#!/usr/bin/env python3
"""Four-node replicated-base DDP LoRA trainer for production SFT qualification."""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import subprocess
import time
from datetime import timedelta
from importlib.metadata import version as package_version
from pathlib import Path
from types import MethodType

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--canonical-trainer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected-samples", required=True, type=int)
    parser.add_argument("--max-seq", type=int, default=4096)
    parser.add_argument("--pad-to-multiple", type=int, default=64)
    parser.add_argument("--short-max", type=int, default=512)
    parser.add_argument("--mid-max", type=int, default=2048)
    parser.add_argument("--short-batch", type=int, default=16)
    parser.add_argument("--mid-batch", type=int, default=4)
    parser.add_argument("--long-batch", type=int, default=1)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--qualification-steps-per-bucket", type=int, default=0)
    parser.add_argument("--qualification-warmup-steps", type=int, default=2)
    parser.add_argument("--min-useful-tps", type=float, default=1000.0)
    parser.add_argument(
        "--min-mem-available-bytes",
        type=int,
        default=40_000_000_000,
    )
    parser.add_argument(
        "--cache-release-interval-steps",
        type=int,
        default=24,
    )
    parser.add_argument(
        "--cache-release-below-available-bytes",
        type=int,
        default=8_000_000_000,
    )
    parser.add_argument(
        "--min-after-cache-release-bytes",
        type=int,
        default=40_000_000_000,
    )
    parser.add_argument("--max-swap-used-bytes", type=int, default=134_217_728)
    parser.add_argument("--max-board-celsius", type=float, default=90.0)
    parser.add_argument("--thermal-persist-steps", type=int, default=3)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--selective-ac-start", type=int, default=0)
    parser.add_argument("--selective-ac-budget", type=int, default=0)
    parser.add_argument("--resume", default="")
    parser.add_argument(
        "--target-modules",
        default=(
            "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,"
            "in_proj_qkv,out_proj"
        ),
    )
    return parser.parse_args()


def load_canonical_trainer(path):
    spec = importlib.util.spec_from_file_location("production_sft_trainer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import canonical trainer from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_distributed_environment():
    required = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise RuntimeError(f"torchrun environment is incomplete: {missing}")
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world != 4:
        raise RuntimeError(f"production DDP requires WORLD_SIZE=4, got {world}")
    if local_rank != 0:
        raise RuntimeError(
            "production topology is one GPU process per Spark; LOCAL_RANK must be 0"
        )
    return rank, world, local_rank


def all_equal(value, world):
    gathered = [None for _ in range(world)]
    dist.all_gather_object(gathered, value)
    if len(set(gathered)) != 1:
        raise RuntimeError(f"rank-divergent receipt: {gathered}")
    return gathered[0]


def assistant_only_selected_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    labels=None,
    use_cache=None,
    cache_position=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    real_mask=None,
    **kwargs,
):
    if input_ids is None or inputs_embeds is not None:
        raise RuntimeError("assistant-only selected loss requires input_ids")
    if labels is None or real_mask is None:
        raise RuntimeError("assistant-only selected loss requires labels and real_mask")
    if labels.shape != input_ids.shape or real_mask.shape != input_ids.shape[:1]:
        raise RuntimeError(
            "assistant-only selected loss received incompatible batch shapes"
        )
    if past_key_values is not None or use_cache not in (None, False):
        raise RuntimeError("assistant-only selected loss cannot run with a KV cache")
    if output_attentions not in (None, False) or output_hidden_states not in (
        None,
        False,
    ):
        raise RuntimeError("assistant-only selected loss does not return layer outputs")
    if return_dict is False:
        raise RuntimeError("assistant-only selected loss returns a scalar tensor")
    unexpected = {key: value for key, value in kwargs.items() if value is not None}
    if unexpected:
        raise RuntimeError(
            f"assistant-only selected loss received unexpected arguments: {unexpected}"
        )

    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
        cache_position=cache_position,
    )
    shift_hidden = outputs.last_hidden_state[:, :-1, :]
    shift_labels = labels[:, 1:]
    label_mask = shift_labels != -100
    per_sample_tokens = label_mask.sum(dim=1)
    if torch.any(real_mask & (per_sample_tokens == 0)):
        raise RuntimeError("real SFT sample has no shifted assistant labels")
    if not torch.any(label_mask):
        return outputs.last_hidden_state.float().sum() * 0.0

    selected_hidden = shift_hidden[label_mask].contiguous()
    selected_targets = shift_labels[label_mask].contiguous()
    selected_logits = F.linear(
        selected_hidden,
        self.lm_head.weight,
    )
    token_losses = F.cross_entropy(
        selected_logits.float(),
        selected_targets,
        reduction="none",
    )
    if token_losses.shape != selected_targets.shape:
        raise RuntimeError(
            "selected loss did not return one loss value per assistant token"
        )
    sample_ids = (
        torch.arange(input_ids.shape[0], device=input_ids.device)
        .unsqueeze(1)
        .expand_as(shift_labels)[label_mask]
    )
    sample_loss_sums = torch.zeros(
        input_ids.shape[0],
        dtype=token_losses.dtype,
        device=token_losses.device,
    ).scatter_add(0, sample_ids, token_losses)
    per_sample_loss = sample_loss_sums / per_sample_tokens.clamp_min(1)
    return (per_sample_loss * real_mask.to(per_sample_loss.dtype)).sum()


def install_assistant_only_selected_loss(model):
    base_model = model.get_base_model()
    if type(base_model).__name__ != "Qwen3_5ForCausalLM":
        raise RuntimeError(
            "assistant-only selected loss requires Qwen3_5ForCausalLM, got "
            f"{type(base_model).__module__}.{type(base_model).__name__}"
        )
    if base_model.lm_head.weight.requires_grad:
        raise RuntimeError("assistant-only selected loss requires a frozen lm_head")
    base_model.forward = MethodType(assistant_only_selected_forward, base_model)
    return {
        "architecture": type(base_model).__name__,
        "implementation": "torch-linear-cross-entropy",
        "reduction": "per_sample_mean",
        "projected_tokens": "assistant_labels_only",
        "temporary_logits": "assistant_tokens_x_vocab",
        "empty_local_partition": "graph-connected-zero",
        "lm_head_trainable": base_model.lm_head.weight.requires_grad,
    }


def install_liger_backbone_kernels(model):
    from liger_kernel.transformers.monkey_patch import (
        apply_liger_kernel_to_qwen3_5,
    )

    if type(model).__name__ != "Qwen3_5ForCausalLM":
        raise RuntimeError(
            "Liger backbone kernels require Qwen3_5ForCausalLM, got "
            f"{type(model).__module__}.{type(model).__name__}"
        )
    apply_liger_kernel_to_qwen3_5(
        rope=False,
        cross_entropy=False,
        fused_linear_cross_entropy=False,
        rms_norm=True,
        swiglu=True,
        model=model,
    )
    text_model = model.model
    rms_norms = [text_model.norm]
    for layer in text_model.layers:
        rms_norms.extend(
            (layer.input_layernorm, layer.post_attention_layernorm)
        )
    unexpected_norms = [
        module._get_name()
        for module in rms_norms
        if module._get_name() != "LigerRMSNorm"
    ]
    unexpected_mlps = [
        layer.mlp._get_name()
        for layer in text_model.layers
        if layer.mlp._get_name() != "LigerQwen3MoeSwiGLUMLP"
    ]
    if unexpected_norms or unexpected_mlps:
        raise RuntimeError(
            "Liger backbone patch receipt is incomplete: "
            f"unexpected_norms={unexpected_norms} "
            f"unexpected_mlps={unexpected_mlps}"
        )
    return {
        "architecture": type(model).__name__,
        "liger_kernel": package_version("liger-kernel"),
        "rms_norm": "LigerRMSNorm",
        "rms_norm_modules": len(rms_norms),
        "swiglu": "LigerQwen3MoeSwiGLUMLP",
        "swiglu_modules": len(text_model.layers),
        "rope": "transformers-qwen3.5",
        "loss": "external-assistant-only-selected",
    }


def prepare_activation_checkpointing(model, full, selective_start, selective_budget):
    text_model = model.model
    layers = list(text_model.layers)
    if full and selective_start:
        raise RuntimeError(
            "full and selective activation checkpointing are mutually exclusive"
        )
    if full or selective_start:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    if selective_start:
        for layer in layers:
            if not hasattr(layer, "_gradient_checkpointing_func"):
                raise RuntimeError(
                    "decoder layer has no checkpoint function after enablement"
                )
            layer.gradient_checkpointing = False
        mode = "adaptive-layer"
    elif full:
        mode = "full-layer"
    else:
        mode = "disabled"
    return layers, {
        "mode": mode,
        "decoder_layers": len(layers),
        "use_reentrant": False,
        "selective_start_padded_tokens": selective_start,
        "selective_activation_budget_tokens": selective_budget,
        "selective_trigger": (
            "padded_tokens_greater_than_or_equal_to_start"
            if selective_start
            else "disabled"
        ),
        "selection": "evenly-spaced",
        "selection_formula": (
            "ceil(decoder_layers*(padded_tokens-budget_tokens)/padded_tokens)"
        ),
    }


def configure_selective_activation_checkpointing(
    layers,
    padded_length,
    selective_start,
    selective_budget,
):
    if padded_length < selective_start:
        checkpoint_count = 0
    else:
        checkpoint_count = math.ceil(
            len(layers) * (padded_length - selective_budget) / padded_length
        )
        checkpoint_count = min(len(layers), checkpoint_count)
    selected = (
        set()
        if checkpoint_count == 0
        else set(evenly_spaced(range(len(layers)), checkpoint_count))
    )
    if len(selected) != checkpoint_count:
        raise RuntimeError(
            "selective activation checkpointing did not select the requested "
            f"layer count: {len(selected)} != {checkpoint_count}"
        )
    for index, layer in enumerate(layers):
        layer.gradient_checkpointing = index in selected
    return checkpoint_count


def dataset_receipt(dataset):
    digest = hashlib.sha256()
    lengths = []
    label_tokens = 0
    for input_ids, labels in dataset.samples:
        length = len(input_ids)
        supervised = sum(label != -100 for label in labels)
        if length <= 0 or length != len(labels) or supervised <= 0:
            raise RuntimeError(
                "canonical SFT cache contains an empty, mismatched, or unsupervised sample"
            )
        digest.update(f"{length}:{supervised}\n".encode("ascii"))
        lengths.append(length)
        label_tokens += supervised
    return {
        "samples": len(lengths),
        "input_tokens": sum(lengths),
        "label_tokens": label_tokens,
        "min_length": min(lengths),
        "max_length": max(lengths),
        "shape_sha256": digest.hexdigest(),
    }, lengths


class ExactSampleDataset(Dataset):
    def __init__(self, canonical):
        self.canonical = canonical

    def __len__(self):
        return len(self.canonical)

    def __getitem__(self, index):
        if index >= 0:
            sample = dict(self.canonical[index])
            sample["is_padding"] = False
            return sample
        sample = dict(self.canonical[0])
        sample["labels"] = [-100] * len(sample["input_ids"])
        sample["is_padding"] = True
        return sample


def evenly_spaced(items, count):
    if count <= 0 or count >= len(items):
        return list(items)
    if count == 1:
        return [items[len(items) // 2]]
    positions = {
        round(index * (len(items) - 1) / (count - 1))
        for index in range(count)
    }
    return [items[position] for position in sorted(positions)]


def build_batch_plan(
    lengths,
    world,
    short_max,
    mid_max,
    short_batch,
    mid_batch,
    long_batch,
    seed,
    qualification_steps_per_bucket,
):
    if not (0 < short_max < mid_max):
        raise ValueError("length thresholds must satisfy 0 < short_max < mid_max")
    bucket_specs = (
        ("short", 0, short_max, short_batch),
        ("mid", short_max + 1, mid_max, mid_batch),
        ("long", mid_max + 1, math.inf, long_batch),
    )
    groups_by_bucket = {}
    for bucket, low, high, local_batch in bucket_specs:
        if local_batch <= 0:
            raise ValueError(f"{bucket} batch must be positive")
        indices = [
            index
            for index, length in enumerate(lengths)
            if low <= length <= high
        ]
        span = local_batch * world
        groups = []
        for offset in range(0, len(indices), span):
            global_indices = indices[offset : offset + span]
            real_samples = len(global_indices)
            global_indices.extend([-1] * (span - real_samples))
            rank_indices = [
                global_indices[rank * local_batch : (rank + 1) * local_batch]
                for rank in range(world)
            ]
            real_lengths = [
                lengths[index] for index in global_indices if index >= 0
            ]
            groups.append(
                {
                    "bucket": bucket,
                    "local_batch": local_batch,
                    "rank_indices": rank_indices,
                    "real_samples": real_samples,
                    "min_length": min(real_lengths),
                    "max_length": max(real_lengths),
                    "useful_tokens": sum(real_lengths),
                }
            )
        if groups:
            groups_by_bucket[bucket] = groups

    full_plan = [
        group
        for bucket in ("short", "mid", "long")
        for group in groups_by_bucket.get(bucket, [])
    ]
    if qualification_steps_per_bucket > 0:
        selected_by_bucket = {}
        late_by_bucket = {}

        def padded_length(group):
            return (
                (group["max_length"] + 15) // 16
            ) * 16

        for bucket in ("short", "mid", "long"):
            bucket_groups = groups_by_bucket.get(bucket, [])
            full_groups = [
                group
                for group in bucket_groups
                if group["real_samples"] == group["local_batch"] * world
            ]
            partial_groups = [
                group
                for group in bucket_groups
                if group["real_samples"] < group["local_batch"] * world
            ]
            maximum_padded_length = max(
                padded_length(group) for group in bucket_groups
            )
            maximum_groups = [
                group
                for group in bucket_groups
                if padded_length(group) == maximum_padded_length
            ]
            if len(maximum_groups) < 2:
                raise RuntimeError(
                    f"qualification bucket {bucket} has fewer than two "
                    "maximum-padded production groups"
                )
            late_groups = maximum_groups[-2:]
            reserved_ids = {
                id(group) for group in partial_groups + late_groups
            }
            reserved_groups = [
                group
                for group in bucket_groups
                if id(group) in reserved_ids
            ]
            remaining_slots = (
                qualification_steps_per_bucket - len(reserved_groups)
            )
            remaining_full_groups = [
                group
                for group in full_groups
                if id(group) not in reserved_ids
            ]
            if (
                remaining_slots < 0
                or remaining_slots > len(remaining_full_groups)
            ):
                raise RuntimeError(
                    f"qualification bucket {bucket} cannot include its "
                    "production partial and maximum-padded groups"
                )
            selected = evenly_spaced(
                sorted(
                    remaining_full_groups,
                    key=lambda group: (
                        group["max_length"],
                        group["min_length"],
                        group["rank_indices"],
                    ),
                ),
                remaining_slots,
            )
            selected.extend(reserved_groups)
            if len(selected) != qualification_steps_per_bucket:
                raise RuntimeError(
                    f"qualification bucket {bucket} produced {len(selected)} "
                    f"groups, expected {qualification_steps_per_bucket}"
                )
            selected_by_bucket[bucket] = selected
            late_by_bucket[bucket] = late_groups

        late_maximum_groups = []
        remaining_groups = []
        for bucket in ("short", "mid", "long"):
            late_ids = {id(group) for group in late_by_bucket[bucket]}
            late_maximum_groups.extend(late_by_bucket[bucket])
            remaining_groups.extend(
                group
                for group in selected_by_bucket[bucket]
                if id(group) not in late_ids
            )
        generator = random.Random(seed)
        generator.shuffle(remaining_groups)
        generator.shuffle(late_maximum_groups)
        plan = remaining_groups + late_maximum_groups
    else:
        plan = list(full_plan)
        random.Random(seed).shuffle(plan)

    if not plan or not full_plan:
        raise RuntimeError("adaptive batch plan is empty")
    payload = {
        "seed": seed,
        "thresholds": [short_max, mid_max],
        "batches": [short_batch, mid_batch, long_batch],
        "qualification_steps_per_bucket": qualification_steps_per_bucket,
        "qualification_full_groups_only": False,
        "groups": [
            {
                "bucket": group["bucket"],
                "local_batch": group["local_batch"],
                "rank_indices": group["rank_indices"],
            }
            for group in plan
        ],
    }
    if qualification_steps_per_bucket > 0:
        payload["qualification_includes_production_partials"] = True
    plan_sha = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return plan, full_plan, plan_sha


class RankBatchSampler(Sampler):
    def __init__(self, plan, rank, start_step):
        self.plan = plan
        self.rank = rank
        self.start_step = start_step

    def __iter__(self):
        for group in self.plan[self.start_step :]:
            yield group["rank_indices"][self.rank]

    def __len__(self):
        return len(self.plan) - self.start_step


def collate(batch, pad_id, pad_to_multiple):
    max_len = max(len(sample["input_ids"]) for sample in batch)
    target = (
        (max_len + pad_to_multiple - 1) // pad_to_multiple
    ) * pad_to_multiple
    count = len(batch)
    input_ids = torch.full((count, target), pad_id, dtype=torch.long)
    labels = torch.full((count, target), -100, dtype=torch.long)
    attention_mask = torch.zeros((count, target), dtype=torch.long)
    real_mask = torch.zeros(count, dtype=torch.bool)
    for index, sample in enumerate(batch):
        ids = sample["input_ids"]
        sample_labels = sample["labels"]
        input_ids[index, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        labels[index, : len(sample_labels)] = torch.tensor(
            sample_labels, dtype=torch.long
        )
        attention_mask[index, : len(ids)] = 1
        real_mask[index] = not sample["is_padding"]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "real_mask": real_mask,
    }


def lora_fingerprint(model):
    digest = hashlib.sha256()
    tensor_count = 0
    element_count = 0
    total_bytes = 0
    dtypes = set()
    finite = True
    absolute_sum = 0.0
    squared_sum = 0.0
    maximum_absolute = 0.0
    for name, parameter in sorted(model.named_parameters()):
        if not parameter.requires_grad:
            continue
        value = parameter.detach().cpu().contiguous()
        finite = finite and bool(torch.isfinite(value).all().item())
        float_value = value.double()
        absolute_sum += float_value.abs().sum().item()
        squared_sum += float_value.square().sum().item()
        maximum_absolute = max(
            maximum_absolute,
            float_value.abs().max().item(),
        )
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(
            json.dumps(list(value.shape), separators=(",", ":")).encode("ascii")
        )
        digest.update(value.view(torch.uint8).numpy().tobytes())
        tensor_count += 1
        element_count += value.numel()
        total_bytes += value.numel() * value.element_size()
        dtypes.add(str(value.dtype))
    if tensor_count == 0:
        raise RuntimeError("LoRA model has no trainable tensors")
    return {
        "sha256": digest.hexdigest(),
        "tensors": tensor_count,
        "elements": element_count,
        "bytes": total_bytes,
        "dtypes": sorted(dtypes),
        "finite": finite,
        "l2_norm": math.sqrt(squared_sum),
        "mean_abs": absolute_sum / element_count,
        "max_abs": maximum_absolute,
    }


def snapshot_trainable_parameters(model):
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in sorted(model.named_parameters())
        if parameter.requires_grad
    }


def lora_delta_receipt(model, reference):
    digest = hashlib.sha256()
    parameter_by_name = dict(model.named_parameters())
    tensor_count = 0
    element_count = 0
    changed_elements = 0
    absolute_sum = 0.0
    squared_sum = 0.0
    maximum_absolute = 0.0
    finite = True
    for name, initial in sorted(reference.items()):
        current = parameter_by_name[name].detach().cpu()
        if current.shape != initial.shape or current.dtype != initial.dtype:
            raise RuntimeError(f"LoRA delta shape/dtype changed for {name}")
        delta = (current - initial).contiguous()
        float_delta = delta.double()
        finite = finite and bool(torch.isfinite(delta).all().item())
        tensor_count += 1
        element_count += delta.numel()
        changed_elements += torch.count_nonzero(delta).item()
        absolute_sum += float_delta.abs().sum().item()
        squared_sum += float_delta.square().sum().item()
        maximum_absolute = max(
            maximum_absolute,
            float_delta.abs().max().item(),
        )
        digest.update(name.encode("utf-8"))
        digest.update(str(delta.dtype).encode("ascii"))
        digest.update(
            json.dumps(list(delta.shape), separators=(",", ":")).encode("ascii")
        )
        digest.update(delta.view(torch.uint8).numpy().tobytes())
    if tensor_count != len(reference) or element_count == 0:
        raise RuntimeError("LoRA delta receipt is incomplete")
    return {
        "sha256": digest.hexdigest(),
        "tensors": tensor_count,
        "elements": element_count,
        "finite": finite,
        "changed_elements": changed_elements,
        "changed_fraction": changed_elements / element_count,
        "mean_abs": absolute_sum / element_count,
        "l2_norm": math.sqrt(squared_sum),
        "max_abs": maximum_absolute,
    }


def scheduler_for(optimizer, warmup_steps, total_steps):
    if warmup_steps <= 0 or warmup_steps >= total_steps:
        raise ValueError(
            f"warmup_steps must be in [1,total_steps), got {warmup_steps}/{total_steps}"
        )

    def multiplier(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.45 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def checkpoint_step(path):
    name = Path(path).name
    if not name.startswith("checkpoint-"):
        raise ValueError(f"resume path is not checkpoint-N: {path}")
    value = name.removeprefix("checkpoint-")
    if not value.isdigit():
        raise ValueError(f"resume path is not checkpoint-N: {path}")
    return int(value)


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mem_available_bytes():
    with open("/proc/meminfo", encoding="ascii") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("/proc/meminfo has no MemAvailable field")


def swap_used_bytes():
    values = {}
    with open("/proc/meminfo", encoding="ascii") as handle:
        for line in handle:
            name, value, *_ = line.split()
            if name in ("SwapTotal:", "SwapFree:"):
                values[name] = int(value) * 1024
    if set(values) != {"SwapTotal:", "SwapFree:"}:
        raise RuntimeError("/proc/meminfo has incomplete swap fields")
    return values["SwapTotal:"] - values["SwapFree:"]


def enforce_uma_headroom(
    device,
    completed_steps,
    last_cache_release_step,
    release_interval,
    release_below_available,
    minimum_after_release,
    maximum_swap_used,
    swap_baseline,
    allocator_retries_baseline,
    oom_baseline,
    allocated_before_step,
    world,
):
    guard_started = time.perf_counter()
    before_observation_started = time.perf_counter()
    cuda_free_before, _ = torch.cuda.mem_get_info(device)
    available_before = mem_available_bytes()
    swap_before = swap_used_bytes()
    before_observation_seconds = (
        time.perf_counter() - before_observation_started
    )
    before = torch.tensor(
        [cuda_free_before, available_before, swap_before],
        dtype=torch.float64,
        device=device,
    )
    before_by_rank = [torch.zeros_like(before) for _ in range(world)]
    pre_collective_started = time.perf_counter()
    dist.all_gather(before_by_rank, before)
    pre_collective_seconds = time.perf_counter() - pre_collective_started
    release_gap_steps = completed_steps - last_cache_release_step
    periodic_release = completed_steps % release_interval == 0
    emergency_release = (
        min(int(values[1].item()) for values in before_by_rank)
        < release_below_available
    )
    cache_reclaimed = periodic_release or emergency_release
    gc_seconds = 0.0
    empty_cache_seconds = 0.0
    reclaim_sync_seconds = 0.0
    if cache_reclaimed:
        reclaim_sync_started = time.perf_counter()
        torch.cuda.synchronize(device)
        reclaim_sync_seconds = time.perf_counter() - reclaim_sync_started
        empty_cache_started = time.perf_counter()
        torch.cuda.empty_cache()
        empty_cache_seconds = time.perf_counter() - empty_cache_started
    after_observation_started = time.perf_counter()
    cuda_free_after, _ = torch.cuda.mem_get_info(device)
    available_after = mem_available_bytes()
    swap_after = swap_used_bytes()
    memory_after = torch.cuda.memory_stats(device)
    after_observation_seconds = time.perf_counter() - after_observation_started
    allocator_retries_after = memory_after["num_alloc_retries"]
    ooms_after = memory_after["num_ooms"]
    swap_growth_bytes = max(
        0,
        swap_before - swap_baseline,
        swap_after - swap_baseline,
    )
    allocator_retry_growth = max(
        0,
        allocator_retries_after - allocator_retries_baseline,
    )
    oom_growth = max(0, ooms_after - oom_baseline)
    required_available = (
        minimum_after_release
        if cache_reclaimed
        else release_below_available
    )
    local_memory_exit = (
        available_after < required_available
        or swap_after > maximum_swap_used
        or swap_growth_bytes
        or allocator_retry_growth
        or oom_growth
    )
    after = torch.tensor(
        [
            cuda_free_after,
            available_after,
            swap_after,
            memory_after["allocated_bytes.all.current"],
            memory_after["reserved_bytes.all.current"],
            memory_after["active_bytes.all.current"],
            memory_after["inactive_split_bytes.all.current"],
            allocator_retries_after,
            ooms_after,
            float(cache_reclaimed),
            float(local_memory_exit),
            float(allocator_retry_growth),
            float(oom_growth),
            float(swap_growth_bytes),
            allocated_before_step,
            memory_after["allocated_bytes.all.peak"],
            memory_after["reserved_bytes.all.peak"],
            max(
                0,
                memory_after["allocated_bytes.all.peak"]
                - allocated_before_step,
            ),
            before_observation_seconds,
            pre_collective_seconds,
            gc_seconds,
            empty_cache_seconds,
            reclaim_sync_seconds,
            after_observation_seconds,
            float(periodic_release),
            float(emergency_release),
            release_gap_steps,
        ],
        dtype=torch.float64,
        device=device,
    )
    after_by_rank = [torch.zeros_like(after) for _ in range(world)]
    post_collective_started = time.perf_counter()
    dist.all_gather(after_by_rank, after)
    post_collective_seconds = time.perf_counter() - post_collective_started
    memory_exit_flag = torch.tensor(
        [int(local_memory_exit)],
        dtype=torch.int32,
        device=device,
    )
    exit_collective_started = time.perf_counter()
    dist.all_reduce(memory_exit_flag, op=dist.ReduceOp.MAX)
    exit_collective_seconds = time.perf_counter() - exit_collective_started
    memory_exit = bool(memory_exit_flag.item())
    gathered = [
        torch.tensor(
            [
                before_values[0].item(),
                after_values[0].item(),
                before_values[1].item(),
                after_values[1].item(),
                before_values[2].item(),
                after_values[2].item(),
                after_values[3].item(),
                after_values[4].item(),
                after_values[5].item(),
                after_values[6].item(),
                after_values[7].item(),
                after_values[8].item(),
                after_values[9].item(),
                after_values[10].item(),
                after_values[11].item(),
                after_values[12].item(),
                after_values[13].item(),
                after_values[14].item(),
                after_values[15].item(),
                after_values[16].item(),
                after_values[17].item(),
                after_values[24].item(),
                after_values[25].item(),
                after_values[26].item(),
            ],
            dtype=torch.float64,
            device=device,
        )
        for before_values, after_values in zip(before_by_rank, after_by_rank)
    ]
    guard_times = {
        "observe_before": before_observation_seconds,
        "pre_collective": pre_collective_seconds,
        "python_gc": gc_seconds,
        "empty_cache": empty_cache_seconds,
        "reclaim_sync": reclaim_sync_seconds,
        "observe_after": after_observation_seconds,
        "post_collective": post_collective_seconds,
        "exit_collective": exit_collective_seconds,
        "total": time.perf_counter() - guard_started,
    }
    return gathered, memory_exit, guard_times


def evict_model_page_cache(model_path):
    if not hasattr(os, "posix_fadvise") or not hasattr(
        os, "POSIX_FADV_DONTNEED"
    ):
        raise RuntimeError("POSIX_FADV_DONTNEED is unavailable")
    model_files = sorted(Path(model_path).glob("*.safetensors"))
    if not model_files:
        raise RuntimeError(f"base model has no safetensor files: {model_path}")
    before = mem_available_bytes()
    total_bytes = 0
    for model_file in model_files:
        descriptor = os.open(model_file, os.O_RDONLY)
        try:
            os.posix_fadvise(
                descriptor,
                0,
                0,
                os.POSIX_FADV_DONTNEED,
            )
            total_bytes += model_file.stat().st_size
        finally:
            os.close(descriptor)
    after = mem_available_bytes()
    return {
        "files": len(model_files),
        "file_bytes": total_bytes,
        "mem_available_before": before,
        "mem_available_after": after,
    }


def save_checkpoint(
    ddp_model,
    optimizer,
    scheduler,
    output_dir,
    completed_steps,
    plan_sha,
    rank,
    world,
    args,
):
    final_dir = Path(output_dir) / f"checkpoint-{completed_steps}"
    temp_dir = Path(output_dir) / f".checkpoint-{completed_steps}.tmp-rank{rank}-{os.getpid()}"
    if final_dir.exists():
        raise RuntimeError(f"refusing to overwrite existing checkpoint: {final_dir}")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)

    model = ddp_model.module
    model.save_pretrained(
        temp_dir,
        safe_serialization=True,
        selected_adapters=["default"],
        save_embedding_layers=False,
    )
    state = {
        "rank": rank,
        "world": world,
        "completed_steps": completed_steps,
        "plan_sha256": plan_sha,
        "production_plan_sha256": args.production_plan_sha,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(),
    }
    torch.save(state, temp_dir / "trainer_state.pt")
    manifest = {
        "format": "replicated-ddp-lora-v1",
        "completed_steps": completed_steps,
        "plan_sha256": plan_sha,
        "production_plan_sha256": args.production_plan_sha,
        "world_size": world,
        "base_model": args.model,
        "corpus": args.data,
        "max_seq": args.max_seq,
        "pad_to_multiple": args.pad_to_multiple,
        "rank": args.rank,
        "alpha": args.alpha,
        "dropout": args.dropout,
        "target_modules": args.target_modules.split(","),
        "optimizer": "AdamW-fp32-adapters",
        "loss_path": "torch-linear-ce-assistant-only",
        "loss_receipt": args.loss_receipt,
        "backbone_kernels": args.backbone_kernel_receipt,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "gradient_checkpointing": args.gradient_checkpointing,
        "activation_checkpointing": args.activation_checkpoint_receipt,
        "max_board_celsius": args.max_board_celsius,
        "thermal_persist_steps": args.thermal_persist_steps,
        "uma_memory": {
            "cache_release_interval_steps": (
                args.cache_release_interval_steps
            ),
            "cache_release_below_available_bytes": (
                args.cache_release_below_available_bytes
            ),
            "minimum_after_cache_release_bytes": (
                args.min_after_cache_release_bytes
            ),
            "minimum_system_available_after_load_bytes": (
                args.min_mem_available_bytes
            ),
            "maximum_swap_used_bytes": args.max_swap_used_bytes,
            "reclaim_python_gc": False,
            "pass_fail_signal": "system_MemAvailable",
            "cuda_free": "telemetry_only",
            "post_step_gate": True,
        },
        "qualification_plan": args.qualification_plan_receipt,
        "uma_baseline": args.uma_baseline,
        "uma_observed": args.uma_observed,
    }
    write_json(temp_dir / "training_manifest.json", manifest)
    adapter_path = temp_dir / "adapter_model.safetensors"
    adapter_sha = file_sha256(adapter_path)
    checksum_names = (
        "adapter_model.safetensors",
        "adapter_config.json",
        "trainer_state.pt",
        "training_manifest.json",
    )
    write_json(
        temp_dir / "SHA256SUMS.json",
        {
            name: file_sha256(temp_dir / name)
            for name in checksum_names
        },
    )
    adapter_sha = all_equal(adapter_sha, world)
    (temp_dir / "COMPLETE").write_text(
        f"adapter_sha256={adapter_sha}\n", encoding="utf-8"
    )
    os.rename(temp_dir, final_dir)
    dist.barrier()
    return str(final_dir), adapter_sha


def load_training_state(
    resume,
    optimizer,
    scheduler,
    plan_sha,
    rank,
    world,
):
    checkpoint_dir = Path(resume)
    state_path = checkpoint_dir / "trainer_state.pt"
    complete_path = checkpoint_dir / "COMPLETE"
    checksums_path = checkpoint_dir / "SHA256SUMS.json"
    if (
        not state_path.is_file()
        or not complete_path.is_file()
        or not checksums_path.is_file()
    ):
        raise RuntimeError(f"resume checkpoint is incomplete: {resume}")
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    required = {
        "adapter_model.safetensors",
        "adapter_config.json",
        "trainer_state.pt",
        "training_manifest.json",
    }
    if not isinstance(checksums, dict) or not required.issubset(checksums):
        raise RuntimeError(
            f"resume checkpoint has incomplete checksums: {resume}"
        )
    for name in sorted(required):
        path = checkpoint_dir / name
        expected_sha = checksums[name]
        if (
            not path.is_file()
            or not isinstance(expected_sha, str)
            or len(expected_sha) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_sha
            )
            or file_sha256(path) != expected_sha
        ):
            raise RuntimeError(
                f"resume checkpoint checksum failed for {name}: {resume}"
            )
    expected_complete = (
        "adapter_sha256="
        + checksums["adapter_model.safetensors"]
    )
    if complete_path.read_text(encoding="utf-8").strip() != expected_complete:
        raise RuntimeError(
            f"resume checkpoint COMPLETE receipt differs: {resume}"
        )
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    expected_step = checkpoint_step(resume)
    expected = {
        "rank": rank,
        "world": world,
        "completed_steps": expected_step,
        "plan_sha256": plan_sha,
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise RuntimeError(
                f"resume state mismatch for {key}: {state.get(key)!r} != {value!r}"
            )
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    random.setstate(state["python_rng"])
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state(state["cuda_rng"])
    return expected_step


def board_metrics(device):
    output = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=power.draw,temperature.gpu,clocks.gr",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    values = [float(value.strip()) for value in output.split(",")]
    if len(values) != 3:
        raise RuntimeError(f"unexpected nvidia-smi telemetry: {output!r}")
    board_temperatures = []
    for thermal_path in sorted(
        Path("/sys/class/thermal").glob("thermal_zone*/temp")
    ):
        try:
            board_temperatures.append(
                float(thermal_path.read_text(encoding="ascii").strip()) / 1000
            )
        except (OSError, ValueError):
            continue
    if not board_temperatures:
        raise RuntimeError("no readable board/SoC thermal zones")
    memory = torch.cuda.memory_stats(device)
    return torch.tensor(
        [
            values[0],
            values[1],
            values[2],
            max(board_temperatures),
            memory["allocated_bytes.all.current"] / 1e9,
            memory["reserved_bytes.all.current"] / 1e9,
            memory["allocated_bytes.all.peak"] / 1e9,
            memory["reserved_bytes.all.peak"] / 1e9,
        ],
        dtype=torch.float64,
        device=device,
    )


def bootstrap_throughput_lower_bounds(
    observations,
    full_bucket_tokens,
    seed,
    draws=2000,
    block_size=8,
):
    if len(observations) < block_size * 4:
        raise RuntimeError(
            "qualification window is too short for block bootstrap"
        )
    generator = random.Random(seed)
    measured_values = []
    projected_values = []
    count = len(observations)
    for _ in range(draws):
        sampled = []
        while len(sampled) < count:
            start = generator.randrange(count)
            sampled.extend(
                observations[(start + offset) % count]
                for offset in range(block_size)
            )
        sampled = sampled[:count]
        total_useful = sum(value["useful"] for value in sampled)
        total_seconds = sum(value["seconds"] for value in sampled)
        measured_values.append(total_useful / total_seconds)
        projected_seconds = 0.0
        for bucket, full_tokens in full_bucket_tokens.items():
            bucket_values = [
                value for value in sampled if value["bucket"] == bucket
            ]
            if not bucket_values:
                raise RuntimeError(
                    f"bootstrap sample omitted bucket {bucket}"
                )
            bucket_useful = sum(
                value["useful"] for value in bucket_values
            )
            bucket_seconds = sum(
                value["seconds"] for value in bucket_values
            )
            projected_seconds += full_tokens / (
                bucket_useful / bucket_seconds
            )
        projected_values.append(
            sum(full_bucket_tokens.values()) / projected_seconds
        )
    lower_index = max(0, math.ceil(draws * 0.05) - 1)
    measured_values.sort()
    projected_values.sort()
    return {
        "confidence": 0.95,
        "draws": draws,
        "block_size": block_size,
        "measured_useful_input_tok_s": measured_values[lower_index],
        "projected_full_pass_useful_input_tok_s": (
            projected_values[lower_index]
        ),
    }


def conservative_linear_projection_lower(values, target_step):
    if len(values) < 32:
        raise RuntimeError(
            "memory window is too short for a session projection"
        )
    x_values = [float(value["step"]) for value in values]
    y_values = [float(value["bytes"]) for value in values]
    count = len(values)
    x_mean = sum(x_values) / count
    y_mean = sum(y_values) / count
    centered_x_sum = sum(
        (value - x_mean) ** 2 for value in x_values
    )
    if centered_x_sum <= 0:
        raise RuntimeError("memory projection has no step span")
    fitted_slope = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x_values, y_values)
    ) / centered_x_sum
    conservative_slope = min(0.0, fitted_slope)
    residuals = [
        y_value
        - (
            y_mean
            + fitted_slope * (x_value - x_mean)
        )
        for x_value, y_value in zip(x_values, y_values)
    ]
    residual_standard_error = math.sqrt(
        sum(value * value for value in residuals)
        / max(1, count - 2)
    )
    projection = y_mean + conservative_slope * (
        target_step - x_mean
    )
    prediction_standard_error = residual_standard_error * math.sqrt(
        1
        + 1 / count
        + (target_step - x_mean) ** 2 / centered_x_sum
    )
    return {
        "observations": count,
        "target_step": target_step,
        "fitted_slope_bytes_per_step": fitted_slope,
        "conservative_slope_bytes_per_step": conservative_slope,
        "point_bytes": projection,
        "one_sided_95pct_lower_bytes": (
            projection - 1.645 * prediction_standard_error
        ),
    }


def main():
    args = parse()
    rank, world, local_rank = require_distributed_environment()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=1800))
    is_main = rank == 0

    if args.expected_samples <= 0 or args.max_seq <= 256:
        raise ValueError("expected samples must be positive and max-seq must exceed 256")
    if args.min_mem_available_bytes <= 0:
        raise ValueError("minimum available memory must be positive")
    if not (
        0
        < args.cache_release_below_available_bytes
        < args.min_after_cache_release_bytes
    ):
        raise ValueError(
            "system no-release floor must be below the post-release floor"
        )
    if args.cache_release_interval_steps <= 0:
        raise ValueError("cache release interval must be positive")
    if args.min_mem_available_bytes < args.min_after_cache_release_bytes:
        raise ValueError(
            "post-load system-memory minimum cannot be below the post-release floor"
        )
    if args.max_swap_used_bytes < 0:
        raise ValueError("maximum swap usage cannot be negative")
    if args.pad_to_multiple < 8 or args.pad_to_multiple % 8:
        raise ValueError("padding multiple must be a positive multiple of 8")
    if (
        args.qualification_steps_per_bucket
        and args.pad_to_multiple != 16
    ):
        raise ValueError(
            "production qualification requires 16-token padding alignment"
        )
    if not 50 <= args.max_board_celsius < 94:
        raise ValueError("max board temperature must be in [50, 94) Celsius")
    if args.thermal_persist_steps <= 0:
        raise ValueError("thermal persistence must be positive")
    if args.max_steps < 0 or args.qualification_steps_per_bucket < 0:
        raise ValueError("step limits cannot be negative")
    if args.selective_ac_start < 0 or args.selective_ac_budget < 0:
        raise ValueError("selective activation checkpoint settings cannot be negative")
    if bool(args.selective_ac_start) != bool(args.selective_ac_budget):
        raise ValueError(
            "selective activation checkpoint start and budget must be set together"
        )
    if args.selective_ac_start:
        if args.selective_ac_start % 64 or args.selective_ac_budget % 64:
            raise ValueError(
                "selective activation checkpoint settings must align to 64 tokens"
            )
        if args.selective_ac_budget >= args.selective_ac_start:
            raise ValueError(
                "selective activation budget must be below its start threshold"
            )
    if args.gradient_checkpointing and args.selective_ac_start:
        raise ValueError(
            "full and selective activation checkpointing are mutually exclusive"
        )
    if args.qualification_steps_per_bucket == 0 and args.min_useful_tps != 1000.0:
        raise ValueError("--min-useful-tps only governs qualification runs")
    for path in (args.model, args.data, args.canonical_trainer):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    canonical = load_canonical_trainer(args.canonical_trainer)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = canonical.BucketSFTDataset(
        args.data,
        tokenizer,
        args.max_seq,
        strict_one_sample_per_row=False,
    )
    receipt, lengths = dataset_receipt(dataset)
    if receipt["samples"] != args.expected_samples:
        raise RuntimeError(
            f"tokenized sample count {receipt['samples']} != expected {args.expected_samples}"
        )
    all_equal(json.dumps(receipt, sort_keys=True), world)

    plan, full_plan, plan_sha = build_batch_plan(
        lengths,
        world,
        args.short_max,
        args.mid_max,
        args.short_batch,
        args.mid_batch,
        args.long_batch,
        args.seed,
        args.qualification_steps_per_bucket,
    )
    all_equal(plan_sha, world)
    if args.qualification_steps_per_bucket:
        _, _, production_plan_sha = build_batch_plan(
            lengths,
            world,
            args.short_max,
            args.mid_max,
            args.short_batch,
            args.mid_batch,
            args.long_batch,
            args.seed,
            0,
        )
    else:
        production_plan_sha = plan_sha
    all_equal(production_plan_sha, world)
    args.production_plan_sha = production_plan_sha
    schedule_steps = len(full_plan)
    warmup_steps = max(1, round(schedule_steps * args.warmup_ratio))
    bucket_summary = {}
    for group in full_plan:
        value = bucket_summary.setdefault(
            group["bucket"],
            {"steps": 0, "samples": 0, "tokens": 0, "partial_steps": 0},
        )
        value["steps"] += 1
        value["samples"] += group["real_samples"]
        value["tokens"] += group["useful_tokens"]
        if group["real_samples"] < group["local_batch"] * world:
            value["partial_steps"] += 1
    qualification_plan_receipt = None
    if args.qualification_steps_per_bucket:
        late_boundary = len(plan) * 2 // 3
        qualification_buckets = {}
        for bucket in ("short", "mid", "long"):
            selected = [
                (index, group)
                for index, group in enumerate(plan)
                if group["bucket"] == bucket
            ]
            padded_lengths = [
                (
                    (group["max_length"] + args.pad_to_multiple - 1)
                    // args.pad_to_multiple
                )
                * args.pad_to_multiple
                for _, group in selected
            ]
            maximum_padded = max(padded_lengths)
            qualification_buckets[bucket] = {
                "steps": len(selected),
                "partial_steps": sum(
                    group["real_samples"] < group["local_batch"] * world
                    for _, group in selected
                ),
                "minimum_padded_length": min(padded_lengths),
                "maximum_padded_length": maximum_padded,
                "late_maximum_steps": sum(
                    index >= late_boundary
                    and (
                        (
                            group["max_length"] + args.pad_to_multiple - 1
                        )
                        // args.pad_to_multiple
                    )
                    * args.pad_to_multiple
                    == maximum_padded
                    for index, group in selected
                ),
            }
        transitions = {
            f"{source}->{target}": 0
            for source in ("short", "mid", "long")
            for target in ("short", "mid", "long")
            if source != target
        }
        for previous, current in zip(plan, plan[1:]):
            if previous["bucket"] != current["bucket"]:
                key = f"{previous['bucket']}->{current['bucket']}"
                transitions[key] += 1
        qualification_plan_receipt = {
            "buckets": qualification_buckets,
            "late_boundary_step": late_boundary + 1,
            "transitions": transitions,
        }
        for bucket, value in qualification_buckets.items():
            if (
                value["steps"] != args.qualification_steps_per_bucket
                or value["partial_steps"]
                != bucket_summary[bucket]["partial_steps"]
                or value["late_maximum_steps"] < 2
            ):
                raise RuntimeError(
                    f"qualification coverage failed for {bucket}: {value}"
                )
    args.qualification_plan_receipt = qualification_plan_receipt
    if is_main:
        print(
            "DATASET_RECEIPT "
            + json.dumps(receipt, sort_keys=True, separators=(",", ":")),
            flush=True,
        )
        print(
            "BATCH_PLAN "
            + json.dumps(
                {
                    "sha256": plan_sha,
                    "production_plan_sha256": production_plan_sha,
                    "qualification_steps": len(plan),
                    "full_schedule_steps": schedule_steps,
                    "warmup_steps": warmup_steps,
                    "pad_to_multiple": args.pad_to_multiple,
                    "buckets": bucket_summary,
                    "qualification_plan": qualification_plan_receipt,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        device_map={"": local_rank},
    )
    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False
    backbone_kernel_receipt = install_liger_backbone_kernels(model)
    all_equal(json.dumps(backbone_kernel_receipt, sort_keys=True), world)
    args.backbone_kernel_receipt = backbone_kernel_receipt
    if is_main:
        print(
            "BACKBONE_KERNEL_RECEIPT "
            + json.dumps(
                backbone_kernel_receipt,
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
    memory_receipt = evict_model_page_cache(args.model)
    memory_receipts = [None for _ in range(world)]
    dist.all_gather_object(memory_receipts, memory_receipt)
    minimum_available = min(
        receipt["mem_available_after"] for receipt in memory_receipts
    )
    if minimum_available < args.min_mem_available_bytes:
        raise RuntimeError(
            "post-load page-cache eviction left insufficient UMA headroom: "
            f"minimum={minimum_available} required={args.min_mem_available_bytes} "
            f"receipts={memory_receipts}"
        )
    if is_main:
        print(
            "POST_LOAD_MEMORY_RECEIPT "
            + json.dumps(
                {
                    "minimum_required": args.min_mem_available_bytes,
                    "ranks": memory_receipts,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
    checkpoint_layers, activation_checkpoint_receipt = (
        prepare_activation_checkpointing(
            model,
            args.gradient_checkpointing,
            args.selective_ac_start,
            args.selective_ac_budget,
        )
    )
    all_equal(json.dumps(activation_checkpoint_receipt, sort_keys=True), world)
    args.activation_checkpoint_receipt = activation_checkpoint_receipt
    if is_main:
        print(
            "ACTIVATION_CHECKPOINT_RECEIPT "
            + json.dumps(
                activation_checkpoint_receipt,
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=args.target_modules.split(","),
        task_type="CAUSAL_LM",
    )
    if args.resume:
        model = PeftModel.from_pretrained(
            model,
            args.resume,
            is_trainable=True,
            autocast_adapter_dtype=True,
        )
    else:
        model = get_peft_model(
            model,
            lora_config,
            autocast_adapter_dtype=True,
        )
    fingerprint = lora_fingerprint(model)
    all_equal(json.dumps(fingerprint, sort_keys=True), world)
    if fingerprint["dtypes"] != ["torch.float32"]:
        raise RuntimeError(
            "stock AdamW is authorized only with FP32 adapter parameters; "
            f"observed {fingerprint['dtypes']}"
        )
    if (
        fingerprint["tensors"] != 704
        or fingerprint["bytes"] != 400_556_032
        or not fingerprint["finite"]
    ):
        raise RuntimeError(
            "production LoRA fingerprint contract failed: "
            f"{fingerprint}"
        )
    if is_main:
        model.print_trainable_parameters()
        print(
            "LORA_RECEIPT "
            + json.dumps(fingerprint, sort_keys=True, separators=(",", ":")),
            flush=True,
        )
    initial_trainable_snapshot = (
        snapshot_trainable_parameters(model)
        if args.qualification_steps_per_bucket
        else None
    )

    loss_receipt = install_assistant_only_selected_loss(model)
    all_equal(json.dumps(loss_receipt, sort_keys=True), world)
    args.loss_receipt = loss_receipt
    if is_main:
        print(
            "LOSS_PATH_RECEIPT "
            + json.dumps(loss_receipt, sort_keys=True, separators=(",", ":")),
            flush=True,
        )

    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    model.train()
    ddp_model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        init_sync=False,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
        bucket_cap_mb=128,
    )
    trainable = [parameter for parameter in ddp_model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
        foreach=False,
    )
    scheduler = scheduler_for(optimizer, warmup_steps, schedule_steps)
    start_step = 0
    if args.resume:
        start_step = load_training_state(
            args.resume,
            optimizer,
            scheduler,
            plan_sha,
            rank,
            world,
        )
    if start_step >= len(plan):
        raise RuntimeError(
            f"resume step {start_step} is at or beyond plan length {len(plan)}"
        )

    wrapped_dataset = ExactSampleDataset(dataset)
    batch_sampler = RankBatchSampler(plan, rank, start_step)
    dataloader = DataLoader(
        wrapped_dataset,
        batch_sampler=batch_sampler,
        collate_fn=lambda batch: collate(
            batch,
            tokenizer.pad_token_id,
            args.pad_to_multiple,
        ),
        pin_memory=False,
        num_workers=0,
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    allocator_baseline = torch.cuda.memory_stats(device)
    local_uma_baseline = {
        "rank": rank,
        "swap_used_bytes": swap_used_bytes(),
        "allocator_retries": allocator_baseline["num_alloc_retries"],
        "ooms": allocator_baseline["num_ooms"],
    }
    uma_baselines = [None for _ in range(world)]
    dist.all_gather_object(uma_baselines, local_uma_baseline)
    if any(
        value["swap_used_bytes"] > args.max_swap_used_bytes
        for value in uma_baselines
    ):
        raise RuntimeError(
            "UMA baseline exceeds the swap contract: "
            f"allowed={args.max_swap_used_bytes} ranks={uma_baselines}"
        )
    args.uma_baseline = uma_baselines
    if is_main:
        print(
            "UMA_BASELINE_RECEIPT "
            + json.dumps(
                {
                    "maximum_swap_used_bytes": args.max_swap_used_bytes,
                    "ranks": uma_baselines,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
    completed_steps = start_step
    last_cache_release_step = start_step
    memory_guard_exit = False
    qualification_useful = 0
    qualification_padded = 0
    qualification_labels = 0
    qualification_seconds = 0.0
    qualification_observations = []
    bucket_metrics = {}
    timing_names = (
        "step_wall",
        "forward_gpu",
        "global_real_collective_gpu",
        "backward_ddp_gpu",
        "optimizer_gpu",
        "stats_collective_gpu",
        "memory_guard",
        "guard_observe_before",
        "guard_pre_collective",
        "python_gc",
        "empty_cache",
        "reclaim_sync",
        "guard_observe_after",
        "guard_post_collective",
        "guard_exit_collective",
        "final_sync",
    )
    qualification_timing_totals = {
        name: 0.0 for name in timing_names
    }
    checkpoint_receipts = []
    memory_observations = []
    shape_memory = {}
    probe_name = next(
        name
        for name, parameter in ddp_model.module.named_parameters()
        if parameter.requires_grad and "lora_B" in name and "down_proj" in name
    )
    probe_reference = dict(ddp_model.module.named_parameters())[
        probe_name
    ].detach().float().clone()
    thermal_hot_steps = 0
    thermal_samples = 0
    graphics_clock_sum_mhz = 0.0
    thermal_observed = {
        "maximum_board_celsius": None,
        "maximum_gpu_celsius": None,
        "maximum_power_watts": None,
        "minimum_graphics_clock_mhz": None,
        "maximum_graphics_clock_mhz": None,
    }
    args.uma_observed = {
        "cache_reclaim_rank_events": 0,
        "periodic_cache_release_rank_events": 0,
        "emergency_cache_release_rank_events": 0,
        "maximum_cache_release_gap_steps": 0,
        "memory_guard_exit_rank_events": 0,
        "maximum_swap_used_bytes": 0,
        "maximum_swap_growth_bytes": 0,
        "maximum_allocator_retry_growth": 0,
        "maximum_oom_growth": 0,
        "maximum_allocated_bytes": 0,
        "maximum_reserved_bytes": 0,
        "maximum_active_bytes": 0,
        "maximum_inactive_split_bytes": 0,
        "maximum_peak_allocated_bytes": 0,
        "maximum_peak_reserved_bytes": 0,
        "maximum_incremental_demand_bytes": 0,
        "minimum_cuda_free_before_bytes": None,
        "minimum_cuda_free_after_bytes": None,
        "minimum_system_available_before_bytes": None,
        "minimum_system_available_after_bytes": None,
        "minimum_cuda_free_after_reclaim_bytes": None,
        "minimum_system_available_after_reclaim_bytes": None,
        "maximum_timing_seconds": {
            name: 0.0 for name in timing_names
        },
    }

    for offset, batch in enumerate(dataloader):
        plan_index = start_step + offset
        if args.max_steps and completed_steps >= args.max_steps:
            break
        group = plan[plan_index]
        batch = {key: value.to(device, non_blocking=False) for key, value in batch.items()}
        padded_group_length = (
            (group["max_length"] + args.pad_to_multiple - 1)
            // args.pad_to_multiple
        ) * args.pad_to_multiple
        if args.selective_ac_start:
            checkpointed_layers = configure_selective_activation_checkpointing(
                checkpoint_layers,
                padded_group_length,
                args.selective_ac_start,
                args.selective_ac_budget,
            )
        elif args.gradient_checkpointing:
            checkpointed_layers = len(checkpoint_layers)
        else:
            checkpointed_layers = 0
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        allocated_before_step = torch.cuda.memory_allocated(device)
        step_started = time.perf_counter()
        forward_started = torch.cuda.Event(enable_timing=True)
        forward_finished = torch.cuda.Event(enable_timing=True)
        global_real_collective_started = torch.cuda.Event(
            enable_timing=True
        )
        global_real_collective_finished = torch.cuda.Event(
            enable_timing=True
        )
        backward_started = torch.cuda.Event(enable_timing=True)
        backward_finished = torch.cuda.Event(enable_timing=True)
        optimizer_started = torch.cuda.Event(enable_timing=True)
        optimizer_finished = torch.cuda.Event(enable_timing=True)
        stats_collective_started = torch.cuda.Event(enable_timing=True)
        stats_collective_finished = torch.cuda.Event(enable_timing=True)
        optimizer.zero_grad(set_to_none=True)

        shift_labels = batch["labels"][:, 1:].contiguous()
        label_mask = shift_labels != -100
        per_sample_tokens = label_mask.sum(dim=1)
        if torch.any(batch["real_mask"] & (per_sample_tokens == 0)):
            raise RuntimeError("real SFT sample has no shifted assistant labels")
        forward_started.record()
        local_loss_sum = ddp_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            real_mask=batch["real_mask"],
        )
        forward_finished.record()
        local_real = batch["real_mask"].sum().to(torch.float64)
        global_real = local_real.clone()
        global_real_collective_started.record()
        dist.all_reduce(global_real, op=dist.ReduceOp.SUM)
        global_real_collective_finished.record()
        loss = local_loss_sum * (world / global_real.item())
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at plan step {plan_index + 1}")

        backward_started.record()
        loss.backward()
        backward_finished.record()
        optimizer_started.record()
        missing_grads = sum(
            parameter.grad is None for parameter in trainable
        )
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        optimizer_finished.record()
        completed_steps = plan_index + 1

        useful_local = (
            batch["attention_mask"].sum(dim=1) * batch["real_mask"]
        ).sum()
        padded_local = torch.tensor(
            batch["input_ids"].numel(),
            dtype=torch.long,
            device=device,
        )
        labels_local = (label_mask.sum(dim=1) * batch["real_mask"]).sum()
        loss_sum_detached = local_loss_sum.detach().to(torch.float64)
        stats = torch.stack(
            [
                useful_local.to(torch.float64),
                padded_local.to(torch.float64),
                labels_local.to(torch.float64),
                loss_sum_detached,
                local_real,
                torch.tensor(float(missing_grads), dtype=torch.float64, device=device),
            ]
        )
        stats_collective_started.record()
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        stats_collective_finished.record()
        useful = int(stats[0].item())
        padded = int(stats[1].item())
        labels = int(stats[2].item())
        mean_loss = stats[3].item() / max(1.0, stats[4].item())
        missing_grads_sum = int(stats[5].item())
        if missing_grads_sum:
            raise RuntimeError(
                f"missing LoRA gradients at plan step {plan_index + 1}: "
                f"{missing_grads_sum}"
            )
        del (
            batch,
            global_real,
            label_mask,
            labels_local,
            local_loss_sum,
            local_real,
            loss,
            loss_sum_detached,
            padded_local,
            per_sample_tokens,
            shift_labels,
            stats,
            useful_local,
        )
        headroom_by_rank, memory_guard_exit, guard_times = enforce_uma_headroom(
            device,
            completed_steps,
            last_cache_release_step,
            args.cache_release_interval_steps,
            args.cache_release_below_available_bytes,
            args.min_after_cache_release_bytes,
            args.max_swap_used_bytes,
            local_uma_baseline["swap_used_bytes"],
            local_uma_baseline["allocator_retries"],
            local_uma_baseline["ooms"],
            allocated_before_step,
            world,
        )
        step_minimum_cuda_before = min(
            int(values[0].item()) for values in headroom_by_rank
        )
        step_minimum_cuda_after = min(
            int(values[1].item()) for values in headroom_by_rank
        )
        step_minimum_system_before = min(
            int(values[2].item()) for values in headroom_by_rank
        )
        step_minimum_system_after = min(
            int(values[3].item()) for values in headroom_by_rank
        )
        step_maximum_swap = max(
            max(int(values[4].item()), int(values[5].item()))
            for values in headroom_by_rank
        )
        reclaimed_by_rank = [
            values
            for values in headroom_by_rank
            if int(values[12].item())
        ]
        args.uma_observed["cache_reclaim_rank_events"] += sum(
            int(values[12].item()) for values in headroom_by_rank
        )
        args.uma_observed["periodic_cache_release_rank_events"] += sum(
            int(values[21].item()) for values in headroom_by_rank
        )
        args.uma_observed["emergency_cache_release_rank_events"] += sum(
            int(values[22].item()) for values in headroom_by_rank
        )
        args.uma_observed["maximum_cache_release_gap_steps"] = max(
            args.uma_observed["maximum_cache_release_gap_steps"],
            max(int(values[23].item()) for values in headroom_by_rank),
        )
        if reclaimed_by_rank:
            last_cache_release_step = completed_steps
        args.uma_observed["memory_guard_exit_rank_events"] += sum(
            int(values[13].item()) for values in headroom_by_rank
        )
        args.uma_observed["maximum_swap_used_bytes"] = max(
            args.uma_observed["maximum_swap_used_bytes"],
            step_maximum_swap,
        )
        for key, value in (
            (
                "minimum_cuda_free_before_bytes",
                step_minimum_cuda_before,
            ),
            (
                "minimum_cuda_free_after_bytes",
                step_minimum_cuda_after,
            ),
            (
                "minimum_system_available_before_bytes",
                step_minimum_system_before,
            ),
            (
                "minimum_system_available_after_bytes",
                step_minimum_system_after,
            ),
        ):
            previous = args.uma_observed[key]
            args.uma_observed[key] = (
                value if previous is None else min(previous, value)
            )
        if reclaimed_by_rank:
            for key, value in (
                (
                    "minimum_cuda_free_after_reclaim_bytes",
                    min(
                        int(values[1].item())
                        for values in reclaimed_by_rank
                    ),
                ),
                (
                    "minimum_system_available_after_reclaim_bytes",
                    min(
                        int(values[3].item())
                        for values in reclaimed_by_rank
                    ),
                ),
            ):
                previous = args.uma_observed[key]
                args.uma_observed[key] = (
                    value if previous is None else min(previous, value)
                )
        for key, index in (
            ("maximum_allocated_bytes", 6),
            ("maximum_reserved_bytes", 7),
            ("maximum_active_bytes", 8),
            ("maximum_inactive_split_bytes", 9),
            ("maximum_allocator_retry_growth", 14),
            ("maximum_oom_growth", 15),
            ("maximum_swap_growth_bytes", 16),
            ("maximum_peak_allocated_bytes", 18),
            ("maximum_peak_reserved_bytes", 19),
            ("maximum_incremental_demand_bytes", 20),
        ):
            args.uma_observed[key] = max(
                args.uma_observed[key],
                max(int(values[index].item()) for values in headroom_by_rank),
            )
        incremental_by_rank = [
            int(values[20].item()) for values in headroom_by_rank
        ]
        shape_key = f"{group['bucket']}:{padded_group_length}"
        shape_value = shape_memory.setdefault(
            shape_key,
            {
                "bucket": group["bucket"],
                "padded_length": padded_group_length,
                "steps": 0,
                "steps_after_128": 0,
                "maximum_incremental_demand_bytes": 0,
                "maximum_cross_rank_spread_bytes": 0,
            },
        )
        shape_value["steps"] += 1
        if completed_steps > 128:
            shape_value["steps_after_128"] += 1
        shape_value["maximum_incremental_demand_bytes"] = max(
            shape_value["maximum_incremental_demand_bytes"],
            max(incremental_by_rank),
        )
        shape_value["maximum_cross_rank_spread_bytes"] = max(
            shape_value["maximum_cross_rank_spread_bytes"],
            max(incremental_by_rank) - min(incremental_by_rank),
        )
        memory_observations.append(
            {
                "step": completed_steps,
                "minimum_cuda_free_after_bytes": (
                    step_minimum_cuda_after
                ),
                "minimum_system_available_after_bytes": (
                    step_minimum_system_after
                ),
                "maximum_incremental_demand_bytes": max(
                    incremental_by_rank
                ),
                "maximum_reserved_unallocated_bytes": max(
                    max(
                        0,
                        int(values[7].item())
                        - int(values[6].item()),
                    )
                    for values in headroom_by_rank
                ),
                "maximum_inactive_split_bytes": max(
                    int(values[9].item())
                    for values in headroom_by_rank
                ),
                "reclaim_rank_events": len(reclaimed_by_rank),
                "shape": shape_key,
            }
        )
        final_sync_started = time.perf_counter()
        torch.cuda.synchronize(device)
        final_sync_seconds = time.perf_counter() - final_sync_started
        local_timing = {
            "step_wall": time.perf_counter() - step_started,
            "forward_gpu": forward_started.elapsed_time(
                forward_finished
            )
            / 1000,
            "global_real_collective_gpu": (
                global_real_collective_started.elapsed_time(
                    global_real_collective_finished
                )
                / 1000
            ),
            "backward_ddp_gpu": backward_started.elapsed_time(
                backward_finished
            )
            / 1000,
            "optimizer_gpu": optimizer_started.elapsed_time(
                optimizer_finished
            )
            / 1000,
            "stats_collective_gpu": stats_collective_started.elapsed_time(
                stats_collective_finished
            )
            / 1000,
            "memory_guard": guard_times["total"],
            "guard_observe_before": guard_times["observe_before"],
            "guard_pre_collective": guard_times["pre_collective"],
            "python_gc": guard_times["python_gc"],
            "empty_cache": guard_times["empty_cache"],
            "reclaim_sync": guard_times["reclaim_sync"],
            "guard_observe_after": guard_times["observe_after"],
            "guard_post_collective": guard_times["post_collective"],
            "guard_exit_collective": guard_times["exit_collective"],
            "final_sync": final_sync_seconds,
        }
        timing_vector = torch.tensor(
            [local_timing[name] for name in timing_names],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(timing_vector, op=dist.ReduceOp.MAX)
        step_timing = {
            name: timing_vector[index].item()
            for index, name in enumerate(timing_names)
        }
        seconds = step_timing["step_wall"]
        for name in timing_names:
            args.uma_observed["maximum_timing_seconds"][name] = max(
                args.uma_observed["maximum_timing_seconds"][name],
                step_timing[name],
            )

        if offset >= args.qualification_warmup_steps:
            qualification_useful += useful
            qualification_padded += padded
            qualification_labels += labels
            qualification_seconds += seconds
            qualification_observations.append(
                {
                    "bucket": group["bucket"],
                    "useful": useful,
                    "seconds": seconds,
                }
            )
            bucket_value = bucket_metrics.setdefault(
                group["bucket"],
                {"useful": 0, "padded": 0, "labels": 0, "seconds": 0.0},
            )
            bucket_value["useful"] += useful
            bucket_value["padded"] += padded
            bucket_value["labels"] += labels
            bucket_value["seconds"] += seconds
            for name in timing_names:
                qualification_timing_totals[name] += step_timing[name]

        should_log = (
            completed_steps <= 3
            or completed_steps % args.log_every == 0
            or completed_steps == len(plan)
        )
        if should_log:
            telemetry = board_metrics(device)
            telemetry_by_rank = [
                torch.zeros_like(telemetry) for _ in range(world)
            ]
            dist.all_gather(telemetry_by_rank, telemetry)
            hottest_board = max(values[3].item() for values in telemetry_by_rank)
            thermal_samples += len(telemetry_by_rank)
            graphics_clock_sum_mhz += sum(
                values[2].item() for values in telemetry_by_rank
            )
            for key, index, reducer in (
                ("maximum_board_celsius", 3, max),
                ("maximum_gpu_celsius", 1, max),
                ("maximum_power_watts", 0, max),
                ("minimum_graphics_clock_mhz", 2, min),
                ("maximum_graphics_clock_mhz", 2, max),
            ):
                observed = reducer(
                    values[index].item()
                    for values in telemetry_by_rank
                )
                previous = thermal_observed[key]
                thermal_observed[key] = (
                    observed
                    if previous is None
                    else reducer(previous, observed)
                )
            if hottest_board >= args.max_board_celsius:
                thermal_hot_steps += 1
            else:
                thermal_hot_steps = 0
            probe = dict(ddp_model.module.named_parameters())[probe_name].detach().float()
            probe_delta = (probe - probe_reference).abs().mean().item()
            if is_main:
                telemetry_text = ";".join(
                    (
                        f"r{metric_rank}:power={values[0].item():.1f}W,"
                        f"temp={values[1].item():.0f}C,"
                        f"clock={values[2].item():.0f}MHz,"
                        f"board={values[3].item():.0f}C,"
                        f"alloc={values[4].item():.1f}GB,"
                        f"res={values[5].item():.1f}GB,"
                        f"peak_alloc={values[6].item():.1f}GB,"
                        f"peak_res={values[7].item():.1f}GB"
                    )
                    for metric_rank, values in enumerate(telemetry_by_rank)
                )
                headroom_text = ";".join(
                    (
                        f"r{metric_rank}:cuda_free_before="
                        f"{values[0].item() / 1e9:.1f}GB,"
                        f"cuda_free_after={values[1].item() / 1e9:.1f}GB,"
                        f"sys_before={values[2].item() / 1e9:.1f}GB,"
                        f"sys_after={values[3].item() / 1e9:.1f}GB,"
                        f"swap_before={values[4].item() / 1e9:.3f}GB,"
                        f"swap_after={values[5].item() / 1e9:.3f}GB,"
                        f"alloc={values[6].item() / 1e9:.1f}GB,"
                        f"reserved={values[7].item() / 1e9:.1f}GB,"
                        f"active={values[8].item() / 1e9:.1f}GB,"
                        f"inactive_split={values[9].item() / 1e9:.1f}GB,"
                        f"allocator_retries={int(values[10].item())},"
                        f"ooms={int(values[11].item())},"
                        f"cache_reclaimed={int(values[12].item())},"
                        f"memory_exit={int(values[13].item())},"
                        f"allocator_retry_growth={int(values[14].item())},"
                        f"oom_growth={int(values[15].item())},"
                        f"swap_growth={int(values[16].item())},"
                        f"alloc_before={values[17].item() / 1e9:.1f}GB,"
                        f"peak_alloc={values[18].item() / 1e9:.1f}GB,"
                        f"peak_reserved={values[19].item() / 1e9:.1f}GB,"
                        f"incremental_demand={values[20].item() / 1e9:.1f}GB,"
                        f"periodic={int(values[21].item())},"
                        f"emergency={int(values[22].item())},"
                        f"release_gap={int(values[23].item())}"
                    )
                    for metric_rank, values in enumerate(headroom_by_rank)
                )
                if completed_steps < len(plan):
                    next_group = plan[completed_steps]
                    next_padded_length = (
                        (
                            next_group["max_length"]
                            + args.pad_to_multiple
                            - 1
                        )
                        // args.pad_to_multiple
                    ) * args.pad_to_multiple
                    next_shape_text = (
                        f"{next_group['bucket']}:{next_padded_length}"
                    )
                else:
                    next_shape_text = "complete"
                timing_text = ",".join(
                    f"{name}={step_timing[name]:.6f}"
                    for name in timing_names
                )
                print(
                    f"[step {completed_steps}/{len(plan)}] "
                    f"bucket={group['bucket']} local_batch={group['local_batch']} "
                    f"length={group['min_length']}..{group['max_length']} "
                    f"padded_length={padded_group_length} "
                    f"checkpointed_layers={checkpointed_layers} "
                    f"loss={mean_loss:.6f} lr={scheduler.get_last_lr()[0]:.3e} "
                    f"step_s={seconds:.3f} useful_tok_s={useful / seconds:.1f} "
                    f"padded_tok_s={padded / seconds:.1f} "
                    f"label_tok_s={labels / seconds:.1f} "
                    f"padding_ratio={padded / max(1, useful):.4f} "
                    f"missing_grads_sum={missing_grads_sum} "
                    f"probe={probe_name} mean_abs_delta={probe_delta:.9e} "
                    f"next_shape={next_shape_text} "
                    f"timing={timing_text} "
                    f"{telemetry_text} uma={headroom_text}",
                    flush=True,
                )
            if thermal_hot_steps >= args.thermal_persist_steps:
                raise RuntimeError(
                    "board/SoC thermal pull-off: "
                    f"{hottest_board:.0f}C >= {args.max_board_celsius:.0f}C "
                    f"for {thermal_hot_steps} consecutive measured steps"
                )

        if args.save_every and completed_steps % args.save_every == 0:
            checkpoint_started = time.perf_counter()
            path, adapter_sha = save_checkpoint(
                ddp_model,
                optimizer,
                scheduler,
                args.out,
                completed_steps,
                plan_sha,
                rank,
                world,
                args,
            )
            checkpoint_fingerprint = None
            if args.qualification_steps_per_bucket:
                checkpoint_fingerprint = lora_fingerprint(
                    ddp_model.module
                )
                all_equal(
                    json.dumps(
                        checkpoint_fingerprint,
                        sort_keys=True,
                    ),
                    world,
                )
            local_checkpoint_seconds = torch.tensor(
                [time.perf_counter() - checkpoint_started],
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(
                local_checkpoint_seconds,
                op=dist.ReduceOp.MAX,
            )
            checkpoint_receipt = {
                "step": completed_steps,
                "adapter_sha256": adapter_sha,
                "maximum_rank_seconds": (
                    local_checkpoint_seconds.item()
                ),
                "lora_fingerprint": checkpoint_fingerprint,
            }
            checkpoint_receipts.append(checkpoint_receipt)
            if is_main:
                print(
                    "CHECKPOINT_COMPLETE "
                    f"path={path} adapter_sha256={adapter_sha} "
                    f"maximum_rank_seconds="
                    f"{checkpoint_receipt['maximum_rank_seconds']:.6f} "
                    "lora_fingerprint="
                    + json.dumps(
                        checkpoint_fingerprint,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
        if memory_guard_exit:
            break

    if completed_steps == start_step:
        raise RuntimeError("trainer completed no optimizer steps")
    final_checkpoint = Path(args.out) / f"checkpoint-{completed_steps}"
    if not final_checkpoint.exists():
        checkpoint_started = time.perf_counter()
        path, adapter_sha = save_checkpoint(
            ddp_model,
            optimizer,
            scheduler,
            args.out,
            completed_steps,
            plan_sha,
            rank,
            world,
            args,
        )
        checkpoint_fingerprint = None
        if args.qualification_steps_per_bucket:
            checkpoint_fingerprint = lora_fingerprint(ddp_model.module)
            all_equal(
                json.dumps(checkpoint_fingerprint, sort_keys=True),
                world,
            )
        local_checkpoint_seconds = torch.tensor(
            [time.perf_counter() - checkpoint_started],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(
            local_checkpoint_seconds,
            op=dist.ReduceOp.MAX,
        )
        checkpoint_receipts.append(
            {
                "step": completed_steps,
                "adapter_sha256": adapter_sha,
                "maximum_rank_seconds": (
                    local_checkpoint_seconds.item()
                ),
                "lora_fingerprint": checkpoint_fingerprint,
            }
        )
        if is_main:
            print(
                "CHECKPOINT_COMPLETE "
                f"path={path} adapter_sha256={adapter_sha} "
                f"maximum_rank_seconds="
                f"{local_checkpoint_seconds.item():.6f} "
                "lora_fingerprint="
                + json.dumps(
                    checkpoint_fingerprint,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                flush=True,
            )
    final_checksums = json.loads(
        (final_checkpoint / "SHA256SUMS.json").read_text(encoding="utf-8")
    )
    final_adapter_sha = final_checksums["adapter_model.safetensors"]
    if file_sha256(
        final_checkpoint / "adapter_model.safetensors"
    ) != final_adapter_sha:
        raise RuntimeError(
            f"final adapter checksum failed: {final_checkpoint}"
        )
    all_equal(final_adapter_sha, world)
    final_lora_fingerprint = (
        checkpoint_receipts[-1]["lora_fingerprint"]
        if checkpoint_receipts
        and checkpoint_receipts[-1]["step"] == completed_steps
        and checkpoint_receipts[-1]["lora_fingerprint"] is not None
        else lora_fingerprint(ddp_model.module)
    )
    all_equal(
        json.dumps(final_lora_fingerprint, sort_keys=True),
        world,
    )
    if (
        not final_lora_fingerprint["finite"]
        or final_lora_fingerprint["sha256"] == fingerprint["sha256"]
    ):
        raise RuntimeError(
            "full-adapter update fingerprint is null or non-finite"
        )
    final_lora_delta = None
    if initial_trainable_snapshot is not None:
        final_lora_delta = lora_delta_receipt(
            ddp_model.module,
            initial_trainable_snapshot,
        )
        all_equal(json.dumps(final_lora_delta, sort_keys=True), world)
        if (
            not final_lora_delta["finite"]
            or final_lora_delta["changed_elements"] == 0
            or final_lora_delta["mean_abs"] <= 0
        ):
            raise RuntimeError(
                "full-adapter delta receipt is null or non-finite"
            )
    if thermal_samples == 0:
        raise RuntimeError("training produced no board/clock telemetry")
    thermal_receipt = {
        **thermal_observed,
        "observations": thermal_samples,
        "mean_graphics_clock_mhz": (
            graphics_clock_sum_mhz / thermal_samples
        ),
        "pull_off_celsius": args.max_board_celsius,
    }
    topology_receipt = {
        "world_size": world,
        "max_seq": args.max_seq,
        "pad_to_multiple": args.pad_to_multiple,
        "thresholds": [args.short_max, args.mid_max],
        "local_batches": [
            args.short_batch,
            args.mid_batch,
            args.long_batch,
        ],
        "lora": {
            "rank": args.rank,
            "alpha": args.alpha,
            "dropout": args.dropout,
            "target_modules": args.target_modules.split(","),
            "initial_fingerprint": fingerprint,
        },
        "optimizer": {
            "name": "AdamW-fp32-adapters",
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
        "schedule_steps": schedule_steps,
        "warmup_steps": warmup_steps,
        "activation_checkpointing": args.activation_checkpoint_receipt,
        "memory": {
            "cache_release_interval_steps": (
                args.cache_release_interval_steps
            ),
            "cache_release_below_available_bytes": (
                args.cache_release_below_available_bytes
            ),
            "minimum_after_cache_release_bytes": (
                args.min_after_cache_release_bytes
            ),
            "maximum_swap_used_bytes": args.max_swap_used_bytes,
            "reclaim_python_gc": False,
            "pass_fail_signal": "system_MemAvailable",
            "cuda_free": "telemetry_only",
        },
        "allocator_configuration": {
            "PYTORCH_CUDA_ALLOC_CONF": os.environ.get(
                "PYTORCH_CUDA_ALLOC_CONF"
            ),
            "PYTORCH_ALLOC_CONF": os.environ.get("PYTORCH_ALLOC_CONF"),
        },
    }

    if args.qualification_steps_per_bucket:
        if memory_guard_exit:
            result = {
                "adapter_checkpoint": str(final_checkpoint),
                "adapter_sha256": final_adapter_sha,
                "qualification_plan_sha256": plan_sha,
                "production_plan_sha256": production_plan_sha,
                "completed_steps": completed_steps,
                "checkpoint_receipts": checkpoint_receipts,
                "dataset_receipt": receipt,
                "final_lora_fingerprint": final_lora_fingerprint,
                "final_lora_delta": final_lora_delta,
                "reason": "post-step-memory-invariant",
                "thresholds": {
                    "system_no_release_floor_bytes": (
                        args.cache_release_below_available_bytes
                    ),
                    "system_post_release_floor_bytes": (
                        args.min_after_cache_release_bytes
                    ),
                    "maximum_swap_used_bytes": args.max_swap_used_bytes,
                    "zero_allocator_retry_growth": True,
                    "zero_oom_growth": True,
                },
                "qualification_plan": args.qualification_plan_receipt,
                "reclaim_python_gc": False,
                "thermal": thermal_receipt,
                "topology": topology_receipt,
                "uma_baseline": args.uma_baseline,
                "uma": args.uma_observed,
            }
            if is_main:
                print(
                    "QUALIFICATION_REJECTED "
                    + json.dumps(result, sort_keys=True, separators=(",", ":")),
                    flush=True,
                )
                write_json(Path(args.out) / "QUALIFICATION_REJECTED.json", result)
            dist.barrier()
            raise SystemExit(2)
        if qualification_seconds <= 0:
            raise RuntimeError(
                "qualification produced no post-warmup throughput window"
            )
        measured_useful_tps = qualification_useful / qualification_seconds
        padded_tps = qualification_padded / qualification_seconds
        label_tps = qualification_labels / qualification_seconds
        projected_seconds = 0.0
        for bucket, full_value in bucket_summary.items():
            measured = bucket_metrics.get(bucket)
            if not measured or measured["seconds"] <= 0:
                raise RuntimeError(
                    f"qualification has no measured throughput for {bucket}"
                )
            bucket_useful_tps = measured["useful"] / measured["seconds"]
            projected_seconds += full_value["tokens"] / bucket_useful_tps
        projected_useful_tps = receipt["input_tokens"] / projected_seconds
        throughput_lower_bounds = bootstrap_throughput_lower_bounds(
            qualification_observations,
            {
                bucket: value["tokens"]
                for bucket, value in bucket_summary.items()
            },
            args.seed + 15_817,
        )
        required_headroom_by_shape = {
            shape: (
                value["maximum_incremental_demand_bytes"]
                + value["maximum_cross_rank_spread_bytes"]
            )
            for shape, value in shape_memory.items()
        }
        worst_next_shape_requirement = max(
            required_headroom_by_shape.values()
        )
        cuda_session_projection = conservative_linear_projection_lower(
            [
                {
                    "step": value["step"],
                    "bytes": value[
                        "minimum_cuda_free_after_bytes"
                    ],
                }
                for value in memory_observations
            ],
            250,
        )
        system_session_projection = (
            conservative_linear_projection_lower(
                [
                    {
                        "step": value["step"],
                        "bytes": value[
                            "minimum_system_available_after_bytes"
                        ],
                    }
                    for value in memory_observations
                ],
                250,
            )
        )
        maximum_shape_coverage = {}
        for bucket in ("short", "mid", "long"):
            bucket_shapes = [
                value
                for value in shape_memory.values()
                if value["bucket"] == bucket
            ]
            maximum_padded = max(
                value["padded_length"] for value in bucket_shapes
            )
            maximum_shape_coverage[bucket] = sum(
                value["steps_after_128"]
                for value in bucket_shapes
                if value["padded_length"] == maximum_padded
            )
        system_memory_projection_diagnostic_passed = (
            system_session_projection[
                "one_sided_95pct_lower_bytes"
            ]
            >= worst_next_shape_requirement
            and all(
                value >= 2
                for value in maximum_shape_coverage.values()
            )
        )
        memory_projection_receipt = {
            "target_session_step": 250,
            "required_headroom_by_shape": (
                required_headroom_by_shape
            ),
            "worst_next_shape_requirement_bytes": (
                worst_next_shape_requirement
            ),
            "maximum_shape_steps_after_128": (
                maximum_shape_coverage
            ),
            "system_available": system_session_projection,
            "maximum_reserved_unallocated_bytes": max(
                value["maximum_reserved_unallocated_bytes"]
                for value in memory_observations
            ),
            "maximum_inactive_split_bytes": max(
                value["maximum_inactive_split_bytes"]
                for value in memory_observations
            ),
            "hard_gate": "none_diagnostic_only",
            "system_projection_diagnostic_passed": (
                system_memory_projection_diagnostic_passed
            ),
            "cuda_free": {
                **cuda_session_projection,
                "hard_gate": False,
            },
        }
        result = {
            "adapter_checkpoint": str(final_checkpoint),
            "adapter_sha256": final_adapter_sha,
            "qualification_plan_sha256": plan_sha,
            "production_plan_sha256": production_plan_sha,
            "completed_steps": completed_steps,
            "checkpoint_receipts": checkpoint_receipts,
            "dataset_receipt": receipt,
            "final_lora_fingerprint": final_lora_fingerprint,
            "final_lora_delta": final_lora_delta,
            "measured_steps": (
                completed_steps - start_step - args.qualification_warmup_steps
            ),
            "measured_useful_input_tok_s": measured_useful_tps,
            "projected_full_pass_useful_input_tok_s": projected_useful_tps,
            "projected_full_pass_seconds": projected_seconds,
            "padded_compute_tok_s": padded_tps,
            "assistant_label_tok_s": label_tps,
            "minimum_useful_input_tok_s": args.min_useful_tps,
            "one_sided_throughput_lower_bounds": (
                throughput_lower_bounds
            ),
            "throughput_contract": {
                "hard_gate": "point_estimates_at_or_above_minimum",
                "bootstrap_lower_bounds": "diagnostic_only",
            },
            "session_memory_projection": (
                memory_projection_receipt
            ),
            "pad_to_multiple": args.pad_to_multiple,
            "qualification_plan": args.qualification_plan_receipt,
            "reclaim_python_gc": False,
            "thermal": thermal_receipt,
            "topology": topology_receipt,
            "uma_baseline": args.uma_baseline,
            "timing": {
                "maximum_rank_totals_seconds": (
                    qualification_timing_totals
                ),
                "maximum_rank_mean_seconds": {
                    name: qualification_timing_totals[name]
                    / (
                        completed_steps
                        - start_step
                        - args.qualification_warmup_steps
                    )
                    for name in timing_names
                },
            },
            "uma": args.uma_observed,
            "buckets": {
                bucket: {
                    "useful_input_tok_s": value["useful"] / value["seconds"],
                    "padded_compute_tok_s": value["padded"] / value["seconds"],
                    "assistant_label_tok_s": value["labels"] / value["seconds"],
                    "seconds": value["seconds"],
                }
                for bucket, value in bucket_metrics.items()
            },
        }
        point_estimates_passed = (
            measured_useful_tps >= args.min_useful_tps
            and projected_useful_tps >= args.min_useful_tps
        )
        bootstrap_lower_bounds_passed = (
            throughput_lower_bounds[
                "measured_useful_input_tok_s"
            ]
            >= args.min_useful_tps
            and throughput_lower_bounds[
                "projected_full_pass_useful_input_tok_s"
            ]
            >= args.min_useful_tps
        )
        throughput_passed = point_estimates_passed
        memory_passed = (
            args.uma_observed["minimum_system_available_after_bytes"]
            >= args.cache_release_below_available_bytes
            and args.uma_observed[
                "minimum_system_available_after_reclaim_bytes"
            ]
            is not None
            and args.uma_observed[
                "minimum_system_available_after_reclaim_bytes"
            ]
            >= args.min_after_cache_release_bytes
            and args.uma_observed["memory_guard_exit_rank_events"] == 0
            and args.uma_observed["maximum_swap_used_bytes"]
            <= args.max_swap_used_bytes
            and args.uma_observed["maximum_swap_growth_bytes"] == 0
            and args.uma_observed[
                "maximum_allocator_retry_growth"
            ]
            == 0
            and args.uma_observed["maximum_oom_growth"] == 0
        )
        result["gates"] = {
            "measured_and_projected_throughput": throughput_passed,
            "point_estimates_at_or_above_minimum": point_estimates_passed,
            "one_sided_95pct_lower_bounds_at_or_above_minimum": (
                bootstrap_lower_bounds_passed
            ),
            "system_no_release_floor_bytes": (
                args.cache_release_below_available_bytes
            ),
            "system_post_release_floor_bytes": (
                args.min_after_cache_release_bytes
            ),
            "cuda_free_hard_gate": False,
            "maximum_swap_used_bytes": args.max_swap_used_bytes,
            "zero_swap_growth": (
                args.uma_observed["maximum_swap_growth_bytes"] == 0
            ),
            "zero_allocator_retry_growth": (
                args.uma_observed["maximum_allocator_retry_growth"] == 0
            ),
            "zero_oom_growth": (
                args.uma_observed["maximum_oom_growth"] == 0
            ),
            "projected_250_step_system_headroom_diagnostic": (
                system_memory_projection_diagnostic_passed
            ),
            "receipt_faithful_system_memory": memory_passed,
        }
        passed = throughput_passed and memory_passed
        marker = "QUALIFICATION_PASSED" if passed else "QUALIFICATION_REJECTED"
        if is_main:
            print(
                marker
                + " "
                + json.dumps(result, sort_keys=True, separators=(",", ":")),
                flush=True,
            )
            write_json(Path(args.out) / f"{marker}.json", result)
        dist.barrier()
        if not passed:
            raise SystemExit(2)
    elif is_main:
        if qualification_seconds <= 0:
            raise RuntimeError("production session has no measured throughput window")
        print(
            f"TRAINING_PASS_COMPLETE steps={completed_steps} "
            f"requested_steps={args.max_steps or len(plan)} "
            f"memory_guard_exit={int(memory_guard_exit)} "
            f"checkpoint={final_checkpoint} "
            f"measured_useful_input_tok_s="
            f"{qualification_useful / qualification_seconds:.9f} "
            f"padded_compute_tok_s="
            f"{qualification_padded / qualification_seconds:.9f} "
            f"assistant_label_tok_s="
            f"{qualification_labels / qualification_seconds:.9f} "
            f"uma_min_cuda_free_after="
            f"{args.uma_observed['minimum_cuda_free_after_bytes']} "
            f"uma_min_cuda_free_before="
            f"{args.uma_observed['minimum_cuda_free_before_bytes']} "
            f"uma_min_system_available_after="
            f"{args.uma_observed['minimum_system_available_after_bytes']} "
            f"uma_min_system_available_after_reclaim="
            f"{args.uma_observed['minimum_system_available_after_reclaim_bytes']} "
            f"uma_max_swap={args.uma_observed['maximum_swap_used_bytes']} "
            f"uma_max_swap_growth="
            f"{args.uma_observed['maximum_swap_growth_bytes']} "
            f"uma_max_allocator_retry_growth="
            f"{args.uma_observed['maximum_allocator_retry_growth']} "
            f"uma_max_oom_growth="
            f"{args.uma_observed['maximum_oom_growth']} "
            f"cache_reclaim_rank_events="
            f"{args.uma_observed['cache_reclaim_rank_events']} "
            f"memory_guard_exit_rank_events="
            f"{args.uma_observed['memory_guard_exit_rank_events']}",
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
