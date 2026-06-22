from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


@dataclass(frozen=True)
class ExpertSliceSpec:
    """One contiguous expert slice inside an unsharded flat parameter."""

    layer_num: int
    param_fqn: str
    expert_id: int
    start: int  # element offset inside the UNsharded flat grad
    end: int    # exclusive element offset inside the UNsharded flat grad


@dataclass
class FlatMaskSpec:
    """Mask plan for one FSDP flat parameter."""

    module_name: str
    layer_num: int
    flat_param_numel_local: int
    flat_param_fqns: Tuple[str, ...]
    ranges: List[Tuple[int, int]]  # element ranges in the UNsharded flat grad
    slices: List[ExpertSliceSpec]
    handle: torch.utils.hooks.RemovableHandle | None = None
    fired_steps: int = 0




def keystone_pairs_to_layer_map(
    pairs: Iterable[Sequence[int]],
    keystone_layers: Sequence[int],
) -> Dict[int, List[int]]:
    """
    Convert JSON-style `[[keystone_idx, expert_id], ...]` pairs into
    `{actual_layer_num: [expert_ids...]}`.
    """
    out: Dict[int, List[int]] = {}
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError(f"Expected [keystone_idx, expert_id], got {pair!r}")
        keystone_idx, expert_id = int(pair[0]), int(pair[1])
        if not (0 <= keystone_idx < len(keystone_layers)):
            raise ValueError(
                f"keystone_idx {keystone_idx} out of range for keystone_layers={tuple(keystone_layers)}"
            )
        layer_num = int(keystone_layers[keystone_idx])
        out.setdefault(layer_num, []).append(int(expert_id))
    for layer_num in list(out):
        out[layer_num] = sorted(set(out[layer_num]))
    return out
def _iter_fsdp_named_modules(root: nn.Module):
    for module_name, module in root.named_modules():
        if isinstance(module, FSDP):
            yield module_name, module


def _parse_layer_num(module_name: str, flat_fqns: Sequence[str]) -> int | None:
    # Prefer the FSDP module path, which in practice looks like
    # `_fsdp_wrapped_module.model.layers.7` for nested decoder-layer wraps.
    m = re.search(r"layers\.(\d+)", module_name)
    if m:
        return int(m.group(1))
    # Fallback: some custom wrapping schemes may leave the layer number only in
    # the flattened FQNs.
    for fqn in flat_fqns:
        m = re.search(r"layers\.(\d+)", fqn)
        if m:
            return int(m.group(1))
    return None


def _build_unsharded_flat_layout(flat_param) -> List[Tuple[str | None, int, int, torch.Size | None]]:
    """
    Returns a list of (fqn_or_None, start, end, shape_or_None) covering the
    UNsharded flat parameter. Padding blocks appear as fqn_or_None=None.

    This relies on FSDP1 private metadata that exists in PyTorch 2.10:
      - _fqns
      - _shapes
      - _numels_with_padding
      - _is_padding_mask
    """
    fqns: Sequence[str] = tuple(flat_param._fqns)  # private, but stable in 2.10
    shapes: Sequence[torch.Size] = tuple(flat_param._shapes)
    numels_with_padding: Sequence[int] = tuple(flat_param._numels_with_padding)
    is_padding_mask: Sequence[bool] = tuple(flat_param._is_padding_mask)

    layout: List[Tuple[str | None, int, int, torch.Size | None]] = []
    param_idx = 0
    offset = 0
    for block_numel, is_padding in zip(numels_with_padding, is_padding_mask):
        start = offset
        end = offset + int(block_numel)
        if is_padding:
            layout.append((None, start, end, None))
        else:
            layout.append((fqns[param_idx], start, end, shapes[param_idx]))
            param_idx += 1
        offset = end

    if param_idx != len(fqns):
        raise RuntimeError(
            f"Flat-param metadata mismatch: consumed {param_idx} params but saw {len(fqns)} fqns"
        )
    return layout


def _is_target_expert_tensor(param_fqn: str) -> bool:
    # Match both decoder-layer flat params (`mlp.experts.gate_up_proj`) and
    # experts-only wraps (`gate_up_proj`). Exclude shared_expert tensors.
    if "shared_expert" in param_fqn:
        return False
    return (
        param_fqn.endswith("mlp.experts.gate_up_proj")
        or param_fqn.endswith("mlp.experts.down_proj")
        or param_fqn == "mlp.experts.gate_up_proj"
        or param_fqn == "mlp.experts.down_proj"
        or param_fqn == "experts.gate_up_proj"
        or param_fqn == "experts.down_proj"
        or param_fqn == "gate_up_proj"
        or param_fqn == "down_proj"
    )


def _coalesce_ranges(ranges: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not ranges:
        return []
    sorted_ranges = sorted((int(s), int(e)) for s, e in ranges)
    out: List[List[int]] = []
    for start, end in sorted_ranges:
        if start >= end:
            continue
        if not out or start > out[-1][1]:
            out.append([start, end])
        else:
            out[-1][1] = max(out[-1][1], end)
    return [(start, end) for start, end in out]


def _build_mask_spec_for_fsdp_module(
    module_name: str,
    fsdp_module: FSDP,
    frozen_expert_ids: Sequence[int],
    num_experts: int = 256,
) -> FlatMaskSpec | None:
    handle = getattr(fsdp_module, "_handle", None)
    if handle is None:
        return None
    flat_param = handle.flat_param
    flat_fqns: Tuple[str, ...] = tuple(flat_param._fqns)
    layer_num = _parse_layer_num(module_name, flat_fqns)
    if layer_num is None:
        return None

    frozen_ids = sorted(set(int(eid) for eid in frozen_expert_ids))
    layout = _build_unsharded_flat_layout(flat_param)

    raw_ranges: List[Tuple[int, int]] = []
    slice_specs: List[ExpertSliceSpec] = []
    for param_fqn, start, end, shape in layout:
        if param_fqn is None or shape is None:
            continue
        if not _is_target_expert_tensor(param_fqn):
            continue
        if len(shape) != 3:
            raise RuntimeError(
                f"Expected expert tensor {param_fqn} to be 3D, got shape={tuple(shape)}"
            )
        if int(shape[0]) != int(num_experts):
            raise RuntimeError(
                f"Expected {param_fqn} first dim to be {num_experts}, got shape={tuple(shape)}"
            )

        expert_stride = math.prod(int(d) for d in shape[1:])
        block_numel = end - start
        expected_numel = int(shape[0]) * expert_stride
        if block_numel != expected_numel:
            raise RuntimeError(
                f"Unexpected block size for {param_fqn}: flat block={block_numel}, expected={expected_numel}"
            )

        for expert_id in frozen_ids:
            if not (0 <= expert_id < num_experts):
                raise ValueError(f"Expert id {expert_id} out of range for {param_fqn}")
            s = start + expert_id * expert_stride
            e = s + expert_stride
            raw_ranges.append((s, e))
            slice_specs.append(
                ExpertSliceSpec(
                    layer_num=layer_num,
                    param_fqn=param_fqn,
                    expert_id=expert_id,
                    start=s,
                    end=e,
                )
            )

    if not raw_ranges:
        return None

    return FlatMaskSpec(
        module_name=module_name,
        layer_num=layer_num,
        flat_param_numel_local=int(flat_param.numel()),
        flat_param_fqns=flat_fqns,
        ranges=_coalesce_ranges(raw_ranges),
        slices=slice_specs,
    )


def install_fsdp_expert_gradient_mask(
    model: nn.Module,
    frozen_experts_by_layer: Mapping[int, Sequence[int]],
    *,
    num_experts: int = 256,
    log_fn=print,
    log_rank0_only: bool = True,
    max_verbose_steps: int = 1,
) -> Dict[int, FlatMaskSpec]:
    """
    Install per-expert gradient masking on FSDP FULL_SHARD + use_orig_params=True.

    The masking point is the *flat parameter*'s post-accumulate hook. In PyTorch
    2.10, this hook sees the full UNsharded 1-D flat grad before FSDP reduce-
    scatters it. We therefore build mask ranges in that unsharded flat-param
    coordinate system.

    Returns a dict[layer_num -> FlatMaskSpec].
    """
    rank = 0
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()

    def _should_log() -> bool:
        return (not log_rank0_only) or rank == 0

    installed: Dict[int, FlatMaskSpec] = {}
    for module_name, fsdp_module in _iter_fsdp_named_modules(model):
        handle = getattr(fsdp_module, "_handle", None)
        if handle is None:
            continue
        maybe_layer = _parse_layer_num(module_name, tuple(handle.flat_param._fqns))
        if maybe_layer is None or maybe_layer not in frozen_experts_by_layer:
            continue
        spec = _build_mask_spec_for_fsdp_module(
            module_name=module_name,
            fsdp_module=fsdp_module,
            frozen_expert_ids=frozen_experts_by_layer[maybe_layer],
            num_experts=num_experts,
        )
        if spec is None:
            continue

        flat_param = fsdp_module._handle.flat_param

        @torch.no_grad()
        def _hook(param, _spec=spec):
            grad = param.grad
            if grad is None:
                return
            if grad.dim() != 1:
                raise RuntimeError(
                    f"Expected flat grad to be 1D for layer {_spec.layer_num}, got shape={tuple(grad.shape)}"
                )
            if _spec.fired_steps < max_verbose_steps and _should_log():
                before = 0.0
                # Sample the first masked slice for diagnostics.
                s0, e0 = _spec.ranges[0]
                before = float(grad[s0:e0].norm().item())
            for start, end in _spec.ranges:
                grad[start:end].zero_()
            if _spec.fired_steps < max_verbose_steps and _should_log():
                after = float(grad[s0:e0].norm().item())
                log_fn(
                    f"[FSDP mask] layer={_spec.layer_num} module={_spec.module_name!r} "
                    f"flat_grad_shape={tuple(grad.shape)} masked_ranges={len(_spec.ranges)} "
                    f"sample_slice_norm_before={before:.6f} after={after:.6f}"
                )
            _spec.fired_steps += 1

        spec.handle = flat_param.register_post_accumulate_grad_hook(_hook)
        installed[spec.layer_num] = spec

        if _should_log():
            masked_elems = sum(end - start for start, end in spec.ranges)
            log_fn(
                f"[FSDP mask install] layer={spec.layer_num} module={module_name!r} "
                f"local_flat_numel={spec.flat_param_numel_local} fqns={len(spec.flat_param_fqns)} "
                f"masked_element_ranges={len(spec.ranges)} masked_elements={masked_elems}"
            )

    if not installed:
        raise RuntimeError(
            "Did not find any nested FSDP modules matching the requested layer map. "
            "Install this AFTER accelerator.prepare()/FSDP wrapping."
        )
    return installed


def remove_fsdp_expert_gradient_mask(specs: Mapping[int, FlatMaskSpec]) -> None:
    for spec in specs.values():
        if spec.handle is not None:
            spec.handle.remove()
            spec.handle = None


def debug_verify_masked_experts(
    model: nn.Module,
    *,
    layer_to_spec: Mapping[int, FlatMaskSpec],
    full_param_name_resolver=None,
    log_fn=print,
    log_rank0_only: bool = True,
) -> None:
    """
    Expensive diagnostic: unshards params+grads and logs one masked expert norm
    per targeted tensor. Call only occasionally (e.g. first optimizer step).

    `full_param_name_resolver` optionally maps `(layer_num, local_param_fqn)` to an
    actual `named_parameters()` key inside `summon_full_params()`. If omitted, we
    try a suffix match, which works for common nested-FSDP naming patterns.
    """
    rank = 0
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()

    def _should_log() -> bool:
        return (not log_rank0_only) or rank == 0

    param_map: Dict[str, nn.Parameter]
    with FSDP.summon_full_params(model, recurse=True, with_grads=True):
        param_map = dict(model.named_parameters())
        for layer_num, spec in layer_to_spec.items():
            grouped: Dict[str, List[ExpertSliceSpec]] = {}
            for sl in spec.slices:
                grouped.setdefault(sl.param_fqn, []).append(sl)
            for local_fqn, slices in grouped.items():
                chosen = slices[0]
                if full_param_name_resolver is not None:
                    full_name = full_param_name_resolver(layer_num, local_fqn, param_map)
                else:
                    suffixes = [
                        f"layers.{layer_num}.{local_fqn}",
                        f"layers.{layer_num}._fsdp_wrapped_module.{local_fqn}",
                        local_fqn,
                    ]
                    full_name = None
                    for cand in param_map:
                        if any(cand.endswith(sfx) for sfx in suffixes):
                            full_name = cand
                            break
                    if full_name is None:
                        if _should_log():
                            log_fn(
                                f"[FSDP mask verify] could not resolve full param name for layer={layer_num} local_fqn={local_fqn}"
                            )
                        continue
                param = param_map[full_name]
                grad = param.grad
                if grad is None:
                    if _should_log():
                        log_fn(f"[FSDP mask verify] param={full_name} grad=None")
                    continue
                if grad.dim() != 3:
                    if _should_log():
                        log_fn(
                            f"[FSDP mask verify] param={full_name} unexpected grad shape={tuple(grad.shape)}"
                        )
                    continue
                frozen_norm = float(grad[chosen.expert_id].norm().item())
                if _should_log():
                    log_fn(
                        f"[FSDP mask verify] layer={layer_num} param={full_name} "
                        f"masked_expert={chosen.expert_id} grad_norm={frozen_norm:.6f}"
                    )
