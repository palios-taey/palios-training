#!/usr/bin/env python3
"""Clean 4-node DDP-LoRA trainer (CPT + SFT) for Qwen3.6-27B careers/knowledge.

Clean-slate rebuild — NO legacy FSDP trainer, no hand-rolled sharding.
Frozen base fits per node (54GB/128GB) => DDP: base replicated, only tiny
LoRA adapter grads all-reduce/step. No per-layer full-model AllGather =>
structurally avoids the FSDP activation/cmdq wedge class.

Launch (per node, via accelerate launch --machine_rank N ...):
  accelerate launch --config_file ddp.yaml --machine_rank $RANK \
    --num_machines 4 --num_processes 4 --main_process_ip <node0> \
    train_ddp_lora.py --mode cpt --data <cpt.jsonl> --model <qwen3.6-27b> ...

Modes:
  cpt : rows {"text": "..."} -> next-token LM on all tokens.
  sft : rows {"messages":[...]} -> loss masked to assistant tokens only.
frozen_regression rows are never trained (held-out probes).
"""
import os, json, argparse, math, time, re
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HOME", "/home/spark/hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from accelerate.utils import set_seed


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["cpt", "sft"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seq", type=int, default=4096)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--max-steps", type=int, default=0, help=">0 caps steps (wedge-probe / smoke)")
    ap.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    return ap.parse_args()


class LMDataset(Dataset):
    def __init__(self, path, tok, max_seq, mode):
        self.tok, self.max_seq, self.mode = tok, max_seq, mode
        self.rows, skipped = [], 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                if ex.get("meta", {}).get("frozen_regression"):
                    skipped += 1
                    continue
                self.rows.append(ex)
        self.skipped = skipped

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        ex = self.rows[i]
        if self.mode == "cpt":
            ids = self.tok(ex["text"], add_special_tokens=True,
                           truncation=True, max_length=self.max_seq)["input_ids"]
            labels = list(ids)  # train on all tokens
        else:  # sft
            m = ex["messages"]
            ptxt = self.tok.apply_chat_template(m[:-1], add_generation_prompt=True, tokenize=False)
            ftxt = self.tok.apply_chat_template(m, add_generation_prompt=False, tokenize=False)
            pids = self.tok(ptxt, add_special_tokens=False)["input_ids"]
            fids = self.tok(ftxt, add_special_tokens=False)["input_ids"][: self.max_seq]
            plen = len(pids) if fids[:len(pids)] == pids else min(len(pids), len(fids))
            ids = fids
            labels = ([-100] * plen + fids[plen:])[: len(ids)]
        return {"input_ids": ids, "labels": labels}


def collate(batch, pad_id):
    m = max(len(b["input_ids"]) for b in batch)
    ids, lbl, att = [], [], []
    for b in batch:
        n = len(b["input_ids"])
        ids.append(b["input_ids"] + [pad_id] * (m - n))
        lbl.append(b["labels"] + [-100] * (m - n))
        att.append([1] * n + [0] * (m - n))
    return torch.tensor(ids), torch.tensor(lbl), torch.tensor(att)


def main():
    a = parse()
    set_seed(0)
    acc = Accelerator(gradient_accumulation_steps=a.grad_accum)
    is_main = acc.is_main_process
    if is_main:
        os.makedirs(a.out, exist_ok=True)
        print(f"[env] torch {torch.__version__} | world_size={acc.num_processes} | mode={a.mode}", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa", low_cpu_mem_usage=True)
    model.config.use_cache = False
    # use_reentrant=False: reentrant GC + PEFT + DDP can deadlock (double-ready reducer
    # hook drift across ranks under grad-accum). Non-reentrant is the PEFT-recommended default.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lc = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout,
                    target_modules=a.target_modules.split(","), task_type="CAUSAL_LM")
    model = get_peft_model(model, lc)
    if is_main:
        model.print_trainable_parameters()

    ds = LMDataset(a.data, tok, a.max_seq, a.mode)
    if is_main:
        print(f"[data] {len(ds)} train rows ({ds.skipped} frozen held out)", flush=True)
    # drop_last=True: without it, an uneven final batch gives ranks different micro-batch
    # counts under grad-accum → NCCL all-reduce blocks forever (the varying-node hard-hang).
    dl = DataLoader(ds, batch_size=1, shuffle=True, drop_last=True,
                    collate_fn=lambda b: collate(b, tok.pad_token_id))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)

    steps_per_epoch = math.ceil(len(dl) / (a.grad_accum * acc.num_processes))
    total = steps_per_epoch * a.epochs
    sched = get_cosine_schedule_with_warmup(opt, int(total * a.warmup_ratio), total)

    model, opt, dl, sched = acc.prepare(model, opt, dl, sched)
    if is_main:
        print(f"[train] ~{total} optim steps (max_steps={a.max_steps or 'off'})", flush=True)

    model.train()
    gstep, t0 = 0, time.time()
    for ep in range(a.epochs):
        for ids, lbl, att in dl:
            with acc.accumulate(model):
                out = model(input_ids=ids, attention_mask=att, labels=lbl)
                acc.backward(out.loss)
                if acc.sync_gradients:
                    acc.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
            if acc.sync_gradients:
                gstep += 1
                if is_main and gstep % a.log_every == 0:
                    free = torch.cuda.mem_get_info()[0] / 1e9
                    print(f"[step {gstep}/{total}] ep{ep} loss={out.loss.item():.4f} "
                          f"lr={sched.get_last_lr()[0]:.2e} free={free:.1f}GB "
                          f"{(time.time()-t0)/gstep:.1f}s/step", flush=True)
                if a.max_steps and gstep >= a.max_steps:
                    if is_main:
                        print(f"[max-steps {a.max_steps} reached — wedge-probe OK, no hang]", flush=True)
                    acc.wait_for_everyone()
                    if is_main:
                        acc.unwrap_model(model).save_pretrained(os.path.join(a.out, "adapter-probe"))
                    return
    acc.wait_for_everyone()
    if is_main:
        acc.unwrap_model(model).save_pretrained(os.path.join(a.out, "adapter-final"))
        print(f"[done] adapter saved {a.out}/adapter-final in {(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
