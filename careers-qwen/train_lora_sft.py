#!/usr/bin/env python3
"""LoRA SFT for Qwen3.6-27B careers offload adapters (knowledge packs + tasks).

Single-node, single-GPU (one Spark). No FSDP, no multi-node => no wedge.
Trains on chat-messages JSONL; loss masked to ASSISTANT tokens only.
Rows with meta.frozen_regression==true are NEVER trained (held-out recall/replay set).

Data format (one JSON object per line):
  {"messages":[{"role":"system","content":...},
               {"role":"user","content":...},
               {"role":"assistant","content":...}],
   "meta":{"task":"k1_history","frozen_regression":false, ...}}
"""
import os, json, argparse, math, time
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HOME", "/home/spark/hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seq", type=int, default=4096)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="cap rows (0=all; for smoke tests)")
    ap.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    ap.add_argument("--resume-adapter", default="", help="path to a CPT adapter dir to CONTINUE (recipe: CPT->SFT->DPO on one growing adapter). Empty = fresh adapter on base.")
    ap.add_argument("--eval-probes", nargs="*", default=[], help="held-out probe jsonl(s) to eval each epoch")
    ap.add_argument("--eval-every-epochs", type=int, default=1)
    ap.add_argument("--eval-max-new", type=int, default=200)
    return ap.parse_args()


import re


def _grade(gen, target, thresh=0.7):
    g, t = re.sub(r"\s+", " ", gen.strip()).lower(), re.sub(r"\s+", " ", target.strip()).lower()
    exact = (g == t) or (t in g)
    toks = [w for w in re.findall(r"[a-z0-9_\-./]+", t) if len(w) > 3]
    contain = (sum(1 for w in toks if w in g) / len(toks)) if toks else 0.0
    return exact, contain >= thresh


@torch.no_grad()
def run_probe_eval(model, tok, probe_files, max_new):
    """Offline generation eval on held-out probes each epoch (no online-loss churn)."""
    model.eval()
    out = []
    for pf in probe_files:
        rows = [json.loads(l) for l in open(pf) if l.strip()]
        rows = [r for r in rows if r.get("meta", {}).get("frozen_regression")]
        n = ex = con = 0
        for r in rows:
            n += 1
            m = r["messages"]
            prompt = tok.apply_chat_template(m[:-1], add_generation_prompt=True, tokenize=False)
            ids = tok(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
            g = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                               pad_token_id=tok.pad_token_id or tok.eos_token_id)
            gen = tok.decode(g[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
            e, c = _grade(gen, m[-1]["content"])
            ex += int(e); con += int(c)
        out.append((os.path.basename(pf), n, ex, con))
    model.train()
    return out


class ChatSFTDataset(Dataset):
    """system+user+assistant single-turn examples; loss on assistant tokens only."""
    def __init__(self, path, tok, max_seq, limit=0):
        self.tok, self.max_seq = tok, max_seq
        self.rows = []
        skipped_frozen = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                if ex.get("meta", {}).get("frozen_regression"):
                    skipped_frozen += 1
                    continue
                if not ex.get("messages"):
                    continue
                self.rows.append(ex["messages"])
        if limit:
            self.rows = self.rows[:limit]
        print(f"[data] {len(self.rows)} train rows loaded ({skipped_frozen} frozen-regression rows held out)")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        msgs = self.rows[i]
        # apply_chat_template return type is unreliable across versions; render TEXT
        # (tokenize=False) then tokenize the string -> plain list[int]. (proven pattern)
        prompt_text = self.tok.apply_chat_template(
            msgs[:-1], add_generation_prompt=True, tokenize=False)
        full_text = self.tok.apply_chat_template(
            msgs, add_generation_prompt=False, tokenize=False)
        prompt_ids = self.tok(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = self.tok(full_text, add_special_tokens=False)["input_ids"]
        # prompt_text is a prefix of full_text (assistant header shared) -> mask prompt
        plen = len(prompt_ids) if full_ids[:len(prompt_ids)] == prompt_ids else min(len(prompt_ids), len(full_ids))
        input_ids = full_ids[: self.max_seq]
        labels = ([-100] * plen + full_ids[plen:])[: len(input_ids)]
        return {"input_ids": input_ids, "labels": labels}


def collate(batch, pad_id):
    m = max(len(b["input_ids"]) for b in batch)
    ids, lbl, att = [], [], []
    for b in batch:
        n = len(b["input_ids"])
        ids.append(b["input_ids"] + [pad_id] * (m - n))
        lbl.append(b["labels"] + [-100] * (m - n))
        att.append([1] * n + [0] * (m - n))
    return (torch.tensor(ids), torch.tensor(lbl), torch.tensor(att))


def main():
    a = parse()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda"
    print(f"[env] torch {torch.__version__} cuda={torch.cuda.is_available()}")

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    print("[model] loading (bf16)...")
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa", low_cpu_mem_usage=True).to(dev)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    if a.resume_adapter:
        # Recipe: CPT -> SFT -> DPO on ONE growing adapter. Continue the CPT adapter
        # (is_trainable=True) rather than starting a fresh LoRA on the base.
        print(f"[model] CONTINUING CPT adapter: {a.resume_adapter}")
        model = PeftModel.from_pretrained(model, a.resume_adapter, is_trainable=True)
    else:
        lc = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout,
                        target_modules=a.target_modules.split(","), task_type="CAUSAL_LM")
        model = get_peft_model(model, lc)
    model.print_trainable_parameters()

    ds = ChatSFTDataset(a.data, tok, a.max_seq, a.limit)
    dl = DataLoader(ds, batch_size=1, shuffle=True,
                    collate_fn=lambda b: collate(b, tok.pad_token_id))

    steps_per_epoch = math.ceil(len(dl) / a.grad_accum)
    total_steps = steps_per_epoch * a.epochs
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    sched = get_cosine_schedule_with_warmup(opt, a.warmup, total_steps)
    print(f"[train] {len(dl)} examples x {a.epochs} epochs = {total_steps} optim steps")

    model.train()
    gstep, t0 = 0, time.time()
    best = {"epoch": -1, "score": -1.0}
    for ep in range(a.epochs):
        opt.zero_grad()
        for i, (ids, lbl, att) in enumerate(dl):
            ids, lbl, att = ids.to(dev), lbl.to(dev), att.to(dev)
            out = model(input_ids=ids, attention_mask=att, labels=lbl)
            (out.loss / a.grad_accum).backward()
            if (i + 1) % a.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); sched.step(); opt.zero_grad()
                gstep += 1
                if gstep % a.log_every == 0:
                    free = torch.cuda.mem_get_info()[0] / 1e9
                    print(f"[step {gstep}/{total_steps}] ep{ep} loss={out.loss.item():.4f} "
                          f"lr={sched.get_last_lr()[0]:.2e} free={free:.1f}GB "
                          f"{(time.time()-t0)/gstep:.1f}s/step", flush=True)
                if gstep % a.save_every == 0:
                    model.save_pretrained(os.path.join(a.out, f"adapter-step{gstep}"))
        # ---- eval-at-intervals: run held-out probes each epoch (no more blind epochs) ----
        if a.eval_probes and (ep + 1) % a.eval_every_epochs == 0:
            res = run_probe_eval(model, tok, a.eval_probes, a.eval_max_new)
            tot_c = tot_n = 0
            for name, n, ex, con in res:
                tot_c += con; tot_n += n
                print(f"[eval ep{ep+1}] {name}: exact={ex}/{n} understand={con}/{n} "
                      f"({100*con/max(n,1):.0f}%)", flush=True)
            score = tot_c / max(tot_n, 1)
            if score > best["score"]:
                best = {"epoch": ep + 1, "score": score}
                model.save_pretrained(os.path.join(a.out, "adapter-best"))
                print(f"[eval ep{ep+1}] NEW BEST understand={100*score:.0f}% -> adapter-best", flush=True)
            else:
                print(f"[eval ep{ep+1}] understand={100*score:.0f}% (best={100*best['score']:.0f}% @ep{best['epoch']}) "
                      f"-- not improving, watch for overfit", flush=True)
    model.save_pretrained(os.path.join(a.out, "adapter-final"))
    if a.eval_probes:
        print(f"[done] BEST checkpoint: adapter-best @ep{best['epoch']} understand={100*best['score']:.0f}% "
              f"(use adapter-best, not adapter-final, if final overfit past best)", flush=True)
    tok.save_pretrained(os.path.join(a.out, "adapter-final"))
    print(f"[done] adapter saved to {a.out}/adapter-final in {(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()
