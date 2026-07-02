#!/usr/bin/env python3
"""Recall-probe eval for a careers knowledge adapter.

Loads base model + LoRA adapter, generates an answer for each held-out probe
(frozen_regression=true), grades against the deterministic target. Reports
pass rates PER probe file (K1 vs K2) — that's the shared-vs-per-pack answer.
"""
import os, json, argparse, re
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HOME", "/home/spark/hf_cache")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def norm(s):
    return re.sub(r"\s+", " ", s.strip()).lower()


def grade(gen, target):
    g, t = norm(gen), norm(target)
    exact = (g == t) or (t in g)          # exact or target fully contained
    # key-content: for JSON/tuple targets, check the salient tokens are present
    toks = [w for w in re.findall(r"[a-z0-9_\-./]+", t) if len(w) > 3]
    hit = sum(1 for w in toks if w in g)
    contain = (hit / len(toks)) if toks else 0.0
    return exact, contain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--probes", required=True, nargs="+", help="one or more probe jsonl files")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--contain-thresh", type=float, default=0.7)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa", low_cpu_mem_usage=True).to("cuda")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    for pf in args.probes:
        rows = [json.loads(l) for l in open(pf) if l.strip()]
        n = ex_pass = con_pass = 0
        for r in rows:
            if not r.get("meta", {}).get("frozen_regression"):
                continue
            n += 1
            msgs = r["messages"]
            target = msgs[-1]["content"]
            prompt = tok.apply_chat_template(msgs[:-1], add_generation_prompt=True, tokenize=False)
            ids = tok(prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=args.max_new, do_sample=False,
                                     pad_token_id=tok.pad_token_id or tok.eos_token_id)
            gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
            exact, contain = grade(gen, target)
            ex_pass += int(exact)
            con_pass += int(contain >= args.contain_thresh)
        name = os.path.basename(pf)
        print(f"[{name}] probes={n} exact_match={ex_pass}/{n} ({100*ex_pass/max(n,1):.0f}%) "
              f"contain>={args.contain_thresh}={con_pass}/{n} ({100*con_pass/max(n,1):.0f}%)")


if __name__ == "__main__":
    main()
