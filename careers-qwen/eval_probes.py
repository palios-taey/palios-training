#!/usr/bin/env python3
"""Recall-probe eval for a careers knowledge adapter.

Loads base model + LoRA adapter, generates an answer for each held-out probe
(frozen_regression=true), grades against the deterministic target. Reports
pass rates PER probe file (K1 vs K2) — that's the shared-vs-per-pack answer.
"""
import os, json, argparse, re
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/hf_cache"))
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
    ap.add_argument("--dump-gens", default="", help="write {id,prompt,gen,target} JSONL for semantic re-grade")
    ap.add_argument("--no-think", action="store_true", help="disable reasoning (some templates hang on this)")
    args = ap.parse_args()
    dump = open(args.dump_gens, "w") if args.dump_gens else None

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # device_map={"":0} loads DIRECT to GPU. Do NOT use low_cpu_mem_usage=True + .to("cuda"):
    # that keeps a CPU copy AND a GPU copy (~2x = ~108GB for the 27B) and OS-OOM-kills mid-load
    # with no traceback. The serve shim uses device_map and loads the same model+adapter fine.
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa", low_cpu_mem_usage=True, device_map={"": 0})
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
            tkw = {"enable_thinking": False} if args.no_think else {}
            try:
                prompt = tok.apply_chat_template(msgs[:-1], add_generation_prompt=True, tokenize=False, **tkw)
            except TypeError:
                prompt = tok.apply_chat_template(msgs[:-1], add_generation_prompt=True, tokenize=False)
            ids = tok(prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=args.max_new, do_sample=False,
                                     pad_token_id=tok.pad_token_id or tok.eos_token_id)
            # reasoning-model aware: decode WITH specials so </think> survives, then take
            # the ANSWER after the reasoning block (grading the <think> preamble = false 0%).
            raw = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=False)
            gen = raw.split("</think>")[-1] if "</think>" in raw else raw
            for t in (tok.eos_token or "", "<|im_end|>", "<|endoftext|>", "<think>"):
                if t:
                    gen = gen.replace(t, "")
            gen = gen.strip()
            exact, contain = grade(gen, target)
            ex_pass += int(exact)
            con_pass += int(contain >= args.contain_thresh)
            if dump:
                dump.write(json.dumps({"id": r.get("meta", {}).get("example_id"),
                                       "user": msgs[-2]["content"], "gen": gen, "target": target}) + "\n")
                dump.flush()
        name = os.path.basename(pf)
        print(f"[{name}] probes={n} exact_match={ex_pass}/{n} ({100*ex_pass/max(n,1):.0f}%) "
              f"contain>={args.contain_thresh}={con_pass}/{n} ({100*con_pass/max(n,1):.0f}%)")


if __name__ == "__main__":
    main()
