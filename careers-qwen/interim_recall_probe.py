#!/usr/bin/env python3
"""Interim per-epoch recall probe — provenance-clean, deterministic, no judge.

Answers "does the CPT'd model recall the corpus content better than base" without waiting for the
full retention-battery design (Chats consult in flight). Draws probes STRAIGHT FROM the registered
corpus slices (each probe traces to a corpus row + slice sha), so no hand-authored/fabricated probes.

Method (per slice-type, deterministic):
  - Sample N rows per slice (stable hash → reproducible; these are the HELD-OUT probe rows).
  - For each: prompt = first PREFIX_CHARS of the doc; expected = salient tokens from the REST.
  - Generate greedily from the served model (trained epoch-N AND untouched base).
  - Score = key-fact containment: fraction of expected salient tokens present in the generation.
  - Report trained_containment vs base_containment PER SLICE, delta, base-vs-base is the σ (run base twice).

This is the INTERIM gate (Jesse: test after every epoch). The Chats' battery (voice-style metric etc.)
upgrades it. Runs against an OpenAI-compatible vLLM endpoint (infra serves epoch-HF + base on Thor).

Usage:
  python3 interim_recall_probe.py --endpoint http://PRIVATE_SERVE_HOST_1:8000/v1 --model <served-name> \
      --slices-dir <dir with the 8 corpus slices> --n-per-slice 20 --out probe_result.json
"""
import argparse, hashlib, json, os, re, sys, urllib.request

# --hf-model mode (2026-07-17): generate DIRECTLY via transformers on the node holding the
# converted HF model — greedy, deterministic, no server. Added when the Thor serve lane
# stalled 17h; the probe must never again be blocked on a serving dependency.
_HF = {"model": None, "tok": None}

def hf_generate(model_dir, prompt, max_tokens=200, adapter=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if _HF["model"] is None:
        _HF["tok"] = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        m = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
        if adapter:
            # GATE-0 base-preservation: attach the LoRA adapter to the frozen base (hot), same
            # serving path. Base weights untouched by construction; this measures the base+module.
            from peft import PeftModel
            m = PeftModel.from_pretrained(m, adapter)
            print(f"[hf] adapter attached: {adapter}", flush=True)
        _HF["model"] = m
        _HF["model"].eval()
    tok = _HF["tok"]
    ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
    with __import__("torch").no_grad():
        out = _HF["model"].generate(ids, max_new_tokens=max_tokens, do_sample=False,
                                    temperature=None, top_p=None, top_k=None,
                                    pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def salient(text, k=12):
    # salient tokens = longer alnum tokens (skip stopwords-ish short ones), dedup, first k
    toks = [w.lower() for w in re.findall(r"[A-Za-z0-9_./-]{4,}", text)]
    seen, out = set(), []
    for w in toks:
        if w not in seen:
            seen.add(w); out.append(w)
        if len(out) >= k:
            break
    return out


def stable_sample(rows, n):
    # deterministic hash-order sample (reproducible held-out probe set)
    idx = sorted(range(len(rows)), key=lambda i: hashlib.sha256(str(i).encode()).hexdigest())
    return [rows[i] for i in idx[:min(n, len(rows))]]


def generate(endpoint, model, prompt, max_tokens=200):
    body = json.dumps({"model": model, "prompt": prompt, "max_tokens": max_tokens,
                       "temperature": 0.0, "stop": None}).encode()
    req = urllib.request.Request(endpoint.rstrip("/") + "/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read())
    return d["choices"][0]["text"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default=None, help="OpenAI-compat server (server mode)")
    ap.add_argument("--hf-model", default=None, help="local HF model dir (direct transformers mode)")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir to attach to --hf-model (GATE-0 base+module)")
    ap.add_argument("--model", required=True, help="model name (server mode) or label (hf mode)")
    ap.add_argument("--slices-dir", required=True)
    ap.add_argument("--n-per-slice", type=int, default=20)
    ap.add_argument("--prefix-chars", type=int, default=400)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not args.endpoint and not args.hf_model:
        sys.exit("ABORT: need --endpoint or --hf-model")

    slices = [f for f in os.listdir(args.slices_dir) if f.endswith(".jsonl")]
    results = {}
    for sl in sorted(slices):
        rows = [json.loads(l) for l in open(os.path.join(args.slices_dir, sl))]
        rows = [r for r in rows if len(r.get("text", "")) > args.prefix_chars + 200]
        probes = stable_sample(rows, args.n_per_slice)
        hits, total = 0.0, 0
        for r in probes:
            text = r["text"]
            prompt = text[:args.prefix_chars]
            exp = salient(text[args.prefix_chars:])
            if not exp:
                continue
            gen = (hf_generate(args.hf_model, prompt, adapter=args.adapter) if args.hf_model
                   else generate(args.endpoint, args.model, prompt)).lower()
            hit = sum(1 for w in exp if w in gen) / len(exp)
            hits += hit; total += 1
        results[sl] = {"n": total, "mean_containment": round(hits / total, 4) if total else None}
        print(f"[probe] {sl}: n={total} mean_containment={results[sl]['mean_containment']}", flush=True)
    json.dump({"model": args.model, "endpoint": args.endpoint or f"hf:{args.hf_model}",
               "per_slice": results}, open(args.out, "w"), indent=2)
    print(f"[probe] DONE → {args.out}", flush=True)


if __name__ == "__main__":
    main()
