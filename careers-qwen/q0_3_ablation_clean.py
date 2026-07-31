#!/usr/bin/env python3
"""q0-3 mechanism ablation (C3): does a factual capability belong in retrieval or
weights? Runs Claude's R2 grid conditions A/B/C against an OpenAI-compatible base
endpoint (+ ISMA for A) on the held-out exact-recall probes. Condition D (small
SFT probe) is a separate training run.

Conditions per probe (each probe = a system+user question with a gold assistant answer):
  C  base alone         : ask the base model, no context. Floor.
  A  base + RAG         : query ISMA with the question, prepend retrieved context. Retrieval path.
  B  in-context ceiling : prepend the GOLD answer as context. Ceiling ("can it USE the fact when present").

Grading is deterministic (exact-match + containment) — appropriate for exact-recall
probes; the exact-vs-contain gap is the "was the pilot's exact-match grade an
artifact" signal (condition C's semantic-regrade question).

Usage:
  python3 q0_3_ablation.py --base-url http://HOST:8000/v1 --model auto \
      --probes k1_career_history_probes.jsonl k2_repo_capabilities_probes.jsonl \
      --isma-url http://localhost:8095 --out q0-3_results.json
"""
import argparse, json, re, sys, urllib.request
import time

def http_json(method, url, payload, timeout=180):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def resolve_model(base_url, model):
    if model != "auto":
        return model
    m = http_json("GET", f"{base_url.rstrip('/')}/models", None, 30)
    return m["data"][0]["id"]

def chat(base_url, model, messages, max_tokens=1024):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    r = http_json("POST", f"{base_url.rstrip('/')}/chat/completions", body)
    return r["choices"][0]["message"]["content"]

def isma_context(isma_url, query, top_k=5):
    # Retry on empty: ISMA's embed model torch.compile-recompiles per new sequence-length;
    # the FIRST /search of a new length can time-out to 0 tiles while recompiling (weaver 2026-07-04).
    # Retry (shape now compiled) so condition A never gets spurious empty context.
    last = ""
    for attempt in range(3):
        try:
            r = http_json("POST", f"{isma_url.rstrip('/')}/search",
                          {"query": query, "top_k": top_k}, 60)
            tiles = r.get("tiles", [])
            if tiles:
                return "\n\n".join(t.get("content", "")[:800] for t in tiles[:top_k])
            last = "[ISMA returned 0 tiles]"
        except Exception as e:
            last = f"[ISMA retrieval error: {e}]"
        time.sleep(3)  # let the recompile finish
    return last

def _norm(s):
    return re.sub(r"\s+", " ", s.strip().lower())

def grade(answer, gold):
    a, g = _norm(answer), _norm(gold)
    exact = g in a or a in g
    gt = set(re.findall(r"[a-z0-9]+", g))
    at = set(re.findall(r"[a-z0-9]+", a))
    contain = (len(gt & at) / max(1, len(gt))) if gt else 0.0
    return exact, contain

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default="auto")
    ap.add_argument("--probes", nargs="+", required=True)
    ap.add_argument("--isma-url", default="http://localhost:8095")
    ap.add_argument("--conditions", default="C,A,B", help="subset of C,A,B to run")
    ap.add_argument("--contain-thresh", type=float, default=0.7)
    ap.add_argument("--out", default="q0-3_results.json")
    a = ap.parse_args()

    model = resolve_model(a.base_url, a.model)
    conds = [c.strip() for c in a.conditions.split(",") if c.strip()]
    print(f"[q0-3] model={model} conditions={conds}", flush=True)

    probes = []
    for pf in a.probes:
        with open(pf) as f:
            for line in f:
                line = line.strip()
                if line:
                    probes.append(json.loads(line))
    print(f"[q0-3] {len(probes)} probes", flush=True)

    rows, agg = [], {c: {"exact": 0, "contain": 0} for c in conds}
    for i, ex in enumerate(probes):
        m = ex["messages"]
        sysmsg = next((x for x in m if x["role"] == "system"), None)
        user = next(x for x in m if x["role"] == "user")
        gold = next(x for x in m if x["role"] == "assistant")["content"]
        base_msgs = ([sysmsg] if sysmsg else []) + [user]
        rec = {"id": ex.get("meta", {}).get("example_id", i), "gold": gold}
        for c in conds:
            if c == "C":
                msgs = base_msgs
            elif c == "A":
                ctx = isma_context(a.isma_url, user["content"])
                msgs = ([sysmsg] if sysmsg else []) + [
                    {"role": "user", "content": f"Context from retrieval:\n{ctx}\n\n{user['content']}"}]
            elif c == "B":
                msgs = ([sysmsg] if sysmsg else []) + [
                    {"role": "user", "content": f"Reference (authoritative):\n{gold}\n\n{user['content']}"}]
            else:
                continue
            try:
                ans = chat(a.base_url, model, msgs)
            except Exception as e:
                ans = f"[gen error: {e}]"
            ex_ok, cont = grade(ans, gold)
            rec[c] = {"exact": ex_ok, "contain": round(cont, 2), "answer": ans}
            agg[c]["exact"] += int(ex_ok)
            agg[c]["contain"] += int(cont >= a.contain_thresh)
        rows.append(rec)
        print(f"  [{i+1}/{len(probes)}] {rec['id']}: " +
              " ".join(f"{c}={'E' if rec[c]['exact'] else ('c' if rec[c]['contain']>=a.contain_thresh else '.')}"
                       for c in conds if c in rec), flush=True)

    n = len(probes)
    summary = {c: {"exact_pct": round(100*agg[c]["exact"]/max(1, n), 1),
                   "contain_pct": round(100*agg[c]["contain"]/max(1, n), 1)} for c in conds}
    out = {"model": model, "n": n, "conditions": conds, "summary": summary, "rows": rows}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print("\n[q0-3 SUMMARY]", flush=True)
    for c in conds:
        print(f"  {c}: exact={summary[c]['exact_pct']}%  contain>={a.contain_thresh}={summary[c]['contain_pct']}%", flush=True)
    print(f"[q0-3] wrote {a.out}", flush=True)

if __name__ == "__main__":
    main()
