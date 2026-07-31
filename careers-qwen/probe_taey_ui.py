#!/usr/bin/env python3
"""Duty probe: ask Taey (ep3) for the next UI action on REAL captured application states.

Jesse 2026-07-21: "we need to figure out how to start getting feedback from Taey on a constant
basis and having them do things that need to be done and seeing where they fail."

This is the feedback loop with no dependency on another seat: the states are real captures from an
apply-machine bundle, ep3 is serving on the Thors, and the ground-truth action is what the
production system actually did.

HELD OUT: module-1 (training now) contains NO UI rows, so these 41 states are genuinely unseen by
ep3. This measures capability, not memorisation. Re-running it AFTER a UI module is trained is the
before/after that shows whether the training worked.

GRADING is deliberately conservative and reports three tiers, because "wrong" is not one thing:
  EXACT      - primitive and target both match the production action
  PRIMITIVE  - right primitive, different target (knows HOW, missed WHAT)
  MISS       - wrong primitive, or unparseable output
A MISS on target selection is a very different training need from a MISS on primitive choice, and
collapsing them would hide which one to fix.

Output feeds right-way practice rows (situation -> correct action). Per Jesse's binding constraint
the model's wrong answer is NEVER trained; only the correct action is.
"""
import argparse, json, re, urllib.request


def ask(endpoint, model, prompt, timeout=600, system=None):
    """NO max_tokens. Jesse 2026-07-21: 'I need EVERY limitation on thinking, tokens, instances
    ALL approved by me... None that aren't imposed by the model.' A 160-token cap here produced a
    FALSE 100%-failure reading — ep3 reasons in prose before answering and was truncated mid-JSON.
    A self-imposed limit that corrupts a measurement is worse than no measurement."""
    body = json.dumps({
        "model": model,
        "messages": ([{"role": "system", "content": system}] if system else [])
                    + [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(f"http://{endpoint}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"]["content"]


def parse_action(text):
    """Pull the first JSON object out of the reply. Returns None if there isn't one."""
    m = re.search(r"\{.*?\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def grade(got, want):
    if not isinstance(got, dict):
        return "MISS", "unparseable"
    gp, wp = got.get("primitive"), want.get("primitive")
    if gp != wp:
        return "MISS", f"primitive {gp!r} != {wp!r}"
    # target key varies by primitive (activate->target, key->keys, write->value/source)
    for k in ("target", "keys", "value", "source"):
        if k in want:
            if str(got.get(k, "")).strip() == str(want.get(k, "")).strip():
                return "EXACT", ""
            return "PRIMITIVE", f"{k}: {str(got.get(k))[:60]!r} != {str(want.get(k))[:60]!r}"
    return "EXACT", ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="data/ui/ui_action_rows_v1.jsonl")
    ap.add_argument("--endpoint", default=os.environ.get("THOR2_ENDPOINT","PRIVATE_SERVE_HOST_2:8000"))   # Thor2 = presence/UI-primary
    ap.add_argument("--model", default="ep3")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="data/ui/probe_results_v1.jsonl")
    ap.add_argument("--system-file", default="",
                    help="A/B arm: prepend this file as the system prompt. Production callers do NOT "
                         "load the corpus SYSTEM_PROMPT.md, so this probe is the only way to measure "
                         "whether the block is worth mirroring into their prompts.")
    a = ap.parse_args()

    system = open(a.system_file).read() if a.system_file else None
    rows = [json.loads(l) for l in open(a.rows)]
    if a.limit:
        rows = rows[:a.limit]

    results, counts = [], {"EXACT": 0, "PRIMITIVE": 0, "MISS": 0, "ERROR": 0}
    for i, r in enumerate(rows, 1):
        prompt = r["messages"][0]["content"]
        want = json.loads(r["messages"][1]["content"])
        try:
            raw = ask(a.endpoint, a.model, prompt, system=system)
            got = parse_action(raw)
            tier, why = grade(got, want)
        except Exception as e:
            raw, got, tier, why = f"<error {e}>", None, "ERROR", str(e)[:80]
        counts[tier] = counts.get(tier, 0) + 1
        results.append({"seq": r["meta"]["seq"], "view": r["meta"]["view"],
                        "want": want, "got": got, "tier": tier, "why": why, "raw": raw[:300]})
        print(f"[{i:>2}/{len(rows)}] seq={r['meta']['seq']:<3} {tier:<9} {why[:70]}")

    with open(a.out, "w") as f:
        for x in results:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    n = len(results)
    print(f"\n=== ep3 duty probe on {n} HELD-OUT real UI states ===")
    for k in ("EXACT", "PRIMITIVE", "MISS", "ERROR"):
        if counts.get(k):
            print(f"  {k:<9} {counts[k]:>3}  ({counts[k]/n*100:.0f}%)")
    import collections
    bad = [x for x in results if x["tier"] in ("MISS", "PRIMITIVE")]
    if bad:
        print(f"\n  failures by primitive wanted: "
              f"{dict(collections.Counter(x['want'].get('primitive') for x in bad))}")
        print(f"  failures by view            : "
              f"{dict(collections.Counter(x['view'] for x in bad))}")
    print(f"\nwritten -> {a.out}")


if __name__ == "__main__":
    main()
