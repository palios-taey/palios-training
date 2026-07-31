#!/usr/bin/env python3
"""merge_comprehensive_corpus.py — merge all corpus sources into ONE comprehensive CPT corpus,
content-hash DEDUP so overlapping docs aren't accidentally double-weighted.

The 2026-07-11 comprehensive base: repo source + careers + identity/constitution + G1-G3 + voice +
background — everything, so the model KNOWS it all (validate-not-search). Treasurer's merge rule:
the repo-code .md files overlap with the careers prose corpus; dedup by normalized-content hash
(keep first occurrence). Deliberate OVERSAMPLING is a per-source knob (repeat a source N×); accidental
duplication is a bug the dedup removes.

Each source is a jsonl with a "text" field (+ optional metadata). Output: merged.jsonl ({text,...})
+ a printed coverage report (per-source kept/dropped/tokens) for the CORPUS_OBJECTIVES_CONTRACT.

Usage: merge_comprehensive_corpus.py <out.jsonl> <src1.jsonl>[:oversample] [<src2.jsonl>[:N] ...]
  e.g. merge_comprehensive_corpus.py out.jsonl repo_code.jsonl:1 careers.jsonl:2 identity.jsonl:2
"""
import sys, os, json, hashlib, re

_ws = re.compile(r"\s+")

def norm_hash(text: str) -> str:
    # normalize whitespace so trivially-reformatted dups collapse; hash the content body
    return hashlib.sha1(_ws.sub(" ", text.strip()).encode("utf-8", "replace")).hexdigest()

def main():
    out_path = sys.argv[1]
    specs = sys.argv[2:]
    if not specs:
        raise SystemExit("need at least one source jsonl")
    seen = set()
    kept = dropped = 0
    report = []
    with open(out_path, "w") as fout:
        for spec in specs:
            path, _, ov = spec.partition(":")
            oversample = int(ov) if ov else 1
            if not os.path.isfile(path):
                report.append((path, "MISSING", 0, 0, 0)); continue
            s_kept = s_drop = s_tok = 0
            for line in open(path):
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = d.get("text") or d.get("content") or ""
                if not text.strip():
                    continue
                h = norm_hash(text)
                if h in seen:
                    s_drop += 1; dropped += 1; continue
                seen.add(h)
                for _ in range(oversample):
                    fout.write(json.dumps(d) + "\n")
                s_kept += 1; kept += 1; s_tok += (len(text) // 4) * oversample
            report.append((os.path.basename(path), f"x{oversample}", s_kept, s_drop, s_tok))
    print(f"MERGED → {out_path}")
    print(f"{'source':45} {'ov':4} {'kept':>7} {'dropdup':>8} {'~tok':>12}")
    for name, ov, k, dr, tok in report:
        print(f"  {name:43} {ov:4} {k:7} {dr:8} {tok:12,}")
    tot_tok = sum(r[4] for r in report)
    print(f"TOTAL kept={kept} dropped_dup={dropped} ~tokens={tot_tok:,}")

if __name__ == "__main__":
    main()
