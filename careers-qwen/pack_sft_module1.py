#!/usr/bin/env python3
"""Pack the FIRST SFT module corpus from the D-verdicted, treasurer-registered PASS lanes.

Deterministic union merge — NO row duplication (mixture is a runtime sampler weight per the Chats'
recipe, not baked in). Every row keeps its source meta + gets meta.lane stamped, so provenance is
preserved and the file traces entirely to treasurer-sanctioned lanes (never-again: tutor packs
sanctioned lanes, does not fabricate/select data). Held-out probes are separate files by design
(D-verdict) and are NOT included here; any row already carrying meta.frozen_regression==true is
dropped as a safety net.

Usage: pack_sft_module1.py --out <train.jsonl> [--manifest <manifest.json>]
"""
import argparse, json, hashlib, os

# The 4 D-verdict PASS lanes (SFT_D_VERDICT_v1.md), treasurer-registered.
PAIRS = os.path.join(os.environ.get("TREASURER_ROOT", os.path.expanduser("~/treasurer")), "foundations/careers/training_data/v2/pairs")
LANES = {
    "stage2_scorer":   f"{PAIRS}/stage2_verdict_trails_v4.jsonl",
    "jesse_voice":     f"{PAIRS}/qwen_verbatim_voice_sft_pairs.jsonl",
    "repo_capability": f"{PAIRS}/b1_repos_laneA.jsonl",
    "values":          f"{PAIRS}/k3_values_train.jsonl",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default="")
    a = ap.parse_args()

    total, per_lane, dropped_frozen, dropped_nomsg = 0, {}, 0, 0
    # DEDUP-AT-PACK (QWEN_FACTORY law: every catch becomes a mechanical rule; catch = module-1 trained
    # 94 exact-dup stage2 rows despite SFT_D_VERDICT_v1 "dedupe at pack"). Rule: an exact-duplicate
    # messages[] payload is DROPPED unless the row is explicitly marked as intentional oversampling
    # (meta.oversampled truthy) — marked oversamples are KEPT and counted separately for the D-verdict
    # to adjudicate. Never silently dedupe a deliberate multiplier; never silently train an accident.
    import hashlib as _hl
    _seen, dropped_dup, kept_oversampled = set(), 0, 0
    with open(a.out, "w") as w:
        for lane, path in LANES.items():
            if not os.path.exists(path):
                raise SystemExit(f"ABORT: lane file missing: {path}")
            n = 0
            for line in open(path):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if "messages" not in r or not r["messages"]:
                    dropped_nomsg += 1
                    continue
                meta = r.setdefault("meta", {})
                if meta.get("frozen_regression") is True:   # safety net; probes live in separate files
                    dropped_frozen += 1
                    continue
                _h = _hl.sha256(json.dumps(r["messages"], ensure_ascii=False,
                                           sort_keys=True).encode()).hexdigest()
                if _h in _seen:
                    if meta.get("oversampled"):
                        kept_oversampled += 1               # deliberate multiplier — keep, report
                    else:
                        dropped_dup += 1                    # unmarked exact-dup — accident, drop
                        continue
                _seen.add(_h)
                meta["lane"] = lane                          # stamp lane for the runtime weighted sampler
                w.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
            per_lane[lane] = n
            total += n

    sha = hashlib.sha256(open(a.out, "rb").read()).hexdigest()
    manifest = {
        "out": a.out, "sha256": sha, "total_rows": total, "per_lane": per_lane,
        "dropped_frozen_regression": dropped_frozen, "dropped_no_messages": dropped_nomsg,
        "dropped_exact_dup_unmarked": dropped_dup, "kept_oversampled_dups": kept_oversampled,
        "source_lanes": LANES,
        "note": "union merge + mechanical dedup (unmarked exact-dups dropped; meta.oversampled kept+reported); mixture = runtime sampler weight",
    }
    print(json.dumps(manifest, indent=2))
    if a.manifest:
        json.dump(manifest, open(a.manifest, "w"), indent=2)

if __name__ == "__main__":
    main()
