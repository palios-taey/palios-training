#!/usr/bin/env python3
"""audit_row_provenance.py — PER-ROW provenance audit + lineage ledger for an SFT corpus.

Why this exists: on 2026-07-24 tutor built a "captured-only" corpus by applying three
lane/flag predicates and called the result "177 captured rows". tutor-codex vetoed it —
correctly — because lane is a HEURISTIC, not evidence, and no per-row lineage existed.
This script replaces that inline one-liner with a tracked, reproducible, evidence-emitting
audit so a corpus claim can be checked rather than believed.

THE CLASSIFICATION AXIS (tutor proposal 2026-07-24, pending tutor-codex ruling):
  classify on whether the SITUATION is real, NOT whether the text is authored.
  An SFT target must be a correct answer, so every target is authored or corrected —
  "captured vs authored" is an incoherent axis for SFT and produced the bad claim.

  REAL_SITUATION    the prompt/context is something that actually happened in production
                    (a real walk, a real Jesse correction on a real artifact, a real file
                    retrieved) and the response is the authored-correct handling of it.
  SYNTH_SITUATION   the prompt itself was manufactured from a seed; the situation never
                    occurred. Model-generated pairs land here.
  UNPROVEN          provenance fields are insufficient to decide. NOT a pass — these are
                    excluded until trace evidence is produced. Silence is not evidence.

Usage:
  python3 audit_row_provenance.py --in <corpus.jsonl> --ledger <ledger.jsonl> [--emit <filtered.jsonl>]
Exit 0 always (an audit reports; it does not decide). --emit writes REAL_SITUATION rows only.
"""
import argparse, json, hashlib, sys, collections

REAL, SYNTH, UNPROVEN = "REAL_SITUATION", "SYNTH_SITUATION", "UNPROVEN"


def classify(meta):
    """Return (class, rule_id, evidence) — evidence is the literal field(s) that decided it."""
    lane = meta.get("lane", "?")

    # 1. explicit model authorship of the SITUATION → synthetic, no appeal
    if meta.get("generator"):
        return SYNTH, "R1-generator", {"generator": meta["generator"]}

    # 2. self-declared synthesized probe: sources may be real commits, but the QUESTION
    #    was manufactured, so the situation never occurred as posed
    if meta.get("probe_is_synthesized") is True:
        return SYNTH, "R2-synth-probe", {"probe_is_synthesized": True,
                                         "source": str(meta.get("source"))[:80]}

    # 3. authored reasoning with self-declared absence of ground truth
    if "no-canonical-source" in str(meta.get("kind", "")):
        return SYNTH, "R3-no-canonical-source", {"kind": meta.get("kind")}

    # 4. explicitly captured from a production surface
    if str(meta.get("origin", "")).lower() == "captured":
        return REAL, "R4-origin-captured", {"origin": meta["origin"],
                                            "view": meta.get("view"), "seq": meta.get("seq")}

    # 5. derived from a real correction: derived_from must carry a concrete seed hash.
    #    A null derived_from means the row IS a seed — which is only real if it also
    #    carries a recurrence/class stamp tying it to an observed event.
    if "derived_from" in meta:
        df = meta.get("derived_from")
        if df:
            return REAL, "R5-derived-from-seed", {"derived_from": df,
                                                  "class": meta.get("class"),
                                                  "recurrence": meta.get("recurrence")}
        if meta.get("class") and meta.get("recurrence") is not None:
            return REAL, "R5b-seed-with-event-stamp", {"class": meta.get("class"),
                                                       "recurrence": meta.get("recurrence"),
                                                       "teaches": str(meta.get("teaches"))[:60]}
        return UNPROVEN, "R5c-seed-no-stamp", {"derived_from": None, "class": meta.get("class")}

    # 6. mechanical/tool-call traces tied to a captured action
    k = str(meta.get("kind", ""))
    if k.startswith("no-think") or "tool-call" in k:
        return REAL, "R6-mechanical-trace", {"kind": k, "primitive": meta.get("primitive"),
                                             "seq": meta.get("seq")}

    # 7. source_file alone proves the FILE is real, not that the retrieval happened.
    #    Explicitly UNPROVEN — this is the class tutor wrongly counted as captured.
    if meta.get("source_file"):
        return UNPROVEN, "R7-source-file-only", {"source_file": meta.get("source_file"),
                                                 "note": "real file != real retrieval event; "
                                                         "needs a production trace to promote"}

    return UNPROVEN, "R8-no-provenance-fields", {"meta_keys": sorted(meta.keys())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--emit", default="")
    a = ap.parse_args()

    counts = collections.Counter()
    by_rule = collections.Counter()
    lane_split = collections.defaultdict(collections.Counter)
    kept = []

    with open(a.ledger, "w") as led:
        for i, line in enumerate(open(a.inp)):
            r = json.loads(line)
            meta = r.get("meta", {})
            cls, rule, ev = classify(meta)
            row_sha = hashlib.sha256(json.dumps(r.get("messages", []), ensure_ascii=False,
                                                sort_keys=True).encode()).hexdigest()[:16]
            led.write(json.dumps({"idx": i, "row_sha16": row_sha, "lane": meta.get("lane"),
                                  "class": cls, "rule": rule, "evidence": ev}) + "\n")
            counts[cls] += 1
            by_rule[(cls, rule)] += 1
            lane_split[meta.get("lane")][cls] += 1
            if cls == REAL:
                kept.append(line)

    print(f"AUDITED {sum(counts.values())} rows from {a.inp}")
    for c in (REAL, SYNTH, UNPROVEN):
        print(f"  {c:16s} {counts[c]:4d}")
    print("\nBY RULE:")
    for (c, rule), n in sorted(by_rule.items()):
        print(f"  {c:16s} {rule:26s} {n:4d}")
    print("\nBY LANE:")
    for lane, cc in sorted(lane_split.items(), key=lambda kv: str(kv[0])):
        print(f"  {str(lane):22s} " + "  ".join(f"{k}={v}" for k, v in sorted(cc.items())))
    print(f"\nledger -> {a.ledger}")

    if a.emit:
        open(a.emit, "w").writelines(kept)
        sha = hashlib.sha256(open(a.emit, "rb").read()).hexdigest()
        print(f"emitted {len(kept)} REAL_SITUATION rows -> {a.emit}  sha256={sha[:16]}")
        print("NOTE: UNPROVEN rows are EXCLUDED. Promote them only with a production trace.")


if __name__ == "__main__":
    main()
