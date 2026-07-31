#!/usr/bin/env python3
"""build_module_corpus.py — THE builder every module corpus goes through.

WHY THIS EXISTS (2026-07-25, and it is one incident, not a style preference):
Taey made two errors in a single production walk that we had ALREADY written rows for.
Three rows, all lane=practice_corrections, all authored + audited REAL_SITUATION +
lane-stamped, none ever delivered to any bake. Root cause turned out to be two defects:

  1. pack_sft_module1.py hardcodes a four-lane dict (stage2_scorer, jesse_voice,
     repo_capability, values) and iterates ONLY those absolute paths. practice_corrections
     was invisible to it BY CONSTRUCTION — not slipped, never wired.
  2. modules 2 and 3 had NO committed builder at all. Their corpora were assembled ad hoc,
     in-session, by tutor's unrecorded judgement. There was no hidden list to blame; there
     was no tool.

So the correction loop ran capture -> author -> audit -> register -> sanction -> VOID. The
highest-value rows we produce, the ones from real production failures, sat in the one lane
that could not reach training. That is why the model repeats lessons we have written down.

THE INVARIANT THIS ENFORCES:
    REGISTERED IMPLIES ELIGIBLE.
A lane cannot be silently absent, because there is no second list to forget to update.
Admission is DERIVED from build_pairs_manifest.STATUS — the same registry authoring already
writes into. If a lane must be kept out of a given module, that is an EXPLICIT zero in the
mixture, recorded in the emitted manifest, never an omission.

Adding a fifth entry to a hardcoded dict would have moved the bug, not killed it: the sixth
lane someone adds next month vanishes identically, silently, and nobody finds out until the
model repeats a lesson weeks later.

STATUS: written 2026-07-25, UNPROVEN. No bake has run through it yet. It is validated when a
real production bake delivers the practice_corrections lane and the rows appear in the
trained corpus — not before, and not by anyone reading this file.

Usage:
    TRAINING_DATA_ROOT=<governed store> python3 build_module_corpus.py \
        --out <corpus.jsonl> --manifest <corpus_manifest.json> \
        [--weights lane=w,lane=w,...] [--exclude lane,lane]
"""
import argparse, hashlib, importlib.util, json, os, sys, collections

TRAINABLE = ("canonical", "derived")   # statuses whose rows may enter a module corpus


def load_registry():
    """The registry IS the source of truth for what exists. Import it rather than
    re-deriving it, so this builder cannot drift from the manifest the way a second
    hardcoded list would."""
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "pairs_manifest", os.path.join(here, "data", "build_pairs_manifest.py"))
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        # the manifest fails loud when TRAINING_DATA_ROOT is unset; surface that verbatim
        raise
    return mod.STATUS, mod.ROOT


def row_key(row):
    return hashlib.sha256(json.dumps(row.get("messages"), sort_keys=True).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--weights", default="",
                    help="lane=w,lane=w — runtime sampler weights. A lane may be 0, but the "
                         "zero must be WRITTEN so the exclusion is a decision on the record.")
    ap.add_argument("--exclude", default="",
                    help="lane,lane — explicit exclusions. Recorded in the manifest with the "
                         "row count that was left out, so an exclusion is never invisible.")
    a = ap.parse_args()

    STATUS, ROOT = load_registry()
    weights = {}
    for kv in filter(None, (s.strip() for s in a.weights.split(","))):
        k, _, v = kv.partition("=")
        weights[k.strip()] = float(v)
    excluded = {s.strip() for s in a.exclude.split(",") if s.strip()}

    # ---- ADMISSION: derived from the registry, never from a list in this file ----
    by_lane = collections.defaultdict(list)
    unstamped, seen, skipped_status = [], set(), collections.Counter()
    for rel, (status, _desc) in sorted(STATUS.items()):
        if status not in TRAINABLE:
            skipped_status[status] += 1
            continue
        path = os.path.join(ROOT, rel)
        if not os.path.isfile(path):
            continue
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row.get("messages"), list) or len(row["messages"]) < 2:
                continue
            k = row_key(row)
            if k in seen:                       # union merge, no duplication
                continue
            seen.add(k)
            lane = (row.get("meta") or {}).get("lane")
            if not lane:
                unstamped.append(rel)
                continue
            row.setdefault("meta", {})["source_file"] = rel
            by_lane[lane].append(row)

    # ---- FAIL LOUD on any registered lane with no weight decision ----
    # This is the whole point: a lane cannot be quietly absent. Either it has a weight
    # (possibly 0), or it is named in --exclude. Silence is refused.
    undecided = sorted(l for l in by_lane if l not in weights and l not in excluded)
    if undecided:
        print("ABORT: registered trainable lanes with NO weight decision:", file=sys.stderr)
        for l in undecided:
            print(f"  {l}  ({len(by_lane[l])} rows)", file=sys.stderr)
        print("\nEvery registered lane must be a DECISION. Give it a weight (0 is allowed "
              "and is a real answer) or name it in --exclude. A lane that is merely absent "
              "is the defect this builder exists to prevent: three practice_corrections rows "
              "were authored, audited, lane-stamped and never trained, and the model then "
              "repeated both lessons in production.", file=sys.stderr)
        return 2

    if unstamped:
        c = collections.Counter(unstamped)
        print("WARNING: rows with no meta.lane were skipped (cannot be weighted):",
              file=sys.stderr)
        for f, n in c.most_common():
            print(f"  {n:5d}  {f}", file=sys.stderr)

    # ---- EMIT ----
    emitted = []
    for lane, rows in sorted(by_lane.items()):
        if lane in excluded or weights.get(lane, 0.0) == 0.0:
            continue
        emitted.extend(rows)
    with open(a.out, "w") as f:
        for r in emitted:
            f.write(json.dumps(r) + "\n")
    sha = hashlib.sha256(open(a.out, "rb").read()).hexdigest()

    manifest = {
        "builder": "build_module_corpus.py",
        "admission": "derived from build_pairs_manifest.STATUS (registered implies eligible)",
        "out": os.path.abspath(a.out),
        "sha256": sha,
        "rows_emitted": len(emitted),
        "lanes_emitted": {l: len(r) for l, r in sorted(by_lane.items())
                          if l not in excluded and weights.get(l, 0.0) != 0.0},
        "lanes_excluded_explicitly": {l: len(by_lane[l]) for l in sorted(excluded)
                                      if l in by_lane},
        "lanes_zero_weighted": {l: len(by_lane[l]) for l in sorted(by_lane)
                                if weights.get(l, None) == 0.0},
        "weights": weights,
        "rows_skipped_unstamped": len(unstamped),
        "statuses_not_trainable": dict(skipped_status),
    }
    with open(a.manifest, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print(f"  emitted {len(emitted)} rows -> {a.out}")
    print(f"  sha256  {sha}")
    for l, n in sorted(manifest["lanes_emitted"].items()):
        print(f"    {n:6d}  {l}")
    for l, n in manifest["lanes_excluded_explicitly"].items():
        print(f"    {n:6d}  {l}  [EXPLICITLY EXCLUDED]")
    for l, n in manifest["lanes_zero_weighted"].items():
        print(f"    {n:6d}  {l}  [WEIGHT 0 — a decision, on the record]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
