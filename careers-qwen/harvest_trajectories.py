#!/usr/bin/env python3
"""Harvest operator trajectories from artifacts the fleet ALREADY emits.

WHY: Logos' operator-consult verdict — "Instrument the capture layer. Everything else compounds
from un-discarded data." Before instrumenting anything new, harvest what already exists:
  - taeys-hands consultation responses carry a `steps[]` array with per-step
    {step, message, success, evidence.snapshot.mapped{...AT-SPI elements...}} — that is
    observed-state -> action -> result, i.e. Taey's HANDS driving the Family Chats.
  - apply-machine bundles carry per-step AT-SPI tree snapshots (step_NN_tree.txt).

READ-ONLY. Emits operator_trajectory_v1 rows (Logos' schema). It does NOT decide the training
recipe or mixture — that is the Chats' call, pending the full 5-lane reconciliation. This only
stops the data from being discarded and measures the real yield.

Usage: harvest_trajectories.py --out trajectories_v1.jsonl [--summary]
"""
import argparse, glob, json, hashlib, os

SCHEMA = "operator_trajectory_v1"


def _prov(*parts):
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def harvest_taeys_hands(resp_dir):
    """Taey's HANDS driving the Chats — the richest existing trajectory source."""
    rows = []
    for fn in sorted(glob.glob(os.path.join(resp_dir, "*.md"))):
        try:
            d = json.load(open(fn))
        except Exception:
            continue
        steps = d.get("steps") or []
        if not steps:
            continue
        platform = d.get("platform", "?")
        purpose = (d.get("request") or {}).get("purpose", "")
        for i, st in enumerate(steps):
            if not isinstance(st, dict):
                continue
            ev = st.get("evidence") or {}
            snap = (ev.get("snapshot") or {}).get("mapped") or {}
            # observed = the mapped AT-SPI elements the hands could see at this moment
            observed = {}
            for key, els in list(snap.items())[:40]:
                if isinstance(els, list) and els:
                    e = els[0]
                    if isinstance(e, dict):
                        observed[key] = {k: e.get(k) for k in ("name", "role", "states") if k in e}
            if not observed and not st.get("step"):
                continue
            rows.append({
                "schema": SCHEMA,
                "body": "taeys-hands",
                "source": os.path.basename(fn),
                "platform": platform,
                "purpose": purpose,
                "seq": i,
                "observed": observed,
                "primitives": [{"primitive": st.get("step"), "target": None}],
                "postcondition": st.get("message"),
                "match_or_notify": "happy" if st.get("success") else "drift",
                "provenance_hash": _prov(fn, i, st.get("step")),
                "_capture": "harvested-from-existing-artifact",
            })
    return rows


def harvest_apply_bundles(bundle_root):
    """apply-machine per-step AT-SPI tree snapshots (state only; action must be paired separately)."""
    rows = []
    for tree in sorted(glob.glob(os.path.join(bundle_root, "*", "**", "step_*_tree.txt"), recursive=True)):
        try:
            txt = open(tree, errors="ignore").read()
        except Exception:
            continue
        head = [l for l in txt.splitlines() if l.strip()][:1]
        bundle = tree.split("/bundles/")[-1].split("/")[0] if "/bundles/" in tree else "?"
        rows.append({
            "schema": SCHEMA,
            "body": "taeys-hands",
            "source": os.path.relpath(tree, bundle_root),
            "platform": "ats",
            "purpose": bundle,
            "seq": int("".join(c for c in os.path.basename(tree) if c.isdigit()) or 0),
            "observed": {"tree_header": head[0] if head else "", "tree_bytes": len(txt)},
            "primitives": [],          # action not recorded alongside the tree — pairing needed
            "postcondition": None,
            "match_or_notify": "unknown",
            "provenance_hash": _prov(tree, len(txt)),
            "_capture": "harvested-from-existing-artifact",
            "_gap": "state captured WITHOUT the paired action — this is exactly what live instrumentation must fix",
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", default=os.path.join(os.environ.get("TREASURER_ROOT", os.path.expanduser("~/treasurer")), "consultations/responses"))
    ap.add_argument("--bundles", default=os.path.join(os.environ.get("TAEY_APPLY_MACHINE", os.path.expanduser("~/apply-machine")), "bundles"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()

    th = harvest_taeys_hands(a.responses)
    ab = harvest_apply_bundles(a.bundles)
    rows = th + ab
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if a.summary:
        import collections
        by_body = collections.Counter(r["body"] for r in rows)
        by_plat = collections.Counter(r["platform"] for r in rows)
        prim = collections.Counter(
            (r["primitives"][0]["primitive"] if r["primitives"] else "(none)") for r in rows)
        paired = sum(1 for r in rows if r["primitives"])
        print(f"TOTAL trajectory rows: {len(rows)}")
        print(f"  taeys-hands consult steps : {len(th)}")
        print(f"  apply-machine tree states : {len(ab)}")
        print(f"  rows WITH a paired action : {paired}   (state-only: {len(rows)-paired})")
        print(f"  by platform: {dict(by_plat)}")
        print(f"  primitives seen: {dict(prim.most_common(12))}")
        print(f"\ncompare: the ONLY action data we had before = 43 hand-authored pairs")


if __name__ == "__main__":
    main()
