#!/usr/bin/env python3
"""measure_cpt_delta.py — weight-diff two HF checkpoints and report BOTH units.

WHY BOTH UNITS, AND WHY THIS SCRIPT EXISTS
------------------------------------------
On 2026-07-27 I quoted a CPT result as "0.38x ULP" for most of an evening. The number was really
0.383 PERCENT relative weight change. Those are different quantities and the confusion is not
cosmetic: "0.38x ULP" reads as a near-null run against a 0.5u floor, while 0.383% relative reads as
a small-but-real update — and a peer was planning a pipeline around the first reading.

The same confusion has a second face. Our documented CPT bake-diff PASS band
(RECIPE_v2_synthesis.md:72) is an ABSOLUTE statistic — decoder mlp mean|dW| between 5e-5 and 8e-4
with 30-80% of elements changed. The 2.176% / 0.291% reference points are RELATIVE (|dW| / |w|).
An absolute band CANNOT be converted into a relative-percentage target by arithmetic, because the
conversion depends on |w|, which differs per tensor. Reporting one while implying the other endorses
it is how a gate gets cited for a claim it never made.

So this prints all three, side by side, and never collapses them:
    absolute  mean|dW|              -> compare with the documented band
    relative  mean|dW| / mean|w|    -> compare with other RUNS measured the same way
    changed   fraction of elements  -> compare with the band's 30-80%

USAGE
    python3 measure_cpt_delta.py --base <hf dir> --cand <hf dir> [--n 8] [--json]
"""

import argparse
import glob
import json
import os
import sys

# THE TWO AXES ARE NOT EQUALS, and treating them as equals was a real gating error.
# Provenance traced by treasurer 2026-07-27 to RECIPE_v2_synthesis.md:72, commit de5d78eb:
#   L65  the 30-80% changed-fraction figure is a PREDICTION — "~27-50% PREDICTED if per-element
#        steps were ~RMS(p)xlr" — widened, never fitted to any outcome.
#   L73  its own author filed it as a watch-item: "POST-RUN CHANGED-FRACTION HISTOGRAM TO QUANTIFY".
#        It was flagged pending and never settled.
# The magnitude axis is different: ep3, our only +3.4-sigma run, lands at 3.98e-04, mid-band. That
# axis has an outcome behind it.
#
# So: magnitude is PASS/FAIL. Changed-fraction is REPORTED, never pass/fail — it is the histogram
# Gaia asked for, not a gate. Grading a run against a predicted-and-TBD axis would have failed ep3.
BAND_ABS_LO, BAND_ABS_HI = 5e-5, 8e-4
FAIL_LOW_ABS = 2e-7

# The two measured reference points, both produced by this script. The REFERENCE-RELATIVE read is
# the one that carries meaning: where does a candidate land between a run we know was good and a run
# we know was under-dosed by a compressed schedule?
REF_GOOD_ABS = 3.976e-04      # Qwen3.6-27B -> prod_v2_ep3_hf   (+3.4 sigma, changed 0.937)
REF_UNDER_ABS = 3.491e-05     # prod_v2_ep3_hf -> cpt_refresh_v3 (compressed schedule, changed 0.527)


def index(d):
    """tensor name -> shard path. Falls back to scanning when there is no index.json."""
    idx = os.path.join(d, "model.safetensors.index.json")
    if os.path.exists(idx):
        return {k: os.path.join(d, v)
                for k, v in json.load(open(idx))["weight_map"].items()}
    from safetensors import safe_open
    m = {}
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        with safe_open(f, "pt") as h:
            for k in h.keys():
                m[k] = f
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="the PRE-run model")
    ap.add_argument("--cand", required=True, help="the POST-run model")
    ap.add_argument("--n", type=int, default=8, help="decoder tensors to sample")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    from safetensors import safe_open

    ia, ib = index(a.base), index(a.cand)
    common = sorted(set(ia) & set(ib))
    if not common:
        raise SystemExit(f"ABORT: no tensors in common between {a.base} and {a.cand}")

    # Sample DECODER mlp/attn weights spread across depth. The band is a decoder-mlp statistic, so
    # sampling embeddings or norms would compare against a band that does not describe them.
    picks = [k for k in common
             if ".layers." in k and k.endswith(".weight")
             and (".mlp." in k or "self_attn." in k or "linear_attn." in k)]
    if not picks:
        raise SystemExit("ABORT: no decoder mlp/attn weights found — check the model layout")
    step = max(1, len(picks) // a.n)
    sample = picks[::step][:a.n]

    rows, abs_sum, rel_sum, chg_sum, n = [], 0.0, 0.0, 0.0, 0
    for k in sample:
        with safe_open(ia[k], "pt") as h:
            x = h.get_tensor(k).float()
        with safe_open(ib[k], "pt") as h:
            y = h.get_tensor(k).float()
        if x.shape != y.shape:
            rows.append({"tensor": k, "note": "SHAPE DIFF — skipped"})
            continue
        d = (x - y).abs()
        mabs = d.mean().item()
        mw = x.abs().mean().item()
        rel = mabs / mw if mw else float("nan")
        chg = (d > 0).float().mean().item()
        abs_sum += mabs; rel_sum += rel; chg_sum += chg; n += 1
        rows.append({"tensor": k, "abs": mabs, "rel": rel, "changed": chg})

    if not n:
        raise SystemExit("ABORT: every sampled tensor was shape-mismatched")
    A, R, C = abs_sum / n, rel_sum / n, chg_sum / n

    if a.json:
        print(json.dumps({"base": a.base, "cand": a.cand, "n": n,
                          "abs_mean_dW": A, "rel_mean_pct": 100 * R, "changed_frac": C,
                          "rows": rows}, indent=2))
        return 0

    print(f"base : {a.base}")
    print(f"cand : {a.cand}")
    print(f"sampled {n} decoder mlp/attn tensors across depth\n")
    for r in rows:
        if "note" in r:
            print(f"  {r['note']:<28} {r['tensor'][:60]}")
        else:
            print(f"  abs={r['abs']:.3e}  rel={100*r['rel']:.3f}%  changed={r['changed']:.3f}  "
                  f"{r['tensor'].split('layers.')[-1][:44]}")

    print(f"\n  ABSOLUTE mean|dW|      {A:.3e}")
    print(f"  RELATIVE mean|dW|/|w|  {100*R:.3f}%")
    print(f"  CHANGED fraction       {C:.3f}")

    # Verdicts are reported PER AXIS. A run can sit inside the band on one and outside on another —
    # which is exactly what the 0.291% refresh did — and collapsing that into one word loses the
    # only information the reader needs.
    # PRIMARY axis — the only pass/fail. Empirically corroborated by ep3 at mid-band.
    print("\n  --- PRIMARY (pass/fail): decoder mlp mean|dW| ---")
    if A < FAIL_LOW_ABS:
        print(f"  FAIL-LOW — below {FAIL_LOW_ABS:.0e}. This is a null run.")
    elif A < BAND_ABS_LO:
        print(f"  BELOW BAND ({BAND_ABS_LO:.0e}-{BAND_ABS_HI:.0e}), "
              f"{A/FAIL_LOW_ABS:.0f}x above FAIL-LOW — real movement, light dose")
    elif A <= BAND_ABS_HI:
        print(f"  IN BAND ({BAND_ABS_LO:.0e}-{BAND_ABS_HI:.0e})")
    else:
        print(f"  ABOVE BAND (>{BAND_ABS_HI:.0e}) — consider reducing the dose")

    # SECONDARY axis — reported, never graded. See the constants block for why.
    print("\n  --- SECONDARY (reported, NOT pass/fail): changed fraction ---")
    print(f"  {C:.3f}    beside ep3 0.937 (good) and cpt_refresh_v3 0.527 (under-dosed)")
    print("  This axis was PREDICTED, never fitted, and its author filed it pending a histogram.")
    print("  It does not fail a run — grading on it would fail ep3, our only +3.4-sigma CPT.")

    # REFERENCE-RELATIVE — the read that carries meaning.
    print("\n  --- REFERENCE-RELATIVE: where does this land between the two known points? ---")
    span = REF_GOOD_ABS - REF_UNDER_ABS
    pos = (A - REF_UNDER_ABS) / span if span else float("nan")
    print(f"  under-dosed {REF_UNDER_ABS:.3e}  ...  this run {A:.3e}  ...  good {REF_GOOD_ABS:.3e}")
    print(f"  position: {100*pos:.1f}% of the way from the under-dosed run toward the good one")
    print(f"  vs under-dosed: {A/REF_UNDER_ABS:.2f}x     vs good: {A/REF_GOOD_ABS:.2f}x")
    if A <= REF_UNDER_ABS * 1.10:
        print("  READ: has NOT moved decisively off the under-dosed point.")
    else:
        print("  READ: has moved off the under-dosed point toward the band.")

    print("\n  NOTE: the RELATIVE percentage above is NOT graded — the band is an absolute")
    print("  statistic and cannot be converted to a percentage target. Compare the relative")
    print("  figure only with other RUNS measured this same way.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
