#!/bin/bash
# TOPOLOGY from gitignored fleet.env (see fleet.env.example). Never hardcode addresses.
[ -f "$(git rev-parse --show-toplevel 2>/dev/null)/fleet.env" ] && . "$(git rev-parse --show-toplevel)/fleet.env"
# run_epoch_probe.sh — one-command per-epoch behavioral test (Jesse: test after every epoch).
# Runs interim_recall_probe against BASE and EPOCH endpoints (parallel — 2 Thors), then emits
# a single per-slice comparison report with deltas.
# Usage: run_epoch_probe.sh <base_endpoint> <base_model> <epoch_endpoint> <epoch_model> <tag>
#   e.g. run_epoch_probe.sh http://${THOR1_HOST}:8000/v1 qwen36-base http://${THOR2_HOST}:8000/v1 prod-v2-ep1 ep1
set -uo pipefail
BASE_EP=$1; BASE_MODEL=$2; EPOCH_EP=$3; EPOCH_MODEL=$4; TAG=$5
DIR=${REPO_ROOT}/careers-qwen
SLICES=$DIR/data/slices_v2_probe          # 6 sha-verified registered slices (repos-v2 + strategy-delta pending re-emit)
OUT=$DIR/probe_results
mkdir -p "$OUT"

echo "[probe-run] BASE pass ($BASE_MODEL @ $BASE_EP) + EPOCH pass ($EPOCH_MODEL @ $EPOCH_EP) in parallel"
python3 "$DIR/interim_recall_probe.py" --endpoint "$BASE_EP" --model "$BASE_MODEL" \
  --slices-dir "$SLICES" --n-per-slice 20 --out "$OUT/probe_${TAG}_base.json" > "$OUT/probe_${TAG}_base.log" 2>&1 &
B=$!
python3 "$DIR/interim_recall_probe.py" --endpoint "$EPOCH_EP" --model "$EPOCH_MODEL" \
  --slices-dir "$SLICES" --n-per-slice 20 --out "$OUT/probe_${TAG}_epoch.json" > "$OUT/probe_${TAG}_epoch.log" 2>&1 &
E=$!
wait $B; brc=$?
wait $E; erc=$?
[ $brc -ne 0 ] && { echo "[probe-run] BASE pass FAILED (rc=$brc):"; tail -5 "$OUT/probe_${TAG}_base.log"; exit 1; }
[ $erc -ne 0 ] && { echo "[probe-run] EPOCH pass FAILED (rc=$erc):"; tail -5 "$OUT/probe_${TAG}_epoch.log"; exit 1; }

python3 - "$OUT/probe_${TAG}_base.json" "$OUT/probe_${TAG}_epoch.json" <<'EOF'
import json, sys
base = json.load(open(sys.argv[1]))["per_slice"]
ep   = json.load(open(sys.argv[2]))["per_slice"]
print(f"\n{'slice':40s} {'n':>3s} {'base':>7s} {'epoch':>7s} {'delta':>8s}")
for sl in sorted(set(base) | set(ep)):
    b = base.get(sl, {}).get("mean_containment"); e = ep.get(sl, {}).get("mean_containment")
    n = ep.get(sl, {}).get("n", base.get(sl, {}).get("n", 0))
    d = (e - b) if (b is not None and e is not None) else None
    print(f"{sl:40s} {n:3d} {b if b is not None else '—':>7} {e if e is not None else '—':>7} "
          f"{('%+.4f' % d) if d is not None else '—':>8}")
print("\nRead: delta>0 = trained model recalls corpus content better than base (deterministic "
      "key-fact containment, greedy, provenance-clean probes drawn from registered slices).")
print("Coverage note: 6/8 slices (cpt_public_repos_v2 + cpt_strategy_research_delta_v1 pending "
      "re-emission by treasurer — gitignored build artifacts).")
EOF
echo "[probe-run] DONE — artifacts: $OUT/probe_${TAG}_{base,epoch}.json"
