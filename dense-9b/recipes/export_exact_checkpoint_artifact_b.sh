#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
FLEET_ENV="${FLEET_ENV:-$REPO_ROOT/fleet.env}"
[ -f "$FLEET_ENV" ] || {
  echo "REFUSE: fleet configuration is missing: $FLEET_ENV" >&2
  exit 1
}
. "$FLEET_ENV"

: "${SPARK_HOME:?fleet.env must define SPARK_HOME}"
: "${SPARK_MGMT_IPS:?fleet.env must define SPARK_MGMT_IPS}"
: "${SPARK_USER:?fleet.env must define SPARK_USER}"
: "${NCCL_IB_HCA:?run manifest must define NCCL_IB_HCA}"
: "${NCCL_NET_GDR_LEVEL:?run manifest must define NCCL_NET_GDR_LEVEL}"
: "${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:?run manifest must define TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC}"

usage() {
  echo "usage:"
  echo "  $0 inspect <step> <checkpoint-root>"
  echo "  $0 export <step> <checkpoint-root> <spark-artifact-b-dir>"
}

mode=${1:-}
step=${2:-}
checkpoint_root=${3:-}
artifact_b_dir=${4:-}
case "$mode" in
  inspect) [ "$#" -eq 3 ] || { usage; exit 2; } ;;
  export) [ "$#" -eq 4 ] || { usage; exit 2; } ;;
  *) usage; exit 2 ;;
esac
[[ "$step" =~ ^[0-9]+$ ]] || {
  echo "REFUSE: step must be a non-negative integer" >&2
  exit 1
}
case "$checkpoint_root" in
  /*) ;;
  *) echo "REFUSE: checkpoint root must be absolute" >&2; exit 1 ;;
esac
[ "$checkpoint_root" != "/" ] && [ "$checkpoint_root" != "$SPARK_HOME" ] || {
  echo "REFUSE: checkpoint root is too broad" >&2
  exit 1
}

nodes=(${SPARK_MGMT_IPS})
[ "${#nodes[@]}" -eq 4 ] || {
  echo "REFUSE: exact Artifact B export requires four rank-ordered nodes" >&2
  exit 1
}
checkpoint="${checkpoint_root%/}/checkpoint-${step}"
remote_repo_root="${REMOTE_PALIOS_TRAINING_ROOT:-${SPARK_HOME}/palios-training}"
remote_inspector="${remote_repo_root}/dense-9b/recipes/inspect_dcp_checkpoint.py"

echo "=== exact DCP checkpoint inspection: step=$step ==="
for rank in 0 1 2 3; do
  host=${nodes[$rank]}
  printf -v remote_command \
    'python3 %q --checkpoint %q --step %q --rank %q --world-size 4' \
    "$remote_inspector" "$checkpoint" "$step" "$rank"
  ssh -o BatchMode=yes -o ConnectTimeout=10 "${SPARK_USER}@${host}" "$remote_command"
done
echo "EXACT CHECKPOINT PASS step=$step ranks=4/4 path=$checkpoint"

if [ "$mode" = inspect ]; then
  exit 0
fi
case "$artifact_b_dir" in
  "${SPARK_HOME}"/exports/*_artifactB) ;;
  *) echo "REFUSE: Artifact B path must be ${SPARK_HOME}/exports/*_artifactB" >&2; exit 1 ;;
esac
unset BAKE_TO_HF
RESUME_DELTA="$checkpoint" EXPORT_DCP="$artifact_b_dir" \
  bash "$SCRIPT_DIR/bake_27b.sh"
