#!/bin/bash
set -euo pipefail

rank=${1:?usage: run_cpt_rank.sh NODE_RANK}
if [[ ! "$rank" =~ ^[0-3]$ ]]; then
  echo "REFUSE: NODE_RANK must be 0, 1, 2, or 3; got $rank" >&2
  exit 1
fi

: "${SPARK_HOME:?SPARK_HOME must be present in the service environment}"

systemd_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
recipe_dir="$(dirname "$systemd_dir")"
log_dir="${SPARK_HOME}/cpt27b_logs"
log_file="${log_dir}/r${rank}.log"

mkdir -p "$log_dir"
# Rotate before every service attempt. A later launch once truncated a completed run's
# forensic log with a short failed-launch stub; systemd restarts must not repeat that loss.
if [ -s "$log_file" ]; then
  log_stamp=$(date -r "$log_file" +%Y%m%dT%H%M%S)
  mv -- "$log_file" "${log_dir}/r${rank}.${log_stamp}.log"
fi

cd "$recipe_dir"
echo "SUPERVISOR START rank=$rank pid=$$ utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

set +e
./launch_cpt_qwen36_27b_fsdp.sh "$rank" 2>&1 | /usr/bin/tee "$log_file"
pipeline_status=("${PIPESTATUS[@]}")
set -e

launcher_status=${pipeline_status[0]}
tee_status=${pipeline_status[1]}
if [ "$tee_status" -ne 0 ]; then
  echo "SUPERVISOR EXIT rank=$rank launcher_status=$launcher_status tee_status=$tee_status" >&2
  exit "$tee_status"
fi

echo "SUPERVISOR EXIT rank=$rank launcher_status=$launcher_status tee_status=$tee_status"
exit "$launcher_status"
