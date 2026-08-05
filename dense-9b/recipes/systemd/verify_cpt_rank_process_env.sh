#!/bin/bash
set -euo pipefail

rank=${1:?usage: verify_cpt_rank_process_env.sh NODE_RANK}
if [[ ! "$rank" =~ ^[0-3]$ ]]; then
  echo "REFUSE: NODE_RANK must be 0, 1, 2, or 3; got $rank" >&2
  exit 1
fi

unit_name="palios-cpt-rank@${rank}.service"
memlock_limit=$(sudo systemctl show --property LimitMEMLOCK --value "$unit_name")
if [ "$memlock_limit" != infinity ]; then
  echo "REFUSE: $unit_name effective LimitMEMLOCK is $memlock_limit, not infinity" >&2
  exit 1
fi

fragment_path=$(sudo systemctl show --property FragmentPath --value "$unit_name")
if [ -z "$fragment_path" ] || [ ! -r "$fragment_path" ]; then
  echo "REFUSE: cannot read the installed unit for $unit_name" >&2
  exit 1
fi

mapfile -t environment_files < <(
  /usr/bin/awk -F= '$1 == "EnvironmentFile" {print substr($0, index($0, "=") + 1)}' "$fragment_path"
)
if [ "${#environment_files[@]}" -ne 1 ] || [[ "${environment_files[0]}" != /* ]]; then
  echo "REFUSE: $unit_name must name exactly one absolute EnvironmentFile" >&2
  exit 1
fi
environment_file=${environment_files[0]}
if ! sudo test -s "$environment_file"; then
  echo "REFUSE: installed environment file is missing or empty: $environment_file" >&2
  exit 1
fi

control_group=$(sudo systemctl show --property ControlGroup --value "$unit_name")
if [ -z "$control_group" ] || [ ! -r "/sys/fs/cgroup${control_group}/cgroup.procs" ]; then
  echo "REFUSE: cannot resolve the live cgroup for $unit_name" >&2
  exit 1
fi

trainer_pid=
while IFS= read -r pid; do
  [ -r "/proc/${pid}/cmdline" ] || continue
  command_line=$(sudo /usr/bin/cat "/proc/${pid}/cmdline" | tr '\0' ' ')
  case "$command_line" in
    *train_fsdp_dense_9b.py*) trainer_pid=$pid; break ;;
  esac
done < "/sys/fs/cgroup${control_group}/cgroup.procs"

if [ -z "$trainer_pid" ]; then
  echo "REFUSE: no trainer process is live inside $unit_name" >&2
  exit 1
fi

environment=$(sudo /usr/bin/cat "/proc/${trainer_pid}/environ" | tr '\0' '\n')
declare -A manifest_keys=()
manifest_count=0
while IFS= read -r expected || [ -n "$expected" ]; do
  if [[ ! "$expected" =~ ^[A-Za-z_][A-Za-z0-9_]*=[^[:space:]]*$ ]]; then
    echo "REFUSE: installed environment contains a malformed line" >&2
    exit 1
  fi
  key=${expected%%=*}
  if [ -n "${manifest_keys[$key]:-}" ]; then
    echo "REFUSE: installed environment contains duplicate key $key" >&2
    exit 1
  fi
  manifest_keys[$key]=1
  match_count=$(/usr/bin/awk -v expected="$expected" '$0 == expected {count++} END {print count + 0}' <<< "$environment")
  if [ "$match_count" -ne 1 ]; then
    echo "REFUSE: trainer pid $trainer_pid does not match installed value for $key" >&2
    exit 1
  fi
  manifest_count=$((manifest_count + 1))
done < <(sudo /usr/bin/cat "$environment_file")

if [ "$manifest_count" -eq 0 ]; then
  echo "REFUSE: installed environment contains no comparable values" >&2
  exit 1
fi

for expected in \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
  PYTORCH_ALLOC_CONF=expandable_segments:False; do
  match_count=$(/usr/bin/awk -v expected="$expected" '$0 == expected {count++} END {print count + 0}' <<< "$environment")
  if [ "$match_count" -ne 1 ]; then
    echo "REFUSE: trainer pid $trainer_pid did not receive launcher-owned $expected" >&2
    exit 1
  fi
done

printf 'PROCESS ENV PASS unit=%s trainer_pid=%s manifest_keys=%s env_file=%s allocator=expandable_segments:False memlock=infinity\n' \
  "$unit_name" "$trainer_pid" "$manifest_count" "$environment_file"
