#!/bin/bash
set -euo pipefail

rank=${1:?usage: start_cpt_rank_service.sh NODE_RANK < run.env}
if [[ ! "$rank" =~ ^[0-3]$ ]]; then
  echo "REFUSE: NODE_RANK must be 0, 1, 2, or 3; got $rank" >&2
  exit 1
fi

: "${SPARK_HOME:?SPARK_HOME must name the run user home directory}"

systemd_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$systemd_dir/../../.." && pwd)"
run_user="${PALIOS_TRAINING_USER:-$(id -un)}"
env_dir="${SPARK_HOME}/.config/palios-training"
env_file="${env_dir}/cpt-rank.env"
unit_template="${systemd_dir}/palios-cpt-rank@.service"
unit_install=/etc/systemd/system/palios-cpt-rank@.service
unit_name="palios-cpt-rank@${rank}.service"
journal_dir=/var/log/journal
journal_dropin_dir=/etc/systemd/journald.conf.d
journal_dropin="${journal_dropin_dir}/90-palios-training-persistent.conf"

for substitution in "$repo_root" "$run_user" "$SPARK_HOME" "$env_file"; do
  if [[ "$substitution" == *'#'* || "$substitution" == *$'\n'* ]]; then
    echo "REFUSE: install-time substitution contains an unsupported character" >&2
    exit 1
  fi
done

if sudo systemctl is-active --quiet "$unit_name"; then
  echo "REFUSE: $unit_name is already active" >&2
  exit 1
fi

mkdir -p "$env_dir"
env_tmp=$(mktemp "${env_dir}/.cpt-rank.env.XXXXXX")
unit_tmp=$(mktemp)
journal_tmp=$(mktemp)
cleanup() {
  rm -f -- "$env_tmp" "$unit_tmp" "$journal_tmp"
}
trap cleanup EXIT

line_count=0
while IFS= read -r line || [ -n "$line" ]; do
  if [[ ! "$line" =~ ^[A-Za-z_][A-Za-z0-9_]*=[^[:space:]]*$ ]]; then
    echo "REFUSE: malformed service environment line: $line" >&2
    exit 1
  fi
  printf '%s\n' "$line" >> "$env_tmp"
  line_count=$((line_count + 1))
done

if [ "$line_count" -eq 0 ]; then
  echo "REFUSE: no service environment was supplied on stdin" >&2
  exit 1
fi

for required_key in \
  SPARK_HOME SPARK_RAIL_MASTER MODEL_PATH CPT_DATA MAX_SEQ \
  NCCL_IB_HCA NCCL_NET_GDR_LEVEL TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC \
  BATCH_SIZE_PER_RANK CPT_PACKED CPT_SHORT_BATCH CPT_MID_BATCH \
  CPT_LONG_BATCH TOTAL_STEPS; do
  if ! /usr/bin/grep -q "^${required_key}=" "$env_tmp"; then
    echo "REFUSE: service environment is missing $required_key" >&2
    exit 1
  fi
done

chmod 600 "$env_tmp"
mv -f -- "$env_tmp" "$env_file"

sed \
  -e "s#@PALIOS_TRAINING_ROOT@#${repo_root}#g" \
  -e "s#@PALIOS_TRAINING_USER@#${run_user}#g" \
  -e "s#@SPARK_HOME@#${SPARK_HOME}#g" \
  -e "s#@CPT_ENV_FILE@#${env_file}#g" \
  "$unit_template" > "$unit_tmp"

if ! /usr/bin/grep -qx 'LimitMEMLOCK=infinity' "$unit_tmp"; then
  echo "REFUSE: rendered unit does not grant the memlock required by RDMA" >&2
  exit 1
fi

printf '%s\n' '[Journal]' 'Storage=persistent' > "$journal_tmp"
sudo install -d -m 0755 "$journal_dropin_dir"
sudo install -d -m 2755 "$journal_dir"
if ! sudo cmp -s "$journal_tmp" "$journal_dropin"; then
  sudo install -m 0644 "$journal_tmp" "$journal_dropin"
  sudo systemctl restart systemd-journald
fi
sudo systemd-tmpfiles --create --prefix "$journal_dir"
sudo journalctl --flush
sudo test -d "$journal_dir"

sudo install -m 0644 "$unit_tmp" "$unit_install"
sudo systemctl daemon-reload
if sudo systemctl is-failed --quiet "$unit_name"; then
  sudo systemctl reset-failed "$unit_name"
fi
sudo systemctl start "$unit_name"

if ! sudo systemctl is-active --quiet "$unit_name"; then
  sudo systemctl status --no-pager --lines=40 "$unit_name" >&2
  exit 1
fi

main_pid=$(sudo systemctl show --property MainPID --value "$unit_name")
if [[ ! "$main_pid" =~ ^[1-9][0-9]*$ ]]; then
  echo "REFUSE: $unit_name is active without a MainPID" >&2
  exit 1
fi

printf 'SUPERVISION ACTIVE unit=%s MainPID=%s env=%s journal=%s\n' \
  "$unit_name" "$main_pid" "$env_file" "$journal_dir"
