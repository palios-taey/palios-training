#!/usr/bin/env bash
# Public-safe multi-node launcher for dense 9B CPT.
#
# This script runs from the control host and starts
# dense-9b/recipes/launch_cpt_phase2_qwen35_9b_fsdp.sh on each worker over SSH.
# Provide cluster details through environment variables or source
# dense-9b/configs/cpt_cluster.env.example after filling in local values.

set -euo pipefail

require_env() {
    local name="$1"
    if [[ -z "${!name:-}" ]]; then
        echo "ERROR: $name must be set; no deployment-specific default is shipped." >&2
        exit 2
    fi
}

split_csv() {
    local value="$1"
    IFS=',' read -r -a _split_csv_result <<< "$value"
}

quote_env_assignment() {
    local name="$1"
    local value="$2"
    printf '%s=%q' "$name" "$value"
}

append_env_if_set() {
    local name="$1"
    local value="${!name:-}"
    if [[ -n "$value" ]]; then
        ENV_FORWARD+=("$(quote_env_assignment "$name" "$value")")
    fi
}

NUM_NODES="${NUM_NODES:-4}"
SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o ConnectTimeout=10}"
REMOTE_REPO_DIR="${REMOTE_REPO_DIR:-}"
REMOTE_LAUNCHER="${REMOTE_LAUNCHER:-}"
LOG_PATH="${LOG_PATH:-}"
REBOOT_COMMAND="${REBOOT_COMMAND:-sudo -n reboot}"
PKILL_PATTERN="${PKILL_PATTERN:-accelerate launch|train_fsdp_dense_9b.py}"

require_env NODE_HOSTS_CSV
require_env REMOTE_REPO_DIR
require_env MODEL_PATH
require_env CPT_DATA
require_env OUTPUT_DIR
require_env TOTAL_STEPS
require_env MASTER_ADDR

REMOTE_LAUNCHER="${REMOTE_LAUNCHER:-$REMOTE_REPO_DIR/dense-9b/recipes/launch_cpt_phase2_qwen35_9b_fsdp.sh}"
LOG_PATH="${LOG_PATH:-$OUTPUT_DIR/train.log}"

split_csv "$NODE_HOSTS_CSV"
NODE_HOSTS=("${_split_csv_result[@]}")
if [[ "$NUM_NODES" -gt "${#NODE_HOSTS[@]}" ]]; then
    echo "ERROR: NUM_NODES=$NUM_NODES but NODE_HOSTS_CSV has only ${#NODE_HOSTS[@]} hosts." >&2
    exit 2
fi

if [[ -n "${NODE_LABELS_CSV:-}" ]]; then
    split_csv "$NODE_LABELS_CSV"
    NODE_LABELS=("${_split_csv_result[@]}")
    if [[ "$NUM_NODES" -gt "${#NODE_LABELS[@]}" ]]; then
        echo "ERROR: NUM_NODES=$NUM_NODES but NODE_LABELS_CSV has only ${#NODE_LABELS[@]} labels." >&2
        exit 2
    fi
else
    NODE_LABELS=()
    for ((rank=0; rank<NUM_NODES; rank++)); do
        NODE_LABELS+=("node-$rank")
    done
fi

if [[ "${1:-}" == "stop" ]]; then
    for ((rank=0; rank<NUM_NODES; rank++)); do
        host="${NODE_HOSTS[$rank]}"
        ssh $SSH_OPTS "$host" "pkill -9 -f '$PKILL_PATTERN' 2>/dev/null || true; echo stopped rank $rank" &
    done
    wait
    exit 0
fi

if [[ "${1:-}" == "reboot" ]]; then
    for ((rank=0; rank<NUM_NODES; rank++)); do
        host="${NODE_HOSTS[$rank]}"
        ssh $SSH_OPTS "$host" "$REBOOT_COMMAND" >/dev/null 2>&1 &
    done
    wait
    echo "reboot commands issued; waiting for SSH and systemd readiness..."
    for ((rank=0; rank<NUM_NODES; rank++)); do
        host="${NODE_HOSTS[$rank]}"
        until ssh $SSH_OPTS "$host" "uptime" >/dev/null 2>&1; do
            sleep 4
        done
        for _ in $(seq 1 30); do
            state="$(ssh $SSH_OPTS "$host" "systemctl is-system-running" 2>&1 || true)"
            case "$state" in
                running|degraded)
                    echo "  ${NODE_LABELS[$rank]} systemd=$state"
                    break
                    ;;
            esac
            sleep 3
        done
    done
    sleep 5
    echo "all $NUM_NODES nodes reachable"
    exit 0
fi

fail=0
for ((rank=0; rank<NUM_NODES; rank++)); do
    host="${NODE_HOSTS[$rank]}"
    ok="$(ssh $SSH_OPTS "$host" \
        "(test -f '$MODEL_PATH/model.safetensors' -o -f '$MODEL_PATH/model.safetensors.index.json') && \
         test -s '$CPT_DATA' && \
         test -x '$REMOTE_LAUNCHER' && \
         echo ok" 2>/dev/null || echo missing)"
    if [[ "$ok" != "ok" ]]; then
        echo "PRE-FLIGHT FAIL on ${NODE_LABELS[$rank]} ($host): check MODEL_PATH, CPT_DATA, and REMOTE_LAUNCHER." >&2
        fail=1
    fi
done
if [[ "$fail" -ne 0 ]]; then
    echo "Aborting; fix missing remote files first." >&2
    exit 2
fi

ENV_FORWARD=()
for var in NUM_NODES MASTER_ADDR MASTER_PORT \
           MODEL_PATH CPT_DATA GENERAL_DIR SFT_DIR SFT_JSONL OUTPUT_DIR \
           MAX_SEQ BATCH_SIZE_PER_RANK GRAD_ACCUM TOTAL_STEPS RESUME_DELTA \
           SAVE_EVERY SESSION_LIMIT WARMUP_STEPS LR ADAFACTOR_CLIP_THRESHOLD \
           LR_MIN_RATIO WEIGHT_DECAY FORCE_BASE; do
    append_env_if_set "$var"
done

echo "Launching dense 9B CPT across $NUM_NODES nodes:"
for ((rank=0; rank<NUM_NODES; rank++)); do
    echo "  rank $rank -> ${NODE_LABELS[$rank]} (${NODE_HOSTS[$rank]})"
done
echo "  repo: $REMOTE_REPO_DIR"
echo "  launcher: $REMOTE_LAUNCHER"
echo "  log: $LOG_PATH"
echo

pids=()
for ((rank=0; rank<NUM_NODES; rank++)); do
    host="${NODE_HOSTS[$rank]}"
    label="${NODE_LABELS[$rank]}"
    rank_env="$(quote_env_assignment NODE_RANK "$rank")"
    env_prefix="${ENV_FORWARD[*]} $rank_env"
    remote_cmd="
        mkdir -p '$(dirname "$LOG_PATH")'
        cd '$REMOTE_REPO_DIR'
        nohup setsid env $env_prefix bash '$REMOTE_LAUNCHER' </dev/null >'$LOG_PATH' 2>&1 &
        disown
        sleep 1
        pgrep -af 'accelerate launch' | head -1 | grep -q . && echo '[$label] launched' || echo '[$label] FAILED to spawn'
    "
    ssh $SSH_OPTS "$host" "$remote_cmd" &
    pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        fail=1
    fi
done
if [[ "$fail" -ne 0 ]]; then
    echo "ERROR: at least one rank failed to launch. Check remote logs." >&2
    exit 3
fi

echo
echo "All $NUM_NODES ranks dispatched."
echo "Tail rank 0:"
echo "  ssh ${NODE_HOSTS[0]} 'tail -F $LOG_PATH'"
echo "Stop:"
echo "  NODE_HOSTS_CSV=... REMOTE_REPO_DIR=... MODEL_PATH=... CPT_DATA=... OUTPUT_DIR=... TOTAL_STEPS=... MASTER_ADDR=... bash $0 stop"
