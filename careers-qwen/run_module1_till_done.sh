#!/bin/bash
# TOPOLOGY from gitignored fleet.env (see fleet.env.example). Never hardcode addresses.
[ -f "$(git rev-parse --show-toplevel 2>/dev/null)/fleet.env" ] && . "$(git rev-parse --show-toplevel)/fleet.env"
# Module-1 full run on the PRODUCTION stack, session-cycled under the 2h thermal wall.
# reboot -> run_4node_27b_cpt.sh (LORA_MODE) -> wait for session exit -> resume -> repeat.
set -uo pipefail
NODES=($SPARK_MGMT_IPS); MASTER=${SPARK_MASTER}
REC=${REPO_ROOT}/dense-9b/recipes
CKPT=${SPARK_HOME}/training_outputs/module1_lora
# SESSION_LIMIT is sized from MEASURED step time against the 2h thermal wall — never guessed.
# Measured 2026-07-21 on this exact config (MAX_SEQ=4096, LoRA, 4 nodes): 16.78s/step over steps
# 2..20. SL=900 (the old value, sized on a stale 6.2s/step figure) = 4.32h = MORE THAN DOUBLE the
# wall, and with SAVE_EVERY=SL the first checkpoint would land at 4.19h — i.e. the node dies at ~2h
# having saved NOTHING. SL=350 -> 1.63h step time + ~0.13h setup = 1.76h, leaving margin for the
# step time to grow (length-sorted bucket batching means later buckets hold longer sequences).
# If step time changes, RE-MEASURE and resize. Do not raise this to "get more done per session".
TOTAL=${TOTAL:-2304}; SL=${SL:-250}
say(){ echo "[m1 $(date +%H:%M:%S)] $*"; }

latest(){ ssh -o ConnectTimeout=6 spark@"$MASTER" "ls -d $CKPT/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1" 2>/dev/null; }

for attempt in $(seq 1 12); do
  cur=$(latest); cur=${cur:-0}
  say "=== session $attempt: at step ${cur}/${TOTAL} ==="
  [ "$cur" -ge "$TOTAL" ] && { say "COMPLETE at $cur"; break; }

  say "reboot all 4 (ALWAYS)"
  for n in "${NODES[@]}"; do timeout 8 ssh -o ConnectTimeout=5 spark@"$n" 'sudo reboot' 2>/dev/null; done
  sleep 30
  up=0
  for t in $(seq 1 45); do
    up=0; for n in "${NODES[@]}"; do timeout 5 ssh -o ConnectTimeout=4 -o BatchMode=yes spark@"$n" true 2>/dev/null && up=$((up+1)); done
    [ "$up" -eq 4 ] && break; sleep 10
  done
  [ "$up" -eq 4 ] || { say "ABORT: only $up/4 back"; exit 1; }
  sleep 20

  # DISK GATE (2026-07-24: ckpt-2295 save died ENOSPC on .68 at 100% — 8-10 retained checkpoints,
  # rotation never mechanized; the rotate-checkpoints lesson now IS mechanical). Fail loud under 40G.
  FREE_G=$(ssh -o ConnectTimeout=6 spark@"$MASTER" "df --output=avail -BG ${SPARK_HOME} | tail -1 | tr -dc 0-9" 2>/dev/null)
  [ "${FREE_G:-0}" -ge 40 ] || { say "ABORT: only ${FREE_G:-?}G free on $MASTER (<40G) — rotate/clean first"; exit 1; }

  RD=""; [ "$cur" -gt 0 ] && RD="$CKPT/checkpoint-$cur"
  # OPTIMIZER SWITCH (Adafactor→AdamW LoRA root fix): checkpoint-1545 is the LAST Adafactor checkpoint;
  # resuming from it (or earlier) needs a MODEL-ONLY load (fresh AdamW optim). AdamW checkpoints (>1545)
  # resume their optimizer normally. This one-line gate makes only the transition session model-only.
  RMO=0; [ "${cur:-0}" -le 2045 ] && RMO=1  # <=1545 Adafactor-era; 2045 tensor-step-format (pre step-coercion fix) — both need model-only
  say "launching session (SESSION_LIMIT=$SL, resume=${RD:-none}, RESUME_MODEL_ONLY=$RMO)"
  ( cd "$REC" && env LORA_MODE=1 LORA_R=16 LORA_ALPHA=32 LORA_DROPOUT=0.05 \
      LORA_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,in_proj_qkv,out_proj" \
      SFT_DIR=${SPARK_HOME}/module1_sft SFT_JSONL=${SPARK_HOME}/module1_sft/module1_train.jsonl \
      MODEL_PATH=${SPARK_HOME}/models/prod_v2_ep3_hf OUTPUT_DIR=$CKPT \
      MAX_SEQ=4096 BATCH_SIZE_PER_RANK=1 \
      LANE_WEIGHTS="stage2_scorer=0.45,jesse_voice=0.35,repo_capability=0.12,values=0.08" \
      TINY_LANE_CAP=3 TINY_LANE_THRESHOLD=500 \
      TOTAL_STEPS=$TOTAL SESSION_LIMIT=$SL SAVE_EVERY=$SL CHECKPOINT_DCP=1 \
      ADAFACTOR_DOSE_LOG=0 \
      NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,PROXY TORCH_NCCL_TRACE_BUFFER_SIZE=20000 \
      LR=1e-4 WARMUP_STEPS=25 CLOCK_CAP=1600 RESUME_MODEL_ONLY=$RMO ${RD:+RESUME_DELTA=$RD} \
      bash run_4node_27b_cpt.sh ) >> ${MIRA_HOME}/.claude/jobs/dd98ecbe/tmp/m1_sessions.log 2>&1

  say "session launched; waiting for exit (max 2.2h)"
  for t in $(seq 1 160); do
    a=0; for n in "${NODES[@]}"; do timeout 6 ssh -o ConnectTimeout=5 -o BatchMode=yes spark@"$n" 'tmux has-session -t cpt27b 2>/dev/null && echo A' 2>/dev/null | grep -q A && a=$((a+1)); done
    [ "$a" -eq 0 ] && { say "session ended"; break; }
    sleep 50
  done
  new=$(latest); say "session $attempt done: step ${cur} -> ${new:-0}"

  # ARCHIVE EVIDENCE before r0.log is overwritten by the next session (Jesse: "it worked" must be
  # SHOWABLE + "logs must flow" — manual archival kept losing the proof; this makes it automatic).
  ARCH=${MIRA_HOME}/treasurer/foundations/careers/training_data/runs/module1_lora_2026-07-22/logs
  mkdir -p "$ARCH"
  ssh -o ConnectTimeout=8 spark@"$MASTER" 'cat $HOME/cpt27b_logs/r0.log' > "$ARCH/r0_session_${cur}to${new:-0}.log" 2>/dev/null \
    && say "archived r0_session_${cur}to${new:-0}.log ($(grep -aoE 'SR-DELTA.*(PASS|FAIL|LIVE)' "$ARCH/r0_session_${cur}to${new:-0}.log" 2>/dev/null | tail -1 | grep -oE 'PASS|FAIL|LIVE' || echo 'no-SR'))"
  # checkpoint hash-chain = the weight-diff evidence (differ = real training)
  [ "${new:-0}" != "$cur" ] && ssh -o ConnectTimeout=8 spark@"$MASTER" "echo \"ckpt-${new} \$(md5sum $CKPT/checkpoint-${new}/dcp/__0_0.distcp 2>/dev/null | cut -c1-16)\"" >> "$ARCH/checkpoint_hashchain.txt" 2>/dev/null

  # CHECKPOINT ROTATION (mechanical, all 4 nodes): after a successful bank, keep only the last TWO
  # COMPLETE checkpoints — each is ~13G/node and un-rotated retention is what filled .68 (ENOSPC
  # mid-save = incomplete checkpoint = lost session). COMPLETE-marker-gated: never deletes the
  # only-good resume point.
  if [ "${new:-0}" != "$cur" ]; then
    for n in "${NODES[@]}"; do
      ssh -o ConnectTimeout=8 spark@"$n" 'cd '"$CKPT"' 2>/dev/null && KEEP=$(for c in checkpoint-*; do [ -f "$c/COMPLETE" ] && echo "${c#checkpoint-}"; done | sort -n | tail -2 | tr "\n" "|" | sed "s/|$//"); for c in checkpoint-*; do s="${c#checkpoint-}"; echo "$s" | grep -qE "^(${KEEP:-NONE})$" || rm -rf "$c"; done' 2>/dev/null
    done
    say "rotated: kept last 2 COMPLETE checkpoints on all nodes"
  fi

  [ "${new:-0}" = "$cur" ] && { say "NO PROGRESS — stopping for diagnosis"; exit 1; }
done
say "MODULE-1 RUN FINISHED at step $(latest)/$TOTAL"
