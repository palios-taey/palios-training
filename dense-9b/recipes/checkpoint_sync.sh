#!/usr/bin/env bash
# TOPOLOGY from gitignored fleet.env (see fleet.env.example). Never hardcode addresses.
[ -f "$(git rev-parse --show-toplevel 2>/dev/null)/fleet.env" ] && . "$(git rev-parse --show-toplevel)/fleet.env"
# checkpoint_sync.sh — durable DCP checkpoint staging for the NO-SHARED-FS GB10 cluster.
#
# WHY (exp11, PERF_EXPERIMENTS_LOG): the 4 Sparks have NO shared filesystem. FSDP2 DCP writes each
# rank's shard (__<rank>_0.distcp) to that node's LOCAL disk + rank0 writes the small .metadata.
# EMPIRICALLY PROVEN (dcp_toy_ml.py, real 4-node): same-world DCP resume reads ONLY-LOCAL — each node
# needs just its OWN shard + a copy of .metadata + trainer_meta.pt + COMPLETE. So:
#   collect  = pull every rank's shard + the metadata into ONE durable dir on MIRA (survives node
#              reboot / the 2-3h hardware death — the checkpoint lives OFF the fragile compute nodes).
#   scatter  = push each rank's own shard back to its node + metadata/meta/COMPLETE to ALL nodes,
#              so a post-reboot resume finds a complete local checkpoint and dcp.load reads only-local.
#
# rank->node is FIXED: .68=0(master) .80=1 .12=2 .19=3.
# Usage:
#   checkpoint_sync.sh collect <step> [remote_output_dir] [mira_store]
#   checkpoint_sync.sh scatter <step> [mira_store] [remote_resume_dir]
set -uo pipefail

RANK_NODE=($SPARK_MGMT_IPS)   # index = rank
REMOTE_OUT_DEFAULT=${SPARK_HOME}/training_outputs/cpt_27b_full_ft
MIRA_STORE_DEFAULT=${REPO_ROOT}/checkpoints_27b
REMOTE_RESUME_DEFAULT=${SPARK_HOME}/training_outputs/cpt_27b_full_ft   # resume reads from here on each node

SCP(){ scp -q -o ConnectTimeout=10 -o ControlMaster=no -o ControlPath=none "$@" 2>&1 | grep -vaE "ControlSocket|mux_client"; }
S(){ ssh -o ConnectTimeout=8 -o ControlMaster=no -o ControlPath=none spark@"$1" "${@:2}" 2>&1 | grep -vaE "ControlSocket|mux_client"; }

cmd="${1:-}"; step="${2:-}"
[ -z "$cmd" ] || [ -z "$step" ] && { echo "usage: $0 collect|scatter <step> [...]"; exit 2; }
CK="checkpoint-${step}"

collect(){
  local remote_out="${3:-$REMOTE_OUT_DEFAULT}" store="${4:-$MIRA_STORE_DEFAULT}"
  local dest="$store/$CK" stage="$store/$CK.staging"
  echo "=== COLLECT $CK  (nodes -> Mira:$dest) ==="
  # pre-flight: master must have the atomic COMPLETE marker (never stage a partial save)
  if [ "$(S "${RANK_NODE[0]}" "test -f $remote_out/$CK/COMPLETE && echo OK")" != "OK" ]; then
    echo "  REFUSE: no COMPLETE marker in ${RANK_NODE[0]}:$remote_out/$CK (partial/missing save)"; return 1
  fi
  # ATOMIC: collect into .staging, rename to final only when fully verified (a crash mid-collect
  # must never leave a half-checkpoint a later scatter would silently corrupt).
  # PER-RANK SELF-CONTAINED BUNDLES (use_collectives=False): each rank R's bundle on node R is
  # dcp/__R.metadata + dcp/__R_0.distcp + trainer_meta.pt + COMPLETE (NO global .metadata; trainer
  # meta/COMPLETE are per-rank). Mira store: dcp/ holds all rank dcp files (unique names); rank-R/
  # holds that rank's trainer_meta.pt + COMPLETE.
  rm -rf "$stage"; mkdir -p "$stage/dcp"
  local dest_final="$dest"; dest="$stage"   # write to staging; committed to $dest_final at the end
  local r ok=1
  for r in 0 1 2 3; do
    local node="${RANK_NODE[$r]}"
    mkdir -p "$dest/rank-$r"
    SCP "spark@$node:$remote_out/$CK/dcp/__${r}.metadata"  "$dest/dcp/"
    SCP "spark@$node:$remote_out/$CK/dcp/__${r}_0.distcp"  "$dest/dcp/"
    SCP "spark@$node:$remote_out/$CK/trainer_meta.pt"      "$dest/rank-$r/"
    SCP "spark@$node:$remote_out/$CK/COMPLETE"             "$dest/rank-$r/"
  done
  # tokenizer (rank0 only, best-effort — a toy save has none)
  if [ "$(S "${RANK_NODE[0]}" "ls $remote_out/$CK/tokenizer* >/dev/null 2>&1 && echo Y")" = "Y" ]; then
    SCP "spark@${RANK_NODE[0]}:$remote_out/$CK/tokenizer*" "$dest/"
  fi
  # verify every rank's full bundle present
  for r in 0 1 2 3; do
    [ -f "$dest/dcp/__${r}.metadata" ]    || { echo "  MISSING __${r}.metadata"; ok=0; }
    [ -f "$dest/dcp/__${r}_0.distcp" ]    || { echo "  MISSING __${r}_0.distcp"; ok=0; }
    [ -f "$dest/rank-$r/trainer_meta.pt" ]|| { echo "  MISSING rank-$r/trainer_meta.pt"; ok=0; }
    [ -f "$dest/rank-$r/COMPLETE" ]       || { echo "  MISSING rank-$r/COMPLETE"; ok=0; }
  done
  if [ "$ok" = 1 ]; then
    rm -rf "$dest_final"; mv "$stage" "$dest_final"     # atomic commit
    du -sh "$dest_final" 2>/dev/null | awk '{print "  collected OK, size="$1}'
    echo "  DURABLE (atomic-committed, 4 per-rank bundles): $dest_final"
  else
    echo "  COLLECT INCOMPLETE — discarding staging, do NOT resume from this"; rm -rf "$stage"; return 1
  fi
}

scatter(){
  local store="${3:-$MIRA_STORE_DEFAULT}" remote_resume="${4:-$REMOTE_RESUME_DEFAULT}"
  local src="$store/$CK"
  echo "=== SCATTER $CK  (Mira:$src -> nodes:$remote_resume) — per-rank self-contained bundles ==="
  [ -f "$src/dcp/__0.metadata" ] && [ -f "$src/rank-0/trainer_meta.pt" ] || { echo "  REFUSE: Mira store $src incomplete/old-format"; return 1; }
  # NOTE: scatter is for node-DEATH durability. A plain REBOOT does NOT need it — the per-rank
  # bundles are self-contained and persist on each node's local ext4 across reboot (resume directly).
  local r ok=1
  for r in 0 1 2 3; do
    local node="${RANK_NODE[$r]}"
    S "$node" "mkdir -p $remote_resume/$CK/dcp"
    SCP "$src/dcp/__${r}.metadata"      "spark@$node:$remote_resume/$CK/dcp/"   # rank R's OWN metadata
    SCP "$src/dcp/__${r}_0.distcp"      "spark@$node:$remote_resume/$CK/dcp/"   # rank R's OWN shard
    SCP "$src/rank-$r/trainer_meta.pt"  "spark@$node:$remote_resume/$CK/"
    SCP "$src/rank-$r/COMPLETE"         "spark@$node:$remote_resume/$CK/"
    local have
    have=$(S "$node" "cd $remote_resume/$CK 2>/dev/null && ls dcp/__${r}.metadata dcp/__${r}_0.distcp trainer_meta.pt COMPLETE 2>/dev/null | wc -l")
    if [ "$have" = "4" ]; then echo "  .${node##*.} rank$r OK (self-contained bundle)"; else echo "  .${node##*.} rank$r INCOMPLETE ($have/4)"; ok=0; fi
  done
  [ "$ok" = 1 ] && echo "  SCATTER COMPLETE — resume with RESUME_DELTA=$remote_resume/$CK on every node" || { echo "  SCATTER INCOMPLETE"; return 1; }
}

case "$cmd" in
  collect) collect "$@";;
  scatter) scatter "$@";;
  *) echo "unknown cmd: $cmd (collect|scatter)"; exit 2;;
esac
