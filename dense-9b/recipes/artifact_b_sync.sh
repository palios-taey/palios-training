#!/usr/bin/env bash
[ -f "$(git rev-parse --show-toplevel 2>/dev/null)/fleet.env" ] && . "$(git rev-parse --show-toplevel)/fleet.env"
set -euo pipefail

: "${SPARK_HOME:?fleet.env did not load}"
: "${SPARK_MGMT_IPS:?fleet.env did not load}"

NODES=(${SPARK_MGMT_IPS})
RECIPE_DIR=$(cd "$(dirname "$0")" && pwd)
VERIFY_TOOL="${RECIPE_DIR}/bake_dcp_offline.py"
TRANSFER_MARGIN_BYTES=${TRANSFER_MARGIN_BYTES:-10737418240}

usage(){
  echo "usage:"
  echo "  $0 collect <spark-export-dir> <controller-dir>"
  echo "  $0 push <controller-dir> <convert-ssh> <convert-dir>"
  echo "  $0 retire-sparks <spark-export-dir> <verified-controller-dir>"
}

safe_local_target(){
  case "$1" in
    /*) ;;
    *) echo "REFUSE: local target must be absolute: $1" >&2; return 1;;
  esac
  [ "$1" != "/" ] && [ "$1" != "$HOME" ] || {
    echo "REFUSE: unsafe local target: $1" >&2
    return 1
  }
}

safe_spark_export(){
  case "$1" in
    "${SPARK_HOME}"/exports/*_artifactB) ;;
    *) echo "REFUSE: Spark export must be ${SPARK_HOME}/exports/*_artifactB: $1" >&2; return 1;;
  esac
}

verify_local(){
  python3 "$VERIFY_TOOL" --assembled "$1" --verify-manifests --verify-only
}

write_sums(){
  python3 - "$1" <<'PY'
import glob
import hashlib
import json
import os
import sys

root = sys.argv[1]
rows = {}
for manifest_path in sorted(glob.glob(os.path.join(root, "manifest.rank*.json"))):
    manifest = json.load(open(manifest_path))
    for item in manifest["files"]:
        rows[item["name"]] = item["sha256"]
for name in [".metadata", *[os.path.basename(p) for p in glob.glob(os.path.join(root, "manifest.rank*.json"))],
             *[os.path.basename(p) for p in glob.glob(os.path.join(root, "READY.rank*"))]]:
    path = os.path.join(root, name)
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    rows[name] = digest
with open(os.path.join(root, "SHA256SUMS"), "w") as handle:
    for name in sorted(rows):
        handle.write(f"{rows[name]}  {name}\n")
PY
}

collect(){
  local spark_export=$1 dest=$2
  safe_spark_export "$spark_export"
  safe_local_target "$dest"
  [ "${#NODES[@]}" = 4 ] || {
    echo "REFUSE: Artifact B requires four rank-ordered nodes, got ${#NODES[@]}" >&2
    return 1
  }

  if [ -e "$dest" ]; then
    verify_local "$dest" && {
      echo "SKIP collect — verified Artifact B already exists at $dest"
      return 0
    }
    echo "REFUSE: existing controller destination is not a verified Artifact B: $dest" >&2
    return 1
  fi

  mkdir -p "$(dirname "$dest")"
  local stage
  stage=$(mktemp -d "${dest}.staging.XXXXXX")
  local cleanup_stage=1
  trap 'if [ "${cleanup_stage:-0}" = 1 ]; then rm -rf -- "$stage"; fi' RETURN

  ssh -o BatchMode=yes -o ConnectTimeout=10 spark@"${NODES[0]}" \
    "test -f '$spark_export/.metadata'"
  local rank
  for rank in 0 1 2 3; do
    ssh -o BatchMode=yes -o ConnectTimeout=10 spark@"${NODES[$rank]}" \
      "test -f '$spark_export/__${rank}_0.distcp' &&
       test -f '$spark_export/manifest.rank${rank}.json' &&
       test -f '$spark_export/READY.rank${rank}'"
  done

  scp -q -o BatchMode=yes -o ConnectTimeout=10 \
    "spark@${NODES[0]}:$spark_export/.metadata" "$stage/.metadata"

  local pids=()
  for rank in 0 1 2 3; do
    (
      scp -q -o BatchMode=yes -o ConnectTimeout=10 \
        "spark@${NODES[$rank]}:$spark_export/__${rank}_0.distcp" "$stage/"
      scp -q -o BatchMode=yes -o ConnectTimeout=10 \
        "spark@${NODES[$rank]}:$spark_export/manifest.rank${rank}.json" "$stage/"
      scp -q -o BatchMode=yes -o ConnectTimeout=10 \
        "spark@${NODES[$rank]}:$spark_export/READY.rank${rank}" "$stage/"
    ) &
    pids+=($!)
  done
  local transfer_pid
  for transfer_pid in "${pids[@]}"; do
    wait "$transfer_pid"
  done

  verify_local "$stage"
  write_sums "$stage"
  mv "$stage" "$dest"
  cleanup_stage=0
  trap - RETURN
  echo "COLLECT COMPLETE — verified, atomic Artifact B at $dest"
}

push(){
  local src=$1 convert_ssh=$2 dest=$3
  safe_local_target "$src"
  case "$dest" in
    /*) ;;
    *) echo "REFUSE: convert destination must be absolute: $dest" >&2; return 1;;
  esac
  [ "$dest" != "/" ] || {
    echo "REFUSE: unsafe convert destination: $dest" >&2
    return 1
  }
  verify_local "$src"
  [ -f "$src/SHA256SUMS" ] || write_sums "$src"

  local src_bytes available required
  src_bytes=$(du -sb "$src" | cut -f1)
  available=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$convert_ssh" \
    "df -B1 --output=avail '$(dirname "$dest")' | tail -1 | tr -d ' '")
  required=$((src_bytes + TRANSFER_MARGIN_BYTES))
  [ "$available" -ge "$required" ] || {
    echo "REFUSE: convert host has $available bytes free; staging requires $required" >&2
    return 1
  }

  if ssh -o BatchMode=yes -o ConnectTimeout=10 "$convert_ssh" "test -e '$dest'"; then
    if ssh -o BatchMode=yes -o ConnectTimeout=10 "$convert_ssh" \
      "cd '$dest' && sha256sum -c SHA256SUMS >/dev/null"; then
      echo "SKIP push — verified Artifact B already exists at $convert_ssh:$dest"
      return 0
    fi
    echo "REFUSE: existing convert destination failed verification: $convert_ssh:$dest" >&2
    return 1
  fi

  local remote_stage="${dest}.staging.$$"
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$convert_ssh" \
    "test ! -e '$remote_stage'; mkdir -p '$(dirname "$dest")' '$remote_stage'"
  if ! rsync -a --partial -e "ssh -o BatchMode=yes -o ConnectTimeout=10" \
      "$src/" "$convert_ssh:$remote_stage/"; then
    echo "PUSH FAILED — partial stage retained for diagnosis: $convert_ssh:$remote_stage" >&2
    return 1
  fi
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$convert_ssh" \
    "cd '$remote_stage' && sha256sum -c SHA256SUMS && mv '$remote_stage' '$dest'"
  echo "PUSH COMPLETE — verified Artifact B at $convert_ssh:$dest"
}

retire_sparks(){
  local spark_export=$1 verified_controller=$2
  safe_spark_export "$spark_export"
  safe_local_target "$verified_controller"
  verify_local "$verified_controller"
  [ "$(basename "$spark_export")" = "$(basename "$verified_controller")" ] || {
    echo "REFUSE: verified copy basename does not match Spark export" >&2
    return 1
  }
  local rank existing=0
  for rank in 0 1 2 3; do
    if ssh -o BatchMode=yes -o ConnectTimeout=10 spark@"${NODES[$rank]}" \
        "test -e '$spark_export'"; then
      existing=$((existing + 1))
      ssh -o BatchMode=yes -o ConnectTimeout=10 spark@"${NODES[$rank]}" \
        "test -f '$spark_export/READY.rank${rank}'; rm -rf -- '$spark_export'"
    fi
  done
  [ "$existing" -gt 0 ] || {
    echo "SKIP Spark retirement — Artifact B is already absent from all four nodes"
    return 0
  }
  echo "SPARK EXPORT RETIRED — verified controller copy remains at $verified_controller"
}

command_name=${1:-}
case "$command_name" in
  collect)
    [ "$#" = 3 ] || { usage; exit 2; }
    collect "$2" "$3"
    ;;
  push)
    [ "$#" = 4 ] || { usage; exit 2; }
    push "$2" "$3" "$4"
    ;;
  retire-sparks)
    [ "$#" = 3 ] || { usage; exit 2; }
    retire_sparks "$2" "$3"
    ;;
  *)
    usage
    exit 2
    ;;
esac
