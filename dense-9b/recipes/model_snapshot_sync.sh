#!/usr/bin/env bash
[ -f "$(git rev-parse --show-toplevel 2>/dev/null)/fleet.env" ] && . "$(git rev-parse --show-toplevel)/fleet.env"
set -euo pipefail

: "${SPARK_HOME:?fleet.env did not load}"
: "${SPARK_MASTER:?fleet.env did not load}"

TRANSFER_MARGIN_BYTES=${TRANSFER_MARGIN_BYTES:-10737418240}

usage(){
  echo "usage:"
  echo "  $0 collect <spark-model-dir> <controller-dir>"
  echo "  $0 push <controller-dir> <convert-ssh> <convert-dir>"
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

safe_spark_model(){
  case "$1" in
    "${SPARK_HOME}"/models/*) ;;
    *) echo "REFUSE: model source must be ${SPARK_HOME}/models/*: $1" >&2; return 1;;
  esac
}

verify_snapshot(){
  local root=$1
  [ -f "$root/SOURCE_SHA256SUMS" ] || {
    echo "REFUSE: snapshot has no SOURCE_SHA256SUMS: $root" >&2
    return 1
  }
  (
    cd "$root"
    sha256sum -c SOURCE_SHA256SUMS
  ) >/dev/null
  python3 - "$root" <<'PY'
import json
import os
import sys

root = sys.argv[1]
index_path = os.path.join(root, "model.safetensors.index.json")
if not os.path.isfile(index_path):
    raise SystemExit(f"REFUSE: snapshot has no safetensors index: {root}")
weight_map = json.load(open(index_path))["weight_map"]
missing = sorted({name for name in weight_map.values()
                  if not os.path.isfile(os.path.join(root, name))})
if missing:
    raise SystemExit(f"REFUSE: snapshot is missing indexed shards: {missing}")
print(f"MODEL SNAPSHOT VERIFIED — {len(weight_map)} tensors at {root}")
PY
}

source_sums(){
  local source=$1 output=$2
  ssh -o BatchMode=yes -o ConnectTimeout=10 spark@"$SPARK_MASTER" \
    "cd '$source' &&
     find . -maxdepth 1 -type f ! -name SOURCE_SHA256SUMS -printf '%P\\0' |
       sort -z | xargs -0 sha256sum" > "$output"
  [ -s "$output" ] || {
    echo "REFUSE: source checksum manifest is empty: $source" >&2
    return 1
  }
}

collect(){
  local source=$1 dest=$2
  safe_spark_model "$source"
  safe_local_target "$dest"

  if [ -e "$dest" ]; then
    verify_snapshot "$dest" && {
      echo "SKIP model collect — verified snapshot already exists at $dest"
      return 0
    }
    echo "REFUSE: existing controller destination is not a verified model snapshot: $dest" >&2
    return 1
  fi

  ssh -o BatchMode=yes -o ConnectTimeout=10 spark@"$SPARK_MASTER" \
    "test -f '$source/model.safetensors.index.json'"
  mkdir -p "$(dirname "$dest")"
  local stage
  stage=$(mktemp -d "${dest}.staging.XXXXXX")
  local cleanup_stage=1
  trap 'if [ "${cleanup_stage:-0}" = 1 ]; then rm -rf -- "$stage"; fi' RETURN

  source_sums "$source" "$stage/SOURCE_SHA256SUMS"
  rsync -rt --partial --no-perms --no-owner --no-group --exclude SOURCE_SHA256SUMS \
    -e "ssh -o BatchMode=yes -o ConnectTimeout=10" \
    "spark@${SPARK_MASTER}:$source/" "$stage/"
  verify_snapshot "$stage"
  mv "$stage" "$dest"
  cleanup_stage=0
  trap - RETURN
  echo "MODEL COLLECT COMPLETE — source-hash-verified, atomic snapshot at $dest"
}

push(){
  local source=$1 convert_ssh=$2 dest=$3
  safe_local_target "$source"
  case "$dest" in
    /*) ;;
    *) echo "REFUSE: convert destination must be absolute: $dest" >&2; return 1;;
  esac
  [ "$dest" != "/" ] || {
    echo "REFUSE: unsafe convert destination: $dest" >&2
    return 1
  }
  verify_snapshot "$source"

  local source_bytes available required
  source_bytes=$(du -sb "$source" | cut -f1)
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$convert_ssh" \
    "mkdir -p '$(dirname "$dest")'"
  available=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$convert_ssh" \
    "df -B1 --output=avail '$(dirname "$dest")' | tail -1 | tr -d ' '")
  required=$((source_bytes + TRANSFER_MARGIN_BYTES))
  [ "$available" -ge "$required" ] || {
    echo "REFUSE: convert host has $available bytes free; model staging requires $required" >&2
    return 1
  }

  if ssh -o BatchMode=yes -o ConnectTimeout=10 "$convert_ssh" "test -e '$dest'"; then
    if ssh -o BatchMode=yes -o ConnectTimeout=10 "$convert_ssh" \
      "cd '$dest' && sha256sum -c SOURCE_SHA256SUMS >/dev/null"; then
      echo "SKIP model push — verified snapshot already exists at $convert_ssh:$dest"
      return 0
    fi
    echo "REFUSE: existing convert model destination failed source verification: $convert_ssh:$dest" >&2
    return 1
  fi

  local remote_stage="${dest}.staging.$$"
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$convert_ssh" \
    "test ! -e '$remote_stage'; mkdir -p '$remote_stage'"
  if ! rsync -a --partial -e "ssh -o BatchMode=yes -o ConnectTimeout=10" \
      "$source/" "$convert_ssh:$remote_stage/"; then
    echo "MODEL PUSH FAILED — partial stage retained: $convert_ssh:$remote_stage" >&2
    return 1
  fi
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$convert_ssh" \
    "cd '$remote_stage' && sha256sum -c SOURCE_SHA256SUMS >/dev/null &&
     mv '$remote_stage' '$dest'"
  echo "MODEL PUSH COMPLETE — source-hash-verified snapshot at $convert_ssh:$dest"
}

case "${1:-}" in
  collect)
    [ "$#" = 3 ] || { usage; exit 2; }
    collect "$2" "$3"
    ;;
  push)
    [ "$#" = 4 ] || { usage; exit 2; }
    push "$2" "$3" "$4"
    ;;
  *)
    usage
    exit 2
    ;;
esac
