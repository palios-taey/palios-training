#!/usr/bin/env bash
# check_no_private_training_material.sh
# Reject reintroduction of private training material classified at c164d35 inventory.
# Does NOT reject generic trainers, loaders, converters, recipes, configs, or validators.
# Public replacements at the same path names (e.g. README.md) are allowed when their
# content sha256 is not in the private blob denylist.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

PATHS_DENY=scripts/private_material_paths.txt
HASH_DENY=scripts/private_material_blob_sha256.txt
fail=0

[ -f "$PATHS_DENY" ] || { echo "missing $PATHS_DENY" >&2; exit 1; }
[ -f "$HASH_DENY" ] || { echo "missing $HASH_DENY" >&2; exit 1; }

echo "private-training-material gate — $(git rev-parse --short HEAD 2>/dev/null || echo nogit)"

declare -A BAD
while read -r h rest; do
  [[ -z "${h:-}" || "$h" =~ ^# ]] && continue
  BAD["$h"]=1
done < "$HASH_DENY"

# 1) path denylist: path may exist only if its content is NOT a private blob hash
while IFS= read -r p || [ -n "${p:-}" ]; do
  [[ -z "$p" || "$p" =~ ^# ]] && continue
  if git ls-files --error-unmatch -- "$p" >/dev/null 2>&1; then
    h=$(sha256sum -- "$p" | awk '{print $1}')
    if [ -n "${BAD[$h]+x}" ]; then
      echo "FAIL private path reintroduced with original bytes: $p ($h)"
      fail=1
    else
      echo "OK path present with non-private bytes (public replacement allowed): $p"
    fi
  fi
done < "$PATHS_DENY"

# 2) content-hash denylist across all tracked files
while IFS= read -r f; do
  [ -f "$f" ] || continue
  case "$f" in
    scripts/private_material_blob_sha256.txt|scripts/private_material_paths.txt) continue ;;
  esac
  h=$(sha256sum -- "$f" | awk '{print $1}')
  if [ -n "${BAD[$h]+x}" ]; then
    echo "FAIL private content hash reintroduced: $f ($h)"
    fail=1
  fi
done < <(git ls-files)

# 3) real operator/service home paths (not /home/user placeholders used as docs)
# Bracket self-non-matching for documentation of this rule is not required here because
# we match concrete account names from this fleet only.
if git grep -nE '/home/(mira|spark|jetson|jesselarose)/' \
  -- ':!scripts/check_no_private_data.sh' \
  ':!scripts/check_no_private_training_material.sh' \
  ':!scripts/private_material_paths.txt' \
  ':!scripts/private_material_blob_sha256.txt' \
  >/tmp/oppath.hits 2>/dev/null; then
  if [ -s /tmp/oppath.hits ]; then
    echo "FAIL operator/service home path literals:"
    cat /tmp/oppath.hits
    fail=1
  fi
fi

if [ "$fail" -ne 0 ]; then
  echo "RESULT: FAIL private training material gate"
  exit 1
fi
echo "RESULT: PASS private-material path+hash denylist clean"
exit 0
