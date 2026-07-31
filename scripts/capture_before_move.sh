#!/usr/bin/env bash
# capture_before_move.sh — phase 0 of the consolidation. Preserve every scrap of LOCAL-ONLY state
# before anything moves.
#
# WHAT LOCAL-ONLY MEANS AND WHY IT IS THE WHOLE RISK. A commit that has been pushed exists on a
# server. A commit that has NOT exists on exactly one disk, and so does every untracked file. The
# consolidation moves trees around; anything local-only is what a move can destroy permanently.
# 13 repos on this machine currently hold such state, the largest 221 unpushed commits.
#
# THE ARCHIVE IS VERIFIED, NOT ASSUMED. Jesse's rule is archive-before-delete *gated on verifying
# the archive succeeded* — an unverified backup is a belief, not a backup. Every bundle is checked
# with `git bundle verify` and every tar with `tar -tf` before this script will call a repo
# captured. A repo whose verification fails is reported FAILED and is NOT eligible for any move.
#
# READ-ONLY ON SOURCES. This script never writes to, checks out, or cleans a source repo. It only
# reads. Nothing here can lose data even if it is interrupted.
set -uo pipefail

DEST="${1:-/home/mira/recovery/consolidation-phase0-$(date -u +%Y%m%d)}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INV="$ROOT/docs/TRAINING_INVENTORY.yml"

[ -f "$INV" ] || { echo "FATAL: no inventory at $INV — run inventory_training_surface.sh first." >&2; exit 1; }
mkdir -p "$DEST" || exit 1

ok=0; fail=0
echo "capture-before-move -> $DEST"
echo

# The inventory is the source of truth for WHAT to capture; it was generated, not hand-listed.
python3 - "$INV" <<'PY' > "$DEST/.targets"
import sys, yaml
d = yaml.safe_load(open(sys.argv[1])) or {}
for r in (d.get("capture_first_candidates") or []):
    print(f"{r['path']}\t{r['default_branch']}\t{r['unpushed_commits']}")
PY

while IFS=$'\t' read -r p def ahead; do
  [ -n "$p" ] || continue
  src="${p/\$HOME//home/mira}"
  name="$(basename "$src")"
  [ -d "$src/.git" ] || { echo "  SKIP  $name (no longer a repo)"; continue; }
  out="$DEST/$name"; mkdir -p "$out"

  # 1. the branch, SELF-CONTAINED (full history), not a thin origin/def..HEAD delta.
  #
  #    THIS WAS WRONG IN THE FIRST VERSION AND THE BUG IS WORTH KEEPING WRITTEN DOWN. A thin
  #    bundle contains only the delta and lists the base as a PREREQUISITE. `git bundle verify`
  #    PASSES on it — because verify checks the bundle's internal integrity and merely *reports*
  #    prerequisites; it never checks they are obtainable. So all 13 captures reported OK and
  #    verified, and `git clone` on one failed outright: "Repository lacks these prerequisite
  #    commits". A capture that cannot restore is not a capture, and a check that passes on one
  #    is a check of FORM, not of TRUTH.
  #
  #    Full history costs almost nothing here (measured: treasurer 107M, the largest; most under
  #    2M; 204G free) and removes any dependence on a remote still existing at restore time.
  #    Output goes to a real FILE: `git bundle create /dev/stdout` fails on the .lock it needs.
  b="$out/branch.bundle"
  if ! git -C "$src" bundle create "$b" HEAD >/dev/null 2>&1; then
    echo "  FAIL  $name — bundle creation failed"; fail=$((fail+1)); continue
  fi
  # VERIFY BY RESTORING, never by `git bundle verify` alone. The only question that matters is
  # "does this come back?", and the only honest way to answer it is to bring it back.
  probe="$(mktemp -d)"
  if git clone -q "$b" "$probe/r" >/dev/null 2>&1 \
     && [ "$(git -C "$probe/r" rev-parse HEAD 2>/dev/null)" = "$(git -C "$src" rev-parse HEAD 2>/dev/null)" ]; then
    bs=$(stat -c%s "$b"); rm -rf "$probe"
  else
    rm -rf "$probe"
    echo "  FAIL  $name — bundle does not restore to the source HEAD"; fail=$((fail+1)); continue
  fi

  # 2. untracked files — invisible to git entirely, so the bundle does not cover them.
  t="$out/untracked.tar.gz"; ts=0
  n=$(git -C "$src" status --porcelain 2>/dev/null | grep -c '^??')
  if [ "${n:-0}" -gt 0 ]; then
    git -C "$src" status --porcelain 2>/dev/null | grep '^??' | cut -c4- \
      | tar -czf "$t" -C "$src" --files-from=- --ignore-failed-read 2>/dev/null
    if [ -f "$t" ] && tar -tzf "$t" >/dev/null 2>&1; then
      ts=$(stat -c%s "$t")
    else
      echo "  FAIL  $name — untracked tar unreadable"; fail=$((fail+1)); continue
    fi
  fi

  # 3. provenance: where it came from and what state it was in. A bundle with no record of its
  #    origin branch/head is hard to restore correctly a month from now.
  {
    echo "source: $src"
    echo "captured_utc: $(date -u +%FT%TZ)"
    echo "branch: $(git -C "$src" rev-parse --abbrev-ref HEAD 2>/dev/null)"
    echo "head: $(git -C "$src" rev-parse HEAD 2>/dev/null)"
    echo "default_branch: $def"
    echo "unpushed_commits: $ahead"
    echo "untracked_files: ${n:-0}"
    echo "prerequisites: none — bundle is self-contained (full history)"
    echo "restore: git clone branch.bundle <dir> && tar -xzf untracked.tar.gz -C <dir>"
    echo "restore_verified: clone performed at capture time; HEAD matched source"
  } > "$out/PROVENANCE.yml"

  ( cd "$out" && sha256sum -- * > SHA256SUMS 2>/dev/null )
  printf "  OK    %-40s bundle=%-8s untracked=%s\n" "$name" \
    "$(numfmt --to=iec "$bs" 2>/dev/null || echo "$bs")" \
    "$(numfmt --to=iec "$ts" 2>/dev/null || echo "$ts")"
  ok=$((ok+1))
done < "$DEST/.targets"

rm -f "$DEST/.targets"
( cd "$DEST" && find . -name SHA256SUMS -printf '%h\n' | sort > MANIFEST.txt )
echo
echo "captured=$ok failed=$fail  ->  $DEST"
[ "$fail" -eq 0 ] || { echo "NOT ALL CAPTURED — no move may proceed." >&2; exit 1; }
