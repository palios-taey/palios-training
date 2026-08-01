#!/usr/bin/env bash
# inventory_training_surface.sh — enumerate EVERYTHING training-related, machine + fleet.
#
# WHY THIS IS A SCRIPT AND NOT A HAND-WRITTEN DOCUMENT. A hand-written inventory is true on the
# day it is typed and silently wrong afterwards, and it cannot be re-verified by anyone who did
# not watch it being made. Today a doc asserted three paths that no longer existed
# (METRICS_PROVENANCE.md, audit_results/, configs/) and every reader believed them for weeks.
# This regenerates, so drift shows up as a diff instead of as a surprise.
#
# CAPTURE-BEFORE-MOVE: this is the record that makes "nothing deleted un-manifested" checkable.
# Run it BEFORE any consolidation phase moves anything.
#
# Emits YAML to stdout. Contains NO literal hosts — nodes are positional and resolve through
# fleet.env, because this file is destined for a repo and the private-data gate is right to
# refuse addresses.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$ROOT/fleet.env" ] && . "$ROOT/fleet.env" 2>/dev/null || true

say(){ printf '%s\n' "$*"; }
# du/find on a missing path must not abort the sweep — absence is itself a finding worth recording.
sz(){ [ -e "$1" ] && du -sh "$1" 2>/dev/null | cut -f1 || echo "ABSENT"; }
ct(){ [ -e "$1" ] && find "$1" -type f 2>/dev/null | wc -l || echo 0; }

say "# Training-surface inventory — generated, do not hand-edit"
say "# Regenerate: scripts/inventory_training_surface.sh > docs/TRAINING_INVENTORY.yml"
say "generated_by: scripts/inventory_training_surface.sh"
say "purpose: capture-before-move record for the consolidation; nothing deleted un-manifested"
say ""

# ---------------------------------------------------------------- repo trees
say "repo_trees:"
say "  # Every tree of the training repo on this machine. 'One production tree per surface' is"
say "  # the goal; this section is the measurement of how far from it we are."
find "$HOME" /media -maxdepth 4 -type d -name '.git' 2>/dev/null | sed 's|/.git$||' | sort | while read -r r; do
  u=$(git -C "$r" config --get remote.origin.url 2>/dev/null)
  case "$u" in *palios-training*) ;; *) continue;; esac
  say "  - path: \"${r/#$HOME/\$HOME}\""
  say "    kind: clone"
  say "    branch: $(git -C "$r" rev-parse --abbrev-ref HEAD 2>/dev/null)"
  say "    head: $(git -C "$r" rev-parse --short HEAD 2>/dev/null)"
  say "    dirty_entries: $(git -C "$r" status --porcelain 2>/dev/null | wc -l)"
  say "    tracked_files: $(git -C "$r" ls-files 2>/dev/null | wc -l)"
done
git -C "$ROOT" worktree list 2>/dev/null | tail -n +2 | while read -r p h b; do
  say "  - path: \"${p/#$HOME/\$HOME}\""
  say "    kind: worktree"
  say "    head: $h"
  say "    branch: \"${b//[\[\]]/}\""
done

# ---------------------------------------------------------------- training data
say ""
say "training_data:"
say "  # NEVER public. In the target design these are MANIFESTED, not stored in git."
for d in \
  "$HOME/treasurer/foundations/careers/training_data/careers_qwen" \
  "$HOME/data/corpus/tier0_infra/raw" \
  "/media/mira/Expansion/training-artifacts/corpora" ; do
  say "  - path: \"${d/#$HOME/\$HOME}\""
  say "    size: $(sz "$d")"
  say "    files: $(ct "$d")"
  say "    public: false"
done

# ---------------------------------------------------------------- artifacts
say ""
say "artifacts:"
for d in "/media/mira/Expansion/training-artifacts" "$HOME/models" ; do
  say "  - path: \"${d/#$HOME/\$HOME}\""
  say "    size: $(sz "$d")"
done

# ---------------------------------------------------------------- fleet nodes
say ""
say "fleet_nodes:"
say "  # Positional identity; addresses resolve through fleet.env (SPARK_MGMT_IPS)."
i=0
for h in ${SPARK_MGMT_IPS:-}; do
  i=$((i+1))
  out=$(ssh -o BatchMode=yes -o ConnectTimeout=8 "spark@$h" \
    "$(printf '%q ' bash -s -- "${SPARK_HOME%/}")" <<'REMOTE' 2>/dev/null
spark_home=$1
echo "$(du -sh "$spark_home/training_outputs" 2>/dev/null|cut -f1):$(du -sh "$spark_home/models" 2>/dev/null|cut -f1):$(du -sh /var/spark/isma/training 2>/dev/null|cut -f1):$(df -h /home 2>/dev/null|tail -1|awk "{print \$4}")"
REMOTE
  )
  say "  - node: $i"
  say "    training_outputs: ${out%%:*}"
  say "    models: $(echo "$out" | cut -d: -f2)"
  say "    corpora: $(echo "$out" | cut -d: -f3)"
  say "    disk_free: $(echo "$out" | cut -d: -f4)"
done
[ "$i" -eq 0 ] && say "  # SPARK_MGMT_IPS unset — fleet section empty; source fleet.env and re-run."

# ---------------------------------------------------------------- adjacent
say ""
say "adjacent_repos:"
say "  # Training-ADJACENT. Scope ruling required before any absorption — folding another seat's"
say "  # repo into a private training repo is a disconnection risk, not a cleanup."
for n in dgx-spark-multinode staging/gb10-vllm-bringup-kit tinygrad-fork tinygrad-blackwell-fork ai_native/sglang-fork; do
  p="$HOME/$n"; [ -d "$p/.git" ] || continue
  say "  - path: \"\$HOME/$n\""
  say "    tracked_files: $(git -C "$p" ls-files 2>/dev/null | wc -l)"
  say "    branch: $(git -C "$p" rev-parse --abbrev-ref HEAD 2>/dev/null)"
  say "    dirty_entries: $(git -C "$p" status --porcelain 2>/dev/null | wc -l)"
  say "    disposition: UNRULED"
done

# ---------------------------------------------------------------- capture-first
say ""
say "capture_first_candidates:"
say "  # Repos holding LOCAL-ONLY state: commits never pushed, or a wandered checkout. This state"
say "  # exists on one disk and nowhere else, so it is what a consolidation move can destroy."
say "  # Found generically (any repo with unpushed commits) rather than from a hand-kept list,"
say "  # because the one that bit us was a seat's own session root that nobody thought to check."
find "$HOME" -maxdepth 2 -type d -name '.git' 2>/dev/null | sed 's|/.git$||' | sort | while read -r r; do
  def=$(git -C "$r" symbolic-ref --short refs/remotes/origin/HEAD 2>/dev/null | sed 's|origin/||')
  def="${def:-main}"
  ahead=$(git -C "$r" rev-list --count "origin/$def..HEAD" 2>/dev/null) || continue
  [ "${ahead:-0}" -gt 0 ] 2>/dev/null || continue
  say "  - path: \"${r/#$HOME/\$HOME}\""
  say "    branch: $(git -C "$r" rev-parse --abbrev-ref HEAD 2>/dev/null)"
  say "    default_branch: $def"
  say "    unpushed_commits: $ahead"
  say "    tracked_modified: $(git -C "$r" status --porcelain 2>/dev/null | grep -vc '^??')"
  say "    untracked: $(git -C "$r" status --porcelain 2>/dev/null | grep -c '^??')"
  say "    action: CAPTURE (bundle + sha256 manifest) BEFORE any consolidation move"
done

# ---------------------------------------------------------------- wheels
say ""
say "wheels:"
find "$HOME" /media/mira/Expansion -maxdepth 5 -name '*.whl' 2>/dev/null | sort | while read -r w; do
  say "  - path: \"${w/#$HOME/\$HOME}\""
  say "    bytes: $(stat -c%s "$w" 2>/dev/null)"
  say "    sha256: $(sha256sum "$w" 2>/dev/null | cut -d' ' -f1)"
done

# ---------------------------------------------------------------- production surface
say ""
say "production_capabilities:"
say "  # From PRODUCTION_MANIFEST.yml — production is defined by EXECUTION RECEIPT, never by name."
python3 - "$ROOT/PRODUCTION_MANIFEST.yml" <<'PY' 2>/dev/null || say "  # manifest unreadable"
import sys, yaml
d = yaml.safe_load(open(sys.argv[1])) or {}
for n, b in (d.get("capabilities") or {}).items():
    print(f"  - capability: {n}")
    print(f"    status: {b.get('status','?')}")
    print(f"    entrypoint: {b.get('entrypoint','')}")
for n, b in (d.get("historical_lines") or {}).items():
    print(f"  - line: {n}")
    print(f"    status: {str(b.get('status','?')).split(chr(8212))[0].strip()}")
    print(f"    launchable: false")
# CONTESTED entries live in their own top-level list, NOT under capabilities. An earlier version
# of this generator read only capabilities + historical_lines and therefore silently omitted
# sft_27b_fullparam — an inventory that drops the one capability flagged as DISPUTED is worse
# than no inventory, because it reads as complete. Enumerate every place the manifest can hold a
# capability, not just the place you remember.
for c in (d.get("contested") or []):
    print(f"  - capability: {c.get('capability','?')}")
    print(f"    status: CONTESTED — not adjudicated")
    print(f"    launchable: false")
PY

# ---------------------------------------------------------------- node dependency residue
say ""
say "node_dependency_residue:"
say "  # Development checkouts SHADOWING installed packages on training nodes. Found 2026-07-31"
say "  # while lending Spark4 for a scorer A/B; both instances were on ONE node while its peers"
say "  # carried none, so this is node-level accumulation rather than a fleet property."
say "  #"
say "  # WHY IT MATTERS AND WHY A PATH LISTING WILL NOT FIND IT: an editable install writes a .pth"
say "  # that executes at interpreter startup and installs a finder redirecting imports AHEAD of"
say "  # site-packages. Inspecting site-packages contents shows the installed version; the import"
say "  # returns the checkout. The only reliable probe is to RESOLVE THROUGH THE RUNNING"
say "  # INTERPRETER - import it, print __file__, hash that."
i=0
for h in ${SPARK_MGMT_IPS:-}; do
  i=$((i+1))
  out=$(ssh -o BatchMode=yes -o ConnectTimeout=8 "spark@$h" \
    "$(printf '%q ' bash -s -- "${SPARK_HOME%/}")" <<'REMOTE' 2>/dev/null
spark_home=$1
echo "$(find "$spark_home" -path "*/vllm/utils/mem_utils.py" 2>/dev/null | wc -l):$(find "$spark_home" -path "*/vllm/utils/mem_utils.py" -exec sha256sum {} \; 2>/dev/null | awk "{print \$1}" | sort -u | wc -l):$(find "$spark_home" -name "__editable__*.pth" 2>/dev/null | wc -l)"
REMOTE
  )
  say "  - node: $i"
  say "    vllm_mem_utils_copies: ${out%%:*}"
  say "    distinct_contents: $(echo "$out" | cut -d: -f2)"
  say "    editable_pth_overrides: $(echo "$out" | cut -d: -f3)"
done
[ "$i" -eq 0 ] && say "  # SPARK_MGMT_IPS unset - section empty; source fleet.env and re-run."
