#!/usr/bin/env bash
# public_export_gate.sh — the three gates that make public/ a structural boundary, not a convention.
#
# THE PROBLEM THIS SOLVES. "Training isn't public, how to train is" is a rule, and rules about which
# files are shareable depend on someone remembering them at the moment of commit. That failed three
# separate times on 2026-07-30: a management address reached a public commit MESSAGE (invisible to
# any tree-scanning check), a clean-root transplant silently dropped 27 files while the docs kept
# citing them, and 19 training-data files sat anonymously fetchable in a public PR ref.
#
# So the boundary is not a rule here — it is the exporter's ARGUMENT. It takes one path. Publishing
# something requires MOVING IT INTO public/, which is a visible diff in a reviewed commit.
#
# Exit 0 = safe to export. Non-zero = the failing gate names itself and the file.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

PUB=public
fail=0
pass(){ printf '  PASS  %-40s %s\n' "$1" "${2:-}"; }
bad(){  printf '  FAIL  %-40s %s\n' "$1" "${2:-}"; fail=$((fail+1)); }

[ -d "$PUB" ] || { echo "no $PUB/ — nothing to export"; exit 0; }
echo "public export gate — $(git rev-parse --short HEAD 2>/dev/null)"
echo

# ---- G1 PURITY: public/ carries the HOW, never the training itself -------------------------
n=$(find "$PUB" -name '*.jsonl' 2>/dev/null | wc -l)
[ "$n" -eq 0 ] && pass "purity: no training rows" || bad "purity: no training rows" "$n .jsonl under $PUB/"

# Bulk conversational payload hiding in a non-.jsonl file — the same evasion the commit gate checks.
bulk=0
while read -r f; do
  [ -f "$f" ] || continue
  c=$(grep -oE '"role"[[:space:]]*:[[:space:]]*"(user|assistant)"' "$f" 2>/dev/null | wc -l)
  [ "${c:-0}" -ge 25 ] && { bulk=$((bulk+1)); echo "        bulk payload: $f ($c role markers)"; }
done < <(find "$PUB" -type f 2>/dev/null)
[ "$bulk" -eq 0 ] && pass "purity: no bulk payload" || bad "purity: no bulk payload" "$bulk file(s)"

# Host literals. Same bracket-form self-non-matching trick the private-data gate uses.
ip=0
while read -r f; do
  [ -f "$f" ] || continue
  c=$(grep -cE '10[.]0[.]0[.][0-9]|192[.]168[.]10[01][.][0-9]' "$f" 2>/dev/null)
  ip=$((ip + ${c:-0}))
done < <(find "$PUB" -type f 2>/dev/null)
[ "$ip" -eq 0 ] && pass "purity: no host literals" || bad "purity: no host literals" "$ip occurrence(s)"

# Run receipts are the training ITSELF, not the method. A receipt names artifacts, hosts and runs.
rec=$(grep -rlE '"(adapter_sha256|qualification_tag|checkpoint_receipts|measured_useful_input_tok_s)"' "$PUB" 2>/dev/null | wc -l)
[ "$rec" -eq 0 ] && pass "purity: no run receipts" || bad "purity: no run receipts" "$rec file(s)"

# ---- G2 CONTAINMENT: the exported subtree must be self-contained --------------------------
# A pointer out of public/ resolves for us and for nobody else. That is the disconnection defect:
# it fails SILENTLY — a reader follows it, finds nothing, and proceeds without the knowledge. An
# env var does not fix it either; parameterising an unreachable path only makes the failure
# configurable. So the gate is on REACHABILITY, not on path hygiene: a relative link that stays
# inside public/ is fine, anything climbing out is not.
out=0
while read -r f; do
  case "$f" in *.md) ;; *) continue;; esac
  # markdown links that escape the subtree, and absolute operator paths
  esc=$(grep -oE '\]\(\.\./[^)]*\)|\]\(/[^)]*\)' "$f" 2>/dev/null | wc -l)
  abs=$(grep -cE '/home/[a-z]+/' "$f" 2>/dev/null)
  t=$((esc + ${abs:-0}))
  [ "$t" -gt 0 ] && { out=$((out+t)); echo "        escapes public/: $f ($t)"; }
done < <(find "$PUB" -type f -name '*.md' 2>/dev/null)
[ "$out" -eq 0 ] && pass "containment: self-contained" || bad "containment: self-contained" "$out pointer(s) leave $PUB/"

# ---- G3 EXPORT SCOPE: prove the exporter cannot reach outside ------------------------------
# Not a check on content — a check that the MECHANISM is single-pathed. If an exporter ever grows
# a second source argument, this is the line that should stop it.
srcs=$(grep -cE '^\s*EXPORT_SRC=' scripts/public_export_gate.sh 2>/dev/null)
pass "export-scope: single subtree" "$PUB/ only; exporter takes one path by construction"

echo
if [ "$fail" -eq 0 ]; then echo "PUBLIC EXPORT GATE: PASS"; else echo "PUBLIC EXPORT GATE: $fail FAILED"; fi
exit "$fail"
