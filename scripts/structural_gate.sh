#!/usr/bin/env bash
# structural_gate.sh — the consolidation's structural checks, as a gate.
#
# Per the execute task: "Structural checks become gates: no script references a path outside the
# repo; no secrets/PII; data present only as manifests; full-history scan; hosts env-configurable
# fail-loud with committed .example."
#
# WHAT "OUTSIDE THE REPO" MEANS, precisely. The target is RESOLVABILITY, not path-scrubbing
# (conductor's clarification, 2026-07-30 — a previous over-scrub of /home/<user> broke 52 scripts).
# A path is a violation when it CANNOT RESOLVE for whoever runs this repo:
#   - a hardcoded operator home that is not env-derived
#   - a pointer into a PRIVATE repo that does not ship
# A path that resolves to content shipped with the repo is FINE and stays untouched.
#
# Exit 0 = all gates pass. Non-zero = at least one gate failed; the failing gate names itself.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

fail=0
pass(){ printf '  PASS  %-46s %s\n' "$1" "${2:-}"; }
bad(){  printf '  FAIL  %-46s %s\n' "$1" "${2:-}"; fail=$((fail+1)); }

echo "structural gate — $(git rev-parse --short HEAD 2>/dev/null)"
echo

# ---- G1: no script points into a private repo or a non-env operator path -------------------
# Private repos on this machine that do NOT ship: a pointer into one silently resolves here and
# nowhere else, which is the disconnection defect class (found in taeys-hands the same day).
PRIVATE_REPOS='treasurer|the-conductor|data/corpus'
hits=$(grep -rnE "/home/[a-z]+/(${PRIVATE_REPOS})/" \
        --include='*.sh' --include='*.py' --include='*.yml' . 2>/dev/null \
        | grep -v '^./.git' | grep -vE '^\./(docs|careers-qwen)/[A-Z_]+\.md' | wc -l)
if [ "$hits" -eq 0 ]; then pass "G1 no pointer into a private repo"
else bad "G1 no pointer into a private repo" "$hits reference(s) — these fail silently for anyone but us"
     grep -rnE "/home/[a-z]+/(${PRIVATE_REPOS})/" --include='*.sh' --include='*.py' --include='*.yml' . 2>/dev/null \
       | grep -v '^./.git' | head -5 | sed 's|^|        |'
fi

# ---- G2: hosts are env-configurable, and the template is committed -------------------------
# SCOPE IS TRACKED FILES, and the scoping is the whole point. The publication bar is "what can
# reach a consumer", and a gitignored or untracked file structurally cannot. The first version of
# this gate scanned the working tree and FAILED on careers-qwen/bake_module3.sh — a file
# .gitignore:31 excludes precisely because it holds addresses. A gate that fails on something that
# cannot leak is a gate that gets waived, and a waived gate protects nothing. Local clutter is
# still worth knowing about, so it is REPORTED below rather than silently dropped.
ip=0
while read -r f; do
  case "$f" in *.sh|*.py|*.yml) ;; *) continue;; esac
  c=$(grep -cE '10[.]0[.]0[.][0-9]|192[.]168[.]10[01][.][0-9]' "$f" 2>/dev/null)
  ip=$((ip + ${c:-0}))
done < <(git ls-files)
[ "$ip" -eq 0 ] && pass "G2a no hardcoded hosts (tracked)" || bad "G2a no hardcoded hosts (tracked)" "$ip literal address(es) WOULD PUBLISH"
# Advisory: gitignored/untracked clutter carrying topology. Never fails the gate — cannot publish.
clutter=$(git ls-files --others --exclude-standard --ignored 2>/dev/null | grep -E '\.(sh|py)$' | while read -r f; do
  c=$(grep -cE '10[.]0[.]0[.][0-9]|192[.]168[.]10[01][.][0-9]' "$f" 2>/dev/null); [ "${c:-0}" -gt 0 ] && echo "$f"; done | wc -l)
[ "${clutter:-0}" -gt 0 ] && printf '  note  %-46s %s\n' "untracked/ignored scripts with addresses" "$clutter — local clutter, cannot publish"
[ -f fleet.env.example ] && pass "G2b fleet.env.example committed" \
  || bad "G2b fleet.env.example committed" "missing — a consumer cannot know what to set"
git check-ignore -q fleet.env 2>/dev/null && pass "G2c fleet.env is gitignored" \
  || bad "G2c fleet.env is gitignored" "the real values would be committed"

# ---- G3: data present only as manifests ----------------------------------------------------
# Tracked, not merely present: gitignored working files are fine, committed rows are not.
rows=$(git ls-files | grep -c '\.jsonl$')
[ "$rows" -eq 0 ] && pass "G3a no training rows tracked" || bad "G3a no training rows tracked" "$rows .jsonl tracked"
# Bulk role-marker payloads hiding in non-.jsonl files.
bulk=0
while read -r f; do
  case "$f" in *.md|*.json) n=$(grep -oE '"role"[[:space:]]*:[[:space:]]*"(user|assistant)"' "$f" 2>/dev/null | wc -l)
    [ "${n:-0}" -ge 25 ] && { bulk=$((bulk+1)); echo "        bulk payload: $f ($n role markers)"; };; esac
done < <(git ls-files)
[ "$bulk" -eq 0 ] && pass "G3b no bulk training payload" || bad "G3b no bulk training payload" "$bulk file(s)"

# ---- G4: the private-data gate itself passes over the whole tree ---------------------------
if bash scripts/check_no_private_data.sh >/dev/null 2>&1; then pass "G4 private-data gate (tree)"
else bad "G4 private-data gate (tree)" "run scripts/check_no_private_data.sh for detail"; fi

# ---- G5: full-history scan ------------------------------------------------------------------
# HEAD-only under-reports: on this repo it covered 407 of 1396 commits. --all is the real bar.
# COUNT-BASED ALLOWANCE WAS WRONG AND IS REPLACED. The first version passed when findings were
# "<= 3", which cannot distinguish THE SAME three characterised false positives from three
# ENTIRELY DIFFERENT findings — and would have silently absorbed a real leak the moment an old
# FP was removed from history. Findings are now matched by FINGERPRINT (file + rule), so every
# known FP must be individually characterised and ANY unrecognised finding fails regardless of
# count. Caught when a legitimate new commit pushed the count to 4 and the gate failed for the
# right outcome but the wrong reason.
if command -v gitleaks >/dev/null 2>&1; then
  glout=$(mktemp)
  gitleaks detect --source . --config .gitleaks.toml --redact --log-opts=--all \
    --no-banner --report-format json --report-path "$glout" >/dev/null 2>&1
  unknown=$(python3 - "$glout" <<'PYGL'
import json, sys
# Each entry: (path, rule) characterised as a NON-secret, with the reason it is not one.
KNOWN = {
 ("careers-qwen/run_stage2_sft_ddp_till_done.sh", "generic-api-key"),   # sha256 of the qualification receipt — a content digest, the receipt identity itself
 ("dense-9b/instrumentation/README.md",           "generic-api-key"),   # sha256 tokenizer-manifest digest quoted in prose
 ("datasets/current/moe-35b/phase1_infra_v2_gated.jsonl", "generic-api-key"),  # 25-char config assignment after the literal word "key parameters"
 ("datasets/current/moe-35b/combined_v1_gated.jsonl",     "generic-api-key"),  # same assignment, same corpus
}
try:
    rows = json.load(open(sys.argv[1]))
except Exception:
    print("SCAN_UNREADABLE"); raise SystemExit
out = [f"{r.get('File')}::{r.get('RuleID')}" for r in rows
       if (r.get("File"), r.get("RuleID")) not in KNOWN]
print("\n".join(out))
PYGL
)
  rm -f "$glout"
  if [ "$unknown" = "SCAN_UNREADABLE" ]; then
    bad "G5 full-history scan (--all)" "gitleaks report unreadable — CANNOT VERIFY, which blocks"
  elif [ -z "$unknown" ]; then
    pass "G5 full-history scan (--all)" "only characterised non-secret digests"
  else
    bad "G5 full-history scan (--all)" "UNCHARACTERISED finding(s):"
    printf '        %s\n' $unknown
  fi
else bad "G5 full-history scan (--all)" "gitleaks not installed — CANNOT VERIFY, which blocks"; fi

# ---- G6: the sealed SFT lane's pinned files still resolve ------------------------------------
# A restructure that moves any of these breaks the runner the pending 0->50 authorization would
# promote. This gate exists so a move cannot silently invalidate it.
miss=0
while read -r f; do [ -n "$f" ] && { [ -f "$f" ] || { miss=$((miss+1)); echo "        missing: $f"; }; }; done < <(
  sed -n '/IMMUTABLE_FILES=(/,/^)/p' careers-qwen/run_stage2_sft_ddp_till_done.sh 2>/dev/null \
    | grep -oE '^[[:space:]]+[a-z0-9-]+/[a-zA-Z0-9_/.-]+' | tr -d ' ')
[ "$miss" -eq 0 ] && pass "G6 sealed SFT pinned files resolve" || bad "G6 sealed SFT pinned files resolve" "$miss missing"

echo
if [ "$fail" -eq 0 ]; then echo "STRUCTURAL GATE: PASS"; else echo "STRUCTURAL GATE: $fail FAILED"; fi
exit "$fail"
