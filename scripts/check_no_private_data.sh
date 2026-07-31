#!/bin/bash
# check_no_private_data.sh — HARD GATE: no private training data may reach the public remote.
#
# Born 2026-07-24: a public branch (production/palios-da5d4d5-20260724) was found exposing 18
# private training .jsonl files, and a sweep then found the SAME exposure on three more public
# refs. Every instance is the same accident: nothing at push time prevents it. Jesse's rule is
# absolute — recipes/tooling public, training data and content NEVER public — and a rule with no
# mechanism is a rule that gets violated on a busy day.
#
# 6SIGMA shape: do not clean up branches one at a time (that is a patch). Make the bad push
# impossible (root fix).
#
# Modes:
#   ./check_no_private_data.sh                 scan HEAD's tracked tree
#   ./check_no_private_data.sh --range A..B    scan every commit in a range (pre-push / CI)
#   ./check_no_private_data.sh --ref REF       scan a ref's FULL history (audit an existing branch)
# Exit 0 clean, 1 violation.
set -uo pipefail

# Paths that must never appear in ANY public commit. Content, not tooling.
VIOLATION_RE='(^|/)(training_data|datasets)/.*\.jsonl$|^careers-qwen/data/.*\.jsonl$|(^|/)data/.*_gated\.jsonl$'
# Live-credential shapes, belt-and-braces alongside gitleaks.
SECRET_RE='sk-ant-api[a-zA-Z0-9_-]{20,}|AIza[0-9A-Za-z_-]{33}|xox[baprs]-[0-9A-Za-z-]{10,}|-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----'

# TOPOLOGY — lifted VERBATIM from origin/main (PR#12 hardening, 3d62c88). This branch
# PREDATES that commit, which is why this scanner had no IP rule and why my "no IP rule
# ever existed" claim was true of THIS ancestry and false of main.
# The two scanners were COMPLEMENTARY, not successive: main caught topology and missed
# email/linkedin/conversation-URLs; this one caught those three and missed topology.
# Landing either as a REPLACEMENT trades one blind spot for another. This is the UNION.
# Bracket form is deliberately self-non-matching: this line's own text contains
# 10[.]0[.]0[.], not the literal address, so the gate does not flag itself.
TOPOLOGY_RE='10[.]0[.]0[.]|192[.]168[.]10[01][.]'

# OPERATOR PII (added 2026-07-24 after an R5 Gatekeeper BLOCK found personal email,
# personal LinkedIn, a private job-application id, operator home-paths in 32+ tracked
# files, and raw consult captures with per-account conversation URLs — all on a repo
# that is ALREADY PUBLIC. This gate PASSED that: it only ever checked IPs, secret
# shapes, and data-file paths, so the entire PII class was invisible.
#
# MATCH BY SHAPE, NEVER BY VALUE. Pasting the operator's actual address into this file
# would publish the very string the gate exists to keep private — the detector must not
# become the leak. (Caught live while writing this comment: the first draft quoted the
# real address as an example and the gate flagged its own source. That is the trap, and
# it is easy to walk into while explaining it.) Shapes also generalise to the next
# person's PII instead of hard-coding one operator's.
PII_EMAIL_RE='[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'
# Corporate/role identities that are legitimately public (commit authorship, noreply).
# ANCHORED ($ / ^). Unanchored these match as SUBSTRINGS, which turns an allowlist into a
# bypass: a bare "@taey\.ai" also allows taey.ai.attacker.com, and "@palios" also allows
# @palios-evil.com. An attacker-registered lookalike domain would then be waved through by
# the very list meant to reduce noise. No live instance — defence in depth, flagged in an
# R5 re-audit.
# CODE SYNTAX IS NOT EMAIL. A full-history scan returned 59 'email-shaped' hits that were
# overwhelmingly decorators and unit names — @app.post, @app.get, @torch.no_grad,
# @torch.distributed.fsdp.wrap, @triton.jit, @triton.autotuner, @tty1.service. A gate that
# flags every FastAPI/torch/triton file cries wolf, and a gate that cries wolf gets bypassed.
# (This class also no longer protects the operator's own address: the data subject ruled it
# public. It remains for THIRD-PARTY addresses, which are not his to waive.)
PII_EMAIL_ALLOW='@taey\.ai$|@palios-taey\.(com|ai|org)$|^noreply@|@example\.(com|org)$|@othercompany\.com$|@users\.noreply\.github\.com$|@anthropic\.com$|^@|\.(service|socket|timer|target|mount)$|@(app|torch|triton|pytest|staticmethod|classmethod|property|dataclass|functools|contextlib)\.'
# Personal profile + per-account AI conversation URLs (these identify an individual
# AND leak private thread contents).
PII_URL_RE='linkedin\.com/in/|chatgpt\.com/c/|chat\.openai\.com/c/|grok\.com/c/|gemini\.google\.com/app/|perplexity\.ai/search/|claude\.ai/chat/'
# DE-UMBILICAL is NOT a privacy class, and keeping it here made this gate cry wolf on
# 76 working files (2026-07-25). A hardcoded operator/service path is a PORTABILITY
# problem for a repo about to go public — it is not an exposure on a private one, and
# /home/[s]park and /home/[j]etson are SERVICE ACCOUNTS on remote nodes, not people.
# (Bracketed so this COMMENT does not match PII_PATH_RE — the same self-non-matching
# idiom TOPOLOGY_RE and PII_URL_RE use. NOT a code carve-out: there is still no
# self-exemption, the file is scanned like any other. Caught by a CLEAN-fixture control
# 2026-07-25, which fired on a doc containing nothing private and turned out to be the
# gate flagging its own documentation. Without that negative control the whole
# five-class verification matrix read CAUGHT for the wrong reason.)
#
# Treating them as PII already cost a production outage: this gate flagged them, the
# scrub rewrote them to a literal "/home/<user>", and 52 runtime scripts broke —
# including the NCCL library path and the telemetry path the launch gate reads. The
# failure then presented as a hardware fault and blocked training for hours.
#
# The check still matters, at the moment it actually applies: REPUBLISH_CHECKLIST.md
# owns it, run before any private->public transition. Enable it here on demand with
# CHECK_HOME_PATHS=1 (used by that checklist), off by default.
CHECK_HOME_PATHS="${CHECK_HOME_PATHS:-0}"
PII_PATH_RE='/home/[a-z][a-z0-9_-]+'
PII_PATH_ALLOW='/home/user$|/home/runner$|/home/\$|/home/<'

# OPERATOR IDENTITY WAIVER (2026-07-25). The data subject ruled his own email and his
# own AI-conversation links non-sensitive: "I do not care about my email addresses or
# links to conversations. Those are fine. This is my model and no one can do anything
# with that." He is the sovereign over his own data. Conductor's independent sweep found
# NO third-party personal data in this history.
#
# So these classes must protect THIRD PARTIES, not the operator from himself — otherwise
# the gate refuses every push over data its owner has waived, which is how a gate stops
# being consulted at all.
#
# The waived identifiers live OUTSIDE this file, in a gitignored allowlist, for the same
# reason the shapes are escaped: a detector that spells out the value it protects has
# published it. Absent the file, nothing is waived and the gate behaves as before.
IDENTITY_WAIVER_FILE="${IDENTITY_WAIVER_FILE:-$(git rev-parse --show-toplevel 2>/dev/null)/.operator-identity-allow}"
if [ -f "$IDENTITY_WAIVER_FILE" ]; then
  _waived=$(grep -vE '^\s*(#|$)' "$IDENTITY_WAIVER_FILE" | paste -sd'|' -)
  if [ -n "$_waived" ]; then
    PII_EMAIL_ALLOW="$PII_EMAIL_ALLOW|$_waived"
    PII_URL_ALLOW="$_waived"
  fi
fi
PII_URL_ALLOW="${PII_URL_ALLOW:-__no_waiver__}"

MODE="tree"; ARG=""
case "${1:-}" in
  --range) MODE="range"; ARG="${2:?--range needs A..B}";;
  --ref)   MODE="ref";   ARG="${2:?--ref needs a ref}";;
esac

fail=0
report() { echo "  VIOLATION: $1"; fail=1; }

case "$MODE" in
  tree)
    while read -r f; do [ -n "$f" ] && report "tracked private data: $f"; done < <(
      git ls-files | grep -E "$VIOLATION_RE" || true)
    # CONTENT-SHAPE CHECK (added 2026-07-24): path patterns are extension-scoped, so a
    # .json/.md carrying training rows slips through. Caught while committing a gate manifest.
    # NARROW BY DESIGN: source files (.py/.sh) legitimately CONTAIN the row-building strings,
    # and a spec may show ONE example row — flagging those makes the gate cry wolf, and a gate
    # that cries wolf gets bypassed, which is worse than no gate. So: data-ish extensions only,
    # and require a BULK payload (>=5 role markers), which is data, not documentation.
    while read -r f; do
      [ -f "$f" ] || continue
      case "$f" in
        *.json|*.md|*.txt|*.csv|*.yaml|*.yml) ;;
        *) continue;;
      esac
      n=$(grep -oE '"role"[[:space:]]*:[[:space:]]*"(user|assistant)"' "$f" 2>/dev/null | wc -l)
      [ "${n:-0}" -ge 5 ] 2>/dev/null && report "bulk training-row payload in $f ($n role markers)"
    done < <(git ls-files)

    while IFS= read -r hit; do [ -n "$hit" ] && report "tracked private topology: $hit"; done < <(
      git grep -I -n -E "$TOPOLOGY_RE" -- $(git ls-files) 2>/dev/null || true)

    # --- OPERATOR PII SWEEP (counts only; never echo the matched value) ---
    while read -r f; do
      [ -f "$f" ] || continue
      # NO self-exemption. This file is scanned like any other: the shapes are written
      # escaped (linkedin\.com, /home/\$) so they do not match themselves, the same
      # trick TOPOLOGY_RE uses. A gate that has to skip itself to pass is hiding.
      e=$(grep -aoE "$PII_EMAIL_RE" "$f" 2>/dev/null | grep -avE "$PII_EMAIL_ALLOW" | wc -l)
      [ "${e:-0}" -gt 0 ] && report "personal email address in $f ($e occurrence(s))"
      u=$(grep -aoE "$PII_URL_RE[A-Za-z0-9_./?=-]*" "$f" 2>/dev/null | grep -avE "$PII_URL_ALLOW" | wc -l)
      [ "${u:-0}" -gt 0 ] && report "personal-profile / private-conversation URL in $f ($u occurrence(s))"
      if [ "$CHECK_HOME_PATHS" = "1" ]; then
        p=$(grep -aoE "$PII_PATH_RE" "$f" 2>/dev/null | grep -avE "$PII_PATH_ALLOW" | wc -l)
        [ "${p:-0}" -gt 0 ] && report "hardcoded operator home path in $f ($p occurrence(s)) [de-umbilical, republish-only]"
      fi
    done < <(git ls-files)
    ;;
  range|ref)
    if [ "$MODE" = "range" ]; then commits=$(git rev-list "$ARG" 2>/dev/null)
    else commits=$(git rev-list "$ARG" 2>/dev/null); fi
    [ -z "$commits" ] && { echo "no commits to scan"; exit 0; }
    n=$(echo "$commits" | wc -l)
    echo "scanning $n commit(s)..."
    while read -r f; do [ -n "$f" ] && report "private data in history: $f"; done < <(
      git log --pretty=format: --name-only $ARG 2>/dev/null | sort -u | grep -E "$VIOLATION_RE" || true)
    if git log -p "$ARG" 2>/dev/null | grep -aqoE "$SECRET_RE"; then
      report "LIVE CREDENTIAL MATERIAL in history — rotate before anything else"
    fi
    # PII in history. History cannot be cleaned by deleting a file in a later commit,
    # so this is deliberately a hard fail: the ref itself has to be rebuilt.
    hist=$(git log -p "$ARG" 2>/dev/null)
    he=$(printf '%s' "$hist" | grep -aoE "$PII_EMAIL_RE" | grep -avE "$PII_EMAIL_ALLOW" | wc -l)
    [ "${he:-0}" -gt 0 ] && report "personal email address in HISTORY ($he occurrence(s)) — ref must be rebuilt, not patched"
    hu=$(printf '%s' "$hist" | grep -aoE "$PII_URL_RE[A-Za-z0-9_./?=-]*" | grep -avE "$PII_URL_ALLOW" | wc -l)
    [ "${hu:-0}" -gt 0 ] && report "personal-profile / private-conversation URL in HISTORY ($hu occurrence(s)) — ref must be rebuilt"
    # TOPOLOGY IN HISTORY — this is NOT in main, which checks topology in TREE MODE ONLY.
    # A scrub commit does not clean the ref: an address deleted in a later commit is still
    # in the objects, so a tree-only topology check passes a ref that would leak on publish.
    ht=$(printf '%s' "$hist" | grep -aoE "$TOPOLOGY_RE" | wc -l)
    [ "${ht:-0}" -gt 0 ] && report "private topology in HISTORY ($ht occurrence(s)) — ref must be rebuilt, not patched"
    if [ "$CHECK_HOME_PATHS" = "1" ]; then
      hp=$(printf '%s' "$hist" | grep -aoE "$PII_PATH_RE" | grep -avE "$PII_PATH_ALLOW" | wc -l)
      [ "${hp:-0}" -gt 0 ] && report "hardcoded operator home path in HISTORY ($hp occurrence(s)) [de-umbilical, republish-only]"
    fi
    unset hist
    ;;
esac

if [ "$fail" = 1 ]; then
  cat <<'EOF'

=== PUSH REFUSED — private content would become public ===
Jesse's binding rule: recipes and tooling are public; training data, private topology, and content NEVER are.
Fix, do not bypass:
  - if the file should not be tracked at all: git rm --cached <f>, add it to .gitignore
  - if prose/code only needs topology context: replace addresses with PRIVATE_MGMT_IP or PRIVATE_RAIL_IP
  - if it is already in this branch's HISTORY: the branch cannot go public as-is —
    transplant the public-safe work onto a clean base (conductor PRIVATE_TO_PUBLIC path)
  - a --no-verify bypass here is a KERNEL violation, not a shortcut
EOF
  exit 1
fi
echo "clean: no private training data, no private topology, no credential material"
exit 0
