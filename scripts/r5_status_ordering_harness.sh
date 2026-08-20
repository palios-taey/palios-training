#!/usr/bin/env bash
# Harness for r5-audit-gate status ordering jq (whole-sequence fail-closed).
# Mirrors the jq program in .github/workflows/r5-audit-gate.yml.
set -euo pipefail

JQ_PROG='
  if type != "array" then
    error("statuses root is not a JSON array")
  else . end
  | if any(.[];
      (.id | type) != "number"
      or (.context | type) != "string"
      or (.state | type) != "string"
      or (.created_at | type) != "string"
    ) then
      error("malformed status object (id/context/state/created_at)")
    else . end
  | group_by(.context)
  | map(
      sort_by(.id)
      | . as $s
      | if any(
          range(0; ($s | length) - 1);
          $s[.].created_at > $s[.+1].created_at
        ) then
          error("ambiguous ordering for context \($s[0].context): id/created_at inversion across sequence")
        else $s end
      | reverse
      | {context: .[0].context, state: .[0].state, id: .[0].id}
    )
'

run_case() {
  local name="$1" expect="$2" input="$3"
  local out rc=0
  set +e
  out="$(printf '%s' "$input" | jq -s -e -c "$JQ_PROG" 2>/tmp/r5-harness.err)"
  rc=$?
  set -e
  if [ "$expect" = "fail" ]; then
    if [ "$rc" -eq 0 ]; then
      echo "FAIL $name: expected nonzero, got: $out"
      return 1
    fi
    echo "PASS $name (nonzero rc=$rc)"
  else
    if [ "$rc" -ne 0 ]; then
      echo "FAIL $name: expected success, stderr=$(cat /tmp/r5-harness.err)"
      return 1
    fi
    local state
    state="$(printf '%s' "$out" | jq -r '([.[] | select(.context=="audit/grok")][0].state) // "missing"')"
    if [ "$state" != "$expect" ]; then
      echo "FAIL $name: expected audit/grok=$expect got $state out=$out"
      return 1
    fi
    echo "PASS $name (audit/grok=$state)"
  fi
}

# Case 1: id30 success 08:03, id20 success 08:02, id10 failure 08:04 => nonzero
# (lower id has newer created_at than a higher id — sequence inversion)
run_case "mid_sequence_conflict" fail "$(cat <<'JSON'
{"id":30,"context":"audit/grok","state":"success","created_at":"2026-08-20T08:03:00Z"}
{"id":20,"context":"audit/grok","state":"success","created_at":"2026-08-20T08:02:00Z"}
{"id":10,"context":"audit/grok","state":"failure","created_at":"2026-08-20T08:04:00Z"}
JSON
)"

# Case 2: top-two inversion => nonzero
run_case "top_two_inversion" fail "$(cat <<'JSON'
{"id":30,"context":"audit/grok","state":"success","created_at":"2026-08-20T08:01:00Z"}
{"id":20,"context":"audit/grok","state":"failure","created_at":"2026-08-20T08:02:00Z"}
JSON
)"

# Case 3: monotonic valid => latest by id
run_case "monotonic_latest" success "$(cat <<'JSON'
{"id":10,"context":"audit/grok","state":"failure","created_at":"2026-08-20T08:01:00Z"}
{"id":20,"context":"audit/grok","state":"pending","created_at":"2026-08-20T08:02:00Z"}
{"id":30,"context":"audit/grok","state":"success","created_at":"2026-08-20T08:03:00Z"}
{"id":11,"context":"audit/gatekeeper","state":"success","created_at":"2026-08-20T08:01:30Z"}
{"id":21,"context":"audit/gatekeeper","state":"success","created_at":"2026-08-20T08:02:30Z"}
JSON
)"

# Case 4: 101+ status objects across one context + noise (pagination stand-in)
# Build 101 monotonic successes; latest id=101 must win.
page_fixture() {
  local i
  for i in $(seq 1 101); do
    printf '{"id":%d,"context":"audit/grok","state":"success","created_at":"2026-08-20T08:%02d:%02dZ"}\n' \
      "$i" "$((i / 60))" "$((i % 60))"
  done
  printf '{"id":1,"context":"audit/gatekeeper","state":"success","created_at":"2026-08-20T07:00:00Z"}\n'
}
run_case "page_101_latest" success "$(page_fixture)"

echo "ALL HARNESS CASES PASS"
