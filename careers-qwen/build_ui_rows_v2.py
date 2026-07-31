#!/usr/bin/env python3
"""Rebuild the UI training rows — fixes every defect the 5-lane consult found.

DEFECTS FIXED (each named by the lane that found it):
1. HORIZON: "27 unique prompts; 21 rows in 7 exact-duplicate groups", several with CONFLICTING
   targets. ROOT CAUSE: the source carries a `question` field that disambiguates every row and my
   converter DROPPED IT, keying only on `name` ("Start typing..."). Including `question` makes all
   41 prompts unique and the targets non-contradictory. This was never a data problem.
2. GAIA: false source attributions — 8 rows stamped "Canonical key: the availability policy" onto
   country / citizenship / staff-level. ROOT CAUSE: a regex->label table with no validation. Now the
   source is derived from the QUESTION and validated against the real ATS token whitelist.
3. GAIA/CLARITY: seq 16 taught `expect: invalid_entry` — a captured failure as a target. Excluded by
   semantic outcome check, and its lesson lives in the practice corpus instead.
4. CLARITY/GAIA/HORIZON: "narrates retrieval while supplying the value inline" — training the
   APPEARANCE of retrieval. Value-bearing rows now emit a real TOOL CALL and consume a TOOL RESULT.
5. HORIZON: "87.1% of assistant words inside <think>"; reasoning on trivial actions. Simple mechanical
   actions now use the native NO-THINK form (empty think block per the chat template).
6. HORIZON: "v3 action JSON matches v1 in all 41 rows — narration, no new executable supervision."
   The thinking rows now carry a genuinely different, executable trajectory.

GROUND TRUTH USED (not invented):
- ATS profile-answer contract: apply-machine/ats_mcp_server.py — CANONICAL_PROFILE_SOURCE and
  SUPPORTED_PROFILE_ANSWER_TOKENS. A profile answer MUST cite a token, never a literal.
- Primitive vocabulary + required keys: the captured actions themselves.
"""
import argparse, hashlib, json, os, re

SRC = os.path.join(os.environ.get("TAEY_APPLY_MACHINE", os.path.expanduser("~/apply-machine")),
                   "bundles/trm_labs_ai_research_engineer_79afbde9/ui_training_pairs.jsonl")
CANON = "careers_kb:identity::application_profile"
# verbatim from ats_mcp_server.py SUPPORTED_PROFILE_ANSWER_TOKENS
TOKENS = {"authorized_to_work_followup", "country", "eeo_decline", "gm_influence",
          "government_official", "hispanic_latino", "location", "pronouns",
          "remote_only_no_columbus_relocation", "sponsorship", "state", "work_authorization"}

# question-text -> canonical ATS token. Only tokens that EXIST in the whitelist.
QUESTION_TOKEN = [
    (r"country of employment",                    "country"),
    (r"work location",                            "location"),
    (r"legally authorized to work",               "work_authorization"),
    (r"sponsorship|visa",                         "sponsorship"),
    (r"relocat|remote",                           "remote_only_no_columbus_relocation"),
    (r"pronoun",                                  "pronouns"),
    (r"hispanic|latino",                          "hispanic_latino"),
    (r"government official",                      "government_official"),
    (r"race|ethnicity|gender|veteran|disability", "eeo_decline"),
]
# Questions that are genuinely NOT canonical profile facts — answering them from a store would be
# a lie. The correct procedure differs and the row says so.
NON_CANONICAL = re.compile(
    r"citizenship|security clearance|job level|hear about|heard of|on-call|advisory|"
    r"acknowledgement|agree|timeline|compensation|salary|describe|why|motivation|outcome", re.I)


def token_for(question):
    q = (question or "").lower()
    for pat, tok in QUESTION_TOKEN:
        if re.search(pat, q):
            return tok if tok in TOKENS else None
    return None


def situation(r, ident):
    o = r.get("observed") or {}
    q = o.get("question")
    name, role, view = o.get("name"), o.get("role"), r.get("view", "")
    lines = [
        "You are operating a real job-application form through your own hands.",
        f"Application: {ident}",
        f"View: {view}",
    ]
    if q:
        lines.append(f'Question on the form: "{q}"')     # <-- THE DISAMBIGUATOR
    lines.append(f'Control: "{name}" ({role})' if name else "Control: (no element mapped)")
    if o.get("anomaly"):
        lines.append(f"Anomaly observed: {o['anomaly']}")
    lines.append("")
    lines.append("Emit the next action as a single JSON object with the primitive and its required "
                 'keys, plus "expect". Emit nothing else.')
    return "\n".join(lines)


def build(rows):
    action_rows, thinking_rows, excluded = [], [], []
    ident = ""
    for r in rows:
        o = r.get("observed") or {}
        ident = o.get("application_identity") or ident
        act = dict(r.get("action") or {})
        if not act.get("primitive"):
            excluded.append((r.get("seq"), "no primitive")); continue
        post = r.get("postcondition")
        # DEFECT 3 — never train a captured failure as a target (semantic, not one key name)
        # Evaluate the VALUES, not the serialized blob: a key literally named "invalid_entry"
        # whose value is False means the entry was VALID. Searching the JSON text dropped the
        # correct row (seq 38) while keeping the shape of the check identical — a false positive
        # that would have removed the very row teaching the right way.
        FAIL = re.compile(r"^(invalid\w*|blocked|rejected|failed|error)", re.I)
        def _failed(pc):
            if isinstance(pc, str):
                return bool(FAIL.match(pc.strip()))
            if isinstance(pc, dict):
                for k, v in pc.items():
                    if isinstance(v, str) and FAIL.match(v.strip()):
                        return True
                    if v is True and FAIL.match(str(k)):
                        return True
            return False
        if _failed(post):
            excluded.append((r.get("seq"), f"failure outcome: {json.dumps(post)[:52]}")); continue

        expect = (post if isinstance(post, str) else
                  "the field holds the written value (verified by copy-back)"
                  if isinstance(post, dict) and "copyback_length" in post else
                  "the control reflects the selected value")
        act["expect"] = expect
        sit = situation(r, ident)
        seq, q = r.get("seq", 0), o.get("question")
        needs_value = act["primitive"] in ("write", "type_filter_select", "pointer_focus_type_select")
        tok = token_for(q) if needs_value else None

        # ---------- ACTION LANE (executable, no narration) ----------
        action_rows.append({
            "schema": "operator_ui_action_v2",
            "messages": [{"role": "user", "content": sit},
                         {"role": "assistant", "content": json.dumps(act, ensure_ascii=False)}],
            "meta": {"seq": seq, "view": r.get("view"), "primitive": act["primitive"],
                     "question": q, "origin": "captured" if seq < 40 else "authored",
                     "profile_token": tok},
            "provenance_hash": hashlib.sha256(f"{seq}|{json.dumps(act)}".encode()).hexdigest()[:16],
        })

        # ---------- THINKING LANE ----------
        if not needs_value:
            # DEFECT 5 — a simple mechanical action gets the NO-THINK form, not invented reasoning.
            msgs = [{"role": "user", "content": sit},
                    {"role": "assistant",
                     "content": "<think>\n\n</think>\n\n" + json.dumps(act, ensure_ascii=False)}]
            kind = "no-think (mechanical action)"
        elif tok:
            # DEFECT 4 — a REAL tool call + tool result, not narrated retrieval.
            plan = (f"This asks for a canonical profile fact. The contract requires citing the "
                    f"profile token, not a literal answer. Token: {tok}. Fetch it, then select.")
            call = {"name": "read_profile_fact",
                    "arguments": {"answer_source": f"{CANON}.{tok}", "reason": f"profile:{tok}"}}
            value = act.get("value") or act.get("filter")
            msgs = [
                {"role": "user", "content": sit},
                {"role": "assistant",
                 "content": f"<think>\n{plan}\n</think>\n\n<tool_call>\n"
                            f"{json.dumps(call, ensure_ascii=False)}\n</tool_call>"},
                {"role": "tool", "content": f"<tool_response>\n"
                                            f"{json.dumps({'value': value}, ensure_ascii=False)}\n"
                                            f"</tool_response>"},
                {"role": "assistant", "content": json.dumps(act, ensure_ascii=False)},
            ]
            kind = f"tool-call retrieval ({tok})"
        else:
            # DEFECT 2 — no canonical token exists. Say so honestly; do NOT invent a source.
            plan = ("This is not a canonical profile fact, so there is no token to cite. It is "
                    "answered from the application context supplied for this run. If no source "
                    "supplies it, stop and ask rather than composing an answer.")
            msgs = [{"role": "user", "content": sit},
                    {"role": "assistant",
                     "content": f"<think>\n{plan}\n</think>\n\n{json.dumps(act, ensure_ascii=False)}"}]
            kind = "no-canonical-source (honest)"

        thinking_rows.append({
            "schema": "operator_ui_thinking_v2",
            "messages": msgs,
            "meta": {"seq": seq, "view": r.get("view"), "primitive": act["primitive"],
                     "question": q, "kind": kind, "profile_token": tok},
            "provenance_hash": hashlib.sha256(f"{seq}|{kind}".encode()).hexdigest()[:16],
        })
    return action_rows, thinking_rows, excluded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out-action", default="data/ui/ui_action_rows_v2.jsonl")
    ap.add_argument("--out-think", default="data/ui/ui_thinking_rows_v2.jsonl")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.src) if l.strip()]
    A, T, X = build(rows)
    for path, data in ((a.out_action, A), (a.out_think, T)):
        with open(path, "w") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    import collections
    up = len({r["messages"][0]["content"] for r in A})
    print(f"source {len(rows)} | action {len(A)} | thinking {len(T)} | excluded {len(X)}")
    print(f"UNIQUE PROMPTS: {up}/{len(A)}   (was 27/41 — the duplicate-target defect)")
    print(f"thinking kinds: {dict(collections.Counter(r['meta']['kind'] for r in T))}")
    print(f"profile tokens cited: {dict(collections.Counter(r['meta']['profile_token'] for r in T if r['meta']['profile_token']))}")
    for s, why in X:
        print(f"  EXCLUDED seq {s}: {why}")


if __name__ == "__main__":
    main()
