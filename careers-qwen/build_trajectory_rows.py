#!/usr/bin/env python3
"""build_trajectory_rows.py — assemble observe→act rows from REAL successful applications.

WHY THIS EXISTS, AND WHY IT IS AN ASSEMBLY JOB
-----------------------------------------------
Module 3 taught the model to WRITE actions as content, in a vocabulary production never
accepted (`{"primitive","target","expect"}` against a `ui_action` schema that has none of those
keys and `additionalProperties: false`). The tuned checkpoint then elected ZERO tools on the
real Upwork unit while the UNTUNED base completed it unaided — the SFT made tool-election worse
than no SFT at all.

The corpus to fix it was never missing. `apply-machine/bundles/*/submit_agent.log` holds one
JSON record per executed action from real applications — 367 of them validate against
production's own schema with ZERO rejections, covering focus/activate/write/navigate/page/
observe/key/verify. They were sitting there while we trained 159 rows containing 2 tool calls.

WHY SINGLE-TURN (instruction, tool_call) PAIRS CANNOT WORK HERE
----------------------------------------------------------------
`ref` matches `^el_[0-9a-f]{24}$` — an opaque handle that exists only inside one accessibility
snapshot. A handle valid in one snapshot is meaningless in the next. So a row with a hardcoded
ref teaches the model to RECALL a fictitious string instead of DERIVING one from what it sees.
The task is inherently multi-turn: observe → receive a snapshot carrying refs → act on a ref
FROM that snapshot.

That is what this builds. The user turn carries the real element list; the assistant turn calls
`ui_action` with a ref that must be read out of it. The join is `tree.revision == action.revision`
— the snapshot is the state the action was taken FROM, not after.

DISCIPLINE
----------
Every emitted arguments object is validated by the SAME `_validate_schema` production runs,
against the SAME `ui_action_tool()` schema, before it is written. A row production would reject
is never emitted — it is REJECTED with a reason. Fail-closed: if the production module cannot be
imported, this writes nothing and exits non-zero. Emitting fewer rows than the source is CORRECT.

USAGE
    python3 build_trajectory_rows.py --bundles <dir> --out rows.jsonl [--dry-run]
"""

import argparse
import glob
import json
import os
import sys

APPLY_MACHINE = os.environ.get("TAEY_APPLY_MACHINE") or os.path.expanduser("~/apply-machine")

# WHICH FIELDS BELONG IN THE CALL IS THE CONTRACT'S ANSWER, NOT A LIST I MAINTAIN.
#
# This was a hardcoded tuple until 2026-07-27 and it carried the identical defect that
# apply-machine's action ledger carried (fixed there in 3ba4e3f, "record the arguments each op
# declares, not a fixed list"). A fixed list is wrong in BOTH directions at once:
#
#   TOO NARROW — it dropped every op-specific argument. Measured against the live schemas:
#       page   -> direction        key    -> key
#       retire -> reason           sweep  -> application_identity, route_sha256
#     `direction` is REQUIRED on page. So when the ledger fix lands and starts recording it,
#     this builder would still have dropped it and the 8 page trajectories would still be
#     rejected — the corpus-growth fix silently defeated one layer downstream. That is the
#     whole reason this is derived now: two hand-maintained lists had to agree, and they did
#     not, and nothing compared them.
#
#   TOO WIDE — a blanket `application_identity` entry put snapshot context into the arguments
#     of activate/focus/write, where the schema has no such key, which is what made all 33
#     rows unservable. Yet `sweep` DOES declare application_identity. A single flat list cannot
#     express "valid on sweep, invalid on write"; the per-op schema states it exactly.
#
# Deriving per-op gets both right and stays right when an op gains an argument.
def passthrough_for(op, schemas):
    """The argument names THIS op declares, per the live contract. Never a maintained list."""
    return set(((schemas.get(op) or {}).get("properties") or {}))


def production_contract():
    """Return (tool_name, schema_for_display, THE PRODUCTION VALIDATOR).

    THE VALIDATOR MUST BE `_validate_ui_action_call`, NOT `_validate_schema`.

    This distinction is not pedantic — it is the whole gate. UI_ACTION_INPUT_SCHEMA is a
    `oneOf` UNION of eleven per-operation branches, and `_validate_schema` DOES NOT IMPLEMENT
    `oneOf`: handed the union it checks `type: object` and returns. Measured 2026-07-27, it
    accepted every one of these against the union —
        {"primitive":"activate","target":"Apply for this Job"}   the module-3 invented vocab
        {"op":"activate", ..., "ref":"Apply for this Job"}       a ref that is not a ref
        {...,"bogus":1}                                          an unknown key
        {}                                                       the empty object
    A validator that accepts the empty object is not validating. Both this builder and
    conformance_gate.py used it, so both emitted a confident PASS over 33 rows that production
    rejects — the exact form-verified-read-as-truth-verified failure that produced module 3.

    Production never validates against the union. `_validate_ui_action_call` (ats_mcp_server.py
    :371) dispatches on `op`, strips it, and validates the remainder against
    UI_ACTION_SCHEMA_BY_NAME[op] — the concrete branch, with its required list, its ref pattern
    and additionalProperties:false. That is the function the running worker calls, so it is the
    only one whose verdict means anything here.
    """
    if APPLY_MACHINE not in sys.path:
        sys.path.insert(0, APPLY_MACHINE)
    import ats_mcp_server as prod  # noqa: E402
    tool = prod.ui_action_tool()

    def validate(args, _schema_ignored):
        prod._validate_ui_action_call(args)

    # The PER-OP schemas, so the builder's notion of "which arguments exist" is the contract's,
    # not a list kept in sync by hand. This is the same object _validate_ui_action_call selects
    # its branch from, so the emitter and the validator cannot disagree.
    return (tool["function"]["name"], tool["function"]["parameters"], validate,
            prod.UI_ACTION_SCHEMA_BY_NAME)


def load_snapshots(bundle):
    """revision -> the snapshot at that revision. Later files win only if revision differs."""
    by_rev = {}
    for path in sorted(glob.glob(os.path.join(bundle, "step_*_tree.txt"))):
        try:
            snap = json.load(open(path))
        except (ValueError, OSError):
            continue
        rev = snap.get("revision")
        if rev and rev not in by_rev:
            by_rev[rev] = (snap, os.path.basename(path))
    return by_rev


def load_actions(bundle):
    acts = []
    for path in glob.glob(os.path.join(bundle, "submit_agent.log")):
        for line in open(path):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("tool"):
                acts.append(rec)
    return acts


def render_observation(snap):
    """What the model can SEE. Refs are shown because the model must pick one from them."""
    lines = [
        "You are operating a real job-application form through your own hands.",
        f"Application: {snap.get('application_identity','(unknown)')}",
        f"View: {snap.get('view','(unknown)')}",
        f"Form revision: {snap.get('revision','(unknown)')}",
        "",
        "Accessible elements in the current snapshot:",
    ]
    for el in (snap.get("elements") or []):
        bits = [f"  ref={el.get('ref')}", f"role={el.get('role')}", f"name={el.get('name')!r}"]
        if el.get("operations"):
            bits.append(f"ops={','.join(el['operations'])}")
        if el.get("answer_token"):
            bits.append(f"answer_token={el['answer_token']}")
        if el.get("states"):
            bits.append(f"states={','.join(el['states'])}")
        lines.append(" ".join(bits))
    if snap.get("missing"):
        lines.append(f"Missing/unmapped: {snap['missing']}")
    lines += ["", "Take the next action."]
    return "\n".join(lines)


def walk_dirs(bundles_root):
    """Every directory holding a submit_agent.log — top-level walks AND superseded retries.

    ADMISSION RULE (treasurer, 2026-07-27, and it is theirs to set, not the builder's):
    a retry under `<app>/.superseded_submit_rN/` is admissible per-ACTION, because the reason a
    walk was superseded is a WALK-level property and does not reach back to invalidate the
    individual steps. Each step still has to earn its place on its own: ok=true, a snapshot at
    its revision, its ref visible in that snapshot, accepted by _validate_ui_action_call.

    Globbing only `bundles/*/submit_agent.log` saw 10 of 27 logs and 39 of 98 action records.
    The corpus was never as small as that made it look; the glob was one level too shallow.
    """
    found = []
    for root, _dirs, files in os.walk(bundles_root):
        if "submit_agent.log" in files:
            found.append(root)
    return sorted(found)


def build(bundle, C):
    tool_name, schema, validate, op_schemas = C
    snaps = load_snapshots(bundle)
    emitted, rejected = [], []
    recs = load_actions(bundle)
    for i, rec in enumerate(recs):
        seq = rec.get("call") or rec.get("action_count")

        # THE WALK'S LAST ACTION IS DROPPED UNCONDITIONALLY (treasurer, 2026-07-27).
        # 13 of the retry dirs are classified `possible_post_action`: the runtime could not
        # determine whether that final action's effect actually landed. An indeterminate effect
        # is the one thing that must never become a training target — the row would assert an
        # outcome nobody observed. This applies to COMPLETED walks too, not only superseded
        # ones: "the walk finished" is a walk-level property and makes its terminal action no
        # more determinate than any other.
        if i == len(recs) - 1:
            rejected.append((seq, "terminal action of the walk — effect indeterminate, dropped by rule"))
            continue

        if not rec.get("ok"):
            rejected.append((seq, "action did not report ok=true — not a success trajectory"))
            continue
        rev = rec.get("revision")
        if rev not in snaps:
            rejected.append((seq, f"no snapshot for revision {rev} — cannot show what was observed"))
            continue
        snap, snap_file = snaps[rev]

        # Only the arguments THIS op declares, read from the live per-op schema. A value the
        # log did not record stays absent — the validator then rejects the row for the missing
        # required field, which is the correct outcome: a reconstructed argument is an
        # authored one, and an authored argument in a captured trajectory is a fabrication.
        args = {"op": rec["tool"]}
        for f in sorted(passthrough_for(rec["tool"], op_schemas)):
            v = rec.get(f)
            if v:
                args[f] = v

        # The ref MUST be visible in the snapshot, or the row teaches recall instead of derivation.
        refs = {el.get("ref") for el in (snap.get("elements") or [])}
        if args.get("ref") and args["ref"] not in refs:
            rejected.append((seq, f"ref {args['ref']} not present in its own snapshot ({snap_file})"))
            continue

        try:
            validate(args, schema)          # production's validator, not a reimplementation
        except Exception as exc:
            rejected.append((seq, f"production would REJECT: {exc}"))
            continue

        call_id = f"call_{seq}"
        result = {"ok": True, "view": rec.get("after_view"), "revision": rec.get("after_revision")}
        emitted.append({
            "schema": "operator_ui_trajectory_v1",
            "messages": [
                {"role": "user", "content": render_observation(snap)},
                # content None — the action is CALLED, never written into the content channel.
                # `arguments` is a MAPPING, not a JSON string. The canonical chat template
                # (dense-9b/inference/qwen3.5-tooluse.jinja:114-117) resolves it two ways:
                #     arguments is mapping  -> use directly           <- what we emit
                #     arguments is string   -> `| fromjson`           <- OpenAI wire format
                # and `fromjson` DOES NOT EXIST in transformers 5.3.0 / jinja2 3.1.6 on the
                # training nodes, so the string form raises
                #     TemplateRuntimeError: No filter named 'fromjson' found
                # and tokenization dies on row 1. The template's own comment says the native
                # form here is a mapping ("soma-proxy normalises to mapping"), so emitting a
                # string was us handing it the one branch this stack cannot execute.
                #
                # This surfaced only now because module 5 is the FIRST corpus containing real
                # tool_calls — modules 1-4 were prose, so the template's tool-call branch had
                # never once been rendered in training. Worth stating plainly: until this run,
                # nothing had ever exercised the path that turns a tool call into tokens.
                {"role": "assistant", "content": None, "tool_calls": [{
                    "id": call_id, "type": "function",
                    "function": {"name": tool_name, "arguments": args},
                }]},
                {"role": "tool", "tool_call_id": call_id,
                 "content": json.dumps(result, sort_keys=True, separators=(",", ":"))},
            ],
            "meta": {
                "bundle": os.path.basename(bundle.rstrip("/")),
                "snapshot": snap_file,
                "call": seq,
                "at": rec.get("at"),
                "tool_contract": tool_name,
                # production runs enable_thinking=False (taey_worker.py:93) — no think block.
                "thinking": False,
            },
        })
    return emitted, rejected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles", default=os.path.join(APPLY_MACHINE, "bundles"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    try:
        C = production_contract()
    except Exception as exc:
        print(f"ABORT — production contract unavailable from {APPLY_MACHINE}: {exc}", file=sys.stderr)
        return 2

    all_rows, all_rej, ops = [], [], {}
    for bundle in walk_dirs(a.bundles):
        rows, rej = build(bundle, C)
        all_rows += rows
        # Label by the path RELATIVE to the bundles root, so a superseded retry is
        # distinguishable from its parent walk in the rejection report instead of both
        # collapsing to the same basename.
        _label = os.path.relpath(bundle, a.bundles)
        all_rej += [(_label, s, w) for s, w in rej]
        for r in rows:
            _a = r["messages"][1]["tool_calls"][0]["function"]["arguments"]
            op = (_a if isinstance(_a, dict) else json.loads(_a))["op"]
            ops[op] = ops.get(op, 0) + 1

    print(f"contract : {C[0]}  required={C[1].get('required')}")
    print(f"EMITTED  : {len(all_rows)} trajectory rows")
    print(f"REJECTED : {len(all_rej)}")
    print(f"ops      : {dict(sorted(ops.items(), key=lambda x: -x[1]))}")
    seen = {}
    for _, _, w in all_rej:
        k = w.split("—")[0].split("(")[0].strip()[:60]
        seen[k] = seen.get(k, 0) + 1
    for k, v in sorted(seen.items(), key=lambda x: -x[1])[:6]:
        print(f"   {v:4d}  {k}")

    if not all_rows:
        print("\nNothing emitted — honest result, not a crash.")
        return 1
    if a.dry_run:
        print("\n--dry-run: nothing written.\nSAMPLE:")
        print(json.dumps(all_rows[0], indent=1)[:900])
        return 0
    with open(a.out, "w") as fh:
        for r in all_rows:
            fh.write(json.dumps(r, ensure_ascii=True) + "\n")
    print(f"\nwrote {len(all_rows)} -> {a.out}")
    print("Now run conformance_gate.py on it — a builder checking its own output is still "
          "one thing marking its own homework.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
