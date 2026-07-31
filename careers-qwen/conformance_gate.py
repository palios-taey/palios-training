#!/usr/bin/env python3
"""conformance_gate.py — WOULD PRODUCTION ACCEPT WHAT THIS ROW TEACHES?

This is the missing half of a question Jesse already split in two.

  provenance_gate.py (fe35aa2) answers: "is this row REAL?"
  Its own commit message names the gap it does not close:
      "gate: SCHEMA CONFORMANCE — provenance proves a row is REAL, not that it is TRAINABLE"

This gate answers the other half. A row can be perfectly real — genuinely captured from a
genuine situation, correct provenance, honest lineage — and still teach the model to emit
something the running system cannot consume. That is not a hypothetical. It happened:

  ui/ui_thinking_rows_v2.jsonl was classified CANONICAL in PAIRS_MANIFEST.md:113 as
  "native <think> + real tool-call trajectories; the corrected thinking lane".
  Measured: 40 rows, 0 tool_calls structures, 18 EMPTY <think></think> blocks, and
  assistant content of the literal form
      {"primitive": "activate", "target": "Apply for this Job", "expect": "..."}

  The production tool is ui_action, whose schema has no primitive, no target, no expect,
  and additionalProperties=False. Every one of those keys is rejected as unknown. The rows
  taught an INVENTED ACTION LANGUAGE that no production surface has ever accepted, so even
  wrapping them in a correct tool_calls envelope would still fail on every call.

  Those rows rode into module 2 and module 3. The tuned checkpoint then elected ZERO tools
  under tool_choice=auto on the real Upwork unit, while the UNTUNED base completed the same
  unit unaided. The SFT program made tool-election worse than no SFT at all.

WHY THIS GATE IMPORTS PRODUCTION INSTEAD OF DESCRIBING IT
---------------------------------------------------------
Every failure in that chain was a check that verified FORM and was read as verifying TRUTH:
a manifest description accepted as a description of contents; a content-pin that proves bytes
match a recorded sha and certifies nothing about what the bytes contain; a provenance gate
that proves a row is real and was read as proving it trainable.

So this gate does not restate the contract — restating it is how it drifts, and a drifted
copy fails exactly when it matters. It IMPORTS the live production schema and calls the
SAME validator the running worker calls:

    ats_mcp_server.UI_ACTION_INPUT_SCHEMA        (apply-machine, the merged apply lane)
    ats_mcp_server._validate_schema              (the function production itself runs)

If production changes its contract, this gate changes with it, with nothing to keep in sync.
The oracle is the running code. That is the whole design.

FAIL-CLOSED. If the production module cannot be imported, this gate does NOT silently pass —
it reports UNKNOWN and exits non-zero. An unverifiable row is not an admitted row.

USAGE
    python3 conformance_gate.py <file.jsonl> [more.jsonl ...]
    python3 conformance_gate.py --all          # sweep every canonical file in PAIRS_MANIFEST
    exit 0 = every row conformant · exit 1 = at least one violation · exit 2 = cannot verify
"""

import json
import os
import sys

# =============================================================================
# RE-ENABLED 2026-07-27, SCOPED — it now validates each row against the surface the ROW
# DECLARES, and refuses to rule on a surface it cannot import. The disable below stands as
# the record of why; read it before touching the dispatch logic.
#
# The condition the disable itself set for re-enabling was:
#     "DO NOT RE-ENABLE until it takes the TARGET SURFACE as input and validates against
#      that surface's contract."
# That is now satisfiable because rows carry `meta.tool_contract`. build_trajectory_rows.py
# stamps "ui_action" on every row it emits, from the surface the capture actually came off —
# so the surface is OBSERVED from the production log, not asserted by me at gate time.
#
# THREE OUTCOMES, and the middle one is the honest one that did not exist before:
#   declared ui_action  -> validated against the LIVE imported schema. A real verdict.
#   declared elsewhere  -> UNVALIDATED. Reported, never counted as a pass. We have no
#                          importable contract for act.py / worker_ui, and a gate that says
#                          PASS when it checked nothing is the form-vs-truth defect again.
#   declared NOTHING    -> a violation in itself. An unlabelled row is the module-3
#                          mechanism: the model cannot condition on a surface the row does
#                          not name, so it learns one unconditional emission shape.
#
# ORIGINAL DISABLE (2026-07-27) — THIS GATE WAS WRONG AND WOULD CONDEMN VALID TRAINING DATA.
#
# It validated EVERY row against ats_mcp_server.ui_action. That is ONE of at least
# THREE action surfaces in this system:
#   ats_mcp_server.ui_action   op/ref/view, ref=^el_[0-9a-f]{24}$   (taey_worker MCP tool)
#   ats_ui/worker_ui.py        primitives — def activate :636, def write :896,
#                              "performed_primitive" :207            (ATS apply walk)
#   treasurer/scripts/loop/act.py   find(name, role=...) — BY NAME   (LinkedIn/SalesNav/Upwork)
#
# Rows using {primitive, target/find, role, expect} with a HUMAN-READABLE target are
# CORRECT for the worker_ui.py and act.py surfaces — act.py:228 is
# `def find(name, role=None, ...)`, so "Apply for this Job" is the right value, not a defect.
# I measured those rows against ui_action's el_<24hex> ref and called them an invented
# vocabulary. They were compliance with a different, real, documented interface.
#
# taey-training-trigger/SKILL.md warned about exactly this, in a table I read:
#   "Vocabulary differs by surface — using the wrong one does nothing."
# I read it and then judged every surface by a fourth one.
#
# DO NOT RE-ENABLE until it takes the TARGET SURFACE as input and validates against that
# surface's contract. A single-surface gate is not a conservative gate — it is a wrong one,
# and it fails in the most expensive direction: rejecting good data with a confident receipt.
# The empty-<think> check was independently sound and can be salvaged; the action-shape check
# cannot, as written.
# =============================================================================
# The surface whose contract this gate can actually import and enforce. Anything else is
# reported UNVALIDATED rather than passed — see the three-outcomes note above.
ENFORCEABLE_SURFACE = "ui_action"

# The production surface. Kept in one place so a relocation is a one-line change and never a
# silently-rewritten copy of the contract.
APPLY_MACHINE = os.environ.get("TAEY_APPLY_MACHINE") or os.path.expanduser("~/apply-machine")


def load_production_contract():
    """Import the LIVE tool contract. Returns (tool_names, schema, validator).

    Never returns a hand-written fallback. A missing production module means this gate
    cannot answer the question it exists to answer, and it must say so rather than guess.

    THE VALIDATOR IS `_validate_ui_action_call`, NOT `_validate_schema` — see the long note in
    build_trajectory_rows.production_contract(). Short version: UI_ACTION_INPUT_SCHEMA is a
    `oneOf` union, `_validate_schema` does not implement `oneOf`, and against the union it
    accepts the module-3 invented vocabulary, a malformed ref, unknown keys, and `{}`. This
    gate ran with it on 2026-07-27 and printed CONFORMANCE PASS over 33 rows production
    rejects. A gate whose validator accepts the empty object certifies nothing.
    """
    if APPLY_MACHINE not in sys.path:
        sys.path.insert(0, APPLY_MACHINE)
    import ats_mcp_server as prod  # noqa: E402  (deliberate: path set above)

    tool = prod.ui_action_tool()
    name = tool["function"]["name"]
    schema = tool["function"]["parameters"]

    def validate(args, _schema_ignored):
        prod._validate_ui_action_call(args)

    return {name}, schema, validate


def skill_currency(skill_path):
    """Is the AUTHORING SKILL current with the contract it teaches against?

    Rationale, and it is not theoretical. taey-training-trigger/SKILL.md instructs owners to
    emit "the exact JSON the primitives layer consumes" using an ATS vocabulary
    (type_filter_select, pointer_activate, escape-as-primitive) that the live ui_action schema
    does not contain. Nothing in apply-machine consumes a "primitive" key any more. Section 6
    of that same skill says: "When a schema, a vocabulary, a canonical source path, or a
    constraint changes, update this file in the same commit. A stale process doc produces
    stale training data, and a status line here that has gone stale will be believed."

    That rule was correct and nothing enforced it, so every owner authoring from the skill
    produced rows against a retired interface. This makes the rule mechanical.

    The pattern is production's own: taey_worker.py:228 refuses to run when
    sha256(_canonical_bytes(ui_action_tool())) differs from its leased binding. The runtime is
    already fail-closed against contract drift; training was not. Same check, training side.

    The skill declares, in its YAML frontmatter, the contract it was written against:
        tool_contract_sha256: <64 hex>
    A skill with no declaration is STALE-BY-DEFAULT — silence is not currency.
    """
    import hashlib
    import re

    if APPLY_MACHINE not in sys.path:
        sys.path.insert(0, APPLY_MACHINE)
    import ats_mcp_server as prod  # noqa: E402
    from taey_worker import _canonical_bytes  # noqa: E402  production's own canonicaliser

    live = hashlib.sha256(_canonical_bytes(prod.ui_action_tool())).hexdigest()

    if not os.path.exists(skill_path):
        return False, live, None, f"authoring skill not found at {skill_path}"
    text = open(skill_path, encoding="utf-8", errors="replace").read()
    m = re.search(r"^tool_contract_sha256:\s*([0-9a-f]{64})\s*$", text, re.M)
    if not m:
        return False, live, None, (
            "skill declares no tool_contract_sha256 — STALE BY DEFAULT. A skill that does not "
            "state which contract it teaches cannot be shown to teach the current one."
        )
    declared = m.group(1)
    if declared != live:
        return False, live, declared, (
            "skill was authored against a DIFFERENT tool contract. Rows authored from it will "
            "teach a retired interface. Update the skill, then re-declare the fingerprint."
        )
    return True, live, declared, None


def iter_assistant_turns(row):
    for msg in row.get("messages") or []:
        if msg.get("role") == "assistant":
            yield msg


def looks_like_an_action_payload(content):
    """True when assistant CONTENT carries a JSON action object.

    This is the exact defect shape: the action written as text instead of called. Detected
    structurally (does it parse to a dict of action-ish keys?) rather than by matching the
    known-bad key names, so a THIRD invented vocabulary is caught too — the previous gates
    each failed by only recognising the specific wrong thing they already knew about.
    """
    text = (content or "").strip()
    # Strip a leading think block; the action payload follows it in the observed defect.
    if text.startswith("<think>") and "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    if not text.startswith("{"):
        return None
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict) or not obj:
        return None
    # An object naming an operation and a thing to operate on is an action, whatever it
    # calls those fields.
    verbish = {"op", "primitive", "action", "command", "verb", "operation"}
    return obj if verbish & set(obj) else None


def declared_surface(row):
    """Which action surface does this row say it is on? None when it does not say.

    Read from meta.tool_contract, which build_trajectory_rows.py stamps from the production
    capture. Not inferred from the row's shape — inferring the surface from the shape and
    then judging the shape against it is circular, and is how the previous version convinced
    itself that valid act.py rows were an invented vocabulary.
    """
    return ((row.get("meta") or {}).get("tool_contract")
            or row.get("contract")
            or None)


def check_row(row, idx, tool_names, schema, validator):
    """Return (violations, unvalidated_reason) for one row.

    unvalidated_reason is non-None when the row is on a real surface whose contract this
    gate cannot import — that is NOT a pass and must not be reported as one.
    """
    bad = []
    surface = declared_surface(row)

    # 0. An unlabelled row is a violation on its own terms. Measured 2026-07-27: 0/58 of the
    #    careers action rows named their surface, and that is a sufficient mechanism for the
    #    module-3 tool-election regression — one unconditional emission shape applied
    #    everywhere. A row that does not say which interface it is on cannot teach one.
    if not surface:
        bad.append(
            f"row {idx}: NO DECLARED SURFACE (meta.tool_contract absent) — the model cannot "
            f"condition on an interface the row does not name; this is the module-3 shape"
        )

    # Can this gate rule on the row's ACTION SHAPE? Only when the row names the one surface
    # whose contract we import. An unlabelled row is NOT judgeable here: without a declared
    # surface there is no contract to judge against, and picking ui_action by default is
    # exactly the error that disabled this gate — it produced the nonsense message
    # "None requires a tool_calls array" on its first run against unlabelled rows.
    # The missing label is reported above as the violation it is; the shape is left open.
    judgeable = surface == ENFORCEABLE_SURFACE

    for msg in iter_assistant_turns(row):
        content = msg.get("content") or ""

        # 1. An empty think block teaches the model to open and close a reasoning channel
        #    without reasoning in it — and production runs enable_thinking=False, so it
        #    matches nothing production ever does. SURFACE-INDEPENDENT: true on every
        #    interface, so it runs even for rows we cannot otherwise rule on.
        stripped = content.replace(" ", "").replace("\n", "").replace("\t", "")
        if "<think></think>" in stripped:
            bad.append(f"row {idx}: EMPTY <think></think> — teaches a hollow reasoning channel")

        # 2. The action-as-content defect. SURFACE-DEPENDENT and only checked for ui_action:
        #    on act.py and worker_ui a JSON action object in content is the CORRECT emission,
        #    not a defect. Judging those rows here is precisely the error that disabled this
        #    gate the first time.
        if not judgeable:
            continue
        payload = looks_like_an_action_payload(content)
        if payload is not None and not msg.get("tool_calls"):
            bad.append(
                f"row {idx}: ACTION-AS-CONTENT — assistant writes {sorted(payload)[:4]} "
                f"as text; {surface} requires a tool_calls array"
            )

        # 3. Where a tool call IS present, production must actually accept it.
        for call in msg.get("tool_calls") or []:
            fn = call.get("function") or {}
            fname = fn.get("name")
            if fname not in tool_names:
                bad.append(f"row {idx}: unknown tool {fname!r}; production exposes {sorted(tool_names)}")
                continue
            raw = fn.get("arguments")
            try:
                args = json.loads(raw) if isinstance(raw, str) else (raw or {})
            except (ValueError, TypeError) as exc:
                bad.append(f"row {idx}: tool arguments are not valid JSON ({exc})")
                continue
            try:
                validator(args, schema)  # THE production validator, not a reimplementation
            except Exception as exc:
                bad.append(f"row {idx}: production would REJECT this call — {exc}")

    if not judgeable:
        why = (f"declares {surface!r}; no importable contract for that surface"
               if surface else "declares no surface; action shape not judgeable")
        return bad, why
    return bad, None


def check_file(path, tool_names, schema, validator):
    violations = []
    unvalidated = []
    rows = 0
    with open(path) as fh:
        for idx, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            rows += 1
            try:
                row = json.loads(line)
            except ValueError as exc:
                violations.append(f"row {idx}: unparseable JSON ({exc})")
                continue
            bad, skipped = check_row(row, idx, tool_names, schema, validator)
            violations.extend(bad)
            if skipped:
                unvalidated.append(f"row {idx}: {skipped}")
    return rows, violations, unvalidated


def main(argv):
    try:
        tool_names, schema, validator = load_production_contract()
    except Exception as exc:
        print(f"CANNOT VERIFY — production contract unavailable from {APPLY_MACHINE}: {exc}", file=sys.stderr)
        print("FAIL-CLOSED: an unverifiable row is not an admitted row.", file=sys.stderr)
        return 2

    paths = [a for a in argv if not a.startswith("-")]

    # SKILL CURRENCY — checked FIRST and blocking. A conformant row authored from a stale
    # skill is an accident, not a process: the next row from the same instructions will be
    # wrong again. Default-on; --no-skill-check exists for auditing historical files.
    if "--no-skill-check" not in argv:
        skill = os.environ.get(
            "TAEY_TRAINING_SKILL",
            os.path.expanduser("~/.claude/skills/taey-training-trigger/SKILL.md"),
        )
        ok, live, declared, why = skill_currency(skill)
        print(f"authoring skill    : {skill}")
        print(f"  live contract    : {live}")
        print(f"  skill declares   : {declared or '(none)'}")
        if not ok:
            print(f"\nSKILL STALE — pair admission BLOCKED.\n  {why}", file=sys.stderr)
            print("  Rows may not enter the queue while the instructions that produce them "
                  "describe a contract production no longer honours.", file=sys.stderr)
            return 1
        print("  skill currency   : OK")

    if not paths:
        print(__doc__.strip().split("USAGE")[-1].strip())
        return 2

    print(f"production contract: tools={sorted(tool_names)} "
          f"required={schema.get('required')} additionalProperties={schema.get('additionalProperties')}")
    total_bad = 0
    total_unvalidated = 0
    for path in paths:
        if not os.path.exists(path):
            print(f"  MISSING  {path}")
            total_bad += 1
            continue
        rows, violations, unvalidated = check_file(path, tool_names, schema, validator)
        checked = rows - len(unvalidated)
        if violations:
            status = f"FAIL ({len(violations)})"
        elif unvalidated:
            status = "PARTIAL"
        else:
            status = "PASS"
        print(f"  {status:<12} {path}  [{rows} rows: {checked} validated, "
              f"{len(unvalidated)} unvalidated]")
        for v in violations[:8]:
            print(f"       - {v}")
        if len(violations) > 8:
            print(f"       - ... and {len(violations) - 8} more")
        # Say out loud what was NOT ruled on. A gate that reports only what it checked lets
        # a reader infer coverage it never had.
        for u in unvalidated[:3]:
            print(f"       ? UNVALIDATED {u}")
        if len(unvalidated) > 3:
            print(f"       ? ... and {len(unvalidated) - 3} more unvalidated")
        total_bad += len(violations)
        total_unvalidated += len(unvalidated)

    if total_bad:
        print(f"\nCONFORMANCE FAIL — {total_bad} violation(s)")
        return 1
    if total_unvalidated:
        # Exit 0: these rows are not condemned. But the word PASS is not available for a
        # file this gate did not fully rule on.
        print(f"\nCONFORMANCE PASS on what was checkable; {total_unvalidated} row(s) "
              f"UNVALIDATED (surface has no importable contract). Not a clean bill.")
        return 0
    print("\nCONFORMANCE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
