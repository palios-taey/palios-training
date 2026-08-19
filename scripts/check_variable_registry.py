#!/usr/bin/env python3
"""The variable registry must describe the launcher, in BOTH directions.

WHY THIS EXISTS. Jesse, 2026-08-18: "ensure the dynamic variables are indeed dynamic and that you
don't lose track of them." Losing track is the documented failure. This launcher reads ~74
variables. Thirteen of them silently ASSIGNED legacy values when unset -- `TOTAL_STEPS:=3000`,
`MAX_SEQ:=2560` -- so an omitted variable trained a different campaign. A packet written ABOUT that
surface then missed an entire bash syntax and reported those thirteen as empty-string defaults,
because nothing connected the code to a statement of what the code was supposed to read.

WHAT MAKES THIS DIFFERENT FROM THE CHECKERS THAT CAME BEFORE IT. tutor-grok's audit found that
scripts/check_manifest_pins.py and scripts/check_docs_index.py can both PASS while the thing they
guard is wrong, because they verify internal consistency rather than correspondence to an
independent source. This one compares the REGISTRY against the CODE'S ACTUAL READS, both ways:

    a variable the launcher reads that no class in the registry names  -> FAIL
    a variable the registry names that the launcher never reads        -> FAIL

Neither side can drift without the other noticing. A new knob added to the launcher fails CI until
someone classifies it, which is precisely "do not lose track of them". A registry entry left behind
by a deleted knob fails too, so the file cannot rot into decoration.

Variables outside the launcher live under an `external` mapping keyed by owning file. They are
checked against that file and may not control a conditional before or containing manifest/content
pin verification. A diagnostic flag is not permission to bypass the production gate.

IT ALSO CHECKS THE CLASS IS TRUE, not merely present:
    dynamic_required          must be enforced by `${VAR:?}` in the code
    invariant                 must be an unconditional assignment, never `:=` or `:-`
    no `${VAR:=...}` may exist at all -- that syntax ASSIGNS, which is the original defect

    python3 scripts/check_variable_registry.py

Exit 0 = the registry describes the launcher. Exit 1 = it does not, and every disagreement prints.
"""
import os
import re
import subprocess
import sys

LAUNCHER = "dense-9b/recipes/run_4node_27b_cpt.sh"

# SCOPE IS NOT ONE FILE. Removing the per-run defaults from the launcher moved them UP into
# run_till_done_v3.sh, which then answered the launcher's ${VAR:?} with values nobody chose. A
# wrapper that supplies a per-run default is the same defect wearing a different filename, so every
# script that INVOKES the launcher is checked too.
# CALLER-DEFAULT DEBT, FROZEN. The scan below found nine caller-side defaults across two SFT
# wrappers on the day it was written. They are NOT fixed here, deliberately: both feed capabilities
# PRODUCTION_MANIFEST.yml marks not-runnable (sft_stage2_lora awaiting authorization,
# sft_27b_fullparam contested), and this same week two production breaks were caused by changing a
# launcher without first enumerating its callers. Fixing them belongs in a dedicated change with its
# own caller validation and its own audit.
#
# The debt is frozen, not ignored: each entry records an exact count. A file that exceeds its
# baseline fails, a NEW file with caller defaults fails, and a baseline higher than reality fails so
# the number must come down as they are fixed. It can shrink. It cannot grow, and it cannot be
# forgotten, which is the only property that matters for work deferred at 3am.
CALLER_DEFAULT_BASELINE = {
    "careers-qwen/launch_stage2_sft.sh": 6,
}


class CannotEnumerate(Exception):
    """The invoker set could not be determined. NOT the same as an empty invoker set."""


def invokers():
    # A FAILED ENUMERATION IS NOT AN EMPTY ENUMERATION. git grep exits 1 when it matches
    # nothing (a real answer) and >1 on error -- a tree with no .git exits 128. This function
    # used to read .stdout without ever looking at .returncode, so an error produced zero
    # callers, every caller-side default became invisible, and the checker reported the
    # baseline as STALE. That false finding was handed to tutor-codex as blocking, and
    # complying with it would have lowered a TRUE baseline of 6 to 0 -- freeing the debt to
    # grow back silently. The checker was at its most confident exactly where it was blind.
    probe = subprocess.run(["git", "grep", "-l", "run_4node_27b_cpt", "--", "*.sh"],
                           capture_output=True, text=True)
    if probe.returncode > 1:
        raise CannotEnumerate(
            f"git grep exited {probe.returncode}: {probe.stderr.strip() or '<no stderr>'}. "
            "Refusing to report an empty caller set, which reads as 'no caller defaults'."
        )
    out = []
    for line in probe.stdout.split("\n"):
        if not line.strip() or line == LAUNCHER:
            continue
        src = open(line, encoding="utf-8", errors="replace").read()
        if re.search(r"^[^\n#]*(bash|sh|exec|\.)\s+[^\n#]*run_4node_27b_cpt\.sh", src, re.M):
            out.append(line)
    return sorted(out)
REGISTRY = "dense-9b/recipes/VARIABLES.yml"

# Read by the launcher's own gate rather than by a ${VAR:?} form, so they are enforced but the
# regex below cannot see them. Kept explicit rather than silently excluded.
GATE_ENFORCED = {"LR", "WARMUP_STEPS", "MODEL_PATH", "RESUME_DELTA"}

# NO HAND-MAINTAINED DENYLIST. The previous version listed nine variables by name, and tutor-grok
# walked straight past it: MODEL_PATH and RESUME_DELTA were never on the list, so either could be
# handed a `:-/legacy` default and this check stayed green. A closed list is a list someone has to
# remember to extend -- the same "lose track of it" failure this whole file exists to prevent.
#
# The rule is now DERIVED from the registry: anything classified dynamic_* varies per run, and a
# variable that varies per run must not carry a default. Add a variable to a dynamic class and it is
# automatically protected; no second place to update.
def never_defaulted(reg):
    out = set()
    for cls, items in reg.items():
        if cls.startswith("dynamic"):
            out |= set(items or [])
    return out


def caller_never_defaulted(reg, source):
    protected = never_defaulted(reg)
    # The launcher is intentionally multi-mode. CPT-only requirements are meaningless to a
    # caller that explicitly supplies an SFT dataset, and the two existing SFT wrappers have
    # their own frozen default debt. Treating OUTPUT_DIR's SFT output convention as a new CPT
    # default is a domain error in the checker, not new production debt.
    if re.search(r"\bSFT_(?:DIR|JSONL)\s*=", source):
        protected -= set(reg.get("dynamic_required_when_cpt") or [])
    return protected


def code_reads(path):
    """Every environment variable the script actually references, comments excluded."""
    src = "\n".join(l for l in open(path).read().split("\n") if not l.strip().startswith("#"))
    # {1,} not {2,}: the previous bound required THREE characters, so `LR` -- the one variable
    # whose silent non-forwarding is already a documented incident (post_cpt_pipeline.sh records
    # LR and WARMUP_STEPS were not forwarded until 2026-07-13) -- was invisible to this enumerator
    # and reported as a stale registry entry. A checker that cannot see a variable cannot protect it.
    names = set(re.findall(r"\$\{?([A-Z][A-Z0-9_]{1,})", src))
    forms = {
        # TOP-LEVEL enforcement only. A `${VAR:?}` nested inside an `if` fires for SOME callers
        # and not others, which is not what dynamic_required claims. CPT_PACKED sat inside an
        # SFT-mode `if` and this check called the class true -- tutor-grok constructed exactly that.
        "required": set(re.findall(r"^: \"\$\{([A-Z][A-Z0-9_]+):\?", src, re.M)),
        "required_conditional": set(re.findall(r"^\s+: \"\$\{([A-Z][A-Z0-9_]+):\?", src, re.M)),
        "assigned": set(re.findall(r"\$\{([A-Z][A-Z0-9_]+):=", src)),
        # `${VAR:-}` with an EMPTY default is a null-check idiom, not a default value.
        # ADAFACTOR_DOSE_LOG is set unconditionally at :240 and then guarded at :241 with
        # `[ -n "${ADAFACTOR_DOSE_LOG:-}" ]` before being forwarded. Counting that guard as a
        # default reported a true invariant as misclassified -- a false positive from this
        # very check, caught on its first run.
        # BOTH forms. `${VAR-x}` (no colon) is also a default -- it applies when VAR is UNSET, which is
        # exactly the case this file cares about. tutor-grok slipped HORIZON_PARTIAL-213 past the
        # colon-only regex.
        "optional": set(re.findall(r"\$\{([A-Z][A-Z0-9_]+):?-[^}\s]", src)),
        "unconditional": set(re.findall(r"^([A-Z][A-Z0-9_]{2,})=", src, re.M)),
    }
    return names, forms


PIN_GATE = re.compile(
    r"\b(?:MANIFEST|RESOLVER|RESOLVED|SHA_[A-Z0-9_]*|sha256sum)\b|CONTENT DRIFT"
)


def conditional_headers(source):
    """Return shell if/elif condition headers and their source lines."""
    headers = []
    active = None
    for line_no, raw in enumerate(source.splitlines(), 1):
        code = raw.split("#", 1)[0]
        if active is not None:
            active["condition"].append(code)
            if re.search(r"\bthen\b", code):
                headers.append(active)
                active = None
            continue
        if re.match(r"^\s*(?:if|elif)\b", code):
            active = {
                "start": line_no,
                "condition": [code],
            }
            if re.search(r"\bthen\b", code):
                headers.append(active)
                active = None
    return headers


def external_failures(external):
    failures = []
    if not isinstance(external, dict):
        return ["EXTERNAL SCHEMA external must map owning file paths to variable lists."]
    for owner, variables in sorted(external.items()):
        normalized = os.path.normpath(owner)
        if os.path.isabs(owner) or normalized == ".." or normalized.startswith("../"):
            failures.append(f"EXTERNAL OWNER {owner!r} is not a repo-relative path.")
            continue
        if not isinstance(variables, list) or not all(isinstance(v, str) for v in variables):
            failures.append(f"EXTERNAL SCHEMA {owner} must map to a list of variable names.")
            continue
        if not os.path.isfile(owner):
            failures.append(f"EXTERNAL OWNER {owner} is absent; its registry entries cannot be verified.")
            continue
        source = open(owner, encoding="utf-8", errors="replace").read()
        names, _ = code_reads(owner)
        executable = "\n".join(
            "" if line.strip().startswith("#") else line.split("#", 1)[0]
            for line in source.splitlines()
        )
        gate_lines = [
            line_no for line_no, line in enumerate(executable.splitlines(), 1)
            if PIN_GATE.search(line)
        ]
        final_gate_line = max(gate_lines, default=0)
        conditions = conditional_headers(source)
        for variable in variables or []:
            if variable not in names:
                failures.append(
                    f"EXTERNAL STALE {owner} does not read registered variable {variable}."
                )
                continue
            variable_ref = re.compile(rf"\$\{{?{re.escape(variable)}\b")
            for condition_header in conditions:
                condition = "\n".join(condition_header["condition"])
                if not variable_ref.search(condition):
                    continue
                if condition_header["start"] <= final_gate_line:
                    failures.append(
                        f"EXTERNAL BYPASS {owner}:{condition_header['start']} {variable} controls a conditional "
                        f"before or containing the final manifest/content pin gate at line "
                        f"{final_gate_line}. External flags may diagnose a gate, never skip it."
                    )
    return failures


def main():
    for f in (LAUNCHER, REGISTRY):
        if not os.path.isfile(f):
            print(f"ABORT: {f} not found", file=sys.stderr)
            return 1
    try:
        import yaml
    except ImportError:
        print("ABORT: pyyaml is required", file=sys.stderr)
        return 1

    reg = yaml.safe_load(open(REGISTRY)) or {}
    external = reg.get("external") or {}
    launcher_reg = {cls: items for cls, items in reg.items() if cls != "external"}
    classified = {}
    for cls, items in launcher_reg.items():
        for v in (items or []):
            classified[v] = cls

    names, forms = code_reads(LAUNCHER)
    failures = external_failures(external)

    # 1. THE ASSIGN SYNTAX MUST NOT EXIST. It is the original defect.
    if forms["assigned"]:
        failures.append(
            f"ASSIGN SYNTAX  {LAUNCHER} still uses ${{VAR:=default}} for: "
            f"{', '.join(sorted(forms['assigned']))}. That ASSIGNS, so an unset variable runs with "
            f"a value the caller never chose. Use ${{VAR:?msg}} or an unconditional assignment."
        )

    # 2. CODE -> REGISTRY. A knob nobody classified is a knob that got lost.
    for v in sorted(names):
        if v not in classified:
            failures.append(
                f"UNCLASSIFIED   {LAUNCHER} reads {v} but {REGISTRY} does not classify it. "
                f"Add it to a class, or this variable is untracked."
            )

    # 3. REGISTRY -> CODE. An entry for a knob that no longer exists is rot.
    # GATE_ENFORCED is NOT an exemption from this direction. It only means the enforcement is the
    # launcher's own gate rather than a ${VAR:?}. The variable must still be READ by the code --
    # tutor-grok deleted LR from a throwaway copy, left it in the registry, and this check stayed
    # green because the old version skipped GATE_ENFORCED entirely.
    for v, cls in sorted(classified.items()):
        if v not in names:
            failures.append(
                f"STALE ENTRY    {REGISTRY} lists {v} under '{cls}' but {LAUNCHER} never reads it."
            )

    # 3b. A variable that varies per run must not carry a default, derived from its class.
    for v in sorted(never_defaulted(launcher_reg)):
        if v in forms["assigned"] or v in forms["optional"]:
            failures.append(
                f"DEFAULT BANNED {v} carries a default in the code. It varies per run, so a default "
                f"is a silent answer to a question nobody asked. This holds whatever the registry "
                f"classifies it as."
            )

    # 3c. A gate suppressor may be unset, but must never carry a NON-EMPTY default.
    src_now = open(LAUNCHER).read()
    for v in reg.get("gate_suppressor") or []:
        # EVERY occurrence, not the first. `re.search` stops at match one, and this launcher has a
        # LEGITIMATE empty `${HORIZON_PARTIAL:-}` null-check at :211 -- so the empty first match
        # satisfied the check and a NON-EMPTY suppressing default added anywhere after that line was
        # never examined. Gatekeeper demonstrated it by execution: inserting the default at :212
        # made this checker exit 0 while a non-empty HORIZON_PARTIAL silently disabled the
        # fail-closed horizon gate.
        #
        # THE RULE, because this is the second time in one file: a check asking "does ANY occurrence
        # violate this" must iterate findall. re.search answers "does at least one match exist",
        # which is only correct when the check is a boolean existence test. I fixed exactly this
        # first-match flaw in the gate-loop check and reintroduced it here in the same edit.
        defaults = [d for d in re.findall(rf"\$\{{{v}:?-([^}}]*)\}}", src_now) if d.strip()]
        if defaults:
            failures.append(
                f"SUPPRESSOR     {v} carries {len(defaults)} non-empty default(s) {defaults}. It disables a "
                f"fail-closed gate, so a default silently disables that gate for every caller who "
                f"never asked for it. Unset is the only safe default."
            )

    # 3d. NO INVOKER MAY SUPPLY A PER-RUN DEFAULT ON THE LAUNCHER'S BEHALF.
    found = {}
    try:
        invoker_files = invokers()
    except CannotEnumerate as exc:
        # Rendered as an ABORT, never as a traceback: a crashing checker and a firing checker
        # both exit non-zero, so the only thing that distinguishes them is what they print.
        print(f"ABORT: cannot enumerate the launcher's invokers -- {exc}", file=sys.stderr)
        return 1
    for inv in invoker_files:
        isrc = "\n".join(
            l for l in open(inv, encoding="utf-8", errors="replace").read().split("\n")
            if not l.strip().startswith("#")
        )
        protected = caller_never_defaulted(launcher_reg, isrc)
        for v in sorted(set(re.findall(r"\$\{([A-Z][A-Z0-9_]+):?-[^}\s]", isrc))):
            if v in protected:
                found.setdefault(inv, set()).add(v)
    for inv, vs in sorted(found.items()):
        allowed = CALLER_DEFAULT_BASELINE.get(inv, 0)
        if len(vs) > allowed:
            failures.append(
                f"CALLER DEFAULT {inv} supplies defaults for {len(vs)} per-run variable(s) "
                f"(baseline {allowed}): {', '.join(sorted(vs))}. Removing the launcher's default "
                f"accomplishes nothing if a wrapper answers the question for you."
            )
    for inv, allowed in sorted(CALLER_DEFAULT_BASELINE.items()):
        actual = len(found.get(inv, ()))
        if actual < allowed:
            failures.append(
                f"BASELINE STALE {inv} now has {actual} caller defaults, baseline says {allowed}. "
                f"Lower it to {actual} so the debt cannot grow back."
            )

    # 4. THE CLASS MUST BE TRUE OF THE CODE, not merely written down.
    # dynamic_required_by_gate must ACTUALLY appear in the launcher's gate, not merely be listed.
    # tutor-grok stripped LR from the gate's for-loop, left a forward-read so it still looked
    # "read", and this check stayed green because nothing verified the gate contained it.
    # EVERY for-loop, not the first. A decoy `for _v in LR ...` earlier in the file satisfied the
    # old single-match search while the real gate no longer checked it.
    launcher_src = "\n".join(
        l for l in open(LAUNCHER).read().split("\n") if not l.strip().startswith("#")
    )
    gate_loop = " ".join(re.findall(r"for _v in ([A-Z_ ]+); do", launcher_src))
    for v in reg.get("dynamic_required_by_gate") or []:
        in_gate = v in gate_loop.split()
        in_pair = v in ("MODEL_PATH", "RESUME_DELTA") and "MODEL_PATH-or-RESUME_DELTA" in launcher_src
        if not (in_gate or in_pair):
            failures.append(
                f"GATE MISSING   {v} is registered dynamic_required_by_gate but the launcher's gate "
                f"does not check it. Being read somewhere is not being enforced."
            )

    for v in reg.get("dynamic_required_when_cpt") or []:
        if v not in forms["required_conditional"] and v not in forms["required"]:
            failures.append(
                f"CLASS UNTRUE   {v} is registered dynamic_required_when_cpt but no ${{{v}:?}} "
                f"enforces it in the CPT branch."
            )
    for v in reg.get("dynamic_required") or []:
        if v in forms["required"]:
            continue
        if v in forms["required_conditional"]:
            failures.append(
                f"CONDITIONAL    {v} is registered dynamic_required but its ${{{v}:?}} is nested "
                f"inside a conditional, so it fires for some callers and not others. Either enforce "
                f"it unconditionally or register it under a mode-scoped class."
            )
        else:
            failures.append(
                f"CLASS UNTRUE   {v} is registered dynamic_required but the code does not enforce "
                f"it with ${{{v}:?...}}. A class nobody enforces is a comment."
            )
    for v in reg.get("invariant") or []:
        if v in forms["assigned"] or v in forms["optional"]:
            failures.append(
                f"CLASS UNTRUE   {v} is registered invariant but the code gives it a default. An "
                f"invariant is set unconditionally so a caller cannot silently get another value."
            )

    external_total = sum(len(variables or []) for variables in external.values())
    total = sum(len(v or []) for v in launcher_reg.values()) + external_total
    print(f"launcher: {LAUNCHER}")
    print(f"registry: {REGISTRY}   variables classified: {total}   code reads: {len(names)}")
    print(f"  dynamic_required enforced by ${{VAR:?}}: {len(forms['required'])}")
    print(f"  invariants set unconditionally         : {len(forms['unconditional'])}")
    print(f"  ${{VAR:=}} assign-defaults remaining      : {len(forms['assigned'])}")
    print(f"  external variables verified              : {external_total}")

    if not failures:
        print("clean: the registry describes the launcher, and every class is true of the code")
        return 0
    print()
    for f in failures:
        print(f"  {f}")
    print()
    print("=== A DYNAMIC VARIABLE HAS BEEN LOST, OR THE REGISTRY HAS ROTTED ===")
    print("This check exists because a launcher silently assigned TOTAL_STEPS=3000 and MAX_SEQ=2560")
    print("for months. Classify the variable or remove it; do not weaken the check to pass.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
