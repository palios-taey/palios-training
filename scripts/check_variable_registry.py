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

IT ALSO CHECKS THE CLASS IS TRUE, not merely present:
    dynamic_required          must be enforced by `${VAR:?}` in the code
    invariant                 must be an unconditional assignment, never `:=` or `:-`
    no `${VAR:=...}` may exist at all -- that syntax ASSIGNS, which is the original defect

    python3 scripts/check_variable_registry.py

Exit 0 = the registry describes the launcher. Exit 1 = it does not, and every disagreement prints.
"""
import os
import re
import sys

LAUNCHER = "dense-9b/recipes/run_4node_27b_cpt.sh"
REGISTRY = "dense-9b/recipes/VARIABLES.yml"

# Read by the launcher's own gate rather than by a ${VAR:?} form, so they are enforced but the
# regex below cannot see them. Kept explicit rather than silently excluded.
GATE_ENFORCED = {"LR", "WARMUP_STEPS", "MODEL_PATH", "RESUME_DELTA"}


def code_reads(path):
    """Every environment variable the script actually references, comments excluded."""
    src = "\n".join(l for l in open(path).read().split("\n") if not l.strip().startswith("#"))
    names = set(re.findall(r"\$\{?([A-Z][A-Z0-9_]{2,})", src))
    forms = {
        "required": set(re.findall(r"\$\{([A-Z][A-Z0-9_]+):\?", src)),
        "assigned": set(re.findall(r"\$\{([A-Z][A-Z0-9_]+):=", src)),
        # `${VAR:-}` with an EMPTY default is a null-check idiom, not a default value.
        # ADAFACTOR_DOSE_LOG is set unconditionally at :240 and then guarded at :241 with
        # `[ -n "${ADAFACTOR_DOSE_LOG:-}" ]` before being forwarded. Counting that guard as a
        # default reported a true invariant as misclassified -- a false positive from this
        # very check, caught on its first run.
        "optional": set(re.findall(r"\$\{([A-Z][A-Z0-9_]+):-[^}\s]", src)),
        "unconditional": set(re.findall(r"^([A-Z][A-Z0-9_]{2,})=", src, re.M)),
    }
    return names, forms


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
    classified = {}
    for cls, items in reg.items():
        for v in (items or []):
            classified[v] = cls

    names, forms = code_reads(LAUNCHER)
    failures = []

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
    for v, cls in sorted(classified.items()):
        if v not in names and v not in GATE_ENFORCED:
            failures.append(
                f"STALE ENTRY    {REGISTRY} lists {v} under '{cls}' but {LAUNCHER} never reads it."
            )

    # 4. THE CLASS MUST BE TRUE OF THE CODE, not merely written down.
    for v in reg.get("dynamic_required") or []:
        if v not in forms["required"]:
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

    total = sum(len(v or []) for v in reg.values())
    print(f"launcher: {LAUNCHER}")
    print(f"registry: {REGISTRY}   variables classified: {total}   code reads: {len(names)}")
    print(f"  dynamic_required enforced by ${{VAR:?}}: {len(forms['required'])}")
    print(f"  invariants set unconditionally         : {len(forms['unconditional'])}")
    print(f"  ${{VAR:=}} assign-defaults remaining      : {len(forms['assigned'])}")

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
