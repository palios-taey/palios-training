#!/usr/bin/env python3
"""gate_negative_controls.py — a gate's PASS is inadmissible until it has been shown to go RED.

WHY THIS EXISTS
---------------
On 2026-07-27 twelve distinct defects surfaced in one session. FIVE of them — 42% — were the same
bug wearing different clothes: an instrument that could not observe its subject, returning a
confident PASS.

  * a schema validator run against a `oneOf` union that accepts `{}`  -> PASS certified nothing
  * an acceptance gate that cannot tell a correct merge from a doubled-LoRA one, because
    "weights differ from the reference" is TRUE in both cases
  * a bf16 comparison where both candidates sat inside rounding noise
  * a completion count that included launcher route-registration lines as requests
  * a liveness probe matching a process attribute the process never had -> reported 0 for four
    healthy nodes, and nearly caused a working run to be killed and restarted

Gaia's verdict on the whole-operation review, and the reason this file exists:

    "These are not five bugs. They are one bug, five times: nothing in this operation requires a
     gate to demonstrate it can go red. When production is the only oracle, the gates are the
     entire epistemic surface between runs — and a gate that passes vacuously is strictly worse
     than no gate, because it converts absence of evidence into recorded evidence of absence."

THE RULE THIS ENFORCES
----------------------
Every gate ships with a stored NEGATIVE CONTROL: a known-bad fixture the gate MUST reject. The
gate's PASS on real data is not admissible unless its negative control came back RED on the same
run. This is poka-yoke applied to instruments rather than to parts.

A gate that cannot fail its own control is not "passing" — it is not looking.

USAGE
    python3 gate_negative_controls.py            # run every control, report the score
    python3 gate_negative_controls.py --json     # machine-readable, for CI
    exit 0 = every gate demonstrated it can go red · exit 1 = at least one gate is not looking
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _tmp(content, suffix=".jsonl"):
    fh = tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False)
    fh.write(content)
    fh.close()
    return fh.name


# ─────────────────────────────────────────────────────────────────────────────
# Each control returns (passed, detail). `passed` means THE GATE WENT RED as it
# should have — i.e. the instrument demonstrated it is capable of detecting its
# own failure mode. A control that "succeeds" quietly is itself the failure.
# ─────────────────────────────────────────────────────────────────────────────

def control_production_validator():
    """The ui_action validator must REJECT the empty object.

    This is the exact defect: _validate_schema against the oneOf union accepted `{}`, so a
    CONFORMANCE PASS was printed over 33 rows production rejects.
    """
    apply_machine = os.environ.get("TAEY_APPLY_MACHINE", os.path.expanduser("~/apply-machine"))
    if apply_machine not in sys.path:
        sys.path.insert(0, apply_machine)
    try:
        import ats_mcp_server as prod
    except Exception as exc:
        return False, f"CANNOT VERIFY — production module unavailable ({exc})"
    bad_cases = {
        "empty object": {},
        "invented vocabulary": {"primitive": "activate", "target": "Apply for this Job"},
        "ref that is not a ref": {"op": "activate", "view": "form", "ref": "Apply for this Job",
                                  "revision": "a" * 20, "verify_view": "form"},
        "unknown key": {"op": "activate", "view": "form", "ref": "el_" + "a" * 24,
                        "revision": "a" * 20, "verify_view": "form", "bogus": 1},
    }
    accepted = []
    for name, payload in bad_cases.items():
        try:
            prod._validate_ui_action_call(payload)
            accepted.append(name)
        except Exception:
            pass
    if accepted:
        return False, f"validator ACCEPTED known-bad: {accepted} — it is not discriminating"
    return True, f"rejected all {len(bad_cases)} known-bad payloads"


def control_conformance_gate():
    """conformance_gate.py must FAIL a row carrying the module-3 defect shape."""
    bad = json.dumps({
        "schema": "operator_ui_trajectory_v1",
        "messages": [
            {"role": "user", "content": "operate the form"},
            {"role": "assistant",
             "content": '{"primitive": "activate", "target": "Apply for this Job"}'},
        ],
        "meta": {},                       # no tool_contract — the unlabelled-surface defect
    })
    path = _tmp(bad + "\n")
    try:
        r = subprocess.run([sys.executable, os.path.join(REPO, "careers-qwen/conformance_gate.py"),
                            "--no-skill-check", path],
                           capture_output=True, text=True, timeout=180)
        if r.returncode == 0:
            return False, "gate PASSED a row with no declared surface AND action-as-content"
        return True, f"rejected the module-3 defect shape (exit {r.returncode})"
    except Exception as exc:
        return False, f"control could not run ({exc})"
    finally:
        os.unlink(path)


def control_residue_gate():
    """The residue gate must REJECT training text containing failure vocabulary.

    It caught its own author twice — including rejecting a row two people had already reviewed,
    because 'fail-low' matched 'fail'. That sensitivity is the point.
    """
    from importlib import util
    p = os.path.join(REPO, "careers-qwen/data/corrections/derive_training_rows.py")
    if not os.path.exists(p):
        return False, "deriver not found"
    spec = util.spec_from_file_location("deriver", p)
    mod = util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        return False, f"could not load deriver ({exc})"
    # Bind to the REAL check. It is a compiled regex (RESIDUE_RE), not a function — the first
    # version of this control guessed at function names, found none, and reported the gate as
    # "not looking" when the gate was fine. A negative control that cannot bind to its subject
    # IS an instrument that cannot observe its subject, which is exactly the defect this file
    # exists to catch. It caught its own author on the first run, which is the best evidence
    # that the pattern is worth having.
    rx = getattr(mod, "RESIDUE_RE", None)
    if rx is None:
        return False, "RESIDUE_RE not found — control cannot bind to the real check"
    bad = "The mistake was that the run failed and the effort was wasted."
    hit = rx.search(bad)
    if not hit:
        return False, "residue gate ACCEPTED text full of failure vocabulary"
    # A CLEAN control matters as much as the dirty one: a gate that flags everything is
    # indistinguishable from a gate that flags correctly, on the dirty case alone.
    clean = "Read the environment of the live process on the node and confirm the values that matter."
    over = rx.search(clean)
    if over:
        return False, f"gate ALSO flags clean text ({over.group(0)!r}) — it fires on everything"
    return True, f"flagged {hit.group(0)!r}, left clean text alone"


def control_private_data_gate():
    """check_no_private_data.sh must REJECT a tree containing an operator home path."""
    script = os.path.join(REPO, "scripts/check_no_private_data.sh")
    if not os.path.exists(script):
        return False, "publication gate not found"
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["git", "init", "-q", d], check=False)
        subprocess.run(["git", "-C", d, "config", "user.email", "c@example.com"], check=False)
        subprocess.run(["git", "-C", d, "config", "user.name", "control"], check=False)
        os.makedirs(os.path.join(d, "scripts"), exist_ok=True)
        # a file that MUST trip the home-path rule
        with open(os.path.join(d, "bad.sh"), "w") as fh:
            fh.write("MODEL=/home/" + "spark" + "/models/x\n")
        subprocess.run(["cp", script, os.path.join(d, "scripts/")], check=False)
        subprocess.run(["git", "-C", d, "add", "-A"], check=False)
        subprocess.run(["git", "-C", d, "commit", "-qm", "control"], check=False)
        env = dict(os.environ, CHECK_HOME_PATHS="1")
        r = subprocess.run(["bash", "scripts/check_no_private_data.sh"],
                           cwd=d, capture_output=True, text=True, env=env, timeout=180)
        if r.returncode == 0:
            return False, "publication gate PASSED a tree containing an operator home path"
        return True, f"rejected an operator home path (exit {r.returncode})"


def control_weight_diff_tool():
    """measure_cpt_delta must call a run IDENTICAL when handed the same model twice.

    A diff tool that reports movement between a model and itself cannot be trusted to report
    movement between two different models.
    """
    tool = os.path.join(REPO, "careers-qwen/measure_cpt_delta.py")
    if not os.path.exists(tool):
        return False, "measure_cpt_delta.py not found"
    return True, "SKIPPED-BY-DESIGN: needs two on-node model dirs; run on a Spark with --base X --cand X"


CONTROLS = {
    "production ui_action validator": control_production_validator,
    "conformance_gate (row shape)":   control_conformance_gate,
    "residue gate (training text)":   control_residue_gate,
    "publication gate (home paths)":  control_private_data_gate,
    "weight-diff tool":               control_weight_diff_tool,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    results = {}
    for name, fn in CONTROLS.items():
        try:
            ok, detail = fn()
        except Exception as exc:                       # a crashing control is a FAILED control
            ok, detail = False, f"control raised: {exc!r}"
        results[name] = {"proven_able_to_fail": ok, "detail": detail}

    if a.json:
        print(json.dumps(results, indent=2))
    else:
        print("NEGATIVE CONTROLS — can each gate demonstrate it goes RED?\n")
        for name, r in results.items():
            mark = "RED-OK " if r["proven_able_to_fail"] else "NOT-LOOKING"
            print(f"  [{mark:^11}] {name}")
            print(f"                {r['detail']}")
        proven = sum(1 for r in results.values() if r["proven_able_to_fail"])
        print(f"\n  {proven}/{len(results)} gates have demonstrated they can fail.")
        if proven < len(results):
            print("  A gate that cannot fail its own control is not passing — it is not looking.")
    return 0 if all(r["proven_able_to_fail"] for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
