#!/usr/bin/env python3
"""Build REASONED UI rows: the steps that should have been gone through, then the action.

Jesse 2026-07-21: "is this enough or do we also have to train thinking? Like what steps they should
have gone through?"

THE GAP THIS FIXES: `ui_action_rows_v1` is state -> JSON with no reasoning. Observed directly when
probing ep3: it reasons in prose BEFORE emitting its answer. Training action-only would teach it to
suppress that deliberation and jump to an answer — fine on states it has seen, poor on novel ones,
because the procedure is what generalises and a bare state->action mapping only memorises.

WHY THIS IS SAFE WHERE SELF-REPORTS ARE NOT: round-2 established that a model's account of its own
reasoning is unreliable and training on it manufactures confident wrong rules. This trains the
PRESCRIBED PROCEDURE — authored from the operating discipline and the row's own observed fields —
not the model's narration of itself. Same principle as the correction corpus: teach the right way,
never the model's story about it.

Emitted as a SEPARATE lane so its dose and its eval can be set independently of the action lane.
"""
import argparse, hashlib, json


# Which surface a view belongs to decides the whole vocabulary.
SURFACE_A_VIEWS = {"navigation", "form", "form/options", "file_dialog", "review"}

# Role -> the precondition that decides the primitive. Authored from the operating discipline,
# not inferred from the model.
ROLE_RULE = {
    "entry": "an entry accepts typed text, but only once it holds focus",
    "push button": "a button is pressed by its accessible name",
    "combo box": "a plain combo is opened and chosen from, never typed into",
    "link": "a link is activated by its accessible name",
    "check box": "a checkbox is toggled by activation, not by writing a value",
}


def reason_steps(r, act):
    obs = r["meta"]
    view = obs.get("view", "")
    prim = act.get("primitive", "")
    name = None
    # recover the element name from the prompt line we authored
    for line in r["messages"][0]["content"].splitlines():
        if line.startswith("Visible element:"):
            name = line.split(":", 1)[1].strip()
    role = ""
    if name and "(" in name:
        role = name.rsplit("(", 1)[1].rstrip(")").strip()

    surface = "A (job application / ATS)" if view in SURFACE_A_VIEWS else "B (LinkedIn / SalesNav / Upwork)"
    steps = [f"Surface: this is a {view} view, so surface {surface} and its vocabulary applies."]
    if name and name != "(no element mapped)":
        steps.append(f"Target: the view exposes {name}.")
    else:
        steps.append("Target: no element is mapped in this view, so no named action is available.")
    if role and role in ROLE_RULE:
        steps.append(f"Precondition: {ROLE_RULE[role]}.")
    if prim == "write":
        steps.append("Because the target is a text entry, it must hold focus before the write lands, "
                     "and the value is read back afterwards to confirm it took.")
    elif prim in ("type_filter_select", "pointer_focus_type_select"):
        steps.append("Because the control filters its options, typing narrows the list and the "
                     "selection is what commits the value.")
    elif prim == "escape":
        steps.append("The view is not the one expected, so the correct move is to leave it and "
                     "return to the last known good state rather than navigate deeper.")
    elif prim in ("pointer_activate",):
        steps.append("No accessible name is exposed for this control, so it is pressed at its "
                     "coordinates rather than by name.")
    steps.append(f"Action: {prim}.")
    steps.append(f"Expected result: {act.get('expect')}. Verify this before advancing.")
    return steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/ui/ui_action_rows_v1.jsonl")
    ap.add_argument("--out", default="data/ui/ui_reasoned_rows_v1.jsonl")
    a = ap.parse_args()

    src = [json.loads(l) for l in open(a.src)]
    out = []
    for r in src:
        act = json.loads(r["messages"][1]["content"])
        steps = reason_steps(r, act)
        body = "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))
        assistant = f"{body}\n\n{json.dumps(act, ensure_ascii=False)}"
        user = r["messages"][0]["content"].replace(
            'Emit the next action as a single JSON object with keys '
            '"primitive", "target", and "expect". Emit nothing else.',
            "Work through the steps, then emit the action as a single JSON object on the final line.")
        out.append({
            "schema": "operator_ui_reasoned_v1",
            "messages": [{"role": "user", "content": user},
                         {"role": "assistant", "content": assistant}],
            "meta": {**r["meta"],
                     "teaches": "the PRESCRIBED procedure then the action — not the model's own narration",
                     "reasoning_source": "authored from the operating discipline + this row's observed fields"},
            "provenance_hash": hashlib.sha256(assistant.encode()).hexdigest()[:16],
        })

    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"EMITTED {len(out)} reasoned rows -> {a.out}")
    if out:
        print("\nsample:")
        print(out[0]["messages"][1]["content"][:700])


if __name__ == "__main__":
    main()
