#!/usr/bin/env python3
"""Convert careers_ui_training_pair_v1 -> {messages, meta} training rows.

WHY THIS SHAPE: the round-1 consult named our core error — ep3 can accurately DESCRIBE taeys-hands
and has never been trained to USE it. So the assistant target here is the EXACT structured action
the primitives layer consumes, not prose about what one might do. CLARITY's L1 lane calls this
(instruction, tool_call) pairs; HORIZON's operator_mission_v1 wraps it as
observed -> action -> expected postcondition.

The postcondition is trained as part of the answer deliberately: stating what the action SHOULD
produce is what makes the next step verifiable. An operator that acts without predicting the result
cannot tell success from drift.

READ-ONLY over the source bundle. Emits, does not tokenize, does not decide dose.
"""
import argparse, os, hashlib, json, os


def render_post(post):
    """A postcondition may be a dict (copyback/state). Render it readably — a raw dict in the
    prompt is unreadable AND, for write rows, leaks the literal answer."""
    if isinstance(post, dict):
        if "copyback_length" in post:
            return "the field holds the written value (verified by copy-back)"
        if "state" in post:
            return str(post.get("reason") or post.get("state"))
        if "value" in post:
            return "the field holds the intended value"
        return "; ".join(f"{k}={v}" for k, v in post.items())
    return str(post or "")


def to_messages(r, ident_carry=""):
    obs = r.get("observed") or {}
    ident = obs.get("application_identity", "") or ident_carry
    name = obs.get("name", "")
    role = obs.get("role", "")
    view = r.get("view", "")
    post = render_post(r.get("postcondition"))
    act = r.get("action") or {}

    seen = f'"{name}" ({role})' if name else "(no element mapped)"
    # The GOAL states the objective and must NEVER contain the literal answer. Deriving it from the
    # postcondition leaked the value on 14/41 rows (write, type_filter_select) — the model was
    # copying the answer out of the question rather than deciding the action.
    if name and name != "(no element mapped)":
        goal = f"advance the application by handling {name}"
    else:
        goal = "recover to a known-good state and continue the application"
    user = (
        f"You are operating a real application UI through your own hands.\n"
        f"Application: {ident or '(identity not exposed on this step)'}\n"
        f"View: {view}\n"
        f"Visible element: {seen}\n"
        f"Goal for this step: {goal}\n\n"
        f"Emit the next action as a single JSON object naming the primitive, its required keys, "
        f'and "expect" (what the screen should show after it succeeds). Emit nothing else.'
    )
    # The action vocabulary is HETEROGENEOUS — each primitive carries its own keys
    # (activate->target, key->keys, write->value/source, escape->reason). Requiring "target"
    # discarded 37 of 43 rows on the first pass. Emit the action as authored, plus the expected
    # postcondition, so the trained vocabulary matches what the primitives layer actually accepts.
    payload = {k: v for k, v in act.items() if v is not None}
    payload["expect"] = post
    assistant = json.dumps(payload, ensure_ascii=False)
    return user, assistant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(
        os.environ.get("TAEY_APPLY_MACHINE", os.path.expanduser("~/apply-machine")),
        "bundles/trm_labs_ai_research_engineer_79afbde9/ui_training_pairs.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--synthetic-from", type=int, default=40,
                    help="seq >= this was authored rather than captured (treasurer-codex 40-43)")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.src)]
    out, skipped, blocked = [], [], []
    ident_carry = ""
    for r in rows:
        act = r.get("action") or {}
        if not act.get("primitive"):
            skipped.append((r.get("seq"), "no primitive"))
            continue
        # Jesse's constraint: teach the right way, never the failure. A row whose postcondition
        # records a BLOCKED outcome is a failure record — its lesson belongs in the practice
        # corpus as correct guidance, not here as an action target.
        post_r = r.get("postcondition")
        if isinstance(post_r, dict) and str(post_r.get("state", "")).lower() == "blocked":
            blocked.append((r.get("seq"), post_r.get("reason")))
            continue
        ident_carry = (r.get('observed') or {}).get('application_identity') or ident_carry
        u, s = to_messages(r, ident_carry)
        seq = r.get("seq", 0)
        out.append({
            "schema": "operator_ui_action_v1",
            "messages": [{"role": "user", "content": u},
                         {"role": "assistant", "content": s}],
            "meta": {
                "seq": seq,
                "view": r.get("view"),
                "primitive": act.get("primitive"),
                # provenance matters: 4 of 43 were authored, not captured from a real run
                "origin": "authored" if seq >= a.synthetic_from else "captured",
                "evidence": r.get("evidence"),
                "teaches": "USE the hands (emit an executable action), not describe them",
            },
            "provenance_hash": hashlib.sha256(
                f"{seq}|{act.get('primitive')}|{act.get('target')}".encode()).hexdigest()[:16],
        })

    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    import collections
    print(f"source rows      : {len(rows)}")
    print(f"EMITTED          : {len(out)} -> {a.out}")
    print(f"skipped (no primitive): {len(skipped)} {skipped if skipped else ''}")
    print(f"EXCLUDED as failure outcomes (blocked): {len(blocked)}")
    for sq, why in blocked:
        print(f"   - seq {sq}: {why}  -> lesson belongs in practice_rows, not as an action target")
    print(f"  by origin  : {dict(collections.Counter(r['meta']['origin'] for r in out))}")
    print(f"  by primitive: {dict(collections.Counter(r['meta']['primitive'] for r in out))}")
    print(f"  by view    : {dict(collections.Counter(r['meta']['view'] for r in out))}")
    if out:
        print("\nsample row:")
        print("  USER:", out[0]["messages"][0]["content"][:220].replace("\n", " | "))
        print("  ASST:", out[0]["messages"][1]["content"])


if __name__ == "__main__":
    main()
