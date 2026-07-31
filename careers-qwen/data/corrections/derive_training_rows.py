#!/usr/bin/env python3
"""Derive TRAINING rows from curated correction rows — teaching the right way only.

BINDING CONSTRAINT (Jesse, 2026-07-21): "100% they should not be taught about failures just how to
do it right next time based on the available tech we have and the hardware that you are on."

So a training row is `situation -> right_way` and NOTHING ELSE. The failure record
(agent_action / correction_verbatim / cost) stays in the curation file for human audit and never
reaches training text.

This is not a filter over the old fields — `correct_action` as originally authored carries failure
RESIDUE ("that was ~30 lines and ran 20x faster", "X had already flagged this as vestigial"), which
is comparative to an incident and therefore still teaches the incident. Rows must be authored
fresh as standalone guidance.

The RESIDUE_RE gate below is the mechanical enforcement: any derived row whose text trips it is
REJECTED, not silently emitted. We are building a process enforcer; it should hold for our own data.
"""
import hashlib, json, os, re, sys

# Vocabulary that indicates the text is narrating a failure rather than teaching a practice.
RESIDUE_RE = re.compile(
    r"\b(fail(ed|ure|s)?|wrong|mistakes?\b|error|broke|broken|bug|defect|instead of|"
    r"should have|shouldn'?t have|don'?t repeat|wasted?|lost \d|cost \d|"
    r"had already flagged|turned out|rediscover|again\b|"
    r"\d+\s*(days?|hours?)\s*(lost|wasted|of)|never called|vestigial|"
    r"i did|we did|my |accidentally|silently (trained|produced))", re.I)


def _h(*p):
    return hashlib.sha256("|".join(str(x) for x in p).encode()).hexdigest()[:16]


def derive(rows):
    """curated correction rows -> training rows. Returns (emitted, rejected)."""
    out, rejected = [], []
    authored = {r.get("class") for r in rows if r.get("situation") and r.get("right_way")}
    for r in rows:
        sit, right = r.get("situation"), r.get("right_way")
        if not sit or not right:
            # One practice row per CLASS is by design — recurrence lives in meta, and emitting the
            # same lesson N times would just weight it by how often we happened to fail. Only a
            # class with NO authored practice text anywhere is a real gap.
            if r.get("class") in authored:
                continue
            rejected.append((r.get("class"), "no practice text authored for this class"))
            continue
        blob = f"{sit}\n{right}"
        hit = RESIDUE_RE.search(blob)
        if hit:
            rejected.append((r.get("class"), f"FAILURE RESIDUE: '{hit.group(0)}'"))
            continue
        out.append({
            "schema": "operator_practice_v1",
            "messages": [
                {"role": "user", "content": sit},
                {"role": "assistant", "content": right},
            ],
            "meta": {
                "class": r.get("class"),
                "recurrence": r.get("recurrence", 1),
                "derived_from": r.get("provenance_hash"),
                "teaches": "right-way-only (no failure narrative — Jesse constraint 2026-07-21)",
            },
            "provenance_hash": _h(sit, right),
        })
    return out, rejected


def derive_spec(rows):
    """spec_knowledge_v1 authored rows -> spec training rows. Returns (emitted, rejected).

    Per TAEY_TRAINING_DOCTRINE §1: the infra-bug-fixed row type. Emitted training text is
    `spec` + `correct_output` ONLY. `story_behind_the_diff` is curation metadata and is NEVER
    emitted (that is what holds the negation hazard at zero). The emitted text is still
    residue-gated — a spec that narrates the failure it came from is malformed.

    The trainable shape is an SFT pair with a NEUTRAL, synthesized probe derived from `subsystem`
    (not a fabricated incident) so the declarative spec trains through the existing messages[]
    pipeline. Whether spec-knowledge should instead ride the CPT slice, and its dose, are the
    Chats' mixture call — this defines the faithful default emission, not the dose.
    """
    out, rejected = [], []
    for r in rows:
        if r.get("schema") != "spec_knowledge_v1":
            continue
        sub, spec, correct = r.get("subsystem"), r.get("spec"), r.get("correct_output")
        if not spec or not correct:
            rejected.append((sub or "?", "missing spec or correct_output"))
            continue
        answer = f"{spec}\n\n{correct}"
        hit = RESIDUE_RE.search(answer)   # story_behind_the_diff deliberately NOT scanned/emitted
        if hit:
            rejected.append((sub or "?", f"FAILURE RESIDUE in emitted text: '{hit.group(0)}'"))
            continue
        probe = f"How does {sub} work, and what does correct output look like?"
        out.append({
            "schema": "spec_knowledge_v1",
            "messages": [
                {"role": "user", "content": probe},
                {"role": "assistant", "content": answer},
            ],
            "meta": {
                "class": "spec-knowledge",
                "subsystem": sub,
                "source": r.get("meta", {}).get("source") or r.get("source"),
                "teaches": "subsystem-spec-only (no failure narrative — story_behind_the_diff never emitted)",
                "probe_is_synthesized": True,
            },
            "provenance_hash": _h(sub, spec, correct),
        })
    return out, rejected


def _data_root():
    """Training DATA lives in the private governed store (Jesse one-store ruling; Q1/Q2 gate
    2026-07-23). Tooling (.py) stays in this public repo; every .jsonl read/write resolves under
    TRAINING_DATA_ROOT. Fail-loud — a silent default would re-create data in the public tree."""
    root = os.environ.get("TRAINING_DATA_ROOT")
    if not root or not os.path.isdir(root):
        raise SystemExit(
            "TRAINING_DATA_ROOT is unset or not a directory. Point it at the governed store "
            "(<MIRA_HOME>/treasurer/foundations/careers/training_data/careers_qwen). "
            "See careers-qwen/TAEY_TRAINING_DOCTRINE.md."
        )
    return root


def main():
    d = os.path.join(_data_root(), "corrections")
    src = os.path.join(d, "corrections_seed_v1.jsonl")
    rows = [json.loads(l) for l in open(src)]
    out, rejected = derive(rows)
    dst = os.path.join(d, "practice_rows_v1.jsonl")
    with open(dst, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"source curated rows : {len(rows)}")
    print(f"EMITTED training rows: {len(out)} -> {os.path.basename(dst)}")
    print(f"REJECTED             : {len(rejected)}")
    for c, why in rejected:
        print(f"   - [{c}] {why}")

    # --- spec_knowledge_v1 (infra-bug-fixed row type) ---
    # Source rows (authored, with story_behind_the_diff) live in ../spec_knowledge/*.jsonl.
    # The emitted, residue-gated, story-stripped file is spec_train_rows_v1.jsonl (skipped as input).
    spec_dir = os.path.join(_data_root(), "spec_knowledge")
    EMITTED = "spec_train_rows_v1.jsonl"
    if os.path.isdir(spec_dir):
        spec_src = []
        for fn in sorted(os.listdir(spec_dir)):
            if not fn.endswith(".jsonl") or fn == EMITTED:
                continue
            for line in open(os.path.join(spec_dir, fn)):
                line = line.strip()
                if line:
                    spec_src.append(json.loads(line))
        s_out, s_rej = derive_spec(spec_src)
        with open(os.path.join(spec_dir, EMITTED), "w") as f:
            for r in s_out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nspec_knowledge source: {len(spec_src)}")
        print(f"EMITTED spec rows    : {len(s_out)} -> spec_knowledge/{EMITTED}")
        print(f"REJECTED             : {len(s_rej)}")
        for sub, why in s_rej:
            print(f"   - [{sub}] {why}")

    # A corpus that emits nothing is honest, not broken — it means the rows need authoring.
    return 0


if __name__ == "__main__":
    sys.exit(main())
