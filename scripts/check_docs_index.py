#!/usr/bin/env python3
"""Every document is reachable from README, and every document that asserts PROCESS says who wins.

WHY THIS EXISTS. Measured 2026-08-18: this repository tracked 62 markdown files and README named
THREE of them. Fifty-nine documents were unreachable from the entry point — not deleted, not wrong,
just invisible. A reader who starts at README cannot find them, and a reader who finds one by
grepping has no way to know whether it is current.

The same measurement found FOUR documents still asserting a production entrypoint after a "one door"
consolidation had supposedly finished, including one pointing at a quarantined trainer. The
consolidation fixed the three documents its author already knew about. The class was never checked.
That is the defect this script exists to make impossible to repeat.

TWO RULES, both mechanical:

  REACHABLE   every tracked .md is named in docs/INDEX.md, and README links docs/INDEX.md. A
              document nobody can navigate to will drift, and its drift will be discovered by
              someone acting on it.

  ATTRIBUTED  every document that names a production ENTRYPOINT carries the authority pointer, so a
              reader who arrives mid-document learns which file wins before they act on this one.
              Naming an entrypoint is what makes a document process-asserting; citing a measurement
              is not.

WHAT THIS DOES NOT DO. It does not judge whether a document is correct, current, or well written.
It checks that it can be found and that it does not silently compete for authority. Those are the
two failures that cost this repository real time.

    python3 scripts/check_docs_index.py [--fix-index]

`--fix-index` regenerates docs/INDEX.md from the tree. Without it the script only reports.
"""
import argparse
import os
import re
import subprocess
import sys

INDEX = "docs/INDEX.md"
README = "README.md"
AUTHORITY = "PRODUCTION AUTHORITY"

# Naming one of these is what makes a document process-asserting rather than measurement-recording.
ENTRYPOINT_TOKENS = [
    "run_till_done_v3",
    "run_4node_27b_cpt.sh",
    "train_fsdp_v3",
    "launch_cpt_qwen36_27b_fsdp.sh",
    "post_cpt_pipeline.sh",
    "bake_27b.sh",
]

# Documents whose entire job is to discuss the entrypoint problem. Exempting them is not a
# loophole: each one either IS the authority or explicitly defers to it in its first lines.
EXEMPT = {
    "CLAUDE.md",            # carries the authority section itself
    "RECIPES.md",           # deprecated in place, header points at the authority
    "docs/INDEX.md",        # the index names everything by construction
    "docs/REMEDIATION_PLAN.md",
}



# ── DEAD REFERENCE CHECK ──────────────────────────────────────────────────────
# A document that cites a path which no longer exists is not merely untidy: it sends a reader to
# something that is not there, and the reader cannot tell whether the doc is stale or the file was
# moved. Measured 2026-08-18: 51 dead in-repo citations across 12 documents, out of 143 citations.
#
# The debt is FROZEN, not ignored. Existing offenders are recorded in DEAD_REF_BASELINE with their
# count. The check fails when a document exceeds its baseline or a new document joins the list, so
# the debt can shrink and cannot grow. Lower a number when you fix references; never raise one.
#
# Citations into OTHER repositories are counted separately. They are not stale — they are
# cross-repo pointers, and per CLAUDE.md a pointer into a private repo is its own defect class.
DEAD_REF_BASELINE = {
    "README.md": 1,
    "careers-qwen/CPT_REFRESH_RECIPE_v0.9.md": 1,
    "careers-qwen/SFT_RECIPE_RECONCILE_v1.md": 1,
    "careers-qwen/TAEY_TRAINING_DOCTRINE.md": 1,
    "careers-qwen/TRAINING_BACKLOG.md": 5,
    "careers-qwen/data/TRAINING_BACKLOG.md": 1,
    "dense-9b/plans/build_launcher_spec.md": 1,
    "docs/METRICS_PROVENANCE.md": 33,
    "docs/SPARK_TOPOLOGY.md": 1,
    "docs/postmortem/PART1_measured_timeline.md": 2,
    "docs/postmortem/RUN_STATE_cpt_qwen38_v3.md": 1,
    "docs/proof_of_run/nccl_synth_probe_results.md": 3,
}

FOREIGN_REPO_PREFIXES = (
    "the-conductor/", "treasurer/", "isma-core/", "taeys-hands/", "apply-machine/",
    "claude-code-fleet", "doge/", "OPERATOR_HOME", "data/corpus/",
)

PATH_CITATION = re.compile(
    r"(?<![\w/])((?:[\w.-]+/)+[\w.-]+\.(?:py|sh|md|yml|yaml|json|jinja))"
)


def dead_references(paths, tracked):
    """Per-document count of cited repo paths that resolve to nothing.

    Resolves each citation BOTH repo-root-relative and relative to the citing document, because a
    naive root-only check reports a doc's own working relative links as broken -- which it did on
    the first run of this very function, and would have reported 19 stale documents instead of 12.
    """
    out = {}
    for d in paths:
        if "/system_prompt/versions/" in d:
            continue  # historical snapshots cite the past on purpose
        base = os.path.dirname(d)
        text = open(d, encoding="utf-8", errors="replace").read()
        missing = set()
        for m in set(PATH_CITATION.findall(text)):
            if m.startswith(("http", "<", ".")):
                continue
            if any(f in m for f in FOREIGN_REPO_PREFIXES):
                continue
            if m in tracked or os.path.exists(m) or os.path.exists(os.path.normpath(os.path.join(base, m))):
                continue
            missing.add(m)
        if missing:
            out[d] = missing
    return out


def tracked_markdown():
    out = subprocess.run(["git", "ls-files", "*.md"], capture_output=True, text=True, check=True)
    return sorted(p for p in out.stdout.split("\n") if p.strip())


def build_index(paths):
    by_dir = {}
    for p in paths:
        by_dir.setdefault(os.path.dirname(p) or ".", []).append(p)
    lines = [
        "# Document index",
        "",
        "Every markdown file tracked in this repository, so none is lost. Generated by",
        "`scripts/check_docs_index.py --fix-index`; CI fails if it is stale.",
        "",
        "**Authority order for anything about HOW TO RUN production:** the PRODUCTION AUTHORITY",
        "section of `CLAUDE.md` wins, then `PRODUCTION_MANIFEST.yml`, then `README.md`, then",
        "`careers-qwen/RUNBOOK_CPT_SFT_BAKE.md`. Any other document is a record, not an instruction.",
        "",
    ]
    for d in sorted(by_dir):
        lines.append(f"## `{d}`")
        lines.append("")
        for p in sorted(by_dir[d]):
            lines.append(f"- [{os.path.basename(p)}]({os.path.relpath(p, 'docs')})")
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix-index", action="store_true")
    a = ap.parse_args()

    paths = tracked_markdown()
    failures = []

    tracked = set(
        subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True).stdout.split("\n")
    )
    dead = dead_references(paths, tracked)
    for doc, missing in sorted(dead.items()):
        allowed = DEAD_REF_BASELINE.get(doc, 0)
        if len(missing) > allowed:
            shown = ", ".join(sorted(missing)[:3])
            failures.append(
                f"DEAD REFS      {doc} cites {len(missing)} path(s) that do not exist "
                f"(baseline {allowed}): {shown}"
            )
    for doc, allowed in sorted(DEAD_REF_BASELINE.items()):
        actual = len(dead.get(doc, ()))
        if actual < allowed:
            failures.append(
                f"BASELINE STALE {doc} now has {actual} dead refs, baseline says {allowed}. "
                f"Lower it to {actual} so the debt cannot silently grow back."
            )

    desired = build_index(paths)
    if a.fix_index:
        os.makedirs("docs", exist_ok=True)
        open(INDEX, "w").write(desired)
        print(f"wrote {INDEX} covering {len(paths)} documents")
    else:
        current = open(INDEX).read() if os.path.isfile(INDEX) else ""
        if current != desired:
            missing = [p for p in paths if os.path.basename(p) not in current]
            failures.append(
                f"INDEX STALE    {INDEX} does not match the tree "
                f"({len(missing)} document(s) absent from it). Run: "
                f"python3 scripts/check_docs_index.py --fix-index"
            )

    readme = open(README).read() if os.path.isfile(README) else ""
    if "docs/INDEX.md" not in readme:
        failures.append(f"UNLINKED INDEX {README} does not link {INDEX}; the index is unreachable")

    for p in paths:
        if p in EXEMPT:
            continue
        text = open(p, encoding="utf-8", errors="replace").read()
        named = [t for t in ENTRYPOINT_TOKENS if t in text]
        if named and AUTHORITY not in text:
            failures.append(
                f"UNATTRIBUTED   {p} names {', '.join(sorted(set(named))[:3])} but never points at "
                f"the {AUTHORITY} section, so a reader cannot tell it does not decide the entrypoint"
            )

    print(f"documents: {len(paths)}   index: {INDEX}")
    if not failures:
        print("clean: every document is indexed, and every process-asserting document names its authority")
        return 0
    print()
    for f in failures:
        print(f"  {f}")
    print()
    print("=== DOCUMENTATION CAN SEND A READER THE WRONG WAY ===")
    print("A document that is unreachable will drift. A document that names an entrypoint without")
    print("naming its authority competes with the real one. Both cost this repo days.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
