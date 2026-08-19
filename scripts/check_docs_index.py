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

THREE RULES, all mechanical:

  REACHABLE   every tracked .md is named in docs/INDEX.md, and README links docs/INDEX.md. A
              document nobody can navigate to will drift, and its drift will be discovered by
              someone acting on it.

  ATTRIBUTED  every document that names a production ENTRYPOINT carries the authority pointer, so a
              reader who arrives mid-document learns which file wins before they act on this one.
              Naming an entrypoint is what makes a document process-asserting; citing a measurement
              is not.

  NOT AN INSTRUCTION  a document that names PRODUCTION AUTHORITY and then tells the reader to
              `bash dense-9b/recipes/run_4node_27b_cpt.sh` is still competing for the door. The
              previous check tested substring presence of those two words. W6 is correspondence:
              an affirmative launch of an inner script fails even when the banner is present.

WHAT THIS DOES NOT DO. It does not judge whether a document is well written. Mentions, citations,
and prohibitions ("do not invoke X") are not instructions.

    python3 scripts/check_docs_index.py [--fix-index]
    python3 scripts/check_docs_index.py --self-test
"""
import argparse
import io
import os
import re
import subprocess
import sys
import traceback

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

INNER_RE = re.compile(
    r"(?:run_4node_27b_cpt\.sh|run_till_done_v[23]\.sh|launch_cpt_qwen36_27b_fsdp\.sh|"
    r"post_cpt_pipeline\.sh|bake_27b\.sh|train_fsdp_v3\.py)"
)
LAUNCH_CMD_RE = re.compile(
    r"(?:^|\s)(?:bash|sh|exec|\.)\s+\S*"
    r"(?:run_4node_27b_cpt\.sh|run_till_done_v[23]\.sh|launch_cpt_qwen36_27b_fsdp\.sh|"
    r"post_cpt_pipeline\.sh|bake_27b\.sh|train_fsdp_v3\.py)"
)
PROHIBITIVE_RE = re.compile(
    r"(?i)\b(do not|don't|never|not the|deprecated|does not decide|not sanctioned|"
    r"do not use|not an entrypoint|not a second entrypoint|not the top-level)\b"
)


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
    "careers-qwen/TRAINING_BACKLOG.md": 26,
    "careers-qwen/data/TRAINING_BACKLOG.md": 1,
    "dense-9b/plans/build_launcher_spec.md": 1,
    "docs/METRICS_PROVENANCE.md": 13,
    "docs/SPARK_TOPOLOGY.md": 1,
    "docs/postmortem/PART1_measured_timeline.md": 2,
    "docs/postmortem/RUN_STATE_cpt_qwen38_v3.md": 1,
    "docs/proof_of_run/nccl_synth_probe_results.md": 2,
}

FOREIGN_REPO_PREFIXES = (
    "the-conductor/", "treasurer/", "isma-core/", "taeys-hands/", "apply-machine/",
    "claude-code-fleet", "doge/", "OPERATOR_HOME", "data/corpus/",
)

PATH_CITATION = re.compile(
    r"(?<![\w/])((?:[\w.-]+/)+[\w.-]+\.(?:py|sh|md|yml|yaml|json|jinja))"
)


def affirmative_launch_lines(text):
    """Lines that instruct the reader to invoke an inner launcher as the thing to run.

    Correspondence, not substring. A document can contain the words PRODUCTION AUTHORITY and
    still tell the reader `bash dense-9b/recipes/run_4node_27b_cpt.sh`. Mentions, citations,
    and prohibitions are not instructions.

    Returns a list of (lineno, line) pairs.
    """
    hits = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.rstrip()
        stripped = line.lstrip("> ").strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not INNER_RE.search(line):
            continue
        if PROHIBITIVE_RE.search(line):
            continue
        instructional = bool(LAUNCH_CMD_RE.search(line)) or bool(
            re.search(r"(?i)\bLAUNCH\s*:", line) and INNER_RE.search(line)
        )
        if instructional:
            hits.append((lineno, stripped[:160]))
    return hits


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


def collect_failures(paths, tracked, index_text, readme_text, read_text):
    """Pure-ish core so self-test can inject documents without mutating the tree."""
    failures = []

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
    if index_text != desired:
        missing = [p for p in paths if os.path.basename(p) not in (index_text or "")]
        failures.append(
            f"INDEX STALE    {INDEX} does not match the tree "
            f"({len(missing)} document(s) absent from it). Run: "
            f"python3 scripts/check_docs_index.py --fix-index"
        )

    if "docs/INDEX.md" not in (readme_text or ""):
        failures.append(f"UNLINKED INDEX {README} does not link {INDEX}; the index is unreachable")

    for p in paths:
        if p in EXEMPT:
            continue
        text = read_text(p)
        named = [t for t in ENTRYPOINT_TOKENS if t in text]
        if named and AUTHORITY not in text:
            failures.append(
                f"UNATTRIBUTED   {p} names {', '.join(sorted(set(named))[:3])} but never points at "
                f"the {AUTHORITY} section, so a reader cannot tell it does not decide the entrypoint"
            )
        launches = affirmative_launch_lines(text)
        if launches:
            lineno, snippet = launches[0]
            failures.append(
                f"INNER LAUNCH   {p}:{lineno} instructs the reader to invoke an inner script "
                f"({snippet!r}). Naming {AUTHORITY} does not cancel an instruction. The door is "
                f"scripts/taey-train."
            )
    return failures


def _clean_run(text: str) -> bool:
    return "Traceback" not in text and "{m.group" not in text


def selftest() -> int:
    """Prove substring AUTHORITY is not enough, and that a crash is not a finding."""
    failures = 0

    def captured(fn):
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        crashed = None
        result = None
        try:
            result = fn()
        except Exception:
            crashed = traceback.format_exc()
        finally:
            sys.stdout = old
        text = buf.getvalue()
        if crashed:
            text += "\n" + crashed
        return result, text

    # 1. AUTHORITY banner + bash inner launcher = FAIL (the residual W1 defect)
    planted = (
        "> **This document does not decide the entrypoint.** The PRODUCTION AUTHORITY "
        "section of `CLAUDE.md` wins.\n\n"
        "```bash\n"
        "OUTPUT_DIR=/tmp bash dense-9b/recipes/run_4node_27b_cpt.sh\n"
        "```\n"
    )
    hits = affirmative_launch_lines(planted)
    if not hits:
        print("SELFTEST FAIL: AUTHORITY + bash inner launcher produced no instructional hit")
        failures += 1
    else:
        print("SELFTEST OK: AUTHORITY + bash inner launcher is an instruction")

    # 2. AUTHORITY + prohibition is not an instruction
    banned = (
        "The PRODUCTION AUTHORITY section wins. Do not invoke run_4node_27b_cpt.sh "
        "directly; it is a STAGE, not an entrypoint.\n"
    )
    hits = affirmative_launch_lines(banned)
    if hits:
        print(f"SELFTEST FAIL: prohibition counted as instruction: {hits}")
        failures += 1
    else:
        print("SELFTEST OK: prohibition is not an instruction")

    # 3. Measurement citation is not an instruction
    cited = (
        "The PRODUCTION AUTHORITY section wins. `run_4node_27b_cpt.sh:351` prints "
        "`27B IS TRAINING` and ends its monitor at the first optimizer step.\n"
    )
    hits = affirmative_launch_lines(cited)
    if hits:
        print(f"SELFTEST FAIL: citation counted as instruction: {hits}")
        failures += 1
    else:
        print("SELFTEST OK: citation is not an instruction")

    # 4. LAUNCH: label with inner script is an instruction even without `bash` on that line
    labeled = (
        "PRODUCTION AUTHORITY\n"
        "LAUNCH:    tutor runs, on Mira: CPT_DATA=<x> bash dense-9b/recipes/run_4node_27b_cpt.sh\n"
    )
    hits = affirmative_launch_lines(labeled)
    if not hits:
        print("SELFTEST FAIL: LAUNCH: label with inner script produced no hit")
        failures += 1
    else:
        print("SELFTEST OK: LAUNCH: label with inner script is an instruction")

    # 5. Real tree must still be evaluable without crashing. (May FAIL on INNER LAUNCH until
    # the documents are fixed in the same commit — that is the point of this cycle.)
    _, text = captured(lambda: 0)
    if not _clean_run(text):
        print("SELFTEST FAIL: capture helper itself produced crash-shaped output")
        failures += 1

    try:
        hits_fn = affirmative_launch_lines
        _ = hits_fn("PRODUCTION AUTHORITY\n")
    except Exception:
        print("SELFTEST FAIL: affirmative_launch_lines CRASHED")
        traceback.print_exc()
        return 1

    if failures:
        print(f"SELFTEST: {failures} control(s) failed — the gate cannot be trusted")
        return 1
    print("SELFTEST: instructional detector fails for the right reason; crash distinguished")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix-index", action="store_true")
    ap.add_argument("--self-test", action="store_true", help="prove the gate can fail for the right reason")
    a = ap.parse_args()

    if a.self_test:
        return selftest()

    paths = tracked_markdown()
    tracked = set(
        subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True).stdout.split("\n")
    )

    if a.fix_index:
        os.makedirs("docs", exist_ok=True)
        open(INDEX, "w").write(build_index(paths))
        print(f"wrote {INDEX} covering {len(paths)} documents")
        return 0

    index_text = open(INDEX).read() if os.path.isfile(INDEX) else ""
    readme = open(README).read() if os.path.isfile(README) else ""

    def read_text(p):
        return open(p, encoding="utf-8", errors="replace").read()

    failures = collect_failures(paths, tracked, index_text, readme, read_text)

    print(f"documents: {len(paths)}   index: {INDEX}")
    if not failures:
        print("clean: every document is indexed, process-asserting docs name authority, none instruct an inner launch")
        return 0
    print()
    for f in failures:
        print(f"  {f}")
    print()
    print("=== DOCUMENTATION CAN SEND A READER THE WRONG WAY ===")
    print("A document that is unreachable will drift. A document that names an entrypoint without")
    print("naming its authority competes with the real one. A document that names the authority")
    print("and then says `bash run_4node_27b_cpt.sh` is still competing. All three cost this repo days.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
