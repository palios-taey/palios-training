#!/usr/bin/env python3
"""verify_repo_usage_rows.py — check AUTHORED repo-usage rows against the real capability surface.

WHAT THIS IS NOT. It is not a generator. Rows that teach a model to operate its own infrastructure
are authored by the Family Chats and the fleet, because choosing WHEN a capability applies and what
a realistic operator situation looks like is authoring judgment. A template filled across 42
programs is one inferential path wearing synonyms — the construction we were explicitly warned is
weak, and the reason an earlier attempt at exactly this was scrapped (Jesse, 2026-08-02: "this
cannot be done by a generator").

WHAT IT IS. The mechanical half. An author — model or human — can write a flag that does not exist,
and that row is worse than no row: it teaches confident invocation of something that fails, and it
reads as correct to a reviewer. Diligence cannot catch it at scale; only a check against harvested
source can. So authorship is judgment and admission is mechanical, and this file is the second one.

  extract_capability_registry.py  -> what provably EXISTS (AST over committed source)
  <authors>                       -> what a row should SAY
  this file                       -> the row may only say what exists

THE CHECKS, and what each one is for:
  FLAG      every `--flag` in emitted text exists in the registry for a named repo. The registry is
            a floor rather than a ceiling (it cannot see dynamically built flags), so an unmatched
            flag is HELD for a human, never silently dropped and never auto-approved.
  RESIDUE   the corrections lane's banned vocabulary. A row that narrates a failure teaches it.
  SHAPE     schema present, exactly one user turn and one assistant turn, non-empty both.
  SOURCE    meta.source names a repo the registry covers, so a reviewer can go read the claim.
  EMPTY     no all-whitespace <think></think>. 18 such blocks shipped in a lane labelled canonical
            and nothing mechanical was looking for them.

Exit non-zero if any row fails. There is no --force: adding one would reopen the hole this closes.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

RESIDUE = ("instead of", "should have", "the mistake", "failed", "wasted",
           "the error was", "went wrong")
FLAG_RE = re.compile(r"(?<![\w-])--[a-z][a-z0-9-]+")
THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)
# Universal argparse/CLI conventions rather than repo capabilities; their absence from a
# repo-specific registry is not evidence of anything.
UNIVERSAL = {"--help", "--version"}


def load_registry(path: Path) -> tuple[dict[str, set[str]], set[str]]:
    reg = json.loads(path.read_text())
    by_repo: dict[str, set[str]] = {}
    for entry in reg["repos"]:
        flags = {f for pargs in entry["programs"].values() for a in pargs for f in a["flags"]}
        by_repo[entry["repo"]] = flags
    return by_repo, set(by_repo)


def emitted(row: dict) -> str:
    """Exactly the text that becomes training tokens — never curation metadata."""
    return "\n".join(m.get("content", "") for m in row.get("messages", []))


CODE_SPAN = re.compile(r"```.*?```|`[^`]*`", re.S)


def prose_only(text: str) -> str:
    """Emitted text with code spans removed, for the residue check.

    Residue is about NARRATIVE vocabulary — a row that recounts a failure teaches the failure. It
    is not about identifiers. A raw substring scan does not know the difference, and the first
    real row it judged proved it: an authored row was rejected for the word "failed" appearing
    inside `dispatch_activation_failed`, which is the actual name of a message the notify daemon
    emits. Naming that message is exactly what the row is FOR. Rejecting it would have deleted a
    correct row and taught the authors to avoid true system vocabulary.

    So the check reads the prose and skips the code. Fenced blocks and inline spans go first, and
    matching is word-boundaried so `failed` in prose still fires while `..._failed` in an
    identifier does not.
    """
    return CODE_SPAN.sub(" ", text)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry", required=True, help="output of extract_capability_registry.py")
    ap.add_argument("--rows", required=True, nargs="+", help="authored .jsonl files")
    args = ap.parse_args()

    by_repo, repos = load_registry(Path(args.registry))
    all_flags = {f for s in by_repo.values() for f in s} | UNIVERSAL

    failures: list[str] = []
    held: list[str] = []
    total = 0

    for rp in args.rows:
        p = Path(rp)
        if not p.is_file():
            failures.append(f"{rp}: no such file")
            continue
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if not line.strip():
                continue
            total += 1
            tag = f"{p.name}:{i}"
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                failures.append(f"{tag}: not valid JSON ({e})")
                continue

            if not row.get("schema"):
                failures.append(f"{tag}: no schema")
            msgs = row.get("messages", [])
            roles = [m.get("role") for m in msgs]
            if roles != ["user", "assistant"]:
                failures.append(f"{tag}: expected one user turn then one assistant turn, got {roles}")
            if any(not (m.get("content") or "").strip() for m in msgs):
                failures.append(f"{tag}: a turn has empty content")

            text = emitted(row)
            low = prose_only(text).lower()
            for w in RESIDUE:
                if re.search(rf"(?<![\w-]){re.escape(w)}(?![\w-])", low):
                    failures.append(f"{tag}: residue vocabulary {w!r}")

            for blk in THINK_RE.findall(text):
                if not blk.strip():
                    failures.append(f"{tag}: empty <think></think> block")

            src = str(row.get("meta", {}).get("source", ""))
            named = src.split("@")[0].split(":")[0]
            if named not in repos:
                failures.append(f"{tag}: meta.source names {named!r}, not a registry repo")

            scoped = by_repo.get(named, set()) | UNIVERSAL
            for f in sorted(set(FLAG_RE.findall(text))):
                if f in scoped:
                    continue
                if f in all_flags:
                    held.append(f"{tag}: {f} exists, but in another repo than {named!r}")
                else:
                    held.append(f"{tag}: {f} not found anywhere in the registry")

    print(f"  checked {total} row(s) across {len(args.rows)} file(s)")
    if held:
        print(f"\n  HELD FOR HUMAN REVIEW ({len(held)}) — the registry cannot see dynamically")
        print("  built flags, so these are unproven rather than proven absent:")
        for h in held[:25]:
            print(f"    {h}")
    if failures:
        print(f"\n  FAILED ({len(failures)}):")
        for f in failures[:30]:
            print(f"    {f}")
        return 1
    if held:
        print("\n  no hard failures, but held rows must be resolved before these train.")
        return 2
    print("  PASS — every emitted flag exists in the named repo; no residue; shapes well-formed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
