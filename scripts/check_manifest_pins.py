#!/usr/bin/env python3
"""Verify PRODUCTION_MANIFEST.yml agrees with the bytes in the tree.

WHY THIS EXISTS. `scripts/taey-train` refuses to launch a capability whose pinned `content_sha`
does not match the file on disk, and there is no force flag. That is the right behaviour. But
NOTHING checked the agreement in CI, so the manifest could drift against its own tree and the only
way to find out was to attempt a production launch and watch it refuse.

Measured 2026-08-18: `origin/main` was drifted on 5 of its 12 pins. `scripts/taey-train` on `main`
would have refused `corpus_pack`, `cpt_27b_4node` and `bake_export` -- three of five capabilities --
and no workflow reported anything, because every workflow that existed checked for secrets, private
paths and data blobs. None checked whether the repository agreed with itself. The drift had been
there for 15 days.

TWO CLASSES OF DEFECT ARE CHECKED, because both were found live in the same file:

  DRIFT     a pinned sha does not match the file's bytes, or the pinned file is absent.
  DUPLICATE a capability declares `content_sha` twice. YAML keeps only the last, so the first is
            decorative text that still reads like a pin, and the two must be hand-synced to stay
            honest. Found on `bake_export`, where both copies happened to agree -- which is why it
            survived. A one-sided edit would have left a losing block asserting a pin the gate does
            not enforce.

RUN IT ANYWHERE. Same script, same exit code, in CI and on a laptop. That is deliberate: this repo
already had a gate whose local hook and CI job enforced DIFFERENT rules, so a commit passed locally
and failed CI, and everyone learned to ignore CI.

    python3 scripts/check_manifest_pins.py [--manifest PRODUCTION_MANIFEST.yml]

Exit 0 = the manifest describes this tree. Exit 1 = it does not, and every disagreement is printed.
"""
import argparse
import hashlib
import os
import re
import sys


def duplicate_content_sha_keys(path):
    """Capabilities that declare content_sha more than once.

    Parsed textually and not with the YAML loader, because the loader is exactly what hides this:
    it silently keeps the last occurrence. To see the duplicate you have to read what was written.
    """
    cap = None
    seen = {}
    dupes = []
    for lineno, line in enumerate(open(path), 1):
        m = re.match(r"^  ([A-Za-z_][\w]*):\s*$", line)
        if m:
            cap, seen = m.group(1), {}
            continue
        m2 = re.match(r"^    ([A-Za-z_][\w]*):", line)
        if m2 and cap:
            key = m2.group(1)
            if key in seen:
                dupes.append((cap, key, seen[key], lineno))
            else:
                seen[key] = lineno
    return dupes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="PRODUCTION_MANIFEST.yml")
    a = ap.parse_args()

    if not os.path.isfile(a.manifest):
        print(f"ABORT: {a.manifest} not found", file=sys.stderr)
        return 1
    try:
        import yaml
    except ImportError:
        print("ABORT: pyyaml is required to verify the manifest", file=sys.stderr)
        return 1

    failures = []

    for capability, key, first, second in duplicate_content_sha_keys(a.manifest):
        failures.append(
            f"DUPLICATE KEY  {capability}: '{key}' declared at line {first} AND line {second}. "
            f"YAML keeps the LAST one, so the first is decorative and does not gate anything."
        )

    doc = yaml.safe_load(open(a.manifest))
    capabilities = (doc or {}).get("capabilities") or {}
    checked = 0
    for capability, body in sorted(capabilities.items()):
        for path, expected in sorted(((body or {}).get("content_sha") or {}).items()):
            checked += 1
            if not os.path.isfile(path):
                failures.append(f"MISSING FILE   {capability}: {path} is pinned but not in the tree")
                continue
            actual = hashlib.sha256(open(path, "rb").read()).hexdigest()
            if actual != expected:
                failures.append(
                    f"DRIFT          {capability}: {path}\n"
                    f"                 pinned {expected}\n"
                    f"                 actual {actual}"
                )

    print(f"manifest: {a.manifest}")
    print(f"capabilities: {len(capabilities)}   content_sha pins checked: {checked}")

    if not failures:
        print("clean: every pinned file is present and its bytes match the manifest")
        return 0

    print()
    for f in failures:
        print(f"  {f}")
    print()
    print("=== THE MANIFEST DOES NOT DESCRIBE THIS TREE ===")
    print("scripts/taey-train verifies these pins at launch and there is no force flag, so every")
    print("capability listed above is UNRUNNABLE from this tree until the disagreement is resolved.")
    print("Fix the BYTES if the file changed by accident. Re-pin only when the change is intended,")
    print("and say in the commit message what changed and why.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
