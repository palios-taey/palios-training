#!/usr/bin/env python3
"""Parse every tracked shell and Python file. A syntax error is a CI failure.

WHY THIS EXISTS. `bash -n` and `python3 -m py_compile` only catch what someone remembers
to run. A printf %q change, a missing `then`, or a truncated Python file can sit in HEAD
while every other gate is green. CI must parse each file independently.

A gate that has never gone RED on a constructed syntax error is not a gate. `--self-test`
plants a broken .sh and a broken .py and requires this script to fail both, and to
distinguish a crash from a finding.

    python3 scripts/check_syntax.py
    python3 scripts/check_syntax.py --self-test

Exit 0 = every tracked shell and Python file parses. Exit 1 = at least one does not,
or the checker could not see its inputs.
"""
from __future__ import annotations

import argparse
import io
import os
import py_compile
import subprocess
import sys
import tempfile
import traceback


class CannotEnumerate(Exception):
    """Tracked files could not be listed. Not the same as an empty file list."""


def tracked_files():
    try:
        proc = subprocess.run(
            ["git", "ls-files", "-z"],
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise CannotEnumerate(f"git is unreadable ({exc})") from exc
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or b"").decode("utf-8", "replace").strip()
        raise CannotEnumerate(
            f"git ls-files exited {proc.returncode}: {err or '<no stderr>'}"
        )
    return [p.decode("utf-8", "replace") for p in proc.stdout.split(b"\0") if p]


def is_shell(path, first_line):
    if path.endswith(".sh"):
        return True
    s = first_line.strip()
    if not s.startswith("#!"):
        return False
    return s.endswith("bash") or s.endswith("/sh") or "/bash " in s or s.endswith(" sh")


def is_python(path, first_line):
    if path.endswith(".py"):
        return True
    s = first_line.strip()
    return s.startswith("#!") and "python" in s


def first_line_of(path):
    try:
        with open(path, "rb") as fh:
            raw = fh.readline()
    except OSError:
        return ""
    return raw.decode("utf-8", "replace")


def classify(paths):
    shell, python = [], []
    for path in paths:
        if not os.path.isfile(path):
            continue
        first = first_line_of(path)
        if is_shell(path, first):
            shell.append(path)
        elif is_python(path, first):
            python.append(path)
    return shell, python


def check_shell(path):
    proc = subprocess.run(["bash", "-n", path], capture_output=True, text=True)
    if proc.returncode == 0:
        return None
    detail = (proc.stderr or proc.stdout or "").strip().split("\n")[0]
    return f"SHELL SYNTAX   {path}" + (f" — {detail}" if detail else "")


def check_python(path):
    try:
        py_compile.compile(path, doraise=True)
    except py_compile.PyCompileError as exc:
        detail = str(exc).strip().split("\n")[0]
        return f"PYTHON SYNTAX  {path}" + (f" — {detail}" if detail else "")
    except OSError as exc:
        return f"PYTHON SYNTAX  {path} — unreadable ({exc})"
    return None


def check(paths=None):
    """Return error strings. Empty = pass. Never raises on a finding."""
    try:
        if paths is None:
            paths = tracked_files()
    except CannotEnumerate as exc:
        return [f"ABORT         cannot enumerate tracked files: {exc}. "
                f"A check that cannot see its inputs must abort, never report clean."]
    shell, python = classify(paths)
    if not shell and not python:
        return [
            "ABORT         no tracked shell or Python files. "
            "An empty enumeration is not a clean tree."
        ]
    failures = []
    for path in shell:
        err = check_shell(path)
        if err:
            failures.append(err)
    for path in python:
        err = check_python(path)
        if err:
            failures.append(err)
    return failures, len(shell), len(python)


def report(result):
    if result and isinstance(result[0], str) and str(result[0]).startswith("ABORT"):
        failures = result
        print()
        for f in failures:
            print(f"  {f}")
        print()
        print("=== THE CHECKER COULD NOT SEE ITS INPUTS ===")
        return 1
    failures, n_shell, n_python = result
    print(f"shell files: {n_shell}   python files: {n_python}")
    if not failures:
        print("clean: every tracked shell and Python file parses")
        return 0
    print()
    for f in failures:
        print(f"  {f}")
    print()
    print("=== A TRACKED FILE DOES NOT PARSE ===")
    print("bash -n / python3 -m py_compile failed on the named file. Fix the syntax.")
    return 1


def _clean_run(text):
    return "Traceback" not in text and "{m.group" not in text


def selftest():
    """Prove the gate can fail for the right reason. Distinguishes crash from detection."""
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

    def expect_fail(label, result, needle, text=""):
        nonlocal failures
        if not _clean_run(text) and text:
            print(f"SELFTEST FAIL: {label} — checker CRASHED instead of reporting:\n{text}")
            failures += 1
            return
        errs = result[0] if isinstance(result, tuple) else result
        if not errs:
            print(f"SELFTEST FAIL: {label} — expected errors, got PASS")
            failures += 1
            return
        blob = "\n".join(errs) if isinstance(errs, list) else str(errs)
        if needle not in blob:
            print(f"SELFTEST FAIL: {label} — expected {needle!r} in errors, got:\n{blob}")
            failures += 1
            return
        print(f"SELFTEST OK: {label}")

    def expect_pass(label, result, text=""):
        nonlocal failures
        if not _clean_run(text) and text:
            print(f"SELFTEST FAIL: {label} — checker CRASHED instead of reporting:\n{text}")
            failures += 1
            return
        errs = result[0] if isinstance(result, tuple) else result
        if errs:
            print(f"SELFTEST FAIL: {label} — expected PASS, got:\n" + "\n".join(errs))
            failures += 1
            return
        print(f"SELFTEST OK: {label}")

    tree, text = captured(lambda: check())
    if tree is None:
        print("SELFTEST FAIL: real tree CRASHED")
        print(text)
        return 1
    if isinstance(tree, list) and tree and str(tree[0]).startswith("ABORT"):
        print("SELFTEST FAIL: real tree ABORT")
        for e in tree:
            print(f"  - {e}")
        return 1
    if tree[0]:
        print("SELFTEST FAIL: real tree must PASS before mutation tests")
        for e in tree[0]:
            print(f"  - {e}")
        return 1
    print("SELFTEST OK: real tree PASS")

    with tempfile.TemporaryDirectory() as td:
        good_sh = os.path.join(td, "good.sh")
        open(good_sh, "w").write("#!/bin/bash\necho ok\n")
        good_py = os.path.join(td, "good.py")
        open(good_py, "w").write("x = 1\n")
        bad_sh = os.path.join(td, "bad.sh")
        open(bad_sh, "w").write("#!/bin/bash\nif true\necho missing then\nfi\n")
        bad_py = os.path.join(td, "bad.py")
        open(bad_py, "w").write("def (\n")

        errs, text = captured(lambda: check([good_sh, good_py]))
        expect_pass("N1 valid shell and python parse", errs, text or "")

        errs, text = captured(lambda: check([bad_sh]))
        expect_fail("N2 constructed shell syntax error", errs, "SHELL SYNTAX", text or "")

        errs, text = captured(lambda: check([bad_py]))
        expect_fail("N3 constructed python syntax error", errs, "PYTHON SYNTAX", text or "")

        errs, text = captured(lambda: check([good_sh, bad_sh, good_py, bad_py]))
        expect_fail("N4 mixed tree names both languages", errs, "SHELL SYNTAX", text or "")
        blob = "\n".join((errs[0] if isinstance(errs, tuple) else errs) or [])
        if "PYTHON SYNTAX" not in blob:
            print(f"SELFTEST FAIL: N4 did not name the python error:\n{blob}")
            failures += 1

    if failures:
        print(f"SELFTEST: {failures} control(s) failed — the gate cannot be trusted")
        return 1
    print("SELFTEST: constructed syntax errors failed the check (correctly); crash distinguished from detection")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("WHY THIS EXISTS.", 1)[0])
    ap.add_argument("--self-test", action="store_true",
                    help="prove the gate can fail for the right reason")
    a = ap.parse_args()
    if a.self_test:
        return selftest()
    return report(check())


if __name__ == "__main__":
    sys.exit(main())
