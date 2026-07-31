#!/usr/bin/env python3
"""Resolve a capability from PRODUCTION_MANIFEST.yml for taey-train.

Kept as a separate file rather than a heredoc inside taey-train for a concrete reason: the
heredoc version collided its own delimiter with the shell's and silently truncated the script.
A resolver that can be mangled by quoting is not a gate.

Contract, deliberately narrow:
  success -> shell assignments on stdout (ENTRY=..., TRAINER=...), exit 0
  refusal -> exactly one line beginning REFUSE:, exit 0 (the CALLER turns that into exit 1)

The refusal path prints rather than exiting non-zero so the caller controls the exit code in one
place. Splitting that decision across two processes is how the first version returned 0 on a
refusal while printing the refusal text.
"""
import shlex
import sys
import yaml


def main() -> int:
    if len(sys.argv) < 3:
        print("REFUSE:resolver called without a capability")
        return 0

    manifest, cap = sys.argv[1], sys.argv[2]
    try:
        doc = yaml.safe_load(open(manifest)) or {}
    except Exception as exc:                      # a manifest that will not parse is not a manifest
        print(f"REFUSE:{manifest} does not parse as YAML: {exc}")
        return 0

    caps = doc.get("capabilities") or {}

    if cap == "--list":
        for name, body in caps.items():
            status = body.get("status", "?")
            mark = "RUNNABLE" if status == "ADJUDICATED" else "blocked"
            print(f"  {name:<22} {status:<34} {mark}")
        for item in doc.get("contested") or []:
            print(f"  {item.get('capability','?'):<22} {'CONTESTED — not adjudicated':<34} blocked")
        return 0

    if cap not in caps:
        known = ", ".join(caps) or "<none>"
        print(f"REFUSE:'{cap}' is not a capability in the manifest. Known: {known}")
        return 0

    body = caps[cap]
    status = body.get("status", "")
    if status != "ADJUDICATED":
        msg = (f"capability '{cap}' has status {status}, not ADJUDICATED. Its gate has not "
               f"passed, and an artifact from a rejected run is not a receipt.")
        open_q = " ".join(str(body.get("open_question", "")).split())
        if open_q:
            msg += f" OPEN QUESTION: {open_q[:300]}"
        print("REFUSE:" + msg)
        return 0

    entry = body.get("entrypoint") or (body.get("stages") or [{}])[0].get("entrypoint", "")
    if not entry:
        print(f"REFUSE:capability '{cap}' is ADJUDICATED but names no entrypoint.")
        return 0

    # shlex.quote would wrap these in quotes the shell then has to strip; these are repo-relative
    # paths under our control, so emit them plainly and let the caller verify existence.
    # Emit the recorded content hashes so the caller can verify the BYTES, not just the path.
    # A manifest that only names files cannot catch drift: the path stays valid while the content
    # changes out from under the receipt that justified it.
    for path, digest in (body.get("content_sha") or {}).items():
        print(f"SHA_{path.replace('/', '_').replace('.', '_').replace('-', '_')}={digest}:{path}")
    # Quote it: entrypoints may legitimately contain spaces (e.g. an env prefix), and an
    # unquoted assignment lets the shell split on whitespace so ENTRY captures only the first
    # word. That produced a launcher whose refusal came from a truncated path rather than from
    # the content check, which never ran — a parsing bug wearing a working gate's clothes.
    print(f"ENTRY={shlex.quote(str(entry))}")
    print(f"TRAINER={body.get('trainer', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
