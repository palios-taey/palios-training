#!/usr/bin/env python3
"""Resolve a capability from PRODUCTION_MANIFEST.yml for taey-train.

Kept as a separate file rather than a heredoc inside taey-train for a concrete reason: the
heredoc version collided its own delimiter with the shell's and silently truncated the script.
A resolver that can be mangled by quoting is not a gate.

Contract, deliberately narrow:
  success -> shell assignments on stdout (ENTRY=..., TRAINER=..., LIFECYCLE=...), exit 0
  refusal -> exactly one line beginning REFUSE:, exit 0 (the CALLER turns that into exit 1)

The refusal path prints rather than exiting non-zero so the caller controls the exit code in one
place. Splitting that decision across two processes is how the first version returned 0 on a
refusal while printing the refusal text.

CANDIDATE_* launch is not a new status and not --force. It is permitted only when a top-level
`authorization` (or one entry of `authorizations`) names this capability, its content_sha pins
match the tree, and no manifest receipt records that campaign_id. ADJUDICATED is unchanged.

Receipt lookup is MANIFEST-ONLY. A `receipt:` / `receipts:` / `campaign_receipts:` entry consumes
a campaign only when it carries `campaign_id`. Historical receipt blocks without that field are
not consumption. Disk files under careers-qwen/receipts/ are not consulted — if the receipt is
not in the manifest, this resolver cannot decide it was consumed, and must not pretend.

    python3 scripts/_resolve_capability.py PRODUCTION_MANIFEST.yml <capability>
    python3 scripts/_resolve_capability.py PRODUCTION_MANIFEST.yml --self-test
"""
from __future__ import annotations

import hashlib
import io
import os
import shlex
import sys
import tempfile
import traceback


def sha256_file(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def campaign_receipt_ids(doc):
    """campaign_id values recorded as receipts in this manifest. Nothing else."""
    ids = set()
    for item in doc.get("campaign_receipts") or []:
        if isinstance(item, dict) and item.get("campaign_id"):
            ids.add(str(item["campaign_id"]))
    for body in (doc.get("capabilities") or {}).values():
        if not isinstance(body, dict):
            continue
        rec = body.get("receipt")
        if isinstance(rec, dict) and rec.get("campaign_id"):
            ids.add(str(rec["campaign_id"]))
        for rec in body.get("receipts") or []:
            if isinstance(rec, dict) and rec.get("campaign_id"):
                ids.add(str(rec["campaign_id"]))
    return ids


def authorizations_in(doc):
    """Every authorization object. Absent / malformed entries are skipped, not guessed."""
    found = []
    raw = doc.get("authorization")
    if isinstance(raw, dict):
        found.append(raw)
    for item in doc.get("authorizations") or []:
        if isinstance(item, dict):
            found.append(item)
    return found


def refuse(msg):
    print("REFUSE:" + msg)
    return 0


def emit_success(cap, body, extra_pins=None):
    """Emit SHA/ENTRY/TRAINER/LIFECYCLE. Missing LIFECYCLE= would silently disable W3."""
    entry = body.get("entrypoint") or (body.get("stages") or [{}])[0].get("entrypoint", "")
    if not entry:
        return refuse(f"capability '{cap}' names no entrypoint.")
    lifecycle = body.get("lifecycle", False)
    if not isinstance(lifecycle, bool):
        return refuse(
            f"capability '{cap}' lifecycle must be YAML true or false, "
            f"not {lifecycle!r}."
        )
    pins = dict(body.get("content_sha") or {})
    if extra_pins:
        for path, digest in extra_pins.items():
            pins.setdefault(path, digest)
    for path, digest in pins.items():
        key = path.replace("/", "_").replace(".", "_").replace("-", "_")
        print(f"SHA_{key}={digest}:{path}")
    print(f"ENTRY={shlex.quote(str(entry))}")
    print(f"TRAINER={body.get('trainer', '')}")
    print(f"LIFECYCLE={'true' if lifecycle else 'false'}")
    return 0


def authorize_candidate(doc, cap, body, root, get_bytes):
    """C1–C5, C7. Returns (ok: bool, extra_pins or None, refuse_msg or None)."""
    status = body.get("status", "")
    matching = [a for a in authorizations_in(doc) if a.get("capability") == cap]
    if not matching:
        others = [a.get("capability") for a in authorizations_in(doc) if a.get("capability")]
        if others:
            return False, None, (
                f"authorization names {others[0]!r}, not '{cap}'. "
                f"An authorization is not transferable across capabilities."
            )
        return False, None, (
            f"capability '{cap}' has status {status}, not ADJUDICATED, and no authorization "
            f"block names it. A CANDIDATE without a human-authorized object is not launchable."
        )
    if len(matching) > 1:
        return False, None, (
            f"capability '{cap}' has {len(matching)} authorization objects. "
            f"Ambiguous authorization is not a gate."
        )
    auth = matching[0]
    campaign_id = auth.get("campaign_id")
    if not campaign_id:
        return False, None, (
            f"authorization for '{cap}' has no campaign_id. "
            f"An authorization without a campaign id cannot be consumed and is a standing bypass."
        )
    if not auth.get("authorized_by"):
        return False, None, (
            f"authorization for '{cap}' has no authorized_by. "
            f"Jesse's authorization is a named act, not an anonymous flag."
        )
    pins = auth.get("content_sha")
    if not isinstance(pins, dict) or not pins:
        return False, None, (
            f"authorization for '{cap}' has no content_sha pins. "
            f"The binding to exact bytes is what stops this becoming a standing path."
        )

    spent = campaign_receipt_ids(doc)
    if str(campaign_id) in spent:
        return False, None, (
            f"campaign_id {campaign_id!r} already has a receipt in the manifest. "
            f"That authorization is consumed. A second campaign needs a new authorization commit."
        )

    cap_pins = body.get("content_sha") or {}
    for path, digest in pins.items():
        if path in cap_pins and cap_pins[path] != digest:
            return False, None, (
                f"{path} authorization pin {digest} disagrees with capability content_sha "
                f"{cap_pins[path]}. The authorization does not describe this capability."
            )
        full = path if os.path.isabs(path) else os.path.join(root, path)
        data = get_bytes(full)
        if data is None:
            return False, None, (
                f"{path} is pinned by the authorization but is not in the tree."
            )
        actual = hashlib.sha256(data).hexdigest()
        if actual != digest:
            return False, None, (
                f"{path} drifted: authorization pin {digest} tree {actual}. "
                f"The authorization binds exact bytes; a one-byte change is a different campaign."
            )
    return True, pins, None


def resolve(doc, cap, *, root=".", get_bytes=None):
    """Return exit code. Prints success assignments or one REFUSE: line. Never raises on a finding."""
    if get_bytes is None:
        def get_bytes(path):  # noqa: F811
            if not os.path.isfile(path):
                return None
            with open(path, "rb") as fh:
                return fh.read()

    caps = doc.get("capabilities") or {}
    if cap not in caps:
        known = ", ".join(caps) or "<none>"
        return refuse(f"'{cap}' is not a capability in the manifest. Known: {known}")

    body = caps[cap] or {}
    status = body.get("status", "")

    if status == "ADJUDICATED":
        return emit_success(cap, body)

    if isinstance(status, str) and status.startswith("CANDIDATE"):
        ok, extra, msg = authorize_candidate(doc, cap, body, root, get_bytes)
        if not ok:
            return refuse(msg)
        return emit_success(cap, body, extra)

    msg = (
        f"capability '{cap}' has status {status}, not ADJUDICATED. Its gate has not "
        f"passed, and an artifact from a rejected run is not a receipt."
    )
    open_q = " ".join(str(body.get("open_question", "")).split())
    if open_q:
        msg += f" OPEN QUESTION: {open_q[:300]}"
    return refuse(msg)


def list_caps(doc):
    caps = doc.get("capabilities") or {}
    for name, body in caps.items():
        body = body or {}
        status = body.get("status", "?")
        mark = "RUNNABLE" if status == "ADJUDICATED" else "blocked"
        print(f"  {name:<22} {status:<34} {mark}")
    for item in doc.get("contested") or []:
        print(f"  {item.get('capability', '?'):<22} {'CONTESTED — not adjudicated':<34} blocked")
    return 0


def _clean(text):
    return "Traceback" not in text and "{m.group" not in text


def selftest():
    """Construct C1–C7. A crash is not a refusal: no Traceback, rendered REFUSE names real values."""
    failures = 0
    digest_a = hashlib.sha256(b"pin-a\n").hexdigest()
    digest_b = hashlib.sha256(b"pin-b\n").hexdigest()

    def run(doc, cap, files):
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        crashed = None
        code = None
        try:
            with tempfile.TemporaryDirectory() as td:
                for rel, data in files.items():
                    path = os.path.join(td, rel)
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, "wb") as fh:
                        fh.write(data)

                def gb(p, _td=td):
                    if not os.path.isfile(p):
                        # resolve joins root+rel; also accept already-joined
                        return None
                    with open(p, "rb") as fh:
                        return fh.read()

                code = resolve(doc, cap, root=td, get_bytes=gb)
        except Exception:
            crashed = traceback.format_exc()
            code = 1
        finally:
            sys.stdout = old
        text = buf.getvalue()
        if crashed:
            text += "\n" + crashed
        return code, text

    def expect_refuse(label, code, text, *needles):
        nonlocal failures
        if not _clean(text):
            print(f"SELFTEST FAIL: {label} — CRASHED rather than refused:\n{text}")
            failures += 1
            return
        if code != 0:
            print(f"SELFTEST FAIL: {label} — refusal must exit 0 (caller maps to 1), got {code}")
            failures += 1
            return
        if not text.startswith("REFUSE:"):
            print(f"SELFTEST FAIL: {label} — expected REFUSE:, got:\n{text}")
            failures += 1
            return
        for n in needles:
            if n not in text:
                print(f"SELFTEST FAIL: {label} — expected {n!r} in refusal, got:\n{text}")
                failures += 1
                return
        print(f"SELFTEST OK: {label}")

    def expect_permit(label, code, text):
        nonlocal failures
        if not _clean(text):
            print(f"SELFTEST FAIL: {label} — CRASHED rather than permitted:\n{text}")
            failures += 1
            return
        if code != 0 or text.startswith("REFUSE:") or "ENTRY=" not in text:
            print(f"SELFTEST FAIL: {label} — expected permit, code={code} text:\n{text}")
            failures += 1
            return
        if "LIFECYCLE=" not in text:
            print(f"SELFTEST FAIL: {label} — missing LIFECYCLE= emission (W3 would silently disable):\n{text}")
            failures += 1
            return
        print(f"SELFTEST OK: {label}")

    files = {"dense-9b/recipes/run_4node_27b_cpt.sh": b"pin-a\n"}
    candidate = {
        "capabilities": {
            "cpt_27b_4node": {
                "status": "CANDIDATE_PENDING_PRODUCTION_RUN",
                "entrypoint": "dense-9b/recipes/run_4node_27b_cpt.sh",
                "content_sha": {"dense-9b/recipes/run_4node_27b_cpt.sh": digest_a},
            }
        }
    }
    adjudicated = {
        "capabilities": {
            "cpt_27b_4node": {
                "status": "ADJUDICATED",
                "entrypoint": "dense-9b/recipes/run_4node_27b_cpt.sh",
                "content_sha": {"dense-9b/recipes/run_4node_27b_cpt.sh": digest_a},
            }
        }
    }
    auth = {
        "capability": "cpt_27b_4node",
        "content_sha": {"dense-9b/recipes/run_4node_27b_cpt.sh": digest_a},
        "authorized_by": "Jesse",
        "campaign_id": "phase2-qwen38",
    }

    code, text = run(candidate, "cpt_27b_4node", files)
    expect_refuse("C1 no authorization", code, text, "CANDIDATE_PENDING_PRODUCTION_RUN", "cpt_27b_4node")

    doc = {**candidate, "authorization": auth}
    code, text = run(doc, "cpt_27b_4node", files)
    expect_permit("C2 pins match tree", code, text)

    bad = dict(auth)
    bad["content_sha"] = {"dense-9b/recipes/run_4node_27b_cpt.sh": digest_b}
    code, text = run({**candidate, "authorization": bad}, "cpt_27b_4node", files)
    expect_refuse(
        "C3 one-byte pin drift",
        code,
        text,
        "dense-9b/recipes/run_4node_27b_cpt.sh",
        digest_b,
        digest_a,
    )

    spent = {
        **candidate,
        "authorization": auth,
        "capabilities": {
            "cpt_27b_4node": {
                **candidate["capabilities"]["cpt_27b_4node"],
                "receipt": {"campaign_id": "phase2-qwen38", "verified_outcome": "THOR_DELIVERED"},
            }
        },
    }
    code, text = run(spent, "cpt_27b_4node", files)
    expect_refuse("C4 receipt exists for campaign_id", code, text, "phase2-qwen38", "receipt")

    # Historical receipt WITHOUT campaign_id must not consume (would false-C4 a real campaign).
    leftover = {
        **candidate,
        "authorization": auth,
        "capabilities": {
            "cpt_27b_4node": {
                **candidate["capabilities"]["cpt_27b_4node"],
                "receipt": {"date": "2026-07-30", "verified_outcome": "old narrative"},
            }
        },
    }
    code, text = run(leftover, "cpt_27b_4node", files)
    expect_permit("C5 leftover receipt without campaign_id is not consumption", code, text)

    code1, text1 = run(doc, "cpt_27b_4node", files)
    code2, text2 = run(doc, "cpt_27b_4node", files)
    if code1 == 0 and code2 == 0 and "ENTRY=" in text1 and "ENTRY=" in text2 and _clean(text1 + text2):
        print("SELFTEST OK: C5 same campaign_id second session, no receipt — both PERMIT")
    else:
        print(f"SELFTEST FAIL: C5 second session\n first={code1} {text1}\n second={code2} {text2}")
        failures += 1

    code, text = run(adjudicated, "cpt_27b_4node", files)
    expect_permit("C6 ADJUDICATED no authorization required", code, text)
    # ADJUDICATED plus a stray authorization for a different cap must still permit.
    code, text = run({**adjudicated, "authorization": {**auth, "capability": "bake_export"}}, "cpt_27b_4node", files)
    expect_permit("C6 ADJUDICATED ignores foreign authorization", code, text)

    foreign = {**candidate, "authorization": {**auth, "capability": "bake_export"}}
    code, text = run(foreign, "cpt_27b_4node", files)
    expect_refuse("C7 authorization names a different capability", code, text, "bake_export", "cpt_27b_4node")

    # Codex lifecycle: YAML bool only. A missing emission is a silent W3 disable.
    code, text = run(adjudicated, "cpt_27b_4node", files)
    if "LIFECYCLE=false" not in text:
        print(f"SELFTEST FAIL: L8 absent lifecycle must emit LIFECYCLE=false\n{text}")
        failures += 1
    else:
        print("SELFTEST OK: L8 absent lifecycle emits LIFECYCLE=false")

    life_true = {
        "capabilities": {
            "cpt_27b_4node": {
                **adjudicated["capabilities"]["cpt_27b_4node"],
                "lifecycle": True,
            }
        }
    }
    code, text = run(life_true, "cpt_27b_4node", files)
    expect_permit("L9 lifecycle true", code, text)
    if "LIFECYCLE=true" not in text:
        print(f"SELFTEST FAIL: L9 expected LIFECYCLE=true\n{text}")
        failures += 1

    life_bad = {
        "capabilities": {
            "cpt_27b_4node": {
                **adjudicated["capabilities"]["cpt_27b_4node"],
                "lifecycle": "yes",
            }
        }
    }
    code, text = run(life_bad, "cpt_27b_4node", files)
    expect_refuse("L10 lifecycle not YAML bool", code, text, "lifecycle", "yes")

    cand_life = {
        **candidate,
        "authorization": auth,
        "capabilities": {
            "cpt_27b_4node": {
                **candidate["capabilities"]["cpt_27b_4node"],
                "lifecycle": True,
            }
        },
    }
    code, text = run(cand_life, "cpt_27b_4node", files)
    expect_permit("L11 authorized CANDIDATE still emits LIFECYCLE", code, text)
    if "LIFECYCLE=true" not in text:
        print(f"SELFTEST FAIL: L11 expected LIFECYCLE=true on authorized CANDIDATE\n{text}")
        failures += 1

    if failures:
        print(f"SELFTEST: {failures} control(s) failed")
        return 1
    print("SELFTEST: C1–C7 + lifecycle constructed; crash distinguished from refusal")
    return 0


def main() -> int:
    if len(sys.argv) < 3:
        return refuse("resolver called without a capability")
    manifest, cap = sys.argv[1], sys.argv[2]
    if cap == "--self-test":
        return selftest()
    try:
        import yaml
    except ImportError:
        return refuse("pyyaml is required to resolve a capability")
    try:
        doc = yaml.safe_load(open(manifest)) or {}
    except Exception as exc:
        return refuse(f"{manifest} does not parse as YAML: {exc}")
    if cap == "--list":
        return list_caps(doc)
    root = os.path.dirname(os.path.abspath(manifest)) or "."
    return resolve(doc, cap, root=root)


if __name__ == "__main__":
    raise SystemExit(main())
