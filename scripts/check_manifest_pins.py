#!/usr/bin/env python3
"""PRODUCTION_MANIFEST.yml must describe the production surface, not merely agree with itself.

WHY THIS EXISTS. `scripts/taey-train` refuses to launch a capability whose pinned `content_sha`
does not match the file on disk, and there is no force flag. That is the right behaviour. But
NOTHING checked the agreement in CI, so the manifest could drift against its own tree and the only
way to find out was to attempt a production launch and watch it refuse.

Measured 2026-08-18: `origin/main` was drifted on 5 of its 12 pins. `scripts/taey-train` on `main`
would have refused `corpus_pack`, `cpt_27b_4node` and `bake_export` -- three of five capabilities --
and no workflow reported anything.

W6 (tutor-grok, 2026-08-18): the first version of this checker hashed only the pinned subset. An
UNPINNED production file could be arbitrarily wrong and it still exited 0. That is consistency
(the pins that exist match) not correspondence (every file the capability actually runs is pinned).
It also never compared NODE-DEPLOYED bytes to the pin — the defect this repo already paid for,
when production ran on node copies that were not the bytes in git.

FOUR CLASSES OF DEFECT ARE CHECKED, because all four were found live or constructed:

  DRIFT          a pinned sha does not match the file's bytes, or the pinned file is absent.
  DUPLICATE      a capability declares `content_sha` twice. YAML keeps only the last, so the first
                 is decorative text that still reads like a pin.
  UNPINNED       a capability that names a structural path (entrypoint / trainer / per_node /
                 config / runner / qualifier / path_family_winner, plus stage entrypoints) does
                 not pin that path in content_sha. Status does not excuse this: a CANDIDATE_*
                 capability whose bytes have never run on hardware is the one that most needs
                 its pins, not the one that skips them. The file can change freely and this
                 checker used to stay green.
  AUTHORIZATION  a leftover authorization object that would become a standing bypass.
                 (1) an authorization names a capability whose status is ADJUDICATED — the
                 promotion commit must delete the authorization; omitting that is the residual.
                 (2) an authorization.campaign_id already appears on a manifest receipt — the
                 resolver already REFUSES that launch; CI failing the dirty manifest is the
                 durable record.
                 Do NOT fail CANDIDATE + authorization + no receipt: that is the valid pre-run
                 window. Failing it makes the authorized launch un-mergeable.

A candidate_update pin that does not match HEAD is judged against git history of that path.
If that history cannot be read — shallow clone, no git, rev-parse failure — the checker
ABORTS. It does not emit CANDIDATE. Absence of evidence is not a finding; that is the
same class as invokers() treating a failed git grep as zero callers. `.github/workflows/
manifest-pins.yml` fetches full history so CI can look; this abort is the durable rule
for every other shallow tree.

`--deployed` additionally hashes the copy on each Spark at `$SPARK_HOME/palios-training/<path>`
and refuses on mismatch. CI cannot SSH, so it does not pass `--deployed`. `scripts/taey-train`
does, whenever `fleet.env` is present. Skipping when fleet.env is absent is announced, not silent.

    python3 scripts/check_manifest_pins.py [--manifest PRODUCTION_MANIFEST.yml] [--deployed]
    python3 scripts/check_manifest_pins.py --self-test

Exit 0 = the manifest describes this tree (and, with --deployed, the nodes). Exit 1 = it does not.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
import re
import subprocess
import sys
import tempfile
import traceback

STRUCTURAL_KEYS = (
    "entrypoint",
    "trainer",
    "per_node",
    "config",
    "runner",
    "qualifier",
    "path_family_winner",
)

PATH_SUFFIXES = (".sh", ".py", ".yaml", ".yml", ".json")


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


def repo_path_from_value(value):
    """Extract a repo-relative production path from a structural YAML value.

    Stage entrypoints may carry an env prefix (`EXPORT_DCP=<dir> dense-9b/recipes/bake_27b.sh`).
    Topology placeholders (`$SPARK_HOME/...`) are not in-repo files.
    """
    if not isinstance(value, str):
        return None
    token = value.strip()
    if " " in token:
        token = token.split()[-1]
    if token.startswith("$") or token.startswith("<") or token.startswith("/"):
        return None
    if any(token.endswith(sfx) for sfx in PATH_SUFFIXES) and "/" in token:
        return token
    return None


def structural_paths(body):
    """Every in-repo file a capability actually names as something it runs."""
    found = []
    for key in STRUCTURAL_KEYS:
        p = repo_path_from_value((body or {}).get(key))
        if p:
            found.append(p)
    for stage in (body or {}).get("stages") or []:
        if not isinstance(stage, dict):
            continue
        p = repo_path_from_value(stage.get("entrypoint"))
        if p:
            found.append(p)
    # Preserve order, drop duplicates.
    out, seen = [], set()
    for p in found:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def load_fleet_env(path):
    env = {}
    for raw in open(path):
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        env[key.strip()] = val.strip().strip('"').strip("'")
    return env


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class HistoryUnreadable(Exception):
    """History could not be enumerated. Not the same as 'enumerated and the digest is absent'."""


def require_complete_git_history():
    """Raise HistoryUnreadable unless this is a non-shallow git repository.

    `git log` in a depth-1 clone succeeds and returns the tip. Treating that
    subset as 'no revision matches' reports absence of evidence as a finding —
    the false claim this checker exists to prevent.
    """
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--is-shallow-repository"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise HistoryUnreadable(f"git is unreadable ({exc})") from exc
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip() or f"exit {proc.returncode}"
        raise HistoryUnreadable(
            f"git rev-parse --is-shallow-repository failed: {err}"
        )
    answer = proc.stdout.strip()
    if answer == "true":
        raise HistoryUnreadable(
            "repository is shallow; candidate_update pins cannot be judged "
            "against complete history"
        )
    if answer != "false":
        raise HistoryUnreadable(
            f"git rev-parse --is-shallow-repository printed {answer!r}, not true or false"
        )


def sha256_in_git_history(path, digest):
    """True iff some committed revision of `path` hashes to digest.

    candidate_update.content_sha records proposed or prior bytes, which need not
    equal HEAD. A pin that matches no revision of the named path is an orphan —
    the same 'pin that corresponds to nothing' this checker exists to close.

    Returns False only after walking a complete (non-shallow) log. Raises
    HistoryUnreadable when the log cannot be trusted as complete. Callers must
    not translate that exception into CANDIDATE.
    """
    require_complete_git_history()
    try:
        revs = subprocess.check_output(
            ["git", "log", "--format=%H", "--", path],
            text=True,
            stderr=subprocess.DEVNULL,
        ).split()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise HistoryUnreadable(f"git log for {path} failed: {exc}") from exc
    for rev in revs:
        try:
            raw = subprocess.check_output(
                ["git", "show", f"{rev}:{path}"],
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.CalledProcessError):
            continue
        if sha256_bytes(raw) == digest:
            return True
    return False


def pin_blocks(body):
    """Every content_sha map this capability carries, labelled."""
    body = body or {}
    yield "content_sha", body.get("content_sha") or {}
    cu = body.get("candidate_update")
    if isinstance(cu, dict) and cu.get("content_sha"):
        yield "candidate_update.content_sha", cu.get("content_sha") or {}


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


def campaign_receipt_ids(doc):
    """campaign_id values recorded as receipts in this manifest. Nothing else.

    Manifest-only. A receipt consumes a campaign only when it carries campaign_id.
    Historical receipt blocks without that field are not consumption. Disk files
    under careers-qwen/receipts/ are not consulted.
    """
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


def authorization_failures(doc):
    """Promotion-side leftover authorization. Empty = none. Never guesses C5.

    (1) authorization names a capability whose status is ADJUDICATED.
    (2) authorization.campaign_id already appears on a manifest receipt.
    CANDIDATE + authorization + no receipt is the valid pre-run window — not a failure.
    """
    failures = []
    capabilities = doc.get("capabilities") or {}
    spent = campaign_receipt_ids(doc)
    for auth in authorizations_in(doc):
        cap = auth.get("capability")
        body = capabilities.get(cap) if cap else None
        status = body.get("status", "") if isinstance(body, dict) else ""
        if cap and isinstance(body, dict) and status == "ADJUDICATED":
            failures.append(
                f"AUTHORIZATION  {cap}: status is ADJUDICATED and an authorization still names it. "
                f"The promotion commit must delete the authorization. Leaving it is a standing "
                f"bypass created by omission — C4 cannot fire if campaign_id is also omitted."
            )
        campaign_id = auth.get("campaign_id")
        if campaign_id and str(campaign_id) in spent:
            named = cap or "?"
            failures.append(
                f"AUTHORIZATION  {named}: campaign_id {campaign_id!r} already has a receipt in "
                f"the manifest. That authorization is consumed. The resolver already REFUSES "
                f"this launch; CI failing the dirty manifest is the durable record."
            )
    return failures


def check(manifest_path, *, deployed=False, get_bytes=None, hash_remote=None):
    """Return a list of error strings. Empty = pass. Never raises on a finding.

    get_bytes(path) -> bytes | None (None means missing). Defaults to reading the tree.
    hash_remote(host, remote_path) -> hex digest | None. Only used with deployed=True.
    """
    if get_bytes is None:
        def get_bytes(path):  # noqa: F811
            if not os.path.isfile(path):
                return None
            with open(path, "rb") as fh:
                return fh.read()

    try:
        import yaml
    except ImportError:
        return ["ABORT: pyyaml is required to verify the manifest"]

    failures = []

    for capability, key, first, second in duplicate_content_sha_keys(manifest_path):
        failures.append(
            f"DUPLICATE KEY  {capability}: '{key}' declared at line {first} AND line {second}. "
            f"YAML keeps the LAST one, so the first is decorative and does not gate anything."
        )

    doc = yaml.safe_load(open(manifest_path)) or {}
    capabilities = doc.get("capabilities") or {}
    contested = doc.get("contested") or []
    checked = 0

    for capability, body in sorted(capabilities.items()):
        body = body or {}
        pins = body.get("content_sha") or {}

        for label, block in pin_blocks(body):
            for path, expected in sorted((block or {}).items()):
                checked += 1
                data = get_bytes(path)
                if data is None:
                    failures.append(
                        f"MISSING FILE   {capability}: {path} is pinned under {label} "
                        f"but not in the tree"
                    )
                    continue
                actual = sha256_bytes(data)
                if actual == expected:
                    continue
                if label == "candidate_update.content_sha":
                    try:
                        in_history = sha256_in_git_history(path, expected)
                    except HistoryUnreadable as exc:
                        failures.append(
                            f"ABORT         {capability}: {path} candidate_update pin {expected} "
                            f"does not match the tree ({actual}), and history cannot be read: {exc}. "
                            f"A check that cannot see its inputs must abort, never report the pin "
                            f"as an orphan."
                        )
                        continue
                    if in_history:
                        continue
                    failures.append(
                        f"CANDIDATE      {capability}: {path} pin {expected} matches neither "
                        f"the tree ({actual}) nor any committed revision of that path. "
                        f"A pin that corresponds to nothing is not a pin."
                    )
                else:
                    failures.append(
                        f"DRIFT          {capability}: {path}\n"
                        f"                 pinned {expected}\n"
                        f"                 actual {actual}"
                    )

        named = structural_paths(body)
        if named and not pins:
            failures.append(
                f"UNPINNED       {capability} names structural paths but has no content_sha. "
                f"Status {body.get('status', '?')!r} does not excuse this: a capability "
                f"whose bytes are proposed-for-production needs pins more, not less."
            )
        for path in named:
            if path not in pins:
                failures.append(
                    f"UNPINNED       {capability} runs {path} (named as a structural path) "
                    f"but does not pin it. Status {body.get('status', '?')!r}. An unpinned "
                    f"production file can be arbitrarily wrong and a pins-only hash still exits 0."
                )

    for item in contested:
        if not isinstance(item, dict):
            continue
        name = item.get("capability") or "?"
        for cand in item.get("candidates") or []:
            path = repo_path_from_value(cand) or (cand if isinstance(cand, str) else None)
            if not path:
                continue
            data = get_bytes(path)
            if data is None:
                continue
            # File exists in the tree and is visible to taey-train --list via contested,
            # but the pin checker would otherwise never see it.
            pinned_anywhere = False
            for body in capabilities.values():
                for _, block in pin_blocks(body or {}):
                    if path in (block or {}):
                        pinned_anywhere = True
            if not pinned_anywhere:
                failures.append(
                    f"UNPINNED       contested {name} names {path}, which is in the tree, "
                    f"but no content_sha pin covers it. The resolver lists this capability; "
                    f"the pin checker must too."
                )

    failures.extend(authorization_failures(doc))

    if deployed:
        fleet_path = os.environ.get("FLEET_ENV", "fleet.env")
        if not os.path.isfile(fleet_path):
            failures.append(
                "DEPLOYED       --deployed was requested but fleet.env is absent. "
                "Refusing rather than skipping: a silent skip is how node drift stays green."
            )
        else:
            fleet = load_fleet_env(fleet_path)
            spark_home = fleet.get("SPARK_HOME")
            hosts = (fleet.get("SPARK_MGMT_IPS") or "").split()
            user = fleet.get("SPARK_USER") or "spark"
            if not spark_home or len(hosts) != 4:
                failures.append(
                    "DEPLOYED       fleet.env did not yield SPARK_HOME and four SPARK_MGMT_IPS. "
                    f"SPARK_HOME={spark_home!r} hosts={hosts!r}"
                )
            else:
                if hash_remote is None:
                    def hash_remote(host, remote_path, _user=user):  # noqa: F811
                        try:
                            proc = subprocess.run(
                                [
                                    "ssh",
                                    "-o", "BatchMode=yes",
                                    "-o", "ConnectTimeout=8",
                                    f"{_user}@{host}",
                                    f"sha256sum {remote_path}",
                                ],
                                capture_output=True,
                                text=True,
                                timeout=20,
                                check=False,
                            )
                        except (OSError, subprocess.TimeoutExpired) as exc:
                            return None, str(exc)
                        if proc.returncode != 0:
                            err = (proc.stderr or proc.stdout or "ssh failed").strip().split("\n")[0]
                            return None, err
                        digest = (proc.stdout or "").split()[0] if proc.stdout else ""
                        if not re.fullmatch(r"[0-9a-f]{64}", digest):
                            return None, f"unparseable sha256sum output: {(proc.stdout or '')[:80]!r}"
                        return digest, ""

                for capability, body in sorted(capabilities.items()):
                    for path, expected in sorted(((body or {}).get("content_sha") or {}).items()):
                        remote = f"{spark_home.rstrip('/')}/palios-training/{path}"
                        for host in hosts:
                            digest, err = hash_remote(host, remote)
                            if digest is None:
                                failures.append(
                                    f"DEPLOYED       {capability}: {path} on {host} could not be "
                                    f"hashed ({err}). Node-unreadable is not 'matches'."
                                )
                            elif digest != expected:
                                failures.append(
                                    f"DEPLOYED       {capability}: {path} on {host}\n"
                                    f"                 pinned  {expected}\n"
                                    f"                 deployed {digest}"
                                )

    return failures, checked, len(capabilities), len(contested)


def report(failures, checked, n_caps, n_contested, manifest):
    print(f"manifest: {manifest}")
    print(
        f"capabilities: {n_caps}   contested: {n_contested}   "
        f"content_sha pins checked: {checked}"
    )
    if not failures:
        print(
            "clean: every production file a capability names is pinned, candidate pins "
            "correspond to a real revision, the pins match, and no leftover authorization "
            "survives promotion or consumption"
        )
        return 0
    print()
    for f in failures:
        print(f"  {f}")
    print()
    aborts = [f for f in failures if f.startswith("ABORT")]
    findings = [f for f in failures if not f.startswith("ABORT")]
    if aborts:
        print("=== THE CHECKER COULD NOT SEE ITS INPUTS ===")
        print("A shallow clone or missing git is not an orphan pin. Fetch the full history")
        print("and re-run. Absence of evidence is not a finding.")
    if findings:
        print("=== THE MANIFEST DOES NOT DESCRIBE THE PRODUCTION SURFACE ===")
        print("scripts/taey-train verifies pinned files at launch and there is no force flag, so a")
        print("drifted pin makes a capability UNRUNNABLE. An UNPINNED production file is worse: the")
        print("capability is runnable on bytes no receipt covers. Fix the BYTES if the file changed")
        print("by accident. Re-pin only when the change is intended, and say why.")
    return 1


def _clean_run(text: str) -> bool:
    """True iff output is a report, not a crash. Exit code alone cannot tell them apart."""
    return "Traceback" not in text and "{m.group" not in text


def selftest() -> int:
    """Prove the gate can fail for the right reason. Distinguishes crash from detection."""
    manifest = "PRODUCTION_MANIFEST.yml"
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

    def expect_fail(label, errs, needle, text=""):
        nonlocal failures
        if not _clean_run(text) and text:
            print(f"SELFTEST FAIL: {label} — checker CRASHED instead of reporting:\n{text}")
            failures += 1
            return
        if not errs:
            print(f"SELFTEST FAIL: {label} — expected errors, got PASS")
            failures += 1
            return
        blob = "\n".join(errs)
        if needle not in blob:
            print(f"SELFTEST FAIL: {label} — expected {needle!r} in errors, got:\n{blob}")
            failures += 1
            return
        print(f"SELFTEST OK: {label}")

    def expect_pass(label, errs, text=""):
        nonlocal failures
        if not _clean_run(text) and text:
            print(f"SELFTEST FAIL: {label} — checker CRASHED instead of reporting:\n{text}")
            failures += 1
            return
        if errs:
            print(f"SELFTEST FAIL: {label} — expected PASS, got:\n" + "\n".join(errs))
            failures += 1
            return
        print(f"SELFTEST OK: {label}")

    clean, text = captured(lambda: check(manifest)[0])
    if clean is None:
        print("SELFTEST FAIL: real tree CRASHED")
        print(text)
        return 1
    if clean:
        print("SELFTEST FAIL: real tree must PASS before mutation tests")
        for e in clean:
            print(f"  - {e}")
        return 1
    print("SELFTEST OK: real tree PASS")

    original = open(manifest).read()

    with tempfile.TemporaryDirectory() as td:
        tdir = os.path.join(td, "manifest.yml")

        # 1. UNPINNED: drop the pin for the CPT operator entrypoint, leave the structural key.
        mutated = "".join(
            line for line in original.splitlines(True)
            if not (
                "dense-9b/recipes/run_4node_27b_cpt.sh:" in line
                and re.match(r"\s+dense-9b/recipes/run_4node_27b_cpt\.sh:", line)
            )
        )
        if mutated == original:
            print("SELFTEST FAIL: could not locate run_4node_27b_cpt.sh pin to drop")
            failures += 1
        else:
            open(tdir, "w").write(mutated)
            errs, text = captured(lambda: check(tdir)[0])
            expect_fail("unpinned ADJUDICATED entrypoint", errs or [], "UNPINNED", text or "")
            if errs and "run_4node_27b_cpt.sh" not in "\n".join(errs):
                print("SELFTEST FAIL: UNPINNED finding did not name the dropped file")
                failures += 1

        # 1b. UNPINNED must still fire after status is no longer ADJUDICATED.
        # tutor-codex W3/W5 will mark cpt_27b_4node CANDIDATE_PENDING_PRODUCTION_RUN;
        # gating UNPINNED on ADJUDICATED would drop the two capabilities whose bytes are new.
        cand_manifest = os.path.join(td, "candidate.yml")
        open(cand_manifest, "w").write(
            "capabilities:\n"
            "  pending_cpt:\n"
            "    status: CANDIDATE_PENDING_PRODUCTION_RUN\n"
            "    entrypoint: dense-9b/recipes/run_4node_27b_cpt.sh\n"
        )
        errs, text = captured(lambda: check(cand_manifest)[0])
        expect_fail(
            "unpinned CANDIDATE_PENDING_PRODUCTION_RUN still UNPINNED",
            errs or [],
            "UNPINNED",
            text or "",
        )

        # 1c. candidate_update pin that matches no tree file and no git revision is an orphan.
        orphan_manifest = os.path.join(td, "orphan.yml")
        open(orphan_manifest, "w").write(
            "capabilities:\n"
            "  corpus_pack:\n"
            "    status: ADJUDICATED\n"
            "    entrypoint: careers-qwen/pack_production_corpus.py\n"
            "    content_sha:\n"
            "      careers-qwen/pack_production_corpus.py: "
            "420bc706f5fd491fcc1b529139e880fbf0a126b2ede69e14c99138588af8532d\n"
            "    candidate_update:\n"
            "      content_sha:\n"
            "        careers-qwen/pack_production_corpus.py: "
            + ("ab" * 32)
            + "\n"
        )
        errs, text = captured(lambda: check(orphan_manifest)[0])
        expect_fail("orphan candidate pin", errs or [], "CANDIDATE", text or "")

        # 2. DRIFT: pin stays, bytes change. Injected via get_bytes, never by mutating the tree.
        target = "dense-9b/recipes/run_4node_27b_cpt.sh"
        real = open(target, "rb").read() if os.path.isfile(target) else b""

        def drifted(path, _real=real, _target=target):
            if path == _target:
                return _real + b"\n# planted-byte-for-selftest\n"
            if not os.path.isfile(path):
                return None
            with open(path, "rb") as fh:
                return fh.read()

        errs, text = captured(lambda: check(manifest, get_bytes=drifted)[0])
        expect_fail("pinned file bytes drifted", errs or [], "DRIFT", text or "")

        # 3. An UNPINNED non-production file being wrong must NOT fail. Correspondence, not
        # hashing the whole tree. README.md is not a structural path of any ADJUDICATED capability.
        def noise(path):
            if path == "README.md":
                return b"this file is not production and may change\n"
            if not os.path.isfile(path):
                return None
            with open(path, "rb") as fh:
                return fh.read()

        errs, text = captured(lambda: check(manifest, get_bytes=noise)[0])
        expect_pass("unpinned non-production file may change", errs, text or "")

        # 4. --deployed without fleet.env fails closed, does not skip.
        old_fleet = os.environ.pop("FLEET_ENV", None)
        try:
            # Point at a guaranteed-absent path even if a real fleet.env exists in this checkout.
            os.environ["FLEET_ENV"] = os.path.join(td, "no-such-fleet.env")
            errs, text = captured(lambda: check(manifest, deployed=True)[0])
            expect_fail("deployed mode without fleet.env refuses", errs or [], "DEPLOYED", text or "")
        finally:
            if old_fleet is None:
                os.environ.pop("FLEET_ENV", None)
            else:
                os.environ["FLEET_ENV"] = old_fleet

        # 5. --deployed with a fleet and a mismatched remote hash fails with DEPLOYED, not crash.
        fake_fleet = os.path.join(td, "fleet.env")
        open(fake_fleet, "w").write(
            'SPARK_HOME="/home/spark"\n'
            'SPARK_MGMT_IPS="PRIVATE_MGMT_IP_0 PRIVATE_MGMT_IP_1 PRIVATE_MGMT_IP_2 PRIVATE_MGMT_IP_3"\n'
            'SPARK_USER="spark"\n'
        )
        os.environ["FLEET_ENV"] = fake_fleet

        def bad_remote(host, remote_path):
            return "0" * 64, ""

        errs, text = captured(lambda: check(manifest, deployed=True, hash_remote=bad_remote)[0])
        expect_fail("deployed bytes disagree with pin", errs or [], "DEPLOYED", text or "")
        os.environ.pop("FLEET_ENV", None)

        # S1–S4. candidate_update vs git history. A shallow clone's git log succeeds
        # and returns the tip; treating that as 'no revision matches' is the false
        # claim this checker exists to prevent. Cannot-look must ABORT, never CANDIDATE.
        def _git(repo, *args):
            subprocess.check_call(
                ["git", "-C", repo, "-c", "user.name=selftest", "-c", "user.email=selftest@test",
                 *args],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        def _write_pin_manifest(root, head_digest, candidate_digest):
            rel = "pack.py"
            open(os.path.join(root, "PRODUCTION_MANIFEST.yml"), "w").write(
                "capabilities:\n"
                "  corpus_pack:\n"
                "    status: ADJUDICATED\n"
                f"    entrypoint: {rel}\n"
                "    content_sha:\n"
                f"      {rel}: {head_digest}\n"
                "    candidate_update:\n"
                "      content_sha:\n"
                f"        {rel}: {candidate_digest}\n"
            )

        src_repo = os.path.join(td, "hist-src")
        os.makedirs(src_repo)
        _git(src_repo, "init")
        v1 = b"candidate-bytes-v1\n"
        v2 = b"tree-bytes-v2\n"
        open(os.path.join(src_repo, "pack.py"), "wb").write(v1)
        _git(src_repo, "add", "pack.py")
        _git(src_repo, "commit", "-m", "v1")
        digest_v1 = sha256_bytes(v1)
        open(os.path.join(src_repo, "pack.py"), "wb").write(v2)
        _git(src_repo, "add", "pack.py")
        _git(src_repo, "commit", "-m", "v2")
        digest_v2 = sha256_bytes(v2)

        here = os.getcwd()
        try:
            _write_pin_manifest(src_repo, digest_v2, digest_v1)
            os.chdir(src_repo)
            errs, text = captured(lambda: check("PRODUCTION_MANIFEST.yml")[0])
            expect_pass("S2 full history: candidate_update matches a parent", errs, text or "")

            shallow = os.path.join(td, "hist-shallow")
            subprocess.check_call(
                # Local clones default to --local and ignore --depth, so the
                # history we are hiding would still be present. --no-local is
                # the same shape as actions/checkout@v4 depth 1.
                ["git", "clone", "--no-local", "--depth", "1", src_repo, shallow],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            shallow_flag = subprocess.check_output(
                ["git", "-C", shallow, "rev-parse", "--is-shallow-repository"],
                text=True,
            ).strip()
            if shallow_flag != "true":
                print(
                    f"SELFTEST FAIL: S1 clone is not shallow "
                    f"(--is-shallow-repository={shallow_flag!r})"
                )
                failures += 1
            _write_pin_manifest(shallow, digest_v2, digest_v1)
            os.chdir(shallow)
            errs, text = captured(lambda: check("PRODUCTION_MANIFEST.yml")[0])
            expect_fail(
                "S1 shallow clone with historical candidate_update ABORTS",
                errs or [],
                "ABORT",
                text or "",
            )
            blob = "\n".join(errs or [])
            if "CANDIDATE" in blob:
                print(
                    "SELFTEST FAIL: S1 — shallow history reported as CANDIDATE "
                    "(absence of evidence as a finding):\n" + blob
                )
                failures += 1

            _write_pin_manifest(shallow, digest_v2, digest_v2)
            errs, text = captured(lambda: check("PRODUCTION_MANIFEST.yml")[0])
            expect_pass(
                "S4 shallow clone whose candidate_update matches HEAD does not need history",
                errs,
                text or "",
            )
        finally:
            os.chdir(here)

        nogit = os.path.join(td, "hist-nogit")
        os.makedirs(nogit)
        open(os.path.join(nogit, "pack.py"), "wb").write(v2)
        _write_pin_manifest(nogit, digest_v2, digest_v1)
        try:
            os.chdir(nogit)
            errs, text = captured(lambda: check("PRODUCTION_MANIFEST.yml")[0])
            expect_fail(
                "S5 no git repo: historical candidate_update ABORTS, not CANDIDATE",
                errs or [],
                "ABORT",
                text or "",
            )
            if any("CANDIDATE" in e for e in (errs or [])):
                print("SELFTEST FAIL: S5 reported CANDIDATE without a git history")
                failures += 1
        finally:
            os.chdir(here)

        # A1. Forgotten promotion: ADJUDICATED + authorization naming it. No structural
        # path so UNPINNED does not fire and the only class is AUTHORIZATION.
        a1 = os.path.join(td, "auth_adjudicated.yml")
        open(a1, "w").write(
            "capabilities:\n"
            "  cpt_27b_4node:\n"
            "    status: ADJUDICATED\n"
            "authorization:\n"
            "  capability: cpt_27b_4node\n"
            "  campaign_id: phase2-qwen38\n"
            "  authorized_by: Jesse\n"
        )
        errs, text = captured(lambda: check(a1)[0])
        expect_fail("A1 ADJUDICATED leftover authorization", errs or [], "AUTHORIZATION", text or "")
        if errs and "cpt_27b_4node" not in "\n".join(errs):
            print("SELFTEST FAIL: A1 did not name the capability")
            failures += 1

        # A2 / (3). Valid pre-run window: CANDIDATE + authorization + no receipt. PASS.
        # The inverted predicate ("auth whose campaign_id appears in no receipt") would fail this.
        a2 = os.path.join(td, "auth_candidate.yml")
        open(a2, "w").write(
            "capabilities:\n"
            "  cpt_27b_4node:\n"
            "    status: CANDIDATE_PENDING_PRODUCTION_RUN\n"
            "authorization:\n"
            "  capability: cpt_27b_4node\n"
            "  campaign_id: phase2-qwen38\n"
            "  authorized_by: Jesse\n"
        )
        errs, text = captured(lambda: check(a2)[0])
        expect_pass("A2 CANDIDATE + authorization + no receipt is the live window", errs, text or "")

        # A3. Guard (2): CANDIDATE + authorization + receipt.campaign_id. Consumed leftover.
        a3 = os.path.join(td, "auth_spent.yml")
        open(a3, "w").write(
            "capabilities:\n"
            "  cpt_27b_4node:\n"
            "    status: CANDIDATE_PENDING_PRODUCTION_RUN\n"
            "    receipt:\n"
            "      campaign_id: phase2-qwen38\n"
            "authorization:\n"
            "  capability: cpt_27b_4node\n"
            "  campaign_id: phase2-qwen38\n"
            "  authorized_by: Jesse\n"
        )
        errs, text = captured(lambda: check(a3)[0])
        expect_fail("A3 consumed authorization still present", errs or [], "AUTHORIZATION", text or "")
        if errs and "phase2-qwen38" not in "\n".join(errs):
            print("SELFTEST FAIL: A3 did not name the campaign_id")
            failures += 1

        # A4. Historical receipt WITHOUT campaign_id is not consumption. Same as resolver C5.
        a4 = os.path.join(td, "auth_leftover_receipt.yml")
        open(a4, "w").write(
            "capabilities:\n"
            "  cpt_27b_4node:\n"
            "    status: CANDIDATE_PENDING_PRODUCTION_RUN\n"
            "    receipt:\n"
            "      date: '2026-07-30'\n"
            "      verified_outcome: old narrative\n"
            "authorization:\n"
            "  capability: cpt_27b_4node\n"
            "  campaign_id: phase2-qwen38\n"
            "  authorized_by: Jesse\n"
        )
        errs, text = captured(lambda: check(a4)[0])
        expect_pass("A4 leftover receipt without campaign_id is not consumption", errs, text or "")

        # A5. Authorization names a CANDIDATE; a different cap is ADJUDICATED. (1) does not fire.
        a5 = os.path.join(td, "auth_foreign.yml")
        open(a5, "w").write(
            "capabilities:\n"
            "  corpus_pack:\n"
            "    status: ADJUDICATED\n"
            "  bake_export:\n"
            "    status: CANDIDATE_PENDING_PRODUCTION_RUN\n"
            "authorization:\n"
            "  capability: bake_export\n"
            "  campaign_id: bake-1\n"
            "  authorized_by: Jesse\n"
        )
        errs, text = captured(lambda: check(a5)[0])
        expect_pass("A5 ADJUDICATED ignores authorization that names a different capability", errs, text or "")

        # A6. Plural authorizations: same as singular for (1).
        a6 = os.path.join(td, "auths_plural.yml")
        open(a6, "w").write(
            "capabilities:\n"
            "  cpt_27b_4node:\n"
            "    status: ADJUDICATED\n"
            "authorizations:\n"
            "  - capability: cpt_27b_4node\n"
            "    campaign_id: phase2-qwen38\n"
            "    authorized_by: Jesse\n"
        )
        errs, text = captured(lambda: check(a6)[0])
        expect_fail("A6 authorizations list leftover on ADJUDICATED", errs or [], "AUTHORIZATION", text or "")

        # A7. Residual exactly: ADJUDICATED + authorization, no campaign_id on anything. (1) still fires.
        a7 = os.path.join(td, "auth_no_campaign.yml")
        open(a7, "w").write(
            "capabilities:\n"
            "  cpt_27b_4node:\n"
            "    status: ADJUDICATED\n"
            "authorization:\n"
            "  capability: cpt_27b_4node\n"
            "  authorized_by: Jesse\n"
        )
        errs, text = captured(lambda: check(a7)[0])
        expect_fail("A7 forgotten promotion omitting campaign_id still FAIL", errs or [], "AUTHORIZATION", text or "")

    if failures:
        print(f"SELFTEST: {failures} control(s) failed — the gate cannot be trusted")
        return 1
    print("SELFTEST: all negative controls failed the check (correctly); crash distinguished from detection")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("WHY THIS EXISTS.", 1)[0])
    ap.add_argument("--manifest", default="PRODUCTION_MANIFEST.yml")
    ap.add_argument(
        "--deployed",
        action="store_true",
        help="also hash $SPARK_HOME/palios-training/<path> on each Spark (requires fleet.env)",
    )
    ap.add_argument("--self-test", action="store_true", help="prove the gate can fail for the right reason")
    a = ap.parse_args()
    if a.self_test:
        return selftest()
    if not os.path.isfile(a.manifest):
        print(f"ABORT: {a.manifest} not found", file=sys.stderr)
        return 1
    failures, checked, n_caps, n_contested = check(a.manifest, deployed=a.deployed)
    return report(failures, checked, n_caps, n_contested, a.manifest)


if __name__ == "__main__":
    sys.exit(main())
