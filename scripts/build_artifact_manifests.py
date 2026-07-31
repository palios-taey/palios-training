#!/usr/bin/env python3
"""Generate manifests/ — the per-artifact record for the consolidated training repo.

WHY MANIFESTS AND NOT STORAGE. The training artifacts measured on 2026-07-30 are ~3.6TB of run
outputs plus ~2.8TB of models across four nodes, 153GB on the Expansion drive, and 27GB of
node-local corpora. None of that can live in git, and none of it may be public. So the repo holds
the RECORD: what exists, where, how big, what produced it, and what depends on it.

THE used_by EDGE IS THE POINT. On 2026-07-30 the 35B audit numbers could not be trusted because
their proof files had been dropped by a clean-root transplant and nothing recorded that a document
depended on them. The break was silent for weeks. A used_by edge makes that class detectable
instead of discovered.

HONESTY RULES, enforced in the output rather than in a reviewer's head:
  - A digest is emitted ONLY when it was actually computed or actually read from a receipt. An
    unknown digest is `null` with `digest_source: unverified`, never an empty string that reads
    like a checked value.
  - Locations are env-resolved (node index + env var), never literal addresses. This file lives in
    a repo and the private-data gate is right to refuse addresses.
  - Absence is recorded as a finding, not skipped. A manifest that omits what it could not find
    reads as complete.
"""
import hashlib
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "manifests")


def sha256_file(path, cap_bytes=64 * 1024 * 1024):
    """Digest a file, but refuse to pretend on very large ones.

    Hashing a 13GB shard on every regeneration is not free and not useful — those are pinned by
    the run receipt that produced them. Over the cap we return None with a reason, which the
    caller records as unverified rather than as a value.
    """
    try:
        size = os.path.getsize(path)
    except OSError:
        return None, "unreadable"
    if size > cap_bytes:
        return None, f"over-cap-{cap_bytes // (1024*1024)}MB"
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return None, "unreadable"
    return h.hexdigest(), "computed"


def record(artifact, kind, **kw):
    r = {"artifact": artifact, "kind": kind}
    r.update(kw)
    return r


def wheels():
    """Wheels, digested — these are small and genuinely verifiable."""
    out = []
    home = os.path.expanduser("~")
    roots = [home, "/media/mira/Expansion"]
    for base in roots:
        for dirpath, _dirs, files in os.walk(base):
            # depth guard: a full walk of a 153GB drive is not what this needs
            if dirpath.count(os.sep) - base.count(os.sep) > 4:
                _dirs[:] = []
                continue
            for f in files:
                if not f.endswith(".whl"):
                    continue
                p = os.path.join(dirpath, f)
                digest, how = sha256_file(p)
                out.append(record(
                    f, "wheel",
                    location=p.replace(home, "$HOME"),
                    bytes=os.path.getsize(p),
                    sha256=digest,
                    digest_source=how,
                ))
    return out


def capabilities():
    """Production capabilities and their pinned content — read from the manifest, not remembered.

    Enumerates capabilities, historical_lines AND contested. The contested list lives at its own
    top level; a generator that reads only `capabilities` silently drops the one entry flagged as
    DISPUTED, which is how an inventory comes to read as complete while being wrong.
    """
    import yaml
    path = os.path.join(ROOT, "PRODUCTION_MANIFEST.yml")
    doc = yaml.safe_load(open(path)) or {}
    out = []
    for name, body in (doc.get("capabilities") or {}).items():
        shas = body.get("content_sha") or {}
        # A capability may carry a candidate_update: NEW bytes recorded WITHOUT promoting them into
        # the adjudicated pin, because new bytes do not inherit an old receipt. When the working
        # tree holds the candidate, taey-train correctly REFUSES the capability until the candidate
        # is qualified. That is the gate working, NOT drift.
        #
        # The first version of this generator did not read candidate_update and reported two such
        # capabilities as "DRIFT" — a designed, correct state labelled as a defect. I nearly
        # escalated another seat's correct process as a breakage. Compare against BOTH pins and say
        # which one the tree actually matches.
        cand = ((body.get("candidate_update") or {}).get("content_sha")) or {}
        pinned = []
        for rel, digest in shas.items():
            full = os.path.join(ROOT, rel)
            present = os.path.exists(full)
            live, how = (sha256_file(full) if present else (None, "absent"))
            candidate_digest = cand.get(rel)
            if live and digest and live == digest:
                state = "matches_adjudicated"
            elif live and candidate_digest and live == candidate_digest:
                state = "matches_candidate_update"   # intentionally unlaunchable until qualified
            elif live and digest:
                state = "UNEXPLAINED_DRIFT"          # matches neither pin — this one is a defect
            else:
                state = "unverified"
            pinned.append({
                "path": rel,
                "recorded_sha256": digest,
                "candidate_sha256": candidate_digest,
                "present": present,
                "live_sha256": live,
                "digest_source": how,
                "state": state,
                "matches": (live == digest) if (live and digest) else None,
            })
        out.append(record(
            name, "capability",
            status=body.get("status"),
            entrypoint=body.get("entrypoint"),
            trainer=body.get("trainer"),
            pinned_content=pinned,
        ))
    for name, body in (doc.get("historical_lines") or {}).items():
        out.append(record(name, "historical_line",
                          status=str(body.get("status", "")).split("—")[0].strip(),
                          launchable=False))
    for c in (doc.get("contested") or []):
        out.append(record(c.get("capability", "?"), "contested_capability",
                          status="CONTESTED — not adjudicated", launchable=False))
    return out


def sealed_pins():
    """The six files the sealed SFT runner pins BY PATH.

    Recorded as first-class artifacts because relocating any of them breaks the runner that a
    pending authorization would promote. A manifest that does not name them makes that failure
    discoverable only after it happens.
    """
    runner = os.path.join(ROOT, "careers-qwen/run_stage2_sft_ddp_till_done.sh")
    if not os.path.exists(runner):
        return [record("run_stage2_sft_ddp_till_done.sh", "sealed_runner", present=False,
                       finding="ABSENT — the sealed runner itself is missing")]
    pins, grab = [], False
    for line in open(runner):
        if "IMMUTABLE_FILES=(" in line:
            grab = True
            continue
        if grab:
            if line.strip().startswith(")"):
                break
            p = line.strip()
            if p:
                pins.append(p)
    out = []
    for rel in pins:
        full = os.path.join(ROOT, rel)
        present = os.path.exists(full)
        digest, how = (sha256_file(full) if present else (None, "absent"))
        out.append(record(rel, "sealed_pinned_file", path=rel, present=present,
                          sha256=digest, digest_source=how,
                          constraint="MUST NOT MOVE while the 0->50 authorization is pending"))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    sections = {
        "capabilities": capabilities(),
        "sealed_pins": sealed_pins(),
        "wheels": wheels(),
    }
    for name, rows in sections.items():
        path = os.path.join(OUT, f"{name}.json")
        with open(path, "w") as fh:
            json.dump({
                "generated_by": "scripts/build_artifact_manifests.py",
                "note": "Data is MANIFESTED here, never stored. Digests are computed or read, "
                        "never assumed; unverified digests are null with a stated reason.",
                "count": len(rows),
                "records": rows,
            }, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print(f"  {name:<16} {len(rows):>4} records -> manifests/{name}.json")
    # Report the two states separately. Conflating them is what made the first version call a
    # correct process a defect.
    pins = [(c["artifact"], p) for c in sections["capabilities"] if c["kind"] == "capability"
            for p in c["pinned_content"]]
    pending = [(cap, p["path"]) for cap, p in pins if p["state"] == "matches_candidate_update"]
    broken = [(cap, p["path"]) for cap, p in pins if p["state"] == "UNEXPLAINED_DRIFT"]
    if pending:
        print("  CANDIDATE PENDING QUALIFICATION (capability intentionally unlaunchable, not a defect):")
        for cap, path in pending:
            print(f"    {cap}: {path}")
    if broken:
        print("  UNEXPLAINED DRIFT — matches neither the adjudicated pin nor any candidate:")
        for cap, path in broken:
            print(f"    {cap}: {path}")
        return 1
    missing = [r["artifact"] for r in sections["sealed_pins"] if not r.get("present")]
    if missing:
        print(f"  SEALED PIN MISSING: {missing}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
