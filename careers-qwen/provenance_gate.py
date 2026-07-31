#!/usr/bin/env python3
"""provenance_gate.py — RELEASE GATE for an SFT corpus. Independent joins, no self-reference.

History of this file matters, because v1 was WORSE than the heuristic auditor it replaced:
it globbed `--seeds/**/*.jsonl`, which indexed the CANDIDATE, the SOURCE and the LEDGER, so
rows resolved against THEMSELVES and the PASS was circular. Its "16 R5b exact matches" were
entirely that artifact — real seed rows carry `situation`/`right_way`, not `messages`, so
hashing `row["messages"]` made every true seed hash as empty and match nothing.
(tutor-codex FULL STOP, 2026-07-24, five defects; all five fixed here.)

THE JOINS — each must be independently provable against a curated seed or the source corpus:
  R4  captured action   -> candidate row must be an EXACT canonical full-row member of --source
  R6  mechanical trace  -> must join a captured source action on (view,seq,primitive)
                           AND share the identical messages[0] user prompt
  R5  derived row       -> meta.derived_from must RESOLVE to a curated seed's top-level
                           provenance_hash, AND the candidate messages must EQUAL that seed's
                           canonical payload [user: situation, assistant: right_way]
  R5b seed-stamped row  -> must match a curated seed's canonical payload exactly; if the seed
                           carries no provenance_hash that limitation is counted SEPARATELY
                           and named in the manifest (never silently accepted)

Seed files are an EXPLICIT ALLOW-LIST (repeatable --seed-file). Any seed path that realpath-
collides with the candidate, source, ledger or manifest is REJECTED — self-indexing cannot
recur by construction.

Exit 0 = PASS. Exit 1 = FAIL. Never exits 0 with unresolved rows.

Usage:
  python3 provenance_gate.py \
    --candidate <candidate.jsonl> --source <source.jsonl> \
    --seed-file <curated_seed.jsonl> [--seed-file ...] \
    --ledger <ledger.jsonl> --manifest <manifest.json> [--expect-rows N]

  --ledger is REQUIRED: pinned by sha in the manifest, but its classifications are never
  trusted — every join is recomputed independently. All four target paths must be distinct.
"""
import argparse, hashlib, json, os, sys, collections


def sha256_of(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def canonical_payload(seed):
    """The payload derive_training_rows.py builds from a seed: situation -> right_way.
    Seeds have NO `messages` field; this reconstructs what a derived row must contain."""
    return [
        {"role": "user", "content": seed.get("situation")},
        {"role": "assistant", "content": seed.get("right_way")},
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--seed-file", action="append", required=True,
                    help="repeatable; EXPLICIT curated-seed allow-list (no globbing)")
    ap.add_argument("--ledger", required=True,
                    help="the audit ledger; pinned in the manifest (the gate never trusts its rows)")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--expect-rows", type=int, default=0)
    a = ap.parse_args()

    # ---- anti-self-reference, TWO checks (v4) ----
    # (a) the four audit targets must be four DISTINCT files. v3 built this set but never
    #     tested its members against each other, so --ledger==--candidate PASSED and
    #     --manifest==--candidate would READ then OVERWRITE the corpus at json.dump.
    #     Checked BEFORE any load or write. (tutor-codex, 2026-07-24 — claim/code mismatch:
    #     the commit message asserted this was enforced when only seeds were being checked.)
    targets = {"candidate": a.candidate, "source": a.source, "ledger": a.ledger, "manifest": a.manifest}
    rp_targets = {k: os.path.realpath(v) for k, v in targets.items()}
    if len(set(rp_targets.values())) != 4:
        dupes = collections.Counter(rp_targets.values())
        for path, n in dupes.items():
            if n > 1:
                names = sorted(k for k, v in rp_targets.items() if v == path)
                print(f"GATE FAIL: {' and '.join(names)} are the SAME file ({os.path.basename(path)}) "
                      f"— audit targets must be distinct")
        sys.exit(1)

    # (b) a seed may never be an audit target
    forbidden = set(rp_targets.values())
    for s in a.seed_file:
        rp = os.path.realpath(s)
        if rp in forbidden:
            print(f"GATE FAIL: seed file {s} collides with an audit target — self-indexing refused")
            sys.exit(1)

    cand, src = load(a.candidate), load(a.source)

    # ---- indexes built ONLY from the curated allow-list ----
    seed_by_ph, seed_payloads, seed_no_ph, seed_rejected = {}, {}, 0, []
    seed_fatal = []
    for p in a.seed_file:
        for idx, sd in enumerate(load(p)):
            # (2) every allowed seed must BE a curated operator correction with real content.
            if sd.get("schema") != "operator_correction_v1" or sd.get("curated") is not True:
                seed_rejected.append(f"{os.path.basename(p)}#{idx}: not schema=operator_correction_v1+curated")
                continue
            if not sd.get("situation") or not sd.get("right_way"):
                seed_rejected.append(f"{os.path.basename(p)}#{idx}: empty situation/right_way")
                continue
            pay_sha = sha256_of(canonical_payload(sd))
            seed_payloads[pay_sha] = p
            ph = sd.get("provenance_hash")
            if ph:
                prior = seed_by_ph.get(ph)
                if prior and prior[1] != pay_sha:
                    # last-write-wins would silently pick a payload; refuse instead.
                    seed_fatal.append(f"seed provenance_hash {ph} maps to TWO different payloads")
                seed_by_ph[ph] = (p, pay_sha)
            else:
                seed_no_ph += 1
    if seed_fatal:
        for f in seed_fatal:
            print("GATE FAIL:", f)
        sys.exit(1)

    # ---- source indexes: exact rows, and captured actions by coordinates+prompt ----
    src_rows = {sha256_of(r) for r in src}
    captured = {}
    for r in src:
        m = r.get("meta", {})
        if str(m.get("origin", "")).lower() == "captured":
            key = (str(m.get("view")), str(m.get("seq")), str(m.get("primitive")))
            prompt = (r.get("messages") or [{}])[0].get("content")
            captured.setdefault(key, set()).add(prompt)

    failures, rules = [], collections.Counter()
    hashes, seen = [], {}

    for i, r in enumerate(cand):
        m = r.get("meta", {})
        # SCHEMA CONFORMANCE (added 2026-07-24): provenance proves a row is REAL; it does not
        # prove it is TRAINABLE. A row authored in a legacy {input,output} shape passed every
        # provenance join and would have been skipped or thrown by the SFT tokenizer, which
        # builds from `messages`. Prove both.
        msgs = r.get("messages")
        if not isinstance(msgs, list) or len(msgs) < 2 \
           or not all(isinstance(x, dict) and x.get("role") and x.get("content") for x in msgs):
            failures.append(f"row {i}: SCHEMA — needs messages[>=2] with role+content "
                            f"(found keys {sorted(r.keys())})")
            continue
        h = sha256_of(r)
        if h in seen:
            failures.append(f"row {i}: DUPLICATE of row {seen[h]}")
        seen[h] = i
        hashes.append(h)

        origin = str(m.get("origin", "")).lower()
        kind = str(m.get("kind", ""))
        df = m.get("derived_from")
        my_prompt = (r.get("messages") or [{}])[0].get("content")
        my_payload_sha = sha256_of(r.get("messages", []))

        if origin == "captured":
            # R4: exact full-row membership in the source corpus — not self-declared meta
            if h in src_rows:
                rules["R4-exact-source-member"] += 1
            else:
                failures.append(f"row {i}: R4 claims captured but is NOT an exact row in --source")

        elif kind.startswith("no-think") or "tool-call" in kind:
            key = (str(m.get("view")), str(m.get("seq")), str(m.get("primitive")))
            prompts = captured.get(key)
            if prompts is None:
                failures.append(f"row {i}: R6 no captured action at {key}")
            elif my_prompt not in prompts:
                failures.append(f"row {i}: R6 joins {key} but user prompt does NOT match the captured action")
            else:
                rules["R6-tuple+exact-prompt"] += 1

        elif df:
            hit = seed_by_ph.get(df)
            if not hit:
                failures.append(f"row {i}: R5 derived_from={str(df)[:20]} UNRESOLVED in the curated allow-list")
            elif my_payload_sha != hit[1]:
                failures.append(f"row {i}: R5 seed hash resolves but MESSAGES DIFFER from the seed payload "
                                f"(fabricated answer would land here)")
            else:
                rules["R5-seedhash+exact-payload"] += 1

        elif m.get("class") and m.get("recurrence") is not None:
            if my_payload_sha in seed_payloads:
                rules["R5b-exact-seed-payload"] += 1
            else:
                failures.append(f"row {i}: R5b class+recurrence stamp but payload matches NO curated seed")
        else:
            failures.append(f"row {i}: NO resolvable provenance (meta keys {sorted(m.keys())})")

    if a.expect_rows and len(cand) != a.expect_rows:
        failures.append(f"ROW COUNT {len(cand)} != expected {a.expect_rows}")

    # (4) PORTABLE: never write operator-absolute paths into a public repo — they leak the
    # umbilical and make the manifest unreproducible on another checkout. Record each file
    # relative to the common governed root, and label the root itself by env var name.
    roots = [os.path.dirname(os.path.realpath(x)) for x in [a.candidate, a.source, a.ledger] + a.seed_file]
    common = os.path.commonpath(roots) if len(set(roots)) > 1 else roots[0]

    def rel(x):
        return os.path.relpath(os.path.realpath(x), common)

    manifest = {
        "gate": "provenance_gate.py (v3 — allow-list + seed schema validation + portable paths)",
        "governed_root": "$TRAINING_DATA_ROOT (paths below are relative to it; absolute paths "
                         "are deliberately NOT recorded — public repo)",
        "candidate": {"path": rel(a.candidate), "sha256": file_sha(a.candidate), "rows": len(cand)},
        "source": {"path": rel(a.source), "sha256": file_sha(a.source), "rows": len(src)},
        "ledger": {"path": rel(a.ledger), "sha256": file_sha(a.ledger), "rows": len(load(a.ledger))},
        "curated_seed_allow_list": [{"path": rel(p), "sha256": file_sha(p)} for p in a.seed_file],
        "seed_rows_rejected_by_schema_validation": seed_rejected,
        "rule_counts": dict(rules),
        "unique_row_hashes": len(set(hashes)),
        "row_sha256_full": hashes,
        "limitations": [
            f"{seed_no_ph} curated seed row(s) carry no provenance_hash; those are provable only by "
            "exact canonical-payload match (counted as R5b-exact-seed-payload), never by hash",
            "the ledger is PINNED by sha but its classifications are NOT trusted — every join in "
            "this gate is computed independently from candidate/source/seeds",
        ],
        "verdict": "PASS" if not failures else "FAIL",
        "failures": failures,
    }
    json.dump(manifest, open(a.manifest, "w"), indent=1)

    print(f"rows={len(cand)} unique={len(set(hashes))} seeds={len(a.seed_file)} file(s)")
    for k, v in sorted(rules.items()):
        print(f"  {k}: {v}")
    if failures:
        print(f"\nGATE FAIL — {len(failures)} unresolved:")
        for f in failures[:12]:
            print("  ", f)
        print(f"manifest -> {a.manifest}")
        sys.exit(1)
    print(f"\nGATE PASS — every row independently resolved. manifest -> {a.manifest}")
    sys.exit(0)


if __name__ == "__main__":
    main()
