#!/usr/bin/env python3
"""Production CPT corpus packer — registered inputs ONLY (governance: tutor packs, never assembles).

Merges the treasurer-registered Cat-1/2/3 CPT slices (2026-07-14 delivery, commits through
b2e33e3 + 745759c + af1f67a on palios-taey/treasurer) into the fixed-length packed format the
proven 27B FSDP2 fit stack trains on ({"input_ids": [SEQ]} rows, MAX_SEQ=2560).

Method (deterministic, no shuffle, reproducible byte-for-byte):
  1. Read slices in the canonical order listed in SLICES (registry order).
  2. Verify each file's sha256:16 against its REGISTERED value — hard abort on mismatch.
  3. Tokenize each row's `text` (add_special_tokens=False), append EOS after every doc.
  4. Concatenate into one stream; emit consecutive SEQ-token blocks. The final partial block is
     CYCLE-PADDED with the start of the stream (Jesse 2026-07-14: NO truncation — zero corpus
     tokens dropped; the pad is a documented replay of already-included content, not new data).

Output: cpt_production_v1_packed_<SEQ>.jsonl + stdout sha256 for the treasurer registry row.
      (SEQ is PACK_SEQ, default 2560; the emitted name and manifest carry the ACTUAL value)
Run on a Spark node (needs the Qwen3.6-27B tokenizer):
  python3 pack_production_corpus.py --slices-dir /var/spark/isma/training/slices_v1 \
      --tokenizer <SPARK_HOME>/models/Qwen3.6-27B \
      --out /var/spark/isma/training/cpt_production_v1_packed_2560.jsonl
"""
import argparse, json, os, re, sys

from corpus_manifest import SCHEMA, sha256_file, write_manifest

# Packed block length, in tokens. DEFAULT UNCHANGED at 2560 — this is a parameterisation,
# not a recipe change; the value is a recipe parameter and recipe parameters are
# Chats-researched, never chosen here.
#
# WHY IT IS NO LONGER A BARE CONSTANT. As a hardcoded literal this silently became the
# binding constraint on every CPT run, and it sits BELOW what the trainer is audited for:
# dense-9b/recipes/launch_cpt_qwen36_27b_fsdp.sh:112 carries
# `MAX_SEQ="${MAX_SEQ:-4096}"  # infra-audited default 2026-07-07`, while this packer emitted
# 2560-token blocks and bake_27b.sh then passed MAX_SEQ=2560 to match the corpus. The audited
# window was never exercised, and nothing in the tree recorded that as a decision — the number
# was simply carried forward. Measured against the repo content a CPT round would consume,
# 27% of files exceed 2560 and get split across blocks; the largest is ~45,000 tokens, i.e. 18
# blocks, so the model sees its lines and never its structure.
#
# Override with PACK_SEQ. Any change to the default is a recipe decision requiring a Chats
# round and a re-pack, and the emitted manifest records `sequence_length` so a corpus always
# carries the value it was packed at.
SEQ = int(os.environ.get("PACK_SEQ", "2560"))
# (filename, registered_rows, registered_sha256_16) — from treasurer REGISTRY.md rows.
# CORPUS V2 pack list (treasurer registry commit 161a7ca9, 8 inputs sha-gated) — the full-mandate
# corpus: repos(19) + career + VOICE + strategy + research. Supersedes the v1 6-slice list.
# V3 PACK LIST (2026-07-28) — the SANCTIONED set, which is the 6 slices treasurer classifies
# `cpt-slice` in build_pairs_manifest.py STATUS, totalling 33,561 rows. It is NOT the 8-entry V2
# list above it in history: cpt_public_repos_v2.jsonl and cpt_strategy_research_delta_v1.jsonl are
# registered in REGISTRY.md but exist on NO machine — not Mira, not any Spark — so a pack against
# the V2 list hard-aborts on a missing file. Verified 2026-07-28 before this edit.
#
# voice_cpt_slice is the SCRUBBED derivative, and that substitution is the load-bearing part:
# the original carried 9 occurrences of 7 distinct live-shaped credentials (Anthropic keys via
# `x-api-key:`/`ANTHROPIC_API_KEY=`, GitHub PATs via `ghp_`), captured from transcripts where the
# operator pasted them. A credential trained into 27B parameters cannot be scrubbed afterwards —
# only retrained away. The scrub replaces each value with treasurer's own
# [REDACTED-LEAKED-KEY-ROTATE] marker (already present x2 in that same file from the 07-27 pass),
# preserves row count exactly 31,920 -> 31,920, and leaves 0 credential shapes.
#
# The 07-27 credential scrub changed voice_cpt_slice's bytes WITHOUT changing its row count, so its
# then-registered sha 352c3b6f12d7d216 stopped matching the on-disk 2f1cb06da65ba497 while the row
# count still read 31,920. ONLY the sha gate caught that; every count-based check passed it silently
# — and behind that mismatch sat 7 live credentials the 07-27 pass had missed. That is why the gate
# is sha-based and not count-based.
# 2026-07-28: those 7 values are scrubbed and the slice is re-registered at 0919e05013d3ef69.
# Treasurer scrubbed the ORIGINAL in place and tutor produced the _SCRUBBED derivative
# independently; both landed byte-identical at that sha, which is the cross-check.
# V4 PACK LIST (2026-07-28, CORRECTED). The V3 list above dropped cpt_public_repos_v2 and
# cpt_strategy_research_delta_v1 on the finding that they "exist on NO machine". THAT FINDING WAS
# WRONG — the search covered the governed store and /var/spark only. Both files live in a treasurer
# peer worktree and match their REGISTERED shas exactly (1245/160110c1, 147/4901efd5). Dropping
# them cost 1,175 packed blocks: the run trained 2,511 blocks against the 3,686 the previous
# production corpus carried, i.e. two thirds of the corpus, while the directive was to retrain
# everything. A file is not absent because one search did not find it.
#
# cpt_strategy_research_delta is the SCRUBBED derivative: the original carried a live Anthropic key
# and a live Google key inside `.env` write-out blocks — and the Anthropic value (sha8 3a90395c) is
# the SAME key the voice slice carried, so one credential was sitting in two separate corpus files.
# CREDENTIAL SCRUB 2026-07-29 — three slice shas re-registered. Row counts preserved EXACTLY
# (330/1245/946 unchanged, 3033 rows all still valid JSON), so this is a credential scrub and not
# a content change; the diff should read that way.
#   cpt_raw_corpus_v4     0388166bee405dd9 -> fd64cb08341238ea   (3,906 bytes)
#   cpt_public_repos_v2   07b88feceba6dee5 -> 871620fa1ca354f8   (2,568 bytes)
#   cpt_consultations_v1  c1840e0a0fe46108 -> 92e6aeef82318b8f  (47,418 bytes)
# WHAT WAS REMOVED: AWS presigned-URL material the earlier passes left behind — 8 distinct
# x-amz-security-token values (longest 1,346 chars, full-length STS session tokens) and 8
# presigned Signature values. Every AWSAccessKeyId was ALREADY redacted by the earlier pass, so
# there was nothing to authenticate as; STS session tokens are time-limited and these came from
# months-old documents. Expired, unusable — and removed anyway rather than trained again.
# SCRUBBED WITH treasurer/scripts/secret_scan.py (commit 5b7bd60c), which redacts NAMED patterns
# only and refuses to write if the line count would change. Its 22 remaining ENTROPY candidates
# are triaged false positives: LinkedIn/YouTube/Glassdoor URLs, a Drive file id, and code
# identifiers like FRESHNESS_HARD_CAP_HOURS=48.0.
# PRE-SCRUB COPIES: slices_v2_probe.prescrub_20260729/ (the slices are not git-tracked, so the
# filesystem copy IS the reversal path).
# CPT v7 trained on the PRE-scrub bytes. Its first provenance record was wrong because it read this
# later source state instead of a receipt emitted with the packed artifact. The sidecar written by
# this packer is now the lineage authority; code state and later environment captures are not.
SLICES = [
    ("cpt_raw_corpus_v4.jsonl", 946, "fd64cb08341238ea"),
    ("cpt_public_repos_v2.jsonl", 1245, "871620fa1ca354f8"),
    ("cpt_careers_kb_v1.jsonl", 316, "ac151e024a3918fa"),
    ("cpt_careers_db_worldmodel_v1.jsonl", 33, "bb80a36f0caf3536"),
    ("cpt_consultations_v1.jsonl", 330, "92e6aeef82318b8f"),
    ("cpt_recaps_v1.jsonl", 16, "a1079b6f37dd848b"),
    ("cpt_strategy_research_delta_v1_SCRUBBED.jsonl", 147, "0a81a0af5e58181d"),
    # VOICE CORPUS REMOVED FROM PRODUCTION 2026-07-29 (Jesse directive). Do not re-add without
    # a rebuilt schema. Reasons, all measured:
    #   - NO SPEAKER LABEL. 24 meta fields (quality_tier, freshness_class, public_safe...) and
    #     zero role/speaker/author/turn. The corpus cannot distinguish Jesse's words from the
    #     assistant's from third-party content he pasted in. Sampled rows include a 3,047-char
    #     assistant response and a quoted LinkedIn post sitting in a corpus meant to teach HIS voice.
    #   - NOT SELECTED. 32,169 source rows -> 31,920 kept. 0.77% dropped, and that filter was for
    #     consult-dispatch markers, not content. A style prior built by taking everything.
    #   - DISPROPORTIONATE. 91.3% of corpus rows, 38.1% of the actual text, at 440 chars/row —
    #     conversational fragments weighted equally with authored documents.
    #   - It carried the live Stripe/Anthropic/GitHub/HF credentials, because it is raw transcript.
    # WRONG OBJECTIVE ANYWAY: CPT on this teaches the model to CONTINUE text like it. "Respond the
    # way Jesse would" is SFT with his turns as the assistant response, sharpened by DPO. Both
    # require knowing which sentences are his, which this schema cannot express.
]

# ---------------------------------------------------------------------------
# REGISTERED PACK SETS
# ---------------------------------------------------------------------------
# SLICES above stays the production_v1 blend, untouched — it is the proven set and its pins are
# the lineage authority for everything already trained.
#
# repos_v1 (2026-08-01, Jesse directive "only the public repos they use"): the public-repo surface
# ALONE. Not a blend. Built by build_corpus_slices.py public-repos-v2 with the retired-directory
# exclusion, so archive/deprecated/superseded content is absent — "the public repos need to be
# clean, that means nothing out of date ever". 19 repos including governance.
PACK_SETS = {
    "production_v1": SLICES,
    "repos_v1": [
        ("cpt_public_repos_v2.jsonl", 1098, "155dc385fb92ed47"),
    ],
    # PRODUCTION SET, 2026-08-02. The repos Taey actually runs on, named by Jesse after the V2
    # corpus trained education and research repos he does not use. Ten repos, and it is LARGER
    # than the 19-repo set it replaces (1,688 rows vs 1,098) because two surfaces Taey is actually
    # driven through had never been in any corpus: apply-machine contributes 709 rows — 42% of
    # this corpus — and linkedin was absent entirely. Training more repos was training less of
    # the right material.
    # Credential-scanned 2026-08-02 with treasurer/scripts/secret_scan.py: 0 NAMED matches; 9
    # entropy candidates triaged to code identifiers and public URLs (linkedin.com/posts activity
    # ids, a tracxn company id), verified by reading them.
    # v2 of the production slice, 2026-08-02. The 1,688-row build was KILLED mid-run and rebuilt:
    # 600 of its rows were apply-machine/bundles/ — the model's own generated resumes and cover
    # letters, 36% of the whole corpus — and 5 came from bundles/_QUARANTINE_fabricated/. Training
    # on generated output reinforces whatever the output already got wrong, which is why Taey kept
    # ignoring a resume standard that was correct and injected on every compose: it had been
    # trained on 76 examples of that standard being violated.
    # Rebuilt with bundles/ excluded at the extractor (treasurer + conductor approved), verified BY
    # SOURCE PATH not row count: 0 from bundles/, 0 from quarantine/fabricated.
    # Cleared on the credential axis too: 0 of the 7 infra-flagged credential files are reachable —
    # they live under treasurer (not one of the 10 repos) AND are .jsonl, which this allowlist does
    # not admit. secret_scan.py: 0 NAMED, 6 entropy candidates all read and triaged to identifiers.
    "prod_v1": [
        ("cpt_public_repos_prod.jsonl", 1089, "82a117538c31ecac"),
    ],
    # prod_v3, 2026-08-17 (Jesse: "train the Qwen3.8 the same CPT content as current model in
    # production with updated production repo content"). SAME ten-repo set, SAME extractor, SAME
    # exclusions as prod_v1 — only the repo content is newer. Named v3 because cpt_prod_v2_packed_*
    # already exists on the nodes; a colliding name would overwrite a real artifact.
    # Rebuilt by build_corpus_slices.py public-repos-prod after fetching all ten to their pinned
    # production refs. Two moved: taey-presence -> c05d8be, apply-machine -> 13a1b60 (its
    # ats-submit-subtraction override); governance -> 1f1415f.
    # 1,167 rows / 815 docs / 10,199,092 chars, +78 vs prod_v1. Growth is where the work was:
    # taeys-hands 280, palios-training 259, taey-presence 125.
    # RE-VERIFIED with prod_v1's own methods rather than assumed to still hold:
    #   bundles/ rows BY SOURCE PATH: 0    quarantine/fabricated: 0
    #   treasurer/scripts/secret_scan.py: 0 NAMED, 6 ENTROPY — same count and class prod_v1 triaged,
    #   all read and all Python identifiers (_strict_children, UniqueKeyLoader.add_constructor,
    #   splitlines(), workday_job).
    "prod_v3": [
        ("cpt_public_repos_prod.jsonl", 1167, "779b4234936bc9fe"),
    ],
    # PROD_V4, 2026-08-19. Same ten-repo PUBLIC_REPOS_PROD set, re-extracted from the repos as
    # they stand today, on Jesse's instruction: "this is just the repos right now, no personal
    # stuff job applications... Repos and the current documentation that is in them, not just the
    # code." Personal/careers material is layered on in a later module, not this base.
    #
    # 1,226 rows / 857 docs / 10,719,871 chars, +59 rows vs prod_v3. Composition verified after
    # extraction rather than assumed: exactly the 10 PROD repos, 31.7% documentation (.md 383,
    # .txt 6) and 68.3% code/config (.py 645, .sh 117, .yml 35, .yaml 31, .toml 9). No careers,
    # cover-letter, or bundles/ content — the personal slices are simply not in this pack set.
    #
    # KNOWN GAP, recorded rather than silently carried: linkedin contributes ONE row. Its repo
    # tracks only .gitignore and CLAUDE.md; scripts/ and foundations/ are untracked or ignored, so
    # the LinkedIn operational surface cannot reach a corpus built from git HEAD blobs. apply-machine
    # and isma-core also have uncommitted CLAUDE.md/AGENTS.md edits that are invisible for the same
    # reason. Extraction is commit-pinned by design; the fix is committing that work, not loosening
    # the extractor.
    "prod_v4": [
        ("cpt_public_repos_prod.jsonl", 1226, "e549870b892d2f72"),
    ],
    # Forward correction, 2026-08-20. Exact same ten source commits as prod_v4, rebuilt through
    # treasurer's source-locked selector after its existing no-tests rule was extended to cover
    # replay_*.py verification harnesses. Historical prod_v4 stays registered above because V5
    # trained on those exact bytes; this entry governs only future packing.
    "prod_v5": [
        ("cpt_public_repos_prod.jsonl", 1209, "fff6dae26ad02e51"),
    ],
}

PACKED_CONTENT_POSITIVE_CONTROL = "# repo:"
PACKED_CONTENT_FORBIDDEN = (
    "WRITE DISPATCH REPLAY",
    "REPLAY: PASS",
)
# Class identity for replay harnesses. Do not enumerate filenames here: a literal
# list is how 6 of 13 replay_*.py rows survived the previous packed-content gate.
REPLAY_HARNESS_DOC_ID = re.compile(r"::replay_[^/]+\.py$")
REPLAY_HARNESS_HEADER = re.compile(
    r"# repo: \S+ file: replay_[^/\s]+\.py(?=\n|$)"
)
_HEADER_DECODE_CARRY = 128


def _count_new_subsequence(values, needle, carry_length):
    if not needle:
        raise ValueError("packed-content marker tokenized to an empty sequence")
    count = 0
    last_start = len(values) - len(needle)
    for start in range(last_start + 1):
        if start + len(needle) <= carry_length:
            continue
        if values[start:start + len(needle)] == needle:
            count += 1
    return count


def _count_new_regex(text, pattern, carry_length):
    count = 0
    for match in pattern.finditer(text):
        if match.end() <= carry_length:
            continue
        count += 1
    return count


def replay_harness_source_row(doc_id, text):
    """True when a source row is a replay_*.py harness by doc_id or extractor header."""
    if REPLAY_HARNESS_DOC_ID.search(str(doc_id or "")):
        return True
    return bool(REPLAY_HARNESS_HEADER.search(str(text or "")))


def _unlink_quiet(*paths):
    for path in paths:
        try:
            os.unlink(path)
        except FileNotFoundError:
            continue


HISTORICAL_PACKED_SHA_PREFIX = "841df5ec"
HISTORICAL_SOURCE_SHA_PREFIX = "e549870b"
CORRECTED_SOURCE_SHA_PREFIX = "fff6dae2"
CORRECTED_PACKED_SHA_PREFIX = "503e18e8"
CORRECTED_MENTION_DOCS = 35
CORRECTED_MENTION_ROWS = 40


def _source_row_mentions_replay(doc_id, source_file, text):
    blob = f"{text}\n{doc_id}\n{source_file}".lower()
    return "replay" in blob


def measure_source_jsonl(path):
    sha = sha256_file(path)
    harness_docs = []
    mention_docs = []
    harness_rows = 0
    mention_rows = 0
    positive_rows = 0
    rows = 0
    seen_harness = set()
    seen_mention = set()
    with open(path) as handle:
        for line in handle:
            payload = json.loads(line)
            rows += 1
            text = payload.get("text") or ""
            meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
            doc_id = str(meta.get("doc_id") or "")
            source_file = str(meta.get("source_file") or "")
            if PACKED_CONTENT_POSITIVE_CONTROL in text:
                positive_rows += 1
            if replay_harness_source_row(doc_id, text):
                harness_rows += 1
                if doc_id not in seen_harness:
                    seen_harness.add(doc_id)
                    harness_docs.append(doc_id)
                continue
            if _source_row_mentions_replay(doc_id, source_file, text):
                mention_rows += 1
                if doc_id not in seen_mention:
                    seen_mention.add(doc_id)
                    mention_docs.append(doc_id)
    return {
        "path": os.path.abspath(path),
        "sha256": sha,
        "rows": rows,
        "positive_control_rows": positive_rows,
        "harness_rows": harness_rows,
        "harness_docs": len(harness_docs),
        "harness_doc_ids": harness_docs,
        "mention_rows": mention_rows,
        "mention_docs": len(mention_docs),
        "mention_doc_ids": mention_docs,
    }


def measure_packed_content(path, tokenizer):
    markers = (PACKED_CONTENT_POSITIVE_CONTROL, *PACKED_CONTENT_FORBIDDEN)
    token_markers = {
        marker: tokenizer(marker, add_special_tokens=False)["input_ids"]
        for marker in markers
    }
    max_marker_tokens = max(len(ids) for ids in token_markers.values())
    token_hits = {marker: 0 for marker in markers}
    decoded_hits = {marker: 0 for marker in markers}
    carry = []
    decoded_carry = ""
    header_hits = 0
    rows = 0
    with open(path) as handle:
        for line in handle:
            row = json.loads(line)
            ids = row.get("input_ids")
            if not isinstance(ids, list) or not ids or any(not isinstance(i, int) for i in ids):
                raise ValueError(f"packed row {rows} does not carry a non-empty integer input_ids list")
            window = carry + ids
            for marker, needle in token_markers.items():
                token_hits[marker] += _count_new_subsequence(window, needle, len(carry))
            decoded = tokenizer.decode(ids, skip_special_tokens=False)
            decoded_window = decoded_carry + decoded
            header_hits += _count_new_regex(
                decoded_window, REPLAY_HARNESS_HEADER, len(decoded_carry)
            )
            for marker in markers:
                decoded_hits[marker] += decoded.count(marker)
            carry = window[-(max_marker_tokens - 1):] if max_marker_tokens > 1 else []
            decoded_carry = decoded_window[-_HEADER_DECODE_CARRY:]
            rows += 1
    if rows == 0:
        raise ValueError("packed corpus has no rows")
    positive = PACKED_CONTENT_POSITIVE_CONTROL
    forbidden_hits = {
        marker: {"token_hits": token_hits[marker], "decoded_hits": decoded_hits[marker]}
        for marker in PACKED_CONTENT_FORBIDDEN
    }
    positive_fired = token_hits[positive] > 0 and decoded_hits[positive] > 0
    forbidden_fired = any(
        result["token_hits"] or result["decoded_hits"] for result in forbidden_hits.values()
    )
    refused = (not positive_fired) or forbidden_fired or header_hits > 0
    return {
        "path": os.path.abspath(path),
        "sha256": sha256_file(path),
        "rows_scanned": rows,
        "positive_control": {
            "marker": positive,
            "token_hits": token_hits[positive],
            "decoded_hits": decoded_hits[positive],
            "fired": positive_fired,
        },
        "forbidden_markers": forbidden_hits,
        "replay_harness_headers": {"decoded_hits": header_hits},
        "refused": refused,
    }


def verify_packed_content(path, tokenizer):
    receipt = measure_packed_content(path, tokenizer)
    positive = receipt["positive_control"]
    if not positive["fired"]:
        raise ValueError(
            f"packed-content positive control is invisible: {positive['marker']!r} "
            f"token_hits={positive['token_hits']} decoded_hits={positive['decoded_hits']}"
        )
    forbidden_hits = receipt["forbidden_markers"]
    if any(result["token_hits"] or result["decoded_hits"] for result in forbidden_hits.values()):
        raise ValueError(f"packed corpus contains forbidden replay-harness markers: {forbidden_hits}")
    header_hits = receipt["replay_harness_headers"]["decoded_hits"]
    if header_hits:
        raise ValueError(
            f"packed corpus contains replay-harness source headers: decoded_hits={header_hits}"
        )
    print(f"[pack] content gate VERIFIED: {json.dumps({
        'positive_control': receipt['positive_control'],
        'forbidden_markers': receipt['forbidden_markers'],
        'replay_harness_headers': receipt['replay_harness_headers'],
        'rows_scanned': receipt['rows_scanned'],
    }, sort_keys=True)}", flush=True)
    return {
        "positive_control": {
            "marker": positive["marker"],
            "token_hits": positive["token_hits"],
            "decoded_hits": positive["decoded_hits"],
        },
        "forbidden_markers": forbidden_hits,
        "replay_harness_headers": receipt["replay_harness_headers"],
        "rows_scanned": receipt["rows_scanned"],
    }


def _require_sha_prefix(label, sha, prefix):
    if not sha.startswith(prefix):
        raise SystemExit(
            f"ABORT: {label} sha256 {sha} does not start with required prefix {prefix}"
        )


def write_class_proof_receipt(
    path,
    *,
    historical_packed,
    historical_source,
    corrected_source,
    corrected_packed,
    tokenizer_path,
):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    hist_source = measure_source_jsonl(historical_source)
    corr_source = measure_source_jsonl(corrected_source)
    _require_sha_prefix("historical-source", hist_source["sha256"], HISTORICAL_SOURCE_SHA_PREFIX)
    _require_sha_prefix("corrected-source", corr_source["sha256"], CORRECTED_SOURCE_SHA_PREFIX)
    hist_packed = measure_packed_content(historical_packed, tok)
    corr_packed = measure_packed_content(corrected_packed, tok)
    _require_sha_prefix("historical-packed", hist_packed["sha256"], HISTORICAL_PACKED_SHA_PREFIX)
    _require_sha_prefix("corrected-packed", corr_packed["sha256"], CORRECTED_PACKED_SHA_PREFIX)

    errors = []
    if hist_source["harness_docs"] != 13:
        errors.append(
            f"historical source harness docs {hist_source['harness_docs']} != 13"
        )
    if not hist_packed["positive_control"]["fired"]:
        errors.append("historical packed positive control did not fire")
    if not hist_packed["refused"]:
        errors.append("historical packed gate passed; it must refuse")
    if corr_source["harness_docs"] != 0:
        errors.append(
            f"corrected source harness docs {corr_source['harness_docs']} != 0"
        )
    if corr_source["mention_docs"] != CORRECTED_MENTION_DOCS:
        errors.append(
            f"corrected mention docs {corr_source['mention_docs']} != {CORRECTED_MENTION_DOCS}"
        )
    if corr_source["mention_rows"] != CORRECTED_MENTION_ROWS:
        errors.append(
            f"corrected mention rows {corr_source['mention_rows']} != {CORRECTED_MENTION_ROWS}"
        )
    if not corr_packed["positive_control"]["fired"]:
        errors.append("corrected packed positive control did not fire")
    if corr_packed["refused"]:
        errors.append("corrected packed gate refused")
    if errors:
        raise SystemExit("ABORT: class proof against real artifacts failed: " + "; ".join(errors))

    receipt = {
        "schema": "palios.replay_harness_class_proof.v2",
        "scanner": {
            "doc_id": REPLAY_HARNESS_DOC_ID.pattern,
            "header": REPLAY_HARNESS_HEADER.pattern,
            "forbidden_literals": list(PACKED_CONTENT_FORBIDDEN),
            "positive_control": PACKED_CONTENT_POSITIVE_CONTROL,
        },
        "historical_source": hist_source,
        "historical_packed": hist_packed,
        "corrected_source": corr_source,
        "corrected_packed": corr_packed,
        "acceptance": {
            "historical_packed_prefix": HISTORICAL_PACKED_SHA_PREFIX,
            "historical_source_prefix": HISTORICAL_SOURCE_SHA_PREFIX,
            "corrected_source_prefix": CORRECTED_SOURCE_SHA_PREFIX,
            "corrected_packed_prefix": CORRECTED_PACKED_SHA_PREFIX,
            "historical_harness_docs": 13,
            "historical_packed_refused": True,
            "corrected_mention_docs": CORRECTED_MENTION_DOCS,
            "corrected_mention_rows": CORRECTED_MENTION_ROWS,
            "corrected_packed_refused": False,
        },
    }
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    stage = path + ".tmp"
    with open(stage, "w") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(stage, path)
    print(json.dumps({
        "historical_source_sha256": hist_source["sha256"],
        "historical_packed_sha256": hist_packed["sha256"],
        "corrected_source_sha256": corr_source["sha256"],
        "corrected_packed_sha256": corr_packed["sha256"],
        "historical_harness_docs": hist_source["harness_docs"],
        "historical_packed_refused": hist_packed["refused"],
        "historical_packed_positive_fired": hist_packed["positive_control"]["fired"],
        "corrected_mention_docs": corr_source["mention_docs"],
        "corrected_mention_rows": corr_source["mention_rows"],
        "corrected_packed_refused": corr_packed["refused"],
        "corrected_packed_positive_fired": corr_packed["positive_control"]["fired"],
    }, sort_keys=True))
    print(f"[pack] class proof VERIFIED against real artifacts: receipt={path}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--class-proof", metavar="RECEIPT",
                    help="write replay-harness class proof from real corpus artifacts and exit")
    ap.add_argument("--historical-packed", help="packed corpus whose sha256 starts 841df5ec")
    ap.add_argument("--historical-source", help="source jsonl whose sha256 starts e549870b")
    ap.add_argument("--corrected-source", help="source jsonl whose sha256 starts fff6dae2")
    ap.add_argument("--corrected-packed", help="packed corpus whose sha256 starts 503e18e8")
    ap.add_argument("--pack-set", default="production_v1", choices=sorted(PACK_SETS))
    ap.add_argument("--slices-dir")
    ap.add_argument("--tokenizer")
    ap.add_argument("--out")
    ap.add_argument("--manifest", help="default: <out>.manifest.json")
    args = ap.parse_args()
    if args.class_proof:
        needed = (
            "historical_packed",
            "historical_source",
            "corrected_source",
            "corrected_packed",
            "tokenizer",
        )
        missing = [name for name in needed if not getattr(args, name)]
        if missing:
            ap.error("--class-proof requires "
                     + ", ".join("--" + name.replace("_", "-") for name in missing))
        return write_class_proof_receipt(
            args.class_proof,
            historical_packed=args.historical_packed,
            historical_source=args.historical_source,
            corrected_source=args.corrected_source,
            corrected_packed=args.corrected_packed,
            tokenizer_path=args.tokenizer,
        )
    missing = [name for name in ("slices_dir", "tokenizer", "out") if not getattr(args, name)]
    if missing:
        ap.error("the following arguments are required unless --class-proof: "
                 + ", ".join("--" + name.replace("_", "-") for name in missing))

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    eos = tok.eos_token_id
    if eos is None:
        sys.exit("ABORT: tokenizer has no eos_token_id")

    stream_docs = 0
    buf = []
    blocks = 0
    head = []          # first SEQ tokens of the stream, kept for cycle-padding the tail
    input_receipts = []
    tmp = args.out + ".tmp"
    with open(tmp, "w") as out:
        for fname, want_rows, want_sha in PACK_SETS[args.pack_set]:
            path = os.path.join(args.slices_dir, fname)
            full_sha = sha256_file(path)
            got_sha = full_sha[:16]
            rows = sum(1 for _ in open(path))
            if got_sha != want_sha or rows != want_rows:
                sys.exit(f"ABORT: {fname} rows={rows}/{want_rows} sha16={got_sha}/{want_sha} — "
                         f"input does not match its REGISTERED identity; refusing to pack.")
            input_receipts.append({
                "name": fname,
                "rows": rows,
                "sha256": full_sha,
                "registered_sha256_prefix": want_sha,
            })
            print(f"[pack] {fname}: rows={rows} sha16={got_sha} VERIFIED", flush=True)
            for line in open(path):
                payload = json.loads(line)
                text = payload["text"]
                meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
                doc_id = str(meta.get("doc_id") or "")
                if replay_harness_source_row(doc_id, text):
                    sys.exit(f"ABORT: source row is a replay harness: {doc_id or '<missing doc_id>'}")
                ids = tok(text, add_special_tokens=False)["input_ids"]
                buf.extend(ids)
                buf.append(eos)
                stream_docs += 1
                if len(head) < SEQ:
                    head = (head + ids + [eos])[:SEQ]
                while len(buf) >= SEQ:
                    out.write(json.dumps({"input_ids": buf[:SEQ]}) + "\n")
                    buf = buf[SEQ:]
                    blocks += 1
        tail_kept = len(buf)
        if buf:  # NO-TRUNCATION: cycle-pad the tail with the stream head to a full block
            pad = head[: SEQ - len(buf)]
            out.write(json.dumps({"input_ids": buf + pad}) + "\n")
            blocks += 1
            buf = []
    dropped = 0
    total_tokens = blocks * SEQ
    done_line = (
        f"[pack] DONE: docs={stream_docs} blocks={blocks} seq={SEQ} "
        f"tokens={total_tokens} tail_dropped={dropped} "
        f"(final block = {tail_kept} corpus-tail tok + {SEQ - tail_kept} cycle-pad)"
        if tail_kept else
        f"[pack] DONE: docs={stream_docs} blocks={blocks} seq={SEQ} tokens={total_tokens} tail_dropped=0"
    )

    # Gates and sidecar staging run against tmp. Dest corpus+manifest are untouched
    # until both candidates exist; a sidecar write failure therefore cannot replace
    # a prior final pair.
    manifest_path = args.manifest or args.out + ".manifest.json"
    sidecar_candidate = manifest_path + ".candidate"
    promoted = False
    try:
        with open(tmp) as handle:
            tmp_rows = sum(1 for _ in handle)
        if tmp_rows != blocks:
            sys.exit(f"ABORT: packed tmp rows={tmp_rows} != emitted blocks={blocks}")

        # SHRINKAGE GATE. Every input's sha was verified above — but a corpus can be perfectly
        # sha-clean and still be missing whole inputs, because the check only validates what you
        # DECIDED to include. On 2026-07-28 two registered slices were dropped on a bad "they exist
        # nowhere" finding and the pack produced 2,511 blocks against the previous production
        # corpus's 3,686. Every per-input check passed. Nothing compared the total.
        prev = os.environ.get("PREV_CORPUS", "")
        if prev and os.path.exists(prev):
            prev_blocks = sum(1 for _ in open(prev, errors="replace"))
            if tmp_rows < prev_blocks * 0.95:
                print(f"[pack] *** SHRINKAGE: {tmp_rows} blocks vs previous corpus {prev_blocks} "
                      f"({tmp_rows/prev_blocks:.0%}). A smaller corpus needs an explicit reason. ***")
                print(f"[pack] set ALLOW_SHRINK=1 with a recorded justification to proceed.")
                if os.environ.get("ALLOW_SHRINK", "") != "1":
                    sys.exit("ABORT: corpus shrank against PREV_CORPUS and ALLOW_SHRINK is not set.")
            else:
                print(f"[pack] shrinkage gate OK: {tmp_rows} vs previous {prev_blocks} "
                      f"({tmp_rows/prev_blocks:.0%})")
        else:
            print("[pack] shrinkage gate SKIPPED (set PREV_CORPUS to the last production corpus)")
        try:
            content_gate = verify_packed_content(tmp, tok)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise SystemExit(f"ABORT: packed-content gate failed: {error}") from error
        corpus_sha = sha256_file(tmp)
        corpus_bytes = os.path.getsize(tmp)
        if os.path.abspath(manifest_path) == os.path.abspath(args.out):
            sys.exit("ABORT: corpus and manifest paths must differ")
        write_manifest(sidecar_candidate, {
            "schema": SCHEMA,
            "corpus_filename": os.path.basename(args.out),
            "corpus_sha256": corpus_sha,
            "corpus_bytes": corpus_bytes,
            "corpus_rows": tmp_rows,
            "sequence_length": SEQ,
            "source_documents": stream_docs,
            "tail_corpus_tokens": tail_kept,
            "cycle_pad_tokens": SEQ - tail_kept if tail_kept else 0,
            "content_gate": content_gate,
            "pack_set": args.pack_set,
            "packer_sha256": sha256_file(os.path.abspath(__file__)),
            "inputs": input_receipts,
        })
        os.replace(tmp, args.out)
        os.replace(sidecar_candidate, manifest_path)
        promoted = True
    finally:
        if not promoted:
            _unlink_quiet(tmp, sidecar_candidate, sidecar_candidate + ".tmp")

    print(done_line, flush=True)
    print(f"[pack] OUTPUT sha256={corpus_sha}")
    print(f"[pack] MANIFEST {manifest_path} sha256={sha256_file(manifest_path)}")
    print(f"[pack] register as: {args.pack_set}_packed_{SEQ} (inputs: "
          + ", ".join(f"{n}@{s}" for n, _, s in PACK_SETS[args.pack_set]) + ")")


if __name__ == "__main__":
    main()
