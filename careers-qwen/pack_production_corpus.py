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
import argparse, json, os, re, subprocess, sys
from pathlib import Path

from corpus_manifest import (
    GENERATION_SCHEMA,
    SCHEMA,
    pointer_member,
    resolve_generation,
    sha256_file,
    write_generation_pointer,
    write_manifest,
)

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
REPLAY_HARNESS_DOC_ID = re.compile(r"::(?:.+/)?replay_[^/]+\.py$")
# Source rows: header only at the start of the document (\A). Packed streams
# join documents with the tokenizer EOS string, not a newline.
REPLAY_HARNESS_HEADER_SOURCE = re.compile(
    r"\A# repo: \S+ file: (?:.+/)?replay_[^/\s]+\.py(?:\n|$)"
)
_HEADER_DECODE_CARRY = 128
_PACKED_EOS_DEFAULT = "<|im_end|>"


def packed_header_pattern(eos_token=None):
    eos = re.escape(eos_token or _PACKED_EOS_DEFAULT)
    return re.compile(
        rf"(?:\A|{eos})# repo: \S+ file: (?:.+/)?replay_[^/\s]+\.py(?:\n|$)"
    )


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
    """True when a source row is a replay_*.py harness by doc_id or start-of-row header."""
    if REPLAY_HARNESS_DOC_ID.search(str(doc_id or "")):
        return True
    return bool(REPLAY_HARNESS_HEADER_SOURCE.search(str(text or "")))


def _unlink_quiet(*paths):
    for path in paths:
        if not path:
            continue
        try:
            os.unlink(path)
        except FileNotFoundError:
            continue


_HEX = frozenset("0123456789abcdef")


def bind_logical_paths(logical_out, logical_manifest):
    """Logical dest paths are never replaced. --manifest is same-dir as --out or defaulted."""
    logical_out = os.path.abspath(logical_out)
    if logical_manifest:
        logical_manifest = os.path.abspath(logical_manifest)
    else:
        logical_manifest = logical_out + ".manifest.json"
    if os.path.dirname(logical_manifest) != os.path.dirname(logical_out):
        raise SystemExit("ABORT: --manifest must be in the same directory as --out")
    return logical_out, logical_manifest, logical_out + ".generation"


def _full_sha256(value, label):
    if not (isinstance(value, str) and len(value) == 64 and set(value) <= _HEX):
        raise SystemExit(f"ABORT: {label} requires a full 64-hex sha256")
    return value


def generation_artifact_paths(logical_out, logical_manifest, corpus_sha, manifest_sha):
    """Gen filenames are sha-stamped logical paths. --manifest names the sidecar."""
    corpus_sha = _full_sha256(corpus_sha, "generation corpus identity")
    manifest_sha = _full_sha256(manifest_sha, "generation manifest identity")
    logical_out = os.path.abspath(logical_out)
    logical_manifest = os.path.abspath(logical_manifest)
    if os.path.dirname(logical_manifest) != os.path.dirname(logical_out):
        raise SystemExit("ABORT: --manifest must be in the same directory as --out")
    return (
        logical_out + ".gen." + corpus_sha,
        logical_manifest + ".gen." + manifest_sha,
    )


def install_if_absent(src, dest):
    """Move src to dest only when dest is new. Never overwrite a live generation file."""
    src = os.path.abspath(src)
    dest = os.path.abspath(dest)
    if src == dest:
        return False
    if os.path.exists(dest):
        if sha256_file(dest) != sha256_file(src):
            raise SystemExit(
                f"ABORT: generation path {dest} already exists with different content"
            )
        os.unlink(src)
        return False
    os.replace(src, dest)
    return True


def publish_generation_artifacts(
    tmp,
    sidecar_candidate,
    logical_out,
    logical_manifest,
    pointer_path,
    *,
    write_pointer,
):
    corpus_sha = sha256_file(tmp)
    manifest_sha = sha256_file(sidecar_candidate)
    corpus_gen, manifest_gen = generation_artifact_paths(
        logical_out, logical_manifest, corpus_sha, manifest_sha
    )
    if os.path.abspath(corpus_gen) == os.path.abspath(manifest_gen):
        raise SystemExit("ABORT: corpus and manifest paths must differ")
    install_if_absent(tmp, corpus_gen)
    install_if_absent(sidecar_candidate, manifest_gen)
    if write_pointer:
        write_generation_pointer(pointer_path, corpus_gen, manifest_gen)
    return corpus_gen, manifest_gen


def unpromoted_staging_paths(tmp, sidecar_candidate, pointer_path):
    """Failure cleanup is staging only. Never the live gen-stamped pair."""
    return (
        tmp,
        sidecar_candidate,
        sidecar_candidate + ".tmp",
        pointer_path + ".tmp",
    )


HISTORICAL_PACKED_SHA = "841df5ec10461d34e6b994b2f858cc3ef943092ed6904aefed16b03427815ddf"
HISTORICAL_SOURCE_SHA = "e549870b892d2f72565981a44d2ef881715cc47fcd84c7c76cdc8ee9a816bc78"
CORRECTED_SOURCE_SHA = "fff6dae26ad02e51614d03e41a1c426932eb55e0716c4771ff0479613e89e685"
CORRECTED_PACKED_SHA = "503e18e8cd67c9bc88cd16bc266381adf13e2666a27c37ef074e1d1d3e2aefba"
CORRECTED_PACKED_NAME = "cpt_prod_src-fff6dae26ad02e51_packed_8192.jsonl"
HISTORICAL_MANIFEST_SHA = "8f2e5da44b461b811f8bc08950808db86326c461d8bdf8865a97ac4b629c8eee"
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
    packed_header = packed_header_pattern(getattr(tokenizer, "eos_token", None))
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
                decoded_window,
                packed_header,
                len(decoded_carry),
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


def _require_sha(label, sha, expected):
    if sha != expected:
        raise SystemExit(
            f"ABORT: {label} sha256 {sha} does not equal required {expected}"
        )


def launcher_corrected_pins(launcher_text):
    """Packed and sidecar pins for the corrected artifact live in the launcher."""
    pattern = re.compile(
        r"\*cpt_prod_src-fff6dae26ad02e51_packed_8192\.jsonl\)\s*(.*?);;",
        re.S,
    )
    arm = None
    for match in pattern.finditer(launcher_text):
        if "EXPECT_CORPUS_SHA=" in match.group(1):
            arm = match.group(1)
            break
    if arm is None:
        raise SystemExit("ABORT: launcher has no content-pin arm for the corrected packed filename")
    packed = re.search(r"EXPECT_CORPUS_SHA=([0-9a-f]{64})", arm)
    sidecar = re.search(r"EXPECT_MANIFEST_SHA=([0-9a-f]{64})", arm)
    if not packed or packed.group(1) != CORRECTED_PACKED_SHA:
        raise SystemExit("ABORT: launcher packed pin is not the corrected corpus sha256")
    if not sidecar:
        raise SystemExit("ABORT: launcher sidecar pin missing for the corrected packed filename")
    return packed.group(1), sidecar.group(1)


def prove_selector_legs():
    harness_id = "repo::apply-machine::harness/replay_nested_contract.py"
    harness_text = (
        "# repo: apply-machine file: harness/replay_nested_contract.py\n\n"
        "Offline replay for a nested path.\n"
    )
    mention_id = "repo::apply-machine::harness/notes.md"
    mention_text = (
        "# repo: apply-machine file: harness/notes.md\n\n"
        "mentions replay machinery without being a replay_*.py file.\n"
    )
    quote_id = "repo::apply-machine::docs/quoting.md"
    quote_text = (
        "# repo: apply-machine file: docs/quoting.md\n\n"
        "Extractor header example:\n"
        "# repo: apply-machine file: replay_write_dispatch.py\n"
        "That quote is documentation, not a harness.\n"
    )
    if not replay_harness_source_row(harness_id, harness_text):
        raise SystemExit("ABORT: nested replay harness was not refused")
    if replay_harness_source_row(mention_id, mention_text):
        raise SystemExit("ABORT: nested non-harness mention was refused")
    if replay_harness_source_row(quote_id, quote_text):
        raise SystemExit("ABORT: prose-quoted header false-refused as a source row")
    packed_re = packed_header_pattern(_PACKED_EOS_DEFAULT)
    if packed_re.search(quote_text):
        raise SystemExit("ABORT: packed predicate matched a mid-document quoted header")
    if not packed_re.search("<|im_end|>" + harness_text):
        raise SystemExit("ABORT: packed predicate missed an EOS-bounded nested header")
    return {
        "nested_harness_refused": True,
        "nested_harness_doc_id": harness_id,
        "nested_mention_survived": True,
        "nested_mention_doc_id": mention_id,
        "quoted_header_survived": True,
        "quoted_header_doc_id": quote_id,
        "packed_eos_header_matched": True,
    }


def _toy_manifest(path, corpus_path, corpus_filename=None):
    write_manifest(path, {
        "schema": SCHEMA,
        "corpus_filename": corpus_filename or os.path.basename(corpus_path),
        "corpus_sha256": sha256_file(corpus_path),
        "corpus_bytes": os.path.getsize(corpus_path),
        "corpus_rows": 1,
        "inputs": [{
            "name": "x.jsonl",
            "rows": 1,
            "sha256": "a" * 64,
            "registered_sha256_prefix": "a" * 16,
        }],
    })


def _forbid_tmp_work_dir(work_dir):
    if not work_dir:
        raise SystemExit("ABORT: --proof-work-dir is required")
    abs_work = os.path.abspath(work_dir)
    if abs_work == "/tmp" or abs_work.startswith("/tmp/") or abs_work.startswith("/var/tmp/"):
        raise SystemExit("ABORT: --proof-work-dir must be a durable directory, not /tmp")
    os.makedirs(abs_work, exist_ok=True)
    return abs_work


def prove_generation_pointer(work_dir):
    td = os.path.join(work_dir, "generation-pointer")
    os.makedirs(td, exist_ok=True)
    logical = os.path.join(td, "logical.jsonl")
    with open(logical, "w") as handle:
        handle.write("stale-logical\n")
    with open(logical + ".manifest.json", "w") as handle:
        handle.write("{}\n")
    gen = os.path.join(td, "logical.jsonl.gen.deadbeef")
    with open(gen, "w") as handle:
        handle.write('{"input_ids":[1,2,3]}\n')
    man = gen + ".manifest.json"
    _toy_manifest(man, gen)

    corpus, _manifest = resolve_generation(logical)
    if os.path.abspath(corpus) != os.path.abspath(logical):
        raise SystemExit("ABORT: pre-pointer resolve published unpublished gen files")
    pre_pointer = True

    incomplete = os.path.join(td, "incomplete.jsonl")
    with open(incomplete, "w") as handle:
        handle.write("stale-incomplete\n")
    with open(incomplete + ".manifest.json", "w") as handle:
        handle.write("{}\n")
    orphan = os.path.join(td, "incomplete.jsonl.gen.orphan")
    with open(orphan, "w") as handle:
        handle.write('{"input_ids":[4,5,6]}\n')
    corpus, _manifest = resolve_generation(incomplete)
    if os.path.abspath(corpus) != os.path.abspath(incomplete):
        raise SystemExit("ABORT: incomplete gen corpus without pointer was resolved")
    if os.path.exists(orphan + ".manifest.json"):
        raise SystemExit("ABORT: incomplete pair unexpectedly has a manifest")
    pre_pointer_incomplete = True

    write_generation_pointer(logical + ".generation", gen, man)
    corpus, manifest = resolve_generation(logical)
    if os.path.abspath(corpus) != os.path.abspath(gen):
        raise SystemExit("ABORT: generation pointer did not resolve to the gen corpus")
    if os.path.abspath(manifest) != os.path.abspath(man):
        raise SystemExit("ABORT: generation pointer did not resolve to the gen manifest")

    with open(gen, "w") as handle:
        handle.write('{"input_ids":[1,2,3]}\n')
    _toy_manifest(man, gen)
    write_generation_pointer(logical + ".generation", gen, man)
    corpus, manifest = resolve_generation(logical)
    if os.path.abspath(corpus) != os.path.abspath(gen):
        raise SystemExit("ABORT: same-generation rerun lost the pointer")
    same_generation_rerun = True

    with open(gen, "w") as handle:
        handle.write('{"input_ids":[9]}\n')
    try:
        resolve_generation(logical)
        raise SystemExit("ABORT: mixed generation pair was accepted")
    except ValueError:
        mixed_pair_refused = True

    trav_logical = os.path.join(td, "trav.jsonl")
    with open(trav_logical, "w") as handle:
        handle.write("stale-trav\n")
    trav_pointer = trav_logical + ".generation"
    with open(trav_pointer, "w") as handle:
        json.dump({
            "schema": GENERATION_SCHEMA,
            "corpus": "../escape.jsonl",
            "manifest": "trav.jsonl.manifest.json",
            "corpus_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
        }, handle)
    try:
        resolve_generation(trav_logical)
        raise SystemExit("ABORT: pointer path traversal was accepted")
    except ValueError:
        path_traversal_refused = True
    try:
        pointer_member(td, "../escape.jsonl", "corpus")
        raise SystemExit("ABORT: pointer_member accepted a parent path")
    except ValueError:
        pass

    return {
        "pointer_resolves": True,
        "pre_pointer_does_not_publish_gen": pre_pointer,
        "pre_pointer_incomplete_pair_not_resolved": pre_pointer_incomplete,
        "same_generation_rerun": same_generation_rerun,
        "mixed_pair_refused": mixed_pair_refused,
        "path_traversal_refused": path_traversal_refused,
    }


def _run_pin_script(script, cpt_data, cwd):
    return subprocess.run(
        ["bash", script, cpt_data],
        capture_output=True,
        text=True,
        cwd=cwd,
    )


def _write_pin_script(dest, block):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w") as handle:
        handle.write("#!/bin/bash\nset -euo pipefail\nCPT_DATA=\"$1\"\n")
        handle.write(block)
        handle.write("\n")
    os.chmod(dest, 0o755)


def prove_launcher_fail_closed(work_dir, launcher_path):
    text = Path(launcher_path).read_text(encoding="utf-8")
    start = text.find("# BEGIN_CORPUS_CONTENT_PIN")
    end = text.find("# END_CORPUS_CONTENT_PIN")
    if start < 0 or end < 0 or end <= start:
        raise SystemExit("ABORT: launcher is missing CORPUS_CONTENT_PIN markers")
    block = text[start:end + len("# END_CORPUS_CONTENT_PIN")]
    tree = os.path.join(work_dir, "launcher-tree")
    careers = os.path.join(tree, "careers-qwen")
    recipes = os.path.join(tree, "dense-9b", "recipes")
    os.makedirs(careers, exist_ok=True)
    os.makedirs(recipes, exist_ok=True)
    manifest_src = Path(__file__).resolve().parent / "corpus_manifest.py"
    Path(careers, "corpus_manifest.py").write_bytes(manifest_src.read_bytes())
    script = os.path.join(recipes, "launch_cpt_qwen36_27b_fsdp.sh")
    _write_pin_script(script, block)
    if os.path.exists(os.path.join(tree, ".git")):
        raise SystemExit("ABORT: launcher proof tree must not be a git checkout")

    # Unknown basename hits the pin-case default, not an empty EXPECT skip.
    unknown = _run_pin_script(
        script, os.path.join(work_dir, "unknown_packed_8192.jsonl"), tree
    )
    if unknown.returncode == 0:
        raise SystemExit("ABORT: launcher accepted an unknown packed filename")
    if "no content pin" not in unknown.stderr:
        raise SystemExit(
            "ABORT: unknown filename did not fail-closed on the pin case: "
            + unknown.stderr.strip()
        )

    packed_pin, sidecar_pin = launcher_corrected_pins(block)
    corrected_name = CORRECTED_PACKED_NAME
    if packed_pin != CORRECTED_PACKED_SHA:
        raise SystemExit("ABORT: launcher packed pin is not the corrected corpus sha256")
    corrected_missing = os.path.join(work_dir, "missing-corrected", corrected_name)
    corrected_run = _run_pin_script(script, corrected_missing, tree)
    if corrected_run.returncode == 0:
        raise SystemExit("ABORT: launcher accepted the corrected packed name with no file")
    if "no content pin" in corrected_run.stderr:
        raise SystemExit("ABORT: corrected packed filename was not admitted by the pin case")
    if "CPT_DATA is missing" not in corrected_run.stderr:
        raise SystemExit(
            "ABORT: corrected packed name did not hit the missing-file gate: "
            + corrected_run.stderr.strip()
        )

    # Name that IS in the pin case, file absent → missing-file gate, not unknown.
    pinned_name = "cpt_prod_v3_packed_8192.jsonl"
    missing = os.path.join(work_dir, "missing", pinned_name)
    missing_run = _run_pin_script(script, missing, tree)
    if missing_run.returncode == 0:
        raise SystemExit("ABORT: launcher accepted a pinned name whose file is missing")
    if "CPT_DATA is missing" not in missing_run.stderr:
        raise SystemExit(
            "ABORT: missing pinned file did not hit the missing-file gate: "
            + missing_run.stderr.strip()
        )

    # Process death before .generation: unpublished gen must not be selected.
    pre_dir = os.path.join(work_dir, "pre-pointer")
    os.makedirs(pre_dir, exist_ok=True)
    logical = os.path.join(pre_dir, pinned_name)
    with open(logical, "w") as handle:
        handle.write("stale-logical-pre-pointer\n")
    unpublished = logical + ".gen.unpublished"
    with open(unpublished, "w") as handle:
        handle.write('{"input_ids":[1,2,3]}\n')
    _toy_manifest(unpublished + ".manifest.json", unpublished)
    pre = _run_pin_script(script, logical, tree)
    if pre.returncode == 0:
        raise SystemExit("ABORT: launcher accepted stale logical with unpublished gen")
    logical_sha = sha256_file(logical)
    unpublished_sha = sha256_file(unpublished)
    if logical_sha not in pre.stderr:
        raise SystemExit(
            "ABORT: pre-pointer launcher did not pin-check the logical file: "
            + pre.stderr.strip()
        )
    if unpublished_sha in pre.stderr:
        raise SystemExit("ABORT: pre-pointer launcher pin-checked unpublished gen")
    if "content pin MISMATCH" not in pre.stderr:
        raise SystemExit(
            "ABORT: pre-pointer launcher did not mismatch the stale logical: "
            + pre.stderr.strip()
        )

    # Same-generation rerun: pointer rewritten to the same gen, then pin-check gen.
    sg_dir = os.path.join(work_dir, "same-generation")
    os.makedirs(sg_dir, exist_ok=True)
    sg_logical = os.path.join(sg_dir, pinned_name)
    with open(sg_logical, "w") as handle:
        handle.write("stale-same-generation\n")
    sg_gen = sg_logical + ".gen.deadbeef"
    with open(sg_gen, "w") as handle:
        handle.write('{"input_ids":[1,2,3]}\n')
    sg_man = sg_gen + ".manifest.json"
    _toy_manifest(sg_man, sg_gen)
    write_generation_pointer(sg_logical + ".generation", sg_gen, sg_man)
    write_generation_pointer(sg_logical + ".generation", sg_gen, sg_man)
    sg = _run_pin_script(script, sg_logical, tree)
    if sg.returncode == 0:
        raise SystemExit("ABORT: launcher accepted same-generation toy gen as pinned v3")
    sg_gen_sha = sha256_file(sg_gen)
    sg_logical_sha = sha256_file(sg_logical)
    if sg_gen_sha not in sg.stderr:
        raise SystemExit(
            "ABORT: same-generation rerun did not pin-check the resolved gen: "
            + sg.stderr.strip()
        )
    if sg_logical_sha != sg_gen_sha and sg_logical_sha in sg.stderr:
        raise SystemExit("ABORT: same-generation rerun still pin-checked the logical file")
    if "content pin MISMATCH" not in sg.stderr:
        raise SystemExit(
            "ABORT: same-generation rerun did not mismatch the resolved gen: "
            + sg.stderr.strip()
        )

    # Mixed pair after publish: mutated gen must fail resolve, not train.
    with open(sg_gen, "w") as handle:
        handle.write('{"input_ids":[9]}\n')
    mixed = _run_pin_script(script, sg_logical, tree)
    if mixed.returncode == 0:
        raise SystemExit("ABORT: launcher accepted a mixed generation pair")
    if "REFUSE" not in mixed.stderr:
        raise SystemExit(
            "ABORT: mixed generation pair did not fail-closed at resolve: "
            + mixed.stderr.strip()
        )

    # Pointer exists, resolver missing: fail-closed, do not skip to stale logical.
    blind = os.path.join(work_dir, "launcher-tree-no-resolver")
    blind_recipes = os.path.join(blind, "dense-9b", "recipes")
    os.makedirs(blind_recipes, exist_ok=True)
    blind_script = os.path.join(blind_recipes, "launch_cpt_qwen36_27b_fsdp.sh")
    _write_pin_script(blind_script, block)
    blind_dir = os.path.join(work_dir, "missing-resolver")
    os.makedirs(blind_dir, exist_ok=True)
    blind_logical = os.path.join(blind_dir, pinned_name)
    with open(blind_logical, "w") as handle:
        handle.write("stale-missing-resolver\n")
    blind_gen = blind_logical + ".gen.deadbeef"
    with open(blind_gen, "w") as handle:
        handle.write('{"input_ids":[1,2,3]}\n')
    blind_man = blind_gen + ".manifest.json"
    _toy_manifest(blind_man, blind_gen)
    write_generation_pointer(blind_logical + ".generation", blind_gen, blind_man)
    missing_resolver = _run_pin_script(blind_script, blind_logical, blind)
    if missing_resolver.returncode == 0:
        raise SystemExit("ABORT: launcher skipped a generation pointer when resolver was missing")
    if "corpus_manifest.py is missing" not in missing_resolver.stderr:
        raise SystemExit(
            "ABORT: missing resolver did not fail-closed on the pointer: "
            + missing_resolver.stderr.strip()
        )

    return {
        "unknown_filename_refused": True,
        "missing_pinned_file_refused": True,
        "pre_pointer_unpublished_gen_not_selected": True,
        "same_generation_rerun_fail_closed": True,
        "mixed_pair_fail_closed": True,
        "outside_git_pointer_resolved": True,
        "missing_resolver_fail_closed": True,
        "corrected_packed_name_admitted": True,
        "corrected_sidecar_pin_present": True,
        "corrected_sidecar_pin_sha256": sidecar_pin,
    }


def prove_publication_invariants(work_dir):
    sha = "ab" * 32
    man_sha = "cd" * 32
    short = sha[:16]
    out = os.path.join(work_dir, "publish", "logical.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    default_manifest = out + ".manifest.json"
    corpus_gen, manifest_gen = generation_artifact_paths(out, default_manifest, sha, man_sha)
    if not corpus_gen.endswith(".gen." + sha):
        raise SystemExit("ABORT: generation artifact path did not use the full corpus sha256")
    if not manifest_gen.endswith(".gen." + man_sha):
        raise SystemExit("ABORT: generation sidecar path did not use the full manifest sha256")
    try:
        generation_artifact_paths(out, default_manifest, short, man_sha)
        raise SystemExit("ABORT: 16-hex generation identity was accepted")
    except SystemExit as error:
        if "full 64-hex" not in str(error):
            raise

    logical_out, logical_manifest, pointer_path = bind_logical_paths(out, None)
    if logical_manifest != logical_out + ".manifest.json":
        raise SystemExit("ABORT: default --manifest path was not bound")
    custom = os.path.join(os.path.dirname(out), "custom.manifest.json")
    _, bound_custom, _ = bind_logical_paths(out, custom)
    if bound_custom != os.path.abspath(custom):
        raise SystemExit("ABORT: --manifest same-dir path was ignored")
    custom_gen = generation_artifact_paths(out, bound_custom, sha, man_sha)[1]
    if os.path.basename(custom_gen) != os.path.basename(custom) + ".gen." + man_sha:
        raise SystemExit("ABORT: --manifest did not name the generation sidecar")
    other = os.path.join(work_dir, "other-dir", "custom.manifest.json")
    os.makedirs(os.path.dirname(other), exist_ok=True)
    try:
        bind_logical_paths(out, other)
        raise SystemExit("ABORT: --manifest in another directory was accepted")
    except SystemExit as error:
        if "same directory" not in str(error):
            raise

    live = os.path.join(work_dir, "publish", "logical.jsonl.gen." + sha)
    with open(live, "w") as handle:
        handle.write("LIVE-GENERATION\n")
    tmp = out + ".tmp"
    sidecar = tmp + ".manifest.json"
    with open(tmp, "w") as handle:
        handle.write("staging\n")
    with open(sidecar, "w") as handle:
        handle.write("{}\n")
    _unlink_quiet(*unpromoted_staging_paths(tmp, sidecar, pointer_path))
    if not os.path.exists(live):
        raise SystemExit("ABORT: unpromoted cleanup deleted the live same-SHA generation")
    with open(live) as handle:
        if handle.read() != "LIVE-GENERATION\n":
            raise SystemExit("ABORT: unpromoted cleanup mutated the live generation")
    if os.path.exists(tmp) or os.path.exists(sidecar):
        raise SystemExit("ABORT: unpromoted cleanup left staging files")

    death_dir = os.path.join(work_dir, "death-before-pointer")
    os.makedirs(death_dir, exist_ok=True)
    death_out = os.path.join(death_dir, "logical.jsonl")
    death_manifest = death_out + ".manifest.json"
    death_pointer = death_out + ".generation"
    tmp_a = death_out + ".tmp"
    sidecar_a = tmp_a + ".manifest.json"
    with open(tmp_a, "w") as handle:
        handle.write('{"input_ids":[1,2,3]}\n')
    death_corpus_sha = sha256_file(tmp_a)
    death_corpus_name = os.path.basename(death_out) + ".gen." + death_corpus_sha
    _toy_manifest(sidecar_a, tmp_a, corpus_filename=death_corpus_name)
    first_corpus, first_manifest = publish_generation_artifacts(
        tmp_a, sidecar_a, death_out, death_manifest, death_pointer, write_pointer=True
    )
    resolved_corpus, resolved_manifest = resolve_generation(death_out)
    if os.path.abspath(resolved_manifest) != os.path.abspath(first_manifest):
        raise SystemExit("ABORT: first publish pointer did not name the first sidecar")
    old_sidecar = Path(first_manifest).read_bytes()
    tmp_b = death_out + ".tmp"
    sidecar_b = tmp_b + ".manifest.json"
    with open(tmp_b, "w") as handle:
        handle.write('{"input_ids":[1,2,3]}\n')
    _toy_manifest(sidecar_b, tmp_b, corpus_filename=death_corpus_name)
    with open(sidecar_b, "a") as handle:
        handle.write(" ")
    second_corpus, second_manifest = publish_generation_artifacts(
        tmp_b, sidecar_b, death_out, death_manifest, death_pointer, write_pointer=False
    )
    if os.path.abspath(second_corpus) != os.path.abspath(first_corpus):
        raise SystemExit("ABORT: same-SHA corpus was not reused")
    if os.path.abspath(second_manifest) == os.path.abspath(first_manifest):
        raise SystemExit("ABORT: changed sidecar reused the live sidecar path")
    if Path(first_manifest).read_bytes() != old_sidecar:
        raise SystemExit("ABORT: death before pointer overwrote the live sidecar")
    still_corpus, still_manifest = resolve_generation(death_out)
    if os.path.abspath(still_manifest) != os.path.abspath(first_manifest):
        raise SystemExit("ABORT: death before pointer invalidated the live pointer")
    write_generation_pointer(death_pointer, second_corpus, second_manifest)
    done_corpus, done_manifest = resolve_generation(death_out)
    if os.path.abspath(done_manifest) != os.path.abspath(second_manifest):
        raise SystemExit("ABORT: completed pointer did not name the new sidecar")

    return {
        "full_sha_generation_identity": True,
        "prefix_identity_refused": True,
        "manifest_flag_bound": True,
        "manifest_names_generation_sidecar": True,
        "manifest_other_dir_refused": True,
        "live_generation_survives_unpromoted_cleanup": True,
        "death_before_pointer_preserves_live_pair": True,
    }


def write_class_proof_receipt(
    path,
    *,
    historical_packed,
    historical_source,
    historical_manifest,
    corrected_source,
    corrected_packed,
    corrected_manifest,
    tokenizer_path,
    proof_work_dir,
    launcher_path,
):
    from transformers import AutoTokenizer

    work_dir = _forbid_tmp_work_dir(proof_work_dir)
    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    nested = prove_selector_legs()
    generation = prove_generation_pointer(work_dir)
    publication = prove_publication_invariants(work_dir)
    launcher = prove_launcher_fail_closed(work_dir, launcher_path)
    hist_source = measure_source_jsonl(historical_source)
    corr_source = measure_source_jsonl(corrected_source)
    _require_sha("historical-source", hist_source["sha256"], HISTORICAL_SOURCE_SHA)
    _require_sha("corrected-source", corr_source["sha256"], CORRECTED_SOURCE_SHA)
    hist_packed = measure_packed_content(historical_packed, tok)
    corr_packed = measure_packed_content(corrected_packed, tok)
    _require_sha("historical-packed", hist_packed["sha256"], HISTORICAL_PACKED_SHA)
    _require_sha("corrected-packed", corr_packed["sha256"], CORRECTED_PACKED_SHA)
    _require_sha("historical-manifest", sha256_file(historical_manifest), HISTORICAL_MANIFEST_SHA)
    _packed_pin, expected_manifest = launcher_corrected_pins(
        Path(launcher_path).read_text(encoding="utf-8")
    )
    _require_sha("launcher-packed-pin", _packed_pin, CORRECTED_PACKED_SHA)
    _require_sha("corrected-manifest", sha256_file(corrected_manifest), expected_manifest)

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
        "schema": "palios.replay_harness_class_proof.v4",
        "scanner": {
            "doc_id": REPLAY_HARNESS_DOC_ID.pattern,
            "header_source": REPLAY_HARNESS_HEADER_SOURCE.pattern,
            "header_packed": packed_header_pattern(_PACKED_EOS_DEFAULT).pattern,
            "forbidden_literals": list(PACKED_CONTENT_FORBIDDEN),
            "positive_control": PACKED_CONTENT_POSITIVE_CONTROL,
        },
        "nested_selector": nested,
        "generation_pointer": generation,
        "publication": publication,
        "launcher": launcher,
        "historical_source": hist_source,
        "historical_packed": hist_packed,
        "historical_manifest_sha256": HISTORICAL_MANIFEST_SHA,
        "corrected_source": corr_source,
        "corrected_packed": corr_packed,
        "corrected_manifest_sha256": expected_manifest,
        "acceptance": {
            "historical_packed_sha256": HISTORICAL_PACKED_SHA,
            "historical_source_sha256": HISTORICAL_SOURCE_SHA,
            "historical_manifest_sha256": HISTORICAL_MANIFEST_SHA,
            "corrected_source_sha256": CORRECTED_SOURCE_SHA,
            "corrected_packed_sha256": CORRECTED_PACKED_SHA,
            "corrected_manifest_sha256": expected_manifest,
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
        "nested_harness_refused": nested["nested_harness_refused"],
        "quoted_header_survived": nested["quoted_header_survived"],
        "mixed_pair_refused": generation["mixed_pair_refused"],
        "pre_pointer_does_not_publish_gen": generation["pre_pointer_does_not_publish_gen"],
        "same_generation_rerun": generation["same_generation_rerun"],
        "unknown_filename_refused": launcher["unknown_filename_refused"],
        "missing_pinned_file_refused": launcher["missing_pinned_file_refused"],
        "pre_pointer_unpublished_gen_not_selected": launcher["pre_pointer_unpublished_gen_not_selected"],
        "same_generation_rerun_fail_closed": launcher["same_generation_rerun_fail_closed"],
        "mixed_pair_fail_closed": launcher["mixed_pair_fail_closed"],
        "path_traversal_refused": generation["path_traversal_refused"],
        "full_sha_generation_identity": publication["full_sha_generation_identity"],
        "manifest_flag_bound": publication["manifest_flag_bound"],
        "manifest_names_generation_sidecar": publication["manifest_names_generation_sidecar"],
        "live_generation_survives_unpromoted_cleanup": publication["live_generation_survives_unpromoted_cleanup"],
        "death_before_pointer_preserves_live_pair": publication["death_before_pointer_preserves_live_pair"],
        "outside_git_pointer_resolved": launcher["outside_git_pointer_resolved"],
        "missing_resolver_fail_closed": launcher["missing_resolver_fail_closed"],
        "corrected_packed_name_admitted": launcher["corrected_packed_name_admitted"],
        "corrected_sidecar_pin_present": launcher["corrected_sidecar_pin_present"],
    }, sort_keys=True))
    print(f"[pack] class proof VERIFIED against real artifacts: receipt={path}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--class-proof", metavar="RECEIPT",
                    help="write replay-harness class proof from real corpus artifacts and exit")
    ap.add_argument("--proof-work-dir", help="durable directory for generation/launcher proofs; not /tmp")
    ap.add_argument("--historical-packed", help="packed corpus sha256 841df5ec…")
    ap.add_argument("--historical-source", help="source jsonl sha256 e549870b…")
    ap.add_argument("--historical-manifest", help="packed sidecar sha256 8f2e5da4…")
    ap.add_argument("--corrected-source", help="source jsonl sha256 fff6dae2…")
    ap.add_argument("--corrected-packed", help="packed corpus sha256 503e18e8…")
    ap.add_argument("--corrected-manifest", help="sidecar whose sha256 equals the launcher sidecar pin")
    ap.add_argument("--launcher", help="path to launch_cpt_qwen36_27b_fsdp.sh")
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
            "historical_manifest",
            "corrected_source",
            "corrected_packed",
            "corrected_manifest",
            "tokenizer",
            "proof_work_dir",
            "launcher",
        )
        missing = [name for name in needed if not getattr(args, name)]
        if missing:
            ap.error("--class-proof requires "
                     + ", ".join("--" + name.replace("_", "-") for name in missing))
        return write_class_proof_receipt(
            args.class_proof,
            historical_packed=args.historical_packed,
            historical_source=args.historical_source,
            historical_manifest=args.historical_manifest,
            corrected_source=args.corrected_source,
            corrected_packed=args.corrected_packed,
            corrected_manifest=args.corrected_manifest,
            tokenizer_path=args.tokenizer,
            proof_work_dir=args.proof_work_dir,
            launcher_path=args.launcher,
        )
    missing = [name for name in ("slices_dir", "tokenizer", "out") if not getattr(args, name)]
    if missing:
        ap.error("the following arguments are required unless --class-proof: "
                 + ", ".join("--" + name.replace("_", "-") for name in missing))
    args.out, logical_manifest, pointer_path = bind_logical_paths(args.out, args.manifest)

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

    # Gates run against tmp. Visibility is one atomic generation pointer; dest
    # logical corpus/manifest paths are never replaced, so process death cannot
    # publish a mixed pair.
    sidecar_candidate = tmp + ".manifest.json"
    corpus_gen = None
    manifest_gen = None
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
        write_manifest(sidecar_candidate, {
            "schema": SCHEMA,
            "corpus_filename": os.path.basename(args.out) + ".gen." + corpus_sha,
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
        corpus_gen, manifest_gen = publish_generation_artifacts(
            tmp,
            sidecar_candidate,
            args.out,
            logical_manifest,
            pointer_path,
            write_pointer=True,
        )
        promoted = True
    finally:
        if not promoted:
            _unlink_quiet(*unpromoted_staging_paths(tmp, sidecar_candidate, pointer_path))

    print(done_line, flush=True)
    print(f"[pack] OUTPUT sha256={corpus_sha}")
    print(f"[pack] MANIFEST {manifest_gen} sha256={sha256_file(manifest_gen)}")
    print(f"[pack] LOGICAL_MANIFEST {logical_manifest} (never replaced)")
    print(f"[pack] GENERATION {pointer_path}")
    print(f"[pack] register as: {args.pack_set}_packed_{SEQ} (inputs: "
          + ", ".join(f"{n}@{s}" for n, _, s in PACK_SETS[args.pack_set]) + ")")


if __name__ == "__main__":
    main()
