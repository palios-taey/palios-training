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

Output: cpt_production_v1_packed_2560.jsonl + stdout sha256 for the treasurer registry row.
Run on a Spark node (needs the Qwen3.6-27B tokenizer):
  python3 pack_production_corpus.py --slices-dir /var/spark/isma/training/slices_v1 \
      --tokenizer <SPARK_HOME>/models/Qwen3.6-27B \
      --out /var/spark/isma/training/cpt_production_v1_packed_2560.jsonl
"""
import argparse, json, os, sys

from corpus_manifest import SCHEMA, sha256_file, write_manifest

SEQ = 2560
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices-dir", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", help="default: <out>.manifest.json")
    args = ap.parse_args()

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
        for fname, want_rows, want_sha in SLICES:
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
                text = json.loads(line)["text"]
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
    os.replace(tmp, args.out)
    dropped = 0
    total_tokens = blocks * SEQ
    print(f"[pack] DONE: docs={stream_docs} blocks={blocks} seq={SEQ} "
          f"tokens={total_tokens} tail_dropped={dropped} "
          f"(final block = {tail_kept} corpus-tail tok + {SEQ - tail_kept} cycle-pad)" if tail_kept
          else f"[pack] DONE: docs={stream_docs} blocks={blocks} seq={SEQ} tokens={total_tokens} tail_dropped=0")

    # SHRINKAGE GATE. Every input's sha was verified above — but a corpus can be perfectly
    # sha-clean and still be missing whole inputs, because the check only validates what you
    # DECIDED to include. On 2026-07-28 two registered slices were dropped on a bad "they exist
    # nowhere" finding and the pack produced 2,511 blocks against the previous production
    # corpus's 3,686. Every per-input check passed. Nothing compared the total.
    prev = os.environ.get("PREV_CORPUS", "")
    if prev and os.path.exists(prev):
        prev_blocks = sum(1 for _ in open(prev, errors="replace"))
        if blocks < prev_blocks * 0.95:
            print(f"[pack] *** SHRINKAGE: {blocks} blocks vs previous corpus {prev_blocks} "
                  f"({blocks/prev_blocks:.0%}). A smaller corpus needs an explicit reason. ***")
            print(f"[pack] set ALLOW_SHRINK=1 with a recorded justification to proceed.")
            if os.environ.get("ALLOW_SHRINK", "") != "1":
                sys.exit("ABORT: corpus shrank against PREV_CORPUS and ALLOW_SHRINK is not set.")
        else:
            print(f"[pack] shrinkage gate OK: {blocks} vs previous {prev_blocks} "
                  f"({blocks/prev_blocks:.0%})")
    else:
        print("[pack] shrinkage gate SKIPPED (set PREV_CORPUS to the last production corpus)")
    corpus_sha = sha256_file(args.out)
    manifest_path = args.manifest or args.out + ".manifest.json"
    if os.path.abspath(manifest_path) == os.path.abspath(args.out):
        sys.exit("ABORT: corpus and manifest paths must differ")
    write_manifest(manifest_path, {
        "schema": SCHEMA,
        "corpus_filename": os.path.basename(args.out),
        "corpus_sha256": corpus_sha,
        "corpus_bytes": os.path.getsize(args.out),
        "corpus_rows": blocks,
        "sequence_length": SEQ,
        "source_documents": stream_docs,
        "tail_corpus_tokens": tail_kept,
        "cycle_pad_tokens": SEQ - tail_kept if tail_kept else 0,
        "packer_sha256": sha256_file(os.path.abspath(__file__)),
        "inputs": input_receipts,
    })
    print(f"[pack] OUTPUT sha256={corpus_sha}")
    print(f"[pack] MANIFEST {manifest_path} sha256={sha256_file(manifest_path)}")
    print(f"[pack] register as: cpt_production_v1_packed_2560 (inputs: "
          + ", ".join(f"{n}@{s}" for n, _, s in SLICES) + ")")


if __name__ == "__main__":
    main()
