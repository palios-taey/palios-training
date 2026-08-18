#!/usr/bin/env python3
"""Build registered, document-preserving CPT chunks (unpadded, EOS-terminated).

COMMITTED 2026-08-18. This script previously existed ONLY on the Spark nodes, untracked by any
repo, with no backup — while producing the corpora that every production 27B CPT run trained on.
That is how the 2026-08-18 Qwen3.8 run reached a completed artifact that the bake pipeline then
refused: rank 0's copy had been edited in place (one REGISTRY line, to admit a refreshed repo
slice), the other three nodes still held the prior version, and nothing recorded the difference.
Tracking it here is the fix; the drift is why.

MANIFEST SCHEMA. The sidecar now declares `schema: palios.cpt_packed_corpus.v1` — the contract
careers-qwen/corpus_manifest.py:verify_manifest enforces and careers-qwen/post_cpt_pipeline.sh
requires before it will export or bake. It previously emitted the same facts under builder-local
key names (`format`/`output_sha256`/`output_bytes`/`rows`), which verify_manifest rejects with
"unsupported corpus manifest schema: None". Every downstream provenance gate binds to this
receipt, so a corpus without one cannot be baked, served, or explained later — which meant each
nopack corpus needed a hand-written sidecar. The serving model's corpus carries exactly such a
hand-made receipt, marked `regenerated_note`. Emitting the canonical shape here removes the need.

The emitted key set deliberately matches that known-good sidecar:
  schema, corpus_filename, corpus_sha256, corpus_bytes, corpus_rows,
  builder_format, builder_sha256, max_seq, overlap, source_documents, tokens, inputs
plus eos_token_id / min_tokens / max_tokens / length_histogram as build diagnostics.

--manifest-only regenerates the receipt for an EXISTING corpus without rebuilding it. Nothing is
hand-entered: corpus_sha256/bytes/rows are recomputed from the corpus file on disk, and the input
receipts are recomputed from the slice files under the same REGISTRY content-pin. Build-time
diagnostics that cannot be recovered without re-tokenising are carried over from the existing
sidecar and are not gate fields. Use it only when the corpus bytes are already final.
"""
import argparse, hashlib, json, os, pathlib, tempfile

SCHEMA = "palios.cpt_packed_corpus.v1"
BUILDER_FORMAT = "cpt_nopack_document_chunks_v2"

REGISTRY = {
    "cpt_public_repos_v2.jsonl": (1167, "779b4234936bc9fe"),
    "cpt_identity_v1.jsonl": (17, "ebdd56e8237ac681"),
    "cpt_raw_corpus_v4.jsonl": (946, "fd64cb08341238ea"),
    "cpt_careers_kb_v1.jsonl": (385, "4743ee60da81dde1"),
    "cpt_careers_db_worldmodel_v1.jsonl": (32, "02c203a3c2526141"),
    "cpt_strategy_research_delta_v1_SCRUBBED.jsonl": (147, "0a81a0af5e58181d"),
}


def _sha256_file(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def _corpus_rows(path):
    """Count rows the way corpus_manifest.corpus_rows does — newline bytes, nothing cleverer.
    Counting any other way here would produce a receipt that its own verifier rejects."""
    rows = 0
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            rows += chunk.count(b"\n")
    return rows


def _input_receipts(slices_dir, inputs):
    """Recompute each input receipt under the REGISTRY content-pin. Shared by build and regenerate
    so the two paths cannot drift apart."""
    receipts = []
    for name in inputs:
        path = pathlib.Path(slices_dir, name)
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        expected_rows, prefix = REGISTRY[name]
        actual_rows = sum(1 for line in raw.splitlines() if line.strip())
        if actual_rows != expected_rows or not digest.startswith(prefix):
            raise SystemExit(f"registry mismatch {name}")
        receipts.append({"name": name, "rows": actual_rows, "sha256": digest,
                         "registered_sha256_prefix": prefix, "bytes": len(raw)})
    return receipts


def _write_manifest(mpath, manifest):
    mt, mtmp = tempfile.mkstemp(prefix=mpath.name + ".", dir=mpath.parent)
    os.close(mt)
    with open(mtmp, "w") as mf:
        json.dump(manifest, mf, indent=2, sort_keys=True)
        mf.write("\n")
        mf.flush()
        os.fsync(mf.fileno())
    os.replace(mtmp, mpath)


def _validate_common(a):
    if a.overlap < 0 or a.overlap >= a.max_seq:
        raise SystemExit("invalid overlap")
    if not a.inputs or any(n not in REGISTRY for n in a.inputs):
        raise SystemExit("unregistered input")
    # A repeated input name passes the registry check every time and contributes its documents once
    # per occurrence, so the manifest would report a corpus larger than the registered slices.
    _dupes = sorted({n for n in a.inputs if a.inputs.count(n) > 1})
    if _dupes:
        raise SystemExit(f"duplicate input name(s), a registered slice may be counted once only: {_dupes}")


def regenerate_manifest(a):
    """Emit a canonical-schema receipt for a corpus whose bytes are already final."""
    out_path = pathlib.Path(a.out)
    if not out_path.is_file():
        raise SystemExit(f"--manifest-only needs an existing corpus at {out_path}")
    mpath = pathlib.Path(str(out_path) + ".manifest.json")
    prior = {}
    if mpath.is_file():
        with open(mpath) as handle:
            prior = json.load(handle)

    receipts = _input_receipts(a.slices_dir, a.inputs)
    digest = _sha256_file(out_path)
    prior_digest = prior.get("corpus_sha256") or prior.get("output_sha256")
    if prior_digest and prior_digest != digest:
        raise SystemExit(
            f"corpus bytes changed since the prior sidecar was written "
            f"(prior {prior_digest}, actual {digest}) — refusing to regenerate a receipt for a "
            f"corpus that is not the one the prior manifest described")

    manifest = {
        "schema": SCHEMA,
        "corpus_filename": out_path.name,
        "corpus_sha256": digest,
        "corpus_bytes": out_path.stat().st_size,
        "corpus_rows": _corpus_rows(out_path),
        "builder_format": prior.get("builder_format") or prior.get("format") or BUILDER_FORMAT,
        "builder_sha256": prior.get("builder_sha256"),
        "max_seq": prior.get("max_seq", a.max_seq),
        "overlap": prior.get("overlap", a.overlap),
        "inputs": receipts,
        "regenerated_note": (
            "canonical-schema sidecar regenerated by build_cpt_nopack_corpus.py --manifest-only; "
            "corpus_sha256/corpus_bytes/corpus_rows recomputed from the corpus file and input "
            "receipts recomputed from the registered slices; the corpus itself was NOT rebuilt"),
    }
    for key in ("eos_token_id", "source_documents", "tokens", "min_tokens", "max_tokens",
                "length_histogram"):
        if key in prior:
            manifest[key] = prior[key]
    _write_manifest(mpath, manifest)
    print(json.dumps({"regenerated": str(mpath), "corpus_sha256": digest,
                      "corpus_rows": manifest["corpus_rows"],
                      "corpus_bytes": manifest["corpus_bytes"]}))


def build(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
    eos = tok.eos_token_id
    if eos is None:
        raise SystemExit("tokenizer has no eos_token_id")
    receipts = _input_receipts(a.slices_dir, a.inputs)
    lengths = []
    docs = tokens = 0
    out_path = pathlib.Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=out_path.name + ".", dir=out_path.parent)
    os.close(fd)
    try:
        with open(tmp, "w") as out:
            for name in a.inputs:
                path = pathlib.Path(a.slices_dir, name)
                with path.open() as src:
                    for original_row_id, line in enumerate(src):
                        if not line.strip():
                            continue
                        # FAIL CLOSED ON EVERY NONBLANK ROW. The registry check counts nonblank
                        # LINES, so a row that parses but carries no usable text was dropped here
                        # silently and the count still matched — source documents could vanish from
                        # a corpus that reported itself complete. And `str(obj.get("text",""))` was
                        # worse than a drop: it COERCED, so {"text": null} became the literal string
                        # "None" and {"text": 123} became "123", both of which tokenise and train as
                        # content. A nonblank row is either a valid document or a hard failure.
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError as e:
                            raise SystemExit(f"{name}:{original_row_id}: not valid JSON ({e})")
                        if not isinstance(obj, dict):
                            raise SystemExit(f"{name}:{original_row_id}: row is {type(obj).__name__}, expected a JSON object")
                        if "text" not in obj:
                            raise SystemExit(f"{name}:{original_row_id}: missing 'text' field")
                        raw_text = obj["text"]
                        if not isinstance(raw_text, str):
                            raise SystemExit(f"{name}:{original_row_id}: 'text' is {type(raw_text).__name__}, expected string (no coercion — it would train as literal text)")
                        text = raw_text.strip()
                        if not text:
                            raise SystemExit(f"{name}:{original_row_id}: 'text' is empty or whitespace-only")
                        docs += 1
                        ids = tok(text, add_special_tokens=False)["input_ids"]
                        start = 0
                        chunk_id = 0
                        while start < len(ids):
                            end = min(start + a.max_seq - 1, len(ids))
                            chunk = ids[start:end] + [eos]
                            row = {"input_ids": chunk, "doc_id": f"{name}:{original_row_id}",
                                   "chunk_id": chunk_id, "token_count": len(chunk),
                                   "original_row_id": original_row_id, "is_continuation": start > 0,
                                   "source_file": name,
                                   "source_type": "constitutional" if "identity" in name else "work_knowledge"}
                            if row["token_count"] != len(chunk) or not chunk or chunk[-1] != eos or len(chunk) > a.max_seq:
                                raise RuntimeError("invalid emitted chunk")
                            out.write(json.dumps(row, separators=(",", ":")) + "\n")
                            lengths.append(len(chunk))
                            tokens += len(chunk)
                            chunk_id += 1
                            if end == len(ids):
                                break
                            start = end - a.overlap
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, out_path)
        digest = _sha256_file(out_path)
        manifest = {
            # canonical contract — corpus_manifest.verify_manifest / post_cpt_pipeline.sh
            "schema": SCHEMA,
            "corpus_filename": out_path.name,
            "corpus_sha256": digest,
            "corpus_bytes": out_path.stat().st_size,
            "corpus_rows": _corpus_rows(out_path),
            # builder identity and settings
            "builder_format": BUILDER_FORMAT,
            "builder_sha256": _sha256_file(pathlib.Path(__file__)),
            "max_seq": a.max_seq,
            "overlap": a.overlap,
            "eos_token_id": eos,
            # build diagnostics
            "source_documents": docs,
            "tokens": tokens,
            "min_tokens": min(lengths),
            "max_tokens": max(lengths),
            "length_histogram": {str(n): lengths.count(n) for n in sorted(set(lengths))},
            "inputs": receipts,
        }
        if manifest["corpus_rows"] != len(lengths):
            raise RuntimeError(
                f"emitted {len(lengths)} chunks but the corpus file has {manifest['corpus_rows']} "
                f"rows — the receipt would not verify against its own corpus")
        _write_manifest(pathlib.Path(str(out_path) + ".manifest.json"), manifest)
        print(json.dumps({"rows": len(lengths), "tokens": tokens, "min": min(lengths),
                          "max": max(lengths), "sha256": digest}))
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices-dir", required=True)
    ap.add_argument("--tokenizer")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seq", type=int, default=8192)
    ap.add_argument("--overlap", type=int, default=256)
    ap.add_argument("--manifest-only", action="store_true",
                    help="regenerate the canonical sidecar for an existing corpus; does not rebuild it")
    ap.add_argument("inputs", nargs="+")
    a = ap.parse_args()
    _validate_common(a)
    if a.manifest_only:
        regenerate_manifest(a)
        return
    if not a.tokenizer:
        raise SystemExit("--tokenizer is required to build a corpus")
    build(a)


if __name__ == "__main__":
    main()
