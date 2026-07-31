#!/usr/bin/env python3
"""Write training_provenance.json INTO an artifact directory, so the artifact carries its own ancestry.

WHY THIS EXISTS
---------------
On 2026-07-28 the served model's lineage could not be established from evidence. LoRA adapters
record their parent because PEFT writes `base_model_name_or_path` into adapter_config.json. CPT
artifacts record NOTHING — config.json is architecture only — and no run log naming a resume base
survived on the nodes. Two questions that should be trivial were both unanswerable:

  "what base did this train from?"   -> unrecoverable
  "what was in the corpus it used?"  -> unrecoverable

The cost was concrete. cpt_refresh_v3 turned out to have resumed directly from prod_v2_ep3_hf,
silently orphaning modules 1 and 3 (4,758 rows). Proving that required a weight-norm argument
against the artifacts themselves, because nothing on disk said so. Separately, the corpus that
trained it (cpt_refresh_v2_packed.jsonl, 805 blocks) appears in no registry, so its contents remain
unknown and unknowable.

A stage that cannot write this record has not completed. That is the whole point: the next person
asking "what is in this model" should read a file, not measure tensors.

USAGE
-----
    python3 emit_training_provenance.py \
        --artifact  <SPARK_HOME>/models/cpt_v5_hf \
        --stage     cpt \
        --base      <SPARK_HOME>/models/module5_merged \
        --corpus    /var/spark/isma/training/cpt_v3_sanctioned_packed_2560.jsonl \
        --total-steps 157 --completed-step 157 --warmup-steps 15 --resumed-step 0 \
        --tooling-commit <controller-git-commit> \
        --sanction  "treasurer task-dfa3fd75 2026-07-28" \
        --corpus-manifest /var/spark/isma/training/cpt_v3_sanctioned_packed_2560.jsonl.manifest.json

Digests are COMPUTED here, never passed in — a digest you were handed is a claim, not a measurement.
"""
import argparse, hashlib, json, os, sys, glob

from corpus_manifest import verify_manifest


def sha256_file(path, limit_mb=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def model_digest(model_dir):
    """Digest of a model directory: the index if sharded, else the concatenated shard digests.

    Hashing 52GB of weights on every stage is not affordable, so this hashes the index (which
    names every tensor and its shard) plus each shard's size. That detects a changed shard set
    and a resharded model. It does NOT detect an in-place edit of shard bytes with identical
    size — recorded here so nobody reads more into this digest than it carries.
    """
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    parts = []
    if os.path.exists(idx):
        parts.append(("index", sha256_file(idx)))
    for f in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        parts.append((os.path.basename(f), str(os.path.getsize(f))))
    if not parts:
        return None, 0
    blob = "|".join(f"{a}:{b}" for a, b in parts).encode()
    n_tensors = 0
    if os.path.exists(idx):
        n_tensors = len(json.load(open(idx))["weight_map"])
    return hashlib.sha256(blob).hexdigest(), n_tensors


def lr_multiplier(step, warmup, total):
    """The trainer's own schedule shape (train_fsdp_dense_9b.py _lr_lambda)."""
    import math
    if step < warmup:
        return step / warmup
    progress = (step - warmup) / max(1, (total - warmup))
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True, help="directory the record is written INTO")
    ap.add_argument("--stage", required=True, help="cpt | sft | merge | graft")
    ap.add_argument("--base", required=True, help="the model this trained FROM")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--total-steps", type=int, required=True,
                    help="schedule horizon supplied to the trainer")
    ap.add_argument("--completed-step", type=int, required=True,
                    help="last checkpoint step this artifact actually contains")
    ap.add_argument("--warmup-steps", type=int, required=True)
    ap.add_argument("--resumed-step", type=int, default=0)
    ap.add_argument("--tooling-commit", required=True,
                    help="controller commit whose exact script bytes were deployed")
    ap.add_argument("--sanction", required=True, help="who sanctioned this corpus, and where")
    ap.add_argument("--corpus-manifest", required=True,
                    help="immutable pack receipt bound to the corpus bytes")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    for p in (args.artifact, args.base):
        if not os.path.isdir(p):
            sys.exit(f"ABORT: not a directory: {p}")
    if not os.path.isfile(args.corpus):
        sys.exit(f"ABORT: corpus not found: {args.corpus}")
    if not 0 <= args.resumed_step <= args.completed_step <= args.total_steps:
        sys.exit("ABORT: require 0 <= resumed-step <= completed-step <= total-steps")
    if len(args.tooling_commit) != 40 or any(c not in "0123456789abcdef" for c in args.tooling_commit):
        sys.exit("ABORT: tooling-commit must be a full lowercase 40-character Git SHA")

    base_digest, base_tensors = model_digest(args.base)
    art_digest, art_tensors = model_digest(args.artifact)
    corpus_sha = sha256_file(args.corpus)
    corpus_rows = sum(1 for _ in open(args.corpus, errors="replace"))
    try:
        corpus_receipt = verify_manifest(args.corpus, args.corpus_manifest)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        sys.exit(f"ABORT: corpus manifest verification failed: {error}")
    if (corpus_receipt["corpus_sha256"] != corpus_sha
            or corpus_receipt["corpus_rows"] != corpus_rows):
        sys.exit("ABORT: corpus receipt changed during provenance generation")

    # dose over the steps this run ACTUALLY executed
    lo = args.resumed_step + 1
    hi = args.completed_step
    dose = sum(lr_multiplier(s, args.warmup_steps, args.total_steps) for s in range(lo, hi + 1))
    peak = max((lr_multiplier(s, args.warmup_steps, args.total_steps) for s in range(lo, hi + 1)),
               default=0.0)

    tooling_sha256 = sha256_file(os.path.abspath(__file__))

    rec = {
        "artifact": os.path.basename(args.artifact.rstrip("/")),
        "artifact_path": args.artifact,
        "artifact_digest": art_digest,
        "artifact_tensors": art_tensors,
        "stage": args.stage,
        "base_model": args.base,
        "base_digest": base_digest,
        "base_tensors": base_tensors,
        "corpus_path": args.corpus,
        "corpus_sha256": corpus_sha,
        "corpus_rows": corpus_rows,
        "corpus_manifest_sha256": corpus_receipt["corpus_manifest_sha256"],
        "corpus_inputs": corpus_receipt["corpus_inputs"],
        "schedule": {
            "total_steps": args.total_steps,
            "warmup_steps": args.warmup_steps,
            "resumed_step": args.resumed_step,
            "completed_step": args.completed_step,
            "steps_executed": max(0, hi - lo + 1),
            "peak_lr_multiplier": round(peak, 6),
            "dose_sum_f": round(dose, 3),
        },
        "sanctioned_by": args.sanction,
        "tooling_commit": args.tooling_commit,
        "tooling_sha256": tooling_sha256,
        "note": args.note,
        "digest_caveat": ("base/artifact digests hash the safetensors index plus per-shard sizes, "
                          "not the full weight bytes. They detect a changed or resharded tensor set, "
                          "NOT an in-place edit preserving shard size."),
    }

    out = os.path.join(args.artifact, "training_provenance.json")
    with open(out, "w") as f:
        json.dump(rec, f, indent=2)
        f.write("\n")

    print(f"WROTE {out}")
    print(f"  stage        {rec['stage']}")
    print(f"  base         {rec['base_model']}  ({base_tensors} tensors, digest {str(base_digest)[:16]})")
    print(f"  corpus       {os.path.basename(args.corpus)}  {corpus_rows} rows  sha {corpus_sha[:16]}")
    print(f"  schedule     total={args.total_steps} warmup={args.warmup_steps} "
          f"resumed={args.resumed_step} completed={args.completed_step} "
          f"executed={rec['schedule']['steps_executed']}")
    print(f"  peak mult    {rec['schedule']['peak_lr_multiplier']}   "
          f"{'(full LR reached)' if peak > 0.99 else '*** PEAK NOT REACHED — annealed-tail run ***'}")
    print(f"  dose sum-f   {rec['schedule']['dose_sum_f']}")
    print(f"  sanction     {rec['sanctioned_by']}")
    if peak <= 0.99:
        print("\nWARNING: this run never reaches full learning rate. That is the exact signature of "
              "cpt_refresh_v3, which executed its entire burst in the annealed tail at f=0.171.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
