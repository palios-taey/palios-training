#!/usr/bin/env python3
"""Derive the 851-tensor causal-LM TRAINING BASE from a 1199-tensor multimodal Qwen3.x checkpoint.

WHY THIS EXISTS. post_cpt_pipeline.sh:425 refuses to convert unless the training-base snapshot has
exactly 851 tensors: `ABORT: exact training-base snapshot has $base_tensors tensors; expected 851.`
A CPT run trains the LANGUAGE tower only, so its checkpoint holds 851 tensors, and a weight-diff
against a 1199-tensor multimodal base is not a comparison — it is two different tensor sets. The
gate is correct.

Until 2026-08-18 that base was produced by hand and left untracked. The s213 campaign used
`<models>/Qwen3.6-27B_training_base_851_causallm`, which exists on the nodes with no provenance
file and no committed deriver. Qwen3.8 had no equivalent, so a completed 218-step run could not be
baked. Same failure class as the untracked corpus builder: a required production input with no
tooling behind it.

THE DERIVATION, measured against the known-good Qwen3.6 base rather than assumed:
    model.language_model.<X>   850  ->  model.<X>      (strip the multimodal wrapper)
    lm_head.weight               1  ->  lm_head.weight
    model.visual.*             333  ->  DROPPED (vision tower)
    mtp.*                       15  ->  DROPPED (multi-token-prediction head)
                              ----
                              1199      850 + 1 = 851 exactly
The known-good Qwen3.6 base is `lm_head:1 + model:850` with names model.embed_tokens.weight,
model.layers.N.*, model.norm.weight — which is precisely what this mapping produces. Qwen3.6 and
Qwen3.8 share the text architecture (hidden 5120, 64 layers), and their config.json differ only in
transformers_version, so the config is copied through unchanged.

SAVE_FILE, NOT SAVE_PRETRAINED — the trap this tool exists to avoid.
save_pretrained on a loaded multimodal model re-emits SERVING names. The count still comes out 851,
so a count check PASSES while the actual name overlap with the trained tensors is 1/851, and the
weight-diff then compares tensors that share no names — silently reporting a delta that means
nothing. This writes raw safetensors with save_file and the names computed above, and VERIFIES BY
NAME, never by count. If a reference base or a run DCP is supplied, the emitted name set must match
it exactly or this aborts.

USAGE
    python3 derive_training_base_851.py --src <1199-dir> --out <851-dir> \
        [--reference-base <known-good 851 dir>] [--verify-dcp <exported artifact dir>]
"""
import argparse
import json
import os
import shutil
import sys

SIDECAR_FILES = ("config.json", "generation_config.json", "tokenizer.json",
                 "tokenizer_config.json", "chat_template.jinja")
LANG_PREFIX = "model.language_model."
KEEP_EXACT = {"lm_head.weight"}
EXPECTED = 851


def target_name(key):
    """Map a source tensor name to its causal-LM name, or None if it is dropped."""
    if key in KEEP_EXACT:
        return key
    if key.startswith(LANG_PREFIX):
        return "model." + key[len(LANG_PREFIX):]
    return None


def read_index(path):
    idx = os.path.join(path, "model.safetensors.index.json")
    if not os.path.isfile(idx):
        raise SystemExit(f"ABORT: no model.safetensors.index.json under {path}")
    return json.load(open(idx))["weight_map"]


def dcp_names(path):
    """Tensor names inside an exported (coordinated) DCP artifact.

    The trainer's own per-rank checkpoint has no global .metadata by design, so this only works on
    the coordinated EXPORT_DCP output. Names there carry an extra FSDP wrapper prefix (model.model.*
    / model.lm_head.weight); it is stripped so the comparison is against the base's naming.
    """
    from torch.distributed.checkpoint import FileSystemReader
    md = FileSystemReader(path).read_metadata()
    out = set()
    for k in md.state_dict_metadata:
        if k.startswith("optim"):
            continue
        out.add(k[len("model."):] if k.startswith("model.") else k)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="1199-tensor multimodal source model dir")
    ap.add_argument("--out", required=True, help="output 851-tensor causal-LM base dir")
    ap.add_argument("--reference-base", help="known-good 851 base to match names against")
    ap.add_argument("--verify-dcp", help="exported DCP artifact to match names against")
    ap.add_argument("--shard-bytes", type=int, default=4 * 1024 ** 3)
    a = ap.parse_args()

    if os.path.abspath(a.src) == os.path.abspath(a.out):
        raise SystemExit("ABORT: --src and --out must differ")
    if os.path.exists(a.out):
        raise SystemExit(f"ABORT: {a.out} already exists — refusing to overwrite a base in place")

    weight_map = read_index(a.src)
    mapping = {}
    for src_key in weight_map:
        dst = target_name(src_key)
        if dst is not None:
            if dst in mapping:
                raise SystemExit(f"ABORT: two source tensors map to {dst}")
            mapping[dst] = src_key
    print(f"[derive] source tensors : {len(weight_map)}")
    print(f"[derive] selected       : {len(mapping)}")
    if len(mapping) != EXPECTED:
        raise SystemExit(
            f"ABORT: mapping produced {len(mapping)} tensors, expected {EXPECTED}. "
            f"The pipeline gate requires exactly {EXPECTED}; refusing to emit a base that cannot "
            f"pass it.")

    # NAME VERIFICATION BEFORE WRITING ANYTHING. A count match is not evidence.
    produced = set(mapping)
    if a.reference_base:
        ref = set(read_index(a.reference_base))
        if produced != ref:
            only_p = sorted(produced - ref)[:5]
            only_r = sorted(ref - produced)[:5]
            raise SystemExit(
                f"ABORT: emitted names do not match the reference base.\n"
                f"  only here: {only_p}\n  only ref : {only_r}")
        print(f"[derive] name-match vs reference base: EXACT ({len(ref)} names)")
    if a.verify_dcp:
        dn = dcp_names(a.verify_dcp)
        if produced != dn:
            only_p = sorted(produced - dn)[:5]
            only_d = sorted(dn - produced)[:5]
            raise SystemExit(
                f"ABORT: emitted names do not match the run's exported DCP.\n"
                f"  only here: {only_p}\n  only dcp : {only_d}")
        print(f"[derive] name-match vs run DCP: EXACT ({len(dn)} names)")

    from safetensors import safe_open
    from safetensors.torch import save_file

    os.makedirs(a.out, exist_ok=True)
    by_shard = {}
    for dst, src_key in mapping.items():
        by_shard.setdefault(weight_map[src_key], []).append((dst, src_key))

    index = {"metadata": {}, "weight_map": {}}
    shard_no = 0
    buf, buf_bytes = {}, 0
    written = []

    def flush():
        nonlocal buf, buf_bytes, shard_no
        if not buf:
            return
        shard_no += 1
        name = f"model-{shard_no:05d}.safetensors"
        # save_file, NOT save_pretrained — see the module docstring.
        save_file(buf, os.path.join(a.out, name), metadata={"format": "pt"})
        for k in buf:
            index["weight_map"][k] = name
        written.append(name)
        print(f"[derive] wrote {name}  ({len(buf)} tensors, {buf_bytes/1024**3:.2f} GB)")
        buf, buf_bytes = {}, 0

    for src_shard in sorted(by_shard):
        with safe_open(os.path.join(a.src, src_shard), framework="pt") as f:
            for dst, src_key in sorted(by_shard[src_shard]):
                t = f.get_tensor(src_key)
                buf[dst] = t
                buf_bytes += t.numel() * t.element_size()
                if buf_bytes >= a.shard_bytes:
                    flush()
    flush()

    total = len(index["weight_map"])
    if total != EXPECTED:
        raise SystemExit(f"ABORT: wrote {total} tensors, expected {EXPECTED}")
    # rename shards to the conventional -of- form now that the count is known
    final_map = {}
    for i, old in enumerate(written, 1):
        new = f"model-{i:05d}-of-{len(written):05d}.safetensors"
        os.rename(os.path.join(a.out, old), os.path.join(a.out, new))
        final_map[old] = new
    index["weight_map"] = {k: final_map[v] for k, v in index["weight_map"].items()}
    with open(os.path.join(a.out, "model.safetensors.index.json"), "w") as fh:
        json.dump(index, fh, indent=2, sort_keys=True)

    for fn in SIDECAR_FILES:
        src = os.path.join(a.src, fn)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(a.out, fn))
            print(f"[derive] copied {fn}")

    with open(os.path.join(a.out, "DERIVED_FROM.json"), "w") as fh:
        json.dump({
            "tool": "careers-qwen/derive_training_base_851.py",
            "source": os.path.abspath(a.src),
            "source_tensors": len(weight_map),
            "emitted_tensors": total,
            "rule": {
                "model.language_model.<X>": "model.<X>",
                "lm_head.weight": "lm_head.weight",
                "model.visual.*": "DROPPED",
                "mtp.*": "DROPPED",
            },
            "writer": "safetensors.torch.save_file (NOT save_pretrained)",
            "name_verified_against_reference_base": bool(a.reference_base),
            "name_verified_against_run_dcp": bool(a.verify_dcp),
        }, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(f"[derive] DONE — {total} tensors at {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
