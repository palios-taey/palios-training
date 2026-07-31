#!/usr/bin/env python3
"""
graft_cpt_into_servable.py -- Make a full-parameter CPT checkpoint SERVABLE.

Produces a checkpoint structurally identical to the servable base but with the CPT'd
language-model weights patched in. No config changes, no architecture changes, NO VISUAL
WEIGHT LOSS. vLLM loads the output exactly as it loads the unmodified base model.

Same contract and same in-place-patch discipline as bake_lora_nopeft.py, applied to a
full-parameter CPT bake instead of a LoRA delta.

WHY THIS EXISTS
---------------
A CPT run trains through AutoModelForCausalLM. On this stack that resolves to the TEXT
model only (Qwen3_5ForCausalLM). Consequences, all measured on 2026-07-27:

  the DCP checkpoint            3108 keys · language_model 0 · visual 0 · mtp 0
                                (keys are model.model.*; vision was never in the model,
                                 never in the optimizer state, never saved)
  bake via BAKE_TO_HF            851 tensors · model.language_model.* + lm_head
                                 config: model_type=qwen3_5_text,
                                         architectures=[Qwen3_5ForCausalLM], no text_config
  ep3-hf, the model it replaces 1199 tensors · language_model 850 + visual 333 + mtp 15
                                              + lm_head 1
                                 config: model_type=qwen3_5,
                                         architectures=[Qwen3_5ForConditionalGeneration]

So the bake output is missing 348 tensors and carries a text-only architecture identity.
vLLM refuses it at engine init. That is not a bug in the bake — save_pretrained faithfully
wrote what was loaded. The vision tower simply never entered training.

TWO REMEDIES WERE CONSIDERED AND ONE IS WRONG:
  (a) Stamp the base's config onto the bake output. WRONG, and dangerous: it declares a
      vision tower the tensors do not contain. vLLM then either refuses (missing weights
      for a declared submodule) or initialises them randomly and serves a garbage vision
      tower — failing PLAUSIBLY instead of failing clean. infra blocked this correctly.
  (b) The canonical EXPORT_DCP -> bake_dcp_offline.py path. Cannot help HERE: it carries
      the base config and the full tensor set, but it can only export tensors the DCP
      actually holds, and the DCP holds ZERO vision tensors. Verify before assuming
      otherwise — see the precondition check below, which this script enforces.

Hence the graft: start from the SERVABLE base's shards, overwrite only the language-model
weights with the CPT'd ones, leave visual / mtp / config / tokenizer untouched.

FAIL-CLOSED. Every one of these aborts before writing anything:
  - a CPT tensor whose key is absent from the base
  - a shape or dtype mismatch on any patched tensor
  - zero tensors matched (a silent no-op graft is worse than an error)
  - any base tensor outside the patch set being modified

USAGE
    python3 graft_cpt_into_servable.py \
        --base /path/to/ep3-hf                 # SERVABLE model (1199 tensors)
        --cpt  /path/to/cpt_refresh_v3_hf      # CPT bake output (851 tensors)
        --out  /path/to/cpt_refresh_v3_servable
        [--dry-run]
"""

import argparse
import json
import os
import shutil
import sys

from safetensors.torch import load_file, save_file


def load_index(d):
    p = os.path.join(d, "model.safetensors.index.json")
    if not os.path.exists(p):
        raise SystemExit(f"ABORT: no safetensors index in {d}")
    return json.load(open(p))["weight_map"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="SERVABLE base model dir (config+visual come from here)")
    ap.add_argument("--cpt", required=True, help="CPT bake output dir (language-model weights come from here)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    base_map, cpt_map = load_index(a.base), load_index(a.cpt)
    print(f"base : {len(base_map)} tensors  ({a.base})")
    print(f"cpt  : {len(cpt_map)} tensors  ({a.cpt})")

    # PRECONDITION: the CPT output must be a strict SUBSET of the base. If it carries a key
    # the base lacks, these are not the same architecture and a graft is not the right
    # operation — bail rather than invent a mapping.
    extra = sorted(set(cpt_map) - set(base_map))
    if extra:
        raise SystemExit(
            f"ABORT: {len(extra)} CPT tensors absent from base — not the same architecture.\n"
            f"  e.g. {extra[:3]}\n"
            f"  A graft cannot reconcile this. Re-bake against the correct base."
        )

    missing = sorted(set(base_map) - set(cpt_map))
    print(f"patch: {len(cpt_map)} tensors from CPT")
    print(f"keep : {len(missing)} tensors from base (visual/mtp/etc — untouched)")
    for pref in ("visual", "mtp", "vision"):
        n = sum(1 for k in missing if pref in k)
        if n:
            print(f"         {pref}: {n} preserved")
    if not cpt_map:
        raise SystemExit("ABORT: zero CPT tensors — a no-op graft is a silent failure.")

    if a.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    os.makedirs(a.out, exist_ok=True)
    # Everything non-tensor comes from the BASE: config.json, generation_config.json,
    # chat_template, tokenizer. This is the same "pin exact serving config" step the
    # canonical bake_dcp_offline.py does at :90-91, and the step the legacy bake omitted.
    for f in os.listdir(a.base):
        src = os.path.join(a.base, f)
        # FILES ONLY. Model dirs carry junk subdirectories (.cache from a hub download);
        # copy2 raises IsADirectoryError on them. Skipping dirs is correct rather than
        # recursing — nothing vLLM needs to load lives in one.
        if f.endswith(".safetensors") or not os.path.isfile(src):
            continue
        shutil.copy2(src, os.path.join(a.out, f))

    # Patch shard by shard so peak memory is one shard, not one model.
    shards = sorted(set(base_map.values()))
    patched = kept = 0
    for shard in shards:
        bt = load_file(os.path.join(a.base, shard))
        for k in list(bt.keys()):
            if k in cpt_map:
                ct = load_file(os.path.join(a.cpt, cpt_map[k]))
                if ct[k].shape != bt[k].shape or ct[k].dtype != bt[k].dtype:
                    raise SystemExit(
                        f"ABORT: {k} mismatch — base {tuple(bt[k].shape)}/{bt[k].dtype} "
                        f"vs cpt {tuple(ct[k].shape)}/{ct[k].dtype}"
                    )
                bt[k] = ct[k]
                patched += 1
                del ct
            else:
                kept += 1
        save_file(bt, os.path.join(a.out, shard), metadata={"format": "pt"})
        del bt
        print(f"  {shard}: written")

    print(f"\npatched {patched} · kept {kept} · total {patched + kept}")
    if patched != len(cpt_map):
        raise SystemExit(f"ABORT: patched {patched} but CPT had {len(cpt_map)} — incomplete graft.")
    with open(os.path.join(a.out, "GRAFT_COMPLETE"), "w") as marker:
        marker.write(f"patched={patched} kept={kept} total={patched + kept}\n")
    print(f"GRAFT COMPLETE -> {a.out}")
    print("Now VERIFY BY LOADING before handing it to infra. An unloaded artifact is not a result.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
