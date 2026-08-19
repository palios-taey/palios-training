# Taey system-prompt INDEX — tutor's section (training + the Sparks)

> **This document does not decide the entrypoint.** It names production scripts because it records or explains them. The PRODUCTION AUTHORITY section of `CLAUDE.md` wins on how to run anything; see [`docs/INDEX.md`](../docs/INDEX.md) for the full authority order.

Every pointer below was `stat`ed on 2026-07-28 before sending. Paths are relative to the
`palios-training` repo unless absolute. Repo is public; training DATA is never in it.

---

PROCESS:   CPT — continued pretraining, full-parameter, 4-node
PLAN:      careers-qwen/CONTINUOUS_TRAINING_RECIPE.md
LAUNCH:    tutor runs, on Mira: scripts/taey-train cpt_27b_4node CPT_DATA=<packed corpus> CPT_PACKED=1 MAX_SEQ=2560
           BATCH_SIZE_PER_RANK=4 MODEL_PATH=<serving merged model> TOTAL_STEPS=<derived>
           WARMUP_STEPS=<10% of total> SESSION_LIMIT=<=TOTAL_STEPS SAVE_EVERY=<=SESSION_LIMIT
           CHECKPOINT_DCP=1 OUTPUT_DIR=<ckpt dir>
           Multi-session horizons still go through the same door; do not invoke run_till_done_v3.sh.
EXPECT:    log line `COVERAGE PROOF: steps/epoch=N ... dataset_blocks=M` where N equals
           TOTAL_STEPS for a one-epoch run and N*global_batch >= M; then `Starting: steps 0->N`;
           then step lines whose lr equals base_lr * (step/warmup) during warmup. Done = a
           checkpoint dir at OUTPUT_DIR/checkpoint-<TOTAL_STEPS>.
ON FAIL:   notify tutor. Review careers-qwen/CONTINUOUS_TRAINING_RECIPE.md sections 4b (sizing)
           and 4d (corpus integrity) FIRST — a mismatch between steps/epoch and TOTAL_STEPS, or a
           corpus sha that differs across ranks, is a BUG. A run that completes with correct
           numbers but poor production behaviour is a TRAINING gap.
NEVER:     launch without rebooting all 4 Sparks first; relaunch on dirty GPUs after a kill;
           read `nvidia-smi` for utilisation or memory on GB10 (UMA makes both meaningless —
           use node telemetry); change BATCH_SIZE_PER_RANK mid-horizon (it redefines global_batch
           and therefore what TOTAL_STEPS means).

---

PROCESS:   SFT / LoRA module training
PLAN:      careers-qwen/launch_stage2_sft.sh (self-documenting; sizing derived at run time)
LAUNCH:    tutor runs, on Mira: SFT_CORPUS=<materialized rows> BASE_MODEL=<servable base>
           bash careers-qwen/launch_stage2_sft.sh
EXPECT:    the script prints rows, sha256, global_batch, TOTAL_STEPS, coverage, peak multiplier
           and dose BEFORE launching. peak mult must read 1.0000; anything lower prints
           "ANNEALED TAIL — DO NOT LAUNCH". Done = an adapter dir whose adapter_config.json
           names BASE_MODEL in base_model_name_or_path.
ON FAIL:   notify tutor. Review careers-qwen/CONTINUOUS_TRAINING_RECIPE.md section 4c (lineage)
           before deciding: an adapter naming the wrong base is a BUG in the launch; an adapter
           on the right base that behaves poorly is a TRAINING gap.
NEVER:     train a module against a base other than the current serving model; merge an old
           module forward onto a new base (re-derive instead); trust an adapter's presence in a
           served model without measuring — norms, not assumptions.

---

PROCESS:   Corpus packing (CPT inputs)
PLAN:      careers-qwen/pack_production_corpus.py
LAUNCH:    tutor runs, ON A SPARK (needs the 27B tokenizer):
           python3 pack_production_corpus.py --slices-dir <dir> --tokenizer <27B model dir>
           --out /var/spark/isma/training/<name>_packed_2560.jsonl
EXPECT:    one `VERIFIED` line per input with its registered sha16, then
           `DONE: docs=D blocks=B seq=2560 tokens=T tail_dropped=0`, then an OUTPUT sha256.
           Done = that sha256 identical on all 4 nodes.
ON FAIL:   notify treasurer (corpus owner), not tutor. Review
           treasurer/foundations/careers/training_data/REGISTRY.md. A sha mismatch against a
           registered input is a BUG in provenance — never a reason to re-register the new value
           without understanding what changed.
NEVER:     pack an input whose sha does not match the registry; rename a corpus to match an
           allowlist glob (that is certification by coincidence — add an explicit entry with its
           provenance instead); tokenize any corpus carrying credential-shaped values, because a
           secret in the weights can only be retrained away.

---

PROCESS:   Post-CPT bake and vision graft
PLAN:      careers-qwen/post_cpt_pipeline.sh
LAUNCH:    tutor runs, on Mira: scripts/taey-train bake_export DCP_DIR=<ckpt dir>
           Graft donor is the run's own 1199-tensor source, not a fleet-pinned foreign tower.
EXPECT:    bake yields exactly 851 tensors (text-only — training loads AutoModelForCausalLM so
           the vision tower is never checkpointed); graft yields exactly 1199 (851 patched + 348
           preserved: visual 333 + mtp 15). Both counts are hard aborts. Done =
           training_provenance.json present in BOTH artifact dirs.
ON FAIL:   notify tutor, and infra if it is a serving question. Review the docstring of
           careers-qwen/graft_cpt_into_servable.py. A tensor count other than 851/1199 is a BUG.
NEVER:     serve an ungrafted 851-tensor bake; stamp a config to claim 1199 without grafting;
           promote any artifact branched from an ancestor for measurement.

---

PROCESS:   Training-pair authoring and registration
PLAN:      $OPERATOR_HOME/.claude/skills/taey-training-trigger/SKILL.md
           (governed store: $OPERATOR_HOME/treasurer/foundations/careers/training_data/careers_qwen)
LAUNCH:    the seat that OWNS the failing surface authors the rows — not tutor. Write to the
           governed store, then: cd careers-qwen/data && TRAINING_DATA_ROOT=<store>
           python3 build_pairs_manifest.py
EXPECT:    `no drift — every pair file is classified`. Done = the file appears in
           PAIRS_MANIFEST.md with a status, and treasurer has sanctioned it.
ON FAIL:   notify the authoring seat, then treasurer. Review
           careers-qwen/TAEY_TRAINING_DOCTRINE.md to decide BUG vs TRAINING: if the capability
           does not exist today it is a BUG to fix, and authoring a pair around it manufactures
           confident failure.
NEVER:     author rows that state the wrong way (right-way-only, always); write a .jsonl into the
           palios-training repo (blocked, and training data is never public); train a corpus
           treasurer has not sanctioned; assemble a corpus as tutor — tutor packs, never assembles.

---

## Two pointers to keep honest

- Whatever is currently serving as `ep3` is the base for the next refresh. Read it from the
  serving endpoint's `/v1/models` `root` field, never from memory or from this document.
- What is IN a model is read from `training_provenance.json` in its artifact directory, or from
  an adapter's `adapter_config.json`. If neither exists, the answer is Unknown and must be
  measured, not inferred — that gap cost 4 modules and an entire night on 2026-07-28.
