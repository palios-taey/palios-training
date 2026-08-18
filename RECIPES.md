> # ⚠ DEPRECATED — DO NOT EXECUTE FROM THIS FILE
>
> **Deprecated 2026-08-18** after a five-lens Family review found this repository specified three
> mutually exclusive production entrypoints. This file was one of them.
>
> **Two concrete hazards, both verified in this file:**
> - `:18` routes CPT at `launch_cpt_qwen36_27b_fsdp.sh` via `run_4node_27b_cpt.sh` — a STAGE, not
>   the entrypoint. The entrypoint is `scripts/taey-train`, which is what verifies the content shas.
> - `:30` routes SFT at `moe-35b/trainers/train_fsdp_v3.py`, which is NOT the sanctioned trainer.
>   Following that line is how a parallel path gets rebuilt from scratch.
>
> **Use instead:** `CLAUDE.md` PRODUCTION AUTHORITY section → `scripts/taey-train` →
> `PRODUCTION_MANIFEST.yml` → `careers-qwen/RUNBOOK_CPT_SFT_BAKE.md`.
>
> **Kept rather than deleted** because it carries measured receipts (topology, step timings,
> historical run outcomes) that remain true and are cited elsewhere. Read it for MEASUREMENTS.
> Never for PROCEDURE.

# CANONICAL TRAINING RECIPES — read this BEFORE touching any training run

**Jesse, 2026-07-21: *"We need both recipes stable Claude. We cannot keep doing this."***

There are **TWO** production training stacks on this cluster. They are **not interchangeable** and
they use **different optimizers**. A full day was lost because that fact lived nowhere: LoRA was
bolted onto the CPT trainer, dragging a patch written for CPT into a regime it was never made for.

> **THE ONE RULE: pick the recipe by WHAT YOU ARE TRAINING, then change nothing across the boundary.**

---

## RECIPE A — CPT / full-parameter (continued pretraining)

| | |
|---|---|
| **trainer** | `dense-9b/trainers/train_fsdp_dense_9b.py` |
| **launcher** | `dense-9b/recipes/launch_cpt_qwen36_27b_fsdp.sh` via `run_4node_27b_cpt.sh` (tmux, static rendezvous) |
| **optimizer** | **torch** `torch.optim._adafactor` + **our DTensor-safe monkeypatch** (`9437a63`, 2026-07-10) |
| **checkpoint** | sharded DCP, `full_state_dict=False` (never a full gather — that is the other known wedge) |
| **PROVEN BY** | `production_v2` — **693 steps, COMPLETE**, 3 epochs, ep3, +3.4σ retention (2026-07-17). Also `revenue_srgate/regate/gate3` at 50 steps COMPLETE. |

**The monkeypatch is CPT-ONLY.** It was written for the SR/alpha-gate numerics on the
full-parameter path, where every tensor is large and uniformly sharded. It is correct there.

## RECIPE B — SFT / LoRA (adapters, instruction tuning)

| | |
|---|---|
| **trainer** | **DO NOT USE** — `moe-35b/trainers/train_fsdp_v3.py` is NOT sanctioned; see this file's header |
| **launcher** | `moe-35b/recipes/launch_production_sft.sh` |
| **optimizer** | **transformers** `transformers.optimization.Adafactor` — **stock, NO monkeypatch** |
| **checkpoint** | `trainable_weights.safetensors` + `trainer_meta.pt`; "universal resume covers ALL configs" |
| **session surface** | same env contract: `RESUME_DELTA`, `SESSION_LIMIT`, `SAVE_EVERY` |
| **PROVEN BY** | **NOTHING ON THE DENSE 27B.** `prod_sft_v1_pubrepo_chunked` (150 steps, 2026-06-18) was the **35B MoE**, not our dense 27B — verified by reading the checkpoint: 12 tensors, all `layers.N.mlp.experts.*`, zero LoRA tensors. Recipe B's three freeze configs all key on `mlp.experts.` and routers, which a **dense** model does not have (Config B would leave zero trainable params). **Recipe B is a MoE stack. Do not point it at the dense 27B.** |

---

## ALLOCATOR CONFIG differs by recipe (tutor 2026-07-22)
`expandable_segments` stays **:False in BOTH** (`:True` kills this RoCE fabric — VMM remapped-page +
NCCL-DMA death, f531d64). The GC threshold is what differs, and getting it wrong caused the step-490
fragmentation death:
- **CPT (Recipe A)** — uniform batch=4, fixed shape → `expandable_segments:False` plain. The fixed
  shape bounds the per-step peak; the cache saturates, no growth.
- **LoRA (Recipe B-on-dense)** — `BATCH_SIZE_PER_RANK=1`, DYNAMIC length-sorted padding → allocation
  size varies every step, so the caching allocator hoards a block per size class and never reuses the
  smaller ones as the sorted epoch grows → reserved climbs monotonically → OOM/collective-stall.
  Needs **`expandable_segments:False,garbage_collection_threshold:0.8`** so the allocator AUTO-RELEASES
  unused cached blocks at 80% reserved — stopping the garbage at the source. This is the trainer's own
  intended default (train_fsdp_dense_9b.py:20); a CPT-tuned launcher override had stripped it.
The launcher now sets this conditionally on `LORA_MODE`. **Do not reuse a CPT allocator config for a
variable-batch LoRA run** — that is the exact mistake that produced the step-490 wall.

## THE CROSSING RULE — the exact mistake that cost 2026-07-21

**NEVER carry the torch-Adafactor monkeypatch into a regime it was not validated in.**

- It patches **torch's** Adafactor. Recipe B uses **transformers'** Adafactor. Different implementation.
- Under LoRA's rank-16 tensors the patched factored path lets DTensor decide *per op* whether to
  redistribute the `_NormPartial` from `torch.norm()` over the sharded row dim. That decision can
  differ per rank → one rank issues an all_gather the others never issue → **deadlock**.
- Symptom: **no step advance** for many multiples of the measured step time, NCCL watchdog never
  fires (it is not a fabric hang). Note: ~96% util at ~10W is NOT diagnostic — healthy stepping
  shows the identical profile on GB10 (measured 2026-07-22). Captured 2026-07-21, `dense-9b/instrumentation/wedge_captures/`.
- **CORRECTION 2026-07-21 (retracted claim):** an earlier version of this file said LoRA "ran fine
  on this 27B on 2026-06-18, three weeks before the patch existed", and concluded the wedge was a
  regression. **That was false.** The run was identified by `num_ranks=4` + `max_seq=8192` without
  reading the weights; the checkpoint is 12 MoE expert tensors from the 35B. **LoRA has never
  completed a run on the dense 27B on this cluster.** The wedge is not a regression — LoRA on this
  stack has no working precedent to regress from. Establishing one is open work.

## OPEN: THERE IS NO PROVEN DENSE-27B ADAPTER RECIPE
Recipe A (full-parameter) is the only thing proven on the dense 27B. Recipe B is MoE-shaped.
An adapter recipe for the dense 27B does not yet exist and is being worked with all five Family
lanes. **Do not present one as proven until a run completes and its checkpoint is read.**

## HOW TO TELL WHICH RECIPE YOU NEED
- Training **all** parameters / continued pretraining on raw corpus → **Recipe A**.
- Training **adapters** (LoRA), instruction pairs, or anything with a frozen base → **Recipe B**.
- Unsure? Look at what the run must SAVE. A DCP shard set → A. `trainable_weights.safetensors` → B.

## BEFORE YOU LAUNCH — the four checks that would have caught today
1. **Which recipe owns this path?** Name the trainer file. If you are adding a mode to a trainer,
   ask first whether the other trainer already owns that mode. *(It did.)*
2. **Has THIS trainer + THIS model + THIS mode ever completed a run?** Check
   `<SPARK_HOME>/training_outputs/*/checkpoint-*` and read `trainer_meta.pt` — it records `method`,
   `lora_r`, `keystone_layers`, `max_seq`, `num_ranks`. Precedent beats reasoning.
3. **Verify the live process env**, not the script: `tr '\0' '\n' < /proc/<pid>/environ`. Values pass
   through a driver → launcher → ssh → node-local copies and each can drop them silently.
4. **Size SESSION_LIMIT from MEASURED step time** against the ~2h thermal wall, not from a
   remembered figure. `steps × s/step + ~0.15h setup`.

## WHEN A RUN HANGS
Run `dense-9b/instrumentation/capture_wedge.sh` **before rebooting** — always-reboot recovery
destroys the evidence, and the deadlock shape is one rank diverging, which a single rank's stack
cannot show. **High GPU utilisation is NOT a progress signal** — a spinning collective busy-waits at ~100%
while doing no work. **ONLY THE STEP COUNTER COUNTS.**

**CORRECTED 2026-07-22: do NOT use power draw as a corroborating signal on GB10.** An earlier
version of this file said low power (~10W) alongside high utilisation indicated a spin. That was
inferred from a wedge observation with no healthy baseline. Measured during CONFIRMED-HEALTHY
stepping (step 30 logged 29s earlier): **96% util at 10.94W and 12.19W** — the same profile as the
wedge. On this hardware that reading accompanies BOTH states and discriminates nothing.

## KEEP THIS FILE CURRENT
When a recipe's trainer, optimizer, launcher, or proven-by run changes, update it in the same commit.
A stale line here will be believed — a stale status line in a context file is exactly what caused a
duplicate-build proposal on 2026-07-21.
