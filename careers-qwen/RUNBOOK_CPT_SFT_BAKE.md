# RUNBOOK — CPT / SFT / bake, in order

> **ENTRYPOINT NOTE (2026-08-18).** This runbook is authoritative for CPT/SFT/bake PROCEDURE. It is
> NOT a second entrypoint. Where it names drivers such as `run_till_done_v3.sh` or
> `run_refresh_gate.sh`, those are INTERNAL drivers in the same sense `PRODUCTION_MANIFEST.yml`
> records for `bake_27b.sh` — "NOT the top-level entrypoint ... this pipeline's internal stage".
> The single legal door is `scripts/taey-train`, per the PRODUCTION AUTHORITY section of
> `CLAUDE.md`, which wins over this file on entrypoint questions.

**Canonical as of 2026-07-30.** Written after a null CPT run and an unservable bake in the same 24h.
Every step here exists because skipping it has cost us something real, and the cost is named.

**THE ONE RULE THAT WOULD HAVE PREVENTED BOTH INCIDENTS:**
> Use the canonical wrapper. Never call the inner launcher with hand-rolled env.
> `run_4node_27b_cpt.sh` has 18 inbound refs and accepts anything. The wrappers
> (`run_till_done_v3.sh`, `run_refresh_gate.sh`) set eight variables correctly. Calling the
> inner launcher directly gets **none** of them, and nothing warns you.

---

## 0. PRECONDITIONS — every run, no exceptions

```bash
# all 4 nodes freshly rebooted + pristine. The launcher assumes this and does NOT do it.
for n in $SPARK_NODES; do ssh spark@$PRIVATE_MGMT_IP 'sudo systemctl reboot'; done
# wait, then verify — pristine is avail ~108G, shm_nccl 0, zero torch procs
for n in $SPARK_NODES; do ssh spark@$PRIVATE_MGMT_IP \
  'echo "avail=$(free -g|awk "/Mem:/{print \$7}")G shm=$(ls /dev/shm/nccl* 2>/dev/null|wc -l) torch=$(ps -eo pid,cmd|grep -cE "[t]orchrun|[t]rain_fsdp") free=$(df -h $HOME|tail -1|awk "{print \$4}")"'; done
```
**Never `pgrep -fc` or a bare `grep torchrun`** — both self-match and report phantom processes.
Use `grep -E "[t]orchrun"`. Disk: a 27B checkpoint needs ~50G; `.68` hit 100% on 2026-07-26 and
that alone can wedge a save.

## 1. CORPUS — sanctioned, content-pinned, verified on every node

- Corpus is **treasurer-sanctioned only**. tutor never assembles training data unilaterally.
- A doctrine sentence describing a corpus is **not** a sanction record. Only a sanction row is.
  (2026-07-27: I read "the sanctioned cpt_refresh_v2" as sanction; the launcher said
  "treasurer sanction NOT yet obtained".)
- Verify the content pin on **all four nodes**, not one:
```bash
for n in $SPARK_NODES; do ssh spark@$PRIVATE_MGMT_IP "sha256sum <SPARK_HOME>/<corpus>.jsonl"; done
```
- Report composition **by parsing the rows**, never from a spec comment. The launcher has twice
  described content the artifact did not contain.
- Production packing is incomplete until `<corpus>.manifest.json` exists. The packer writes that
  sidecar with the packed-corpus SHA plus every source file's full SHA and registered prefix.
  `post_cpt_pipeline.sh` verifies the sidecar against the corpus bytes before reboot or export,
  carries it to the conversion host, and provenance generation reads inputs only from that
  receipt. A later `CORPUS_INPUTS` environment value is a cross-check, never the lineage source.
  This would have stopped CPT v7's first provenance record, which named post-scrub sources even
  though the packed artifact had been built from their pre-scrub predecessors.
- **A NOPACK corpus needs that same receipt, and until 2026-08-18 its builder did not write one.**
  `build_cpt_nopack_corpus.py` emitted its facts under builder-local key names
  (`format`/`output_sha256`/`output_bytes`/`rows`), while `corpus_manifest.verify_manifest`
  requires `schema: palios.cpt_packed_corpus.v1` with `corpus_sha256`/`corpus_bytes`/`corpus_rows`
  and rejects anything else — *"unsupported corpus manifest schema: None"*. The name says packed;
  the contract is not packing-specific and applies to every CPT corpus. So a nopack corpus trained
  fine and then could not be baked, and each one needed a hand-written sidecar. The served model's
  corpus carries exactly such a hand-made receipt, marked `regenerated_note`.
  The builder now emits the canonical shape. For a corpus whose bytes are already final:
```bash
python3 careers-qwen/build_cpt_nopack_corpus.py --manifest-only \
  --slices-dir <slices> --out <corpus>.jsonl <slice>.jsonl ...
python3 careers-qwen/corpus_manifest.py verify --corpus <corpus>.jsonl --manifest <corpus>.jsonl.manifest.json
```
  It recomputes SHA, byte count and row count from the corpus file and the input receipts from the
  registered slices; nothing is hand-entered, and it refuses if the corpus bytes differ from what
  the prior sidecar described. **The builder itself was untracked until 2026-08-18** — it existed
  only on the nodes, had drifted between ranks (rank 0 edited in place, three ranks on the prior
  version), and produced every production corpus including the served model's. It is committed now;
  edit it in the repo and ship it, never in place on a node.

## 2. CPT — via the canonical wrapper

```bash
scripts/taey-train cpt_27b_4node [VAR=val ...]     # the ONE door — verifies content shas
```

The manifest declares `lifecycle: true` only for `cpt_27b_4node` and `bake_export`; the launcher
never infers lifecycle ownership from a name or argument. Those runs append every transition to
their `lifecycle_events.jsonl` journal under the selected output directory. `taey-train` returns
non-zero after an intermediate fragmentation exit or `CHECKPOINT_SAVED`; that is an honest
incomplete lifecycle, not a failed checkpoint. Exit 0 for a lifecycle-declaring operation is
reserved for `bake_export` after the exact sealed artifact has been verified on Thor and
`THOR_DELIVERED` is appended. Capabilities without the declaration return their entrypoint status
and do not require Spark topology or a lifecycle journal.

**AN UNRESOLVED TENSION, STATED RATHER THAN PAPERED OVER (2026-08-18).** `run_till_done_v3.sh`
exists because it sets the eight variables below correctly, and that is real value — the launcher
hard-aborts on ZERO variables and silently ASSIGNS legacy values for 13 of them
(`run_4node_27b_cpt.sh:27-32,142,146`: `TOTAL_STEPS:=3000`, `MAX_SEQ:=2560`, `SESSION_LIMIT:=200`).
So "always use `taey-train`" and "always use the wrapper that sets the eight" are BOTH good advice
and they currently point at different commands.

The resolution is a code change, not a doc edit: either `taey-train` gains a capability that wraps
the parameter-setting driver, or the driver is invoked through `taey-train`. Until that lands:
**go through `taey-train`, and set the eight explicitly and verify them**, because the gate that
checks the code is worth more than the convenience that sets the parameters — and an unset variable
here does not fail, it runs a different campaign. Tracked as P1/P2 in `docs/REMEDIATION_PLAN.md`.
If you must parameterise, these are the eight the wrapper sets and **all of them matter**:
```
TOTAL_STEPS=<campaign horizon>   SESSION_LIMIT=<burst>   SAVE_EVERY=$SESSION_LIMIT
LR=1e-5   WARMUP_STEPS=15   CLOCK_CAP=1600
ADAFACTOR_ALPHA_MODE=absolute   ADAFACTOR_EPS1=fp32   ADAFACTOR_DOSE_LOG=1
RESUME_DELTA=<ckpt for a continuation>   MODEL_PATH=<base architecture; always required>
```
**`TOTAL_STEPS` is the DECAY HORIZON, not the burst.** `SESSION_LIMIT` is the burst. Setting
`TOTAL_STEPS` equal to one burst compresses a whole cosine schedule into it — warmup eats half,
decay kills the rest. That is what produced the 2026-07-26 null run.

**A continuation requires BOTH `MODEL_PATH` and `RESUME_DELTA`.** The controller's early gate accepts
either one, but the per-node launcher requires `MODEL_PATH` unconditionally because it constructs the
model before the trainer loads the DCP checkpoint post-prepare. Missing `MODEL_PATH` aborts loudly on
every rank. Missing `RESUME_DELTA` is worse: it silently trains the raw base and nothing warns. Name the
same base architecture that created the checkpoint, name the completed DCP checkpoint, verify both in
the resolved-config banner, and require `Scheduler fast-forwarded to opt-step N` before accepting the
resume. `SAVE_EVERY=SESSION_LIMIT` means final-only — **no mid-run saves, they break things** (Jesse,
standing).

**Verify while it runs:**
- `COVERAGE PROOF: blocks/epoch ≈ dataset_blocks` — the corpus is fully consumed
- `Scheduler fast-forwarded to opt-step N` — the resume actually took
- `[SR-DELTA] mean|dW| = X ULP` — **this is the oracle, not the loss.** `<0.5u` = FAIL-LOW.
  Loss wobbling 2.4→1.0→2.8 is batch noise on a model that is not learning.

## 3. BAKE — Artifact B assembled and baked node-local on one Spark

```bash
scripts/taey-train bake_export DCP_DIR=<completed-run>
```

**CHECK THE CHECKPOINT IT PICKED. Read the launch banner before you walk away.** The wrapper prints
`checkpoint selection: ...` and then `27B BAKE launch <time> — <path> →`. That path is the model you
are shipping. Until 2026-08-18 the selector globbed `checkpoint-*` and sorted numerically, but
`train_fsdp_dense_9b.py:3643` names a COMPLETED run's save `final`, not `checkpoint-<step>`:
`ckpt_name = "final" if final else f"checkpoint-{step}"`. So the glob could not see the finished
model and silently chose the last INTERMEDIATE checkpoint. Observed on cpt_qwen38_v3, which held
checkpoint-73, checkpoint-146 and final(step=218): it selected 146 and launched the bake on
two-thirds of the run. **Every downstream gate passed** — the artifact handed to them was internally
consistent, just the wrong model. The defect had been latent since the pipeline was written because
every prior bake ran on an INTERRUPTED run, whose last save genuinely was `checkpoint-<N>`; the
adjudicated receipt (cpt_v7_eps1fix, checkpoint-148) is one of those. **A run that completes is the
one that exposes it.** Fixed: `final/` wins when present, its step read from its own
`trainer_meta.pt`, and the pipeline aborts rather than defaulting a completed-step it cannot read.

`fleet.env` supplies the Thor delivery host/root and immutable conversion-image digest. Durable
controller storage is optional and is used only after Thor delivery. The 1199-tensor donor is never
read from `fleet.env`: the wrapper resolves it from the run's captured `TRAIN_BASE` (1199 uses
itself; 851 must name its source in `DERIVED_FROM.json`; any other shape or missing sidecar aborts).
It verifies the packed-corpus input manifest and checkpoint, reboots all four nodes and proves
changed boot IDs plus zero trainers, deploys the export runtime byte-exact, then calls canonical
`EXPORT_DCP` in `bake_27b.sh`.
`artifact_b_sync.sh` verifies every shard against its rank manifest and assembles Artifact B on
rank 0 without staging it on controller disk. Conversion, weight-diff, graft, provenance and the
1199/851 gates run node-local in the pinned container. The completed servable artifact is sealed
with a full-content manifest, pushed Spark→Thor directly and verified there before the lifecycle
records `THOR_DELIVERED`. Only then is durable storage populated and Spark transients retired.

Offline conversion runs in the pinned torch 2.10 / transformers 5.3 container. The wrapper writes
the measured diff into the candidate and stops before graft, handoff, or SFT unless
`5e-05 <= mean|dW| <= 8e-04`. `BAKE_TO_HF` is legacy full-gather mode and is not production.

If the exact result is below band but Jesse directs a reviewed trial, do not rerun export and do
not bypass the gate in place. Run the approved Chats consultation, create the hash-bound
`palios.post_cpt_below_band_review.v1` receipt, then continue from the retained 851-tensor HF/base:

```bash
DCP_DIR=<completed-run> \
BELOW_BAND_REVIEW_RECEIPT=<absolute-review-receipt.json> \
bash careers-qwen/finalize_post_cpt_candidate.sh
```

The continuation refuses above-band weights. For below-band weights it verifies that the receipt
matches the exact `weight_diff.json`, Jesse's trial instruction, the review packet, independent
peer review, and at least two Chat outputs. It then uses the same pinned conversion image and
1199-tensor donor, writes the actual completed checkpoint separately from the schedule horizon,
and retires the reproducible transients only after candidate verification.

The text-only conversion emits **851 tensors**. Production serves **1199**. Training loads
`AutoModelForCausalLM`, so the vision tower is never in the model, optimizer state, or DCP.

## 3a. ARTIFACT MOVEMENT — bake node-local, Thors FIRST, Expansion LAST

**Jesse-directed 2026-08-18, verbatim: "You finish training, you bake it in the most efficient manner
possible, and you get it on Thors and then you can back it up on Expansion." And: "When something
finishes training, we need to get it off Sparks and on Thors so we can do next round of training as
fast as possible."** The Sparks are the scarce resource — every hour an artifact sits on them is an
hour the next run cannot start. Backup is durability, not delivery, and durability is never on the
critical path.

**THE ORDER:**

1. **Bake on ONE Spark, node-local.** The conversion and the graft are single-process, single-machine
   CPU work. They do not need four nodes, a controller, or an off-cluster host. Put the exported
   artifact, the training base, the donor and the tools on one node and run both steps there.
2. **Push the finished servable STRAIGHT to the Thors,** Spark → Thor directly. Do not stage through
   the controller first.
3. **Then** copy to Expansion for durable backup, off the critical path, with the model already live.

**MEASURED, on cpt_qwen38_v3 (2026-08-18), so this is not a preference:**

| step | where | measured |
|---|---|---|
| DCP → HF convert (851) | one Spark, node-local | **107 s** (20:46:41 → 20:48:28) |
| graft 851 → 1199 | same Spark, node-local | **50 s** (20:56:20 → 20:57:10) |
| servable → Thor1, 52G | Spark → Thor direct rsync | **112 MB/s, ~8 min** |
| the same 52G via the controller artifact store | USB-attached | **19 MB/s, ~46 min each way** |

Routing a 52G artifact through the controller store costs roughly an hour and a half of round trip to
move bytes that never needed to go there. The direct path is ~8 minutes.

`post_cpt_pipeline.sh` now enforces this order mechanically. `ARTIFACT_STORE` may be absent; when it
is present the copy is explicitly stage 8, after the serving-host content gate and
`THOR_DELIVERED`. The export-skip predicate executes the per-rank manifest/hash verifier, so
`.metadata` plus four READY markers cannot conceal a missing shard.

**MEASURE YOUR OWN PROCESSES BEFORE BLAMING THE NETWORK.** During this transfer the rate fell from
112 MB/s to 7.5 MB/s and the ETA jumped from 4 minutes to 90. The cause was not the fabric or the
Thor — it was `scp` and two `safetensors` reads this seat was running against the receiving host at
that moment. It recovered to 112 MB/s within 75 seconds of stopping them. A receiving host that is
also serving a 27B model has no spare IO to lend; do not read from it while a transfer lands.

## 3b. THE WEIGHT-DIFF BASE MUST SHARE NAMING WITH THE BAKE OUTPUT

`measure_cpt_delta.py` selects decoder tensors from the names COMMON to base and candidate
(`:86-90`). Hand it two artifacts that name the same weights differently and the intersection is
empty, and it exits `ABORT: no decoder mlp/attn weights found — check the model layout`. **That abort
is about the pair you gave it, not about the model.** Observed on cpt_qwen38_v3: base and candidate
were both exactly 851 tensors and the common-name count was **1** (`lm_head.weight`).

The cause is benign and worth knowing, because the instinct is to suspect the bake:

```
851 training base (derived) :  model.layers.N.*          causal-LM naming
bake output (save_pretrained):  model.language_model.N.*  SERVING naming
```

`Qwen3_5ForCausalLM.__init__` does `self.model = Qwen3_5TextModel(config)`, so the model's own
state_dict keys are `model.layers.N.*` — which is what both the training base and the DCP use
(`dcp key = "model." + state_dict key`, verified: `model.model.layers.22.mlp.up_proj.weight`). The
load path matches exactly. It is `save_pretrained` that applies the reverse checkpoint-conversion
mapping on the way out and writes the SERVING names. **The output being serving-named is correct and
is exactly what the graft wants** — it is not a defect to be fixed.

**So compare against the run's own 1199-tensor SOURCE model, not against the derived 851 base.** The
source shares all 851 language names with the bake output, needs no mapping, and is the artifact
training actually started from:

```bash
python3 careers-qwen/measure_cpt_delta.py \
  --base <SPARK_HOME>/models/<the model the run loaded> \   # 1199-tensor source
  --cand <bake output>
```

**Do not "fix" this by renaming tensors in either artifact to make the tool agree.** Renaming to
satisfy a comparison is how a diff gets computed between weights that were never the same weights.

## 4. GRAFT — mandatory, or the artifact cannot serve

```bash
python3 careers-qwen/graft_cpt_into_servable.py \
  --base <SPARK_HOME>/models/Qwen3.6-27B \        # 1199-tensor structural donor
  --cpt  <bake output> --out <servable> [--dry-run]
```

**THE DONOR FOLLOWS THE RUN'S BASE MODEL — and `fleet.env` does not.** This runbook has always said
the donor is the run's own source model (above). `fleet.env` pins
`POST_CPT_GRAFT_BASE=<serve-models>/module5_merged`, a fixed artifact from an earlier lineage. Both
are 1199 tensors and both are `Qwen3_5ForConditionalGeneration` / `model_type qwen3_5`, so a graft
either way is architecturally valid and the tensor-count gates pass either way. All 851 language
tensors are replaced regardless, so the ONLY thing the donor contributes is the vision tower, the mtp
head and the config. On cpt_qwen38_v3 the run's own source was used, per this runbook.

**And the two vision towers are NOT the same weights — measured, not assumed.** sha256 of the raw
tensor bytes, three vision tensors sampled across the tower:

| tensor | run's own source | `module5_merged` (the pin) |
|---|---|---|
| `model.visual.blocks.0.attn.proj.bias` | `ea9264ae519e8a02` | `54d0cecb31e424a8` |
| `model.visual.blocks.15.attn.proj.bias` | `848c549105fae0b4` | `03a381eecb0992cc` |
| `model.visual.patch_embed.proj.bias` | `8acebc63dfde0d75` | `f5250290478a5bfc` |

Three of three differ. So the pin is not a harmless default — following it on a new base model
grafts a **foreign vision tower** onto the trained language weights. Tensor count is still 1199,
visual is still 333, mtp is still 15, the name sets still match and **every gate in this runbook
still passes.** Nothing mechanical catches it. That is the same defect shape as a cache keyed by run
name instead of source digest: correct until the day the lineage changes, then silently wrong with
every gate green. **Resolve the donor from the run's base model, never from a fixed path.**
**Never fix this by stamping the base config onto the 851-tensor bake.** That declares a vision
tower the tensors lack; vLLM then refuses, or worse initialises it randomly and serves a garbage
tower — failing *plausibly* instead of clean.

## 5. VERIFY BEFORE HANDOFF — an unverified artifact is not a result

```
tensors 1199 · visual 333 · mtp 15
model_type qwen3_5 · architectures ['Qwen3_5ForConditionalGeneration'] · text_config PRESENT
CPT delta present: sample language-model tensors vs the PRE-CPT base — they MUST differ,
  or you are shipping the old weights with extra steps
```
State exactly what you checked. **`AutoConfig` loading is not a model load, and a model load is
not a vLLM serve.** Say which one you did.

## 6. HANDOFF — infra owns serving

tutor produces artifacts; **infra decides what gets stood up**. Never replace a known-good served
model with an unmeasured one — stand up **alongside**, on a **separate node** (a Thor cannot serve
two 27B models; vLLM reserves ~92% of unified memory).

## 7. EVALUATION — production only

No synthetic batteries. The simple one-tool matrix **passed for both the healthy and the degraded
model**, so a pass on it is *no* evidence, not weak evidence. Evaluate on the real apply unit.
Baseline from `ep3-hf`'s verified behaviour: one tool call · `tool_name=ui_action` · zero prose ·
every argument key a real schema field.

---

## SFT / LoRA — the differences

The current Stage-2 path is four-node replicated-base DDP LoRA. The old FSDP Stage-2 wrapper is
not the CPT-v7 production path.

Before any full production SFT, run the single bounded receipt-faithful candidate from a clean
committed branch:

```bash
FLEET_ENV=<absolute-private-fleet.env> \
RUN_TAG=cpt_v7_eps1fix \
SFT_CORPUS=<absolute-sanctioned-jsonl-on-every-Spark> \
EXPECTED_SFT_SAMPLES=10033 \
QUAL_TAG=<unique-tag> \
DEPLOY_SHA=<exact-commit> \
bash careers-qwen/qualify_stage2_sft_ddp.sh
```

The candidate performs 192 optimizer steps: 64 short, 64 mid, and 64 long production groups. It
uses the exact 979-step production topology: max sequence 1792, padding 16, thresholds 160/512,
local batches 8/2/1, selective activation checkpointing 1536/1472, 2000 MHz, dual-rail NCCL,
packing off, and
`expandable_segments:False,garbage_collection_threshold:0.8`. It observes system
`MemAvailable` after every completed step; synchronizes and calls `empty_cache` after fixed steps
24, 48, 72, 96, 120, 144, 168, and 192; and performs the same release immediately if any rank is
below 8 GB before the next step. A sub-8 GB pre-release observation is a trigger, not a failure.
Every actual release must restore every rank to at least 40 GB. A step with no release must remain
at or above 8 GB. Python cyclic GC is off. CUDA-free is retained as telemetry and has no
independent 8/40 GB gate. Each node may begin with at most 128 MiB of swap, and training may add
zero bytes above that node's baseline. It checkpoints at 96 and 192.

The driver verifies the CPT-v7 1199-tensor base, corpus, full production-plan hash, topology,
partial batches, late maximum shapes, useful throughput, the full-adapter delta from step 0 to
step 192, zero swap growth, zero allocator retries/OOMs, full-adapter cross-rank equality and
checkpoint fingerprints, per-step board/SoC temperature and actual graphics clock, and all four
checkpoint files after a reboot. CUDA-free and the 250-step memory projection are retained as
diagnostics, not hard gates. The driver retains logs and receipts off-cluster, then removes the
measurement outputs from every Spark. It cannot launch full SFT.

Both measured and corpus-projected point useful-input throughput must be at least 1000 all-reduced
input tokens/s. Their one-sided 95% block-bootstrap lower bounds are calculated and retained as
diagnostics, not silently promoted into a stronger hard gate. A measured-only or projected-only
pass is not authorization.

The 8/40 GB and bounded-baseline swap policy is the 2026-07-30 Grok + ChatGPT majority ruling.
The raw consultation-receipt identities are
`332135825e56f31289b2902fa883e811ae4adeeed87a3651609cf223b7bbfe75` (Grok) and
`523cecc1ba30793a8f5cf5c55f22d68380bee500e12425f404b4ef4bcd68fdd6` (ChatGPT).
No adapter delta band was supplied, so the qualifier requires a finite non-zero full-adapter
delta receipt but does not invent a numerical acceptance threshold.

The exact candidate `ddp-r16-family-a-20260730T174711Z` passed on 2026-07-30 from commit
`45a967b6e70f579dae62744e58aa2013a4e8c615`: measured/projected useful throughput was
1032.0168/1027.0452 tok/s, the full-adapter mean absolute delta was 0.0011275692, the no-release
minimum was 8,178,622,464 bytes, and the post-release minimum was 53,476,806,656 bytes. It had
zero swap growth, allocator-retry growth, OOM growth, and memory exits. The 250-step
system-headroom projection was false and remains visible as the diagnostic-only field the Family
specified. The immutable public receipt is
`careers-qwen/receipts/sft_ddp_r16_qualification.manifest.json`.

After a candidate pass, bind `run_stage2_sft_ddp_till_done.sh` to the exact qualification receipt
before use.
The production sequence is 0→50, reboot and verify, then 50→300 as one 250-step session. Only a
passing checkpoint-300 receipt authorizes continuation to step 979. `run_stage2_sft_and_bake_production.sh`
then runs the validated DDP bake; it never uses `BAKE_TO_HF`.

- The adapter is a delta against this exact CPT-v7 base. Never apply it to a different or
  previously merged base.
- The canonical data path is one declared pass with exactly-once supervised-window coverage.
- Gated-DeltaNet samples remain unpacked until recurrent-state separation is proven.
- Private corpus, environment, logs, checkpoints, and receipts stay outside Git. Production code,
  contracts, and runbook changes stay current in the repository.

### Legacy FSDP only: extract the adapter offline
`careers-qwen/extract_lora_adapter_offline.py --ckpt <ckpt>/final --out <adapter dir>`
Single process, no cluster, no collective. Safe because the FSDP policy (`Qwen3_5DecoderLayer`,
no LoRA wrap) leaves adapter tensors unsharded — the script verifies `chunks=1` and refuses
otherwise. Filter is `model.*` prefix AND has-size: a checkpoint has ~2816 keys matching `lora_`,
of which only ~704 are weights; the rest is optimizer state including non-tensor entries.

**The in-trainer `save_lora_only_fsdp` deadlocks against a multimodal base.** It calls
`get_model_state_dict(full_state_dict=True, ignore_frozen_params=True)` — a full-model collective
over 1199 tensors, 348 of them frozen and unwrapped. All four ranks sat in it at ~10W for 45 min
on 2026-07-27. That gather exists to materialise a *merged* model; when the deliverable is an
adapter it is work nobody needs. **Still unfixed in the trainer** — this is a route around it.

### THIS MODEL IS HYBRID-ATTENTION: LoRA must be MERGED, never served dynamically
vLLM's dynamic LoRA path has **no kernel for linear-attention modules**, and it fails loud:
`ValueError: ...linear_attn.in_proj_qkv.lora_A... is unsupported LoRA weight` in `init_static_loras`.

| module family | tensors | vLLM dynamic LoRA |
|---|---:|---|
| `mlp` gate/up/down | 384 | serves |
| `self_attn` q/k/v/o | 128 | serves |
| `linear_attn` in_proj_qkv/out_proj | 192 | **cannot serve** |

48 linear-attention layers against 16 full-attention — the 3:1 hybrid — so ~27% of any adapter
targets modules the dynamic path cannot mount. **No retrain is needed**: merged weights are just
weights. `bake_lora_nopeft.py` applies `W_new = W_base + (alpha/r)·B@A` into the shards and
explicitly handles `.linear_attn.in_proj_qkv.` and `.linear_attn.out_proj.` — it was written for
this architecture, preserves structure, and outputs a 1199-tensor servable model with **no visual
weight loss**. That script is why LoRA never hit the CPT servability bug.

## MULTI-EPOCH STRICT SFT — use the multiplier, never bypass strict mode

For legacy one-sample-per-row modules already using `EXACT_SFT_EPOCH=1`, set `EPOCHS=<n>`
(default 1). `expected_steps = EPOCHS * steps_per_epoch`, and all four strict guards keep working.
Turning strict mode off merely to get multiple epochs is a bypass: it forfeits the
coverage-equality check, `TOTAL_STEPS==SESSION_LIMIT==expected`, `SAVE_EVERY>expected`, and
`resume_step==0`.

The windowed Stage-2 campaign above is a different contract: one natural cycle, DCP session
continuation, tokenizer-sample and horizon receipts, and no `EXACT_SFT_EPOCH` mode.

```
EPOCHS=2 EXPECTED_REAL_SAMPLES=<rows> TOTAL_STEPS=<2*spe> SESSION_LIMIT=<2*spe> SAVE_EVERY=999
```
Worked example (module 5): 60 rows / gb 4 → 15 steps/epoch → `EPOCHS=2 TOTAL_STEPS=30`.
Verify in the log: `DOSE PASS: optimizer_steps=30 real=120 padding=0` — `real` is rows × EPOCHS.

## LAUNCH IN DETACHED TMUX, NEVER `nohup … &` OVER SSH

`run_4node_27b_cpt.sh:91` uses `tmux new-session -d`. That is not stylistic. A `nohup`'d process
launched through ssh dies of **SIGTERM** when systemd tears down the user slice after the last
session for that user closes — which happens the moment you stop polling. Killed a run at step
20/30 on 2026-07-27, with `SAVE_EVERY` final-only, so nothing was saved.
```
ssh spark@$n "cd $RR && tmux new-session -d -s <name> \"env <ENV> bash <launcher> $rank > r$rank.log 2>&1\""
```
Rank 0 first — it binds `:29500` before the workers dial in.

## TOOL-CALL ROWS — two things that had never once run before 2026-07-27

Modules 1–4 were prose. Module 5 was the first corpus with real `tool_calls`, and it exposed both:

- **`arguments` must be a MAPPING, not a JSON string.** The template resolves a mapping directly and
  falls back to `| fromjson` for the OpenAI wire form — and `fromjson` does not exist in
  transformers 5.3.0 / jinja2 3.1.6. A string dies at tokenization on row 1.
- **A saved adapter must NOT carry the adapter name in its keys.** `set_peft_model_state_dict`
  inserts `default` itself; a file already containing `...lora_A.default.weight` loads as
  `...default.default.weight`, and every tensor comes back simultaneously missing AND unexpected.
  This hid for months because the only consumer was `bake_lora_nopeft.py` (raw tensor math, no
  module-path resolution): **merging worked, resuming did not.** It surfaces the first time a module
  is genuinely cumulative.

## MERGE TARGET IS ALWAYS THE CPT BASE — NEVER A PREVIOUSLY-MERGED MODEL

A resumed adapter encodes the **cumulative** delta from the frozen base, not a delta on top of the
previous module. `LORA_MODE` freezes the base; the only tensors that train are the adapter's A/B,
and `RESUME_DELTA` initialises them from module N-1. So module N's adapter *is* module N-1's
adapter continued.

```
base + (m4+m5)                merge module5 into cpt_refresh_v3_servable    CORRECT
base + m4 + (m4+m5)           merge module5 into module4_merged             m4 APPLIED TWICE
```

**A merged model is a SERVING artifact only.** It is never a merge target and never a training base.

**Why this is dangerous rather than merely wrong:** the corrupted artifact does not crash. It merges
cleanly, carries the right 1199 tensors, has matching key sets, and its weights genuinely *differ
from the reference* — which is TRUE in both the correct and the doubled case, so a structural
acceptance gate cannot separate them. It serves, quietly degraded. Caught 2026-07-27 by infra
inspecting their own gate's blind spot **before** building the artifact.

To check a merge target: read `adapter_config.json → base_model_name_or_path`. Merge into THAT.

## VERIFY A CUMULATIVE ADAPTER BY DIFFING IT AGAINST ITS PARENT

`lora_B` non-zero proves *trained*. It does NOT prove *this run trained it* — a resumed adapter
inherits a non-zero B from its parent. The check that means something:

```
CHANGED vs module<N-1>: 704/704 tensors   IDENTICAL: 0   mean|delta| 1.21e-04
```
Identical tensors would mean the run resumed and then contributed nothing. Diff the child against
the parent, per-tensor, and state the count both ways.

## DIAGNOSING A RUN THAT LOOKS DEAD — check your instrument before you conclude

**`ps -eo comm` CANNOT SEE THE TRAINER.** The launcher is `python3 -m torch.distributed.run`, so its
`comm` is `python3` — a pattern of `torchrun` or `pt_main_thread` returns **0 for four healthy
nodes**. Use the command line, not the comm:
```bash
ps -eo args | grep -c '[t]rain_fsdp_dense_9b.py'      # 2 per node when healthy
```
A probe that cannot match a live process returns the same `0` whether the run died or the pattern is
wrong, and those are indistinguishable from the output alone. On 2026-07-27 this nearly caused a
working run to be killed and restarted at step 6.

**`earlyoom` IS A RED HERRING ON THESE NODES.** Its log reads
`low memory! at or below SIGTERM limits`, which looks exactly like a kill. The next two lines are
`find_largest_process: selected myself` and `Could not find a process to kill.` — it cannot see the
training processes at all (proc hidepid). It logged **1825 low-memory events between 07-20 and 07-26,
during the ep3 campaign that COMPLETED**. Its presence proves nothing about a death.

**A CPT step legitimately runs at ~118–119G of 122.5G with ~7G free.** That is the normal envelope,
not distress — `free` holding flat across steps is health. A leak trending to death looks different:
monotonically falling free memory with rising `frag`. Distinguish them before intervening.

**Silent death with no traceback** means SIGKILL or a hard node fault, not a Python error. Check, in
order: `ps -eo args` (is it actually dead?), `sudo dmesg -T | grep -i oom` (kernel OOM leaves a
record), `sudo journalctl --since` around the timestamp, then the WORKER rank logs — rank 0 often
dies silently while a worker holds the real error.

**TOPOLOGY MUST BE FORWARDED TO THE NODES.** `fleet.env` is gitignored and is NOT on the Sparks, but
`launch_cpt_qwen36_27b_fsdp.sh:399` dereferences `SPARK_RAIL_MASTER` under `set -u`. `RUN_ENV` in the
launchers is an **allowlist** — exporting a var on the Mira side does not reach the node unless it is
named there. Symptom: every rank dies instantly *after* `corpus schema OK` prints, while the
orchestrator reports `launched rank 0..3` and `alive=4/4`. Fixed in both `run_4node_27b_cpt.sh` and
`bake_27b.sh` on 2026-07-27.

## NEVER

- Never call the inner launcher with hand-rolled env when a wrapper exists.
- Never trust a description of an artifact. Open it. A manifest said "real tool-call trajectories"
  for a file with zero tool calls; a launcher described substrate-physics content the corpus did
  not contain; a truncated log was read as the run it replaced.
- Never assert a row is defective without naming the **surface** it was measured against. There are
  at least three (`ui_action` ref-based, `worker_ui.py` ref-based, `act.py` name-based).
  "Invalid against ui_action" is a fact; "invalid" is not.
- Never report a run healthy from the loss curve. `SR-DELTA` vs ULP is the oracle.
