# Qwen3.8-27B CPT and bake: the configuration surface

Part 2 of 2. Scope: the non-public configuration inputs the training and bake paths read, and what a
reader holding only the public repository can and cannot determine. The measured run timeline is
Part 1 and is outside this scope.

`Observed` = read from a repository object or a live command. `Inferred` = arithmetic over observed
values. `Unknown` = not determined from the evidence available.

**Disclosure rule applied to this document.** Variable names, semantics, call sites, unset-behaviour
and measured properties are given verbatim, because they are the diagnostic content. Host addresses
and any credential-bearing value are represented by role-stating placeholders
(`<SPARK_NODE_N_MGMT>`, `<THOR1_MGMT>`, `<CONVERT_HOST_SSH>`). No octet, key, or token appears here.
Operator filesystem paths appear as `$SPARK_HOME` / `$OPERATOR_HOME`, matching the repository after
commit `c89fce5`.

---

## 1. The declared surface and the read surface are different sizes

`fleet.env` is gitignored. `fleet.env.example` is the public declaration of it.

| measure | count | register |
|---|---:|---|
| variables declared in `fleet.env.example` | 23 | Observed |
| distinct variables read by `dense-9b/recipes/run_4node_27b_cpt.sh` | 73 | Observed |
| …of those, declared in `fleet.env.example` | **4** (`SPARK_HOME`, `SPARK_MASTER`, `SPARK_MGMT_IPS`, `SPARK_RAIL_MASTER`) | Observed |
| …of those, declared nowhere public | **69** | Observed |
| distinct variables read by `careers-qwen/post_cpt_pipeline.sh` | 58 | Observed |
| …of those, declared in `fleet.env.example` | 9 | Observed |

**Observed:** the 69 undeclared CPT variables include every one that sets the run's shape —
`MODEL_PATH`, `TOTAL_STEPS`, `EPOCHS`, `LR`, `WARMUP_STEPS`, `SESSION_LIMIT`, `MAX_SEQ`,
`BATCH_SIZE_PER_RANK`, `TOKEN_BUDGET_PER_STEP`, `CPT_SHORT_BATCH`, `CPT_MID_BATCH`,
`CPT_LONG_BATCH`, `CPT_PACKED`, `HORIZON_PARTIAL`, `SAVE_EVERY`, `OUTPUT_DIR`, `FP32_MASTER`,
`ADAFACTOR_EPS1`, `ADAFACTOR_ALPHA_MODE`, `CLOCK_CAP`.

---

## 2. Unset behaviour: the two surfaces behave oppositely

Counted mechanically over `${VAR:?…}` (hard abort) and `${VAR:-…}` (silent default).

| surface | hard-abort if unset | silent default | register |
|---|---:|---:|---|
| `run_4node_27b_cpt.sh` (CPT) | **0** | 47 | Observed |
| `post_cpt_pipeline.sh` (bake) | **10** | 10 | Observed |

**Observed:** the CPT launcher has no required variable. Of its 47 defaults, **42 default to the
empty string** — among them `MODEL_PATH`, `LR`, `EPOCHS`, `WARMUP_STEPS`, `OUTPUT_DIR`,
`HORIZON_PARTIAL`, `FP32_MASTER`, `LORA_MODE`, `LORA_R`, `RESUME_MODEL_ONLY`, `BAKE_TO_HF`.
Five default to a value: `CLOCK_CAP=2000`, `GATE_PREFLIGHT=1`, `GEMM_PREFLIGHT_ONLY=0`,
`GEMM_PREFLIGHT_MIN_PEER_RATIO=0.80`, and `CPT_DATA` to an absolute node corpus path.

**Observed:** the bake pipeline aborts with a stated message when any of `ARTIFACT_STORE`, `CKPT`,
`CONVERT_GRAFT_BASE`, `CONVERT_IMAGE`, `CONVERT_ROOT`, `CONVERT_SSH`, `DCP_DIR`, `SPARK_HOME`,
`SPARK_MASTER` or `SPARK_MGMT_IPS` is unset (`careers-qwen/post_cpt_pipeline.sh:20` and the
`: "${VAR:?…}"` guards around it).

**Observed:** the same variable name can be both, because the bake pipeline layers them — e.g.
`ARTIFACT_STORE=${ARTIFACT_STORE:-${POST_CPT_ARTIFACT_STORE:-}}` at `:13`, then
`: "${ARTIFACT_STORE:?…}"` at `:20`. The default supplies the `fleet.env` value; the guard rejects
the case where neither exists.

---

## 3. Load-bearing variables, with call site and unset behaviour

| variable | read at | controls | unset |
|---|---|---|---|
| `SPARK_HOME` | `run_4node_27b_cpt.sh`, `post_cpt_pipeline.sh` | node-side root for models, corpora, outputs | bake aborts; CPT does not |
| `SPARK_MASTER`, `SPARK_MGMT_IPS`, `SPARK_RAIL_MASTER` | both surfaces | rank-0 election and the two fabrics (`<SPARK_NODE_N_MGMT>` / `<SPARK_NODE_N_RAIL>`) | bake aborts; CPT does not |
| `MODEL_PATH` | `run_4node_27b_cpt.sh` | the base the trainer loads | **empty string** |
| `TOTAL_STEPS`, `EPOCHS`, `LR`, `WARMUP_STEPS` | `run_4node_27b_cpt.sh` | the entire schedule | **empty string** |
| `HORIZON_PARTIAL` | `run_4node_27b_cpt.sh` | suppresses the fail-closed horizon gate | **empty string** (gate active) |
| `SESSION_LIMIT` | `run_4node_27b_cpt.sh` | steps before `FRAGMENTATION EXIT` | **empty string** |
| `FP32_MASTER`, `ADAFACTOR_EPS1`, `ADAFACTOR_ALPHA_MODE` | `run_4node_27b_cpt.sh` | optimizer master-weight and normalisation path | **empty string** |
| `CLOCK_CAP` | `run_4node_27b_cpt.sh` | `nvidia-smi -lgc` cap, which PERSISTS across jobs | defaults `2000` |
| `ARTIFACT_STORE` / `POST_CPT_ARTIFACT_STORE` | `post_cpt_pipeline.sh:13,20,97,98` | where Artifact B and the training base are staged before conversion | aborts |
| `CONVERT_SSH` / `POST_CPT_CONVERT_SSH` | `post_cpt_pipeline.sh` | the off-cluster conversion host (`<CONVERT_HOST_SSH>`) | aborts |
| `CONVERT_IMAGE` / `POST_CPT_CONVERT_IMAGE` | `post_cpt_pipeline.sh` | the pinned conversion container digest | aborts |
| `CONVERT_GRAFT_BASE` / `POST_CPT_GRAFT_BASE` | `post_cpt_pipeline.sh` | the 1199-tensor donor supplying vision tower, mtp head and config | aborts |
| `SANCTION` / `POST_CPT_SANCTION` | `post_cpt_pipeline.sh` | the sanctioning task reference | defaults to a task id |

**Measured properties that are not in the repository at all:**

- **Observed:** the artifact store named by `POST_CPT_ARTIFACT_STORE` is a USB-attached volume on the
  controller, measured **19 MB/s**.
- **Observed:** the Spark-to-Spark rail measures **666 MB/s**; the management LAN measures
  **96–112 MB/s** (112 MB/s sustained for the 52 GB Spark→Thor transfer on 2026-08-18).
- **Observed:** the controller has no interface on the rail subnet, so controller-initiated transfers
  cannot use the 666 MB/s path.

---

## 4. What the public repository alone determines

For each GOLDEN_PATH surface: holding only the public repo, is the behaviour DETERMINED?

| question | surface | determined? | decided at |
|---|---|---|---|
| Which corpus is packed? | `corpus_pack` | **No** — `CPT_DATA` default is an absolute node path; overrides are undeclared | `run_4node_27b_cpt.sh` default |
| Which base model is trained? | `cpt_27b_4node` | **No** — `MODEL_PATH`, empty default | `run_4node_27b_cpt.sh` |
| How many steps, at what LR? | `cpt_27b_4node` | **No** — `TOTAL_STEPS`/`EPOCHS`/`LR`, empty defaults | `run_4node_27b_cpt.sh` |
| Is the horizon gate active? | `cpt_27b_4node` | **Yes** — active unless `HORIZON_PARTIAL` is set | `run_4node_27b_cpt.sh` |
| What GPU clock is applied? | `cpt_27b_4node` | **Yes**, as a default — `CLOCK_CAP=2000`, and it persists across jobs | `run_4node_27b_cpt.sh` |
| Which checkpoint is baked? | `bake_export` | **Yes** — `final/` when present, else highest `checkpoint-N` | `post_cpt_pipeline.sh` (`6b63716`) |
| Where do artifacts stage? | `bake_export` | **No** — `ARTIFACT_STORE`, aborts if unset | `post_cpt_pipeline.sh:13,20` |
| Which host converts? | `bake_export` | **No** — `CONVERT_SSH`, aborts if unset | `post_cpt_pipeline.sh` |
| **Which model is grafted onto?** | `bake_export` | **No** — `CONVERT_GRAFT_BASE`, aborts if unset | `post_cpt_pipeline.sh` |
| Are the tensor counts gated? | `bake_export` | **Yes** — 851 required, 1199 emitted | `post_cpt_pipeline.sh:425` |
| Is the weight-diff band enforced? | `bake_export` | **Yes** — `5e-05 .. 8e-04` | `careers-qwen/measure_cpt_delta.py` |

**Observed:** the gates are determined from the public repo. The inputs those gates run against are
not. A reader can verify that a bake refuses a base that is not 851 tensors, and cannot determine
which base was supplied.

---

## 5. Where configuration and documented process disagree

Reported as disagreements. Which side is correct is outside this document's scope.

**5.1 — the graft donor.**
`fleet.env` sets `POST_CPT_GRAFT_BASE` to one fixed artifact path.
`careers-qwen/RUNBOOK_CPT_SFT_BAKE.md` section 4 gives the donor as the run's own source model.
Measured (`4fcf70a`), sha256 over raw tensor bytes:

| tensor | run's own source | the pinned artifact |
|---|---|---|
| `model.visual.blocks.0.attn.proj.bias` | `ea9264ae519e8a02` | `54d0cecb31e424a8` |
| `model.visual.blocks.15.attn.proj.bias` | `848c549105fae0b4` | `03a381eecb0992cc` |
| `model.visual.patch_embed.proj.bias` | `8acebc63dfde0d75` | `f5250290478a5bfc` |

**Observed:** 3 of 3 sampled vision tensors differ. Both candidates carry 1199 tensors and both are
`Qwen3_5ForConditionalGeneration` / `model_type qwen3_5`. **Observed:** the graft replaces all 851
language tensors, so the donor contributes the vision tower, the mtp head and the config only.

**5.2 — artifact routing.**
`post_cpt_pipeline.sh:20` requires `ARTIFACT_STORE` and `:97-98` derive `LOCAL_ARTIFACT` and
`LOCAL_BASE` beneath it, so artifacts stage on controller storage before conversion.
`careers-qwen/RUNBOOK_CPT_SFT_BAKE.md` section 3a states bake node-local, then Thors, then Expansion
last. **Observed:** on 2026-08-18 the conversion ran node-local (107 s) and the graft node-local
(50 s), and the servable moved Spark→Thor directly at 112 MB/s; the pipeline's own path was not used
for those stages.

**5.3 — a search for other instances of the same shape.**
`CPT_DATA` is the one other variable found whose default is a fixed absolute artifact path rather
than a per-run derivation. No further instances were found in the two surfaces examined.
**Unknown:** the SFT surfaces were not enumerated for this document.

---

## 6. Residual Unknowns

- **Unknown:** whether any consumer outside these two surfaces and `scripts/taey-train` reads
  `fleet.env`. Enumeration covered the three GOLDEN_PATH surfaces only.
- **Unknown:** the values held by the 69 undeclared CPT variables during any historical run.
  Per README rule 5 they existed only in the live process environment, and the processes have exited.
- **Unknown:** whether `fleet.env.example` was ever intended to enumerate the full read surface, or
  only the topology subset.
