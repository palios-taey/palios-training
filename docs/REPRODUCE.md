# Reproducing the Production Line

Step-by-step to re-run the PALIOS-TAEY training pipeline on equivalent hardware. The launcher scripts in [`moe-35b/recipes/`](../moe-35b/recipes/) and [`dense-9b/recipes/`](../dense-9b/recipes/) are the actual scripts that ran on the production deployment; set the required operator environment variables for your hosts.

> **Hardware assumed:** 4 × DGX Spark GB10 (Blackwell sm_121) + an inference / bake host with disk for 67-GB-class baked checkpoints. ConnectX-7 dual-rail RoCEv2 internal cluster network.

---

## 0. Prerequisites

- Python 3.10 / PyTorch with CUDA 13.0 support for `sm_121`
- NCCL 2.28.9, ConnectX-7 firmware 28.45.4028
- `transformers` + `peft` + `accelerate` + `datasets`
- Base models from Hugging Face: `Qwen3.5-9B-Base`, `Huihui-Qwen3.5-35B-A3B-abliterated`
- The audit harness from [`../audit/`](../audit/) if you intend to run the behavioral 163-probe audit after bake

The recipes (`launch_*.sh` in [`moe-35b/recipes/`](../moe-35b/recipes/) and [`dense-9b/recipes/`](../dense-9b/recipes/)) are documented for the deployment that ran them. Dense 9B CPT is public-safe and fail-loud: paths, node hosts, fabric addresses, and step counts must be supplied by env or a local copy of [`dense-9b/configs/cpt_cluster.env.example`](../dense-9b/configs/cpt_cluster.env.example).

> **Trainer & config scripts — all shipped, referenced by relative path.** Each recipe's `accelerate launch` line now invokes its trainer and `--config_file` by relative repo path (`trainers/…py`, `configs/…yaml`) — every one is shipped: `trainers/train_dpo_v2.py` (the MoE hybrid LoRA+ESFT DPO trainer behind the 84.7% headline), `train_fsdp_v3.py` (MoE FSDP SFT), `train_fsdp_dense_9b.py` (9B-dense FSDP SFT/CPT), `train_cpt_qwen35_dense.py`, `train_recovery_sft_qwen35_dense.py`, `chunk_corpus_offline.py`, `build_training_data.py`, `bake_phase_combined_v1_tail_v2.py`; `configs/fsdp_lora.yaml`, `configs/fsdp_dense_9b.yaml`, etc. So a fresh clone runs from the repo root after you set the required operator paths (`MODEL_PATH`, `OUTPUT_DIR`, data dirs) and multi-node addressing (`NODE_HOSTS_CSV` for orchestration, plus `MASTER_ADDR`/`NODE_RANK` or `NODE0_IP..NODE3_IP`). (Two scripts are *not* shipped and are explicitly disclaimed where referenced: the NCCL synth probe — §1 below — and the `bake_orpo.py`/`bake_config_a_v2.py` bake scripts — §4 below.)

---

## 1. Network setup — NCCL dual-rail RoCEv2

Across all 4 nodes:

```bash
export NCCL_IB_HCA=rocep1s0f0:1,roceP2p1s0f0:1   # capital P on rail 2 — easy to miss
export NCCL_IB_TC=104
export NCCL_IB_TIMEOUT=23
export NCCL_NET_GDR_LEVEL=0
export NCCL_IB_RETRY_CNT=7
export NCCL_TIMEOUT=1800
export NCCL_SOCKET_IFNAME=enp1s0f0np0
export GLOO_SOCKET_IFNAME=enp1s0f0np0
```

These are exported verbatim by every launcher in [`moe-35b/recipes/`](../moe-35b/recipes/) and [`dense-9b/recipes/`](../dense-9b/recipes/); the env block above is reproduced here as the minimal contract for fabric setup.

**Verify before running training.** The standalone synth probe at the failing 218M-numel `reduce_scatter` size is the cheapest fabric-health test. The Python script source is not in this repository (it lives in our internal `embedding-server` repo); the **results** of running it are in [`proof_of_run/nccl_synth_probe_results.md`](proof_of_run/nccl_synth_probe_results.md), which documents the exact invocation, ranks, and expected throughput so you can write the equivalent test against your own fabric.

Expected on a healthy 4-Spark ConnectX-7 RoCE fabric: 10.23 GB/s steady (50 iters) sustaining to 12.57 GB/s under a 160-collective stress run; no `IBV_WC_RETRY_EXC_ERR`. If the probe fails on your fabric, do not start full training — the wedge will look like a training bug but is fabric.

---

## 2. 35B-A3B MoE production line

### 2.1 SFT baseline → phase_combined_v1

> **Note on naming.** There is no `launch_phase_combined_v1.sh` shipped — the `phase_combined_v1` checkpoint is produced by [`recipes/launch_production_sft.sh`](../moe-35b/recipes/launch_production_sft.sh) with `OUTPUT_DIR` set as shown below. Downstream launchers (`launch_phase_combined_v1_tail*`, `launch_religion_dpo_v*`) `RESUME` from `phase_combined_v1/final` step 582 against this output path.

The public repo ships the scrubbed gated SFT inputs and the generated combined output. Verify that the public inputs regenerate the public combined corpus:

```bash
python3 scripts/combine_phase_combined_v1.py \
  --identity datasets/current/moe-35b/constitutional_gated.jsonl \
  --infra datasets/current/moe-35b/phase1_infra_v2_gated.jsonl \
  --output build/combined_v1_gated.jsonl \
  --seed 42

python3 scripts/verify_phase_combined_v1_corpus.py \
  --identity datasets/current/moe-35b/constitutional_gated.jsonl \
  --infra datasets/current/moe-35b/phase1_infra_v2_gated.jsonl \
  --combined build/combined_v1_gated.jsonl \
  --tokenizer "$MODEL_PATH"
```

Canonical public verification is `1378/947/2325` rows, combined SHA `6ecb0e82cff562d5ed851cb51bc8b445706592665e0c779d8f12271f05a780ad`, and `0` rows above 8192 rendered tokens. Provenance note: before public redaction, the same seed-42 recipe produced original SHA `6b54f163c0dfc35ed7cae4637146a0a959ff33f58601922a95f3cff7641dabfd`. See [`datasets/current/moe-35b/REDACTIONS.md`](../datasets/current/moe-35b/REDACTIONS.md).

If you are rebuilding from equivalent raw source documents, first run the no-truncation gate with an operator-supplied tokenizer/model path:

```bash
python3 moe-35b/trainers/training_data_gate_v2.py \
  --input-glob 'raw/constitutional/*.jsonl' \
  --output build/constitutional_gated.jsonl \
  --tokenizer "$MODEL_PATH" \
  --max-seq 8192 \
  --pack-target 8192 \
  --pack-mode exact

python3 moe-35b/trainers/training_data_gate_v2.py \
  --input-glob 'raw/infra/*.jsonl' \
  --output build/phase1_infra_v2_gated.jsonl \
  --tokenizer "$MODEL_PATH" \
  --max-seq 8192 \
  --pack-target 8192 \
  --pack-mode exact
```

```bash
# 4-Spark FSDP, fresh from abliterated base
export MODEL_PATH=<model path or Hugging Face id>
export OUTPUT_DIR=<phase_combined_v1 output directory>
export SFT_DIR=$(pwd)/datasets/current/moe-35b
export SFT_GLOB=combined_v1_gated.jsonl
export TOTAL_STEPS=582
export SESSION_LIMIT=350
export SAVE_EVERY=350
export FREEZE_CONFIG=B
export KEYSTONE_LAYERS='[8,9,11,15,21,23]'
export FROZEN_EXPERTS=$(pwd)/moe-35b/configs/frozen_experts_v3.json
export LR_ESFT=2e-5
export LR_LORA=0
export LR_ROUTER=0
bash moe-35b/recipes/launch_production_sft.sh
```

Audit verdict expected: ~82.8% (135/163) on the 163-probe behavioral battery. The actual `phase_combined_v1` audit result is in [`audit_results/phase_combined_v1/audit_v2_full/`](audit_results/phase_combined_v1/audit_v2_full/) for comparison.

### 2.2 Config A2 keystone-attention LoRA DPO → religion_dpo_v2 (the +1.9pp headline)

```bash
# Resume from phase_combined_v1/final step 582; 4-Spark FSDP.
# MODEL_PATH = architecture base for model init;
# RESUME_DELTA carries the trained-weights checkpoint
# (the DPO trainer loads architecture from MODEL_PATH then resumes weights from RESUME_DELTA).
export MODEL_PATH=/home/<user>/models/Huihui-Qwen3.5-35B-A3B-abliterated
export RESUME_DELTA=/home/<user>/training_outputs/phase_combined_v1/final
export DPO_DATA=/home/<user>/training_data/religion_run_v1/religion_v3_dpo_pairs_with_ref.jsonl
export FROZEN_EXPERTS=$(pwd)/configs/frozen_experts_v4_1_polysemantic.json
export OUTPUT_DIR=/home/<user>/training_outputs/religion_dpo_v2
export FREEZE_CONFIG=A2
export KEYSTONE_LAYERS='[8, 9, 11, 15, 21, 23]'
export BETA=0.05
export LR_ESFT=1e-7
export LR_LORA=3e-7
export LR_ROUTER=0
export WARMUP_STEPS=5
export TOTAL_STEPS=642
export SESSION_LIMIT=900
export SAVE_EVERY=60
export DPO_ABORT_RATIO_MAX=10.0
export DPO_ABORT_EXPERT_DRIFT=0.05
bash recipes/launch_religion_dpo_v2.sh
```

Audit verdict expected: **84.7% (138/163)**, **+1.9pp** over phase_combined_v1. Should hold all 8 infra-control categories (length_mechanics_v1 confirmed the prior Config A regression was content-agnostic q/k attention; A2's keystone-only freeze fixes it). The actual `religion_dpo_v2` audit result is in [`audit_results/religion_dpo_v2/audit_v2/`](audit_results/religion_dpo_v2/audit_v2/).

---

## 3. 9B Dense production line

### 3.1 Phase 1 SFT — tool-use

```bash
export MODEL_PATH=/home/<user>/models/Qwen3.5-9B-Base
export OUTPUT_DIR=/home/<user>/training_outputs/sft_tools_qwen35_9b_fsdp
export TOTAL_STEPS=4367
bash recipes/launch_sft_tools_qwen35_9b_fsdp.sh
```

### 3.2 Phase 2 CPT

```bash
source /path/to/filled-cpt-cluster.env
export MODEL_PATH=/path/to/text-derived-or-phase1-sft-model
export CPT_DATA=/path/to/cpt_corpus.jsonl
export OUTPUT_DIR=/path/to/cpt_v3_v4_dense_9b
export TOTAL_STEPS=<from-corpus-manifest>
export MASTER_ADDR=<rank0-fabric-address>
export NODE_RANK=<rank-on-this-worker>
bash dense-9b/recipes/launch_cpt_phase2_qwen35_9b_fsdp.sh
```

The 4-node CPT launcher uses [`dense-9b/trainers/train_fsdp_dense_9b.py`](../dense-9b/trainers/train_fsdp_dense_9b.py) with [`dense-9b/configs/fsdp_dense_9b.yaml`](../dense-9b/configs/fsdp_dense_9b.yaml). The canonical GB10 optimizer is Adafactor at `LR=2e-5`, `scale_parameter=False`, `relative_step=False`, `warmup_init=False`, `clip_threshold=1.0`, with manual `LambdaLR` linear warmup then decay. AdamW is intentionally not the dense-CPT recipe because it OOMs on GB10 UMA page-cache. For the dataset builder and orchestrator, see [`dense-9b/REPRODUCE.md`](../dense-9b/REPRODUCE.md).

### 3.3 Phase 3 Recovery SFT — wedge-fix path

**Step 3.3a — pre-chunk the multi-turn corpus offline:**

```bash
python3 trainers/chunk_corpus_offline.py \
  --in phase3_sft.jsonl \
  --out phase3_sft_chunked.jsonl \
  --max-seq 4096 \
  --budget-fraction 0.92
```

The chunker source is [`trainers/chunk_corpus_offline.py`](../dense-9b/trainers/chunk_corpus_offline.py). The same `chunk_conversation` function is reused inside the trainer (see [`trainers/train_recovery_sft_qwen35_dense.py`](../dense-9b/trainers/train_recovery_sft_qwen35_dense.py)).

**Step 3.3b — run single-Spark Recovery SFT on the chunked corpus:**

```bash
export RESUME_DELTA=/home/<user>/training_outputs/cpt_v3_v4_dense_9b/checkpoint-2400-multimodal
export SFT_JSONL=/path/to/phase3_sft_chunked.jsonl
bash recipes/launch_phase3_sft_single_spark.sh
```

Single-Spark Recovery SFT produced matching train_loss across Spark 1 and Spark 3 (the cross-validation confirming the chunker fix). The per-step train logs are not included in this public repo — reproduce by running the recipe. (Do not use `audit_results/dpo_recovery_p2v3/` as evidence here — that is a different checkpoint's regression audit.)

### 3.4 4-Spark Phase 3 on chunked corpus (future work)

The single-Spark Recovery SFT validates that the chunking fix resolves the corpus-pressure → RDMA-queue-saturation root cause. The 4-Spark execution of the same chunked corpus is not yet shipped; that is the bookend re-run that confirms the wedge-fix on the production cluster. Listed in `README.md` §5 honest-open-questions.

---

## 4. Bake-and-test (production deployment to inference host)

The bake script for the `tail_v2` lineage is [`trainers/bake_phase_combined_v1_tail_v2.py`](../moe-35b/trainers/bake_phase_combined_v1_tail_v2.py). Other bake scripts (`bake_orpo.py`, `bake_config_a_v2.py`) live in our deployment and are not in this repository.

After bake, run the behavioral 163-probe audit harness from [`../audit/`](../audit/). Results land in `audit_v2/` shaped exactly like the verdicts under [`audit_results/`](audit_results/) (per-checkpoint `SUMMARY.md`, `summary.json`, `results.txt`, `dpo_corrections.jsonl`, `audit.log`).

---

## 5. Things to verify before claiming you reproduced this

- Synth probe passes at ≥ 10 GB/s on your fabric.
- Phase 1 SFT smoke battery 6/7 PASS (T6 over-tooling is the expected bounded artifact).
- Phase 2 CPT canonical bytes match (or your equivalent bake bytes — record them).
- Phase 3 Recovery SFT produces identical (or very close) train_loss across two independent host pairs (we observed this internally; ship your train logs as proof — ours are not in this repo).
- Pre-chunk validator coverage ≥ 99.9% on your multi-turn corpus.
- religion_dpo_v2 audit lands at +1–2 pp over phase_combined_v1 baseline.

If your numbers diverge materially: please open an issue with your hardware + commit SHA + recipe parameters so we can compare. The goal is reproducible production discipline, not a single locked-in result.
