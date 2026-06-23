# Reproduce Dense 9B Training

This directory is self-contained for the dense Qwen3.5-9B line: launchers live
in `recipes/`, Accelerate/FSDP config lives in `configs/`, and trainers plus
dataset builders live in `trainers/`.

## Public-Safe Configuration

No production paths, management hosts, fabric IPs, or operator usernames are
embedded in the CPT launcher. Fill a local copy of
`configs/cpt_cluster.env.example`, source it, then run the recipe.

Required CPT launch inputs:

- `MODEL_PATH`: text-derived or Phase-1-SFT model directory.
- `CPT_DATA`: prebuilt JSONL corpus with raw causal-LM text rows.
- `OUTPUT_DIR`: checkpoint/output directory.
- `TOTAL_STEPS`: explicit step count from the corpus manifest.
- `MASTER_ADDR`: rank-0 fabric address or hostname.
- `NODE_RANK` on each worker, or `NODE0_IP..NODE{N-1}_IP` for local IP matching.

The canonical GB10 CPT optimizer is Adafactor, not AdamW:

```bash
export LR=2e-5
export ADAFACTOR_CLIP_THRESHOLD=1.0
export WARMUP_STEPS=100
export LR_MIN_RATIO=0.0
```

`trainers/train_fsdp_dense_9b.py` uses
`Adafactor(scale_parameter=False, relative_step=False, warmup_init=False)` and a
manual `LambdaLR` with linear warmup followed by linear decay. AdamW is not the
canonical dense-CPT path because its optimizer state OOMs on GB10 UMA once page
cache and FSDP shards are present.

## Build Training Data

`trainers/build_training_data.py` is the migrated dataset builder. All paths are
explicit CLI args or `PALIOS_*` environment variables:

```bash
python3 dense-9b/trainers/build_training_data.py \
  --tokenizer "$MODEL_PATH" \
  --output-dir "$PALIOS_TRAINING_OUTPUT_DIR" \
  --phase cpt \
  --corpus-base /path/to/corpus/root \
  --infra-cpt-jsonl /path/to/infra_cpt.jsonl
```

For SFT and DPO, add the corresponding explicit inputs:

```bash
python3 dense-9b/trainers/build_training_data.py \
  --tokenizer "$MODEL_PATH" \
  --output-dir "$PALIOS_TRAINING_OUTPUT_DIR" \
  --phase sft \
  --sft-v2-dir /path/to/sft_v2 \
  --sft-clean-dir /path/to/sft_clean \
  --general-instruction-jsonl /path/to/general_instruction.jsonl

python3 dense-9b/trainers/build_training_data.py \
  --tokenizer "$MODEL_PATH" \
  --output-dir "$PALIOS_TRAINING_OUTPUT_DIR" \
  --phase dpo \
  --dpo-jsonl /path/to/dpo_pairs.jsonl
```

The builder fails if a selected phase lacks its required inputs; it never falls
back to deployment-local directories.

## Launch CPT

Per-worker launch:

```bash
source /path/to/filled-cpt-cluster.env
NODE_RANK=0 bash dense-9b/recipes/launch_cpt_phase2_qwen35_9b_fsdp.sh
```

Control-host orchestration:

```bash
source /path/to/filled-cpt-cluster.env
bash dense-9b/recipes/launch_fsdp_orchestrator_cpt_v0.sh
```

Stop launched ranks:

```bash
source /path/to/filled-cpt-cluster.env
bash dense-9b/recipes/launch_fsdp_orchestrator_cpt_v0.sh stop
```

Reboot mode uses `REBOOT_COMMAND`, defaulting to `sudo -n reboot`; configure
passwordless sudo or override the command. No password is stored in this repo.
