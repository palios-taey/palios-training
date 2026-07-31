# 6SIGMA IMPROVE spec — SFT-27B: multimodal-aware full-param trainer (train_fsdp_dense_9b.py)

**Task:** (fresh) · **Owner:** infra design (tutor precise spec) → Codex implement → tutor production-run oracle → Conductor merge
**Branch off main. Anchor edits on CONTENT, not line numbers (they drift).** Parallel: production-run is node-gated (needs a 3rd healthy Spark), so land the CODE now.

## Goal
`train_fsdp_dense_9b.py` loads `AutoModelForCausalLM` → DROPS the 27B multimodal vision tower (the phase-s bake lesson). For 27B (Qwen3.6-27B = `Qwen3_5ForConditionalGeneration`, multimodal), load the FULL model, FREEZE the vision tower, full-param the 64 text `Qwen3_5DecoderLayer`s. dcp sharded save + metadata-broadcast (already merged fix) UNCHANGED.

## Changes (all in train_fsdp_dense_9b.py)

### (1) IMPORT — add alongside `from transformers import AutoModelForCausalLM, AutoTokenizer`:
```python
from transformers import AutoModelForImageTextToText, AutoConfig
```

### (2) MODEL LOAD — the two `AutoModelForCausalLM.from_pretrained(...)` sites. Make it CONDITIONAL on the base architecture so the text-only 9B path is NOT broken (SAFETY — infra decision; tutor: flag if you want 27B-only-unconditional instead):
```python
_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
_is_multimodal = "ForConditionalGeneration" in (_cfg.architectures or [""])[0] or hasattr(_cfg, "vision_config")
if _is_multimodal:
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True, ...<keep existing kwargs: device_map/attn_impl/etc>)
else:
    model = AutoModelForCausalLM.from_pretrained(model_path, ...<existing kwargs unchanged>)
```
Apply at BOTH load sites. Preserve every existing kwarg (attn_implementation, low_cpu_mem_usage, etc.).

### (3) FREEZE VISION + full-param text — replace the existing 'unfreeze every param' block (the full-FT requires_grad=True loop):
```python
for n, p in model.named_parameters():
    p.requires_grad_(not (("visual" in n) or ("vision_" in n) or (".visual." in n)))   # vision FROZEN; text (language_model.layers, embed, lm_head) trainable
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
log.info(f"SFT-27B trainable={n_train/1e9:.2f}B frozen(vision)={n_frozen/1e9:.2f}B")  # frozen should be the ~348 vision tensors, NOT the text
```
(For the text-only 9B path, this freezes nothing — no 'visual' names — so it's a no-op there. Correct.)

### (4) FSDP WRAP — UNCHANGED: `transformer_layer_cls={Qwen3_5DecoderLayer}` (already correct, confirmed = 27B text decoder layer). Do NOT touch.

### (5) DCP SAVE + metadata-broadcast — UNCHANGED (the merged fix: dcp.save sharded + rank0 .metadata broadcast). Do NOT touch _save_checkpoint.

## VERIFY (the one runtime unknown — CHECK statically, tutor's oracle confirms at runtime)
Text-only forward: `model(input_ids=..., labels=...)` with NO `pixel_values` must not trip the multimodal image path. Statically check the ConditionalGeneration `forward` signature: does it require pixel_values, or handle None? If it forces the image path, route the loss through `model.model.language_model` (the text decoder) OR set a text-only flag (e.g. the base config's `language_model_only`). DO NOT guess a runtime fix — if the forward looks like it needs pixel_values, FLAG it in the handoff; tutor tests the real forward in the production oracle.

## Constraints
- 6SIGMA root-cause: replace the load/freeze cleanly; the CONDITIONAL keeps both 9B-text + 27B-multimodal working (no silent break).
- Config: 27B run uses fsdp_dense_27b_3node.yaml (tutor deploys to nodes) — not your concern; just the trainer code.
- NO synthetic tests. Production run is the oracle (tutor, on {.12,.19,+3rd node when it lands}).
- Report the diff + gitnexus_impact + the forward-signature finding (does text-only forward need a route/flag?) in the handoff.
