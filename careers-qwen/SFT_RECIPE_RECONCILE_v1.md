# SFT RECIPE RECONCILE v1 — architecture-grounded (tutor, 2026-07-19)

The two harvested SFT-recipe consults (`sft_recipe_gaia.md`, `sft_recipe_logos.md`) DIVERGE on
several knobs. Most of the divergence traces to ONE architecture fact that Gaia flagged and Logos
did not account for. I VERIFIED that fact from the model config + safetensors index (documents-first;
receipts below). This doc records what the fact resolves and what still needs a final Gaia
reconciliation. **Recipe = Chats-only (binding): I do not solo-pick the residual divergences — they
go back to Gaia.** This doc is the evidence packet for that reconciliation.

## VERIFIED architecture (receipts)

`<SPARK_HOME>/models/Qwen3.6-27B/config.json` → `text_config`:
- `model_type = qwen3_5_text`, `num_hidden_layers = 64`, `hidden_size = 5120`.
- `layer_types` = `[linear_attention ×3, full_attention, …]` repeating; `full_attention_interval = 4`.
  → **48 linear-attention (Gated-DeltaNet/SSM) layers (75%) + 16 full-attention (25%)**. Gaia's
  "≈3:1 linear:full hybrid" is **GENUINE/verified**.
- **No `num_experts` key → the 27B is DENSE, not MoE.** (Resolves Gaia's open branch: `all-linear`
  is safe re: experts; the MoE-ESFT caution does not apply.)
- SSM markers present: `mamba_ssm_dtype`, `linear_conv_kernel_dim=4`, `linear_num_key_heads=16`.

Per-layer weight names (safetensors index, no model load):
- **linear_attention layer (e.g. layer 0):** `linear_attn.in_proj_a`, `linear_attn.in_proj_b`,
  `linear_attn.in_proj_qkv`, `linear_attn.in_proj_z`, `linear_attn.out_proj` (all `.weight`-only →
  nn.Linear, NOT fused), `linear_attn.conv1d` (conv), `linear_attn.norm`; PLUS `self_attn.{q,k,v,o}_proj`
  and `mlp.{gate,up,down}_proj`.
- **full_attention layer (e.g. layer 3):** `self_attn.{q,k,v,o}_proj` + `mlp.{gate,up,down}_proj` only
  (no `linear_attn.*`).

## What the fact RESOLVES (not a preference — architecture-determined)

1. **target_modules — the standard list is WRONG for this base.** A
   `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` list (the `train_lora_sft.py` DEFAULT and
   Logos's explicit list) adapts only the 16 full-attention layers' token-mixing + all MLPs — it
   **skips the GDN token-mixing on 48/64 layers (75% of the model)**. The GDN projections are named
   `linear_attn.in_proj_{a,b,qkv,z}` + `linear_attn.out_proj`. **Correct target set** = either PEFT
   `all-linear` (Gaia; dense base makes it safe), OR an explicit augmented list that ADDS the GDN
   projections:
   ```
   q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,in_proj_a,in_proj_b,in_proj_qkv,in_proj_z,out_proj
   ```
   (PEFT matches by suffix, so `out_proj`→`linear_attn.out_proj`, `o_proj`→`self_attn.o_proj` — both
   covered; the existing harness `--target-modules a,b,c` split works with this augmented list, **no
   code change needed**, just the correct launch string). EXCLUDE always: `*conv1d*`, `*norm*`,
   `embed_tokens`, `lm_head`. `modules_to_save: []` (purely additive; base stays bit-identical).
2. **Packing OFF for module #1.** The 48 GDN layers carry an O(1) recurrent state that **bleeds
   across packed document boundaries** unless the FLA kernels reset on `cu_seqlens` (a block-diagonal
   attention mask fixes only the 16 full-attn layers, not the SSM state — silent corruption). Gaia's
   call stands; Logos's "packing ON" was architecture-blind. One sample per sequence, right-pad,
   `group_by_length`. (`train_lora_sft.py` is already one-sample-per-seq — matches.)

## RESIDUAL divergences → final Gaia reconciliation (NOT solo-decided)

| knob | Gaia | Logos | note |
|---|---|---|---|
| LoRA rank / alpha | 32 / 64 | 16 / 32 | Gaia: r32 for 3 heterogeneous jobs (scorer+voice+capability), r16 underfits scorer; Logos: r16 = base-conservatism + greener. |
| mixture | single-stage weighted (scorer .45/voice .35/repo .12/values .08) | staged curriculum (Ph1 skill+ethics, Ph2 voice) | Gaia: staged invites voice eroding scorer in one adapter; Logos: staged protects scorer from 62% voice majority. |
| max_seq | 4096 (memory not the constraint; drop >4096 outliers, don't truncate) | 3072 (defer 4096; trail-aware truncation of the minority) | both: never mid-schema truncate scorer trails. |
| LR / epochs | 1e-4 cosine, 2 epochs | 2e-5 cosine, ≤3 epochs, early-stop | both: assistant-only mask, warmup 5%, grad-clip 1.0. |
| values lane | don't gate on 61 pairs; move to DPO/system layer | audit 61 pairs vs KERNEL first; oversample ×5 | CONVERGE: 61 SFT pairs won't durably instill values — light touch only. |

**Convergences (both agree — lock these):** LoRA-not-full-param; base strict-frozen; assistant-only
loss masking; save-at-session-end (LoRA ~1–1.5GB sidesteps the DCP-save→resume OOM); cosine + 5%
warmup + grad-clip 1.0; re-validate SR/absolute-alpha on a 30-step LoRA probe before trusting it
(it was validated for CPT full-param, not LoRA); dry-run memory profile before the real run;
NEVER `expandable_segments:True`; **GATE-0 base-preservation** (re-run 20/20 + +3.4σ battery WITH
the adapter attached — pass = ≥19/20 and within −0.5σ; the module's prime directive).

## Loose end Gaia flagged (verify before trusting `all-linear`)
Layer 0 shows BOTH `linear_attn.*` AND `self_attn.*` projections on a layer typed `linear_attention`
— unusual. Confirm at build time via `named_modules()` whether the linear-attn layers' `self_attn.*`
are live or vestigial, and that `linear_attn.in_proj_*`/`out_proj` register as `nn.Linear` under
PEFT (so `all-linear` actually catches them). If the explicit augmented list is used instead, this
is moot — the suffixes match regardless.

## RESOLVED — Gaia reconciliation (2026-07-19, `sft_recipe_reconcile_gaia.md`, verified real)

Dispatched to Gaia (Claude Opus + extended-thinking) via taeys-hands; lint-clean packet. Reconciled
config (BINDING — implemented verbatim in `train_lora_sft.py` defaults + launch):

```
target_modules: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,in_proj_qkv,out_proj
  # self_attn q/k/v/o (16 full layers) + mlp (all 64) + GDN CONTENT path in_proj_qkv,out_proj (48
  # linear layers). EXCLUDES in_proj_a/b/z (GDN decay/write/output gates) — adapt what ENTERS the
  # recurrent memory, freeze how it decays/writes. Narrower than the full augmented list.
rank 16 / alpha 32 / dropout 0.05        # min perturbation under the 20/20 retention gate
sampler: single_stage_weighted (NOT staged) — scorer .45 / voice .35 / repo .12 / values .08
  + tiny-lane cap (values 21@≤3x, repo 160@≤3x) — mandatory anti-memorization
max_seq 4096 / packing OFF (GENUINE — GDN scan corrupts across packed boundaries; harness is 1-seq/sample)
LR 1e-4 cosine / warmup 5% / grad-clip 1.0 / 2 epochs
convergences: LoRA-only; base frozen; assistant-only mask; save-session-end; expandable_segments OFF
```

**Verifications (Gaia's V1–V4):** V1 (GDN projs are nn.Linear/PEFT-targetable) + V3 (self_attn only on
the 16 full layers) — BOTH already cleared by the smoke (commit a5c4d9e: adapters attached to 48 GDN
`in_proj_qkv`/`out_proj` + 16 self_attn, loss 6.70→4.43). V2 (a/b/z = decay/beta/gate) — moot, all
three frozen. V4 (optimizer/mask/param-count/step-time/mem probe) — gated by watching the first ~30
steps of the binding run. **GATE-0 base-preservation:** retention battery WITH adapter attached, PASS =
20/20 within the +3.4σ band, zero probe regressed >3.4σ, frozen tensors bit-identical.

*Epistemic: architecture facts = operational_verified (config + safetensors index + smoke). Gaia's
knob choices = the Chats' reconciled recommendation (mostly INFERRED, target_modules gated on V1/V3
now cleared). Implemented verbatim per recipe=Chats-only.*
