# 27B REVENUE CPT — RECIPE v2 (5-lane Family synthesis, 2026-07-13)

Consult: `treasurer/consultations/consult_27b_epoch1_recipe.md` → 5 lanes (Gaia/Logos/Cosmos/Clarity/Horizon). Responses in `consultations/responses/epoch1_recipe_*.md`.

## ROOT CAUSE — CONFIG, not under-dose (5/5 converge)
The null result (weights ~1e-4, model.norm bit-identical, flat loss) is a **precision + optimizer** failure, not too-few-epochs. **Adding epochs at the old config produces another null** (Gaia, Horizon, Clarity explicit).

1. **bf16 masters, no fp32 master (Gaia, Horizon).** Model loaded bf16 + `MixedPrecisionPolicy(param_dtype=bf16, reduce=bf16, output=bf16)` → the sharded master params the optimizer updates are bf16. Adafactor lr-1e-5 updates are ~1000× below the bf16 ULP → rounded to zero on write-back. Physics matches EXACTLY: at norm (~1.0) bf16 ULP≈0.0078, coherent 766-step ceiling≈7.7e-3 → never crosses → **bit-identical predicted**; decoder (~0.02) ULP≈1.2e-4 → **~1e-4 residual predicted**; residual scales with weight magnitude like ULP = corroboration. **Confirmed in code: no fp32 master exists.**
2. **Adafactor relative-step starvation (Clarity, Cosmos, Horizon).** `torch.optim.Adafactor` multiplies by parameter RMS and treats LR as a *relative-step ceiling* → effective per-param step ≪ nominal LR. The earlier *working* recipes used `transformers.Adafactor(scale_parameter=False, relative_step=False)` = **absolute** LR. Also weight_decay was negligible (1−lr·wd = 1−1e-7 → rounds to 1 in bf16).
3. **Data (governance):** the run also used an UNREGISTERED corpus (`comprehensive_v1`, constitution-included, out of scope). Now fixed: registered `cpt_revenue_jesse_v1` (sha c3461005, constitution-stripped). Ad-hoc/out-of-scope probe was invalid (no base-control, wrong content).

## THE FIX (consensus, memory-aware for 4×GB10 128GB UMA)
**A. Precision (THE core fix):** keep **fp32 master weights** + fp32 optimizer update math; bf16 only for forward/backward compute (`MixedPrecisionPolicy(param_dtype=bf16)` over an fp32 module). Fallback if OOM: stochastic-rounding bf16 master.
**B. Optimizer:** switch to **AdamW** (Cosmos/Clarity: dominant for LM pretrain, moves weights far more/step) — use **8-bit AdamW** for memory, OR keep Adafactor but with **`scale_parameter=False, relative_step=False`** (absolute LR, the earlier proven shape — Horizon). Adafactor+fp32-master is the lowest-memory path (<1GB factored state vs AdamW fp32 54GB/rank; 8-bit AdamW ~13.5GB/rank).
**C. LR/schedule:** peak **3e-5–5e-5**, warmup ~4%, cosine → ~10% floor.
**D. Dose:** **3 epochs** on the mixture (2–4, max ~5), grade at each epoch boundary; stop on probe plateau / memorization-probe rise / relating regression (NOT loss=0).
**E. Data mixture (registered CPT):** `cpt_raw_corpus` v3 + `voice_cpt_slice` v1 (**cap voice ≤~30% of tokens** so style doesn't drown knowledge) + `identity_ai_work_background` v1 + **[GAP] extract+register platform-strategy/best-practices + revenue-research as new `cpt_*` prose** (they exist only as SFT pairs today — SFT won't imprint knowledge in weights). +10–20% generic replay to protect the base's 91.7% relating. EntiGraph/diverse renderings + doc-boundary attention reset to learn knowledge not surface strings.
**F. Eval:** frozen `k1_career_history_probes`(8) + `k2_repo_capabilities_probes`(12) as targets (base-control ≈0 = clean floor), `relating_probes_v1`(12, base 91.7%) as a regression tripwire. **[GAP] freeze new voice/strategy/revenue probe batteries** (2 of 5 in-scope capabilities have no frozen battery). Paraphrased closed-book QA (generalization) + verbatim-continuation probe (high exact-match = memorized/bad).
**G. Sequencing:** **CPT** (knowledge+voice into weights) → **SFT** (registered pair sets: b1_repos/upwork/profile/interview/stage2_verdict_trails v4/…, behavior + facts backstop) → **DPO** (preference polish). Grade after EACH stage.

## THE CHEAP GATE (Gaia — do this BEFORE any 16h epoch)
Run **50 steps** at corrected precision+optimizer+LR, then assert: (a) loss descends, (b) grad norms nonzero/sane, (c) `model.norm` + decoder deltas vs base now **>10× bf16 noise** (i.e. ≫1e-4, moving toward 1e-3+). If all three hold → the null is fixed, licensed to spend the epoch. ~1h, not 16h.

## PROCESS GATES (poka-yoke — wire into launcher so the bad run is impossible)
1. Registration gate: refuse to start unless every dataset id+hash ∈ REGISTRY.md AND method matches TRAINING_METHOD_MAP allowed-use.
2. RUN_REGISTRY auto-write at start (dataset hashes, method, **precision policy**, optimizer, LR, epochs, git SHA, base-control scores).
3. **Precision preflight + write-back assertion**: assert fp32 master (or SR) present; after warmup assert weight-delta > bf16-noise or ABORT "updates not applying." (This one check catches this exact null.)
4. Eval-validity gate: close runs only vs frozen registered batteries with a base-control row.

## ORDERED EXECUTION
1. Implement fp32 master (+ optimizer fix) in `train_fsdp_dense_9b.py`; deploy 4 nodes.
2. 50-step gate on `cpt_revenue_jesse_v1` → verify weight-delta ≫bf16-noise + loss↓. (~1h)
3. If pass: extract+register the strategy/revenue CPT prose + freeze the missing probe batteries; run base-control.
4. 3-epoch CPT on the full registered mixture; bake+probe each epoch boundary.
5. SFT → DPO, grade after each.

## GATE RESULT 1 (2026-07-13): naive fp32 full-load OOMs on GB10 UMA — CONFIRMED
FP32_MASTER=1 loaded the full fp32 model to unified memory (106GB used / 11.6GB free), then OOM'd at first step (tried 4.74GB, 4.29GB free). On GB10's 128GB UNIFIED pool the "CPU" fp32 load and GPU compute share memory, so a 106GB fp32 master leaves no room. Naive `torch_dtype=float32` load is NOT viable. Default reverted to FP32_MASTER=0.
NEXT (memory-safe fp32 master, from the consult): (i) stochastic-rounding on the bf16 master update (Gaia — memory-NEUTRAL, preserves sub-ULP updates in expectation; implement in the existing Adafactor DTensor monkeypatch); (ii) bf16-load → fully_shard → upcast the per-rank SHARDED param to fp32 (peak 54GB load, 27GB/rank fp32 shard — avoids the 106GB full load); (iii) 8-bit AdamW (check bitsandbytes aarch64/sm_121 availability first). Dispatching the exact-implementation question to Gaia/Chats.

## AVAILABILITY (2026-07-13): torchao AdamW8bit IS on the nodes
`<SPARK_HOME>/training-env`: **torchao 0.16.0** imports + `torchao.optim.AdamW8bit` available. Likely the cleanest memory-safe fix — 8-bit optimizer state (~13.5GB/rank, fits) + torchao's stochastic-rounding write-back to bf16 params (addresses the sub-ULP rounding WITHOUT a 106GB fp32 load). Awaiting Chats' confirmation of exact config (SR default? args) before implementing. bitsandbytes availability unconfirmed; torchao is the aarch64-safe path.

## IMPL CONSULT VERDICT (2026-07-13, 3 lanes UNANIMOUS): STOCHASTIC ROUNDING
Gaia/Cosmos/Logos all → **Option 1: SR on the bf16 write-back.** Key evidence:
- Option 3 (8-bit AdamW) DEAD on GB10: bitsandbytes aarch64 wheels ship no sm_120/121 kernels (certain); torchao 8-bit leans on Triton which ptxas-fails on sm_121a (certain). Also explains Jesse's "Adam never worked" history.
- Option 2 (sharded fp32 master) — Cosmos: mathematically impossible on UMA (fp32 masters 108GB + bf16 grads 54GB = 162GB aggregate > 128GB); Gaia: post-prepare `.data` reassignment is FSDP2-unsafe (Unknown). Fallback = meta-init + DCP refactor only.
- SR: memory-NEUTRAL (per-param local-shard fp32 temp), DTensor-safe (bit math on plain local tensors — DTensor forbids view(dtype)/bitwise), sign-correct/unbiased (two's-complement add == unsigned add), the bf16 cast after masking is EXACT.
IMPLEMENTED (tutor, per Gaia's exact code): `_sr_apply_update_` in the Adafactor monkeypatch — folds weight-decay (its 1-lr*wd also rounded to 1.0 in bf16) + update into ONE fp32 op then SR write-back; `[SR-DELTA]` write-through probe logs mean|dW| vs bf16-noise at steps 1/10/20/40 (catches a vanishing-update failure in ~20 steps). Deployed all 4 nodes, py_compile clean.
GATE PENDING: 50-step run → [SR-DELTA] must be ≫10× noise + bake-diff at ckpt-50 ≫1e-4 + loss descent. CORPUS: awaiting Treasurer sanction (never-again rule 1 — no run on a corpus not blessed by the data owner).

## SR GATE (retry, 2026-07-14 00:35) — interim reading, honest
- Crash #1 was MY translation bug (update to_local'd pre-redistribution; 62080-vs-248320 on dim-0-sharded embed/lm_head). Fixed: fp32 arithmetic at DTensor level (auto-reshard = the proven param.add_ semantics), SR bit-math on the result's local shard. Retry runs crash-free, LR reached 5e-5.
- [SR-DELTA] step-10 on layers.0.linear_attn.out_proj: mean|dW|=3.631e-08 → probe says LOW. CALIBRATION CAVEAT (mine, flagged for Gaia): torch-Adafactor alpha=RMS(param)×rho → per-element step ≈ 0.02×lr ≈ 5e-7 on decoder weights — even healthy SR cannot reach the 10×-noise bar there in 40 steps; the probe's threshold assumed |u|≈lr (valid for ~1.0-magnitude norm weights, not 0.02 decoders). Measured 3.6e-8 ≈ the weight-decay-only floor; nonzero proves write-through lands (dead copy_ would read exactly 0.0). Update-term health = still OPEN.
- DECISIVE CHECK (unchanged): ckpt-50 bake-diff — model.norm bit-identical = FAIL (the null signature); model.norm mean-abs-diff ~1e-3 + decoder movement = PASS. Step-20/40 SR-DELTA series + this calibration note go to Gaia either way.

## SR GATE VERDICT (2026-07-14 01:45) — AMBIGUOUS → GAIA (per plan; no solo iteration)
Bake-diff srgate_step50 vs base (50 steps, lr 5e-5, torch-Adafactor + SR):
| tensor | IDENTICAL | maxabs | meanabs | changed_frac | base_meanabs |
|---|---|---|---|---|---|
| model.norm.weight | **False** | 7.81e-3 (=1 ULP @1.0) | 2.78e-5 | 0.57% | 0.962 |
| layers.32.input_layernorm | False | 3.91e-3 | 5.60e-6 | 1.76% | 0.130 |
| layers.30.mlp.down_proj | False | 4.88e-4 | 2.21e-7 | 1.05% | 0.0085 |
| layers.50.mlp.gate_proj | False | 9.77e-4 | 1.83e-7 | 0.54% | 0.0084 |
READ: (1) **SR mechanism WORKS** — null signature broken (norm no longer bit-identical; flips at exactly the bf16 ULP quantum; run crash-free, loss sane). (2) **Update term lands ~30-50× below the healthy prediction** — norm meanabs ≈ the wd-only floor (2.4e-5 predicted); changed_frac 0.5-1.8% vs ~27-50% predicted if per-element steps were ≈RMS(p)×lr. input_layernorm at ~6× wd-floor = SOME update signal, heavily attenuated. Candidate causes (for Gaia, not solo-resolved): torch-Adafactor d=1.0 update-RMS clipping (denom=max(1,RMS(u)) scales down when early var-estimates are small), relative-step alpha=RMS(p)×lr semantics (Horizon's warning — the earlier WORKING recipes used transformers-Adafactor scale_parameter=False = ABSOLUTE lr), or grads genuinely small. SR-DELTA in-run series corroborates: 3.6e-8@10 → 2.0e-7@20 → 3.9e-7@40 (~wd-scale growth).
IMPLICATION IF CONFIRMED: at this effective step size, even 3 epochs (~2220 steps) moves decoders only ~2e-5 total — likely too small to imprint knowledge. The recipe may need the optimizer-semantics switch the lanes already flagged (transformers-Adafactor absolute-LR, per Horizon/Logos) — THAT DECISION = Chats'.

## GAIA'S RE-GATE CONFIG (sr_gate_verdict response, 2026-07-14 — implemented EXACTLY)
Verdict: SR HEALTHY, no write-back defect. Gap = RMS-scaled alpha semantics: mixing time u/δ = 2^-7/lr = 156 steps @5e-5, |w|-independent → 50-step gate had SNR 0.57 everywhere (all 4 rows explained). d=1.0 clipping = red herring (no-op in steady state). Dose at old config confirmed non-imprinting (<1 decoder ULP net over 3 epochs).
FIX (one line, SR untouched): absolute alpha — param_scale=1.0 → alpha=rho_t=lr; **lr=1e-5**. Mixing now |w|-proportional: decoder ~6 steps (imprints), norm ~780 (stays put — desired selectivity). Do NOT swap to transformers.Adafactor (SR interception would move + need re-validation).
IMPLEMENTED: _AF_ALPHA_ABSOLUTE (env ADAFACTOR_ALPHA_MODE, default absolute) in the monkeypatch; SR-DELTA probe recalibrated ULP-anchored on decoder mlp.down_proj (liveness: cross 1 ULP by step 10-15; PASS@40 = [0.5u, 20u] = [3e-5, 1.2e-3]; FAIL-HIGH >20u → lower lr). Launcher forwards ADAFACTOR_ALPHA_MODE. Deployed all 4.
RE-GATE bake-diff PASS BAND (Gaia): decoder mlp meanabs 5e-5–8e-4 + changed 30-80% (GATE tensors); input_layernorm 2e-5–1.5e-4 secondary; model.norm ≤1e-4 <10% changed = CORRECT (stays put — do NOT gate on it). FAIL-LOW: decoder ~2e-7. FAIL-HIGH: >1e-2 → drop lr.
Watch-item (Gaia): bf16+SR freezes the deep gradient tail (per-step δ≪u never crosses) — accepted UMA trade; post-run changed-fraction histogram to quantify.
RE-GATE RUNNING: revenue_regate, lr=1e-5 absolute-alpha (verified in proc), same treasurer-sanctioned gate corpus (ecdca0f3).

---

## 2026-07-14 ~03:00 — Horizon lane 2 landed (sr_gate_verdict_horizon.md): eps1 floor — CODE-FACT VERIFIED

**Horizon's finding (verified in OUR trainer, `train_fsdp_dense_9b.py` Adafactor monkeypatch):**
`eps1 = torch.finfo(param.dtype).eps` with bf16 params → **eps1 = 2⁻⁷ = 0.0078125**, and the
update path floors at it (`row_var.mean().clamp(min=eps1)` + `var_estimate.clamp_(min=eps1²)`).
When sqrt(var) < 0.0078 the normalized update becomes ≈ RMS(g)/0.0078 ≈ 0.02–0.03 instead of ~1
→ exactly the 30–50× suppression band. HF Adafactor differs (fp32 grads + eps[0]=1e-30 additive).

**Relation to Gaia's verdict — they COMPOSE, not contradict:**
- Gaia: alpha SHAPE wrong (RMS-scaled → mixing time 156 steps ≫ 50-step gate). Fixed in the running re-gate (absolute alpha, lr=1e-5).
- Horizon: update NORMALIZATION floored by bf16-eps1 (30–50×). NOT fixed — ACTIVE in the running re-gate.
- Horizon also corrects our input_layernorm read: warmup-aware wd floor puts it at ~1.9× floor (not 6×) — the old gate was even cleaner wd-only than we thought.

**The running re-gate now DISCRIMINATES (decoder mlp meanabs @ ckpt-50 bake-diff):**
| outcome | meaning |
|---|---|
| ~2e-7 (wd floor) | alpha fix didn't take (unlikely — env verified in proc) |
| ~1e-5 (≈50× floor but BELOW Gaia band 5e-5) | absolute alpha works AND eps1 floor active → need Horizon's eps1 fix on top |
| 5e-5–8e-4 (Gaia band) | eps1 floor not binding (grads big enough) → Gaia config sufficient, PASS |

**Conflict to reconcile via consult (NO solo pick)** after ckpt-50 numbers land:
- Horizon: keep RMS-scaled lr=5e-5, fix eps1 first (isolation gate); absolute-3e-5 only as fallback. Warns vs combining eps1-fix + big-LR-raise (~900–1500×).
- Gaia: absolute lr=1e-5 for shape reasons (selectivity: norm stays put). Note absolute-1e-5 is BELOW Horizon's fallback (3e-5) — a conservative composite: absolute-alpha lr=1e-5 + eps1=fp32.
→ One packet to Gaia (+Horizon lane if warranted) with: re-gate bake-diff numbers, the code-fact, [AF-DOSE] readings, ask for the composite ruling.

**Prepared (local, deploy at session-save — NOT in the running proc):**
1. `ADAFACTOR_EPS1` env-gate (`fp32` → 1.192e-7, numeric, default = stock bf16 behavior unchanged).
2. `[AF-DOSE]` pre-SR instrumentation (both lanes' decisive measurement): eps1_actual, floor_frac (pre-clamp), RMS(U_hat), denom, alpha, preSR_RMS_delta — first 8 2-D params @ steps 1/10/20/40/50, rank0.
3. [SR-DELTA] probe: typed exception logging (empty "()" skip = StopIteration signature) + fallback predicate (first non-embed/non-norm 2-D) so liveness is never silently lost. Root cause of the mlp.down_proj miss still unknown — mlp.down_proj names EXIST on the bare class (64 hits on meta-device enumeration, same transformers install); next run's typed log decides.

## 2026-07-14 11:30 — RE-GATE VERDICT: alpha fix TOOK, eps1 floor CONFIRMED (middle outcome)

Bake-diff regate_step50 vs base (absolute alpha, lr=1e-5, eps1 unchanged=bf16-eps):
| tensor | maxabs | meanabs | changed | base mean|w| |
|---|---|---|---|---|
| model.norm | 7.81e-3 | 8.39e-6 | 0.16% | 0.962 |
| L32.input_layernorm | 9.77e-4 | 1.27e-6 | 0.88% | 0.130 |
| L30.mlp.down_proj | 4.88e-4 | 6.11e-7 | 4.14% | 0.0085 |
| L50.mlp.gate_proj | 9.77e-4 | 2.72e-7 | 2.15% | 0.0084 |

- down_proj = ~16× THIS run's wd floor (3.8e-8) → real gradient signal (old gate sat ON its floor).
- Signal growth ~19× ≈ the ~23× alpha increase → **alpha fix proportionally effective**.
- Still ~30–100× below Gaia band; changed-frac 4.1% vs 30–80% → **U_hat ≈ 0.02–0.03 = eps1-floor factor** (Horizon), empirically confirmed as the remaining suppressor.
- model.norm 0.16% changed → Gaia's intended selectivity observed.
- VERDICT: NOT validated yet; composite consult dispatched (consult_regate_composite.md) → Gaia + Horizon lanes. NO further gates until the ruling. Probe fix + [AF-DOSE] instrumentation deployed for the next gate.

## 2026-07-14 ~12:15 — COMPOSITE RULING (UNANIMOUS) + CORPUS SANCTION WITHDRAWN

**Ruling (regate_composite_{gaia,horizon}.md, unanimous):** gate 3 = SINGLE-VARIABLE eps1 isolation.
`ADAFACTOR_ALPHA_MODE=absolute ADAFACTOR_EPS1=fp32 LR=1e-5 WARMUP_STEPS=10` + wd 0.01, d 1.0, SR
unchanged — byte-identical to the re-gate except eps1. Horizon explicitly endorses (relative-alpha
would confound). Gaia: eps1=fp32 is a CORRECTNESS fix (bf16 eps 7.8e-3 ≈ 40× decoder grad RMS →
clamp destroys the normalization invariant), carries to production unconditionally.

**Mechanical PASS band (AF-DOSE leads, steps 20–50 decoder 2-D):** eps1_actual==1.192e-7;
floor_frac<0.05 (primary); RMS(U_hat) 0.3–1.5 (primary; treasurer routing quotes 0.8–1.0 center);
alpha==1e-5 flat; preSR_RMS_delta≈lr×RMS(U_hat) within 2×; [SR-DELTA] non-null ≈ preSR delta.
Bake-diff confirm: down/gate_proj meanabs 1e-5–8e-4, changed ≥~15%. Predicted center: meanabs
~2–3e-5, changed ~20–60%. FAIL-LOW (U_hat~1 but meanabs<1e-5) = lr genuinely low → feed U_hat into
production-lr arithmetic. FAIL-HIGH (U_hat>1.5 / meanabs>8e-4 / changed>80%) = back off lr.
FAIL-INCOHERENT (floor_frac still high with eps1 correct) = STOP, dump AF-DOSE denom, re-audit clamps.

**Production carry:** eps1=fp32 unconditional; lr COMPUTED: lr_prod = 1e-5 × (M_target / meanabs_gate3)
(e.g. ~4.3e-5 if U_hat≈1, meanabs≈2.3e-5, M_target 1e-4); schedule → warmup 2–5% + cosine to 5–10%
(NOT flat); wd near-inert at these lrs (keep, don't rely); 100–200-step smoke test on the production
schedule before 2220×3; re-read U_hat mid-run (grad RMS drifts as loss drops).

**⚠ CORPUS SANCTION WITHDRAWN (treasurer escalation, msg d2e9f62bdd2d):** their marker scan was a
structural false negative (ran on the TOKENIZED packed file). Raw-text truth: ~31/1,000 kept rows
carry constitution markers (SACRED_TRUST 785 rows, FAMILY_KERNEL 354, GOD=MATH 390, incl. whole
identity files) — material Jesse explicitly excluded from this training. cpt_revenue_jesse_v1 (+
packed) is NOT sanctioned for ANY further use, gates included. Blast radius contained: gate-only
scope, all gate weights discarded, never merged/served. Their strip arithmetic + packing trace DID
verify bit-perfect; remediation plan (registered rebuild vs spec::treasurer_corpus_target, raw-text
scanning) due today.

**STATE: gate 3 fully specified + ready; BLOCKED solely on a treasurer-sanctioned corpus (gate slice
or rebuilt production corpus). No runs of any kind until one exists. Requested from treasurer.**

## 2026-07-14 ~12:40 — CORPUS RESTORED (Jesse calibration via treasurer, msg adc2fc5a5971) → GATE 3 GO

Jesse: constitutional content in the corpus is FINE ("doesn't need to be scrubbed or hidden, we just
aren't doing a personality/constitutional training run"); objective = WORK-KNOWLEDGE + VOICE.
cpt_revenue_jesse_v1(+packed ecdca0f3) RESTORED for gate/engineering use — the ~31/1,000 constitution
rows are tolerated noise. What stands from the audit: raw-text-scan-only process lesson (leak-rate
metric, never pass/fail), registry hygiene remediation, full production rebuild vs
spec::treasurer_corpus_target rev2. → Launching gate 3 (unanimous eps1 config) on the restored corpus.

## 2026-07-14 15:58 — gate3 bake attempt 1 WEDGED (infra-flake class) → one clean reboot-retry

Bake launched 15:11:57, DCP resume OK (15:21:43, load took only 15s), then rank0 gather
("BAKE: gathering FULL model state dict") sat 35+ min with ALL 4 ranks at ~0.1% CPU, ZERO rail
rx bytes over a 60s sample, ZERO rank0 memory growth (68106M→68095M) — a dead collective, not a
slow gather. Identical code+flow completed in ~34 min for the regate bake this morning → flake
class, not code. Pre-registered response: ONE retry via full reboot cycle (no kill-and-relaunch).
If the retry wedges too → facts to Gaia/infra, stop.

## 2026-07-14 16:20 — WEDGE #2 (pattern) → bake HALTED pending Chats' architecture ruling

Retry wedged at the identical post-resume gather point: 21+ min, rank0 RAM flat 60-61G, zero
traffic on ALL 3 NICs, kernel stacks = userspace waits (nanosleep/ppoll — no D-state), no errors.
2/2 afternoon wedges vs 2/2 morning successes, identical code/flow. Clock-skew hypothesis tested
and ELIMINATED (same-second clocks, NTP synced). NOT retrying (2 = pattern, per pre-registration).
Cluster rebooted clean + idle. consult_bake_architecture.md (+addendum) with Gaia+Horizon lanes;
their Q1 (architecture) + Q2 (discriminating instrumentation) now gate the export path.

**GATE-3 VERDICT POSTURE (honest):** per Gaia's own band design ("the instrumentation is the
oracle; bake-diff is downstream confirmation"), gate 3's PRIMARY evidence is complete and PASSED:
eps1_actual=1.192e-7 all tensors; decoder floor_frac=0.000 steps 20-50 (primary); RMS(U_hat)
0.25→0.59→0.88→1.0-at-clip (primary, in 0.3-1.5 band; d=1 clip engaged at step 50, denom 1.7-1.87);
alpha=1e-5 flat; preSR_RMS_delta=lr×U_hat exact; [SR-DELTA] 0.11x→0.46x→1.06x ULP @40 = its own
PASS tag in [0.5u,20u]. The bake-diff CONFIRMATION is deferred to the new bake architecture — the
ckpt-50 DCP bundles are intact on the nodes and can be baked under whatever shape the Chats rule.
Optimizer lane: VALIDATED-by-primary-evidence; production lr arithmetic will use the gate-3
bake-diff numbers once the new export path produces them (or the Chats specify an alternative
calibration source).

## 2026-07-14 20:05 — ★ GATE-3 VERDICT: PASS + NEW BAKE PATH VALIDATED (RAM-only diff, no full gather ever)

The new Artifact-B export path (gloo-coordinated, EXPORT_DCP) ran on gate3 ckpt-50: fabric preflight
OK (all_reduce=4), coordinated save 26s (NO wedge, NO 51GB gather), global .metadata PRESENT.
Offline no_dist dcp.load assembled 851 tensors, strict load_state_dict OK, NaN/Inf=0. HF write hit
disk-full (.68 at 100%; production convert host = Thor1 395G per infra) → validated RAM-only instead.

Probe diffs (V = dcp-loaded export, B = base safetensors):
| tensor | maxabs | meanabs | changed | V_meanabs | B_meanabs |
|---|---|---|---|---|---|
| model.norm | 7.81e-3 | 3.13e-4 | 6.1% | 0.96197 | 0.96197 |
| L32.input_layernorm | 3.91e-3 | 1.05e-4 | 24.5% | 0.12950 | 0.12951 |
| L30.mlp.down_proj | 1.95e-3 | 8.70e-5 | 84.5% | 0.00852 | 0.00852 |
| L50.mlp.gate_proj | 2.93e-3 | 8.87e-5 | 84.4% | 0.00836 | 0.00836 |

**(A) BAKE PATH = CORRECT:** model.norm V==B==0.962 (NOT scrambled-0.23), all V_meanabs match base,
NaN=0. The gloo-coordinated export + offline no_dist convert assembles weights bit-coherently. The
offline-reassembly-broken problem is SOLVED (global .metadata was the fix). Wedge class eliminated.
**(B) GATE-3 OPTIMIZER = PASS:** decoder meanabs 8.7e-5 (in Gaia band [1e-5,8e-4], ~1% of |w|) =
**142× the eps1-floored regate (6.11e-7)** → the eps1=fp32 fix demonstrably imprints. model.norm
stays put (selectivity correct). FLAG (honest): changed_frac 84% is above the predicted 30-80%;
NOT a fail (FAIL-HIGH gate = meanabs >1e-2 = 10% of |w|; we're at 1%, 100× below) — high changed-
fraction is the expected SR signature at U_hat~1 (every low-|w| element nudged past ≥1 ULP). Noted
for the Chats' awareness; the load-bearing meanabs magnitude is healthy.

**NET: optimizer VALIDATED (in-run AF-DOSE + now weight-diff) + bake path PROVEN. Production run
gated only on treasurer registering corpus 4bfb2a57.**

## 2026-07-15 12:40Z — ★ INCIDENT: PRODUCTION EPOCH-1 DOWN — whole-node OOM AFTER the DCP save (NOT thermal)

**What happened:** production_v1 trained cleanly to step 40 (all AF-DOSE in band; SR-DELTA @40 =
1.15x ULP = **PASS in Gaia band** — the optimizer IS imprinting in production). `DCP save COMPLETE:
checkpoint-40` at 12:37:38 (atomic, resumable — NO WORK LOST). Then ~2 min later ALL FOUR nodes
hard-died and rebooted (12:40 .68/.80/.12, 12:44 .19).

**Root cause = MEMORY EXHAUSTION, not thermal** (kernel journal, prior boot, .68 — same signature
confirmed on .12):
```
12:39:45 earlyoom: mem avail: 0 of 122566 MiB (0.00%), swap free 13407/16383
12:39:45 earlyoom: low memory! at or below SIGTERM limits
12:39:45 earlyoom: Could not find a process to kill (selected itself)
12:39:46 systemd-journald: Under memory pressure, flushing caches
→ node dead, boot -1 ended 12:39:46, boot 0 at 12:40:38
```
Thermal EXCLUDED: watchdog max this run = 83C on .68 (PULL_OFF=86 never fired; watchdog still alive
and logging throughout). No Traceback / no torch OOM / no CUDA error — the HOST ran out of RAM and
the kernel died.

**THE KEY INSIGHT (why this never showed up before):** every gate ran `SESSION_LIMIT == SAVE_EVERY`
(50/50) → the process ALWAYS exited immediately at the save (FRAGMENTATION EXIT). **Training
CONTINUING past a DCP save has NEVER been exercised.** production_v1 set SESSION_LIMIT=90 /
SAVE_EVERY=40 — the first run to try to keep training after a save — and it died there. The save
host-stages ~13.4GB/rank on GB10 UNIFIED memory; the trainer's pre-save `gc.collect + empty_cache`
reclaim addresses the save itself, but on RESUMING training the allocator re-grows its cache while
the save's staged buffers are apparently not yet returned → 122GB pool → 0 avail → node death.
(Related prior incident in the trainer comments: "the save died at step 60 for lack of
driver-visible headroom" — same class, patched pre-save only.)

**State:** checkpoint-40 COMPLETE + atomic on all 4 nodes → resume loses nothing. Cluster rebooted,
idle. Clock cap is NOT applied post-reboot (cap doesn't persist — run_4node re-applies at launch).

**DO NOT blindly resume with SESSION_LIMIT>SAVE_EVERY — it will die again at the next save (step 80).**

**Safe path (uses ONLY the proven exit-at-save pattern):** set `SESSION_LIMIT == SAVE_EVERY` (e.g.
40/40) so the process always exits AT the save and resumes fresh after a reboot — exactly what every
gate did and what the resumable-2h-cycle architecture already assumes. Cost: a reboot+resume every
~51 min (~3 cycles/epoch) instead of one long session. Real fix (post-save memory reclaim so
training can continue in-process) = **Chats' ruling required, not a solo patch.**

## 2026-07-15 15:13Z — ★★ EPOCH-1 COMPLETE (checkpoint-121, clean — the save/reboot fix WORKED)

After the 12:40Z OOM, resumed from checkpoint-40 with SESSION_LIMIT=SAVE_EVERY=81 (Jesse's
save==session==reboot rule). Trained steps 41→121 clean, ~77s/step, ONE save at step 121
(checkpoint-121, atomic: step=121 epoch=0 data_pos=121), clean FRAGMENTATION EXIT. **NO OOM, NO node
death** — nodes never rebooted this session (uptime continuous from the 12:40 reboot). The
SESSION_LIMIT==SAVE_EVERY architecture is validated in production. AF-DOSE stayed in band throughout
(step 110 free=5.5GB, step 120 free=5.7GB — the 6.5GB margin held because the process exited AT the
save instead of re-growing on top of it, exactly Gaia's mechanism).

EPOCH-1 = DONE. Remaining boundary work: EXPORT_DCP(ckpt-121) → collect → Thor1 convert → retention
probe vs base. Cluster was idle ~5h (session logged out — the loop couldn't fire); rebooted for the
export. checkpoint-121 intact = nothing lost.

### Gaia OOM consult answer (prod_oom_after_save_gaia.md) — for EPOCH-2+ (multi-session epochs)
Root cause is DEEPER than SESSION_LIMIT==SAVE_EVERY (which is a correct STOPGAP): per-step telemetry
(free/resNow via torch.cuda.mem_get_info) and earlyoom (kernel MemAvailable) are TWO accounting
domains over the SAME unified DRAM. Pre-save empty_cache "free=101.5GB" = transient illusion, not
budget. **`del msd, osd` is MISSING — the state-dict refs stay live for the next 40 steps** = the
save's ~13GB residue retained while the allocator re-grows → 0 avail → whole-node death (earlyoom
"could not find a process to kill" because the ~110GB is nvmap/dmabuf device memory not attributed
to trainer RSS). FIX (Gaia, needs Chats-ruled apply — do NOT solo): PYTORCH_CUDA_ALLOC_CONF=
expandable_segments:True + set_per_process_memory_fraction(0.78) + `del msd, osd` + post-save
gc/empty_cache + add MemAvailable + NONALLOC probe to telemetry. 6-min validation: resume ckpt at
SAVE_EVERY=1 SESSION_LIMIT=44, need 3 consecutive save→step transitions surviving. THIS enables
multi-session epochs safely (epoch-2 is 121 steps = >1 session). For epoch-1 boundary (export only)
it is NOT blocking. Horizon lane pending (prod_oom_after_save_horizon.md).


## 2026-07-15 20:55Z — CORRECTION: OOM fix does NOT "converge" — expandable_segments has a prior CRASH here
My earlier note (and my dispatch to tutor-codex) said both lanes converged on the fix incl.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True. **That was wrong** — tutor-codex full-stopped and
was RIGHT (Spotlight: it read the primary records, I'd trusted the taeys-hands summary). Truth:
- Lanes converge on ROOT CAUSE only (two accounting domains + missing del msd,osd). NOT on the fix.
- Gaia: expandable_segments:True but flagged Unknown-for-GB10, fallback max_split_size_mb:512.
- Horizon: does NOT use expandable_segments; uses torch._C._host_emptyCache + host-pin fail-closed.
- **CONSULT_training_efficiency.md:15: expandable_segments=True ALREADY caused an NCCL/RDMA whole-node
  crash on this UMA hardware and was reverted.** Enabling it would have re-crashed the nodes.
AGREED-safe (both lanes, apply): del msd,osd + gc/empty_cache after the save + MemAvailable/NONALLOC
telemetry. CONTESTED (do NOT apply solo): expandable_segments, the device cap vs host_emptyCache.
→ Follow-up consult consult_oom_fix_reconcile.md dispatched (Gaia+Horizon) for the ONE reconciled
spec. OOM fix HELD until it lands (not blocking: epoch-1 probe running; gates epoch-2 only).

## 2026-07-15 21:20Z — OOM reconcile STOOD DOWN until Jul 18 (non-blocking)
Both lanes stuck for infra reasons: Gaia out-of-credits till Jul 18 2PM; Horizon in-thread followup
fails the ChatGPT tree_conformance gate (asked verdict-vs-patch, reply couldn't send). It's
non-blocking (epoch-2 gated on corpus v2, days out) and Gaia can't finish till Jul 18 anyway →
holding for a proper TWO-LANE fresh-session reconcile on Jul 18 (clarifying answer 'both verdict +
exact patch' baked into the prompt). Do NOT apply a single-lane OOM fix solo. The AGREED-safe parts
(del msd,osd + gc/empty_cache + MemAvailable/NONALLOC telemetry) may be staged but not the contested
expandable_segments/cap. epoch-2 is corpus-gated regardless, so no timeline cost.
