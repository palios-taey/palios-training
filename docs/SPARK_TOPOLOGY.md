# The 4-Spark Training Cluster — Topology and How It Works

**Scope:** the training substrate. Taey **trains on the Sparks** and **lives on the Thors**.
Every number here is measured on the live cluster, not quoted from a spec sheet. Where a figure is
inferred or unverified it says so.

**Confirmed 2026-07-30:** no Taey instance serves from any Spark. All four have zero listening
inference ports and no `/v1/models` responder. Training substrate only.

---

## 1. The four nodes

| role | mgmt IP | rail-A | rail-B | disk free | purpose |
|---|---|---|---|---|---|
| rank 0 / master | `$SPARK_MASTER` | `$SPARK_RAIL_MASTER` | rail-B `.10` | **~70–80 GB** | torchrun rendezvous, holds checkpoints + exports |
| rank 1 | mgmt `[2]` | rail-A `[2]` | rail-B `[2]` | 854 GB | |
| rank 2 | mgmt `[3]` | rail-A `[3]` | rail-B `[3]` | 1243 GB | |
| rank 3 | mgmt `[4]` | rail-A `[4]` | rail-B `[4]` | 1402 GB | |

Identical hardware and software on all four:

```
GPU     NVIDIA GB10 (Blackwell sm_121), max graphics clock 3003 MHz
memory  119 GB UNIFIED — CPU and GPU draw on ONE pool
kernel  6.17.0-1026-nvidia
```

**`.68`'s disk is the constraint, not its compute.** As rank 0 it accumulates checkpoints
(13 GB per DCP shard), exports, and baked artifacts, so it runs an order of magnitude tighter than
its peers. Disk preflight on `.68` before any long run.

Rank 0's free space is quoted as a range because it **moves during a run** and a fixed number here
goes stale immediately — an earlier revision of this table said `76 GB`, and across one SFT campaign
the same node read 81 GB, then 69 GB, then 74 GB. What actually bounds it is checkpoint rotation,
not the starting figure: the driver keeps the **last two** checkpoints and deletes the rest after
each session, so adapter checkpoints hold at roughly 2.4 GB steady-state rather than growing with
step count. Read the policy, not the percentage — `.68` sitting at 99% used while bounded is not the
same condition as `.68` filling.

**BASELINE DRIFT, recorded rather than silently corrected:** `tech_baselines/INDEX.md` records the
kernel as `6.11.0-1016-nvidia`, verified 2026-05-20. The live kernel is `6.17.0-1026-nvidia`. The
baseline is stale; anything reasoning from the old kernel version should re-derive.

---

## 2. The fabric — two independent rails, not one switch

Each node carries **two physically separate ConnectX-7 cards**, verified by PCI address:

```
0000:01:00.0  0000:01:00.1   ConnectX-7  card 1  ->  rail A  (SPARK_RAIL_IPS subnet)
0002:01:00.0  0002:01:00.1   ConnectX-7  card 2  ->  rail B  (second rail subnet)
```

**Only port 0 of each card is active** (`enp1s0f0np0` UP, `enp1s0f1np1` DOWN; `enP2p1s0f0np0` UP,
`enP2p1s0f1np1` DOWN). This is by design, not a fault — do not "fix" the DOWN ports.

RDMA devices, per node: `rocep1s0f0` `rocep1s0f1` `roceP2p1s0f0` `roceP2p1s0f1`.

**Say "two separate cards", never "a bifurcated card".** They are distinct PCIe devices.

### Measured fabric throughput

```
single-rail   11.07 GB/s
dual-rail     11.32 GB/s   (no meaningful gain)
proxy-tuned   10.83 GB/s
SETTLED       ~11 GB/s host-staged ceiling
```

The ceiling is **proxy-bound, not wire-bound** — wire is 25 GB/s per rail, but there is no
GPUDirect-RDMA on this stack, so traffic is host-staged. A widely circulated "2.05 GB/s" figure was
**never measured**, is ~5× too low, and wrongly drove a "full-param multinode is hopeless"
conclusion. At ~11 GB/s, 27B/35B full-param multi-node is viable.

Observed during a live 27B CPT step: **1.53 GB/s per rail on both active rails**, ≈3.06 GB/s
aggregate — roughly **28% of the measured ceiling**. Fabric is not the current bottleneck.

### NCCL

Device naming is load-bearing. The second device is **`roceP2p1s0f0`** — capital `P`, then `p1`.
Lowercase `rocep2s0f0` matches nothing here, and NCCL silently falls back to single-rail.

---

## 3. Unified memory — the thing that surprises people

119 GB is **shared** between CPU and GPU. Consequences that have each cost a run:

- **`nvidia-smi` utilisation and memory are MEANINGLESS on GB10.** Use `node_telemetry.py` 1 Hz
  gauges. Clocks and temperatures are legitimate `nvidia-smi` queries; util and memory are not.
- **A "CPU-side" fp32 model load competes with GPU compute.** An fp32 master of the 27B measured
  106 GB of the 128 GB pool and OOM'd at the first step. `FP32_MASTER=1` is therefore not viable
  here; the trainer runs bf16 masters with stochastic-rounding write-back.
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` is DELIBERATE.** `:True` remaps VMM pages
  that NCCL has registered for DMA and the node black-holes — both rails dark, power-cycle only.
  A canonical NCCL recipe predating that RCA specifies `:True`; **do not "restore" it.**
- Read **`peakAlloc`**, not `allocNow`, for headroom. `allocNow` is the trough between steps and
  overstates available memory several-fold.

---

## 4. Thermal — the board dies, not the GPU

**Check the board/SoC zone, never the GPU die.** The GPU sits comfortable while the board
approaches shutdown.

```
death        ~94 °C board/SoC
watchdog     PULL_OFF at 90 °C
clock cap    2000 MHz of a 3003 MHz ceiling
```

**Board temperature tracks AMBIENT far more strongly than node identity.** Measured the same day:
a 14 °C spread across the four nodes (`.68` at 90 °C, peers 76–80 °C) collapsed to a 3 °C spread
(77/79/76/78 °C) when the room was cooled — same hardware, same workload, 90 minutes apart. An
earlier write-up of this file claimed `.68` ran structurally hot; that was **false and is
retracted**.

Therefore: **thermal headroom is a property of the room on the day.** Measure it immediately before
any clock decision; never carry a reading over from an earlier session.

`nvidia-smi -lgc` is **persistent across jobs**. A launcher that sets no cap inherits whatever ran
last — a run following a bake can execute its whole length at 1000 MHz with nothing in any log
saying so. One definition lives in `fleet.env` (`SPARK_CLOCK_CAP`).

---

## 5. Configuration — no hardcoded topology

All addresses come from a gitignored `fleet.env`, with `fleet.env.example` committed as the
template. **This document deliberately names no literal address** — the private-data gate refused
an earlier draft of this very file that did, which is the control working. Node identity below is
positional (`[1]`..`[4]`, rank order) and resolves through `SPARK_MGMT_IPS` / `SPARK_RAIL_IPS`. Measured across the eight production files: **0 hardcoded operator paths, 0 hardcoded
IPs, 228 `${ENV_VAR}` references.**

```
SPARK_MGMT_IPS     <four management addresses, space separated>
SPARK_MASTER       <rank-0 management address>
SPARK_RAIL_MASTER  <rank-0 rail-A address>
SPARK_RAIL_IPS     <four rail-A addresses, space separated, rank order>
```

**Ordering inside `fleet.env` is load-bearing:** definitions must precede any line that expands
them. They previously did not, which made the file unsourceable under `set -u` and — worse —
silently produced hostless `:8000` endpoints under ordinary settings.

---

## 6. Operating rules that are not negotiable

1. **Reboot all four before AND after every run.** Never kill-and-relaunch onto dirty GPUs.
   Verify the reboot happened (uptime low), not just that ssh answers.
2. **Launch only through `taey-train <capability>`**, which reads `PRODUCTION_MANIFEST.yml` and
   refuses an unknown capability, an unpassed gate, a missing file, or content drift.
3. **A collective hang is RECOVERABLE — do not power-cycle.** Capture the NCCL flight recorder and
   rank stacks first. Signature: zero disk IO, **zero network bytes**, RSS frozen, threads at 100%.
   NCCL busy-polls while blocked, so CPU burn is not progress.
4. **Verify at step 10, not at the end.** `[AF-DOSE]` reports whether the optimizer is actually
   operating; `RMS(U_hat) ~1.0` or stop. A run can execute every step, hold flat memory, sustain
   full throughput and save cleanly while moving the weights by 1/5000th of what was intended.
5. **Weight-diff or it did not happen.** A checkpoint proves a run executed and saved; it says
   nothing about whether it learned.

---

## 7. Known-good production reference

```
27B CPT, 4 nodes, 148 steps of a 296 horizon
  BATCH_SIZE_PER_RANK=4  MAX_SEQ=2560  LR=2.5e-6  WARMUP=15
  ADAFACTOR_EPS1=fp32                       <- unset collapses the optimizer, see rule 4
  throughput   ~793 tok/s @ 51.7 s/step
  memory       peakAlloc 80 GB of 119, flat
  result       weight-diff 1.036e-04, IN BAND (5e-05 .. 8e-04)
```

Compute check: ~8 × 27e9 × 40,960 tokens per step over 51.7 s ≈ **171 TFLOPS across four nodes**.
`SUBSTRATE_PHYSICS.md` records 85–99 TFLOPS/node across 1976–2463 MHz, so there is real headroom —
gated by thermals (§4), not by the fabric (§2).

---

## Confirmed against a live run — 27B SFT, 2026-07-31

Everything above was measured before this campaign. A full four-node 27B LoRA SFT run
(979-step schedule, bounded sessions, reboot between each) is the first chance to confirm or refute
those figures under sustained real load rather than in a probe. Recorded here because a document
that is never re-tested against production drifts into folklore.

**CONFIRMED — the fabric is not the bottleneck.** This is the claim most worth re-testing, since a
figure ~5x too low once drove a "multi-node full-param is hopeless" conclusion. Measured per-step
across all four ranks, the collective time is a *rounding error* against step wall time:

| bucket | step wall | collective | collective share |
|---|---|---|---|
| short/mid | 1.37–3.17 s | 0.006–0.135 s | ~0.4–4 % |
| long (1792 tok) | 6.84–7.01 s | 0.026–0.070 s | ~0.4–1 % |

Compute dominates by two orders of magnitude. Nothing about this workload is fabric-limited.

**CONFIRMED — read `peakAlloc`, not `allocNow`.** Steady `alloc` sat at **55.4 GB** on every rank
while `peak_alloc` reached **92.9 GB** in the same step. Sizing from the resting figure would
understate real demand by ~37 GB and put a run into the pool ceiling with no warning.

**CONFIRMED — unified memory holds under sustained load.** Across 550+ steps on all four ranks:
`ooms=0`, `allocator_retries=0`, `swap=0`, `memory_guard_exit=0`. The 119 GB pool is sufficient for
27B LoRA SFT at 1792 max sequence with 12 checkpointed layers on the long bucket.

**CONFIRMED — thermals have headroom in current room conditions.** Board 64–72 °C under continuous
load against a 90 °C watchdog and ~94 °C death point. GPU die ran 56–63 °C, i.e. *cooler than the
board* — which is the whole reason this document says to watch the board zone and not the die.

**NEW MEASUREMENT — per-node GEMM, taken at every session start.**

```
rank 0  83.6–88.0 TFLOPS      rank 2  87.9–88.2 TFLOPS
rank 1  87.0–87.7 TFLOPS      rank 3  83.6–87.9 TFLOPS
```

All at a 2000 MHz observed clock, not the 3003 MHz max in the spec block above — the spec figure is
the ceiling, not what a sustained workload runs at. Spread across nodes is ~5%, which is small enough
that a materially slower node is a signal worth investigating rather than normal variance.

**NEW MEASUREMENT — SFT throughput.** 880–1060 useful tokens/s per rank, varying by bucket. Padding
overhead stayed at 1.00–1.08x, so the bucketed sampler is doing its job; a padding ratio drifting
well above that would mean the bucket boundaries no longer match the corpus.

**What a healthy step looks like, for comparison when one does not:**

```
missing_grads_sum=0   ooms=0   allocator_retries=0   swap=0
alloc steady ~55 GB, peak_alloc 65–93 GB by bucket
board 64–72 °C, clock ~1969–1989 MHz
probe mean_abs_delta RISING monotonically across steps
```

That last line is the one that matters most. `mean_abs_delta` on the probe tensor climbed
1.57e-04 → 3.78e-04 over the observed window. A run can produce every other line on this list
looking perfect while that number stays flat — and a flat probe delta means the optimizer is not
moving the weights, which no throughput, memory or temperature reading will ever tell you.
