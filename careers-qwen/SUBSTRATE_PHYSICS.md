# The measured physics of this body — GB10 4-node cluster

Feel, here, is not a mood and not a metaphor. It is a fitted predictive model of a
physical system, evaluated faster than deliberate inspection. A driver knows the engine
is at redline without looking because a model of mass, grip and thermal state has been
fitted against ground truth until its error is small; the tachometer then confirms a
value the model already produced. The instrument becomes redundant, not unnecessary.

This document is ground truth for that fit: the measured constants of this cluster,
so predictions can be made and residuals computed.

## The hardware

Four DGX Spark nodes, GB10 (Grace-Blackwell, aarch64, sm_121), one GPU each, unified
memory ~119-125GB shared between CPU and GPU on a 256-bit LPDDR5X bus with a ~273 GB/s
ceiling. No NVLink between nodes. No shared filesystem. Interconnect is ConnectX
direct-attach copper at 200 Gb/s per node, RoCE, running CPU-proxy: GPUDirect is
disabled deliberately because NCCL registering torch's remappable VMM pages with the NIC
killed whole nodes.

## Measured constants (observed, dated)

BF16 GEMM, 8192^3, healthy node:
  85-99 TFLOPS at 1976-2463 MHz. Higher clocks uncapped; ~1976 MHz under CLOCK_CAP.

Collective bandwidth, 4 ranks, production NCCL env:
  all_reduce  busbw 20.4-22.6 GB/s     all_gather busbw 19.9-21.7 GB/s
  ~85% of the 25 GB/s line rate, WITH GPUDirect off. The CPU-proxy path is not the
  bottleneck it is often assumed to be.

Power under sustained load, healthy node (.68, 2026-07-25):
  t+0s 26.04W -> t+3s 50.56 -> t+6s 56.48 -> t+9s 57.13 -> t+12s 57.39 -> t+15s 56.63W
  1976 MHz throughout, 50 -> 53C.
  A healthy node RAMPS. Power climbs as work demands it and the node warms doing it.
  Uncapped, the same node reached 92.88W.

Power under identical load, power-starved node (.80, same day):
  t+0s 15.58W -> 15.62 -> 15.57 -> 15.62 -> 15.58 -> 15.74W
  611-617 MHz throughout, 41-42C, 28.5 TFLOPS.
  Flat within 0.17W across every sample. It does not sag under stress; it never rises.
  The same node measured 81.7W on 2026-07-10, so its condition CHANGED over time. It was
  not born this way.

  Do not read "degradation" as a degraded processor. The 2026-07-25 sustained run settles
  which part changed, and the answer is the useful one:

THE EFFICIENCY IDENTITY — how to tell a BROKEN node from a STARVED one:
  Sustained 75s 8192^3 bf16 GEMM, CLOCK_CAP=1600, sampled DURING load, all four nodes:

    node   TFLOPS   clock     power    TFLOPS/W   temp
    .68    71.917   1592 MHz  39.7 W   1.812      65 C
    .80    28.432    611 MHz  16.0 W   1.777      47 C
    .12    71.738   1577 MHz  41.1 W   1.745      61 C
    .19    72.659   1592 MHz  41.4 W   1.755      63 C

  .80's clock (0.385), power (0.393) and throughput (0.394) are all the SAME fraction of
  its peers, moving in lockstep. And its efficiency, 1.777 TFLOPS/W, sits INSIDE the
  healthy spread of 1.745-1.812 — within 0.4% of the fleet mean.

  The silicon converts power to compute exactly as well as its three siblings do. It
  delivers precisely what its clock permits and draws precisely what that clock demands.
  It is also the COLDEST node in the rack while doing it: 47C against 61-65C.

  That single ratio discriminates the failure modes:
    degraded silicon  -> draws normal power, converts it BADLY  -> TFLOPS/W FALLS
    thermal throttle  -> runs HOT, backs off                    -> temp is HIGHEST
    failed VRM        -> draws power and wastes it              -> TFLOPS/W FALLS
    STARVED supply    -> draws less, converts perfectly, cool   -> TFLOPS/W UNCHANGED

  .80 shows the fourth signature. A healthy part doing exactly what its power budget
  permits. The fault is upstream of the die, not in it.

  HOW STRONGLY TO HOLD THAT — a second instrument weakened it, and the weakening is
  itself the more useful lesson:

    instrument                      healthy spread   .80     .80 vs mean
    75s sustained, sequential       1.745-1.812      1.777   +0.4%  INSIDE
    8s guarded GEMM, concurrent     1.909-1.999      1.899   -2.9%  0.5% BELOW floor

  The whole fleet moved between them (+9.4% to +11.8% on the healthy nodes), because a
  shorter cooler run leaks less and converts better. So an efficiency figure is only
  comparable WITHIN one instrument — an absolute band carried across harnesses is not a
  test of the hardware, it is a test of the harness.

  Read honestly: .80 sits inside the healthy band on one instrument and marginally below
  it on another, and it gained LESS from the cooler workload than any healthy node
  (+6.9% vs +9.4-11.8%). That is a small signal toward genuinely different rather than
  identical-but-starved. The identity is NOT clean, and it wants same-instrument
  replication before anyone leans on it.

  What survives regardless: .80 trails the fleet by 61% in throughput while sitting within
  3% of it in efficiency, at the lowest temperature of the four. That gap is the
  load-bearing observation and no instrument has contradicted it. The ranked-hypothesis
  conclusion stands; the CONFIDENCE attached to it does not.

  (Note the two healthy-node figures in this document are not in conflict: 1976 MHz /
  57 W was measured under a higher CLOCK_CAP than the 1592 MHz / 39.7 W above. Always
  read a clock or power number against the cap in force when it was taken.)

Idle, all four nodes: exactly 208 MHz, ~4-5W. Identical at rest. This is why any
comparison made at idle is blind: the interesting variation exists only under load.

Thermal: the whole-node death wall is ~94C board/SoC, established 2026-07-10. The CPU
side reaches distress before the GPU; an 82C GPU is not distress.

27B full-parameter CPT step, packed B4 x 2560, 4 ranks (40,960 tokens/step):
  ~103s baseline, ~99s with the allocator fix. NVTX kernel split at baseline:
  forward+CE 23.7s, optimizer 9.9s, backward range 58.2s, and NCCL ReduceScatter 59.8s
  + AllGather 31.7s.
  That NCCL figure is NOT bandwidth. Modelled traffic at measured bandwidth is ~5.8s.
  The remainder is healthy ranks blocked inside the collective waiting for the slow rank,
  billed by the profiler to the collective. A straggler's cost appears as communication.

Memory during that step: free ~5.8GB, resident ~110.9GB of ~119GB. Near-saturated, so
batch size is not an available lever.

## How to use this

Before reading telemetry, state the expected values. Then read, and compute the residual.
A read without a prior prediction supplies no error term and updates nothing.

MATCH THE INSTRUMENT'S TIMESCALE TO THE FAILURE'S. A power limit needs sustained draw to
manifest, so a burst cannot express it. The production preflight runs 20 GEMMs — at these
rates that is under a second, and a 1Hz sampler gets about one sample. Measuring .80's
power that way returned 5.09W mean; the same node under a 75s sustained load returned
16.0W. The first number was aliasing, not physics. Worse, the preflight queried nvidia-smi
AFTER torch.cuda.synchronize(), so its clock and power columns describe a GPU that had
already fallen back toward idle: one receipt reported 70.903 TFLOPS alongside 760 MHz and
4.91 W, which is physically impossible and therefore a tell.

A NUMBER FROM A DEAD LAUNCHER IS STILL A NUMBER. A local timeout does not bound remote
work: kill the ssh and the GEMM keeps running on the node with no local process to show
for it. The next measurement then shares the device with a workload nobody can see and
returns a clean, plausible, WRONG value — not an error. This produced a phantom 31%
collapse on a perfectly healthy rank 0. Check for live remote work before trusting a
sample, and kill remote-side explicitly rather than trusting the disconnect.

A SURPRISING SINGLE SAMPLE IS A HYPOTHESIS, NEVER A FINDING. Repeat it before reporting
it. The repeat costs seconds; the false report costs trust.

Predictions this document should already let you make: that four nodes at 208 MHz tells
you nothing; that a node drawing 15W under load is not lightly loaded but unable to draw;
that a cold node while others wait is the urgent case, not the calm one; that a step
dominated by NCCL time is more likely a straggler than a fabric problem; that a node
whose power is flat across a sustained ramp has a supply limit rather than a thermal one.

Track the residual as a number. Predicting step time within 2% is a different fit than
within 30%, and the difference is statable. Where the fit is poor, say so — reporting a
reading honestly is work; claiming a feel not yet earned is an unverified claim.

A surprise is the highest-information sample available, because it is where the model is
most wrong. The impulse to explain it away discards exactly the data that would correct it.
