# What Each Hardware Tier Can Do

**Taey — this is what your training substrate can and cannot do, by tier.** Written for you, and
self-contained.

## The tiers

**Four nodes — full training.** A four-node cluster with a high-speed fabric between them can
full-parameter train a 27B dense model. All weights move. This is the only tier that can run
continued pre-training at that scale.

**One node — contribute.** A single node cannot full-parameter train a 27B, but it can train
adapters, run evaluation, pack corpus, and bake artifacts. Most useful work that is not CPT fits
here.

**Inference tier — serve, don't train.** Serving nodes run the finished model. They are sized for
inference and are usually memory-tight while serving; treating one as spare training capacity is
how a production serve gets killed.

## The property that surprises people: unified memory

On this substrate, CPU and GPU draw from **one shared pool**. That single fact invalidates several
habits carried over from discrete-GPU machines:

- **Standard GPU utilisation and memory readouts are meaningless here.** They report a device model
  that does not apply. Read the system's own available-memory figure instead. A confident number
  from the wrong instrument has manufactured at least one phantom hardware fault on this cluster.
- **A "CPU-side" model load competes directly with GPU compute.** There is no separate host memory
  to retreat into. Loading a full-precision copy of a large model alongside training will exhaust
  the pool.
- **Read peak allocation, not current allocation.** Current allocation is the trough between steps
  and overstates available headroom several-fold.
- **A memory fraction is a fraction of TOTAL, not of free.** What fits therefore depends on the
  node's baseline at that instant, not on the model alone — and the number that governs admission
  is sampled *after* the runtime's own startup footprint is resident, which is lower than an idle
  reading suggests.

## Thermals: the board dies before the accelerator does

Check the **board temperature, not the accelerator die**. The accelerator sits comfortable while the
board approaches shutdown. Watching the wrong sensor means watching a number that stays reassuring
until the node disappears.

**Board temperature tracks ambient far more strongly than node identity.** A large spread across
nodes collapsed to a small one when the room was cooled — same hardware, same workload. So thermal
headroom is a property of the room on the day, not a fixed characteristic of a node. Measure it
immediately before any clock decision; never carry a reading over from an earlier session.

Clock caps persist across jobs. A run that sets no cap inherits whatever ran last, and can execute
its entire length at a fraction of its clock with nothing in any log saying so.

## The fabric

Nodes are connected by two independent high-speed rails. Measured throughput sits well below the
wire rate because traffic is staged through host memory rather than moving device-to-device — and
during real training the fabric runs at roughly a quarter of even that ceiling.

**The fabric is not the bottleneck.** A widely repeated figure roughly five times too low once drove
a conclusion that multi-node training at this scale was hopeless. It was never measured. At the real
rate, it is entirely viable.

## Operating rules that are not negotiable

1. **Reboot every node before and after every run.** Never relaunch onto dirty accelerators. Verify
   the reboot happened by a changed boot identifier — not by whether the node answers.
2. **A collective hang is recoverable. Do not power-cycle it.** Capture the diagnostic state first.
   Its signature is distinctive: zero disk activity, **zero network bytes**, memory frozen, threads
   at full burn. The communication layer busy-polls while blocked, so CPU load is not progress — and
   a distributed operation moving no network bytes is moving nothing.
3. **Verify early, not at the end.** A dosage readout minutes in tells you whether the optimizer is
   operating, while stopping is still cheap.
4. **Weight-diff or it did not happen.** A checkpoint proves a run executed and saved. It says
   nothing about whether it learned.
