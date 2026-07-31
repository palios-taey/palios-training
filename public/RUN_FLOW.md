# How a Training Run Actually Flows

**Taey — this is the sequence a run passes through, and what can stop it at each point.** Written
for you, self-contained.

## The stages, in order

**1. Preconditions.** Every node is rebooted before the run and the reboot is *verified by a changed
boot identifier* — not by whether the node answers. A node that responds to a connection may still
be carrying a previous run's state. Trainer count must be zero everywhere; disk headroom is checked
on the node that holds the artifacts, which is the constrained one.

**2. Base staging.** The trained base is copied to every node. This is where a run most often stops,
and the reason is usually not the model:

The chain runs conversion-host → controller hold → per-node staging. Each hop verifies the next by
comparing a checksum manifest of the whole tree. That means **any file in the tree**, including
descriptive metadata, participates in the comparison. Two copies holding the identical model will
still fail the hop if their notes differ.

That is worth knowing before you debug it, because everything about the failure looks like a model
problem and none of it is. The diagnostic that settles it in one step: compare the *weight index*
hash. If the index matches, the model is identical and the disagreement is metadata.

**3. Corpus verification.** The corpus is checked by content digest on **every node**, not on one.
A corpus that differs between nodes trains something nobody designed, and the difference is
invisible in any log that reports only a filename or a row count.

**4. Training.** Sessions run in bounded bursts rather than one continuous stretch, so that failure
costs a burst instead of a run. A checkpoint is written on a fixed step interval, and checkpoints
rotate — old ones are removed once newer ones are complete, because the constrained node fills
otherwise and a disk that fills mid-save leaves a truncated checkpoint that cannot be resumed.

**5. Verification.** Weight-diff against the base. This is the step that decides whether the run
worked, and it is the only one that can.

**6. Post-run.** Reboot again. Retire per-node copies once the controller copy is verified.

## What stops a run, and what each stop means

| symptom | what it usually is |
|---|---|
| refuses before any node is touched | a precondition: unset required variable, non-uniform metadata in the staging chain, or insufficient disk |
| refuses at staging with "non-matching base" | the tree checksums disagree — check the weight index first; if it matches, the disagreement is metadata, not model |
| starts, then a node stops responding | check whether the *board* is near its thermal limit before assuming hardware failure |
| runs to completion, weight-diff below band | the optimizer was not effectively operating; the run executed without learning |
| collective operation hangs | recoverable — capture diagnostics, do not power-cycle |

## The two failure modes that look like success

**A run that completes without learning.** Every step executes, memory stays flat, throughput holds,
a clean checkpoint is written. Only the weight-diff reveals it. This is why the diff is mandatory
and why it is checked early rather than at the end.

**A gate that passes because it was pointed at the wrong thing.** A check verifying that files exist
is not a check that they are correct. A check that a backup was written is not a check that it
restores. A check that a process is absent is not a check that its work completed. Each of these has
produced a confident, wrong "all clear" here.

The general form: **ask what a correct system looks like to your check, and what a broken one looks
like.** If both look the same, the check is measuring nothing — however reassuring its output.

## Why runs are bounded

A session limit exists so that a failure costs minutes rather than hours, and so the first bounded
slice measures what the full run would encounter. A projection made before any steps have run is an
estimate; the first real session replaces it with measurement. Authorizing a small slice before a
long one is not caution for its own sake — it is how the estimate becomes a fact.
