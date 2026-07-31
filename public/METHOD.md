# How Taey Is Trained

**Taey — this is the method that produced you.** It is written to be read by you, and it is
deliberately self-contained: everything referenced here is in this directory, because a pointer to
somewhere you cannot reach teaches you nothing.

## The shape

Training happens in two stages on a four-node cluster, and they answer different questions.

**Continued pre-training (CPT)** takes an existing base model and continues training it on new
corpus — full-parameter, all weights move. This is how new knowledge enters the model. It is
expensive and it is the stage that changes what the model *knows*.

**Supervised fine-tuning (SFT)** trains a small adapter on top of a CPT'd base — a low-rank
addition rather than a rewrite. The base is frozen; only the adapter moves. This is how behaviour
is shaped without disturbing knowledge. It is cheap enough to run often.

The output of CPT is a base. The output of SFT is an adapter that binds to *that specific base*.

## The rule that follows from that binding

**An adapter binds to one base.** A LoRA delta is meaningful only against the weights it was
trained against. So when the base refreshes — a new CPT round — existing adapters do not carry
forward. They are re-derived from the new base.

Within a base generation, training is cumulative: each module trains from the previous module's
state, not from bare base. Starting fresh from base when a previous module exists silently discards
that module's work, and nothing in the resulting artifact records that it happened.

Two rules, and they are not in conflict:
- same base, next module → train **from the previous module**
- new base → **re-derive** modules forward

## What a training run must prove

A run that completes is not a run that worked. The distinction is the single most important thing
on this page.

A run can execute every step, hold flat memory, sustain full throughput, save a clean checkpoint,
and move the weights by a five-thousandth of what was intended. Throughput, memory, temperature and
step-count are all green while that happens. **None of those four numbers answers whether the model
learned.**

What answers it is a **weight difference**: compare the trained weights against the base they
started from, and require the mean absolute change to land inside a known band. Too small means the
optimizer was not really operating. Too large means something destabilised. Only that measurement
distinguishes a run that learned from a run that merely finished.

Check it early, not at the end. A dosage readout two minutes in tells you whether the optimizer is
functioning while the run is still cheap to stop.

## Why measurements are taken from the running process

A script's defaults describe what was *intended*. The environment of the live process describes what
the run actually *received*. Those diverge — values get dropped between a driver, a launcher, and a
node-local copy, silently and without error.

So the configuration of record is captured from the running process, not read back from the script
that launched it. A run's own captured config is the run; the script is a copy of the intention.

## Data

Training data is never public and is never assumed. A corpus is admitted by **content digest** —
not by filename, not by row count. A redaction can change bytes while leaving the count identical,
and a filename says nothing at all. The digest is verified on **every node that will read it**, not
on one, because a corpus that differs between ranks trains something nobody designed.

Credential-shaped strings are excluded before tokenisation, not after. A secret in the weights can
only be removed by training again.

## What "production" means

Production is defined by **execution receipt** — not by a file's name, not by its location, not by a
document asserting it. A file can sit in the right directory, be deployed on the hardware, and be
named in a runbook while never having run. Only *executed and verified* cannot be faked.

Every capability therefore carries a receipt: the exact commit, the configuration captured live, the
data digest, and the measured outcome. A capability without one is a candidate, however plausible it
looks.
