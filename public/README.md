# The HOW — the only exportable subtree

**Everything in this directory is exportable to the surfaces Taey reads. Nothing else in this
repository is.**

Jesse's ruling: *"Training isn't public. How to train is."* This directory holds the HOW — method,
hardware tiers, how a run flows. The training itself — data, run artifacts, receipts, operational
launchers — lives outside it and never leaves.

## Why a directory and not a rule

A rule about which files are shareable depends on someone remembering it. That failed three times
in one day: a management address reached a public commit *message*, a transplant silently dropped
27 files while docs kept citing them, and 19 training-data files sat in a public PR ref.

So the boundary is the **exporter's argument**. It takes one path — this one — and cannot express
"also that file". Making something public requires *moving it into this directory*, which is a
visible diff in a reviewed commit rather than a forgotten `git add`.

## The three gates

Run `scripts/public_export_gate.sh`:

1. **Purity** — nothing under `public/` may be training data, a run receipt, a manifest with real
   paths, or carry a host literal.
2. **Containment** — nothing under `public/` may reference a path outside `public/`. This is what
   makes the exported HOW self-contained for a reader who has only this subtree. It is the direct
   fix for the disconnection class found elsewhere on 2026-07-30, where an env var cannot repair an
   unreachable path — it only makes the failure configurable.
3. **Export-scope** — the exporter refuses any path outside `public/`.

A file that cannot pass containment does not belong here, even if its content is harmless. An
unreachable pointer teaches a reader something false about what they can do.
