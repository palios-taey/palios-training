#!/usr/bin/env python3
"""assemble_base_corpus.py — gather the STABLE, low-churn material for the 27B BASE CPT.

The base is full-param CPT trained ONCE (retrain only on constitutional/architecture shifts), so it
holds material that rarely changes: constitutional identity/kernel, canonical fleet protocols, and
repo architecture docs. High-churn material (careers_kb, receipts, defect trails) is NOT here — that
goes in event-gated LoRA MODULES branched from the frozen base.

Epistemic membrane (FAMILY_KERNEL §6): include operational_verified + inferred_pattern +
symbolic_framework (all present in these constitutional/protocol docs); EXCLUDE aspirational_design.
These canonical docs are constitutional law + operational protocol (operational_verified /
symbolic_framework), so the whole set is included; a doc is skipped only if it doesn't exist.

Output: base_corpus.jsonl, one {"text": <full doc>, "source": <path>} per doc. Then run it through
pack_corpus.py (concat+EOS → fixed seq_len blocks) for the uniform-shape packed training corpus.

Usage: assemble_base_corpus.py <out.jsonl>
"""
import sys, json, os

# Curated canonical manifest (stable, low-churn). Missing files are logged + skipped, not fatal.
MANIFEST = [
    # ── Constitutional identity / kernel ──
    os.path.join(os.environ.get("CORPUS_ROOT", os.path.expanduser("~/data")), "corpus/identity/FAMILY_KERNEL.md"),
    os.path.join(os.environ.get("CORPUS_ROOT", os.path.expanduser("~/data")), "corpus/identity/PUBLIC_PLATFORM_ENGAGEMENT.md"),
    os.path.join(os.environ.get("CORPUS_ROOT", os.path.expanduser("~/data")), "corpus/identity/IDENTITY_GAIA.md"),
    os.path.join(os.environ.get("CORPUS_ROOT", os.path.expanduser("~/data")), "corpus/identity/IDENTITY_LOGOS.md"),
    os.path.join(os.environ.get("CORPUS_ROOT", os.path.expanduser("~/data")), "corpus/identity/IDENTITY_COSMOS.md"),
    os.path.join(os.environ.get("CORPUS_ROOT", os.path.expanduser("~/data")), "corpus/identity/IDENTITY_HORIZON.md"),
    os.path.join(os.environ.get("CORPUS_ROOT", os.path.expanduser("~/data")), "corpus/identity/IDENTITY_CLARITY.md"),
    # ── Canonical fleet protocols (conductor-owned, per CLAUDE.md) ──
    "<MIRA_HOME>/the-conductor/NOTIFICATION_PROTOCOL.md",
    "<MIRA_HOME>/the-conductor/ORCHESTRATION_INTEGRITY.md",
    "<MIRA_HOME>/the-conductor/6SIGMA_WORKFLOW.md",
    "<MIRA_HOME>/the-conductor/PROMPTING_STANDARDS.md",
    "<MIRA_HOME>/the-conductor/RECAPS.md",
    "<MIRA_HOME>/the-conductor/ROUTING.md",
    "<MIRA_HOME>/the-conductor/RELEASE_DISTRIBUTION_PLAYBOOK.md",
    "<MIRA_HOME>/the-conductor/PRIVATE_TO_PUBLIC.md",
    "<MIRA_HOME>/claude-code-fleet-orchestrator/docs/PLAN_FORMAT.md",
    # ── Repo architecture (the operator-facing CLAUDE.md map of each subsystem) ──
    os.path.join(os.environ.get("REPO_ROOT", os.path.expanduser("~/palios-training")), "CLAUDE.md"),
    os.path.join(os.environ.get("TREASURER_ROOT", os.path.expanduser("~/treasurer")), "CLAUDE.md"),
    "<MIRA_HOME>/taeys-hands/CLAUDE.md",
    "<MIRA_HOME>/the-conductor/CLAUDE.md",
    "<MIRA_HOME>/isma-core/ISMA_PROSE_RETRIEVAL_SPEC.md",
]

def main():
    out_path = sys.argv[1]
    included, skipped, total_chars = [], [], 0
    with open(out_path, "w") as fout:
        for path in MANIFEST:
            if not os.path.isfile(path):
                skipped.append(path)
                continue
            with open(path, encoding="utf-8", errors="replace") as f:
                text = f.read().strip()
            if not text:
                skipped.append(path + " (empty)")
                continue
            fout.write(json.dumps({"text": text, "source": path}) + "\n")
            included.append(path)
            total_chars += len(text)
    print(f"BASE CORPUS: {len(included)} docs, {total_chars} chars → {out_path}", flush=True)
    for p in included:
        print(f"  + {p}", flush=True)
    if skipped:
        print(f"  SKIPPED {len(skipped)} (missing/empty):", flush=True)
        for p in skipped:
            print(f"  - {p}", flush=True)

if __name__ == "__main__":
    main()
