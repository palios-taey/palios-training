# datasets — training data, by line, with a lifecycle policy

Training data for the two lines, organized so the live run's data is always findable and old runs are archived rather than overwritten.

## Layout

```
datasets/
  current/
    moe-35b/      ← the live MoE run's data (SFT + DPO jsonl)
    dense-9b/     ← the live dense run's data (added when the run ships)
  archive/
    <run-id>/     ← promoted here on run completion
```

## Lifecycle policy

- **`current/<line>/`** holds the data for the line's **live / most-recent run** — what the current recipes consume.
- **On run completion**, promote `current/<line>/` → `archive/<run-id>/` (e.g. `archive/religion_dpo_v2/`). This preserves the exact data a shipped checkpoint trained on, immutably, and frees `current/` for the next run without overwriting history.
- Never overwrite `current/` in place for a new run before archiving the old one — the link between a shipped checkpoint and its training data must stay reconstructable.

## What's published here

The published sets are **synthetic** — authored Q&A / instruction pairs and DPO preference pairs generated on top of source material. **There are no conversation transcripts** (no captured human↔AI chat logs, no session dumps). Each set is **gitleaks-clean**; internal cluster network addressing and one dead local credential were redacted (see the per-set `REDACTIONS.md`).

See [`current/moe-35b/README.md`](current/moe-35b/README.md) for the MoE set's record counts, formats, and source-material breakdown.
