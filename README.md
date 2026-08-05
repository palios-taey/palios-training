# palios-training

Public executable training tooling for Taey: trainers, recipes, configs, launchers,
validators, and gates. **Training material (prompts, pairs, standards maps, provenance
ledgers, inventories, traces, receipts) is private** and lives only in a local governed
SFT authority. This repository never embeds that material and never falls back to an
operator home path.

## What stays public

| class | examples |
|---|---|
| trainers | `dense-9b/trainers/`, LoRA/SFT entry points |
| recipes / launchers | `dense-9b/recipes/`, `scripts/taey-train` |
| configs | model/train YAML under `dense-9b/configs/` |
| gates / scanners | `scripts/check_no_private_data.sh`, `scripts/check_no_private_training_material.sh` |
| instrumentation | telemetry and probes without private payloads |

## Private authority contract

Private inputs resolve only through a fail-loud configuration root:

```bash
: "${GOVERNED_SFT_ROOT:?set GOVERNED_SFT_ROOT to the local governed-SFT Git repository}"
[ -d "$GOVERNED_SFT_ROOT/.git" ] || { echo "GOVERNED_SFT_ROOT is not a Git repository" >&2; exit 1; }
```

No fallback to `/home/...`, sibling checkouts, or hard-coded machine paths is permitted.
An absent or invalid root is a configuration error, not a reason to search elsewhere.

## Launch door

```
scripts/taey-train <capability> [VAR=val ...]
```

Capabilities are listed in `PRODUCTION_MANIFEST.yml` as **public executable entry points
only** (script path + short description). Adjudicated private receipts and dose/ledger
state stay under `GOVERNED_SFT_ROOT`.

## Privacy gates

```
scripts/check_no_private_data.sh                 # jsonl / secrets / topology shapes
scripts/check_no_private_training_material.sh    # inventory path + content-hash denylist
scripts/public_export_gate.sh
scripts/structural_gate.sh
```

`check_no_private_training_material.sh` rejects any reintroduction of the 92 private-material
blob hashes or paths from the c164d35 privacy inventory. It does **not** reject generic
trainers, loaders, converters, recipes, configs, or validators.

## Non-goals

- Do not commit training rows (`.jsonl` corpora), system prompts, pair builders with embedded
  private payloads, standards maps, or run receipts.
- Do not rewrite history or publish private content.
- Do not mutate live Spark training checkouts from this repository alone.
