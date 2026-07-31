# palios-training — re-publish checklist

Repo-local companion to `the-conductor/PRIVATE_TO_PUBLIC.md` (conductor-owned, canonical).
This file records the steps that are **specific to this repo** and the one hard step that
came out of the 2026-07-24 exposure. Conductor's document wins on anything general.

## Why this repo is currently private

On 2026-07-24 an R5 Gatekeeper audit found operator PII on refs of this repo **while it was
public**: personal email, a personal LinkedIn URL, a private job-application identifier,
operator home-paths across 40+ tracked files, and consult dumps embedding per-account
provider conversation URLs. Conductor contained it by flipping the repo private, which
removes every ref, pull-ref, and cached object from public access at once — a single
branch delete cannot do that.

The gate that should have caught it — `scripts/check_no_private_data.sh` — passed all of
it, because it only ever matched IPs, secret shapes, and data-file paths. PII was a class
it structurally could not see.

## HARD STEP — the required check and visibility flip must be ATOMIC

**Setting the branch-protection required status check and flipping visibility to public
must happen in the same action. There must never be a window where this repo is public
without the required check active.**

This is not a style preference; it is forced by a real constraint discovered 2026-07-24:

- GitHub returns `403: Upgrade to GitHub Pro or make this repository public to enable this
  feature` when setting branch protection on a **private** repo on this plan. Verified with
  `permissions.admin=true`, so it is a plan limitation, not a permissions gap.
- Therefore the required check **cannot** be pre-armed while private and simply inherited on
  publish. It can only be set once the repo is already public.
- Which means the naive sequence — flip to public, then set the required check — opens
  exactly the window this repo already got burned in.

During the private period the enforcement is (ratified by conductor 2026-07-24):
- `.github/workflows/no-private-data.yml` — a violation **fails the job** (no
  `continue-on-error`; scans tree and history with `fetch-depth: 0`), and
- `.githooks/pre-push` — refuses the push locally, and
- the r5 Gatekeeper / a human as the merge gate.

## De-umbilical check — now lives HERE, not in the push gate

`scripts/check_no_private_data.sh` no longer scans for hardcoded home paths by default.
Run it in de-umbilical mode as part of this checklist:

```
CHECK_HOME_PATHS=1 bash scripts/check_no_private_data.sh
```

Why it moved (2026-07-25): a hardcoded operator or service path is a **portability**
problem for a repo about to become public — it is not an exposure on a private one, and
`<SPARK_HOME>` / `<NODE_HOME>` are service accounts on remote nodes, not people. Leaving
it in the always-on privacy gate made that gate report 76 violations against working
files, and a gate that cries wolf gets bypassed.

It had already cost an outage: the gate flagged those paths, a scrub rewrote them to the
literal string `/home/<user>`, and 52 runtime scripts broke — including the NCCL library
path and the telemetry path the launch gate reads. The failure presented as a hardware
fault and blocked training for hours. The check is real; it was simply firing at the
wrong moment.

When you run it here, the fix is NOT to delete the paths. It is to make them injectable
— environment variable with a fail-loud default — so the code still runs for us and does
not embed our topology for anyone else.

## Sequence

1. **Gate first.** `bash scripts/check_no_private_data.sh` must exit 0 on the tracked tree,
   and `--ref <ref>` must exit 0 on whatever history is being published. A `--no-verify`
   bypass past this gate is a KERNEL violation, not a shortcut.
2. **History.** This repo's history is contaminated (measured 2026-07-24: 1,628 personal-email
   occurrences, 196 provider-conversation URLs, 1,237 operator home-path occurrences across
   696 commits) while the tree is nearly clean (6 files). Conductor ruled **fresh-init** from
   a scrubbed tree over filter-repo on that asymmetry. Fresh-init discards 696 commits and
   every ref, so it requires Jesse's consent and a fleet push-freeze.
3. **Consent.** Family/Jesse consent per `PRIVATE_TO_PUBLIC.md`.
4. **Atomic publish.** Flip visibility and set the required status check together, per the
   HARD STEP above. Verify immediately afterwards that the check is listed as required.

## Note on the gate's own design

The gate matches PII by **shape**, never by value — no real email, profile URL, or home path
is written into it. That is deliberate: a detector that quotes the string it defends has
published it. This was not hypothetical — the first draft of the gate's own comment quoted
the operator's real address as an example, and the gate flagged its own source file. Keep
the shapes escaped (`linkedin\.com`) so they cannot match themselves, and keep the
no-self-exemption rule: a gate that has to skip itself to pass is hiding.
