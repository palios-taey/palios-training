# Autonomous release contract

**Status:** additive, training-side contract slice. It defines release evidence; it does not
replace the production launcher, change campaign quarantine, or authorize a manifest edit.

## Router

1. Start with an immutable
   [`campaign spec`](schemas/release/campaign-spec.schema.json). Its `content_sha` pins are the
   declared intent and its `consumer_plane` names the stable consumer aliases `taey` and `ep3`.
2. For each requested transition (`train`, `bake`, then `promote`), require a matching
   [`Hub decision receipt`](schemas/release/hub-decision-receipt.schema.json). Its exact
   `authority` object names the `the-hub` surface, a safe attributable Taey or Family Chat actor,
   signer identity, the fixed `taey-release` detached-signature namespace, and a trust-policy
   digest. The receipt binds the exact campaign-spec digest, repository commit, attributable
   evidence receipt, and subject artifact. Promotion additionally binds the rollback artifact and
   ordered consumer aliases `["taey", "ep3"]`. A rejected, absent, malformed, or mismatched receipt
   blocks the transition.
3. Append, never rewrite,
   [`lifecycle observations`](schemas/release/lifecycle-observation.schema.json), then collect their
   ordered digests in a [`lifecycle chain`](schemas/release/lifecycle-chain.schema.json). Genesis has
   no predecessor; every later observation names its predecessor. They record what occurred after
   authorization; they are not declared intent and cannot grant a later transition.
4. At the train/bake boundary, collect
   [`checkpoint manifests`](schemas/release/checkpoint-manifest.schema.json): each node contributes
   its own fragment, and the collector is represented by a digest-bound collector receipt.
5. Bind the checks required for the next decision with an
   [`evaluation contract`](schemas/release/evaluation-contract.schema.json).
6. Only after the approved promotion and complete evidence chain, generate the
   [`terminal release receipt`](schemas/release/terminal-release-receipt.schema.json). It is the
   sole terminal state object and references the evidence by digest rather than copying it.

The contract state is therefore `DECLARED → TRAIN_AUTHORIZED → TRAIN_OBSERVED →
BAKE_AUTHORIZED → BAKE_OBSERVED → PROMOTION_AUTHORIZED → RELEASED`. Any missing link is
`BLOCKED`, not an implied success.

## Authority and failure policy

The existing production authority still wins: `CLAUDE.md`, `PRODUCTION_MANIFEST.yml`, and
`scripts/taey-train` remain the only production launch path. This contract is deliberately not an
alternative entrypoint and does not implement trainer capture in this change.

Validation fails closed. **Schema validity is not authorization.** Consumers MUST verify the detached
OpenSSH signature in namespace `taey-release` against signer identities pinned by the named trust
policy, and MUST reject replay or freshness-policy violations. They must also verify every referenced
SHA-256 against the bytes or receipt it names and require exact equality of `campaign_id`,
`transition`, and campaign-spec digest across the chain. No schema accepts a user, human, or
`approved_by` field. Authorization is a `the-hub` decision with a Taey/Family Chat authority object,
grounded in a repository commit and attributable receipts.

`taey` and `ep3` are consumer-plane aliases only. Control-plane identity is always the bound
SHA-256 digest; an alias is never evidence that two artifacts are identical.

## Object ownership

| Object | Mutability | Purpose |
|---|---|---|
| campaign spec | immutable declared input | campaign pins, requested transition, and consumer aliases |
| Hub decision receipt | immutable decision evidence | transition authorization from The Hub |
| lifecycle observation / chain | append-only | observed transition progress, predecessor links, and ordered digest index |
| checkpoint manifest | collected immutable evidence | node-local fragments plus collector receipt |
| evaluation contract | immutable declared gate | assertions required before the next decision |
| terminal release receipt | generated once | digest-bound terminal release evidence |

The schemas specify individual data shape. `scripts/validate_release_chain.py` is the repository-side
fail-closed cross-object validator for a terminal receipt and its dereferenced records; signature
verification and record collection remain separate required consumers. Neither may modify
`PRODUCTION_MANIFEST.yml` authorization blocks or historical run configuration captures.
