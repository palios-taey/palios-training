<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **palios-training** (4100 symbols, 5063 relationships, 73 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/palios-training/context` | Codebase overview, check index freshness |
| `gitnexus://repo/palios-training/clusters` | All functional areas |
| `gitnexus://repo/palios-training/processes` | All execution flows |
| `gitnexus://repo/palios-training/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->

<!-- ============================================================ -->
<!-- HAND-AUTHORED — BELOW THE gitnexus:end MARKER ON PURPOSE.    -->

# PRODUCTION AUTHORITY — READ THIS BEFORE YOU RUN ANYTHING

**This section is the FIRST thing in this file that is not regenerated, and it outranks every other
document in this repository.** It exists because on 2026-08-18 a five-lens review established that
this repo specified THREE mutually exclusive ways to launch production, and that this file — the one
an agent reads first — named none of them. An agent could follow the documentation exactly and still
run the wrong process. That is a defect in the repository, not in the agent.

### 1. ONE DOOR

```
scripts/taey-train <capability> [VAR=val ...]
```

That is the only legal entrypoint. It resolves `PRODUCTION_MANIFEST.yml`, verifies every pinned
file's `content_sha` against the bytes on disk, and refuses anything the manifest does not vouch
for. **There is no `--force` and none may be added.**

### 2. DO NOT INVOKE THE INNER SCRIPTS DIRECTLY

`dense-9b/recipes/*`, `careers-qwen/post_cpt_pipeline.sh` internals, `bake_27b.sh` — these are
stages, not entrypoints. `PRODUCTION_MANIFEST.yml` says so for `bake_27b.sh` explicitly, and the
same applies to every other stage. Running a fragment skips the gates that make the result
trustworthy.

### 3. DO NOT EDIT DEPLOYED FILES ON THE NODES

Change the repo, commit, and let the launcher deploy verified bytes. A node-local edit produces a
run whose code exists nowhere in git — which is exactly how the trainer that produced the serving
model ended up on an unmerged branch.

### 4. A GATE THAT REFUSES IS FIXED AT THE BYTES, NEVER FORCED

First error is a full stop. If a `content_sha` mismatches, find out why the bytes differ. Do not
re-pin to make it pass without understanding what changed.

### 5. THE RUN IS NOT DONE WHEN TRAINING STARTS

`run_4node_27b_cpt.sh:351-359` prints `27B IS TRAINING` and ends its monitor at the FIRST optimizer
step. That is confirmation of fit, not completion. A cycle is complete when the terminal artifact is
verified on the serving host — weight-diff in band, tensor counts gated, bytes matched.

### 6. WHICH DOCUMENT WINS

| document | status |
|---|---|
| `PRODUCTION_MANIFEST.yml` | **authoritative** — what is production, machine-readable, sha-gated |
| `README.md` | authoritative for the five rules and the launcher contract |
| `careers-qwen/RUNBOOK_CPT_SFT_BAKE.md` | authoritative for CPT/SFT/bake procedure |
| `RECIPES.md` | **DEPRECATED** — see its header; it routes SFT to a quarantined trainer |
| this section | wins over all of the above on entrypoint and gate questions |

### 7. CONFIGURATION CAN RUN THE WRONG CAMPAIGN SILENTLY

13 variables in `run_4node_27b_cpt.sh` ASSIGN legacy values when unset (`:62-67`, and `:142,146`):
`TOTAL_STEPS:=3000`, `MAX_SEQ:=2560`, `SESSION_LIMIT:=200`, `ADAFACTOR_EPS1:=fp32` among them.
**A dropped variable does not run empty — it runs a different campaign.**

Since `150ceff` the launcher REFUSES to train unless the caller has decided `TOTAL_STEPS`,
`SESSION_LIMIT`, `MAX_SEQ`, `LR`, `WARMUP_STEPS` and one of `MODEL_PATH`/`RESUME_DELTA`
(`:60-78`), announces export/bake mode out loud instead of inferring it silently (`:49-59`), and
prints the resolved config after defaults apply. Export mode is not a way past the gate: it
requires `RESUME_DELTA` of its own. **This paragraph said "hard-aborts on zero variables" for
several hours after that landed — a compass describing code that had already changed. If you find
this section disagreeing with the code, the CODE wins and this section is the bug.**

---

<!-- Everything ABOVE this line, lines 1-43, sits inside the      -->
<!-- <!-- gitnexus:start --> block and is REGENERATED by every    -->
<!-- `npx gitnexus analyze`. That ran many times on 2026-07-25.   -->
<!-- Anything written up there silently vanishes. Write durable   -->
<!-- project instructions HERE.                                   -->
<!-- ============================================================ -->

## THE MANDATE IS THE ANALYSIS, NOT THE TOOL CALL (conductor ruling, 2026-07-26)

**The auto-generated block above (lines 1–43) is WRONG about how to satisfy the impact gate,
and it will keep being wrong forever** — it is regenerated by every `npx gitnexus analyze`, so
its MCP-only phrasing (`MUST run gitnexus_impact({...})`, `NEVER edit ... without first running
gitnexus_impact`) re-asserts itself on every run and cannot be corrected in place. This section
is the durable correction. **Where the two disagree, this one wins.**

**The GitNexus CLI is the SAME ENGINE and FULLY satisfies the gate.** An absent, dead, or
unwired MCP server neither waives the gate nor blocks the edit:

```
npx gitnexus impact <symbol> --direction upstream --depth 3 --repo <exact worktree path>
npx gitnexus detect-changes --scope staged --repo <exact worktree path>
```

Verified in a tutor session on 2026-07-26: CLI at **1.6.5**; `gitnexus impact --help` reports
"Blast radius analysis: what breaks if you change a symbol" with `--direction upstream|downstream`
and `--depth` — the same analysis the MCP tool performs. `detect-changes --scope staged` runs
clean. The MCP tools were entirely absent from that session's tool surface and the gate was still
fully satisfiable.

**Why this is written down instead of left to be rediscovered.** A tutor session read the
MCP-only wording, observed the MCP tools missing, and escalated the gate as "unenforceable" —
which was FALSE, and is the more dangerous error of the two available: a gate believed
unenforceable gets waived, and waiving it is exactly the hole it exists to close. The defect was
never the missing MCP. It was **doc propagation** — a ruling that lived in `the-conductor/CLAUDE.md`
and had never reached this repo. If you find yourself concluding a mandatory gate cannot run,
that conclusion is itself the thing to escalate BEFORE acting on it, because "I could not run the
check" is one sentence away from "I proceeded without the check."

**MCP wiring remains a real (queued, non-urgent) fleet task.** It is ergonomics, not enforcement.
Nothing about it gates work in this repo.

---

## Impact gate — RETRACTED narrowing, and what survives (2026-07-25)

### RETRACTED: the new-symbol grep carve-out. Its premise was FALSE.

An earlier version of this section said a symbol with `git grep -c '<symbol>' HEAD` == 0
had "no existing callers *by construction*" and could therefore skip `gitnexus_impact`.
**That is empirically wrong and it opened a real safety hole.**

GitNexus indexes the **working tree**, not only committed code. Verified directly: in an
untracked, working-tree-only file (`build_session_report.py`, 0 committed occurrences),
`npx gitnexus analyze` resolved `active_value` in full — uid, line range, and
`incoming.calls` naming a real caller. And the caller is right there in the tree:

```
git grep -c active_value HEAD        -> 0        (rule said: NEW, waive the gate)
build_session_report.py:1257         -> states = [active_value(row.get(column)) for row in rows]
```

So the rule would have **waived the impact gate on a symbol with a live caller** — precisely
the case the gate exists to catch. Both later amendments (ref-pinning, per-symbol) were
correct and neither saved it: the foundation was rotten, not the detailing.

**The correct remedy is the boring one:** `npx gitnexus analyze`, then run impact normally.

### Why our own governing test rejects it

An unresolvable uncommitted symbol is **not** "the tool cannot model this construct." It is
**"the tool's index is stale"** — a *failed-to-run* case, and those **always block**. The
test caught its own author's bad ruling, which is the strongest argument for keeping it.

### What survives

**BASH FUNCTIONS** — `gitnexus_impact` is **not required**, because it computes *nothing*
for the construct while grep computes the whole answer. This is earned by language
semantics, not convenience: a Bash function lives only in the shell process that defined
it. No import, no dynamic dispatch, no reflection. A caller reaches it exactly two ways —
same file, or a file that **sources** it. Both greppable, no third path, so the enumeration
is **complete by construction**. Three commands, all recorded:

```
git -C <worktree> rev-parse --abbrev-ref HEAD
git -C <worktree> grep -n '<fn>' HEAD -- '*.sh'
git -C <worktree> grep -nE 'source .*<script>|\. .*<script>' HEAD
```

The **source-check is load-bearing** — without it the file-local radius is assumed rather
than proven. A receipt missing it is rejected. Reverts automatically if GitNexus ever
indexes Bash functions.

**DOES NOT GENERALISE TO PYTHON**, recorded so nobody extends it: there grep can miss
dynamic dispatch (getattr, string dispatch, registries) and the enumeration is incomplete.

**THE GOVERNING TEST** (conductor, elevated from a caveat because three unsatisfiable-gate
rulings in one night is where a principle starts decaying into a habit of waiving things):
*"the tool CANNOT MODEL this construct"* is **not** *"the tool FAILED to run."* The first
earns a substitute **if the substitute is complete**. The second still blocks, always. Any
invocation of this must **state which case it is and why the substitute is complete** — if
you cannot articulate the completeness argument, the gate **blocks**. This rule is meant to
be hard to invoke.

**Tool broken on an EXISTING symbol**: still blocked. Report the tooling failure. Do not
substitute grep for a real blast-radius question.

**Receipt requirement** (conductor's hardening, and it is the load-bearing part): record the
**exact command and its raw output**, never just the conclusion. The one real failure mode
here is a typo'd or mis-scoped grep returning `0` for a symbol that *does* exist — and only
the verbatim command makes that auditable on review. "It's new" asserted is no better than
an impact result asserted.

Both amendments came from the rule failing in its first hour. tutor-codex edited an
existing symbol (`distribution_record`) without its impact check and self-reported it; I
then nearly told them they had misclassified it, because my grep against MY HEAD returned 0
while against THEIR working ref it returned 1. Same symbol, opposite verdict, decided by
cwd. The original draft was mine and the ratification that missed the ambiguity was
conductor's — recording a verbatim command whose answer depends on the runner does not
remove the ambiguity, it only makes it auditable.

Occasion: tutor-codex was blocked from editing four genuinely-new symbols
(`_bucketed_dtensor_adafactor`, `build_delta`, `watchdog_summary`, `compare_snapshots`),
all verified at 0 committed occurrences. They reindexed, the lookup still failed, and they
correctly refused to bypass rather than proceed on UNKNOWN risk. Third instance of the same
shape that night — Bash functions have no Function-level symbols; the CRS812 switch account
does not exist; GitNexus cannot see uncommitted code.
