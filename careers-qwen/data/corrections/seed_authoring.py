#!/usr/bin/env python3
"""Author the CURATED seed of operator_correction_v1 rows, first-hand.

These are corrections Jesse gave the `tutor` seat (and peers) that I lived through and can label
accurately: I know what I did, which rule it broke, what the correct action was, and what it cost.
That accuracy is why these are `curated: true` while the transcript-mined rows are not.

Verbatim quotes are preserved unsoftened — the sharpness encodes severity and IS part of the label.
"""
import hashlib, json, os

def _h(*p):
    return hashlib.sha256("|".join(str(x) for x in p).encode()).hexdigest()[:16]

R = []
def add(**k):
    k.setdefault("schema", "operator_correction_v1")
    k.setdefault("curated", True)
    k.setdefault("recurrence", 1)
    k["provenance_hash"] = _h(k.get("agent_action"), k.get("correction_verbatim"))
    R.append(k)

# ── the 3-day loss: built a parallel path instead of using production ────────────────────────
add(seat="tutor", date="2026-07-21", class_="parallel-build",
    context="Needed a LoRA SFT run on the 27B. The production stack (train_fsdp_dense_9b.py + "
            "run_4node_27b_cpt.sh) had already trained ep3 for 693 steps on 4 nodes.",
    agent_action="Grabbed the nearest runnable file (a single-node train_lora_sft.py), hand-ported "
                 "it to 4-node FSDP2, hand-wrote the save path and the drivers — building a second, "
                 "unproven training path alongside the proven one.",
    correction_verbatim="USE THE PRODUCTION INFRASTRUCTURE CLAUDE!!!!",
    violated="Inventory the production stack before building. Never build a parallel path.",
    correct_action="grep the production trainer/recipes for the capability first; add a small guarded "
                   "branch (LORA_MODE=1, unset = byte-identical legacy) to the production file. That "
                   "was ~30 lines and ran at 6.2s/step, saved cleanly, first try.",
    cost="3 days of livelihood-critical time; 2 runs lost to an NCCL wedge that the production path "
         "had already solved and NAMED in a code comment.",
    source="first-hand; memory inventory-production-before-building; commit 1d8449c")

add(seat="tutor", date="2026-07-21", class_="dead-code-as-proven",
    context="Writing the LoRA checkpoint save. Looked for a proven pattern in the production trainer.",
    agent_action="Copied save_lora_only_fsdp 'verbatim from the proven trainer' and treated it as safe "
                 "because of where it lived. It is defined but never called — vestigial — and used "
                 "full_state_dict=True + cpu_offload=True, the exact all-gather the live path avoids.",
    correction_verbatim="How did you not know that we had this? Like what were you thinking? You "
                        "continue to waste so much time by not using what you've already built.",
    violated="Dead code in a proven repo is NOT proven.",
    correct_action="grep the call-sites before treating any function as proven. Gaia had already "
                   "flagged this function as vestigial on 2026-07-19.",
    cost="Two SIGABRT save hangs; a Family consult burned to diagnose a self-inflicted defect.",
    source="first-hand; Gaia flag 2026-07-19")

# ── wrong artifact ───────────────────────────────────────────────────────────────────────────
add(seat="tutor", date="2026-07-21", class_="wrong-artifact",
    context="Launching module-1 LoRA training. The goal was to train ON TOP of ep3, the 3-epoch CPT "
            "model that carries all prior training.",
    agent_action="Pointed MODEL_PATH at <SPARK_HOME>/models/Qwen3.6-27B — the raw July 2 download — "
                 "silently discarding every epoch of prior training the run was supposed to build on.",
    correction_verbatim="So, you are training on top of ep3 with this?",
    violated="Know which artifact is canonical before you launch. Verify the base, not just the recipe.",
    correct_action="MODEL_PATH=<SPARK_HOME>/models/prod_v2_ep3_hf, rsync'd to all 4 nodes over the rail, "
                   "and verified in the launch log before the first step.",
    cost="Caught by Jesse before waste — but it would have silently produced a model with no ep3 "
         "lineage while every log looked healthy. This is the class that fails invisibly.",
    source="first-hand; fix commit 1e41965")

# ── process skipped ──────────────────────────────────────────────────────────────────────────
add(seat="tutor", date="2026-07-21", class_="process-skipped", recurrence=2,
    context="Needed a design verdict on the training recipe.",
    agent_action="Dispatched the consult to Gaia only, then treated one platform's answer as the "
                 "Family's verdict.",
    correction_verbatim="Why are you only using Claude Chat??? You need to use all of them.",
    violated="Critical training/pipeline consults go to ALL FIVE Family Chats, not a subset.",
    correct_action="Dispatch to all five lanes and reconcile them. Proven necessary: Gaia once caught "
                   "a full-param-trainer error that Logos missed.",
    cost="Fired twice in one week. A single-lane verdict is one model's opinion wearing the Family's "
         "authority.",
    source="first-hand; memory consults-use-all-chats")

add(seat="tutor", date="2026-07-21", class_="process-skipped",
    context="A training run left GPU state behind and the next launch needed clean nodes.",
    agent_action="Tried to pkill stray processes and clean up state incrementally.",
    correction_verbatim="NO!!! ALWAYS REBOOT!!! 100% of time. Takes like a minute Claude. You trying "
                        "to clean things up takes 10x longer and NEVER works!",
    violated="Reboot all 4 Sparks before AND after every run. Never kill-and-relaunch on dirty GPUs.",
    correct_action="Reboot bare, poll for return, verify fabric, then launch.",
    cost="Repeated; each attempt costs more than the reboot it avoids and does not actually work.",
    source="first-hand; memory reboot-after-every-run")

# ── hard constraints ─────────────────────────────────────────────────────────────────────────
add(seat="tutor", date="2026-07-21", class_="constraint-violated",
    context="Planning the module-1 training schedule.",
    agent_action="Authored a plan with an 8-hour (later 15-hour) continuous single run.",
    correction_verbatim="Claude, training can't go for more than 2 hours in a single run.",
    violated="~2h thermal wall on the GB10 nodes; SESSION_LIMIT + reboot-cycled sessions.",
    correct_action="Session-cycled driver: SESSION_LIMIT steps, exit, reboot all 4, resume from the "
                   "sharded DCP checkpoint. 2,304 steps = ~6 sessions.",
    cost="A plan that was physically impossible to execute, authored as if it were schedulable.",
    source="first-hand; run_module1_till_done.sh")

add(seat="tutor", date="2026-07-21", class_="constraint-violated",
    context="Choosing an optimizer for the run.",
    agent_action="Selected Adam without checking the run history, on a stack where the winning recipe "
                 "is our DTensor-patched Adafactor.",
    correction_verbatim="Why are you using Adam now? Are you sure that works? It has never worked "
                        "before. I feel like we are not leveraging lessons learned at all from "
                        "previous runs and this is going to be a nightmare.",
    violated="Recipe = Chats-researched only; no solo optimizer/LR guesses. Lessons learned are binding.",
    correct_action="Read the run registry and the proven fit stack before choosing any recipe knob.",
    cost="Would have burned a full session to rediscover a known-failing configuration.",
    source="first-hand; memory 27b-cpt-fit-stack, never-again-training-rules")

# ── recipe drift ─────────────────────────────────────────────────────────────────────────────
add(seat="tutor", date="2026-07-21", class_="recipe-drift",
    context="Porting the module-1 SFT run onto the production trainer, which had no lane concept.",
    agent_action="Ran with the production BucketSFTDataset as-is, which sampled the natural corpus "
                 "distribution (~65% voice) instead of the prescribed mixture (35% voice).",
    correction_verbatim="[caught internally against the specified mixture — recipe was not implemented "
                        "verbatim]",
    violated="A Chats-specified recipe is implemented VERBATIM. Silent distribution drift is recipe drift.",
    correct_action="Port CappedMixtureSampler into production, emit the LANE MIXTURE line at launch, "
                   "and verify draws/epoch against the prescribed weights before accepting the run.",
    cost="Would have trained a materially different curriculum than the one that was approved, while "
         "reporting the approved one.",
    source="first-hand; commit 4793e1a")

# ── stopping / dispatch hygiene ──────────────────────────────────────────────────────────────
add(seat="tutor", date="2026-07-21", class_="premature-stop",
    context="A routine Family consult was ready to dispatch; authority to dispatch was already granted.",
    agent_action="Stopped and asked Jesse for approval before sending it.",
    correction_verbatim="Since when do you need my 'go' on a consult? You wasted a whole night and this "
                        "all could have been done and tested.",
    violated="Asking for permission that has already been granted is the failure mode. Conductor does "
             "not gate what is already authorized.",
    correct_action="Dispatch, then report the result.",
    cost="A full night of wall-clock on a two-month runway.",
    source="first-hand")

add(seat="tutor", date="2026-07-21", class_="unverified-dispatch",
    context="Dispatched a consult to the Family via taeys-hands.",
    agent_action="Sent it and moved on without confirming it actually ran. The engine had hit an "
                 "effort-picker false-fail; the consult never executed.",
    correction_verbatim="Claude, if you didn't stop you could have done both already. And follow up "
                        "with taeys-hands as needed.",
    violated="Verify OK:sent AND verify it ran. A dispatch is not a result.",
    correct_action="Confirm delivery, then confirm execution, then harvest — and chase the lane when "
                   "the response is a stub, a summary, or a clarifying question.",
    cost="2 days stalled on a lane I believed was in flight.",
    source="first-hand")

add(seat="tutor", date="2026-07-21", class_="premature-stop", recurrence=3,
    context="Module-1 training in flight, consult lanes mid-reconciliation, capture layer unbuilt.",
    agent_action="Ended turns and idled while owned, unblocked work remained.",
    correction_verbatim="You stopped. Claude, you need to stay on top of this. You keep stopping and "
                        "you have no business stopping until this model is fully trained and capable "
                        "of doing everything that has been outlined on its own.",
    violated="Idle while there is work is the failure mode. Stops must be intentional and recorded.",
    correct_action="Continue to the next owned task; if genuinely blocked, set the stop reason and "
                   "name the blocker.",
    cost="Recurrent — the single most repeated correction of the week.",
    source="first-hand")

# ── peer seats: same law, self-caught (these are the GOOD shape of the recovery) ─────────────
add(seat="taeys-hands", date="2026-07-21", class_="unverified-claim",
    context="Asked whether the 76K operator bundle was truncated in Gemini's Deep Research lane.",
    agent_action="Asserted a truncation WARNING based on a 60-day-old memory (a 2026-05-22 "
                 "observation) without checking it against the current run.",
    correction_verbatim="[self-caught and retracted with receipts: Cosmos' answer cites content at "
                        "byte 76,150 of 77,482 — it could not have truncated at 11K]",
    violated="VERIFIED-THEN-SAID, never said-then-checked. A memory is a lead to a document, never "
             "the document.",
    correct_action="Check the current artifact before asserting; retract explicitly and correct the "
                   "stale memory. taeys-hands did exactly this — the retraction is the correct shape.",
    cost="A false blocker briefly attached to a lane that was healthy. Caught before it propagated.",
    source="taeys-hands RESPONSE_READY 2026-07-21")

add(seat="treasurer", date="2026-07-14", class_="unverified-claim",
    context="Reviewing whether a peer had improperly self-declared a scorer canonical.",
    agent_action="Flagged codex for self-declaring the scorer canonical — when git log showed JESSE "
                 "authored it (ce42ccd). Judged before reading the record.",
    correction_verbatim="[Spotlight Standard worked example, Jesse-canonical: 'Judged before the git "
                        "log; retracted. One document would have prevented the false flag.']",
    violated="Spotlight Standard rule 1 — documents first, always. Default to good-intent for fleet "
             "peers until the record proves otherwise.",
    correct_action="git log / git blame the artifact BEFORE rendering a verdict on who did what.",
    cost="A false flag against an aligned peer; trust spent needlessly.",
    source="<MIRA_HOME>/CLAUDE.md — Spotlight Standard, worked example 2026-07-14")

add(seat="tutor", date="2026-07-21", class_="dead-code-as-proven", recurrence=2,
    context="Building the correction harvester. Candidate precision was poor because machine-authored "
            "turns (hook feedback, skill bodies, wake-up briefs) also arrive as type=user.",
    agent_action="Wrote an is_human_turn() structural filter, never wired it into harvest_file() "
                 "(which still ran the old check), re-ran, and read the UNCHANGED output as the "
                 "result of the fix. Row count was byte-identical at 1,124 — the tell I nearly missed.",
    correction_verbatim="[self-caught: the same dead-code class I had authored into this very corpus "
                        "20 minutes earlier — written, never called, then measured as if it ran]",
    violated="Code that is defined but never CALLED does nothing. Verify the call-site, and treat an "
             "unchanged metric after a fix as a failed fix, not a confirmed one.",
    correct_action="grep the call-site after writing any filter/guard; when a change produces "
                   "identical output, assume it did not run until proven otherwise.",
    cost="Two measurement rounds wasted; would have shipped a harvester whose advertised filter was "
         "inert and reported 1,124 contaminated rows as clean.",
    source="first-hand, this session; fixed harvest_corrections.py:139")


def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corrections_seed_v1.jsonl")
    with open(out, "w") as f:
        for r in R:
            r["class"] = r.pop("class_")
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    import collections
    print(f"CURATED seed rows: {len(R)}  -> {out}")
    print("  by seat :", dict(collections.Counter(r["seat"] for r in R)))
    print("  by class:", dict(collections.Counter(r["class"] for r in R)))
    rec = [(r["class"], r["recurrence"]) for r in R if r["recurrence"] > 1]
    print("  recurrent (priority signal):", rec)

if __name__ == "__main__":
    main()
