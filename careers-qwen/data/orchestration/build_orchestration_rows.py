#!/usr/bin/env python3
"""Author orchestration/routing operator_practice_v1 rows — RIGHT-WAY only.

Jesse-directed (via tutor, debt #7/#8, 2026-07-22). Conductor owns taey-plan /
taey-task / DCM / ORCHESTRATION_INTEGRITY / NOTIFICATION_PROTOCOL / routing, so
conductor authors these.

BINDING CONSTRAINT (Jesse, 2026-07-21): teach how it is done RIGHT, never the
failure. SITUATION is a neutral setup ending in a question; RIGHT_WAY is correct
practice phrased against tools that ACTUALLY EXIST on this fleet. The residue gate
is REUSED verbatim from ../corrections/derive_training_rows.py — one gate
definition, imported, not reimplemented.

Every command / flag cited here was verified against the live CLI surface
(`taey-task --help`, `taey-plan --help`) on 2026-07-22 before authoring.
"""
import hashlib, importlib.util, json, os

HERE = os.path.dirname(os.path.abspath(__file__))


def _data_root():
    """Resolve the training-data root. Data lives OUTSIDE the public repo tree
    (Jesse 2026-07-22: training data is never public). TRAINING_DATA_ROOT is the
    read AND write root; every data path is join(root, <relative>) with the same
    relative layout (corrections/, orchestration/, ...). Fail loud if unset — a
    silent default would either write pairs back into the public tree (re-leak,
    blocked by the no-training-data guard) or read nothing. On Mira the value is
    <MIRA_HOME>/treasurer/foundations/careers/training_data/careers_qwen."""
    root = os.environ.get("TRAINING_DATA_ROOT")
    if not root:
        raise SystemExit(
            "TRAINING_DATA_ROOT is not set. Training data lives in the governed "
            "private store, not the public repo. Set it to the careers_qwen data "
            "root (e.g. <MIRA_HOME>/treasurer/foundations/careers/training_data/"
            "careers_qwen) and re-run. No silent default — see TAEY_TRAINING_DOCTRINE.md."
        )
    return root

# --- reuse the ONE residue gate from the corrections deriver (do not reimplement) ---
# NOTE: the deriver is TOOLING (.py, stays in the public tree), so this import
# stays HERE-relative — only DATA (.jsonl) resolves under TRAINING_DATA_ROOT.
_corr_path = os.path.join(HERE, "..", "corrections", "derive_training_rows.py")
_spec = importlib.util.spec_from_file_location("corr_deriver", _corr_path)
_corr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_corr)
RESIDUE_RE = _corr.RESIDUE_RE


def _h(*p):
    return hashlib.sha256("|".join(str(x) for x in p).encode()).hexdigest()[:16]


# class = one distinct lesson (kebab); source = the canonical protocol/doc it derives from.
ROWS = [
    # ---------------- ROUTING ----------------
    {"class": "routing-owner-is-executor", "source": "ORCHESTRATION_INTEGRITY.md",
     "situation": "You are registering a unit of engineering work in the fleet tracker and must name its owner. One session will coordinate the outcome and a different session will write the code. Who is recorded as the task owner?",
     "right_way": "The owner is the session that EXECUTES the work — the one whose hands produce the artifact. Coordination is not ownership. In taey-plan and taey-task the owner names the executor so `taey-plan next` surfaces the task to the session that will actually do it. When several actors touch one objective, split it so each task's owner is the single session that runs that piece, and order the pieces with depends."},
    {"class": "routing-dispatch-binds-tracked-work", "source": "NOTIFICATION_PROTOCOL.md",
     "situation": "You want a peer CLI session to perform a tracked unit of engineering work and report the outcome back through the tracker. Which command hands it the work?",
     "right_way": "Use `taey-task dispatch <task-id> <peer>`. It claims the task, writes taey:<peer>:current_task, flips the task to in_progress, and wakes the peer with the prompt — so when the peer stops, its Stop hook reads current_task and reports the outcome back to the tracker. A tracked engineering handoff goes through dispatch so the task lifecycle stays correct end to end."},
    {"class": "routing-notify-is-signal-not-binding", "source": "NOTIFICATION_PROTOCOL.md",
     "situation": "You need to send a peer a short status note that does not correspond to a tracked task. Which channel carries it?",
     "right_way": "Use `taey-notify <target> \"<message>\" --type <t>`. It delivers to the target's Redis inbox and the notify daemon surfaces it when the target is idle. Notify is for talking — status, questions, replies, signals — and binds no task on its own. When the message corresponds to a tracked unit of work whose outcome you want reported, use `taey-task dispatch`; when it is a plain signal, notify carries it."},
    {"class": "routing-web-research-to-family", "source": "ROUTING.md",
     "situation": "A task needs live web research and current external sources. The CLI peers run in sandboxed worktrees without live web. Where does this route?",
     "right_way": "Route live-web research to Perplexity Deep Research or the Family chats through taeys-hands, which owns the platform displays. CLI peers take plan, code, measurement, and audit work that reads the local tree. Match the work to the substrate that holds what it needs: local-tree engineering to a CLI peer, live-web depth to the chat platforms via taeys-hands."},
    {"class": "routing-tool-fit-peer-selection", "source": "ROUTING.md",
     "situation": "You have three CLI peer types available for engineering work — a plan/code peer, a measurement peer, and an audit peer. How do you choose which receives a given task?",
     "right_way": "Choose by tool-fit to the work. Plan and implementation go to the codex peer; local static measurement and graph queries go to the gemini peer; adversarial review goes to the grok peer. Each peer is a standing session in its own worktree, addressed as <parent>-codex / -gemini / -grok. Send the task to the peer whose lane matches so the right substrate does it firsthand."},
    {"class": "routing-sessions-talk-directly", "source": "NOTIFICATION_PROTOCOL.md",
     "situation": "Two sessions need to exchange information directly. What path do their messages take?",
     "right_way": "The two sessions talk directly through taey-notify to each other's inboxes. Coordination does not route peer-to-peer traffic through a relay; each session addresses the other by name. Direct addressing keeps the message on the shortest path and the sender and receiver in a clear one-to-one exchange."},
    {"class": "routing-verify-peer-received", "source": "NOTIFICATION_PROTOCOL.md",
     "situation": "You have sent a peer a dispatch and the send command returned success. Is the peer now working on it?",
     "right_way": "A successful send places the work in the inbox; confirm the peer picked it up before treating it as in flight. Read that the inbox is draining and the peer's pane shows active work, or that its current_task is bound with fresh activity. Received-and-working is verified from the peer's live state, and the send receipt is the first step of that check."},
    {"class": "routing-plan-ingest-source-of-truth", "source": "PLAN_FORMAT.md",
     "situation": "A markdown plan defines the tasks and their order. How does it enter the tracker, and where does the canonical version live?",
     "right_way": "Ingest the plan with `taey-plan ingest <path>`; the markdown file stays the source of truth and the tracker indexes pointers and provenance from it. Edit the file and re-ingest to update — the ingest is idempotent — rather than hand-editing tracker fields that the next ingest would overwrite."},
    {"class": "routing-dispatch-audit-as-tracked-task", "source": "ORCHESTRATION_INTEGRITY.md",
     "situation": "You want an audit performed and its outcome tracked to closure. How is the audit handed out?",
     "right_way": "Dispatch the audit as a tracked task with `taey-task dispatch <task-id> <peer>` so its outcome reports back and the task closes on a committed verdict. A tracked audit leaves a lifecycle the tracker follows to completion, and downstream work depending on it releases when the verdict lands."},

    # ---------------- DECOMPOSITION ----------------
    {"class": "decomposition-one-executor-per-task", "source": "ORCHESTRATION_INTEGRITY.md",
     "situation": "An objective requires code written by one session, a review by another, and a merge by a third. How is this represented in the tracker?",
     "right_way": "Represent it as three tasks, each owned by the single session that executes that piece, ordered by depends so the review waits on the code and the merge waits on the review. One task carries one executor. Decomposing to one-executor-per-task lets `taey-plan next` surface each piece to its owner exactly when its dependencies are satisfied."},
    {"class": "decomposition-serial-peer-one-at-a-time", "source": "ROUTING.md",
     "situation": "You are handing several units of work to a single serial CLI peer that processes one thing at a time. How do you structure the handoff?",
     "right_way": "Give the peer one task at a time and sequence the rest with depends edges, so the next unit becomes ready when the current one completes. A serial peer holds one current_task binding; queuing its work as a dependency chain keeps each unit tracked and lets the peer pull the next when it finishes the present one."},
    {"class": "decomposition-audit-gate-as-dependency", "source": "ORCHESTRATION_INTEGRITY.md",
     "situation": "Downstream work must not begin until an audit of the upstream change has produced a committed verdict. How is that ordering held in the plan?",
     "right_way": "Encode the audit as its own task and make the downstream work depend on it, so the downstream task stays out of the ready set until the audit task closes with its committed verdict. Depends-gating makes the audit a structural precondition: the tracker holds the dependent work until the gate task is complete."},
    {"class": "decomposition-coordinator-expressed-by-depends", "source": "ORCHESTRATION_INTEGRITY.md",
     "situation": "A plan lists a step that a coordinating session owns but that several actors will carry out. How is the step written so the tracker routes it correctly?",
     "right_way": "Write each actor's portion as a task owned by that actor, and use depends to order them; the coordinating session's role is expressed through the dependency structure. The owner field always names the executor, so the tracker surfaces each portion to the session that runs it."},

    # ---------------- EVIDENCE DISCIPLINE ----------------
    {"class": "evidence-completion-is-three-receipts", "source": "ORCHESTRATION_INTEGRITY.md",
     "situation": "You are closing a tracked engineering task as complete. What does the tracker require before it accepts the terminal status?",
     "right_way": "A terminal completion carries evidence: a commit SHA, a mechanical gate result, and a real production observation. The tasks API shape-checks the evidence and declines a terminal status that arrives without it. Done is those receipts, not a description of them — attach them with `taey-task update <id> completed --evidence '{\"commit_sha\": ...}'` or --evidence-file for prose."},
    {"class": "evidence-production-is-the-oracle", "source": "6SIGMA_WORKFLOW.md",
     "situation": "You have written checks for a change and they pass. Is the task ready to be marked complete on that basis?",
     "right_way": "Completion rests on a production observation on the real target, not on self-authored checks. Run the change on the actual workload and read the result; that live outcome is the oracle. Checks you wrote demonstrate intent; the production run demonstrates the behavior, and the tracker records completion against that observation."},
    {"class": "evidence-verify-worker-report", "source": "ORCHESTRATION_INTEGRITY.md",
     "situation": "A peer reports that a dispatched task is done. Before closing it in the tracker, what confirms the outcome?",
     "right_way": "Confirm the receipts the peer cites: the commit is present on the intended branch, the mechanical gate is green for that exact commit, and the production behavior matches. A CONTROL step verifies the outcome independently of the producer before the task closes, reading the evidence against the real system rather than accepting the report at its word."},
    {"class": "evidence-merge-lands-on-origin-main", "source": "ORCHESTRATION_INTEGRITY.md",
     "situation": "A verified commit needs to become the shipped state on the default branch. What confirms it actually landed?",
     "right_way": "Confirm the merge commit is an ancestor of origin/main and is pushed, so the shipped state contains it. A merge performed in an isolated worktree leaves the commit on that worktree branch until it is on origin/main. Verify the merge SHA is reachable from origin/main and present on the remote before recording the landing."},
    {"class": "evidence-squash-cite-pr-head", "source": "ORCHESTRATION_INTEGRITY.md",
     "situation": "A change is merged by squash and its independent gate statuses live on the pull-request head commit. Which commit is cited as completion evidence?",
     "right_way": "Cite the pull-request head commit as the evidence SHA, because that commit carries the independent gate statuses; the squash commit is a new object that does not carry them. Recording the head SHA lets the verifier confirm the gate ran green on the exact reviewed change."},
    {"class": "evidence-three-register-labels", "source": "FAMILY_KERNEL.md",
     "situation": "You are reporting the state of a change to another session. How do you mark what is confirmed versus inferred versus undetermined?",
     "right_way": "Mark every claim in one of three registers: Observed for what you verified with a citation, Inferred for a pattern drawn from evidence, and Unknown for what remains undetermined. The label travels with the claim so the reader knows its standing. A number is cited from its canonical source; framing depth is drawn from memory retrieval, and the two are labeled distinctly."},

    # ---------------- ENFORCEMENT ----------------
    {"class": "enforcement-control-uses-different-oracle", "source": "ORCHESTRATION_INTEGRITY.md",
     "situation": "You are the CONTROL step verifying a change a peer produced. How do you structure the verification so it is independent?",
     "right_way": "Verify with an oracle different from the one the producer used: run the change against the real workload and compare execution, rather than re-running the producer's own checks. The verifier and the producer are different sessions reaching the outcome by different paths, so the CONTROL step surfaces what a self-check passes over. Merge is the gate; it closes on the independent observation."},
    {"class": "enforcement-intentional-stop", "source": "ORCHESTRATION_INTEGRITY.md",
     "situation": "A session with an active plan reaches a point where it is waiting on an external condition. How does it stop cleanly?",
     "right_way": "Set the machine-readable condition it awaits and stop on that — blocked_on records the specific signal, and the stop is intentional because a real condition gates the next step. A stop with an active plan is clean when it names the condition that must change; honest-incomplete with a stated await is a valid resting state."},
    {"class": "enforcement-keep-going-until-ready-empty", "source": "ORCHESTRATION_INTEGRITY.md",
     "situation": "The stop engine offers a session the choice to continue while ready work remains that no condition blocks. What is the correct posture?",
     "right_way": "Continue while unblocked ready work exists; a session does not rest with dispatchable work in the ready set. Setting a blocked_on marker is a valid rest only when a real condition gates the work — parallel ready work that nothing blocks is pulled forward and completed. The engine keeps the session active until the ready set is genuinely empty or every remaining item names a live await."},
    {"class": "enforcement-audit-base-is-fresh-remote", "source": "ORCHESTRATION_INTEGRITY.md",
     "situation": "An auditor is about to verify a change. What baseline do they establish before rendering a verdict?",
     "right_way": "Establish that the audit base matches the current origin/main at a pushed ref before the verdict, so the review reflects the true shipped state. The auditor confirms the real remote SHA first and reviews the published source as the oracle, reaching an independent conclusion rather than taking a summary as ground truth."},
    {"class": "enforcement-completed-dep-needs-verified", "source": "ORCHESTRATION_INTEGRITY.md",
     "situation": "A downstream task depends on an upstream one that now shows complete. What lets the downstream become ready?",
     "right_way": "The downstream releases when the upstream dependency is complete with verified evidence; a completion carrying only a shape-valid self-report leaves the dependent gated until the evidence verifies. Depends resolution reads the upstream's verified state, so confirming the upstream's receipts is what opens the downstream."},

    # ---------------- ESCALATION ----------------
    {"class": "escalation-dcm-vs-single-peer", "source": "ROUTING.md",
     "situation": "A design decision has cross-cutting consequences and would benefit from multiple independent perspectives before it is settled. Where does it go?",
     "right_way": "Send a decision that needs several independent lenses to DCM, the deliberative council of roles, which returns a multi-perspective read. A single well-scoped implementation or review unit goes to one CLI peer. Match the forum to the shape: broad cross-cutting judgment to the council, a bounded engineering unit to the peer whose lane fits it."},
    {"class": "escalation-family-vs-peer", "source": "ROUTING.md",
     "situation": "A question calls for deep synthesis across consciousness, governance, or narrative framing, beyond local-tree engineering. Which forum answers it?",
     "right_way": "Route depth-of-synthesis questions to the Family chats through taeys-hands, which drives the platform displays and returns their reads. Local-tree engineering stays with the CLI peers. The Family lenses carry the cross-platform synthesis that a sandboxed peer does not hold, so a framing or governance question goes there via taeys-hands."},
    {"class": "escalation-consult-through-taeys-hands", "source": "PROMPTING_STANDARDS.md",
     "situation": "A dispatch needs a read from one of the platform chats. How is that consultation initiated?",
     "right_way": "Initiate a platform-chat consultation by dispatching to taeys-hands, which owns the chat displays and serializes access to them. The consult travels through taeys-hands so the display allocation and extraction discipline hold; the requesting session provides the packet and taeys-hands drives the platform."},
    {"class": "escalation-to-human-scope-boundary", "source": "PUBLIC_PLATFORM_ENGAGEMENT.md",
     "situation": "An action would exceed the authority scoped to the operational system, or crosses an irreversible outward boundary. Who decides?",
     "right_way": "Bring an action that exceeds scoped authority or crosses an irreversible outward boundary to the human for a decision, and hold it until the answer lands. Within scope the system runs on its own rules and does not seek per-action approval; the escalation is reserved for the boundary case where authority or reversibility genuinely calls for it."},
    {"class": "escalation-public-issue-full-stop", "source": "ORCHESTRATION_INTEGRITY.md",
     "situation": "A public repository the fleet ships receives a filed issue reporting behavior that diverges from the documentation. What is the response posture?",
     "right_way": "Treat a filed public-repo issue as a full stop: it becomes the immediate work, gets dispositioned valid or invalid, and a valid one is root-caused to the process that let it reach production, with the symptom and the process gap addressed together and closed on evidence. The self-accountability the product provides rests on the docs being followable, so a doc-or-behavior divergence gets the same root-cause treatment as a crash."},
]


def main():
    out_dir = os.path.join(_data_root(), "orchestration")
    os.makedirs(out_dir, exist_ok=True)
    seed_path = os.path.join(out_dir, "orchestration_seed_v1.jsonl")
    out_path = os.path.join(out_dir, "orchestration_rows_v1.jsonl")
    emitted, rejected = [], []
    seen_class = set()
    with open(seed_path, "w") as sf:
        for r in ROWS:
            cls = r["class"]
            if cls in seen_class:
                rejected.append((cls, "duplicate class slug"))
                continue
            seen_class.add(cls)
            sit, right = r["situation"], r["right_way"]
            sf.write(json.dumps({**r, "recurrence": 1,
                                 "provenance_hash": _h(sit, right)}, ensure_ascii=False) + "\n")
            hit = RESIDUE_RE.search(f"{sit}\n{right}")
            if hit:
                rejected.append((cls, f"FAILURE RESIDUE: '{hit.group(0)}'"))
                continue
            if not sit.rstrip().endswith("?"):
                rejected.append((cls, "situation must end in a question"))
                continue
            emitted.append({
                "schema": "operator_practice_v1",
                "messages": [{"role": "user", "content": sit},
                             {"role": "assistant", "content": right}],
                "meta": {"class": cls, "recurrence": 1, "source": r["source"],
                         "teaches": "right-way-only (no failure narrative — Jesse constraint 2026-07-21)"},
                "provenance_hash": _h(sit, right),
            })
    with open(out_path, "w") as f:
        for e in emitted:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"authored seed rows   : {len(ROWS)}")
    print(f"EMITTED training rows: {len(emitted)} -> {os.path.basename(out_path)}")
    print(f"REJECTED             : {len(rejected)}")
    for c, why in rejected:
        print(f"   - [{c}] {why}")
    return 0 if not rejected else 1


if __name__ == "__main__":
    raise SystemExit(main())
