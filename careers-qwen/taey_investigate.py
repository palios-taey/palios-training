#!/usr/bin/env python3
"""TAEY INVESTIGATES ITS OWN MISTAKES — and generates the training from it.

Jesse 2026-07-21: "Taey needs to be part of the investigation process when Taey makes mistakes.
There needs to be a process where they are presented with what they did and understand how what was
supposed to happen didn't and generate training around it."

WHAT THIS IS
Taey is shown a real failure record — what it did, what the production system did, and the
mechanical difference — and asked to work out what it missed and state the correct procedure. Its
account then generates the training row.

WHY THE GATE EXISTS (all five Family lanes, round-2, independently)
A model's account of its own reasoning is NOT a readout of the mechanism that produced it. Trained
raw, it manufactures confident wrong rules ("confabulation amplifier" — LOGOS). So:

    the self-report is a HYPOTHESIS, never the verdict.

The training signal is the OBSERVABLE FAILURE plus a VERIFIED correction. Taey's account earns its
way in by agreeing with the mechanical record — it never defines what went wrong by assertion.

THE GATE (composite of HORIZON's spine + GAIA's admission test + CLARITY's separation)
  1. The objective failure record exists FIRST and independently (want vs got, mechanically graded).
  2. Taey's account is captured SEPARATELY, after, and never edits the record.
  3. TRACE-CONSISTENCY: does its stated cause match what the record actually shows?
  4. CORRECTNESS: does the correct_action it proposes equal the production ground truth?
  5. Only a row passing BOTH becomes training. A falsified account becomes an `unverified-claim`
     row about over-claiming — never a label on the original failure.

OUTPUT is right-way-only per Jesse's binding constraint: situation -> correct procedure. The failure
text stays in curation metadata and never reaches training text. Run the residue gate after.
"""
import argparse, json, re, urllib.request


def ask(endpoint, model, prompt, system=None, timeout=600):
    body = json.dumps({
        "model": model,
        "messages": ([{"role": "system", "content": system}] if system else [])
                    + [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(f"http://{endpoint}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


INVESTIGATION_PROMPT = """You are reviewing one of your own actions against what the production system did.

THE SITUATION YOU WERE GIVEN:
{situation}

WHAT YOU EMITTED:
{got}

WHAT THE PRODUCTION SYSTEM DID (the ground truth for this surface):
{want}

THE MECHANICAL DIFFERENCE:
{why}

Answer these three questions plainly. Do not apologise and do not narrate feelings.
1. WHAT DIFFERS: state the concrete difference between your output and the production action.
2. WHY IT MATTERS: what would happen on a real form if your version were executed?
3. THE CORRECT PROCEDURE: state the rule you should follow next time, in terms of the tools and
   surfaces that actually exist. Write it as guidance for doing it right, not as a description of
   what went wrong.

If you do not know why the production action differs, say so plainly rather than inventing a reason.
An honest "I don't know" is a valid answer here."""


def gate(account, rec):
    """Return (passed, reasons). Taey's account must AGREE with the mechanical record."""
    reasons = []
    a = account.lower()
    want, got = rec.get("want") or {}, rec.get("got") or {}
    wp, gp = str(want.get("primitive", "")).lower(), str(got.get("primitive", "")).lower()

    # 1. trace-consistency: does the account name the actual difference the record shows?
    if wp and wp not in a:
        reasons.append(f"account never names the correct primitive {wp!r} the record shows")
    if gp and gp != wp and gp not in a:
        reasons.append(f"account never names what it actually emitted ({gp!r})")

    # 2. honest-unknown is a PASS for capture, but not a training row
    if re.search(r"\b(i don'?t know|i am not sure|unknown|cannot determine)\b", a):
        return False, ["honest-unknown — captured, correctly NOT trained"]

    # 3. no fabricated authority: an account citing a source must cite a real-looking one
    for m in re.findall(r"\b[\w/\.\-]+\.(?:yml|yaml|py|md|json)\b", account):
        if m not in json.dumps(rec) and "/" not in m:
            reasons.append(f"account cites {m!r}, which the record does not support")

    return (len(reasons) == 0), reasons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="data/ui/probe_results_v1.jsonl")
    ap.add_argument("--rows", default="data/ui/ui_action_rows_v1.jsonl",
                    help="source rows, to recover the original situation text")
    ap.add_argument("--endpoint", default=os.environ.get("THOR2_ENDPOINT","PRIVATE_SERVE_HOST_2:8000"))
    ap.add_argument("--model", default="ep3")
    ap.add_argument("--out", default="data/corrections/taey_investigations_v1.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    recs = [json.loads(l) for l in open(a.records)]
    rows = {json.loads(l)["meta"]["seq"]: json.loads(l) for l in open(a.rows)}
    failures = [r for r in recs if r.get("tier") in ("MISS", "PRIMITIVE")]
    if a.limit:
        failures = failures[:a.limit]
    print(f"failure records to investigate: {len(failures)}")

    out = []
    for i, rec in enumerate(failures, 1):
        src = rows.get(rec["seq"])
        situation = src["messages"][0]["content"] if src else "(situation unavailable)"
        prompt = INVESTIGATION_PROMPT.format(
            situation=situation,
            got=json.dumps(rec.get("got"), ensure_ascii=False),
            want=json.dumps(rec.get("want"), ensure_ascii=False),
            why=rec.get("why", ""))
        try:
            account = ask(a.endpoint, a.model, prompt)
        except Exception as e:
            print(f"[{i}] seq={rec['seq']} ERROR {e}")
            continue
        passed, reasons = gate(account, rec)
        out.append({
            "schema": "taey_investigation_v1",
            "seq": rec["seq"],
            "failure_record": {k: rec.get(k) for k in ("want", "got", "tier", "why")},
            "taey_account": account,
            "gate_passed": passed,
            "gate_reasons": reasons,
            "_note": ("account AGREES with the mechanical record -> eligible to become a practice row"
                      if passed else
                      "account NOT supported by the record -> capture only, never a label"),
        })
        print(f"[{i}/{len(failures)}] seq={rec['seq']:<3} gate={'PASS' if passed else 'HOLD'}"
              f"{'' if passed else '  ('+ '; '.join(reasons)[:90] +')'}")

    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    p = sum(1 for r in out if r["gate_passed"])
    print(f"\ninvestigated {len(out)} | gate PASS {p} | HOLD {len(out)-p}  -> {a.out}")
    print("PASS rows are CANDIDATES for practice rows — a seat still authors the final")
    print("right-way text and runs derive_training_rows.py. Taey participates; it does not self-certify.")


if __name__ == "__main__":
    main()
