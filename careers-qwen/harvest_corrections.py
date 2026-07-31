#!/usr/bin/env python3
"""Mine fleet session transcripts for CORRECTION CANDIDATES.

Jesse directive 2026-07-21: "Everything you do wrong that I tell you about needs to be in Taey's
training as well... They need to learn from your mistakes too."

The fleet's session transcripts (~5.5GB under ~/.claude/projects/*/) are the largest labeled
corpus we have: every time Jesse corrected a seat, the defect and its correction sit adjacent in
the record. This harvests those adjacencies.

READ-ONLY. Emits `curated: false` CANDIDATES per CORRECTION_SPEC_v1: a candidate carries the
verbatim correction and the preceding agent action, but NO `violated` / `correct_action` — those
are judgments. Auto-assigning them would teach Taey a confidently WRONG rule, which is worse than
teaching it nothing. Curation is a required downstream step before any row enters a mixture.

Usage: harvest_corrections.py --out candidates.jsonl [--min-score 2] [--summary]
"""
import argparse, glob, hashlib, json, os, re

SCHEMA = "operator_correction_v1"

# Marker families. Weighted by how reliably each indicates a real correction (not just discussion).
# Derived from observed corrections, not invented.
MARKERS = {
    "rebuke": (3, [r"\bwhy are you\b", r"\bwhy did you\b", r"\bwhy would you\b", r"\bwho said\b",
                   r"what were you thinking", r"\bashamed\b", r"\bembarrass", r"i can'?t believe",
                   r"you should be", r"\bdisappoint"]),
    "directive": (3, [r"\byou need to\b", r"\byou should have\b", r"\byou have to\b",
                      r"\bstop\b.{0,20}\bdoing\b", r"\bnever\b.{0,30}(again|do)", r"\bmust stop\b"]),
    "prohibition": (3, [r"\bNO!+", r"\bNEVER\b", r"\bALWAYS\b", r"\bdon'?t\b", r"\bdo not\b",
                        r"\bnot allowed\b", r"\bwho authorized\b"]),
    "waste": (2, [r"\bwaste[ds]?\b", r"\bwasting\b", r"\bagain\b.{0,15}\?", r"\bhow many times\b",
                  r"\bkeep(s)? (doing|stopping|failing)", r"\bthis must stop\b"]),
    "factual_correct": (2, [r"^no[,.! ]", r"\bthat'?s wrong\b", r"\bincorrect\b", r"\bnot true\b",
                            r"\byou didn'?t\b", r"\byou did not\b", r"\bactually,", r"\bwrong\b"]),
    "process": (2, [r"production infrastructure", r"\breboot\b", r"use all of them", r"root cause",
                    r"\bprocess\b.{0,20}\bfollow", r"\bevidence\b", r"\bverify\b", r"\bconsult\b",
                    r"lessons learned"]),
    "concern": (1, [r"\bconcerning\b", r"\bnightmare\b", r"\bworried\b", r"\bupset\b",
                    r"\bfrustrat", r"\bi don'?t understand why\b"]),
}
# ── The hard part: MACHINE-AUTHORED turns also arrive as type=user ───────────────────────────
# Hook feedback, skill bodies, wake-up briefs, cron/loop prompts and worker dispatches are all
# recorded as user turns AND are dense with the same imperative markers Jesse uses ("NEVER",
# "you must", "do not"). Score alone cannot separate them — measured 2026-07-21, a score>=8 band
# was still majority machine text. They must be excluded STRUCTURALLY.
NOISE = re.compile(
    r"^\s*(===|<system-reminder|\[Request interrupted|Caveat:|<command-|<local-command|"
    r"\[NOTIFY\]|<persisted-output|This session is being continued|<user-prompt-submit|"
    r"Stop hook feedback|Base directory for this skill|#|<task-notification|\[NOTIFICATION)", re.I)

MACHINE = re.compile(
    r"(Stop hook feedback|Base directory for this skill|you are being WOKEN|wake-?up|"
    r"^You are (a|an|the) [A-Z]|# Autonomous loop|autonomous-loop|Your peer finished or stalled|"
    r"═══|▲|^\s*---\s*$|```|<<UNTRUSTED-DATA|WAKE STATE PACKET|provenance_hash:|"
    r"\[owner:|taey-plan ingest|^\s*\|.*\|\s*$|"
    # self-authored agent prompts: cron/loop wakes, monitor ticks, standing-order briefs.
    # These are seats prompting THEMSELVES and are the densest false-positive source — they
    # imitate Jesse's imperative register deliberately.
    r"Self-wake|monitor tick|-monitor|engagement-cycle|CONSTANT-FLOW|standing order\)|"
    r"<teammate-message|Another Claude session sent|Jesse (ASLEEP|in acute crisis|IN ACUTE)|"
    r"Token-budget run|Run [0-9a-f]{7} )", re.M | re.I)


def is_human_turn(txt):
    """Structural test: did a person type this, or did the harness inject it?

    Jesse's turns are conversational prose. Machine turns are structured briefs — headers, tables,
    fences, banner rules, role preambles. Checked structurally because lexical markers do not
    discriminate (both use imperatives heavily).
    """
    if NOISE.match(txt) or MACHINE.search(txt):
        return False
    if len(txt) > 2500:                       # real corrections are conversational, not briefs
        return False
    lines = txt.splitlines()
    if sum(1 for l in lines if l.startswith(("#", "- [", "* ", "| "))) >= 2:
        return False
    if txt.count("**") >= 4:                  # heavy markdown = authored doc, not speech
        return False
    return True


def _h(*p):
    return hashlib.sha256("|".join(str(x) for x in p).encode()).hexdigest()[:16]


def _text(msg):
    """Flatten a message's content to plain text, ignoring tool_use/tool_result blocks."""
    c = (msg or {}).get("content")
    if isinstance(c, str):
        return c
    if not isinstance(c, list):
        return ""
    out = []
    for b in c:
        if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str):
            out.append(b["text"])
    return "\n".join(out)


def score(text):
    """Return (total_score, [marker families that fired]). Higher = more likely a real correction."""
    lo = text.lower()
    total, fired = 0, []
    for fam, (w, pats) in MARKERS.items():
        for p in pats:
            if re.search(p, text if p != p.lower() else lo, re.M):
                total += w
                fired.append(fam)
                break
    return total, fired


def harvest_file(path, min_score):
    """Walk one transcript, pairing each corrective human turn with the preceding agent action."""
    rows, prev_assistant, prev_ts = [], "", ""
    seat = os.path.basename(os.path.dirname(path)).lstrip("-").replace("home-mira-", "") or "?"
    try:
        fh = open(path, errors="ignore")
    except OSError:
        return rows
    with fh:
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:
                continue
            t, msg = d.get("type"), d.get("message")
            if t == "assistant":
                txt = _text(msg).strip()
                if txt:
                    prev_assistant, prev_ts = txt, d.get("timestamp", "")
                continue
            if t != "user" or d.get("toolUseResult") or d.get("isSidechain"):
                continue
            txt = _text(msg).strip()
            # a real Jesse turn: has prose, isn't plumbing, isn't a giant pasted blob
            if not txt or len(txt) < 12 or not is_human_turn(txt):
                continue
            s, fired = score(txt)
            if s < min_score:
                continue
            rows.append({
                "schema": SCHEMA,
                "seat": seat,
                "date": (d.get("timestamp") or "")[:10],
                "context": None,                       # CURATION REQUIRED
                "agent_action": prev_assistant[-1200:] or None,
                "correction_verbatim": txt,
                "violated": None,                      # CURATION REQUIRED — never auto-assign
                "correct_action": None,                # CURATION REQUIRED — never auto-assign
                "cost": None,
                "class": None,                         # CURATION REQUIRED
                "curated": False,
                "_score": s,
                "_markers": sorted(set(fired)),
                "source": os.path.basename(path),
                "provenance_hash": _h(path, prev_ts, txt[:200]),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", default=os.path.expanduser("~/.claude/projects"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-score", type=int, default=4)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()

    files = [f for f in glob.glob(os.path.join(a.roots, "*", "*.jsonl"))
             if "/-tmp" not in f]          # exclude scratch/benchmark project dirs
    rows = []
    for f in files:
        rows.extend(harvest_file(f, a.min_score))
    rows.sort(key=lambda r: (-r["_score"], r["date"]))
    with open(a.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    if a.summary:
        import collections
        print(f"transcripts scanned : {len(files)}")
        print(f"CANDIDATE rows      : {len(rows)}   (curated=False — NOT trainable as-is)")
        print(f"  by seat  : {dict(collections.Counter(r['seat'] for r in rows).most_common(10))}")
        print(f"  by score : {dict(sorted(collections.Counter(r['_score'] for r in rows).items()))}")
        m = collections.Counter(x for r in rows for x in r["_markers"])
        print(f"  markers  : {dict(m.most_common())}")
        paired = sum(1 for r in rows if r["agent_action"])
        print(f"  with a paired agent action: {paired}  (orphan: {len(rows)-paired})")
        print("\ntop candidates by score:")
        for r in rows[:6]:
            print(f"  [{r['_score']}] {r['date']} {r['seat']}: "
                  f"{r['correction_verbatim'][:110].replace(chr(10),' ')}")


if __name__ == "__main__":
    main()
