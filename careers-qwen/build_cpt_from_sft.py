#!/usr/bin/env python3
"""build_cpt_from_sft.py — derive a CPT delta from what an SFT corpus actually touched.

Jesse's directive (2026-07-24): "we need to get everything in there on the updates, diffs,
current code that was changed due to SFT, the current correct documentation on everything
touched by SFT, all of that needs to be tracked when a training pair is developed and pulled
into CPT."

The coupling this creates: an SFT row teaches BEHAVIOR ("when X, do Y"). CPT must carry the
WORLD that behavior operates on — the current code, the current documentation, and the diff
that changed it. Train the behavior without the world and the model applies a correct rule to
a system it does not know; train the world without the behavior and it knows without acting.

WHAT IT EXTRACTS from every row's `meta`:
  - commit SHAs  -> the DIFF (what changed, and why the row exists)
  - file paths   -> the CURRENT CONTENT (the correct state as of now, not as of the incident)
  - doc refs     -> the canonical documentation for the touched surface

ORDER MATTERS (Jesse, same directive): run CPT FIRST, then SFT. The model should learn the
current correct world, then learn the behavior on top of it. Doing SFT first teaches conduct
about a system the weights have never seen.

Usage:
  python3 build_cpt_from_sft.py --sft <corpus.jsonl> --repo <git repo> --out <delta.jsonl>
                                [--max-diff-lines 400]
"""
import argparse, json, os, re, subprocess, hashlib, collections

SHA_RE = re.compile(r"\b[0-9a-f]{7,40}\b")
PATH_RE = re.compile(r"[\w./-]+\.(?:py|sh|md|ya?ml|jinja)")
# a bare hex word is only a commit if git can resolve it; guard against hashes-of-things
NOISE = {"probe_is_synthesized", "origin", "captured"}


def git(repo, *args):
    try:
        return subprocess.run(["git", "-C", repo, *args], capture_output=True,
                              text=True, timeout=30).stdout
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", required=True)
    ap.add_argument("--repo", default=os.environ.get("REPO_ROOT", os.path.expanduser("~/palios-training")))
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-diff-lines", type=int, default=400)
    ap.add_argument("--seed-file", action="append",
                    help="curated seed file(s); lets derived_from resolve to the seed's own source")
    ap.add_argument("--max-diff-bytes", type=int, default=200_000,
                    help="a .jsonl diff is huge SINGLE lines; the line cap alone is not a guard")
    a = ap.parse_args()

    # A row's provenance is not always in a field literally named `source`. Follow the chain:
    #   meta.source            -> direct
    #   meta.derived_from      -> the curated seed's provenance_hash; the SEED carries source/date
    #   origin=captured + view/seq/primitive -> a real UI walk, provenance in the coordinates
    # Looking only for `source` undercounts coverage badly and drops real world-material.
    seed_src = {}
    for sp in a.seed_file or []:
        for sd in (json.loads(l) for l in open(sp) if l.strip()):
            ph = sd.get("provenance_hash")
            if ph:
                seed_src[ph] = " ".join(str(sd.get(k, "")) for k in ("source", "date", "seat", "context"))

    shas, paths, rows_with_source = set(), set(), 0
    for line in open(a.sft):
        if not line.strip():
            continue
        meta = json.loads(line).get("meta", {})
        blob = json.dumps(meta)
        resolved = ""
        if meta.get("source"):
            resolved = str(meta["source"])
        elif meta.get("derived_from") and seed_src.get(meta["derived_from"]):
            resolved = seed_src[meta["derived_from"]]          # follow into the seed
        elif str(meta.get("origin", "")).lower() == "captured" and meta.get("view"):
            resolved = f"captured walk view={meta.get('view')} seq={meta.get('seq')}"
        if resolved:
            rows_with_source += 1
            blob += " " + resolved                              # mine the RESOLVED text too
        shas.update(SHA_RE.findall(blob))
        paths.update(PATH_RE.findall(blob))

    # keep only SHAs git actually resolves to a commit — the rest are content hashes
    real = []
    for s in sorted(shas):
        if s in NOISE:
            continue
        t = git(a.repo, "cat-file", "-t", s).strip()
        if t == "commit":
            real.append(s)

    emitted, seen = [], set()

    # 1. DIFFS — why each row exists.
    # HARD EXCLUSIONS, learned the hard way: a diff can carry private training data straight
    # into a corpus. The line cap does NOT protect you — a .jsonl diff is enormous SINGLE
    # lines, so 400 lines can be 8 MB of training rows. Filter by PATH and by BYTES, and
    # refuse any commit that touches a data path at all.
    DATA_RE = re.compile(r"(training_data|datasets)/|\.jsonl$|_gated\.jsonl")
    skipped_data, skipped_big = [], []
    for s in real:
        names = git(a.repo, "show", "--name-only", "--format=", s).split()
        if any(DATA_RE.search(n) for n in names):
            skipped_data.append(s[:12])
            continue
        d = git(a.repo, "show", "--stat", "--patch", s)
        if not d:
            continue
        if len(d) > a.max_diff_bytes:
            skipped_big.append(f"{s[:12]}({len(d)//1024}KB)")
            continue
        lines = d.splitlines()
        if len(lines) > a.max_diff_lines:
            lines = lines[:a.max_diff_lines] + [f"... [diff truncated at {a.max_diff_lines} lines]"]
        emitted.append({"text": f"[CHANGE THAT PRODUCED A TRAINING PAIR: commit {s[:12]}]\n\n"
                                + "\n".join(lines)})

    # 2. CURRENT CONTENT — the correct state now, which is what the model must know
    for p in sorted(paths):
        if DATA_RE.search(p):
            continue                      # never inline a data file as "current state"
        cand = p if os.path.isabs(p) else os.path.join(a.repo, p)
        if not os.path.isfile(cand) or os.path.getsize(cand) > a.max_diff_bytes:
            continue
        key = os.path.realpath(cand)
        if key in seen:
            continue
        seen.add(key)
        try:
            body = open(cand, errors="ignore").read()
        except Exception:
            continue
        if not body.strip():
            continue
        emitted.append({"text": f"[CURRENT STATE OF A SURFACE SFT TOUCHED: {os.path.basename(cand)}]\n\n{body}"})

    with open(a.out, "w") as f:
        for e in emitted:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    sha = hashlib.sha256(open(a.out, "rb").read()).hexdigest()
    print(f"SFT rows scanned      : {sum(1 for _ in open(a.sft))}")
    print(f"  rows carrying source: {rows_with_source}")
    print(f"resolved commits      : {len(real)} -> diffs emitted")
    print(f"current files emitted : {len(seen)}")
    if skipped_data:
        print(f"EXCLUDED (data-bearing commits): {', '.join(skipped_data)}")
    if skipped_big:
        print(f"EXCLUDED (oversize diffs)      : {', '.join(skipped_big)}")
    print(f"CPT delta             : {a.out}  {os.path.getsize(a.out)} bytes  sha256 {sha[:16]}")
    if rows_with_source < sum(1 for _ in open(a.sft)) * 0.8:
        print("\nCOVERAGE WARNING: under 80% of rows name a source. Rows without one contribute "
              "NOTHING to this delta — the world behind them is invisible. Stamp meta.source "
              "with the commit/file/doc at authoring time, not afterwards.")


if __name__ == "__main__":
    main()
