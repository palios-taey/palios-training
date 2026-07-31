#!/usr/bin/env python3
"""assemble_full_corpus.py — walk OUR OWN repos and emit their FULL source+docs as CPT corpus.

The 2026-07-11 corpus miss: the base was trained WITHOUT the repo source code — the single most
important knowledge. This fixes the repo-code half (tutor owns dataset assembly; the repos live
locally under <MIRA_HOME>). Treasurer supplies the voice + background + G3 halves separately; all
merge into ONE comprehensive corpus for a fresh base.

Whitelist = OUR-code only (public + private-tagged), confirmed by Treasurer. Vendor forks EXCLUDED
(pure-vendor code is noise). Each file → {"text": <provenance header + content>, "source": <path>,
"repo": <name>, "visibility": public|private}. The visibility tag preserves the product-boundary
(private repos stay Jesse-only; public repos may ship in a product model later).

Usage: assemble_full_corpus.py <out.jsonl>
"""
import sys, os, json

BASE = "<MIRA_HOME>"

# Treasurer-confirmed whitelist (repo dir name under <MIRA_HOME> → visibility). Names resolved
# fuzzily below (some local dirs differ from the canonical repo name).
# EXACTLY Treasurer's confirmed whitelist (own-code only), resolved to local dir names.
# (hunter/x-claude/ai_native/career-ops-study were NOT on the whitelist — removed; hunter's
#  archive/+pr-prep/ were 200M tok of archived/vendored junk anyway.)
PUBLIC = ["claude-code-fleet-orchestrator", "claude-code-fleet-notify", "claude-code-api-watchdog",
          "mcp-reconnect", "taeys-hands", "dcm", "isma-core", "restart-safe-agents", "palios-training",
          "taey-ed", "taey-presence-validate", "claude-code-fleet-cockpit-template", "dgx-spark-multinode",
          "doge", "claude-code-fleet-support"]
PRIVATE = ["treasurer", "careers-apply", "careers-linkedin", "the-conductor", "isma", "embedding-server",
           "infra-soul"]

# include these source/doc extensions (code + docs + config)
KEEP_EXT = {".py", ".md", ".sh", ".bash", ".js", ".ts", ".tsx", ".jsx", ".txt", ".yaml", ".yml",
            ".toml", ".cfg", ".ini", ".jinja", ".j2", ".sql", ".html", ".css", ".rs", ".go", ".c",
            ".cpp", ".h", ".hpp", ".cu", ".rst", ".env.example"}
# skip these dirs entirely (vendor / junk / data / build)
SKIP_DIR = {".git", "node_modules", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache",
            "dist", "build", ".next", "target", "vendor", "third_party", ".cache", "site-packages",
            "data", "datasets", "corpora", "checkpoints", "models", "wandb", ".ruff_cache", "eggs",
            ".tox", "htmlcov", "coverage", ".idea", ".vscode",
            # ops-dir junk: archived material + vendored PR-prep + scratch + runtime output (not source)
            "archive", "pr-prep", "pr_prep", "tmp", "_deploy_tmp", ".staging", "backups", "backup",
            "scraped", "downloads", "logs", "runtime", "code.bak", "consultations", "dispatches",
            "runs", "outputs", "output", "results", "artifacts"}


def _is_junk_dir(d: str) -> bool:
    dl = d.lower()
    # EPISTEMIC MEMBRANE (Treasurer-required 2026-07-11): superseded/draft-tier = correction-corpus
    # only, NEVER CPT. Draft dirs hold retracted/scrubbed numbers (fabrication risk). Exclude ALL
    # draft dirs wholesale (nvidia_drafts/reddit_drafts/careers_drafts/drafts + any *_drafts).
    if dl == "drafts" or dl.endswith("_drafts") or dl.endswith("-drafts"):
        return True
    return d in SKIP_DIR or dl.endswith(".bak") or dl.endswith("_bak") or "backup" in dl
SKIP_NAME_SUBSTR = ("package-lock.json", "yarn.lock", "poetry.lock", "pnpm-lock", ".min.js", ".min.css")
MAX_FILE_BYTES = 400_000   # skip huge single files (generated/data)

# EPISTEMIC MEMBRANE — Treasurer's authoritative superseded-path list (raw-corpus MANIFEST tier map).
# Belt-and-suspenders over the draft-dir exclusion: catches superseded .md files OUTSIDE draft dirs
# (e.g. treasurer/foundations/action_logs/*). Clean join: exclude any repo file whose abs path is listed.
SUPERSEDED_LIST = os.path.join(os.environ.get("TREASURER_ROOT", os.path.expanduser("~/treasurer")), "foundations/careers/training_data/v2/raw_corpus/superseded_md_paths.txt")
SUPERSEDED_PATHS = set()
if os.path.isfile(SUPERSEDED_LIST):
    with open(SUPERSEDED_LIST) as _f:
        SUPERSEDED_PATHS = {ln.strip() for ln in _f if ln.strip()}


def iter_repo_files(repo_dir):
    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if not _is_junk_dir(d) and not d.endswith(".egg-info")]
        for fn in files:
            if any(s in fn for s in SKIP_NAME_SUBSTR):
                continue
            ext = os.path.splitext(fn)[1].lower()
            if ext not in KEEP_EXT and fn not in ("Dockerfile", "Makefile", ".env.example"):
                continue
            p = os.path.join(root, fn)
            if p in SUPERSEDED_PATHS:   # membrane: authoritative superseded-tier exclusion
                continue
            try:
                if os.path.getsize(p) > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            yield p


def main():
    out_path = sys.argv[1]
    n_files = n_repos = n_missing = total_chars = 0
    missing = []
    with open(out_path, "w") as fout:
        for vis, repos in (("public", PUBLIC), ("private", PRIVATE)):
            for name in repos:
                repo_dir = os.path.join(BASE, name)
                if not os.path.isdir(repo_dir):
                    n_missing += 1; missing.append(name); continue
                n_repos += 1
                for p in iter_repo_files(repo_dir):
                    try:
                        text = open(p, encoding="utf-8", errors="replace").read().strip()
                    except Exception:
                        continue
                    if not text:
                        continue
                    rel = os.path.relpath(p, BASE)
                    header = f"# repo: {name} ({vis})\n# file: {rel}\n\n"
                    fout.write(json.dumps({"text": header + text, "source": rel,
                                           "repo": name, "visibility": vis}) + "\n")
                    n_files += 1; total_chars += len(text)
    print(f"repos included: {n_repos}  files: {n_files}  ~tokens: {total_chars//4:,}  "
          f"({total_chars/1e6:.1f}M chars) → {out_path}")
    if missing:
        print(f"MISSING (not found under {BASE}, verify name): {missing}")


if __name__ == "__main__":
    main()
