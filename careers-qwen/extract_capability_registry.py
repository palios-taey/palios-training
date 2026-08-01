#!/usr/bin/env python3
"""Harvest the REAL capability surface of the repos Taey uses.

WHY THIS EXISTS. SFT pairs teach Taey to invoke things. A pair naming a flag that does not
exist is worse than no pair: it teaches confident invocation of something that fails, and it
reads as correct to a reviewer. The defence cannot be diligence — it has to be mechanical. So
the capability inventory is HARVESTED FROM THE CODE, and every pair is later checked against
it. Nothing in this registry is authored by a model, so nothing in it can be fabricated.

WHY STATIC AST, NOT `--help`. Running `--help` would be the obvious way to learn a CLI's flags,
and it is the wrong way here: it imports the module, which runs import-time side effects across
19 repositories — servers, clients, file writes. An extractor must not be able to damage what it
is reading. AST parsing sees the same `add_argument` calls without executing anything.

WHAT IT CANNOT SEE, recorded so nobody trusts it further than it earns:
  - flags built dynamically (a loop over a config, `**kwargs`, argparse parents)
  - routes registered at runtime rather than by decorator
  - anything behind a conditional import
The registry is therefore a floor, not a ceiling: everything in it is real, but absence is not
proof of absence. A pair citing something absent gets held for a human, never auto-approved.

Output: one JSON registry, every entry carrying repo@commit:file:line provenance.
"""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

SKIP_DIR_PARTS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "site-packages",
    # the two exclusions the CPT corpus already enforces, for the same reasons
    "archive", "archived", "deprecated", "superseded", "obsolete", "legacy",
    "tests", "test", "testing", "__tests__", "spec", "specs",
}
ROUTE_DECORATORS = {"get", "post", "put", "delete", "patch", "route", "head", "options"}


def git(repo: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def wanted(rel: str) -> bool:
    p = Path(rel)
    if any(part.lower() in SKIP_DIR_PARTS for part in p.parts[:-1]):
        return False
    if p.stem.lower().startswith("test_") or p.stem.lower().endswith("_test"):
        return False
    return p.suffix == ".py"


def const(node) -> str | None:
    """Literal value of an AST node, or None when it is computed.

    Computed values are deliberately dropped rather than guessed: a flag whose name is built at
    runtime is exactly the case where a plausible reconstruction would be wrong.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def extract_file(repo: Path, rel: str, source: str) -> tuple[list[dict], list[dict]]:
    """Return (cli_arguments, http_routes) found in one file."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [], []

    clis: list[dict] = []
    routes: list[dict] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func

        # parser.add_argument("--flag", ...)
        if isinstance(fn, ast.Attribute) and fn.attr == "add_argument":
            flags = [c for c in (const(a) for a in node.args) if c]
            if not flags:
                continue
            kw = {k.arg: k.value for k in node.keywords if k.arg}
            entry = {
                "flags": flags,
                "required": bool(
                    isinstance(kw.get("required"), ast.Constant) and kw["required"].value is True
                ),
                "has_default": "default" in kw,
                "choices": [
                    c for c in (const(e) for e in kw["choices"].elts)
                    if c
                ] if isinstance(kw.get("choices"), (ast.List, ast.Tuple)) else [],
                "help": const(kw.get("help")) if "help" in kw else None,
                "file": rel,
                "line": node.lineno,
            }
            clis.append(entry)

        # @app.get("/path") / @router.post("/path")
        if isinstance(fn, ast.Attribute) and fn.attr in ROUTE_DECORATORS:
            path = next((c for c in (const(a) for a in node.args) if c), None)
            if path and path.startswith("/"):
                routes.append({
                    "method": fn.attr.upper(),
                    "path": path,
                    "file": rel,
                    "line": node.lineno,
                })

    return clis, routes


def extract_repo(name: str, repo: Path) -> dict:
    head = git(repo, "rev-parse", "HEAD")
    # The registry reads `git show HEAD:<file>`, so it describes COMMITTED state. A dirty tree
    # therefore produces a registry that certifies flags which may not match the code on disk --
    # and it would look complete either way. Found by spot-check: a --pack-set flag added and not
    # yet committed was absent from the registry with no indication anything was missing.
    # Reported per repo rather than fatal: a clean registry of committed code is still valid and
    # useful; what must never happen is that the divergence goes unrecorded.
    dirty = [l for l in git(repo, "status", "--porcelain").splitlines() if l.strip()]
    if dirty:
        print(f"  WARNING {name}: {len(dirty)} uncommitted change(s) — registry reflects "
              f"HEAD {head[:8]}, NOT the working tree", file=sys.stderr)
    files = [f for f in git(repo, "ls-files").splitlines() if wanted(f)]
    programs: dict[str, list[dict]] = {}
    routes: list[dict] = []
    unreadable: list[str] = []

    for rel in files:
        blob = subprocess.run(
            ["git", "-C", str(repo), "show", f"HEAD:{rel}"], capture_output=True
        )
        if blob.returncode != 0:
            unreadable.append(rel)
            continue
        try:
            source = blob.stdout.decode("utf-8")
        except UnicodeDecodeError:
            unreadable.append(rel)
            continue
        c, r = extract_file(repo, rel, source)
        if c:
            programs.setdefault(rel, []).extend(c)
        routes.extend(r)

    return {
        "repo": name,
        "commit": head,
        "checkout_path": str(repo),
        "files_scanned": len(files),
        "working_tree_dirty": len(dirty),
        "programs": programs,
        "routes": routes,
        "unreadable": unreadable,
        "counts": {
            "programs": len(programs),
            "arguments": sum(len(v) for v in programs.values()),
            "routes": len(routes),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repos", required=True,
                    help="comma-separated name=path pairs, or a JSON file mapping name->path")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    spec = Path(args.repos)
    if spec.is_file():
        repo_map = {k: Path(v) for k, v in json.loads(spec.read_text()).items()}
    else:
        repo_map = {}
        for item in args.repos.split(","):
            k, _, v = item.partition("=")
            repo_map[k.strip()] = Path(v.strip())

    missing = [n for n, p in repo_map.items() if not (p / ".git").is_dir()]
    if missing:
        sys.exit(f"REFUSE: not a git checkout: {', '.join(missing)} — "
                 f"a registry built from a partial set would look complete.")

    registry = {"repos": [], "totals": {"programs": 0, "arguments": 0, "routes": 0}}
    for name, path in sorted(repo_map.items()):
        entry = extract_repo(name, path)
        registry["repos"].append(entry)
        for k in registry["totals"]:
            registry["totals"][k] += entry["counts"][k]
        print(f"  {name:36s} programs={entry['counts']['programs']:4d} "
              f"args={entry['counts']['arguments']:5d} routes={entry['counts']['routes']:4d}")

    Path(args.out).write_text(json.dumps(registry, indent=2))
    t = registry["totals"]
    print(f"\n  TOTAL programs={t['programs']} arguments={t['arguments']} routes={t['routes']}")
    print(f"  registry -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
