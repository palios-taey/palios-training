#!/usr/bin/env python3
"""Prelaunch tool-call wire-format gate — FAIL-CLOSED.

Why this exists: the 9B tool-calling broke because three surfaces disagreed on the
tool-call wire format — the training data used Hermes-JSON (`<tool_call>{"name":..,
"arguments":..}</tool_call>`) while the inference template (`qwen3.5-fixed.jinja`)
commanded XML (`<tool_call><function=..><parameter=..></tool_call>`). The model trained
on JSON but was told at inference to emit XML. This gate makes that class of bug
structurally impossible to ship: it asserts all three surfaces (training data,
inference template, eval probes) speak the SAME canonical format, and exits non-zero
(blocks the run) on any divergence.

Run it BEFORE any training/serving launch:
    python3 toolcall_format_gate.py \
        --train-sample /path/to/tools_sft.jsonl \
        --template /path/to/qwen3.5-fixed.jinja \
        --eval-probes /path/to/tool_call_probes.jsonl   # optional

Canonical format = Hermes-JSON: a single JSON object inside <tool_call></tool_call> tags.
"""
import argparse
import json
import re
import sys

# Hermes-JSON: <tool_call> ... {json object with "name"} ... </tool_call>
HERMES_RE = re.compile(r"<tool_call>\s*\{.*?\"name\".*?\}\s*</tool_call>", re.DOTALL)
# XML function/parameter form — the format that caused the collision; must NOT appear.
XML_TOOLCALL_RE = re.compile(r"<tool_call>\s*<function=")


def _fail(msg: str) -> None:
    print(f"GATE FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def check_training_data(path: str, sample: int = 5000) -> None:
    hermes = xml = neither = 0
    with open(path, encoding="utf-8") as f:
        for i, ln in enumerate(f):
            if i >= sample:
                break
            try:
                d = json.loads(ln)
            except Exception:
                continue
            blob = "".join(
                m.get("content") or ""
                for m in d.get("messages", [])
                if isinstance(m.get("content"), str)
            )
            if XML_TOOLCALL_RE.search(blob):
                xml += 1
            elif HERMES_RE.search(blob):
                hermes += 1
            else:
                neither += 1
    if xml:
        _fail(f"training data has {xml} XML-format tool calls (must be Hermes-JSON)")
    print(f"  train: {hermes} Hermes-JSON tool-call records, 0 XML, {neither} no-tool (sampled {sample})")


def check_template(path: str) -> None:
    src = open(path, encoding="utf-8").read()
    if XML_TOOLCALL_RE.search(src) or "<function=' + tool_call.name" in src or "<parameter=' ~" in src:
        _fail(f"inference template {path} still renders/instructs XML <function=>/<parameter=> tool calls")
    # must emit the JSON form
    if "tool_call>" not in src or "name" not in src:
        _fail(f"inference template {path} has no recognizable <tool_call> JSON render")
    print(f"  template: {path} renders Hermes-JSON, 0 XML <function=>/<parameter=>")


def check_eval_probes(path: str) -> None:
    src = open(path, encoding="utf-8").read()
    if XML_TOOLCALL_RE.search(src):
        _fail(f"eval probes {path} reference XML-format tool calls")
    print(f"  eval probes: {path} consistent (no XML tool-call format)")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--train-sample", required=True)
    p.add_argument("--template", required=True)
    p.add_argument("--eval-probes", default=None)
    p.add_argument("--sample", type=int, default=5000)
    a = p.parse_args()
    print("Tool-call wire-format gate (canonical = Hermes-JSON):")
    check_training_data(a.train_sample, a.sample)
    check_template(a.template)
    if a.eval_probes:
        check_eval_probes(a.eval_probes)
    print("GATE PASS: train == inference-template == eval all on Hermes-JSON.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
