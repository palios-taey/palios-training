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
from pathlib import Path

from one_wire import render_record


TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
# XML function/parameter form — the format that caused the collision; must NOT appear.
XML_TOOLCALL_RE = re.compile(r"<tool_call>\s*<function=|<function=|<parameter=", re.DOTALL)
JINJA_COMMENT_RE = re.compile(r"{#.*?#}", re.DOTALL)


def _fail(msg: str) -> None:
    print(f"GATE FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def _validate_tool_call_body(body: str, source: str) -> None:
    try:
        payload = json.loads(body.strip())
    except json.JSONDecodeError as exc:
        _fail(f"{source}: <tool_call> body is not valid JSON: {exc}")
    if not isinstance(payload, dict):
        _fail(f"{source}: <tool_call> body must be a JSON object")
    name = payload.get("name")
    arguments = payload.get("arguments")
    if not isinstance(name, str) or not name.strip():
        _fail(f"{source}: <tool_call> JSON missing non-empty string 'name'")
    if not isinstance(arguments, dict):
        _fail(f"{source}: <tool_call> JSON missing object 'arguments'")


def _validate_rendered_wire(rendered: str, source: str) -> int:
    if XML_TOOLCALL_RE.search(rendered):
        _fail(f"{source}: rendered XML-format tool call (<function=>/<parameter=>)")
    bodies = TOOL_CALL_BLOCK_RE.findall(rendered)
    for body_index, body in enumerate(bodies, 1):
        _validate_tool_call_body(body, f"{source} tool_call[{body_index}]")
    return len(bodies)


def check_training_data(path: str, sample: int = 0, template_path: str | None = None) -> None:
    tool_records = no_tool_records = inspected = 0
    with open(path, encoding="utf-8") as f:
        for i, ln in enumerate(f):
            if sample > 0 and inspected >= sample:
                break
            if not ln.strip():
                continue
            try:
                d = json.loads(ln)
            except json.JSONDecodeError as exc:
                _fail(f"{path}:{i + 1}: invalid JSONL row: {exc}")
            try:
                rendered = render_record(d, template_path=template_path)
            except Exception as exc:
                _fail(f"{path}:{i + 1}: canonical render failed: {exc}")
            n_calls = _validate_rendered_wire(rendered, f"{path}:{i + 1}")
            if n_calls:
                tool_records += 1
            else:
                no_tool_records += 1
            inspected += 1
    scope = "all rows" if sample <= 0 else f"sample={sample}"
    print(
        f"  train: {tool_records} rendered Hermes-JSON tool-call records, "
        f"0 XML, {no_tool_records} no-tool ({inspected} inspected; {scope})"
    )


def check_template(path: str) -> None:
    src = open(path, encoding="utf-8").read()
    runtime_src = JINJA_COMMENT_RE.sub("", src)
    if (
        XML_TOOLCALL_RE.search(runtime_src)
        or "<function=' + tool_call.name" in runtime_src
        or "<parameter=' ~" in runtime_src
    ):
        _fail(f"inference template {path} still renders/instructs XML <function=>/<parameter=> tool calls")
    # must emit the JSON form
    if "tool_call>" not in runtime_src or "name" not in runtime_src:
        _fail(f"inference template {path} has no recognizable <tool_call> JSON render")
    fixture = {
        "messages": [
            {"role": "user", "content": "Call lookup."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "lookup", "arguments": "{\"query\":\"taey\"}"}}],
            },
        ],
        "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
    }
    rendered = render_record(fixture, template_path=path)
    if _validate_rendered_wire(rendered, f"template fixture {path}") < 1:
        _fail(f"inference template {path} did not render the structured tool_call fixture")
    print(f"  template: {path} renders parseable Hermes-JSON, 0 XML <function=>/<parameter=>")


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
    p.add_argument("--sample", type=int, default=0, help="Optional max rendered records to inspect; default 0 means all")
    a = p.parse_args()
    template_path = Path(a.template)
    print("Tool-call wire-format gate (canonical = Hermes-JSON):")
    check_training_data(a.train_sample, a.sample, str(template_path))
    check_template(str(template_path))
    if a.eval_probes:
        check_eval_probes(a.eval_probes)
    print("GATE PASS: train == inference-template == eval all on Hermes-JSON.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
