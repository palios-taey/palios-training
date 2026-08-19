#!/usr/bin/env python3
"""Append and validate the production training lifecycle journal."""

import argparse
import fcntl
import json
import os
import re
import sys
from datetime import datetime, timezone


STATES = (
    "SPEC_VALID",
    "NODE_DEPLOY",
    "TRAINING",
    "FRAGMENTATION_EXIT",
    "CHECKPOINT_SAVED",
    "BAKE_COMPLETE",
    "THOR_DELIVERED",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class LifecycleError(ValueError):
    pass


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_ts(value):
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise LifecycleError(f"invalid UTC timestamp {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise LifecycleError(f"timestamp is not UTC: {value!r}")
    return parsed


def allowed_next(previous, state):
    if previous is None:
        return state == "SPEC_VALID"
    return state in {
        "SPEC_VALID": {"NODE_DEPLOY"},
        "NODE_DEPLOY": {"TRAINING"},
        "TRAINING": {"FRAGMENTATION_EXIT", "CHECKPOINT_SAVED"},
        "FRAGMENTATION_EXIT": {"FRAGMENTATION_EXIT", "CHECKPOINT_SAVED"},
        "CHECKPOINT_SAVED": {"BAKE_COMPLETE"},
        "BAKE_COMPLETE": {"THOR_DELIVERED"},
        "THOR_DELIVERED": set(),
    }[previous]


def validate_event(event, line_no):
    if not isinstance(event, dict):
        raise LifecycleError(f"line {line_no} is not a JSON object")
    for field in ("ts", "state", "run_tag"):
        if not isinstance(event.get(field), str) or not event[field]:
            raise LifecycleError(f"line {line_no} has no non-empty {field}")
    parse_ts(event["ts"])
    state = event["state"]
    if state not in STATES:
        raise LifecycleError(f"line {line_no} has unknown state {state!r}")
    if state in {"TRAINING", "FRAGMENTATION_EXIT", "CHECKPOINT_SAVED"}:
        if not isinstance(event.get("step"), int) or event["step"] < 0:
            raise LifecycleError(f"line {line_no} state {state} has no non-negative integer step")
    if state in {"FRAGMENTATION_EXIT", "CHECKPOINT_SAVED"}:
        if not isinstance(event.get("checkpoint_path"), str) or not event["checkpoint_path"]:
            raise LifecycleError(f"line {line_no} state {state} has no checkpoint_path")
    if state in {"BAKE_COMPLETE", "THOR_DELIVERED"}:
        digest = event.get("artifact_sha256")
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            raise LifecycleError(f"line {line_no} state {state} has no valid artifact_sha256")
    if state == "THOR_DELIVERED":
        if not isinstance(event.get("served_root"), str) or not event["served_root"]:
            raise LifecycleError(f"line {line_no} state THOR_DELIVERED has no served_root")


def validate_transition_evidence(previous, event, line_no):
    if previous and event["state"] == "THOR_DELIVERED":
        if event["artifact_sha256"] != previous.get("artifact_sha256"):
            raise LifecycleError(
                f"line {line_no} THOR_DELIVERED artifact_sha256 differs from BAKE_COMPLETE"
            )
    if previous and "step" in previous and "step" in event:
        if event["step"] < previous["step"]:
            raise LifecycleError(
                f"line {line_no} step {event['step']} precedes prior step {previous['step']}"
            )


def load_events(handle):
    handle.seek(0)
    events = []
    previous_ts = None
    previous_state = None
    run_tag = None
    for line_no, raw in enumerate(handle, 1):
        if not raw.strip():
            raise LifecycleError(f"line {line_no} is empty")
        try:
            event = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LifecycleError(f"line {line_no} is not JSON: {exc.msg}") from exc
        validate_event(event, line_no)
        current_ts = parse_ts(event["ts"])
        if previous_ts is not None and current_ts < previous_ts:
            raise LifecycleError(
                f"line {line_no} timestamp {event['ts']} precedes the prior event"
            )
        if run_tag is None:
            run_tag = event["run_tag"]
        elif event["run_tag"] != run_tag:
            raise LifecycleError(
                f"line {line_no} run_tag={event['run_tag']!r} differs from {run_tag!r}"
            )
        if not allowed_next(previous_state, event["state"]):
            raise LifecycleError(
                f"line {line_no} transition {previous_state or '<start>'} -> {event['state']} is invalid"
            )
        validate_transition_evidence(events[-1] if events else None, event, line_no)
        events.append(event)
        previous_ts = current_ts
        previous_state = event["state"]
    return events


def add_if_set(event, name, value, converter=None):
    if value is not None:
        event[name] = converter(value) if converter else value


def append_event(args):
    os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
    with open(args.log, "a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        events = load_events(handle)
        event = {"ts": utc_now(), "state": args.state, "run_tag": args.run_tag}
        add_if_set(event, "step", args.step, int)
        add_if_set(event, "checkpoint_path", args.checkpoint_path)
        add_if_set(event, "artifact_sha256", args.artifact_sha256)
        add_if_set(event, "served_root", args.served_root)
        add_if_set(event, "total_steps", args.total_steps, int)
        add_if_set(event, "session_limit", args.session_limit, int)
        add_if_set(event, "nodes", args.nodes, int)
        validate_event(event, len(events) + 1)
        previous_state = events[-1]["state"] if events else None
        if not allowed_next(previous_state, event["state"]):
            raise LifecycleError(
                f"transition {previous_state or '<start>'} -> {event['state']} is invalid"
            )
        validate_transition_evidence(events[-1] if events else None, event, len(events) + 1)
        if events and parse_ts(event["ts"]) < parse_ts(events[-1]["ts"]):
            raise LifecycleError("new timestamp precedes the last journal timestamp")
        line = json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
        handle.seek(0, os.SEEK_END)
        os.write(handle.fileno(), line.encode("utf-8"))
        os.fsync(handle.fileno())
        print(
            f"LIFECYCLE APPEND state={event['state']} run_tag={event['run_tag']} "
            f"line={len(events) + 1}"
        )


def validate_log(args):
    with open(args.log, encoding="utf-8") as handle:
        events = load_events(handle)
    if not events:
        raise LifecycleError("journal has no transitions")
    if args.expected_fragmentation is not None:
        observed = sum(event["state"] == "FRAGMENTATION_EXIT" for event in events)
        if observed != args.expected_fragmentation:
            raise LifecycleError(
                f"fragmentation exits observed={observed} expected={args.expected_fragmentation}"
            )
    print(
        f"LIFECYCLE VALID transitions={len(events)} last={events[-1]['state']} "
        f"fragmentation_exits={sum(event['state'] == 'FRAGMENTATION_EXIT' for event in events)}"
    )


def last_state(args):
    if not os.path.isfile(args.log):
        print("UNKNOWN")
        return
    with open(args.log, encoding="utf-8") as handle:
        events = load_events(handle)
    if not events:
        print("UNKNOWN")
    elif args.json:
        print(json.dumps(events[-1], sort_keys=True, separators=(",", ":")))
    else:
        print(events[-1]["state"])


def build_parser():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    append = sub.add_parser("append")
    append.add_argument("--log", required=True)
    append.add_argument("--state", required=True, choices=STATES)
    append.add_argument("--run-tag", required=True)
    append.add_argument("--step")
    append.add_argument("--checkpoint-path")
    append.add_argument("--artifact-sha256")
    append.add_argument("--served-root")
    append.add_argument("--total-steps")
    append.add_argument("--session-limit")
    append.add_argument("--nodes")
    append.set_defaults(func=append_event)
    validate = sub.add_parser("validate")
    validate.add_argument("--log", required=True)
    validate.add_argument("--expected-fragmentation", type=int)
    validate.set_defaults(func=validate_log)
    last = sub.add_parser("last")
    last.add_argument("--log", required=True)
    last.add_argument("--json", action="store_true")
    last.set_defaults(func=last_state)
    return parser


def main():
    args = build_parser().parse_args()
    try:
        args.func(args)
    except (LifecycleError, OSError, ValueError) as exc:
        print(f"LIFECYCLE INVALID: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
