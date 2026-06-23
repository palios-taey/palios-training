#!/usr/bin/env python3
"""Canonical Qwen3.5 tool-use wire compiler.

This module is the single Python consolidation point for the dense-9B chat wire:
training tokenization/masking, format gating, and render-equivalence checks all
use the same Jinja template file that serving consumes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


TEMPLATE_PATH = Path(__file__).resolve().with_name("qwen3.5-tooluse.jinja")


class EmptyAssistantMask(ValueError):
    """Raised when a rendered SFT record has no assistant target tokens."""


def load_chat_template(template_path: str | Path | None = None) -> str:
    return Path(template_path or TEMPLATE_PATH).read_text(encoding="utf-8")


def install_chat_template(tokenizer: Any, template_path: str | Path | None = None) -> Path:
    path = Path(template_path or TEMPLATE_PATH)
    tokenizer.chat_template = load_chat_template(path)
    return path


def _messages_and_tools(record_or_messages: Any, tools: Any = None) -> tuple[list[dict], Any]:
    if isinstance(record_or_messages, dict):
        messages = record_or_messages.get("messages")
        record_tools = record_or_messages.get("tools")
    else:
        messages = record_or_messages
        record_tools = None
    if not isinstance(messages, list) or not messages:
        raise ValueError("record has no messages list")
    return messages, record_tools if tools is None else tools


def _render_with_jinja2(
    messages: list[dict],
    tools: Any,
    *,
    add_generation_prompt: bool,
    enable_thinking: bool,
    template_path: str | Path | None,
) -> str:
    from jinja2 import Environment

    def raise_exception(message: str) -> None:
        raise ValueError(message)

    def fromjson(value: str) -> Any:
        return json.loads(value)

    env = Environment(autoescape=False)
    env.filters["fromjson"] = fromjson
    env.filters["tojson"] = lambda value: json.dumps(value, ensure_ascii=False)
    template = env.from_string(load_chat_template(template_path))
    return template.render(
        messages=messages,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
        add_vision_id=False,
        raise_exception=raise_exception,
    )


def render_record(
    record_or_messages: Any,
    tokenizer: Any | None = None,
    *,
    tools: Any = None,
    add_generation_prompt: bool = False,
    enable_thinking: bool = False,
    template_path: str | Path | None = None,
) -> str:
    messages, row_tools = _messages_and_tools(record_or_messages, tools)
    if tokenizer is not None:
        if not getattr(tokenizer, "chat_template", None):
            install_chat_template(tokenizer, template_path)
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            tools=row_tools,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
    return _render_with_jinja2(
        messages,
        row_tools,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
        template_path=template_path,
    )


def tokenize_sft_record(
    record_or_messages: Any,
    tokenizer: Any,
    *,
    tools: Any = None,
    template_path: str | Path | None = None,
) -> tuple[list[int], list[int]]:
    messages, row_tools = _messages_and_tools(record_or_messages, tools)
    record = {"messages": messages}
    if row_tools:
        record["tools"] = row_tools

    full_text = render_record(record, tokenizer, template_path=template_path)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    labels = [-100] * len(full_ids)

    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        prefix_messages = messages[:index]
        inclusive_messages = messages[: index + 1]
        prefix_record = {"messages": prefix_messages}
        inclusive_record = {"messages": inclusive_messages}
        if row_tools:
            prefix_record["tools"] = row_tools
            inclusive_record["tools"] = row_tools
        prefix_text = (
            render_record(prefix_record, tokenizer, template_path=template_path)
            if prefix_messages
            else ""
        )
        inclusive_text = render_record(
            inclusive_record,
            tokenizer,
            template_path=template_path,
        )
        start = len(tokenizer.encode(prefix_text, add_special_tokens=False)) if prefix_text else 0
        end = len(tokenizer.encode(inclusive_text, add_special_tokens=False))
        for token_index in range(start, min(end, len(full_ids))):
            labels[token_index] = full_ids[token_index]

    if all(label == -100 for label in labels):
        raise EmptyAssistantMask("rendered record has no assistant target tokens")
    return full_ids, labels


def templated_token_len(record_or_messages: Any, tokenizer: Any, *, tools: Any = None) -> int:
    text = render_record(record_or_messages, tokenizer, tools=tools)
    return len(tokenizer.encode(text, add_special_tokens=False))
