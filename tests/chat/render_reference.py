#!/usr/bin/env python3
"""Render the test cases with the real Qwen3.8 chat template.

This is the ground truth the C++ renderer (src/chat.cpp, via tools/chat_test.cpp) must match
byte for byte.  Run it with the project venv:

    .work/venv/Scripts/python tests/chat/render_reference.py tests/chat/cases.json

It writes one JSON object per line (JSON Lines) with "name" plus either "output" or "error",
which is exactly what tools/chat_test.cpp writes.

The environment mirrors what HuggingFace transformers uses for chat templates:
ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols]),
a tojson filter equal to json.dumps(..., ensure_ascii=False, separators=(", ", ": ")), and a
raise_exception global.  A real OpenAI-compatible server parses function arguments that arrive
as JSON strings before templating (the template iterates `arguments|items`, which needs a
mapping), so we do the same here.
"""

import sys as _sys
_sys.stdout.reconfigure(encoding="utf-8")
import copy
import json
import os
import sys

import jinja2
import jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment


class TemplateError(Exception):
    pass


def raise_exception(message):
    raise TemplateError(message)


def build_template():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "..", ".work", "ref", "chat_template.jinja")
    with open(path, encoding="utf-8") as handle:
        source = handle.read()

    env = ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=[jinja2.ext.loopcontrols],
    )
    env.filters["tojson"] = lambda value: json.dumps(
        value, ensure_ascii=False, indent=None, separators=(", ", ": "), sort_keys=False
    )
    env.globals["raise_exception"] = raise_exception
    return env.from_string(source)


def normalize_messages(messages):
    """Parse OpenAI-style function-call arguments that are JSON strings into objects."""
    messages = copy.deepcopy(messages)
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            target = tool_call
            if isinstance(tool_call, dict) and "function" in tool_call:
                target = tool_call["function"]
            if isinstance(target, dict) and isinstance(target.get("arguments"), str):
                try:
                    target["arguments"] = json.loads(target["arguments"])
                except ValueError:
                    pass
    return messages


def render_case(template, case):
    options = case.get("options", {})
    kwargs = {"messages": normalize_messages(case.get("messages", []))}
    tools = case.get("tools", None)
    if tools is not None:
        kwargs["tools"] = tools
    kwargs["add_generation_prompt"] = options.get("add_generation_prompt", True)
    kwargs["enable_thinking"] = options.get("enable_thinking", True)
    kwargs["reasoning_effort"] = options.get("reasoning_effort", "xhigh")
    kwargs["preserve_thinking"] = options.get("preserve_thinking", True)
    return template.render(**kwargs)


def main():
    if len(sys.argv) != 2:
        sys.stderr.write("usage: render_reference.py cases.json\n")
        return 2
    with open(sys.argv[1], encoding="utf-8") as handle:
        cases = json.load(handle)

    template = build_template()
    for case in cases:
        record = {"name": case["name"]}
        try:
            record["output"] = render_case(template, case)
        except Exception as exc:  # noqa: BLE001 - the message is the contract
            record["error"] = str(exc)
        sys.stdout.write(json.dumps(record, ensure_ascii=False))
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
