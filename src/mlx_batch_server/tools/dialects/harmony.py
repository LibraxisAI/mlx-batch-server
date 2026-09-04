# SPDX-License-Identifier: Apache-2.0
"""Cumulative Harmony channel and tool-recipient dialect.

The scheduler-facing behavior and marker contract are adapted from oMLX
``omlx/adapter/output_parser.py`` at
e467261edc786efd33b1e9023d5c4a827f8aa1c1. This target-owned implementation
operates on cumulative text rather than tokenizer sessions.
"""

from __future__ import annotations

import re

from ..parser import DialectParse, DialectToolCall
from ._common import (
    hold_partial_marker,
    json_value_end,
    skip_ascii_space,
    stable_call_id,
)

_ASSISTANT = "<|start|>assistant<|channel|>"
_MESSAGE = "<|message|>"
_CALL_END = "<|call|>"
_TURN_END = "<|end|>"
_RETURN = "<|return|>"
_CONTROL_MARKERS = (
    _ASSISTANT,
    "<|start|>",
    "<|channel|>",
    _MESSAGE,
    "<|constrain|>",
    _CALL_END,
    _TURN_END,
    _RETURN,
)
_CONTROL_RE = re.compile(r"<\|(?:start|channel|message|constrain|call|end|return)\|>")
_RECIPIENT_RE = re.compile(r"(?:to|recipient)=functions\.([A-Za-z_][\w.-]*)")


class HarmonyDialectError(ValueError):
    """Harmony emitted a final tool record that cannot be normalized."""


def _outside_text(text: str) -> str:
    safe = hold_partial_marker(text, _CONTROL_MARKERS)
    return _CONTROL_RE.sub("", safe)


def _tool_call_end(text: str, body_start: int) -> int | None:
    """Return the structural call marker, never one inside JSON text."""

    value_start = skip_ascii_space(text, body_start)
    if value_start < len(text) and text[value_start] in "{[":
        value_end = json_value_end(text, value_start)
        if value_end is None:
            return None
        marker_start = skip_ascii_space(text, value_end)
        return marker_start if text.startswith(_CALL_END, marker_start) else None

    quoted = False
    escaped = False
    cursor = body_start
    while cursor < len(text):
        char = text[cursor]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            cursor += 1
            continue
        if char == '"':
            quoted = True
            cursor += 1
            continue
        if text.startswith(_CALL_END, cursor):
            return cursor
        cursor += 1
    return None


class HarmonyToolDialect:
    """Expose final-channel text and normalize Harmony tool recipients."""

    def parse(self, text: str, *, final: bool) -> DialectParse:
        visible: list[str] = []
        calls: list[DialectToolCall] = []
        cursor = 0

        while True:
            start = text.find(_ASSISTANT, cursor)
            if start < 0:
                visible.append(_outside_text(text[cursor:]))
                break

            visible.append(_outside_text(text[cursor:start]))
            header_start = start + len(_ASSISTANT)
            message_start = text.find(_MESSAGE, header_start)
            if message_start < 0:
                if final:
                    raise HarmonyDialectError("unterminated Harmony channel header")
                break

            header = text[header_start:message_start]
            body_start = message_start + len(_MESSAGE)
            recipient = _RECIPIENT_RE.search(header)
            channel = header.split(None, 1)[0].split("<|", 1)[0].strip()

            if recipient is not None:
                call_end = _tool_call_end(text, body_start)
                if call_end is None:
                    arguments = hold_partial_marker(
                        text[body_start:], (_CALL_END, _TURN_END)
                    )
                    calls.append(
                        DialectToolCall(
                            index=len(calls),
                            call_id=stable_call_id("harmony", len(calls)),
                            name=recipient.group(1),
                            arguments=arguments,
                            complete=False,
                        )
                    )
                    if final:
                        raise HarmonyDialectError("unterminated Harmony tool call")
                    break

                calls.append(
                    DialectToolCall(
                        index=len(calls),
                        call_id=stable_call_id("harmony", len(calls)),
                        name=recipient.group(1),
                        arguments=text[body_start:call_end],
                        complete=True,
                    )
                )
                cursor = call_end + len(_CALL_END)
                continue

            turn_end = text.find(_TURN_END, body_start)
            if turn_end < 0:
                if channel == "final":
                    body = text[body_start:]
                    visible.append(_outside_text(body))
                break

            if channel == "final":
                visible.append(text[body_start:turn_end])
            cursor = turn_end + len(_TURN_END)

        return DialectParse(visible_text="".join(visible), calls=tuple(calls))
