# SPDX-License-Identifier: Apache-2.0
"""Cumulative Qwen XML/JSON tool-call dialect.

Protocol behavior is adapted from oMLX ``omlx/api/tool_calling.py`` at
e467261edc786efd33b1e9023d5c4a827f8aa1c1 and the MTPLX bridge at
6d0ddf0575faa9acf77e63c57e48ea1602a7e4ab. This target-owned version is
stateless, emits cumulative snapshots, and never exposes control markers.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..parser import DialectParse, DialectToolCall
from ._common import (
    hold_partial_marker,
    json_value_end,
    json_value_prefix,
    skip_ascii_space,
    stable_call_id,
)

_OPEN = "<tool_call>"
_CLOSE = "</tool_call>"
_FUNCTION_CLOSE = "</function>"
_FUNCTION_RE = re.compile(r"<function=([A-Za-z_][\w.-]*)>")
_PARAMETER_OPEN_RE = re.compile(r"<parameter=([A-Za-z_][\w.-]*)>")
_PARAMETER_CLOSE = "</parameter>"


class QwenDialectError(ValueError):
    """Qwen emitted a final tool envelope that cannot be normalized."""


def _decode_json_string(payload: str) -> str | None:
    try:
        value = json.loads(payload)
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, str) and value else None


def _json_string_end(text: str, start: int) -> int | None:
    if start >= len(text) or text[start] != '"':
        return None
    escaped = False
    for index in range(start + 1, len(text)):
        char = text[index]
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            return index + 1
    return None


def _json_value_span_end(text: str, start: int) -> int | None:
    start = skip_ascii_space(text, start)
    if start >= len(text):
        return None
    if text[start] in "{[":
        return json_value_end(text, start)
    if text[start] == '"':
        return _json_string_end(text, start)

    cursor = start
    while cursor < len(text) and text[cursor] not in ",}":
        cursor += 1
    if cursor == len(text):
        return None
    candidate = text[start:cursor].strip()
    try:
        json.loads(candidate)
    except (TypeError, ValueError):
        return None
    return cursor


def _top_level_json_members(
    payload: str,
) -> dict[str, tuple[int, int | None]]:
    """Lex top-level object fields without mistaking nested keys for them."""

    cursor = skip_ascii_space(payload, 0)
    if cursor >= len(payload) or payload[cursor] != "{":
        return {}
    cursor += 1
    members: dict[str, tuple[int, int | None]] = {}

    while True:
        cursor = skip_ascii_space(payload, cursor)
        if cursor >= len(payload) or payload[cursor] == "}":
            return members
        key_end = _json_string_end(payload, cursor)
        if key_end is None:
            return members
        key = _decode_json_string(payload[cursor:key_end])
        if key is None:
            return members
        if key in members:
            raise QwenDialectError(f"duplicate JSON tool field: {key}")

        cursor = skip_ascii_space(payload, key_end)
        if cursor >= len(payload) or payload[cursor] != ":":
            return members
        value_start = skip_ascii_space(payload, cursor + 1)
        value_end = _json_value_span_end(payload, value_start)
        members[key] = (value_start, value_end)
        if value_end is None:
            return members

        cursor = skip_ascii_space(payload, value_end)
        if cursor >= len(payload) or payload[cursor] == "}":
            return members
        if payload[cursor] != ",":
            return members
        cursor += 1


def _unquoted_marker_start(text: str, marker: str) -> int | None:
    quoted = False
    escaped = False
    cursor = 0
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
        if text.startswith(marker, cursor):
            return cursor
        cursor += 1
    return None


def _safe_partial_json_payload(payload: str) -> str:
    marker_start = _unquoted_marker_start(payload, _CLOSE)
    return payload if marker_start is None else payload[:marker_start]


def _json_name(payload: str) -> str | None:
    safe_payload = _safe_partial_json_payload(payload)
    span = _top_level_json_members(safe_payload).get("name")
    if span is None or span[1] is None:
        return None
    return _decode_json_string(safe_payload[span[0] : span[1]])


def _json_arguments(payload: str) -> str:
    safe_payload = _safe_partial_json_payload(payload)
    span = _top_level_json_members(safe_payload).get("arguments")
    if span is None:
        return ""
    value_start, value_end = span
    if value_start >= len(safe_payload) or safe_payload[value_start] not in "{[":
        return ""
    if value_end is None:
        return json_value_prefix(safe_payload, value_start)
    return safe_payload[value_start:value_end]


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    decoded: dict[str, Any] = {}
    for key, value in pairs:
        if key in decoded:
            raise QwenDialectError(f"duplicate JSON tool field: {key}")
        decoded[key] = value
    return decoded


def _visible_text(text: str) -> str:
    without_orphan_closers = text.replace(_CLOSE, "")
    return hold_partial_marker(without_orphan_closers, (_OPEN, _CLOSE))


def _envelope_end(text: str, payload_start: int) -> tuple[int, int] | None:
    """Find the structural close, ignoring close-marker text in JSON strings."""

    content_start = skip_ascii_space(text, payload_start)
    if content_start < len(text) and text[content_start] in "{[":
        payload_end = json_value_end(text, content_start)
        if payload_end is None:
            return None
        close_start = skip_ascii_space(text, payload_end)
        if text.startswith(_CLOSE, close_start):
            return payload_end, close_start + len(_CLOSE)
        return None

    if text.startswith("<function=", content_start):
        search = content_start
        for _ in range(32):
            function_end = text.find(_FUNCTION_CLOSE, search)
            if function_end < 0:
                return None
            payload_end = function_end + len(_FUNCTION_CLOSE)
            close_start = skip_ascii_space(text, payload_end)
            candidate = text[content_start:function_end]
            function = _FUNCTION_RE.match(candidate)
            if (
                text.startswith(_CLOSE, close_start)
                and function is not None
                and _parameter_body_is_complete(candidate[function.end() :])
            ):
                return payload_end, close_start + len(_CLOSE)
            search = payload_end
        return None

    close_start = text.find(_CLOSE, payload_start)
    if close_start < 0:
        return None
    return close_start, close_start + len(_CLOSE)


def _serialize_arguments(value: Any) -> str:
    if value is None:
        value = {}
    if not isinstance(value, dict | list):
        value = {"value": value}
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _parameter_value_end(body: str, start: int) -> int | None:
    search = start
    while True:
        close = body.find(_PARAMETER_CLOSE, search)
        if close < 0:
            return None
        after = skip_ascii_space(body, close + len(_PARAMETER_CLOSE))
        if after >= len(body) or body.startswith("<parameter=", after):
            return close
        search = close + len(_PARAMETER_CLOSE)


def _parameter_body_is_complete(body: str) -> bool:
    cursor = 0
    while cursor < len(body):
        cursor = skip_ascii_space(body, cursor)
        if cursor >= len(body):
            return True
        parameter = _PARAMETER_OPEN_RE.match(body, cursor)
        if parameter is None:
            return False
        value_end = _parameter_value_end(body, parameter.end())
        if value_end is None:
            return False
        cursor = value_end + len(_PARAMETER_CLOSE)
    return True


def _complete_call(payload: str, index: int) -> DialectToolCall:
    stripped = payload.strip()
    if stripped.startswith("{"):
        try:
            decoded = json.loads(stripped, object_pairs_hook=_unique_object)
        except QwenDialectError:
            raise
        except (TypeError, ValueError) as exc:
            raise QwenDialectError("invalid JSON tool envelope") from exc
        if not isinstance(decoded, dict) or not isinstance(decoded.get("name"), str):
            raise QwenDialectError("JSON tool envelope requires a string name")
        name = decoded["name"].strip()
        if not name:
            raise QwenDialectError("JSON tool envelope requires a non-empty name")
        arguments_json = _json_arguments(stripped)
        if not arguments_json:
            arguments_json = _serialize_arguments(decoded.get("arguments", {}))
        return DialectToolCall(
            index=index,
            call_id=stable_call_id("qwen", index),
            name=name,
            arguments=arguments_json,
            complete=True,
        )

    function = _FUNCTION_RE.match(stripped)
    if function is None or not stripped.endswith(_FUNCTION_CLOSE):
        raise QwenDialectError("unsupported Qwen tool envelope")
    name = function.group(1)
    body = stripped[function.end() : -len(_FUNCTION_CLOSE)]
    parameter_arguments: dict[str, Any] = {}
    cursor = 0
    while cursor < len(body):
        cursor = skip_ascii_space(body, cursor)
        if cursor >= len(body):
            break
        parameter = _PARAMETER_OPEN_RE.match(body, cursor)
        if parameter is None:
            raise QwenDialectError("text outside Qwen parameter elements")
        key = parameter.group(1)
        if key in parameter_arguments:
            raise QwenDialectError(f"duplicate Qwen parameter: {key}")
        value_end = _parameter_value_end(body, parameter.end())
        if value_end is None:
            raise QwenDialectError(f"unterminated Qwen parameter: {key}")
        raw_value = body[parameter.end() : value_end].strip()
        try:
            parameter_arguments[key] = json.loads(raw_value)
        except (TypeError, ValueError):
            parameter_arguments[key] = raw_value
        cursor = value_end + len(_PARAMETER_CLOSE)
    return DialectToolCall(
        index=index,
        call_id=stable_call_id("qwen", index),
        name=name,
        arguments=_serialize_arguments(parameter_arguments),
        complete=True,
    )


def _partial_json_call(payload: str, index: int) -> DialectToolCall | None:
    name = _json_name(payload)
    if name is None:
        return None
    return DialectToolCall(
        index=index,
        call_id=stable_call_id("qwen", index),
        name=name,
        arguments=_json_arguments(payload),
        complete=False,
    )


class QwenToolDialect:
    """Parse cumulative Qwen tool envelopes without owning stream state."""

    def parse(self, text: str, *, final: bool) -> DialectParse:
        visible: list[str] = []
        calls: list[DialectToolCall] = []
        cursor = 0

        while True:
            start = text.find(_OPEN, cursor)
            if start < 0:
                tail = text[cursor:]
                safe_tail = _visible_text(tail)
                if final and safe_tail != tail.replace(_CLOSE, ""):
                    raise QwenDialectError("unterminated Qwen tool marker")
                visible.append(safe_tail)
                break

            visible.append(_visible_text(text[cursor:start]))
            payload_start = start + len(_OPEN)
            envelope_end = _envelope_end(text, payload_start)
            if envelope_end is None:
                if final:
                    raise QwenDialectError("unterminated Qwen tool envelope")
                partial = _partial_json_call(text[payload_start:], len(calls))
                if partial is not None:
                    calls.append(partial)
                break

            payload_end, span_end = envelope_end
            calls.append(_complete_call(text[payload_start:payload_end], len(calls)))
            cursor = span_end

        return DialectParse(visible_text="".join(visible), calls=tuple(calls))
