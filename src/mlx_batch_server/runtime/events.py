"""Closed, protocol-neutral event family for one generation turn.

Adapted from MTPLX
`mtplx/server/core/events.py@6d0ddf0575faa9acf77e63c57e48ea1602a7e4ab`
(Apache-2.0). Modified by LibraxisAI for the complete Responses output-item
and content-part lifecycle, immutable payloads, target token names, and
transport-neutral sequencing.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, TypeAlias

TEXT_CONTENT_KIND = "output_text"
REASONING_CONTENT_KIND = "reasoning_summary_text"
CONTENT_KINDS = frozenset((TEXT_CONTENT_KIND, REASONING_CONTENT_KIND))
OUTPUT_ITEM_KINDS = frozenset(("message", "reasoning", "function_call"))


class _FrozenDict(dict[str, Any]):
    """A JSON-serializable mapping that rejects consumer mutation."""

    @staticmethod
    def _immutable(*_args: Any, **_kwargs: Any) -> None:
        raise TypeError("event mappings are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable


def _require_identity(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must not be empty")


def _require_text(name: str, value: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")


def _require_index(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("event mapping keys must be strings")
            frozen[key] = _deep_freeze(item)
        return _FrozenDict(frozen)
    if isinstance(value, list | tuple):
        return tuple(_deep_freeze(item) for item in value)
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise TypeError(
        f"event payload value {type(value).__name__} is not JSON-compatible"
    )


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    frozen = _deep_freeze(value)
    if not isinstance(frozen, Mapping):  # pragma: no cover - type guard
        raise TypeError("event payload must be a mapping")
    return frozen


@dataclass(frozen=True, slots=True)
class TurnStarted:
    response_id: str
    model: str
    created_at: int

    def __post_init__(self) -> None:
        _require_identity("response_id", self.response_id)
        _require_identity("model", self.model)
        _require_index("created_at", self.created_at)


@dataclass(frozen=True, slots=True)
class OutputItemStarted:
    kind: str
    index: int
    item_id: str

    def __post_init__(self) -> None:
        if self.kind not in OUTPUT_ITEM_KINDS:
            raise ValueError(f"unsupported output item kind {self.kind!r}")
        _require_index("index", self.index)
        _require_identity("item_id", self.item_id)


@dataclass(frozen=True, slots=True)
class OutputItemCompleted:
    kind: str
    index: int
    item_id: str
    text: str | None = None
    call_id: str | None = None
    name: str | None = None
    arguments: str | None = None
    status: str = "completed"

    def __post_init__(self) -> None:
        if self.kind not in OUTPUT_ITEM_KINDS:
            raise ValueError(f"unsupported output item kind {self.kind!r}")
        _require_index("index", self.index)
        _require_identity("item_id", self.item_id)
        if self.status not in {"completed", "incomplete"}:
            raise ValueError("output item status must be completed or incomplete")
        if self.kind in {"message", "reasoning"}:
            if self.text is None:
                raise ValueError(f"{self.kind} completion requires final text")
            _require_text("text", self.text)
            if any(
                value is not None for value in (self.call_id, self.name, self.arguments)
            ):
                raise ValueError(
                    f"{self.kind} completion cannot carry function-call fields"
                )
            return
        if self.text is not None:
            raise ValueError("function_call completion cannot carry text")
        if self.call_id is None or self.name is None or self.arguments is None:
            raise ValueError(
                "function_call completion requires call_id, name, and arguments"
            )
        _require_identity("call_id", self.call_id)
        _require_identity("name", self.name)
        _require_text("arguments", self.arguments)


@dataclass(frozen=True, slots=True)
class ContentPartStarted:
    kind: str
    output_index: int
    content_index: int
    item_id: str

    def __post_init__(self) -> None:
        if self.kind not in CONTENT_KINDS:
            raise ValueError(f"unsupported content part kind {self.kind!r}")
        _require_index("output_index", self.output_index)
        _require_index("content_index", self.content_index)
        _require_identity("item_id", self.item_id)


@dataclass(frozen=True, slots=True)
class ContentPartCompleted:
    kind: str
    output_index: int
    content_index: int
    item_id: str
    text: str

    def __post_init__(self) -> None:
        if self.kind not in CONTENT_KINDS:
            raise ValueError(f"unsupported content part kind {self.kind!r}")
        _require_index("output_index", self.output_index)
        _require_index("content_index", self.content_index)
        _require_identity("item_id", self.item_id)
        _require_text("text", self.text)


@dataclass(frozen=True, slots=True)
class ReasoningDelta:
    delta: str
    item_id: str
    output_index: int
    content_index: int

    def __post_init__(self) -> None:
        _require_text("delta", self.delta)
        _require_identity("item_id", self.item_id)
        _require_index("output_index", self.output_index)
        _require_index("content_index", self.content_index)


@dataclass(frozen=True, slots=True)
class ReasoningCompleted:
    text: str
    item_id: str
    output_index: int
    content_index: int

    def __post_init__(self) -> None:
        _require_text("text", self.text)
        _require_identity("item_id", self.item_id)
        _require_index("output_index", self.output_index)
        _require_index("content_index", self.content_index)


@dataclass(frozen=True, slots=True)
class TextDelta:
    delta: str
    item_id: str
    output_index: int
    content_index: int

    def __post_init__(self) -> None:
        _require_text("delta", self.delta)
        _require_identity("item_id", self.item_id)
        _require_index("output_index", self.output_index)
        _require_index("content_index", self.content_index)


@dataclass(frozen=True, slots=True)
class TextCompleted:
    text: str
    item_id: str
    output_index: int
    content_index: int

    def __post_init__(self) -> None:
        _require_text("text", self.text)
        _require_identity("item_id", self.item_id)
        _require_index("output_index", self.output_index)
        _require_index("content_index", self.content_index)


@dataclass(frozen=True, slots=True)
class ToolDelta:
    index: int
    call_id: str
    item_id: str
    name: str | None = None
    arguments_delta: str = ""

    def __post_init__(self) -> None:
        _require_index("index", self.index)
        _require_identity("call_id", self.call_id)
        _require_identity("item_id", self.item_id)
        if self.name is not None:
            _require_identity("name", self.name)
        _require_text("arguments_delta", self.arguments_delta)


@dataclass(frozen=True, slots=True)
class ToolCompleted:
    index: int
    call_id: str
    item_id: str
    name: str
    arguments: str

    def __post_init__(self) -> None:
        _require_index("index", self.index)
        _require_identity("call_id", self.call_id)
        _require_identity("item_id", self.item_id)
        _require_identity("name", self.name)
        _require_text("arguments", self.arguments)


@dataclass(frozen=True, slots=True)
class UsageUpdate:
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_input_tokens: int = 0
    cache_write_input_tokens: int = 0
    reasoning_output_tokens: int = 0

    def __post_init__(self) -> None:
        for field_name in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cached_input_tokens",
            "cache_write_input_tokens",
            "reasoning_output_tokens",
        ):
            _require_index(field_name, getattr(self, field_name))
        if self.total_tokens != self.input_tokens + self.output_tokens:
            raise ValueError("total_tokens must equal input_tokens + output_tokens")
        if self.cached_input_tokens > self.input_tokens:
            raise ValueError("cached_input_tokens cannot exceed input_tokens")
        if self.cache_write_input_tokens > self.input_tokens:
            raise ValueError("cache_write_input_tokens cannot exceed input_tokens")
        if self.reasoning_output_tokens > self.output_tokens:
            raise ValueError("reasoning_output_tokens cannot exceed output_tokens")


@dataclass(frozen=True, slots=True)
class ProgressUpdate:
    phase: str
    detail: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_identity("phase", self.phase)
        object.__setattr__(self, "detail", _freeze_mapping(self.detail))


@dataclass(frozen=True, slots=True)
class TurnCompleted:
    finish_reason: str
    usage: UsageUpdate | None = None
    backend_stats: Mapping[str, Any] = field(default_factory=dict)
    stop_sequence: str | None = None

    def __post_init__(self) -> None:
        _require_identity("finish_reason", self.finish_reason)
        if self.usage is not None and not isinstance(self.usage, UsageUpdate):
            raise TypeError("usage must be a UsageUpdate")
        if self.finish_reason == "stop_sequence":
            if not isinstance(self.stop_sequence, str) or not self.stop_sequence:
                raise ValueError("stop_sequence must not be empty")
        elif self.stop_sequence is not None:
            raise ValueError("non-stop completion cannot carry stop_sequence")
        object.__setattr__(self, "backend_stats", _freeze_mapping(self.backend_stats))


@dataclass(frozen=True, slots=True)
class TurnFailed:
    error: str
    code: str = "internal_error"
    status_code: int = 500

    def __post_init__(self) -> None:
        _require_identity("error", self.error)
        _require_identity("code", self.code)
        if (
            isinstance(self.status_code, bool)
            or not isinstance(self.status_code, int)
            or not 400 <= self.status_code <= 599
        ):
            raise ValueError("status_code must be an HTTP error status")


@dataclass(frozen=True, slots=True)
class TurnCancelled:
    reason: str

    def __post_init__(self) -> None:
        _require_identity("reason", self.reason)


TerminalEvent: TypeAlias = TurnCompleted | TurnFailed | TurnCancelled
TurnEvent: TypeAlias = (
    TurnStarted
    | OutputItemStarted
    | OutputItemCompleted
    | ContentPartStarted
    | ContentPartCompleted
    | ReasoningDelta
    | ReasoningCompleted
    | TextDelta
    | TextCompleted
    | ToolDelta
    | ToolCompleted
    | UsageUpdate
    | ProgressUpdate
    | TerminalEvent
)

TERMINAL_EVENT_TYPES = (TurnCompleted, TurnFailed, TurnCancelled)
TURN_EVENT_TYPES = (
    TurnStarted,
    OutputItemStarted,
    OutputItemCompleted,
    ContentPartStarted,
    ContentPartCompleted,
    ReasoningDelta,
    ReasoningCompleted,
    TextDelta,
    TextCompleted,
    ToolDelta,
    ToolCompleted,
    UsageUpdate,
    ProgressUpdate,
    *TERMINAL_EVENT_TYPES,
)


@dataclass(frozen=True, slots=True)
class SequencedTurnEvent:
    sequence_number: int
    event: TurnEvent

    def __post_init__(self) -> None:
        _require_index("sequence_number", self.sequence_number)
