"""Canonical incremental tool-call parsing with pluggable model dialects.

The dialect recognizes model-specific syntax and returns a cumulative snapshot.
This module owns incremental emission, stable call identity, and final
normalization so protocol adapters do not each grow their own parser state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class ParsedToolDelta:
    index: int
    call_id: str
    name: str | None = None
    arguments_delta: str = ""


@dataclass(frozen=True, slots=True)
class ParsedToolCall:
    index: int
    call_id: str
    name: str
    arguments: str


@runtime_checkable
class ToolParser(Protocol):
    def feed(self, text: str) -> tuple[str, tuple[ParsedToolDelta, ...]]: ...

    def finish(self) -> tuple[str, tuple[ParsedToolCall, ...]]: ...


class ToolParseError(ValueError):
    """A dialect produced a non-monotonic or otherwise ambiguous snapshot."""


@dataclass(frozen=True, slots=True)
class DialectToolCall:
    """One cumulative tool-call snapshot emitted by a model dialect."""

    index: int
    call_id: str
    name: str | None = None
    arguments: str = ""
    complete: bool = False

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("tool call index must be non-negative")
        if not self.call_id.strip():
            raise ValueError("tool call_id must not be empty")
        if self.name is not None and not self.name.strip():
            raise ValueError("tool name must not be empty")


@dataclass(frozen=True, slots=True)
class DialectParse:
    """Cumulative parser view for all source text received so far."""

    visible_text: str = ""
    calls: tuple[DialectToolCall, ...] = ()


@runtime_checkable
class ToolDialect(Protocol):
    """Model-specific syntax recognizer without incremental ownership."""

    def parse(self, text: str, *, final: bool) -> DialectParse: ...


@dataclass(slots=True)
class _CallState:
    index: int
    call_id: str
    name: str | None
    arguments: str
    complete: bool


@dataclass(frozen=True, slots=True)
class _ApplyResult:
    visible_delta: str
    deltas: tuple[ParsedToolDelta, ...]
    visible_text: str
    calls: dict[str, _CallState]
    call_ids_by_index: dict[int, str]


class IncrementalToolParser:
    """Normalize cumulative dialect snapshots into exactly-once deltas/calls."""

    def __init__(self, dialect: ToolDialect) -> None:
        self._dialect = dialect
        self._source = ""
        self._visible_text = ""
        self._calls: dict[str, _CallState] = {}
        self._call_ids_by_index: dict[int, str] = {}
        self._finished = False
        self._final_result: tuple[str, tuple[ParsedToolCall, ...]] | None = None

    def feed(self, text: str) -> tuple[str, tuple[ParsedToolDelta, ...]]:
        if self._finished:
            raise RuntimeError("cannot feed a finished tool parser")
        if not isinstance(text, str):
            raise TypeError("tool parser input must be text")
        next_source = self._source + text
        result = self._apply(self._dialect.parse(next_source, final=False))
        self._commit(result)
        self._source = next_source
        return result.visible_delta, result.deltas

    def finish(self) -> tuple[str, tuple[ParsedToolCall, ...]]:
        if self._final_result is not None:
            return self._final_result

        result = self._apply(self._dialect.parse(self._source, final=True))
        unfinished = [
            state.call_id
            for state in result.calls.values()
            if not state.complete or state.name is None
        ]
        if unfinished:
            joined = ", ".join(sorted(unfinished))
            raise ToolParseError(f"unfinished tool calls at end of stream: {joined}")

        calls = tuple(
            ParsedToolCall(
                index=state.index,
                call_id=state.call_id,
                name=state.name,
                arguments=state.arguments,
            )
            for state in sorted(result.calls.values(), key=lambda item: item.index)
            if state.name is not None
        )
        self._commit(result)
        self._finished = True
        self._final_result = (result.visible_delta, calls)
        return self._final_result

    def _apply(
        self,
        snapshot: DialectParse,
    ) -> _ApplyResult:
        if not snapshot.visible_text.startswith(self._visible_text):
            raise ToolParseError("dialect visible text must grow monotonically")

        visible_delta = snapshot.visible_text[len(self._visible_text) :]
        next_calls = {
            call_id: _CallState(
                index=state.index,
                call_id=state.call_id,
                name=state.name,
                arguments=state.arguments,
                complete=state.complete,
            )
            for call_id, state in self._calls.items()
        }
        next_call_ids_by_index = dict(self._call_ids_by_index)
        current_ids: set[str] = set()
        current_indices: set[int] = set()
        deltas: list[ParsedToolDelta] = []

        for call in sorted(snapshot.calls, key=lambda item: item.index):
            if call.call_id in current_ids:
                raise ToolParseError(f"duplicate call_id in snapshot: {call.call_id}")
            if call.index in current_indices:
                raise ToolParseError(f"duplicate tool index in snapshot: {call.index}")
            current_ids.add(call.call_id)
            current_indices.add(call.index)

            indexed_call_id = next_call_ids_by_index.get(call.index)
            if indexed_call_id is not None and indexed_call_id != call.call_id:
                raise ToolParseError(
                    f"tool index {call.index} changed call_id from "
                    f"{indexed_call_id} to {call.call_id}"
                )

            state = next_calls.get(call.call_id)
            if state is None:
                state = _CallState(
                    index=call.index,
                    call_id=call.call_id,
                    name=None,
                    arguments="",
                    complete=False,
                )
                next_calls[call.call_id] = state
                next_call_ids_by_index[call.index] = call.call_id
            elif state.index != call.index:
                raise ToolParseError(
                    f"call_id {call.call_id} changed index from "
                    f"{state.index} to {call.index}"
                )

            if state.complete and (
                call.name != state.name
                or call.arguments != state.arguments
                or not call.complete
            ):
                raise ToolParseError(
                    f"completed tool call changed after completion: {call.call_id}"
                )
            if state.name is not None and call.name != state.name:
                raise ToolParseError(f"tool name changed for call_id {call.call_id}")
            if not call.arguments.startswith(state.arguments):
                raise ToolParseError(
                    f"tool arguments rewound for call_id {call.call_id}"
                )
            name_delta = call.name if state.name is None else None
            arguments_delta = call.arguments[len(state.arguments) :]
            if name_delta is not None or arguments_delta:
                deltas.append(
                    ParsedToolDelta(
                        index=call.index,
                        call_id=call.call_id,
                        name=name_delta,
                        arguments_delta=arguments_delta,
                    )
                )

            state.name = call.name
            state.arguments = call.arguments
            state.complete = call.complete

        missing = set(next_calls).difference(current_ids)
        if missing:
            joined = ", ".join(sorted(missing))
            raise ToolParseError(f"dialect dropped previously observed calls: {joined}")

        return _ApplyResult(
            visible_delta=visible_delta,
            deltas=tuple(deltas),
            visible_text=snapshot.visible_text,
            calls=next_calls,
            call_ids_by_index=next_call_ids_by_index,
        )

    def _commit(self, result: _ApplyResult) -> None:
        self._visible_text = result.visible_text
        self._calls = result.calls
        self._call_ids_by_index = result.call_ids_by_index
