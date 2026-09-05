"""Accumulate canonical runtime events into one terminal Responses envelope."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..runtime.events import (
    REASONING_CONTENT_KIND,
    TEXT_CONTENT_KIND,
    ContentPartCompleted,
    ContentPartStarted,
    OutputItemCompleted,
    OutputItemStarted,
    ProgressUpdate,
    ReasoningCompleted,
    ReasoningDelta,
    SequencedTurnEvent,
    TextCompleted,
    TextDelta,
    ToolCompleted,
    ToolDelta,
    TurnCancelled,
    TurnCompleted,
    TurnEvent,
    TurnFailed,
    TurnStarted,
    UsageUpdate,
)
from .controller import PreparedResponse
from .transport import (
    ResponseSnapshotIdentity,
    build_response_snapshot,
    request_settings_from,
)


class RuntimeProjectionError(ValueError):
    """A runtime event cannot be reconciled with this response lifecycle."""


class _FrozenDict(dict[str, Any]):
    @staticmethod
    def _immutable(*_args: Any, **_kwargs: Any) -> None:
        raise TypeError("terminal response envelope is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __ior__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable


class _FrozenList(list[Any]):
    @staticmethod
    def _immutable(*_args: Any, **_kwargs: Any) -> None:
        raise TypeError("terminal response envelope is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __iadd__ = _immutable
    __imul__ = _immutable
    append = _immutable
    clear = _immutable
    extend = _immutable
    insert = _immutable
    pop = _immutable
    remove = _immutable
    reverse = _immutable
    sort = _immutable


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _FrozenDict(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list | tuple):
        return _FrozenList(_deep_freeze(item) for item in value)
    return value


@dataclass(slots=True)
class _ContentState:
    kind: str
    chunks: list[str] = field(default_factory=list)
    done_text: str | None = None
    flow_completed: bool = False
    completed: bool = False


@dataclass(slots=True)
class _ItemState:
    kind: str
    item_id: str
    contents: dict[int, _ContentState] = field(default_factory=dict)
    completed: bool = False
    call_id: str | None = None
    name: str | None = None
    argument_chunks: list[str] = field(default_factory=list)
    done_arguments: str | None = None
    tool_completed: bool = False
    completion_status: str | None = None


class RuntimeResponseProjection:
    """Concrete controller projection for one canonical runtime response."""

    def __init__(
        self,
        prepared: PreparedResponse,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not isinstance(prepared, PreparedResponse):
            raise TypeError("runtime projection requires PreparedResponse")
        if not callable(clock):
            raise TypeError("clock must be callable")
        request = prepared.request
        response_id = request.response_id
        model = request.runtime.model_id
        if not isinstance(response_id, str) or not response_id.strip():
            raise ValueError("prepared response_id must not be blank")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("prepared model must not be blank")
        requested_model = request.metadata.get("requested_model")
        if requested_model is not None and (
            not isinstance(requested_model, str) or not requested_model.strip()
        ):
            raise ValueError("prepared requested_model must not be blank")

        self._response_id = response_id
        # Physical runtime identity: every runtime event keeps validating
        # against this and it never reaches the wire.
        self._model = model
        # Public protocol identity: the alias the client requested, and the
        # only model value this projection publishes.
        self._public_model = requested_model if requested_model is not None else model
        self._request_settings = request_settings_from(
            tools=request.tools,
            sampling=request.sampling,
            reasoning=request.reasoning,
            metadata=request.metadata,
        )
        self._created_at = int(clock())
        self._last_sequence_number: int | None = None
        self._started: TurnStarted | None = None
        self._items: dict[int, _ItemState] = {}
        self._item_ids: set[str] = set()
        self._usage: UsageUpdate | None = None
        self._terminal: Mapping[str, Any] | None = None

    def observe(self, sequenced: SequencedTurnEvent) -> None:
        if not isinstance(sequenced, SequencedTurnEvent):
            raise TypeError("projection requires SequencedTurnEvent")
        if self._terminal is not None:
            raise RuntimeProjectionError("event cannot follow terminal response")
        previous = self._last_sequence_number
        if previous is not None and sequenced.sequence_number <= previous:
            raise RuntimeProjectionError(
                "response event sequence numbers must increase monotonically"
            )

        self._apply(sequenced.event)
        self._last_sequence_number = sequenced.sequence_number

    def terminal_envelope(self) -> Mapping[str, Any]:
        if self._terminal is None:
            raise RuntimeProjectionError(
                "terminal response envelope requested before terminal event"
            )
        return self._terminal

    def _apply(self, event: TurnEvent) -> None:
        if isinstance(event, TurnStarted):
            self._start(event)
            return
        if isinstance(event, TurnFailed):
            self._finish(event)
            return
        if self._started is None:
            raise RuntimeProjectionError(f"{type(event).__name__} requires TurnStarted")
        if isinstance(event, OutputItemStarted):
            self._start_item(event)
        elif isinstance(event, OutputItemCompleted):
            self._complete_item(event)
        elif isinstance(event, ContentPartStarted):
            self._start_content(event)
        elif isinstance(event, ContentPartCompleted):
            self._complete_content(event)
        elif isinstance(event, TextDelta):
            self._append_content(event, TEXT_CONTENT_KIND, event.delta)
        elif isinstance(event, TextCompleted):
            self._complete_content_flow(event, TEXT_CONTENT_KIND, event.text)
        elif isinstance(event, ReasoningDelta):
            self._append_content(event, REASONING_CONTENT_KIND, event.delta)
        elif isinstance(event, ReasoningCompleted):
            self._complete_content_flow(
                event,
                REASONING_CONTENT_KIND,
                event.text,
            )
        elif isinstance(event, ToolDelta):
            self._append_tool(event)
        elif isinstance(event, ToolCompleted):
            self._complete_tool(event)
        elif isinstance(event, UsageUpdate):
            self._accept_usage(event)
        elif isinstance(event, ProgressUpdate):
            return
        elif isinstance(event, TurnCompleted | TurnCancelled):
            self._finish(event)
        else:
            raise TypeError(f"unsupported runtime event {type(event).__name__}")

    def _start(self, event: TurnStarted) -> None:
        if self._started is not None:
            raise RuntimeProjectionError("response has already started")
        if event.response_id != self._response_id:
            raise RuntimeProjectionError(
                "TurnStarted response_id does not match prepared response"
            )
        if event.model != self._model:
            raise RuntimeProjectionError(
                "TurnStarted model does not match prepared response"
            )
        if (
            event.requested_model is not None
            and event.requested_model != self._public_model
        ):
            raise RuntimeProjectionError(
                "TurnStarted requested_model does not match prepared response"
            )
        self._started = event
        self._created_at = event.created_at

    def _start_item(self, event: OutputItemStarted) -> None:
        if event.index != len(self._items):
            raise RuntimeProjectionError(
                f"output item index must be contiguous; expected {len(self._items)}"
            )
        if event.item_id in self._item_ids:
            raise RuntimeProjectionError(f"duplicate output item id {event.item_id!r}")
        # A function call is admitted with its identity already fixed; nothing
        # downstream may introduce, change or drop it later.
        self._items[event.index] = _ItemState(
            event.kind,
            event.item_id,
            call_id=event.call_id,
            name=event.name,
        )
        self._item_ids.add(event.item_id)

    def _complete_item(self, event: OutputItemCompleted) -> None:
        item = self._require_open_item(event.index, event.item_id)
        if item.kind != event.kind:
            raise RuntimeProjectionError("output item kind changed before completion")
        if any(not content.completed for content in item.contents.values()):
            raise RuntimeProjectionError("output item has an open content part")
        if item.kind == "function_call" and not item.tool_completed:
            raise RuntimeProjectionError("function-call output item has no done event")
        if item.kind in {"message", "reasoning"}:
            completed_text = "".join(
                content.done_text or "" for content in item.contents.values()
            )
            if event.text != completed_text:
                raise RuntimeProjectionError(
                    "output item completion text does not match its content"
                )
        elif (
            event.call_id != item.call_id
            or event.name != item.name
            or event.arguments != item.done_arguments
        ):
            raise RuntimeProjectionError(
                "function-call output item completion does not match its done event"
            )
        item.completed = True
        item.completion_status = event.status

    def _start_content(self, event: ContentPartStarted) -> None:
        item = self._require_open_item(event.output_index, event.item_id)
        expected_item_kind = (
            "message" if event.kind == TEXT_CONTENT_KIND else "reasoning"
        )
        if item.kind != expected_item_kind:
            raise RuntimeProjectionError(
                f"{event.kind} content requires a {expected_item_kind} item"
            )
        if event.content_index != len(item.contents):
            raise RuntimeProjectionError(
                f"content part index must be contiguous; expected {len(item.contents)}"
            )
        item.contents[event.content_index] = _ContentState(event.kind)

    def _complete_content(self, event: ContentPartCompleted) -> None:
        content = self._require_open_content(
            event.output_index,
            event.item_id,
            event.content_index,
            event.kind,
        )
        if not content.flow_completed:
            raise RuntimeProjectionError(
                "content part requires its text done event first"
            )
        if event.text != content.done_text:
            raise RuntimeProjectionError(
                "content part completion text does not match its done event"
            )
        content.completed = True

    def _append_content(
        self,
        event: TextDelta | ReasoningDelta,
        expected_kind: str,
        delta: str,
    ) -> None:
        content = self._require_open_content(
            event.output_index,
            event.item_id,
            event.content_index,
            expected_kind,
        )
        if content.flow_completed:
            raise RuntimeProjectionError("content delta cannot follow done event")
        content.chunks.append(delta)

    def _complete_content_flow(
        self,
        event: TextCompleted | ReasoningCompleted,
        expected_kind: str,
        text: str,
    ) -> None:
        content = self._require_open_content(
            event.output_index,
            event.item_id,
            event.content_index,
            expected_kind,
        )
        if content.flow_completed:
            raise RuntimeProjectionError("content text done event already emitted")
        if text != "".join(content.chunks):
            raise RuntimeProjectionError(
                "content done text does not match emitted deltas"
            )
        content.done_text = text
        content.flow_completed = True

    def _append_tool(self, event: ToolDelta) -> None:
        item = self._require_open_item(event.index, event.item_id)
        if item.kind != "function_call":
            raise RuntimeProjectionError(
                "tool arguments require a function_call output item"
            )
        if item.tool_completed:
            raise RuntimeProjectionError(
                "tool arguments delta cannot follow its done event"
            )
        if item.call_id is None:
            raise RuntimeProjectionError(
                "function_call output item started without its tool identity"
            )
        if event.call_id != item.call_id:
            raise RuntimeProjectionError(
                "tool call id changed during argument streaming"
            )
        elif event.name is not None and event.name != item.name:
            raise RuntimeProjectionError("tool name changed during argument streaming")
        item.argument_chunks.append(event.arguments_delta)

    def _complete_tool(self, event: ToolCompleted) -> None:
        item = self._require_open_item(event.index, event.item_id)
        if item.kind != "function_call":
            raise RuntimeProjectionError(
                "tool done requires a function_call output item"
            )
        if item.call_id is None or item.name is None:
            raise RuntimeProjectionError(
                "tool done requires an output item started with its tool identity"
            )
        if item.tool_completed:
            raise RuntimeProjectionError("tool done event already emitted")
        if event.call_id != item.call_id or event.name != item.name:
            raise RuntimeProjectionError("tool done identity does not match its deltas")
        if event.arguments != "".join(item.argument_chunks):
            raise RuntimeProjectionError(
                "tool done arguments do not match emitted deltas"
            )
        item.done_arguments = event.arguments
        item.tool_completed = True

    def _accept_usage(self, event: UsageUpdate) -> None:
        previous = self._usage
        if previous is not None and (
            event.input_tokens < previous.input_tokens
            or event.output_tokens < previous.output_tokens
            or event.total_tokens < previous.total_tokens
            or event.cached_input_tokens < previous.cached_input_tokens
            or (event.cache_write_input_tokens < previous.cache_write_input_tokens)
            or event.reasoning_output_tokens < previous.reasoning_output_tokens
        ):
            raise RuntimeProjectionError(
                "usage updates must be cumulative and monotonic"
            )
        self._usage = event

    def _finish(
        self,
        event: TurnCompleted | TurnFailed | TurnCancelled,
    ) -> None:
        if isinstance(event, TurnCompleted):
            if any(not item.completed for item in self._items.values()):
                raise RuntimeProjectionError(
                    "TurnCompleted requires every output item to be done"
                )
            if self._usage is not None and event.usage != self._usage:
                raise RuntimeProjectionError(
                    "TurnCompleted usage must equal the latest UsageUpdate"
                )
            if self._usage is None:
                self._usage = event.usage
            if event.finish_reason == "length":
                status = "incomplete"
                incomplete_details = {"reason": "max_output_tokens"}
            else:
                status = "completed"
                incomplete_details = None
            error = None
        elif isinstance(event, TurnFailed):
            status = "failed"
            error = {"code": event.code, "message": event.error}
            incomplete_details = None
        else:
            status = "incomplete" if event.reason == "steered" else "cancelled"
            error = None
            incomplete_details = (
                {"reason": "steered"} if event.reason == "steered" else None
            )

        response = build_response_snapshot(
            identity=ResponseSnapshotIdentity(
                response_id=self._response_id,
                created_at=self._created_at,
                public_model=self._public_model,
                physical_model=self._model,
            ),
            status=status,
            request_settings=self._request_settings,
            output=[
                self._project_item(self._items[index])
                for index in range(len(self._items))
            ],
            usage=self._project_usage(),
            error=error,
            incomplete_details=incomplete_details,
        )
        self._terminal = _deep_freeze(response)

    def _project_item(self, item: _ItemState) -> dict[str, Any]:
        status = item.completion_status or "incomplete"
        if item.kind == "message":
            return {
                "id": item.item_id,
                "type": "message",
                "status": status,
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": self._content_text(item.contents[index]),
                        "annotations": [],
                    }
                    for index in range(len(item.contents))
                ],
            }
        if item.kind == "reasoning":
            return {
                "id": item.item_id,
                "type": "reasoning",
                "status": status,
                "summary": [
                    {
                        "type": "summary_text",
                        "text": self._content_text(item.contents[index]),
                    }
                    for index in range(len(item.contents))
                ],
            }

        if item.call_id is None or item.name is None:  # pragma: no cover - guarded
            raise RuntimeProjectionError(
                "function_call output item has no admitted tool identity"
            )
        return {
            "id": item.item_id,
            "type": "function_call",
            "status": status,
            "call_id": item.call_id,
            "name": item.name,
            "arguments": (
                item.done_arguments
                if item.done_arguments is not None
                else "".join(item.argument_chunks)
            ),
        }

    def _project_usage(self) -> dict[str, Any] | None:
        if self._usage is None:
            return None
        return {
            "input_tokens": self._usage.input_tokens,
            "input_tokens_details": {
                "cache_write_tokens": self._usage.cache_write_input_tokens,
                "cached_tokens": self._usage.cached_input_tokens,
            },
            "output_tokens": self._usage.output_tokens,
            "output_tokens_details": {
                "reasoning_tokens": self._usage.reasoning_output_tokens,
            },
            "total_tokens": self._usage.total_tokens,
        }

    @staticmethod
    def _content_text(content: _ContentState) -> str:
        if content.done_text is not None:
            return content.done_text
        return "".join(content.chunks)

    def _require_open_item(self, index: int, item_id: str) -> _ItemState:
        item = self._items.get(index)
        if item is None:
            raise RuntimeProjectionError(f"output item {index} has not been started")
        if item.item_id != item_id:
            raise RuntimeProjectionError("output item id does not match its index")
        if item.completed:
            raise RuntimeProjectionError("output item is already done")
        return item

    def _require_open_content(
        self,
        output_index: int,
        item_id: str,
        content_index: int,
        expected_kind: str,
    ) -> _ContentState:
        item = self._require_open_item(output_index, item_id)
        content = item.contents.get(content_index)
        if content is None:
            raise RuntimeProjectionError(
                f"content part {content_index} has not been started"
            )
        if content.kind != expected_kind:
            raise RuntimeProjectionError(
                "content event kind does not match its content part"
            )
        if content.completed:
            raise RuntimeProjectionError("content part is already done")
        return content


def create_runtime_projection(
    prepared: PreparedResponse,
) -> RuntimeResponseProjection:
    """ProjectionFactory entry point for CanonicalResponsesMapper."""

    return RuntimeResponseProjection(prepared)


__all__ = (
    "RuntimeProjectionError",
    "RuntimeResponseProjection",
    "create_runtime_projection",
)
