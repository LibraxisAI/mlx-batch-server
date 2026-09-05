"""One temporal and finality owner for a generation turn.

Adapted from MTPLX
`mtplx/server/core/generation_turn.py@6d0ddf0575faa9acf77e63c57e48ea1602a7e4ab`
(Apache-2.0). Modified by LibraxisAI with a strict Responses lifecycle,
bounded acknowledged producer handoff, and isolated bounded subscribers.
"""

from __future__ import annotations

import asyncio
import threading
import weakref
from collections import deque
from collections.abc import AsyncIterator, Mapping
from concurrent.futures import Future
from enum import StrEnum
from typing import Any, TypeAlias

from .events import (
    HOSTED_CALL_ITEM_KIND,
    REASONING_CONTENT_KIND,
    TERMINAL_EVENT_TYPES,
    TEXT_CONTENT_KIND,
    TURN_EVENT_TYPES,
    ContentPartCompleted,
    ContentPartStarted,
    HostedCallCompleted,
    HostedCallProgress,
    HostedCallResult,
    HostedCallStarted,
    HostedCitation,
    OutputItemCompleted,
    OutputItemStarted,
    ProgressUpdate,
    ReasoningCompleted,
    ReasoningDelta,
    SequencedTurnEvent,
    TerminalEvent,
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


class TurnState(StrEnum):
    IDLE = "idle"
    STARTED = "started"
    TERMINAL = "terminal"


class TurnSubscriberOverflow(RuntimeError):
    """One subscriber exceeded its bound; sibling subscribers remain live."""


class TurnSubscriberLimit(RuntimeError):
    """The turn already has its maximum number of active subscribers."""


class TurnProducerOverflow(RuntimeError):
    """The cross-thread producer bridge has no free acknowledgement slot."""


_QueueItem: TypeAlias = SequencedTurnEvent | BaseException | None

MAX_CITATIONS_PER_ITEM = 64

_HOSTED_TOOL_ACTION_KINDS = {"web_search": "search", "web_fetch": "fetch"}


class _ContentLifecycle:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.text = ""
        self.flow_completed = False
        self.completed = False
        self.citation_end_max = 0


class _ItemLifecycle:
    """Bounded lifecycle bookkeeping: identities, digest and booleans only.

    Fetched content, snippets and result arrays never enter this object; the
    event stream is the sole carrier of hosted payloads (memory-lifetime law).
    """

    def __init__(self, kind: str, item_id: str) -> None:
        self.kind = kind
        self.item_id = item_id
        self.contents: dict[int, _ContentLifecycle] = {}
        self.call_id: str | None = None
        self.tool_name: str | None = None
        self.tool_arguments = ""
        self.tool_completed = False
        self.hosted_started = False
        self.hosted_action: Mapping[str, Any] | None = None
        self.hosted_status: str | None = None
        self.hosted_result_seen = False
        self.hosted_result_digest: str | None = None
        self.hosted_result_identities: tuple[str, ...] = ()
        self.citation_count = 0
        self.completed = False


class GenerationTurn:
    def __init__(
        self,
        *,
        max_pending_events: int = 4096,
        replay_events: int | None = None,
        max_subscribers: int = 64,
        max_thread_bridge_events: int | None = None,
    ) -> None:
        if max_pending_events < 1:
            raise ValueError("max_pending_events must be at least 1")
        if replay_events is not None and replay_events < 1:
            raise ValueError("replay_events must be at least 1")
        if max_subscribers < 1:
            raise ValueError("max_subscribers must be at least 1")
        bridge_limit = (
            max_pending_events
            if max_thread_bridge_events is None
            else int(max_thread_bridge_events)
        )
        if bridge_limit < 1:
            raise ValueError("max_thread_bridge_events must be at least 1")

        self._loop = asyncio.get_running_loop()
        self._owner_thread_id = threading.get_ident()
        self._max_pending_events = int(max_pending_events)
        self._max_subscribers = int(max_subscribers)
        history_limit = (
            self._max_pending_events if replay_events is None else int(replay_events)
        )
        self._replay_events = history_limit
        self._history: deque[SequencedTurnEvent] = deque(maxlen=history_limit)
        self._subscribers: weakref.WeakSet[_Subscriber] = weakref.WeakSet()
        self._state = TurnState.IDLE
        self._sequence_number = 0
        self._started: SequencedTurnEvent | None = None
        self._terminal_event: SequencedTurnEvent | None = None
        self._items: dict[int, _ItemLifecycle] = {}
        self._item_ids: set[str] = set()
        self._usage: UsageUpdate | None = None
        # call_id -> proven URL identities; O(#calls) identity ledger, no content.
        self._hosted_success_identities: dict[str, tuple[str, ...]] = {}

        self._thread_bridge_slots = threading.BoundedSemaphore(bridge_limit)
        self._thread_bridge_lock = threading.Lock()
        self._pending_thread_events = 0

    @property
    def state(self) -> TurnState:
        return self._state

    @property
    def terminal_event(self) -> SequencedTurnEvent | None:
        return self._terminal_event

    @property
    def usage(self) -> UsageUpdate | None:
        return self._usage

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    @property
    def pending_thread_events(self) -> int:
        with self._thread_bridge_lock:
            return self._pending_thread_events

    def start(self, event: TurnStarted) -> None:
        self._dispatch(event)

    def emit(self, event: TurnEvent) -> None:
        self._dispatch(event)

    def terminal(self, event: TerminalEvent) -> None:
        if not isinstance(event, TERMINAL_EVENT_TYPES):
            raise TypeError(f"terminal event required, got {type(event).__name__}")
        self._dispatch(event)

    def complete(self, event: TurnCompleted) -> None:
        if not isinstance(event, TurnCompleted):
            raise TypeError(f"TurnCompleted required, got {type(event).__name__}")
        self._dispatch(event)

    def fail(self, event: TurnFailed) -> None:
        if not isinstance(event, TurnFailed):
            raise TypeError(f"TurnFailed required, got {type(event).__name__}")
        self._dispatch(event)

    def cancel(self, event: TurnCancelled) -> None:
        if not isinstance(event, TurnCancelled):
            raise TypeError(f"TurnCancelled required, got {type(event).__name__}")
        self._dispatch(event)

    def subscribe(
        self,
        *,
        max_pending_events: int | None = None,
    ) -> _Subscriber:
        self._require_owner_loop("subscribe")
        size = (
            self._max_pending_events
            if max_pending_events is None
            else int(max_pending_events)
        )
        if size < 1:
            raise ValueError("max_pending_events must be at least 1")
        if (
            self._state is not TurnState.TERMINAL
            and len(self._subscribers) >= self._max_subscribers
        ):
            raise TurnSubscriberLimit("turn subscriber limit reached")

        subscriber = _Subscriber(self, size)
        for item in self._bounded_replay(size):
            subscriber.put_nowait(item)
        if self._state is TurnState.TERMINAL:
            subscriber.close_nowait()
        else:
            self._subscribers.add(subscriber)
        return subscriber

    def _dispatch(self, event: TurnEvent) -> None:
        if not isinstance(event, TURN_EVENT_TYPES):
            raise TypeError(f"TurnEvent required, got {type(event).__name__}")
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is self._loop:
            self._accept(event)
            return
        if threading.get_ident() == self._owner_thread_id:
            raise RuntimeError("turn mutation requires its running owner event loop")
        if self._loop.is_closed() or not self._loop.is_running():
            raise RuntimeError("turn owner event loop is not running")
        if not self._thread_bridge_slots.acquire(blocking=False):
            raise TurnProducerOverflow("turn producer bridge overflow")

        acknowledgement: Future[None] = Future()
        self._increment_pending_thread_events()

        def accept_on_owner() -> None:
            try:
                self._accept(event)
            except BaseException as error:
                acknowledgement.set_exception(error)
            else:
                acknowledgement.set_result(None)
            finally:
                self._decrement_pending_thread_events()
                self._thread_bridge_slots.release()

        try:
            self._loop.call_soon_threadsafe(accept_on_owner)
        except BaseException:
            self._decrement_pending_thread_events()
            self._thread_bridge_slots.release()
            raise
        acknowledgement.result()

    def _increment_pending_thread_events(self) -> None:
        with self._thread_bridge_lock:
            self._pending_thread_events += 1

    def _decrement_pending_thread_events(self) -> None:
        with self._thread_bridge_lock:
            self._pending_thread_events -= 1

    def _require_owner_loop(self, operation: str) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not self._loop:
            raise RuntimeError(f"{operation} must run on the owning event loop")

    def _accept(self, event: TurnEvent) -> None:
        if isinstance(event, TurnStarted):
            if self._state is not TurnState.IDLE:
                raise RuntimeError(
                    f"TurnStarted requires idle state, got {self._state.value}"
                )
            self._state = TurnState.STARTED
        elif isinstance(event, TERMINAL_EVENT_TYPES):
            if self._state is TurnState.TERMINAL:
                raise RuntimeError("terminal event already emitted")
            if self._state is TurnState.IDLE and not isinstance(event, TurnFailed):
                raise RuntimeError("only TurnFailed may terminate an idle turn")
            if isinstance(event, TurnCompleted):
                self._validate_completion(event)
            self._state = TurnState.TERMINAL
        else:
            if self._state is not TurnState.STARTED:
                raise RuntimeError(
                    f"turn event requires started state, got {self._state.value}"
                )
            self._apply_intermediate(event)

        sequenced = SequencedTurnEvent(self._sequence_number, event)
        self._sequence_number += 1
        self._history.append(sequenced)
        if isinstance(event, TurnStarted):
            self._started = sequenced
        if isinstance(event, TERMINAL_EVENT_TYPES):
            self._terminal_event = sequenced

        overflowed: list[_Subscriber] = []
        for subscriber in tuple(self._subscribers):
            if not subscriber.put_nowait(sequenced):
                overflowed.append(subscriber)
        for subscriber in overflowed:
            self._subscribers.discard(subscriber)
            subscriber.fail_nowait(
                TurnSubscriberOverflow("turn event subscriber overflow")
            )

        if self._state is TurnState.TERMINAL:
            for subscriber in tuple(self._subscribers):
                subscriber.close_nowait()
            self._subscribers.clear()

    def _apply_intermediate(self, event: TurnEvent) -> None:
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
        elif isinstance(event, HostedCallStarted):
            self._start_hosted_call(event)
        elif isinstance(event, HostedCallProgress):
            self._progress_hosted_call(event)
        elif isinstance(event, HostedCallResult):
            self._result_hosted_call(event)
        elif isinstance(event, HostedCitation):
            self._cite(event)
        elif isinstance(event, HostedCallCompleted):
            self._complete_hosted_call(event)
        elif isinstance(event, UsageUpdate):
            self._accept_usage(event)
        elif isinstance(event, ProgressUpdate):
            return
        else:  # pragma: no cover - closed-union guard
            raise TypeError(f"unsupported intermediate event {type(event).__name__}")

    def _start_item(self, event: OutputItemStarted) -> None:
        if event.index != len(self._items):
            raise RuntimeError(
                f"output item index must be contiguous; expected {len(self._items)}"
            )
        if event.item_id in self._item_ids:
            raise RuntimeError(f"duplicate output item id {event.item_id!r}")
        item = _ItemLifecycle(event.kind, event.item_id)
        if event.kind == HOSTED_CALL_ITEM_KIND:
            item.call_id = event.call_id
            item.tool_name = event.name
        self._items[event.index] = item
        self._item_ids.add(event.item_id)

    def _complete_item(self, event: OutputItemCompleted) -> None:
        item = self._require_open_item(event.index, event.item_id)
        if item.kind != event.kind:
            raise RuntimeError("output item kind changed before completion")
        if any(not content.completed for content in item.contents.values()):
            raise RuntimeError("output item has an open content part")
        if item.kind == "function_call" and not item.tool_completed:
            raise RuntimeError("function-call output item has no done event")
        if item.kind == HOSTED_CALL_ITEM_KIND:
            if item.hosted_status is None:
                raise RuntimeError("hosted_call output item has no receipt event")
            if event.call_id != item.call_id or event.name != item.tool_name:
                raise RuntimeError(
                    "hosted_call output item completion does not match its call"
                )
            if event.status != item.hosted_status:
                raise RuntimeError(
                    "hosted_call output item status must match its receipt"
                )
            self._verify_hosted_action(item, event)
            item.completed = True
            return
        if item.kind in {"message", "reasoning"}:
            completed_text = "".join(content.text for content in item.contents.values())
            if event.text != completed_text:
                raise RuntimeError(
                    "output item completion text does not match its content"
                )
        elif (
            event.call_id != item.call_id
            or event.name != item.tool_name
            or event.arguments != item.tool_arguments
        ):
            raise RuntimeError(
                "function-call output item completion does not match its done event"
            )
        item.completed = True

    def _start_content(self, event: ContentPartStarted) -> None:
        item = self._require_open_item(event.output_index, event.item_id)
        expected_item_kind = (
            "message" if event.kind == TEXT_CONTENT_KIND else "reasoning"
        )
        if item.kind != expected_item_kind:
            raise RuntimeError(
                f"{event.kind} content requires a {expected_item_kind} output item"
            )
        if event.content_index != len(item.contents):
            raise RuntimeError(
                f"content part index must be contiguous; expected {len(item.contents)}"
            )
        item.contents[event.content_index] = _ContentLifecycle(event.kind)

    def _complete_content(self, event: ContentPartCompleted) -> None:
        content = self._require_open_content(
            event.output_index,
            event.item_id,
            event.content_index,
            event.kind,
        )
        if not content.flow_completed:
            raise RuntimeError("content part requires its text done event first")
        if event.text != content.text:
            raise RuntimeError(
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
            raise RuntimeError("content delta cannot follow its done event")
        content.text += delta

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
            raise RuntimeError("content text done event already emitted")
        if text != content.text:
            raise RuntimeError("content done text does not match emitted deltas")
        if content.citation_end_max > len(text):
            raise RuntimeError("hosted citation output range exceeds its final text")
        content.flow_completed = True

    def _append_tool(self, event: ToolDelta) -> None:
        item = self._require_open_item(event.index, event.item_id)
        if item.kind != "function_call":
            raise RuntimeError("tool arguments require a function_call output item")
        if item.tool_completed:
            raise RuntimeError("tool arguments delta cannot follow its done event")
        if item.call_id is None:
            if event.name is None:
                raise RuntimeError("first tool delta must include the tool name")
            item.call_id = event.call_id
            item.tool_name = event.name
        elif event.call_id != item.call_id:
            raise RuntimeError("tool call id changed during argument streaming")
        elif event.name is not None and event.name != item.tool_name:
            raise RuntimeError("tool name changed during argument streaming")
        item.tool_arguments += event.arguments_delta

    def _complete_tool(self, event: ToolCompleted) -> None:
        item = self._require_open_item(event.index, event.item_id)
        if item.kind != "function_call":
            raise RuntimeError("tool done requires a function_call output item")
        if item.call_id is None:
            raise RuntimeError("tool done requires a preceding tool delta")
        if item.tool_completed:
            raise RuntimeError("tool done event already emitted")
        if event.call_id != item.call_id or event.name != item.tool_name:
            raise RuntimeError("tool done identity does not match its deltas")
        if event.arguments != item.tool_arguments:
            raise RuntimeError("tool done arguments do not match emitted deltas")
        item.tool_completed = True

    def _require_hosted_item(
        self,
        index: int,
        item_id: str,
        call_id: str,
    ) -> _ItemLifecycle:
        item = self._require_open_item(index, item_id)
        if item.kind != HOSTED_CALL_ITEM_KIND:
            raise RuntimeError("hosted call events require a hosted_call output item")
        if call_id != item.call_id:
            raise RuntimeError("hosted call id does not match its output item")
        return item

    def _start_hosted_call(self, event: HostedCallStarted) -> None:
        item = self._require_hosted_item(event.index, event.item_id, event.call_id)
        if event.tool_name != item.tool_name:
            raise RuntimeError("hosted call tool name does not match its output item")
        if item.hosted_started:
            raise RuntimeError("hosted call already started")
        item.hosted_started = True
        item.hosted_action = event.action

    def _verify_hosted_action(
        self,
        item: _ItemLifecycle,
        event: OutputItemCompleted,
    ) -> None:
        action = event.action
        if action is None:  # pragma: no cover - the event layer already refuses
            raise RuntimeError("hosted_call output item completion carries no action")
        kind = action["kind"]
        expected_kind = _HOSTED_TOOL_ACTION_KINDS.get(item.tool_name or "")
        if expected_kind is not None and kind != expected_kind:
            raise RuntimeError("hosted completion action kind does not match its tool")
        started: Mapping[str, Any] = item.hosted_action or {}
        if kind == "search":
            started_query = started.get("query")
            if started_query is not None and action["query"] != started_query:
                raise RuntimeError(
                    "hosted completion action contradicts its started query"
                )
            sources = action["sources"]
            if sources and not set(sources) <= set(item.hosted_result_identities):
                raise RuntimeError(
                    "hosted completion sources are not proven by its result"
                )
            return
        started_url = started.get("url")
        if started_url is not None and action["url"] != started_url:
            raise RuntimeError("hosted completion action contradicts its started url")

    def _result_hosted_call(self, event: HostedCallResult) -> None:
        item = self._require_hosted_item(event.index, event.item_id, event.call_id)
        if event.tool_name != item.tool_name:
            raise RuntimeError("hosted result tool name does not match its output item")
        if not item.hosted_started:
            raise RuntimeError("hosted result requires a started hosted call")
        if item.hosted_status is not None:
            raise RuntimeError("hosted result cannot follow its receipt")
        if item.hosted_result_seen:
            raise RuntimeError("hosted result already emitted")
        item.hosted_result_seen = True
        item.hosted_result_digest = event.result["digest"]
        item.hosted_result_identities = event.identities
        self._hosted_success_identities[event.call_id] = event.identities

    def _cite(self, event: HostedCitation) -> None:
        item = self._require_open_item(event.output_index, event.item_id)
        if item.kind != "message":
            raise RuntimeError("hosted citation requires a message output item")
        content = item.contents.get(event.content_index)
        if content is None:
            raise RuntimeError(
                f"content part {event.content_index} has not been started"
            )
        if content.kind != TEXT_CONTENT_KIND:
            raise RuntimeError("hosted citation requires an output_text content part")
        if content.completed:
            raise RuntimeError("hosted citation cannot follow its content completion")
        identities = self._hosted_success_identities.get(event.source_call_id)
        if identities is None:
            raise RuntimeError("hosted citation references an unknown hosted result")
        if event.source_url not in identities:
            raise RuntimeError("hosted citation url is not proven by its result")
        if item.citation_count >= MAX_CITATIONS_PER_ITEM:
            raise RuntimeError("hosted citation count exceeds its item bound")
        if content.flow_completed:
            if event.output_end > len(content.text):
                raise RuntimeError(
                    "hosted citation output range exceeds its final text"
                )
        else:
            content.citation_end_max = max(content.citation_end_max, event.output_end)
        item.citation_count += 1

    def _progress_hosted_call(self, event: HostedCallProgress) -> None:
        item = self._require_hosted_item(event.index, event.item_id, event.call_id)
        if not item.hosted_started:
            raise RuntimeError("hosted call progress requires a started hosted call")
        if item.hosted_status is not None:
            raise RuntimeError("hosted call progress cannot follow its receipt")

    def _complete_hosted_call(self, event: HostedCallCompleted) -> None:
        item = self._require_hosted_item(event.index, event.item_id, event.call_id)
        if event.tool_name != item.tool_name:
            raise RuntimeError("hosted call tool name does not match its output item")
        if not item.hosted_started:
            raise RuntimeError("hosted call receipt requires a started hosted call")
        if item.hosted_status is not None:
            raise RuntimeError("hosted call receipt already emitted")
        receipt_call_id = event.receipt.get("call_id")
        if receipt_call_id is not None and receipt_call_id != event.call_id:
            raise RuntimeError("hosted call receipt does not match its call id")
        if event.status == "completed" and not item.hosted_result_seen:
            raise RuntimeError(
                "hosted call completed receipt requires its result event"
            )
        if event.status == "failed" and item.hosted_result_seen:
            raise RuntimeError(
                "hosted call failed receipt cannot follow a result event"
            )
        receipt_digest = event.receipt.get("result_digest")
        if (
            item.hosted_result_seen
            and receipt_digest is not None
            and receipt_digest != item.hosted_result_digest
        ):
            raise RuntimeError("hosted call receipt digest does not match its result")
        item.hosted_status = event.status

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
            raise RuntimeError("usage updates must be cumulative and monotonic")
        self._usage = event

    def _validate_completion(self, event: TurnCompleted) -> None:
        if self._state is not TurnState.STARTED:
            return
        if any(not item.completed for item in self._items.values()):
            raise RuntimeError("TurnCompleted requires every output item to be done")
        if self._usage is not None and event.usage != self._usage:
            raise RuntimeError("TurnCompleted usage must equal the latest UsageUpdate")
        if self._usage is None and event.usage is not None:
            self._usage = event.usage

    def _require_open_item(self, index: int, item_id: str) -> _ItemLifecycle:
        item = self._items.get(index)
        if item is None:
            raise RuntimeError(f"output item {index} has not been started")
        if item.item_id != item_id:
            raise RuntimeError("output item id does not match its index")
        if item.completed:
            raise RuntimeError("output item is already done")
        return item

    def _require_open_content(
        self,
        output_index: int,
        item_id: str,
        content_index: int,
        expected_kind: str,
    ) -> _ContentLifecycle:
        item = self._require_open_item(output_index, item_id)
        content = item.contents.get(content_index)
        if content is None:
            raise RuntimeError(f"content part {content_index} has not been started")
        if content.kind != expected_kind:
            raise RuntimeError("content event kind does not match its content part")
        if content.completed:
            raise RuntimeError("content part is already done")
        return content

    def _bounded_replay(self, max_pending_events: int) -> list[SequencedTurnEvent]:
        capacity = min(max_pending_events, self._replay_events)
        history = list(self._history)
        middle = [
            item
            for item in history
            if item is not self._started and item is not self._terminal_event
        ]
        replay: list[SequencedTurnEvent] = []
        if self._started is not None and capacity > 1:
            replay.append(self._started)
        terminal_slots = 1 if self._terminal_event is not None else 0
        middle_capacity = max(0, capacity - len(replay) - terminal_slots)
        if middle_capacity:
            replay.extend(middle[-middle_capacity:])
        if self._terminal_event is not None:
            replay.append(self._terminal_event)
        return replay[-capacity:]

    def _unsubscribe(self, subscriber: _Subscriber) -> None:
        self._subscribers.discard(subscriber)


class _Subscriber:
    def __init__(self, owner: GenerationTurn, maxsize: int) -> None:
        self._owner = owner
        self._event_capacity = maxsize
        self._queue: asyncio.Queue[_QueueItem] = asyncio.Queue(maxsize=maxsize + 1)
        self._accepting = True
        self._closed = False

    def __aiter__(self) -> AsyncIterator[SequencedTurnEvent]:
        return self

    async def __anext__(self) -> SequencedTurnEvent:
        if self._closed:
            raise StopAsyncIteration
        try:
            item = await self._queue.get()
        except asyncio.CancelledError:
            self._accepting = False
            self._closed = True
            self._owner._unsubscribe(self)
            raise
        if item is None:
            self._closed = True
            self._owner._unsubscribe(self)
            raise StopAsyncIteration
        if isinstance(item, BaseException):
            self._closed = True
            self._owner._unsubscribe(self)
            raise item
        return item

    async def aclose(self) -> None:
        if self._closed:
            return
        self._accepting = False
        self._closed = True
        self._owner._unsubscribe(self)

    def put_nowait(self, item: SequencedTurnEvent) -> bool:
        if not self._accepting:
            return True
        if self._queue.qsize() >= self._event_capacity:
            return False
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            return False
        return True

    def close_nowait(self) -> None:
        self._put_control_nowait(None)

    def fail_nowait(self, error: BaseException) -> None:
        self._put_control_nowait(error)

    def _put_control_nowait(self, item: BaseException | None) -> None:
        if not self._accepting:
            return
        self._accepting = False
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull as error:  # pragma: no cover - invariant guard
            raise RuntimeError("subscriber control slot was not reserved") from error
