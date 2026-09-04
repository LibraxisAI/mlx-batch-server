"""Concrete legacy port over the target-owned wrapper cache and stream adapter.

Imports of the legacy MLX stack are deliberately deferred until ``acquire`` or
the first request. Importing this module is therefore tensor-free. The provider
never constructs a cache and never unloads a borrowed runtime.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..contracts import (
    BackendKind,
    CancelToken,
    GenerationRequest,
    LoadConfig,
    ModelSpec,
    RuntimeKey,
)
from ..events import (
    REASONING_CONTENT_KIND,
    TEXT_CONTENT_KIND,
    ContentPartCompleted,
    ContentPartStarted,
    OutputItemCompleted,
    OutputItemStarted,
    ReasoningCompleted,
    ReasoningDelta,
    TextCompleted,
    TextDelta,
    ToolCompleted,
    ToolDelta,
    TurnEvent,
    UsageUpdate,
)
from .legacy_mlx import LegacyCapability, LegacyPortContractError

__all__ = ["CachedLegacyPortProvider"]


class _WrapperCache(Protocol):
    """The small part of ``MLXWrapperCache`` this adapter is allowed to borrow."""

    def get_wrapper(
        self,
        model_id: str,
        adapter_path: str | None = None,
        draft_model_id: str | None = None,
        surface: str | None = None,
    ) -> Any: ...

    def is_runtime_loaded(
        self,
        model_id: str,
        adapter_path: str | None = None,
        draft_model_id: str | None = None,
    ) -> bool: ...

    def get_cache_info(self) -> Mapping[str, Any]: ...


class _ResponseStreamFactory(Protocol):
    def __call__(
        self,
        cache: _WrapperCache,
        payload: Mapping[str, Any],
    ) -> AsyncIterator[Mapping[str, Any]]: ...


class LegacyProviderExecutionError(RuntimeError):
    """Raised when the borrowed legacy stream cannot satisfy the port contract."""


def _global_wrapper_cache() -> _WrapperCache:
    # Runtime-only import: this module remains safe to inspect under the embargo.
    from ...chat.mlx.wrapper_cache import wrapper_cache

    return wrapper_cache


def _responses_stream(
    cache: _WrapperCache,
    payload: Mapping[str, Any],
) -> AsyncIterator[Mapping[str, Any]]:
    # The old Responses adapter is authoritative only when it uses this exact
    # process-global cache. Refuse an injected second residency owner.
    from ...chat.mlx.wrapper_cache import wrapper_cache
    from ...responses.adapter import ResponsesAdapter
    from ...responses.schema import ResponseRequest

    if cache is not wrapper_cache:
        raise LegacyProviderExecutionError(
            "default legacy stream requires the process-global MLXWrapperCache"
        )
    request = ResponseRequest.model_validate(dict(payload))
    return ResponsesAdapter(model_id=request.model).generate_stream(request)


class CachedLegacyPortProvider:
    """Bind ``LegacyMlxBackend`` to the existing cache and streaming surface.

    A cache may be injected for tests or composition, but it is always borrowed.
    The default is the existing process-global ``wrapper_cache``. The provider
    binds revision-bearing keys only through an existing immutable snapshot
    directory because the cache itself has no revision dimension.
    """

    def __init__(
        self,
        *,
        cache: _WrapperCache | None = None,
        stream_factory: _ResponseStreamFactory | None = None,
    ) -> None:
        self._cache = cache
        self._stream_factory = stream_factory or _responses_stream

    def probe(self, model: ModelSpec) -> LegacyCapability:
        disabled = model.metadata.get("legacy_streaming") is False
        revision_unkeyed = model.revision is not None and not _model_spec_is_bound(
            model
        )
        vision = bool(
            model.metadata.get("supports_multimodal")
            or model.metadata.get("vision")
            or _looks_multimodal(model)
        )
        reasons: list[str] = []
        if disabled:
            reasons.append("legacy model explicitly disables streaming generation")
        if revision_unkeyed:
            reasons.append("legacy cache cannot key or verify model revisions")
        return LegacyCapability(
            supported=(
                bool(model.model_id.strip()) and not disabled and not revision_unkeyed
            ),
            text=True,
            vision=vision,
            tools=True,
            continuous_batching=False,
            cache_modes=("wrapper_lru", "runtime_lease"),
            rejection_reasons=tuple(reasons),
            facts={
                "residency_owner": "MLXWrapperCache",
                "generation_surface": "ResponsesAdapter.generate_stream",
                "cooperative_cancel": True,
                "sequential_generation": False,
                "revision_identity": "snapshot_directory_required",
            },
        )

    async def acquire(
        self,
        runtime: RuntimeKey,
        config: LoadConfig,
    ) -> _CachedLegacyExecutionPort:
        if runtime.backend is not BackendKind.LEGACY_MLX:
            raise LegacyPortContractError(
                "cached legacy provider requires backend='legacy_mlx'"
            )
        _reject_sequential_options(config.options)
        cancel_timeout_s = _positive_float(
            config.options.get("legacy_cancel_timeout_s", 10.0),
            "legacy_cancel_timeout_s",
        )
        event_queue_size = _positive_int(
            config.options.get("legacy_event_queue_size", 256),
            "legacy_event_queue_size",
        )
        cache = self._cache or _global_wrapper_cache()
        model_ref = _verified_model_ref(runtime, config.options)

        # ``llm`` is the existing shared product attachment. It keeps max_size=0
        # deployments from creating one uncached wrapper here and another in the
        # Responses adapter. This provider does not own or release that surface.
        wrapper = cache.get_wrapper(
            model_ref,
            adapter_path=runtime.adapter_path,
            draft_model_id=runtime.draft_model_id,
            surface="llm",
        )
        if not cache.is_runtime_loaded(
            model_ref,
            adapter_path=runtime.adapter_path,
            draft_model_id=runtime.draft_model_id,
        ):
            raise LegacyPortContractError(
                "legacy cache returned an untracked wrapper; refusing duplicate residency"
            )

        supports_vision = bool(
            getattr(getattr(wrapper, "model", None), "supports_multimodal", False)
        )
        return _CachedLegacyExecutionPort(
            runtime=runtime,
            model_ref=model_ref,
            cache=cache,
            stream_factory=self._stream_factory,
            supports_vision=supports_vision,
            max_active=config.max_admitted_requests,
            cancel_timeout_s=cancel_timeout_s,
            event_queue_size=event_queue_size,
        )


class _CachedLegacyExecutionPort:
    def __init__(
        self,
        *,
        runtime: RuntimeKey,
        model_ref: str,
        cache: _WrapperCache,
        stream_factory: _ResponseStreamFactory,
        supports_vision: bool,
        max_active: int,
        cancel_timeout_s: float,
        event_queue_size: int,
    ) -> None:
        if max_active < 1:
            raise LegacyPortContractError("max_admitted_requests must be positive")
        self._runtime = runtime
        self._model_ref = model_ref
        self._cache = cache
        self._stream_factory = stream_factory
        self._supports_vision = supports_vision
        self._max_active = max_active
        self._cancel_timeout_s = cancel_timeout_s
        self._event_queue_size = event_queue_size
        self._execution_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._workers: dict[str, _TurnWorker] = {}
        self._closed = False

    @property
    def runtime_key(self) -> RuntimeKey:
        return self._runtime

    async def events(
        self,
        request: GenerationRequest,
        cancel: CancelToken,
    ) -> AsyncIterator[TurnEvent]:
        del cancel  # LegacyMlxBackend owns cancellation sequencing and reason.
        if request.runtime != self._runtime:
            raise LegacyPortContractError(
                "request RuntimeKey does not match legacy port"
            )
        if request.media and not self._supports_vision:
            raise LegacyPortContractError(
                "resident legacy runtime does not expose a multimodal stream"
            )

        payload = _responses_payload(request, model_ref=self._model_ref)
        worker = _TurnWorker(
            request.response_id,
            cache=self._cache,
            payload=payload,
            stream_factory=self._stream_factory,
            execution_lock=self._execution_lock,
            event_queue_size=self._event_queue_size,
        )
        with self._state_lock:
            if self._closed:
                raise LegacyPortContractError("legacy execution port is closed")
            if request.response_id in self._workers:
                raise LegacyPortContractError("legacy response is already active")
            if len(self._workers) >= self._max_active:
                raise LegacyPortContractError(
                    "legacy port admission capacity exhausted"
                )
            self._workers[request.response_id] = worker
        worker.start()

        try:
            while True:
                try:
                    item = await asyncio.to_thread(worker.output.get, True, 0.1)
                except queue.Empty:
                    if worker.done.is_set():
                        break
                    continue
                if isinstance(item, _WorkerFailure):
                    raise item.error
                yield item
        finally:
            if not worker.done.is_set():
                await asyncio.to_thread(
                    worker.cancel,
                    "legacy_event_consumer_closed",
                    self._cancel_timeout_s,
                )
            with self._state_lock:
                if self._workers.get(request.response_id) is worker:
                    self._workers.pop(request.response_id, None)

    def cancel(self, response_id: str, reason: str) -> bool:
        with self._state_lock:
            worker = self._workers.get(response_id)
        if worker is None:
            return False
        return worker.cancel(reason, self._cancel_timeout_s)

    def stats(self) -> Mapping[str, Any]:
        with self._state_lock:
            active = len(self._workers)
            closed = self._closed
        cache_info = self._cache.get_cache_info()
        return {
            "residency_owner": "MLXWrapperCache",
            "runtime_key": {
                "model_id": self._runtime.model_id,
                "model_ref": self._model_ref,
                "adapter_path": self._runtime.adapter_path,
                "draft_model_id": self._runtime.draft_model_id,
            },
            "active_executions": active,
            "single_flight": True,
            "event_queue_size": self._event_queue_size,
            "cooperative_cancel": True,
            "sequential_generation": False,
            "closed": closed,
            "cache_size": cache_info.get("cache_size"),
        }

    async def close(self, deadline_s: float) -> None:
        if deadline_s < 0:
            raise ValueError("deadline_s must be non-negative")
        with self._state_lock:
            if self._closed and not self._workers:
                return
            self._closed = True
            workers = tuple(self._workers.values())
        deadline_at = time.monotonic() + deadline_s
        for worker in workers:
            if worker.done.is_set():
                continue
            remaining = max(0.0, deadline_at - time.monotonic())
            accepted = await asyncio.to_thread(
                worker.cancel,
                "legacy_port_closed",
                remaining,
            )
            if not accepted:
                raise LegacyPortContractError(
                    "legacy stream did not stop and clean up before close deadline"
                )


@dataclass(frozen=True, slots=True)
class _WorkerFailure:
    error: Exception


_RESPONSES_STREAM_SAMPLING = frozenset(
    {
        "max_output_tokens",
        "temperature",
        "top_p",
        "stop",
        "tool_choice",
        "response_format",
        "text",
    }
)


class _TurnWorker:
    def __init__(
        self,
        response_id: str,
        *,
        cache: _WrapperCache,
        payload: Mapping[str, Any],
        stream_factory: _ResponseStreamFactory,
        execution_lock: threading.Lock,
        event_queue_size: int,
    ) -> None:
        self.response_id = response_id
        self.output: queue.Queue[TurnEvent | _WorkerFailure] = queue.Queue(
            maxsize=event_queue_size
        )
        self.done = threading.Event()
        self._cache = cache
        self._payload = payload
        self._stream_factory = stream_factory
        self._execution_lock = execution_lock
        self._cancel_requested = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task[None] | None = None
        self._cleanup_error: Exception | None = None
        self._completed_naturally = False
        self._thread = threading.Thread(
            target=self._thread_main,
            name=f"legacy-port:{response_id}",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def cancel(self, reason: str, timeout_s: float) -> bool:
        if not reason.strip() or timeout_s < 0:
            return False
        if self.done.is_set():
            return (
                self._cancel_requested.is_set()
                and not self._completed_naturally
                and self._cleanup_error is None
            )
        self._cancel_requested.set()
        loop = self._loop
        task = self._task
        if loop is not None and task is not None:
            loop.call_soon_threadsafe(task.cancel)
        if not self.done.wait(timeout_s):
            return False
        return not self._completed_naturally and self._cleanup_error is None

    def _thread_main(self) -> None:
        acquired = False
        try:
            while not self._cancel_requested.is_set():
                acquired = self._execution_lock.acquire(timeout=0.05)
                if acquired:
                    break
            if not acquired:
                return
            asyncio.run(self._consume())
        except Exception as exc:
            self._cleanup_error = exc
            self._put(_WorkerFailure(exc))
        finally:
            if acquired:
                self._execution_lock.release()
            self.done.set()

    async def _consume(self) -> None:
        if self._cancel_requested.is_set():
            return
        self._loop = asyncio.get_running_loop()
        self._task = asyncio.current_task()
        translator = _ResponsesEventTranslator(self.response_id)
        stream: AsyncIterator[Mapping[str, Any]] | None = None
        try:
            stream = self._stream_factory(self._cache, self._payload)
            async for raw in stream:
                if self._cancel_requested.is_set():
                    raise asyncio.CancelledError
                for event in translator.feed(raw):
                    if not self._put(event):
                        raise asyncio.CancelledError
                await asyncio.sleep(0)
            translator.finish()
            self._completed_naturally = True
        except asyncio.CancelledError:
            if not self._cancel_requested.is_set():
                raise
        finally:
            if stream is not None:
                close = getattr(stream, "aclose", None)
                if callable(close):
                    try:
                        await close()
                    except Exception as exc:
                        self._cleanup_error = exc
                        raise

    def _put(self, item: TurnEvent | _WorkerFailure) -> bool:
        while not self._cancel_requested.is_set():
            try:
                self.output.put(item, timeout=0.05)
            except queue.Full:
                continue
            return True
        return False


class _ResponsesEventTranslator:
    """Translate the existing Responses SSE lifecycle without owning terminal EOF."""

    _IGNORED = frozenset({"response.created", "response.in_progress"})

    def __init__(self, response_id: str) -> None:
        self._response_id = response_id
        self._items: dict[int, tuple[str, str]] = {}
        self._content_started: set[tuple[int, int]] = set()
        self._content_completed: set[tuple[int, int]] = set()
        self._text_completed: set[tuple[int, int]] = set()
        self._completed_text: dict[tuple[int, int], str] = {}
        self._item_completed: set[int] = set()
        self._tool_state: dict[int, dict[str, str]] = {}
        self._tool_completed: set[int] = set()
        self._saw_completed = False

    def feed(self, raw: Mapping[str, Any]) -> tuple[TurnEvent, ...]:
        if not isinstance(raw, Mapping):
            raise LegacyProviderExecutionError("legacy stream event must be a mapping")
        kind = raw.get("type")
        if self._saw_completed:
            raise LegacyProviderExecutionError("event arrived after response.completed")
        if kind in self._IGNORED:
            return ()
        if kind == "error":
            error = raw.get("error")
            message = error.get("message") if isinstance(error, Mapping) else error
            raise LegacyProviderExecutionError(str(message or "legacy stream failed"))
        if kind == "response.output_item.added":
            return self._item_started(raw)
        if kind == "response.content_part.added":
            return (self._content_start(raw),)
        if kind == "response.reasoning_summary_text.delta":
            return self._reasoning_delta(raw)
        if kind == "response.reasoning_summary_text.done":
            return self._reasoning_done(raw)
        if kind == "response.output_text.delta":
            return (self._text_delta(raw),)
        if kind == "response.output_text.done":
            return (self._text_done(raw),)
        if kind == "response.function_call_arguments.delta":
            return (self._tool_delta(raw),)
        if kind == "response.function_call_arguments.done":
            return (self._tool_done(raw),)
        if kind == "response.content_part.done":
            return (self._content_done(raw),)
        if kind == "response.output_item.done":
            return self._item_done(raw)
        if kind == "response.completed":
            response = _mapping(raw, "response")
            if response.get("status") != "completed":
                raise LegacyProviderExecutionError(
                    "response.completed did not carry completed status"
                )
            self._saw_completed = True
            usage = _usage_from(raw)
            return (usage,) if usage is not None else ()
        raise LegacyProviderExecutionError(f"unsupported legacy stream event {kind!r}")

    def finish(self) -> None:
        if not self._saw_completed:
            raise LegacyProviderExecutionError(
                "legacy Responses stream ended without response.completed"
            )
        open_items = set(self._items) - self._item_completed
        if open_items:
            raise LegacyProviderExecutionError(
                f"legacy Responses stream left output items open: {sorted(open_items)}"
            )
        open_content = self._content_started - self._content_completed
        if open_content:
            raise LegacyProviderExecutionError(
                f"legacy Responses stream left content parts open: {sorted(open_content)}"
            )

    def _item_started(self, raw: Mapping[str, Any]) -> tuple[TurnEvent, ...]:
        index = _index(raw, "output_index")
        item = _mapping(raw, "item")
        item_kind = _required_text(item, "type")
        item_id = _required_text(item, "id")
        if index in self._items:
            raise LegacyProviderExecutionError("duplicate output item start")
        self._items[index] = (item_kind, item_id)
        if item_kind == "function_call":
            self._tool_state[index] = {
                "call_id": _required_text(item, "call_id"),
                "item_id": item_id,
                "name": _required_text(item, "name"),
                "arguments": str(item.get("arguments") or ""),
            }
        return (OutputItemStarted(kind=item_kind, index=index, item_id=item_id),)

    def _content_start(self, raw: Mapping[str, Any]) -> ContentPartStarted:
        output_index = _index(raw, "output_index")
        content_index = _index(raw, "content_index")
        item_kind, item_id = self._item(output_index)
        part_kind = _required_text(_mapping(raw, "part"), "type")
        expected = (
            TEXT_CONTENT_KIND if item_kind == "message" else REASONING_CONTENT_KIND
        )
        if part_kind != expected:
            raise LegacyProviderExecutionError(
                "content kind does not match output item"
            )
        key = (output_index, content_index)
        if key in self._content_started:
            raise LegacyProviderExecutionError("duplicate content part start")
        self._content_started.add(key)
        return ContentPartStarted(part_kind, output_index, content_index, item_id)

    def _reasoning_delta(self, raw: Mapping[str, Any]) -> tuple[TurnEvent, ...]:
        output_index = _index(raw, "output_index")
        _, item_id = self._item(output_index, expected="reasoning")
        start = self._ensure_reasoning_start(output_index, item_id)
        delta = ReasoningDelta(str(raw.get("delta") or ""), item_id, output_index, 0)
        return (*start, delta)

    def _reasoning_done(self, raw: Mapping[str, Any]) -> tuple[TurnEvent, ...]:
        output_index = _index(raw, "output_index")
        _, item_id = self._item(output_index, expected="reasoning")
        if (output_index, 0) in self._content_completed:
            raise LegacyProviderExecutionError("duplicate reasoning completion")
        start = self._ensure_reasoning_start(output_index, item_id)
        completed = ReasoningCompleted(
            str(raw.get("text") or ""), item_id, output_index, 0
        )
        self._completed_text[(output_index, 0)] = completed.text
        end = self._complete_content(
            output_index,
            0,
            REASONING_CONTENT_KIND,
            item_id,
            completed.text,
        )
        return (*start, completed, *end)

    def _text_delta(self, raw: Mapping[str, Any]) -> TextDelta:
        output_index = _index(raw, "output_index")
        content_index = _index(raw, "content_index")
        _, item_id = self._item(output_index, expected="message")
        self._require_content(output_index, content_index)
        return TextDelta(
            str(raw.get("delta") or ""), item_id, output_index, content_index
        )

    def _text_done(self, raw: Mapping[str, Any]) -> TextCompleted:
        output_index = _index(raw, "output_index")
        content_index = _index(raw, "content_index")
        _, item_id = self._item(output_index, expected="message")
        self._require_content(output_index, content_index)
        key = (output_index, content_index)
        if key in self._text_completed:
            raise LegacyProviderExecutionError("duplicate text completion")
        self._text_completed.add(key)
        completed = TextCompleted(
            str(raw.get("text") or ""), item_id, output_index, content_index
        )
        self._completed_text[key] = completed.text
        return completed

    def _tool_delta(self, raw: Mapping[str, Any]) -> ToolDelta:
        index = _index(raw, "output_index")
        state = self._tool(index)
        if index in self._tool_completed:
            raise LegacyProviderExecutionError("tool delta arrived after completion")
        _validate_optional_identity(raw, "item_id", state["item_id"])
        _validate_optional_identity(raw, "call_id", state["call_id"])
        delta = str(raw.get("delta") or "")
        state["arguments"] += delta
        return ToolDelta(
            index, state["call_id"], state["item_id"], state["name"], delta
        )

    def _tool_done(self, raw: Mapping[str, Any]) -> ToolCompleted:
        index = _index(raw, "output_index")
        state = self._tool(index)
        if index in self._tool_completed:
            raise LegacyProviderExecutionError("duplicate tool completion")
        _validate_optional_identity(raw, "item_id", state["item_id"])
        _validate_optional_identity(raw, "call_id", state["call_id"])
        _validate_optional_identity(raw, "name", state["name"])
        arguments = str(raw.get("arguments", state["arguments"]))
        state["arguments"] = arguments
        self._tool_completed.add(index)
        return ToolCompleted(
            index, state["call_id"], state["item_id"], state["name"], arguments
        )

    def _content_done(self, raw: Mapping[str, Any]) -> ContentPartCompleted:
        output_index = _index(raw, "output_index")
        content_index = _index(raw, "content_index")
        item_kind, item_id = self._item(output_index)
        expected = (
            TEXT_CONTENT_KIND if item_kind == "message" else REASONING_CONTENT_KIND
        )
        part = _mapping(raw, "part")
        part_kind = _required_text(part, "type")
        if part_kind == "summary_text":
            part_kind = REASONING_CONTENT_KIND
        if part_kind != expected:
            raise LegacyProviderExecutionError("completed content kind mismatch")
        self._require_content(output_index, content_index)
        if (output_index, content_index) in self._content_completed:
            raise LegacyProviderExecutionError("duplicate content part completion")
        text = str(part.get("text") or "")
        completed_text = self._completed_text.get((output_index, content_index))
        if completed_text is not None and text != completed_text:
            raise LegacyProviderExecutionError(
                "completed content text does not match its text done event"
            )
        self._content_completed.add((output_index, content_index))
        return ContentPartCompleted(
            part_kind, output_index, content_index, item_id, text
        )

    def _item_done(self, raw: Mapping[str, Any]) -> tuple[TurnEvent, ...]:
        index = _index(raw, "output_index")
        item_kind, item_id = self._item(index)
        completed_item = _mapping(raw, "item")
        if _required_text(completed_item, "id") != item_id:
            raise LegacyProviderExecutionError("completed output item id mismatch")
        if _required_text(completed_item, "type") != item_kind:
            raise LegacyProviderExecutionError("completed output item kind mismatch")
        if index in self._item_completed:
            raise LegacyProviderExecutionError("duplicate output item completion")
        synthesized: list[TurnEvent] = []
        if item_kind == "reasoning" and (index, 0) not in self._content_completed:
            summary = completed_item.get("summary")
            text = ""
            if (
                isinstance(summary, Sequence)
                and summary
                and isinstance(summary[0], Mapping)
            ):
                text = str(summary[0].get("text") or "")
            synthesized.extend(
                self._reasoning_done({"output_index": index, "text": text})
            )
        if item_kind == "message":
            message_content = {key for key in self._content_started if key[0] == index}
            open_content = message_content - self._content_completed
            if open_content:
                raise LegacyProviderExecutionError(
                    "message output item completed before its content part"
                )
            if message_content - self._text_completed:
                raise LegacyProviderExecutionError(
                    "message output item completed before output_text.done"
                )
        if item_kind == "function_call" and index not in self._tool_completed:
            raise LegacyProviderExecutionError(
                "function_call output item completed before its arguments"
            )
        self._item_completed.add(index)
        if item_kind == "function_call":
            state = self._tool(index)
            completed = OutputItemCompleted(
                item_kind,
                index,
                item_id,
                call_id=state["call_id"],
                name=state["name"],
                arguments=state["arguments"],
            )
        else:
            text = "".join(
                self._completed_text[key]
                for key in sorted(self._completed_text)
                if key[0] == index
            )
            completed = OutputItemCompleted(item_kind, index, item_id, text=text)
        synthesized.append(completed)
        return tuple(synthesized)

    def _ensure_reasoning_start(
        self, index: int, item_id: str
    ) -> tuple[TurnEvent, ...]:
        key = (index, 0)
        if key in self._content_started:
            return ()
        self._content_started.add(key)
        return (ContentPartStarted(REASONING_CONTENT_KIND, index, 0, item_id),)

    def _complete_content(
        self,
        index: int,
        content_index: int,
        kind: str,
        item_id: str,
        text: str,
    ) -> tuple[TurnEvent, ...]:
        key = (index, content_index)
        if key in self._content_completed:
            return ()
        self._content_completed.add(key)
        return (ContentPartCompleted(kind, index, content_index, item_id, text),)

    def _require_content(self, index: int, content_index: int) -> None:
        key = (index, content_index)
        if key not in self._content_started:
            raise LegacyProviderExecutionError(
                "content delta arrived before part start"
            )
        if key in self._content_completed:
            raise LegacyProviderExecutionError(
                "content event arrived after part completion"
            )

    def _item(self, index: int, expected: str | None = None) -> tuple[str, str]:
        try:
            item = self._items[index]
        except KeyError as exc:
            raise LegacyProviderExecutionError(
                "event references an unopened output item"
            ) from exc
        if expected is not None and item[0] != expected:
            raise LegacyProviderExecutionError(
                "event references the wrong output item kind"
            )
        return item

    def _tool(self, index: int) -> dict[str, str]:
        self._item(index, expected="function_call")
        return self._tool_state[index]


def _responses_payload(
    request: GenerationRequest, *, model_ref: str
) -> Mapping[str, Any]:
    unsupported_sampling = set(request.sampling) - _RESPONSES_STREAM_SAMPLING
    if unsupported_sampling:
        raise LegacyPortContractError(
            "legacy Responses stream cannot preserve sampling fields: "
            + ", ".join(sorted(unsupported_sampling))
        )
    messages = [_copy_message(message) for message in request.messages]
    media_by_message: dict[int, list[tuple[int, Mapping[str, Any]]]] = {}
    for raw_part in request.media:
        part = dict(raw_part)
        message_index = part.pop("_message_index", None)
        content_index = part.pop("_content_index", None)
        part.pop("_role", None)
        if not isinstance(message_index, int) or not isinstance(content_index, int):
            raise LegacyPortContractError("canonical media is missing source position")
        if message_index < 0 or message_index >= len(messages) or content_index < 0:
            raise LegacyPortContractError("canonical media source position is invalid")
        media_by_message.setdefault(message_index, []).append((content_index, part))

    for message_index, positioned in media_by_message.items():
        message = messages[message_index]
        if message.get("type") == "function_call_output":
            raise LegacyPortContractError("function_call_output cannot own media")
        text_parts = list(message.get("content") or [])
        total = len(text_parts) + len(positioned)
        media = dict(positioned)
        if len(media) != len(positioned) or any(index >= total for index in media):
            raise LegacyPortContractError("canonical media positions are ambiguous")
        rebuilt: list[Mapping[str, Any]] = []
        text_iter = iter(text_parts)
        for index in range(total):
            rebuilt.append(media[index] if index in media else next(text_iter))
        message["content"] = rebuilt

    payload: dict[str, Any] = {
        "model": model_ref,
        "input": messages,
        "stream": True,
        "store": False,
    }
    if request.runtime.adapter_path is not None:
        payload["adapter_path"] = request.runtime.adapter_path
    if request.runtime.draft_model_id is not None:
        payload["draft_model_id"] = request.runtime.draft_model_id
    if request.tools:
        payload["tools"] = [dict(tool) for tool in request.tools]
    payload.update(dict(request.sampling))
    if request.reasoning:
        payload["reasoning"] = dict(request.reasoning)
    return payload


def _copy_message(message: Mapping[str, Any]) -> dict[str, Any]:
    if message.get("type") == "function_call_output":
        return {
            "type": "function_call_output",
            "call_id": _required_text(message, "call_id"),
            "output": _required_text(message, "output"),
        }
    copied: dict[str, Any] = {
        "role": _required_text(message, "role"),
        "content": [dict(part) for part in message.get("content", ())],
    }
    if message.get("type") == "message":
        copied["type"] = "message"
    return copied


def _looks_multimodal(model: ModelSpec) -> bool:
    identity = " ".join(
        value for value in (model.architecture, model.model_type) if value
    ).lower()
    return any(token in identity for token in ("vision", "_vl", "-vl", " vl"))


def _model_spec_is_bound(model: ModelSpec) -> bool:
    if model.revision is None or model.local_path is None:
        return False
    path = Path(model.local_path).expanduser().resolve()
    return path.is_dir() and path.name == model.revision


def _verified_model_ref(runtime: RuntimeKey, options: Mapping[str, Any]) -> str:
    raw_model_dir = options.get("model_dir")
    if raw_model_dir is None:
        if runtime.revision is not None:
            raise LegacyPortContractError(
                "revision-bearing legacy runtime requires an exact model_dir snapshot"
            )
        return runtime.model_id
    if not isinstance(raw_model_dir, str) or not raw_model_dir.strip():
        raise LegacyPortContractError("model_dir must be a non-empty path")
    path = Path(raw_model_dir).expanduser().resolve()
    if not path.is_dir():
        raise LegacyPortContractError("legacy model_dir must be an existing directory")
    if runtime.revision is not None and path.name != runtime.revision:
        raise LegacyPortContractError(
            "legacy model_dir does not resolve to the requested revision snapshot"
        )
    return str(path)


def _reject_sequential_options(options: Mapping[str, Any]) -> None:
    if options.get("stream") is False or options.get("legacy_generation_mode") in {
        "generate",
        "sequential",
        "non_stream",
    }:
        raise LegacyPortContractError(
            "legacy sequential generation has no truthful cooperative cancellation"
        )


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise LegacyPortContractError(f"{name} must be a positive number")
    return float(value)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise LegacyPortContractError(f"{name} must be a positive integer")
    return value


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise LegacyProviderExecutionError(f"legacy event {key} must be a mapping")
    return result


def _required_text(value: Mapping[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise LegacyProviderExecutionError(f"legacy event {key} must not be empty")
    return result


def _index(value: Mapping[str, Any], key: str) -> int:
    result = value.get(key)
    if isinstance(result, bool) or not isinstance(result, int) or result < 0:
        raise LegacyProviderExecutionError(f"legacy event {key} must be non-negative")
    return result


def _validate_optional_identity(
    value: Mapping[str, Any], key: str, expected: str
) -> None:
    actual = value.get(key)
    if actual is not None and actual != expected:
        raise LegacyProviderExecutionError(f"legacy event {key} identity mismatch")


def _usage_from(raw: Mapping[str, Any]) -> UsageUpdate | None:
    response = raw.get("response")
    usage = response.get("usage") if isinstance(response, Mapping) else raw.get("usage")
    if not isinstance(usage, Mapping):
        return None
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
    counters = (input_tokens, output_tokens)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in counters
    ):
        raise LegacyProviderExecutionError(
            "legacy usage counters must be non-negative integers"
        )
    return UsageUpdate(input_tokens, output_tokens, input_tokens + output_tokens)
