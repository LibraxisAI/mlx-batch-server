from __future__ import annotations

import asyncio
import contextlib
import threading
import time
import uuid
from collections.abc import AsyncGenerator, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
from PIL import Image

from ..chat.mlx.model_types import reset_request_local_runtime_state
from ..chat.mlx.runtime_aliases import resolve_runtime_target
from ..core.config import get_settings
from ..utils.logger import logger
from ..utils.model_limits import extract_context_length, resolve_max_tokens
from .vlm_cache import (
    get_vlm_backend,
    resolve_vlm_model_id,
    vlm_execution,
)

try:  # Optional dependency
    from mlx_vlm import apply_chat_template as _vlm_apply_chat_template
    from mlx_vlm.generate import (
        BatchGenerator as _VlmBatchGenerator,
    )
    from mlx_vlm.generate import (
        batch_generate as _vlm_batch_generate,
    )
    from mlx_vlm.sample_utils import top_p_sampling as _vlm_top_p_sampling
    from mlx_vlm.utils import prepare_inputs as _vlm_prepare_inputs
except Exception:  # pragma: no cover - optional dependency
    _vlm_apply_chat_template = None
    _VlmBatchGenerator = None
    _vlm_batch_generate = None
    _vlm_top_p_sampling = None
    _vlm_prepare_inputs = None


@dataclass
class VlmBatchResult:
    text: str
    prompt_tokens: int = 0
    generation_tokens: int = 0
    total_tokens: int = 0


@dataclass
class PendingVlmRequest:
    request_id: str
    messages: list[dict[str, Any]]
    images: list[Any] | None
    max_tokens: int | None
    temperature: float | None
    top_p: float | None
    future: asyncio.Future[VlmBatchResult]
    created_at: float


class VlmBatchCoordinator:
    """Micro-batch coordinator for eligible single-image VLM requests."""

    def __init__(
        self,
        model_id: str,
        adapter_path: str | None = None,
        draft_model_id: str | None = None,
        batch_window_ms: int = 50,
        max_batch_size: int = 4,
        group_by_shape: bool = True,
    ) -> None:
        target = resolve_runtime_target(
            model_id,
            adapter_path=adapter_path,
            draft_model_id=draft_model_id,
        )
        self.model_id = target.model_id
        self.adapter_path = target.adapter_path
        self.draft_model_id = target.draft_model_id
        self._batch_window_ms = batch_window_ms
        self._max_batch_size = max_batch_size
        self._group_by_shape = group_by_shape

        self._pending: dict[str, PendingVlmRequest] = {}
        self._request_lock = asyncio.Lock()
        self._new_request_event = asyncio.Event()
        self._shutdown_event = asyncio.Event()
        self._worker_task: asyncio.Task | None = None

        self._total_requests = 0
        self._total_batches = 0
        self._last_batch_size = 0

    async def _ensure_worker_running(self) -> None:
        if self._worker_task is None or self._worker_task.done():
            self._shutdown_event.clear()
            self._worker_task = asyncio.create_task(self._worker_loop())
            logger.info("Started VLM batch coordinator worker (%s)", self.model_id)

    async def shutdown(self) -> None:
        self._shutdown_event.set()
        self._new_request_event.set()
        if self._worker_task:
            self._worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_task

        async with self._request_lock:
            pending = list(self._pending.values())
            self._pending.clear()

        for item in pending:
            if not item.future.done():
                item.future.set_exception(RuntimeError("VLM batch worker shut down"))

    async def submit_request(
        self,
        *,
        messages: list[dict[str, Any]],
        images: list[Any] | None,
        max_tokens: int | None,
        temperature: float | None,
        top_p: float | None,
    ) -> VlmBatchResult:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[VlmBatchResult] = loop.create_future()
        request_id = uuid.uuid4().hex
        pending = PendingVlmRequest(
            request_id=request_id,
            messages=messages,
            images=images,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            future=future,
            created_at=time.time(),
        )

        async with self._request_lock:
            self._pending[request_id] = pending
            self._total_requests += 1

        self._new_request_event.set()
        await self._ensure_worker_running()
        return await future

    async def _worker_loop(self) -> None:
        try:
            while not self._shutdown_event.is_set():
                try:
                    await asyncio.wait_for(self._new_request_event.wait(), timeout=1.0)
                    self._new_request_event.clear()
                except TimeoutError:
                    continue

                await asyncio.sleep(self._batch_window_ms / 1000.0)

                batch_items = await self._drain_pending_requests()
                if not batch_items:
                    continue

                for group in self._group_requests(batch_items):
                    for chunk in _chunk_list(group, self._max_batch_size):
                        await self._process_batch(chunk)

        except asyncio.CancelledError:
            logger.info("VLM batch coordinator cancelled (%s)", self.model_id)
        except Exception as exc:
            logger.error("VLM batch coordinator error (%s): %s", self.model_id, exc)
            raise
        finally:
            logger.info("VLM batch coordinator stopped (%s)", self.model_id)

    async def _drain_pending_requests(self) -> list[PendingVlmRequest]:
        now = time.time()
        max_wait = get_settings().local_timeout
        expired: list[PendingVlmRequest] = []
        batch_items: list[PendingVlmRequest] = []

        async with self._request_lock:
            for request_id, request in list(self._pending.items()):
                if now - request.created_at > max_wait:
                    expired.append(self._pending.pop(request_id))
                else:
                    batch_items.append(request)
            self._pending.clear()

        for item in expired:
            if not item.future.done():
                item.future.set_exception(TimeoutError("VLM batch request timed out"))

        return batch_items

    def _group_requests(
        self, requests: list[PendingVlmRequest]
    ) -> list[list[PendingVlmRequest]]:
        grouped: dict[tuple, list[PendingVlmRequest]] = {}
        for item in requests:
            key = (
                bool(item.images),
                _normalize_temperature(item.temperature),
                _normalize_top_p(item.top_p),
            )
            grouped.setdefault(key, []).append(item)
        return list(grouped.values())

    async def _process_batch(self, batch: list[PendingVlmRequest]) -> None:
        if not batch:
            return

        self._total_batches += 1
        self._last_batch_size = len(batch)

        try:
            _require_vlm_batch_support()

            settings = get_settings()
            resize_shape = _parse_resize_shape(settings.vlm_batch_resize_shape)

            prompts, images = _collect_vlm_batch_inputs(batch)
            with vlm_execution(
                self.model_id,
                adapter_path=self.adapter_path,
                draft_model_id=self.draft_model_id,
            ):
                model, processor = get_vlm_backend(
                    self.model_id,
                    adapter_path=self.adapter_path,
                    draft_model_id=self.draft_model_id,
                    surface="llm",
                )
                reset_request_local_runtime_state(model)
                context_length = _infer_vlm_context_length(model, processor)
                prompt_lengths = _estimate_prompt_lengths(
                    model=model,
                    processor=processor,
                    batch=batch,
                )
                max_tokens = _collect_vlm_max_tokens(
                    batch,
                    context_length=context_length,
                    prompt_lengths=prompt_lengths,
                )
                sampler = _build_sampler(
                    temperature=batch[0].temperature,
                    top_p=batch[0].top_p,
                )

                kwargs: dict[str, Any] = {"group_by_shape": self._group_by_shape}
                if sampler is not None:
                    kwargs["sampler"] = sampler
                if resize_shape is not None:
                    kwargs["resize_shape"] = resize_shape

                response = _vlm_batch_generate(
                    model,
                    processor,
                    images=images,
                    prompts=prompts,
                    max_tokens=max_tokens,
                    **kwargs,
                )
                texts = response.texts

            prompt_tokens = int(getattr(response, "prompt_tokens", 0) or 0)
            generation_tokens = int(getattr(response, "generation_tokens", 0) or 0)
            total_tokens = int(getattr(response, "total_tokens", 0) or 0)

            for item, text in zip(batch, texts, strict=False):
                if not item.future.done():
                    item.future.set_result(
                        VlmBatchResult(
                            text=text,
                            prompt_tokens=prompt_tokens,
                            generation_tokens=generation_tokens,
                            total_tokens=total_tokens,
                        )
                    )

        except Exception as exc:
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(exc)

    def stats(self) -> dict[str, Any]:
        pending = len(self._pending)
        avg_batch = (
            self._total_requests / self._total_batches if self._total_batches else 0
        )
        return {
            "model_id": self.model_id,
            "pending": pending,
            "total_requests": self._total_requests,
            "total_batches": self._total_batches,
            "avg_batch_size": round(avg_batch, 2),
            "last_batch_size": self._last_batch_size,
        }


@dataclass
class VlmStreamChunk:
    text: str
    finish_reason: str | None = None


@dataclass
class PendingVlmStreamRequest:
    request_id: str
    messages: list[dict[str, Any]]
    images: list[Any] | None
    max_tokens: int | None
    temperature: float | None
    top_p: float | None
    response_queue: asyncio.Queue
    created_at: float


class VlmStreamBatchCoordinator:
    """Batch coordinator for token-level streaming VLM responses."""

    def __init__(
        self,
        model_id: str,
        adapter_path: str | None = None,
        draft_model_id: str | None = None,
        batch_window_ms: int = 50,
        max_batch_size: int = 4,
        group_by_shape: bool = True,
    ) -> None:
        target = resolve_runtime_target(
            model_id,
            adapter_path=adapter_path,
            draft_model_id=draft_model_id,
        )
        self.model_id = target.model_id
        self.adapter_path = target.adapter_path
        self.draft_model_id = target.draft_model_id
        self._batch_window_ms = batch_window_ms
        self._max_batch_size = max_batch_size
        self._group_by_shape = group_by_shape

        self._pending_requests: dict[str, PendingVlmStreamRequest] = {}
        self._active_requests: dict[str, asyncio.Queue] = {}
        self._request_lock = asyncio.Lock()
        self._new_request_event = asyncio.Event()
        self._shutdown_event = asyncio.Event()
        self._worker_task: asyncio.Task | None = None

        self._total_requests = 0
        self._total_batches = 0
        self._last_batch_size = 0

    async def _ensure_worker_running(self) -> None:
        if self._worker_task is None or self._worker_task.done():
            self._shutdown_event.clear()
            self._worker_task = asyncio.create_task(self._worker_loop())
            logger.info("Started VLM stream batch worker (%s)", self.model_id)

    async def shutdown(self) -> None:
        self._shutdown_event.set()
        self._new_request_event.set()
        if self._worker_task:
            self._worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_task

        async with self._request_lock:
            pending = list(self._pending_requests.values())
            active = list(self._active_requests.values())
            self._pending_requests.clear()
            self._active_requests.clear()

        for req in pending:
            await req.response_queue.put(Exception("VLM stream worker shut down"))
        for queue in active:
            await queue.put(Exception("VLM stream worker shut down"))

    async def stream_request(
        self,
        *,
        messages: list[dict[str, Any]],
        images: list[Any] | None,
        max_tokens: int | None,
        temperature: float | None,
        top_p: float | None,
    ) -> AsyncGenerator[VlmStreamChunk, None]:
        request_id = f"vlm_{uuid.uuid4().hex[:12]}"
        response_queue: asyncio.Queue = asyncio.Queue()

        pending = PendingVlmStreamRequest(
            request_id=request_id,
            messages=messages,
            images=images,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            response_queue=response_queue,
            created_at=time.time(),
        )

        async with self._request_lock:
            self._pending_requests[request_id] = pending
            self._total_requests += 1

        await self._ensure_worker_running()
        self._new_request_event.set()

        try:
            while True:
                try:
                    item = await asyncio.wait_for(response_queue.get(), timeout=300.0)
                except TimeoutError:
                    logger.warning(
                        "VLM stream request %s timed out waiting for response",
                        request_id,
                    )
                    break

                if isinstance(item, Exception):
                    raise item

                yield item

                if item.finish_reason is not None:
                    break

        except asyncio.CancelledError:
            await self._cancel_request(request_id)
            raise
        finally:
            async with self._request_lock:
                self._pending_requests.pop(request_id, None)
                self._active_requests.pop(request_id, None)

    async def _cancel_request(self, request_id: str) -> None:
        async with self._request_lock:
            self._pending_requests.pop(request_id, None)
            self._active_requests.pop(request_id, None)

    async def _worker_loop(self) -> None:
        try:
            while not self._shutdown_event.is_set():
                try:
                    await asyncio.wait_for(self._new_request_event.wait(), timeout=1.0)
                    self._new_request_event.clear()
                except TimeoutError:
                    continue

                await asyncio.sleep(self._batch_window_ms / 1000.0)

                batch = await self._drain_stream_pending()
                if not batch:
                    continue

                for group in self._group_stream_requests(batch):
                    for chunk in _chunk_list(group, self._max_batch_size):
                        await self._process_stream_batch(chunk)

        except asyncio.CancelledError:
            logger.info("VLM stream worker cancelled (%s)", self.model_id)
        except Exception as exc:
            logger.error("VLM stream worker error (%s): %s", self.model_id, exc)
            raise
        finally:
            logger.info("VLM stream worker stopped (%s)", self.model_id)

    async def _drain_stream_pending(self) -> list[PendingVlmStreamRequest]:
        now = time.time()
        max_wait = get_settings().local_timeout
        expired: list[PendingVlmStreamRequest] = []
        batch: list[PendingVlmStreamRequest] = []

        async with self._request_lock:
            for request_id, request in list(self._pending_requests.items()):
                if now - request.created_at > max_wait:
                    expired.append(self._pending_requests.pop(request_id))
                else:
                    batch.append(request)
            self._pending_requests.clear()

        for item in expired:
            await item.response_queue.put(TimeoutError("VLM stream request timed out"))

        return batch

    async def _process_stream_batch(
        self,
        batch: list[PendingVlmStreamRequest],
    ) -> None:
        if not batch:
            return

        self._total_batches += 1
        self._last_batch_size = len(batch)

        try:
            with vlm_execution(
                self.model_id,
                adapter_path=self.adapter_path,
                draft_model_id=self.draft_model_id,
            ):
                state = _init_stream_batch_state(
                    model_id=self.model_id,
                    adapter_path=self.adapter_path,
                    draft_model_id=self.draft_model_id,
                    batch=batch,
                )
                await self._register_active_requests(batch)

                finished = await _dispatch_stream_tokens(
                    batch=batch,
                    gen=state["gen"],
                    tokenizer=state["tokenizer"],
                    uids=state["uids"],
                )

            await self._emit_stream_finalizers(batch, finished)
        except Exception as exc:
            logger.error("VLM stream batch failed: %s", exc)
            async with self._request_lock:
                for req in batch:
                    queue = self._active_requests.pop(req.request_id, None)
                    if queue is not None:
                        await queue.put(Exception(str(exc)))
        finally:
            async with self._request_lock:
                for req in batch:
                    self._active_requests.pop(req.request_id, None)

    async def _register_active_requests(
        self,
        batch: list[PendingVlmStreamRequest],
    ) -> None:
        async with self._request_lock:
            for req in batch:
                self._active_requests[req.request_id] = req.response_queue

    async def _emit_stream_finalizers(
        self,
        batch: list[PendingVlmStreamRequest],
        finished: list[bool],
    ) -> None:
        for idx, req in enumerate(batch):
            if req.request_id in self._active_requests and not finished[idx]:
                await req.response_queue.put(
                    VlmStreamChunk(
                        text="",
                        finish_reason="stop",
                    )
                )

    def stats(self) -> dict[str, Any]:
        pending = len(self._pending_requests)
        avg_batch = (
            self._total_requests / self._total_batches if self._total_batches else 0
        )
        return {
            "model_id": self.model_id,
            "pending": pending,
            "total_requests": self._total_requests,
            "total_batches": self._total_batches,
            "avg_batch_size": round(avg_batch, 2),
            "last_batch_size": self._last_batch_size,
        }

    def _group_stream_requests(
        self, requests: list[PendingVlmStreamRequest]
    ) -> list[list[PendingVlmStreamRequest]]:
        grouped: dict[tuple, list[PendingVlmStreamRequest]] = {}
        for item in requests:
            key = (
                bool(item.images),
                _normalize_temperature(item.temperature),
                _normalize_top_p(item.top_p),
                _stream_shape_group_key(item.images) if self._group_by_shape else None,
            )
            grouped.setdefault(key, []).append(item)
        return list(grouped.values())


VlmRuntimeKey = tuple[str, str | None, str | None]


def _resolve_vlm_runtime_key(
    model_id: str,
    *,
    adapter_path: str | None = None,
    draft_model_id: str | None = None,
) -> VlmRuntimeKey:
    target = resolve_runtime_target(
        model_id,
        adapter_path=adapter_path,
        draft_model_id=draft_model_id,
    )
    return (
        resolve_vlm_model_id(target.model_id),
        target.adapter_path,
        target.draft_model_id,
    )


def _format_vlm_runtime_key(runtime_key: VlmRuntimeKey) -> str:
    model_id, adapter_path, draft_model_id = runtime_key
    parts = [model_id]
    if adapter_path is not None:
        parts.append(f"adapter={adapter_path}")
    if draft_model_id is not None:
        parts.append(f"draft={draft_model_id}")
    return " | ".join(parts)


_VLM_COORDINATORS: dict[VlmRuntimeKey, VlmBatchCoordinator] = {}
_VLM_LOCK = threading.Lock()
_VLM_STREAM_COORDINATORS: dict[VlmRuntimeKey, VlmStreamBatchCoordinator] = {}
_VLM_STREAM_LOCK = threading.Lock()


def get_vlm_batch_coordinator(
    model_id: str,
    *,
    adapter_path: str | None = None,
    draft_model_id: str | None = None,
    batch_window_ms: int,
    max_batch_size: int,
    group_by_shape: bool,
) -> VlmBatchCoordinator:
    runtime_key = _resolve_vlm_runtime_key(
        model_id,
        adapter_path=adapter_path,
        draft_model_id=draft_model_id,
    )
    with _VLM_LOCK:
        coordinator = _VLM_COORDINATORS.get(runtime_key)
        if coordinator is None:
            coordinator = VlmBatchCoordinator(
                model_id=runtime_key[0],
                adapter_path=runtime_key[1],
                draft_model_id=runtime_key[2],
                batch_window_ms=batch_window_ms,
                max_batch_size=max_batch_size,
                group_by_shape=group_by_shape,
            )
            _VLM_COORDINATORS[runtime_key] = coordinator
        return coordinator


def get_vlm_stream_coordinator(
    model_id: str,
    *,
    adapter_path: str | None = None,
    draft_model_id: str | None = None,
    batch_window_ms: int,
    max_batch_size: int,
    group_by_shape: bool = True,
) -> VlmStreamBatchCoordinator:
    runtime_key = _resolve_vlm_runtime_key(
        model_id,
        adapter_path=adapter_path,
        draft_model_id=draft_model_id,
    )
    with _VLM_STREAM_LOCK:
        coordinator = _VLM_STREAM_COORDINATORS.get(runtime_key)
        if coordinator is None:
            coordinator = VlmStreamBatchCoordinator(
                model_id=runtime_key[0],
                adapter_path=runtime_key[1],
                draft_model_id=runtime_key[2],
                batch_window_ms=batch_window_ms,
                max_batch_size=max_batch_size,
                group_by_shape=group_by_shape,
            )
            _VLM_STREAM_COORDINATORS[runtime_key] = coordinator
        return coordinator


def get_vlm_batch_stats() -> dict[str, Any]:
    stats: dict[str, Any] = {}
    with _VLM_LOCK:
        stats.update(
            {
                _format_vlm_runtime_key(runtime_key): coord.stats()
                for runtime_key, coord in _VLM_COORDINATORS.items()
            }
        )
    with _VLM_STREAM_LOCK:
        stream_stats = {
            _format_vlm_runtime_key(runtime_key): coord.stats()
            for runtime_key, coord in _VLM_STREAM_COORDINATORS.items()
        }
    if stream_stats:
        stats["stream"] = stream_stats
    return stats


def get_loaded_vlm_batch_models() -> list[str]:
    """Return model IDs with active VLM batch or stream coordinators."""
    with _VLM_LOCK:
        loaded = [coord.model_id for coord in _VLM_COORDINATORS.values()]
    with _VLM_STREAM_LOCK:
        loaded.extend(coord.model_id for coord in _VLM_STREAM_COORDINATORS.values())
    return list(dict.fromkeys(loaded))


async def shutdown_all_vlm_coordinators() -> None:
    with _VLM_LOCK:
        coordinators = list(_VLM_COORDINATORS.values())
        _VLM_COORDINATORS.clear()
    for coord in coordinators:
        await coord.shutdown()

    with _VLM_STREAM_LOCK:
        stream_coords = list(_VLM_STREAM_COORDINATORS.values())
        _VLM_STREAM_COORDINATORS.clear()
    for coord in stream_coords:
        await coord.shutdown()


async def shutdown_vlm_coordinator(
    model_id: str,
    *,
    adapter_path: str | None = None,
    draft_model_id: str | None = None,
) -> int:
    runtime_key = _resolve_vlm_runtime_key(
        model_id,
        adapter_path=adapter_path,
        draft_model_id=draft_model_id,
    )
    removed = 0

    with _VLM_LOCK:
        if adapter_path is None and draft_model_id is None:
            keys = [key for key in _VLM_COORDINATORS if key[0] == runtime_key[0]]
        else:
            keys = [runtime_key]
        coords = [
            _VLM_COORDINATORS.pop(key) for key in keys if key in _VLM_COORDINATORS
        ]
    for coord in coords:
        removed += 1
        await coord.shutdown()

    with _VLM_STREAM_LOCK:
        if adapter_path is None and draft_model_id is None:
            stream_keys = [
                key for key in _VLM_STREAM_COORDINATORS if key[0] == runtime_key[0]
            ]
        else:
            stream_keys = [runtime_key]
        stream_coords = [
            _VLM_STREAM_COORDINATORS.pop(key)
            for key in stream_keys
            if key in _VLM_STREAM_COORDINATORS
        ]
    for stream_coord in stream_coords:
        removed += 1
        await stream_coord.shutdown()

    return removed


def _normalize_temperature(value: float | None) -> float:
    if value is None:
        return 0.0
    return float(value)


def _normalize_top_p(value: float | None) -> float:
    if value is None:
        return 1.0
    return float(value)


def _build_sampler(
    *,
    temperature: float | None,
    top_p: float | None,
) -> Callable[[mx.array], mx.array] | None:
    temp = _normalize_temperature(temperature)
    if temp <= 0:
        return None

    tp = _normalize_top_p(top_p)
    if _vlm_top_p_sampling is None:
        raise RuntimeError("mlx-vlm is required for VLM batching")

    def sampler(logprobs: mx.array) -> mx.array:
        if 0 < tp < 1.0:
            return _vlm_top_p_sampling(logprobs, tp, temp)
        return mx.random.categorical(logprobs * (1 / temp))

    return sampler


def _chunk_list(items: list[Any], size: int) -> list[list[Any]]:
    if size <= 0:
        return [items]
    return [items[i : i + size] for i in range(0, len(items), size)]


def _single_image_size(image: Any) -> tuple[int, int] | None:
    if hasattr(image, "size"):
        size = image.size
        if (
            isinstance(size, tuple)
            and len(size) == 2
            and all(isinstance(value, int) for value in size)
        ):
            return size

    if isinstance(image, str | Path):
        path = Path(image).expanduser()
        if path.exists() and path.is_file():
            with Image.open(path) as opened:
                return opened.size

    return None


def _stream_shape_group_key(
    images: list[Any] | None,
) -> tuple[tuple[int, int] | None, ...]:
    if not images:
        return ()
    return tuple(_single_image_size(image) for image in images)


def _init_stream_batch_state(
    *,
    model_id: str,
    adapter_path: str | None = None,
    draft_model_id: str | None = None,
    batch: list[PendingVlmStreamRequest],
) -> dict[str, Any]:
    apply_chat_template, batch_generator, prepare_inputs = _require_vlm_stream_support()
    settings = get_settings()
    resize_shape = _parse_resize_shape(settings.vlm_batch_resize_shape)
    pad_uniform = settings.vlm_batch_pad_to_uniform_size

    model, processor = get_vlm_backend(
        model_id,
        adapter_path=adapter_path,
        draft_model_id=draft_model_id,
        surface="llm",
    )
    reset_request_local_runtime_state(model)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    if hasattr(processor, "detokenizer"):
        processor.detokenizer.reset()

    batch_inputs = _prepare_stream_batch_inputs(
        apply_chat_template=apply_chat_template,
        prepare_inputs=prepare_inputs,
        model=model,
        processor=processor,
        batch=batch,
        resize_shape=resize_shape,
        pad_uniform=pad_uniform,
    )

    input_ids = batch_inputs["input_ids"]
    input_ids_list = batch_inputs["input_ids_list"]
    pixel_values = batch_inputs["pixel_values"]
    data_kwargs = batch_inputs["data_kwargs"]
    max_tokens = batch_inputs["max_tokens"]

    sampler = _build_sampler(
        temperature=batch[0].temperature,
        top_p=batch[0].top_p,
    )

    language_model = getattr(model, "language_model", model)
    gen = batch_generator(
        language_model,
        processor,
        prefill_batch_size=len(batch),
        completion_batch_size=len(batch),
        sampler=sampler,
    )

    prompt_kwargs = _build_stream_prompt_kwargs(
        model,
        input_ids,
        pixel_values,
        data_kwargs,
    )

    return {
        "tokenizer": tokenizer,
        "gen": gen,
        # Newer mlx-vlm BatchGenerator expects precomputed prompt kwargs,
        # especially inputs_embeds, at insert-time for VLM prefill.
        "uids": gen.insert(
            input_ids_list,
            max_tokens,
            prompt_kwargs=[prompt_kwargs] * len(input_ids_list),
        ),
    }


def _require_vlm_batch_support() -> None:
    if _vlm_batch_generate is None:
        raise RuntimeError("mlx-vlm is required for VLM batch generation")


def _require_vlm_stream_support():
    if (
        _vlm_apply_chat_template is None
        or _VlmBatchGenerator is None
        or _vlm_prepare_inputs is None
    ):
        raise RuntimeError("mlx-vlm is required for VLM stream batching")
    return _vlm_apply_chat_template, _VlmBatchGenerator, _vlm_prepare_inputs


def _collect_vlm_batch_inputs(
    batch: list[PendingVlmRequest],
) -> tuple[list[list[dict[str, Any]]], list[Any] | None]:
    prompts = [item.messages for item in batch]
    images = None
    if batch[0].images:
        images = [item.images[0] if item.images else None for item in batch]
    return prompts, images


def _infer_vlm_context_length(model: Any, processor: Any) -> int | None:
    config = getattr(model, "config", None)
    tokenizer = getattr(processor, "tokenizer", None)
    return extract_context_length(config, tokenizer)


def _estimate_prompt_lengths(
    *,
    model: Any,
    processor: Any,
    batch: list[PendingVlmRequest],
) -> list[int] | None:
    if _vlm_apply_chat_template is None:
        return None

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None or not hasattr(tokenizer, "encode"):
        return None

    num_images_list = [1 if req.images else 0 for req in batch]
    lengths: list[int] = []
    for idx, req in enumerate(batch):
        try:
            formatted = _vlm_apply_chat_template(
                processor,
                model.config,
                req.messages,
                num_images=num_images_list[idx],
            )
            lengths.append(len(tokenizer.encode(formatted)))
        except Exception:
            return None
    return lengths


def _collect_vlm_max_tokens(
    batch: Sequence[PendingVlmRequest | PendingVlmStreamRequest],
    *,
    context_length: int | None,
    prompt_lengths: Sequence[int] | None,
) -> list[int]:
    max_tokens_list: list[int] = []
    for idx, req in enumerate(batch):
        prompt_tokens = prompt_lengths[idx] if prompt_lengths else None
        max_tokens_list.append(
            resolve_max_tokens(
                requested=req.max_tokens,
                context_length=context_length,
                prompt_tokens=prompt_tokens,
                context_label=getattr(req, "request_id", "vlm"),
            )
        )
    return max_tokens_list


def _prepare_stream_batch_inputs(
    *,
    apply_chat_template,
    prepare_inputs,
    model,
    processor,
    batch: list[PendingVlmStreamRequest],
    resize_shape: tuple[int, int] | None,
    pad_uniform: bool,
) -> dict[str, Any]:
    prompts = [req.messages for req in batch]
    images = None
    if batch[0].images:
        images = [req.images[0] if req.images else None for req in batch]

    num_images_list = [1 if req.images else 0 for req in batch]
    formatted_prompts = [
        apply_chat_template(
            processor,
            model.config,
            prompt,
            num_images=num_images_list[idx],
        )
        for idx, prompt in enumerate(prompts)
    ]

    add_special_tokens = (
        not hasattr(processor, "chat_template")
        if model.config.model_type in ["gemma3", "gemma3n"]
        else True
    )

    image_token_index = getattr(model.config, "image_token_index", None)
    inputs = prepare_inputs(
        processor,
        images=images,
        audio=None,
        prompts=formatted_prompts,
        image_token_index=image_token_index,
        resize_shape=resize_shape,
        add_special_tokens=add_special_tokens,
        pad_to_uniform_size=pad_uniform,
    )

    input_ids = inputs.get("input_ids")
    pixel_values = inputs.get("pixel_values")
    data_kwargs = {
        k: v
        for k, v in inputs.items()
        if k not in ["input_ids", "pixel_values", "attention_mask"]
    }

    input_ids_list = input_ids.tolist()
    if input_ids.ndim == 1:
        input_ids_list = [input_ids_list]

    context_length = _infer_vlm_context_length(model, processor)
    prompt_lengths = [len(ids) for ids in input_ids_list]
    max_tokens = _collect_vlm_max_tokens(
        list(batch),
        context_length=context_length,
        prompt_lengths=prompt_lengths,
    )

    return {
        "input_ids": input_ids,
        "input_ids_list": input_ids_list,
        "pixel_values": pixel_values,
        "data_kwargs": data_kwargs,
        "max_tokens": max_tokens,
    }


def _build_stream_prompt_kwargs(
    model,
    input_ids: mx.array,
    pixel_values: mx.array | None,
    data_kwargs: dict[str, Any],
) -> dict[str, Any]:
    if pixel_values is None:
        return {}

    embedding_output = model.get_input_embeddings(
        input_ids, pixel_values, **data_kwargs
    )

    if isinstance(embedding_output, dict):
        embed_kwargs = embedding_output
    elif hasattr(embedding_output, "to_dict"):
        embed_kwargs = {
            k: v for k, v in embedding_output.to_dict().items() if v is not None
        }
    else:
        embed_kwargs = {"inputs_embeds": embedding_output}

    return {"pixel_values": pixel_values, **data_kwargs, **embed_kwargs}


async def _dispatch_stream_tokens(
    *,
    batch: list[PendingVlmStreamRequest],
    gen,
    tokenizer,
    uids: list[int],
) -> list[bool]:
    uid_to_idx = {uid: idx for idx, uid in enumerate(uids)}
    token_buffers: list[list[int]] = [[] for _ in batch]
    last_texts = [""] * len(batch)
    finished = [False] * len(batch)

    while responses := gen.next():
        for resp in responses:
            idx = uid_to_idx.get(resp.uid)
            if idx is None:
                continue

            delta = ""
            if resp.token is not None:
                token_buffers[idx].append(resp.token)
                new_text = tokenizer.decode(token_buffers[idx])
                delta = new_text[len(last_texts[idx]) :]
                last_texts[idx] = new_text

            if delta or resp.finish_reason is not None:
                await batch[idx].response_queue.put(
                    VlmStreamChunk(
                        text=delta,
                        finish_reason=resp.finish_reason,
                    )
                )
            if resp.finish_reason is not None:
                finished[idx] = True

    return finished


def _parse_resize_shape(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    raw = value.strip().lower()
    if "x" in raw:
        parts = raw.split("x")
    elif "," in raw:
        parts = raw.split(",")
    else:
        parts = [raw]

    try:
        nums = [int(p.strip()) for p in parts if p.strip()]
    except ValueError:
        return None

    if not nums:
        return None
    if len(nums) == 1:
        return (nums[0], nums[0])
    return (nums[0], nums[1])
