"""Batch Request Coordinator - Transparent batching for concurrent requests.

This module provides a coordinator that handles multiple concurrent chat
completion requests using BatchChatGenerator. The batching is transparent
to clients - each request gets its own streaming response as if it were
handled individually.

Architecture:
- Single BatchChatGenerator instance shared across requests
- Per-request async queues for token dispatch
- Background worker runs the batch generation loop
- Automatic batch collection with configurable window

Vibecrafted with AI Agents by VetCoders (c)2026 VetCoders
Co-Authored-By: [Maciej](void@div0.space) & [Klaudiusz](the1st@whoai.am)
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..utils.logger import logger

if TYPE_CHECKING:
    from .generator import BatchChatGenerator, BatchStreamChunk

__all__ = [
    "BatchRequestCoordinator",
    "get_batch_coordinator",
    "get_loaded_batch_models",
    "shutdown_all_coordinators",
    "shutdown_batch_coordinator",
]


@dataclass
class PendingRequest:
    """A request waiting to be processed."""

    request_id: str
    messages: list[dict[str, Any]]
    max_tokens: int | None
    sampler_config: dict[str, Any] | None
    template_kwargs: dict[str, Any] | None
    response_queue: asyncio.Queue
    created_at: float


class BatchRequestCoordinator:
    """Coordinates multiple concurrent requests through batch inference.

    This coordinator provides a transparent interface for handling concurrent
    chat completion requests. Requests are automatically batched and processed
    together for improved throughput.

    The interface is designed to be a drop-in replacement for single-request
    processing:

    ```python
    coordinator = get_batch_coordinator(model_id)

    # Process a single request (but internally batched with others)
    async for chunk in coordinator.stream_request(messages, max_tokens=100):
        print(chunk.text)
    ```

    Key features:
    - Automatic request batching with configurable window
    - Per-request streaming with independent cancellation
    - Graceful handling of client disconnects
    - Statistics and monitoring
    """

    def __init__(
        self,
        model_id: str,
        adapter_path: str | None = None,
        completion_batch_size: int = 32,
        prefill_batch_size: int = 8,
        prefill_step_size: int = 2048,
        batch_window_ms: int = 50,
        max_batch_size: int = 10,
    ):
        """Initialize BatchRequestCoordinator.

        Args:
            model_id: Model name/path for BatchChatGenerator
            adapter_path: Optional LoRA adapter path
            completion_batch_size: Sequences per batch step
            prefill_batch_size: Sequences to prefill together
            prefill_step_size: Tokens per prefill step
            batch_window_ms: Time window to collect requests (ms)
            max_batch_size: Maximum requests per batch
        """
        self.model_id = model_id
        self.adapter_path = adapter_path
        self._completion_batch_size = completion_batch_size
        self._prefill_batch_size = prefill_batch_size
        self._prefill_step_size = prefill_step_size
        self._batch_window_ms = batch_window_ms
        self._max_batch_size = max_batch_size

        # Generator instance - created lazily
        self._generator: BatchChatGenerator | None = None
        self._generator_lock = threading.Lock()

        # Request management
        self._pending_requests: dict[str, PendingRequest] = {}
        self._active_requests: dict[str, asyncio.Queue] = {}
        self._request_lock = asyncio.Lock()

        # Worker state
        self._worker_task: asyncio.Task | None = None
        self._shutdown_event = asyncio.Event()
        self._new_request_event = asyncio.Event()

        # Statistics
        self._total_requests = 0
        self._total_batches = 0
        self._total_tokens = 0

    def _get_or_create_generator(self) -> BatchChatGenerator:
        """Get existing or create new BatchChatGenerator."""
        with self._generator_lock:
            if self._generator is None:
                from ..chat.mlx.chat_generator import ChatGenerator
                from .generator import BatchChatGenerator

                # Reuse the cached chat wrapper so batch mode shares a single hot
                # MLX model instance with the rest of the text stack.
                shared_wrapper = ChatGenerator.get_or_create(
                    model_id=self.model_id,
                    adapter_path=self.adapter_path,
                    draft_model_id=None,
                )
                self._generator = BatchChatGenerator.from_chat_generator(
                    shared_wrapper,
                    completion_batch_size=self._completion_batch_size,
                    prefill_batch_size=self._prefill_batch_size,
                    prefill_step_size=self._prefill_step_size,
                )
                logger.info(
                    "Created BatchChatGenerator for coordinator from shared wrapper: "
                    f"model={self.model_id}"
                )
            return self._generator

    async def _ensure_worker_running(self) -> None:
        """Ensure the background worker is running."""
        if self._worker_task is None or self._worker_task.done():
            self._shutdown_event.clear()
            self._worker_task = asyncio.create_task(self._worker_loop())
            logger.info("Started batch coordinator worker")

    async def _worker_loop(self) -> None:
        """Background worker that processes batches."""
        logger.info("Batch coordinator worker started")

        try:
            while not self._shutdown_event.is_set():
                # Wait for new requests or shutdown
                try:
                    await asyncio.wait_for(
                        self._new_request_event.wait(),
                        timeout=1.0,
                    )
                    self._new_request_event.clear()
                except TimeoutError:
                    continue

                # Collect batch with time window
                await asyncio.sleep(self._batch_window_ms / 1000.0)

                # Collect pending requests
                async with self._request_lock:
                    if not self._pending_requests:
                        continue

                    # Take up to max_batch_size requests
                    batch_ids = list(self._pending_requests.keys())[
                        : self._max_batch_size
                    ]
                    batch = [self._pending_requests.pop(rid) for rid in batch_ids]

                if not batch:
                    continue

                # Process batch
                await self._process_batch(batch)

        except asyncio.CancelledError:
            logger.info("Batch coordinator worker cancelled")
        except Exception as e:
            logger.error(f"Batch coordinator worker error: {e}")
            raise
        finally:
            logger.info("Batch coordinator worker stopped")

    async def _process_batch(self, batch: list[PendingRequest]) -> None:
        """Process a batch of requests."""
        from .generator import BatchRequest

        if not batch:
            return

        logger.info(f"Processing batch of {len(batch)} requests")
        self._total_batches += 1

        # Convert to BatchRequest objects
        batch_requests = [
            BatchRequest(
                id=req.request_id,
                messages=req.messages,
                max_tokens=req.max_tokens,
                sampler_config=req.sampler_config,
                template_kwargs=req.template_kwargs,
            )
            for req in batch
        ]

        # Register queues for active requests
        async with self._request_lock:
            for req in batch:
                self._active_requests[req.request_id] = req.response_queue

        # Get generator and stream batch
        generator = self._get_or_create_generator()

        try:
            async for chunk in generator.stream_batch(batch_requests):
                self._total_tokens += 1

                # Dispatch to appropriate queue
                queue = self._active_requests.get(chunk.request_id)
                if queue is not None:
                    try:
                        await queue.put(chunk)
                    except Exception as e:
                        logger.warning(
                            f"Failed to dispatch chunk to {chunk.request_id}: {e}"
                        )

                # Clean up completed requests
                if chunk.finish_reason is not None:
                    async with self._request_lock:
                        self._active_requests.pop(chunk.request_id, None)

        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            # Signal error to all waiting requests
            async with self._request_lock:
                for req in batch:
                    queue = self._active_requests.pop(req.request_id, None)
                    if queue is not None:
                        await queue.put(Exception(str(e)))

    async def stream_request(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int | None = None,
        sampler_config: dict[str, Any] | None = None,
        template_kwargs: dict[str, Any] | None = None,
    ) -> AsyncGenerator[BatchStreamChunk, None]:
        """Stream tokens for a single request.

        This is the main interface for processing requests. The request will
        be automatically batched with other concurrent requests for improved
        throughput.

        Args:
            messages: Chat messages in standard format
            max_tokens: Maximum tokens to generate
            sampler_config: Optional sampler configuration
            template_kwargs: Optional template parameters

        Yields:
            BatchStreamChunk for each generated token

        Example:
            ```python
            async for chunk in coordinator.stream_request(
                messages=[{"role": "user", "content": "Hello"}], max_tokens=100
            ):
                print(chunk.text, end="", flush=True)
                if chunk.finish_reason:
                    print(f"\\nDone: {chunk.finish_reason}")
            ```
        """
        request_id = f"batch_{uuid.uuid4().hex[:12]}"
        response_queue: asyncio.Queue = asyncio.Queue()

        # Create pending request
        pending = PendingRequest(
            request_id=request_id,
            messages=messages,
            max_tokens=max_tokens,
            sampler_config=sampler_config,
            template_kwargs=template_kwargs,
            response_queue=response_queue,
            created_at=time.time(),
        )

        # Add to pending queue
        async with self._request_lock:
            self._pending_requests[request_id] = pending
            self._total_requests += 1

        # Ensure worker is running
        await self._ensure_worker_running()

        # Signal new request
        self._new_request_event.set()

        # Stream results
        try:
            while True:
                try:
                    item = await asyncio.wait_for(response_queue.get(), timeout=300.0)
                except TimeoutError:
                    logger.warning(
                        f"Request {request_id} timed out waiting for response"
                    )
                    break

                # Check for error
                if isinstance(item, Exception):
                    raise item

                yield item

                # Check if done
                if item.finish_reason is not None:
                    break

        except asyncio.CancelledError:
            # Client cancelled - remove from pending/active
            await self.cancel_request(request_id)
            raise
        finally:
            # Cleanup
            async with self._request_lock:
                self._pending_requests.pop(request_id, None)
                self._active_requests.pop(request_id, None)

    async def cancel_request(self, request_id: str) -> bool:
        """Cancel a specific request.

        Args:
            request_id: The request ID to cancel

        Returns:
            True if request was found and cancelled
        """
        cancelled = False

        async with self._request_lock:
            # Remove from pending
            if request_id in self._pending_requests:
                self._pending_requests.pop(request_id)
                cancelled = True
                logger.debug(f"Cancelled pending request: {request_id}")

            # Remove from active
            if request_id in self._active_requests:
                self._active_requests.pop(request_id)
                cancelled = True
                logger.debug(f"Cancelled active request: {request_id}")

        # Also cancel in generator
        if self._generator is not None:
            self._generator.cancel(request_id)

        return cancelled

    async def shutdown(self) -> None:
        """Shutdown the coordinator and release resources."""
        logger.info("Shutting down batch coordinator...")

        # Signal shutdown
        self._shutdown_event.set()
        self._new_request_event.set()  # Wake up worker

        # Wait for worker to stop
        if self._worker_task is not None:
            try:
                await asyncio.wait_for(self._worker_task, timeout=5.0)
            except TimeoutError:
                self._worker_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._worker_task

        # Close generator
        with self._generator_lock:
            if self._generator is not None:
                self._generator.close()
                self._generator = None

        logger.info("Batch coordinator shutdown complete")

    def stats(self) -> dict[str, Any]:
        """Get coordinator statistics.

        Returns:
            Dictionary with coordinator stats
        """
        generator_stats = {}
        if self._generator is not None:
            generator_stats = self._generator.stats_dict()

        return {
            "coordinator": {
                "total_requests": self._total_requests,
                "total_batches": self._total_batches,
                "total_tokens": self._total_tokens,
                "pending_requests": len(self._pending_requests),
                "active_requests": len(self._active_requests),
                "batch_window_ms": self._batch_window_ms,
                "max_batch_size": self._max_batch_size,
            },
            "generator": generator_stats,
        }

    @property
    def is_active(self) -> bool:
        """Check if coordinator has active work."""
        return bool(self._pending_requests or self._active_requests)

    @property
    def has_loaded_generator(self) -> bool:
        """Check whether the coordinator currently owns a batch generator."""
        return self._generator is not None


# Global coordinator cache
_coordinators: dict[str, BatchRequestCoordinator] = {}
_coordinator_lock = threading.Lock()


def get_batch_coordinator(
    model_id: str,
    adapter_path: str | None = None,
    completion_batch_size: int = 32,
    prefill_batch_size: int = 8,
    prefill_step_size: int = 2048,
    batch_window_ms: int = 50,
    max_batch_size: int = 10,
) -> BatchRequestCoordinator:
    """Get or create a BatchRequestCoordinator for the given model.

    This function provides a singleton coordinator per model configuration.
    Multiple calls with the same parameters return the same coordinator.

    Args:
        model_id: Model name/path
        adapter_path: Optional LoRA adapter path
        completion_batch_size: Sequences per batch step
        prefill_batch_size: Sequences to prefill together
        prefill_step_size: Tokens per prefill step
        batch_window_ms: Time window to collect requests
        max_batch_size: Maximum requests per batch

    Returns:
        BatchRequestCoordinator instance
    """
    cache_key = f"{model_id}:{adapter_path or ''}"

    with _coordinator_lock:
        if cache_key not in _coordinators:
            _coordinators[cache_key] = BatchRequestCoordinator(
                model_id=model_id,
                adapter_path=adapter_path,
                completion_batch_size=completion_batch_size,
                prefill_batch_size=prefill_batch_size,
                prefill_step_size=prefill_step_size,
                batch_window_ms=batch_window_ms,
                max_batch_size=max_batch_size,
            )
            logger.info(f"Created batch coordinator for: {cache_key}")

        return _coordinators[cache_key]


async def shutdown_all_coordinators() -> None:
    """Shutdown all active coordinators.

    Call this during application shutdown to cleanly release resources.
    """
    with _coordinator_lock:
        coords = list(_coordinators.values())
        _coordinators.clear()

    for coord in coords:
        await coord.shutdown()

    logger.info("All batch coordinators shutdown")


async def shutdown_batch_coordinator(model_id: str) -> int:
    """Shutdown coordinator instances for a specific model ID."""
    with _coordinator_lock:
        keys_to_remove = [
            key for key, coord in _coordinators.items() if coord.model_id == model_id
        ]
        coords = [_coordinators.pop(key) for key in keys_to_remove]

    for coord in coords:
        await coord.shutdown()

    if coords:
        logger.info(
            "Shutdown %s batch coordinator(s) for model=%s",
            len(coords),
            model_id,
        )

    return len(coords)


def get_loaded_batch_models() -> list[str]:
    """Return model IDs with an instantiated batch generator."""
    with _coordinator_lock:
        loaded = [
            coord.model_id
            for coord in _coordinators.values()
            if coord.has_loaded_generator
        ]

    return list(dict.fromkeys(loaded))
