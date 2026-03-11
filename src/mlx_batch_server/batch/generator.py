"""Batch Chat Generator - Wrapper for mlx-lm BatchGenerator.

This module provides a wrapper around mlx-lm's BatchGenerator for efficient
parallel inference with streaming support. It enables handling multiple
concurrent chat requests with a single model instance.

Key features:
- Dynamic batching: insert/remove requests on the fly
- Streaming support: yield tokens as they're generated
- Request cancellation: cancel individual requests
- Statistics tracking: prompt/generation tokens, throughput

Target metrics:
- 10+ concurrent streaming requests
- 500+ tok/s total throughput for 70B model
- <500ms time-to-first-token per user
- <150MB overhead per concurrent request

Vibecrafted with AI Agents by VetCoders (c)2026 VetCoders
Co-Authored-By: [Maciej](void@div0.space) & [Klaudiusz](the1st@whoai.am)
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mlx_lm.generate import BatchGenerator
from mlx_lm.sample_utils import make_sampler

from ..chat.mlx.model_types import MLXModel
from ..utils.logger import logger
from ..utils.model_limits import extract_context_length, resolve_max_tokens

if TYPE_CHECKING:
    from ..chat.mlx.chat_generator import ChatGenerator

__all__ = [
    "BatchChatGenerator",
    "BatchGenerationStats",
    "BatchRequest",
    "BatchStreamChunk",
]


@dataclass
class BatchStreamChunk:
    """A single token chunk from batch generation.

    Attributes:
        request_id: The unique identifier for the request
        token: The generated token ID
        text: The decoded text for this token
        finish_reason: None if still generating, "stop" or "length" when done
        logprobs: Optional log probabilities for the token
    """

    request_id: str
    token: int
    text: str
    finish_reason: str | None = None
    logprobs: dict[str, Any] | None = None


@dataclass
class BatchRequest:
    """A request to be processed in the batch.

    Attributes:
        id: Unique identifier for this request
        messages: Chat messages in standard format
        max_tokens: Maximum tokens to generate
        tools: Optional tools for function calling
        sampler_config: Optional sampler configuration
        template_kwargs: Optional template parameters
    """

    id: str
    messages: list[dict[str, Any]]
    max_tokens: int | None = None
    tools: list[dict[str, Any]] | None = None
    sampler_config: dict[str, Any] | None = None
    template_kwargs: dict[str, Any] | None = None


@dataclass
class BatchGenerationStats:
    """Statistics for batch generation.

    Attributes:
        prompt_tokens: Total prompt tokens processed
        generation_tokens: Total tokens generated
        prompt_tps: Prompt tokens per second
        generation_tps: Generation tokens per second
        peak_memory_gb: Peak memory usage in GB
        active_requests: Number of currently active requests
        completed_requests: Number of completed requests
    """

    prompt_tokens: int = 0
    generation_tokens: int = 0
    prompt_tps: float = 0.0
    generation_tps: float = 0.0
    peak_memory_gb: float = 0.0
    active_requests: int = 0
    completed_requests: int = 0


class BatchChatGenerator:
    """Wrapper around mlx-lm BatchGenerator for chat completions.

    This class provides a high-level API for batch inference with streaming
    support. It handles prompt formatting, token decoding, and request
    lifecycle management.

    Example usage:
        ```python
        # Create generator
        gen = BatchChatGenerator.create("mlx-community/Qwen3-0.6B-4bit")

        # Stream multiple requests
        requests = [
            BatchRequest(id="req1", messages=[{"role": "user", "content": "Hi"}]),
            BatchRequest(id="req2", messages=[{"role": "user", "content": "Hello"}]),
        ]

        async for chunk in gen.stream_batch(requests):
            print(f"{chunk.request_id}: {chunk.text}")
            if chunk.finish_reason:
                print(f"{chunk.request_id} finished: {chunk.finish_reason}")
        ```
    """

    def __init__(
        self,
        model: MLXModel,
        completion_batch_size: int = 32,
        prefill_batch_size: int = 8,
        prefill_step_size: int = 2048,
    ):
        """Initialize BatchChatGenerator.

        Args:
            model: MLXModel instance with loaded model and tokenizer
            completion_batch_size: Number of sequences to process per step
            prefill_batch_size: Number of sequences to prefill together
            prefill_step_size: Number of tokens to prefill per step
        """
        self.model = model
        self.tokenizer = model.tokenizer
        self.chat_template = model.chat_template
        self._completion_batch_size = completion_batch_size
        self._prefill_batch_size = prefill_batch_size
        self._prefill_step_size = prefill_step_size
        self._context_length = getattr(
            model, "context_length", None
        ) or extract_context_length(
            model.config,
            model.tokenizer,
        )

        # Generator instance - created lazily
        self._generator: BatchGenerator | None = None
        self._generator_lock = threading.Lock()

        # Request tracking
        self._uid_to_request: dict[int, str] = {}
        self._request_to_uid: dict[str, int] = {}
        self._active_requests: set[str] = set()

        # Delta decoding state (for proper space handling with SentencePiece)
        self._request_tokens: dict[str, list[int]] = {}
        self._request_text: dict[str, str] = {}

        # Statistics
        self._stats = BatchGenerationStats()
        self._start_time: float | None = None

    @classmethod
    def create(
        cls,
        model_id: str,
        adapter_path: str | None = None,
        completion_batch_size: int = 32,
        prefill_batch_size: int = 8,
        prefill_step_size: int = 2048,
    ) -> BatchChatGenerator:
        """Factory method to create BatchChatGenerator.

        Args:
            model_id: Model name/path (HuggingFace model ID or local path)
            adapter_path: Optional path to LoRA adapter
            completion_batch_size: Number of sequences to process per step
            prefill_batch_size: Number of sequences to prefill together
            prefill_step_size: Number of tokens to prefill per step

        Returns:
            BatchChatGenerator instance ready for use
        """
        model = MLXModel.load(model_id=model_id, adapter_path=adapter_path)
        return cls(
            model=model,
            completion_batch_size=completion_batch_size,
            prefill_batch_size=prefill_batch_size,
            prefill_step_size=prefill_step_size,
        )

    @classmethod
    def from_chat_generator(
        cls,
        chat_generator: ChatGenerator,
        completion_batch_size: int = 32,
        prefill_batch_size: int = 8,
        prefill_step_size: int = 2048,
    ) -> BatchChatGenerator:
        """Create BatchChatGenerator from existing ChatGenerator.

        This allows sharing the underlying model between sequential and batch
        generation, which is useful for memory efficiency.

        Args:
            chat_generator: Existing ChatGenerator instance
            completion_batch_size: Number of sequences to process per step
            prefill_batch_size: Number of sequences to prefill together
            prefill_step_size: Number of tokens to prefill per step

        Returns:
            BatchChatGenerator sharing the same model
        """
        return cls(
            model=chat_generator.model,
            completion_batch_size=completion_batch_size,
            prefill_batch_size=prefill_batch_size,
            prefill_step_size=prefill_step_size,
        )

    def _get_or_create_generator(
        self,
        max_tokens: int,
    ) -> BatchGenerator:
        """Get existing or create new BatchGenerator.

        Note: generator-level max_tokens is only a default. Per-request limits
        and samplers are passed to ``BatchGenerator.insert()`` so concurrent
        requests can keep independent decoding settings.

        Args:
            max_tokens: Default max tokens for new requests

        Returns:
            BatchGenerator instance
        """
        with self._generator_lock:
            # Get stop tokens from tokenizer
            stop_tokens = set()
            if hasattr(self.tokenizer, "eos_token_ids"):
                stop_tokens = self.tokenizer.eos_token_ids
            elif hasattr(self.tokenizer, "_eos_token_ids"):
                stop_tokens = self.tokenizer._eos_token_ids

            if self._generator is None:
                self._generator = BatchGenerator(
                    model=self.model.text_model,
                    max_tokens=max_tokens,
                    stop_tokens=stop_tokens,
                    completion_batch_size=self._completion_batch_size,
                    prefill_batch_size=self._prefill_batch_size,
                    prefill_step_size=self._prefill_step_size,
                )
                self._start_time = time.perf_counter()
                logger.info(
                    f"BatchGenerator created: batch={self._completion_batch_size}, "
                    f"prefill={self._prefill_batch_size}"
                )

            return self._generator

    def _format_prompt(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        template_kwargs: dict[str, Any] | None = None,
    ) -> str:
        """Format messages into a prompt string.

        Args:
            messages: Chat messages in standard format
            tools: Optional tools for function calling
            template_kwargs: Optional template parameters

        Returns:
            Formatted prompt string
        """
        kwargs = template_kwargs or {}

        # Map enable_thinking to enable_thinking_parse for new parameter name
        if "enable_thinking" in kwargs:
            kwargs["enable_thinking_parse"] = kwargs.pop("enable_thinking")

        request_template = (
            self.model.new_chat_template()
            if hasattr(self.model, "new_chat_template")
            else self.chat_template
        )

        return request_template.apply_chat_template(
            messages=messages,
            tools=tools,
            **kwargs,
        )

    def _prepare_batch_requests(
        self, requests: list[BatchRequest]
    ) -> tuple[list[list[int]], list[int], list[Any | None]]:
        """Prepare and tokenize batch requests.

        Args:
            requests: List of BatchRequest objects

        Returns:
            Tuple of (prompts, max_tokens_list, samplers)
        """
        prompts = []
        max_tokens_list = []
        samplers = []

        for req in requests:
            prompt_str = self._format_prompt(
                messages=req.messages,
                tools=req.tools,
                template_kwargs=req.template_kwargs,
            )
            tokenized = self.tokenizer.encode(prompt_str)
            prompts.append(tokenized)
            resolved_max_tokens = resolve_max_tokens(
                requested=req.max_tokens,
                context_length=self._context_length,
                prompt_tokens=len(tokenized),
                context_label=self.model.model_id,
            )
            max_tokens_list.append(resolved_max_tokens)
            samplers.append(
                make_sampler(**req.sampler_config) if req.sampler_config else None
            )

            logger.debug(
                f"Request {req.id}: prompt_length={len(tokenized)}, "
                f"max_tokens={resolved_max_tokens}"
            )

        return prompts, max_tokens_list, samplers

    def _process_response(self, response) -> BatchStreamChunk | None:
        """Process a single response from BatchGenerator.

        Uses delta decoding to preserve spaces for SentencePiece tokenizers.
        Instead of decoding each token individually (which loses spaces),
        we decode all tokens together and emit the difference.

        Args:
            response: Response from BatchGenerator.next()

        Returns:
            BatchStreamChunk or None if unknown UID
        """
        request_id = self._uid_to_request.get(response.uid)
        if request_id is None:
            logger.warning(f"Unknown UID in batch response: {response.uid}")
            return None

        # Delta decoding: decode all tokens together to preserve spaces
        # Initialize tracking state for new requests
        if request_id not in self._request_tokens:
            self._request_tokens[request_id] = []
            self._request_text[request_id] = ""

        # Append token and decode full sequence
        self._request_tokens[request_id].append(response.token)
        full_text = self.tokenizer.decode(self._request_tokens[request_id])

        # Calculate delta (new text since last decode)
        prev_text = self._request_text[request_id]
        token_text = full_text[len(prev_text) :]
        self._request_text[request_id] = full_text

        # Build logprobs if available
        logprobs = None
        if response.logprobs is not None:
            logprobs = {
                "token": response.token,
                "logprob": float(response.logprobs[response.token]),
            }

        # Track completion and cleanup delta state
        if response.finish_reason is not None:
            self._uid_to_request.pop(response.uid, None)
            self._request_to_uid.pop(request_id, None)
            self._active_requests.discard(request_id)
            # Clean up delta decoding state
            self._request_tokens.pop(request_id, None)
            self._request_text.pop(request_id, None)
            self._stats.completed_requests += 1
            self._stats.active_requests = len(self._active_requests)
            logger.debug(f"Request {request_id} completed: {response.finish_reason}")

        return BatchStreamChunk(
            request_id=request_id,
            token=response.token,
            text=token_text,
            finish_reason=response.finish_reason,
            logprobs=logprobs,
        )

    async def stream_batch(
        self,
        requests: list[BatchRequest],
    ) -> AsyncGenerator[BatchStreamChunk, None]:
        """Stream tokens for multiple requests concurrently.

        This method processes multiple requests in parallel, yielding tokens
        as they are generated. Requests can have different max_tokens limits.

        Args:
            requests: List of BatchRequest objects to process

        Yields:
            BatchStreamChunk for each generated token

        Example:
            ```python
            requests = [
                BatchRequest(id="req1", messages=[...], max_tokens=100),
                BatchRequest(id="req2", messages=[...], max_tokens=200),
            ]

            async for chunk in gen.stream_batch(requests):
                if chunk.request_id == "req1":
                    handle_req1_token(chunk)
                elif chunk.request_id == "req2":
                    handle_req2_token(chunk)

                if chunk.finish_reason:
                    print(f"Request {chunk.request_id} completed")
            ```
        """
        if not requests:
            return

        # Prepare and insert requests
        prompts, max_tokens_list, samplers = self._prepare_batch_requests(requests)

        # The generator-level max_tokens is only the default. Per-request limits
        # and samplers are applied at insert time.
        gen = self._get_or_create_generator(max_tokens=max(max_tokens_list))
        uids = gen.insert(prompts, max_tokens=max_tokens_list, samplers=samplers)

        # Map UIDs to request IDs
        for uid, req in zip(uids, requests, strict=True):
            self._uid_to_request[uid] = req.id
            self._request_to_uid[req.id] = uid
            self._active_requests.add(req.id)

        self._stats.active_requests = len(self._active_requests)
        logger.info(
            f"Inserted {len(requests)} requests, active={self._stats.active_requests}"
        )

        # Stream tokens
        try:
            while self._active_requests:
                responses = gen.next()
                if not responses:
                    break

                for response in responses:
                    chunk = self._process_response(response)
                    if chunk is not None:
                        yield chunk

                await asyncio.sleep(0)

        except AttributeError as e:
            # MambaCache doesn't have extract() method required by batch generation
            if "extract" in str(e):
                model_name = getattr(self.model, "model_id", "unknown")
                error_msg = (
                    f"Model '{model_name}' uses MambaCache which doesn't support "
                    f"batch mode. Hybrid Mamba-Transformer models (like Nemotron) "
                    f"require single-request mode. Original error: {e}"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e
            logger.error(f"Error in batch streaming: {e}")
            raise
        except Exception as e:
            logger.error(f"Error in batch streaming: {e}")
            raise
        finally:
            # Update stats
            if self._generator:
                mlx_stats = self._generator.stats()
                self._stats.prompt_tokens = mlx_stats.prompt_tokens
                self._stats.generation_tokens = mlx_stats.generation_tokens
                self._stats.prompt_tps = mlx_stats.prompt_tps
                self._stats.generation_tps = mlx_stats.generation_tps
                self._stats.peak_memory_gb = mlx_stats.peak_memory

    def cancel(self, request_id: str) -> bool:
        """Cancel a specific request.

        Removes the request from the batch, stopping generation for it.

        Args:
            request_id: The unique identifier for the request to cancel

        Returns:
            True if request was found and cancelled, False otherwise
        """
        uid = self._request_to_uid.get(request_id)
        if uid is None:
            logger.warning(f"Cannot cancel unknown request: {request_id}")
            return False

        if self._generator is None:
            logger.warning(f"Cannot cancel request {request_id}: no active generator")
            return False

        try:
            self._generator.remove([uid])
            self._uid_to_request.pop(uid, None)
            self._request_to_uid.pop(request_id, None)
            self._active_requests.discard(request_id)
            self._stats.active_requests = len(self._active_requests)

            logger.info(f"Cancelled request: {request_id}")
            return True
        except Exception as e:
            logger.error(f"Error cancelling request {request_id}: {e}")
            return False

    def cancel_all(self) -> int:
        """Cancel all active requests.

        Returns:
            Number of requests cancelled
        """
        cancelled = 0
        for request_id in list(self._active_requests):
            if self.cancel(request_id):
                cancelled += 1
        return cancelled

    def close(self) -> None:
        """Close the generator and release resources.

        This should be called when the BatchChatGenerator is no longer needed.
        After calling close(), the generator can be reused - a new BatchGenerator
        will be created on the next stream_batch call.
        """
        with self._generator_lock:
            if self._generator is not None:
                self._generator.close()
                self._generator = None

            # Clear all tracking
            self._uid_to_request.clear()
            self._request_to_uid.clear()
            self._active_requests.clear()
            self._start_time = None

            logger.info("BatchChatGenerator closed")

    def stats(self) -> BatchGenerationStats:
        """Get current generation statistics.

        Returns:
            BatchGenerationStats with current metrics
        """
        # Update stats from underlying generator if available
        if self._generator is not None:
            try:
                mlx_stats = self._generator.stats()
                self._stats.prompt_tokens = mlx_stats.prompt_tokens
                self._stats.generation_tokens = mlx_stats.generation_tokens
                self._stats.prompt_tps = mlx_stats.prompt_tps
                self._stats.generation_tps = mlx_stats.generation_tps
                self._stats.peak_memory_gb = mlx_stats.peak_memory
            except Exception as e:
                logger.debug(f"Could not get generator stats: {e}")

        return self._stats

    def stats_dict(self) -> dict[str, Any]:
        """Get statistics as dictionary.

        Returns:
            Dictionary with generation stats
        """
        stats = self.stats()
        return {
            "prompt_tokens": stats.prompt_tokens,
            "generation_tokens": stats.generation_tokens,
            "prompt_tps": stats.prompt_tps,
            "generation_tps": stats.generation_tps,
            "peak_memory_gb": stats.peak_memory_gb,
            "active_requests": stats.active_requests,
            "completed_requests": stats.completed_requests,
        }
