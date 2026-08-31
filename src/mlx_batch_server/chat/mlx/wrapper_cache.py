"""Centralized MLX Runtime Cache.

This cache is the single residency owner for MLX runtimes. Multimodal-capable
models are loaded once through the normal ChatGenerator path; text batching uses
their ``language_model`` tower and media requests reuse the same resident
runtime/processor in single-flight mode.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from contextlib import contextmanager, suppress
from dataclasses import dataclass

from ...batch.coordinator import retire_idle_batch_runtime
from ...core.config import get_settings
from ...utils.logger import logger
from ...utils.memory import force_mlx_cleanup
from .chat_generator import ChatGenerator
from .runtime_aliases import (
    normalize_runtime_model_id,
    resolve_runtime_target,
)
from .runtime_attachments import (
    attach_runtime_surface,
    clear_runtime_surface_attachments,
    get_runtime_surface_attachments,
    list_runtime_surface_attachments_by_runtime,
    release_runtime_surface,
)
from .runtime_leases import (
    acquire_runtime_lease,
    active_runtime_lease_count,
    list_runtime_leases,
    release_runtime_lease,
    runtime_retirement_guard,
)


def normalize_model_id(model_id: str) -> str:
    """Normalize model IDs for stable cache keys."""
    return normalize_runtime_model_id(model_id)


def normalize_runtime_key(
    model_id: str,
    adapter_path: str | None = None,
    draft_model_id: str | None = None,
) -> WrapperCacheKey:
    """Normalize all runtime-key fields so residency checks are stable."""
    target = resolve_runtime_target(
        model_id,
        adapter_path=adapter_path,
        draft_model_id=draft_model_id,
    )
    return WrapperCacheKey(
        model_id=target.model_id,
        adapter_path=target.adapter_path,
        draft_model_id=target.draft_model_id,
    )


def serialize_runtime_key(key: WrapperCacheKey) -> dict[str, str | None]:
    """Convert one runtime key into a structured operator-facing payload."""
    return {
        "model_id": key.model_id,
        "adapter_path": key.adapter_path,
        "draft_model_id": key.draft_model_id,
    }


@dataclass(frozen=True)
class WrapperCacheKey:
    """Cache key for ChatGenerator instances.

    Uses all parameters that affect model loading to ensure proper cache invalidation
    when any of these parameters change.
    """

    model_id: str
    adapter_path: str | None = None
    draft_model_id: str | None = None


class MLXWrapperCache:
    """Thread-safe LRU cache for MLX runtimes.

    One cached ChatGenerator owns one loaded model runtime. If that runtime is
    multimodal, the same wrapper also owns the processor stack used by
    ``mlx-vlm`` requests. Text batching and multimodal single-flight therefore
    share a single model residency.
    """

    def __init__(
        self,
        max_size: int = 1,
        ttl_seconds: int = 600,
        cleanup_interval: int = 5,
        pinned_models: list[str] | None = None,
    ):
        """Initialize cache with LRU eviction, TTL support, and pinned models.

        Args:
            max_size: Maximum number of NON-PINNED models to cache (default: 1)
            ttl_seconds: TTL for non-pinned models (default: 600s = 10 min)
            cleanup_interval: Background cleanup interval in seconds (default: 5)
            pinned_models: List of model IDs that are NEVER evicted (always available)
        """
        self._cache: OrderedDict[WrapperCacheKey, ChatGenerator] = OrderedDict()
        self._access_times: dict[WrapperCacheKey, float] = {}
        self._vlm_execution_locks: dict[WrapperCacheKey, threading.Lock] = {}

        self._lock = threading.Lock()
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._cleanup_interval = cleanup_interval
        self._pinned_models: set[str] = {
            normalize_model_id(model_id) for model_id in (pinned_models or [])
        }
        self._stop_event = threading.Event()
        self._cleanup_thread = None

        logger.info(
            f"MLXWrapperCache initialized: max_size={max_size}, ttl={ttl_seconds}s, "
            f"pinned={list(self._pinned_models)}"
        )

        # Start background cleanup thread if TTL is enabled
        if self._ttl_seconds > 0:
            self._cleanup_thread = threading.Thread(
                target=self._periodic_cleanup, daemon=True
            )
            self._cleanup_thread.start()

    def _is_pinned(self, key: WrapperCacheKey) -> bool:
        """Check if a model is statically pinned (should never be evicted)."""
        return key.model_id in self._pinned_models

    def _is_eviction_protected(self, key: WrapperCacheKey) -> bool:
        """Return True when runtime residency is protected from TTL/LRU eviction.

        Static pinned models always stay resident. Passive product-surface
        attachments describe ownership but do not defeat idle TTL. Only queued or
        active inference jobs hold a runtime lease that blocks automatic eviction.
        """
        return self._is_pinned(key) or active_runtime_lease_count(
            key.model_id,
            adapter_path=key.adapter_path,
            draft_model_id=key.draft_model_id,
        )

    def _should_cache_runtime(self, key: WrapperCacheKey) -> bool:
        """Return True when one runtime must stay resident after creation."""
        return (
            self._is_pinned(key)
            or self._max_size > 0
            or bool(
                get_runtime_surface_attachments(
                    key.model_id,
                    adapter_path=key.adapter_path,
                    draft_model_id=key.draft_model_id,
                )
            )
        )

    def _release_memory(
        self, wrapper: ChatGenerator | None, key: WrapperCacheKey
    ) -> None:
        """Release MLX memory after text cache eviction or unload."""
        if key.draft_model_id is None:
            with suppress(Exception):
                retire_idle_batch_runtime(
                    key.model_id,
                    adapter_path=key.adapter_path,
                )
        if wrapper is None:
            return

        model_id = key.model_id
        try:
            with suppress(Exception):
                wrapper._prompt_cache = None
                wrapper.model = None

            del wrapper
            force_mlx_cleanup(f"wrapper:{model_id}", passes=3)
        except Exception as e:
            logger.warning(f"Error releasing memory for {model_id}: {e}")

    def _evict_expired_items(self) -> None:
        """Evict items that have exceeded their TTL.

        PINNED models are NEVER evicted regardless of TTL.
        This method should be called while holding the lock.
        """
        if self._ttl_seconds <= 0:
            return  # TTL disabled

        current_time = time.time()

        # --- Text cache ---
        expired_keys = []
        for key, access_time in self._access_times.items():
            if self._is_eviction_protected(key):
                continue
            if current_time - access_time > self._ttl_seconds:
                expired_keys.append(key)

        for key in expired_keys:
            with runtime_retirement_guard(
                key.model_id,
                adapter_path=key.adapter_path,
                draft_model_id=key.draft_model_id,
            ) as idle:
                if not idle:
                    continue
                wrapper = self._cache.pop(key, None)
                self._access_times.pop(key, None)
                logger.info(
                    f"Evicted expired model from cache (TTL={self._ttl_seconds}s): {key}"
                )
                self._release_memory(wrapper, key)

    def _evict_lru_if_needed(self) -> None:
        """Evict least recently used NON-PINNED item if cache is at capacity.

        PINNED models don't count against max_size and are NEVER evicted.
        This method should be called while holding the lock.
        """
        if self._max_size <= 0:
            return

        # Count only evictable models against max_size. Queued/active runtimes
        # are temporarily protected, just like statically pinned runtimes.
        evictable_count = sum(
            1 for key in self._cache if not self._is_eviction_protected(key)
        )

        if evictable_count >= self._max_size:
            # Find the least recently used evictable key.
            evictable_keys = [
                k for k in self._access_times if not self._is_eviction_protected(k)
            ]
            if not evictable_keys:
                return  # Only pinned/active runtimes are resident, nothing to evict

            lru_key = min(evictable_keys, key=lambda k: self._access_times[k])

            with runtime_retirement_guard(
                lru_key.model_id,
                adapter_path=lru_key.adapter_path,
                draft_model_id=lru_key.draft_model_id,
            ) as idle:
                if not idle:
                    return
                # Remove from cache and access times
                wrapper = self._cache.pop(lru_key, None)
                self._access_times.pop(lru_key, None)
                logger.info(f"Evicted LRU model from cache: {lru_key}")
                self._release_memory(wrapper, lru_key)

    def _update_access_time(self, key: WrapperCacheKey) -> None:
        """Update access time for LRU tracking.

        This method should be called while holding the lock.
        """
        self._access_times[key] = time.time()

    def _has_runtime_for_model_locked(self, model_id: str) -> bool:
        """Return True while any cache entry still owns this canonical model id."""
        return any(key.model_id == model_id for key in self._cache)

    def _periodic_cleanup(self) -> None:
        """Background thread method for periodic cleanup of expired items.

        This method runs in a daemon thread and periodically checks for expired items.
        """
        while not self._stop_event.wait(self._cleanup_interval):
            try:
                with self._lock:
                    self._evict_expired_items()
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")

    def _stop_cleanup_thread(self) -> None:
        """Stop the background cleanup thread gracefully."""
        if self._cleanup_thread is not None:
            self._stop_event.set()
            self._cleanup_thread.join(timeout=1.0)
            self._cleanup_thread = None
            logger.info("Background cleanup thread stopped")

    def get_wrapper(
        self,
        model_id: str,
        adapter_path: str | None = None,
        draft_model_id: str | None = None,
        surface: str | None = None,
    ) -> ChatGenerator:
        """Get or create ChatGenerator instance.

        Args:
            model_id: Model name/path (HuggingFace model ID or local path)
            adapter_path: Optional path to LoRA adapter
            draft_model_id: Optional draft model name/path for speculative decoding
            surface: Optional product surface requesting sticky residency

        Returns:
            Cached or newly created ChatGenerator instance

        Note:
            This method is thread-safe and will only create one wrapper instance
            per unique parameter combination, even under concurrent access.
        """
        key = normalize_runtime_key(model_id, adapter_path, draft_model_id)
        normalized_model_id = key.model_id
        attachment_state = (
            attach_runtime_surface(
                key.model_id,
                surface,
                adapter_path=key.adapter_path,
                draft_model_id=key.draft_model_id,
            )
            if surface is not None
            else None
        )

        # Double-checked locking pattern for performance
        if key in self._cache:
            with self._lock:
                # Evict expired items before checking cache
                self._evict_expired_items()

                # Check if key still exists after expiry cleanup
                if key in self._cache:
                    # Update access time for LRU and TTL
                    self._update_access_time(key)
                    logger.debug(f"Cache hit for ChatGenerator: {key}")
                    return self._cache[key]

        with self._lock:
            # Evict expired items first
            self._evict_expired_items()

            # Check again inside lock in case another thread created it
            if key in self._cache:
                self._update_access_time(key)
                logger.debug(f"Cache hit (after lock) for ChatGenerator: {key}")
                return self._cache[key]

            # Cache miss - evict LRU if needed before creating new wrapper
            self._evict_lru_if_needed()

            # Create new wrapper
            logger.info(f"Creating new ChatGenerator for: {key}")
            try:
                wrapper = ChatGenerator.create(
                    model_id=normalized_model_id,
                    adapter_path=key.adapter_path,
                    draft_model_id=key.draft_model_id,
                )

                # Pinned models must remain tracked even in max_size=0
                # deployments. Surface-retained runtimes must also enter the
                # cache so ownership/observability remains truthful; unlike
                # pinned or leased runtimes, idle TTL may retire their weights.
                if self._should_cache_runtime(key):
                    self._cache[key] = wrapper
                    self._update_access_time(key)
                    logger.info(
                        f"Cached ChatGenerator: {key} "
                        f"(size={len(self._cache)}, max_non_pinned={self._max_size}, "
                        f"pinned={self._is_pinned(key)})"
                    )
                else:
                    logger.warning(
                        f"ChatGenerator created but NOT cached (non-pinned, max_size=0): {key}"
                    )

                return wrapper
            except Exception as e:
                if attachment_state is not None and not attachment_state.was_attached:
                    release_runtime_surface(
                        key.model_id,
                        attachment_state.surface,
                        adapter_path=key.adapter_path,
                        draft_model_id=key.draft_model_id,
                    )
                logger.error(f"Failed to create ChatGenerator for {key}: {e}")
                raise

    def cleanup_expired_items(self) -> int:
        """Manually trigger cleanup of expired items.

        This can be called periodically by a background task or manually
        to clean up expired items without waiting for cache access.

        Returns:
            Number of items that were evicted
        """
        with self._lock:
            initial_size = len(self._cache)
            self._evict_expired_items()
            evicted_count = initial_size - len(self._cache)

            if evicted_count > 0:
                logger.info(f"Manual cleanup evicted {evicted_count} expired items")

            return evicted_count

    def clear_cache(self) -> None:
        """Clear all cached model instances and release MLX memory.

        This can be useful for memory management or testing purposes.
        """
        # Stop the cleanup thread first
        self._stop_cleanup_thread()

        with self._lock:
            cache_size = len(self._cache)
            wrappers_to_release = list(self._cache.items())
            self._cache.clear()
            self._access_times.clear()
            self._vlm_execution_locks.clear()
            clear_runtime_surface_attachments()
            logger.info(f"Cleared cache ({cache_size} runtime entries)")

        for key, wrapper in wrappers_to_release:
            self._release_memory(wrapper, key)

    def unload_model(
        self,
        model_id: str,
        adapter_path: str | None = None,
        draft_model_id: str | None = None,
    ) -> bool:
        """Unload a specific model or one exact runtime key from cache.

        Args:
            model_id: The model ID to unload
            adapter_path: Optional adapter path for exact runtime eviction
            draft_model_id: Optional speculative draft runtime identity

        Returns:
            True if model was found and unloaded, False otherwise
        """
        exact_key = None
        normalized_model_id = normalize_model_id(model_id)
        if adapter_path is not None or draft_model_id is not None:
            exact_key = normalize_runtime_key(
                model_id,
                adapter_path=adapter_path,
                draft_model_id=draft_model_id,
            )
            normalized_model_id = exact_key.model_id
        wrappers_to_release: list[tuple[WrapperCacheKey, ChatGenerator | None]] = []
        found = False

        with self._lock:
            if exact_key is None:
                keys_to_remove = [
                    key for key in self._cache if key.model_id == normalized_model_id
                ]
            else:
                keys_to_remove = [key for key in self._cache if key == exact_key]
            for key in keys_to_remove:
                wrapper = self._cache.pop(key, None)
                self._access_times.pop(key, None)
                wrappers_to_release.append((key, wrapper))
                clear_runtime_surface_attachments(
                    key.model_id,
                    adapter_path=key.adapter_path,
                    draft_model_id=key.draft_model_id,
                )
                logger.info(f"Unloaded model from cache: {key}")

            if exact_key is None:
                lock_keys_to_remove = [
                    lock_key
                    for lock_key in self._vlm_execution_locks
                    if lock_key.model_id == normalized_model_id
                ]
            else:
                lock_keys_to_remove = [exact_key]

            for lock_key in lock_keys_to_remove:
                self._vlm_execution_locks.pop(lock_key, None)

            if not any(key.model_id == normalized_model_id for key in self._cache):
                sibling_lock_keys = [
                    lock_key
                    for lock_key in self._vlm_execution_locks
                    if lock_key.model_id == normalized_model_id
                ]
                for lock_key in sibling_lock_keys:
                    self._vlm_execution_locks.pop(lock_key, None)
            found = bool(keys_to_remove)

        if not found:
            logger.info(f"Model {normalized_model_id} not found in cache")

        for key, wrapper in wrappers_to_release:
            self._release_memory(wrapper, key)

        return found

    def is_model_loaded(self, model_id: str) -> bool:
        """Check if a model is currently loaded in cache.

        Args:
            model_id: The model ID to check

        Returns:
            True if model is in cache, False otherwise
        """
        normalized_model_id = normalize_model_id(model_id)
        with self._lock:
            return any(key.model_id == normalized_model_id for key in self._cache)

    def is_runtime_loaded(
        self,
        model_id: str,
        adapter_path: str | None = None,
        draft_model_id: str | None = None,
    ) -> bool:
        """Check if one exact runtime key is currently resident in cache."""
        key = normalize_runtime_key(model_id, adapter_path, draft_model_id)
        with self._lock:
            return key in self._cache

    def get_loaded_models(self) -> list[str]:
        """Get list of currently loaded model IDs."""
        with self._lock:
            return list({key.model_id for key in self._cache})

    def get_runtime_keys(self) -> list[WrapperCacheKey]:
        """Return the concrete runtime cache keys currently resident."""
        with self._lock:
            return list(self._cache.keys())

    def renew_runtime_ttl(
        self,
        model_id: str,
        *,
        adapter_path: str | None = None,
        draft_model_id: str | None = None,
    ) -> bool:
        """Restart idle TTL for an exact runtime after active work finishes."""
        key = normalize_runtime_key(model_id, adapter_path, draft_model_id)
        with self._lock:
            if key not in self._cache:
                return False
            self._update_access_time(key)
            return True

    # ------------------------------------------------------------------
    # Multimodal helpers on top of the shared runtime cache
    # ------------------------------------------------------------------

    def _loaded_multimodal_model_ids(self) -> list[str]:
        return sorted(
            {
                key.model_id
                for key, wrapper in self._cache.items()
                if getattr(wrapper.model, "supports_multimodal", False)
            }
        )

    def get_vlm_backend(
        self,
        model_id: str,
        *,
        adapter_path: str | None = None,
        draft_model_id: str | None = None,
        surface: str | None = None,
    ):
        """Return the shared multimodal backend from the cached wrapper runtime.

        Args:
            model_id: Model name/path
            adapter_path: Optional adapter path tied to the resident runtime alias
            draft_model_id: Optional speculative draft runtime identity
            surface: Optional product surface requesting sticky residency

        Returns:
            ``(model, processor)`` tuple from the resident VLM runtime
        """
        wrapper = self.get_wrapper(
            model_id=model_id,
            adapter_path=adapter_path,
            draft_model_id=draft_model_id,
            surface=surface,
        )
        if (
            not getattr(wrapper.model, "supports_multimodal", False)
            or wrapper.model.processor is None
        ):
            raise RuntimeError(
                f"Model {model_id} is not loaded as a multimodal MLX-VLM runtime"
            )
        return wrapper.model.model, wrapper.model.processor

    def get_loaded_vlm_models(self) -> list[str]:
        """Return model IDs currently resident as multimodal-capable runtimes."""
        with self._lock:
            return self._loaded_multimodal_model_ids()

    def unload_vlm_model(
        self,
        model_id: str | None = None,
        *,
        adapter_path: str | None = None,
        draft_model_id: str | None = None,
    ) -> list[str]:
        """Unload one or all multimodal-capable runtimes.

        Args:
            model_id: Specific model to unload, or ``None`` to clear all.
            adapter_path: Optional adapter path for exact runtime eviction
            draft_model_id: Optional speculative draft runtime identity

        Returns:
            List of model IDs that were actually evicted.
        """
        if model_id is not None:
            runtime_key = normalize_runtime_key(
                model_id,
                adapter_path=adapter_path,
                draft_model_id=draft_model_id,
            )
            normalized = runtime_key.model_id
            with self._lock:
                is_multimodal = any(
                    (
                        key == runtime_key
                        if adapter_path is not None or draft_model_id is not None
                        else key.model_id == normalized
                    )
                    and getattr(wrapper.model, "supports_multimodal", False)
                    for key, wrapper in self._cache.items()
                )
            if not is_multimodal:
                return []
            return (
                [normalized]
                if self.unload_model(
                    normalized,
                    adapter_path=runtime_key.adapter_path,
                    draft_model_id=runtime_key.draft_model_id,
                )
                else []
            )

        with self._lock:
            multimodal_ids = self._loaded_multimodal_model_ids()
        unloaded: list[str] = []
        for mid in multimodal_ids:
            if self.unload_model(mid):
                unloaded.append(mid)
        return unloaded

    @contextmanager
    def vlm_execution(
        self,
        model_id: str,
        *,
        adapter_path: str | None = None,
        draft_model_id: str | None = None,
    ):
        """Serialize multimodal generation on one exact resident VLM runtime."""
        runtime_key = normalize_runtime_key(
            model_id,
            adapter_path=adapter_path,
            draft_model_id=draft_model_id,
        )
        with self._lock:
            lock = self._vlm_execution_locks.setdefault(runtime_key, threading.Lock())
        acquire_runtime_lease(
            runtime_key.model_id,
            adapter_path=runtime_key.adapter_path,
            draft_model_id=runtime_key.draft_model_id,
        )
        lock.acquire()
        try:
            yield
        finally:
            self.renew_runtime_ttl(
                runtime_key.model_id,
                adapter_path=runtime_key.adapter_path,
                draft_model_id=runtime_key.draft_model_id,
            )
            release_runtime_lease(
                runtime_key.model_id,
                adapter_path=runtime_key.adapter_path,
                draft_model_id=runtime_key.draft_model_id,
            )
            lock.release()

    def get_cache_info(self) -> dict[str, any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics including LRU and TTL information
        """
        with self._lock:
            # Clean up expired items first to get accurate stats
            self._evict_expired_items()

            current_time = time.time()
            sorted_keys = sorted(
                self._access_times.items(), key=lambda x: x[1], reverse=True
            )

            # Calculate TTL remaining for each item
            ttl_info = []
            if self._ttl_seconds > 0:
                for key, access_time in sorted_keys:
                    remaining_ttl = self._ttl_seconds - (current_time - access_time)
                    ttl_info.append(
                        {
                            "key": str(key),
                            "remaining_ttl_seconds": max(0, remaining_ttl),
                            "expires_at": access_time + self._ttl_seconds,
                        }
                    )

            return {
                "cache_size": len(self._cache),
                "max_size": self._max_size,
                "ttl_seconds": self._ttl_seconds,
                "cached_keys": [str(key) for key in self._cache],
                "runtime_keys": [serialize_runtime_key(key) for key in self._cache],
                "vlm_cached_keys": self._loaded_multimodal_model_ids(),
                "surface_runtime_attachments": list_runtime_surface_attachments_by_runtime(),
                "active_runtime_leases": list_runtime_leases(),
                "lru_order": [str(key) for key, _ in sorted_keys],  # Most recent first
                "ttl_info": ttl_info,
            }

    def set_max_size(self, max_size: int) -> None:
        """Update the maximum cache size.

        Args:
            max_size: New maximum cache size

        Note:
            If the new size is smaller than current cache size,
            LRU items will be evicted immediately.
        """
        with self._lock:
            self._max_size = max_size

            # Evict items if current cache exceeds new limit
            while len(self._cache) > self._max_size:
                self._evict_lru_if_needed()

            logger.info(
                f"Updated cache max_size to {max_size}, current size: {len(self._cache)}"
            )

    def __del__(self) -> None:
        """Destructor to ensure cleanup thread is stopped."""
        self._stop_cleanup_thread()


# Global cache instance - shared across all API endpoints
# Reads configuration from environment via Settings
def _create_wrapper_cache() -> MLXWrapperCache:
    """Create cache instance with settings from environment."""
    settings = get_settings()
    return MLXWrapperCache(
        max_size=settings.model_cache_max_size,
        ttl_seconds=settings.model_cache_ttl,
        pinned_models=settings.get_pinned_models(),
    )


wrapper_cache = _create_wrapper_cache()
