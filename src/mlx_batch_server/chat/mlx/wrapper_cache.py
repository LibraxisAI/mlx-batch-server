"""Centralized ChatGenerator Cache

This module provides a unified caching system for ChatGenerator instances
to avoid expensive model reloading when the same model configuration is used
across different API endpoints.

Supports pinned models that are NEVER evicted (always available).
"""

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

from ...core.config import get_settings
from ...utils.logger import logger
from .chat_generator import ChatGenerator


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
    """Thread-safe LRU cache for ChatGenerator instances with TTL support.

    This cache ensures that expensive model loading only happens once per unique
    combination of (model_id, adapter_path, draft_model_id). All API endpoints
    (OpenAI, Anthropic) can share the same cached wrapper instance.

    Uses LRU (Least Recently Used) eviction policy and TTL (Time To Live)
    to manage memory usage automatically.
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
        self._lock = threading.Lock()
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._cleanup_interval = cleanup_interval
        self._pinned_models: set[str] = set(pinned_models or [])
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
        """Check if a model is pinned (should never be evicted)."""
        return key.model_id in self._pinned_models

    def _evict_expired_items(self) -> None:
        """Evict items that have exceeded their TTL.

        PINNED models are NEVER evicted regardless of TTL.
        This method should be called while holding the lock.
        """
        if self._ttl_seconds <= 0:
            return  # TTL disabled

        current_time = time.time()
        expired_keys = []

        for key, access_time in self._access_times.items():
            # Skip pinned models - they never expire
            if self._is_pinned(key):
                continue
            if current_time - access_time > self._ttl_seconds:
                expired_keys.append(key)

        for key in expired_keys:
            self._cache.pop(key, None)
            self._access_times.pop(key, None)
            logger.info(
                f"Evicted expired model from cache (TTL={self._ttl_seconds}s): {key}"
            )

    def _evict_lru_if_needed(self) -> None:
        """Evict least recently used NON-PINNED item if cache is at capacity.

        PINNED models don't count against max_size and are NEVER evicted.
        This method should be called while holding the lock.
        """
        if self._max_size <= 0:
            return

        # Count only non-pinned models against max_size
        non_pinned_count = sum(1 for k in self._cache if not self._is_pinned(k))

        if non_pinned_count >= self._max_size:
            # Find the least recently used NON-PINNED key
            non_pinned_keys = [k for k in self._access_times if not self._is_pinned(k)]
            if not non_pinned_keys:
                return  # Only pinned models in cache, nothing to evict

            lru_key = min(non_pinned_keys, key=lambda k: self._access_times[k])

            # Remove from cache and access times
            self._cache.pop(lru_key, None)
            self._access_times.pop(lru_key, None)

            logger.info(f"Evicted LRU model from cache: {lru_key}")

    def _update_access_time(self, key: WrapperCacheKey) -> None:
        """Update access time for LRU tracking.

        This method should be called while holding the lock.
        """
        self._access_times[key] = time.time()

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
    ) -> ChatGenerator:
        """Get or create ChatGenerator instance.

        Args:
            model_id: Model name/path (HuggingFace model ID or local path)
            adapter_path: Optional path to LoRA adapter
            draft_model_id: Optional draft model name/path for speculative decoding

        Returns:
            Cached or newly created ChatGenerator instance

        Note:
            This method is thread-safe and will only create one wrapper instance
            per unique parameter combination, even under concurrent access.
        """
        key = WrapperCacheKey(
            model_id=model_id,
            adapter_path=adapter_path,
            draft_model_id=draft_model_id,
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
                    model_id=model_id,
                    adapter_path=adapter_path,
                    draft_model_id=draft_model_id,
                )

                # Only cache if max_size > 0
                if self._max_size > 0:
                    self._cache[key] = wrapper
                    self._update_access_time(key)
                    logger.info(
                        f"Successfully cached ChatGenerator: {key} (cache size: {len(self._cache)}/{self._max_size})"
                    )
                else:
                    logger.info(
                        f"Created ChatGenerator but not cached (max_size=0): {key}"
                    )

                return wrapper
            except Exception as e:
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
        """Clear all cached wrapper instances.

        This can be useful for memory management or testing purposes.
        """
        # Stop the cleanup thread first
        self._stop_cleanup_thread()

        with self._lock:
            cache_size = len(self._cache)
            self._cache.clear()
            self._access_times.clear()
            logger.info(f"Cleared ChatGenerator cache ({cache_size} entries)")

    def unload_model(self, model_id: str) -> bool:
        """Unload a specific model from the cache.

        Args:
            model_id: The model ID to unload

        Returns:
            True if model was found and unloaded, False if not in cache
        """
        with self._lock:
            # Find all keys matching this model_id
            keys_to_remove = [key for key in self._cache if key.model_id == model_id]

            if not keys_to_remove:
                logger.info(f"Model {model_id} not found in cache")
                return False

            for key in keys_to_remove:
                self._cache.pop(key, None)
                self._access_times.pop(key, None)
                logger.info(f"Unloaded model from cache: {key}")

            return True

    def is_model_loaded(self, model_id: str) -> bool:
        """Check if a model is currently loaded in cache.

        Args:
            model_id: The model ID to check

        Returns:
            True if model is in cache, False otherwise
        """
        with self._lock:
            return any(key.model_id == model_id for key in self._cache)

    def get_loaded_models(self) -> list[str]:
        """Get list of currently loaded model IDs.

        Returns:
            List of model IDs currently in cache
        """
        with self._lock:
            return list({key.model_id for key in self._cache})

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
