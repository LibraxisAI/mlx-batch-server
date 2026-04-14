"""Unit tests for MLXWrapperCache.

This module tests the MLXWrapperCache class to ensure proper caching behavior,
LRU eviction, thread safety, and edge case handling.
"""

import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from mlx_batch_server.chat.mlx import runtime_aliases as runtime_aliases_module
from mlx_batch_server.chat.mlx import runtime_attachments as runtime_attachments_module
from mlx_batch_server.chat.mlx.wrapper_cache import MLXWrapperCache, WrapperCacheKey


class MockChatGenerator:
    """Mock wrapper for testing without actual model loading."""

    def __init__(self, model_id: str, *, multimodal: bool = False):
        self.model_id = model_id
        runtime = Mock()
        runtime.processor = Mock() if multimodal else None
        runtime.supports_multimodal = multimodal
        runtime.model = Mock()
        self.model = runtime

    def __str__(self):
        return f"MockWrapper({self.model_id})"


class TestWrapperCacheKey:
    """Test WrapperCacheKey functionality."""

    def test_key_behavior(self):
        """Test key equality, hashing, immutability, and string representation."""
        key1 = WrapperCacheKey("model1", None, None)
        key2 = WrapperCacheKey("model1", None, None)  # Same as key1
        key3 = WrapperCacheKey("model1", "/adapter", None)  # Different

        # Test equality and hashing
        assert key1 == key2
        assert key1 != key3
        assert hash(key1) == hash(key2)
        assert hash(key1) != hash(key3)

        # Test immutability
        with pytest.raises(Exception):  # Should raise FrozenInstanceError
            key1.model_id = "model2"

        # Test string representation
        assert "model1" in str(key1)
        assert "adapter" in str(key3)


class TestMLXWrapperCache:
    """Test MLXWrapperCache functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        runtime_attachments_module.clear_runtime_surface_attachments()
        self.cache = MLXWrapperCache(max_size=3)

    @patch("mlx_batch_server.chat.mlx.wrapper_cache.ChatGenerator.create")
    def test_caching_and_lru_eviction(self, mock_create):
        """Test caching behavior, different keys, LRU eviction, and order tracking."""
        mock_create.side_effect = [
            MockChatGenerator("model1"),
            MockChatGenerator("model1_adapter"),
            MockChatGenerator("model2"),
            MockChatGenerator("model3"),
            MockChatGenerator("model4"),
            MockChatGenerator("model1_adapter_again"),  # For LRU order test
        ]

        # Test cache miss and hit
        wrapper1 = self.cache.get_wrapper("model1")
        wrapper1_again = self.cache.get_wrapper("model1")
        assert wrapper1 is wrapper1_again
        assert mock_create.call_count == 1

        # Test different keys create separate entries
        wrapper_adapter = self.cache.get_wrapper("model1", adapter_path="/adapter")
        assert wrapper1 is not wrapper_adapter
        assert mock_create.call_count == 2

        # Fill cache to capacity and test LRU order
        time.sleep(0.01)
        self.cache.get_wrapper("model2")
        time.sleep(0.01)
        self.cache.get_wrapper("model3")

        info = self.cache.get_cache_info()
        assert info["cache_size"] == 3
        assert info["max_size"] == 3
        # Most recent (model3) should be first in LRU order
        assert "model3" in info["lru_order"][0]

        # Test LRU eviction - add fourth item should evict oldest (model1)
        self.cache.get_wrapper("model4")
        final_info = self.cache.get_cache_info()
        assert final_info["cache_size"] == 3

        cached_models = final_info["cached_keys"]
        assert not any(
            "model1" in key and "/adapter" not in key for key in cached_models
        )
        assert any("model4" in key for key in cached_models)

        # Test LRU order update - access model_adapter again
        self.cache.get_wrapper("model1", adapter_path="/adapter")
        updated_info = self.cache.get_cache_info()
        # model1_adapter should now be most recent
        assert "adapter" in updated_info["lru_order"][0]

    @patch("mlx_batch_server.chat.mlx.wrapper_cache.ChatGenerator.create")
    def test_cache_management(self, mock_create):
        """Test cache size changes, clearing, and error handling."""
        mock_create.side_effect = [
            MockChatGenerator("model1"),
            MockChatGenerator("model2"),
            RuntimeError("Model loading failed"),
        ]

        # Test cache operations
        self.cache.get_wrapper("model1")
        self.cache.get_wrapper("model2")
        assert self.cache.get_cache_info()["cache_size"] == 2

        # Test changing max size
        self.cache.set_max_size(1)
        info = self.cache.get_cache_info()
        assert info["max_size"] == 1
        assert info["cache_size"] == 1  # Should evict oldest

        # Test clear cache
        self.cache.clear_cache()
        assert self.cache.get_cache_info()["cache_size"] == 0

        # Test error handling
        with pytest.raises(RuntimeError, match="Model loading failed"):
            self.cache.get_wrapper("broken_model")
        assert self.cache.get_cache_info()["cache_size"] == 0

    @patch("mlx_batch_server.chat.mlx.wrapper_cache.ChatGenerator.create")
    def test_surface_attached_runtime_survives_lru_pressure(self, mock_create):
        """Surface-retained VLM runtimes should not be silently evicted by LRU."""
        cache = MLXWrapperCache(max_size=1)
        mock_create.side_effect = [
            MockChatGenerator("model-vlm", multimodal=True),
            MockChatGenerator("model-b"),
            MockChatGenerator("model-c"),
        ]

        cache.get_wrapper("model-vlm")
        runtime_attachments_module.attach_runtime_surface("model-vlm", "visual")

        cache.get_wrapper("model-b")
        info = cache.get_cache_info()
        assert info["cache_size"] == 2
        assert any("model-vlm" in key for key in info["cached_keys"])
        assert any("model-b" in key for key in info["cached_keys"])

        cache.get_wrapper("model-c")
        info = cache.get_cache_info()
        assert info["cache_size"] == 2
        assert any("model-vlm" in key for key in info["cached_keys"])
        assert any("model-c" in key for key in info["cached_keys"])
        assert all("model-b" not in key for key in info["cached_keys"])


class TestMLXWrapperCacheThreadSafety:
    """Test thread safety of MLXWrapperCache."""

    def setup_method(self):
        """Set up test fixtures."""
        runtime_attachments_module.clear_runtime_surface_attachments()
        self.cache = MLXWrapperCache(max_size=10)
        self.results = []
        self.creation_count = 0

    def mock_create_with_delay(self, model_id, **kwargs):
        """Mock create method with artificial delay to test concurrency."""
        time.sleep(0.01)  # Small delay to increase chance of race conditions
        self.creation_count += 1
        return MockChatGenerator(f"{model_id}_{self.creation_count}")

    def test_concurrent_access(self):
        """Test concurrent access for same and different cache keys."""
        with patch(
            "mlx_batch_server.chat.mlx.wrapper_cache.ChatGenerator.create"
        ) as mock_create:
            mock_create.side_effect = self.mock_create_with_delay

            # Test same key - all threads should get same instance
            def worker_same_key():
                wrapper = self.cache.get_wrapper("shared_model")
                self.results.append(("same", wrapper))

            threads = [threading.Thread(target=worker_same_key) for _ in range(5)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            # All should get same wrapper instance
            same_key_wrappers = [
                wrapper for key_type, wrapper in self.results if key_type == "same"
            ]
            assert len({id(wrapper) for wrapper in same_key_wrappers}) == 1
            first_creation_count = self.creation_count
            assert first_creation_count == 1

            # Test different keys - should create different instances
            def worker_different_key(model_id):
                wrapper = self.cache.get_wrapper(f"unique_model_{model_id}")
                self.results.append(("different", wrapper))

            threads = [
                threading.Thread(target=worker_different_key, args=(i,))
                for i in range(3)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            # Should create different wrappers for different keys
            different_key_wrappers = [
                wrapper for key_type, wrapper in self.results if key_type == "different"
            ]
            assert len({id(wrapper) for wrapper in different_key_wrappers}) == 3
            assert self.creation_count == first_creation_count + 3


class TestMLXWrapperCacheTTL:
    """Test TTL (Time To Live) functionality of MLXWrapperCache."""

    def setup_method(self):
        """Set up test fixtures."""
        # Use short TTL for faster testing
        runtime_attachments_module.clear_runtime_surface_attachments()
        self.cache = MLXWrapperCache(max_size=5, ttl_seconds=1)

    @patch("mlx_batch_server.chat.mlx.wrapper_cache.ChatGenerator.create")
    def test_ttl_expiration_and_renewal(self, mock_create):
        """Test TTL expiration, renewal on access, and cache info."""
        mock_create.return_value = MockChatGenerator("model1")

        # Add item and verify TTL info
        self.cache.get_wrapper("model1")
        info = self.cache.get_cache_info()
        assert info["cache_size"] == 1
        assert info["ttl_seconds"] == 1
        assert len(info["ttl_info"]) == 1
        assert info["ttl_info"][0]["remaining_ttl_seconds"] <= 1.0

        # Test TTL renewal - access after half TTL
        time.sleep(0.5)
        self.cache.get_wrapper("model1")  # Should renew TTL
        time.sleep(0.8)  # Total 1.3s, but last access was 0.8s ago
        assert self.cache.get_cache_info()["cache_size"] == 1  # Still cached

        # Test actual expiration
        time.sleep(0.5)  # Now 1.3s since last access, should expire
        info = self.cache.get_cache_info()
        assert info["cache_size"] == 0

    @patch("mlx_batch_server.chat.mlx.wrapper_cache.ChatGenerator.create")
    def test_ttl_management(self, mock_create):
        """Test TTL disabled, manual cleanup, and TTL+LRU interaction."""
        mock_create.side_effect = [
            MockChatGenerator("model1"),
            MockChatGenerator("model2"),
        ]

        # Test TTL disabled
        cache_no_ttl = MLXWrapperCache(max_size=3, ttl_seconds=0)
        with patch(
            "mlx_batch_server.chat.mlx.wrapper_cache.ChatGenerator.create"
        ) as mock_no_ttl:
            mock_no_ttl.return_value = MockChatGenerator("model_no_ttl")
            cache_no_ttl.get_wrapper("model_no_ttl")
            time.sleep(0.1)
            info = cache_no_ttl.get_cache_info()
            assert info["cache_size"] == 1
            assert info["ttl_seconds"] == 0
            assert info["ttl_info"] == []

        # Test manual cleanup
        self.cache.get_wrapper("model1")
        time.sleep(0.5)
        self.cache.get_wrapper("model2")
        time.sleep(0.8)  # model1 should be expired

        evicted_count = self.cache.cleanup_expired_items()
        assert evicted_count >= 1
        info = self.cache.get_cache_info()
        assert all("model1" not in key for key in info["cached_keys"])
        if info["cache_size"] == 1:
            assert any("model2" in key for key in info["cached_keys"])

        # Test TTL + LRU interaction
        ttl_lru_cache = MLXWrapperCache(max_size=2, ttl_seconds=0.5)
        with patch(
            "mlx_batch_server.chat.mlx.wrapper_cache.ChatGenerator.create"
        ) as mock_ttl_lru:
            mock_ttl_lru.side_effect = [
                MockChatGenerator(f"model{i}") for i in range(1, 4)
            ]
            ttl_lru_cache.get_wrapper("model1")
            time.sleep(0.1)
            ttl_lru_cache.get_wrapper("model2")
            time.sleep(0.6)  # Both should expire
            ttl_lru_cache.get_wrapper("model3")  # Should trigger cleanup
            info = ttl_lru_cache.get_cache_info()
            assert info["cache_size"] == 1
            assert any("model3" in key for key in info["cached_keys"])

    @patch("mlx_batch_server.chat.mlx.wrapper_cache.ChatGenerator.create")
    def test_ttl_does_not_expire_surface_attached_runtime(self, mock_create):
        """TTL must not drop a runtime while a product surface still retains it."""
        mock_create.return_value = MockChatGenerator("model-vlm", multimodal=True)

        self.cache.get_wrapper("model-vlm")
        runtime_attachments_module.attach_runtime_surface("model-vlm", "embeddings")

        time.sleep(1.2)
        info = self.cache.get_cache_info()

        assert info["cache_size"] == 1
        assert any("model-vlm" in key for key in info["cached_keys"])
        assert runtime_attachments_module.get_runtime_surface_attachments(
            "model-vlm"
        ) == ["embeddings"]


class TestMLXWrapperCacheEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_edge_cases(self):
        """Test zero size, single item cache, and large cache configurations."""
        # Test zero max size - should work but not cache
        zero_cache = MLXWrapperCache(max_size=0)
        with patch(
            "mlx_batch_server.chat.mlx.wrapper_cache.ChatGenerator.create"
        ) as mock_zero:
            mock_zero.return_value = MockChatGenerator("model1")
            wrapper = zero_cache.get_wrapper("model1")
            assert wrapper is not None
            assert zero_cache.get_cache_info()["cache_size"] == 0

        # Test single item cache - should evict on second addition
        single_cache = MLXWrapperCache(max_size=1)
        with patch(
            "mlx_batch_server.chat.mlx.wrapper_cache.ChatGenerator.create"
        ) as mock_single:
            mock_single.side_effect = [
                MockChatGenerator("model1"),
                MockChatGenerator("model2"),
            ]
            single_cache.get_wrapper("model1")
            assert single_cache.get_cache_info()["cache_size"] == 1

            single_cache.get_wrapper("model2")
            info = single_cache.get_cache_info()
            assert info["cache_size"] == 1
            assert not any("model1" in key for key in info["cached_keys"])
            assert any("model2" in key for key in info["cached_keys"])

        # Test very large max size
        large_cache = MLXWrapperCache(max_size=1000)
        info = large_cache.get_cache_info()
        assert info["max_size"] == 1000
        assert info["cache_size"] == 0


class TestMLXWrapperCacheModelManagement:
    """Test model management methods: unload_model, is_model_loaded, get_loaded_models."""

    def setup_method(self):
        """Set up test fixtures."""
        runtime_attachments_module.clear_runtime_surface_attachments()
        self.cache = MLXWrapperCache(max_size=5)

    @patch("mlx_batch_server.chat.mlx.wrapper_cache.ChatGenerator.create")
    def test_unload_model_success(self, mock_create):
        """Test unloading a specific model from cache."""
        mock_create.side_effect = [
            MockChatGenerator("model1"),
            MockChatGenerator("model2"),
        ]

        # Load two models
        self.cache.get_wrapper("model1")
        self.cache.get_wrapper("model2")
        assert self.cache.get_cache_info()["cache_size"] == 2

        # Unload model1
        result = self.cache.unload_model("model1")
        assert result is True
        assert self.cache.get_cache_info()["cache_size"] == 1
        assert not self.cache.is_model_loaded("model1")
        assert self.cache.is_model_loaded("model2")

    @patch("mlx_batch_server.chat.mlx.wrapper_cache.ChatGenerator.create")
    def test_unload_model_not_found(self, mock_create):
        """Test unloading a model that isn't in cache."""
        mock_create.return_value = MockChatGenerator("model1")

        self.cache.get_wrapper("model1")

        # Try to unload non-existent model
        result = self.cache.unload_model("nonexistent")
        assert result is False
        assert self.cache.get_cache_info()["cache_size"] == 1

    @patch("mlx_batch_server.chat.mlx.wrapper_cache.ChatGenerator.create")
    def test_unload_model_with_adapter(self, mock_create):
        """Test unloading model also removes adapter variants."""
        mock_create.side_effect = [
            MockChatGenerator("model1"),
            MockChatGenerator("model1_adapter"),
        ]

        # Load same model with and without adapter
        self.cache.get_wrapper("model1")
        self.cache.get_wrapper("model1", adapter_path="/adapter")
        assert self.cache.get_cache_info()["cache_size"] == 2

        # Unload model1 - should remove both entries
        result = self.cache.unload_model("model1")
        assert result is True
        assert self.cache.get_cache_info()["cache_size"] == 0

    @patch("mlx_batch_server.chat.mlx.wrapper_cache.ChatGenerator.create")
    def test_unload_model_clears_runtime_surface_attachments(self, mock_create):
        """Actual runtime eviction should remove stale surface ownership metadata."""
        mock_create.return_value = MockChatGenerator("model-vlm", multimodal=True)

        self.cache.get_wrapper("model-vlm")
        runtime_attachments_module.attach_runtime_surface("model-vlm", "visual")
        runtime_attachments_module.attach_runtime_surface("model-vlm", "embeddings")

        assert runtime_attachments_module.get_runtime_surface_attachments(
            "model-vlm"
        ) == ["embeddings", "visual"]

        assert self.cache.unload_model("model-vlm") is True
        assert (
            runtime_attachments_module.get_runtime_surface_attachments("model-vlm")
            == []
        )

    @patch("mlx_batch_server.chat.mlx.wrapper_cache.ChatGenerator.create")
    def test_is_model_loaded(self, mock_create):
        """Test checking if a model is loaded."""
        mock_create.return_value = MockChatGenerator("model1")

        assert self.cache.is_model_loaded("model1") is False

        self.cache.get_wrapper("model1")
        assert self.cache.is_model_loaded("model1") is True

        self.cache.clear_cache()
        assert self.cache.is_model_loaded("model1") is False

    @patch("mlx_batch_server.chat.mlx.wrapper_cache.ChatGenerator.create")
    def test_is_runtime_loaded_tracks_exact_runtime_key(self, mock_create):
        """Exact runtime checks must distinguish adapter and draft variants."""
        mock_create.side_effect = [
            MockChatGenerator("model1"),
            MockChatGenerator("model1_adapter"),
        ]

        self.cache.get_wrapper("model1")
        self.cache.get_wrapper("model1", adapter_path="/adapter")

        assert self.cache.is_runtime_loaded("model1") is True
        assert self.cache.is_runtime_loaded("model1", adapter_path="/adapter") is True
        assert (
            self.cache.is_runtime_loaded(
                "model1",
                adapter_path="/other-adapter",
            )
            is False
        )
        assert (
            self.cache.is_runtime_loaded(
                "model1",
                adapter_path="/adapter",
                draft_model_id="draft-a",
            )
            is False
        )

    @patch("mlx_batch_server.chat.mlx.wrapper_cache.ChatGenerator.create")
    def test_get_loaded_models(self, mock_create):
        """Test getting list of loaded model IDs."""
        mock_create.side_effect = [
            MockChatGenerator("model1"),
            MockChatGenerator("model2"),
            MockChatGenerator("model1_adapter"),
        ]

        assert self.cache.get_loaded_models() == []

        self.cache.get_wrapper("model1")
        self.cache.get_wrapper("model2")
        self.cache.get_wrapper("model1", adapter_path="/adapter")

        loaded = self.cache.get_loaded_models()
        # Should return unique model IDs (model1 appears twice with different adapters)
        assert set(loaded) == {"model1", "model2"}

    @patch("mlx_batch_server.chat.mlx.wrapper_cache.ChatGenerator.create")
    def test_unload_empty_cache(self, mock_create):
        """Test unloading from empty cache."""
        result = self.cache.unload_model("any_model")
        assert result is False
        assert self.cache.get_cache_info()["cache_size"] == 0

    @patch("mlx_batch_server.chat.mlx.wrapper_cache.ChatGenerator.create")
    def test_multimodal_backend_reuses_shared_wrapper_runtime(self, mock_create):
        """VLM backend access should reuse the same cached wrapper runtime."""
        wrapper = MockChatGenerator("model-vlm", multimodal=True)
        mock_create.return_value = wrapper

        first = self.cache.get_wrapper("model-vlm")
        model, processor = self.cache.get_vlm_backend("model-vlm")

        assert first is wrapper
        assert model is wrapper.model.model
        assert processor is wrapper.model.processor
        assert self.cache.get_loaded_vlm_models() == ["model-vlm"]

    @patch("mlx_batch_server.chat.mlx.wrapper_cache.ChatGenerator.create")
    def test_runtime_alias_reuses_single_cached_wrapper(self, mock_create):
        """Dynamic runtime aliases should not create duplicate residency."""
        runtime_aliases_module.clear_runtime_aliases()
        try:
            runtime_aliases_module.register_runtime_alias(
                "qwen3.5-vl-crack",
                "libraxisai/gpt-oss-120b-mlx-mxfp4",
            )
            mock_create.return_value = MockChatGenerator(
                "libraxisai/gpt-oss-120b-mlx-mxfp4",
                multimodal=True,
            )

            canonical = self.cache.get_wrapper("libraxisai/gpt-oss-120b-mlx-mxfp4")
            alias = self.cache.get_wrapper("qwen3.5-vl-crack")

            assert alias is canonical
            assert mock_create.call_count == 1
            assert self.cache.get_loaded_models() == [
                "libraxisai/gpt-oss-120b-mlx-mxfp4"
            ]
            assert self.cache.get_loaded_vlm_models() == [
                "libraxisai/gpt-oss-120b-mlx-mxfp4"
            ]
        finally:
            runtime_aliases_module.clear_runtime_aliases()

    @patch("mlx_batch_server.chat.mlx.wrapper_cache.ChatGenerator.create")
    def test_home_relative_path_reuses_single_cached_wrapper(self, mock_create):
        """Home-relative model paths should collapse to one cache identity."""
        model_path = "~/models/frontier-vlm"
        expanded_path = str(Path(model_path).expanduser())
        mock_create.return_value = MockChatGenerator(expanded_path, multimodal=True)

        first = self.cache.get_wrapper(model_path)
        second = self.cache.get_wrapper(expanded_path)

        assert second is first
        assert mock_create.call_count == 1
        assert self.cache.get_loaded_models() == [expanded_path]

    @patch("mlx_batch_server.chat.mlx.wrapper_cache.ChatGenerator.create")
    def test_home_relative_adapter_path_reuses_single_cached_wrapper(self, mock_create):
        """Home-relative adapter paths should collapse to one cache identity."""
        adapter_path = "~/adapters/frontier-lora"
        expanded_adapter_path = str(Path(adapter_path).expanduser())
        mock_create.return_value = MockChatGenerator("model-with-adapter")

        first = self.cache.get_wrapper("model-with-adapter", adapter_path=adapter_path)
        second = self.cache.get_wrapper(
            "model-with-adapter",
            adapter_path=expanded_adapter_path,
        )

        assert second is first
        assert mock_create.call_count == 1
        assert self.cache.get_cache_info()["cache_size"] == 1

    @patch("mlx_batch_server.chat.mlx.wrapper_cache.ChatGenerator.create")
    def test_home_relative_adapter_path_is_canonicalized_before_wrapper_creation(
        self,
        mock_create,
    ):
        """Cache identity and downstream runtime args should share one adapter path."""
        adapter_path = "~/adapters/frontier-lora"
        expanded_adapter_path = str(Path(adapter_path).expanduser())
        mock_create.return_value = MockChatGenerator("model-with-adapter")

        self.cache.get_wrapper("model-with-adapter", adapter_path=adapter_path)

        assert mock_create.call_count == 1
        assert mock_create.call_args.kwargs["adapter_path"] == expanded_adapter_path


class TestMLXWrapperCachePinnedOnly:
    """Test pinned-only mode (max_size=0 + PINNED_MODELS)."""

    @patch("mlx_batch_server.chat.mlx.wrapper_cache.ChatGenerator.create")
    def test_pinned_model_cached_with_max_size_zero(self, mock_create):
        """Pinned model must be cached even when max_size=0."""
        cache = MLXWrapperCache(
            max_size=0, pinned_models=["libraxisai/gpt-oss-120b-mlx-mxfp4"]
        )
        mock_create.return_value = MockChatGenerator(
            "libraxisai/gpt-oss-120b-mlx-mxfp4"
        )

        wrapper1 = cache.get_wrapper("libraxisai/gpt-oss-120b-mlx-mxfp4")
        wrapper2 = cache.get_wrapper("libraxisai/gpt-oss-120b-mlx-mxfp4")

        assert wrapper2 is wrapper1
        assert cache.get_cache_info()["cache_size"] == 1
        assert cache.is_model_loaded("libraxisai/gpt-oss-120b-mlx-mxfp4")
        assert mock_create.call_count == 1

    @patch("mlx_batch_server.chat.mlx.wrapper_cache.ChatGenerator.create")
    def test_non_pinned_not_cached_with_max_size_zero(self, mock_create):
        """Non-pinned model should not be cached in pinned-only mode."""
        cache = MLXWrapperCache(
            max_size=0, pinned_models=["libraxisai/gpt-oss-120b-mlx-mxfp4"]
        )
        mock_create.return_value = MockChatGenerator("some-random-model")

        wrapper = cache.get_wrapper("some-random-model")

        assert wrapper is not None
        assert cache.get_cache_info()["cache_size"] == 0

    @patch("mlx_batch_server.chat.mlx.wrapper_cache.ChatGenerator.create")
    def test_pinned_survives_ttl(self, mock_create):
        """Pinned models must survive TTL expiration."""
        cache = MLXWrapperCache(
            max_size=0, ttl_seconds=1, pinned_models=["pinned-model"]
        )
        mock_create.return_value = MockChatGenerator("pinned-model")

        cache.get_wrapper("pinned-model")
        time.sleep(1.5)

        assert cache.get_cache_info()["cache_size"] == 1
        assert cache.is_model_loaded("pinned-model")

    @patch("mlx_batch_server.chat.mlx.wrapper_cache.ChatGenerator.create")
    def test_pinned_model_case_insensitive(self, mock_create):
        """Pinned Hugging Face IDs should match regardless of case."""
        cache = MLXWrapperCache(
            max_size=0, pinned_models=["LibraxisAI/GPT-OSS-120B-mlx-mxfp4"]
        )
        mock_create.return_value = MockChatGenerator(
            "libraxisai/gpt-oss-120b-mlx-mxfp4"
        )

        cache.get_wrapper("libraxisai/gpt-oss-120b-mlx-mxfp4")
        cache.get_wrapper("LibraxisAI/GPT-OSS-120B-mlx-mxfp4")

        assert cache.get_cache_info()["cache_size"] == 1
        assert mock_create.call_count == 1
