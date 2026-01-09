"""Tests for BatchRequestCoordinator.

Created by M&K (c)2026 VetCoders
"""

import asyncio

from mlx_omni_server.batch.coordinator import (
    BatchRequestCoordinator,
    PendingRequest,
    get_batch_coordinator,
)


class TestPendingRequest:
    """Tests for PendingRequest dataclass."""

    def test_create_pending_request(self):
        """PendingRequest can be created with required fields."""
        queue = asyncio.Queue()
        req = PendingRequest(
            request_id="test_123",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=100,
            sampler_config=None,
            template_kwargs=None,
            response_queue=queue,
            created_at=1234567890.0,
        )

        assert req.request_id == "test_123"
        assert req.messages == [{"role": "user", "content": "Hello"}]
        assert req.max_tokens == 100
        assert req.response_queue is queue


class TestBatchRequestCoordinator:
    """Tests for BatchRequestCoordinator."""

    def test_coordinator_initialization(self):
        """Coordinator initializes with correct parameters."""
        coord = BatchRequestCoordinator(
            model_id="test-model",
            adapter_path=None,
            batch_window_ms=100,
            max_batch_size=5,
        )

        assert coord.model_id == "test-model"
        assert coord._batch_window_ms == 100
        assert coord._max_batch_size == 5
        assert not coord.is_active

    def test_stats_empty_coordinator(self):
        """Stats returns valid structure for empty coordinator."""
        coord = BatchRequestCoordinator(
            model_id="test-model",
        )

        stats = coord.stats()

        assert "coordinator" in stats
        assert "generator" in stats
        assert stats["coordinator"]["total_requests"] == 0
        assert stats["coordinator"]["total_batches"] == 0
        assert stats["coordinator"]["pending_requests"] == 0


class TestGetBatchCoordinator:
    """Tests for coordinator factory function."""

    def test_returns_same_instance(self):
        """Same model returns same coordinator instance."""
        coord1 = get_batch_coordinator("model-a")
        coord2 = get_batch_coordinator("model-a")

        assert coord1 is coord2

    def test_different_models_different_instances(self):
        """Different models return different coordinators."""
        coord1 = get_batch_coordinator("model-a")
        coord2 = get_batch_coordinator("model-b")

        assert coord1 is not coord2

    def test_adapter_path_affects_cache_key(self):
        """Different adapter paths create different coordinators."""
        coord1 = get_batch_coordinator("model-c", adapter_path=None)
        coord2 = get_batch_coordinator("model-c", adapter_path="/path/to/adapter")

        assert coord1 is not coord2
