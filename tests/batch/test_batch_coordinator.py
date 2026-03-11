"""Tests for BatchRequestCoordinator.

Vibecrafted with AI Agents by VetCoders (c)2026 VetCoders
"""

import asyncio
from types import SimpleNamespace

import pytest

from mlx_batch_server.batch.coordinator import (
    BatchRequestCoordinator,
    PendingRequest,
    get_batch_coordinator,
    get_loaded_batch_models,
    shutdown_batch_coordinator,
)
from mlx_batch_server.batch.generator import BatchChatGenerator
from mlx_batch_server.chat.mlx.chat_generator import ChatGenerator


class TestPendingRequest:
    """Tests for PendingRequest dataclass."""

    def test_create_pending_request(self):
        """PendingRequest can be created with required fields."""
        queue = asyncio.Queue()
        req = PendingRequest(
            request_id="test_123",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=100,
            tools=None,
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

    def test_get_or_create_generator_reuses_shared_vlm_wrapper(self, monkeypatch):
        """Coordinator should keep using the shared chat wrapper for VLM text paths."""
        coord = BatchRequestCoordinator(model_id="test-vlm")
        fake_wrapper = SimpleNamespace(model=SimpleNamespace(text_model=object()))
        fake_batch_generator = SimpleNamespace()
        seen = {}

        monkeypatch.setattr(
            ChatGenerator,
            "get_or_create",
            lambda model_id, adapter_path=None, draft_model_id=None: fake_wrapper,
        )
        monkeypatch.setattr(
            BatchChatGenerator,
            "from_chat_generator",
            lambda chat_generator,
            completion_batch_size=32,
            prefill_batch_size=8,
            prefill_step_size=2048: (
                seen.setdefault("chat_generator", chat_generator),
                fake_batch_generator,
            )[1],
        )

        result = coord._get_or_create_generator()

        assert result is fake_batch_generator
        assert seen["chat_generator"] is fake_wrapper


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

    def test_loaded_batch_models_only_include_initialized_generators(self):
        """Only coordinators with live generators should be reported as loaded."""
        loaded_model = "model-loaded-generator"
        idle_model = "model-without-generator"

        loaded = get_batch_coordinator(loaded_model)
        idle = get_batch_coordinator(idle_model)

        fake_generator = SimpleNamespace(
            stats_dict=lambda: {},
            close=lambda: None,
        )

        loaded._generator = fake_generator
        idle._generator = None

        loaded_models = get_loaded_batch_models()

        assert loaded_model in loaded_models
        assert idle_model not in loaded_models

        loaded._generator = None

    @pytest.mark.asyncio
    async def test_shutdown_batch_coordinator_removes_matching_model(self):
        """Shutdown should remove all coordinators for the selected model."""
        target_model = "model-shutdown-target"
        other_model = "model-shutdown-other"
        target = get_batch_coordinator(target_model)
        other = get_batch_coordinator(other_model)

        state = {"target_shutdown": False, "other_shutdown": False}

        async def target_shutdown():
            state["target_shutdown"] = True

        async def other_shutdown():
            state["other_shutdown"] = True

        target.shutdown = target_shutdown
        other.shutdown = other_shutdown

        removed = await shutdown_batch_coordinator(target_model)

        assert removed == 1
        assert state["target_shutdown"] is True
        assert state["other_shutdown"] is False
        assert get_batch_coordinator(other_model) is other
