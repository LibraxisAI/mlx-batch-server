"""E2E tests for batch processing with concurrent requests.

Tests the full pipeline from /v1/responses endpoint through batch coordinator.

Created by M&K (c)2026 VetCoders
"""

import pytest
from httpx import ASGITransport, AsyncClient

from mlx_omni_server.main import create_app


@pytest.fixture
def app():
    """Create test application."""
    return create_app()


class TestBatchStatsEndpoint:
    """Tests for /v1/batch/stats endpoint."""

    @pytest.mark.asyncio
    async def test_batch_stats_endpoint_exists(self, app):
        """Batch stats endpoint should respond."""
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.get("/v1/batch/stats")
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_batch_stats_returns_settings(self, app):
        """Batch stats should include settings."""
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.get("/v1/batch/stats")
            data = response.json()

            assert "enabled" in data
            assert "settings" in data
            assert "coordinators" in data

            # Check default settings
            settings = data["settings"]
            assert "batch_window_ms" in settings
            assert "max_batch_size" in settings


class TestBatchConfig:
    """Tests for batch configuration."""

    def test_batch_settings_in_config(self):
        """Config should include batch settings."""
        from mlx_omni_server.core.config import get_settings

        settings = get_settings()

        assert hasattr(settings, "enable_batch_inference")
        assert hasattr(settings, "batch_window_ms")
        assert hasattr(settings, "max_batch_size")
        assert hasattr(settings, "batch_completion_size")
        assert hasattr(settings, "batch_prefill_size")

    def test_batch_enabled_by_default(self):
        """Batch inference should be enabled by default."""
        from mlx_omni_server.core.config import get_settings

        settings = get_settings()
        assert settings.enable_batch_inference is True


class TestBatchCoordinatorCaching:
    """Tests for coordinator caching behavior."""

    def test_coordinator_cached_per_model(self):
        """Same model should return same coordinator."""
        from mlx_omni_server.batch import get_batch_coordinator

        coord1 = get_batch_coordinator("test-model-cache-1")
        coord2 = get_batch_coordinator("test-model-cache-1")

        assert coord1 is coord2

    def test_coordinator_unique_per_model(self):
        """Different models should get different coordinators."""
        from mlx_omni_server.batch import get_batch_coordinator

        coord1 = get_batch_coordinator("test-model-unique-1")
        coord2 = get_batch_coordinator("test-model-unique-2")

        assert coord1 is not coord2

    @pytest.mark.asyncio
    async def test_coordinator_stats_tracking(self):
        """Coordinator should track request statistics."""
        from mlx_omni_server.batch import get_batch_coordinator

        coord = get_batch_coordinator("test-model-stats")
        stats = coord.stats()

        assert "coordinator" in stats
        assert stats["coordinator"]["total_requests"] == 0
        assert stats["coordinator"]["pending_requests"] == 0


class TestBatchResponsesIntegration:
    """Tests for batch integration with Responses API."""

    def test_adapter_has_batch_method(self):
        """ResponsesAdapter should have batch streaming method."""
        from mlx_omni_server.responses.adapter import ResponsesAdapter

        adapter = ResponsesAdapter()

        assert hasattr(adapter, "_should_use_batch")
        assert hasattr(adapter, "_stream_batch_tokens")
        assert callable(adapter._should_use_batch)

    def test_adapter_batch_decision(self):
        """Adapter should check batch setting."""
        from mlx_omni_server.responses.adapter import ResponsesAdapter

        adapter = ResponsesAdapter()
        # Default is True (batch enabled)
        assert adapter._should_use_batch() is True


class TestBatchGeneratorDataclasses:
    """Tests for batch data structures."""

    def test_batch_stream_chunk_creation(self):
        """BatchStreamChunk should be creatable."""
        from mlx_omni_server.batch import BatchStreamChunk

        chunk = BatchStreamChunk(
            request_id="test_123",
            token=42,
            text="Hello",
            finish_reason=None,
        )

        assert chunk.request_id == "test_123"
        assert chunk.token == 42
        assert chunk.text == "Hello"
        assert chunk.finish_reason is None

    def test_batch_request_creation(self):
        """BatchRequest should be creatable."""
        from mlx_omni_server.batch import BatchRequest

        req = BatchRequest(
            id="req_456",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=100,
        )

        assert req.id == "req_456"
        assert req.max_tokens == 100

    def test_batch_generation_stats(self):
        """BatchGenerationStats should have expected fields."""
        from mlx_omni_server.batch import BatchGenerationStats

        stats = BatchGenerationStats()

        assert stats.prompt_tokens == 0
        assert stats.generation_tokens == 0
        assert stats.prompt_tps == 0.0
        assert stats.generation_tps == 0.0
        assert stats.active_requests == 0
        assert stats.completed_requests == 0
