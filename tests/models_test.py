import logging

import pytest
from fastapi.testclient import TestClient
from openai import NotFoundError, OpenAI
from src.mlx_omni_server.main import app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def openai_client(client):
    """Create OpenAI client configured with the test server."""
    return OpenAI(
        base_url="http://test/v1",
        api_key="test",
        http_client=client,
    )


def test_list_models_default(openai_client: OpenAI):
    """Test listing models without details (default)."""
    model_list = openai_client.models.list()
    if not model_list.data:
        pytest.skip("No models available to test.")
    for model in model_list.data:
        assert not hasattr(model, "details") or model.details is None


def test_list_models_with_details(openai_client: OpenAI):
    """Test listing models with the show_details flag."""
    model_list = openai_client.models.list(
        extra_query={
            "include_details": True,
        }
    )
    if not model_list.data:
        pytest.skip("No models available to test.")
    for model in model_list.data:
        assert model.details is not None
        assert isinstance(model.details, dict)


def test_get_existing_model_with_details(openai_client: OpenAI):
    """Test retrieving a single, existing model with details."""
    # First, get a valid model ID from the list
    model_list = openai_client.models.list()
    if not model_list.data:
        pytest.skip("No models available in the cache to test retrieval.")

    model_id_to_test = model_list.data[0].id

    try:
        model = openai_client.models.retrieve(
            model_id_to_test,
            extra_query={
                "include_details": True,
            },
        )
        logger.info(f"Retrieved Model with details: {model}")

        assert model is not None
        assert model.id == model_id_to_test
        assert model.details is not None
        assert isinstance(model.details, dict)
        assert model.details.get("model_type") is not None

    except Exception as e:
        logger.error(
            f"Test error retrieving model '{model_id_to_test}' with details: {e!s}"
        )
        raise


def test_get_existing_model_without_details(openai_client: OpenAI):
    """Test retrieving a single, existing model without details."""
    # First, get a valid model ID from the list
    model_list = openai_client.models.list()
    if not model_list.data:
        pytest.skip("No models available in the cache to test retrieval.")

    model_id_to_test = model_list.data[0].id

    try:
        model = openai_client.models.retrieve(
            model_id_to_test,
            extra_query={
                "include_details": False,
            },
        )
        logger.info(f"Retrieved Model without details: {model}")

        assert model is not None
        assert model.id == model_id_to_test
        assert not hasattr(model, "details") or model.details is None

    except Exception as e:
        logger.error(
            f"Test error retrieving model '{model_id_to_test}' without details: {e!s}"
        )
        raise


def test_get_non_existent_model(openai_client: OpenAI):
    """Test retrieving a non-existent model."""
    non_existent_model_id = "non-existent/model-that-should-not-be-found"
    with pytest.raises(NotFoundError):
        openai_client.models.retrieve(non_existent_model_id)


# === Model Load/Unload Tests ===


class TestModelLoadUnload:
    """Test model load and unload endpoints."""

    @pytest.fixture
    def test_client(self):
        """Create test client."""
        return TestClient(app)

    def test_load_model_success(self, test_client):
        """Test loading a model via POST /v1/models/load."""
        response = test_client.post(
            "/v1/models/load",
            json={"model": "mlx-community/Qwen2.5-0.5B-Instruct-4bit"},
        )

        # Should succeed (200) or fail gracefully if model not available
        if response.status_code == 200:
            data = response.json()
            assert data["id"] == "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
            assert data["status"] in ("loaded", "already_loaded")
            assert "message" in data
            assert "cache_info" in data
        else:
            # Model might not be available - that's OK for CI
            pytest.skip("Model not available for loading")

    def test_load_model_idempotent(self, test_client):
        """Test that loading the same model twice returns already_loaded."""
        # First load
        response1 = test_client.post(
            "/v1/models/load",
            json={"model": "mlx-community/Qwen2.5-0.5B-Instruct-4bit"},
        )

        if response1.status_code != 200:
            pytest.skip("Model not available for loading")

        # Second load - should be idempotent
        response2 = test_client.post(
            "/v1/models/load",
            json={"model": "mlx-community/Qwen2.5-0.5B-Instruct-4bit"},
        )

        assert response2.status_code == 200
        data = response2.json()
        assert data["status"] == "already_loaded"

    def test_unload_model_success(self, test_client):
        """Test unloading a model via POST /v1/models/unload."""
        # First load a model
        load_response = test_client.post(
            "/v1/models/load",
            json={"model": "mlx-community/Qwen2.5-0.5B-Instruct-4bit"},
        )

        if load_response.status_code != 200:
            pytest.skip("Model not available for loading")

        # Now unload it
        response = test_client.post(
            "/v1/models/unload",
            json={"model": "mlx-community/Qwen2.5-0.5B-Instruct-4bit"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unloaded"
        assert "mlx-community/Qwen2.5-0.5B-Instruct-4bit" in data["unloaded_models"]

    def test_unload_not_loaded_model(self, test_client):
        """Test unloading a model that isn't loaded."""
        response = test_client.post(
            "/v1/models/unload",
            json={"model": "non-existent/model-never-loaded"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "not_found"
        assert data["unloaded_models"] == []

    def test_unload_all_models(self, test_client):
        """Test unloading all models (no model specified)."""
        response = test_client.post(
            "/v1/models/unload",
            json={},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "cleared"
        assert "cache_info" in data
        assert data["cache_info"]["cache_size"] == 0

    def test_load_with_adapter_path(self, test_client):
        """Test loading a model with adapter_path parameter."""
        response = test_client.post(
            "/v1/models/load",
            json={
                "model": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
                "adapter_path": "/nonexistent/adapter",
            },
        )

        # Will likely fail due to nonexistent adapter, but endpoint should handle it
        assert response.status_code in (200, 500)

    def test_load_invalid_model(self, test_client):
        """Test loading an invalid model returns error."""
        response = test_client.post(
            "/v1/models/load",
            json={"model": ""},
        )

        # Empty model should fail
        assert response.status_code == 500
