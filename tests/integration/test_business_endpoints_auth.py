"""Business inference endpoints respect SECURITY_LEVEL auth gates."""

from __future__ import annotations

from contextlib import asynccontextmanager
from importlib import import_module
from typing import Any


class _DumpableResponse:
    def __init__(self, payload: dict[str, Any]):
        self._payload = payload

    def model_dump(self, **_kwargs: Any) -> dict[str, Any]:
        return self._payload


@asynccontextmanager
async def _noop_runtime_session(*_args: Any, **_kwargs: Any):
    yield {"switched": False, "evicted_models": []}


def _chat_payload() -> dict[str, Any]:
    return {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }


def _anthropic_payload() -> dict[str, Any]:
    return {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 8,
        "stream": False,
    }


def test_business_endpoints_require_auth(inference_client_with_auth):
    """No business inference surface should pass unauthenticated at level 2."""
    with inference_client_with_auth() as client:
        cases = [
            ("post", "/v1/responses", {"json": {"model": "m", "input": "hi"}}),
            ("post", "/v1/chat/completions", {"json": _chat_payload()}),
            ("post", "/anthropic/v1/messages", {"json": _anthropic_payload()}),
            ("get", "/v1/models", {}),
            ("post", "/v1/embeddings", {"json": {"model": "m", "input": "hi"}}),
            (
                "post",
                "/v1/maxsim",
                {"json": {"query_embedding": [[1.0]], "doc_embedding": [[1.0]]}},
            ),
            ("post", "/v1/images/generations", {"json": {"prompt": "hi"}}),
            (
                "post",
                "/v1/audio/transcriptions",
                {
                    "files": {"file": ("sample.wav", b"fake wav", "audio/wav")},
                    "data": {"model": "whisper-test"},
                },
            ),
            (
                "post",
                "/v1/audio/speech",
                {"json": {"model": "tts-test", "input": "hello"}},
            ),
            ("get", "/v1/batch/stats", {}),
        ]

        for method, path, kwargs in cases:
            response = getattr(client, method)(path, **kwargs)
            assert response.status_code == 401, path


def test_business_endpoints_accept_static_api_key(
    inference_client_with_auth, monkeypatch
):
    """With the static key, auth lets each business router reach its handler."""
    from mlx_batch_server.chat.openai.models import models as model_routes
    from mlx_batch_server.chat.openai.models.schema import Model, ModelList
    from mlx_batch_server.embeddings import visual_router
    from mlx_batch_server.embeddings.schema import (
        EmbeddingData,
        EmbeddingResponse,
        EmbeddingUsage,
    )
    from mlx_batch_server.images.schema import ImageObject

    anthropic_router = import_module("mlx_batch_server.chat.anthropic.router")
    batch_coordinator = import_module("mlx_batch_server.batch.coordinator")
    chat_router = import_module("mlx_batch_server.chat.openai.router")
    embeddings_router = import_module("mlx_batch_server.embeddings.router")
    images_router = import_module("mlx_batch_server.images.images")
    responses_router = import_module("mlx_batch_server.responses.router")
    stt_router = import_module("mlx_batch_server.stt.stt")
    tts_router = import_module("mlx_batch_server.tts.tts")

    class FakeResponsesAdapter:
        async def generate(self, request):
            return _DumpableResponse(
                {
                    "id": "resp_test",
                    "object": "response",
                    "created_at": 1,
                    "model": request.model,
                    "status": "completed",
                    "output": [],
                    "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                }
            )

    class FakeTextModel:
        def generate(self, request):
            return _DumpableResponse(
                {
                    "id": "chatcmpl_test",
                    "object": "chat.completion",
                    "created": 1,
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                }
            )

    class FakeAnthropicModel:
        def generate(self, request):
            return _DumpableResponse(
                {
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "model": request.model,
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                }
            )

    class FakeModelsService:
        def list_models(self, include_details: bool = False):
            return ModelList(
                data=[
                    Model(
                        id="test-model",
                        created=1,
                        owned_by="test",
                        details={} if include_details else None,
                    )
                ]
            )

    class FakeEmbeddingsService:
        def uses_shared_vlm_runtime(self, _model: str) -> bool:
            return False

        def generate_embeddings(self, request):
            return EmbeddingResponse(
                data=[EmbeddingData(embedding=[0.0], index=0)],
                model=request.model,
                usage=EmbeddingUsage(prompt_tokens=0, total_tokens=0),
            )

    class FakeImagesService:
        def generate_images(self, _request):
            return [ImageObject(b64_json="ZmFrZQ==")]

    class FakeSTTService:
        async def transcribe(self, _request):
            return {"language": "en", "duration": 0.0, "text": "ok"}

    class FakeTTSService:
        def __init__(self, _model: str):
            pass

        async def generate_speech(self, request):
            return request.input.encode()

    def get_fake_responses_adapter():
        return FakeResponsesAdapter()

    def get_fake_text_model(*_args: Any):
        return FakeTextModel()

    def get_fake_anthropic_model(*_args: Any):
        return FakeAnthropicModel()

    def get_fake_models_service():
        return FakeModelsService()

    def get_fake_images_service():
        return FakeImagesService()

    def get_fake_stt_service():
        return FakeSTTService()

    def fake_maxsim_score(*_args: Any):
        return 1.0

    monkeypatch.setattr(responses_router, "get_adapter", get_fake_responses_adapter)
    monkeypatch.setattr(chat_router, "endpoint_runtime_session", _noop_runtime_session)
    monkeypatch.setattr(chat_router, "_create_text_model", get_fake_text_model)
    monkeypatch.setattr(
        anthropic_router, "endpoint_runtime_session", _noop_runtime_session
    )
    monkeypatch.setattr(
        anthropic_router, "_create_anthropic_model", get_fake_anthropic_model
    )
    monkeypatch.setattr(model_routes, "get_models_service", get_fake_models_service)
    monkeypatch.setattr(
        embeddings_router, "embeddings_service", FakeEmbeddingsService()
    )
    monkeypatch.setattr(
        visual_router.Qwen3VLEmbedder, "maxsim_score", fake_maxsim_score
    )
    monkeypatch.setattr(images_router, "get_images_service", get_fake_images_service)
    monkeypatch.setattr(stt_router, "STTService", get_fake_stt_service)
    monkeypatch.setattr(tts_router, "TTSService", FakeTTSService)
    monkeypatch.setattr(batch_coordinator, "_coordinators", {})

    headers = {"x-api-key": "test-key"}
    with inference_client_with_auth() as client:
        cases = [
            (
                "post",
                "/v1/responses",
                {
                    "headers": headers,
                    "json": {"model": "m", "input": "hi", "store": False},
                },
            ),
            (
                "post",
                "/v1/chat/completions",
                {"headers": headers, "json": _chat_payload()},
            ),
            (
                "post",
                "/anthropic/v1/messages",
                {"headers": headers, "json": _anthropic_payload()},
            ),
            ("get", "/v1/models", {"headers": headers}),
            (
                "post",
                "/v1/embeddings",
                {"headers": headers, "json": {"model": "m", "input": "hi"}},
            ),
            (
                "post",
                "/v1/maxsim",
                {
                    "headers": headers,
                    "json": {"query_embedding": [[1.0]], "doc_embedding": [[1.0]]},
                },
            ),
            (
                "post",
                "/v1/images/generations",
                {"headers": headers, "json": {"prompt": "hi"}},
            ),
            (
                "post",
                "/v1/audio/transcriptions",
                {
                    "headers": headers,
                    "files": {"file": ("sample.wav", b"fake wav", "audio/wav")},
                    "data": {"model": "whisper-test"},
                },
            ),
            (
                "post",
                "/v1/audio/speech",
                {"headers": headers, "json": {"model": "tts-test", "input": "hello"}},
            ),
            ("get", "/v1/batch/stats", {"headers": headers}),
        ]

        for method, path, kwargs in cases:
            response = getattr(client, method)(path, **kwargs)
            assert 200 <= response.status_code < 300, (path, response.text)


def test_business_endpoints_stay_open_at_security_level_zero(monkeypatch):
    """Backward compatibility: open mode does not demand credentials."""
    from fastapi.testclient import TestClient

    from mlx_batch_server.core.config import get_settings as get_core_settings
    from mlx_batch_server.main import create_app

    monkeypatch.delenv("SECURITY_LEVEL", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    get_core_settings.cache_clear()

    with TestClient(create_app()) as client:
        response = client.get("/v1/batch/stats")
        assert response.status_code == 200
