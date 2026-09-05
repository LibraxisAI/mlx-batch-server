"""Contracts for the canonical OpenAI Responses error boundary."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from openai.types.responses.response_error_event import ResponseErrorEvent

from mlx_batch_server.main import (
    _responses_request_validation_exception_handler,
    create_app,
)
from mlx_batch_server.responses import errors
from mlx_batch_server.responses.errors import (
    OpenAIError,
    render_http_error,
    render_sse_error,
)
from mlx_batch_server.responses.runtime_resolver import ManifestRuntimeResolver
from mlx_batch_server.runtime.contracts import BackendKind, RoleName, RoleSpec
from mlx_batch_server.runtime.readiness import ReadinessService
from mlx_batch_server.runtime.roles import RoleDirectory


def _error(**overrides: Any) -> OpenAIError:
    values: dict[str, Any] = {
        "message": "Invalid request.",
        "type": "invalid_request_error",
        "code": "invalid_request",
        "param": None,
        "status_code": 400,
        "sequence_number": None,
    }
    values.update(overrides)
    return OpenAIError(**values)


@pytest.mark.parametrize(
    ("field", "value", "exception_type"),
    [
        ("message", 1, TypeError),
        ("message", "", ValueError),
        ("type", None, TypeError),
        ("type", " ", ValueError),
        ("code", [], TypeError),
        ("code", "", ValueError),
        ("param", 1, TypeError),
        ("param", "", ValueError),
        ("status_code", True, TypeError),
        ("status_code", 399, ValueError),
        ("status_code", 600, ValueError),
        ("sequence_number", True, TypeError),
        ("sequence_number", -1, ValueError),
    ],
)
def test_typed_error_rejects_invalid_values(
    field: str,
    value: object,
    exception_type: type[Exception],
) -> None:
    with pytest.raises(exception_type):
        _error(**{field: value})


def test_typed_error_is_immutable() -> None:
    error = _error()

    with pytest.raises(FrozenInstanceError):
        error.message = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "error",
    [
        _error(message="Bad request.", param="input"),
        _error(
            message="Response not found.",
            code="response_not_found",
            param="response_id",
            status_code=404,
        ),
        _error(
            message="Response state conflict.",
            type="conflict_error",
            code="response_conflict",
            status_code=409,
        ),
        _error(
            message="Rate limit exceeded.",
            type="rate_limit_error",
            code="rate_limit_exceeded",
            status_code=429,
        ),
        _error(
            message="Internal server error.",
            type="server_error",
            code="internal_error",
            status_code=500,
        ),
    ],
)
def test_http_renderer_emits_exact_nested_envelope(error: OpenAIError) -> None:
    response = render_http_error(error)

    assert response.status_code == error.status_code
    assert json.loads(response.body) == {
        "error": {
            "message": error.message,
            "type": error.type,
            "param": error.param,
            "code": error.code,
        }
    }


def test_sse_renderer_is_flat_and_parses_with_official_sdk() -> None:
    error = _error(
        message="Response transport failed.",
        type="server_error",
        code="responses_transport_failed",
        status_code=500,
        sequence_number=7,
    )

    event = render_sse_error(error)
    parsed = ResponseErrorEvent.model_validate(event)

    assert set(event) == {"type", "code", "message", "param", "sequence_number"}
    assert event == {
        "type": "error",
        "code": "responses_transport_failed",
        "message": "Response transport failed.",
        "param": None,
        "sequence_number": 7,
    }
    assert parsed.type == "error"
    assert parsed.sequence_number == 7
    assert "error" not in event


def test_sse_renderer_requires_sequence_number() -> None:
    with pytest.raises(ValueError, match="sequence_number"):
        render_sse_error(_error())


def test_internal_exception_constructor_never_leaks_exception_text() -> None:
    hostile = RuntimeError(
        "secret=sk-test-sentinel body={'password':'hunter2'} "
        "/Users/founder/private/model.safetensors\nTraceback (most recent call last)"
    )

    error = OpenAIError.from_exception(
        hostile,
        status_code=503,
        code="responses_runtime_unavailable",
        sequence_number=9,
    )
    rendered = json.dumps(
        {
            "http": json.loads(render_http_error(error).body),
            "sse": render_sse_error(error),
        }
    )

    assert error.message == "Internal server error."
    assert error.code == "responses_runtime_unavailable"
    for sentinel in ("sk-test-sentinel", "hunter2", "/Users/", "Traceback"):
        assert sentinel not in rendered


def test_module_exposes_one_canonical_error_and_two_encoders() -> None:
    assert errors.__all__ == ["OpenAIError", "render_http_error", "render_sse_error"]


class _RuntimeManager:
    def __init__(self) -> None:
        self.acquired: list[RoleName] = []

    async def acquire_role(self, role: RoleName) -> object:
        self.acquired.append(role)
        return object()

    def role_capabilities(self, _role: RoleName) -> None:
        return None

    def role_stats(self, _role: RoleName) -> dict[str, Any]:
        return {}


class _CanonicalRuntime:
    def __init__(self) -> None:
        roles = RoleDirectory(
            (
                RoleSpec(
                    name=RoleName.MAIN,
                    port=8100,
                    requested_model="test/flash",
                    backend=BackendKind.FUSED_MTP_MLX,
                    pinned=True,
                ),
            )
        )
        manager = _RuntimeManager()
        self.responses = SimpleNamespace(
            responses_controller=object(),
            response_registry=object(),
            requires_single_worker=True,
            process_role=RoleName.MAIN,
            process_port=8100,
            role_manifest_sha256="manifest-sha",
            role_directory=roles,
            readiness_service=ReadinessService(roles),
            runtime_manager=manager,
            runtime_resolver=ManifestRuntimeResolver(
                roles,
                {"test/flash": RoleName.MAIN},
            ),
        )
        self.shutdown_deadlines: list[float] = []

    async def shutdown(self, *, deadline_s: float) -> None:
        self.shutdown_deadlines.append(deadline_s)


RESPONSES_BODY_PATHS = (
    "/v1/responses",
    "/v1/responses/compact",
    "/v1/responses/input_tokens",
)


@pytest.mark.parametrize("path", RESPONSES_BODY_PATHS)
@pytest.mark.parametrize(
    ("raw_body", "message", "code"),
    [
        (b'{"model":', "Malformed JSON request body.", "invalid_json"),
        (b"[]", "Request body must be a JSON object.", "invalid_request"),
        (b"", "Request body is required.", "invalid_request"),
    ],
)
def test_responses_raw_body_failures_use_canonical_http_contract(
    path: str,
    raw_body: bytes,
    message: str,
    code: str,
) -> None:
    app = create_app(responses_runtime=_CanonicalRuntime())

    with TestClient(app) as client:
        response = client.post(
            path,
            content=raw_body,
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "param": "body",
            "code": code,
        }
    }
    assert "detail" not in response.json()


def test_create_app_registers_the_scoped_validation_handler() -> None:
    app = create_app(responses_runtime=_CanonicalRuntime())

    assert (
        app.exception_handlers[RequestValidationError]
        is _responses_request_validation_exception_handler
    )


def test_fastapi_field_validation_uses_the_canonical_responses_contract() -> None:
    app = create_app()

    with TestClient(app) as client:
        response = client.post("/v1/responses", json={})

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "message": "Missing required parameter: 'model'.",
            "type": "invalid_request_error",
            "param": "model",
            "code": "invalid_request",
        }
    }


def test_adjacent_route_error_contracts_are_untouched() -> None:
    app = create_app(responses_runtime=_CanonicalRuntime())

    with TestClient(app) as client:
        model_control = client.post("/v1/models/load", json={})
        admin = client.post("/api/admin/models/load", json={})
        anthropic = client.post(
            "/anthropic/v1/messages",
            content=b'{"model":',
            headers={"content-type": "application/json"},
        )
        health = client.get("/health")

    assert model_control.status_code == 422
    assert set(model_control.json()) == {"detail"}
    assert admin.status_code == 422
    assert set(admin.json()) == {"detail"}
    assert anthropic.status_code == 400
    assert anthropic.json()["type"] == "error"
    assert anthropic.json()["error"]["type"] == "invalid_request_error"
    assert "request_id" in anthropic.json()
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert "error" not in health.json()
