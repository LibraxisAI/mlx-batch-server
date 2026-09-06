"""Protocol-boundary truth for native Anthropic Messages."""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import time
from collections.abc import Iterator

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request as StarletteRequest

from mlx_batch_server.auth.hmac import compute_signature
from mlx_batch_server.auth.rate_limit import RateLimitMiddleware
from mlx_batch_server.chat.anthropic.errors import AnthropicAPIError
from mlx_batch_server.chat.anthropic.runtime_source import RuntimeAnthropicTurnSource
from mlx_batch_server.chat.anthropic.turn_source import (
    AnthropicTurn,
    clear_turn_source,
    register_turn_source,
)
from mlx_batch_server.core.config import get_settings
from mlx_batch_server.main import app
from mlx_batch_server.runtime.contracts import (
    BackendKind,
    GenerationRequest,
    RuntimeKey,
    TurnSink,
)
from mlx_batch_server.runtime.events import TurnFailed, TurnStarted
from mlx_batch_server.runtime.service import FirstWriterCancelToken, RuntimeStartService

anthropic_router = importlib.import_module("mlx_batch_server.chat.anthropic.router")
auth_dependency = importlib.import_module("mlx_batch_server.auth.dependency")
hmac_auth = importlib.import_module("mlx_batch_server.auth.hmac")
session_auth_module = importlib.import_module("mlx_batch_server.auth.session")

VERSION = "2023-06-01"
MESSAGES_PATHS = ("/anthropic/messages", "/anthropic/v1/messages")
BODY = {
    "model": "protocol-boundary-test",
    "max_tokens": 8,
    "messages": [{"role": "user", "content": "hi"}],
}
RUNTIME = RuntimeKey(
    model_id="private/physical-model",
    revision="private-revision",
    backend=BackendKind.FUSED_MTP_MLX,
)


class _FailingTurnSource:
    def stream(self, turn: AnthropicTurn):
        async def events():
            yield TurnStarted(
                response_id="anthropic_protocol_failure",
                model=turn.model_alias,
                created_at=1,
            )
            yield TurnFailed(
                error="runtime unavailable",
                code="overloaded_error",
                status_code=529,
            )

        return events()


class _IteratorExceptionTurnSource:
    def stream(self, turn: AnthropicTurn):
        async def events():
            yield TurnStarted(
                response_id="anthropic_iterator_failure",
                model=turn.model_alias,
                created_at=1,
            )
            raise RuntimeError("STREAM_EXCEPTION_SECRET_MARKER")

        return events()


class _ImmediateExceptionTurnSource:
    def stream(self, turn: AnthropicTurn):
        del turn
        raise AnthropicAPIError("SOURCE_SECRET_MARKER")


class _RawTurnFailedSource:
    def stream(self, turn: AnthropicTurn):
        async def events():
            yield TurnStarted(
                response_id="anthropic_raw_turn_failure",
                model=turn.model_alias,
                created_at=1,
            )
            yield TurnFailed(
                error="TURN_FAILED_SECRET_MARKER",
                code="overloaded_error",
                status_code=529,
            )

        return events()


class _ExplodingStarter(RuntimeStartService):
    def __init__(self) -> None:
        pass

    async def start(
        self,
        request: GenerationRequest,
        sink: TurnSink,
        *,
        cancel: FirstWriterCancelToken | None = None,
    ):
        del request, sink, cancel
        raise RuntimeError("STARTER_SECRET_MARKER")


class _ExplodingHandle:
    def __init__(self, response_id: str) -> None:
        self._response_id = response_id

    @property
    def response_id(self) -> str:
        return self._response_id

    def cancel(self, reason: str) -> bool:
        del reason
        return True

    async def wait_closed(self) -> None:
        raise RuntimeError("HANDLE_SECRET_MARKER")


class _HandleStarter(RuntimeStartService):
    def __init__(self) -> None:
        pass

    async def start(
        self,
        request: GenerationRequest,
        sink: TurnSink,
        *,
        cancel: FirstWriterCancelToken | None = None,
    ) -> _ExplodingHandle:
        assert cancel is not None
        sink.emit(
            TurnStarted(
                response_id=request.response_id,
                model=request.runtime.model_id,
                created_at=1,
            )
        )
        return _ExplodingHandle(request.response_id)


def _runtime_failure_source(kind: str):
    if kind == "source":
        return _ImmediateExceptionTurnSource(), "SOURCE_SECRET_MARKER"
    if kind == "iterator":
        return _IteratorExceptionTurnSource(), "STREAM_EXCEPTION_SECRET_MARKER"
    if kind == "turn_failed":
        return _RawTurnFailedSource(), "TURN_FAILED_SECRET_MARKER"
    if kind == "alias":

        def fail_alias(alias: str):
            del alias
            raise RuntimeError("ALIAS_SECRET_MARKER")

        return (
            RuntimeAnthropicTurnSource(
                starter=_HandleStarter(),
                resolve_model=fail_alias,
            ),
            "ALIAS_SECRET_MARKER",
        )
    if kind == "starter":
        return (
            RuntimeAnthropicTurnSource(
                starter=_ExplodingStarter(),
                resolve_model=lambda alias: (RUNTIME, alias),
            ),
            "STARTER_SECRET_MARKER",
        )
    if kind == "handle":
        return (
            RuntimeAnthropicTurnSource(
                starter=_HandleStarter(),
                resolve_model=lambda alias: (RUNTIME, alias),
            ),
            "HANDLE_SECRET_MARKER",
        )
    raise AssertionError(f"unknown hostile runtime kind: {kind}")


@pytest.fixture
def authenticated_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("SECURITY_LEVEL", "2")
    monkeypatch.setenv("API_KEY", "anthropic-boundary-secret")
    get_settings.cache_clear()
    try:
        yield TestClient(app)
    finally:
        monkeypatch.undo()
        get_settings.cache_clear()


def _assert_correlated_error(response, expected_type: str) -> dict:
    payload = response.json()
    assert payload["type"] == "error"
    assert payload["error"]["type"] == expected_type
    assert "detail" not in payload
    assert response.headers["request-id"]
    assert payload["request_id"] == response.headers["request-id"]
    assert response.headers.get_list("request-id") == [payload["request_id"]]
    return payload


def _raw_body() -> bytes:
    return json.dumps(BODY, separators=(",", ":")).encode()


def _hmac_headers(
    path: str,
    *,
    timestamp: int,
    signature: str | None = None,
) -> dict[str, str]:
    body = _raw_body()
    body_hash = hashlib.sha256(body).hexdigest()
    return {
        "anthropic-version": VERSION,
        "content-type": "application/json",
        "x-client-id": "protocol-client",
        "x-timestamp": str(timestamp),
        "x-signature": signature
        or compute_signature(
            "protocol-secret",
            timestamp,
            "POST",
            path,
            body_hash,
        ),
    }


def _post_hmac(client: TestClient, path: str, headers: dict[str, str]):
    return client.post(path, headers=headers, content=_raw_body())


@pytest.mark.parametrize("path", MESSAGES_PATHS)
@pytest.mark.parametrize(
    "credential",
    [None, "wrong-key"],
    ids=["missing", "bad"],
)
def test_auth_failures_use_anthropic_envelope_on_both_native_paths(
    authenticated_client: TestClient,
    path: str,
    credential: str | None,
) -> None:
    headers = {"anthropic-version": VERSION}
    if credential is not None:
        headers["x-api-key"] = credential

    response = authenticated_client.post(path, headers=headers, json=BODY)

    assert response.status_code == 401
    _assert_correlated_error(response, "authentication_error")
    assert response.headers["www-authenticate"] == "Bearer, ApiKey"


@pytest.mark.parametrize(
    "authorization",
    ["Basic dXNlcjpwYXNz", "Token alternate", "Bearer wrong-key"],
    ids=["basic", "alternate", "bad-bearer"],
)
def test_alternate_or_bad_authorization_cannot_escape_as_fastapi_detail(
    authenticated_client: TestClient,
    authorization: str,
) -> None:
    response = authenticated_client.post(
        "/anthropic/v1/messages",
        headers={
            "anthropic-version": VERSION,
            "authorization": authorization,
        },
        json=BODY,
    )

    assert response.status_code == 401
    _assert_correlated_error(response, "authentication_error")


@pytest.mark.parametrize(
    ("headers", "message_fragment"),
    [
        ({}, "header is required"),
        ({"anthropic-version": "2024-01-01"}, "unsupported anthropic-version"),
    ],
    ids=["missing", "unsupported"],
)
def test_version_is_required_and_closed(
    authenticated_client: TestClient,
    headers: dict[str, str],
    message_fragment: str,
) -> None:
    response = authenticated_client.post(
        "/anthropic/v1/messages",
        headers={"x-api-key": "anthropic-boundary-secret", **headers},
        json=BODY,
    )

    assert response.status_code == 400
    payload = _assert_correlated_error(response, "invalid_request_error")
    assert message_fragment in payload["error"]["message"]


@pytest.mark.parametrize("path", MESSAGES_PATHS)
@pytest.mark.parametrize(
    "versions",
    [
        (VERSION, "2099-01-01"),
        ("2099-01-01", VERSION),
        (VERSION, VERSION),
        (f"{VERSION}, 2099-01-01",),
    ],
    ids=["valid-invalid", "invalid-valid", "duplicate-valid", "comma-combined"],
)
def test_version_requires_one_physical_scalar_occurrence_regardless_of_order(
    authenticated_client: TestClient,
    path: str,
    versions: tuple[str, ...],
) -> None:
    headers = [("x-api-key", "anthropic-boundary-secret")]
    headers.extend(("anthropic-version", value) for value in versions)

    response = authenticated_client.post(path, headers=headers, json=BODY)

    assert response.status_code == 400
    _assert_correlated_error(response, "invalid_request_error")


@pytest.mark.parametrize(
    "beta",
    ["prompt-caching-2024-07-31", "future-a, future-b", "future-a,"],
    ids=["single", "multiple", "empty-token"],
)
def test_unimplemented_and_malformed_beta_tokens_fail_closed(
    authenticated_client: TestClient,
    beta: str,
) -> None:
    response = authenticated_client.post(
        "/anthropic/v1/messages",
        headers={
            "x-api-key": "anthropic-boundary-secret",
            "anthropic-version": VERSION,
            "anthropic-beta": beta,
        },
        json=BODY,
    )

    assert response.status_code == 400
    payload = _assert_correlated_error(response, "invalid_request_error")
    assert "anthropic-beta" in payload["error"]["message"]


@pytest.mark.parametrize("path", MESSAGES_PATHS)
@pytest.mark.parametrize(
    "betas",
    [("future-a", "future-b"), ("future-b", "future-a"), ("future-a", "")],
    ids=["forward", "reverse", "physical-empty"],
)
def test_duplicate_physical_beta_headers_fail_closed(
    authenticated_client: TestClient,
    path: str,
    betas: tuple[str, str],
) -> None:
    headers: list[tuple[str, str]] = [
        ("x-api-key", "anthropic-boundary-secret"),
        ("anthropic-version", VERSION),
    ]
    headers.extend(("anthropic-beta", value) for value in betas)

    response = authenticated_client.post(path, headers=headers, json=BODY)

    assert response.status_code == 400
    _assert_correlated_error(response, "invalid_request_error")


@pytest.mark.parametrize("path", MESSAGES_PATHS)
def test_unexpected_canonical_auth_failure_is_safe_and_correlated(
    authenticated_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
) -> None:
    async def fail_auth(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("AUTHORITY_SECRET_MARKER")

    monkeypatch.setattr(anthropic_router, "verify_auth", fail_auth)
    response = authenticated_client.post(
        path,
        headers={"anthropic-version": VERSION},
        json=BODY,
    )

    assert response.status_code == 500
    payload = _assert_correlated_error(response, "api_error")
    assert payload["error"]["message"] == "authentication service unavailable"
    assert "AUTHORITY_SECRET_MARKER" not in response.text


@pytest.mark.parametrize("path", MESSAGES_PATHS)
@pytest.mark.parametrize(
    ("status_code", "expected_status", "expected_type"),
    [
        (400, 400, "invalid_request_error"),
        (401, 401, "authentication_error"),
        (402, 402, "billing_error"),
        (403, 403, "permission_error"),
        (404, 404, "not_found_error"),
        (413, 413, "request_too_large"),
        (429, 429, "rate_limit_error"),
        (500, 500, "api_error"),
        (504, 504, "timeout_error"),
        (529, 529, "overloaded_error"),
        (418, 500, "api_error"),
    ],
)
def test_canonical_http_auth_failure_uses_closed_safe_mapping(
    authenticated_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    status_code: int,
    expected_status: int,
    expected_type: str,
) -> None:
    async def fail_auth(*args, **kwargs):
        del args, kwargs
        raise anthropic_router.HTTPException(
            status_code=status_code,
            detail="AUTH_HTTP_SECRET_MARKER",
            headers={
                "Request-ID": "foreign-upper",
                "rEqUeSt-Id": "foreign-mixed",
                "Retry-After": "7",
                "WWW-Authenticate": "Bearer",
            },
        )

    monkeypatch.setattr(anthropic_router, "verify_auth", fail_auth)
    response = authenticated_client.post(
        path,
        headers={"anthropic-version": VERSION},
        json=BODY,
    )

    assert response.status_code == expected_status
    payload = _assert_correlated_error(response, expected_type)
    assert payload["error"]["message"] != "AUTH_HTTP_SECRET_MARKER"
    assert "AUTH_HTTP_SECRET_MARKER" not in response.text
    assert response.headers["retry-after"] == "7"
    assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize("path", MESSAGES_PATHS)
@pytest.mark.parametrize("stream", [False, True], ids=["unary", "stream"])
@pytest.mark.parametrize(
    "failure_kind",
    ["source", "alias", "starter", "handle", "iterator", "turn_failed"],
)
def test_real_engine_runtime_failures_never_disclose_private_detail(
    authenticated_client: TestClient,
    path: str,
    stream: bool,
    failure_kind: str,
) -> None:
    source, marker = _runtime_failure_source(failure_kind)
    register_turn_source(source)
    try:
        response = authenticated_client.post(
            path,
            headers={
                "x-api-key": "anthropic-boundary-secret",
                "anthropic-version": VERSION,
            },
            json={**BODY, "stream": stream},
        )
    finally:
        clear_turn_source(source)

    assert marker not in response.text
    assert response.headers.get_list("request-id") == [response.headers["request-id"]]
    expected_status = 529 if failure_kind == "turn_failed" else 500
    expected_type = "overloaded_error" if failure_kind == "turn_failed" else "api_error"
    expected_message = (
        "the inference runtime is temporarily overloaded"
        if failure_kind == "turn_failed"
        else "message generation failed"
    )
    if not stream:
        assert response.status_code == expected_status
        payload = _assert_correlated_error(response, expected_type)
        assert payload["error"]["message"] == expected_message
        return

    assert response.status_code == 200
    error_frames = [
        frame
        for frame in response.text.split("\n\n")
        if frame.startswith("event: error\n")
    ]
    assert len(error_frames) == 1
    data_line = next(
        line for line in error_frames[0].splitlines() if line.startswith("data: ")
    )
    payload = json.loads(data_line.removeprefix("data: "))
    assert payload["error"] == {
        "type": expected_type,
        "message": expected_message,
    }
    assert payload["request_id"] == response.headers["request-id"]


@pytest.mark.parametrize("path", MESSAGES_PATHS)
def test_unexpected_body_decoder_failure_is_safe_and_correlated(
    authenticated_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
) -> None:
    async def fail_json(self):
        del self
        raise RuntimeError("BODY_SECRET_MARKER")

    monkeypatch.setattr(StarletteRequest, "json", fail_json)
    response = authenticated_client.post(
        path,
        headers={
            "x-api-key": "anthropic-boundary-secret",
            "anthropic-version": VERSION,
        },
        content=_raw_body(),
    )

    assert response.status_code == 500
    payload = _assert_correlated_error(response, "api_error")
    assert payload["error"]["message"] == "request body could not be read"
    assert "BODY_SECRET_MARKER" not in response.text


@pytest.mark.parametrize("path", MESSAGES_PATHS)
@pytest.mark.parametrize(
    ("timestamp", "signature", "secret", "expected_status"),
    [
        ("not-int", "bad", "protocol-secret", 400),
        (str(int(time.time())), "bad", None, 401),
        ("1", "bad", "protocol-secret", 401),
        (str(int(time.time())), "bad", "protocol-secret", 401),
    ],
    ids=["malformed-timestamp", "unknown-client", "stale", "bad-signature"],
)
def test_hmac_failure_matrix_uses_closed_anthropic_types(
    authenticated_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    timestamp: str,
    signature: str,
    secret: str | None,
    expected_status: int,
) -> None:
    async def read_secret(client_id: str) -> str | None:
        assert client_id == "protocol-client"
        return secret

    monkeypatch.setattr(hmac_auth, "_read_secret", read_secret)
    headers = {
        "anthropic-version": VERSION,
        "content-type": "application/json",
        "x-client-id": "protocol-client",
        "x-timestamp": timestamp,
        "x-signature": signature,
    }
    response = _post_hmac(authenticated_client, path, headers)

    assert response.status_code == expected_status
    expected_type = (
        "invalid_request_error" if expected_status == 400 else "authentication_error"
    )
    _assert_correlated_error(response, expected_type)


@pytest.mark.parametrize("path", MESSAGES_PATHS)
def test_hmac_backend_failure_is_safe_and_correlated(
    authenticated_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
) -> None:
    async def fail_secret(client_id: str) -> str | None:
        del client_id
        raise RuntimeError("HMAC_STORE_SECRET_MARKER")

    monkeypatch.setattr(hmac_auth, "_read_secret", fail_secret)
    headers = _hmac_headers(path, timestamp=int(time.time()))
    response = _post_hmac(authenticated_client, path, headers)

    assert response.status_code == 500
    _assert_correlated_error(response, "api_error")
    assert "HMAC_STORE_SECRET_MARKER" not in response.text


@pytest.mark.parametrize("path", MESSAGES_PATHS)
def test_hmac_body_read_failure_is_safe_and_correlated(
    authenticated_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
) -> None:
    async def fail_body(self):
        del self
        raise RuntimeError("HMAC_BODY_SECRET_MARKER")

    monkeypatch.setattr(StarletteRequest, "body", fail_body)
    headers = _hmac_headers(path, timestamp=int(time.time()))
    response = _post_hmac(authenticated_client, path, headers)

    assert response.status_code == 500
    _assert_correlated_error(response, "api_error")
    assert "HMAC_BODY_SECRET_MARKER" not in response.text


@pytest.mark.parametrize("path", MESSAGES_PATHS)
def test_valid_hmac_precedes_conflicting_api_key(
    authenticated_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
) -> None:
    async def read_secret(client_id: str) -> str | None:
        assert client_id == "protocol-client"
        return "protocol-secret"

    monkeypatch.setattr(hmac_auth, "_read_secret", read_secret)
    headers = _hmac_headers(path, timestamp=int(time.time()))
    headers["x-api-key"] = "conflicting-invalid-key"
    source = _FailingTurnSource()
    register_turn_source(source)
    try:
        response = _post_hmac(authenticated_client, path, headers)
    finally:
        clear_turn_source(source)

    assert response.status_code == 529
    _assert_correlated_error(response, "overloaded_error")


@pytest.mark.parametrize("path", MESSAGES_PATHS)
def test_session_rate_limit_preserves_retry_header_and_anthropic_type(
    authenticated_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
) -> None:
    monkeypatch.setenv("SESSION_AUTH_ENABLED", "true")
    get_settings.cache_clear()

    async def valid_session(session_id: str) -> dict:
        assert session_id == "session-token"
        return {"user_id": "session-user", "custom_metadata": {}}

    async def reject_rate(user_id: str, tier: str) -> bool:
        assert (user_id, tier) == ("session-user", "default")
        return False

    monkeypatch.setattr(
        session_auth_module.session_auth, "validate_session", valid_session
    )
    monkeypatch.setattr(
        session_auth_module.session_auth, "check_rate_limit", reject_rate
    )
    response = authenticated_client.post(
        path,
        headers={
            "authorization": "Bearer session-token",
            "anthropic-version": VERSION,
        },
        json=BODY,
    )

    assert response.status_code == 429
    _assert_correlated_error(response, "rate_limit_error")
    assert response.headers["retry-after"] == "60"


@pytest.mark.parametrize("path", MESSAGES_PATHS)
def test_valid_session_reaches_protocol_owner(
    authenticated_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
) -> None:
    monkeypatch.setenv("SESSION_AUTH_ENABLED", "true")
    get_settings.cache_clear()

    async def valid_session(session_id: str) -> dict:
        assert session_id == "session-token"
        return {"user_id": "session-user", "custom_metadata": {}}

    async def allow_rate(user_id: str, tier: str) -> bool:
        assert (user_id, tier) == ("session-user", "default")
        return True

    monkeypatch.setattr(
        session_auth_module.session_auth, "validate_session", valid_session
    )
    monkeypatch.setattr(
        session_auth_module.session_auth, "check_rate_limit", allow_rate
    )
    source = _FailingTurnSource()
    register_turn_source(source)
    try:
        response = authenticated_client.post(
            path,
            headers={
                "authorization": "Bearer session-token",
                "anthropic-version": VERSION,
            },
            json=BODY,
        )
    finally:
        clear_turn_source(source)

    assert response.status_code == 529
    _assert_correlated_error(response, "overloaded_error")


@pytest.mark.parametrize("path", MESSAGES_PATHS)
def test_session_backend_failure_is_safe_and_correlated(
    authenticated_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
) -> None:
    monkeypatch.setenv("SESSION_AUTH_ENABLED", "true")
    get_settings.cache_clear()

    async def fail_session(session_id: str, *, enforce_rate_limit: bool = True):
        del session_id, enforce_rate_limit
        raise RuntimeError("SESSION_STORE_SECRET_MARKER")

    monkeypatch.setattr(auth_dependency, "_resolve_session_auth", fail_session)
    response = authenticated_client.post(
        path,
        headers={
            "authorization": "Bearer session-token",
            "anthropic-version": VERSION,
        },
        json=BODY,
    )

    assert response.status_code == 500
    _assert_correlated_error(response, "api_error")
    assert "SESSION_STORE_SECRET_MARKER" not in response.text


@pytest.mark.parametrize("path", MESSAGES_PATHS)
def test_open_auth_mode_reaches_protocol_owner_without_credentials(
    authenticated_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
) -> None:
    monkeypatch.setenv("SECURITY_LEVEL", "0")
    monkeypatch.delenv("API_KEY", raising=False)
    get_settings.cache_clear()
    source = _FailingTurnSource()
    register_turn_source(source)
    try:
        response = authenticated_client.post(
            path,
            headers={"anthropic-version": VERSION},
            json=BODY,
        )
    finally:
        clear_turn_source(source)

    assert response.status_code == 529
    _assert_correlated_error(response, "overloaded_error")


def _rate_limited_test_app(*, requests_per_minute: int, concurrent_limit: int):
    application = FastAPI()

    async def accepted():
        return {"ok": True}

    for path in MESSAGES_PATHS:
        application.add_api_route(path, accepted, methods=["POST"])
    application.add_api_route("/v1/responses", accepted, methods=["POST"])
    application.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=requests_per_minute,
        concurrent_limit=concurrent_limit,
    )
    return application


@pytest.mark.asyncio
@pytest.mark.parametrize("path", MESSAGES_PATHS)
async def test_global_ordinary_rate_limit_is_protocol_aware(
    monkeypatch: pytest.MonkeyPatch,
    path: str,
) -> None:
    async def no_redis(self):
        del self

    monkeypatch.setattr(RateLimitMiddleware, "_get_redis", no_redis)
    application = _rate_limited_test_app(requests_per_minute=0, concurrent_limit=10)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application),
        base_url="http://test",
    ) as client:
        response = await client.post(path)

    assert response.status_code == 429
    _assert_correlated_error(response, "rate_limit_error")
    assert int(response.headers["retry-after"]) > 0
    assert response.headers["x-ratelimit-limit"] == "0"
    assert response.headers["x-ratelimit-remaining"] == "0"
    assert int(response.headers["x-ratelimit-reset"]) > 0


@pytest.mark.asyncio
async def test_global_rate_limit_keeps_non_anthropic_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_redis(self):
        del self

    monkeypatch.setattr(RateLimitMiddleware, "_get_redis", no_redis)
    application = _rate_limited_test_app(requests_per_minute=0, concurrent_limit=10)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application),
        base_url="http://test",
    ) as client:
        response = await client.post("/v1/responses")

    assert response.status_code == 429
    assert response.json()["error"] == "Too Many Requests"
    assert "request-id" not in response.headers


@pytest.mark.asyncio
@pytest.mark.parametrize("path", MESSAGES_PATHS)
async def test_global_concurrent_rate_limit_is_protocol_aware(
    monkeypatch: pytest.MonkeyPatch,
    path: str,
) -> None:
    async def no_redis(self):
        del self

    monkeypatch.setattr(RateLimitMiddleware, "_get_redis", no_redis)
    entered = asyncio.Event()
    release = asyncio.Event()
    application = FastAPI()

    async def held_request():
        entered.set()
        await release.wait()
        return {"ok": True}

    for route in MESSAGES_PATHS:
        application.add_api_route(route, held_request, methods=["POST"])
    application.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=100,
        concurrent_limit=1,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application),
        base_url="http://test",
    ) as client:
        first = asyncio.create_task(client.post(path))
        await asyncio.wait_for(entered.wait(), timeout=1)
        second = await client.post(path)
        release.set()
        assert (await first).status_code == 200

    assert second.status_code == 429
    _assert_correlated_error(second, "rate_limit_error")
    assert second.headers["retry-after"] == "5"


@pytest.mark.parametrize("path", MESSAGES_PATHS)
def test_runtime_sse_error_uses_transport_request_id(
    authenticated_client: TestClient,
    path: str,
) -> None:
    source = _FailingTurnSource()
    register_turn_source(source)
    try:
        response = authenticated_client.post(
            path,
            headers={
                "x-api-key": "anthropic-boundary-secret",
                "anthropic-version": VERSION,
            },
            json={**BODY, "stream": True},
        )
    finally:
        clear_turn_source(source)

    assert response.status_code == 200
    frames = [frame for frame in response.text.split("\n\n") if frame]
    error_frames = [frame for frame in frames if frame.startswith("event: error\n")]
    assert len(error_frames) == 1
    data_line = next(
        line for line in error_frames[0].splitlines() if line.startswith("data: ")
    )
    payload = json.loads(data_line.removeprefix("data: "))
    assert [frame.splitlines()[0] for frame in frames] == [
        "event: message_start",
        "event: error",
    ]
    assert payload["error"]["type"] == "overloaded_error"
    assert payload["request_id"] == response.headers["request-id"]
    assert response.text.count(payload["request_id"]) == 1


def test_openai_auth_failure_keeps_its_existing_envelope(
    authenticated_client: TestClient,
) -> None:
    response = authenticated_client.post(
        "/v1/responses",
        json={"model": "not-entered", "input": "not-entered"},
    )

    assert response.status_code == 401
    assert response.json() == {
        "detail": "Authentication required. Provide either API key or session token."
    }
    assert "request-id" not in response.headers
