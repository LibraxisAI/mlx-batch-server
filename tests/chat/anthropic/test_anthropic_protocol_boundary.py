"""Protocol-boundary truth for native Anthropic Messages."""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from mlx_batch_server.chat.anthropic.turn_source import (
    AnthropicTurn,
    clear_turn_source,
    register_turn_source,
)
from mlx_batch_server.core.config import get_settings
from mlx_batch_server.main import app
from mlx_batch_server.runtime.events import TurnFailed, TurnStarted

VERSION = "2023-06-01"
MESSAGES_PATHS = ("/anthropic/messages", "/anthropic/v1/messages")
BODY = {
    "model": "protocol-boundary-test",
    "max_tokens": 8,
    "messages": [{"role": "user", "content": "hi"}],
}


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
    return payload


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
