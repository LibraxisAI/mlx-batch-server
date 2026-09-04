"""RED contracts for the lifespan-owned Responses transport router."""

from __future__ import annotations

import ast
import asyncio
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from mlx_batch_server.auth.dependency import verify_auth, verify_websocket_auth
from mlx_batch_server.responses.runtime_router import (
    build_runtime_responses_router,
)
from mlx_batch_server.responses.transport import ResponseEventSource
from mlx_batch_server.runtime.events import (
    SequencedTurnEvent,
    TurnCompleted,
    TurnStarted,
)

OWNER = "resp-owner:v1:api-key:" + "a" * 64
TERMINAL = {
    "id": "resp_runtime_router",
    "object": "response",
    "created_at": 1,
    "model": "buddy",
    "status": "completed",
    "output": [
        {
            "id": "msg_runtime_router",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "hej"}],
        }
    ],
    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
}


async def _events() -> AsyncIterator[SequencedTurnEvent]:
    yield SequencedTurnEvent(
        sequence_number=0,
        event=TurnStarted(
            response_id="resp_runtime_router",
            model="buddy",
            created_at=1,
        ),
    )
    yield SequencedTurnEvent(
        sequence_number=1,
        event=TurnCompleted(finish_reason="stop"),
    )


async def _terminal() -> Mapping[str, Any]:
    return TERMINAL


def _source() -> ResponseEventSource:
    return ResponseEventSource(
        events=_events(),
        cancel=lambda _reason: None,
        terminal_response=asyncio.create_task(_terminal()),
    )


async def _events_without_terminal() -> AsyncIterator[SequencedTurnEvent]:
    yield SequencedTurnEvent(
        sequence_number=0,
        event=TurnStarted(
            response_id="resp_runtime_router",
            model="buddy",
            created_at=1,
        ),
    )


def _source_without_terminal_event() -> ResponseEventSource:
    return ResponseEventSource(
        events=_events_without_terminal(),
        cancel=lambda _reason: None,
        terminal_response=asyncio.create_task(_terminal()),
    )


class _Controller:
    def __init__(self) -> None:
        self.create_calls: list[tuple[dict[str, Any], str]] = []
        self.cancel_calls: list[tuple[str, str, str]] = []
        self.source_factory = _source

    async def create(
        self,
        payload: Mapping[str, Any],
        *,
        owner_id: str,
    ) -> ResponseEventSource:
        self.create_calls.append((dict(payload), owner_id))
        return self.source_factory()

    def cancel(self, response_id: str, *, owner_id: str, reason: str) -> None:
        self.cancel_calls.append((response_id, owner_id, reason))


class _Registry:
    def __init__(self) -> None:
        self.owners: list[str] = []

    def _owner(self, owner_id: str) -> None:
        self.owners.append(owner_id)
        assert owner_id == OWNER

    def get(self, _response_id: str, *, owner_id: str) -> dict[str, Any]:
        self._owner(owner_id)
        return dict(TERMINAL)

    def delete(self, response_id: str, *, owner_id: str) -> dict[str, Any]:
        self._owner(owner_id)
        return {"id": response_id, "object": "response.deleted", "deleted": True}

    def parent_messages(
        self,
        _response_id: str,
        *,
        owner_id: str,
    ) -> list[dict[str, Any]]:
        self._owner(owner_id)
        return [{"type": "message", "role": "user", "content": "hej"}]

    def wait_terminal(
        self,
        _response_id: str,
        _timeout_s: float,
        *,
        owner_id: str,
    ) -> dict[str, Any]:
        self._owner(owner_id)
        return dict(TERMINAL)


class _Runtime:
    def __init__(self) -> None:
        self.responses_controller = _Controller()
        self.response_registry = _Registry()


def _app(runtime: _Runtime) -> FastAPI:
    app = FastAPI()
    app.include_router(build_runtime_responses_router(runtime))

    def auth():
        return {"response_owner_id": OWNER}

    app.dependency_overrides[verify_auth] = auth
    app.dependency_overrides[verify_websocket_auth] = auth
    return app


def test_non_stream_returns_controller_terminal_and_verified_owner() -> None:
    runtime = _Runtime()
    client = TestClient(_app(runtime))

    response = client.post(
        "/v1/responses",
        json={"model": "buddy", "input": "hej"},
    )

    assert response.status_code == 200
    assert response.json() == TERMINAL
    assert runtime.responses_controller.create_calls == [
        ({"model": "buddy", "input": "hej"}, OWNER)
    ]


def test_sse_terminal_contains_the_same_full_response() -> None:
    runtime = _Runtime()
    client = TestClient(_app(runtime))

    response = client.post(
        "/v1/responses",
        json={"model": "buddy", "input": "hej", "stream": True},
    )

    assert response.status_code == 200
    assert '"type":"response.completed"' in response.text
    assert '"id":"resp_runtime_router"' in response.text
    assert '"text":"hej"' in response.text
    assert response.text.endswith("data: [DONE]\n\n")


def test_lifecycle_routes_use_one_verified_owner_and_terminal_writer() -> None:
    runtime = _Runtime()
    client = TestClient(_app(runtime))

    assert client.get("/v1/responses/resp_runtime_router").json() == TERMINAL
    assert client.get("/v1/responses/resp_runtime_router/input_items").json()[
        "data"
    ] == [
        {
            "id": "input_0",
            "type": "message",
            "role": "user",
            "content": "hej",
        }
    ]
    assert client.post("/v1/responses/resp_runtime_router/cancel").json() == TERMINAL
    assert client.delete("/v1/responses/resp_runtime_router").json() == {
        "id": "resp_runtime_router",
        "object": "response.deleted",
        "deleted": True,
    }
    assert runtime.responses_controller.cancel_calls == [
        ("resp_runtime_router", OWNER, "http_cancel_requested")
    ]
    assert runtime.response_registry.owners == [OWNER, OWNER, OWNER, OWNER]


def test_websocket_uses_the_same_owner_and_full_terminal_response() -> None:
    runtime = _Runtime()
    client = TestClient(_app(runtime))

    with client.websocket_connect("/v1/responses") as websocket:
        websocket.send_json(
            {
                "type": "response.create",
                "stream_id": "studio",
                "model": "buddy",
                "input": "hej",
            }
        )
        created = websocket.receive_json()
        completed = websocket.receive_json()

    assert created["type"] == "response.created"
    assert created["stream_id"] == "studio"
    assert completed["type"] == "response.completed"
    assert completed["stream_id"] == "studio"
    assert completed["response"] == TERMINAL
    assert runtime.responses_controller.create_calls == [
        ({"model": "buddy", "input": "hej"}, OWNER)
    ]


def test_websocket_renders_non_object_json_as_protocol_error() -> None:
    runtime = _Runtime()
    client = TestClient(_app(runtime))

    with client.websocket_connect("/v1/responses") as websocket:
        websocket.send_json("not-an-object")
        error = websocket.receive_json()

    assert error == {
        "type": "error",
        "status": 400,
        "error": {
            "type": "invalid_request_error",
            "code": "transport_protocol_error",
            "message": "WebSocket event must be a JSON object",
            "param": "type",
        },
    }


def test_http_and_sse_fail_closed_without_canonical_terminal_event() -> None:
    runtime = _Runtime()
    runtime.responses_controller.source_factory = _source_without_terminal_event
    client = TestClient(_app(runtime))

    response = client.post(
        "/v1/responses",
        json={"model": "buddy", "input": "hej"},
    )
    stream = client.post(
        "/v1/responses",
        json={"model": "buddy", "input": "hej", "stream": True},
    )

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert '"code":"responses_transport_failed"' in stream.text
    assert "data: [DONE]" not in stream.text


def test_runtime_router_has_no_legacy_responses_dependencies() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "src/mlx_batch_server/responses/runtime_router.py"
    )
    tree = ast.parse(path.read_text())
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_modules.update(
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    )

    assert not any(
        module.endswith((".adapter", ".store", ".context_builder"))
        for module in imported_modules
    )
