"""RED contracts for the lifespan-owned Responses transport router."""

from __future__ import annotations

import ast
import asyncio
import socket
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient
from openai import AsyncOpenAI, BadRequestError

from mlx_batch_server.auth.dependency import verify_auth, verify_websocket_auth
from mlx_batch_server.responses.runtime_router import (
    build_runtime_responses_router,
)
from mlx_batch_server.responses.transport import ResponseEventSource
from mlx_batch_server.runtime.events import (
    SequencedTurnEvent,
    TurnCancelled,
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

    def input_messages(
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


@asynccontextmanager
async def _serve(app: FastAPI) -> AsyncIterator[str]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(128)
    sock.setblocking(False)
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    while not server.started:
        await asyncio.sleep(0.01)
    try:
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, timeout=2)


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


@pytest.mark.asyncio
async def test_official_openai_sdk_parses_http_and_sse_contracts() -> None:
    runtime = _Runtime()
    transport = httpx.ASGITransport(app=_app(runtime))
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as http_client:
        client = AsyncOpenAI(
            api_key="test",
            base_url="http://test/v1",
            http_client=http_client,
        )
        response = await client.responses.create(model="buddy", input="hej")
        stream = await client.responses.create(
            model="buddy",
            input="hej",
            stream=True,
        )
        event_types = [event.type async for event in stream]

    assert response.id == "resp_runtime_router"
    assert response.status == "completed"
    assert response.output_text == "hej"
    assert event_types == ["response.created", "response.completed"]


@pytest.mark.asyncio
async def test_official_openai_sdk_receives_precise_unsupported_field_error() -> None:
    runtime = _Runtime()

    async def reject(
        payload: Mapping[str, Any],
        *,
        owner_id: str,
    ) -> ResponseEventSource:
        del payload, owner_id
        from mlx_batch_server.responses.runtime_mapper import ResponsesMappingError

        raise ResponsesMappingError(
            "unsupported Responses parameter: background",
            code="unsupported_parameter",
            param="background",
        )

    runtime.responses_controller.create = reject  # type: ignore[method-assign]
    transport = httpx.ASGITransport(app=_app(runtime))
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as http_client:
        client = AsyncOpenAI(
            api_key="test",
            base_url="http://test/v1",
            http_client=http_client,
        )
        with pytest.raises(BadRequestError) as raised:
            await client.responses.create(
                model="buddy",
                input="hej",
                extra_body={"background": True},
            )

    assert raised.value.code == "unsupported_parameter"
    assert raised.value.param == "background"


@pytest.mark.asyncio
async def test_official_openai_sdk_parses_websocket_steering_lifecycle() -> None:
    runtime = _Runtime()
    cancelled = asyncio.Event()
    create_calls: list[dict[str, Any]] = []

    async def parent_events() -> AsyncIterator[SequencedTurnEvent]:
        yield SequencedTurnEvent(
            0,
            TurnStarted(response_id="resp_parent", model="buddy", created_at=1),
        )
        await cancelled.wait()
        yield SequencedTurnEvent(1, TurnCancelled("steered"))

    async def successor_events() -> AsyncIterator[SequencedTurnEvent]:
        yield SequencedTurnEvent(
            0,
            TurnStarted(response_id="resp_successor", model="buddy", created_at=2),
        )
        yield SequencedTurnEvent(1, TurnCompleted("stop"))

    async def terminal(response_id: str, status: str) -> Mapping[str, Any]:
        result = dict(TERMINAL)
        result["id"] = response_id
        result["status"] = status
        result["output"] = []
        result["incomplete_details"] = (
            {"reason": "steered"} if status == "incomplete" else None
        )
        return result

    async def create(
        payload: Mapping[str, Any],
        *,
        owner_id: str,
    ) -> ResponseEventSource:
        assert owner_id == OWNER
        create_calls.append(dict(payload))
        if len(create_calls) == 1:
            return ResponseEventSource(
                parent_events(),
                cancel=lambda _reason: cancelled.set(),
                terminal_response=asyncio.create_task(
                    terminal("resp_parent", "incomplete")
                ),
                response_id="resp_parent",
            )
        return ResponseEventSource(
            successor_events(),
            terminal_response=asyncio.create_task(
                terminal("resp_successor", "completed")
            ),
            response_id="resp_successor",
        )

    runtime.responses_controller.create = create  # type: ignore[method-assign]
    async with _serve(_app(runtime)) as base_url:
        client = AsyncOpenAI(api_key="test", base_url=base_url)
        try:
            async with client.responses.connect(max_retries=0) as connection:
                await connection.send(
                    {
                        "type": "response.create",
                        "stream_id": "studio",
                        "model": "buddy",
                        "input": "original",
                        "instructions": "Buddy policy",
                    }
                )
                created = await connection.recv()
                await connection.send(
                    {
                        "type": "response.steer",
                        "previous_response_id": "resp_parent",
                        "input": "change direction",
                    }
                )
                accepted = await connection.recv()
                incomplete = await connection.recv()
                successor = await connection.recv()
        finally:
            await client.close()

    assert created.type == "response.created"
    assert accepted.type == "response.steer.accepted"
    assert accepted.stream_id == "studio"
    assert incomplete.type == "response.incomplete"
    assert incomplete.response.incomplete_details.reason == "steered"
    assert successor.type == "response.created"
    assert successor.response.id == "resp_successor"
    assert create_calls[1] == {
        "model": "buddy",
        "input": "change direction",
        "instructions": "Buddy policy",
        "previous_response_id": "resp_parent",
    }


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
