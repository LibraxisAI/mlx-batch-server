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
from openai import AsyncOpenAI, BadRequestError, OpenAI

from mlx_batch_server.auth.dependency import verify_auth, verify_websocket_auth
from mlx_batch_server.responses.runtime_router import (
    build_runtime_responses_router,
)
from mlx_batch_server.responses.transport import (
    SNAPSHOT_EMBEDDING_EVENT_TYPES,
    PublishedResponseEvent,
    ResponseEventSource,
    ResponseSnapshotBuilder,
)
from mlx_batch_server.runtime.events import (
    SequencedTurnEvent,
    TurnCancelled,
    TurnCompleted,
    TurnEvent,
    TurnStarted,
    UsageUpdate,
)


def _published(
    builder: ResponseSnapshotBuilder,
    sequence_number: int,
    event: TurnEvent,
) -> PublishedResponseEvent:
    """Fold one event exactly as the controller publishes it."""

    builder.observe(event)
    snapshot = (
        builder.snapshot(event)
        if isinstance(event, SNAPSHOT_EMBEDDING_EVENT_TYPES)
        else None
    )
    return PublishedResponseEvent(sequence_number, event, snapshot)


async def _publish(*events: TurnEvent) -> AsyncIterator[PublishedResponseEvent]:
    builder = ResponseSnapshotBuilder()
    for sequence_number, event in enumerate(events):
        yield _published(builder, sequence_number, event)


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


def _events() -> AsyncIterator[PublishedResponseEvent]:
    return _publish(
        TurnStarted(
            response_id="resp_runtime_router",
            model="buddy",
            created_at=1,
        ),
        TurnCompleted(finish_reason="stop"),
    )


async def _terminal() -> Mapping[str, Any]:
    return TERMINAL


def _source() -> ResponseEventSource:
    return ResponseEventSource(
        events=_events(),
        cancel=lambda _reason: None,
        terminal_response=asyncio.create_task(_terminal()),
    )


def _events_without_terminal() -> AsyncIterator[PublishedResponseEvent]:
    return _publish(
        TurnStarted(
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

    async def create_background(
        self,
        payload: Mapping[str, Any],
        *,
        owner_id: str,
    ) -> Mapping[str, Any]:
        self.create_calls.append((dict(payload), owner_id))
        return {**TERMINAL, "status": "queued", "background": True, "output": []}

    def cancel(self, response_id: str, *, owner_id: str, reason: str) -> None:
        self.cancel_calls.append((response_id, owner_id, reason))


class _Registry:
    def __init__(self) -> None:
        self.owners: list[str] = []
        self.background = False
        self.items: list[dict[str, Any]] = [
            {
                "id": "input_stable_0",
                "type": "message",
                "role": "user",
                "content": "hej",
            }
        ]

    def _owner(self, owner_id: str) -> None:
        self.owners.append(owner_id)
        assert owner_id == OWNER

    def get(self, _response_id: str, *, owner_id: str) -> dict[str, Any]:
        self._owner(owner_id)
        response = dict(TERMINAL)
        if self.background:
            response["background"] = True
        return response

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

    def input_items(
        self,
        _response_id: str,
        *,
        owner_id: str,
    ) -> list[dict[str, Any]]:
        self._owner(owner_id)
        return [dict(item) for item in self.items]

    def wait_terminal(
        self,
        _response_id: str,
        _timeout_s: float,
        *,
        owner_id: str,
    ) -> dict[str, Any]:
        self._owner(owner_id)
        return dict(TERMINAL)


class _Operations:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any], str]] = []

    async def count_input_tokens(
        self,
        payload: Mapping[str, Any],
        *,
        owner_id: str,
    ) -> Mapping[str, Any]:
        self.calls.append(("input_tokens", dict(payload), owner_id))
        return {"object": "response.input_tokens", "input_tokens": 17}

    async def compact(
        self,
        payload: Mapping[str, Any],
        *,
        owner_id: str,
    ) -> Mapping[str, Any]:
        self.calls.append(("compact", dict(payload), owner_id))
        return {
            "id": "resp_compact_test",
            "created_at": 1,
            "object": "response.compaction",
            "output": [
                {
                    "id": "cmp_test",
                    "type": "compaction",
                    "encrypted_content": "mlxbr1.test",
                    "created_by": "mlx-batch-server",
                }
            ],
            "usage": {
                "input_tokens": 17,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 0,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 17,
            },
        }


class _Runtime:
    def __init__(self) -> None:
        self.responses_controller = _Controller()
        self.response_registry = _Registry()
        self.responses_operations = _Operations()


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


def test_official_sync_openai_sdk_parses_http_and_sse_contracts() -> None:
    runtime = _Runtime()
    http_client = TestClient(_app(runtime))
    client = OpenAI(
        api_key="test",
        base_url="http://testserver/v1",
        http_client=http_client,
        max_retries=0,
    )
    try:
        response = client.responses.create(model="buddy", input="hej")
        stream = client.responses.create(model="buddy", input="hej", stream=True)
        event_types = [event.type for event in stream]
    finally:
        client.close()

    assert response.id == "resp_runtime_router"
    assert response.output_text == "hej"
    assert event_types == ["response.created", "response.completed"]


@pytest.mark.asyncio
async def test_official_openai_sdk_parses_lifecycle_and_local_operations() -> None:
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
        retrieved = await client.responses.retrieve("resp_runtime_router")
        runtime.response_registry.background = True
        cancelled = await client.responses.cancel("resp_runtime_router")
        input_items = await client.responses.input_items.list("resp_runtime_router")
        deleted = await client.responses.delete("resp_runtime_router")
        counted = await client.responses.input_tokens.count(
            model="buddy",
            input="hej",
        )
        compacted = await client.responses.compact(model="buddy", input="hej")

    assert retrieved.id == "resp_runtime_router"
    assert cancelled.status == "completed"
    assert input_items.data[0].role == "user"
    # openai-python 2.32 models Responses DELETE as ``None`` even though the
    # wire endpoint returns the documented response.deleted JSON object. The
    # raw transport assertion below protects that body independently.
    assert deleted is None
    assert counted.object == "response.input_tokens"
    assert counted.input_tokens == 17
    assert compacted.object == "response.compaction"
    assert compacted.output[-1].type == "compaction"
    assert runtime.responses_operations.calls == [
        ("input_tokens", {"model": "buddy", "input": "hej"}, OWNER),
        ("compact", {"model": "buddy", "input": "hej"}, OWNER),
    ]


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
            "unsupported Responses parameter: max_tool_calls",
            code="unsupported_parameter",
            param="max_tool_calls",
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
                max_tool_calls=1,
            )

    assert raised.value.code == "unsupported_parameter"
    assert raised.value.param == "max_tool_calls"


@pytest.mark.asyncio
async def test_official_openai_sdk_parses_websocket_steering_lifecycle() -> None:
    runtime = _Runtime()
    cancelled = asyncio.Event()
    create_calls: list[dict[str, Any]] = []

    async def parent_events() -> AsyncIterator[PublishedResponseEvent]:
        builder = ResponseSnapshotBuilder()
        yield _published(
            builder,
            0,
            TurnStarted(response_id="resp_parent", model="buddy", created_at=1),
        )
        await cancelled.wait()
        yield _published(builder, 1, TurnCancelled("steered"))

    def successor_events() -> AsyncIterator[PublishedResponseEvent]:
        return _publish(
            TurnStarted(response_id="resp_successor", model="buddy", created_at=2),
            TurnCompleted("stop"),
        )

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


def test_pending_steer_waits_for_client_tool_output_and_starts_once() -> None:
    runtime = _Runtime()
    create_calls: list[dict[str, Any]] = []

    def events(
        response_id: str,
        finish_reason: str,
    ) -> AsyncIterator[PublishedResponseEvent]:
        return _publish(
            TurnStarted(response_id=response_id, model="buddy", created_at=1),
            TurnCompleted(finish_reason),
        )

    async def terminal(
        response_id: str,
        output: list[dict[str, Any]],
    ) -> Mapping[str, Any]:
        result = dict(TERMINAL)
        result["id"] = response_id
        result["output"] = output
        return result

    async def create(
        payload: Mapping[str, Any],
        *,
        owner_id: str,
    ) -> ResponseEventSource:
        assert owner_id == OWNER
        create_calls.append(dict(payload))
        if len(create_calls) == 1:
            output = [
                {
                    "id": "fc_1",
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "lookup_case",
                    "arguments": "{}",
                    "status": "completed",
                }
            ]
            return ResponseEventSource(
                events("resp_tool_stop", "tool_calls"),
                terminal_response=asyncio.create_task(
                    terminal("resp_tool_stop", output)
                ),
                response_id="resp_tool_stop",
            )
        return ResponseEventSource(
            events("resp_after_tool", "stop"),
            terminal_response=asyncio.create_task(terminal("resp_after_tool", [])),
            response_id="resp_after_tool",
        )

    runtime.responses_controller.create = create  # type: ignore[method-assign]
    client = TestClient(_app(runtime))
    with client.websocket_connect("/v1/responses") as websocket:
        websocket.send_json(
            {
                "type": "response.create",
                "stream_id": "studio",
                "model": "buddy",
                "input": "find the case",
            }
        )
        assert websocket.receive_json()["type"] == "response.created"
        stopped = websocket.receive_json()
        assert stopped["type"] == "response.completed"

        websocket.send_json(
            {
                "type": "response.steer",
                "previous_response_id": "resp_tool_stop",
                "input": "then summarize it",
            }
        )
        accepted = websocket.receive_json()
        pending = websocket.receive_json()
        assert accepted["type"] == "response.steer.accepted"
        assert pending["type"] == "response.steer.pending"
        assert pending["sequence_number"] == accepted["sequence_number"] + 1
        assert len(create_calls) == 1

        websocket.send_json(
            {
                "type": "response.create",
                "stream_id": "studio",
                "model": "buddy",
                "previous_response_id": "resp_tool_stop",
                "input": {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "case data",
                },
            }
        )
        assert websocket.receive_json()["type"] == "response.created"
        assert websocket.receive_json()["type"] == "response.completed"

    assert len(create_calls) == 2
    assert create_calls[1]["input"] == [
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "case data",
        },
        "then summarize it",
    ]


def test_lifecycle_routes_use_one_verified_owner_and_terminal_writer() -> None:
    runtime = _Runtime()
    client = TestClient(_app(runtime))

    assert client.get("/v1/responses/resp_runtime_router").json() == TERMINAL
    assert client.get("/v1/responses/resp_runtime_router/input_items").json()[
        "data"
    ] == [
        {
            "id": "input_stable_0",
            "type": "message",
            "role": "user",
            "content": "hej",
        }
    ]
    not_background = client.post("/v1/responses/resp_runtime_router/cancel")
    assert not_background.status_code == 409
    assert not_background.json()["error"] == {
        "message": (
            "response 'resp_runtime_router' was not created in background mode"
        ),
        "type": "invalid_request_error",
        "param": None,
        "code": "response_not_cancellable",
    }
    assert runtime.responses_controller.cancel_calls == []

    runtime.response_registry.background = True
    assert client.post("/v1/responses/resp_runtime_router/cancel").json() == TERMINAL
    assert client.delete("/v1/responses/resp_runtime_router").json() == {
        "id": "resp_runtime_router",
        "object": "response.deleted",
        "deleted": True,
    }
    assert runtime.responses_controller.cancel_calls == [
        ("resp_runtime_router", OWNER, "http_cancel_requested")
    ]
    assert runtime.response_registry.owners == [
        OWNER,
        OWNER,
        OWNER,
        OWNER,
        OWNER,
        OWNER,
    ]


def test_static_route_census_precedes_the_dynamic_response_id_route() -> None:
    runtime = _Runtime()
    client = TestClient(_app(runtime))

    compacted = client.post(
        "/v1/responses/compact",
        json={"model": "buddy", "input": "hej"},
    )
    counted = client.post(
        "/v1/responses/input_tokens",
        json={"model": "buddy", "input": "hej"},
    )
    capabilities = client.get("/v1/responses/capabilities")
    arbitrary = client.get("/v1/responses/resp_arbitrary")

    assert compacted.status_code == 200
    assert compacted.json()["object"] == "response.compaction"
    assert counted.status_code == 200
    assert counted.json()["object"] == "response.input_tokens"
    assert capabilities.status_code == 200
    assert capabilities.json()["version"] == "responses.request.capability/1"
    assert arbitrary.json()["id"] == "resp_runtime_router"


def test_malformed_bodies_and_retrieve_options_use_canonical_errors() -> None:
    runtime = _Runtime()
    client = TestClient(_app(runtime))

    malformed = client.post(
        "/v1/responses",
        content=b'{"model":',
        headers={"content-type": "application/json"},
    )
    non_object = client.post("/v1/responses", json=["not", "an", "object"])
    unsupported = client.get(
        "/v1/responses/resp_runtime_router",
        params={"starting_after": 1},
    )
    unsupported_include = client.get(
        "/v1/responses/resp_runtime_router",
        params={"include": "reasoning.encrypted_content"},
    )
    ordinary = client.get(
        "/v1/responses/resp_runtime_router",
        params={"stream": "false"},
    )

    assert malformed.status_code == 400
    assert malformed.json() == {
        "error": {
            "message": "request body must contain valid JSON",
            "type": "invalid_request_error",
            "param": None,
            "code": "invalid_json",
        }
    }
    assert non_object.status_code == 400
    assert "detail" not in non_object.json()
    assert non_object.json()["error"]["code"] == "invalid_responses_request"
    assert unsupported.status_code == 400
    assert unsupported.json()["error"]["code"] == "unsupported_parameter"
    assert unsupported.json()["error"]["param"] == "starting_after"
    assert unsupported_include.status_code == 400
    assert unsupported_include.json()["error"]["param"] == "include[0]"
    assert ordinary.status_code == 200


@pytest.mark.asyncio
async def test_official_sync_and_async_auto_pagination_is_gap_free() -> None:
    runtime = _Runtime()
    runtime.response_registry.items = [
        {
            "id": f"input_stable_{index:02d}",
            "type": "message",
            "role": "user",
            "content": f"item {index}",
        }
        for index in range(27)
    ]
    expected = [item["id"] for item in runtime.response_registry.items]

    sync_http = TestClient(_app(runtime))
    sync_client = OpenAI(
        api_key="test",
        base_url="http://testserver/v1",
        http_client=sync_http,
        max_retries=0,
    )
    try:
        sync_ids = [
            item.id
            for item in sync_client.responses.input_items.list(
                "resp_runtime_router",
                limit=7,
                order="asc",
            )
        ]
    finally:
        sync_client.close()

    transport = httpx.ASGITransport(app=_app(runtime))
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as async_http:
        async_client = AsyncOpenAI(
            api_key="test",
            base_url="http://test/v1",
            http_client=async_http,
            max_retries=0,
        )
        async_ids = [
            item.id
            async for item in async_client.responses.input_items.list(
                "resp_runtime_router",
                limit=6,
                order="desc",
            )
        ]

    assert sync_ids == expected
    assert async_ids == list(reversed(expected))
    assert len(set(sync_ids)) == len(expected)
    assert len(set(async_ids)) == len(expected)


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
    error_line = next(
        line
        for line in stream.text.splitlines()
        if line.startswith("data: {") and '"type":"error"' in line
    )
    error = httpx.Response(
        200,
        content=error_line.removeprefix("data: "),
    ).json()
    assert set(error) == {"type", "code", "message", "param", "sequence_number"}
    assert error["sequence_number"] == 1
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


# --- Lossless official tool continuation over the real controller stack -------
#
# These contracts refuse a fake controller: they wire the real registry, the
# real canonical mapper and the real controller behind the shipped router, and
# assert the GenerationRequest the backend actually receives.

CALL_ID = "call_weather_1"
CALL_ARGUMENTS = '{"city":"Kielce"}'
TOOL_RESULT = '{"temp_c":18}'
_CALL_ITEM = {
    "id": "fc_weather_1",
    "type": "function_call",
    "status": "completed",
    "call_id": CALL_ID,
    "name": "get_weather",
    "arguments": CALL_ARGUMENTS,
}


class _ScriptedProjection:
    """Terminal envelopes are scripted; the projector itself is out of scope."""

    def __init__(self, response_id: str, output: list[dict[str, Any]]) -> None:
        self._response_id = response_id
        self._output = output

    def observe(self, event: SequencedTurnEvent) -> None:
        del event

    def terminal_envelope(self) -> Mapping[str, Any]:
        return {
            "id": self._response_id,
            "object": "response",
            "created_at": 1,
            "model": "buddy",
            "status": "completed",
            "output": self._output,
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        }


class _ToolHandle:
    def __init__(self, response_id: str) -> None:
        self._response_id = response_id
        self.closed = asyncio.Event()

    @property
    def response_id(self) -> str:
        return self._response_id

    def cancel(self, reason: str) -> None:
        del reason
        self.closed.set()

    async def wait_closed(self) -> None:
        await self.closed.wait()


class _CapturingStarter:
    """Stand in for the model runtime and keep every canonical request seen."""

    def __init__(self) -> None:
        self.requests: list[Any] = []

    async def start(self, request: Any, sink: Any, *, cancel: Any) -> _ToolHandle:
        del cancel
        self.requests.append(request)
        sink.emit(
            TurnStarted(
                response_id=request.response_id,
                model=request.runtime.model_id,
                created_at=1,
            )
        )
        sink.emit(TurnCompleted("stop"))
        handle = _ToolHandle(request.response_id)
        handle.closed.set()
        return handle


class _ToolRuntime:
    """Real registry + real canonical mapper + real controller behind the router."""

    def __init__(self) -> None:
        from mlx_batch_server.responses.controller import ResponsesController
        from mlx_batch_server.responses.registry import ResponseRegistry
        from mlx_batch_server.responses.runtime_mapper import (
            CanonicalResponsesMapper,
        )
        from mlx_batch_server.runtime.contracts import BackendKind, RuntimeKey

        self.starter = _CapturingStarter()
        self.rounds = 0

        def resolve_runtime(**kwargs: Any) -> RuntimeKey:
            return RuntimeKey(
                model_id=kwargs["model"],
                revision=kwargs["revision"],
                adapter_path=kwargs["adapter_path"],
                draft_model_id=kwargs["draft_model_id"],
                backend=BackendKind.FUSED_MTP_MLX,
            )

        def projection_factory(prepared: Any) -> _ScriptedProjection:
            self.rounds += 1
            output: list[dict[str, Any]] = (
                [dict(_CALL_ITEM)]
                if self.rounds == 1
                else [
                    {
                        "id": "msg_2",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "18 stopni."}],
                    }
                ]
            )
            return _ScriptedProjection(prepared.request.response_id, output)

        self.response_registry = ResponseRegistry()
        self.responses_controller = ResponsesController(
            registry=self.response_registry,
            mapper=CanonicalResponsesMapper(
                resolve_runtime=resolve_runtime,
                projection_factory=projection_factory,
            ),
            starter=self.starter,
        )


def _assert_continuation_history(runtime: _ToolRuntime) -> None:
    """The successor must reach the backend as an ordered call/result pair."""

    assert len(runtime.starter.requests) == 2
    messages = [dict(message) for message in runtime.starter.requests[1].messages]

    assert [message["role"] for message in messages] == ["user", "assistant", "tool"]
    # The call reaches the backend as the typed item; its rendered text belongs
    # to canonical history and never becomes visible assistant content.
    assert messages[1]["type"] == "function_call"
    assert messages[1]["content"] == ()
    assert messages[1]["call_id"] == CALL_ID
    assert messages[1]["name"] == "get_weather"
    assert messages[1]["arguments"] == CALL_ARGUMENTS
    assert messages[2]["type"] == "function_call_output"
    assert messages[2]["call_id"] == CALL_ID
    assert messages[2]["output"] == TOOL_RESULT

    # Canonical history keeps the typed call item, not only its rendering.
    first = runtime.response_registry.parent_messages(
        runtime.starter.requests[0].response_id,
        owner_id=OWNER,
    )
    assert first[-1] == {
        "type": "function_call",
        "role": "assistant",
        "call_id": CALL_ID,
        "name": "get_weather",
        "arguments": CALL_ARGUMENTS,
        "id": "fc_weather_1",
        "status": "completed",
    }


def test_http_successor_receives_the_official_call_with_its_result() -> None:
    runtime = _ToolRuntime()
    client = TestClient(_app(runtime))

    first = client.post(
        "/v1/responses",
        json={"model": "buddy", "input": "Jaka pogoda w Kielcach?"},
    )
    assert first.status_code == 200
    first_id = first.json()["id"]

    second = client.post(
        "/v1/responses",
        json={
            "model": "buddy",
            "previous_response_id": first_id,
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": CALL_ID,
                    "output": TOOL_RESULT,
                }
            ],
        },
    )
    assert second.status_code == 200

    _assert_continuation_history(runtime)


def test_websocket_successor_shares_the_same_canonical_mapping_and_registry() -> None:
    runtime = _ToolRuntime()
    client = TestClient(_app(runtime))

    with client.websocket_connect("/v1/responses") as websocket:
        websocket.send_json(
            {
                "type": "response.create",
                "stream_id": "studio",
                "model": "buddy",
                "input": "Jaka pogoda w Kielcach?",
            }
        )
        assert websocket.receive_json()["type"] == "response.created"
        completed = websocket.receive_json()
        assert completed["type"] == "response.completed"
        first_id = completed["response"]["id"]

        websocket.send_json(
            {
                "type": "response.create",
                "stream_id": "studio",
                "model": "buddy",
                "previous_response_id": first_id,
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": CALL_ID,
                        "output": TOOL_RESULT,
                    }
                ],
            }
        )
        assert websocket.receive_json()["type"] == "response.created"
        assert websocket.receive_json()["type"] == "response.completed"

    _assert_continuation_history(runtime)


def test_http_successor_rejects_a_tool_result_the_lineage_cannot_explain() -> None:
    runtime = _ToolRuntime()
    client = TestClient(_app(runtime))

    first = client.post(
        "/v1/responses",
        json={"model": "buddy", "input": "Jaka pogoda w Kielcach?"},
    )
    first_id = first.json()["id"]

    orphan = client.post(
        "/v1/responses",
        json={
            "model": "buddy",
            "previous_response_id": first_id,
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_foreign",
                    "output": TOOL_RESULT,
                }
            ],
        },
    )

    assert orphan.status_code == 400
    body = orphan.json()["error"]
    assert body["code"] == "invalid_responses_request"
    assert body["param"] == "input[0].call_id"
    assert len(runtime.starter.requests) == 1


class _DeferredHandle:
    def __init__(self, response_id: str, sink: Any) -> None:
        self._response_id = response_id
        self.sink = sink
        self.closed = asyncio.Event()
        self.cancel_reasons: list[str] = []

    @property
    def response_id(self) -> str:
        return self._response_id

    def cancel(self, reason: str) -> None:
        if self.closed.is_set():
            return
        self.cancel_reasons.append(reason)
        self.sink.emit(TurnCancelled(reason))
        self.closed.set()

    async def wait_closed(self) -> None:
        await self.closed.wait()


class _DeferredStarter:
    def __init__(self, *, fail: bool = False) -> None:
        self.entered = asyncio.Event()
        self.fail = fail
        self.handles: dict[str, _DeferredHandle] = {}

    async def start(self, request: Any, sink: Any, *, cancel: Any) -> _DeferredHandle:
        del cancel
        sink.emit(
            TurnStarted(
                response_id=request.response_id,
                model=request.runtime.model_id,
                created_at=17,
                requested_model=request.metadata.get("requested_model"),
            )
        )
        self.entered.set()
        if self.fail:
            raise RuntimeError("scripted backend failure")
        handle = _DeferredHandle(request.response_id, sink)
        self.handles[request.response_id] = handle
        return handle

    def complete(self, response_id: str) -> None:
        handle = self.handles[response_id]
        handle.sink.emit(TurnCompleted("stop"))
        handle.closed.set()


class _DeferredRuntime:
    def __init__(self, *, fail: bool = False) -> None:
        from mlx_batch_server.responses.controller import ResponsesController
        from mlx_batch_server.responses.registry import ResponseRegistry
        from mlx_batch_server.responses.runtime_mapper import CanonicalResponsesMapper
        from mlx_batch_server.responses.runtime_projection import (
            create_runtime_projection,
        )
        from mlx_batch_server.runtime.contracts import RuntimeKey

        self.response_registry = ResponseRegistry()
        self.starter = _DeferredStarter(fail=fail)

        def resolve_runtime(**kwargs: Any) -> RuntimeKey:
            return RuntimeKey(model_id=kwargs["model"])

        self.responses_controller = ResponsesController(
            registry=self.response_registry,
            mapper=CanonicalResponsesMapper(
                resolve_runtime=resolve_runtime,
                projection_factory=create_runtime_projection,
            ),
            starter=self.starter,
        )


def _deferred_app(runtime: _DeferredRuntime) -> FastAPI:
    app = FastAPI()
    app.include_router(build_runtime_responses_router(runtime))

    def auth() -> dict[str, str]:
        return {"response_owner_id": OWNER}

    app.dependency_overrides[verify_auth] = auth
    app.dependency_overrides[verify_websocket_auth] = auth
    return app


async def _wait_for_status(
    client: AsyncOpenAI,
    response_id: str,
    expected: str,
) -> Any:
    for _ in range(100):
        response = await client.responses.retrieve(response_id)
        if response.status == expected:
            return response
        await asyncio.sleep(0)
    raise AssertionError(f"response {response_id} never reached {expected}")


@pytest.mark.asyncio
async def test_background_create_returns_before_release_and_get_advances() -> None:
    runtime = _DeferredRuntime()
    transport = httpx.ASGITransport(app=_deferred_app(runtime))
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as http_client:
        client = AsyncOpenAI(
            api_key="test",
            base_url="http://test/v1",
            http_client=http_client,
            max_retries=0,
        )
        created = await asyncio.wait_for(
            client.responses.create(
                model="buddy",
                input="blocked until released",
                background=True,
            ),
            timeout=0.25,
        )

        assert created.status == "queued"
        assert created.background is True
        assert created.model == "buddy"
        await runtime.starter.entered.wait()
        running = await _wait_for_status(client, created.id, "in_progress")
        assert running.background is True
        assert running.model == "buddy"
        task_names = {task.get_name() for task in runtime.responses_controller._tasks}
        assert any("-runtime-" in name for name in task_names)
        assert any("-relay-" in name for name in task_names)
        assert any("-background-" in name for name in task_names)

        runtime.starter.complete(created.id)
        terminal = await _wait_for_status(client, created.id, "completed")

    assert terminal.background is True
    assert terminal.model == "buddy"
    await runtime.responses_controller.shutdown(timeout_s=1.0)
    assert not runtime.responses_controller._tasks


@pytest.mark.asyncio
async def test_background_cancel_is_idempotent_and_owner_bound() -> None:
    runtime = _DeferredRuntime()
    transport = httpx.ASGITransport(app=_deferred_app(runtime))
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as http_client:
        client = AsyncOpenAI(
            api_key="test",
            base_url="http://test/v1",
            http_client=http_client,
            max_retries=0,
        )
        created = await client.responses.create(
            model="buddy",
            input="cancel me",
            background=True,
        )
        await runtime.starter.entered.wait()
        await _wait_for_status(client, created.id, "in_progress")
        first = await client.responses.cancel(created.id)
        second = await client.responses.cancel(created.id)

    assert first.status == "cancelled"
    assert second.status == "cancelled"
    assert first.id == second.id == created.id
    assert runtime.starter.handles[created.id].cancel_reasons == [
        "http_cancel_requested"
    ]
    await runtime.responses_controller.shutdown(timeout_s=1.0)


@pytest.mark.asyncio
async def test_background_failure_and_shutdown_leave_no_controller_tasks() -> None:
    failed_runtime = _DeferredRuntime(fail=True)
    failed_transport = httpx.ASGITransport(app=_deferred_app(failed_runtime))
    async with httpx.AsyncClient(
        transport=failed_transport,
        base_url="http://test",
    ) as http_client:
        failed_client = AsyncOpenAI(
            api_key="test",
            base_url="http://test/v1",
            http_client=http_client,
            max_retries=0,
        )
        created = await failed_client.responses.create(
            model="buddy",
            input="fail",
            background=True,
        )
        terminal = await _wait_for_status(failed_client, created.id, "failed")

    assert terminal.error is not None
    await failed_runtime.responses_controller.shutdown(timeout_s=1.0)
    assert not failed_runtime.responses_controller._tasks

    active_runtime = _DeferredRuntime()
    active_transport = httpx.ASGITransport(app=_deferred_app(active_runtime))
    async with httpx.AsyncClient(
        transport=active_transport,
        base_url="http://test",
    ) as http_client:
        active_client = AsyncOpenAI(
            api_key="test",
            base_url="http://test/v1",
            http_client=http_client,
            max_retries=0,
        )
        active = await active_client.responses.create(
            model="buddy",
            input="shutdown",
            background=True,
        )
        await active_runtime.starter.entered.wait()
        await _wait_for_status(active_client, active.id, "in_progress")
        await active_runtime.responses_controller.shutdown(timeout_s=1.0)

    assert active_runtime.starter.handles[active.id].cancel_reasons == [
        "registry_shutdown"
    ]
    assert not active_runtime.responses_controller._tasks


def test_every_sse_in_progress_body_is_a_complete_sdk_response() -> None:
    """Mutation falsifier: a status/usage-only fake Response must be impossible."""

    import json as _json

    from openai.types.responses import Response

    runtime = _Runtime()
    runtime.responses_controller.source_factory = lambda: ResponseEventSource(
        _publish(
            TurnStarted(
                response_id="resp_runtime_router",
                model="buddy",
                created_at=1,
            ),
            UsageUpdate(2, 3, 5),
            TurnCompleted(finish_reason="stop"),
        ),
        cancel=lambda _reason: None,
        terminal_response=asyncio.create_task(_terminal()),
    )
    client = TestClient(_app(runtime))

    stream = client.post(
        "/v1/responses",
        json={"model": "buddy", "input": "hej", "stream": True},
    )

    assert stream.status_code == 200
    payloads = [
        _json.loads(line.removeprefix("data: "))
        for line in stream.text.splitlines()
        if line.startswith("data: {")
    ]
    lifecycle = [
        item
        for item in payloads
        if item["type"]
        in {"response.created", "response.in_progress", "response.completed"}
    ]
    assert [item["type"] for item in lifecycle] == [
        "response.created",
        "response.in_progress",
        "response.completed",
    ]
    # The terminal body is the awaited committed envelope (fixture-owned);
    # the snapshot spine owns the created/in_progress bodies validated here.
    for item in lifecycle[:2]:
        body = item["response"]
        parsed = Response.model_validate(body)
        assert parsed.id == "resp_runtime_router"
        assert parsed.model == "buddy"
    assert lifecycle[2]["response"] == TERMINAL
    in_progress = lifecycle[1]["response"]
    assert in_progress["usage"]["total_tokens"] == 5
    assert set(in_progress) >= {
        "id",
        "object",
        "created_at",
        "model",
        "status",
        "output",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
    }
