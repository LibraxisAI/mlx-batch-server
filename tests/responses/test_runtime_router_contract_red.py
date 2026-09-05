"""RED contracts for the lifespan-owned Responses transport router."""

from __future__ import annotations

import ast
import asyncio
import json
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


def test_pending_steer_waits_for_client_tool_output_and_starts_once() -> None:
    runtime = _Runtime()
    create_calls: list[dict[str, Any]] = []

    async def events(
        response_id: str,
        finish_reason: str,
    ) -> AsyncIterator[SequencedTurnEvent]:
        yield SequencedTurnEvent(
            0,
            TurnStarted(response_id=response_id, model="buddy", created_at=1),
        )
        yield SequencedTurnEvent(1, TurnCompleted(finish_reason))

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
    assert json.loads(messages[1]["content"][0]["text"]) == {
        "type": "function_call",
        "call_id": CALL_ID,
        "name": "get_weather",
        "arguments": CALL_ARGUMENTS,
    }
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
