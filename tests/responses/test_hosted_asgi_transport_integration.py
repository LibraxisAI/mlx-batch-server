"""Joined hosted OpenPage proof through the actual ASGI Responses routes.

Only model generation and remote HTTP I/O are scripted. Mapping, hosted
execution, SafePublicFetch attestation, continuation, owner snapshots and all
three public transports use their production implementations.
"""

from __future__ import annotations

import asyncio
import json
import socket
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from openai.types.responses import Response, ResponseStreamEvent
from pydantic import TypeAdapter

from mlx_batch_server.auth.dependency import verify_auth, verify_websocket_auth
from mlx_batch_server.responses.controller import ResponsesController
from mlx_batch_server.responses.registry import ResponseRegistry
from mlx_batch_server.responses.runtime_mapper import CanonicalResponsesMapper
from mlx_batch_server.responses.runtime_projection import create_runtime_projection
from mlx_batch_server.responses.runtime_router import build_runtime_responses_router
from mlx_batch_server.runtime.agentic import (
    FAILURE_CONTINUATION_PREPARATION,
    HostedAgenticRuntimeStarter,
)
from mlx_batch_server.runtime.contracts import GenerationRequest, RuntimeKey, TurnSink
from mlx_batch_server.runtime.events import (
    ContentPartCompleted,
    ContentPartStarted,
    OutputItemCompleted,
    OutputItemStarted,
    TextCompleted,
    TextDelta,
    ToolCompleted,
    ToolDelta,
    TurnCompleted,
    TurnFailed,
    TurnStarted,
    UsageUpdate,
)
from mlx_batch_server.runtime.service import FirstWriterCancelToken, RuntimeStartService
from mlx_batch_server.tools.hosted import HostedToolCatalog, HostedToolExecutor
from mlx_batch_server.tools.hosted_web import HostedOpenPageTool
from mlx_batch_server.utils.safe_public_fetch import (
    SafePublicFetch,
    SafePublicFetchLimits,
)

_OWNER = "resp-owner:v1:api-key:" + "a" * 64
_URL = "https://example.com/public"
_DOCUMENT = "Loctree maps repository structure. PRIVATE_FETCH_RESULT_SENTINEL"
_SUCCESS_REPLY = "The document says Loctree maps repository structure."
_FAILURE_REPLY = "The page fetch failed, so I cannot verify the document."
_CALL_ID = "call_page"


class _ScriptedTurn:
    def __init__(
        self,
        request: GenerationRequest,
        sink: TurnSink,
        *,
        round_index: int,
        failed: bool,
    ) -> None:
        self.response_id = request.response_id
        self._task = asyncio.create_task(
            self._emit(request, sink, round_index=round_index, failed=failed)
        )

    def cancel(self, reason: str) -> bool:
        return self._task.cancel(reason)

    async def wait_closed(self) -> None:
        await asyncio.shield(self._task)

    async def _emit(
        self,
        request: GenerationRequest,
        sink: TurnSink,
        *,
        round_index: int,
        failed: bool,
    ) -> None:
        sink.emit(TurnStarted(request.response_id, request.runtime.model_id, 1))
        if round_index == 0:
            arguments = json.dumps({"url": _URL}, separators=(",", ":"))
            sink.emit(
                OutputItemStarted("function_call", 0, "fc_page", _CALL_ID, "open_page")
            )
            sink.emit(ToolDelta(0, _CALL_ID, "fc_page", "open_page", arguments))
            sink.emit(ToolCompleted(0, _CALL_ID, "fc_page", "open_page", arguments))
            sink.emit(
                OutputItemCompleted(
                    "function_call",
                    0,
                    "fc_page",
                    call_id=_CALL_ID,
                    name="open_page",
                    arguments=arguments,
                )
            )
        elif round_index == 1:
            text = _FAILURE_REPLY if failed else _SUCCESS_REPLY
            sink.emit(OutputItemStarted("message", 0, "msg_answer"))
            sink.emit(ContentPartStarted("output_text", 0, 0, "msg_answer"))
            # Multiple model deltas exercise projection, not a prebuilt response.
            for delta in (text[:12], text[12:]):
                sink.emit(TextDelta(delta, "msg_answer", 0, 0))
            sink.emit(TextCompleted(text, "msg_answer", 0, 0))
            sink.emit(ContentPartCompleted("output_text", 0, 0, "msg_answer", text))
            sink.emit(OutputItemCompleted("message", 0, "msg_answer", text=text))
        else:
            sink.emit(TurnFailed("unexpected extra model round", "fixture_round_limit"))
            return
        usage = UsageUpdate(10, 5, 15)
        sink.emit(usage)
        sink.emit(TurnCompleted("tool_calls" if round_index == 0 else "stop", usage))


class _ScriptedModel(RuntimeStartService):
    """Implements only the generation boundary; never constructs a model manager."""

    def __init__(self, *, failed: bool) -> None:
        self.failed = failed
        self.requests: list[GenerationRequest] = []

    async def start(
        self,
        request: GenerationRequest,
        sink: TurnSink,
        *,
        cancel: FirstWriterCancelToken | None = None,
    ) -> _ScriptedTurn:
        round_index = len(self.requests)
        self.requests.append(request)
        return _ScriptedTurn(request, sink, round_index=round_index, failed=self.failed)


class _HostedRuntime:
    def __init__(self, *, failed: bool) -> None:
        self.fetches: list[httpx.Request] = []
        self.model = _ScriptedModel(failed=failed)

        def remote(request: httpx.Request) -> httpx.Response:
            self.fetches.append(request)
            return httpx.Response(
                503 if failed else 200,
                headers={"content-type": "text/plain"},
                stream=httpx.ByteStream(_DOCUMENT.encode()),
                extensions={"mlx_batch_server.connected_peer": request.url.host},
            )

        def resolve(host: str, port: int, *args: Any, **kwargs: Any) -> list[Any]:
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.1.1.1", port))]

        fetch = SafePublicFetch(
            limits=SafePublicFetchLimits(max_bytes=4096, timeout=2.0),
            transport=httpx.MockTransport(remote),
            getaddrinfo=resolve,
        )
        catalog = HostedToolCatalog((HostedOpenPageTool(fetch=fetch),))
        starter = HostedAgenticRuntimeStarter(
            self.model,
            catalog=catalog,
            executor=HostedToolExecutor(catalog),
        )
        self.response_registry = ResponseRegistry()
        self.responses_controller = ResponsesController(
            registry=self.response_registry,
            mapper=CanonicalResponsesMapper(
                resolve_runtime=lambda **kwargs: RuntimeKey(model_id=kwargs["model"]),
                projection_factory=create_runtime_projection,
            ),
            starter=starter,
        )

    def app(self) -> FastAPI:
        @asynccontextmanager
        async def lifespan(app: FastAPI) -> AsyncIterator[None]:
            yield
            await self.responses_controller.shutdown()

        app = FastAPI(lifespan=lifespan)
        app.include_router(build_runtime_responses_router(self))

        def auth() -> dict[str, str]:
            return {"response_owner_id": _OWNER}

        app.dependency_overrides[verify_auth] = auth
        app.dependency_overrides[verify_websocket_auth] = auth
        return app


def _run_transport(
    transport: str, *, failed: bool
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    runtime = _HostedRuntime(failed=failed)
    payload = {
        "model": "scripted-model",
        "input": f"Open {_URL} and explain what the document says.",
        "tools": [{"type": "web_search"}],
    }
    events: list[dict[str, Any]] = []
    with TestClient(runtime.app()) as client:
        if transport == "websocket":
            with client.websocket_connect("/v1/responses") as websocket:
                websocket.send_json(
                    {"type": "response.create", "stream_id": "page", **payload}
                )
                for _ in range(100):
                    event = websocket.receive_json()
                    assert event.pop("stream_id") == "page"
                    events.append(event)
                    if event["type"] in {
                        "response.completed",
                        "response.failed",
                        "response.incomplete",
                    }:
                        break
                else:
                    pytest.fail("hosted response did not reach a terminal event")
            terminal = events[-1]["response"]
        else:
            response = client.post(
                "/v1/responses", json={**payload, "stream": transport == "sse"}
            )
            assert response.status_code == 200, response.text
            if transport == "sse":
                data = [
                    line[6:]
                    for line in response.text.splitlines()
                    if line.startswith("data: ")
                ]
                assert data[-1] == "[DONE]"
                events = [json.loads(item) for item in data[:-1]]
                terminal = events[-1]["response"]
            else:
                terminal = response.json()
        stored = client.get(f"/v1/responses/{terminal['id']}")
        assert stored.status_code == 200
        assert stored.json() == terminal

    assert len(runtime.fetches) == 1
    assert runtime.fetches[0].url.host == "1.1.1.1"
    assert runtime.fetches[0].headers["host"] == "example.com"
    assert len(runtime.model.requests) == 2
    first, continuation = runtime.model.requests
    assert first.response_id == terminal["id"]
    assert continuation.response_id == f"{terminal['id']}-hosted-round-1"
    assert [tool["name"] for tool in first.tools] == ["open_page"]
    tool_messages = [
        message for message in continuation.messages if message["role"] == "tool"
    ]
    assert len(tool_messages) == 1
    assert tool_messages[0]["call_id"] == _CALL_ID
    result = json.loads(tool_messages[0]["output"])
    if failed:
        assert result["error"]["code"] == "fetch_url_fetch_status"
        assert continuation.tools == ()
        assert continuation.sampling["tool_choice"] == "none"
        assert (
            sum(
                message.get("content") == FAILURE_CONTINUATION_PREPARATION
                for message in continuation.messages
            )
            == 1
        )
    else:
        assert result["content"] == _DOCUMENT
        assert result["url"] == _URL
    assert not runtime.responses_controller._tasks
    return terminal, events


def _without_run_identity(response: dict[str, Any]) -> dict[str, Any]:
    """Independent requests differ only in their allocated id and wall clock."""

    return {**response, "id": "resp_independent_run", "created_at": 1}


def _comparable_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {**event, "response": _without_run_identity(event["response"])}
        if "response" in event
        else event
        for event in events
    ]


@pytest.mark.parametrize("failed", [False, True], ids=["success", "fetch-failure"])
def test_hosted_open_page_has_one_asgi_truth_and_model_continuation(
    failed: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_network(*args: Any, **kwargs: Any) -> None:
        pytest.fail("hosted ASGI fixture attempted real network I/O")

    monkeypatch.setattr(socket.socket, "connect", forbidden_network)
    monkeypatch.setattr(socket.socket, "bind", forbidden_network)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden_network)
    results = [
        _run_transport(transport, failed=failed)
        for transport in ("http", "sse", "websocket")
    ]
    expected_text = _FAILURE_REPLY if failed else _SUCCESS_REPLY
    for terminal, events in results:
        parsed = Response.model_validate(terminal)
        assert parsed.status == "completed"
        assert parsed.output_text == expected_text
        assert [item.type for item in parsed.output] == ["web_search_call", "message"]
        assert terminal["output"][0]["status"] == ("failed" if failed else "completed")
        assert terminal["output"][0]["action"] == {"type": "open_page", "url": _URL}
        assert terminal["usage"]["input_tokens"] == 20
        assert terminal["usage"]["output_tokens"] == 10
        assert terminal["usage"]["total_tokens"] == 30
        assert "PRIVATE_FETCH_RESULT_SENTINEL" not in json.dumps(terminal)
        if not events:
            continue
        adapter = TypeAdapter(ResponseStreamEvent)
        for event in events:
            adapter.validate_python(event)
        types = [event["type"] for event in events]
        assert types.count("response.created") == 1
        assert types.count("response.completed") == 1
        assert types.count("response.output_item.done") == 2
        assert types.count("response.web_search_call.in_progress") == 1
        assert types.count("response.web_search_call.searching") == 1
        assert types.count("response.web_search_call.completed") == (0 if failed else 1)
        assert "response.function_call_arguments.done" not in types
        assert "response.failed" not in types
        opening = next(
            event["item"]
            for event in events
            if event["type"] == "response.output_item.added"
            and event["item"]["type"] == "web_search_call"
        )
        assert opening["status"] == "in_progress"
        assert opening["action"] == {"type": "open_page", "url": _URL}
        assert expected_text not in json.dumps(opening)
        numbers = [event["sequence_number"] for event in events]
        assert numbers == sorted(set(numbers))
        assert (
            "".join(
                event["delta"]
                for event in events
                if event["type"] == "response.output_text.delta"
            )
            == expected_text
        )
        assert "PRIVATE_FETCH_RESULT_SENTINEL" not in json.dumps(events)
    http_terminal, sse_terminal, ws_terminal = (
        _without_run_identity(terminal) for terminal, _ in results
    )
    assert http_terminal == sse_terminal == ws_terminal
    assert _comparable_events(results[1][1]) == _comparable_events(results[2][1])
