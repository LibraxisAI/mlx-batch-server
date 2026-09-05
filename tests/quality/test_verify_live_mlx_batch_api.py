from __future__ import annotations

import argparse
import asyncio
import json
import socket
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import uvicorn
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import JSONResponse, StreamingResponse
from scripts.quality.verify_live_mlx_batch_api import (
    ANTHROPIC_SDK_VERSION,
    DEFAULT_PROMPT,
    ReceivedEvent,
    VerificationConfig,
    VerificationError,
    _parser,
    _receipt_integrity_errors,
    _redact_receipt,
    _require_anthropic_sdk_version,
    _run_cli,
    _run_probe,
    _safe_url,
    _validate_openai_events,
    probe_openai_public_admission,
    probe_openai_sse,
    probe_openai_websocket,
    probe_safe_fetch_private_redirect,
    run_verification,
)

from mlx_batch_server.utils.safe_public_fetch import (
    FetchedResource,
    SafePublicFetchError,
)

MODEL = "buddy"
TEXT = "alpha beta"


def _terminal_response() -> dict[str, Any]:
    return {
        "id": "resp_live_acceptance",
        "object": "response",
        "created_at": 1,
        "model": MODEL,
        "status": "completed",
        "tools": [],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "output": [
            {
                "id": "msg_live_acceptance",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": TEXT}],
            }
        ],
        "usage": {"input_tokens": 2, "output_tokens": 2, "total_tokens": 4},
    }


def _openai_events(mode: str, *, stream_id: str | None = None) -> list[dict[str, Any]]:
    deltas = ["alpha ", "beta"]
    if mode == "single_delta":
        deltas = [TEXT]
    elif mode == "duplicate":
        deltas = ["alpha ", TEXT]

    item_content: list[dict[str, Any]] = []
    if mode == "prefill":
        item_content = [{"type": "output_text", "text": TEXT}]

    events: list[dict[str, Any]] = [
        {
            "type": "response.created",
            "response": {
                "id": "resp_live_acceptance",
                "object": "response",
                "created_at": 1,
                "model": MODEL,
                "status": "in_progress",
                "output": [],
                "usage": None,
            },
        },
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "id": "msg_live_acceptance",
                "type": "message",
                "role": "assistant",
                "status": "in_progress",
                "content": item_content,
            },
        },
        {
            "type": "response.content_part.added",
            "item_id": "msg_live_acceptance",
            "output_index": 0,
            "content_index": 0,
            "part": {
                "type": "output_text",
                "text": "",
                "annotations": [],
                "logprobs": [],
            },
        },
    ]
    events.extend(
        {
            "type": "response.output_text.delta",
            "item_id": "msg_live_acceptance",
            "output_index": 0,
            "content_index": 0,
            "delta": delta,
            "logprobs": [],
        }
        for delta in deltas
    )
    events.extend(
        [
            {
                "type": "response.output_text.done",
                "item_id": "msg_live_acceptance",
                "output_index": 0,
                "content_index": 0,
                "text": TEXT,
                "logprobs": [],
            },
            {
                "type": "response.content_part.done",
                "item_id": "msg_live_acceptance",
                "output_index": 0,
                "content_index": 0,
                "part": {
                    "type": "output_text",
                    "text": TEXT,
                    "annotations": [],
                    "logprobs": [],
                },
            },
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": _terminal_response()["output"][0],
            },
            {"type": "response.completed", "response": _terminal_response()},
        ]
    )
    for index, event in enumerate(events):
        event["sequence_number"] = index + (
            1 if mode == "bad_sequence" and index >= 3 else 0
        )
        if stream_id is not None:
            event["stream_id"] = stream_id
    return events


def _anthropic_message(
    *,
    content: list[dict[str, Any]],
    stop_reason: str | None = None,
    stop_sequence: str | None = None,
) -> dict[str, Any]:
    return {
        "id": "msg_anthropic_live",
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": MODEL,
        "stop_reason": stop_reason or ("end_turn" if content else None),
        "stop_sequence": stop_sequence,
        "usage": {
            "input_tokens": 2,
            "output_tokens": 2 if content else 0,
            "service_tier": "standard",
        },
    }


def _anthropic_events(
    *,
    text: str = TEXT,
    stop_reason: str = "end_turn",
    stop_sequence: str | None = None,
) -> list[dict[str, Any]]:
    midpoint = max(1, len(text) // 2)
    return [
        {"type": "message_start", "message": _anthropic_message(content=[])},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": text[:midpoint]},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": text[midpoint:]},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": stop_sequence},
            "usage": {"input_tokens": 2, "output_tokens": 2},
        },
        {"type": "message_stop"},
    ]


def _sse(event: dict[str, Any]) -> bytes:
    payload = json.dumps(event, separators=(",", ":"))
    return f"event: {event['type']}\ndata: {payload}\n\n".encode()


def _mode(prompt: Any) -> str:
    if isinstance(prompt, str) and prompt.startswith("mode:"):
        return prompt.removeprefix("mode:")
    return "success"


def _unsupported_field(body: dict[str, Any]) -> str | None:
    unsupported = next(
        (
            field
            for field in ("cache_control", "container", "inference_geo", "effort")
            if field in body
        ),
        None,
    )
    output_config = body.get("output_config")
    if (
        unsupported is None
        and isinstance(output_config, dict)
        and output_config.get("format") is not None
    ):
        unsupported = "output_config.format"
    thinking = body.get("thinking")
    if (
        unsupported is None
        and isinstance(thinking, dict)
        and thinking.get("type") == "enabled"
    ):
        unsupported = "thinking.type"
    tools = body.get("tools")
    if (
        unsupported is None
        and isinstance(tools, list)
        and any(
            isinstance(tool, dict) and tool.get("type") not in (None, "custom")
            for tool in tools
        )
    ):
        unsupported = "tools.0.type"
    messages = body.get("messages")
    if unsupported is None and isinstance(messages, list):
        for message_index, message in enumerate(messages):
            if not isinstance(message, dict) or not isinstance(
                message.get("content"), list
            ):
                continue
            for block_index, block in enumerate(message["content"]):
                if not isinstance(block, dict):
                    continue
                if block.get("cache_control") is not None:
                    unsupported = (
                        f"messages.{message_index}.content.{block_index}.cache_control"
                    )
                    break
                block_type = block.get("type")
                if block_type in {
                    "server_tool_use",
                    "web_search_tool_result",
                    "container_upload",
                    "thinking",
                    "redacted_thinking",
                }:
                    unsupported = f"messages.{message_index}.content.{block_index}.type"
                    break
                citations = block.get("citations")
                if isinstance(citations, dict) and citations.get("enabled") is True:
                    unsupported = f"messages.{message_index}.content.{block_index}.citations.enabled"
                    break
            if unsupported is not None:
                break
    return unsupported


def _validate_supported_fixture(body: dict[str, Any]) -> None:
    messages = body.get("messages")
    if not isinstance(messages, list):
        return
    for message in messages:
        if not isinstance(message, dict) or not isinstance(
            message.get("content"), list
        ):
            continue
        content = message["content"]
        types = [block.get("type") for block in content if isinstance(block, dict)]
        if "search_result" in types:
            assert types == [
                "text",
                "image",
                "text",
                "document",
                "search_result",
                "text",
            ]
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            assert block["tool_use_id"] == "toolu_admission_1"
            assert block["is_error"] is False
            assert [item["type"] for item in block["content"]] == ["text", "image"]


def _fake_app(*, refusal_mode: str = "strict") -> FastAPI:
    app = FastAPI()
    states: dict[str, dict[str, Any]] = {}
    next_response = 0
    app.state.inference_starts = 0

    def error(
        message: str,
        *,
        code: str,
        param: str | None,
        status_code: int = 400,
    ) -> JSONResponse:
        return JSONResponse(
            {
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                    "param": param,
                    "code": code,
                }
            },
            status_code=status_code,
        )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "role_runtime": {
                "runtime_stats": {
                    "executor": {
                        "active_requests": 0,
                        "tombstones": app.state.inference_starts,
                    }
                }
            }
        }

    @app.post("/v1/responses", response_model=None)
    async def responses(request: Request) -> JSONResponse | StreamingResponse:
        nonlocal next_response
        try:
            body = json.loads(await request.body())
        except json.JSONDecodeError:
            return error(
                "request body must contain valid JSON",
                code="invalid_json",
                param=None,
            )
        if not isinstance(body, dict):
            return error(
                "request body must be a JSON object",
                code="invalid_responses_request",
                param=None,
            )
        if "max_tool_calls" in body:
            return error(
                "unsupported Responses parameter: max_tool_calls",
                code="unsupported_parameter",
                param="max_tool_calls",
            )
        if body.get("background") is True:
            next_response += 1
            response_id = f"resp_background_{next_response}"
            raw_items = body.get("input")
            sequence = raw_items if isinstance(raw_items, list) else [raw_items]
            items = [
                {"id": f"input_stable_{index:02d}", **dict(item)}
                if isinstance(item, dict)
                else {
                    "id": f"input_stable_{index:02d}",
                    "type": "message",
                    "role": "user",
                    "content": str(item),
                }
                for index, item in enumerate(sequence)
            ]
            snapshot = {
                **_terminal_response(),
                "id": response_id,
                "status": "queued",
                "background": True,
                "output": [],
                "usage": None,
            }
            states[response_id] = {
                "snapshot": snapshot,
                "items": items,
                "get_count": 0,
                "deleted": False,
            }
            return JSONResponse(snapshot)
        if not body.get("stream"):
            return JSONResponse(_terminal_response())
        mode = _mode(body.get("input"))

        async def generate() -> AsyncIterator[bytes]:
            events = _openai_events(mode)
            if mode == "batched_receives":
                yield b"".join(_sse(event) for event in events)
            else:
                for event in events:
                    yield _sse(event)
                    await asyncio.sleep(0.002)
            yield b"data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    @app.get("/v1/responses/{response_id}/input_items")
    async def input_items(response_id: str, request: Request) -> JSONResponse:
        state = states.get(response_id)
        if state is None or state["deleted"]:
            return error(
                "response not found",
                code="response_not_found",
                param=None,
                status_code=404,
            )
        ordered = list(state["items"])
        if request.query_params.get("order", "desc") == "desc":
            ordered.reverse()
        after = request.query_params.get("after")
        start = 0
        if after is not None:
            start = next(
                index + 1 for index, item in enumerate(ordered) if item["id"] == after
            )
        limit = int(request.query_params.get("limit", "20"))
        data = ordered[start : start + limit]
        return JSONResponse(
            {
                "object": "list",
                "data": data,
                "first_id": data[0]["id"] if data else None,
                "last_id": data[-1]["id"] if data else None,
                "has_more": start + len(data) < len(ordered),
            }
        )

    @app.post("/v1/responses/{response_id}/cancel")
    async def cancel_response(response_id: str) -> JSONResponse:
        state = states.get(response_id)
        if state is None or state["deleted"]:
            return error(
                "response not found",
                code="response_not_found",
                param=None,
                status_code=404,
            )
        snapshot = state["snapshot"]
        if snapshot["status"] != "cancelled":
            snapshot.update(
                {
                    "status": "cancelled",
                    "output": [],
                    "usage": None,
                }
            )
        return JSONResponse(snapshot)

    @app.delete("/v1/responses/{response_id}")
    async def delete_response(response_id: str) -> JSONResponse:
        state = states.get(response_id)
        if state is None:
            return error(
                "response not found",
                code="response_not_found",
                param=None,
                status_code=404,
            )
        state["deleted"] = True
        return JSONResponse(
            {"id": response_id, "object": "response.deleted", "deleted": True}
        )

    @app.get("/v1/responses/{response_id}")
    async def retrieve_response(response_id: str) -> JSONResponse:
        state = states.get(response_id)
        if state is None or state["deleted"]:
            return error(
                "response not found",
                code="response_not_found",
                param=None,
                status_code=404,
            )
        snapshot = state["snapshot"]
        if snapshot["status"] == "queued":
            snapshot["status"] = "in_progress"
        elif snapshot["status"] == "in_progress":
            snapshot.update(
                {
                    **_terminal_response(),
                    "id": response_id,
                    "background": True,
                }
            )
        state["get_count"] += 1
        return JSONResponse(snapshot)

    @app.websocket("/v1/responses")
    async def responses_websocket(websocket: WebSocket) -> None:
        await websocket.accept()
        command = await websocket.receive_json()
        mode = _mode(command.get("input"))
        stream_id = command.get("stream_id")
        for event in _openai_events(mode, stream_id=stream_id):
            await websocket.send_json(event)

    @app.post("/anthropic/v1/messages", response_model=None)
    async def messages(body: dict[str, Any]) -> JSONResponse | StreamingResponse:
        unsupported = _unsupported_field(body)
        request_id = "req_fixture_admission"
        if unsupported is not None and refusal_mode in {"strict", "started_400"}:
            if refusal_mode == "started_400":
                app.state.inference_starts += 1
            return JSONResponse(
                status_code=400,
                headers={"request-id": request_id},
                content={
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": f"{unsupported} is not supported",
                    },
                    "request_id": request_id,
                },
            )
        if unsupported is not None and refusal_mode == "late_sse":
            app.state.inference_starts += 1

            async def late_error() -> AsyncIterator[bytes]:
                yield _sse(
                    {
                        "type": "error",
                        "error": {
                            "type": "invalid_request_error",
                            "message": f"{unsupported} is not supported",
                        },
                        "request_id": request_id,
                    }
                )

            return StreamingResponse(late_error(), media_type="text/event-stream")

        _validate_supported_fixture(body)
        app.state.inference_starts += 1
        stop_sequence = " delta" if body.get("stop_sequences") == [" delta"] else None
        stop_reason = "stop_sequence" if stop_sequence else "end_turn"
        text = "alpha beta gamma" if stop_sequence else TEXT
        if not body.get("stream"):
            return JSONResponse(
                _anthropic_message(
                    content=[{"type": "text", "text": text}],
                    stop_reason=stop_reason,
                    stop_sequence=stop_sequence,
                )
            )

        async def generate() -> AsyncIterator[bytes]:
            for event in _anthropic_events(
                text=text,
                stop_reason=stop_reason,
                stop_sequence=stop_sequence,
            ):
                yield _sse(event)
                await asyncio.sleep(0.001)

        return StreamingResponse(generate(), media_type="text/event-stream")

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
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, timeout=2)


class _FakeSafeFetcher:
    async def fetch(
        self,
        url: str,
        *,
        accepted_media_types: Sequence[str],
        max_bytes: int | None = None,
    ) -> FetchedResource:
        assert accepted_media_types
        assert max_bytes is not None
        if "private-redirect" in url:
            raise SafePublicFetchError(
                "url_target_blocked",
                "URL target is not a public address",
            )
        return FetchedResource(
            content=b"public fixture",
            media_type="text/plain",
            final_url="https://public.example/final?secret=hidden",
        )


class _UnsafeFakeFetcher(_FakeSafeFetcher):
    async def fetch(
        self,
        url: str,
        *,
        accepted_media_types: Sequence[str],
        max_bytes: int | None = None,
    ) -> FetchedResource:
        return FetchedResource(
            content=b"private fixture",
            media_type="text/plain",
            final_url=url,
        )


def _config(
    base_url: str, receipt: Path, *, mode: str = "success"
) -> VerificationConfig:
    return VerificationConfig(
        base_url=base_url,
        model=MODEL,
        api_key="test-openai-key",
        anthropic_api_key="test-anthropic-key",
        public_url="https://public.example/file.txt?token=do-not-record",
        private_redirect_url="https://public.example/private-redirect?token=secret",
        receipt_path=receipt,
        prompt=f"mode:{mode}" if mode != "success" else "fixture prompt",
        timeout_s=3,
    )


@pytest.mark.asyncio
async def test_full_fake_http_sse_websocket_success_writes_receipt(
    tmp_path: Path,
) -> None:
    receipt_path = tmp_path / "receipt.json"
    config: VerificationConfig
    async with _serve(_fake_app()) as base_url:
        config = _config(base_url, receipt_path)
        receipt = await run_verification(
            config,
            safe_fetcher=_FakeSafeFetcher(),
        )

    assert receipt["overall"] is True
    assert receipt["finalized"] is True
    assert receipt["integrity"]["ok"] is True
    assert len(receipt["probes"]) == 12
    assert all(probe["ok"] for probe in receipt["probes"].values())
    assert receipt["probes"]["openai_sse"]["details"]["text_delta_count"] == 2
    assert receipt["probes"]["openai_sse"]["details"]["text_delta_receive_count"] >= 2
    assert receipt["probes"]["openai_websocket"]["details"]["socket_count"] == 1
    public = receipt["probes"]["openai_public_admission"]["details"]
    assert public["sdk_version"] == "2.32.0"
    assert public["pagination_item_count"] == 25
    assert public["pagination_unique_ids"] == 25
    assert public["cancel_status"] == "cancelled"
    assert public["malformed_error_code"] == "invalid_json"
    assert public["unsupported_error_param"] == "max_tool_calls"
    assert (
        receipt["probes"]["anthropic_sdk_async_sse"]["details"]["text_delta_count"] == 2
    )
    for probe_id in (
        "anthropic_sdk_sync_non_stream",
        "anthropic_sdk_async_non_stream",
        "anthropic_sdk_sync_sse",
        "anthropic_sdk_async_sse",
    ):
        assert (
            receipt["probes"][probe_id]["details"]["sdk_version"]
            == ANTHROPIC_SDK_VERSION
        )
    supported = receipt["probes"]["anthropic_supported_matrix"]["details"]
    refused = receipt["probes"]["anthropic_refusal_matrix"]["details"]
    assert len(supported["cells"]) == 12
    assert len(refused["cells"]) == 30
    assert {cell["cell_id"] for cell in supported["cells"]} == set(
        supported["required_cell_ids"]
    )
    assert {cell["cell_id"] for cell in refused["cells"]} == set(
        refused["required_cell_ids"]
    )
    assert all(cell["http_status"] == 400 for cell in refused["cells"])
    assert all(cell["sse_event_bytes"] == 0 for cell in refused["cells"])
    assert all(cell["inference_start_count"] == 0 for cell in refused["cells"])
    assert json.loads(receipt_path.read_text()) == receipt
    assert "do-not-record" not in receipt_path.read_text()
    assert "secret" not in receipt_path.read_text()
    assert "test-openai-key" not in receipt_path.read_text()
    assert "test-anthropic-key" not in receipt_path.read_text()

    missing_probe = deepcopy(receipt["probes"])
    missing_probe.pop("anthropic_sdk_sync_sse")
    assert any(
        "missing required probes" in error
        for error in _receipt_integrity_errors(missing_probe, config)
    )

    missing_cell = deepcopy(receipt["probes"])
    missing_cell["anthropic_refusal_matrix"]["details"]["cells"].pop()
    assert any(
        "missing required cells" in error
        for error in _receipt_integrity_errors(missing_cell, config)
    )

    duplicate_cell = deepcopy(receipt["probes"])
    duplicate_cell["anthropic_supported_matrix"]["details"]["cells"].append(
        duplicate_cell["anthropic_supported_matrix"]["details"]["cells"][0]
    )
    assert any(
        "duplicate matrix cells" in error
        for error in _receipt_integrity_errors(duplicate_cell, config)
    )

    assert any(
        "duplicate probe ids" in error
        for error in _receipt_integrity_errors(
            receipt["probes"],
            config,
            declared_probe_ids=[*receipt["probes"], "openai_non_stream"],
        )
    )

    skipped_probe = deepcopy(receipt["probes"])
    skipped_probe["anthropic_sdk_sync_sse"] = {
        "ok": True,
        "status": "skipped",
        "details": {},
    }
    assert any(
        "required probes were skipped" in error
        for error in _receipt_integrity_errors(skipped_probe, config)
    )


def test_anthropic_sdk_version_is_an_exact_admission_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "scripts.quality.verify_live_mlx_batch_api.anthropic.__version__", "0.95.0"
    )
    with pytest.raises(VerificationError, match=r"must be 0\.96\.0, found 0\.95\.0"):
        _require_anthropic_sdk_version()


def test_receipt_redaction_removes_credentials_from_nested_failures() -> None:
    redacted = _redact_receipt(
        {
            "error": "server echoed sk-openai-secret",
            "nested": [{"error": "sk-anthropic-secret"}],
        },
        ("sk-openai-secret", "sk-anthropic-secret"),
    )

    serialized = json.dumps(redacted)
    assert "sk-openai-secret" not in serialized
    assert "sk-anthropic-secret" not in serialized
    assert serialized.count("[redacted]") == 2


@pytest.mark.asyncio
async def test_public_admission_probe_covers_lifecycle_pagination_and_errors(
    tmp_path: Path,
) -> None:
    async with _serve(_fake_app()) as base_url:
        details = await probe_openai_public_admission(
            _config(base_url, tmp_path / "unused.json")
        )

    assert details["lifecycle_statuses"] == [
        "queued",
        "in_progress",
        "completed",
    ]
    assert details["pagination_item_count"] == 25
    assert details["pagination_unique_ids"] == 25
    assert details["delete_status"] == 404


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "error"),
    [
        ("duplicate", "delta, done, and terminal text differ"),
        ("prefill", "prefilled final text"),
        ("single_delta", "at least two non-empty text deltas"),
        ("batched_receives", "buffered into one raw HTTP receive"),
        ("bad_sequence", "non-contiguous sequence numbers"),
    ],
)
async def test_raw_sse_falsifies_invalid_streams(
    tmp_path: Path,
    mode: str,
    error: str,
) -> None:
    async with _serve(_fake_app()) as base_url:
        with pytest.raises(VerificationError, match=error):
            await probe_openai_sse(
                _config(base_url, tmp_path / "unused.json", mode=mode)
            )


@pytest.mark.asyncio
async def test_official_websocket_sdk_falsifies_bad_sequence(tmp_path: Path) -> None:
    async with _serve(_fake_app()) as base_url:
        with pytest.raises(VerificationError, match="non-contiguous sequence numbers"):
            await probe_openai_websocket(
                _config(base_url, tmp_path / "unused.json", mode="bad_sequence")
            )


def test_validator_falsifies_non_monotonic_timestamps() -> None:
    events = [
        ReceivedEvent(event, 100 + index, 20 - index, float(index), index + 1)
        for index, event in enumerate(_openai_events("success"))
    ]
    with pytest.raises(VerificationError, match="timestamps are not monotonic"):
        _validate_openai_events(
            events,
            model=MODEL,
            require_separate_receives=False,
        )


@pytest.mark.asyncio
async def test_private_redirect_rejection_is_required_and_audit_safe(
    tmp_path: Path,
) -> None:
    config = _config("http://127.0.0.1:1", tmp_path / "unused.json")
    details = await probe_safe_fetch_private_redirect(config, _FakeSafeFetcher())

    assert details["fetch_status"] == "rejected"
    assert details["error_code"] == "url_target_blocked"
    assert "token=" not in details["requested_url"]
    with pytest.raises(VerificationError, match="unexpectedly succeeded"):
        await probe_safe_fetch_private_redirect(config, _UnsafeFakeFetcher())


@pytest.mark.asyncio
async def test_failure_still_writes_finalized_receipt(tmp_path: Path) -> None:
    receipt_path = tmp_path / "failed.json"
    async with _serve(_fake_app()) as base_url:
        receipt = await run_verification(
            _config(base_url, receipt_path, mode="single_delta"),
            safe_fetcher=_FakeSafeFetcher(),
        )

    persisted = json.loads(receipt_path.read_text())
    assert receipt["overall"] is False
    assert persisted["finalized"] is True
    assert persisted["probes"]["openai_sse"]["status"] == "failed"
    assert persisted["probes"]["openai_websocket"]["status"] == "failed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("refusal_mode", "error"),
    (
        ("silent", "expected HTTP 400"),
        ("late_sse", "expected HTTP 400"),
        ("started_400", "crossed the inference boundary"),
    ),
)
async def test_refusal_matrix_falsifies_silent_late_and_started_requests(
    tmp_path: Path,
    refusal_mode: str,
    error: str,
) -> None:
    receipt_path = tmp_path / f"{refusal_mode}.json"
    async with _serve(_fake_app(refusal_mode=refusal_mode)) as base_url:
        receipt = await run_verification(
            _config(base_url, receipt_path),
            safe_fetcher=_FakeSafeFetcher(),
        )

    persisted = json.loads(receipt_path.read_text())
    assert receipt["overall"] is False
    assert persisted["finalized"] is True
    assert persisted["probes"]["anthropic_refusal_matrix"]["status"] == "failed"
    assert error in persisted["probes"]["anthropic_refusal_matrix"]["error"]


@pytest.mark.asyncio
async def test_probe_timeout_is_bounded_and_structured() -> None:
    async def hangs() -> dict[str, Any]:
        await asyncio.sleep(60)
        return {}

    result = await _run_probe("hangs", 0.01, hangs)

    assert result["ok"] is False
    assert result["status"] == "timeout"
    assert result["finished_at_unix_ns"] >= result["started_at_unix_ns"]


@pytest.mark.asyncio
async def test_missing_credential_writes_configuration_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MISSING_LIVE_KEY", raising=False)
    path = tmp_path / "configuration-failed.json"
    args = argparse.Namespace(
        base_url="http://127.0.0.1:10240",
        anthropic_base_url=None,
        model=MODEL,
        api_key_env="MISSING_LIVE_KEY",
        anthropic_api_key_env=None,
        public_url="https://public.example/file.txt",
        private_redirect_url="https://public.example/private-redirect",
        receipt=path,
        prompt="fixture",
        max_output_tokens=64,
        timeout=3.0,
        safe_fetch_max_bytes=1024,
        accepted_media_types=None,
        stream_id="live_acceptance",
    )

    assert await _run_cli(args) == 1
    receipt = json.loads(path.read_text())
    assert receipt["finalized"] is True
    assert receipt["overall"] is False
    assert receipt["probes"]["configuration"]["status"] == "failed"
    assert "MISSING_LIVE_KEY" in receipt["probes"]["configuration"]["error"]


def test_url_redaction_removes_credentials_query_and_fragment() -> None:
    assert (
        _safe_url("https://user:password@example.test:8443/path?token=secret#piece")
        == "https://example.test:8443/path"
    )


def test_cli_uses_a_real_default_prompt() -> None:
    args = _parser().parse_args(
        [
            "--base-url",
            "http://127.0.0.1:10240",
            "--model",
            MODEL,
            "--api-key-env",
            "LIVE_KEY",
            "--public-url",
            "https://public.example/file.txt",
            "--private-redirect-url",
            "https://public.example/private-redirect",
            "--receipt",
            "receipt.json",
        ]
    )

    assert args.prompt == DEFAULT_PROMPT
