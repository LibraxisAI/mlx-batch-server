from __future__ import annotations

import argparse
import asyncio
import json
import socket
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse, StreamingResponse
from scripts.quality.verify_live_mlx_batch_api import (
    DEFAULT_PROMPT,
    ReceivedEvent,
    VerificationConfig,
    VerificationError,
    _parser,
    _run_cli,
    _run_probe,
    _safe_url,
    _validate_openai_events,
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


def _anthropic_message(*, content: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": "msg_anthropic_live",
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": MODEL,
        "stop_reason": "end_turn" if content else None,
        "stop_sequence": None,
        "usage": {"input_tokens": 2, "output_tokens": 2 if content else 0},
    }


def _anthropic_events() -> list[dict[str, Any]]:
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
            "delta": {"type": "text_delta", "text": "alpha "},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "beta"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
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


def _fake_app() -> FastAPI:
    app = FastAPI()

    @app.post("/v1/responses", response_model=None)
    async def responses(body: dict[str, Any]) -> JSONResponse | StreamingResponse:
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
        if not body.get("stream"):
            return JSONResponse(
                _anthropic_message(content=[{"type": "text", "text": TEXT}])
            )

        async def generate() -> AsyncIterator[bytes]:
            for event in _anthropic_events():
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
    async with _serve(_fake_app()) as base_url:
        receipt = await run_verification(
            _config(base_url, receipt_path),
            safe_fetcher=_FakeSafeFetcher(),
        )

    assert receipt["overall"] is True
    assert receipt["finalized"] is True
    assert len(receipt["probes"]) == 7
    assert all(probe["ok"] for probe in receipt["probes"].values())
    assert receipt["probes"]["openai_sse"]["details"]["text_delta_count"] == 2
    assert receipt["probes"]["openai_sse"]["details"]["text_delta_receive_count"] >= 2
    assert receipt["probes"]["openai_websocket"]["details"]["socket_count"] == 1
    assert receipt["probes"]["anthropic_sse"]["details"]["text_delta_count"] == 2
    assert json.loads(receipt_path.read_text()) == receipt
    assert "do-not-record" not in receipt_path.read_text()
    assert "secret" not in receipt_path.read_text()
    assert "test-openai-key" not in receipt_path.read_text()
    assert "test-anthropic-key" not in receipt_path.read_text()


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
