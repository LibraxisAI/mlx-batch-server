#!/usr/bin/env python3
"""Verify an already-running mlx-batch API and write a durable JSON receipt.

The verifier never starts a server or loads a model. OpenAI and Anthropic
semantic compatibility is exercised with their official SDKs. OpenAI SSE is
also consumed as raw HTTP bytes so receive timing and chunk separation remain
observable instead of being inferred from an already-buffered SDK object.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit, urlunsplit

import anthropic
import httpx
from openai import AsyncOpenAI

from mlx_batch_server.utils.safe_public_fetch import (
    FetchedResource,
    SafePublicFetch,
    SafePublicFetchError,
    SafePublicFetchLimits,
)

SCHEMA_VERSION = "mlx-batch-server.live-api-acceptance.v1"
DEFAULT_PROMPT = "Reply with exactly these words: alpha beta gamma delta epsilon."
TERMINAL_RESPONSE_EVENTS = frozenset(
    {
        "response.completed",
        "response.failed",
        "response.incomplete",
        "response.cancelled",
    }
)
DEFAULT_MEDIA_TYPES = (
    "image/png",
    "image/jpeg",
    "text/plain",
    "text/html",
    "application/json",
)


class VerificationError(RuntimeError):
    """A live response violated an acceptance invariant."""


class PublicFetcher(Protocol):
    async def fetch(
        self,
        url: str,
        *,
        accepted_media_types: Sequence[str],
        max_bytes: int | None = None,
    ) -> FetchedResource: ...


@dataclass(frozen=True, slots=True)
class VerificationConfig:
    base_url: str
    model: str
    api_key: str
    anthropic_api_key: str
    public_url: str
    private_redirect_url: str
    receipt_path: Path
    prompt: str = DEFAULT_PROMPT
    max_output_tokens: int = 64
    timeout_s: float = 120.0
    safe_fetch_max_bytes: int = 2 * 1024 * 1024
    accepted_media_types: tuple[str, ...] = DEFAULT_MEDIA_TYPES
    anthropic_base_url: str | None = None
    stream_id: str = "live_acceptance"

    def __post_init__(self) -> None:
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must use http or https")
        if not self.model:
            raise ValueError("model must not be empty")
        if not self.api_key or not self.anthropic_api_key:
            raise ValueError("API keys must not be empty")
        if self.timeout_s <= 0:
            raise ValueError("timeout must be positive")
        if self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        if self.safe_fetch_max_bytes < 1:
            raise ValueError("safe_fetch_max_bytes must be positive")
        if not self.stream_id:
            raise ValueError("stream_id must not be empty")


@dataclass(frozen=True, slots=True)
class ReceivedEvent:
    payload: dict[str, Any]
    received_at_unix_ns: int
    received_at_monotonic_ns: int
    elapsed_ms: float
    receive_index: int


Probe = Callable[[], Awaitable[dict[str, Any]]]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _api_base(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/v1"


def _anthropic_base(config: VerificationConfig) -> str:
    if config.anthropic_base_url:
        return config.anthropic_base_url.rstrip("/")
    return f"{config.base_url.rstrip('/')}/anthropic"


def _safe_url(value: str) -> str:
    """Keep receipt URLs useful without retaining credentials or query secrets."""

    parsed = urlsplit(value)
    hostname = parsed.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    netloc = hostname
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def _safe_error(error: Exception) -> str:
    if isinstance(error, SafePublicFetchError):
        return str(error)
    return f"{type(error).__name__}: {error}"


def _model_dump(value: Any) -> dict[str, Any]:
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        result = dump(mode="json")
        if isinstance(result, dict):
            return result
    if isinstance(value, Mapping):
        return dict(value)
    raise VerificationError(f"SDK event is not object-shaped: {type(value).__name__}")


def _text_from_response(response: Mapping[str, Any]) -> str:
    parts: list[str] = []
    output = response.get("output")
    if not isinstance(output, list):
        return ""
    for item in output:
        if not isinstance(item, Mapping) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, Mapping) or part.get("type") != "output_text":
                continue
            text = part.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def _prefilled_added_text(event: Mapping[str, Any]) -> str:
    item = event.get("item")
    if not isinstance(item, Mapping):
        return ""
    parts: list[str] = []
    for field in ("content", "summary"):
        content = item.get(field)
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, Mapping) and isinstance(part.get("text"), str):
                parts.append(str(part["text"]))
    return "".join(parts)


def _sequence_numbers(events: Sequence[ReceivedEvent]) -> list[int]:
    numbers: list[int] = []
    for event in events:
        value = event.payload.get("sequence_number")
        if not isinstance(value, int) or isinstance(value, bool):
            raise VerificationError("missing sequence_number")
        numbers.append(value)
    return numbers


def _require_contiguous(numbers: Sequence[int]) -> None:
    _require(bool(numbers), "stream contained no sequenced events")
    _require(numbers[0] == 0, f"sequence numbers must start at zero: {list(numbers)}")
    expected = list(range(numbers[0], numbers[0] + len(numbers)))
    _require(
        list(numbers) == expected, f"non-contiguous sequence numbers: {list(numbers)}"
    )


def _event_receipt(events: Sequence[ReceivedEvent]) -> list[dict[str, Any]]:
    return [
        {
            "type": str(event.payload.get("type") or ""),
            "sequence_number": event.payload.get("sequence_number"),
            "stream_id": event.payload.get("stream_id"),
            "received_at_unix_ns": event.received_at_unix_ns,
            "received_at_monotonic_ns": event.received_at_monotonic_ns,
            "elapsed_ms": event.elapsed_ms,
            "receive_index": event.receive_index,
        }
        for event in events
    ]


def _parse_sse_block(block: bytes) -> dict[str, Any] | None:
    data_lines: list[bytes] = []
    for line in block.replace(b"\r\n", b"\n").split(b"\n"):
        if line.startswith(b"data:"):
            data_lines.append(line[5:].lstrip(b" "))
    if not data_lines:
        return None
    data = b"\n".join(data_lines).decode("utf-8")
    if data == "[DONE]":
        return {"type": "[DONE]"}
    parsed = json.loads(data)
    if not isinstance(parsed, dict):
        raise VerificationError("SSE data must be a JSON object")
    return parsed


async def _read_raw_sse(
    response: httpx.Response,
    *,
    started_monotonic_ns: int,
) -> tuple[list[ReceivedEvent], int]:
    events: list[ReceivedEvent] = []
    buffer = b""
    receive_count = 0
    done_count = 0
    async for chunk in response.aiter_bytes():
        if not chunk:
            continue
        receive_count += 1
        received_ns = time.time_ns()
        received_monotonic_ns = time.monotonic_ns()
        buffer += chunk
        normalized = buffer.replace(b"\r\n", b"\n")
        blocks = normalized.split(b"\n\n")
        buffer = blocks.pop()
        for block in blocks:
            payload = _parse_sse_block(block)
            if payload is None:
                continue
            if payload["type"] == "[DONE]":
                done_count += 1
                continue
            events.append(
                ReceivedEvent(
                    payload=payload,
                    received_at_unix_ns=received_ns,
                    received_at_monotonic_ns=received_monotonic_ns,
                    elapsed_ms=(received_monotonic_ns - started_monotonic_ns)
                    / 1_000_000,
                    receive_index=receive_count,
                )
            )
    _require(not buffer.strip(), "SSE stream ended with an incomplete event")
    return events, done_count


def _validate_openai_events(
    events: Sequence[ReceivedEvent],
    *,
    model: str,
    require_separate_receives: bool,
    stream_id: str | None = None,
) -> dict[str, Any]:
    _require(bool(events), "OpenAI stream emitted no events")
    timestamps = [event.received_at_monotonic_ns for event in events]
    _require(
        timestamps == sorted(timestamps), "event receive timestamps are not monotonic"
    )
    numbers = _sequence_numbers(events)
    _require_contiguous(numbers)

    if stream_id is not None:
        _require(
            all(event.payload.get("stream_id") == stream_id for event in events),
            "WebSocket event omitted or changed the requested stream_id",
        )

    terminal = [
        event
        for event in events
        if event.payload.get("type") in TERMINAL_RESPONSE_EVENTS
    ]
    _require(
        len(terminal) == 1, f"expected one terminal event, received {len(terminal)}"
    )
    _require(
        terminal[0].payload.get("type") == "response.completed",
        "response did not complete",
    )

    deltas = [
        event
        for event in events
        if event.payload.get("type") == "response.output_text.delta"
        and isinstance(event.payload.get("delta"), str)
        and bool(event.payload["delta"])
    ]
    _require(
        len(deltas) >= 2,
        f"expected at least two non-empty text deltas, received {len(deltas)}",
    )
    if require_separate_receives:
        receive_indexes = {event.receive_index for event in deltas}
        _require(
            len(receive_indexes) >= 2,
            "text deltas were buffered into one raw HTTP receive",
        )

    added = [
        event
        for event in events
        if event.payload.get("type") == "response.output_item.added"
    ]
    _require(bool(added), "stream omitted response.output_item.added")
    _require(
        all(not _prefilled_added_text(event.payload) for event in added),
        "response.output_item.added contained prefilled final text",
    )

    done = [
        event
        for event in events
        if event.payload.get("type") == "response.output_text.done"
    ]
    _require(
        len(done) == 1, f"expected one response.output_text.done, received {len(done)}"
    )
    done_text = done[0].payload.get("text")
    _require(
        isinstance(done_text, str) and bool(done_text), "output_text.done text is empty"
    )

    terminal_response = terminal[0].payload.get("response")
    if not isinstance(terminal_response, Mapping):
        raise VerificationError("terminal response is missing")
    terminal_text = _text_from_response(terminal_response)
    reconstructed = "".join(str(event.payload["delta"]) for event in deltas)
    _require(
        reconstructed == done_text == terminal_text,
        "delta, done, and terminal text differ",
    )
    _require(terminal_response.get("model") == model, "terminal model alias changed")

    created = [
        event for event in events if event.payload.get("type") == "response.created"
    ]
    _require(
        len(created) == 1, f"expected one response.created, received {len(created)}"
    )
    created_response = created[0].payload.get("response")
    if not isinstance(created_response, Mapping):
        raise VerificationError("created response is missing")
    _require(created_response.get("model") == model, "created model alias changed")

    return {
        "event_count": len(events),
        "sequence_numbers": numbers,
        "text_delta_count": len(deltas),
        "text_delta_receive_count": len({event.receive_index for event in deltas}),
        "terminal_count": len(terminal),
        "reconstructed_text": reconstructed,
        "done_text": done_text,
        "terminal_text": terminal_text,
        "events": _event_receipt(events),
    }


async def probe_openai_non_stream(config: VerificationConfig) -> dict[str, Any]:
    async with AsyncOpenAI(
        api_key=config.api_key,
        base_url=_api_base(config.base_url),
        timeout=config.timeout_s,
        max_retries=0,
    ) as client:
        response = await client.responses.create(
            model=config.model,
            input=config.prompt,
            max_output_tokens=config.max_output_tokens,
        )
    _require(
        response.status == "completed", f"unexpected response status: {response.status}"
    )
    _require(response.model == config.model, "non-stream model alias changed")
    _require(bool(response.output_text), "non-stream output is empty")
    return {
        "response_id": response.id,
        "status": response.status,
        "model": response.model,
        "output_text": response.output_text,
    }


async def probe_openai_sse(config: VerificationConfig) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {config.api_key}"}
    payload = {
        "model": config.model,
        "input": config.prompt,
        "max_output_tokens": config.max_output_tokens,
        "stream": True,
    }
    started_monotonic_ns = time.monotonic_ns()
    timeout = httpx.Timeout(config.timeout_s, connect=min(config.timeout_s, 10.0))
    async with (
        httpx.AsyncClient(timeout=timeout, trust_env=False) as client,
        client.stream(
            "POST",
            f"{config.base_url.rstrip('/')}/v1/responses",
            headers=headers,
            json=payload,
        ) as response,
    ):
        response.raise_for_status()
        events, done_count = await _read_raw_sse(
            response,
            started_monotonic_ns=started_monotonic_ns,
        )
    _require(done_count == 1, f"expected one SSE [DONE] marker, received {done_count}")
    result = _validate_openai_events(
        events,
        model=config.model,
        require_separate_receives=True,
    )
    result["done_marker_count"] = done_count
    return result


async def probe_openai_websocket(config: VerificationConfig) -> dict[str, Any]:
    events: list[ReceivedEvent] = []
    started_monotonic_ns = time.monotonic_ns()
    async with (
        AsyncOpenAI(
            api_key=config.api_key,
            base_url=_api_base(config.base_url),
            timeout=config.timeout_s,
            max_retries=0,
        ) as client,
        client.responses.connect(max_retries=0) as connection,
    ):
        command: Any = {
            "type": "response.create",
            "stream_id": config.stream_id,
            "model": config.model,
            "input": config.prompt,
            "max_output_tokens": config.max_output_tokens,
        }
        await connection.send(command)
        while True:
            sdk_event = await connection.recv()
            received_ns = time.time_ns()
            received_monotonic_ns = time.monotonic_ns()
            payload = _model_dump(sdk_event)
            events.append(
                ReceivedEvent(
                    payload=payload,
                    received_at_unix_ns=received_ns,
                    received_at_monotonic_ns=received_monotonic_ns,
                    elapsed_ms=(received_monotonic_ns - started_monotonic_ns)
                    / 1_000_000,
                    receive_index=len(events) + 1,
                )
            )
            if payload.get("type") in TERMINAL_RESPONSE_EVENTS:
                break
    result = _validate_openai_events(
        events,
        model=config.model,
        require_separate_receives=False,
        stream_id=config.stream_id,
    )
    result.update({"socket_count": 1, "stream_id": config.stream_id})
    return result


def _anthropic_text(message: Any) -> str:
    parts: list[str] = []
    for block in getattr(message, "content", ()):
        if getattr(block, "type", None) == "text":
            text = getattr(block, "text", None)
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def _usage_receipt(usage: Any) -> dict[str, int | None]:
    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
    }


def _validate_usage(usage: Any) -> None:
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    _require(
        isinstance(input_tokens, int) and input_tokens >= 0, "input usage is missing"
    )
    _require(
        isinstance(output_tokens, int) and output_tokens > 0, "output usage is missing"
    )


async def probe_anthropic_non_stream(config: VerificationConfig) -> dict[str, Any]:
    async with anthropic.AsyncAnthropic(
        api_key=config.anthropic_api_key,
        base_url=_anthropic_base(config),
        timeout=config.timeout_s,
        max_retries=0,
    ) as client:
        message = await client.messages.create(
            model=config.model,
            max_tokens=config.max_output_tokens,
            messages=[{"role": "user", "content": config.prompt}],
        )
    text = _anthropic_text(message)
    _require(message.model == config.model, "Anthropic non-stream model alias changed")
    _require(bool(text), "Anthropic non-stream output is empty")
    _validate_usage(message.usage)
    return {
        "message_id": message.id,
        "model": message.model,
        "stop_reason": message.stop_reason,
        "output_text": text,
        "usage": _usage_receipt(message.usage),
    }


def _validate_anthropic_lifecycle(events: Sequence[ReceivedEvent]) -> None:
    timestamps = [event.received_at_monotonic_ns for event in events]
    _require(
        timestamps == sorted(timestamps),
        "Anthropic event receive timestamps are not monotonic",
    )
    types = [str(event.payload.get("type") or "") for event in events]
    _require(types.count("message_start") == 1, "expected one message_start")
    _require(types.count("message_delta") == 1, "expected one message_delta")
    _require(types.count("message_stop") == 1, "expected one message_stop")
    _require(types[0] == "message_start", "message_start must be first")
    _require(types[-1] == "message_stop", "message_stop must be last")
    message_delta_index = types.index("message_delta")
    _require(
        message_delta_index == len(types) - 2,
        "message_delta must immediately precede message_stop",
    )

    open_blocks: set[int] = set()
    closed_blocks: set[int] = set()
    for position, event in enumerate(events):
        event_type = event.payload.get("type")
        if event_type == "content_block_start":
            index = event.payload.get("index")
            if not isinstance(index, int):
                raise VerificationError("content_block_start omitted index")
            _require(
                index not in open_blocks | closed_blocks, "content block started twice"
            )
            open_blocks.add(index)
        elif event_type == "content_block_delta":
            index = event.payload.get("index")
            _require(
                index in open_blocks, "content delta arrived outside an open block"
            )
        elif event_type == "content_block_stop":
            index = event.payload.get("index")
            if not isinstance(index, int):
                raise VerificationError("content_block_stop omitted index")
            _require(index in open_blocks, "content block stopped before it started")
            open_blocks.remove(index)
            closed_blocks.add(index)
        elif event_type == "message_delta":
            _require(
                not open_blocks, "message_delta arrived before content blocks closed"
            )
            _require(
                position == message_delta_index, "unexpected message_delta position"
            )
        elif event_type not in {"message_start", "message_stop", "ping"}:
            raise VerificationError(f"unexpected Anthropic event: {event_type}")
    _require(bool(closed_blocks), "Anthropic stream contained no content block")
    _require(not open_blocks, "Anthropic stream left a content block open")


async def probe_anthropic_sse(config: VerificationConfig) -> dict[str, Any]:
    events: list[ReceivedEvent] = []
    started_monotonic_ns = time.monotonic_ns()
    async with (
        anthropic.AsyncAnthropic(
            api_key=config.anthropic_api_key,
            base_url=_anthropic_base(config),
            timeout=config.timeout_s,
            max_retries=0,
        ) as client,
        client.messages.stream(
            model=config.model,
            max_tokens=config.max_output_tokens,
            messages=[{"role": "user", "content": config.prompt}],
        ) as stream,
    ):
        async for sdk_event in stream:
            received_ns = time.time_ns()
            received_monotonic_ns = time.monotonic_ns()
            events.append(
                ReceivedEvent(
                    payload=_model_dump(sdk_event),
                    received_at_unix_ns=received_ns,
                    received_at_monotonic_ns=received_monotonic_ns,
                    elapsed_ms=(received_monotonic_ns - started_monotonic_ns)
                    / 1_000_000,
                    receive_index=len(events) + 1,
                )
            )
        final_message = await stream.get_final_message()

    protocol_types = {
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
        "ping",
        "error",
    }
    protocol_events = [
        event for event in events if event.payload.get("type") in protocol_types
    ]
    _validate_anthropic_lifecycle(protocol_events)
    starts = [
        event
        for event in protocol_events
        if event.payload.get("type") == "message_start"
    ]
    start_message = starts[0].payload.get("message")
    if not isinstance(start_message, Mapping):
        raise VerificationError("message_start omitted its message")
    _require(
        start_message.get("model") == config.model,
        "Anthropic stream model alias changed",
    )
    _require(final_message.model == config.model, "Anthropic final model alias changed")
    _validate_usage(final_message.usage)

    text_deltas = []
    for event in protocol_events:
        if event.payload.get("type") != "content_block_delta":
            continue
        delta = event.payload.get("delta")
        if isinstance(delta, Mapping) and delta.get("type") == "text_delta":
            text = delta.get("text")
            if isinstance(text, str) and text:
                text_deltas.append(text)
    _require(
        len(text_deltas) >= 2,
        f"expected at least two Anthropic text deltas, received {len(text_deltas)}",
    )
    reconstructed = "".join(text_deltas)
    final_text = _anthropic_text(final_message)
    _require(reconstructed == final_text, "Anthropic delta and final text differ")
    return {
        "sdk_event_count": len(events),
        "protocol_event_count": len(protocol_events),
        "event_types": [event.payload.get("type") for event in protocol_events],
        "sdk_synthetic_event_types": [
            event.payload.get("type")
            for event in events
            if event.payload.get("type") not in protocol_types
        ],
        "text_delta_count": len(text_deltas),
        "reconstructed_text": reconstructed,
        "final_text": final_text,
        "model": final_message.model,
        "usage": _usage_receipt(final_message.usage),
        "events": _event_receipt(protocol_events),
    }


async def probe_safe_fetch_public(
    config: VerificationConfig,
    fetcher: PublicFetcher,
) -> dict[str, Any]:
    resource = await fetcher.fetch(
        config.public_url,
        accepted_media_types=config.accepted_media_types,
        max_bytes=config.safe_fetch_max_bytes,
    )
    _require(bool(resource.content), "public URL returned an empty body")
    return {
        "fetch_status": "accepted",
        "requested_url": _safe_url(config.public_url),
        "final_url": _safe_url(resource.final_url),
        "media_type": resource.media_type,
        "byte_count": len(resource.content),
    }


async def probe_safe_fetch_private_redirect(
    config: VerificationConfig,
    fetcher: PublicFetcher,
) -> dict[str, Any]:
    try:
        await fetcher.fetch(
            config.private_redirect_url,
            accepted_media_types=config.accepted_media_types,
            max_bytes=config.safe_fetch_max_bytes,
        )
    except SafePublicFetchError as error:
        _require(
            error.code == "url_target_blocked",
            f"unexpected rejection code: {error.code}",
        )
        return {
            "fetch_status": "rejected",
            "requested_url": _safe_url(config.private_redirect_url),
            "error_type": type(error).__name__,
            "error_code": error.code,
            "error": str(error),
        }
    raise VerificationError("private redirect unexpectedly succeeded")


async def _run_probe(name: str, timeout_s: float, probe: Probe) -> dict[str, Any]:
    started_ns = time.time_ns()
    started_perf = time.perf_counter_ns()
    try:
        async with asyncio.timeout(timeout_s):
            details = await probe()
    except TimeoutError:
        finished_ns = time.time_ns()
        return {
            "ok": False,
            "status": "timeout",
            "started_at_unix_ns": started_ns,
            "finished_at_unix_ns": finished_ns,
            "duration_ms": (time.perf_counter_ns() - started_perf) / 1_000_000,
            "error": f"{name} exceeded {timeout_s:g}s",
        }
    except Exception as error:
        finished_ns = time.time_ns()
        return {
            "ok": False,
            "status": "failed",
            "started_at_unix_ns": started_ns,
            "finished_at_unix_ns": finished_ns,
            "duration_ms": (time.perf_counter_ns() - started_perf) / 1_000_000,
            "error": _safe_error(error),
        }
    finished_ns = time.time_ns()
    return {
        "ok": True,
        "status": "passed",
        "started_at_unix_ns": started_ns,
        "finished_at_unix_ns": finished_ns,
        "duration_ms": (time.perf_counter_ns() - started_perf) / 1_000_000,
        "details": details,
    }


def _receipt_config(config: VerificationConfig) -> dict[str, Any]:
    return {
        "base_url": _safe_url(config.base_url),
        "anthropic_base_url": _safe_url(_anthropic_base(config)),
        "model": config.model,
        "public_url": _safe_url(config.public_url),
        "private_redirect_url": _safe_url(config.private_redirect_url),
        "max_output_tokens": config.max_output_tokens,
        "timeout_s": config.timeout_s,
        "safe_fetch_max_bytes": config.safe_fetch_max_bytes,
        "accepted_media_types": list(config.accepted_media_types),
        "stream_id": config.stream_id,
        "credentials": "redacted",
    }


def write_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


async def run_verification(
    config: VerificationConfig,
    *,
    safe_fetcher: PublicFetcher | None = None,
) -> dict[str, Any]:
    started_ns = time.time_ns()
    fetcher = safe_fetcher or SafePublicFetch(
        limits=SafePublicFetchLimits(
            max_bytes=config.safe_fetch_max_bytes,
            timeout=config.timeout_s,
            connect_timeout=min(config.timeout_s, 10.0),
            write_timeout=min(config.timeout_s, 10.0),
            pool_timeout=min(config.timeout_s, 10.0),
        )
    )
    probes: tuple[tuple[str, Probe], ...] = (
        ("openai_non_stream", lambda: probe_openai_non_stream(config)),
        ("openai_sse", lambda: probe_openai_sse(config)),
        ("openai_websocket", lambda: probe_openai_websocket(config)),
        ("anthropic_non_stream", lambda: probe_anthropic_non_stream(config)),
        ("anthropic_sse", lambda: probe_anthropic_sse(config)),
        ("safe_public_fetch_public", lambda: probe_safe_fetch_public(config, fetcher)),
        (
            "safe_public_fetch_private_redirect",
            lambda: probe_safe_fetch_private_redirect(config, fetcher),
        ),
    )
    results: dict[str, Any] = {}
    for name, probe in probes:
        results[name] = await _run_probe(name, config.timeout_s, probe)

    finished_ns = time.time_ns()
    overall = all(bool(result.get("ok")) for result in results.values())
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "finalized": True,
        "overall": overall,
        "started_at_unix_ns": started_ns,
        "finished_at_unix_ns": finished_ns,
        "duration_ms": (finished_ns - started_ns) / 1_000_000,
        "config": _receipt_config(config),
        "probes": results,
    }
    write_receipt(config.receipt_path, receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        required=True,
        help="server root, for example http://127.0.0.1:10240",
    )
    parser.add_argument(
        "--anthropic-base-url",
        help="override the default <base-url>/anthropic SDK root",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--api-key-env",
        required=True,
        help="environment variable containing the OpenAI-compatible API key",
    )
    parser.add_argument(
        "--anthropic-api-key-env",
        help="Anthropic key environment variable; defaults to --api-key-env",
    )
    parser.add_argument("--public-url", required=True)
    parser.add_argument("--private-redirect-url", required=True)
    parser.add_argument(
        "--receipt",
        required=True,
        type=Path,
        help="durable JSON receipt path",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-output-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--safe-fetch-max-bytes", type=int, default=2 * 1024 * 1024)
    parser.add_argument(
        "--safe-fetch-media-type",
        action="append",
        dest="accepted_media_types",
    )
    parser.add_argument("--stream-id", default="live_acceptance")
    return parser


def _configuration_failure(
    args: argparse.Namespace, error: Exception
) -> dict[str, Any]:
    now = time.time_ns()
    return {
        "schema_version": SCHEMA_VERSION,
        "finalized": True,
        "overall": False,
        "started_at_unix_ns": now,
        "finished_at_unix_ns": now,
        "duration_ms": 0.0,
        "config": {
            "base_url": _safe_url(args.base_url),
            "model": args.model,
            "credentials": "redacted",
        },
        "probes": {
            "configuration": {
                "ok": False,
                "status": "failed",
                "started_at_unix_ns": now,
                "finished_at_unix_ns": now,
                "duration_ms": 0.0,
                "error": _safe_error(error),
            }
        },
    }


async def _run_cli(args: argparse.Namespace) -> int:
    try:
        api_key = os.environ[args.api_key_env]
        anthropic_key_env = args.anthropic_api_key_env or args.api_key_env
        anthropic_api_key = os.environ[anthropic_key_env]
        config = VerificationConfig(
            base_url=args.base_url.rstrip("/"),
            anthropic_base_url=args.anthropic_base_url,
            model=args.model,
            api_key=api_key,
            anthropic_api_key=anthropic_api_key,
            public_url=args.public_url,
            private_redirect_url=args.private_redirect_url,
            receipt_path=args.receipt,
            prompt=args.prompt,
            max_output_tokens=args.max_output_tokens,
            timeout_s=args.timeout,
            safe_fetch_max_bytes=args.safe_fetch_max_bytes,
            accepted_media_types=tuple(
                args.accepted_media_types or DEFAULT_MEDIA_TYPES
            ),
            stream_id=args.stream_id,
        )
    except (KeyError, ValueError) as error:
        receipt = _configuration_failure(args, error)
        write_receipt(args.receipt, receipt)
        print(json.dumps(receipt, sort_keys=True))
        return 1

    receipt = await run_verification(config)
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt["overall"] else 1


def main() -> int:
    return asyncio.run(_run_cli(_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
