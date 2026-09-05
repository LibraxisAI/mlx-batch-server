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
import openai
from openai import APIStatusError, AsyncOpenAI, OpenAI

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
ANTHROPIC_SDK_VERSION = "0.96.0"
_TINY_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9Zg70"
    "AAAAASUVORK5CYII="
)
_DOCUMENT_BASE64 = "V2F2ZSA0IEFudGhyb3BpYyBwdWJsaWMgY29tcGF0aWJpbGl0eSByZWNlaXB0Lg=="
_THINKING_BLOCK_TYPES = frozenset(
    {"thinking", "thinking_delta", "signature_delta", "redacted_thinking"}
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


@dataclass(frozen=True, slots=True)
class AnthropicMatrixCase:
    case_id: str
    overrides: Mapping[str, Any]
    expected_field: str


Probe = Callable[[], Awaitable[dict[str, Any]]]


def _unsupported_anthropic_cases() -> tuple[AnthropicMatrixCase, ...]:
    text = {"type": "text", "text": "matrix probe"}
    return (
        AnthropicMatrixCase(
            "cache_control",
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "matrix probe",
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    }
                ]
            },
            "messages.0.content.0.cache_control",
        ),
        AnthropicMatrixCase("container", {"container": "container_1"}, "container"),
        AnthropicMatrixCase("inference_geo", {"inference_geo": "us"}, "inference_geo"),
        AnthropicMatrixCase(
            "output_config_format",
            {
                "output_config": {
                    "format": {
                        "type": "json_schema",
                        "schema": {"type": "object"},
                    }
                }
            },
            "output_config.format",
        ),
        AnthropicMatrixCase("effort_high", {"effort": "high"}, "effort"),
        AnthropicMatrixCase("effort_max", {"effort": "max"}, "effort"),
        AnthropicMatrixCase(
            "citations_enabled",
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "document",
                                "source": {
                                    "type": "base64",
                                    "media_type": "text/plain",
                                    "data": _DOCUMENT_BASE64,
                                },
                                "citations": {"enabled": True},
                            }
                        ],
                    }
                ]
            },
            "messages.0.content.0.citations.enabled",
        ),
        AnthropicMatrixCase(
            "server_tool_use",
            {
                "messages": [
                    {"role": "user", "content": "search"},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "server_tool_use",
                                "id": "srvtoolu_1",
                                "name": "web_search",
                                "input": {"query": "local"},
                            }
                        ],
                    },
                    {"role": "user", "content": "continue"},
                ]
            },
            "messages.1.content.0.type",
        ),
        AnthropicMatrixCase(
            "hosted_tool_definition",
            {"tools": [{"type": "web_search_20250305", "name": "web_search"}]},
            "tools.0.type",
        ),
        AnthropicMatrixCase(
            "hosted_tool_result",
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "web_search_tool_result",
                                "tool_use_id": "srvtoolu_1",
                                "content": [{"title": "result"}],
                            }
                        ],
                    }
                ]
            },
            "messages.0.content.0.type",
        ),
        AnthropicMatrixCase(
            "container_upload",
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "container_upload", "file_id": "file_1"}],
                    }
                ]
            },
            "messages.0.content.0.type",
        ),
        AnthropicMatrixCase(
            "thinking_budget_min",
            {"thinking": {"type": "enabled", "budget_tokens": 1024}},
            "thinking.type",
        ),
        AnthropicMatrixCase(
            "thinking_budget_large",
            {"thinking": {"type": "enabled", "budget_tokens": 4096}},
            "thinking.type",
        ),
        AnthropicMatrixCase(
            "thinking_continuation",
            {
                "messages": [
                    {"role": "user", "content": "first"},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "thinking",
                                "thinking": "carried reasoning",
                                "signature": "sig_untrusted",
                            },
                            text,
                        ],
                    },
                    {"role": "user", "content": "continue"},
                ]
            },
            "messages.1.content.0.type",
        ),
        AnthropicMatrixCase(
            "redacted_thinking",
            {
                "messages": [
                    {"role": "user", "content": "first"},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "redacted_thinking",
                                "data": "opaque_untrusted",
                            },
                            text,
                        ],
                    },
                    {"role": "user", "content": "continue"},
                ]
            },
            "messages.1.content.0.type",
        ),
    )


def _matrix_cell_id(case_id: str, stream: bool) -> str:
    return f"{case_id}.{'stream' if stream else 'unary'}"


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


def _redact_receipt(
    receipt: Mapping[str, Any], secrets: Sequence[str]
) -> dict[str, Any]:
    active_secrets = tuple(secret for secret in secrets if secret)

    def redact(value: Any) -> Any:
        if isinstance(value, str):
            for secret in active_secrets:
                value = value.replace(secret, "[redacted]")
            return value
        if isinstance(value, Mapping):
            return {key: redact(item) for key, item in value.items()}
        if isinstance(value, list):
            return [redact(item) for item in value]
        if isinstance(value, tuple):
            return [redact(item) for item in value]
        return value

    redacted = redact(receipt)
    if not isinstance(redacted, dict):
        raise TypeError("receipt redaction changed the root shape")
    return redacted


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


def _sync_input_item_ids(
    config: VerificationConfig,
    response_id: str,
    *,
    limit: int,
    order: str,
) -> list[str]:
    with OpenAI(
        api_key=config.api_key,
        base_url=_api_base(config.base_url),
        timeout=config.timeout_s,
        max_retries=0,
    ) as client:
        return [
            item.id
            for item in client.responses.input_items.list(
                response_id,
                limit=limit,
                order=order,  # type: ignore[arg-type]
            )
        ]


async def probe_openai_public_admission(  # noqa: PLR0915 - one receipt spans the lifecycle
    config: VerificationConfig,
) -> dict[str, Any]:
    """Exercise the complete public Responses lifecycle with official clients."""

    input_items: list[dict[str, Any]] = [
        {
            "type": "message",
            "role": "user",
            "content": (
                f"context item {index}"
                if index % 2 == 0
                else [{"type": "input_text", "text": f"context item {index}"}]
            ),
        }
        for index in range(25)
    ]
    headers = {"Authorization": f"Bearer {config.api_key}"}
    started = time.perf_counter_ns()
    async with AsyncOpenAI(
        api_key=config.api_key,
        base_url=_api_base(config.base_url),
        timeout=config.timeout_s,
        max_retries=0,
    ) as client:
        created = await client.responses.create(
            model=config.model,
            input=input_items,  # type: ignore[arg-type]
            background=True,
            max_output_tokens=config.max_output_tokens,
        )
        create_latency_ms = (time.perf_counter_ns() - started) / 1_000_000
        _require(
            created.status in {"queued", "in_progress"},
            "background create did not return before terminal completion",
        )
        _require(created.background is True, "background policy was not echoed")
        _require(created.model == config.model, "background model alias changed")

        first_get = await client.responses.retrieve(created.id)
        _require(
            first_get.status in {"queued", "in_progress"},
            "first background GET did not expose a live lifecycle state",
        )
        statuses = [str(created.status), str(first_get.status)]

        async_ascending = [
            item.id
            async for item in client.responses.input_items.list(
                created.id,
                limit=6,
                order="asc",
            )
        ]
        sync_descending = await asyncio.to_thread(
            _sync_input_item_ids,
            config,
            created.id,
            limit=7,
            order="desc",
        )
        _require(len(async_ascending) == 25, "async pagination lost input items")
        _require(len(sync_descending) == 25, "sync pagination lost input items")
        _require(
            len(set(async_ascending)) == len(async_ascending),
            "async pagination returned duplicate IDs",
        )
        _require(
            sync_descending == list(reversed(async_ascending)),
            "sync and async pagination disagree on canonical order",
        )

        deadline = time.monotonic() + max(0.1, config.timeout_s / 2)
        terminal = first_get
        while terminal.status in {"queued", "in_progress"}:
            if time.monotonic() >= deadline:
                raise VerificationError(
                    "background response did not reach terminal state"
                )
            await asyncio.sleep(0.02)
            terminal = await client.responses.retrieve(created.id)
            statuses.append(str(terminal.status))
        _require(terminal.status == "completed", "background response did not complete")
        _require(terminal.model == config.model, "terminal model alias changed")

        cancellable = await client.responses.create(
            model=config.model,
            input="Generate a deliberately long answer for cancellation verification.",
            background=True,
            max_output_tokens=config.max_output_tokens,
        )
        _require(
            cancellable.status in {"queued", "in_progress"},
            "cancellation fixture completed before cancel admission",
        )
        cancelled = await client.responses.cancel(cancellable.id)
        cancelled_again = await client.responses.cancel(cancellable.id)
        _require(cancelled.status == "cancelled", "background cancel did not settle")
        _require(
            cancelled_again.id == cancelled.id
            and cancelled_again.status == cancelled.status,
            "background cancel was not idempotent",
        )

        unsupported_error: Mapping[str, Any] | None = None
        try:
            await client.responses.create(
                model=config.model,
                input="unsupported field",
                max_tool_calls=1,
            )
        except APIStatusError as error:
            payload = error.response.json()
            if isinstance(payload, Mapping):
                nested = payload.get("error")
                if isinstance(nested, Mapping):
                    unsupported_error = nested
        if unsupported_error is None:
            raise VerificationError("unsupported field was accepted")
        _require(
            unsupported_error.get("code") == "unsupported_parameter"
            and unsupported_error.get("param") == "max_tool_calls",
            "unsupported field returned the wrong typed error",
        )

        await client.responses.delete(created.id)

    timeout = httpx.Timeout(config.timeout_s, connect=min(config.timeout_s, 10.0))
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as raw_client:
        malformed = await raw_client.post(
            f"{config.base_url.rstrip('/')}/v1/responses",
            headers={**headers, "Content-Type": "application/json"},
            content=b'{"model":',
        )
        deleted_get = await raw_client.get(
            f"{config.base_url.rstrip('/')}/v1/responses/{created.id}",
            headers=headers,
        )
    malformed_payload = malformed.json()
    _require(malformed.status_code == 400, "malformed JSON did not return HTTP 400")
    _require(
        isinstance(malformed_payload, Mapping)
        and set(malformed_payload) == {"error"}
        and "detail" not in malformed_payload,
        "malformed JSON leaked a framework error envelope",
    )
    malformed_error = malformed_payload.get("error")
    _require(
        isinstance(malformed_error, Mapping)
        and set(malformed_error) == {"message", "type", "param", "code"}
        and malformed_error.get("code") == "invalid_json",
        "malformed JSON returned the wrong typed error",
    )
    _require(deleted_get.status_code == 404, "deleted response remained retrievable")

    return {
        "sdk_version": openai.__version__,
        "background_response_id": created.id,
        "background_create_latency_ms": create_latency_ms,
        "lifecycle_statuses": statuses,
        "pagination_item_count": len(async_ascending),
        "pagination_first_id": async_ascending[0],
        "pagination_last_id": async_ascending[-1],
        "pagination_unique_ids": len(set(async_ascending)),
        "cancelled_response_id": cancelled.id,
        "cancel_status": cancelled.status,
        "delete_status": deleted_get.status_code,
        "malformed_error_code": malformed_error.get("code"),
        "unsupported_error_code": unsupported_error.get("code"),
        "unsupported_error_param": unsupported_error.get("param"),
    }


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


def _require_anthropic_sdk_version() -> str:
    version = str(getattr(anthropic, "__version__", ""))
    _require(
        version == ANTHROPIC_SDK_VERSION,
        f"Anthropic SDK must be {ANTHROPIC_SDK_VERSION}, found {version or 'unknown'}",
    )
    return version


def _anthropic_block_types(message: Any) -> list[str]:
    return [
        str(block_type)
        for block in getattr(message, "content", ())
        if (block_type := getattr(block, "type", None)) is not None
    ]


def _anthropic_message_receipt(message: Any, *, model: str) -> dict[str, Any]:
    text = _anthropic_text(message)
    block_types = _anthropic_block_types(message)
    _require(message.model == model, "Anthropic non-stream model alias changed")
    _require(bool(text), "Anthropic non-stream output is empty")
    _require(
        not (_THINKING_BLOCK_TYPES & set(block_types)),
        "omitted thinking emitted a thinking or signature block",
    )
    _validate_usage(message.usage)
    _require(
        getattr(message.usage, "service_tier", None) == "standard",
        "Anthropic response did not report actual service tier standard",
    )
    return {
        "message_id": message.id,
        "model": message.model,
        "stop_reason": message.stop_reason,
        "stop_sequence": message.stop_sequence,
        "content_block_types": block_types,
        "output_text": text,
        "usage": {
            **_usage_receipt(message.usage),
            "service_tier": getattr(message.usage, "service_tier", None),
        },
    }


async def probe_anthropic_sync_non_stream(
    config: VerificationConfig,
) -> dict[str, Any]:
    version = _require_anthropic_sdk_version()

    def run() -> dict[str, Any]:
        with anthropic.Anthropic(
            api_key=config.anthropic_api_key,
            base_url=_anthropic_base(config),
            timeout=config.timeout_s,
            max_retries=0,
        ) as client:
            message = client.messages.create(
                model=config.model,
                max_tokens=config.max_output_tokens,
                messages=[{"role": "user", "content": config.prompt}],
            )
        return _anthropic_message_receipt(message, model=config.model)

    result = await asyncio.to_thread(run)
    result["sdk_version"] = version
    result["client_mode"] = "sync"
    return result


async def probe_anthropic_non_stream(config: VerificationConfig) -> dict[str, Any]:
    version = _require_anthropic_sdk_version()
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
    result = _anthropic_message_receipt(message, model=config.model)
    result["sdk_version"] = version
    result["client_mode"] = "async"
    return result


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


def _anthropic_stream_receipt(
    events: Sequence[ReceivedEvent],
    final_message: Any,
    *,
    model: str,
) -> dict[str, Any]:
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
        start_message.get("model") == model, "Anthropic stream model alias changed"
    )
    _require(final_message.model == model, "Anthropic final model alias changed")
    _validate_usage(final_message.usage)
    _require(
        getattr(final_message.usage, "service_tier", None) == "standard",
        "Anthropic final stream message did not report service tier standard",
    )

    text_deltas: list[str] = []
    forbidden_types: set[str] = set()
    for event in protocol_events:
        payload = event.payload
        event_type = str(payload.get("type") or "")
        if event_type in _THINKING_BLOCK_TYPES:
            forbidden_types.add(event_type)
        content_block = payload.get("content_block")
        if isinstance(content_block, Mapping):
            block_type = str(content_block.get("type") or "")
            if block_type in _THINKING_BLOCK_TYPES:
                forbidden_types.add(block_type)
        delta = payload.get("delta")
        if not isinstance(delta, Mapping):
            continue
        delta_type = str(delta.get("type") or "")
        if delta_type in _THINKING_BLOCK_TYPES:
            forbidden_types.add(delta_type)
        if event_type == "content_block_delta" and delta_type == "text_delta":
            text = delta.get("text")
            if isinstance(text, str) and text:
                text_deltas.append(text)
    final_block_types = _anthropic_block_types(final_message)
    forbidden_types.update(_THINKING_BLOCK_TYPES & set(final_block_types))
    _require(
        not forbidden_types,
        f"omitted thinking emitted forbidden block/event types: {sorted(forbidden_types)}",
    )
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
        "content_block_types": final_block_types,
        "forbidden_thinking_types": sorted(forbidden_types),
        "text_delta_count": len(text_deltas),
        "reconstructed_text": reconstructed,
        "final_text": final_text,
        "model": final_message.model,
        "usage": {
            **_usage_receipt(final_message.usage),
            "service_tier": getattr(final_message.usage, "service_tier", None),
        },
        "events": _event_receipt(protocol_events),
    }


async def probe_anthropic_sync_sse(config: VerificationConfig) -> dict[str, Any]:
    version = _require_anthropic_sdk_version()

    def run() -> dict[str, Any]:
        events: list[ReceivedEvent] = []
        started_monotonic_ns = time.monotonic_ns()
        with (
            anthropic.Anthropic(
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
            for sdk_event in stream:
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
            final_message = stream.get_final_message()
        return _anthropic_stream_receipt(events, final_message, model=config.model)

    result = await asyncio.to_thread(run)
    result["sdk_version"] = version
    result["client_mode"] = "sync"
    return result


async def probe_anthropic_sse(config: VerificationConfig) -> dict[str, Any]:
    version = _require_anthropic_sdk_version()
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
    result = _anthropic_stream_receipt(events, final_message, model=config.model)
    result["sdk_version"] = version
    result["client_mode"] = "async"
    return result


def _anthropic_body(config: VerificationConfig) -> dict[str, Any]:
    return {
        "model": config.model,
        "max_tokens": config.max_output_tokens,
        "messages": [{"role": "user", "content": config.prompt}],
    }


def _body_with_overrides(
    config: VerificationConfig,
    overrides: Mapping[str, Any],
    *,
    stream: bool,
) -> dict[str, Any]:
    body = _anthropic_body(config)
    body.update(json.loads(json.dumps(dict(overrides))))
    body["stream"] = stream
    return body


def _anthropic_headers(config: VerificationConfig) -> dict[str, str]:
    return {
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
        "x-api-key": config.anthropic_api_key,
    }


def _sse_payloads(content: bytes) -> tuple[list[dict[str, Any]], int]:
    normalized = content.replace(b"\r\n", b"\n")
    payloads: list[dict[str, Any]] = []
    event_bytes = 0
    for line in normalized.splitlines(keepends=True):
        if line.startswith((b"event:", b"data:")):
            event_bytes += len(line)
    for block in normalized.split(b"\n\n"):
        if not block.strip():
            continue
        payload = _parse_sse_block(block)
        if payload is not None and payload.get("type") != "[DONE]":
            payloads.append(payload)
    return payloads, event_bytes


def _raw_anthropic_success(
    response: httpx.Response,
    *,
    model: str,
    stream: bool,
) -> dict[str, Any]:
    _require(
        response.status_code == 200, f"expected HTTP 200, got {response.status_code}"
    )
    if not stream:
        payload = response.json()
        if not isinstance(payload, dict):
            raise VerificationError("Anthropic response is not object-shaped")
        content = payload.get("content")
        if not isinstance(content, list) or not content:
            raise VerificationError("response content is empty")
        unary_block_types = [
            str(block.get("type") or "")
            for block in content
            if isinstance(block, Mapping)
        ]
        _require(
            not (_THINKING_BLOCK_TYPES & set(unary_block_types)),
            "supported request emitted thinking/signature content",
        )
        usage = payload.get("usage")
        usage = usage if isinstance(usage, Mapping) else {}
        return {
            "http_status": response.status_code,
            "content_type": response.headers.get("content-type", ""),
            "model": payload.get("model"),
            "content_block_types": unary_block_types,
            "service_tier": usage.get("service_tier"),
            "stop_reason": payload.get("stop_reason"),
            "stop_sequence": payload.get("stop_sequence"),
            "output_text": "".join(
                str(block.get("text") or "")
                for block in content
                if isinstance(block, Mapping) and block.get("type") == "text"
            ),
        }

    _require(
        response.headers.get("content-type", "").startswith("text/event-stream"),
        "stream=true did not return an SSE content type",
    )
    payloads, event_bytes = _sse_payloads(response.content)
    now = time.monotonic_ns()
    events = [
        ReceivedEvent(payload, now + index, now + index, float(index), index + 1)
        for index, payload in enumerate(payloads)
    ]
    _validate_anthropic_lifecycle(events)
    _require(
        not any(payload.get("type") == "error" for payload in payloads),
        "supported Anthropic stream emitted an error event",
    )
    starts = [payload for payload in payloads if payload.get("type") == "message_start"]
    start_message = starts[0].get("message")
    start_message = start_message if isinstance(start_message, Mapping) else {}
    usage = start_message.get("usage")
    usage = usage if isinstance(usage, Mapping) else {}
    stream_block_types: list[str] = []
    text_deltas: list[str] = []
    stop_reason: object = None
    stop_sequence: object = None
    for payload in payloads:
        content_block = payload.get("content_block")
        if isinstance(content_block, Mapping):
            stream_block_types.append(str(content_block.get("type") or ""))
        delta = payload.get("delta")
        if not isinstance(delta, Mapping):
            continue
        delta_type = str(delta.get("type") or "")
        if delta_type == "text_delta" and isinstance(delta.get("text"), str):
            text_deltas.append(str(delta["text"]))
        if payload.get("type") == "message_delta":
            stop_reason = delta.get("stop_reason")
            stop_sequence = delta.get("stop_sequence")
    wire_types = set(stream_block_types)
    wire_types.update(
        str(payload.get("delta", {}).get("type") or "")
        for payload in payloads
        if isinstance(payload.get("delta"), Mapping)
    )
    _require(
        not (_THINKING_BLOCK_TYPES & wire_types),
        "supported stream emitted thinking/signature content",
    )
    return {
        "http_status": response.status_code,
        "content_type": response.headers.get("content-type", ""),
        "model": start_message.get("model"),
        "content_block_types": stream_block_types,
        "service_tier": usage.get("service_tier"),
        "stop_reason": stop_reason,
        "stop_sequence": stop_sequence,
        "output_text": "".join(text_deltas),
        "event_types": [payload.get("type") for payload in payloads],
        "sse_event_bytes": event_bytes,
    }


def _supported_anthropic_cells(
    config: VerificationConfig,
) -> tuple[tuple[str, dict[str, Any], bool], ...]:
    rich_content = [
        {"type": "text", "text": "first"},
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": _TINY_PNG_BASE64,
            },
        },
        {"type": "text", "text": "second"},
        {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "text/plain",
                "data": _DOCUMENT_BASE64,
            },
            "title": "admission.txt",
        },
        {
            "type": "search_result",
            "source": "https://must-not-fetch.invalid/result",
            "title": "caller supplied",
            "content": [{"type": "text", "text": "third"}],
        },
        {"type": "text", "text": "fourth; answer briefly"},
    ]
    tool_messages = [
        {"role": "user", "content": "use the lookup tool"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_admission_1",
                    "name": "lookup",
                    "input": {"key": "alpha"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_admission_1",
                    "is_error": False,
                    "content": [
                        {"type": "text", "text": "result-alpha"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": _TINY_PNG_BASE64,
                            },
                        },
                    ],
                }
            ],
        },
    ]
    cells: list[tuple[str, dict[str, Any], bool]] = []
    for tier in ("auto", "standard_only"):
        cells.append((f"service_tier_{tier}.unary", {"service_tier": tier}, False))
    for label, overrides in (
        ("thinking_omitted", {}),
        ("thinking_disabled", {"thinking": {"type": "disabled"}}),
        (
            "rich_content_order",
            {"messages": [{"role": "user", "content": rich_content}]},
        ),
        (
            "stop_sequence",
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Reply exactly: alpha beta gamma delta epsilon.",
                    }
                ],
                "stop_sequences": [" delta"],
            },
        ),
        ("tool_result_fidelity", {"messages": tool_messages}),
    ):
        for stream in (False, True):
            cells.append((_matrix_cell_id(label, stream), dict(overrides), stream))
    return tuple(cells)


async def probe_anthropic_supported_matrix(
    config: VerificationConfig,
) -> dict[str, Any]:
    timeout = httpx.Timeout(config.timeout_s, connect=min(config.timeout_s, 10.0))
    cells: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        for cell_id, overrides, stream in _supported_anthropic_cells(config):
            body = _body_with_overrides(config, overrides, stream=stream)
            response = await client.post(
                f"{_anthropic_base(config)}/v1/messages",
                headers=_anthropic_headers(config),
                json=body,
            )
            details = _raw_anthropic_success(
                response,
                model=config.model,
                stream=stream,
            )
            if cell_id.startswith("service_tier_"):
                _require(
                    details["service_tier"] == "standard",
                    f"{cell_id} did not report actual tier standard",
                )
            if cell_id.startswith("stop_sequence."):
                _require(
                    details["stop_reason"] == "stop_sequence"
                    and details["stop_sequence"] == " delta"
                    and " delta" not in str(details["output_text"]),
                    f"{cell_id} did not preserve exact stop-sequence semantics",
                )
            cells.append(
                {
                    "cell_id": cell_id,
                    "stream": stream,
                    "request_content_types": [
                        block.get("type")
                        for message in body["messages"]
                        if isinstance(message.get("content"), list)
                        for block in message["content"]
                        if isinstance(block, Mapping)
                    ],
                    **details,
                }
            )
    return {
        "required_cell_ids": [cell[0] for cell in _supported_anthropic_cells(config)],
        "cells": cells,
    }


def _inference_start_counter(payload: Mapping[str, Any]) -> int:
    try:
        executor = payload["role_runtime"]["runtime_stats"]["executor"]
        active = executor["active_requests"]
        tombstones = executor["tombstones"]
    except (KeyError, TypeError) as error:
        raise VerificationError(
            "health omits role_runtime.runtime_stats.executor inference counters"
        ) from error
    if (
        not isinstance(active, int)
        or isinstance(active, bool)
        or active < 0
        or not isinstance(tombstones, int)
        or isinstance(tombstones, bool)
        or tombstones < 0
    ):
        raise VerificationError("health inference counters are invalid")
    return active + tombstones


async def _live_inference_start_counter(
    client: httpx.AsyncClient,
    config: VerificationConfig,
) -> int:
    response = await client.get(f"{config.base_url.rstrip('/')}/health")
    _require(
        response.status_code == 200, "health counter probe did not return HTTP 200"
    )
    payload = response.json()
    if not isinstance(payload, Mapping):
        raise VerificationError("health counter response is not object-shaped")
    return _inference_start_counter(payload)


async def probe_anthropic_refusal_matrix(
    config: VerificationConfig,
) -> dict[str, Any]:
    timeout = httpx.Timeout(config.timeout_s, connect=min(config.timeout_s, 10.0))
    cells: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        for case in _unsupported_anthropic_cases():
            for stream in (False, True):
                cell_id = _matrix_cell_id(case.case_id, stream)
                before = await _live_inference_start_counter(client, config)
                response = await client.post(
                    f"{_anthropic_base(config)}/v1/messages",
                    headers=_anthropic_headers(config),
                    json=_body_with_overrides(config, case.overrides, stream=stream),
                )
                after = await _live_inference_start_counter(client, config)
                payloads, event_bytes = _sse_payloads(response.content)
                _require(
                    response.status_code == 400,
                    f"{cell_id} expected HTTP 400, got {response.status_code}",
                )
                _require(
                    not response.headers.get("content-type", "").startswith(
                        "text/event-stream"
                    ),
                    f"{cell_id} opened an SSE response",
                )
                _require(
                    event_bytes == 0 and not payloads, f"{cell_id} emitted SSE bytes"
                )
                body = response.json()
                _require(
                    isinstance(body, Mapping)
                    and body.get("type") == "error"
                    and isinstance(body.get("error"), Mapping)
                    and body["error"].get("type") == "invalid_request_error",
                    f"{cell_id} did not return the structured invalid_request_error envelope",
                )
                message = str(body["error"].get("message") or "")
                _require(
                    case.expected_field in message,
                    f"{cell_id} error did not name {case.expected_field}",
                )
                header_request_id = response.headers.get("request-id")
                body_request_id = body.get("request_id")
                _require(
                    isinstance(header_request_id, str)
                    and header_request_id.startswith("req_")
                    and header_request_id == body_request_id,
                    f"{cell_id} request-id header/body mismatch",
                )
                inference_start_count = after - before
                _require(
                    inference_start_count == 0,
                    f"{cell_id} crossed the inference boundary ({inference_start_count})",
                )
                cells.append(
                    {
                        "cell_id": cell_id,
                        "stream": stream,
                        "http_status": response.status_code,
                        "error_type": body["error"].get("type"),
                        "field": case.expected_field,
                        "request_id": body_request_id,
                        "request_id_matches_header": True,
                        "sse_event_bytes": event_bytes,
                        "inference_start_count": inference_start_count,
                    }
                )
    required = [
        _matrix_cell_id(case.case_id, stream)
        for case in _unsupported_anthropic_cases()
        for stream in (False, True)
    ]
    return {"required_cell_ids": required, "cells": cells}


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
        "anthropic_sdk_version": str(getattr(anthropic, "__version__", "unknown")),
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


_REQUIRED_PROBE_IDS = (
    "openai_non_stream",
    "openai_sse",
    "openai_websocket",
    "openai_public_admission",
    "anthropic_sdk_sync_non_stream",
    "anthropic_sdk_async_non_stream",
    "anthropic_sdk_sync_sse",
    "anthropic_sdk_async_sse",
    "anthropic_supported_matrix",
    "anthropic_refusal_matrix",
    "safe_public_fetch_public",
    "safe_public_fetch_private_redirect",
)


def _receipt_integrity_errors(
    results: Mapping[str, Any],
    config: VerificationConfig,
    *,
    declared_probe_ids: Sequence[str] | None = None,
) -> list[str]:
    errors: list[str] = []
    actual_probe_ids = list(declared_probe_ids or results)
    if len(actual_probe_ids) != len(set(actual_probe_ids)):
        errors.append("duplicate probe ids")
    missing = sorted(set(_REQUIRED_PROBE_IDS) - set(actual_probe_ids))
    unexpected = sorted(set(actual_probe_ids) - set(_REQUIRED_PROBE_IDS))
    if missing:
        errors.append(f"missing required probes: {missing}")
    if unexpected:
        errors.append(f"unexpected probes: {unexpected}")
    skipped = sorted(
        probe_id
        for probe_id, result in results.items()
        if isinstance(result, Mapping) and result.get("status") == "skipped"
    )
    if skipped:
        errors.append(f"required probes were skipped: {skipped}")
    for probe_id, expected_ids in (
        (
            "anthropic_supported_matrix",
            [cell[0] for cell in _supported_anthropic_cells(config)],
        ),
        (
            "anthropic_refusal_matrix",
            [
                _matrix_cell_id(case.case_id, stream)
                for case in _unsupported_anthropic_cases()
                for stream in (False, True)
            ],
        ),
    ):
        result = results.get(probe_id)
        details = result.get("details") if isinstance(result, Mapping) else None
        cells = details.get("cells") if isinstance(details, Mapping) else None
        actual_ids: list[str] = []
        if isinstance(cells, list):
            actual_ids = [
                cell_id
                for cell in cells
                if isinstance(cell, Mapping)
                and isinstance((cell_id := cell.get("cell_id")), str)
            ]
        if len(actual_ids) != len(set(actual_ids)):
            errors.append(f"{probe_id} contains duplicate matrix cells")
        missing_cells = sorted(set(expected_ids) - set(actual_ids))
        unexpected_cells = sorted(set(actual_ids) - set(expected_ids))
        if missing_cells:
            errors.append(f"{probe_id} missing required cells: {missing_cells}")
        if unexpected_cells:
            errors.append(f"{probe_id} contains unexpected cells: {unexpected_cells}")
        if len(actual_ids) != len(expected_ids):
            errors.append(
                f"{probe_id} cell cardinality is {len(actual_ids)}, expected {len(expected_ids)}"
            )
    return errors


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
        (
            "openai_public_admission",
            lambda: probe_openai_public_admission(config),
        ),
        (
            "anthropic_sdk_sync_non_stream",
            lambda: probe_anthropic_sync_non_stream(config),
        ),
        (
            "anthropic_sdk_async_non_stream",
            lambda: probe_anthropic_non_stream(config),
        ),
        ("anthropic_sdk_sync_sse", lambda: probe_anthropic_sync_sse(config)),
        ("anthropic_sdk_async_sse", lambda: probe_anthropic_sse(config)),
        (
            "anthropic_supported_matrix",
            lambda: probe_anthropic_supported_matrix(config),
        ),
        (
            "anthropic_refusal_matrix",
            lambda: probe_anthropic_refusal_matrix(config),
        ),
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
    integrity_errors = _receipt_integrity_errors(
        results,
        config,
        declared_probe_ids=[name for name, _probe in probes],
    )
    overall = not integrity_errors and all(
        bool(result.get("ok")) for result in results.values()
    )
    receipt = _redact_receipt(
        {
            "schema_version": SCHEMA_VERSION,
            "finalized": True,
            "overall": overall,
            "started_at_unix_ns": started_ns,
            "finished_at_unix_ns": finished_ns,
            "duration_ms": (finished_ns - started_ns) / 1_000_000,
            "config": _receipt_config(config),
            "integrity": {
                "ok": not integrity_errors,
                "required_probe_ids": list(_REQUIRED_PROBE_IDS),
                "errors": integrity_errors,
            },
            "probes": results,
        },
        (config.api_key, config.anthropic_api_key),
    )
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
