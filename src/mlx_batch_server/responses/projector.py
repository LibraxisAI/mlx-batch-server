"""OpenAI Responses wire projection from protocol-neutral turn events."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from ..runtime.events import (
    REASONING_CONTENT_KIND,
    ContentPartCompleted,
    ContentPartStarted,
    OutputItemCompleted,
    OutputItemStarted,
    ProgressUpdate,
    ReasoningCompleted,
    ReasoningDelta,
    TextCompleted,
    TextDelta,
    ToolCompleted,
    ToolDelta,
    TurnCancelled,
    TurnCompleted,
    TurnFailed,
    TurnStarted,
    UsageUpdate,
)

if TYPE_CHECKING:
    from .transport import TransportEnvelope


def project_event(envelope: TransportEnvelope) -> dict[str, Any]:
    """Project one turn event onto an official Responses streaming event type."""

    event = envelope.event
    base: dict[str, Any] = {"sequence_number": envelope.sequence_number}
    if envelope.stream_id is not None:
        base["stream_id"] = envelope.stream_id.value

    if isinstance(event, TurnStarted):
        return {
            **base,
            "type": "response.created",
            "response": {
                "id": event.response_id,
                "object": "response",
                "created_at": event.created_at,
                "model": event.model,
                "status": "in_progress",
                "output": [],
                "usage": None,
            },
        }
    if isinstance(event, OutputItemStarted):
        return {
            **base,
            "type": "response.output_item.added",
            "output_index": event.index,
            "item": _output_item(event.kind, event.item_id, "in_progress"),
        }
    if isinstance(event, OutputItemCompleted):
        return {
            **base,
            "type": "response.output_item.done",
            "output_index": event.index,
            "item": _completed_output_item(event),
        }
    if isinstance(event, ContentPartStarted):
        if event.kind == REASONING_CONTENT_KIND:
            return {
                **base,
                "type": "response.reasoning_summary_part.added",
                "item_id": event.item_id,
                "output_index": event.output_index,
                "summary_index": event.content_index,
                "part": {"type": "summary_text", "text": ""},
            }
        return {
            **base,
            "type": "response.content_part.added",
            "item_id": event.item_id,
            "output_index": event.output_index,
            "content_index": event.content_index,
            "part": {
                "type": "output_text",
                "text": "",
                "annotations": [],
                "logprobs": [],
            },
        }
    if isinstance(event, ContentPartCompleted):
        if event.kind == REASONING_CONTENT_KIND:
            return {
                **base,
                "type": "response.reasoning_summary_part.done",
                "item_id": event.item_id,
                "output_index": event.output_index,
                "summary_index": event.content_index,
                "part": {"type": "summary_text", "text": event.text},
            }
        return {
            **base,
            "type": "response.content_part.done",
            "item_id": event.item_id,
            "output_index": event.output_index,
            "content_index": event.content_index,
            "part": {
                "type": "output_text",
                "text": event.text,
                "annotations": [],
                "logprobs": [],
            },
        }
    if isinstance(event, ReasoningDelta):
        return {
            **base,
            "type": "response.reasoning_summary_text.delta",
            "item_id": event.item_id,
            "output_index": event.output_index,
            "summary_index": event.content_index,
            "delta": event.delta,
        }
    if isinstance(event, ReasoningCompleted):
        return {
            **base,
            "type": "response.reasoning_summary_text.done",
            "item_id": event.item_id,
            "output_index": event.output_index,
            "summary_index": event.content_index,
            "text": event.text,
        }
    if isinstance(event, TextDelta):
        return {
            **base,
            "type": "response.output_text.delta",
            "item_id": event.item_id,
            "output_index": event.output_index,
            "content_index": event.content_index,
            "delta": event.delta,
            "logprobs": [],
        }
    if isinstance(event, TextCompleted):
        return {
            **base,
            "type": "response.output_text.done",
            "item_id": event.item_id,
            "output_index": event.output_index,
            "content_index": event.content_index,
            "text": event.text,
            "logprobs": [],
        }
    if isinstance(event, ToolDelta):
        return {
            **base,
            "type": "response.function_call_arguments.delta",
            "item_id": event.item_id,
            "output_index": event.index,
            "delta": event.arguments_delta,
        }
    if isinstance(event, ToolCompleted):
        return {
            **base,
            "type": "response.function_call_arguments.done",
            "item_id": event.item_id,
            "output_index": event.index,
            "name": event.name,
            "arguments": event.arguments,
        }
    if isinstance(event, UsageUpdate):
        return {
            **base,
            "type": "response.in_progress",
            "response": {
                "status": "in_progress",
                "usage": _usage(event),
            },
        }
    if isinstance(event, ProgressUpdate):
        return {
            **base,
            "type": "response.in_progress",
            "response": {"status": "in_progress"},
        }
    if isinstance(event, TurnCompleted):
        incomplete = event.finish_reason == "length"
        response: dict[str, Any] = {
            "status": "incomplete" if incomplete else "completed"
        }
        if incomplete:
            response["incomplete_details"] = {"reason": "max_output_tokens"}
        if event.usage is not None:
            response["usage"] = _usage(event.usage)
        event_type = "response.incomplete" if incomplete else "response.completed"
        return {**base, "type": event_type, "response": response}
    if isinstance(event, TurnFailed):
        return {
            **base,
            "type": "response.failed",
            "response": {
                "status": "failed",
                "error": {"message": event.error, "code": event.code},
            },
        }
    if isinstance(event, TurnCancelled):
        return {
            **base,
            "type": "response.incomplete",
            "response": {
                "status": "incomplete",
                "incomplete_details": {"reason": event.reason},
            },
        }
    raise TypeError(f"unsupported turn event: {type(event).__name__}")


def encode_sse(envelope: TransportEnvelope) -> bytes:
    projected = project_event(envelope)
    projected.pop("stream_id", None)
    event_type = str(projected["type"])
    payload = json.dumps(projected, separators=(",", ":"), ensure_ascii=False)
    return f"event: {event_type}\ndata: {payload}\n\n".encode()


def encode_websocket(envelope: TransportEnvelope) -> str:
    """Encode the exact projected object used as the SSE data payload."""

    return json.dumps(
        project_event(envelope),
        separators=(",", ":"),
        ensure_ascii=False,
    )


def encode_sse_done() -> bytes:
    return b"data: [DONE]\n\n"


def _usage(event: UsageUpdate) -> dict[str, int]:
    return {
        "input_tokens": event.input_tokens,
        "output_tokens": event.output_tokens,
        "total_tokens": event.total_tokens,
    }


def _output_item(kind: str, item_id: str, status: str) -> dict[str, Any]:
    if kind == "message":
        return {
            "id": item_id,
            "type": "message",
            "status": status,
            "role": "assistant",
            "content": [],
        }
    if kind == "reasoning":
        return {
            "id": item_id,
            "type": "reasoning",
            "status": status,
            "summary": [],
        }
    return {
        "id": item_id,
        "type": "function_call",
        "status": status,
        "arguments": "",
    }


def _completed_output_item(event: OutputItemCompleted) -> dict[str, Any]:
    if event.kind == "message":
        return {
            "id": event.item_id,
            "type": "message",
            "status": event.status,
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": event.text,
                    "annotations": [],
                    "logprobs": [],
                }
            ],
        }
    if event.kind == "reasoning":
        return {
            "id": event.item_id,
            "type": "reasoning",
            "status": event.status,
            "summary": [{"type": "summary_text", "text": event.text}],
        }
    return {
        "id": event.item_id,
        "type": "function_call",
        "status": event.status,
        "call_id": event.call_id,
        "name": event.name,
        "arguments": event.arguments,
    }
