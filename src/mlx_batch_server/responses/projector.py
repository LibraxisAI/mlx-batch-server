"""OpenAI Responses wire projection from protocol-neutral turn events."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

from ..runtime.events import (
    REASONING_CONTENT_KIND,
    ContentPartCompleted,
    ContentPartStarted,
    HostedCallCompleted,
    HostedCallProgress,
    HostedCallResult,
    HostedCallStarted,
    HostedCitation,
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
from .transport import (
    render_completed_item,
    render_started_item,
    render_url_citation,
    render_usage,
)

if TYPE_CHECKING:
    from .transport import TransportEnvelope


@dataclass(frozen=True, slots=True)
class NoWireResponseEvent:
    """An admitted neutral event with deliberately no legal OpenAI wire event."""

    sequence_number: int
    reason: Literal["hosted_result_internal", "failed_hosted_lifecycle_absent"]


@dataclass(slots=True)
class OpenAIProjectionState:
    """Per-response wire bookkeeping that is not response snapshot truth."""

    annotation_counts: dict[tuple[int, str, int], int] = field(default_factory=dict)
    annotations: dict[tuple[int, str, int], list[dict[str, Any]]] = field(
        default_factory=dict
    )

    def add_citation(self, event: HostedCitation) -> tuple[int, dict[str, Any]]:
        key = (event.output_index, event.item_id, event.content_index)
        annotation_index = self.annotation_counts.get(key, 0)
        annotation = render_url_citation(event)
        self.annotation_counts[key] = annotation_index + 1
        self.annotations.setdefault(key, []).append(annotation)
        return annotation_index, annotation

    def content_annotations(
        self,
        output_index: int,
        item_id: str,
        content_index: int,
    ) -> tuple[dict[str, Any], ...]:
        return tuple(self.annotations.get((output_index, item_id, content_index), ()))


ProjectedResponseEvent: TypeAlias = dict[str, Any] | NoWireResponseEvent
_HOSTED_EVENT_TYPES = (
    HostedCallStarted,
    HostedCallProgress,
    HostedCallResult,
    HostedCallCompleted,
    HostedCitation,
)


def project_event(
    envelope: TransportEnvelope,
    *,
    state: OpenAIProjectionState | None = None,
) -> ProjectedResponseEvent:
    """Project one turn event onto an official Responses streaming event type."""

    event = envelope.event
    base: dict[str, Any] = {"sequence_number": envelope.sequence_number}
    if envelope.stream_id is not None:
        base["stream_id"] = envelope.stream_id.value

    if isinstance(event, TurnStarted):
        return {
            **base,
            "type": "response.created",
            "response": _snapshot(envelope, "in_progress"),
        }
    if isinstance(event, OutputItemStarted):
        return {
            **base,
            "type": "response.output_item.added",
            "output_index": event.index,
            "item": render_started_item(event),
        }
    if isinstance(event, OutputItemCompleted):
        annotations = ()
        if event.kind == "message" and state is not None:
            annotations = state.content_annotations(event.index, event.item_id, 0)
        return {
            **base,
            "type": "response.output_item.done",
            "output_index": event.index,
            "item": render_completed_item(event, annotations=annotations),
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
                "annotations": (
                    []
                    if state is None
                    else list(
                        state.content_annotations(
                            event.output_index,
                            event.item_id,
                            event.content_index,
                        )
                    )
                ),
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
    if isinstance(event, _HOSTED_EVENT_TYPES):
        return _project_hosted_event(envelope, state=state)
    if isinstance(event, UsageUpdate):
        response = _snapshot(envelope, "in_progress")
        response.setdefault("usage", None)
        if response["usage"] is None:
            response["usage"] = render_usage(event)
        return {**base, "type": "response.in_progress", "response": response}
    if isinstance(event, ProgressUpdate):
        return {
            **base,
            "type": "response.in_progress",
            "response": _snapshot(envelope, "in_progress"),
        }
    if isinstance(event, TurnCompleted):
        incomplete = event.finish_reason == "length"
        status = "incomplete" if incomplete else "completed"
        response = _snapshot(envelope, status)
        if incomplete:
            response["incomplete_details"] = {"reason": "max_output_tokens"}
        if event.usage is not None:
            response["usage"] = render_usage(event.usage)
        event_type = "response.incomplete" if incomplete else "response.completed"
        return {**base, "type": event_type, "response": response}
    if isinstance(event, TurnFailed):
        response = _snapshot(envelope, "failed")
        response["error"] = {"message": event.error, "code": event.code}
        return {**base, "type": "response.failed", "response": response}
    if isinstance(event, TurnCancelled):
        response = _snapshot(envelope, "incomplete")
        response["incomplete_details"] = {"reason": event.reason}
        return {**base, "type": "response.incomplete", "response": response}
    raise TypeError(f"unsupported turn event: {type(event).__name__}")


def _project_hosted_event(
    envelope: TransportEnvelope,
    *,
    state: OpenAIProjectionState | None,
) -> ProjectedResponseEvent:
    event = envelope.event
    base: dict[str, Any] = {"sequence_number": envelope.sequence_number}
    if envelope.stream_id is not None:
        base["stream_id"] = envelope.stream_id.value
    if isinstance(event, HostedCallStarted):
        if event.tool_name != "web_search":
            raise TypeError(f"unsupported hosted tool for Responses: {event.tool_name}")
        return {
            **base,
            "type": "response.web_search_call.in_progress",
            "item_id": event.item_id,
            "output_index": event.index,
        }
    if isinstance(event, HostedCallProgress):
        if event.phase not in {"executing", "searching"}:
            raise TypeError(
                f"unsupported hosted web_search progress phase: {event.phase}"
            )
        return {
            **base,
            "type": "response.web_search_call.searching",
            "item_id": event.item_id,
            "output_index": event.index,
        }
    if isinstance(event, HostedCallResult):
        return NoWireResponseEvent(
            envelope.sequence_number,
            "hosted_result_internal",
        )
    if isinstance(event, HostedCallCompleted):
        if event.tool_name != "web_search":
            raise TypeError(f"unsupported hosted tool for Responses: {event.tool_name}")
        if event.status == "failed":
            return NoWireResponseEvent(
                envelope.sequence_number,
                "failed_hosted_lifecycle_absent",
            )
        return {
            **base,
            "type": "response.web_search_call.completed",
            "item_id": event.item_id,
            "output_index": event.index,
        }
    if isinstance(event, HostedCitation):
        projection_state = state if state is not None else OpenAIProjectionState()
        annotation_index, annotation = projection_state.add_citation(event)
        return {
            **base,
            "type": "response.output_text.annotation.added",
            "item_id": event.item_id,
            "output_index": event.output_index,
            "content_index": event.content_index,
            "annotation_index": annotation_index,
            "annotation": annotation,
        }
    raise TypeError(f"unsupported hosted turn event: {type(event).__name__}")


def _snapshot(envelope: TransportEnvelope, status: str) -> dict[str, Any]:
    """Return the complete Response snapshot this lifecycle event must embed.

    The controller owns the only snapshot fold. Any response-embedding event
    without that complete published snapshot is an integrity failure.
    """

    snapshot = envelope.snapshot
    if snapshot is None:
        raise TypeError("response-embedding event is missing its full snapshot")
    projected = dict(snapshot)
    projected["status"] = status
    return projected


def encode_sse(
    envelope: TransportEnvelope,
    *,
    state: OpenAIProjectionState | None = None,
) -> bytes | None:
    projected = project_event(envelope, state=state)
    if isinstance(projected, NoWireResponseEvent):
        return None
    projected.pop("stream_id", None)
    event_type = str(projected["type"])
    payload = json.dumps(projected, separators=(",", ":"), ensure_ascii=False)
    return f"event: {event_type}\ndata: {payload}\n\n".encode()


def encode_websocket(
    envelope: TransportEnvelope,
    *,
    state: OpenAIProjectionState | None = None,
) -> str | None:
    """Encode the exact projected object used as the SSE data payload."""

    projected = project_event(envelope, state=state)
    if isinstance(projected, NoWireResponseEvent):
        return None
    return json.dumps(
        projected,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def encode_sse_done() -> bytes:
    return b"data: [DONE]\n\n"
