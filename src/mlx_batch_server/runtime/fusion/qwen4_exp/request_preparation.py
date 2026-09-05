"""Backend-owned Qwen4Exp request preparation before admission.

The canonical Responses mapper remains synchronous and resolution-free. This
module reconstructs message-local mixed content, performs injected asynchronous
media resolution, and seals the result before any scheduler or tensor mutation.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ....vision.input import (
    MultimodalInputCapabilities,
    MultimodalInputPlan,
    MultimodalInputPlanner,
)
from ...contracts import (
    CancelToken,
    GenerationRequest,
    PreparedGenerationRequest,
    RequestModality,
)
from .media import (
    TERMINAL_ITEM_STATUSES,
    PreparedPromptItem,
    PreparedQwen4Message,
    PreparedQwen4Prompt,
    Qwen4PromptBuilder,
    bind_prepared_messages,
)

if TYPE_CHECKING:
    from .media_resolver import ResolvedMediaBundle

# Every field that only a typed `function_call` may own. Anything else carrying
# one is refused rather than silently dropped, so nothing reaches the seal
# unauthenticated.
_CALL_ONLY_FIELDS = ("id", "status", "name", "arguments")


class Qwen4ExpRequestPreparationError(ValueError):
    """The canonical request cannot be sealed for Qwen4Exp execution."""


@runtime_checkable
class Qwen4ExpMediaResolverPort(Protocol):
    async def resolve(self, plan: MultimodalInputPlan) -> ResolvedMediaBundle: ...


@runtime_checkable
class Qwen4ExpRequestPreparerPort(Protocol):
    async def prepare(
        self,
        request: GenerationRequest,
        cancel: CancelToken,
    ) -> PreparedGenerationRequest: ...


@dataclass(frozen=True, slots=True)
class _MessageLayout:
    message_index: int
    role: str
    part_indices: tuple[int, ...]
    item_type: str | None = None
    id: str | None = None
    status: str | None = None
    call_id: str | None = None
    output: str | None = None
    is_error: bool | None = None
    name: str | None = None
    arguments: str | None = None


class Qwen4ExpRequestPreparer:
    """Resolve and seal one request without entering the inference mailbox."""

    def __init__(
        self,
        *,
        resolver: Qwen4ExpMediaResolverPort,
        capabilities: MultimodalInputCapabilities,
        builder: Qwen4PromptBuilder | None = None,
    ) -> None:
        if not isinstance(capabilities, MultimodalInputCapabilities):
            raise TypeError("capabilities must be MultimodalInputCapabilities")
        self._resolver = resolver
        self._planner = MultimodalInputPlanner(capabilities)
        self._builder = builder or Qwen4PromptBuilder()

    async def prepare(
        self,
        request: GenerationRequest,
        cancel: CancelToken,
    ) -> PreparedGenerationRequest:
        if not isinstance(request, GenerationRequest):
            raise TypeError("request must be a GenerationRequest")
        _raise_if_cancelled(cancel)
        if not request.media:
            return PreparedGenerationRequest(
                request=request,
                modality=RequestModality.TEXT,
            )

        parts, layouts = _reconstruct_mixed_messages(request)
        plan = self._planner.plan(parts)
        if not plan.media:
            raise Qwen4ExpRequestPreparationError(
                "canonical request media did not produce media descriptors"
            )
        resolution = await self._resolver.resolve(plan)
        _raise_if_cancelled(cancel)
        prompt = self._builder.build(
            response_id=request.response_id,
            runtime=request.runtime,
            plan=plan,
            resolution=resolution,
        )
        prompt = bind_prepared_messages(
            prompt,
            _prepared_messages(prompt, layouts),
        )
        return PreparedGenerationRequest(
            request=request,
            modality=RequestModality.VISION,
            backend_payload=prompt,
        )


def _reconstruct_mixed_messages(
    request: GenerationRequest,
) -> tuple[tuple[Mapping[str, object], ...], tuple[_MessageLayout, ...]]:
    messages = tuple(request.messages)
    media_by_message: dict[int, dict[int, Mapping[str, object]]] = {}
    for raw_media in request.media:
        if not isinstance(raw_media, Mapping):
            raise Qwen4ExpRequestPreparationError("media item must be a mapping")
        message_index = _required_index(raw_media, "_message_index")
        content_index = _required_index(raw_media, "_content_index")
        if message_index >= len(messages):
            raise Qwen4ExpRequestPreparationError(
                "media item refers to a missing message"
            )
        role = raw_media.get("_role")
        if not isinstance(role, str) or not role.strip():
            raise Qwen4ExpRequestPreparationError(
                "media item is missing canonical role provenance"
            )
        slots = media_by_message.setdefault(message_index, {})
        if content_index in slots:
            raise Qwen4ExpRequestPreparationError(
                "media items reuse one message content index"
            )
        slots[content_index] = raw_media

    parts: list[Mapping[str, object]] = []
    layouts: list[_MessageLayout] = []
    for message_index, raw_message in enumerate(messages):
        if not isinstance(raw_message, Mapping):
            raise Qwen4ExpRequestPreparationError("message must be a mapping")
        role_value = raw_message.get("role")
        if not isinstance(role_value, str) or not role_value.strip():
            raise Qwen4ExpRequestPreparationError("message role must not be empty")
        role = role_value.strip().lower()
        indexed_media = media_by_message.get(message_index, {})
        item_type = raw_message.get("type")
        call_id: str | None = None
        output: str | None = None
        is_error: bool | None = None
        if item_type not in (
            None,
            "message",
            "function_call",
            "function_call_output",
        ):
            raise Qwen4ExpRequestPreparationError(
                "canonical message type is unsupported"
            )
        if item_type == "function_call":
            # The call keeps its exact typed identity through preparation; it is
            # never flattened into an ordinary assistant text message.
            layouts.append(
                _function_call_layout(
                    raw_message,
                    message_index=message_index,
                    role=role,
                    media=indexed_media,
                )
            )
            continue
        if any(field in raw_message for field in _CALL_ONLY_FIELDS):
            raise Qwen4ExpRequestPreparationError(
                "id, status, name and arguments belong only to function_call"
            )
        text_parts = _canonical_text_parts(raw_message.get("content"))
        if item_type == "function_call_output":
            call_id, output, is_error = _function_call_output_identity(
                raw_message,
                role=role,
                text_parts=text_parts,
            )
        elif "call_id" in raw_message or "output" in raw_message:
            raise Qwen4ExpRequestPreparationError(
                "call_id and output belong only to function_call_output"
            )
        elif "is_error" in raw_message:
            raise Qwen4ExpRequestPreparationError(
                "is_error belongs only to function_call_output"
            )
        for item in indexed_media.values():
            if str(item.get("_role", "")).strip().lower() != role:
                raise Qwen4ExpRequestPreparationError(
                    "media role provenance does not match its message"
                )
        content_count = len(text_parts) + len(indexed_media)
        if any(index >= content_count for index in indexed_media):
            raise Qwen4ExpRequestPreparationError(
                "media content index is outside the reconstructed message"
            )

        text_cursor = 0
        part_indices: list[int] = []
        for content_index in range(content_count):
            raw_part = indexed_media.get(content_index)
            if raw_part is None:
                if text_cursor >= len(text_parts):
                    raise Qwen4ExpRequestPreparationError(
                        "message content provenance has an unfillable gap"
                    )
                part: Mapping[str, object] = text_parts[text_cursor]
                text_cursor += 1
            else:
                part = {
                    key: value
                    for key, value in raw_part.items()
                    if not str(key).startswith("_")
                }
            part_indices.append(len(parts))
            parts.append(part)
        if text_cursor != len(text_parts):
            raise Qwen4ExpRequestPreparationError(
                "message text provenance was not consumed exactly once"
            )
        layouts.append(
            _MessageLayout(
                message_index=message_index,
                role=role,
                part_indices=tuple(part_indices),
                item_type=item_type,
                call_id=call_id,
                output=output,
                is_error=is_error,
            )
        )

    if set(media_by_message) - set(range(len(messages))):
        raise Qwen4ExpRequestPreparationError("media refers to a foreign message")
    _validate_call_lineage(layouts)
    return tuple(parts), tuple(layouts)


def _function_call_layout(
    raw_message: Mapping[str, object],
    *,
    message_index: int,
    role: str,
    media: Mapping[int, Mapping[str, object]],
) -> _MessageLayout:
    """Admit one typed tool call, or refuse it — never degrade it."""

    if role != "assistant":
        raise Qwen4ExpRequestPreparationError(
            "function_call must keep its assistant role"
        )
    if media:
        raise Qwen4ExpRequestPreparationError("function_call cannot own media")
    if raw_message.get("content"):
        raise Qwen4ExpRequestPreparationError(
            "function_call cannot carry message content"
        )
    for field in ("output", "is_error"):
        if field in raw_message:
            raise Qwen4ExpRequestPreparationError(
                f"{field} belongs only to function_call_output"
            )
    call_id = raw_message.get("call_id")
    if not isinstance(call_id, str) or not call_id.strip():
        raise Qwen4ExpRequestPreparationError("function_call call_id must not be empty")
    name = raw_message.get("name")
    if not isinstance(name, str) or not name.strip():
        raise Qwen4ExpRequestPreparationError("function_call name must not be empty")
    arguments = raw_message.get("arguments")
    if not isinstance(arguments, str):
        raise Qwen4ExpRequestPreparationError("function_call arguments must be text")
    return _MessageLayout(
        message_index=message_index,
        role=role,
        part_indices=(),
        item_type="function_call",
        id=_optional_call_identity(raw_message, "id"),
        status=_terminal_call_status(raw_message),
        call_id=call_id,
        name=name,
        arguments=arguments,
    )


def _optional_call_identity(
    raw_message: Mapping[str, object],
    field: str,
) -> str | None:
    """Absent means absent; present means a normalized non-blank identity."""

    if field not in raw_message:
        return None
    value = raw_message.get(field)
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise Qwen4ExpRequestPreparationError(
            f"function_call {field} must be a normalized identity"
        )
    return value


def _terminal_call_status(raw_message: Mapping[str, object]) -> str | None:
    if "status" not in raw_message:
        return None
    status = raw_message.get("status")
    if status not in TERMINAL_ITEM_STATUSES:
        raise Qwen4ExpRequestPreparationError(
            "function_call status must be a terminal item status"
        )
    return str(status)


def _validate_call_lineage(layouts: Sequence[_MessageLayout]) -> None:
    """Refuse call/result lineage no typed call in this request can explain.

    A request whose upstream still flattens calls into assistant text carries no
    typed identity to judge, and its lineage was validated before it arrived.
    Once typed calls are present the whole chain is judged: one call per
    ``call_id``, one result per call, and no result for a call that never
    completed.
    """

    calls: dict[str, _MessageLayout] = {}
    for layout in layouts:
        if layout.item_type != "function_call" or layout.call_id is None:
            continue
        if layout.call_id in calls:
            raise Qwen4ExpRequestPreparationError("function_call call_id is duplicated")
        calls[layout.call_id] = layout
    if not calls:
        return
    identities = [layout.id for layout in calls.values() if layout.id is not None]
    if len(set(identities)) != len(identities):
        raise Qwen4ExpRequestPreparationError("function_call id is duplicated")

    seen: set[str] = set()
    settled: set[str] = set()
    for layout in layouts:
        if layout.item_type == "function_call" and layout.call_id is not None:
            seen.add(layout.call_id)
            continue
        if layout.item_type != "function_call_output":
            continue
        call_id = layout.call_id
        if call_id is None or call_id not in seen:
            raise Qwen4ExpRequestPreparationError(
                "function_call_output has no preceding function_call"
            )
        if call_id in settled:
            raise Qwen4ExpRequestPreparationError(
                "function_call_output is duplicated for one function_call"
            )
        if calls[call_id].status == "incomplete":
            raise Qwen4ExpRequestPreparationError(
                "function_call_output follows a function_call that never completed"
            )
        settled.add(call_id)


def _function_call_output_identity(
    raw_message: Mapping[str, object],
    *,
    role: str,
    text_parts: tuple[Mapping[str, object], ...],
) -> tuple[str, str, bool | None]:
    call_id_value = raw_message.get("call_id")
    output_value = raw_message.get("output")
    if role != "tool":
        raise Qwen4ExpRequestPreparationError(
            "function_call_output must use the tool role"
        )
    if not isinstance(call_id_value, str) or not call_id_value.strip():
        raise Qwen4ExpRequestPreparationError(
            "function_call_output call_id must not be empty"
        )
    if not isinstance(output_value, str):
        raise Qwen4ExpRequestPreparationError(
            "function_call_output output must be text"
        )
    texts = tuple(part.get("text") for part in text_parts)
    if texts:
        if len(texts) == 1:
            if texts[0] != output_value:
                raise Qwen4ExpRequestPreparationError(
                    "function_call_output content must preserve its exact output"
                )
        elif "\n".join(str(item) for item in texts) != output_value:
            raise Qwen4ExpRequestPreparationError(
                "function_call_output content must preserve its exact output"
            )
    elif output_value:
        raise Qwen4ExpRequestPreparationError(
            "function_call_output content must preserve its exact output"
        )
    is_error: bool | None
    if "is_error" not in raw_message:
        is_error = None
    else:
        is_error_value = raw_message.get("is_error")
        if not isinstance(is_error_value, bool):
            raise Qwen4ExpRequestPreparationError(
                "function_call_output is_error must be a boolean"
            )
        is_error = is_error_value
    return call_id_value, output_value, is_error


def _canonical_text_parts(content: object) -> tuple[Mapping[str, object], ...]:
    if isinstance(content, str):
        return ({"type": "input_text", "text": content},)
    if not isinstance(content, Sequence) or isinstance(content, str | bytes):
        raise Qwen4ExpRequestPreparationError(
            "canonical message content must be text or a sequence"
        )
    result: list[Mapping[str, object]] = []
    for part in content:
        if not isinstance(part, Mapping) or part.get("type") != "input_text":
            raise Qwen4ExpRequestPreparationError(
                "request.messages may contain only canonical text parts"
            )
        result.append(dict(part))
    return tuple(result)


def _prepared_messages(
    prompt: PreparedQwen4Prompt,
    layouts: tuple[_MessageLayout, ...],
) -> tuple[PreparedQwen4Message, ...]:
    by_part: dict[int, list[PreparedPromptItem]] = {}
    for item in prompt.items:
        by_part.setdefault(item.part_index, []).append(item)
    messages: list[PreparedQwen4Message] = []
    consumed: list[PreparedPromptItem] = []
    for layout in layouts:
        items = tuple(
            item
            for part_index in layout.part_indices
            for item in by_part.get(part_index, ())
        )
        consumed.extend(items)
        messages.append(
            PreparedQwen4Message(
                message_index=layout.message_index,
                role=layout.role,
                items=items,
                item_type=layout.item_type,
                id=layout.id,
                status=layout.status,
                call_id=layout.call_id,
                output=layout.output,
                is_error=layout.is_error,
                name=layout.name,
                arguments=layout.arguments,
            )
        )
    if tuple(consumed) != prompt.items:
        raise Qwen4ExpRequestPreparationError(
            "prepared prompt items do not match reconstructed messages"
        )
    return tuple(messages)


def _required_index(value: Mapping[str, object], field: str) -> int:
    index = value.get(field)
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise Qwen4ExpRequestPreparationError(
            f"media {field} must be a non-negative integer"
        )
    return index


def _raise_if_cancelled(cancel: CancelToken) -> None:
    if cancel.cancelled:
        raise asyncio.CancelledError(cancel.reason or "request preparation cancelled")


__all__ = [
    "Qwen4ExpMediaResolverPort",
    "Qwen4ExpRequestPreparationError",
    "Qwen4ExpRequestPreparer",
    "Qwen4ExpRequestPreparerPort",
]
