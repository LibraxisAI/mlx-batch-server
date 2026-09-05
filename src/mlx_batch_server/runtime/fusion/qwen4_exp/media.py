"""Whole-request Qwen4Exp prompt assembly after bounded media resolution.

Source fetching, file expansion, and all byte/image/pixel limits belong to
``SourceMediaResolver``. This module only validates that one complete bundle
matches one immutable input plan and preserves prompt order for the future
Qwen4Exp preprocessor. It never resolves one descriptor in isolation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import TypeAlias

from ....vision.input import (
    AudioInput,
    FileInput,
    ImageInput,
    MediaInput,
    MultimodalInputPlan,
    PromptText,
    VideoInput,
)
from ...contracts import RuntimeKey
from .media_resolver import (
    ResolvedImage,
    ResolvedMediaBundle,
    ResolvedMediaItem,
    ResolvedText,
    source_identity_digest,
)


@dataclass(frozen=True, slots=True)
class PreparedTextItem:
    part_index: int
    text: str

    def __post_init__(self) -> None:
        if self.part_index < 0:
            raise ValueError("text part_index must be non-negative")
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("prepared text must not be empty")


@dataclass(frozen=True, slots=True)
class PreparedMediaItem:
    """One ordered image or text produced from an input media descriptor."""

    part_index: int
    item_index: int
    descriptor: ImageInput | FileInput
    resolution: ResolvedMediaItem

    def __post_init__(self) -> None:
        if self.part_index != self.descriptor.part_index:
            raise ValueError("prepared media descriptor identity mismatch")
        if self.part_index != self.resolution.part_index:
            raise ValueError("prepared media resolution part mismatch")
        if self.item_index != self.resolution.item_index:
            raise ValueError("prepared media resolution item mismatch")


PreparedPromptItem: TypeAlias = PreparedTextItem | PreparedMediaItem


@dataclass(frozen=True, slots=True)
class PreparedQwen4Message:
    """One role-preserving message after whole-request media expansion."""

    message_index: int
    role: str
    items: tuple[PreparedPromptItem, ...]
    item_type: str | None = None
    call_id: str | None = None
    output: str | None = None
    is_error: bool | None = None
    name: str | None = None
    arguments: str | None = None

    def __post_init__(self) -> None:
        if self.message_index < 0:
            raise ValueError("message_index must be non-negative")
        if not isinstance(self.role, str) or not self.role.strip():
            raise ValueError("prepared message role must not be empty")
        object.__setattr__(self, "role", self.role.strip().lower())
        object.__setattr__(self, "items", tuple(self.items))
        if self.item_type not in (
            None,
            "message",
            "function_call",
            "function_call_output",
        ):
            raise ValueError("prepared message item_type is unsupported")
        if self.item_type != "function_call" and (
            self.name is not None or self.arguments is not None
        ):
            raise ValueError("name and arguments belong only to function_call")
        if self.item_type == "function_call":
            _validate_function_call(self)
        elif self.item_type == "function_call_output":
            _validate_function_call_output(self)
        elif self.call_id is not None or self.output is not None:
            raise ValueError("call_id and output belong only to function_call_output")
        elif self.is_error is not None:
            raise ValueError("is_error belongs only to function_call_output")


@dataclass(frozen=True, slots=True)
class PreparedQwen4Prompt:
    """Identity-bound prompt plus its single whole-request media receipt."""

    response_id: str
    runtime: RuntimeKey
    items: tuple[PreparedPromptItem, ...]
    media: tuple[PreparedMediaItem, ...]
    resolution: ResolvedMediaBundle
    content_digest: str
    media_digest: str
    messages: tuple[PreparedQwen4Message, ...] = ()

    def __post_init__(self) -> None:
        if not self.response_id:
            raise ValueError("response_id must not be empty")
        if not isinstance(self.runtime, RuntimeKey):
            raise TypeError("runtime must be a RuntimeKey")
        actual_media = tuple(
            item for item in self.items if isinstance(item, PreparedMediaItem)
        )
        if actual_media != self.media:
            raise ValueError("media must preserve its order in items")
        _require_digest(self.content_digest, "content_digest")
        _require_digest(self.media_digest, "media_digest")
        if self.messages:
            indices = tuple(item.message_index for item in self.messages)
            if indices != tuple(range(len(self.messages))):
                raise ValueError("prepared message indices must be contiguous")
            flattened = tuple(
                item for message in self.messages for item in message.items
            )
            if flattened != self.items:
                raise ValueError(
                    "prepared messages must exactly partition prompt items"
                )
            expected_digest = _messages_digest(self.messages)
            if self.content_digest != expected_digest:
                raise ValueError("content_digest does not seal message layout")


class Qwen4PromptError(ValueError):
    """Structured fail-closed prompt/bundle mismatch."""

    def __init__(self, code: str, message: str, part_index: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.part_index = part_index


class Qwen4PromptBuilder:
    """Join one input plan with one atomically resolved media bundle."""

    def build(
        self,
        *,
        response_id: str,
        runtime: RuntimeKey,
        plan: MultimodalInputPlan,
        resolution: ResolvedMediaBundle,
    ) -> PreparedQwen4Prompt:
        if not response_id:
            raise Qwen4PromptError("invalid_response_id", "response_id is required")
        if not isinstance(runtime, RuntimeKey):
            raise Qwen4PromptError("invalid_runtime", "runtime must be a RuntimeKey")
        if not isinstance(plan, MultimodalInputPlan):
            raise Qwen4PromptError("invalid_plan", "plan must be multimodal")
        if not isinstance(resolution, ResolvedMediaBundle):
            raise Qwen4PromptError(
                "invalid_resolution",
                "resolution must be one whole-request media bundle",
            )

        descriptors = _validate_plan(plan)
        resolved_by_part = _validate_bundle(descriptors, resolution)
        items: list[PreparedPromptItem] = [
            PreparedTextItem(part_index=item.part_index, text=item.text)
            for item in plan.prompt
        ]
        for part_index, descriptor in descriptors.items():
            for resolved in resolved_by_part[part_index]:
                items.append(
                    PreparedMediaItem(
                        part_index=part_index,
                        item_index=resolved.item_index,
                        descriptor=descriptor,
                        resolution=resolved,
                    )
                )

        items.sort(key=_item_order)
        ordered_media = tuple(
            item for item in items if isinstance(item, PreparedMediaItem)
        )
        content_payload = [_content_payload(item) for item in items]
        return PreparedQwen4Prompt(
            response_id=response_id,
            runtime=runtime,
            items=tuple(items),
            media=ordered_media,
            resolution=resolution,
            content_digest=_json_digest(content_payload),
            media_digest=_json_digest(
                {
                    "bundle": resolution.digest,
                    "items": [_media_payload(item) for item in ordered_media],
                }
            ),
        )


def bind_prepared_messages(
    prompt: PreparedQwen4Prompt,
    messages: tuple[PreparedQwen4Message, ...],
) -> PreparedQwen4Prompt:
    """Seal role and message boundaries into an already resolved prompt."""

    normalized = tuple(messages)
    return replace(
        prompt,
        messages=normalized,
        content_digest=_messages_digest(normalized),
    )


def _validate_plan(
    plan: MultimodalInputPlan,
) -> dict[int, ImageInput | FileInput]:
    all_parts: tuple[PromptText | MediaInput, ...] = (*plan.prompt, *plan.media)
    indices = [item.part_index for item in all_parts]
    if any(index < 0 for index in indices) or len(set(indices)) != len(indices):
        raise Qwen4PromptError(
            "invalid_part_order",
            "part indices must be unique and non-negative",
        )
    descriptors: dict[int, ImageInput | FileInput] = {}
    for descriptor in plan.media:
        if isinstance(descriptor, AudioInput | VideoInput):
            raise Qwen4PromptError(
                "unsupported_media",
                "audio and video are outside the 3more launch contract",
                descriptor.part_index,
            )
        if not isinstance(descriptor, ImageInput | FileInput):
            raise Qwen4PromptError(
                "unsupported_media",
                "unknown media descriptor",
                descriptor.part_index,
            )
        descriptors[descriptor.part_index] = descriptor
    return descriptors


def _validate_bundle(
    descriptors: dict[int, ImageInput | FileInput],
    bundle: ResolvedMediaBundle,
) -> dict[int, tuple[ResolvedMediaItem, ...]]:
    item_order = [(item.part_index, item.item_index) for item in bundle.items]
    if item_order != sorted(item_order):
        raise Qwen4PromptError(
            "non_canonical_bundle_order",
            "media bundle items must use canonical order",
        )
    grouped: dict[int, list[ResolvedMediaItem]] = {
        part_index: [] for part_index in descriptors
    }
    seen: set[tuple[int, int]] = set()
    for item in bundle.items:
        key = (item.part_index, item.item_index)
        if key in seen:
            raise Qwen4PromptError(
                "duplicate_resolution_item",
                "media bundle contains a duplicate item identity",
                item.part_index,
            )
        seen.add(key)
        descriptor = descriptors.get(item.part_index)
        if descriptor is None:
            raise Qwen4PromptError(
                "foreign_resolution_part",
                "media bundle contains an unrequested part",
                item.part_index,
            )
        if item.source_digest != source_identity_digest(descriptor):
            raise Qwen4PromptError(
                "source_identity_mismatch",
                "media bundle does not match the requested source",
                item.part_index,
            )
        grouped[item.part_index].append(item)

    result: dict[int, tuple[ResolvedMediaItem, ...]] = {}
    for part_index, descriptor in descriptors.items():
        resolved = sorted(grouped[part_index], key=lambda item: item.item_index)
        if not resolved:
            raise Qwen4PromptError(
                "missing_resolution_part",
                "media bundle omitted a requested part",
                part_index,
            )
        if [item.item_index for item in resolved] != list(range(len(resolved))):
            raise Qwen4PromptError(
                "non_contiguous_resolution_items",
                "file expansion item indices must be contiguous",
                part_index,
            )
        if isinstance(descriptor, ImageInput) and (
            len(resolved) != 1 or not isinstance(resolved[0], ResolvedImage)
        ):
            raise Qwen4PromptError(
                "invalid_image_expansion",
                "input_image must resolve to exactly one image",
                part_index,
            )
        if isinstance(descriptor, FileInput) and not all(
            isinstance(item, ResolvedImage | ResolvedText) for item in resolved
        ):
            raise Qwen4PromptError(
                "invalid_file_expansion",
                "input_file produced an unsupported item",
                part_index,
            )
        result[part_index] = tuple(resolved)
    return result


def _item_order(item: PreparedPromptItem) -> tuple[int, int]:
    if isinstance(item, PreparedTextItem):
        return item.part_index, -1
    return item.part_index, item.item_index


def _content_payload(item: PreparedPromptItem) -> dict[str, object]:
    if isinstance(item, PreparedTextItem):
        return {"part_index": item.part_index, "kind": "text", "text": item.text}
    payload = _media_payload(item)
    payload["kind"] = (
        "image" if isinstance(item.resolution, ResolvedImage) else "file_text"
    )
    return payload


def render_function_call_output_text(
    output: str,
    is_error: bool | None,
) -> str:
    """Template-visible tool receipt. Success text is unchanged."""

    if is_error is not True:
        return output
    return json.dumps(
        {"is_error": True, "content": output},
        ensure_ascii=True,
        separators=(",", ":"),
    )


def _validate_function_call(message: PreparedQwen4Message) -> None:
    if message.role != "assistant":
        raise ValueError("function_call must keep its assistant role")
    if not isinstance(message.call_id, str) or not message.call_id.strip():
        raise ValueError("function_call call_id must not be empty")
    if not isinstance(message.name, str) or not message.name.strip():
        raise ValueError("function_call name must not be empty")
    if not isinstance(message.arguments, str):
        raise ValueError("function_call arguments must be text")
    if message.output is not None or message.is_error is not None:
        raise ValueError("output and is_error belong only to function_call_output")
    if message.items:
        raise ValueError("function_call cannot carry prompt items")


def _validate_function_call_output(message: PreparedQwen4Message) -> None:
    if message.role != "tool":
        raise ValueError("function_call_output must use the tool role")
    if not isinstance(message.call_id, str) or not message.call_id.strip():
        raise ValueError("function_call_output call_id must not be empty")
    if not isinstance(message.output, str):
        raise ValueError("function_call_output output must be text")
    if message.is_error is not None and not isinstance(message.is_error, bool):
        raise ValueError("function_call_output is_error must be a boolean")
    texts = tuple(
        item.text for item in message.items if isinstance(item, PreparedTextItem)
    )
    if texts:
        joined = "\n".join(texts)
        if len(texts) == 1:
            if texts[0] != message.output:
                raise ValueError(
                    "function_call_output items must preserve the exact output"
                )
        elif joined != message.output:
            raise ValueError(
                "function_call_output items must preserve the exact output"
            )
    elif message.output:
        raise ValueError("function_call_output items must preserve the exact output")


def _messages_digest(messages: tuple[PreparedQwen4Message, ...]) -> str:
    return _json_digest(
        [
            {
                "message_index": message.message_index,
                "role": message.role,
                "type": message.item_type,
                "call_id": message.call_id,
                "output": message.output,
                "is_error": message.is_error,
                "items": [_content_payload(item) for item in message.items],
            }
            for message in messages
        ]
    )


def _media_payload(item: PreparedMediaItem) -> dict[str, object]:
    resolved = item.resolution
    return {
        "part_index": item.part_index,
        "item_index": item.item_index,
        "source_digest": resolved.source_digest,
        "content_digest": resolved.content_digest,
    }


def _json_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_digest(value: str, field_name: str) -> None:
    if len(value) != 64 or value != value.lower():
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest") from exc
    if len(decoded) != 32:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


__all__ = [
    "PreparedMediaItem",
    "PreparedPromptItem",
    "PreparedQwen4Message",
    "PreparedQwen4Prompt",
    "PreparedTextItem",
    "Qwen4PromptBuilder",
    "Qwen4PromptError",
    "bind_prepared_messages",
    "render_function_call_output_text",
]
