"""Map a validated Anthropic Messages request onto a protocol-neutral turn.

Two rules govern this module. Anthropic semantics are *mapped*, never
flattened away: a tool result stays a tool result, reasoning stays reasoning.
And anything this runtime cannot actually honour is refused with an explicit
Anthropic error, because a silently dropped field is a lie about what was
inferred.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .anthropic_schema import (
    AnthropicTool,
    InputMessage,
    MessageRole,
    MessagesRequest,
    RequestContentBlock,
    RequestDocumentBlock,
    RequestImageBlock,
    RequestSearchResultBlock,
    RequestTextBlock,
    RequestThinkingBlock,
    RequestToolReferenceBlock,
    RequestToolResultBlock,
    RequestToolUseBlock,
    SystemPrompt,
    ThinkingConfigEnabled,
    ToolChoiceAny,
    ToolChoiceAuto,
    ToolChoiceNone,
    ToolChoiceTool,
)
from .errors import AnthropicAPIError, UnsupportedCapabilityError
from .turn_source import AnthropicTurn

_IMAGE_MEDIA_TYPES = frozenset({"image/jpeg", "image/png", "image/gif", "image/webp"})
_FILE_MEDIA_TYPES = frozenset({"application/pdf", "text/plain"})


def build_turn(request: MessagesRequest) -> AnthropicTurn:
    """Translate one Anthropic request into a runtime-neutral turn."""

    _reject_unsupported(request)
    _validate_tool_conversation(request.messages)
    messages, media = _build_messages(request.system, request.messages)
    tools = tuple(_map_tool(tool) for tool in request.tools or ())
    tool_choice = _map_tool_choice(request)
    return AnthropicTurn(
        model_alias=request.model,
        messages=messages,
        media=media,
        tools=tools,
        tool_choice=tool_choice,
        sampling=_map_sampling(request),
        reasoning=_map_reasoning(request),
        metadata=_map_metadata(request),
    )


# ---------------------------------------------------------------------------
# Fail-closed capability checks
# ---------------------------------------------------------------------------


def _reject_unsupported(request: MessagesRequest) -> None:
    if not request.messages:
        raise AnthropicAPIError(
            "messages must contain at least one entry",
            error_type="invalid_request_error",
        )
    for sequence in request.stop_sequences or ():
        if not sequence:
            raise AnthropicAPIError(
                "stop_sequences entries must not be empty",
                error_type="invalid_request_error",
            )
    for index, message in enumerate(request.messages):
        if isinstance(message.content, str):
            continue
        for block_index, block in enumerate(message.content):
            path = f"messages.{index}.content.{block_index}"
            if isinstance(block, RequestImageBlock):
                raise UnsupportedCapabilityError(
                    "Image content on the Anthropic Messages surface",
                    "This runtime has no media resolution bound to this "
                    "protocol; the request is refused rather than answered "
                    "from text alone.",
                )
            if isinstance(block, RequestToolResultBlock):
                _reject_unsupported_tool_result(block, path)


def _reject_unsupported_tool_result(
    block: RequestToolResultBlock,
    path: str,
) -> None:
    if isinstance(block.content, str):
        return
    for item_index, item in enumerate(block.content):
        item_path = f"{path}.content.{item_index}"
        if isinstance(item, RequestSearchResultBlock):
            raise UnsupportedCapabilityError(
                "Anthropic search_result blocks in tool_result",
                f"{item_path} cannot be represented on the local Qwen ABI.",
            )
        if isinstance(item, RequestToolReferenceBlock):
            raise UnsupportedCapabilityError(
                "Anthropic tool_reference blocks in tool_result",
                f"{item_path} cannot be represented on the local Qwen ABI.",
            )
        if isinstance(item, RequestDocumentBlock):
            _require_mappable_document(item, item_path)


def _require_mappable_document(block: RequestDocumentBlock, path: str) -> None:
    if block.citations is not None and block.citations.enabled:
        raise UnsupportedCapabilityError(
            "Anthropic document citations in tool_result",
            f"{path}.citations cannot be represented on the local Qwen ABI.",
        )
    source = block.source
    if source.type == "content":
        raise UnsupportedCapabilityError(
            "Anthropic nested document content in tool_result",
            f"{path}.source.type=content cannot be represented exactly.",
        )
    if source.type == "text":
        raise UnsupportedCapabilityError(
            "Anthropic plaintext document blocks in tool_result",
            f"{path}.source.type=text is not an exact local file ABI.",
        )
    if source.type == "url" and not (source.url or "").strip():
        raise _field_error(f"{path}.source.url must not be empty")
    if source.type == "file" and not (source.file_id or "").strip():
        raise _field_error(f"{path}.source.file_id must not be empty")
    if source.type == "base64":
        if not (source.data or "").strip():
            raise _field_error(f"{path}.source.data must not be empty")
        media_type = source.media_type or ""
        if media_type not in _FILE_MEDIA_TYPES:
            raise UnsupportedCapabilityError(
                "Anthropic document media_type in tool_result",
                f"{path}.source.media_type={media_type!r} is not an exact "
                "local file ABI.",
            )


def _validate_tool_conversation(messages: list[InputMessage]) -> None:
    pending: tuple[str, ...] | None = None
    pending_path = "messages"
    for index, message in enumerate(messages):
        path = f"messages.{index}"
        tool_use_ids = _tool_use_ids(message, path)
        result_ids = _tool_result_ids(message, path)
        if message.role is MessageRole.ASSISTANT:
            if result_ids:
                raise _field_error(
                    f"{path}.content tool_result is not valid on assistant"
                )
            if pending is not None:
                raise _field_error(
                    f"{pending_path} tool_use requires tool_result before {path}"
                )
            pending = tool_use_ids or None
            pending_path = path
            continue
        if tool_use_ids:
            raise _field_error(f"{path}.content tool_use is not valid on user")
        if result_ids:
            _require_tool_result_prefix(message, path)
            if pending is None:
                raise _field_error(
                    f"{path}.content tool_result has no preceding tool_use"
                )
            _require_matching_tool_ids(result_ids, pending, path)
            pending = None
            continue
        if pending is not None:
            raise _field_error(
                f"{pending_path} tool_use requires tool_result before {path}"
            )
    if pending is not None:
        raise _field_error(
            f"{pending_path} tool_use requires matching tool_result blocks"
        )


def _tool_use_ids(message: InputMessage, path: str) -> tuple[str, ...]:
    if isinstance(message.content, str):
        return ()
    ids: list[str] = []
    seen: set[str] = set()
    for block_index, block in enumerate(message.content):
        if not isinstance(block, RequestToolUseBlock):
            continue
        if block.id in seen:
            raise _field_error(
                f"{path}.content.{block_index}.id duplicates {block.id!r}"
            )
        seen.add(block.id)
        ids.append(block.id)
    return tuple(ids)


def _tool_result_ids(message: InputMessage, path: str) -> tuple[str, ...]:
    if isinstance(message.content, str):
        return ()
    ids: list[str] = []
    seen: set[str] = set()
    for block_index, block in enumerate(message.content):
        if not isinstance(block, RequestToolResultBlock):
            continue
        if block.tool_use_id in seen:
            raise _field_error(
                f"{path}.content.{block_index}.tool_use_id duplicates "
                f"{block.tool_use_id!r}"
            )
        seen.add(block.tool_use_id)
        ids.append(block.tool_use_id)
    return tuple(ids)


def _require_tool_result_prefix(message: InputMessage, path: str) -> None:
    if isinstance(message.content, str):
        return
    seen_other = False
    for block_index, block in enumerate(message.content):
        if isinstance(block, RequestToolResultBlock):
            if seen_other:
                raise _field_error(
                    f"{path}.content.{block_index} tool_result must precede "
                    "later text or other content; the request is refused "
                    "rather than reordered"
                )
            continue
        seen_other = True


def _require_matching_tool_ids(
    result_ids: tuple[str, ...],
    pending: tuple[str, ...],
    path: str,
) -> None:
    extra = [item for item in result_ids if item not in pending]
    missing = [item for item in pending if item not in result_ids]
    if extra or missing:
        raise _field_error(
            f"{path}.content tool_result ids {result_ids!r} do not match "
            f"the immediately preceding tool_use ids {pending!r}"
        )


def _field_error(message: str) -> AnthropicAPIError:
    return AnthropicAPIError(message, error_type="invalid_request_error")


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


def _build_messages(
    system: SystemPrompt | None,
    messages: list[InputMessage],
) -> tuple[tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...]]:
    mapped: list[Mapping[str, Any]] = []
    media: list[Mapping[str, Any]] = []
    system_text = _system_text(system)
    if system_text:
        mapped.append({"role": "system", "content": _text_content(system_text)})
    for message in messages:
        new_messages, new_media = _map_message(message, start_index=len(mapped))
        mapped.extend(new_messages)
        media.extend(new_media)
    return tuple(mapped), tuple(media)


def _system_text(system: SystemPrompt | None) -> str:
    if system is None:
        return ""
    if isinstance(system, str):
        return system
    return "\n".join(block.text for block in system)


def _map_message(
    message: InputMessage,
    *,
    start_index: int,
) -> tuple[tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...]]:
    if isinstance(message.content, str):
        return (
            (
                {
                    "role": message.role.value,
                    "content": _text_content(message.content),
                },
            ),
            (),
        )
    if message.role is MessageRole.USER:
        return _map_user_content(message.content, start_index=start_index)
    return _map_assistant_content(message.content), ()


def _map_user_content(
    blocks: Sequence[RequestContentBlock],
    *,
    start_index: int,
) -> tuple[tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...]]:
    mapped: list[Mapping[str, Any]] = []
    media: list[Mapping[str, Any]] = []
    text_parts: list[str] = []
    for block in blocks:
        if isinstance(block, RequestToolResultBlock):
            message_index = start_index + len(mapped)
            tool_message, tool_media = _map_tool_result(block, message_index)
            mapped.append(tool_message)
            media.extend(tool_media)
            continue
        if isinstance(block, RequestTextBlock):
            text_parts.append(block.text)
            continue
        if isinstance(block, RequestThinkingBlock):
            raise _field_error("thinking blocks are not valid on user turns")
        if isinstance(block, RequestToolUseBlock):
            raise _field_error("tool_use blocks are not valid on user turns")
        # Reached only when a block type is represented on the wire but has
        # no mapping here. Refuse rather than drop it: a skipped block is
        # exactly the silent lie the capability preflight exists to prevent.
        raise _field_error(f"{block.type} blocks have no mapping on user turns")
    if text_parts:
        mapped.append(
            {
                "role": "user",
                "content": _text_content("\n".join(text_parts)),
            }
        )
    elif not mapped:
        mapped.append({"role": "user", "content": _text_content("")})
    return tuple(mapped), tuple(media)


def _map_assistant_content(
    blocks: Sequence[RequestContentBlock],
) -> tuple[Mapping[str, Any], ...]:
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[Mapping[str, Any]] = []
    for block in blocks:
        if isinstance(block, RequestTextBlock):
            text_parts.append(block.text)
        elif isinstance(block, RequestThinkingBlock):
            reasoning_parts.append(block.thinking)
        elif isinstance(block, RequestToolUseBlock):
            tool_calls.append(
                {
                    "id": block.id,
                    "type": "function",
                    "function": {"name": block.name, "arguments": block.input},
                }
            )
        elif isinstance(block, RequestToolResultBlock):
            raise _field_error("tool_result blocks are not valid on assistant")
        else:
            raise _field_error(
                f"{block.type} blocks have no mapping on assistant turns"
            )
    primary: dict[str, Any] = {"role": "assistant"}
    if text_parts:
        primary["content"] = _text_content("\n".join(text_parts))
    if reasoning_parts:
        primary["reasoning"] = "\n".join(reasoning_parts)
    if tool_calls:
        primary["tool_calls"] = tool_calls
    if len(primary) == 1:
        primary["content"] = _text_content("")
    else:
        primary.setdefault("content", _text_content(""))
    return (primary,)


def _map_tool_result(
    block: RequestToolResultBlock,
    message_index: int,
) -> tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...]]:
    text_parts, media = _map_tool_result_parts(block, message_index)
    output = "\n".join(part["text"] for part in text_parts)
    if text_parts:
        content: tuple[Mapping[str, str], ...] = tuple(text_parts)
    elif output:
        content = _text_content(output)
    else:
        content = ()
    message: dict[str, Any] = {
        "type": "function_call_output",
        "role": "tool",
        "call_id": block.tool_use_id,
        "output": output,
        "content": content,
        "is_error": bool(block.is_error),
    }
    return message, media


def _map_tool_result_parts(
    block: RequestToolResultBlock,
    message_index: int,
) -> tuple[tuple[dict[str, str], ...], tuple[Mapping[str, Any], ...]]:
    if isinstance(block.content, str):
        return (_text_part(block.content),), ()
    has_media = any(not isinstance(item, RequestTextBlock) for item in block.content)
    if not has_media:
        joined = "\n".join(
            item.text for item in block.content if isinstance(item, RequestTextBlock)
        )
        if not joined:
            return (), ()
        return (_text_part(joined),), ()
    texts: list[dict[str, str]] = []
    media: list[Mapping[str, Any]] = []
    for content_index, item in enumerate(block.content):
        if isinstance(item, RequestTextBlock):
            texts.append(_text_part(item.text))
            continue
        if isinstance(item, RequestImageBlock):
            media.append(
                _image_media(
                    item, message_index=message_index, content_index=content_index
                )
            )
            continue
        if isinstance(item, RequestDocumentBlock):
            media.append(
                _document_media(
                    item,
                    message_index=message_index,
                    content_index=content_index,
                )
            )
            continue
        raise UnsupportedCapabilityError(
            f"Anthropic {item.type} blocks in tool_result",
            "Nested tool_result content must be text, image, or a mappable document.",
        )
    return tuple(texts), tuple(media)


def _text_part(text: str) -> dict[str, str]:
    return {"type": "input_text", "text": text}


def _image_media(
    block: RequestImageBlock,
    *,
    message_index: int,
    content_index: int,
) -> Mapping[str, Any]:
    source = block.source
    item: dict[str, Any] = {
        "type": "input_image",
        "_role": "tool",
        "_message_index": message_index,
        "_content_index": content_index,
    }
    if source.type == "url":
        if not (source.url or "").strip():
            raise _field_error("tool_result image url must not be empty")
        item["image_url"] = source.url
        return item
    if source.type == "base64":
        if not (source.data or "").strip():
            raise _field_error("tool_result image data must not be empty")
        media_type = source.media_type or "image/png"
        if media_type not in _IMAGE_MEDIA_TYPES:
            raise UnsupportedCapabilityError(
                "Anthropic tool_result image media_type",
                f"{media_type!r} is outside the local image ABI.",
            )
        item["image_base64"] = f"data:{media_type};base64,{source.data}"
        return item
    if source.type == "file":
        if not (source.file_id or "").strip():
            raise _field_error("tool_result image file_id must not be empty")
        item["file_id"] = source.file_id
        return item
    raise UnsupportedCapabilityError(
        "Anthropic tool_result image source",
        f"source.type={source.type!r} cannot be represented exactly.",
    )


def _document_media(
    block: RequestDocumentBlock,
    *,
    message_index: int,
    content_index: int,
) -> Mapping[str, Any]:
    source = block.source
    item: dict[str, Any] = {
        "type": "input_file",
        "_role": "tool",
        "_message_index": message_index,
        "_content_index": content_index,
    }
    if block.title:
        item["filename"] = block.title
    if source.type == "url":
        item["file_url"] = source.url
        return item
    if source.type == "file":
        item["file_id"] = source.file_id
        return item
    if source.type == "base64":
        item["file_data"] = source.data
        return item
    raise UnsupportedCapabilityError(
        "Anthropic document source in tool_result",
        f"source.type={source.type!r} cannot be represented exactly.",
    )


# ---------------------------------------------------------------------------
# Tools and sampling
# ---------------------------------------------------------------------------


def _map_tool(tool: AnthropicTool) -> Mapping[str, Any]:
    if tool.input_schema is None:
        raise _field_error(f"tools entry {tool.name!r} requires input_schema")
    schema = tool.input_schema.model_dump(mode="json", exclude_none=True)
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description or "",
        "parameters": schema,
    }


def _map_tool_choice(request: MessagesRequest) -> str | Mapping[str, Any] | None:
    choice = request.tool_choice
    if choice is None:
        return None
    if isinstance(choice, ToolChoiceTool):
        declared = {tool.name for tool in request.tools or ()}
        if choice.name not in declared:
            raise AnthropicAPIError(
                f"tool_choice names {choice.name!r}, which is not in tools",
                error_type="invalid_request_error",
            )
        return {"type": "function", "name": choice.name}
    if isinstance(choice, ToolChoiceAuto):
        return "auto"
    if isinstance(choice, ToolChoiceAny):
        return "required"
    if isinstance(choice, ToolChoiceNone):
        return "none"
    raise TypeError(f"unsupported Anthropic tool choice: {type(choice).__name__}")


def _map_sampling(request: MessagesRequest) -> Mapping[str, Any]:
    # Anthropic calls this max_tokens; the shared runtime ABI calls the same
    # output budget max_output_tokens.
    sampling: dict[str, Any] = {"max_output_tokens": request.max_tokens}
    if request.temperature is not None:
        sampling["temperature"] = request.temperature
    if request.top_p is not None:
        sampling["top_p"] = request.top_p
    if request.top_k is not None:
        sampling["top_k"] = request.top_k
    if request.stop_sequences:
        sampling["stop"] = tuple(request.stop_sequences)
    if request.tool_choice is not None:
        sampling["parallel_tool_calls"] = not getattr(
            request.tool_choice, "disable_parallel_tool_use", False
        )
    return sampling


def _text_content(text: str) -> tuple[Mapping[str, str], ...]:
    return ({"type": "input_text", "text": text},)


def _map_reasoning(request: MessagesRequest) -> Mapping[str, Any]:
    thinking = request.thinking
    if thinking is None:
        return {}
    if isinstance(thinking, ThinkingConfigEnabled):
        return {"enabled": True, "budget_tokens": thinking.budget_tokens}
    return {"enabled": False}


def _map_metadata(request: MessagesRequest) -> Mapping[str, Any]:
    metadata: dict[str, Any] = {}
    if request.metadata is not None and request.metadata.user_id:
        metadata["user_id"] = request.metadata.user_id
    if request.service_tier is not None:
        metadata["service_tier"] = request.service_tier.value
    return metadata


__all__ = ["build_turn"]
