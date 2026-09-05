"""Map a validated Anthropic Messages request onto a protocol-neutral turn.

Two rules govern this module. Anthropic semantics are *mapped*, never
flattened away: a tool result stays a tool result, reasoning stays reasoning.
And anything this runtime cannot actually honour is refused with an explicit
Anthropic error, because a silently dropped field is a lie about what was
inferred.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .anthropic_schema import (
    AnthropicTool,
    InputMessage,
    MessageRole,
    MessagesRequest,
    RequestImageBlock,
    RequestTextBlock,
    RequestThinkingBlock,
    RequestToolResultBlock,
    RequestToolUseBlock,
    SystemPrompt,
    ThinkingConfigEnabled,
    ToolChoiceTool,
)
from .errors import AnthropicAPIError, UnsupportedCapabilityError
from .turn_source import AnthropicTurn


def build_turn(request: MessagesRequest) -> AnthropicTurn:
    """Translate one Anthropic request into a runtime-neutral turn."""

    _reject_unsupported(request)
    messages = _build_messages(request.system, request.messages)
    tools = tuple(_map_tool(tool) for tool in request.tools or ())
    tool_choice = _map_tool_choice(request)
    return AnthropicTurn(
        model_alias=request.model,
        messages=messages,
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
    for message in request.messages:
        if isinstance(message.content, str):
            continue
        for block in message.content:
            if isinstance(block, RequestImageBlock):
                raise UnsupportedCapabilityError(
                    "Image content on the Anthropic Messages surface",
                    "This runtime has no media resolution bound to this "
                    "protocol; the request is refused rather than answered "
                    "from text alone.",
                )


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


def _build_messages(
    system: SystemPrompt | None,
    messages: list[InputMessage],
) -> tuple[Mapping[str, Any], ...]:
    mapped: list[Mapping[str, Any]] = []
    system_text = _system_text(system)
    if system_text:
        mapped.append({"role": "system", "content": system_text})
    for message in messages:
        mapped.extend(_map_message(message))
    return tuple(mapped)


def _system_text(system: SystemPrompt | None) -> str:
    if system is None:
        return ""
    if isinstance(system, str):
        return system
    return "\n".join(block.text for block in system)


def _map_message(message: InputMessage) -> tuple[Mapping[str, Any], ...]:
    if isinstance(message.content, str):
        return ({"role": message.role.value, "content": message.content},)

    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[Mapping[str, Any]] = []
    tool_results: list[Mapping[str, Any]] = []

    for block in message.content:
        if isinstance(block, RequestTextBlock):
            text_parts.append(block.text)
        elif isinstance(block, RequestThinkingBlock):
            # Anthropic requires prior reasoning to be echoed back on
            # tool-use continuations. Keep it on its own channel so it is
            # never concatenated into visible assistant text.
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
            tool_results.append(_map_tool_result(block))

    mapped: list[Mapping[str, Any]] = []
    # Anthropic packs tool results into a user turn; the chat-template shape
    # expects them as their own tool-role messages, ahead of any new text.
    mapped.extend(tool_results)

    primary: dict[str, Any] = {"role": message.role.value}
    if text_parts:
        primary["content"] = "\n".join(text_parts)
    if reasoning_parts:
        primary["reasoning"] = "\n".join(reasoning_parts)
    if tool_calls:
        primary["tool_calls"] = tool_calls
    if len(primary) > 1:
        primary.setdefault("content", "")
        mapped.append(primary)
    elif not mapped:
        # A content list that produced nothing at all still has to occupy the
        # turn, otherwise the conversation silently loses a role boundary.
        mapped.append({"role": message.role.value, "content": ""})
    return tuple(mapped)


def _map_tool_result(block: RequestToolResultBlock) -> Mapping[str, Any]:
    if isinstance(block.content, str):
        content = block.content
    else:
        content = "\n".join(
            item.text for item in block.content if isinstance(item, RequestTextBlock)
        )
    result: dict[str, Any] = {
        "role": "tool",
        "tool_call_id": block.tool_use_id,
        "content": content,
    }
    if block.is_error:
        # Preserved rather than collapsed into the text: an errored tool
        # result is a different fact than a tool result that says "error".
        result["is_error"] = True
    return result


# ---------------------------------------------------------------------------
# Tools and sampling
# ---------------------------------------------------------------------------


def _map_tool(tool: AnthropicTool) -> Mapping[str, Any]:
    schema = tool.input_schema.model_dump(mode="json", exclude_none=True)
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": schema,
        },
    }


def _map_tool_choice(request: MessagesRequest) -> Mapping[str, Any] | None:
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
    return choice.model_dump(mode="json")


def _map_sampling(request: MessagesRequest) -> Mapping[str, Any]:
    # max_tokens is carried through exactly as the client asked. No ceiling is
    # imposed here; budget policy belongs to the runtime owner, not to a
    # protocol adapter.
    sampling: dict[str, Any] = {"max_tokens": request.max_tokens}
    if request.temperature is not None:
        sampling["temperature"] = request.temperature
    if request.top_p is not None:
        sampling["top_p"] = request.top_p
    if request.top_k is not None:
        sampling["top_k"] = request.top_k
    if request.stop_sequences:
        sampling["stop_sequences"] = list(request.stop_sequences)
    return sampling


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
