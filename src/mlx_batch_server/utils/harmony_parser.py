"""
OpenAI Harmony format parser and builder.

Provides utilities for working with the Harmony response format
used by GPT-OSS models. Uses openai-harmony package for parsing.

Includes:
- Tool call extraction from Harmony format
- Reasoning channel parsing (analysis/final sections)
- Output entry building for Responses API
- Harmony model detection

Created by M&K (c)2025 The LibraxisAI Team
"""

from __future__ import annotations

import re
import uuid
from typing import Any

# Import openai-harmony components
try:
    from openai_harmony import (
        Conversation,
        HarmonyEncodingName,
        Message,
        Role,
        TextContent,
        load_harmony_encoding,
    )

    HARMONY_AVAILABLE = True
    _HARMONY_ENCODING = None

    def _get_encoding():
        """Lazy load Harmony encoding."""
        global _HARMONY_ENCODING
        if _HARMONY_ENCODING is None:
            _HARMONY_ENCODING = load_harmony_encoding(
                HarmonyEncodingName.HARMONY_GPT_OSS.value
            )
        return _HARMONY_ENCODING

except ImportError:
    Conversation = Message = Role = TextContent = None  # type: ignore[assignment]
    load_harmony_encoding = None  # type: ignore[assignment]
    HarmonyEncodingName = None  # type: ignore[assignment]
    HARMONY_AVAILABLE = False

    def _get_encoding():
        return None


# Keywords that indicate a Harmony-format model
HARMONY_KEYWORDS = ["gpt-oss", "harmony"]


def is_harmony_model(model_name: str) -> bool:
    """
    Check if model uses Harmony response format.

    Args:
        model_name: Model identifier

    Returns:
        True if model uses Harmony format
    """
    model_lower = model_name.lower()
    return any(kw in model_lower for kw in HARMONY_KEYWORDS)


def _cleanup_harmony_content(content: str) -> str:
    """
    Clean up common Harmony format issues before parsing.

    Fixes:
    - Double special tokens (<|call|><|call|>)
    - Missing <|start|> before assistant
    - Truncated sequences
    """
    # Fix double tokens
    content = re.sub(r"<\|call\|>\s*<\|call\|>", "<|call|>", content)
    content = re.sub(r"<\|end\|>\s*<\|end\|>", "<|end|>", content)

    # Fix missing <|start|> before assistant after <|call|>
    content = re.sub(
        r"<\|call\|>\s*assistant<\|channel\|>",
        "<|call|><|start|>assistant<|channel|>",
        content,
    )

    return content


def parse_harmony_output(content: str) -> dict[str, Any]:
    """
    Parse Harmony format output using openai-harmony.

    Extracts:
    - Tool calls (channel='commentary', recipient='functions.X')
    - Reasoning (channel='analysis')
    - Final response (channel='final')

    Args:
        content: Raw model output in Harmony format

    Returns:
        Dict with 'tool_calls', 'reasoning', 'final_text' keys
    """
    result = {
        "tool_calls": [],
        "reasoning": None,
        "final_text": "",
    }

    if not HARMONY_AVAILABLE:
        # Fallback: return content as-is
        result["final_text"] = content
        return result

    encoding = _get_encoding()
    if encoding is None:
        result["final_text"] = content
        return result

    # Clean up common format issues
    content = _cleanup_harmony_content(content)

    # Ensure content has start token for parsing
    if not content.strip().startswith("<|start|>"):
        content = "<|start|>assistant" + content

    try:
        # Encode to tokens
        tokens = encoding.encode(content, allowed_special="all")

        # Parse messages
        messages = encoding.parse_messages_from_completion_tokens(tokens)

        reasoning_parts = []
        final_parts = []

        for msg in messages:
            channel = getattr(msg, "channel", None)
            recipient = getattr(msg, "recipient", None)
            msg_content = getattr(msg, "content", [])

            # Extract text from content
            text = ""
            for c in msg_content:
                if hasattr(c, "text"):
                    text = c.text
                    break

            # Tool call: channel='commentary', recipient='functions.X'
            if (
                channel == "commentary"
                and recipient
                and recipient.startswith("functions.")
            ):
                func_name = recipient.replace("functions.", "")
                result["tool_calls"].append(
                    {
                        "id": f"call_{uuid.uuid4().hex[:24]}",
                        "name": func_name,
                        "arguments": text,
                    }
                )

            # Extract reasoning from analysis channel
            elif channel == "analysis":
                reasoning_parts.append(text)

            # Final response: channel='final'
            elif channel == "final" or not channel:
                final_parts.append(text)

        result["reasoning"] = "\n".join(reasoning_parts) if reasoning_parts else None
        result["final_text"] = "\n".join(final_parts) if final_parts else ""

    except Exception:
        # openai-harmony parsing failed - use regex fallback
        result = _parse_harmony_regex_fallback(content)

    return result


# Regex patterns for fallback parsing
_TOOL_CALL_RE = re.compile(
    r"<\|channel\|>commentary\s+to=functions\.(\w+)\s*"
    r"(?:<\|constrain\|>json)?\s*"
    r"<\|message\|>(.*?)<\|call\|>",
    re.DOTALL,
)
_ANALYSIS_RE = re.compile(
    r"<\|channel\|>analysis<\|message\|>(.*?)(?:<\|end\|>|<\|start\|>|<\|channel\|>|$)",
    re.DOTALL,
)
_FINAL_RE = re.compile(
    r"<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|<\|start\|>|$)",
    re.DOTALL,
)


def _parse_harmony_regex_fallback(content: str) -> dict[str, Any]:
    """
    Fallback regex parser for malformed Harmony output.

    Used when openai-harmony parser fails due to truncated/malformed content.
    """
    result = {
        "tool_calls": [],
        "reasoning": None,
        "final_text": "",
    }

    # Extract tool calls
    for match in _TOOL_CALL_RE.finditer(content):
        func_name = match.group(1)
        args_str = match.group(2).strip()
        result["tool_calls"].append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "name": func_name,
                "arguments": args_str,
            }
        )

    # Extract reasoning (first analysis block)
    analysis_match = _ANALYSIS_RE.search(content)
    if analysis_match:
        result["reasoning"] = analysis_match.group(1).strip()

    # Extract final text
    final_match = _FINAL_RE.search(content)
    if final_match:
        result["final_text"] = final_match.group(1).strip()

    return result


def parse_harmony_tool_calls(content: str) -> list[dict[str, Any]]:
    """
    Extract tool calls from Harmony format output.

    Args:
        content: Raw model output containing Harmony tool calls

    Returns:
        List of tool call dicts with 'id', 'name', 'arguments' keys
    """
    parsed = parse_harmony_output(content)
    return parsed["tool_calls"]


def extract_harmony_content(
    content: str,
) -> tuple[str, str | None, list[dict[str, Any]]]:
    """
    Parse full Harmony output into components.

    Args:
        content: Raw model output

    Returns:
        Tuple of (final_text, reasoning_text, tool_calls)
    """
    parsed = parse_harmony_output(content)
    return parsed["final_text"], parsed["reasoning"], parsed["tool_calls"]


# Legacy regex pattern for fallback
CHANNEL_PATTERN = re.compile(r"^\s*(analysis|final)\s*:?\s*$", re.IGNORECASE)


def parse_reasoning_channels(reasoning: str | None) -> tuple[str | None, str | None]:
    """
    Split Harmony-style plain text into analysis/final sections.

    Harmony models emit reasoning in format:
        analysis:
        [analysis/thinking text]
        final:
        [final response text]

    Args:
        reasoning: Raw reasoning text from model

    Returns:
        Tuple of (analysis_text, final_text), either may be None
    """
    if not reasoning:
        return None, None

    current_channel: str | None = None
    buffers: dict[str, list[str]] = {"analysis": [], "final": []}

    for line in reasoning.splitlines():
        match = CHANNEL_PATTERN.match(line)
        if match:
            current_channel = match.group(1).lower()
            # Include any text after the marker on same line
            remainder = line[match.end() :].strip()
            if remainder:
                buffers[current_channel].append(remainder)
            continue

        if current_channel:
            buffers[current_channel].append(line)

    analysis_text = "\n".join(buffers["analysis"]).strip() or None
    final_text = "\n".join(buffers["final"]).strip() or None

    return analysis_text, final_text


def build_harmony_output_entries(
    *,
    final_text: str | None,
    reasoning_text: str | None,
) -> list[dict[str, Any]]:
    """
    Create OpenAI Responses output[] entries with reasoning + message.

    Args:
        final_text: Main response text
        reasoning_text: Reasoning/analysis text (optional)

    Returns:
        List of output item dictionaries
    """
    outputs: list[dict[str, Any]] = []

    if reasoning_text:
        outputs.append(
            {
                "id": f"rs_{uuid.uuid4().hex[:24]}",
                "type": "reasoning",
                "status": "completed",
                "summary": [],
                "content": [{"type": "reasoning_text", "text": reasoning_text}],
            }
        )

    if final_text is not None:
        outputs.append(
            {
                "id": f"msg_{uuid.uuid4().hex[:24]}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": final_text}],
            }
        )

    return outputs


def apply_harmony_parsing(result: dict[str, Any], model: str) -> dict[str, Any]:
    """
    Apply Harmony format parsing to chat completion result.

    If model uses Harmony format, extracts reasoning channels,
    tool calls, and restructures the response accordingly.

    Args:
        result: Raw chat completion result
        model: Model name/identifier

    Returns:
        Modified result with parsed Harmony content and tool_calls
    """
    if not is_harmony_model(model):
        return result

    # Extract content from result
    choices = result.get("choices", [])
    if not choices:
        return result

    choice = choices[0]
    message = choice.get("message", {})
    content = message.get("content", "")

    if not content:
        return result

    # Use openai-harmony parser
    parsed = parse_harmony_output(content)

    # Update message with parsed content
    if parsed["final_text"]:
        message["content"] = parsed["final_text"]

    if parsed["reasoning"]:
        message["reasoning"] = parsed["reasoning"]

    # Add tool calls if present
    if parsed["tool_calls"]:
        message["tool_calls"] = [
            {
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": tc["arguments"],
                },
            }
            for tc in parsed["tool_calls"]
        ]

    return result


def create_harmony_conversation(
    messages: list[dict[str, str]],
) -> Conversation | None:
    """
    Create an openai_harmony Conversation from chat messages.

    Requires openai-harmony package to be installed.

    Args:
        messages: List of {"role": str, "content": str} dicts

    Returns:
        Conversation object or None if harmony not available
    """
    if not HARMONY_AVAILABLE:
        return None

    harmony_messages: list[Message] = []

    for msg in messages:
        role_str = msg.get("role", "user").lower()

        try:
            role = Role(role_str)
        except ValueError:
            role = Role.USER

        content = msg.get("content", "")
        harmony_messages.append(
            Message.from_role_and_content(role, TextContent(text=content))
        )

    return Conversation.from_messages(harmony_messages)


def render_harmony_prompt(conversation: Conversation) -> str:
    """
    Render Harmony conversation to prompt string.

    Args:
        conversation: Harmony Conversation object

    Returns:
        Formatted prompt string
    """
    if not HARMONY_AVAILABLE or conversation is None:
        return ""

    return conversation.render()


__all__ = [
    "HARMONY_AVAILABLE",
    "apply_harmony_parsing",
    "build_harmony_output_entries",
    "create_harmony_conversation",
    "extract_harmony_content",
    "is_harmony_model",
    "parse_harmony_output",
    "parse_harmony_tool_calls",
    "parse_reasoning_channels",
    "render_harmony_prompt",
]
