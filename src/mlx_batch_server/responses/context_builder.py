"""
Context Builder - Build conversation context from response chains.

This module handles building conversation context from previous_response_id chains,
supporting mixed response types including STT transcriptions and LLM responses.

Key features:
- STT transcriptions become system context with metadata
- LLM responses become conversation history
- Chain walking with depth and token limits
- Handles mixed chains: STT -> LLM -> LLM -> ...

Created by M&K (c)2026 The LibraxisAI Team
Co-Authored-By: Maciej (void@div0.space) & Klaudiusz (the1st@whoai.am)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from ..utils.logger import logger
from .store import StoredResponse
from .store import get_response as store_get_response

# Configuration
MAX_CHAIN_DEPTH = int(os.getenv("CONTEXT_CHAIN_MAX_DEPTH", "20"))
MAX_CONTEXT_TOKENS = int(os.getenv("CONTEXT_MAX_TOKENS", "100000"))
# Rough estimate: 4 chars per token
CHARS_PER_TOKEN = 4


@dataclass
class ContextChainEntry:
    """Single entry in a context chain."""

    response_id: str
    response_type: str  # "transcription" or "response"
    messages: list[dict[str, Any]]
    estimated_tokens: int


@dataclass
class BuiltContext:
    """Result of building context from a response chain."""

    messages: list[dict[str, Any]]
    chain: list[str]  # Response IDs in chain order
    total_tokens: int
    truncated: bool
    chain_depth: int


def _estimate_tokens(text: str) -> int:
    """Rough token estimate based on character count."""
    return (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


def _extract_text_from_output(output: list[dict[str, Any]]) -> str:
    """Extract all text content from a response's output array."""
    texts = []
    for item in output:
        if not isinstance(item, dict):
            continue

        item_type = item.get("type", "")

        # Transcription output
        if item_type == "transcription":
            texts.append(item.get("text", ""))

        # Message output (from LLM)
        elif item_type == "message":
            content = item.get("content", [])
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") in ("text", "output_text"):
                            texts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        texts.append(part)

        # Direct text output
        elif item_type in ("text", "output_text"):
            texts.append(item.get("text", ""))

    return "\n".join(filter(None, texts))


def _build_transcription_context(stored: StoredResponse) -> list[dict[str, Any]]:
    """
    Build context messages from a transcription response.

    Transcriptions become a system message with metadata to inform the LLM
    that this is audio content being analyzed.
    """
    response = stored.response
    output = response.get("output", [])

    # Extract transcription details
    text = ""
    language = "unknown"
    duration = None

    for item in output:
        if isinstance(item, dict) and item.get("type") == "transcription":
            text = item.get("text", "")
            language = item.get("language", "unknown")
            duration = item.get("duration")
            break

    # Also check audio_metadata
    audio_meta = response.get("audio_metadata", {})
    if duration is None and audio_meta:
        duration = audio_meta.get("duration_seconds")

    # Build informative system message
    metadata_parts = []
    if duration is not None:
        metadata_parts.append(f"Duration: {duration:.1f}s")
    if language != "unknown":
        metadata_parts.append(f"Language: {language}")

    metadata_str = ", ".join(metadata_parts) if metadata_parts else ""

    system_content = f"Audio transcription from previous interaction:\n\n{text}"
    if metadata_str:
        system_content += f"\n\n[{metadata_str}]"

    return [
        {
            "role": "system",
            "content": [{"type": "input_text", "text": system_content}],
        }
    ]


def _filter_media_from_content(
    content: list[dict[str, Any]] | str,
) -> list[dict[str, Any]]:
    """
    Filter out media (images, audio, video) from message content.

    When building context from previous responses, we only want text content.
    Including large base64 media from previous requests causes:
    - Massive context sizes (43k+ tokens)
    - Corrupted base64 data when re-processed
    - Backend "illegal base64 data" errors
    """
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]

    filtered = []
    media_types = {"input_image", "input_audio", "input_video", "image_url"}

    for part in content:
        if not isinstance(part, dict):
            continue

        part_type = part.get("type", "")

        # Skip media parts - they bloat context and cause corruption
        if part_type in media_types:
            continue

        # Keep text parts
        if part_type in ("input_text", "text"):
            filtered.append(part)

    return filtered


def _build_response_context(stored: StoredResponse) -> list[dict[str, Any]]:
    """
    Build context messages from a regular LLM response.

    Extracts the assistant's response and any prior context from the stored request.
    IMPORTANT: Filters out media (images/audio/video) from previous requests to prevent
    context bloat and base64 corruption issues.
    """
    messages = []

    # Include prior input from the request (user messages, system, etc.)
    # BUT filter out media content to prevent context bloat and corruption
    request = stored.request
    input_items = request.get("input", [])

    # Handle string input (simple case)
    if isinstance(input_items, str):
        if input_items.strip():
            messages.append({"role": "user", "content": input_items})
    elif isinstance(input_items, list):
        for item in input_items:
            if not isinstance(item, dict):
                continue

            role = item.get("role", "user")
            content = item.get("content", [])

            # Filter media from content
            filtered_content = _filter_media_from_content(content)

            # Only add message if there's text content remaining
            if filtered_content:
                messages.append({"role": role, "content": filtered_content})

    # Add assistant response
    response = stored.response
    output = response.get("output", [])
    assistant_text = _extract_text_from_output(output)

    if assistant_text:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "output_text", "text": assistant_text}],
            }
        )

    return messages


async def build_context_from_response_chain(
    *,
    response_id: str,
    api_key: str | None,
    max_depth: int | None = None,
    max_tokens: int | None = None,
) -> BuiltContext | None:
    """
    Build full context by walking the previous_response_id chain.

    This walks backwards through the chain, collecting context from each response.
    STT transcriptions become system context, LLM responses become conversation history.

    Args:
        response_id: The response ID to start from
        api_key: API key for access control
        max_depth: Maximum chain depth (default: MAX_CHAIN_DEPTH)
        max_tokens: Maximum total context tokens (default: MAX_CONTEXT_TOKENS)

    Returns:
        BuiltContext with messages, chain info, and token counts, or None if not found
    """
    max_depth = max_depth if max_depth is not None else MAX_CHAIN_DEPTH
    max_tokens = max_tokens if max_tokens is not None else MAX_CONTEXT_TOKENS

    # Walk the chain backwards, collecting entries
    chain_entries: list[ContextChainEntry] = []
    current_id: str | None = response_id
    visited: set[str] = set()
    total_tokens = 0
    truncated = False

    while current_id and len(chain_entries) < max_depth:
        # Prevent cycles
        if current_id in visited:
            logger.warning(f"Cycle detected in response chain at {current_id}")
            break
        visited.add(current_id)

        # Fetch the response
        stored = await store_get_response(response_id=current_id, api_key=api_key)
        if stored is None:
            # Chain is broken - stop here
            if chain_entries:
                logger.warning(f"Chain broken at {current_id} - response not found")
            break

        # Determine response type and build messages
        response = stored.response
        response_type = response.get("type", "response")

        if response_type == "transcription":
            messages = _build_transcription_context(stored)
        else:
            messages = _build_response_context(stored)

        # Estimate tokens
        content_text = ""
        for msg in messages:
            content = msg.get("content", [])
            if isinstance(content, str):
                content_text += content
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        content_text += part.get("text", "")
                    elif isinstance(part, str):
                        content_text += part

        entry_tokens = _estimate_tokens(content_text)

        # Check token limit
        if total_tokens + entry_tokens > max_tokens:
            truncated = True
            logger.info(
                f"Context truncated at {current_id}: "
                f"{total_tokens + entry_tokens} > {max_tokens}"
            )
            break

        chain_entries.append(
            ContextChainEntry(
                response_id=current_id,
                response_type=response_type,
                messages=messages,
                estimated_tokens=entry_tokens,
            )
        )
        total_tokens += entry_tokens

        # Move to previous response in chain
        current_id = stored.request.get("previous_response_id")

    if not chain_entries:
        return None

    # Reverse to get chronological order (oldest first)
    chain_entries.reverse()

    # Flatten messages
    all_messages: list[dict[str, Any]] = []
    chain_ids: list[str] = []

    for entry in chain_entries:
        all_messages.extend(entry.messages)
        chain_ids.append(entry.response_id)

    return BuiltContext(
        messages=all_messages,
        chain=chain_ids,
        total_tokens=total_tokens,
        truncated=truncated,
        chain_depth=len(chain_entries),
    )


async def build_context_from_previous_response(
    *,
    previous_response_id: str,
    api_key: str | None,
    current_input: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Build context from a single previous response (simple case).

    This is a convenience function for the common case where you just need
    to include the previous response as context for a new request.

    For STT transcriptions:
        - Adds transcript as system context

    For LLM responses:
        - Includes the full conversation history from that response

    Args:
        previous_response_id: The response ID to build context from
        api_key: API key for access control
        current_input: Current request's input items to append

    Returns:
        List of messages combining previous context and current input
    """
    context = await build_context_from_response_chain(
        response_id=previous_response_id,
        api_key=api_key,
        max_depth=1,  # Just the immediate previous response
    )

    if context is None:
        # Previous response not found - just return current input
        return current_input or []

    messages = list(context.messages)

    # Append current input
    if current_input:
        messages.extend(current_input)

    return messages


async def get_response_type(
    response_id: str,
    api_key: str | None,
) -> str | None:
    """
    Get the type of a stored response.

    Args:
        response_id: The response ID to check
        api_key: API key for access control

    Returns:
        Response type ("transcription", "response", etc.) or None if not found
    """
    stored = await store_get_response(response_id=response_id, api_key=api_key)
    if stored is None:
        return None

    return stored.response.get("type", "response")
