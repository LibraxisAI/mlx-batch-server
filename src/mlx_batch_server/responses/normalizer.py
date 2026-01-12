"""
Request normalization for Responses API.

Converts various input formats to canonical structure for processing.

"""

from __future__ import annotations

import json
from collections.abc import Iterable
from copy import deepcopy
from typing import Any

# Content type sets for classification
_TEXT_KEYS = {"text", "input_text", "output_text"}
_IMAGE_KEYS = {"input_image", "image_url", "image_base64"}
_AUDIO_KEYS = {"input_audio", "audio_url"}
_VIDEO_KEYS = {"input_video", "video_url"}

DEFAULT_MODALITIES = ["text"]


def _stringify(value: Any) -> str:
    """Convert any value to string."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def _normalise_part_dict(part: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize a single content part dictionary."""
    part_type = part.get("type")

    # Text content
    if part_type in _TEXT_KEYS or (part_type is None and "text" in part):
        text_value = part.get("text")
        if text_value is None:
            return None
        return {"type": "input_text", "text": _stringify(text_value)}

    # Image content
    if part_type in _IMAGE_KEYS or "image_url" in part or "image_base64" in part:
        source = (
            part.get("image_base64")
            or part.get("image_url")
            or part.get("url")
            or part.get("file_id")
        )
        if isinstance(source, dict):
            source = source.get("url") or source.get("file_id")
        if not source:
            return None

        normalised: dict[str, Any] = {"type": "input_image"}
        if isinstance(source, str) and source.startswith("data:"):
            normalised["image_base64"] = source
        else:
            normalised["image_url"] = source

        if "detail" in part:
            normalised["detail"] = part["detail"]
        return normalised

    # Audio content
    if part_type in _AUDIO_KEYS or "audio_url" in part:
        source = part.get("audio_url") or part.get("file_id")
        if not source:
            return None
        return {"type": "input_audio", "audio_url": source}

    # Video content
    if part_type in _VIDEO_KEYS or "video_url" in part:
        source = part.get("video_url") or part.get("file_id")
        if not source:
            return None
        return {"type": "input_video", "video_url": source}

    # Unknown - treat as text
    if part:
        return {"type": "input_text", "text": _stringify(part)}
    return None


def _normalise_content_parts(content: Any) -> list[dict[str, Any]]:
    """Normalize content to list of canonical parts."""
    parts: list[dict[str, Any]] = []

    if content is None:
        return parts

    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]

    if isinstance(content, dict):
        normalized = _normalise_part_dict(content)
        return [normalized] if normalized else []

    if isinstance(content, Iterable):
        for raw in content:
            if isinstance(raw, str):
                parts.append({"type": "input_text", "text": raw})
            elif isinstance(raw, dict):
                normalized = _normalise_part_dict(raw)
                if normalized:
                    parts.append(normalized)
            elif raw is not None:
                parts.append({"type": "input_text", "text": _stringify(raw)})
        return parts

    return [{"type": "input_text", "text": _stringify(content)}]


def _build_turn(role: str, content: Any) -> dict[str, Any]:
    """Build a turn dictionary from role and content."""
    normalised_parts = _normalise_content_parts(content)
    return {"role": role, "content": normalised_parts}


def _normalise_input(raw_input: Any) -> list[dict[str, Any]]:
    """Normalize input to list of turns."""
    turns: list[dict[str, Any]] = []

    if isinstance(raw_input, str):
        turns.append(_build_turn("user", raw_input))
    elif isinstance(raw_input, dict):
        role = raw_input.get("role", "user")
        turns.append(_build_turn(role, raw_input.get("content")))
    elif isinstance(raw_input, Iterable):
        for entry in raw_input:
            if isinstance(entry, dict):
                role = entry.get("role", "user")
                turns.append(_build_turn(role, entry.get("content")))
            elif isinstance(entry, str):
                turns.append(_build_turn("user", entry))
            elif entry is not None:
                turns.append(_build_turn("user", _stringify(entry)))
    elif raw_input is not None:
        turns.append(_build_turn("user", _stringify(raw_input)))

    if not turns:
        turns.append(_build_turn("user", ""))
    return turns


def _normalise_modalities(modalities: Any, default: list[str]) -> list[str]:
    """Normalize modalities list."""
    if isinstance(modalities, list) and modalities:
        return [str(mod).lower() for mod in modalities if mod]
    if isinstance(modalities, str) and modalities.strip():
        return [modalities.strip().lower()]
    return default


def normalise_responses_payload(body: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize a Responses API request body to canonical format.

    Args:
        body: Raw request body

    Returns:
        Normalized body with canonical structure
    """
    copy = deepcopy(body)

    # Normalize input turns
    copy["input"] = _normalise_input(body.get("input"))

    # Normalize modalities
    copy.setdefault("modalities", DEFAULT_MODALITIES)
    copy["modalities"] = _normalise_modalities(
        copy.get("modalities"), DEFAULT_MODALITIES
    )

    output_modalities = copy.get("output_modalities")
    copy["output_modalities"] = _normalise_modalities(
        output_modalities, copy["modalities"]
    )

    return copy


def has_media_content(normalised_body: dict[str, Any]) -> bool:
    """
    Check if request contains any media content (images, audio, video).

    Args:
        normalised_body: Normalized request body

    Returns:
        True if media content is present
    """
    # Check modalities
    modalities = set(normalised_body.get("modalities", []) or [])
    output_modalities = set(normalised_body.get("output_modalities", []) or [])
    media_modalities = {"image", "audio", "video"}

    if media_modalities & (modalities | output_modalities):
        return True

    # Check input turns for media parts
    media_types = {"input_image", "input_audio", "input_video"}
    for turn in normalised_body.get("input", []):
        if not isinstance(turn, dict):
            continue
        contents = turn.get("content", [])
        for part in contents:
            if isinstance(part, dict) and part.get("type") in media_types:
                return True

    return False


def parts_to_plaintext(parts: Iterable[dict[str, Any]] | Any) -> str:
    """
    Collapse normalized content parts into plaintext string.

    Used for converting multimodal content to text-only for
    models that don't support media.
    """
    if isinstance(parts, str):
        return parts

    if not isinstance(parts, Iterable):
        return _stringify(parts)

    lines: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            lines.append(_stringify(part))
            continue

        part_type = part.get("type")
        if part_type == "input_text" and part.get("text"):
            lines.append(_stringify(part["text"]))
        elif part_type == "input_image":
            url = part.get("image_url") or part.get("image_base64")
            if url:
                lines.append(
                    f"[Image: {url[:50]}...]"
                    if len(str(url)) > 50
                    else f"[Image: {url}]"
                )
        elif part_type == "input_audio":
            url = part.get("audio_url")
            if url:
                lines.append(f"[Audio: {url}]")
        elif part_type == "input_video":
            url = part.get("video_url")
            if url:
                lines.append(f"[Video: {url}]")
        elif part_type and part_type.startswith("output_"):
            text = part.get("text")
            if text:
                lines.append(_stringify(text))
        elif "text" in part:
            lines.append(_stringify(part["text"]))

    return "\n".join(lines)


def collect_system_preamble(body: dict[str, Any]) -> list[str]:
    """
    Collect system-level instructions from request.

    Extracts:
    - system_instruction / instructions field
    - text.format (JSON schema) as formatting instruction
    - reasoning guidance
    """
    preamble: list[str] = []

    # System instruction
    system_instruction = body.get("system_instruction") or body.get("instructions")
    if isinstance(system_instruction, str) and system_instruction.strip():
        preamble.append(system_instruction.strip())
    elif isinstance(system_instruction, dict):
        preamble.append(_stringify(system_instruction))

    # JSON schema format
    formatter = body.get("text", {}).get("format")
    if formatter:
        schema_text = _stringify(formatter)
        preamble.append(
            "When responding, conform strictly to the following JSON schema. "
            "Do not include any prose outside the JSON.\n" + schema_text
        )

    # Reasoning guidance
    reasoning = body.get("reasoning")
    if reasoning:
        preamble.append("Reasoning guidance: " + _stringify(reasoning))

    return preamble


def responses_to_chat_messages(normalised_body: dict[str, Any]) -> list[dict[str, str]]:
    """
    Convert normalized Responses format to chat messages format.

    Used for backends that only support chat/completions API.
    """
    messages: list[dict[str, str]] = []

    # Add system preamble
    preamble = collect_system_preamble(normalised_body)
    if preamble:
        messages.append({"role": "system", "content": "\n\n".join(preamble)})

    # Convert input turns
    for turn in normalised_body.get("input", []):
        if not isinstance(turn, dict):
            continue
        text = parts_to_plaintext(turn.get("content"))
        if not text:
            continue
        role = str(turn.get("role", "user")).lower()
        if role not in {"system", "user", "assistant", "tool", "developer"}:
            role = "user"
        messages.append({"role": role, "content": text})

    if not messages:
        messages.append({"role": "user", "content": ""})

    return messages
