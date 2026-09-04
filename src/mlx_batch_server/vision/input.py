"""Pure multimodal input planning for target-owned inference adapters.

This module deliberately stops before source resolution. It validates canonical
Responses content parts and carries opaque references to a capability-matched
runtime adapter without fetching, decoding, or opening them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeAlias


class MediaSourceField(StrEnum):
    """Exact normalized wire field an adapter is able to resolve."""

    IMAGE_URL = "image_url"
    IMAGE_BASE64 = "image_base64"
    FILE_ID = "file_id"
    FILE_URL = "file_url"
    FILE_DATA = "file_data"
    AUDIO_URL = "audio_url"
    VIDEO_URL = "video_url"


@dataclass(frozen=True, slots=True)
class MultimodalInputCapabilities:
    """Source forms explicitly accepted by one runtime adapter.

    The empty default is intentionally text-only. Capability inference from a
    related source form is forbidden: accepting ``file_url`` does not imply
    support for ``file_id`` or inline ``file_data``.
    """

    accepted_sources: frozenset[MediaSourceField] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        try:
            normalized = frozenset(
                MediaSourceField(value) for value in self.accepted_sources
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "accepted_sources contains an unknown source field"
            ) from exc
        object.__setattr__(self, "accepted_sources", normalized)

    def accepts(self, source: MediaSourceField) -> bool:
        return source in self.accepted_sources


@dataclass(frozen=True, slots=True)
class PromptText:
    part_index: int
    text: str


@dataclass(frozen=True, slots=True)
class ImageInput:
    part_index: int
    file_id: str | None = None
    image_url: str | None = None
    image_base64: str | None = None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class FileInput:
    part_index: int
    file_id: str | None = None
    file_url: str | None = None
    file_data: str | None = None
    filename: str | None = None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class AudioInput:
    part_index: int
    audio_url: str


@dataclass(frozen=True, slots=True)
class VideoInput:
    part_index: int
    video_url: str


MediaInput: TypeAlias = ImageInput | FileInput | AudioInput | VideoInput


@dataclass(frozen=True, slots=True)
class MultimodalInputPlan:
    """Immutable, ordered prompt and media descriptors for one content list."""

    prompt: tuple[PromptText, ...]
    media: tuple[MediaInput, ...]


class MultimodalInputError(ValueError):
    """Structured fail-closed boundary error."""

    def __init__(
        self,
        code: str,
        part_index: int,
        part_type: object,
        message: str,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.part_index = part_index
        self.part_type = part_type


_SOURCE_FIELDS: dict[str, tuple[MediaSourceField, ...]] = {
    "input_image": (
        MediaSourceField.FILE_ID,
        MediaSourceField.IMAGE_URL,
        MediaSourceField.IMAGE_BASE64,
    ),
    "input_file": (
        MediaSourceField.FILE_ID,
        MediaSourceField.FILE_URL,
        MediaSourceField.FILE_DATA,
    ),
    "input_audio": (MediaSourceField.AUDIO_URL,),
    "input_video": (MediaSourceField.VIDEO_URL,),
}
_MEDIA_DETAIL_LEVELS = frozenset({"auto", "low", "high", "original"})


class MultimodalInputPlanner:
    """Validate normalized parts and produce a resolution-free runtime plan."""

    def __init__(self, capabilities: MultimodalInputCapabilities) -> None:
        self._capabilities = capabilities

    def plan(self, parts: Sequence[Mapping[str, object]]) -> MultimodalInputPlan:
        prompt: list[PromptText] = []
        media: list[MediaInput] = []

        for part_index, part in enumerate(parts):
            if not isinstance(part, Mapping):
                raise MultimodalInputError(
                    "invalid_part",
                    part_index,
                    None,
                    f"content part {part_index} must be a mapping",
                )

            part_type = part.get("type")
            if part_type == "input_text":
                prompt.append(
                    PromptText(
                        part_index=part_index,
                        text=self._required_text(part, "text", part_index, part_type),
                    )
                )
                continue

            if not isinstance(part_type, str):
                raise MultimodalInputError(
                    "unsupported_part_type",
                    part_index,
                    part_type,
                    f"content part {part_index} has unsupported type {part_type!r}",
                )
            expected_sources = _SOURCE_FIELDS.get(part_type)
            if expected_sources is None:
                raise MultimodalInputError(
                    "unsupported_part_type",
                    part_index,
                    part_type,
                    f"content part {part_index} has unsupported type {part_type!r}",
                )

            source, value = self._select_source(
                part,
                expected_sources,
                part_index,
                part_type,
            )
            if not self._capabilities.accepts(source):
                raise MultimodalInputError(
                    "unsupported_source",
                    part_index,
                    part_type,
                    f"runtime does not accept {part_type} via {source.value}",
                )

            if part_type == "input_image":
                media.append(
                    ImageInput(
                        part_index=part_index,
                        file_id=(value if source is MediaSourceField.FILE_ID else None),
                        image_url=(
                            value if source is MediaSourceField.IMAGE_URL else None
                        ),
                        image_base64=(
                            value if source is MediaSourceField.IMAGE_BASE64 else None
                        ),
                        detail=self._optional_detail(
                            part,
                            "detail",
                            part_index,
                            part_type,
                        ),
                    )
                )
            elif part_type == "input_file":
                media.append(
                    FileInput(
                        part_index=part_index,
                        file_id=value if source is MediaSourceField.FILE_ID else None,
                        file_url=value if source is MediaSourceField.FILE_URL else None,
                        file_data=value
                        if source is MediaSourceField.FILE_DATA
                        else None,
                        filename=self._optional_text(
                            part,
                            "filename",
                            part_index,
                            part_type,
                        ),
                        detail=self._optional_detail(
                            part,
                            "detail",
                            part_index,
                            part_type,
                        ),
                    )
                )
            elif part_type == "input_audio":
                media.append(AudioInput(part_index=part_index, audio_url=value))
            else:
                media.append(VideoInput(part_index=part_index, video_url=value))

        return MultimodalInputPlan(prompt=tuple(prompt), media=tuple(media))

    @staticmethod
    def _select_source(
        part: Mapping[str, object],
        expected: tuple[MediaSourceField, ...],
        part_index: int,
        part_type: str,
    ) -> tuple[MediaSourceField, str]:
        present: list[tuple[MediaSourceField, str]] = []
        for source in MediaSourceField:
            raw = part.get(source.value)
            if raw is None:
                continue
            if not isinstance(raw, str) or not raw:
                raise MultimodalInputError(
                    "invalid_source",
                    part_index,
                    part_type,
                    f"{source.value} must be a non-empty string",
                )
            present.append((source, raw))

        if not present:
            names = ", ".join(source.value for source in expected)
            raise MultimodalInputError(
                "missing_source",
                part_index,
                part_type,
                f"{part_type} requires exactly one of: {names}",
            )
        if len(present) > 1:
            names = ", ".join(source.value for source, _ in present)
            raise MultimodalInputError(
                "ambiguous_source",
                part_index,
                part_type,
                f"{part_type} supplied multiple sources: {names}",
            )
        source, value = present[0]
        if source not in expected:
            names = ", ".join(candidate.value for candidate in expected)
            raise MultimodalInputError(
                "invalid_source_field",
                part_index,
                part_type,
                f"{source.value} is invalid for {part_type}; expected one of: {names}",
            )
        return source, value

    @staticmethod
    def _required_text(
        part: Mapping[str, object],
        key: str,
        part_index: int,
        part_type: object,
    ) -> str:
        value = part.get(key)
        if not isinstance(value, str):
            raise MultimodalInputError(
                "invalid_text",
                part_index,
                part_type,
                f"{key} must be a string",
            )
        return value

    @staticmethod
    def _optional_text(
        part: Mapping[str, object],
        key: str,
        part_index: int,
        part_type: object,
    ) -> str | None:
        value = part.get(key)
        if value is None:
            return None
        if not isinstance(value, str) or not value:
            raise MultimodalInputError(
                "invalid_metadata",
                part_index,
                part_type,
                f"{key} must be a non-empty string when provided",
            )
        return value

    @classmethod
    def _optional_detail(
        cls,
        part: Mapping[str, object],
        key: str,
        part_index: int,
        part_type: object,
    ) -> str | None:
        value = cls._optional_text(part, key, part_index, part_type)
        if value is not None and value not in _MEDIA_DETAIL_LEVELS:
            raise MultimodalInputError(
                "invalid_metadata",
                part_index,
                part_type,
                "detail must be auto, low, high, or original",
            )
        return value


def plan_multimodal_input(
    parts: Sequence[Mapping[str, object]],
    capabilities: MultimodalInputCapabilities,
) -> MultimodalInputPlan:
    """Convenience entry point for the pure planning boundary."""

    return MultimodalInputPlanner(capabilities).plan(parts)
