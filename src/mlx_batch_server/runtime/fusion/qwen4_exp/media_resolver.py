"""Bounded, source-only media resolution for the 3more Qwen4Exp lane.

This module owns source policy, ordering, identity, and resource budgets. It
does not perform network or filesystem I/O by default and deliberately stops
before model preprocessing. Any operation that can touch external state or
interpret a file/image is supplied through an explicit port.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from dataclasses import dataclass, field
from typing import Protocol, TypeAlias, runtime_checkable
from urllib.parse import urlsplit

from ....vision.input import (
    AudioInput,
    FileInput,
    ImageInput,
    MediaInput,
    MediaSourceField,
    MultimodalInputPlan,
    VideoInput,
)

_DEFAULT_IMAGE_MEDIA_TYPES = frozenset(
    {"image/gif", "image/jpeg", "image/png", "image/webp"}
)
_DEFAULT_FILE_MEDIA_TYPES = frozenset(
    {*_DEFAULT_IMAGE_MEDIA_TYPES, "application/pdf", "text/plain"}
)
_MEDIA_DETAIL_LEVELS = frozenset({"auto", "low", "high", "original"})


class MediaResolverError(ValueError):
    """Structured fail-closed error from the source-resolution boundary."""

    def __init__(
        self,
        code: str,
        message: str,
        part_index: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.part_index = part_index


@dataclass(frozen=True, slots=True)
class MediaResolverLimits:
    """Hard per-request limits, all checked before model preprocessing."""

    max_direct_images: int = 8
    max_images: int = 16
    max_source_bytes: int = 32 * 1024 * 1024
    max_materialized_image_bytes: int = 64 * 1024 * 1024
    max_materialized_text_bytes: int = 16 * 1024 * 1024
    max_image_pixels: int = 16_777_216
    max_total_pixels: int = 67_108_864

    def __post_init__(self) -> None:
        for name in (
            "max_direct_images",
            "max_images",
            "max_source_bytes",
            "max_materialized_image_bytes",
            "max_materialized_text_bytes",
            "max_image_pixels",
            "max_total_pixels",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class AllowedUrlPolicy:
    """Exact-origin allowlist for injected URL fetchers.

    The empty default denies every URL. Origins must be canonical HTTP(S)
    origins such as ``https://media.3more.ai`` and cannot carry paths,
    credentials, queries, or fragments.
    """

    origins: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        normalized = frozenset(_canonical_origin(origin) for origin in self.origins)
        object.__setattr__(self, "origins", normalized)

    def permits(self, url: str) -> bool:
        try:
            parsed = urlsplit(url)
            origin = _origin_from_split(parsed)
        except ValueError:
            return False
        return not parsed.fragment and origin in self.origins


@dataclass(frozen=True, slots=True)
class SourceBlob:
    """Immutable bytes returned by an injected source adapter."""

    content: bytes
    media_type: str
    final_url: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.content, bytes) or not self.content:
            raise ValueError("source content must be non-empty immutable bytes")
        object.__setattr__(self, "media_type", _normalize_media_type(self.media_type))
        if self.final_url is not None and not self.final_url:
            raise ValueError("final_url must be non-empty when provided")


@dataclass(frozen=True, slots=True)
class UrlFetchRequest:
    url: str
    max_bytes: int
    accepted_media_types: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FileIdResolutionRequest:
    file_id: str
    max_bytes: int
    accepted_media_types: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ImageMaterializationRequest:
    part_index: int
    source: SourceBlob
    max_bytes: int
    max_pixels: int
    detail: str = "auto"

    def __post_init__(self) -> None:
        _validate_materialization_request(
            part_index=self.part_index,
            max_values=(self.max_bytes, self.max_pixels),
            detail=self.detail,
        )


@dataclass(frozen=True, slots=True)
class MaterializedImage:
    """Decoded image receipt, still prior to model-specific preprocessing."""

    content: bytes
    media_type: str
    width: int
    height: int

    def __post_init__(self) -> None:
        if not isinstance(self.content, bytes) or not self.content:
            raise ValueError("image content must be non-empty immutable bytes")
        object.__setattr__(self, "media_type", _normalize_media_type(self.media_type))
        if self.width < 1 or self.height < 1:
            raise ValueError("image dimensions must be positive")


@dataclass(frozen=True, slots=True)
class MaterializedText:
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("materialized text must be non-empty")


FileMaterializedItem: TypeAlias = MaterializedImage | MaterializedText


@dataclass(frozen=True, slots=True)
class FileMaterializationRequest:
    part_index: int
    filename: str | None
    source: SourceBlob
    max_images: int
    max_image_bytes: int
    max_text_bytes: int
    max_image_pixels: int
    max_total_pixels: int
    detail: str = "auto"

    def __post_init__(self) -> None:
        _validate_materialization_request(
            part_index=self.part_index,
            max_values=(
                self.max_images,
                self.max_image_bytes,
                self.max_text_bytes,
                self.max_image_pixels,
                self.max_total_pixels,
            ),
            detail=self.detail,
        )


@dataclass(frozen=True, slots=True)
class FileMaterialization:
    """Ordered file expansion supplied by an injected adapter.

    A PDF adapter may return page images and extracted text here. The resolver
    remains independent of the concrete parser and renderer.
    """

    items: tuple[FileMaterializedItem, ...]

    def __post_init__(self) -> None:
        items = tuple(self.items)
        if not items:
            raise ValueError("file materialization must not be empty")
        if any(
            not isinstance(item, MaterializedImage | MaterializedText) for item in items
        ):
            raise ValueError("file materialization contains an unsupported item")
        object.__setattr__(self, "items", items)


@runtime_checkable
class UrlFetcherPort(Protocol):
    async def fetch(self, request: UrlFetchRequest) -> SourceBlob: ...


@runtime_checkable
class FileIdResolverPort(Protocol):
    async def resolve(self, request: FileIdResolutionRequest) -> SourceBlob: ...


@runtime_checkable
class ImageMaterializerPort(Protocol):
    async def materialize(
        self,
        request: ImageMaterializationRequest,
    ) -> MaterializedImage: ...


@runtime_checkable
class FileMaterializerPort(Protocol):
    async def materialize(
        self,
        request: FileMaterializationRequest,
    ) -> FileMaterialization: ...


@dataclass(frozen=True, slots=True)
class ResolvedImage:
    part_index: int
    item_index: int
    media_type: str
    content: bytes
    width: int
    height: int
    source_digest: str
    content_digest: str

    def __post_init__(self) -> None:
        if self.part_index < 0 or self.item_index < 0:
            raise ValueError("resolved image indices must be non-negative")
        if not isinstance(self.content, bytes) or not self.content:
            raise ValueError("resolved image content must be immutable bytes")
        object.__setattr__(self, "media_type", _normalize_media_type(self.media_type))
        if self.width < 1 or self.height < 1:
            raise ValueError("resolved image dimensions must be positive")
        _require_digest(self.source_digest, "source_digest")
        _require_digest(self.content_digest, "content_digest")


@dataclass(frozen=True, slots=True)
class ResolvedText:
    part_index: int
    item_index: int
    text: str
    source_digest: str
    content_digest: str

    def __post_init__(self) -> None:
        if self.part_index < 0 or self.item_index < 0:
            raise ValueError("resolved text indices must be non-negative")
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("resolved text must be non-empty")
        _require_digest(self.source_digest, "source_digest")
        _require_digest(self.content_digest, "content_digest")


ResolvedMediaItem: TypeAlias = ResolvedImage | ResolvedText


@dataclass(frozen=True, slots=True)
class ResolvedMediaBundle:
    """Immutable, order-preserving handoff to a future preprocessor."""

    items: tuple[ResolvedMediaItem, ...]
    images: tuple[ResolvedImage, ...]
    texts: tuple[ResolvedText, ...]
    source_bytes: int
    materialized_image_bytes: int
    materialized_text_bytes: int
    total_pixels: int
    digest: str

    def __post_init__(self) -> None:
        items = tuple(self.items)
        images = tuple(self.images)
        texts = tuple(self.texts)
        if tuple(item for item in items if isinstance(item, ResolvedImage)) != images:
            raise ValueError("images must preserve their order in items")
        if tuple(item for item in items if isinstance(item, ResolvedText)) != texts:
            raise ValueError("texts must preserve their order in items")
        if (
            min(
                self.source_bytes,
                self.materialized_image_bytes,
                self.materialized_text_bytes,
                self.total_pixels,
            )
            < 0
        ):
            raise ValueError("bundle counters must be non-negative")
        _require_digest(self.digest, "digest")
        object.__setattr__(self, "items", items)
        object.__setattr__(self, "images", images)
        object.__setattr__(self, "texts", texts)


@dataclass(slots=True)
class _BudgetLedger:
    limits: MediaResolverLimits
    source_bytes: int = 0
    image_bytes: int = 0
    text_bytes: int = 0
    image_count: int = 0
    total_pixels: int = 0

    @property
    def remaining_source_bytes(self) -> int:
        return self.limits.max_source_bytes - self.source_bytes

    @property
    def remaining_image_bytes(self) -> int:
        return self.limits.max_materialized_image_bytes - self.image_bytes

    @property
    def remaining_text_bytes(self) -> int:
        return self.limits.max_materialized_text_bytes - self.text_bytes

    @property
    def remaining_images(self) -> int:
        return self.limits.max_images - self.image_count

    @property
    def remaining_pixels(self) -> int:
        return self.limits.max_total_pixels - self.total_pixels

    def add_source(self, blob: SourceBlob, part_index: int) -> None:
        self.source_bytes += len(blob.content)
        if self.source_bytes > self.limits.max_source_bytes:
            raise MediaResolverError(
                "source_bytes_exceeded",
                "media sources exceed the request byte budget",
                part_index,
            )

    def add_image(self, image: MaterializedImage, part_index: int) -> None:
        pixels = image.width * image.height
        if pixels > self.limits.max_image_pixels:
            raise MediaResolverError(
                "image_pixels_exceeded",
                "one image exceeds the per-image pixel budget",
                part_index,
            )
        self.image_count += 1
        self.image_bytes += len(image.content)
        self.total_pixels += pixels
        if self.image_count > self.limits.max_images:
            raise MediaResolverError(
                "image_count_exceeded",
                "media expands beyond the input image limit",
                part_index,
            )
        if self.image_bytes > self.limits.max_materialized_image_bytes:
            raise MediaResolverError(
                "image_bytes_exceeded",
                "materialized images exceed the request byte budget",
                part_index,
            )
        if self.total_pixels > self.limits.max_total_pixels:
            raise MediaResolverError(
                "total_pixels_exceeded",
                "materialized images exceed the request pixel budget",
                part_index,
            )

    def add_text(self, text: MaterializedText, part_index: int) -> None:
        self.text_bytes += len(text.text.encode("utf-8"))
        if self.text_bytes > self.limits.max_materialized_text_bytes:
            raise MediaResolverError(
                "text_bytes_exceeded",
                "materialized text exceeds the request byte budget",
                part_index,
            )


class SourceMediaResolver:
    """Resolve approved image/file sources without hidden I/O capabilities."""

    def __init__(
        self,
        *,
        limits: MediaResolverLimits = MediaResolverLimits(),
        url_policy: AllowedUrlPolicy = AllowedUrlPolicy(),
        url_fetcher: UrlFetcherPort | None = None,
        file_id_resolver: FileIdResolverPort | None = None,
        image_materializer: ImageMaterializerPort | None = None,
        file_materializer: FileMaterializerPort | None = None,
        image_media_types: frozenset[str] = _DEFAULT_IMAGE_MEDIA_TYPES,
        file_media_types: frozenset[str] = _DEFAULT_FILE_MEDIA_TYPES,
    ) -> None:
        self._limits = limits
        self._url_policy = url_policy
        self._url_fetcher = url_fetcher
        self._file_id_resolver = file_id_resolver
        self._image_materializer = image_materializer
        self._file_materializer = file_materializer
        self._image_media_types = _normalize_media_types(image_media_types)
        self._file_media_types = _normalize_media_types(file_media_types)

    async def resolve(self, plan: MultimodalInputPlan) -> ResolvedMediaBundle:
        """Materialize one plan atomically and return only validated outputs."""

        ordered = self._preflight(plan)
        ledger = _BudgetLedger(self._limits)
        resolved: list[ResolvedMediaItem] = []

        for descriptor in ordered:
            source_field, source_value = _select_source(descriptor)
            source_digest = source_identity_digest(descriptor)
            if isinstance(descriptor, ImageInput):
                image = await self._resolve_image_source(
                    descriptor,
                    source_field,
                    source_value,
                    ledger,
                )
                ledger.add_image(image, descriptor.part_index)
                resolved.append(
                    _resolved_image(descriptor.part_index, 0, source_digest, image)
                )
                continue

            if not isinstance(descriptor, FileInput):
                raise MediaResolverError(
                    "unsupported_media",
                    "only input_image and input_file are supported",
                    descriptor.part_index,
                )
            source = await self._resolve_file_source(
                descriptor,
                source_field,
                source_value,
                ledger,
            )
            if self._file_materializer is None:
                raise MediaResolverError(
                    "file_materializer_required",
                    "input_file requires an injected file materializer",
                    descriptor.part_index,
                )
            request = FileMaterializationRequest(
                part_index=descriptor.part_index,
                filename=descriptor.filename,
                source=source,
                max_images=ledger.remaining_images,
                max_image_bytes=ledger.remaining_image_bytes,
                max_text_bytes=ledger.remaining_text_bytes,
                max_image_pixels=self._limits.max_image_pixels,
                max_total_pixels=ledger.remaining_pixels,
                detail=_canonical_detail(descriptor.detail),
            )
            materialization = await self._file_materializer.materialize(request)
            if not isinstance(materialization, FileMaterialization):
                raise MediaResolverError(
                    "invalid_file_materialization",
                    "file materializer returned an invalid receipt",
                    descriptor.part_index,
                )
            for item_index, item in enumerate(materialization.items):
                if isinstance(item, MaterializedImage):
                    self._validate_image_type(item.media_type, descriptor.part_index)
                    ledger.add_image(item, descriptor.part_index)
                    resolved.append(
                        _resolved_image(
                            descriptor.part_index,
                            item_index,
                            source_digest,
                            item,
                        )
                    )
                elif isinstance(item, MaterializedText):
                    ledger.add_text(item, descriptor.part_index)
                    resolved.append(
                        ResolvedText(
                            part_index=descriptor.part_index,
                            item_index=item_index,
                            text=item.text,
                            source_digest=source_digest,
                            content_digest=_bytes_digest(item.text.encode("utf-8")),
                        )
                    )
                else:
                    raise MediaResolverError(
                        "unsupported_file_output",
                        "file materializer returned unsupported media",
                        descriptor.part_index,
                    )

        items = tuple(resolved)
        images = tuple(item for item in items if isinstance(item, ResolvedImage))
        texts = tuple(item for item in items if isinstance(item, ResolvedText))
        digest = _json_digest([_item_digest_payload(item) for item in items])
        return ResolvedMediaBundle(
            items=items,
            images=images,
            texts=texts,
            source_bytes=ledger.source_bytes,
            materialized_image_bytes=ledger.image_bytes,
            materialized_text_bytes=ledger.text_bytes,
            total_pixels=ledger.total_pixels,
            digest=digest,
        )

    def _preflight(self, plan: MultimodalInputPlan) -> tuple[MediaInput, ...]:
        if not isinstance(plan, MultimodalInputPlan):
            raise MediaResolverError("invalid_plan", "plan must be multimodal")
        all_indices = [item.part_index for item in (*plan.prompt, *plan.media)]
        if any(index < 0 for index in all_indices) or len(set(all_indices)) != len(
            all_indices
        ):
            raise MediaResolverError(
                "invalid_part_order",
                "part indices must be unique and non-negative",
            )
        direct_images = 0
        for descriptor in plan.media:
            _select_source(descriptor)
            if isinstance(descriptor, AudioInput | VideoInput):
                raise MediaResolverError(
                    "unsupported_media",
                    "audio and video inputs are outside the 3more launch contract",
                    descriptor.part_index,
                )
            if not isinstance(descriptor, ImageInput | FileInput):
                raise MediaResolverError(
                    "unsupported_media",
                    "unknown media descriptor",
                    descriptor.part_index,
                )
            direct_images += isinstance(descriptor, ImageInput)
        if direct_images > self._limits.max_direct_images:
            raise MediaResolverError(
                "image_count_exceeded",
                "input contains more than the allowed number of direct images",
            )
        return tuple(sorted(plan.media, key=lambda item: item.part_index))

    async def _resolve_image_source(
        self,
        descriptor: ImageInput,
        source_field: MediaSourceField,
        source_value: str,
        ledger: _BudgetLedger,
    ) -> MaterializedImage:
        source = await self._resolve_blob(
            source_field=source_field,
            source_value=source_value,
            part_index=descriptor.part_index,
            accepted_media_types=self._image_media_types,
            ledger=ledger,
        )
        if self._image_materializer is None:
            raise MediaResolverError(
                "image_materializer_required",
                "input_image requires an injected image materializer",
                descriptor.part_index,
            )
        request = ImageMaterializationRequest(
            part_index=descriptor.part_index,
            source=source,
            max_bytes=ledger.remaining_image_bytes,
            max_pixels=min(
                self._limits.max_image_pixels,
                ledger.remaining_pixels,
            ),
            detail=_canonical_detail(descriptor.detail),
        )
        image = await self._image_materializer.materialize(request)
        if not isinstance(image, MaterializedImage):
            raise MediaResolverError(
                "invalid_image_materialization",
                "image materializer returned an invalid receipt",
                descriptor.part_index,
            )
        self._validate_image_type(image.media_type, descriptor.part_index)
        return image

    async def _resolve_file_source(
        self,
        descriptor: FileInput,
        source_field: MediaSourceField,
        source_value: str,
        ledger: _BudgetLedger,
    ) -> SourceBlob:
        if source_field is MediaSourceField.FILE_ID:
            if self._file_id_resolver is None:
                raise MediaResolverError(
                    "file_id_resolver_required",
                    "file_id requires an injected resolver",
                    descriptor.part_index,
                )
            file_request = FileIdResolutionRequest(
                file_id=source_value,
                max_bytes=ledger.remaining_source_bytes,
                accepted_media_types=tuple(sorted(self._file_media_types)),
            )
            blob = await self._file_id_resolver.resolve(file_request)
            return self._accept_blob(
                blob,
                self._file_media_types,
                descriptor.part_index,
                ledger,
            )
        return await self._resolve_blob(
            source_field=source_field,
            source_value=source_value,
            part_index=descriptor.part_index,
            accepted_media_types=self._file_media_types,
            ledger=ledger,
        )

    async def _resolve_blob(
        self,
        *,
        source_field: MediaSourceField,
        source_value: str,
        part_index: int,
        accepted_media_types: frozenset[str],
        ledger: _BudgetLedger,
    ) -> SourceBlob:
        if source_field is MediaSourceField.FILE_ID:
            if self._file_id_resolver is None:
                raise MediaResolverError(
                    "file_id_resolver_required",
                    "file_id requires an injected resolver",
                    part_index,
                )
            request = FileIdResolutionRequest(
                file_id=source_value,
                max_bytes=ledger.remaining_source_bytes,
                accepted_media_types=tuple(sorted(accepted_media_types)),
            )
            blob = await self._file_id_resolver.resolve(request)
            return self._accept_blob(
                blob,
                accepted_media_types,
                part_index,
                ledger,
            )
        if source_value.startswith("data:"):
            blob = _decode_data_url(
                source_value,
                ledger.remaining_source_bytes,
                part_index,
            )
            return self._accept_blob(blob, accepted_media_types, part_index, ledger)
        if source_field in (
            MediaSourceField.IMAGE_BASE64,
            MediaSourceField.FILE_DATA,
        ):
            raise MediaResolverError(
                "data_url_required",
                f"{source_field.value} must be an explicit base64 data URL",
                part_index,
            )
        if source_field not in (
            MediaSourceField.IMAGE_URL,
            MediaSourceField.FILE_URL,
        ):
            raise MediaResolverError(
                "unsupported_source",
                f"cannot resolve {source_field.value}",
                part_index,
            )
        if not self._url_policy.permits(source_value):
            raise MediaResolverError(
                "url_not_allowed",
                "URL origin is not explicitly allowed",
                part_index,
            )
        if self._url_fetcher is None:
            raise MediaResolverError(
                "url_fetcher_required",
                "allowed URL requires an injected fetcher",
                part_index,
            )
        url_request = UrlFetchRequest(
            url=source_value,
            max_bytes=ledger.remaining_source_bytes,
            accepted_media_types=tuple(sorted(accepted_media_types)),
        )
        blob = await self._url_fetcher.fetch(url_request)
        if not isinstance(blob, SourceBlob) or blob.final_url is None:
            raise MediaResolverError(
                "invalid_fetch_receipt",
                "URL fetcher must return a SourceBlob with final_url",
                part_index,
            )
        if not self._url_policy.permits(blob.final_url):
            raise MediaResolverError(
                "redirect_not_allowed",
                "URL fetcher crossed the configured origin boundary",
                part_index,
            )
        return self._accept_blob(blob, accepted_media_types, part_index, ledger)

    @staticmethod
    def _accept_blob(
        blob: object,
        accepted_media_types: frozenset[str],
        part_index: int,
        ledger: _BudgetLedger,
    ) -> SourceBlob:
        if not isinstance(blob, SourceBlob):
            raise MediaResolverError(
                "invalid_source_receipt",
                "source adapter returned an invalid receipt",
                part_index,
            )
        if blob.media_type not in accepted_media_types:
            raise MediaResolverError(
                "unsupported_media_type",
                f"media type {blob.media_type!r} is not accepted",
                part_index,
            )
        ledger.add_source(blob, part_index)
        return blob

    def _validate_image_type(self, media_type: str, part_index: int) -> None:
        if _normalize_media_type(media_type) not in self._image_media_types:
            raise MediaResolverError(
                "unsupported_image_type",
                f"materialized image type {media_type!r} is not accepted",
                part_index,
            )


def _select_source(descriptor: MediaInput) -> tuple[MediaSourceField, str]:
    if isinstance(descriptor, ImageInput):
        candidates = (
            (MediaSourceField.FILE_ID, descriptor.file_id),
            (MediaSourceField.IMAGE_URL, descriptor.image_url),
            (MediaSourceField.IMAGE_BASE64, descriptor.image_base64),
        )
    elif isinstance(descriptor, FileInput):
        candidates = (
            (MediaSourceField.FILE_ID, descriptor.file_id),
            (MediaSourceField.FILE_URL, descriptor.file_url),
            (MediaSourceField.FILE_DATA, descriptor.file_data),
        )
    elif isinstance(descriptor, AudioInput):
        candidates = ((MediaSourceField.AUDIO_URL, descriptor.audio_url),)
    elif isinstance(descriptor, VideoInput):
        candidates = ((MediaSourceField.VIDEO_URL, descriptor.video_url),)
    else:
        raise MediaResolverError("unsupported_media", "unknown media descriptor")
    present = [(field, value) for field, value in candidates if value is not None]
    if len(present) != 1:
        raise MediaResolverError(
            "ambiguous_source",
            "media descriptor must contain exactly one source",
            descriptor.part_index,
        )
    source_field, source_value = present[0]
    if not isinstance(source_value, str) or not source_value:
        raise MediaResolverError(
            "invalid_source",
            "media source must be a non-empty string",
            descriptor.part_index,
        )
    return source_field, source_value


def source_identity_digest(descriptor: MediaInput) -> str:
    """Return the canonical source identity used by bundle consumers."""

    source_field, source_value = _select_source(descriptor)
    detail = (
        _canonical_detail(descriptor.detail)
        if isinstance(descriptor, ImageInput | FileInput)
        else None
    )
    return _json_digest(
        {"detail": detail, "field": source_field.value, "value": source_value}
    )


def _canonical_detail(value: str | None) -> str:
    detail = "auto" if value is None else value
    if detail not in _MEDIA_DETAIL_LEVELS:
        raise ValueError("detail must be auto, low, high, or original")
    return detail


def _validate_materialization_request(
    *,
    part_index: int,
    max_values: tuple[int, ...],
    detail: str,
) -> None:
    if part_index < 0:
        raise ValueError("part_index must be non-negative")
    if any(value < 0 for value in max_values):
        raise ValueError("materialization budgets must not be negative")
    _canonical_detail(detail)


def _decode_data_url(value: str, max_bytes: int, part_index: int) -> SourceBlob:
    header, separator, payload = value.partition(",")
    if not separator or not header.endswith(";base64"):
        raise MediaResolverError(
            "invalid_data_url",
            "only explicit base64 data URLs are supported",
            part_index,
        )
    media_type = header[5:-7]
    try:
        normalized_type = _normalize_media_type(media_type)
    except ValueError as exc:
        raise MediaResolverError(
            "invalid_data_url",
            "data URL requires an explicit media type",
            part_index,
        ) from exc
    estimated_size = (len(payload) * 3) // 4
    if estimated_size > max_bytes + 2:
        raise MediaResolverError(
            "source_bytes_exceeded",
            "data URL exceeds the remaining source byte budget",
            part_index,
        )
    try:
        content = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise MediaResolverError(
            "invalid_data_url",
            "data URL payload is not valid base64",
            part_index,
        ) from exc
    try:
        return SourceBlob(content=content, media_type=normalized_type)
    except ValueError as exc:
        raise MediaResolverError(
            "invalid_data_url",
            "data URL payload must not be empty",
            part_index,
        ) from exc


def _resolved_image(
    part_index: int,
    item_index: int,
    source_digest: str,
    image: MaterializedImage,
) -> ResolvedImage:
    return ResolvedImage(
        part_index=part_index,
        item_index=item_index,
        media_type=image.media_type,
        content=image.content,
        width=image.width,
        height=image.height,
        source_digest=source_digest,
        content_digest=_bytes_digest(image.content),
    )


def _item_digest_payload(item: ResolvedMediaItem) -> dict[str, object]:
    payload: dict[str, object] = {
        "part_index": item.part_index,
        "item_index": item.item_index,
        "source_digest": item.source_digest,
        "content_digest": item.content_digest,
    }
    if isinstance(item, ResolvedImage):
        payload.update(
            {
                "kind": "image",
                "media_type": item.media_type,
                "width": item.width,
                "height": item.height,
            }
        )
    else:
        payload["kind"] = "text"
    return payload


def _canonical_origin(value: str) -> str:
    parsed = urlsplit(value)
    origin = _origin_from_split(parsed)
    if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise ValueError("allowed URL origins cannot include paths or parameters")
    return origin


def _origin_from_split(parsed: object) -> str:
    scheme = getattr(parsed, "scheme", "").lower()
    hostname = getattr(parsed, "hostname", None)
    username = getattr(parsed, "username", None)
    password = getattr(parsed, "password", None)
    if scheme not in ("http", "https") or not hostname or username or password:
        raise ValueError("URL must have a credential-free HTTP(S) origin")
    port = getattr(parsed, "port", None)
    default_port = 80 if scheme == "http" else 443
    suffix = "" if port in (None, default_port) else f":{port}"
    return f"{scheme}://{hostname.lower()}{suffix}"


def _normalize_media_type(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("media type must be a string")
    normalized = value.split(";", 1)[0].strip().lower()
    if not normalized or "/" not in normalized:
        raise ValueError("media type must be explicit")
    return normalized


def _normalize_media_types(values: frozenset[str]) -> frozenset[str]:
    normalized = frozenset(_normalize_media_type(value) for value in values)
    if not normalized:
        raise ValueError("accepted media types must not be empty")
    return normalized


def _bytes_digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return _bytes_digest(encoded)


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
    "AllowedUrlPolicy",
    "FileIdResolutionRequest",
    "FileIdResolverPort",
    "FileMaterialization",
    "FileMaterializationRequest",
    "FileMaterializedItem",
    "FileMaterializerPort",
    "ImageMaterializationRequest",
    "ImageMaterializerPort",
    "MaterializedImage",
    "MaterializedText",
    "MediaResolverError",
    "MediaResolverLimits",
    "ResolvedImage",
    "ResolvedMediaBundle",
    "ResolvedMediaItem",
    "ResolvedText",
    "SourceBlob",
    "SourceMediaResolver",
    "UrlFetchRequest",
    "UrlFetcherPort",
    "source_identity_digest",
]
