"""Concrete, bounded adapters for the Qwen4Exp media-resolution ports.

Network acquisition is owned by SafePublicFetch: public HTTP(S) URLs work
without an allowlist, while SSRF, redirect rebinding, and resource limits
fail closed. Exact-origin lockdown remains optional. Image and file
interpretation runs outside the event loop, before runtime admission. The
default composition includes a bounded PDF renderer but does not invent a
``file_id`` store; foreign identifiers remain an explicit product-edge port.
"""

from __future__ import annotations

import asyncio
import io
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from PIL import Image, ImageOps, UnidentifiedImageError

if TYPE_CHECKING:
    import httpx

from ....utils.safe_public_fetch import (
    SafePublicFetch,
    SafePublicFetchError,
    SafePublicFetchLimits,
)
from .media_resolver import (
    AllowedUrlPolicy,
    FileIdResolverPort,
    FileMaterialization,
    FileMaterializationRequest,
    FileMaterializerPort,
    ImageMaterializationRequest,
    MaterializedImage,
    MaterializedText,
    MediaResolverError,
    MediaResolverLimits,
    SourceBlob,
    SourceMediaResolver,
    UrlFetchRequest,
)
from .pdf_materializer import PyMuPDFFileMaterializer

_IMAGE_FORMAT_MEDIA_TYPES: Final = {
    "GIF": "image/gif",
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}


@dataclass(frozen=True, slots=True)
class HttpFetchPolicy:
    """Transport limits applied independently to every source fetch."""

    connect_timeout_s: float = 5.0
    read_timeout_s: float = 20.0
    write_timeout_s: float = 5.0
    pool_timeout_s: float = 5.0
    max_redirects: int = 3
    chunk_bytes: int = 64 * 1024

    def __post_init__(self) -> None:
        if (
            min(
                self.connect_timeout_s,
                self.read_timeout_s,
                self.write_timeout_s,
                self.pool_timeout_s,
            )
            <= 0
        ):
            raise ValueError("HTTP timeouts must be positive")
        if self.max_redirects < 0:
            raise ValueError("max_redirects must not be negative")
        if self.chunk_bytes < 1:
            raise ValueError("chunk_bytes must be positive")


class HttpxUrlFetcher:
    """Fetch public HTTP(S) media through SafePublicFetch.

    Exact-origin lockdown is optional. The default empty policy does not
    bypass SSRF, redirect, or resource checks owned by SafePublicFetch.
    """

    def __init__(
        self,
        *,
        url_policy: AllowedUrlPolicy,
        fetch_policy: HttpFetchPolicy = HttpFetchPolicy(),
        transport: httpx.AsyncBaseTransport | None = None,
        getaddrinfo: Callable[..., object] | None = None,
    ) -> None:
        self._safe = SafePublicFetch(
            limits=SafePublicFetchLimits(
                timeout=fetch_policy.read_timeout_s,
                connect_timeout=fetch_policy.connect_timeout_s,
                write_timeout=fetch_policy.write_timeout_s,
                pool_timeout=fetch_policy.pool_timeout_s,
                max_redirects=fetch_policy.max_redirects,
                chunk_bytes=fetch_policy.chunk_bytes,
            ),
            allowed_origins=url_policy.origins,
            transport=transport,
            getaddrinfo=getaddrinfo,
        )

    async def fetch(self, request: UrlFetchRequest) -> SourceBlob:
        try:
            resource = await self._safe.fetch(
                request.url,
                accepted_media_types=request.accepted_media_types,
                max_bytes=request.max_bytes,
            )
        except SafePublicFetchError as exc:
            raise MediaResolverError(exc.code, str(exc)) from exc
        return SourceBlob(
            content=resource.content,
            media_type=resource.media_type,
            final_url=resource.final_url,
        )


class PillowImageMaterializer:
    """Validate image bytes and return geometry without retaining PIL state."""

    async def materialize(
        self,
        request: ImageMaterializationRequest,
    ) -> MaterializedImage:
        return await asyncio.to_thread(self._materialize_sync, request)

    @staticmethod
    def _materialize_sync(
        request: ImageMaterializationRequest,
    ) -> MaterializedImage:
        content = request.source.content
        if len(content) > request.max_bytes:
            raise MediaResolverError(
                "image_bytes_exceeded",
                "image exceeds the remaining materialized byte budget",
                request.part_index,
            )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(content)) as opened:
                    image_format = opened.format
                    opened.load()
                    oriented = ImageOps.exif_transpose(opened)
                    width, height = oriented.size
        except (
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
            UnidentifiedImageError,
            OSError,
        ) as exc:
            raise MediaResolverError(
                "image_decode_failed",
                "image payload cannot be decoded safely",
                request.part_index,
            ) from exc

        media_type = _IMAGE_FORMAT_MEDIA_TYPES.get(str(image_format).upper())
        if media_type is None:
            raise MediaResolverError(
                "unsupported_image_format",
                "decoded image format is not supported",
                request.part_index,
            )
        if media_type != request.source.media_type:
            raise MediaResolverError(
                "image_media_type_mismatch",
                "declared and decoded image media types differ",
                request.part_index,
            )
        if width * height > request.max_pixels:
            raise MediaResolverError(
                "image_pixels_exceeded",
                "image exceeds the remaining pixel budget",
                request.part_index,
            )
        return MaterializedImage(
            content=content,
            media_type=media_type,
            width=width,
            height=height,
        )


class StandardFileMaterializer:
    """Materialize image and UTF-8 files, with explicit PDF delegation."""

    def __init__(
        self,
        *,
        image_materializer: PillowImageMaterializer,
        pdf_materializer: FileMaterializerPort | None = None,
    ) -> None:
        self._image_materializer = image_materializer
        self._pdf_materializer = pdf_materializer

    async def materialize(
        self,
        request: FileMaterializationRequest,
    ) -> FileMaterialization:
        media_type = request.source.media_type
        if media_type.startswith("image/"):
            image = await self._image_materializer.materialize(
                ImageMaterializationRequest(
                    part_index=request.part_index,
                    source=request.source,
                    max_bytes=request.max_image_bytes,
                    max_pixels=min(
                        request.max_image_pixels,
                        request.max_total_pixels,
                    ),
                    detail=request.detail,
                )
            )
            return FileMaterialization(items=(image,))
        if media_type == "text/plain":
            text = await asyncio.to_thread(self._decode_text, request)
            return FileMaterialization(items=(text,))
        if media_type == "application/pdf":
            if self._pdf_materializer is None:
                raise MediaResolverError(
                    "pdf_materializer_required",
                    "PDF input requires an injected product-owned materializer",
                    request.part_index,
                )
            return await self._pdf_materializer.materialize(request)
        raise MediaResolverError(
            "unsupported_file_type",
            f"file media type {media_type!r} is not supported",
            request.part_index,
        )

    @staticmethod
    def _decode_text(request: FileMaterializationRequest) -> MaterializedText:
        if len(request.source.content) > request.max_text_bytes:
            raise MediaResolverError(
                "text_bytes_exceeded",
                "text file exceeds the remaining materialized byte budget",
                request.part_index,
            )
        try:
            text = request.source.content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise MediaResolverError(
                "text_decode_failed",
                "text/plain input must be valid UTF-8",
                request.part_index,
            ) from exc
        if not text:
            raise MediaResolverError(
                "empty_text_file",
                "text/plain input must not be empty",
                request.part_index,
            )
        return MaterializedText(text=text)


def compose_source_media_resolver(
    *,
    allowed_url_origins: Iterable[str] = (),
    limits: MediaResolverLimits = MediaResolverLimits(),
    fetch_policy: HttpFetchPolicy = HttpFetchPolicy(),
    file_id_resolver: FileIdResolverPort | None = None,
    pdf_materializer: FileMaterializerPort | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    getaddrinfo: Callable[..., object] | None = None,
) -> SourceMediaResolver:
    """Compose bounded image, text, and PDF ports without file-id ownership.

    Public HTTP(S) URLs are fetched through SafePublicFetch by default.
    ``allowed_url_origins`` is an optional exact-origin lockdown.
    """

    url_policy = AllowedUrlPolicy(frozenset(allowed_url_origins))
    image_materializer = PillowImageMaterializer()
    file_materializer = StandardFileMaterializer(
        image_materializer=image_materializer,
        pdf_materializer=pdf_materializer or PyMuPDFFileMaterializer(),
    )
    url_fetcher = HttpxUrlFetcher(
        url_policy=url_policy,
        fetch_policy=fetch_policy,
        transport=transport,
        getaddrinfo=getaddrinfo,
    )
    return SourceMediaResolver(
        limits=limits,
        url_policy=url_policy,
        url_fetcher=url_fetcher,
        file_id_resolver=file_id_resolver,
        image_materializer=image_materializer,
        file_materializer=file_materializer,
    )


__all__ = [
    "HttpFetchPolicy",
    "HttpxUrlFetcher",
    "PillowImageMaterializer",
    "PyMuPDFFileMaterializer",
    "StandardFileMaterializer",
    "compose_source_media_resolver",
]
