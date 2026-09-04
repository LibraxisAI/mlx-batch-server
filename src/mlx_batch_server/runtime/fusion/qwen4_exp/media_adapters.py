"""Concrete, bounded adapters for the Qwen4Exp media-resolution ports.

Network acquisition remains restricted to explicitly trusted origins. Image
and file interpretation runs outside the event loop, before runtime admission.
The default composition includes a bounded PDF renderer but does not invent a
``file_id`` store; foreign identifiers remain an explicit product-edge port.
"""

from __future__ import annotations

import asyncio
import io
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final
from urllib.parse import urljoin

import httpx
from PIL import Image, ImageOps, UnidentifiedImageError

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

_REDIRECT_STATUSES: Final = frozenset({301, 302, 303, 307, 308})
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
    """Fetch approved HTTP(S) media without ambient proxy or auth state."""

    def __init__(
        self,
        *,
        url_policy: AllowedUrlPolicy,
        fetch_policy: HttpFetchPolicy = HttpFetchPolicy(),
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._url_policy = url_policy
        self._fetch_policy = fetch_policy
        self._transport = transport

    async def fetch(self, request: UrlFetchRequest) -> SourceBlob:
        if request.max_bytes < 1:
            raise MediaResolverError(
                "invalid_fetch_budget",
                "URL fetch byte budget must be positive",
            )
        normalized_types = {
            _normalize_media_type(value) for value in request.accepted_media_types
        }
        accepted = tuple(sorted(normalized_types))
        if not accepted:
            raise MediaResolverError(
                "invalid_fetch_media_types",
                "URL fetch requires accepted media types",
            )
        if not self._url_policy.permits(request.url):
            raise MediaResolverError(
                "url_not_allowed",
                "URL origin is not explicitly allowed",
            )

        timeout = httpx.Timeout(
            connect=self._fetch_policy.connect_timeout_s,
            read=self._fetch_policy.read_timeout_s,
            write=self._fetch_policy.write_timeout_s,
            pool=self._fetch_policy.pool_timeout_s,
        )
        try:
            async with httpx.AsyncClient(
                follow_redirects=False,
                timeout=timeout,
                transport=self._transport,
                trust_env=False,
            ) as client:
                return await self._fetch_with_client(
                    client=client,
                    request=request,
                    accepted=accepted,
                )
        except MediaResolverError:
            raise
        except httpx.TimeoutException as exc:
            raise MediaResolverError(
                "url_fetch_timeout",
                "URL fetch exceeded its transport deadline",
            ) from exc
        except httpx.HTTPError as exc:
            raise MediaResolverError(
                "url_fetch_failed",
                "URL fetch failed",
            ) from exc

    async def _fetch_with_client(
        self,
        *,
        client: httpx.AsyncClient,
        request: UrlFetchRequest,
        accepted: tuple[str, ...],
    ) -> SourceBlob:
        current_url = request.url
        for redirects in range(self._fetch_policy.max_redirects + 1):
            if not self._url_policy.permits(current_url):
                raise MediaResolverError(
                    "redirect_not_allowed",
                    "URL redirect crossed the configured origin boundary",
                )
            async with client.stream(
                "GET",
                current_url,
                headers={
                    "Accept": ", ".join(accepted),
                    "User-Agent": "mlx-batch-server-media/1",
                },
            ) as response:
                if response.status_code in _REDIRECT_STATUSES:
                    if redirects >= self._fetch_policy.max_redirects:
                        raise MediaResolverError(
                            "redirect_limit_exceeded",
                            "URL fetch exceeded its redirect limit",
                        )
                    location = response.headers.get("location")
                    if not location:
                        raise MediaResolverError(
                            "invalid_redirect",
                            "URL redirect is missing a location",
                        )
                    current_url = urljoin(str(response.url), location)
                    continue
                if response.status_code < 200 or response.status_code >= 300:
                    raise MediaResolverError(
                        "url_fetch_status",
                        f"URL fetch returned HTTP {response.status_code}",
                    )

                media_type = _response_media_type(response)
                if media_type not in accepted:
                    raise MediaResolverError(
                        "unsupported_media_type",
                        f"URL returned unsupported media type {media_type!r}",
                    )
                _validate_content_length(response, request.max_bytes)
                content = await self._read_bounded(response, request.max_bytes)
                return SourceBlob(
                    content=content,
                    media_type=media_type,
                    final_url=str(response.url),
                )

        raise AssertionError("redirect loop must terminate")

    async def _read_bounded(
        self,
        response: httpx.Response,
        max_bytes: int,
    ) -> bytes:
        content = bytearray()
        async for chunk in response.aiter_bytes(self._fetch_policy.chunk_bytes):
            content.extend(chunk)
            if len(content) > max_bytes:
                raise MediaResolverError(
                    "source_bytes_exceeded",
                    "URL response exceeds the remaining source byte budget",
                )
        if not content:
            raise MediaResolverError(
                "empty_source",
                "URL response body must not be empty",
            )
        return bytes(content)


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
) -> SourceMediaResolver:
    """Compose bounded image, text, and PDF ports without file-id ownership."""

    url_policy = AllowedUrlPolicy(frozenset(allowed_url_origins))
    image_materializer = PillowImageMaterializer()
    file_materializer = StandardFileMaterializer(
        image_materializer=image_materializer,
        pdf_materializer=pdf_materializer or PyMuPDFFileMaterializer(),
    )
    url_fetcher = (
        HttpxUrlFetcher(
            url_policy=url_policy,
            fetch_policy=fetch_policy,
            transport=transport,
        )
        if url_policy.origins
        else None
    )
    return SourceMediaResolver(
        limits=limits,
        url_policy=url_policy,
        url_fetcher=url_fetcher,
        file_id_resolver=file_id_resolver,
        image_materializer=image_materializer,
        file_materializer=file_materializer,
    )


def _response_media_type(response: httpx.Response) -> str:
    value = response.headers.get("content-type")
    if value is None:
        raise MediaResolverError(
            "missing_media_type",
            "URL response requires an explicit Content-Type",
        )
    return _normalize_media_type(value)


def _normalize_media_type(value: str) -> str:
    normalized = value.split(";", 1)[0].strip().lower()
    if not normalized or "/" not in normalized:
        raise MediaResolverError(
            "invalid_media_type",
            "media type must be explicit",
        )
    return normalized


def _validate_content_length(response: httpx.Response, max_bytes: int) -> None:
    value = response.headers.get("content-length")
    if value is None:
        return
    try:
        content_length = int(value)
    except ValueError as exc:
        raise MediaResolverError(
            "invalid_content_length",
            "URL response Content-Length must be an integer",
        ) from exc
    if content_length < 0:
        raise MediaResolverError(
            "invalid_content_length",
            "URL response Content-Length must not be negative",
        )
    if content_length > max_bytes:
        raise MediaResolverError(
            "source_bytes_exceeded",
            "URL response exceeds the remaining source byte budget",
        )


__all__ = [
    "HttpFetchPolicy",
    "HttpxUrlFetcher",
    "PillowImageMaterializer",
    "PyMuPDFFileMaterializer",
    "StandardFileMaterializer",
    "compose_source_media_resolver",
]
