"""Bounded PDF-to-text-and-page-image materialization for Qwen4Exp."""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import math
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from .media_resolver import (
    FileMaterialization,
    FileMaterializationRequest,
    MaterializedImage,
    MaterializedText,
    MediaResolverError,
)


@dataclass(frozen=True, slots=True)
class PdfRenderPolicy:
    """Target-owned mapping from Responses detail to bounded raster density."""

    low_dpi: int = 72
    auto_dpi: int = 144
    high_dpi: int = 200
    original_dpi: int = 300

    def __post_init__(self) -> None:
        if (
            min(
                self.low_dpi,
                self.auto_dpi,
                self.high_dpi,
                self.original_dpi,
            )
            < 1
        ):
            raise ValueError("PDF render DPI values must be positive")

    def dpi_for(self, detail: str) -> int:
        try:
            return {
                "low": self.low_dpi,
                "auto": self.auto_dpi,
                "high": self.high_dpi,
                "original": self.original_dpi,
            }[detail]
        except KeyError as exc:
            raise ValueError("PDF detail must be auto, low, high, or original") from exc


class PyMuPDFFileMaterializer:
    """Extract page text and rasterize every page without hidden truncation."""

    def __init__(self, policy: PdfRenderPolicy = PdfRenderPolicy()) -> None:
        self._policy = policy

    async def materialize(
        self,
        request: FileMaterializationRequest,
    ) -> FileMaterialization:
        if request.source.media_type != "application/pdf":
            raise MediaResolverError(
                "unsupported_file_type",
                "PyMuPDF materializer accepts only application/pdf",
                request.part_index,
            )
        return await asyncio.to_thread(self._materialize_sync, request)

    def _materialize_sync(
        self,
        request: FileMaterializationRequest,
    ) -> FileMaterialization:
        pymupdf = _load_pymupdf()
        document: Any | None = None
        try:
            document = pymupdf.open(
                stream=request.source.content,
                filetype="pdf",
            )
            return self._materialize_document(pymupdf, document, request)
        except MediaResolverError:
            raise
        except Exception as exc:
            raise MediaResolverError(
                "pdf_decode_failed",
                "PDF payload cannot be decoded safely",
                request.part_index,
            ) from exc
        finally:
            if document is not None:
                with contextlib.suppress(Exception):
                    document.close()

    def _materialize_document(
        self,
        pymupdf: ModuleType,
        document: Any,
        request: FileMaterializationRequest,
    ) -> FileMaterialization:
        if bool(getattr(document, "needs_pass", False)):
            raise MediaResolverError(
                "encrypted_pdf",
                "password-protected PDF input is not supported",
                request.part_index,
            )

        page_count = int(document.page_count)
        if page_count < 1:
            raise MediaResolverError(
                "empty_pdf",
                "PDF input must contain at least one page",
                request.part_index,
            )
        if page_count > request.max_images:
            raise MediaResolverError(
                "image_count_exceeded",
                "PDF page count exceeds the remaining image budget",
                request.part_index,
            )

        dpi = self._policy.dpi_for(request.detail)
        scale = dpi / 72.0
        matrix = pymupdf.Matrix(scale, scale)
        items: list[MaterializedImage | MaterializedText] = []
        image_bytes = 0
        text_bytes = 0
        total_pixels = 0

        for page_index in range(page_count):
            page = document.load_page(page_index)
            text = page.get_text("text")
            if not isinstance(text, str):
                raise MediaResolverError(
                    "pdf_text_decode_failed",
                    "PDF text extraction returned an invalid value",
                    request.part_index,
                )
            if text and not text.isspace():
                text_bytes += len(text.encode("utf-8"))
                if text_bytes > request.max_text_bytes:
                    raise MediaResolverError(
                        "text_bytes_exceeded",
                        "PDF text exceeds the remaining text byte budget",
                        request.part_index,
                    )
                items.append(MaterializedText(text=text))

            estimated_pixels = _estimated_page_pixels(page, scale)
            _validate_pixel_budget(
                estimated_pixels,
                total_pixels,
                request,
            )
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            width = int(pixmap.width)
            height = int(pixmap.height)
            pixels = width * height
            _validate_pixel_budget(pixels, total_pixels, request)

            content = pixmap.tobytes("png")
            if not isinstance(content, bytes) or not content:
                raise MediaResolverError(
                    "pdf_page_render_failed",
                    "PDF page renderer returned invalid PNG bytes",
                    request.part_index,
                )
            image_bytes += len(content)
            if image_bytes > request.max_image_bytes:
                raise MediaResolverError(
                    "image_bytes_exceeded",
                    "PDF page images exceed the remaining image byte budget",
                    request.part_index,
                )
            total_pixels += pixels
            items.append(
                MaterializedImage(
                    content=content,
                    media_type="image/png",
                    width=width,
                    height=height,
                )
            )

        return FileMaterialization(items=tuple(items))


def _estimated_page_pixels(page: Any, scale: float) -> int:
    width = max(1, math.ceil(float(page.rect.width) * scale))
    height = max(1, math.ceil(float(page.rect.height) * scale))
    return width * height


def _validate_pixel_budget(
    pixels: int,
    previous_pixels: int,
    request: FileMaterializationRequest,
) -> None:
    if pixels < 1:
        raise MediaResolverError(
            "pdf_page_render_failed",
            "PDF page dimensions must be positive",
            request.part_index,
        )
    if pixels > request.max_image_pixels:
        raise MediaResolverError(
            "image_pixels_exceeded",
            "one PDF page exceeds the per-image pixel budget",
            request.part_index,
        )
    if previous_pixels + pixels > request.max_total_pixels:
        raise MediaResolverError(
            "total_pixels_exceeded",
            "PDF page images exceed the remaining pixel budget",
            request.part_index,
        )


def _load_pymupdf() -> ModuleType:
    try:
        return importlib.import_module("pymupdf")
    except ModuleNotFoundError:
        try:
            return importlib.import_module("fitz")
        except ModuleNotFoundError as exc:
            raise MediaResolverError(
                "pdf_dependency_missing",
                "PDF input requires the PyMuPDF package",
            ) from exc


__all__ = ["PdfRenderPolicy", "PyMuPDFFileMaterializer"]
