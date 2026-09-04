from __future__ import annotations

from types import SimpleNamespace

import pytest

from mlx_batch_server.runtime.fusion.qwen4_exp import pdf_materializer as subject
from mlx_batch_server.runtime.fusion.qwen4_exp.media_resolver import (
    FileMaterializationRequest,
    MaterializedImage,
    MaterializedText,
    MediaResolverError,
    SourceBlob,
)
from mlx_batch_server.runtime.fusion.qwen4_exp.pdf_materializer import (
    PdfRenderPolicy,
    PyMuPDFFileMaterializer,
)


class _Pixmap:
    def __init__(self, width: int, height: int, content: bytes) -> None:
        self.width = width
        self.height = height
        self._content = content

    def tobytes(self, output: str) -> bytes:
        assert output == "png"
        return self._content


class _Page:
    def __init__(self, text: str, content: bytes) -> None:
        self._text = text
        self._content = content
        self.rect = SimpleNamespace(width=72, height=36)
        self.matrices: list[tuple[float, float]] = []

    def get_text(self, kind: str) -> str:
        assert kind == "text"
        return self._text

    def get_pixmap(
        self,
        *,
        matrix: tuple[float, float],
        alpha: bool,
    ) -> _Pixmap:
        assert alpha is False
        self.matrices.append(matrix)
        return _Pixmap(200, 100, self._content)


class _Document:
    def __init__(self, pages: list[_Page], *, needs_pass: bool = False) -> None:
        self._pages = pages
        self.needs_pass = needs_pass
        self.closed = False

    @property
    def page_count(self) -> int:
        return len(self._pages)

    def load_page(self, index: int) -> _Page:
        return self._pages[index]

    def close(self) -> None:
        self.closed = True


def _request(*, detail: str = "high", max_images: int = 8):
    return FileMaterializationRequest(
        part_index=4,
        filename="brief.pdf",
        source=SourceBlob(
            content=b"%PDF-1.7 test",
            media_type="application/pdf",
        ),
        max_images=max_images,
        max_image_bytes=1024,
        max_text_bytes=1024,
        max_image_pixels=100_000,
        max_total_pixels=200_000,
        detail=detail,
    )


def _module(document: _Document):
    return SimpleNamespace(
        Matrix=lambda x, y: (x, y),
        open=lambda **kwargs: document,
    )


@pytest.mark.asyncio
async def test_pdf_emits_text_then_png_for_each_page_in_order(monkeypatch) -> None:
    pages = [_Page("page one\n", b"png-1"), _Page("", b"png-2")]
    document = _Document(pages)
    monkeypatch.setattr(subject, "_load_pymupdf", lambda: _module(document))

    result = await PyMuPDFFileMaterializer().materialize(_request())

    assert [type(item) for item in result.items] == [
        MaterializedText,
        MaterializedImage,
        MaterializedImage,
    ]
    assert result.items[0].text == "page one\n"
    assert [item.content for item in result.items[1:]] == [b"png-1", b"png-2"]
    assert pages[0].matrices == [(200 / 72, 200 / 72)]
    assert document.closed


@pytest.mark.asyncio
async def test_pdf_page_count_fails_before_any_page_is_loaded(monkeypatch) -> None:
    document = _Document([_Page("", b"png")] * 2)
    monkeypatch.setattr(subject, "_load_pymupdf", lambda: _module(document))

    with pytest.raises(MediaResolverError) as error:
        await PyMuPDFFileMaterializer().materialize(_request(max_images=1))

    assert error.value.code == "image_count_exceeded"
    assert all(page.matrices == [] for page in document._pages)
    assert document.closed


@pytest.mark.asyncio
async def test_encrypted_pdf_fails_closed(monkeypatch) -> None:
    document = _Document([_Page("", b"png")], needs_pass=True)
    monkeypatch.setattr(subject, "_load_pymupdf", lambda: _module(document))

    with pytest.raises(MediaResolverError) as error:
        await PyMuPDFFileMaterializer().materialize(_request())

    assert error.value.code == "encrypted_pdf"
    assert document.closed


def test_pdf_render_policy_rejects_unknown_detail() -> None:
    with pytest.raises(ValueError, match="detail"):
        PdfRenderPolicy().dpi_for("ultra")
