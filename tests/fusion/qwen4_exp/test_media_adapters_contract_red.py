from __future__ import annotations

import base64
import socket

import httpx
import pytest

from mlx_batch_server.runtime.fusion.qwen4_exp.media_adapters import (
    HttpxUrlFetcher,
    PillowImageMaterializer,
    StandardFileMaterializer,
    compose_source_media_resolver,
)
from mlx_batch_server.runtime.fusion.qwen4_exp.media_resolver import (
    AllowedUrlPolicy,
    FileMaterialization,
    FileMaterializationRequest,
    ImageMaterializationRequest,
    MaterializedImage,
    MaterializedText,
    MediaResolverError,
    SourceBlob,
    UrlFetchRequest,
)
from mlx_batch_server.vision.input import FileInput, MultimodalInputPlan

_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "/x8AAusB9Wl2nWQAAAAASUVORK5CYII="
)
_PUBLIC_TEST_IP = "1.1.1.1"


def _public_addrinfo(host: str, port: int, *args: object, **kwargs: object):
    return [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", (_PUBLIC_TEST_IP, port or 0)),
    ]


def _url_request(url: str, *, max_bytes: int = 1024) -> UrlFetchRequest:
    return UrlFetchRequest(
        url=url,
        max_bytes=max_bytes,
        accepted_media_types=("image/png",),
    )


@pytest.mark.asyncio
async def test_url_fetcher_checks_every_redirect_origin() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            302,
            headers={"location": "https://foreign.example/image.png"},
            request=request,
        )

    fetcher = HttpxUrlFetcher(
        url_policy=AllowedUrlPolicy(frozenset({"https://media.example"})),
        transport=httpx.MockTransport(handler),
        getaddrinfo=_public_addrinfo,
    )

    with pytest.raises(MediaResolverError, match="origin boundary") as error:
        await fetcher.fetch(_url_request("https://media.example/start"))

    assert error.value.code == "redirect_not_allowed"


@pytest.mark.asyncio
async def test_url_fetcher_enforces_decoded_stream_budget() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "image/png"},
            content=_PNG,
            request=request,
        )

    fetcher = HttpxUrlFetcher(
        url_policy=AllowedUrlPolicy(frozenset({"https://media.example"})),
        transport=httpx.MockTransport(handler),
        getaddrinfo=_public_addrinfo,
    )

    with pytest.raises(MediaResolverError) as error:
        await fetcher.fetch(
            _url_request("https://media.example/image.png", max_bytes=4)
        )

    assert error.value.code == "source_bytes_exceeded"


@pytest.mark.asyncio
async def test_url_fetcher_returns_an_immutable_final_url_receipt() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/start":
            return httpx.Response(
                307,
                headers={"location": "/image.png"},
                request=request,
            )
        return httpx.Response(
            200,
            headers={"content-type": "image/png"},
            content=_PNG,
            request=request,
        )

    fetcher = HttpxUrlFetcher(
        url_policy=AllowedUrlPolicy(frozenset({"https://media.example"})),
        transport=httpx.MockTransport(handler),
        getaddrinfo=_public_addrinfo,
    )

    blob = await fetcher.fetch(_url_request("https://media.example/start"))

    assert blob.content == _PNG
    assert blob.media_type == "image/png"
    assert blob.final_url == "https://media.example/image.png"


@pytest.mark.asyncio
async def test_url_fetcher_allows_public_https_without_allowlist() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "image/png"},
            content=_PNG,
            request=request,
        )

    fetcher = HttpxUrlFetcher(
        url_policy=AllowedUrlPolicy(),
        transport=httpx.MockTransport(handler),
        getaddrinfo=_public_addrinfo,
    )

    blob = await fetcher.fetch(_url_request("https://cdn.example/image.png"))

    assert blob.content == _PNG
    assert blob.final_url == "https://cdn.example/image.png"


@pytest.mark.asyncio
async def test_image_materializer_rejects_declared_type_spoofing() -> None:
    materializer = PillowImageMaterializer()
    request = ImageMaterializationRequest(
        part_index=7,
        source=SourceBlob(content=_PNG, media_type="image/jpeg"),
        max_bytes=1024,
        max_pixels=100,
    )

    with pytest.raises(MediaResolverError) as error:
        await materializer.materialize(request)

    assert error.value.code == "image_media_type_mismatch"
    assert error.value.part_index == 7


@pytest.mark.asyncio
async def test_standard_file_materializer_preserves_utf8_text() -> None:
    materializer = StandardFileMaterializer(
        image_materializer=PillowImageMaterializer()
    )
    request = FileMaterializationRequest(
        part_index=3,
        filename="note.txt",
        source=SourceBlob(
            content="za\u017c\u00f3\u0142\u0107".encode(),
            media_type="text/plain",
        ),
        max_images=8,
        max_image_bytes=1024,
        max_text_bytes=1024,
        max_image_pixels=100,
        max_total_pixels=100,
    )

    result = await materializer.materialize(request)

    assert isinstance(result, FileMaterialization)
    assert isinstance(result.items[0], MaterializedText)
    assert result.items[0].text == "za\u017c\u00f3\u0142\u0107"


@pytest.mark.asyncio
async def test_standard_file_materializer_requires_explicit_pdf_port() -> None:
    materializer = StandardFileMaterializer(
        image_materializer=PillowImageMaterializer()
    )
    request = FileMaterializationRequest(
        part_index=4,
        filename="report.pdf",
        source=SourceBlob(content=b"%PDF-1.7", media_type="application/pdf"),
        max_images=8,
        max_image_bytes=1024,
        max_text_bytes=1024,
        max_image_pixels=100,
        max_total_pixels=100,
    )

    with pytest.raises(MediaResolverError) as error:
        await materializer.materialize(request)

    assert error.value.code == "pdf_materializer_required"


@pytest.mark.asyncio
async def test_composition_does_not_invent_file_id_ownership() -> None:
    resolver = compose_source_media_resolver(
        allowed_url_origins=("https://media.example",)
    )
    plan = MultimodalInputPlan(
        prompt=(),
        media=(FileInput(part_index=0, file_id="file_missing"),),
    )

    with pytest.raises(MediaResolverError) as error:
        await resolver.resolve(plan)

    assert error.value.code == "file_id_resolver_required"


@pytest.mark.asyncio
async def test_file_materializer_delegates_images_through_same_validator() -> None:
    materializer = StandardFileMaterializer(
        image_materializer=PillowImageMaterializer()
    )
    request = FileMaterializationRequest(
        part_index=5,
        filename="pixel.png",
        source=SourceBlob(content=_PNG, media_type="image/png"),
        max_images=8,
        max_image_bytes=1024,
        max_text_bytes=1024,
        max_image_pixels=100,
        max_total_pixels=100,
    )

    result = await materializer.materialize(request)

    assert isinstance(result.items[0], MaterializedImage)
    assert result.items[0].width == 1
    assert result.items[0].height == 1
