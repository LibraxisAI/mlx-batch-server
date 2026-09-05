"""RED contracts for bounded 3more source-media resolution.

These tests are intentionally authored but not executed during Compile Embargo
HOLD. They contain no MLX or donor imports.
"""

from dataclasses import FrozenInstanceError

import pytest

from mlx_batch_server.runtime.fusion.qwen4_exp.media_resolver import (
    AllowedUrlPolicy,
    FileIdResolutionRequest,
    FileMaterialization,
    FileMaterializationRequest,
    ImageMaterializationRequest,
    MaterializedImage,
    MaterializedText,
    MediaResolverError,
    MediaResolverLimits,
    ResolvedImage,
    ResolvedText,
    SourceBlob,
    SourceMediaResolver,
    UrlFetchRequest,
    source_identity_digest,
)
from mlx_batch_server.vision.input import (
    AudioInput,
    FileInput,
    ImageInput,
    MultimodalInputPlan,
    PromptText,
    VideoInput,
)

PNG_A = "data:image/png;base64,aW1hZ2UtYQ=="
PNG_B = "data:image/png;base64,aW1hZ2UtYg=="
PDF = "data:application/pdf;base64,cGRm"


class RecordingImageMaterializer:
    def __init__(self) -> None:
        self.requests: list[ImageMaterializationRequest] = []

    async def materialize(
        self,
        request: ImageMaterializationRequest,
    ) -> MaterializedImage:
        self.requests.append(request)
        return MaterializedImage(
            content=request.source.content,
            media_type=request.source.media_type,
            width=32,
            height=24,
        )


class RecordingUrlFetcher:
    def __init__(self, *, final_url: str | None = None) -> None:
        self.requests: list[UrlFetchRequest] = []
        self.final_url = final_url

    async def fetch(self, request: UrlFetchRequest) -> SourceBlob:
        self.requests.append(request)
        return SourceBlob(
            content=b"remote-image",
            media_type="image/png",
            final_url=self.final_url or request.url,
        )


class RecordingFileIdResolver:
    def __init__(self, blob: SourceBlob | None = None) -> None:
        self.requests: list[FileIdResolutionRequest] = []
        self.blob = blob or SourceBlob(
            content=b"pdf",
            media_type="application/pdf",
        )

    async def resolve(self, request: FileIdResolutionRequest) -> SourceBlob:
        self.requests.append(request)
        return self.blob


class RecordingFileMaterializer:
    def __init__(self, items: tuple[object, ...]) -> None:
        self.requests: list[FileMaterializationRequest] = []
        self.items = items

    async def materialize(
        self,
        request: FileMaterializationRequest,
    ) -> FileMaterialization:
        self.requests.append(request)
        return FileMaterialization(items=self.items)


def test_detail_is_part_of_canonical_source_identity() -> None:
    source = "data:application/pdf;base64,cGRm"

    assert source_identity_digest(FileInput(part_index=0, file_data=source)) != (
        source_identity_digest(FileInput(part_index=0, file_data=source, detail="high"))
    )


def _plan(*media: object) -> MultimodalInputPlan:
    return MultimodalInputPlan(
        prompt=(PromptText(part_index=0, text="Compare the evidence."),),
        media=tuple(media),
    )


@pytest.mark.asyncio
async def test_data_urls_preserve_order_and_have_deterministic_digests() -> None:
    materializer_a = RecordingImageMaterializer()
    materializer_b = RecordingImageMaterializer()
    plan = _plan(
        ImageInput(part_index=2, image_base64=PNG_B),
        ImageInput(part_index=1, image_url=PNG_A),
    )

    first = await SourceMediaResolver(image_materializer=materializer_a).resolve(plan)
    second = await SourceMediaResolver(image_materializer=materializer_b).resolve(plan)

    assert tuple(item.part_index for item in first.items) == (1, 2)
    assert first.digest == second.digest
    assert first.images[0].content_digest != first.images[1].content_digest
    assert first.source_bytes == len(b"image-a") + len(b"image-b")
    assert first.total_pixels == 2 * 32 * 24
    assert tuple(request.part_index for request in materializer_a.requests) == (1, 2)


@pytest.mark.asyncio
async def test_ninth_direct_image_fails_before_any_materializer_call() -> None:
    materializer = RecordingImageMaterializer()
    plan = _plan(
        *(ImageInput(part_index=index, image_url=PNG_A) for index in range(1, 10))
    )

    with pytest.raises(MediaResolverError) as caught:
        await SourceMediaResolver(image_materializer=materializer).resolve(plan)

    assert caught.value.code == "image_count_exceeded"
    assert materializer.requests == []


@pytest.mark.asyncio
async def test_file_id_expands_to_ordered_images_and_text_via_injected_ports() -> None:
    file_resolver = RecordingFileIdResolver()
    file_materializer = RecordingFileMaterializer(
        (
            MaterializedImage(b"page-1", "image/png", 100, 80),
            MaterializedText("clinical note"),
            MaterializedImage(b"page-2", "image/png", 120, 90),
        )
    )
    resolver = SourceMediaResolver(
        file_id_resolver=file_resolver,
        file_materializer=file_materializer,
    )

    bundle = await resolver.resolve(
        _plan(FileInput(part_index=1, file_id="file_case", filename="case.pdf"))
    )

    assert tuple(type(item) for item in bundle.items) == (
        ResolvedImage,
        ResolvedText,
        ResolvedImage,
    )
    assert tuple(item.item_index for item in bundle.items) == (0, 1, 2)
    assert tuple(image.content for image in bundle.images) == (b"page-1", b"page-2")
    assert tuple(text.text for text in bundle.texts) == ("clinical note",)
    assert file_resolver.requests[0].file_id == "file_case"
    assert file_materializer.requests[0].filename == "case.pdf"
    assert file_materializer.requests[0].max_images == 16
    assert file_materializer.requests[0].max_text_bytes == 16 * 1024 * 1024


@pytest.mark.asyncio
async def test_image_file_id_uses_the_injected_resolver_and_image_materializer() -> (
    None
):
    file_resolver = RecordingFileIdResolver(
        SourceBlob(content=b"image", media_type="image/png")
    )
    image_materializer = RecordingImageMaterializer()
    resolver = SourceMediaResolver(
        file_id_resolver=file_resolver,
        image_materializer=image_materializer,
    )

    bundle = await resolver.resolve(
        _plan(ImageInput(part_index=2, file_id="file_image"))
    )

    assert tuple(image.content for image in bundle.images) == (b"image",)
    assert file_resolver.requests[0].file_id == "file_image"
    assert file_resolver.requests[0].accepted_media_types == (
        "image/gif",
        "image/jpeg",
        "image/png",
        "image/webp",
    )


@pytest.mark.asyncio
async def test_pdf_has_no_builtin_rasterization_or_hidden_file_io() -> None:
    with pytest.raises(MediaResolverError) as caught:
        await SourceMediaResolver().resolve(
            _plan(FileInput(part_index=1, file_data=PDF, filename="case.pdf"))
        )

    assert caught.value.code == "file_materializer_required"


@pytest.mark.asyncio
async def test_remote_url_succeeds_without_allowlist_and_keeps_optional_lockdown() -> (
    None
):
    image = ImageInput(
        part_index=1,
        image_url="https://media.3more.test/cases/a.png",
    )
    materializer = RecordingImageMaterializer()

    with pytest.raises(MediaResolverError) as missing_fetcher:
        await SourceMediaResolver(image_materializer=materializer).resolve(_plan(image))
    assert missing_fetcher.value.code == "url_fetcher_required"

    fetcher = RecordingUrlFetcher()
    bundle = await SourceMediaResolver(
        url_fetcher=fetcher,
        image_materializer=materializer,
    ).resolve(_plan(image))
    assert bundle.images[0].content == b"remote-image"
    assert fetcher.requests[0].accepted_media_types == (
        "image/gif",
        "image/jpeg",
        "image/png",
        "image/webp",
    )

    lockdown = AllowedUrlPolicy(frozenset({"https://media.3more.test"}))
    with pytest.raises(MediaResolverError) as denied:
        await SourceMediaResolver(
            url_policy=lockdown,
            url_fetcher=RecordingUrlFetcher(),
            image_materializer=materializer,
        ).resolve(
            _plan(
                ImageInput(
                    part_index=1,
                    image_url="https://elsewhere.test/cases/a.png",
                )
            )
        )
    assert denied.value.code == "url_not_allowed"


@pytest.mark.asyncio
async def test_redirect_to_unapproved_origin_fails_closed() -> None:
    fetcher = RecordingUrlFetcher(final_url="https://elsewhere.test/a.png")
    resolver = SourceMediaResolver(
        url_policy=AllowedUrlPolicy(frozenset({"https://media.3more.test"})),
        url_fetcher=fetcher,
        image_materializer=RecordingImageMaterializer(),
    )

    with pytest.raises(MediaResolverError) as caught:
        await resolver.resolve(
            _plan(
                ImageInput(
                    part_index=1,
                    image_url="https://media.3more.test/a.png",
                )
            )
        )

    assert caught.value.code == "redirect_not_allowed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "descriptor",
    (
        AudioInput(part_index=1, audio_url="https://media.3more.test/a.wav"),
        VideoInput(part_index=1, video_url="https://media.3more.test/a.mov"),
    ),
)
async def test_audio_and_video_fail_before_external_io(descriptor: object) -> None:
    fetcher = RecordingUrlFetcher()
    resolver = SourceMediaResolver(
        url_policy=AllowedUrlPolicy(frozenset({"https://media.3more.test"})),
        url_fetcher=fetcher,
    )

    with pytest.raises(MediaResolverError) as caught:
        await resolver.resolve(_plan(descriptor))

    assert caught.value.code == "unsupported_media"
    assert fetcher.requests == []


@pytest.mark.asyncio
async def test_ambiguous_source_fails_before_external_io() -> None:
    materializer = RecordingImageMaterializer()
    resolver = SourceMediaResolver(image_materializer=materializer)

    with pytest.raises(MediaResolverError) as caught:
        await resolver.resolve(
            _plan(
                ImageInput(
                    part_index=1,
                    image_url=PNG_A,
                    image_base64=PNG_B,
                )
            )
        )

    assert caught.value.code == "ambiguous_source"
    assert materializer.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("limits", "image", "expected_code"),
    (
        (
            MediaResolverLimits(max_source_bytes=2),
            MaterializedImage(b"ok", "image/png", 1, 1),
            "source_bytes_exceeded",
        ),
        (
            MediaResolverLimits(max_materialized_image_bytes=3),
            MaterializedImage(b"four", "image/png", 1, 1),
            "image_bytes_exceeded",
        ),
        (
            MediaResolverLimits(max_image_pixels=99),
            MaterializedImage(b"ok", "image/png", 10, 10),
            "image_pixels_exceeded",
        ),
        (
            MediaResolverLimits(max_total_pixels=99),
            MaterializedImage(b"ok", "image/png", 10, 10),
            "total_pixels_exceeded",
        ),
    ),
)
async def test_byte_and_pixel_budgets_fail_closed(
    limits: MediaResolverLimits,
    image: MaterializedImage,
    expected_code: str,
) -> None:
    file_materializer = RecordingFileMaterializer((image,))
    resolver = SourceMediaResolver(
        limits=limits,
        file_materializer=file_materializer,
    )

    with pytest.raises(MediaResolverError) as caught:
        await resolver.resolve(
            _plan(FileInput(part_index=1, file_data=PDF, filename="case.pdf"))
        )

    assert caught.value.code == expected_code


@pytest.mark.asyncio
async def test_file_text_expansion_is_bounded_by_utf8_bytes() -> None:
    resolver = SourceMediaResolver(
        limits=MediaResolverLimits(max_materialized_text_bytes=3),
        file_materializer=RecordingFileMaterializer((MaterializedText("zloty"),)),
    )

    with pytest.raises(MediaResolverError) as caught:
        await resolver.resolve(
            _plan(FileInput(part_index=1, file_data=PDF, filename="case.pdf"))
        )

    assert caught.value.code == "text_bytes_exceeded"


@pytest.mark.asyncio
async def test_eight_direct_images_leave_a_bounded_eight_page_pdf_budget() -> None:
    pages = tuple(
        MaterializedImage(f"page-{index}".encode(), "image/png", 1, 1)
        for index in range(8)
    )
    file_materializer = RecordingFileMaterializer(pages)
    resolver = SourceMediaResolver(
        image_materializer=RecordingImageMaterializer(),
        file_materializer=file_materializer,
    )

    bundle = await resolver.resolve(
        _plan(
            *(ImageInput(part_index=index + 1, image_url=PNG_A) for index in range(8)),
            FileInput(part_index=9, file_data=PDF, filename="eight-pages.pdf"),
        )
    )

    assert len(bundle.images) == 16
    assert file_materializer.requests[0].max_images == 8


@pytest.mark.asyncio
async def test_file_expansion_cannot_bypass_materialized_image_limit() -> None:
    pages = tuple(
        MaterializedImage(f"page-{index}".encode(), "image/png", 1, 1)
        for index in range(9)
    )
    resolver = SourceMediaResolver(
        image_materializer=RecordingImageMaterializer(),
        file_materializer=RecordingFileMaterializer(pages),
    )

    with pytest.raises(MediaResolverError) as caught:
        await resolver.resolve(
            _plan(
                *(
                    ImageInput(part_index=index + 1, image_url=PNG_A)
                    for index in range(8)
                ),
                FileInput(part_index=9, file_data=PDF, filename="nine-pages.pdf"),
            )
        )

    assert caught.value.code == "image_count_exceeded"


@pytest.mark.asyncio
async def test_resolved_bundle_and_nested_outputs_are_immutable() -> None:
    bundle = await SourceMediaResolver(
        image_materializer=RecordingImageMaterializer()
    ).resolve(_plan(ImageInput(part_index=1, image_url=PNG_A)))

    with pytest.raises(FrozenInstanceError):
        bundle.digest = "changed"
    with pytest.raises(FrozenInstanceError):
        bundle.images[0].width = 99
