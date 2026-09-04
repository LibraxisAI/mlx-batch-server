"""RED contracts for the target-owned multimodal input boundary."""

from dataclasses import FrozenInstanceError

import pytest

from mlx_batch_server.vision.input import (
    AudioInput,
    FileInput,
    ImageInput,
    MediaSourceField,
    MultimodalInputCapabilities,
    MultimodalInputError,
    PromptText,
    VideoInput,
    plan_multimodal_input,
)

ALL_SOURCES = MultimodalInputCapabilities(
    accepted_sources=frozenset(MediaSourceField),
)


def test_plan_separates_prompt_from_multiple_lossless_media_inputs() -> None:
    file_data = "data:application/pdf;base64,JVBERi0xLjQK"
    parts = [
        {"type": "input_text", "text": "Compare both scans with the report."},
        {
            "type": "input_image",
            "image_url": "https://media.3more.test/scan-a.png",
            "detail": "high",
        },
        {
            "type": "input_image",
            "image_base64": "data:image/png;base64,iVBORw0KGgo=",
            "detail": "original",
        },
        {
            "type": "input_file",
            "file_id": "file_lab_a",
            "filename": "lab-a.pdf",
            "detail": "high",
        },
        {
            "type": "input_file",
            "file_data": file_data,
            "filename": "lab-b.pdf",
        },
    ]

    plan = plan_multimodal_input(parts, ALL_SOURCES)

    assert plan.prompt == (
        PromptText(part_index=0, text="Compare both scans with the report."),
    )
    assert plan.media == (
        ImageInput(
            part_index=1,
            image_url="https://media.3more.test/scan-a.png",
            detail="high",
        ),
        ImageInput(
            part_index=2,
            image_base64="data:image/png;base64,iVBORw0KGgo=",
            detail="original",
        ),
        FileInput(
            part_index=3,
            file_id="file_lab_a",
            filename="lab-a.pdf",
            detail="high",
        ),
        FileInput(
            part_index=4,
            file_data=file_data,
            filename="lab-b.pdf",
        ),
    )
    assert all(not isinstance(item, PromptText) for item in plan.media)


def test_all_normalized_media_shapes_remain_opaque_and_ordered() -> None:
    parts = [
        {"type": "input_audio", "audio_url": "file:///private/tmp/exam.wav"},
        {"type": "input_video", "video_url": "file:///private/tmp/exam.mov"},
        {
            "type": "input_file",
            "file_url": "https://media.3more.test/record.pdf",
            "filename": "record.pdf",
        },
    ]

    plan = plan_multimodal_input(parts, ALL_SOURCES)

    assert plan.media == (
        AudioInput(part_index=0, audio_url="file:///private/tmp/exam.wav"),
        VideoInput(part_index=1, video_url="file:///private/tmp/exam.mov"),
        FileInput(
            part_index=2,
            file_url="https://media.3more.test/record.pdf",
            filename="record.pdf",
        ),
    )


def test_descriptors_and_plan_are_immutable_snapshots() -> None:
    part = {"type": "input_image", "image_url": "https://example.test/a.png"}
    plan = plan_multimodal_input([part], ALL_SOURCES)
    part["image_url"] = "https://example.test/changed.png"

    assert plan.media == (
        ImageInput(part_index=0, image_url="https://example.test/a.png"),
    )
    with pytest.raises(FrozenInstanceError):
        plan.media[0].image_url = "https://example.test/mutated.png"


def test_default_capabilities_are_text_only_and_fail_closed() -> None:
    capabilities = MultimodalInputCapabilities()

    with pytest.raises(MultimodalInputError) as caught:
        plan_multimodal_input(
            [{"type": "input_image", "image_url": "https://example.test/a.png"}],
            capabilities,
        )

    assert caught.value.code == "unsupported_source"
    assert caught.value.part_index == 0
    assert caught.value.part_type == "input_image"


def test_capability_is_exact_per_source_field() -> None:
    capabilities = MultimodalInputCapabilities(
        accepted_sources=frozenset({MediaSourceField.FILE_URL}),
    )

    accepted = plan_multimodal_input(
        [{"type": "input_file", "file_url": "https://example.test/a.pdf"}],
        capabilities,
    )
    assert accepted.media == (
        FileInput(part_index=0, file_url="https://example.test/a.pdf"),
    )

    with pytest.raises(MultimodalInputError, match="does not accept") as caught:
        plan_multimodal_input(
            [{"type": "input_file", "file_id": "file_a"}],
            capabilities,
        )
    assert caught.value.code == "unsupported_source"


def test_input_image_file_id_is_preserved_for_an_injected_resolver() -> None:
    capabilities = MultimodalInputCapabilities(
        accepted_sources=frozenset({MediaSourceField.FILE_ID}),
    )

    plan = plan_multimodal_input(
        [{"type": "input_image", "file_id": "file_image_a", "detail": "high"}],
        capabilities,
    )

    assert plan.media == (
        ImageInput(part_index=0, file_id="file_image_a", detail="high"),
    )


@pytest.mark.parametrize(
    "part",
    [
        {"type": "input_image"},
        {"type": "input_file", "filename": "orphan.pdf"},
        {"type": "input_audio"},
        {"type": "input_video"},
    ],
)
def test_missing_media_source_is_rejected(part: dict[str, str]) -> None:
    with pytest.raises(MultimodalInputError) as caught:
        plan_multimodal_input([part], ALL_SOURCES)

    assert caught.value.code == "missing_source"


@pytest.mark.parametrize(
    "part",
    [
        {
            "type": "input_image",
            "image_url": "https://example.test/a.png",
            "image_base64": "data:image/png;base64,AAAA",
        },
        {
            "type": "input_file",
            "file_id": "file_a",
            "file_url": "https://example.test/a.pdf",
        },
        {
            "type": "input_file",
            "file_url": "https://example.test/a.pdf",
            "file_data": "data:application/pdf;base64,AAAA",
        },
    ],
)
def test_ambiguous_media_sources_are_rejected(part: dict[str, str]) -> None:
    with pytest.raises(MultimodalInputError) as caught:
        plan_multimodal_input([part], ALL_SOURCES)

    assert caught.value.code == "ambiguous_source"


@pytest.mark.parametrize(
    ("part", "code"),
    [
        ({"type": "input_image", "image_url": ""}, "invalid_source"),
        ({"type": "input_file", "file_data": 42}, "invalid_source"),
        (
            {"type": "input_file", "file_id": "file_a", "filename": 42},
            "invalid_metadata",
        ),
        ({"type": "input_text", "text": None}, "invalid_text"),
        (
            {
                "type": "input_file",
                "file_data": "data:application/pdf;base64,AAAA",
                "detail": "extreme",
            },
            "invalid_metadata",
        ),
        ({"type": "output_text", "text": "not input"}, "unsupported_part_type"),
    ],
)
def test_malformed_or_non_input_parts_fail_closed(
    part: dict[str, object],
    code: str,
) -> None:
    with pytest.raises(MultimodalInputError) as caught:
        plan_multimodal_input([part], ALL_SOURCES)

    assert caught.value.code == code
