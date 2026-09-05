"""RED contracts for whole-request Qwen4Exp media handoff."""

from __future__ import annotations

import hashlib

import pytest

from mlx_batch_server.runtime.contracts import BackendKind, RuntimeKey
from mlx_batch_server.runtime.fusion.qwen4_exp.media import (
    PreparedMediaItem,
    PreparedQwen4Message,
    PreparedTextItem,
    Qwen4PromptBuilder,
    Qwen4PromptError,
    bind_prepared_messages,
    render_function_call_output_text,
)
from mlx_batch_server.runtime.fusion.qwen4_exp.media_resolver import (
    ResolvedImage,
    ResolvedMediaBundle,
    ResolvedText,
    source_identity_digest,
)
from mlx_batch_server.vision.input import (
    FileInput,
    ImageInput,
    MultimodalInputPlan,
    PromptText,
)

RUNTIME = RuntimeKey(
    model_id="grant-ai/Qwen3.8-Flash-Next-Abliterated-MLX-4bit",
    revision="000544f8cddcbde27c1bc302deac2b5b4d45a5b1",
    backend=BackendKind.FUSED_MTP_MLX,
)


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _image(descriptor: ImageInput, item_index: int = 0) -> ResolvedImage:
    content = f"image-{descriptor.part_index}-{item_index}".encode()
    return ResolvedImage(
        part_index=descriptor.part_index,
        item_index=item_index,
        media_type="image/png",
        content=content,
        width=64,
        height=64,
        source_digest=source_identity_digest(descriptor),
        content_digest=_digest(content),
    )


def _file_image(descriptor: FileInput, item_index: int) -> ResolvedImage:
    content = f"page-{descriptor.part_index}-{item_index}".encode()
    return ResolvedImage(
        part_index=descriptor.part_index,
        item_index=item_index,
        media_type="image/png",
        content=content,
        width=80,
        height=60,
        source_digest=source_identity_digest(descriptor),
        content_digest=_digest(content),
    )


def _text(descriptor: FileInput, item_index: int, value: str) -> ResolvedText:
    return ResolvedText(
        part_index=descriptor.part_index,
        item_index=item_index,
        text=value,
        source_digest=source_identity_digest(descriptor),
        content_digest=_digest(value.encode()),
    )


def _bundle(*items: ResolvedImage | ResolvedText) -> ResolvedMediaBundle:
    images = tuple(item for item in items if isinstance(item, ResolvedImage))
    texts = tuple(item for item in items if isinstance(item, ResolvedText))
    return ResolvedMediaBundle(
        items=tuple(items),
        images=images,
        texts=texts,
        source_bytes=100,
        materialized_image_bytes=sum(len(item.content) for item in images),
        materialized_text_bytes=sum(len(item.text.encode()) for item in texts),
        total_pixels=sum(item.width * item.height for item in images),
        digest=_digest(
            b"bundle" + b"".join(item.content_digest.encode() for item in items)
        ),
    )


def test_builder_joins_one_whole_bundle_without_per_item_resolution() -> None:
    first = ImageInput(part_index=1, image_url="https://media.3more.ai/a.png")
    document = FileInput(part_index=2, file_id="file_lab")
    last = ImageInput(part_index=4, image_url="https://media.3more.ai/b.png")
    plan = MultimodalInputPlan(
        prompt=(PromptText(part_index=0, text="inspect"),),
        media=(first, document, last),
    )
    bundle = _bundle(
        _image(first),
        _text(document, 0, "page one"),
        _file_image(document, 1),
        _image(last),
    )

    prepared = Qwen4PromptBuilder().build(
        response_id="resp_3more",
        runtime=RUNTIME,
        plan=plan,
        resolution=bundle,
    )

    assert prepared.resolution is bundle
    assert [item.part_index for item in prepared.items] == [0, 1, 2, 2, 4]
    assert isinstance(prepared.items[0], PreparedTextItem)
    assert all(isinstance(item, PreparedMediaItem) for item in prepared.items[1:])
    assert [item.item_index for item in prepared.media] == [0, 0, 1, 0]


def test_direct_image_must_resolve_to_one_image() -> None:
    descriptor = ImageInput(part_index=0, image_url="https://media.3more.ai/a.png")
    plan = MultimodalInputPlan(prompt=(), media=(descriptor,))
    wrong = ResolvedText(
        part_index=0,
        item_index=0,
        text="not an image",
        source_digest=source_identity_digest(descriptor),
        content_digest=_digest(b"not an image"),
    )

    with pytest.raises(Qwen4PromptError, match="exactly one image"):
        Qwen4PromptBuilder().build(
            response_id="resp_3more",
            runtime=RUNTIME,
            plan=plan,
            resolution=_bundle(wrong),
        )


def test_file_expansion_indices_must_be_contiguous() -> None:
    descriptor = FileInput(part_index=0, file_id="file_pdf")
    plan = MultimodalInputPlan(prompt=(), media=(descriptor,))

    with pytest.raises(Qwen4PromptError, match="contiguous"):
        Qwen4PromptBuilder().build(
            response_id="resp_3more",
            runtime=RUNTIME,
            plan=plan,
            resolution=_bundle(_text(descriptor, 1, "page two")),
        )


def test_bundle_cannot_substitute_a_different_source() -> None:
    expected = ImageInput(part_index=0, image_url="https://media.3more.ai/a.png")
    foreign = ImageInput(part_index=0, image_url="https://media.3more.ai/b.png")
    plan = MultimodalInputPlan(prompt=(), media=(expected,))

    with pytest.raises(Qwen4PromptError, match="requested source"):
        Qwen4PromptBuilder().build(
            response_id="resp_3more",
            runtime=RUNTIME,
            plan=plan,
            resolution=_bundle(_image(foreign)),
        )


def test_bundle_must_cover_every_requested_part_and_no_foreign_part() -> None:
    first = ImageInput(part_index=0, image_url="https://media.3more.ai/a.png")
    second = ImageInput(part_index=1, image_url="https://media.3more.ai/b.png")
    plan = MultimodalInputPlan(prompt=(), media=(first, second))

    with pytest.raises(Qwen4PromptError, match="omitted"):
        Qwen4PromptBuilder().build(
            response_id="resp_3more",
            runtime=RUNTIME,
            plan=plan,
            resolution=_bundle(_image(first)),
        )

    foreign = ImageInput(part_index=2, image_url="https://media.3more.ai/c.png")
    with pytest.raises(Qwen4PromptError, match="unrequested"):
        Qwen4PromptBuilder().build(
            response_id="resp_3more",
            runtime=RUNTIME,
            plan=plan,
            resolution=_bundle(_image(first), _image(second), _image(foreign)),
        )


def test_bundle_item_order_is_canonical_and_digest_is_bounded() -> None:
    first = ImageInput(part_index=0, image_url="https://media.3more.ai/a.png")
    second = ImageInput(part_index=1, image_url="https://media.3more.ai/b.png")
    plan = MultimodalInputPlan(prompt=(), media=(first, second))
    builder = Qwen4PromptBuilder()
    prepared = builder.build(
        response_id="resp_3more",
        runtime=RUNTIME,
        plan=plan,
        resolution=_bundle(_image(first), _image(second)),
    )

    with pytest.raises(Qwen4PromptError, match="canonical order"):
        builder.build(
            response_id="resp_3more",
            runtime=RUNTIME,
            plan=plan,
            resolution=_bundle(_image(second), _image(first)),
        )

    assert len(prepared.content_digest) == 64
    assert len(prepared.media_digest) == 64


def test_message_digest_changes_with_is_error_and_rejects_tampering() -> None:
    plan = MultimodalInputPlan(
        prompt=(PromptText(part_index=0, text="boom"),),
        media=(),
    )
    prepared = Qwen4PromptBuilder().build(
        response_id="resp_tool_digest",
        runtime=RUNTIME,
        plan=plan,
        resolution=_bundle(),
    )
    success = PreparedQwen4Message(
        message_index=0,
        role="tool",
        items=(prepared.items[0],),
        item_type="function_call_output",
        call_id="toolu_1",
        output="boom",
        is_error=False,
    )
    error = PreparedQwen4Message(
        message_index=0,
        role="tool",
        items=(prepared.items[0],),
        item_type="function_call_output",
        call_id="toolu_1",
        output="boom",
        is_error=True,
    )
    success_prompt = bind_prepared_messages(prepared, (success,))
    error_prompt = bind_prepared_messages(prepared, (error,))
    assert success_prompt.content_digest != error_prompt.content_digest
    assert render_function_call_output_text("boom", False) == "boom"
    assert render_function_call_output_text("boom", True) == (
        '{"is_error":true,"content":"boom"}'
    )

    tampered = PreparedQwen4Message(
        message_index=0,
        role="tool",
        items=(prepared.items[0],),
        item_type="function_call_output",
        call_id="toolu_other",
        output="boom",
        is_error=False,
    )
    tampered_prompt = bind_prepared_messages(prepared, (tampered,))
    assert tampered_prompt.content_digest != success_prompt.content_digest

    with pytest.raises(ValueError, match="does not seal message layout"):
        success_prompt.__class__(
            response_id=success_prompt.response_id,
            runtime=success_prompt.runtime,
            items=success_prompt.items,
            media=success_prompt.media,
            resolution=success_prompt.resolution,
            content_digest=error_prompt.content_digest,
            media_digest=success_prompt.media_digest,
            messages=success_prompt.messages,
        )
