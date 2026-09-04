"""RED contracts for per-request Qwen4Exp vision state.

These tests are authored but intentionally not executed while Compile Embargo
HOLD remains in force. They contain no tensor or donor imports.
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import FrozenInstanceError

import pytest

from mlx_batch_server.runtime.contracts import RuntimeKey
from mlx_batch_server.runtime.fusion.qwen4_exp.media_resolver import (
    ResolvedImage,
    ResolvedMediaBundle,
)
from mlx_batch_server.runtime.fusion.qwen4_exp.vision import (
    DeepstackFeatureReceipt,
    ImageGridReceipt,
    OpaqueRows,
    ProcessedVisionBatch,
    VisionContractError,
    VisionEmbeddingSlice,
    VisionPreprocessorPort,
    VisionProcessingRequest,
    VisionRequestIdentity,
    VisionSpliceCursor,
    VisionTowerOutput,
    VisionTowerPort,
    VisionTowerRequest,
    build_content_key_surrogates,
    build_image_spans,
    build_mrope_plan,
    build_vision_splice_plan,
    mrope_plan_digest,
    validate_preprocessing_output,
    validate_tower_output,
)

PAD = 99


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _bundle(image_count: int = 2) -> ResolvedMediaBundle:
    images = tuple(
        ResolvedImage(
            part_index=index + 1,
            item_index=0,
            media_type="image/png",
            content=f"image-{index}".encode(),
            width=64 + (index * 32),
            height=64,
            source_digest=_digest(f"source-{index}"),
            content_digest=_digest(f"content-{index}"),
        )
        for index in range(image_count)
    )
    return ResolvedMediaBundle(
        items=images,
        images=images,
        texts=(),
        source_bytes=sum(len(image.content) for image in images),
        materialized_image_bytes=sum(len(image.content) for image in images),
        materialized_text_bytes=0,
        total_pixels=sum(image.width * image.height for image in images),
        digest=_digest(f"bundle-{image_count}"),
    )


def _request(image_count: int = 2) -> VisionProcessingRequest:
    bundle = _bundle(image_count)
    identity = VisionRequestIdentity(
        response_id="resp_vision",
        runtime=RuntimeKey(model_id="grant-ai/Qwen3.8-Flash-Next"),
        bundle_digest=bundle.digest,
    )
    return VisionProcessingRequest(
        identity=identity,
        bundle=bundle,
        spatial_merge_size=2,
    )


def _processed(
    request: VisionProcessingRequest,
    grids: tuple[tuple[int, int, int], ...] | None = None,
) -> ProcessedVisionBatch:
    if grids is None:
        grids = tuple(
            (1, 4, 4 + (2 * index)) for index in range(len(request.bundle.images))
        )
    receipts = []
    patch_cursor = 0
    for index, (source, grid) in enumerate(
        zip(request.bundle.images, grids, strict=True)
    ):
        patch_rows = grid[0] * grid[1] * grid[2]
        receipts.append(
            ImageGridReceipt(
                identity=request.identity,
                image_index=index,
                part_index=source.part_index,
                item_index=source.item_index,
                content_digest=source.content_digest,
                grid_thw=grid,
                spatial_merge_size=2,
                patch_start=patch_cursor,
                patch_end=patch_cursor + patch_rows,
                pad_rows=patch_rows // 4,
            )
        )
        patch_cursor += patch_rows
    return ProcessedVisionBatch(
        identity=request.identity,
        images=tuple(receipts),
        pixel_values=OpaqueRows(handle=object(), row_count=patch_cursor),
    )


def _tower(processed: ProcessedVisionBatch) -> VisionTowerOutput:
    slices = []
    row_cursor = 0
    for image in processed.images:
        slices.append(
            VisionEmbeddingSlice(
                identity=processed.identity,
                image_index=image.image_index,
                content_digest=image.content_digest,
                start=row_cursor,
                end=row_cursor + image.pad_rows,
            )
        )
        row_cursor += image.pad_rows
    return VisionTowerOutput(
        identity=processed.identity,
        embeddings=OpaqueRows(handle=object(), row_count=row_cursor),
        image_slices=tuple(slices),
        deepstack=(
            DeepstackFeatureReceipt(
                identity=processed.identity,
                injection_index=0,
                rows=OpaqueRows(handle=object(), row_count=row_cursor),
            ),
        ),
    )


def _plan():
    request = _request()
    processed = _processed(request)
    tower = _tower(processed)
    prompt = (1, PAD, PAD, PAD, PAD, 2, PAD, PAD, PAD, PAD, PAD, PAD, 3)
    plan = build_vision_splice_plan(
        processed,
        tower,
        prompt_token_ids=prompt,
        image_pad_token_id=PAD,
    )
    assert plan is not None
    return request, processed, tower, prompt, plan


def test_processing_boundary_is_one_identity_bound_resolved_bundle() -> None:
    request = _request()

    assert request.bundle.images[0].part_index == 1
    assert request.bundle.images[1].part_index == 2
    assert request.identity.bundle_digest == request.bundle.digest
    assert tuple(VisionProcessingRequest.__dataclass_fields__) == (
        "identity",
        "bundle",
        "spatial_merge_size",
    )

    with pytest.raises(FrozenInstanceError):
        request.identity.response_id = "resp_foreign"  # type: ignore[misc]


def test_processing_rejects_ninth_image_and_foreign_bundle_identity() -> None:
    with pytest.raises(VisionContractError, match=r"1\.\.8 images"):
        _request(image_count=9)

    bundle = _bundle()
    identity = VisionRequestIdentity(
        response_id="resp_vision",
        runtime=RuntimeKey(model_id="model"),
        bundle_digest=_digest("different-bundle"),
    )
    with pytest.raises(VisionContractError) as caught:
        VisionProcessingRequest(identity, bundle, 2)
    assert caught.value.code == "bundle_identity_mismatch"


def test_grid_receipt_requires_divisible_grid_and_exact_pad_rows() -> None:
    request = _request(image_count=1)
    source = request.bundle.images[0]
    common = {
        "identity": request.identity,
        "image_index": 0,
        "part_index": source.part_index,
        "item_index": source.item_index,
        "content_digest": source.content_digest,
        "spatial_merge_size": 2,
        "patch_start": 0,
    }

    with pytest.raises(VisionContractError) as indivisible:
        ImageGridReceipt(
            **common,
            grid_thw=(1, 5, 4),
            patch_end=20,
            pad_rows=5,
        )
    assert indivisible.value.code == "grid_not_divisible"

    with pytest.raises(VisionContractError) as wrong_pads:
        ImageGridReceipt(
            **common,
            grid_thw=(1, 4, 4),
            patch_end=16,
            pad_rows=5,
        )
    assert wrong_pads.value.code == "pad_row_mismatch"


def test_preprocessor_validation_preserves_image_metadata_and_patch_rows() -> None:
    request = _request()
    processed = _processed(request)

    assert validate_preprocessing_output(request, processed) is processed
    assert processed.pixel_values.row_count == 16 + 24
    assert processed.total_pad_rows == 4 + 6

    first, second = processed.images
    foreign_second = ImageGridReceipt(
        identity=request.identity,
        image_index=1,
        part_index=second.part_index,
        item_index=second.item_index,
        content_digest=_digest("wrong-image"),
        grid_thw=second.grid_thw,
        spatial_merge_size=2,
        patch_start=second.patch_start,
        patch_end=second.patch_end,
        pad_rows=second.pad_rows,
    )
    wrong = ProcessedVisionBatch(
        identity=request.identity,
        images=(first, foreign_second),
        pixel_values=processed.pixel_values,
    )
    with pytest.raises(VisionContractError) as caught:
        validate_preprocessing_output(request, wrong)
    assert caught.value.code == "preprocessor_image_metadata_mismatch"

    with pytest.raises(VisionContractError) as row_error:
        ProcessedVisionBatch(
            identity=request.identity,
            images=processed.images,
            pixel_values=OpaqueRows(handle=object(), row_count=39),
        )
    assert row_error.value.code == "preprocessor_row_mismatch"


def test_protocols_are_structural_and_outputs_are_still_validated() -> None:
    class Preprocessor:
        def preprocess(self, request):
            return _processed(request)

    class Tower:
        def forward(self, request):
            return _tower(request.processed)

    assert isinstance(Preprocessor(), VisionPreprocessorPort)
    assert isinstance(Tower(), VisionTowerPort)

    request = _request()
    processed = Preprocessor().preprocess(request)
    tower_request = VisionTowerRequest(request.identity, processed)
    output = Tower().forward(tower_request)
    assert validate_tower_output(tower_request, output) is output


def test_tower_rejects_wrong_rows_with_self_consistent_total() -> None:
    request = _request()
    processed = _processed(request)
    output = VisionTowerOutput(
        identity=request.identity,
        embeddings=OpaqueRows(handle=object(), row_count=9),
        image_slices=(
            VisionEmbeddingSlice(
                request.identity,
                0,
                processed.images[0].content_digest,
                0,
                4,
            ),
            VisionEmbeddingSlice(
                request.identity,
                1,
                processed.images[1].content_digest,
                4,
                9,
            ),
        ),
    )

    with pytest.raises(VisionContractError) as caught:
        validate_tower_output(VisionTowerRequest(request.identity, processed), output)
    assert caught.value.code == "tower_image_row_mismatch"


def test_content_keys_and_spans_are_exact_and_fail_closed() -> None:
    request, processed, _tower_output, prompt, plan = _plan()

    assert tuple((span.start, span.end) for span in plan.image_spans) == (
        (1, 5),
        (6, 12),
    )
    assert plan.keyed_prompt_token_ids[0] == prompt[0]
    assert plan.keyed_prompt_token_ids[5] == prompt[5]
    assert all(
        token != PAD
        for token in plan.keyed_prompt_token_ids[1:5]
        + plan.keyed_prompt_token_ids[6:12]
    )

    digests = tuple(image.content_digest for image in processed.images)
    counts = tuple(image.pad_rows for image in processed.images)
    assert (
        build_content_key_surrogates(
            prompt,
            image_pad_token_id=PAD,
            image_digests=digests,
            pad_counts=counts,
        )
        == plan.keyed_prompt_token_ids
    )
    assert (
        build_content_key_surrogates(
            prompt,
            image_pad_token_id=PAD,
            image_digests=None,
            pad_counts=counts,
        )
        is None
    )
    assert (
        build_content_key_surrogates(
            prompt,
            image_pad_token_id=PAD,
            image_digests=digests[:1],
            pad_counts=counts,
        )
        is None
    )
    assert (
        build_image_spans(
            request.identity,
            (PAD, 1, PAD),
            image_pad_token_id=PAD,
            image_digests=(digests[0],),
            pad_counts=(2,),
        )
        is None
    )


def test_splice_cursor_is_sequential_identity_bound_and_mtp_lookup_is_pure() -> None:
    request, _processed_batch, _tower_output, prompt, plan = _plan()
    cursor = VisionSpliceCursor(plan)

    first = cursor.consume(
        identity=request.identity,
        plan_digest=plan.plan_digest,
        token_ids=prompt[:3],
    )
    assert first is not None
    assert (first.row_start, first.row_end, first.pad_offsets) == (0, 2, (1, 2))
    before_lookup = (cursor.prompt_cursor, cursor.row_cursor)
    lookup = cursor.lookup_window(
        identity=request.identity,
        plan_digest=plan.plan_digest,
        prompt_start=2,
        token_ids=prompt[2:8],
    )
    assert lookup is not None
    assert (lookup.row_start, lookup.row_end) == (1, 6)
    assert (cursor.prompt_cursor, cursor.row_cursor) == before_lookup

    with pytest.raises(VisionContractError) as mismatch:
        cursor.consume(
            identity=request.identity,
            plan_digest=plan.plan_digest,
            token_ids=(123,),
        )
    assert mismatch.value.code == "splice_prompt_mismatch"

    tail = cursor.consume(
        identity=request.identity,
        plan_digest=plan.plan_digest,
        token_ids=prompt[3:],
    )
    assert tail is not None
    cursor.assert_complete(
        identity=request.identity,
        plan_digest=plan.plan_digest,
    )

    cursor.reset(identity=request.identity, plan_digest=plan.plan_digest)
    assert (cursor.prompt_cursor, cursor.row_cursor) == (0, 0)


def test_splice_cursor_rejects_foreign_identity_and_non_owner_thread() -> None:
    request, _processed_batch, _tower_output, prompt, plan = _plan()
    cursor = VisionSpliceCursor(plan)
    foreign = VisionRequestIdentity(
        response_id="resp_foreign",
        runtime=request.identity.runtime,
        bundle_digest=request.identity.bundle_digest,
    )

    with pytest.raises(VisionContractError) as identity_error:
        cursor.consume(
            identity=foreign,
            plan_digest=plan.plan_digest,
            token_ids=prompt[:1],
        )
    assert identity_error.value.code == "splice_identity_mismatch"

    errors = []

    def reset_from_foreign_thread() -> None:
        try:
            cursor.reset(
                identity=request.identity,
                plan_digest=plan.plan_digest,
            )
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=reset_from_foreign_thread)
    thread.start()
    thread.join()
    assert len(errors) == 1
    assert isinstance(errors[0], VisionContractError)
    assert errors[0].code == "owner_thread_violation"


def test_mrope_single_image_matches_frozen_donor_table_exactly() -> None:
    request = _request(image_count=1)
    processed = _processed(request, grids=((1, 4, 4),))
    ids = (1, 2, PAD, PAD, PAD, PAD, 3)

    plan = build_mrope_plan(
        request.identity,
        ids,
        image_token_id=PAD,
        image_grids=processed.images,
        spatial_merge_size=2,
    )

    assert plan is not None
    assert plan.position_table[0] == (0, 1, 2, 2, 2, 2, 4)
    assert plan.position_table[1] == (0, 1, 2, 2, 3, 3, 4)
    assert plan.position_table[2] == (0, 1, 2, 3, 2, 3, 4)
    assert plan.rope_delta == -2
    assert len(plan.position_table_digest) == 64
    assert len(mrope_plan_digest(plan)) == 64


def test_mrope_multiple_images_preserves_grid_order_and_refuses_mismatch() -> None:
    request = _request()
    processed = _processed(request)
    ids = (7, PAD, PAD, PAD, PAD, 8, PAD, PAD, PAD, PAD, PAD, PAD)

    plan = build_mrope_plan(
        request.identity,
        ids,
        image_token_id=PAD,
        image_grids=processed.images,
        spatial_merge_size=2,
    )

    assert plan is not None
    assert plan.position_table[0] == (
        0,
        1,
        1,
        1,
        1,
        3,
        4,
        4,
        4,
        4,
        4,
        4,
    )
    assert plan.rope_delta == -5

    assert (
        build_mrope_plan(
            request.identity,
            (*ids, PAD),
            image_token_id=PAD,
            image_grids=processed.images,
            spatial_merge_size=2,
        )
        is None
    )
    assert (
        build_mrope_plan(
            request.identity,
            (*ids, 55),
            image_token_id=PAD,
            image_grids=processed.images,
            spatial_merge_size=2,
            video_token_id=55,
        )
        is None
    )
