# SPDX-License-Identifier: Apache-2.0
# Derived from youssofal/mtplx@6d0ddf0575faa9acf77e63c57e48ea1602a7e4ab
# mtplx/vision/qwen3_vl_tower.py, itself adapted from mlx-vlm (Apache-2.0).
"""Tensor-free Qwen4Exp vision tower input and output contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .processing import (
    OpaqueRows,
    ProcessedVisionBatch,
    VisionContractError,
    VisionRequestIdentity,
)


@dataclass(frozen=True, slots=True)
class VisionTowerRequest:
    identity: VisionRequestIdentity
    processed: ProcessedVisionBatch

    def __post_init__(self) -> None:
        if self.processed.identity != self.identity:
            raise VisionContractError(
                "tower_request_identity_mismatch",
                "processed vision batch belongs to another request",
            )


@dataclass(frozen=True, slots=True)
class VisionEmbeddingSlice:
    """Rows in the concatenated tower output belonging to one image."""

    identity: VisionRequestIdentity
    image_index: int
    content_digest: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.image_index < 0 or self.start < 0 or self.end <= self.start:
            raise VisionContractError(
                "invalid_embedding_slice",
                "embedding slice indices must form a positive half-open range",
            )
        if len(self.content_digest) != 64:
            raise VisionContractError(
                "invalid_embedding_digest",
                "embedding slice requires an image content digest",
            )

    @property
    def row_count(self) -> int:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class DeepstackFeatureReceipt:
    identity: VisionRequestIdentity
    injection_index: int
    rows: OpaqueRows

    def __post_init__(self) -> None:
        if self.injection_index < 0:
            raise VisionContractError(
                "invalid_deepstack_index",
                "deepstack injection index must be non-negative",
            )


@dataclass(frozen=True, slots=True)
class VisionTowerOutput:
    """Ordered merged embeddings and optional deepstack row handles."""

    identity: VisionRequestIdentity
    embeddings: OpaqueRows
    image_slices: tuple[VisionEmbeddingSlice, ...]
    deepstack: tuple[DeepstackFeatureReceipt, ...] = ()

    def __post_init__(self) -> None:
        image_slices = tuple(self.image_slices)
        deepstack = tuple(self.deepstack)
        cursor = 0
        for expected_index, image_slice in enumerate(image_slices):
            if image_slice.identity != self.identity:
                raise VisionContractError(
                    "tower_slice_identity_mismatch",
                    "tower image slice belongs to another request",
                )
            if image_slice.image_index != expected_index:
                raise VisionContractError(
                    "tower_image_order_mismatch",
                    "tower image slices must preserve prompt order",
                )
            if image_slice.start != cursor:
                raise VisionContractError(
                    "tower_slice_gap",
                    "tower image slices must be contiguous",
                )
            cursor = image_slice.end
        if not image_slices or self.embeddings.row_count != cursor:
            raise VisionContractError(
                "tower_row_mismatch",
                "tower embedding rows must equal all image slice rows",
            )
        for expected_index, feature in enumerate(deepstack):
            if feature.identity != self.identity:
                raise VisionContractError(
                    "deepstack_identity_mismatch",
                    "deepstack receipt belongs to another request",
                )
            if feature.injection_index != expected_index:
                raise VisionContractError(
                    "deepstack_order_mismatch",
                    "deepstack receipts must use contiguous injection order",
                )
            if feature.rows.row_count != self.embeddings.row_count:
                raise VisionContractError(
                    "deepstack_row_mismatch",
                    "every deepstack feature must preserve merged row count",
                )
        object.__setattr__(self, "image_slices", image_slices)
        object.__setattr__(self, "deepstack", deepstack)


@runtime_checkable
class VisionTowerPort(Protocol):
    """Injected Qwen vision tower implementation behind opaque row handles."""

    def forward(self, request: VisionTowerRequest) -> VisionTowerOutput: ...


def validate_tower_output(
    request: VisionTowerRequest,
    output: VisionTowerOutput,
) -> VisionTowerOutput:
    """Validate tower order and merged rows against preprocessing receipts."""

    if not isinstance(output, VisionTowerOutput):
        raise VisionContractError(
            "invalid_tower_output",
            "vision tower must return VisionTowerOutput",
        )
    if output.identity != request.identity:
        raise VisionContractError(
            "tower_identity_mismatch",
            "vision tower output belongs to another request",
        )
    if len(output.image_slices) != len(request.processed.images):
        raise VisionContractError(
            "tower_image_count_mismatch",
            "vision tower must preserve every processed image",
        )
    for grid, image_slice in zip(
        request.processed.images,
        output.image_slices,
        strict=True,
    ):
        if image_slice.content_digest != grid.content_digest:
            raise VisionContractError(
                "tower_image_digest_mismatch",
                "vision tower changed image identity or order",
            )
        if image_slice.row_count != grid.pad_rows:
            raise VisionContractError(
                "tower_image_row_mismatch",
                "tower rows per image must equal t*h*w/merge^2",
            )
    if output.embeddings.row_count != request.processed.total_pad_rows:
        raise VisionContractError(
            "tower_total_row_mismatch",
            "tower output rows must equal the sum of image pad rows",
        )
    return output
