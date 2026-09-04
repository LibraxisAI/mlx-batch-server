# SPDX-License-Identifier: Apache-2.0
# Derived from youssofal/mtplx@6d0ddf0575faa9acf77e63c57e48ea1602a7e4ab
# mtplx/vision/processing.py, itself adapted from mlx-vlm (Apache-2.0).
"""Tensor-free, whole-request Qwen4Exp vision preprocessing contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ....contracts import RuntimeKey
from ..media_resolver import ResolvedMediaBundle

MAX_REQUEST_IMAGES = 8


class VisionContractError(ValueError):
    """Fail-closed vision contract error with a stable machine code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class VisionRequestIdentity:
    """Identity shared by every receipt produced for one response request."""

    response_id: str
    runtime: RuntimeKey
    bundle_digest: str

    def __post_init__(self) -> None:
        if not self.response_id:
            raise VisionContractError(
                "invalid_response_id",
                "response_id must not be empty",
            )
        if not isinstance(self.runtime, RuntimeKey):
            raise VisionContractError(
                "invalid_runtime",
                "runtime must be a RuntimeKey",
            )
        _require_sha256(self.bundle_digest, "bundle_digest")

    @property
    def digest(self) -> str:
        return _sha256_json(
            {
                "response_id": self.response_id,
                "runtime": _runtime_payload(self.runtime),
                "bundle_digest": self.bundle_digest,
            }
        )


@dataclass(frozen=True, slots=True)
class OpaqueRows:
    """Opaque tensor-like handle accompanied by an asserted row count."""

    handle: object
    row_count: int

    def __post_init__(self) -> None:
        if self.handle is None:
            raise VisionContractError(
                "missing_opaque_handle",
                "opaque row handle must not be None",
            )
        if self.row_count < 1:
            raise VisionContractError(
                "invalid_row_count",
                "opaque row count must be positive",
            )


@dataclass(frozen=True, slots=True)
class VisionProcessingRequest:
    """One whole resolved bundle entering model-specific preprocessing."""

    identity: VisionRequestIdentity
    bundle: ResolvedMediaBundle
    spatial_merge_size: int

    def __post_init__(self) -> None:
        if not isinstance(self.bundle, ResolvedMediaBundle):
            raise VisionContractError(
                "invalid_media_bundle",
                "vision preprocessing requires one ResolvedMediaBundle",
            )
        if self.bundle.digest != self.identity.bundle_digest:
            raise VisionContractError(
                "bundle_identity_mismatch",
                "resolved bundle digest does not match request identity",
            )
        if not 1 <= len(self.bundle.images) <= MAX_REQUEST_IMAGES:
            raise VisionContractError(
                "image_count_out_of_bounds",
                f"vision request must contain 1..{MAX_REQUEST_IMAGES} images",
            )
        if self.spatial_merge_size < 1:
            raise VisionContractError(
                "invalid_merge_size",
                "spatial merge size must be positive",
            )
        order = tuple(
            (image.part_index, image.item_index) for image in self.bundle.images
        )
        if len(set(order)) != len(order) or order != tuple(sorted(order)):
            raise VisionContractError(
                "invalid_image_order",
                "resolved images must have unique deterministic bundle order",
            )


@dataclass(frozen=True, slots=True)
class ImageGridReceipt:
    """Immutable image/grid receipt in whole-request prompt order."""

    identity: VisionRequestIdentity
    image_index: int
    part_index: int
    item_index: int
    content_digest: str
    grid_thw: tuple[int, int, int]
    spatial_merge_size: int
    patch_start: int
    patch_end: int
    pad_rows: int

    def __post_init__(self) -> None:
        if self.image_index < 0 or self.part_index < 0 or self.item_index < 0:
            raise VisionContractError(
                "invalid_image_index",
                "image, part, and item indices must be non-negative",
            )
        _require_sha256(self.content_digest, "content_digest")
        if len(self.grid_thw) != 3 or any(value < 1 for value in self.grid_thw):
            raise VisionContractError(
                "invalid_grid",
                "grid_thw must contain three positive dimensions",
            )
        if self.spatial_merge_size < 1:
            raise VisionContractError(
                "invalid_merge_size",
                "spatial merge size must be positive",
            )
        t, h, w = self.grid_thw
        merge = self.spatial_merge_size
        if h % merge or w % merge:
            raise VisionContractError(
                "grid_not_divisible",
                "spatial grid dimensions must be divisible by merge size",
            )
        patch_rows = t * h * w
        expected_pad_rows = patch_rows // (merge * merge)
        if self.patch_start < 0 or self.patch_end - self.patch_start != patch_rows:
            raise VisionContractError(
                "patch_row_mismatch",
                "patch range must equal t*h*w rows",
            )
        if self.pad_rows != expected_pad_rows:
            raise VisionContractError(
                "pad_row_mismatch",
                "pad rows must equal t*h*w/merge^2",
            )

    @property
    def patch_rows(self) -> int:
        return self.patch_end - self.patch_start


@dataclass(frozen=True, slots=True)
class ProcessedVisionBatch:
    """Validated whole-request preprocessing output without tensor knowledge."""

    identity: VisionRequestIdentity
    images: tuple[ImageGridReceipt, ...]
    pixel_values: OpaqueRows

    def __post_init__(self) -> None:
        images = tuple(self.images)
        if not 1 <= len(images) <= MAX_REQUEST_IMAGES:
            raise VisionContractError(
                "image_count_out_of_bounds",
                f"processed batch must contain 1..{MAX_REQUEST_IMAGES} images",
            )
        patch_cursor = 0
        for expected_index, image in enumerate(images):
            if image.identity != self.identity:
                raise VisionContractError(
                    "image_identity_mismatch",
                    "image grid receipt belongs to another request",
                )
            if image.image_index != expected_index:
                raise VisionContractError(
                    "image_order_mismatch",
                    "image grid receipts must use contiguous prompt order",
                )
            if image.patch_start != patch_cursor:
                raise VisionContractError(
                    "patch_range_gap",
                    "image patch ranges must be contiguous",
                )
            patch_cursor = image.patch_end
        if self.pixel_values.row_count != patch_cursor:
            raise VisionContractError(
                "preprocessor_row_mismatch",
                "pixel value rows must equal the sum of t*h*w",
            )
        object.__setattr__(self, "images", images)

    @property
    def total_pad_rows(self) -> int:
        return sum(image.pad_rows for image in self.images)


@runtime_checkable
class VisionPreprocessorPort(Protocol):
    """Injected tensor implementation; source resolution has already finished."""

    def preprocess(self, request: VisionProcessingRequest) -> ProcessedVisionBatch: ...


def validate_preprocessing_output(
    request: VisionProcessingRequest,
    output: ProcessedVisionBatch,
) -> ProcessedVisionBatch:
    """Validate an injected preprocessor's metadata against the whole bundle."""

    if not isinstance(output, ProcessedVisionBatch):
        raise VisionContractError(
            "invalid_preprocessor_output",
            "preprocessor must return ProcessedVisionBatch",
        )
    if output.identity != request.identity:
        raise VisionContractError(
            "preprocessor_identity_mismatch",
            "preprocessor output belongs to another request",
        )
    if len(output.images) != len(request.bundle.images):
        raise VisionContractError(
            "preprocessor_image_count_mismatch",
            "preprocessor output must preserve every bundle image",
        )
    for source, receipt in zip(request.bundle.images, output.images, strict=True):
        expected = (source.part_index, source.item_index, source.content_digest)
        actual = (receipt.part_index, receipt.item_index, receipt.content_digest)
        if actual != expected:
            raise VisionContractError(
                "preprocessor_image_metadata_mismatch",
                "preprocessor output changed image identity or order",
            )
        if receipt.spatial_merge_size != request.spatial_merge_size:
            raise VisionContractError(
                "preprocessor_merge_mismatch",
                "preprocessor output changed spatial merge size",
            )
    return output


def _runtime_payload(runtime: RuntimeKey) -> dict[str, str | None]:
    return {
        "model_id": runtime.model_id,
        "revision": runtime.revision,
        "adapter_path": runtime.adapter_path,
        "draft_model_id": runtime.draft_model_id,
        "backend": runtime.backend.value,
    }


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: str, field_name: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise VisionContractError(
            "invalid_digest",
            f"{field_name} must be a lowercase SHA-256 digest",
        )
    try:
        parsed = bytes.fromhex(value)
    except ValueError as exc:
        raise VisionContractError(
            "invalid_digest",
            f"{field_name} must be a lowercase SHA-256 digest",
        ) from exc
    if len(parsed) != 32 or value != value.lower():
        raise VisionContractError(
            "invalid_digest",
            f"{field_name} must be a lowercase SHA-256 digest",
        )
