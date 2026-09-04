# SPDX-License-Identifier: Apache-2.0
# Derived from youssofal/mtplx@6d0ddf0575faa9acf77e63c57e48ea1602a7e4ab
# mtplx/vision/processing.py, itself adapted from mlx-vlm (Apache-2.0).
# Modified by LibraxisAI for ordered whole-request tensor preprocessing.
"""Executable MLX/Pillow/NumPy preprocessing behind vision contracts."""

from __future__ import annotations

import hashlib
import io
import json
import math
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from ..model.load_plan import Qwen4ExpModelLoadPlan
from .processing import (
    MAX_REQUEST_IMAGES,
    ImageGridReceipt,
    OpaqueRows,
    ProcessedVisionBatch,
    VisionContractError,
    VisionProcessingRequest,
    validate_preprocessing_output,
)

MAX_IMAGE_BYTES = 50 * 1024 * 1024
MAX_IMAGE_DIM = 8000
_DEFAULT_MEAN = (0.5, 0.5, 0.5)
_DEFAULT_STD = (0.5, 0.5, 0.5)


@dataclass(frozen=True, slots=True)
class TensorVisionPreprocessorConfig:
    """The checkpoint image-processor values that affect tensor layout."""

    patch_size: int = 16
    merge_size: int = 2
    temporal_patch_size: int = 2
    min_pixels: int = 56 * 56
    max_pixels: int = 14 * 14 * 4 * 1280
    do_rescale: bool = True
    rescale_factor: float = 1.0 / 255.0
    do_normalize: bool = True
    image_mean: tuple[float, float, float] = _DEFAULT_MEAN
    image_std: tuple[float, float, float] = _DEFAULT_STD
    max_image_bytes: int = MAX_IMAGE_BYTES
    max_image_dim: int = MAX_IMAGE_DIM

    def __post_init__(self) -> None:
        positive = (
            self.patch_size,
            self.merge_size,
            self.temporal_patch_size,
            self.min_pixels,
            self.max_pixels,
            self.max_image_bytes,
            self.max_image_dim,
        )
        if any(value < 1 for value in positive):
            raise VisionContractError(
                "invalid_preprocessor_config",
                "vision preprocessing geometry and limits must be positive",
            )
        if self.min_pixels > self.max_pixels:
            raise VisionContractError(
                "invalid_pixel_bounds",
                "minimum image pixels cannot exceed maximum image pixels",
            )
        if len(self.image_mean) != 3 or len(self.image_std) != 3:
            raise VisionContractError(
                "invalid_normalization",
                "image normalization requires exactly three channels",
            )
        if any(not math.isfinite(value) for value in self.image_mean):
            raise VisionContractError(
                "invalid_normalization",
                "image means must be finite",
            )
        if any(not math.isfinite(value) or value == 0.0 for value in self.image_std):
            raise VisionContractError(
                "invalid_normalization",
                "image standard deviations must be finite and non-zero",
            )
        if not math.isfinite(self.rescale_factor):
            raise VisionContractError(
                "invalid_rescale_factor",
                "image rescale factor must be finite",
            )

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
    ) -> TensorVisionPreprocessorConfig:
        """Read the mlx-vlm-compatible preprocessor configuration."""

        size = raw.get("size") or {}
        if not isinstance(size, Mapping):
            raise VisionContractError(
                "invalid_preprocessor_config",
                "preprocessor size must be a mapping",
            )
        min_pixels = size.get("shortest_edge", 56 * 56)
        max_pixels = size.get("longest_edge", 14 * 14 * 4 * 1280)
        if raw.get("min_pixels") is not None:
            min_pixels = raw["min_pixels"]
        if raw.get("max_pixels") is not None:
            max_pixels = raw["max_pixels"]
        return cls(
            patch_size=_config_integer(raw, "patch_size", 16),
            merge_size=_config_integer(raw, "merge_size", 2),
            temporal_patch_size=_config_integer(
                raw,
                "temporal_patch_size",
                2,
            ),
            min_pixels=_typed_integer("min_pixels", min_pixels),
            max_pixels=_typed_integer("max_pixels", max_pixels),
            do_rescale=_config_boolean(raw, "do_rescale", True),
            rescale_factor=_config_number(
                raw,
                "rescale_factor",
                1.0 / 255.0,
            ),
            do_normalize=_config_boolean(raw, "do_normalize", True),
            image_mean=_channel_tuple(
                "image_mean",
                raw.get("image_mean", _DEFAULT_MEAN),
            ),
            image_std=_channel_tuple(
                "image_std",
                raw.get("image_std", _DEFAULT_STD),
            ),
            max_image_bytes=_config_integer(
                raw,
                "max_image_bytes",
                MAX_IMAGE_BYTES,
            ),
            max_image_dim=_config_integer(
                raw,
                "max_image_dim",
                MAX_IMAGE_DIM,
            ),
        )

    @classmethod
    def from_load_plan(
        cls,
        plan: Qwen4ExpModelLoadPlan,
    ) -> TensorVisionPreprocessorConfig:
        if not isinstance(plan, Qwen4ExpModelLoadPlan):
            raise VisionContractError(
                "invalid_load_plan",
                "vision preprocessing requires Qwen4ExpModelLoadPlan",
            )
        raw = json.loads(plan.preprocessor_config_json)
        if not isinstance(raw, Mapping):
            raise VisionContractError(
                "invalid_preprocessor_config",
                "load-plan preprocessor metadata must be a mapping",
            )
        config = cls.from_mapping(raw)
        vision = plan.config.vision
        if (
            config.patch_size != vision.patch_size
            or config.merge_size != vision.spatial_merge_size
            or config.temporal_patch_size != vision.temporal_patch_size
        ):
            raise VisionContractError(
                "preprocessor_model_mismatch",
                "preprocessor and model vision geometry disagree",
            )
        return config


class Qwen4ExpTensorPreprocessor:
    """Owner-thread implementation of ``VisionPreprocessorPort``."""

    __slots__ = ("_config", "_owner_thread_id")

    def __init__(self, config: TensorVisionPreprocessorConfig) -> None:
        if not isinstance(config, TensorVisionPreprocessorConfig):
            raise VisionContractError(
                "invalid_preprocessor_config",
                "tensor preprocessor requires its validated configuration",
            )
        self._config = config
        self._owner_thread_id = threading.get_ident()

    @classmethod
    def from_load_plan(
        cls,
        plan: Qwen4ExpModelLoadPlan,
    ) -> Qwen4ExpTensorPreprocessor:
        return cls(TensorVisionPreprocessorConfig.from_load_plan(plan))

    def preprocess(self, request: VisionProcessingRequest) -> ProcessedVisionBatch:
        self._assert_owner()
        if not isinstance(request, VisionProcessingRequest):
            raise VisionContractError(
                "invalid_preprocessor_request",
                "tensor preprocessor requires VisionProcessingRequest",
            )
        if request.spatial_merge_size != self._config.merge_size:
            raise VisionContractError(
                "preprocessor_merge_mismatch",
                "request and checkpoint spatial merge sizes disagree",
            )
        if not 1 <= len(request.bundle.images) <= MAX_REQUEST_IMAGES:
            raise VisionContractError(
                "image_count_out_of_bounds",
                f"tensor preprocessing requires 1..{MAX_REQUEST_IMAGES} images",
            )

        patches: list[np.ndarray] = []
        receipts: list[ImageGridReceipt] = []
        patch_cursor = 0
        for image_index, source in enumerate(request.bundle.images):
            if hashlib.sha256(source.content).hexdigest() != source.content_digest:
                raise VisionContractError(
                    "image_digest_mismatch",
                    "resolved image bytes do not match their content digest",
                )
            image = decode_image(
                source.content,
                max_image_bytes=self._config.max_image_bytes,
                max_image_dim=self._config.max_image_dim,
            )
            if image.size != (source.width, source.height):
                raise VisionContractError(
                    "image_dimension_mismatch",
                    "decoded image dimensions changed after materialization",
                )
            image_patches, grid = _preprocess_image(image, self._config)
            patch_rows = int(image_patches.shape[0])
            t, h, w = grid
            expected_rows = t * h * w
            if patch_rows != expected_rows:
                raise VisionContractError(
                    "patch_row_mismatch",
                    "preprocessed rows must equal the exact t*h*w grid",
                )
            pad_rows = expected_rows // (self._config.merge_size**2)
            receipts.append(
                ImageGridReceipt(
                    identity=request.identity,
                    image_index=image_index,
                    part_index=source.part_index,
                    item_index=source.item_index,
                    content_digest=source.content_digest,
                    grid_thw=grid,
                    spatial_merge_size=self._config.merge_size,
                    patch_start=patch_cursor,
                    patch_end=patch_cursor + patch_rows,
                    pad_rows=pad_rows,
                )
            )
            patches.append(image_patches)
            patch_cursor += patch_rows

        pixel_values = mx.array(np.concatenate(patches, axis=0))
        output = ProcessedVisionBatch(
            identity=request.identity,
            images=tuple(receipts),
            pixel_values=OpaqueRows(
                handle=pixel_values,
                row_count=int(pixel_values.shape[0]),
            ),
        )
        return validate_preprocessing_output(request, output)

    def _assert_owner(self) -> None:
        if threading.get_ident() != self._owner_thread_id:
            raise VisionContractError(
                "owner_thread_violation",
                "tensor preprocessing must run on its inference owner thread",
            )


def decode_image(
    data: bytes,
    *,
    max_image_bytes: int = MAX_IMAGE_BYTES,
    max_image_dim: int = MAX_IMAGE_DIM,
) -> Image.Image:
    """Decode immutable bytes without acquiring any source or control plane."""

    if not isinstance(data, bytes) or not data:
        raise VisionContractError(
            "invalid_image_bytes",
            "image payload must be non-empty immutable bytes",
        )
    if len(data) > max_image_bytes:
        raise VisionContractError(
            "image_bytes_exceeded",
            f"image payload is {len(data)} bytes, limit is {max_image_bytes}",
        )
    try:
        with Image.open(io.BytesIO(data)) as opened:
            width, height = opened.size
            if width > max_image_dim or height > max_image_dim:
                raise VisionContractError(
                    "image_dimension_exceeded",
                    f"image is {width}x{height}, limit is {max_image_dim} per side",
                )
            oriented = ImageOps.exif_transpose(opened)
            return oriented.convert("RGB")
    except VisionContractError:
        raise
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
        raise VisionContractError(
            "image_decode_failed",
            f"cannot decode image: {exc}",
        ) from exc


def smart_resize(
    height: int,
    width: int,
    *,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    """Return the exact mlx-vlm factor-aligned image geometry."""

    if min(height, width, factor, min_pixels, max_pixels) < 1:
        raise VisionContractError(
            "invalid_resize_geometry",
            "resize geometry and pixel bounds must be positive",
        )
    ratio = max(height, width) / min(height, width)
    if ratio > 200:
        raise VisionContractError(
            "image_aspect_ratio_exceeded",
            f"absolute aspect ratio must be smaller than 200, got {ratio}",
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _preprocess_image(
    image: Image.Image,
    config: TensorVisionPreprocessorConfig,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    factor = config.patch_size * config.merge_size
    width, height = image.size
    resized_h, resized_w = smart_resize(
        height,
        width,
        factor=factor,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
    )
    if (resized_w, resized_h) != (width, height):
        image = image.resize(
            (resized_w, resized_h),
            resample=Image.Resampling.BICUBIC,
        )

    array = np.transpose(np.asarray(image, dtype=np.uint8), (2, 0, 1))
    channels = int(array.shape[0])
    if channels != 3:
        raise VisionContractError(
            "invalid_image_channels",
            "Qwen4Exp vision preprocessing requires RGB images",
        )
    array = array.astype(np.float32)
    if config.do_rescale:
        array = array * config.rescale_factor
    if config.do_normalize:
        mean = np.array(config.image_mean, dtype=np.float32)[:, None, None]
        std = np.array(config.image_std, dtype=np.float32)[:, None, None]
        array = (array - mean) / std

    temporal = config.temporal_patch_size
    patch = config.patch_size
    merge = config.merge_size
    patches = np.repeat(array[None, None, ...], temporal, axis=1)
    grid_t = 1
    grid_h = resized_h // patch
    grid_w = resized_w // patch
    if grid_h % merge or grid_w % merge:
        raise VisionContractError(
            "grid_not_divisible",
            "preprocessed spatial grid must be divisible by merge size",
        )
    patches = patches.reshape(
        1,
        grid_t,
        temporal,
        channels,
        grid_h // merge,
        merge,
        patch,
        grid_w // merge,
        merge,
        patch,
    )
    patches = patches.transpose(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
    rows = patches.reshape(
        grid_t * grid_h * grid_w,
        channels * temporal * patch * patch,
    )
    return rows, (grid_t, grid_h, grid_w)


def _channel_tuple(name: str, value: object) -> tuple[float, float, float]:
    try:
        channels = tuple(float(item) for item in value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise VisionContractError(
            "invalid_normalization",
            f"{name} must contain exactly three numbers",
        ) from exc
    if len(channels) != 3:
        raise VisionContractError(
            "invalid_normalization",
            f"{name} must contain exactly three numbers",
        )
    return channels


def _typed_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise VisionContractError(
            "invalid_preprocessor_config",
            f"{name} must be an integer",
        )
    return value


def _config_integer(raw: Mapping[str, Any], name: str, default: int) -> int:
    return _typed_integer(name, raw.get(name, default))


def _config_boolean(raw: Mapping[str, Any], name: str, default: bool) -> bool:
    value = raw.get(name, default)
    if type(value) is not bool:
        raise VisionContractError(
            "invalid_preprocessor_config",
            f"{name} must be a boolean",
        )
    return value


def _config_number(
    raw: Mapping[str, Any],
    name: str,
    default: float,
) -> float:
    value = raw.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise VisionContractError(
            "invalid_preprocessor_config",
            f"{name} must be numeric",
        )
    return float(value)
