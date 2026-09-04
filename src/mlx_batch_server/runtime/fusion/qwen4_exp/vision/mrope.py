# SPDX-License-Identifier: Apache-2.0
# Derived from youssofal/mtplx@6d0ddf0575faa9acf77e63c57e48ea1602a7e4ab
# mtplx/vision/mrope.py, transitively adapted from mlx-vlm (MIT).
"""Pure-Python per-request Qwen4Exp M-RoPE planning."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass

from .processing import (
    ImageGridReceipt,
    VisionContractError,
    VisionRequestIdentity,
    _runtime_payload,
)


@dataclass(frozen=True, slots=True)
class MropePlan:
    """Immutable exact three-axis positions and decode-time rope delta."""

    identity: VisionRequestIdentity
    image_token_id: int
    spatial_merge_size: int
    input_token_count: int
    position_table: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    position_table_digest: str
    rope_delta: int

    def __post_init__(self) -> None:
        table = tuple(
            tuple(int(value) for value in axis) for axis in self.position_table
        )
        if self.image_token_id < 0 or self.spatial_merge_size < 1:
            raise VisionContractError(
                "invalid_mrope_config",
                "M-RoPE token id and merge size must be valid",
            )
        if self.input_token_count < 1:
            raise VisionContractError(
                "invalid_mrope_length",
                "M-RoPE input token count must be positive",
            )
        if len(table) != 3 or any(
            len(axis) != self.input_token_count for axis in table
        ):
            raise VisionContractError(
                "invalid_mrope_table",
                "M-RoPE table must have shape [3, input_token_count]",
            )
        expected_digest = _sha256_json(table)
        if self.position_table_digest != expected_digest:
            raise VisionContractError(
                "mrope_digest_mismatch",
                "M-RoPE digest must match the exact immutable position table",
            )
        object.__setattr__(self, "position_table", table)


def build_mrope_plan(
    identity: VisionRequestIdentity,
    input_ids: Sequence[int],
    *,
    image_token_id: int,
    image_grids: Sequence[ImageGridReceipt],
    spatial_merge_size: int,
    video_token_id: int | None = None,
) -> MropePlan | None:
    """Build exact multi-image positions or refuse video/layout mismatches."""

    ids = _token_tuple(input_ids)
    grids = tuple(image_grids)
    if not ids or not grids or len(grids) > 8:
        return None
    if spatial_merge_size < 1 or image_token_id < 0:
        raise VisionContractError(
            "invalid_mrope_config",
            "M-RoPE token id and merge size must be valid",
        )
    if video_token_id is not None and int(video_token_id) in ids:
        return None
    for expected_index, grid in enumerate(grids):
        if grid.identity != identity or grid.image_index != expected_index:
            raise VisionContractError(
                "mrope_grid_identity_mismatch",
                "M-RoPE grids must belong to this request in prompt order",
            )
        if grid.spatial_merge_size != spatial_merge_size:
            raise VisionContractError(
                "mrope_merge_mismatch",
                "M-RoPE merge size must match every grid receipt",
            )
        t, h, w = grid.grid_thw
        if any(value < 1 for value in (t, h, w)):
            raise VisionContractError(
                "invalid_mrope_grid",
                "M-RoPE grid dimensions must be positive",
            )
        if h % spatial_merge_size or w % spatial_merge_size:
            raise VisionContractError(
                "mrope_grid_not_divisible",
                "M-RoPE spatial grids must be divisible by merge size",
            )
        expected_rows = t * h * w // (spatial_merge_size**2)
        if grid.pad_rows != expected_rows:
            raise VisionContractError(
                "mrope_pad_row_mismatch",
                "M-RoPE pad rows must equal t*h*w/merge^2",
            )
    expected_pads = sum(grid.pad_rows for grid in grids)
    if sum(token == image_token_id for token in ids) != expected_pads:
        return None

    axes: list[list[int]] = [[], [], []]
    prompt_cursor = 0
    next_position = 0
    for grid in grids:
        try:
            image_start = ids.index(image_token_id, prompt_cursor)
        except ValueError:
            return None
        _append_text_positions(
            axes,
            start=next_position,
            count=image_start - prompt_cursor,
        )
        next_position += image_start - prompt_cursor

        block = grid.pad_rows
        image_end = image_start + block
        if image_end > len(ids) or any(
            token != image_token_id for token in ids[image_start:image_end]
        ):
            return None
        t, h, w = grid.grid_thw
        llm_h = h // spatial_merge_size
        llm_w = w // spatial_merge_size
        for temporal in range(t):
            for row in range(llm_h):
                for column in range(llm_w):
                    axes[0].append(next_position + temporal)
                    axes[1].append(next_position + row)
                    axes[2].append(next_position + column)
        next_position += max(t, llm_h, llm_w)
        prompt_cursor = image_end

    if image_token_id in ids[prompt_cursor:]:
        return None
    tail = len(ids) - prompt_cursor
    _append_text_positions(axes, start=next_position, count=tail)
    next_position += tail
    table = (tuple(axes[0]), tuple(axes[1]), tuple(axes[2]))
    if any(len(axis) != len(ids) for axis in table):
        return None
    table_digest = _sha256_json(table)
    return MropePlan(
        identity=identity,
        image_token_id=image_token_id,
        spatial_merge_size=spatial_merge_size,
        input_token_count=len(ids),
        position_table=table,
        position_table_digest=table_digest,
        rope_delta=next_position - len(ids),
    )


def mrope_plan_digest(plan: MropePlan) -> str:
    """Digest identity plus the exact immutable position table."""

    return _sha256_json(
        {
            "identity": {
                "response_id": plan.identity.response_id,
                "runtime": _runtime_payload(plan.identity.runtime),
                "bundle_digest": plan.identity.bundle_digest,
            },
            "image_token_id": plan.image_token_id,
            "spatial_merge_size": plan.spatial_merge_size,
            "position_table_digest": plan.position_table_digest,
            "rope_delta": plan.rope_delta,
        }
    )


def _append_text_positions(
    axes: list[list[int]],
    *,
    start: int,
    count: int,
) -> None:
    positions = list(range(start, start + count))
    for axis in axes:
        axis.extend(positions)


def _token_tuple(token_ids: Sequence[int]) -> tuple[int, ...]:
    try:
        tokens = tuple(int(token) for token in token_ids)
    except (TypeError, ValueError) as exc:
        raise VisionContractError(
            "invalid_mrope_tokens",
            "M-RoPE input ids must be integers",
        ) from exc
    if any(token < 0 for token in tokens):
        raise VisionContractError(
            "invalid_mrope_tokens",
            "M-RoPE input ids must be non-negative",
        )
    return tokens


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
