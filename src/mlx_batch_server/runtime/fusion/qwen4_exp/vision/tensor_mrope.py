# SPDX-License-Identifier: Apache-2.0
# Derived from youssofal/mtplx@6d0ddf0575faa9acf77e63c57e48ea1602a7e4ab
# mtplx/vision/mrope.py and mtplx/models/qwen4_exp.py (Apache-2.0).
# M-RoPE positioning is transitively adapted from mlx-vlm (MIT),
# Copyright Prince Canuma and mlx-vlm contributors.
# Modified by LibraxisAI to replace request-global implicit state.
"""Per-request MLX application of an immutable Qwen4Exp M-RoPE plan."""

from __future__ import annotations

import math
import threading
from collections.abc import Sequence

import mlx.core as mx

from .mrope import MropePlan
from .processing import VisionContractError


class Qwen4ExpTensorMrope:
    """Owner-thread M-RoPE tensors for exactly one vision request."""

    __slots__ = (
        "_axes",
        "_owner_thread_id",
        "_plan",
        "_rotary_dim",
        "_table",
    )

    def __init__(
        self,
        plan: MropePlan,
        *,
        mrope_section: Sequence[int],
        mrope_interleaved: bool,
        rotary_dim: int,
    ) -> None:
        if not isinstance(plan, MropePlan):
            raise VisionContractError(
                "invalid_mrope_plan",
                "tensor M-RoPE requires an immutable MropePlan",
            )
        try:
            section = tuple(int(item) for item in mrope_section)
        except (TypeError, ValueError) as exc:
            raise VisionContractError(
                "invalid_mrope_section",
                "M-RoPE section must contain three positive integers",
            ) from exc
        if len(section) != 3 or any(item < 1 for item in section):
            raise VisionContractError(
                "invalid_mrope_section",
                "M-RoPE section must contain three positive integers",
            )
        if rotary_dim < 2 or rotary_dim % 2:
            raise VisionContractError(
                "invalid_rotary_dimension",
                "M-RoPE rotary dimension must be positive and even",
            )
        if sum(section) != rotary_dim // 2:
            raise VisionContractError(
                "mrope_section_width_mismatch",
                "M-RoPE frequency sections must cover half the rotary width",
            )
        axes = _build_mrope_axes(section, bool(mrope_interleaved))
        self._plan = plan
        self._rotary_dim = rotary_dim
        self._table = mx.array(plan.position_table, dtype=mx.int32)
        self._axes = mx.array(axes, dtype=mx.int32)
        self._owner_thread_id = threading.get_ident()

    @property
    def plan(self) -> MropePlan:
        return self._plan

    @property
    def rope_delta(self) -> int:
        return self._plan.rope_delta

    @property
    def position_table(self) -> mx.array:
        """Return the request-owned immutable-position tensor for scoped use."""

        self._assert_owner()
        return self._table

    def apply(
        self,
        query: mx.array,
        key: mx.array,
        *,
        position_start: int,
        inverse_frequency: mx.array,
        attention_scaling: float = 1.0,
    ) -> tuple[mx.array, mx.array]:
        """Apply table positions in prefill and shifted text RoPE in decode."""

        self._assert_owner()
        query_shape = _shape(query, "query")
        key_shape = _shape(key, "key")
        if (
            len(query_shape) != 4
            or len(key_shape) != 4
            or query_shape[0] != 1
            or key_shape[0] != 1
            or query_shape[1] != key_shape[1]
            or query_shape[3] != key_shape[3]
        ):
            raise VisionContractError(
                "invalid_attention_shape",
                "Qwen4Exp attention tensors must be batch-1 [B,S,H,D]",
            )
        if query_shape[3] < self._rotary_dim:
            raise VisionContractError(
                "rotary_dimension_exceeded",
                "attention head width is smaller than the rotary dimension",
            )
        frequency_shape = _shape(inverse_frequency, "inverse frequency")
        if frequency_shape != (self._rotary_dim // 2,):
            raise VisionContractError(
                "inverse_frequency_mismatch",
                "inverse frequency width does not match M-RoPE sections",
            )
        if position_start < 0:
            raise VisionContractError(
                "invalid_position_start",
                "M-RoPE position start must be non-negative",
            )
        scaling = float(attention_scaling)
        if not math.isfinite(scaling) or scaling <= 0.0:
            raise VisionContractError(
                "invalid_attention_scaling",
                "RoPE attention scaling must be finite and positive",
            )

        sequence_length = query_shape[1]
        position_end = position_start + sequence_length
        if position_start < self._plan.input_token_count < position_end:
            raise VisionContractError(
                "mrope_window_crosses_prompt",
                "one attention call cannot mix prompt-table and decode positions",
            )
        if position_end <= self._plan.input_token_count:
            cosine, sine = _mrope_cos_sin(
                self._table[:, position_start:position_end],
                inverse_frequency,
                self._axes,
            )
        else:
            shifted = mx.arange(
                position_start + self._plan.rope_delta,
                position_end + self._plan.rope_delta,
                dtype=mx.int32,
            )
            cosine, sine = _rope_cos_sin(
                shifted,
                inverse_frequency,
                scaling,
            )
        return (
            _apply_partial_rope(query, cosine, sine),
            _apply_partial_rope(key, cosine, sine),
        )

    def _assert_owner(self) -> None:
        if threading.get_ident() != self._owner_thread_id:
            raise VisionContractError(
                "owner_thread_violation",
                "M-RoPE tensors must run on their request owner thread",
            )


def select_qwen4_attention_mask(
    mrope_state: Qwen4ExpTensorMrope | None,
    sparse_selection: object,
) -> object:
    """Vision uses dense-causal attention; text keeps QSA sparse selection."""

    if mrope_state is None:
        return sparse_selection
    if not isinstance(mrope_state, Qwen4ExpTensorMrope):
        raise VisionContractError(
            "invalid_mrope_state",
            "vision attention requires Qwen4ExpTensorMrope state",
        )
    return None


def _build_mrope_axes(
    section: tuple[int, int, int],
    interleaved: bool,
) -> tuple[int, ...]:
    remaining = list(section)
    axes: list[int] = []
    if interleaved:
        axis = 0
        while sum(remaining) > 0:
            if remaining[axis] > 0:
                axes.append(axis)
                remaining[axis] -= 1
            axis = (axis + 1) % len(remaining)
    else:
        for axis, count in enumerate(remaining):
            axes.extend([axis] * count)
    return tuple(axes)


def _mrope_cos_sin(
    positions: mx.array,
    inverse_frequency: mx.array,
    axes: mx.array,
) -> tuple[mx.array, mx.array]:
    selected = mx.take(positions.astype(mx.float32), axes, axis=0)
    angles = selected.transpose(1, 0) * inverse_frequency[None, :]
    embedding = mx.concatenate([angles, angles], axis=-1)
    return mx.cos(embedding), mx.sin(embedding)


def _rope_cos_sin(
    positions: mx.array,
    inverse_frequency: mx.array,
    attention_scaling: float,
) -> tuple[mx.array, mx.array]:
    angles = positions.astype(mx.float32)[:, None] * inverse_frequency[None, :]
    embedding = mx.concatenate([angles, angles], axis=-1)
    cosine = mx.cos(embedding)
    sine = mx.sin(embedding)
    if attention_scaling != 1.0:
        cosine = cosine * attention_scaling
        sine = sine * attention_scaling
    return cosine, sine


def _apply_partial_rope(
    value: mx.array,
    cosine: mx.array,
    sine: mx.array,
) -> mx.array:
    rotary_width = int(cosine.shape[-1])
    rotary = value[..., :rotary_width]
    passthrough = value[..., rotary_width:]
    half = rotary_width // 2
    first = rotary[..., :half]
    second = rotary[..., half:]
    rotated = mx.concatenate([-second, first], axis=-1)
    cosine = cosine[:, None, :]
    sine = sine[:, None, :]
    rotary = (
        rotary.astype(mx.float32) * cosine + rotated.astype(mx.float32) * sine
    ).astype(value.dtype)
    return mx.concatenate([rotary, passthrough], axis=-1)


def _shape(value: object, label: str) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise VisionContractError(
            "invalid_tensor_handle",
            f"{label} is not an MLX tensor-like value",
        )
    try:
        return tuple(int(item) for item in shape)
    except (TypeError, ValueError) as exc:
        raise VisionContractError(
            "invalid_tensor_handle",
            f"{label} has an invalid shape",
        ) from exc
