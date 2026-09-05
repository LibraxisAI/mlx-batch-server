# SPDX-License-Identifier: Apache-2.0
# Adapted from MTPLX commit 6d0ddf0575faa9acf77e63c57e48ea1602a7e4ab.
# Copyright © 2026 MTPLX.
#
# Qwen4-Exp (Qwen3.8-Flash-Next) target-owned MLX tensor backend.
#
# The pinned mlx-lm has no implementation for model_type "qwen4_exp"
# (Qwen4ExpForConditionalGeneration). This module implements the text trunk
# natively, reusing the pinned mlx-lm building blocks where the architecture
# genuinely overlaps (GatedDeltaNet, the qwen3_next MoE block) and adding the
# four genuinely new pieces:
#
#   * Gated Residual ("hyper-connections"): hc_count widened residual streams
#     with a learned low-rank read mix and per-stream scalar write gates.
#     There are NO input/post-attention layernorms and no final model.norm in
#     this family — the per-block hc_norm and the final hyper_connection_mixer
#     play those roles.
#   * QSA (Qwen Sparse Attention): standard gated GQA whose causal mask is
#     intersected with a per-query token selection produced by a
#     DeepSeek-V3.2-class indexer (relu-scored mean-pooled key blocks,
#     top-(budget/ratio) blocks + the incomplete tail block).
#   * PLE (Per-Layer Embedding): a hashed n-gram lookup memory (~51B params,
#     320M rows x 160) injected on one early linear-attention layer through a
#     per-stream sigmoid gate and a dilated depthwise convolution. This frozen
#     checkpoint stores the table as 128 embedded quantized shard modules.
#   * mrope carried by the family config; for text-only serving with equal
#     t/h/w positions the interleaved mrope is numerically identical to the
#     standard partial rotary embedding, which is what this module applies
#     (same treatment the pinned mlx-lm gives qwen3_5).
#
# Reference: transformers' modular_qwen4_exp.py (read 2026-08-26, T+9h after
# the weight drop). Norm convention: the Qwen4ExpTextRMSNorm family is stored
# zero-centered and applies (1+w) at runtime; the GDN gated norm is stored
# one-centered. Legacy direct-gamma conversions are recentered in sanitize.

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import math
import os
import re
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlx.core as mx
import numpy as np
from mlx import nn
from mlx_lm.models.base import create_ssm_mask
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.qwen3_5 import GatedDeltaNet as _Qwen3_5GatedDeltaNet
from mlx_lm.models.qwen3_next import (
    Qwen3NextSparseMoeBlock as _Qwen3NextSparseMoeBlock,
)
from mlx_lm.tokenizer_utils import load as load_tokenizer

from .....utils.model_limits import resolve_max_tokens
from .....utils.streaming_detokenizer import new_streaming_detokenizer
from ....backends.fused_mtp_mlx import FusedStepResult
from ....contracts import (
    GenerationRequest,
    LoadConfig,
    ModelSpec,
    PreparedGenerationRequest,
    RequestModality,
    RuntimeKey,
)
from ....events import UsageUpdate
from ...cache import CacheCleanupReceipt, CacheReleaseReason, CacheTier
from ...mtp import MtpAlignment, MtpDecision, MtpDisableReason, MtpPolicy
from ...output import Qwen4OutputChunk, Qwen4TurnEventEncoderFactory
from ...scheduler import DecodeResult, PrefillResult, SchedulerConfig, SchedulerPlan
from ..execution import Qwen4ExpExecutionBinding
from ..media import PreparedMediaItem, PreparedQwen4Prompt, PreparedTextItem
from ..media_resolver import ResolvedImage, ResolvedText
from ..prefix_store import (
    TEXT_CONTEXT_FINGERPRINT,
    Qwen4ExpPendingBoundaryCheckpoint,
    Qwen4ExpPrefixLeaseIdentity,
    Qwen4ExpPrefixLookupReceipt,
    Qwen4ExpPrefixReleaseReason,
    Qwen4ExpWholeBoundaryPrefixStore,
)
from ..vision.mrope import build_mrope_plan
from ..vision.processing import (
    VisionContractError,
    VisionProcessingRequest,
    VisionRequestIdentity,
)
from ..vision.splice import build_vision_splice_plan
from ..vision.tensor_mrope import Qwen4ExpTensorMrope
from ..vision.tensor_processing import Qwen4ExpTensorPreprocessor
from ..vision.tensor_splice import Qwen4ExpTensorSplicer
from ..vision.tensor_tower import Qwen4ExpVisionTensorTower
from ..vision.tower import VisionTowerRequest
from .load_plan import Qwen4ExpModelLoadPlan, load_qwen4_exp_plan
from .multirow import MultirowBatchPlan
from .sampling import (
    Distribution,
    SamplerConfig,
    SparseDistribution,
    acceptance_probability,
    distribution_from_logits,
    residual_distribution,
    sample_from_distribution,
)
from .tensor_support import (
    Qwen4ExpTensorCapabilities,
    attention_phase_scope,
    current_attention_phase,
    current_tensor_capabilities,
    tensor_capability_scope,
    vision_rope_scope,
    vision_rope_state,
)

if TYPE_CHECKING:
    from .config import Qwen4ExpCheckpointConfig, Qwen4ExpTextConfig


class TextArgs:
    """Read-only tensor view over the canonical checkpoint configuration."""

    __slots__ = ("_config", "capabilities")

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        capabilities: Qwen4ExpTensorCapabilities,
    ) -> None:
        self._config = config
        self.capabilities = capabilities

    def __getattr__(self, name: str) -> Any:
        return getattr(self._config, name)

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def eos_id(self) -> int:
        return self.eos_token_ids[0]


class MtpTextArgs:
    """MTP decoder view over the same immutable checkpoint configuration."""

    __slots__ = ("_text",)

    def __init__(self, text: TextArgs) -> None:
        self._text = text

    def __getattr__(self, name: str) -> Any:
        return getattr(self._text, name)

    @property
    def layer_types(self) -> tuple[str, ...]:
        return self._text.mtp.layer_types

    @property
    def num_hidden_layers(self) -> int:
        return self._text.mtp.num_hidden_layers

    @property
    def rope_theta(self) -> int:
        return self._text.mtp.rope_theta

    @property
    def ple_layer_ids(self) -> tuple[int, ...]:
        return ()


def _rope_inv_freq_and_scaling(args: TextArgs) -> tuple[mx.array, float]:
    """Build the exact default RoPE frequencies carried by the frozen plan."""

    rotary_dim = int(args.rotary_dim)
    if rotary_dim <= 0 or rotary_dim % 2:
        raise ValueError(
            f"rotary_dim must be a positive even integer, got {rotary_dim}"
        )

    base = float(args.rope_theta)
    if not math.isfinite(base) or base <= 1.0:
        raise ValueError(f"rope_theta must be finite and greater than 1, got {base}")

    positions = mx.arange(0, rotary_dim, 2, dtype=mx.float32)
    position_frequencies = base ** (positions / rotary_dim)
    return (1.0 / position_frequencies).astype(mx.float32), 1.0


def _rope_cos_sin(
    positions: mx.array,
    inv_freq: mx.array,
    attention_scaling: float = 1.0,
) -> tuple[mx.array, mx.array]:
    """Non-interleaved RoPE tables, including static-YaRN amplitude scaling."""

    angles = positions.astype(mx.float32)[..., None] * inv_freq
    emb = mx.concatenate([angles, angles], axis=-1)
    cosine = mx.cos(emb)
    sine = mx.sin(emb)
    if attention_scaling != 1.0:
        cosine = cosine * float(attention_scaling)
        sine = sine * float(attention_scaling)
    return cosine, sine


def _build_mrope_axes(section: list, interleaved: bool) -> list[int]:
    """Per-frequency axis assignment (0=t, 1=h, 2=w) over the rotary pairs.

    Interleaved layout is round-robin t,h,w while each axis has budget left
    (Qwen3.8-Flash-Next [11,11,10] -> t@0,3..30, h@1,4..31, w@2,5..29);
    non-interleaved is contiguous section blocks. Matches the reference
    Qwen-VL family layout (mlx-vlm / transformers).
    """
    remaining = [int(x) for x in section]
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
    return axes


def _mrope_cos_sin(
    positions3: mx.array, inv_freq: mx.array, axes: mx.array
) -> tuple[mx.array, mx.array]:
    """Rope tables for 3-axis (t, h, w) positions.

    ``positions3`` is [3, S] int32; ``axes`` maps each of the len(inv_freq)
    frequencies to its position axis. With equal axes this reduces exactly
    to ``_rope_cos_sin`` (the text case), which is why text-only serving
    never needs it.
    """
    pos = mx.take(positions3.astype(mx.float32), axes, axis=0)  # [F, S]
    angles = pos.transpose(1, 0) * inv_freq[None, :]  # [S, F]
    emb = mx.concatenate([angles, angles], axis=-1)
    return mx.cos(emb), mx.sin(emb)


def _apply_partial_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Rotate the first `2 * inv_freq.size` features of the last axis of
    x[..., S, H, D] with per-position tables cos/sin of shape [S, rot]."""
    rot = cos.shape[-1]
    x_rope = x[..., :rot]
    x_pass = x[..., rot:]
    half = rot // 2
    x1 = x_rope[..., :half]
    x2 = x_rope[..., half:]
    rotated = mx.concatenate([-x2, x1], axis=-1)
    if cos.ndim == 2:
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
    elif cos.ndim == 3:
        cos = cos[:, :, None, :]
        sin = sin[:, :, None, :]
    else:
        raise ValueError(f"RoPE tables must be rank 2 or 3, got {cos.ndim}")
    x_rope = (
        x_rope.astype(mx.float32) * cos + rotated.astype(mx.float32) * sin
    ).astype(x.dtype)
    return mx.concatenate([x_rope, x_pass], axis=-1)


class GroupedRMSNorm(nn.Module):
    """Qwen4 RMSNorm with zero-centered checkpoint weights."""

    def __init__(
        self,
        dims: int,
        group_size: int | None = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        if group_size is not None and dims % group_size:
            raise ValueError(
                f"dims ({dims}) not divisible by group_size ({group_size})"
            )
        self.weight = mx.zeros((dims,))
        self.group_size = group_size
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        normalized = x.astype(mx.float32)
        if self.group_size is not None:
            normalized = normalized.reshape(
                *normalized.shape[:-1],
                -1,
                self.group_size,
            )
            weight = self.weight.reshape(-1, self.group_size)
        else:
            weight = self.weight
        normalized = normalized * mx.rsqrt(
            mx.mean(mx.square(normalized), axis=-1, keepdims=True) + self.eps
        )
        normalized = normalized * (1.0 + weight.astype(mx.float32))
        return normalized.reshape(x.shape).astype(dtype)


_RMSNORM_CENTER_ANCHOR_RE = re.compile(
    r"^language_model\.model\.layers\.\d+" r"\.attn_hyper_connection\.hc_norm\.weight$"
)
_RMSNORM_CENTER_MIN_ANCHORS = 8


def _normalize_ones_centered_rmsnorm_weights(
    model: nn.Module,
    weights: dict[str, mx.array],
) -> None:
    """Convert legacy direct-gamma Qwen4 norms to residual weights."""

    anchors = [
        value
        for key, value in weights.items()
        if _RMSNORM_CENTER_ANCHOR_RE.fullmatch(key)
        and isinstance(value, mx.array)
        and mx.issubdtype(value.dtype, mx.floating)
    ]
    if len(anchors) < _RMSNORM_CENTER_MIN_ANCHORS:
        return

    means = mx.stack([mx.mean(value.astype(mx.float32)) for value in anchors])
    median = mx.median(means)
    ones_vote = mx.mean((means > 0.5).astype(mx.float32))
    mx.eval(median, ones_vote)
    if not (0.75 <= float(median.item()) <= 1.5):
        return
    if float(ones_vote.item()) < 0.9:
        return

    target_keys = {
        f"{path}.weight"
        for path, module in model.named_modules()
        if isinstance(module, GroupedRMSNorm)
    }
    for key in target_keys:
        value = weights.get(key)
        if not isinstance(value, mx.array) or not mx.issubdtype(
            value.dtype,
            mx.floating,
        ):
            continue
        weights[key] = value.astype(mx.float32) - 1.0


class SigmoidRMSNormGated(nn.Module):
    """GDN output norm with a sigmoid (not silu) gate — output_gate_type of
    this family. Stored one-centered; NOT +1-shifted in sanitize."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.eps = eps

    def __call__(self, hidden_states: mx.array, gate: mx.array | None = None):
        x = mx.fast.rms_norm(hidden_states, self.weight, self.eps)
        if gate is None:
            return x.astype(hidden_states.dtype)
        g = mx.sigmoid(gate.astype(mx.float32))
        return (g * x.astype(mx.float32)).astype(hidden_states.dtype)


class GatedDeltaNet(_Qwen3_5GatedDeltaNet):
    """qwen3_5's GDN with the family's output gate activation (sigmoid) and
    the reference q/k normalization.

    mlx-lm folds the attention scale through mx.fast.rms_norm, whose eps sits
    on mean(x²) — an effective d²·1e-6 on Σx² versus the reference FLA
    l2norm's d·1e-6 (transformers qwen3_5 l2norm: x·rsqrt(Σx²+1e-6)). At
    d=128 that skew is a measured, systematic ~1e-4-class divergence per
    layer (pinned by CPU-exact stage bisection, 2026-08-26), so this forward
    is mlx-lm's verbatim except the two q/k lines reproduce l2norm exactly.
    """

    def __init__(self, args: TextArgs):
        super().__init__(args)
        if getattr(args, "output_gate_type", "sigmoid") == "sigmoid":
            self.norm = SigmoidRMSNormGated(
                self.head_v_dim, eps=self.layer_norm_epsilon
            )

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        from mlx_lm.models.gated_delta import gated_delta_update

        B, S, _ = inputs.shape

        fused_in = getattr(self, "in_proj_fused", None)
        if fused_in is not None:
            qkv, z, b, a = fused_in(inputs)
            z = z.reshape(B, S, self.num_v_heads, self.head_v_dim)
        else:
            qkv = self.in_proj_qkv(inputs)
            z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
            b = self.in_proj_b(inputs)
            a = self.in_proj_a(inputs)

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim), dtype=inputs.dtype
            )

        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        if self._fused_step_applies(B, S, mask, cache):
            # One-dispatch GDN step: conv+silu+l2norm + g/beta + delta +
            # gated norm in a single kernel between the two library GEMVs.
            # Verify rows (S>1) never take this branch, so capture-commit's
            # stash contract below is untouched.
            from .tensor_support import fused_gdn_step

            y, new_conv, new_delta = fused_gdn_step(
                qkv.reshape(-1),
                z.reshape(-1),
                a.reshape(-1),
                b.reshape(-1),
                conv_state.reshape(self.conv_kernel_size - 1, self.conv_dim),
                self.conv1d.weight,
                self.A_log,
                self.dt_bias,
                cache[1],
                self.norm.weight,
            )
            cache[0] = new_conv.reshape(B, self.conv_kernel_size - 1, self.conv_dim)
            cache[1] = new_delta.reshape(
                B, self.num_v_heads, self.head_v_dim, self.head_k_dim
            )
            cache.advance(S)
            return self.out_proj(y.reshape(B, S, -1))
        if self._fused_conv_norm_applies(B, S, mask, cache):
            from .tensor_support import fused_gdn_conv_norm

            q_f, k_f, v_f, new_state = fused_gdn_conv_norm(
                qkv.reshape(-1),
                conv_state.reshape(self.conv_kernel_size - 1, self.conv_dim),
                self.conv1d.weight,
            )
            cache[0] = new_state.reshape(B, self.conv_kernel_size - 1, self.conv_dim)
            q = q_f.reshape(B, S, self.num_k_heads, self.head_k_dim)
            k = k_f.reshape(B, S, self.num_k_heads, self.head_k_dim)
            v = v_f.reshape(B, S, self.num_v_heads, self.head_v_dim)
            state = cache[1] if cache else None
        elif self._fused_conv_norm_rows_applies(B, S, mask, cache):
            from .tensor_support import fused_gdn_conv_norm_rows

            q_f, k_f, v_f, new_state = fused_gdn_conv_norm_rows(
                qkv.reshape(S, -1),
                conv_state.reshape(self.conv_kernel_size - 1, self.conv_dim),
                self.conv1d.weight,
            )
            cache[0] = new_state.reshape(B, self.conv_kernel_size - 1, self.conv_dim)
            q = q_f.reshape(B, S, self.num_k_heads, self.head_k_dim)
            k = k_f.reshape(B, S, self.num_k_heads, self.head_k_dim)
            v = v_f.reshape(B, S, self.num_v_heads, self.head_v_dim)
            state = cache[1] if cache else None
        else:
            conv_input = mx.concatenate([conv_state, qkv], axis=1)
            if cache is not None:
                n_keep = self.conv_kernel_size - 1
                if cache.lengths is not None:
                    ends = mx.clip(cache.lengths, 0, S)
                    positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                    cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
                else:
                    cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
            conv_out = nn.silu(self.conv1d(conv_input))

            q, k, v = [
                t.reshape(B, S, h, d)
                for t, h, d in zip(
                    mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                    [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                    [self.head_k_dim, self.head_k_dim, self.head_v_dim],
                    strict=False,
                )
            ]

            state = cache[1] if cache else None
            inv_scale = k.shape[-1] ** -0.5

            def _l2norm(x: mx.array) -> mx.array:
                xf = x.astype(mx.float32)
                return (xf * mx.rsqrt((xf * xf).sum(-1, keepdims=True) + 1e-6)).astype(
                    x.dtype
                )

            q = inv_scale * _l2norm(q)
            k = _l2norm(k)

        if cache is not None and _VERIFY_CAPTURE.get():
            # Family capture-commit: retain the exact rows gated_delta_update
            # consumed (plus the pre-conv stream for the conv-state tail) so a
            # rejected speculative window commits by replaying ONLY this
            # recurrence from the pre-verify state — no trunk re-forward.
            # These references are already materialized by this forward; at
            # mx.compile trace time they are tracers the compiled step
            # surfaces as extra outputs.
            cache._qwen4_exp_verify_rows = (qkv, q, k, v, a, b)

        out, state = gated_delta_update(
            q,
            k,
            v,
            a,
            b,
            self.A_log,
            self.dt_bias,
            state,
            mask,
            use_kernel=not self.training,
        )

        if cache is not None:
            cache[1] = state
            cache.advance(S)

        if self._fused_out_applies(B, S):
            from .tensor_support import fused_gdn_out

            proj = self.out_proj
            y = fused_gdn_out(
                out.reshape(-1),
                z.reshape(-1),
                self.norm.weight,
                proj.weight,
                proj.scales,
                proj.biases,
                group_size=int(proj.group_size),
            )
            return y.reshape(B, S, -1)

        out = self.norm(out, z)
        return self.out_proj(out.reshape(B, S, -1))

    def _fused_conv_norm_applies(self, B, S, mask, cache) -> bool:
        # Capability-gated fused conv+silu+l2norm for the decode row.
        # chain between the input GEMV and gated_delta_update. Family
        # geometry only (conv_dim 10240 / key_dim 2048 / heads of 128 — the
        # kernel's TG alignment depends on 2*key_dim being a threadgroup
        # multiple), dense rows, no conv bias, no ragged lengths. bf16
        # rounding happens after the norm instead of before (tolerance
        # class, same as the fallback re-forward's own noise).
        if B != 1 or S != 1 or mask is not None or cache is None:
            return False
        if not _fused_gdn_conv_norm_enabled() or self.training:
            return False
        if getattr(cache, "lengths", None) is not None:
            return False
        if self.conv_dim != 10240 or self.key_dim != 2048:
            return False
        if self.conv_kernel_size != 4 or self.head_k_dim != 128:
            return False
        if getattr(self.conv1d, "bias", None) is not None:
            return False
        from .tensor_support import device_supports_gdn_conv_norm

        # G14-class GPUs cap this 1024-thread pipeline below 1024, so the
        # capability's device gate must sit here. Cached one-shot probe.
        return device_supports_gdn_conv_norm()

    def _fused_conv_norm_rows_applies(self, B, S, mask, cache) -> bool:
        # Capability-gated verify-width fused conv+silu+l2norm.
        # same chain the S=1 kernel replaces, for speculative verify blocks
        # of 2..6 sequential rows. Deliberately ALLOWED under the capture
        # scope — the kernel produces exactly the q/k/v rows the
        # capture-commit stash retains, in the S=1 kernel's tolerance class.
        # The recurrence stays in the library gated_delta_update dispatch.
        if B != 1 or S < 2 or S > 6 or mask is not None or cache is None:
            return False
        if not _fused_conv_norm_rows_enabled() or self.training:
            return False
        if getattr(cache, "lengths", None) is not None:
            return False
        if self.conv_dim != 10240 or self.key_dim != 2048:
            return False
        if self.conv_kernel_size != 4 or self.head_k_dim != 128:
            return False
        if getattr(self.conv1d, "bias", None) is not None:
            return False
        from .tensor_support import device_supports_gdn_conv_norm_rows

        # Same G14 device gate as the S=1 kernel (issue #400).
        return device_supports_gdn_conv_norm_rows()

    def _fused_step_applies(self, B, S, mask, cache) -> bool:
        # Capability-gated one-dispatch GDN step for decode rows only.
        # family geometry, sigmoid-gated norm, live fp32 delta state. Mirrors
        # _fused_conv_norm_applies plus the recurrence/epilogue requirements;
        # anything else runs the staged chain.
        if B != 1 or S != 1 or mask is not None or cache is None:
            return False
        if not _fused_gdn_step_enabled() or self.training:
            return False
        if _VERIFY_CAPTURE.get():
            return False
        if getattr(cache, "lengths", None) is not None:
            return False
        if cache[1] is None:
            return False
        if self.conv_dim != 10240 or self.key_dim != 2048:
            return False
        if self.conv_kernel_size != 4 or self.head_k_dim != 128:
            return False
        if self.num_v_heads != 48 or self.head_v_dim != 128 or self.num_k_heads != 16:
            return False
        if getattr(self.conv1d, "bias", None) is not None:
            return False
        if not isinstance(self.norm, SigmoidRMSNormGated):
            return False
        return cache[1].dtype == mx.float32

    def _fused_out_applies(self, B: int, S: int) -> bool:
        # Capability-gated fused norm, gate, and output projection.
        # family geometry (48x128 values -> 2560), 4-bit affine out_proj at a
        # shipped forge group size, sigmoid-gated norm. Anything else runs
        # the stock chain. The capture-commit stash is upstream of this
        # boundary (it retains the gated_delta_update INPUTS), so the fused
        # output path is invisible to replay.
        if B * S != 1 or not _fused_gdn_out_enabled():
            return False
        if self.num_v_heads != 48 or self.head_v_dim != 128:
            return False
        if not isinstance(self.norm, SigmoidRMSNormGated):
            return False
        proj = self.out_proj
        return (
            getattr(proj, "bits", None) == 4
            and getattr(proj, "group_size", None) in (32, 64)
            and getattr(proj, "weight", None) is not None
            and proj.weight.dtype == mx.uint32
        )


class GatedResidual(nn.Module):
    """The Gated Residual read/write mixer (hyper-connections)."""

    def __init__(self, args: TextArgs, use_combine: bool = True):
        super().__init__()
        self.hc_count = args.hc_count
        self.hidden_size = args.hidden_size
        hc_hidden = self.hc_count * self.hidden_size
        self.hc_norm = GroupedRMSNorm(
            hc_hidden, args.hidden_size, eps=args.rms_norm_eps
        )
        self.input_mix_weight_down = nn.Linear(hc_hidden, args.hc_lowrank, bias=False)
        self.input_mix_weight_up = nn.Linear(args.hc_lowrank, hc_hidden, bias=False)
        if use_combine:
            self.block_inject_weight = nn.Linear(hc_hidden, self.hc_count, bias=False)

    def _fused_read_applies(self, hyper_input: mx.array) -> bool:
        # The fused kernel hardcodes the family geometry and reads bf16
        # module weights directly; anything else (quantized hc mixes, other
        # dims, prefill widths) stays on the eager chain.
        if not _fused_hc_enabled():
            return False
        if self.hc_count != 4 or self.hidden_size != 2560:
            return False
        down = self.input_mix_weight_down
        if hasattr(down, "scales") or down.weight.shape[0] != 320:
            return False
        if down.weight.dtype != hyper_input.dtype:
            return False
        rows = 1
        for s in hyper_input.shape[:-1]:
            rows *= s
        return 1 <= rows <= 8

    def _v3_read_applies(self, hyper_input: mx.array) -> bool:
        # v3 (two-dispatch, kernel-private 8-bit pack): single-row decode
        # reads on the combine variant with bf16 module weights. Verify
        # widths (rows 2..8) and prefill stay on the eager chain.
        if not _fused_hc_v3_enabled():
            return False
        if self.hc_count != 4 or self.hidden_size != 2560:
            return False
        if "block_inject_weight" not in self:
            return False
        if hasattr(self.input_mix_weight_down, "scales"):
            return False
        rows = 1
        for s in hyper_input.shape[:-1]:
            rows *= s
        if rows != 1:
            return False
        from .tensor_support import device_supports_hyper_v3, prepare_v3_pack

        # G14 device gate before paying for the pack (issue #400).
        if not device_supports_hyper_v3():
            return False
        if getattr(self, "_v3_pack", None) is None:
            self._v3_pack = prepare_v3_pack(self)
        return True

    def __call__(self, hyper_input: mx.array):
        if self._v3_read_applies(hyper_input):
            from .tensor_support import fused_hyper_read_v3

            x2 = hyper_input.reshape(-1)
            mixed, inject = fused_hyper_read_v3(x2, self.hc_norm.weight, self._v3_pack)
            mixed = mixed.reshape(*hyper_input.shape[:-1], self.hidden_size)
            inject = inject.reshape(*hyper_input.shape[:-1], self.hc_count)
            return mixed, hyper_input, inject
        if self._fused_read_applies(hyper_input):
            from .tensor_support import fused_hyper_read

            combine = "block_inject_weight" in self
            x2 = hyper_input.reshape(-1, self.hc_count * self.hidden_size)
            mixed, inject = fused_hyper_read(
                x2,
                self.hc_norm.weight,
                self.input_mix_weight_down.weight,
                self.input_mix_weight_up.weight,
                self.block_inject_weight.weight if combine else None,
            )
            mixed = mixed.reshape(*hyper_input.shape[:-1], self.hidden_size)
            if not combine:
                return mixed
            inject = inject.reshape(*hyper_input.shape[:-1], self.hc_count)
            return mixed, hyper_input, inject
        normed = self.hc_norm(hyper_input)
        mix = nn.silu(self.input_mix_weight_down(normed) / self.hc_count)
        mix = mx.sigmoid(self.input_mix_weight_up(mix))
        mix = mix.reshape(*mix.shape[:-1], self.hc_count, self.hidden_size)
        grouped = normed.reshape(*normed.shape[:-1], self.hc_count, self.hidden_size)
        mixed_input = mx.mean(mix * grouped, axis=-2)
        if "block_inject_weight" not in self:
            return mixed_input
        inject = 2.0 * mx.sigmoid(self.block_inject_weight(normed) / self.hc_count)
        return mixed_input, hyper_input, inject


class SparseMoeBlock(_Qwen3NextSparseMoeBlock):
    def __call__(self, x: mx.array) -> mx.array:
        # Capability-gated fused decode path plus sanitize-fused gate/up
        # weights): collapses gate_up -> GLU -> down -> weighted-sum into two
        # dispatches. Requires 4-bit affine at a shipped forge group size
        # (32 or 64 — the 2026-08-27 01:35 reforge moved the pack to g64);
        # anything else runs the stock chain.
        sw = self.switch_mlp
        if (
            x.shape[-2] == 1
            and x.size == x.shape[-1]  # B*S == 1
            and _fused_moe_decode_enabled()
            and isinstance(sw, _FusedGateUpSwitchGLU)
            and sw.bits == 4
            and sw.group_size in (32, 64)
            and getattr(sw.down_proj, "bits", None) == 4
            and getattr(sw.down_proj, "group_size", None) in (32, 64)
        ):
            from .tensor_support import moe_glu_decode

            flat = x.reshape(-1)
            # Routing math mirrors the parent exactly: softmax over ALL
            # experts first, then top-k of the probabilities.
            gates = mx.softmax(self.gate(x), axis=-1, precise=True)
            idx = mx.argpartition(gates, kth=-self.top_k, axis=-1)[..., -self.top_k :]
            w = mx.take_along_axis(gates, idx, axis=-1)
            if self.norm_topk_prob:
                w = w / w.sum(axis=-1, keepdims=True)
            dn = sw.down_proj
            y = moe_glu_decode(
                flat,
                sw.gu_weight,
                sw.gu_scales,
                sw.gu_biases,
                dn.weight,
                dn.scales,
                dn.biases,
                idx.reshape(-1).astype(mx.uint32),
                w.reshape(-1).astype(mx.float32),
                gu_group_size=int(sw.group_size),
                dn_group_size=int(dn.group_size),
            ).reshape(x.shape)
            shared = mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
            return (y + shared).astype(x.dtype)
        if (
            # Capability-gated fused verify path for the M=2..4
            # MTP verify forward pays the same per-layer dependency-gap
            # serialization the M=1 pair removed from AR decode, inside the
            # 23 ms verify_hidden_eval wall (round anatomy 2026-08-31). Same
            # kernels, M-batched grid: each token's rows are bit-identical
            # to the M=1 kernel on that row alone.
            x.ndim >= 2
            and 2 <= x.shape[-2] <= 4
            and x.size == x.shape[-2] * x.shape[-1]  # B == 1
            and _fused_moe_verify_enabled()
            and isinstance(sw, _FusedGateUpSwitchGLU)
            and sw.bits == 4
            and sw.group_size in (32, 64)
            and getattr(sw.down_proj, "bits", None) == 4
            and getattr(sw.down_proj, "group_size", None) in (32, 64)
        ):
            from .tensor_support import moe_glu_verify

            x2 = x.reshape(-1, x.shape[-1])
            gates = mx.softmax(self.gate(x), axis=-1, precise=True)
            idx = mx.argpartition(gates, kth=-self.top_k, axis=-1)[..., -self.top_k :]
            w = mx.take_along_axis(gates, idx, axis=-1)
            if self.norm_topk_prob:
                w = w / w.sum(axis=-1, keepdims=True)
            dn = sw.down_proj
            y = moe_glu_verify(
                x2,
                sw.gu_weight,
                sw.gu_scales,
                sw.gu_biases,
                dn.weight,
                dn.scales,
                dn.biases,
                idx.reshape(-1).astype(mx.uint32),
                w.reshape(-1).astype(mx.float32),
                gu_group_size=int(sw.group_size),
                dn_group_size=int(dn.group_size),
            ).reshape(x.shape)
            shared = mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
            return (y + shared).astype(x.dtype)
        return super().__call__(x)


class _FusedGateUpSwitchGLU(nn.Module):
    """SwitchGLU with gate_proj and up_proj concatenated into ONE
    gather_qmm (N=2*moe_intermediate) for the small-M decode/verify regime.

    Rationale (2026-08-26 attribution campaign): at qL=1 the MoE runs three
    gather_qmm dispatches per layer at N=640 — grids too small to fill the
    M5's 40 cores. Concatenating gate+up along the output-rows axis halves
    the large dispatches and doubles rows in flight. Per-row dot products
    are unchanged, so results match the split path up to within-row
    accumulation order. Large-M (prefill) calls fall through to the original
    SwitchGLU, keeping its expert-sorted access pattern."""

    def __init__(
        self, down_proj, gu_weight, gu_scales, gu_biases, group_size, bits, mode
    ):
        super().__init__()
        self.gu_weight = gu_weight
        self.gu_scales = gu_scales
        self.gu_biases = gu_biases
        self.group_size = group_size
        self.bits = bits
        self.mode = mode
        self.down_proj = down_proj
        # Built ONLY at sanitize time with placeholder params that strict
        # load_weights replaces by the lazily concatenated pack tensors —
        # the per-projection originals never materialize. A mid-session
        # module swap cannot reclaim their memory (freed tensors keep their
        # multi-GB safetensors shard buffers pinned via siblings; measured
        # +0.31G per fused module straight into a Metal OOM).

    def _gu(self, x, idx, sorted_indices=False):
        gu = mx.gather_qmm(
            x,
            self.gu_weight,
            self.gu_scales,
            self.gu_biases,
            rhs_indices=idx,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=sorted_indices,
        )
        return mx.split(gu, 2, axis=-1)

    def __call__(self, x, indices) -> mx.array:
        from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort

        x = mx.expand_dims(x, (-2, -3))
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        gate, up = self._gu(x, idx, sorted_indices=do_sort)
        x = self.down_proj(nn.silu(gate) * up, idx, sorted_indices=do_sort)
        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)
        return x.squeeze(-2)


class _FusedGateUpMLP(nn.Module):
    """Shared-expert MLP with gate_proj+up_proj as one quantized matmul
    (same fusion rationale and build-time contract as
    _FusedGateUpSwitchGLU; N=640 -> 1280)."""

    def __init__(
        self, down_proj, gu_weight, gu_scales, gu_biases, group_size, bits, mode
    ):
        super().__init__()
        self.gu_weight = gu_weight
        self.gu_scales = gu_scales
        self.gu_biases = gu_biases
        self.group_size = group_size
        self.bits = bits
        self.mode = mode
        self.down_proj = down_proj

    def __call__(self, x) -> mx.array:
        gu = mx.quantized_matmul(
            x,
            self.gu_weight,
            self.gu_scales,
            self.gu_biases,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )
        gate, up = mx.split(gu, 2, axis=-1)
        return self.down_proj(nn.silu(gate) * up)


class _FusedGDNInProj(nn.Module):
    """GDN qkv/z/b/a input projections as ONE quantized matmul.

    All four share the layer input row; at qL=1 they are four separate GEMV
    dispatches per GDN layer (35 layers = 140 dispatches/step). Row-axis
    concat of quantized packs is bit-exact per output row — each row's dot
    and its quant groups are unchanged — so the fused output just splits at
    the recorded row offsets. Same placeholder-at-build/load-fills contract
    as _FusedGateUpSwitchGLU."""

    def __init__(self, weight, scales, biases, group_size, bits, mode, splits):
        super().__init__()
        self.weight = weight
        self.scales = scales
        self.biases = biases
        self.group_size = group_size
        self.bits = bits
        self.mode = mode
        self._splits = list(splits)  # cumulative row offsets: qkv|z|b|a

    def __call__(self, x):
        y = mx.quantized_matmul(
            x,
            self.weight,
            self.scales,
            self.biases,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )
        return mx.split(y, self._splits, axis=-1)


_LAYER_GDN_RE = re.compile(r"^(.*\.layers\.(\d+)\.linear_attn)\.in_proj_qkv\.weight$")
_LAYER_MLP_RE = re.compile(r"^(.*\.layers\.(\d+)\.mlp)\.switch_mlp\.gate_proj\.weight$")
_GDN_IN_PROJS = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")


def _fuse_gdn_in_proj_sanitize(model, out: dict) -> dict:
    """Capability-gated sanitize-time GDN input fusion.

    Concatenates the four quantized input projections of every GDN layer
    along the output-rows axis on the LAZY weight dict (originals never
    materialize) and swaps in a _FusedGDNInProj child. Fuses only when all
    four are affine-quantized at one (group_size, bits); anything else keeps
    the stock modules."""
    if not _fused_gdn_in_proj_enabled():
        return out
    hits = [
        (m.group(1), int(m.group(2)))
        for m in (_LAYER_GDN_RE.match(k) for k in list(out))
        if m is not None
    ]
    for prefix, idx in hits:
        parts = []
        for sub in _GDN_IN_PROJS:
            w = out.get(f"{prefix}.{sub}.weight")
            s = out.get(f"{prefix}.{sub}.scales")
            b = out.get(f"{prefix}.{sub}.biases")
            if w is None or s is None or b is None:
                parts = None
                break
            parts.append((w, s, b))
        if parts is None:
            continue
        k_words = parts[0][0].shape[-1]
        n_groups = parts[0][1].shape[-1]
        if any(w.shape[-1] != k_words or s.shape[-1] != n_groups for w, s, _ in parts):
            continue  # mixed packing: stay stock
        gdn = model.layers[idx].linear_attn
        k_in = model.args.hidden_size
        group_size = k_in // n_groups
        bits = (k_words * 32) // k_in
        if bits not in (4, 8):
            continue
        rows = [w.shape[0] for w, _, _ in parts]
        splits = [rows[0], rows[0] + rows[1], rows[0] + rows[1] + rows[2]]
        f_w = mx.concatenate([w for w, _, _ in parts], axis=0)
        f_s = mx.concatenate([s for _, s, _ in parts], axis=0)
        f_b = mx.concatenate([b for _, _, b in parts], axis=0)
        gdn.in_proj_fused = _FusedGDNInProj(
            mx.zeros(f_w.shape, dtype=f_w.dtype),
            mx.zeros(f_s.shape, dtype=f_s.dtype),
            mx.zeros(f_b.shape, dtype=f_b.dtype),
            group_size,
            bits,
            "affine",
            splits,
        )
        for sub in _GDN_IN_PROJS:
            gdn.pop(sub, None)
        out[f"{prefix}.in_proj_fused.weight"] = f_w
        out[f"{prefix}.in_proj_fused.scales"] = f_s
        out[f"{prefix}.in_proj_fused.biases"] = f_b
        for sub in _GDN_IN_PROJS:
            for part in ("weight", "scales", "biases"):
                out.pop(f"{prefix}.{sub}.{part}", None)
    return out


_LAYER_ATTN_RE = re.compile(r"^(.*\.layers\.(\d+)\.self_attn)\.q_proj\.weight$")
_QSA_QKV_PROJS = ("q_proj", "k_proj", "v_proj", "indexer.index_qk_proj")


def _fuse_qsa_qkv_sanitize(model, out: dict) -> dict:
    """Capability-gated sanitize-time QSA attention input fusion.

    q/k/v and the indexer's qk projection all consume the layer input row;
    row-axis concat of the quantized packs is bit-exact per output row (same
    contract as the GDN in_proj fusion), so the 13 attention layers run one
    shared-input GEMV instead of four. Biased checkpoints and mixed packings
    keep the stock chain."""
    if not _fused_qsa_qkv_enabled():
        return out
    hits = [
        (m.group(1), int(m.group(2)))
        for m in (_LAYER_ATTN_RE.match(k) for k in list(out))
        if m is not None
    ]
    for prefix, idx in hits:
        if any(f"{prefix}.{p}.bias" in out for p in _QSA_QKV_PROJS):
            continue
        parts = []
        fused_subs = []
        for sub in _QSA_QKV_PROJS:
            w = out.get(f"{prefix}.{sub}.weight")
            s = out.get(f"{prefix}.{sub}.scales")
            b = out.get(f"{prefix}.{sub}.biases")
            if w is None or s is None or b is None:
                if sub == "indexer.index_qk_proj":
                    continue  # optional member
                parts = None
                break
            parts.append((w, s, b))
            fused_subs.append(sub)
        if parts is None:
            continue
        k_words = parts[0][0].shape[-1]
        n_groups = parts[0][1].shape[-1]
        if any(
            w.shape[-1] != k_words or s.shape[-1] != n_groups for w, s, _ in parts[:3]
        ):
            continue
        # Include the indexer projection only when its actual pack geometry
        # matches q/k/v; never infer this from names or config.  The v2.10
        # production artifact is 4-bit group-64 here, despite the stale older
        # 8-bit artifact assumption.
        if len(parts) == 4 and (
            parts[3][0].shape[-1] != k_words or parts[3][1].shape[-1] != n_groups
        ):
            parts = parts[:3]
            fused_subs = fused_subs[:3]
        attn = model.layers[idx].self_attn
        if getattr(attn, "indexer", None) is None:
            continue
        k_in = model.args.hidden_size
        group_size = k_in // n_groups
        bits = (k_words * 32) // k_in
        if bits not in (4, 8):
            continue
        rows = [w.shape[0] for w, _, _ in parts]
        splits = [sum(rows[: i + 1]) for i in range(len(rows) - 1)]
        f_w = mx.concatenate([w for w, _, _ in parts], axis=0)
        f_s = mx.concatenate([s for _, s, _ in parts], axis=0)
        f_b = mx.concatenate([b for _, _, b in parts], axis=0)
        attn.qkv_fused = _FusedGDNInProj(
            mx.zeros(f_w.shape, dtype=f_w.dtype),
            mx.zeros(f_s.shape, dtype=f_s.dtype),
            mx.zeros(f_b.shape, dtype=f_b.dtype),
            group_size,
            bits,
            "affine",
            splits,
        )
        for name in ("q_proj", "k_proj", "v_proj"):
            attn.pop(name, None)
        if "indexer.index_qk_proj" in fused_subs:
            attn.indexer.pop("index_qk_proj", None)
        out[f"{prefix}.qkv_fused.weight"] = f_w
        out[f"{prefix}.qkv_fused.scales"] = f_s
        out[f"{prefix}.qkv_fused.biases"] = f_b
        for sub in fused_subs:
            for part in ("weight", "scales", "biases"):
                out.pop(f"{prefix}.{sub}.{part}", None)
    return out


def _capability(name: str) -> bool:
    return current_tensor_capabilities().has(name)


def _fused_gate_up_enabled() -> bool:
    return _capability("fused_gate_up")


def _fused_qsa_qkv_enabled() -> bool:
    return _capability("fused_qsa_qkv")


def _fused_gdn_in_proj_enabled() -> bool:
    return _capability("fused_gdn_in_proj")


def _fused_gdn_out_enabled() -> bool:
    return _capability("fused_gdn_out")


def _fused_gdn_conv_norm_enabled() -> bool:
    return _capability("fused_gdn_conv_norm")


def _fused_gdn_step_enabled() -> bool:
    return _capability("fused_gdn_step")


def _fused_conv_norm_rows_enabled() -> bool:
    return _capability("fused_gdn_conv_norm_rows")


def _qsa_gather_enabled() -> bool:
    return _capability("qsa_gather")


def _qsa_gather_decode_enabled() -> bool:
    return _capability("qsa_gather_decode")


def _qsa_gather_min_context() -> int:
    return current_tensor_capabilities().qsa_gather_min_context


def _qsa_gather_max_rows() -> int:
    return current_tensor_capabilities().qsa_gather_max_rows


def _qsa_flash_enabled() -> bool:
    return _capability("qsa_flash_skip")


def _qsa_score_tile_rows() -> int:
    return current_tensor_capabilities().qsa_score_tile_rows


def _fused_qsa_indexer_enabled() -> bool:
    return _capability("qsa_indexer_select")


def _compiled_qsa_indexer_enabled() -> bool:
    return _capability("qsa_compiled_indexer")


def qsa_prefill_lane_auto_supported() -> bool:
    return _capability("qsa_prefill_auto")


def _qsa_prefill_enabled() -> bool:
    return _capability("qsa_prefill")


def _qsa_prefill_min_rows() -> int:
    return current_tensor_capabilities().qsa_prefill_min_rows


def _qsa_prefill_min_context() -> int:
    return current_tensor_capabilities().qsa_prefill_min_context


def _qsa_prefill_flash_min_context() -> int:
    return current_tensor_capabilities().qsa_prefill_flash_min_context


def _qsa_prefill_score_workspace_bytes() -> int:
    return current_tensor_capabilities().qsa_prefill_score_workspace_bytes


def _qsa_prefill_compile_rows() -> int:
    return current_tensor_capabilities().qsa_prefill_compile_rows


def _qsa_large_prefill_enabled(rows: int, total_tokens: int) -> bool:
    return (
        _qsa_prefill_enabled()
        and rows >= _qsa_prefill_min_rows()
        and total_tokens >= _qsa_prefill_min_context()
    )


def _qsa_prefill_flash_attention_enabled(rows: int, total_tokens: int) -> bool:
    return (
        _capability("qsa_prefill_flash")
        and rows >= _qsa_prefill_min_rows()
        and total_tokens >= _qsa_prefill_flash_min_context()
    )


_QSA_PREFILL_ENGAGEMENT: dict[str, int] = {}


def _qsa_prefill_count(lane: str) -> None:
    _QSA_PREFILL_ENGAGEMENT[lane] = _QSA_PREFILL_ENGAGEMENT.get(lane, 0) + 1


def qsa_prefill_engagement() -> dict[str, int]:
    return dict(_QSA_PREFILL_ENGAGEMENT)


def _qsa_prefill_gather_enabled() -> bool:
    return _capability("qsa_prefill_gather")


def _qsa_prefill_gather_tile_rows() -> int:
    return current_tensor_capabilities().qsa_prefill_gather_tile_rows


def _fused_hc_enabled() -> bool:
    return _capability("fused_hyper_read")


def _fused_hc_v3_enabled() -> bool:
    return _capability("fused_hyper_read_v3")


def _fused_moe_decode_enabled() -> bool:
    return _capability("moe_glu_decode")


def _fused_moe_verify_enabled() -> bool:
    return _capability("moe_glu_verify")


def _fuse_gate_up_sanitize(model, out: dict) -> dict:
    """Capability-gated sanitize-time MoE gate and up fusion.

    Runs on the LAZY, file-backed weight dict before quantize/load: swaps
    each layer's switch_mlp and shared_expert modules for the fused
    variants (placeholder params), moves the pack tensors into the dict as
    lazy concatenations under the fused names, and drops the per-projection
    keys. Materialization then only ever builds fused buffers — the split
    originals never come off the shards. down_proj children keep their
    stock modules and tree paths, so the quantize predicate and strict
    load_weights treat them exactly as before."""
    if not _fused_gate_up_enabled():
        return out
    layer_hits = [
        (m.group(1), int(m.group(2)))
        for m in (_LAYER_MLP_RE.match(k) for k in list(out))
        if m is not None
    ]
    for prefix, idx in layer_hits:
        layer = model.layers[idx]
        for sub, cat_axis in (("switch_mlp", 1), ("shared_expert", 0)):
            base = f"{prefix}.{sub}"
            gw = out.get(f"{base}.gate_proj.weight")
            uw = out.get(f"{base}.up_proj.weight")
            gs = out.get(f"{base}.gate_proj.scales")
            us = out.get(f"{base}.up_proj.scales")
            gb = out.get(f"{base}.gate_proj.biases")
            ub = out.get(f"{base}.up_proj.biases")
            if gw is None or uw is None or gs is None or us is None:
                continue  # bf16/tiny checkpoints: stock path
            if (gb is None) != (ub is None):
                continue
            k_in = model.args.hidden_size
            group_size = k_in // gs.shape[-1]
            bits = (gw.shape[-1] * 32) // k_in
            if gb is None:
                continue  # non-affine packing: unknown mode, stay stock
            mod = getattr(layer.mlp, sub)
            cls = _FusedGateUpSwitchGLU if sub == "switch_mlp" else _FusedGateUpMLP
            gu_w = mx.concatenate([gw, uw], axis=cat_axis)
            gu_s = mx.concatenate([gs, us], axis=cat_axis)
            gu_b = mx.concatenate([gb, ub], axis=cat_axis)
            setattr(
                layer.mlp,
                sub,
                cls(
                    mod.down_proj,
                    mx.zeros(gu_w.shape, dtype=gu_w.dtype),
                    mx.zeros(gu_s.shape, dtype=gu_s.dtype),
                    mx.zeros(gu_b.shape, dtype=gu_b.dtype),
                    group_size,
                    bits,
                    "affine",
                ),
            )
            out[f"{base}.gu_weight"] = gu_w
            out[f"{base}.gu_scales"] = gu_s
            out[f"{base}.gu_biases"] = gu_b
            for proj in ("gate_proj", "up_proj"):
                for part in ("weight", "scales", "biases"):
                    out.pop(f"{base}.{proj}.{part}", None)
    return out


# Armed around a speculative verify forward: GDN layers retain the exact
# recurrence rows so a rejected window commits by replaying only the
# gated-delta recurrence (see Qwen4ExpTextModel.commit_verified_window).
_VERIFY_CAPTURE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "qwen4_exp_verify_capture", default=False
)


@contextlib.contextmanager
def verify_capture_scope():
    with attention_phase_scope("verify"):
        token = _VERIFY_CAPTURE.set(True)
        try:
            yield
        finally:
            _VERIFY_CAPTURE.reset(token)


class QSACache:
    """Cache for one QSA layer: the attention KV plus the indexer's raw key
    stream and the incrementally maintained pooled (mean->norm->rope) block
    keys. Single-sequence.

    The raw/pooled streams are POSITIONAL buffers keyed to ``kv.offset``, not
    append-only logs: every write lands at the absolute row range of the
    tokens being forwarded and the valid lengths derive from ``kv.offset``.
    That keeps the indexer streams in lockstep with the KV through every
    mutation the runtime performs — per-round speculative rollback,
    verified-window trims, and session-bank ``state`` round-trips — which
    also makes the layer trimmable like a plain ``KVCache`` instead of being
    deep-cloned every verify round. (Append-only streams desynced from the
    KV on the first rollback and, past the indexer's engage threshold, built
    a selection mask longer than the KV: the 2026-08-27 OpenCode
    ``broadcast_shapes (1,1,4,3719) vs (1,24,4,3715)`` crash.)"""

    step = 256

    def __init__(self, compress_ratio: int = 4):
        self.kv = KVCache()
        self.ratio = max(1, int(compress_ratio))
        self.raw_keys: mx.array | None = None  # [1, cap, index_head_dim]
        self.pooled: mx.array | None = None  # [1, cap_blocks, index_head_dim]
        self.pooled_len = 0  # valid pooled blocks
        # fp32-transposed mirror of ``pooled`` [1, 1, D, cap_blocks], kept in
        # lockstep by write_pooled. The indexer used to upcast + transpose the
        # ENTIRE pooled table on every forward of every QSA layer — 33.5 MB
        # allocated and freed per layer per decoded token at 262K context
        # (#393 audit). Same values (astype of the same bf16 blocks), so
        # selection is bit-identical; this is allocation hygiene only.
        self.pooled_f32_t: mx.array | None = None
        # Host-planned graph buckets can be reserved before the first array
        # write, when dtype/head width are not known yet.  The next write
        # materializes the pending capacity with the actual projected dtype.
        self._reserved_raw_capacity = 0
        self._reserved_pooled_capacity = 0

    @property
    def offset(self) -> int:
        return self.kv.offset

    @staticmethod
    def _grown_cap(end: int, current: int, step: int) -> int:
        """Geometric (doubling) growth, step-aligned.

        The previous fixed +``step`` growth full-copied the buffer every 256
        rows: Θ(N²) memcpy — ~34 GB of pure copy traffic per QSA layer over
        a 262K decode, times 13 caches (#393 audit). Doubling bounds total
        copy traffic at O(N) with at most 2x capacity overshoot; ``nbytes``
        keeps reporting real capacity so memory accounting stays honest."""
        cap = ((end + step - 1) // step) * step
        return max(cap, 2 * current)

    def write_raw(self, keys: mx.array) -> None:
        """Store this forward's indexer keys at their absolute positions.

        Called before ``kv.update_and_fetch`` advances the offset, so
        ``kv.offset`` IS the absolute position of ``keys[:, 0]``. After a
        trim the same positions are simply overwritten."""
        start = self.kv.offset
        end = start + keys.shape[1]
        if self.raw_keys is None or end > self.raw_keys.shape[1]:
            current = 0 if self.raw_keys is None else self.raw_keys.shape[1]
            # Geometric growth bounds copy traffic; a staged host reservation
            # (compiled-indexer graph buckets) may demand a wider backing.
            cap = max(
                self._grown_cap(end, current, self.step), self._reserved_raw_capacity
            )
            grown = mx.zeros((1, cap, keys.shape[2]), keys.dtype)
            if self.raw_keys is not None:
                grown[:, : self.raw_keys.shape[1], :] = self.raw_keys
            self.raw_keys = grown
        self.raw_keys[:, start:end, :] = keys

    def write_pooled(self, blocks: mx.array, nb_start: int, nb_total: int) -> None:
        if self.pooled is None or nb_total > self.pooled.shape[1]:
            current = 0 if self.pooled is None else self.pooled.shape[1]
            cap = max(
                self._grown_cap(nb_total, current, self.step),
                self._reserved_pooled_capacity,
            )
            grown = mx.zeros((1, cap, blocks.shape[2]), blocks.dtype)
            if self.pooled is not None:
                grown[:, : self.pooled.shape[1], :] = self.pooled
            self.pooled = grown
        self.pooled[:, nb_start:nb_total, :] = blocks
        # Keep the fp32-transposed mirror in lockstep (same capacity). When
        # the mirror is absent (fresh cache, or dropped by a state restore)
        # it must seed from the pooled buffer's CONTENT — zeros would blank
        # every previously valid block's scores (caught by the state
        # round-trip gate).
        cap_blocks = self.pooled.shape[1]
        if self.pooled_f32_t is None:
            self.pooled_f32_t = mx.swapaxes(self.pooled.astype(mx.float32), 1, 2)[
                :, None
            ]
        elif self.pooled_f32_t.shape[3] < cap_blocks:
            grown_t = mx.zeros((1, 1, blocks.shape[2], cap_blocks), mx.float32)
            grown_t[..., : self.pooled_f32_t.shape[3]] = self.pooled_f32_t
            self.pooled_f32_t = grown_t
            self.pooled_f32_t[..., nb_start:nb_total] = mx.swapaxes(
                blocks.astype(mx.float32), 1, 2
            )[:, None]
        else:
            self.pooled_f32_t[..., nb_start:nb_total] = mx.swapaxes(
                blocks.astype(mx.float32), 1, 2
            )[:, None]
        self.pooled_len = nb_total

    def pooled_f32_view(self, nb: int) -> mx.array:
        """[1, 1, D, nb] fp32 view of the valid pooled blocks.

        Rebuilds the mirror from ``pooled`` after a state restore (setter
        drops it) or a compiled-indexer commit (which replaces ``pooled``
        wholesale and nulls the mirror); otherwise a zero-copy slice of the
        maintained buffer."""
        if self.pooled_f32_t is None or self.pooled_f32_t.shape[3] < nb:
            self.pooled_f32_t = mx.swapaxes(self.pooled.astype(mx.float32), 1, 2)[
                :, None
            ]
        return self.pooled_f32_t[..., :nb]

    def reserve_indexer_capacity(
        self,
        *,
        raw_capacity: int,
        pooled_capacity: int,
    ) -> None:
        """Reserve fixed backing shapes before an indexer graph is traced.

        The MTP replay planner calls this on the host.  Existing allocations
        grow immediately and retain their active prefix; a pristine cache
        records the request until its first projected rows establish dtype and
        head width.  No reservation may truncate a live logical frontier.
        """

        raw_requested = int(raw_capacity)
        pooled_requested = int(pooled_capacity)
        if raw_requested < 0 or pooled_requested < 0:
            raise ValueError(
                "QSA reserved capacities must be non-negative; got "
                f"raw={raw_requested}, pooled={pooled_requested}"
            )

        raw_existing = 0 if self.raw_keys is None else int(self.raw_keys.shape[1])
        pooled_existing = 0 if self.pooled is None else int(self.pooled.shape[1])
        raw_target = max(
            raw_requested,
            raw_existing,
            self._reserved_raw_capacity,
        )
        pooled_target = max(
            pooled_requested,
            pooled_existing,
            self._reserved_pooled_capacity,
        )
        if raw_target < self.offset:
            raise ValueError(
                f"raw capacity {raw_target} cannot cover QSA offset {self.offset}"
            )
        if pooled_target < self.pooled_len:
            raise ValueError(
                "pooled capacity cannot truncate the valid QSA frontier: "
                f"{pooled_target} < {self.pooled_len}"
            )

        self._reserved_raw_capacity = raw_target
        self._reserved_pooled_capacity = pooled_target
        if self.raw_keys is not None and raw_target > raw_existing:
            grown = mx.zeros(
                (1, raw_target, self.raw_keys.shape[2]),
                self.raw_keys.dtype,
            )
            grown[:, :raw_existing, :] = self.raw_keys
            self.raw_keys = grown
        if self.pooled is not None and pooled_target > pooled_existing:
            grown = mx.zeros(
                (1, pooled_target, self.pooled.shape[2]),
                self.pooled.dtype,
            )
            grown[:, :pooled_existing, :] = self.pooled
            self.pooled = grown

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        trimmed = self.kv.trim(n)
        # Pooled blocks past the new frontier were built from now-rejected
        # rows. The raw buffer needs no touch: future writes land at the same
        # absolute positions and overwrite.
        self.pooled_len = min(self.pooled_len, self.kv.offset // self.ratio)
        return trimmed

    @property
    def nbytes(self) -> int:
        total = self.kv.nbytes
        if self.raw_keys is not None:
            total += self.raw_keys.nbytes
        if self.pooled is not None:
            total += self.pooled.nbytes
        return total

    @property
    def state(self):
        off = self.kv.offset
        nb = min(self.pooled_len, off // self.ratio)
        raw = None if self.raw_keys is None else self.raw_keys[:, :off, :]
        pooled = None if self.pooled is None or nb == 0 else self.pooled[:, :nb, :]
        return (*self.kv.state, raw, pooled)

    @state.setter
    def state(self, v):
        if len(v) != 4:
            raise ValueError(
                "QSACache.state expects (keys, values, raw_keys, pooled); got "
                f"{len(v)} leaves — a session snapshot from an older build; "
                "drop it and re-prefill"
            )
        keys, values, raw, pooled = v
        self.kv.state = (keys, values)
        self.raw_keys = raw
        self.pooled = pooled
        self.pooled_len = 0 if pooled is None else pooled.shape[1]
        # Derived mirror: rebuilt lazily on the first pooled_f32_view read.
        # Restored snapshots stay 4-leaf — the state contract is unchanged.
        self.pooled_f32_t = None
        self._reserved_raw_capacity = 0 if raw is None else int(raw.shape[1])
        self._reserved_pooled_capacity = 0 if pooled is None else int(pooled.shape[1])


class _QSAKVBatch:
    """Transient right-padded KV view over independent QSA row caches.

    Logical offsets stay on the host as one integer per row. The physical
    tensor is shared only for the duration of one model forward; every write
    is scattered at that row's own frontier and the owner cache is restored
    immediately afterwards.
    """

    def __init__(self, rows: Sequence[QSACache]) -> None:
        if not rows:
            raise ValueError("QSA batch requires at least one row")
        self._rows = tuple(rows)
        self.offsets = tuple(int(row.kv.offset) for row in rows)
        self.keys = self._pack_leaf("keys")
        self.values = self._pack_leaf("values")

    def _pack_leaf(self, name: str) -> mx.array | None:
        source = next(
            (
                getattr(row.kv, name)
                for row in self._rows
                if getattr(row.kv, name) is not None
            ),
            None,
        )
        if source is None:
            if any(self.offsets):
                raise RuntimeError("non-empty QSA row is missing its KV backing")
            return None
        width = max(self.offsets)
        packed = mx.zeros(
            (len(self._rows), source.shape[1], width, source.shape[3]),
            dtype=source.dtype,
        )
        for index, (row, offset) in enumerate(
            zip(self._rows, self.offsets, strict=True)
        ):
            leaf = getattr(row.kv, name)
            if offset:
                if leaf is None:
                    raise RuntimeError("non-empty QSA row is missing its KV backing")
                packed[index : index + 1, :, :offset, :] = leaf[..., :offset, :]
        return packed

    def update_and_fetch(
        self,
        keys: mx.array,
        values: mx.array,
    ) -> tuple[mx.array, mx.array]:
        batch, heads, rows, key_width = keys.shape
        if batch != len(self.offsets):
            raise ValueError("QSA KV update batch size changed")
        starts = self.offsets
        ends = tuple(offset + rows for offset in starts)
        capacity = max(ends)

        if self.keys is None:
            self.keys = mx.zeros((batch, heads, capacity, key_width), keys.dtype)
            self.values = mx.zeros(
                (batch, values.shape[1], capacity, values.shape[3]),
                values.dtype,
            )
        elif capacity > int(self.keys.shape[2]):
            grown_keys = mx.zeros((batch, heads, capacity, key_width), keys.dtype)
            grown_values = mx.zeros(
                (batch, values.shape[1], capacity, values.shape[3]),
                values.dtype,
            )
            grown_keys[..., : self.keys.shape[2], :] = self.keys
            grown_values[..., : self.values.shape[2], :] = self.values
            self.keys = grown_keys
            self.values = grown_values

        indices = mx.array(starts, dtype=mx.int32)[:, None, None, None]
        indices = indices + mx.arange(rows, dtype=mx.int32)[None, None, :, None]
        key_indices = mx.broadcast_to(indices, keys.shape)
        value_indices = mx.broadcast_to(indices, values.shape)
        self.keys = mx.put_along_axis(self.keys, key_indices, keys, axis=2)
        self.values = mx.put_along_axis(self.values, value_indices, values, axis=2)
        self.offsets = ends
        return self.keys[..., :capacity, :], self.values[..., :capacity, :]


class _QSABatchCache:
    """Ephemeral ragged QSA cache used by one multi-row tensor call."""

    def __init__(self, rows: Sequence[QSACache]) -> None:
        if not rows:
            raise ValueError("QSA batch requires at least one row")
        ratios = {row.ratio for row in rows}
        if len(ratios) != 1:
            raise ValueError("QSA rows must share one compression ratio")
        self._rows = tuple(rows)
        self.ratio = ratios.pop()
        self.kv = _QSAKVBatch(rows)
        self.raw_keys = self._pack_raw()
        self.pooled: mx.array | None = None

    @property
    def offsets(self) -> tuple[int, ...]:
        return self.kv.offsets

    def _pack_raw(self) -> mx.array | None:
        source = next(
            (row.raw_keys for row in self._rows if row.raw_keys is not None),
            None,
        )
        if source is None:
            if any(self.offsets):
                raise RuntimeError("non-empty QSA row is missing indexer keys")
            return None
        width = max(self.offsets)
        packed = mx.zeros(
            (len(self._rows), width, source.shape[2]),
            dtype=source.dtype,
        )
        for index, (row, offset) in enumerate(
            zip(self._rows, self.offsets, strict=True)
        ):
            if offset:
                if row.raw_keys is None:
                    raise RuntimeError("non-empty QSA row is missing indexer keys")
                packed[index : index + 1, :offset, :] = row.raw_keys[:, :offset, :]
        return packed

    def write_raw(self, keys: mx.array) -> None:
        batch, rows, width = keys.shape
        if batch != len(self.offsets):
            raise ValueError("QSA index update batch size changed")
        starts = self.offsets
        capacity = max(offset + rows for offset in starts)
        if self.raw_keys is None:
            self.raw_keys = mx.zeros((batch, capacity, width), keys.dtype)
        elif capacity > int(self.raw_keys.shape[1]):
            grown = mx.zeros((batch, capacity, width), keys.dtype)
            grown[:, : self.raw_keys.shape[1], :] = self.raw_keys
            self.raw_keys = grown
        indices = mx.array(starts, dtype=mx.int32)[:, None, None]
        indices = indices + mx.arange(rows, dtype=mx.int32)[None, :, None]
        indices = mx.broadcast_to(indices, keys.shape)
        self.raw_keys = mx.put_along_axis(self.raw_keys, indices, keys, axis=1)

    def scatter_rows(self) -> None:
        """Return the evaluated batch leaves to their identity-stable owners."""

        for index, (row, offset) in enumerate(
            zip(self._rows, self.offsets, strict=True)
        ):
            row.kv.keys = self.kv.keys[index : index + 1, :, :offset, :]
            row.kv.values = self.kv.values[index : index + 1, :, :offset, :]
            row.kv.offset = offset
            row.raw_keys = self.raw_keys[index : index + 1, :offset, :]
            pooled_len = offset // self.ratio
            row.pooled = (
                None
                if pooled_len == 0 or self.pooled is None
                else self.pooled[index : index + 1, :pooled_len, :]
            )
            row.pooled_len = pooled_len
            row.pooled_f32_t = None
            row._reserved_raw_capacity = offset
            row._reserved_pooled_capacity = pooled_len


class QSAIndexer(nn.Module):
    """Vectorized exact port of the reference indexer for the single-sequence
    causal case (B=1, no padding): every query selects its top
    (budget/compress_ratio) complete key blocks by relu-scored pooled keys,
    plus the visible incomplete tail."""

    # The selector carries one private float32 score row per pooled backing
    # block. Bound that hidden output per Metal dispatch without imposing any
    # cap on logical history length. An irreducible one-row dispatch can exceed
    # the target at extreme capacities; MLX may also retain multiple lazy
    # chunks until their concatenated consumer is evaluated, so this is not a
    # claim about graph-wide peak memory.
    _fused_score_scratch_bytes = 32 * 1024 * 1024

    def __init__(self, args: TextArgs):
        super().__init__()
        self.n_heads = args.indexer_n_heads
        self.kv_heads = args.indexer_kv_heads
        self.head_dim = args.indexer_head_dim
        self.budget = args.indexer_budget
        self.ratio = args.indexer_compress_ratio
        self.block_topk = self.budget // self.ratio
        self.rms_norm_eps = float(args.rms_norm_eps)
        self.index_qk_proj = nn.Linear(
            args.hidden_size, (self.n_heads + self.kv_heads) * self.head_dim, bias=False
        )
        self.q_layernorm = GroupedRMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_layernorm = GroupedRMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self._inv_freq, self._rope_attention_scaling = _rope_inv_freq_and_scaling(args)
        # Kept outside the nn.Module parameter tree.  The graph bank is built
        # lazily on the first eligible inference call, after checkpoint load
        # and sanitize-time projection fusion have finalized every weight.
        object.__setattr__(self, "_compiled_indexer_core", None)
        object.__setattr__(self, "_compiled_indexer_parameter_signature", None)

    def _pool_keys_eager(
        self,
        fresh: mx.array,
        nb_old: int,
        nb_total: int,
    ) -> mx.array:
        """Stock completed-block preparation kept as the numeric oracle."""

        fresh = fresh.reshape(1, nb_total - nb_old, self.ratio, self.head_dim)
        pooled = mx.mean(fresh.astype(mx.float32), axis=2).astype(fresh.dtype)
        pooled = self.k_layernorm(pooled)
        starts = mx.arange(nb_old, nb_total, dtype=mx.int32) * self.ratio
        cos, sin = _rope_cos_sin(
            starts,
            self._inv_freq,
            self._rope_attention_scaling,
        )
        return _apply_partial_rope(pooled[:, :, None, :], cos, sin)[:, :, 0, :]

    def _prepare_kernel_supported(
        self,
        values: mx.array,
        norm_weight: mx.array,
        *,
        expected_ndim: int,
    ) -> bool:
        if not _fused_qsa_indexer_enabled():
            return False
        from .tensor_support import qsa_indexer_prepare_supported

        return qsa_indexer_prepare_supported(
            values,
            norm_weight,
            self._inv_freq,
            expected_ndim=expected_ndim,
        )

    def _extend_pooled(self, cache: QSACache, total: int) -> mx.array | None:
        nb_total = total // self.ratio
        nb_old = min(cache.pooled_len, nb_total)
        if nb_total > nb_old:
            fresh = cache.raw_keys[:, nb_old * self.ratio : nb_total * self.ratio, :]
            if self._prepare_kernel_supported(
                fresh,
                self.k_layernorm.weight,
                expected_ndim=3,
            ):
                from .tensor_support import qsa_indexer_pool_keys_metal

                pooled = qsa_indexer_pool_keys_metal(
                    fresh,
                    self.k_layernorm.weight,
                    self._inv_freq,
                    block_start=nb_old,
                    compress_ratio=self.ratio,
                    eps=self.rms_norm_eps,
                    attention_scaling=self._rope_attention_scaling,
                )
            else:
                pooled = self._pool_keys_eager(fresh, nb_old, nb_total)
            cache.write_pooled(pooled, nb_old, nb_total)
        if nb_total == 0:
            return None
        return cache.pooled[:, :nb_total, :]

    def _tiled_topk(
        self,
        q: mx.array,
        pooled_t: mx.array,
        nb_q: mx.array,
        blk: mx.array,
        neg: mx.array,
        k_eff: int,
        nb_total: int,
        tile: int,
    ) -> mx.array:
        """Per-tile scoring + top-k; concatenated [S, k_eff] indices.

        Row math is identical to the whole-chunk path (each row's dot,
        relu-sum, validity mask, tie-break and top-k involve no other row);
        the per-tile mx.eval is the point — it retires each tile's fp32
        score intermediates before the next tile is built, so the live
        transient is bounded by ONE tile instead of the whole chunk."""
        parts = []
        S = q.shape[1]
        tie = blk.astype(mx.float32)[None, :] * 1e-12
        for s0 in range(0, S, tile):
            s1 = min(s0 + tile, S)
            sc = mx.matmul(q[:, s0:s1].astype(mx.float32), pooled_t)
            sc = mx.maximum(sc, 0.0).sum(axis=2) / math.sqrt(self.head_dim)
            sc = sc[0]  # [s1-s0, nb]
            valid_t = blk[None, :] < nb_q[s0:s1, None]
            masked_t = mx.where(valid_t, sc, neg) - tie
            top_t = mx.argpartition(masked_t, kth=nb_total - k_eff, axis=-1)[
                :, nb_total - k_eff :
            ]
            mx.eval(top_t)
            parts.append(top_t)
        return mx.concatenate(parts, axis=0)

    def _select_eager(
        self,
        q: mx.array,
        pos_start: int,
        cache: QSACache,
        pooled: mx.array,
        total: int,
    ):
        """Stock MLX selector kept as the correctness oracle and kill-switch."""

        S = q.shape[1]
        nb_total = pooled.shape[1]
        # Cached fp32-transposed pooled view: same values as the old per-call
        # astype+swapaxes of the whole table (astype of the same bf16 blocks
        # -> bit-identical scores), without re-materializing 33.5 MB per
        # layer per token at 262K (#393).
        pooled_t = cache.pooled_f32_view(nb_total)  # [1,1,D,nb]

        qpos = mx.arange(pos_start, pos_start + S, dtype=mx.int32)  # abs position
        nb_q = (qpos + 1) // self.ratio  # complete blocks visible per query [S]
        blk = mx.arange(nb_total, dtype=mx.int32)
        valid = blk[None, :] < nb_q[:, None]  # [S, nb]
        neg = mx.array(-mx.inf, dtype=mx.float32)
        k_eff = min(self.block_topk, nb_total)

        tile = _qsa_score_tile_rows()
        if S > 1 and 0 < tile < S:
            # Tiled scoring (see _qsa_score_tile_rows): bounds the live fp32
            # score transient at one tile; per-row selection math identical.
            top_idx = self._tiled_topk(
                q, pooled_t, nb_q, blk, neg, k_eff, nb_total, tile
            )
        else:
            scores = mx.matmul(q.astype(mx.float32), pooled_t)  # [1,S,H,nb]
            scores = mx.maximum(scores, 0.0).sum(axis=2) / math.sqrt(self.head_dim)
            scores = scores[0]  # [S, nb]
            masked_scores = mx.where(valid, scores, neg)
            # torch.topk tie-break (lowest index wins). Exact ties are
            # common: a block whose every head-dot is negative relu-scores
            # exactly 0.0.
            masked_scores = masked_scores - blk.astype(mx.float32)[None, :] * 1e-12
            top_idx = mx.argpartition(masked_scores, kth=nb_total - k_eff, axis=-1)[
                :, nb_total - k_eff :
            ]

        if S > 1 and _qsa_large_prefill_enabled(S, total):
            # Preserve the eager score/top-k expression as an independently
            # selectable oracle while handing attention the compact block set.
            # IDs are chronological so the online-softmax consumer has a
            # deterministic traversal; validity distinguishes the padded
            # top-k slots on early rows whose complete prefix has < K blocks.
            block_ids = mx.sort(top_idx.astype(mx.int32), axis=-1)
            block_valid = mx.take_along_axis(
                valid,
                block_ids.astype(mx.int64),
                axis=-1,
            )
            block_ids = mx.where(
                block_valid,
                block_ids,
                mx.array(0, dtype=mx.int32),
            )
            _qsa_prefill_count("eager_selector")
            return ("flash_prefill", block_ids, block_valid)

        selected = mx.zeros((S, nb_total), dtype=mx.bool_)
        selected = mx.put_along_axis(
            selected, top_idx.astype(mx.int64), mx.array(True), axis=-1
        )
        selected = selected & valid  # -inf padding rows never select

        if S == 1 and _qsa_flash_enabled():
            # Flash-skip lane: hand attention the sorted
            # selected BLOCK ids + host-side tail bounds; the block-sparse
            # flash kernel iterates exactly that visible set in place — no
            # dense mask staged, no gathered copies (both measured slower:
            # dense = full-context reads, gather = -5.25% from two
            # materialized copies per layer per token, d6171d2c).
            _qsa_prefill_count("decode_flash_skip")
            blk_idx = mx.sort(top_idx[0].astype(mx.int32))
            tail_start = ((pos_start + 1) // self.ratio) * self.ratio
            return ("flash", blk_idx, tail_start)

        if S == 1 and _qsa_gather_decode_enabled():
            # Decode gather lane, an explicitly enabled optimization:
            # FALSIFIED d6171d2c, clean A/B/A -5.25% at 22.9k, so the
            # rows-gather family default must never arm it): return the
            # selected TOKEN INDICES instead of a dense [T] mask so
            # attention reads only budget+tail keys/values. Every returned
            # token is visible by construction (complete selected blocks
            # are < the tail start; the tail runs to the current position),
            # so the gathered SDPA needs no mask — identical math to the
            # masked dense product over the same visible set.
            blk_idx = mx.sort(top_idx[0].astype(mx.int32))
            tok_from_blocks = (
                blk_idx[:, None] * self.ratio + mx.arange(self.ratio, dtype=mx.int32)
            ).reshape(-1)
            # Host-side int (no .item() sync — a per-layer eval would stall
            # the AR pipeline): for the single decode row qpos == pos_start,
            # so the visible-complete-block count is (pos_start+1) // ratio.
            _qsa_prefill_count("decode_gather")
            tail_start = ((pos_start + 1) // self.ratio) * self.ratio
            tail_ids = mx.arange(tail_start, total, dtype=mx.int32)
            return mx.concatenate([tok_from_blocks, tail_ids])

        if (
            S > 1
            and not (0 < tile < S)  # tiled branch produced no top_idx
            and _qsa_gather_enabled()
            and _qsa_gather_max_rows() >= S
            and total >= _qsa_gather_min_context()
        ):
            # Rows-gather lane at S>1, adapting the
            # per-query gather + GQA-broadcast attention from community PR
            # #380 by @maceip. Every S>1 forward previously staged a dense
            # [S, T] bool mask and read the FULL KV through fused SDPA in
            # each of the 12 QSA layers, an O(T)-per-round chain that grows
            # with the generation. Here each row hands attention its own
            # token list instead: k_eff selected blocks plus its visible
            # tail, padded to one constant width (k_eff*ratio + ratio) so
            # gather shapes stay stable across the whole generation.
            # Selected blocks are complete blocks strictly below each row's
            # tail start, so the lists never double-count a token; invalid
            # slots carry valid=False and are re-pointed at token 0 for the
            # take.
            blk_ok = mx.take_along_axis(valid, top_idx.astype(mx.int64), axis=-1)
            tok_blocks = (
                top_idx.astype(mx.int32)[:, :, None] * self.ratio
                + mx.arange(self.ratio, dtype=mx.int32)
            ).reshape(S, -1)
            blocks_ok = mx.repeat(blk_ok, self.ratio, axis=1)
            tail_tok = nb_q[:, None] * self.ratio + mx.arange(
                self.ratio, dtype=mx.int32
            )
            tail_ok = tail_tok <= qpos[:, None]
            token_idx = mx.concatenate([tok_blocks, tail_tok], axis=1)
            token_ok = mx.concatenate([blocks_ok, tail_ok], axis=1)
            token_idx = mx.where(token_ok, token_idx, mx.array(0, dtype=mx.int32))
            return ("gather_rows", token_idx, token_ok)

        # Blocks -> tokens, plus the visible tail, intersected with causal.
        if S == 1:
            _qsa_prefill_count("decode_dense_mask")
        tok_sel = mx.repeat(selected, self.ratio, axis=1)
        if nb_total * self.ratio < total:
            pad = mx.zeros((S, total - nb_total * self.ratio), dtype=mx.bool_)
            tok_sel = mx.concatenate([tok_sel, pad], axis=1)
        tpos = mx.arange(total, dtype=mx.int32)
        tail = tpos[None, :] >= (nb_q[:, None] * self.ratio)
        causal = tpos[None, :] <= qpos[:, None]
        mask = (tok_sel | tail) & causal
        return mask[None, None]

    def _fused_selector_supported(self, q: mx.array, pooled: mx.array) -> bool:
        """Static fail-closed eligibility; kernel failures remain visible."""

        if not mx.metal.is_available() or mx.default_device() != mx.gpu:
            return False
        supported_dtypes = (mx.float16, mx.bfloat16, mx.float32)
        if q.dtype not in supported_dtypes or pooled.dtype not in supported_dtypes:
            return False
        if q.ndim != 4 or pooled.ndim != 3:
            return False
        if q.shape[0] != 1 or pooled.shape[0] != 1:
            return False
        if q.shape[1] <= 0 or q.shape[2] <= 0 or q.shape[3] <= 0:
            return False
        if pooled.shape[1] <= 0 or pooled.shape[2] != q.shape[3]:
            return False
        if not (1 <= self.block_topk <= 512) or self.ratio <= 0:
            return False
        if mx.float32 in (q.dtype, pooled.dtype):
            from .tensor_support import qsa_indexer_select_nax_available

            if not qsa_indexer_select_nax_available():
                return False
        return True

    def _prefill_selector_supported(self, q: mx.array, pooled: mx.array) -> bool:
        """Fail-closed eligibility for the vectorized large-S selector."""

        if not mx.metal.is_available() or mx.default_device() != mx.gpu:
            return False
        supported_dtypes = (mx.float16, mx.bfloat16, mx.float32)
        if q.dtype not in supported_dtypes or pooled.dtype not in supported_dtypes:
            return False
        if q.ndim != 4 or pooled.ndim != 3:
            return False
        if int(q.shape[0]) != 1 or int(pooled.shape[0]) != 1:
            return False
        if int(q.shape[1]) <= 1 or int(q.shape[2]) <= 0 or int(q.shape[3]) <= 0:
            return False
        if int(pooled.shape[1]) <= 0 or int(pooled.shape[2]) != int(q.shape[3]):
            return False
        return 1 <= self.block_topk <= 512 and self.ratio > 0

    def _fused_query_chunk_rows(self, rows: int, backing_blocks: int) -> int:
        scratch_per_row = max(1, int(backing_blocks)) * 4
        return min(
            int(rows),
            max(1, self._fused_score_scratch_bytes // scratch_per_row),
        )

    def _select_fused(
        self,
        q: mx.array,
        pos_start: int,
        pooled_backing: mx.array,
        logical_blocks: int,
        total: int,
        mode: str,
    ):
        """Dispatch the selector, chunking query rows but never history."""

        from .tensor_support import (
            qsa_indexer_select_blocks_metal,
            qsa_indexer_select_dense_mask_metal,
            qsa_indexer_select_row_tokens_metal,
        )

        rows = int(q.shape[1])
        chunk_rows = self._fused_query_chunk_rows(rows, pooled_backing.shape[1])
        # Keep the custom-kernel output specialization stable while a pooled
        # cache allocation remains in place. A logical prefix can occupy all
        # backing blocks and still have up to ratio-1 visible tail tokens, so
        # one extra ratio-sized block is the smallest safe capacity.
        dense_output_capacity = (
            (int(pooled_backing.shape[1]) + 1) * self.ratio
            if mode == "dense_mask"
            else None
        )
        chunks = []
        for row_start in range(0, rows, chunk_rows):
            q_chunk = q[:, row_start : row_start + chunk_rows]
            kwargs = {
                "pos_start": pos_start + row_start,
                "total_tokens": total,
                "block_topk": self.block_topk,
                "compress_ratio": self.ratio,
                "logical_blocks": logical_blocks,
            }
            if mode == "blocks":
                chunk = qsa_indexer_select_blocks_metal(
                    q_chunk, pooled_backing, **kwargs
                )
            elif mode == "row_tokens":
                chunk = qsa_indexer_select_row_tokens_metal(
                    q_chunk, pooled_backing, **kwargs
                )
            elif mode == "dense_mask":
                chunk = qsa_indexer_select_dense_mask_metal(
                    q_chunk,
                    pooled_backing,
                    output_total_tokens=dense_output_capacity,
                    **kwargs,
                )
            else:
                raise ValueError(f"unknown fused QSA selector mode {mode!r}")
            chunks.append(chunk)

        if mode == "dense_mask":
            mask = chunks[0] if len(chunks) == 1 else mx.concatenate(chunks, axis=2)
            return mask[..., :total]
        if len(chunks) == 1:
            return chunks[0]
        return tuple(
            mx.concatenate([chunk[leaf] for chunk in chunks], axis=0)
            for leaf in range(len(chunks[0]))
        )

    def _prepare_queries_eager(self, q: mx.array, pos_start: int) -> mx.array:
        """Stock query preparation kept as the numeric oracle."""

        q = self.q_layernorm(q)
        positions = mx.arange(pos_start, pos_start + q.shape[1], dtype=mx.int32)
        cos, sin = _rope_cos_sin(
            positions,
            self._inv_freq,
            self._rope_attention_scaling,
        )
        return _apply_partial_rope(q, cos, sin)

    def _prepare_queries(self, q: mx.array, pos_start: int) -> mx.array:
        if not self._prepare_kernel_supported(
            q,
            self.q_layernorm.weight,
            expected_ndim=4,
        ):
            return self._prepare_queries_eager(q, pos_start)

        from .tensor_support import qsa_indexer_prepare_queries_metal

        return qsa_indexer_prepare_queries_metal(
            q,
            self.q_layernorm.weight,
            self._inv_freq,
            pos_start=pos_start,
            eps=self.rms_norm_eps,
            attention_scaling=self._rope_attention_scaling,
        )

    def _compiled_mode(
        self,
        *,
        decode: bool,
        rows: int,
        total: int,
        last_nb: int,
    ) -> str:
        """Choose one fixed-shape graph/output contract on the host."""

        if last_nb <= self.block_topk:
            # Cache maintenance still belongs to the captured indexer even
            # while sparse selection is mathematically the dense causal mask.
            return "update_only"
        if decode and _qsa_flash_enabled():
            return "blocks"
        if not decode and _qsa_large_prefill_enabled(rows, total):
            return "prefill_blocks"
        if (
            not decode
            and _qsa_gather_enabled()
            and rows <= _qsa_gather_max_rows()
            and total >= _qsa_gather_min_context()
        ):
            return "row_tokens"
        return "dense_mask"

    def _projection_output_matches_dtype(self, dtype: mx.Dtype) -> bool:
        """Fail closed when a hidden-source projection may promote dtype."""

        projection = getattr(self, "index_qk_proj", None)
        if projection is None or not callable(projection):
            return False
        weight = getattr(projection, "weight", None)
        if not isinstance(weight, mx.array):
            return False
        expected_width = (self.n_heads + self.kv_heads) * self.head_dim
        if weight.ndim != 2 or int(weight.shape[0]) != expected_width:
            return False

        # QuantizedLinear's U32 packed weight does not determine the output
        # dtype; its scales/biases do.  Dense Linear uses weight dtype.
        scales = getattr(projection, "scales", None)
        if isinstance(scales, mx.array):
            if scales.dtype != dtype:
                return False
            biases = getattr(projection, "biases", None)
            return not isinstance(biases, mx.array) or biases.dtype == dtype
        return weight.dtype == dtype

    def _compiled_route_supported(
        self,
        source: mx.array,
        cache: QSACache,
        *,
        pos_start: int,
        qk_rows_supplied: bool,
        decode: bool,
        mode: str,
    ) -> bool:
        """Static eligibility check; dispatched graph failures are not hidden."""

        if not (_fused_qsa_indexer_enabled() and _compiled_qsa_indexer_enabled()):
            return False
        if not mx.metal.is_available() or mx.default_device() != mx.gpu:
            return False
        if source.ndim != 3 or int(source.shape[0]) != 1:
            return False
        if int(source.shape[1]) <= 0 or self.kv_heads != 1:
            return False
        rows = int(source.shape[1])
        # Capturing cache maintenance at the dense==sparse boundary created a
        # fresh graph per layer/capacity bucket without doing any selection.
        # Keep prefill update-only work eager; decode retains its established
        # compiled cache lane.
        if (
            not decode
            and mode == "update_only"
            and current_attention_phase() == "prefill"
        ):
            return False
        # Never send a matrix-shaped prefill through the legacy
        # one-threadgroup-per-row scorer.  The dedicated mode is captured only
        # at the canonical chunk width; arbitrary suffix tails use the same
        # Metal prefill selector outside mx.compile, avoiding a new trace for
        # every tail shape.
        if (
            not decode
            and rows >= _qsa_prefill_min_rows()
            and (
                mode not in ("prefill_blocks", "update_only")
                or rows != _qsa_prefill_compile_rows()
            )
        ):
            return False
        if cache.ratio != self.ratio or pos_start != cache.offset:
            return False
        supported_dtypes = (mx.float16, mx.bfloat16, mx.float32)
        if source.dtype not in supported_dtypes:
            return False
        if not (0 < self.head_dim <= 128):
            return False
        rotary_dim = 2 * int(self._inv_freq.shape[0])
        if (
            self._inv_freq.ndim != 1
            or self._inv_freq.dtype != mx.float32
            or rotary_dim <= 0
            or rotary_dim > self.head_dim
            or rotary_dim % 2
        ):
            return False
        if not (1 <= self.block_topk <= 512) or self.ratio <= 0:
            return False
        if self.q_layernorm.weight.dtype != source.dtype:
            return False
        if self.k_layernorm.weight.dtype != source.dtype:
            return False
        if tuple(self.q_layernorm.weight.shape) != (self.head_dim,):
            return False
        if tuple(self.k_layernorm.weight.shape) != (self.head_dim,):
            return False

        if qk_rows_supplied:
            expected_width = (self.n_heads + self.kv_heads) * self.head_dim
            if int(source.shape[2]) != expected_width:
                return False
        elif not self._projection_output_matches_dtype(source.dtype):
            return False

        for backing in (cache.raw_keys, cache.pooled):
            if backing is None:
                continue
            if (
                backing.ndim != 3
                or int(backing.shape[0]) != 1
                or int(backing.shape[2]) != self.head_dim
                or backing.dtype != source.dtype
            ):
                return False
        if pos_start > 0 and (
            cache.raw_keys is None or int(cache.raw_keys.shape[1]) < pos_start
        ):
            return False
        if cache.pooled is None:
            if cache.pooled_len != 0:
                return False
        elif not 0 <= cache.pooled_len <= int(cache.pooled.shape[1]):
            return False
        logical_blocks = (pos_start + int(source.shape[1])) // self.ratio
        pooled_frontier = min(cache.pooled_len, logical_blocks)
        max_new_blocks = (int(source.shape[1]) + self.ratio - 1) // self.ratio
        if logical_blocks - pooled_frontier > max_new_blocks:
            return False

        # Preserve the variable-width decode-gather experiment in the eager
        # oracle.  When flash is also enabled it wins first and its fixed block
        # output is safe for the compiled path.
        if (
            decode
            and mode not in {"update_only", "blocks"}
            and _qsa_gather_decode_enabled()
        ):
            return False

        if mode not in ("update_only", "prefill_blocks") and source.dtype == mx.float32:
            from .tensor_support import qsa_indexer_select_nax_available

            if not qsa_indexer_select_nax_available():
                return False
        return True

    def _ensure_compiled_backings(
        self,
        cache: QSACache,
        *,
        dtype: mx.Dtype,
        pos_start: int,
        rows: int,
    ) -> tuple[mx.array, mx.array]:
        """Reserve/materialize shape-stable raw and pooled cache leaves."""

        from .qsa_replay import (
            precompute_qsa_replay_capacity,
            qsa_indexer_capacity_bucket,
        )

        raw_existing = 0 if cache.raw_keys is None else int(cache.raw_keys.shape[1])
        pooled_existing = 0 if cache.pooled is None else int(cache.pooled.shape[1])
        # Phase-3 staging can reserve a wider pristine cache before dtype/head
        # width are known.  Include those pending extents rather than
        # accidentally materializing the smaller immediate-call plan.
        raw_existing = max(raw_existing, cache._reserved_raw_capacity)
        pooled_existing = max(pooled_existing, cache._reserved_pooled_capacity)
        plan = precompute_qsa_replay_capacity(
            start_offset=pos_start,
            window_tokens=rows,
            compress_ratio=self.ratio,
            allocation_step=cache.step,
            current_raw_capacity=raw_existing,
            current_pooled_capacity=pooled_existing,
        )
        # The compiled pool stage has one fixed ceil(S/ratio)-block window.
        # For an unaligned first prefill that can be one row wider than the
        # logical complete-block frontier (for example S=1025, ratio=4:
        # logical=256 but staging=257). Bucket the physical requirement itself,
        # rather than taking max() after bucketing 256, which would stay 256.
        max_new_blocks = (rows + self.ratio - 1) // self.ratio
        pooled_capacity = qsa_indexer_capacity_bucket(
            max(
                1,
                plan.complete_blocks,
                pooled_existing,
                max_new_blocks,
            ),
            minimum=cache.step,
        )
        cache.reserve_indexer_capacity(
            raw_capacity=plan.raw_capacity,
            pooled_capacity=pooled_capacity,
        )
        if cache.raw_keys is None:
            cache.raw_keys = mx.zeros(
                (1, cache._reserved_raw_capacity, self.head_dim),
                dtype=dtype,
            )
        if cache.pooled is None:
            cache.pooled = mx.zeros(
                (1, cache._reserved_pooled_capacity, self.head_dim),
                dtype=dtype,
            )
        return cache.raw_keys, cache.pooled

    def _compiled_parameter_signature(self) -> tuple[int, ...]:
        """Identity seal for arrays captured by the lazy graph manager."""

        projection = getattr(self, "index_qk_proj", None)
        leaves = [
            self.q_layernorm.weight,
            self.k_layernorm.weight,
            self._inv_freq,
            getattr(projection, "weight", None),
            getattr(projection, "scales", None),
            getattr(projection, "biases", None),
        ]
        return tuple(id(leaf) for leaf in leaves if isinstance(leaf, mx.array))

    def _get_compiled_indexer_core(self):
        """Build the graph bank only after final checkpoint weights exist."""

        signature = self._compiled_parameter_signature()
        core = self._compiled_indexer_core
        if core is not None and signature == self._compiled_indexer_parameter_signature:
            return core

        from .tensor_support import QSACompiledIndexerCore

        projection = getattr(self, "index_qk_proj", None)
        core = QSACompiledIndexerCore(
            n_heads=self.n_heads,
            kv_heads=self.kv_heads,
            head_dim=self.head_dim,
            block_topk=self.block_topk,
            compress_ratio=self.ratio,
            q_norm_weight=self.q_layernorm.weight,
            k_norm_weight=self.k_layernorm.weight,
            inv_freq=self._inv_freq,
            rms_norm_eps=self.rms_norm_eps,
            rope_attention_scaling=self._rope_attention_scaling,
            project_qk=projection if callable(projection) else None,
            selector_scratch_bytes=self._fused_score_scratch_bytes,
            prefill_score_workspace_bytes=_qsa_prefill_score_workspace_bytes(),
        )
        object.__setattr__(self, "_compiled_indexer_core", core)
        object.__setattr__(
            self,
            "_compiled_indexer_parameter_signature",
            signature,
        )
        return core

    def _call_rows_compiled(
        self,
        hidden: mx.array,
        pos_start: int,
        cache: QSACache,
        qk_rows: mx.array | None,
        *,
        mode: str,
    ):
        """Run and commit one pure explicit-state compiled indexer graph."""

        rows = int(hidden.shape[1])
        total = pos_start + rows
        logical_blocks = total // self.ratio
        source = hidden if qk_rows is None else qk_rows
        raw_keys, pooled = self._ensure_compiled_backings(
            cache,
            dtype=source.dtype,
            pos_start=pos_start,
            rows=rows,
        )
        core = self._get_compiled_indexer_core()
        kwargs = {
            "pos_start": pos_start,
            "total_tokens": total,
            "logical_blocks": logical_blocks,
            "pooled_len": min(cache.pooled_len, logical_blocks),
            "mode": mode,
        }
        if qk_rows is None:
            result = core.select_hidden(hidden, raw_keys, pooled, **kwargs)
        else:
            result = core.select_qk_rows(qk_rows, raw_keys, pooled, **kwargs)

        # The graph is pure: updated cache arrays are explicit outputs.  The
        # host already knows the logical frontier, so committing it needs no
        # .item() synchronization. Attention advances cache.kv.offset later.
        cache.raw_keys = result.raw_keys
        cache.pooled = result.pooled
        cache.pooled_len = logical_blocks
        # The compiled graph rebuilt ``pooled`` wholesale, so the eager lane's
        # fp32-transposed mirror is stale; drop it and let pooled_f32_view
        # rebuild lazily on the next eager read.
        cache.pooled_f32_t = None

        if mode == "update_only":
            return None
        if mode == "blocks":
            block_ids, _, _ = result.selection
            tail_start = ((pos_start + 1) // self.ratio) * self.ratio
            return ("flash", block_ids[0], tail_start)
        if mode == "prefill_blocks":
            block_ids, block_valid, _ = result.selection
            _qsa_prefill_count("compiled_selector")
            return ("flash_prefill", block_ids, block_valid)
        if mode == "row_tokens":
            token_idx, token_ok = result.selection
            return ("gather_rows", token_idx, token_ok)
        if mode == "dense_mask":
            return result.selection[..., :total]
        raise ValueError(f"unknown compiled QSA indexer mode {mode!r}")

    def _call_rows(
        self,
        hidden: mx.array,
        pos_start: int,
        cache: QSACache,
        qk_rows: mx.array | None,
        *,
        decode: bool,
    ) -> mx.array | None:
        """Shared arithmetic behind the explicit prefill/decode entry points."""

        B, S, _ = hidden.shape
        if decode != (S == 1):
            raise ValueError(
                f"QSA decode route requires S=1 and prefill requires S>1; got S={S}"
            )
        T = pos_start + S  # == the KV length after this forward's update
        last_nb = T // self.ratio
        compiled_mode = self._compiled_mode(
            decode=decode,
            rows=S,
            total=T,
            last_nb=last_nb,
        )
        compiled_source = hidden if qk_rows is None else qk_rows
        if self._compiled_route_supported(
            compiled_source,
            cache,
            pos_start=pos_start,
            qk_rows_supplied=qk_rows is not None,
            decode=decode,
            mode=compiled_mode,
        ):
            return self._call_rows_compiled(
                hidden,
                pos_start,
                cache,
                qk_rows,
                mode=compiled_mode,
            )

        # qk_rows: the layer's fused shared-input GEMV already produced this
        # projection capability — same rows bit-exactly. Keeping
        # projection outside the Metal preparation kernel is also the vLLM
        # boundary and preserves packed/quantized Linear dispatches.
        qk = self.index_qk_proj(hidden) if qk_rows is None else qk_rows
        q, k = mx.split(qk, [self.n_heads * self.head_dim], axis=-1)
        q = q.reshape(B, S, self.n_heads, self.head_dim)
        k = k.reshape(B, S, self.head_dim)
        q = self._prepare_queries(q, pos_start)

        cache.write_raw(k)
        pooled = self._extend_pooled(cache, T)
        nb_total = 0 if pooled is None else pooled.shape[1]

        # Per-query complete-block counts. If every visible prefix fits inside
        # the budget the selection is the full causal mask — skip the work.
        if last_nb <= self.block_topk:
            return None  # dense == sparse in this regime

        pooled_backing = cache.pooled
        large_prefill = not decode and _qsa_large_prefill_enabled(S, T)
        if large_prefill and self._prefill_selector_supported(q, pooled):
            from .tensor_support import qsa_indexer_prefill_blocks_metal

            block_ids, block_valid, _ = qsa_indexer_prefill_blocks_metal(
                q,
                pooled,
                pos_start=pos_start,
                total_tokens=T,
                block_topk=self.block_topk,
                compress_ratio=self.ratio,
                logical_blocks=nb_total,
                score_workspace_bytes=_qsa_prefill_score_workspace_bytes(),
            )
            _qsa_prefill_count("metal_selector")
            return ("flash_prefill", block_ids, block_valid)

        # The original custom selector scores a row serially inside one
        # threadgroup.  Keep it for decode/small verify only; large-S prefill
        # either takes the tiled scorer above or the stock vectorized oracle.
        legacy_fused = (
            _fused_qsa_indexer_enabled()
            and pooled_backing is not None
            and (decode or _qsa_prefill_min_rows() > S)
            # Preserve the dormant variable-width decode-gather lane only in
            # the eager oracle. Flash still wins there when both knobs are on.
            and not (decode and _qsa_gather_decode_enabled())
            and self._fused_selector_supported(q, pooled_backing)
        )
        if not legacy_fused:
            return self._select_eager(q, pos_start, cache, pooled, T)

        if decode and _qsa_flash_enabled():
            block_ids, _, _ = self._select_fused(
                q,
                pos_start,
                pooled_backing,
                nb_total,
                T,
                "blocks",
            )
            tail_start = ((pos_start + 1) // self.ratio) * self.ratio
            return ("flash", block_ids[0], tail_start)

        if (
            not decode
            and _qsa_gather_enabled()
            and _qsa_gather_max_rows() >= S
            and _qsa_gather_min_context() <= T
        ):
            token_idx, token_ok = self._select_fused(
                q,
                pos_start,
                pooled_backing,
                nb_total,
                T,
                "row_tokens",
            )
            return ("gather_rows", token_idx, token_ok)

        return self._select_fused(
            q,
            pos_start,
            pooled_backing,
            nb_total,
            T,
            "dense_mask",
        )

    def _call_batch(
        self,
        hidden: mx.array,
        cache: _QSABatchCache,
        qk_rows: mx.array | None,
    ) -> mx.array:
        """Vectorized QSA selection for right-padded independent rows.

        The batch dimension is request identity. Sequence width is shared for
        the current forward, while absolute positions and visible key lengths
        remain per-row. Custom B=1 kernels deliberately stay out of this path.
        """

        batch, rows, _ = hidden.shape
        if batch != len(cache.offsets):
            raise ValueError("QSA hidden/cache batch size changed")
        source = self.index_qk_proj(hidden) if qk_rows is None else qk_rows
        q, k = mx.split(source, [self.n_heads * self.head_dim], axis=-1)
        q = q.reshape(batch, rows, self.n_heads, self.head_dim)
        k = k.reshape(batch, rows, self.head_dim)
        q = self.q_layernorm(q)

        starts = mx.array(cache.offsets, dtype=mx.int32)
        positions = starts[:, None] + mx.arange(rows, dtype=mx.int32)[None, :]
        cos, sin = _rope_cos_sin(
            positions,
            self._inv_freq,
            self._rope_attention_scaling,
        )
        q = _apply_partial_rope(q, cos, sin)
        cache.write_raw(k)

        ends = tuple(offset + rows for offset in cache.offsets)
        total = max(ends)
        block_count = total // self.ratio
        pooled: mx.array | None = None
        if block_count:
            pooled_source = cache.raw_keys[:, : block_count * self.ratio, :]
            pooled_source = pooled_source.reshape(
                batch,
                block_count,
                self.ratio,
                self.head_dim,
            )
            pooled = mx.mean(pooled_source.astype(mx.float32), axis=2).astype(
                pooled_source.dtype
            )
            pooled = self.k_layernorm(pooled)
            block_starts = mx.arange(block_count, dtype=mx.int32) * self.ratio
            block_cos, block_sin = _rope_cos_sin(
                block_starts,
                self._inv_freq,
                self._rope_attention_scaling,
            )
            pooled = _apply_partial_rope(pooled[:, :, None, :], block_cos, block_sin)[
                :, :, 0, :
            ]
        cache.pooled = pooled

        tpos = mx.arange(total, dtype=mx.int32)
        causal = tpos[None, None, :] <= positions[:, :, None]
        complete_for_query = (positions + 1) // self.ratio
        if block_count <= self.block_topk:
            return causal[:, None]

        block_ids = mx.arange(block_count, dtype=mx.int32)
        valid = block_ids[None, None, :] < complete_for_query[:, :, None]
        pooled_t = mx.swapaxes(pooled.astype(mx.float32), 1, 2)[:, None]
        scores = mx.matmul(q.astype(mx.float32), pooled_t)
        scores = mx.maximum(scores, 0.0).sum(axis=2) / math.sqrt(self.head_dim)
        neg = mx.array(-mx.inf, dtype=mx.float32)
        ranked = mx.where(valid, scores, neg)
        ranked = ranked - block_ids.astype(mx.float32)[None, None, :] * 1e-12
        keep = min(self.block_topk, block_count)
        selected_ids = mx.argpartition(
            ranked,
            kth=block_count - keep,
            axis=-1,
        )[..., block_count - keep :]
        selected = mx.zeros((batch, rows, block_count), dtype=mx.bool_)
        selected = mx.put_along_axis(
            selected,
            selected_ids.astype(mx.int64),
            mx.array(True),
            axis=-1,
        )
        selected = selected & valid
        token_selected = mx.repeat(selected, self.ratio, axis=-1)
        if token_selected.shape[-1] < total:
            token_selected = mx.concatenate(
                (
                    token_selected,
                    mx.zeros(
                        (batch, rows, total - token_selected.shape[-1]),
                        dtype=mx.bool_,
                    ),
                ),
                axis=-1,
            )
        tail_start = complete_for_query * self.ratio
        tail = tpos[None, None, :] >= tail_start[:, :, None]
        return ((token_selected | tail) & causal)[:, None]

    def _call_decode(
        self,
        hidden: mx.array,
        pos_start: int,
        cache: QSACache,
        qk_rows: mx.array | None,
    ) -> mx.array | None:
        return self._call_rows(hidden, pos_start, cache, qk_rows, decode=True)

    def _call_prefill(
        self,
        hidden: mx.array,
        pos_start: int,
        cache: QSACache,
        qk_rows: mx.array | None,
    ) -> mx.array | None:
        return self._call_rows(hidden, pos_start, cache, qk_rows, decode=False)

    def __call__(
        self,
        hidden: mx.array,
        pos_start: int | tuple[int, ...],
        cache: QSACache | _QSABatchCache,
        qk_rows: mx.array | None = None,
    ) -> mx.array | None:
        B, S, _ = hidden.shape
        if isinstance(cache, _QSABatchCache):
            return self._call_batch(hidden, cache, qk_rows)
        if B != 1 or not isinstance(pos_start, int):
            raise ValueError("multi-row QSA requires an explicit batch cache")
        if S == 1:
            return self._call_decode(hidden, pos_start, cache, qk_rows)
        return self._call_prefill(hidden, pos_start, cache, qk_rows)


def _qsa_rows_gather_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    token_idx: mx.array,
    token_ok: mx.array,
    scale: float,
) -> mx.array:
    """Attention over per-row gathered tokens (rows-gather lane).

    Adapting the per-query single-take gather + GQA head-group broadcast
    from community PR #380 by @maceip, minus its mask re-gather (the
    selection itself carries validity here, so no second gathered copy is
    ever built — the S=1 lane's receipt priced each extra copy at -5.25%).

    q is [1, H, S, D]; k/v are the cache's [1, H_kv, T, D] slices;
    token_idx/token_ok are [S, K]. Keys differ per row, so fused SDPA over
    a shared KV sequence cannot serve this; the whole S stays in-graph as
    one broadcast GEMM pair, and the 12x-repeated K/V working set is never
    materialized: q is viewed [1, H_kv, rep, S, 1, D] against
    [1, H_kv, 1, S, D, K]. Invalid slots score -inf before the fp32
    softmax, identical math to the dense bool-mask product over the same
    visible set.
    """
    _B, H, S, D = q.shape
    H_kv = int(k.shape[1])
    K = int(token_idx.shape[-1])
    flat = token_idx.reshape(-1)
    k_sel = mx.take(k, flat, axis=2).reshape(1, H_kv, S, K, D)
    v_sel = mx.take(v, flat, axis=2).reshape(1, H_kv, S, K, D)
    neg = mx.array(-mx.inf, dtype=mx.float32)
    if H_kv != H:
        rep = H // H_kv
        q_view = q.reshape(1, H_kv, rep, S, 1, D)
        k_view = k_sel.swapaxes(-1, -2).reshape(1, H_kv, 1, S, D, K)
        scores = mx.matmul(q_view, k_view).squeeze(-2).astype(mx.float32) * scale
        scores = mx.where(token_ok[None, None, None], scores, neg)
        probs = mx.softmax(scores, axis=-1).astype(q.dtype)
        v_view = v_sel.reshape(1, H_kv, 1, S, K, D)
        out = mx.matmul(probs[..., None, :], v_view).squeeze(-2)
        return out.reshape(1, H, S, D)
    scores = (
        mx.matmul(q[..., None, :], k_sel.swapaxes(-1, -2))
        .squeeze(-2)
        .astype(mx.float32)
        * scale
    )
    scores = mx.where(token_ok[None, None], scores, neg)
    probs = mx.softmax(scores, axis=-1).astype(q.dtype)
    return mx.matmul(probs[..., None, :], v_sel).squeeze(-2)


def _qsa_prefill_gather_attention(
    q: mx.array,
    keys: mx.array,
    values: mx.array,
    block_ids: mx.array,
    block_valid: mx.array,
    *,
    pos_start: int,
    total_tokens: int,
    compress_ratio: int,
    scale: float,
    tile_rows: int,
) -> mx.array:
    """Portable bounded attention over the flash_prefill block contract.

    The universal tier between the NAX flash kernel and the dense-mask
    reconstruction (approach from oMLX PR #3244's portable lane, Apache-2.0,
    reimplemented on our block/validity contract): each query row attends to
    its selected complete blocks plus its visible tail — a constant
    ``topk*ratio + ratio`` token width — instead of the full [S, T] masked
    context. Row tiles bound the gathered K/V working set; a per-tile eval
    retires each tile's gathered copies before the next tile is built.

    q is [1, H, S, D]; keys/values are the FULL cache backings
    [1, H_kv, cap, D] (never sliced-contiguous — gathers index absolute
    rows, all < total_tokens). Selection semantics match the dense mask
    exactly: selected blocks are complete (< each row's tail start), the
    tail runs to the row's own position, and invalid slots score -inf.
    """

    S = int(q.shape[2])
    ratio = int(compress_ratio)
    arange_ratio = mx.arange(ratio, dtype=mx.int32)
    outputs = []
    for r0 in range(0, S, tile_rows):
        r1 = min(r0 + tile_rows, S)
        qpos = mx.arange(pos_start + r0, pos_start + r1, dtype=mx.int32)
        nb_q = (qpos + 1) // ratio
        ids_t = block_ids[r0:r1]
        ok_t = block_valid[r0:r1]
        tok_blocks = (
            ids_t.astype(mx.int32)[:, :, None] * ratio + arange_ratio
        ).reshape(r1 - r0, -1)
        blocks_ok = mx.repeat(ok_t, ratio, axis=1)
        tail_tok = nb_q[:, None] * ratio + arange_ratio
        tail_ok = tail_tok <= qpos[:, None]
        token_idx = mx.concatenate([tok_blocks, tail_tok], axis=1)
        token_ok = mx.concatenate([blocks_ok, tail_ok], axis=1)
        token_idx = mx.where(token_ok, token_idx, mx.array(0, dtype=mx.int32))
        out_t = _qsa_rows_gather_attention(
            q[:, :, r0:r1],
            keys,
            values,
            token_idx,
            token_ok,
            scale,
        )
        mx.eval(out_t)
        outputs.append(out_t)
    return outputs[0] if len(outputs) == 1 else mx.concatenate(outputs, axis=2)


def _qsa_blocks_to_dense_mask(
    block_ids: mx.array,
    block_valid: mx.array,
    *,
    pos_start: int,
    total_tokens: int,
    compress_ratio: int,
) -> mx.array:
    """Reconstruct the exact dense QSA mask for an unsupported consumer.

    The normal large-prefill route never calls this: its sparse attention
    kernel consumes block IDs directly.  Keeping the reconstruction here makes
    unsupported geometry/dtype combinations correctness-preserving without
    hiding a Metal dispatch failure.  A sentinel column absorbs every invalid
    top-k slot so no padded entry can accidentally select block zero.
    """

    rows = int(block_ids.shape[0])
    ratio = int(compress_ratio)
    logical_blocks = int(total_tokens) // ratio
    qpos = mx.arange(pos_start, pos_start + rows, dtype=mx.int32)
    complete_for_row = (qpos + 1) // ratio
    in_range = (
        (block_ids >= 0)
        & (block_ids < logical_blocks)
        & (block_ids < complete_for_row[:, None])
    )
    valid = block_valid & in_range
    sentinel = mx.array(logical_blocks, dtype=mx.int32)
    safe_ids = mx.where(valid, block_ids, sentinel)
    selected = mx.zeros((rows, logical_blocks + 1), dtype=mx.bool_)
    selected = mx.put_along_axis(
        selected,
        safe_ids.astype(mx.int64),
        mx.array(True),
        axis=-1,
    )[:, :logical_blocks]
    token_selected = mx.repeat(selected, ratio, axis=-1)
    complete_token_count = logical_blocks * ratio
    if complete_token_count < int(total_tokens):
        token_selected = mx.concatenate(
            [
                token_selected,
                mx.zeros(
                    (rows, int(total_tokens) - complete_token_count),
                    dtype=mx.bool_,
                ),
            ],
            axis=-1,
        )

    tail_start = complete_for_row * ratio
    tpos = mx.arange(total_tokens, dtype=mx.int32)
    tail = (tpos[None, :] >= tail_start[:, None]) & (tpos[None, :] <= qpos[:, None])
    causal = tpos[None, :] <= qpos[:, None]
    return ((token_selected | tail) & causal)[None, None]


class Attention(nn.Module):
    """Gated GQA (qwen3_5 style: double-width q_proj, sigmoid output gate,
    per-head q/k RMSNorm, partial rotary) masked by the QSA indexer."""

    # The QSA indexer mask is part of this module's semantics (and __call__
    # takes (x, cache)): any generic dense-SDPA rewrite that replaces
    # __call__ would silently drop the sparse selection. attention_split
    # honors this and never hooks the class.
    _qwen4_exp_generic_sdpa_rewrites_unsupported = True

    def __init__(self, args: TextArgs):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(
            args.hidden_size, self.n_heads * self.head_dim * 2, bias=args.attention_bias
        )
        self.k_proj = nn.Linear(
            args.hidden_size, self.n_kv_heads * self.head_dim, bias=args.attention_bias
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.n_kv_heads * self.head_dim, bias=args.attention_bias
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, args.hidden_size, bias=args.attention_bias
        )
        self.q_norm = GroupedRMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = GroupedRMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.indexer = QSAIndexer(args) if args.indexer_n_heads else None
        self._inv_freq, self._rope_attention_scaling = _rope_inv_freq_and_scaling(args)
        self._mrope_axes = (
            mx.array(
                _build_mrope_axes(args.mrope_section, args.mrope_interleaved),
                dtype=mx.int32,
            )
            if args.mrope_section
            and sum(args.mrope_section) == int(args.rotary_dim) // 2
            else None
        )

    def __call__(
        self,
        x: mx.array,
        cache: QSACache | _QSABatchCache,
    ) -> mx.array:
        B, S, _ = x.shape
        is_multirow = isinstance(cache, _QSABatchCache)
        pos_start = cache.offsets if is_multirow else cache.offset
        vrope = vision_rope_state()
        if is_multirow and vrope is not None:
            raise ValueError("multi-row QSA does not flatten request-local M-RoPE")

        fused = getattr(self, "qkv_fused", None)
        if fused is not None:
            # One shared-input GEMV replaces q/k/v (+ indexer qk when its
            # pack precision matches; the v2.10 artifact's 4-bit group-64
            # indexer can therefore join this dispatch). Row-concat is
            # bit-exact per row under the QSA QKV sanitize fusion.
            outs = fused(x)
            if len(outs) == 4:
                q, k, v, idx_rows = outs
            else:
                q, k, v = outs
                idx_rows = None
            sel_mask = (
                self.indexer(x, pos_start, cache, qk_rows=idx_rows)
                if self.indexer is not None
                else None
            )
        else:
            sel_mask = None
            if self.indexer is not None:
                sel_mask = self.indexer(x, pos_start, cache)
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)

        q, gate = mx.split(q.reshape(B, S, self.n_heads, -1), 2, axis=-1)
        gate = gate.reshape(B, S, -1)
        k = k.reshape(B, S, self.n_kv_heads, -1)
        v = v.reshape(B, S, self.n_kv_heads, -1)

        q = self.q_norm(q)
        k = self.k_norm(k)
        if vrope is not None and self._mrope_axes is not None:
            # Vision request: image tokens rope at (t, h, w) grid positions
            # from the request's M-RoPE table; spans past the table (decode
            # continuation) are equal-axes at sequence_index + delta, which
            # is plain rope shifted by delta. Table and delta are derived
            # per request from content — nothing new rides cache state.
            table, delta = vrope
            end = pos_start + S
            if table is not None and end <= int(table.shape[1]):
                cos, sin = _mrope_cos_sin(
                    table[:, pos_start:end], self._inv_freq, self._mrope_axes
                )
            else:
                positions = mx.arange(
                    pos_start + delta, pos_start + delta + S, dtype=mx.int32
                )
                cos, sin = _rope_cos_sin(
                    positions, self._inv_freq, self._rope_attention_scaling
                )
        elif is_multirow:
            positions = mx.array(pos_start, dtype=mx.int32)[:, None]
            positions = positions + mx.arange(S, dtype=mx.int32)[None, :]
            cos, sin = _rope_cos_sin(
                positions, self._inv_freq, self._rope_attention_scaling
            )
        else:
            positions = mx.arange(pos_start, pos_start + S, dtype=mx.int32)
            cos, sin = _rope_cos_sin(
                positions, self._inv_freq, self._rope_attention_scaling
            )
        q = _apply_partial_rope(q, cos, sin)
        k = _apply_partial_rope(k, cos, sin)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        k, v = cache.kv.update_and_fetch(k, v)
        T = k.shape[2]

        if vrope is not None:
            # Vision request: QSA sparse selection is bypassed and attention
            # runs dense-causal — the reference qwen4_exp implementation
            # serves multimodal exactly this way (its sparse fast paths
            # exclude M-RoPE). The indexer above still ran, so the QSA cache
            # streams (raw/pooled) stay byte-identical with text serving and
            # bank state keeps one format. Masks key on sequence order,
            # which remains correct under M-RoPE; only rope reads the axes.
            sel_mask = None

        if isinstance(sel_mask, tuple) and sel_mask and sel_mask[0] == "flash":
            # Block-sparse flash attention over the indexer's exact visible
            # set. Reads the cache BACKING arrays in place at their
            # allocation stride — the :T slice above is non-contiguous, and
            # forcing it contiguous would copy the entire KV.
            from .tensor_support import qsa_flash_skip

            _, blk_idx, tail_start = sel_mask
            out = qsa_flash_skip(
                q.reshape(self.n_heads, self.head_dim),
                cache.kv.keys,
                cache.kv.values,
                blk_idx,
                T,
                tail_start,
                self.scale,
            )
            out = out.reshape(B, S, -1)
            return self.o_proj(out * mx.sigmoid(gate))

        if isinstance(sel_mask, tuple) and sel_mask and sel_mask[0] == "flash_prefill":
            # Large-S prefill consumes compact per-row block selections
            # directly from the full cache backing.  This is the point of the
            # prefill port: no [S,T] selection expansion and no per-row K/V
            # gather on the supported production geometry.
            from .tensor_support import qsa_prefill_flash, qsa_prefill_flash_supported

            _, block_ids, block_valid = sel_mask
            if _qsa_prefill_flash_attention_enabled(
                S, T
            ) and qsa_prefill_flash_supported(
                q,
                cache.kv.keys,
                cache.kv.values,
                block_ids,
                block_valid,
                pos_start=pos_start,
                total_tokens=T,
                scale=self.scale,
            ):
                out = qsa_prefill_flash(
                    q,
                    cache.kv.keys,
                    cache.kv.values,
                    block_ids,
                    block_valid,
                    pos_start=pos_start,
                    total_tokens=T,
                    scale=self.scale,
                )
                _qsa_prefill_count("flash_kernel")
                out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)
                return self.o_proj(out * mx.sigmoid(gate))

            # Portable prefill gather tier: same compact block
            # contract, bounded gathered attention on any Metal device — the
            # universal lane for machines without Metal 4 TensorOps. Same
            # visible set as the dense mask; only reduction order differs.
            if _qsa_prefill_gather_enabled():
                out = _qsa_prefill_gather_attention(
                    q,
                    cache.kv.keys,
                    cache.kv.values,
                    block_ids,
                    block_valid,
                    pos_start=pos_start,
                    total_tokens=T,
                    compress_ratio=self.indexer.ratio,
                    scale=self.scale,
                    tile_rows=_qsa_prefill_gather_tile_rows(),
                )
                _qsa_prefill_count("gather_tier")
                out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)
                return self.o_proj(out * mx.sigmoid(gate))

            # Static unsupported geometry falls back exactly.  Once the
            # supported kernel is dispatched, failures propagate instead of
            # being hidden behind a dense retry.
            _qsa_prefill_count("dense_fallback")
            sel_mask = _qsa_blocks_to_dense_mask(
                block_ids,
                block_valid,
                pos_start=pos_start,
                total_tokens=T,
                compress_ratio=self.indexer.ratio,
            )

        if isinstance(sel_mask, tuple) and sel_mask and sel_mask[0] == "gather_rows":
            # Rows-gather lane (S>1): each verify/pipeline row reads only
            # its own selected blocks + tail instead of the full context
            # through a dense [S, T] mask. See _qsa_rows_gather_attention
            # (adapting community PR #380 by @maceip).
            _, tok_idx, tok_ok = sel_mask
            out = _qsa_rows_gather_attention(q, k, v, tok_idx, tok_ok, self.scale)
            out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)
            return self.o_proj(out * mx.sigmoid(gate))

        if sel_mask is not None and sel_mask.ndim == 1:
            # QSA gather lane (decode): the indexer returned the selected
            # token indices — attention reads budget+tail keys/values
            # instead of the full context through a dense bool mask. All
            # gathered tokens are visible, so no mask is needed; the
            # softmax over the same visible set is identical math.
            k = mx.take(k, sel_mask, axis=2)
            v = mx.take(v, sel_mask, axis=2)
            mask = None
        elif sel_mask is not None:
            mask = sel_mask
        elif S > 1:
            qpos = mx.arange(pos_start, pos_start + S, dtype=mx.int32)
            tpos = mx.arange(T, dtype=mx.int32)
            mask = (tpos[None, :] <= qpos[:, None])[None, None]
        else:
            mask = None

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)
        return self.o_proj(out * mx.sigmoid(gate))


_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _build_layer_multipliers(vocab: int, ngram_size: int, ple_index: int, seed: int):
    max_long = (1 << 63) - 1
    half_bound = max(1, (max_long // max(vocab, 1)) // 2)
    base_seed = seed + _PRIME_1 * ple_index
    out = []
    for i in range(ngram_size):
        v = (base_seed + _SPLITMIX_GAMMA * (i + 1)) & _MASK64
        out.append(2 * (_splitmix64(v) % half_bound) + 1)
    return out


def _is_prime(v: int) -> bool:
    if v < 2:
        return False
    if v % 2 == 0:
        return v == 2
    return all(v % d != 0 for d in range(3, math.isqrt(v) + 1, 2))


def _head_vocab_layout(base: int, heads: int, ple_index: int):
    sizes, offsets, total = [], [], 0
    prime = base - 1
    # global head index runs across PLE layers; sizes are consecutive primes
    for h in range(ple_index * heads + heads):
        prime += 1
        while not _is_prime(prime):
            prime += 1
        if h >= ple_index * heads:
            sizes.append(prime)
            offsets.append(total)
            total += prime
    return sizes, offsets, total


class NGramTable(nn.Module):
    """Embedded, independently quantized PLE shards from the indexed snapshot."""

    def __init__(self, rows: int, dim: int, shard_count: int):
        super().__init__()
        if rows < 1 or dim < 1 or shard_count < 1 or rows % shard_count:
            raise ValueError("embedded n-gram shard geometry is invalid")
        self.rows = rows
        self.dim = dim
        self.shard_count = shard_count
        self.rows_per_shard = rows // shard_count
        self.shards = [
            nn.Embedding(self.rows_per_shard, dim) for _ in range(shard_count)
        ]
        # oMLX Qwen4Exp checkpoints carry one shared post-dequant scale for
        # every PLE shard. It is a real checkpoint parameter, not metadata.
        self.weight_scale = mx.ones((1,), dtype=mx.bfloat16)

    def __call__(self, ids: mx.array) -> mx.array:
        fused = getattr(self, "fused", None)
        if fused is not None:
            return fused(ids) * self.weight_scale

        flat = ids.reshape(-1)
        mx.eval(flat)
        host_ids = [int(index) for index in flat.tolist()]
        if not host_ids:
            return self.shards[0](flat).reshape(*ids.shape, self.dim)
        if any(index < 0 or index >= self.rows for index in host_ids):
            raise IndexError("embedding index is outside the sharded vocabulary")

        shard_ids = [index // self.rows_per_shard for index in host_ids]
        out = None
        for shard_index in sorted(set(shard_ids)):
            positions_list = [
                position
                for position, current_shard in enumerate(shard_ids)
                if current_shard == shard_index
            ]
            local_ids = [
                host_ids[position] - shard_index * self.rows_per_shard
                for position in positions_list
            ]
            positions = mx.array(positions_list, dtype=mx.int32)
            selected = self.shards[shard_index](mx.array(local_ids, dtype=mx.int32))
            if out is None:
                out = mx.zeros((len(host_ids), self.dim), dtype=selected.dtype)
            out = out.at[positions].add(selected)
        if out is None:
            raise RuntimeError("embedded n-gram table has no shards")
        return out.reshape(*ids.shape, self.dim) * self.weight_scale

    def fuse_quantized_shards(self) -> bool:
        """Join compatible packed shards without dequantizing the PLE table."""

        if getattr(self, "fused", None) is not None:
            return False
        shards = list(self.shards)
        if not shards or not all(
            type(shard) is nn.QuantizedEmbedding for shard in shards
        ):
            return False
        first = shards[0]
        if not all(
            shard.dims == first.dims
            and shard.group_size == first.group_size
            and shard.bits == first.bits
            and shard.mode == first.mode
            and shard.weight.dtype == first.weight.dtype
            and shard.scales.dtype == first.scales.dtype
            and (shard.biases is None) == (first.biases is None)
            and (shard.biases is None or shard.biases.dtype == first.biases.dtype)
            for shard in shards
        ):
            return False

        total_rows = sum(int(shard.weight.shape[0]) for shard in shards)
        if total_rows != self.rows or first.dims != self.dim:
            return False

        fused = nn.QuantizedEmbedding(
            1,
            self.dim,
            group_size=first.group_size,
            bits=first.bits,
            mode=first.mode,
        )
        fused.weight = mx.concatenate([shard.weight for shard in shards], axis=0)
        fused.scales = mx.concatenate([shard.scales for shard in shards], axis=0)
        if first.biases is None:
            fused.biases = None
        else:
            fused.biases = mx.concatenate([shard.biases for shard in shards], axis=0)
        fused.num_embeddings = total_rows
        arrays = [fused.weight, fused.scales]
        if fused.biases is not None:
            arrays.append(fused.biases)
        mx.eval(*arrays)
        self.fused = fused
        self.shards = []
        return True


def _fuse_resident_ple_embeddings(
    model: nn.Module,
    *,
    minimum_physical_memory: int = 192 * 1024**3,
) -> int:
    """Fuse resident PLE shards only when their temporary peak is admissible."""

    physical_memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    if physical_memory < minimum_physical_memory:
        return 0

    fused = 0
    language_model = getattr(model, "language_model", None)
    layers = getattr(getattr(language_model, "model", None), "layers", ())
    for layer in layers:
        table = getattr(
            getattr(getattr(layer, "ple", None), "ple_embedding", None),
            "ngram_embedding",
            None,
        )
        if type(table) is NGramTable and table.fuse_quantized_shards():
            fused += 1
    return fused


class NGramEmbedding(nn.Module):
    def __init__(self, args: TextArgs, ple_index: int):
        super().__init__()
        self.ngram_size = args.ngram_size
        self.context_len = args.ngram_size - 1
        self.heads_per_ngram = args.heads_per_ngram
        self.ngram_heads = (args.ngram_size - 1) * args.heads_per_ngram
        self.eos_id = args.eos_id
        head_dim = args.ple_embed_dim // self.ngram_heads
        sizes, offsets, total = _head_vocab_layout(
            args.ngram_vocab_size_base, self.ngram_heads, ple_index
        )
        div = args.make_ngram_vocab_size_divisible_by
        padded = math.ceil(total / div) * div
        self.layer_multipliers = mx.array(
            _build_layer_multipliers(
                args.vocab_size, args.ngram_size, ple_index, args.seed
            ),
            dtype=mx.int64,
        )
        self.ngram_heads_vocab_sizes = mx.array(sizes, dtype=mx.int64)
        self.ngram_heads_offsets = mx.array(offsets, dtype=mx.int64)
        self.ngram_embedding = NGramTable(
            padded,
            head_dim,
            args.split_ngram_parts,
        )

    def stage(
        self,
        input_ids: mx.array,
        cache: ArraysCache | None,
        state_idx: int,
    ) -> None:
        # Embedded shards stay in the MLX graph; no external table is opened.
        del input_ids, cache, state_idx

    def _shift_ignore_eos(self, ids: mx.array, shift: int) -> mx.array:
        if shift == 0:
            return ids
        batch, length = ids.shape
        pos = mx.arange(length, dtype=mx.int64)[None, :]
        eos_pos = mx.where(
            ids == self.eos_id,
            pos,
            mx.array(-1, dtype=mx.int64),
        )
        previous_inclusive = mx.cummax(eos_pos, axis=1)
        previous = mx.concatenate(
            [
                mx.full((batch, 1), -1, dtype=mx.int64),
                previous_inclusive[:, :-1],
            ],
            axis=1,
        )
        source = pos - shift
        shifted = mx.take_along_axis(ids, mx.maximum(source, 0), axis=1)
        valid = ((pos - (previous + 1)) >= shift) & (source >= 0)
        return mx.where(
            valid,
            shifted,
            mx.array(self.eos_id, dtype=mx.int64),
        )

    def __call__(
        self,
        input_ids: mx.array,
        cache: ArraysCache | None,
        state_idx: int,
    ) -> mx.array:
        ids = input_ids.astype(mx.int64)
        batch, width = ids.shape
        if cache is not None and cache[state_idx] is not None:
            previous = cache[state_idx]
        else:
            previous = mx.full(
                (batch, self.context_len),
                self.eos_id,
                dtype=mx.int64,
            )
        history = mx.concatenate([previous, ids], axis=1)
        if cache is not None:
            cache[state_idx] = history[:, -self.context_len :]

        shifted = [
            self._shift_ignore_eos(history, shift) for shift in range(self.ngram_size)
        ]
        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start = (ngram - 2) * self.heads_per_ngram
            end = start + self.heads_per_ngram
            mixed = shifted[0] * self.layer_multipliers[0]
            for position in range(1, ngram):
                mixed = mx.bitwise_xor(
                    mixed,
                    shifted[position] * self.layer_multipliers[position],
                )
            sizes = self.ngram_heads_vocab_sizes[start:end]
            offsets = self.ngram_heads_offsets[start:end]
            head_ids = mx.remainder(
                mixed[..., None],
                sizes.reshape(1, 1, -1),
            )
            blocks.append(head_ids + offsets.reshape(1, 1, -1))
        ngram_ids = mx.concatenate(blocks, axis=-1)[:, -width:]
        embedded = self.ngram_embedding(ngram_ids)
        return embedded.reshape(batch, width, -1)


class PLELayer(nn.Module):
    """Per-Layer Embedding injection (runs on one linear-attention layer,
    before its hyper-connections). Cache slots: state_idx 2 = conv state,
    state_idx 3 = n-gram context ids."""

    CONV_IDX = 2
    NGRAM_IDX = 3

    def __init__(self, args: TextArgs, ple_index: int):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.hc_count = args.hc_count
        hc_hidden = args.hidden_size * args.hc_count
        self.ple_embedding = NGramEmbedding(args, ple_index)
        self.conv_kernel_size = args.ple_conv_kernel_size
        self.conv_dilation = args.ngram_size
        self.conv_state_len = (self.conv_kernel_size - 1) * self.conv_dilation
        self.key_proj = nn.Linear(args.ple_embed_dim, hc_hidden, bias=False)
        self.value_proj = nn.Linear(args.ple_embed_dim, args.hidden_size, bias=False)
        self.norm_key = GroupedRMSNorm(
            hc_hidden, args.hidden_size, eps=args.rms_norm_eps
        )
        self.norm_query = GroupedRMSNorm(
            hc_hidden, args.hidden_size, eps=args.rms_norm_eps
        )
        self.norm_conv = GroupedRMSNorm(
            hc_hidden, args.hidden_size, eps=args.rms_norm_eps
        )
        # Depthwise dilated conv, stored [channels, kernel, 1] (mlx layout).
        self.conv_weight = mx.zeros((hc_hidden, self.conv_kernel_size, 1))

    def _short_conv(self, x: mx.array, cache: ArraysCache | None) -> mx.array:
        B, S, C = x.shape
        if cache is not None and cache[self.CONV_IDX] is not None:
            state = cache[self.CONV_IDX]
        else:
            state = mx.zeros((B, self.conv_state_len, C), dtype=x.dtype)
        window = mx.concatenate([state, x], axis=1)
        if cache is not None:
            cache[self.CONV_IDX] = window[:, -self.conv_state_len :, :]
        out = mx.conv1d(
            window,
            self.conv_weight,
            stride=1,
            padding=0,
            dilation=self.conv_dilation,
            groups=C,
        )
        return nn.silu(out[:, -S:, :])

    def __call__(self, hidden: mx.array, input_ids: mx.array, cache) -> mx.array:
        emb = self.ple_embedding(input_ids, cache, self.NGRAM_IDX)
        emb = emb.astype(hidden.dtype)
        key = self.norm_key(self.key_proj(emb))
        key = key.reshape(*key.shape[:-1], self.hc_count, self.hidden_size)
        value = self.value_proj(emb)
        query = self.norm_query(hidden)
        query = query.reshape(*query.shape[:-1], self.hc_count, self.hidden_size)
        gate = (key * query).sum(axis=-1, keepdims=True) / math.sqrt(self.hidden_size)
        gate = mx.sqrt(mx.maximum(mx.abs(gate), 1e-6)) * mx.sign(gate)
        gated = mx.sigmoid(gate) * value[..., None, :]
        gated = gated.reshape(*hidden.shape)
        return gated + self._short_conv(self.norm_conv(gated), cache)


class DecoderLayer(nn.Module):
    def __init__(self, args: TextArgs, layer_idx: int):
        super().__init__()
        self.layer_type = args.layer_types[layer_idx]
        self.is_linear = self.layer_type == "linear_attention"
        if self.is_linear:
            self.linear_attn = GatedDeltaNet(args)
        else:
            self.self_attn = Attention(args)
        self.mlp = SparseMoeBlock(args)
        self.attn_hyper_connection = GatedResidual(args)
        self.mlp_hyper_connection = GatedResidual(args)
        if (layer_idx + 1) in args.ple_layer_ids:
            self.ple = PLELayer(args, args.ple_layer_ids.index(layer_idx + 1))
        self._hc = args.hc_count

    def __call__(self, hidden, *, input_ids, ssm_mask, cache):
        if "ple" in self:
            hidden = hidden + self.ple(hidden, input_ids, cache)

        mixed, hyper, inject = self.attn_hyper_connection(hidden)
        if self.is_linear:
            block_out = self.linear_attn(mixed, ssm_mask, cache)
        else:
            block_out = self.self_attn(mixed, cache)
        hidden = hyper + (block_out[..., None, :] * inject[..., :, None]).reshape(
            *hyper.shape
        )

        mixed, hyper, inject = self.mlp_hyper_connection(hidden)
        block_out = self.mlp(mixed)
        hidden = hyper + (block_out[..., None, :] * inject[..., :, None]).reshape(
            *hyper.shape
        )
        return hidden


class Qwen4ExpTextModel(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.hyper_connection_mixer = GatedResidual(args, use_combine=False)
        self.ssm_idx = (
            args.layer_types.index("linear_attention")
            if "linear_attention" in args.layer_types
            else 0
        )
        self._ple_stage_idx = next(
            (
                i
                for i, layer in enumerate(self.layers)
                if getattr(layer, "ple", None) is not None
            ),
            None,
        )
        self.fa_idx = next(
            (i for i, t in enumerate(args.layer_types) if t != "linear_attention"),
            self.ssm_idx,
        )
        capabilities = current_tensor_capabilities()
        self._gdn_compile_explicit_off = not capabilities.has("compiled_gdn")
        self._gdn_compiled_enabled = capabilities.has("compiled_gdn")
        self._gdn_compiled_lane = False
        self._decode_runs = None
        self._decode_run_fns = {}

    def __call__(self, inputs, cache=None, input_embeddings=None):
        h = (
            input_embeddings
            if input_embeddings is not None
            else self.embed_tokens(inputs)
        )
        if cache is None:
            cache = [None] * len(self.layers)
        ssm_mask = create_ssm_mask(h, cache[self.ssm_idx])
        if self._ple_stage_idx is not None:
            ple = self.layers[self._ple_stage_idx].ple
            ple.ple_embedding.stage(inputs, cache[self._ple_stage_idx], ple.NGRAM_IDX)
        h = mx.tile(h, (1, 1, self.args.hc_count))
        if (
            # S<=4 covers AR decode (S=1) and MTP verify widths (S=2..4,
            # depth ceiling 3): mx.compile keys its trace cache on input
            # shapes, so each S gets one retrace then C++ replay, and the
            # GDN states are S-invariant so the same run fns serve all
            # widths. Prefill and masked/padded forwards stay eager.
            1 <= h.shape[1] <= 4
            and h.shape[0] == 1
            and ssm_mask is None
            and not self._gdn_compile_explicit_off
            and (self._gdn_compiled_enabled or self._gdn_compiled_lane)
            and cache[self.ssm_idx] is not None
        ):
            h = self._decode_layers_compiled(h, inputs, cache)
        else:
            capture = _VERIFY_CAPTURE.get()
            for layer, c in zip(self.layers, cache, strict=False):
                if (
                    capture
                    and c is not None
                    and getattr(layer, "ple", None) is not None
                ):
                    c._qwen4_exp_verify_ple = (h, inputs)
                h = layer(h, input_ids=inputs, ssm_mask=ssm_mask, cache=c)
        # The MTP head consumes the pre-mixer widened stream; keep the last
        # one reachable (lazy ref, freed on the next step).
        self._last_widened = h
        return self.hyper_connection_mixer(h)

    # ---- compiled GDN decode runs ----------------------------------------
    # The qL=1 decode step is CPU-dispatch-bound: ~20.8ms of Python graph
    # construction per token against <=14ms of GPU work (measured 2026-08-27,
    # ar-lane census: build=20.79ms wait=0.00ms). GDN layers have FIXED state
    # shapes at decode (conv tape + SSM state), so contiguous non-PLE GDN
    # runs compile once and replay in C++. QSA layers grow their caches every
    # step (KV slab + raw-key concat) and the PLE layer consumes token ids —
    # both stay eager until the slab/graphbank arc.

    def _build_decode_runs(self):
        runs = []
        cur = []
        for i, layer in enumerate(self.layers):
            if layer.is_linear and "ple" not in layer:
                cur.append(i)
            else:
                if cur:
                    runs.append(("run", tuple(cur)))
                    cur = []
                runs.append(("eager", i))
        if cur:
            runs.append(("run", tuple(cur)))
        return runs

    def _compiled_run_fn(self, idxs, capture: bool = False):
        layers = [self.layers[i] for i in idxs]

        def step(h, *flat):
            out_states = []
            rows = []
            k = 0
            for layer in layers:
                c = ArraysCache(size=2)
                c[0], c[1] = flat[k], flat[k + 1]
                k += 2
                h = layer(h, input_ids=None, ssm_mask=None, cache=c)
                out_states.extend((c[0], c[1]))
                if capture:
                    # __call__ ran under the capture scope during THIS trace,
                    # so the temp cache carries the tracer rows — surface
                    # them as compiled outputs.
                    rows.extend(c._qwen4_exp_verify_rows)
            return (h, *out_states, *rows)

        return mx.compile(step)

    def _get_run_fn(self, idxs, capture: bool):
        key = (idxs, bool(capture))
        fn = self._decode_run_fns.get(key)
        if fn is None:
            fn = self._compiled_run_fn(idxs, capture=capture)
            self._decode_run_fns[key] = fn
        return fn

    def _decode_layers_compiled(self, h, inputs, cache):
        if self._decode_runs is None:
            self._decode_runs = self._build_decode_runs()
        capture = _VERIFY_CAPTURE.get()
        for kind, payload in self._decode_runs:
            if kind == "eager":
                i = payload
                if capture and getattr(self.layers[i], "ple", None) is not None:
                    cache[i]._qwen4_exp_verify_ple = (h, inputs)
                h = self.layers[i](h, input_ids=inputs, ssm_mask=None, cache=cache[i])
                continue
            idxs = payload
            flat = []
            usable = True
            for i in idxs:
                s0, s1 = cache[i][0], cache[i][1]
                if s0 is None or s1 is None:
                    usable = False
                    break
                flat.extend((s0, s1))
            if not usable:
                for i in idxs:
                    h = self.layers[i](
                        h, input_ids=inputs, ssm_mask=None, cache=cache[i]
                    )
                continue
            out = self._get_run_fn(idxs, capture)(h, *flat)
            h = out[0]
            k = 1
            for i in idxs:
                cache[i][0] = out[k]
                cache[i][1] = out[k + 1]
                k += 2
            if capture:
                for i in idxs:
                    cache[i]._qwen4_exp_verify_rows = tuple(out[k : k + 6])
                    k += 6
        return h

    def clear_verify_capture(self, cache) -> None:
        for entry in cache:
            if entry is None:
                continue
            for attr in ("_qwen4_exp_verify_rows", "_qwen4_exp_verify_ple"):
                if getattr(entry, attr, None) is not None:
                    setattr(entry, attr, None)

    def _refuse_commit(self, layer_index: int, reason: str) -> bool:
        """Refuse before mutation so the target can rollback and re-forward."""
        del layer_index, reason
        return False

    def commit_verified_window(
        self,
        cache,
        snapshot_states,
        *,
        keep_tokens: int,
        verified_tokens: int,
    ) -> bool:
        """Repair-free commit of a speculative verify window.

        Trimmable entries (QSA attention) trim their uncommitted tail; each
        pure-GDN layer replays ONLY its gated-delta recurrence over the kept
        rows from the pre-verify snapshot state; the single PLE-carrying
        layer replays its full (cheap, <=window-rows) layer forward from its
        snapshot slots. Everything is lazy — no eval here; the next round's
        eval pulls the replay. Validates every entry before mutating any so
        a refusal leaves the cache intact for the rollback+re-forward
        fallback. Returns True when the commit landed.
        """
        from mlx_lm.models.gated_delta import gated_delta_update

        keep_tokens = int(keep_tokens)
        verified_tokens = int(verified_tokens)
        trim_n = verified_tokens - keep_tokens
        if keep_tokens < 1 or trim_n < 0 or len(cache) != len(self.layers):
            return False

        plan = []
        for i, (layer, entry) in enumerate(zip(self.layers, cache, strict=False)):
            if entry is None:
                return self._refuse_commit(i, "entry_missing")
            if callable(getattr(entry, "is_trimmable", None)) and entry.is_trimmable():
                plan.append(("trim", i, None))
                continue
            pre = snapshot_states[i] if snapshot_states is not None else None
            if pre is None:
                return self._refuse_commit(i, "snapshot_missing")
            if getattr(layer, "ple", None) is not None:
                cap = getattr(entry, "_qwen4_exp_verify_ple", None)
                if cap is None:
                    return self._refuse_commit(i, "ple_rows_missing")
                if cap[0].shape[1] != verified_tokens:
                    return self._refuse_commit(
                        i, f"ple_rows_width_{cap[0].shape[1]}_vs_{verified_tokens}"
                    )
                if len(pre) < 4:
                    return self._refuse_commit(i, "ple_snapshot_short")
                plan.append(("ple", i, cap))
                continue
            rows = getattr(entry, "_qwen4_exp_verify_rows", None)
            if rows is None:
                return self._refuse_commit(i, "gdn_rows_missing")
            if rows[0].shape[1] != verified_tokens:
                return self._refuse_commit(
                    i, f"gdn_rows_width_{rows[0].shape[1]}_vs_{verified_tokens}"
                )
            if len(pre) < 2 or pre[1] is None:
                return self._refuse_commit(i, "gdn_snapshot_short")
            plan.append(("gdn", i, rows))

        for kind, i, payload in plan:
            entry = cache[i]
            layer = self.layers[i]
            if kind == "trim":
                if trim_n:
                    entry.trim(trim_n)
                continue
            pre = snapshot_states[i]
            if kind == "gdn":
                qkv, q, k, v, a, b = payload
                gdn = layer.linear_attn
                conv_pre = pre[0]
                if conv_pre is None:
                    conv_pre = mx.zeros(
                        (qkv.shape[0], gdn.conv_kernel_size - 1, qkv.shape[2]),
                        dtype=qkv.dtype,
                    )
                _, new_state = gated_delta_update(
                    q[:, :keep_tokens],
                    k[:, :keep_tokens],
                    v[:, :keep_tokens],
                    a[:, :keep_tokens],
                    b[:, :keep_tokens],
                    gdn.A_log,
                    gdn.dt_bias,
                    pre[1],
                    None,
                    use_kernel=not gdn.training,
                )
                conv_input = mx.concatenate([conv_pre, qkv[:, :keep_tokens]], axis=1)
                entry[0] = mx.contiguous(
                    conv_input[:, -(gdn.conv_kernel_size - 1) :, :]
                )
                entry[1] = new_state
                entry._qwen4_exp_verify_rows = None
            else:  # ple
                h_in, ids = payload
                for j in range(len(pre)):
                    entry[j] = pre[j]
                ple = layer.ple
                kept_ids = ids[:, :keep_tokens]
                ple.ple_embedding.stage(kept_ids, entry, ple.NGRAM_IDX)
                layer(
                    h_in[:, :keep_tokens],
                    input_ids=kept_ids,
                    ssm_mask=None,
                    cache=entry,
                )
                entry._qwen4_exp_verify_ple = None
        return True


class TextModel(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.args = args
        self.model = Qwen4ExpTextModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs,
        cache=None,
        input_embeddings=None,
        return_hidden: bool = False,
        hidden_variant: str | None = None,
        emit_logits: bool = True,
        logits_keep: int = 0,
    ):
        # hidden_variant is accepted for the runtime contract's sake but this
        # family has exactly one draft input: the pre-mixer WIDENED stream.
        # emit_logits/logits_keep are the sustained-prefill contract: a
        # cache-only chunk skips the [1, S, 248320] head matmul entirely
        # (~1.02 GB per 2048-token chunk that used to be built and thrown
        # away 128 times per 262K cold prefill — the #393 audit receipt).
        del hidden_variant
        out = self.model(inputs, cache, input_embeddings)
        if not emit_logits:
            return (None, self.model._last_widened) if return_hidden else None
        if logits_keep:
            out = out[:, -max(1, int(logits_keep)) :]
        if self.args.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(out)
        else:
            logits = self.lm_head(out)
        if return_hidden:
            return logits, self.model._last_widened
        return logits

    def _head_logits(self, h):
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(h)
        return self.lm_head(h)

    def make_mtp_cache(self):
        return [
            QSACache(self.model.args.indexer_compress_ratio)
            for _ in range(self.model.args.mtp.num_hidden_layers)
        ]

    def make_cache(self):
        ratio = self.model.args.indexer_compress_ratio
        caches = []
        for _i, layer in enumerate(self.model.layers):
            if not layer.is_linear:
                caches.append(QSACache(ratio))
            elif "ple" in layer:
                caches.append(ArraysCache(size=4))
            else:
                caches.append(ArraysCache(size=2))
        return caches


class Qwen4ExpMTP(nn.Module):
    """Flash-Next MTP head, reconstructed from the shipped tensors (no public
    reference implements it — transformers ships only the trunk).

    Wiring (the only reading consistent with every tensor shape): the trunk's
    pre-mixer WIDENED stream [B,S,hc*d] is RMS-normed at full width
    (pre_fc_norm_hidden is [hc*d]); each 2560-wide substream goes through the
    SHARED fc_hidden [d,d]; the normed+projected token embedding
    (pre_fc_norm_embedding -> fc_embedding, both [d]-sized) is broadcast-added
    into every substream; the fused widened stream runs the configured MTP
    full-attention layer stack (one layer in this snapshot), then this head's
    own mixer collapses back to d for the SHARED trunk lm_head.

    Correctness is graded by measured acceptance — the probability-ratio
    verify contract keeps outputs exact for ANY draft head, so a mis-wiring
    can only cost speed, never quality.
    """

    def __init__(self, args: TextArgs):
        super().__init__()
        if args.mtp_use_dedicated_embeddings:
            raise ValueError(
                "dedicated MTP embeddings are not present in this tensor ABI"
            )
        d = args.hidden_size
        self.pre_fc_norm_embedding = GroupedRMSNorm(d, eps=args.rms_norm_eps)
        self.pre_fc_norm_hidden = GroupedRMSNorm(
            d * args.hc_count,
            eps=args.rms_norm_eps,
        )
        self.fc_embedding = nn.Linear(d, d, bias=False)
        self.fc_hidden = nn.Linear(d, d, bias=False)
        mtp_args = MtpTextArgs(args)
        self.layers = [
            DecoderLayer(mtp_args, index) for index in range(mtp_args.num_hidden_layers)
        ]
        self.hyper_connection_mixer = GatedResidual(args, use_combine=False)
        self._hc = args.hc_count

    def fuse_and_run(self, widened: mx.array, tok_emb: mx.array, cache) -> mx.array:
        """Fuse (widened, token embedding) and run the head's layer; returns
        the PRE-mixer widened output — the recursion state for deeper drafts."""
        B, S, W = widened.shape
        hn = self.pre_fc_norm_hidden(widened).reshape(B, S, self._hc, -1)
        en = self.fc_embedding(self.pre_fc_norm_embedding(tok_emb))
        fused = self.fc_hidden(hn) + en[:, :, None, :]
        h = fused.reshape(B, S, W)
        for index, layer in enumerate(self.layers):
            layer_cache = cache[index] if cache is not None else None
            h = layer(h, input_ids=None, ssm_mask=None, cache=layer_cache)
        return h

    def __call__(self, widened: mx.array, tok_emb: mx.array, cache) -> mx.array:
        return self.hyper_connection_mixer(self.fuse_and_run(widened, tok_emb, cache))


class Model(nn.Module):
    """Target-owned text and embedded-MTP tensor tree for one immutable plan."""

    def __init__(
        self,
        config: Qwen4ExpCheckpointConfig,
        capabilities: Qwen4ExpTensorCapabilities,
    ) -> None:
        super().__init__()
        self.config = config
        self.capabilities = capabilities
        self.model_type = config.model_type
        args = TextArgs(config.text, capabilities)
        with tensor_capability_scope(capabilities):
            self.language_model = TextModel(args)
            self.mtp = Qwen4ExpMTP(args)

    def __call__(
        self,
        inputs,
        cache=None,
        input_embeddings=None,
        return_hidden: bool = False,
        hidden_variant: str | None = None,
        emit_logits: bool = True,
        logits_keep: int = 0,
    ):
        _require_supported_batch(inputs, "text inputs", cache)
        if input_embeddings is not None:
            _require_supported_batch(input_embeddings, "text input embeddings", cache)
            if input_embeddings.shape[0] != inputs.shape[0]:
                raise ValueError("text inputs and embeddings must share batch size")
        with tensor_capability_scope(self.capabilities):
            return self.language_model(
                inputs,
                cache,
                input_embeddings,
                return_hidden=return_hidden,
                hidden_variant=hidden_variant,
                emit_logits=emit_logits,
                logits_keep=logits_keep,
            )

    @property
    def layers(self):
        return self.language_model.model.layers

    def make_cache(self):
        return self.language_model.make_cache()

    def make_mtp_cache(self):
        return self.language_model.make_mtp_cache()

    def mtp_forward(
        self,
        hidden_states,
        next_token_ids,
        mtp_cache=None,
        concat_order: str | None = None,
        return_hidden: bool = False,
        mtp_hidden_variant: str | None = None,
        position_offset: int | None = None,
    ):
        del concat_order, mtp_hidden_variant, position_offset
        _require_supported_batch(hidden_states, "MTP hidden states", mtp_cache)
        _require_supported_batch(next_token_ids, "MTP token ids", mtp_cache)
        if hidden_states.shape[0] != next_token_ids.shape[0]:
            raise ValueError("MTP hidden states and token ids must share batch size")
        with tensor_capability_scope(self.capabilities):
            emb = self.language_model.model.embed_tokens(next_token_ids)
            hidden = self.mtp.fuse_and_run(
                hidden_states,
                emb,
                mtp_cache,
            )
            logits = self.language_model._head_logits(
                self.mtp.hyper_connection_mixer(hidden)
            )
            if return_hidden:
                return logits, hidden
            return logits

    def mtp_update_cache(
        self,
        hidden_states,
        next_token_ids,
        mtp_cache=None,
        concat_order: str | None = None,
        mtp_hidden_variant: str | None = None,
        position_offset: int | None = None,
        input_embeddings=None,
    ):
        del concat_order, mtp_hidden_variant, position_offset
        _require_supported_batch(hidden_states, "MTP hidden states", mtp_cache)
        _require_supported_batch(next_token_ids, "MTP token ids", mtp_cache)
        if hidden_states.shape[0] != next_token_ids.shape[0]:
            raise ValueError("MTP hidden states and token ids must share batch size")
        if input_embeddings is not None:
            _require_supported_batch(
                input_embeddings,
                "MTP input embeddings",
                mtp_cache,
            )
            if input_embeddings.shape[0] != next_token_ids.shape[0]:
                raise ValueError("MTP embeddings and token ids must share batch size")
        with tensor_capability_scope(self.capabilities):
            embedding = (
                input_embeddings
                if input_embeddings is not None
                else self.language_model.model.embed_tokens(next_token_ids)
            )
            return self.mtp.fuse_and_run(
                hidden_states,
                embedding,
                mtp_cache,
            )

    def mtp_draft_logits(self, widened: mx.array, tok_emb: mx.array, cache):
        with tensor_capability_scope(self.capabilities):
            hidden = self.mtp(widened, tok_emb, cache)
            return self.language_model._head_logits(hidden)

    def verify_capture_scope(self):
        return verify_capture_scope()

    def clear_verify_capture(self, cache) -> None:
        self.language_model.model.clear_verify_capture(cache)

    def commit_verified_window(
        self,
        cache,
        snapshot_states,
        *,
        keep_tokens: int,
        verified_tokens: int,
    ) -> bool:
        with tensor_capability_scope(self.capabilities):
            return self.language_model.model.commit_verified_window(
                cache,
                snapshot_states,
                keep_tokens=keep_tokens,
                verified_tokens=verified_tokens,
            )

    def snapshot(self, bundle: object) -> object:
        cache = _require_cache_bundle(bundle)
        return tuple(_snapshot_value(getattr(entry, "state", None)) for entry in cache)

    def begin_capture(self, bundle: object) -> object:
        _require_cache_bundle(bundle)
        scope = self.verify_capture_scope()
        scope.__enter__()
        return scope

    def end_capture(self, bundle: object, capture_token: object) -> None:
        _require_cache_bundle(bundle)
        capture_token.__exit__(None, None, None)

    def rollback_after_verify(
        self,
        bundle: object,
        snapshot: object,
        verified_tokens: int,
    ) -> None:
        cache = _require_cache_bundle(bundle)
        states = tuple(snapshot)
        if len(states) != len(cache):
            raise ValueError("cache snapshot shape changed")
        for entry, state in zip(cache, states, strict=True):
            if callable(getattr(entry, "is_trimmable", None)) and entry.is_trimmable():
                entry.trim(verified_tokens)
            if state is not None:
                entry.state = _snapshot_value(state)

    def sanitize(self, weights: Mapping[str, mx.array]) -> dict[str, mx.array]:
        """Normalize HF names while retaining embedded PLE and MTP tensors."""

        raw = any(
            key.endswith("conv1d.weight") and value.shape[-1] != 1
            for key, value in weights.items()
        ) or any(key.startswith("model.language_model.") for key in weights)
        out: dict[str, mx.array] = {}
        stacked: dict[str, dict[int, mx.array]] = {}
        for original_key, source_value in weights.items():
            key = original_key
            value = source_value
            if key.startswith("model.visual.") or key.startswith("vision_tower."):
                continue
            if key.startswith("mtp."):
                out[key] = value
                continue
            if key.startswith("model.language_model."):
                key = key.replace(
                    "model.language_model.",
                    "language_model.model.",
                    1,
                )
            elif key == "lm_head.weight":
                key = "language_model.lm_head.weight"
            elif not key.startswith("language_model."):
                key = "language_model." + key

            if raw:
                if (
                    ".mlp.experts." in key
                    and ".weight" in key
                    and "scale_inv" not in key
                ):
                    prefix, rest = key.split(".mlp.experts.", 1)
                    index_text, projection_rest = rest.split(".", 1)
                    projection = projection_rest.rsplit(".weight", 1)[0]
                    destination = f"{prefix}.mlp.switch_mlp.{projection}.weight"
                    stacked.setdefault(destination, {})[int(index_text)] = value
                    continue
                if ".mlp.experts.gate_up_proj" in key:
                    prefix = key.split(".mlp.experts.", 1)[0]
                    hidden = self.language_model.args.hidden_size
                    if value.shape[1] == hidden:
                        gate, up = mx.split(value, 2, axis=-1)
                        gate = gate.swapaxes(1, 2)
                        up = up.swapaxes(1, 2)
                    else:
                        gate, up = mx.split(value, 2, axis=1)
                    out[f"{prefix}.mlp.switch_mlp.gate_proj.weight"] = gate
                    out[f"{prefix}.mlp.switch_mlp.up_proj.weight"] = up
                    continue
                if ".mlp.experts.down_proj" in key:
                    prefix = key.split(".mlp.experts.", 1)[0]
                    hidden = self.language_model.args.hidden_size
                    if value.shape[2] == hidden:
                        value = value.swapaxes(1, 2)
                    out[f"{prefix}.mlp.switch_mlp.down_proj.weight"] = value
                    continue
                if key.endswith("ple.conv1d.weight"):
                    out[key.replace("ple.conv1d.weight", "ple.conv_weight")] = (
                        value.moveaxis(2, 1)
                    )
                    continue
                if key.endswith("linear_attn.conv1d.weight") and value.shape[-1] != 1:
                    value = value.moveaxis(2, 1)
            elif key.endswith("ple.conv1d.weight"):
                key = key.replace("ple.conv1d.weight", "ple.conv_weight")
            if key in out:
                raise ValueError(f"duplicate sanitized weight: {key}")
            out[key] = value

        for destination, parts in stacked.items():
            expected = tuple(range(len(parts)))
            if tuple(sorted(parts)) != expected:
                raise ValueError(f"expert weights are not contiguous: {destination}")
            out[destination] = mx.stack([parts[index] for index in expected])
        _normalize_ones_centered_rmsnorm_weights(self, out)
        trunk = self.language_model.model
        out = _fuse_gate_up_sanitize(trunk, out)
        out = _fuse_gdn_in_proj_sanitize(trunk, out)
        return _fuse_qsa_qkv_sanitize(trunk, out)


def _snapshot_value(value: Any) -> Any:
    if isinstance(value, mx.array):
        return value[...]
    if isinstance(value, tuple):
        return tuple(_snapshot_value(item) for item in value)
    if isinstance(value, list):
        return [_snapshot_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _snapshot_value(item) for key, item in value.items()}
    return value


def _restore_cache_bundle(bundle: object, snapshot: object) -> tuple[object, ...]:
    cache = _require_cache_bundle(bundle)
    states = tuple(snapshot)
    if len(states) != len(cache):
        raise ValueError("prefix cache snapshot shape changed")
    restored: list[object] = []
    for entry, state in zip(cache, states, strict=True):
        if state is None:
            raise ValueError("prefix cache snapshot contains an empty layer state")
        copied = _snapshot_value(state)
        entry.state = copied
        restored.append(copied)
    return tuple(restored)


def _eval_tensor_values(*values: object) -> None:
    leaves: list[Any] = []

    def collect(value: object) -> None:
        if isinstance(value, mx.array):
            leaves.append(value)
        elif isinstance(value, Mapping):
            for item in value.values():
                collect(item)
        elif isinstance(value, tuple | list):
            for item in value:
                collect(item)

    for value in values:
        collect(value)
    if leaves:
        mx.eval(*leaves)


def _require_supported_batch(value: Any, name: str, cache: object) -> None:
    shape = getattr(value, "shape", None)
    if shape is None or not shape or shape[0] < 1:
        raise ValueError(f"{name} must have a positive batch size")
    if shape[0] == 1:
        return
    if not isinstance(cache, list) or not any(
        isinstance(entry, _QSABatchCache) for entry in cache
    ):
        raise ValueError(f"{name} multi-row input requires a packed QSA cache")


def _merge_batch_cache(rows: Sequence[list[Any]]) -> list[Any]:
    """Pack independent semantic caches without changing their owners."""

    if not rows:
        raise ValueError("cache batch requires at least one row")
    layer_count = len(rows[0])
    if any(len(row) != layer_count for row in rows):
        raise ValueError("cache rows must share one layer topology")
    merged: list[Any] = []
    for layer_index in range(layer_count):
        entries = tuple(row[layer_index] for row in rows)
        if all(isinstance(entry, QSACache) for entry in entries):
            merged.append(_QSABatchCache(entries))
        elif all(isinstance(entry, ArraysCache) for entry in entries):
            merged.append(_merge_arrays_cache(entries))
        else:
            kinds = ", ".join(type(entry).__name__ for entry in entries)
            raise TypeError(
                f"cache layer {layer_index} cannot form a semantic batch: {kinds}"
            )
    return merged


def _merge_arrays_cache(entries: Sequence[ArraysCache]) -> ArraysCache:
    """Merge independent array caches while preserving sparse empty slots."""

    if not entries:
        raise ValueError("array cache batch requires at least one row")
    slot_count = len(entries[0].cache)
    if any(len(entry.cache) != slot_count for entry in entries):
        raise ValueError("array cache rows must share one slot topology")
    merged = ArraysCache(slot_count)
    for slot in range(slot_count):
        source = next(
            (entry[slot] for entry in entries if entry[slot] is not None),
            None,
        )
        if source is None:
            continue
        if any(
            entry[slot] is not None and entry[slot].shape[1:] != source.shape[1:]
            for entry in entries
        ):
            raise ValueError(f"array cache slot {slot} changed shape across rows")
        packed = mx.zeros(
            (len(entries), *source.shape[1:]),
            dtype=source.dtype,
        )
        for row_index, entry in enumerate(entries):
            if entry[slot] is not None:
                packed[row_index : row_index + 1] = entry[slot]
        merged[slot] = packed
    for attr in ("left_padding", "lengths"):
        values = tuple(getattr(entry, attr, None) for entry in entries)
        source = next((value for value in values if value is not None), None)
        if source is not None:
            setattr(
                merged,
                attr,
                mx.concatenate(
                    tuple(
                        mx.zeros_like(source[:1]) if value is None else value
                        for value in values
                    )
                ),
            )
    return merged


def _scatter_batch_cache(batch: list[Any], rows: Sequence[list[Any]]) -> None:
    """Split a transient cache batch back into the same row identities."""

    for layer_index, entry in enumerate(batch):
        if isinstance(entry, _QSABatchCache):
            entry.scatter_rows()
            continue
        if not isinstance(entry, ArraysCache):
            raise TypeError(
                f"cache layer {layer_index} has unsupported batch type "
                f"{type(entry).__name__}"
            )
        for row_index, row in enumerate(rows):
            extracted = ArraysCache(len(entry.cache))
            extracted.cache = [
                None if leaf is None else leaf[row_index : row_index + 1]
                for leaf in entry.cache
            ]
            extracted.left_padding = (
                None
                if entry.left_padding is None
                else entry.left_padding[row_index : row_index + 1]
            )
            extracted.lengths = (
                None
                if entry.lengths is None
                else entry.lengths[row_index : row_index + 1]
            )
            for attr in ("_qwen4_exp_verify_rows", "_qwen4_exp_verify_ple"):
                captured = getattr(entry, attr, None)
                if captured is not None:
                    setattr(
                        extracted,
                        attr,
                        _slice_batch_capture(captured, row_index),
                    )
            row[layer_index] = extracted


def _slice_batch_capture(value: Any, row_index: int) -> Any:
    """Take one identity-preserving row from a verify capture tree."""

    if isinstance(value, mx.array):
        return value[row_index : row_index + 1]
    if isinstance(value, tuple):
        return tuple(_slice_batch_capture(item, row_index) for item in value)
    if isinstance(value, list):
        return [_slice_batch_capture(item, row_index) for item in value]
    raise TypeError(
        f"Qwen4Exp verify capture contains an unsupported value: {type(value).__name__}"
    )


def _require_cache_bundle(bundle: object) -> list[Any]:
    if not isinstance(bundle, list):
        raise TypeError("Qwen4Exp semantic cache bundle must be a list")
    return bundle


def _quantize_for_plan(model: Model, plan: Qwen4ExpModelLoadPlan) -> None:
    config = plan.config
    quantized_paths = {
        key.rsplit(".", 1)[0]
        for key in plan.artifacts.weight_keys
        if key.endswith(".scales")
        and not key.startswith(("model.visual.", "vision_tower."))
    }
    matched_paths: set[str] = set()

    def predicate(path: str, module: nn.Module) -> bool | dict:
        if path not in quantized_paths:
            return False
        if not hasattr(module, "to_quantized"):
            raise TypeError(f"planned quantized tensor has no module support: {path}")
        matched_paths.add(path)
        bits, group_size, mode = config.quantization_recipe(path)
        return {"bits": bits, "group_size": group_size, "mode": mode}

    nn.quantize(
        model,
        group_size=config.quantization_group_size,
        bits=config.quantization_bits,
        mode=config.quantization_mode,
        class_predicate=predicate,
    )
    missing = sorted(quantized_paths - matched_paths)
    if missing:
        raise ValueError(f"planned quantized tensors have no modules: {missing!r}")


def _read_indexed_weights(plan: Qwen4ExpModelLoadPlan) -> dict[str, mx.array]:
    weights: dict[str, mx.array] = {}
    for shard_name in plan.artifacts.weight_shards:
        shard = mx.load(str(Path(plan.model_dir) / shard_name))
        for key, value in shard.items():
            if key in weights:
                raise ValueError(f"duplicate checkpoint tensor: {key}")
            weights[key] = value
    observed = tuple(sorted(weights))
    if observed != plan.artifacts.weight_keys:
        missing = sorted(set(plan.artifacts.weight_keys) - set(observed))
        extra = sorted(set(observed) - set(plan.artifacts.weight_keys))
        raise ValueError(
            f"checkpoint index mismatch: missing={missing!r}, extra={extra!r}"
        )
    return weights


def load_qwen4_exp_tensor(
    plan: Qwen4ExpModelLoadPlan,
    capabilities: Qwen4ExpTensorCapabilities,
) -> Model:
    """Build, quantize and strictly load exactly the immutable planned snapshot."""

    with tensor_capability_scope(capabilities):
        model = Model(plan.config, capabilities)
        _quantize_for_plan(model, plan)
        weights = model.sanitize(_read_indexed_weights(plan))
        model.load_weights(list(weights.items()), strict=True)
        if capabilities.has("fused_ple"):
            _fuse_resident_ple_embeddings(model)
        model.eval()
    return model


@dataclass(frozen=True, slots=True)
class _TensorSamplingConfig:
    max_output_tokens: int
    target: SamplerConfig
    draft: SamplerConfig
    seed: int

    @property
    def needs_rng(self) -> bool:
        return self.target.temperature > 0 or self.draft.temperature > 0


_PREFIX_PAYLOAD_SCHEMA = "qwen4exp-qsa-gdn-ple-mtp-v1"


@dataclass(frozen=True, slots=True)
class _TensorPrefixConfig:
    block_size_tokens: int
    max_hot_entries: int
    max_hot_tokens: int
    max_active_leases: int


@dataclass(frozen=True, slots=True)
class _TensorPrefixPayload:
    target_cache_state: object
    mtp_cache_state: object
    logits: Any
    hidden: Any
    position: int


@dataclass(slots=True)
class _TensorOutputState:
    stop_token_ids: frozenset[int]
    think_start_id: int | None
    think_end_id: int | None
    in_reasoning: bool
    trim_visible_prefix: bool = False
    reasoning_tokens: int = 0

    def is_protocol_token(self, token_id: int) -> bool:
        return token_id in self.stop_token_ids or token_id in {
            self.think_start_id,
            self.think_end_id,
        }

    def route(self, token_id: int, text: str) -> tuple[str, str]:
        if token_id in self.stop_token_ids:
            return "", ""
        if self.think_start_id is not None and token_id == self.think_start_id:
            self.in_reasoning = True
            return "", ""
        if self.think_end_id is not None and token_id == self.think_end_id:
            self.in_reasoning = False
            self.trim_visible_prefix = True
            return "", ""
        if self.in_reasoning:
            self.reasoning_tokens += 1
            return "", text
        if self.trim_visible_prefix and text.isspace():
            return "", ""
        self.trim_visible_prefix = False
        return text, ""


@dataclass(slots=True)
class _TensorReservation:
    request: GenerationRequest
    prepared_request: PreparedGenerationRequest
    lease_id: str
    cache: list[Any]
    mtp_cache: list[Any]
    prompt_tokens: tuple[int, ...]
    max_output_tokens: int
    sampler: SamplerConfig
    draft_sampler: SamplerConfig
    rng: np.random.Generator | None
    prefix_lease: Qwen4ExpPrefixLeaseIdentity
    prefix_context_fingerprint: str
    output_state: _TensorOutputState
    detokenizer: Any
    prefix_hit: bool = False
    pending_prefix: Qwen4ExpPendingBoundaryCheckpoint | None = None
    pending_prefix_tokens: tuple[int, ...] = ()
    input_embeddings: Any = None
    position_table: Any = None
    mrope: Qwen4ExpTensorMrope | None = None
    position: int = 0
    output_tokens: int = 0
    pending_primary: int | None = None
    logits: Any = None
    hidden: Any = None
    aborted: bool = False


@dataclass(frozen=True, slots=True)
class _TensorDecodeOutcome:
    tokens: tuple[int, ...]
    finished: bool
    finish_reason: str | None = None
    ar_decode_tokens: int = 0
    mtp_rounds: int = 0
    mtp_drafted_tokens: int = 0
    mtp_accepted_tokens: int = 0
    mtp_rejected_tokens: int = 0


_TENSOR_FORWARD_SCHEMA = "qwen4-exp.tensor-forward.v1"
_TENSOR_FORWARD_RESET = "per_tensor_runtime_instance"
_TENSOR_FORWARD_PHASES = (
    "target_decode",
    "mtp_draft",
    "target_verify",
    "target_correction",
    "mtp_history_update",
)


@dataclass(slots=True)
class _TensorForwardPhaseTelemetry:
    completed_calls: int = 0
    completed_rows: int = 0
    max_completed_rows: int = 0
    completed_calls_by_shape: dict[str, int] = field(default_factory=dict)

    def record(self, *, batch_rows: int, sequence_length: int) -> None:
        if batch_rows < 1 or sequence_length < 1:
            raise ValueError("completed tensor forwards require positive BxS shape")
        shape = f"{batch_rows}x{sequence_length}"
        self.completed_calls += 1
        self.completed_rows += batch_rows
        self.max_completed_rows = max(self.max_completed_rows, batch_rows)
        self.completed_calls_by_shape[shape] = (
            self.completed_calls_by_shape.get(shape, 0) + 1
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "completed_calls": self.completed_calls,
            "completed_rows": self.completed_rows,
            "max_completed_rows": self.max_completed_rows,
            "completed_calls_by_shape": dict(self.completed_calls_by_shape),
        }


@dataclass(slots=True)
class _TensorForwardTelemetry:
    runtime_instance_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    phases: dict[str, _TensorForwardPhaseTelemetry] = field(
        default_factory=lambda: {
            phase: _TensorForwardPhaseTelemetry() for phase in _TENSOR_FORWARD_PHASES
        }
    )

    def record(
        self,
        phase: str,
        *,
        batch_rows: int,
        sequence_length: int,
    ) -> None:
        try:
            counters = self.phases[phase]
        except KeyError as error:
            raise ValueError(f"unknown tensor forward phase: {phase}") from error
        counters.record(batch_rows=batch_rows, sequence_length=sequence_length)

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema": _TENSOR_FORWARD_SCHEMA,
            "runtime_instance_id": self.runtime_instance_id,
            "reset_semantics": _TENSOR_FORWARD_RESET,
            "phases": {
                phase: self.phases[phase].snapshot() for phase in _TENSOR_FORWARD_PHASES
            },
        }


_VISION_IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"


def _require_prepared_vision_prompt(
    request: GenerationRequest,
    value: object,
) -> PreparedQwen4Prompt:
    if not isinstance(value, PreparedQwen4Prompt):
        raise ValueError("vision execution requires a sealed PreparedQwen4Prompt")
    if value.response_id != request.response_id:
        raise ValueError("prepared prompt response identity changed")
    if value.runtime != request.runtime:
        raise ValueError("prepared prompt runtime identity changed")
    if not value.messages:
        raise ValueError("prepared prompt must seal its message layout")
    if tuple(item.resolution for item in value.media) != value.resolution.items:
        raise ValueError("prepared prompt resolution layout changed")

    raw_messages = tuple(request.messages)
    if len(value.messages) != len(raw_messages):
        raise ValueError("prepared prompt message count changed")
    media_slots: dict[int, set[int]] = {}
    for raw_media in request.media:
        if not isinstance(raw_media, Mapping):
            raise ValueError("canonical media item must be a mapping")
        message_index = raw_media.get("_message_index")
        content_index = raw_media.get("_content_index")
        if (
            isinstance(message_index, bool)
            or not isinstance(message_index, int)
            or isinstance(content_index, bool)
            or not isinstance(content_index, int)
            or message_index < 0
            or content_index < 0
        ):
            raise ValueError("canonical media provenance is invalid")
        if message_index >= len(raw_messages):
            raise ValueError("canonical media refers to a missing message")
        role = raw_media.get("_role")
        if not isinstance(role, str) or not role.strip():
            raise ValueError("canonical media role provenance is invalid")
        slots = media_slots.setdefault(message_index, set())
        if content_index in slots:
            raise ValueError("canonical media content index is duplicated")
        slots.add(content_index)

    for index, (raw_message, sealed_message) in enumerate(
        zip(raw_messages, value.messages, strict=True)
    ):
        if not isinstance(raw_message, Mapping):
            raise ValueError("canonical message must be a mapping")
        role = raw_message.get("role")
        if not isinstance(role, str) or not role.strip():
            raise ValueError("canonical message role must not be empty")
        normalized_role = role.strip().lower()
        if (
            sealed_message.message_index != index
            or sealed_message.role != normalized_role
        ):
            raise ValueError("prepared prompt role/message layout changed")
        item_type = raw_message.get("type")
        if item_type not in (None, "message", "function_call_output"):
            raise ValueError("canonical message type is unsupported")
        if sealed_message.item_type != item_type:
            raise ValueError("prepared prompt message type changed")
        expected_call_id = raw_message.get("call_id")
        expected_output = raw_message.get("output")
        if (
            sealed_message.call_id != expected_call_id
            or sealed_message.output != expected_output
        ):
            raise ValueError("prepared prompt tool receipt identity changed")
        if any(
            str(item.get("_role", "")).strip().lower() != normalized_role
            for item in request.media
            if isinstance(item, Mapping) and item.get("_message_index") == index
        ):
            raise ValueError("canonical media role provenance changed")
        expected_text = _canonical_message_text(raw_message.get("content"))
        expected_kinds = _expected_message_item_kinds(
            len(expected_text),
            media_slots.get(index, set()),
        )
        actual_kinds, actual_text = _sealed_message_layout(sealed_message.items)
        if actual_kinds != expected_kinds or actual_text != expected_text:
            raise ValueError("prepared prompt message content layout changed")
    return value


def _canonical_message_text(content: object) -> tuple[str, ...]:
    if isinstance(content, str):
        return (content,)
    if not isinstance(content, Sequence) or isinstance(content, str | bytes):
        raise ValueError("canonical message content must be text or a sequence")
    texts: list[str] = []
    for part in content:
        if not isinstance(part, Mapping) or part.get("type") != "input_text":
            raise ValueError("canonical message content is not text-only")
        text = part.get("text")
        if not isinstance(text, str):
            raise ValueError("canonical input_text must carry text")
        texts.append(text)
    return tuple(texts)


def _expected_message_item_kinds(
    text_count: int,
    media_slots: set[int],
) -> tuple[str, ...]:
    content_count = text_count + len(media_slots)
    if any(slot >= content_count for slot in media_slots):
        raise ValueError("canonical media content index is outside its message")
    return tuple(
        "media" if content_index in media_slots else "text"
        for content_index in range(content_count)
    )


def _sealed_message_layout(
    items: tuple[PreparedTextItem | PreparedMediaItem, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    kinds: list[str] = []
    texts: list[str] = []
    previous_media_part: int | None = None
    for item in items:
        if isinstance(item, PreparedTextItem):
            kinds.append("text")
            texts.append(item.text)
            previous_media_part = None
        elif isinstance(item, PreparedMediaItem):
            if item.part_index != previous_media_part:
                kinds.append("media")
                previous_media_part = item.part_index
        else:
            raise TypeError("prepared message contains an unsupported item")
    return tuple(kinds), tuple(texts)


def _render_prepared_messages(
    prompt: PreparedQwen4Prompt,
) -> tuple[list[dict[str, object]], int]:
    rendered: list[dict[str, object]] = []
    image_count = 0
    for message in prompt.messages:
        content: list[str] = []
        for item in message.items:
            if isinstance(item, PreparedTextItem):
                content.append(item.text)
                continue
            if not isinstance(item, PreparedMediaItem):
                raise TypeError("prepared message contains an unsupported item")
            resolved = item.resolution
            if isinstance(resolved, ResolvedText):
                content.append(resolved.text)
            elif isinstance(resolved, ResolvedImage):
                content.append(_VISION_IMAGE_PLACEHOLDER)
                image_count += 1
            else:
                raise TypeError("prepared media contains an unsupported resolution")
        rendered_message: dict[str, object] = {
            "role": message.role,
            "content": "".join(content),
        }
        if message.item_type is not None:
            rendered_message["type"] = message.item_type
        if message.item_type == "function_call_output":
            rendered_message["call_id"] = message.call_id
            rendered_message["output"] = message.output
        rendered.append(rendered_message)
    return rendered, image_count


def _chat_template_instruction_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, Sequence) or isinstance(content, str | bytes):
        raise ValueError("Qwen4Exp instruction content must be text")
    parts: list[str] = []
    for item in content:
        if not isinstance(item, Mapping):
            raise ValueError("Qwen4Exp instruction parts must be mappings")
        text = item.get("text")
        if not isinstance(text, str):
            raise ValueError("Qwen4Exp instruction messages cannot contain media")
        parts.append(text)
    return "".join(parts)


def _chat_template_messages(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, object]]:
    """Project canonical Responses roles onto the Qwen checkpoint template."""

    instructions: list[str] = []
    rendered: list[dict[str, object]] = []
    conversation_started = False
    for message in messages:
        role = message.get("role")
        if role in {"system", "developer"}:
            if conversation_started:
                raise ValueError(
                    "Qwen4Exp system/developer messages must precede conversation"
                )
            instructions.append(_chat_template_instruction_text(message.get("content")))
            continue
        conversation_started = True
        rendered.append(dict(message))
    if instructions:
        rendered.insert(
            0,
            {
                "role": "system",
                "content": "\n\n".join(part for part in instructions if part),
            },
        )
    return rendered


def _expand_image_pad_tokens(
    token_ids: tuple[int, ...],
    *,
    image_pad_token_id: int,
    vision_start_token_id: int,
    vision_end_token_id: int,
    pad_rows: tuple[int, ...],
) -> tuple[int, ...]:
    pad_offsets = tuple(
        index for index, token in enumerate(token_ids) if token == image_pad_token_id
    )
    if len(pad_offsets) != len(pad_rows):
        raise VisionContractError(
            "unsealed_image_placeholders",
            "tokenized image placeholders do not match sealed image count",
        )
    expanded: list[int] = []
    image_index = 0
    for index, token in enumerate(token_ids):
        if token != image_pad_token_id:
            expanded.append(token)
            continue
        if (
            index == 0
            or index + 1 >= len(token_ids)
            or token_ids[index - 1] != vision_start_token_id
            or token_ids[index + 1] != vision_end_token_id
        ):
            raise VisionContractError(
                "invalid_image_placeholder",
                "image_pad must be enclosed by vision_start and vision_end",
            )
        rows = pad_rows[image_index]
        if rows < 1:
            raise VisionContractError(
                "invalid_image_pad_rows",
                "every sealed image requires at least one embedding row",
            )
        expanded.extend((image_pad_token_id,) * rows)
        image_index += 1
    return tuple(expanded)


def _tokenizer_marker_id(
    tokenizer: Any,
    attribute: str,
    marker: str,
) -> int | None:
    value = getattr(tokenizer, attribute, None)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    converter = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(converter):
        return None
    value = converter(marker)
    unknown = getattr(tokenizer, "unk_token_id", None)
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
        or value == unknown
    ):
        return None
    return value


def _tokenizer_stop_token_ids(
    tokenizer: Any,
    checkpoint_ids: Sequence[int],
) -> frozenset[int]:
    values = {int(token) for token in checkpoint_ids}
    tokenizer_ids = getattr(tokenizer, "eos_token_id", None)
    if isinstance(tokenizer_ids, int) and not isinstance(tokenizer_ids, bool):
        values.add(tokenizer_ids)
    elif isinstance(tokenizer_ids, Sequence) and not isinstance(
        tokenizer_ids,
        str | bytes,
    ):
        values.update(
            int(token)
            for token in tokenizer_ids
            if isinstance(token, int) and not isinstance(token, bool)
        )
    return frozenset(token for token in values if token >= 0)


def _prompt_has_open_thinking(
    token_ids: Sequence[int],
    think_start_id: int | None,
    think_end_id: int | None,
) -> bool:
    if think_start_id is None:
        return False
    last_start = max(
        (index for index, token in enumerate(token_ids) if token == think_start_id),
        default=-1,
    )
    if last_start < 0:
        return False
    last_end = (
        max(
            (index for index, token in enumerate(token_ids) if token == think_end_id),
            default=-1,
        )
        if think_end_id is not None
        else -1
    )
    return last_start > last_end


class _Qwen4ExpTensorRuntime:
    """Identity-stable continuous batch over the target scheduler port."""

    def __init__(
        self,
        *,
        plan: Qwen4ExpModelLoadPlan,
        model: Model,
        tokenizer: Any,
        model_spec: ModelSpec,
        capabilities: Qwen4ExpTensorCapabilities,
        prefix_config: _TensorPrefixConfig,
    ) -> None:
        self.plan = plan
        self.model = model
        self.tokenizer = tokenizer
        self._model_spec = model_spec
        self.capabilities = capabilities
        self._stop_token_ids = _tokenizer_stop_token_ids(
            tokenizer,
            plan.config.text.eos_token_ids,
        )
        self._think_start_id = _tokenizer_marker_id(
            tokenizer,
            "think_start_id",
            "<think>",
        )
        self._think_end_id = _tokenizer_marker_id(
            tokenizer,
            "think_end_id",
            "</think>",
        )
        self._prefix_config = prefix_config
        self._prefix_store = Qwen4ExpWholeBoundaryPrefixStore(
            namespace_signature=_prefix_namespace_signature(
                plan,
                capabilities,
                prefix_config.block_size_tokens,
            ),
            block_size_tokens=prefix_config.block_size_tokens,
            max_hot_entries=prefix_config.max_hot_entries,
            max_hot_tokens=prefix_config.max_hot_tokens,
            max_active_leases=prefix_config.max_active_leases,
            ssd=None,
        )
        self._reservations: dict[str, _TensorReservation] = {}
        self._encoders: dict[str, Any] = {}
        self._closed = False
        self._prefill_rows = 0
        self._decode_rows = 0
        self._tensor_forward_telemetry = _TensorForwardTelemetry()
        self._vision_preprocessor: Qwen4ExpTensorPreprocessor | None = None
        self._vision_tower: Qwen4ExpVisionTensorTower | None = None
        if not plan.config.language_model_only:
            self._vision_preprocessor = Qwen4ExpTensorPreprocessor.from_load_plan(plan)
            self._vision_tower = Qwen4ExpVisionTensorTower.from_load_plan(plan)

    @property
    def model_spec(self) -> ModelSpec:
        return self._model_spec

    def reserve(self, request: PreparedGenerationRequest, lease_id: str) -> object:
        self._require_open()
        if not isinstance(request, PreparedGenerationRequest):
            raise TypeError("tensor reserve requires PreparedGenerationRequest")
        canonical = request.request
        if canonical.media and not isinstance(
            request.backend_payload,
            PreparedQwen4Prompt,
        ):
            raise ValueError("raw media requires a sealed PreparedQwen4Prompt")
        if canonical.response_id in self._reservations:
            raise ValueError("duplicate tensor reservation")
        input_embeddings = None
        mrope = None
        if request.modality is RequestModality.TEXT:
            if canonical.media:
                raise ValueError("text modality cannot carry canonical media")
            prompt = self.tokenizer.apply_chat_template(
                _chat_template_messages(canonical.messages),
                tools=_chat_template_tools(canonical),
                tokenize=True,
                add_generation_prompt=True,
                **_chat_template_reasoning(canonical),
            )
            tokens = tuple(int(token) for token in prompt)
        elif request.modality is RequestModality.VISION:
            payload = _require_prepared_vision_prompt(
                canonical, request.backend_payload
            )
            tokens, input_embeddings, mrope = self._prepare_vision_prompt(
                canonical,
                payload,
            )
        else:
            raise ValueError(f"unsupported request modality: {request.modality!r}")
        if not tokens:
            raise ValueError("tokenized prompt must not be empty")
        sampling = _parse_tensor_sampling(
            canonical,
            context_length=self.plan.config.text.max_position_embeddings,
            prompt_tokens=len(tokens),
        )
        prefix_context = _prefix_context_fingerprint(request)
        cache = self.model.make_cache()
        mtp_cache = self.model.make_mtp_cache()
        prefix_lease = self._prefix_store.begin_request(
            canonical.response_id,
            context_fingerprint=prefix_context,
        ).lease
        reservation = _TensorReservation(
            request=canonical,
            prepared_request=request,
            lease_id=lease_id,
            cache=cache,
            mtp_cache=mtp_cache,
            prompt_tokens=tokens,
            max_output_tokens=sampling.max_output_tokens,
            sampler=sampling.target,
            draft_sampler=sampling.draft,
            rng=(np.random.default_rng(sampling.seed) if sampling.needs_rng else None),
            prefix_lease=prefix_lease,
            prefix_context_fingerprint=prefix_context,
            output_state=_TensorOutputState(
                stop_token_ids=self._stop_token_ids,
                think_start_id=self._think_start_id,
                think_end_id=self._think_end_id,
                in_reasoning=_prompt_has_open_thinking(
                    tokens,
                    self._think_start_id,
                    self._think_end_id,
                ),
            ),
            detokenizer=new_streaming_detokenizer(self.tokenizer),
            input_embeddings=input_embeddings,
            position_table=mrope.position_table if mrope is not None else None,
            mrope=mrope,
        )
        self._reservations[canonical.response_id] = reservation
        return reservation

    def _prepare_vision_prompt(
        self,
        request: GenerationRequest,
        prompt: PreparedQwen4Prompt,
    ) -> tuple[tuple[int, ...], Any, Qwen4ExpTensorMrope | None]:
        rendered, image_count = _render_prepared_messages(prompt)
        if image_count != len(prompt.resolution.images):
            raise VisionContractError(
                "prepared_image_count_mismatch",
                "sealed message images must match the resolved bundle",
            )
        tokenized = self.tokenizer.apply_chat_template(
            _chat_template_messages(rendered),
            tools=_chat_template_tools(request),
            tokenize=True,
            add_generation_prompt=True,
            **_chat_template_reasoning(request),
        )
        base_tokens = tuple(int(token) for token in tokenized)
        if not base_tokens:
            raise VisionContractError(
                "empty_prompt", "tokenized prompt must not be empty"
            )
        if image_count == 0:
            token_tensor = mx.array([base_tokens])
            embeddings = self.model.language_model.model.embed_tokens(token_tensor)
            return base_tokens, embeddings, None
        if self._vision_preprocessor is None or self._vision_tower is None:
            raise VisionContractError(
                "vision_runtime_unavailable",
                "language-model-only load plan cannot execute image input",
            )

        identity = VisionRequestIdentity(
            response_id=prompt.response_id,
            runtime=prompt.runtime,
            bundle_digest=prompt.resolution.digest,
        )
        processing_request = VisionProcessingRequest(
            identity=identity,
            bundle=prompt.resolution,
            spatial_merge_size=self.plan.config.vision.spatial_merge_size,
        )
        processed = self._vision_preprocessor.preprocess(processing_request)
        tower_output = self._vision_tower.forward(
            VisionTowerRequest(identity=identity, processed=processed)
        )
        # The frozen donor path validates deepstack receipts but does not inject them.
        _ignored_deepstack = tower_output.deepstack
        del _ignored_deepstack

        config = self.plan.config
        tokens = _expand_image_pad_tokens(
            base_tokens,
            image_pad_token_id=config.image_token_id,
            vision_start_token_id=config.vision_start_token_id,
            vision_end_token_id=config.vision_end_token_id,
            pad_rows=tuple(image.pad_rows for image in processed.images),
        )
        splice_plan = build_vision_splice_plan(
            processed,
            tower_output,
            prompt_token_ids=tokens,
            image_pad_token_id=config.image_token_id,
        )
        if splice_plan is None:
            raise VisionContractError(
                "invalid_splice_layout",
                "expanded prompt cannot be sealed to vision embedding rows",
            )
        token_tensor = mx.array([tokens])
        splicer = Qwen4ExpTensorSplicer(splice_plan)
        embeddings = splicer.splice_chunk(
            identity=identity,
            plan_digest=splice_plan.plan_digest,
            token_ids=tokens,
            token_tensor=token_tensor,
            embed_tokens=self.model.language_model.model.embed_tokens,
        )
        if embeddings is None:
            raise VisionContractError(
                "missing_splice_rows",
                "vision prompt did not consume its image embeddings",
            )
        splicer.assert_complete(
            identity=identity,
            plan_digest=splice_plan.plan_digest,
        )

        mrope_plan = build_mrope_plan(
            identity,
            tokens,
            image_token_id=config.image_token_id,
            image_grids=processed.images,
            spatial_merge_size=config.vision.spatial_merge_size,
            video_token_id=config.video_token_id,
        )
        if mrope_plan is None:
            raise VisionContractError(
                "invalid_mrope_layout",
                "expanded prompt cannot produce an exact M-RoPE table",
            )
        text = config.text
        mrope = Qwen4ExpTensorMrope(
            mrope_plan,
            mrope_section=text.mrope_section,
            mrope_interleaved=text.mrope_interleaved,
            rotary_dim=int(text.head_dim * text.partial_rotary_factor),
        )
        return tokens, embeddings, mrope

    def execute(
        self,
        plan: SchedulerPlan,
        reservations: Mapping[str, object],
        requests: Mapping[str, GenerationRequest],
        mtp_policy: MtpPolicy,
    ) -> FusedStepResult:
        self._require_open()
        prefill_results: list[PrefillResult] = []
        decode_results: list[DecodeResult] = []
        events: dict[str, tuple[Any, ...]] = {}
        prefill_started = time.perf_counter()
        for row in plan.prefill_rows:
            reservation = self._reservation(row.request_id, reservations, requests)
            self._prefill(reservation)
            prefill_results.append(
                PrefillResult(
                    request_id=row.request_id,
                    position=reservation.position,
                    complete=True,
                )
            )
            self._prefill_rows += 1
        prefill_elapsed = time.perf_counter() - prefill_started

        decode_started = time.perf_counter()
        decode_states = tuple(
            self._reservation(row.request_id, reservations, requests)
            for row in plan.decode_rows
        )
        mtp_decision = (
            self._mtp_decision(plan, decode_states, mtp_policy)
            if decode_states
            else None
        )
        decode_outcomes, fallbacks = (
            self._decode_batch(
                decode_states,
                mtp_decision=mtp_decision,
                draft_depth=mtp_policy.draft_depth,
            )
            if decode_states
            else ((), ())
        )

        ar_decode_tokens = 0
        mtp_rounds = 0
        mtp_drafted_tokens = 0
        mtp_accepted_tokens = 0
        mtp_rejected_tokens = 0
        for row, reservation, outcome in zip(
            plan.decode_rows,
            decode_states,
            decode_outcomes,
            strict=True,
        ):
            ar_decode_tokens += outcome.ar_decode_tokens
            mtp_rounds += outcome.mtp_rounds
            mtp_drafted_tokens += outcome.mtp_drafted_tokens
            mtp_accepted_tokens += outcome.mtp_accepted_tokens
            mtp_rejected_tokens += outcome.mtp_rejected_tokens
            encoder = self._encoders.get(row.request_id)
            if encoder is None:
                encoder = Qwen4TurnEventEncoderFactory().create(reservation.request)
                self._encoders[row.request_id] = encoder
            first_output_index = reservation.output_tokens - len(outcome.tokens) + 1
            row_events: list[Any] = []
            for index, token in enumerate(outcome.tokens):
                if reservation.output_state.is_protocol_token(token):
                    reservation.detokenizer.finalize()
                    pending_text = reservation.detokenizer.last_segment
                    if pending_text:
                        text_delta, reasoning_delta = reservation.output_state.route(
                            -1,
                            pending_text,
                        )
                        row_events.extend(
                            encoder.feed(
                                Qwen4OutputChunk(
                                    text_delta=text_delta,
                                    reasoning_delta=reasoning_delta,
                                    usage=_usage(
                                        reservation, first_output_index + index
                                    ),
                                )
                            )
                        )
                    reservation.detokenizer.reset()
                    decoded = ""
                else:
                    reservation.detokenizer.add_token(token)
                    decoded = reservation.detokenizer.last_segment
                text_delta, reasoning_delta = reservation.output_state.route(
                    token,
                    decoded,
                )
                chunk = Qwen4OutputChunk(
                    text_delta=text_delta,
                    reasoning_delta=reasoning_delta,
                    usage=_usage(reservation, first_output_index + index),
                )
                row_events.extend(encoder.feed(chunk))
            if outcome.finished:
                reservation.detokenizer.finalize()
                final_text = reservation.detokenizer.last_segment
                if final_text:
                    text_delta, reasoning_delta = reservation.output_state.route(
                        -1,
                        final_text,
                    )
                    row_events.extend(
                        encoder.feed(
                            Qwen4OutputChunk(
                                text_delta=text_delta,
                                reasoning_delta=reasoning_delta,
                                usage=_usage(reservation),
                            )
                        )
                    )
                row_events.extend(
                    encoder.finish(
                        _usage(reservation),
                        finish_reason=outcome.finish_reason or "stop",
                    )
                )
                self._encoders.pop(row.request_id, None)
            events[row.request_id] = tuple(row_events)
            decode_results.append(
                DecodeResult(
                    request_id=row.request_id,
                    position=reservation.position,
                    finished=outcome.finished,
                    finish_reason=outcome.finish_reason,
                )
            )
            self._decode_rows += 1
        decode_elapsed = time.perf_counter() - decode_started
        return FusedStepResult(
            prefill_results=tuple(prefill_results),
            decode_results=tuple(decode_results),
            events=events,
            prefill_elapsed_s=prefill_elapsed,
            decode_elapsed_s=decode_elapsed,
            ar_decode_steps=1 if ar_decode_tokens else 0,
            ar_decode_tokens=ar_decode_tokens,
            mtp_rounds=mtp_rounds,
            mtp_drafted_tokens=mtp_drafted_tokens,
            mtp_accepted_tokens=mtp_accepted_tokens,
            mtp_rejected_tokens=mtp_rejected_tokens,
            mtp_fallbacks=fallbacks,
        )

    def abort(self, reservation: object, reason: str) -> None:
        state = self._typed_reservation(reservation)
        if not reason:
            raise ValueError("abort reason must not be empty")
        state.aborted = True
        self._encoders.pop(state.request.response_id, None)

    def cleanup(
        self,
        reservation: object,
        reason: CacheReleaseReason,
    ) -> CacheCleanupReceipt:
        state = self._typed_reservation(reservation)
        request_id = state.request.response_id
        current = self._reservations.get(request_id)
        if current is not state:
            raise ValueError("foreign tensor reservation")
        prefix_published = False
        if reason is CacheReleaseReason.COMPLETED and state.pending_prefix is not None:
            commit = self._prefix_store.commit(
                state.prefix_lease,
                state.pending_prefix_tokens,
                state.pending_prefix,
                context_fingerprint=state.prefix_context_fingerprint,
            )
            prefix_published = commit.published or commit.already_published
        prefix_release = self._prefix_store.release_request(
            state.prefix_lease,
            context_fingerprint=state.prefix_context_fingerprint,
            reason=_prefix_release_reason(reason),
        )
        self._reservations.pop(request_id)
        self._encoders.pop(request_id, None)
        state.cache.clear()
        state.mtp_cache.clear()
        retained = bool(
            reason is CacheReleaseReason.COMPLETED
            and (prefix_published or state.prefix_hit)
        )
        return CacheCleanupReceipt(
            request_id=request_id,
            lease_id=state.lease_id,
            reason=reason,
            released_tiers=(CacheTier.PREFIX,),
            released_references=prefix_release.released_references,
            pending_writes_quiesced=True,
            retained_reusable_blocks=retained,
        )

    def stats(self) -> Mapping[str, Any]:
        prefix = self._prefix_store.stats()
        return {
            "qwen4_exp_plan_sha256": self.plan.plan_sha256,
            "tensor_batch_mode": self.plan.topology.tensor_batch_mode.value,
            "active_reservations": len(self._reservations),
            "prefill_rows": self._prefill_rows,
            "decode_rows": self._decode_rows,
            "tensor_forward": self._forward_telemetry().snapshot(),
            "native_capabilities": tuple(sorted(self.capabilities.enabled)),
            "prefix_cache_mode": "whole_boundary_hot",
            "paged_cache_enabled": False,
            "prefix_cache_ssd_enabled": False,
            "prefix_cache_block_size_tokens": self._prefix_config.block_size_tokens,
            "prefix_cache_hot_entries": prefix.hot_entries,
            "prefix_cache_hot_tokens": prefix.hot_tokens,
            "prefix_cache_hot_hits": prefix.hot_hits,
            "prefix_cache_ssd_hits": prefix.ssd_hits,
            "prefix_cache_misses": prefix.misses,
            "prefix_cache_commits": prefix.commits,
        }

    def _forward_telemetry(self) -> _TensorForwardTelemetry:
        telemetry = getattr(self, "_tensor_forward_telemetry", None)
        if telemetry is None:
            telemetry = _TensorForwardTelemetry()
            self._tensor_forward_telemetry = telemetry
        return telemetry

    def _record_tensor_forward(
        self,
        phase: str,
        *,
        batch_rows: int,
        sequence_length: int,
    ) -> None:
        self._forward_telemetry().record(
            phase,
            batch_rows=batch_rows,
            sequence_length=sequence_length,
        )

    def shutdown(self, deadline_s: float) -> None:
        if deadline_s < 0:
            raise ValueError("deadline_s must be non-negative")
        self._reservations.clear()
        self._encoders.clear()
        self._closed = True

    def _prefill(self, reservation: _TensorReservation) -> None:
        if reservation.aborted:
            raise RuntimeError("cannot prefill an aborted reservation")
        if reservation.position != 0:
            raise RuntimeError("tensor reservation was already prefilled")
        lookup = self._prefix_store.lookup(
            reservation.prefix_lease,
            reservation.prompt_tokens,
            context_fingerprint=reservation.prefix_context_fingerprint,
        )
        reservation.prefix_hit = lookup.hit
        if lookup.hit:
            self._restore_prefix_checkpoint(reservation, lookup)
            self._prefix_store.detach_lookup(
                reservation.prefix_lease,
                context_fingerprint=reservation.prefix_context_fingerprint,
            )

        boundary = (
            len(reservation.prompt_tokens)
            // self._prefix_config.block_size_tokens
            * self._prefix_config.block_size_tokens
        )
        if boundary > reservation.position:
            self._prefill_segment(reservation, reservation.position, boundary)
            self._stage_prefix_checkpoint(reservation, boundary)
        if reservation.position < len(reservation.prompt_tokens):
            self._prefill_segment(
                reservation,
                reservation.position,
                len(reservation.prompt_tokens),
            )
        if reservation.position != len(reservation.prompt_tokens):
            raise RuntimeError("prefill did not reach the sealed prompt frontier")

    def _prefill_segment(
        self,
        reservation: _TensorReservation,
        start: int,
        end: int,
    ) -> None:
        if not 0 <= start < end <= len(reservation.prompt_tokens):
            raise ValueError("prefill segment is outside the sealed prompt")
        if reservation.position != start:
            raise RuntimeError("prefill segment does not start at cache frontier")
        token_ids = reservation.prompt_tokens[start:end]
        tokens = mx.array([token_ids])
        input_embeddings = (
            reservation.input_embeddings[:, start:end, :]
            if reservation.input_embeddings is not None
            else None
        )
        previous_hidden = reservation.hidden
        with (
            self._vision_rope_scope(reservation),
            tensor_capability_scope(self.capabilities),
            attention_phase_scope("prefill"),
        ):
            logits, hidden = self.model(
                tokens,
                cache=reservation.cache,
                input_embeddings=input_embeddings,
                return_hidden=True,
                logits_keep=1,
            )
        mx.eval(logits, hidden)
        if start == 0:
            history_token_ids = token_ids[1:]
            history_hidden = hidden[:, :-1, :]
            history_embeddings = (
                input_embeddings[:, 1:, :] if input_embeddings is not None else None
            )
        else:
            if previous_hidden is None:
                raise RuntimeError("restored prefix is missing its hidden frontier")
            history_token_ids = token_ids
            history_hidden = mx.concatenate(
                (previous_hidden, hidden[:, :-1, :]),
                axis=1,
            )
            history_embeddings = input_embeddings
        if history_token_ids:
            history_tokens = mx.array([history_token_ids])
            with (
                self._vision_rope_scope(reservation, history_shift=1),
                attention_phase_scope("prefill"),
            ):
                mtp_hidden = self.model.mtp_update_cache(
                    history_hidden,
                    history_tokens,
                    mtp_cache=reservation.mtp_cache,
                    input_embeddings=history_embeddings,
                )
            mx.eval(mtp_hidden)
            self._record_tensor_forward(
                "mtp_history_update",
                batch_rows=1,
                sequence_length=len(history_token_ids),
            )
        reservation.logits = logits
        reservation.hidden = hidden[:, -1:, :]
        reservation.position = end

    def _stage_prefix_checkpoint(
        self,
        reservation: _TensorReservation,
        boundary: int,
    ) -> None:
        if reservation.position != boundary:
            raise RuntimeError("checkpoint boundary does not match cache frontier")
        if reservation.logits is None or reservation.hidden is None:
            raise RuntimeError("checkpoint boundary is missing model frontier state")
        target_state = self.model.snapshot(reservation.cache)
        mtp_state = self.model.snapshot(reservation.mtp_cache)
        logits = _snapshot_value(reservation.logits)
        hidden = _snapshot_value(reservation.hidden)
        _eval_tensor_values(target_state, mtp_state, logits, hidden)
        payload = _TensorPrefixPayload(
            target_cache_state=target_state,
            mtp_cache_state=mtp_state,
            logits=logits,
            hidden=hidden,
            position=boundary,
        )
        tokens = reservation.prompt_tokens[:boundary]
        reservation.pending_prefix = self._prefix_store.create_checkpoint(
            tokens,
            context_fingerprint=reservation.prefix_context_fingerprint,
            payload=payload,
        )
        reservation.pending_prefix_tokens = tokens

    def _restore_prefix_checkpoint(
        self,
        reservation: _TensorReservation,
        lookup: Qwen4ExpPrefixLookupReceipt,
    ) -> None:
        checkpoint = lookup.checkpoint
        if checkpoint is None or not lookup.hit:
            raise ValueError("prefix restore requires a cache hit")
        payload = checkpoint.payload
        if not isinstance(payload, _TensorPrefixPayload):
            raise TypeError("prefix checkpoint payload has a foreign tensor ABI")
        if payload.position != lookup.matched_tokens:
            raise ValueError("prefix checkpoint frontier does not match its identity")
        target_state = _restore_cache_bundle(
            reservation.cache,
            payload.target_cache_state,
        )
        mtp_state = _restore_cache_bundle(
            reservation.mtp_cache,
            payload.mtp_cache_state,
        )
        reservation.logits = _snapshot_value(payload.logits)
        reservation.hidden = _snapshot_value(payload.hidden)
        reservation.position = payload.position
        _eval_tensor_values(
            reservation.logits,
            reservation.hidden,
            target_state,
            mtp_state,
        )

    def _decode_batch(
        self,
        reservations: tuple[_TensorReservation, ...],
        *,
        mtp_decision: MtpDecision | None,
        draft_depth: int,
    ) -> tuple[tuple[_TensorDecodeOutcome, ...], tuple[MtpDisableReason, ...]]:
        """Decode one scheduler tick while preserving row-owned state.

        Plain-text AR rows share one ``[B, 1]`` target forward. Sampling and
        output commit remain host-side per-row operations, so stochastic RNG
        order, pending-primary semantics, cancellation compaction, and stream
        identity match the former row-serial oracle.

        An admitted aligned text cohort shares each recursive MTP draft depth
        and one equal-width target verify call. Request-local M-RoPE remains on
        the exact singleton lane because its three-axis position state cannot
        be flattened into the text batch representation.
        """

        if not reservations:
            return (), ()
        if len(reservations) == 1:
            reservation = reservations[0]
            if mtp_decision is not None and mtp_decision.enabled:
                return (
                    (
                        self._decode_mtp_one(
                            reservation,
                            draft_depth=draft_depth,
                        ),
                    ),
                    (),
                )
            outcome = self._decode_ar_one(reservation)
            fallback = self._mtp_fallback(mtp_decision)
            return (outcome,), fallback

        if any(reservation.position_table is not None for reservation in reservations):
            # M-RoPE tables are request-local three-axis state. Until a typed
            # batch representation exists, keep these rows on the exact
            # singleton oracle instead of flattening their semantics.
            if mtp_decision is not None and mtp_decision.enabled:
                return (
                    tuple(
                        self._decode_mtp_one(
                            reservation,
                            draft_depth=draft_depth,
                        )
                        for reservation in reservations
                    ),
                    (),
                )
            return (
                tuple(self._decode_ar_one(reservation) for reservation in reservations),
                self._mtp_fallback(mtp_decision),
            )

        if mtp_decision is not None and mtp_decision.enabled:
            return (
                self._decode_mtp_batch(
                    reservations,
                    draft_depth=draft_depth,
                ),
                (),
            )

        primaries: list[int] = []
        primary_is_new: list[bool] = []
        emitted: list[tuple[int, ...]] = []
        finished: list[bool] = []
        for reservation in reservations:
            primary, is_new = self._take_primary(reservation)
            row_emitted = (primary,) if is_new else ()
            if not is_new and self._pending_primary_is_terminal(reservation, primary):
                raise RuntimeError("terminal pending primary was scheduled for decode")
            row_finished = self._would_finish(reservation, row_emitted)
            primaries.append(primary)
            primary_is_new.append(is_new)
            emitted.append(row_emitted)
            finished.append(row_finished)
        batch_plan = MultirowBatchPlan[tuple[Any, Any]].compact(
            tuple(not row_finished for row_finished in finished)
        )
        forwarded = batch_plan.execute(
            lambda ordinals: self._target_forward_batch(
                tuple((primaries[index],) for index in ordinals),
                tuple(reservations[index] for index in ordinals),
            )
        )
        if batch_plan.active_ordinals:
            active_rows = tuple(
                reservations[index] for index in batch_plan.active_ordinals
            )
            self._mtp_update_batch(
                tuple(
                    reservations[index].hidden for index in batch_plan.active_ordinals
                ),
                tuple(primaries[index] for index in batch_plan.active_ordinals),
                active_rows,
            )
            for index in batch_plan.active_ordinals:
                row_forward = forwarded[index]
                if row_forward is None:
                    raise RuntimeError("active tensor row lost its forwarded state")
                logits, hidden = row_forward
                reservations[index].logits = logits[:, -1:, :]
                reservations[index].hidden = hidden[:, -1:, :]

        outcomes: list[_TensorDecodeOutcome] = []
        for index, reservation in enumerate(reservations):
            row_emitted = emitted[index]
            row_finished = finished[index]
            primary = primaries[index]
            finish_reason = (
                self._finish_reason(reservation, primary) if row_finished else None
            )
            self._commit_output(reservation, row_emitted)
            outcomes.append(
                _TensorDecodeOutcome(
                    tokens=row_emitted,
                    finished=row_finished,
                    finish_reason=finish_reason,
                    ar_decode_tokens=int(primary_is_new[index]),
                )
            )
        return tuple(outcomes), self._mtp_fallback(mtp_decision)

    def _decode_mtp_batch(
        self,
        reservations: tuple[_TensorReservation, ...],
        *,
        draft_depth: int,
    ) -> tuple[_TensorDecodeOutcome, ...]:
        """Run one exact recursive-MTP cycle over an aligned text cohort."""

        if len(reservations) < 2:
            raise ValueError("multi-row MTP requires at least two reservations")
        if any(reservation.position_table is not None for reservation in reservations):
            raise ValueError("multi-row MTP cannot flatten request-local M-RoPE")

        target_snapshots = tuple(
            self.model.snapshot(reservation.cache) for reservation in reservations
        )
        mtp_snapshots = tuple(
            self.model.snapshot(reservation.mtp_cache) for reservation in reservations
        )
        entry_pending = tuple(
            reservation.pending_primary for reservation in reservations
        )
        entry_positions = tuple(reservation.position for reservation in reservations)
        entry_output_counts = tuple(
            reservation.output_tokens for reservation in reservations
        )
        entry_logits = tuple(
            _snapshot_value(reservation.logits) for reservation in reservations
        )
        entry_hidden = tuple(
            _snapshot_value(reservation.hidden) for reservation in reservations
        )
        entry_rng_states = tuple(
            (
                None
                if reservation.rng is None
                else _snapshot_value(reservation.rng.bit_generator.state)
            )
            for reservation in reservations
        )

        staged_rngs: list[np.random.Generator | None] = []
        for rng_state in entry_rng_states:
            if rng_state is None:
                staged_rngs.append(None)
                continue
            staged_rng = np.random.default_rng()
            staged_rng.bit_generator.state = _snapshot_value(rng_state)
            staged_rngs.append(staged_rng)

        def clear_captures() -> None:
            for reservation in reservations:
                self.model.clear_verify_capture(reservation.cache)

        def restore_entry_state() -> None:
            restored_target = tuple(
                _restore_cache_bundle(reservation.cache, snapshot)
                for reservation, snapshot in zip(
                    reservations,
                    target_snapshots,
                    strict=True,
                )
            )
            restored_mtp = tuple(
                _restore_cache_bundle(reservation.mtp_cache, snapshot)
                for reservation, snapshot in zip(
                    reservations,
                    mtp_snapshots,
                    strict=True,
                )
            )
            for ordinal, reservation in enumerate(reservations):
                reservation.pending_primary = entry_pending[ordinal]
                reservation.position = entry_positions[ordinal]
                reservation.output_tokens = entry_output_counts[ordinal]
                reservation.logits = _snapshot_value(entry_logits[ordinal])
                reservation.hidden = _snapshot_value(entry_hidden[ordinal])
                rng_state = entry_rng_states[ordinal]
                if reservation.rng is not None and rng_state is not None:
                    reservation.rng.bit_generator.state = _snapshot_value(rng_state)
            _eval_tensor_values(
                restored_target,
                restored_mtp,
                entry_logits,
                entry_hidden,
            )

        def finish_reason(
            reservation: _TensorReservation,
            emitted_tokens: tuple[int, ...],
            output_count: int,
        ) -> str | None:
            if not emitted_tokens:
                return None
            if emitted_tokens[-1] in self._stop_token_ids:
                return "stop"
            if output_count >= reservation.max_output_tokens:
                return "length"
            return None

        try:
            primaries: list[int] = []
            primary_is_new: list[bool] = []
            emitted_rows: list[list[int]] = []
            terminal_rows: list[bool] = []
            active_ordinals: list[int] = []
            for ordinal, reservation in enumerate(reservations):
                pending = entry_pending[ordinal]
                if pending is not None:
                    primary = pending
                    is_new = False
                elif reservation.sampler.temperature <= 0:
                    primary = self._next_greedy_token(reservation)
                    is_new = True
                else:
                    self._require_decode_ready(reservation)
                    primary, _ = _sample_from_logits(
                        reservation.logits[0, -1],
                        reservation.sampler,
                        staged_rngs[ordinal],
                    )
                    is_new = True
                emitted = [primary] if is_new else []
                if not is_new and self._pending_primary_is_terminal(
                    reservation,
                    primary,
                ):
                    raise RuntimeError(
                        "terminal pending primary was scheduled for decode"
                    )
                terminal = self._would_finish(reservation, tuple(emitted))
                primaries.append(primary)
                primary_is_new.append(is_new)
                emitted_rows.append(emitted)
                terminal_rows.append(terminal)
                if not terminal:
                    active_ordinals.append(ordinal)

            staged_logits = list(entry_logits)
            staged_hidden = list(entry_hidden)
            staged_pending: list[int | None] = [None] * len(reservations)
            outcomes: list[_TensorDecodeOutcome | None] = [None] * len(reservations)
            for ordinal, terminal in enumerate(terminal_rows):
                if not terminal:
                    continue
                emitted_tokens = tuple(emitted_rows[ordinal])
                output_count = entry_output_counts[ordinal] + len(emitted_tokens)
                outcomes[ordinal] = _TensorDecodeOutcome(
                    tokens=emitted_tokens,
                    finished=True,
                    finish_reason=finish_reason(
                        reservations[ordinal],
                        emitted_tokens,
                        output_count,
                    ),
                    ar_decode_tokens=int(primary_is_new[ordinal]),
                )

            if active_ordinals:
                active_rows = tuple(reservations[index] for index in active_ordinals)
                active_rngs = tuple(staged_rngs[index] for index in active_ordinals)
                max_drafts = min(
                    draft_depth,
                    *(
                        reservation.max_output_tokens
                        - (entry_output_counts[ordinal] + len(emitted_rows[ordinal]))
                        for ordinal, reservation in zip(
                            active_ordinals,
                            active_rows,
                            strict=True,
                        )
                    ),
                )
                if max_drafts < 1:
                    raise RuntimeError(
                        "non-terminal multi-row MTP requires draft capacity"
                    )

                hidden_before_primary = tuple(
                    reservation.hidden for reservation in active_rows
                )
                draft_hidden = hidden_before_primary
                next_tokens = tuple(primaries[index] for index in active_ordinals)
                draft_tokens: list[list[int]] = [[] for _ in active_rows]
                draft_distributions: list[list[Distribution | None]] = [
                    [] for _ in active_rows
                ]
                for _ in range(max_drafts):
                    draft_forwards = self._mtp_forward_batch(
                        draft_hidden,
                        next_tokens,
                        active_rows,
                    )
                    next_hidden: list[Any] = []
                    sampled_tokens: list[int] = []
                    for row_index, (reservation, row_forward) in enumerate(
                        zip(active_rows, draft_forwards, strict=True)
                    ):
                        draft_logits, draft_hidden_next = row_forward
                        draft, draft_q = _sample_draft_from_logits(
                            draft_logits[0, -1],
                            reservation.draft_sampler,
                            active_rngs[row_index],
                            need_distribution=reservation.sampler.temperature > 0,
                        )
                        draft_tokens[row_index].append(draft)
                        draft_distributions[row_index].append(draft_q)
                        next_hidden.append(draft_hidden_next[:, -1:, :])
                        sampled_tokens.append(draft)
                    draft_hidden = tuple(next_hidden)
                    next_tokens = tuple(sampled_tokens)

                verify_token_rows = tuple(
                    (primaries[ordinal], *draft_tokens[row_index])
                    for row_index, ordinal in enumerate(active_ordinals)
                )
                capture = self.model.begin_capture(active_rows[0].cache)
                try:
                    verify_forwards = self._target_forward_batch(
                        verify_token_rows,
                        active_rows,
                        phase="verify",
                        telemetry_phase=None,
                    )
                finally:
                    self.model.end_capture(active_rows[0].cache, capture)

                accepted_counts: list[int] = []
                rejected_rows: list[bool] = []
                corrections: list[int | None] = []
                committed_token_rows: list[tuple[int, ...]] = []
                for row_index, reservation in enumerate(active_rows):
                    verify_logits = verify_forwards[row_index][0]
                    accepted_count = 0
                    rejected = False
                    correction: int | None = None
                    for index, (draft, draft_q) in enumerate(
                        zip(
                            draft_tokens[row_index],
                            draft_distributions[row_index],
                            strict=True,
                        )
                    ):
                        if reservation.sampler.temperature <= 0:
                            accepted = draft == int(
                                mx.argmax(verify_logits[0, index]).item()
                            )
                        else:
                            target_p = _distribution_from_mlx_logits(
                                verify_logits[0, index],
                                reservation.sampler,
                            )
                            if draft_q is None:
                                raise RuntimeError(
                                    "stochastic MTP requires a draft distribution"
                                )
                            rng = _require_sampling_rng(active_rngs[row_index])
                            accept_p = acceptance_probability(
                                target_p,
                                draft_q,
                                draft,
                            )
                            accepted = float(rng.random()) <= accept_p
                            if not accepted:
                                correction = sample_from_distribution(
                                    residual_distribution(target_p, draft_q),
                                    rng,
                                )
                        if not accepted:
                            rejected = True
                            break
                        accepted_count += 1
                        if draft in self._stop_token_ids:
                            break
                    accepted_counts.append(accepted_count)
                    rejected_rows.append(rejected)
                    corrections.append(correction)
                    committed_tokens = (
                        primaries[active_ordinals[row_index]],
                        *draft_tokens[row_index][:accepted_count],
                    )
                    if rejected and reservation.sampler.temperature > 0:
                        if correction is None:
                            raise RuntimeError(
                                "stochastic MTP rejection requires correction"
                            )
                        committed_tokens = (*committed_tokens, correction)
                    committed_token_rows.append(committed_tokens)

                committed_from_capture = True
                for row_index, reservation in enumerate(active_rows):
                    committed = self.model.commit_verified_window(
                        reservation.cache,
                        target_snapshots[active_ordinals[row_index]],
                        keep_tokens=1 + accepted_counts[row_index],
                        verified_tokens=len(verify_token_rows[row_index]),
                    )
                    if not committed:
                        committed_from_capture = False
                        break
                clear_captures()

                authoritative_logits: list[Any] = [None] * len(active_rows)
                authoritative_hidden: list[Any] = [None] * len(active_rows)
                history_hidden_rows: list[Any] = [None] * len(active_rows)
                if committed_from_capture:
                    self._record_tensor_forward(
                        "target_verify",
                        batch_rows=len(active_rows),
                        sequence_length=len(verify_token_rows[0]),
                    )
                    for row_index, (verify_logits, verify_hidden) in enumerate(
                        verify_forwards
                    ):
                        keep_tokens = 1 + accepted_counts[row_index]
                        authoritative_logits[row_index] = verify_logits[
                            :, keep_tokens - 1 : keep_tokens, :
                        ]
                        authoritative_hidden[row_index] = verify_hidden[
                            :, :keep_tokens, :
                        ]
                        history_hidden_rows[row_index] = verify_hidden[
                            :, :keep_tokens, :
                        ]

                    correction_indices = tuple(
                        index
                        for index, (reservation, rejected) in enumerate(
                            zip(active_rows, rejected_rows, strict=True)
                        )
                        if rejected and reservation.sampler.temperature > 0
                    )
                    if correction_indices:
                        correction_forwards = self._target_forward_batch(
                            tuple(
                                (committed_token_rows[index][-1],)
                                for index in correction_indices
                            ),
                            tuple(active_rows[index] for index in correction_indices),
                            telemetry_phase="target_correction",
                        )
                        if len(correction_forwards) != len(correction_indices):
                            raise RuntimeError("stochastic correction batch lost a row")
                        for row_index, row_forward in zip(
                            correction_indices,
                            correction_forwards,
                            strict=True,
                        ):
                            authoritative_logits[row_index] = row_forward[0]
                            authoritative_hidden[row_index] = row_forward[1]
                else:
                    restored_target = tuple(
                        _restore_cache_bundle(reservation.cache, snapshot)
                        for reservation, snapshot in zip(
                            reservations,
                            target_snapshots,
                            strict=True,
                        )
                    )
                    _eval_tensor_values(restored_target)
                    width_groups: dict[int, list[int]] = {}
                    for row_index, committed_tokens in enumerate(committed_token_rows):
                        width_groups.setdefault(len(committed_tokens), []).append(
                            row_index
                        )
                    for group_indices in width_groups.values():
                        if len(group_indices) >= 2:
                            group_forwards = self._target_forward_batch(
                                tuple(
                                    committed_token_rows[index]
                                    for index in group_indices
                                ),
                                tuple(active_rows[index] for index in group_indices),
                            )
                        else:
                            row_index = group_indices[0]
                            group_forwards = (
                                self._target_forward(
                                    committed_token_rows[row_index],
                                    active_rows[row_index],
                                ),
                            )
                        for row_index, row_forward in zip(
                            group_indices,
                            group_forwards,
                            strict=True,
                        ):
                            authoritative_logits[row_index] = row_forward[0]
                            authoritative_hidden[row_index] = row_forward[1]
                            history_hidden_rows[row_index] = row_forward[1]

                restored_mtp = tuple(
                    _restore_cache_bundle(reservation.mtp_cache, snapshot)
                    for reservation, snapshot in zip(
                        reservations,
                        mtp_snapshots,
                        strict=True,
                    )
                )
                _eval_tensor_values(restored_mtp)

                history_rows: list[list[tuple[Any, int]]] = []
                for row_index, ordinal in enumerate(active_ordinals):
                    history_hidden = history_hidden_rows[row_index]
                    accepted = accepted_counts[row_index]
                    history: list[tuple[Any, int]] = [
                        (hidden_before_primary[row_index], primaries[ordinal])
                    ]
                    history.extend(
                        (history_hidden[:, index : index + 1, :], draft)
                        for index, draft in enumerate(
                            draft_tokens[row_index][:accepted]
                        )
                    )
                    correction = corrections[row_index]
                    if correction is not None:
                        history.append(
                            (
                                history_hidden[:, accepted : accepted + 1, :],
                                correction,
                            )
                        )
                    history_rows.append(history)

                for history_depth in range(max(map(len, history_rows))):
                    history_indices = tuple(
                        index
                        for index, history in enumerate(history_rows)
                        if history_depth < len(history)
                    )
                    self._mtp_update_batch(
                        tuple(
                            history_rows[index][history_depth][0]
                            for index in history_indices
                        ),
                        tuple(
                            history_rows[index][history_depth][1]
                            for index in history_indices
                        ),
                        tuple(active_rows[index] for index in history_indices),
                    )

                for row_index, ordinal in enumerate(active_ordinals):
                    reservation = active_rows[row_index]
                    staged_logits[ordinal] = authoritative_logits[row_index][:, -1:, :]
                    staged_hidden[ordinal] = authoritative_hidden[row_index][:, -1:, :]
                    accepted = accepted_counts[row_index]
                    emitted = emitted_rows[ordinal]
                    emitted.extend(draft_tokens[row_index][:accepted])
                    correction = corrections[row_index]
                    if correction is not None:
                        emitted.append(correction)
                    elif accepted == len(
                        draft_tokens[row_index]
                    ) and not self._would_finish(reservation, tuple(emitted)):
                        bonus, _ = _sample_from_logits(
                            staged_logits[ordinal][0, -1],
                            reservation.sampler,
                            active_rngs[row_index],
                        )
                        emitted.append(bonus)
                        staged_pending[ordinal] = bonus

                    emitted_tokens = tuple(emitted)
                    finished = self._would_finish(reservation, emitted_tokens)
                    output_count = entry_output_counts[ordinal] + len(emitted_tokens)
                    outcomes[ordinal] = _TensorDecodeOutcome(
                        tokens=emitted_tokens,
                        finished=finished,
                        finish_reason=(
                            finish_reason(
                                reservation,
                                emitted_tokens,
                                output_count,
                            )
                            if finished
                            else None
                        ),
                        mtp_rounds=1,
                        mtp_drafted_tokens=len(draft_tokens[row_index]),
                        mtp_accepted_tokens=accepted,
                        mtp_rejected_tokens=int(rejected_rows[row_index]),
                    )

            if any(outcome is None for outcome in outcomes):
                raise RuntimeError("multi-row MTP lost a scheduler row")

            for ordinal, reservation in enumerate(reservations):
                reservation.pending_primary = staged_pending[ordinal]
                reservation.position = entry_positions[ordinal] + len(
                    emitted_rows[ordinal]
                )
                reservation.output_tokens = entry_output_counts[ordinal] + len(
                    emitted_rows[ordinal]
                )
                reservation.logits = staged_logits[ordinal]
                reservation.hidden = staged_hidden[ordinal]
                staged_rng = staged_rngs[ordinal]
                if reservation.rng is not None and staged_rng is not None:
                    reservation.rng.bit_generator.state = _snapshot_value(
                        staged_rng.bit_generator.state
                    )
            _eval_tensor_values(staged_logits, staged_hidden)
            return tuple(outcome for outcome in outcomes if outcome is not None)
        except Exception:
            restore_entry_state()
            raise
        finally:
            clear_captures()

    @staticmethod
    def _mtp_fallback(
        decision: MtpDecision | None,
    ) -> tuple[MtpDisableReason, ...]:
        if decision is None:
            return ()
        reason = decision.disable_reason
        if reason is None:
            return ()
        return (reason,)

    def _target_forward_batch(
        self,
        token_rows: tuple[tuple[int, ...], ...],
        reservations: tuple[_TensorReservation, ...],
        *,
        phase: str = "decode",
        telemetry_phase: str | None = "target_decode",
    ) -> tuple[tuple[Any, Any], ...]:
        """Issue exactly one target/QSA forward for compatible active rows."""

        if not token_rows or len(token_rows) != len(reservations):
            raise ValueError("target batch requires one token row per reservation")
        widths = {len(tokens) for tokens in token_rows}
        if len(widths) != 1 or not next(iter(widths)):
            raise ValueError("target batch token rows must share a positive width")
        if any(reservation.position_table is not None for reservation in reservations):
            raise ValueError("target tensor batch cannot flatten request-local M-RoPE")

        token_array = mx.stack(tuple(mx.array(tokens) for tokens in token_rows))
        cache_rows = tuple(reservation.cache for reservation in reservations)
        batch_cache = _merge_batch_cache(cache_rows)
        with (
            tensor_capability_scope(self.capabilities),
            attention_phase_scope(phase),
        ):
            logits, hidden = self.model(
                token_array,
                cache=batch_cache,
                return_hidden=True,
                logits_keep=0,
            )
        mx.eval(logits, hidden)
        _scatter_batch_cache(batch_cache, cache_rows)
        if telemetry_phase is not None:
            self._record_tensor_forward(
                telemetry_phase,
                batch_rows=len(reservations),
                sequence_length=len(token_rows[0]),
            )
        return tuple(
            (logits[index : index + 1], hidden[index : index + 1])
            for index in range(len(reservations))
        )

    def _mtp_forward_batch(
        self,
        hidden_rows: tuple[Any, ...],
        token_ids: tuple[int, ...],
        reservations: tuple[_TensorReservation, ...],
    ) -> tuple[tuple[Any, Any], ...]:
        """Issue one recursive draft-head call for an aligned cohort depth."""

        if not hidden_rows or not (
            len(hidden_rows) == len(token_ids) == len(reservations)
        ):
            raise ValueError("MTP draft inputs must have the same positive row count")
        hidden = mx.concatenate(hidden_rows, axis=0)
        tokens = mx.stack(tuple(mx.array([token]) for token in token_ids))
        cache_rows = tuple(reservation.mtp_cache for reservation in reservations)
        batch_cache = _merge_batch_cache(cache_rows)
        with (
            tensor_capability_scope(self.capabilities),
            attention_phase_scope("decode"),
        ):
            logits, next_hidden = self.model.mtp_forward(
                hidden,
                tokens,
                mtp_cache=batch_cache,
                return_hidden=True,
            )
        mx.eval(logits, next_hidden)
        _scatter_batch_cache(batch_cache, cache_rows)
        self._record_tensor_forward(
            "mtp_draft",
            batch_rows=len(reservations),
            sequence_length=1,
        )
        return tuple(
            (
                logits[index : index + 1],
                next_hidden[index : index + 1],
            )
            for index in range(len(reservations))
        )

    def _mtp_update_batch(
        self,
        hidden_rows: tuple[Any, ...],
        token_ids: tuple[int, ...],
        reservations: tuple[_TensorReservation, ...],
    ) -> None:
        """Advance every row's draft history in one independent-state call."""

        if not hidden_rows or not (
            len(hidden_rows) == len(token_ids) == len(reservations)
        ):
            raise ValueError("MTP batch inputs must have the same positive row count")
        hidden = mx.concatenate(hidden_rows, axis=0)
        tokens = mx.stack(tuple(mx.array([token]) for token in token_ids))
        cache_rows = tuple(reservation.mtp_cache for reservation in reservations)
        batch_cache = _merge_batch_cache(cache_rows)
        with (
            tensor_capability_scope(self.capabilities),
            attention_phase_scope("decode"),
        ):
            mtp_hidden = self.model.mtp_update_cache(
                hidden,
                tokens,
                mtp_cache=batch_cache,
            )
        mx.eval(mtp_hidden)
        _scatter_batch_cache(batch_cache, cache_rows)
        self._record_tensor_forward(
            "mtp_history_update",
            batch_rows=len(reservations),
            sequence_length=1,
        )

    def _decode_ar_one(self, reservation: _TensorReservation) -> _TensorDecodeOutcome:
        primary, primary_is_new = self._take_primary(reservation)
        emitted_tokens = (primary,) if primary_is_new else ()
        if not primary_is_new and self._pending_primary_is_terminal(
            reservation, primary
        ):
            raise RuntimeError("terminal pending primary was scheduled for decode")
        finished = self._would_finish(reservation, emitted_tokens)
        if not finished:
            pre_forward_hidden = reservation.hidden
            logits, hidden = self._target_forward((primary,), reservation)
            with self._vision_rope_scope(reservation, history_shift=1):
                mtp_hidden = self.model.mtp_update_cache(
                    pre_forward_hidden,
                    mx.array([[primary]]),
                    mtp_cache=reservation.mtp_cache,
                )
            mx.eval(mtp_hidden)
            self._record_tensor_forward(
                "mtp_history_update",
                batch_rows=1,
                sequence_length=1,
            )
            reservation.logits = logits[:, -1:, :]
            reservation.hidden = hidden[:, -1:, :]
        self._commit_output(reservation, emitted_tokens)
        return _TensorDecodeOutcome(
            tokens=emitted_tokens,
            finished=finished,
            finish_reason=self._finish_reason(reservation, primary),
            ar_decode_tokens=int(primary_is_new),
        )

    def _decode_mtp_one(
        self,
        reservation: _TensorReservation,
        *,
        draft_depth: int,
    ) -> _TensorDecodeOutcome:
        """Run one exact singleton recursive-MTP cycle."""

        primary, primary_is_new = self._take_primary(reservation)
        emitted: list[int] = [primary] if primary_is_new else []
        if not primary_is_new and self._pending_primary_is_terminal(
            reservation, primary
        ):
            raise RuntimeError("terminal pending primary was scheduled for decode")
        if self._would_finish(reservation, tuple(emitted)):
            emitted_tokens = tuple(emitted)
            self._commit_output(reservation, emitted_tokens)
            return _TensorDecodeOutcome(
                tokens=emitted_tokens,
                finished=True,
                finish_reason=self._finish_reason(reservation, primary),
                ar_decode_tokens=int(primary_is_new),
            )

        remaining = reservation.max_output_tokens - (
            reservation.output_tokens + len(emitted)
        )
        max_drafts = min(draft_depth, remaining)
        if max_drafts < 1:
            raise RuntimeError("non-terminal MTP cycle requires draft capacity")

        hidden_before_primary = reservation.hidden
        mtp_snapshot = self.model.snapshot(reservation.mtp_cache)
        draft_hidden = hidden_before_primary
        next_token = primary
        draft_tokens: list[int] = []
        draft_distributions: list[Distribution | None] = []
        for _ in range(max_drafts):
            with self._vision_rope_scope(reservation, history_shift=1):
                draft_logits, draft_hidden_next = self.model.mtp_forward(
                    draft_hidden,
                    mx.array([[next_token]]),
                    mtp_cache=reservation.mtp_cache,
                    return_hidden=True,
                )
            mx.eval(draft_logits, draft_hidden_next)
            self._record_tensor_forward(
                "mtp_draft",
                batch_rows=1,
                sequence_length=1,
            )
            draft, draft_q = _sample_draft_from_logits(
                draft_logits[0, -1],
                reservation.draft_sampler,
                reservation.rng,
                need_distribution=reservation.sampler.temperature > 0,
            )
            draft_tokens.append(draft)
            draft_distributions.append(draft_q)
            draft_hidden = draft_hidden_next[:, -1:, :]
            next_token = draft

        verify_tokens = (primary, *draft_tokens)
        snapshot = self.model.snapshot(reservation.cache)
        capture = self.model.begin_capture(reservation.cache)
        try:
            verify_logits, verify_hidden = self._target_forward(
                verify_tokens,
                reservation,
                phase="verify",
                telemetry_phase=None,
            )
        finally:
            self.model.end_capture(reservation.cache, capture)

        accepted_count = 0
        rejected = False
        correction: int | None = None
        for index, (draft, draft_q) in enumerate(
            zip(draft_tokens, draft_distributions, strict=True)
        ):
            if reservation.sampler.temperature <= 0:
                accepted = draft == int(mx.argmax(verify_logits[0, index]).item())
            else:
                target_p = _distribution_from_mlx_logits(
                    verify_logits[0, index], reservation.sampler
                )
                if draft_q is None:
                    raise RuntimeError("stochastic MTP requires a draft distribution")
                rng = _require_sampling_rng(reservation.rng)
                accept_p = acceptance_probability(target_p, draft_q, draft)
                accepted = float(rng.random()) <= accept_p
                if not accepted:
                    correction = sample_from_distribution(
                        residual_distribution(target_p, draft_q), rng
                    )
            if not accepted:
                rejected = True
                break
            accepted_count += 1
            if draft in self._stop_token_ids:
                break

        accepted_drafts = tuple(draft_tokens[:accepted_count])
        stochastic_rejection = rejected and reservation.sampler.temperature > 0
        keep_tokens = 1 + accepted_count
        try:
            with self._vision_rope_scope(reservation):
                committed_from_capture = self.model.commit_verified_window(
                    reservation.cache,
                    snapshot,
                    keep_tokens=keep_tokens,
                    verified_tokens=len(verify_tokens),
                )
        finally:
            self.model.clear_verify_capture(reservation.cache)

        if committed_from_capture:
            self._record_tensor_forward(
                "target_verify",
                batch_rows=1,
                sequence_length=len(verify_tokens),
            )

        committed_tokens = (primary, *accepted_drafts)
        if stochastic_rejection:
            if correction is None:
                raise RuntimeError("stochastic MTP rejection requires correction")
            committed_tokens = (*committed_tokens, correction)

        if committed_from_capture and stochastic_rejection:
            authoritative_logits, authoritative_hidden = self._target_forward(
                (correction,),
                reservation,
                telemetry_phase="target_correction",
            )
        elif committed_from_capture:
            authoritative_logits = verify_logits[:, keep_tokens - 1 : keep_tokens, :]
            authoritative_hidden = verify_hidden[:, :keep_tokens, :]
        else:
            self.model.rollback_after_verify(
                reservation.cache,
                snapshot,
                verified_tokens=len(verify_tokens),
            )
            authoritative_logits, authoritative_hidden = self._target_forward(
                committed_tokens,
                reservation,
            )

        mtp_state = _restore_cache_bundle(reservation.mtp_cache, mtp_snapshot)
        _eval_tensor_values(*mtp_state)
        history_hidden = (
            verify_hidden if committed_from_capture else authoritative_hidden
        )
        mtp_history: list[tuple[Any, int]] = [(hidden_before_primary, primary)]
        mtp_history.extend(
            (history_hidden[:, index : index + 1, :], draft)
            for index, draft in enumerate(accepted_drafts)
        )
        if stochastic_rejection:
            mtp_history.append(
                (
                    history_hidden[:, accepted_count : accepted_count + 1, :],
                    correction,
                )
            )
        for hidden_before_token, token in mtp_history:
            with self._vision_rope_scope(reservation, history_shift=1):
                mtp_hidden = self.model.mtp_update_cache(
                    hidden_before_token,
                    mx.array([[token]]),
                    mtp_cache=reservation.mtp_cache,
                )
            mx.eval(mtp_hidden)
            self._record_tensor_forward(
                "mtp_history_update",
                batch_rows=1,
                sequence_length=1,
            )

        reservation.logits = authoritative_logits[:, -1:, :]
        reservation.hidden = authoritative_hidden[:, -1:, :]
        emitted.extend(accepted_drafts)
        if stochastic_rejection:
            emitted.append(correction)
        elif accepted_count == len(draft_tokens) and not self._would_finish(
            reservation, tuple(emitted)
        ):
            bonus, _ = _sample_from_logits(
                reservation.logits[0, -1],
                reservation.sampler,
                reservation.rng,
            )
            emitted.append(bonus)
            reservation.pending_primary = bonus

        emitted_tokens = tuple(emitted)
        finished = self._would_finish(reservation, emitted_tokens)
        self._commit_output(reservation, emitted_tokens)
        finish_reason = (
            self._finish_reason(reservation, emitted_tokens[-1])
            if emitted_tokens
            else None
        )
        return _TensorDecodeOutcome(
            tokens=emitted_tokens,
            finished=finished,
            finish_reason=finish_reason,
            mtp_rounds=1,
            mtp_drafted_tokens=len(draft_tokens),
            mtp_accepted_tokens=accepted_count,
            mtp_rejected_tokens=int(rejected),
        )

    def _take_primary(
        self,
        reservation: _TensorReservation,
    ) -> tuple[int, bool]:
        if reservation.pending_primary is not None:
            primary = reservation.pending_primary
            reservation.pending_primary = None
            return primary, False
        return self._next_target_token(reservation), True

    def _next_target_token(self, reservation: _TensorReservation) -> int:
        if reservation.sampler.temperature <= 0:
            return self._next_greedy_token(reservation)
        self._require_decode_ready(reservation)
        token, _ = _sample_from_logits(
            reservation.logits[0, -1],
            reservation.sampler,
            reservation.rng,
        )
        return token

    def _next_greedy_token(self, reservation: _TensorReservation) -> int:
        self._require_decode_ready(reservation)
        return int(mx.argmax(reservation.logits[0, -1]).item())

    @staticmethod
    def _require_decode_ready(reservation: _TensorReservation) -> None:
        if reservation.aborted:
            raise RuntimeError("cannot decode an aborted reservation")
        if reservation.logits is None:
            raise RuntimeError("decode requires completed prefill")
        if reservation.output_tokens >= reservation.max_output_tokens:
            raise RuntimeError("decode scheduled after max_output_tokens")

    def _pending_primary_is_terminal(
        self,
        reservation: _TensorReservation,
        primary: int,
    ) -> bool:
        return (
            primary in self._stop_token_ids
            or reservation.output_tokens >= reservation.max_output_tokens
        )

    def _target_forward(
        self,
        token_ids: tuple[int, ...],
        reservation: _TensorReservation,
        *,
        phase: str = "decode",
        telemetry_phase: str | None = "target_decode",
    ) -> tuple[Any, Any]:
        token_array = mx.array([token_ids])
        with (
            self._vision_rope_scope(reservation),
            tensor_capability_scope(self.capabilities),
            attention_phase_scope(phase),
        ):
            logits, hidden = self.model(
                token_array,
                cache=reservation.cache,
                return_hidden=True,
                logits_keep=0,
            )
        mx.eval(logits, hidden)
        if telemetry_phase is not None:
            self._record_tensor_forward(
                telemetry_phase,
                batch_rows=1,
                sequence_length=len(token_ids),
            )
        return logits, hidden

    @staticmethod
    def _vision_rope_scope(
        reservation: _TensorReservation,
        *,
        history_shift: int = 0,
    ) -> contextlib.AbstractContextManager[None]:
        if reservation.position_table is None or reservation.mrope is None:
            return contextlib.nullcontext()
        table = reservation.position_table
        if history_shift:
            table = table[:, history_shift:]
        return vision_rope_scope(
            table,
            reservation.mrope.rope_delta + history_shift,
        )

    def _would_finish(
        self,
        reservation: _TensorReservation,
        token_ids: tuple[int, ...],
    ) -> bool:
        return (
            any(token in self._stop_token_ids for token in token_ids)
            or reservation.output_tokens + len(token_ids)
            >= reservation.max_output_tokens
        )

    def _finish_reason(
        self,
        reservation: _TensorReservation,
        last_token: int,
    ) -> str | None:
        if last_token in self._stop_token_ids:
            return "stop"
        if reservation.output_tokens >= reservation.max_output_tokens:
            return "length"
        return None

    @staticmethod
    def _commit_output(
        reservation: _TensorReservation,
        token_ids: tuple[int, ...],
    ) -> None:
        reservation.output_tokens += len(token_ids)
        reservation.position += len(token_ids)

    def _mtp_decision(
        self,
        plan: SchedulerPlan,
        reservations: tuple[_TensorReservation, ...],
        mtp_policy: MtpPolicy,
    ) -> MtpDecision:
        alignment = MtpAlignment(
            runtime_keys=tuple(
                _runtime_identity(reservation.request.runtime)
                for reservation in reservations
            ),
            cache_positions=tuple(
                reservation.position - int(reservation.pending_primary is not None)
                for reservation in reservations
            ),
            pending_prompt_work=bool(plan.prefill_rows),
        )
        return mtp_policy.decide(
            alignment=alignment,
            model_supported=(
                self.plan.config.text.mtp.num_hidden_layers > 0
                and self.plan.topology.max_verified_mtp_rows >= 1
            ),
            head_attached=self.plan.artifacts.has_embedded_mtp,
            decode_enabled=True,
            verifier_available=all(
                self._has_exact_sampling_verifier(reservation)
                for reservation in reservations
            ),
            grammar_constrained=any(
                _is_grammar_constrained(reservation.request)
                for reservation in reservations
            ),
        )

    @staticmethod
    def _has_exact_sampling_verifier(reservation: _TensorReservation) -> bool:
        return (
            reservation.sampler.temperature <= 0 or reservation.rng is not None
        ) and (
            reservation.draft_sampler.temperature <= 0 or reservation.rng is not None
        )

    def _reservation(
        self,
        request_id: str,
        reservations: Mapping[str, object],
        requests: Mapping[str, GenerationRequest],
    ) -> _TensorReservation:
        state = self._typed_reservation(reservations[request_id])
        if state is not self._reservations.get(request_id):
            raise ValueError("reservation is not owned by this tensor runtime")
        if state.request is not requests[request_id]:
            raise ValueError("generation request changed after reservation")
        return state

    @staticmethod
    def _typed_reservation(value: object) -> _TensorReservation:
        if not isinstance(value, _TensorReservation):
            raise TypeError("foreign tensor reservation")
        return value

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Qwen4Exp tensor runtime is closed")


def _usage(
    reservation: _TensorReservation,
    output_tokens: int | None = None,
) -> UsageUpdate:
    input_tokens = len(reservation.prompt_tokens)
    generated = reservation.output_tokens if output_tokens is None else output_tokens
    return UsageUpdate(
        input_tokens=input_tokens,
        output_tokens=generated,
        total_tokens=input_tokens + generated,
        reasoning_output_tokens=min(
            reservation.output_state.reasoning_tokens,
            generated,
        ),
    )


_SAMPLER_FIELDS = {
    "temperature",
    "top_p",
    "top_k",
    "presence_penalty",
    "frequency_penalty",
}


def _parse_tensor_sampling(
    request: GenerationRequest,
    *,
    context_length: int,
    prompt_tokens: int,
) -> _TensorSamplingConfig:
    sampling = request.sampling
    allowed = _SAMPLER_FIELDS | {
        "max_output_tokens",
        "max_tokens",
        "seed",
        "draft_sampler",
        "parallel_tool_calls",
        "tool_choice",
    }
    unsupported = sorted(
        key
        for key, value in sampling.items()
        if value is not None and key not in allowed
    )
    if unsupported:
        raise ValueError(
            "Q4-TENSOR-TEXT cannot preserve unsupported sampling controls: "
            f"controls: {unsupported!r}"
        )

    canonical_limit = sampling.get("max_output_tokens")
    legacy_limit = sampling.get("max_tokens")
    if (
        canonical_limit is not None
        and legacy_limit is not None
        and canonical_limit != legacy_limit
    ):
        raise ValueError("max_output_tokens and max_tokens must agree")
    max_output_tokens = canonical_limit if canonical_limit is not None else legacy_limit
    if max_output_tokens is not None and (
        isinstance(max_output_tokens, bool)
        or not isinstance(max_output_tokens, int)
        or max_output_tokens < 1
    ):
        raise ValueError("Q4-TENSOR-TEXT max_output_tokens must be a positive integer")
    if prompt_tokens >= context_length:
        raise ValueError(
            "Q4-TENSOR-TEXT tokenized prompt length "
            f"{prompt_tokens} must be smaller than model context length "
            f"{context_length}"
        )
    max_output_tokens = resolve_max_tokens(
        requested=max_output_tokens,
        context_length=context_length,
        prompt_tokens=prompt_tokens,
        fallback=None,
        context_label=request.runtime.model_id,
    )
    target_values = {
        key: sampling[key] for key in _SAMPLER_FIELDS if sampling.get(key) is not None
    }
    target = _sampler_from_mapping(target_values, label="target")

    raw_draft = sampling.get("draft_sampler")
    if raw_draft is None:
        draft = target
    elif isinstance(raw_draft, Mapping):
        draft = _sampler_from_mapping(raw_draft, label="draft")
    else:
        raise ValueError("Q4-TENSOR-TEXT draft_sampler must be a mapping")

    seed = sampling.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Q4-TENSOR-TEXT seed must be an integer")
    return _TensorSamplingConfig(
        max_output_tokens=max_output_tokens,
        target=target,
        draft=draft,
        seed=seed,
    )


def _chat_template_tools(
    request: GenerationRequest,
) -> list[Mapping[str, Any]] | None:
    tools = list(request.tools)
    choice = request.sampling.get("tool_choice")
    if choice is None or (isinstance(choice, str) and choice in {"auto", "required"}):
        return tools or None
    if choice == "none":
        return None
    if not isinstance(choice, Mapping):
        raise ValueError(f"unsupported tool_choice: {choice!r}")
    if choice.get("type") != "function":
        raise ValueError("Q4-TENSOR-TEXT only supports function tool_choice")
    name = choice.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("function tool_choice requires a non-empty name")
    selected = [tool for tool in tools if tool.get("name") == name]
    if len(selected) != 1:
        raise ValueError("function tool_choice must name exactly one request tool")
    return selected


def _chat_template_reasoning(request: GenerationRequest) -> dict[str, object]:
    """Translate Responses reasoning controls to the checkpoint template ABI."""

    reasoning = request.reasoning
    enabled = reasoning.get("enabled")
    if enabled is not None and not isinstance(enabled, bool):
        raise ValueError("reasoning.enabled must be a boolean")

    effort = reasoning.get("effort")
    if effort is not None and not isinstance(effort, str):
        raise ValueError("reasoning.effort must be text")
    if enabled is False or effort in {"none", "off"}:
        return {"enable_thinking": False}
    if effort is None:
        return {"enable_thinking": True} if enabled is True else {}

    template_effort = "xhigh" if effort == "high" else effort
    if template_effort not in {"low", "medium", "xhigh"}:
        raise ValueError(
            "reasoning.effort must be one of none, off, low, medium, high, xhigh"
        )
    return {
        "enable_thinking": True,
        "reasoning_effort": template_effort,
    }


def _sampler_from_mapping(
    values: Mapping[str, Any],
    *,
    label: str,
) -> SamplerConfig:
    unsupported = sorted(
        key
        for key, value in values.items()
        if value is not None and key not in _SAMPLER_FIELDS
    )
    if unsupported:
        raise ValueError(
            f"Q4-TENSOR-TEXT {label} sampler has unsupported controls: {unsupported!r}"
        )

    defaults = SamplerConfig()
    temperature = _finite_number(
        values.get("temperature", defaults.temperature),
        name=f"{label}.temperature",
    )
    top_p = _finite_number(
        values.get("top_p", defaults.top_p),
        name=f"{label}.top_p",
    )
    if not 0 <= top_p <= 1:
        raise ValueError(f"Q4-TENSOR-TEXT {label}.top_p must be between 0 and 1")
    top_k = values.get("top_k", defaults.top_k)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
        raise ValueError(f"Q4-TENSOR-TEXT {label}.top_k must be a non-negative integer")

    presence = _finite_number(
        values.get("presence_penalty", defaults.presence_penalty),
        name=f"{label}.presence_penalty",
    )
    frequency = _finite_number(
        values.get("frequency_penalty", defaults.frequency_penalty),
        name=f"{label}.frequency_penalty",
    )
    if presence != 0.0 or frequency != 0.0:
        raise ValueError(
            "Q4-TENSOR-TEXT completion penalties are not wired into exact "
            "tensor sampling"
        )
    return SamplerConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        presence_penalty=presence,
        frequency_penalty=frequency,
    )


def _finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"Q4-TENSOR-TEXT {name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Q4-TENSOR-TEXT {name} must be finite")
    return number


def _distribution_from_mlx_logits(
    logits: Any,
    config: SamplerConfig,
) -> Distribution:
    logits = logits.astype(mx.float32)
    mx.eval(logits)
    dense = np.asarray(logits, dtype=np.float32).astype(np.float64).reshape(-1)
    return distribution_from_logits(dense, config)


def _require_sampling_rng(
    rng: np.random.Generator | None,
) -> np.random.Generator:
    if rng is None:
        raise RuntimeError("stochastic tensor sampling requires a request RNG")
    return rng


def _sample_from_logits(
    logits: Any,
    config: SamplerConfig,
    rng: np.random.Generator | None,
) -> tuple[int, Distribution | None]:
    if config.temperature <= 0:
        mx.eval(logits)
        return int(mx.argmax(logits, axis=-1).item()), None
    distribution = _distribution_from_mlx_logits(logits, config)
    token = sample_from_distribution(distribution, _require_sampling_rng(rng))
    return token, distribution


def _sample_draft_from_logits(
    logits: Any,
    config: SamplerConfig,
    rng: np.random.Generator | None,
    *,
    need_distribution: bool,
) -> tuple[int, Distribution | None]:
    if config.temperature > 0:
        return _sample_from_logits(logits, config, rng)
    mx.eval(logits)
    token = int(mx.argmax(logits, axis=-1).item())
    if not need_distribution:
        return token, None
    return token, SparseDistribution.one_hot(token, int(logits.shape[-1]))


def _runtime_identity(runtime: RuntimeKey) -> str:
    return repr(
        (
            runtime.model_id,
            runtime.revision,
            runtime.adapter_path,
            runtime.draft_model_id,
            runtime.backend.value,
        )
    )


def _prefix_namespace_signature(
    plan: Qwen4ExpModelLoadPlan,
    capabilities: Qwen4ExpTensorCapabilities,
    block_size_tokens: int,
) -> str:
    identity = repr(
        (
            _PREFIX_PAYLOAD_SCHEMA,
            plan.plan_sha256,
            plan.tokenizer_fingerprint,
            plan.topology.layer_types,
            plan.topology.ple_layers,
            plan.topology.mtp_layers,
            tuple(sorted(capabilities.enabled)),
            block_size_tokens,
        )
    ).encode("utf-8")
    return f"qwen4exp-wb-v1:{hashlib.sha256(identity).hexdigest()}"


def _prefix_context_fingerprint(request: PreparedGenerationRequest) -> str:
    if request.modality is RequestModality.TEXT:
        return TEXT_CONTEXT_FINGERPRINT
    payload = request.backend_payload
    if not isinstance(payload, PreparedQwen4Prompt):
        raise ValueError("vision prefix identity requires a sealed prompt")
    digest = payload.resolution.digest
    if not digest:
        raise ValueError("vision prefix identity requires a media bundle digest")
    return digest


def _prefix_release_reason(
    reason: CacheReleaseReason,
) -> Qwen4ExpPrefixReleaseReason:
    return Qwen4ExpPrefixReleaseReason(reason.value)


def _positive_prefix_option(
    options: Mapping[str, Any],
    name: str,
    default: int,
) -> int:
    value = options.get(name, default)
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _prefix_config(config: LoadConfig) -> _TensorPrefixConfig:
    options = config.options
    if options.get("qwen4_exp_paged_cache_enabled", False) is not False:
        raise ValueError("Qwen4Exp paged cache is outside the row-serial v1 runtime")
    if options.get("qwen4_exp_prefix_ssd_enabled", False) is not False:
        raise ValueError(
            "Qwen4Exp whole-boundary SSD cache has no concrete serializer port"
        )
    return _TensorPrefixConfig(
        block_size_tokens=_positive_prefix_option(
            options,
            "qwen4_exp_prefix_block_size_tokens",
            256,
        ),
        max_hot_entries=_positive_prefix_option(
            options,
            "qwen4_exp_prefix_max_hot_entries",
            8,
        ),
        max_hot_tokens=_positive_prefix_option(
            options,
            "qwen4_exp_prefix_max_hot_tokens",
            32768,
        ),
        max_active_leases=config.max_admitted_requests,
    )


def _is_grammar_constrained(request: GenerationRequest) -> bool:
    grammar_keys = {
        "compiled_grammar",
        "grammar",
        "guided_json",
        "response_format",
        "text",
    }
    return any(request.sampling.get(key) is not None for key in grammar_keys)


class Qwen4ExpExecutionFactory:
    """Sole concrete Qwen4ExpExecutionFactoryPort implementation."""

    def prepare(
        self,
        runtime: RuntimeKey,
        config: LoadConfig,
        scheduler_config: SchedulerConfig,
    ) -> _PreparedQwen4ExpExecutionFactory:
        try:
            model_dir = config.options["model_dir"]
        except KeyError as error:
            raise ValueError(
                'config.options["model_dir"] must be an absolute path'
            ) from error
        if not isinstance(model_dir, str) or not Path(model_dir).is_absolute():
            raise ValueError('config.options["model_dir"] must be an absolute path')
        if not runtime.revision:
            raise ValueError("Qwen4Exp runtime requires an exact revision")
        plan = load_qwen4_exp_plan(
            model_dir=model_dir,
            model_id=runtime.model_id,
            revision=runtime.revision,
        )
        capabilities = Qwen4ExpTensorCapabilities.from_options(config.options)
        return _PreparedQwen4ExpExecutionFactory(
            runtime=runtime,
            config=config,
            scheduler_config=scheduler_config,
            model_plan=plan,
            capabilities=capabilities,
            prefix_config=_prefix_config(config),
        )


@dataclass(frozen=True, slots=True)
class _PreparedQwen4ExpExecutionFactory:
    """Immutable metadata receipt consumed by the inference owner thread."""

    runtime: RuntimeKey
    config: LoadConfig
    scheduler_config: SchedulerConfig
    model_plan: Qwen4ExpModelLoadPlan
    capabilities: Qwen4ExpTensorCapabilities
    prefix_config: _TensorPrefixConfig

    def __post_init__(self) -> None:
        plan = self.model_plan
        configured_model_dir = self.config.options.get("model_dir")
        if configured_model_dir != plan.model_dir:
            raise ValueError(
                "prepared Qwen4Exp model_dir does not match the canonical plan"
            )
        if plan.model_id != self.runtime.model_id:
            raise ValueError("prepared Qwen4Exp plan has a different model id")
        if not self.runtime.revision or plan.revision != self.runtime.revision:
            raise ValueError("prepared Qwen4Exp plan has a different revision")

    def load(self) -> Qwen4ExpExecutionBinding:
        plan = self.model_plan
        with tensor_capability_scope(self.capabilities):
            model = load_qwen4_exp_tensor(plan, self.capabilities)
            tokenizer = load_tokenizer(Path(plan.model_dir))
        build_facts = {
            "qwen4_exp_plan_sha256": plan.plan_sha256,
            "config_sha256": plan.config_sha256,
            "index_sha256": plan.index_sha256,
            "artifact_inventory_sha256": plan.artifacts.digest,
            "tokenizer_fingerprint": plan.tokenizer_fingerprint,
            "tensor_batch_mode": plan.topology.tensor_batch_mode.value,
            "prefix_cache_mode": "whole_boundary_hot",
            "prefix_cache_payload_schema": _PREFIX_PAYLOAD_SCHEMA,
            "prefix_cache_block_size_tokens": self.prefix_config.block_size_tokens,
            "prefix_cache_ssd_enabled": False,
        }
        model_spec = ModelSpec(
            model_id=plan.model_id,
            revision=plan.revision,
            architecture=plan.config.architectures[0],
            model_type=plan.config.model_type,
            quantization=(
                f"{plan.config.quantization_bits}-bit/"
                f"g{plan.config.quantization_group_size}/"
                f"{plan.config.quantization_mode}"
            ),
            local_path=plan.model_dir,
            metadata={
                "qwen4_exp_plan_sha256": plan.plan_sha256,
                "build_facts": build_facts,
                "embedded_mtp": plan.artifacts.has_embedded_mtp,
                "embedded_ple": plan.artifacts.has_embedded_ple,
                "max_qsa_batch_rows": plan.topology.max_qsa_batch_rows,
            },
        )
        execution = _Qwen4ExpTensorRuntime(
            plan=plan,
            model=model,
            tokenizer=tokenizer,
            model_spec=model_spec,
            capabilities=self.capabilities,
            prefix_config=self.prefix_config,
        )
        return Qwen4ExpExecutionBinding(
            execution=execution,
            runtime=self.runtime,
            config=self.config,
            scheduler_config=self.scheduler_config,
            model=model_spec,
        )


__all__ = [
    "Attention",
    "GatedDeltaNet",
    "GatedResidual",
    "Model",
    "NGramEmbedding",
    "NGramTable",
    "PLELayer",
    "QSACache",
    "QSAIndexer",
    "Qwen4ExpExecutionFactory",
    "Qwen4ExpMTP",
    "Qwen4ExpTextModel",
    "SparseMoeBlock",
    "TextModel",
    "load_qwen4_exp_tensor",
    "qsa_prefill_engagement",
    "verify_capture_scope",
]
