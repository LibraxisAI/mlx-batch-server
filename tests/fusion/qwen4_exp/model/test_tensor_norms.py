from __future__ import annotations

import mlx.core as mx

from mlx_batch_server.runtime.fusion.qwen4_exp.model.tensor import (
    GroupedRMSNorm,
    _normalize_ones_centered_rmsnorm_weights,
)


def test_qwen4_rmsnorm_applies_zero_centered_residual_weight() -> None:
    norm = GroupedRMSNorm(4, group_size=2, eps=0.0)
    norm.weight = mx.array([0.0, 0.0, 1.0, -0.5])
    values = mx.array([[3.0, 4.0, 5.0, 12.0]])

    actual = norm(values)
    base = values.reshape(1, 2, 2)
    base = base * mx.rsqrt(mx.mean(mx.square(base), axis=-1, keepdims=True))
    expected = base.reshape(values.shape) * mx.array([1.0, 1.0, 2.0, 0.5])

    assert mx.allclose(actual, expected)


def test_legacy_ones_centered_checkpoint_norms_are_recentered_once() -> None:
    anchors = [
        (
            f"language_model.model.layers.{index}.attn_hyper_connection.hc_norm",
            GroupedRMSNorm(2),
        )
        for index in range(8)
    ]
    mtp = ("mtp.pre_fc_norm_embedding", GroupedRMSNorm(2))

    class _Model:
        @staticmethod
        def named_modules():
            return (*anchors, mtp)

    weights = {
        f"{path}.weight": mx.ones((2,), dtype=mx.bfloat16)
        for path, _module in (*anchors, mtp)
    }

    _normalize_ones_centered_rmsnorm_weights(_Model(), weights)

    for value in weights.values():
        assert mx.allclose(value, mx.zeros((2,)))


def test_zero_centered_checkpoint_norms_are_not_shifted() -> None:
    anchors = [
        (
            f"language_model.model.layers.{index}.attn_hyper_connection.hc_norm",
            GroupedRMSNorm(2),
        )
        for index in range(8)
    ]

    class _Model:
        @staticmethod
        def named_modules():
            return tuple(anchors)

    weights = {
        f"{path}.weight": mx.array([0.125, -0.25], dtype=mx.float32)
        for path, _module in anchors
    }

    _normalize_ones_centered_rmsnorm_weights(_Model(), weights)

    for value in weights.values():
        assert mx.allclose(value, mx.array([0.125, -0.25]))
