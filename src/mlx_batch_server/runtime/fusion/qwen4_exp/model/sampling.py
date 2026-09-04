# SPDX-License-Identifier: Apache-2.0
# Adapted from MTPLX commit 6d0ddf0575faa9acf77e63c57e48ea1602a7e4ab.
"""Pure sampling contract for exact stochastic speculative decoding."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias, cast

import numpy as np

PENALTY_MIN = -2.0
PENALTY_MAX = 2.0


@dataclass(frozen=True)
class SamplerConfig:
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0


@dataclass(frozen=True)
class SparseDistribution:
    token_ids: np.ndarray
    probs: np.ndarray
    vocab_size: int

    def __post_init__(self) -> None:
        token_ids = np.asarray(self.token_ids, dtype=np.int64)
        probs = np.asarray(self.probs, dtype=np.float64)
        if token_ids.ndim != 1 or probs.ndim != 1:
            raise ValueError("SparseDistribution expects 1D token_ids and probs")
        if token_ids.shape[0] != probs.shape[0]:
            raise ValueError("SparseDistribution token_ids/probs length mismatch")
        if token_ids.shape[0] == 0:
            raise ValueError("SparseDistribution cannot be empty")
        if np.any(np.isfinite(probs) & (probs < 0)):
            raise ValueError("SparseDistribution probabilities must be non-negative")
        sanitized = np.where(np.isfinite(probs) & (probs > 0), probs, 0.0)
        total = sanitized.sum()
        if not np.isfinite(total) or total <= 0:
            valid_ids = token_ids[(token_ids >= 0) & (token_ids < int(self.vocab_size))]
            if valid_ids.shape[0] == 0:
                raise ValueError(
                    "SparseDistribution probabilities must have positive mass"
                )
            token_ids = np.array([int(valid_ids[0])], dtype=np.int64)
            sanitized = np.array([1.0], dtype=np.float64)
            total = 1.0
        object.__setattr__(self, "token_ids", token_ids)
        object.__setattr__(self, "probs", sanitized / total)

    @classmethod
    def one_hot(cls, token_id: int, vocab_size: int) -> SparseDistribution:
        return cls(
            np.array([int(token_id)], dtype=np.int64),
            np.array([1.0], dtype=np.float64),
            vocab_size,
        )

    def probability(self, token_id: int) -> float:
        hits = np.nonzero(self.token_ids == int(token_id))[0]
        if hits.size == 0:
            return 0.0
        return float(self.probs[int(hits[0])])

    def to_dense(self) -> np.ndarray:
        dense = np.zeros(int(self.vocab_size), dtype=np.float64)
        dense[self.token_ids] = self.probs
        return dense


Distribution: TypeAlias = np.ndarray | SparseDistribution


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    if temperature <= 0:
        out = np.zeros_like(logits, dtype=np.float64)
        out[int(np.argmax(logits))] = 1.0
        return out
    scaled = logits / float(temperature)
    scaled = scaled - np.max(scaled)
    exp = np.exp(scaled)
    total = np.sum(exp)
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Cannot normalize logits into a probability distribution")
    return exp / total


def deterministic_top_k_order(values: np.ndarray, top_k: int) -> np.ndarray:
    """Return highest-value token ids, breaking exact ties by vocabulary id."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("Expected a 1D value vector")
    size = int(values.shape[0])
    count = min(max(int(top_k), 0), size)
    if count == 0:
        return np.empty(0, dtype=np.int64)
    token_ids = np.arange(size, dtype=np.int64)
    if count == size:
        return np.lexsort((token_ids, -values)).astype(np.int64, copy=False)
    cutoff = np.partition(values, size - count)[size - count]
    higher = np.flatnonzero(values > cutoff)
    tied = np.flatnonzero(values == cutoff)
    chosen = np.concatenate((higher, tied[: count - higher.size]))
    order = np.lexsort((chosen, -values[chosen]))
    return chosen[order].astype(np.int64, copy=False)


def apply_top_p_top_k(
    probs: np.ndarray,
    top_p: float = 1.0,
    top_k: int = 0,
) -> np.ndarray:
    """Apply the frozen donor's top-p then top-k filtering semantics."""

    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 1:
        raise ValueError("Expected a 1D probability vector")
    size = int(probs.shape[0])
    bounded_top_k = int(top_k) if top_k and 0 < int(top_k) < size else 0
    mask = np.ones(size, dtype=bool)
    ranked: np.ndarray | None = None
    if bounded_top_k:
        ranked = deterministic_top_k_order(probs, bounded_top_k)
    if 0 < top_p < 1.0:
        order = ranked
        if order is None:
            order = deterministic_top_k_order(probs, size)
        sorted_probs = probs[order]
        cumulative = np.cumsum(sorted_probs)
        cumulative_before = np.concatenate(([0.0], cumulative[:-1]))
        keep_sorted = cumulative_before < top_p
        nucleus_mask = np.zeros_like(mask)
        nucleus_mask[order[keep_sorted]] = True
        mask &= nucleus_mask
    if bounded_top_k:
        top_mask = np.zeros_like(mask)
        top_mask[ranked] = True
        mask &= top_mask
    filtered = np.where(mask, probs, 0.0)
    total = filtered.sum()
    if total <= 0:
        filtered[int(np.argmax(probs))] = 1.0
        return filtered
    return filtered / total


def apply_top_k_top_p(
    probs: np.ndarray,
    top_k: int = 0,
    top_p: float = 1.0,
) -> np.ndarray:
    """Backward-compatible name for the donor's top-p then top-k contract."""

    return apply_top_p_top_k(probs, top_p=top_p, top_k=top_k)


def apply_penalties(
    logits: np.ndarray,
    token_counts: Mapping[int, int] | None,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    penalty_overlay: Mapping[int, float] | None = None,
) -> np.ndarray:
    """Apply additive completion-token penalties to raw logits."""

    presence = float(np.clip(presence_penalty, PENALTY_MIN, PENALTY_MAX))
    frequency = float(np.clip(frequency_penalty, PENALTY_MIN, PENALTY_MAX))
    counts_active = bool(token_counts) and (presence != 0.0 or frequency != 0.0)
    overlay_active = bool(penalty_overlay)
    if not counts_active and not overlay_active:
        return logits
    out = np.array(logits, dtype=np.float64, copy=True)
    if counts_active and token_counts is not None:
        ids = np.fromiter(token_counts.keys(), dtype=np.int64, count=len(token_counts))
        counts = np.fromiter(
            token_counts.values(), dtype=np.float64, count=len(token_counts)
        )
        out[ids] -= frequency * counts + presence * (counts > 0)
    if overlay_active and penalty_overlay is not None:
        overlay_ids = np.fromiter(
            penalty_overlay.keys(),
            dtype=np.int64,
            count=len(penalty_overlay),
        )
        overlay_values = np.fromiter(
            penalty_overlay.values(),
            dtype=np.float64,
            count=len(penalty_overlay),
        )
        out[overlay_ids] -= overlay_values
    return out


def distribution_from_logits(
    logits: np.ndarray,
    config: SamplerConfig,
    *,
    token_counts: Mapping[int, int] | None = None,
    penalty_overlay: Mapping[int, float] | None = None,
) -> np.ndarray:
    logits = apply_penalties(
        logits,
        token_counts,
        config.presence_penalty,
        config.frequency_penalty,
        penalty_overlay=penalty_overlay,
    )
    probs = softmax(logits, temperature=config.temperature)
    return apply_top_p_top_k(probs, top_p=config.top_p, top_k=config.top_k)


def _probability(distribution: Distribution, token_id: int) -> float:
    if isinstance(distribution, SparseDistribution):
        return distribution.probability(token_id)
    return float(distribution[token_id])


def _vocab_size(distribution: Distribution) -> int:
    if isinstance(distribution, SparseDistribution):
        return int(distribution.vocab_size)
    return int(np.asarray(distribution).shape[0])


def _as_dense(distribution: Distribution) -> np.ndarray:
    if isinstance(distribution, SparseDistribution):
        return distribution.to_dense()
    return np.asarray(distribution, dtype=np.float64)


def acceptance_probability(
    target_p: Distribution,
    draft_q: Distribution,
    token_id: int,
) -> float:
    p = _probability(target_p, token_id)
    q = _probability(draft_q, token_id)
    if q <= 0:
        return 1.0 if p > 0 else 0.0
    return min(1.0, p / q)


def residual_distribution(
    target_p: Distribution,
    draft_q: Distribution,
) -> Distribution:
    if isinstance(target_p, SparseDistribution) or isinstance(
        draft_q, SparseDistribution
    ):
        if isinstance(target_p, SparseDistribution) and isinstance(
            draft_q, SparseDistribution
        ):
            token_ids = np.union1d(target_p.token_ids, draft_q.token_ids).astype(
                np.int64
            )
            residual = np.array(
                [
                    max(
                        target_p.probability(int(token))
                        - draft_q.probability(int(token)),
                        0.0,
                    )
                    for token in token_ids
                ],
                dtype=np.float64,
            )
            residual = np.where(np.isfinite(residual) & (residual > 0), residual, 0.0)
            keep = residual > 0
            total = residual[keep].sum()
            if not np.isfinite(total) or total <= 0:
                return target_p
            return SparseDistribution(
                token_ids[keep],
                residual[keep] / total,
                _vocab_size(target_p),
            )

        dense_target = _as_dense(target_p)
        dense_draft = _as_dense(draft_q)
        residual = np.maximum(dense_target - dense_draft, 0.0)
        residual = np.where(np.isfinite(residual) & (residual > 0), residual, 0.0)
        total = residual.sum()
        if not np.isfinite(total) or total <= 0:
            residual = np.where(
                np.isfinite(dense_target) & (dense_target > 0),
                dense_target,
                0.0,
            )
            total = residual.sum()
        if not np.isfinite(total) or total <= 0:
            raise ValueError("Cannot build residual distribution from empty target")
        return residual / total

    residual = np.maximum(np.asarray(target_p) - np.asarray(draft_q), 0.0)
    residual = np.where(np.isfinite(residual) & (residual > 0), residual, 0.0)
    total = residual.sum()
    if not np.isfinite(total) or total <= 0:
        dense_target = np.asarray(target_p, dtype=np.float64)
        residual = np.where(
            np.isfinite(dense_target) & (dense_target > 0), dense_target, 0.0
        )
        total = residual.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Cannot build residual distribution from empty target")
    return residual / total


def sample_from_distribution(
    probs: Distribution,
    rng: np.random.Generator | None = None,
) -> int:
    rng = rng or np.random.default_rng()
    if isinstance(probs, SparseDistribution):
        return int(rng.choice(probs.token_ids, p=probs.probs))
    dense_probs = np.asarray(cast("np.ndarray", probs), dtype=np.float64)
    dense_probs = dense_probs / dense_probs.sum()
    return int(rng.choice(np.arange(dense_probs.shape[0]), p=dense_probs))


@dataclass(frozen=True)
class SpeculativeDecision:
    accepted: bool
    token_id: int
    accept_probability: float


def verify_one_token(
    target_p: Distribution,
    draft_q: Distribution,
    draft_token: int,
    rng: np.random.Generator | None = None,
) -> SpeculativeDecision:
    rng = rng or np.random.default_rng()
    accept_p = acceptance_probability(target_p, draft_q, draft_token)
    if float(rng.random()) <= accept_p:
        return SpeculativeDecision(True, int(draft_token), accept_p)
    corrected = sample_from_distribution(residual_distribution(target_p, draft_q), rng)
    return SpeculativeDecision(False, corrected, accept_p)


def speculative_output_marginal(
    target_p: Distribution,
    draft_q: Distribution,
) -> np.ndarray:
    """Return the exact output marginal induced by one-token speculation."""

    target_dense = _as_dense(target_p)
    draft_dense = _as_dense(draft_q)
    target_dense = target_dense / target_dense.sum()
    draft_dense = draft_dense / draft_dense.sum()

    output = np.zeros_like(target_dense)
    for token_id, q_value in enumerate(draft_dense):
        accept_p = acceptance_probability(target_dense, draft_dense, token_id)
        output[token_id] += q_value * accept_p
        if accept_p < 1.0:
            residual = residual_distribution(target_dense, draft_dense)
            output += q_value * (1.0 - accept_p) * residual
    return output / output.sum()
