# SPDX-License-Identifier: Apache-2.0
"""Pure QSA capacity and MTP-history replay plans.

Adapted from MTPLX ``mtplx/qsa_mtp_precompute.py`` at
``6d0ddf0575faa9acf77e63c57e48ea1602a7e4ab``. LibraxisAI removed environment
gates and cache duck typing; the tensor adapter must explicitly consume these
immutable plans after the MLX ABI is selected.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeVar


def _nonnegative(name: str, value: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _positive(name: str, value: int) -> int:
    value = _nonnegative(name, value)
    if value == 0:
        raise ValueError(f"{name} must be positive")
    return value


def qsa_indexer_is_bucket_capacity(capacity: int, *, minimum: int = 256) -> bool:
    capacity = _nonnegative("capacity", capacity)
    minimum = _positive("minimum", minimum)
    if minimum & (minimum - 1):
        raise ValueError("minimum must be a power of two")
    return capacity == 0 or (capacity >= minimum and capacity & (capacity - 1) == 0)


def qsa_indexer_capacity_bucket(required: int, *, minimum: int = 256) -> int:
    required = _nonnegative("required", required)
    minimum = _positive("minimum", minimum)
    if minimum & (minimum - 1):
        raise ValueError("minimum must be a power of two")
    if required == 0:
        return 0
    return max(minimum, 1 << (required - 1).bit_length())


@dataclass(frozen=True, slots=True)
class QSAReplayCapacity:
    start_offset: int
    window_tokens: int
    end_offset: int
    complete_blocks: int
    raw_capacity: int
    pooled_capacity: int
    compress_ratio: int
    allocation_step: int

    @property
    def graph_key(self) -> tuple[int, int]:
        return (self.raw_capacity, self.pooled_capacity)


def precompute_qsa_replay_capacity(
    *,
    start_offset: int,
    window_tokens: int,
    compress_ratio: int,
    allocation_step: int = 256,
    current_raw_capacity: int = 0,
    current_pooled_capacity: int = 0,
) -> QSAReplayCapacity:
    start = _nonnegative("start_offset", start_offset)
    width = _nonnegative("window_tokens", window_tokens)
    ratio = _positive("compress_ratio", compress_ratio)
    step = _positive("allocation_step", allocation_step)
    if step & (step - 1):
        raise ValueError("allocation_step must be a power of two")
    current_raw = _nonnegative("current_raw_capacity", current_raw_capacity)
    current_pooled = _nonnegative("current_pooled_capacity", current_pooled_capacity)
    end = start + width
    complete_blocks = end // ratio
    staging_blocks = (width + ratio - 1) // ratio
    staging_tokens = staging_blocks * ratio
    return QSAReplayCapacity(
        start_offset=start,
        window_tokens=width,
        end_offset=end,
        complete_blocks=complete_blocks,
        raw_capacity=qsa_indexer_capacity_bucket(
            max(current_raw, end, staging_tokens), minimum=step
        ),
        pooled_capacity=qsa_indexer_capacity_bucket(
            max(current_pooled, complete_blocks, staging_blocks), minimum=step
        ),
        compress_ratio=ratio,
        allocation_step=step,
    )


_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class MTPIndexerReplayPlan:
    cycle_offset: int
    observed_offset: int
    speculative_rows: int
    primary_staged: bool
    reusable_rows: int
    rollback_offset: int
    reappend_start: int

    def reappend_tokens(self, committed: Sequence[_T]) -> tuple[_T, ...]:
        return tuple(committed[self.reappend_start :])

    def authoritative_hidden_rows(self, committed_count: int) -> int:
        count = _nonnegative("committed_count", committed_count)
        if count < self.reappend_start:
            raise ValueError("committed_count is smaller than the retained prefix")
        return count - self.reappend_start


def precompute_mtp_indexer_replay(
    *, cycle_offset: int, observed_offset: int
) -> MTPIndexerReplayPlan:
    base = _nonnegative("cycle_offset", cycle_offset)
    observed = _nonnegative("observed_offset", observed_offset)
    if observed < base:
        raise ValueError("observed_offset cannot precede cycle_offset")
    speculative_rows = observed - base
    reusable = 1 if speculative_rows else 0
    return MTPIndexerReplayPlan(
        cycle_offset=base,
        observed_offset=observed,
        speculative_rows=speculative_rows,
        primary_staged=bool(reusable),
        reusable_rows=reusable,
        rollback_offset=base + reusable,
        reappend_start=reusable,
    )


__all__ = [
    "MTPIndexerReplayPlan",
    "QSAReplayCapacity",
    "precompute_mtp_indexer_replay",
    "precompute_qsa_replay_capacity",
    "qsa_indexer_capacity_bucket",
    "qsa_indexer_is_bucket_capacity",
]
