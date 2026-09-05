# SPDX-License-Identifier: Apache-2.0
"""Pure host-side compaction contract for Qwen4-Exp tensor batches."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class MultirowBatchPlan(Generic[_T]):
    """Stable ordinal map between a scheduler cohort and an active tensor.

    Finished or cancelled rows never enter the tensor call, but their removal
    must not renumber the output slots consumed by scheduler/event identity.
    """

    row_count: int
    active_ordinals: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.row_count < 0:
            raise ValueError("row_count must be non-negative")
        if self.active_ordinals != tuple(sorted(set(self.active_ordinals))):
            raise ValueError("active ordinals must be unique and ordered")
        if any(
            not 0 <= ordinal < self.row_count for ordinal in self.active_ordinals
        ):
            raise ValueError("active ordinal is outside the scheduler cohort")

    @classmethod
    def compact(cls, active: Sequence[bool]) -> MultirowBatchPlan[_T]:
        return cls(
            row_count=len(active),
            active_ordinals=tuple(
                ordinal for ordinal, keep in enumerate(active) if keep
            ),
        )

    def execute(
        self,
        forward: Callable[[tuple[int, ...]], Sequence[_T]],
    ) -> tuple[_T | None, ...]:
        """Call the tensor owner once, then scatter by original ordinal."""

        scattered: list[_T | None] = [None] * self.row_count
        if not self.active_ordinals:
            return tuple(scattered)
        values = tuple(forward(self.active_ordinals))
        if len(values) != len(self.active_ordinals):
            raise ValueError("tensor output count does not match active row count")
        for ordinal, value in zip(self.active_ordinals, values, strict=True):
            scattered[ordinal] = value
        return tuple(scattered)


__all__ = ["MultirowBatchPlan"]
