"""Protocol-neutral contracts for exact speculative decoding.

The contract follows MTPLX's probability-ratio verification model and oMLX's
continuous-batch handoff boundary. Both donor implementations are Apache-2.0;
this target-owned contract is a new, deliberately smaller adaptation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable


class MtpMode(StrEnum):
    DISABLED = "disabled"
    EXACT_SINGLETON = "exact_singleton"
    EXACT_ALIGNED_COHORT = "exact_aligned_cohort"


class MtpDisableReason(StrEnum):
    MODEL_UNSUPPORTED = "model_unsupported"
    HEAD_MISSING = "head_missing"
    DECODE_DISABLED = "decode_disabled"
    VERIFIER_MISSING = "verifier_missing"
    PROMPT_MERGE_PENDING = "prompt_merge_pending"
    GRAMMAR_PROCESSOR_UNSUPPORTED = "grammar_processor_unsupported"
    STOP_SEQUENCE_CONSTRAINED = "stop_sequence_constrained"
    MULTIROW_NOT_PROVEN = "multirow_not_proven"
    ROWS_UNALIGNED = "rows_unaligned"
    EMPTY_COHORT = "empty_cohort"


@dataclass(frozen=True, slots=True)
class MtpAlignment:
    """Facts required before a cohort may share the experimental MTP lane."""

    runtime_keys: tuple[str, ...]
    cache_positions: tuple[int, ...]
    pending_prompt_work: bool = False

    @property
    def row_count(self) -> int:
        return len(self.runtime_keys)

    @property
    def is_aligned(self) -> bool:
        if self.row_count == 0 or len(self.cache_positions) != self.row_count:
            return False
        return (
            not self.pending_prompt_work
            and len(set(self.runtime_keys)) == 1
            and len(set(self.cache_positions)) == 1
        )


@dataclass(frozen=True, slots=True)
class MtpDecision:
    enabled: bool
    mode: MtpMode
    exact: bool
    row_count: int
    disable_reason: MtpDisableReason | None = None
    facts: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MtpStats:
    rounds: int = 0
    drafted_tokens: int = 0
    accepted_tokens: int = 0
    rejected_tokens: int = 0
    fallback_counts: dict[MtpDisableReason, int] = field(default_factory=dict)

    def record_fallback(self, reason: MtpDisableReason) -> None:
        self.fallback_counts[reason] = self.fallback_counts.get(reason, 0) + 1

    @property
    def acceptance_rate(self) -> float:
        if self.drafted_tokens == 0:
            return 0.0
        return self.accepted_tokens / self.drafted_tokens


@dataclass(frozen=True, slots=True)
class VerificationResult:
    accepted_token_ids: tuple[int, ...]
    correction_token_id: int | None
    accepted_draft_count: int
    exact: bool = True


@runtime_checkable
class MtpVerifier(Protocol):
    """Exact target-distribution verifier; draft quality may only affect speed."""

    def verify(
        self,
        *,
        draft_token_ids: Sequence[int],
        draft_logprobs: Sequence[Any],
        target_logprobs: Sequence[Any],
        random_values: Sequence[float],
    ) -> VerificationResult: ...
