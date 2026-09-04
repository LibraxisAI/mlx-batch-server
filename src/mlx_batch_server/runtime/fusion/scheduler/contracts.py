"""Protocol-neutral contracts for the fused runtime scheduler.

The lifecycle shape is adapted from oMLX ``omlx/request.py`` and
``omlx/scheduler.py`` at commit
``e467261edc786efd33b1e9023d5c4a827f8aa1c1`` (Apache-2.0). LibraxisAI
reduced it to target-owned row planning: there are no MLX, HTTP, model-pool,
cache, parser, or protocol dependencies in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class WorkKind(StrEnum):
    TEXT = "text"
    VISION = "vision"


class RequestPhase(StrEnum):
    WAITING = "waiting"
    PREFILL = "prefill"
    DECODE = "decode"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SubmitDisposition(StrEnum):
    ACCEPTED = "accepted"
    DUPLICATE = "duplicate"
    CAPACITY = "capacity"


class CancelDisposition(StrEnum):
    REQUESTED = "requested"
    ALREADY_REQUESTED = "already_requested"
    ALREADY_TERMINAL = "already_terminal"
    NOT_FOUND = "not_found"


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    """Static row limits after the process-level admission lease is acquired."""

    max_admitted_requests: int = 8
    max_decode_rows: int = 4
    max_prefill_rows: int = 2
    max_vision_prefills: int = 2
    decode_fair_share: float = 0.5
    terminal_history_size: int = 128

    def __post_init__(self) -> None:
        positive = {
            "max_admitted_requests": self.max_admitted_requests,
            "max_decode_rows": self.max_decode_rows,
            "max_prefill_rows": self.max_prefill_rows,
            "max_vision_prefills": self.max_vision_prefills,
            "terminal_history_size": self.terminal_history_size,
        }
        for name, value in positive.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if self.max_decode_rows > self.max_admitted_requests:
            raise ValueError("max_decode_rows cannot exceed max_admitted_requests")
        if self.max_prefill_rows > self.max_decode_rows:
            raise ValueError("max_prefill_rows cannot exceed max_decode_rows")
        if self.max_vision_prefills > self.max_prefill_rows:
            raise ValueError("max_vision_prefills cannot exceed max_prefill_rows")
        if self.decode_fair_share < 0:
            raise ValueError("decode_fair_share must be non-negative")


@dataclass(frozen=True, slots=True)
class SchedulerRequest:
    request_id: str
    kind: WorkKind = WorkKind.TEXT
    initial_position: int = 0

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if self.initial_position < 0:
            raise ValueError("initial_position must be non-negative")


@dataclass(frozen=True, slots=True)
class SubmitResult:
    request_id: str
    disposition: SubmitDisposition
    reason: str


@dataclass(frozen=True, slots=True)
class CancelResult:
    request_id: str
    disposition: CancelDisposition
    reason: str


@dataclass(frozen=True, slots=True)
class CancelledRequest:
    request_id: str
    previous_phase: RequestPhase
    reason: str


@dataclass(frozen=True, slots=True)
class ScheduledRequest:
    request_id: str
    kind: WorkKind
    position: int


@dataclass(frozen=True, slots=True)
class SchedulerPlan:
    step_id: int
    prefill_rows: tuple[ScheduledRequest, ...] = ()
    decode_rows: tuple[ScheduledRequest, ...] = ()
    cancelled_requests: tuple[CancelledRequest, ...] = ()

    @property
    def requires_completion(self) -> bool:
        return bool(self.prefill_rows or self.decode_rows)

    @property
    def prefill_request_ids(self) -> tuple[str, ...]:
        return tuple(row.request_id for row in self.prefill_rows)

    @property
    def decode_request_ids(self) -> tuple[str, ...]:
        return tuple(row.request_id for row in self.decode_rows)


@dataclass(frozen=True, slots=True)
class PrefillResult:
    request_id: str
    position: int
    complete: bool = False
    failed_reason: str | None = None

    def __post_init__(self) -> None:
        if self.complete and self.failed_reason is not None:
            raise ValueError("a prefill result cannot be complete and failed")


@dataclass(frozen=True, slots=True)
class DecodeResult:
    request_id: str
    position: int
    finished: bool = False
    finish_reason: str | None = None
    failed_reason: str | None = None

    def __post_init__(self) -> None:
        if self.finished and self.failed_reason is not None:
            raise ValueError("a decode result cannot be finished and failed")


@dataclass(frozen=True, slots=True)
class TerminalRequest:
    request_id: str
    phase: RequestPhase
    reason: str


@dataclass(frozen=True, slots=True)
class SchedulerUpdate:
    step_id: int
    terminal_requests: tuple[TerminalRequest, ...] = ()
    decode_time_owed_s: float = 0.0


@dataclass(frozen=True, slots=True)
class RequestSnapshot:
    request_id: str
    kind: WorkKind
    phase: RequestPhase
    position: int
    arrival_sequence: int


@dataclass(frozen=True, slots=True)
class SchedulerSnapshot:
    step_id: int
    admitted_requests: int
    waiting_requests: int
    prefilling_requests: int
    decoding_requests: int
    decode_time_owed_s: float
    requests: tuple[RequestSnapshot, ...]
    recent_terminal: tuple[TerminalRequest, ...]
