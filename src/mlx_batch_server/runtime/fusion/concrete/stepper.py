"""Tensor-free bridge from target scheduler plans to fused batch operations.

The low-level operation shape is adapted from oMLX ``omlx/scheduler.py`` and
``omlx/patches/mlx_lm_mtp/batch_generator.py`` at
``e467261edc786efd33b1e9023d5c4a827f8aa1c1`` (Apache-2.0). Exact MTP facts
and cleanup requirements follow MTPLX at
``6d0ddf0575faa9acf77e63c57e48ea1602a7e4ab`` (Apache-2.0). LibraxisAI
reduced both to an injected owner-thread port: no donor server, control plane,
model pool, tensor object, or optimistic capability declaration lives here.
"""

from __future__ import annotations

import threading
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from ...backends.fused_mtp_mlx import FusedStepResult
from ...contracts import GenerationRequest, ModelSpec, RuntimeKey
from ...events import TurnEvent, UsageUpdate
from ..mtp import MtpAlignment, MtpDecision, MtpDisableReason, MtpPolicy
from ..output import (
    Qwen4OutputChunk,
    Qwen4OutputEncoderFactoryPort,
    Qwen4OutputEncoderPort,
    Qwen4TurnEventEncoderFactory,
)
from ..scheduler import (
    DecodeResult,
    PrefillResult,
    ScheduledRequest,
    SchedulerPlan,
    WorkKind,
)


class OmlxBatchStepperError(RuntimeError):
    """Base failure at the target-to-owner stepping boundary."""


class OmlxBatchContractError(OmlxBatchStepperError):
    """The injected driver returned evidence that does not match its lease."""


class OmlxBatchCleanupError(OmlxBatchStepperError):
    """Cancellation cleanup did not acknowledge every owned resource."""


class LowLevelBatchKind(StrEnum):
    TEXT_PREFILL = "text_prefill"
    VISION_PREFILL = "vision_prefill"
    AR_DECODE = "ar_decode"
    MTP_DECODE = "mtp_decode"


class CleanupDisposition(StrEnum):
    RELEASED = "released"
    NOT_ALLOCATED = "not_allocated"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class MtpRuntimeFacts:
    """Live-proven MTP facts; all defaults deliberately disable MTP."""

    model_supported: bool = False
    head_attached: bool = False
    decode_enabled: bool = False
    verifier_available: bool = False

    def __post_init__(self) -> None:
        facts = (
            self.model_supported,
            self.head_attached,
            self.decode_enabled,
            self.verifier_available,
        )
        if any(type(value) is not bool for value in facts):
            raise TypeError("MTP runtime facts must be observed booleans")


@dataclass(frozen=True, slots=True)
class LowLevelBatchRow:
    request_id: str
    request: GenerationRequest
    kind: WorkKind
    position: int

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if self.request.response_id != self.request_id:
            raise ValueError("batch row request_id does not match GenerationRequest")
        if self.position < 0:
            raise ValueError("batch row position must be non-negative")
        expected_kind = WorkKind.VISION if self.request.media else WorkKind.TEXT
        if self.kind is not expected_kind:
            raise ValueError("batch row kind does not match request media")


@dataclass(frozen=True, slots=True)
class LowLevelBatchStep:
    scheduler_step_id: int
    kind: LowLevelBatchKind
    rows: tuple[LowLevelBatchRow, ...]
    mtp_decision: MtpDecision | None = None

    def __post_init__(self) -> None:
        if self.scheduler_step_id < 1:
            raise ValueError("scheduler_step_id must be positive")
        if not self.rows:
            raise ValueError("low-level batch step requires at least one row")
        request_ids = tuple(row.request_id for row in self.rows)
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("low-level batch rows must have unique request ids")

        if self.kind is LowLevelBatchKind.TEXT_PREFILL and any(
            row.kind is not WorkKind.TEXT for row in self.rows
        ):
            raise ValueError("text prefill cannot contain vision rows")
        if self.kind is LowLevelBatchKind.VISION_PREFILL and any(
            row.kind is not WorkKind.VISION for row in self.rows
        ):
            raise ValueError("vision prefill cannot contain text rows")

        is_decode = self.kind in (
            LowLevelBatchKind.AR_DECODE,
            LowLevelBatchKind.MTP_DECODE,
        )
        decision = self.mtp_decision
        if is_decode != (decision is not None):
            raise ValueError("decode steps require exactly one MTP decision")
        if self.kind is LowLevelBatchKind.MTP_DECODE and (
            decision is None or not decision.enabled or not decision.exact
        ):
            raise ValueError("MTP decode requires an enabled exact decision")
        if (
            self.kind is LowLevelBatchKind.AR_DECODE
            and decision is not None
            and decision.enabled
        ):
            raise ValueError("AR decode requires a disabled MTP decision")


@dataclass(frozen=True, slots=True)
class LowLevelRowResult:
    request_id: str
    position: int
    prefill_complete: bool = False
    finished: bool = False
    finish_reason: str | None = None
    failed_reason: str | None = None
    emitted_tokens: int = 0
    chunks: tuple[Qwen4OutputChunk, ...] = ()
    final_usage: UsageUpdate | None = None

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if self.position < 0:
            raise ValueError("row result position must be non-negative")
        if self.emitted_tokens < 0:
            raise ValueError("emitted_tokens must be non-negative")
        terminal_flags = int(self.prefill_complete) + int(self.finished)
        if terminal_flags > 1:
            raise ValueError("row cannot finish prefill and decode together")
        if self.failed_reason is not None and terminal_flags:
            raise ValueError("failed row cannot also complete")
        if self.finished and not self.finish_reason:
            raise ValueError("finished decode row requires a finish_reason")
        if not self.finished and self.finish_reason is not None:
            raise ValueError("unfinished row cannot carry a finish_reason")
        if any(not isinstance(chunk, Qwen4OutputChunk) for chunk in self.chunks):
            raise TypeError("row chunks must be Qwen4OutputChunk values")
        if self.final_usage is not None and not isinstance(
            self.final_usage,
            UsageUpdate,
        ):
            raise TypeError("final_usage must be a UsageUpdate")


@dataclass(frozen=True, slots=True)
class LowLevelBatchReceipt:
    scheduler_step_id: int
    kind: LowLevelBatchKind
    rows: tuple[LowLevelRowResult, ...]
    elapsed_s: float
    ar_decode_steps: int = 0
    ar_decode_tokens: int = 0
    mtp_rounds: int = 0
    mtp_drafted_tokens: int = 0
    mtp_accepted_tokens: int = 0
    mtp_rejected_tokens: int = 0

    def __post_init__(self) -> None:
        counters = (
            self.elapsed_s,
            self.ar_decode_steps,
            self.ar_decode_tokens,
            self.mtp_rounds,
            self.mtp_drafted_tokens,
            self.mtp_accepted_tokens,
            self.mtp_rejected_tokens,
        )
        if any(value < 0 for value in counters):
            raise ValueError("low-level receipt counters must be non-negative")
        if self.mtp_accepted_tokens > self.mtp_drafted_tokens:
            raise ValueError("accepted MTP tokens cannot exceed drafted tokens")


@dataclass(frozen=True, slots=True)
class CancellationCleanupReceipt:
    request_id: str
    reason: str
    execution_stopped: bool
    uid: CleanupDisposition
    kv: CleanupDisposition
    parser: CleanupDisposition
    media: CleanupDisposition
    errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("cleanup request_id must not be empty")
        if not self.reason:
            raise ValueError("cleanup reason must not be empty")
        dispositions = (self.uid, self.kv, self.parser, self.media)
        failed = any(item is CleanupDisposition.FAILED for item in dispositions)
        if self.errors and not failed:
            raise ValueError("cleanup errors require a failed component")
        if failed and not self.errors:
            raise ValueError("failed cleanup component requires an error")

    @property
    def succeeded(self) -> bool:
        complete = {CleanupDisposition.RELEASED, CleanupDisposition.NOT_ALLOCATED}
        dispositions = (self.uid, self.kv, self.parser, self.media)
        return (
            self.execution_stopped
            and not self.errors
            and all(item in complete for item in dispositions)
        )


@runtime_checkable
class OmlxBatchKernelPort(Protocol):
    """Synchronous batch mechanics called only on the tensor-owner thread.

    This is deliberately below ``Qwen4ExpExecutionPort``: the owner-level port
    owns request reservations and cache receipts, while this port only runs
    translated homogeneous prefill/decode batches and performs cancellation
    cleanup. It must never introduce an event loop or a second scheduler.
    """

    def run(self, step: LowLevelBatchStep) -> LowLevelBatchReceipt: ...

    def cleanup_cancelled(
        self,
        request_id: str,
        reason: str,
    ) -> CancellationCleanupReceipt: ...

    def stats(self) -> Mapping[str, Any]: ...

    def close(self, deadline_s: float) -> None: ...


class OmlxBatchStepper:
    """Translate scheduler leases into serial owner-thread batch mechanics.

    The surrounding ``Qwen4ExpTensorOwner`` supplies all cross-thread
    serialization. This object is synchronous by construction so the future
    MLX implementation cannot smuggle a second asyncio ownership domain onto
    the inference thread.
    """

    def __init__(
        self,
        *,
        model_spec: ModelSpec,
        driver: OmlxBatchKernelPort,
        mtp_facts: MtpRuntimeFacts | None = None,
        output_factory: Qwen4OutputEncoderFactoryPort | None = None,
    ) -> None:
        if not isinstance(model_spec, ModelSpec):
            raise TypeError("model_spec must be a ModelSpec")
        self._model_spec = model_spec
        self._driver = driver
        self._mtp_facts = mtp_facts or MtpRuntimeFacts()
        self._output_factory = output_factory or Qwen4TurnEventEncoderFactory()
        self._encoders: dict[str, Qwen4OutputEncoderPort] = {}
        self._cleanup_reasons: dict[str, str] = {}
        self._cleaned: set[str] = set()
        self._owner_thread_id = threading.get_ident()
        self._closed = False
        self._low_level_steps = 0
        self._fallbacks: Counter[MtpDisableReason] = Counter()

    @property
    def model_spec(self) -> ModelSpec:
        return self._model_spec

    def translate_plan(
        self,
        plan: SchedulerPlan,
        requests: Mapping[str, GenerationRequest],
        mtp_policy: MtpPolicy,
    ) -> tuple[LowLevelBatchStep, ...]:
        """Build homogeneous prefills plus one common decode batch."""
        self._validate_plan_requests(plan, requests)
        steps: list[LowLevelBatchStep] = []

        seen_kinds: list[WorkKind] = []
        for row in plan.prefill_rows:
            if row.kind not in seen_kinds:
                seen_kinds.append(row.kind)
        for work_kind in seen_kinds:
            rows = tuple(
                self._translate_row(row, requests[row.request_id])
                for row in plan.prefill_rows
                if row.kind is work_kind
            )
            step_kind = (
                LowLevelBatchKind.VISION_PREFILL
                if work_kind is WorkKind.VISION
                else LowLevelBatchKind.TEXT_PREFILL
            )
            steps.append(LowLevelBatchStep(plan.step_id, step_kind, rows))

        if plan.decode_rows:
            decode_rows = tuple(
                self._translate_row(row, requests[row.request_id])
                for row in plan.decode_rows
            )
            decision = self._mtp_decision(plan, decode_rows, mtp_policy)
            decode_kind = (
                LowLevelBatchKind.MTP_DECODE
                if decision.enabled
                else LowLevelBatchKind.AR_DECODE
            )
            steps.append(
                LowLevelBatchStep(
                    plan.step_id,
                    decode_kind,
                    decode_rows,
                    mtp_decision=decision,
                )
            )
        return tuple(steps)

    def execute(
        self,
        plan: SchedulerPlan,
        requests: Mapping[str, GenerationRequest],
        mtp_policy: MtpPolicy,
    ) -> FusedStepResult:
        self._require_owner_thread()
        self._require_open()
        steps = self.translate_plan(plan, requests, mtp_policy)
        receipts: list[tuple[LowLevelBatchStep, LowLevelBatchReceipt]] = []
        try:
            for step in steps:
                receipt = self._driver.run(step)
                self._validate_receipt(step, receipt)
                receipts.append((step, receipt))
                self._low_level_steps += 1
            return self._fused_result(plan, requests, receipts)
        except Exception:
            for request_id in plan.decode_request_ids:
                self._encoders.pop(request_id, None)
            raise

    def cleanup_cancelled(self, request_id: str, reason: str) -> None:
        if not request_id:
            raise ValueError("request_id must not be empty")
        if not reason:
            raise ValueError("cleanup reason must not be empty")
        self._require_owner_thread()
        self._require_open()
        first_reason = self._cleanup_reasons.setdefault(request_id, reason)
        if first_reason != reason:
            raise OmlxBatchCleanupError(
                "cleanup reason differs from the first cancellation reason"
            )
        if request_id in self._cleaned:
            return
        receipt = self._driver.cleanup_cancelled(request_id, reason)
        if receipt.request_id != request_id or receipt.reason != reason:
            raise OmlxBatchCleanupError(
                "cleanup receipt identity does not match the cancellation"
            )
        if not receipt.succeeded:
            detail = "; ".join(receipt.errors) or "cleanup was not acknowledged"
            raise OmlxBatchCleanupError(detail)
        self._encoders.pop(request_id, None)
        self._cleaned.add(request_id)

    def stats(self) -> Mapping[str, Any]:
        self._require_owner_thread()
        return {
            "low_level_steps": self._low_level_steps,
            "active_output_encoders": len(self._encoders),
            "cleanup_acknowledged": len(self._cleaned),
            "mtp_fallback_counts": {
                reason.value: count
                for reason, count in sorted(
                    self._fallbacks.items(),
                    key=lambda item: item[0].value,
                )
            },
            "driver": dict(self._driver.stats()),
        }

    def close(self, deadline_s: float) -> None:
        if deadline_s < 0:
            raise ValueError("deadline_s must be non-negative")
        self._require_owner_thread()
        if self._closed:
            return
        if self._encoders:
            raise OmlxBatchStepperError(
                "cannot close stepper with active output encoders"
            )
        self._driver.close(deadline_s)
        self._closed = True

    def _fused_result(
        self,
        plan: SchedulerPlan,
        requests: Mapping[str, GenerationRequest],
        receipts: list[tuple[LowLevelBatchStep, LowLevelBatchReceipt]],
    ) -> FusedStepResult:
        prefill_results: dict[str, PrefillResult] = {}
        decode_results: dict[str, DecodeResult] = {}
        events: dict[str, list[TurnEvent]] = {}
        prefill_elapsed_s = 0.0
        decode_elapsed_s = 0.0
        ar_decode_steps = 0
        ar_decode_tokens = 0
        mtp_rounds = 0
        mtp_drafted_tokens = 0
        mtp_accepted_tokens = 0
        mtp_rejected_tokens = 0
        fallbacks: list[MtpDisableReason] = []

        for step, receipt in receipts:
            if step.kind in (
                LowLevelBatchKind.TEXT_PREFILL,
                LowLevelBatchKind.VISION_PREFILL,
            ):
                prefill_elapsed_s += receipt.elapsed_s
                for observed in receipt.rows:
                    prefill_results[observed.request_id] = PrefillResult(
                        request_id=observed.request_id,
                        position=observed.position,
                        complete=observed.prefill_complete,
                        failed_reason=observed.failed_reason,
                    )
                continue

            decode_elapsed_s += receipt.elapsed_s
            ar_decode_steps += receipt.ar_decode_steps
            ar_decode_tokens += receipt.ar_decode_tokens
            mtp_rounds += receipt.mtp_rounds
            mtp_drafted_tokens += receipt.mtp_drafted_tokens
            mtp_accepted_tokens += receipt.mtp_accepted_tokens
            mtp_rejected_tokens += receipt.mtp_rejected_tokens
            decision = step.mtp_decision
            if decision is not None and not decision.enabled:
                if decision.disable_reason is None:
                    raise OmlxBatchContractError(
                        "disabled MTP decision has no fallback reason"
                    )
                fallbacks.append(decision.disable_reason)
                self._fallbacks[decision.disable_reason] += 1

            for observed in receipt.rows:
                request_id = observed.request_id
                row_events = events.setdefault(request_id, [])
                if observed.chunks or observed.finished:
                    encoder = self._encoders.get(request_id)
                    if encoder is None:
                        encoder = self._output_factory.create(requests[request_id])
                        self._encoders[request_id] = encoder
                    for chunk in observed.chunks:
                        row_events.extend(encoder.feed(chunk))
                    if observed.finished:
                        row_events.extend(
                            encoder.finish(
                                observed.final_usage,
                                finish_reason=observed.finish_reason or "stop",
                            )
                        )
                        self._encoders.pop(request_id, None)
                if observed.failed_reason is not None:
                    self._encoders.pop(request_id, None)
                decode_results[request_id] = DecodeResult(
                    request_id=request_id,
                    position=observed.position,
                    finished=observed.finished,
                    finish_reason=observed.finish_reason,
                    failed_reason=observed.failed_reason,
                )

        return FusedStepResult(
            prefill_results=tuple(
                prefill_results[row.request_id] for row in plan.prefill_rows
            ),
            decode_results=tuple(
                decode_results[row.request_id] for row in plan.decode_rows
            ),
            events={key: tuple(value) for key, value in events.items()},
            prefill_elapsed_s=prefill_elapsed_s,
            decode_elapsed_s=decode_elapsed_s,
            ar_decode_steps=ar_decode_steps,
            ar_decode_tokens=ar_decode_tokens,
            mtp_rounds=mtp_rounds,
            mtp_drafted_tokens=mtp_drafted_tokens,
            mtp_accepted_tokens=mtp_accepted_tokens,
            mtp_rejected_tokens=mtp_rejected_tokens,
            mtp_fallbacks=tuple(fallbacks),
        )

    def _mtp_decision(
        self,
        plan: SchedulerPlan,
        rows: tuple[LowLevelBatchRow, ...],
        policy: MtpPolicy,
    ) -> MtpDecision:
        alignment = MtpAlignment(
            runtime_keys=tuple(_runtime_identity(row.request.runtime) for row in rows),
            cache_positions=tuple(row.position for row in rows),
            pending_prompt_work=bool(plan.prefill_rows),
        )
        grammar_constrained = any(_is_grammar_constrained(row.request) for row in rows)
        return policy.decide(
            alignment=alignment,
            model_supported=self._mtp_facts.model_supported,
            head_attached=self._mtp_facts.head_attached,
            decode_enabled=self._mtp_facts.decode_enabled,
            verifier_available=self._mtp_facts.verifier_available,
            grammar_constrained=grammar_constrained,
        )

    @staticmethod
    def _translate_row(
        row: ScheduledRequest,
        request: GenerationRequest,
    ) -> LowLevelBatchRow:
        return LowLevelBatchRow(
            request_id=row.request_id,
            request=request,
            kind=row.kind,
            position=row.position,
        )

    @staticmethod
    def _validate_plan_requests(
        plan: SchedulerPlan,
        requests: Mapping[str, GenerationRequest],
    ) -> None:
        active_rows = plan.prefill_rows + plan.decode_rows
        expected = tuple(row.request_id for row in active_rows)
        if len(expected) != len(set(expected)):
            raise OmlxBatchContractError(
                "scheduler plan repeats a request across prefill and decode"
            )
        if set(requests) != set(expected):
            raise OmlxBatchContractError(
                "GenerationRequest mapping does not match scheduler rows"
            )
        cancelled = {item.request_id for item in plan.cancelled_requests}
        if cancelled.intersection(expected):
            raise OmlxBatchContractError(
                "cancelled request cannot remain in an executable scheduler row"
            )
        for row in active_rows:
            request = requests[row.request_id]
            if not isinstance(request, GenerationRequest):
                raise TypeError("request mapping values must be GenerationRequest")
            if request.response_id != row.request_id:
                raise OmlxBatchContractError(
                    "scheduler request_id does not match GenerationRequest"
                )
            expected_kind = WorkKind.VISION if request.media else WorkKind.TEXT
            if row.kind is not expected_kind:
                raise OmlxBatchContractError(
                    "scheduler work kind does not match GenerationRequest media"
                )

    @staticmethod
    def _validate_receipt(
        step: LowLevelBatchStep,
        receipt: LowLevelBatchReceipt,
    ) -> None:
        if not isinstance(receipt, LowLevelBatchReceipt):
            raise OmlxBatchContractError(
                "low-level driver must return LowLevelBatchReceipt"
            )
        if receipt.scheduler_step_id != step.scheduler_step_id:
            raise OmlxBatchContractError("receipt has a foreign scheduler step id")
        if receipt.kind is not step.kind:
            raise OmlxBatchContractError("receipt has a different batch kind")
        expected = tuple(row.request_id for row in step.rows)
        observed = tuple(row.request_id for row in receipt.rows)
        if observed != expected or len(observed) != len(set(observed)):
            raise OmlxBatchContractError(
                "receipt rows do not exactly match the low-level batch lease"
            )
        scheduled = {row.request_id: row for row in step.rows}
        for row in receipt.rows:
            if row.position < scheduled[row.request_id].position:
                raise OmlxBatchContractError("row position moved backwards")

        prefill = step.kind in (
            LowLevelBatchKind.TEXT_PREFILL,
            LowLevelBatchKind.VISION_PREFILL,
        )
        if prefill:
            if any(
                row.finished or row.chunks or row.final_usage for row in receipt.rows
            ):
                raise OmlxBatchContractError(
                    "prefill receipt cannot contain decode output"
                )
            if any(
                (
                    receipt.ar_decode_steps,
                    receipt.ar_decode_tokens,
                    receipt.mtp_rounds,
                    receipt.mtp_drafted_tokens,
                    receipt.mtp_accepted_tokens,
                    receipt.mtp_rejected_tokens,
                )
            ):
                raise OmlxBatchContractError(
                    "prefill receipt cannot claim decode accounting"
                )
            return

        if any(row.prefill_complete for row in receipt.rows):
            raise OmlxBatchContractError("decode receipt cannot complete prefill")
        if step.kind is LowLevelBatchKind.AR_DECODE:
            if any(
                (
                    receipt.mtp_rounds,
                    receipt.mtp_drafted_tokens,
                    receipt.mtp_accepted_tokens,
                    receipt.mtp_rejected_tokens,
                )
            ):
                raise OmlxBatchContractError("AR receipt cannot claim MTP work")
            emitted = sum(row.emitted_tokens for row in receipt.rows)
            if emitted != receipt.ar_decode_tokens:
                raise OmlxBatchContractError(
                    "AR token accounting does not match observed row emissions"
                )
        elif receipt.ar_decode_steps or receipt.ar_decode_tokens:
            raise OmlxBatchContractError("MTP receipt cannot claim AR work")

    def _require_open(self) -> None:
        if self._closed:
            raise OmlxBatchStepperError("batch stepper is closed")

    def _require_owner_thread(self) -> None:
        if threading.get_ident() != self._owner_thread_id:
            raise OmlxBatchStepperError(
                "batch stepper must run on its tensor-owner thread"
            )


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


def _is_grammar_constrained(request: GenerationRequest) -> bool:
    grammar_keys = {
        "compiled_grammar",
        "grammar",
        "guided_json",
        "response_format",
    }
    return any(request.sampling.get(key) is not None for key in grammar_keys)


__all__ = [
    "CancellationCleanupReceipt",
    "CleanupDisposition",
    "LowLevelBatchKind",
    "LowLevelBatchReceipt",
    "LowLevelBatchRow",
    "LowLevelBatchStep",
    "LowLevelRowResult",
    "MtpRuntimeFacts",
    "OmlxBatchCleanupError",
    "OmlxBatchContractError",
    "OmlxBatchKernelPort",
    "OmlxBatchStepper",
    "OmlxBatchStepperError",
]
