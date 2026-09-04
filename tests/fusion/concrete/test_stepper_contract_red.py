"""RED contracts for the tensor-free oMLX/MTPLX batch stepper.

These tests are authored but must not be executed while Compile Embargo is
HOLD.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from mlx_batch_server.runtime.contracts import (
    BackendKind,
    GenerationRequest,
    ModelSpec,
    RuntimeKey,
)
from mlx_batch_server.runtime.events import (
    ContentPartCompleted,
    OutputItemCompleted,
    TextCompleted,
    TextDelta,
    UsageUpdate,
)
from mlx_batch_server.runtime.fusion.concrete.stepper import (
    CancellationCleanupReceipt,
    CleanupDisposition,
    LowLevelBatchKind,
    LowLevelBatchReceipt,
    LowLevelBatchStep,
    LowLevelRowResult,
    MtpRuntimeFacts,
    OmlxBatchCleanupError,
    OmlxBatchContractError,
    OmlxBatchStepper,
)
from mlx_batch_server.runtime.fusion.mtp import (
    MtpDisableReason,
    MtpMode,
    MtpPolicy,
)
from mlx_batch_server.runtime.fusion.output import Qwen4OutputChunk, Qwen4OutputError
from mlx_batch_server.runtime.fusion.scheduler import (
    ScheduledRequest,
    SchedulerPlan,
    WorkKind,
)

RUNTIME = RuntimeKey(
    model_id="grant-ai/Qwen3.8-Flash-Next-Abliterated-MLX-4bit",
    revision="000544f8cddcbde27c1bc302deac2b5b4d45a5b1",
    backend=BackendKind.FUSED_MTP_MLX,
)
MODEL = ModelSpec(
    model_id=RUNTIME.model_id,
    revision=RUNTIME.revision,
    architecture="Qwen4ExpForConditionalGeneration",
    model_type="qwen4_exp",
)
PROVEN_MTP = MtpRuntimeFacts(
    model_supported=True,
    head_attached=True,
    decode_enabled=True,
    verifier_available=True,
)


def _request(response_id: str, *, vision: bool = False) -> GenerationRequest:
    return GenerationRequest(
        response_id=response_id,
        runtime=RUNTIME,
        messages=({"role": "user", "content": "diagnose"},),
        media=({"type": "input_image", "image_url": "file:///patient.png"},)
        if vision
        else (),
    )


def _clean_receipt(request_id: str, reason: str) -> CancellationCleanupReceipt:
    return CancellationCleanupReceipt(
        request_id=request_id,
        reason=reason,
        execution_stopped=True,
        uid=CleanupDisposition.RELEASED,
        kv=CleanupDisposition.RELEASED,
        parser=CleanupDisposition.RELEASED,
        media=CleanupDisposition.NOT_ALLOCATED,
    )


class RecordingDriver:
    def __init__(self) -> None:
        self.steps: list[LowLevelBatchStep] = []
        self.cleanup_calls: list[tuple[str, str]] = []
        self.cleanup_receipt: CancellationCleanupReceipt | None = None
        self.closed_with: list[float] = []

    def run(self, step: LowLevelBatchStep) -> LowLevelBatchReceipt:
        self.steps.append(step)
        if step.kind in (
            LowLevelBatchKind.TEXT_PREFILL,
            LowLevelBatchKind.VISION_PREFILL,
        ):
            return LowLevelBatchReceipt(
                scheduler_step_id=step.scheduler_step_id,
                kind=step.kind,
                rows=tuple(
                    LowLevelRowResult(
                        request_id=row.request_id,
                        position=row.position + 8,
                        prefill_complete=True,
                    )
                    for row in step.rows
                ),
                elapsed_s=0.2,
            )
        if step.kind is LowLevelBatchKind.AR_DECODE:
            return LowLevelBatchReceipt(
                scheduler_step_id=step.scheduler_step_id,
                kind=step.kind,
                rows=tuple(
                    LowLevelRowResult(
                        request_id=row.request_id,
                        position=row.position + 7,
                        emitted_tokens=1,
                        finished=True,
                        finish_reason="stop",
                        chunks=(Qwen4OutputChunk(text_delta="done"),),
                        final_usage=UsageUpdate(8, 1, 9),
                    )
                    for row in step.rows
                ),
                elapsed_s=0.1,
                ar_decode_steps=1,
                ar_decode_tokens=len(step.rows),
            )
        return LowLevelBatchReceipt(
            scheduler_step_id=step.scheduler_step_id,
            kind=step.kind,
            rows=tuple(
                LowLevelRowResult(
                    request_id=row.request_id,
                    position=row.position + 3,
                    emitted_tokens=3,
                )
                for row in step.rows
            ),
            elapsed_s=0.05,
            mtp_rounds=1,
            mtp_drafted_tokens=4,
            mtp_accepted_tokens=2,
            mtp_rejected_tokens=1,
        )

    def cleanup_cancelled(
        self,
        request_id: str,
        reason: str,
    ) -> CancellationCleanupReceipt:
        self.cleanup_calls.append((request_id, reason))
        return self.cleanup_receipt or _clean_receipt(request_id, reason)

    def stats(self) -> Mapping[str, Any]:
        return {"driver_steps": len(self.steps)}

    def close(self, deadline_s: float) -> None:
        self.closed_with.append(deadline_s)


class MalformedOutputDriver(RecordingDriver):
    def run(self, step: LowLevelBatchStep) -> LowLevelBatchReceipt:
        self.steps.append(step)
        row = step.rows[0]
        return LowLevelBatchReceipt(
            scheduler_step_id=step.scheduler_step_id,
            kind=step.kind,
            rows=(
                LowLevelRowResult(
                    request_id=row.request_id,
                    position=row.position + 1,
                    emitted_tokens=1,
                    finished=True,
                    finish_reason="stop",
                    chunks=(
                        Qwen4OutputChunk(
                            text_delta='<tool_call>{"name":"lookup","arguments":{'
                        ),
                    ),
                ),
            ),
            elapsed_s=0.01,
            ar_decode_steps=1,
            ar_decode_tokens=1,
        )


def _stepper(
    driver: RecordingDriver | None = None,
    *,
    facts: MtpRuntimeFacts | None = None,
) -> tuple[OmlxBatchStepper, RecordingDriver]:
    selected = driver or RecordingDriver()
    return (
        OmlxBatchStepper(model_spec=MODEL, driver=selected, mtp_facts=facts),
        selected,
    )


def test_translation_separates_text_and_vision_prefill_from_common_decode() -> None:
    stepper, _ = _stepper()
    requests = {
        "text-prefill": _request("text-prefill"),
        "vision-prefill": _request("vision-prefill", vision=True),
        "text-decode": _request("text-decode"),
        "vision-decode": _request("vision-decode", vision=True),
    }
    plan = SchedulerPlan(
        step_id=7,
        prefill_rows=(
            ScheduledRequest("text-prefill", WorkKind.TEXT, 0),
            ScheduledRequest("vision-prefill", WorkKind.VISION, 0),
        ),
        decode_rows=(
            ScheduledRequest("text-decode", WorkKind.TEXT, 32),
            ScheduledRequest("vision-decode", WorkKind.VISION, 32),
        ),
    )

    steps = stepper.translate_plan(plan, requests, MtpPolicy())

    assert tuple(step.kind for step in steps) == (
        LowLevelBatchKind.TEXT_PREFILL,
        LowLevelBatchKind.VISION_PREFILL,
        LowLevelBatchKind.AR_DECODE,
    )
    assert tuple(row.request_id for row in steps[0].rows) == ("text-prefill",)
    assert tuple(row.request_id for row in steps[1].rows) == ("vision-prefill",)
    assert tuple(row.request_id for row in steps[2].rows) == (
        "text-decode",
        "vision-decode",
    )
    assert steps[2].mtp_decision is not None
    assert steps[2].mtp_decision.disable_reason is MtpDisableReason.MODEL_UNSUPPORTED
    assert steps[0].rows[0].request is requests["text-prefill"]


def test_only_live_proven_aligned_cohort_enters_exact_mtp_decode() -> None:
    stepper, _ = _stepper(facts=PROVEN_MTP)
    requests = {name: _request(name) for name in ("first", "second")}
    policy = MtpPolicy(allow_proven_multirow=True, max_proven_rows=2)
    aligned = SchedulerPlan(
        step_id=1,
        decode_rows=(
            ScheduledRequest("first", WorkKind.TEXT, 64),
            ScheduledRequest("second", WorkKind.TEXT, 64),
        ),
    )

    step = stepper.translate_plan(aligned, requests, policy)[0]

    assert step.kind is LowLevelBatchKind.MTP_DECODE
    assert step.mtp_decision is not None
    assert step.mtp_decision.enabled is True
    assert step.mtp_decision.exact is True
    assert step.mtp_decision.mode is MtpMode.EXACT_ALIGNED_COHORT


def test_aligned_multirow_cohort_stays_ar_without_explicit_multirow_proof() -> None:
    stepper, _ = _stepper(facts=PROVEN_MTP)
    requests = {name: _request(name) for name in ("first", "second")}
    plan = SchedulerPlan(
        step_id=2,
        decode_rows=(
            ScheduledRequest("first", WorkKind.TEXT, 64),
            ScheduledRequest("second", WorkKind.TEXT, 64),
        ),
    )

    decode = stepper.translate_plan(plan, requests, MtpPolicy())[0]

    assert decode.kind is LowLevelBatchKind.AR_DECODE
    assert decode.mtp_decision is not None
    assert decode.mtp_decision.disable_reason is MtpDisableReason.MULTIROW_NOT_PROVEN


@pytest.mark.parametrize(
    ("plan", "expected_reason"),
    [
        (
            SchedulerPlan(
                step_id=2,
                decode_rows=(
                    ScheduledRequest("first", WorkKind.TEXT, 64),
                    ScheduledRequest("second", WorkKind.TEXT, 65),
                ),
            ),
            MtpDisableReason.ROWS_UNALIGNED,
        ),
        (
            SchedulerPlan(
                step_id=3,
                prefill_rows=(ScheduledRequest("joining", WorkKind.TEXT, 0),),
                decode_rows=(ScheduledRequest("first", WorkKind.TEXT, 64),),
            ),
            MtpDisableReason.PROMPT_MERGE_PENDING,
        ),
    ],
)
def test_unaligned_or_late_join_decode_falls_back_to_ar_with_exact_reason(
    plan: SchedulerPlan,
    expected_reason: MtpDisableReason,
) -> None:
    stepper, _ = _stepper(facts=PROVEN_MTP)
    requests = {
        request_id: _request(request_id)
        for request_id in plan.prefill_request_ids + plan.decode_request_ids
    }
    policy = MtpPolicy(allow_proven_multirow=True, max_proven_rows=2)

    decode = stepper.translate_plan(plan, requests, policy)[-1]

    assert decode.kind is LowLevelBatchKind.AR_DECODE
    assert decode.mtp_decision is not None
    assert decode.mtp_decision.disable_reason is expected_reason


def test_fused_result_uses_only_observed_driver_accounting_and_events() -> None:
    stepper, driver = _stepper()
    requests = {
        "joining": _request("joining", vision=True),
        "running": _request("running"),
    }
    plan = SchedulerPlan(
        step_id=9,
        prefill_rows=(ScheduledRequest("joining", WorkKind.VISION, 0),),
        decode_rows=(ScheduledRequest("running", WorkKind.TEXT, 40),),
    )

    result = stepper.execute(plan, requests, MtpPolicy())

    assert tuple(step.kind for step in driver.steps) == (
        LowLevelBatchKind.VISION_PREFILL,
        LowLevelBatchKind.AR_DECODE,
    )
    assert result.prefill_results[0].position == 8
    assert result.decode_results[0].position == 47
    assert result.ar_decode_steps == 1
    assert result.ar_decode_tokens == 1
    assert result.mtp_rounds == 0
    assert result.mtp_fallbacks == (MtpDisableReason.MODEL_UNSUPPORTED,)
    emitted = result.events["running"]
    assert any(isinstance(event, TextDelta) for event in emitted)
    assert any(isinstance(event, TextCompleted) for event in emitted)
    assert any(isinstance(event, ContentPartCompleted) for event in emitted)
    assert any(isinstance(event, OutputItemCompleted) for event in emitted)
    assert emitted[-1] == UsageUpdate(8, 1, 9)
    assert stepper.stats()["mtp_fallback_counts"] == {"model_unsupported": 1}


def test_mtp_accounting_is_copied_from_observation_not_inferred() -> None:
    stepper, _ = _stepper(facts=PROVEN_MTP)
    requests = {"one": _request("one")}
    plan = SchedulerPlan(
        step_id=4,
        decode_rows=(ScheduledRequest("one", WorkKind.TEXT, 100),),
    )

    result = stepper.execute(plan, requests, MtpPolicy())

    assert result.ar_decode_steps == 0
    assert result.ar_decode_tokens == 0
    assert result.mtp_rounds == 1
    assert result.mtp_drafted_tokens == 4
    assert result.mtp_accepted_tokens == 2
    assert result.mtp_rejected_tokens == 1
    assert result.decode_results[0].position == 103
    assert result.mtp_fallbacks == ()


def test_output_contract_failure_discards_request_parser_state() -> None:
    driver = MalformedOutputDriver()
    stepper, _ = _stepper(driver)
    requests = {"broken": _request("broken")}
    plan = SchedulerPlan(
        step_id=5,
        decode_rows=(ScheduledRequest("broken", WorkKind.TEXT, 4),),
    )

    with pytest.raises(Qwen4OutputError, match="unterminated"):
        stepper.execute(plan, requests, MtpPolicy())

    assert stepper.stats()["active_output_encoders"] == 0


def test_cancellation_requires_exact_full_cleanup_ack_and_is_idempotent() -> None:
    stepper, driver = _stepper()

    stepper.cleanup_cancelled("resp_cancel", "client disconnected")
    stepper.cleanup_cancelled("resp_cancel", "client disconnected")

    assert driver.cleanup_calls == [("resp_cancel", "client disconnected")]
    with pytest.raises(OmlxBatchCleanupError, match="first cancellation reason"):
        stepper.cleanup_cancelled("resp_cancel", "different reason")


def test_failed_or_foreign_cleanup_receipt_never_counts_as_acknowledged() -> None:
    driver = RecordingDriver()
    stepper, _ = _stepper(driver)
    driver.cleanup_receipt = CancellationCleanupReceipt(
        request_id="resp_cancel",
        reason="stop",
        execution_stopped=False,
        uid=CleanupDisposition.RELEASED,
        kv=CleanupDisposition.FAILED,
        parser=CleanupDisposition.RELEASED,
        media=CleanupDisposition.NOT_ALLOCATED,
        errors=("KV refs remain",),
    )

    with pytest.raises(OmlxBatchCleanupError, match="KV refs remain"):
        stepper.cleanup_cancelled("resp_cancel", "stop")
    assert stepper.stats()["cleanup_acknowledged"] == 0

    driver.cleanup_receipt = _clean_receipt("foreign", "stop")
    with pytest.raises(OmlxBatchCleanupError, match="identity"):
        stepper.cleanup_cancelled("resp_cancel", "stop")
    assert driver.cleanup_calls == [
        ("resp_cancel", "stop"),
        ("resp_cancel", "stop"),
    ]


def test_request_mapping_and_work_kind_fail_closed() -> None:
    stepper, _ = _stepper()
    plan = SchedulerPlan(
        step_id=1,
        prefill_rows=(ScheduledRequest("vision", WorkKind.VISION, 0),),
    )

    with pytest.raises(OmlxBatchContractError, match="does not match"):
        stepper.translate_plan(plan, {}, MtpPolicy())
    with pytest.raises(OmlxBatchContractError, match="work kind"):
        stepper.translate_plan(
            plan,
            {"vision": _request("vision", vision=False)},
            MtpPolicy(),
        )
