"""RED contracts for the source-only fused scheduler chassis.

These tests are intentionally authored but not executed while the Compile
Embargo is HOLD.
"""

import pytest

from mlx_batch_server.runtime.fusion.scheduler import (
    CancelDisposition,
    DecodeResult,
    PrefillResult,
    RequestPhase,
    SchedulerChassis,
    SchedulerConfig,
    SchedulerRequest,
    SubmitDisposition,
    WorkKind,
)


def _complete_prefill(
    scheduler: SchedulerChassis,
    request_id: str,
    *,
    position: int = 16,
) -> None:
    plan = scheduler.next_plan()
    assert plan.prefill_request_ids == (request_id,)
    scheduler.complete_step(
        plan,
        prefill_results=(PrefillResult(request_id, position, complete=True),),
    )


def test_admission_is_bounded_and_duplicate_ids_are_rejected() -> None:
    scheduler = SchedulerChassis(
        SchedulerConfig(max_admitted_requests=2, max_decode_rows=2)
    )

    assert (
        scheduler.submit(SchedulerRequest("one")).disposition
        is SubmitDisposition.ACCEPTED
    )
    assert (
        scheduler.submit(SchedulerRequest("one")).disposition
        is SubmitDisposition.DUPLICATE
    )
    assert (
        scheduler.submit(SchedulerRequest("two")).disposition
        is SubmitDisposition.ACCEPTED
    )
    assert (
        scheduler.submit(SchedulerRequest("three")).disposition
        is SubmitDisposition.CAPACITY
    )
    assert scheduler.snapshot().admitted_requests == 2


def test_late_vision_join_prefills_then_enters_common_decode_batch() -> None:
    scheduler = SchedulerChassis(
        SchedulerConfig(
            max_admitted_requests=4,
            max_decode_rows=3,
            max_prefill_rows=1,
            max_vision_prefills=1,
        )
    )
    scheduler.submit(SchedulerRequest("text"))
    _complete_prefill(scheduler, "text")

    first_decode = scheduler.next_plan()
    assert first_decode.decode_request_ids == ("text",)
    scheduler.complete_step(
        first_decode,
        decode_results=(DecodeResult("text", 17),),
        decode_elapsed_s=0.1,
    )

    scheduler.submit(SchedulerRequest("vision", WorkKind.VISION))
    joined = scheduler.next_plan()
    assert joined.prefill_request_ids == ("vision",)
    assert joined.decode_request_ids == ("text",)
    assert joined.prefill_rows[0].kind is WorkKind.VISION
    assert joined.prefill_rows[0].position == 0
    scheduler.complete_step(
        joined,
        prefill_results=(PrefillResult("vision", 32, complete=True),),
        decode_results=(DecodeResult("text", 18),),
        prefill_elapsed_s=0.1,
        decode_elapsed_s=0.1,
    )

    common_decode = scheduler.next_plan()
    assert common_decode.decode_request_ids == ("text", "vision")


def test_vision_prefill_parallelism_has_an_independent_bound() -> None:
    scheduler = SchedulerChassis(
        SchedulerConfig(
            max_admitted_requests=4,
            max_decode_rows=4,
            max_prefill_rows=3,
            max_vision_prefills=2,
        )
    )
    for request_id in ("v1", "v2", "v3"):
        scheduler.submit(SchedulerRequest(request_id, WorkKind.VISION))

    plan = scheduler.next_plan()
    assert plan.prefill_request_ids == ("v1", "v2")
    assert scheduler.snapshot().waiting_requests == 1


def test_prefill_debt_forces_decode_progress_before_another_late_join() -> None:
    scheduler = SchedulerChassis(
        SchedulerConfig(
            max_admitted_requests=4,
            max_decode_rows=3,
            max_prefill_rows=1,
            max_vision_prefills=1,
            decode_fair_share=0.5,
        )
    )
    scheduler.submit(SchedulerRequest("running"))
    _complete_prefill(scheduler, "running")
    scheduler.submit(SchedulerRequest("first-join"))

    contended = scheduler.next_plan()
    scheduler.complete_step(
        contended,
        prefill_results=(PrefillResult("first-join", 8, complete=True),),
        decode_results=(DecodeResult("running", 17),),
        prefill_elapsed_s=0.6,
        decode_elapsed_s=0.1,
    )
    assert scheduler.snapshot().decode_time_owed_s == pytest.approx(0.2)

    scheduler.submit(SchedulerRequest("second-join"))
    repayment = scheduler.next_plan()
    assert repayment.prefill_request_ids == ()
    assert repayment.decode_request_ids == ("running", "first-join")
    scheduler.complete_step(
        repayment,
        decode_results=(
            DecodeResult("running", 18),
            DecodeResult("first-join", 9),
        ),
        decode_elapsed_s=0.2,
    )

    admitted = scheduler.next_plan()
    assert admitted.prefill_request_ids == ("second-join",)


def test_cancel_is_idempotent_and_reports_cleanup_phase() -> None:
    scheduler = SchedulerChassis(
        SchedulerConfig(
            max_admitted_requests=3,
            max_decode_rows=2,
            max_prefill_rows=1,
            max_vision_prefills=1,
        )
    )
    scheduler.submit(SchedulerRequest("prefilling", WorkKind.VISION))
    scheduler.submit(SchedulerRequest("waiting"))
    prefill = scheduler.next_plan()
    scheduler.complete_step(
        prefill,
        prefill_results=(PrefillResult("prefilling", 4, complete=False),),
    )

    requested = scheduler.request_cancel("prefilling", "client disconnected")
    repeated = scheduler.request_cancel("prefilling", "different reason")
    scheduler.request_cancel("waiting", "response cancelled")
    assert requested.disposition is CancelDisposition.REQUESTED
    assert repeated.disposition is CancelDisposition.ALREADY_REQUESTED
    assert repeated.reason == "client disconnected"

    cleanup = scheduler.next_plan()
    assert {
        (item.request_id, item.previous_phase, item.reason)
        for item in cleanup.cancelled_requests
    } == {
        ("prefilling", RequestPhase.PREFILL, "client disconnected"),
        ("waiting", RequestPhase.WAITING, "response cancelled"),
    }
    assert scheduler.snapshot().admitted_requests == 0
    assert (
        scheduler.request_cancel("prefilling", "late cancel").disposition
        is CancelDisposition.ALREADY_TERMINAL
    )


def test_cancelled_decode_frees_capacity_for_oldest_waiter() -> None:
    scheduler = SchedulerChassis(
        SchedulerConfig(
            max_admitted_requests=3,
            max_decode_rows=1,
            max_prefill_rows=1,
            max_vision_prefills=1,
        )
    )
    scheduler.submit(SchedulerRequest("active"))
    _complete_prefill(scheduler, "active")
    scheduler.submit(SchedulerRequest("oldest"))
    scheduler.submit(SchedulerRequest("newest"))

    scheduler.request_cancel("active", "stop")
    after_cancel = scheduler.next_plan()
    assert after_cancel.cancelled_requests[0].previous_phase is RequestPhase.DECODE
    assert after_cancel.prefill_request_ids == ("oldest",)


def test_completed_decode_frees_admission_capacity() -> None:
    scheduler = SchedulerChassis(
        SchedulerConfig(
            max_admitted_requests=1,
            max_decode_rows=1,
            max_prefill_rows=1,
            max_vision_prefills=1,
        )
    )
    scheduler.submit(SchedulerRequest("done"))
    _complete_prefill(scheduler, "done")
    plan = scheduler.next_plan()
    update = scheduler.complete_step(
        plan,
        decode_results=(DecodeResult("done", 17, finished=True, finish_reason="stop"),),
    )

    assert update.terminal_requests[0].phase is RequestPhase.COMPLETED
    assert (
        scheduler.submit(SchedulerRequest("next")).disposition
        is SubmitDisposition.ACCEPTED
    )


def test_plan_must_be_completed_before_next_step() -> None:
    scheduler = SchedulerChassis()
    scheduler.submit(SchedulerRequest("request"))
    plan = scheduler.next_plan()

    with pytest.raises(RuntimeError, match="still in flight"):
        scheduler.next_plan()
    with pytest.raises(ValueError, match="do not match"):
        scheduler.complete_step(plan)


def test_foreign_plan_with_same_step_id_cannot_commit_state() -> None:
    scheduler = SchedulerChassis()
    scheduler.submit(SchedulerRequest("request"))
    plan = scheduler.next_plan()
    foreign = type(plan)(step_id=plan.step_id)

    with pytest.raises(RuntimeError, match="stale or foreign"):
        scheduler.complete_step(foreign)
