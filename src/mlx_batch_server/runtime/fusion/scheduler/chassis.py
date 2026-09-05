"""Small target-owned scheduler chassis for text and VLM rows.

Adapted from the request queues, deferred abort boundary, bounded admission,
late-join flow, and prefill/decode fairness in oMLX ``omlx/scheduler.py`` at
``e467261edc786efd33b1e9023d5c4a827f8aa1c1`` (Apache-2.0). Modified by
LibraxisAI into a dependency-free planner. Backend adapters remain responsible
for MLX execution and cache cleanup; ``GenerationTurn`` remains the sole
terminal/event writer.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock

from .contracts import (
    CancelDisposition,
    CancelledRequest,
    CancelResult,
    DecodeResult,
    PrefillResult,
    RequestPhase,
    RequestSnapshot,
    ScheduledRequest,
    SchedulerConfig,
    SchedulerPlan,
    SchedulerRequest,
    SchedulerSnapshot,
    SchedulerUpdate,
    SubmitDisposition,
    SubmitResult,
    TerminalRequest,
    WorkKind,
)


@dataclass(slots=True)
class _RequestState:
    request: SchedulerRequest
    arrival_sequence: int
    phase: RequestPhase
    position: int


class SchedulerChassis:
    """Plan scheduler rows while keeping model execution outside this module.

    All mutations are serialized by one lock so ``request_cancel`` may be
    called from the event-loop thread while planning and completion happen on
    the inference owner thread. A plan is a lease: callers must complete it
    before requesting another plan.
    """

    def __init__(self, config: SchedulerConfig | None = None) -> None:
        self._config = config or SchedulerConfig()
        self._lock = RLock()
        self._states: dict[str, _RequestState] = {}
        self._waiting: list[str] = []
        self._pending_cancels: OrderedDict[str, str] = OrderedDict()
        self._terminal: OrderedDict[str, TerminalRequest] = OrderedDict()
        self._arrival_sequence = 0
        self._step_id = 0
        self._decode_time_owed_s = 0.0
        self._inflight_plan: SchedulerPlan | None = None

    @property
    def config(self) -> SchedulerConfig:
        return self._config

    def submit(self, request: SchedulerRequest) -> SubmitResult:
        """Admit one backend row after the target admission lease is held."""
        with self._lock:
            if (
                request.request_id in self._states
                or request.request_id in self._terminal
            ):
                return SubmitResult(
                    request.request_id,
                    SubmitDisposition.DUPLICATE,
                    "request_id is already known to this scheduler",
                )
            if len(self._states) >= self._config.max_admitted_requests:
                return SubmitResult(
                    request.request_id,
                    SubmitDisposition.CAPACITY,
                    "scheduler admission bound reached",
                )

            state = _RequestState(
                request=request,
                arrival_sequence=self._arrival_sequence,
                phase=RequestPhase.WAITING,
                position=request.initial_position,
            )
            self._arrival_sequence += 1
            self._states[request.request_id] = state
            self._waiting.append(request.request_id)
            return SubmitResult(
                request.request_id,
                SubmitDisposition.ACCEPTED,
                "request admitted to bounded scheduler queue",
            )

    def request_cancel(self, request_id: str, reason: str) -> CancelResult:
        """Request owner-thread cleanup at the next scheduler boundary."""
        if not reason:
            raise ValueError("cancel reason must not be empty")
        with self._lock:
            if request_id in self._pending_cancels:
                return CancelResult(
                    request_id,
                    CancelDisposition.ALREADY_REQUESTED,
                    self._pending_cancels[request_id],
                )
            if request_id in self._states:
                self._pending_cancels[request_id] = reason
                return CancelResult(
                    request_id,
                    CancelDisposition.REQUESTED,
                    reason,
                )
            terminal = self._terminal.get(request_id)
            if terminal is not None:
                return CancelResult(
                    request_id,
                    CancelDisposition.ALREADY_TERMINAL,
                    terminal.reason,
                )
            return CancelResult(
                request_id,
                CancelDisposition.NOT_FOUND,
                "request_id is unknown",
            )

    def next_plan(self) -> SchedulerPlan:
        """Drain cancels, admit late joiners, and lease one execution step."""
        with self._lock:
            if self._inflight_plan is not None:
                raise RuntimeError("the previous scheduler plan is still in flight")

            self._step_id += 1
            cancelled = self._drain_pending_cancels()
            decode_states = self._states_in_phase(RequestPhase.DECODE)
            if not decode_states:
                self._decode_time_owed_s = 0.0

            prefill_gate_open = not decode_states or self._decode_time_owed_s <= 0.0
            if prefill_gate_open:
                self._fill_prefill_slots(len(decode_states))

            prefill_rows = ()
            if prefill_gate_open:
                prefill_rows = tuple(
                    self._scheduled_row(state)
                    for state in self._states_in_phase(RequestPhase.PREFILL)
                )
            decode_rows = tuple(
                self._scheduled_row(state)
                for state in self._states_in_phase(RequestPhase.DECODE)
            )
            plan = SchedulerPlan(
                step_id=self._step_id,
                prefill_rows=prefill_rows,
                decode_rows=decode_rows,
                cancelled_requests=tuple(cancelled),
            )
            if plan.requires_completion:
                self._inflight_plan = plan
            return plan

    def complete_step(
        self,
        plan: SchedulerPlan,
        *,
        prefill_results: tuple[PrefillResult, ...] = (),
        decode_results: tuple[DecodeResult, ...] = (),
        prefill_elapsed_s: float = 0.0,
        decode_elapsed_s: float = 0.0,
    ) -> SchedulerUpdate:
        """Commit one leased plan after the backend finishes its MLX step."""
        with self._lock:
            if self._inflight_plan is None:
                raise RuntimeError("no scheduler plan is in flight")
            if plan != self._inflight_plan:
                raise RuntimeError("scheduler plan is stale or foreign")
            if prefill_elapsed_s < 0 or decode_elapsed_s < 0:
                raise ValueError("step durations must be non-negative")

            self._require_exact_results(
                plan.prefill_request_ids,
                tuple(result.request_id for result in prefill_results),
                "prefill",
            )
            self._require_exact_results(
                plan.decode_request_ids,
                tuple(result.request_id for result in decode_results),
                "decode",
            )

            prefill_states = tuple(
                self._validated_result_state(
                    result.request_id,
                    RequestPhase.PREFILL,
                    result.position,
                )
                for result in prefill_results
            )
            decode_states = tuple(
                self._validated_result_state(
                    result.request_id,
                    RequestPhase.DECODE,
                    result.position,
                )
                for result in decode_results
            )

            terminal: list[TerminalRequest] = []
            for state, prefill_result in zip(
                prefill_states, prefill_results, strict=True
            ):
                state.position = prefill_result.position
                if prefill_result.failed_reason is not None:
                    terminal.append(
                        self._finish(
                            state,
                            RequestPhase.FAILED,
                            prefill_result.failed_reason,
                        )
                    )
                elif prefill_result.complete:
                    state.phase = RequestPhase.DECODE

            for state, decode_result in zip(decode_states, decode_results, strict=True):
                state.position = decode_result.position
                if decode_result.failed_reason is not None:
                    terminal.append(
                        self._finish(
                            state,
                            RequestPhase.FAILED,
                            decode_result.failed_reason,
                        )
                    )
                elif decode_result.finished:
                    terminal.append(
                        self._finish(
                            state,
                            RequestPhase.COMPLETED,
                            decode_result.finish_reason or "completed",
                            stop_sequence=decode_result.stop_sequence,
                        )
                    )

            if plan.prefill_request_ids and plan.decode_request_ids:
                self._decode_time_owed_s += (
                    prefill_elapsed_s * self._config.decode_fair_share
                )
            self._decode_time_owed_s = max(
                0.0,
                self._decode_time_owed_s - decode_elapsed_s,
            )
            self._inflight_plan = None
            return SchedulerUpdate(
                step_id=plan.step_id,
                terminal_requests=tuple(terminal),
                decode_time_owed_s=self._decode_time_owed_s,
            )

    def snapshot(self) -> SchedulerSnapshot:
        with self._lock:
            states = sorted(
                self._states.values(), key=lambda item: item.arrival_sequence
            )
            requests = tuple(
                RequestSnapshot(
                    request_id=state.request.request_id,
                    kind=state.request.kind,
                    phase=state.phase,
                    position=state.position,
                    arrival_sequence=state.arrival_sequence,
                )
                for state in states
            )
            return SchedulerSnapshot(
                step_id=self._step_id,
                admitted_requests=len(states),
                waiting_requests=sum(
                    state.phase is RequestPhase.WAITING for state in states
                ),
                prefilling_requests=sum(
                    state.phase is RequestPhase.PREFILL for state in states
                ),
                decoding_requests=sum(
                    state.phase is RequestPhase.DECODE for state in states
                ),
                decode_time_owed_s=self._decode_time_owed_s,
                requests=requests,
                recent_terminal=tuple(self._terminal.values()),
            )

    def has_work(self) -> bool:
        with self._lock:
            return bool(
                self._states or self._pending_cancels or self._inflight_plan is not None
            )

    def _states_in_phase(self, phase: RequestPhase) -> list[_RequestState]:
        return sorted(
            (state for state in self._states.values() if state.phase is phase),
            key=lambda state: state.arrival_sequence,
        )

    @staticmethod
    def _scheduled_row(state: _RequestState) -> ScheduledRequest:
        return ScheduledRequest(
            request_id=state.request.request_id,
            kind=state.request.kind,
            position=state.position,
        )

    def _fill_prefill_slots(self, decode_count: int) -> None:
        prefilling = self._states_in_phase(RequestPhase.PREFILL)
        row_capacity = self._config.max_decode_rows - decode_count - len(prefilling)
        prefill_capacity = self._config.max_prefill_rows - len(prefilling)
        slots = max(0, min(row_capacity, prefill_capacity))
        vision_prefills = sum(
            state.request.kind is WorkKind.VISION for state in prefilling
        )

        for request_id in tuple(self._waiting):
            if slots <= 0:
                break
            state = self._states[request_id]
            if (
                state.request.kind is WorkKind.VISION
                and vision_prefills >= self._config.max_vision_prefills
            ):
                continue
            self._waiting.remove(request_id)
            state.phase = RequestPhase.PREFILL
            slots -= 1
            if state.request.kind is WorkKind.VISION:
                vision_prefills += 1

    def _drain_pending_cancels(self) -> list[CancelledRequest]:
        cancelled: list[CancelledRequest] = []
        pending = tuple(self._pending_cancels.items())
        self._pending_cancels.clear()
        for request_id, reason in pending:
            state = self._states.get(request_id)
            if state is None:
                continue
            previous_phase = state.phase
            self._finish(state, RequestPhase.CANCELLED, reason)
            cancelled.append(CancelledRequest(request_id, previous_phase, reason))
        return cancelled

    def _finish(
        self,
        state: _RequestState,
        phase: RequestPhase,
        reason: str,
        *,
        stop_sequence: str | None = None,
    ) -> TerminalRequest:
        request_id = state.request.request_id
        if request_id in self._waiting:
            self._waiting.remove(request_id)
        self._states.pop(request_id, None)
        terminal = TerminalRequest(
            request_id=request_id,
            phase=phase,
            reason=reason,
            stop_sequence=stop_sequence,
        )
        self._terminal[request_id] = terminal
        while len(self._terminal) > self._config.terminal_history_size:
            self._terminal.popitem(last=False)
        return terminal

    def _validated_result_state(
        self,
        request_id: str,
        expected_phase: RequestPhase,
        position: int,
    ) -> _RequestState:
        state = self._states.get(request_id)
        if state is None:
            raise RuntimeError(f"result references unknown request {request_id}")
        if state.phase is not expected_phase:
            raise RuntimeError(
                f"request {request_id} is {state.phase.value}, expected "
                f"{expected_phase.value}"
            )
        if position < state.position:
            raise ValueError("request position cannot move backwards")
        return state

    @staticmethod
    def _require_exact_results(
        expected: tuple[str, ...],
        observed: tuple[str, ...],
        phase: str,
    ) -> None:
        if len(observed) != len(set(observed)):
            raise ValueError(f"duplicate {phase} result")
        if set(expected) != set(observed):
            raise ValueError(
                f"{phase} results do not match leased rows: "
                f"expected={expected!r} observed={observed!r}"
            )
