from __future__ import annotations

from types import MethodType

from mlx_batch_server.runtime.contracts import (
    BackendKind,
    GenerationRequest,
    PreparedGenerationRequest,
    RequestModality,
    RuntimeKey,
)
from mlx_batch_server.runtime.events import TextCompleted, TextDelta
from mlx_batch_server.runtime.fusion.mtp import MtpPolicy
from mlx_batch_server.runtime.fusion.qwen4_exp.model.sampling import SamplerConfig
from mlx_batch_server.runtime.fusion.qwen4_exp.model.tensor import (
    _prompt_has_open_thinking,
    _Qwen4ExpTensorRuntime,
    _TensorDecodeOutcome,
    _TensorOutputState,
    _TensorReservation,
    _tokenizer_marker_id,
    _tokenizer_stop_token_ids,
    _usage,
)
from mlx_batch_server.runtime.fusion.scheduler import (
    ScheduledRequest,
    SchedulerPlan,
    WorkKind,
)


class _Tokenizer:
    eos_token_id = 248046
    think_start_id = 248068
    think_end_id = 248069
    unk_token_id = 0

    @staticmethod
    def convert_tokens_to_ids(marker: str) -> int:
        return {
            "<think>": 248068,
            "</think>": 248069,
        }.get(marker, 0)


def test_tokenizer_stop_ids_include_checkpoint_and_chat_template_eos() -> None:
    tokenizer = _Tokenizer()

    assert _tokenizer_stop_token_ids(tokenizer, (248044,)) == frozenset(
        {248044, 248046}
    )
    assert _tokenizer_marker_id(tokenizer, "think_start_id", "<think>") == 248068
    assert _tokenizer_marker_id(tokenizer, "missing", "</think>") == 248069


def test_prompt_thinking_state_uses_the_last_open_or_close_marker() -> None:
    assert _prompt_has_open_thinking((1, 248068, 2), 248068, 248069)
    assert not _prompt_has_open_thinking((248068, 2, 248069, 3), 248068, 248069)
    assert not _prompt_has_open_thinking((1, 2, 3), 248068, 248069)


def test_output_state_separates_reasoning_and_hides_protocol_tokens() -> None:
    state = _TensorOutputState(
        stop_token_ids=frozenset({248044, 248046}),
        think_start_id=248068,
        think_end_id=248069,
        in_reasoning=True,
    )

    assert state.route(11, "inspect") == ("", "inspect")
    assert state.route(12, "\n") == ("", "\n")
    assert state.route(248069, "</think>") == ("", "")
    assert state.route(13, "\n") == ("", "")
    assert state.route(14, "\n") == ("", "")
    assert state.route(15, "answer") == ("answer", "")
    assert state.route(248046, "<|im_end|>") == ("", "")
    assert state.reasoning_tokens == 2


def test_usage_reports_routed_reasoning_tokens() -> None:
    class _Reservation:
        prompt_tokens = (1, 2, 3)
        output_tokens = 5
        output_state = _TensorOutputState(
            stop_token_ids=frozenset(),
            think_start_id=None,
            think_end_id=248069,
            in_reasoning=True,
        )

    reservation = _Reservation()
    reservation.output_state.route(11, "inspect")
    reservation.output_state.route(12, "\n")

    usage = _usage(reservation)  # type: ignore[arg-type]

    assert usage.output_tokens == 5
    assert usage.reasoning_output_tokens == 2


def test_generated_think_marker_can_start_reasoning_without_prompt_marker() -> None:
    state = _TensorOutputState(
        stop_token_ids=frozenset(),
        think_start_id=248068,
        think_end_id=248069,
        in_reasoning=False,
    )

    assert state.route(248068, "<think>") == ("", "")
    assert state.route(11, "inspect") == ("", "inspect")


def test_stop_filter_never_moves_buffered_reasoning_into_visible_text() -> None:
    state = _TensorOutputState(
        stop_token_ids=frozenset(),
        think_start_id=248068,
        think_end_id=248069,
        in_reasoning=True,
        stop_sequences=("END",),
    )

    text, reasoning = state.route(11, "reason EN")
    assert state.filter_stops(text, reasoning) == ("", "reason ", None)
    assert state.flush_channel_boundary(248069) == ("", "EN")
    assert state.route(248069, "</think>") == ("", "")
    assert state.route(12, "D visible") == ("D visible", "")
    assert state.filter_stops("D visible", "") == ("D visible", "", None)


def test_stop_filter_discards_exact_match_and_same_chunk_tail() -> None:
    state = _TensorOutputState(
        stop_token_ids=frozenset(),
        think_start_id=None,
        think_end_id=None,
        in_reasoning=False,
        stop_sequences=("STOP",),
    )

    assert state.filter_stops("before ST", "") == ("before ", "", None)
    assert state.filter_stops("OP discarded", "") == ("", "", "STOP")


class _PieceDetokenizer:
    def __init__(self) -> None:
        self.last_segment = ""

    def add_token(self, token: int) -> None:
        self.last_segment = {11: "before ST", 12: "OP discarded"}[token]

    def finalize(self) -> None:
        pass

    def reset(self) -> None:
        self.last_segment = ""


def test_tensor_execute_terminates_on_first_exact_text_match_without_leakage() -> None:
    request = GenerationRequest(
        response_id="resp_stop",
        runtime=RuntimeKey(
            model_id="test/qwen4",
            revision="snapshot",
            backend=BackendKind.FUSED_MTP_MLX,
        ),
        messages=({"role": "user", "content": "continue"},),
        sampling={"stop": ("STOP",)},
    )
    prepared = PreparedGenerationRequest(request, RequestModality.TEXT)
    sampler = SamplerConfig()
    reservation = _TensorReservation(
        request=request,
        prepared_request=prepared,
        lease_id="lease",
        cache=[],
        mtp_cache=[],
        prompt_tokens=(1, 2),
        max_output_tokens=16,
        sampler=sampler,
        draft_sampler=sampler,
        rng=None,
        prefix_lease=object(),  # type: ignore[arg-type]
        prefix_context_fingerprint="",
        output_state=_TensorOutputState(
            stop_token_ids=frozenset(),
            think_start_id=None,
            think_end_id=None,
            in_reasoning=False,
            stop_sequences=("STOP",),
        ),
        detokenizer=_PieceDetokenizer(),
    )
    runtime = object.__new__(_Qwen4ExpTensorRuntime)
    runtime._closed = False
    runtime._reservations = {request.response_id: reservation}
    runtime._encoders = {}
    runtime._prefill_rows = 0
    runtime._decode_rows = 0
    outcomes = iter(
        (
            _TensorDecodeOutcome(tokens=(11,), finished=False),
            _TensorDecodeOutcome(tokens=(12,), finished=False),
        )
    )

    def decode_batch(self, reservations, *, mtp_decision, draft_depth):
        del self, mtp_decision, draft_depth
        outcome = next(outcomes)
        reservations[0].output_tokens += len(outcome.tokens)
        reservations[0].position += len(outcome.tokens)
        return (outcome,), ()

    runtime._decode_batch = MethodType(decode_batch, runtime)
    runtime._mtp_decision = MethodType(
        lambda self, plan, reservations, policy: None,
        runtime,
    )
    reservations = {request.response_id: reservation}
    requests = {request.response_id: request}

    first = runtime.execute(
        SchedulerPlan(
            step_id=1,
            decode_rows=(ScheduledRequest("resp_stop", WorkKind.TEXT, 0),),
        ),
        reservations,
        requests,
        MtpPolicy(),
    )
    second = runtime.execute(
        SchedulerPlan(
            step_id=2,
            decode_rows=(ScheduledRequest("resp_stop", WorkKind.TEXT, 1),),
        ),
        reservations,
        requests,
        MtpPolicy(),
    )

    assert first.decode_results[0].finished is False
    terminal = second.decode_results[0]
    assert terminal.finished is True
    assert terminal.finish_reason == "stop_sequence"
    assert terminal.stop_sequence == "STOP"
    events = (*first.events["resp_stop"], *second.events["resp_stop"])
    assert "".join(event.delta for event in events if isinstance(event, TextDelta)) == (
        "before "
    )
    completed = [event for event in events if isinstance(event, TextCompleted)]
    assert completed[0].text == "before "
    assert all(
        "STOP" not in repr(event) and "discarded" not in repr(event) for event in events
    )
