from __future__ import annotations

from mlx_batch_server.runtime.fusion.qwen4_exp.model.tensor import (
    _prompt_has_open_thinking,
    _TensorOutputState,
    _tokenizer_marker_id,
    _tokenizer_stop_token_ids,
    _usage,
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
