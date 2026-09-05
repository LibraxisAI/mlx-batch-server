from __future__ import annotations

from types import MappingProxyType

import pytest

from mlx_batch_server.runtime.contracts import (
    BackendKind,
    GenerationRequest,
    RuntimeKey,
)
from mlx_batch_server.runtime.fusion.qwen4_exp.model.tensor import (
    _chat_template_messages,
    _chat_template_reasoning,
    _chat_template_tools,
    _parse_tensor_sampling,
)


def _request(
    tool_choice: object,
    *,
    reasoning: MappingProxyType[str, object] | None = None,
    max_output_tokens: int | None = 32,
) -> GenerationRequest:
    sampling: dict[str, object] = {
        "parallel_tool_calls": True,
        "tool_choice": tool_choice,
    }
    if max_output_tokens is not None:
        sampling["max_output_tokens"] = max_output_tokens
    return GenerationRequest(
        response_id="resp_tools",
        runtime=RuntimeKey(
            model_id="test/qwen4",
            revision="snapshot",
            backend=BackendKind.FUSED_MTP_MLX,
        ),
        messages=({"role": "user", "content": "inspect"},),
        tools=(
            MappingProxyType({"type": "function", "name": "first"}),
            MappingProxyType({"type": "function", "name": "second"}),
        ),
        sampling=MappingProxyType(sampling),
        reasoning=reasoning or MappingProxyType({}),
    )


def test_tensor_sampling_accepts_prompt_only_tool_controls() -> None:
    sampling = _parse_tensor_sampling(
        _request("auto"),
        context_length=128,
        prompt_tokens=16,
    )

    assert sampling.max_output_tokens == 32


def test_tensor_sampling_preserves_exact_stop_sequence_order() -> None:
    request = _request("auto")
    request = GenerationRequest(
        response_id=request.response_id,
        runtime=request.runtime,
        messages=request.messages,
        tools=request.tools,
        sampling={**request.sampling, "stop": ("END", "end", "e\u0301")},
        reasoning=request.reasoning,
    )

    sampling = _parse_tensor_sampling(
        request,
        context_length=128,
        prompt_tokens=16,
    )

    assert sampling.stop_sequences == ("END", "end", "e\u0301")


def test_tensor_sampling_defaults_to_discovered_remaining_context() -> None:
    sampling = _parse_tensor_sampling(
        _request("auto", max_output_tokens=None),
        context_length=128,
        prompt_tokens=16,
    )

    assert sampling.max_output_tokens == 112


def test_tensor_sampling_clamps_explicit_limit_to_remaining_context() -> None:
    sampling = _parse_tensor_sampling(
        _request("auto", max_output_tokens=256),
        context_length=128,
        prompt_tokens=16,
    )

    assert sampling.max_output_tokens == 112


def test_tensor_sampling_rejects_prompt_that_fills_model_context() -> None:
    with pytest.raises(ValueError, match="tokenized prompt length 128"):
        _parse_tensor_sampling(
            _request("auto", max_output_tokens=None),
            context_length=128,
            prompt_tokens=128,
        )


def test_tool_choice_controls_exact_template_tool_set() -> None:
    assert _chat_template_tools(_request("none")) is None
    assert [tool["name"] for tool in _chat_template_tools(_request("auto")) or []] == [
        "first",
        "second",
    ]
    assert [
        tool["name"]
        for tool in _chat_template_tools(
            _request(MappingProxyType({"type": "function", "name": "second"}))
        )
        or []
    ] == ["second"]


def test_forced_tool_choice_must_name_a_request_tool() -> None:
    with pytest.raises(ValueError, match="exactly one request tool"):
        _chat_template_tools(
            _request(MappingProxyType({"type": "function", "name": "missing"}))
        )


@pytest.mark.parametrize("effort", ("none", "off"))
def test_reasoning_off_closes_thinking_in_the_checkpoint_template(effort: str) -> None:
    request = _request("none", reasoning=MappingProxyType({"effort": effort}))

    assert _chat_template_reasoning(request) == {"enable_thinking": False}


@pytest.mark.parametrize(
    ("effort", "template_effort"),
    (("low", "low"), ("medium", "medium"), ("high", "xhigh"), ("xhigh", "xhigh")),
)
def test_reasoning_effort_maps_to_the_checkpoint_template(
    effort: str,
    template_effort: str,
) -> None:
    request = _request("none", reasoning=MappingProxyType({"effort": effort}))

    assert _chat_template_reasoning(request) == {
        "enable_thinking": True,
        "reasoning_effort": template_effort,
    }


def test_invalid_reasoning_controls_fail_before_tokenization() -> None:
    with pytest.raises(ValueError, match=r"reasoning\.enabled"):
        _chat_template_reasoning(
            _request("none", reasoning=MappingProxyType({"enabled": "yes"}))
        )
    with pytest.raises(ValueError, match=r"reasoning\.effort"):
        _chat_template_reasoning(
            _request("none", reasoning=MappingProxyType({"effort": "minimal"}))
        )


def test_chat_template_projects_developer_without_mutating_lineage() -> None:
    messages = (
        MappingProxyType(
            {
                "role": "developer",
                "content": (
                    MappingProxyType({"type": "input_text", "text": "Be precise."}),
                ),
            }
        ),
        MappingProxyType({"role": "user", "content": "Hello"}),
    )

    rendered = _chat_template_messages(messages)

    assert rendered == [
        {"role": "system", "content": "Be precise."},
        {"role": "user", "content": "Hello"},
    ]
    assert messages[0]["role"] == "developer"


def test_chat_template_coalesces_leading_instruction_roles_in_order() -> None:
    rendered = _chat_template_messages(
        (
            {"role": "system", "content": "Platform policy."},
            {"role": "developer", "content": "Product voice."},
            {
                "role": "tool",
                "content": "receipt",
                "call_id": "call_1",
                "type": "function_call_output",
            },
        )
    )

    assert rendered == [
        {"role": "system", "content": "Platform policy.\n\nProduct voice."},
        {
            "role": "tool",
            "content": "receipt",
            "call_id": "call_1",
            "type": "function_call_output",
        },
    ]


def test_chat_template_rejects_late_developer_role() -> None:
    with pytest.raises(ValueError, match="must precede conversation"):
        _chat_template_messages(
            (
                {"role": "user", "content": "Hello"},
                {"role": "developer", "content": "Replace policy."},
            )
        )
