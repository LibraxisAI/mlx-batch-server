from __future__ import annotations

from types import MappingProxyType

import pytest

from mlx_batch_server.runtime.contracts import (
    BackendKind,
    GenerationRequest,
    RuntimeKey,
)
from mlx_batch_server.runtime.fusion.qwen4_exp.model.tensor import (
    _chat_template_reasoning,
    _chat_template_tools,
    _parse_tensor_sampling,
)


def _request(
    tool_choice: object,
    *,
    reasoning: MappingProxyType[str, object] | None = None,
) -> GenerationRequest:
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
        sampling=MappingProxyType(
            {
                "max_output_tokens": 32,
                "parallel_tool_calls": True,
                "tool_choice": tool_choice,
            }
        ),
        reasoning=reasoning or MappingProxyType({}),
    )


def test_tensor_sampling_accepts_prompt_only_tool_controls() -> None:
    sampling = _parse_tensor_sampling(_request("auto"))

    assert sampling.max_output_tokens == 32


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
