from __future__ import annotations

from typing import Any

import pytest

from mlx_batch_server.chat.anthropic import router as anthropic_router
from mlx_batch_server.chat.anthropic.errors import AnthropicAPIError
from mlx_batch_server.chat.anthropic.runtime_source import RuntimeAnthropicTurnSource
from mlx_batch_server.chat.anthropic.turn_source import AnthropicTurn
from mlx_batch_server.runtime.contracts import (
    BackendKind,
    GenerationRequest,
    RuntimeKey,
    TurnSink,
)
from mlx_batch_server.runtime.events import TurnCompleted, TurnFailed, TurnStarted
from mlx_batch_server.runtime.service import FirstWriterCancelToken, RuntimeStartService

RUNTIME = RuntimeKey(
    model_id="local/qwen",
    revision="pinned",
    backend=BackendKind.FUSED_MTP_MLX,
)


class _Turn:
    def __init__(self, response_id: str) -> None:
        self._response_id = response_id
        self.cancelled: list[str] = []

    @property
    def response_id(self) -> str:
        return self._response_id

    def cancel(self, reason: str) -> bool:
        self.cancelled.append(reason)
        return True

    async def wait_closed(self) -> None:
        return None


class _Starter(RuntimeStartService):
    def __init__(self, *, terminal: bool = True) -> None:
        self.requests: list[GenerationRequest] = []
        self.terminal = terminal

    async def start(
        self,
        request: GenerationRequest,
        sink: TurnSink,
        *,
        cancel: FirstWriterCancelToken | None = None,
    ) -> _Turn:
        assert cancel is not None
        self.requests.append(request)
        sink.emit(
            TurnStarted(
                response_id=request.response_id,
                model=request.runtime.model_id,
                created_at=1,
            )
        )
        if self.terminal:
            sink.emit(TurnCompleted(finish_reason="stop"))
        return _Turn(request.response_id)


async def _collect(
    source: RuntimeAnthropicTurnSource, turn: AnthropicTurn
) -> list[Any]:
    return [event async for event in source.stream(turn)]


@pytest.mark.asyncio
async def test_runtime_source_maps_one_turn_to_the_canonical_owner() -> None:
    starter = _Starter()
    source = RuntimeAnthropicTurnSource(
        starter=starter,
        resolve_model=lambda alias: (RUNTIME, "main"),
    )
    turn = AnthropicTurn(
        model_alias="buddy",
        messages=({"role": "user", "content": "hello"},),
        tools=({"type": "function", "function": {"name": "lookup"}},),
        tool_choice={"type": "auto"},
        sampling={"max_tokens": 32},
        reasoning={"enabled": True, "budget_tokens": 8},
        metadata={"user_id": "vet-1"},
    )

    events = await _collect(source, turn)

    assert [type(event) for event in events] == [TurnStarted, TurnCompleted]
    assert len(starter.requests) == 1
    request = starter.requests[0]
    assert request.runtime is RUNTIME
    assert request.messages == turn.messages
    assert request.tools == turn.tools
    assert request.sampling == {"max_tokens": 32, "tool_choice": {"type": "auto"}}
    assert request.reasoning == turn.reasoning
    assert request.metadata == {
        "user_id": "vet-1",
        "requested_model": "buddy",
        "resolved_model": "local/qwen",
        "runtime_role": "main",
        "protocol": "anthropic_messages",
    }


@pytest.mark.asyncio
async def test_runtime_source_synthesizes_a_missing_terminal_failure() -> None:
    source = RuntimeAnthropicTurnSource(
        starter=_Starter(terminal=False),
        resolve_model=lambda alias: (RUNTIME, "main"),
    )

    events = await _collect(
        source,
        AnthropicTurn(
            model_alias="buddy",
            messages=({"role": "user", "content": "hello"},),
        ),
    )

    assert isinstance(events[-1], TurnFailed)
    assert events[-1].code == "runtime_missing_terminal"


@pytest.mark.asyncio
async def test_runtime_source_maps_alias_failure_to_anthropic_error() -> None:
    def reject(_alias: str):
        raise ValueError("unknown model alias")

    source = RuntimeAnthropicTurnSource(
        starter=_Starter(),
        resolve_model=reject,
    )

    with pytest.raises(AnthropicAPIError, match="unknown model alias"):
        await _collect(
            source,
            AnthropicTurn(
                model_alias="missing",
                messages=({"role": "user", "content": "hello"},),
            ),
        )


def test_anthropic_router_has_no_legacy_runtime_lifecycle_owner() -> None:
    assert not hasattr(anthropic_router, "endpoint_runtime_session")
