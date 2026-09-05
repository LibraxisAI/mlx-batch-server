from __future__ import annotations

import asyncio
from types import SimpleNamespace
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


class _BlockingTurn(_Turn):
    def __init__(self, response_id: str) -> None:
        super().__init__(response_id)
        self.closed = asyncio.Event()

    def cancel(self, reason: str) -> bool:
        accepted = super().cancel(reason)
        self.closed.set()
        return accepted

    async def wait_closed(self) -> None:
        await self.closed.wait()


class _BlockingStarter(RuntimeStartService):
    def __init__(self) -> None:
        self.turn: _BlockingTurn | None = None

    async def start(
        self,
        request: GenerationRequest,
        sink: TurnSink,
        *,
        cancel: FirstWriterCancelToken | None = None,
    ) -> _BlockingTurn:
        assert cancel is not None
        sink.emit(
            TurnStarted(
                response_id=request.response_id,
                model=request.runtime.model_id,
                created_at=1,
            )
        )
        self.turn = _BlockingTurn(request.response_id)
        return self.turn


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
        tools=({"type": "function", "name": "lookup"},),
        tool_choice="auto",
        sampling={"max_output_tokens": 32},
        reasoning={"enabled": True, "budget_tokens": 8},
        metadata={"user_id": "vet-1"},
    )

    events = await _collect(source, turn)

    assert [type(event) for event in events] == [TurnStarted, TurnCompleted]
    assert len(starter.requests) == 1
    request = starter.requests[0]
    assert request.runtime is RUNTIME
    assert request.messages == turn.messages
    assert request.media == ()
    assert request.tools == turn.tools
    assert request.sampling == {"max_output_tokens": 32, "tool_choice": "auto"}
    assert request.reasoning == turn.reasoning
    assert request.metadata == {
        "user_id": "vet-1",
        "requested_model": "buddy",
        "resolved_model": "local/qwen",
        "runtime_role": "main",
        "protocol": "anthropic_messages",
    }


@pytest.mark.asyncio
async def test_runtime_source_copies_canonical_tool_result_media() -> None:
    starter = _Starter()
    source = RuntimeAnthropicTurnSource(
        starter=starter,
        resolve_model=lambda alias: (RUNTIME, "main"),
    )
    media = (
        {
            "type": "input_image",
            "image_url": "https://media.3more.ai/a.png",
            "_role": "tool",
            "_message_index": 0,
            "_content_index": 1,
        },
    )
    turn = AnthropicTurn(
        model_alias="buddy",
        messages=(
            {
                "type": "function_call_output",
                "role": "tool",
                "call_id": "toolu_1",
                "output": "see photo",
                "is_error": False,
                "content": ({"type": "input_text", "text": "see photo"},),
            },
        ),
        media=media,
    )

    await _collect(source, turn)

    assert starter.requests[0].media == media


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


@pytest.mark.asyncio
async def test_client_disconnect_is_delivered_to_the_started_backend() -> None:
    starter = _BlockingStarter()
    source = RuntimeAnthropicTurnSource(
        starter=starter,
        resolve_model=lambda alias: (RUNTIME, "main"),
    )
    events = source.stream(
        AnthropicTurn(
            model_alias="buddy",
            messages=({"role": "user", "content": "hello"},),
        )
    )

    assert isinstance(await anext(events), TurnStarted)
    await events.aclose()

    assert starter.turn is not None
    assert starter.turn.cancelled == ["anthropic_client_disconnected"]


def test_canonical_app_binds_the_engine_to_its_own_receipt() -> None:
    source = RuntimeAnthropicTurnSource(
        starter=_Starter(),
        resolve_model=lambda alias: (RUNTIME, "main"),
    )
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                responses_runtime=SimpleNamespace(
                    responses=SimpleNamespace(anthropic_turn_source=source)
                )
            )
        )
    )

    engine = anthropic_router._create_request_engine(request, "buddy")

    assert engine._turn_source is source


def test_anthropic_router_has_no_legacy_runtime_lifecycle_owner() -> None:
    assert not hasattr(anthropic_router, "endpoint_runtime_session")
