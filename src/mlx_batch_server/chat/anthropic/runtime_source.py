"""Bind Anthropic Messages to the one canonical inference runtime owner."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import suppress

from mlx_batch_server.runtime.contracts import (
    BackendTurn,
    GenerationRequest,
    RuntimeKey,
)
from mlx_batch_server.runtime.events import (
    TurnCancelled,
    TurnEvent,
    TurnFailed,
    TurnStarted,
)
from mlx_batch_server.runtime.service import FirstWriterCancelToken, RuntimeStartService
from mlx_batch_server.runtime.turn import GenerationTurn, TurnState

from .errors import AnthropicAPIError
from .turn_source import AnthropicTurn

ResolvedModel = tuple[RuntimeKey, str]
ResolveModel = Callable[[str], ResolvedModel]


class RuntimeAnthropicTurnSource:
    """Translate one Anthropic turn and consume events from one runtime owner."""

    def __init__(
        self,
        *,
        starter: RuntimeStartService,
        resolve_model: ResolveModel,
        max_pending_events: int = 4096,
    ) -> None:
        if not isinstance(starter, RuntimeStartService):
            raise TypeError("starter must be a RuntimeStartService")
        if not callable(resolve_model):
            raise TypeError("resolve_model must be callable")
        if max_pending_events < 2:
            raise ValueError("max_pending_events must be at least 2")
        self._starter = starter
        self._resolve_model = resolve_model
        self._max_pending_events = int(max_pending_events)

    async def stream(self, turn: AnthropicTurn) -> AsyncIterator[TurnEvent]:
        """Yield unsequenced canonical events and drain the admitted turn."""

        request = self._request(turn)
        event_turn = GenerationTurn(max_pending_events=self._max_pending_events)
        subscription = event_turn.subscribe(max_pending_events=self._max_pending_events)
        cancel = FirstWriterCancelToken()
        backend: list[BackendTurn] = []
        driver = asyncio.create_task(
            self._drive(request, event_turn, cancel, backend),
            name=f"anthropic-runtime-{request.response_id}",
        )
        try:
            async for sequenced in subscription:
                yield sequenced.event
        finally:
            if event_turn.state is not TurnState.TERMINAL:
                reason = "anthropic_client_disconnected"
                cancel.cancel(reason)
                if backend:
                    backend[0].cancel(reason)
            with suppress(asyncio.CancelledError):
                await asyncio.shield(driver)

    def _request(self, turn: AnthropicTurn) -> GenerationRequest:
        if not isinstance(turn, AnthropicTurn):
            raise TypeError("turn must be an AnthropicTurn")
        try:
            runtime, role = self._resolve_model(turn.model_alias)
        except AnthropicAPIError:
            raise
        except Exception as error:
            raise AnthropicAPIError(
                str(error) or "model alias could not be resolved",
                error_type="invalid_request_error",
            ) from error
        if not isinstance(runtime, RuntimeKey):
            raise TypeError("resolve_model must return a RuntimeKey")
        if not isinstance(role, str) or not role.strip():
            raise TypeError("resolve_model must return a non-empty runtime role")

        metadata = dict(turn.metadata)
        expected = {
            "requested_model": turn.model_alias,
            "resolved_model": runtime.model_id,
            "runtime_role": role.strip(),
            "protocol": "anthropic_messages",
        }
        for key, value in expected.items():
            claimed = metadata.get(key)
            if claimed is not None and claimed != value:
                raise AnthropicAPIError(
                    f"metadata.{key} conflicts with canonical runtime resolution",
                    error_type="invalid_request_error",
                )
            metadata[key] = value

        sampling = dict(turn.sampling)
        if turn.tool_choice is not None:
            sampling["tool_choice"] = dict(turn.tool_choice)
        return GenerationRequest(
            response_id=f"anthropic_{uuid.uuid4().hex}",
            runtime=runtime,
            messages=tuple(dict(message) for message in turn.messages),
            tools=tuple(dict(tool) for tool in turn.tools),
            sampling=sampling,
            reasoning=dict(turn.reasoning),
            metadata=metadata,
        )

    async def _drive(
        self,
        request: GenerationRequest,
        event_turn: GenerationTurn,
        cancel: FirstWriterCancelToken,
        backend: list[BackendTurn],
    ) -> None:
        try:
            handle = await self._starter.start(request, event_turn, cancel=cancel)
            backend.append(handle)
            if handle.response_id != request.response_id:
                handle.cancel("runtime_response_id_mismatch")
                raise RuntimeError("runtime returned a foreign response id")
            await handle.wait_closed()
            await asyncio.sleep(0)
            if event_turn.state is not TurnState.TERMINAL:
                event_turn.fail(
                    TurnFailed(
                        error="runtime turn closed without a terminal event",
                        code="runtime_missing_terminal",
                        status_code=500,
                    )
                )
        except asyncio.CancelledError:
            if event_turn.state is TurnState.IDLE:
                event_turn.emit(
                    TurnStarted(
                        response_id=request.response_id,
                        model=request.runtime.model_id,
                        created_at=int(time.time()),
                    )
                )
            if event_turn.state is not TurnState.TERMINAL:
                event_turn.cancel(
                    TurnCancelled(cancel.reason or "anthropic_turn_cancelled")
                )
        except Exception as error:
            if event_turn.state is not TurnState.TERMINAL:
                event_turn.fail(
                    TurnFailed(
                        error=str(error) or type(error).__name__,
                        code="runtime_start_failed",
                        status_code=500,
                    )
                )


__all__ = ["ResolveModel", "ResolvedModel", "RuntimeAnthropicTurnSource"]
