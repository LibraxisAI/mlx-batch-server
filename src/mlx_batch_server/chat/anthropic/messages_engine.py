"""Drive one Anthropic Messages turn over the typed runtime seam.

The engine owns no inference. It validates intent (``request_mapper``), asks
the bound inference owner (``turn_source``) for typed runtime events, and lets
the projector turn those events into Anthropic protocol shapes. Both
transports walk the same projector, so a streamed message and its non-stream
equivalent describe the same generation.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from mlx_batch_server.utils.logger import logger

from .errors import AnthropicAPIError
from .projector import AnthropicMessageProjector
from .request_mapper import build_turn
from .turn_source import AnthropicTurnSource, require_turn_source

if TYPE_CHECKING:
    from mlx_batch_server.runtime.events import TurnEvent

    from .anthropic_schema import (
        AnthropicStreamEvent,
        MessagesRequest,
        MessagesResponse,
    )


def new_message_id() -> str:
    """Mint one message identifier in Anthropic's ``msg_`` shape."""

    return f"msg_{uuid.uuid4().hex[:24]}"


class AnthropicMessagesEngine:
    """Protocol-side owner of one Anthropic Messages turn."""

    def __init__(self, *, turn_source: AnthropicTurnSource | None = None) -> None:
        self._turn_source = turn_source

    def _source(self) -> AnthropicTurnSource:
        return self._turn_source or require_turn_source()

    def _prepare(
        self, request: MessagesRequest
    ) -> tuple[AnthropicMessageProjector, AsyncIterator[TurnEvent]]:
        turn = build_turn(request)
        projector = AnthropicMessageProjector(
            message_id=new_message_id(),
            # The alias the client asked for, held stable for the whole turn.
            model_alias=request.model,
        )
        return projector, self._source().stream(turn).__aiter__()

    async def generate(self, request: MessagesRequest) -> MessagesResponse:
        """Run one turn and return the terminal Anthropic message."""

        projector, events = self._prepare(request)
        async for event in events:
            projector.observe(event)
        failure = projector.failure
        if failure is not None:
            raise AnthropicAPIError(failure.message, error_type=failure.type)
        if not projector.stopped:
            raise AnthropicAPIError(
                "the runtime turn ended without a terminal event",
                error_type="api_error",
            )
        return projector.terminal_message()

    async def generate_stream(
        self, request: MessagesRequest
    ) -> AsyncIterator[AnthropicStreamEvent]:
        """Run one turn and yield its Anthropic streaming lifecycle."""

        projector, events = self._prepare(request)
        # message_start opens every Anthropic stream, before any runtime event
        # is observed, so the lifecycle is well-formed even if the inference
        # owner starts by reporting a failure.
        yield projector.message_start_event()
        projector.observe_started()
        try:
            async for event in events:
                for projected in projector.observe(event):
                    yield projected
        except AnthropicAPIError as error:
            logger.error("Anthropic turn failed: %s", error.message)
            for projected in projector.fail(error.error_type, error.message):
                yield projected
            return
        except Exception as error:
            logger.error("Anthropic turn failed: %s", error, exc_info=True)
            for projected in projector.fail("api_error", str(error)):
                yield projected
            return
        if not projector.stopped:
            # A stream that simply stops is indistinguishable from a truncated
            # connection. Say so explicitly instead of ending mid-message.
            for projected in projector.fail(
                "api_error",
                "the runtime turn ended without a terminal event",
            ):
                yield projected


__all__ = ["AnthropicMessagesEngine", "new_message_id"]
