"""Drive one Anthropic Messages turn over the typed runtime seam.

The engine owns no inference. It validates intent (``request_mapper``), asks
the bound inference owner (``turn_source``) for typed runtime events, and lets
the projector turn those events into Anthropic protocol shapes. Both
transports walk the same projector, so a streamed message and its non-stream
equivalent describe the same generation.

Capability policy is decided once, by ``capabilities``, and travels here as a
``CapabilityAdmission``. The engine consumes that receipt rather than
re-deciding, so the streaming and non-streaming paths cannot drift apart. A
caller that drives the engine directly — the documented substitution seam —
gets the ``detached`` profile classified here, so no request can reach the
turn source unclassified.

Three protocol decisions are resolved here and handed to the projector, which
is not allowed to guess any of them: whether Anthropic thinking may reach the
wire (and who signs it), which capacity lane actually served the turn, and
whether the validated web-fetch tool enabled citations. Runtime events cannot
widen any of those request and admission decisions.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Final

from mlx_batch_server.utils.logger import logger

from .anthropic_schema import ResponseServiceTier, ThinkingConfigEnabled
from .capabilities import (
    AnthropicCapabilityProfile,
    CapabilityAdmission,
    CapabilityStatus,
    detached_profile,
    enforce_capabilities,
)
from .errors import AnthropicAPIError, UnsupportedCapabilityError
from .projector import (
    AnthropicMessageProjector,
    ThinkingProjection,
    ThinkingSignatureOwner,
)
from .request_mapper import build_turn
from .turn_source import AnthropicTurnSource, require_turn_source

if TYPE_CHECKING:
    from mlx_batch_server.runtime.events import TurnEvent

    from .anthropic_schema import (
        AnthropicStreamEvent,
        MessagesRequest,
        MessagesResponse,
    )


#: The one capacity lane this process serves. Local inference has no priority
#: queue and no batch lane, so ``auto`` and ``standard_only`` both land here —
#: and the response says so instead of echoing the request back.
DELIVERED_SERVICE_TIER: Final = ResponseServiceTier.STANDARD


def new_message_id() -> str:
    """Mint one message identifier in Anthropic's ``msg_`` shape."""

    return f"msg_{uuid.uuid4().hex[:24]}"


class AnthropicMessagesEngine:
    """Protocol-side owner of one Anthropic Messages turn."""

    def __init__(
        self,
        *,
        turn_source: AnthropicTurnSource | None = None,
        thinking_signature_owner: ThinkingSignatureOwner | None = None,
    ) -> None:
        self._turn_source = turn_source
        # No production owner is bound anywhere in this repository, and the
        # public capability profile says enabled thinking is unsupported. The
        # parameter exists so admitting thinking later is one explicit
        # binding — not so that today's surface can pretend it has one.
        self._thinking_signature_owner = thinking_signature_owner

    def _source(self) -> AnthropicTurnSource:
        return self._turn_source or require_turn_source()

    def _prepare(
        self,
        request: MessagesRequest,
        admission: CapabilityAdmission | None,
    ) -> tuple[AnthropicMessageProjector, AsyncIterator[TurnEvent]]:
        if admission is None:
            admission = enforce_capabilities(request, detached_profile(request.model))
        profile = admission.profile
        # Both decisions are taken before ``build_turn`` and before the turn
        # source is touched, so a refusal is an HTTP failure — on the
        # streaming transport too, where the caller has not yet been handed a
        # StreamingResponse and no SSE byte can exist.
        thinking = self._thinking_projection(request, profile)
        service_tier = _admitted_service_tier(request, profile)
        citations_enabled = any(
            tool.type == "web_fetch_20250910"
            and tool.name == "web_fetch"
            and tool.citations is not None
            and tool.citations.enabled is True
            for tool in request.tools or ()
        )
        turn = build_turn(request)
        projector = AnthropicMessageProjector(
            message_id=new_message_id(),
            # The alias the client asked for, held stable for the whole turn.
            model_alias=request.model,
            thinking=thinking,
            service_tier=service_tier,
            citations_enabled=citations_enabled,
        )
        return projector, self._source().stream(turn).__aiter__()

    def _thinking_projection(
        self,
        request: MessagesRequest,
        profile: AnthropicCapabilityProfile,
    ) -> ThinkingProjection:
        """Decide, from the profile alone, whether thinking may be projected.

        Omitted and disabled thinking are refused *output*, not refused
        requests: the turn runs normally and simply carries no thinking on the
        wire, however much the runtime reasons internally.
        """

        if request.thinking is None or not isinstance(
            request.thinking, ThinkingConfigEnabled
        ):
            return ThinkingProjection.refused()

        entry = profile.classification("thinking.enabled", "thinking.type")
        if entry.status is CapabilityStatus.UNSUPPORTED:
            # The same refusal W3-AA raises at preflight, restated for callers
            # that drive the engine directly through the substitution seam.
            raise UnsupportedCapabilityError(
                "Anthropic thinking.enabled at thinking.type",
                f"{entry.detail} Owner: {entry.owner}.",
            )

        owner = self._thinking_signature_owner
        if owner is None:
            # A profile that claims enabled thinking without naming anything
            # that can sign a block has claimed a capability, not proved one.
            # ``budget_tokens`` is refused rather than accepted as decoration.
            raise UnsupportedCapabilityError(
                "Anthropic thinking.enabled at thinking.type",
                "The capability profile admits extended thinking but no "
                "signature owner is bound to this engine, so a thinking block "
                "could not carry the integrity signature the protocol "
                "requires, and budget_tokens has nothing enforcing it.",
            )
        return ThinkingProjection.signed_by(owner)

    async def generate(
        self,
        request: MessagesRequest,
        *,
        admission: CapabilityAdmission | None = None,
    ) -> MessagesResponse:
        """Run one turn and return the terminal Anthropic message."""

        projector, events = self._prepare(request, admission)
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
        self,
        request: MessagesRequest,
        *,
        admission: CapabilityAdmission | None = None,
    ) -> AsyncIterator[AnthropicStreamEvent]:
        """Run one turn and yield its Anthropic streaming lifecycle."""

        projector, events = self._prepare(request, admission)
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


def _admitted_service_tier(
    request: MessagesRequest,
    profile: AnthropicCapabilityProfile,
) -> ResponseServiceTier:
    """Resolve the tier that will actually be reported for this turn.

    A requested tier the profile does not admit is a field-specific refusal,
    not a silent downgrade: the client learns that ``service_tier`` was the
    problem rather than receiving a turn served by an unreported lane.
    """

    if request.service_tier is None:
        return DELIVERED_SERVICE_TIER
    entry = profile.classification("service_tier", "service_tier")
    if entry.status is CapabilityStatus.UNSUPPORTED:
        raise UnsupportedCapabilityError(
            "Anthropic service_tier at service_tier",
            f"{entry.detail} Owner: {entry.owner}.",
        )
    return DELIVERED_SERVICE_TIER


__all__ = [
    "DELIVERED_SERVICE_TIER",
    "AnthropicMessagesEngine",
    "new_message_id",
]
