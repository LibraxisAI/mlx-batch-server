"""Proofs that Anthropic thinking and service tier are truthful at the wire.

Three tiers exist for thinking on this runtime, and only the first two are
reachable in production:

omitted / disabled
    The turn runs. The runtime may reason as much as it likes on its own
    reasoning channel; not one ``thinking``, ``thinking_delta``,
    ``signature_delta`` or ``redacted_thinking`` reaches the client, and the
    dropped reasoning is not smuggled into visible text either.
enabled
    Refused before a single SSE byte exists, because no owner can enforce a
    token budget or sign a block.
admitted
    Reachable only by binding a signature owner, which nothing in this
    repository does. The tests below bind a local double to prove the shape
    of that tier — including that a *broken* signer makes the turn fail
    rather than emit an unsigned block.

Service tier is the same discipline in a smaller frame: ``auto`` and
``standard_only`` are both accepted, and both envelopes report the lane that
actually ran the turn.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from mlx_batch_server.chat.anthropic.anthropic_schema import (
    MessagesRequest,
    ResponseServiceTier,
)
from mlx_batch_server.chat.anthropic.capabilities import (
    AnthropicCapabilityProfile,
    CapabilityStatus,
    FieldClassification,
    detached_profile,
    enforce_capabilities,
)
from mlx_batch_server.chat.anthropic.errors import (
    AnthropicAPIError,
    UnsupportedCapabilityError,
)
from mlx_batch_server.chat.anthropic.messages_engine import (
    DELIVERED_SERVICE_TIER,
    AnthropicMessagesEngine,
)
from mlx_batch_server.chat.anthropic.projector import (
    AnthropicMessageProjector,
    ThinkingProjection,
    ThinkingSignature,
)
from mlx_batch_server.chat.anthropic.turn_source import (
    AnthropicTurn,
    clear_turn_source,
    register_turn_source,
)
from mlx_batch_server.main import app
from mlx_batch_server.runtime.events import (
    REASONING_CONTENT_KIND,
    TEXT_CONTENT_KIND,
    ContentPartStarted,
    ReasoningCompleted,
    ReasoningDelta,
    TextCompleted,
    TextDelta,
    TurnCompleted,
    TurnStarted,
    UsageUpdate,
)

ALIAS = "Qwen/Qwen3-0.6B-MLX-4bit"

#: Every wire shape that would tell a client this runtime produced extended
#: thinking. The point of the absent/disabled tiers is that none of these can
#: appear, whatever the runtime does on its own reasoning channel.
THINKING_WIRE_SHAPES = frozenset(
    {"thinking", "thinking_delta", "signature_delta", "redacted_thinking"}
)


class _AlwaysReasoningTurnSource:
    """A runtime that reasons on every turn, whatever the client asked for.

    This is the adversarial case: if the projector inferred protocol
    capability from the presence of a reasoning event, this owner would drag
    thinking onto the wire for requests that never asked for it.
    """

    def stream(self, turn: AnthropicTurn):
        del turn

        async def events():
            yield TurnStarted(response_id="resp_reasoning", model=ALIAS, created_at=1)
            yield ContentPartStarted(
                kind=REASONING_CONTENT_KIND,
                output_index=0,
                content_index=0,
                item_id="item_r",
            )
            yield ReasoningDelta(
                delta="deliberating privately",
                item_id="item_r",
                output_index=0,
                content_index=0,
            )
            yield ReasoningCompleted(
                text="deliberating privately",
                item_id="item_r",
                output_index=0,
                content_index=0,
            )
            yield ContentPartStarted(
                kind=TEXT_CONTENT_KIND,
                output_index=1,
                content_index=0,
                item_id="item_t",
            )
            yield TextDelta(
                delta="42", item_id="item_t", output_index=1, content_index=0
            )
            yield TextCompleted(
                text="42", item_id="item_t", output_index=1, content_index=0
            )
            yield TurnCompleted(
                finish_reason="stop",
                usage=UsageUpdate(input_tokens=5, output_tokens=1, total_tokens=6),
            )

        return events()


class _TestSignatureOwner:
    """The only thing in this repository that can sign a thinking block."""

    __test__ = False

    def sign_thinking(
        self, *, message_id: str, index: int, thinking: str
    ) -> ThinkingSignature:
        del thinking
        return ThinkingSignature(owner="tests.signature-double", value=f"sig-{index}")


class _EmptySignatureOwner:
    """The mutation the verifier has to catch: a signer that signs nothing."""

    __test__ = False

    def sign_thinking(
        self, *, message_id: str, index: int, thinking: str
    ) -> ThinkingSignature:
        del message_id, index, thinking
        return ThinkingSignature(owner="tests.signature-double", value="")


@pytest.fixture
def reasoning_client() -> Iterator[TestClient]:
    """An HTTP client whose bound runtime always emits reasoning events."""

    source = _AlwaysReasoningTurnSource()
    register_turn_source(source)
    try:
        yield TestClient(app)
    finally:
        clear_turn_source(source)


def _body(**overrides) -> dict:
    payload = {
        "model": ALIAS,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "What is 6 times 7?"}],
    }
    payload.update(overrides)
    return payload


def _sse_events(text: str) -> list[dict]:
    events = []
    for frame in text.split("\n\n"):
        for line in frame.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: ") :]))
    return events


def _projector(*, thinking: ThinkingProjection | None = None):
    return AnthropicMessageProjector(
        message_id="msg_integrity", model_alias=ALIAS, thinking=thinking
    )


def _drain(projector, events) -> list:
    emitted = []
    for event in events:
        emitted.extend(projector.observe(event))
    return emitted


def _reasoning_then_text() -> list:
    return [
        TurnStarted(response_id="resp_reasoning", model=ALIAS, created_at=1),
        ContentPartStarted(
            kind=REASONING_CONTENT_KIND,
            output_index=0,
            content_index=0,
            item_id="item_r",
        ),
        ReasoningDelta(
            delta="deliberating privately",
            item_id="item_r",
            output_index=0,
            content_index=0,
        ),
        ReasoningCompleted(
            text="deliberating privately",
            item_id="item_r",
            output_index=0,
            content_index=0,
        ),
        ContentPartStarted(
            kind=TEXT_CONTENT_KIND,
            output_index=1,
            content_index=0,
            item_id="item_t",
        ),
        TextDelta(delta="42", item_id="item_t", output_index=1, content_index=0),
        TextCompleted(text="42", item_id="item_t", output_index=1, content_index=0),
        TurnCompleted(finish_reason="stop"),
    ]


# ---------------------------------------------------------------------------
# The absent and disabled tiers emit nothing, at either transport
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "thinking"),
    [("omitted", None), ("disabled", {"type": "disabled"})],
)
def test_unrequested_thinking_never_reaches_the_projector(label, thinking):
    """A reasoning-emitting runtime produces zero thinking wire output."""

    del label, thinking  # the projector tier is the same for both requests
    projector = _projector()
    emitted = _drain(projector, _reasoning_then_text())

    assert not projector.emits_thinking
    for event in emitted:
        assert event.type != "content_block_start" or (
            event.content_block.type not in THINKING_WIRE_SHAPES
        )
        if event.type == "content_block_delta":
            assert event.delta.type not in THINKING_WIRE_SHAPES

    terminal = projector.terminal_message()
    assert [block.type for block in terminal.content] == ["text"]
    # The dropped reasoning is dropped, not relocated: the visible text is
    # exactly what the runtime produced as text.
    assert terminal.content[0].text == "42"


@pytest.mark.parametrize(
    ("label", "overrides"),
    [
        ("omitted", {}),
        ("disabled", {"thinking": {"type": "disabled"}}),
    ],
)
def test_no_thinking_on_the_non_stream_wire(reasoning_client, label, overrides):
    """The HTTP envelope carries no thinking block for either tier."""

    del label
    response = reasoning_client.post("/anthropic/v1/messages", json=_body(**overrides))

    assert response.status_code == 200
    payload = response.json()
    assert [block["type"] for block in payload["content"]] == ["text"]
    assert "deliberating privately" not in json.dumps(payload)


@pytest.mark.parametrize(
    ("label", "overrides"),
    [
        ("omitted", {}),
        ("disabled", {"thinking": {"type": "disabled"}}),
    ],
)
def test_no_thinking_on_the_sse_wire(reasoning_client, label, overrides):
    """The streaming lifecycle carries no thinking event for either tier."""

    del label
    response = reasoning_client.post(
        "/anthropic/v1/messages", json=_body(stream=True, **overrides)
    )

    assert response.status_code == 200
    events = _sse_events(response.text)
    assert events, "the stream produced no events at all"
    for event in events:
        assert event["type"] != "error"
        if event["type"] == "content_block_start":
            assert event["content_block"]["type"] not in THINKING_WIRE_SHAPES
        if event["type"] == "content_block_delta":
            assert event["delta"]["type"] not in THINKING_WIRE_SHAPES
    assert "deliberating privately" not in response.text


# ---------------------------------------------------------------------------
# The enabled tier is refused before SSE, and so is prior thinking
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stream", [False, True])
def test_enabled_budget_is_refused_before_any_stream_byte(reasoning_client, stream):
    """``budget_tokens`` nothing enforces is a 400, not silent decoration."""

    response = reasoning_client.post(
        "/anthropic/v1/messages",
        json=_body(stream=stream, thinking={"type": "enabled", "budget_tokens": 4096}),
    )

    assert response.status_code == 400
    assert "text/event-stream" not in response.headers.get("content-type", "")
    body = response.json()
    assert body["error"]["type"] == "invalid_request_error"
    assert "thinking.type" in body["error"]["message"]
    # The budget is never echoed back as though it had been recorded.
    assert "4096" not in response.text


@pytest.mark.parametrize(
    ("label", "block", "payload"),
    [
        (
            "thinking_continuation",
            {
                "type": "thinking",
                "thinking": "CARRIED-REASONING-9f2a",
                "signature": "CARRIED-SIGNATURE-9f2a",
            },
            "CARRIED-REASONING-9f2a",
        ),
        (
            "redacted_thinking",
            {"type": "redacted_thinking", "data": "CARRIED-REDACTED-9f2a"},
            "CARRIED-REDACTED-9f2a",
        ),
    ],
)
def test_prior_thinking_input_is_refused_not_concatenated(
    reasoning_client, label, block, payload
):
    """Replayed reasoning is refused at its exact wire location."""

    del label
    response = reasoning_client.post(
        "/anthropic/v1/messages",
        json=_body(
            stream=True,
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": [block]},
                {"role": "user", "content": "continue"},
            ],
        ),
    )

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["type"] == "invalid_request_error"
    assert "messages.1.content.0.type" in body["error"]["message"]
    # Refused, not folded into the prompt: the replayed payload itself never
    # comes back on the wire. (The refusal *explains* the capability in prose;
    # what must not appear is the client's own carried reasoning.)
    assert payload not in response.text


def test_local_profile_does_not_claim_enabled_thinking():
    """The truthful tier is a property of the profile, not of the tests."""

    profile = detached_profile(ALIAS)

    assert not profile.supports("thinking.enabled")
    assert not profile.supports("content.thinking")
    assert not profile.supports("content.redacted_thinking")
    assert profile.supports("thinking.disabled")


# ---------------------------------------------------------------------------
# A profile double that claims thinking still cannot produce a block
# ---------------------------------------------------------------------------


def _profile_claiming_thinking() -> AnthropicCapabilityProfile:
    """A future/hostile profile that admits enabled thinking on paper."""

    base = detached_profile(ALIAS)
    fields = dict(base.fields)
    fields["thinking.enabled"] = FieldClassification(
        key="thinking.enabled",
        status=CapabilityStatus.IMPLEMENTED,
        owner="a profile that claims more than it can prove",
    )
    return AnthropicCapabilityProfile(
        alias=base.alias,
        role=base.role,
        backend=base.backend,
        declared_capabilities=base.declared_capabilities,
        fields=fields,
    )


@pytest.mark.asyncio
async def test_claiming_profile_without_a_signer_is_still_refused():
    """A capability claim with no signed receipt buys nothing."""

    request = MessagesRequest.model_validate(
        _body(thinking={"type": "enabled", "budget_tokens": 4096})
    )
    admission = enforce_capabilities(request, _profile_claiming_thinking())
    engine = AnthropicMessagesEngine(turn_source=_AlwaysReasoningTurnSource())

    with pytest.raises(UnsupportedCapabilityError) as failure:
        await engine.generate(request, admission=admission)

    assert failure.value.status_code == 400
    assert failure.value.error_type == "invalid_request_error"
    assert "signature owner" in failure.value.message


@pytest.mark.asyncio
async def test_claiming_profile_with_a_signer_emits_one_signed_block():
    """Bound signature owner — and only then — thinking becomes projectable."""

    request = MessagesRequest.model_validate(
        _body(thinking={"type": "enabled", "budget_tokens": 4096})
    )
    admission = enforce_capabilities(request, _profile_claiming_thinking())
    engine = AnthropicMessagesEngine(
        turn_source=_AlwaysReasoningTurnSource(),
        thinking_signature_owner=_TestSignatureOwner(),
    )

    terminal = await engine.generate(request, admission=admission)

    thinking_blocks = [block for block in terminal.content if block.type == "thinking"]
    assert len(thinking_blocks) == 1
    assert thinking_blocks[0].thinking == "deliberating privately"
    assert thinking_blocks[0].signature == "sig-0"


# ---------------------------------------------------------------------------
# An empty signature is structurally impossible, and a broken signer fails closed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("owner", "value"),
    [("", "sig"), ("   ", "sig"), ("owner", ""), ("owner", "   ")],
)
def test_empty_signature_cannot_be_constructed(owner, value):
    """The signature type refuses to exist without both halves."""

    with pytest.raises(ValueError):
        ThinkingSignature(owner=owner, value=value)


def test_a_signer_returning_an_empty_signature_fails_the_turn():
    """The mutation the verifier exists to catch turns the turn red."""

    projector = _projector(
        thinking=ThinkingProjection.signed_by(_EmptySignatureOwner())
    )

    with pytest.raises(ValueError):
        _drain(projector, _reasoning_then_text())


def test_refused_projection_cannot_sign():
    """No admitted owner means no signature is obtainable at all."""

    with pytest.raises(AnthropicAPIError) as failure:
        ThinkingProjection.refused().sign(
            message_id="msg_integrity", index=0, thinking="anything"
        )

    assert failure.value.error_type == "api_error"


def test_a_non_signing_owner_fails_closed():
    """An owner that returns something other than a signature signs nothing."""

    class _Impostor:
        def sign_thinking(self, *, message_id: str, index: int, thinking: str):
            del message_id, index, thinking
            return "looks-like-a-signature"

    projector = _projector(thinking=ThinkingProjection.signed_by(_Impostor()))

    with pytest.raises(AnthropicAPIError) as failure:
        _drain(projector, _reasoning_then_text())

    assert failure.value.error_type == "api_error"


# ---------------------------------------------------------------------------
# Service tier reports what was served
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("requested", ["auto", "standard_only"])
def test_requested_tier_is_accepted_and_reported_as_standard(
    reasoning_client, requested
):
    """Both accepted preferences report the one lane that actually ran."""

    response = reasoning_client.post(
        "/anthropic/v1/messages", json=_body(service_tier=requested)
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["usage"]["service_tier"] == ResponseServiceTier.STANDARD.value
    # Never the requested preference echoed back as a delivered tier.
    assert payload["usage"]["service_tier"] != requested


@pytest.mark.parametrize("requested", ["auto", "standard_only"])
def test_opening_envelope_reports_the_delivered_tier(reasoning_client, requested):
    """message_start says the same thing the terminal envelope will say."""

    response = reasoning_client.post(
        "/anthropic/v1/messages", json=_body(stream=True, service_tier=requested)
    )

    assert response.status_code == 200
    events = _sse_events(response.text)
    opening = next(event for event in events if event["type"] == "message_start")
    assert (
        opening["message"]["usage"]["service_tier"]
        == ResponseServiceTier.STANDARD.value
    )


@pytest.mark.asyncio
async def test_a_profile_that_refuses_service_tier_is_a_field_specific_400():
    """A tier the profile does not admit is refused, never silently changed."""

    base = detached_profile(ALIAS)
    fields = dict(base.fields)
    fields["service_tier"] = FieldClassification(
        key="service_tier",
        status=CapabilityStatus.UNSUPPORTED,
        owner="a runtime with no capacity lanes",
        detail="No lane exists.",
    )
    profile = AnthropicCapabilityProfile(
        alias=base.alias,
        role=base.role,
        backend=base.backend,
        declared_capabilities=base.declared_capabilities,
        fields=fields,
    )
    request = MessagesRequest.model_validate(_body(service_tier="auto"))

    with pytest.raises(UnsupportedCapabilityError) as failure:
        enforce_capabilities(request, profile)

    assert failure.value.status_code == 400
    assert "service_tier" in failure.value.message


def test_delivered_tier_is_the_only_lane_this_process_serves():
    """The constant is the single place the delivered lane is decided."""

    assert DELIVERED_SERVICE_TIER is ResponseServiceTier.STANDARD
