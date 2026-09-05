"""Protocol-level proofs for the Anthropic Messages projector.

These tests drive the projector with synthetic typed runtime events, so they
assert the wire contract itself rather than the behaviour of any model. No
MLX runtime, no network and no Anthropic SDK are involved.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from mlx_batch_server.chat.anthropic.anthropic_schema import (
    MessagesRequest,
    StopReason,
)
from mlx_batch_server.chat.anthropic.capabilities import (
    detached_profile,
    enforce_capabilities,
)
from mlx_batch_server.chat.anthropic.errors import (
    REQUEST_ID_FIELD,
    AnthropicAPIError,
    InferenceOwnerUnavailableError,
    UnsupportedCapabilityError,
)
from mlx_batch_server.chat.anthropic.messages_engine import AnthropicMessagesEngine
from mlx_batch_server.chat.anthropic.projector import (
    AnthropicMessageProjector,
    ThinkingProjection,
    ThinkingSignature,
)
from mlx_batch_server.chat.anthropic.request_mapper import build_turn
from mlx_batch_server.chat.anthropic.turn_source import (
    AnthropicTurn,
    clear_turn_source,
    register_turn_source,
    require_turn_source,
)
from mlx_batch_server.runtime.events import (
    REASONING_CONTENT_KIND,
    TEXT_CONTENT_KIND,
    ContentPartStarted,
    OutputItemCompleted,
    ReasoningDelta,
    TextCompleted,
    TextDelta,
    ToolCompleted,
    ToolDelta,
    TurnCompleted,
    TurnFailed,
    TurnStarted,
    UsageUpdate,
)

ALIAS = "qwen-flash"
PHYSICAL = "/Volumes/models/Qwen3-Next-80B-A3B-Instruct-4bit"


class _SigningOwner:
    """A signature owner that exists only inside this test module.

    No production owner is bound anywhere. Admitting thinking in a test is
    therefore an explicit act with a visible signer, which is the point: the
    projector cannot reach a thinking block without one.
    """

    def sign_thinking(
        self, *, message_id: str, index: int, thinking: str
    ) -> ThinkingSignature:
        del thinking
        return ThinkingSignature(
            owner="test-signer", value=f"sig::{message_id}::{index}"
        )


def _projector(
    *, thinking: ThinkingProjection | None = None
) -> AnthropicMessageProjector:
    return AnthropicMessageProjector(
        message_id="msg_test", model_alias=ALIAS, thinking=thinking
    )


def _admitted_projector() -> AnthropicMessageProjector:
    return _projector(thinking=ThinkingProjection.signed_by(_SigningOwner()))


def _drain(projector: AnthropicMessageProjector, events) -> list:
    emitted: list = []
    for event in events:
        emitted.extend(projector.observe(event))
    return emitted


def _of_type(emitted: list, name: str) -> list:
    return [event for event in emitted if event.type == name]


class _ScriptedTurnSource:
    """A turn source that replays a fixed typed event script."""

    def __init__(self, script) -> None:
        self.script = list(script)
        self.turns: list[AnthropicTurn] = []

    def stream(self, turn: AnthropicTurn):
        self.turns.append(turn)

        async def _iterator():
            for event in self.script:
                yield event

        return _iterator()


@pytest.fixture(autouse=True)
def _unbound_turn_source():
    clear_turn_source()
    yield
    clear_turn_source()


# ---------------------------------------------------------------------------
# Tool streaming
# ---------------------------------------------------------------------------


def test_tool_arguments_stream_as_partial_json_exactly_once():
    """Partial tool JSON assembles once, even when the turn re-reports it.

    The first ToolDelta carries no name, so the block cannot open yet; the
    buffered fragment must still reach the client exactly once, and the
    duplicate OutputItemCompleted must not replay the payload.
    """

    arguments = '{"city": "Kielce", "unit": "c"}'
    projector = _projector()
    emitted = _drain(
        projector,
        [
            TurnStarted(response_id="resp_1", model=PHYSICAL, created_at=1),
            ToolDelta(
                index=0,
                call_id="toolu_1",
                item_id="item_1",
                arguments_delta='{"city": ',
            ),
            ToolDelta(
                index=0,
                call_id="toolu_1",
                item_id="item_1",
                name="get_weather",
                arguments_delta='"Kielce", ',
            ),
            ToolDelta(
                index=0,
                call_id="toolu_1",
                item_id="item_1",
                arguments_delta='"unit": "c"}',
            ),
            ToolCompleted(
                index=0,
                call_id="toolu_1",
                item_id="item_1",
                name="get_weather",
                arguments=arguments,
            ),
            # The turn also completes the output item for the same call.
            OutputItemCompleted(
                kind="function_call",
                index=0,
                item_id="item_1",
                call_id="toolu_1",
                name="get_weather",
                arguments=arguments,
            ),
            TurnCompleted(
                finish_reason="tool_calls",
                usage=UsageUpdate(input_tokens=11, output_tokens=7, total_tokens=18),
            ),
        ],
    )

    starts = _of_type(emitted, "content_block_start")
    assert len(starts) == 1
    assert starts[0].content_block.type == "tool_use"
    assert starts[0].content_block.id == "toolu_1"
    assert starts[0].content_block.name == "get_weather"
    # Anthropic opens a tool block with an empty input object; the payload
    # arrives only through input_json_delta.
    assert starts[0].content_block.input == {}

    deltas = _of_type(emitted, "content_block_delta")
    assert [delta.delta.type for delta in deltas] == ["input_json_delta"] * len(deltas)
    assembled = "".join(delta.delta.partial_json for delta in deltas)
    assert assembled == arguments
    assert json.loads(assembled) == {"city": "Kielce", "unit": "c"}

    # Exactly one stop for the single block, and no duplicate from the
    # repeated completion.
    assert len(_of_type(emitted, "content_block_stop")) == 1
    assert all(event.index == 0 for event in _of_type(emitted, "content_block_stop"))

    message_delta = _of_type(emitted, "message_delta")
    assert len(message_delta) == 1
    assert message_delta[0].delta.stop_reason is StopReason.TOOL_USE
    assert message_delta[0].usage.output_tokens == 7
    assert message_delta[0].usage.input_tokens == 11
    assert [event.type for event in emitted[-2:]] == ["message_delta", "message_stop"]

    terminal = projector.terminal_message()
    assert [block.type for block in terminal.content] == ["tool_use"]
    assert terminal.content[0].input == {"city": "Kielce", "unit": "c"}
    assert terminal.stop_reason is StopReason.TOOL_USE


def test_truncated_tool_call_reports_max_tokens_not_tool_use():
    """Truncation outranks tool use when the turn was cut off."""

    projector = _projector()
    emitted = _drain(
        projector,
        [
            TurnStarted(response_id="resp_2", model=PHYSICAL, created_at=1),
            ToolDelta(
                index=0,
                call_id="toolu_2",
                item_id="item_2",
                name="search",
                arguments_delta='{"q": "unfinis',
            ),
            TurnCompleted(finish_reason="length"),
        ],
    )

    assert _of_type(emitted, "message_delta")[0].delta.stop_reason is (
        StopReason.MAX_TOKENS
    )
    # An open block is still closed before the message ends.
    assert len(_of_type(emitted, "content_block_stop")) == 1


def test_exact_stop_sequence_projects_identically_to_stream_and_envelope() -> None:
    projector = _projector()
    emitted = _drain(
        projector,
        [
            TurnStarted(response_id="resp_stop", model=PHYSICAL, created_at=1),
            TurnCompleted(
                finish_reason="stop_sequence",
                stop_sequence="Exact END",
            ),
        ],
    )

    delta = _of_type(emitted, "message_delta")[0].delta
    assert delta.stop_reason is StopReason.STOP_SEQUENCE
    assert delta.stop_sequence == "Exact END"
    terminal = projector.terminal_message()
    assert terminal.stop_reason is StopReason.STOP_SEQUENCE
    assert terminal.stop_sequence == "Exact END"


def test_reasoning_and_text_never_share_a_block():
    """Thinking and output text stay on separate, non-duplicating channels.

    Admitted thinking is the only tier where this question can be asked at
    all; the unadmitted tiers emit no thinking to confuse with text, and
    ``test_anthropic_thinking_integrity`` owns that proof.
    """

    projector = _admitted_projector()
    emitted = _drain(
        projector,
        [
            TurnStarted(response_id="resp_3", model=PHYSICAL, created_at=1),
            ContentPartStarted(
                kind=REASONING_CONTENT_KIND,
                output_index=0,
                content_index=0,
                item_id="item_r",
            ),
            ReasoningDelta(
                delta="weighing options",
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
            TextDelta(delta="Hello", item_id="item_t", output_index=1, content_index=0),
            # The completion repeats the whole text; only the unseen tail may
            # be emitted, otherwise the client accumulates "HelloHello there".
            TextCompleted(
                text="Hello there",
                item_id="item_t",
                output_index=1,
                content_index=0,
            ),
            TurnCompleted(finish_reason="stop"),
        ],
    )

    thinking_deltas = [
        event
        for event in _of_type(emitted, "content_block_delta")
        if event.delta.type == "thinking_delta"
    ]
    text_deltas = [
        event
        for event in _of_type(emitted, "content_block_delta")
        if event.delta.type == "text_delta"
    ]
    assert "".join(delta.delta.thinking for delta in thinking_deltas) == (
        "weighing options"
    )
    assert "".join(delta.delta.text for delta in text_deltas) == "Hello there"
    assert {delta.index for delta in thinking_deltas} == {0}
    assert {delta.index for delta in text_deltas} == {1}

    terminal = projector.terminal_message()
    assert [block.type for block in terminal.content] == ["thinking", "text"]
    assert terminal.content[1].text == "Hello there"
    assert terminal.stop_reason is StopReason.END_TURN
    # The admitted block is signed, and the signature travelled on the wire
    # before the block closed.
    assert terminal.content[0].signature == "sig::msg_test::0"
    signature_deltas = [
        event
        for event in _of_type(emitted, "content_block_delta")
        if event.delta.type == "signature_delta"
    ]
    assert [delta.delta.signature for delta in signature_deltas] == ["sig::msg_test::0"]


def test_runtime_alias_never_leaks_the_physical_model():
    """The alias a client asked for is the alias it sees start to finish."""

    projector = _projector()
    emitted = _drain(
        projector,
        [
            TurnStarted(response_id="resp_4", model=PHYSICAL, created_at=1),
            ContentPartStarted(
                kind=TEXT_CONTENT_KIND,
                output_index=0,
                content_index=0,
                item_id="item_t",
            ),
            TextDelta(delta="hi", item_id="item_t", output_index=0, content_index=0),
            TurnCompleted(finish_reason="stop"),
        ],
    )

    start = _of_type(emitted, "message_start")[0]
    assert start.message.model == ALIAS
    assert PHYSICAL not in json.dumps(start.message.model_dump(mode="json"))
    assert projector.terminal_message().model == ALIAS


# ---------------------------------------------------------------------------
# Typed error behaviour
# ---------------------------------------------------------------------------


def test_turn_failure_projects_a_documented_error_event():
    """An undocumented runtime code fails closed onto ``api_error``."""

    projector = _projector()
    emitted = _drain(
        projector,
        [
            TurnStarted(response_id="resp_5", model=PHYSICAL, created_at=1),
            TurnFailed(error="backend exploded", code="internal_error"),
        ],
    )

    errors = _of_type(emitted, "error")
    assert len(errors) == 1
    assert errors[0].error.type == "api_error"
    assert errors[0].error.message == "backend exploded"
    assert projector.stopped is True

    with pytest.raises(AnthropicAPIError) as raised:
        projector.terminal_message()
    assert raised.value.error_type == "api_error"
    assert raised.value.status_code == 500


def test_error_payload_carries_type_status_and_request_id():
    """Every failure body is correlatable and carries a documented type."""

    error = AnthropicAPIError("no capacity", error_type="overloaded_error")
    payload = error.payload("req_abc")

    assert payload["type"] == "error"
    assert payload["error"]["type"] == "overloaded_error"
    assert payload[REQUEST_ID_FIELD] == "req_abc"
    assert error.status_code == 529

    # An invented type is refused rather than forwarded to the client.
    assert AnthropicAPIError("x", error_type="teapot_error").error_type == "api_error"


def test_unknown_request_field_fails_closed():
    """A field this runtime cannot honour is an error, not a silent no-op."""

    with pytest.raises(ValidationError):
        MessagesRequest.model_validate(
            {
                "model": ALIAS,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
                "not_an_anthropic_field": "srv_unsupported",
            }
        )


def test_a_represented_field_still_needs_a_capability_owner():
    """Representation is not admission.

    ``container`` is now named by the schema so the preflight can refuse it
    precisely. Parsing it must not be mistaken for honouring it.
    """

    request = MessagesRequest.model_validate(
        {
            "model": ALIAS,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
            "container": "srv_unsupported",
        }
    )

    with pytest.raises(UnsupportedCapabilityError, match="container"):
        enforce_capabilities(request, detached_profile(ALIAS))


def test_image_content_maps_instead_of_being_dropped():
    """Direct media enters the canonical ABI instead of disappearing."""

    request = MessagesRequest.model_validate(
        {
            "model": ALIAS,
            "max_tokens": 16,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "aGk=",
                            },
                        }
                    ],
                }
            ],
        }
    )

    turn = build_turn(request)

    assert turn.messages == ({"role": "user", "content": ()},)
    assert turn.media[0]["type"] == "input_image"
    assert turn.media[0]["image_base64"] == "data:image/png;base64,aGk="
    assert turn.media[0]["_content_index"] == 0


def test_unbound_inference_owner_fails_closed():
    """With no typed inference owner the surface refuses, never improvises."""

    with pytest.raises(InferenceOwnerUnavailableError) as raised:
        require_turn_source()
    assert raised.value.error_type == "overloaded_error"
    assert raised.value.status_code == 529


def test_tool_choice_must_name_a_declared_tool():
    request = MessagesRequest.model_validate(
        {
            "model": ALIAS,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {
                    "name": "get_weather",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
            "tool_choice": {"type": "tool", "name": "not_declared"},
        }
    )

    with pytest.raises(AnthropicAPIError) as raised:
        build_turn(request)
    assert raised.value.error_type == "invalid_request_error"


@pytest.mark.parametrize(
    ("choice", "expected_choice", "expected_parallel"),
    [
        ({"type": "auto"}, "auto", True),
        ({"type": "any", "disable_parallel_tool_use": True}, "required", False),
        ({"type": "none"}, "none", True),
        (
            {"type": "tool", "name": "get_weather"},
            {"type": "function", "name": "get_weather"},
            True,
        ),
    ],
)
def test_tool_controls_are_normalized_to_the_shared_runtime_abi(
    choice, expected_choice, expected_parallel
):
    request = MessagesRequest.model_validate(
        {
            "model": ALIAS,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Look up weather",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
            "tool_choice": choice,
        }
    )

    turn = build_turn(request)

    assert turn.tools == (
        {
            "type": "function",
            "name": "get_weather",
            "description": "Look up weather",
            "parameters": {"type": "object", "properties": {}},
        },
    )
    assert turn.tool_choice == expected_choice
    assert turn.sampling["parallel_tool_calls"] is expected_parallel


# ---------------------------------------------------------------------------
# Engine over the seam
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_engine_stream_opens_and_closes_the_anthropic_lifecycle():
    source = _ScriptedTurnSource(
        [
            TurnStarted(response_id="resp_6", model=PHYSICAL, created_at=1),
            ContentPartStarted(
                kind=TEXT_CONTENT_KIND,
                output_index=0,
                content_index=0,
                item_id="item_t",
            ),
            TextDelta(delta="ok", item_id="item_t", output_index=0, content_index=0),
            TurnCompleted(
                finish_reason="stop",
                usage=UsageUpdate(input_tokens=3, output_tokens=1, total_tokens=4),
            ),
        ]
    )
    register_turn_source(source)
    request = MessagesRequest.model_validate(
        {
            "model": ALIAS,
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }
    )

    engine = AnthropicMessagesEngine()
    events = [event async for event in engine.generate_stream(request)]
    types = [event.type for event in events]

    assert types[0] == "message_start"
    assert types[-1] == "message_stop"
    assert types.count("message_start") == 1
    assert types.count("message_delta") == 1
    assert types.index("content_block_start") < types.index("content_block_stop")
    # Anthropic's max_tokens is normalized to the shared runtime ABI.
    assert source.turns[0].sampling["max_output_tokens"] == 32


@pytest.mark.asyncio
async def test_engine_reports_a_truncated_turn_instead_of_ending_silently():
    """A stream that stops without a terminal event says so."""

    source = _ScriptedTurnSource(
        [TurnStarted(response_id="resp_7", model=PHYSICAL, created_at=1)]
    )
    register_turn_source(source)
    request = MessagesRequest.model_validate(
        {
            "model": ALIAS,
            "max_tokens": 8,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }
    )

    engine = AnthropicMessagesEngine()
    events = [event async for event in engine.generate_stream(request)]
    assert events[-1].type == "error"
    assert events[-1].error.type == "api_error"


@pytest.mark.asyncio
async def test_engine_non_stream_raises_on_a_failed_turn():
    source = _ScriptedTurnSource(
        [
            TurnStarted(response_id="resp_8", model=PHYSICAL, created_at=1),
            TurnFailed(error="no capacity", code="overloaded_error", status_code=529),
        ]
    )
    register_turn_source(source)
    request = MessagesRequest.model_validate(
        {
            "model": ALIAS,
            "max_tokens": 8,
            "messages": [{"role": "user", "content": "hi"}],
        }
    )

    with pytest.raises(AnthropicAPIError) as raised:
        await AnthropicMessagesEngine().generate(request)
    assert raised.value.error_type == "overloaded_error"


def _tool_result_request(
    *,
    result_content: object = "boom",
    is_error: bool = True,
    extra_user_blocks: list | None = None,
    tool_use_id: str = "toolu_1",
) -> MessagesRequest:
    user_content = [
        {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": result_content,
            "is_error": is_error,
        }
    ]
    user_content.extend(extra_user_blocks or [])
    return MessagesRequest.model_validate(
        {
            "model": ALIAS,
            "max_tokens": 8,
            "messages": [
                {"role": "user", "content": "lookup Kielce"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "lookup",
                            "input": {"city": "Kielce"},
                        }
                    ],
                },
                {"role": "user", "content": user_content},
            ],
        }
    )


def test_tool_results_map_to_tool_messages_without_losing_the_error_flag():
    request = _tool_result_request(
        extra_user_blocks=[{"type": "text", "text": "what now?"}]
    )

    turn = build_turn(request)
    roles = [message["role"] for message in turn.messages]
    assert roles == ["user", "assistant", "tool", "user"]
    receipt = turn.messages[2]
    assert receipt["type"] == "function_call_output"
    assert receipt["call_id"] == "toolu_1"
    assert receipt["output"] == "boom"
    assert receipt["is_error"] is True
    assert receipt["content"] == ({"type": "input_text", "text": "boom"},)
    assert turn.messages[3]["content"] == ({"type": "input_text", "text": "what now?"},)


def test_explicit_is_error_false_is_preserved_and_success_text_is_unchanged():
    request = _tool_result_request(result_content="15 degrees", is_error=False)

    turn = build_turn(request)
    receipt = turn.messages[2]
    assert receipt["is_error"] is False
    assert receipt["output"] == "15 degrees"
    assert receipt["content"] == ({"type": "input_text", "text": "15 degrees"},)


def test_tool_result_after_user_text_is_refused_not_reordered():
    request = MessagesRequest.model_validate(
        {
            "model": ALIAS,
            "max_tokens": 8,
            "messages": [
                {"role": "user", "content": "lookup"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "lookup",
                            "input": {},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "ignore this order"},
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "ok",
                        },
                    ],
                },
            ],
        }
    )

    with pytest.raises(AnthropicAPIError, match="must precede later text") as raised:
        build_turn(request)
    assert raised.value.error_type == "invalid_request_error"


def test_orphan_duplicate_and_mismatched_tool_results_fail_closed():
    orphan = MessagesRequest.model_validate(
        {
            "model": ALIAS,
            "max_tokens": 8,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_missing",
                            "content": "nope",
                        }
                    ],
                }
            ],
        }
    )
    with pytest.raises(AnthropicAPIError, match="no preceding tool_use"):
        build_turn(orphan)

    duplicate = MessagesRequest.model_validate(
        {
            "model": ALIAS,
            "max_tokens": 8,
            "messages": [
                {"role": "user", "content": "lookup"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "lookup",
                            "input": {},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "one",
                        },
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "two",
                        },
                    ],
                },
            ],
        }
    )
    with pytest.raises(AnthropicAPIError, match="duplicates"):
        build_turn(duplicate)

    mismatched = _tool_result_request(tool_use_id="toolu_other")
    with pytest.raises(AnthropicAPIError, match="do not match"):
        build_turn(mismatched)


def test_nested_tool_result_image_maps_through_canonical_media():
    request = _tool_result_request(
        result_content=[
            {"type": "text", "text": "see photo"},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "aW1hZ2U=",
                },
            },
        ],
        is_error=False,
    )

    turn = build_turn(request)
    receipt = turn.messages[2]
    assert receipt["output"] == "see photo"
    assert receipt["is_error"] is False
    assert len(turn.media) == 1
    assert turn.media[0]["type"] == "input_image"
    assert turn.media[0]["_role"] == "tool"
    assert turn.media[0]["_message_index"] == 2
    assert turn.media[0]["_content_index"] == 1
    assert turn.media[0]["image_base64"].startswith("data:image/png;base64,")


def test_nested_search_result_and_plaintext_document_fail_closed():
    search = _tool_result_request(
        result_content=[
            {
                "type": "search_result",
                "source": "https://example.test/doc",
                "title": "Doc",
                "content": [{"type": "text", "text": "hit"}],
            }
        ]
    )
    with pytest.raises(UnsupportedCapabilityError, match="search_result") as search_err:
        build_turn(search)
    assert search_err.value.error_type == "invalid_request_error"

    document = _tool_result_request(
        result_content=[
            {
                "type": "document",
                "source": {
                    "type": "text",
                    "media_type": "text/plain",
                    "data": "plain notes",
                },
            }
        ]
    )
    with pytest.raises(UnsupportedCapabilityError, match="plaintext document"):
        build_turn(document)


def test_nested_pdf_document_maps_to_canonical_file_media():
    request = _tool_result_request(
        result_content=[
            {
                "type": "document",
                "title": "lab.pdf",
                "source": {
                    "type": "url",
                    "url": "https://media.3more.ai/lab.pdf",
                },
            }
        ],
        is_error=False,
    )

    turn = build_turn(request)
    assert turn.media[0]["type"] == "input_file"
    assert turn.media[0]["file_url"] == "https://media.3more.ai/lab.pdf"
    assert turn.media[0]["filename"] == "lab.pdf"


def test_official_sdk_shaped_tool_results_parse():
    payload = {
        "model": ALIAS,
        "max_tokens": 8,
        "messages": [
            {"role": "user", "content": "lookup"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "lookup",
                        "input": {"q": "1"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "is_error": False,
                        "content": [
                            {"type": "text", "text": "ok"},
                            {
                                "type": "image",
                                "source": {
                                    "type": "url",
                                    "url": "https://media.3more.ai/a.png",
                                },
                            },
                        ],
                    }
                ],
            },
        ],
    }
    parsed = MessagesRequest.model_validate(payload)
    turn = build_turn(parsed)
    assert turn.messages[2]["is_error"] is False
    assert turn.media[0]["image_url"] == "https://media.3more.ai/a.png"
