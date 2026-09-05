"""RED contracts for the sole temporal/finality writer."""

from __future__ import annotations

import asyncio
import gc
import threading
import time
from typing import Any, cast

import pytest

from mlx_batch_server.runtime.events import (
    ContentPartCompleted,
    ContentPartStarted,
    HostedCallCompleted,
    HostedCallProgress,
    HostedCallResult,
    HostedCallStarted,
    HostedCitation,
    OutputItemCompleted,
    OutputItemStarted,
    ProgressUpdate,
    ReasoningCompleted,
    ReasoningDelta,
    TextCompleted,
    TextDelta,
    ToolCompleted,
    ToolDelta,
    TurnCancelled,
    TurnCompleted,
    TurnEvent,
    TurnFailed,
    TurnStarted,
    UsageUpdate,
)
from mlx_batch_server.runtime.turn import (
    MAX_CITATIONS_PER_ITEM,
    GenerationTurn,
    TurnProducerOverflow,
    TurnState,
    TurnSubscriberLimit,
    TurnSubscriberOverflow,
)


def _start(turn: GenerationTurn, response_id: str = "resp_test") -> None:
    turn.start(TurnStarted(response_id=response_id, model="model", created_at=1))


def _complete_text_item(
    turn: GenerationTurn,
    *,
    output_index: int,
    item_id: str,
    text: str,
) -> None:
    turn.emit(OutputItemStarted("message", output_index, item_id))
    turn.emit(ContentPartStarted("output_text", output_index, 0, item_id))
    turn.emit(TextDelta(text, item_id, output_index, 0))
    turn.emit(TextCompleted(text, item_id, output_index, 0))
    turn.emit(ContentPartCompleted("output_text", output_index, 0, item_id, text))
    turn.emit(OutputItemCompleted("message", output_index, item_id, text=text))


@pytest.mark.asyncio
async def test_full_responses_lifecycle_has_stable_item_and_content_identity() -> None:
    turn = GenerationTurn(max_pending_events=32)
    stream = turn.subscribe()
    _start(turn)

    turn.emit(OutputItemStarted("reasoning", 0, "reasoning_0"))
    turn.emit(
        ContentPartStarted(
            "reasoning_summary_text",
            0,
            0,
            "reasoning_0",
        )
    )
    turn.emit(ReasoningDelta("because ", "reasoning_0", 0, 0))
    turn.emit(ReasoningDelta("evidence", "reasoning_0", 0, 0))
    turn.emit(ReasoningCompleted("because evidence", "reasoning_0", 0, 0))
    turn.emit(
        ContentPartCompleted(
            "reasoning_summary_text",
            0,
            0,
            "reasoning_0",
            "because evidence",
        )
    )
    turn.emit(
        OutputItemCompleted("reasoning", 0, "reasoning_0", text="because evidence")
    )

    _complete_text_item(
        turn,
        output_index=1,
        item_id="message_1",
        text="answer",
    )

    turn.emit(OutputItemStarted("function_call", 2, "tool_2", "call_2", "search"))
    turn.emit(ToolDelta(2, "call_2", "tool_2", "search", '{"q":'))
    turn.emit(ToolDelta(2, "call_2", "tool_2", arguments_delta='"mlx"}'))
    turn.emit(ToolCompleted(2, "call_2", "tool_2", "search", '{"q":"mlx"}'))
    turn.emit(
        OutputItemCompleted(
            "function_call",
            2,
            "tool_2",
            call_id="call_2",
            name="search",
            arguments='{"q":"mlx"}',
        )
    )

    usage = UsageUpdate(input_tokens=7, output_tokens=5, total_tokens=12)
    turn.emit(usage)
    turn.complete(TurnCompleted("stop", usage=usage))

    observed = [event async for event in stream]
    assert [item.sequence_number for item in observed] == list(range(len(observed)))
    assert isinstance(observed[-1].event, TurnCompleted)
    assert turn.usage == usage
    assert turn.state is TurnState.TERMINAL
    assert turn.subscriber_count == 0


@pytest.mark.asyncio
async def test_event_order_rejects_unscoped_duplicate_and_incomplete_flows() -> None:
    turn = GenerationTurn(max_pending_events=32)
    _start(turn)
    turn.emit(OutputItemStarted("message", 0, "message_0"))

    with pytest.raises(RuntimeError, match="content part 0 has not been started"):
        turn.emit(TextDelta("early", "message_0", 0, 0))
    with pytest.raises(RuntimeError, match="contiguous"):
        turn.emit(OutputItemStarted("message", 0, "duplicate_index"))

    turn.emit(ContentPartStarted("output_text", 0, 0, "message_0"))
    with pytest.raises(RuntimeError, match="kind does not match"):
        turn.emit(ReasoningDelta("wrong channel", "message_0", 0, 0))
    turn.emit(TextDelta("one", "message_0", 0, 0))
    with pytest.raises(RuntimeError, match="does not match emitted deltas"):
        turn.emit(TextCompleted("wrong", "message_0", 0, 0))
    with pytest.raises(RuntimeError, match="text done event first"):
        turn.emit(ContentPartCompleted("output_text", 0, 0, "message_0", ""))
    with pytest.raises(RuntimeError, match="every output item"):
        turn.complete(TurnCompleted("stop"))

    turn.emit(TextDelta(" two", "message_0", 0, 0))
    turn.emit(TextCompleted("one two", "message_0", 0, 0))
    turn.emit(ContentPartCompleted("output_text", 0, 0, "message_0", "one two"))
    turn.emit(OutputItemCompleted("message", 0, "message_0", text="one two"))
    with pytest.raises(RuntimeError, match="duplicate output item id"):
        turn.emit(OutputItemStarted("message", 1, "message_0"))
    turn.complete(TurnCompleted("stop"))
    assert turn.terminal_event is not None
    assert turn.terminal_event.sequence_number == 8


@pytest.mark.asyncio
async def test_tool_done_requires_a_stable_call_identity_and_exact_arguments() -> None:
    turn = GenerationTurn(max_pending_events=16)
    _start(turn)
    turn.emit(OutputItemStarted("function_call", 0, "tool_0", "call_0", "search"))

    with pytest.raises(RuntimeError, match="preceding tool delta"):
        turn.emit(ToolCompleted(0, "call_0", "tool_0", "search", "{}"))
    with pytest.raises(RuntimeError, match="first tool delta"):
        turn.emit(ToolDelta(0, "call_0", "tool_0", arguments_delta="{"))

    turn.emit(ToolDelta(0, "call_0", "tool_0", "search", "{"))
    with pytest.raises(RuntimeError, match="call id changed"):
        turn.emit(ToolDelta(0, "foreign", "tool_0", arguments_delta="}"))
    with pytest.raises(RuntimeError, match="do not match emitted deltas"):
        turn.emit(ToolCompleted(0, "call_0", "tool_0", "search", "{}"))

    turn.emit(ToolDelta(0, "call_0", "tool_0", arguments_delta="}"))
    turn.emit(ToolCompleted(0, "call_0", "tool_0", "search", "{}"))
    turn.emit(
        OutputItemCompleted(
            "function_call",
            0,
            "tool_0",
            call_id="call_0",
            name="search",
            arguments="{}",
        )
    )
    turn.complete(TurnCompleted("tool_call"))


@pytest.mark.asyncio
async def test_usage_updates_are_monotonic_and_terminal_usage_matches() -> None:
    turn = GenerationTurn(max_pending_events=8)
    _start(turn)
    first = UsageUpdate(4, 1, 5, cached_input_tokens=2)
    final = UsageUpdate(
        4,
        3,
        7,
        cached_input_tokens=2,
        reasoning_output_tokens=1,
    )
    turn.emit(first)

    with pytest.raises(RuntimeError, match="cumulative and monotonic"):
        turn.emit(UsageUpdate(3, 2, 5))
    with pytest.raises(RuntimeError, match="cumulative and monotonic"):
        turn.emit(UsageUpdate(4, 2, 6, cached_input_tokens=1))
    turn.emit(final)
    with pytest.raises(RuntimeError, match="must equal the latest UsageUpdate"):
        turn.complete(TurnCompleted("stop", usage=first))

    turn.complete(TurnCompleted("stop", usage=final))
    assert turn.usage is final


@pytest.mark.asyncio
async def test_terminal_only_usage_becomes_authoritative() -> None:
    usage = UsageUpdate(3, 2, 5)
    turn = GenerationTurn(max_pending_events=2)
    _start(turn)
    turn.complete(TurnCompleted("stop", usage=usage))
    assert turn.usage is usage


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method_name", "terminal_event"),
    [
        ("complete", TurnCompleted("stop")),
        ("fail", TurnFailed("backend failed", code="backend_error")),
        ("cancel", TurnCancelled("client_cancelled")),
    ],
)
async def test_each_terminal_path_is_exactly_once(
    method_name: str,
    terminal_event: TurnCompleted | TurnFailed | TurnCancelled,
) -> None:
    turn = GenerationTurn(max_pending_events=4)
    stream = turn.subscribe()
    _start(turn, "resp_terminal")
    getattr(turn, method_name)(terminal_event)

    observed = [item async for item in stream]
    assert turn.state is TurnState.TERMINAL
    assert turn.terminal_event is observed[-1]
    assert observed[-1].event is terminal_event
    with pytest.raises(RuntimeError, match="terminal event already emitted"):
        turn.fail(TurnFailed("second terminal"))


@pytest.mark.asyncio
async def test_idle_failure_is_the_only_terminal_allowed_before_start() -> None:
    turn = GenerationTurn(max_pending_events=2)
    with pytest.raises(RuntimeError, match="only TurnFailed"):
        turn.complete(TurnCompleted("stop"))
    with pytest.raises(RuntimeError, match="only TurnFailed"):
        turn.cancel(TurnCancelled("cancelled_before_start"))

    turn.fail(TurnFailed("load failed", code="load_failed"))
    observed = [item async for item in turn.subscribe()]
    assert [item.sequence_number for item in observed] == [0]
    assert isinstance(observed[0].event, TurnFailed)


@pytest.mark.asyncio
async def test_slow_subscriber_overflow_is_explicit_and_reclaimed() -> None:
    turn = GenerationTurn(max_pending_events=8)
    slow = turn.subscribe(max_pending_events=2)
    healthy = turn.subscribe(max_pending_events=8)
    _start(turn, "resp_slow")
    turn.emit(ProgressUpdate("prefill"))
    turn.emit(ProgressUpdate("decode"))
    assert turn.subscriber_count == 1
    turn.complete(TurnCompleted("stop"))
    assert turn.subscriber_count == 0

    with pytest.raises(TurnSubscriberOverflow):
        _ = [event async for event in slow]
    observed = [event async for event in healthy]
    assert isinstance(observed[-1].event, TurnCompleted)


@pytest.mark.asyncio
async def test_subscriber_count_is_bounded_and_abandoned_subscribers_reclaim() -> None:
    turn = GenerationTurn(max_subscribers=1)
    first = turn.subscribe()
    with pytest.raises(TurnSubscriberLimit, match="subscriber limit"):
        turn.subscribe()
    await first.aclose()
    assert turn.subscriber_count == 0

    abandoned = turn.subscribe()
    assert turn.subscriber_count == 1
    del abandoned
    gc.collect()
    assert turn.subscriber_count == 0


@pytest.mark.asyncio
async def test_foreign_thread_invalid_transition_is_acknowledged_to_producer() -> None:
    turn = GenerationTurn(max_pending_events=4)
    _start(turn, "resp_thread_error")

    with pytest.raises(RuntimeError, match="has not been started"):
        await asyncio.to_thread(
            turn.emit,
            TextDelta("invalid", "missing_item", 0, 0),
        )
    assert turn.terminal_event is None
    turn.complete(TurnCompleted("stop"))


@pytest.mark.asyncio
async def test_foreign_thread_terminal_race_acknowledges_one_loser() -> None:
    turn = GenerationTurn(max_pending_events=8, max_thread_bridge_events=2)
    _start(turn, "resp_thread_terminal")

    results = await asyncio.gather(
        asyncio.to_thread(turn.complete, TurnCompleted("stop")),
        asyncio.to_thread(turn.fail, TurnFailed("lost race")),
        return_exceptions=True,
    )
    assert sum(result is None for result in results) == 1
    errors = [result for result in results if isinstance(result, BaseException)]
    assert len(errors) == 1
    assert "terminal event already emitted" in str(errors[0])


@pytest.mark.asyncio
async def test_foreign_thread_bridge_is_bounded_before_owner_acknowledgement() -> None:
    turn = GenerationTurn(max_thread_bridge_events=1)
    _start(turn, "resp_thread_bound")
    first_errors: list[BaseException] = []
    second_errors: list[BaseException] = []

    def emit(detail: str, errors: list[BaseException]) -> None:
        try:
            turn.emit(ProgressUpdate("decode", {"detail": detail}))
        except BaseException as error:
            errors.append(error)

    first = threading.Thread(target=emit, args=("first", first_errors))
    first.start()
    deadline = time.monotonic() + 1
    while turn.pending_thread_events != 1 and time.monotonic() < deadline:
        threading.Event().wait(0.001)
    assert turn.pending_thread_events == 1

    second = threading.Thread(target=emit, args=("second", second_errors))
    second.start()
    second.join(timeout=1)
    assert not second.is_alive()
    assert len(second_errors) == 1
    assert isinstance(second_errors[0], TurnProducerOverflow)

    await asyncio.sleep(0)
    first.join(timeout=1)
    assert not first.is_alive()
    assert first_errors == []
    assert turn.pending_thread_events == 0
    turn.complete(TurnCompleted("stop"))


@pytest.mark.asyncio
async def test_replay_payloads_are_deeply_immutable() -> None:
    mutable_detail = {"nested": {"values": [1, 2]}}
    mutable_stats = {"cache": {"tiers": ["ssd", "paged"]}}
    progress = ProgressUpdate("decode", mutable_detail)
    completed = TurnCompleted("stop", backend_stats=mutable_stats)
    mutable_detail["nested"]["values"].append(3)
    mutable_stats["cache"]["tiers"].append("prefix")

    assert progress.detail["nested"]["values"] == (1, 2)
    assert completed.backend_stats["cache"]["tiers"] == ("ssd", "paged")
    with pytest.raises(TypeError):
        cast("dict[str, Any]", progress.detail)["new"] = "mutation"
    with pytest.raises(TypeError):
        cast("dict[str, Any]", progress.detail["nested"])["new"] = "mutation"

    turn = GenerationTurn(max_pending_events=4)
    _start(turn, "resp_immutable")
    turn.emit(progress)
    turn.complete(completed)
    replay = [item async for item in turn.subscribe()]
    replayed = cast("ProgressUpdate", replay[1].event)
    assert replayed.detail["nested"]["values"] == (1, 2)


def test_turn_completed_stop_sequence_invariants_are_typed() -> None:
    completed = TurnCompleted("stop_sequence", stop_sequence="Exact END")

    assert completed.stop_sequence == "Exact END"
    assert TurnCompleted("stop_sequence", stop_sequence="  ").stop_sequence == "  "
    with pytest.raises(ValueError, match="stop_sequence must not be empty"):
        TurnCompleted("stop_sequence")
    with pytest.raises(ValueError, match="non-stop completion"):
        TurnCompleted("stop", stop_sequence="END")


@pytest.mark.asyncio
async def test_late_subscriber_gets_bounded_start_recent_and_terminal_replay() -> None:
    turn = GenerationTurn(max_pending_events=8, replay_events=4)
    _start(turn, "resp_replay")
    for index in range(6):
        turn.emit(ProgressUpdate("decode", {"index": index}))
    turn.complete(TurnCompleted("stop"))

    observed = [item async for item in turn.subscribe(max_pending_events=3)]
    assert [item.sequence_number for item in observed] == [0, 6, 7]
    assert isinstance(observed[0].event, TurnStarted)
    assert isinstance(observed[-1].event, TurnCompleted)


@pytest.mark.asyncio
async def test_bounds_accept_one_and_reject_zero() -> None:
    turn = GenerationTurn(
        max_pending_events=1,
        replay_events=1,
        max_subscribers=1,
        max_thread_bridge_events=1,
    )
    stream = turn.subscribe(max_pending_events=1)
    await stream.aclose()

    with pytest.raises(ValueError, match="max_pending_events"):
        GenerationTurn(max_pending_events=0)
    with pytest.raises(ValueError, match="replay_events"):
        GenerationTurn(replay_events=0)
    with pytest.raises(ValueError, match="max_subscribers"):
        GenerationTurn(max_subscribers=0)
    with pytest.raises(ValueError, match="max_thread_bridge_events"):
        GenerationTurn(max_thread_bridge_events=0)
    with pytest.raises(ValueError, match="max_pending_events"):
        turn.subscribe(max_pending_events=0)


def test_event_payload_validation_fails_closed() -> None:
    with pytest.raises(ValueError, match="total_tokens"):
        UsageUpdate(2, 2, 5)
    with pytest.raises(ValueError, match="item_id"):
        TextDelta("text", "", 0, 0)
    with pytest.raises(ValueError, match="unsupported output item kind"):
        OutputItemStarted("unknown", 0, "item")
    with pytest.raises(TypeError, match="mapping keys"):
        ProgressUpdate("decode", cast("dict[str, Any]", {1: "invalid"}))
    with pytest.raises(TypeError, match="not JSON-compatible"):
        ProgressUpdate("decode", {"invalid": {"set"}})


@pytest.mark.asyncio
async def test_closed_event_union_rejects_foreign_objects() -> None:
    turn = GenerationTurn()
    with pytest.raises(TypeError, match="TurnEvent required"):
        turn.emit(cast("TurnEvent", object()))


def test_function_call_item_start_requires_immutable_tool_identity() -> None:
    """A tool call is admitted with its identity or not admitted at all."""

    with pytest.raises(ValueError, match="requires call_id and name"):
        OutputItemStarted("function_call", 0, "tool_0")
    with pytest.raises(ValueError, match="requires call_id and name"):
        OutputItemStarted("function_call", 0, "tool_0", "call_0", None)
    with pytest.raises(ValueError, match="requires call_id and name"):
        OutputItemStarted("function_call", 0, "tool_0", None, "search")
    with pytest.raises(ValueError, match="call_id"):
        OutputItemStarted("function_call", 0, "tool_0", "  ", "search")
    with pytest.raises(ValueError, match="name"):
        OutputItemStarted("function_call", 0, "tool_0", "call_0", " ")

    started = OutputItemStarted("function_call", 0, "tool_0", "call_0", "search")
    assert (started.call_id, started.name) == ("call_0", "search")
    with pytest.raises(AttributeError):
        started.call_id = "other"  # type: ignore[misc]


def test_message_and_reasoning_starts_cannot_carry_tool_identity() -> None:
    for kind in ("message", "reasoning"):
        with pytest.raises(ValueError, match="cannot carry tool identity"):
            OutputItemStarted(kind, 0, f"{kind}_0", "call_0", "search")
        with pytest.raises(ValueError, match="cannot carry tool identity"):
            OutputItemStarted(kind, 0, f"{kind}_0", name="search")
        assert OutputItemStarted(kind, 0, f"{kind}_0").call_id is None


def test_turn_started_keeps_public_alias_and_physical_identity_apart() -> None:
    physical = TurnStarted("resp_1", "grant-ai/Qwen3.8-Flash-Next", 7)
    assert physical.public_model == "grant-ai/Qwen3.8-Flash-Next"

    aliased = TurnStarted(
        "resp_1",
        "grant-ai/Qwen3.8-Flash-Next",
        7,
        requested_model="buddy",
        request_settings={"tools": [], "tool_choice": "auto"},
    )
    assert aliased.public_model == "buddy"
    assert aliased.model == "grant-ai/Qwen3.8-Flash-Next"
    with pytest.raises(TypeError, match="immutable"):
        aliased.request_settings["tools"] = ["mutated"]  # type: ignore[index]
    with pytest.raises(ValueError, match="requested_model"):
        TurnStarted("resp_1", "physical", 7, requested_model=" ")


def _open_hosted_item(
    turn: GenerationTurn,
    *,
    index: int = 0,
    item_id: str = "hosted_0",
    call_id: str = "call_0",
    name: str = "web_search",
) -> None:
    turn.emit(
        OutputItemStarted(
            "hosted_call",
            index,
            item_id,
            call_id=call_id,
            name=name,
        )
    )
    turn.emit(HostedCallStarted(index, item_id, call_id, name, {"query": "q"}))


_RESULT_URL = "https://example.com/result"
_RESULT_DIGEST = "sha256:feedface"


def _search_result(
    *,
    index: int = 0,
    item_id: str = "hosted_0",
    call_id: str = "call_0",
    tool_name: str = "web_search",
    url: str = _RESULT_URL,
    snippet: str = "snippet",
    digest: str = _RESULT_DIGEST,
) -> HostedCallResult:
    return HostedCallResult(
        index,
        item_id,
        call_id,
        tool_name,
        {
            "kind": "search_results",
            "query": "q",
            "results": [{"title": "t", "url": url, "snippet": snippet}],
            "digest": digest,
        },
    )


def _search_action(
    *,
    query: str = "q",
    sources: tuple[str, ...] = (_RESULT_URL,),
) -> dict[str, Any]:
    return {"kind": "search", "query": query, "sources": list(sources)}


def _hosted_receipt(
    status: str = "completed",
    **extras: Any,
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "call_id": "call_0",
        "status": status,
        "attempt": 1,
    }
    receipt.update(extras)
    return receipt


@pytest.mark.asyncio
async def test_hosted_call_lifecycle_opens_progresses_and_closes_once() -> None:
    turn = GenerationTurn(max_pending_events=32)
    _start(turn)
    _open_hosted_item(turn)
    turn.emit(HostedCallProgress(0, "hosted_0", "call_0", "executing"))
    turn.emit(_search_result())
    turn.emit(
        HostedCallCompleted(
            0,
            "hosted_0",
            "call_0",
            "web_search",
            "completed",
            _hosted_receipt(result_digest=_RESULT_DIGEST),
        )
    )
    turn.emit(
        OutputItemCompleted(
            "hosted_call",
            0,
            "hosted_0",
            call_id="call_0",
            name="web_search",
            status="completed",
            action=_search_action(),
        )
    )
    turn.complete(TurnCompleted("stop"))
    assert turn.state is TurnState.TERMINAL


@pytest.mark.asyncio
async def test_hosted_call_duplicate_receipt_raises() -> None:
    turn = GenerationTurn(max_pending_events=32)
    _start(turn)
    _open_hosted_item(turn)
    receipt = {"call_id": "call_0"}
    turn.emit(
        HostedCallCompleted(0, "hosted_0", "call_0", "web_search", "failed", receipt)
    )
    with pytest.raises(RuntimeError, match="receipt already emitted"):
        turn.emit(
            HostedCallCompleted(
                0,
                "hosted_0",
                "call_0",
                "web_search",
                "completed",
                receipt,
            )
        )


@pytest.mark.asyncio
async def test_hosted_call_receipt_requires_a_started_call() -> None:
    turn = GenerationTurn(max_pending_events=32)
    _start(turn)
    turn.emit(
        OutputItemStarted(
            "hosted_call",
            0,
            "hosted_0",
            call_id="call_0",
            name="web_search",
        )
    )
    with pytest.raises(RuntimeError, match="requires a started hosted call"):
        turn.emit(
            HostedCallCompleted(0, "hosted_0", "call_0", "web_search", "completed")
        )


@pytest.mark.asyncio
async def test_hosted_call_item_completion_requires_matching_receipt_status() -> None:
    turn = GenerationTurn(max_pending_events=32)
    _start(turn)
    _open_hosted_item(turn)
    turn.emit(HostedCallCompleted(0, "hosted_0", "call_0", "web_search", "failed"))
    with pytest.raises(RuntimeError, match="status must match its receipt"):
        turn.emit(
            OutputItemCompleted(
                "hosted_call",
                0,
                "hosted_0",
                call_id="call_0",
                name="web_search",
                status="completed",
                action=_search_action(sources=()),
            )
        )


@pytest.mark.asyncio
async def test_hosted_call_item_completion_requires_a_receipt() -> None:
    turn = GenerationTurn(max_pending_events=32)
    _start(turn)
    _open_hosted_item(turn)
    with pytest.raises(RuntimeError, match="no receipt event"):
        turn.emit(
            OutputItemCompleted(
                "hosted_call",
                0,
                "hosted_0",
                call_id="call_0",
                name="web_search",
                status="completed",
                action=_search_action(sources=()),
            )
        )


@pytest.mark.asyncio
async def test_hosted_call_receipt_call_id_mismatch_raises() -> None:
    turn = GenerationTurn(max_pending_events=32)
    _start(turn)
    _open_hosted_item(turn)
    with pytest.raises(RuntimeError, match="does not match its call id"):
        turn.emit(
            HostedCallCompleted(
                0,
                "hosted_0",
                "call_0",
                "web_search",
                "completed",
                {"call_id": "call_other"},
            )
        )


@pytest.mark.asyncio
async def test_hosted_call_events_require_a_hosted_call_item() -> None:
    turn = GenerationTurn(max_pending_events=32)
    _start(turn)
    turn.emit(OutputItemStarted("message", 0, "msg_0"))
    with pytest.raises(RuntimeError, match="hosted_call output item"):
        turn.emit(HostedCallStarted(0, "msg_0", "call_0", "web_search"))


def _complete_hosted_success(
    turn: GenerationTurn,
    *,
    sources: tuple[str, ...] = (_RESULT_URL,),
) -> None:
    _open_hosted_item(turn)
    turn.emit(_search_result())
    turn.emit(
        HostedCallCompleted(
            0,
            "hosted_0",
            "call_0",
            "web_search",
            "completed",
            _hosted_receipt(),
        )
    )
    turn.emit(
        OutputItemCompleted(
            "hosted_call",
            0,
            "hosted_0",
            call_id="call_0",
            name="web_search",
            status="completed",
            action=_search_action(sources=sources),
        )
    )


def _citation(**overrides: Any) -> HostedCitation:
    payload: dict[str, Any] = {
        "output_index": 1,
        "item_id": "message_1",
        "content_index": 0,
        "source_call_id": "call_0",
        "source_url": _RESULT_URL,
        "cited_text": "cite",
        "source_start": 0,
        "source_end": 4,
        "output_start": 0,
        "output_end": 4,
    }
    payload.update(overrides)
    return HostedCitation(**payload)


@pytest.mark.asyncio
async def test_hosted_result_is_legal_exactly_once() -> None:
    turn = GenerationTurn(max_pending_events=32)
    _start(turn)
    _open_hosted_item(turn)
    turn.emit(_search_result())
    with pytest.raises(RuntimeError, match="hosted result already emitted"):
        turn.emit(_search_result())


@pytest.mark.asyncio
async def test_hosted_result_requires_a_started_call_and_exact_identity() -> None:
    turn = GenerationTurn(max_pending_events=32)
    _start(turn)
    turn.emit(
        OutputItemStarted(
            "hosted_call",
            0,
            "hosted_0",
            call_id="call_0",
            name="web_search",
        )
    )
    with pytest.raises(RuntimeError, match="requires a started hosted call"):
        turn.emit(_search_result())
    turn.emit(HostedCallStarted(0, "hosted_0", "call_0", "web_search", {"query": "q"}))
    with pytest.raises(RuntimeError, match="does not match its output item"):
        turn.emit(_search_result(call_id="call_other"))
    with pytest.raises(
        RuntimeError, match="result tool name does not match its output item"
    ):
        turn.emit(_search_result(tool_name="web_fetch"))


@pytest.mark.asyncio
async def test_completed_receipt_without_result_is_rejected() -> None:
    turn = GenerationTurn(max_pending_events=32)
    _start(turn)
    _open_hosted_item(turn)
    with pytest.raises(RuntimeError, match="requires its result event"):
        turn.emit(
            HostedCallCompleted(
                0,
                "hosted_0",
                "call_0",
                "web_search",
                "completed",
                _hosted_receipt(),
            )
        )


@pytest.mark.asyncio
async def test_failed_receipt_with_result_is_rejected() -> None:
    turn = GenerationTurn(max_pending_events=32)
    _start(turn)
    _open_hosted_item(turn)
    turn.emit(_search_result())
    with pytest.raises(RuntimeError, match="cannot follow a result event"):
        turn.emit(
            HostedCallCompleted(
                0,
                "hosted_0",
                "call_0",
                "web_search",
                "failed",
                _hosted_receipt("failed", error={"code": "tool_timeout"}),
            )
        )


@pytest.mark.asyncio
async def test_hosted_result_after_its_receipt_is_rejected() -> None:
    turn = GenerationTurn(max_pending_events=32)
    _start(turn)
    _open_hosted_item(turn)
    turn.emit(_search_result())
    turn.emit(
        HostedCallCompleted(
            0,
            "hosted_0",
            "call_0",
            "web_search",
            "completed",
            _hosted_receipt(),
        )
    )
    with pytest.raises(RuntimeError, match="cannot follow its receipt"):
        turn.emit(_search_result(digest="sha256:other"))


@pytest.mark.asyncio
async def test_hosted_result_after_terminal_is_rejected() -> None:
    turn = GenerationTurn(max_pending_events=32)
    _start(turn)
    _complete_hosted_success(turn)
    turn.complete(TurnCompleted("stop"))
    with pytest.raises(RuntimeError, match="requires started state"):
        turn.emit(_search_result(digest="sha256:late"))


@pytest.mark.asyncio
async def test_receipt_digest_must_agree_with_the_result_digest() -> None:
    turn = GenerationTurn(max_pending_events=32)
    _start(turn)
    _open_hosted_item(turn)
    turn.emit(_search_result())
    with pytest.raises(RuntimeError, match="digest does not match its result"):
        turn.emit(
            HostedCallCompleted(
                0,
                "hosted_0",
                "call_0",
                "web_search",
                "completed",
                _hosted_receipt(result_digest="sha256:disagrees"),
            )
        )


@pytest.mark.asyncio
async def test_hosted_completion_action_cannot_contradict_its_start() -> None:
    turn = GenerationTurn(max_pending_events=32)
    _start(turn)
    _open_hosted_item(turn)
    turn.emit(_search_result())
    turn.emit(
        HostedCallCompleted(
            0,
            "hosted_0",
            "call_0",
            "web_search",
            "completed",
            _hosted_receipt(),
        )
    )
    with pytest.raises(RuntimeError, match="contradicts its started query"):
        turn.emit(
            OutputItemCompleted(
                "hosted_call",
                0,
                "hosted_0",
                call_id="call_0",
                name="web_search",
                status="completed",
                action=_search_action(query="different"),
            )
        )
    with pytest.raises(RuntimeError, match="action kind does not match its tool"):
        turn.emit(
            OutputItemCompleted(
                "hosted_call",
                0,
                "hosted_0",
                call_id="call_0",
                name="web_search",
                status="completed",
                action={"kind": "fetch", "url": "https://example.com"},
            )
        )


@pytest.mark.asyncio
async def test_fetch_completion_action_cannot_contradict_its_started_url() -> None:
    turn = GenerationTurn(max_pending_events=32)
    _start(turn)
    turn.emit(
        OutputItemStarted(
            "hosted_call",
            0,
            "hosted_0",
            call_id="call_0",
            name="web_fetch",
        )
    )
    turn.emit(
        HostedCallStarted(
            0,
            "hosted_0",
            "call_0",
            "web_fetch",
            {"url": "https://example.com/doc"},
        )
    )
    turn.emit(
        HostedCallResult(
            0,
            "hosted_0",
            "call_0",
            "web_fetch",
            {
                "kind": "document",
                "url": "https://example.com/doc",
                "digest": _RESULT_DIGEST,
            },
        )
    )
    turn.emit(
        HostedCallCompleted(
            0,
            "hosted_0",
            "call_0",
            "web_fetch",
            "completed",
            _hosted_receipt(),
        )
    )
    with pytest.raises(RuntimeError, match="contradicts its started url"):
        turn.emit(
            OutputItemCompleted(
                "hosted_call",
                0,
                "hosted_0",
                call_id="call_0",
                name="web_fetch",
                status="completed",
                action={"kind": "fetch", "url": "https://example.com/other"},
            )
        )


@pytest.mark.asyncio
async def test_hosted_completion_sources_must_be_proven_by_the_result() -> None:
    turn = GenerationTurn(max_pending_events=32)
    _start(turn)
    _open_hosted_item(turn)
    turn.emit(_search_result())
    turn.emit(
        HostedCallCompleted(
            0,
            "hosted_0",
            "call_0",
            "web_search",
            "completed",
            _hosted_receipt(),
        )
    )
    with pytest.raises(RuntimeError, match="not proven by its result"):
        turn.emit(
            OutputItemCompleted(
                "hosted_call",
                0,
                "hosted_0",
                call_id="call_0",
                name="web_search",
                status="completed",
                action=_search_action(
                    sources=(_RESULT_URL, "https://unproven.example"),
                ),
            )
        )


@pytest.mark.asyncio
async def test_item_lifecycle_retains_identities_but_never_result_payload() -> None:
    sentinel_snippet = "FETCHED-PAYLOAD-SENTINEL-BYTES"
    turn = GenerationTurn(max_pending_events=32)
    _start(turn)
    _open_hosted_item(turn)
    turn.emit(_search_result(snippet=sentinel_snippet))
    turn.emit(
        HostedCallCompleted(
            0,
            "hosted_0",
            "call_0",
            "web_search",
            "completed",
            _hosted_receipt(),
        )
    )

    lifecycle = turn._items[0]  # the retention law is the contract under test
    assert lifecycle.hosted_result_seen is True
    assert lifecycle.hosted_result_digest == _RESULT_DIGEST
    assert lifecycle.hosted_result_identities == (_RESULT_URL,)
    retained = repr(vars(lifecycle))
    assert sentinel_snippet not in retained
    assert "search_results" not in retained


def _open_cited_message(turn: GenerationTurn, text: str = "cite and more") -> None:
    turn.emit(OutputItemStarted("message", 1, "message_1"))
    turn.emit(ContentPartStarted("output_text", 1, 0, "message_1"))
    turn.emit(TextDelta(text, "message_1", 1, 0))


@pytest.mark.asyncio
async def test_citation_requires_a_proven_result_identity() -> None:
    turn = GenerationTurn(max_pending_events=64)
    _start(turn)
    _complete_hosted_success(turn)
    _open_cited_message(turn)

    with pytest.raises(RuntimeError, match="unknown hosted result"):
        turn.emit(_citation(source_call_id="call_unknown"))
    with pytest.raises(RuntimeError, match="url is not proven by its result"):
        turn.emit(_citation(source_url="https://unproven.example"))
    turn.emit(_citation())


@pytest.mark.asyncio
async def test_citation_requires_an_open_message_text_part() -> None:
    turn = GenerationTurn(max_pending_events=64)
    _start(turn)
    _complete_hosted_success(turn)

    turn.emit(OutputItemStarted("reasoning", 1, "reasoning_1"))
    turn.emit(ContentPartStarted("reasoning_summary_text", 1, 0, "reasoning_1"))
    with pytest.raises(RuntimeError, match="requires a message output item"):
        turn.emit(_citation(item_id="reasoning_1"))
    turn.emit(ReasoningCompleted("", "reasoning_1", 1, 0))
    turn.emit(ContentPartCompleted("reasoning_summary_text", 1, 0, "reasoning_1", ""))
    turn.emit(OutputItemCompleted("reasoning", 1, "reasoning_1", text=""))

    turn.emit(OutputItemStarted("message", 2, "message_2"))
    with pytest.raises(RuntimeError, match="has not been started"):
        turn.emit(_citation(output_index=2, item_id="message_2"))


@pytest.mark.asyncio
async def test_citation_cannot_follow_content_completion_or_item_done() -> None:
    turn = GenerationTurn(max_pending_events=64)
    _start(turn)
    _complete_hosted_success(turn)
    _open_cited_message(turn)
    turn.emit(TextCompleted("cite and more", "message_1", 1, 0))
    turn.emit(ContentPartCompleted("output_text", 1, 0, "message_1", "cite and more"))
    with pytest.raises(RuntimeError, match="cannot follow its content completion"):
        turn.emit(_citation())
    turn.emit(OutputItemCompleted("message", 1, "message_1", text="cite and more"))
    with pytest.raises(RuntimeError, match="already done"):
        turn.emit(_citation())


@pytest.mark.asyncio
async def test_citation_output_ranges_must_fit_the_final_text() -> None:
    turn = GenerationTurn(max_pending_events=64)
    _start(turn)
    _complete_hosted_success(turn)
    _open_cited_message(turn, text="cite")

    # Accepted while streaming, falsified at TextCompleted reconciliation.
    turn.emit(_citation(output_start=90, output_end=94))
    with pytest.raises(RuntimeError, match="exceeds its final text"):
        turn.emit(TextCompleted("cite", "message_1", 1, 0))


@pytest.mark.asyncio
async def test_citation_after_text_done_is_checked_immediately() -> None:
    turn = GenerationTurn(max_pending_events=64)
    _start(turn)
    _complete_hosted_success(turn)
    _open_cited_message(turn, text="cite")
    turn.emit(TextCompleted("cite", "message_1", 1, 0))

    turn.emit(_citation())
    with pytest.raises(RuntimeError, match="exceeds its final text"):
        turn.emit(_citation(output_start=90, output_end=94))


@pytest.mark.asyncio
async def test_citations_per_item_are_bounded() -> None:
    turn = GenerationTurn(max_pending_events=256)
    _start(turn)
    _complete_hosted_success(turn)
    _open_cited_message(turn)
    for _ in range(MAX_CITATIONS_PER_ITEM):
        turn.emit(_citation())
    with pytest.raises(RuntimeError, match="exceeds its item bound"):
        turn.emit(_citation())
