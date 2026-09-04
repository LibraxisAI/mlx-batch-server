"""RED contracts for canonical parsing and bounded target-owned tool rounds.

These tests are intentionally authored but not executed while Compile Embargo
is HOLD.
"""

from __future__ import annotations

import asyncio

import pytest

from mlx_batch_server.tools.agent_loop import (
    AgentLoop,
    AgentLoopLimitExceeded,
    AgentLoopPolicy,
    AgentRound,
    DuplicateToolCallError,
    ToolExecutionContractError,
    ToolExecutionError,
    ToolExecutionResult,
)
from mlx_batch_server.tools.parser import (
    DialectParse,
    DialectToolCall,
    IncrementalToolParser,
    ParsedToolCall,
    ToolParseError,
)


class _MarkerDialect:
    """Tiny cumulative dialect fixture; production dialects remain pluggable."""

    def parse(self, text: str, *, final: bool) -> DialectParse:
        marker = "<tool "
        if marker not in text:
            return DialectParse(visible_text=text)

        visible, encoded = text.split(marker, 1)
        complete = encoded.endswith(">")
        if complete:
            encoded = encoded[:-1]
        name_part, separator, arguments = encoded.partition(";args=")
        name = name_part.removeprefix("name=") if separator else None
        return DialectParse(
            visible_text=visible,
            calls=(
                DialectToolCall(
                    index=0,
                    call_id="call_lookup",
                    name=name,
                    arguments=arguments,
                    complete=complete,
                ),
            ),
        )


class _TrackingExecutor:
    def __init__(self) -> None:
        self.calls: list[ParsedToolCall] = []

    async def execute(self, call: ParsedToolCall) -> ToolExecutionResult:
        self.calls.append(call)
        return ToolExecutionResult(call_id=call.call_id, output=f"ran:{call.name}")


class _TransactionalDialect:
    def parse(self, text: str, *, final: bool) -> DialectParse:
        del final
        if text == "a":
            return DialectParse(calls=(DialectToolCall(0, "call_tx", "lookup", "{"),))
        if text == "abad":
            return DialectParse(
                calls=(
                    DialectToolCall(0, "call_tx", "lookup", '{"leaked":'),
                    DialectToolCall(0, "call_duplicate", "lookup", "{}"),
                )
            )
        if text == "agood":
            return DialectParse(
                calls=(
                    DialectToolCall(
                        0,
                        "call_tx",
                        "lookup",
                        '{"ok":1}',
                        complete=True,
                    ),
                )
            )
        raise AssertionError(f"unexpected source: {text}")


class _CompletedGrowthDialect:
    def parse(self, text: str, *, final: bool) -> DialectParse:
        del final
        arguments = "{}" if text == "a" else "{} trailing"
        return DialectParse(
            calls=(
                DialectToolCall(
                    0,
                    "call_complete",
                    "lookup",
                    arguments,
                    complete=True,
                ),
            )
        )


def _call(
    call_id: str,
    *,
    name: str = "lookup",
    arguments: str = '{"q":"cats"}',
    index: int = 0,
) -> ParsedToolCall:
    return ParsedToolCall(
        index=index,
        call_id=call_id,
        name=name,
        arguments=arguments,
    )


def test_incremental_parser_hides_dialect_markers_and_emits_suffixes_once() -> None:
    parser = IncrementalToolParser(_MarkerDialect())

    visible, first = parser.feed("visible <tool name=lookup;args={")
    assert visible == "visible "
    assert first[0].name == "lookup"
    assert first[0].arguments_delta == "{"

    visible, second = parser.feed('"q":"cats"}>')
    assert visible == ""
    assert second[0].name is None
    assert second[0].arguments_delta == '"q":"cats"}'

    final_visible, calls = parser.finish()
    assert final_visible == ""
    assert calls == (_call("call_lookup"),)
    assert parser.finish() == ("", calls)


def test_parser_rejects_snapshot_transactionally_without_poisoning_source() -> None:
    parser = IncrementalToolParser(_TransactionalDialect())

    parser.feed("a")
    with pytest.raises(ToolParseError, match="duplicate tool index"):
        parser.feed("bad")

    _, delta = parser.feed("good")
    assert delta[0].arguments_delta == '"ok":1}'
    assert parser.finish()[1] == (_call("call_tx", arguments='{"ok":1}'),)


def test_completed_tool_call_cannot_grow_after_completion() -> None:
    parser = IncrementalToolParser(_CompletedGrowthDialect())

    parser.feed("a")
    with pytest.raises(ToolParseError, match="changed after completion"):
        parser.feed("b")

    assert parser.finish()[1] == (_call("call_complete", arguments="{}"),)


@pytest.mark.asyncio
async def test_repeated_source_call_id_is_unique_across_model_rounds() -> None:
    executor = _TrackingExecutor()
    loop = AgentLoop(
        executor,
        AgentLoopPolicy(max_rounds=2),
        loop_id="response-123",
    )
    call = _call("call_repeat")
    continuations: list[tuple[ToolExecutionResult, ...]] = []

    async def continue_round(
        results: tuple[ToolExecutionResult, ...],
        *,
        round_index: int,
    ) -> AgentRound:
        continuations.append(results)
        if round_index == 1:
            return AgentRound(tool_calls=(call, call))
        return AgentRound(output="done")

    result = await loop.run(AgentRound(tool_calls=(call,)), continue_round)

    assert len(executor.calls) == 2
    assert executor.calls[0].call_id != executor.calls[1].call_id
    assert all(item.call_id.startswith("call_") for item in executor.calls)
    assert result.tool_rounds == 2
    assert result.final_round.output == "done"
    assert [group[0].call_id for group in continuations] == [
        executor.calls[0].call_id,
        executor.calls[1].call_id,
    ]


@pytest.mark.asyncio
async def test_reused_call_id_with_changed_payload_is_rejected() -> None:
    executor = _TrackingExecutor()
    loop = AgentLoop(executor, AgentLoopPolicy(max_rounds=2))
    original = _call("call_conflict")
    changed = _call("call_conflict", arguments='{"q":"dogs"}')

    await loop.execute_round((original,), round_id="retryable-round")
    with pytest.raises(DuplicateToolCallError, match="different payload"):
        await loop.execute_round((changed,), round_id="retryable-round")
    assert len(executor.calls) == 1
    assert executor.calls[0].arguments == original.arguments


@pytest.mark.asyncio
async def test_max_rounds_blocks_the_next_tool_side_effect() -> None:
    executor = _TrackingExecutor()
    loop = AgentLoop(executor, AgentLoopPolicy(max_rounds=1))
    first = _call("call_first")
    blocked = _call("call_blocked", index=1)

    async def continue_round(
        results: tuple[ToolExecutionResult, ...],
        *,
        round_index: int,
    ) -> AgentRound:
        del results, round_index
        return AgentRound(tool_calls=(blocked,))

    with pytest.raises(AgentLoopLimitExceeded, match="limit"):
        await loop.run(AgentRound(tool_calls=(first,)), continue_round)
    assert [call.name for call in executor.calls] == [first.name]


@pytest.mark.asyncio
async def test_executor_cannot_return_a_foreign_call_id() -> None:
    class _WrongExecutor:
        async def execute(self, call: ParsedToolCall) -> ToolExecutionResult:
            return ToolExecutionResult(call_id="call_foreign", output=call.name)

    loop = AgentLoop(_WrongExecutor())

    with pytest.raises(ToolExecutionContractError, match="call_foreign"):
        await loop.execute_round((_call("call_expected"),))


@pytest.mark.asyncio
async def test_concurrent_round_requests_cannot_duplicate_a_tool_side_effect() -> None:
    class _YieldingExecutor(_TrackingExecutor):
        async def execute(self, call: ParsedToolCall) -> ToolExecutionResult:
            await asyncio.sleep(0)
            return await super().execute(call)

    executor = _YieldingExecutor()
    loop = AgentLoop(executor, AgentLoopPolicy(max_rounds=2))
    call = _call("call_concurrent")

    first, second = await asyncio.gather(
        loop.execute_round((call,), round_id="same-round-retry"),
        loop.execute_round((call,), round_id="same-round-retry"),
    )

    assert len(executor.calls) == 1
    assert first == second


@pytest.mark.asyncio
async def test_parallel_contract_failure_preserves_success_and_blocks_reexecution() -> (
    None
):
    class _PartialContractExecutor(_TrackingExecutor):
        async def execute(self, call: ParsedToolCall) -> ToolExecutionResult:
            self.calls.append(call)
            if call.name == "broken":
                return ToolExecutionResult(call_id="call_foreign", output="side effect")
            return ToolExecutionResult(call_id=call.call_id, output="receipt")

    executor = _PartialContractExecutor()
    loop = AgentLoop(executor, loop_id="response-contract-failure")
    calls = (
        _call("call_local_0", name="success", index=0),
        _call("call_local_1", name="broken", index=1),
    )

    with pytest.raises(ToolExecutionContractError) as first_failure:
        await loop.execute_round(calls, round_id="model-0")

    assert [result.output for result in first_failure.value.results] == ["receipt"]
    assert loop.receipts == first_failure.value.results
    assert len(executor.calls) == 2

    with pytest.raises(ToolExecutionContractError) as retry_failure:
        await loop.execute_round(calls, round_id="model-0")

    assert retry_failure.value.results == first_failure.value.results
    assert len(executor.calls) == 2


@pytest.mark.asyncio
async def test_failed_tool_receipt_is_stable_across_same_round_retry() -> None:
    class _MixedExecutor(_TrackingExecutor):
        async def execute(self, call: ParsedToolCall) -> ToolExecutionResult:
            self.calls.append(call)
            if call.name == "broken":
                raise RuntimeError("tool unavailable")
            return ToolExecutionResult(call_id=call.call_id, output="receipt")

    executor = _MixedExecutor()
    loop = AgentLoop(executor, loop_id="response-tool-failure")
    calls = (
        _call("call_local_0", name="success", index=0),
        _call("call_local_1", name="broken", index=1),
    )

    with pytest.raises(ToolExecutionError) as first_failure:
        await loop.execute_round(calls, round_id="model-0")
    with pytest.raises(ToolExecutionError) as retry_failure:
        await loop.execute_round(calls, round_id="model-0")

    assert retry_failure.value.results == first_failure.value.results
    assert len(executor.calls) == 2
