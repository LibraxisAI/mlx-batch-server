"""Bounded target-owned agent rounds and exactly-once tool execution."""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .parser import ParsedToolCall


@dataclass(frozen=True, slots=True)
class AgentLoopPolicy:
    max_rounds: int = 8
    parallel_tool_calls: bool = True
    stop_on_tool_error: bool = True

    def __post_init__(self) -> None:
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")


def hosted_agent_loop_policy(*, max_rounds: int = 8) -> AgentLoopPolicy:
    """The one hosted-execution policy: an error receipt must reach the model.

    ``stop_on_tool_error=False`` is the single semantic flip that implements the
    failure-continuation contract at the loop layer; the default raise would
    burn the model's one explanatory continuation.
    """

    return AgentLoopPolicy(max_rounds=max_rounds, stop_on_tool_error=False)


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    call_id: str
    output: str
    metadata: Mapping[str, Any] | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@runtime_checkable
class ToolExecutor(Protocol):
    def execute(self, call: ParsedToolCall) -> Awaitable[ToolExecutionResult]: ...


@dataclass(frozen=True, slots=True)
class AgentRound:
    """One model result, before or after tool execution."""

    tool_calls: tuple[ParsedToolCall, ...] = ()
    output: Any = None


@dataclass(frozen=True, slots=True)
class AgentLoopResult:
    """Terminal model round plus the ordered tool results sent back to it."""

    final_round: AgentRound
    tool_rounds: int
    executions: tuple[ToolExecutionResult, ...]


@runtime_checkable
class AgentRoundContinuation(Protocol):
    def __call__(
        self,
        results: tuple[ToolExecutionResult, ...],
        *,
        round_index: int,
    ) -> Awaitable[AgentRound]: ...


class AgentLoopError(RuntimeError):
    """Base error for target-owned agent-loop invariants."""


class AgentLoopLimitExceeded(AgentLoopError):
    """The model requested another tool round beyond the configured bound."""


class DuplicateToolCallError(AgentLoopError):
    """A call_id was reused for a different tool invocation."""


class ToolExecutionContractError(AgentLoopError):
    """A ToolExecutor returned a result for a different call_id."""

    def __init__(
        self,
        expected_call_id: str,
        actual_call_id: str,
        *,
        results: tuple[ToolExecutionResult, ...] = (),
    ) -> None:
        self.expected_call_id = expected_call_id
        self.actual_call_id = actual_call_id
        self.results = results
        super().__init__(
            f"executor returned call_id {actual_call_id} for {expected_call_id}"
        )


class ToolExecutionError(AgentLoopError):
    """One or more tool calls failed under a stop-on-error policy."""

    def __init__(self, results: tuple[ToolExecutionResult, ...]) -> None:
        self.results = results
        failed = ", ".join(result.call_id for result in results if not result.ok)
        super().__init__(f"tool execution failed for call_id(s): {failed}")


@dataclass(frozen=True, slots=True)
class _ToolClaim:
    index: int
    name: str
    arguments: str


class AgentLoop:
    """Own bounded continuation rounds and de-duplicate tool side effects."""

    def __init__(
        self,
        executor: ToolExecutor,
        policy: AgentLoopPolicy | None = None,
        *,
        loop_id: str | None = None,
    ) -> None:
        if loop_id is not None and not loop_id.strip():
            raise ValueError("loop_id must not be empty")
        self._executor = executor
        self._policy = policy or AgentLoopPolicy()
        self._loop_id = loop_id or uuid.uuid4().hex
        self._claims: dict[str, _ToolClaim] = {}
        self._results: dict[str, ToolExecutionResult] = {}
        self._failures: dict[str, BaseException] = {}
        self._receipt_order: list[str] = []
        self._round_call_ids: dict[str, tuple[str, ...]] = {}
        self._direct_round_sequence = 0
        self._rounds = 0
        self._round_lock = asyncio.Lock()
        self._started = False
        self._running = False
        self._completed = False

    @property
    def rounds(self) -> int:
        return self._rounds

    @property
    def loop_id(self) -> str:
        return self._loop_id

    @property
    def receipts(self) -> tuple[ToolExecutionResult, ...]:
        return tuple(self._results[call_id] for call_id in self._receipt_order)

    async def run(
        self,
        initial_round: AgentRound,
        continuation: AgentRoundContinuation,
    ) -> AgentLoopResult:
        if self._started:
            raise RuntimeError("an AgentLoop instance may only be run once")

        self._started = True
        self._running = True
        current = initial_round
        executions: list[ToolExecutionResult] = []
        try:
            while current.tool_calls:
                results = await self.execute_round(
                    current.tool_calls,
                    round_id=f"model-{self._rounds}",
                )
                executions.extend(results)
                current = await continuation(results, round_index=self._rounds)
            self._completed = True
            return AgentLoopResult(
                final_round=current,
                tool_rounds=self._rounds,
                executions=tuple(executions),
            )
        finally:
            self._running = False

    async def execute_round(
        self,
        calls: Sequence[ParsedToolCall],
        *,
        round_id: str | None = None,
    ) -> tuple[ToolExecutionResult, ...]:
        async with self._round_lock:
            if round_id is None:
                round_id = f"direct-{self._direct_round_sequence}"
                self._direct_round_sequence += 1
            if not round_id.strip():
                raise ValueError("round_id must not be empty")
            return await self._execute_round(calls, round_id=round_id)

    async def _execute_round(
        self,
        calls: Sequence[ParsedToolCall],
        *,
        round_id: str,
    ) -> tuple[ToolExecutionResult, ...]:
        scoped_calls = self._scope_calls(calls, round_id=round_id)
        round_call_ids = tuple(dict.fromkeys(call.call_id for call in scoped_calls))
        previous_round_call_ids = self._round_call_ids.get(round_id)
        is_new_round = previous_round_call_ids is None
        if (
            previous_round_call_ids is not None
            and previous_round_call_ids != round_call_ids
        ):
            raise DuplicateToolCallError(
                f"round_id {round_id} was reused with a different call set"
            )
        if is_new_round and self._rounds >= self._policy.max_rounds:
            raise AgentLoopLimitExceeded(
                f"tool round limit reached: {self._policy.max_rounds}"
            )

        ordered_calls = self._claim_unique_calls(scoped_calls)
        if not ordered_calls:
            return ()
        if is_new_round:
            self._round_call_ids[round_id] = round_call_ids
            self._rounds += 1

        previous_failure = next(
            (
                self._failures[call.call_id]
                for call in ordered_calls
                if call.call_id in self._failures
            ),
            None,
        )
        if previous_failure is not None:
            self._raise_failure(previous_failure, ordered_calls)

        pending = [
            call
            for call in ordered_calls
            if call.call_id not in self._results and call.call_id not in self._failures
        ]

        if self._policy.parallel_tool_calls:
            await self._execute_parallel(pending, ordered_calls=ordered_calls)
        else:
            await self._execute_sequential(pending, ordered_calls=ordered_calls)

        available_results = tuple(
            self._results[call.call_id]
            for call in ordered_calls
            if call.call_id in self._results
        )
        if self._policy.stop_on_tool_error and any(
            not result.ok for result in available_results
        ):
            raise ToolExecutionError(available_results)
        results = tuple(self._results[call.call_id] for call in ordered_calls)
        return results

    def _scope_calls(
        self,
        calls: Sequence[ParsedToolCall],
        *,
        round_id: str,
    ) -> tuple[ParsedToolCall, ...]:
        scoped: list[ParsedToolCall] = []
        for call in calls:
            digest = hashlib.sha256(
                f"{self._loop_id}\0{round_id}\0{call.call_id}".encode()
            ).hexdigest()[:24]
            scoped.append(
                ParsedToolCall(
                    index=call.index,
                    call_id=f"call_{digest}",
                    name=call.name,
                    arguments=call.arguments,
                )
            )
        return tuple(scoped)

    def _claim_unique_calls(
        self,
        calls: Sequence[ParsedToolCall],
    ) -> tuple[ParsedToolCall, ...]:
        unique: list[ParsedToolCall] = []
        seen_this_round: set[str] = set()
        next_claims = dict(self._claims)

        for call in calls:
            if call.index < 0:
                raise ValueError("tool call index must be non-negative")
            if not call.call_id.strip():
                raise ValueError("tool call_id must not be empty")
            if not call.name.strip():
                raise ValueError("tool name must not be empty")

            claim = _ToolClaim(
                index=call.index,
                name=call.name,
                arguments=call.arguments,
            )
            previous = next_claims.get(call.call_id)
            if previous is not None and previous != claim:
                raise DuplicateToolCallError(
                    f"call_id {call.call_id} was reused with different payload"
                )
            next_claims.setdefault(call.call_id, claim)

            if call.call_id not in seen_this_round:
                unique.append(call)
                seen_this_round.add(call.call_id)

        self._claims = next_claims
        return tuple(unique)

    async def _execute_parallel(
        self,
        calls: Sequence[ParsedToolCall],
        *,
        ordered_calls: Sequence[ParsedToolCall],
    ) -> None:
        if not calls:
            return
        gathered = await asyncio.gather(
            *(self._execute_one(call) for call in calls),
            return_exceptions=True,
        )
        first_failure: BaseException | None = None
        for call, item in zip(calls, gathered, strict=True):
            if isinstance(item, BaseException):
                self._failures.setdefault(call.call_id, item)
                if first_failure is None:
                    first_failure = item
                continue
            self._record_result(item)
        if first_failure is not None:
            self._raise_failure(first_failure, ordered_calls)

    async def _execute_sequential(
        self,
        calls: Sequence[ParsedToolCall],
        *,
        ordered_calls: Sequence[ParsedToolCall],
    ) -> None:
        for call in calls:
            try:
                result = await self._execute_one(call)
            except BaseException as error:
                self._failures.setdefault(call.call_id, error)
                self._raise_failure(error, ordered_calls)
            self._record_result(result)
            if self._policy.stop_on_tool_error and not result.ok:
                break

    def _record_result(self, result: ToolExecutionResult) -> None:
        if result.call_id not in self._results:
            self._results[result.call_id] = result
            self._receipt_order.append(result.call_id)

    def _raise_failure(
        self,
        error: BaseException,
        ordered_calls: Sequence[ParsedToolCall],
    ) -> None:
        available_results = tuple(
            self._results[call.call_id]
            for call in ordered_calls
            if call.call_id in self._results
        )
        if isinstance(error, ToolExecutionContractError):
            raise ToolExecutionContractError(
                error.expected_call_id,
                error.actual_call_id,
                results=available_results,
            ) from error
        raise error

    async def _execute_one(self, call: ParsedToolCall) -> ToolExecutionResult:
        try:
            result = await self._executor.execute(call)
        except Exception as error:
            return ToolExecutionResult(
                call_id=call.call_id,
                output="",
                metadata={"exception_type": type(error).__name__},
                error=str(error),
            )

        if result.call_id != call.call_id:
            raise ToolExecutionContractError(
                call.call_id,
                result.call_id,
            )
        return result
