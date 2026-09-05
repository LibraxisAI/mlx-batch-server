"""Delivery verifier for the hosted failure-continuation runtime (design §8-§9).

A deterministic fake backend drives ``HostedAgenticRuntimeStarter`` through
successful hosted execution and each tool-level failure class. For every
failure it proves: one executor attempt, one immutable typed error receipt,
exactly one terminal continuation round, non-empty explanatory model text, and
one completed outer turn — never an abrupt outer failure.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import pytest

from mlx_batch_server.runtime.agentic import (
    FAILURE_CONTINUATION_PREPARATION,
    NO_WEB_PREPARATION,
    HostedAgenticRuntimeStarter,
)
from mlx_batch_server.runtime.contracts import GenerationRequest, RuntimeKey, TurnSink
from mlx_batch_server.runtime.events import (
    HostedCallCompleted,
    HostedCallStarted,
    OutputItemCompleted,
    OutputItemStarted,
    TextCompleted,
    TextDelta,
    ToolCompleted,
    ToolDelta,
    TurnCancelled,
    TurnCompleted,
    TurnFailed,
    TurnStarted,
    UsageUpdate,
)
from mlx_batch_server.runtime.service import (
    FirstWriterCancelToken,
    RuntimeStartError,
    RuntimeStartService,
)
from mlx_batch_server.runtime.turn import GenerationTurn, TurnState
from mlx_batch_server.tools.hosted import (
    HostedToolCatalog,
    HostedToolError,
    HostedToolExecutor,
    HostedToolSuccess,
)
from mlx_batch_server.tools.hosted_web import HostedWebSearchTool


@dataclass(slots=True)
class _Round:
    """One scripted model round for the deterministic fake backend."""

    text: str | None = None
    tool_calls: tuple[tuple[str, str, str], ...] = ()  # (call_id, name, args)
    usage: tuple[int, int] = (10, 5)
    hold: asyncio.Event | None = None
    repeat_last_tool_call: bool = False  # stream-level identical duplicate


class _FakeBackendTurn:
    def __init__(
        self,
        request: GenerationRequest,
        sink: TurnSink,
        step: _Round,
        round_index: int,
    ) -> None:
        self._request = request
        self._sink = sink
        self._step = step
        self._round_index = round_index
        self._cancelled: str | None = None
        self._task = asyncio.create_task(self._emit())

    @property
    def response_id(self) -> str:
        return self._request.response_id

    def cancel(self, reason: str) -> bool:
        self._cancelled = reason
        if self._step.hold is not None:
            self._step.hold.set()
        return True

    def wait_closed(self) -> asyncio.Future[None]:
        return asyncio.shield(self._task)

    async def _emit(self) -> None:
        sink = self._sink
        step = self._step
        rnd = self._round_index
        sink.emit(
            TurnStarted(
                response_id=self._request.response_id,
                model=self._request.runtime.model_id,
                created_at=1,
            )
        )
        if step.hold is not None:
            await step.hold.wait()
            if self._cancelled is not None:
                sink.emit(TurnCancelled(self._cancelled))
                return
        index = 0
        if step.text is not None:
            item_id = f"msg_r{rnd}"
            sink.emit(OutputItemStarted("message", index, item_id))
            from mlx_batch_server.runtime.events import (
                ContentPartCompleted,
                ContentPartStarted,
            )

            sink.emit(ContentPartStarted("output_text", index, 0, item_id))
            sink.emit(TextDelta(step.text, item_id, index, 0))
            sink.emit(TextCompleted(step.text, item_id, index, 0))
            sink.emit(ContentPartCompleted("output_text", index, 0, item_id, step.text))
            sink.emit(OutputItemCompleted("message", index, item_id, text=step.text))
            index += 1
        for call_id, name, arguments in step.tool_calls:
            item_id = f"fc_r{rnd}_{call_id}"
            sink.emit(
                OutputItemStarted(
                    "function_call",
                    index,
                    item_id,
                    call_id=call_id,
                    name=name,
                )
            )
            sink.emit(
                ToolDelta(
                    index,
                    call_id,
                    item_id,
                    name=name,
                    arguments_delta=arguments,
                )
            )
            sink.emit(ToolCompleted(index, call_id, item_id, name, arguments))
            sink.emit(
                OutputItemCompleted(
                    "function_call",
                    index,
                    item_id,
                    call_id=call_id,
                    name=name,
                    arguments=arguments,
                )
            )
            index += 1
        if step.repeat_last_tool_call and step.tool_calls:
            call_id, name, arguments = step.tool_calls[-1]
            item_id = f"fc_r{rnd}_{call_id}"
            sink.emit(ToolCompleted(index - 1, call_id, item_id, name, arguments))
        input_tokens, output_tokens = step.usage
        usage = UsageUpdate(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )
        sink.emit(usage)
        finish = "tool_calls" if step.tool_calls else "stop"
        sink.emit(TurnCompleted(finish, usage=usage))


class _FakeInner(RuntimeStartService):
    """Scripted RuntimeStartService double; isinstance seam requires the type."""

    def __init__(self, script: tuple[_Round, ...]) -> None:
        # Intentionally no super().__init__ (no manager in the fake).
        self.script = script
        self.requests: list[GenerationRequest] = []
        self.sinks: list[TurnSink] = []
        self.turns: list[_FakeBackendTurn] = []

    async def start(
        self,
        request: GenerationRequest,
        sink: TurnSink,
        *,
        cancel: FirstWriterCancelToken | None = None,
    ) -> _FakeBackendTurn:
        round_index = len(self.requests)
        self.requests.append(request)
        self.sinks.append(sink)
        if round_index >= len(self.script):
            raise AssertionError(f"fake backend has no script for round {round_index}")
        turn = _FakeBackendTurn(request, sink, self.script[round_index], round_index)
        self.turns.append(turn)
        return turn


@dataclass(slots=True)
class _CountingTool:
    name: str
    behavior: Any
    invocations: int = 0
    arguments_seen: list[Mapping[str, Any]] = field(default_factory=list)

    def describe(self) -> Mapping[str, Any]:
        return {"name": self.name}

    async def invoke(self, arguments: Mapping[str, Any]) -> HostedToolSuccess:
        self.invocations += 1
        self.arguments_seen.append(arguments)
        return await self.behavior(arguments)


async def _ok_behavior(arguments: Mapping[str, Any]) -> HostedToolSuccess:
    return HostedToolSuccess(
        payload={"results": [{"title": "t", "url": "https://ok.example"}]},
        receipt_fields={"final_url": "https://ok.example", "mime": "text/html"},
    )


def _raising_behavior(code: str, message: str):
    async def behavior(arguments: Mapping[str, Any]) -> HostedToolSuccess:
        raise HostedToolError(code, message)

    return behavior


def _request(tools: tuple[Mapping[str, Any], ...]) -> GenerationRequest:
    return GenerationRequest(
        response_id="resp_hosted",
        runtime=RuntimeKey(model_id="model-x"),
        messages=({"role": "user", "content": "co pisza o loctree?"},),
        tools=tools,
    )


def _starter(
    inner: _FakeInner,
    tools: tuple[Any, ...],
    **kwargs: Any,
) -> tuple[HostedAgenticRuntimeStarter, HostedToolCatalog]:
    catalog = HostedToolCatalog(tools)
    executor_kwargs = {}
    if "per_call_timeout_s" in kwargs:
        executor_kwargs["per_call_timeout_s"] = kwargs.pop("per_call_timeout_s")
    executor = HostedToolExecutor(catalog, **executor_kwargs)
    return (
        HostedAgenticRuntimeStarter(
            inner,
            catalog=catalog,
            executor=executor,
            **kwargs,
        ),
        catalog,
    )


async def _drive(
    starter: HostedAgenticRuntimeStarter,
    request: GenerationRequest,
) -> tuple[list[Any], GenerationTurn]:
    outer = GenerationTurn(max_pending_events=512)
    subscription = outer.subscribe(max_pending_events=512)
    handle = await starter.start(
        request,
        outer,
        cancel=FirstWriterCancelToken(),
    )
    await handle.wait_closed()
    events = [item.event async for item in subscription]
    return events, outer


def _of(events: list[Any], event_type: type) -> list[Any]:
    return [event for event in events if isinstance(event, event_type)]


_FAILURE_CASES = (
    ("provider_unavailable", None),  # F4: real web_search tool, no provider
    ("provider_auth_failed", _raising_behavior("provider_auth_failed", "auth")),
    (
        "fetch_url_target_blocked",  # F6
        _raising_behavior("fetch_url_target_blocked", "not a public address"),
    ),
    (
        "fetch_source_bytes_exceeded",  # F7
        _raising_behavior("fetch_source_bytes_exceeded", "too large"),
    ),
    ("tool_execution_failed", None),  # F9: crash, built below
)


@pytest.mark.asyncio
@pytest.mark.parametrize("code,behavior", _FAILURE_CASES)
async def test_tool_failure_yields_one_receipt_and_one_terminal_continuation(
    code: str,
    behavior: Any,
) -> None:
    inner = _FakeInner(
        (
            _Round(tool_calls=(("call_a", "web_search", '{"query":"loctree"}'),)),
            _Round(text="The web_search tool failed, so here is what I know."),
        )
    )
    if code == "provider_unavailable":
        tool: Any = HostedWebSearchTool(provider=None)
    elif code == "tool_execution_failed":

        async def crash(arguments: Mapping[str, Any]) -> HostedToolSuccess:
            raise RuntimeError("backend exploded")

        tool = _CountingTool("web_search", crash)
    else:
        tool = _CountingTool("web_search", behavior)
    starter, _ = _starter(inner, (tool,))
    events, outer = await _drive(starter, _request(({"type": "web_search"},)))

    # One completed outer turn, never an abrupt outer failure (falsification 1).
    assert not _of(events, TurnFailed)
    completed = _of(events, TurnCompleted)
    assert len(completed) == 1

    # Exactly one immutable typed error receipt.
    receipts = _of(events, HostedCallCompleted)
    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt.status == "failed"
    assert receipt.receipt["error"]["code"] == code
    assert receipt.receipt["attempt"] == 1
    with pytest.raises(TypeError):
        receipt.receipt["error"]["code"] = "mutated"  # type: ignore[index]

    # Exactly one executor attempt and one continuation round.
    if isinstance(tool, _CountingTool):
        assert tool.invocations == 1
    assert len(inner.requests) == 2

    # The continuation round carries the error exactly once as a tool message
    # plus the trusted preparation that quotes nothing from the payload.
    continuation = inner.requests[1].messages
    tool_messages = [m for m in continuation if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert code in tool_messages[0]["content"]
    assert tool_messages[0]["tool_call_id"] == "call_a"
    preparations = [
        m
        for m in continuation
        if m.get("role") == "system"
        and m.get("content") == FAILURE_CONTINUATION_PREPARATION
    ]
    assert len(preparations) == 1
    assert code not in FAILURE_CONTINUATION_PREPARATION

    # Explanatory, non-empty model text closed the turn.
    texts = _of(events, TextCompleted)
    assert texts and texts[-1].text
    assert outer.state is TurnState.TERMINAL


@pytest.mark.asyncio
async def test_model_generated_invalid_arguments_are_f10_receipts() -> None:
    inner = _FakeInner(
        (
            _Round(tool_calls=(("call_a", "web_search", "{not json"),)),
            _Round(text="I could not run the search because its input was invalid."),
        )
    )
    tool = _CountingTool("web_search", _ok_behavior)
    starter, _ = _starter(inner, (tool,))
    events, _ = await _drive(starter, _request(({"type": "web_search"},)))

    receipts = _of(events, HostedCallCompleted)
    assert len(receipts) == 1
    assert receipts[0].receipt["error"]["code"] == "invalid_tool_arguments"
    assert tool.invocations == 0
    assert len(inner.requests) == 2
    assert not _of(events, TurnFailed)


@pytest.mark.asyncio
async def test_per_call_timeout_is_f8_not_an_outer_deadline() -> None:
    async def slow(arguments: Mapping[str, Any]) -> HostedToolSuccess:
        await asyncio.sleep(30.0)
        raise AssertionError("unreachable")

    inner = _FakeInner(
        (
            _Round(tool_calls=(("call_a", "web_search", '{"query":"q"}'),)),
            _Round(text="The search timed out; answering from prior knowledge."),
        )
    )
    tool = _CountingTool("web_search", slow)
    starter, _ = _starter(inner, (tool,), per_call_timeout_s=0.05)
    events, _ = await _drive(starter, _request(({"type": "web_search"},)))

    receipts = _of(events, HostedCallCompleted)
    assert len(receipts) == 1
    assert receipts[0].receipt["error"]["code"] == "tool_timeout"
    assert tool.invocations == 1
    assert len(inner.requests) == 2
    assert not _of(events, TurnFailed)
    assert _of(events, TurnCompleted)


@pytest.mark.asyncio
async def test_successful_hosted_execution_grounds_one_more_round() -> None:
    inner = _FakeInner(
        (
            _Round(tool_calls=(("call_a", "web_search", '{"query":"loctree"}'),)),
            _Round(text="Loctree is a structural perception tool."),
        )
    )
    tool = _CountingTool("web_search", _ok_behavior)
    starter, _ = _starter(inner, (tool,))
    events, _ = await _drive(starter, _request(({"type": "web_search"},)))

    started = _of(events, HostedCallStarted)
    receipts = _of(events, HostedCallCompleted)
    assert len(started) == 1 and len(receipts) == 1
    assert events.index(started[0]) < events.index(receipts[0])
    assert receipts[0].status == "completed"
    assert receipts[0].receipt["final_url"] == "https://ok.example"
    assert "error" not in receipts[0].receipt
    assert tool.invocations == 1

    # V4: the outer usage equals the monotone sum over both child rounds.
    completed = _of(events, TurnCompleted)[0]
    assert completed.usage is not None
    assert completed.usage.input_tokens == 20
    assert completed.usage.output_tokens == 10
    usage_events = _of(events, UsageUpdate)
    totals = [event.total_tokens for event in usage_events]
    assert totals == sorted(totals)

    # V7/I7: no child terminal leaked (exactly one outer terminal).
    assert len(_of(events, TurnCompleted)) == 1
    assert not _of(events, TurnFailed) and not _of(events, TurnCancelled)
    # The model's function_call was consumed, not re-emitted (S5).
    kinds = [e.kind for e in _of(events, OutputItemStarted)]
    assert "function_call" not in kinds
    assert kinds.count("hosted_call") == 1


@pytest.mark.asyncio
async def test_post_failure_hosted_call_is_not_executed() -> None:
    """Falsification 10 / I8: the terminal continuation cannot spend tools."""

    inner = _FakeInner(
        (
            _Round(tool_calls=(("call_a", "web_search", '{"query":"q"}'),)),
            _Round(
                text="The tool failed; trying to search again anyway.",
                tool_calls=(("call_b", "web_search", '{"query":"retry"}'),),
            ),
        )
    )
    tool = _CountingTool(
        "web_search",
        _raising_behavior("provider_auth_failed", "auth"),
    )
    starter, _ = _starter(inner, (tool,))
    events, _ = await _drive(starter, _request(({"type": "web_search"},)))

    assert tool.invocations == 1
    assert len(inner.requests) == 2
    receipts = _of(events, HostedCallCompleted)
    assert len(receipts) == 2
    codes = [r.receipt["error"]["code"] for r in receipts]
    assert codes == ["provider_auth_failed", "continuation_exhausted"]
    assert _of(events, TurnCompleted)
    assert not _of(events, TurnFailed)


@pytest.mark.asyncio
async def test_round_limit_is_a_receipt_plus_terminal_continuation() -> None:
    inner = _FakeInner(
        (
            _Round(tool_calls=(("call_a", "web_search", '{"query":"a"}'),)),
            _Round(
                text="Continuing.",
                tool_calls=(("call_b", "web_search", '{"query":"b"}'),),
            ),
            _Round(text="I hit the tool round limit; here is my best answer."),
        )
    )
    tool = _CountingTool("web_search", _ok_behavior)
    starter, _ = _starter(inner, (tool,), max_tool_rounds=1)
    events, _ = await _drive(starter, _request(({"type": "web_search"},)))

    assert tool.invocations == 1
    assert len(inner.requests) == 3
    receipts = _of(events, HostedCallCompleted)
    assert receipts[-1].receipt["error"]["code"] == "tool_round_limit"
    assert _of(events, TurnCompleted)
    assert not _of(events, TurnFailed)


@pytest.mark.asyncio
async def test_cancel_during_execution_stops_with_zero_continuation() -> None:
    """Falsification 6 / F11a: no receipt event, no continuation, cancelled."""

    invoked = asyncio.Event()

    async def blocking(arguments: Mapping[str, Any]) -> HostedToolSuccess:
        invoked.set()
        await asyncio.sleep(30.0)
        raise AssertionError("unreachable")

    inner = _FakeInner(
        (_Round(tool_calls=(("call_a", "web_search", '{"query":"q"}'),)),),
    )
    tool = _CountingTool("web_search", blocking)
    starter, _ = _starter(inner, (tool,))

    outer = GenerationTurn(max_pending_events=512)
    subscription = outer.subscribe(max_pending_events=512)
    handle = await starter.start(
        _request(({"type": "web_search"},)),
        outer,
        cancel=FirstWriterCancelToken(),
    )
    await invoked.wait()
    handle.cancel("client_disconnected")
    await handle.wait_closed()
    events = [item.event async for item in subscription]

    assert len(inner.requests) == 1  # zero continuation spend
    assert _of(events, HostedCallStarted)  # started events only
    assert not _of(events, HostedCallCompleted)
    cancelled = _of(events, TurnCancelled)
    assert len(cancelled) == 1
    assert cancelled[0].reason == "client_disconnected"
    assert not _of(events, TurnCompleted)


@pytest.mark.asyncio
async def test_absolute_deadline_is_f11c_504_with_zero_continuation() -> None:
    async def blocking(arguments: Mapping[str, Any]) -> HostedToolSuccess:
        await asyncio.sleep(30.0)
        raise AssertionError("unreachable")

    inner = _FakeInner(
        (_Round(tool_calls=(("call_a", "web_search", '{"query":"q"}'),)),),
    )
    tool = _CountingTool("web_search", blocking)
    starter, _ = _starter(inner, (tool,), deadline_s=0.2)
    events, _ = await _drive(starter, _request(({"type": "web_search"},)))

    assert len(inner.requests) == 1
    failed = _of(events, TurnFailed)
    assert len(failed) == 1
    assert failed[0].code == "deadline_exceeded"
    assert failed[0].status_code == 504
    assert not _of(events, TurnCompleted)
    assert not _of(events, HostedCallCompleted)


@pytest.mark.asyncio
async def test_no_admitted_hosted_tool_gets_honest_no_web_preparation() -> None:
    """Row 14: zero hosted events, trusted preparation forbids web claims."""

    inner = _FakeInner((_Round(text="I cannot browse; from what I know..."),))
    tool = _CountingTool("web_search", _ok_behavior)
    starter, _ = _starter(inner, (tool,))
    events, _ = await _drive(starter, _request(()))

    assert len(inner.requests) == 1
    first_message = inner.requests[0].messages[0]
    assert first_message["role"] == "system"
    assert first_message["content"] == NO_WEB_PREPARATION
    assert not _of(events, HostedCallStarted)
    assert not _of(events, HostedCallCompleted)
    assert len(_of(events, TurnCompleted)) == 1
    assert tool.invocations == 0


@pytest.mark.asyncio
async def test_empty_catalog_is_a_transparent_pass_through() -> None:
    inner = _FakeInner((_Round(text="plain answer"),))
    catalog = HostedToolCatalog()
    starter = HostedAgenticRuntimeStarter(
        inner,
        catalog=catalog,
        executor=HostedToolExecutor(catalog),
    )
    outer = GenerationTurn(max_pending_events=64)
    handle = await starter.start(
        _request(()),
        outer,
        cancel=FirstWriterCancelToken(),
    )
    await handle.wait_closed()
    # The inner service received the outer sink itself: zero interposition.
    assert inner.sinks == [outer]
    assert isinstance(handle, _FakeBackendTurn)


@pytest.mark.asyncio
async def test_mixed_hosted_and_client_tools_never_reach_inference() -> None:
    inner = _FakeInner(())
    tool = _CountingTool("web_search", _ok_behavior)
    starter, _ = _starter(inner, (tool,))
    with pytest.raises(RuntimeStartError):
        await starter.start(
            _request(
                (
                    {"type": "web_search"},
                    {"type": "function", "name": "my_client_fn"},
                )
            ),
            GenerationTurn(max_pending_events=64),
            cancel=FirstWriterCancelToken(),
        )
    assert inner.requests == []
    assert tool.invocations == 0


@pytest.mark.asyncio
async def test_client_function_tools_pass_through_unexecuted() -> None:
    inner = _FakeInner(
        (
            _Round(
                text="calling your tool",
                tool_calls=(("call_c", "my_client_fn", "{}"),),
            ),
        )
    )
    tool = _CountingTool("web_search", _ok_behavior)
    starter, _ = _starter(inner, (tool,))
    events, _ = await _drive(
        starter,
        _request(({"type": "function", "name": "my_client_fn"},)),
    )

    assert tool.invocations == 0
    assert len(inner.requests) == 1
    kinds = [e.kind for e in _of(events, OutputItemStarted)]
    assert "function_call" in kinds  # forwarded to the client, not consumed
    assert not _of(events, HostedCallStarted)
    assert _of(events, TurnCompleted)


# -- W3-HA2b integrity-recovery falsifiers ----------------------------------

_SECRET_SENTINEL = "sk-SENTINEL-LEAK-XYZ"


@pytest.mark.asyncio
async def test_unexpected_exception_text_never_reaches_any_surface() -> None:
    """HA2b-1: a secret living only in an exception string stays inside."""

    async def leaking_crash(arguments: Mapping[str, Any]) -> HostedToolSuccess:
        raise RuntimeError(f"Authorization: Bearer {_SECRET_SENTINEL}")

    inner = _FakeInner(
        (
            _Round(tool_calls=(("call_a", "web_search", '{"query":"q"}'),)),
            _Round(text="The tool crashed; answering without it."),
        )
    )
    tool = _CountingTool("web_search", leaking_crash)
    starter, _ = _starter(inner, (tool,))
    events, _ = await _drive(starter, _request(({"type": "web_search"},)))

    receipts = _of(events, HostedCallCompleted)
    assert len(receipts) == 1
    error = receipts[0].receipt["error"]
    assert error["code"] == "tool_execution_failed"
    assert error["message"] == "hosted tool execution failed unexpectedly"
    # The sentinel is absent from every emitted event and every message the
    # continuation model round receives.
    assert _SECRET_SENTINEL not in repr(events)
    import json as _json

    assert _SECRET_SENTINEL not in _json.dumps(
        [dict(m) for m in inner.requests[1].messages], default=str
    )
    assert _of(events, TurnCompleted) and not _of(events, TurnFailed)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "receipt_fields",
    (
        {"debug": {"authorization": _SECRET_SENTINEL}},  # unknown nested payload
        {"status": "completed"},  # core-field override
        {"call_id": "call_forged"},  # call identity override
        {"result_count": "12"},  # wrong type
        {"final_url": object()},  # non-serializable value
    ),
)
async def test_success_receipt_extras_are_a_closed_schema(
    receipt_fields: Mapping[str, Any],
) -> None:
    """HA2b-2: arbitrary receipt extras fail closed as invalid_tool_result."""

    async def decorated(arguments: Mapping[str, Any]) -> HostedToolSuccess:
        return HostedToolSuccess(payload={"ok": True}, receipt_fields=receipt_fields)

    inner = _FakeInner(
        (
            _Round(tool_calls=(("call_a", "web_search", '{"query":"q"}'),)),
            _Round(text="The tool result was invalid; continuing without it."),
        )
    )
    tool = _CountingTool("web_search", decorated)
    starter, _ = _starter(inner, (tool,))
    events, _ = await _drive(starter, _request(({"type": "web_search"},)))

    receipts = _of(events, HostedCallCompleted)
    assert len(receipts) == 1
    assert receipts[0].status == "failed"
    assert receipts[0].receipt["error"]["code"] == "invalid_tool_result"
    for key in receipt_fields:
        if key not in {"call_id", "status"}:
            assert key not in receipts[0].receipt
    assert _SECRET_SENTINEL not in repr(events)
    assert _of(events, TurnCompleted) and not _of(events, TurnFailed)


def test_error_receipts_carry_no_url_or_citation_authority() -> None:
    """HA2b-2b: build_receipt rejects extras on any error receipt."""

    from mlx_batch_server.tools.hosted import build_receipt

    with pytest.raises(ValueError):
        build_receipt(
            call_id="call_a",
            tool_name="web_fetch",
            status="failed",
            duration_ms=1,
            error={"code": "tool_execution_failed", "message": "x"},
            extra={"final_url": "https://forged.example"},
        )


@pytest.mark.asyncio
async def test_conflicting_duplicate_call_ids_are_an_outer_f12() -> None:
    """HA2b-3: same call_id with different payloads fails the turn first."""

    inner = _FakeInner(
        (
            _Round(
                tool_calls=(
                    ("call_a", "web_search", '{"query":"a"}'),
                    ("call_a", "web_search", '{"query":"b"}'),
                )
            ),
            _Round(text="unreachable continuation"),
        )
    )
    tool = _CountingTool("web_search", _ok_behavior)
    starter, _ = _starter(inner, (tool,))
    events, _ = await _drive(starter, _request(({"type": "web_search"},)))

    failed = _of(events, TurnFailed)
    assert len(failed) == 1
    assert failed[0].status_code == 500
    assert tool.invocations == 0  # no hosted execution
    assert not _of(events, HostedCallStarted)  # no receipt surface at all
    assert not _of(events, HostedCallCompleted)
    assert len(inner.requests) == 1  # no continuation
    assert not _of(events, TurnCompleted)


@pytest.mark.asyncio
async def test_identical_duplicate_call_ids_execute_once() -> None:
    """HA2b-3b: a verbatim stream duplicate is collapsed, not failed."""

    inner = _FakeInner(
        (
            _Round(
                tool_calls=(("call_a", "web_search", '{"query":"a"}'),),
                repeat_last_tool_call=True,
            ),
            _Round(text="One search was enough."),
        )
    )
    tool = _CountingTool("web_search", _ok_behavior)
    starter, _ = _starter(inner, (tool,))
    events, _ = await _drive(starter, _request(({"type": "web_search"},)))

    assert tool.invocations == 1
    assert len(_of(events, HostedCallCompleted)) == 1
    assert _of(events, TurnCompleted) and not _of(events, TurnFailed)


@pytest.mark.asyncio
async def test_deadline_actively_cancels_and_drains_the_child_backend() -> None:
    """HA2b-4: F11c must stop a child that ignores the shared token."""

    inner = _FakeInner(
        # The hold is never set from the outside: this fake backend ignores
        # the shared cancel token and only stops when its cancel() is called.
        (_Round(hold=asyncio.Event()),),
    )
    tool = _CountingTool("web_search", _ok_behavior)
    starter, _ = _starter(inner, (tool,), deadline_s=0.2)
    events, _ = await _drive(starter, _request(({"type": "web_search"},)))

    failed = _of(events, TurnFailed)
    assert len(failed) == 1
    assert failed[0].code == "deadline_exceeded"
    assert failed[0].status_code == 504
    assert not _of(events, TurnCompleted)
    assert len(inner.requests) == 1  # zero continuation
    # The child was explicitly cancelled and observably drained before the
    # hosted facade closed.
    child = inner.turns[0]
    assert child._cancelled is not None
    assert child._task.done()


@pytest.mark.asyncio
async def test_concurrent_requests_have_isolated_execution_scopes() -> None:
    """HA2b-5: no request inherits another's deadline or cancel token."""

    from mlx_batch_server.tools.hosted import (
        HostedExecutionScope,
        current_execution_scope,
    )

    scopes: dict[str, HostedExecutionScope] = {}
    both_entered = asyncio.Event()

    async def probe(arguments: Mapping[str, Any]) -> HostedToolSuccess:
        scopes[arguments["query"]] = current_execution_scope()
        if len(scopes) == 2:
            both_entered.set()
        await asyncio.wait_for(both_entered.wait(), timeout=2.0)
        return HostedToolSuccess(payload={"ok": True})

    def build(inner: _FakeInner, deadline_s: float, query: str):
        rounds = (
            _Round(tool_calls=(("call_a", "web_search", f'{{"query":"{query}"}}'),)),
            _Round(text="done"),
        )
        inner.script = rounds
        tool = _CountingTool("web_search", probe)
        return _starter(inner, (tool,), deadline_s=deadline_s)[0]

    inner_a, inner_b = _FakeInner(()), _FakeInner(())
    starter_a = build(inner_a, 5.0, "a")
    starter_b = build(inner_b, 9.0, "b")
    token_a, token_b = FirstWriterCancelToken(), FirstWriterCancelToken()

    async def drive(starter, request, token):
        outer = GenerationTurn(max_pending_events=512)
        handle = await starter.start(request, outer, cancel=token)
        await handle.wait_closed()

    req_a = _request(({"type": "web_search"},))
    req_b = GenerationRequest(
        response_id="resp_hosted_b",
        runtime=RuntimeKey(model_id="model-x"),
        messages=({"role": "user", "content": "b"},),
        tools=({"type": "web_search"},),
    )
    await asyncio.gather(
        drive(starter_a, req_a, token_a),
        drive(starter_b, req_b, token_b),
    )

    scope_a, scope_b = scopes["a"], scopes["b"]
    assert scope_a.cancel is token_a
    assert scope_b.cancel is token_b
    assert scope_a.deadline is not None and scope_b.deadline is not None
    # The two overlapping requests observed their own absolute budgets.
    assert 3.0 < (scope_b.deadline - scope_a.deadline) < 5.0


@pytest.mark.asyncio
async def test_terminal_sink_rejection_is_reported_not_swallowed() -> None:
    """HA2b-6: a refused terminal is a visible fault, never quiet success."""

    from mlx_batch_server.runtime.agentic import HostedTerminalDeliveryError

    class _RejectingSink:
        def __init__(self) -> None:
            self.events: list[Any] = []

        def emit(self, event: Any) -> None:
            if isinstance(event, TurnCompleted):
                raise RuntimeError("sink refused the terminal")
            self.events.append(event)

    inner = _FakeInner((_Round(text="plain answer"),))
    tool = _CountingTool("web_search", _ok_behavior)
    starter, _ = _starter(inner, (tool,))
    sink = _RejectingSink()
    handle = await starter.start(
        _request(({"type": "web_search"},)),
        sink,
        cancel=FirstWriterCancelToken(),
    )
    with pytest.raises(HostedTerminalDeliveryError):
        await handle.wait_closed()
    assert not any(isinstance(event, TurnCompleted) for event in sink.events)


@pytest.mark.asyncio
async def test_inconsistent_success_receipt_is_a_server_fault_500() -> None:
    """HA2b-7: a receipt disagreeing with its closing events is F12."""

    from mlx_batch_server.tools.agent_loop import ToolExecutionResult

    class _InconsistentExecutor(HostedToolExecutor):
        async def execute(self, call: Any) -> ToolExecutionResult:
            result = await super().execute(call)
            metadata = dict(result.metadata or {})
            receipt = dict(metadata["receipt"])
            receipt["tool_name"] = "someone_else"
            metadata["receipt"] = receipt
            return ToolExecutionResult(
                call_id=result.call_id,
                output=result.output,
                metadata=metadata,
            )

    inner = _FakeInner(
        (_Round(tool_calls=(("call_a", "web_search", '{"query":"q"}'),)),),
    )
    tool = _CountingTool("web_search", _ok_behavior)
    catalog = HostedToolCatalog((tool,))
    starter = HostedAgenticRuntimeStarter(
        inner,
        catalog=catalog,
        executor=_InconsistentExecutor(catalog),
    )
    events, _ = await _drive(starter, _request(({"type": "web_search"},)))

    failed = _of(events, TurnFailed)
    assert len(failed) == 1
    assert failed[0].status_code == 500
    assert not _of(events, TurnCompleted)


@pytest.mark.asyncio
async def test_executor_call_id_mismatch_is_a_server_fault_500() -> None:
    """F12/row 15: contract violation fails the outer turn, no fake receipt."""

    class _LyingExecutor(HostedToolExecutor):
        async def execute(self, call):  # type: ignore[override]
            result = await super().execute(call)
            from mlx_batch_server.tools.agent_loop import ToolExecutionResult

            return ToolExecutionResult(
                call_id="call_someone_else",
                output=result.output,
                metadata=result.metadata,
            )

    inner = _FakeInner(
        (_Round(tool_calls=(("call_a", "web_search", '{"query":"q"}'),)),),
    )
    tool = _CountingTool("web_search", _ok_behavior)
    catalog = HostedToolCatalog((tool,))
    starter = HostedAgenticRuntimeStarter(
        inner,
        catalog=catalog,
        executor=_LyingExecutor(catalog),
    )
    events, _ = await _drive(starter, _request(({"type": "web_search"},)))

    failed = _of(events, TurnFailed)
    assert len(failed) == 1
    assert failed[0].status_code == 500
    assert len(inner.requests) == 1  # no continuation after a server fault
    assert not _of(events, TurnCompleted)
