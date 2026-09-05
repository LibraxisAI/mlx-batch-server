"""Delivery verifier for the hosted failure-continuation runtime (design §8-§9).

A deterministic fake backend drives ``HostedAgenticRuntimeStarter`` through
successful hosted execution and each tool-level failure class. For every
failure it proves: one executor attempt, one immutable typed error receipt,
exactly one terminal continuation round, non-empty explanatory model text, and
one completed outer turn — never an abrupt outer failure.
"""

from __future__ import annotations

import asyncio
import gc
import hashlib
import weakref
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import pytest

from mlx_batch_server.runtime import agentic as agentic_module
from mlx_batch_server.runtime.agentic import (
    CITATIONS_METADATA_KEY,
    FAILURE_CONTINUATION_PREPARATION,
    NO_WEB_PREPARATION,
    HostedAgenticRuntimeStarter,
)
from mlx_batch_server.runtime.citations import (
    CITATION_PREPARATION,
    CitationStreamFilter,
)
from mlx_batch_server.runtime.contracts import GenerationRequest, RuntimeKey, TurnSink
from mlx_batch_server.runtime.events import (
    ContentPartCompleted,
    ContentPartStarted,
    HostedCallCompleted,
    HostedCallResult,
    HostedCallStarted,
    HostedCitation,
    OutputItemCompleted,
    OutputItemStarted,
    ReasoningCompleted,
    ReasoningDelta,
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
from mlx_batch_server.tools.agent_loop import ToolExecutionResult
from mlx_batch_server.tools.hosted import (
    HostedToolCatalog,
    HostedToolError,
    HostedToolExecutor,
    HostedToolSuccess,
    build_receipt,
    canonical_json,
)
from mlx_batch_server.tools.hosted_web import HostedWebSearchTool
from mlx_batch_server.tools.parser import ParsedToolCall


@dataclass(slots=True)
class _Round:
    """One scripted model round for the deterministic fake backend."""

    text: str | None = None
    reasoning: str | None = None
    tool_calls: tuple[tuple[str, str, str], ...] = ()  # (call_id, name, args)
    usage: tuple[int, int] = (10, 5)
    hold: asyncio.Event | None = None
    repeat_last_tool_call: bool = False  # stream-level identical duplicate
    text_deltas: tuple[str, ...] | None = None  # arbitrary delta chunking


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
        if step.reasoning is not None:
            item_id = f"reasoning_r{rnd}"
            sink.emit(OutputItemStarted("reasoning", index, item_id))
            sink.emit(ContentPartStarted("reasoning_summary_text", index, 0, item_id))
            sink.emit(ReasoningDelta(step.reasoning, item_id, index, 0))
            sink.emit(ReasoningCompleted(step.reasoning, item_id, index, 0))
            sink.emit(
                ContentPartCompleted(
                    "reasoning_summary_text",
                    index,
                    0,
                    item_id,
                    step.reasoning,
                )
            )
            sink.emit(
                OutputItemCompleted("reasoning", index, item_id, text=step.reasoning)
            )
            index += 1
        if step.text is not None:
            item_id = f"msg_r{rnd}"
            sink.emit(OutputItemStarted("message", index, item_id))
            sink.emit(ContentPartStarted("output_text", index, 0, item_id))
            for delta in step.text_deltas or (step.text,):
                sink.emit(TextDelta(delta, item_id, index, 0))
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


def _search_success(
    query: str,
    results: list[dict[str, str]],
) -> HostedToolSuccess:
    digest = (
        "sha256:"
        + hashlib.sha256(
            canonical_json({"query": query, "results": results}).encode("utf-8")
        ).hexdigest()
    )
    return HostedToolSuccess(
        payload={"query": query, "results": results},
        receipt_fields={"result_count": len(results), "result_digest": digest},
        result={
            "kind": "search_results",
            "query": query,
            "results": [dict(entry) for entry in results],
            "digest": digest,
        },
    )


def _document_success(
    url: str,
    content: str,
    media_type: str = "text/plain",
) -> HostedToolSuccess:
    digest = f"sha256:{hashlib.sha256(content.encode('utf-8')).hexdigest()}"
    return HostedToolSuccess(
        payload={"url": url, "media_type": media_type, "content": content},
        receipt_fields={
            "final_url": url,
            "mime": media_type,
            "result_digest": digest,
        },
        result={
            "kind": "document",
            "url": url,
            "media_type": media_type,
            "content": content,
            "digest": digest,
            "retrieved_at": 1,
        },
    )


_OK_RESULTS = [
    {
        "title": "t",
        "url": "https://ok.example",
        "snippet": "Loctree is a structural perception tool.",
    }
]


async def _ok_behavior(arguments: Mapping[str, Any]) -> HostedToolSuccess:
    return _search_success(str(arguments.get("query", "q")), _OK_RESULTS)


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
    assert receipts[0].receipt["result_count"] == 1
    assert receipts[0].receipt["result_digest"].startswith("sha256:")
    assert "error" not in receipts[0].receipt
    assert tool.invocations == 1

    # HR2-4: exactly one HostedCallResult between started and receipt, and the
    # closing item carries the validated sealed action with proven sources.
    results = _of(events, HostedCallResult)
    assert len(results) == 1
    assert events.index(started[0]) < events.index(results[0])
    assert events.index(results[0]) < events.index(receipts[0])
    assert results[0].result["kind"] == "search_results"
    assert results[0].result["digest"] == receipts[0].receipt["result_digest"]
    hosted_items = [
        e for e in _of(events, OutputItemCompleted) if e.kind == "hosted_call"
    ]
    assert len(hosted_items) == 1
    action = hosted_items[0].action
    assert action is not None
    assert action["kind"] == "search"
    assert action["query"] == "loctree"
    assert action["sources"] == ("https://ok.example",)
    assert events.index(receipts[0]) < events.index(hosted_items[0])

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
        return _search_success(str(arguments["query"]), [])

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


# -- W3-HR2-4 result emission, aggregate budget, citation filter -------------

_DOC_URL = "https://doc.example/loctree"
_DOC_CONTENT = (
    "Loctree gives structural sight before you touch anything.\n"
    "It maps  the blast radius."
)
_QUOTE = "It maps the blast radius."


async def _doc_behavior(arguments: Mapping[str, Any]) -> HostedToolSuccess:
    return _document_success(str(arguments["url"]), _DOC_CONTENT)


def _cited_request(tools: tuple[Mapping[str, Any], ...]) -> GenerationRequest:
    return GenerationRequest(
        response_id="resp_hosted",
        runtime=RuntimeKey(model_id="model-x"),
        messages=({"role": "user", "content": "co pisza o loctree?"},),
        tools=tools,
        metadata={CITATIONS_METADATA_KEY: True},
    )


@pytest.mark.asyncio
async def test_success_without_result_payload_is_f12() -> None:
    """HRPD §1.3: a hosted success carrying no canonical result is F12."""

    async def bare(arguments: Mapping[str, Any]) -> HostedToolSuccess:
        return HostedToolSuccess(payload={"ok": True})

    inner = _FakeInner(
        (_Round(tool_calls=(("call_a", "web_search", '{"query":"q"}'),)),),
    )
    tool = _CountingTool("web_search", bare)
    starter, _ = _starter(inner, (tool,))
    events, _ = await _drive(starter, _request(({"type": "web_search"},)))

    failed = _of(events, TurnFailed)
    assert len(failed) == 1 and failed[0].status_code == 500
    assert not _of(events, HostedCallResult)
    assert not _of(events, HostedCallCompleted)
    assert len(inner.requests) == 1
    assert not _of(events, TurnCompleted)


class _ResultMutatingExecutor(HostedToolExecutor):
    """Post-validation mutation double: forges one result payload field."""

    mutate_field = "digest"
    mutate_value: Any = "sha256:" + "0" * 64

    async def execute(self, call: Any) -> ToolExecutionResult:
        result = await super().execute(call)
        metadata = dict(result.metadata or {})
        payload = dict(metadata["result"])
        payload[type(self).mutate_field] = type(self).mutate_value
        metadata["result"] = payload
        return ToolExecutionResult(
            call_id=result.call_id,
            output=result.output,
            metadata=metadata,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    (
        ("digest", "sha256:" + "0" * 64),
        ("url", "https://forged.example"),
        ("media_type", "text/forged"),
    ),
)
async def test_result_receipt_identity_disagreement_is_f12(
    field: str,
    value: str,
) -> None:
    """Digest and fetch URL/MIME identity laws: disagreement is never success."""

    executor_cls = type(
        f"_Mutate_{field}",
        (_ResultMutatingExecutor,),
        {"mutate_field": field, "mutate_value": value},
    )
    inner = _FakeInner(
        (_Round(tool_calls=(("call_a", "web_fetch", f'{{"url":"{_DOC_URL}"}}'),)),),
    )
    tool = _CountingTool("web_fetch", _doc_behavior)
    catalog = HostedToolCatalog((tool,))
    starter = HostedAgenticRuntimeStarter(
        inner,
        catalog=catalog,
        executor=executor_cls(catalog),
    )
    events, _ = await _drive(starter, _request(({"type": "web_fetch"},)))

    failed = _of(events, TurnFailed)
    assert len(failed) == 1 and failed[0].status_code == 500
    assert not _of(events, HostedCallResult)
    assert not _of(events, HostedCallCompleted)
    assert len(inner.requests) == 1


@pytest.mark.asyncio
async def test_aggregate_budget_drops_only_the_overflowing_call() -> None:
    """Deterministic charging in model call order; prior results stay valid."""

    contents = {
        "https://a.example": "A" * 40,
        "https://b.example": "B" * 40,
    }

    async def by_url(arguments: Mapping[str, Any]) -> HostedToolSuccess:
        url = str(arguments["url"])
        return _document_success(url, contents[url])

    inner = _FakeInner(
        (
            _Round(
                tool_calls=(
                    ("call_a", "web_fetch", '{"url":"https://a.example"}'),
                    ("call_b", "web_fetch", '{"url":"https://b.example"}'),
                )
            ),
            _Round(text="The second fetch exceeded the result budget."),
        )
    )
    tool = _CountingTool("web_fetch", by_url)
    starter, _ = _starter(inner, (tool,), max_result_chars_total=60)
    events, _ = await _drive(starter, _request(({"type": "web_fetch"},)))

    results = _of(events, HostedCallResult)
    assert len(results) == 1
    assert results[0].call_id == "call_a"
    receipts = _of(events, HostedCallCompleted)
    assert [r.call_id for r in receipts] == ["call_a", "call_b"]
    assert [r.status for r in receipts] == ["completed", "failed"]
    assert receipts[1].receipt["error"]["code"] == "result_budget_exceeded"

    # One terminal continuation with the trusted failure preparation.
    continuation = inner.requests[1].messages
    preparations = [
        m
        for m in continuation
        if m.get("role") == "system"
        and m.get("content") == FAILURE_CONTINUATION_PREPARATION
    ]
    assert len(preparations) == 1
    assert _of(events, TurnCompleted) and not _of(events, TurnFailed)

    # The failed item still closes with its sealed fetch action.
    failed_items = [
        e
        for e in _of(events, OutputItemCompleted)
        if e.kind == "hosted_call" and e.status == "failed"
    ]
    assert len(failed_items) == 1
    assert failed_items[0].action == {"kind": "fetch", "url": "https://b.example"}


@pytest.mark.asyncio
async def test_f12_result_mismatch_precedes_aggregate_overflow() -> None:
    """A forged success that would overflow is F12, not a budget failure."""
    inner = _FakeInner(
        (_Round(tool_calls=(("call_a", "web_fetch", f'{{"url":"{_DOC_URL}"}}'),)),),
    )
    tool = _CountingTool("web_fetch", _doc_behavior)
    catalog = HostedToolCatalog((tool,))
    starter = HostedAgenticRuntimeStarter(
        inner,
        catalog=catalog,
        executor=_ResultMutatingExecutor(catalog),
        max_result_chars_total=1,
    )
    events, _ = await _drive(starter, _request(({"type": "web_fetch"},)))

    failed = _of(events, TurnFailed)
    assert len(failed) == 1 and failed[0].status_code == 500
    assert not _of(events, HostedCallResult)
    assert not _of(events, HostedCallCompleted)
    assert len(inner.requests) == 1


@pytest.mark.asyncio
async def test_late_result_after_cancel_forwards_nothing() -> None:
    """F11: a call completing after the token fired emits zero events."""

    inner = _FakeInner((_Round(text="plain answer"),))
    tool = _CountingTool("web_search", _ok_behavior)
    starter, _ = _starter(inner, (tool,))
    recorded: list[Any] = []

    class _Recorder:
        def emit(self, event: Any) -> None:
            recorded.append(event)

    token = FirstWriterCancelToken()
    handle = await starter.start(
        _request(({"type": "web_search"},)),
        _Recorder(),
        cancel=token,
    )
    await handle.wait_closed()
    before = len(recorded)

    token.cancel("client_disconnected")
    success = _search_success("q", _OK_RESULTS)
    receipt = build_receipt(
        call_id="call_z",
        tool_name="web_search",
        status="completed",
        duration_ms=1,
        extra=success.receipt_fields,
    )
    late = ToolExecutionResult(
        call_id="call_z",
        output="{}",
        metadata={
            "tool_name": "web_search",
            "receipt": receipt,
            "result": success.result,
        },
    )
    item = agentic_module._HostedItem(
        index=99,
        item_id="hosted_late",
        call_id="call_z",
        tool_name="web_search",
    )
    call = ParsedToolCall(
        index=0,
        call_id="call_z",
        name="web_search",
        arguments='{"query":"q"}',
    )
    with pytest.raises(asyncio.CancelledError):
        handle._emit_hosted_result_and_receipt(item, call, late)
    assert len(recorded) == before  # no payload, no receipt, no item


_RAW_CITED = (
    'Grounded: <cite url="https://doc.example/loctree">'
    "It maps the blast radius.</cite> Indeed."
)
_FILTERED_CITED = "Grounded: It maps the blast radius. Indeed."


def _cited_rounds(raw: str, deltas: tuple[str, ...]) -> tuple[_Round, ...]:
    return (
        _Round(tool_calls=(("call_a", "web_fetch", f'{{"url":"{_DOC_URL}"}}'),)),
        _Round(text=raw, text_deltas=deltas),
    )


@pytest.mark.asyncio
async def test_armed_filter_grounds_one_citation_and_strips_markup() -> None:
    deltas = (
        "Grounded: <ci",
        'te url="https://doc.example/loctree">It maps the blast',
        " radius.</ci",
        "te> Indeed.",
    )
    inner = _FakeInner(_cited_rounds(_RAW_CITED, deltas))
    tool = _CountingTool("web_fetch", _doc_behavior)
    starter, _ = _starter(inner, (tool,))
    events, outer = await _drive(starter, _cited_request(({"type": "web_fetch"},)))

    # No raw control markup on any surface; the turn contract stayed green.
    assert "<cite" not in repr(events)
    assert outer.state is TurnState.TERMINAL
    assert _of(events, TurnCompleted) and not _of(events, TurnFailed)

    # Exactly one execution, one result event, one proven citation.
    assert tool.invocations == 1  # no second fetch anywhere (filter included)
    assert len(_of(events, HostedCallResult)) == 1
    citations = _of(events, HostedCitation)
    assert len(citations) == 1
    citation = citations[0]
    assert citation.source_call_id == "call_a"
    assert citation.source_url == _DOC_URL
    assert citation.cited_text == _QUOTE
    assert citation.output_start == len("Grounded: ")
    assert citation.output_end == len("Grounded: ") + len(_QUOTE)
    assert citation.source_start == _DOC_CONTENT.index("It maps")
    assert citation.source_end == len(_DOC_CONTENT)

    # Filtered deltas, done-text and item completion agree exactly.
    message_texts = [e for e in _of(events, TextCompleted) if e.text == _FILTERED_CITED]
    assert len(message_texts) == 1
    part = (message_texts[0].output_index, message_texts[0].content_index)
    delta_concat = "".join(
        e.delta
        for e in _of(events, TextDelta)
        if (e.output_index, e.content_index) == part
    )
    assert delta_concat == _FILTERED_CITED
    completed_parts = [
        e
        for e in _of(events, ContentPartCompleted)
        if e.output_index == part[0] and e.content_index == part[1]
    ]
    assert len(completed_parts) == 1
    assert completed_parts[0].text == _FILTERED_CITED
    message_items = [
        e
        for e in _of(events, OutputItemCompleted)
        if e.kind == "message" and e.text == _FILTERED_CITED
    ]
    assert len(message_items) == 1

    # The trusted citation preparation entered the continuation exactly once.
    preparations = [
        m
        for m in inner.requests[1].messages
        if m.get("role") == "system" and m.get("content") == CITATION_PREPARATION
    ]
    assert len(preparations) == 1


@pytest.mark.asyncio
async def test_split_sentinels_produce_identical_output_for_every_boundary() -> None:
    """Property: delta boundary placement can never change the public output."""

    baseline_events: list[Any] | None = None
    for split in range(1, len(_RAW_CITED)):
        deltas = (_RAW_CITED[:split], _RAW_CITED[split:])
        inner = _FakeInner(_cited_rounds(_RAW_CITED, deltas))
        tool = _CountingTool("web_fetch", _doc_behavior)
        starter, _ = _starter(inner, (tool,))
        events, _ = await _drive(starter, _cited_request(({"type": "web_fetch"},)))
        assert "<cite" not in repr(events)
        texts = [e.text for e in _of(events, TextCompleted)]
        citations = [
            (
                c.source_url,
                c.cited_text,
                c.source_start,
                c.source_end,
                c.output_start,
                c.output_end,
            )
            for c in _of(events, HostedCitation)
        ]
        observed = (texts, citations)
        if baseline_events is None:
            baseline_events = observed
        else:
            assert observed == baseline_events, f"split at {split} diverged"
    assert baseline_events is not None
    assert baseline_events[0].count(_FILTERED_CITED) == 1
    assert len(baseline_events[1]) == 1


@pytest.mark.asyncio
async def test_unknown_url_citation_is_stripped_without_event() -> None:
    raw = 'See <cite url="https://evil.example">It maps the blast radius.</cite>!'
    inner = _FakeInner(_cited_rounds(raw, (raw,)))
    tool = _CountingTool("web_fetch", _doc_behavior)
    starter, _ = _starter(inner, (tool,))
    events, _ = await _drive(starter, _cited_request(({"type": "web_fetch"},)))

    assert "<cite" not in repr(events)
    assert not _of(events, HostedCitation)
    texts = [e.text for e in _of(events, TextCompleted)]
    assert "See It maps the blast radius.!" in texts
    assert _of(events, TurnCompleted) and not _of(events, TurnFailed)


@pytest.mark.asyncio
async def test_malformed_and_nested_markup_never_reaches_public_events() -> None:
    """Malformed and nested markers preserve human text with zero citation."""

    raw = (
        "Broken <cite>human</cite>; nested "
        f'<cite url="{_DOC_URL}">outer '
        f'<cite url="{_DOC_URL}">{_QUOTE}</cite> tail</cite>.'
    )
    clean = f"Broken human; nested outer {_QUOTE} tail."
    inner = _FakeInner(_cited_rounds(raw, tuple(raw)))
    tool = _CountingTool("web_fetch", _doc_behavior)
    starter, _ = _starter(inner, (tool,))
    events, _ = await _drive(starter, _cited_request(({"type": "web_fetch"},)))

    assert all("<cite" not in repr(event) for event in events)
    assert all("</cite>" not in repr(event) for event in events)
    assert not _of(events, HostedCitation)
    assert clean in [event.text for event in _of(events, TextCompleted)]
    assert _of(events, TurnCompleted) and not _of(events, TurnFailed)


@pytest.mark.asyncio
async def test_unarmed_sentinel_text_passes_through_byte_identically() -> None:
    """No citations metadata: the baseline pass-through path is untouched."""

    inner = _FakeInner(_cited_rounds(_RAW_CITED, (_RAW_CITED,)))
    tool = _CountingTool("web_fetch", _doc_behavior)
    starter, _ = _starter(inner, (tool,))
    events, _ = await _drive(starter, _request(({"type": "web_fetch"},)))

    texts = [e.text for e in _of(events, TextCompleted)]
    assert _RAW_CITED in texts  # raw model text, byte-identical
    assert not _of(events, HostedCitation)
    preparations = [
        m
        for m in inner.requests[1].messages
        if m.get("content") == CITATION_PREPARATION
    ]
    assert not preparations


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metadata",
    (
        {},
        {CITATIONS_METADATA_KEY: False},
        {CITATIONS_METADATA_KEY: 1},
        {CITATIONS_METADATA_KEY: "true"},
        {CITATIONS_METADATA_KEY: {"enabled": True}},
        {"citations": True},
    ),
)
async def test_only_internal_literal_true_arms_citations(
    metadata: Mapping[str, Any],
) -> None:
    """Absent, truthy and client-owned metadata all fail closed."""

    assert CITATIONS_METADATA_KEY == "mlx_batch_server.internal.citations_requested"
    request = GenerationRequest(
        response_id="resp_hosted",
        runtime=RuntimeKey(model_id="model-x"),
        messages=({"role": "user", "content": "co pisza o loctree?"},),
        tools=({"type": "web_fetch"},),
        metadata=metadata,
    )
    inner = _FakeInner(_cited_rounds(_RAW_CITED, (_RAW_CITED,)))
    tool = _CountingTool("web_fetch", _doc_behavior)
    starter, _ = _starter(inner, (tool,))
    events, _ = await _drive(starter, request)

    assert _RAW_CITED in [event.text for event in _of(events, TextCompleted)]
    assert not _of(events, HostedCitation)
    assert not [
        message
        for message in inner.requests[1].messages
        if message.get("content") == CITATION_PREPARATION
    ]


@pytest.mark.asyncio
async def test_armed_ordinary_text_is_byte_identical_on_every_completion() -> None:
    """Arming the filter cannot perturb ordinary continuation text."""

    text = "Ordinary text with <citrus> and no control sentinel."
    inner = _FakeInner(_cited_rounds(text, tuple(text)))
    tool = _CountingTool("web_fetch", _doc_behavior)
    starter, _ = _starter(inner, (tool,))
    events, _ = await _drive(starter, _cited_request(({"type": "web_fetch"},)))

    completed = [event for event in _of(events, TextCompleted) if event.text == text]
    assert len(completed) == 1
    part = (completed[0].output_index, completed[0].content_index)
    assert (
        "".join(
            event.delta
            for event in _of(events, TextDelta)
            if (event.output_index, event.content_index) == part
        )
        == text
    )
    assert any(event.text == text for event in _of(events, ContentPartCompleted))
    assert any(
        event.kind == "message" and event.text == text
        for event in _of(events, OutputItemCompleted)
    )
    assert not _of(events, HostedCitation)


@pytest.mark.asyncio
async def test_armed_reasoning_path_is_byte_identical_and_never_filtered() -> None:
    """Citation-shaped reasoning remains on its untouched baseline channel."""

    reasoning = f'Reason about <cite url="{_DOC_URL}">{_QUOTE}</cite> privately.'
    inner = _FakeInner(
        (
            _Round(tool_calls=(("call_a", "web_fetch", f'{{"url":"{_DOC_URL}"}}'),)),
            _Round(reasoning=reasoning, text="Public answer."),
        )
    )
    tool = _CountingTool("web_fetch", _doc_behavior)
    starter, _ = _starter(inner, (tool,))
    events, _ = await _drive(starter, _cited_request(({"type": "web_fetch"},)))

    assert [event.delta for event in _of(events, ReasoningDelta)] == [reasoning]
    assert [event.text for event in _of(events, ReasoningCompleted)] == [reasoning]
    reasoning_parts = [
        event
        for event in _of(events, ContentPartCompleted)
        if event.kind == "reasoning_summary_text"
    ]
    assert len(reasoning_parts) == 1 and reasoning_parts[0].text == reasoning
    assert any(
        event.kind == "reasoning" and event.text == reasoning
        for event in _of(events, OutputItemCompleted)
    )
    assert not _of(events, HostedCitation)


@pytest.mark.asyncio
async def test_continuation_messages_match_baseline_exactly() -> None:
    """Golden equality vs baseline 35b95ae: armed adds only the preparation."""

    def build() -> tuple[_FakeInner, HostedAgenticRuntimeStarter]:
        inner = _FakeInner(
            (
                _Round(tool_calls=(("call_a", "web_search", '{"query":"loctree"}'),)),
                _Round(text="Loctree is a structural perception tool."),
            )
        )
        starter, _ = _starter(inner, (_CountingTool("web_search", _ok_behavior),))
        return inner, starter

    inner_plain, starter_plain = build()
    await _drive(starter_plain, _request(({"type": "web_search"},)))
    inner_armed, starter_armed = build()
    await _drive(starter_armed, _cited_request(({"type": "web_search"},)))

    expected_baseline = [
        {"role": "user", "content": "co pisza o loctree?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_a",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": '{"query":"loctree"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_a",
            "name": "web_search",
            "content": canonical_json({"query": "loctree", "results": _OK_RESULTS}),
        },
    ]
    plain = [dict(m) for m in inner_plain.requests[1].messages]
    armed = [dict(m) for m in inner_armed.requests[1].messages]
    assert plain == expected_baseline
    assert armed == [
        *expected_baseline,
        {"role": "system", "content": CITATION_PREPARATION},
    ]


@pytest.mark.asyncio
async def test_filter_state_dies_with_the_turn(monkeypatch: Any) -> None:
    """Memory-lifetime law: no filter object survives the turn (weakref probe)."""

    created: list[weakref.ref[Any]] = []

    class _Recording(CitationStreamFilter):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            created.append(weakref.ref(self))

    monkeypatch.setattr(agentic_module, "CitationStreamFilter", _Recording)
    inner = _FakeInner(_cited_rounds(_RAW_CITED, (_RAW_CITED,)))
    tool = _CountingTool("web_fetch", _doc_behavior)
    starter, _ = _starter(inner, (tool,))

    outer = GenerationTurn(max_pending_events=512)
    handle = await starter.start(
        _cited_request(({"type": "web_fetch"},)),
        outer,
        cancel=FirstWriterCancelToken(),
    )
    await handle.wait_closed()
    assert created  # the armed round really built filters

    inner.sinks.clear()
    inner.turns.clear()
    inner.requests.clear()
    del handle, starter, outer
    gc.collect()
    assert all(ref() is None for ref in created)
