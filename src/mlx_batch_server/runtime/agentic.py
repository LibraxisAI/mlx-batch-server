"""The one protocol-neutral hosted agentic runtime owner.

``HostedAgenticRuntimeStarter`` implements design/HOSTED_FAILURE_CONTINUATION:
it owns the outer turn lifecycle, runs child generation rounds through the
wrapped ``RuntimeStartService`` with private child sinks, drives ``AgentLoop``
for exactly-once hosted execution, converts every receipt into typed hosted
events, builds the single continuation input after success or failure, and
enforces the one absolute deadline plus the cancel/disconnect stop.

The class subclasses ``RuntimeStartService`` only because the Anthropic turn
source validates its starter with ``isinstance(starter, RuntimeStartService)``
and that seam is outside this cut's fence; every child round delegates to the
wrapped inner service, and no inherited state is used.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from ..tools.agent_loop import (
    AgentLoop,
    AgentLoopLimitExceeded,
    ToolExecutionResult,
    hosted_agent_loop_policy,
)
from ..tools.hosted import (
    ACTION_KIND_FOR_TOOL,
    HostedExecutionScope,
    HostedToolCatalog,
    HostedToolExecutor,
    canonical_json,
    failure_result,
    reset_execution_scope,
    result_identities,
    set_execution_scope,
    validate_result_payload,
    validate_sealed_action,
)
from ..tools.parser import ParsedToolCall
from .citations import (
    CITATION_PREPARATION,
    CitationSource,
    CitationStreamFilter,
    ItemCitationBudget,
    PreparedCitationCorpus,
    ProvenCitation,
)
from .events import (
    HOSTED_CALL_ITEM_KIND,
    TERMINAL_EVENT_TYPES,
    ContentPartCompleted,
    ContentPartStarted,
    HostedCallCompleted,
    HostedCallProgress,
    HostedCallResult,
    HostedCallStarted,
    HostedCitation,
    OutputItemCompleted,
    OutputItemStarted,
    ReasoningCompleted,
    ReasoningDelta,
    TerminalEvent,
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
from .service import FirstWriterCancelToken, RuntimeStartError, RuntimeStartService

if TYPE_CHECKING:
    from .contracts import BackendTurn, GenerationRequest, TurnSink

FAILURE_CONTINUATION_PREPARATION = (
    "A hosted tool call failed. Use the tool error result to tell the user "
    "what failed, then answer as well as you can without the tool output. "
    "Do not fabricate tool results, fetched content, or citations."
)
NO_WEB_PREPARATION = (
    "No hosted web tools are available for this request. You have no web, "
    "search, or network access. Do not claim to have browsed, fetched, or "
    "searched the web."
)

# The one internal admitted-request truth for "citations requested". It is not
# a client metadata field: W4 is the only future owner allowed to set it after
# protocol validation. Until then it is absent, so filtering fails closed and
# every public request remains byte-identical to the unfiltered baseline.
CITATIONS_METADATA_KEY = "mlx_batch_server.internal.citations_requested"


def _citations_requested(request: GenerationRequest) -> bool:
    return request.metadata.get(CITATIONS_METADATA_KEY) is True


class HostedRuntimeIntegrityError(RuntimeError):
    """A server-integrity fault (F12): outer TurnFailed 500, never a receipt."""


class HostedTerminalDeliveryError(RuntimeError):
    """The outer terminal could not be delivered to the sink.

    Never silently suppressed into apparent success: it escapes the turn task
    so the backend facade (``wait_closed``) reports the delivery fault.
    """


# Fixed audit-safe F12 text: an arbitrary exception's message may carry
# internals or secrets and never reaches the outer TurnFailed verbatim.
INTERNAL_FAILURE_MESSAGE = "hosted runtime encountered an internal error"


class HostedAgenticRuntimeStarter(RuntimeStartService):
    """Single owner of hosted failure-continuation semantics (design §1.1)."""

    def __init__(
        self,
        inner: RuntimeStartService,
        *,
        catalog: HostedToolCatalog,
        executor: HostedToolExecutor,
        max_tool_rounds: int = 8,
        deadline_s: float | None = None,
        max_result_chars_total: int = 786_432,
    ) -> None:
        # Intentionally no super().__init__: only the type is inherited (see
        # module docstring); the wrapped inner service owns backend turns.
        if not isinstance(inner, RuntimeStartService):
            raise TypeError("inner must be a RuntimeStartService")
        if not isinstance(catalog, HostedToolCatalog):
            raise TypeError("catalog must be a HostedToolCatalog")
        if not isinstance(executor, HostedToolExecutor):
            raise TypeError("executor must be a HostedToolExecutor")
        if executor.catalog is not catalog:
            raise ValueError("executor must execute exactly this hosted catalog")
        if max_tool_rounds < 1:
            raise ValueError("max_tool_rounds must be at least 1")
        if deadline_s is not None and deadline_s <= 0:
            raise ValueError("deadline_s must be positive")
        if max_result_chars_total < 1:
            raise ValueError("max_result_chars_total must be positive")
        self._inner = inner
        self._catalog = catalog
        self._executor = executor
        self._max_tool_rounds = max_tool_rounds
        self._deadline_s = deadline_s
        self._max_result_chars_total = max_result_chars_total

    @property
    def hosted_catalog(self) -> HostedToolCatalog:
        return self._catalog

    @property
    def inner(self) -> RuntimeStartService:
        return self._inner

    async def start(
        self,
        request: GenerationRequest,
        sink: TurnSink,
        *,
        cancel: FirstWriterCancelToken | None = None,
    ) -> BackendTurn:
        token = cancel or FirstWriterCancelToken()
        if not isinstance(token, FirstWriterCancelToken):
            raise TypeError("cancel must be a FirstWriterCancelToken")
        if not self._catalog:
            # No hosted capability composed: the starter is a transparent
            # pass-through and the deployment's behavior is unchanged.
            return await self._inner.start(request, sink, cancel=token)
        hosted_names = self._admitted_hosted_names(request)
        if hosted_names and self._has_client_tools(request):
            raise RuntimeStartError(
                "mixed hosted and client tools must be rejected before the runtime"
            )
        turn = _HostedAgenticTurn(
            starter=self,
            request=request,
            sink=sink,
            token=token,
            hosted_names=hosted_names,
        )
        turn.launch()
        return turn

    def _admitted_hosted_names(self, request: GenerationRequest) -> frozenset[str]:
        admitted: set[str] = set()
        for name in self._request_tool_names(request):
            if name in self._catalog.names:
                admitted.add(name)
        return frozenset(admitted)

    def _has_client_tools(self, request: GenerationRequest) -> bool:
        return any(
            name not in self._catalog.names
            for name in self._request_tool_names(request)
        )

    @staticmethod
    def _request_tool_names(request: GenerationRequest) -> tuple[str, ...]:
        names: list[str] = []
        for tool in request.tools:
            if not isinstance(tool, Mapping):
                continue
            kind = tool.get("type")
            if kind == "function":
                name = tool.get("name")
                if name is None and isinstance(tool.get("function"), Mapping):
                    name = tool["function"].get("name")
            else:
                name = tool.get("name") or kind
            if isinstance(name, str) and name:
                names.append(name)
        return tuple(names)


@dataclass(frozen=True, slots=True)
class _ChildRound:
    terminal: TerminalEvent
    tool_calls: tuple[ParsedToolCall, ...]
    saw_text: bool


@dataclass(slots=True)
class _HostedItem:
    index: int
    item_id: str
    call_id: str
    tool_name: str


class _HostedAgenticTurn:
    """BackendTurn facade owning one outer hosted lifecycle."""

    def __init__(
        self,
        *,
        starter: HostedAgenticRuntimeStarter,
        request: GenerationRequest,
        sink: TurnSink,
        token: FirstWriterCancelToken,
        hosted_names: frozenset[str],
    ) -> None:
        self._starter = starter
        self._request = request
        self._sink = sink
        self._token = token
        self._hosted_names = hosted_names
        self._loop = asyncio.get_running_loop()
        self._lock = threading.Lock()
        self._task: asyncio.Task[None] | None = None
        self._current_child: BackendTurn | None = None
        self._outer_started = False
        self._terminal_emitted = False
        self._next_index = 0
        self._used_item_ids: set[str] = set()
        self._usage_base: UsageUpdate | None = None
        self._last_merged_usage: UsageUpdate | None = None
        self._deadline: float | None = None
        self._result_chars_remaining = starter._max_result_chars_total
        # Within-turn only: the frozen success result events this turn emitted
        # (citation source authority). Dies with the turn; no store, no cache.
        self._success_results: list[HostedCallResult] = []
        # Persistent immutable snapshots let every filter share each prepared
        # source while later successful rounds extend without recomputing it.
        self._citation_corpus = PreparedCitationCorpus()
        self._citations_requested = _citations_requested(request)
        self._citation_preparation_added = False

    # -- BackendTurn surface -------------------------------------------------

    @property
    def response_id(self) -> str:
        return self._request.response_id

    def cancel(self, reason: str) -> bool:
        self._token.cancel(reason)
        child = self._current_child
        if child is not None:
            with contextlib.suppress(Exception):
                child.cancel(reason)
        task = self._task
        if task is not None and not task.done():
            task.cancel()
        return True

    def wait_closed(self) -> asyncio.Future[None]:
        task = self._task
        if task is None:  # pragma: no cover - launch() precedes exposure
            raise RuntimeError("hosted turn was not launched")
        return asyncio.shield(task)

    # -- lifecycle -----------------------------------------------------------

    def launch(self) -> None:
        self._task = asyncio.create_task(
            self._run(),
            name=f"hosted-agentic:{self._request.response_id}",
        )

    async def _run(self) -> None:
        # One absolute deadline instant on the loop clock covers every child
        # generation round and all hosted work; the immutable scope propagates
        # it (plus this request's cancel token) via a context variable, so
        # concurrent requests can never inherit each other's context.
        deadline: float | None = None
        if self._starter._deadline_s is not None:
            deadline = self._loop.time() + self._starter._deadline_s
        self._deadline = deadline
        scope_token = set_execution_scope(
            HostedExecutionScope(deadline=deadline, cancel=self._token)
        )
        try:
            if deadline is None:
                await self._run_rounds()
                return
            try:
                async with asyncio.timeout_at(deadline):
                    await self._run_rounds()
            except TimeoutError:
                # F11c: the absolute deadline stops work; zero continuation.
                # The current child was cancelled and drained during unwind
                # (_run_child_round) before this terminal is emitted.
                self._token.cancel("deadline_exceeded")
                self._emit_failed(
                    "hosted turn exceeded its absolute deadline",
                    code="deadline_exceeded",
                    status_code=504,
                )
        except HostedTerminalDeliveryError:
            raise  # the facade, not a swallowed success, reports the fault
        except asyncio.CancelledError:
            # F11a/F11b: immediate stop, no continuation, outer TurnCancelled.
            self._emit_cancelled(self._token.reason or "client_cancelled")
        except HostedRuntimeIntegrityError as error:  # F12, authored text
            self._emit_failed(str(error) or INTERNAL_FAILURE_MESSAGE)
        except Exception:  # F12: the owner of last resort, fixed text only
            self._emit_failed(INTERNAL_FAILURE_MESSAGE)
        finally:
            reset_execution_scope(scope_token)

    async def _run_rounds(self) -> None:
        request = self._request
        messages: list[Mapping[str, Any]] = [dict(m) for m in request.messages]
        if not self._hosted_names:
            messages.insert(0, {"role": "system", "content": NO_WEB_PREPARATION})
        loop = AgentLoop(
            self._starter._executor,
            hosted_agent_loop_policy(max_rounds=self._starter._max_tool_rounds),
            loop_id=request.response_id,
        )
        terminal_continuation = False
        hosted_attempted = False
        round_index = 0
        while True:
            self._raise_if_cancelled()
            child = await self._run_child_round(messages, round_index)
            if isinstance(child.terminal, TurnFailed):
                self._emit_terminal(child.terminal)
                return
            if isinstance(child.terminal, TurnCancelled):
                self._emit_cancelled(child.terminal.reason)
                return
            if not isinstance(child.terminal, TurnCompleted):
                raise HostedRuntimeIntegrityError(  # pragma: no cover - guard
                    "child round produced an unknown terminal event"
                )
            if terminal_continuation and not child.saw_text:
                # I10: an empty success after a hosted failure is illegal.
                raise HostedRuntimeIntegrityError(
                    "failure continuation produced no explanatory text"
                )
            calls = _unique_calls(child.tool_calls)
            if not self._hosted_names or not calls:
                self._complete_outer(child.terminal)
                return
            if terminal_continuation:
                self._refuse_post_failure_calls(calls)
                self._complete_outer(child.terminal)
                return
            self._raise_if_cancelled()
            hosted_attempted = True
            results, limit_hit = await self._execute_hosted_round(
                loop,
                calls,
                round_index,
            )
            messages.append(_assistant_tool_call_message(calls))
            for call, result in zip(calls, results, strict=True):
                messages.append(_tool_result_message(call, result))
            if limit_hit or any(not result.ok for result in results):
                # T7: the first error receipt arms exactly one terminal
                # continuation; the trusted preparation quotes nothing from
                # the untrusted error payload.
                terminal_continuation = True
                messages.append(
                    {
                        "role": "system",
                        "content": FAILURE_CONTINUATION_PREPARATION,
                    }
                )
            if (
                self._citations_requested
                and self._success_results
                and not self._citation_preparation_added
            ):
                # The trusted citation preparation quotes nothing from any
                # payload; outside this one message the continuation input
                # stays byte-identical to the unfiltered baseline.
                self._citation_preparation_added = True
                messages.append({"role": "system", "content": CITATION_PREPARATION})
            round_index += 1
            if hosted_attempted and round_index > 2 * self._starter._max_tool_rounds:
                raise HostedRuntimeIntegrityError(  # pragma: no cover - guard
                    "hosted round accounting exceeded its bound"
                )

    def _refuse_post_failure_calls(
        self,
        calls: Sequence[ParsedToolCall],
    ) -> None:
        # I8: the failure continuation may not execute hosted tools.
        for call in calls:
            item = self._emit_hosted_started(call)
            self._emit_hosted_result_and_receipt(
                item,
                call,
                failure_result(
                    call_id=call.call_id,
                    tool_name=call.name,
                    code="continuation_exhausted",
                    message=(
                        "the terminal failure continuation may not execute hosted tools"
                    ),
                ),
            )

    async def _execute_hosted_round(
        self,
        loop: AgentLoop,
        calls: tuple[ParsedToolCall, ...],
        round_index: int,
    ) -> tuple[tuple[ToolExecutionResult, ...], bool]:
        items = {call.call_id: self._emit_hosted_started(call) for call in calls}
        limit_hit = False
        try:
            results = await loop.execute_round(
                calls,
                round_id=f"model-{round_index}",
            )
        except AgentLoopLimitExceeded:
            limit_hit = True
            results = tuple(
                failure_result(
                    call_id=call.call_id,
                    tool_name=call.name,
                    code="tool_round_limit",
                    message="the hosted tool round limit was reached",
                )
                for call in calls
            )
        if len(results) != len(calls):  # pragma: no cover - loop contract
            raise HostedRuntimeIntegrityError(
                "hosted execution returned a mismatched receipt set"
            )
        results = self._charge_result_budget(calls, results)
        for call, result in zip(calls, results, strict=True):
            self._emit_hosted_result_and_receipt(items[call.call_id], call, result)
        return results, limit_hit

    def _charge_result_budget(
        self,
        calls: tuple[ParsedToolCall, ...],
        results: tuple[ToolExecutionResult, ...],
    ) -> tuple[ToolExecutionResult, ...]:
        """Charge the one aggregate result budget in model call order.

        Only the would-overflow payload is dropped: it becomes one typed
        ``result_budget_exceeded`` error receipt (arming the one terminal
        continuation downstream) while every previously proven result stays
        valid and charged.
        """

        charged: list[ToolExecutionResult] = []
        for call, result in zip(calls, results, strict=True):
            outcome = result
            if outcome.ok:
                receipt = self._validated_receipt(call, outcome, "completed")
                payload = self._validated_success_payload(call, outcome, receipt)
                cost = _result_charge(payload)
                if cost > self._result_chars_remaining:
                    outcome = failure_result(
                        call_id=call.call_id,
                        tool_name=call.name,
                        code="result_budget_exceeded",
                        message=(
                            "hosted tool result exceeds the aggregate result "
                            "budget of this turn"
                        ),
                    )
                else:
                    self._result_chars_remaining -= cost
            charged.append(outcome)
        return tuple(charged)

    async def _run_child_round(
        self,
        messages: Sequence[Mapping[str, Any]],
        round_index: int,
    ) -> _ChildRound:
        child_request = replace(self._request, messages=tuple(messages))
        collector = _ChildSink(self, first_round=round_index == 0)
        handle = await self._starter._inner.start(
            child_request,
            collector,
            cancel=self._token,
        )
        self._current_child = handle
        try:
            terminal = await collector.wait_terminal()
            await handle.wait_closed()
        except asyncio.CancelledError:
            # F11a/b/c: actively stop the child backend (a backend may ignore
            # the shared token until its own cancel() is invoked) and observe
            # its closure before the outer terminal can be considered closed.
            with contextlib.suppress(Exception):
                handle.cancel(self._token.reason or "hosted_turn_stopped")
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await asyncio.shield(handle.wait_closed())
            raise
        finally:
            self._current_child = None
        child_usage = collector.last_child_usage
        if child_usage is not None:
            self._usage_base = _add_usage(self._usage_base, child_usage)
        return _ChildRound(
            terminal=terminal,
            tool_calls=collector.tool_calls(),
            saw_text=collector.saw_text,
        )

    # -- outer event emission ------------------------------------------------

    def _forward(self, event: TurnEvent) -> None:
        self._sink.emit(event)

    def _mark_started(self, event: TurnStarted) -> bool:
        with self._lock:
            if self._outer_started:
                return False
            self._outer_started = True
        self._forward(event)
        return True

    def _alloc_index(self) -> int:
        with self._lock:
            index = self._next_index
            self._next_index += 1
            return index

    def _alloc_item_id(self, item_id: str) -> str:
        with self._lock:
            candidate = item_id
            attempt = 1
            while candidate in self._used_item_ids:
                candidate = f"{item_id}-x{attempt}"
                attempt += 1
            self._used_item_ids.add(candidate)
            return candidate

    def _merged_usage(self, child_usage: UsageUpdate) -> UsageUpdate:
        merged = _add_usage(self._usage_base, child_usage)
        self._last_merged_usage = merged
        return merged

    def _emit_hosted_started(self, call: ParsedToolCall) -> _HostedItem:
        index = self._alloc_index()
        item_id = self._alloc_item_id(f"hosted_{call.call_id}")
        opening_action = _call_action(call)
        item = _HostedItem(
            index=index,
            item_id=item_id,
            call_id=call.call_id,
            tool_name=call.name,
        )
        self._forward(
            OutputItemStarted(
                kind=HOSTED_CALL_ITEM_KIND,
                index=index,
                item_id=item_id,
                call_id=call.call_id,
                name=call.name,
                action=opening_action,
            )
        )
        self._forward(
            HostedCallStarted(
                index=index,
                item_id=item_id,
                call_id=call.call_id,
                tool_name=call.name,
                action=opening_action,
            )
        )
        self._forward(
            HostedCallProgress(
                index=index,
                item_id=item_id,
                call_id=call.call_id,
                phase="executing",
            )
        )
        return item

    def _emit_hosted_result_and_receipt(
        self,
        item: _HostedItem,
        call: ParsedToolCall,
        result: ToolExecutionResult,
    ) -> None:
        # F11: a call completing after cancel/disconnect/deadline forwards
        # nothing — no payload, no receipt, no continuation input.
        self._raise_if_cancelled()
        self._raise_if_deadline_expired()
        status = "completed" if result.ok else "failed"
        metadata = result.metadata or {}
        receipt = self._validated_receipt(call, result, status)
        result_event: HostedCallResult | None = None
        if status == "completed":
            payload = self._validated_success_payload(call, result, receipt)
            result_event = HostedCallResult(
                index=item.index,
                item_id=item.item_id,
                call_id=call.call_id,
                tool_name=call.name,
                result=payload,
            )
        elif metadata.get("result") is not None:
            raise HostedRuntimeIntegrityError("hosted failure carried a result payload")
        sealed_action = self._sealed_action(
            call,
            result_event.result if result_event is not None else None,
            status,
        )
        extended_corpus = self._citation_corpus
        if result_event is not None and self._citations_requested:
            extended_corpus = extended_corpus.extend(_citation_sources((result_event,)))
        if result_event is not None:
            self._forward(result_event)
            self._success_results.append(result_event)
            self._citation_corpus = extended_corpus
        self._forward(
            HostedCallCompleted(
                index=item.index,
                item_id=item.item_id,
                call_id=call.call_id,
                tool_name=call.name,
                status=status,
                receipt=receipt,
            )
        )
        self._forward(
            OutputItemCompleted(
                kind=HOSTED_CALL_ITEM_KIND,
                index=item.index,
                item_id=item.item_id,
                call_id=call.call_id,
                name=call.name,
                status=status,
                action=sealed_action,
            )
        )

    @staticmethod
    def _validated_receipt(
        call: ParsedToolCall,
        result: ToolExecutionResult,
        status: str,
    ) -> dict[str, Any]:
        metadata = result.metadata or {}
        raw_receipt = metadata.get("receipt")
        if not isinstance(raw_receipt, Mapping):
            raise HostedRuntimeIntegrityError(
                "hosted execution result carried no typed receipt"
            )
        receipt = dict(raw_receipt)
        scoped = receipt.get("call_id")
        if scoped is not None and scoped != call.call_id:
            receipt["scoped_call_id"] = scoped
        receipt["call_id"] = call.call_id
        # Receipt/event consistency (§3.4): a receipt disagreeing with the
        # events it closes is a server fault, never a terminal success.
        if receipt.get("tool_name") != call.name:
            raise HostedRuntimeIntegrityError(
                "hosted receipt tool_name disagrees with the closing events"
            )
        if receipt.get("status") != status:
            raise HostedRuntimeIntegrityError(
                "hosted receipt status disagrees with the closing events"
            )
        if ("error" in receipt) != (status == "failed"):
            raise HostedRuntimeIntegrityError(
                "hosted receipt error presence disagrees with its status"
            )
        return receipt

    def _validated_success_payload(
        self,
        call: ParsedToolCall,
        result: ToolExecutionResult,
        receipt: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        payload = (result.metadata or {}).get("result")
        if not isinstance(payload, Mapping):
            raise HostedRuntimeIntegrityError(
                "hosted success carried no result payload"
            )
        try:
            validated = validate_result_payload(call.name, payload)
        except (TypeError, ValueError) as error:
            raise HostedRuntimeIntegrityError(
                "hosted success carried an invalid result payload"
            ) from error
        self._verify_result_receipt_identity(validated, receipt)
        return validated

    @staticmethod
    def _verify_result_receipt_identity(
        payload: Mapping[str, Any],
        receipt: Mapping[str, Any],
    ) -> None:
        # Payload/receipt identity law: any disagreement between the public
        # result and the audit receipt is F12, never a terminal success.
        if payload.get("digest") != receipt.get("result_digest"):
            raise HostedRuntimeIntegrityError(
                "hosted result digest disagrees with its receipt"
            )
        if payload.get("kind") == "document":
            if payload.get("url") != receipt.get("final_url"):
                raise HostedRuntimeIntegrityError(
                    "hosted result url disagrees with its receipt final_url"
                )
            if payload.get("media_type") != receipt.get("mime"):
                raise HostedRuntimeIntegrityError(
                    "hosted result media_type disagrees with its receipt mime"
                )

    def _sealed_action(
        self,
        call: ParsedToolCall,
        result: Mapping[str, Any] | None,
        status: str,
    ) -> Mapping[str, Any]:
        """Build the final immutable sealed action (design D-B §2.2).

        The model input supplies the requested query/url; on success the
        proven result identities supply the search sources. A hosted call
        whose arguments carry no usable identity seals the raw argument
        string instead — deterministic, never fabricated semantics.
        """

        kind = ACTION_KIND_FOR_TOOL.get(call.name)
        model_action = _call_action(call)
        if kind == "fetch":
            url = model_action.get("url")
            if not isinstance(url, str) or not url.strip():
                url = call.arguments.strip() or "{}"
            action: dict[str, Any] = {"kind": "fetch", "url": url}
        else:
            query = model_action.get("query")
            if not isinstance(query, str) or not query.strip():
                query = call.arguments.strip() or "{}"
            sources: list[str] = []
            if status == "completed" and result is not None:
                seen: set[str] = set()
                for identity in result_identities(result):
                    if identity not in seen:
                        seen.add(identity)
                        sources.append(identity)
            action = {"kind": "search", "query": query, "sources": sources}
        if kind is not None:
            # The producer validator proves the closed schema and, on
            # success, the sources-subset law; the event layer re-freezes it.
            action = validate_sealed_action(call.name, action, result=result)
        return action

    def _raise_if_deadline_expired(self) -> None:
        if self._deadline is not None and self._loop.time() >= self._deadline:
            # Raised inside the _run timeout_at block: the same F11c 504 path.
            raise TimeoutError("hosted turn exceeded its absolute deadline")

    def _complete_outer(self, child_terminal: TurnCompleted) -> None:
        self._emit_terminal(
            TurnCompleted(
                finish_reason=child_terminal.finish_reason,
                usage=self._last_merged_usage,
                backend_stats=child_terminal.backend_stats,
                stop_sequence=child_terminal.stop_sequence,
            )
        )

    def _emit_failed(
        self,
        error: str,
        *,
        code: str = "internal_error",
        status_code: int = 500,
    ) -> None:
        self._emit_terminal(TurnFailed(error, code=code, status_code=status_code))

    def _emit_cancelled(self, reason: str) -> None:
        with self._lock:
            needs_start = not self._outer_started
        if needs_start:
            # TurnCancelled may not terminate an idle turn; open it honestly.
            with contextlib.suppress(Exception):
                self._mark_started(
                    TurnStarted(
                        response_id=self._request.response_id,
                        model=self._request.runtime.model_id,
                        created_at=int(time.time()),
                    )
                )
        self._emit_terminal(TurnCancelled(reason))

    def _emit_terminal(self, event: TerminalEvent) -> None:
        with self._lock:
            if self._terminal_emitted:
                return
            self._terminal_emitted = True
        try:
            self._forward(event)
        except BaseException as error:
            # Never suppressed into apparent success: the fault escapes the
            # turn task so wait_closed() observably reports it.
            raise HostedTerminalDeliveryError(
                "outer terminal event could not be delivered to the sink"
            ) from error

    def _raise_if_cancelled(self) -> None:
        if self._token.cancelled:
            raise asyncio.CancelledError(self._token.reason or "cancelled")


class _ChildSink:
    """Private per-round sink: child terminals never reach the outer stream."""

    def __init__(self, owner: _HostedAgenticTurn, *, first_round: bool) -> None:
        self._owner = owner
        self._first_round = first_round
        self._lock = threading.Lock()
        self._terminal: asyncio.Future[TerminalEvent] = owner._loop.create_future()
        self._index_map: dict[int, tuple[int, str]] = {}
        self._suppressed_indices: set[int] = set()
        self._tool_calls: list[ParsedToolCall] = []
        self._last_child_usage: UsageUpdate | None = None
        self.saw_text = False
        # The citation filter arms only for a continuation round that follows
        # at least one immutable success result with citations requested; on
        # every other path this sink is byte-identical to the baseline.
        self._citation_armed = bool(
            owner._citations_requested and owner._citation_corpus
        )
        self._citation_corpus = owner._citation_corpus
        self._filters: dict[tuple[int, int], CitationStreamFilter] = {}
        self._item_budgets: dict[int, ItemCitationBudget] = {}

    @property
    def last_child_usage(self) -> UsageUpdate | None:
        return self._last_child_usage

    def tool_calls(self) -> tuple[ParsedToolCall, ...]:
        with self._lock:
            return tuple(self._tool_calls)

    async def wait_terminal(self) -> TerminalEvent:
        return await self._terminal

    def emit(self, event: TurnEvent) -> None:
        try:
            self._emit(event)
        except BaseException as error:
            # A forwarding fault must not strand the driver on a terminal
            # that will never arrive: surface it as the round outcome (F12).
            self._fail_terminal(error)
            raise

    def _emit(self, event: TurnEvent) -> None:
        owner = self._owner
        if isinstance(event, TurnStarted):
            if self._first_round:
                owner._mark_started(event)
        elif isinstance(event, TERMINAL_EVENT_TYPES):
            self._resolve_terminal(event)
        elif isinstance(event, UsageUpdate):
            with self._lock:
                self._last_child_usage = event
            owner._forward(owner._merged_usage(event))
        elif isinstance(event, OutputItemStarted):
            self._emit_item_started(event)
        elif isinstance(event, OutputItemCompleted | ToolDelta | ToolCompleted):
            self._emit_item_scoped(event)
        elif isinstance(
            event,
            ContentPartStarted
            | ContentPartCompleted
            | TextDelta
            | TextCompleted
            | ReasoningDelta
            | ReasoningCompleted,
        ):
            self._emit_content_scoped(event)
        else:
            # ProgressUpdate and any other neutral intermediate: forward as-is.
            owner._forward(event)

    def _emit_item_started(self, event: OutputItemStarted) -> None:
        owner = self._owner
        if self._suppress_kind(event.kind):
            with self._lock:
                self._suppressed_indices.add(event.index)
            return
        outer_index = owner._alloc_index()
        outer_item_id = owner._alloc_item_id(event.item_id)
        with self._lock:
            self._index_map[event.index] = (outer_index, outer_item_id)
        owner._forward(replace(event, index=outer_index, item_id=outer_item_id))

    def _emit_item_scoped(
        self,
        event: OutputItemCompleted | ToolDelta | ToolCompleted,
    ) -> None:
        if self._is_suppressed(event.index):
            if isinstance(event, ToolCompleted):
                with self._lock:
                    self._tool_calls.append(
                        ParsedToolCall(
                            index=event.index,
                            call_id=event.call_id,
                            name=event.name,
                            arguments=event.arguments,
                        )
                    )
            return
        if (
            self._citation_armed
            and isinstance(event, OutputItemCompleted)
            and event.kind == "message"
        ):
            filtered = self._filtered_item_text(event.index)
            if filtered is not None:
                event = replace(event, text=filtered)
        outer_index, outer_item_id = self._mapped(event.index)
        self._owner._forward(replace(event, index=outer_index, item_id=outer_item_id))

    def _emit_content_scoped(
        self,
        event: (
            ContentPartStarted
            | ContentPartCompleted
            | TextDelta
            | TextCompleted
            | ReasoningDelta
            | ReasoningCompleted
        ),
    ) -> None:
        if self._citation_armed and (
            isinstance(event, TextDelta | TextCompleted)
            or (isinstance(event, ContentPartCompleted) and event.kind == "output_text")
        ):
            self._emit_filtered_content(event)
            return
        if isinstance(event, TextDelta | TextCompleted) and (
            event.delta if isinstance(event, TextDelta) else event.text
        ):
            self.saw_text = True
        outer_index, outer_item_id = self._mapped(event.output_index)
        self._owner._forward(
            replace(event, output_index=outer_index, item_id=outer_item_id)
        )

    def _emit_filtered_content(
        self,
        event: TextDelta | TextCompleted | ContentPartCompleted,
    ) -> None:
        """Route message text through the armed causal citation filter.

        Held bytes are the only bytes not yet emitted; markup is only ever
        held or stripped, so the concatenated clean deltas, the rewritten
        TextCompleted/ContentPartCompleted texts and the rewritten message
        OutputItemCompleted text stay exactly equal — the existing turn
        equality checks remain the enforcement of this property.
        """

        outer_index, outer_item_id = self._mapped(event.output_index)
        content_filter = self._filter_for(event.output_index, event.content_index)
        if isinstance(event, TextDelta):
            self._forward_filter_output(
                content_filter.feed(event.delta),
                event,
                outer_index,
                outer_item_id,
            )
            return
        self._forward_filter_output(
            content_filter.finish() if isinstance(event, TextCompleted) else (),
            event,
            outer_index,
            outer_item_id,
        )
        filtered_text = content_filter.filtered_text
        if filtered_text:
            self.saw_text = True
        self._owner._forward(
            replace(
                event,
                output_index=outer_index,
                item_id=outer_item_id,
                text=filtered_text,
            )
        )

    def _forward_filter_output(
        self,
        output: Sequence[str | ProvenCitation],
        event: TextDelta | TextCompleted | ContentPartCompleted,
        outer_index: int,
        outer_item_id: str,
    ) -> None:
        for piece in output:
            if isinstance(piece, str):
                if piece:
                    self.saw_text = True
                self._owner._forward(
                    TextDelta(
                        delta=piece,
                        item_id=outer_item_id,
                        output_index=outer_index,
                        content_index=event.content_index,
                    )
                )
                continue
            self._owner._forward(
                HostedCitation(
                    output_index=outer_index,
                    item_id=outer_item_id,
                    content_index=event.content_index,
                    source_call_id=piece.source_call_id,
                    source_url=piece.source_url,
                    cited_text=piece.cited_text,
                    source_start=piece.source_start,
                    source_end=piece.source_end,
                    output_start=piece.output_start,
                    output_end=piece.output_end,
                )
            )

    def _filter_for(
        self,
        output_index: int,
        content_index: int,
    ) -> CitationStreamFilter:
        key = (output_index, content_index)
        with self._lock:
            content_filter = self._filters.get(key)
            if content_filter is None:
                budget = self._item_budgets.get(output_index)
                if budget is None:
                    budget = ItemCitationBudget()
                    self._item_budgets[output_index] = budget
                content_filter = CitationStreamFilter(
                    self._citation_corpus,
                    budget=budget,
                )
                self._filters[key] = content_filter
        return content_filter

    def _filtered_item_text(self, item_index: int) -> str | None:
        with self._lock:
            parts = sorted(
                (key[1], content_filter)
                for key, content_filter in self._filters.items()
                if key[0] == item_index
            )
        if not parts:
            return None
        return "".join(content_filter.filtered_text for _, content_filter in parts)

    def _suppress_kind(self, kind: str) -> bool:
        # In hosted mode the model's function_call items are consumed by the
        # runtime (they become hosted_call items); without hosted tools the
        # client owns them and they pass through untouched.
        return kind == "function_call" and bool(self._owner._hosted_names)

    def _is_suppressed(self, index: int) -> bool:
        with self._lock:
            return index in self._suppressed_indices

    def _mapped(self, index: int) -> tuple[int, str]:
        with self._lock:
            mapped = self._index_map.get(index)
        if mapped is None:
            raise HostedRuntimeIntegrityError(
                "child event references an item that was never started"
            )
        return mapped

    def _resolve_terminal(self, event: TerminalEvent) -> None:
        def resolve() -> None:
            if not self._terminal.done():
                self._terminal.set_result(event)

        self._owner._loop.call_soon_threadsafe(resolve)

    def _fail_terminal(self, error: BaseException) -> None:
        def resolve() -> None:
            if not self._terminal.done():
                self._terminal.set_exception(error)

        self._owner._loop.call_soon_threadsafe(resolve)


def _unique_calls(calls: Sequence[ParsedToolCall]) -> tuple[ParsedToolCall, ...]:
    """Collapse identical duplicates; a conflicting call_id reuse is F12.

    Silent first-wins deduplication would hide from ``AgentLoop`` claim
    validation a call_id claimed twice with different payloads; the conflict
    fails the outer turn before any hosted execution or receipt instead.
    """

    unique: dict[str, ParsedToolCall] = {}
    for call in calls:
        previous = unique.get(call.call_id)
        if previous is None:
            unique[call.call_id] = call
        elif (previous.index, previous.name, previous.arguments) != (
            call.index,
            call.name,
            call.arguments,
        ):
            raise HostedRuntimeIntegrityError(
                f"tool call_id {call.call_id} was reused with a conflicting payload"
            )
    return tuple(unique.values())


def _result_charge(payload: Any) -> int:
    """The deterministic aggregate-budget cost of one canonical result."""

    if not isinstance(payload, Mapping):
        # Absence of a success payload is F12 at emission; charge nothing.
        return 0
    if payload.get("kind") == "document":
        content = payload.get("content")
        return len(content) if isinstance(content, str) else 0
    results = payload.get("results")
    if results is None:
        return 0
    try:
        return len(canonical_json(results))
    except (TypeError, ValueError):  # pragma: no cover - producer-validated
        return 0


def _citation_sources(
    results: Sequence[HostedCallResult],
) -> tuple[CitationSource, ...]:
    """Quotable proven sources: document content and search snippets."""

    sources: list[CitationSource] = []
    for event in results:
        result = event.result
        if result["kind"] == "document":
            sources.append(
                CitationSource(
                    call_id=event.call_id,
                    url=result["url"],
                    content=result["content"],
                )
            )
            continue
        for entry in result["results"]:
            sources.append(
                CitationSource(
                    call_id=event.call_id,
                    url=entry["url"],
                    content=entry["snippet"],
                )
            )
    return tuple(sources)


def _call_action(call: ParsedToolCall) -> Mapping[str, Any]:
    try:
        parsed = json.loads(call.arguments)
    except (TypeError, ValueError):
        return {"arguments": call.arguments}
    if isinstance(parsed, dict):
        return parsed
    return {"arguments": call.arguments}


def _assistant_tool_call_message(
    calls: Sequence[ParsedToolCall],
) -> Mapping[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call.call_id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in calls
        ],
    }


def _tool_result_message(
    call: ParsedToolCall,
    result: ToolExecutionResult,
) -> Mapping[str, Any]:
    if result.ok:
        content = result.output
    else:
        metadata = result.metadata or {}
        code = str(metadata.get("error_code") or "tool_execution_failed")
        content = json.dumps(
            {
                "error": {"code": code, "message": result.error or "tool failed"},
                "tool_name": call.name,
                "call_id": call.call_id,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    return {
        "role": "tool",
        "tool_call_id": call.call_id,
        "name": call.name,
        "content": content,
    }


def _add_usage(base: UsageUpdate | None, child: UsageUpdate) -> UsageUpdate:
    if base is None:
        return child
    return UsageUpdate(
        input_tokens=base.input_tokens + child.input_tokens,
        output_tokens=base.output_tokens + child.output_tokens,
        total_tokens=base.total_tokens + child.total_tokens,
        cached_input_tokens=base.cached_input_tokens + child.cached_input_tokens,
        cache_write_input_tokens=(
            base.cache_write_input_tokens + child.cache_write_input_tokens
        ),
        reasoning_output_tokens=(
            base.reasoning_output_tokens + child.reasoning_output_tokens
        ),
    )


__all__ = [
    "CITATIONS_METADATA_KEY",
    "FAILURE_CONTINUATION_PREPARATION",
    "INTERNAL_FAILURE_MESSAGE",
    "NO_WEB_PREPARATION",
    "HostedAgenticRuntimeStarter",
    "HostedRuntimeIntegrityError",
    "HostedTerminalDeliveryError",
]
