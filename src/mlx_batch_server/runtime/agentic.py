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
import re
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

from ..tools.agent_loop import (
    AgentLoop,
    AgentLoopLimitExceeded,
    ToolExecutionResult,
    hosted_agent_loop_policy,
)
from ..tools.hosted import (
    ACTION_KIND_FOR_TOOL,
    HostedExecutionScope,
    HostedRoundMode,
    HostedToolCatalog,
    HostedToolExecutor,
    HostedToolPlan,
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
from .hosted_evidence import (
    HostedCallEvidence,
    HostedEvidenceRegistry,
    HostedRoundEvidence,
    RequestState,
)
from .service import FirstWriterCancelToken, RuntimeStartError, RuntimeStartService

if TYPE_CHECKING:
    from .contracts import BackendTurn, GenerationRequest, TurnSink

FAILURE_CONTINUATION_PREPARATION = (
    "A hosted tool call failed. Use the tool error result to tell the user "
    "what failed, then answer as well as you can without the tool output. "
    "Do not fabricate tool results, fetched content, or citations."
)
HOSTED_FAILURE_DISCLOSURE = (
    "The hosted tool failed, so I could not verify the requested external "
    "information. I will continue with that limitation stated explicitly."
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


class HostedEvidenceError(HostedRuntimeIntegrityError):
    """Admission evidence could not be prepared before public delivery."""


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
        evidence_registry: HostedEvidenceRegistry | None = None,
        acceptance_profile: str | None = None,
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
        self._evidence_registry = evidence_registry
        self._acceptance_profile = acceptance_profile
        if (evidence_registry is None) != (acceptance_profile is None):
            raise ValueError(
                "evidence registry and acceptance profile are one capability"
            )

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
        try:
            plan = self._catalog.plan_for(request)
        except ValueError as error:
            raise RuntimeStartError(
                str(error) or "hosted tool plan could not be admitted"
            ) from error
        turn = _HostedAgenticTurn(
            starter=self,
            request=request,
            sink=sink,
            token=token,
            plan=plan,
        )
        turn.launch()
        return turn


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
    started_monotonic_ns: int


class _HostedAgenticTurn:
    """BackendTurn facade owning one outer hosted lifecycle."""

    def __init__(
        self,
        *,
        starter: HostedAgenticRuntimeStarter,
        request: GenerationRequest,
        sink: TurnSink,
        token: FirstWriterCancelToken,
        plan: HostedToolPlan | None,
    ) -> None:
        self._starter = starter
        self._request = request
        self._sink = sink
        self._token = token
        self._plan = plan
        self._hosted_names = frozenset() if plan is None else plan.executable_names
        self._prior_urls = set(_message_urls(request.messages))
        self._hosted_uses = 0
        # Request-global authority. AgentLoop round ids are scheduling detail,
        # never a namespace in which an outward call id may become a new effect.
        self._claimed_hosted_calls: dict[str, tuple[str, str]] = {}
        self._event_sequence = 0
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
        self._evidence = starter._evidence_registry
        if self._evidence is not None and self._plan is not None:
            trace_id = str(
                request.metadata.get(
                    "mlx_batch_server.internal.hosted_trace_id",
                    request.response_id,
                )
            )
            try:
                self._evidence.begin(
                    response_id=request.response_id,
                    trace_id=trace_id,
                    protocol=self._plan.protocol,
                    profile=starter._acceptance_profile,  # type: ignore[arg-type]
                )
                self._evidence.record_consumer_event(
                    request.response_id,
                    "waiting",
                )
            except Exception as error:
                raise HostedEvidenceError("admission evidence failed") from error

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
            HostedExecutionScope(
                deadline=deadline,
                cancel=self._token,
                policy=(
                    None
                    if self._plan is None
                    else self._plan.policy.with_prior_urls(tuple(self._prior_urls))
                ),
            )
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
            self._emit_failed(
                str(error) or INTERNAL_FAILURE_MESSAGE,
                code=(
                    "admission_evidence_failed"
                    if isinstance(error, HostedEvidenceError)
                    else "internal_error"
                ),
            )
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
            mode = (
                HostedRoundMode.FAILURE_CONTINUATION
                if terminal_continuation
                else (
                    HostedRoundMode.ACTION_SELECTION
                    if round_index == 0
                    else HostedRoundMode.SUCCESS_FOLLOWUP
                )
            )
            child = await self._run_child_round(messages, round_index, mode=mode)
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
            calls = _unique_calls(child.tool_calls)
            if not self._hosted_names or not calls:
                # The deterministic disclosure is already a complete,
                # server-authored answer. A successfully completed no-tools
                # continuation may add no prose; that is still a completed
                # outer turn. Actual child failure/cancellation/deadline paths
                # are handled above and remain failures.
                self._complete_outer(child.terminal)
                return
            if terminal_continuation:
                raise HostedRuntimeIntegrityError(
                    "failure continuation emitted a forbidden hosted tool call"
                )
            self._claim_request_calls(calls)
            self._raise_if_cancelled()
            hosted_attempted = True
            current_scope = HostedExecutionScope(
                deadline=self._deadline,
                cancel=self._token,
                policy=(
                    None
                    if self._plan is None
                    else self._plan.policy.with_prior_urls(tuple(self._prior_urls))
                ),
            )
            scope_token = set_execution_scope(current_scope)
            try:
                results, limit_hit = await self._execute_hosted_round(
                    loop,
                    calls,
                    round_index,
                )
            finally:
                reset_execution_scope(scope_token)
            messages.append(_assistant_tool_call_message(calls))
            for call, result in zip(calls, results, strict=True):
                messages.append(_tool_result_message(call, result))
            if limit_hit or any(not result.ok for result in results):
                # T7: the first error receipt arms exactly one terminal
                # continuation; the trusted preparation quotes nothing from
                # the untrusted error payload.
                terminal_continuation = True
                self._emit_failure_disclosure()
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

    async def _execute_hosted_round(
        self,
        loop: AgentLoop,
        calls: tuple[ParsedToolCall, ...],
        round_index: int,
    ) -> tuple[tuple[ToolExecutionResult, ...], bool]:
        if self._plan is None:
            raise HostedRuntimeIntegrityError("hosted execution has no request plan")
        if any(call.name not in self._plan.executable_names for call in calls):
            raise HostedRuntimeIntegrityError(
                "model selected a tool outside its round plan"
            )
        if self._hosted_uses + len(calls) > self._plan.policy.max_uses:
            results = tuple(
                failure_result(
                    call_id=call.call_id,
                    tool_name=call.name,
                    code="tool_round_limit",
                    message="hosted tool maximum uses was reached",
                )
                for call in calls
            )
            items = {call.call_id: self._emit_hosted_started(call) for call in calls}
            for call, result in zip(calls, results, strict=True):
                self._emit_hosted_result_and_receipt(items[call.call_id], call, result)
            return results, True
        self._hosted_uses += len(calls)
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
        *,
        mode: HostedRoundMode,
    ) -> _ChildRound:
        started_ns = time.monotonic_ns()
        if self._plan is None:
            child_request = replace(self._request, messages=tuple(messages))
        else:
            sampling = dict(self._request.sampling)
            if mode is HostedRoundMode.FAILURE_CONTINUATION:
                tools: tuple[Mapping[str, Any], ...] = ()
                sampling["tool_choice"] = "none"
            else:
                tools = self._plan.model_tools
                if mode is HostedRoundMode.SUCCESS_FOLLOWUP:
                    sampling["tool_choice"] = "auto"
            child_request = replace(
                self._request,
                messages=tuple(messages),
                tools=tools,
                sampling=sampling,
            )
        collector = _ChildSink(
            self,
            first_round=round_index == 0,
            failure_continuation=mode is HostedRoundMode.FAILURE_CONTINUATION,
        )
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
        collector.finalize_action_selection()
        if child_usage is not None:
            self._usage_base = _add_usage(self._usage_base, child_usage)
        if self._evidence is not None and self._plan is not None:
            terminal_kind: Literal["completed", "failed", "cancelled"] = (
                "completed"
                if isinstance(terminal, TurnCompleted)
                else "cancelled"
                if isinstance(terminal, TurnCancelled)
                else "failed"
            )
            try:
                self._evidence.record_round(
                    self.response_id,
                    HostedRoundEvidence(
                        round_index=round_index,
                        started_monotonic_ns=started_ns,
                        ended_monotonic_ns=time.monotonic_ns(),
                        terminal_kind=terminal_kind,
                        usage=(
                            None if child_usage is None else _usage_mapping(child_usage)
                        ),
                        call_ids=tuple(call.call_id for call in collector.tool_calls()),
                    ),
                )
            except Exception as error:
                raise HostedEvidenceError("admission evidence failed") from error
        return _ChildRound(
            terminal=terminal,
            tool_calls=collector.tool_calls(),
            saw_text=collector.saw_text,
        )

    def _claim_request_calls(self, calls: Sequence[ParsedToolCall]) -> None:
        for call in calls:
            identity = (call.name, call.arguments)
            previous = self._claimed_hosted_calls.get(call.call_id)
            if previous is not None:
                raise HostedRuntimeIntegrityError(
                    f"tool call_id {call.call_id} was reused in a later round"
                )
            self._claimed_hosted_calls[call.call_id] = identity

    def _emit_failure_disclosure(self) -> None:
        index = self._alloc_index()
        item_id = self._alloc_item_id(f"msg_hosted_failure_{index}")
        text = HOSTED_FAILURE_DISCLOSURE
        self._forward(OutputItemStarted(index=index, item_id=item_id, kind="message"))
        self._forward(
            ContentPartStarted(
                output_index=index,
                item_id=item_id,
                content_index=0,
                kind="output_text",
            )
        )
        self._forward(
            TextDelta(
                delta=text,
                output_index=index,
                item_id=item_id,
                content_index=0,
            )
        )
        self._forward(
            TextCompleted(
                text=text,
                output_index=index,
                item_id=item_id,
                content_index=0,
            )
        )
        self._forward(
            ContentPartCompleted(
                output_index=index,
                item_id=item_id,
                content_index=0,
                kind="output_text",
                text=text,
            )
        )
        self._forward(
            OutputItemCompleted(
                index=index,
                item_id=item_id,
                kind="message",
                text=text,
            )
        )

    # -- outer event emission ------------------------------------------------

    def _forward(self, event: TurnEvent) -> None:
        self._sink.emit(event)
        self._event_sequence += 1

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
        opening_action = _opening_action(call, _call_action(call))
        item = _HostedItem(
            index=index,
            item_id=item_id,
            call_id=call.call_id,
            tool_name=call.name,
            started_monotonic_ns=time.monotonic_ns(),
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
        completed_event = HostedCallCompleted(
            index=item.index,
            item_id=item.item_id,
            call_id=call.call_id,
            tool_name=call.name,
            status=status,
            receipt=receipt,
        )
        item_event = OutputItemCompleted(
            kind=HOSTED_CALL_ITEM_KIND,
            index=item.index,
            item_id=item.item_id,
            call_id=call.call_id,
            name=call.name,
            status=status,
            action=sealed_action,
        )
        evidence_token: str | None = None
        if self._evidence is not None:
            transport_receipt = receipt.get("transport_receipt")
            requested_url = (
                transport_receipt.get("requested_url")
                if isinstance(transport_receipt, Mapping)
                else None
            )
            error = receipt.get("error")
            try:
                evidence_token = self._evidence.prepare_call(
                    self.response_id,
                    HostedCallEvidence(
                        call_id=call.call_id,
                        tool_name=call.name,
                        action_kind=str(sealed_action["kind"]),
                        status=status,  # type: ignore[arg-type]
                        error_code=(
                            str(error.get("code"))
                            if isinstance(error, Mapping)
                            else None
                        ),
                        requested_url=(
                            str(requested_url)
                            if isinstance(requested_url, str)
                            else None
                        ),
                        final_url=(
                            str(receipt["final_url"])
                            if isinstance(receipt.get("final_url"), str)
                            else None
                        ),
                        result_digest=(
                            str(receipt["result_digest"])
                            if isinstance(receipt.get("result_digest"), str)
                            else None
                        ),
                        mime=(
                            str(receipt["mime"])
                            if isinstance(receipt.get("mime"), str)
                            else None
                        ),
                        started_monotonic_ns=item.started_monotonic_ns,
                        ended_monotonic_ns=time.monotonic_ns(),
                        delivery_state="prepared",
                        first_event_sequence=self._event_sequence,
                        last_event_sequence=(
                            self._event_sequence
                            + (2 if result_event is not None else 1)
                        ),
                        transport_receipt=(
                            transport_receipt
                            if isinstance(transport_receipt, Mapping)
                            else None
                        ),
                    ),
                )
            except Exception as evidence_error:
                raise HostedEvidenceError(
                    "admission evidence failed"
                ) from evidence_error
        try:
            if result_event is not None:
                self._forward(result_event)
                self._success_results.append(result_event)
                self._citation_corpus = extended_corpus
                self._prior_urls.update(result_identities(result_event.result))
            self._forward(completed_event)
            self._forward(item_event)
        except BaseException:
            if evidence_token is not None and self._evidence is not None:
                self._evidence.mark_call_delivery_failed(
                    self.response_id, evidence_token
                )
            raise
        if evidence_token is not None and self._evidence is not None:
            self._evidence.mark_call_delivered(self.response_id, evidence_token)

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
        if scoped != call.call_id:
            raise HostedRuntimeIntegrityError(
                "hosted receipt call_id disagrees with the admitted call identity"
            )
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
        if kind in {"fetch", "open_page"}:
            url = model_action.get("url")
            if not isinstance(url, str) or not url.strip():
                url = call.arguments.strip() or "{}"
            action: dict[str, Any] = {"kind": kind, "url": url}
        elif kind == "find_in_page":
            url = model_action.get("url")
            pattern = model_action.get("pattern")
            if not isinstance(url, str) or not url.strip():
                url = call.arguments.strip() or "{}"
            if not isinstance(pattern, str) or not pattern:
                pattern = call.arguments.strip() or "{}"
            action = {"kind": "find_in_page", "url": url, "pattern": pattern}
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
        state: RequestState = (
            "completed"
            if isinstance(event, TurnCompleted)
            else "cancelled"
            if isinstance(event, TurnCancelled)
            else "failed"
        )
        usage = event.usage if isinstance(event, TurnCompleted) else None
        evidence_token: str | None = None
        if self._evidence is not None:
            try:
                evidence_token = self._evidence.prepare_terminal(
                    self.response_id,
                    state=state,
                    cancel_reason=(
                        event.reason if isinstance(event, TurnCancelled) else None
                    ),
                    terminal_usage=(
                        None if usage is None else dict(_usage_mapping(usage))
                    ),
                )
            except Exception as evidence_error:
                raise HostedEvidenceError(
                    "admission evidence failed"
                ) from evidence_error
        with self._lock:
            if self._terminal_emitted:
                return
            self._terminal_emitted = True
        try:
            self._forward(event)
        except BaseException as error:
            if evidence_token is not None and self._evidence is not None:
                self._evidence.mark_terminal(
                    self.response_id,
                    evidence_token,
                    delivered=False,
                )
            # Never suppressed into apparent success: the fault escapes the
            # turn task so wait_closed() observably reports it.
            raise HostedTerminalDeliveryError(
                "outer terminal event could not be delivered to the sink"
            ) from error
        if evidence_token is not None and self._evidence is not None:
            self._evidence.mark_terminal(
                self.response_id,
                evidence_token,
                delivered=True,
            )

    def _raise_if_cancelled(self) -> None:
        if self._token.cancelled:
            raise asyncio.CancelledError(self._token.reason or "cancelled")


class _ChildSink:
    """Private per-round sink: child terminals never reach the outer stream."""

    def __init__(
        self,
        owner: _HostedAgenticTurn,
        *,
        first_round: bool,
        failure_continuation: bool,
    ) -> None:
        self._owner = owner
        self._first_round = first_round
        self._lock = threading.Lock()
        self._terminal: asyncio.Future[TerminalEvent] = owner._loop.create_future()
        self._index_map: dict[int, tuple[int, str]] = {}
        self._suppressed_indices: set[int] = set()
        self._tool_calls: list[ParsedToolCall] = []
        self._last_child_usage: UsageUpdate | None = None
        self.saw_text = False
        self._quarantine = bool(owner._hosted_names)
        self._quarantined_events: list[TurnEvent] = []
        self._failure_continuation = failure_continuation
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

    def finalize_action_selection(self) -> None:
        """Publish model prose iff this round selected no hosted action."""
        with self._lock:
            events = tuple(self._quarantined_events)
            self._quarantined_events.clear()
            has_hosted_call = bool(self._tool_calls)
        if not has_hosted_call:
            if self._failure_continuation:
                text = " ".join(
                    event.text for event in events if isinstance(event, TextCompleted)
                ).casefold()
                forbidden = (
                    "tool succeeded",
                    "search succeeded",
                    "fetch succeeded",
                    "successfully searched",
                    "successfully fetched",
                )
                if any(claim in text for claim in forbidden):
                    # The runtime-authored disclosure is already a non-empty,
                    # honest assistant answer. This bounded guard suppresses
                    # only the listed literal contradictions; it is not a
                    # general semantic-honesty classifier.
                    return
            for event in events:
                self._owner._forward(event)
            return
        for event in events:
            if isinstance(event, UsageUpdate):
                self._owner._forward(event)
        # Nothing in the quarantine reached the outer stream. Reclaim its
        # provisional identities before hosted items are allocated, keeping
        # outward indices contiguous and preventing snapshot ghosts.
        with self._lock:
            mapped = tuple(self._index_map.values())
            self._index_map.clear()
        with self._owner._lock:
            for _, item_id in mapped:
                self._owner._used_item_ids.discard(item_id)
            self._owner._next_index -= len(mapped)

    def _publish(self, event: TurnEvent) -> None:
        if self._quarantine:
            with self._lock:
                self._quarantined_events.append(event)
            return
        self._owner._forward(event)

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
            self._publish(owner._merged_usage(event))
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
        self._publish(replace(event, index=outer_index, item_id=outer_item_id))

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
        self._publish(replace(event, index=outer_index, item_id=outer_item_id))

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
        self._publish(replace(event, output_index=outer_index, item_id=outer_item_id))

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
        self._publish(
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
                self._publish(
                    TextDelta(
                        delta=piece,
                        item_id=outer_item_id,
                        output_index=outer_index,
                        content_index=event.content_index,
                    )
                )
                continue
            self._publish(
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
            content = (
                result["extracted_text"]
                if result.get("representation") == "base64"
                else result["content"]
            )
            sources.append(
                CitationSource(
                    call_id=event.call_id,
                    url=result["url"],
                    content=content,
                )
            )
            continue
        if result["kind"] == "find_matches":
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


def _opening_action(
    call: ParsedToolCall,
    model_action: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Project a total opening action without claiming successful validation."""

    fallback = call.arguments.strip() or "{}"
    if call.name == "web_search":
        query = model_action.get("query")
        return {
            "query": query if isinstance(query, str) and query.strip() else fallback
        }
    if call.name in {"web_fetch", "open_page"}:
        url = model_action.get("url")
        return {"url": url if isinstance(url, str) and url.strip() else fallback}
    if call.name == "find_in_page":
        url = model_action.get("url")
        pattern = model_action.get("pattern")
        return {
            "url": url if isinstance(url, str) and url.strip() else fallback,
            "pattern": (pattern if isinstance(pattern, str) and pattern else fallback),
        }
    raise HostedRuntimeIntegrityError("hosted opening action has an unknown tool")


_URL_PATTERN = re.compile(r"https?://[^\s<>\"']+")


def _message_urls(messages: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    """Extract only literal user-supplied URL identities from request content."""

    urls: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, str):
            urls.extend(match.rstrip(".,);]}") for match in _URL_PATTERN.findall(value))
        elif isinstance(value, Mapping):
            for item in value.values():
                visit(item)
        elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
            for item in value:
                visit(item)

    for message in messages:
        if str(message.get("role", "")).lower() == "user":
            visit(message.get("content"))
    return tuple(dict.fromkeys(urls))


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


def _usage_mapping(usage: UsageUpdate) -> MappingProxyType:
    return MappingProxyType(
        {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
            "cached_input_tokens": usage.cached_input_tokens,
            "cache_write_input_tokens": usage.cache_write_input_tokens,
            "reasoning_output_tokens": usage.reasoning_output_tokens,
        }
    )


__all__ = [
    "CITATIONS_METADATA_KEY",
    "FAILURE_CONTINUATION_PREPARATION",
    "HOSTED_FAILURE_DISCLOSURE",
    "INTERNAL_FAILURE_MESSAGE",
    "NO_WEB_PREPARATION",
    "HostedAgenticRuntimeStarter",
    "HostedRuntimeIntegrityError",
    "HostedTerminalDeliveryError",
]
