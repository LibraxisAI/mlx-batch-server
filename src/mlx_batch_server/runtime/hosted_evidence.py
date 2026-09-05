"""Bounded protocol-neutral admission evidence for hosted tool turns.

This store is deliberately not a response store: its closed dataclasses have
no place for prompts, bodies, headers, provider payloads, IP addresses or
credentials. Candidates become acceptance evidence only after sink delivery.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Literal


DeliveryState = Literal["prepared", "delivered", "delivery_failed"]
RequestState = Literal["in_progress", "completed", "failed", "cancelled"]


@dataclass(frozen=True, slots=True)
class HostedConsumerEvent:
    state: Literal[
        "queued",
        "waiting",
        "tool_failed_but_continuing",
        "terminal",
    ]
    monotonic_ns: int
    call_id: str | None = None
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class HostedRoundEvidence:
    round_index: int
    started_monotonic_ns: int
    ended_monotonic_ns: int
    terminal_kind: Literal["completed", "failed", "cancelled"]
    usage: MappingProxyType | None
    call_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HostedCallEvidence:
    call_id: str
    tool_name: str
    action_kind: str
    status: Literal["completed", "failed"]
    error_code: str | None
    requested_url: str | None
    final_url: str | None
    result_digest: str | None
    mime: str | None
    started_monotonic_ns: int
    ended_monotonic_ns: int
    delivery_state: DeliveryState
    first_event_sequence: int
    last_event_sequence: int


@dataclass(frozen=True, slots=True)
class HostedRequestEvidence:
    response_id: str
    trace_id: str
    protocol: Literal["openai_responses", "anthropic_messages"]
    profile: Literal["provider-present", "provider-absent", "deadline-short"]
    state: RequestState
    terminal_delivery_state: DeliveryState | None
    cancel_reason: str | None
    consumer_events: tuple[HostedConsumerEvent, ...]
    rounds: tuple[HostedRoundEvidence, ...]
    calls: tuple[HostedCallEvidence, ...]
    terminal_usage: MappingProxyType | None
    created_monotonic_ns: int
    sealed_monotonic_ns: int | None


@dataclass(slots=True)
class _Record:
    snapshot: HostedRequestEvidence
    pending_calls: dict[str, _PreparedCall] = field(default_factory=dict)
    pending_terminal: _PreparedTerminal | None = None


@dataclass(frozen=True, slots=True)
class _PreparedCall:
    call_id: str
    delivered: HostedRequestEvidence
    delivery_failed: HostedRequestEvidence


@dataclass(frozen=True, slots=True)
class _PreparedTerminal:
    token: str
    delivered: HostedRequestEvidence
    delivery_failed: HostedRequestEvidence


class HostedEvidenceRegistry:
    """Process-local TTL/capacity store with idempotent delivery marks."""

    def __init__(self, *, capacity: int = 256, ttl_s: float = 900.0) -> None:
        if capacity < 1 or ttl_s <= 0:
            raise ValueError("hosted evidence capacity and ttl must be positive")
        self._capacity = capacity
        self._ttl_ns = int(ttl_s * 1_000_000_000)
        self._records: dict[str, _Record] = {}
        self._trace_ids: dict[str, str] = {}
        self._lock = threading.Lock()

    def begin(
        self,
        *,
        response_id: str,
        trace_id: str,
        protocol: Literal["openai_responses", "anthropic_messages"],
        profile: Literal["provider-present", "provider-absent", "deadline-short"],
    ) -> None:
        now = time.monotonic_ns()
        with self._lock:
            self._evict(now)
            if response_id in self._records or trace_id in self._trace_ids:
                raise ValueError("hosted evidence identities must be unique")
            if len(self._records) >= self._capacity:
                raise RuntimeError("hosted evidence capacity is exhausted")
            snapshot = HostedRequestEvidence(
                response_id=response_id,
                trace_id=trace_id,
                protocol=protocol,
                profile=profile,
                state="in_progress",
                terminal_delivery_state=None,
                cancel_reason=None,
                consumer_events=(HostedConsumerEvent("queued", now),),
                rounds=(),
                calls=(),
                terminal_usage=None,
                created_monotonic_ns=now,
                sealed_monotonic_ns=None,
            )
            self._records[response_id] = _Record(snapshot)
            self._trace_ids[trace_id] = response_id

    def record_consumer_event(
        self,
        response_id: str,
        state: Literal["waiting", "tool_failed_but_continuing"],
        *,
        call_id: str | None = None,
        error_code: str | None = None,
    ) -> None:
        with self._lock:
            record = self._open(response_id)
            event = HostedConsumerEvent(
                state=state,
                monotonic_ns=time.monotonic_ns(),
                call_id=call_id,
                error_code=error_code,
            )
            record.snapshot = replace(
                record.snapshot,
                consumer_events=(*record.snapshot.consumer_events, event),
            )

    def prepare_call(self, response_id: str, evidence: HostedCallEvidence) -> str:
        if evidence.delivery_state != "prepared":
            raise ValueError("call evidence must begin prepared")
        token = uuid.uuid4().hex
        with self._lock:
            record = self._open(response_id)
            if evidence.call_id in {call.call_id for call in record.snapshot.calls} or any(
                item.call_id == evidence.call_id for item in record.pending_calls.values()
            ):
                raise ValueError("hosted call evidence id is duplicated")
            delivered = replace(evidence, delivery_state="delivered")
            delivery_failed = replace(evidence, delivery_state="delivery_failed")
            failure_event = (
                HostedConsumerEvent(
                    "tool_failed_but_continuing",
                    time.monotonic_ns(),
                    call_id=evidence.call_id,
                    error_code=evidence.error_code,
                ),
            ) if evidence.status == "failed" else ()
            record.pending_calls[token] = _PreparedCall(
                call_id=evidence.call_id,
                delivered=replace(
                    record.snapshot,
                    calls=(*record.snapshot.calls, delivered),
                    consumer_events=(*record.snapshot.consumer_events, *failure_event),
                ),
                delivery_failed=replace(
                    record.snapshot,
                    calls=(*record.snapshot.calls, delivery_failed),
                ),
            )
        return token

    def mark_call_delivered(self, response_id: str, token: str) -> None:
        self._mark_call(response_id, token, "delivered")

    def mark_call_delivery_failed(self, response_id: str, token: str) -> None:
        self._mark_call(response_id, token, "delivery_failed")

    def _mark_call(self, response_id: str, token: str, state: DeliveryState) -> None:
        with self._lock:
            record = self._records.get(response_id)
            if record is None:
                return
            prepared = record.pending_calls.pop(token, None)
            if prepared is None:
                return
            record.snapshot = (
                prepared.delivered
                if state == "delivered"
                else prepared.delivery_failed
            )

    def record_round(self, response_id: str, evidence: HostedRoundEvidence) -> None:
        with self._lock:
            record = self._open(response_id)
            if evidence.round_index != len(record.snapshot.rounds):
                raise ValueError("hosted round indices must be contiguous")
            if evidence.ended_monotonic_ns < evidence.started_monotonic_ns:
                raise ValueError("hosted round time is reversed")
            record.snapshot = replace(
                record.snapshot,
                rounds=(*record.snapshot.rounds, evidence),
            )

    def prepare_terminal(
        self,
        response_id: str,
        *,
        state: RequestState,
        cancel_reason: str | None = None,
        terminal_usage: dict[str, int] | None = None,
    ) -> str:
        token = uuid.uuid4().hex
        with self._lock:
            record = self._open(response_id)
            if record.pending_terminal is not None:
                raise ValueError("terminal evidence was already prepared")
            now = time.monotonic_ns()
            common = {
                "cancel_reason": cancel_reason,
                "terminal_usage": (
                    None
                    if terminal_usage is None
                    else MappingProxyType(dict(terminal_usage))
                ),
                "consumer_events": (
                    *record.snapshot.consumer_events,
                    HostedConsumerEvent("terminal", now),
                ),
                "sealed_monotonic_ns": now,
            }
            record.pending_terminal = _PreparedTerminal(
                token=token,
                delivered=replace(
                    record.snapshot,
                    state=state,
                    terminal_delivery_state="delivered",
                    **common,
                ),
                delivery_failed=replace(
                    record.snapshot,
                    state="failed",
                    terminal_delivery_state="delivery_failed",
                    **common,
                ),
            )
        return token

    def mark_terminal(
        self,
        response_id: str,
        token: str,
        *,
        delivered: bool,
    ) -> None:
        with self._lock:
            record = self._records.get(response_id)
            if (
                record is None
                or record.pending_terminal is None
                or token != record.pending_terminal.token
            ):
                return
            prepared = record.pending_terminal
            record.pending_terminal = None
            record.snapshot = (
                prepared.delivered if delivered else prepared.delivery_failed
            )

    def by_response(self, response_id: str) -> HostedRequestEvidence | None:
        with self._lock:
            record = self._records.get(response_id)
            return None if record is None else record.snapshot

    def by_trace(self, trace_id: str) -> HostedRequestEvidence | None:
        with self._lock:
            response_id = self._trace_ids.get(trace_id)
            record = None if response_id is None else self._records.get(response_id)
            return None if record is None else record.snapshot

    def _open(self, response_id: str) -> _Record:
        record = self._records.get(response_id)
        if record is None or record.snapshot.state != "in_progress":
            raise ValueError("hosted evidence request is not open")
        return record

    def _evict(self, now: int) -> None:
        expired = [
            response_id
            for response_id, record in self._records.items()
            if record.snapshot.sealed_monotonic_ns is not None
            and now - record.snapshot.sealed_monotonic_ns >= self._ttl_ns
        ]
        for response_id in expired:
            trace_id = self._records[response_id].snapshot.trace_id
            del self._records[response_id]
            self._trace_ids.pop(trace_id, None)
        if len(self._records) < self._capacity:
            return
        sealed = sorted(
            (
                (record.snapshot.sealed_monotonic_ns, response_id)
                for response_id, record in self._records.items()
                if record.snapshot.sealed_monotonic_ns is not None
            ),
            key=lambda item: item[0],
        )
        if sealed:
            _, response_id = sealed[0]
            trace_id = self._records[response_id].snapshot.trace_id
            del self._records[response_id]
            self._trace_ids.pop(trace_id, None)


__all__ = [
    "HostedCallEvidence",
    "HostedConsumerEvent",
    "HostedEvidenceRegistry",
    "HostedRequestEvidence",
    "HostedRoundEvidence",
]
