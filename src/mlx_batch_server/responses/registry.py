"""Bounded response lifecycle, lineage, ownership, and cancel intent.

Adapted from MTPLX
``mtplx/server/protocols/responses/store.py@6d0ddf0575faa9acf77e63c57e48ea1602a7e4ab``
(Apache-2.0). Modified by LibraxisAI for owner isolation, reason-bearing
cancellation, and target-owned terminal waits.

Only JSON-compatible protocol data crosses the storage boundary. Generation
objects remain outside the registry; the one exception is an opaque in-process
cancel callback bound to an in-flight response.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

DEFAULT_MAX_ENTRIES = 256
DEFAULT_MAX_IN_FLIGHT = 64
DEFAULT_MAX_BYTES = 32 * 1024 * 1024
DEFAULT_MAX_ENTRY_BYTES = 4 * 1024 * 1024
DEFAULT_IDLE_TTL_S = 3600.0
DEFAULT_IN_FLIGHT_TTL_S = 900.0
DEFAULT_TOMBSTONE_ENTRIES = 512


class ResponseRegistryError(LookupError):
    """Structured lifecycle error suitable for a Responses wire adapter."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        status_code: int,
        param: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = int(status_code)
        self.param = param

    def payload(self) -> dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": "invalid_request_error",
                "param": self.param,
                "code": self.code,
            }
        }


class CancelDeliveryRejected(RuntimeError):
    """The bound cancel callback explicitly rejected a retryable delivery."""


# Keep the donor's error name available for the serial integration cut.
ResponseStoreError = ResponseRegistryError


@dataclass(slots=True)
class _StoredResponse:
    owner_id: str
    envelope: dict[str, Any]
    materialized_messages: list[dict[str, Any]]
    committed_at: float
    last_access_at: float
    size_bytes: int


@dataclass(slots=True)
class _InFlightResponse:
    response_id: str
    owner_id: str
    store: bool
    materialized_messages: list[dict[str, Any]]
    cancel: Callable[[str], Any] | None
    started_at: float
    size_bytes: int
    cancel_requested: bool = False
    cancel_reason: str | None = None
    cancel_delivered: bool = False
    cancel_delivery_id: str | None = None
    deleted: bool = False
    done: threading.Event = field(default_factory=threading.Event)
    terminal_envelope: dict[str, Any] | None = None


@dataclass(slots=True)
class _Tombstone:
    owner_id: str
    reason: str
    created_at: float
    cancel_pending: bool = False
    cancel_reason: str | None = None
    cancel_delivery_id: str | None = None
    terminal_digest: str | None = None


@dataclass(frozen=True, slots=True)
class _CancelDelivery:
    response_id: str
    owner_id: str
    cancel: Callable[[str], Any]
    reason: str
    delivery_id: str
    state: _InFlightResponse | _Tombstone


def _json_clone(value: Any) -> Any:
    """Copy and validate the JSON-only protocol-state boundary."""

    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ResponseRegistryError(
            "response registry state must be JSON-compatible",
            code="response_invalid_state",
            status_code=400,
        ) from exc


def _assistant_output_messages(envelope: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Materialize assistant text output for the next response in a chain."""

    output = envelope.get("output")
    if not isinstance(output, list):
        return []

    messages: list[dict[str, Any]] = []
    for item in output:
        if not isinstance(item, Mapping) or item.get("type") != "message":
            continue
        content = item.get("content")
        if isinstance(content, str):
            parts = [{"type": "input_text", "text": content}] if content else []
        elif isinstance(content, list):
            parts = [
                {"type": "input_text", "text": part["text"]}
                for part in content
                if isinstance(part, Mapping)
                and part.get("type") in {"text", "input_text", "output_text"}
                and isinstance(part.get("text"), str)
                and part["text"]
            ]
        else:
            parts = []
        if parts:
            messages.append({"role": "assistant", "content": parts})
    return messages


def _json_size(*values: Any) -> int:
    try:
        return sum(
            len(
                json.dumps(
                    value,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            )
            for value in values
        )
    except (TypeError, ValueError) as exc:
        raise ResponseRegistryError(
            "response registry state must be JSON-compatible",
            code="response_invalid_state",
            status_code=400,
        ) from exc


def _json_digest(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ResponseRegistryError(
            "response registry state must be JSON-compatible",
            code="response_invalid_state",
            status_code=400,
        ) from exc
    return sha256(encoded).hexdigest()


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _positive_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _owner_id(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ResponseRegistryError(
            "owner_id must be a non-empty string",
            code="invalid_owner_id",
            status_code=400,
            param="owner_id",
        )
    return value.strip()


def _cancel_reason(value: str) -> str:
    if not isinstance(value, str):
        raise ResponseRegistryError(
            "cancel reason must be a string",
            code="invalid_cancel_reason",
            status_code=400,
            param="reason",
        )
    reason = value.strip() or "client_cancelled"
    if len(reason) > 512:
        raise ResponseRegistryError(
            "cancel reason exceeds 512 characters",
            code="invalid_cancel_reason",
            status_code=400,
            param="reason",
        )
    return reason


class ResponseRegistry:
    """Thread-safe, bounded registry for local Responses protocol state."""

    def __init__(
        self,
        *,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        max_in_flight: int = DEFAULT_MAX_IN_FLIGHT,
        max_bytes: int = DEFAULT_MAX_BYTES,
        max_entry_bytes: int = DEFAULT_MAX_ENTRY_BYTES,
        idle_ttl_s: float = DEFAULT_IDLE_TTL_S,
        in_flight_ttl_s: float = DEFAULT_IN_FLIGHT_TTL_S,
        max_tombstones: int = DEFAULT_TOMBSTONE_ENTRIES,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if (
            min(
                max_entries,
                max_in_flight,
                max_bytes,
                max_entry_bytes,
                max_tombstones,
            )
            <= 0
        ):
            raise ValueError("response registry bounds must be positive")
        if idle_ttl_s <= 0 or in_flight_ttl_s <= 0:
            raise ValueError("response registry TTLs must be positive")

        self.max_entries = int(max_entries)
        self.max_in_flight = int(max_in_flight)
        self.max_bytes = int(max_bytes)
        self.max_entry_bytes = int(max_entry_bytes)
        self.idle_ttl_s = float(idle_ttl_s)
        self.in_flight_ttl_s = float(in_flight_ttl_s)
        self.max_tombstones = int(max_tombstones)
        self._clock = clock
        self._lock = threading.RLock()
        self._stored: OrderedDict[str, _StoredResponse] = OrderedDict()
        self._in_flight: dict[str, _InFlightResponse] = {}
        self._tombstones: OrderedDict[str, _Tombstone] = OrderedDict()
        self._bytes = 0
        self._in_flight_bytes = 0
        self._shutting_down = False
        self._counters = {
            "committed_total": 0,
            "deleted_total": 0,
            "evicted_total": 0,
            "expired_total": 0,
            "cancel_requests_total": 0,
            "cancel_delivery_failures_total": 0,
            "cancel_settled_total": 0,
            "timed_out_total": 0,
        }

    @classmethod
    def from_env(cls) -> ResponseRegistry:
        prefix = "MLX_BATCH_RESPONSE_REGISTRY_"
        return cls(
            max_entries=_positive_int_env(f"{prefix}MAX_ENTRIES", DEFAULT_MAX_ENTRIES),
            max_in_flight=_positive_int_env(
                f"{prefix}MAX_IN_FLIGHT", DEFAULT_MAX_IN_FLIGHT
            ),
            max_bytes=_positive_int_env(f"{prefix}MAX_BYTES", DEFAULT_MAX_BYTES),
            max_entry_bytes=_positive_int_env(
                f"{prefix}MAX_ENTRY_BYTES", DEFAULT_MAX_ENTRY_BYTES
            ),
            idle_ttl_s=_positive_float_env(f"{prefix}IDLE_TTL_S", DEFAULT_IDLE_TTL_S),
            in_flight_ttl_s=_positive_float_env(
                f"{prefix}IN_FLIGHT_TTL_S", DEFAULT_IN_FLIGHT_TTL_S
            ),
            max_tombstones=_positive_int_env(
                f"{prefix}MAX_TOMBSTONES", DEFAULT_TOMBSTONE_ENTRIES
            ),
        )

    def allocate_id(self, preferred: str | None = None) -> str:
        """Return a collision-safe response id, retaining a safe unused hint."""

        self._prune()
        with self._lock:
            self._require_accepting_locked()
            if preferred and not self._known_locked(preferred):
                return preferred
            while True:
                candidate = f"resp_{uuid.uuid4().hex}"
                if not self._known_locked(candidate):
                    return candidate

    def begin(
        self,
        response_id: str,
        *,
        owner_id: str,
        store: bool,
        materialized_messages: Sequence[Mapping[str, Any]],
        cancel: Callable[[str], Any] | None = None,
    ) -> None:
        owner = _owner_id(owner_id)
        if cancel is not None and not callable(cancel):
            raise TypeError("cancel must be callable")
        messages = _json_clone(list(materialized_messages))
        size_bytes = _json_size(messages)
        now = self._clock()
        self._prune(now)

        with self._lock:
            self._require_accepting_locked()
            if self._known_locked(response_id):
                raise ResponseRegistryError(
                    f"response id {response_id!r} already exists",
                    code="response_id_conflict",
                    status_code=409,
                )
            if size_bytes > self.max_entry_bytes or size_bytes > self.max_bytes:
                raise ResponseRegistryError(
                    "response input exceeds the local lineage byte limit",
                    code="response_too_large",
                    status_code=413,
                    param="input",
                )
            if len(self._in_flight) >= self.max_in_flight:
                raise ResponseRegistryError(
                    "response registry has reached its in-flight limit",
                    code="response_store_capacity",
                    status_code=429,
                )
            if self._in_flight_bytes + size_bytes > self.max_bytes:
                raise ResponseRegistryError(
                    "response registry has reached its in-flight byte limit",
                    code="response_store_capacity",
                    status_code=429,
                )
            self._in_flight[response_id] = _InFlightResponse(
                response_id=response_id,
                owner_id=owner,
                store=bool(store),
                materialized_messages=messages,
                cancel=cancel,
                started_at=now,
                size_bytes=size_bytes,
            )
            self._in_flight_bytes += size_bytes

    def bind_cancel(
        self,
        response_id: str,
        cancel: Callable[[str], Any],
        *,
        owner_id: str,
    ) -> bool:
        """Bind the real worker handle and replay one earlier cancel intent."""

        owner = _owner_id(owner_id)
        if not callable(cancel):
            raise TypeError("cancel must be callable")
        self._prune()
        delivery: _CancelDelivery | None = None

        with self._lock:
            inflight = self._owned_in_flight_locked(response_id, owner)
            if inflight is not None:
                if (
                    inflight.cancel is not None
                    and inflight.cancel_delivery_id is not None
                    and inflight.cancel != cancel
                ):
                    raise RuntimeError(
                        f"response {response_id!r} has another cancel delivery "
                        "in progress"
                    )
                if inflight.cancel_delivered:
                    return True
                inflight.cancel = cancel
                delivery = self._reserve_in_flight_cancel_locked(inflight)
            else:
                tombstone = self._owned_tombstone_locked(response_id, owner)
                if tombstone is None or not tombstone.cancel_pending:
                    return False
                delivery = self._reserve_tombstone_cancel_locked(
                    response_id,
                    tombstone,
                    cancel,
                )

        if delivery is not None:
            self._deliver_cancel(delivery)
        return True

    def commit(
        self,
        response_id: str,
        envelope: Mapping[str, Any],
        *,
        owner_id: str,
        materialized_messages: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        owner = _owner_id(owner_id)
        terminal = _json_clone(dict(envelope))
        terminal_digest = _json_digest(terminal)
        messages = _json_clone(list(materialized_messages))
        now = self._clock()
        self._prune(now)

        with self._lock:
            inflight = self._owned_in_flight_locked(response_id, owner)
            if inflight is None:
                stored = self._owned_stored_locked(response_id, owner)
                if stored is not None:
                    return _json_clone(stored.envelope)
                tombstone = self._owned_tombstone_locked(response_id, owner)
                if tombstone is not None:
                    if tombstone.terminal_digest is None:
                        self._raise_lookup_locked(response_id, owner)
                    if tombstone.terminal_digest != terminal_digest:
                        raise ResponseRegistryError(
                            f"response {response_id!r} already has a different "
                            "terminal envelope",
                            code="response_terminal_conflict",
                            status_code=409,
                        )
                    return terminal
                self._raise_lookup_locked(response_id, owner)

            self._in_flight.pop(response_id, None)
            self._in_flight_bytes -= inflight.size_bytes
            inflight.terminal_envelope = terminal
            inflight.done.set()

            if inflight.cancel_requested:
                self._counters["cancel_settled_total"] += 1

            tombstone = self._owned_tombstone_locked(response_id, owner)
            if inflight.deleted or (
                tombstone is not None and tombstone.reason == "deleted"
            ):
                self._add_tombstone_locked(
                    response_id,
                    owner,
                    "deleted",
                    now,
                    cancel_pending=(
                        inflight.cancel_requested and not inflight.cancel_delivered
                    ),
                    cancel_reason=inflight.cancel_reason,
                    cancel_delivery_id=inflight.cancel_delivery_id,
                    terminal_digest=terminal_digest,
                )
                return _json_clone(terminal)

            if not inflight.store:
                self._add_tombstone_locked(
                    response_id,
                    owner,
                    "not_stored",
                    now,
                    terminal_digest=terminal_digest,
                )
                return _json_clone(terminal)

            size_bytes = _json_size(terminal, messages)
            if size_bytes > self.max_entry_bytes or size_bytes > self.max_bytes:
                self._add_tombstone_locked(
                    response_id,
                    owner,
                    "evicted",
                    now,
                    terminal_digest=terminal_digest,
                )
                self._counters["evicted_total"] += 1
                return _json_clone(terminal)

            self._stored[response_id] = _StoredResponse(
                owner_id=owner,
                envelope=terminal,
                materialized_messages=messages,
                committed_at=now,
                last_access_at=now,
                size_bytes=size_bytes,
            )
            self._bytes += size_bytes
            self._counters["committed_total"] += 1
            self._evict_pressure_locked(now)
            return _json_clone(terminal)

    def get(self, response_id: str, *, owner_id: str) -> dict[str, Any]:
        owner = _owner_id(owner_id)
        now = self._clock()
        self._prune(now)
        with self._lock:
            stored = self._owned_stored_locked(response_id, owner)
            if stored is not None:
                stored.last_access_at = now
                self._stored.move_to_end(response_id)
                return _json_clone(stored.envelope)
            self._raise_lookup_locked(response_id, owner)
        raise AssertionError("unreachable")

    def parent_messages(
        self, response_id: str, *, owner_id: str
    ) -> list[dict[str, Any]]:
        owner = _owner_id(owner_id)
        now = self._clock()
        self._prune(now)
        with self._lock:
            stored = self._owned_stored_locked(response_id, owner)
            if stored is not None:
                stored.last_access_at = now
                self._stored.move_to_end(response_id)
                messages = _json_clone(stored.materialized_messages)
                messages.extend(_assistant_output_messages(stored.envelope))
                return messages
            self._raise_lookup_locked(
                response_id,
                owner,
                param="previous_response_id",
            )
        raise AssertionError("unreachable")

    def input_messages(
        self, response_id: str, *, owner_id: str
    ) -> list[dict[str, Any]]:
        """Return model inputs without adding this response's output."""

        owner = _owner_id(owner_id)
        now = self._clock()
        self._prune(now)
        with self._lock:
            stored = self._owned_stored_locked(response_id, owner)
            if stored is not None:
                stored.last_access_at = now
                self._stored.move_to_end(response_id)
                return _json_clone(stored.materialized_messages)
            self._raise_lookup_locked(response_id, owner)
        raise AssertionError("unreachable")

    def request_cancel(
        self,
        response_id: str,
        reason: str,
        *,
        owner_id: str,
    ) -> _InFlightResponse:
        """Record one cancel intent; the first reason and delivery always win."""

        owner = _owner_id(owner_id)
        requested_reason = _cancel_reason(reason)
        self._prune()
        delivery: _CancelDelivery | None = None

        with self._lock:
            inflight = self._owned_in_flight_locked(response_id, owner)
            if inflight is None:
                stored = self._owned_stored_locked(response_id, owner)
                if stored is not None and self._is_cancelled(stored.envelope):
                    return self._terminal_waiter_from_stored(response_id, stored)
                if stored is not None:
                    raise ResponseRegistryError(
                        f"response {response_id!r} is already terminal",
                        code="response_not_cancellable",
                        status_code=409,
                    )
                self._raise_lookup_locked(response_id, owner)

            if inflight.deleted:
                self._raise_lookup_locked(response_id, owner)
            first_request = not inflight.cancel_requested
            if first_request:
                inflight.cancel_requested = True
                inflight.cancel_reason = requested_reason
                self._counters["cancel_requests_total"] += 1
            delivery = self._reserve_in_flight_cancel_locked(inflight)

        if delivery is not None:
            self._deliver_cancel(delivery)
        return inflight

    def wait_terminal(
        self,
        response: str | _InFlightResponse,
        timeout_s: float,
        *,
        owner_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Wait for terminal state by cancel handle or by owned response id."""

        if isinstance(response, _InFlightResponse):
            inflight = response
            if owner_id is not None and inflight.owner_id != _owner_id(owner_id):
                self._raise_not_found(inflight.response_id)
        elif isinstance(response, str):
            if owner_id is None:
                raise TypeError("owner_id is required when waiting by response id")
            owner = _owner_id(owner_id)
            self._prune()
            with self._lock:
                inflight = self._owned_in_flight_locked(response, owner)
                if inflight is None:
                    stored = self._owned_stored_locked(response, owner)
                    if stored is not None:
                        return _json_clone(stored.envelope)
                    self._raise_lookup_locked(response, owner)
        else:
            raise TypeError("response must be a response id or cancel wait handle")

        inflight.done.wait(max(0.0, float(timeout_s)))
        envelope = inflight.terminal_envelope
        return _json_clone(envelope) if envelope is not None else None

    def delete(self, response_id: str, *, owner_id: str) -> dict[str, Any]:
        owner = _owner_id(owner_id)
        now = self._clock()
        self._prune(now)
        delivery: _CancelDelivery | None = None

        with self._lock:
            stored = self._owned_stored_locked(response_id, owner)
            inflight = self._owned_in_flight_locked(response_id, owner)
            tombstone = self._owned_tombstone_locked(response_id, owner)
            already_deleted = tombstone is not None and tombstone.reason == "deleted"
            if already_deleted and inflight is None:
                return self._deleted_payload(response_id)
            if stored is None and inflight is None:
                self._raise_lookup_locked(response_id, owner)

            if stored is not None:
                self._stored.pop(response_id, None)
                self._bytes -= stored.size_bytes

            if inflight is not None:
                inflight.deleted = True
                if not inflight.cancel_requested:
                    inflight.cancel_requested = True
                    inflight.cancel_reason = "response_deleted"
                    self._counters["cancel_requests_total"] += 1
                delivery = self._reserve_in_flight_cancel_locked(inflight)

            self._add_tombstone_locked(
                response_id,
                owner,
                "deleted",
                now,
                cancel_pending=bool(
                    inflight is not None
                    and inflight.cancel_requested
                    and not inflight.cancel_delivered
                ),
                cancel_reason=(
                    inflight.cancel_reason if inflight is not None else None
                ),
                cancel_delivery_id=(
                    inflight.cancel_delivery_id if inflight is not None else None
                ),
            )
            if not already_deleted:
                self._counters["deleted_total"] += 1

        if delivery is not None:
            self._deliver_cancel(delivery)
        return self._deleted_payload(response_id)

    def stats(self) -> dict[str, Any]:
        self._prune()
        with self._lock:
            return {
                "entries": len(self._stored),
                "bytes": self._bytes,
                "in_flight": len(self._in_flight),
                "in_flight_bytes": self._in_flight_bytes,
                "tombstones": len(self._tombstones),
                "max_entries": self.max_entries,
                "max_in_flight": self.max_in_flight,
                "max_bytes": self.max_bytes,
                "max_entry_bytes": self.max_entry_bytes,
                "max_tombstones": self.max_tombstones,
                "idle_ttl_s": self.idle_ttl_s,
                "in_flight_ttl_s": self.in_flight_ttl_s,
                "shutting_down": self._shutting_down,
                **self._counters,
            }

    def is_in_flight(self, response_id: str, *, owner_id: str) -> bool:
        owner = _owner_id(owner_id)
        self._prune()
        with self._lock:
            return self._owned_in_flight_locked(response_id, owner) is not None

    def terminal_for(self, response_id: str, *, owner_id: str) -> dict[str, Any] | None:
        owner = _owner_id(owner_id)
        self._prune()
        with self._lock:
            stored = self._owned_stored_locked(response_id, owner)
            if stored is not None:
                return _json_clone(stored.envelope)
            inflight = self._owned_in_flight_locked(response_id, owner)
            if inflight is not None and inflight.terminal_envelope is not None:
                return _json_clone(inflight.terminal_envelope)
            return None

    def request_shutdown(self) -> tuple[threading.Event, ...]:
        """Stop new entries and deliver cancellation on the caller's thread."""
        with self._lock:
            self._shutting_down = True
            inflight = list(self._in_flight.values())
            deliveries: list[_CancelDelivery] = []
            for item in inflight:
                if not item.cancel_requested:
                    item.cancel_requested = True
                    item.cancel_reason = "registry_shutdown"
                delivery = self._reserve_in_flight_cancel_locked(item)
                if delivery is not None:
                    deliveries.append(delivery)

        for delivery in deliveries:
            self._deliver_cancel(delivery, suppress_errors=True)

        return tuple(item.done for item in inflight)

    @staticmethod
    def wait_for_shutdown(
        waiters: Sequence[threading.Event],
        timeout_s: float = 1.0,
    ) -> bool:
        """Wait without delivering callbacks; safe to move to a worker thread."""

        if timeout_s < 0:
            raise ValueError("timeout_s must be non-negative")
        deadline = time.monotonic() + timeout_s
        for waiter in waiters:
            if not waiter.wait(max(0.0, deadline - time.monotonic())):
                return False
        return True

    def shutdown(self, timeout_s: float = 1.0) -> None:
        """Synchronous compatibility wrapper for non-event-loop owners."""

        waiters = self.request_shutdown()
        self.wait_for_shutdown(waiters, timeout_s)

    def _require_accepting_locked(self) -> None:
        if self._shutting_down:
            raise ResponseRegistryError(
                "response registry is shutting down",
                code="response_registry_shutting_down",
                status_code=503,
            )

    def _owned_stored_locked(
        self, response_id: str, owner_id: str
    ) -> _StoredResponse | None:
        stored = self._stored.get(response_id)
        return stored if stored is not None and stored.owner_id == owner_id else None

    def _owned_in_flight_locked(
        self, response_id: str, owner_id: str
    ) -> _InFlightResponse | None:
        inflight = self._in_flight.get(response_id)
        return (
            inflight if inflight is not None and inflight.owner_id == owner_id else None
        )

    def _owned_tombstone_locked(
        self, response_id: str, owner_id: str
    ) -> _Tombstone | None:
        tombstone = self._tombstones.get(response_id)
        return (
            tombstone
            if tombstone is not None and tombstone.owner_id == owner_id
            else None
        )

    def _known_locked(self, response_id: str) -> bool:
        return (
            response_id in self._stored
            or response_id in self._in_flight
            or response_id in self._tombstones
        )

    def _raise_lookup_locked(
        self,
        response_id: str,
        owner_id: str,
        *,
        param: str | None = None,
    ) -> None:
        tombstone = self._owned_tombstone_locked(response_id, owner_id)
        inflight = self._owned_in_flight_locked(response_id, owner_id)
        if tombstone is not None and tombstone.reason == "deleted":
            raise ResponseRegistryError(
                f"response {response_id!r} was deleted",
                code="response_deleted",
                status_code=410,
                param=param,
            )
        if inflight is not None:
            if inflight.deleted:
                raise ResponseRegistryError(
                    f"response {response_id!r} was deleted",
                    code="response_deleted",
                    status_code=410,
                    param=param,
                )
            raise ResponseRegistryError(
                f"response {response_id!r} is still in progress",
                code="response_in_progress",
                status_code=409,
                param=param,
            )
        if tombstone is not None:
            if tombstone.reason == "not_stored":
                self._raise_not_found(response_id, param=param)
            raise ResponseRegistryError(
                f"response {response_id!r} was {tombstone.reason}",
                code=f"response_{tombstone.reason}",
                status_code=410,
                param=param,
            )
        self._raise_not_found(response_id, param=param)

    @staticmethod
    def _raise_not_found(response_id: str, *, param: str | None = None) -> None:
        raise ResponseRegistryError(
            f"response {response_id!r} was not found",
            code="response_not_found",
            status_code=404,
            param=param,
        )

    def _prune(self, now: float | None = None) -> None:
        with self._lock:
            deliveries = self._prune_locked(now)
        for delivery in deliveries:
            self._deliver_cancel(delivery, suppress_errors=True)

    def _prune_locked(self, now: float | None = None) -> list[_CancelDelivery]:
        current = self._clock() if now is None else now
        deliveries: list[_CancelDelivery] = []
        stale_in_flight = [
            response_id
            for response_id, inflight in self._in_flight.items()
            if current - inflight.started_at >= self.in_flight_ttl_s
        ]
        for response_id in stale_in_flight:
            inflight = self._in_flight.pop(response_id)
            self._in_flight_bytes -= inflight.size_bytes
            already_requested = inflight.cancel_requested
            inflight.cancel_requested = True
            if inflight.cancel_reason is None:
                inflight.cancel_reason = "response_timeout"
            delivery = self._reserve_in_flight_cancel_locked(inflight)
            if delivery is not None:
                deliveries.append(delivery)
            inflight.terminal_envelope = {
                "id": response_id,
                "object": "response",
                "status": "failed",
                "error": {
                    "code": "response_timeout",
                    "message": "response exceeded the local in-flight lifetime",
                },
            }
            inflight.done.set()
            reason = "deleted" if inflight.deleted else "timeout"
            self._add_tombstone_locked(
                response_id,
                inflight.owner_id,
                reason,
                current,
                cancel_pending=not inflight.cancel_delivered,
                cancel_reason=inflight.cancel_reason,
                cancel_delivery_id=inflight.cancel_delivery_id,
                terminal_digest=_json_digest(inflight.terminal_envelope),
            )
            self._counters["timed_out_total"] += 1
            if already_requested:
                self._counters["cancel_settled_total"] += 1

        expired = [
            response_id
            for response_id, stored in self._stored.items()
            if current - stored.last_access_at >= self.idle_ttl_s
        ]
        for response_id in expired:
            stored = self._stored.pop(response_id)
            self._bytes -= stored.size_bytes
            self._add_tombstone_locked(
                response_id,
                stored.owner_id,
                "evicted",
                current,
                terminal_digest=_json_digest(stored.envelope),
            )
            self._counters["expired_total"] += 1
            self._counters["evicted_total"] += 1

        stale_tombstones = [
            response_id
            for response_id, tombstone in self._tombstones.items()
            if current - tombstone.created_at >= self.idle_ttl_s
        ]
        for response_id in stale_tombstones:
            self._tombstones.pop(response_id, None)
        return deliveries

    def _reserve_in_flight_cancel_locked(
        self,
        inflight: _InFlightResponse,
    ) -> _CancelDelivery | None:
        if (
            not inflight.cancel_requested
            or inflight.cancel is None
            or inflight.cancel_delivered
            or inflight.cancel_delivery_id is not None
        ):
            return None
        delivery_id = uuid.uuid4().hex
        inflight.cancel_delivery_id = delivery_id
        tombstone = self._owned_tombstone_locked(
            inflight.response_id,
            inflight.owner_id,
        )
        if tombstone is not None and tombstone.cancel_pending:
            tombstone.cancel_delivery_id = delivery_id
        return _CancelDelivery(
            response_id=inflight.response_id,
            owner_id=inflight.owner_id,
            cancel=inflight.cancel,
            reason=inflight.cancel_reason or "client_cancelled",
            delivery_id=delivery_id,
            state=inflight,
        )

    def _reserve_tombstone_cancel_locked(
        self,
        response_id: str,
        tombstone: _Tombstone,
        cancel: Callable[[str], Any],
    ) -> _CancelDelivery | None:
        if not tombstone.cancel_pending or tombstone.cancel_delivery_id is not None:
            return None
        delivery_id = uuid.uuid4().hex
        tombstone.cancel_delivery_id = delivery_id
        return _CancelDelivery(
            response_id=response_id,
            owner_id=tombstone.owner_id,
            cancel=cancel,
            reason=tombstone.cancel_reason or "client_cancelled",
            delivery_id=delivery_id,
            state=tombstone,
        )

    def _deliver_cancel(
        self,
        delivery: _CancelDelivery,
        *,
        suppress_errors: bool = False,
    ) -> bool:
        try:
            result = delivery.cancel(delivery.reason)
            if result is not None and not isinstance(result, bool):
                raise TypeError("cancel callback must return None, True, or False")
            if result is False:
                raise CancelDeliveryRejected(
                    f"cancel delivery for {delivery.response_id!r} was rejected"
                )
        except BaseException as error:
            self._finish_cancel_delivery(delivery, accepted=False)
            if suppress_errors and isinstance(error, Exception):
                return False
            raise
        self._finish_cancel_delivery(delivery, accepted=True)
        return True

    def _finish_cancel_delivery(
        self,
        delivery: _CancelDelivery,
        *,
        accepted: bool,
    ) -> None:
        with self._lock:
            states: list[_InFlightResponse | _Tombstone] = [delivery.state]
            inflight = self._owned_in_flight_locked(
                delivery.response_id,
                delivery.owner_id,
            )
            tombstone = self._owned_tombstone_locked(
                delivery.response_id,
                delivery.owner_id,
            )
            if inflight is not None:
                states.append(inflight)
            if tombstone is not None:
                states.append(tombstone)

            seen: set[int] = set()
            for state in states:
                identity = id(state)
                if identity in seen or state.cancel_delivery_id != delivery.delivery_id:
                    continue
                seen.add(identity)
                state.cancel_delivery_id = None
                if not accepted:
                    continue
                if isinstance(state, _InFlightResponse):
                    state.cancel_delivered = True
                else:
                    state.cancel_pending = False
            if not accepted:
                self._counters["cancel_delivery_failures_total"] += 1

    def _evict_pressure_locked(self, now: float) -> None:
        while len(self._stored) > self.max_entries or self._bytes > self.max_bytes:
            response_id, stored = self._stored.popitem(last=False)
            self._bytes -= stored.size_bytes
            self._add_tombstone_locked(
                response_id,
                stored.owner_id,
                "evicted",
                now,
                terminal_digest=_json_digest(stored.envelope),
            )
            self._counters["evicted_total"] += 1

    def _add_tombstone_locked(
        self,
        response_id: str,
        owner_id: str,
        reason: str,
        now: float,
        *,
        cancel_pending: bool = False,
        cancel_reason: str | None = None,
        cancel_delivery_id: str | None = None,
        terminal_digest: str | None = None,
    ) -> None:
        previous = self._tombstones.get(response_id)
        if terminal_digest is None and previous is not None:
            terminal_digest = previous.terminal_digest
        self._tombstones.pop(response_id, None)
        self._tombstones[response_id] = _Tombstone(
            owner_id=owner_id,
            reason=reason,
            created_at=now,
            cancel_pending=cancel_pending,
            cancel_reason=cancel_reason,
            cancel_delivery_id=cancel_delivery_id,
            terminal_digest=terminal_digest,
        )
        while len(self._tombstones) > self.max_tombstones:
            self._tombstones.popitem(last=False)

    @staticmethod
    def _terminal_waiter_from_stored(
        response_id: str, stored: _StoredResponse
    ) -> _InFlightResponse:
        error = stored.envelope.get("error") or {}
        reason = str(error.get("message") or "client_cancelled")
        waiter = _InFlightResponse(
            response_id=response_id,
            owner_id=stored.owner_id,
            store=True,
            materialized_messages=_json_clone(stored.materialized_messages),
            cancel=None,
            started_at=stored.committed_at,
            size_bytes=0,
            cancel_requested=True,
            cancel_reason=reason,
            cancel_delivered=True,
            terminal_envelope=_json_clone(stored.envelope),
        )
        waiter.done.set()
        return waiter

    @staticmethod
    def _is_cancelled(envelope: Mapping[str, Any]) -> bool:
        error = envelope.get("error") or {}
        return envelope.get("status") == "cancelled" or (
            isinstance(error, Mapping) and error.get("code") == "request_cancelled"
        )

    @staticmethod
    def _deleted_payload(response_id: str) -> dict[str, Any]:
        return {"id": response_id, "object": "response.deleted", "deleted": True}
