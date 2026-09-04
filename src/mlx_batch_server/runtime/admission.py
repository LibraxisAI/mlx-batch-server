"""Bounded process/role admission with idempotent request leases."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field

from .contracts import (
    AdmissionDecision,
    AdmissionDisposition,
    AdmissionLease,
    RoleName,
)


class AdmissionRejected(RuntimeError):
    """Raised when bounded admission cannot grant a lease."""

    def __init__(self, decision: AdmissionDecision) -> None:
        super().__init__(decision.reason)
        self.decision = decision


@dataclass(slots=True)
class _RoleGate:
    limit: int
    max_waiters: int
    active: int = 0
    waiters: deque[asyncio.Future[None]] = field(default_factory=deque)


class _RequestLease:
    def __init__(
        self,
        controller: AdmissionController,
        role: str,
        decision: AdmissionDecision,
    ) -> None:
        self._controller = controller
        self._role = role
        self._decision = decision
        self._released = False

    @property
    def decision(self) -> AdmissionDecision:
        return self._decision

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._controller._release(self._role)

    def __enter__(self) -> _RequestLease:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()

    async def __aenter__(self) -> _RequestLease:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        self.release()


class AdmissionController:
    """Apply bounded FIFO admission independently to each configured role.

    A granted slot is transferred directly to the oldest waiter on release, so
    active work never exceeds the role limit. Waiting is bounded separately;
    queue overflow rejects immediately and a caller deadline returns RETRY.
    """

    def __init__(
        self,
        max_active_requests: int = 8,
        *,
        max_waiters: int = 64,
        role_limits: Mapping[RoleName | str, int] | None = None,
    ) -> None:
        if max_active_requests < 1:
            raise ValueError("max_active_requests must be positive")
        if max_waiters < 0:
            raise ValueError("max_waiters must be non-negative")
        self._default_limit = max_active_requests
        self._max_waiters = max_waiters
        self._role_limits = {
            self._role_key(role): self._validated_limit(limit)
            for role, limit in (role_limits or {}).items()
        }
        self._gates: dict[str, _RoleGate] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    @staticmethod
    def _role_key(role: RoleName | str) -> str:
        value = role.value if isinstance(role, RoleName) else str(role)
        value = value.strip().lower()
        if not value:
            raise ValueError("role must not be empty")
        return value

    @staticmethod
    def _validated_limit(limit: int) -> int:
        if limit < 1:
            raise ValueError("role admission limit must be positive")
        return limit

    def _gate(self, role: str) -> _RoleGate:
        gate = self._gates.get(role)
        if gate is None:
            gate = _RoleGate(
                limit=self._role_limits.get(role, self._default_limit),
                max_waiters=self._max_waiters,
            )
            self._gates[role] = gate
        return gate

    def _bind_loop(self) -> asyncio.AbstractEventLoop:
        loop = asyncio.get_running_loop()
        if self._loop is None or self._loop.is_closed():
            self._loop = loop
        elif self._loop is not loop:
            raise RuntimeError("AdmissionController cannot span event loops")
        return loop

    @staticmethod
    def _discard_finished_waiters(gate: _RoleGate) -> None:
        gate.waiters = deque(waiter for waiter in gate.waiters if not waiter.done())

    def decide(self, role: RoleName | str) -> AdmissionDecision:
        """Return the current non-reserving admission decision for a role."""
        role_key = self._role_key(role)
        gate = self._gate(role_key)
        self._discard_finished_waiters(gate)
        if gate.active < gate.limit and not gate.waiters:
            return AdmissionDecision(
                AdmissionDisposition.ADMIT,
                "capacity_available",
            )
        if len(gate.waiters) >= gate.max_waiters:
            return AdmissionDecision(
                AdmissionDisposition.REJECT,
                "admission_queue_full",
            )
        return AdmissionDecision(AdmissionDisposition.WAIT, "capacity_in_use")

    async def acquire(
        self,
        role: RoleName | str,
        *,
        timeout_s: float | None = None,
    ) -> AdmissionLease:
        """Wait for and return one admission lease, bounded by queue and timeout."""
        if timeout_s is not None and timeout_s < 0:
            raise ValueError("timeout_s must be non-negative")

        loop = self._bind_loop()
        role_key = self._role_key(role)
        gate = self._gate(role_key)
        self._discard_finished_waiters(gate)

        if gate.active < gate.limit and not gate.waiters:
            gate.active += 1
            return self._lease(role_key)

        if len(gate.waiters) >= gate.max_waiters:
            raise AdmissionRejected(
                AdmissionDecision(
                    AdmissionDisposition.REJECT,
                    "admission_queue_full",
                )
            )

        waiter: asyncio.Future[None] = loop.create_future()
        gate.waiters.append(waiter)
        deadline = None if timeout_s is None else loop.time() + timeout_s
        try:
            if timeout_s is None:
                await asyncio.shield(waiter)
            else:
                await asyncio.wait_for(asyncio.shield(waiter), timeout_s)
        except TimeoutError as exc:
            if not self._remove_waiter(gate, waiter):
                self._release_now(role_key)
            raise AdmissionRejected(
                AdmissionDecision(
                    AdmissionDisposition.RETRY,
                    "admission_deadline_exceeded",
                    retry_after_s=0.0,
                    deadline_s=deadline,
                )
            ) from exc
        except asyncio.CancelledError:
            if not self._remove_waiter(gate, waiter):
                self._release_now(role_key)
            raise
        return self._lease(role_key)

    def _lease(self, role: str) -> _RequestLease:
        return _RequestLease(
            self,
            role,
            AdmissionDecision(AdmissionDisposition.ADMIT, "capacity_acquired"),
        )

    @staticmethod
    def _remove_waiter(
        gate: _RoleGate,
        waiter: asyncio.Future[None],
    ) -> bool:
        try:
            gate.waiters.remove(waiter)
        except ValueError:
            return False
        waiter.cancel()
        return True

    def _release(self, role: str) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            self._release_now(role)
            return
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is loop:
            self._release_now(role)
        else:
            loop.call_soon_threadsafe(self._release_now, role)

    def _release_now(self, role: str) -> None:
        gate = self._gate(role)
        while gate.waiters:
            waiter = gate.waiters.popleft()
            if waiter.done():
                continue
            waiter.set_result(None)
            return
        if gate.active <= 0:
            raise RuntimeError(f"admission lease underflow for role {role!r}")
        gate.active -= 1

    def snapshot(self, role: RoleName | str) -> dict[str, int]:
        """Return observer-only counters without reserving capacity."""
        role_key = self._role_key(role)
        gate = self._gate(role_key)
        self._discard_finished_waiters(gate)
        return {
            "active": gate.active,
            "waiting": len(gate.waiters),
            "limit": gate.limit,
            "max_waiters": gate.max_waiters,
        }


__all__ = [
    "AdmissionController",
    "AdmissionDecision",
    "AdmissionDisposition",
    "AdmissionLease",
    "AdmissionRejected",
]
