"""Target-owned runtime start service for one admitted backend turn."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable
from typing import TYPE_CHECKING

from .contracts import (
    AdmissionLease,
    BackendHandle,
    BackendTurn,
    CancelToken,
    GenerationRequest,
    PreparedBackendHandle,
    PreparedGenerationRequest,
    RoleName,
    TurnSink,
)

if TYPE_CHECKING:
    from .manager import RuntimeManager


class RuntimeStartError(RuntimeError):
    """Raised when a runtime turn cannot be started without changing identity."""


class FirstWriterCancelToken:
    """Cancellation state committed only after backend delivery is accepted."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reason: str | None = None
        self._pending_reason: str | None = None
        self._delivery_reserved = False

    @property
    def cancelled(self) -> bool:
        with self._lock:
            return self._reason is not None

    @property
    def reason(self) -> str | None:
        with self._lock:
            return self._reason

    def cancel(self, reason: str) -> bool:
        canonical = self._canonical_reason(reason)
        with self._lock:
            if self._reason is not None or self._pending_reason is not None:
                return False
            self._reason = canonical
            return True

    def reserve_delivery(self, reason: str) -> str | None:
        """Reserve the first reason without exposing cancellation before ACK."""

        canonical = self._canonical_reason(reason)
        with self._lock:
            if self._reason is not None or self._delivery_reserved:
                return None
            if self._pending_reason is None:
                self._pending_reason = canonical
            self._delivery_reserved = True
            return self._pending_reason

    def acknowledge_delivery(self, reason: str) -> None:
        with self._lock:
            self._require_reservation_locked(reason)
            self._reason = self._pending_reason
            self._pending_reason = None
            self._delivery_reserved = False

    def reject_delivery(self, reason: str) -> None:
        with self._lock:
            self._require_reservation_locked(reason)
            self._delivery_reserved = False

    def _require_reservation_locked(self, reason: str) -> None:
        if not self._delivery_reserved or self._pending_reason != reason:
            raise RuntimeStartError("cancel delivery reservation is no longer current")

    @staticmethod
    def _canonical_reason(reason: str) -> str:
        if not isinstance(reason, str):
            raise TypeError("cancellation reason must be a string")
        canonical = reason.strip()
        if not canonical:
            raise ValueError("cancellation reason must not be empty")
        return canonical


def _cancel_acknowledged(result: object) -> bool:
    """Normalize the bounded BackendTurn cancel acknowledgement contract.

    Existing ``BackendTurn.cancel`` implementations return ``None`` and are
    therefore acknowledged for compatibility. New implementations may return
    ``True`` for accepted or ``False`` for retryable rejection. No other return
    shape is admitted.
    """

    if result is None or result is True:
        return True
    if result is False:
        return False
    raise RuntimeStartError("BackendTurn.cancel must return None, True, or False")


class _AdmissionGuard:
    """Call the underlying admission release exactly once."""

    def __init__(self, lease: AdmissionLease) -> None:
        self._lease = lease
        self._lock = threading.Lock()
        self._released = False

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
        self._lease.release()


class _ManagedBackendTurn:
    """Keep admission alive while the backend owns work for this response."""

    def __init__(
        self,
        backend_turn: BackendTurn,
        cancel_token: FirstWriterCancelToken,
        admission: _AdmissionGuard,
    ) -> None:
        self._backend_turn = backend_turn
        self._cancel_token = cancel_token
        self._admission = admission
        self._cancel_delivery_lock = threading.Lock()
        self._closed_task = asyncio.create_task(
            self._wait_for_backend(),
            name=f"runtime-turn-drain:{backend_turn.response_id}",
        )

    @property
    def response_id(self) -> str:
        return self._backend_turn.response_id

    @property
    def cancel_token(self) -> CancelToken:
        return self._cancel_token

    def cancel(self, reason: str) -> bool:
        """Deliver the first reason once, committing the token only after ACK."""

        with self._cancel_delivery_lock:
            canonical = self._cancel_token.reserve_delivery(reason)
            if canonical is None:
                return self._cancel_token.cancelled
            try:
                accepted = _cancel_acknowledged(self._backend_turn.cancel(canonical))
            except BaseException:
                self._cancel_token.reject_delivery(canonical)
                raise
            if not accepted:
                self._cancel_token.reject_delivery(canonical)
                return False
            self._cancel_token.acknowledge_delivery(canonical)
            return True

    def wait_closed(self) -> Awaitable[None]:
        return asyncio.shield(self._closed_task)

    async def _wait_for_backend(self) -> None:
        try:
            await self._backend_turn.wait_closed()
        finally:
            self._admission.release()


class RuntimeStartService:
    """Resolve a trusted role and start exactly one admitted backend turn.

    Role-aware acquisition is mandatory by default. Setting
    ``direct_acquire_without_role_services`` is an explicit deployment choice
    for a manager constructed without role/readiness services; it never changes
    the request's RuntimeKey or selects another backend.
    """

    def __init__(
        self,
        manager: RuntimeManager,
        *,
        default_role: RoleName | str = RoleName.MAIN,
        role_metadata_field: str = "runtime_role",
        admission_timeout_s: float | None = None,
        direct_acquire_without_role_services: bool = False,
    ) -> None:
        field = role_metadata_field.strip()
        if not field:
            raise ValueError("role_metadata_field must not be empty")
        if admission_timeout_s is not None and admission_timeout_s < 0:
            raise ValueError("admission_timeout_s must be non-negative")

        self._manager = manager
        self._default_role = self._normalize_role(default_role)
        self._role_metadata_field = field
        self._admission_timeout_s = admission_timeout_s
        self._direct_acquire_without_role_services = (
            direct_acquire_without_role_services
        )

    async def start(
        self,
        request: GenerationRequest,
        sink: TurnSink,
        *,
        cancel: FirstWriterCancelToken | None = None,
    ) -> BackendTurn:
        role = self._request_role(request)
        handle = await self._acquire_exact_runtime(role, request)
        if handle.runtime_key != request.runtime:
            raise RuntimeStartError(
                "runtime manager returned a handle for a different RuntimeKey"
            )

        cancel_token = cancel or FirstWriterCancelToken()
        if not isinstance(cancel_token, FirstWriterCancelToken):
            raise TypeError("cancel must be a FirstWriterCancelToken")
        _raise_if_cancelled(cancel_token)
        prepared: PreparedGenerationRequest | None = None
        prepare_request = getattr(handle, "prepare_request", None)
        start_prepared_turn = getattr(handle, "start_prepared_turn", None)
        supports_prepare = callable(prepare_request)
        supports_prepared_start = callable(start_prepared_turn)
        if supports_prepare != supports_prepared_start:
            raise RuntimeStartError(
                "backend must implement both prepare_request and start_prepared_turn"
            )
        if supports_prepare:
            assert isinstance(handle, PreparedBackendHandle)
            try:
                prepared = await handle.prepare_request(request, cancel_token)
            except asyncio.CancelledError:
                cancel_token.cancel("request preparation cancelled")
                raise
            if not isinstance(prepared, PreparedGenerationRequest):
                raise RuntimeStartError(
                    "backend request preparation returned an invalid envelope"
                )
            if prepared.request is not request:
                raise RuntimeStartError(
                    "backend request preparation replaced the canonical request"
                )

        _raise_if_cancelled(cancel_token)
        lease = await self._manager.admit(
            role,
            timeout_s=self._admission_timeout_s,
        )
        admission = _AdmissionGuard(lease)
        try:
            if prepared is None:
                backend_turn = await handle.start_turn(request, sink, cancel_token)
            else:
                assert isinstance(handle, PreparedBackendHandle)
                backend_turn = await handle.start_prepared_turn(
                    prepared,
                    sink,
                    cancel_token,
                )
        except BaseException:
            admission.release()
            raise
        return _ManagedBackendTurn(backend_turn, cancel_token, admission)

    async def _acquire_exact_runtime(
        self,
        role: RoleName,
        request: GenerationRequest,
    ) -> BackendHandle:
        if self._direct_acquire_without_role_services:
            return await self._manager.acquire(request.runtime)
        return await self._manager.acquire_role(role, runtime=request.runtime)

    def _request_role(self, request: GenerationRequest) -> RoleName:
        role = request.metadata.get(self._role_metadata_field, self._default_role)
        return self._normalize_role(role)

    @staticmethod
    def _normalize_role(role: object) -> RoleName:
        if isinstance(role, RoleName):
            return role
        if not isinstance(role, str):
            raise RuntimeStartError("trusted runtime role must be a string")
        try:
            return RoleName(role.strip().lower())
        except ValueError as exc:
            raise RuntimeStartError(f"unknown runtime role {role!r}") from exc


def _raise_if_cancelled(cancel: CancelToken) -> None:
    if cancel.cancelled:
        raise asyncio.CancelledError(
            cancel.reason or "request cancelled before admission"
        )


__all__ = [
    "FirstWriterCancelToken",
    "RuntimeStartError",
    "RuntimeStartService",
]
