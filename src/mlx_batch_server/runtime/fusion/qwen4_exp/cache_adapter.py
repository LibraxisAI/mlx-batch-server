# SPDX-License-Identifier: Apache-2.0
"""Atomic semantic-cache transactions for Qwen4Exp exact MTP verification.

The target contract is adapted from MTPLX ``mtplx/models/qwen4_exp.py`` and
``mtplx/cache_state.py`` at commit
``6d0ddf0575faa9acf77e63c57e48ea1602a7e4ab``. The opaque bundle represents
QSA attention state, GDN recurrence, and PLE state together. Generic paged,
prefix, or SSD cache adapters may budget and retain the bundle, but cannot
commit or roll back its members independently.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from ...contracts import RuntimeKey


class Qwen4ExpSemanticCacheError(RuntimeError):
    """Base failure for the Qwen4Exp semantic-cache boundary."""


class Qwen4ExpSemanticCacheIdentityError(Qwen4ExpSemanticCacheError):
    """A request, lease, runtime, or verify transaction did not match."""


class Qwen4ExpSemanticCacheStateError(Qwen4ExpSemanticCacheError):
    """A semantic-cache operation was attempted in an invalid phase."""


class Qwen4ExpSemanticCachePoisonedError(Qwen4ExpSemanticCacheError):
    """An uncertain partial mutation requires request-level abort and cleanup."""


class Qwen4ExpVerifyPhase(StrEnum):
    IDLE = "idle"
    CAPTURING = "capturing"
    FORWARDED = "forwarded"
    POISONED = "poisoned"
    CLOSED = "closed"


class Qwen4ExpVerifyDisposition(StrEnum):
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"


@dataclass(frozen=True, slots=True)
class Qwen4ExpSemanticCacheBinding:
    """Exact request identity and its opaque MTPLX semantic-cache bundle."""

    request_id: str
    lease_id: str
    runtime: RuntimeKey
    bundle: object

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if not self.lease_id:
            raise ValueError("lease_id must not be empty")
        if not isinstance(self.runtime, RuntimeKey):
            raise TypeError("runtime must be a RuntimeKey")
        if self.bundle is None:
            raise ValueError("semantic cache bundle must not be None")


@dataclass(frozen=True, slots=True)
class Qwen4ExpVerifyReceipt:
    """Observed resolution of one exact speculative verify transaction."""

    request_id: str
    lease_id: str
    verify_id: int
    verified_tokens: int
    keep_tokens: int
    disposition: Qwen4ExpVerifyDisposition
    capture_cleared: bool
    requires_reforward: bool

    def __post_init__(self) -> None:
        if not self.request_id or not self.lease_id:
            raise ValueError("receipt identity must not be empty")
        if self.verify_id < 1:
            raise ValueError("verify_id must be positive")
        if self.verified_tokens < 1:
            raise ValueError("verified_tokens must be positive")
        if not 0 <= self.keep_tokens <= self.verified_tokens:
            raise ValueError("keep_tokens must fit inside the verified window")
        if not self.capture_cleared:
            raise ValueError("successful receipts require cleared capture state")
        expected_reforward = self.disposition is Qwen4ExpVerifyDisposition.ROLLED_BACK
        if self.requires_reforward is not expected_reforward:
            raise ValueError("requires_reforward must match the disposition")


@runtime_checkable
class Qwen4ExpSemanticCacheKernelPort(Protocol):
    """Injected tensor implementation called only on the inference thread.

    ``commit_verified_window`` returning ``False`` promises that validation
    refused before mutation. Raising does not make that promise and poisons
    the adapter, preventing an unsafe automatic rollback over partial state.
    """

    def snapshot(self, bundle: object) -> object: ...

    def begin_capture(self, bundle: object) -> object: ...

    def end_capture(self, bundle: object, capture_token: object) -> None: ...

    def commit_verified_window(
        self,
        bundle: object,
        snapshot: object,
        *,
        keep_tokens: int,
        verified_tokens: int,
    ) -> bool: ...

    def rollback_after_verify(
        self,
        bundle: object,
        snapshot: object,
        verified_tokens: int,
    ) -> None: ...

    def clear_verify_capture(self, bundle: object) -> None: ...


@dataclass(slots=True)
class _ActiveVerify:
    verify_id: int
    verified_tokens: int
    snapshot: object
    capture_token: object


class Qwen4ExpVerifyLease:
    """Stale-safe handle for one active semantic-cache transaction."""

    def __init__(
        self,
        adapter: Qwen4ExpSemanticCacheAdapter,
        verify_id: int,
    ) -> None:
        self._adapter = adapter
        self._verify_id = verify_id

    @property
    def verify_id(self) -> int:
        return self._verify_id

    def forward_complete(self) -> None:
        self._adapter._forward_complete(self._verify_id)

    def resolve(self, *, keep_tokens: int) -> Qwen4ExpVerifyReceipt:
        return self._adapter._resolve(self._verify_id, keep_tokens=keep_tokens)

    def abort(self) -> Qwen4ExpVerifyReceipt:
        return self._adapter._abort(self._verify_id)


class Qwen4ExpSemanticCacheAdapter:
    """Serialize one request's exact MTP semantic-cache transactions."""

    def __init__(
        self,
        *,
        binding: Qwen4ExpSemanticCacheBinding,
        kernel: Qwen4ExpSemanticCacheKernelPort,
    ) -> None:
        if not isinstance(binding, Qwen4ExpSemanticCacheBinding):
            raise TypeError("binding must be a Qwen4ExpSemanticCacheBinding")
        self._binding = binding
        self._kernel = kernel
        self._owner_thread_id = threading.get_ident()
        self._phase = Qwen4ExpVerifyPhase.IDLE
        self._active: _ActiveVerify | None = None
        self._next_verify_id = 1
        self._poison_reason: str | None = None

    @property
    def binding(self) -> Qwen4ExpSemanticCacheBinding:
        return self._binding

    @property
    def phase(self) -> Qwen4ExpVerifyPhase:
        return self._phase

    @property
    def poison_reason(self) -> str | None:
        return self._poison_reason

    def begin_verify(self, *, verified_tokens: int) -> Qwen4ExpVerifyLease:
        self._require_owner_thread()
        self._require_usable()
        if self._active is not None or self._phase is not Qwen4ExpVerifyPhase.IDLE:
            raise Qwen4ExpSemanticCacheStateError(
                "a semantic-cache verify transaction is already active"
            )
        if type(verified_tokens) is not int or verified_tokens < 1:
            raise ValueError("verified_tokens must be a positive integer")

        bundle = self._binding.bundle
        snapshot = self._kernel.snapshot(bundle)
        if snapshot is None:
            raise Qwen4ExpSemanticCacheStateError(
                "semantic cache snapshot must not be None"
            )
        try:
            capture_token = self._kernel.begin_capture(bundle)
        except BaseException:
            try:
                self._kernel.clear_verify_capture(bundle)
            except BaseException as clear_error:
                self._poison("capture setup and cleanup both failed", clear_error)
            raise
        if capture_token is None:
            try:
                self._kernel.clear_verify_capture(bundle)
            except BaseException as clear_error:
                self._poison("capture token missing and cleanup failed", clear_error)
            raise Qwen4ExpSemanticCacheStateError(
                "semantic cache capture token must not be None"
            )

        verify_id = self._next_verify_id
        self._next_verify_id += 1
        self._active = _ActiveVerify(
            verify_id=verify_id,
            verified_tokens=verified_tokens,
            snapshot=snapshot,
            capture_token=capture_token,
        )
        self._phase = Qwen4ExpVerifyPhase.CAPTURING
        return Qwen4ExpVerifyLease(self, verify_id)

    def close(self) -> None:
        self._require_owner_thread()
        self._require_usable()
        if self._active is not None:
            raise Qwen4ExpSemanticCacheStateError(
                "cannot close with an active verify transaction"
            )
        self._phase = Qwen4ExpVerifyPhase.CLOSED

    def _forward_complete(self, verify_id: int) -> None:
        self._require_owner_thread()
        active = self._require_active(verify_id, Qwen4ExpVerifyPhase.CAPTURING)
        try:
            self._kernel.end_capture(
                self._binding.bundle,
                active.capture_token,
            )
        except BaseException as error:
            self._poison("ending verify capture failed", error)
        self._phase = Qwen4ExpVerifyPhase.FORWARDED

    def _resolve(
        self,
        verify_id: int,
        *,
        keep_tokens: int,
    ) -> Qwen4ExpVerifyReceipt:
        self._require_owner_thread()
        active = self._require_active(verify_id, Qwen4ExpVerifyPhase.FORWARDED)
        valid_keep = type(keep_tokens) is int and (
            0 <= keep_tokens <= active.verified_tokens
        )
        if not valid_keep:
            raise ValueError("keep_tokens must fit inside the verified window")

        bundle = self._binding.bundle
        committed = False
        if keep_tokens > 0:
            try:
                committed = self._kernel.commit_verified_window(
                    bundle,
                    active.snapshot,
                    keep_tokens=keep_tokens,
                    verified_tokens=active.verified_tokens,
                )
            except BaseException as error:
                self._poison("semantic cache commit raised", error)
            if type(committed) is not bool:
                self._poison(
                    "semantic cache commit returned a non-boolean result",
                    TypeError(type(committed).__name__),
                )

        disposition = Qwen4ExpVerifyDisposition.COMMITTED
        if not committed:
            try:
                self._kernel.rollback_after_verify(
                    bundle,
                    active.snapshot,
                    active.verified_tokens,
                )
            except BaseException as error:
                self._poison("semantic cache rollback failed", error)
            disposition = Qwen4ExpVerifyDisposition.ROLLED_BACK

        self._clear_capture_or_poison()
        return self._finish(active, keep_tokens, disposition)

    def _abort(self, verify_id: int) -> Qwen4ExpVerifyReceipt:
        self._require_owner_thread()
        active = self._require_active(
            verify_id,
            Qwen4ExpVerifyPhase.CAPTURING,
            Qwen4ExpVerifyPhase.FORWARDED,
        )
        bundle = self._binding.bundle
        if self._phase is Qwen4ExpVerifyPhase.CAPTURING:
            try:
                self._kernel.end_capture(bundle, active.capture_token)
            except BaseException as error:
                self._poison("ending aborted verify capture failed", error)
        try:
            self._kernel.rollback_after_verify(
                bundle,
                active.snapshot,
                active.verified_tokens,
            )
        except BaseException as error:
            self._poison("aborted semantic cache rollback failed", error)
        self._clear_capture_or_poison()
        return self._finish(active, 0, Qwen4ExpVerifyDisposition.ROLLED_BACK)

    def _finish(
        self,
        active: _ActiveVerify,
        keep_tokens: int,
        disposition: Qwen4ExpVerifyDisposition,
    ) -> Qwen4ExpVerifyReceipt:
        receipt = Qwen4ExpVerifyReceipt(
            request_id=self._binding.request_id,
            lease_id=self._binding.lease_id,
            verify_id=active.verify_id,
            verified_tokens=active.verified_tokens,
            keep_tokens=keep_tokens,
            disposition=disposition,
            capture_cleared=True,
            requires_reforward=(disposition is Qwen4ExpVerifyDisposition.ROLLED_BACK),
        )
        self._active = None
        self._phase = Qwen4ExpVerifyPhase.IDLE
        return receipt

    def _clear_capture_or_poison(self) -> None:
        try:
            self._kernel.clear_verify_capture(self._binding.bundle)
        except BaseException as error:
            self._poison("clearing verify capture failed", error)

    def _require_active(
        self,
        verify_id: int,
        *phases: Qwen4ExpVerifyPhase,
    ) -> _ActiveVerify:
        self._require_usable()
        active = self._active
        if active is None or active.verify_id != verify_id:
            raise Qwen4ExpSemanticCacheIdentityError(
                "stale or foreign semantic-cache verify lease"
            )
        if self._phase not in phases:
            expected = ", ".join(phase.value for phase in phases)
            raise Qwen4ExpSemanticCacheStateError(
                f"verify transaction is {self._phase.value}; expected {expected}"
            )
        return active

    def _require_owner_thread(self) -> None:
        if threading.get_ident() != self._owner_thread_id:
            raise Qwen4ExpSemanticCacheStateError(
                "semantic-cache operations require the inference owner thread"
            )

    def _require_usable(self) -> None:
        if self._phase is Qwen4ExpVerifyPhase.POISONED:
            raise Qwen4ExpSemanticCachePoisonedError(
                self._poison_reason or "semantic cache is poisoned"
            )
        if self._phase is Qwen4ExpVerifyPhase.CLOSED:
            raise Qwen4ExpSemanticCacheStateError("semantic cache is closed")

    def _poison(self, message: str, error: BaseException) -> None:
        detail = f"{message}: {type(error).__name__}: {error}"
        self._phase = Qwen4ExpVerifyPhase.POISONED
        self._poison_reason = detail
        raise Qwen4ExpSemanticCachePoisonedError(detail) from error


__all__ = [
    "Qwen4ExpSemanticCacheAdapter",
    "Qwen4ExpSemanticCacheBinding",
    "Qwen4ExpSemanticCacheError",
    "Qwen4ExpSemanticCacheIdentityError",
    "Qwen4ExpSemanticCacheKernelPort",
    "Qwen4ExpSemanticCachePoisonedError",
    "Qwen4ExpSemanticCacheStateError",
    "Qwen4ExpVerifyDisposition",
    "Qwen4ExpVerifyLease",
    "Qwen4ExpVerifyPhase",
    "Qwen4ExpVerifyReceipt",
]
