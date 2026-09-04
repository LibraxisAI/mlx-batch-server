# SPDX-License-Identifier: Apache-2.0
"""Request lease and cleanup coordinator for the fused cache tiers.

Adapted from the scheduler/cache ownership split in oMLX ``scheduler.py``,
``paged_cache.py``, ``prefix_cache.py``, and ``paged_ssd_cache.py`` at commit
``e467261edc786efd33b1e9023d5c4a827f8aa1c1``. This module never owns cache
tensors. Tier adapters execute tensor-affecting operations on their scheduler
owner thread.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field, replace
from itertools import count
from threading import RLock
from typing import TYPE_CHECKING

from .contracts import (
    CacheBindingReceipt,
    CacheCleanupReceipt,
    CacheInvalidationReceipt,
    CacheLeaseState,
    CacheNamespace,
    CacheReleaseReason,
    CacheTier,
    PagedCachePort,
    PrefixCachePort,
    SSDCachePort,
)

if TYPE_CHECKING:
    from ....cache.contracts import CacheIdentity


@dataclass(slots=True)
class _CleanupProgress:
    """Successful cleanup work retained across retryable failures."""

    reason: CacheReleaseReason | None = None
    attempted_tiers: set[CacheTier] = field(default_factory=set)
    released_tiers: set[CacheTier] = field(default_factory=set)
    released_references: int = 0
    pending_writes_quiesced: bool = False


class FusionCacheLease:
    """One request's idempotent claim on a single cache namespace."""

    def __init__(
        self,
        owner: FusionCacheCoordinator,
        request_id: str,
        lease_id: str,
    ) -> None:
        self._owner = owner
        self._request_id = request_id
        self._lease_id = lease_id
        self._state = CacheLeaseState.OPENING

    @property
    def identity(self) -> CacheIdentity:
        return self._owner.namespace.identity

    @property
    def request_id(self) -> str:
        return self._request_id

    @property
    def lease_id(self) -> str:
        return self._lease_id

    @property
    def state(self) -> CacheLeaseState:
        return self._state

    def release(self) -> None:
        self._owner.complete(self._request_id, lease_id=self._lease_id)

    def abort(
        self,
        reason: CacheReleaseReason = CacheReleaseReason.ABORTED,
    ) -> CacheCleanupReceipt:
        if reason is CacheReleaseReason.COMPLETED:
            raise ValueError("abort requires a non-completed release reason")
        return self._owner.cleanup(
            self._request_id,
            reason=reason,
            lease_id=self._lease_id,
        )


class FusionCacheCoordinator:
    """Couple three cache tiers without becoming a second tensor owner."""

    def __init__(
        self,
        *,
        namespace: CacheNamespace,
        paged: PagedCachePort,
        prefix: PrefixCachePort,
        ssd: SSDCachePort,
        max_completed_leases: int = 4096,
    ) -> None:
        if max_completed_leases <= 0:
            raise ValueError("max_completed_leases must be positive")
        self.namespace = namespace
        self._paged = paged
        self._prefix = prefix
        self._ssd = ssd
        self._max_completed_leases = max_completed_leases
        self._active: dict[str, FusionCacheLease] = {}
        self._completed: OrderedDict[str, CacheCleanupReceipt] = OrderedDict()
        self._cleanup_progress: dict[str, _CleanupProgress] = {}
        self._lease_sequence = count(1)
        self._lock = RLock()
        self._operation_lock = RLock()
        self._binding: CacheBindingReceipt | None = None
        self._binding_progress: dict[CacheTier, CacheInvalidationReceipt] = {}

    @property
    def binding(self) -> CacheBindingReceipt | None:
        return self._binding

    def activate(self) -> CacheBindingReceipt:
        """Bind persistent identity before in-memory indexes can accept work."""
        with self._operation_lock:
            return self._activate()

    def _activate(self) -> CacheBindingReceipt:
        with self._lock:
            if self._active:
                raise RuntimeError("cannot rebind cache namespace with active leases")
            if self._binding is not None:
                return self._binding

        for tier, bind in (
            (CacheTier.SSD, self._ssd.bind_namespace),
            (CacheTier.PAGED, self._paged.bind_namespace),
            (CacheTier.PREFIX, self._prefix.bind_namespace),
        ):
            if tier in self._binding_progress:
                continue
            invalidated_entries = bind(self.namespace)
            with self._lock:
                self._binding_progress[tier] = CacheInvalidationReceipt(
                    tier,
                    self.namespace.signature,
                    invalidated_entries,
                )

        with self._lock:
            invalidations = tuple(
                self._binding_progress[tier]
                for tier in (CacheTier.SSD, CacheTier.PAGED, CacheTier.PREFIX)
            )
            receipt = CacheBindingReceipt(self.namespace.signature, invalidations)
            self._binding = receipt
            return receipt

    def acquire(self, request_id: str) -> FusionCacheLease:
        with self._operation_lock:
            return self._acquire(request_id)

    def _acquire(self, request_id: str) -> FusionCacheLease:
        if not request_id:
            raise ValueError("request_id must be non-empty")
        with self._lock:
            if self._binding is None:
                raise RuntimeError("cache namespace must be activated before acquire")
            if request_id in self._active or request_id in self._completed:
                raise RuntimeError(f"cache lease already exists for {request_id}")
            lease_id = f"cache_{next(self._lease_sequence):016x}"
            lease = FusionCacheLease(self, request_id, lease_id)
            self._active[request_id] = lease

        attempted: list[CacheTier] = []
        try:
            attempted.append(CacheTier.PAGED)
            self._paged.open_request(request_id, lease_id)
            attempted.append(CacheTier.PREFIX)
            self._prefix.open_request(request_id, lease_id)
            attempted.append(CacheTier.SSD)
            self._ssd.open_request(request_id, lease_id)
        except Exception:
            progress, errors = self._rollback_open(request_id, attempted)
            with self._lock:
                if errors:
                    lease._state = CacheLeaseState.CLEANUP_FAILED
                    self._cleanup_progress[request_id] = progress
                else:
                    self._active.pop(request_id, None)
            raise

        with self._lock:
            self._cleanup_progress[request_id] = _CleanupProgress(
                attempted_tiers=set(attempted)
            )
            lease._state = CacheLeaseState.ACTIVE
        return lease

    def complete(
        self,
        request_id: str,
        *,
        lease_id: str | None = None,
    ) -> CacheCleanupReceipt:
        return self.cleanup(
            request_id,
            reason=CacheReleaseReason.COMPLETED,
            lease_id=lease_id,
        )

    def cleanup(
        self,
        request_id: str,
        *,
        reason: CacheReleaseReason,
        lease_id: str | None = None,
    ) -> CacheCleanupReceipt:
        with self._operation_lock:
            return self._cleanup(request_id, reason=reason, lease_id=lease_id)

    def _cleanup(
        self,
        request_id: str,
        *,
        reason: CacheReleaseReason,
        lease_id: str | None,
    ) -> CacheCleanupReceipt:
        with self._lock:
            previous = self._completed.get(request_id)
            if previous is not None:
                self._validate_lease_id(previous.lease_id, lease_id)
                if previous.reason is not reason:
                    raise ValueError(
                        "cache release reason does not match the completed lease"
                    )
                self._completed.move_to_end(request_id)
                return replace(previous, already_released=True)
            lease = self._active.get(request_id)
            if lease is None:
                raise KeyError(request_id)
            self._validate_lease_id(lease.lease_id, lease_id)
            if lease._state is CacheLeaseState.RELEASING:
                raise RuntimeError(
                    f"cache cleanup already in progress for {request_id}"
                )
            progress = self._cleanup_progress.setdefault(
                request_id,
                _CleanupProgress(),
            )
            if progress.reason is not None and progress.reason is not reason:
                raise ValueError(
                    "cache release reason does not match the cleanup in progress"
                )
            progress.reason = reason
            lease._state = CacheLeaseState.RELEASING

        commit = reason is CacheReleaseReason.COMPLETED
        errors: list[str] = []

        if (
            CacheTier.SSD in progress.attempted_tiers
            and not progress.pending_writes_quiesced
        ):
            try:
                self._ssd.quiesce_request(request_id, commit=commit)
                progress.pending_writes_quiesced = True
            except Exception as error:
                lease._state = CacheLeaseState.CLEANUP_FAILED
                return self._cleanup_receipt(
                    lease,
                    progress,
                    reason=reason,
                    errors=(f"ssd.quiesce_request: {error}",),
                )

        for tier, cleanup in (
            (
                CacheTier.PAGED,
                lambda: self._paged.release_request(
                    request_id,
                    retain_reusable_blocks=commit,
                ),
            ),
            (
                CacheTier.PREFIX,
                lambda: self._prefix.clear_request_entry(
                    request_id,
                    retain_reusable_blocks=commit,
                ),
            ),
            (CacheTier.SSD, lambda: self._ssd.cleanup_request(request_id)),
        ):
            if tier not in progress.attempted_tiers:
                continue
            if tier in progress.released_tiers:
                continue
            try:
                progress.released_references += cleanup()
                progress.released_tiers.add(tier)
            except Exception as error:
                errors.append(f"{tier.value}.cleanup: {error}")
                break

        receipt = self._cleanup_receipt(
            lease,
            progress,
            reason=reason,
            errors=tuple(errors),
        )
        if errors:
            lease._state = CacheLeaseState.CLEANUP_FAILED
            return receipt

        lease._state = CacheLeaseState.RELEASED
        with self._lock:
            self._active.pop(request_id, None)
            self._cleanup_progress.pop(request_id, None)
            self._completed[request_id] = receipt
            self._completed.move_to_end(request_id)
            while len(self._completed) > self._max_completed_leases:
                self._completed.popitem(last=False)
        return receipt

    def _cleanup_receipt(
        self,
        lease: FusionCacheLease,
        progress: _CleanupProgress,
        *,
        reason: CacheReleaseReason,
        errors: tuple[str, ...],
    ) -> CacheCleanupReceipt:
        released_tiers = tuple(
            tier
            for tier in (CacheTier.PAGED, CacheTier.PREFIX, CacheTier.SSD)
            if tier in progress.released_tiers
        )
        return CacheCleanupReceipt(
            request_id=lease.request_id,
            lease_id=lease.lease_id,
            reason=reason,
            released_tiers=released_tiers,
            released_references=progress.released_references,
            pending_writes_quiesced=progress.pending_writes_quiesced,
            retained_reusable_blocks=reason is CacheReleaseReason.COMPLETED,
            errors=errors,
        )

    def _rollback_open(
        self,
        request_id: str,
        opened: list[CacheTier],
    ) -> tuple[_CleanupProgress, tuple[str, ...]]:
        progress = _CleanupProgress(attempted_tiers=set(opened))
        errors: list[str] = []
        if CacheTier.SSD in progress.attempted_tiers:
            try:
                self._ssd.quiesce_request(request_id, commit=False)
                progress.pending_writes_quiesced = True
            except Exception as error:
                errors.append(f"{CacheTier.SSD.value}.rollback: {error}")
                return progress, tuple(errors)

        for tier, cleanup in (
            (
                CacheTier.PAGED,
                lambda: self._paged.release_request(
                    request_id,
                    retain_reusable_blocks=False,
                ),
            ),
            (
                CacheTier.PREFIX,
                lambda: self._prefix.clear_request_entry(
                    request_id,
                    retain_reusable_blocks=False,
                ),
            ),
            (CacheTier.SSD, lambda: self._ssd.cleanup_request(request_id)),
        ):
            if tier not in progress.attempted_tiers:
                continue
            try:
                progress.released_references += cleanup()
                progress.released_tiers.add(tier)
            except Exception as error:
                # Preserve the original open failure and stop at the first
                # incomplete phase so a retry keeps cross-tier ordering exact.
                errors.append(f"{tier.value}.rollback: {error}")
                break
        return progress, tuple(errors)

    @staticmethod
    def _validate_lease_id(expected: str, supplied: str | None) -> None:
        if supplied is not None and supplied != expected:
            raise RuntimeError("stale cache lease cannot mutate a newer request lease")


__all__ = ["FusionCacheCoordinator", "FusionCacheLease"]
