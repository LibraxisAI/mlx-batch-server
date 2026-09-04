# SPDX-License-Identifier: Apache-2.0
"""Tensor-free contracts for the fused scheduler cache bundle.

Behavior is adapted from oMLX cache lifecycle contracts at commit
``e467261edc786efd33b1e9023d5c4a827f8aa1c1``. The target rewrite keeps
cache identity and request cleanup visible to the target-owned scheduler while
leaving all KV tensors inside the eventual donor-derived tier adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from ....cache.contracts import CacheBudget, CacheIdentity, CacheLease


class CacheTier(StrEnum):
    PAGED = "paged"
    PREFIX = "prefix"
    SSD = "ssd"


class CacheReleaseReason(StrEnum):
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ABORTED = "aborted"
    REJECTED = "rejected"
    FAILED = "failed"
    SHUTDOWN = "shutdown"


class CacheLeaseState(StrEnum):
    OPENING = "opening"
    ACTIVE = "active"
    RELEASING = "releasing"
    CLEANUP_FAILED = "cleanup_failed"
    RELEASED = "released"


@dataclass(frozen=True, slots=True)
class CacheLayout:
    """Every model/cache shape dimension that can make persisted state unsafe."""

    num_layers: int
    block_size_tokens: int
    layer_cache_types: tuple[str, ...]
    payload_layout: str = "embedded"
    format_version: int = 1
    adapter_fingerprint: str | None = None
    draft_model_id: str | None = None
    draft_model_revision: str | None = None
    mtp_layout: str | None = None
    turboquant_kv_bits: float | None = None
    cachelist_subtypes: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def __post_init__(self) -> None:
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.block_size_tokens <= 0:
            raise ValueError("block_size_tokens must be positive")
        if len(self.layer_cache_types) != self.num_layers:
            raise ValueError("layer_cache_types must have one entry per layer")
        if not self.payload_layout:
            raise ValueError("payload_layout must be non-empty")
        if self.format_version <= 0:
            raise ValueError("format_version must be positive")
        if self.turboquant_kv_bits is not None and self.turboquant_kv_bits <= 0:
            raise ValueError("turboquant_kv_bits must be positive")


@dataclass(frozen=True, slots=True)
class CacheNamespace:
    """Exact target identity shared by paged, prefix, and SSD tiers."""

    identity: CacheIdentity
    layout: CacheLayout
    signature: str


@dataclass(frozen=True, slots=True)
class CacheInvalidationReceipt:
    tier: CacheTier
    signature: str
    invalidated_entries: int


@dataclass(frozen=True, slots=True)
class CacheBindingReceipt:
    signature: str
    invalidations: tuple[CacheInvalidationReceipt, ...]


@dataclass(frozen=True, slots=True)
class CacheCleanupReceipt:
    request_id: str
    lease_id: str
    reason: CacheReleaseReason
    released_tiers: tuple[CacheTier, ...]
    released_references: int
    pending_writes_quiesced: bool
    retained_reusable_blocks: bool
    already_released: bool = False
    errors: tuple[str, ...] = ()

    @property
    def succeeded(self) -> bool:
        return not self.errors


@runtime_checkable
class PagedCachePort(Protocol):
    """Scheduler-thread adapter over request block tables and refcounts.

    Request cleanup must be idempotent and exception-atomic: when an operation
    raises it must not have released references that its return value would
    otherwise report. A coordinator retry resumes at that same tier.
    """

    def bind_namespace(self, namespace: CacheNamespace) -> int: ...

    def open_request(self, request_id: str, lease_id: str) -> None: ...

    def release_request(
        self,
        request_id: str,
        *,
        retain_reusable_blocks: bool,
    ) -> int: ...


@runtime_checkable
class PrefixCachePort(Protocol):
    """Adapter over prefix request bookkeeping and reusable hash indexes.

    ``retain_reusable_blocks=False`` discards request-local, uncommitted state;
    it does not invalidate independently committed prefixes in the namespace.
    Cleanup must be idempotent and exception-atomic: a raised operation must
    not have released unreported references.
    """

    def bind_namespace(self, namespace: CacheNamespace) -> int: ...

    def open_request(self, request_id: str, lease_id: str) -> None: ...

    def clear_request_entry(
        self,
        request_id: str,
        *,
        retain_reusable_blocks: bool,
    ) -> int: ...


@runtime_checkable
class SSDCachePort(Protocol):
    """Adapter over persistent indexes, pending writes, and request scratch.

    Quiescing must make all request-scoped persistence readers and writers stop
    using paged references before the coordinator releases those references.
    Quiesce and cleanup must both be idempotent and exception-atomic: a raised
    operation must not have completed unreported release work.
    """

    def bind_namespace(self, namespace: CacheNamespace) -> int: ...

    def open_request(self, request_id: str, lease_id: str) -> None: ...

    def quiesce_request(self, request_id: str, *, commit: bool) -> None: ...

    def cleanup_request(self, request_id: str) -> int: ...


__all__ = [
    "CacheBindingReceipt",
    "CacheBudget",
    "CacheCleanupReceipt",
    "CacheIdentity",
    "CacheInvalidationReceipt",
    "CacheLayout",
    "CacheLease",
    "CacheLeaseState",
    "CacheNamespace",
    "CacheReleaseReason",
    "CacheTier",
    "PagedCachePort",
    "PrefixCachePort",
    "SSDCachePort",
]
