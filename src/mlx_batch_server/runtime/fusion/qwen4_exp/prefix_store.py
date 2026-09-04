# SPDX-License-Identifier: Apache-2.0
"""Whole-boundary prefix index for Qwen4Exp semantic-cache snapshots.

This is the tensor-free baseline for reusable Qwen4Exp prompt state.  It borrows
oMLX's namespace, parent-hash, lease/refcount, hot-LRU, and optional SSD
ownership concepts without claiming arbitrary per-layer KV slicing.  Each
payload is one opaque owner-thread snapshot that preserves QSA, GDN, and PLE
state atomically at a complete token-block boundary.

The index never serializes or mutates a payload.  A deployment-owned SSD port
may persist the immutable checkpoint record, including the opaque payload.
"""

from __future__ import annotations

import hashlib
import struct
import threading
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable

TEXT_CONTEXT_FINGERPRINT = ""
_FORMAT_DOMAIN = b"mlx-batch-qwen4exp-whole-boundary-v1\x00"
_MAX_TOKEN_ID = (1 << 64) - 1


class Qwen4ExpPrefixStoreError(RuntimeError):
    """Base failure for the whole-boundary prefix-store contract."""


class Qwen4ExpPrefixStoreIdentityError(Qwen4ExpPrefixStoreError):
    """A namespace, context, lease, chain, or checkpoint did not match."""


class Qwen4ExpPrefixStoreStateError(Qwen4ExpPrefixStoreError):
    """A prefix-store operation was attempted in an invalid state."""


class Qwen4ExpPrefixStoreCapacityError(Qwen4ExpPrefixStoreStateError):
    """A bounded store cannot admit an entry or another active lease."""


class Qwen4ExpPrefixLookupSource(StrEnum):
    HOT = "hot"
    SSD = "ssd"
    MISS = "miss"


class Qwen4ExpPrefixReleaseReason(StrEnum):
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ABORTED = "aborted"
    FAILED = "failed"
    REJECTED = "rejected"
    SHUTDOWN = "shutdown"


@dataclass(frozen=True, slots=True)
class Qwen4ExpPrefixBlockIdentity:
    """One deterministic full token block in a parent-hash chain."""

    token_start: int
    token_end: int
    parent_hash: str
    block_hash: str


@dataclass(frozen=True, slots=True)
class Qwen4ExpPrefixCheckpointIdentity:
    """Immutable identity of one complete semantic-cache boundary."""

    namespace_signature: str
    context_fingerprint: str
    token_count: int
    terminal_block_hash: str
    checkpoint_key: str


@dataclass(frozen=True, slots=True)
class Qwen4ExpWholeBoundaryCheckpoint:
    """Opaque QSA/GDN/PLE snapshot plus its immutable boundary identity."""

    identity: Qwen4ExpPrefixCheckpointIdentity
    payload: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.identity, Qwen4ExpPrefixCheckpointIdentity):
            raise TypeError("identity must be a Qwen4ExpPrefixCheckpointIdentity")
        if self.payload is None:
            raise ValueError("checkpoint payload must not be None")


@dataclass(frozen=True, slots=True)
class Qwen4ExpPendingBoundaryCheckpoint:
    """Owner-thread-created checkpoint that is not published until commit."""

    checkpoint: Qwen4ExpWholeBoundaryCheckpoint
    _issuer: object = field(repr=False, compare=False)
    _owner_thread_id: int = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class Qwen4ExpPrefixLeaseIdentity:
    request_id: str
    lease_id: str
    namespace_signature: str
    context_fingerprint: str
    generation: int


@dataclass(frozen=True, slots=True)
class Qwen4ExpPrefixLeaseReceipt:
    lease: Qwen4ExpPrefixLeaseIdentity
    already_open: bool


@dataclass(frozen=True, slots=True)
class Qwen4ExpPrefixLookupReceipt:
    request_id: str
    lease_id: str
    namespace_signature: str
    context_fingerprint: str
    source: Qwen4ExpPrefixLookupSource
    matched_tokens: int
    checkpoint_key: str | None
    checkpoint: Qwen4ExpWholeBoundaryCheckpoint | None = field(
        repr=False,
        compare=False,
    )
    refcount: int = 0

    @property
    def hit(self) -> bool:
        return self.source is not Qwen4ExpPrefixLookupSource.MISS


@dataclass(frozen=True, slots=True)
class Qwen4ExpPrefixCommitReceipt:
    request_id: str
    lease_id: str
    namespace_signature: str
    context_fingerprint: str
    checkpoint_key: str
    token_count: int
    published: bool
    already_published: bool
    persisted: bool
    evicted_checkpoint_keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Qwen4ExpPrefixReleaseReceipt:
    request_id: str
    lease_id: str
    namespace_signature: str
    context_fingerprint: str
    reason: Qwen4ExpPrefixReleaseReason
    released_references: int
    already_released: bool


@dataclass(frozen=True, slots=True)
class Qwen4ExpPrefixInvalidationReceipt:
    previous_signature: str | None
    namespace_signature: str
    invalidated_hot_entries: int
    invalidated_active_leases: int
    invalidated_persistent_entries: int
    generation: int


@dataclass(frozen=True, slots=True)
class Qwen4ExpPrefixStoreStats:
    namespace_signature: str
    generation: int
    hot_entries: int
    hot_tokens: int
    active_leases: int
    retained_references: int
    hot_hits: int
    ssd_hits: int
    misses: int
    commits: int
    namespace_invalidations: int
    invalidated_entries: int
    evictions: int


@runtime_checkable
class Qwen4ExpWholeBoundarySSDPort(Protocol):
    """Injected persistence for immutable whole-boundary checkpoints.

    The port owns serialization and storage location.  ``bind_namespace`` must
    invalidate metadata that is incompatible with the exact supplied signature
    and return the number of invalidated persistent entries.
    """

    def bind_namespace(self, namespace_signature: str) -> int: ...

    def load(
        self,
        checkpoint_key: str,
    ) -> Qwen4ExpWholeBoundaryCheckpoint | None: ...

    def store(self, checkpoint: Qwen4ExpWholeBoundaryCheckpoint) -> None: ...

    def delete(self, checkpoint_key: str) -> bool: ...


@dataclass(slots=True)
class _ActiveLease:
    identity: Qwen4ExpPrefixLeaseIdentity
    held_checkpoint_key: str | None = None


class Qwen4ExpWholeBoundaryPrefixStore:
    """Bounded prefix index for atomic Qwen4Exp semantic snapshots.

    All public operations are owner-thread only.  This keeps opaque tensor
    snapshots under the same ownership rule as the Qwen4Exp execution runtime;
    the index itself performs only hashing and metadata bookkeeping.
    """

    def __init__(
        self,
        *,
        namespace_signature: str,
        block_size_tokens: int,
        max_hot_entries: int,
        max_hot_tokens: int,
        max_active_leases: int,
        max_released_leases: int | None = None,
        ssd: Qwen4ExpWholeBoundarySSDPort | None = None,
    ) -> None:
        self._validate_signature(namespace_signature)
        self._require_positive_integer(block_size_tokens, "block_size_tokens")
        self._require_positive_integer(max_hot_entries, "max_hot_entries")
        self._require_positive_integer(max_hot_tokens, "max_hot_tokens")
        self._require_positive_integer(max_active_leases, "max_active_leases")
        if max_released_leases is None:
            max_released_leases = max_active_leases * 2
        self._require_positive_integer(max_released_leases, "max_released_leases")
        if ssd is not None and not isinstance(ssd, Qwen4ExpWholeBoundarySSDPort):
            raise TypeError("ssd must implement Qwen4ExpWholeBoundarySSDPort")

        self._owner_thread_id = threading.get_ident()
        self._issuer = object()
        self._namespace_signature: str | None = None
        self._generation = 0
        self._block_size_tokens = block_size_tokens
        self._max_hot_entries = max_hot_entries
        self._max_hot_tokens = max_hot_tokens
        self._max_active_leases = max_active_leases
        self._max_released_leases = max_released_leases
        self._ssd = ssd
        self._hot: OrderedDict[str, Qwen4ExpWholeBoundaryCheckpoint] = OrderedDict()
        self._hot_tokens = 0
        self._refcounts: dict[str, int] = {}
        self._active_by_id: dict[str, _ActiveLease] = {}
        self._active_by_request: dict[str, str] = {}
        self._released: OrderedDict[
            str,
            tuple[Qwen4ExpPrefixLeaseIdentity, Qwen4ExpPrefixReleaseReceipt],
        ] = OrderedDict()
        self._next_lease_id = 1
        self._hot_hits = 0
        self._ssd_hits = 0
        self._misses = 0
        self._commits = 0
        self._namespace_invalidations = 0
        self._invalidated_entries = 0
        self._evictions = 0
        self._binding_receipt = self.bind_namespace(namespace_signature)

    @property
    def namespace_signature(self) -> str:
        signature = self._namespace_signature
        if signature is None:
            raise Qwen4ExpPrefixStoreStateError("prefix store is not bound")
        return signature

    @property
    def block_size_tokens(self) -> int:
        return self._block_size_tokens

    @property
    def binding_receipt(self) -> Qwen4ExpPrefixInvalidationReceipt:
        return self._binding_receipt

    def bind_namespace(
        self,
        namespace_signature: str,
    ) -> Qwen4ExpPrefixInvalidationReceipt:
        """Bind one exact namespace and stale every incompatible identity."""

        self._require_owner_thread()
        self._validate_signature(namespace_signature)
        previous = self._namespace_signature
        if previous == namespace_signature:
            return Qwen4ExpPrefixInvalidationReceipt(
                previous_signature=previous,
                namespace_signature=namespace_signature,
                invalidated_hot_entries=0,
                invalidated_active_leases=0,
                invalidated_persistent_entries=0,
                generation=self._generation,
            )

        persistent = 0
        if self._ssd is not None:
            persistent = self._ssd.bind_namespace(namespace_signature)
            if type(persistent) is not int or persistent < 0:
                raise Qwen4ExpPrefixStoreStateError(
                    "SSD bind_namespace must return a non-negative integer"
                )

        hot = len(self._hot)
        active = len(self._active_by_id)
        self._hot.clear()
        self._hot_tokens = 0
        self._refcounts.clear()
        self._active_by_id.clear()
        self._active_by_request.clear()
        self._released.clear()
        self._namespace_signature = namespace_signature
        self._generation += 1
        if previous is not None:
            self._namespace_invalidations += 1
        self._invalidated_entries += hot + persistent
        receipt = Qwen4ExpPrefixInvalidationReceipt(
            previous_signature=previous,
            namespace_signature=namespace_signature,
            invalidated_hot_entries=hot,
            invalidated_active_leases=active,
            invalidated_persistent_entries=persistent,
            generation=self._generation,
        )
        self._binding_receipt = receipt
        return receipt

    def token_chain(
        self,
        tokens: Sequence[int],
        *,
        context_fingerprint: str,
    ) -> tuple[Qwen4ExpPrefixBlockIdentity, ...]:
        """Return deterministic complete blocks for one exact context domain."""

        self._require_owner_thread()
        fingerprint = self._validate_context_fingerprint(context_fingerprint)
        token_ids = self._validate_tokens(tokens)
        parent = self._genesis_hash(fingerprint)
        blocks: list[Qwen4ExpPrefixBlockIdentity] = []
        complete = len(token_ids) - (len(token_ids) % self._block_size_tokens)
        for start in range(0, complete, self._block_size_tokens):
            end = start + self._block_size_tokens
            block_hash = self._next_block_hash(parent, token_ids[start:end])
            blocks.append(
                Qwen4ExpPrefixBlockIdentity(
                    token_start=start,
                    token_end=end,
                    parent_hash=parent,
                    block_hash=block_hash,
                )
            )
            parent = block_hash
        return tuple(blocks)

    def begin_request(
        self,
        request_id: str,
        *,
        context_fingerprint: str,
    ) -> Qwen4ExpPrefixLeaseReceipt:
        self._require_owner_thread()
        request_id = self._validate_request_id(request_id)
        fingerprint = self._validate_context_fingerprint(context_fingerprint)
        existing_id = self._active_by_request.get(request_id)
        if existing_id is not None:
            existing = self._active_by_id[existing_id].identity
            if existing.context_fingerprint != fingerprint:
                raise Qwen4ExpPrefixStoreIdentityError(
                    "active request belongs to a different context fingerprint"
                )
            return Qwen4ExpPrefixLeaseReceipt(existing, already_open=True)
        if len(self._active_by_id) >= self._max_active_leases:
            raise Qwen4ExpPrefixStoreCapacityError(
                "maximum active prefix-store leases reached"
            )

        lease_id = f"prefix-{self._generation}-{self._next_lease_id}"
        self._next_lease_id += 1
        identity = Qwen4ExpPrefixLeaseIdentity(
            request_id=request_id,
            lease_id=lease_id,
            namespace_signature=self.namespace_signature,
            context_fingerprint=fingerprint,
            generation=self._generation,
        )
        self._active_by_id[lease_id] = _ActiveLease(identity)
        self._active_by_request[request_id] = lease_id
        return Qwen4ExpPrefixLeaseReceipt(identity, already_open=False)

    def create_checkpoint(
        self,
        tokens: Sequence[int],
        *,
        context_fingerprint: str,
        payload: object,
    ) -> Qwen4ExpPendingBoundaryCheckpoint:
        """Seal an owner-thread snapshot without publishing it."""

        self._require_owner_thread()
        if payload is None:
            raise ValueError("checkpoint payload must not be None")
        fingerprint = self._validate_context_fingerprint(context_fingerprint)
        token_ids = self._validate_tokens(tokens)
        if not token_ids or len(token_ids) % self._block_size_tokens:
            raise ValueError("checkpoint tokens must end at a complete block boundary")
        identity = self._checkpoint_identity(token_ids, fingerprint)
        checkpoint = Qwen4ExpWholeBoundaryCheckpoint(identity, payload)
        return Qwen4ExpPendingBoundaryCheckpoint(
            checkpoint=checkpoint,
            _issuer=self._issuer,
            _owner_thread_id=self._owner_thread_id,
        )

    def lookup(
        self,
        lease: Qwen4ExpPrefixLeaseIdentity,
        tokens: Sequence[int],
        *,
        context_fingerprint: str,
    ) -> Qwen4ExpPrefixLookupReceipt:
        """Acquire the longest fully validated checkpoint in one context."""

        self._require_owner_thread()
        fingerprint = self._validate_context_fingerprint(context_fingerprint)
        active = self._require_active(lease, fingerprint)
        token_ids = self._validate_tokens(tokens)
        blocks = self.token_chain(token_ids, context_fingerprint=fingerprint)

        for block in reversed(blocks):
            identity = self._identity_for_block(block, fingerprint)
            checkpoint = self._hot.get(identity.checkpoint_key)
            if checkpoint is not None:
                if checkpoint.identity != identity:
                    self._discard_hot(identity.checkpoint_key, invalidated=True)
                else:
                    self._hot.move_to_end(identity.checkpoint_key)
                    refcount = self._switch_reference(
                        active,
                        identity.checkpoint_key,
                    )
                    self._hot_hits += 1
                    return self._lookup_receipt(
                        active,
                        Qwen4ExpPrefixLookupSource.HOT,
                        checkpoint,
                        refcount,
                    )

            if self._ssd is None:
                continue
            restored = self._ssd.load(identity.checkpoint_key)
            if restored is None:
                continue
            if (
                not isinstance(restored, Qwen4ExpWholeBoundaryCheckpoint)
                or restored.identity != identity
            ):
                self._ssd.delete(identity.checkpoint_key)
                self._invalidated_entries += 1
                continue
            evictions = self._plan_evictions(restored.identity.token_count)
            if evictions is None:
                continue
            self._apply_evictions(evictions)
            self._insert_hot(restored)
            refcount = self._switch_reference(active, identity.checkpoint_key)
            self._ssd_hits += 1
            return self._lookup_receipt(
                active,
                Qwen4ExpPrefixLookupSource.SSD,
                restored,
                refcount,
            )

        self._switch_reference(active, None)
        self._misses += 1
        return Qwen4ExpPrefixLookupReceipt(
            request_id=lease.request_id,
            lease_id=lease.lease_id,
            namespace_signature=self.namespace_signature,
            context_fingerprint=fingerprint,
            source=Qwen4ExpPrefixLookupSource.MISS,
            matched_tokens=0,
            checkpoint_key=None,
            checkpoint=None,
            refcount=0,
        )

    def detach_lookup(
        self,
        lease: Qwen4ExpPrefixLeaseIdentity,
        *,
        context_fingerprint: str,
    ) -> bool:
        """Release a copied lookup payload while keeping the request lease open."""

        self._require_owner_thread()
        fingerprint = self._validate_context_fingerprint(context_fingerprint)
        active = self._require_active(lease, fingerprint)
        held = active.held_checkpoint_key is not None
        self._switch_reference(active, None)
        return held

    def commit(
        self,
        lease: Qwen4ExpPrefixLeaseIdentity,
        tokens: Sequence[int],
        pending: Qwen4ExpPendingBoundaryCheckpoint,
        *,
        context_fingerprint: str,
    ) -> Qwen4ExpPrefixCommitReceipt:
        """Atomically publish one explicit, complete owner checkpoint."""

        self._require_owner_thread()
        fingerprint = self._validate_context_fingerprint(context_fingerprint)
        self._require_active(lease, fingerprint)
        if not isinstance(pending, Qwen4ExpPendingBoundaryCheckpoint):
            raise TypeError("pending must be a Qwen4ExpPendingBoundaryCheckpoint")
        if (
            pending._issuer is not self._issuer
            or pending._owner_thread_id != self._owner_thread_id
        ):
            raise Qwen4ExpPrefixStoreIdentityError(
                "checkpoint was not created by this store owner"
            )
        token_ids = self._validate_tokens(tokens)
        if not token_ids or len(token_ids) % self._block_size_tokens:
            raise ValueError("commit tokens must end at a complete block boundary")
        expected = self._checkpoint_identity(token_ids, fingerprint)
        checkpoint = pending.checkpoint
        if checkpoint.identity != expected:
            raise Qwen4ExpPrefixStoreIdentityError(
                "checkpoint identity does not match commit tokens and context"
            )

        key = expected.checkpoint_key
        existing = self._hot.get(key)
        if existing is not None:
            if existing.identity != expected:
                raise Qwen4ExpPrefixStoreIdentityError(
                    "hot checkpoint key has a conflicting identity"
                )
            self._hot.move_to_end(key)
            return Qwen4ExpPrefixCommitReceipt(
                request_id=lease.request_id,
                lease_id=lease.lease_id,
                namespace_signature=self.namespace_signature,
                context_fingerprint=fingerprint,
                checkpoint_key=key,
                token_count=expected.token_count,
                published=False,
                already_published=True,
                persisted=self._ssd is not None,
                evicted_checkpoint_keys=(),
            )

        evictions = self._plan_evictions(expected.token_count)
        if evictions is None:
            raise Qwen4ExpPrefixStoreCapacityError(
                "hot checkpoint budget is pinned or checkpoint is too large"
            )
        if self._ssd is not None:
            self._ssd.store(checkpoint)
        self._apply_evictions(evictions)
        self._insert_hot(checkpoint)
        self._commits += 1
        return Qwen4ExpPrefixCommitReceipt(
            request_id=lease.request_id,
            lease_id=lease.lease_id,
            namespace_signature=self.namespace_signature,
            context_fingerprint=fingerprint,
            checkpoint_key=key,
            token_count=expected.token_count,
            published=True,
            already_published=False,
            persisted=self._ssd is not None,
            evicted_checkpoint_keys=tuple(evictions),
        )

    def release_request(
        self,
        lease: Qwen4ExpPrefixLeaseIdentity,
        *,
        context_fingerprint: str,
        reason: Qwen4ExpPrefixReleaseReason,
    ) -> Qwen4ExpPrefixReleaseReceipt:
        """Release one request reference; never publish pending state."""

        self._require_owner_thread()
        fingerprint = self._validate_context_fingerprint(context_fingerprint)
        if not isinstance(reason, Qwen4ExpPrefixReleaseReason):
            raise TypeError("reason must be a Qwen4ExpPrefixReleaseReason")
        released = self._released.get(getattr(lease, "lease_id", ""))
        if released is not None:
            released_identity, receipt = released
            if released_identity != lease or lease.context_fingerprint != fingerprint:
                raise Qwen4ExpPrefixStoreIdentityError(
                    "stale or foreign prefix-store lease"
                )
            return Qwen4ExpPrefixReleaseReceipt(
                request_id=receipt.request_id,
                lease_id=receipt.lease_id,
                namespace_signature=receipt.namespace_signature,
                context_fingerprint=receipt.context_fingerprint,
                reason=receipt.reason,
                released_references=0,
                already_released=True,
            )

        active = self._require_active(lease, fingerprint)
        released_references = 0
        if active.held_checkpoint_key is not None:
            self._decrement_reference(active.held_checkpoint_key)
            released_references = 1
        del self._active_by_id[lease.lease_id]
        del self._active_by_request[lease.request_id]
        receipt = Qwen4ExpPrefixReleaseReceipt(
            request_id=lease.request_id,
            lease_id=lease.lease_id,
            namespace_signature=self.namespace_signature,
            context_fingerprint=fingerprint,
            reason=reason,
            released_references=released_references,
            already_released=False,
        )
        self._released[lease.lease_id] = (lease, receipt)
        self._released.move_to_end(lease.lease_id)
        while len(self._released) > self._max_released_leases:
            self._released.popitem(last=False)
        return receipt

    def stats(self) -> Qwen4ExpPrefixStoreStats:
        self._require_owner_thread()
        return Qwen4ExpPrefixStoreStats(
            namespace_signature=self.namespace_signature,
            generation=self._generation,
            hot_entries=len(self._hot),
            hot_tokens=self._hot_tokens,
            active_leases=len(self._active_by_id),
            retained_references=sum(self._refcounts.values()),
            hot_hits=self._hot_hits,
            ssd_hits=self._ssd_hits,
            misses=self._misses,
            commits=self._commits,
            namespace_invalidations=self._namespace_invalidations,
            invalidated_entries=self._invalidated_entries,
            evictions=self._evictions,
        )

    def _checkpoint_identity(
        self,
        token_ids: tuple[int, ...],
        context_fingerprint: str,
    ) -> Qwen4ExpPrefixCheckpointIdentity:
        blocks = self.token_chain(
            token_ids,
            context_fingerprint=context_fingerprint,
        )
        if not blocks or blocks[-1].token_end != len(token_ids):
            raise ValueError("checkpoint identity requires a complete token boundary")
        return self._identity_for_block(blocks[-1], context_fingerprint)

    def _identity_for_block(
        self,
        block: Qwen4ExpPrefixBlockIdentity,
        context_fingerprint: str,
    ) -> Qwen4ExpPrefixCheckpointIdentity:
        key_digest = hashlib.sha256()
        key_digest.update(_FORMAT_DOMAIN)
        self._update_text(key_digest, self.namespace_signature)
        self._update_text(key_digest, context_fingerprint)
        key_digest.update(struct.pack(">Q", block.token_end))
        key_digest.update(bytes.fromhex(block.block_hash))
        return Qwen4ExpPrefixCheckpointIdentity(
            namespace_signature=self.namespace_signature,
            context_fingerprint=context_fingerprint,
            token_count=block.token_end,
            terminal_block_hash=block.block_hash,
            checkpoint_key=f"qwen4exp-wb-v1:{key_digest.hexdigest()}",
        )

    def _genesis_hash(self, context_fingerprint: str) -> str:
        digest = hashlib.sha256()
        digest.update(_FORMAT_DOMAIN)
        self._update_text(digest, self.namespace_signature)
        self._update_text(digest, context_fingerprint)
        return digest.hexdigest()

    @staticmethod
    def _next_block_hash(parent_hash: str, tokens: Sequence[int]) -> str:
        digest = hashlib.sha256()
        digest.update(_FORMAT_DOMAIN)
        digest.update(b"block\x00")
        digest.update(bytes.fromhex(parent_hash))
        digest.update(struct.pack(">Q", len(tokens)))
        for token in tokens:
            digest.update(struct.pack(">Q", token))
        return digest.hexdigest()

    @staticmethod
    def _update_text(digest: object, value: str) -> None:
        encoded = value.encode("utf-8")
        digest.update(struct.pack(">Q", len(encoded)))  # type: ignore[attr-defined]
        digest.update(encoded)  # type: ignore[attr-defined]

    def _lookup_receipt(
        self,
        active: _ActiveLease,
        source: Qwen4ExpPrefixLookupSource,
        checkpoint: Qwen4ExpWholeBoundaryCheckpoint,
        refcount: int,
    ) -> Qwen4ExpPrefixLookupReceipt:
        identity = active.identity
        return Qwen4ExpPrefixLookupReceipt(
            request_id=identity.request_id,
            lease_id=identity.lease_id,
            namespace_signature=identity.namespace_signature,
            context_fingerprint=identity.context_fingerprint,
            source=source,
            matched_tokens=checkpoint.identity.token_count,
            checkpoint_key=checkpoint.identity.checkpoint_key,
            checkpoint=checkpoint,
            refcount=refcount,
        )

    def _require_active(
        self,
        lease: Qwen4ExpPrefixLeaseIdentity,
        context_fingerprint: str,
    ) -> _ActiveLease:
        if not isinstance(lease, Qwen4ExpPrefixLeaseIdentity):
            raise TypeError("lease must be a Qwen4ExpPrefixLeaseIdentity")
        if (
            lease.namespace_signature != self.namespace_signature
            or lease.generation != self._generation
            or lease.context_fingerprint != context_fingerprint
        ):
            raise Qwen4ExpPrefixStoreIdentityError(
                "stale or foreign prefix-store lease"
            )
        active = self._active_by_id.get(lease.lease_id)
        if active is None or active.identity != lease:
            raise Qwen4ExpPrefixStoreIdentityError(
                "stale or foreign prefix-store lease"
            )
        return active

    def _switch_reference(
        self,
        active: _ActiveLease,
        checkpoint_key: str | None,
    ) -> int:
        current = active.held_checkpoint_key
        if current == checkpoint_key:
            return self._refcounts.get(checkpoint_key, 0) if checkpoint_key else 0
        if current is not None:
            self._decrement_reference(current)
        active.held_checkpoint_key = checkpoint_key
        if checkpoint_key is None:
            return 0
        refcount = self._refcounts.get(checkpoint_key, 0) + 1
        self._refcounts[checkpoint_key] = refcount
        return refcount

    def _decrement_reference(self, checkpoint_key: str) -> None:
        current = self._refcounts.get(checkpoint_key, 0)
        if current <= 0:
            raise Qwen4ExpPrefixStoreStateError(
                "checkpoint reference count is inconsistent"
            )
        if current == 1:
            del self._refcounts[checkpoint_key]
        else:
            self._refcounts[checkpoint_key] = current - 1

    def _plan_evictions(self, incoming_tokens: int) -> list[str] | None:
        if incoming_tokens > self._max_hot_tokens:
            return None
        entries = len(self._hot) + 1
        tokens = self._hot_tokens + incoming_tokens
        evictions: list[str] = []
        for key, checkpoint in self._hot.items():
            if entries <= self._max_hot_entries and tokens <= self._max_hot_tokens:
                break
            if self._refcounts.get(key, 0) > 0:
                continue
            evictions.append(key)
            entries -= 1
            tokens -= checkpoint.identity.token_count
        if entries > self._max_hot_entries or tokens > self._max_hot_tokens:
            return None
        return evictions

    def _apply_evictions(self, checkpoint_keys: Sequence[str]) -> None:
        for key in checkpoint_keys:
            self._discard_hot(key, invalidated=False)
            self._evictions += 1

    def _insert_hot(self, checkpoint: Qwen4ExpWholeBoundaryCheckpoint) -> None:
        key = checkpoint.identity.checkpoint_key
        if key in self._hot:
            raise Qwen4ExpPrefixStoreStateError("checkpoint is already hot")
        self._hot[key] = checkpoint
        self._hot_tokens += checkpoint.identity.token_count

    def _discard_hot(self, checkpoint_key: str, *, invalidated: bool) -> None:
        if self._refcounts.get(checkpoint_key, 0) > 0:
            raise Qwen4ExpPrefixStoreStateError(
                "cannot discard a referenced checkpoint"
            )
        checkpoint = self._hot.pop(checkpoint_key, None)
        if checkpoint is not None:
            self._hot_tokens -= checkpoint.identity.token_count
            if invalidated:
                self._invalidated_entries += 1

    def _require_owner_thread(self) -> None:
        if threading.get_ident() != self._owner_thread_id:
            raise Qwen4ExpPrefixStoreStateError(
                "prefix-store operations require the inference owner thread"
            )

    @staticmethod
    def _validate_signature(namespace_signature: str) -> None:
        if not isinstance(namespace_signature, str):
            raise TypeError("namespace_signature must be a string")
        if not namespace_signature:
            raise ValueError("namespace_signature must not be empty")

    @staticmethod
    def _validate_context_fingerprint(context_fingerprint: str) -> str:
        if not isinstance(context_fingerprint, str):
            raise TypeError("context_fingerprint must be a string")
        return context_fingerprint

    @staticmethod
    def _validate_request_id(request_id: str) -> str:
        if not isinstance(request_id, str):
            raise TypeError("request_id must be a string")
        if not request_id:
            raise ValueError("request_id must not be empty")
        return request_id

    @staticmethod
    def _validate_tokens(tokens: Sequence[int]) -> tuple[int, ...]:
        if isinstance(tokens, str | bytes | bytearray):
            raise TypeError("tokens must be a sequence of integers")
        token_ids = tuple(tokens)
        for token in token_ids:
            if type(token) is not int or not 0 <= token <= _MAX_TOKEN_ID:
                raise ValueError("token IDs must be unsigned 64-bit integers")
        return token_ids

    @staticmethod
    def _require_positive_integer(value: int, name: str) -> None:
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")


__all__ = [
    "TEXT_CONTEXT_FINGERPRINT",
    "Qwen4ExpPendingBoundaryCheckpoint",
    "Qwen4ExpPrefixBlockIdentity",
    "Qwen4ExpPrefixCheckpointIdentity",
    "Qwen4ExpPrefixCommitReceipt",
    "Qwen4ExpPrefixInvalidationReceipt",
    "Qwen4ExpPrefixLeaseIdentity",
    "Qwen4ExpPrefixLeaseReceipt",
    "Qwen4ExpPrefixLookupReceipt",
    "Qwen4ExpPrefixLookupSource",
    "Qwen4ExpPrefixReleaseReason",
    "Qwen4ExpPrefixReleaseReceipt",
    "Qwen4ExpPrefixStoreCapacityError",
    "Qwen4ExpPrefixStoreError",
    "Qwen4ExpPrefixStoreIdentityError",
    "Qwen4ExpPrefixStoreStateError",
    "Qwen4ExpPrefixStoreStats",
    "Qwen4ExpWholeBoundaryCheckpoint",
    "Qwen4ExpWholeBoundaryPrefixStore",
    "Qwen4ExpWholeBoundarySSDPort",
]
