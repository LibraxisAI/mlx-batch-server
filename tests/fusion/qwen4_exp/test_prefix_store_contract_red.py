"""RED contracts for the Qwen4Exp whole-boundary prefix store."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from mlx_batch_server.runtime.fusion.qwen4_exp.prefix_store import (
    TEXT_CONTEXT_FINGERPRINT,
    Qwen4ExpPrefixLookupSource,
    Qwen4ExpPrefixReleaseReason,
    Qwen4ExpPrefixStoreCapacityError,
    Qwen4ExpPrefixStoreIdentityError,
    Qwen4ExpWholeBoundaryCheckpoint,
    Qwen4ExpWholeBoundaryPrefixStore,
)

NAMESPACE_A = "qwen4exp:flash:revision-a:layout-v1"
NAMESPACE_B = "qwen4exp:flash:revision-b:layout-v1"
VISION_A = "media-bundle:sha256:aaaa"
VISION_B = "media-bundle:sha256:bbbb"


class MemorySSD:
    def __init__(self) -> None:
        self.namespace_signature: str | None = None
        self.entries: dict[str, Qwen4ExpWholeBoundaryCheckpoint] = {}
        self.binds: list[str] = []
        self.loads: list[str] = []
        self.stores: list[str] = []
        self.deletes: list[str] = []

    def bind_namespace(self, namespace_signature: str) -> int:
        self.binds.append(namespace_signature)
        if self.namespace_signature in (None, namespace_signature):
            self.namespace_signature = namespace_signature
            return 0
        invalidated = len(self.entries)
        self.entries.clear()
        self.namespace_signature = namespace_signature
        return invalidated

    def load(self, checkpoint_key: str) -> Qwen4ExpWholeBoundaryCheckpoint | None:
        self.loads.append(checkpoint_key)
        return self.entries.get(checkpoint_key)

    def store(self, checkpoint: Qwen4ExpWholeBoundaryCheckpoint) -> None:
        self.stores.append(checkpoint.identity.checkpoint_key)
        self.entries[checkpoint.identity.checkpoint_key] = checkpoint

    def delete(self, checkpoint_key: str) -> bool:
        self.deletes.append(checkpoint_key)
        return self.entries.pop(checkpoint_key, None) is not None


def _store(
    *,
    namespace: str = NAMESPACE_A,
    block_size: int = 2,
    max_entries: int = 4,
    max_tokens: int = 64,
    max_leases: int = 8,
    ssd: MemorySSD | None = None,
) -> Qwen4ExpWholeBoundaryPrefixStore:
    return Qwen4ExpWholeBoundaryPrefixStore(
        namespace_signature=namespace,
        block_size_tokens=block_size,
        max_hot_entries=max_entries,
        max_hot_tokens=max_tokens,
        max_active_leases=max_leases,
        ssd=ssd,
    )


def _lease(
    store: Qwen4ExpWholeBoundaryPrefixStore,
    request_id: str,
    fingerprint: str = TEXT_CONTEXT_FINGERPRINT,
):
    return store.begin_request(
        request_id,
        context_fingerprint=fingerprint,
    ).lease


def _commit(
    store: Qwen4ExpWholeBoundaryPrefixStore,
    request_id: str,
    tokens: list[int],
    payload: object,
    fingerprint: str = TEXT_CONTEXT_FINGERPRINT,
):
    lease = _lease(store, request_id, fingerprint)
    pending = store.create_checkpoint(
        tokens,
        context_fingerprint=fingerprint,
        payload=payload,
    )
    receipt = store.commit(
        lease,
        tokens,
        pending,
        context_fingerprint=fingerprint,
    )
    store.release_request(
        lease,
        context_fingerprint=fingerprint,
        reason=Qwen4ExpPrefixReleaseReason.COMPLETED,
    )
    return pending.checkpoint, receipt


def test_token_blocks_are_deterministic_parent_hash_chains() -> None:
    store = _store(block_size=2)
    tokens = [11, 12, 13, 14, 99]

    first = store.token_chain(tokens, context_fingerprint=TEXT_CONTEXT_FINGERPRINT)
    second = store.token_chain(tuple(tokens), context_fingerprint="")

    assert first == second
    assert len(first) == 2
    assert first[0].token_start == 0
    assert first[0].token_end == 2
    assert first[1].token_start == 2
    assert first[1].token_end == 4
    assert first[1].parent_hash == first[0].block_hash
    assert first[0].block_hash != first[1].block_hash


def test_context_fingerprint_is_part_of_every_chain_and_checkpoint_key() -> None:
    store = _store()
    text = store.token_chain([1, 2], context_fingerprint="")
    vision_a = store.token_chain([1, 2], context_fingerprint=VISION_A)
    vision_b = store.token_chain([1, 2], context_fingerprint=VISION_B)

    assert text[0].parent_hash != vision_a[0].parent_hash
    assert vision_a[0].parent_hash != vision_b[0].parent_hash

    text_checkpoint = store.create_checkpoint(
        [1, 2],
        context_fingerprint="",
        payload=object(),
    ).checkpoint
    vision_checkpoint = store.create_checkpoint(
        [1, 2],
        context_fingerprint=VISION_A,
        payload=object(),
    ).checkpoint
    assert text_checkpoint.identity.checkpoint_key != (
        vision_checkpoint.identity.checkpoint_key
    )


def test_only_explicit_commit_publishes_a_complete_owner_checkpoint() -> None:
    store = _store()
    payload = {"qsa": [1], "gdn": [2], "ple": [3]}
    lease = _lease(store, "cancelled", VISION_A)
    pending = store.create_checkpoint(
        [1, 2],
        context_fingerprint=VISION_A,
        payload=payload,
    )

    release = store.release_request(
        lease,
        context_fingerprint=VISION_A,
        reason=Qwen4ExpPrefixReleaseReason.CANCELLED,
    )
    probe = _lease(store, "probe", VISION_A)
    miss = store.lookup(
        probe,
        [1, 2],
        context_fingerprint=VISION_A,
    )

    assert release.released_references == 0
    assert miss.source is Qwen4ExpPrefixLookupSource.MISS
    assert store.stats().commits == 0
    assert pending.checkpoint.payload is payload


@pytest.mark.parametrize("tokens", [[], [1], [1, 2, 3]])
def test_checkpoint_creation_rejects_non_boundaries(tokens: list[int]) -> None:
    store = _store(block_size=2)
    with pytest.raises(ValueError, match="complete block boundary"):
        store.create_checkpoint(
            tokens,
            context_fingerprint="",
            payload=object(),
        )


def test_longest_prefix_returns_exact_boundary_and_same_opaque_payload() -> None:
    store = _store()
    short_payload = object()
    long_payload = {"opaque": ["do-not-touch"]}
    _commit(store, "store-short", [1, 2], short_payload)
    long_checkpoint, _ = _commit(
        store,
        "store-long",
        [1, 2, 3, 4],
        long_payload,
    )

    tokens = [1, 2, 3, 4, 5]
    before = list(tokens)
    lease = _lease(store, "lookup")
    hit = store.lookup(lease, tokens, context_fingerprint="")

    assert hit.source is Qwen4ExpPrefixLookupSource.HOT
    assert hit.matched_tokens == 4
    assert hit.checkpoint is long_checkpoint
    assert hit.checkpoint.payload is long_payload
    assert long_payload == {"opaque": ["do-not-touch"]}
    assert tokens == before
    assert hit.refcount == 1


def test_detach_lookup_unpins_payload_without_closing_request_lease() -> None:
    store = _store(max_entries=1, max_tokens=4)
    _commit(store, "seed", [1, 2], object())
    lease = _lease(store, "lookup")
    hit = store.lookup(lease, [1, 2, 3], context_fingerprint="")

    assert hit.hit is True
    assert store.detach_lookup(lease, context_fingerprint="") is True
    assert store.detach_lookup(lease, context_fingerprint="") is False

    pending = store.create_checkpoint(
        [1, 2, 3, 4],
        context_fingerprint="",
        payload=object(),
    )
    receipt = store.commit(
        lease,
        [1, 2, 3, 4],
        pending,
        context_fingerprint="",
    )
    assert receipt.published is True


def test_identical_tokens_never_cross_context_fingerprint_domains() -> None:
    store = _store()
    vision_checkpoint, _ = _commit(
        store,
        "vision-store",
        [7, 8, 9, 10],
        object(),
        VISION_A,
    )

    text_lease = _lease(store, "text-lookup", "")
    text_result = store.lookup(
        text_lease,
        [7, 8, 9, 10],
        context_fingerprint="",
    )
    other_lease = _lease(store, "other-vision", VISION_B)
    other_result = store.lookup(
        other_lease,
        [7, 8, 9, 10],
        context_fingerprint=VISION_B,
    )
    exact_lease = _lease(store, "exact-vision", VISION_A)
    exact_result = store.lookup(
        exact_lease,
        [7, 8, 9, 10],
        context_fingerprint=VISION_A,
    )

    assert text_result.source is Qwen4ExpPrefixLookupSource.MISS
    assert other_result.source is Qwen4ExpPrefixLookupSource.MISS
    assert exact_result.checkpoint is vision_checkpoint


def test_lookup_and_commit_reject_context_fingerprint_substitution() -> None:
    store = _store()
    lease = _lease(store, "vision", VISION_A)
    pending = store.create_checkpoint(
        [1, 2],
        context_fingerprint=VISION_A,
        payload=object(),
    )

    with pytest.raises(Qwen4ExpPrefixStoreIdentityError, match="stale or foreign"):
        store.lookup(lease, [1, 2], context_fingerprint=VISION_B)
    with pytest.raises(Qwen4ExpPrefixStoreIdentityError, match="stale or foreign"):
        store.commit(
            lease,
            [1, 2],
            pending,
            context_fingerprint=VISION_B,
        )


def test_namespace_rebind_invalidates_hot_ssd_and_active_leases() -> None:
    ssd = MemorySSD()
    store = _store(ssd=ssd)
    _commit(store, "seed", [1, 2], object())
    stale = _lease(store, "stale")

    receipt = store.bind_namespace(NAMESPACE_B)

    assert receipt.previous_signature == NAMESPACE_A
    assert receipt.invalidated_hot_entries == 1
    assert receipt.invalidated_active_leases == 1
    assert receipt.invalidated_persistent_entries == 1
    assert store.stats().hot_entries == 0
    assert ssd.entries == {}
    with pytest.raises(Qwen4ExpPrefixStoreIdentityError, match="stale or foreign"):
        store.lookup(stale, [1, 2], context_fingerprint="")


def test_release_is_idempotent_but_old_lease_cannot_control_reopened_request() -> None:
    store = _store()
    lease = _lease(store, "request")
    first = store.release_request(
        lease,
        context_fingerprint="",
        reason=Qwen4ExpPrefixReleaseReason.ABORTED,
    )
    second = store.release_request(
        lease,
        context_fingerprint="",
        reason=Qwen4ExpPrefixReleaseReason.ABORTED,
    )
    reopened = _lease(store, "request")

    assert first.already_released is False
    assert second.already_released is True
    assert reopened.lease_id != lease.lease_id
    with pytest.raises(Qwen4ExpPrefixStoreIdentityError, match="stale or foreign"):
        store.lookup(lease, [1, 2], context_fingerprint="")


def test_active_leases_are_bounded() -> None:
    store = _store(max_leases=1)
    _lease(store, "one")
    with pytest.raises(Qwen4ExpPrefixStoreCapacityError, match="maximum active"):
        _lease(store, "two")


def test_hot_lru_respects_entry_token_and_refcount_bounds() -> None:
    store = _store(max_entries=2, max_tokens=6)
    first, _ = _commit(store, "first", [1, 2], object())
    second, _ = _commit(store, "second", [3, 4, 5, 6], object())
    pinned = _lease(store, "pinned")
    pin = store.lookup(pinned, [1, 2], context_fingerprint="")
    assert pin.checkpoint is first

    writer = _lease(store, "writer")
    too_large_while_pinned = store.create_checkpoint(
        [10, 11, 12, 13, 14, 15],
        context_fingerprint="",
        payload=object(),
    )
    with pytest.raises(Qwen4ExpPrefixStoreCapacityError, match="pinned"):
        store.commit(
            writer,
            [10, 11, 12, 13, 14, 15],
            too_large_while_pinned,
            context_fingerprint="",
        )
    assert store.stats().hot_entries == 2
    assert store.stats().hot_tokens == 6

    store.release_request(
        pinned,
        context_fingerprint="",
        reason=Qwen4ExpPrefixReleaseReason.COMPLETED,
    )
    receipt = store.commit(
        writer,
        [10, 11, 12, 13, 14, 15],
        too_large_while_pinned,
        context_fingerprint="",
    )
    assert set(receipt.evicted_checkpoint_keys) == {
        first.identity.checkpoint_key,
        second.identity.checkpoint_key,
    }
    assert store.stats().hot_entries == 1
    assert store.stats().hot_tokens == 6
    assert store.stats().evictions == 2


def test_ssd_rehydrates_exact_checkpoint_without_serialization_guessing() -> None:
    ssd = MemorySSD()
    first_store = _store(ssd=ssd)
    payload = {"whole-semantic-bundle": object()}
    checkpoint, _ = _commit(
        first_store,
        "persist",
        [20, 21, 22, 23],
        payload,
        VISION_A,
    )
    second_store = _store(ssd=ssd)
    lease = _lease(second_store, "rehydrate", VISION_A)

    hit = second_store.lookup(
        lease,
        [20, 21, 22, 23, 24],
        context_fingerprint=VISION_A,
    )

    assert hit.source is Qwen4ExpPrefixLookupSource.SSD
    assert hit.matched_tokens == 4
    assert hit.checkpoint is checkpoint
    assert hit.checkpoint.payload is payload
    assert second_store.stats().ssd_hits == 1
    assert second_store.stats().hot_entries == 1


def test_corrupt_ssd_identity_is_deleted_and_never_returned_as_a_holey_hit() -> None:
    ssd = MemorySSD()
    producer = _store(ssd=ssd)
    checkpoint, _ = _commit(producer, "persist", [1, 2, 3, 4], object())
    expected_key = checkpoint.identity.checkpoint_key
    wrong_identity = replace(
        checkpoint.identity,
        terminal_block_hash="0" * 64,
    )
    ssd.entries[expected_key] = Qwen4ExpWholeBoundaryCheckpoint(
        wrong_identity,
        object(),
    )
    consumer = _store(ssd=ssd)
    lease = _lease(consumer, "lookup")

    result = consumer.lookup(lease, [1, 2, 3, 4], context_fingerprint="")

    assert result.source is Qwen4ExpPrefixLookupSource.MISS
    assert result.matched_tokens == 0
    assert expected_key in ssd.deletes
    assert consumer.stats().invalidated_entries == 1


def test_checkpoint_receipts_and_identities_are_immutable() -> None:
    store = _store()
    checkpoint, receipt = _commit(store, "immutable", [1, 2], object())

    with pytest.raises(FrozenInstanceError):
        checkpoint.identity.token_count = 99  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        receipt.published = False  # type: ignore[misc]


def test_foreign_store_cannot_publish_an_owner_checkpoint() -> None:
    first = _store()
    second = _store()
    pending = first.create_checkpoint(
        [1, 2],
        context_fingerprint="",
        payload=object(),
    )
    lease = _lease(second, "foreign")

    with pytest.raises(Qwen4ExpPrefixStoreIdentityError, match="store owner"):
        second.commit(
            lease,
            [1, 2],
            pending,
            context_fingerprint="",
        )
