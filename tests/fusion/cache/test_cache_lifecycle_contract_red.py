"""RED contracts for the source-only fused cache lifecycle.

These tests are intentionally authored but not executed while the Compile
Embargo is HOLD.
"""

from __future__ import annotations

from dataclasses import replace
from threading import Event, Thread
from typing import Any

import pytest

from mlx_batch_server.cache.contracts import CacheIdentity
from mlx_batch_server.runtime.contracts import BackendKind
from mlx_batch_server.runtime.fusion.cache.contracts import (
    CacheLayout,
    CacheLeaseState,
    CacheReleaseReason,
    CacheTier,
)
from mlx_batch_server.runtime.fusion.cache.identity import build_cache_namespace
from mlx_batch_server.runtime.fusion.cache.lifecycle import FusionCacheCoordinator


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.actions: dict[str, list[int | Exception | None]] = {}

    def script(self, operation: str, *actions: int | Exception | None) -> None:
        self.actions[operation] = list(actions)

    def record(
        self,
        operation: str,
        *details: Any,
        default: int | None = None,
    ) -> int | None:
        self.calls.append((operation, *details))
        scripted = self.actions.get(operation)
        action = scripted.pop(0) if scripted else default
        if isinstance(action, Exception):
            raise action
        return action


class _PagedPort:
    def __init__(self, recorder: _Recorder) -> None:
        self.recorder = recorder

    def bind_namespace(self, namespace: Any) -> int:
        result = self.recorder.record("paged.bind", namespace.signature, default=2)
        assert isinstance(result, int)
        return result

    def open_request(self, request_id: str, lease_id: str) -> None:
        self.recorder.record("paged.open", request_id, lease_id)

    def release_request(
        self,
        request_id: str,
        *,
        retain_reusable_blocks: bool,
    ) -> int:
        result = self.recorder.record(
            "paged.release",
            request_id,
            retain_reusable_blocks,
            default=3,
        )
        assert isinstance(result, int)
        return result


class _PrefixPort:
    def __init__(self, recorder: _Recorder) -> None:
        self.recorder = recorder

    def bind_namespace(self, namespace: Any) -> int:
        result = self.recorder.record("prefix.bind", namespace.signature, default=5)
        assert isinstance(result, int)
        return result

    def open_request(self, request_id: str, lease_id: str) -> None:
        self.recorder.record("prefix.open", request_id, lease_id)

    def clear_request_entry(
        self,
        request_id: str,
        *,
        retain_reusable_blocks: bool,
    ) -> int:
        result = self.recorder.record(
            "prefix.clear",
            request_id,
            retain_reusable_blocks,
            default=7,
        )
        assert isinstance(result, int)
        return result


class _SSDPort:
    def __init__(self, recorder: _Recorder) -> None:
        self.recorder = recorder

    def bind_namespace(self, namespace: Any) -> int:
        result = self.recorder.record("ssd.bind", namespace.signature, default=11)
        assert isinstance(result, int)
        return result

    def open_request(self, request_id: str, lease_id: str) -> None:
        self.recorder.record("ssd.open", request_id, lease_id)

    def quiesce_request(self, request_id: str, *, commit: bool) -> None:
        self.recorder.record("ssd.quiesce", request_id, commit)

    def cleanup_request(self, request_id: str) -> int:
        result = self.recorder.record("ssd.cleanup", request_id, default=13)
        assert isinstance(result, int)
        return result


def _identity() -> CacheIdentity:
    return CacheIdentity(
        backend=BackendKind.FUSED_MTP_MLX,
        model_id="grant-ai/Qwen3.8-Flash-Next-Abliterated-MLX-4bit",
        model_revision="000544f8",
        quantization="4bit",
        adapter_path="/models/adapters/main",
        kv_layout="qwen4-exp-hybrid-v1",
        tokenizer_fingerprint="tokenizer-sha256",
    )


def _layout() -> CacheLayout:
    return CacheLayout(
        num_layers=2,
        block_size_tokens=256,
        layer_cache_types=("recurrent", "attention"),
        payload_layout="embedded-vlm",
        format_version=3,
        adapter_fingerprint="adapter-sha256",
        draft_model_id="embedded-mtp",
        draft_model_revision="mtp-revision",
        mtp_layout="qwen4-exp-mtp-v1",
        turboquant_kv_bits=4.0,
        cachelist_subtypes=(
            ("attention", ("keys", "values")),
            ("recurrent", ("conv", "state")),
        ),
    )


def _coordinator(
    *,
    recorder: _Recorder | None = None,
    max_completed_leases: int = 4,
) -> tuple[FusionCacheCoordinator, _Recorder]:
    recorder = recorder or _Recorder()
    coordinator = FusionCacheCoordinator(
        namespace=build_cache_namespace(_identity(), _layout()),
        paged=_PagedPort(recorder),
        prefix=_PrefixPort(recorder),
        ssd=_SSDPort(recorder),
        max_completed_leases=max_completed_leases,
    )
    return coordinator, recorder


def _operation_names(recorder: _Recorder) -> list[str]:
    return [str(call[0]) for call in recorder.calls]


def test_cache_namespace_is_deterministic_and_normalizes_cachelist_order() -> None:
    identity = _identity()
    layout = _layout()
    reordered = replace(
        layout,
        cachelist_subtypes=tuple(reversed(layout.cachelist_subtypes)),
    )

    first = build_cache_namespace(identity, layout)
    second = build_cache_namespace(replace(identity), replace(layout))
    normalized = build_cache_namespace(identity, reordered)

    assert first.signature == second.signature
    assert first.signature == normalized.signature
    assert first.signature.startswith("mlx-batch-fusion-cache-v1:")
    assert len(first.signature.rsplit(":", 1)[1]) == 64


def test_every_cache_compatibility_dimension_separates_the_namespace() -> None:
    identity = _identity()
    layout = _layout()
    variants = (
        (replace(identity, backend=BackendKind.LEGACY_MLX), layout),
        (replace(identity, model_id="another/model"), layout),
        (replace(identity, model_revision="another-revision"), layout),
        (replace(identity, quantization="8bit"), layout),
        (replace(identity, adapter_path="/models/adapters/other"), layout),
        (replace(identity, kv_layout="different-layout"), layout),
        (replace(identity, tokenizer_fingerprint="different-tokenizer"), layout),
        (
            identity,
            replace(layout, num_layers=3, layer_cache_types=("a", "b", "c")),
        ),
        (identity, replace(layout, block_size_tokens=128)),
        (identity, replace(layout, layer_cache_types=("attention", "recurrent"))),
        (identity, replace(layout, payload_layout="detached-vlm")),
        (identity, replace(layout, format_version=4)),
        (identity, replace(layout, adapter_fingerprint="different-adapter")),
        (identity, replace(layout, draft_model_id="external-draft")),
        (identity, replace(layout, draft_model_revision="different-draft")),
        (identity, replace(layout, mtp_layout="different-mtp")),
        (identity, replace(layout, turboquant_kv_bits=8.0)),
        (
            identity,
            replace(
                layout,
                cachelist_subtypes=(("attention", ("keys", "values", "scales")),),
            ),
        ),
    )
    baseline = build_cache_namespace(identity, layout).signature
    variant_signatures = {
        build_cache_namespace(candidate_identity, candidate_layout).signature
        for candidate_identity, candidate_layout in variants
    }

    assert baseline not in variant_signatures
    assert len(variant_signatures) == len(variants)


def test_activation_binds_ssd_before_paged_and_prefix_and_is_idempotent() -> None:
    coordinator, recorder = _coordinator()

    first = coordinator.activate()
    second = coordinator.activate()

    assert _operation_names(recorder) == ["ssd.bind", "paged.bind", "prefix.bind"]
    assert first is second
    assert tuple(item.tier for item in first.invalidations) == (
        CacheTier.SSD,
        CacheTier.PAGED,
        CacheTier.PREFIX,
    )
    assert tuple(item.invalidated_entries for item in first.invalidations) == (
        11,
        2,
        5,
    )


def test_partial_activation_retries_only_the_failed_and_unbound_tiers() -> None:
    recorder = _Recorder()
    recorder.script("paged.bind", RuntimeError("paged bind failed"), 2)
    coordinator, _ = _coordinator(recorder=recorder)

    with pytest.raises(RuntimeError, match="paged bind failed"):
        coordinator.activate()

    recovered = coordinator.activate()

    assert _operation_names(recorder) == [
        "ssd.bind",
        "paged.bind",
        "paged.bind",
        "prefix.bind",
    ]
    assert tuple(item.tier for item in recovered.invalidations) == (
        CacheTier.SSD,
        CacheTier.PAGED,
        CacheTier.PREFIX,
    )


def test_acquire_opens_paged_then_prefix_then_ssd() -> None:
    coordinator, recorder = _coordinator()
    coordinator.activate()
    recorder.calls.clear()

    lease = coordinator.acquire("response-1")

    assert lease.state is CacheLeaseState.ACTIVE
    assert _operation_names(recorder) == ["paged.open", "prefix.open", "ssd.open"]
    assert {call[2] for call in recorder.calls} == {lease.lease_id}


@pytest.mark.parametrize(
    ("reason", "commit"),
    (
        (CacheReleaseReason.COMPLETED, True),
        (CacheReleaseReason.CANCELLED, False),
        (CacheReleaseReason.ABORTED, False),
    ),
)
def test_cleanup_quiesces_ssd_before_releasing_paged_prefix_and_scratch(
    reason: CacheReleaseReason,
    commit: bool,
) -> None:
    coordinator, recorder = _coordinator()
    coordinator.activate()
    lease = coordinator.acquire("response-1")
    recorder.calls.clear()

    receipt = coordinator.cleanup("response-1", reason=reason)

    assert _operation_names(recorder) == [
        "ssd.quiesce",
        "paged.release",
        "prefix.clear",
        "ssd.cleanup",
    ]
    assert recorder.calls[0] == ("ssd.quiesce", "response-1", commit)
    assert recorder.calls[1][-1] is commit
    assert recorder.calls[2][-1] is commit
    assert receipt.released_tiers == (
        CacheTier.PAGED,
        CacheTier.PREFIX,
        CacheTier.SSD,
    )
    assert receipt.released_references == 23
    assert receipt.pending_writes_quiesced
    assert receipt.retained_reusable_blocks is commit
    assert receipt.succeeded
    assert lease.state is CacheLeaseState.RELEASED


def test_successful_cleanup_is_idempotent_without_revisiting_tiers() -> None:
    coordinator, recorder = _coordinator()
    coordinator.activate()
    coordinator.acquire("response-1")

    first = coordinator.cleanup("response-1", reason=CacheReleaseReason.COMPLETED)
    call_count = len(recorder.calls)
    repeated = coordinator.cleanup("response-1", reason=CacheReleaseReason.COMPLETED)

    assert len(recorder.calls) == call_count
    assert not first.already_released
    assert repeated.already_released
    assert replace(repeated, already_released=False) == first


def test_partial_cleanup_retry_preserves_prior_progress_in_final_receipt() -> None:
    recorder = _Recorder()
    recorder.script("paged.release", 3, 0)
    recorder.script("prefix.clear", RuntimeError("prefix clear failed"), 7)
    recorder.script("ssd.cleanup", 13, 0)
    coordinator, _ = _coordinator(recorder=recorder)
    coordinator.activate()
    lease = coordinator.acquire("response-1")

    failed = coordinator.cleanup("response-1", reason=CacheReleaseReason.ABORTED)
    first_cleanup_calls = _operation_names(recorder)[-3:]
    recovered = coordinator.cleanup("response-1", reason=CacheReleaseReason.ABORTED)

    assert not failed.succeeded
    assert failed.released_references == 3
    assert first_cleanup_calls == ["ssd.quiesce", "paged.release", "prefix.clear"]
    assert lease.state is CacheLeaseState.RELEASED
    assert recovered.succeeded
    assert recovered.reason is CacheReleaseReason.ABORTED
    assert recovered.released_references == 23
    assert recovered.released_tiers == (
        CacheTier.PAGED,
        CacheTier.PREFIX,
        CacheTier.SSD,
    )


def test_partial_cleanup_cannot_be_retried_with_a_different_release_reason() -> None:
    recorder = _Recorder()
    recorder.script("prefix.clear", RuntimeError("prefix clear failed"), 7)
    coordinator, _ = _coordinator(recorder=recorder)
    coordinator.activate()
    coordinator.acquire("response-1")
    coordinator.cleanup("response-1", reason=CacheReleaseReason.ABORTED)

    with pytest.raises(ValueError, match="release reason"):
        coordinator.cleanup("response-1", reason=CacheReleaseReason.COMPLETED)


def test_completed_tombstones_are_bounded_and_protect_recent_request_ids() -> None:
    coordinator, recorder = _coordinator(max_completed_leases=2)
    coordinator.activate()
    for request_id in ("one", "two", "three"):
        coordinator.acquire(request_id)
        coordinator.complete(request_id)

    before = len(recorder.calls)
    recent = coordinator.cleanup("three", reason=CacheReleaseReason.COMPLETED)
    assert recent.already_released
    assert len(recorder.calls) == before

    with pytest.raises(KeyError):
        coordinator.cleanup("one", reason=CacheReleaseReason.COMPLETED)

    recycled = coordinator.acquire("one")
    assert recycled.state is CacheLeaseState.ACTIVE


def test_evicted_tombstone_cannot_turn_an_old_lease_into_an_aba_release() -> None:
    coordinator, _ = _coordinator(max_completed_leases=1)
    coordinator.activate()
    old = coordinator.acquire("one")
    old.release()
    coordinator.acquire("two").release()
    current = coordinator.acquire("one")

    with pytest.raises(RuntimeError, match="stale cache lease"):
        old.abort(CacheReleaseReason.CANCELLED)

    assert current.state is CacheLeaseState.ACTIVE
    current.abort(CacheReleaseReason.CANCELLED)


def test_successful_partial_open_rollback_allows_a_clean_retry() -> None:
    recorder = _Recorder()
    recorder.script("prefix.open", RuntimeError("prefix open failed"), None)
    coordinator, _ = _coordinator(recorder=recorder)
    coordinator.activate()

    with pytest.raises(RuntimeError, match="prefix open failed"):
        coordinator.acquire("response-1")

    assert _operation_names(recorder)[-3:] == [
        "prefix.open",
        "paged.release",
        "prefix.clear",
    ]
    retried = coordinator.acquire("response-1")
    assert retried.state is CacheLeaseState.ACTIVE


def test_failed_partial_open_rollback_is_quarantined_until_cleanup_succeeds() -> None:
    recorder = _Recorder()
    recorder.script("prefix.open", RuntimeError("prefix open failed"), None)
    recorder.script("paged.release", RuntimeError("paged rollback failed"), 3)
    coordinator, _ = _coordinator(recorder=recorder)
    coordinator.activate()

    with pytest.raises(RuntimeError, match="prefix open failed"):
        coordinator.acquire("response-1")

    with pytest.raises(RuntimeError, match=r"already exists|cleanup incomplete"):
        coordinator.acquire("response-1")

    recovered = coordinator.cleanup(
        "response-1",
        reason=CacheReleaseReason.REJECTED,
    )
    assert recovered.succeeded
    assert recovered.reason is CacheReleaseReason.REJECTED


def test_cleanup_waits_for_opening_to_finish_before_observing_the_lease() -> None:
    recorder = _Recorder()
    entered = Event()
    release_open = Event()

    class _BlockingPagedPort(_PagedPort):
        def open_request(self, request_id: str, lease_id: str) -> None:
            super().open_request(request_id, lease_id)
            entered.set()
            assert release_open.wait(timeout=1)

    coordinator = FusionCacheCoordinator(
        namespace=build_cache_namespace(_identity(), _layout()),
        paged=_BlockingPagedPort(recorder),
        prefix=_PrefixPort(recorder),
        ssd=_SSDPort(recorder),
    )
    coordinator.activate()
    acquired: list[Any] = []
    cleaned: list[Any] = []
    acquire_thread = Thread(target=lambda: acquired.append(coordinator.acquire("one")))
    cleanup_thread = Thread(
        target=lambda: cleaned.append(
            coordinator.cleanup("one", reason=CacheReleaseReason.CANCELLED)
        )
    )

    acquire_thread.start()
    assert entered.wait(timeout=1)
    cleanup_thread.start()
    assert not cleaned
    release_open.set()
    acquire_thread.join(timeout=1)
    cleanup_thread.join(timeout=1)

    assert acquired[0].state is CacheLeaseState.RELEASED
    assert cleaned[0].succeeded


def test_persisted_namespace_identity_fails_closed_on_missing_dimensions() -> None:
    identity = _identity()
    layout = _layout()
    invalid = (
        (replace(identity, model_id=""), layout, "model_id"),
        (replace(identity, model_revision=None), layout, "model_revision"),
        (replace(identity, kv_layout=" "), layout, "kv_layout"),
        (
            replace(identity, tokenizer_fingerprint=""),
            layout,
            "tokenizer_fingerprint",
        ),
        (identity, replace(layout, adapter_fingerprint=None), "adapter_fingerprint"),
        (
            identity,
            replace(layout, draft_model_revision=None),
            "draft_model_revision",
        ),
        (
            replace(identity, adapter_path=None),
            layout,
            "adapter_fingerprint requires adapter_path",
        ),
        (
            identity,
            replace(layout, draft_model_id=None),
            "draft_model_revision requires draft_model_id",
        ),
    )

    for candidate_identity, candidate_layout, message in invalid:
        with pytest.raises(ValueError, match=message):
            build_cache_namespace(candidate_identity, candidate_layout)
