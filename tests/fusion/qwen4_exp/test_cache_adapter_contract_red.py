"""RED contracts for atomic Qwen4Exp semantic-cache verification."""

from __future__ import annotations

import threading

import pytest

from mlx_batch_server.runtime.contracts import BackendKind, RuntimeKey
from mlx_batch_server.runtime.fusion.qwen4_exp.cache_adapter import (
    Qwen4ExpSemanticCacheAdapter,
    Qwen4ExpSemanticCacheBinding,
    Qwen4ExpSemanticCacheIdentityError,
    Qwen4ExpSemanticCachePoisonedError,
    Qwen4ExpSemanticCacheStateError,
    Qwen4ExpVerifyDisposition,
    Qwen4ExpVerifyPhase,
)

RUNTIME = RuntimeKey(
    model_id="grant-ai/Qwen3.8-Flash-Next-Abliterated-MLX-4bit",
    revision="000544f8cddcbde27c1bc302deac2b5b4d45a5b1",
    backend=BackendKind.FUSED_MTP_MLX,
)


class RecordingKernel:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.commit_result = True
        self.fail: str | None = None
        self.snapshot_result: object = {"qsa": "q", "gdn": "g", "ple": "p"}
        self.capture_result: object = "capture-token"

    def _record(self, name: str, *values: object) -> None:
        self.calls.append((name, *values))
        if self.fail == name:
            raise RuntimeError(f"{name} failed")

    def snapshot(self, bundle: object) -> object:
        self._record("snapshot", bundle)
        return self.snapshot_result

    def begin_capture(self, bundle: object) -> object:
        self._record("begin_capture", bundle)
        return self.capture_result

    def end_capture(self, bundle: object, capture_token: object) -> None:
        self._record("end_capture", bundle, capture_token)

    def commit_verified_window(
        self,
        bundle: object,
        snapshot: object,
        *,
        keep_tokens: int,
        verified_tokens: int,
    ) -> bool:
        self._record(
            "commit",
            bundle,
            snapshot,
            keep_tokens,
            verified_tokens,
        )
        return self.commit_result

    def rollback_after_verify(
        self,
        bundle: object,
        snapshot: object,
        verified_tokens: int,
    ) -> None:
        self._record("rollback", bundle, snapshot, verified_tokens)

    def clear_verify_capture(self, bundle: object) -> None:
        self._record("clear", bundle)


def _adapter(
    kernel: RecordingKernel | None = None,
) -> tuple[Qwen4ExpSemanticCacheAdapter, RecordingKernel, object]:
    selected = kernel or RecordingKernel()
    bundle = object()
    adapter = Qwen4ExpSemanticCacheAdapter(
        binding=Qwen4ExpSemanticCacheBinding(
            request_id="response_3more",
            lease_id="lease_1",
            runtime=RUNTIME,
            bundle=bundle,
        ),
        kernel=selected,
    )
    return adapter, selected, bundle


def test_commit_is_one_atomic_qsa_gdn_ple_transaction() -> None:
    adapter, kernel, bundle = _adapter()

    lease = adapter.begin_verify(verified_tokens=4)
    lease.forward_complete()
    receipt = lease.resolve(keep_tokens=3)

    assert [call[0] for call in kernel.calls] == [
        "snapshot",
        "begin_capture",
        "end_capture",
        "commit",
        "clear",
    ]
    assert kernel.calls[0][1] is bundle
    assert kernel.calls[3][3:] == (3, 4)
    assert receipt.disposition is Qwen4ExpVerifyDisposition.COMMITTED
    assert receipt.keep_tokens == 3
    assert receipt.requires_reforward is False
    assert adapter.phase is Qwen4ExpVerifyPhase.IDLE


def test_refused_commit_rolls_back_before_requesting_reforward() -> None:
    kernel = RecordingKernel()
    kernel.commit_result = False
    adapter, _, _ = _adapter(kernel)

    lease = adapter.begin_verify(verified_tokens=4)
    lease.forward_complete()
    receipt = lease.resolve(keep_tokens=2)

    assert [call[0] for call in kernel.calls] == [
        "snapshot",
        "begin_capture",
        "end_capture",
        "commit",
        "rollback",
        "clear",
    ]
    assert receipt.disposition is Qwen4ExpVerifyDisposition.ROLLED_BACK
    assert receipt.requires_reforward is True


def test_zero_kept_tokens_skips_commit_and_rolls_back() -> None:
    adapter, kernel, _ = _adapter()

    lease = adapter.begin_verify(verified_tokens=3)
    lease.forward_complete()
    receipt = lease.resolve(keep_tokens=0)

    assert "commit" not in [call[0] for call in kernel.calls]
    assert [call[0] for call in kernel.calls][-2:] == ["rollback", "clear"]
    assert receipt.requires_reforward is True


def test_abort_exits_capture_then_rolls_back_the_whole_bundle() -> None:
    adapter, kernel, _ = _adapter()

    lease = adapter.begin_verify(verified_tokens=2)
    receipt = lease.abort()

    assert [call[0] for call in kernel.calls] == [
        "snapshot",
        "begin_capture",
        "end_capture",
        "rollback",
        "clear",
    ]
    assert receipt.keep_tokens == 0
    assert receipt.disposition is Qwen4ExpVerifyDisposition.ROLLED_BACK


def test_only_one_verify_is_active_and_old_handles_are_stale() -> None:
    adapter, _, _ = _adapter()
    first = adapter.begin_verify(verified_tokens=2)

    with pytest.raises(Qwen4ExpSemanticCacheStateError, match="already active"):
        adapter.begin_verify(verified_tokens=2)
    first.abort()

    second = adapter.begin_verify(verified_tokens=2)
    with pytest.raises(Qwen4ExpSemanticCacheIdentityError, match="stale"):
        first.forward_complete()
    second.abort()


@pytest.mark.parametrize("operation", ["commit", "rollback", "clear"])
def test_uncertain_mutation_poisons_instead_of_guessing_recovery(
    operation: str,
) -> None:
    kernel = RecordingKernel()
    adapter, _, _ = _adapter(kernel)
    lease = adapter.begin_verify(verified_tokens=3)
    lease.forward_complete()
    kernel.fail = operation
    if operation == "rollback":
        kernel.commit_result = False

    with pytest.raises(Qwen4ExpSemanticCachePoisonedError):
        lease.resolve(keep_tokens=2)

    assert adapter.phase is Qwen4ExpVerifyPhase.POISONED
    with pytest.raises(Qwen4ExpSemanticCachePoisonedError):
        adapter.begin_verify(verified_tokens=1)


def test_all_mutation_is_owner_thread_only() -> None:
    adapter, _, _ = _adapter()
    errors: list[BaseException] = []

    def foreign_call() -> None:
        try:
            adapter.begin_verify(verified_tokens=2)
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=foreign_call)
    thread.start()
    thread.join()

    assert len(errors) == 1
    assert isinstance(errors[0], Qwen4ExpSemanticCacheStateError)
    assert "owner thread" in str(errors[0])


def test_close_requires_idle_and_prevents_reuse() -> None:
    adapter, _, _ = _adapter()
    lease = adapter.begin_verify(verified_tokens=2)
    with pytest.raises(Qwen4ExpSemanticCacheStateError, match="active"):
        adapter.close()
    lease.abort()

    adapter.close()
    assert adapter.phase is Qwen4ExpVerifyPhase.CLOSED
    with pytest.raises(Qwen4ExpSemanticCacheStateError, match="closed"):
        adapter.begin_verify(verified_tokens=1)


def test_missing_snapshot_fails_before_capture_without_mutation() -> None:
    kernel = RecordingKernel()
    kernel.snapshot_result = None
    adapter, _, _ = _adapter(kernel)

    with pytest.raises(Qwen4ExpSemanticCacheStateError, match="snapshot"):
        adapter.begin_verify(verified_tokens=2)

    assert [call[0] for call in kernel.calls] == ["snapshot"]
    assert adapter.phase is Qwen4ExpVerifyPhase.IDLE


def test_missing_capture_token_is_cleared_and_rejected() -> None:
    kernel = RecordingKernel()
    kernel.capture_result = None
    adapter, _, _ = _adapter(kernel)

    with pytest.raises(Qwen4ExpSemanticCacheStateError, match="capture token"):
        adapter.begin_verify(verified_tokens=2)

    assert [call[0] for call in kernel.calls] == [
        "snapshot",
        "begin_capture",
        "clear",
    ]
    assert adapter.phase is Qwen4ExpVerifyPhase.IDLE


def test_missing_capture_token_with_failed_cleanup_poisons() -> None:
    kernel = RecordingKernel()
    kernel.capture_result = None
    kernel.fail = "clear"
    adapter, _, _ = _adapter(kernel)

    with pytest.raises(Qwen4ExpSemanticCachePoisonedError):
        adapter.begin_verify(verified_tokens=2)

    assert adapter.phase is Qwen4ExpVerifyPhase.POISONED
