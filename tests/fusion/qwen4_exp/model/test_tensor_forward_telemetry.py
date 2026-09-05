from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.cache import ArraysCache
from scripts.benchmark_live_responses import _tensor_forward_requirements

from mlx_batch_server.runtime.fusion.qwen4_exp.model import tensor as tensor_module
from mlx_batch_server.runtime.fusion.qwen4_exp.model.sampling import SamplerConfig
from mlx_batch_server.runtime.fusion.qwen4_exp.model.tensor import (
    _Qwen4ExpTensorRuntime,
    _snapshot_value,
)
from mlx_batch_server.runtime.fusion.qwen4_exp.model.tensor_support import (
    Qwen4ExpTensorCapabilities,
    current_attention_phase,
)

_VOCAB_SIZE = 64
_PHASES = (
    "target_decode",
    "mtp_draft",
    "target_verify",
    "target_correction",
    "mtp_history_update",
)


def _cache() -> list[ArraysCache]:
    return [ArraysCache(size=2)]


def _logits_for(token_id: int) -> mx.array:
    logits = np.full((1, 1, _VOCAB_SIZE), -100.0, dtype=np.float32)
    logits[0, 0, token_id] = 100.0
    return mx.array(logits)


def _reservation(primary: int) -> SimpleNamespace:
    sampler = SamplerConfig(temperature=1.0, top_p=1.0, top_k=1)
    return SimpleNamespace(
        cache=_cache(),
        mtp_cache=_cache(),
        position_table=None,
        mrope=None,
        hidden=mx.array([[[float(primary)]]]),
        logits=_logits_for(primary),
        sampler=sampler,
        draft_sampler=sampler,
        rng=np.random.default_rng(primary),
        pending_primary=None,
        max_output_tokens=16,
        output_tokens=0,
        position=5,
        aborted=False,
    )


class _FakeTensorModel:
    def __init__(self, *, refuse_commit: bool = False) -> None:
        self.refuse_commit = refuse_commit
        self.raise_on_target = False
        self.draft_calls: list[tuple[int, ...]] = []
        self.target_calls: list[tuple[str, tuple[tuple[int, ...], ...]]] = []
        self.history_calls: list[tuple[int, ...]] = []

    @staticmethod
    def snapshot(bundle: list[ArraysCache]) -> tuple[object, ...]:
        return tuple(_snapshot_value(entry.state) for entry in bundle)

    @staticmethod
    def begin_capture(bundle: list[ArraysCache]) -> object:
        assert bundle
        return object()

    @staticmethod
    def end_capture(bundle: list[ArraysCache], token: object) -> None:
        assert bundle
        assert token is not None

    def mtp_forward(
        self,
        hidden_states: mx.array,
        next_token_ids: mx.array,
        *,
        mtp_cache: list[ArraysCache],
        return_hidden: bool,
    ) -> tuple[mx.array, mx.array]:
        assert return_hidden
        assert mtp_cache
        tokens = tuple(int(token) for token in np.asarray(next_token_ids).reshape(-1))
        self.draft_calls.append(tokens)
        desired = np.asarray(tokens, dtype=np.int64) + 1
        logits = np.full((len(tokens), 1, _VOCAB_SIZE), -100.0, dtype=np.float32)
        logits[np.arange(len(tokens)), 0, desired] = 100.0
        return mx.array(logits), hidden_states + 1

    def __call__(
        self,
        inputs: mx.array,
        *,
        cache: list[ArraysCache],
        return_hidden: bool,
        logits_keep: int,
    ) -> tuple[mx.array, mx.array]:
        assert return_hidden
        assert logits_keep == 0
        if self.raise_on_target:
            raise RuntimeError("target forward failed")
        tokens = np.asarray(inputs, dtype=np.int64)
        rows = tuple(tuple(int(token) for token in row) for row in tokens)
        phase = current_attention_phase()
        self.target_calls.append((phase, rows))
        logits = np.full((*tokens.shape, _VOCAB_SIZE), -100.0, dtype=np.float32)
        for row_index, row in enumerate(rows):
            for token_index, token in enumerate(row):
                desired = token + (10 if phase == "verify" and token_index == 0 else 1)
                logits[row_index, token_index, desired] = 100.0
        hidden = mx.array(tokens[..., None].astype(np.float32) + 0.5)
        if phase == "verify":
            capture = mx.array(tokens[..., None].astype(np.float32))
            cache[0]._qwen4_exp_verify_rows = (capture,) * 6
            cache[0]._qwen4_exp_verify_ple = (capture, inputs)
        return mx.array(logits), hidden

    def commit_verified_window(
        self,
        cache: list[ArraysCache],
        snapshot: object,
        *,
        keep_tokens: int,
        verified_tokens: int,
    ) -> bool:
        assert cache
        assert snapshot is not None
        assert keep_tokens == 1
        assert verified_tokens == 3
        return not self.refuse_commit

    @staticmethod
    def clear_verify_capture(cache: list[ArraysCache]) -> None:
        for attr in ("_qwen4_exp_verify_rows", "_qwen4_exp_verify_ple"):
            if hasattr(cache[0], attr):
                setattr(cache[0], attr, None)

    def mtp_update_cache(
        self,
        hidden_states: mx.array,
        next_token_ids: mx.array,
        *,
        mtp_cache: list[ArraysCache],
    ) -> mx.array:
        assert mtp_cache
        tokens = tuple(int(token) for token in np.asarray(next_token_ids).reshape(-1))
        self.history_calls.append(tokens)
        return hidden_states + 1


def _runtime(model: _FakeTensorModel) -> _Qwen4ExpTensorRuntime:
    runtime = object.__new__(_Qwen4ExpTensorRuntime)
    runtime.model = model
    runtime.capabilities = Qwen4ExpTensorCapabilities()
    runtime._stop_token_ids = frozenset({0})
    runtime._reservations = {}
    runtime._encoders = {}
    runtime._closed = False
    runtime._prefill_rows = 0
    runtime._decode_rows = 0
    runtime.plan = SimpleNamespace(
        plan_sha256="plan-sha",
        topology=SimpleNamespace(tensor_batch_mode=SimpleNamespace(value="row_serial")),
    )
    runtime._prefix_config = SimpleNamespace(block_size_tokens=256)
    runtime._prefix_store = SimpleNamespace(
        stats=lambda: SimpleNamespace(
            hot_entries=0,
            hot_tokens=0,
            hot_hits=0,
            ssd_hits=0,
            misses=0,
            commits=0,
        )
    )
    return runtime


def _telemetry(runtime: _Qwen4ExpTensorRuntime) -> dict[str, Any]:
    value = runtime.stats()["tensor_forward"]
    assert isinstance(value, dict)
    return value


def test_batched_physical_calls_publish_exact_phase_histogram_deltas() -> None:
    model = _FakeTensorModel()
    runtime = _runtime(model)
    rows = (_reservation(10), _reservation(20))
    before = _telemetry(runtime)

    runtime._target_forward_batch(((1,), (2,)), rows)
    runtime._decode_mtp_batch(rows, draft_depth=2)

    after = _telemetry(runtime)
    phases = after["phases"]
    assert model.target_calls == [
        ("decode", ((1,), (2,))),
        ("verify", ((10, 11, 12), (20, 21, 22))),
        ("decode", ((20,), (30,))),
    ]
    assert model.draft_calls == [(10, 20), (11, 21)]
    assert model.history_calls == [(10, 20), (20, 30)]
    assert phases == {
        "target_decode": {
            "completed_calls": 1,
            "completed_rows": 2,
            "max_completed_rows": 2,
            "completed_calls_by_shape": {"2x1": 1},
        },
        "mtp_draft": {
            "completed_calls": 2,
            "completed_rows": 4,
            "max_completed_rows": 2,
            "completed_calls_by_shape": {"2x1": 2},
        },
        "target_verify": {
            "completed_calls": 1,
            "completed_rows": 2,
            "max_completed_rows": 2,
            "completed_calls_by_shape": {"2x3": 1},
        },
        "target_correction": {
            "completed_calls": 1,
            "completed_rows": 2,
            "max_completed_rows": 2,
            "completed_calls_by_shape": {"2x1": 1},
        },
        "mtp_history_update": {
            "completed_calls": 2,
            "completed_rows": 4,
            "max_completed_rows": 2,
            "completed_calls_by_shape": {"2x1": 2},
        },
    }
    requirement = _tensor_forward_requirements(
        before,
        after,
        target_rows=2,
        mtp_rows=2,
    )
    assert requirement["passed"] is True
    assert requirement["delta"]["same_runtime_instance"] is True
    assert requirement["delta"]["phases"] == phases


def test_stats_reads_are_stable_and_new_runtime_resets_with_new_uuid() -> None:
    first_runtime = _runtime(_FakeTensorModel())
    first = _telemetry(first_runtime)
    repeated = _telemetry(first_runtime)

    assert first == repeated
    assert first is not repeated
    assert uuid.UUID(first["runtime_instance_id"]).version == 4
    assert first["schema"] == "qwen4-exp.tensor-forward.v1"
    assert first["reset_semantics"] == "per_tensor_runtime_instance"
    assert tuple(first["phases"]) == _PHASES
    first["phases"]["target_decode"]["completed_calls_by_shape"]["99x99"] = 1
    assert (
        "99x99"
        not in _telemetry(first_runtime)["phases"]["target_decode"][
            "completed_calls_by_shape"
        ]
    )

    reloaded = _telemetry(_runtime(_FakeTensorModel()))
    assert reloaded["runtime_instance_id"] != first["runtime_instance_id"]
    assert all(phase["completed_calls"] == 0 for phase in reloaded["phases"].values())


def test_live_requirements_reject_stale_maxima_and_runtime_reload() -> None:
    runtime = _runtime(_FakeTensorModel())
    rows = (_reservation(10), _reservation(20))
    runtime._target_forward_batch(((1,), (2,)), rows)
    stale_before = _telemetry(runtime)
    stale_after = _telemetry(runtime)

    stale = _tensor_forward_requirements(
        stale_before,
        stale_after,
        target_rows=2,
        mtp_rows=None,
    )
    assert stale["passed"] is False
    assert stale["delta"]["phases"]["target_decode"]["completed_calls_by_shape"] == {
        "2x1": 0
    }
    assert any("no positive target" in error for error in stale["errors"])

    reloaded_after = _telemetry(_runtime(_FakeTensorModel()))
    reloaded = _tensor_forward_requirements(
        stale_before,
        reloaded_after,
        target_rows=2,
        mtp_rows=None,
    )
    assert reloaded["passed"] is False
    assert reloaded["delta"]["same_runtime_instance"] is False
    assert any("runtime instance changed" in error for error in reloaded["errors"])


def test_validation_model_eval_and_scatter_failures_do_not_increment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _FakeTensorModel()
    runtime = _runtime(model)
    rows = (_reservation(10), _reservation(20))

    with pytest.raises(ValueError, match="positive width"):
        runtime._target_forward_batch(((), ()), rows)
    assert _telemetry(runtime)["phases"]["target_decode"]["completed_calls"] == 0

    model.raise_on_target = True
    with pytest.raises(RuntimeError, match="target forward failed"):
        runtime._target_forward_batch(((1,), (2,)), rows)
    model.raise_on_target = False
    assert _telemetry(runtime)["phases"]["target_decode"]["completed_calls"] == 0

    with monkeypatch.context() as patch:
        patch.setattr(
            tensor_module.mx,
            "eval",
            lambda *_values: (_ for _ in ()).throw(RuntimeError("eval failed")),
        )
        with pytest.raises(RuntimeError, match="eval failed"):
            runtime._target_forward_batch(((1,), (2,)), rows)
    assert _telemetry(runtime)["phases"]["target_decode"]["completed_calls"] == 0

    with monkeypatch.context() as patch:
        patch.setattr(
            tensor_module,
            "_scatter_batch_cache",
            lambda *_values: (_ for _ in ()).throw(RuntimeError("scatter failed")),
        )
        with pytest.raises(RuntimeError, match="scatter failed"):
            runtime._target_forward_batch(((1,), (2,)), rows)
    assert _telemetry(runtime)["phases"]["target_decode"]["completed_calls"] == 0


def test_refused_verified_commit_never_increments_target_verify_evidence() -> None:
    runtime = _runtime(_FakeTensorModel(refuse_commit=True))

    with pytest.raises(RuntimeError, match="verified-window commit was refused"):
        runtime._decode_mtp_batch((_reservation(10), _reservation(20)), draft_depth=2)

    phases = _telemetry(runtime)["phases"]
    assert phases["mtp_draft"]["completed_calls"] == 2
    assert phases["target_verify"]["completed_calls"] == 0
    assert phases["target_correction"]["completed_calls"] == 0
    assert phases["mtp_history_update"]["completed_calls"] == 0
