from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.cache import ArraysCache

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


def _logits_for(token_id: int) -> mx.array:
    logits = np.full((1, 1, _VOCAB_SIZE), -100.0, dtype=np.float32)
    logits[0, 0, token_id] = 100.0
    return mx.array(logits)


def _cache() -> list[ArraysCache]:
    return [ArraysCache(size=2)]


def _reservation(
    primary: int,
    *,
    stochastic: bool = False,
    max_output_tokens: int = 16,
    seed: int = 7,
) -> SimpleNamespace:
    sampler = SamplerConfig(
        temperature=1.0 if stochastic else 0.0,
        top_p=1.0,
        top_k=1 if stochastic else 0,
    )
    return SimpleNamespace(
        cache=_cache(),
        mtp_cache=_cache(),
        position_table=None,
        mrope=None,
        hidden=mx.array([[[float(primary)]]]),
        logits=_logits_for(primary),
        sampler=sampler,
        draft_sampler=sampler,
        rng=np.random.default_rng(seed) if stochastic else None,
        pending_primary=None,
        max_output_tokens=max_output_tokens,
        output_tokens=0,
        position=5,
        aborted=False,
    )


class _FakeBatchModel:
    def __init__(
        self,
        *,
        verify_overrides: dict[tuple[int, int], int] | None = None,
        refuse_primary: int | None = None,
        raise_commit_primary: int | None = None,
        fail_decode_primary: int | None = None,
    ) -> None:
        self.verify_overrides = verify_overrides or {}
        self.refuse_primary = refuse_primary
        self.raise_commit_primary = raise_commit_primary
        self.fail_decode_primary = fail_decode_primary
        self.draft_calls: list[tuple[int, ...]] = []
        self.target_calls: list[tuple[str, tuple[tuple[int, ...], ...]]] = []
        self.decode_cache_inputs: list[tuple[int | None, ...]] = []
        self.history_calls: list[tuple[int, ...]] = []
        self.commit_calls: list[tuple[int, int, int]] = []
        self.capture_rows: list[tuple[int, int]] = []

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
        mtp_cache[0][0] = mx.array(np.asarray(tokens, dtype=np.float32)[:, None, None])
        desired = np.asarray(tokens, dtype=np.int64) + 1
        logits = np.full(
            (len(tokens), 1, _VOCAB_SIZE),
            -100.0,
            dtype=np.float32,
        )
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
        tokens = np.asarray(inputs, dtype=np.int64)
        rows = tuple(tuple(int(token) for token in row) for row in tokens)
        phase = current_attention_phase()
        self.target_calls.append((phase, rows))
        if phase == "decode":
            slot = cache[0][0]
            self.decode_cache_inputs.append(
                ((None,) * len(rows))
                if slot is None
                else tuple(int(value) for value in np.asarray(slot).reshape(-1))
            )
            if self.fail_decode_primary is not None and any(
                row[0] == self.fail_decode_primary for row in rows
            ):
                raise RuntimeError("injected exact fallback failure")
            cache[0][0] = mx.array(
                np.asarray([row[-1] for row in rows], dtype=np.float32)[:, None, None]
            )
        logits = np.full(
            (tokens.shape[0], tokens.shape[1], _VOCAB_SIZE),
            -100.0,
            dtype=np.float32,
        )
        for row_index, row in enumerate(rows):
            primary = row[0]
            for token_index, token in enumerate(row):
                desired = self.verify_overrides.get(
                    (primary, token_index),
                    token + 1,
                )
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
        assert snapshot is not None
        rows = cache[0]._qwen4_exp_verify_rows
        ple = cache[0]._qwen4_exp_verify_ple
        primary = int(rows[0][0, 0, 0].item())
        ple_primary = int(ple[0][0, 0, 0].item())
        self.capture_rows.append((primary, ple_primary))
        self.commit_calls.append((primary, keep_tokens, verified_tokens))
        if primary == self.raise_commit_primary:
            raise RuntimeError("injected verified-window commit failure")
        if primary == self.refuse_primary:
            return False
        cache[0][0] = rows[0][:, keep_tokens - 1 : keep_tokens, :]
        return True

    @staticmethod
    def rollback_after_verify(
        cache: list[ArraysCache],
        snapshot: object,
        verified_tokens: int,
    ) -> None:
        assert verified_tokens > 0
        for entry, state in zip(cache, tuple(snapshot), strict=True):
            entry.state = _snapshot_value(state)

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
        mtp_cache[0][0] = mx.array(np.asarray(tokens, dtype=np.float32)[:, None, None])
        return hidden_states + 1


def _runtime(model: _FakeBatchModel) -> _Qwen4ExpTensorRuntime:
    runtime = object.__new__(_Qwen4ExpTensorRuntime)
    runtime.model = model
    runtime.capabilities = Qwen4ExpTensorCapabilities()
    runtime._stop_token_ids = frozenset({0})
    return runtime


def _assert_reservation_frontier_matches(
    actual: SimpleNamespace,
    expected: SimpleNamespace,
) -> None:
    assert actual.position == expected.position
    assert actual.output_tokens == expected.output_tokens
    assert actual.pending_primary == expected.pending_primary
    assert np.array_equal(np.asarray(actual.logits), np.asarray(expected.logits))
    assert np.array_equal(np.asarray(actual.hidden), np.asarray(expected.hidden))
    if actual.rng is None or expected.rng is None:
        assert actual.rng is expected.rng
    else:
        assert actual.rng.bit_generator.state == expected.rng.bit_generator.state
    for cache_name in ("cache", "mtp_cache"):
        actual_bundle = getattr(actual, cache_name)
        expected_bundle = getattr(expected, cache_name)
        for actual_entry, expected_entry in zip(
            actual_bundle,
            expected_bundle,
            strict=True,
        ):
            for actual_leaf, expected_leaf in zip(
                actual_entry.cache,
                expected_entry.cache,
                strict=True,
            ):
                if actual_leaf is None or expected_leaf is None:
                    assert actual_leaf is expected_leaf
                else:
                    assert np.array_equal(
                        np.asarray(actual_leaf),
                        np.asarray(expected_leaf),
                    )


def test_multirow_mtp_batches_recursive_drafts_and_matches_singleton_oracle() -> None:
    batch_model = _FakeBatchModel()
    batch_runtime = _runtime(batch_model)
    batch_rows = (_reservation(10), _reservation(20))

    outcomes = batch_runtime._decode_mtp_batch(batch_rows, draft_depth=2)

    oracle_rows = (_reservation(10), _reservation(20))
    oracle_outcomes = tuple(
        _runtime(_FakeBatchModel())._decode_mtp_one(row, draft_depth=2)
        for row in oracle_rows
    )
    assert outcomes == oracle_outcomes
    for actual, expected in zip(batch_rows, oracle_rows, strict=True):
        _assert_reservation_frontier_matches(actual, expected)
    assert batch_model.draft_calls == [(10, 20), (11, 21)]
    assert batch_model.target_calls == [("verify", ((10, 11, 12), (20, 21, 22)))]
    assert batch_model.history_calls == [(10, 20), (11, 21), (12, 22)]
    assert batch_model.commit_calls == [(10, 3, 3), (20, 3, 3)]
    assert batch_model.capture_rows == [(10, 10), (20, 20)]
    assert all(
        getattr(row.cache[0], attr, None) is None
        for row in batch_rows
        for attr in ("_qwen4_exp_verify_rows", "_qwen4_exp_verify_ple")
    )


def test_multirow_mtp_batches_mixed_rejection_corrections() -> None:
    model = _FakeBatchModel(verify_overrides={(20, 0): 30})
    runtime = _runtime(model)
    rows = (
        _reservation(10, stochastic=True, seed=11),
        _reservation(20, stochastic=True, seed=13),
    )

    outcomes = runtime._decode_mtp_batch(rows, draft_depth=2)

    assert outcomes[0].tokens == (10, 11, 12, 13)
    assert outcomes[0].mtp_accepted_tokens == 2
    assert outcomes[0].mtp_rejected_tokens == 0
    assert outcomes[1].tokens == (20, 30)
    assert outcomes[1].mtp_accepted_tokens == 0
    assert outcomes[1].mtp_rejected_tokens == 1
    assert model.draft_calls == [(10, 20), (11, 21)]
    assert model.target_calls == [
        ("verify", ((10, 11, 12), (20, 21, 22))),
        ("decode", ((30,),)),
    ]
    assert model.commit_calls == [(10, 3, 3), (20, 1, 3)]
    assert model.capture_rows == [(10, 10), (20, 20)]
    assert model.history_calls == [(10, 20), (11, 30), (12,)]


def test_multirow_mtp_commit_refusal_restores_then_matches_singleton_oracle() -> None:
    model = _FakeBatchModel(refuse_primary=20)
    runtime = _runtime(model)
    rows = (_reservation(10), _reservation(20))

    outcomes = runtime._decode_mtp_batch(rows, draft_depth=1)

    oracle_rows = (_reservation(10), _reservation(20))
    oracle_outcomes = tuple(
        _runtime(_FakeBatchModel(refuse_primary=20))._decode_mtp_one(
            row,
            draft_depth=1,
        )
        for row in oracle_rows
    )
    assert outcomes == oracle_outcomes
    for actual, expected in zip(rows, oracle_rows, strict=True):
        _assert_reservation_frontier_matches(actual, expected)
    assert model.target_calls == [
        ("verify", ((10, 11), (20, 21))),
        ("decode", ((10, 11), (20, 21))),
    ]
    assert model.history_calls == [(10, 20), (11, 21)]
    # Row 10 committed before row 20 refused. Exact fallback saw both entry caches.
    assert model.decode_cache_inputs == [(None, None)]
    assert all(
        getattr(row.cache[0], attr, None) is None
        for row in rows
        for attr in ("_qwen4_exp_verify_rows", "_qwen4_exp_verify_ple")
    )


def test_multirow_mtp_refusal_reforwards_stable_equal_width_groups() -> None:
    model = _FakeBatchModel(
        verify_overrides={(20, 0): 50, (30, 0): 51, (40, 1): 52},
        refuse_primary=20,
    )
    rows = tuple(_reservation(primary) for primary in (10, 20, 30, 40))

    outcomes = _runtime(model)._decode_mtp_batch(rows, draft_depth=2)

    assert tuple(outcome.tokens for outcome in outcomes) == (
        (10, 11, 12, 13),
        (20,),
        (30,),
        (40, 41),
    )
    assert model.target_calls == [
        ("verify", ((10, 11, 12), (20, 21, 22), (30, 31, 32), (40, 41, 42))),
        ("decode", ((10, 11, 12),)),
        ("decode", ((20,), (30,))),
        ("decode", ((40, 41),)),
    ]
    assert model.decode_cache_inputs == [(None,), (None, None), (None,)]
    assert model.history_calls == [(10, 20, 30, 40), (11, 41), (12,)]


def test_multirow_mtp_refusal_preserves_pending_primary_semantics() -> None:
    model = _FakeBatchModel(refuse_primary=20)
    rows = (_reservation(10), _reservation(20))
    oracle_rows = (_reservation(10), _reservation(20))
    for row in (*rows, *oracle_rows):
        row.pending_primary = int(np.argmax(np.asarray(row.logits)))
        row.output_tokens = 1
        row.position = 6

    outcomes = _runtime(model)._decode_mtp_batch(rows, draft_depth=1)
    oracle_outcomes = tuple(
        _runtime(_FakeBatchModel(refuse_primary=20))._decode_mtp_one(
            row,
            draft_depth=1,
        )
        for row in oracle_rows
    )

    assert outcomes == oracle_outcomes
    for actual, expected in zip(rows, oracle_rows, strict=True):
        _assert_reservation_frontier_matches(actual, expected)


def test_seeded_stochastic_refusal_matches_singleton_rng_and_frontier() -> None:
    model = _FakeBatchModel(
        verify_overrides={(20, 0): 30},
        refuse_primary=20,
    )
    rows = (
        _reservation(10, stochastic=True, seed=11),
        _reservation(20, stochastic=True, seed=13),
    )

    outcomes = _runtime(model)._decode_mtp_batch(rows, draft_depth=2)

    oracle_rows = (
        _reservation(10, stochastic=True, seed=11),
        _reservation(20, stochastic=True, seed=13),
    )
    oracle_outcomes = tuple(
        _runtime(
            _FakeBatchModel(
                verify_overrides={(20, 0): 30},
                refuse_primary=20,
            )
        )._decode_mtp_one(row, draft_depth=2)
        for row in oracle_rows
    )
    assert outcomes == oracle_outcomes
    for actual, expected in zip(rows, oracle_rows, strict=True):
        _assert_reservation_frontier_matches(actual, expected)
    assert model.target_calls == [
        ("verify", ((10, 11, 12), (20, 21, 22))),
        ("decode", ((10, 11, 12),)),
        ("decode", ((20, 30),)),
    ]


def test_commit_exception_restores_every_row_entry_state() -> None:
    model = _FakeBatchModel(raise_commit_primary=20)
    rows = (
        _reservation(10, stochastic=True, seed=11),
        _reservation(20, stochastic=True, seed=13),
    )
    entry_rows = (
        _reservation(10, stochastic=True, seed=11),
        _reservation(20, stochastic=True, seed=13),
    )
    rows[0].pending_primary = 10
    rows[0].output_tokens = 1
    rows[0].position = 6
    entry_rows[0].pending_primary = 10
    entry_rows[0].output_tokens = 1
    entry_rows[0].position = 6

    with pytest.raises(
        RuntimeError,
        match="injected verified-window commit failure",
    ):
        _runtime(model)._decode_mtp_batch(rows, draft_depth=2)

    for actual, expected in zip(rows, entry_rows, strict=True):
        _assert_reservation_frontier_matches(actual, expected)
    assert all(
        getattr(row.cache[0], attr, None) is None
        for row in rows
        for attr in ("_qwen4_exp_verify_rows", "_qwen4_exp_verify_ple")
    )


def test_fallback_failure_does_not_publish_staged_terminal_row() -> None:
    model = _FakeBatchModel(refuse_primary=20, fail_decode_primary=20)
    rows = (
        _reservation(5, max_output_tokens=1),
        _reservation(10, stochastic=True, seed=11),
        _reservation(20, stochastic=True, seed=13),
    )
    entry_rows = (
        _reservation(5, max_output_tokens=1),
        _reservation(10, stochastic=True, seed=11),
        _reservation(20, stochastic=True, seed=13),
    )

    with pytest.raises(RuntimeError, match="injected exact fallback failure"):
        _runtime(model)._decode_mtp_batch(rows, draft_depth=1)

    for actual, expected in zip(rows, entry_rows, strict=True):
        _assert_reservation_frontier_matches(actual, expected)
    assert all(
        getattr(row.cache[0], attr, None) is None
        for row in rows
        for attr in ("_qwen4_exp_verify_rows", "_qwen4_exp_verify_ple")
    )


def test_terminal_and_omitted_cancelled_rows_do_not_perturb_survivor() -> None:
    model = _FakeBatchModel()
    runtime = _runtime(model)
    terminal = _reservation(5, max_output_tokens=1)
    cancelled = _reservation(40)
    cancelled.aborted = True
    survivor = _reservation(20)
    oracle = _reservation(20)

    outcomes = runtime._decode_mtp_batch((terminal, survivor), draft_depth=2)
    oracle_outcome = _runtime(_FakeBatchModel())._decode_mtp_one(
        oracle,
        draft_depth=2,
    )

    assert outcomes[0].tokens == (5,)
    assert outcomes[0].finished
    assert outcomes[1] == oracle_outcome
    _assert_reservation_frontier_matches(survivor, oracle)
    assert cancelled.output_tokens == 0
    assert cancelled.position == 5
    assert model.draft_calls == [(20,), (21,)]
    assert model.target_calls == [("verify", ((20, 21, 22),))]
