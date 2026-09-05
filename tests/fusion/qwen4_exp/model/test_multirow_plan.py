from __future__ import annotations

from mlx_batch_server.runtime.fusion.qwen4_exp.model.multirow import (
    MultirowBatchPlan,
)


class _FakeModel:
    def __init__(self, tokens: tuple[int, ...]) -> None:
        self.tokens = tokens
        self.calls = 0
        self.seen_ordinals: tuple[int, ...] = ()

    def forward(self, ordinals: tuple[int, ...]) -> tuple[int, ...]:
        self.calls += 1
        self.seen_ordinals = ordinals
        return tuple(self.tokens[ordinal] + 1 for ordinal in ordinals)


def test_two_active_rows_use_one_fake_model_call_and_match_serial_reference() -> None:
    tokens = (10, 20, 30)
    model = _FakeModel(tokens)
    plan = MultirowBatchPlan[int].compact((True, False, True))

    actual = plan.execute(model.forward)
    serial_reference = tuple(token + 1 for token in tokens)

    assert model.calls == 1
    assert model.seen_ordinals == (0, 2)
    assert actual == (serial_reference[0], None, serial_reference[2])


def test_finished_and_cancelled_rows_compact_without_identity_drift() -> None:
    row_ids = ("row-a", "row-b", "row-c", "row-d")
    cancelled = {"row-b"}
    finished = {"row-c"}
    active = tuple(
        row_id not in cancelled and row_id not in finished for row_id in row_ids
    )
    model = _FakeModel((3, 5, 7, 11))

    scattered = MultirowBatchPlan[int].compact(active).execute(model.forward)

    assert model.calls == 1
    assert model.seen_ordinals == (0, 3)
    assert dict(zip(row_ids, scattered, strict=True)) == {
        "row-a": 4,
        "row-b": None,
        "row-c": None,
        "row-d": 12,
    }


def test_empty_active_set_does_not_call_fake_model() -> None:
    model = _FakeModel((1, 2))

    assert MultirowBatchPlan[int].compact((False, False)).execute(model.forward) == (
        None,
        None,
    )
    assert model.calls == 0
