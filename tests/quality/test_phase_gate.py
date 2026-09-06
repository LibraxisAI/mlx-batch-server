from __future__ import annotations

from pathlib import Path

import pytest
from scripts.quality import phase_gate

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / ".vibecrafted/embargo.toml"
CONFIG = ROOT / ".pre-commit-config.yaml"
SOURCE = ROOT / "scripts/quality/phase_gate.py"


def _state(
    *,
    schema: str = phase_gate.STATE_SCHEMA,
    plan_id: str = phase_gate.PLAN_ID,
    phase: str = phase_gate.W1_PHASE,
    deferred: tuple[str, ...] = phase_gate.DEFERRED_GATES,
    recovery_ref: str = phase_gate.RECOVERY_REF,
    extra: str = "",
) -> str:
    values = ", ".join(f'"{item}"' for item in deferred)
    return (
        f'schema = "{schema}"\n'
        f'plan_id = "{plan_id}"\n'
        f'phase = "{phase}"\n'
        f"deferred_gates = [{values}]\n"
        f'recovery_ref = "{recovery_ref}"\n'
        f"{extra}"
    )


def test_tracked_state_is_the_exact_w1_contract() -> None:
    """The historical W1 gate node follows the exact committed phase receipt."""

    expected = _state(phase=phase_gate.RELEASE_PHASE, deferred=())

    assert STATE.read_text(encoding="utf-8") == expected
    assert phase_gate.load_index_state() == phase_gate.parse_state_text(expected)


@pytest.mark.parametrize("phase", (phase_gate.W1_PHASE, phase_gate.W2_PHASE))
def test_open_phases_defer_exact_allowlist(phase: str) -> None:
    state = phase_gate.parse_state_text(_state(phase=phase))

    deferred = tuple(
        gate
        for gate in phase_gate.DEFERRED_GATES
        if phase_gate.decide_gate(gate, state) is phase_gate.GateDecision.DEFER
    )
    assert deferred == phase_gate.DEFERRED_GATES


@pytest.mark.parametrize("gate", phase_gate.NON_DEFERRED_PROBES)
def test_security_provenance_and_hygiene_never_defer(gate: str) -> None:
    state = phase_gate.parse_state_text(_state())

    assert phase_gate.decide_gate(gate, state) is phase_gate.GateDecision.RUN


def test_w2_structural_close_releases_every_gate() -> None:
    state = phase_gate.parse_state_text(
        _state(phase=phase_gate.RELEASE_PHASE, deferred=())
    )

    for gate in (*phase_gate.DEFERRED_GATES, *phase_gate.NON_DEFERRED_PROBES):
        assert phase_gate.decide_gate(gate, state) is phase_gate.GateDecision.RUN
    phase_gate.require_closed_at_ref_boundary(state, surface="test")


@pytest.mark.parametrize(
    "raw",
    (
        "not-toml",
        _state(schema="another-schema"),
        _state(plan_id="another-plan"),
        _state(recovery_ref="0" * 40),
        _state(phase="UNKNOWN"),
        _state(deferred=()),
        _state(deferred=(*phase_gate.DEFERRED_GATES, "bandit")),
        _state(phase=phase_gate.RELEASE_PHASE),
        _state(extra='extra = "forbidden"\n'),
    ),
)
def test_tampered_or_ambiguous_state_fails_closed(raw: str) -> None:
    with pytest.raises(phase_gate.PhaseGateError):
        phase_gate.parse_state_text(raw)


@pytest.mark.parametrize("phase", (phase_gate.W1_PHASE, phase_gate.W2_PHASE))
def test_pre_push_boundary_rejects_open_phase(phase: str) -> None:
    state = phase_gate.parse_state_text(_state(phase=phase))

    with pytest.raises(phase_gate.PhaseGateError, match="forbidden"):
        phase_gate.require_closed_at_ref_boundary(state, surface="pre-push")


def test_head_loader_uses_committed_state(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []

    def fake_git_blob(spec: str, *, source: str) -> bytes:
        calls.append((spec, source))
        return _state(phase=phase_gate.RELEASE_PHASE, deferred=()).encode()

    monkeypatch.setattr(phase_gate, "_git_blob", fake_git_blob)

    assert phase_gate.load_head_state().phase == phase_gate.RELEASE_PHASE
    assert calls == [("HEAD:.vibecrafted/embargo.toml", "HEAD")]


def test_index_worktree_mismatch_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_path = tmp_path / "embargo.toml"
    state_path.write_text(_state(), encoding="utf-8")
    monkeypatch.setattr(phase_gate, "STATE_PATH", state_path)
    monkeypatch.setattr(
        phase_gate,
        "_git_blob",
        lambda spec, *, source: _state(plan_id="another-plan").encode(),
    )

    with pytest.raises(phase_gate.PhaseGateError, match="differs"):
        phase_gate.load_index_state()


def test_missing_or_symlink_state_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "missing.toml"
    monkeypatch.setattr(phase_gate, "STATE_PATH", missing)
    with pytest.raises(phase_gate.PhaseGateError, match="missing"):
        phase_gate.load_index_state()

    target = tmp_path / "target.toml"
    target.write_text(_state(), encoding="utf-8")
    missing.symlink_to(target)
    with pytest.raises(phase_gate.PhaseGateError, match="non-symlink"):
        phase_gate.load_index_state()


def test_pre_commit_wiring_wraps_only_three_deferred_hooks() -> None:
    config = CONFIG.read_text(encoding="utf-8")

    for gate in phase_gate.DEFERRED_GATES:
        assert config.count(f"--gate {gate} --") == 1
    for gate in phase_gate.NON_DEFERRED_PROBES:
        assert f"--gate {gate} --" not in config
    assert "--validate-state" in config
    assert "--forbid-open --surface pre-push" in config


def test_env_marker_policy_is_removed() -> None:
    assert "MLX_BATCH_EMBARGO" not in SOURCE.read_text(encoding="utf-8")
    assert "MLX_BATCH_EMBARGO" not in CONFIG.read_text(encoding="utf-8")
