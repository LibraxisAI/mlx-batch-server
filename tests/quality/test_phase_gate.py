from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/quality/phase_gate.py"
CONFIG = ROOT / ".pre-commit-config.yaml"
MARKER_ENV = "MLX_BATCH_EMBARGO_V1"
DEFERRED = ("mypy", "ruff", "ruff-format")
FORBIDDEN = (
    "bandit",
    "semgrep",
    "detect-private-key",
    "check-ast",
    "check-merge-conflict",
    "merge-markers-block",
    "check-added-large-files",
    "commit-msg",
    "pre-push",
    "ref-safety",
)


def _marker(**changes: object) -> str:
    payload: dict[str, object] = {
        "schema": "mlx-batch-compile-embargo.v1",
        "plan_id": "mlx-batch-api-conformance-v1",
        "phase": "W1-source-shape",
        "deferred_gates": list(DEFERRED),
        "release_attestation": "I1_STRUCTURALLY_CLOSED",
    }
    payload.update(changes)
    return json.dumps(payload)


def _run(*args: str, marker: str | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if marker is None:
        env.pop(MARKER_ENV, None)
    else:
        env[MARKER_ENV] = marker
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def _probe_command(gate: str) -> tuple[str, ...]:
    commands = {
        "ruff": (
            "ruff",
            "check",
            "--output-format",
            "json",
            "scripts/quality/phase_gate.py",
        ),
        "ruff-format": (
            "ruff",
            "format",
            "--check",
            "scripts/quality/phase_gate.py",
        ),
        "mypy": ("mypy", "--version"),
        "bandit": ("bandit", "--version"),
    }
    return ("--gate", gate, "--", *commands.get(gate, ("bandit", "--version")))


def test_no_marker_runs_the_ordinary_gate_command() -> None:
    result = _run(*_probe_command("ruff"))

    assert result.returncode == 0
    assert "PHASE_GATE=run gate=ruff" in result.stdout
    assert "[]" in result.stdout


@pytest.mark.parametrize("gate", DEFERRED)
def test_valid_w1_marker_defers_exact_allowlist(gate: str) -> None:
    result = _run(*_probe_command(gate), marker=_marker())

    assert result.returncode == 0
    assert f"PHASE_GATE=deferred gate={gate}" in result.stdout


@pytest.mark.parametrize(
    "marker",
    (
        "not-json",
        _marker(plan_id="another-plan"),
        _marker(phase="I1"),
        _marker(release_attestation="release-now"),
        _marker(extra="not-allowed"),
    ),
)
def test_marker_tampering_fails_closed(marker: str) -> None:
    result = _run(*_probe_command("ruff"), marker=marker)

    assert result.returncode == 2
    assert "PHASE_GATE=blocked" in result.stderr


@pytest.mark.parametrize("gate", FORBIDDEN)
def test_forbidden_gate_cannot_be_added_to_deferral(gate: str) -> None:
    attempted = _marker(deferred_gates=[*DEFERRED, gate])
    result = _run(*_probe_command(gate), marker=attempted)

    assert result.returncode == 2
    assert "does not exactly attest" in result.stderr


def test_valid_marker_runs_a_non_deferred_security_gate() -> None:
    result = _run(*_probe_command("bandit"), marker=_marker())

    assert result.returncode == 0
    assert "PHASE_GATE=run gate=bandit" in result.stdout
    assert "bandit " in result.stdout.lower()


def test_pre_push_ref_boundary_rejects_even_valid_marker() -> None:
    result = _run(
        "--forbid-marker",
        "--surface",
        "pre-push",
        marker=_marker(),
    )

    assert result.returncode == 2
    assert "forbidden at the pre-push ref boundary" in result.stderr


def test_pre_commit_wiring_wraps_only_the_three_deferred_hooks() -> None:
    config = CONFIG.read_text(encoding="utf-8")

    for gate in DEFERRED:
        assert f"--gate {gate} --" in config
    for gate in FORBIDDEN[:7]:
        assert f"--gate {gate} --" not in config
    assert "compile-embargo-marker-policy" in config
    assert "compile-embargo-ref-guard" in config
