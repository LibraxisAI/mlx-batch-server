from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/quality/verify_mlx_batch_api_contract.py"
EXPECTED_MARKERS = (
    "OPENAI_SOURCE_CONTRACT=red",
    "ANTHROPIC_SOURCE_CONTRACT=red",
    "SAFE_PUBLIC_FETCH_SOURCE_CONTRACT=red",
    "MULTIROW_SOURCE_CONTRACT=red",
)


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args, "--no-imports"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_all_baseline_contracts_are_named_and_red_for_a_missing_capability() -> None:
    result = _run("--all", "--expect", "red")

    assert result.returncode == 0, result.stderr
    for marker in EXPECTED_MARKERS:
        assert marker in result.stdout
    assert result.stdout.count("  missing:") >= 4


@pytest.mark.parametrize(
    ("section", "reason"),
    (
        ("openai", "compact-endpoint"),
        ("anthropic", "shared-runtime-owner"),
        ("safe-fetch", "owned-interface"),
        ("multirow", "no-row-serial-model-loop"),
    ),
)
def test_each_section_exposes_its_deterministic_red_reason(
    section: str,
    reason: str,
) -> None:
    result = _run("--section", section, "--expect", "red")

    assert result.returncode == 0, result.stderr
    assert reason in result.stdout


def test_green_expectation_rejects_the_red_baseline() -> None:
    result = _run("--section", "openai", "--expect", "green")

    assert result.returncode == 1
    assert "OPENAI_SOURCE_CONTRACT=red" in result.stdout
