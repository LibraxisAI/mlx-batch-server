from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from scripts.quality.verify_mlx_batch_api_contract import evaluate_section

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/quality/verify_mlx_batch_api_contract.py"
EXPECTED_MARKERS = (
    "OPENAI_SOURCE_CONTRACT=red",
    "ANTHROPIC_SOURCE_CONTRACT=red",
    "SAFE_PUBLIC_FETCH_SOURCE_CONTRACT=red",
    "MULTIROW_SOURCE_CONTRACT=green",
)
TENSOR_RELATIVE = Path("src/mlx_batch_server/runtime/fusion/qwen4_exp/model/tensor.py")


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args, "--no-imports"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_all_contracts_are_named_and_report_independent_states() -> None:
    result = _run("--all", "--expect", "red")

    assert result.returncode == 1, result.stderr
    for marker in EXPECTED_MARKERS:
        assert marker in result.stdout
    assert result.stdout.count("  missing:") >= 3


@pytest.mark.parametrize(
    ("section", "reason"),
    (
        ("openai", "compact-endpoint"),
        ("anthropic", "shared-runtime-owner"),
        ("safe-fetch", "owned-interface"),
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


def test_multirow_green_requires_the_concrete_recursive_mtp_seam() -> None:
    result = _run("--section", "multirow", "--expect", "green")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "MULTIROW_SOURCE_CONTRACT=green" in result.stdout
    assert "batched-recursive-drafts" in result.stdout
    assert "shared-verify-and-variable-commit" in result.stdout


def _mutated_multirow_result(tmp_path: Path, source: str):
    tensor = tmp_path / TENSOR_RELATIVE
    tensor.parent.mkdir(parents=True)
    tensor.write_text(source, encoding="utf-8")
    return evaluate_section(tmp_path, "multirow")


def test_multirow_verifier_rejects_missing_decode_mtp_call(tmp_path: Path) -> None:
    source = (ROOT / TENSOR_RELATIVE).read_text(encoding="utf-8")
    mutated = source.replace(
        "self._decode_mtp_batch(",
        "self._missing_decode_mtp_batch(",
        1,
    )

    result = _mutated_multirow_result(tmp_path, mutated)

    assert not result.green
    assert any(
        "_decode_batch does not route admitted cohorts" in failure
        for failure in result.failures
    )


def test_multirow_verifier_rejects_restored_explicit_fallback(
    tmp_path: Path,
) -> None:
    source = (ROOT / TENSOR_RELATIVE).read_text(encoding="utf-8")
    mutated = source.replace(
        "        if not reservations:\n",
        "        fallback = MtpDisableReason.MULTIROW_NOT_PROVEN\n"
        "        if not reservations:\n",
        1,
    )

    result = _mutated_multirow_result(tmp_path, mutated)

    assert not result.green
    assert any(
        "_decode_batch does not route admitted cohorts" in failure
        for failure in result.failures
    )


def test_multirow_verifier_rejects_missing_batched_draft_forward(
    tmp_path: Path,
) -> None:
    source = (ROOT / TENSOR_RELATIVE).read_text(encoding="utf-8")
    prefix, method = source.split("    def _mtp_forward_batch(\n", 1)
    mutated = (
        prefix
        + "    def _mtp_forward_batch(\n"
        + method.replace(
            "self.model.mtp_forward(",
            "self.model.missing_mtp_forward(",
            1,
        )
    )

    result = _mutated_multirow_result(tmp_path, mutated)

    assert not result.green
    assert any("one batched MTP model call" in failure for failure in result.failures)


def test_multirow_verifier_rejects_missing_verified_window_commit(
    tmp_path: Path,
) -> None:
    source = (ROOT / TENSOR_RELATIVE).read_text(encoding="utf-8")
    prefix, method = source.split("    def _decode_mtp_batch(\n", 1)
    mutated = (
        prefix
        + "    def _decode_mtp_batch(\n"
        + method.replace(
            "self.model.commit_verified_window(",
            "self.model.missing_verified_window_commit(",
            1,
        )
    )

    result = _mutated_multirow_result(tmp_path, mutated)

    assert not result.green
    assert any(
        "row-local verified-window commits" in failure for failure in result.failures
    )
