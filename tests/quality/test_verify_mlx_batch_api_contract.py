from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from shutil import copy2, copytree

import pytest
from scripts.quality.verify_mlx_batch_api_contract import evaluate_section

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/quality/verify_mlx_batch_api_contract.py"
EXPECTED_MARKERS = (
    "OPENAI_SOURCE_CONTRACT=green",
    "ANTHROPIC_SOURCE_CONTRACT=green",
    "SAFE_PUBLIC_FETCH_SOURCE_CONTRACT=green",
    "MULTIROW_SOURCE_CONTRACT=green",
)
TENSOR_RELATIVE = Path("src/mlx_batch_server/runtime/fusion/qwen4_exp/model/tensor.py")
ANTHROPIC_RELATIVE = Path("src/mlx_batch_server/chat/anthropic")


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
    assert "  missing:" not in result.stdout


@pytest.mark.parametrize(
    ("section", "reason"),
    (
        ("openai", "compact-endpoint"),
        ("anthropic", "shared-runtime-owner"),
        ("safe-fetch", "owned-interface"),
    ),
)
def test_each_section_exposes_its_deterministic_green_reason(
    section: str,
    reason: str,
) -> None:
    result = _run("--section", section, "--expect", "green")

    assert result.returncode == 0, result.stderr
    assert reason in result.stdout


def test_red_expectation_rejects_the_green_baseline() -> None:
    result = _run("--section", "openai", "--expect", "red")

    assert result.returncode == 1
    assert "OPENAI_SOURCE_CONTRACT=green" in result.stdout


def test_multirow_green_requires_the_concrete_recursive_mtp_seam() -> None:
    result = _run("--section", "multirow", "--expect", "green")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "MULTIROW_SOURCE_CONTRACT=green" in result.stdout
    assert "batched-recursive-drafts" in result.stdout
    assert "shared-verify-and-variable-commit" in result.stdout


def _mutated_anthropic_result(
    tmp_path: Path,
    relative: str,
    before: str,
    after: str,
):
    target = tmp_path / ANTHROPIC_RELATIVE
    copytree(ROOT / ANTHROPIC_RELATIVE, target)
    rate_limit = tmp_path / "src/mlx_batch_server/auth/rate_limit.py"
    rate_limit.parent.mkdir(parents=True, exist_ok=True)
    copy2(ROOT / "src/mlx_batch_server/auth/rate_limit.py", rate_limit)
    if relative.startswith("auth/"):
        path = tmp_path / "src/mlx_batch_server" / relative
    else:
        path = target / relative
    source = path.read_text(encoding="utf-8")
    assert before in source
    path.write_text(source.replace(before, after, 1), encoding="utf-8")
    return evaluate_section(tmp_path, "anthropic")


@pytest.mark.parametrize(
    ("check_name", "relative", "before", "after"),
    (
        (
            "single-capability-owner",
            "capabilities.py",
            "def enforce_capabilities(\n",
            "def enforce_capabilities_legacy(\n",
        ),
        (
            "pre-sse-capability-rejection",
            "router.py",
            "admission = enforce_capabilities(request, profile)",
            "admission = bypass_capabilities(request, profile)",
        ),
        (
            "thinking-signature-gate",
            "capabilities.py",
            '"thinking.enabled",',
            '"thinking.unverified",',
        ),
        (
            "ordered-rich-content-mapping",
            "request_mapper.py",
            "canonical = map_anthropic_content(\n",
            "canonical = map_anthropic_content_unordered(\n",
        ),
        (
            "explicit-no-silent-ignore",
            "capabilities.py",
            '"output_config.format", "structured-output execution"',
            '"output_config.unclassified", "structured-output execution"',
        ),
        (
            "protocol-aware-global-rate-limit",
            "auth/rate_limit.py",
            "return _rate_limit_response(\n                request,",
            "return _legacy_rate_limit_response(\n                request,",
        ),
        (
            "protocol-aware-global-rate-limit",
            "auth/rate_limit.py",
            "headers={**headers, REQUEST_ID_HEADER: request_id}",
            "headers=headers",
        ),
        (
            "safe-protocol-boundary-failures",
            "router.py",
            'AnthropicAPIError("authentication service unavailable")',
            "AnthropicAPIError(str(error))",
        ),
        (
            "safe-protocol-boundary-failures",
            "router.py",
            'AnthropicAPIError("request body could not be read")',
            "AnthropicAPIError(str(error))",
        ),
        (
            "complete-physical-header-validation",
            "router.py",
            'http_request.headers.getlist("anthropic-version")',
            '[http_request.headers.get("anthropic-version")]',
        ),
        (
            "complete-physical-header-validation",
            "router.py",
            'http_request.headers.getlist("anthropic-beta")',
            '[http_request.headers.get("anthropic-beta")]',
        ),
        (
            "closed-auth-status-map",
            "router.py",
            '400: ("invalid_request_error", "authentication request is invalid")',
            '400: ("api_error", "authentication request is invalid")',
        ),
        (
            "protocol-aware-global-rate-limit",
            "auth/rate_limit.py",
            'if request.method == "POST" and request.url.path in _ANTHROPIC_MESSAGES_PATHS:',
            'if False and request.method == "POST" and request.url.path in _ANTHROPIC_MESSAGES_PATHS:',
        ),
        (
            "safe-protocol-boundary-failures",
            "router.py",
            "raise _ProtocolAuthError(error) from error\n"
            "    except Exception as error:\n"
            "        logger.error(",
            "raise _ProtocolAuthError(error) from error\n"
            "    except Exception as error:\n"
            "        raise\n"
            "        logger.error(",
        ),
        (
            "complete-physical-header-validation",
            "router.py",
            "if len(version_values) != 1:",
            "if False and len(version_values) != 1:",
        ),
        (
            "closed-auth-status-map",
            "router.py",
            "admitted = _AUTH_FAILURES.get(error.status_code)",
            "admitted = {}.get(error.status_code)",
        ),
        (
            "safe-protocol-boundary-failures",
            "messages_engine.py",
            'projector.fail("api_error", diagnostic=type(error).__name__)',
            'projector.fail("api_error", diagnostic=str(error))',
        ),
        (
            "safe-protocol-boundary-failures",
            "runtime_source.py",
            'error="message generation failed",',
            "error=str(error),",
        ),
        (
            "safe-protocol-boundary-failures",
            "projector.py",
            "message=public.message,",
            "message=event.error,",
        ),
        (
            "safe-protocol-boundary-failures",
            "projector.py",
            "def _public_runtime_failure(event: TurnFailed) -> _PublicRuntimeFailure:\n"
            "    admitted = _PUBLIC_RUNTIME_FAILURES.get(event.code)",
            "def _public_runtime_failure(event: TurnFailed) -> _PublicRuntimeFailure:\n"
            "    if event.error:\n"
            "        return _PublicRuntimeFailure(\n"
            '            error_type="api_error", message=event.error\n'
            "        )\n"
            "    admitted = _PUBLIC_RUNTIME_FAILURES.get(event.code)",
        ),
        (
            "safe-protocol-boundary-failures",
            "router.py",
            "        self._public_projection = (error_type, message, status_code)",
            "        self.message = str(error.detail)\n"
            "        self._public_projection = (error_type, message, status_code)",
        ),
        (
            "safe-protocol-boundary-failures",
            "messages_engine.py",
            '            projector.fail("api_error", diagnostic=type(error).__name__)',
            '            projector.fail("api_error", diagnostic=type(error).__name__)\n'
            "            projector._failed = StreamErrorBody(\n"
            '                type="api_error", message=error.args[0]\n'
            "            )",
        ),
    ),
)
def test_anthropic_verifier_independently_falsifies_each_admission_clause(
    tmp_path: Path,
    check_name: str,
    relative: str,
    before: str,
    after: str,
) -> None:
    baseline = evaluate_section(ROOT, "anthropic")
    assert baseline.green, baseline.failures

    result = _mutated_anthropic_result(tmp_path, relative, before, after)

    checks = {check.name: check.passed for check in result.checks}
    assert checks[check_name] is False
    assert sum(not passed for passed in checks.values()) == 1, checks


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


@pytest.mark.parametrize(
    "method_name",
    (
        "_target_forward_batch",
        "_mtp_forward_batch",
        "_mtp_update_batch",
        "_decode_mtp_batch",
        "_decode_mtp_one",
    ),
)
def test_multirow_verifier_rejects_missing_physical_recorders(
    tmp_path: Path,
    method_name: str,
) -> None:
    source = (ROOT / TENSOR_RELATIVE).read_text(encoding="utf-8")
    prefix, method = source.split(f"    def {method_name}(\n", 1)
    mutated = (
        prefix
        + f"    def {method_name}(\n"
        + method.replace(
            "self._record_tensor_forward(",
            "self._missing_tensor_forward_recorder(",
            1,
        )
    )

    result = _mutated_multirow_result(tmp_path, mutated)

    assert not result.green
    assert any(
        "completion recorders are missing" in failure for failure in result.failures
    )


@pytest.mark.parametrize(
    ("method_name", "assignment"),
    (
        ("_target_forward_batch", "logits, hidden = self.model("),
        ("_mtp_forward_batch", "logits, next_hidden = self.model.mtp_forward("),
        ("_mtp_update_batch", "mtp_hidden = self.model.mtp_update_cache("),
    ),
)
def test_multirow_verifier_rejects_per_row_physical_model_calls(
    tmp_path: Path,
    method_name: str,
    assignment: str,
) -> None:
    source = (ROOT / TENSOR_RELATIVE).read_text(encoding="utf-8")
    prefix, method = source.split(f"    def {method_name}(\n", 1)
    mutated = (
        prefix
        + f"    def {method_name}(\n"
        + method.replace(
            f"            {assignment}\n",
            f"            for _reservation in reservations:\n"
            f"                {assignment}\n",
            1,
        )
    )

    result = _mutated_multirow_result(tmp_path, mutated)

    assert not result.green
    assert any(
        "model forward" in failure or "model call" in failure or "row-serial" in failure
        for failure in result.failures
    )
