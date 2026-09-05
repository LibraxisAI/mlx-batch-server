"""RED source contract for the W1 fused multirow tensor cut."""

from pathlib import Path

from scripts.quality.verify_mlx_batch_api_contract import evaluate_section

ROOT = Path(__file__).resolve().parents[2]


def test_multirow_tensor_source_contract_is_green() -> None:
    result = evaluate_section(ROOT, "multirow")

    assert result.green, "\n".join(result.failures)
