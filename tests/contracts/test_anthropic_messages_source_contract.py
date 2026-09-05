"""RED source contract for the W1 Anthropic Messages cut."""

from pathlib import Path

from scripts.quality.verify_mlx_batch_api_contract import evaluate_section

ROOT = Path(__file__).resolve().parents[2]


def test_anthropic_messages_source_contract_is_green() -> None:
    result = evaluate_section(ROOT, "anthropic")

    assert result.green, "\n".join(result.failures)
