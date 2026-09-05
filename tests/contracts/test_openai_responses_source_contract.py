"""RED source contract for the W1 OpenAI Responses cut."""

from pathlib import Path

from scripts.quality.verify_mlx_batch_api_contract import evaluate_section

ROOT = Path(__file__).resolve().parents[2]


def test_openai_responses_source_contract_is_green() -> None:
    result = evaluate_section(ROOT, "openai")

    assert result.green, "\n".join(result.failures)
