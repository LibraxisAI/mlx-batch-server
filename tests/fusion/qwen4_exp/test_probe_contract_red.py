"""RED contracts for the static Flash-Next family probe."""

from mlx_batch_server.runtime.contracts import BackendKind, ModelSpec
from mlx_batch_server.runtime.fusion.qwen4_exp import probe_qwen4_exp


def test_flash_next_probe_reports_only_statically_proven_capabilities() -> None:
    report = probe_qwen4_exp(
        ModelSpec(
            model_id="grant-ai/Qwen3.8-Flash-Next-Abliterated-MLX-4bit",
            revision="000544f8cddcbde27c1bc302deac2b5b4d45a5b1",
            architecture="Qwen4ExpForConditionalGeneration",
            model_type="qwen4_exp",
            quantization="4bit",
        )
    )

    assert report.supported is True
    assert report.backend is BackendKind.FUSED_MTP_MLX
    assert report.text is True
    assert report.vision is True
    assert report.tools is True
    assert report.mtp is False
    assert report.continuous_batching is False
    assert report.cache_modes == ()
    assert report.facts["mtp_contract_available"] is True
    assert report.facts["mtp_runtime_proven"] is False
    assert report.facts["continuous_batching_contract_available"] is True
    assert report.facts["continuous_batching_runtime_proven"] is False


def test_probe_rejects_a_different_qwen_family() -> None:
    report = probe_qwen4_exp(
        ModelSpec(
            model_id="some/qwen3-next",
            architecture="Qwen3NextForCausalLM",
            model_type="qwen3_next",
        )
    )

    assert report.supported is False
    assert report.mtp is False
    assert report.rejection_reasons == ("unsupported_qwen4_exp_family",)
