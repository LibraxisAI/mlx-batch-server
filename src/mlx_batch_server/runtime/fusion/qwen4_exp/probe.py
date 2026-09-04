"""Static Qwen4Exp capability probe with no model imports or loading.

Family recognition and MTP degradation semantics are adapted from MTPLX's
``models/qwen4_exp.py``. Continuous text/VLM batching capability comes from
the oMLX scheduler chassis; neither donor's HTTP control plane is imported.
"""

from __future__ import annotations

from ...contracts import BackendKind, CapabilityReport, ModelSpec

_MODEL_TYPES = {"qwen4_exp", "qwen4_exp_text"}
_ARCHITECTURES = {
    "qwen4expforcausallm",
    "qwen4expforconditionalgeneration",
    "qwen4expmodel",
}


def probe_qwen4_exp(model: ModelSpec) -> CapabilityReport:
    model_type = (model.model_type or "").strip().lower()
    architecture = (model.architecture or "").strip()
    architecture_key = architecture.lower()
    supported = model_type in _MODEL_TYPES or architecture_key in _ARCHITECTURES
    if not supported:
        return CapabilityReport(
            supported=False,
            backend=BackendKind.FUSED_MTP_MLX,
            architecture=model.architecture,
            text=False,
            mtp=False,
            continuous_batching=False,
            rejection_reasons=("unsupported_qwen4_exp_family",),
            facts={"model_type": model.model_type},
        )

    is_conditional = architecture_key == "qwen4expforconditionalgeneration"
    has_vision_metadata = bool(model.metadata.get("vision_config"))
    vision = is_conditional or has_vision_metadata or model_type == "qwen4_exp"
    return CapabilityReport(
        supported=True,
        backend=BackendKind.FUSED_MTP_MLX,
        architecture=model.architecture or "Qwen4Exp",
        text=True,
        vision=vision,
        tools=True,
        mtp=False,
        continuous_batching=False,
        cache_modes=(),
        facts={
            "model_family": "qwen4_exp",
            "mtp_head_required_at_load": True,
            "mtp_contract_available": True,
            "mtp_runtime_proven": False,
            "mtp_multirow_requires_live_proof": True,
            "continuous_batching_contract_available": True,
            "continuous_batching_runtime_proven": False,
            "paged_prefix_ssd_contract_available": True,
            "vision_prefill_cap_initial": 2,
        },
    )
