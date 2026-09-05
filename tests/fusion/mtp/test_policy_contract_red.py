"""RED contracts for fail-closed MTP lane selection."""

from mlx_batch_server.runtime.fusion.mtp import (
    MtpAlignment,
    MtpDisableReason,
    MtpMode,
    MtpPolicy,
)


def _decide(policy: MtpPolicy, alignment: MtpAlignment):
    return policy.decide(
        alignment=alignment,
        model_supported=True,
        head_attached=True,
        decode_enabled=True,
        verifier_available=True,
    )


def test_singleton_exact_mtp_is_the_initial_guarantee() -> None:
    decision = _decide(
        MtpPolicy(),
        MtpAlignment(runtime_keys=("flash@rev",), cache_positions=(4096,)),
    )

    assert decision.enabled is True
    assert decision.exact is True
    assert decision.mode is MtpMode.EXACT_SINGLETON
    assert decision.facts["draft_depth"] == 3


def test_load_time_disable_forces_autoregressive_fallback() -> None:
    decision = _decide(
        MtpPolicy(enabled=False),
        MtpAlignment(runtime_keys=("flash@rev",), cache_positions=(4096,)),
    )

    assert decision.enabled is False
    assert decision.disable_reason is MtpDisableReason.DECODE_DISABLED


def test_text_stop_constraint_forces_truthful_autoregressive_fallback() -> None:
    policy = MtpPolicy(allow_proven_multirow=True, max_proven_rows=2)
    decision = policy.decide(
        alignment=MtpAlignment(
            runtime_keys=("flash@rev", "flash@rev"),
            cache_positions=(4096, 4096),
        ),
        model_supported=True,
        head_attached=True,
        decode_enabled=True,
        verifier_available=True,
        stop_sequence_constrained=True,
    )

    assert decision.enabled is False
    assert decision.disable_reason is MtpDisableReason.STOP_SEQUENCE_CONSTRAINED


def test_text_stop_constraint_is_the_explicit_reason_when_also_grammar_bound() -> None:
    decision = MtpPolicy().decide(
        alignment=MtpAlignment(
            runtime_keys=("flash@rev",),
            cache_positions=(4096,),
        ),
        model_supported=True,
        head_attached=True,
        decode_enabled=True,
        verifier_available=True,
        grammar_constrained=True,
        stop_sequence_constrained=True,
    )

    assert decision.disable_reason is MtpDisableReason.STOP_SEQUENCE_CONSTRAINED


def test_multirow_mtp_falls_back_until_live_proof_is_enabled() -> None:
    decision = _decide(
        MtpPolicy(),
        MtpAlignment(
            runtime_keys=("flash@rev", "flash@rev"),
            cache_positions=(4096, 4096),
        ),
    )

    assert decision.enabled is False
    assert decision.disable_reason is MtpDisableReason.MULTIROW_NOT_PROVEN


def test_proven_multirow_still_rejects_unaligned_rows() -> None:
    decision = _decide(
        MtpPolicy(allow_proven_multirow=True, max_proven_rows=4),
        MtpAlignment(
            runtime_keys=("flash@rev", "flash@rev"),
            cache_positions=(4096, 4097),
        ),
    )

    assert decision.enabled is False
    assert decision.disable_reason is MtpDisableReason.ROWS_UNALIGNED


def test_aligned_proven_cohort_has_an_explicit_exact_mode() -> None:
    decision = _decide(
        MtpPolicy(allow_proven_multirow=True, max_proven_rows=4),
        MtpAlignment(
            runtime_keys=("flash@rev", "flash@rev"),
            cache_positions=(4096, 4096),
        ),
    )

    assert decision.enabled is True
    assert decision.exact is True
    assert decision.mode is MtpMode.EXACT_ALIGNED_COHORT
