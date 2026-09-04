# SPDX-License-Identifier: Apache-2.0
# Contract derived from MTPLX commit 6d0ddf0575faa9acf77e63c57e48ea1602a7e4ab.

from __future__ import annotations

import numpy as np

from mlx_batch_server.runtime.fusion.qwen4_exp.model.sampling import (
    SamplerConfig,
    SparseDistribution,
    acceptance_probability,
    apply_penalties,
    apply_top_p_top_k,
    distribution_from_logits,
    residual_distribution,
    sample_from_distribution,
    softmax,
    speculative_output_marginal,
    verify_one_token,
)


def test_sampler_defaults_match_frozen_mtplx() -> None:
    assert SamplerConfig() == SamplerConfig(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        presence_penalty=0.0,
        frequency_penalty=0.0,
    )


def test_distribution_from_logits_normalizes_after_filtering() -> None:
    logits = np.array([4.0, 3.0, 2.0, 1.0])
    probs = distribution_from_logits(
        logits,
        SamplerConfig(temperature=0.6, top_p=0.9, top_k=2),
    )

    assert np.isclose(probs.sum(), 1.0)
    assert np.count_nonzero(probs) <= 2


def test_non_positive_temperature_is_exact_greedy_one_hot() -> None:
    np.testing.assert_array_equal(
        softmax(np.array([1.0, 4.0, 3.0]), temperature=0.0),
        np.array([0.0, 1.0, 0.0]),
    )


def test_top_p_precedes_top_k_without_intermediate_renormalization() -> None:
    probs = np.array([0.5, 0.3, 0.2])

    filtered = apply_top_p_top_k(probs, top_p=0.6, top_k=2)

    np.testing.assert_allclose(filtered, np.array([0.625, 0.375, 0.0]))


def test_top_p_keeps_first_token_crossing_threshold() -> None:
    probs = np.array([0.6, 0.25, 0.15])

    filtered = apply_top_p_top_k(probs, top_p=0.7)

    np.testing.assert_allclose(filtered, np.array([12 / 17, 5 / 17, 0.0]))


def test_top_k_breaks_exact_ties_by_lowest_token_id() -> None:
    probs = np.array([0.4, 0.2, 0.2, 0.2])

    filtered = apply_top_p_top_k(probs, top_p=1.0, top_k=2)

    np.testing.assert_allclose(filtered, np.array([2 / 3, 1 / 3, 0.0, 0.0]))


def test_penalties_are_additive_on_raw_logits_and_clamped() -> None:
    logits = np.array([5.0, 4.0, 3.0])

    penalized = apply_penalties(
        logits,
        {0: 2, 2: 1},
        presence_penalty=3.0,
        frequency_penalty=0.5,
    )

    np.testing.assert_allclose(penalized, np.array([2.0, 4.0, 0.5]))
    np.testing.assert_array_equal(logits, np.array([5.0, 4.0, 3.0]))


def test_acceptance_probability_is_minimum_of_one_and_ratio() -> None:
    target = np.array([0.8, 0.2])
    draft = np.array([0.4, 0.6])

    assert acceptance_probability(target, draft, 0) == 1.0
    assert np.isclose(acceptance_probability(target, draft, 1), 1.0 / 3.0)


def test_acceptance_probability_handles_zero_draft_mass() -> None:
    target = np.array([1.0, 0.0])
    draft = np.array([0.0, 1.0])

    assert acceptance_probability(target, draft, 0) == 1.0
    assert acceptance_probability(target, draft, 1) == 0.0


def test_dense_residual_is_normalized_positive_target_minus_draft() -> None:
    target = np.array([0.6, 0.3, 0.1])
    draft = np.array([0.2, 0.5, 0.3])

    residual = residual_distribution(target, draft)

    np.testing.assert_allclose(residual, np.array([1.0, 0.0, 0.0]))


def test_degenerate_dense_residual_falls_back_to_target() -> None:
    target = np.array([0.25, 0.75])

    np.testing.assert_allclose(residual_distribution(target, target), target)
    np.testing.assert_allclose(
        residual_distribution(target, np.array([np.nan, np.nan])),
        target,
    )


def test_sparse_acceptance_and_residual_preserve_token_ids() -> None:
    target = SparseDistribution(
        token_ids=np.array([2, 5, 9]),
        probs=np.array([0.5, 0.3, 0.2]),
        vocab_size=12,
    )
    draft = SparseDistribution.one_hot(5, vocab_size=12)

    assert np.isclose(acceptance_probability(target, draft, 5), 0.3)
    residual = residual_distribution(target, draft)

    assert isinstance(residual, SparseDistribution)
    assert residual.token_ids.tolist() == [2, 9]
    np.testing.assert_allclose(residual.probs, np.array([5 / 7, 2 / 7]))


def test_sparse_invalid_mass_falls_back_to_first_valid_token() -> None:
    distribution = SparseDistribution(
        token_ids=np.array([7, 11]),
        probs=np.array([0.0, np.nan]),
        vocab_size=12,
    )

    assert distribution.token_ids.tolist() == [7]
    np.testing.assert_allclose(distribution.probs, np.array([1.0]))


def test_degenerate_sparse_residual_falls_back_to_target() -> None:
    target = SparseDistribution(
        token_ids=np.array([2, 5]),
        probs=np.array([0.25, 0.75]),
        vocab_size=8,
    )

    residual = residual_distribution(target, target)

    assert residual is target


def test_sparse_sampling_returns_original_token_id() -> None:
    distribution = SparseDistribution(
        token_ids=np.array([7, 11]),
        probs=np.array([0.0, 1.0]),
        vocab_size=12,
    )

    assert sample_from_distribution(distribution, np.random.default_rng(0)) == 11


def test_seed_replays_dense_sampling_sequence() -> None:
    distribution = np.array([0.2, 0.3, 0.5])
    first_rng = np.random.default_rng(90210)
    second_rng = np.random.default_rng(90210)

    first = [sample_from_distribution(distribution, first_rng) for _ in range(20)]
    second = [sample_from_distribution(distribution, second_rng) for _ in range(20)]

    assert first == second


def test_verify_one_token_rejects_into_residual() -> None:
    target = np.array([0.2, 0.8])
    draft = np.array([0.8, 0.2])

    decision = verify_one_token(target, draft, 0, np.random.default_rng(1))

    assert decision.accept_probability == 0.25
    assert not decision.accepted
    assert decision.token_id == 1


def test_speculative_output_marginal_recovers_target_distribution() -> None:
    target = np.array([0.6, 0.3, 0.1])
    draft = np.array([0.2, 0.5, 0.3])

    marginal = speculative_output_marginal(target, draft)

    np.testing.assert_allclose(marginal, target, atol=1e-12)


def test_draft_temperature_changes_acceptance_not_output_marginal() -> None:
    logits = np.array([2.5, 1.0, 0.5, -1.0])
    target = distribution_from_logits(
        logits,
        SamplerConfig(temperature=0.6, top_p=1.0, top_k=0),
    )

    for draft_temperature in (0.1, 0.3, 0.8, 1.2):
        draft = distribution_from_logits(
            logits,
            SamplerConfig(
                temperature=draft_temperature,
                top_p=1.0,
                top_k=0,
            ),
        )
        marginal = speculative_output_marginal(target, draft)
        np.testing.assert_allclose(marginal, target, atol=1e-12)
