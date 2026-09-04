# SPDX-License-Identifier: Apache-2.0
"""Static execution contract for exact stochastic singleton MTP sampling."""

from __future__ import annotations

import ast
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]
_TENSOR_PATH = _ROOT / "src/mlx_batch_server/runtime/fusion/qwen4_exp/model/tensor.py"


def _source() -> str:
    return _TENSOR_PATH.read_text(encoding="utf-8")


def _tree() -> ast.Module:
    return ast.parse(_source(), filename=str(_TENSOR_PATH))


def _class(name: str) -> ast.ClassDef:
    matches = [
        node
        for node in _tree().body
        if isinstance(node, ast.ClassDef) and node.name == name
    ]
    assert len(matches) == 1
    return matches[0]


def _function(name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in _tree().body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1
    return matches[0]


def _method(class_name: str, method_name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in _class(class_name).body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    ]
    assert len(matches) == 1
    return matches[0]


def test_tensor_imports_target_owned_sampling_contract() -> None:
    source = _source()

    assert "import numpy as np" in source
    assert "from .sampling import (" in source
    for symbol in (
        "SamplerConfig",
        "SparseDistribution",
        "acceptance_probability",
        "distribution_from_logits",
        "residual_distribution",
        "sample_from_distribution",
    ):
        assert symbol in source
    assert "from mtplx" not in source


def test_reservation_owns_sampler_rng_and_pending_primary() -> None:
    reservation = _class("_TensorReservation")
    fields = {
        node.target.id
        for node in reservation.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }

    assert {
        "sampler",
        "draft_sampler",
        "rng",
        "pending_primary",
        "input_embeddings",
        "position_table",
        "mrope",
    } <= fields


def test_reserve_builds_one_seeded_rng_only_when_sampling_needs_it() -> None:
    reserve = ast.unparse(_method("_Qwen4ExpTensorRuntime", "reserve"))
    module = _tree()
    module_level_rngs = [
        node
        for node in module.body
        if isinstance(node, ast.Assign | ast.AnnAssign)
        and "default_rng" in ast.unparse(node)
    ]

    assert "sampling = _parse_tensor_sampling(" in reserve
    assert "context_length=self.plan.config.text.max_position_embeddings" in reserve
    assert "prompt_tokens=len(tokens)" in reserve
    assert reserve.index("if not tokens") < reserve.index(
        "sampling = _parse_tensor_sampling("
    )
    assert "np.random.default_rng(sampling.seed)" in reserve
    assert "if sampling.needs_rng" in reserve
    assert "else None" in reserve
    assert "rng=" in reserve
    assert module_level_rngs == []


def test_parser_uses_donor_defaults_and_optional_draft_sampler() -> None:
    parser = ast.unparse(_function("_parse_tensor_sampling"))
    sampler_parser = ast.unparse(_function("_sampler_from_mapping"))

    assert "target = _sampler_from_mapping" in parser
    assert "draft = target" in parser
    assert "isinstance(raw_draft, Mapping)" in parser
    assert "SamplerConfig()" in sampler_parser
    assert "temperature=temperature" in sampler_parser
    assert "top_p=top_p" in sampler_parser
    assert "top_k=top_k" in sampler_parser


def test_parser_fails_unwired_penalties_and_unknown_controls() -> None:
    parser = ast.unparse(_function("_parse_tensor_sampling"))
    sampler_parser = ast.unparse(_function("_sampler_from_mapping"))

    assert "unsupported sampling controls" in parser
    assert "sampler has unsupported controls" in sampler_parser
    assert "presence != 0.0 or frequency != 0.0" in sampler_parser
    assert "completion penalties are not wired" in sampler_parser


def test_greedy_sampler_uses_argmax_without_distribution_or_rng() -> None:
    function = _function("_sample_from_logits")
    greedy = function.body[0]

    assert isinstance(greedy, ast.If)
    assert ast.unparse(greedy.test) == "config.temperature <= 0"
    greedy_source = ast.unparse(greedy)
    assert "mx.argmax" in greedy_source
    assert "distribution_from_logits" not in greedy_source
    assert "sample_from_distribution" not in greedy_source
    assert "_require_sampling_rng" not in greedy_source


def test_ar_fallback_samples_target_and_consumes_pending_primary_once() -> None:
    method = ast.unparse(_method("_Qwen4ExpTensorRuntime", "_decode_ar_one"))
    take_primary = ast.unparse(_method("_Qwen4ExpTensorRuntime", "_take_primary"))
    next_target = ast.unparse(_method("_Qwen4ExpTensorRuntime", "_next_target_token"))

    assert "primary, primary_is_new = self._take_primary(reservation)" in method
    assert "self._next_greedy_token(reservation)" not in method
    assert "reservation.pending_primary = None" in take_primary
    assert "return (primary, False)" in take_primary
    assert "_sample_from_logits(" in next_target
    assert "reservation.logits[0, -1]" in next_target
    assert "reservation.rng" in next_target


def test_stochastic_mtp_samples_p_and_q_then_ratio_verifies() -> None:
    method = ast.unparse(_method("_Qwen4ExpTensorRuntime", "_decode_mtp_one"))

    for marker in (
        "_sample_draft_from_logits(",
        "_distribution_from_mlx_logits(verify_logits[0, index]",
        "acceptance_probability(target_p, draft_q, draft)",
        "float(rng.random()) <= accept_p",
        "residual_distribution(target_p, draft_q)",
        "sample_from_distribution(",
    ):
        assert marker in method


def test_rejection_replaces_draft_with_authoritative_correction() -> None:
    method = ast.unparse(_method("_Qwen4ExpTensorRuntime", "_decode_mtp_one"))

    assert "committed_tokens = (*committed_tokens, correction)" in method
    assert "emitted.append(correction)" in method
    assert "self.model.mtp_update_cache(" in method
    assert "history_hidden[:, accepted_count:accepted_count + 1, :]" in method


def test_acceptance_commits_draft_once_and_preserves_bonus_rng_order() -> None:
    method = ast.unparse(_method("_Qwen4ExpTensorRuntime", "_decode_mtp_one"))

    assert "emitted.extend(accepted_drafts)" in method
    assert "bonus, _ = _sample_from_logits(" in method
    assert "reservation.pending_primary = bonus" in method
    bonus_sample = method.index("bonus, _ = _sample_from_logits(")
    assert method.rindex("self._would_finish", 0, bonus_sample) < bonus_sample


def test_stop_or_budget_guard_precedes_draft_and_bonus_rng() -> None:
    method = ast.unparse(_method("_Qwen4ExpTensorRuntime", "_decode_mtp_one"))

    terminal_guard = method.index("if self._would_finish(reservation, tuple(emitted))")
    draft_sample = method.index("draft, draft_q = _sample_draft_from_logits(")
    bonus_sample = method.index("bonus, _ = _sample_from_logits(")
    bonus_guard = method.rindex("self._would_finish", 0, bonus_sample)

    assert terminal_guard < draft_sample
    assert bonus_guard < bonus_sample


def test_primary_draft_and_correction_frontiers_are_explicit() -> None:
    method = ast.unparse(_method("_Qwen4ExpTensorRuntime", "_decode_mtp_one"))

    assert "verify_tokens = (primary, *draft_tokens)" in method
    assert "keep_tokens = 1 + accepted_count" in method
    assert "verified_tokens=len(verify_tokens)" in method
    assert (
        "authoritative_logits, authoritative_hidden = self._target_forward((correction,), reservation)"
        in method
    )
    assert "committed_tokens = (primary, *accepted_drafts)" in method
    assert "_restore_cache_bundle(reservation.mtp_cache, mtp_snapshot)" in method
    assert method.count("self.model.mtp_update_cache(") == 1


def test_recursive_drafts_chain_hidden_state_and_stop_at_request_boundaries() -> None:
    method = ast.unparse(_method("_Qwen4ExpTensorRuntime", "_decode_mtp_one"))

    assert "max_drafts = min(draft_depth, remaining)" in method
    assert "for _ in range(max_drafts)" in method
    assert "return_hidden=True" in method
    assert "draft_hidden = draft_hidden_next[:, -1:, :]" in method
    assert "next_token = draft" in method
    assert method.count("if draft in self._stop_token_ids") == 1
    assert method.index("if draft in self._stop_token_ids") > method.index(
        "for index, (draft, draft_q) in enumerate("
    )
    assert "for index, (draft, draft_q) in enumerate(" in method


def test_mtp_repair_uses_authoritative_hidden_and_counts_first_rejection_once() -> None:
    method = ast.unparse(_method("_Qwen4ExpTensorRuntime", "_decode_mtp_one"))

    assert (
        "history_hidden = verify_hidden if committed_from_capture else authoritative_hidden"
        in method
    )
    assert "mtp_rejected_tokens=int(rejected)" in method


def test_verifier_availability_depends_on_per_request_exact_state() -> None:
    decision = ast.unparse(_method("_Qwen4ExpTensorRuntime", "_mtp_decision"))
    verifier = ast.unparse(
        _method("_Qwen4ExpTensorRuntime", "_has_exact_sampling_verifier")
    )

    assert "verifier_available=True" not in decision
    assert "self._has_exact_sampling_verifier(reservation)" in decision
    assert "reservation.rng is not None" in verifier


def test_sampling_wiring_retains_plato_vision_contract() -> None:
    source = _source()
    reserve = ast.unparse(_method("_Qwen4ExpTensorRuntime", "reserve"))
    mtp_decode = ast.unparse(_method("_Qwen4ExpTensorRuntime", "_decode_mtp_one"))

    for marker in (
        "PreparedGenerationRequest",
        "RequestModality.VISION",
        "self._prepare_vision_prompt(",
        "input_embeddings=input_embeddings",
        "position_table=mrope.position_table",
        "Qwen4ExpTensorPreprocessor",
        "Qwen4ExpVisionTensorTower",
    ):
        assert marker in source
    assert "canonical = request.request" in reserve
    assert "with self._vision_rope_scope(reservation, history_shift=1)" in mtp_decode
