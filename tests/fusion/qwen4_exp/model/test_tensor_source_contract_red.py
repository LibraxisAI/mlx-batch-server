# SPDX-License-Identifier: Apache-2.0
"""Static source contract for the Qwen4Exp text and embedded-MTP trunk."""

from __future__ import annotations

import ast
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]
_TENSOR_PATH = _ROOT / "src/mlx_batch_server/runtime/fusion/qwen4_exp/model/tensor.py"
_LOAD_PLAN_PATH = (
    _ROOT / "src/mlx_batch_server/runtime/fusion/qwen4_exp/model/load_plan.py"
)
_VISION_TOWER_PATH = (
    _ROOT / "src/mlx_batch_server/runtime/fusion/qwen4_exp/vision/tensor_tower.py"
)
_SUPPORT_PATH = (
    _ROOT / "src/mlx_batch_server/runtime/fusion/qwen4_exp/model/tensor_support.py"
)
_EXECUTION_PATH = _ROOT / "src/mlx_batch_server/runtime/fusion/qwen4_exp/execution.py"
_OWNER_PATH = _ROOT / "src/mlx_batch_server/runtime/fusion/concrete/owner.py"
_COMPOSITION_PATH = (
    _ROOT / "src/mlx_batch_server/runtime/fusion/concrete/composition.py"
)
_BOOTSTRAP_PATH = _ROOT / "src/mlx_batch_server/responses/runtime_bootstrap.py"
_RUNTIME_INIT_PATH = _ROOT / "src/mlx_batch_server/runtime/__init__.py"
_DONOR_COMMIT = "6d0ddf0575faa9acf77e63c57e48ea1602a7e4ab"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _tree(path: Path) -> ast.Module:
    return ast.parse(_source(path), filename=str(path))


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == name
    ]
    assert len(matches) == 1
    return matches[0]


def _method(class_node: ast.ClassDef, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1
    return matches[0]


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1
    return matches[0]


def _call_leaf(call: ast.Call) -> str | None:
    function = call.func
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return None


def _call_path(call: ast.Call) -> str | None:
    parts: list[str] = []
    function = call.func
    while isinstance(function, ast.Attribute):
        parts.append(function.attr)
        function = function.value
    if not isinstance(function, ast.Name):
        return None
    parts.append(function.id)
    return ".".join(reversed(parts))


def _calls_named(node: ast.AST, path: str) -> list[ast.Call]:
    return [
        child
        for child in ast.walk(node)
        if isinstance(child, ast.Call) and _call_path(child) == path
    ]


def test_tensor_source_retains_frozen_provenance_without_donor_imports() -> None:
    source = _source(_TENSOR_PATH)
    tree = _tree(_TENSOR_PATH)

    assert source.startswith("# SPDX-License-Identifier: Apache-2.0\n")
    assert _DONOR_COMMIT in source
    assert "Copyright © 2026 MTPLX." in source
    imported_roots = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_roots.update(
        node.module.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    assert imported_roots.isdisjoint({"mtplx", "omlx"})
    assert "os.environ" not in source
    assert "os.getenv" not in source
    assert "ngram-table.safetensors" not in source
    assert "ngram_sidecar" not in source
    assert "BaseModelArgs" not in source


def test_text_args_is_only_a_view_over_canonical_checkpoint_config() -> None:
    tree = _tree(_TENSOR_PATH)
    text_args = _class(tree, "TextArgs")

    assert not text_args.decorator_list
    assert list(text_args.bases) == []
    assignments = [
        node for node in text_args.body if isinstance(node, ast.Assign | ast.AnnAssign)
    ]
    assert len(assignments) == 1
    assert ast.unparse(assignments[0]) == "__slots__ = ('_config', 'capabilities')"
    init = _method(text_args, "__init__")
    assert any(
        isinstance(node, ast.Name) and node.id == "Qwen4ExpTextConfig"
        for node in ast.walk(init)
    )


def test_embedded_ple_and_mtp_are_part_of_the_strict_tensor_tree() -> None:
    source = _source(_TENSOR_PATH)
    load = _function(_tree(_TENSOR_PATH), "load_qwen4_exp_tensor")

    assert "class NGramTable(nn.Module):" in source
    assert "args.split_ngram_parts" in source
    assert "nn.Embedding(self.rows_per_shard, dim)" in source
    assert "self.weight_scale = mx.ones((1,), dtype=mx.bfloat16)" in source
    assert "out.reshape(*ids.shape, self.dim) * self.weight_scale" in source
    assert "self.mtp = Qwen4ExpMTP(args)" in source
    assert 'if key.startswith("mtp."):' in source
    sanitize_calls = _calls_named(load, "model.sanitize")
    assert len(sanitize_calls) == 1
    assert len(sanitize_calls[0].args) == 1
    read_weights = sanitize_calls[0].args[0]
    assert isinstance(read_weights, ast.Call)
    assert _call_path(read_weights) == "_read_indexed_weights"
    assert [ast.unparse(argument) for argument in read_weights.args] == [
        "plan",
        "shard_set",
    ]
    assert "strict=True" in source


def test_weight_loading_uses_exact_index_and_per_module_recipes() -> None:
    source = _source(_TENSOR_PATH)

    assert "for shard_name in plan.artifacts.weight_shards:" in source
    assert "if observed != plan.artifacts.weight_keys:" in source
    assert "duplicate checkpoint tensor" in source
    assert 'if key.endswith(".scales")' in source
    assert "config.quantization_recipe(path)" in source
    assert "quantized_paths - matched_paths" in source


def test_factory_prepares_the_single_plan_before_owner_thread_tensor_load() -> None:
    source = _source(_TENSOR_PATH)
    tree = _tree(_TENSOR_PATH)
    factory = _class(tree, "_Qwen4ExpExecutionFactory")
    prepare = _method(factory, "prepare")
    prepared_factory = _class(tree, "_PreparedQwen4ExpExecutionFactory")
    load = _method(prepared_factory, "load")
    plan_calls = [
        node
        for node in ast.walk(prepare)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_qwen4_exp_plan"
    ]
    load_plan_calls = [
        node
        for node in ast.walk(load)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_qwen4_exp_plan"
    ]

    assert len(plan_calls) == 1
    assert load_plan_calls == []
    assert 'config.options["model_dir"] must be an absolute path' in source
    assert "load_qwen4_exp_tensor(plan, self.capabilities, shard_set)" in source
    assert '"qwen4_exp_plan_sha256": plan.plan_sha256' in source
    assert '"build_facts": build_facts' in source
    assert '"config.json"' not in source


def test_materialization_receipt_is_issued_once_after_complete_chunked_eval() -> None:
    tree = _tree(_TENSOR_PATH)
    materialize = _function(tree, "_materialize_qwen4_exp_parameters")
    eval_calls = [
        node
        for node in ast.walk(materialize)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "mx"
        and node.func.attr == "eval"
    ]
    receipt_calls = [
        node
        for node in ast.walk(materialize)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_issue"
    ]

    assert len(eval_calls) == 1
    assert len(receipt_calls) == 1
    assert eval_calls[0].lineno < receipt_calls[0].lineno
    source = ast.unparse(materialize)
    assert "range(0, len(unique_leaves), _MATERIALIZATION_EVAL_CHUNK_SIZE)" in source
    assert "model.parameters()" in source
    assert "vision_tower.parameters()" in source
    assert "text_mtp.language_model." in source
    assert "text_mtp.mtp." in source
    assert "mx.eval(*unique_leaves" in source
    assert source.index("mx.eval(*unique_leaves") < source.index(
        "shard_set.verify_after_eval()"
    )
    assert source.index("shard_set.verify_after_eval()") < source.index(
        "materialization_authority._issue("
    )
    assert "verify_qwen4_exp_shard_content" not in source
    assert "checkpoint_content_sha256=plan.artifacts.weight_content_sha256" in source
    assert "_issuer_authority" not in source


def test_every_production_mx_load_consumes_one_held_verified_shard_stream() -> None:
    consumers: list[tuple[Path, ast.Call]] = []
    for path in (_ROOT / "src").rglob("*.py"):
        for node in ast.walk(_tree(path)):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "mx"
                and node.func.attr == "load"
            ):
                consumers.append((path, node))

    assert {path for path, _ in consumers} == {_TENSOR_PATH, _VISION_TOWER_PATH}
    assert len(consumers) == 2
    for _, call in consumers:
        assert len(call.args) == 1
        stream = call.args[0]
        assert isinstance(stream, ast.Call)
        assert _call_path(stream) == "shard_set.stream_for_load"
        assert any(
            keyword.arg == "format"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value == "safetensors"
            for keyword in call.keywords
        )

    prepared_load = _method(
        _class(_tree(_TENSOR_PATH), "_PreparedQwen4ExpExecutionFactory"),
        "load",
    )
    open_shards = _calls_named(prepared_load, "open_qwen4_exp_shards")
    load_text = _calls_named(prepared_load, "load_qwen4_exp_tensor")
    load_vision = _calls_named(
        prepared_load,
        "Qwen4ExpVisionTensorTower.from_load_plan",
    )
    materialize = _calls_named(prepared_load, "_materialize_qwen4_exp_parameters")
    assert all(
        len(calls) == 1 for calls in (open_shards, load_text, load_vision, materialize)
    )
    assert (
        open_shards[0].lineno
        < load_text[0].lineno
        < load_vision[0].lineno
        < materialize[0].lineno
    )

    load_plan_source = _source(_LOAD_PLAN_PATH)
    assert 'getattr(os, "O_NOFOLLOW", 0)' in load_plan_source
    assert "os.fstat(stream.fileno())" in load_plan_source


def test_prepared_factory_propagates_one_receipt_after_text_mtp_and_vision() -> None:
    tree = _tree(_TENSOR_PATH)
    prepared = _class(tree, "_PreparedQwen4ExpExecutionFactory")
    load = _method(prepared, "load")
    load_vision = _calls_named(load, "Qwen4ExpVisionTensorTower.from_load_plan")
    materialize = _calls_named(load, "_materialize_qwen4_exp_parameters")
    runtime = _calls_named(load, "_Qwen4ExpTensorRuntime")
    binding = _calls_named(load, "Qwen4ExpExecutionBinding")

    assert all(
        len(calls) == 1 for calls in (load_vision, materialize, runtime, binding)
    )
    assert (
        load_vision[0].lineno
        < materialize[0].lineno
        < runtime[0].lineno
        < binding[0].lineno
    )
    for constructor in (runtime[0], binding[0]):
        receipt_keywords = [
            keyword
            for keyword in constructor.keywords
            if keyword.arg == "materialization_receipt"
        ]
        assert len(receipt_keywords) == 1
        assert ast.unparse(receipt_keywords[0].value) == "materialization_receipt"


def test_no_other_production_surface_can_mint_materialization_receipts() -> None:
    constructors: list[tuple[Path, int]] = []
    for path in (_ROOT / "src").rglob("*.py"):
        tree = _tree(path)
        constructors.extend(
            (path, node.lineno)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and _call_leaf(node) == "TensorMaterializationReceipt"
        )

    assert len(constructors) == 1
    assert constructors[0][0] == _EXECUTION_PATH


def test_production_composition_owns_the_private_issuer_and_no_injection_seam() -> None:
    composition_tree = _tree(_COMPOSITION_PATH)
    compose = _function(composition_tree, "compose_qwen4_exp_backend")
    bootstrap_tree = _tree(_BOOTSTRAP_PATH)
    compose_role = _function(bootstrap_tree, "compose_role_responses_runtime")
    composition_source = ast.unparse(compose)
    owner_source = _source(_OWNER_PATH)
    execution_source = _source(_EXECUTION_PATH)
    runtime_init_source = _source(_RUNTIME_INIT_PATH)

    assert "execution_factory" not in {
        argument.arg for argument in compose.args.kwonlyargs
    }
    assert "execution_factory" not in {
        argument.arg for argument in compose_role.args.kwonlyargs
    }
    assert composition_source.count("_TensorMaterializationIssuerAuthority()") == 1
    assert composition_source.count("_Qwen4ExpExecutionFactory(") == 1
    assert "_bind_trusted_fused_backend_factory(" in composition_source
    assert "materialization_authority=materialization_authority" in composition_source
    assert "__seal" in execution_source
    assert "def _issue(" in execution_source
    assert "def _authenticates(" in execution_source
    assert (
        "_TensorMaterializationIssuerAuthority"
        not in execution_source.split("__all__ =", 1)[1]
    )
    assert "TensorMaterializationReceipt" not in runtime_init_source
    assert (
        owner_source.count("materialization_receipt = owner.materialization_receipt")
        == 1
    )
    assert "materialization_receipt=materialization_receipt" in owner_source


def test_owner_consumes_prepared_factory_after_strict_pre_mailbox_identity() -> None:
    execution_tree = _tree(_EXECUTION_PATH)
    raw_factory = _class(execution_tree, "Qwen4ExpExecutionFactoryPort")
    prepared_factory = _class(
        execution_tree,
        "Qwen4ExpPreparedExecutionFactoryPort",
    )
    owner_source = _source(_OWNER_PATH)

    _method(raw_factory, "prepare")
    _method(prepared_factory, "load")
    assert owner_source.index(
        "prepared_factory = execution_factory.prepare("
    ) < owner_source.index("owner = cls(")
    assert "self._prepared_execution_factory.load()" in owner_source
    assert "self._execution_factory.load(" not in owner_source
    for marker in (
        "prepared factory returned a different runtime identity",
        "prepared factory returned a different load config identity",
        "prepared factory returned a different scheduler config identity",
    ):
        assert marker in owner_source


def test_tensor_runtime_preserves_qsa_gdn_ple_mtp_and_rollback_contracts() -> None:
    source = _source(_TENSOR_PATH)
    tree = _tree(_TENSOR_PATH)

    for name in (
        "GatedDeltaNet",
        "QSAIndexer",
        "QSACache",
        "PLELayer",
        "Qwen4ExpMTP",
        "Model",
    ):
        _class(tree, name)
    for marker in (
        "commit_verified_window",
        "verify_capture_scope",
        "rollback_after_verify",
        "mtp_forward",
        "mtp_update_cache",
        '_require_supported_batch(inputs, "text inputs", cache)',
    ):
        assert marker in source
    assert "multi-row input requires a packed QSA cache" in source


def test_tensor_execution_consumes_policy_and_runs_exact_recursive_mtp() -> None:
    source = _source(_TENSOR_PATH)
    tree = _tree(_TENSOR_PATH)
    runtime = _class(tree, "_Qwen4ExpTensorRuntime")
    execute = _method(runtime, "execute")
    mtp_decode = _method(runtime, "_decode_mtp_one")
    decision = _method(runtime, "_mtp_decision")

    assert "del mtp_policy" not in ast.unparse(execute)
    assert "self._mtp_decision(plan, decode_states, mtp_policy)" in source
    assert "draft_depth=mtp_policy.draft_depth" in source
    assert "mtp_policy.decide(" in ast.unparse(decision)
    assert "MtpAlignment(" in ast.unparse(decision)
    assert "reservation.pending_primary is not None" in ast.unparse(decision)
    for marker in (
        "self.model.mtp_forward(",
        "return_hidden=True",
        "self.model.snapshot(",
        "self.model.begin_capture(",
        "self.model.end_capture(",
        "self.model.commit_verified_window(",
        "self.model.rollback_after_verify(",
        "self.model.mtp_update_cache(",
        "_restore_cache_bundle(reservation.mtp_cache",
    ):
        assert marker in ast.unparse(mtp_decode)


def test_tensor_execution_reports_mtp_and_fallback_accounting() -> None:
    source = _source(_TENSOR_PATH)
    runtime = _class(_tree(_TENSOR_PATH), "_Qwen4ExpTensorRuntime")
    fallback = ast.unparse(_method(runtime, "_mtp_fallback"))
    mtp_batch = ast.unparse(_method(runtime, "_decode_mtp_batch"))

    for marker in (
        "mtp_rounds=mtp_rounds",
        "mtp_drafted_tokens=mtp_drafted_tokens",
        "mtp_accepted_tokens=mtp_accepted_tokens",
        "mtp_rejected_tokens=mtp_rejected_tokens",
        "mtp_fallbacks=fallbacks",
    ):
        assert marker in source
    assert "reason = decision.disable_reason" in fallback
    assert "MULTIROW_NOT_PROVEN" not in fallback
    assert "mtp_rounds=1" in mtp_batch
    assert "mtp_rejected_tokens=int(rejected_rows[row_index])" in mtp_batch


def test_tensor_prefix_cache_restores_whole_semantic_boundaries_only() -> None:
    source = _source(_TENSOR_PATH)
    tree = _tree(_TENSOR_PATH)
    runtime = _class(tree, "_Qwen4ExpTensorRuntime")
    prefill = ast.unparse(_method(runtime, "_prefill"))
    segment = ast.unparse(_method(runtime, "_prefill_segment"))
    stage = ast.unparse(_method(runtime, "_stage_prefix_checkpoint"))
    restore = ast.unparse(_method(runtime, "_restore_prefix_checkpoint"))
    cleanup = ast.unparse(_method(runtime, "cleanup"))

    assert "Qwen4ExpWholeBoundaryPrefixStore(" in source
    assert "plan.tokenizer_fingerprint" in source
    assert "plan.topology.layer_types" in source
    assert "payload.resolution.digest" in source
    assert "self._prefix_store.lookup(" in prefill
    assert "self._restore_prefix_checkpoint(reservation, lookup)" in prefill
    assert "self._prefix_store.detach_lookup(" in prefill
    assert (
        "self._prefill_segment(reservation, reservation.position, boundary)" in prefill
    )
    assert "self.model.mtp_update_cache(" in segment
    assert "target_cache_state=target_state" in stage
    assert "mtp_cache_state=mtp_state" in stage
    assert "_restore_cache_bundle(reservation.cache" in restore
    assert "_restore_cache_bundle(reservation.mtp_cache" in restore
    assert "reason is CacheReleaseReason.COMPLETED" in cleanup
    assert "self._prefix_store.commit(" in cleanup
    assert '"paged_cache_enabled": False' in source
    assert '"prefix_cache_ssd_enabled": False' in source
    assert "paged cache is outside the row-serial v1 runtime" in source
    assert "whole-boundary SSD cache has no concrete serializer port" in source


def test_ar_fallback_keeps_future_mtp_history_exact() -> None:
    tree = _tree(_TENSOR_PATH)
    runtime = _class(tree, "_Qwen4ExpTensorRuntime")
    method = _method(runtime, "_decode_ar_one")
    ar_decode = ast.unparse(method)
    nonterminal = next(
        node
        for node in method.body
        if isinstance(node, ast.If) and ast.unparse(node.test) == "not finished"
    )
    nonterminal_source = ast.unparse(nonterminal)

    assert "pre_forward_hidden = reservation.hidden" in ar_decode
    assert "self._target_forward((primary,), reservation)" in ar_decode
    assert "self.model.mtp_update_cache(" in nonterminal_source
    assert "pre_forward_hidden" in nonterminal_source
    assert "mtp_cache=reservation.mtp_cache" in nonterminal_source
    assert "mx.eval(mtp_hidden)" in nonterminal_source
    assert ar_decode.index("pre_forward_hidden = reservation.hidden") < ar_decode.index(
        "self.model.mtp_update_cache("
    )
    assert ar_decode.index("self.model.mtp_update_cache(") < ar_decode.index(
        "reservation.hidden = hidden[:, -1:, :]"
    )


def test_tensor_runtime_admits_only_sampling_it_can_preserve_exactly() -> None:
    source = _source(_TENSOR_PATH)

    assert "sampling = _parse_tensor_sampling(" in source
    assert "context_length=self.plan.config.text.max_position_embeddings" in source
    assert "prompt_tokens=len(tokens)" in source
    assert "_require_exact_greedy_sampling" not in source
    assert "acceptance_probability(target_p, draft_q, draft)" in source
    assert "residual_distribution(target_p, draft_q)" in source
    assert "completion penalties are not wired into exact" in source
    assert 'sampling.get("max_tokens", 256)' not in source


def test_optional_native_kernels_are_injected_capabilities() -> None:
    source = _source(_SUPPORT_PATH)
    tree = _tree(_SUPPORT_PATH)

    _class(tree, "Qwen4ExpTensorCapabilities")
    imported_roots = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert imported_roots <= {
        "collections",
        "contextlib",
        "contextvars",
        "dataclasses",
        "types",
        "typing",
    }
    assert "MappingProxyType" in source
    assert "native kernel was not injected" in source
    assert "def tensor_capability_scope(" in source
    assert "importlib" not in source
