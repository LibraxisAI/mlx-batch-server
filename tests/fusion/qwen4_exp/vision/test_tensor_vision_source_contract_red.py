"""Static RED contracts for the embargoed Qwen4Exp tensor vision cut."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
VISION = ROOT / "src/mlx_batch_server/runtime/fusion/qwen4_exp/vision"
SOURCES = {
    name: VISION / f"tensor_{name}.py"
    for name in ("processing", "tower", "splice", "mrope")
}
FROZEN_MTPLX = "6d0ddf0575faa9acf77e63c57e48ea1602a7e4ab"


def _source(name: str) -> str:
    return SOURCES[name].read_text()


def _tree(name: str) -> ast.Module:
    return ast.parse(_source(name), filename=str(SOURCES[name]))


def _imports(name: str) -> set[str]:
    imported: set[str] = set()
    for node in ast.walk(_tree(name)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    return imported


def _definitions(name: str) -> set[str]:
    return {
        node.name
        for node in _tree(name).body
        if isinstance(node, ast.ClassDef | ast.FunctionDef)
    }


def test_tensor_sources_preserve_frozen_transitive_provenance() -> None:
    for name in SOURCES:
        source = _source(name)
        assert "SPDX-License-Identifier: Apache-2.0" in source
        assert FROZEN_MTPLX in source
        assert "Modified by LibraxisAI" in source
    assert "mlx-vlm (Apache-2.0)" in _source("processing")
    assert "mlx-vlm (Apache-2.0)" in _source("tower")
    assert "mlx-vlm (MIT)" in _source("mrope")
    assert "Prince Canuma" in _source("mrope")


def test_tensor_sources_use_runtime_libraries_without_donor_imports() -> None:
    imports = {name: _imports(name) for name in SOURCES}
    assert {"mlx.core", "numpy", "PIL"} <= imports["processing"]
    assert {"mlx.core", "mlx.nn"} <= imports["tower"]
    assert "mlx.core" in imports["splice"]
    assert "mlx.core" in imports["mrope"]
    for names in imports.values():
        assert not any(
            imported == "mtplx"
            or imported.startswith("mtplx.")
            or imported == "omlx"
            or imported.startswith("omlx.")
            for imported in names
        )
        assert not any(
            imported in {"httpx", "requests", "urllib.request"} for imported in names
        )


def test_processing_is_whole_request_ordered_and_exact() -> None:
    definitions = _definitions("processing")
    source = _source("processing")
    assert {
        "TensorVisionPreprocessorConfig",
        "Qwen4ExpTensorPreprocessor",
        "decode_image",
        "smart_resize",
    } <= definitions
    assert "request.bundle.images" in source
    assert "def from_load_plan(" in source
    assert "plan.preprocessor_config_json" in source
    assert "preprocessor and model vision geometry disagree" in source
    assert "MAX_REQUEST_IMAGES" in source
    assert "hashlib.sha256(source.content)" in source
    assert "patch_rows != expected_rows" in source
    assert "validate_preprocessing_output(request, output)" in source


def test_tower_keeps_frame_attention_and_exact_output_receipts() -> None:
    definitions = _definitions("tower")
    source = _source("tower")
    assert {
        "PatchEmbed",
        "PatchMerger",
        "Attention",
        "VisionBlock",
        "Qwen4ExpVisionTensorTower",
        "resolve_vision_prefix",
    } <= definitions
    assert "mx.split(tensor, split_indices, axis=2)" in source
    assert "for _ in range(temporal)" in source
    assert "validate_tower_output(request, output)" in source
    assert "strict=True" in source
    assert "_TOWER_CACHE" not in source


def test_tower_uses_only_immutable_load_plan_checkpoint_truth() -> None:
    source = _source("tower")
    assert "Qwen4ExpModelLoadPlan" in source
    assert "class Qwen4ExpVisionTensorTower" in source
    assert "Qwen3VLTensorTower" not in source
    assert "def from_load_plan(" in source
    assert "plan.config.vision" in source
    assert "plan.artifacts.weight_keys" in source
    assert "plan.artifacts.shards_for_prefix(prefix)" in source
    assert "_planned_shard_path(plan, shard)" in source
    assert "config.quantization_overrides" in source
    assert "config.quantization_recipe(selected)" in source
    assert "nn.quantize(" in source
    assert "config.hidden_act" in source
    assert "from_model_dir" not in source
    assert "config.json" not in source
    assert "model.safetensors.index.json" not in source
    assert "json.loads" not in source
    assert ".glob(" not in source
    assert ".iterdir(" not in source


def test_splice_is_per_request_owner_bound_and_window_exact() -> None:
    source = _source("splice")
    assert "class Qwen4ExpTensorSplicer" in source
    assert "VisionSpliceCursor(plan)" in source
    assert "threading.get_ident()" in source
    assert "owner_thread_violation" in source
    assert "lookup_window(" in source
    assert "window.row_start : window.row_end" in source
    assert "token tensor must exactly match" in source
    assert "ContextVar" not in source


def test_mrope_is_explicit_per_request_and_preserves_attention_semantics() -> None:
    source = _source("mrope")
    definitions = _definitions("mrope")
    assert {
        "Qwen4ExpTensorMrope",
        "select_qwen4_attention_mask",
        "_build_mrope_axes",
        "_mrope_cos_sin",
        "_rope_cos_sin",
        "_apply_partial_rope",
    } <= definitions
    assert "self._table[:, position_start:position_end]" in source
    assert "position_start + self._plan.rope_delta" in source
    assert "mrope_window_crosses_prompt" in source
    assert "if mrope_state is None:" in source
    assert "return sparse_selection" in source
    assert "return None" in source
    assert "ContextVar" not in source


def test_only_contract_modules_are_imported_from_target_vision_package() -> None:
    local_imports = {
        imported
        for name in SOURCES
        for imported in _imports(name)
        if imported.startswith(".")
    }
    assert local_imports == set()
    for name in SOURCES:
        source = _source(name)
        assert "from .processing import" in source or name == "processing"
