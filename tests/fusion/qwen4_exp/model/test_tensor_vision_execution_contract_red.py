"""Static contracts for the embargoed Qwen4Exp vision execution seam."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
TENSOR_PATH = ROOT / "src/mlx_batch_server/runtime/fusion/qwen4_exp/model/tensor.py"
MROPE_PATH = (
    ROOT / "src/mlx_batch_server/runtime/fusion/qwen4_exp/vision/tensor_mrope.py"
)


def _source(path: Path) -> str:
    return path.read_text()


def _tree(path: Path) -> ast.Module:
    return ast.parse(_source(path), filename=str(path))


def _definition(path: Path, name: str) -> ast.AST:
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.ClassDef | ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing definition: {name}")


def _segment(path: Path, name: str) -> str:
    segment = ast.get_source_segment(_source(path), _definition(path, name))
    assert segment is not None
    return segment


def test_reserve_is_prepared_only_and_fails_raw_media_before_state_mutation() -> None:
    runtime = _definition(TENSOR_PATH, "_Qwen4ExpTensorRuntime")
    assert isinstance(runtime, ast.ClassDef)
    reserve = next(
        node
        for node in runtime.body
        if isinstance(node, ast.FunctionDef) and node.name == "reserve"
    )
    annotation = ast.unparse(reserve.args.args[1].annotation)
    assert annotation == "PreparedGenerationRequest"
    source = ast.get_source_segment(_source(TENSOR_PATH), reserve)
    assert source is not None
    guard = "raw media requires a sealed PreparedQwen4Prompt"
    mutation = "self._reservations[canonical.response_id] = reservation"
    assert guard in source
    assert mutation in source
    assert source.index(guard) < source.index(mutation)
    assert "RequestModality.TEXT" in source
    assert "RequestModality.VISION" in source


def test_sealed_messages_render_all_roles_parts_and_images_in_order() -> None:
    validate = _segment(TENSOR_PATH, "_require_prepared_vision_prompt")
    render = _segment(TENSOR_PATH, "_render_prepared_messages")
    layout = _segment(TENSOR_PATH, "_sealed_message_layout")
    assert "PreparedQwen4Prompt" in validate
    assert "value.response_id != request.response_id" in validate
    assert "value.runtime != request.runtime" in validate
    assert "value.messages" in validate
    assert "prepared prompt role/message layout changed" in validate
    assert "for message in prompt.messages" in render
    assert "for item in message.items" in render
    assert '"role": message.role' in render
    assert '"content": "".join(content)' in render
    assert 'rendered_message["type"] = message.item_type' in render
    assert 'rendered_message["call_id"] = message.call_id' in render
    assert 'rendered_message["output"] = message.output' in render
    assert "ResolvedText" in render
    assert "ResolvedImage" in render
    assert "<|vision_start|><|image_pad|><|vision_end|>" in _source(TENSOR_PATH)
    assert "previous_media_part" in layout


def test_owner_thread_builds_existing_vision_components_from_load_plan() -> None:
    runtime = _segment(TENSOR_PATH, "_Qwen4ExpTensorRuntime")
    prepare = _segment(TENSOR_PATH, "_prepare_vision_prompt")
    assert "Qwen4ExpTensorPreprocessor.from_load_plan(plan)" in runtime
    assert "Qwen4ExpVisionTensorTower.from_load_plan(plan)" in runtime
    assert "VisionProcessingRequest(" in prepare
    assert "VisionTowerRequest(" in prepare
    assert "build_vision_splice_plan(" in prepare
    assert "Qwen4ExpTensorSplicer(" in prepare
    assert "build_mrope_plan(" in prepare
    assert "Qwen4ExpTensorMrope(" in prepare
    assert "_ignored_deepstack = tower_output.deepstack" in prepare
    assert "async " not in prepare
    assert "await " not in prepare


def test_image_pad_expansion_embeddings_and_history_alignment_are_explicit() -> None:
    expand = _segment(TENSOR_PATH, "_expand_image_pad_tokens")
    prepare = _segment(TENSOR_PATH, "_prepare_vision_prompt")
    runtime = _segment(TENSOR_PATH, "_Qwen4ExpTensorRuntime")
    assert "pad_offsets" in expand
    assert "vision_start_token_id" in expand
    assert "vision_end_token_id" in expand
    assert "expanded.extend((image_pad_token_id,) * rows)" in expand
    assert "embed_tokens=self.model.language_model.model.embed_tokens" in prepare
    assert "splicer.assert_complete(" in prepare
    assert "reservation.input_embeddings[:, start:end, :]" in runtime
    assert "input_embeddings=input_embeddings" in runtime
    assert "input_embeddings[:, 1:, :]" in runtime
    assert "input_embeddings=history_embeddings" in runtime


def test_per_request_mrope_table_scopes_prefill_decode_and_verify() -> None:
    reservation = _segment(TENSOR_PATH, "_TensorReservation")
    runtime = _segment(TENSOR_PATH, "_Qwen4ExpTensorRuntime")
    mrope = _segment(MROPE_PATH, "Qwen4ExpTensorMrope")
    assert "position_table: Any = None" in reservation
    assert "mrope: Qwen4ExpTensorMrope | None = None" in reservation
    assert "def position_table(self) -> mx.array" in mrope
    assert "return self._table" in mrope
    assert "vision_rope_scope(" in runtime
    assert 'phase="verify"' in runtime
    assert "history_shift=1" in runtime
    assert "reservation.mrope.rope_delta + history_shift" in runtime


def test_request_local_preparation_does_not_reread_checkpoint_metadata() -> None:
    prepare = _segment(TENSOR_PATH, "_prepare_vision_prompt")
    forbidden = (
        "load_qwen4_exp_plan",
        "load_tokenizer",
        "config.json",
        "model.safetensors.index.json",
        ".glob(",
        ".iterdir(",
    )
    assert not any(value in prepare for value in forbidden)
    assert "self.plan.config" in prepare


def test_runtime_remains_honest_b1_row_serial() -> None:
    runtime = _segment(TENSOR_PATH, "_Qwen4ExpTensorRuntime")
    assert "B=1 row-serial baseline" in runtime
    assert "for row in plan.prefill_rows" in runtime
    assert "for row, reservation in zip(" in runtime
    assert "tensor batch" not in runtime.lower()
