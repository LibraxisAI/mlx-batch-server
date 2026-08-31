from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from mlx_vlm.utils import get_model_and_args as get_vlm_model_and_args

from mlx_batch_server.chat.mlx import model_types as model_types_module
from mlx_batch_server.chat.mlx import runtime_aliases as runtime_aliases_module
from mlx_batch_server.chat.mlx.model_types import MLXLMCompatibleLanguageModel
from mlx_batch_server.core.config import get_settings


@pytest.fixture(autouse=True)
def _reset_runtime_aliases_and_settings():
    runtime_aliases_module.clear_runtime_aliases()
    get_settings.cache_clear()
    yield
    runtime_aliases_module.clear_runtime_aliases()
    get_settings.cache_clear()


def test_patch_transformers_auto_docstring_for_vlm_swallow_index_error(
    monkeypatch,
):
    calls = {"count": 0}
    fake_module = SimpleNamespace()

    def broken_get_placeholders_dict(*args, **kwargs):
        del args, kwargs
        calls["count"] += 1
        raise IndexError("tuple index out of range")

    fake_module.get_placeholders_dict = broken_get_placeholders_dict
    real_import_module = model_types_module.importlib.import_module

    def fake_import_module(name: str):
        if name == "transformers.utils.auto_docstring":
            return fake_module
        return real_import_module(name)

    monkeypatch.setattr(
        model_types_module.importlib, "import_module", fake_import_module
    )

    model_types_module._patch_transformers_auto_docstring_for_vlm()

    assert fake_module.get_placeholders_dict("demo") == {}
    assert calls["count"] == 1


@pytest.mark.parametrize(
    ("model_type", "expected"),
    [
        ("qwen3.6-vl", "qwen3_vl"),
        ("qwen3_6_vl", "qwen3_vl"),
        ("qwen3.6-vl-moe", "qwen3_vl_moe"),
        ("qwen3_6_vl_moe", "qwen3_vl_moe"),
    ],
)
def test_mlx_vlm_supports_qwen36_aliases(model_type: str, expected: str):
    _, resolved = get_vlm_model_and_args({"model_type": model_type})

    assert resolved == expected


def test_vlm_language_tower_without_cache_metadata_returns_empty_cache():
    tower = SimpleNamespace()
    text_model = MLXLMCompatibleLanguageModel(tower)

    assert text_model.make_cache() == []


def test_vlm_language_wrapper_uses_nested_model_cache_metadata():
    inner_tower = SimpleNamespace(
        layers=[object(), object()],
        head_dim=16,
        n_kv_heads=2,
    )

    class WrappedLanguageTower:
        def __init__(self):
            self.model = inner_tower

        def __call__(self, inputs, cache=None, **kwargs):
            del cache, kwargs
            return SimpleNamespace(logits=inputs)

    text_model = MLXLMCompatibleLanguageModel(WrappedLanguageTower())

    assert len(text_model.make_cache()) == 2
    assert text_model.layers == inner_tower.layers
    assert text_model.head_dim == 16
    assert text_model.n_kv_heads == 2


def test_reset_request_local_runtime_state_clears_nested_language_model():
    inner_tower = SimpleNamespace(
        _position_ids="stale-inner",
        _rope_deltas="stale-inner",
    )
    language_model = SimpleNamespace(
        model=inner_tower,
        _position_ids="stale-outer",
    )
    runtime = SimpleNamespace(language_model=language_model)

    cleared = model_types_module.reset_request_local_runtime_state(runtime)

    assert cleared is True
    assert language_model._position_ids is None
    assert inner_tower._position_ids is None
    assert inner_tower._rope_deltas is None


def test_load_mlx_model_rejects_non_pinned_vlm_in_pinned_only_mode(monkeypatch):
    monkeypatch.setenv("PINNED_MODELS", "mlx-community/Qwen3-VL-30B-A3B-Instruct-8bit")
    monkeypatch.setenv("MODEL_CACHE_MAX_SIZE", "0")
    get_settings.cache_clear()

    monkeypatch.setattr(
        model_types_module,
        "get_model_path",
        lambda model_id: Path("/tmp/fake-vlm"),
    )
    monkeypatch.setattr(
        model_types_module,
        "load_text_config",
        lambda path: {"model_type": "qwen3_vl", "vision_config": {"hidden_size": 1}},
    )
    monkeypatch.setattr(
        model_types_module,
        "_should_use_vlm_runtime",
        lambda config: True,
    )

    load_calls: list[tuple[str, str | None]] = []

    def forbidden_vlm_load(model_id: str, adapter_path: str | None):
        load_calls.append((model_id, adapter_path))
        raise AssertionError(
            "VLM loader should not run for forbidden pinned-only loads"
        )

    monkeypatch.setattr(model_types_module, "_load_vlm_runtime", forbidden_vlm_load)

    with pytest.raises(ValueError, match="not allowed in pinned-only mode"):
        model_types_module.load_mlx_model("Qwen/Qwen3-VL-30B-A3B-Instruct")

    assert load_calls == []


def test_load_mlx_model_allows_pinned_vlm_alias_in_pinned_only_mode(monkeypatch):
    monkeypatch.setenv("PINNED_MODELS", "MLX-Community/Qwen3-VL-30B-A3B-Instruct-8bit")
    monkeypatch.setenv("MODEL_CACHE_MAX_SIZE", "0")
    get_settings.cache_clear()

    runtime_aliases_module.register_runtime_alias(
        "frontier-vlm",
        "mlx-community/qwen3-vl-30b-a3b-instruct-8bit",
    )

    fake_model = SimpleNamespace()
    fake_tokenizer = SimpleNamespace()
    fake_processor = SimpleNamespace()
    fake_chat_template_source = SimpleNamespace()
    seen_loads: list[tuple[str, str | None]] = []

    monkeypatch.setattr(
        model_types_module,
        "get_model_path",
        lambda model_id: Path("/tmp/fake-vlm"),
    )
    monkeypatch.setattr(
        model_types_module,
        "load_text_config",
        lambda path: {"model_type": "qwen3_vl", "vision_config": {"hidden_size": 1}},
    )
    monkeypatch.setattr(
        model_types_module,
        "_should_use_vlm_runtime",
        lambda config: True,
    )
    monkeypatch.setattr(
        model_types_module,
        "_load_vlm_runtime",
        lambda model_id, adapter_path: (
            seen_loads.append((model_id, adapter_path))
            or (fake_model, fake_tokenizer, fake_processor, fake_chat_template_source)
        ),
    )
    monkeypatch.setattr(
        model_types_module,
        "ChatTemplate",
        lambda model_type, source: ("chat-template", model_type, source),
    )
    monkeypatch.setattr(
        model_types_module,
        "extract_context_length",
        lambda config, tokenizer: 8192,
    )

    loaded = model_types_module.load_mlx_model("frontier-vlm")

    assert seen_loads == [("mlx-community/qwen3-vl-30b-a3b-instruct-8bit", None)]
    assert loaded.model_id == "mlx-community/qwen3-vl-30b-a3b-instruct-8bit"
    assert loaded.processor is fake_processor
    assert loaded.supports_multimodal is True


def test_load_mlx_model_expands_home_relative_adapter_path_before_loader(monkeypatch):
    fake_model = SimpleNamespace()
    fake_tokenizer = SimpleNamespace()
    seen_loads: list[tuple[str, str | None]] = []
    expanded_adapter = str(Path("~/adapters/frontier-lora").expanduser())

    monkeypatch.setattr(
        model_types_module,
        "get_model_path",
        lambda model_id: Path("/tmp/fake-llm"),
    )
    monkeypatch.setattr(
        model_types_module,
        "load_text_config",
        lambda path: {"model_type": "llama"},
    )
    monkeypatch.setattr(
        model_types_module,
        "_should_use_vlm_runtime",
        lambda config: False,
    )
    monkeypatch.setattr(
        model_types_module,
        "load_text_runtime",
        lambda model_id, tokenizer_config=None, adapter_path=None: (
            seen_loads.append((model_id, adapter_path)) or (fake_model, fake_tokenizer)
        ),
    )
    monkeypatch.setattr(
        model_types_module,
        "ChatTemplate",
        lambda model_type, source: ("chat-template", model_type, source),
    )
    monkeypatch.setattr(
        model_types_module,
        "extract_context_length",
        lambda config, tokenizer: 8192,
    )
    monkeypatch.setattr(
        model_types_module,
        "_wrap_tokenizer",
        lambda tokenizer_like: tokenizer_like,
    )
    monkeypatch.setattr(
        model_types_module,
        "_fix_tokenizer_eos",
        lambda tokenizer: None,
    )

    loaded = model_types_module.load_mlx_model(
        "mlx-community/llama-3.1-8b",
        adapter_path="~/adapters/frontier-lora",
    )

    assert seen_loads == [("mlx-community/llama-3.1-8b", expanded_adapter)]
    assert loaded.adapter_path == expanded_adapter


def test_resolves_to_multimodal_runtime_honors_alias(monkeypatch):
    runtime_aliases_module.register_runtime_alias(
        "frontier-vlm",
        "mlx-community/pixtral-12b-4bit",
    )

    monkeypatch.setattr(
        model_types_module,
        "get_model_path",
        lambda model_id: (
            Path("/tmp/fake-vlm")
            if model_id == "mlx-community/pixtral-12b-4bit"
            else Path("/tmp/other")
        ),
    )
    monkeypatch.setattr(
        model_types_module,
        "load_text_config",
        lambda path: (
            {
                "model_type": "pixtral",
                "vision_config": {"hidden_size": 1},
            }
            if path == Path("/tmp/fake-vlm")
            else {"model_type": "llama"}
        ),
    )

    assert model_types_module.resolves_to_multimodal_runtime("frontier-vlm") is True
    assert (
        model_types_module.resolves_to_multimodal_runtime("mlx-community/llama-3.1-8b")
        is False
    )
