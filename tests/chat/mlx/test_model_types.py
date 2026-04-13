from __future__ import annotations

from types import SimpleNamespace

from mlx_batch_server.chat.mlx import model_types as model_types_module
from mlx_batch_server.chat.mlx.model_types import MLXLMCompatibleLanguageModel


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


def test_vlm_language_tower_without_cache_metadata_returns_empty_cache():
    tower = SimpleNamespace()
    text_model = MLXLMCompatibleLanguageModel(tower)

    assert text_model.make_cache() == []
