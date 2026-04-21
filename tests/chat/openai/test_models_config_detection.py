from mlx_batch_server.chat.openai.models import models as models_module


def test_is_embeddings_config_accepts_qwen3_vl_moe() -> None:
    assert models_module._is_embeddings_config({"model_type": "qwen3_vl_moe"}) is True


def test_is_embeddings_config_accepts_future_qwen36_vl_alias() -> None:
    assert models_module._is_embeddings_config({"model_type": "qwen3_6_vl_moe"}) is True
