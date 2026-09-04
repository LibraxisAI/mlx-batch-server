from __future__ import annotations

import json
from pathlib import Path

import pytest

from mlx_batch_server.runtime.fusion.qwen4_exp.model.load_plan import (
    Qwen4ExpLoadPlanError,
    load_qwen4_exp_plan,
)


def _config() -> dict[str, object]:
    return {
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "model_type": "qwen4_exp",
        "image_token_id": 28,
        "video_token_id": 29,
        "vision_start_token_id": 30,
        "vision_end_token_id": 31,
        "language_model_only": False,
        "tie_word_embeddings": False,
        "text_config": {
            "model_type": "qwen4_exp_text",
            "hidden_size": 8,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "rms_norm_eps": 1e-6,
            "vocab_size": 32,
            "max_position_embeddings": 128,
            "tie_word_embeddings": False,
            "attention_bias": False,
            "hidden_act": "silu",
            "dtype": "bfloat16",
            "mamba_ssm_dtype": "float32",
            "use_cache": True,
            "linear_num_value_heads": 2,
            "linear_num_key_heads": 1,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "linear_conv_kernel_dim": 2,
            "output_gate_type": "sigmoid",
            "num_experts": 4,
            "num_experts_per_tok": 1,
            "moe_intermediate_size": 8,
            "shared_expert_intermediate_size": 8,
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
            "full_attention_interval": 4,
            "hc_count": 2,
            "hc_lowrank": 2,
            "indexer_n_heads": 2,
            "indexer_kv_heads": 1,
            "indexer_head_dim": 4,
            "indexer_budget": 8,
            "indexer_compress_ratio": 2,
            "ple_layer_ids": [2],
            "ple_embed_dim": 8,
            "ple_conv_kernel_size": 2,
            "ngram_size": 2,
            "heads_per_ngram": 1,
            "ngram_vocab_size_base": 32,
            "make_ngram_vocab_size_divisible_by": 2,
            "split_ngram_parts": 2,
            "seed": 1234,
            "eos_token_id": [30, 31],
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": False,
            "mtp": {
                "hybrid": True,
                "layer_types": ["full_attention"],
                "num_hidden_layers": 1,
                "rope_theta": 10000,
            },
            "rope_parameters": {
                "mrope_interleaved": True,
                "mrope_section": [1, 1],
                "partial_rotary_factor": 0.5,
                "rope_theta": 10000,
            },
        },
        "vision_config": {
            "depth": 2,
            "hidden_size": 4,
            "intermediate_size": 8,
            "num_heads": 1,
            "in_channels": 3,
            "num_position_embeddings": 16,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 8,
            "hidden_act": "gelu_pytorch_tanh",
            "deepstack_visual_indexes": [],
        },
        "quantization_config": {
            "bits": 4,
            "group_size": 32,
            "mode": "affine",
            "language_model.model.embed_tokens": {
                "bits": 8,
                "group_size": 64,
                "mode": "affine",
            },
        },
    }


def _write_snapshot(root: Path) -> None:
    weight_map = {
        "language_model.embed_tokens.weight": "model-00001-of-00001.safetensors",
        "language_model.model.layers.1.ple.conv1d.weight": (
            "model-00001-of-00001.safetensors"
        ),
        "mtp.fc_embedding.weight": "model-00001-of-00001.safetensors",
        "vision_tower.patch_embed.proj.weight": ("model-00001-of-00001.safetensors"),
    }
    (root / "config.json").write_text(json.dumps(_config()))
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    (root / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "patch_size": 2,
                "temporal_patch_size": 2,
                "merge_size": 2,
                "size": {"shortest_edge": 16, "longest_edge": 256},
                "image_mean": [0.5, 0.5, 0.5],
                "image_std": [0.5, 0.5, 0.5],
            }
        )
    )
    for name in (
        "model-00001-of-00001.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ):
        (root / name).write_bytes(b"fixture")


def test_load_plan_binds_metadata_artifacts_and_row_serial_truth(
    tmp_path: Path,
) -> None:
    _write_snapshot(tmp_path)

    first = load_qwen4_exp_plan(
        model_dir=tmp_path,
        model_id="grant-ai/flash",
        revision="revision-a",
    )
    second = load_qwen4_exp_plan(
        model_dir=tmp_path,
        model_id="grant-ai/flash",
        revision="revision-a",
    )

    assert first == second
    assert first.topology.max_qsa_batch_rows == 1
    assert first.topology.max_verified_mtp_rows == 1
    assert first.artifacts.has_embedded_mtp
    assert first.artifacts.has_embedded_vision
    assert len(first.config.quantization_overrides) == 1
    assert json.loads(first.preprocessor_config_json)["patch_size"] == 2
    assert len(first.preprocessor_sha256) == 64
    assert len(first.tokenizer_fingerprint) == 64
    assert len(first.plan_sha256) == 64


def test_load_plan_digest_changes_with_runtime_identity(tmp_path: Path) -> None:
    _write_snapshot(tmp_path)
    first = load_qwen4_exp_plan(
        model_dir=tmp_path,
        model_id="grant-ai/flash",
        revision="revision-a",
    )
    second = load_qwen4_exp_plan(
        model_dir=tmp_path,
        model_id="grant-ai/flash",
        revision="revision-b",
    )

    assert first.plan_sha256 != second.plan_sha256


def test_load_plan_digest_changes_with_active_tokenizer_surface(
    tmp_path: Path,
) -> None:
    _write_snapshot(tmp_path)
    first = load_qwen4_exp_plan(
        model_dir=tmp_path,
        model_id="grant-ai/flash",
        revision="revision-a",
    )

    (tmp_path / "chat_template.jinja").write_text("changed-template")
    second = load_qwen4_exp_plan(
        model_dir=tmp_path,
        model_id="grant-ai/flash",
        revision="revision-a",
    )

    assert first.tokenizer_fingerprint != second.tokenizer_fingerprint
    assert first.plan_sha256 != second.plan_sha256


def test_load_plan_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    _write_snapshot(tmp_path)
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map": {}, "weight_map": {}}'
    )

    with pytest.raises(Qwen4ExpLoadPlanError, match="duplicate JSON key"):
        load_qwen4_exp_plan(
            model_dir=tmp_path,
            model_id="grant-ai/flash",
            revision="revision-a",
        )


def test_load_plan_rejects_relative_or_oversized_metadata(tmp_path: Path) -> None:
    _write_snapshot(tmp_path)
    with pytest.raises(Qwen4ExpLoadPlanError, match="absolute"):
        load_qwen4_exp_plan(
            model_dir="relative/model",
            model_id="grant-ai/flash",
            revision="revision-a",
        )

    with pytest.raises(Qwen4ExpLoadPlanError, match="size"):
        load_qwen4_exp_plan(
            model_dir=tmp_path,
            model_id="grant-ai/flash",
            revision="revision-a",
            max_metadata_bytes=1,
        )
