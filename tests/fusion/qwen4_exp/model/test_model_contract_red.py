from __future__ import annotations

from copy import deepcopy

import pytest

from mlx_batch_server.runtime.fusion.qwen4_exp.model import (
    Qwen4ExpArtifactError,
    Qwen4ExpConfigError,
    Qwen4ExpTensorBatchMode,
    build_qwen4_exp_topology,
    inspect_qwen4_exp_artifacts,
    parse_qwen4_exp_config,
    precompute_mtp_indexer_replay,
    precompute_qsa_replay_capacity,
)


def _config() -> dict[str, object]:
    layer_types = tuple(
        "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
        for index in range(48)
    )
    return {
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "model_type": "qwen4_exp",
        "image_token_id": 248056,
        "video_token_id": 248057,
        "vision_start_token_id": 248053,
        "vision_end_token_id": 248054,
        "language_model_only": False,
        "tie_word_embeddings": False,
        "text_config": {
            "model_type": "qwen4_exp_text",
            "hidden_size": 2560,
            "num_hidden_layers": 48,
            "num_attention_heads": 24,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "rms_norm_eps": 1e-6,
            "vocab_size": 248320,
            "max_position_embeddings": 262144,
            "tie_word_embeddings": False,
            "attention_bias": False,
            "hidden_act": "silu",
            "dtype": "bfloat16",
            "mamba_ssm_dtype": "float32",
            "use_cache": True,
            "linear_num_value_heads": 48,
            "linear_num_key_heads": 16,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "output_gate_type": "sigmoid",
            "num_experts": 512,
            "num_experts_per_tok": 10,
            "moe_intermediate_size": 640,
            "shared_expert_intermediate_size": 640,
            "layer_types": layer_types,
            "full_attention_interval": 4,
            "hc_count": 4,
            "hc_lowrank": 320,
            "indexer_n_heads": 4,
            "indexer_kv_heads": 1,
            "indexer_head_dim": 128,
            "indexer_budget": 2048,
            "indexer_compress_ratio": 4,
            "ple_layer_ids": [2],
            "ple_embed_dim": 2560,
            "ple_conv_kernel_size": 4,
            "ngram_size": 3,
            "heads_per_ngram": 8,
            "ngram_vocab_size_base": 20_000_000,
            "make_ngram_vocab_size_divisible_by": 128,
            "split_ngram_parts": 128,
            "seed": None,
            "eos_token_id": 248044,
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": False,
            "mtp": {
                "hybrid": True,
                "layer_types": ["full_attention"],
                "num_hidden_layers": 1,
                "rope_theta": 10_000_000,
            },
            "rope_parameters": {
                "mrope_interleaved": True,
                "mrope_section": [11, 11, 10],
                "partial_rotary_factor": 0.25,
                "rope_theta": 10_000_000,
            },
        },
        "vision_config": {
            "depth": 27,
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "num_heads": 16,
            "in_channels": 3,
            "num_position_embeddings": 2304,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 2560,
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


def test_flash_checkpoint_topology_is_explicitly_row_serial() -> None:
    config = parse_qwen4_exp_config(_config())
    topology = build_qwen4_exp_topology(config)

    assert topology.full_attention_layers == tuple(range(3, 48, 4))
    assert topology.ple_layers == (1,)
    assert topology.tensor_batch_mode is Qwen4ExpTensorBatchMode.ROW_SERIAL
    assert topology.max_qsa_batch_rows == 1
    assert topology.max_verified_mtp_rows == 1
    assert config.text.seed == 1234
    assert config.text.eos_token_ids == (248044,)
    assert config.quantization_recipe("language_model.model.embed_tokens") == (
        8,
        64,
        "affine",
    )
    assert config.quantization_recipe("language_model.model.layers.0.mlp") == (
        4,
        32,
        "affine",
    )


def test_config_rejects_mtp_depth_disagreement() -> None:
    raw = deepcopy(_config())
    text_config = raw["text_config"]
    assert isinstance(text_config, dict)
    text_config["mtp_num_hidden_layers"] = 2

    with pytest.raises(Qwen4ExpConfigError, match="depth declarations"):
        parse_qwen4_exp_config(raw)


def test_config_rejects_divergent_or_malformed_quantization_recipes() -> None:
    divergent = deepcopy(_config())
    divergent["quantization"] = {
        "bits": 3,
        "group_size": 32,
        "mode": "affine",
    }
    with pytest.raises(Qwen4ExpConfigError, match="declarations disagree"):
        parse_qwen4_exp_config(divergent)

    malformed = deepcopy(_config())
    quantization = malformed["quantization_config"]
    assert isinstance(quantization, dict)
    quantization["language_model.model.embed_tokens"] = {
        "bits": 8,
        "group_size": 64,
        "mode": "affine",
        "foreign": True,
    }
    with pytest.raises(Qwen4ExpConfigError, match="unknown shape"):
        parse_qwen4_exp_config(malformed)


def test_config_rejects_layer_pattern_that_disagrees_with_interval() -> None:
    raw = deepcopy(_config())
    text_config = raw["text_config"]
    assert isinstance(text_config, dict)
    layer_types = list(text_config["layer_types"])
    layer_types[0] = "full_attention"
    text_config["layer_types"] = layer_types
    config = parse_qwen4_exp_config(raw)

    with pytest.raises(Qwen4ExpConfigError, match="full-attention layers"):
        build_qwen4_exp_topology(config)


def test_embedded_flash_artifact_pack_is_complete_and_deterministic() -> None:
    config = parse_qwen4_exp_config(_config())
    files = (
        "config.json",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "preprocessor_config.json",
        "tokenizer.json",
    )
    weight_map = {
        "language_model.embed_tokens.weight": "model-00001-of-00002.safetensors",
        "language_model.model.layers.1.ple.conv1d.weight": (
            "model-00001-of-00002.safetensors"
        ),
        "mtp.fc_embedding.weight": "model-00002-of-00002.safetensors",
        "vision_tower.patch_embed.proj.weight": ("model-00002-of-00002.safetensors"),
    }

    first = inspect_qwen4_exp_artifacts(
        config=config,
        file_names=files,
        weight_map=weight_map,
    )
    second = inspect_qwen4_exp_artifacts(
        config=config,
        file_names=tuple(reversed(files)),
        weight_map=dict(reversed(tuple(weight_map.items()))),
    )

    assert first == second
    assert first.has_embedded_mtp
    assert first.has_embedded_vision
    assert first.has_embedded_ple
    assert first.shards_for_prefix("vision_tower.") == (
        "model-00002-of-00002.safetensors",
    )
    assert len(first.digest) == 64


def test_artifact_inventory_rejects_unindexed_or_missing_components() -> None:
    config = parse_qwen4_exp_config(_config())
    files = (
        "config.json",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "preprocessor_config.json",
        "tokenizer.json",
    )
    incomplete = {
        "language_model.embed_tokens.weight": "model-00001-of-00002.safetensors"
    }
    with pytest.raises(Qwen4ExpArtifactError, match="components"):
        inspect_qwen4_exp_artifacts(
            config=config,
            file_names=tuple(
                name for name in files if name != "model-00002-of-00002.safetensors"
            ),
            weight_map=incomplete,
        )

    complete = dict(incomplete)
    complete.update(
        {
            "language_model.model.layers.1.ple.conv1d.weight": (
                "model-00001-of-00002.safetensors"
            ),
            "mtp.fc_embedding.weight": "model-00001-of-00002.safetensors",
            "vision_tower.patch_embed.proj.weight": (
                "model-00001-of-00002.safetensors"
            ),
        }
    )
    with pytest.raises(Qwen4ExpArtifactError, match="unindexed"):
        inspect_qwen4_exp_artifacts(
            config=config,
            file_names=files,
            weight_map=complete,
        )


def test_artifact_inventory_rejects_duplicate_snapshot_names() -> None:
    config = parse_qwen4_exp_config(_config())
    files = (
        "config.json",
        "config.json",
        "model.safetensors.index.json",
        "model-00001-of-00001.safetensors",
        "preprocessor_config.json",
        "tokenizer.json",
    )
    weight_map = {
        "language_model.embed_tokens.weight": "model-00001-of-00001.safetensors",
        "language_model.model.layers.1.ple.conv1d.weight": (
            "model-00001-of-00001.safetensors"
        ),
        "mtp.fc_embedding.weight": "model-00001-of-00001.safetensors",
        "vision_tower.patch_embed.proj.weight": ("model-00001-of-00001.safetensors"),
    }

    with pytest.raises(Qwen4ExpArtifactError, match="duplicates"):
        inspect_qwen4_exp_artifacts(
            config=config,
            file_names=files,
            weight_map=weight_map,
        )


def test_qsa_capacity_preserves_unaligned_staging_row() -> None:
    plan = precompute_qsa_replay_capacity(
        start_offset=0,
        window_tokens=1025,
        compress_ratio=4,
    )

    assert plan.end_offset == 1025
    assert plan.complete_blocks == 256
    assert plan.raw_capacity == 2048
    assert plan.pooled_capacity == 512
    assert plan.graph_key == (2048, 512)


def test_mtp_replay_retains_only_authoritative_primary() -> None:
    plan = precompute_mtp_indexer_replay(cycle_offset=10, observed_offset=14)

    assert plan.speculative_rows == 4
    assert plan.reusable_rows == 1
    assert plan.rollback_offset == 11
    assert plan.reappend_tokens((20, 21, 22)) == (21, 22)
    assert plan.authoritative_hidden_rows(3) == 2


def test_prompt_lookup_replay_retains_no_unobserved_primary() -> None:
    plan = precompute_mtp_indexer_replay(cycle_offset=10, observed_offset=10)

    assert not plan.primary_staged
    assert plan.reusable_rows == 0
    assert plan.reappend_tokens((20, 21)) == (20, 21)
