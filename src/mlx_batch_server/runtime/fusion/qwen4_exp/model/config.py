# SPDX-License-Identifier: Apache-2.0
"""Tensor-free Qwen4Exp checkpoint configuration.

The field set follows MTPLX ``mtplx/models/qwen4_exp.py`` at
``6d0ddf0575faa9acf77e63c57e48ea1602a7e4ab``. LibraxisAI keeps this parser
independent from MLX and MLX-LM so checkpoint truth can be validated before
the dependency ABI is selected.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


class Qwen4ExpConfigError(ValueError):
    """The checkpoint configuration cannot satisfy the target contract."""


@dataclass(frozen=True, slots=True)
class Qwen4ExpQuantizationRecipe:
    module_path: str
    bits: int
    group_size: int
    mode: str

    def __post_init__(self) -> None:
        path_parts = self.module_path.split(".")
        if (
            not self.module_path
            or "/" in self.module_path
            or "\\" in self.module_path
            or any(not part for part in path_parts)
        ):
            raise Qwen4ExpConfigError("quantized module path is invalid")
        if self.bits < 1 or self.group_size < 1:
            raise Qwen4ExpConfigError("quantization geometry must be positive")
        if not self.mode:
            raise Qwen4ExpConfigError("quantization mode must not be empty")


@dataclass(frozen=True, slots=True)
class Qwen4ExpMtpConfig:
    num_hidden_layers: int
    layer_types: tuple[str, ...]
    hybrid: bool
    rope_theta: int

    def __post_init__(self) -> None:
        if self.num_hidden_layers < 1:
            raise Qwen4ExpConfigError("MTP requires at least one hidden layer")
        if len(self.layer_types) != self.num_hidden_layers:
            raise Qwen4ExpConfigError("MTP layer_types do not match its depth")
        if any(kind != "full_attention" for kind in self.layer_types):
            raise Qwen4ExpConfigError("Qwen4Exp MTP requires full-attention layers")
        if self.rope_theta < 1:
            raise Qwen4ExpConfigError("MTP rope theta must be positive")


@dataclass(frozen=True, slots=True)
class Qwen4ExpTextConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    vocab_size: int
    max_position_embeddings: int
    tie_word_embeddings: bool
    attention_bias: bool
    hidden_act: str
    dtype: str
    mamba_ssm_dtype: str
    use_cache: bool
    linear_num_value_heads: int
    linear_num_key_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    output_gate_type: str
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    norm_topk_prob: bool
    decoder_sparse_step: int
    layer_types: tuple[str, ...]
    full_attention_interval: int
    hc_count: int
    hc_lowrank: int
    indexer_n_heads: int
    indexer_kv_heads: int
    indexer_head_dim: int
    indexer_budget: int
    indexer_compress_ratio: int
    ple_layer_ids: tuple[int, ...]
    ple_embed_dim: int
    ple_conv_kernel_size: int
    ngram_size: int
    heads_per_ngram: int
    ngram_vocab_size_base: int
    make_ngram_vocab_size_divisible_by: int
    split_ngram_parts: int
    seed: int
    eos_token_ids: tuple[int, ...]
    partial_rotary_factor: float
    rope_theta: int
    mrope_section: tuple[int, ...]
    mrope_interleaved: bool
    mtp_use_dedicated_embeddings: bool
    mtp: Qwen4ExpMtpConfig

    def __post_init__(self) -> None:
        positive = {
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "vocab_size": self.vocab_size,
            "max_position_embeddings": self.max_position_embeddings,
            "linear_num_value_heads": self.linear_num_value_heads,
            "linear_num_key_heads": self.linear_num_key_heads,
            "linear_key_head_dim": self.linear_key_head_dim,
            "linear_value_head_dim": self.linear_value_head_dim,
            "linear_conv_kernel_dim": self.linear_conv_kernel_dim,
            "num_experts": self.num_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "moe_intermediate_size": self.moe_intermediate_size,
            "shared_expert_intermediate_size": (self.shared_expert_intermediate_size),
            "decoder_sparse_step": self.decoder_sparse_step,
            "full_attention_interval": self.full_attention_interval,
            "hc_count": self.hc_count,
            "hc_lowrank": self.hc_lowrank,
            "indexer_n_heads": self.indexer_n_heads,
            "indexer_kv_heads": self.indexer_kv_heads,
            "indexer_head_dim": self.indexer_head_dim,
            "indexer_budget": self.indexer_budget,
            "indexer_compress_ratio": self.indexer_compress_ratio,
            "ple_embed_dim": self.ple_embed_dim,
            "ple_conv_kernel_size": self.ple_conv_kernel_size,
            "ngram_size": self.ngram_size,
            "heads_per_ngram": self.heads_per_ngram,
            "ngram_vocab_size_base": self.ngram_vocab_size_base,
            "make_ngram_vocab_size_divisible_by": (
                self.make_ngram_vocab_size_divisible_by
            ),
            "split_ngram_parts": self.split_ngram_parts,
            "rope_theta": self.rope_theta,
        }
        invalid = sorted(name for name, value in positive.items() if value < 1)
        if invalid:
            raise Qwen4ExpConfigError(
                f"positive Qwen4Exp fields are invalid: {invalid!r}"
            )
        if self.rms_norm_eps <= 0.0:
            raise Qwen4ExpConfigError("rms_norm_eps must be positive")
        if not self.hidden_act or not self.dtype or not self.mamba_ssm_dtype:
            raise Qwen4ExpConfigError("text tensor dtypes and activation are required")
        if not self.eos_token_ids or any(item < 0 for item in self.eos_token_ids):
            raise Qwen4ExpConfigError("at least one non-negative EOS token is required")
        if len(self.layer_types) != self.num_hidden_layers:
            raise Qwen4ExpConfigError("layer_types do not match num_hidden_layers")
        allowed = {"linear_attention", "full_attention"}
        if any(kind not in allowed for kind in self.layer_types):
            raise Qwen4ExpConfigError("unknown Qwen4Exp layer type")
        if self.output_gate_type != "sigmoid":
            raise Qwen4ExpConfigError("Qwen4Exp GDN output gate must be sigmoid")
        if self.num_experts_per_tok > self.num_experts:
            raise Qwen4ExpConfigError("active experts cannot exceed total experts")
        if not self.ple_layer_ids:
            raise Qwen4ExpConfigError("Qwen4Exp requires at least one PLE layer")
        if tuple(sorted(set(self.ple_layer_ids))) != self.ple_layer_ids:
            raise Qwen4ExpConfigError("PLE layer ids must be sorted and unique")
        if any(not 1 <= item <= self.num_hidden_layers for item in self.ple_layer_ids):
            raise Qwen4ExpConfigError("PLE layer ids are one-based and in range")
        if not self.mrope_section or any(item < 1 for item in self.mrope_section):
            raise Qwen4ExpConfigError("M-RoPE section must contain positive axes")
        if not 0.0 < self.partial_rotary_factor <= 1.0:
            raise Qwen4ExpConfigError("partial rotary factor must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class Qwen4ExpVisionConfig:
    depth: int
    hidden_size: int
    intermediate_size: int
    num_heads: int
    in_channels: int
    num_position_embeddings: int
    patch_size: int
    spatial_merge_size: int
    temporal_patch_size: int
    out_hidden_size: int
    hidden_act: str
    deepstack_visual_indexes: tuple[int, ...]

    def __post_init__(self) -> None:
        values = (
            self.depth,
            self.hidden_size,
            self.intermediate_size,
            self.num_heads,
            self.in_channels,
            self.num_position_embeddings,
            self.patch_size,
            self.spatial_merge_size,
            self.temporal_patch_size,
            self.out_hidden_size,
        )
        if any(value < 1 for value in values):
            raise Qwen4ExpConfigError("vision geometry must be positive")
        if not self.hidden_act:
            raise Qwen4ExpConfigError("vision activation must not be empty")
        if tuple(sorted(set(self.deepstack_visual_indexes))) != (
            self.deepstack_visual_indexes
        ):
            raise Qwen4ExpConfigError(
                "deepstack visual indexes must be sorted and unique"
            )
        if any(not 0 <= item < self.depth for item in self.deepstack_visual_indexes):
            raise Qwen4ExpConfigError("deepstack visual index is out of range")


@dataclass(frozen=True, slots=True)
class Qwen4ExpCheckpointConfig:
    architectures: tuple[str, ...]
    model_type: str
    image_token_id: int
    video_token_id: int
    vision_start_token_id: int
    vision_end_token_id: int
    language_model_only: bool
    tie_word_embeddings: bool
    text: Qwen4ExpTextConfig
    vision: Qwen4ExpVisionConfig
    quantization_bits: int
    quantization_group_size: int
    quantization_mode: str
    quantization_overrides: tuple[Qwen4ExpQuantizationRecipe, ...]

    def __post_init__(self) -> None:
        if "Qwen4ExpForConditionalGeneration" not in self.architectures:
            raise Qwen4ExpConfigError("Qwen4Exp architecture marker is missing")
        if self.model_type != "qwen4_exp":
            raise Qwen4ExpConfigError("model_type must be qwen4_exp")
        if self.text.hidden_size != self.vision.out_hidden_size:
            raise Qwen4ExpConfigError(
                "vision output width must match the text hidden size"
            )
        token_ids = (
            self.image_token_id,
            self.video_token_id,
            self.vision_start_token_id,
            self.vision_end_token_id,
        )
        if any(item < 0 for item in token_ids) or len(set(token_ids)) != 4:
            raise Qwen4ExpConfigError(
                "vision token ids must be unique and non-negative"
            )
        if self.tie_word_embeddings != self.text.tie_word_embeddings:
            raise Qwen4ExpConfigError("embedding tie declarations disagree")
        if self.quantization_bits < 1 or self.quantization_group_size < 1:
            raise Qwen4ExpConfigError("quantization geometry must be positive")
        if not self.quantization_mode:
            raise Qwen4ExpConfigError("quantization mode must not be empty")
        paths = tuple(item.module_path for item in self.quantization_overrides)
        if paths != tuple(sorted(set(paths))):
            raise Qwen4ExpConfigError(
                "quantization override paths must be sorted and unique"
            )

    def quantization_recipe(
        self,
        module_path: str,
    ) -> tuple[int, int, str]:
        for recipe in self.quantization_overrides:
            if recipe.module_path == module_path:
                return recipe.bits, recipe.group_size, recipe.mode
        return (
            self.quantization_bits,
            self.quantization_group_size,
            self.quantization_mode,
        )


def parse_qwen4_exp_config(raw: Mapping[str, Any]) -> Qwen4ExpCheckpointConfig:
    """Parse only fields that determine target execution and artifact truth."""

    root = _mapping("config", raw)
    text = _mapping("text_config", root.get("text_config"))
    vision = _mapping("vision_config", root.get("vision_config"))
    quantization = _mapping("quantization_config", root.get("quantization_config"))
    legacy_quantization = root.get("quantization")
    if legacy_quantization is not None and legacy_quantization != quantization:
        raise Qwen4ExpConfigError("quantization declarations disagree")
    rope = _mapping("text_config.rope_parameters", text.get("rope_parameters"))
    mtp_raw = _mapping("text_config.mtp", text.get("mtp"))

    text_model_type = _string("text_config.model_type", text.get("model_type"))
    if text_model_type != "qwen4_exp_text":
        raise Qwen4ExpConfigError("text_config.model_type must be qwen4_exp_text")

    mtp_depth = _integer(
        "text_config.mtp.num_hidden_layers",
        mtp_raw.get("num_hidden_layers"),
    )
    declared_mtp_depth = _integer(
        "text_config.mtp_num_hidden_layers",
        text.get("mtp_num_hidden_layers"),
    )
    if mtp_depth != declared_mtp_depth:
        raise Qwen4ExpConfigError("MTP depth declarations disagree")

    text_config = Qwen4ExpTextConfig(
        hidden_size=_integer("text_config.hidden_size", text.get("hidden_size")),
        num_hidden_layers=_integer(
            "text_config.num_hidden_layers", text.get("num_hidden_layers")
        ),
        num_attention_heads=_integer(
            "text_config.num_attention_heads", text.get("num_attention_heads")
        ),
        num_key_value_heads=_integer(
            "text_config.num_key_value_heads", text.get("num_key_value_heads")
        ),
        head_dim=_integer("text_config.head_dim", text.get("head_dim")),
        rms_norm_eps=_number("text_config.rms_norm_eps", text.get("rms_norm_eps")),
        vocab_size=_integer("text_config.vocab_size", text.get("vocab_size")),
        max_position_embeddings=_integer(
            "text_config.max_position_embeddings",
            text.get("max_position_embeddings"),
        ),
        tie_word_embeddings=_boolean(
            "text_config.tie_word_embeddings",
            text.get("tie_word_embeddings"),
        ),
        attention_bias=_boolean(
            "text_config.attention_bias", text.get("attention_bias")
        ),
        hidden_act=_string("text_config.hidden_act", text.get("hidden_act")),
        dtype=_string("text_config.dtype", text.get("dtype")),
        mamba_ssm_dtype=_string(
            "text_config.mamba_ssm_dtype", text.get("mamba_ssm_dtype")
        ),
        use_cache=_boolean("text_config.use_cache", text.get("use_cache")),
        linear_num_value_heads=_integer(
            "text_config.linear_num_value_heads",
            text.get("linear_num_value_heads"),
        ),
        linear_num_key_heads=_integer(
            "text_config.linear_num_key_heads",
            text.get("linear_num_key_heads"),
        ),
        linear_key_head_dim=_integer(
            "text_config.linear_key_head_dim", text.get("linear_key_head_dim")
        ),
        linear_value_head_dim=_integer(
            "text_config.linear_value_head_dim",
            text.get("linear_value_head_dim"),
        ),
        linear_conv_kernel_dim=_integer(
            "text_config.linear_conv_kernel_dim",
            text.get("linear_conv_kernel_dim"),
        ),
        output_gate_type=_string(
            "text_config.output_gate_type", text.get("output_gate_type")
        ),
        num_experts=_integer("text_config.num_experts", text.get("num_experts")),
        num_experts_per_tok=_integer(
            "text_config.num_experts_per_tok", text.get("num_experts_per_tok")
        ),
        moe_intermediate_size=_integer(
            "text_config.moe_intermediate_size",
            text.get("moe_intermediate_size"),
        ),
        shared_expert_intermediate_size=_integer(
            "text_config.shared_expert_intermediate_size",
            text.get("shared_expert_intermediate_size"),
        ),
        norm_topk_prob=_boolean_default(
            "text_config.norm_topk_prob",
            text.get("norm_topk_prob"),
            True,
        ),
        decoder_sparse_step=_integer_default(
            "text_config.decoder_sparse_step",
            text.get("decoder_sparse_step"),
            1,
        ),
        layer_types=_strings("text_config.layer_types", text.get("layer_types")),
        full_attention_interval=_integer(
            "text_config.full_attention_interval",
            text.get("full_attention_interval"),
        ),
        hc_count=_integer("text_config.hc_count", text.get("hc_count")),
        hc_lowrank=_integer("text_config.hc_lowrank", text.get("hc_lowrank")),
        indexer_n_heads=_integer(
            "text_config.indexer_n_heads", text.get("indexer_n_heads")
        ),
        indexer_kv_heads=_integer(
            "text_config.indexer_kv_heads", text.get("indexer_kv_heads")
        ),
        indexer_head_dim=_integer(
            "text_config.indexer_head_dim", text.get("indexer_head_dim")
        ),
        indexer_budget=_integer(
            "text_config.indexer_budget", text.get("indexer_budget")
        ),
        indexer_compress_ratio=_integer(
            "text_config.indexer_compress_ratio",
            text.get("indexer_compress_ratio"),
        ),
        ple_layer_ids=_integers("text_config.ple_layer_ids", text.get("ple_layer_ids")),
        ple_embed_dim=_integer("text_config.ple_embed_dim", text.get("ple_embed_dim")),
        ple_conv_kernel_size=_integer(
            "text_config.ple_conv_kernel_size",
            text.get("ple_conv_kernel_size"),
        ),
        ngram_size=_integer("text_config.ngram_size", text.get("ngram_size")),
        heads_per_ngram=_integer(
            "text_config.heads_per_ngram", text.get("heads_per_ngram")
        ),
        ngram_vocab_size_base=_integer(
            "text_config.ngram_vocab_size_base",
            text.get("ngram_vocab_size_base"),
        ),
        make_ngram_vocab_size_divisible_by=_integer(
            "text_config.make_ngram_vocab_size_divisible_by",
            text.get("make_ngram_vocab_size_divisible_by"),
        ),
        split_ngram_parts=_integer(
            "text_config.split_ngram_parts", text.get("split_ngram_parts")
        ),
        seed=_integer_default("text_config.seed", text.get("seed"), 1234),
        eos_token_ids=_integer_or_integers(
            "text_config.eos_token_id", text.get("eos_token_id")
        ),
        partial_rotary_factor=_number(
            "text_config.rope_parameters.partial_rotary_factor",
            rope.get("partial_rotary_factor"),
        ),
        rope_theta=_integer(
            "text_config.rope_parameters.rope_theta", rope.get("rope_theta")
        ),
        mrope_section=_integers(
            "text_config.rope_parameters.mrope_section",
            rope.get("mrope_section"),
        ),
        mrope_interleaved=_boolean(
            "text_config.rope_parameters.mrope_interleaved",
            rope.get("mrope_interleaved"),
        ),
        mtp_use_dedicated_embeddings=_boolean(
            "text_config.mtp_use_dedicated_embeddings",
            text.get("mtp_use_dedicated_embeddings"),
        ),
        mtp=Qwen4ExpMtpConfig(
            num_hidden_layers=mtp_depth,
            layer_types=_strings(
                "text_config.mtp.layer_types", mtp_raw.get("layer_types")
            ),
            hybrid=_boolean("text_config.mtp.hybrid", mtp_raw.get("hybrid")),
            rope_theta=_integer(
                "text_config.mtp.rope_theta", mtp_raw.get("rope_theta")
            ),
        ),
    )
    vision_config = Qwen4ExpVisionConfig(
        depth=_integer("vision_config.depth", vision.get("depth")),
        hidden_size=_integer("vision_config.hidden_size", vision.get("hidden_size")),
        intermediate_size=_integer(
            "vision_config.intermediate_size", vision.get("intermediate_size")
        ),
        num_heads=_integer("vision_config.num_heads", vision.get("num_heads")),
        in_channels=_integer("vision_config.in_channels", vision.get("in_channels")),
        num_position_embeddings=_integer(
            "vision_config.num_position_embeddings",
            vision.get("num_position_embeddings"),
        ),
        patch_size=_integer("vision_config.patch_size", vision.get("patch_size")),
        spatial_merge_size=_integer(
            "vision_config.spatial_merge_size", vision.get("spatial_merge_size")
        ),
        temporal_patch_size=_integer(
            "vision_config.temporal_patch_size",
            vision.get("temporal_patch_size"),
        ),
        out_hidden_size=_integer(
            "vision_config.out_hidden_size", vision.get("out_hidden_size")
        ),
        hidden_act=_string("vision_config.hidden_act", vision.get("hidden_act")),
        deepstack_visual_indexes=_integers(
            "vision_config.deepstack_visual_indexes",
            vision.get("deepstack_visual_indexes", ()),
        ),
    )
    return Qwen4ExpCheckpointConfig(
        architectures=_strings("architectures", root.get("architectures")),
        model_type=_string("model_type", root.get("model_type")),
        image_token_id=_integer("image_token_id", root.get("image_token_id")),
        video_token_id=_integer("video_token_id", root.get("video_token_id")),
        vision_start_token_id=_integer(
            "vision_start_token_id", root.get("vision_start_token_id")
        ),
        vision_end_token_id=_integer(
            "vision_end_token_id", root.get("vision_end_token_id")
        ),
        language_model_only=_boolean(
            "language_model_only", root.get("language_model_only")
        ),
        tie_word_embeddings=_boolean(
            "tie_word_embeddings", root.get("tie_word_embeddings")
        ),
        text=text_config,
        vision=vision_config,
        quantization_bits=_integer(
            "quantization_config.bits", quantization.get("bits")
        ),
        quantization_group_size=_integer(
            "quantization_config.group_size", quantization.get("group_size")
        ),
        quantization_mode=_string("quantization_config.mode", quantization.get("mode")),
        quantization_overrides=_quantization_overrides(quantization),
    )


def _mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise Qwen4ExpConfigError(f"{name} must be a string-keyed mapping")
    return value


def _string(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise Qwen4ExpConfigError(f"{name} must be a non-empty string")
    return value


def _integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Qwen4ExpConfigError(f"{name} must be an integer")
    return value


def _boolean(name: str, value: Any) -> bool:
    if type(value) is not bool:
        raise Qwen4ExpConfigError(f"{name} must be a boolean")
    return value


def _boolean_default(name: str, value: Any, default: bool) -> bool:
    if value is None:
        return default
    return _boolean(name, value)


def _integer_default(name: str, value: Any, default: int) -> int:
    if value is None:
        return default
    return _integer(name, value)


def _number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise Qwen4ExpConfigError(f"{name} must be numeric")
    return float(value)


def _sequence(name: str, value: Any) -> Sequence[Any]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise Qwen4ExpConfigError(f"{name} must be a sequence")
    return value


def _strings(name: str, value: Any) -> tuple[str, ...]:
    items = _sequence(name, value)
    if any(not isinstance(item, str) or not item for item in items):
        raise Qwen4ExpConfigError(f"{name} must contain non-empty strings")
    return tuple(items)


def _integers(name: str, value: Any) -> tuple[int, ...]:
    items = _sequence(name, value)
    if any(isinstance(item, bool) or not isinstance(item, int) for item in items):
        raise Qwen4ExpConfigError(f"{name} must contain integers")
    return tuple(items)


def _integer_or_integers(name: str, value: Any) -> tuple[int, ...]:
    if isinstance(value, bool):
        raise Qwen4ExpConfigError(f"{name} must be an integer or sequence")
    if isinstance(value, int):
        return (value,)
    return _integers(name, value)


def _quantization_overrides(
    raw: Mapping[str, Any],
) -> tuple[Qwen4ExpQuantizationRecipe, ...]:
    base_keys = {"bits", "group_size", "mode"}
    recipes: list[Qwen4ExpQuantizationRecipe] = []
    for module_path, value in raw.items():
        if module_path in base_keys:
            continue
        recipe = _mapping(
            f"quantization_config.{module_path}",
            value,
        )
        if set(recipe) != base_keys:
            raise Qwen4ExpConfigError(
                f"quantization recipe has unknown shape: {module_path}"
            )
        recipes.append(
            Qwen4ExpQuantizationRecipe(
                module_path=module_path,
                bits=_integer(f"{module_path}.bits", recipe.get("bits")),
                group_size=_integer(
                    f"{module_path}.group_size",
                    recipe.get("group_size"),
                ),
                mode=_string(f"{module_path}.mode", recipe.get("mode")),
            )
        )
    return tuple(sorted(recipes, key=lambda item: item.module_path))


__all__ = [
    "Qwen4ExpCheckpointConfig",
    "Qwen4ExpConfigError",
    "Qwen4ExpMtpConfig",
    "Qwen4ExpQuantizationRecipe",
    "Qwen4ExpTextConfig",
    "Qwen4ExpVisionConfig",
    "parse_qwen4_exp_config",
]
