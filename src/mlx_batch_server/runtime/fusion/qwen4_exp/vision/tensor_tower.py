# SPDX-License-Identifier: Apache-2.0
# Derived from youssofal/mtplx@6d0ddf0575faa9acf77e63c57e48ea1602a7e4ab
# mtplx/vision/qwen3_vl_tower.py, itself adapted from mlx-vlm (Apache-2.0).
# Modified by LibraxisAI to implement the target's opaque whole-request port.
"""Executable Qwen3VL vision tower for Qwen4Exp checkpoints."""

from __future__ import annotations

import math
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn

from ..model.load_plan import Qwen4ExpModelLoadPlan
from .processing import OpaqueRows, VisionContractError
from .tower import (
    DeepstackFeatureReceipt,
    VisionEmbeddingSlice,
    VisionTowerOutput,
    VisionTowerRequest,
    validate_tower_output,
)

if TYPE_CHECKING:
    from ..model.config import Qwen4ExpCheckpointConfig, Qwen4ExpVisionConfig

_FUSED_SDPA_DIMS = (64, 80, 128)
_VISION_PREFIXES = ("vision_tower.", "model.visual.")


def resolve_vision_prefix(
    weights: Mapping[str, object] | Sequence[str],
) -> str | None:
    weight_keys = weights.keys() if isinstance(weights, Mapping) else weights
    for prefix in _VISION_PREFIXES:
        if any(key.startswith(prefix) for key in weight_keys):
            return prefix
    return None


def _validate_tower_config(config: Qwen4ExpVisionConfig) -> None:
    if config.hidden_size % config.num_heads:
        raise VisionContractError(
            "invalid_tower_heads",
            "vision hidden size must be divisible by head count",
        )
    side = math.isqrt(config.num_position_embeddings)
    if side * side != config.num_position_embeddings:
        raise VisionContractError(
            "invalid_position_embeddings",
            "vision position embedding count must be a perfect square",
        )


def _activation(name: str) -> nn.Module:
    if name in {"gelu_pytorch_tanh", "gelu_new"}:
        return nn.GELU(approx="tanh")
    if name == "gelu":
        return nn.GELU()
    if name in {"silu", "swish"}:
        return nn.SiLU()
    raise VisionContractError(
        "unsupported_vision_activation",
        f"unsupported checkpoint vision activation: {name}",
    )


def _fused_sdpa(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float,
) -> mx.array:
    head_dim = int(query.shape[-1])
    target = next(
        (candidate for candidate in _FUSED_SDPA_DIMS if head_dim <= candidate),
        head_dim,
    )
    if target != head_dim:
        padding = [(0, 0)] * (query.ndim - 1) + [(0, target - head_dim)]
        query = mx.pad(query, padding)
        key = mx.pad(key, padding)
        value = mx.pad(value, padding)
    output = mx.fast.scaled_dot_product_attention(
        query,
        key,
        value,
        scale=scale,
    )
    return output[..., :head_dim]


def _rotate_half(value: mx.array) -> mx.array:
    first = value[..., : value.shape[-1] // 2]
    second = value[..., value.shape[-1] // 2 :]
    return mx.concatenate([-second, first], axis=-1)


def _apply_rotary_pos_emb_vision(
    tensor: mx.array,
    frequencies: mx.array,
) -> mx.array:
    original_dtype = tensor.dtype
    cosine = mx.expand_dims(mx.cos(frequencies), axis=1)
    cosine = mx.expand_dims(mx.tile(cosine, (1, 1, 2)), axis=0)
    sine = mx.expand_dims(mx.sin(frequencies), axis=1)
    sine = mx.expand_dims(mx.tile(sine, (1, 1, 2)), axis=0)
    output = tensor * cosine + _rotate_half(tensor) * sine
    return output.astype(original_dtype)


def _conv_weight_in_mlx_layout(array: mx.array) -> bool:
    shape = tuple(int(value) for value in array.shape)
    if len(shape) != 5:
        return False
    _, out_channels, kernel_h, kernel_w, temporal = shape
    if temporal == 3:
        return True
    return (
        out_channels >= kernel_h and out_channels >= kernel_w and kernel_h == kernel_w
    )


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta

    def __call__(self, sequence_length: int) -> mx.array:
        inverse_frequency = 1.0 / (
            self.theta ** (mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim)
        )
        sequence = mx.arange(sequence_length, dtype=inverse_frequency.dtype)
        return mx.outer(sequence, inverse_frequency)


class PatchEmbed(nn.Module):
    def __init__(self, config: Qwen4ExpVisionConfig) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.hidden_size = config.hidden_size
        kernel_size = (
            config.temporal_patch_size,
            config.patch_size,
            config.patch_size,
        )
        self.proj = nn.Conv3d(
            config.in_channels,
            config.hidden_size,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = hidden_states.reshape(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        ).moveaxis(1, 4)
        hidden_states = self.proj(hidden_states)
        return hidden_states.reshape(-1, self.hidden_size)


class PatchMerger(nn.Module):
    def __init__(
        self,
        config: Qwen4ExpVisionConfig,
        *,
        use_postshuffle_norm: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        norm_width = self.hidden_size if use_postshuffle_norm else config.hidden_size
        self.norm = nn.LayerNorm(norm_width, eps=1e-6)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = _activation(config.hidden_act)
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

    def __call__(self, value: mx.array) -> mx.array:
        if self.use_postshuffle_norm:
            value = value.reshape(-1, self.hidden_size)
        value = self.norm(value).reshape(-1, self.hidden_size)
        return self.linear_fc2(self.act_fn(self.linear_fc1(value)))


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def __call__(
        self,
        value: mx.array,
        split_indices: list[int],
        rotary_pos_emb: mx.array,
    ) -> mx.array:
        sequence_length = int(value.shape[0])
        qkv = (
            self.qkv(value)
            .reshape(sequence_length, 3, self.num_heads, -1)
            .transpose(1, 0, 2, 3)
        )
        query, key, projected_value = mx.split(qkv, 3)
        query = _apply_rotary_pos_emb_vision(
            mx.expand_dims(query, 0),
            rotary_pos_emb,
        )[0]
        key = _apply_rotary_pos_emb_vision(
            mx.expand_dims(key, 0),
            rotary_pos_emb,
        )[0]
        query = query.transpose(0, 2, 1, 3)
        key = key.transpose(0, 2, 1, 3)
        projected_value = projected_value.transpose(0, 2, 1, 3)

        # Each image frame is independent inside the vision tower. This is
        # not the language trunk's causal attention policy.
        splits = [
            mx.split(tensor, split_indices, axis=2)
            for tensor in (query, key, projected_value)
        ]
        outputs = [
            _fused_sdpa(q_part, k_part, v_part, self.scale)
            for q_part, k_part, v_part in zip(*splits, strict=True)
        ]
        output = mx.concatenate(outputs, axis=2)
        output = output.transpose(0, 2, 1, 3).reshape(sequence_length, -1)
        return self.proj(output)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, activation: str) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(dim, hidden_dim, bias=True)
        self.linear_fc2 = nn.Linear(hidden_dim, dim, bias=True)
        self.act_fn = _activation(activation)

    def __call__(self, value: mx.array) -> mx.array:
        return self.linear_fc2(self.act_fn(self.linear_fc1(value)))


class VisionBlock(nn.Module):
    def __init__(self, config: Qwen4ExpVisionConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Attention(config.hidden_size, config.num_heads)
        self.mlp = MLP(
            config.hidden_size,
            config.intermediate_size,
            config.hidden_act,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        split_indices: list[int],
        rotary_pos_emb: mx.array,
    ) -> mx.array:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            split_indices,
            rotary_pos_emb,
        )
        return hidden_states + self.mlp(self.norm2(hidden_states))


class Qwen4ExpVisionTensorTower(nn.Module):
    """Qwen3-VL-derived tower implementing the Qwen4Exp vision port."""

    def __init__(self, plan: Qwen4ExpModelLoadPlan) -> None:
        super().__init__()
        config = plan.config.vision
        _validate_tower_config(config)
        self.load_plan = plan
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_embed = PatchEmbed(config)
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)
        self.pos_embed = nn.Embedding(
            config.num_position_embeddings,
            config.hidden_size,
        )
        self.num_grid_per_side = math.isqrt(config.num_position_embeddings)
        self.blocks = [VisionBlock(config) for _ in range(config.depth)]
        self.merger = PatchMerger(config)
        self.deepstack_visual_indexes = config.deepstack_visual_indexes
        self.deepstack_merger_list = [
            PatchMerger(config, use_postshuffle_norm=True)
            for _ in config.deepstack_visual_indexes
        ]
        self._owner_thread_id = threading.get_ident()

    @classmethod
    def from_load_plan(
        cls,
        plan: Qwen4ExpModelLoadPlan,
    ) -> Qwen4ExpVisionTensorTower:
        """Open only shards sealed by the immutable checkpoint load plan."""

        if not isinstance(plan, Qwen4ExpModelLoadPlan):
            raise VisionContractError(
                "invalid_load_plan",
                "vision tower requires Qwen4ExpModelLoadPlan",
            )
        planned_keys = plan.artifacts.weight_keys
        prefix = resolve_vision_prefix(planned_keys)
        if prefix is None:
            raise VisionContractError(
                "missing_vision_weights",
                "load plan has no supported vision tensor prefix",
            )
        tower = cls(plan)
        vision_keys = frozenset(key for key in planned_keys if key.startswith(prefix))
        weights: dict[str, mx.array] = {}
        vision_shards = plan.artifacts.shards_for_prefix(prefix)
        if not vision_shards:
            raise VisionContractError(
                "missing_vision_shards",
                "load plan maps no approved shard to vision tensors",
            )
        for shard in vision_shards:
            loaded = mx.load(str(_planned_shard_path(plan, shard)))
            for key, value in loaded.items():
                if key not in vision_keys:
                    continue
                local_key = key[len(prefix) :]
                if local_key in weights:
                    raise VisionContractError(
                        "duplicate_vision_weight",
                        f"planned vision tensor appears twice: {key}",
                    )
                weights[local_key] = value
        expected_local_keys = {key[len(prefix) :] for key in vision_keys}
        missing = sorted(expected_local_keys - set(weights))
        if missing:
            raise VisionContractError(
                "missing_planned_vision_weights",
                f"planned vision tensors are absent from approved shards: {missing!r}",
            )
        conv_key = "patch_embed.proj.weight"
        if conv_key in weights and not _conv_weight_in_mlx_layout(weights[conv_key]):
            weights[conv_key] = weights[conv_key].transpose(0, 2, 3, 4, 1)
        _quantize_from_plan(
            tower,
            plan.config,
            prefix,
            expected_local_keys,
        )
        tower.load_weights(list(weights.items()), strict=True)
        mx.eval(tower.parameters())
        return tower

    def forward(self, request: VisionTowerRequest) -> VisionTowerOutput:
        self._assert_owner()
        if not isinstance(request, VisionTowerRequest):
            raise VisionContractError(
                "invalid_tower_request",
                "tensor tower requires VisionTowerRequest",
            )
        processed = request.processed
        if any(
            image.spatial_merge_size != self.spatial_merge_size
            for image in processed.images
        ):
            raise VisionContractError(
                "tower_merge_mismatch",
                "tower and preprocessor merge sizes disagree",
            )
        grids = [image.grid_thw for image in processed.images]
        pixel_values = processed.pixel_values.handle
        pixel_shape = _shape(pixel_values, "pixel_values")
        if len(pixel_shape) != 2 or pixel_shape[0] != processed.pixel_values.row_count:
            raise VisionContractError(
                "tower_pixel_shape_mismatch",
                "pixel tensor rows do not match preprocessing receipt",
            )
        embeddings, deepstack = self._forward_tensors(pixel_values, grids)
        embedding_shape = _shape(embeddings, "vision embeddings")
        if len(embedding_shape) != 2:
            raise VisionContractError(
                "invalid_tower_embedding_shape",
                "vision embeddings must have shape [rows, hidden]",
            )
        if embedding_shape[1] != self.config.out_hidden_size:
            raise VisionContractError(
                "tower_hidden_size_mismatch",
                "vision embedding width does not match checkpoint config",
            )

        slices: list[VisionEmbeddingSlice] = []
        cursor = 0
        for image in processed.images:
            slices.append(
                VisionEmbeddingSlice(
                    identity=request.identity,
                    image_index=image.image_index,
                    content_digest=image.content_digest,
                    start=cursor,
                    end=cursor + image.pad_rows,
                )
            )
            cursor += image.pad_rows
        deepstack_receipts: list[DeepstackFeatureReceipt] = []
        for injection_index, rows in deepstack:
            row_shape = _shape(rows, "deepstack rows")
            if len(row_shape) != 2 or row_shape[0] != cursor:
                raise VisionContractError(
                    "deepstack_row_mismatch",
                    "deepstack rows must match merged vision embeddings",
                )
            deepstack_receipts.append(
                DeepstackFeatureReceipt(
                    identity=request.identity,
                    injection_index=injection_index,
                    rows=OpaqueRows(handle=rows, row_count=row_shape[0]),
                )
            )
        output = VisionTowerOutput(
            identity=request.identity,
            embeddings=OpaqueRows(
                handle=embeddings,
                row_count=embedding_shape[0],
            ),
            image_slices=tuple(slices),
            deepstack=tuple(deepstack_receipts),
        )
        return validate_tower_output(request, output)

    def _forward_tensors(
        self,
        pixel_values: mx.array,
        grids: list[tuple[int, int, int]],
    ) -> tuple[mx.array, list[tuple[int, mx.array]]]:
        hidden_states = self.patch_embed(
            pixel_values.astype(self.patch_embed.proj.weight.dtype)
        )
        hidden_states = hidden_states + self._position_embeddings(grids)
        rotary_pos_emb = self._rotary_positions(grids)
        split_indices: list[int] = []
        total = 0
        for temporal, height, width in grids:
            for _ in range(temporal):
                total += height * width
                split_indices.append(total)
        split_indices = split_indices[:-1]

        deepstack: list[tuple[int, mx.array]] = []
        for layer_number, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                split_indices,
                rotary_pos_emb,
            )
            if layer_number in self.deepstack_visual_indexes:
                index = self.deepstack_visual_indexes.index(layer_number)
                deepstack.append(
                    (index, self.deepstack_merger_list[index](hidden_states))
                )
        return self.merger(hidden_states), deepstack

    def _rotary_positions(
        self,
        grids: list[tuple[int, int, int]],
    ) -> mx.array:
        merge = self.spatial_merge_size
        max_hw = max(max(height, width) for _, height, width in grids)
        frequency_table = self.rotary_pos_emb(max_hw)
        position_ids: list[mx.array] = []
        for frames, height, width in grids:
            merged_h = height // merge
            merged_w = width // merge
            block_rows = mx.arange(merged_h)
            block_columns = mx.arange(merged_w)
            intra_row = mx.arange(merge)
            intra_column = mx.arange(merge)
            row_index = (
                block_rows[:, None, None, None] * merge + intra_row[None, None, :, None]
            )
            column_index = (
                block_columns[None, :, None, None] * merge
                + intra_column[None, None, None, :]
            )
            shape = (merged_h, merged_w, merge, merge)
            row_index = mx.broadcast_to(row_index, shape).reshape(-1)
            column_index = mx.broadcast_to(column_index, shape).reshape(-1)
            coordinates = mx.stack([row_index, column_index], axis=-1)
            if frames > 1:
                coordinates = mx.tile(coordinates, (frames, 1))
            position_ids.append(coordinates)
        positions = mx.concatenate(position_ids, axis=0)
        height_embeddings = frequency_table[positions[:, 0]]
        width_embeddings = frequency_table[positions[:, 1]]
        return mx.concatenate([height_embeddings, width_embeddings], axis=-1)

    def _position_embeddings(
        self,
        grids: list[tuple[int, int, int]],
    ) -> mx.array:
        index_lists: list[list[int]] = [[] for _ in range(4)]
        weight_lists: list[list[float]] = [[] for _ in range(4)]
        for _, height, width in grids:
            height_indexes = mx.linspace(0, self.num_grid_per_side - 1, height)
            width_indexes = mx.linspace(0, self.num_grid_per_side - 1, width)
            height_floor = height_indexes.astype(mx.int32)
            width_floor = width_indexes.astype(mx.int32)
            height_ceil = mx.minimum(
                height_floor + 1,
                self.num_grid_per_side - 1,
            )
            width_ceil = mx.minimum(
                width_floor + 1,
                self.num_grid_per_side - 1,
            )
            delta_h = height_indexes - height_floor.astype(mx.float32)
            delta_w = width_indexes - width_floor.astype(mx.float32)
            base_h = height_floor * self.num_grid_per_side
            base_h_ceil = height_ceil * self.num_grid_per_side
            indexes = (
                (base_h[:, None] + width_floor[None, :]).flatten(),
                (base_h[:, None] + width_ceil[None, :]).flatten(),
                (base_h_ceil[:, None] + width_floor[None, :]).flatten(),
                (base_h_ceil[:, None] + width_ceil[None, :]).flatten(),
            )
            weights = (
                ((1 - delta_h)[:, None] * (1 - delta_w)[None, :]).flatten(),
                ((1 - delta_h)[:, None] * delta_w[None, :]).flatten(),
                (delta_h[:, None] * (1 - delta_w)[None, :]).flatten(),
                (delta_h[:, None] * delta_w[None, :]).flatten(),
            )
            for index in range(4):
                index_lists[index].extend(indexes[index].tolist())
                weight_lists[index].extend(weights[index].tolist())

        index_tensor = mx.array(index_lists, dtype=mx.int32)
        weight_tensor = mx.array(
            weight_lists,
            dtype=self.pos_embed.weight.dtype,
        )
        embeddings = self.pos_embed(index_tensor) * weight_tensor[:, :, None]
        patch_embeddings = embeddings[0] + embeddings[1] + embeddings[2] + embeddings[3]
        split_sizes = [height * width for _, height, width in grids]
        split_indices: list[int] = []
        total = 0
        for size in split_sizes[:-1]:
            total += size
            split_indices.append(total)
        split_embeddings = (
            mx.split(patch_embeddings, split_indices, axis=0)
            if split_indices
            else [patch_embeddings]
        )
        merge = self.spatial_merge_size
        permuted: list[mx.array] = []
        for embedding, (frames, height, width) in zip(
            split_embeddings,
            grids,
            strict=True,
        ):
            feature_dim = int(embedding.shape[-1])
            merged_embedding = mx.tile(embedding, (frames, 1))
            merged_embedding = merged_embedding.reshape(
                frames, height, width, feature_dim
            )
            merged_embedding = (
                merged_embedding.reshape(
                    frames,
                    height // merge,
                    merge,
                    width // merge,
                    merge,
                    feature_dim,
                )
                .transpose(0, 1, 3, 2, 4, 5)
                .reshape(-1, feature_dim)
            )
            permuted.append(merged_embedding)
        return mx.concatenate(permuted, axis=0)

    def _assert_owner(self) -> None:
        if threading.get_ident() != self._owner_thread_id:
            raise VisionContractError(
                "owner_thread_violation",
                "vision tower must run on its inference owner thread",
            )


def _planned_shard_path(plan: Qwen4ExpModelLoadPlan, shard: str) -> Path:
    if shard not in plan.artifacts.weight_shards:
        raise VisionContractError(
            "unplanned_weight_shard",
            f"vision tensor load attempted an unplanned shard: {shard}",
        )
    if Path(shard).name != shard:
        raise VisionContractError(
            "invalid_weight_shard",
            "load-plan shard names must be checkpoint basenames",
        )
    model_dir = Path(plan.model_dir)
    if not model_dir.is_absolute():
        raise VisionContractError(
            "invalid_load_plan",
            "load-plan model directory must be absolute",
        )
    return model_dir / shard


def _quantize_from_plan(
    tower: Qwen4ExpVisionTensorTower,
    config: Qwen4ExpCheckpointConfig,
    prefix: str,
    local_weight_keys: set[str],
) -> None:
    quantized_paths = {
        key[: -len(".scales")] for key in local_weight_keys if key.endswith(".scales")
    }
    if not quantized_paths:
        return
    matched_paths: set[str] = set()

    def predicate(path: str, module: nn.Module) -> bool | dict[str, object]:
        if path not in quantized_paths:
            return False
        if not hasattr(module, "to_quantized"):
            raise VisionContractError(
                "invalid_quantized_vision_module",
                f"planned quantized tensor has unsupported module: {path}",
            )
        matched_paths.add(path)
        bits, group_size, mode = _vision_quantization_recipe(
            config,
            prefix,
            path,
        )
        return {
            "bits": bits,
            "group_size": group_size,
            "mode": mode,
        }

    nn.quantize(
        tower,
        group_size=config.quantization_group_size,
        bits=config.quantization_bits,
        mode=config.quantization_mode,
        class_predicate=predicate,
    )
    missing_modules = sorted(quantized_paths - matched_paths)
    if missing_modules:
        raise VisionContractError(
            "missing_quantized_vision_modules",
            f"planned quantized tensors have no tower modules: {missing_modules!r}",
        )


def _vision_quantization_recipe(
    config: Qwen4ExpCheckpointConfig,
    checkpoint_prefix: str,
    local_path: str,
) -> tuple[int, int, str]:
    overrides = {
        recipe.module_path: (recipe.bits, recipe.group_size, recipe.mode)
        for recipe in config.quantization_overrides
    }
    candidates = (
        f"{checkpoint_prefix}{local_path}",
        f"vision_tower.{local_path}",
        f"model.visual.{local_path}",
        local_path,
    )
    matched = tuple(path for path in candidates if path in overrides)
    recipes = {overrides[path] for path in matched}
    if len(recipes) > 1:
        raise VisionContractError(
            "conflicting_vision_quantization",
            f"vision quantization aliases disagree for {local_path}",
        )
    selected = matched[0] if matched else candidates[0]
    return config.quantization_recipe(selected)


def _shape(value: object, label: str) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise VisionContractError(
            "invalid_tensor_handle",
            f"{label} is not an MLX tensor-like value",
        )
    try:
        return tuple(int(item) for item in shape)
    except (TypeError, ValueError) as exc:
        raise VisionContractError(
            "invalid_tensor_handle",
            f"{label} has an invalid shape",
        ) from exc
