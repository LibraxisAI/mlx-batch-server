# SPDX-License-Identifier: Apache-2.0
"""Pure checkpoint inventory for a complete Qwen4Exp text+vision+MTP pack."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Qwen4ExpCheckpointConfig


class Qwen4ExpArtifactError(ValueError):
    """Checkpoint filenames or weight ownership are incomplete or ambiguous."""


@dataclass(frozen=True, slots=True)
class Qwen4ExpArtifactInventory:
    files: tuple[str, ...]
    weight_shards: tuple[str, ...]
    weight_keys: tuple[str, ...]
    weight_map: tuple[tuple[str, str], ...]
    has_language_trunk: bool
    has_embedded_mtp: bool
    has_embedded_vision: bool
    has_embedded_ple: bool
    digest: str

    def __post_init__(self) -> None:
        if tuple(sorted(set(self.files))) != self.files:
            raise Qwen4ExpArtifactError("artifact files must be sorted and unique")
        if tuple(sorted(set(self.weight_shards))) != self.weight_shards:
            raise Qwen4ExpArtifactError("weight shards must be sorted and unique")
        if tuple(sorted(set(self.weight_keys))) != self.weight_keys:
            raise Qwen4ExpArtifactError("weight keys must be sorted and unique")
        mapped_keys = tuple(key for key, _ in self.weight_map)
        if mapped_keys != self.weight_keys:
            raise Qwen4ExpArtifactError("weight map must cover every weight key")
        if any(shard not in self.weight_shards for _, shard in self.weight_map):
            raise Qwen4ExpArtifactError("weight map names an unknown shard")
        if len(self.digest) != 64:
            raise Qwen4ExpArtifactError("inventory digest must be SHA-256")

    def shards_for_prefix(self, prefix: str) -> tuple[str, ...]:
        if not prefix:
            raise Qwen4ExpArtifactError("weight prefix must not be empty")
        return tuple(
            sorted({shard for key, shard in self.weight_map if key.startswith(prefix)})
        )


def inspect_qwen4_exp_artifacts(
    *,
    config: Qwen4ExpCheckpointConfig,
    file_names: Sequence[str],
    weight_map: Mapping[str, str],
) -> Qwen4ExpArtifactInventory:
    """Validate an already parsed model index without opening model shards."""

    files = _normalized_names(
        "file_names",
        file_names,
        allow_duplicates=False,
    )
    required_metadata = {
        "config.json",
        "model.safetensors.index.json",
        "preprocessor_config.json",
        "tokenizer.json",
    }
    missing = sorted(required_metadata - set(files))
    if missing:
        raise Qwen4ExpArtifactError(
            f"required checkpoint files are missing: {missing!r}"
        )
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise Qwen4ExpArtifactError("weight_map must be a non-empty mapping")
    if any(not isinstance(key, str) or not key for key in weight_map):
        raise Qwen4ExpArtifactError("weight_map keys must be non-empty strings")
    if any(not isinstance(value, str) or not value for value in weight_map.values()):
        raise Qwen4ExpArtifactError("weight_map shard names must be non-empty strings")

    weight_keys = tuple(sorted(weight_map))
    frozen_weight_map = tuple(sorted(weight_map.items()))
    shards = _normalized_names(
        "weight shards",
        tuple(weight_map.values()),
        allow_duplicates=True,
    )
    missing_shards = sorted(set(shards) - set(files))
    if missing_shards:
        raise Qwen4ExpArtifactError(
            f"weight-map shards are missing from the snapshot: {missing_shards!r}"
        )
    unexpected_shards = sorted(
        name
        for name in files
        if name.startswith("model-")
        and name.endswith(".safetensors")
        and name not in shards
    )
    if unexpected_shards:
        raise Qwen4ExpArtifactError(
            f"unindexed model shards are present: {unexpected_shards!r}"
        )

    has_language = any(key.startswith("language_model.") for key in weight_keys)
    has_mtp = any(key.startswith("mtp.") for key in weight_keys)
    has_vision = any(key.startswith("vision_tower.") for key in weight_keys)
    has_ple = any(".ple." in key for key in weight_keys)
    required_components = {
        "language trunk": has_language,
        "embedded MTP": has_mtp if config.text.mtp.num_hidden_layers else True,
        "embedded vision tower": has_vision,
        "embedded PLE": has_ple if config.text.ple_layer_ids else True,
    }
    absent = sorted(
        name for name, present in required_components.items() if not present
    )
    if absent:
        raise Qwen4ExpArtifactError(
            f"checkpoint weight components are missing: {absent!r}"
        )

    payload = {
        "schema": "qwen4-exp-artifact-inventory-v1",
        "files": files,
        "weight_shards": shards,
        "weight_keys": weight_keys,
        "weight_map": frozen_weight_map,
        "components": required_components,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return Qwen4ExpArtifactInventory(
        files=files,
        weight_shards=shards,
        weight_keys=weight_keys,
        weight_map=frozen_weight_map,
        has_language_trunk=has_language,
        has_embedded_mtp=has_mtp,
        has_embedded_vision=has_vision,
        has_embedded_ple=has_ple,
        digest=digest,
    )


def _normalized_names(
    name: str,
    values: Sequence[str],
    *,
    allow_duplicates: bool,
) -> tuple[str, ...]:
    if isinstance(values, str | bytes) or not isinstance(values, Sequence):
        raise Qwen4ExpArtifactError(f"{name} must be a sequence")
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value:
            raise Qwen4ExpArtifactError(f"{name} must contain non-empty strings")
        if value in {".", ".."} or "/" in value or "\\" in value:
            raise Qwen4ExpArtifactError(f"{name} must contain snapshot basenames")
        normalized.append(value)
    if not allow_duplicates and len(set(normalized)) != len(normalized):
        raise Qwen4ExpArtifactError(f"{name} must not contain duplicates")
    return tuple(sorted(set(normalized)))


__all__ = [
    "Qwen4ExpArtifactError",
    "Qwen4ExpArtifactInventory",
    "inspect_qwen4_exp_artifacts",
]
