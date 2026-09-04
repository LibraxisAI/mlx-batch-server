# SPDX-License-Identifier: Apache-2.0
"""Fail-closed filesystem plan for loading one Qwen4Exp checkpoint.

This module performs metadata I/O only. It deliberately does not import MLX,
open safetensor shards, instantiate tokenizers, or select a runtime backend.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import Qwen4ExpArtifactInventory, inspect_qwen4_exp_artifacts
from .config import Qwen4ExpCheckpointConfig, parse_qwen4_exp_config
from .topology import Qwen4ExpTopology, build_qwen4_exp_topology

_DEFAULT_MAX_METADATA_BYTES = 64 * 1024 * 1024


class Qwen4ExpLoadPlanError(ValueError):
    """Checkpoint metadata cannot produce an immutable tensor load plan."""


@dataclass(frozen=True, slots=True)
class Qwen4ExpModelLoadPlan:
    model_id: str
    revision: str
    model_dir: str
    config: Qwen4ExpCheckpointConfig
    topology: Qwen4ExpTopology
    artifacts: Qwen4ExpArtifactInventory
    preprocessor_config_json: str
    config_sha256: str
    index_sha256: str
    preprocessor_sha256: str
    tokenizer_fingerprint: str
    plan_sha256: str

    def __post_init__(self) -> None:
        if not self.model_id or not self.revision or not self.model_dir:
            raise Qwen4ExpLoadPlanError("model identity must be complete")
        for name, digest in (
            ("config", self.config_sha256),
            ("index", self.index_sha256),
            ("preprocessor", self.preprocessor_sha256),
            ("tokenizer", self.tokenizer_fingerprint),
            ("plan", self.plan_sha256),
        ):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise Qwen4ExpLoadPlanError(f"{name} digest must be SHA-256")
        if not self.preprocessor_config_json:
            raise Qwen4ExpLoadPlanError("preprocessor metadata must not be empty")
        try:
            preprocessor = json.loads(self.preprocessor_config_json)
        except json.JSONDecodeError as error:
            raise Qwen4ExpLoadPlanError(
                "preprocessor metadata must be canonical JSON"
            ) from error
        if not isinstance(preprocessor, Mapping) or self.preprocessor_config_json != (
            json.dumps(preprocessor, sort_keys=True, separators=(",", ":"))
        ):
            raise Qwen4ExpLoadPlanError(
                "preprocessor metadata must be a canonical JSON object"
            )


def load_qwen4_exp_plan(
    *,
    model_dir: str | Path,
    model_id: str,
    revision: str,
    max_metadata_bytes: int = _DEFAULT_MAX_METADATA_BYTES,
) -> Qwen4ExpModelLoadPlan:
    """Read bounded metadata and freeze the exact future tensor load input."""

    if not model_id or not revision:
        raise Qwen4ExpLoadPlanError("model_id and revision must not be empty")
    if (
        isinstance(max_metadata_bytes, bool)
        or not isinstance(max_metadata_bytes, int)
        or max_metadata_bytes < 1
    ):
        raise Qwen4ExpLoadPlanError("max_metadata_bytes must be positive")

    root = Path(model_dir).expanduser()
    if not root.is_absolute():
        raise Qwen4ExpLoadPlanError("model_dir must be an absolute path")
    if not root.is_dir():
        raise Qwen4ExpLoadPlanError("model_dir must be an existing directory")
    root = root.resolve(strict=True)

    config_path = root / "config.json"
    index_path = root / "model.safetensors.index.json"
    preprocessor_path = root / "preprocessor_config.json"
    tokenizer_path = root / "tokenizer.json"
    tokenizer_config_path = root / "tokenizer_config.json"
    chat_template_path = root / "chat_template.jinja"
    config_bytes = _read_bounded(config_path, max_metadata_bytes)
    index_bytes = _read_bounded(index_path, max_metadata_bytes)
    preprocessor_bytes = _read_bounded(preprocessor_path, max_metadata_bytes)
    tokenizer_bytes = _read_bounded(tokenizer_path, max_metadata_bytes)
    tokenizer_config_bytes = _read_bounded(
        tokenizer_config_path,
        max_metadata_bytes,
    )
    chat_template_bytes = _read_bounded(chat_template_path, max_metadata_bytes)
    config_raw = _json_mapping("config.json", config_bytes)
    index_raw = _json_mapping("model.safetensors.index.json", index_bytes)
    preprocessor_raw = _json_mapping(
        "preprocessor_config.json",
        preprocessor_bytes,
    )
    weight_map = index_raw.get("weight_map")
    if not isinstance(weight_map, Mapping):
        raise Qwen4ExpLoadPlanError("model index requires a weight_map")

    config = parse_qwen4_exp_config(config_raw)
    topology = build_qwen4_exp_topology(config)
    files = tuple(
        sorted(
            item.name
            for item in root.iterdir()
            if item.is_file() and not item.name.startswith(".")
        )
    )
    artifacts = inspect_qwen4_exp_artifacts(
        config=config,
        file_names=files,
        weight_map=_string_mapping("weight_map", weight_map),
    )
    config_digest = hashlib.sha256(config_bytes).hexdigest()
    index_digest = hashlib.sha256(index_bytes).hexdigest()
    preprocessor_digest = hashlib.sha256(preprocessor_bytes).hexdigest()
    tokenizer_fingerprint = _component_fingerprint(
        (
            (tokenizer_path.name, tokenizer_bytes),
            (tokenizer_config_path.name, tokenizer_config_bytes),
            (chat_template_path.name, chat_template_bytes),
        )
    )
    preprocessor_json = json.dumps(
        preprocessor_raw,
        sort_keys=True,
        separators=(",", ":"),
    )
    plan_payload = {
        "schema": "qwen4-exp-model-load-plan-v1",
        "model_id": model_id,
        "revision": revision,
        "model_dir": str(root),
        "config_sha256": config_digest,
        "index_sha256": index_digest,
        "preprocessor_sha256": preprocessor_digest,
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "artifact_inventory_sha256": artifacts.digest,
        "tensor_batch_mode": topology.tensor_batch_mode.value,
        "max_qsa_batch_rows": topology.max_qsa_batch_rows,
        "max_verified_mtp_rows": topology.max_verified_mtp_rows,
    }
    plan_digest = hashlib.sha256(
        json.dumps(plan_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return Qwen4ExpModelLoadPlan(
        model_id=model_id,
        revision=revision,
        model_dir=str(root),
        config=config,
        topology=topology,
        artifacts=artifacts,
        preprocessor_config_json=preprocessor_json,
        config_sha256=config_digest,
        index_sha256=index_digest,
        preprocessor_sha256=preprocessor_digest,
        tokenizer_fingerprint=tokenizer_fingerprint,
        plan_sha256=plan_digest,
    )


def _read_bounded(path: Path, limit: int) -> bytes:
    try:
        size = path.stat().st_size
    except OSError as error:
        raise Qwen4ExpLoadPlanError(
            f"cannot stat checkpoint metadata: {path.name}"
        ) from error
    if size < 1 or size > limit:
        raise Qwen4ExpLoadPlanError(f"checkpoint metadata size is invalid: {path.name}")
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise Qwen4ExpLoadPlanError(
            f"cannot read checkpoint metadata: {path.name}"
        ) from error
    if len(payload) != size:
        raise Qwen4ExpLoadPlanError(
            f"checkpoint metadata changed while reading: {path.name}"
        )
    return payload


def _component_fingerprint(parts: Sequence[tuple[str, bytes]]) -> str:
    digest = hashlib.sha256()
    for name, payload in parts:
        name_bytes = name.encode("utf-8")
        digest.update(len(name_bytes).to_bytes(4, "big"))
        digest.update(name_bytes)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _json_mapping(name: str, payload: bytes) -> Mapping[str, Any]:
    try:
        value = json.loads(payload, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Qwen4ExpLoadPlanError(f"{name} must be valid UTF-8 JSON") from error
    if not isinstance(value, Mapping):
        raise Qwen4ExpLoadPlanError(f"{name} must contain a JSON object")
    return value


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise Qwen4ExpLoadPlanError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _string_mapping(name: str, value: Mapping[Any, Any]) -> Mapping[str, str]:
    if any(
        not isinstance(key, str) or not key or not isinstance(item, str) or not item
        for key, item in value.items()
    ):
        raise Qwen4ExpLoadPlanError(f"{name} must map strings to strings")
    return dict(value)


__all__ = [
    "Qwen4ExpLoadPlanError",
    "Qwen4ExpModelLoadPlan",
    "load_qwen4_exp_plan",
]
