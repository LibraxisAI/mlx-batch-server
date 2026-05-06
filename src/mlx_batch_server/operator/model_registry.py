"""Model cache inspection for local Hugging Face/MLX assets."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class CachedModel:
    model_id: str
    config_path: Path
    snapshot_dir: Path
    size_bytes: int
    config: dict[str, Any]


def _hub_root() -> Path:
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _decode_repo_id(dirname: str) -> str:
    stem = dirname.removeprefix("models--")
    parts = [segment for segment in stem.split("--") if segment]
    return "/".join(parts)


def scan_local_models() -> list[CachedModel]:
    root = _hub_root()
    if not root.exists():
        return []

    models: list[CachedModel] = []
    for model_dir in sorted(root.glob("models--*")):
        repo_id = _decode_repo_id(model_dir.name)
        for config_path in model_dir.glob("snapshots/*/config.json"):
            snapshot_dir = config_path.parent
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                config = None
            if config is None:
                continue
            size_bytes = sum(
                file.stat().st_size
                for file in snapshot_dir.rglob("*")
                if file.is_file()
            )
            models.append(
                CachedModel(
                    model_id=repo_id,
                    config_path=config_path,
                    snapshot_dir=snapshot_dir,
                    size_bytes=size_bytes,
                    config=config,
                )
            )
            break
    return models


def registry_rows(
    assignments: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    assignments = assignments or {}
    for model in scan_local_models():
        quantization = model.config.get("quantization")
        if isinstance(quantization, dict):
            quantization_text = ", ".join(
                f"{key}={value}" for key, value in sorted(quantization.items())
            )
        else:
            quantization_text = str(quantization) if quantization is not None else ""

        rows.append(
            {
                "model_id": model.model_id,
                "model_type": model.config.get("model_type"),
                "image_token_id": model.config.get("image_token_id"),
                "quantization": quantization_text or None,
                "size_bytes": model.size_bytes,
                "assigned_to": assignments.get(model.model_id, []),
                "config_path": str(model.config_path),
            }
        )
    return rows
