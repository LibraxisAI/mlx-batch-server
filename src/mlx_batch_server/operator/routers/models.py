"""Model cache and registry endpoints for the operator backend."""

from __future__ import annotations

from fastapi import APIRouter

from mlx_batch_server.operator.model_registry import registry_rows, scan_local_models

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("/cache")
async def models_cache() -> list[dict[str, int | str]]:
    return [
        {
            "model_id": item.model_id,
            "snapshot_dir": str(item.snapshot_dir),
            "config_path": str(item.config_path),
            "size_bytes": item.size_bytes,
        }
        for item in scan_local_models()
    ]


@router.get("/registry")
async def models_registry() -> list[dict]:
    return registry_rows()
