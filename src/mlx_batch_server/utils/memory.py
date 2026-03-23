from __future__ import annotations

import gc
import time
from contextlib import suppress
from typing import Any

import mlx.core as mx

from .logger import logger


def _to_gb(value: Any) -> float | None:
    try:
        return round(float(value) / (1024**3), 2)
    except Exception:
        return None


def get_mlx_memory_snapshot() -> dict[str, float | None]:
    """Return current MLX/Metal memory telemetry in GB."""
    active = None
    cache = None

    with suppress(Exception):
        active = _to_gb(mx.get_active_memory())
        cache = _to_gb(mx.get_cache_memory())

    metal = getattr(mx, "metal", None)
    if metal is not None:
        with suppress(Exception):
            active = _to_gb(metal.get_active_memory())
            cache = _to_gb(metal.get_cache_memory())

    return {
        "mlx_active_memory_gb": active,
        "mlx_cache_memory_gb": cache,
    }


def force_mlx_cleanup(label: str | None = None, *, passes: int = 2) -> dict[str, dict]:
    """Best-effort memory cleanup for MLX + Metal allocator caches."""
    before = get_mlx_memory_snapshot()
    metal = getattr(mx, "metal", None)

    for idx in range(max(1, passes)):
        gc.collect()
        with suppress(Exception):
            mx.clear_cache()

        if metal is not None and hasattr(metal, "clear_cache"):
            with suppress(Exception):
                metal.clear_cache()

        gc.collect()
        if idx < passes - 1:
            time.sleep(0.05)

    after = get_mlx_memory_snapshot()

    if label:
        logger.info(
            "MLX cleanup (%s): active %sGB -> %sGB, cache %sGB -> %sGB",
            label,
            before.get("mlx_active_memory_gb"),
            after.get("mlx_active_memory_gb"),
            before.get("mlx_cache_memory_gb"),
            after.get("mlx_cache_memory_gb"),
        )

    return {"before": before, "after": after}
