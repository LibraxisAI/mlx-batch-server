"""Batch Processing API Router.

Provides endpoints for batch processing statistics and control.

Vibecrafted. with AI Agents by VetCoders (c)2024-2026 The LibraxisAI Team
"""

from fastapi import APIRouter
from pydantic import BaseModel

from ..core.config import get_settings

router = APIRouter(prefix="/v1/batch", tags=["batch"])


class BatchStatsResponse(BaseModel):
    """Response model for batch statistics."""

    enabled: bool
    coordinators: dict
    settings: dict
    coverage: dict[str, str]


@router.get("/stats", response_model=BatchStatsResponse)
async def get_batch_stats() -> BatchStatsResponse:
    """Get batch processing statistics.

    Returns:
        Batch coordinator statistics including:
        - Whether batch processing is enabled
        - Per-coordinator stats (requests, batches, tokens)
        - Current batch settings
    """
    settings = get_settings()

    # Get stats from all active coordinators
    coordinator_stats = {}
    try:
        from .coordinator import _coordinators

        for key, coord in _coordinators.items():
            coordinator_stats[key] = coord.stats()
    except ImportError:
        pass

    return BatchStatsResponse(
        enabled=settings.enable_batch_inference,
        coordinators=coordinator_stats,
        settings={
            "batch_window_ms": settings.batch_window_ms,
            "max_batch_size": settings.max_batch_size,
            "batch_completion_size": settings.batch_completion_size,
            "batch_prefill_size": settings.batch_prefill_size,
            "batch_prefill_step_size": settings.batch_prefill_step_size,
        },
        coverage={
            "text": "Text requests can use the shared batch coordinator lane.",
            "tools": "Tool-capable text requests stay on the same model runtime but currently fall back to the single-request lane.",
            "multimodal": "Image and video requests use mlx-vlm single-flight and are intentionally absent from batch stats.",
        },
    )
