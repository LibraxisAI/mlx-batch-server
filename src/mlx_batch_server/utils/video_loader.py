"""Video input compatibility for the current mlx-vlm generation API."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from mlx_vlm.generate.video import resolve_video_inputs

logger = logging.getLogger(__name__)

_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

DEFAULT_FPS = 2.0
DEFAULT_MAX_FRAMES = 16


@dataclass(frozen=True)
class ResolvedVlmMedia:
    """Media ready for ``mlx_vlm.generate``."""

    images: list[Any]
    videos: list[Any]
    used_fallback: bool = False
    sampled_count: int = 0
    selected_count: int = 0


def is_video_model(model: Any) -> bool:
    """Check whether *model* exposes a native video token."""
    config = getattr(model, "config", None)
    if config is None:
        return False
    return hasattr(config, "video_token_id") or hasattr(config, "video_token_index")


def resolve_vlm_media(
    processor: Any,
    *,
    images: list[Any] | None = None,
    videos: list[Any] | None = None,
    fps: float = DEFAULT_FPS,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> ResolvedVlmMedia:
    """Keep native video inputs or convert them to a bounded image sequence."""
    image_list = list(images or [])
    video_list = list(videos or [])
    if not video_list:
        return ResolvedVlmMedia(images=image_list, videos=[])

    resolution = resolve_video_inputs(
        processor,
        video_list,
        images=image_list,
        fps=fps,
        max_frames=max_frames,
    )
    return ResolvedVlmMedia(
        images=list(resolution.images or []),
        videos=list(resolution.videos or []),
        used_fallback=bool(getattr(resolution, "used_fallback", False)),
        sampled_count=int(getattr(resolution, "sampled_count", 0) or 0),
        selected_count=int(getattr(resolution, "selected_count", 0) or 0),
    )
