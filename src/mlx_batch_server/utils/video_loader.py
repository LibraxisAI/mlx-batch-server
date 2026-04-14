"""Video loading utilities for vision-language model inference.

Bridges between the Responses API ``input_video`` parts and the mlx-vlm
video generation pipeline.  The heavy lifting (frame extraction, resizing,
processor invocation) is delegated to ``mlx_vlm.video_generate``.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Supported video extensions (matches mlx-vlm)
_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov"}

DEFAULT_FPS = 4.0
DEFAULT_MAX_PIXELS = 500_000  # ~500k pixels per frame for decent resolution


def is_video_model(model: Any) -> bool:
    """Check whether *model* has video token support (e.g. Qwen3-VL)."""
    cfg = getattr(model, "config", None)
    if cfg is None:
        return False
    return hasattr(cfg, "video_token_id") or hasattr(cfg, "video_token_index")


def prepare_video_inputs(
    video_sources: list[str],
    processor: Any,
    *,
    fps: float = DEFAULT_FPS,
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> dict[str, Any]:
    """Load video file(s) and run them through the VLM processor.

    Parameters
    ----------
    video_sources:
        Paths or URLs to video files.
    processor:
        A HuggingFace-compatible processor (e.g. ``AutoProcessor`` for Qwen3-VL).
    fps:
        Frames-per-second to sample from the video.
    max_pixels:
        Maximum pixels per frame for resizing.

    Returns
    -------
    dict with keys that can be unpacked into ``generate()`` kwargs:
        - ``pixel_values``: mx.array of processed video frames
        - ``video_grid_thw``: mx.array of (Time, Height, Width) grid
        - ``image_grid_thw``: mx.array if present
        - ``input_ids``: mx.array of tokenised prompt (optional)
        - ``mask``: mx.array attention mask (optional)
    """
    try:
        from mlx_vlm.video_generate import process_vision_info
    except ImportError as exc:
        raise RuntimeError(
            "mlx-vlm with video support is required for video responses"
        ) from exc

    # Build messages in the format mlx-vlm expects
    video_content: list[dict[str, Any]] = []
    for src in video_sources:
        video_content.append(
            {
                "type": "video",
                "video": src,
                "fps": fps,
                "max_pixels": max_pixels,
            }
        )

    messages = [{"role": "user", "content": video_content}]

    # process_vision_info extracts frames and returns numpy arrays
    image_inputs, video_inputs = process_vision_info(messages)

    return {
        "image_inputs": image_inputs,
        "video_inputs": video_inputs,
    }


def build_video_prompt_and_inputs(
    video_sources: list[str],
    text_prompt: str,
    processor: Any,
    model_config: Any,
    *,
    fps: float = DEFAULT_FPS,
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> dict[str, Any]:
    """Full pipeline: build chat template, process video, return ready-to-generate inputs.

    Returns
    -------
    dict with:
        - ``prompt``: str — formatted prompt with video tokens
        - ``pixel_values``: mx.array
        - ``video_grid_thw``: mx.array (if present)
        - ``image_grid_thw``: mx.array (if present)
        - ``mask``: mx.array
        - ``input_ids``: mx.array
    """
    try:
        import mlx.core as mx
        from mlx_vlm.video_generate import process_vision_info
    except ImportError as exc:
        raise RuntimeError(
            "mlx-vlm with video support is required for video responses"
        ) from exc

    # Build messages with both video and text content
    content: list[dict[str, Any]] = []
    for src in video_sources:
        content.append(
            {
                "type": "video",
                "video": src,
                "fps": fps,
                "max_pixels": max_pixels,
            }
        )
    content.append({"type": "text", "text": text_prompt})

    messages = [{"role": "user", "content": content}]

    # Apply chat template to get prompt with video tokens
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Extract video frames
    image_inputs, video_inputs = process_vision_info(messages)

    # Run through processor to get pixel values and grid info.
    # IMPORTANT: do_sample_frames=False prevents the processor from
    # re-sampling frames down to 4 (which collapses temporal to 2).
    # We already sampled frames in process_vision_info via smart_nframes.
    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        do_sample_frames=False,
    )

    result: dict[str, Any] = {"prompt": prompt}

    # Extract pixel values (video or image)
    pixel_values = inputs.get("pixel_values_videos", inputs.get("pixel_values", None))
    if pixel_values is not None:
        result["pixel_values"] = mx.array(pixel_values)

    if inputs.get("video_grid_thw") is not None:
        result["video_grid_thw"] = mx.array(inputs["video_grid_thw"])
    if inputs.get("image_grid_thw") is not None:
        result["image_grid_thw"] = mx.array(inputs["image_grid_thw"])

    result["input_ids"] = mx.array(inputs["input_ids"])
    result["mask"] = mx.array(inputs["attention_mask"])

    return result
