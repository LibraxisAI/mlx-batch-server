from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class VideoPipeline(StrEnum):
    DISTILLED = "distilled"
    DEV = "dev"
    DEV_TWO_STAGE = "dev-two-stage"
    DEV_TWO_STAGE_HQ = "dev-two-stage-hq"


class VideoSize(StrEnum):
    LANDSCAPE = "768x512"
    PORTRAIT = "512x768"
    SQUARE = "512x512"


class VideoGenerationRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    model: Literal[
        "prince-canuma/LTX-2-distilled",
        "prince-canuma/LTX-2.3-distilled",
        "prince-canuma/LTX-2-dev",
        "prince-canuma/LTX-2.3-dev",
    ] = Field(
        default="prince-canuma/LTX-2.3-distilled",
        description="Supported LTX model repository",
    )
    image: str | None = Field(
        default=None,
        description="Optional data:image base64 source for image-to-video",
    )
    end_image: str | None = Field(
        default=None,
        description="Optional data:image base64 final-frame reference",
    )
    duration: Literal[6, 10, 15] = 6
    size: VideoSize = VideoSize.LANDSCAPE
    fps: int = Field(default=24, ge=8, le=30)
    seed: int | None = Field(default=None, ge=0)
    pipeline: VideoPipeline = VideoPipeline.DISTILLED
    steps: int | None = Field(default=None, ge=1, le=100)
    cfg_scale: float | None = Field(default=None, ge=0, le=20)
    tiling: Literal[
        "auto", "none", "default", "aggressive", "conservative", "spatial", "temporal"
    ] = "auto"

    @field_validator("image", "end_image")
    @classmethod
    def validate_image_input(cls, value: str | None) -> str | None:
        if value is not None and not value.startswith("data:image/"):
            raise ValueError("MLX video inputs must be data:image base64 URLs")
        return value

    @model_validator(mode="after")
    def validate_end_image(self):
        if self.end_image is not None and self.image is None:
            raise ValueError("end_image requires image")
        return self


class VideoArtifact(BaseModel):
    id: str
    url: str
    mime_type: str = "video/mp4"
    bytes: int
    sha256: str
    duration: Literal[6, 10, 15]
    model: str
    revised_prompt: str


class VideoGenerationResponse(BaseModel):
    created: int
    data: list[VideoArtifact]


class VideoCapabilities(BaseModel):
    available: bool
    backend: str = "mlx-video-cli"
    generation: bool = True
    image_to_video: bool = True
    end_frame: bool = True
    video_edit: bool = False
    durations: list[int] = Field(default_factory=lambda: [6, 10, 15])
    models: list[str]
    reason: str | None = None
