"""Subprocess adapter for the sibling mlx-video checkout.

The adapter intentionally isolates MLX video imports from the FastAPI process.
Each job gets a fresh child process today; the request/response contract stays
stable when this is replaced by a persistent native worker later.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import os
import subprocess  # nosec B404
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from PIL import Image

from ..core.config import get_settings
from .schema import VideoArtifact, VideoCapabilities, VideoGenerationRequest

_DATA_IMAGE_PREFIX = "data:image/"
_MAX_IMAGE_BYTES = 20 * 1024 * 1024
_SUPPORTED_MODELS = [
    "prince-canuma/LTX-2-distilled",
    "prince-canuma/LTX-2.3-distilled",
    "prince-canuma/LTX-2-dev",
    "prince-canuma/LTX-2.3-dev",
]


def default_mlx_video_root() -> Path:
    return Path(__file__).resolve().parents[3].parent / "mlx-video"


class MlxVideoAdapter:
    def __init__(
        self,
        *,
        root: Path | None = None,
        python: Path | None = None,
        artifact_dir: Path | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        settings = get_settings()
        configured_root = settings.mlx_video_root
        self.root = (
            root or Path(configured_root or default_mlx_video_root())
        ).resolve()
        configured_python = settings.mlx_video_python
        selected_python = python or (
            Path(configured_python)
            if configured_python
            else self.root / ".venv/bin/python"
        )
        self.python = selected_python.resolve()
        selected_artifact_dir = artifact_dir or (
            Path(settings.mlx_video_artifact_dir)
            if settings.mlx_video_artifact_dir
            else Path(tempfile.gettempdir()) / "mlx_batch_server" / "videos"
        )
        self.artifact_dir = selected_artifact_dir.resolve()
        self.timeout_seconds = float(
            timeout_seconds or settings.mlx_video_timeout_seconds
        )

    def capabilities(self) -> VideoCapabilities:
        reason = None
        cached_models = [
            model for model in _SUPPORTED_MODELS if self._model_cached(model)
        ]
        if not self.root.is_dir():
            reason = f"mlx-video checkout not found at {self.root}"
        elif not self.python.is_file():
            reason = f"mlx-video Python not found at {self.python}"
        elif self._offline() and not cached_models:
            reason = "HF_HUB_OFFLINE=1 and no supported LTX model is cached"
        return VideoCapabilities(
            available=reason is None,
            models=_SUPPORTED_MODELS,
            cached_models=cached_models,
            reason=reason,
        )

    def generate(self, request: VideoGenerationRequest) -> VideoArtifact:
        capabilities = self.capabilities()
        if not capabilities.available:
            raise RuntimeError(capabilities.reason)
        if self._offline() and not self._model_cached(request.model):
            raise RuntimeError(
                f"HF_HUB_OFFLINE=1 and requested model {request.model} is not cached"
            )
        self.artifact_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        artifact_id = f"video_{time.time_ns()}_{uuid.uuid4().hex[:8]}"
        output_path = self.artifact_dir / f"{artifact_id}.mp4"

        with tempfile.TemporaryDirectory(prefix="mlx-video-input-") as raw_dir:
            input_dir = Path(raw_dir)
            image = self._materialize_image(request.image, input_dir, "first")
            end_image = self._materialize_image(request.end_image, input_dir, "last")
            command = self._ltx_command(request, output_path, image, end_image)
            # shell=False; executable/module are server-owned and every request
            # value is one argv element validated by VideoGenerationRequest.
            completed = subprocess.run(  # nosec B603  # nosemgrep: python.django.security.injection.command.subprocess-injection.subprocess-injection
                command,
                cwd=self.root,
                env=self._environment(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        if completed.returncode != 0:
            output_path.unlink(missing_ok=True)
            tail = completed.stdout[-4000:].strip()
            raise RuntimeError(
                f"mlx-video exited with status {completed.returncode}: {tail}"
            )
        if not output_path.is_file() or output_path.stat().st_size == 0:
            output_path.unlink(missing_ok=True)
            raise RuntimeError("mlx-video completed without a non-empty MP4 artifact")
        payload = output_path.read_bytes()
        return VideoArtifact(
            id=artifact_id,
            url=f"/v1/videos/artifacts/{artifact_id}",
            bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
            duration=request.duration,
            model=request.model,
            revised_prompt=request.prompt,
        )

    def artifact_path(self, artifact_id: str) -> Path:
        if not artifact_id.startswith("video_") or any(
            char
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
            for char in artifact_id
        ):
            raise ValueError("Invalid video artifact id")
        return self.artifact_dir / f"{artifact_id}.mp4"

    def _ltx_command(
        self,
        request: VideoGenerationRequest,
        output_path: Path,
        image: Path | None,
        end_image: Path | None,
    ) -> list[str]:
        width, height = request.size.value.split("x", 1)
        frames = request.duration * request.fps + 1
        command = [
            str(self.python),
            "-m",
            "mlx_video.models.ltx_2.generate",
            "--prompt",
            request.prompt,
            "--pipeline",
            request.pipeline.value,
            "--model-repo",
            request.model,
            "--width",
            width,
            "--height",
            height,
            "--fps",
            str(request.fps),
            "--num-frames",
            str(frames),
            "--tiling",
            request.tiling,
            "--output-path",
            str(output_path),
        ]
        if image is not None:
            command.extend(["--image", str(image)])
        if end_image is not None:
            command.extend(["--end-image", str(end_image)])
        if request.seed is not None:
            command.extend(["--seed", str(request.seed)])
        if request.steps is not None:
            command.extend(["--steps", str(request.steps)])
        if request.cfg_scale is not None:
            command.extend(["--cfg-scale", str(request.cfg_scale)])
        return command

    @staticmethod
    def _materialize_image(
        source: str | None,
        directory: Path,
        stem: str,
    ) -> Path | None:
        if source is None:
            return None
        if not source.startswith(_DATA_IMAGE_PREFIX) or ";base64," not in source:
            raise ValueError("MLX video inputs must be base64 data:image URLs")
        header, encoded = source.split(",", 1)
        mime = header.removeprefix("data:").split(";", 1)[0]
        try:
            payload = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ValueError("Invalid base64 video reference image") from error
        if len(payload) > _MAX_IMAGE_BYTES:
            raise ValueError("Video reference image exceeds 20 MiB")
        suffix = ".png" if mime == "image/png" else ".jpg"
        path = directory / f"{stem}{suffix}"
        path.write_bytes(payload)
        try:
            with Image.open(path) as bitmap:
                bitmap.verify()
        except Exception as error:
            path.unlink(missing_ok=True)
            raise ValueError("Video reference is not a valid bitmap") from error
        return path

    @staticmethod
    def _environment() -> dict[str, str]:
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        return env

    @staticmethod
    def _offline() -> bool:
        return os.environ.get("HF_HUB_OFFLINE", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    @staticmethod
    def _hub_cache() -> Path:
        if path := os.environ.get("HF_HUB_CACHE"):
            return Path(path).expanduser()
        if path := os.environ.get("HF_HOME"):
            return Path(path).expanduser() / "hub"
        if path := os.environ.get("XDG_CACHE_HOME"):
            return Path(path).expanduser() / "huggingface" / "hub"
        return Path.home() / ".cache" / "huggingface" / "hub"

    @classmethod
    def _model_cached(cls, model: str) -> bool:
        snapshots = (
            cls._hub_cache() / f"models--{model.replace('/', '--')}" / "snapshots"
        )
        try:
            return any(entry.is_dir() for entry in snapshots.iterdir())
        except OSError:
            return False


_video_adapter: MlxVideoAdapter | None = None


def get_video_adapter() -> MlxVideoAdapter:
    global _video_adapter
    if _video_adapter is None:
        _video_adapter = MlxVideoAdapter()
    return _video_adapter


def run_video_worker_operation(
    operation: str, payload: dict[str, Any]
) -> dict[str, Any]:
    adapter = get_video_adapter()
    if operation == "generate":
        artifact = adapter.generate(VideoGenerationRequest.model_validate(payload))
        return {"artifact": artifact.model_dump(), "worker_pid": os.getpid()}
    if operation == "capabilities":
        return {
            "capabilities": adapter.capabilities().model_dump(),
            "worker_pid": os.getpid(),
        }
    raise ValueError(f"Unknown video operation '{operation}'")
