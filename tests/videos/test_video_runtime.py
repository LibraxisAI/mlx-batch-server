from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from mlx_batch_server.videos.schema import VideoGenerationRequest
from mlx_batch_server.videos.video_runtime import VideoRuntime


@pytest.mark.asyncio
async def test_runtime_returns_structured_artifact_and_tracks_activity():
    runtime: VideoRuntime

    def worker(_operation, payload):
        assert runtime.snapshot()["active_operations"] == 1
        return {
            "artifact": {
                "id": "video_1_a",
                "url": "/v1/videos/artifacts/video_1_a",
                "mime_type": "video/mp4",
                "bytes": 12,
                "sha256": "0" * 64,
                "duration": payload["duration"],
                "model": payload["model"],
                "revised_prompt": payload["prompt"],
            }
        }

    executor = ThreadPoolExecutor(max_workers=1)
    runtime = VideoRuntime(executor=executor, worker_operation=worker)
    artifact = await runtime.generate(VideoGenerationRequest(prompt="move"))
    assert artifact.bytes == 12
    assert runtime.snapshot()["active_operations"] == 0
    executor.shutdown()
