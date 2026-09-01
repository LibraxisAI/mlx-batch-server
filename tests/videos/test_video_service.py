from __future__ import annotations

import base64
from pathlib import Path

import pytest
from PIL import Image

from mlx_batch_server.videos.schema import VideoGenerationRequest
from mlx_batch_server.videos.video_service import MlxVideoAdapter


def data_image(tmp_path: Path) -> str:
    path = tmp_path / "source.png"
    Image.new("RGB", (8, 8), "red").save(path)
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def test_ltx_command_maps_duration_to_four_n_plus_one_frames(tmp_path):
    python = tmp_path / "python"
    python.touch()
    adapter = MlxVideoAdapter(
        root=tmp_path,
        python=python,
        artifact_dir=tmp_path / "out",
    )
    request = VideoGenerationRequest(prompt="move", duration=10, fps=24)

    command = adapter._ltx_command(request, tmp_path / "out.mp4", None, None)

    assert command[command.index("--num-frames") + 1] == "241"
    assert command[command.index("--model-repo") + 1] == request.model
    assert command[command.index("--output-path") + 1] == str(tmp_path / "out.mp4")


def test_reference_images_are_validated_and_materialized(tmp_path):
    source = data_image(tmp_path)
    materialized = MlxVideoAdapter._materialize_image(source, tmp_path, "input")
    assert materialized is not None
    assert materialized.is_file()
    with Image.open(materialized) as image:
        assert image.size == (8, 8)


def test_request_rejects_paths_and_end_image_without_first_image(tmp_path):
    with pytest.raises(ValueError, match="data:image"):
        VideoGenerationRequest(prompt="move", image=str(tmp_path / "source.png"))
    with pytest.raises(ValueError, match="end_image requires image"):
        VideoGenerationRequest(prompt="move", end_image=data_image(tmp_path))


def test_artifact_id_rejects_path_traversal(tmp_path):
    python = tmp_path / "python"
    python.touch()
    adapter = MlxVideoAdapter(root=tmp_path, python=python, artifact_dir=tmp_path)
    with pytest.raises(ValueError, match="Invalid video artifact id"):
        adapter.artifact_path("../../secret")


def test_offline_capabilities_require_a_cached_supported_model(tmp_path, monkeypatch):
    python = tmp_path / "python"
    python.touch()
    cache = tmp_path / "cache"
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    adapter = MlxVideoAdapter(root=tmp_path, python=python, artifact_dir=tmp_path)

    missing = adapter.capabilities()
    assert missing.available is False
    assert missing.cached_models == []
    assert "no supported LTX model" in (missing.reason or "")

    snapshot = cache / "models--prince-canuma--LTX-2.3-distilled" / "snapshots" / "sha"
    snapshot.mkdir(parents=True)
    ready = adapter.capabilities()
    assert ready.available is True
    assert ready.cached_models == ["prince-canuma/LTX-2.3-distilled"]
