from types import SimpleNamespace

from mlx_batch_server.responses import adapter as adapter_module
from mlx_batch_server.responses.adapter import ResponsesAdapter
from mlx_batch_server.utils import video_loader
from mlx_batch_server.utils.video_loader import ResolvedVlmMedia


def test_resolve_vlm_media_preserves_native_video(monkeypatch):
    captured = {}

    def fake_resolve(processor, videos, **kwargs):
        captured.update(processor=processor, videos=videos, **kwargs)
        return SimpleNamespace(
            images=kwargs["images"],
            videos=videos,
            used_fallback=False,
            sampled_count=0,
            selected_count=0,
        )

    monkeypatch.setattr(video_loader, "resolve_video_inputs", fake_resolve)
    processor = object()
    resolved = video_loader.resolve_vlm_media(
        processor,
        images=["cover.png"],
        videos=["clip.mov"],
    )

    assert resolved.images == ["cover.png"]
    assert resolved.videos == ["clip.mov"]
    assert captured == {
        "processor": processor,
        "videos": ["clip.mov"],
        "images": ["cover.png"],
        "fps": 2.0,
        "max_frames": 16,
    }


def test_resolve_vlm_media_exposes_frame_fallback(monkeypatch):
    monkeypatch.setattr(
        video_loader,
        "resolve_video_inputs",
        lambda *_args, **_kwargs: SimpleNamespace(
            images=["frame-1", "frame-2"],
            videos=[],
            used_fallback=True,
            sampled_count=40,
            selected_count=2,
        ),
    )

    resolved = video_loader.resolve_vlm_media(object(), videos=["clip.mp4"])

    assert resolved == ResolvedVlmMedia(
        images=["frame-1", "frame-2"],
        videos=[],
        used_fallback=True,
        sampled_count=40,
        selected_count=2,
    )


def test_stream_request_routes_video_around_image_batcher(monkeypatch):
    adapter = ResponsesAdapter()
    processor = object()
    chat_template = object()
    model = SimpleNamespace(config=SimpleNamespace(model_type="qwen3_vl"))
    body = {"input": []}

    monkeypatch.setattr(adapter, "_extract_image_inputs", lambda _body: [])
    monkeypatch.setattr(adapter, "_extract_video_inputs", lambda _body: ["clip.webm"])
    monkeypatch.setattr(adapter, "_multimodal_validation_error", lambda _body: None)
    monkeypatch.setattr(adapter, "_require_vlm_chat_template", lambda: chat_template)
    monkeypatch.setattr(
        adapter,
        "_get_vlm_backend",
        lambda *_args, **_kwargs: (model, processor),
    )
    monkeypatch.setattr(adapter, "_vlm_generation_kwargs", lambda _body: {})
    monkeypatch.setattr(
        adapter,
        "_build_vlm_prompt",
        lambda *_args, **_kwargs: "<video prompt>",
    )
    monkeypatch.setattr(
        adapter_module,
        "resolve_vlm_media",
        lambda *_args, **_kwargs: ResolvedVlmMedia(images=[], videos=["clip.webm"]),
    )

    result = adapter._prepare_vlm_stream_request("demo-vlm", body)

    assert result == (
        model,
        processor,
        "<video prompt>",
        [],
        {"video": ["clip.webm"]},
    )
