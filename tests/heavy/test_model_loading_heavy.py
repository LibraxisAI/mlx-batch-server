import base64
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from src.mlx_batch_server.main import app

RUN_HEAVY = os.getenv("RUN_HEAVY_TESTS") == "1"

FLUX_MODEL = os.getenv("HEAVY_FLUX_MODEL", "dhairyashil/FLUX.1-schnell-mflux-4bit")
DIA_MODEL = os.getenv("HEAVY_TTS_MODEL", "mlx-community/Dia-1.6b")
VISUAL_MODEL = os.getenv("HEAVY_VISUAL_MODEL", "mlx-community/Qwen3-VL-4B-mlx")
WHISPER_MODEL = os.getenv("HEAVY_STT_MODEL", "mlx-community/whisper-base-mlx")
VISUAL_PROCESSOR = os.getenv("HEAVY_VISUAL_PROCESSOR")
VISUAL_PROJECTION = os.getenv("HEAVY_VISUAL_PROJECTION")
USE_DIA_TTS = os.getenv("HEAVY_DIA_TTS") == "1"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not RUN_HEAVY,
        reason="Set RUN_HEAVY_TESTS=1 to run heavy model download tests.",
    ),
]


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def _cleanup_models(client: TestClient):
    yield
    client.post("/v1/models/unload", json={})


def _assert_loaded(resp):
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] in ("loaded", "already_loaded")
    return data


def test_heavy_flux_and_qwen3_vl_image_path(client: TestClient):
    _assert_loaded(
        client.post(
            "/v1/models/load",
            json={"model": FLUX_MODEL, "task": "images"},
        )
    )

    image_resp = client.post(
        "/v1/images/generations",
        json={
            "model": FLUX_MODEL,
            "prompt": "a red apple on a wooden table",
            "n": 1,
            "size": "256x256",
            "response_format": "b64_json",
            "seed": 123,
        },
    )
    assert image_resp.status_code == 200, image_resp.text
    image_data = image_resp.json()
    assert image_data.get("data")
    image_b64 = image_data["data"][0].get("b64_json")
    assert image_b64
    base64.b64decode(image_b64)

    _assert_loaded(
        client.post(
            "/v1/models/load",
            json={"model": VISUAL_MODEL, "task": "visual"},
        )
    )

    payload = {"model": VISUAL_MODEL, "images": [image_b64]}
    if VISUAL_PROCESSOR:
        payload["processor_id"] = VISUAL_PROCESSOR
    if VISUAL_PROJECTION:
        payload["projection_path"] = VISUAL_PROJECTION

    embed_resp = client.post("/v1/visual-embeddings", json=payload)
    assert embed_resp.status_code == 200, embed_resp.text
    embed_data = embed_resp.json()
    assert embed_data.get("image_embeddings")
    embeddings = embed_data["image_embeddings"][0]["embedding"]
    assert isinstance(embeddings, list)
    assert embeddings
    if embed_data.get("dim") is not None:
        assert embed_data["dim"] > 0


def test_heavy_dia_and_whisper_transcription(client: TestClient):
    _assert_loaded(
        client.post(
            "/v1/models/load",
            json={"model": DIA_MODEL, "task": "tts"},
        )
    )
    _assert_loaded(
        client.post(
            "/v1/models/load",
            json={"model": WHISPER_MODEL, "task": "stt"},
        )
    )

    if USE_DIA_TTS:
        tts_resp = client.post(
            "/v1/audio/speech",
            json={
                "model": DIA_MODEL,
                "input": "hello from mlx batch server",
                "voice": "demo",
                "response_format": "wav",
            },
        )
        assert tts_resp.status_code == 200, tts_resp.text
        audio_bytes = tts_resp.content
    else:
        audio_path = Path(__file__).resolve().parents[1] / "test_audio.wav"
        if not audio_path.exists():
            pytest.skip(f"Audio file {audio_path} does not exist")
        audio_bytes = audio_path.read_bytes()

    files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
    data = {"model": WHISPER_MODEL, "response_format": "json"}
    stt_resp = client.post("/v1/audio/transcriptions", data=data, files=files)
    assert stt_resp.status_code == 200, stt_resp.text
    stt_data = stt_resp.json()
    assert "text" in stt_data
    assert stt_data["text"].strip()
