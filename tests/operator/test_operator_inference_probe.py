from __future__ import annotations

import asyncio

from mlx_batch_server.operator.config import get_settings
from mlx_batch_server.operator.services.inference_probe import probe_inference


class _FakeResponse:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload
        self.headers = {"content-type": "application/json"}

    @property
    def is_success(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self):
        return self._payload


class _ProbeClient:
    def __init__(self, *args, **kwargs):
        return None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def get(self, url):
        if url.endswith("/health"):
            return _FakeResponse(200, {"status": "ok", "version": "test"})
        if url.endswith("/v1/models/loaded"):
            return _FakeResponse(200, {"data": [{"id": "mlx-test"}]})
        return _FakeResponse(404, {})


def test_probe_inference_handles_upstream(monkeypatch):
    from mlx_batch_server.operator.services import inference_probe

    monkeypatch.setattr(inference_probe.httpx, "AsyncClient", _ProbeClient)
    status = asyncio.run(probe_inference(get_settings()))
    assert status.healthy is True
    assert status.version == "test"
    assert status.loaded_models == ["mlx-test"]
