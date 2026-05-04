from __future__ import annotations


class _FakeStreamResponse:
    headers = {"content-type": "text/event-stream"}

    def raise_for_status(self) -> None:
        return None

    async def aiter_lines(self):
        yield 'data: {"response_id":"resp_1","model":"mlx-test"}'
        yield "data: [DONE]"


class _FakeStreamContext:
    async def __aenter__(self):
        return _FakeStreamResponse()

    async def __aexit__(self, exc_type, exc, tb):
        return None


class _FakeAsyncClient:
    def __init__(self, *args, **kwargs):
        self.requests = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    def stream(self, method, url, headers=None, json=None):
        self.requests.append((method, url, headers, json))
        return _FakeStreamContext()


def test_playground_sse_proxies_to_local_inference(
    monkeypatch,
    operator_client,
):
    monkeypatch.setenv("MLX_BATCH_INTERNAL_API_KEY", "secret")
    from mlx_batch_server.operator.routers import playground

    monkeypatch.setattr(playground.httpx, "AsyncClient", _FakeAsyncClient)
    response = operator_client.post(
        "/api/playground/responses",
        json={
            "model": "mlx-test",
            "input": [{"role": "user", "content": "hello"}],
            "stream": True,
            "session_id": "default",
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "resp_1" in response.text


def test_playground_history_endpoint(operator_client):
    response = operator_client.get("/api/playground/history?session_id=default")
    assert response.status_code == 200
    assert response.json() == {"session_id": "default", "items": []}
