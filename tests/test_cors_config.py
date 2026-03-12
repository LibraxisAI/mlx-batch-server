import pytest
from httpx import ASGITransport, AsyncClient

from mlx_batch_server.main import _build_cors_config, create_app


def test_build_cors_config_supports_wildcards():
    origins, regex = _build_cors_config(
        "https://*.example.ts.net,http://*.example.ts.net,https://host-a.example.ts.net"
    )

    assert origins == ["https://host-a.example.ts.net"]
    assert regex is not None
    assert regex.startswith("^(?:")
    assert r"example\.ts\.net" in regex


@pytest.mark.asyncio
async def test_cors_allows_tailnet_wildcards(monkeypatch):
    monkeypatch.setenv(
        "MLX_BATCH_CORS",
        "https://*.example.ts.net,http://*.example.ts.net",
    )
    app = create_app()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.options(
            "/v1/models",
            headers={
                "Origin": "https://host-b.example.ts.net",
                "Access-Control-Request-Method": "POST",
            },
        )

    assert response.status_code == 200
    assert (
        response.headers.get("access-control-allow-origin")
        == "https://host-b.example.ts.net"
    )
