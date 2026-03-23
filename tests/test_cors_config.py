import pytest
from httpx import ASGITransport, AsyncClient

from mlx_batch_server.main import _build_cors_config, create_app


def test_build_cors_config_supports_wildcards():
    origins, regex = _build_cors_config(
        "https://*.fold-antares.ts.net,http://*.fold-antares.ts.net,https://dragon.fold-antares.ts.net"
    )

    assert origins == ["https://dragon.fold-antares.ts.net"]
    assert regex is not None
    assert regex.startswith("^(?:")
    assert r"fold\-antares\.ts\.net" in regex


@pytest.mark.asyncio
async def test_cors_allows_tailnet_wildcards(monkeypatch):
    monkeypatch.setenv(
        "MLX_BATCH_CORS",
        "https://*.fold-antares.ts.net,http://*.fold-antares.ts.net",
    )
    app = create_app()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.options(
            "/v1/models",
            headers={
                "Origin": "https://mgbook16.fold-antares.ts.net",
                "Access-Control-Request-Method": "POST",
            },
        )

    assert response.status_code == 200
    assert (
        response.headers.get("access-control-allow-origin")
        == "https://mgbook16.fold-antares.ts.net"
    )
