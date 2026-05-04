from fastapi.testclient import TestClient

from mlx_batch_server.main import _build_cors_config, create_app


def test_build_cors_config_supports_wildcards():
    origins, regex = _build_cors_config(
        "https://*.fold-antares.ts.net,http://*.fold-antares.ts.net,https://dragon.fold-antares.ts.net"
    )

    assert origins == ["https://dragon.fold-antares.ts.net"]
    assert regex is not None
    assert regex.startswith("^(?:")
    assert r"fold\-antares\.ts\.net" in regex


def test_default_cors_allows_tailscale_100_space(monkeypatch):
    monkeypatch.delenv("MLX_BATCH_CORS", raising=False)
    app = create_app()

    cors = [
        middleware
        for middleware in app.user_middleware
        if middleware.cls.__name__ == "CORSMiddleware"
    ]

    assert cors
    assert "100\\." in (cors[0].kwargs.get("allow_origin_regex") or "")


def test_cors_allows_tailnet_wildcards(monkeypatch):
    monkeypatch.setenv(
        "MLX_BATCH_CORS",
        "https://*.fold-antares.ts.net,http://*.fold-antares.ts.net",
    )
    app = create_app()

    with TestClient(app) as client:
        response = client.options(
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
