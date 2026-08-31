from fastapi.testclient import TestClient

from mlx_batch_server.main import create_app


def _get_json(path: str):
    app = create_app()
    with TestClient(app) as client:
        response = client.get(path)
    assert response.status_code == 200
    return response.json()


def test_admin_panel_is_registered():
    app = create_app()
    paths = set(app.openapi()["paths"])
    assert "/admin" in paths
    assert "/api/admin/summary" in paths


def test_admin_summary_exposes_runtime_state():
    payload = _get_json("/api/admin/summary")

    assert payload["pid"] > 0
    assert "health" in payload
    assert "readiness" in payload
