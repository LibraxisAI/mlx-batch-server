from __future__ import annotations


def test_operator_app_boots_and_health_routes_work(operator_client):
    assert operator_client.get("/health").json() == {"status": "ok"}
    assert operator_client.get("/api/health").json() == {"status": "ok"}
    response = operator_client.get("/admin/")
    assert response.status_code == 200
    assert "MLX Batch Server Operator" in response.text


def test_operator_static_mount_serves_htmx(operator_client):
    response = operator_client.get("/admin/static/htmx.js")
    assert response.status_code == 200
    assert "htmx" in response.text.lower()
