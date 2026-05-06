from __future__ import annotations


def test_admin_routes_return_html(operator_client):
    for path in [
        "/admin/",
        "/admin/sessions",
        "/admin/logs",
        "/admin/lifecycle",
        "/admin/playground",
    ]:
        response = operator_client.get(path)
        assert response.status_code == 200, path
        assert "text/html" in response.headers["content-type"]


def test_fleet_partial_renders_inference_status(monkeypatch, operator_client):
    from mlx_batch_server.operator.routers import admin
    from mlx_batch_server.operator.services.inference_probe import InferenceStatus

    async def fake_probe(settings=None):
        return InferenceStatus(
            healthy=True,
            base_url="http://127.0.0.1:10240",
            loaded_models=["mlx-test"],
        )

    monkeypatch.setattr(admin, "probe_inference", fake_probe)
    response = operator_client.get("/admin/_partials/fleet")
    assert response.status_code == 200
    assert "mlx-test" in response.text
