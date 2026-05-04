from __future__ import annotations


def test_tail_reads_configured_log_file(operator_client):
    response = operator_client.get("/api/logs/tail?lines=1")
    assert response.status_code == 200
    assert response.json()["lines"] == ["line two"]


def test_missing_log_returns_404(operator_client):
    response = operator_client.get("/api/logs/tail?service=missing")
    assert response.status_code == 404


def test_admin_logs_page_renders(operator_client):
    response = operator_client.get("/admin/logs")
    assert response.status_code == 200
    assert "Live logs" in response.text
