from __future__ import annotations


def test_lifecycle_status_contains_pid_and_uptime(operator_client):
    data = operator_client.get("/api/lifecycle/status").json()
    assert data["pid"] > 0
    assert data["version"] == "0.6.0-dev"
    assert data["uptime_seconds"] >= 0


def test_restart_requires_supervisor_by_default(operator_client):
    response = operator_client.post("/api/lifecycle/restart-backend")
    assert response.status_code == 501
    assert "dev mode" in response.text.lower()
