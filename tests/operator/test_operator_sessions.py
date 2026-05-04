from __future__ import annotations


def test_session_crud_requires_delete_guard(operator_client):
    created = operator_client.post("/api/sessions/new").json()
    session_id = created["session_id"]

    recent = operator_client.get("/api/sessions/recent").json()
    assert recent[0]["session_id"] == session_id

    detail = operator_client.get(f"/api/sessions/{session_id}").json()
    assert detail["session_id"] == session_id

    guarded = operator_client.delete(f"/api/sessions/{session_id}")
    assert guarded.status_code == 400

    deleted = operator_client.delete(
        f"/api/sessions/{session_id}",
        headers={"X-Confirm-Delete": "true"},
    )
    assert deleted.status_code == 200
    assert deleted.json() == {"deleted": True, "session_id": session_id}


def test_admin_sessions_partials_render(operator_client):
    operator_client.post("/api/sessions/new")
    response = operator_client.get("/admin/_partials/sessions")
    assert response.status_code == 200
    assert "view chain" in response.text
