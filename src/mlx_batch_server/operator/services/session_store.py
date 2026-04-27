"""Thread-safe in-process store for operator playground sessions."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class SessionRecord:
    session_id: str
    created_at: str
    response_count: int = 0
    last_model: str | None = None
    responses: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "response_count": self.response_count,
            "last_model": self.last_model,
        }

    def detail(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "responses": list(self.responses),
        }


class SessionStore:
    """Single owner for lightweight session state shared by routers and UI flows."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionRecord] = {}
        self._lock = threading.Lock()

    def create(self) -> dict[str, str]:
        session_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()
        with self._lock:
            self._sessions[session_id] = SessionRecord(
                session_id=session_id, created_at=now
            )
        return {"session_id": session_id, "created_at": now}

    def list_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            items = sorted(
                self._sessions.values(),
                key=lambda session: session.created_at,
                reverse=True,
            )[:limit]
            return [session.summary() for session in items]

    def get(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            entry = self._sessions.get(session_id)
            if entry is None:
                return None
            return entry.detail()

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def remember_response(self, session_id: str, response: dict[str, Any]) -> bool:
        with self._lock:
            entry = self._sessions.get(session_id)
            if entry is None:
                return False
            entry.responses.append(dict(response))
            entry.response_count = len(entry.responses)
            model = response.get("model")
            entry.last_model = model if isinstance(model, str) else None
            return True

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()


session_store = SessionStore()
