"""In-process runtime aliases for resident model identities."""

from __future__ import annotations

import threading
from pathlib import Path

from ...core.config import get_settings

_runtime_aliases_lock = threading.Lock()
_runtime_aliases: dict[str, str] = {}


def _normalize_identifier(model_id: str) -> str:
    normalized = model_id.strip()
    if (
        "/" in normalized
        and not normalized.startswith("/")
        and not normalized.startswith(".")
        and not normalized.startswith("~")
    ):
        return normalized.lower()
    return normalized


def normalize_runtime_path(path: str | None) -> str | None:
    """Normalize path-like runtime fields so cache keys stay stable."""
    if path is None:
        return None

    normalized = path.strip()
    if not normalized:
        return None
    if normalized.startswith("~"):
        normalized = str(Path(normalized).expanduser())
    return normalized


def resolve_runtime_model_id(model_id: str) -> str:
    """Resolve static and dynamic aliases to a canonical runtime model id."""
    resolved = (model_id or "").strip()
    if not resolved:
        return resolved

    resolved = get_settings().get_model_alias(resolved).strip()
    seen: set[str] = set()

    while True:
        key = _normalize_identifier(resolved)
        if key in seen:
            return resolved
        seen.add(key)

        with _runtime_aliases_lock:
            target = _runtime_aliases.get(key)

        if not target:
            return resolved

        resolved = target


def normalize_runtime_model_id(model_id: str) -> str:
    """Resolve aliases and canonicalize runtime model identifiers."""
    normalized = normalize_runtime_path(resolve_runtime_model_id(model_id)) or ""
    if (
        "/" in normalized
        and not normalized.startswith("/")
        and not normalized.startswith(".")
        and not normalized.startswith("~")
    ):
        return normalized.lower()
    return normalized


def register_runtime_alias(alias: str, model_id: str) -> str:
    """Register an in-process alias that resolves to the canonical model id."""
    alias_key = _normalize_identifier(alias)
    if not alias_key:
        raise ValueError("alias cannot be empty")

    canonical_model_id = normalize_runtime_model_id(model_id)
    if not canonical_model_id:
        raise ValueError("model_id cannot be empty")

    with _runtime_aliases_lock:
        _runtime_aliases[alias_key] = canonical_model_id

    return canonical_model_id


def get_runtime_aliases() -> dict[str, str]:
    """Return a copy of the current runtime alias map."""
    with _runtime_aliases_lock:
        return dict(_runtime_aliases)


def clear_runtime_aliases() -> None:
    """Clear runtime aliases (test helper)."""
    with _runtime_aliases_lock:
        _runtime_aliases.clear()
