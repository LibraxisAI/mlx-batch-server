"""In-process runtime aliases for resident model identities."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

from ...core.config import get_settings

_runtime_aliases_lock = threading.Lock()
_runtime_aliases: dict[str, RuntimeAliasTarget] = {}


@dataclass(frozen=True)
class RuntimeAliasTarget:
    """Resolved runtime identity for one aliasable resident model target."""

    model_id: str
    adapter_path: str | None = None
    draft_model_id: str | None = None


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


def _canonicalize_runtime_identifier(model_id: str | None) -> str | None:
    normalized = normalize_runtime_path(model_id)
    if not normalized:
        return None
    if (
        "/" in normalized
        and not normalized.startswith("/")
        and not normalized.startswith(".")
        and not normalized.startswith("~")
    ):
        return normalized.lower()
    return normalized


def resolve_runtime_target(
    model_id: str,
    *,
    adapter_path: str | None = None,
    draft_model_id: str | None = None,
) -> RuntimeAliasTarget:
    """Resolve one runtime identity, preserving alias-scoped adapter/draft info."""
    resolved_model = (model_id or "").strip()
    if not resolved_model:
        return RuntimeAliasTarget(
            model_id="",
            adapter_path=normalize_runtime_path(adapter_path),
            draft_model_id=_canonicalize_runtime_identifier(draft_model_id),
        )

    resolved_model = get_settings().get_model_alias(resolved_model).strip()
    resolved_adapter = normalize_runtime_path(adapter_path)
    resolved_draft = _canonicalize_runtime_identifier(draft_model_id)
    seen: set[str] = set()

    while True:
        key = _normalize_identifier(resolved_model)
        if key in seen:
            break
        seen.add(key)

        with _runtime_aliases_lock:
            target = _runtime_aliases.get(key)

        if target is None:
            break

        resolved_model = target.model_id
        if resolved_adapter is None:
            resolved_adapter = target.adapter_path
        if resolved_draft is None:
            resolved_draft = target.draft_model_id

    return RuntimeAliasTarget(
        model_id=_canonicalize_runtime_identifier(resolved_model) or "",
        adapter_path=resolved_adapter,
        draft_model_id=resolved_draft,
    )


def resolve_runtime_model_id(model_id: str) -> str:
    """Resolve static and dynamic aliases to a canonical runtime model id."""
    return resolve_runtime_target(model_id).model_id


def normalize_runtime_model_id(model_id: str) -> str:
    """Resolve aliases and canonicalize runtime model identifiers."""
    return resolve_runtime_target(model_id).model_id


def register_runtime_alias(
    alias: str,
    model_id: str,
    *,
    adapter_path: str | None = None,
    draft_model_id: str | None = None,
) -> str:
    """Register an in-process alias that resolves to the canonical runtime."""
    alias_key = _normalize_identifier(alias)
    if not alias_key:
        raise ValueError("alias cannot be empty")

    target = resolve_runtime_target(
        model_id,
        adapter_path=adapter_path,
        draft_model_id=draft_model_id,
    )
    if not target.model_id:
        raise ValueError("model_id cannot be empty")

    with _runtime_aliases_lock:
        _runtime_aliases[alias_key] = target

    return target.model_id


def get_runtime_aliases() -> dict[str, str]:
    """Return runtime aliases collapsed to canonical model ids."""
    with _runtime_aliases_lock:
        return {alias: target.model_id for alias, target in _runtime_aliases.items()}


def clear_runtime_aliases() -> None:
    """Clear runtime aliases (test helper)."""
    with _runtime_aliases_lock:
        _runtime_aliases.clear()
