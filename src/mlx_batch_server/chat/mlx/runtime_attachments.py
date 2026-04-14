"""Operator-facing surface attachments for shared MLX/VLM runtimes.

This registry answers a product-truth question the raw wrapper cache cannot:
which product surfaces intentionally retain a shared resident runtime right now.

The runtime weights still live in ``wrapper_cache``. We only track higher-level
attachments such as ``llm``, ``visual``, and ``embeddings`` so one surface can
detach cleanly without accidentally evicting a runtime another surface still
expects to stay hot.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from .runtime_aliases import (
    RuntimeAliasTarget,
    normalize_runtime_model_id,
    resolve_runtime_target,
)

_VALID_SURFACES = frozenset({"llm", "embeddings", "visual"})
_runtime_attachments_lock = threading.Lock()
_runtime_attachments: dict[RuntimeAliasTarget, set[str]] = {}


def _validate_surface(surface: str) -> str:
    normalized = (surface or "").strip().lower()
    if normalized not in _VALID_SURFACES:
        allowed = ", ".join(sorted(_VALID_SURFACES))
        raise ValueError(f"Unsupported runtime surface '{surface}'. Allowed: {allowed}")
    return normalized


@dataclass(frozen=True)
class RuntimeSurfaceState:
    """Result of mutating one surface attachment for a runtime."""

    model_id: str
    adapter_path: str | None
    draft_model_id: str | None
    surface: str
    was_attached: bool
    remaining_surfaces: tuple[str, ...]


def _normalize_attachment_target(
    model_id: str,
    *,
    adapter_path: str | None = None,
    draft_model_id: str | None = None,
) -> RuntimeAliasTarget:
    return resolve_runtime_target(
        model_id,
        adapter_path=adapter_path,
        draft_model_id=draft_model_id,
    )


def _serialize_target(target: RuntimeAliasTarget) -> dict[str, str | None]:
    return {
        "model_id": target.model_id,
        "adapter_path": target.adapter_path,
        "draft_model_id": target.draft_model_id,
    }


def attach_runtime_surface(
    model_id: str,
    surface: str,
    *,
    adapter_path: str | None = None,
    draft_model_id: str | None = None,
) -> RuntimeSurfaceState:
    """Attach one product surface to a canonical runtime model id."""
    target = _normalize_attachment_target(
        model_id,
        adapter_path=adapter_path,
        draft_model_id=draft_model_id,
    )
    normalized_surface = _validate_surface(surface)

    with _runtime_attachments_lock:
        surfaces = _runtime_attachments.setdefault(target, set())
        was_attached = normalized_surface in surfaces
        surfaces.add(normalized_surface)
        remaining = tuple(sorted(surfaces))

    return RuntimeSurfaceState(
        model_id=target.model_id,
        adapter_path=target.adapter_path,
        draft_model_id=target.draft_model_id,
        surface=normalized_surface,
        was_attached=was_attached,
        remaining_surfaces=remaining,
    )


def release_runtime_surface(
    model_id: str,
    surface: str,
    *,
    adapter_path: str | None = None,
    draft_model_id: str | None = None,
) -> RuntimeSurfaceState:
    """Detach one product surface from a canonical runtime model id."""
    target = _normalize_attachment_target(
        model_id,
        adapter_path=adapter_path,
        draft_model_id=draft_model_id,
    )
    normalized_surface = _validate_surface(surface)

    with _runtime_attachments_lock:
        surfaces = _runtime_attachments.get(target)
        was_attached = surfaces is not None and normalized_surface in surfaces

        if surfaces is not None:
            surfaces.discard(normalized_surface)
            if surfaces:
                remaining = tuple(sorted(surfaces))
            else:
                _runtime_attachments.pop(target, None)
                remaining = ()
        else:
            remaining = ()

    return RuntimeSurfaceState(
        model_id=target.model_id,
        adapter_path=target.adapter_path,
        draft_model_id=target.draft_model_id,
        surface=normalized_surface,
        was_attached=was_attached,
        remaining_surfaces=remaining,
    )


def get_runtime_surface_attachments(
    model_id: str,
    *,
    adapter_path: str | None = None,
    draft_model_id: str | None = None,
) -> list[str]:
    """Return sorted surface attachments for one runtime model."""
    target = _normalize_attachment_target(
        model_id,
        adapter_path=adapter_path,
        draft_model_id=draft_model_id,
    )
    with _runtime_attachments_lock:
        return sorted(_runtime_attachments.get(target, set()))


def get_remaining_runtime_surfaces(
    model_id: str,
    releasing_surface: str | None = None,
    *,
    adapter_path: str | None = None,
    draft_model_id: str | None = None,
) -> list[str]:
    """Return surfaces that would remain after one surface detaches."""
    remaining = set(
        get_runtime_surface_attachments(
            model_id,
            adapter_path=adapter_path,
            draft_model_id=draft_model_id,
        )
    )
    if releasing_surface is not None:
        remaining.discard(_validate_surface(releasing_surface))
    return sorted(remaining)


def list_runtime_surface_attachments() -> dict[str, list[str]]:
    """Return aggregated attachments collapsed to canonical model ids."""
    with _runtime_attachments_lock:
        by_model: dict[str, set[str]] = {}
        for target, surfaces in _runtime_attachments.items():
            bucket = by_model.setdefault(target.model_id, set())
            bucket.update(surfaces)
        return {model_id: sorted(surfaces) for model_id, surfaces in by_model.items()}


def list_runtime_surface_attachments_by_runtime() -> list[dict[str, object]]:
    """Return exact runtime-key attachments for operator/debug tooling."""
    with _runtime_attachments_lock:
        items = [
            {
                **_serialize_target(target),
                "surfaces": sorted(surfaces),
            }
            for target, surfaces in _runtime_attachments.items()
        ]
    return sorted(
        items,
        key=lambda item: (
            item["model_id"] or "",
            item["adapter_path"] or "",
            item["draft_model_id"] or "",
        ),
    )


def get_attached_models(surface: str) -> list[str]:
    """Return canonical model ids with at least one attached runtime for a surface."""
    normalized_surface = _validate_surface(surface)
    with _runtime_attachments_lock:
        return sorted(
            {
                target.model_id
                for target, surfaces in _runtime_attachments.items()
                if normalized_surface in surfaces
            }
        )


def get_attached_runtime_targets(
    surface: str,
    *,
    model_id: str | None = None,
) -> list[RuntimeAliasTarget]:
    """Return exact runtime targets currently attached to one product surface."""
    normalized_surface = _validate_surface(surface)
    normalized_model_id = (
        normalize_runtime_model_id(model_id) if model_id is not None else None
    )
    with _runtime_attachments_lock:
        targets = [
            target
            for target, surfaces in _runtime_attachments.items()
            if normalized_surface in surfaces
            and (normalized_model_id is None or target.model_id == normalized_model_id)
        ]
    return sorted(
        targets,
        key=lambda target: (
            target.model_id,
            target.adapter_path or "",
            target.draft_model_id or "",
        ),
    )


def clear_runtime_surface_attachments(
    model_id: str | None = None,
    *,
    adapter_path: str | None = None,
    draft_model_id: str | None = None,
) -> None:
    """Clear tracked runtime attachments globally, for one model, or one exact key."""
    with _runtime_attachments_lock:
        if model_id is None:
            _runtime_attachments.clear()
            return

        if adapter_path is None and draft_model_id is None:
            canonical_model_id = normalize_runtime_model_id(model_id)
            keys_to_remove = [
                target
                for target in _runtime_attachments
                if target.model_id == canonical_model_id
            ]
            for target in keys_to_remove:
                _runtime_attachments.pop(target, None)
            return

        target = _normalize_attachment_target(
            model_id,
            adapter_path=adapter_path,
            draft_model_id=draft_model_id,
        )
        _runtime_attachments.pop(target, None)
