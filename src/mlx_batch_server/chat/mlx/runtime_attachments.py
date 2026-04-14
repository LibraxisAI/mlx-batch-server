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

from .runtime_aliases import normalize_runtime_model_id

_VALID_SURFACES = frozenset({"llm", "embeddings", "visual"})
_runtime_attachments_lock = threading.Lock()
_runtime_attachments: dict[str, set[str]] = {}


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
    surface: str
    was_attached: bool
    remaining_surfaces: tuple[str, ...]


def attach_runtime_surface(model_id: str, surface: str) -> RuntimeSurfaceState:
    """Attach one product surface to a canonical runtime model id."""
    canonical_model_id = normalize_runtime_model_id(model_id)
    normalized_surface = _validate_surface(surface)

    with _runtime_attachments_lock:
        surfaces = _runtime_attachments.setdefault(canonical_model_id, set())
        was_attached = normalized_surface in surfaces
        surfaces.add(normalized_surface)
        remaining = tuple(sorted(surfaces))

    return RuntimeSurfaceState(
        model_id=canonical_model_id,
        surface=normalized_surface,
        was_attached=was_attached,
        remaining_surfaces=remaining,
    )


def release_runtime_surface(model_id: str, surface: str) -> RuntimeSurfaceState:
    """Detach one product surface from a canonical runtime model id."""
    canonical_model_id = normalize_runtime_model_id(model_id)
    normalized_surface = _validate_surface(surface)

    with _runtime_attachments_lock:
        surfaces = _runtime_attachments.get(canonical_model_id)
        was_attached = surfaces is not None and normalized_surface in surfaces

        if surfaces is not None:
            surfaces.discard(normalized_surface)
            if surfaces:
                remaining = tuple(sorted(surfaces))
            else:
                _runtime_attachments.pop(canonical_model_id, None)
                remaining = ()
        else:
            remaining = ()

    return RuntimeSurfaceState(
        model_id=canonical_model_id,
        surface=normalized_surface,
        was_attached=was_attached,
        remaining_surfaces=remaining,
    )


def get_runtime_surface_attachments(model_id: str) -> list[str]:
    """Return sorted surface attachments for one runtime model."""
    canonical_model_id = normalize_runtime_model_id(model_id)
    with _runtime_attachments_lock:
        return sorted(_runtime_attachments.get(canonical_model_id, set()))


def get_remaining_runtime_surfaces(
    model_id: str,
    releasing_surface: str | None = None,
) -> list[str]:
    """Return surfaces that would remain after one surface detaches."""
    remaining = set(get_runtime_surface_attachments(model_id))
    if releasing_surface is not None:
        remaining.discard(_validate_surface(releasing_surface))
    return sorted(remaining)


def list_runtime_surface_attachments() -> dict[str, list[str]]:
    """Return a copy of all runtime surface attachments."""
    with _runtime_attachments_lock:
        return {
            model_id: sorted(surfaces)
            for model_id, surfaces in _runtime_attachments.items()
        }


def get_attached_models(surface: str) -> list[str]:
    """Return canonical model ids currently attached to one product surface."""
    normalized_surface = _validate_surface(surface)
    with _runtime_attachments_lock:
        return sorted(
            model_id
            for model_id, surfaces in _runtime_attachments.items()
            if normalized_surface in surfaces
        )


def clear_runtime_surface_attachments(model_id: str | None = None) -> None:
    """Clear tracked runtime attachments globally or for one canonical model."""
    with _runtime_attachments_lock:
        if model_id is None:
            _runtime_attachments.clear()
            return

        canonical_model_id = normalize_runtime_model_id(model_id)
        _runtime_attachments.pop(canonical_model_id, None)
