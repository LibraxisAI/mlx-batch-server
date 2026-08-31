"""Thread-safe active-job leases for exact MLX runtime identities."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

from .runtime_aliases import RuntimeAliasTarget, resolve_runtime_target

_runtime_leases_lock = threading.RLock()
_runtime_leases: dict[RuntimeAliasTarget, int] = {}


def _target(
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


def acquire_runtime_lease(
    model_id: str,
    *,
    adapter_path: str | None = None,
    draft_model_id: str | None = None,
) -> int:
    """Increment the queued/active job count for one exact runtime."""
    target = _target(
        model_id,
        adapter_path=adapter_path,
        draft_model_id=draft_model_id,
    )
    with _runtime_leases_lock:
        count = _runtime_leases.get(target, 0) + 1
        _runtime_leases[target] = count
        return count


def release_runtime_lease(
    model_id: str,
    *,
    adapter_path: str | None = None,
    draft_model_id: str | None = None,
) -> int:
    """Decrement a runtime job count and return the remaining leases."""
    target = _target(
        model_id,
        adapter_path=adapter_path,
        draft_model_id=draft_model_id,
    )
    with _runtime_leases_lock:
        count = _runtime_leases.get(target, 0)
        if count <= 1:
            _runtime_leases.pop(target, None)
            return 0
        count -= 1
        _runtime_leases[target] = count
        return count


def active_runtime_lease_count(
    model_id: str,
    *,
    adapter_path: str | None = None,
    draft_model_id: str | None = None,
) -> int:
    target = _target(
        model_id,
        adapter_path=adapter_path,
        draft_model_id=draft_model_id,
    )
    with _runtime_leases_lock:
        return _runtime_leases.get(target, 0)


def list_runtime_leases() -> list[dict[str, object]]:
    """Return observer-only exact runtime lease counts."""
    with _runtime_leases_lock:
        items = [
            {
                "model_id": target.model_id,
                "adapter_path": target.adapter_path,
                "draft_model_id": target.draft_model_id,
                "active_jobs": count,
            }
            for target, count in _runtime_leases.items()
        ]
    return sorted(
        items,
        key=lambda item: (
            str(item["model_id"]),
            str(item["adapter_path"] or ""),
            str(item["draft_model_id"] or ""),
        ),
    )


def clear_runtime_leases() -> None:
    """Clear lease state during explicit cache teardown and tests."""
    with _runtime_leases_lock:
        _runtime_leases.clear()


@contextmanager
def runtime_retirement_guard(
    model_id: str,
    *,
    adapter_path: str | None = None,
    draft_model_id: str | None = None,
) -> Iterator[bool]:
    """Atomically exclude new jobs while an idle runtime is being retired."""
    target = _target(
        model_id,
        adapter_path=adapter_path,
        draft_model_id=draft_model_id,
    )
    with _runtime_leases_lock:
        yield _runtime_leases.get(target, 0) == 0
