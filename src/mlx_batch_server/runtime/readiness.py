"""Canonical, observer-safe process and model readiness snapshots."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from .contracts import (
    BackendKind,
    CapabilityReport,
    ModelState,
    ProcessState,
    RoleName,
    RoleSnapshot,
)

if TYPE_CHECKING:
    from .roles import RoleDirectory


class ReadinessService:
    """Own role state without deriving health from loaded-model counts."""

    def __init__(
        self,
        roles: RoleDirectory,
        *,
        receipt: Mapping[str, Any] | None = None,
    ) -> None:
        self._roles = roles
        self._lock = threading.RLock()
        frozen_receipt = None if receipt is None else dict(receipt)
        self._snapshots = {
            spec.name: RoleSnapshot(
                role=spec.name,
                process_state=ProcessState.ALIVE,
                model_state=ModelState.COLD,
                requested_model=spec.requested_model,
                loaded_model=None,
                backend=spec.backend,
                capabilities=None,
                transition="configured",
                error=None,
                receipt=frozen_receipt,
            )
            for spec in roles.specs()
        }

    @property
    def roles(self) -> RoleDirectory:
        return self._roles

    def snapshot(self, role: RoleName | str) -> RoleSnapshot:
        name = self._roles.resolve(role).name
        with self._lock:
            return self._snapshots[name]

    def snapshots(self) -> tuple[RoleSnapshot, ...]:
        with self._lock:
            return tuple(self._snapshots[spec.name] for spec in self._roles.specs())

    def is_loadable(self, role: RoleName | str) -> bool:
        snapshot = self.snapshot(role)
        return (
            snapshot.process_state is ProcessState.ALIVE
            and snapshot.model_state is ModelState.COLD
        )

    def is_available(self, role: RoleName | str) -> bool:
        """Treat an alive cold/loading role as available, never as dead."""
        snapshot = self.snapshot(role)
        return (
            snapshot.process_state is ProcessState.ALIVE
            and snapshot.model_state
            in {ModelState.COLD, ModelState.LOADING, ModelState.READY}
        )

    def is_ready(self, role: RoleName | str) -> bool:
        snapshot = self.snapshot(role)
        return (
            snapshot.process_state is ProcessState.ALIVE
            and snapshot.model_state is ModelState.READY
            and snapshot.loaded_model == snapshot.requested_model
            and snapshot.error is None
        )

    def mark_loading(self, role: RoleName | str) -> RoleSnapshot:
        name = self._roles.resolve(role).name
        with self._lock:
            current = self._snapshots[name]
            if current.process_state is ProcessState.DEAD:
                raise RuntimeError(f"cannot load dead role {name.value!r}")
            return self._store(
                name,
                replace(
                    current,
                    model_state=ModelState.LOADING,
                    loaded_model=None,
                    capabilities=None,
                    transition="loading",
                    error=None,
                ),
            )

    def mark_ready(
        self,
        role: RoleName | str,
        *,
        loaded_model: str,
        backend: BackendKind,
        capabilities: CapabilityReport | None = None,
        receipt: Mapping[str, Any] | None = None,
    ) -> RoleSnapshot:
        name = self._roles.resolve(role).name
        with self._lock:
            current = self._snapshots[name]
            if current.process_state is ProcessState.DEAD:
                raise RuntimeError(f"cannot ready dead role {name.value!r}")
            if loaded_model != current.requested_model:
                raise ValueError(
                    f"role {name.value!r} requested {current.requested_model!r}, "
                    f"loaded {loaded_model!r}"
                )
            return self._store(
                name,
                replace(
                    current,
                    model_state=ModelState.READY,
                    loaded_model=loaded_model,
                    backend=backend,
                    capabilities=capabilities,
                    transition="ready",
                    error=None,
                    receipt=current.receipt if receipt is None else dict(receipt),
                ),
            )

    def mark_unloading(self, role: RoleName | str) -> RoleSnapshot:
        name = self._roles.resolve(role).name
        with self._lock:
            current = self._snapshots[name]
            if current.loaded_model is None:
                raise RuntimeError(f"role {name.value!r} has no loaded model")
            return self._store(
                name,
                replace(
                    current,
                    model_state=ModelState.UNLOADING,
                    transition="unloading",
                    error=None,
                ),
            )

    def mark_cold(self, role: RoleName | str) -> RoleSnapshot:
        name = self._roles.resolve(role).name
        with self._lock:
            current = self._snapshots[name]
            return self._store(
                name,
                replace(
                    current,
                    model_state=ModelState.COLD,
                    loaded_model=None,
                    capabilities=None,
                    transition="cold",
                    error=None,
                ),
            )

    def mark_degraded(
        self,
        role: RoleName | str,
        error: str,
        *,
        transition: str = "load_failed",
    ) -> RoleSnapshot:
        if not error.strip():
            raise ValueError("degraded readiness requires a non-empty error")
        if not transition.strip():
            raise ValueError("degraded readiness requires a non-empty transition")
        name = self._roles.resolve(role).name
        with self._lock:
            current = self._snapshots[name]
            return self._store(
                name,
                replace(
                    current,
                    model_state=ModelState.DEGRADED,
                    transition=transition,
                    error=error,
                ),
            )

    def mark_dead(self, role: RoleName | str, error: str) -> RoleSnapshot:
        if not error.strip():
            raise ValueError("dead readiness requires a non-empty error")
        name = self._roles.resolve(role).name
        with self._lock:
            current = self._snapshots[name]
            return self._store(
                name,
                replace(
                    current,
                    process_state=ProcessState.DEAD,
                    model_state=ModelState.COLD,
                    loaded_model=None,
                    capabilities=None,
                    transition="process_dead",
                    error=error,
                ),
            )

    def mark_alive(self, role: RoleName | str) -> RoleSnapshot:
        name = self._roles.resolve(role).name
        with self._lock:
            current = self._snapshots[name]
            return self._store(
                name,
                replace(
                    current,
                    process_state=ProcessState.ALIVE,
                    model_state=ModelState.COLD,
                    loaded_model=None,
                    capabilities=None,
                    transition="process_alive",
                    error=None,
                ),
            )

    def _store(self, name: RoleName, snapshot: RoleSnapshot) -> RoleSnapshot:
        self._snapshots[name] = snapshot
        return snapshot


__all__ = [
    "ModelState",
    "ProcessState",
    "ReadinessService",
    "RoleSnapshot",
]
