"""Bounded target-owned model lifecycle and single-flight loading."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from .admission import AdmissionController
from .contracts import (
    AdmissionLease,
    BackendFactory,
    BackendHandle,
    BackendKind,
    CapabilityReport,
    LoadConfig,
    ModelState,
    ProcessState,
    RoleName,
    RoleSpec,
    RuntimeKey,
)

if TYPE_CHECKING:
    from .readiness import ReadinessService
    from .roles import RoleDirectory


class RuntimeManagerError(RuntimeError):
    """Base error for target runtime lifecycle failures."""


class RuntimeCapacityError(RuntimeManagerError):
    """Raised when the bounded runtime registry cannot admit another key."""


class RuntimeUnavailableError(RuntimeManagerError):
    """Raised when a dead role cannot perform a local wake."""


@dataclass(slots=True)
class _RuntimeRecord:
    state: ModelState = ModelState.COLD
    handle: BackendHandle | None = None
    error: str | None = None
    load_task: asyncio.Task[BackendHandle] | None = None
    unload_task: asyncio.Task[bool] | None = None


class RuntimeManager:
    """Own runtime load/unload state without owning model stepping.

    Concurrent callers for the same ``RuntimeKey`` await one shielded load task.
    A cancelled waiter therefore cannot cancel the shared model load. Failures
    remain visible as DEGRADED until a later successful acquisition clears them.
    """

    def __init__(
        self,
        factories: Mapping[BackendKind | str, BackendFactory],
        *,
        roles: RoleDirectory | None = None,
        readiness: ReadinessService | None = None,
        admission: AdmissionController | None = None,
        default_load_config: LoadConfig | None = None,
        max_runtimes: int = 8,
        max_parallel_loads: int = 1,
        rejected_handle_close_deadline_s: float = 5.0,
    ) -> None:
        if max_runtimes < 1:
            raise ValueError("max_runtimes must be positive")
        if max_parallel_loads < 1:
            raise ValueError("max_parallel_loads must be positive")
        if rejected_handle_close_deadline_s < 0:
            raise ValueError("rejected handle close deadline must be non-negative")
        if readiness is not None and roles is not None and readiness.roles is not roles:
            raise ValueError("readiness and manager must share one RoleDirectory")

        self._factories = {
            key if isinstance(key, BackendKind) else BackendKind(key): factory
            for key, factory in factories.items()
        }
        self._roles = roles or (readiness.roles if readiness is not None else None)
        self._readiness = readiness
        self._admission = admission or AdmissionController()
        self._default_load_config = default_load_config or LoadConfig()
        self._max_runtimes = max_runtimes
        self._rejected_handle_close_deadline_s = rejected_handle_close_deadline_s
        self._load_slots = asyncio.Semaphore(max_parallel_loads)
        self._lock = asyncio.Lock()
        self._records: dict[RuntimeKey, _RuntimeRecord] = {}
        self._role_runtimes: dict[RoleName, RuntimeKey] = {}
        self._closing = False
        self._closed = False
        self._shutdown_task: asyncio.Task[None] | None = None

    async def acquire(
        self,
        runtime: RuntimeKey,
        config: LoadConfig | None = None,
    ) -> BackendHandle:
        """Return a resident handle, joining one in-flight load for this key."""
        while True:
            wait_for_unload: asyncio.Task[bool] | None = None
            async with self._lock:
                self._require_accepting()
                record = self._record(runtime)
                if record.handle is not None and record.state is ModelState.READY:
                    return record.handle
                if record.unload_task is not None:
                    wait_for_unload = record.unload_task
                    load_task = None
                elif record.load_task is not None:
                    load_task = record.load_task
                else:
                    record.state = ModelState.LOADING
                    record.error = None
                    load_task = asyncio.create_task(
                        self._load_and_publish(
                            runtime,
                            record,
                            config or self._default_load_config,
                        ),
                        name=f"runtime-load:{runtime.model_id}",
                    )
                    record.load_task = load_task

            if wait_for_unload is not None:
                await asyncio.shield(wait_for_unload)
                continue
            assert load_task is not None
            return await asyncio.shield(load_task)

    async def acquire_role(
        self,
        role: RoleName | str,
        *,
        runtime: RuntimeKey | None = None,
        config: LoadConfig | None = None,
        capabilities: CapabilityReport | None = None,
        receipt: Mapping[str, Any] | None = None,
    ) -> BackendHandle:
        """Wake one configured role and publish its exact readiness transition."""
        if self._roles is None or self._readiness is None:
            raise RuntimeManagerError("role acquisition requires roles and readiness")
        async with self._lock:
            self._require_accepting()
        spec = self._roles.resolve(role)
        snapshot = self._readiness.snapshot(spec.name)
        if snapshot.process_state is ProcessState.DEAD:
            raise RuntimeUnavailableError(
                f"role {spec.name.value!r} process is dead: {snapshot.error}"
            )

        runtime_key = runtime or self._roles.runtime_key(spec.name)
        if runtime_key.model_id != spec.requested_model:
            raise ValueError(
                f"role {spec.name.value!r} requested {spec.requested_model!r}, "
                f"runtime key names {runtime_key.model_id!r}"
            )
        if runtime_key.backend is not spec.backend:
            raise ValueError(
                f"role {spec.name.value!r} requires backend {spec.backend.value!r}"
            )
        if runtime_key.revision != spec.revision:
            raise ValueError(
                f"role {spec.name.value!r} requires revision {spec.revision!r}"
            )

        load_config = self._role_load_config(
            spec,
            config or self._default_load_config,
        )

        self._role_runtimes[spec.name] = runtime_key
        if self.status(runtime_key)["state"] != ModelState.READY.value:
            self._readiness.mark_loading(spec.name)
        try:
            handle = await self.acquire(runtime_key, load_config)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self._closing:
                raise
            self._readiness.mark_degraded(spec.name, self._error_text(exc))
            raise
        self._readiness.mark_ready(
            spec.name,
            loaded_model=runtime_key.model_id,
            backend=runtime_key.backend,
            capabilities=(
                capabilities
                if capabilities is not None
                else self._handle_capabilities(handle)
            ),
            receipt=receipt,
        )
        return handle

    def role_capabilities(
        self,
        role: RoleName | str,
    ) -> CapabilityReport | None:
        """Return current handle facts instead of a load-time capability copy."""

        if self._roles is None:
            return None
        name = self._roles.resolve(role).name
        runtime = self._role_runtimes.get(name)
        if runtime is None:
            return None
        record = self._records.get(runtime)
        if record is None or record.handle is None:
            return None
        return self._handle_capabilities(record.handle)

    def role_stats(self, role: RoleName | str) -> Mapping[str, Any] | None:
        """Return an observer-only snapshot from the role's resident handle."""

        if self._roles is None:
            return None
        name = self._roles.resolve(role).name
        runtime = self._role_runtimes.get(name)
        if runtime is None:
            return None
        record = self._records.get(runtime)
        if record is None or record.handle is None:
            return None
        return dict(record.handle.stats())

    @staticmethod
    def _handle_capabilities(handle: BackendHandle) -> CapabilityReport | None:
        capabilities = getattr(handle, "capabilities", None)
        return capabilities if isinstance(capabilities, CapabilityReport) else None

    @staticmethod
    def _role_load_config(spec: RoleSpec, config: LoadConfig) -> LoadConfig:
        if spec.model_dir is None:
            return config
        options = dict(config.options)
        configured_model_dir = options.get("model_dir")
        if configured_model_dir not in (None, spec.model_dir):
            raise ValueError(
                f"role {spec.name.value!r} model_dir is controlled by the role manifest"
            )
        options["model_dir"] = spec.model_dir
        return replace(config, options=MappingProxyType(options))

    async def admit(
        self,
        role: RoleName | str,
        *,
        timeout_s: float | None = None,
    ) -> AdmissionLease:
        async with self._lock:
            self._require_accepting()
        if self._roles is not None:
            role = self._roles.resolve(role).name
        return await self._admission.acquire(role, timeout_s=timeout_s)

    async def shutdown(self, *, deadline_s: float) -> None:
        """Stop new work and close every loaded runtime within one deadline."""

        if deadline_s < 0:
            raise ValueError("deadline_s must be non-negative")
        async with self._lock:
            if self._closed:
                return
            self._closing = True
            shutdown_task = self._shutdown_task
            if shutdown_task is None:
                shutdown_task = asyncio.create_task(
                    self._shutdown_and_publish(deadline_s),
                    name="runtime-manager-shutdown",
                )
                self._shutdown_task = shutdown_task
        await asyncio.shield(shutdown_task)

    async def unload(self, runtime: RuntimeKey, *, deadline_s: float) -> bool:
        """Close one resident runtime once, joining concurrent unload callers."""
        if deadline_s < 0:
            raise ValueError("deadline_s must be non-negative")
        while True:
            wait_for_load: asyncio.Task[BackendHandle] | None = None
            async with self._lock:
                record = self._records.get(runtime)
                if record is None:
                    return False
                if record.load_task is not None:
                    wait_for_load = record.load_task
                    unload_task = None
                elif record.unload_task is not None:
                    unload_task = record.unload_task
                elif record.handle is None:
                    return record.state is ModelState.COLD and record.error is None
                else:
                    record.state = ModelState.UNLOADING
                    record.error = None
                    self._mark_roles_unloading(runtime)
                    unload_task = asyncio.create_task(
                        self._unload_and_publish(runtime, record, deadline_s),
                        name=f"runtime-unload:{runtime.model_id}",
                    )
                    record.unload_task = unload_task

            if wait_for_load is not None:
                await asyncio.shield(wait_for_load)
                continue
            assert unload_task is not None
            return await asyncio.shield(unload_task)

    def status(self, runtime: RuntimeKey) -> dict[str, Any]:
        """Return observer-only lifecycle state for one exact runtime key."""
        record = self._records.get(runtime)
        if record is None:
            return {
                "state": ModelState.COLD.value,
                "loaded": False,
                "loading": False,
                "unloading": False,
                "error": None,
            }
        return {
            "state": record.state.value,
            "loaded": record.handle is not None,
            "loading": record.load_task is not None,
            "unloading": record.unload_task is not None,
            "error": record.error,
        }

    def loaded_runtime_keys(self) -> tuple[RuntimeKey, ...]:
        return tuple(
            runtime
            for runtime, record in self._records.items()
            if record.handle is not None and record.state is ModelState.READY
        )

    def _record(self, runtime: RuntimeKey) -> _RuntimeRecord:
        record = self._records.get(runtime)
        if record is not None:
            return record
        if len(self._records) >= self._max_runtimes:
            raise RuntimeCapacityError(
                f"runtime registry limit {self._max_runtimes} reached"
            )
        record = _RuntimeRecord()
        self._records[runtime] = record
        return record

    def _require_accepting(self) -> None:
        if self._closing or self._closed:
            raise RuntimeUnavailableError("runtime manager is shutting down")

    async def _shutdown_and_publish(self, deadline_s: float) -> None:
        loop = asyncio.get_running_loop()
        deadline_at = loop.time() + deadline_s
        failures: list[str] = []
        try:
            async with self._lock:
                runtimes = tuple(self._records)

            for runtime in runtimes:
                async with self._lock:
                    record = self._records[runtime]
                    load_task = record.load_task
                if load_task is not None:
                    remaining = deadline_at - loop.time()
                    if remaining <= 0:
                        failures.append(f"{runtime.model_id}: load drain timed out")
                        continue
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(load_task),
                            timeout=remaining,
                        )
                    except TimeoutError:
                        failures.append(f"{runtime.model_id}: load drain timed out")
                        continue
                    except Exception:
                        pass

                remaining = max(0.0, deadline_at - loop.time())
                try:
                    await self.unload(runtime, deadline_s=remaining)
                except Exception as exc:
                    failures.append(f"{runtime.model_id}: {self._error_text(exc)}")

            async with self._lock:
                live = tuple(
                    runtime.model_id
                    for runtime, record in self._records.items()
                    if record.handle is not None
                    or record.load_task is not None
                    or record.unload_task is not None
                )
                if not failures and not live:
                    self._closed = True
            if failures or live:
                details = [*failures]
                if live:
                    details.append("still live: " + ", ".join(live))
                raise RuntimeManagerError(
                    "runtime manager shutdown incomplete: " + "; ".join(details)
                )
        finally:
            async with self._lock:
                if self._shutdown_task is asyncio.current_task():
                    self._shutdown_task = None

    async def _load_and_publish(
        self,
        runtime: RuntimeKey,
        record: _RuntimeRecord,
        config: LoadConfig,
    ) -> BackendHandle:
        try:
            factory = self._factories.get(runtime.backend)
            if factory is None:
                raise RuntimeUnavailableError(
                    f"no factory registered for backend {runtime.backend.value!r}"
                )
            async with self._load_slots:
                handle = await factory.load(runtime, config)
            if handle.runtime_key != runtime:
                mismatch = RuntimeManagerError(
                    "backend returned a handle for a different runtime key"
                )
                try:
                    await handle.close(self._rejected_handle_close_deadline_s)
                except Exception as close_error:
                    raise RuntimeManagerError(
                        f"{mismatch}; rejected handle cleanup failed: "
                        f"{self._error_text(close_error)}"
                    ) from close_error
                raise mismatch
        except Exception as exc:
            async with self._lock:
                if record.load_task is asyncio.current_task():
                    record.load_task = None
                    record.state = ModelState.DEGRADED
                    record.error = self._error_text(exc)
            raise

        async with self._lock:
            if record.load_task is asyncio.current_task():
                record.handle = handle
                record.load_task = None
                record.state = ModelState.READY
                record.error = None
        return handle

    async def _unload_and_publish(
        self,
        runtime: RuntimeKey,
        record: _RuntimeRecord,
        deadline_s: float,
    ) -> bool:
        handle = record.handle
        assert handle is not None
        try:
            await handle.close(deadline_s)
        except Exception as exc:
            error = self._error_text(exc)
            async with self._lock:
                if record.unload_task is asyncio.current_task():
                    record.unload_task = None
                    record.state = ModelState.DEGRADED
                    record.error = error
            self._mark_roles_degraded(runtime, error)
            raise

        async with self._lock:
            if record.unload_task is asyncio.current_task():
                record.handle = None
                record.unload_task = None
                record.state = ModelState.COLD
                record.error = None
        self._mark_roles_cold(runtime)
        return True

    def _role_names_for(self, runtime: RuntimeKey) -> tuple[RoleName, ...]:
        return tuple(
            role
            for role, role_runtime in self._role_runtimes.items()
            if role_runtime == runtime
        )

    def _mark_roles_unloading(self, runtime: RuntimeKey) -> None:
        if self._readiness is None:
            return
        for role in self._role_names_for(runtime):
            self._readiness.mark_unloading(role)

    def _mark_roles_degraded(self, runtime: RuntimeKey, error: str) -> None:
        if self._readiness is None:
            return
        for role in self._role_names_for(runtime):
            self._readiness.mark_degraded(
                role,
                error,
                transition="unload_failed",
            )

    def _mark_roles_cold(self, runtime: RuntimeKey) -> None:
        if self._readiness is None:
            return
        for role in self._role_names_for(runtime):
            self._readiness.mark_cold(role)

    @staticmethod
    def _error_text(exc: BaseException) -> str:
        detail = str(exc).strip()
        return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__


__all__ = [
    "RuntimeCapacityError",
    "RuntimeManager",
    "RuntimeManagerError",
    "RuntimeUnavailableError",
]
