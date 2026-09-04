"""Single-owner provider seam for the future concrete tensor runtime.

This module deliberately contains no tensor-library or donor-runtime imports.
An injected loader creates one opaque tensor owner and exposes only the two
target protocols consumed by ``MtpMlxBackend``. The registry then keeps those
incompatible factory calls on one exact runtime/config identity.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ...backends.fused_mtp_mlx import (
        FusedCacheLeasePort,
        FusedCachePort,
        FusedExecutorPort,
        FusedStepResult,
    )
    from ...contracts import (
        CancelToken,
        GenerationRequest,
        LoadConfig,
        ModelSpec,
        PreparedGenerationRequest,
        RuntimeKey,
    )
    from ..mtp import MtpPolicy
    from ..scheduler import SchedulerConfig, SchedulerPlan


class FusedTensorRegistryError(RuntimeError):
    """Base failure for the target-owned concrete tensor provider seam."""


class FusedTensorIdentityError(FusedTensorRegistryError):
    """A caller or loader supplied a conflicting runtime identity."""


class FusedTensorCapacityError(FusedTensorRegistryError):
    """The bounded registry cannot admit another tensor owner."""


class FusedTensorRegistryClosedError(FusedTensorRegistryError):
    """The registry or requested owner is closing and cannot accept leases."""


@dataclass(frozen=True, slots=True)
class FusedTensorOwnerBinding:
    """Validated target views over one loader-owned opaque tensor owner."""

    owner: object
    runtime: RuntimeKey
    config: LoadConfig
    scheduler_config: SchedulerConfig
    model: ModelSpec
    executor: FusedExecutorPort
    cache: FusedCachePort


@runtime_checkable
class FusedTensorOwnerLoaderPort(Protocol):
    """Dependency-injected loader for one concrete tensor owner.

    The implementation may know tensor libraries. This source-only seam does
    not. ``close`` is the sole owner-level teardown operation.
    """

    async def load(
        self,
        runtime: RuntimeKey,
        config: LoadConfig,
        scheduler_config: SchedulerConfig,
    ) -> FusedTensorOwnerBinding: ...

    async def close(self, owner: object, deadline_s: float) -> None: ...


_IdentityAtom = tuple[Any, ...]
_LeaseKind = Literal["executor", "cache"]


@dataclass(frozen=True, slots=True)
class _LoadIdentity:
    runtime: RuntimeKey
    config: _IdentityAtom


@dataclass(slots=True)
class _Entry:
    identity: _LoadIdentity
    runtime: RuntimeKey
    config: LoadConfig
    scheduler_config: SchedulerConfig
    load_task: asyncio.Task[FusedTensorOwnerBinding]
    waiters: int = 0
    executor_leases: int = 0
    cache_leases: int = 0
    closing_task: asyncio.Task[None] | None = None

    @property
    def lease_count(self) -> int:
        return self.executor_leases + self.cache_leases


class FusedTensorRuntimeRegistry:
    """Bounded single-flight registry for opaque concrete tensor owners."""

    def __init__(
        self,
        *,
        owner_loader: FusedTensorOwnerLoaderPort,
        max_entries: int = 2,
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self._owner_loader = owner_loader
        self._max_entries = max_entries
        self._entries: dict[_LoadIdentity, _Entry] = {}
        self._lock = asyncio.Lock()
        self._closed = False

    @property
    def max_entries(self) -> int:
        return self._max_entries

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    @property
    def closed(self) -> bool:
        return self._closed

    async def acquire_executor(
        self,
        runtime: RuntimeKey,
        config: LoadConfig,
        scheduler_config: SchedulerConfig,
    ) -> FusedExecutorPort:
        entry, binding = await self._acquire(
            "executor",
            runtime,
            config,
            scheduler_config=scheduler_config,
        )
        return _ExecutorLease(self, entry, binding.executor)

    async def acquire_cache(
        self,
        runtime: RuntimeKey,
        config: LoadConfig,
        model: ModelSpec,
    ) -> FusedCachePort:
        entry, binding = await self._acquire(
            "cache",
            runtime,
            config,
            model=model,
        )
        return _CachePortLease(self, entry, binding.cache)

    async def shutdown(self, deadline_s: float) -> None:
        """Reject new leases and close every loading or loaded owner once."""
        _validate_deadline(deadline_s)
        async with self._lock:
            self._closed = True
            closing = tuple(
                self._begin_close_locked(entry, deadline_s)
                for entry in self._entries.values()
            )
        if not closing:
            return
        results = await asyncio.gather(
            *(asyncio.shield(task) for task in closing),
            return_exceptions=True,
        )
        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            raise FusedTensorRegistryError(
                f"failed to close {len(failures)} tensor owner(s)"
            ) from failures[0]

    async def _acquire(
        self,
        kind: _LeaseKind,
        runtime: RuntimeKey,
        config: LoadConfig,
        *,
        scheduler_config: SchedulerConfig | None = None,
        model: ModelSpec | None = None,
    ) -> tuple[_Entry, FusedTensorOwnerBinding]:
        identity = _load_identity(runtime, config)
        async with self._lock:
            if self._closed:
                raise FusedTensorRegistryClosedError("tensor registry is closed")
            entry = self._entries.get(identity)
            if entry is None:
                if kind == "cache":
                    if any(item.runtime == runtime for item in self._entries.values()):
                        raise FusedTensorIdentityError(
                            "cache config does not match the established owner identity"
                        )
                    raise FusedTensorIdentityError(
                        "cache cannot establish tensor owner identity before executor"
                    )
                assert scheduler_config is not None
                if len(self._entries) >= self._max_entries:
                    raise FusedTensorCapacityError(
                        "concrete tensor owner registry is at capacity"
                    )
                task = asyncio.create_task(
                    self._load_owner(runtime, config, scheduler_config, identity),
                    name=f"tensor-owner-load:{runtime.model_id}",
                )
                entry = _Entry(
                    identity=identity,
                    runtime=runtime,
                    config=config,
                    scheduler_config=scheduler_config,
                    load_task=task,
                )
                self._entries[identity] = entry
            elif entry.closing_task is not None:
                raise FusedTensorRegistryClosedError(
                    "tensor owner for this identity is closing"
                )
            elif (
                scheduler_config is not None
                and scheduler_config != entry.scheduler_config
            ):
                raise FusedTensorIdentityError(
                    "scheduler config does not match the established owner identity"
                )
            entry.waiters += 1

        try:
            binding = await asyncio.shield(entry.load_task)
        except BaseException:
            await self._leave_failed_waiter(entry)
            raise

        close_task: asyncio.Task[None] | None = None
        try:
            async with self._lock:
                entry.waiters -= 1
                current = self._entries.get(identity)
                if (
                    current is not entry
                    or self._closed
                    or entry.closing_task is not None
                ):
                    if entry.waiters == 0 and entry.lease_count == 0:
                        close_task = self._begin_close_locked(entry, 0.0)
                    raise FusedTensorRegistryClosedError(
                        "tensor owner closed while its lease was loading"
                    )
                if model is not None and _model_identity(model) != _model_identity(
                    binding.model
                ):
                    if entry.waiters == 0 and entry.lease_count == 0:
                        close_task = self._begin_close_locked(entry, 0.0)
                    raise FusedTensorIdentityError(
                        "cache model does not match the loaded tensor owner"
                    )
                if kind == "executor":
                    entry.executor_leases += 1
                else:
                    entry.cache_leases += 1
                return entry, binding
        finally:
            if close_task is not None:
                await asyncio.shield(close_task)

    async def _load_owner(
        self,
        runtime: RuntimeKey,
        config: LoadConfig,
        scheduler_config: SchedulerConfig,
        identity: _LoadIdentity,
    ) -> FusedTensorOwnerBinding:
        binding = await self._owner_loader.load(runtime, config, scheduler_config)
        try:
            self._validate_binding(binding, identity, scheduler_config)
        except Exception as identity_error:
            try:
                await self._owner_loader.close(binding.owner, 0.0)
            except Exception as close_error:
                raise FusedTensorRegistryError(
                    "invalid tensor owner also failed defensive close"
                ) from close_error
            raise identity_error
        return binding

    def _validate_binding(
        self,
        binding: FusedTensorOwnerBinding,
        identity: _LoadIdentity,
        scheduler_config: SchedulerConfig,
    ) -> None:
        if binding.owner is None:
            raise FusedTensorIdentityError("tensor owner must be opaque, not absent")
        if _load_identity(binding.runtime, binding.config) != identity:
            raise FusedTensorIdentityError(
                "loader returned a different runtime/config identity"
            )
        if binding.scheduler_config != scheduler_config:
            raise FusedTensorIdentityError(
                "loader returned a different scheduler config"
            )
        if binding.model.model_id != identity.runtime.model_id:
            raise FusedTensorIdentityError("loader returned a different model id")
        if (
            identity.runtime.revision is not None
            and binding.model.revision != identity.runtime.revision
        ):
            raise FusedTensorIdentityError("loader returned a different model revision")
        if binding.executor.model_spec is not binding.model:
            raise FusedTensorIdentityError(
                "executor model must be the binding's canonical model object"
            )

    async def _leave_failed_waiter(self, entry: _Entry) -> None:
        async with self._lock:
            entry.waiters -= 1
            current = self._entries.get(entry.identity)
            if current is not entry or entry.waiters or entry.lease_count:
                return
            if (
                entry.load_task.done()
                and not entry.load_task.cancelled()
                and entry.load_task.exception() is not None
            ):
                self._entries.pop(entry.identity, None)
                return
            self._begin_close_locked(entry, 0.0)

    async def _release(
        self,
        entry: _Entry,
        kind: _LeaseKind,
        deadline_s: float,
    ) -> None:
        _validate_deadline(deadline_s)
        close_task: asyncio.Task[None] | None = None
        async with self._lock:
            current = self._entries.get(entry.identity)
            if current is not entry:
                return
            if kind == "executor":
                if entry.executor_leases < 1:
                    raise FusedTensorRegistryError(
                        "executor lease accounting underflow"
                    )
                entry.executor_leases -= 1
            else:
                if entry.cache_leases < 1:
                    raise FusedTensorRegistryError("cache lease accounting underflow")
                entry.cache_leases -= 1
            if entry.lease_count == 0 and entry.waiters == 0:
                close_task = self._begin_close_locked(entry, deadline_s)
        if close_task is not None:
            await asyncio.shield(close_task)

    def _begin_close_locked(
        self,
        entry: _Entry,
        deadline_s: float,
    ) -> asyncio.Task[None]:
        if entry.closing_task is None:
            entry.closing_task = asyncio.create_task(
                self._close_entry(entry, deadline_s),
                name=f"tensor-owner-close:{entry.runtime.model_id}",
            )
        return entry.closing_task

    async def _close_entry(self, entry: _Entry, deadline_s: float) -> None:
        try:
            try:
                binding = await entry.load_task
            except Exception:
                return
            await self._owner_loader.close(binding.owner, deadline_s)
        finally:
            async with self._lock:
                if self._entries.get(entry.identity) is entry:
                    self._entries.pop(entry.identity, None)


class _ExecutorLease:
    def __init__(
        self,
        registry: FusedTensorRuntimeRegistry,
        entry: _Entry,
        executor: FusedExecutorPort,
    ) -> None:
        self._registry = registry
        self._entry = entry
        self._executor = executor
        self._closed = False

    @property
    def model_spec(self) -> ModelSpec:
        self._require_open()
        return self._executor.model_spec

    async def prepare_request(
        self,
        request: GenerationRequest,
        cancel: CancelToken,
    ) -> PreparedGenerationRequest:
        self._require_open()
        return await self._executor.prepare_request(request, cancel)

    async def execute(
        self,
        plan: SchedulerPlan,
        requests: Mapping[str, GenerationRequest],
        mtp_policy: MtpPolicy,
    ) -> FusedStepResult:
        self._require_open()
        return await self._executor.execute(plan, requests, mtp_policy)

    async def cleanup_cancelled(self, request_id: str, reason: str) -> None:
        self._require_open()
        await self._executor.cleanup_cancelled(request_id, reason)

    def stats(self) -> Mapping[str, Any]:
        self._require_open()
        return self._executor.stats()

    async def close(self, deadline_s: float) -> None:
        _validate_deadline(deadline_s)
        if self._closed:
            return
        self._closed = True
        await self._registry._release(self._entry, "executor", deadline_s)

    def _require_open(self) -> None:
        if self._closed or self._entry.closing_task is not None:
            raise FusedTensorRegistryClosedError("executor lease is closed")


class _CachePortLease:
    def __init__(
        self,
        registry: FusedTensorRuntimeRegistry,
        entry: _Entry,
        cache: FusedCachePort,
    ) -> None:
        self._registry = registry
        self._entry = entry
        self._cache = cache
        self._closed = False

    async def acquire(
        self,
        request: PreparedGenerationRequest,
    ) -> FusedCacheLeasePort:
        self._require_open()
        return await self._cache.acquire(request)

    def stats(self) -> Mapping[str, Any]:
        self._require_open()
        return self._cache.stats()

    async def close(self, deadline_s: float) -> None:
        _validate_deadline(deadline_s)
        if self._closed:
            return
        self._closed = True
        await self._registry._release(self._entry, "cache", deadline_s)

    def _require_open(self) -> None:
        if self._closed or self._entry.closing_task is not None:
            raise FusedTensorRegistryClosedError("cache lease is closed")


class OmlxMtplxExecutorFactory:
    """Executor factory facade over one target-owned owner registry."""

    def __init__(self, registry: FusedTensorRuntimeRegistry) -> None:
        self._registry = registry

    async def load(
        self,
        runtime: RuntimeKey,
        config: LoadConfig,
        scheduler_config: SchedulerConfig,
    ) -> FusedExecutorPort:
        return await self._registry.acquire_executor(
            runtime,
            config,
            scheduler_config,
        )


class OmlxMtplxCacheFactory:
    """Cache factory facade converging on the executor's tensor owner."""

    def __init__(self, registry: FusedTensorRuntimeRegistry) -> None:
        self._registry = registry

    async def load(
        self,
        runtime: RuntimeKey,
        config: LoadConfig,
        model: ModelSpec,
    ) -> FusedCachePort:
        return await self._registry.acquire_cache(runtime, config, model)


def _load_identity(runtime: RuntimeKey, config: LoadConfig) -> _LoadIdentity:
    return _LoadIdentity(
        runtime=runtime,
        config=(
            "load-config",
            _freeze_identity(config.max_admitted_requests),
            _freeze_identity(config.max_decode_rows),
            _freeze_identity(config.max_vision_prefills),
            _freeze_identity(config.memory_budget_bytes),
            _freeze_identity(config.cache_directory),
            _freeze_identity(config.options),
        ),
    )


def _model_identity(model: ModelSpec) -> _IdentityAtom:
    return (
        "model-spec",
        model.model_id,
        model.revision,
        model.architecture,
        model.model_type,
        model.quantization,
        model.local_path,
        _freeze_identity(model.metadata),
    )


def _freeze_identity(value: Any) -> _IdentityAtom:
    if value is None:
        return ("none",)
    if isinstance(value, Enum):
        return ("enum", type(value).__module__, type(value).__qualname__, value.value)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, float):
        return ("float", value.hex())
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, Mapping):
        items: list[tuple[str, _IdentityAtom]] = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise FusedTensorIdentityError("identity mappings require string keys")
            items.append((key, _freeze_identity(item)))
        return ("mapping", tuple(sorted(items, key=lambda item: item[0])))
    if isinstance(value, tuple):
        return ("tuple", tuple(_freeze_identity(item) for item in value))
    if isinstance(value, list):
        return ("list", tuple(_freeze_identity(item) for item in value))
    raise FusedTensorIdentityError(
        f"unsupported mutable identity value: {type(value).__qualname__}"
    )


def _validate_deadline(deadline_s: float) -> None:
    if deadline_s < 0:
        raise ValueError("deadline_s must be non-negative")


__all__ = [
    "FusedTensorCapacityError",
    "FusedTensorIdentityError",
    "FusedTensorOwnerBinding",
    "FusedTensorOwnerLoaderPort",
    "FusedTensorRegistryClosedError",
    "FusedTensorRegistryError",
    "FusedTensorRuntimeRegistry",
    "OmlxMtplxCacheFactory",
    "OmlxMtplxExecutorFactory",
]
