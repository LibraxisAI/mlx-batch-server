"""Explicit composition root for one concrete Qwen4Exp fused backend."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType

from ...backends.fused_mtp_mlx import MtpMlxBackend
from ...contracts import BackendHandle, BackendKind, LoadConfig, RuntimeKey
from ..mtp import MtpPolicy
from ..qwen4_exp.execution import Qwen4ExpExecutionFactoryPort
from ..qwen4_exp.request_preparation import Qwen4ExpRequestPreparerPort
from ..scheduler import SchedulerConfig
from .owner import Qwen4ExpTensorOwnerLoader
from .provider import (
    FusedTensorRuntimeRegistry,
    OmlxMtplxCacheFactory,
    OmlxMtplxExecutorFactory,
)


@dataclass(frozen=True, slots=True)
class Qwen4ExpBackendCompositionReceipt:
    """Immutable references for one inert fused Qwen4Exp object graph."""

    execution_factory: Qwen4ExpExecutionFactoryPort
    request_preparer: Qwen4ExpRequestPreparerPort
    owner_loader: Qwen4ExpTensorOwnerLoader
    registry: FusedTensorRuntimeRegistry
    executor_factory: OmlxMtplxExecutorFactory
    cache_factory: OmlxMtplxCacheFactory
    backend: MtpMlxBackend
    scheduler_config: SchedulerConfig
    mtp_policy: MtpPolicy
    capacity: int
    _shutdown_task: asyncio.Task[None] | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    async def shutdown(self, *, deadline_s: float) -> None:
        """Close the sole registry exactly once and share its terminal result."""

        _validate_deadline(deadline_s)
        task = self._shutdown_task
        if task is None:
            task = asyncio.create_task(self.registry.shutdown(deadline_s))
            object.__setattr__(self, "_shutdown_task", task)
        await asyncio.shield(task)


class _ConfiguredQwen4ExpBackend(MtpMlxBackend):
    """Bind MtpMlxBackend's option-derived policy to composition-time truth."""

    def __init__(
        self,
        *,
        executor_factory: OmlxMtplxExecutorFactory,
        cache_factory: OmlxMtplxCacheFactory,
        scheduler_config: SchedulerConfig,
        mtp_policy: MtpPolicy,
    ) -> None:
        super().__init__(
            executor_factory=executor_factory,
            cache_factory=cache_factory,
        )
        self._composition_scheduler_config = scheduler_config
        self._composition_mtp_policy = mtp_policy

    async def load(self, runtime: RuntimeKey, config: LoadConfig) -> BackendHandle:
        _validate_runtime_identity(runtime)
        canonical_config = _canonical_load_config(
            config,
            scheduler_config=self._composition_scheduler_config,
            mtp_policy=self._composition_mtp_policy,
        )
        return await super().load(runtime, canonical_config)


def compose_qwen4_exp_backend(
    *,
    request_preparer: Qwen4ExpRequestPreparerPort,
    scheduler_config: SchedulerConfig,
    mtp_policy: MtpPolicy,
    capacity: int,
    execution_factory: Qwen4ExpExecutionFactoryPort | None = None,
) -> Qwen4ExpBackendCompositionReceipt:
    """Build one inert, target-owned Qwen4Exp backend graph.

    Composition does not read checkpoint metadata, create an owner thread, load
    tensors, or touch runtime state. Those effects remain behind ``backend.load``.
    """

    if not isinstance(request_preparer, Qwen4ExpRequestPreparerPort):
        raise TypeError("request_preparer must satisfy Qwen4ExpRequestPreparerPort")
    if not isinstance(scheduler_config, SchedulerConfig):
        raise TypeError("scheduler_config must be a SchedulerConfig")
    if not isinstance(mtp_policy, MtpPolicy):
        raise TypeError("mtp_policy must be an MtpPolicy")
    _validate_composition_options(scheduler_config, mtp_policy, capacity)

    if execution_factory is None:
        from ..qwen4_exp.model.tensor import Qwen4ExpExecutionFactory

        resolved_execution_factory = Qwen4ExpExecutionFactory()
    else:
        resolved_execution_factory = execution_factory
    if not isinstance(resolved_execution_factory, Qwen4ExpExecutionFactoryPort):
        raise TypeError("execution_factory must satisfy Qwen4ExpExecutionFactoryPort")

    owner_loader = Qwen4ExpTensorOwnerLoader(
        resolved_execution_factory,
        request_preparer=request_preparer,
    )
    registry = FusedTensorRuntimeRegistry(
        owner_loader=owner_loader,
        max_entries=capacity,
    )
    executor_factory = OmlxMtplxExecutorFactory(registry)
    cache_factory = OmlxMtplxCacheFactory(registry)
    backend = _ConfiguredQwen4ExpBackend(
        executor_factory=executor_factory,
        cache_factory=cache_factory,
        scheduler_config=scheduler_config,
        mtp_policy=mtp_policy,
    )
    return Qwen4ExpBackendCompositionReceipt(
        execution_factory=resolved_execution_factory,
        request_preparer=request_preparer,
        owner_loader=owner_loader,
        registry=registry,
        executor_factory=executor_factory,
        cache_factory=cache_factory,
        backend=backend,
        scheduler_config=scheduler_config,
        mtp_policy=mtp_policy,
        capacity=capacity,
    )


def _canonical_load_config(
    config: LoadConfig,
    *,
    scheduler_config: SchedulerConfig,
    mtp_policy: MtpPolicy,
) -> LoadConfig:
    if not isinstance(config, LoadConfig):
        raise TypeError("config must be a LoadConfig")
    expected_limits = {
        "max_admitted_requests": scheduler_config.max_admitted_requests,
        "max_decode_rows": scheduler_config.max_decode_rows,
        "max_vision_prefills": scheduler_config.max_vision_prefills,
    }
    for name, expected in expected_limits.items():
        if getattr(config, name) != expected:
            raise ValueError(f"LoadConfig {name} conflicts with composition")

    if not isinstance(config.options, Mapping):
        raise TypeError("LoadConfig options must be a mapping")
    options = dict(config.options)
    expected_options: dict[str, bool | int | float] = {
        "max_prefill_rows": scheduler_config.max_prefill_rows,
        "decode_fair_share": scheduler_config.decode_fair_share,
        "terminal_history_size": scheduler_config.terminal_history_size,
        "mtp_enabled": mtp_policy.enabled,
        "mtp_draft_depth": mtp_policy.draft_depth,
        "mtp_multirow_live_proven": mtp_policy.allow_proven_multirow,
        "mtp_max_proven_rows": mtp_policy.max_proven_rows,
    }
    for name, expected in expected_options.items():
        if name in options:
            _require_exact_option(name, options[name], expected)
        options[name] = expected

    model_dir = options.get("model_dir")
    if not isinstance(model_dir, str) or not Path(model_dir).is_absolute():
        raise ValueError('LoadConfig options require absolute "model_dir"')
    return replace(config, options=MappingProxyType(options))


def _validate_runtime_identity(runtime: RuntimeKey) -> None:
    if not isinstance(runtime, RuntimeKey):
        raise TypeError("runtime must be a RuntimeKey")
    if runtime.backend is not BackendKind.FUSED_MTP_MLX:
        raise ValueError("Qwen4Exp composition requires fused_mtp_mlx")
    if not runtime.model_id or runtime.model_id.strip() != runtime.model_id:
        raise ValueError("Qwen4Exp runtime requires an exact model id")
    if not runtime.revision or runtime.revision.strip() != runtime.revision:
        raise ValueError("Qwen4Exp runtime requires an exact revision")
    if runtime.adapter_path is not None:
        raise ValueError("Qwen4Exp composition does not admit adapters")
    if runtime.draft_model_id is not None:
        raise ValueError("Qwen4Exp composition requires the embedded MTP head")


def _validate_composition_options(
    scheduler_config: SchedulerConfig,
    mtp_policy: MtpPolicy,
    capacity: int,
) -> None:
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
        raise ValueError("capacity must be a positive int")
    scheduler_ints = (
        scheduler_config.max_admitted_requests,
        scheduler_config.max_decode_rows,
        scheduler_config.max_prefill_rows,
        scheduler_config.max_vision_prefills,
        scheduler_config.terminal_history_size,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in scheduler_ints
    ):
        raise ValueError("scheduler limits must be ints")
    if (
        isinstance(scheduler_config.decode_fair_share, bool)
        or not isinstance(scheduler_config.decode_fair_share, int | float)
        or not math.isfinite(scheduler_config.decode_fair_share)
    ):
        raise ValueError("decode_fair_share must be finite")
    if not isinstance(mtp_policy.allow_proven_multirow, bool):
        raise ValueError("mtp allow_proven_multirow must be a bool")
    if not isinstance(mtp_policy.enabled, bool):
        raise ValueError("mtp enabled must be a bool")
    if (
        isinstance(mtp_policy.draft_depth, bool)
        or not isinstance(mtp_policy.draft_depth, int)
        or not 1 <= mtp_policy.draft_depth <= 8
    ):
        raise ValueError("mtp draft_depth must be an int between 1 and 8")
    if (
        isinstance(mtp_policy.max_proven_rows, bool)
        or not isinstance(mtp_policy.max_proven_rows, int)
        or mtp_policy.max_proven_rows < 1
    ):
        raise ValueError("mtp max_proven_rows must be a positive int")


def _require_exact_option(name: str, actual: object, expected: object) -> None:
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(f'LoadConfig option "{name}" conflicts with composition')


def _validate_deadline(deadline_s: float) -> None:
    if isinstance(deadline_s, bool) or not isinstance(deadline_s, int | float):
        raise TypeError("deadline_s must be numeric")
    if not math.isfinite(deadline_s) or deadline_s < 0:
        raise ValueError("deadline_s must be finite and non-negative")


__all__ = [
    "Qwen4ExpBackendCompositionReceipt",
    "compose_qwen4_exp_backend",
]
