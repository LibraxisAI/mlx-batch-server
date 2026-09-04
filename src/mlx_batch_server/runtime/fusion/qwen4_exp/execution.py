# SPDX-License-Identifier: Apache-2.0
"""Single execution contract for the target-owned Qwen4Exp runtime.

The contract is a LibraxisAI boundary informed by MTPLX Qwen4Exp/MTP
semantics at ``6d0ddf0575faa9acf77e63c57e48ea1602a7e4ab`` and oMLX batch
mechanics at ``e467261edc786efd33b1e9023d5c4a827f8aa1c1``. Implementations are
synchronous because every call is serialized by the target tensor owner.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ...backends.fused_mtp_mlx import FusedStepResult
    from ...contracts import (
        GenerationRequest,
        LoadConfig,
        ModelSpec,
        PreparedGenerationRequest,
        RuntimeKey,
    )
    from ..cache import CacheCleanupReceipt, CacheReleaseReason
    from ..mtp import MtpPolicy
    from ..scheduler import SchedulerConfig, SchedulerPlan
    from .model.load_plan import Qwen4ExpModelLoadPlan


@runtime_checkable
class Qwen4ExpExecutionPort(Protocol):
    """All mutable model execution called exclusively by the owner thread."""

    @property
    def model_spec(self) -> ModelSpec: ...

    def reserve(self, request: PreparedGenerationRequest, lease_id: str) -> object: ...

    def execute(
        self,
        plan: SchedulerPlan,
        reservations: Mapping[str, object],
        requests: Mapping[str, GenerationRequest],
        mtp_policy: MtpPolicy,
    ) -> FusedStepResult: ...

    def abort(self, reservation: object, reason: str) -> None: ...

    def cleanup(
        self,
        reservation: object,
        reason: CacheReleaseReason,
    ) -> CacheCleanupReceipt: ...

    def stats(self) -> Mapping[str, Any]: ...

    def shutdown(self, deadline_s: float) -> None: ...


@dataclass(frozen=True, slots=True)
class Qwen4ExpExecutionBinding:
    """Exact result of loading the unresolved Qwen4Exp tensor ABI."""

    execution: Qwen4ExpExecutionPort
    runtime: RuntimeKey
    config: LoadConfig
    scheduler_config: SchedulerConfig
    model: ModelSpec


@runtime_checkable
class Qwen4ExpPreparedExecutionFactoryPort(Protocol):
    """Metadata-frozen factory whose load runs only on the owner thread."""

    @property
    def runtime(self) -> RuntimeKey: ...

    @property
    def config(self) -> LoadConfig: ...

    @property
    def scheduler_config(self) -> SchedulerConfig: ...

    @property
    def model_plan(self) -> Qwen4ExpModelLoadPlan: ...

    def load(
        self,
    ) -> Qwen4ExpExecutionBinding: ...


@runtime_checkable
class Qwen4ExpExecutionFactoryPort(Protocol):
    """Prepare immutable metadata before any tensor-owner mailbox exists."""

    def prepare(
        self,
        runtime: RuntimeKey,
        config: LoadConfig,
        scheduler_config: SchedulerConfig,
    ) -> Qwen4ExpPreparedExecutionFactoryPort: ...


__all__ = [
    "Qwen4ExpExecutionBinding",
    "Qwen4ExpExecutionFactoryPort",
    "Qwen4ExpExecutionPort",
    "Qwen4ExpPreparedExecutionFactoryPort",
]
