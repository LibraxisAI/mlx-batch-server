"""Protocol-neutral contracts for the single target-owned inference runtime."""

from __future__ import annotations

from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .events import TurnEvent


class BackendKind(StrEnum):
    FUSED_MTP_MLX = "fused_mtp_mlx"
    LEGACY_MLX = "legacy_mlx"


class RoleName(StrEnum):
    MAIN = "main"
    CANARY = "canary"
    FLEX = "flex"
    VISION = "vision"


class ProcessState(StrEnum):
    DEAD = "dead"
    ALIVE = "alive"


class ModelState(StrEnum):
    COLD = "cold"
    LOADING = "loading"
    READY = "ready"
    UNLOADING = "unloading"
    DEGRADED = "degraded"


class AdmissionDisposition(StrEnum):
    ADMIT = "admit"
    WAIT = "wait"
    RETRY = "retry"
    REJECT = "reject"


class RequestModality(StrEnum):
    TEXT = "text"
    VISION = "vision"


@dataclass(frozen=True, slots=True)
class RuntimeKey:
    """Identity of one loadable model runtime, including backend-sensitive state."""

    model_id: str
    revision: str | None = None
    adapter_path: str | None = None
    draft_model_id: str | None = None
    backend: BackendKind = BackendKind.FUSED_MTP_MLX


@dataclass(frozen=True, slots=True)
class TensorMaterializationReceipt:
    """Immutable proof that one exact runtime crossed its MLX eval barrier."""

    schema: str
    load_id: str
    runtime: RuntimeKey
    qwen4_exp_plan_sha256: str
    artifact_inventory_sha256: str
    parameter_manifest_sha256: str
    parameter_path_count: int
    evaluated_leaf_count: int
    evaluated_logical_bytes: int
    owner_thread_id: int
    completed_at_monotonic_ns: int
    checkpoint_content_sha256: str
    _issuer_authority: object = field(repr=False, compare=False)
    barrier: str = "mx.eval"

    def __post_init__(self) -> None:
        if self.schema != "mlx-tensor-materialization.v1":
            raise ValueError("unsupported tensor materialization receipt schema")
        if not self.load_id:
            raise ValueError("tensor materialization load_id must not be empty")
        if self.runtime.backend is not BackendKind.FUSED_MTP_MLX:
            raise ValueError("tensor materialization requires the fused backend")
        for name, digest in (
            ("Qwen4Exp plan", self.qwen4_exp_plan_sha256),
            ("artifact inventory", self.artifact_inventory_sha256),
            ("parameter manifest", self.parameter_manifest_sha256),
            ("checkpoint content", self.checkpoint_content_sha256),
        ):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"{name} digest must be lowercase SHA-256")
        if self.parameter_path_count < self.evaluated_leaf_count:
            raise ValueError("parameter paths cannot be fewer than evaluated leaves")
        if self.evaluated_leaf_count < 1 or self.evaluated_logical_bytes < 1:
            raise ValueError("tensor materialization must evaluate non-empty leaves")
        if self.owner_thread_id < 1 or self.completed_at_monotonic_ns < 1:
            raise ValueError(
                "tensor materialization owner and completion must be known"
            )
        if self.barrier != "mx.eval":
            raise ValueError("unsupported tensor materialization barrier")
        if self._issuer_authority is None:
            raise ValueError("tensor materialization issuer authority is required")

    def _issued_by(self, authority: object) -> bool:
        """Authenticate the private concrete issuer without serializing its seal."""

        return self._issuer_authority is authority


@dataclass(frozen=True, slots=True)
class ModelSpec:
    model_id: str
    revision: str | None = None
    architecture: str | None = None
    model_type: str | None = None
    quantization: str | None = None
    local_path: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CapabilityReport:
    supported: bool
    backend: BackendKind
    architecture: str | None = None
    text: bool = True
    vision: bool = False
    tools: bool = False
    mtp: bool = False
    continuous_batching: bool = False
    cache_modes: tuple[str, ...] = ()
    rejection_reasons: tuple[str, ...] = ()
    facts: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LoadConfig:
    max_admitted_requests: int = 8
    max_decode_rows: int = 4
    max_vision_prefills: int = 2
    memory_budget_bytes: int | None = None
    cache_directory: str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RoleSpec:
    name: RoleName
    port: int
    requested_model: str
    backend: BackendKind
    revision: str | None = None
    model_dir: str | None = None
    pinned: bool = False
    local_required: bool = True
    capabilities: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RoleSnapshot:
    role: RoleName
    process_state: ProcessState
    model_state: ModelState
    requested_model: str
    loaded_model: str | None
    backend: BackendKind | None
    capabilities: CapabilityReport | None
    transition: str | None = None
    error: str | None = None
    receipt: Mapping[str, Any] | None = None
    materialization: TensorMaterializationReceipt | None = None


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    response_id: str
    runtime: RuntimeKey
    messages: Sequence[Mapping[str, Any]]
    media: Sequence[Mapping[str, Any]] = ()
    tools: Sequence[Mapping[str, Any]] = ()
    sampling: Mapping[str, Any] = field(default_factory=dict)
    reasoning: Mapping[str, Any] = field(default_factory=dict)
    lineage: Sequence[Mapping[str, Any]] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PreparedGenerationRequest:
    """Backend-sealed request produced before admission and tensor mutation."""

    request: GenerationRequest
    modality: RequestModality
    backend_payload: object | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.request, GenerationRequest):
            raise TypeError("request must be a GenerationRequest")
        if not isinstance(self.modality, RequestModality):
            raise TypeError("modality must be a RequestModality")
        if self.request.media and self.modality is not RequestModality.VISION:
            raise ValueError("canonical request media requires vision modality")
        if self.modality is RequestModality.VISION:
            if not self.request.media:
                raise ValueError("vision modality requires canonical request media")
            if self.backend_payload is None:
                raise ValueError("vision requests require a sealed backend payload")


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    disposition: AdmissionDisposition
    reason: str
    retry_after_s: float | None = None
    deadline_s: float | None = None


@runtime_checkable
class CancelToken(Protocol):
    @property
    def cancelled(self) -> bool: ...

    @property
    def reason(self) -> str | None: ...

    def cancel(self, reason: str) -> bool: ...


@runtime_checkable
class TurnSink(Protocol):
    def emit(self, event: TurnEvent) -> None: ...


@runtime_checkable
class AdmissionLease(Protocol):
    @property
    def decision(self) -> AdmissionDecision: ...

    def release(self) -> None: ...


@runtime_checkable
class BackendTurn(Protocol):
    @property
    def response_id(self) -> str: ...

    def cancel(self, reason: str) -> bool | None: ...

    def wait_closed(self) -> Awaitable[None]: ...


@runtime_checkable
class BackendHandle(Protocol):
    @property
    def runtime_key(self) -> RuntimeKey: ...

    @property
    def materialization_receipt(self) -> TensorMaterializationReceipt | None: ...

    def start_turn(
        self,
        request: GenerationRequest,
        sink: TurnSink,
        cancel: CancelToken,
    ) -> Awaitable[BackendTurn]: ...

    def stats(self) -> Mapping[str, Any]: ...

    def close(self, deadline_s: float) -> Awaitable[None]: ...


@runtime_checkable
class PreparedBackendHandle(Protocol):
    """Optional backend extension for asynchronous pre-admission preparation."""

    def prepare_request(
        self,
        request: GenerationRequest,
        cancel: CancelToken,
    ) -> Awaitable[PreparedGenerationRequest]: ...

    def start_prepared_turn(
        self,
        prepared: PreparedGenerationRequest,
        sink: TurnSink,
        cancel: CancelToken,
    ) -> Awaitable[BackendTurn]: ...


@runtime_checkable
class BackendFactory(Protocol):
    def probe(self, model: ModelSpec) -> CapabilityReport: ...

    def load(
        self,
        runtime: RuntimeKey,
        config: LoadConfig,
    ) -> Awaitable[BackendHandle]: ...
