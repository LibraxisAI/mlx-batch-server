"""Explicit source-only composition for the canonical Responses runtime."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

from ..chat.anthropic.runtime_source import RuntimeAnthropicTurnSource
from ..chat.anthropic.turn_source import (
    AnthropicTurnSource,
    clear_turn_source,
    register_turn_source,
)
from ..provenance import BuildReceipt
from ..runtime.admission import AdmissionController
from ..runtime.agentic import HostedAgenticRuntimeStarter
from ..runtime.backends.legacy_mlx import (
    LegacyMlxBackend,
    LegacyPortProvider,
)
from ..runtime.backends.legacy_provider import CachedLegacyPortProvider
from ..runtime.contracts import (
    BackendFactory,
    BackendKind,
    RoleName,
    RoleSpec,
    RuntimeKey,
)
from ..runtime.fusion.concrete.composition import (
    Qwen4ExpBackendCompositionReceipt,
    compose_qwen4_exp_backend,
)
from ..runtime.fusion.mtp import MtpPolicy
from ..runtime.fusion.qwen4_exp.media_adapters import compose_source_media_resolver
from ..runtime.fusion.qwen4_exp.request_preparation import (
    Qwen4ExpRequestPreparer,
    Qwen4ExpRequestPreparerPort,
)
from ..runtime.fusion.scheduler import SchedulerConfig
from ..runtime.manager import RuntimeManager
from ..runtime.readiness import ReadinessService
from ..runtime.role_manifest import (
    SignedRoleManifest,
    load_role_manifest,
    packaged_role_manifest_path,
)
from ..runtime.roles import RoleDirectory
from ..runtime.service import RuntimeStartService
from ..tools.hosted import HostedToolCatalog, HostedToolExecutor
from ..vision.input import MediaSourceField, MultimodalInputCapabilities
from .compaction import LocalCompactionCodec
from .controller import ResponsesController
from .operations import LocalResponsesOperations, LocalResponsesTokenCounter
from .registry import ResponseRegistry
from .runtime_mapper import CanonicalResponsesMapper
from .runtime_projection import create_runtime_projection
from .runtime_resolver import ManifestRuntimeResolver

if TYPE_CHECKING:
    from ..runtime.fusion.qwen4_exp.execution import Qwen4ExpExecutionFactoryPort
    from ..runtime.fusion.qwen4_exp.media_resolver import FileIdResolverPort


@dataclass(frozen=True, slots=True)
class RuntimeCompositionReceipt:
    """Immutable references and manifest identity for one composed runtime graph."""

    role_manifest_path: Path
    role_manifest_sha256: str
    role_manifest: SignedRoleManifest
    process_role: RoleName
    process_port: int
    public_aliases: Mapping[str, RoleName]
    role_directory: RoleDirectory
    media_source_fields: Mapping[RoleName, frozenset[MediaSourceField]]
    readiness_service: ReadinessService
    admission_controller: AdmissionController
    runtime_manager: RuntimeManager
    runtime_start_service: RuntimeStartService
    hosted_catalog: HostedToolCatalog
    anthropic_turn_source: AnthropicTurnSource
    response_registry: ResponseRegistry
    runtime_resolver: ManifestRuntimeResolver
    responses_mapper: CanonicalResponsesMapper
    responses_controller: ResponsesController
    responses_operations: LocalResponsesOperations
    build_receipt: BuildReceipt | None = None
    response_store_scope: str = "process_local"
    requires_single_worker: bool = True
    response_history_survives_restart: bool = False

    async def shutdown(self, *, deadline_s: float) -> None:
        """Drain Responses first, then close model runtimes on one deadline."""

        if deadline_s < 0:
            raise ValueError("deadline_s must be non-negative")
        loop = asyncio.get_running_loop()
        deadline_at = loop.time() + deadline_s
        clear_turn_source(self.anthropic_turn_source)
        await self.responses_controller.shutdown(timeout_s=deadline_s)
        await self.runtime_manager.shutdown(
            deadline_s=max(0.0, deadline_at - loop.time())
        )


@dataclass(frozen=True, slots=True)
class RoleRuntimeCompositionReceipt:
    """One process-local Responses graph with exactly one backend family."""

    responses: RuntimeCompositionReceipt
    qwen4_exp: Qwen4ExpBackendCompositionReceipt | None = None
    legacy_provider: LegacyPortProvider | None = None
    legacy_backend: LegacyMlxBackend | None = None
    _shutdown_task: asyncio.Task[None] | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    async def shutdown(self, *, deadline_s: float) -> None:
        """Drain protocol/runtime ownership, then close the fused registry."""

        if deadline_s < 0:
            raise ValueError("deadline_s must be non-negative")
        task = self._shutdown_task
        if task is None:
            task = asyncio.create_task(self._shutdown_once(deadline_s))
            object.__setattr__(self, "_shutdown_task", task)
        await asyncio.shield(task)

    async def _shutdown_once(self, deadline_s: float) -> None:
        loop = asyncio.get_running_loop()
        deadline_at = loop.time() + deadline_s
        try:
            await self.responses.shutdown(deadline_s=deadline_s)
        finally:
            if self.qwen4_exp is not None:
                await self.qwen4_exp.shutdown(
                    deadline_s=max(0.0, deadline_at - loop.time())
                )


def compose_responses_runtime(
    *,
    process_role: RoleName | str,
    role_manifest_path: str | Path | None = None,
    backend_factories: Mapping[BackendKind | str, BackendFactory],
    public_aliases: Mapping[str, RoleName | str] | None = None,
    build_receipt: BuildReceipt | None = None,
    hosted_tools: HostedToolCatalog | None = None,
) -> RuntimeCompositionReceipt:
    """Build one inert process-local graph from explicit trusted inputs.

    Composition reads and verifies only the supplied manifest. It does not probe
    factories, load models, start services, or mount donor protocol surfaces.
    """

    manifest = _load_manifest(role_manifest_path)
    return _compose_responses_runtime(
        manifest=manifest,
        process_role=process_role,
        backend_factories=backend_factories,
        public_aliases=public_aliases,
        build_receipt=build_receipt,
        media_source_fields=frozenset(),
        hosted_tools=hosted_tools,
    )


def _load_manifest(role_manifest_path: str | Path | None) -> SignedRoleManifest:
    return load_role_manifest(
        packaged_role_manifest_path()
        if role_manifest_path is None
        else role_manifest_path
    )


def _compose_responses_runtime(
    *,
    manifest: SignedRoleManifest,
    process_role: RoleName | str,
    backend_factories: Mapping[BackendKind | str, BackendFactory],
    public_aliases: Mapping[str, RoleName | str] | None,
    build_receipt: BuildReceipt | None,
    media_source_fields: frozenset[MediaSourceField],
    hosted_tools: HostedToolCatalog | None = None,
) -> RuntimeCompositionReceipt:
    topology = manifest.role_directory()
    process_spec = topology.resolve(process_role)
    roles = RoleDirectory((process_spec,))
    factories = _validated_backend_factories((process_spec,), backend_factories)
    aliases = (
        {process_spec.requested_model: process_spec.name}
        if public_aliases is None
        else public_aliases
    )
    receipt_fields = dict(manifest.build_receipt_fields())
    if build_receipt is not None:
        if not isinstance(build_receipt, BuildReceipt):
            raise TypeError("build_receipt must be a BuildReceipt")
        if build_receipt.role_manifest_sha256 != manifest.role_manifest_sha256:
            raise ValueError("build receipt role manifest digest does not match")
        receipt_fields.update(build_receipt.health_fields())
    readiness = ReadinessService(roles, receipt=receipt_fields)
    admission = AdmissionController()
    manager = RuntimeManager(
        factories,
        roles=roles,
        readiness=readiness,
        admission=admission,
    )
    inner_starter = RuntimeStartService(manager, default_role=process_spec.name)
    catalog = HostedToolCatalog() if hosted_tools is None else hosted_tools
    # One immutable catalog/executor/starter owner handed to both protocol
    # paths; with an empty catalog the starter is a transparent pass-through.
    starter = HostedAgenticRuntimeStarter(
        inner_starter,
        catalog=catalog,
        executor=HostedToolExecutor(catalog),
    )
    registry = ResponseRegistry()
    resolver = ManifestRuntimeResolver(roles, aliases)
    anthropic_turn_source = RuntimeAnthropicTurnSource(
        starter=starter,
        resolve_model=lambda model: _resolve_anthropic_model(resolver, model),
    )
    register_turn_source(anthropic_turn_source)
    compaction_codec = LocalCompactionCodec()
    mapper = CanonicalResponsesMapper(
        resolve_runtime=resolver,
        projection_factory=create_runtime_projection,
        compaction_codec=compaction_codec,
    )
    controller = ResponsesController(
        registry=registry,
        mapper=mapper,
        starter=starter,
    )
    token_counter = LocalResponsesTokenCounter(
        {
            spec.requested_model: spec.model_dir
            for spec in roles.specs()
            if spec.model_dir is not None
        }
    )
    operations = LocalResponsesOperations(
        controller=controller,
        compaction_codec=compaction_codec,
        token_counter=token_counter,
    )

    source_path = manifest.source.path
    if source_path is None:
        raise RuntimeError("loaded role manifest is missing its explicit source path")
    return RuntimeCompositionReceipt(
        role_manifest_path=source_path,
        role_manifest_sha256=manifest.role_manifest_sha256,
        role_manifest=manifest,
        process_role=process_spec.name,
        process_port=process_spec.port,
        public_aliases=resolver.aliases,
        role_directory=roles,
        media_source_fields=MappingProxyType(
            {process_spec.name: frozenset(media_source_fields)}
        ),
        readiness_service=readiness,
        admission_controller=admission,
        runtime_manager=manager,
        runtime_start_service=starter,
        hosted_catalog=catalog,
        anthropic_turn_source=anthropic_turn_source,
        response_registry=registry,
        runtime_resolver=resolver,
        responses_mapper=mapper,
        responses_controller=controller,
        responses_operations=operations,
        build_receipt=build_receipt,
    )


def _resolve_anthropic_model(
    resolver: ManifestRuntimeResolver,
    model: str,
) -> tuple[RuntimeKey, str]:
    resolved = resolver(
        model=model,
        role=None,
        revision=None,
        adapter_path=None,
        draft_model_id=None,
        backend=None,
    )
    return resolved.runtime, resolved.role


def compose_role_responses_runtime(
    *,
    process_role: RoleName | str,
    role_manifest_path: str | Path | None = None,
    public_aliases: Mapping[str, RoleName | str] | None = None,
    request_preparer: Qwen4ExpRequestPreparerPort | None = None,
    legacy_provider: LegacyPortProvider | None = None,
    scheduler_config: SchedulerConfig | None = None,
    mtp_policy: MtpPolicy | None = None,
    fused_capacity: int = 2,
    execution_factory: Qwen4ExpExecutionFactoryPort | None = None,
    allowed_url_origins: Iterable[str] = (),
    file_id_resolver: FileIdResolverPort | None = None,
    build_receipt: BuildReceipt | None = None,
    hosted_tools: HostedToolCatalog | None = None,
) -> RoleRuntimeCompositionReceipt:
    """Compose exactly the backend family selected by one manifest role.

    Composition remains inert: no checkpoint, wrapper, tensor, service, URL,
    file, or model is opened until the selected backend is acquired.
    """

    manifest = _load_manifest(role_manifest_path)
    process_spec = manifest.role_directory().resolve(process_role)
    qwen4_exp: Qwen4ExpBackendCompositionReceipt | None = None
    resolved_legacy_provider: LegacyPortProvider | None = None
    legacy_backend: LegacyMlxBackend | None = None
    media_source_fields: frozenset[MediaSourceField] = frozenset()

    if process_spec.backend is BackendKind.FUSED_MTP_MLX:
        if legacy_provider is not None:
            raise ValueError("a fused process cannot register a legacy provider")
        origins = tuple(allowed_url_origins)
        if request_preparer is None:
            media_capabilities = _default_qwen4_exp_capabilities(
                allowed_url_origins=origins,
                file_id_resolver=file_id_resolver,
            )
            resolved_preparer = _default_qwen4_exp_preparer(
                allowed_url_origins=origins,
                file_id_resolver=file_id_resolver,
                capabilities=media_capabilities,
            )
            media_source_fields = media_capabilities.accepted_sources
        else:
            # An injected preparer is an opaque port. Without a separately
            # trusted composition receipt, inspecting its private resolver or
            # planner would invent capability truth, so it stays text-only.
            resolved_preparer = request_preparer
        if not isinstance(resolved_preparer, Qwen4ExpRequestPreparerPort):
            raise TypeError("request_preparer must satisfy Qwen4ExpRequestPreparerPort")
        qwen4_exp = compose_qwen4_exp_backend(
            request_preparer=resolved_preparer,
            scheduler_config=scheduler_config or SchedulerConfig(),
            mtp_policy=mtp_policy or MtpPolicy(),
            capacity=fused_capacity,
            execution_factory=execution_factory,
        )
        factories: Mapping[BackendKind, BackendFactory] = {
            BackendKind.FUSED_MTP_MLX: qwen4_exp.backend,
        }
    else:
        if request_preparer is not None or execution_factory is not None:
            raise ValueError("a legacy process cannot register fused runtime ports")
        if scheduler_config is not None or mtp_policy is not None:
            raise ValueError("a legacy process cannot register fused runtime policy")
        if tuple(allowed_url_origins) or file_id_resolver is not None:
            raise ValueError("a legacy process cannot register fused media policy")
        resolved_legacy_provider = (
            CachedLegacyPortProvider() if legacy_provider is None else legacy_provider
        )
        if not isinstance(resolved_legacy_provider, LegacyPortProvider):
            raise TypeError("legacy_provider must satisfy LegacyPortProvider")
        legacy_backend = LegacyMlxBackend(resolved_legacy_provider)
        factories = {BackendKind.LEGACY_MLX: legacy_backend}

    responses = _compose_responses_runtime(
        manifest=manifest,
        process_role=process_spec.name,
        backend_factories=factories,
        public_aliases=public_aliases,
        build_receipt=build_receipt,
        media_source_fields=media_source_fields,
        hosted_tools=hosted_tools,
    )
    return RoleRuntimeCompositionReceipt(
        responses=responses,
        qwen4_exp=qwen4_exp,
        legacy_provider=resolved_legacy_provider,
        legacy_backend=legacy_backend,
    )


def _default_qwen4_exp_preparer(
    *,
    allowed_url_origins: tuple[str, ...],
    file_id_resolver: FileIdResolverPort | None,
    capabilities: MultimodalInputCapabilities,
) -> Qwen4ExpRequestPreparer:
    resolver = compose_source_media_resolver(
        allowed_url_origins=allowed_url_origins,
        file_id_resolver=file_id_resolver,
    )
    return Qwen4ExpRequestPreparer(
        resolver=resolver,
        capabilities=capabilities,
    )


def _default_qwen4_exp_capabilities(
    *,
    allowed_url_origins: tuple[str, ...],
    file_id_resolver: FileIdResolverPort | None,
) -> MultimodalInputCapabilities:
    """Derive the default preparer's exact source contract from trusted inputs."""

    accepted_sources = {
        MediaSourceField.IMAGE_URL,
        MediaSourceField.IMAGE_BASE64,
        MediaSourceField.FILE_DATA,
    }
    if allowed_url_origins:
        accepted_sources.add(MediaSourceField.FILE_URL)
    if file_id_resolver is not None:
        accepted_sources.add(MediaSourceField.FILE_ID)
    return MultimodalInputCapabilities(accepted_sources=frozenset(accepted_sources))


def _validated_backend_factories(
    specs: Iterable[RoleSpec],
    backend_factories: Mapping[BackendKind | str, BackendFactory],
) -> Mapping[BackendKind, BackendFactory]:
    if not isinstance(backend_factories, Mapping) or not backend_factories:
        raise ValueError("backend_factories must be a non-empty mapping")

    normalized: dict[BackendKind, BackendFactory] = {}
    for raw_kind, factory in backend_factories.items():
        kind = raw_kind if isinstance(raw_kind, BackendKind) else BackendKind(raw_kind)
        if kind in normalized:
            raise ValueError(f"duplicate backend factory for {kind.value!r}")
        if not isinstance(factory, BackendFactory):
            raise TypeError(
                f"backend factory for {kind.value!r} does not satisfy BackendFactory"
            )
        normalized[kind] = factory

    required = {spec.backend for spec in specs}
    missing = sorted(kind.value for kind in required - normalized.keys())
    if missing:
        raise ValueError(
            "backend_factories is missing manifest backend(s): " + ", ".join(missing)
        )
    unexpected = sorted(kind.value for kind in normalized.keys() - required)
    if unexpected:
        raise ValueError(
            "backend_factories contains backend(s) outside the process role: "
            + ", ".join(unexpected)
        )
    return normalized


__all__ = [
    "RoleRuntimeCompositionReceipt",
    "RuntimeCompositionReceipt",
    "compose_responses_runtime",
    "compose_role_responses_runtime",
]
