"""RED contracts for explicit, inert Responses runtime composition."""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from mlx_batch_server.chat.anthropic.runtime_source import RuntimeAnthropicTurnSource
from mlx_batch_server.chat.anthropic.turn_source import (
    clear_turn_source,
    current_turn_source,
)
from mlx_batch_server.provenance import BuildReceipt
from mlx_batch_server.responses import runtime_bootstrap
from mlx_batch_server.responses.controller import ResponsesController
from mlx_batch_server.responses.registry import ResponseRegistry
from mlx_batch_server.responses.runtime_bootstrap import (
    RoleRuntimeCompositionReceipt,
    compose_responses_runtime,
    compose_role_responses_runtime,
)
from mlx_batch_server.responses.runtime_mapper import CanonicalResponsesMapper
from mlx_batch_server.responses.runtime_projection import create_runtime_projection
from mlx_batch_server.responses.runtime_resolver import ManifestRuntimeResolver
from mlx_batch_server.runtime.admission import AdmissionController
from mlx_batch_server.runtime.agentic import HostedAgenticRuntimeStarter
from mlx_batch_server.runtime.backends.legacy_mlx import LegacyMlxBackend
from mlx_batch_server.runtime.contracts import BackendKind, RoleName
from mlx_batch_server.runtime.fusion.mtp import MtpPolicy
from mlx_batch_server.runtime.fusion.scheduler import SchedulerConfig
from mlx_batch_server.runtime.manager import RuntimeManager
from mlx_batch_server.runtime.readiness import ReadinessService
from mlx_batch_server.runtime.role_manifest import packaged_role_manifest_path
from mlx_batch_server.runtime.roles import RoleDirectory
from mlx_batch_server.runtime.service import RuntimeStartService
from mlx_batch_server.vision.input import MediaSourceField

ROOT = Path(__file__).resolve().parents[2]
ROLE_MANIFEST = packaged_role_manifest_path()


@pytest.fixture(autouse=True)
def _reset_anthropic_turn_source():
    clear_turn_source()
    yield
    clear_turn_source()


class _DormantBackendFactory:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def probe(self, model: object) -> Any:
        self.calls.append(("probe", model))
        raise AssertionError("composition must not probe backends")

    async def load(self, runtime: object, config: object) -> Any:
        self.calls.append(("load", (runtime, config)))
        raise AssertionError("composition must not load backends")


class _DormantRequestPreparer:
    def __init__(self) -> None:
        self.calls: list[tuple[object, object]] = []

    async def prepare(self, request: object, cancel: object) -> Any:
        self.calls.append((request, cancel))
        raise AssertionError("composition must not prepare requests")


class _DormantExecutionFactory:
    def __init__(self) -> None:
        self.calls: list[tuple[object, object, object]] = []

    def prepare(
        self,
        runtime: object,
        config: object,
        scheduler_config: object,
    ) -> Any:
        self.calls.append((runtime, config, scheduler_config))
        raise AssertionError("composition must not prepare checkpoint loads")


class _DormantLegacyProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def probe(self, model: object) -> Any:
        self.calls.append(("probe", model))
        raise AssertionError("composition must not probe legacy models")

    async def acquire(self, runtime: object, config: object) -> Any:
        self.calls.append(("acquire", (runtime, config)))
        raise AssertionError("composition must not acquire legacy models")


class _DormantFileIdResolver:
    async def resolve(self, request: object) -> Any:
        raise AssertionError("composition must not resolve file identities")


_FactoryBundle = tuple[
    dict[BackendKind, _DormantBackendFactory],
    _DormantBackendFactory,
    _DormantBackendFactory,
]


def _factories() -> _FactoryBundle:
    fused = _DormantBackendFactory()
    legacy = _DormantBackendFactory()
    return (
        {
            BackendKind.FUSED_MTP_MLX: fused,
            BackendKind.LEGACY_MLX: legacy,
        },
        fused,
        legacy,
    )


def test_composition_wires_one_graph_from_the_explicit_signed_manifest() -> None:
    _, fused, legacy = _factories()

    receipt = compose_responses_runtime(
        process_role=RoleName.MAIN,
        role_manifest_path=ROLE_MANIFEST,
        backend_factories={BackendKind.FUSED_MTP_MLX: fused},
        public_aliases={"buddy": RoleName.MAIN},
    )

    assert receipt.role_manifest_path == ROLE_MANIFEST.resolve()
    assert receipt.role_manifest_sha256 == (
        "e9630032e90ff2fdf8b17b779e0276ef326afd39fd97889e023727be4ef0176a"
    )
    assert receipt.role_manifest_sha256 == receipt.role_manifest.role_manifest_sha256
    assert receipt.process_role is RoleName.MAIN
    assert receipt.process_port == 8100
    assert tuple(receipt.role_directory.specs()) == (
        receipt.role_manifest.role_directory().resolve(RoleName.MAIN),
    )
    assert receipt.public_aliases == {
        "buddy": RoleName.MAIN,
    }

    assert isinstance(receipt.role_directory, RoleDirectory)
    assert isinstance(receipt.readiness_service, ReadinessService)
    assert isinstance(receipt.admission_controller, AdmissionController)
    assert isinstance(receipt.runtime_manager, RuntimeManager)
    assert isinstance(receipt.runtime_start_service, RuntimeStartService)
    assert isinstance(receipt.anthropic_turn_source, RuntimeAnthropicTurnSource)
    assert current_turn_source() is receipt.anthropic_turn_source
    assert isinstance(receipt.response_registry, ResponseRegistry)
    assert isinstance(receipt.runtime_resolver, ManifestRuntimeResolver)
    assert isinstance(receipt.responses_mapper, CanonicalResponsesMapper)
    assert isinstance(receipt.responses_controller, ResponsesController)

    assert receipt.readiness_service.roles is receipt.role_directory
    assert receipt.runtime_manager._roles is receipt.role_directory
    assert receipt.runtime_manager._readiness is receipt.readiness_service
    assert receipt.runtime_manager._admission is receipt.admission_controller
    assert isinstance(receipt.runtime_start_service, HostedAgenticRuntimeStarter)
    assert receipt.runtime_start_service.inner._manager is receipt.runtime_manager
    assert receipt.anthropic_turn_source._starter is receipt.runtime_start_service
    assert receipt.runtime_resolver._roles is receipt.role_directory
    assert receipt.responses_mapper._resolve_runtime is receipt.runtime_resolver
    assert receipt.responses_mapper._projection_factory is create_runtime_projection
    assert receipt.responses_controller._registry is receipt.response_registry
    assert receipt.responses_controller._mapper is receipt.responses_mapper
    assert receipt.responses_controller._starter is receipt.runtime_start_service
    assert receipt.response_store_scope == "process_local"
    assert receipt.requires_single_worker is True
    assert receipt.response_history_survives_restart is False
    assert fused.calls == []
    assert legacy.calls == []

    with pytest.raises(FrozenInstanceError):
        receipt.role_manifest_sha256 = "0" * 64  # type: ignore[misc]
    with pytest.raises(TypeError):
        receipt.public_aliases["buddy"] = RoleName.VISION  # type: ignore[index]


def test_composition_uses_the_packaged_manifest_by_default() -> None:
    factories, _, _ = _factories()

    receipt = compose_responses_runtime(
        process_role=RoleName.MAIN,
        backend_factories={
            BackendKind.FUSED_MTP_MLX: factories[BackendKind.FUSED_MTP_MLX]
        },
        public_aliases={"buddy": RoleName.MAIN},
    )

    assert receipt.role_manifest_path == packaged_role_manifest_path()


def test_build_receipt_is_bound_to_the_same_readiness_graph() -> None:
    factories, _, _ = _factories()
    build_receipt = BuildReceipt(
        target_sha="a" * 40,
        target_version="0.6-test",
        source_dirty=True,
        omlx_sha="b" * 40,
        mtplx_sha="c" * 40,
        source_origins_sha256="d" * 64,
        dependency_lock_sha256="e" * 64,
        role_manifest_sha256=(
            "e9630032e90ff2fdf8b17b779e0276ef326afd39fd97889e023727be4ef0176a"
        ),
    )

    receipt = compose_responses_runtime(
        process_role=RoleName.MAIN,
        backend_factories={
            BackendKind.FUSED_MTP_MLX: factories[BackendKind.FUSED_MTP_MLX]
        },
        build_receipt=build_receipt,
    )

    assert receipt.build_receipt is build_receipt
    readiness_receipt = receipt.readiness_service.snapshot(RoleName.MAIN).receipt
    assert readiness_receipt is not None
    assert readiness_receipt["dependency_lock_sha256"] == "e" * 64
    assert readiness_receipt["role_manifest_sha256"] == (
        "e9630032e90ff2fdf8b17b779e0276ef326afd39fd97889e023727be4ef0176a"
    )


def test_role_composition_places_only_the_selected_backend_behind_manager() -> None:
    request_preparer = _DormantRequestPreparer()
    execution_factory = _DormantExecutionFactory()
    scheduler = SchedulerConfig()
    mtp_policy = MtpPolicy()

    receipt = compose_role_responses_runtime(
        process_role=RoleName.MAIN,
        role_manifest_path=ROLE_MANIFEST,
        public_aliases={"buddy": RoleName.MAIN},
        request_preparer=request_preparer,
        scheduler_config=scheduler,
        mtp_policy=mtp_policy,
        fused_capacity=2,
        execution_factory=execution_factory,
    )

    assert isinstance(receipt, RoleRuntimeCompositionReceipt)
    assert receipt.responses.process_role is RoleName.MAIN
    assert receipt.qwen4_exp is not None
    assert receipt.qwen4_exp.request_preparer is request_preparer
    assert receipt.qwen4_exp.execution_factory is execution_factory
    assert receipt.qwen4_exp.scheduler_config is scheduler
    assert receipt.qwen4_exp.mtp_policy is mtp_policy
    assert receipt.legacy_provider is None
    assert receipt.legacy_backend is None
    assert receipt.responses.runtime_manager._factories == {
        BackendKind.FUSED_MTP_MLX: receipt.qwen4_exp.backend,
    }
    assert receipt.responses.responses_controller._starter is (
        receipt.responses.runtime_start_service
    )
    assert request_preparer.calls == []
    assert execution_factory.calls == []


def test_default_fused_composition_publishes_exact_inert_media_sources() -> None:
    execution_factory = _DormantExecutionFactory()

    receipt = compose_role_responses_runtime(
        process_role=RoleName.MAIN,
        role_manifest_path=ROLE_MANIFEST,
        execution_factory=execution_factory,
    )

    assert receipt.responses.media_source_fields == {
        RoleName.MAIN: frozenset(
            {
                MediaSourceField.IMAGE_URL,
                MediaSourceField.IMAGE_BASE64,
                MediaSourceField.FILE_DATA,
            }
        )
    }
    assert execution_factory.calls == []
    with pytest.raises(TypeError):
        receipt.responses.media_source_fields[RoleName.MAIN] = frozenset()  # type: ignore[index]


def test_fused_media_receipt_comes_only_from_canonical_composition_inputs() -> None:
    resolver = _DormantFileIdResolver()
    receipt = compose_role_responses_runtime(
        process_role=RoleName.MAIN,
        role_manifest_path=ROLE_MANIFEST,
        allowed_url_origins=("https://media.example",),
        file_id_resolver=resolver,
        execution_factory=_DormantExecutionFactory(),
    )

    assert receipt.responses.media_source_fields[RoleName.MAIN] == frozenset(
        {
            MediaSourceField.IMAGE_URL,
            MediaSourceField.IMAGE_BASE64,
            MediaSourceField.FILE_DATA,
            MediaSourceField.FILE_URL,
            MediaSourceField.FILE_ID,
        }
    )


def test_injected_preparer_and_generic_composition_fail_closed_to_text_only() -> None:
    injected = compose_role_responses_runtime(
        process_role=RoleName.MAIN,
        role_manifest_path=ROLE_MANIFEST,
        request_preparer=_DormantRequestPreparer(),
        execution_factory=_DormantExecutionFactory(),
    )
    generic = compose_responses_runtime(
        process_role=RoleName.MAIN,
        role_manifest_path=ROLE_MANIFEST,
        backend_factories={BackendKind.FUSED_MTP_MLX: _DormantBackendFactory()},
    )

    assert injected.responses.media_source_fields == {RoleName.MAIN: frozenset()}
    assert generic.media_source_fields == {RoleName.MAIN: frozenset()}


def test_legacy_role_composition_does_not_construct_the_fused_graph() -> None:
    legacy_provider = _DormantLegacyProvider()

    receipt = compose_role_responses_runtime(
        process_role=RoleName.VISION,
        role_manifest_path=ROLE_MANIFEST,
        legacy_provider=legacy_provider,
    )

    assert receipt.responses.process_role is RoleName.VISION
    assert receipt.responses.process_port == 8102
    assert receipt.qwen4_exp is None
    assert receipt.legacy_provider is legacy_provider
    assert isinstance(receipt.legacy_backend, LegacyMlxBackend)
    assert receipt.responses.runtime_manager._factories == {
        BackendKind.LEGACY_MLX: receipt.legacy_backend,
    }
    assert receipt.responses.media_source_fields == {RoleName.VISION: frozenset()}


@pytest.mark.asyncio
async def test_role_composition_shutdown_closes_both_owned_graphs_once() -> None:
    receipt = compose_role_responses_runtime(
        process_role=RoleName.MAIN,
        role_manifest_path=ROLE_MANIFEST,
        public_aliases={"buddy": RoleName.MAIN},
        request_preparer=_DormantRequestPreparer(),
        execution_factory=_DormantExecutionFactory(),
    )

    await receipt.shutdown(deadline_s=1.0)
    await receipt.shutdown(deadline_s=0.0)

    assert receipt.responses.response_registry.stats()["shutting_down"] is True
    assert receipt.responses.runtime_manager._closed is True
    assert receipt.qwen4_exp is not None
    assert receipt.qwen4_exp.registry._closed is True


def test_composition_never_falls_back_from_a_missing_explicit_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factories, _, _ = _factories()
    monkeypatch.setenv("MLX_BATCH_ROLE_MANIFEST", str(ROLE_MANIFEST))

    with pytest.raises(FileNotFoundError):
        compose_responses_runtime(
            process_role=RoleName.MAIN,
            role_manifest_path=tmp_path / "missing-role-manifest.json",
            backend_factories={
                BackendKind.FUSED_MTP_MLX: factories[BackendKind.FUSED_MTP_MLX]
            },
            public_aliases={"buddy": RoleName.MAIN},
        )


def test_bootstrap_has_no_runtime_singleton_or_tensor_donor_import() -> None:
    component_types = (
        RoleDirectory,
        ReadinessService,
        AdmissionController,
        RuntimeManager,
        RuntimeStartService,
        ResponseRegistry,
        ManifestRuntimeResolver,
        CanonicalResponsesMapper,
        ResponsesController,
    )
    assert not any(
        isinstance(value, component_types) for value in vars(runtime_bootstrap).values()
    )

    source = Path(runtime_bootstrap.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_modules.update(
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    )
    forbidden_roots = ("mlx", "mlx_lm", "mlx_vlm", "omlx", "mtplx")
    assert not any(
        module == root or module.startswith(f"{root}.")
        for module in imported_modules
        for root in forbidden_roots
    )


def test_composition_rejects_backends_outside_the_process_role() -> None:
    with pytest.raises(ValueError, match="outside the process role"):
        compose_responses_runtime(
            process_role=RoleName.MAIN,
            role_manifest_path=ROLE_MANIFEST,
            backend_factories={
                BackendKind.FUSED_MTP_MLX: _DormantBackendFactory(),
                BackendKind.LEGACY_MLX: _DormantBackendFactory(),
            },
            public_aliases={"buddy": RoleName.MAIN},
        )


def test_process_role_rejects_an_alias_owned_by_another_process() -> None:
    with pytest.raises(ValueError, match="unknown role"):
        compose_responses_runtime(
            process_role=RoleName.MAIN,
            role_manifest_path=ROLE_MANIFEST,
            backend_factories={BackendKind.FUSED_MTP_MLX: _DormantBackendFactory()},
            public_aliases={"vision": RoleName.VISION},
        )


def test_process_role_defaults_to_its_exact_manifest_model_alias() -> None:
    receipt = compose_responses_runtime(
        process_role=RoleName.CANARY,
        role_manifest_path=ROLE_MANIFEST,
        backend_factories={BackendKind.FUSED_MTP_MLX: _DormantBackendFactory()},
    )

    spec = receipt.role_directory.resolve(RoleName.CANARY)
    assert receipt.public_aliases == {spec.requested_model.casefold(): RoleName.CANARY}


@pytest.mark.asyncio
async def test_composition_receipt_closes_controller_then_runtime_once() -> None:
    factories, _, _ = _factories()
    receipt = compose_responses_runtime(
        process_role=RoleName.MAIN,
        role_manifest_path=ROLE_MANIFEST,
        backend_factories={
            BackendKind.FUSED_MTP_MLX: factories[BackendKind.FUSED_MTP_MLX]
        },
        public_aliases={"buddy": RoleName.MAIN},
    )

    await receipt.shutdown(deadline_s=1.0)
    await receipt.shutdown(deadline_s=0.0)

    assert receipt.response_registry.stats()["shutting_down"] is True
    with pytest.raises(RuntimeError, match="shutting down"):
        await receipt.responses_controller.create(
            {"model": "buddy", "input": "late"},
            owner_id="principal:test",
        )


def test_composition_hands_one_hosted_owner_to_both_protocol_paths() -> None:
    """HAD-4: one catalog/executor/starter instance, no second registry."""

    from mlx_batch_server.tools.hosted import HostedToolCatalog
    from mlx_batch_server.tools.hosted_web import HostedWebSearchTool

    catalog = HostedToolCatalog((HostedWebSearchTool(provider=None),))
    receipt = compose_responses_runtime(
        process_role=RoleName.MAIN,
        role_manifest_path=ROLE_MANIFEST,
        backend_factories={BackendKind.FUSED_MTP_MLX: _DormantBackendFactory()},
        hosted_tools=catalog,
    )
    try:
        starter = receipt.runtime_start_service
        assert isinstance(starter, HostedAgenticRuntimeStarter)
        assert receipt.hosted_catalog is catalog
        assert starter.hosted_catalog is catalog
        # Both protocol paths hold the same owner object: no HTTP loopback,
        # no second scheduler.
        assert receipt.responses_controller._starter is starter
        assert receipt.anthropic_turn_source._starter is starter
    finally:
        clear_turn_source()


def test_default_composition_has_an_empty_hosted_catalog() -> None:
    """Capability truth unchanged: no hosted claim flips in this cut."""

    receipt = compose_responses_runtime(
        process_role=RoleName.MAIN,
        role_manifest_path=ROLE_MANIFEST,
        backend_factories={BackendKind.FUSED_MTP_MLX: _DormantBackendFactory()},
    )
    try:
        assert not receipt.hosted_catalog
        assert isinstance(receipt.runtime_start_service, HostedAgenticRuntimeStarter)
    finally:
        clear_turn_source()
