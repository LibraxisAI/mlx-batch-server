"""RED contracts for atomic application mounting of the Responses runtime."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from mlx_batch_server import main as main_module
from mlx_batch_server import provenance
from mlx_batch_server.core import config as core_config
from mlx_batch_server.main import (
    _compose_process_runtime,
    build_parser,
    create_app,
)
from mlx_batch_server.responses import runtime_bootstrap
from mlx_batch_server.responses.runtime_resolver import ManifestRuntimeResolver
from mlx_batch_server.runtime.contracts import BackendKind, RoleName, RoleSpec
from mlx_batch_server.runtime.readiness import ReadinessService
from mlx_batch_server.runtime.role_manifest import packaged_role_manifest_path
from mlx_batch_server.runtime.roles import RoleDirectory

ROOT = Path(__file__).resolve().parents[2]


def _expanded_routes(routes: list[Any]) -> list[Any]:
    expanded: list[Any] = []
    for route in routes:
        original_router = getattr(route, "original_router", None)
        if original_router is None:
            expanded.append(route)
        else:
            expanded.extend(_expanded_routes(original_router.routes))
    return expanded


class _ApplicationManager:
    def __init__(self) -> None:
        self.acquired: list[RoleName] = []

    async def acquire_role(self, role: RoleName) -> object:
        self.acquired.append(role)
        return object()


class _ApplicationRuntime:
    def __init__(self, *, nested: bool) -> None:
        roles = RoleDirectory(
            (
                RoleSpec(
                    name=RoleName.MAIN,
                    port=8100,
                    requested_model="test/flash",
                    backend=BackendKind.FUSED_MTP_MLX,
                    pinned=True,
                ),
            )
        )
        manager = _ApplicationManager()
        route_runtime = SimpleNamespace(
            responses_controller=object(),
            response_registry=object(),
            requires_single_worker=True,
            process_role=RoleName.MAIN,
            process_port=8100,
            role_manifest_sha256="manifest-sha",
            role_directory=roles,
            readiness_service=ReadinessService(roles),
            runtime_manager=manager,
            runtime_resolver=ManifestRuntimeResolver(
                roles,
                {"test/flash": RoleName.MAIN},
            ),
        )
        if nested:
            self.responses = route_runtime
        else:
            self.responses_controller = route_runtime.responses_controller
            self.response_registry = route_runtime.response_registry
            self.requires_single_worker = True
            self.process_role = route_runtime.process_role
            self.process_port = route_runtime.process_port
            self.role_manifest_sha256 = route_runtime.role_manifest_sha256
            self.role_directory = route_runtime.role_directory
            self.readiness_service = route_runtime.readiness_service
            self.runtime_manager = route_runtime.runtime_manager
            self.runtime_resolver = route_runtime.runtime_resolver
        self.shutdown_deadlines: list[float] = []
        self.manager = manager

    async def shutdown(self, *, deadline_s: float) -> None:
        self.shutdown_deadlines.append(deadline_s)


@pytest.mark.parametrize("nested", [False, True])
def test_runtime_receipt_atomically_replaces_legacy_responses_router(
    nested: bool,
) -> None:
    runtime = _ApplicationRuntime(nested=nested)
    app = create_app(
        responses_runtime=runtime,
        responses_shutdown_timeout_s=7.5,
    )
    routes = _expanded_routes(app.routes)

    response_routes = [
        route for route in routes if getattr(route, "path", None) == "/v1/responses"
    ]
    endpoint_modules = {
        route.endpoint.__module__
        for route in response_routes
        if hasattr(route, "endpoint")
    }

    assert len(response_routes) == 2
    assert endpoint_modules == {"mlx_batch_server.responses.runtime_router"}
    assert app.state.responses_runtime is runtime
    expected_route_runtime = getattr(runtime, "responses", runtime)
    assert app.state.responses_route_runtime is expected_route_runtime
    assert app.state.role_control_service.role is RoleName.MAIN

    model_control_modules = {
        route.endpoint.__module__
        for route in routes
        if getattr(route, "path", None)
        in {"/health", "/v1/models/load", "/v1/models/unload"}
    }
    assert model_control_modules == {"mlx_batch_server.responses.runtime_control"}
    canonical_paths = {getattr(route, "path", None) for route in routes}
    assert "/v1/chat/completions" not in canonical_paths
    assert "/v1/audio/transcriptions" not in canonical_paths
    assert "/v1/images/generations" not in canonical_paths
    assert "/v1/embeddings" not in canonical_paths
    assert str(app.url_path_for("create_message")) == "/anthropic/v1/messages"

    with TestClient(app):
        assert runtime.manager.acquired == [RoleName.MAIN]

    assert runtime.shutdown_deadlines == [7.5]


def test_invalid_runtime_fails_before_application_mount() -> None:
    runtime = SimpleNamespace(shutdown=lambda **_: None)

    with pytest.raises(TypeError, match="controller/registry"):
        create_app(responses_runtime=runtime)


def test_process_local_runtime_rejects_multiple_workers() -> None:
    runtime = _ApplicationRuntime(nested=True)

    with pytest.raises(RuntimeError, match="exactly one worker"):
        create_app(responses_runtime=runtime, worker_count=2)

    assert runtime.shutdown_deadlines == []


def test_production_ports_require_an_explicit_process_role() -> None:
    for port in (8100, 8101, 8102):
        with pytest.raises(RuntimeError, match="requires an explicit runtime role"):
            _compose_process_runtime(runtime_role=None, port=port)

    assert _compose_process_runtime(runtime_role=None, port=10240) is None


def test_process_role_must_own_the_bound_port() -> None:
    with pytest.raises(RuntimeError, match="owns port 8100, not 8101"):
        _compose_process_runtime(runtime_role="main", port=8101)


def test_process_role_composes_the_exact_manifest_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    catalog_calls: list[dict[str, Any]] = []
    settings_calls: list[None] = []
    sentinel = object()
    build_receipt = object()
    hosted_catalog = object()

    class _Secret:
        def get_secret_value(self) -> str:
            return "brave-test-key"

    def get_settings() -> object:
        settings_calls.append(None)
        return SimpleNamespace(brave_api_key=_Secret())

    def compose_catalog(**kwargs: Any) -> object:
        catalog_calls.append(kwargs)
        return hosted_catalog

    def compose(**kwargs: Any) -> object:
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(core_config, "get_settings", get_settings)
    monkeypatch.setattr(
        runtime_bootstrap,
        "compose_production_hosted_catalog",
        compose_catalog,
    )
    monkeypatch.setattr(runtime_bootstrap, "compose_role_responses_runtime", compose)
    monkeypatch.setattr(
        provenance,
        "compose_source_build_receipt",
        lambda **_kwargs: build_receipt,
    )

    result = _compose_process_runtime(
        runtime_role="main",
        port=8100,
        media_url_origins=("https://media.example",),
    )

    assert result is sentinel
    assert settings_calls == [None]
    assert catalog_calls == [{"brave_api_key": "brave-test-key"}]
    assert calls == [
        {
            "process_role": RoleName.MAIN,
            "role_manifest_path": packaged_role_manifest_path(),
            "allowed_url_origins": ("https://media.example",),
            "build_receipt": build_receipt,
            "hosted_tools": hosted_catalog,
        }
    ]


def test_lazy_application_path_delegates_role_composition_to_the_one_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    runtime = object()
    application = object()

    def compose(**kwargs: Any) -> object:
        calls.append(kwargs)
        return runtime

    monkeypatch.setenv("MLX_BATCH_RUNTIME_ROLE", "main")
    monkeypatch.setenv("MLX_BATCH_PORT", "8100")
    monkeypatch.setenv(
        "MLX_BATCH_MEDIA_URL_ORIGINS",
        "https://media.example, https://images.example",
    )
    monkeypatch.setattr(main_module, "_compose_process_runtime", compose)
    monkeypatch.setattr(
        main_module,
        "create_app",
        lambda **kwargs: (
            application if kwargs == {"responses_runtime": runtime} else None
        ),
    )
    monkeypatch.setattr(main_module, "_app_instance", None)

    assert main_module._get_app() is application
    assert calls == [
        {
            "runtime_role": "main",
            "port": 8100,
            "media_url_origins": (
                "https://media.example",
                "https://images.example",
            ),
        }
    ]


def test_cli_exposes_explicit_role_and_repeatable_media_origins() -> None:
    args = build_parser().parse_args(
        [
            "--port",
            "8100",
            "--runtime-role",
            "main",
            "--media-url-origin",
            "https://one.example",
            "--media-url-origin",
            "https://two.example",
        ]
    )

    assert args.runtime_role == "main"
    assert args.media_url_origin == [
        "https://one.example",
        "https://two.example",
    ]


def test_canonical_import_path_has_no_eager_legacy_graph() -> None:
    package_tree = ast.parse(
        (ROOT / "src/mlx_batch_server/responses/__init__.py").read_text()
    )
    routers_tree = ast.parse((ROOT / "src/mlx_batch_server/routers.py").read_text())

    eager_package_imports = [
        node
        for node in ast.walk(package_tree)
        if isinstance(node, ast.ImportFrom)
        and node.module in {"adapter", "context_builder", "router", "store"}
    ]
    eager_router_singletons = [
        node
        for node in routers_tree.body
        if isinstance(node, ast.Assign | ast.AnnAssign)
        and _assigned_names(node) == {"api_router"}
    ]

    assert eager_package_imports == []
    assert eager_router_singletons == []


def _assigned_names(node: ast.Assign | ast.AnnAssign) -> set[str]:
    targets: list[Any]
    targets = list(node.targets) if isinstance(node, ast.Assign) else [node.target]
    return {target.id for target in targets if isinstance(target, ast.Name)}
