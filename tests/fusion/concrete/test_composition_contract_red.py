"""RED contracts for the explicit concrete Qwen4Exp composition root.

These tests are authored but deliberately not executed while Compile Embargo
is HOLD.
"""

from __future__ import annotations

import ast
import asyncio
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any

import pytest

from mlx_batch_server.runtime.backends.fused_mtp_mlx import MtpMlxBackend
from mlx_batch_server.runtime.contracts import (
    BackendKind,
    GenerationRequest,
    LoadConfig,
    PreparedGenerationRequest,
    RuntimeKey,
)
from mlx_batch_server.runtime.fusion.concrete import composition
from mlx_batch_server.runtime.fusion.concrete.composition import (
    compose_qwen4_exp_backend,
)
from mlx_batch_server.runtime.fusion.concrete.owner import Qwen4ExpTensorOwnerLoader
from mlx_batch_server.runtime.fusion.concrete.provider import (
    FusedTensorRuntimeRegistry,
    OmlxMtplxCacheFactory,
    OmlxMtplxExecutorFactory,
)
from mlx_batch_server.runtime.fusion.mtp import MtpPolicy
from mlx_batch_server.runtime.fusion.scheduler import SchedulerConfig

RUNTIME = RuntimeKey(
    model_id="grant-ai/Qwen3.8-Flash-Next-Abliterated-MLX-4bit",
    revision="000544f8cddcbde27c1bc302deac2b5b4d45a5b1",
    backend=BackendKind.FUSED_MTP_MLX,
)
SCHEDULER = SchedulerConfig(
    max_admitted_requests=8,
    max_decode_rows=4,
    max_prefill_rows=2,
    max_vision_prefills=2,
    decode_fair_share=0.75,
    terminal_history_size=64,
)
MTP_POLICY = MtpPolicy(
    draft_depth=3,
    allow_proven_multirow=False,
    max_proven_rows=1,
)
CONFIG = LoadConfig(
    max_admitted_requests=8,
    max_decode_rows=4,
    max_vision_prefills=2,
    memory_budget_bytes=32_000_000_000,
    cache_directory="/var/cache/mlx-batch-server",
    options={
        "model_dir": (
            "/models/grant-ai--Qwen3.8-Flash-Next-Abliterated-MLX-4bit/"
            "000544f8cddcbde27c1bc302deac2b5b4d45a5b1"
        ),
    },
)


class _PreparedFactory:
    runtime = RUNTIME
    config = CONFIG
    scheduler_config = SCHEDULER
    model_plan = object()

    def load(self) -> Any:
        raise AssertionError("composition must not load tensors")


class _ExecutionFactory:
    def __init__(self) -> None:
        self.calls: list[tuple[RuntimeKey, LoadConfig, SchedulerConfig]] = []

    def prepare(
        self,
        runtime: RuntimeKey,
        config: LoadConfig,
        scheduler_config: SchedulerConfig,
    ) -> _PreparedFactory:
        self.calls.append((runtime, config, scheduler_config))
        return _PreparedFactory()


class _RequestPreparer:
    def __init__(self) -> None:
        self.calls: list[GenerationRequest] = []

    async def prepare(
        self,
        request: GenerationRequest,
        cancel: Any,
    ) -> PreparedGenerationRequest:
        del cancel
        self.calls.append(request)
        raise AssertionError("composition must not prepare requests")


def _compose() -> tuple[Any, _ExecutionFactory, _RequestPreparer]:
    execution_factory = _ExecutionFactory()
    request_preparer = _RequestPreparer()
    receipt = compose_qwen4_exp_backend(
        execution_factory=execution_factory,
        request_preparer=request_preparer,
        scheduler_config=SCHEDULER,
        mtp_policy=MTP_POLICY,
        capacity=2,
    )
    return receipt, execution_factory, request_preparer


def test_composition_wires_one_inert_fused_qwen4_exp_graph() -> None:
    receipt, execution_factory, request_preparer = _compose()

    assert isinstance(receipt.owner_loader, Qwen4ExpTensorOwnerLoader)
    assert isinstance(receipt.registry, FusedTensorRuntimeRegistry)
    assert isinstance(receipt.executor_factory, OmlxMtplxExecutorFactory)
    assert isinstance(receipt.cache_factory, OmlxMtplxCacheFactory)
    assert isinstance(receipt.backend, MtpMlxBackend)
    assert receipt.execution_factory is execution_factory
    assert receipt.request_preparer is request_preparer
    assert receipt.scheduler_config is SCHEDULER
    assert receipt.mtp_policy is MTP_POLICY
    assert receipt.capacity == 2
    assert receipt.registry.max_entries == 2

    assert receipt.owner_loader._execution_factory is execution_factory
    assert receipt.owner_loader._request_preparer is request_preparer
    assert receipt.registry._owner_loader is receipt.owner_loader
    assert receipt.executor_factory._registry is receipt.registry
    assert receipt.cache_factory._registry is receipt.registry
    assert receipt.backend._executor_factory is receipt.executor_factory
    assert receipt.backend._cache_factory is receipt.cache_factory
    assert execution_factory.calls == []
    assert request_preparer.calls == []
    assert receipt.registry.entry_count == 0

    with pytest.raises(FrozenInstanceError):
        receipt.capacity = 3  # type: ignore[misc]


@pytest.mark.asyncio
async def test_receipt_shutdown_delegates_to_registry_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt, _, _ = _compose()
    calls: list[float] = []
    entered = asyncio.Event()
    release = asyncio.Event()

    async def shutdown(deadline_s: float) -> None:
        calls.append(deadline_s)
        entered.set()
        await release.wait()

    monkeypatch.setattr(receipt.registry, "shutdown", shutdown)
    first = asyncio.create_task(receipt.shutdown(deadline_s=4.0))
    await entered.wait()
    second = asyncio.create_task(receipt.shutdown(deadline_s=1.0))
    await asyncio.sleep(0)
    release.set()
    await asyncio.gather(first, second)
    await receipt.shutdown(deadline_s=0.0)

    assert calls == [4.0]


@pytest.mark.asyncio
async def test_backend_rejects_identity_and_option_drift_before_prepare() -> None:
    receipt, execution_factory, _ = _compose()

    with pytest.raises(ValueError, match="exact revision"):
        await receipt.backend.load(replace(RUNTIME, revision=None), CONFIG)
    with pytest.raises(ValueError, match="embedded MTP"):
        await receipt.backend.load(
            replace(RUNTIME, draft_model_id="foreign/draft"),
            CONFIG,
        )
    with pytest.raises(ValueError, match="max_decode_rows"):
        await receipt.backend.load(
            RUNTIME,
            replace(CONFIG, max_decode_rows=3),
        )
    with pytest.raises(ValueError, match="decode_fair_share"):
        await receipt.backend.load(
            RUNTIME,
            replace(CONFIG, options={**CONFIG.options, "decode_fair_share": 0.5}),
        )
    with pytest.raises(ValueError, match="model_dir"):
        await receipt.backend.load(
            RUNTIME,
            replace(CONFIG, options={}),
        )

    assert execution_factory.calls == []


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5])
def test_composition_rejects_invalid_capacity(capacity: Any) -> None:
    with pytest.raises(ValueError, match="capacity"):
        compose_qwen4_exp_backend(
            execution_factory=_ExecutionFactory(),
            request_preparer=_RequestPreparer(),
            scheduler_config=SCHEDULER,
            mtp_policy=MTP_POLICY,
            capacity=capacity,
        )


@pytest.mark.parametrize("draft_depth", [0, 9, True, 1.5])
def test_composition_rejects_invalid_mtp_draft_depth(draft_depth: Any) -> None:
    with pytest.raises(ValueError, match="draft_depth"):
        compose_qwen4_exp_backend(
            execution_factory=_ExecutionFactory(),
            request_preparer=_RequestPreparer(),
            scheduler_config=SCHEDULER,
            mtp_policy=replace(MTP_POLICY, draft_depth=draft_depth),
            capacity=2,
        )


def test_canonical_load_config_seals_mtp_draft_depth() -> None:
    canonical = composition._canonical_load_config(
        CONFIG,
        scheduler_config=SCHEDULER,
        mtp_policy=MTP_POLICY,
    )

    assert canonical.options["mtp_draft_depth"] == 3
    with pytest.raises(ValueError, match="mtp_draft_depth"):
        composition._canonical_load_config(
            replace(CONFIG, options={**CONFIG.options, "mtp_draft_depth": 1}),
            scheduler_config=SCHEDULER,
            mtp_policy=MTP_POLICY,
        )


def test_composition_has_no_singleton_or_donor_control_plane_imports() -> None:
    source_path = Path(composition.__file__)
    source = source_path.read_text(encoding="utf-8")
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
    forbidden = ("omlx", "mtplx")
    forbidden_surfaces = ("http", "admin", "store")
    assert not any(
        module == root or module.startswith(f"{root}.")
        for module in imported_modules
        for root in forbidden
    )
    assert not any(
        surface in module.split(".")
        for module in imported_modules
        for surface in forbidden_surfaces
    )

    component_types = (FusedTensorRuntimeRegistry, MtpMlxBackend)
    assert not any(
        isinstance(value, component_types) for value in vars(composition).values()
    )

    compose_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "compose_qwen4_exp_backend"
    )
    called_names = [
        node.func.id
        for node in ast.walk(compose_node)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    for required in (
        "Qwen4ExpExecutionFactory",
        "Qwen4ExpTensorOwnerLoader",
        "FusedTensorRuntimeRegistry",
        "OmlxMtplxExecutorFactory",
        "OmlxMtplxCacheFactory",
        "_ConfiguredQwen4ExpBackend",
    ):
        assert called_names.count(required) == 1
