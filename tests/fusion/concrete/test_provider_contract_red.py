"""RED contracts for the source-only concrete tensor provider seam.

These tests are intentionally authored but not executed while the Compile
Embargo is HOLD.
"""

from __future__ import annotations

import ast
import asyncio
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from mlx_batch_server.runtime.backends.fused_mtp_mlx import (
    FusedCacheFactoryPort,
    FusedExecutorFactoryPort,
    FusedStepResult,
)
from mlx_batch_server.runtime.contracts import (
    BackendKind,
    GenerationRequest,
    LoadConfig,
    ModelSpec,
    PreparedGenerationRequest,
    RequestModality,
    RuntimeKey,
)
from mlx_batch_server.runtime.fusion.concrete import (
    FusedTensorCapacityError,
    FusedTensorIdentityError,
    FusedTensorOwnerBinding,
    FusedTensorRegistryClosedError,
    FusedTensorRuntimeRegistry,
    OmlxMtplxCacheFactory,
    OmlxMtplxExecutorFactory,
)
from mlx_batch_server.runtime.fusion.scheduler import (
    SchedulerConfig,
    SchedulerPlan,
)
from mlx_batch_server.runtime.service import FirstWriterCancelToken

if TYPE_CHECKING:
    from mlx_batch_server.runtime.fusion.mtp import MtpPolicy

RUNTIME = RuntimeKey(
    model_id="grant-ai/Qwen3.8-Flash-Next-Abliterated-MLX-4bit",
    revision="000544f8cddcbde27c1bc302deac2b5b4d45a5b1",
    backend=BackendKind.FUSED_MTP_MLX,
)
CONFIG = LoadConfig(
    max_admitted_requests=8,
    max_decode_rows=4,
    max_vision_prefills=2,
    memory_budget_bytes=32_000_000_000,
    cache_directory="/var/cache/mlx-batch-server",
    options={"decode_fair_share": 0.5, "modes": ["paged", "prefix"]},
)
SCHEDULER = SchedulerConfig(
    max_admitted_requests=8,
    max_decode_rows=4,
    max_prefill_rows=2,
    max_vision_prefills=2,
)
MODEL = ModelSpec(
    model_id=RUNTIME.model_id,
    revision=RUNTIME.revision,
    architecture="Qwen4ExpForConditionalGeneration",
    model_type="qwen4_exp",
    quantization="4bit",
    metadata={"tensor_layout": ["qsa", "attention"]},
)


class _Executor:
    def __init__(self, model: ModelSpec) -> None:
        self._model = model
        self.prepared: list[tuple[GenerationRequest, FirstWriterCancelToken]] = []

    @property
    def model_spec(self) -> ModelSpec:
        return self._model

    async def prepare_request(
        self,
        request: GenerationRequest,
        cancel: FirstWriterCancelToken,
    ) -> PreparedGenerationRequest:
        self.prepared.append((request, cancel))
        return PreparedGenerationRequest(request, RequestModality.TEXT)

    async def execute(
        self,
        plan: SchedulerPlan,
        requests: dict[str, GenerationRequest],
        mtp_policy: MtpPolicy,
    ) -> FusedStepResult:
        del plan, requests, mtp_policy
        return FusedStepResult()

    async def cleanup_cancelled(self, request_id: str, reason: str) -> None:
        del request_id, reason

    def stats(self) -> dict[str, Any]:
        return {"observed_steps": 0}

    async def close(self, deadline_s: float) -> None:
        raise AssertionError(f"registry bypassed owner close: {deadline_s}")


class _CacheLease:
    request_id = "response_1"

    async def cleanup(self, reason: Any) -> Any:
        del reason
        raise AssertionError("not used by provider contract")


class _Cache:
    async def acquire(self, request: PreparedGenerationRequest) -> _CacheLease:
        del request
        return _CacheLease()

    def stats(self) -> dict[str, Any]:
        return {"active_leases": 0}

    async def close(self, deadline_s: float) -> None:
        raise AssertionError(f"registry bypassed owner close: {deadline_s}")


class _OwnerLoader:
    def __init__(self) -> None:
        self.calls: list[tuple[RuntimeKey, LoadConfig, SchedulerConfig]] = []
        self.closed: list[tuple[object, float]] = []
        self.started = asyncio.Event()
        self.proceed = asyncio.Event()
        self.proceed.set()
        self.failures: list[Exception] = []
        self.binding_transform: Any = None
        self.owners: list[object] = []

    async def load(
        self,
        runtime: RuntimeKey,
        config: LoadConfig,
        scheduler_config: SchedulerConfig,
    ) -> FusedTensorOwnerBinding:
        self.calls.append((runtime, config, scheduler_config))
        self.started.set()
        await self.proceed.wait()
        if self.failures:
            raise self.failures.pop(0)
        owner = object()
        self.owners.append(owner)
        model = replace(
            MODEL,
            model_id=runtime.model_id,
            revision=runtime.revision,
        )
        binding = FusedTensorOwnerBinding(
            owner=owner,
            runtime=runtime,
            config=config,
            scheduler_config=scheduler_config,
            model=model,
            executor=_Executor(model),
            cache=_Cache(),
        )
        if self.binding_transform is not None:
            binding = self.binding_transform(binding)
        return binding

    async def close(self, owner: object, deadline_s: float) -> None:
        self.closed.append((owner, deadline_s))


def _factories(
    *,
    loader: _OwnerLoader | None = None,
    max_entries: int = 2,
) -> tuple[
    FusedTensorRuntimeRegistry,
    OmlxMtplxExecutorFactory,
    OmlxMtplxCacheFactory,
    _OwnerLoader,
]:
    loader = loader or _OwnerLoader()
    registry = FusedTensorRuntimeRegistry(
        owner_loader=loader,
        max_entries=max_entries,
    )
    return (
        registry,
        OmlxMtplxExecutorFactory(registry),
        OmlxMtplxCacheFactory(registry),
        loader,
    )


@pytest.mark.asyncio
async def test_incompatible_factories_share_exactly_one_owner_single_flight() -> None:
    registry, executor_factory, cache_factory, loader = _factories()
    loader.proceed.clear()

    executor_task = asyncio.create_task(
        executor_factory.load(RUNTIME, CONFIG, SCHEDULER)
    )
    await loader.started.wait()
    cache_task = asyncio.create_task(cache_factory.load(RUNTIME, CONFIG, MODEL))
    await asyncio.sleep(0)
    loader.proceed.set()

    executor, cache = await asyncio.gather(executor_task, cache_task)

    assert isinstance(executor_factory, FusedExecutorFactoryPort)
    assert isinstance(cache_factory, FusedCacheFactoryPort)
    assert len(loader.calls) == 1
    assert registry.entry_count == 1
    assert not hasattr(executor, "owner")
    assert not hasattr(cache, "owner")

    request = GenerationRequest("response_1", RUNTIME, ())
    cancel = FirstWriterCancelToken()
    prepared = await executor.prepare_request(request, cancel)
    assert prepared.request is request
    assert prepared.modality is RequestModality.TEXT
    assert loader.calls

    await executor.close(3.0)
    assert loader.closed == []
    await cache.close(2.0)
    assert loader.closed == [(loader.owners[0], 2.0)]
    assert registry.entry_count == 0


@pytest.mark.asyncio
async def test_cache_cannot_create_an_owner_without_executor_identity() -> None:
    _, _, cache_factory, loader = _factories()

    with pytest.raises(FusedTensorIdentityError, match="before executor"):
        await cache_factory.load(RUNTIME, CONFIG, MODEL)

    assert loader.calls == []


@pytest.mark.asyncio
async def test_config_scheduler_and_model_mismatches_fail_closed() -> None:
    _, executor_factory, cache_factory, loader = _factories()
    executor = await executor_factory.load(RUNTIME, CONFIG, SCHEDULER)

    different_config = replace(CONFIG, max_decode_rows=3)
    with pytest.raises(FusedTensorIdentityError, match="config"):
        await cache_factory.load(RUNTIME, different_config, MODEL)

    different_scheduler = replace(SCHEDULER, decode_fair_share=0.75)
    with pytest.raises(FusedTensorIdentityError, match="scheduler config"):
        await executor_factory.load(RUNTIME, CONFIG, different_scheduler)

    different_model = replace(MODEL, quantization="8bit")
    with pytest.raises(FusedTensorIdentityError, match="cache model"):
        await cache_factory.load(RUNTIME, CONFIG, different_model)

    assert len(loader.calls) == 1
    await executor.close(0.0)


@pytest.mark.asyncio
async def test_loader_identity_mismatch_is_closed_and_not_cached() -> None:
    registry, executor_factory, _, loader = _factories()
    loader.binding_transform = lambda binding: replace(
        binding,
        runtime=replace(RUNTIME, revision="wrong-revision"),
    )

    with pytest.raises(FusedTensorIdentityError, match="runtime/config"):
        await executor_factory.load(RUNTIME, CONFIG, SCHEDULER)

    assert len(loader.calls) == 1
    assert loader.closed == [(loader.owners[0], 0.0)]
    assert registry.entry_count == 0


@pytest.mark.asyncio
async def test_loader_failure_is_single_flight_and_next_attempt_retries() -> None:
    registry, executor_factory, _, loader = _factories()
    loader.proceed.clear()
    loader.failures.append(RuntimeError("load failed"))

    first = asyncio.create_task(executor_factory.load(RUNTIME, CONFIG, SCHEDULER))
    second = asyncio.create_task(executor_factory.load(RUNTIME, CONFIG, SCHEDULER))
    await loader.started.wait()
    await asyncio.sleep(0)
    loader.proceed.set()
    results = await asyncio.gather(first, second, return_exceptions=True)

    assert len(loader.calls) == 1
    assert all(isinstance(result, RuntimeError) for result in results)
    assert registry.entry_count == 0

    recovered = await executor_factory.load(RUNTIME, CONFIG, SCHEDULER)
    assert len(loader.calls) == 2
    await recovered.close(0.0)


@pytest.mark.asyncio
async def test_bounded_entries_reopen_capacity_after_explicit_release() -> None:
    registry, executor_factory, _, _loader = _factories(max_entries=1)
    first = await executor_factory.load(RUNTIME, CONFIG, SCHEDULER)
    other_runtime = replace(RUNTIME, model_id="grant-ai/other-model")

    with pytest.raises(FusedTensorCapacityError, match="capacity"):
        await executor_factory.load(other_runtime, CONFIG, SCHEDULER)

    await first.close(1.0)
    assert registry.entry_count == 0

    second = await executor_factory.load(other_runtime, CONFIG, SCHEDULER)
    await second.close(0.0)


@pytest.mark.asyncio
async def test_release_is_idempotent_and_closes_only_after_both_facets() -> None:
    _, executor_factory, cache_factory, loader = _factories()
    executor = await executor_factory.load(RUNTIME, CONFIG, SCHEDULER)
    cache = await cache_factory.load(RUNTIME, CONFIG, MODEL)

    await cache.close(5.0)
    await cache.close(5.0)
    assert loader.closed == []

    await executor.close(4.0)
    await executor.close(4.0)
    assert loader.closed == [(loader.owners[0], 4.0)]


@pytest.mark.asyncio
async def test_shutdown_closes_active_owner_once_and_rejects_future_work() -> None:
    registry, executor_factory, cache_factory, loader = _factories()
    executor = await executor_factory.load(RUNTIME, CONFIG, SCHEDULER)
    cache = await cache_factory.load(RUNTIME, CONFIG, MODEL)

    await registry.shutdown(7.0)
    await registry.shutdown(7.0)

    assert registry.closed
    assert registry.entry_count == 0
    assert loader.closed == [(loader.owners[0], 7.0)]
    with pytest.raises(FusedTensorRegistryClosedError):
        await executor_factory.load(RUNTIME, CONFIG, SCHEDULER)
    with pytest.raises(FusedTensorRegistryClosedError):
        executor.stats()
    with pytest.raises(FusedTensorRegistryClosedError):
        cache.stats()

    await executor.close(0.0)
    await cache.close(0.0)
    assert len(loader.closed) == 1


def test_provider_import_graph_has_no_tensor_or_donor_runtime_dependency() -> None:
    provider_path = (
        Path(__file__).parents[3]
        / "src/mlx_batch_server/runtime/fusion/concrete/provider.py"
    )
    tree = ast.parse(provider_path.read_text(encoding="utf-8"))
    forbidden_roots = {"mlx", "mlx_lm", "mlx_vlm", "omlx", "mtplx"}
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])

    assert imported_roots.isdisjoint(forbidden_roots)


def test_registry_does_not_advertise_unproven_capabilities() -> None:
    registry, executor_factory, cache_factory, _ = _factories()

    for subject in (registry, executor_factory, cache_factory):
        assert not hasattr(subject, "probe")
        assert not hasattr(subject, "capabilities")
        assert not hasattr(subject, "supports_mtp")
        assert not hasattr(subject, "supports_continuous_batching")
