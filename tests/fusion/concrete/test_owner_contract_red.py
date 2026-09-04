"""RED contracts for the single-thread Qwen4Exp tensor owner.

These tests are authored but deliberately not executed while the Compile
Embargo is HOLD.
"""

from __future__ import annotations

import ast
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from mlx_batch_server.runtime.backends.fused_mtp_mlx import FusedStepResult
from mlx_batch_server.runtime.contracts import (
    BackendKind,
    GenerationRequest,
    LoadConfig,
    ModelSpec,
    PreparedGenerationRequest,
    RequestModality,
    RuntimeKey,
)
from mlx_batch_server.runtime.fusion.cache import (
    CacheCleanupReceipt,
    CacheReleaseReason,
    CacheTier,
)
from mlx_batch_server.runtime.fusion.concrete.owner import (
    Qwen4ExpDriverContractError,
    Qwen4ExpRequestPhase,
    Qwen4ExpTensorIdentityError,
    Qwen4ExpTensorOwner,
    Qwen4ExpTensorOwnerClosedError,
    Qwen4ExpTensorOwnerLoader,
)
from mlx_batch_server.runtime.fusion.mtp import MtpPolicy
from mlx_batch_server.runtime.fusion.qwen4_exp import Qwen4ExpExecutionBinding
from mlx_batch_server.runtime.fusion.scheduler import (
    DecodeResult,
    PrefillResult,
    ScheduledRequest,
    SchedulerConfig,
    SchedulerPlan,
    WorkKind,
)

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
    options={"tensor_owner_tombstones": 1},
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
    metadata={"vision_config": {"hidden_size": 1280}},
)


def _request(
    response_id: str = "response_1",
    *,
    messages: list[dict[str, Any]] | None = None,
) -> GenerationRequest:
    return GenerationRequest(
        response_id=response_id,
        runtime=RUNTIME,
        messages=messages or [{"role": "user", "content": "hello"}],
    )


def _prepared(request: GenerationRequest) -> PreparedGenerationRequest:
    return PreparedGenerationRequest(request, RequestModality.TEXT)


def _receipt(
    request_id: str,
    lease_id: str,
    reason: CacheReleaseReason,
    *,
    errors: tuple[str, ...] = (),
) -> CacheCleanupReceipt:
    return CacheCleanupReceipt(
        request_id=request_id,
        lease_id=lease_id,
        reason=reason,
        released_tiers=(CacheTier.PAGED, CacheTier.PREFIX, CacheTier.SSD),
        released_references=3,
        pending_writes_quiesced=True,
        retained_reusable_blocks=reason is CacheReleaseReason.COMPLETED,
        errors=errors,
    )


class _Driver:
    def __init__(self, model: ModelSpec) -> None:
        self._model = model
        self.thread_ids: list[int] = []
        self.calls: list[tuple[Any, ...]] = []
        self.abort_failures = 0
        self.cleanup_failures = 0
        self.foreign_result = False
        self.shutdown_deadlines: list[float] = []

    @property
    def model_spec(self) -> ModelSpec:
        return self._model

    def _record(self, *call: Any) -> None:
        self.thread_ids.append(threading.get_ident())
        self.calls.append(call)

    def reserve(self, request: PreparedGenerationRequest, lease_id: str) -> object:
        self._record("reserve", request.request.response_id, lease_id)
        return {"request_id": request.request.response_id, "lease_id": lease_id}

    def execute(
        self,
        plan: SchedulerPlan,
        reservations: dict[str, object],
        requests: dict[str, GenerationRequest],
        mtp_policy: MtpPolicy,
    ) -> FusedStepResult:
        del requests, mtp_policy
        self._record("execute", plan.step_id, tuple(reservations))
        if self.foreign_result:
            return FusedStepResult(
                prefill_results=(PrefillResult("foreign", 1, complete=True),)
            )
        return FusedStepResult(
            prefill_results=tuple(
                PrefillResult(row.request_id, row.position + 4, complete=True)
                for row in plan.prefill_rows
            ),
            decode_results=tuple(
                DecodeResult(
                    row.request_id,
                    row.position + 1,
                    finished=True,
                    finish_reason="stop",
                )
                for row in plan.decode_rows
            ),
            ar_decode_steps=len(plan.decode_rows),
            ar_decode_tokens=len(plan.decode_rows),
        )

    def abort(self, reservation: object, reason: str) -> None:
        assert isinstance(reservation, dict)
        self._record("abort", reservation["request_id"], reason)
        if self.abort_failures:
            self.abort_failures -= 1
            raise RuntimeError("abort failed")

    def cleanup(
        self,
        reservation: object,
        reason: CacheReleaseReason,
    ) -> CacheCleanupReceipt:
        assert isinstance(reservation, dict)
        request_id = reservation["request_id"]
        lease_id = reservation["lease_id"]
        self._record("cleanup", request_id, reason)
        if self.cleanup_failures:
            self.cleanup_failures -= 1
            return _receipt(request_id, lease_id, reason, errors=("busy",))
        return _receipt(request_id, lease_id, reason)

    def stats(self) -> dict[str, Any]:
        self._record("stats")
        return {"driver_calls": len(self.calls)}

    def shutdown(self, deadline_s: float) -> None:
        self._record("shutdown", deadline_s)
        self.shutdown_deadlines.append(deadline_s)


@dataclass(frozen=True, slots=True)
class _PreparedDriverFactory:
    factory: _DriverFactory
    runtime: RuntimeKey
    config: LoadConfig
    scheduler_config: SchedulerConfig
    model_plan: object

    def load(self) -> Qwen4ExpExecutionBinding:
        owner = self.factory
        owner.thread_ids.append(threading.get_ident())
        binding = Qwen4ExpExecutionBinding(
            execution=owner.driver,
            runtime=self.runtime,
            config=self.config,
            scheduler_config=self.scheduler_config,
            model=owner.driver.model_spec,
        )
        if owner.binding_transform is not None:
            binding = owner.binding_transform(binding)
        return binding


class _DriverFactory:
    def __init__(self) -> None:
        self.driver = _Driver(MODEL)
        self.prepare_thread_ids: list[int] = []
        self.thread_ids: list[int] = []
        self.binding_transform: Any = None
        self.prepared_transform: Any = None

    def prepare(
        self,
        runtime: RuntimeKey,
        config: LoadConfig,
        scheduler_config: SchedulerConfig,
    ) -> _PreparedDriverFactory:
        self.prepare_thread_ids.append(threading.get_ident())
        prepared = _PreparedDriverFactory(
            factory=self,
            runtime=runtime,
            config=config,
            scheduler_config=scheduler_config,
            model_plan=object(),
        )
        if self.prepared_transform is not None:
            prepared = self.prepared_transform(prepared)
        return prepared


async def _binding(
    factory: _DriverFactory | None = None,
) -> tuple[Qwen4ExpTensorOwnerLoader, Any, _DriverFactory]:
    factory = factory or _DriverFactory()
    loader = Qwen4ExpTensorOwnerLoader(factory)
    binding = await loader.load(RUNTIME, CONFIG, SCHEDULER)
    return loader, binding, factory


@pytest.mark.asyncio
async def test_driver_lifecycle_is_serialized_on_one_inference_thread() -> None:
    event_loop_thread = threading.get_ident()
    loader, binding, factory = await _binding()
    owner = binding.owner
    assert isinstance(owner, Qwen4ExpTensorOwner)
    request = _request()
    lease = await binding.cache.acquire(_prepared(request))

    prefill = SchedulerPlan(
        step_id=1,
        prefill_rows=(ScheduledRequest(request.response_id, WorkKind.TEXT, 0),),
    )
    await binding.executor.execute(
        prefill,
        {request.response_id: request},
        MtpPolicy(),
    )
    assert (await owner.request_snapshot(request.response_id)).phase is (
        Qwen4ExpRequestPhase.DECODE
    )
    with pytest.raises(Qwen4ExpTensorIdentityError, match="strictly increasing"):
        await binding.executor.execute(
            prefill,
            {request.response_id: request},
            MtpPolicy(),
        )

    decode = SchedulerPlan(
        step_id=2,
        decode_rows=(ScheduledRequest(request.response_id, WorkKind.TEXT, 4),),
    )
    await binding.executor.execute(
        decode,
        {request.response_id: request},
        MtpPolicy(),
    )
    assert (await owner.request_snapshot(request.response_id)).phase is (
        Qwen4ExpRequestPhase.FINISHED
    )
    receipt = await lease.cleanup(CacheReleaseReason.COMPLETED)
    assert receipt.succeeded
    assert (await owner.request_snapshot(request.response_id)).phase is (
        Qwen4ExpRequestPhase.CLEANED
    )

    await loader.close(owner, 3.0)
    assert factory.prepare_thread_ids == [event_loop_thread]
    observed_threads = set(factory.thread_ids + factory.driver.thread_ids)
    assert observed_threads == {owner.owner_thread_id}
    assert event_loop_thread not in observed_threads
    assert len(factory.driver.shutdown_deadlines) == 1
    assert 0.0 < factory.driver.shutdown_deadlines[0] <= 3.0


@pytest.mark.asyncio
async def test_request_runtime_and_mutable_payload_identity_fail_closed() -> None:
    loader, binding, _ = await _binding()
    owner = binding.owner
    foreign = replace(_request(), runtime=replace(RUNTIME, revision="foreign"))
    with pytest.raises(Qwen4ExpTensorIdentityError, match="request runtime"):
        await binding.cache.acquire(_prepared(foreign))

    messages = [{"role": "user", "content": ["stable"]}]
    request = _request(messages=messages)
    lease = await binding.cache.acquire(_prepared(request))
    messages[0]["content"].append("mutated")
    plan = SchedulerPlan(
        step_id=1,
        prefill_rows=(ScheduledRequest(request.response_id, WorkKind.TEXT, 0),),
    )
    with pytest.raises(Qwen4ExpTensorIdentityError, match="changed"):
        await binding.executor.execute(
            plan,
            {request.response_id: request},
            MtpPolicy(),
        )

    await lease.cleanup(CacheReleaseReason.REJECTED)
    await loader.close(owner, 0.0)


@pytest.mark.asyncio
async def test_plan_and_driver_result_identities_are_exact() -> None:
    loader, binding, factory = await _binding()
    owner = binding.owner
    request = _request()
    lease = await binding.cache.acquire(_prepared(request))
    plan = SchedulerPlan(
        step_id=1,
        prefill_rows=(ScheduledRequest(request.response_id, WorkKind.TEXT, 0),),
    )

    with pytest.raises(Qwen4ExpTensorIdentityError, match="exactly match"):
        await binding.executor.execute(plan, {}, MtpPolicy())

    factory.driver.foreign_result = True
    with pytest.raises(Qwen4ExpDriverContractError, match="prefill results"):
        await binding.executor.execute(
            plan,
            {request.response_id: request},
            MtpPolicy(),
        )

    await lease.cleanup(CacheReleaseReason.FAILED)
    await loader.close(owner, 0.0)


@pytest.mark.asyncio
async def test_abort_is_retryable_and_first_reason_wins() -> None:
    loader, binding, factory = await _binding()
    owner = binding.owner
    request = _request()
    lease = await binding.cache.acquire(_prepared(request))
    factory.driver.abort_failures = 1

    with pytest.raises(RuntimeError, match="abort failed"):
        await binding.executor.cleanup_cancelled(request.response_id, "disconnect")
    await binding.executor.cleanup_cancelled(request.response_id, "later reason")
    abort_calls = [call for call in factory.driver.calls if call[0] == "abort"]
    assert [call[2] for call in abort_calls] == ["disconnect", "disconnect"]

    receipt = await lease.cleanup(CacheReleaseReason.CANCELLED)
    assert receipt.succeeded
    await loader.close(owner, 0.0)


@pytest.mark.asyncio
async def test_cleanup_failure_retries_and_success_is_idempotent() -> None:
    loader, binding, factory = await _binding()
    owner = binding.owner
    request = _request()
    lease = await binding.cache.acquire(_prepared(request))
    factory.driver.cleanup_failures = 1

    failed = await lease.cleanup(CacheReleaseReason.COMPLETED)
    assert not failed.succeeded
    snapshot = await owner.request_snapshot(request.response_id)
    assert snapshot.phase is Qwen4ExpRequestPhase.CLEANUP_FAILED

    completed = await lease.cleanup(CacheReleaseReason.COMPLETED)
    repeated = await lease.cleanup(CacheReleaseReason.COMPLETED)
    assert completed.succeeded
    assert repeated.succeeded
    assert repeated.already_released
    cleanup_calls = [call for call in factory.driver.calls if call[0] == "cleanup"]
    assert len(cleanup_calls) == 2
    await loader.close(owner, 0.0)


@pytest.mark.asyncio
async def test_stale_lease_cannot_cleanup_reused_response_id() -> None:
    loader, binding, _ = await _binding()
    owner = binding.owner
    first = _request("response_reused")
    stale = await binding.cache.acquire(_prepared(first))
    await stale.cleanup(CacheReleaseReason.REJECTED)

    eviction = await binding.cache.acquire(_prepared(_request("response_evictor")))
    await eviction.cleanup(CacheReleaseReason.REJECTED)
    current = await binding.cache.acquire(_prepared(first))
    with pytest.raises(Qwen4ExpTensorIdentityError, match="stale cache lease"):
        await stale.cleanup(CacheReleaseReason.REJECTED)

    await current.cleanup(CacheReleaseReason.REJECTED)
    await loader.close(owner, 0.0)


@pytest.mark.asyncio
async def test_shutdown_aborts_and_cleans_every_active_request_before_driver() -> None:
    loader, binding, factory = await _binding()
    owner = binding.owner
    request = _request()
    await binding.cache.acquire(_prepared(request))

    await loader.close(owner, 5.0)

    significant = [call[0] for call in factory.driver.calls if call[0] != "stats"]
    assert significant == ["reserve", "abort", "cleanup", "shutdown"]
    cleanup = next(call for call in factory.driver.calls if call[0] == "cleanup")
    assert cleanup[2] is CacheReleaseReason.SHUTDOWN
    assert owner.closed
    with pytest.raises(Qwen4ExpTensorOwnerClosedError):
        await binding.cache.acquire(_prepared(_request("response_after_close")))


@pytest.mark.asyncio
async def test_driver_binding_mismatch_stops_owner_thread() -> None:
    factory = _DriverFactory()
    factory.binding_transform = lambda binding: replace(
        binding,
        runtime=replace(RUNTIME, revision="foreign"),
    )
    loader = Qwen4ExpTensorOwnerLoader(factory)

    with pytest.raises(Qwen4ExpTensorIdentityError, match="runtime identity"):
        await loader.load(RUNTIME, CONFIG, SCHEDULER)
    assert factory.driver.shutdown_deadlines == [0.0]


@pytest.mark.asyncio
async def test_prepared_factory_identity_fails_before_owner_mailbox() -> None:
    factory = _DriverFactory()
    event_loop_thread = threading.get_ident()
    factory.prepared_transform = lambda prepared: replace(
        prepared,
        runtime=replace(RUNTIME, revision="foreign"),
    )
    loader = Qwen4ExpTensorOwnerLoader(factory)

    with pytest.raises(Qwen4ExpTensorIdentityError, match="prepared factory"):
        await loader.load(RUNTIME, CONFIG, SCHEDULER)
    assert factory.prepare_thread_ids == [event_loop_thread]
    assert factory.thread_ids == []
    assert factory.driver.calls == []


def test_owner_source_records_frozen_donors_without_runtime_imports_or_stubs() -> None:
    owner_path = (
        Path(__file__).parents[3]
        / "src/mlx_batch_server/runtime/fusion/concrete/owner.py"
    )
    source = owner_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])

    assert imported_roots.isdisjoint({"mlx", "mlx_lm", "mlx_vlm", "omlx", "mtplx"})
    assert "e467261edc786efd33b1e9023d5c4a827f8aa1c1" in source
    assert "6d0ddf0575faa9acf77e63c57e48ea1602a7e4ab" in source
    assert "NotImplemented" not in source
    assert not any(isinstance(node, ast.Pass) for node in ast.walk(tree))
