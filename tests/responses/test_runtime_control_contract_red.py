"""RED contracts for role-local readiness and model lifecycle HTTP control."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mlx_batch_server.chat.openai.models.schema import (
    ModelAliasRequest,
    ModelLoadRequest,
    ModelUnloadRequest,
)
from mlx_batch_server.responses.runtime_control import RoleControlService
from mlx_batch_server.responses.runtime_resolver import ManifestRuntimeResolver
from mlx_batch_server.runtime.contracts import (
    BackendKind,
    CapabilityReport,
    RoleName,
    RoleSpec,
)
from mlx_batch_server.runtime.readiness import ReadinessService
from mlx_batch_server.runtime.roles import RoleDirectory

FLASH = "grant-ai/Qwen3.8-Flash-Next-Abliterated-MLX-4bit"


class _Manager:
    def __init__(self, readiness: ReadinessService) -> None:
        self.readiness = readiness
        self.acquired: list[RoleName] = []
        self.unloaded = False
        self.capabilities: CapabilityReport | None = None
        self.stats: dict[str, object] | None = None

    def role_capabilities(self, role: RoleName) -> CapabilityReport | None:
        del role
        return self.capabilities

    def role_stats(self, role: RoleName) -> dict[str, object] | None:
        del role
        return self.stats

    async def acquire_role(self, role: RoleName) -> object:
        self.acquired.append(role)
        self.readiness.mark_loading(role)
        self.readiness.mark_ready(
            role,
            loaded_model=FLASH,
            backend=BackendKind.FUSED_MTP_MLX,
            capabilities=self.capabilities,
        )
        return object()

    async def unload(self, runtime: object, *, deadline_s: float) -> bool:
        del runtime, deadline_s
        if not self.readiness.is_ready(RoleName.MAIN):
            return False
        self.readiness.mark_unloading(RoleName.MAIN)
        self.readiness.mark_cold(RoleName.MAIN)
        self.unloaded = True
        return True


def _service(*, pinned: bool = True) -> tuple[RoleControlService, _Manager]:
    roles = RoleDirectory(
        (
            RoleSpec(
                name=RoleName.MAIN,
                port=8100,
                requested_model=FLASH,
                revision="snapshot-sha",
                model_dir="/models/snapshot-sha",
                backend=BackendKind.FUSED_MTP_MLX,
                pinned=pinned,
                capabilities=("text", "vision", "tools", "mtp"),
            ),
        )
    )
    readiness = ReadinessService(
        roles,
        receipt={"role_manifest_sha256": "manifest-sha"},
    )
    manager = _Manager(readiness)
    resolver = ManifestRuntimeResolver(roles, {FLASH: RoleName.MAIN})
    runtime = SimpleNamespace(
        process_role=RoleName.MAIN,
        process_port=8100,
        role_manifest_sha256="manifest-sha",
        role_directory=roles,
        readiness_service=readiness,
        runtime_manager=manager,
        runtime_resolver=resolver,
    )
    return RoleControlService(runtime), manager


@pytest.mark.asyncio
async def test_startup_makes_only_a_manifest_pinned_role_resident() -> None:
    pinned_service, pinned_manager = _service(pinned=True)
    cold_service, cold_manager = _service(pinned=False)

    assert await pinned_service.start_pinned_role() is True
    assert pinned_manager.acquired == [RoleName.MAIN]
    assert await cold_service.start_pinned_role() is False
    assert cold_manager.acquired == []


def test_cold_role_is_alive_available_and_wakeable_but_not_resident() -> None:
    service, _ = _service()

    status_code, ready = service.ready_payload()
    loaded = service.loaded_models_payload()

    assert status_code == 200
    assert ready["ready"] is True
    assert ready["role_runtime"]["model_state"] == "cold"
    assert ready["role_runtime"]["wakeable"] is True
    assert ready["role_runtime"]["model_ready"] is False
    assert loaded["loaded_models_count"] == 0
    assert loaded["runtime_contract"]["available"] is True
    assert loaded["runtime_contract"]["text"]["capable"] is True
    assert loaded["runtime_contract"]["multimodal"]["capable"] is True


@pytest.mark.asyncio
async def test_runtime_contract_reads_live_capabilities_from_loaded_handle() -> None:
    service, manager = _service()
    manager.capabilities = CapabilityReport(
        supported=True,
        backend=BackendKind.FUSED_MTP_MLX,
        text=True,
        vision=True,
        tools=True,
        mtp=True,
        continuous_batching=False,
        facts={
            "mtp_runtime_proven": True,
            "mtp_policy_enabled": True,
            "mtp_rounds": 7,
        },
    )
    manager.stats = {
        "mtp": {"rounds": 7, "accepted_tokens": 6},
        "scheduler": {"max_decode_rows": 4},
    }

    await service.load_model(ModelLoadRequest(model=FLASH))
    status = service.role_status()
    contract = service.runtime_contract()

    assert status["observed_capabilities"]["facts"]["mtp_rounds"] == 7
    assert status["runtime_stats"] == manager.stats
    assert contract["mtp"] == {
        "capable": True,
        "enabled": True,
        "active": True,
    }
    assert contract["text"]["batch_capable"] is False


@pytest.mark.asyncio
async def test_load_and_unload_drive_the_same_role_readiness_owner() -> None:
    service, manager = _service()

    loaded = await service.load_model(ModelLoadRequest(model=FLASH))
    resident = service.loaded_models_payload()

    assert loaded["status"] == "loaded"
    assert manager.acquired == [RoleName.MAIN]
    assert resident["loaded_models"] == [FLASH]
    assert resident["data"][0]["runtime"]["active_lanes"] == [
        "text",
        "multimodal",
    ]
    assert resident["runtime_contract"]["model_state"] == "ready"

    unloaded = await service.unload_model(ModelUnloadRequest(model=FLASH))

    assert unloaded["status"] == "unloaded"
    assert manager.unloaded is True
    assert service.role_status()["model_state"] == "cold"


def test_runtime_aliases_can_only_name_the_manifest_owned_model() -> None:
    service, _ = _service()

    receipt = service.register_alias(ModelAliasRequest(alias="buddy", model=FLASH))

    assert receipt == {
        "alias": "buddy",
        "model": FLASH,
        "adapter_path": None,
        "draft_model_id": None,
        "status": "registered",
    }
    assert service.model_payload("BUDDY")["id"] == FLASH
    aliases = service.aliases_payload()
    assert aliases["aliases"]["buddy"] == FLASH
    assert {"alias": "buddy", "model": FLASH} in aliases["data"]

    with pytest.raises(ValueError, match="not owned"):
        service.register_alias(ModelAliasRequest(alias="foreign", model="other/model"))


def test_manifest_identity_rejects_adapter_and_draft_overrides() -> None:
    service, _ = _service()

    with pytest.raises(ValueError, match="controlled by the role manifest"):
        service.register_alias(
            ModelAliasRequest(
                alias="buddy",
                model=FLASH,
                adapter_path="/foreign/adapter",
            )
        )
    with pytest.raises(ValueError, match="controlled by the role manifest"):
        service.register_alias(
            ModelAliasRequest(
                alias="buddy",
                model=FLASH,
                draft_model_id="foreign/draft",
            )
        )
