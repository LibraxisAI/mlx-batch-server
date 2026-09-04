"""HTTP model lifecycle for one manifest-owned runtime role."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from fastapi import APIRouter, Depends, HTTPException

from ..auth.dependency import verify_auth
from ..chat.openai.models.schema import (
    ModelAliasRequest,
    ModelLoadRequest,
    ModelUnloadRequest,
)
from ..runtime.contracts import (
    CapabilityReport,
    ModelState,
    ProcessState,
    RoleName,
    RoleSnapshot,
)
from .runtime_mapper import ResponsesMappingError

if TYPE_CHECKING:
    from ..runtime.manager import RuntimeManager
    from ..runtime.readiness import ReadinessService
    from ..runtime.roles import RoleDirectory
    from .runtime_resolver import ManifestRuntimeResolver


@runtime_checkable
class RuntimeControlRuntime(Protocol):
    """References required by the role-local HTTP control plane."""

    process_role: RoleName
    process_port: int
    role_manifest_sha256: str
    role_directory: RoleDirectory
    readiness_service: ReadinessService
    runtime_manager: RuntimeManager
    runtime_resolver: ManifestRuntimeResolver


class RoleControlService:
    """Expose one role's immutable identity and mutable residency state."""

    def __init__(self, runtime: RuntimeControlRuntime) -> None:
        if not isinstance(runtime, RuntimeControlRuntime):
            raise TypeError("runtime must satisfy RuntimeControlRuntime")
        spec = runtime.role_directory.resolve(runtime.process_role)
        if spec.port != runtime.process_port:
            raise ValueError("runtime process port does not match its role")
        self._runtime = runtime
        self._spec = spec

    @property
    def role(self) -> RoleName:
        return self._spec.name

    async def start_pinned_role(self) -> bool:
        """Make manifest-pinned roles resident before the process serves traffic."""

        if not self._spec.pinned:
            return False
        await self._runtime.runtime_manager.acquire_role(self._spec.name)
        return True

    def role_status(self) -> dict[str, Any]:
        snapshot = self._runtime.readiness_service.snapshot(self._spec.name)
        declared = frozenset(self._spec.capabilities)
        model_ready = _model_ready(snapshot)
        available = self._runtime.readiness_service.is_available(self._spec.name)
        observed = self._runtime.runtime_manager.role_capabilities(self._spec.name)
        if observed is None:
            observed = snapshot.capabilities
        return {
            "role": self._spec.name.value,
            "port": self._spec.port,
            "process_state": snapshot.process_state.value,
            "model_state": snapshot.model_state.value,
            "available": available,
            "wakeable": snapshot.model_state in {ModelState.COLD, ModelState.LOADING},
            "model_ready": model_ready,
            "requested_model": snapshot.requested_model,
            "loaded_model": snapshot.loaded_model,
            "backend": snapshot.backend.value if snapshot.backend else None,
            "revision": self._spec.revision,
            "model_dir": self._spec.model_dir,
            "pinned": self._spec.pinned,
            "local_required": self._spec.local_required,
            "declared_capabilities": sorted(declared),
            "observed_capabilities": _capability_payload(observed),
            "runtime_stats": _json_value(
                self._runtime.runtime_manager.role_stats(self._spec.name)
            ),
            "transition": snapshot.transition,
            "error": snapshot.error,
            "receipt": _json_value(snapshot.receipt),
            "role_manifest_sha256": self._runtime.role_manifest_sha256,
        }

    def runtime_contract(self) -> dict[str, Any]:
        status = self.role_status()
        declared = frozenset(status["declared_capabilities"])
        observed = self._runtime.runtime_manager.role_capabilities(self._spec.name)
        if observed is None:
            observed = self._runtime.readiness_service.snapshot(
                self._spec.name
            ).capabilities
        text_capable = "text" in declared
        vision_capable = "vision" in declared
        tools_capable = "tools" in declared
        mtp_capable = "mtp" in declared
        batch_capable = bool(observed is not None and observed.continuous_batching)
        mtp_enabled = bool(
            observed is not None and observed.facts.get("mtp_policy_enabled", False)
        )
        resident = bool(status["model_ready"])
        return {
            "schema_version": "mlx-batch-server.role-runtime.v1",
            "role": status["role"],
            "port": status["port"],
            "process_state": status["process_state"],
            "model_state": status["model_state"],
            "available": status["available"],
            "wakeable": status["wakeable"],
            "requested_model": status["requested_model"],
            "loaded_model": status["loaded_model"],
            "backend": status["backend"],
            "revision": status["revision"],
            "mtp": {
                "capable": mtp_capable,
                "enabled": mtp_enabled,
                "active": bool(observed is not None and observed.mtp),
            },
            "text": (
                {
                    "capable": True,
                    "resident": resident,
                    "tool_capable": tools_capable,
                    "batch_capable": batch_capable,
                }
                if text_capable
                else {}
            ),
            "multimodal": (
                {"capable": True, "resident": resident} if vision_capable else {}
            ),
            "role_manifest_sha256": status["role_manifest_sha256"],
        }

    def health_payload(self) -> dict[str, Any]:
        status = self.role_status()
        healthy = (
            status["process_state"] == ProcessState.ALIVE.value
            and status["model_state"] != ModelState.DEGRADED.value
        )
        loaded = [status["loaded_model"]] if status["model_ready"] else []
        return {
            "status": "ok" if healthy else "degraded",
            "ok": healthy,
            "loaded_models": loaded,
            "loaded_models_count": len(loaded),
            "runtime_contract": self.runtime_contract(),
            "role_runtime": status,
        }

    def ready_payload(self) -> tuple[int, dict[str, Any]]:
        status = self.role_status()
        ready = bool(status["available"])
        payload = self.health_payload()
        payload.update(
            {
                "ready": ready,
                "checks": {
                    "process": status["process_state"] == ProcessState.ALIVE.value,
                    "role_available": ready,
                    "model_loaded": status["model_ready"],
                },
            }
        )
        return (200 if ready else 503), payload

    def list_models_payload(self) -> dict[str, Any]:
        return {
            "object": "list",
            "data": [self._model_payload()],
            "runtime_contract": self.runtime_contract(),
        }

    def loaded_models_payload(self) -> dict[str, Any]:
        status = self.role_status()
        loaded = [status["loaded_model"]] if status["model_ready"] else []
        entries = [self._loaded_model_payload(status)] if loaded else []
        return {
            "object": "list",
            "data": entries,
            "loaded_models": loaded,
            "loaded_models_count": len(loaded),
            "runtime_contract": self.runtime_contract(),
            "role_runtime": status,
        }

    def aliases_payload(self) -> dict[str, Any]:
        aliases = {
            alias: self._spec.requested_model
            for alias, role in self._runtime.runtime_resolver.aliases.items()
            if role is self._spec.name
        }
        return {
            "object": "list",
            "data": [
                {"alias": alias, "model": model}
                for alias, model in sorted(aliases.items())
            ],
            "aliases": aliases,
            "role": self._spec.name.value,
            "model": self._spec.requested_model,
        }

    async def load_model(self, request: ModelLoadRequest) -> dict[str, Any]:
        self._validate_runtime_request(
            model=request.model,
            adapter_path=request.adapter_path,
            draft_model_id=request.draft_model_id,
        )
        before = self._runtime.readiness_service.snapshot(self._spec.name)
        await self._runtime.runtime_manager.acquire_role(self._spec.name)
        if request.alias is not None:
            self._runtime.runtime_resolver.register_alias(
                request.alias,
                self._spec.requested_model,
            )
        status = self.role_status()
        already_loaded = _model_ready(before)
        return {
            "id": self._spec.requested_model,
            "object": "model",
            "task": request.task or "llm",
            "status": "already_loaded" if already_loaded else "loaded",
            "message": (
                "Manifest-owned runtime was already resident"
                if already_loaded
                else "Manifest-owned runtime is ready"
            ),
            "cache_info": status,
        }

    async def unload_model(
        self,
        request: ModelUnloadRequest | None,
        *,
        deadline_s: float = 30.0,
    ) -> dict[str, Any]:
        request = request or ModelUnloadRequest()
        if request.model is not None:
            self._validate_runtime_request(
                model=request.model,
                adapter_path=request.adapter_path,
                draft_model_id=request.draft_model_id,
            )
        elif request.adapter_path is not None or request.draft_model_id is not None:
            raise ValueError("adapter_path and draft_model_id require model")
        runtime_key = self._runtime.role_directory.runtime_key(self._spec.name)
        unloaded = await self._runtime.runtime_manager.unload(
            runtime_key,
            deadline_s=deadline_s,
        )
        status = self.role_status()
        return {
            "task": request.task or "llm",
            "status": "unloaded" if unloaded else "not_loaded",
            "message": (
                "Manifest-owned runtime is cold"
                if unloaded
                else "Manifest-owned runtime was not resident"
            ),
            "unloaded_models": [self._spec.requested_model] if unloaded else [],
            "cache_info": status,
        }

    def register_alias(self, request: ModelAliasRequest) -> dict[str, Any]:
        self._validate_runtime_request(
            model=request.model,
            adapter_path=request.adapter_path,
            draft_model_id=request.draft_model_id,
        )
        self._runtime.runtime_resolver.register_alias(
            request.alias,
            self._spec.requested_model,
        )
        return {
            "alias": request.alias.strip(),
            "model": self._spec.requested_model,
            "adapter_path": None,
            "draft_model_id": None,
            "status": "registered",
        }

    def model_payload(self, model: str) -> dict[str, Any]:
        self._validate_runtime_request(model=model)
        return self._model_payload()

    def _validate_runtime_request(
        self,
        *,
        model: str,
        adapter_path: str | None = None,
        draft_model_id: str | None = None,
    ) -> None:
        if adapter_path is not None or draft_model_id is not None:
            raise ValueError(
                "adapter_path and draft_model_id are controlled by the role manifest"
            )
        candidate = model.strip() if isinstance(model, str) else ""
        if not candidate:
            raise ValueError("model must be a non-empty string")
        if candidate == self._spec.requested_model:
            return
        role = self._runtime.runtime_resolver.aliases.get(candidate.casefold())
        if role is not self._spec.name:
            raise ValueError("model is not owned by this runtime role")

    def _model_payload(self) -> dict[str, Any]:
        return {
            "id": self._spec.requested_model,
            "object": "model",
            "created": 0,
            "owned_by": "libraxisai",
            "details": {
                "role": self._spec.name.value,
                "port": self._spec.port,
                "revision": self._spec.revision,
                "backend": self._spec.backend.value,
                "capabilities": list(self._spec.capabilities),
            },
        }

    def _loaded_model_payload(self, status: Mapping[str, Any]) -> dict[str, Any]:
        active_lanes: list[str] = []
        if "text" in self._spec.capabilities:
            active_lanes.append("text")
        if "vision" in self._spec.capabilities:
            active_lanes.append("multimodal")
        contract = self.runtime_contract()
        return {
            "id": self._spec.requested_model,
            "object": "model",
            "created": 0,
            "owned_by": "libraxisai",
            "runtime": {
                "active_lanes": active_lanes,
                "text": contract["text"],
                "multimodal": contract["multimodal"],
                "product_residency": status["model_state"],
                "role": self._spec.name.value,
                "backend": self._spec.backend.value,
                "revision": self._spec.revision,
            },
        }


def build_role_control_router(service: RoleControlService) -> APIRouter:
    """Build the only model/readiness control routes for a canonical role."""

    if not isinstance(service, RoleControlService):
        raise TypeError("service must be a RoleControlService")
    router = APIRouter()

    @router.get("/health")
    async def health() -> dict[str, Any]:
        return service.health_payload()

    @router.get("/models")
    @router.get("/v1/models")
    async def list_models(
        _auth: dict[str, Any] = Depends(verify_auth),
    ) -> dict[str, Any]:
        return service.list_models_payload()

    @router.get("/models/loaded")
    @router.get("/v1/models/loaded")
    async def list_loaded_models(
        _auth: dict[str, Any] = Depends(verify_auth),
    ) -> dict[str, Any]:
        return service.loaded_models_payload()

    @router.post("/models/load")
    @router.post("/v1/models/load")
    async def load_model(
        request: ModelLoadRequest,
        _auth: dict[str, Any] = Depends(verify_auth),
    ) -> dict[str, Any]:
        try:
            return await service.load_model(request)
        except Exception as exc:
            raise _control_error(exc) from exc

    @router.get("/models/aliases")
    @router.get("/v1/models/aliases")
    async def list_aliases(
        _auth: dict[str, Any] = Depends(verify_auth),
    ) -> dict[str, Any]:
        return service.aliases_payload()

    @router.post("/models/alias")
    @router.post("/v1/models/alias")
    async def register_alias(
        request: ModelAliasRequest,
        _auth: dict[str, Any] = Depends(verify_auth),
    ) -> dict[str, Any]:
        try:
            return service.register_alias(request)
        except Exception as exc:
            raise _control_error(exc) from exc

    @router.post("/models/unload")
    @router.post("/v1/models/unload")
    async def unload_model(
        request: ModelUnloadRequest | None = None,
        _auth: dict[str, Any] = Depends(verify_auth),
    ) -> dict[str, Any]:
        try:
            return await service.unload_model(request)
        except Exception as exc:
            raise _control_error(exc) from exc

    @router.get("/models/{model_id:path}")
    @router.get("/v1/models/{model_id:path}")
    async def get_model(
        model_id: str,
        _auth: dict[str, Any] = Depends(verify_auth),
    ) -> dict[str, Any]:
        try:
            return service.model_payload(model_id)
        except Exception as exc:
            raise _control_error(exc, status_code=404) from exc

    @router.delete("/models/{model_id:path}")
    @router.delete("/v1/models/{model_id:path}")
    async def delete_model(
        model_id: str,
        _auth: dict[str, Any] = Depends(verify_auth),
    ) -> None:
        try:
            service.model_payload(model_id)
        except Exception as exc:
            raise _control_error(exc, status_code=404) from exc
        raise HTTPException(
            status_code=409,
            detail="manifest-owned model snapshots cannot be deleted by this process",
        )

    return router


def _model_ready(snapshot: RoleSnapshot) -> bool:
    return (
        snapshot.process_state is ProcessState.ALIVE
        and snapshot.model_state is ModelState.READY
        and snapshot.loaded_model == snapshot.requested_model
        and snapshot.error is None
    )


def _capability_payload(value: CapabilityReport | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "supported": value.supported,
        "backend": value.backend.value,
        "architecture": value.architecture,
        "text": value.text,
        "vision": value.vision,
        "tools": value.tools,
        "mtp": value.mtp,
        "continuous_batching": value.continuous_batching,
        "cache_modes": list(value.cache_modes),
        "rejection_reasons": list(value.rejection_reasons),
        "facts": _json_value(value.facts),
    }


def _json_value(value: object) -> object:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list | set | frozenset):
        return [_json_value(item) for item in value]
    return str(value)


def _control_error(exc: Exception, *, status_code: int = 400) -> HTTPException:
    if isinstance(exc, ResponsesMappingError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, TypeError | ValueError):
        return HTTPException(status_code=status_code, detail=str(exc))
    return HTTPException(status_code=503, detail=str(exc) or type(exc).__name__)


__all__ = [
    "RoleControlService",
    "RuntimeControlRuntime",
    "build_role_control_router",
]
