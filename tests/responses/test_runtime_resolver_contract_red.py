"""RED contracts for manifest-owned public model alias resolution."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mlx_batch_server.responses.runtime_mapper import (
    CanonicalResponsesMapper,
    ResponsesMappingError,
)
from mlx_batch_server.responses.runtime_resolver import ManifestRuntimeResolver
from mlx_batch_server.runtime.contracts import BackendKind, RoleName, RoleSpec
from mlx_batch_server.runtime.roles import RoleDirectory

if TYPE_CHECKING:
    from mlx_batch_server.responses.controller import PreparedResponse

FLASH = "grant-ai/Qwen3.8-Flash-Next-Abliterated-MLX-4bit"
VISION = "LibraxisAI/Huihui-Qwen3.6-35B-VLM"


class _Projection:
    def observe(self, event: object) -> None:
        del event

    def terminal_envelope(self) -> dict[str, object]:
        return {"id": "unused", "status": "completed"}


def _roles() -> RoleDirectory:
    return RoleDirectory(
        (
            RoleSpec(
                name=RoleName.MAIN,
                port=8100,
                requested_model=FLASH,
                backend=BackendKind.FUSED_MTP_MLX,
                pinned=True,
            ),
            RoleSpec(
                name=RoleName.CANARY,
                port=8101,
                requested_model=FLASH,
                backend=BackendKind.FUSED_MTP_MLX,
            ),
            RoleSpec(
                name=RoleName.VISION,
                port=8102,
                requested_model=VISION,
                backend=BackendKind.LEGACY_MLX,
                pinned=True,
            ),
        )
    )


def _mapper() -> CanonicalResponsesMapper:
    resolver = ManifestRuntimeResolver(
        _roles(),
        {
            "buddy": RoleName.MAIN,
            "programmer": RoleName.MAIN,
            "vision": RoleName.VISION,
            "flash-canary": RoleName.CANARY,
        },
    )
    return CanonicalResponsesMapper(
        resolve_runtime=resolver,
        projection_factory=lambda _: _Projection(),
    )


def _prepare(payload: dict[str, object]) -> PreparedResponse:
    return _mapper().prepare(
        payload,
        response_id="resp_owned",
        owner_id="principal:owner",
        parent_messages=(),
    )


def test_public_multimodal_alias_resolves_to_manifest_owned_main_role() -> None:
    prepared = _prepare(
        {
            "model": "buddy",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Read this."},
                        {"type": "input_image", "file_id": "file_lab"},
                    ],
                }
            ],
            "tools": [{"type": "function", "name": "record_lab_values"}],
        }
    )

    assert prepared.request.runtime.model_id == FLASH
    assert prepared.request.runtime.backend is BackendKind.FUSED_MTP_MLX
    assert prepared.request.metadata["requested_model"] == "buddy"
    assert prepared.request.metadata["resolved_model"] == FLASH
    assert prepared.request.metadata["runtime_role"] == "main"
    assert prepared.request.media[0]["file_id"] == "file_lab"


def test_runtime_role_may_confirm_but_cannot_redirect_an_alias() -> None:
    assert (
        _prepare(
            {"model": "buddy", "runtime_role": "main", "input": "hello"}
        ).request.runtime.model_id
        == FLASH
    )

    with pytest.raises(ResponsesMappingError) as error:
        _prepare({"model": "buddy", "runtime_role": "vision", "input": "hello"})

    assert error.value.code == "runtime_role_mismatch"
    assert error.value.param == "runtime_role"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("revision", "foreign-revision"),
        ("adapter_path", "/foreign/adapter"),
        ("draft_model_id", "foreign-draft"),
        ("backend", "legacy_mlx"),
    ],
)
def test_client_cannot_override_manifest_runtime_identity(
    field: str, value: str
) -> None:
    with pytest.raises(ResponsesMappingError) as error:
        _prepare({"model": "buddy", "input": "hello", field: value})

    assert error.value.code == "runtime_override_forbidden"
    assert error.value.param == field


def test_unknown_alias_and_alias_configuration_fail_closed() -> None:
    with pytest.raises(ResponsesMappingError) as error:
        _prepare({"model": "not-configured", "input": "hello"})
    assert error.value.code == "unknown_model_alias"

    with pytest.raises(ValueError, match="duplicate runtime alias"):
        ManifestRuntimeResolver(
            _roles(),
            {"Buddy": RoleName.MAIN, "buddy": RoleName.VISION},
        )

    with pytest.raises(ValueError, match="unknown role"):
        ManifestRuntimeResolver(_roles(), {"buddy": "flex"})


def test_process_alias_registration_cannot_change_manifest_model_identity() -> None:
    roles = RoleDirectory(
        (
            RoleSpec(
                name=RoleName.MAIN,
                port=8100,
                requested_model=FLASH,
                backend=BackendKind.FUSED_MTP_MLX,
            ),
        )
    )
    resolver = ManifestRuntimeResolver(roles, {FLASH: RoleName.MAIN})

    assert resolver.register_alias("buddy", FLASH) is RoleName.MAIN
    resolved = resolver(
        model="BUDDY",
        role=None,
        revision=None,
        adapter_path=None,
        draft_model_id=None,
        backend=None,
    )

    assert resolved.runtime == roles.runtime_key(RoleName.MAIN)
    assert resolver.aliases["buddy"] is RoleName.MAIN

    with pytest.raises(ResponsesMappingError) as error:
        resolver.register_alias("foreign", "other/model")
    assert error.value.code == "runtime_alias_target_forbidden"
