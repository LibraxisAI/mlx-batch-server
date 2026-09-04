from __future__ import annotations

import copy
import json
from dataclasses import FrozenInstanceError
from hashlib import sha256
from pathlib import Path

import pytest

from mlx_batch_server.provenance import BuildReceipt
from mlx_batch_server.runtime.contracts import BackendKind, RoleName
from mlx_batch_server.runtime.role_manifest import (
    ACCEPTED_ROLE_SPECS,
    RoleManifestError,
    RoleManifestSignatureError,
    RoleManifestTopologyError,
    canonical_role_manifest_bytes,
    load_role_manifest,
    packaged_role_manifest_path,
    parse_role_manifest,
    parse_role_manifest_json,
)

MANIFEST_PATH = packaged_role_manifest_path()


def _signed_mapping() -> dict[str, object]:
    roles = [
        {
            "role": spec.name.value,
            "port": spec.port,
            "requested_model": spec.requested_model,
            "revision": spec.revision,
            "model_dir": spec.model_dir,
            "backend": spec.backend.value,
            "pinned": spec.pinned,
            "local_required": spec.local_required,
            "capabilities": list(spec.capabilities),
        }
        for spec in ACCEPTED_ROLE_SPECS
    ]
    digest = sha256(canonical_role_manifest_bytes(ACCEPTED_ROLE_SPECS)).hexdigest()
    return {
        "schema_version": 2,
        "roles": roles,
        "signature": {"algorithm": "sha256", "sha256": digest},
    }


def test_checked_in_manifest_is_signed_and_build_receipt_compatible() -> None:
    manifest = load_role_manifest(MANIFEST_PATH)

    assert manifest.specs == ACCEPTED_ROLE_SPECS
    assert "src/mlx_batch_server/runtime/manifests" in MANIFEST_PATH.as_posix()
    assert manifest.source.path == MANIFEST_PATH.resolve()
    assert manifest.source.canonical_json == canonical_role_manifest_bytes(
        ACCEPTED_ROLE_SPECS
    )
    assert (
        manifest.role_manifest_sha256
        == sha256(manifest.source.canonical_json).hexdigest()
    )
    assert dict(manifest.build_receipt_fields()) == {
        "role_manifest_sha256": manifest.role_manifest_sha256
    }

    receipt = BuildReceipt(
        target_sha="a" * 40,
        target_version="test",
        source_dirty=False,
        omlx_sha="b" * 40,
        mtplx_sha="c" * 40,
        source_origins_sha256="d" * 64,
        dependency_lock_sha256="e" * 64,
        role_manifest_sha256=manifest.role_manifest_sha256,
    )
    assert receipt.role_manifest_sha256 == manifest.role_manifest_sha256


def test_checked_in_manifest_builds_exact_role_directory() -> None:
    directory = load_role_manifest(MANIFEST_PATH).role_directory()

    assert len(directory) == 3
    assert directory.resolve(RoleName.MAIN).port == 8100
    assert directory.resolve(RoleName.MAIN).backend is BackendKind.FUSED_MTP_MLX
    assert directory.resolve(RoleName.MAIN).revision == (
        "000544f8cddcbde27c1bc302deac2b5b4d45a5b1"
    )
    assert directory.runtime_key(RoleName.MAIN).revision == (
        "000544f8cddcbde27c1bc302deac2b5b4d45a5b1"
    )
    assert directory.resolve(RoleName.MAIN).model_dir.endswith(
        "/snapshots/000544f8cddcbde27c1bc302deac2b5b4d45a5b1"
    )
    assert directory.resolve(RoleName.MAIN).pinned is True
    assert directory.resolve(RoleName.MAIN).capabilities == (
        "text",
        "vision",
        "tools",
        "mtp",
    )
    assert directory.resolve(RoleName.CANARY).port == 8101
    assert directory.resolve(RoleName.CANARY).pinned is False
    assert directory.resolve(RoleName.CANARY).capabilities == (
        "text",
        "vision",
        "tools",
        "mtp",
    )
    assert directory.resolve(RoleName.VISION).port == 8102
    assert directory.resolve(RoleName.VISION).backend is BackendKind.LEGACY_MLX
    assert directory.resolve(RoleName.VISION).capabilities == ("vision",)


def test_manifest_source_and_specs_are_immutable_snapshots() -> None:
    source_mapping = _signed_mapping()
    manifest = parse_role_manifest(source_mapping)
    source_mapping["roles"][0]["port"] = 9999  # type: ignore[index]

    assert manifest.specs == ACCEPTED_ROLE_SPECS
    with pytest.raises(FrozenInstanceError):
        manifest.source.sha256 = "0" * 64  # type: ignore[misc]
    with pytest.raises(TypeError):
        manifest.build_receipt_fields()["role_manifest_sha256"] = "0" * 64


@pytest.mark.parametrize(
    "raw_json",
    [
        b'{"schema_version":1,"schema_version":1}',
        (
            b'{"schema_version":1,"roles":[{"role":"main","role":"canary"}],'
            b'"signature":{"algorithm":"sha256","sha256":"00"}}'
        ),
        (
            b'{"schema_version":1,"roles":[],"signature":'
            b'{"algorithm":"sha256","sha256":"00","sha256":"11"}}'
        ),
    ],
)
def test_duplicate_json_fields_are_rejected(raw_json: bytes) -> None:
    with pytest.raises(RoleManifestError, match="duplicate JSON field"):
        parse_role_manifest_json(raw_json)


@pytest.mark.parametrize("location", ["root", "role", "signature"])
def test_unknown_fields_are_rejected_at_every_boundary(location: str) -> None:
    payload = _signed_mapping()
    if location == "root":
        payload["unexpected"] = True
    elif location == "role":
        payload["roles"][0]["unexpected"] = True  # type: ignore[index]
    else:
        payload["signature"]["unexpected"] = True  # type: ignore[index]

    with pytest.raises(RoleManifestError, match="unknown"):
        parse_role_manifest(payload)


@pytest.mark.parametrize("role", ["MAIN", " main", "flex", "unknown"])
def test_noncanonical_or_unaccepted_roles_are_rejected(role: str) -> None:
    payload = _signed_mapping()
    payload["roles"][0]["role"] = role  # type: ignore[index]

    with pytest.raises((RoleManifestError, RoleManifestTopologyError)):
        parse_role_manifest(payload)


def test_duplicate_roles_are_rejected_before_signature_validation() -> None:
    payload = _signed_mapping()
    payload["roles"][1] = copy.deepcopy(payload["roles"][0])  # type: ignore[index]

    with pytest.raises(RoleManifestTopologyError, match="duplicate role"):
        parse_role_manifest(payload)


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_role_set_drift_is_rejected(mutation: str) -> None:
    payload = _signed_mapping()
    if mutation == "missing":
        payload["roles"].pop()  # type: ignore[union-attr]
    else:
        extra = copy.deepcopy(payload["roles"][0])  # type: ignore[index]
        extra["role"] = "flex"
        payload["roles"].append(extra)  # type: ignore[union-attr]

    with pytest.raises(RoleManifestTopologyError):
        parse_role_manifest(payload)


@pytest.mark.parametrize(
    ("role_index", "field", "replacement"),
    [
        (0, "port", 8110),
        (0, "requested_model", "grant-ai/other"),
        (0, "revision", "other-revision"),
        (0, "model_dir", "/other/snapshot"),
        (0, "backend", "legacy_mlx"),
        (0, "pinned", False),
        (0, "local_required", False),
        (0, "capabilities", ["text", "vision"]),
        (1, "port", 8100),
        (1, "pinned", True),
        (2, "requested_model", "LibraxisAI/other"),
        (2, "backend", "fused_mtp_mlx"),
        (2, "capabilities", ["text", "vision"]),
    ],
)
def test_any_accepted_topology_drift_is_rejected(
    role_index: int,
    field: str,
    replacement: object,
) -> None:
    payload = _signed_mapping()
    payload["roles"][role_index][field] = replacement  # type: ignore[index]

    with pytest.raises(RoleManifestTopologyError, match="drifted"):
        parse_role_manifest(payload)


def test_role_order_and_object_key_order_do_not_change_canonical_digest() -> None:
    payload = _signed_mapping()
    payload["roles"] = list(reversed(payload["roles"]))  # type: ignore[arg-type]
    reordered = {
        "signature": payload["signature"],
        "roles": payload["roles"],
        "schema_version": payload["schema_version"],
    }

    manifest = parse_role_manifest(reordered)

    assert (
        manifest.role_manifest_sha256
        == sha256(canonical_role_manifest_bytes(ACCEPTED_ROLE_SPECS)).hexdigest()
    )


def test_tampered_or_noncanonical_signature_is_rejected() -> None:
    payload = _signed_mapping()
    payload["signature"]["sha256"] = "0" * 64  # type: ignore[index]
    with pytest.raises(RoleManifestSignatureError, match="does not match"):
        parse_role_manifest(payload)

    payload = _signed_mapping()
    payload["signature"]["sha256"] = "A" * 64  # type: ignore[index]
    with pytest.raises(RoleManifestSignatureError, match="lowercase hex"):
        parse_role_manifest(payload)

    payload = _signed_mapping()
    payload["signature"]["algorithm"] = "SHA-256"  # type: ignore[index]
    with pytest.raises(RoleManifestSignatureError, match="unsupported"):
        parse_role_manifest(payload)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("schema_version", True),
        ("roles", ()),
    ],
)
def test_manifest_container_types_are_strict(field: str, replacement: object) -> None:
    payload = _signed_mapping()
    payload[field] = replacement

    with pytest.raises(RoleManifestError):
        parse_role_manifest(payload)


def test_role_field_types_are_strict() -> None:
    payload = _signed_mapping()
    payload["roles"][0]["port"] = True  # type: ignore[index]
    with pytest.raises(RoleManifestError, match="port must be an integer"):
        parse_role_manifest(payload)

    payload = _signed_mapping()
    payload["roles"][0]["capabilities"] = ("text",)  # type: ignore[index]
    with pytest.raises(RoleManifestError, match="capabilities must be an array"):
        parse_role_manifest(payload)

    payload = _signed_mapping()
    payload["roles"][0]["model_dir"] = "relative/model"  # type: ignore[index]
    with pytest.raises(RoleManifestError, match="absolute path"):
        parse_role_manifest(payload)


def test_loader_requires_explicit_file_and_never_uses_environment_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fallback = tmp_path / "fallback.json"
    fallback.write_text(json.dumps(_signed_mapping()), encoding="utf-8")
    monkeypatch.setenv("MLX_BATCH_ROLE_MANIFEST", str(fallback))

    with pytest.raises(FileNotFoundError):
        load_role_manifest(tmp_path / "missing.json")
    with pytest.raises(TypeError, match="path is required"):
        load_role_manifest(None)  # type: ignore[arg-type]
