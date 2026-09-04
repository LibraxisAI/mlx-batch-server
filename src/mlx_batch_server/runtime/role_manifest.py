"""Signed, fail-closed role topology for the 8100-8102 target runtime."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from hmac import compare_digest
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from .contracts import BackendKind, RoleName, RoleSpec
from .roles import RoleDirectory

SCHEMA_VERSION: Final = 2
SIGNATURE_ALGORITHM: Final = "sha256"
PACKAGED_ROLE_MANIFEST: Final = "runtime-roles-8100-8102.json"
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")
_ROOT_FIELDS = frozenset({"schema_version", "roles", "signature"})
_ROLE_FIELDS = frozenset(
    {
        "role",
        "port",
        "requested_model",
        "revision",
        "model_dir",
        "backend",
        "pinned",
        "local_required",
        "capabilities",
    }
)
_SIGNATURE_FIELDS = frozenset({"algorithm", "sha256"})

_FLASH_MODEL = "grant-ai/Qwen3.8-Flash-Next-Abliterated-MLX-4bit"
_FLASH_REVISION = "000544f8cddcbde27c1bc302deac2b5b4d45a5b1"
_FLASH_MODEL_DIR = (
    "/Volumes/Maciejowe/mlx_lm/models/huggingface/hub/"
    "models--grant-ai--Qwen3.8-Flash-Next-Abliterated-MLX-4bit/"
    "snapshots/000544f8cddcbde27c1bc302deac2b5b4d45a5b1"
)
_VISION_MODEL = (
    "LibraxisAI/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-vmlx-mxfp8"
)
_VISION_REVISION = "bf82a9016781681d7b6d77f08c400ef917d82383"
_VISION_MODEL_DIR = (
    "/Volumes/Maciejowe/mlx_lm/models/huggingface/hub/"
    "models--LibraxisAI--Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-"
    "abliterated-vmlx-mxfp8/snapshots/"
    "bf82a9016781681d7b6d77f08c400ef917d82383"
)
_FLASH_CAPABILITIES = (
    "text",
    "vision",
    "tools",
    "mtp",
)

ACCEPTED_ROLE_SPECS: Final = (
    RoleSpec(
        name=RoleName.MAIN,
        port=8100,
        requested_model=_FLASH_MODEL,
        backend=BackendKind.FUSED_MTP_MLX,
        revision=_FLASH_REVISION,
        model_dir=_FLASH_MODEL_DIR,
        pinned=True,
        local_required=True,
        capabilities=_FLASH_CAPABILITIES,
    ),
    RoleSpec(
        name=RoleName.CANARY,
        port=8101,
        requested_model=_FLASH_MODEL,
        backend=BackendKind.FUSED_MTP_MLX,
        revision=_FLASH_REVISION,
        model_dir=_FLASH_MODEL_DIR,
        pinned=False,
        local_required=True,
        capabilities=_FLASH_CAPABILITIES,
    ),
    RoleSpec(
        name=RoleName.VISION,
        port=8102,
        requested_model=_VISION_MODEL,
        backend=BackendKind.LEGACY_MLX,
        revision=_VISION_REVISION,
        model_dir=_VISION_MODEL_DIR,
        pinned=True,
        local_required=True,
        capabilities=("vision",),
    ),
)


class RoleManifestError(ValueError):
    """Base class for malformed or untrusted role manifests."""


class RoleManifestSignatureError(RoleManifestError):
    """Raised when the manifest integrity seal is absent or invalid."""


class RoleManifestTopologyError(RoleManifestError):
    """Raised when a validly shaped manifest changes accepted rollout truth."""


@dataclass(frozen=True, slots=True)
class RoleManifestSource:
    """Immutable source and digest used to populate a build receipt."""

    path: Path | None
    raw_json: bytes
    canonical_json: bytes
    sha256: str


@dataclass(frozen=True, slots=True)
class SignedRoleManifest:
    """Verified topology with target runtime contracts and immutable source truth."""

    schema_version: int
    specs: tuple[RoleSpec, ...]
    source: RoleManifestSource

    @property
    def role_manifest_sha256(self) -> str:
        """Digest intended for ``BuildReceipt.role_manifest_sha256``."""

        return self.source.sha256

    def role_directory(self) -> RoleDirectory:
        """Create the target-owned immutable role lookup from verified specs."""

        return RoleDirectory(self.specs)

    def build_receipt_fields(self) -> Mapping[str, str]:
        """Return an immutable fragment accepted by ``BuildReceipt``."""

        return MappingProxyType({"role_manifest_sha256": self.role_manifest_sha256})


def _exact_fields(
    value: Mapping[object, Any],
    expected: frozenset[str],
    *,
    location: str,
) -> None:
    keys = set(value)
    if any(not isinstance(key, str) for key in keys):
        raise RoleManifestError(f"{location} contains a non-string field name")
    string_keys = {key for key in keys if isinstance(key, str)}
    missing = expected - string_keys
    unknown = string_keys - expected
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing={sorted(missing)!r}")
        if unknown:
            details.append(f"unknown={sorted(unknown)!r}")
        raise RoleManifestError(
            f"{location} fields are not canonical: {', '.join(details)}"
        )


def _mapping(value: object, *, location: str) -> Mapping[object, Any]:
    if not isinstance(value, Mapping):
        raise RoleManifestError(f"{location} must be an object")
    return value


def _canonical_string(value: object, *, location: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise RoleManifestError(f"{location} must be a non-empty canonical string")
    return value


def _role_payload(spec: RoleSpec) -> dict[str, Any]:
    return {
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


def _canonical_payload_bytes(specs: Sequence[RoleSpec]) -> bytes:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "roles": [_role_payload(spec) for spec in specs],
    }
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def canonical_role_manifest_bytes(specs: Sequence[RoleSpec]) -> bytes:
    """Return deterministic signed payload bytes for the accepted topology."""

    ordered = _validate_topology(tuple(specs))
    return _canonical_payload_bytes(ordered)


def _parse_role(value: object, *, index: int) -> RoleSpec:
    location = f"roles[{index}]"
    role = _mapping(value, location=location)
    _exact_fields(role, _ROLE_FIELDS, location=location)

    role_name_raw = _canonical_string(role["role"], location=f"{location}.role")
    try:
        role_name = RoleName(role_name_raw)
    except ValueError as exc:
        raise RoleManifestTopologyError(
            f"{location}.role is not a canonical target role: {role_name_raw!r}"
        ) from exc
    if role_name not in {RoleName.MAIN, RoleName.CANARY, RoleName.VISION}:
        raise RoleManifestTopologyError(
            f"{location}.role is outside the accepted 8100-8102 topology"
        )

    port = role["port"]
    if type(port) is not int:
        raise RoleManifestError(f"{location}.port must be an integer")
    requested_model = _canonical_string(
        role["requested_model"], location=f"{location}.requested_model"
    )
    revision = _canonical_string(role["revision"], location=f"{location}.revision")
    model_dir = _canonical_string(role["model_dir"], location=f"{location}.model_dir")
    if not Path(model_dir).is_absolute():
        raise RoleManifestError(f"{location}.model_dir must be an absolute path")
    backend_raw = _canonical_string(role["backend"], location=f"{location}.backend")
    try:
        backend = BackendKind(backend_raw)
    except ValueError as exc:
        raise RoleManifestTopologyError(
            f"{location}.backend is not a canonical backend: {backend_raw!r}"
        ) from exc

    pinned = role["pinned"]
    local_required = role["local_required"]
    if type(pinned) is not bool:
        raise RoleManifestError(f"{location}.pinned must be a boolean")
    if type(local_required) is not bool:
        raise RoleManifestError(f"{location}.local_required must be a boolean")

    capabilities_raw = role["capabilities"]
    if not isinstance(capabilities_raw, list):
        raise RoleManifestError(f"{location}.capabilities must be an array")
    capabilities = tuple(
        _canonical_string(item, location=f"{location}.capabilities[{offset}]")
        for offset, item in enumerate(capabilities_raw)
    )
    if len(set(capabilities)) != len(capabilities):
        raise RoleManifestError(f"{location}.capabilities contains duplicates")

    return RoleSpec(
        name=role_name,
        port=port,
        requested_model=requested_model,
        backend=backend,
        revision=revision,
        model_dir=model_dir,
        pinned=pinned,
        local_required=local_required,
        capabilities=capabilities,
    )


def _validate_topology(specs: tuple[RoleSpec, ...]) -> tuple[RoleSpec, ...]:
    by_name: dict[RoleName, RoleSpec] = {}
    for spec in specs:
        if spec.name in by_name:
            raise RoleManifestTopologyError(f"duplicate role {spec.name.value!r}")
        by_name[spec.name] = spec

    accepted_names = {spec.name for spec in ACCEPTED_ROLE_SPECS}
    actual_names = set(by_name)
    if actual_names != accepted_names:
        missing = sorted(name.value for name in accepted_names - actual_names)
        extra = sorted(name.value for name in actual_names - accepted_names)
        raise RoleManifestTopologyError(
            f"role set drifted from accepted topology: missing={missing!r}, extra={extra!r}"
        )

    ordered = tuple(by_name[expected.name] for expected in ACCEPTED_ROLE_SPECS)
    for actual, expected in zip(ordered, ACCEPTED_ROLE_SPECS, strict=True):
        if actual != expected:
            raise RoleManifestTopologyError(
                f"role {expected.name.value!r} drifted from accepted rollout truth"
            )
    return ordered


def _signature_digest(value: object) -> str:
    signature = _mapping(value, location="signature")
    _exact_fields(signature, _SIGNATURE_FIELDS, location="signature")
    algorithm = _canonical_string(
        signature["algorithm"], location="signature.algorithm"
    )
    if algorithm != SIGNATURE_ALGORITHM:
        raise RoleManifestSignatureError(
            f"unsupported signature algorithm {algorithm!r}"
        )
    digest = _canonical_string(signature["sha256"], location="signature.sha256")
    if _SHA256_HEX.fullmatch(digest) is None:
        raise RoleManifestSignatureError(
            "signature.sha256 must be 64 lowercase hex digits"
        )
    return digest


def _canonical_signed_bytes(specs: Sequence[RoleSpec], digest: str) -> bytes:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "roles": [_role_payload(spec) for spec in specs],
        "signature": {"algorithm": SIGNATURE_ALGORITHM, "sha256": digest},
    }
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def parse_role_manifest(
    value: Mapping[object, Any],
    *,
    source_path: Path | None = None,
    raw_json: bytes | None = None,
) -> SignedRoleManifest:
    """Validate a mapping without retaining any caller-owned mutable object."""

    root = _mapping(value, location="manifest")
    _exact_fields(root, _ROOT_FIELDS, location="manifest")
    version = root["schema_version"]
    if type(version) is not int or version != SCHEMA_VERSION:
        raise RoleManifestError(f"schema_version must be the integer {SCHEMA_VERSION}")

    roles = root["roles"]
    if not isinstance(roles, list):
        raise RoleManifestError("roles must be an array")
    specs = _validate_topology(
        tuple(_parse_role(role, index=index) for index, role in enumerate(roles))
    )

    claimed_digest = _signature_digest(root["signature"])
    canonical_json = _canonical_payload_bytes(specs)
    actual_digest = sha256(canonical_json).hexdigest()
    if not compare_digest(claimed_digest, actual_digest):
        raise RoleManifestSignatureError(
            "role manifest signature does not match canonical payload"
        )

    immutable_raw = (
        bytes(raw_json)
        if raw_json is not None
        else _canonical_signed_bytes(specs, claimed_digest)
    )
    source = RoleManifestSource(
        path=source_path,
        raw_json=immutable_raw,
        canonical_json=canonical_json,
        sha256=actual_digest,
    )
    return SignedRoleManifest(
        schema_version=version,
        specs=specs,
        source=source,
    )


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise RoleManifestError(f"duplicate JSON field {key!r}")
        value[key] = item
    return value


def parse_role_manifest_json(
    raw_json: bytes,
    *,
    source_path: Path | None = None,
) -> SignedRoleManifest:
    """Parse strict UTF-8 JSON while preserving duplicate-field evidence."""

    if not isinstance(raw_json, bytes):
        raise TypeError("raw_json must be bytes")
    try:
        text = raw_json.decode("utf-8")
        value = json.loads(text, object_pairs_hook=_object_without_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RoleManifestError("role manifest must be valid UTF-8 JSON") from exc
    root = _mapping(value, location="manifest")
    return parse_role_manifest(root, source_path=source_path, raw_json=raw_json)


def load_role_manifest(path: str | Path) -> SignedRoleManifest:
    """Load exactly ``path``; environment variables are never consulted."""

    if path is None:
        raise TypeError("role manifest path is required")
    manifest_path = Path(path)
    raw_json = manifest_path.read_bytes()
    return parse_role_manifest_json(
        raw_json,
        source_path=manifest_path.resolve(),
    )


def packaged_role_manifest_path() -> Path:
    """Return the single role manifest shipped inside the runtime package."""

    path = Path(__file__).with_name("manifests") / PACKAGED_ROLE_MANIFEST
    if not path.is_file():
        raise FileNotFoundError(f"packaged role manifest is missing: {path}")
    return path.resolve()


__all__ = [
    "ACCEPTED_ROLE_SPECS",
    "PACKAGED_ROLE_MANIFEST",
    "SCHEMA_VERSION",
    "SIGNATURE_ALGORITHM",
    "RoleManifestError",
    "RoleManifestSignatureError",
    "RoleManifestSource",
    "RoleManifestTopologyError",
    "SignedRoleManifest",
    "canonical_role_manifest_bytes",
    "load_role_manifest",
    "packaged_role_manifest_path",
    "parse_role_manifest",
    "parse_role_manifest_json",
]
