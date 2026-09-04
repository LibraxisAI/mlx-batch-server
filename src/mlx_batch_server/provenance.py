"""Immutable source provenance for one server process."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Any

SOURCE_SHA_ENV = "MLX_BATCH_SOURCE_SHA"
SOURCE_DIRTY_ENV = "MLX_BATCH_SOURCE_DIRTY"
DEPENDENCY_LOCK_SHA_ENV = "MLX_BATCH_DEPENDENCY_LOCK_SHA256"
WHEEL_SHA_ENV = "MLX_BATCH_WHEEL_SHA256"
_FULL_GIT_SHA = re.compile(r"[0-9a-f]{40,64}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
MTPLX_ATTRIBUTION = {
    "text": "Powered by MTPLX",
    "url": "https://github.com/youssofal/mtplx",
}


@dataclass(frozen=True)
class RuntimeProvenance:
    """Source identity captured before the runtime begins serving requests."""

    source_sha: str | None
    source_dirty: bool | None

    def health_fields(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "third_party_attributions": [dict(MTPLX_ATTRIBUTION)],
        }


@dataclass(frozen=True)
class BuildReceipt:
    """Immutable identity of the target and every admitted source dependency."""

    target_sha: str
    target_version: str
    source_dirty: bool
    omlx_sha: str
    mtplx_sha: str
    source_origins_sha256: str
    dependency_lock_sha256: str
    role_manifest_sha256: str
    wheel_sha256: str | None = None
    donor_wheel_sha256: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if not self.target_version.strip():
            raise ValueError("target_version must not be empty")
        for field_name in ("target_sha", "omlx_sha", "mtplx_sha"):
            value = getattr(self, field_name)
            if _FULL_GIT_SHA.fullmatch(value) is None:
                raise ValueError(f"{field_name} must be a full commit hash")
        for field_name in (
            "source_origins_sha256",
            "dependency_lock_sha256",
            "role_manifest_sha256",
        ):
            value = getattr(self, field_name)
            if _SHA256.fullmatch(value) is None:
                raise ValueError(f"{field_name} must be a SHA-256 digest")
        if (
            self.wheel_sha256 is not None
            and _SHA256.fullmatch(self.wheel_sha256) is None
        ):
            raise ValueError("wheel_sha256 must be a SHA-256 digest")
        if self.donor_wheel_sha256 is not None:
            for name, digest in self.donor_wheel_sha256.items():
                if not isinstance(name, str) or not name.strip():
                    raise ValueError("donor wheel names must not be empty")
                if _SHA256.fullmatch(digest) is None:
                    raise ValueError(
                        "donor_wheel_sha256 values must be SHA-256 digests"
                    )

    def health_fields(self) -> dict[str, Any]:
        return asdict(self)


def _checkout_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _target_version(repo_root: Path) -> str:
    try:
        return metadata.version("mlx-batch-server")
    except metadata.PackageNotFoundError:
        pyproject = repo_root / "pyproject.toml"
        if not pyproject.is_file():
            raise RuntimeError(
                "mlx-batch-server package version is unavailable"
            ) from None
        payload = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        version = payload.get("project", {}).get("version")
        if not isinstance(version, str) or not version.strip():
            raise RuntimeError("pyproject.toml has no valid project version") from None
        return version.strip()


def compose_source_build_receipt(
    *,
    role_manifest_sha256: str,
    repo_root: Path | None = None,
) -> BuildReceipt:
    """Create one startup receipt from checked-in source and resolved lock truth."""

    root = (repo_root or _checkout_root()).resolve()
    provenance = get_runtime_provenance()
    if provenance.source_sha is None or provenance.source_dirty is None:
        raise RuntimeError("runtime source provenance is unavailable")

    origins_path = Path(__file__).with_name("SOURCE_ORIGINS.json")
    origins_raw = origins_path.read_bytes()
    origins = json.loads(origins_raw)
    donors = origins.get("donors")
    if not isinstance(donors, Mapping):
        raise RuntimeError("SOURCE_ORIGINS.json has no donor map")
    try:
        omlx_sha = donors["omlx"]["commit"]
        mtplx_sha = donors["mtplx"]["commit"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("SOURCE_ORIGINS.json has incomplete donor commits") from exc

    dependency_lock_sha256 = os.environ.get(DEPENDENCY_LOCK_SHA_ENV)
    if dependency_lock_sha256 is None:
        lock_path = root / "uv.lock"
        if not lock_path.is_file():
            raise RuntimeError(
                f"{DEPENDENCY_LOCK_SHA_ENV} is required without a source uv.lock"
            )
        dependency_lock_sha256 = _sha256_file(lock_path)

    wheel_sha256 = os.environ.get(WHEEL_SHA_ENV)
    return BuildReceipt(
        target_sha=provenance.source_sha,
        target_version=_target_version(root),
        source_dirty=provenance.source_dirty,
        omlx_sha=omlx_sha,
        mtplx_sha=mtplx_sha,
        source_origins_sha256=hashlib.sha256(origins_raw).hexdigest(),
        dependency_lock_sha256=dependency_lock_sha256,
        role_manifest_sha256=role_manifest_sha256,
        wheel_sha256=wheel_sha256,
    )


def _read_git_checkout(repo_root: Path) -> RuntimeProvenance:
    try:
        sha = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--verify", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
        if _FULL_GIT_SHA.fullmatch(sha) is None:
            return RuntimeProvenance(source_sha=None, source_dirty=None)
        status = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return RuntimeProvenance(source_sha=None, source_dirty=None)
    return RuntimeProvenance(source_sha=sha, source_dirty=bool(status))


def _provenance_from_environment() -> RuntimeProvenance | None:
    sha = os.environ.get(SOURCE_SHA_ENV)
    if sha is None:
        return None
    if _FULL_GIT_SHA.fullmatch(sha) is None:
        return RuntimeProvenance(source_sha=None, source_dirty=None)

    dirty_value = os.environ.get(SOURCE_DIRTY_ENV)
    dirty = {"0": False, "1": True}.get(dirty_value or "")
    return RuntimeProvenance(source_sha=sha, source_dirty=dirty)


def stamp_runtime_environment(repo_root: Path | None = None) -> RuntimeProvenance:
    """Stamp source identity into the environment once, before Uvicorn starts.

    Packagers and supervisors may provide an authoritative full SHA. Otherwise
    the source checkout is inspected once at process start. An installed wheel
    without either source leaves the fields unknown rather than inventing an
    identity.
    """
    provenance = _provenance_from_environment()
    if provenance is None:
        provenance = _read_git_checkout(repo_root or _checkout_root())
        if provenance.source_sha is not None:
            os.environ[SOURCE_SHA_ENV] = provenance.source_sha
            os.environ[SOURCE_DIRTY_ENV] = "1" if provenance.source_dirty else "0"
    return provenance


@lru_cache(maxsize=1)
def get_runtime_provenance() -> RuntimeProvenance:
    """Return the immutable provenance snapshot for this server process."""
    return stamp_runtime_environment()
