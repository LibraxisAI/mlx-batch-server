"""Immutable source provenance for one server process."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

SOURCE_SHA_ENV = "MLX_BATCH_SOURCE_SHA"
SOURCE_DIRTY_ENV = "MLX_BATCH_SOURCE_DIRTY"
_FULL_GIT_SHA = re.compile(r"[0-9a-f]{40,64}")


@dataclass(frozen=True)
class RuntimeProvenance:
    """Source identity captured before the runtime begins serving requests."""

    source_sha: str | None
    source_dirty: bool | None

    def health_fields(self) -> dict[str, Any]:
        return asdict(self)


def _checkout_root() -> Path:
    return Path(__file__).resolve().parents[2]


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
