"""Cache identity, budget, and lease types without tensor ownership."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..runtime.contracts import BackendKind


@dataclass(frozen=True, slots=True)
class CacheIdentity:
    backend: BackendKind
    model_id: str
    model_revision: str | None
    quantization: str | None
    adapter_path: str | None
    kv_layout: str
    tokenizer_fingerprint: str


@dataclass(frozen=True, slots=True)
class CacheBudget:
    memory_bytes: int
    ssd_bytes: int = 0


@runtime_checkable
class CacheLease(Protocol):
    @property
    def identity(self) -> CacheIdentity: ...

    def release(self) -> None: ...
