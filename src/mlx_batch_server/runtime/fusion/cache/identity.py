# SPDX-License-Identifier: Apache-2.0
"""Stable cache namespace construction for the fused backend.

The compatibility dimensions follow oMLX ``paged_ssd_cache.py`` at commit
``e467261edc786efd33b1e9023d5c4a827f8aa1c1`` and are extended with target
runtime revision, tokenizer, adapter, draft-model, and MTP identity.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

from .contracts import CacheLayout, CacheNamespace

if TYPE_CHECKING:
    from ....cache.contracts import CacheIdentity

CACHE_SIGNATURE_SCHEMA = "mlx-batch-fusion-cache-v1"


def build_cache_namespace(
    identity: CacheIdentity,
    layout: CacheLayout,
) -> CacheNamespace:
    _require_nonempty("model_id", identity.model_id)
    _require_nonempty("model_revision", identity.model_revision)
    _require_nonempty("kv_layout", identity.kv_layout)
    _require_nonempty("tokenizer_fingerprint", identity.tokenizer_fingerprint)
    if identity.adapter_path is not None:
        _require_nonempty("adapter_path", identity.adapter_path)
        _require_nonempty("adapter_fingerprint", layout.adapter_fingerprint)
    elif layout.adapter_fingerprint is not None:
        raise ValueError("adapter_fingerprint requires adapter_path")
    if layout.draft_model_id is not None:
        _require_nonempty("draft_model_id", layout.draft_model_id)
        _require_nonempty("draft_model_revision", layout.draft_model_revision)
    elif layout.draft_model_revision is not None:
        raise ValueError("draft_model_revision requires draft_model_id")

    payload: dict[str, Any] = {
        "schema": CACHE_SIGNATURE_SCHEMA,
        "backend": identity.backend.value,
        "model_id": identity.model_id,
        "model_revision": identity.model_revision,
        "quantization": identity.quantization,
        "adapter_path": identity.adapter_path,
        "adapter_fingerprint": layout.adapter_fingerprint,
        "kv_layout": identity.kv_layout,
        "tokenizer_fingerprint": identity.tokenizer_fingerprint,
        "num_layers": layout.num_layers,
        "block_size_tokens": layout.block_size_tokens,
        "layer_cache_types": list(layout.layer_cache_types),
        "payload_layout": layout.payload_layout,
        "format_version": layout.format_version,
        "draft_model_id": layout.draft_model_id,
        "draft_model_revision": layout.draft_model_revision,
        "mtp_layout": layout.mtp_layout,
        "turboquant_kv_bits": layout.turboquant_kv_bits,
        "cachelist_subtypes": [
            [layer, list(subtypes)]
            for layer, subtypes in sorted(layout.cachelist_subtypes)
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return CacheNamespace(
        identity=identity,
        layout=layout,
        signature=f"{CACHE_SIGNATURE_SCHEMA}:{digest}",
    )


def _require_nonempty(name: str, value: str | None) -> None:
    if value is None or not value.strip():
        raise ValueError(f"{name} must be a non-empty persisted identity field")


__all__ = ["CACHE_SIGNATURE_SCHEMA", "build_cache_namespace"]
