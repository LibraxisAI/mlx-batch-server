# SPDX-License-Identifier: Apache-2.0
"""Fail-closed filesystem plan for loading one Qwen4Exp checkpoint.

This module performs filesystem planning only. It deliberately does not import
MLX, instantiate tokenizers, or select a runtime backend. It hashes every
planned safetensor shard so later reopen and materialization can verify bytes.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from .artifacts import (
    Qwen4ExpArtifactInventory,
    inspect_qwen4_exp_artifacts,
    qwen4_exp_weight_shards,
)
from .config import Qwen4ExpCheckpointConfig, parse_qwen4_exp_config
from .topology import Qwen4ExpTopology, build_qwen4_exp_topology

_DEFAULT_MAX_METADATA_BYTES = 64 * 1024 * 1024
_CONTENT_HASH_CHUNK_BYTES = 8 * 1024 * 1024


class Qwen4ExpLoadPlanError(ValueError):
    """Checkpoint metadata cannot produce an immutable tensor load plan."""


@dataclass(frozen=True, slots=True)
class _DescriptorIdentity:
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int
    link_count: int

    def admits_held_descriptor(self, observed: _DescriptorIdentity) -> bool:
        stable_identity = (self.device, self.inode, self.size, self.mtime_ns)
        observed_stable_identity = (
            observed.device,
            observed.inode,
            observed.size,
            observed.mtime_ns,
        )
        if observed_stable_identity != stable_identity:
            return False
        if (
            observed.ctime_ns == self.ctime_ns
            and observed.link_count == self.link_count
        ):
            return True
        return (
            sys.platform == "darwin"
            and self.link_count > 0
            and observed.link_count == self.link_count - 1
        )


@dataclass(slots=True)
class Qwen4ExpShardLease:
    """Open-file identity whose admitted bytes are consumed by tensor loading."""

    name: str
    stream: BinaryIO
    expected_sha256: str
    _identity: _DescriptorIdentity
    _closed: bool = False

    def prepare_for_load(self) -> BinaryIO:
        """Authenticate and rewind the held identity immediately before MLX."""

        self._verify_held_identity(stage="before tensor load")
        self.stream.seek(0)
        return self.stream

    def verify_after_eval(self) -> None:
        """Rehash the same open file after MLX has evaluated its lazy tensors."""

        self._verify_held_identity(stage="after tensor eval")
        self.stream.seek(0)

    def _verify_held_identity(self, *, stage: str) -> None:
        if self._closed:
            raise Qwen4ExpLoadPlanError(
                f"checkpoint shard lease is closed: {self.name}"
            )
        before = _descriptor_identity(self.stream, self.name)
        observed = _stream_sha256(self.stream, self.name)
        after = _descriptor_identity(self.stream, self.name)
        if not self._identity.admits_held_descriptor(
            before
        ) or not self._identity.admits_held_descriptor(after):
            raise Qwen4ExpLoadPlanError(
                f"checkpoint shard identity changed {stage}: {self.name}"
            )
        if observed != self.expected_sha256:
            raise Qwen4ExpLoadPlanError(
                f"checkpoint shard bytes changed {stage}: {self.name}"
            )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.stream.close()

    def __enter__(self) -> Qwen4ExpShardLease:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


@dataclass(slots=True)
class Qwen4ExpShardSet:
    """All checkpoint shard identities held until final model materialization."""

    leases: Mapping[str, Qwen4ExpShardLease]
    _closed: bool = False

    def stream_for_load(self, shard_name: str) -> BinaryIO:
        if self._closed:
            raise Qwen4ExpLoadPlanError("checkpoint shard set is closed")
        try:
            lease = self.leases[shard_name]
        except KeyError as error:
            raise Qwen4ExpLoadPlanError(
                f"unplanned checkpoint shard: {shard_name}"
            ) from error
        return lease.prepare_for_load()

    def verify_after_eval(self) -> None:
        if self._closed:
            raise Qwen4ExpLoadPlanError("checkpoint shard set is closed")
        for lease in self.leases.values():
            lease.verify_after_eval()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for lease in self.leases.values():
            lease.close()

    def __enter__(self) -> Qwen4ExpShardSet:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


@dataclass(frozen=True, slots=True)
class Qwen4ExpModelLoadPlan:
    model_id: str
    revision: str
    model_dir: str
    config: Qwen4ExpCheckpointConfig
    topology: Qwen4ExpTopology
    artifacts: Qwen4ExpArtifactInventory
    preprocessor_config_json: str
    config_sha256: str
    index_sha256: str
    preprocessor_sha256: str
    tokenizer_fingerprint: str
    plan_sha256: str

    def __post_init__(self) -> None:
        if not self.model_id or not self.revision or not self.model_dir:
            raise Qwen4ExpLoadPlanError("model identity must be complete")
        for name, digest in (
            ("config", self.config_sha256),
            ("index", self.index_sha256),
            ("preprocessor", self.preprocessor_sha256),
            ("tokenizer", self.tokenizer_fingerprint),
            ("plan", self.plan_sha256),
        ):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise Qwen4ExpLoadPlanError(f"{name} digest must be SHA-256")
        if not self.preprocessor_config_json:
            raise Qwen4ExpLoadPlanError("preprocessor metadata must not be empty")
        try:
            preprocessor = json.loads(self.preprocessor_config_json)
        except json.JSONDecodeError as error:
            raise Qwen4ExpLoadPlanError(
                "preprocessor metadata must be canonical JSON"
            ) from error
        if not isinstance(preprocessor, Mapping) or self.preprocessor_config_json != (
            json.dumps(preprocessor, sort_keys=True, separators=(",", ":"))
        ):
            raise Qwen4ExpLoadPlanError(
                "preprocessor metadata must be a canonical JSON object"
            )


def load_qwen4_exp_plan(
    *,
    model_dir: str | Path,
    model_id: str,
    revision: str,
    max_metadata_bytes: int = _DEFAULT_MAX_METADATA_BYTES,
) -> Qwen4ExpModelLoadPlan:
    """Read bounded metadata and freeze the exact future tensor load input."""

    if not model_id or not revision:
        raise Qwen4ExpLoadPlanError("model_id and revision must not be empty")
    if (
        isinstance(max_metadata_bytes, bool)
        or not isinstance(max_metadata_bytes, int)
        or max_metadata_bytes < 1
    ):
        raise Qwen4ExpLoadPlanError("max_metadata_bytes must be positive")

    root = Path(model_dir).expanduser()
    if not root.is_absolute():
        raise Qwen4ExpLoadPlanError("model_dir must be an absolute path")
    if not root.is_dir():
        raise Qwen4ExpLoadPlanError("model_dir must be an existing directory")
    root = root.resolve(strict=True)

    config_path = root / "config.json"
    index_path = root / "model.safetensors.index.json"
    preprocessor_path = root / "preprocessor_config.json"
    tokenizer_path = root / "tokenizer.json"
    tokenizer_config_path = root / "tokenizer_config.json"
    chat_template_path = root / "chat_template.jinja"
    config_bytes = _read_bounded(config_path, max_metadata_bytes)
    index_bytes = _read_bounded(index_path, max_metadata_bytes)
    preprocessor_bytes = _read_bounded(preprocessor_path, max_metadata_bytes)
    tokenizer_bytes = _read_bounded(tokenizer_path, max_metadata_bytes)
    tokenizer_config_bytes = _read_bounded(
        tokenizer_config_path,
        max_metadata_bytes,
    )
    chat_template_bytes = _read_bounded(chat_template_path, max_metadata_bytes)
    config_raw = _json_mapping("config.json", config_bytes)
    index_raw = _json_mapping("model.safetensors.index.json", index_bytes)
    preprocessor_raw = _json_mapping(
        "preprocessor_config.json",
        preprocessor_bytes,
    )
    raw_weight_map = index_raw.get("weight_map")
    if not isinstance(raw_weight_map, Mapping):
        raise Qwen4ExpLoadPlanError("model index requires a weight_map")
    weight_map = _string_mapping("weight_map", raw_weight_map)

    config = parse_qwen4_exp_config(config_raw)
    topology = build_qwen4_exp_topology(config)
    files = tuple(
        sorted(
            item.name
            for item in root.iterdir()
            if item.is_file() and not item.name.startswith(".")
        )
    )
    weight_shards = qwen4_exp_weight_shards(weight_map)
    shard_sha256 = {
        shard_name: _stable_file_sha256(root / shard_name)
        for shard_name in weight_shards
    }
    artifacts = inspect_qwen4_exp_artifacts(
        config=config,
        file_names=files,
        weight_map=weight_map,
        weight_shard_sha256=shard_sha256,
    )
    config_digest = hashlib.sha256(config_bytes).hexdigest()
    index_digest = hashlib.sha256(index_bytes).hexdigest()
    preprocessor_digest = hashlib.sha256(preprocessor_bytes).hexdigest()
    tokenizer_fingerprint = _component_fingerprint(
        (
            (tokenizer_path.name, tokenizer_bytes),
            (tokenizer_config_path.name, tokenizer_config_bytes),
            (chat_template_path.name, chat_template_bytes),
        )
    )
    preprocessor_json = json.dumps(
        preprocessor_raw,
        sort_keys=True,
        separators=(",", ":"),
    )
    plan_payload = {
        "schema": "qwen4-exp-model-load-plan-v1",
        "model_id": model_id,
        "revision": revision,
        "model_dir": str(root),
        "config_sha256": config_digest,
        "index_sha256": index_digest,
        "preprocessor_sha256": preprocessor_digest,
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "artifact_inventory_sha256": artifacts.digest,
        "weight_content_sha256": artifacts.weight_content_sha256,
        "tensor_batch_mode": topology.tensor_batch_mode.value,
        "max_qsa_batch_rows": topology.max_qsa_batch_rows,
        "max_verified_mtp_rows": topology.max_verified_mtp_rows,
    }
    plan_digest = hashlib.sha256(
        json.dumps(plan_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return Qwen4ExpModelLoadPlan(
        model_id=model_id,
        revision=revision,
        model_dir=str(root),
        config=config,
        topology=topology,
        artifacts=artifacts,
        preprocessor_config_json=preprocessor_json,
        config_sha256=config_digest,
        index_sha256=index_digest,
        preprocessor_sha256=preprocessor_digest,
        tokenizer_fingerprint=tokenizer_fingerprint,
        plan_sha256=plan_digest,
    )


def verify_qwen4_exp_shard_content(
    plan: Qwen4ExpModelLoadPlan,
    shard_name: str | None = None,
) -> None:
    """Fail closed when a planned shard no longer has its admitted bytes."""

    expected = dict(plan.artifacts.weight_shard_sha256)
    names = plan.artifacts.weight_shards if shard_name is None else (shard_name,)
    for name in names:
        planned = expected.get(name)
        if planned is None:
            raise Qwen4ExpLoadPlanError(f"unplanned checkpoint shard: {name}")
        observed = _stable_file_sha256(Path(plan.model_dir) / name)
        if observed != planned:
            raise Qwen4ExpLoadPlanError(
                f"checkpoint shard content changed after planning: {name}"
            )


def open_qwen4_exp_shard(
    plan: Qwen4ExpModelLoadPlan,
    shard_name: str,
) -> Qwen4ExpShardLease:
    """Open, authenticate, and hold the exact file object MLX must consume."""

    expected = dict(plan.artifacts.weight_shard_sha256).get(shard_name)
    if expected is None:
        raise Qwen4ExpLoadPlanError(f"unplanned checkpoint shard: {shard_name}")
    if Path(shard_name).name != shard_name:
        raise Qwen4ExpLoadPlanError(
            "load-plan shard names must be checkpoint basenames"
        )
    path = _resolved_checkpoint_file(Path(plan.model_dir) / shard_name)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        stream = os.fdopen(descriptor, "rb", closefd=True)
    except (OSError, ValueError) as error:
        raise Qwen4ExpLoadPlanError(
            f"cannot open checkpoint shard: {shard_name}"
        ) from error
    try:
        identity = _descriptor_identity(stream, shard_name)
        observed = _stream_sha256(stream, shard_name)
        if observed != expected:
            raise Qwen4ExpLoadPlanError(
                f"checkpoint shard content changed after planning: {shard_name}"
            )
        stream.seek(0)
        return Qwen4ExpShardLease(
            name=shard_name,
            stream=stream,
            expected_sha256=expected,
            _identity=identity,
        )
    except BaseException:
        stream.close()
        raise


def open_qwen4_exp_shards(plan: Qwen4ExpModelLoadPlan) -> Qwen4ExpShardSet:
    """Open every planned shard now and retain all identities through eval."""

    leases: dict[str, Qwen4ExpShardLease] = {}
    try:
        for shard_name in plan.artifacts.weight_shards:
            leases[shard_name] = open_qwen4_exp_shard(plan, shard_name)
    except BaseException:
        for lease in leases.values():
            lease.close()
        raise
    return Qwen4ExpShardSet(leases=leases)


def _descriptor_identity(
    stream: BinaryIO,
    name: str,
) -> _DescriptorIdentity:
    try:
        observed = os.fstat(stream.fileno())
    except (OSError, ValueError) as error:
        raise Qwen4ExpLoadPlanError(
            f"cannot stat open checkpoint shard: {name}"
        ) from error
    if not stat.S_ISREG(observed.st_mode) or observed.st_size < 1:
        raise Qwen4ExpLoadPlanError(
            f"checkpoint shard must be a non-empty regular file: {name}"
        )
    return _DescriptorIdentity(
        device=observed.st_dev,
        inode=observed.st_ino,
        size=observed.st_size,
        mtime_ns=observed.st_mtime_ns,
        ctime_ns=observed.st_ctime_ns,
        link_count=observed.st_nlink,
    )


def _stream_sha256(stream: BinaryIO, name: str) -> str:
    digest = hashlib.sha256()
    try:
        stream.seek(0)
        while chunk := stream.read(_CONTENT_HASH_CHUNK_BYTES):
            digest.update(chunk)
    except (OSError, ValueError) as error:
        raise Qwen4ExpLoadPlanError(
            f"cannot hash open checkpoint shard: {name}"
        ) from error
    return digest.hexdigest()


def _stable_file_sha256(path: Path) -> str:
    path = _resolved_checkpoint_file(path)
    if not path.is_file():
        raise Qwen4ExpLoadPlanError(
            f"checkpoint shard must resolve to a regular file: {path.name}"
        )
    try:
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(_CONTENT_HASH_CHUNK_BYTES):
                digest.update(chunk)
        after = path.stat()
    except OSError as error:
        raise Qwen4ExpLoadPlanError(
            f"cannot hash checkpoint shard: {path.name}"
        ) from error
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before.st_size < 1 or identity_before != identity_after:
        raise Qwen4ExpLoadPlanError(
            f"checkpoint shard changed while hashing: {path.name}"
        )
    return digest.hexdigest()


def _resolved_checkpoint_file(path: Path) -> Path:
    """Resolve the bounded Hugging Face snapshot indirection for one shard."""

    if not path.is_symlink():
        return path
    snapshot_dir = path.parent
    snapshots_dir = snapshot_dir.parent
    if snapshots_dir.name != "snapshots":
        raise Qwen4ExpLoadPlanError(
            f"checkpoint shard symlink is not in a Hugging Face snapshot: {path.name}"
        )
    try:
        model_cache_root = snapshots_dir.parent.resolve(strict=True)
        resolved = path.resolve(strict=True)
        resolved.relative_to(model_cache_root)
    except (OSError, RuntimeError, ValueError) as error:
        raise Qwen4ExpLoadPlanError(
            f"checkpoint shard symlink escapes its model cache root: {path.name}"
        ) from error
    if resolved.is_symlink() or not resolved.is_file():
        raise Qwen4ExpLoadPlanError(
            f"checkpoint shard symlink must resolve to a regular file: {path.name}"
        )
    return resolved


def _read_bounded(path: Path, limit: int) -> bytes:
    try:
        size = path.stat().st_size
    except OSError as error:
        raise Qwen4ExpLoadPlanError(
            f"cannot stat checkpoint metadata: {path.name}"
        ) from error
    if size < 1 or size > limit:
        raise Qwen4ExpLoadPlanError(f"checkpoint metadata size is invalid: {path.name}")
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise Qwen4ExpLoadPlanError(
            f"cannot read checkpoint metadata: {path.name}"
        ) from error
    if len(payload) != size:
        raise Qwen4ExpLoadPlanError(
            f"checkpoint metadata changed while reading: {path.name}"
        )
    return payload


def _component_fingerprint(parts: Sequence[tuple[str, bytes]]) -> str:
    digest = hashlib.sha256()
    for name, payload in parts:
        name_bytes = name.encode("utf-8")
        digest.update(len(name_bytes).to_bytes(4, "big"))
        digest.update(name_bytes)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _json_mapping(name: str, payload: bytes) -> Mapping[str, Any]:
    try:
        value = json.loads(payload, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Qwen4ExpLoadPlanError(f"{name} must be valid UTF-8 JSON") from error
    if not isinstance(value, Mapping):
        raise Qwen4ExpLoadPlanError(f"{name} must contain a JSON object")
    return value


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise Qwen4ExpLoadPlanError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _string_mapping(name: str, value: Mapping[Any, Any]) -> Mapping[str, str]:
    if any(
        not isinstance(key, str) or not key or not isinstance(item, str) or not item
        for key, item in value.items()
    ):
        raise Qwen4ExpLoadPlanError(f"{name} must map strings to strings")
    return dict(value)


__all__ = [
    "Qwen4ExpLoadPlanError",
    "Qwen4ExpModelLoadPlan",
    "Qwen4ExpShardLease",
    "Qwen4ExpShardSet",
    "load_qwen4_exp_plan",
    "open_qwen4_exp_shard",
    "open_qwen4_exp_shards",
    "verify_qwen4_exp_shard_content",
]
