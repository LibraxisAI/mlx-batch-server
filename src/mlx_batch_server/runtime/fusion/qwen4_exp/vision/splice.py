# SPDX-License-Identifier: Apache-2.0
# Derived from youssofal/mtplx@6d0ddf0575faa9acf77e63c57e48ea1602a7e4ab
# mtplx/vision/splice.py; adapted to immutable plans and per-request cursors.
"""Content-keyed Qwen4Exp splice plans and owner-thread cursors."""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Sequence
from dataclasses import dataclass

from .processing import (
    ImageGridReceipt,
    OpaqueRows,
    ProcessedVisionBatch,
    VisionContractError,
    VisionRequestIdentity,
    _runtime_payload,
)
from .tower import VisionTowerOutput, VisionTowerRequest, validate_tower_output

_SURROGATE_FLAG = 1 << 256
_SURROGATE_DOMAIN = b"mlx-batch-server/qwen4-exp/vision-key/v1\0"


@dataclass(frozen=True, slots=True)
class VisionImageSpan:
    identity: VisionRequestIdentity
    image_index: int
    content_digest: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.image_index < 0 or self.start < 0 or self.end <= self.start:
            raise VisionContractError(
                "invalid_image_span",
                "image span must be a positive half-open prompt range",
            )
        if len(self.content_digest) != 64:
            raise VisionContractError(
                "missing_image_digest",
                "image span requires a content digest",
            )

    @property
    def pad_rows(self) -> int:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class VisionSplicePlan:
    """Immutable splice truth for one expanded prompt and one tower output."""

    identity: VisionRequestIdentity
    plan_digest: str
    image_pad_token_id: int
    prompt_token_ids: tuple[int, ...]
    keyed_prompt_token_ids: tuple[int, ...]
    image_spans: tuple[VisionImageSpan, ...]
    image_grids: tuple[ImageGridReceipt, ...]
    embeddings: OpaqueRows

    def __post_init__(self) -> None:
        prompt = _token_tuple(self.prompt_token_ids)
        keyed = _token_tuple(self.keyed_prompt_token_ids)
        if self.image_pad_token_id < 0:
            raise VisionContractError(
                "invalid_image_pad_token",
                "image pad token id must be non-negative",
            )
        if not prompt:
            raise VisionContractError(
                "empty_prompt",
                "splice plan requires an expanded prompt",
            )
        if not _is_sha256(self.plan_digest):
            raise VisionContractError(
                "invalid_plan_digest",
                "splice plan digest must be SHA-256",
            )
        if len(keyed) != len(prompt):
            raise VisionContractError(
                "keyed_prompt_length_mismatch",
                "content-keyed prompt must preserve prompt length",
            )
        grids = tuple(self.image_grids)
        spans = tuple(self.image_spans)
        if len(grids) != len(spans):
            raise VisionContractError(
                "splice_image_count_mismatch",
                "splice grids and spans must have equal image count",
            )
        total_rows = 0
        for index, (grid, span) in enumerate(zip(grids, spans, strict=True)):
            if grid.identity != self.identity or span.identity != self.identity:
                raise VisionContractError(
                    "splice_identity_mismatch",
                    "splice metadata belongs to another request",
                )
            if grid.image_index != index or span.image_index != index:
                raise VisionContractError(
                    "splice_image_order_mismatch",
                    "splice metadata must preserve prompt image order",
                )
            if grid.content_digest != span.content_digest:
                raise VisionContractError(
                    "splice_digest_mismatch",
                    "splice grid and span content digests disagree",
                )
            if grid.pad_rows != span.pad_rows:
                raise VisionContractError(
                    "splice_pad_layout_mismatch",
                    "image span length must equal grid pad rows",
                )
            total_rows += grid.pad_rows
        if self.embeddings.row_count != total_rows:
            raise VisionContractError(
                "splice_embedding_row_mismatch",
                "splice embeddings must contain exactly all image pad rows",
            )
        digests = tuple(grid.content_digest for grid in grids)
        pad_counts = tuple(grid.pad_rows for grid in grids)
        expected_keys = build_content_key_surrogates(
            prompt,
            image_pad_token_id=self.image_pad_token_id,
            image_digests=digests,
            pad_counts=pad_counts,
        )
        expected_spans = build_image_spans(
            self.identity,
            prompt,
            image_pad_token_id=self.image_pad_token_id,
            image_digests=digests,
            pad_counts=pad_counts,
        )
        if expected_keys is None or keyed != expected_keys:
            raise VisionContractError(
                "splice_content_keys_mismatch",
                "splice content keys do not match image identity and pad layout",
            )
        if expected_spans is None or spans != expected_spans:
            raise VisionContractError(
                "splice_image_spans_mismatch",
                "splice image spans do not match the exact pad layout",
            )
        expected_digest = _splice_plan_digest(
            self.identity,
            self.image_pad_token_id,
            prompt,
            keyed,
            grids,
        )
        if self.plan_digest != expected_digest:
            raise VisionContractError(
                "splice_plan_digest_mismatch",
                "splice plan digest does not match its immutable content",
            )
        object.__setattr__(self, "prompt_token_ids", prompt)
        object.__setattr__(self, "keyed_prompt_token_ids", keyed)
        object.__setattr__(self, "image_grids", grids)
        object.__setattr__(self, "image_spans", spans)

    @property
    def total_rows(self) -> int:
        return self.embeddings.row_count


@dataclass(frozen=True, slots=True)
class VisionSpliceWindow:
    identity: VisionRequestIdentity
    plan_digest: str
    prompt_start: int
    prompt_end: int
    row_start: int
    row_end: int
    pad_offsets: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.prompt_start < 0 or self.prompt_end <= self.prompt_start:
            raise VisionContractError(
                "invalid_prompt_window",
                "prompt window must be a positive half-open range",
            )
        if self.row_start < 0 or self.row_end <= self.row_start:
            raise VisionContractError(
                "invalid_row_window",
                "row window must be a positive half-open range",
            )
        if self.row_end - self.row_start != len(self.pad_offsets):
            raise VisionContractError(
                "window_row_mismatch",
                "one splice row is required for every pad offset",
            )


class VisionSpliceCursor:
    """Mutable sequential cursor owned by exactly one request thread."""

    __slots__ = ("_owner_thread_id", "_plan", "_prompt_cursor", "_row_cursor")

    def __init__(self, plan: VisionSplicePlan) -> None:
        if not isinstance(plan, VisionSplicePlan):
            raise VisionContractError(
                "invalid_splice_plan",
                "cursor requires a VisionSplicePlan",
            )
        self._owner_thread_id = threading.get_ident()
        self._plan = plan
        self._prompt_cursor = 0
        self._row_cursor = 0

    @property
    def prompt_cursor(self) -> int:
        return self._prompt_cursor

    @property
    def row_cursor(self) -> int:
        return self._row_cursor

    @property
    def remaining_rows(self) -> int:
        return self._plan.total_rows - self._row_cursor

    def consume(
        self,
        *,
        identity: VisionRequestIdentity,
        plan_digest: str,
        token_ids: Sequence[int],
    ) -> VisionSpliceWindow | None:
        """Consume the next exact prompt chunk and advance sequential state."""

        self._check_access(identity, plan_digest)
        chunk = _token_tuple(token_ids)
        if not chunk:
            raise VisionContractError(
                "empty_splice_chunk",
                "splice chunks must not be empty",
            )
        start = self._prompt_cursor
        end = start + len(chunk)
        if end > len(self._plan.prompt_token_ids):
            raise VisionContractError(
                "splice_prompt_overflow",
                "splice chunk extends beyond the expanded prompt",
            )
        if chunk != self._plan.prompt_token_ids[start:end]:
            raise VisionContractError(
                "splice_prompt_mismatch",
                "splice chunks must consume the exact prompt sequentially",
            )
        window = self._window(start, end, chunk, self._row_cursor)
        self._prompt_cursor = end
        if window is not None:
            self._row_cursor = window.row_end
        return window

    def lookup_window(
        self,
        *,
        identity: VisionRequestIdentity,
        plan_digest: str,
        prompt_start: int,
        token_ids: Sequence[int],
    ) -> VisionSpliceWindow | None:
        """Return an MTP/history row window without mutating either cursor."""

        self._check_access(identity, plan_digest)
        chunk = _token_tuple(token_ids)
        if prompt_start < 0 or not chunk:
            raise VisionContractError(
                "invalid_lookup_window",
                "lookup requires a non-negative start and non-empty tokens",
            )
        prompt_end = prompt_start + len(chunk)
        if prompt_end > len(self._plan.prompt_token_ids):
            raise VisionContractError(
                "splice_window_overflow",
                "lookup window extends beyond the expanded prompt",
            )
        if chunk != self._plan.prompt_token_ids[prompt_start:prompt_end]:
            raise VisionContractError(
                "splice_window_mismatch",
                "lookup window must match the exact expanded prompt",
            )
        rows_before = sum(
            token == self._plan.image_pad_token_id
            for token in self._plan.prompt_token_ids[:prompt_start]
        )
        return self._window(prompt_start, prompt_end, chunk, rows_before)

    def reset(
        self,
        *,
        identity: VisionRequestIdentity,
        plan_digest: str,
    ) -> None:
        self._check_access(identity, plan_digest)
        self._prompt_cursor = 0
        self._row_cursor = 0

    def assert_complete(
        self,
        *,
        identity: VisionRequestIdentity,
        plan_digest: str,
    ) -> None:
        self._check_access(identity, plan_digest)
        if self._prompt_cursor != len(self._plan.prompt_token_ids):
            raise VisionContractError(
                "splice_prompt_incomplete",
                "expanded prompt was not fully consumed",
            )
        if self._row_cursor != self._plan.total_rows:
            raise VisionContractError(
                "splice_rows_incomplete",
                "vision embeddings were not fully consumed",
            )

    def _window(
        self,
        prompt_start: int,
        prompt_end: int,
        chunk: tuple[int, ...],
        row_start: int,
    ) -> VisionSpliceWindow | None:
        offsets = tuple(
            offset
            for offset, token in enumerate(chunk)
            if token == self._plan.image_pad_token_id
        )
        if not offsets:
            return None
        row_end = row_start + len(offsets)
        if row_end > self._plan.total_rows:
            raise VisionContractError(
                "splice_row_overflow",
                "prompt requires more rows than the tower produced",
            )
        return VisionSpliceWindow(
            identity=self._plan.identity,
            plan_digest=self._plan.plan_digest,
            prompt_start=prompt_start,
            prompt_end=prompt_end,
            row_start=row_start,
            row_end=row_end,
            pad_offsets=offsets,
        )

    def _check_access(
        self,
        identity: VisionRequestIdentity,
        plan_digest: str,
    ) -> None:
        if threading.get_ident() != self._owner_thread_id:
            raise VisionContractError(
                "owner_thread_violation",
                "vision splice cursor may only run on its owner thread",
            )
        if identity != self._plan.identity or plan_digest != self._plan.plan_digest:
            raise VisionContractError(
                "splice_identity_mismatch",
                "vision splice cursor identity does not match its plan",
            )


def build_vision_splice_plan(
    processed: ProcessedVisionBatch,
    tower_output: VisionTowerOutput,
    *,
    prompt_token_ids: Sequence[int],
    image_pad_token_id: int,
) -> VisionSplicePlan | None:
    """Build a plan or refuse absent/inconsistent digest and pad metadata."""

    validate_tower_output(
        VisionTowerRequest(identity=processed.identity, processed=processed),
        tower_output,
    )
    prompt = _token_tuple(prompt_token_ids)
    digests = tuple(image.content_digest for image in processed.images)
    pad_counts = tuple(image.pad_rows for image in processed.images)
    keyed = build_content_key_surrogates(
        prompt,
        image_pad_token_id=image_pad_token_id,
        image_digests=digests,
        pad_counts=pad_counts,
    )
    spans = build_image_spans(
        processed.identity,
        prompt,
        image_pad_token_id=image_pad_token_id,
        image_digests=digests,
        pad_counts=pad_counts,
    )
    if keyed is None or spans is None:
        return None
    digest = _splice_plan_digest(
        processed.identity,
        image_pad_token_id,
        prompt,
        keyed,
        processed.images,
    )
    return VisionSplicePlan(
        identity=processed.identity,
        plan_digest=digest,
        image_pad_token_id=image_pad_token_id,
        prompt_token_ids=prompt,
        keyed_prompt_token_ids=keyed,
        image_spans=spans,
        image_grids=processed.images,
        embeddings=tower_output.embeddings,
    )


def build_content_key_surrogates(
    prompt_token_ids: Sequence[int],
    *,
    image_pad_token_id: int,
    image_digests: Sequence[str] | None,
    pad_counts: Sequence[int] | None,
) -> tuple[int, ...] | None:
    """Return content-true cache keys, or None when identity is incomplete."""

    prompt = _token_tuple(prompt_token_ids)
    normalized = _normalize_layout(image_digests, pad_counts)
    if normalized is None:
        return None
    digests, counts = normalized
    if sum(token == image_pad_token_id for token in prompt) != sum(counts):
        return None
    keyed = list(prompt)
    image_index = 0
    row_in_image = 0
    for position, token in enumerate(keyed):
        if token != image_pad_token_id:
            continue
        while row_in_image >= counts[image_index]:
            image_index += 1
            row_in_image = 0
        payload = (
            _SURROGATE_DOMAIN
            + bytes.fromhex(digests[image_index])
            + row_in_image.to_bytes(8, "big")
        )
        keyed[position] = _SURROGATE_FLAG | int.from_bytes(
            hashlib.sha256(payload).digest(),
            "big",
        )
        row_in_image += 1
    return tuple(keyed)


def build_image_spans(
    identity: VisionRequestIdentity,
    prompt_token_ids: Sequence[int],
    *,
    image_pad_token_id: int,
    image_digests: Sequence[str] | None,
    pad_counts: Sequence[int] | None,
) -> tuple[VisionImageSpan, ...] | None:
    """Return exact contiguous image spans, refusing ambiguous pad layouts."""

    prompt = _token_tuple(prompt_token_ids)
    normalized = _normalize_layout(image_digests, pad_counts)
    if normalized is None:
        return None
    digests, counts = normalized
    positions = tuple(
        index for index, token in enumerate(prompt) if token == image_pad_token_id
    )
    if len(positions) != sum(counts):
        return None
    spans: list[VisionImageSpan] = []
    position_cursor = 0
    for image_index, (digest, count) in enumerate(zip(digests, counts, strict=True)):
        run = positions[position_cursor : position_cursor + count]
        if len(run) != count:
            return None
        expected = tuple(range(run[0], run[0] + count))
        if run != expected:
            return None
        spans.append(
            VisionImageSpan(
                identity=identity,
                image_index=image_index,
                content_digest=digest,
                start=run[0],
                end=run[-1] + 1,
            )
        )
        position_cursor += count
    return tuple(spans)


def _normalize_layout(
    image_digests: Sequence[str] | None,
    pad_counts: Sequence[int] | None,
) -> tuple[tuple[str, ...], tuple[int, ...]] | None:
    if image_digests is None or pad_counts is None:
        return None
    digests = tuple(image_digests)
    try:
        counts = tuple(int(count) for count in pad_counts)
    except (TypeError, ValueError):
        return None
    if not digests or len(digests) != len(counts) or len(digests) > 8:
        return None
    if any(count < 1 for count in counts):
        return None
    for digest in digests:
        if not isinstance(digest, str) or len(digest) != 64:
            return None
        try:
            raw = bytes.fromhex(digest)
        except ValueError:
            return None
        if len(raw) != 32 or digest != digest.lower():
            return None
    return digests, counts


def _token_tuple(token_ids: Sequence[int]) -> tuple[int, ...]:
    try:
        tokens = tuple(int(token) for token in token_ids)
    except (TypeError, ValueError) as exc:
        raise VisionContractError(
            "invalid_prompt_tokens",
            "prompt token ids must be integers",
        ) from exc
    if any(token < 0 for token in tokens):
        raise VisionContractError(
            "invalid_prompt_tokens",
            "prompt token ids must be non-negative",
        )
    return tokens


def _splice_plan_digest(
    identity: VisionRequestIdentity,
    image_pad_token_id: int,
    prompt: tuple[int, ...],
    keyed: tuple[int, ...],
    images: Sequence[ImageGridReceipt],
) -> str:
    return _sha256_json(
        {
            "identity": {
                "response_id": identity.response_id,
                "runtime": _runtime_payload(identity.runtime),
                "bundle_digest": identity.bundle_digest,
            },
            "image_pad_token_id": image_pad_token_id,
            "prompt_token_ids": prompt,
            "keyed_prompt_token_ids": keyed,
            "images": [
                {
                    "index": image.image_index,
                    "digest": image.content_digest,
                    "grid_thw": image.grid_thw,
                    "pad_rows": image.pad_rows,
                }
                for image in images
            ],
        }
    )


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        return False
    try:
        return len(bytes.fromhex(value)) == 32
    except ValueError:
        return False


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
