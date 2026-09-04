# SPDX-License-Identifier: Apache-2.0
# Derived from youssofal/mtplx@6d0ddf0575faa9acf77e63c57e48ea1602a7e4ab
# mtplx/vision/splice.py; adapted to target-owned per-request cursors.
# Modified by LibraxisAI for explicit tensor windows and owner-thread affinity.
"""Executable tensor splice behind the immutable vision splice plan."""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence

import mlx.core as mx

from .processing import VisionContractError, VisionRequestIdentity
from .splice import VisionSpliceCursor, VisionSplicePlan, VisionSpliceWindow


class Qwen4ExpTensorSplicer:
    """Mutable tensor splice state owned by one request and one thread."""

    __slots__ = ("_cursor", "_owner_thread_id", "_plan")

    def __init__(self, plan: VisionSplicePlan) -> None:
        if not isinstance(plan, VisionSplicePlan):
            raise VisionContractError(
                "invalid_splice_plan",
                "tensor splicer requires a VisionSplicePlan",
            )
        embedding_shape = _shape(plan.embeddings.handle, "vision embeddings")
        if len(embedding_shape) != 2 or embedding_shape[0] != plan.embeddings.row_count:
            raise VisionContractError(
                "splice_embedding_shape_mismatch",
                "vision embeddings must have shape [receipt rows, hidden]",
            )
        self._plan = plan
        self._cursor = VisionSpliceCursor(plan)
        self._owner_thread_id = threading.get_ident()

    @property
    def prompt_cursor(self) -> int:
        return self._cursor.prompt_cursor

    @property
    def row_cursor(self) -> int:
        return self._cursor.row_cursor

    @property
    def remaining_rows(self) -> int:
        return self._cursor.remaining_rows

    def splice_chunk(
        self,
        *,
        identity: VisionRequestIdentity,
        plan_digest: str,
        token_ids: Sequence[int],
        token_tensor: mx.array,
        embed_tokens: Callable[[mx.array], mx.array],
    ) -> mx.array | None:
        """Consume one sequential prefill chunk and splice exact image rows."""

        self._assert_owner()
        tokens = _validate_token_tensor(token_ids, token_tensor)
        window = self._cursor.consume(
            identity=identity,
            plan_digest=plan_digest,
            token_ids=tokens,
        )
        if window is None:
            return None
        return self._splice_window(token_tensor, embed_tokens, window)

    def splice_history_window(
        self,
        *,
        identity: VisionRequestIdentity,
        plan_digest: str,
        prompt_start: int,
        token_ids: Sequence[int],
        token_tensor: mx.array,
        embed_tokens: Callable[[mx.array], mx.array],
    ) -> mx.array | None:
        """Splice an MTP/history window without advancing prefill state."""

        self._assert_owner()
        tokens = _validate_token_tensor(token_ids, token_tensor)
        window = self._cursor.lookup_window(
            identity=identity,
            plan_digest=plan_digest,
            prompt_start=prompt_start,
            token_ids=tokens,
        )
        if window is None:
            return None
        return self._splice_window(token_tensor, embed_tokens, window)

    def reset(
        self,
        *,
        identity: VisionRequestIdentity,
        plan_digest: str,
    ) -> None:
        self._assert_owner()
        self._cursor.reset(identity=identity, plan_digest=plan_digest)

    def assert_complete(
        self,
        *,
        identity: VisionRequestIdentity,
        plan_digest: str,
    ) -> None:
        self._assert_owner()
        self._cursor.assert_complete(
            identity=identity,
            plan_digest=plan_digest,
        )

    def _splice_window(
        self,
        token_tensor: mx.array,
        embed_tokens: Callable[[mx.array], mx.array],
        window: VisionSpliceWindow,
    ) -> mx.array:
        embedded = embed_tokens(token_tensor)
        embedded_shape = _shape(embedded, "token embeddings")
        token_shape = _shape(token_tensor, "token tensor")
        if (
            len(embedded_shape) != 3
            or embedded_shape[0] != 1
            or embedded_shape[1] != token_shape[1]
        ):
            raise VisionContractError(
                "invalid_token_embeddings",
                "token embeddings must have shape [1, sequence, hidden]",
            )
        rows = self._plan.embeddings.handle[window.row_start : window.row_end]
        row_shape = _shape(rows, "vision splice rows")
        if (
            len(row_shape) != 2
            or row_shape[0] != len(window.pad_offsets)
            or row_shape[1] != embedded_shape[2]
        ):
            raise VisionContractError(
                "splice_row_shape_mismatch",
                "vision rows must match pad count and token embedding width",
            )
        positions = mx.array(window.pad_offsets, dtype=mx.int32)
        flattened = embedded.reshape(embedded_shape[1], embedded_shape[2])
        flattened[positions] = rows.astype(embedded.dtype)
        return flattened.reshape(1, embedded_shape[1], embedded_shape[2])

    def _assert_owner(self) -> None:
        if threading.get_ident() != self._owner_thread_id:
            raise VisionContractError(
                "owner_thread_violation",
                "tensor splice must run on its request owner thread",
            )


def _validate_token_tensor(
    token_ids: Sequence[int],
    token_tensor: mx.array,
) -> tuple[int, ...]:
    try:
        tokens = tuple(int(token) for token in token_ids)
    except (TypeError, ValueError) as exc:
        raise VisionContractError(
            "invalid_splice_tokens",
            "splice token ids must be integers",
        ) from exc
    shape = _shape(token_tensor, "token tensor")
    if len(shape) != 2 or shape[0] != 1 or shape[1] != len(tokens):
        raise VisionContractError(
            "invalid_token_tensor_shape",
            "token tensor must have shape [1, len(token_ids)]",
        )
    try:
        tensor_tokens = tuple(int(value) for value in token_tensor[0].tolist())
    except (TypeError, ValueError) as exc:
        raise VisionContractError(
            "invalid_token_tensor",
            "token tensor values must be integral",
        ) from exc
    if tensor_tokens != tokens:
        raise VisionContractError(
            "token_tensor_mismatch",
            "token tensor must exactly match the cursor token ids",
        )
    return tokens


def _shape(value: object, label: str) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise VisionContractError(
            "invalid_tensor_handle",
            f"{label} is not an MLX tensor-like value",
        )
    try:
        return tuple(int(item) for item in shape)
    except (TypeError, ValueError) as exc:
        raise VisionContractError(
            "invalid_tensor_handle",
            f"{label} has an invalid shape",
        ) from exc
