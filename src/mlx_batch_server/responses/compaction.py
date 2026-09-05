"""Process-local authenticated compaction capsules for Responses input."""

from __future__ import annotations

import base64
import binascii
import json
import os
from collections.abc import Mapping, Sequence
from typing import Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

_AAD_PREFIX = b"mlx-batch-responses-compaction-v1\x00"
_CAPSULE_PREFIX = "mlxbr1."


class CompactionError(ValueError):
    """A compaction item is malformed, foreign, or cannot be authenticated."""

    def __init__(self, message: str, *, code: str, param: str = "input") -> None:
        super().__init__(message)
        self.code = code
        self.param = param


class LocalCompactionCodec:
    """Seal canonical messages for this process and authenticated owner only."""

    def __init__(self, key: bytes | None = None) -> None:
        resolved = os.urandom(32) if key is None else bytes(key)
        if len(resolved) != 32:
            raise ValueError("compaction key must contain exactly 32 bytes")
        self._cipher = AESGCM(resolved)

    def seal(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        owner_id: str,
    ) -> str:
        owner = _owner(owner_id)
        payload = json.dumps(
            {
                "version": 1,
                "messages": [dict(message) for message in messages],
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        nonce = os.urandom(12)
        encrypted = self._cipher.encrypt(nonce, payload, _aad(owner))
        encoded = base64.urlsafe_b64encode(nonce + encrypted).decode("ascii")
        return _CAPSULE_PREFIX + encoded.rstrip("=")

    def open(
        self,
        encrypted_content: str,
        *,
        owner_id: str,
    ) -> tuple[Mapping[str, Any], ...]:
        owner = _owner(owner_id)
        if not isinstance(encrypted_content, str) or not encrypted_content.startswith(
            _CAPSULE_PREFIX
        ):
            raise CompactionError(
                "compaction item was not created by this local runtime",
                code="invalid_compaction_item",
            )
        raw = encrypted_content.removeprefix(_CAPSULE_PREFIX)
        try:
            padded = raw + "=" * (-len(raw) % 4)
            sealed = base64.urlsafe_b64decode(padded.encode("ascii"))
            if len(sealed) < 29:
                raise ValueError("sealed compaction item is too short")
            clear = self._cipher.decrypt(sealed[:12], sealed[12:], _aad(owner))
            document = json.loads(clear)
        except (
            InvalidTag,
            ValueError,
            UnicodeError,
            binascii.Error,
            json.JSONDecodeError,
        ):
            raise CompactionError(
                "compaction item is invalid for the authenticated owner",
                code="invalid_compaction_item",
            ) from None
        if not isinstance(document, Mapping) or document.get("version") != 1:
            raise CompactionError(
                "compaction item has an unsupported version",
                code="invalid_compaction_item",
            )
        messages = document.get("messages")
        if not isinstance(messages, list) or not all(
            isinstance(message, Mapping) for message in messages
        ):
            raise CompactionError(
                "compaction item does not contain canonical messages",
                code="invalid_compaction_item",
            )
        return tuple(dict(message) for message in messages)


def expand_compaction_input(
    raw_input: Any,
    *,
    owner_id: str,
    codec: LocalCompactionCodec | None,
) -> Any:
    """Replace one official compaction item with its authenticated context."""

    if not isinstance(raw_input, Sequence) or isinstance(raw_input, str | bytes):
        return raw_input
    items = list(raw_input)
    offsets = [
        index
        for index, item in enumerate(items)
        if isinstance(item, Mapping) and item.get("type") == "compaction"
    ]
    if not offsets:
        return raw_input
    if len(offsets) != 1:
        raise CompactionError(
            "input may contain exactly one compaction item",
            code="invalid_compaction_item",
        )
    if codec is None:
        raise CompactionError(
            "compaction input is unavailable on this runtime",
            code="unsupported_input_item",
        )
    offset = offsets[0]
    item = items[offset]
    if not isinstance(item, Mapping):  # pragma: no cover - selected above
        raise CompactionError(
            "compaction item must be a mapping",
            code="invalid_compaction_item",
        )
    unknown = set(item) - {"type", "id", "encrypted_content", "created_by"}
    encrypted = item.get("encrypted_content")
    if unknown or not isinstance(encrypted, str) or not encrypted:
        raise CompactionError(
            "compaction item has invalid fields",
            code="invalid_compaction_item",
            param=f"input[{offset}]",
        )
    restored = codec.open(encrypted, owner_id=owner_id)
    expected_prefix = compacted_user_messages(restored)
    if items[:offset] != list(expected_prefix):
        raise CompactionError(
            "user-message prefix does not match the compaction item",
            code="invalid_compaction_item",
            param=f"input[{offset}]",
        )
    return [*restored, *items[offset + 1 :]]


def compacted_user_messages(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Render the user-message prefix required by the compact response shape."""

    rendered: list[Mapping[str, Any]] = []
    for index, message in enumerate(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, Sequence) or isinstance(content, str | bytes):
            raise CompactionError(
                "canonical user message content is invalid",
                code="invalid_compaction_state",
            )
        rendered.append(
            {
                "id": f"msg_compact_{index}",
                "type": "message",
                "role": "user",
                "status": "completed",
                "content": [dict(part) for part in content],
            }
        )
    return tuple(rendered)


def _owner(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CompactionError(
            "owner_id must be a non-empty string",
            code="invalid_owner_id",
            param="owner_id",
        )
    return value.strip()


def _aad(owner_id: str) -> bytes:
    return _AAD_PREFIX + owner_id.encode("utf-8")


__all__ = [
    "CompactionError",
    "LocalCompactionCodec",
    "compacted_user_messages",
    "expand_compaction_input",
]
