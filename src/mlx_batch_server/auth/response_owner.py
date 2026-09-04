"""Stable, non-secret owner identifiers for the Responses API."""

from __future__ import annotations

import hashlib
from typing import Final, Literal, TypeAlias

ResponseOwnerKind: TypeAlias = Literal["api-key", "session", "hmac", "open"]

_DOMAIN: Final = b"mlx-batch-server/responses-owner/v1\0"
_KINDS: Final = frozenset({"api-key", "session", "hmac", "open"})


def build_response_owner_id(
    *, kind: ResponseOwnerKind, authenticated_subject: str
) -> str:
    """Derive an owner ID from an explicit, already-authenticated subject.

    This function performs no authentication and never reads transport data. The
    caller owns verification of the subject before passing it here.
    """
    if not isinstance(kind, str) or kind not in _KINDS:
        raise ValueError("unsupported response owner kind")
    if not isinstance(authenticated_subject, str):
        raise TypeError("authenticated subject must be a string")
    if not authenticated_subject.strip():
        raise ValueError("authenticated subject must not be blank")

    payload = _DOMAIN + kind.encode("ascii") + b"\0" + authenticated_subject.encode()
    digest = hashlib.sha256(payload).hexdigest()
    return f"resp-owner:v1:{kind}:{digest}"


def build_api_key_response_owner(api_key: str) -> str:
    """Derive an owner from an already-verified API key."""
    return build_response_owner_id(kind="api-key", authenticated_subject=api_key)


def build_session_response_owner(session_id: str) -> str:
    """Derive an owner from the verified session ID, never its claimed user ID."""
    return build_response_owner_id(kind="session", authenticated_subject=session_id)


def build_hmac_response_owner(client_id: str) -> str:
    """Derive an owner from the verified client ID, independent of secret rotation."""
    return build_response_owner_id(kind="hmac", authenticated_subject=client_id)


def build_open_response_owner(client_host: str) -> str:
    """Derive a development owner from the direct client host.

    Behind a reverse proxy, multiple clients may share the same observed host and
    therefore the same owner. Open identity is suitable only for level 0/dev use.
    """
    return build_response_owner_id(kind="open", authenticated_subject=client_host)
