"""RED contract for stable, non-secret Responses owner identifiers."""

from __future__ import annotations

import hashlib
from typing import Any

import pytest

from mlx_batch_server.auth.response_owner import (
    build_api_key_response_owner,
    build_hmac_response_owner,
    build_open_response_owner,
    build_response_owner_id,
    build_session_response_owner,
)


def _expected(kind: str, subject: str) -> str:
    payload = (
        b"mlx-batch-server/responses-owner/v1\0"
        + kind.encode("ascii")
        + b"\0"
        + subject.encode()
    )
    return f"resp-owner:v1:{kind}:{hashlib.sha256(payload).hexdigest()}"


def test_owner_id_is_deterministic_and_matches_v1_contract():
    first = build_response_owner_id(
        kind="api-key", authenticated_subject="verified-key"
    )
    second = build_response_owner_id(
        kind="api-key", authenticated_subject="verified-key"
    )

    assert first == second == _expected("api-key", "verified-key")


@pytest.mark.parametrize("kind", ["api-key", "session", "hmac", "open"])
def test_supported_kinds_are_explicit(kind: Any):
    owner_id = build_response_owner_id(kind=kind, authenticated_subject="subject")
    assert owner_id == _expected(kind, "subject")


def test_kind_domain_separation_changes_owner_id():
    owners = {
        build_response_owner_id(kind=kind, authenticated_subject="same-subject")
        for kind in ("api-key", "session", "hmac", "open")
    }
    assert len(owners) == 4


def test_subject_separation_changes_owner_id():
    first = build_session_response_owner("session-one")
    second = build_session_response_owner("session-two")
    assert first != second


def test_api_key_secret_is_not_exposed_in_owner_id():
    secret = "sk-private-material-that-must-never-leak"
    owner_id = build_api_key_response_owner(secret)

    assert secret not in owner_id
    assert owner_id == _expected("api-key", secret)


def test_session_identity_uses_session_id_not_user_id():
    session_id = "sess-verified-123"
    user_id = "founder-supplied-user-label"

    owner_id = build_session_response_owner(session_id)

    assert owner_id == _expected("session", session_id)
    assert owner_id != _expected("session", user_id)


def test_hmac_identity_is_stable_by_client_id():
    client_id = "three-more-production"

    assert build_hmac_response_owner(client_id) == build_hmac_response_owner(client_id)
    assert build_hmac_response_owner(client_id) == _expected("hmac", client_id)


def test_open_identity_uses_explicit_client_host():
    owner_id = build_open_response_owner("127.0.0.1")

    assert owner_id == _expected("open", "127.0.0.1")
    assert owner_id != build_open_response_owner("10.0.0.8")


@pytest.mark.parametrize("subject", ["", " ", "\t", "\n"])
def test_blank_subject_fails_closed(subject: str):
    with pytest.raises(ValueError, match="must not be blank"):
        build_response_owner_id(kind="session", authenticated_subject=subject)


@pytest.mark.parametrize("subject", [None, b"bytes", 123])
def test_non_string_subject_fails_closed(subject: Any):
    with pytest.raises(TypeError, match="must be a string"):
        build_response_owner_id(kind="session", authenticated_subject=subject)


@pytest.mark.parametrize("kind", ["", "api_key", "bearer", "SESSION", None])
def test_unknown_or_blank_kind_fails_closed(kind: Any):
    with pytest.raises(ValueError, match="unsupported response owner kind"):
        build_response_owner_id(kind=kind, authenticated_subject="trusted-subject")
