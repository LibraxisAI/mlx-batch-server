"""Session manager: create/validate/extend/expire flow (memory backend)."""

from __future__ import annotations

import asyncio

from mlx_batch_server.auth.session import session_auth


def test_create_and_validate_session():
    sid = asyncio.run(session_auth.create_session(user_id="alice", user_tier="default"))
    assert sid

    info = asyncio.run(session_auth.validate_session(sid))
    assert info is not None
    assert info["user_id"] == "alice"
    assert info["custom_metadata"]["user_tier"] == "default"


def test_extend_then_delete_session():
    sid = asyncio.run(session_auth.create_session(user_id="bob"))
    info = asyncio.run(session_auth.validate_session(sid))
    original_exp = info["expires_at"]

    assert asyncio.run(session_auth.extend_session(sid, hours=48)) is True
    info2 = asyncio.run(session_auth.get_session_info(sid))
    assert info2["expires_at"] >= original_exp

    assert asyncio.run(session_auth.delete_session(sid)) is True
    assert asyncio.run(session_auth.validate_session(sid)) is None


def test_rate_limit_blocks_after_threshold():
    """Lower the limit and verify check_rate_limit flips."""
    session_auth.rate_limit_configs["default"].requests_per_minute = 2
    try:
        assert asyncio.run(session_auth.check_rate_limit("user-x")) is True
        assert asyncio.run(session_auth.check_rate_limit("user-x")) is True
        assert asyncio.run(session_auth.check_rate_limit("user-x")) is False
    finally:
        session_auth.rate_limit_configs["default"].requests_per_minute = 60
