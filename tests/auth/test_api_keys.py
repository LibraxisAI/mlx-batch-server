"""API key issuance / validation / revocation against in-memory backend."""

from __future__ import annotations

import asyncio

from mlx_batch_server.auth import api_keys


def test_issue_validate_revoke_inmem(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://0.0.0.0:1/0")  # unreachable -> falls back
    record = asyncio.run(api_keys.issue_api_key(subject="alice", ttl_hours=1))
    assert record["api_key"].startswith("mlx-")
    assert record["subject"] == "alice"

    assert asyncio.run(api_keys.validate_api_key(record["api_key"])) is True
    assert asyncio.run(api_keys.validate_api_key("mlx-bogus-not-issued")) is False

    assert asyncio.run(api_keys.revoke_api_key(record["api_key"])) is True
    assert asyncio.run(api_keys.validate_api_key(record["api_key"])) is False


def test_validate_returns_false_for_none(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://0.0.0.0:1/0")
    assert asyncio.run(api_keys.validate_api_key(None)) is False
    assert asyncio.run(api_keys.validate_api_key("")) is False


def test_keys_are_stored_as_hashes(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://0.0.0.0:1/0")
    record = asyncio.run(api_keys.issue_api_key(subject="bob", ttl_hours=1))
    plaintext = record["api_key"]
    # Plaintext must NOT appear anywhere in storage.
    for stored in api_keys._memory_store.values():
        assert plaintext not in stored.values()
        assert "api_key" not in stored
        assert "key_hash" in stored
