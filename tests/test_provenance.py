from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from mlx_batch_server import main, provenance
from mlx_batch_server.chat.openai.models import models as models_module


@pytest.fixture(autouse=True)
def _reset_provenance(monkeypatch):
    monkeypatch.delenv(provenance.SOURCE_SHA_ENV, raising=False)
    monkeypatch.delenv(provenance.SOURCE_DIRTY_ENV, raising=False)
    provenance.get_runtime_provenance.cache_clear()
    yield
    provenance.get_runtime_provenance.cache_clear()


def test_start_stamps_checkout_before_uvicorn_admission(monkeypatch):
    sha = "a" * 40
    calls: list[tuple[str, str | None]] = []

    def stamp():
        monkeypatch.setenv(provenance.SOURCE_SHA_ENV, sha)
        monkeypatch.setenv(provenance.SOURCE_DIRTY_ENV, "1")
        calls.append(("stamp", None))
        return provenance.RuntimeProvenance(sha, True)

    def run(*args, **kwargs):
        calls.append(("uvicorn", provenance.os.environ.get(provenance.SOURCE_SHA_ENV)))

    monkeypatch.setattr(main, "stamp_runtime_environment", stamp)
    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=run))
    monkeypatch.setattr(sys, "argv", ["mlx-batch-server"])

    main.start()

    assert calls == [("stamp", None), ("uvicorn", sha)]


def test_direct_app_construction_freezes_provenance_at_start(monkeypatch):
    calls = 0

    def get_provenance():
        nonlocal calls
        calls += 1
        return provenance.RuntimeProvenance("f" * 40, False)

    monkeypatch.setattr(main, "get_runtime_provenance", get_provenance)

    main.create_app()

    assert calls == 1


def test_checkout_provenance_is_stamped_once_and_frozen(monkeypatch):
    sha = "b" * 40
    reads = 0

    def read_checkout(_repo_root):
        nonlocal reads
        reads += 1
        return provenance.RuntimeProvenance(sha, True)

    monkeypatch.setattr(provenance, "_read_git_checkout", read_checkout)

    first = provenance.get_runtime_provenance()
    monkeypatch.setenv(provenance.SOURCE_SHA_ENV, "c" * 40)
    second = provenance.get_runtime_provenance()

    assert first == provenance.RuntimeProvenance(sha, True)
    assert second is first
    assert reads == 1


def test_explicit_build_stamp_does_not_inspect_git(monkeypatch):
    sha = "d" * 40
    monkeypatch.setenv(provenance.SOURCE_SHA_ENV, sha)
    monkeypatch.setenv(provenance.SOURCE_DIRTY_ENV, "0")

    def unexpected_git(_repo_root):
        raise AssertionError("authoritative build stamp must bypass git")

    monkeypatch.setattr(provenance, "_read_git_checkout", unexpected_git)

    assert provenance.get_runtime_provenance() == provenance.RuntimeProvenance(
        sha, False
    )


@pytest.mark.asyncio
async def test_health_reports_frozen_source_provenance(monkeypatch):
    sha = "e" * 40
    monkeypatch.setattr(
        models_module,
        "get_runtime_provenance",
        lambda: provenance.RuntimeProvenance(sha, True),
    )

    payload = await models_module.health_check()

    assert payload["source_sha"] == sha
    assert payload["source_dirty"] is True
