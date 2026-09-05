"""HTTP verifier for source-specific Anthropic rich-input admission."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from mlx_batch_server.chat.anthropic import router as anthropic_router
from mlx_batch_server.chat.anthropic.turn_source import clear_turn_source
from mlx_batch_server.main import app
from mlx_batch_server.responses.runtime_bootstrap import (
    RoleRuntimeCompositionReceipt,
    compose_role_responses_runtime,
)
from mlx_batch_server.runtime.contracts import RoleName
from mlx_batch_server.runtime.events import (
    TEXT_CONTENT_KIND,
    ContentPartStarted,
    TextCompleted,
    TextDelta,
    TurnCompleted,
    TurnStarted,
    UsageUpdate,
)

MESSAGES_PATH = "/anthropic/v1/messages"
ALIAS = "flash-main"


class _DormantExecutionFactory:
    def __init__(self) -> None:
        self.calls: list[tuple[object, object, object]] = []

    def prepare(
        self,
        runtime: object,
        config: object,
        scheduler_config: object,
    ) -> Any:
        self.calls.append((runtime, config, scheduler_config))
        raise AssertionError("HTTP preflight must not acquire a model")


class _DormantFileIdResolver:
    def __init__(self) -> None:
        self.calls: list[object] = []

    async def resolve(self, request: object) -> Any:
        self.calls.append(request)
        raise AssertionError("HTTP admission must not resolve file identities")


class _RecordingTurnSource:
    def __init__(self) -> None:
        self.entries: list[Any] = []

    def stream(self, turn: Any) -> Any:
        self.entries.append(turn)

        async def events() -> Any:
            yield TurnStarted(response_id="rich_input_probe", model=ALIAS, created_at=1)
            yield ContentPartStarted(
                kind=TEXT_CONTENT_KIND,
                output_index=0,
                content_index=0,
                item_id="text_0",
            )
            yield TextDelta(
                delta="ok",
                item_id="text_0",
                output_index=0,
                content_index=0,
            )
            yield TextCompleted(
                text="ok",
                item_id="text_0",
                output_index=0,
                content_index=0,
            )
            yield TurnCompleted(
                finish_reason="stop",
                usage=UsageUpdate(input_tokens=1, output_tokens=1, total_tokens=2),
            )

        return events()


def _runtime(
    source: _RecordingTurnSource,
    *,
    allowed_url_origins: tuple[str, ...] = (),
    file_id_resolver: _DormantFileIdResolver | None = None,
) -> tuple[RoleRuntimeCompositionReceipt, _DormantExecutionFactory]:
    execution = _DormantExecutionFactory()
    composed = compose_role_responses_runtime(
        process_role=RoleName.MAIN,
        public_aliases={ALIAS: RoleName.MAIN},
        allowed_url_origins=allowed_url_origins,
        file_id_resolver=file_id_resolver,
        execution_factory=execution,
    )
    responses = replace(composed.responses, anthropic_turn_source=source)
    return replace(composed, responses=responses), execution


@contextmanager
def _client(runtime: RoleRuntimeCompositionReceipt) -> Iterator[TestClient]:
    previous = getattr(app.state, "responses_runtime", None)
    app.state.responses_runtime = runtime
    try:
        with TestClient(app) as client:
            yield client
    finally:
        clear_turn_source()
        if previous is None:
            delattr(app.state, "responses_runtime")
        else:
            app.state.responses_runtime = previous


def _body(content: list[dict[str, Any]], *, stream: bool) -> dict[str, Any]:
    return {
        "model": ALIAS,
        "max_tokens": 16,
        "stream": stream,
        "messages": [{"role": "user", "content": content}],
    }


SUPPORTED_DEFAULT: tuple[tuple[str, dict[str, Any], str], ...] = (
    (
        "image_url",
        {"type": "image", "source": {"type": "url", "url": "https://x/a.png"}},
        "image_url",
    ),
    (
        "image_base64",
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "AAAA",
            },
        },
        "image_base64",
    ),
    (
        "document_file_data",
        {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": "JVBERg==",
            },
        },
        "file_data",
    ),
)


@pytest.mark.parametrize("stream", [False, True], ids=["unary", "stream"])
@pytest.mark.parametrize(
    ("case_id", "block", "canonical_field"),
    SUPPORTED_DEFAULT,
    ids=[case[0] for case in SUPPORTED_DEFAULT],
)
def test_default_composition_admits_only_its_exact_source_field(
    case_id: str,
    block: dict[str, Any],
    canonical_field: str,
    stream: bool,
) -> None:
    source = _RecordingTurnSource()
    runtime, execution = _runtime(source)

    with _client(runtime) as client:
        response = client.post(MESSAGES_PATH, json=_body([block], stream=stream))

    assert response.status_code == 200, (case_id, response.text)
    assert len(source.entries) == 1
    assert canonical_field in source.entries[0].media[0]
    assert execution.calls == []
    if stream:
        assert response.headers["content-type"].startswith("text/event-stream")
        assert "event: message_stop" in response.text


UNSUPPORTED_DEFAULT: tuple[tuple[str, dict[str, Any], str], ...] = (
    (
        "image_file_id",
        {"type": "image", "source": {"type": "file", "file_id": "file_image"}},
        "messages.0.content.0.source.file_id",
    ),
    (
        "document_file_url",
        {
            "type": "document",
            "source": {"type": "url", "url": "https://x/report.pdf"},
        },
        "messages.0.content.0.source.url",
    ),
    (
        "document_file_id",
        {
            "type": "document",
            "source": {"type": "file", "file_id": "file_report"},
        },
        "messages.0.content.0.source.file_id",
    ),
    (
        "document_plaintext",
        {
            "type": "document",
            "source": {"type": "text", "data": "not a canonical file"},
        },
        "messages.0.content.0.source.type",
    ),
)


@pytest.mark.parametrize("stream", [False, True], ids=["unary", "stream"])
@pytest.mark.parametrize(
    ("case_id", "block", "wire_path"),
    UNSUPPORTED_DEFAULT,
    ids=[case[0] for case in UNSUPPORTED_DEFAULT],
)
def test_unsupported_source_fails_before_mapper_model_and_sse(
    monkeypatch: pytest.MonkeyPatch,
    case_id: str,
    block: dict[str, Any],
    wire_path: str,
    stream: bool,
) -> None:
    source = _RecordingTurnSource()
    runtime, execution = _runtime(source)
    mapper_calls: list[object] = []

    def forbidden_mapper(request: object) -> Any:
        mapper_calls.append(request)
        raise AssertionError("unsupported source reached build_turn")

    monkeypatch.setattr(anthropic_router, "build_turn", forbidden_mapper)
    with _client(runtime) as client:
        response = client.post(MESSAGES_PATH, json=_body([block], stream=stream))

    assert response.status_code == 400, case_id
    assert response.headers["content-type"].startswith("application/json")
    assert "text/event-stream" not in response.headers["content-type"]
    assert "event:" not in response.text
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert wire_path in response.json()["error"]["message"]
    assert mapper_calls == []
    assert source.entries == []
    assert execution.calls == []


@pytest.mark.parametrize("stream", [False, True], ids=["unary", "stream"])
def test_url_and_file_id_forms_require_their_canonical_wiring(stream: bool) -> None:
    file_resolver = _DormantFileIdResolver()
    source = _RecordingTurnSource()
    runtime, execution = _runtime(
        source,
        allowed_url_origins=("https://x",),
        file_id_resolver=file_resolver,
    )
    content = [
        {
            "type": "document",
            "source": {"type": "url", "url": "https://x/report.pdf"},
        },
        {"type": "image", "source": {"type": "file", "file_id": "image_1"}},
        {
            "type": "document",
            "source": {"type": "file", "file_id": "document_1"},
        },
    ]

    with _client(runtime) as client:
        response = client.post(MESSAGES_PATH, json=_body(content, stream=stream))

    assert response.status_code == 200, response.text
    assert [
        set(item) & {"file_url", "file_id"} for item in source.entries[0].media
    ] == [
        {"file_url"},
        {"file_id"},
        {"file_id"},
    ]
    assert file_resolver.calls == []
    assert execution.calls == []


@pytest.mark.parametrize("stream", [False, True], ids=["unary", "stream"])
def test_search_result_is_only_delimited_untrusted_text(stream: bool) -> None:
    source = _RecordingTurnSource()
    runtime, execution = _runtime(source)
    search_url = "https://must-not-fetch.example/result"
    block = {
        "type": "search_result",
        "source": search_url,
        "title": "caller result",
        "content": [{"type": "text", "text": "untrusted passage"}],
    }

    with _client(runtime) as client:
        response = client.post(MESSAGES_PATH, json=_body([block], stream=stream))

    assert response.status_code == 200, response.text
    assert len(source.entries) == 1
    turn = source.entries[0]
    assert turn.media == ()
    assert "CALLER-SUPPLIED UNTRUSTED SEARCH RESULT" in str(turn.messages)
    assert search_url in str(turn.messages)
    assert execution.calls == []


UNHONOURED_RICH_CONTROLS: tuple[tuple[dict[str, Any], str], ...] = (
    (
        {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": "JVBERg==",
            },
            "citations": {"enabled": True},
        },
        "messages.0.content.0.citations.enabled",
    ),
    (
        {
            "type": "image",
            "source": {"type": "url", "url": "https://x/a.png"},
            "cache_control": {"type": "ephemeral"},
        },
        "messages.0.content.0.cache_control",
    ),
    (
        {
            "type": "search_result",
            "source": "https://must-not-fetch.example/result",
            "title": "caller result",
            "content": [
                {
                    "type": "text",
                    "text": "untrusted passage",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        "messages.0.content.0.content.0.cache_control",
    ),
)


@pytest.mark.parametrize("stream", [False, True], ids=["unary", "stream"])
@pytest.mark.parametrize(("block", "wire_path"), UNHONOURED_RICH_CONTROLS)
def test_rich_controls_remain_pre_sse_field_specific_refusals(
    block: dict[str, Any], wire_path: str, stream: bool
) -> None:
    source = _RecordingTurnSource()
    runtime, execution = _runtime(source)

    with _client(runtime) as client:
        response = client.post(MESSAGES_PATH, json=_body([block], stream=stream))

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert "event:" not in response.text
    assert wire_path in response.json()["error"]["message"]
    assert source.entries == []
    assert execution.calls == []


def test_coarse_image_capability_cannot_admit_an_unreceipted_source() -> None:
    source = _RecordingTurnSource()
    runtime, _ = _runtime(source)
    receipt = anthropic_router.role_receipt(runtime)
    assert receipt is not None
    profile = anthropic_router.resolve_capability_profile(ALIAS, receipt=receipt)

    assert not profile.supports("content.image")
    assert not profile.supports("content.image.file_id")
    assert profile.supports("content.image.image_url")
    assert profile.supports("content.image.image_base64")
