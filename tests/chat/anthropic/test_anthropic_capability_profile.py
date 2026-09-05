"""Delivery verifier for the Anthropic capability profile and preflight.

Every case below is submitted over real HTTP, in both stream modes, against
two distinct alias/runtime profiles. The verifier proves three things at
once: an unsupported official field is refused with a structured HTTP 400
*before* the turn source is entered and *before* any SSE byte exists, a
supported field still executes, and the two profiles genuinely disagree
where the signed role manifest says they should.

The turn source counts its own entries, so "rejected pre-inference" is a
measured fact here, not a claim about control flow.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from mlx_batch_server.chat.anthropic.anthropic_schema import MessagesRequest
from mlx_batch_server.chat.anthropic.capabilities import (
    DETACHED_ROLE,
    AnthropicCapabilityProfile,
    CapabilityPolicyError,
    CapabilityStatus,
    detached_profile,
    enforce_capabilities,
    resolve_capability_profile,
    role_receipt,
)
from mlx_batch_server.chat.anthropic.errors import (
    REQUEST_ID_FIELD,
    REQUEST_ID_HEADER,
)
from mlx_batch_server.main import app
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

#: Mirrors the two shapes the signed role manifest actually publishes: a
#: fused role that declares tools, and a legacy role that does not.
TOOLED_ALIAS = "flash-main"
TOOLLESS_ALIAS = "vision-legacy"

_ROLE_CAPABILITIES: dict[str, tuple[str, ...]] = {
    "main": ("text", "vision", "tools", "mtp"),
    "vision": ("vision",),
}
_ROLE_BACKENDS: dict[str, str] = {
    "main": "fused_mtp_mlx",
    "vision": "legacy_mlx",
}


class _CountingTurnSource:
    """A turn source that records every entry into inference."""

    def __init__(self) -> None:
        self.entries: list[str] = []

    def stream(self, turn: Any) -> Any:
        self.entries.append(turn.model_alias)

        async def events() -> Any:
            yield TurnStarted(
                response_id="anthropic_capability_probe",
                model=turn.model_alias,
                created_at=1,
            )
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


def _fake_receipt(source: _CountingTurnSource) -> SimpleNamespace:
    """A read-only composition receipt: aliases, roles, and a bound source.

    Nothing here loads a model. The preflight reads exactly the two surfaces
    the real ``RuntimeCompositionReceipt`` already publishes.
    """

    specs = (
        SimpleNamespace(
            name=SimpleNamespace(value="main"),
            capabilities=_ROLE_CAPABILITIES["main"],
            backend=SimpleNamespace(value="fused_mtp_mlx"),
        ),
        SimpleNamespace(
            name=SimpleNamespace(value="vision"),
            capabilities=_ROLE_CAPABILITIES["vision"],
            backend=SimpleNamespace(value="legacy_mlx"),
        ),
    )
    return SimpleNamespace(
        public_aliases=MappingProxyType(
            {TOOLED_ALIAS: "main", TOOLLESS_ALIAS: "vision"}
        ),
        role_directory=SimpleNamespace(specs=lambda: specs),
        anthropic_turn_source=source,
    )


@pytest.fixture
def source() -> _CountingTurnSource:
    return _CountingTurnSource()


@pytest.fixture
def client(source: _CountingTurnSource) -> Iterator[TestClient]:
    previous = getattr(app.state, "responses_runtime", None)
    app.state.responses_runtime = _fake_receipt(source)
    try:
        yield TestClient(app)
    finally:
        if previous is None:
            delattr(app.state, "responses_runtime")
        else:
            app.state.responses_runtime = previous


def _body(alias: str, **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": alias,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hello"}],
    }
    payload.update(overrides)
    return payload


def _sse_events(text: str) -> list[str]:
    return [
        line.removeprefix("event: ").strip()
        for line in text.splitlines()
        if line.startswith("event: ")
    ]


# ---------------------------------------------------------------------------
# The classification matrix
# ---------------------------------------------------------------------------

_TEXT_BLOCK = {"type": "text", "text": "hi"}
_CACHE = {"type": "ephemeral"}
_CUSTOM_TOOL = {
    "name": "lookup",
    "description": "look something up",
    "input_schema": {"type": "object", "properties": {}},
}

#: ``(case id, request overrides, expected wire path fragment)``. Every entry
#: names an official field that has no semantic owner on this runtime.
UNSUPPORTED_CASES: tuple[tuple[str, dict[str, Any], str], ...] = (
    (
        "thinking_enabled",
        {"thinking": {"type": "enabled", "budget_tokens": 2048}},
        "thinking.type",
    ),
    (
        "thinking_continuation",
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "step", "signature": "sig"}
                    ],
                },
                {"role": "user", "content": "go on"},
            ]
        },
        "messages.1.content.0.type",
    ),
    (
        "redacted_thinking",
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": [{"type": "redacted_thinking", "data": "opaque"}],
                },
                {"role": "user", "content": "go on"},
            ]
        },
        "messages.1.content.0.type",
    ),
    (
        "image_input",
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "AAAA",
                            },
                        }
                    ],
                }
            ]
        },
        "messages.0.content.0.source.data",
    ),
    (
        "document_input",
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {"type": "url", "url": "https://x/y.pdf"},
                        }
                    ],
                }
            ]
        },
        "messages.0.content.0.source.url",
    ),
    (
        "server_tool_use",
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "server_tool_use",
                            "id": "srvtoolu_1",
                            "name": "web_search",
                            "input": {"query": "x"},
                        }
                    ],
                },
                {"role": "user", "content": "go on"},
            ]
        },
        "messages.1.content.0.type",
    ),
    (
        "web_search_tool_result",
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "web_search_tool_result",
                            "tool_use_id": "srvtoolu_1",
                            "content": [{"title": "t"}],
                        }
                    ],
                }
            ]
        },
        "messages.0.content.0.type",
    ),
    (
        "container_upload",
        {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "container_upload", "file_id": "file_1"}],
                }
            ]
        },
        "messages.0.content.0.type",
    ),
    (
        "cache_control_on_content",
        {
            "messages": [
                {
                    "role": "user",
                    "content": [{**_TEXT_BLOCK, "cache_control": _CACHE}],
                }
            ]
        },
        "messages.0.content.0.cache_control",
    ),
    (
        "cache_control_on_system",
        {"system": [{"type": "text", "text": "be brief", "cache_control": _CACHE}]},
        "system.0.cache_control",
    ),
    (
        "cache_control_on_tool",
        {"tools": [{**_CUSTOM_TOOL, "cache_control": _CACHE}]},
        "tools.0.cache_control",
    ),
    (
        "cache_control_in_tool_result",
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "lookup",
                            "input": {},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": [{**_TEXT_BLOCK, "cache_control": _CACHE}],
                        }
                    ],
                },
            ]
        },
        "messages.2.content.0.content.0.cache_control",
    ),
    ("container", {"container": "container_1"}, "container"),
    ("inference_geo", {"inference_geo": "us"}, "inference_geo"),
    (
        "output_config_format",
        {"output_config": {"format": {"type": "json_schema"}}},
        "output_config.format",
    ),
    (
        "output_config_effort_high",
        {"output_config": {"effort": "high"}},
        "output_config.effort",
    ),
    (
        "output_config_effort_max",
        {"output_config": {"effort": "max"}},
        "output_config.effort",
    ),
    ("effort_high", {"effort": "high"}, "effort"),
    ("effort_max", {"effort": "max"}, "effort"),
    (
        "citations_enabled",
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {"type": "url", "url": "https://x/y.pdf"},
                            "citations": {"enabled": True},
                        }
                    ],
                }
            ]
        },
        "messages.0.content.0",
    ),
    (
        "hosted_tool_definition",
        {
            "tools": [
                {"type": "web_search_20250305", "name": "web_search"},
            ]
        },
        "tools.0.type",
    ),
)

#: Official fields that do have a named semantic owner and must still run.
SUPPORTED_CASES: tuple[tuple[str, dict[str, Any]], ...] = (
    ("plain_text", {}),
    ("sampling", {"temperature": 0.2, "top_p": 0.9, "top_k": 40}),
    ("stop_sequences", {"stop_sequences": ["STOP"]}),
    ("system_string", {"system": "be brief"}),
    ("system_blocks", {"system": [{"type": "text", "text": "be brief"}]}),
    ("metadata_user_id", {"metadata": {"user_id": "vet-1"}}),
    ("service_tier_auto", {"service_tier": "auto"}),
    ("service_tier_standard_only", {"service_tier": "standard_only"}),
    ("thinking_disabled", {"thinking": {"type": "disabled"}}),
    (
        "custom_tools",
        {"tools": [_CUSTOM_TOOL], "tool_choice": {"type": "auto"}},
    ),
    (
        "tool_result_round_trip",
        {
            "tools": [_CUSTOM_TOOL],
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "lookup",
                            "input": {},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "42",
                        }
                    ],
                },
            ],
        },
    ),
    (
        "search_result_as_untrusted_text",
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "search_result",
                            "source": "https://x",
                            "title": "t",
                            "content": [_TEXT_BLOCK],
                        }
                    ],
                }
            ]
        },
    ),
)


# ---------------------------------------------------------------------------
# HTTP receipts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stream", [False, True], ids=["unary", "stream"])
@pytest.mark.parametrize(
    ("case_id", "overrides", "wire_path"),
    UNSUPPORTED_CASES,
    ids=[case[0] for case in UNSUPPORTED_CASES],
)
def test_unsupported_field_fails_closed_before_inference(
    client: TestClient,
    source: _CountingTurnSource,
    case_id: str,
    overrides: dict[str, Any],
    wire_path: str,
    stream: bool,
) -> None:
    response = client.post(
        MESSAGES_PATH,
        json=_body(TOOLED_ALIAS, stream=stream, **overrides),
    )

    assert response.status_code == 400, case_id
    assert "text/event-stream" not in response.headers.get("content-type", "")
    payload = response.json()
    assert payload["type"] == "error"
    assert payload["error"]["type"] == "invalid_request_error"
    assert wire_path in payload["error"]["message"], payload["error"]["message"]
    request_id = response.headers[REQUEST_ID_HEADER]
    assert payload[REQUEST_ID_FIELD] == request_id
    assert request_id.startswith("req_")
    # The decisive receipt: inference was never entered.
    assert source.entries == []


@pytest.mark.parametrize("stream", [False, True], ids=["unary", "stream"])
@pytest.mark.parametrize(
    ("case_id", "overrides"),
    SUPPORTED_CASES,
    ids=[case[0] for case in SUPPORTED_CASES],
)
def test_supported_field_still_executes(
    client: TestClient,
    source: _CountingTurnSource,
    case_id: str,
    overrides: dict[str, Any],
    stream: bool,
) -> None:
    response = client.post(
        MESSAGES_PATH,
        json=_body(TOOLED_ALIAS, stream=stream, **overrides),
    )

    assert response.status_code == 200, (case_id, response.text)
    assert source.entries == [TOOLED_ALIAS]
    if stream:
        events = _sse_events(response.text)
        assert events[0] == "message_start"
        assert events[-1] == "message_stop"
        assert "error" not in events
    else:
        assert response.json()["type"] == "message"


def test_two_profiles_disagree_on_the_same_field(
    client: TestClient,
    source: _CountingTurnSource,
) -> None:
    """The tool capability is decided by the role, not by the schema."""

    tooled = client.post(
        MESSAGES_PATH,
        json=_body(TOOLED_ALIAS, tools=[_CUSTOM_TOOL]),
    )
    assert tooled.status_code == 200
    assert source.entries == [TOOLED_ALIAS]

    toolless = client.post(
        MESSAGES_PATH,
        json=_body(TOOLLESS_ALIAS, tools=[_CUSTOM_TOOL]),
    )
    assert toolless.status_code == 400
    message = toolless.json()["error"]["message"]
    assert "tools" in message
    assert "runtime role" in message
    # The refused request added no entry.
    assert source.entries == [TOOLED_ALIAS]

    # Plain text still runs on the tool-less profile: the role is narrower,
    # not broken.
    assert client.post(MESSAGES_PATH, json=_body(TOOLLESS_ALIAS)).status_code == 200
    assert source.entries == [TOOLED_ALIAS, TOOLLESS_ALIAS]


@pytest.mark.parametrize("stream", [False, True], ids=["unary", "stream"])
def test_unknown_alias_fails_closed(
    client: TestClient,
    source: _CountingTurnSource,
    stream: bool,
) -> None:
    response = client.post(
        MESSAGES_PATH,
        json=_body("not-configured", stream=stream),
    )

    assert response.status_code == 400
    assert "not configured" in response.json()["error"]["message"]
    assert source.entries == []


def test_a_refused_request_leaves_no_sse_bytes(client: TestClient) -> None:
    response = client.post(
        MESSAGES_PATH,
        json=_body(TOOLED_ALIAS, stream=True, inference_geo="eu"),
    )

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert "event:" not in response.text
    json.loads(response.text)


def test_profile_is_not_inherited_between_requests(
    client: TestClient,
    source: _CountingTurnSource,
) -> None:
    """A tool-bearing request on `main` must not license one on `vision`."""

    assert (
        client.post(
            MESSAGES_PATH, json=_body(TOOLED_ALIAS, tools=[_CUSTOM_TOOL])
        ).status_code
        == 200
    )
    assert (
        client.post(
            MESSAGES_PATH, json=_body(TOOLLESS_ALIAS, tools=[_CUSTOM_TOOL])
        ).status_code
        == 400
    )
    assert (
        client.post(
            MESSAGES_PATH, json=_body(TOOLED_ALIAS, tools=[_CUSTOM_TOOL])
        ).status_code
        == 200
    )
    assert source.entries == [TOOLED_ALIAS, TOOLED_ALIAS]


def test_streaming_and_unary_share_one_classification(
    client: TestClient,
    source: _CountingTurnSource,
) -> None:
    unary = client.post(
        MESSAGES_PATH, json=_body(TOOLED_ALIAS, effort="high", stream=False)
    )
    streamed = client.post(
        MESSAGES_PATH, json=_body(TOOLED_ALIAS, effort="high", stream=True)
    )

    assert unary.status_code == streamed.status_code == 400
    assert unary.json()["error"] == streamed.json()["error"]
    assert source.entries == []


# ---------------------------------------------------------------------------
# The profile itself
# ---------------------------------------------------------------------------


def test_profile_resolves_role_and_backend_without_acquiring_a_model(
    source: _CountingTurnSource,
) -> None:
    receipt = role_receipt(_fake_receipt(source))
    assert receipt is not None

    main = resolve_capability_profile(TOOLED_ALIAS, receipt=receipt)
    vision = resolve_capability_profile(TOOLLESS_ALIAS, receipt=receipt)

    assert (main.role, main.backend) == ("main", "fused_mtp_mlx")
    assert (vision.role, vision.backend) == ("vision", "legacy_mlx")
    assert main.supports("tools") and not vision.supports("tools")
    assert main.media_source_fields == vision.media_source_fields == frozenset()
    assert source.entries == []


def test_detached_role_is_named_not_absent() -> None:
    profile = resolve_capability_profile("anything", receipt=None)

    assert profile.role == DETACHED_ROLE
    assert profile.supports("tools")
    assert profile.supports("content.search_result")
    assert profile.media_source_fields == frozenset()
    assert not profile.supports("thinking.enabled")
    assert detached_profile("anything").role == DETACHED_ROLE


def test_a_blank_alias_is_refused() -> None:
    with pytest.raises(Exception, match="configured alias"):
        resolve_capability_profile("   ", receipt=None)


def test_an_unknown_role_fails_closed(source: _CountingTurnSource) -> None:
    receipt = role_receipt(
        SimpleNamespace(
            public_aliases={"ghost": "not-a-role"},
            role_directory=SimpleNamespace(specs=lambda: ()),
            anthropic_turn_source=source,
        )
    )
    assert receipt is not None

    with pytest.raises(Exception, match="canonical runtime role"):
        resolve_capability_profile("ghost", receipt=receipt)


@pytest.mark.parametrize("alias", [TOOLED_ALIAS, TOOLLESS_ALIAS, "detached"])
def test_every_request_field_carries_a_classification(
    alias: str,
    source: _CountingTurnSource,
) -> None:
    """No field is admitted merely because a Pydantic model contains it."""

    receipt = None if alias == "detached" else role_receipt(_fake_receipt(source))
    profile = resolve_capability_profile(alias, receipt=receipt)

    missing = [
        name for name in MessagesRequest.model_fields if name not in profile.fields
    ]
    assert missing == []
    for entry in profile.fields.values():
        assert entry.status in set(CapabilityStatus)
        assert entry.owner


def test_removing_a_classification_makes_the_request_fail_closed() -> None:
    """The mutation guard: a stripped table must not silently admit."""

    profile = detached_profile(TOOLED_ALIAS)
    stripped = AnthropicCapabilityProfile(
        alias=profile.alias,
        role=profile.role,
        backend=profile.backend,
        declared_capabilities=profile.declared_capabilities,
        media_source_fields=profile.media_source_fields,
        fields=MappingProxyType(
            {
                key: value
                for key, value in profile.fields.items()
                if key != "temperature"
            }
        ),
    )
    request = MessagesRequest.model_validate(_body(TOOLED_ALIAS, temperature=0.5))

    with pytest.raises(CapabilityPolicyError) as failure:
        enforce_capabilities(request, stripped)

    assert failure.value.error_type == "api_error"
    assert failure.value.field_key == "temperature"


def test_the_admission_records_what_was_normalized() -> None:
    profile = detached_profile(TOOLED_ALIAS)
    request = MessagesRequest.model_validate(
        _body(TOOLED_ALIAS, service_tier="auto", system=["one", "two"][0])
    )

    admission = enforce_capabilities(request, profile)

    normalized = {entry.key for entry in admission.normalized}
    assert {"service_tier", "system"} <= normalized
    assert admission.profile is profile
    for entry in admission.admitted:
        assert entry.status is not CapabilityStatus.UNSUPPORTED


def test_the_profile_table_is_read_only() -> None:
    profile = detached_profile(TOOLED_ALIAS)

    with pytest.raises(TypeError):
        profile.fields["temperature"] = profile.fields["top_p"]  # type: ignore[index]
