"""Delivery verifier for Anthropic direct rich-content mapping."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from mlx_batch_server.chat.anthropic.anthropic_schema import MessagesRequest
from mlx_batch_server.chat.anthropic.errors import (
    AnthropicAPIError,
    UnsupportedCapabilityError,
)
from mlx_batch_server.chat.anthropic.request_mapper import build_turn
from mlx_batch_server.runtime.contracts import GenerationRequest, RuntimeKey
from mlx_batch_server.runtime.fusion.concrete.owner import _request_identity
from mlx_batch_server.runtime.fusion.qwen4_exp.request_preparation import (
    _reconstruct_mixed_messages,
)


def _request(content: list[dict[str, Any]]) -> MessagesRequest:
    return MessagesRequest.model_validate(
        {
            "model": "flash-main",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": content}],
        }
    )


def _rich_content() -> list[dict[str, Any]]:
    return [
        {"type": "text", "text": "Compare the supplied records."},
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "aW1hZ2UtYQ==",
            },
        },
        {
            "type": "image",
            "source": {
                "type": "url",
                "media_type": "image/jpeg",
                "url": "https://media.example/scan-b.jpg",
            },
        },
        {
            "type": "document",
            "title": "lab.pdf",
            "context": "Caller labels this as the September laboratory report.",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": "JVBERi0xLjQ=",
            },
        },
        {
            "type": "search_result",
            "source": "https://search.example/result/one",
            "title": "First caller result — źródło",
            "content": [
                {"type": "text", "text": "first passage"},
                {"type": "text", "text": "second passage"},
            ],
        },
        {
            "type": "search_result",
            "source": "file:///caller-supplied/not-a-fetch.txt",
            "title": "Second caller result",
            "content": [{"type": "text", "text": "third passage"}],
        },
    ]


def _generation_request(request: MessagesRequest) -> GenerationRequest:
    turn = build_turn(request)
    return GenerationRequest(
        response_id="anthropic_integrity_probe",
        runtime=RuntimeKey("local/qwen"),
        messages=turn.messages,
        media=turn.media,
    )


def test_official_rich_fixture_preserves_order_identity_and_backend_content() -> None:
    turn = build_turn(_request(_rich_content()))

    assert len(turn.messages) == 1
    assert [item["_content_index"] for item in turn.media] == [1, 2, 4]
    assert [item["type"] for item in turn.media] == [
        "input_image",
        "input_image",
        "input_file",
    ]
    assert turn.media[0]["image_base64"] == ("data:image/png;base64,aW1hZ2UtYQ==")
    assert turn.media[1]["image_url"] == "https://media.example/scan-b.jpg"
    assert turn.media[2]["file_data"] == ("data:application/pdf;base64,JVBERi0xLjQ=")
    assert turn.media[2]["filename"] == "lab.pdf"
    assert turn.media[2]["_anthropic_source"] == {
        "context": "Caller labels this as the September laboratory report.",
        "media_type": "application/pdf",
        "title": "lab.pdf",
        "type": "base64",
    }

    parts, layouts = _reconstruct_mixed_messages(
        _generation_request(_request(_rich_content()))
    )
    assert [part["type"] for part in parts] == [
        "input_text",
        "input_image",
        "input_image",
        "input_text",
        "input_file",
        "input_text",
        "input_text",
    ]
    assert layouts[0].part_indices == tuple(range(7))
    assert "CALLER-SUPPLIED DOCUMENT CONTEXT" in parts[3]["text"]
    assert "CALLER-SUPPLIED UNTRUSTED SEARCH RESULT" in parts[5]["text"]
    assert "First caller result — źródło" in parts[5]["text"]
    assert "https://search.example/result/one" in parts[5]["text"]
    assert "file:///caller-supplied/not-a-fetch.txt" in parts[6]["text"]
    assert all(
        "search.example" not in str(item) and "not-a-fetch" not in str(item)
        for item in turn.media
    )


@pytest.mark.parametrize(
    ("source", "field", "expected"),
    [
        (
            {"type": "url", "url": "https://media.example/a.png"},
            "image_url",
            "https://media.example/a.png",
        ),
        ({"type": "file", "file_id": "file_image_a"}, "file_id", "file_image_a"),
        (
            {
                "type": "base64",
                "media_type": "image/webp",
                "data": "aW1hZ2U=",
            },
            "image_base64",
            "data:image/webp;base64,aW1hZ2U=",
        ),
    ],
)
def test_image_source_kinds_reach_one_canonical_media_owner(
    source: dict[str, str], field: str, expected: str
) -> None:
    turn = build_turn(_request([{"type": "image", "source": source}]))

    assert len(turn.media) == 1
    assert turn.media[0][field] == expected
    assert turn.media[0]["_anthropic_source"]["type"] == source["type"]


@pytest.mark.parametrize(
    ("source", "field", "expected"),
    [
        (
            {"type": "url", "url": "https://media.example/report.pdf"},
            "file_url",
            "https://media.example/report.pdf",
        ),
        ({"type": "file", "file_id": "file_report"}, "file_id", "file_report"),
        (
            {
                "type": "base64",
                "media_type": "text/plain",
                "data": "cmVwb3J0",
            },
            "file_data",
            "data:text/plain;base64,cmVwb3J0",
        ),
    ],
)
def test_document_source_kinds_preserve_canonical_identity(
    source: dict[str, str], field: str, expected: str
) -> None:
    turn = build_turn(
        _request(
            [
                {
                    "type": "document",
                    "title": "report",
                    "context": "caller context",
                    "source": source,
                }
            ]
        )
    )

    assert turn.media[0][field] == expected
    assert turn.media[0]["_anthropic_source"]["type"] == source["type"]
    assert turn.media[0]["_anthropic_source"]["context"] == "caller context"


@pytest.mark.parametrize(
    ("content", "match"),
    [
        (
            [
                {
                    "type": "document",
                    "source": {"type": "url", "url": "https://example/doc.pdf"},
                    "citations": {"enabled": True},
                }
            ],
            "citations.enabled",
        ),
        (
            [
                {
                    "type": "text",
                    "text": "cached",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "cache_control",
        ),
        (
            [
                {
                    "type": "search_result",
                    "source": "https://example/result",
                    "title": "result",
                    "content": [
                        {
                            "type": "text",
                            "text": "cached nested text",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ],
            "content.0.cache_control",
        ),
        (
            [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/zip",
                        "data": "emlw",
                    },
                }
            ],
            "application/zip",
        ),
    ],
)
def test_unhonoured_rich_controls_fail_before_generation(
    content: list[dict[str, Any]], match: str
) -> None:
    with pytest.raises(UnsupportedCapabilityError, match=match):
        build_turn(_request(content))


def test_ambiguous_source_fields_are_not_silently_stripped() -> None:
    with pytest.raises(AnthropicAPIError, match="exactly 'url'"):
        build_turn(
            _request(
                [
                    {
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": "https://media.example/a.png",
                            "file_id": "foreign",
                        },
                    }
                ]
            )
        )


def test_rich_provenance_mutations_change_existing_request_integrity() -> None:
    baseline = _rich_content()
    baseline_identity = _request_identity(_generation_request(_request(baseline)))

    for mutation in ("source", "title", "content"):
        changed = deepcopy(baseline)
        if mutation == "source":
            changed[2]["source"]["url"] = "https://media.example/changed.jpg"
        elif mutation == "title":
            changed[3]["title"] = "changed.pdf"
        else:
            changed[4]["content"][0]["text"] = "changed passage"
        assert _request_identity(_generation_request(_request(changed))) != (
            baseline_identity
        )


def test_w2_nested_tool_result_payload_keeps_its_exact_mapping() -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "flash-main",
            "max_tokens": 32,
            "messages": [
                {"role": "user", "content": "inspect"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "inspect",
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
                            "is_error": False,
                            "content": [
                                {"type": "text", "text": "see photo"},
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "url",
                                        "url": "https://media.example/tool.png",
                                    },
                                },
                            ],
                        }
                    ],
                },
            ],
        }
    )

    turn = build_turn(request)

    assert turn.messages[2] == {
        "type": "function_call_output",
        "role": "tool",
        "call_id": "toolu_1",
        "output": "see photo",
        "content": ({"type": "input_text", "text": "see photo"},),
        "is_error": False,
    }
    assert turn.media == (
        {
            "type": "input_image",
            "_role": "tool",
            "_message_index": 2,
            "_content_index": 1,
            "image_url": "https://media.example/tool.png",
        },
    )


def test_text_only_mapping_remains_newline_joined() -> None:
    turn = build_turn(
        _request(
            [
                {"type": "text", "text": "first"},
                {"type": "text", "text": "second"},
            ]
        )
    )

    assert turn.messages == (
        {
            "role": "user",
            "content": ({"type": "input_text", "text": "first\nsecond"},),
        },
    )
    assert turn.media == ()
