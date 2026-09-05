"""RED contracts for the canonical Responses request mapper.

These tests are source-only while the Compile Embargo remains HOLD.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from mlx_batch_server.responses.compaction import (
    LocalCompactionCodec,
    compacted_user_messages,
)
from mlx_batch_server.responses.runtime_mapper import (
    CanonicalResponsesMapper,
    ResolvedRuntime,
    ResponsesMappingError,
)
from mlx_batch_server.runtime.contracts import BackendKind, RuntimeKey
from mlx_batch_server.runtime.fusion.qwen4_exp.request_preparation import (
    _reconstruct_mixed_messages,
)

INSPECT_CALL_ID = "${INSPECT_CANVAS_CALL_ID}"
INSPECT_ARGUMENTS = '{"target":"canvas"}'
_CALL_LINEAGE = {
    "type": "function_call",
    "role": "assistant",
    "call_id": INSPECT_CALL_ID,
    "name": "inspect_canvas",
    "arguments": INSPECT_ARGUMENTS,
    "id": "fc_inspect_canvas",
    "status": "completed",
}
_CALL_TEXT = json.dumps(
    {
        "type": "function_call",
        "call_id": INSPECT_CALL_ID,
        "name": "inspect_canvas",
        "arguments": INSPECT_ARGUMENTS,
    },
    ensure_ascii=False,
    separators=(",", ":"),
)

if TYPE_CHECKING:
    from mlx_batch_server.responses.controller import PreparedResponse


class _Projection:
    def observe(self, event: Any) -> None:
        del event

    def terminal_envelope(self) -> Mapping[str, Any]:
        return {"id": "unused", "status": "completed"}


class _Resolver:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.override: RuntimeKey | None = None

    def __call__(self, **kwargs: Any) -> RuntimeKey:
        self.calls.append(dict(kwargs))
        if self.override is not None:
            return self.override
        return RuntimeKey(
            model_id=kwargs["model"],
            revision=kwargs["revision"],
            adapter_path=kwargs["adapter_path"],
            draft_model_id=kwargs["draft_model_id"],
            backend=BackendKind.FUSED_MTP_MLX,
        )


def _mapper(
    resolver: _Resolver | None = None,
) -> tuple[CanonicalResponsesMapper, _Resolver, list[PreparedResponse]]:
    exact = resolver or _Resolver()
    projected: list[PreparedResponse] = []

    def projection_factory(prepared: PreparedResponse) -> _Projection:
        projected.append(prepared)
        return _Projection()

    return (
        CanonicalResponsesMapper(
            resolve_runtime=exact,
            projection_factory=projection_factory,
        ),
        exact,
        projected,
    )


def _prepare(
    mapper: CanonicalResponsesMapper,
    payload: Mapping[str, Any],
    *,
    parents: tuple[Mapping[str, Any], ...] = (),
) -> PreparedResponse:
    return mapper.prepare(
        payload,
        response_id="resp_owned",
        owner_id="principal:owner",
        parent_messages=parents,
    )


def test_materializes_all_instructions_before_parent_and_current_conversation() -> None:
    mapper, _, _ = _mapper()
    parent = {"role": "assistant", "content": "remembered"}

    prepared = _prepare(
        mapper,
        {
            "model": "flash-next",
            "instructions": "Be exact.",
            "input": [
                {"role": "developer", "content": "Use the lab schema."},
                {"role": "user", "content": "Read this result."},
            ],
        },
        parents=(parent,),
    )

    assert [item["role"] for item in prepared.materialized_messages] == [
        "developer",
        "developer",
        "assistant",
        "user",
    ]
    assert prepared.materialized_messages[0]["content"][0]["text"] == "Be exact."
    assert prepared.materialized_messages[2]["content"][0]["text"] == "remembered"
    assert [item["role"] for item in prepared.lineage_messages or ()] == [
        "developer",
        "assistant",
        "user",
    ]
    assert all(
        part.get("text") != "Be exact."
        for item in prepared.lineage_messages or ()
        for part in item["content"]
    )
    assert prepared.request.lineage[0]["content"][0]["text"] == "remembered"
    assert prepared.request.messages == tuple(
        {
            "role": item["role"],
            "content": tuple(
                part for part in item["content"] if part["type"] == "input_text"
            ),
        }
        for item in prepared.materialized_messages
    )


def test_preserves_multiple_media_parts_without_stringifying_them() -> None:
    mapper, _, _ = _mapper()
    image_a = "data:image/png;base64,AAAA"
    image_b = "https://example.test/right.png"
    file_a = "file_lab_pdf"
    file_b = "data:application/pdf;base64,JVBERg=="

    prepared = _prepare(
        mapper,
        {
            "model": "flash-next",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Compare all records."},
                        {"type": "input_image", "image_url": image_a, "detail": "high"},
                        {"type": "input_image", "image_url": image_b},
                        {"type": "input_file", "file_id": file_a, "filename": "a.pdf"},
                        {
                            "type": "input_file",
                            "file_data": file_b,
                            "filename": "b.pdf",
                        },
                        {"type": "input_audio", "audio_url": "file_audio"},
                        {"type": "input_video", "video_url": "file_video"},
                    ],
                }
            ],
        },
    )

    assert [part["type"] for part in prepared.request.media] == [
        "input_image",
        "input_image",
        "input_file",
        "input_file",
        "input_audio",
        "input_video",
    ]
    assert prepared.request.media[0]["image_base64"] == image_a
    assert prepared.request.media[1]["image_url"] == image_b
    assert prepared.request.media[2]["file_id"] == file_a
    assert prepared.request.media[3]["file_data"] == file_b
    assert all(part["_role"] == "user" for part in prepared.request.media)
    assert [part["_content_index"] for part in prepared.request.media] == [
        1,
        2,
        3,
        4,
        5,
        6,
    ]
    assert prepared.request.messages[0]["content"] == (
        {"type": "input_text", "text": "Compare all records."},
    )
    assert image_a not in str(prepared.request.messages)
    assert file_b not in str(prepared.request.messages)


def test_preserves_function_output_before_typed_message_and_image() -> None:
    mapper, _, _ = _mapper()
    fixture_path = Path(__file__).parents[1] / "fixtures" / "three_more_north_star.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    command = fixture["commands"]["photo_tool_continuation"]
    payload = {
        key: value for key, value in command.items() if key not in {"type", "stream_id"}
    }

    # The command carries previous_response_id, so the matching call reaches the
    # mapper as inherited lineage exactly as the registry materializes it.
    prepared = _prepare(mapper, payload, parents=(_CALL_LINEAGE,))

    assert prepared.request.messages[0] == {
        "type": "message",
        "role": "assistant",
        "content": ({"type": "input_text", "text": _CALL_TEXT},),
    }
    function_output = prepared.request.messages[1]
    assert function_output == {
        "type": "function_call_output",
        "role": "tool",
        "call_id": "${INSPECT_CANVAS_CALL_ID}",
        "output": "${INSPECT_CANVAS_JSON_RECEIPT}",
        "content": (
            {
                "type": "input_text",
                "text": "${INSPECT_CANVAS_JSON_RECEIPT}",
            },
        ),
    }
    assert prepared.request.messages[2] == {
        "type": "message",
        "role": "user",
        "content": (
            {
                "type": "input_text",
                "text": "Widok kanwy po inspect_canvas. Ocen go wzrokiem.",
            },
        ),
    }
    assert prepared.request.media == (
        {
            "type": "input_image",
            "image_url": "${CANVAS_VIEW_DATA_URL}",
            "detail": "high",
            "_role": "user",
            "_message_index": 2,
            "_content_index": 1,
        },
    )
    assert [item["type"] for item in prepared.materialized_messages] == [
        "function_call",
        "function_call_output",
        "message",
    ]
    assert payload["input"][0]["output"] == "${INSPECT_CANVAS_JSON_RECEIPT}"
    assert (
        json.loads(json.dumps(list(prepared.request.messages)))[1]["call_id"]
        == "${INSPECT_CANVAS_CALL_ID}"
    )
    with pytest.raises(TypeError):
        function_output["call_id"] = "call_changed"  # type: ignore[index]

    # Delivery proof: the fused mixed-content preparer seals this exact request.
    _parts, layouts = _reconstruct_mixed_messages(prepared.request)
    assert [layout.item_type for layout in layouts] == [
        "message",
        "function_call_output",
        "message",
    ]


@pytest.mark.parametrize(
    ("item", "param"),
    [
        (
            {"type": "function_call_output", "output": "ok"},
            "input[0].call_id",
        ),
        (
            {"type": "function_call_output", "call_id": 17, "output": "ok"},
            "input[0].call_id",
        ),
        (
            {"type": "function_call_output", "call_id": " ", "output": "ok"},
            "input[0].call_id",
        ),
        (
            {"type": "function_call_output", "call_id": "call_1"},
            "input[0].output",
        ),
        (
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": {"unsafe": "coercion"},
            },
            "input[0].output",
        ),
        (
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "ok",
                "status": "finished",
            },
            "input[0].status",
        ),
        (
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "ok",
                "id": " ",
            },
            "input[0].id",
        ),
        (
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": [{"type": "output_text", "text": "ok"}],
            },
            "input[0].output[0].type",
        ),
        (
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": [{"type": "input_text"}],
            },
            "input[0].output[0].text",
        ),
        (
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": [],
            },
            "input[0].output",
        ),
        (
            {"type": "function_call", "call_id": "call_1", "name": "t"},
            "input[0].arguments",
        ),
        (
            {"type": "function_call", "name": "t", "arguments": "{}"},
            "input[0].call_id",
        ),
        (
            {"type": "function_call", "call_id": "call_1", "arguments": "{}"},
            "input[0].name",
        ),
        (
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "t",
                "arguments": "{}",
                "status": "done",
            },
            "input[0].status",
        ),
        (
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "t",
                "arguments": "{}",
                "unknown": 1,
            },
            "input[0]",
        ),
    ],
)
def test_invalid_or_incomplete_function_items_fail_closed(
    item: Mapping[str, Any], param: str
) -> None:
    mapper, _, _ = _mapper()

    with pytest.raises(ResponsesMappingError) as error:
        _prepare(mapper, {"model": "flash-next", "input": [item]})

    assert error.value.code == "invalid_responses_request"
    assert error.value.param == param


def test_duplicate_function_output_call_ids_fail_closed() -> None:
    mapper, _, _ = _mapper()

    with pytest.raises(ResponsesMappingError) as error:
        _prepare(
            mapper,
            {
                "model": "flash-next",
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": "call_1",
                        "output": "first",
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_1",
                        "output": "second",
                    },
                ],
            },
        )

    assert error.value.code == "invalid_responses_request"
    assert error.value.param == "input[1].call_id"


def test_unknown_media_detail_fails_closed() -> None:
    mapper, _, _ = _mapper()

    with pytest.raises(ResponsesMappingError) as error:
        _prepare(
            mapper,
            {
                "model": "flash-next",
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_file",
                                "file_data": "data:application/pdf;base64,AAAA",
                                "detail": "extreme",
                            }
                        ],
                    }
                ],
            },
        )

    assert error.value.code == "invalid_responses_request"
    assert error.value.param == "input[0].content[0].detail"


def test_maps_runtime_tools_sampling_reasoning_metadata_and_lifecycle_flags() -> None:
    mapper, resolver, _ = _mapper()
    payload = {
        "model": "flash-next",
        "runtime_role": "main",
        "revision": "snapshot-1",
        "adapter_path": "/models/adapters/vet",
        "draft_model": "flash-draft",
        "input": "hello",
        "tools": [
            {
                "type": "function",
                "name": "record_lab_values",
                "parameters": {"type": "object", "required": ["wbc"]},
            }
        ],
        "tool_choice": {"type": "function", "name": "record_lab_values"},
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 40,
        "max_tokens": 321,
        "stop": ["END"],
        "reasoning": {"effort": "high", "summary": "auto"},
        "metadata": {"case_id": "LBRX-42"},
        "store": False,
        "cancel_on_disconnect": False,
    }

    prepared = _prepare(mapper, payload)

    assert resolver.calls == [
        {
            "model": "flash-next",
            "role": "main",
            "revision": "snapshot-1",
            "adapter_path": "/models/adapters/vet",
            "draft_model_id": "flash-draft",
            "backend": None,
        }
    ]
    assert prepared.request.runtime == RuntimeKey(
        model_id="flash-next",
        revision="snapshot-1",
        adapter_path="/models/adapters/vet",
        draft_model_id="flash-draft",
        backend=BackendKind.FUSED_MTP_MLX,
    )
    assert prepared.request.tools[0]["parameters"]["required"] == ("wbc",)
    assert prepared.request.sampling["max_output_tokens"] == 321
    assert prepared.request.sampling["tool_choice"]["name"] == "record_lab_values"
    assert prepared.request.reasoning == {"effort": "high", "summary": "auto"}
    assert prepared.request.metadata["case_id"] == "LBRX-42"
    assert prepared.request.metadata["runtime_role"] == "main"
    assert prepared.request.metadata["requested_model"] == "flash-next"
    assert prepared.request.metadata["resolved_model"] == "flash-next"
    assert prepared.store is False
    assert prepared.cancel_on_disconnect is False


def test_alias_resolution_may_supply_manifest_owned_runtime_identity() -> None:
    class AliasResolver:
        def __call__(self, **kwargs: Any) -> ResolvedRuntime:
            return ResolvedRuntime(
                runtime=RuntimeKey(
                    model_id="grant-ai/flash",
                    revision="snapshot-1",
                    backend=BackendKind.FUSED_MTP_MLX,
                ),
                requested_model=kwargs["model"],
                role="main",
            )

    mapper = CanonicalResponsesMapper(
        resolve_runtime=AliasResolver(),
        projection_factory=lambda _prepared: _Projection(),
    )

    prepared = _prepare(mapper, {"model": "buddy", "input": "hello"})

    assert prepared.request.runtime == RuntimeKey(
        model_id="grant-ai/flash",
        revision="snapshot-1",
        backend=BackendKind.FUSED_MTP_MLX,
    )
    assert prepared.request.metadata["requested_model"] == "buddy"
    assert prepared.request.metadata["resolved_model"] == "grant-ai/flash"
    assert prepared.request.metadata["runtime_role"] == "main"


@pytest.mark.parametrize(
    ("override", "param"),
    [
        (
            RuntimeKey(
                model_id="foreign-model",
                revision="snapshot-1",
                adapter_path="/adapter",
                draft_model_id="draft",
            ),
            "model",
        ),
        (
            RuntimeKey(
                model_id="flash-next",
                revision="foreign",
                adapter_path="/adapter",
                draft_model_id="draft",
            ),
            "revision",
        ),
        (
            RuntimeKey(
                model_id="flash-next",
                revision="snapshot-1",
                adapter_path="foreign",
                draft_model_id="draft",
            ),
            "adapter_path",
        ),
        (
            RuntimeKey(
                model_id="flash-next",
                revision="snapshot-1",
                adapter_path="/adapter",
                draft_model_id="foreign",
            ),
            "draft_model_id",
        ),
    ],
)
def test_resolver_cannot_change_request_owned_runtime_identity(
    override: RuntimeKey, param: str
) -> None:
    resolver = _Resolver()
    resolver.override = override
    mapper, _, _ = _mapper(resolver)

    with pytest.raises(ResponsesMappingError) as error:
        _prepare(
            mapper,
            {
                "model": "flash-next",
                "revision": "snapshot-1",
                "adapter_path": "/adapter",
                "draft_model_id": "draft",
                "input": "hello",
            },
        )

    assert error.value.code == "runtime_ownership_mismatch"
    assert error.value.param == param


@pytest.mark.parametrize(
    "claim",
    [
        {"response_id": "resp_foreign"},
        {"id": "resp_foreign"},
        {"owner_id": "principal:foreign"},
        {"metadata": {"owner_id": "principal:foreign"}},
        {"metadata": {"response_id": "resp_foreign"}},
    ],
)
def test_response_and_owner_claims_fail_closed(claim: Mapping[str, Any]) -> None:
    mapper, _, _ = _mapper()
    payload = {"model": "flash-next", "input": "hello", **claim}

    with pytest.raises(ResponsesMappingError) as error:
        _prepare(mapper, payload)

    assert error.value.code == "response_ownership_mismatch"


@pytest.mark.parametrize(
    ("part", "param"),
    [
        ({"type": "unknown", "value": "x"}, "type"),
        ({"text": "x", "image_url": "https://example.test/x.png"}, "type"),
        (
            {
                "type": "input_image",
                "image_url": "https://example.test/x.png",
                "file_id": "file_x",
            },
            "content",
        ),
        (
            {
                "type": "input_file",
                "file_id": "file_x",
                "file_url": "https://example.test/x.pdf",
            },
            "content",
        ),
        ({"type": "input_text", "text": {"unsafe": "coercion"}}, "text"),
    ],
)
def test_unknown_ambiguous_or_lossy_parts_fail_closed(
    part: Mapping[str, Any], param: str
) -> None:
    mapper, _, _ = _mapper()

    with pytest.raises(ResponsesMappingError) as error:
        _prepare(
            mapper,
            {
                "model": "flash-next",
                "input": [{"role": "user", "content": [part]}],
            },
        )

    assert error.value.code == "invalid_responses_request"
    assert param in str(error.value.param)


def test_aliases_and_instruction_sources_must_not_disagree() -> None:
    mapper, _, _ = _mapper()

    with pytest.raises(ResponsesMappingError) as token_error:
        _prepare(
            mapper,
            {
                "model": "flash-next",
                "input": "hello",
                "max_tokens": 5,
                "max_output_tokens": 6,
            },
        )
    assert token_error.value.code == "ambiguous_request_alias"

    with pytest.raises(ResponsesMappingError) as instruction_error:
        _prepare(
            mapper,
            {
                "model": "flash-next",
                "input": "hello",
                "system_instruction": "one",
                "instructions": "two",
            },
        )
    assert instruction_error.value.code == "ambiguous_instructions"


def test_caller_objects_are_not_mutated_and_boundaries_are_immutable() -> None:
    mapper, _, _ = _mapper()
    parent = {"role": "assistant", "content": [{"type": "input_text", "text": "old"}]}
    payload = {
        "model": "flash-next",
        "input": [{"role": "user", "content": [{"type": "input_text", "text": "new"}]}],
        "metadata": {"nested": {"items": [1, 2]}},
    }
    original_parent = {
        "role": "assistant",
        "content": [{"type": "input_text", "text": "old"}],
    }
    original_payload = {
        "model": "flash-next",
        "input": [{"role": "user", "content": [{"type": "input_text", "text": "new"}]}],
        "metadata": {"nested": {"items": [1, 2]}},
    }

    prepared = _prepare(mapper, payload, parents=(parent,))

    assert parent == original_parent
    assert payload == original_payload
    with pytest.raises(TypeError):
        prepared.request.metadata["new"] = True  # type: ignore[index]
    with pytest.raises(TypeError):
        prepared.request.metadata["nested"]["new"] = True  # type: ignore[index]
    assert prepared.request.metadata["nested"]["items"] == (1, 2)
    assert json.loads(json.dumps(list(prepared.materialized_messages)))[0] == parent


def test_projection_is_created_only_by_the_injected_factory() -> None:
    mapper, _, projected = _mapper()
    prepared = _prepare(mapper, {"model": "flash-next", "input": "hello"})

    projection = mapper.start_projection(prepared)

    assert isinstance(projection, _Projection)
    assert projected == [prepared]

    invalid = CanonicalResponsesMapper(
        resolve_runtime=_Resolver(),
        projection_factory=lambda _: object(),  # type: ignore[arg-type]
    )
    with pytest.raises(TypeError, match="ResponseProjection"):
        invalid.start_projection(prepared)


@pytest.mark.parametrize(
    "field",
    [
        "background",
        "conversation",
        "include",
        "max_tool_calls",
        "modalities",
        "output_modalities",
        "prompt",
        "prompt_cache_key",
        "service_tier",
        "stream_options",
        "truncation",
        "unknown_extension",
    ],
)
def test_unsupported_top_level_fields_fail_closed(field: str) -> None:
    mapper, _, _ = _mapper()

    with pytest.raises(ResponsesMappingError) as error:
        _prepare(
            mapper,
            {
                "model": "flash-next",
                "input": "hello",
                field: True,
            },
        )

    assert error.value.code == "unsupported_parameter"
    assert error.value.param == field


def test_unsupported_reasoning_fields_fail_closed() -> None:
    mapper, _, _ = _mapper()

    with pytest.raises(ResponsesMappingError) as error:
        _prepare(
            mapper,
            {
                "model": "flash-next",
                "input": "hello",
                "reasoning": {"generate_summary": "auto"},
            },
        )

    assert error.value.code == "unsupported_parameter"
    assert error.value.param == "reasoning.generate_summary"


def test_owner_bound_compaction_restores_context_before_new_input() -> None:
    codec = LocalCompactionCodec(key=b"c" * 32)
    prior = (
        {
            "role": "user",
            "content": ({"type": "input_text", "text": "first"},),
        },
        {
            "role": "assistant",
            "content": ({"type": "input_text", "text": "remembered"},),
        },
    )
    capsule = codec.seal(prior, owner_id="principal:owner")
    mapper = CanonicalResponsesMapper(
        resolve_runtime=_Resolver(),
        projection_factory=lambda _prepared: _Projection(),
        compaction_codec=codec,
    )

    prepared = _prepare(
        mapper,
        {
            "model": "flash-next",
            "input": [
                *compacted_user_messages(prior),
                {
                    "id": "cmp_1",
                    "type": "compaction",
                    "encrypted_content": capsule,
                    "created_by": "mlx-batch-server",
                },
                {"role": "user", "content": "new"},
            ],
        },
    )

    assert [message["role"] for message in prepared.materialized_messages] == [
        "user",
        "assistant",
        "user",
    ]
    assert prepared.materialized_messages[-1]["content"][0]["text"] == "new"


@pytest.mark.parametrize("tool_type", ["web_search", "file_search", "computer"])
def test_hosted_tools_fail_closed_with_stable_error(tool_type: str) -> None:
    mapper, _, _ = _mapper()

    with pytest.raises(ResponsesMappingError) as error:
        _prepare(
            mapper,
            {
                "model": "flash-next",
                "input": "hello",
                "tools": [{"type": tool_type}],
            },
        )

    assert error.value.code == "unsupported_tool"
    assert error.value.param == "tools[0].type"


def test_unpreserved_reasoning_summary_style_fails_closed() -> None:
    mapper, _, _ = _mapper()

    with pytest.raises(ResponsesMappingError) as error:
        _prepare(
            mapper,
            {
                "model": "flash-next",
                "input": "hello",
                "reasoning": {"summary": "concise"},
            },
        )

    assert error.value.code == "invalid_responses_request"
    assert error.value.param == "reasoning.summary"


def test_direct_official_function_call_reaches_the_backend_with_its_result() -> None:
    """Round two must carry the call that explains the result it supplies."""

    mapper, _, _ = _mapper()

    prepared = _prepare(
        mapper,
        {
            "model": "flash-next",
            "input": [
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": '{"city":"Kielce"}',
                    "status": "completed",
                },
                {
                    "type": "function_call_output",
                    "id": "fco_1",
                    "call_id": "call_1",
                    "output": "18C",
                    "status": "completed",
                },
                {"type": "message", "role": "user", "content": "I ciepło?"},
            ],
        },
    )

    call = prepared.materialized_messages[0]
    assert call == {
        "type": "function_call",
        "role": "assistant",
        "call_id": "call_1",
        "name": "get_weather",
        "arguments": '{"city":"Kielce"}',
        "content": (
            {
                "type": "input_text",
                "text": json.dumps(
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "get_weather",
                        "arguments": '{"city":"Kielce"}',
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
        ),
        "id": "fc_1",
        "status": "completed",
    }
    assert prepared.materialized_messages[1]["id"] == "fco_1"
    assert prepared.materialized_messages[1]["status"] == "completed"
    assert [item["type"] for item in prepared.materialized_messages] == [
        "function_call",
        "function_call_output",
        "message",
    ]

    # The backend receives the ordered call/result pair, not a lone result.
    backend = list(prepared.request.messages)
    assert [message["role"] for message in backend] == ["assistant", "tool", "user"]
    # The rendered call is the official item verbatim, so it round-trips.
    assert json.loads(backend[0]["content"][0]["text"]) == {
        "type": "function_call",
        "call_id": "call_1",
        "name": "get_weather",
        "arguments": '{"city":"Kielce"}',
    }
    assert backend[1]["call_id"] == "call_1"
    assert backend[1]["output"] == "18C"


def test_single_text_content_output_is_delivered_without_stringifying() -> None:
    mapper, _, _ = _mapper()

    prepared = _prepare(
        mapper,
        {
            "model": "flash-next",
            "input": [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "t",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": [{"type": "input_text", "text": "receipt"}],
                },
            ],
        },
    )

    assert prepared.request.messages[1]["output"] == "receipt"
    assert prepared.request.messages[1]["content"] == (
        {"type": "input_text", "text": "receipt"},
    )


@pytest.mark.parametrize(
    "output",
    [
        [{"type": "input_image", "image_url": "https://example.test/a.png"}],
        [{"type": "input_file", "file_id": "file_1"}],
        [
            {"type": "input_text", "text": "a"},
            {"type": "input_text", "text": "b"},
        ],
    ],
)
def test_unrenderable_official_output_variants_fail_with_capability_error(
    output: list[Mapping[str, Any]],
) -> None:
    """Official but unrenderable variants must not stringify or disappear."""

    mapper, _, _ = _mapper()

    with pytest.raises(ResponsesMappingError) as error:
        _prepare(
            mapper,
            {
                "model": "flash-next",
                "input": [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "t",
                        "arguments": "{}",
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_1",
                        "output": output,
                    },
                ],
            },
        )

    assert error.value.code == "unsupported_capability"
    assert error.value.param == "input[1].output"


def test_orphan_and_mismatched_tool_results_fail_closed() -> None:
    mapper, _, _ = _mapper()

    with pytest.raises(ResponsesMappingError) as orphan:
        _prepare(
            mapper,
            {
                "model": "flash-next",
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": "call_1",
                        "output": "ok",
                    }
                ],
            },
        )
    assert orphan.value.code == "invalid_responses_request"
    assert orphan.value.param == "input[0].call_id"

    with pytest.raises(ResponsesMappingError) as mismatch:
        _prepare(
            mapper,
            {
                "model": "flash-next",
                "input": [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "t",
                        "arguments": "{}",
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_2",
                        "output": "ok",
                    },
                ],
            },
        )
    assert mismatch.value.param == "input[1].call_id"

    with pytest.raises(ResponsesMappingError) as reordered:
        _prepare(
            mapper,
            {
                "model": "flash-next",
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": "call_1",
                        "output": "ok",
                    },
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "t",
                        "arguments": "{}",
                    },
                ],
            },
        )
    assert reordered.value.param == "input[0].call_id"


def test_inherited_call_lineage_explains_a_direct_tool_result() -> None:
    """A previous_response_id chain supplies the call round two never resends."""

    mapper, _, _ = _mapper()

    prepared = _prepare(
        mapper,
        {
            "model": "flash-next",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": INSPECT_CALL_ID,
                    "output": "receipt",
                }
            ],
        },
        parents=(_CALL_LINEAGE,),
    )

    assert [item["type"] for item in prepared.materialized_messages] == [
        "function_call",
        "function_call_output",
    ]
    assert prepared.materialized_messages[0]["call_id"] == INSPECT_CALL_ID
    assert prepared.materialized_messages[0]["arguments"] == INSPECT_ARGUMENTS
    assert prepared.request.messages[0]["content"][0]["text"] == _CALL_TEXT

    with pytest.raises(ResponsesMappingError) as foreign:
        _prepare(
            mapper,
            {
                "model": "flash-next",
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": "call_foreign",
                        "output": "receipt",
                    }
                ],
            },
            parents=(_CALL_LINEAGE,),
        )
    assert foreign.value.param == "input[0].call_id"


def test_inherited_tool_result_keeps_its_identity_for_a_third_round() -> None:
    """Re-reading stored lineage must not degrade a result into a bare message."""

    mapper, _, _ = _mapper()

    prepared = _prepare(
        mapper,
        {"model": "flash-next", "input": "and now?"},
        parents=(
            _CALL_LINEAGE,
            {
                "type": "function_call_output",
                "role": "tool",
                "call_id": INSPECT_CALL_ID,
                "output": "receipt",
                "content": [{"type": "input_text", "text": "receipt"}],
            },
        ),
    )

    assert [item.get("type") for item in prepared.materialized_messages[:2]] == [
        "function_call",
        "function_call_output",
    ]
    inherited = prepared.request.messages[1]
    assert inherited["type"] == "function_call_output"
    assert inherited["call_id"] == INSPECT_CALL_ID
    assert inherited["output"] == "receipt"
