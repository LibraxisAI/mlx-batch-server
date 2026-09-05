"""RED contracts for the OpenAI Responses request capability taxonomy.

The delivery verifier lives here: the installed SDK field universe is compared
to the published profile, then every classified field is driven through the real
mapper and must show either an observable canonical destination or an exact
`unsupported_parameter` error.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import pytest

from mlx_batch_server.responses.request_contract import (
    ACCEPTED_TOP_LEVEL_FIELDS,
    CAPABILITY_PROFILE_VERSION,
    EXTENSION_FIELD_NAMES,
    FIELD_CONTRACTS,
    OFFICIAL_FIELD_NAMES,
    RESERVED_METADATA_KEYS,
    FieldSupport,
    ResponsesMappingError,
    capability_profile,
    official_sdk_fields,
    sdk_version,
)
from mlx_batch_server.responses.runtime_mapper import CanonicalResponsesMapper
from mlx_batch_server.runtime.contracts import BackendKind, RuntimeKey

if TYPE_CHECKING:
    from mlx_batch_server.responses.controller import PreparedResponse

RESPONSE_ID = "resp_contract"
OWNER_ID = "principal:contract"
PARENT = {"role": "assistant", "content": [{"type": "input_text", "text": "earlier"}]}


class _Projection:
    def observe(self, event: Any) -> None:
        del event

    def terminal_envelope(self) -> Mapping[str, Any]:
        return {"id": RESPONSE_ID, "status": "completed"}


def _resolve(**kwargs: Any) -> RuntimeKey:
    return RuntimeKey(
        model_id=kwargs["model"],
        revision=kwargs["revision"],
        adapter_path=kwargs["adapter_path"],
        draft_model_id=kwargs["draft_model_id"],
        backend=BackendKind.FUSED_MTP_MLX,
    )


def _mapper() -> CanonicalResponsesMapper:
    return CanonicalResponsesMapper(
        resolve_runtime=_resolve,
        projection_factory=lambda prepared: _Projection(),
    )


def _prepare(payload: Mapping[str, Any]) -> PreparedResponse:
    return _mapper().prepare(
        payload,
        response_id=RESPONSE_ID,
        owner_id=OWNER_ID,
        parent_messages=(PARENT,),
    )


def _base(**extra: Any) -> dict[str, Any]:
    return {"model": "flash-next", "input": "hello", **extra}


# One fixture per classified field: the sentinel payload fragment and the exact
# canonical destination it must reach. Unsupported fields carry no destination.
_ACCEPTED_FIXTURES: Mapping[str, tuple[Mapping[str, Any], Any, Any]] = {
    # official, implemented
    "input": (
        {"input": "hello"},
        lambda p: p.request.messages[-1]["content"][0]["text"],
        "hello",
    ),
    "model": ({}, lambda p: p.request.runtime.model_id, "flash-next"),
    "instructions": (
        {"instructions": "be exact"},
        lambda p: p.request.messages[0]["content"][0]["text"],
        "be exact",
    ),
    "max_output_tokens": (
        {"max_output_tokens": 64},
        lambda p: p.request.sampling["max_output_tokens"],
        64,
    ),
    "metadata": (
        {"metadata": {"case_id": "LBRX-42"}},
        lambda p: p.request.metadata["case_id"],
        "LBRX-42",
    ),
    "parallel_tool_calls": (
        {"parallel_tool_calls": False},
        lambda p: p.request.sampling["parallel_tool_calls"],
        False,
    ),
    "previous_response_id": (
        {"previous_response_id": "resp_parent"},
        lambda p: p.request.lineage[0]["content"][0]["text"],
        "earlier",
    ),
    "reasoning": (
        {"reasoning": {"effort": "high"}},
        lambda p: p.request.reasoning["effort"],
        "high",
    ),
    "store": ({"store": False}, lambda p: p.store, False),
    "temperature": (
        {"temperature": 0.25},
        lambda p: p.request.sampling["temperature"],
        0.25,
    ),
    "text": (
        {"text": {"format": {"type": "json_object"}}},
        lambda p: p.request.sampling["text"]["format"]["type"],
        "json_object",
    ),
    "tool_choice": (
        {
            "tools": [{"type": "function", "name": "record"}],
            "tool_choice": {"type": "function", "name": "record"},
        },
        lambda p: p.request.sampling["tool_choice"]["name"],
        "record",
    ),
    "tools": (
        {"tools": [{"type": "function", "name": "record"}]},
        lambda p: p.request.tools[0]["name"],
        "record",
    ),
    "top_p": ({"top_p": 0.8}, lambda p: p.request.sampling["top_p"], 0.8),
    # official, locally interpreted
    "background": (
        {"background": False},
        lambda p: p.request.metadata["background"],
        False,
    ),
    "include": ({"include": []}, lambda p: p.request.metadata["include"], ()),
    "service_tier": (
        {"service_tier": "default"},
        lambda p: p.request.metadata["service_tier"],
        "default",
    ),
    "stream": ({"stream": True}, lambda p: p.request.metadata["stream"], True),
    "stream_options": (
        {"stream": True, "stream_options": {"include_obfuscation": False}},
        lambda p: p.request.metadata["stream_options"]["include_obfuscation"],
        False,
    ),
    "truncation": (
        {"truncation": "disabled"},
        lambda p: p.request.metadata["truncation"],
        "disabled",
    ),
    # Libraxis extensions
    "adapter_path": (
        {"adapter_path": "/models/adapters/vet"},
        lambda p: p.request.runtime.adapter_path,
        "/models/adapters/vet",
    ),
    "backend": (
        {"backend": "fused_mtp_mlx"},
        lambda p: p.request.runtime.backend,
        BackendKind.FUSED_MTP_MLX,
    ),
    "cancel_on_disconnect": (
        {"cancel_on_disconnect": False},
        lambda p: p.cancel_on_disconnect,
        False,
    ),
    "draft_model": (
        {"draft_model": "flash-draft"},
        lambda p: p.request.runtime.draft_model_id,
        "flash-draft",
    ),
    "draft_model_id": (
        {"draft_model_id": "flash-draft"},
        lambda p: p.request.runtime.draft_model_id,
        "flash-draft",
    ),
    "frequency_penalty": (
        {"frequency_penalty": 0.1},
        lambda p: p.request.sampling["frequency_penalty"],
        0.1,
    ),
    "id": ({"id": RESPONSE_ID}, lambda p: p.request.response_id, RESPONSE_ID),
    "max_tokens": (
        {"max_tokens": 32},
        lambda p: p.request.sampling["max_output_tokens"],
        32,
    ),
    "model_revision": (
        {"model_revision": "snapshot-1"},
        lambda p: p.request.runtime.revision,
        "snapshot-1",
    ),
    "owner_id": ({"owner_id": OWNER_ID}, lambda p: p.request.response_id, RESPONSE_ID),
    "presence_penalty": (
        {"presence_penalty": 0.2},
        lambda p: p.request.sampling["presence_penalty"],
        0.2,
    ),
    "response_format": (
        {"response_format": {"type": "json_object"}},
        lambda p: p.request.sampling["response_format"]["type"],
        "json_object",
    ),
    "response_id": (
        {"response_id": RESPONSE_ID},
        lambda p: p.request.response_id,
        RESPONSE_ID,
    ),
    "revision": (
        {"revision": "snapshot-1"},
        lambda p: p.request.runtime.revision,
        "snapshot-1",
    ),
    "runtime_role": (
        {"runtime_role": "main"},
        lambda p: p.request.metadata["runtime_role"],
        "main",
    ),
    "seed": ({"seed": 7}, lambda p: p.request.sampling["seed"], 7),
    "stop": ({"stop": ["END"]}, lambda p: p.request.sampling["stop"], ("END",)),
    "system_instruction": (
        {"system_instruction": "stay local"},
        lambda p: p.request.messages[0]["content"][0]["text"],
        "stay local",
    ),
    "top_k": ({"top_k": 40}, lambda p: p.request.sampling["top_k"], 40),
}

# A sentinel value per unsupported official field. None of them may be accepted.
_UNSUPPORTED_SENTINELS: Mapping[str, Any] = {
    "context_management": {"strategy": "auto"},
    "conversation": "conv_sentinel",
    "max_tool_calls": 3,
    "prompt": {"id": "pmpt_sentinel"},
    "prompt_cache_key": "cache-sentinel",
    "prompt_cache_retention": "24h",
    "safety_identifier": "user-sentinel",
    "top_logprobs": 5,
    "user": "user-sentinel",
}

_PROFILE_SNAPSHOT: tuple[tuple[str, str, str], ...] = (
    ("background", "official", "locally_interpreted"),
    ("context_management", "official", "unsupported"),
    ("conversation", "official", "unsupported"),
    ("include", "official", "locally_interpreted"),
    ("input", "official", "implemented"),
    ("instructions", "official", "implemented"),
    ("max_output_tokens", "official", "implemented"),
    ("max_tool_calls", "official", "unsupported"),
    ("metadata", "official", "implemented"),
    ("model", "official", "implemented"),
    ("parallel_tool_calls", "official", "implemented"),
    ("previous_response_id", "official", "implemented"),
    ("prompt", "official", "unsupported"),
    ("prompt_cache_key", "official", "unsupported"),
    ("prompt_cache_retention", "official", "unsupported"),
    ("reasoning", "official", "implemented"),
    ("safety_identifier", "official", "unsupported"),
    ("service_tier", "official", "locally_interpreted"),
    ("store", "official", "implemented"),
    ("stream", "official", "locally_interpreted"),
    ("stream_options", "official", "locally_interpreted"),
    ("temperature", "official", "implemented"),
    ("text", "official", "implemented"),
    ("tool_choice", "official", "implemented"),
    ("tools", "official", "implemented"),
    ("top_logprobs", "official", "unsupported"),
    ("top_p", "official", "implemented"),
    ("truncation", "official", "locally_interpreted"),
    ("user", "official", "unsupported"),
    ("adapter_path", "extension", "implemented"),
    ("backend", "extension", "implemented"),
    ("cancel_on_disconnect", "extension", "implemented"),
    ("draft_model", "extension", "implemented"),
    ("draft_model_id", "extension", "implemented"),
    ("frequency_penalty", "extension", "implemented"),
    ("id", "extension", "implemented"),
    ("max_tokens", "extension", "implemented"),
    ("model_revision", "extension", "implemented"),
    ("owner_id", "extension", "implemented"),
    ("presence_penalty", "extension", "implemented"),
    ("response_format", "extension", "implemented"),
    ("response_id", "extension", "implemented"),
    ("revision", "extension", "implemented"),
    ("runtime_role", "extension", "implemented"),
    ("seed", "extension", "implemented"),
    ("stop", "extension", "implemented"),
    ("system_instruction", "extension", "implemented"),
    ("top_k", "extension", "implemented"),
)


def test_every_installed_sdk_field_is_classified_exactly_once() -> None:
    universe = official_sdk_fields()

    assert sorted(OFFICIAL_FIELD_NAMES) == sorted(universe)
    assert not OFFICIAL_FIELD_NAMES & EXTENSION_FIELD_NAMES
    assert len(FIELD_CONTRACTS) == len(OFFICIAL_FIELD_NAMES) + len(
        EXTENSION_FIELD_NAMES
    )
    for name, contract in FIELD_CONTRACTS.items():
        assert contract.name == name


def test_capability_profile_is_versioned_stable_and_partitioned() -> None:
    profile = capability_profile()

    assert profile["version"] == CAPABILITY_PROFILE_VERSION
    assert profile["sdk"] == {
        "distribution": "openai",
        "version": sdk_version(),
        "params_module": "openai.types.responses.response_create_params",
        "official_field_count": len(official_sdk_fields()),
    }
    assert profile["counts"] == {
        "official": 29,
        "extension": 19,
        "implemented": 33,
        "locally_interpreted": 6,
        "unsupported": 9,
    }
    assert (
        tuple(
            (entry["name"], entry["surface"], entry["support"])
            for entry in profile["fields"]
        )
        == _PROFILE_SNAPSHOT
    )
    assert profile["metadata_limits"] == {
        "max_entries": 16,
        "max_key_length": 64,
        "max_string_value_length": 512,
        "non_string_values": "extension",
    }
    assert profile["reserved_metadata_keys"] == sorted(RESERVED_METADATA_KEYS)
    assert [rule["id"] for rule in profile["combination_rules"]] == [
        "stream_options_requires_stream",
        "conversation_conflicts_with_previous_response_id",
        "background_requires_store",
        "tool_choice_requires_declared_tool",
        "aliases_must_agree",
    ]


def test_every_unsupported_field_states_a_reason_and_no_destination() -> None:
    for contract in FIELD_CONTRACTS.values():
        if contract.support is FieldSupport.UNSUPPORTED:
            assert contract.reason
            assert contract.destination is None
        else:
            assert contract.destination


def test_the_fixture_table_covers_the_whole_classified_universe() -> None:
    assert set(_ACCEPTED_FIXTURES) == set(ACCEPTED_TOP_LEVEL_FIELDS)
    assert set(_UNSUPPORTED_SENTINELS) == {
        name
        for name, contract in FIELD_CONTRACTS.items()
        if contract.support is FieldSupport.UNSUPPORTED
    }


@pytest.mark.parametrize("field", sorted(_ACCEPTED_FIXTURES))
def test_every_accepted_field_reaches_its_declared_destination(field: str) -> None:
    fragment, probe, expected = _ACCEPTED_FIXTURES[field]

    prepared = _prepare(_base(**fragment))

    assert probe(prepared) == expected


@pytest.mark.parametrize("field", sorted(_UNSUPPORTED_SENTINELS))
def test_every_unsupported_field_rejects_its_sentinel(field: str) -> None:
    with pytest.raises(ResponsesMappingError) as error:
        _prepare(_base(**{field: _UNSUPPORTED_SENTINELS[field]}))

    assert error.value.code == "unsupported_parameter"
    assert error.value.param == field
    assert FIELD_CONTRACTS[field].reason in str(error.value)


def test_unclassified_fields_can_never_be_accepted() -> None:
    with pytest.raises(ResponsesMappingError) as error:
        _prepare(_base(future_sdk_field="sentinel"))

    assert error.value.code == "unsupported_parameter"
    assert error.value.param == "future_sdk_field"
