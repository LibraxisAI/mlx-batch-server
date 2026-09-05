"""One auditable capability contract for official Responses request fields.

The mapper used to carry a hand-written allowlist that mixed official
`ResponseCreateParams` fields with Libraxis runtime selectors, so a reader could
not tell which parameters this server actually honors. This module owns that
truth instead: every field the installed `openai` SDK declares is classified
exactly once as implemented, locally interpreted or explicitly unsupported, and
the classification is the same data the mapper validates against, so the
published profile cannot drift into documentation-only truth.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from openai.types.responses.response_create_params import (
    ResponseCreateParamsBase,
    ResponseCreateParamsNonStreaming,
    ResponseCreateParamsStreaming,
)

CAPABILITY_PROFILE_VERSION = "responses.request.capability/1"
SDK_DISTRIBUTION = "openai"
SDK_PARAMS_MODULE = "openai.types.responses.response_create_params"

METADATA_MAX_ENTRIES = 16
METADATA_MAX_KEY_LENGTH = 64
METADATA_MAX_VALUE_LENGTH = 512


class ResponsesMappingError(ValueError):
    """A request cannot be mapped without guessing or changing ownership."""

    def __init__(self, message: str, *, code: str, param: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.param = param


class FieldSurface(StrEnum):
    """Which contract a field belongs to."""

    OFFICIAL = "official"
    EXTENSION = "extension"


class FieldSupport(StrEnum):
    """How much of a field this server can actually honor."""

    IMPLEMENTED = "implemented"
    LOCAL = "locally_interpreted"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class FieldContract:
    """One field, one classification, one observable consequence."""

    name: str
    surface: FieldSurface
    support: FieldSupport
    destination: str | None = None
    accepted: tuple[Any, ...] | None = None
    reason: str | None = None
    requires: tuple[str, ...] = ()

    def as_profile_entry(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "surface": self.surface.value,
            "support": self.support.value,
            "destination": self.destination,
            "accepted": list(self.accepted) if self.accepted is not None else None,
            "reason": self.reason,
            "requires": list(self.requires),
        }


def _official(
    name: str,
    support: FieldSupport,
    *,
    destination: str | None = None,
    accepted: tuple[Any, ...] | None = None,
    reason: str | None = None,
    requires: tuple[str, ...] = (),
) -> FieldContract:
    return FieldContract(
        name=name,
        surface=FieldSurface.OFFICIAL,
        support=support,
        destination=destination,
        accepted=accepted,
        reason=reason,
        requires=requires,
    )


def _extension(
    name: str, support: FieldSupport, *, destination: str, reason: str | None = None
) -> FieldContract:
    return FieldContract(
        name=name,
        surface=FieldSurface.EXTENSION,
        support=support,
        destination=destination,
        reason=reason,
    )


_IMPLEMENTED = FieldSupport.IMPLEMENTED
_LOCAL = FieldSupport.LOCAL
_UNSUPPORTED = FieldSupport.UNSUPPORTED

_OFFICIAL_CONTRACTS: tuple[FieldContract, ...] = (
    _official(
        "background",
        _LOCAL,
        destination="request.metadata['background']",
        accepted=(False,),
        reason=(
            "this server executes one response synchronously; there is no "
            "background scheduler that could own a deferred run"
        ),
        requires=("store",),
    ),
    _official(
        "context_management",
        _UNSUPPORTED,
        reason="no automatic context-management owner exists in this runtime",
    ),
    _official(
        "conversation",
        _UNSUPPORTED,
        reason=(
            "the OpenAI conversation store is not hosted here; local chaining is "
            "owned by previous_response_id"
        ),
    ),
    _official(
        "include",
        _LOCAL,
        destination="request.metadata['include']",
        accepted=((),),
        reason=(
            "no includable side channel (logprobs, encrypted reasoning, hosted "
            "tool artifacts) is produced locally, so only an empty include is true"
        ),
    ),
    _official(
        "input",
        _IMPLEMENTED,
        destination="request.messages + request.media",
    ),
    _official(
        "instructions",
        _IMPLEMENTED,
        destination="request.messages[developer]",
    ),
    _official(
        "max_output_tokens",
        _IMPLEMENTED,
        destination="request.sampling['max_output_tokens']",
    ),
    _official(
        "max_tool_calls",
        _UNSUPPORTED,
        reason=(
            "tool calls are executed by the client, so this server cannot enforce "
            "a per-response tool-call budget"
        ),
    ),
    _official("metadata", _IMPLEMENTED, destination="request.metadata"),
    _official(
        "model",
        _IMPLEMENTED,
        destination="request.runtime.model_id + request.metadata['requested_model']",
    ),
    _official(
        "parallel_tool_calls",
        _IMPLEMENTED,
        destination="request.sampling['parallel_tool_calls']",
    ),
    _official(
        "previous_response_id",
        _IMPLEMENTED,
        destination="request.lineage",
    ),
    _official(
        "prompt",
        _UNSUPPORTED,
        reason="no prompt-template service backs this server",
    ),
    _official(
        "prompt_cache_key",
        _UNSUPPORTED,
        reason=(
            "prefix cache partitions are keyed by runtime identity, not by a "
            "caller-supplied key; accepting one would not change execution"
        ),
    ),
    _official(
        "prompt_cache_retention",
        _UNSUPPORTED,
        reason="local prefix cache lifetime is owned by the runtime, not the request",
    ),
    _official("reasoning", _IMPLEMENTED, destination="request.reasoning"),
    _official(
        "safety_identifier",
        _UNSUPPORTED,
        reason=(
            "no abuse-monitoring owner consumes an end-user identifier; storing it "
            "would not affect execution"
        ),
    ),
    _official(
        "service_tier",
        _LOCAL,
        destination="request.metadata['service_tier']",
        accepted=("auto", "default"),
        reason=(
            "one local tier serves every request; flex, scale and priority have no "
            "distinct scheduling class here"
        ),
    ),
    _official("store", _IMPLEMENTED, destination="prepared.store"),
    _official(
        "stream",
        _LOCAL,
        destination="request.metadata['stream']",
        accepted=(False, True),
        reason="delivery is owned by the transport; the mapper records the claim",
    ),
    _official(
        "stream_options",
        _LOCAL,
        destination="request.metadata['stream_options']",
        accepted=({}, {"include_obfuscation": False}),
        reason="this server never emits obfuscation padding",
        requires=("stream",),
    ),
    _official(
        "temperature", _IMPLEMENTED, destination="request.sampling['temperature']"
    ),
    _official("text", _IMPLEMENTED, destination="request.sampling['text']"),
    _official(
        "tool_choice", _IMPLEMENTED, destination="request.sampling['tool_choice']"
    ),
    _official("tools", _IMPLEMENTED, destination="request.tools"),
    _official(
        "top_logprobs",
        _UNSUPPORTED,
        reason="the runtime does not emit logprobs on any output surface",
    ),
    _official("top_p", _IMPLEMENTED, destination="request.sampling['top_p']"),
    _official(
        "truncation",
        _LOCAL,
        destination="request.metadata['truncation']",
        accepted=("disabled",),
        reason=(
            "the runtime never drops context silently; an over-long prompt fails "
            "closed instead of being auto-truncated"
        ),
    ),
    _official(
        "user",
        _UNSUPPORTED,
        reason=(
            "deprecated by the SDK in favour of safety_identifier and equally "
            "unowned locally"
        ),
    ),
)

_EXTENSION_CONTRACTS: tuple[FieldContract, ...] = (
    _extension(
        "adapter_path", _IMPLEMENTED, destination="request.runtime.adapter_path"
    ),
    _extension("backend", _IMPLEMENTED, destination="request.runtime.backend"),
    _extension(
        "cancel_on_disconnect",
        _IMPLEMENTED,
        destination="prepared.cancel_on_disconnect",
    ),
    _extension(
        "draft_model",
        _IMPLEMENTED,
        destination="request.runtime.draft_model_id",
        reason="alias of draft_model_id",
    ),
    _extension(
        "draft_model_id", _IMPLEMENTED, destination="request.runtime.draft_model_id"
    ),
    _extension(
        "frequency_penalty",
        _IMPLEMENTED,
        destination="request.sampling['frequency_penalty']",
    ),
    _extension(
        "id",
        _IMPLEMENTED,
        destination="ownership claim checked against the issued response id",
    ),
    _extension(
        "max_tokens",
        _IMPLEMENTED,
        destination="request.sampling['max_output_tokens']",
        reason="alias of max_output_tokens",
    ),
    _extension(
        "model_revision",
        _IMPLEMENTED,
        destination="request.runtime.revision",
        reason="alias of revision",
    ),
    _extension(
        "owner_id",
        _IMPLEMENTED,
        destination="ownership claim checked against the authenticated owner",
    ),
    _extension(
        "presence_penalty",
        _IMPLEMENTED,
        destination="request.sampling['presence_penalty']",
    ),
    _extension(
        "response_format",
        _IMPLEMENTED,
        destination="request.sampling['response_format']",
        reason="legacy Chat Completions structured output, aliased by text",
    ),
    _extension(
        "response_id",
        _IMPLEMENTED,
        destination="ownership claim checked against the issued response id",
    ),
    _extension("revision", _IMPLEMENTED, destination="request.runtime.revision"),
    _extension(
        "runtime_role", _IMPLEMENTED, destination="request.metadata['runtime_role']"
    ),
    _extension("seed", _IMPLEMENTED, destination="request.sampling['seed']"),
    _extension("stop", _IMPLEMENTED, destination="request.sampling['stop']"),
    _extension(
        "system_instruction",
        _IMPLEMENTED,
        destination="request.messages[system]",
        reason="alias of instructions",
    ),
    _extension("top_k", _IMPLEMENTED, destination="request.sampling['top_k']"),
)


def _indexed(contracts: Sequence[FieldContract]) -> Mapping[str, FieldContract]:
    index: dict[str, FieldContract] = {}
    for contract in contracts:
        if contract.name in index:  # pragma: no cover - guarded by parity tests
            raise ValueError(f"duplicate field contract: {contract.name}")
        index[contract.name] = contract
    return index


OFFICIAL_CONTRACTS: Mapping[str, FieldContract] = _indexed(_OFFICIAL_CONTRACTS)
EXTENSION_CONTRACTS: Mapping[str, FieldContract] = _indexed(_EXTENSION_CONTRACTS)
FIELD_CONTRACTS: Mapping[str, FieldContract] = _indexed(
    (*_OFFICIAL_CONTRACTS, *_EXTENSION_CONTRACTS)
)

OFFICIAL_FIELD_NAMES = frozenset(OFFICIAL_CONTRACTS)
EXTENSION_FIELD_NAMES = frozenset(EXTENSION_CONTRACTS)
LOCAL_FIELD_NAMES = frozenset(
    name
    for name, contract in OFFICIAL_CONTRACTS.items()
    if contract.support is FieldSupport.LOCAL
)
UNSUPPORTED_FIELD_NAMES = frozenset(
    name
    for name, contract in FIELD_CONTRACTS.items()
    if contract.support is FieldSupport.UNSUPPORTED
)
ACCEPTED_TOP_LEVEL_FIELDS = frozenset(FIELD_CONTRACTS) - UNSUPPORTED_FIELD_NAMES

RESERVED_METADATA_KEYS = frozenset(
    {"requested_model", "resolved_model", "runtime_role", *LOCAL_FIELD_NAMES}
)


def official_sdk_fields() -> frozenset[str]:
    """The field universe the installed SDK declares for `responses.create`."""

    universe: set[str] = set()
    for params in (
        ResponseCreateParamsBase,
        ResponseCreateParamsNonStreaming,
        ResponseCreateParamsStreaming,
    ):
        universe.update(params.__annotations__)
    return frozenset(universe)


def sdk_version() -> str:
    try:
        return version(SDK_DISTRIBUTION)
    except PackageNotFoundError:  # pragma: no cover - packaging accident
        return "unknown"


_ALIAS_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("max_output_tokens", "max_tokens", "max_output_tokens"),
    ("revision", "model_revision", "revision"),
    ("draft_model_id", "draft_model", "draft_model_id"),
    ("instructions", "system_instruction", "instructions"),
    ("text", "response_format", "text"),
)

_COMBINATION_RULES: tuple[Mapping[str, Any], ...] = (
    {
        "id": "stream_options_requires_stream",
        "fields": ("stream_options", "stream"),
        "rule": "stream_options is only meaningful while stream is true",
        "param": "stream_options",
    },
    {
        "id": "conversation_conflicts_with_previous_response_id",
        "fields": ("conversation", "previous_response_id"),
        "rule": "a response may inherit from a conversation or a previous response, never both",
        "param": "conversation",
    },
    {
        "id": "background_requires_store",
        "fields": ("background", "store"),
        "rule": "a background response must be stored to remain retrievable",
        "param": "store",
    },
    {
        "id": "tool_choice_requires_declared_tool",
        "fields": ("tool_choice", "tools"),
        "rule": "a named tool_choice must reference a declared function tool",
        "param": "tool_choice.name",
    },
    {
        "id": "aliases_must_agree",
        "fields": tuple(sorted({name for pair in _ALIAS_PAIRS for name in pair[:2]})),
        "rule": "alias fields carrying the same meaning must not disagree",
        "param": "<the canonical alias name>",
    },
)


def capability_profile() -> Mapping[str, Any]:
    """A stable, versioned, machine-readable statement of what is honored."""

    fields = [
        contract.as_profile_entry()
        for contract in (
            *sorted(_OFFICIAL_CONTRACTS, key=lambda item: item.name),
            *sorted(_EXTENSION_CONTRACTS, key=lambda item: item.name),
        )
    ]
    counts = {
        support.value: sum(
            1 for contract in FIELD_CONTRACTS.values() if contract.support is support
        )
        for support in FieldSupport
    }
    return {
        "version": CAPABILITY_PROFILE_VERSION,
        "sdk": {
            "distribution": SDK_DISTRIBUTION,
            "version": sdk_version(),
            "params_module": SDK_PARAMS_MODULE,
            "official_field_count": len(official_sdk_fields()),
        },
        "counts": {
            "official": len(OFFICIAL_CONTRACTS),
            "extension": len(EXTENSION_CONTRACTS),
            **counts,
        },
        "fields": fields,
        "combination_rules": [dict(rule) for rule in _COMBINATION_RULES],
        "metadata_limits": {
            "max_entries": METADATA_MAX_ENTRIES,
            "max_key_length": METADATA_MAX_KEY_LENGTH,
            "max_string_value_length": METADATA_MAX_VALUE_LENGTH,
            "non_string_values": "extension",
        },
        "reserved_metadata_keys": sorted(RESERVED_METADATA_KEYS),
    }


def _unsupported(field: str, contract: FieldContract) -> ResponsesMappingError:
    reason = contract.reason or "this server cannot honor the parameter"
    return ResponsesMappingError(
        f"unsupported Responses parameter: {field} ({reason})",
        code="unsupported_parameter",
        param=field,
    )


def _conflict(message: str, param: str) -> ResponsesMappingError:
    return ResponsesMappingError(message, code="invalid_field_combination", param=param)


def validate_top_level_fields(payload: Mapping[str, Any]) -> None:
    """Reject anything this server would otherwise accept and quietly ignore."""

    for field in sorted(payload):
        contract = FIELD_CONTRACTS.get(field)
        if contract is None:
            raise ResponsesMappingError(
                f"unsupported Responses parameter: {field}",
                code="unsupported_parameter",
                param=field,
            )
        if contract.support is FieldSupport.UNSUPPORTED:
            raise _unsupported(field, contract)


def validate_field_combinations(payload: Mapping[str, Any]) -> None:
    """Cross-field legality, evaluated before single-field classification."""

    if "stream_options" in payload and not payload.get("stream"):
        raise _conflict(
            "stream_options requires a streaming response",
            "stream_options",
        )
    if (
        payload.get("conversation") is not None
        and payload.get("previous_response_id") is not None
    ):
        raise _conflict(
            "conversation and previous_response_id cannot both own the context",
            "conversation",
        )
    if payload.get("background") and payload.get("store") is False:
        raise _conflict(
            "a background response must be stored to remain retrievable",
            "store",
        )


def local_setting(field: str, value: Any) -> Any:
    """Canonicalize one locally interpreted field or fail with its exact param."""

    contract = OFFICIAL_CONTRACTS[field]
    if field == "include":
        return _local_include(value)
    if field == "stream_options":
        return _local_stream_options(value)
    if field == "stream":
        if not isinstance(value, bool):
            raise ResponsesMappingError(
                "stream must be a boolean",
                code="invalid_responses_request",
                param=field,
            )
        return value
    if field == "background" and not isinstance(value, bool):
        raise ResponsesMappingError(
            "background must be a boolean",
            code="invalid_responses_request",
            param=field,
        )
    accepted = contract.accepted or ()
    if value not in accepted:
        raise _unsupported(field, contract)
    return value


def _local_include(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        raise ResponsesMappingError(
            "include must be a sequence",
            code="invalid_responses_request",
            param="include",
        )
    if value:
        raise ResponsesMappingError(
            f"unsupported Responses include entry: {value[0]!r} "
            f"({OFFICIAL_CONTRACTS['include'].reason})",
            code="unsupported_parameter",
            param="include[0]",
        )
    return ()


def _local_stream_options(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ResponsesMappingError(
            "stream_options must be a mapping",
            code="invalid_responses_request",
            param="stream_options",
        )
    unknown = sorted(set(value) - {"include_obfuscation"})
    if unknown:
        raise ResponsesMappingError(
            f"unsupported Responses parameter: stream_options.{unknown[0]}",
            code="unsupported_parameter",
            param=f"stream_options.{unknown[0]}",
        )
    obfuscation = value.get("include_obfuscation")
    if obfuscation is None:
        return {}
    if obfuscation is not False:
        raise ResponsesMappingError(
            "unsupported Responses parameter: stream_options.include_obfuscation "
            "(this server never emits obfuscation padding)",
            code="unsupported_parameter",
            param="stream_options.include_obfuscation",
        )
    return {"include_obfuscation": False}


def local_settings(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Every accepted locally interpreted field, canonicalized for metadata."""

    settings: dict[str, Any] = {}
    for field in sorted(LOCAL_FIELD_NAMES):
        if field not in payload or payload[field] is None:
            continue
        settings[field] = local_setting(field, payload[field])
    return settings


def validate_client_metadata(raw: Any, *, derived: Mapping[str, Any]) -> None:
    """Official metadata limits plus a fence around server-owned keys."""

    if raw is None:
        return
    if not isinstance(raw, Mapping):
        raise ResponsesMappingError(
            "metadata must be a mapping",
            code="invalid_responses_request",
            param="metadata",
        )
    if len(raw) > METADATA_MAX_ENTRIES:
        raise ResponsesMappingError(
            f"metadata accepts at most {METADATA_MAX_ENTRIES} entries",
            code="invalid_responses_request",
            param="metadata",
        )
    for key in sorted(raw):
        _validate_metadata_entry(key, raw[key])
    _validate_reserved_metadata(raw, derived=derived)


def _validate_metadata_entry(key: str, value: Any) -> None:
    if not isinstance(key, str) or len(key) > METADATA_MAX_KEY_LENGTH:
        raise ResponsesMappingError(
            f"metadata keys accept at most {METADATA_MAX_KEY_LENGTH} characters",
            code="invalid_responses_request",
            param=f"metadata.{key}",
        )
    if isinstance(value, str) and len(value) > METADATA_MAX_VALUE_LENGTH:
        raise ResponsesMappingError(
            f"metadata values accept at most {METADATA_MAX_VALUE_LENGTH} characters",
            code="invalid_responses_request",
            param=f"metadata.{key}",
        )


def _validate_reserved_metadata(
    raw: Mapping[str, Any], *, derived: Mapping[str, Any]
) -> None:
    for key in sorted(set(raw) & LOCAL_FIELD_NAMES):
        if key not in derived or raw[key] != derived[key]:
            raise ResponsesMappingError(
                f"metadata.{key} is owned by the server request contract",
                code="reserved_metadata_key",
                param=f"metadata.{key}",
            )


__all__ = [
    "ACCEPTED_TOP_LEVEL_FIELDS",
    "CAPABILITY_PROFILE_VERSION",
    "EXTENSION_CONTRACTS",
    "EXTENSION_FIELD_NAMES",
    "FIELD_CONTRACTS",
    "LOCAL_FIELD_NAMES",
    "METADATA_MAX_ENTRIES",
    "METADATA_MAX_KEY_LENGTH",
    "METADATA_MAX_VALUE_LENGTH",
    "OFFICIAL_CONTRACTS",
    "OFFICIAL_FIELD_NAMES",
    "RESERVED_METADATA_KEYS",
    "SDK_DISTRIBUTION",
    "SDK_PARAMS_MODULE",
    "UNSUPPORTED_FIELD_NAMES",
    "FieldContract",
    "FieldSupport",
    "FieldSurface",
    "ResponsesMappingError",
    "capability_profile",
    "local_setting",
    "local_settings",
    "official_sdk_fields",
    "sdk_version",
    "validate_client_metadata",
    "validate_field_combinations",
    "validate_top_level_fields",
]
