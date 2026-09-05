"""Fail-closed Responses request mapping for the canonical runtime."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from ..runtime.contracts import GenerationRequest, RuntimeKey
from .compaction import LocalCompactionCodec, expand_compaction_input
from .controller import PreparedResponse, ResponseProjection
from .normalizer import normalise_responses_payload

_ROLES = frozenset(("system", "developer", "user", "assistant", "tool"))
_TEXT_TYPES = frozenset(("text", "input_text", "output_text"))
_MEDIA_TYPES = frozenset(("input_image", "input_file", "input_audio", "input_video"))
_PART_TYPES = _TEXT_TYPES | _MEDIA_TYPES
_MESSAGE_ITEM_TYPE = "message"
_FUNCTION_CALL_ITEM_TYPE = "function_call"
_FUNCTION_OUTPUT_ITEM_TYPE = "function_call_output"
_ITEM_STATUSES = frozenset(("in_progress", "completed", "incomplete"))
_FUNCTION_CALL_FIELDS = frozenset(
    {"type", "id", "call_id", "name", "arguments", "status", "namespace"}
)
_FUNCTION_OUTPUT_FIELDS = frozenset({"type", "id", "call_id", "output", "status"})
# Inherited lineage arrives already canonical, so it also carries the canonical
# role and rendered content the mapper itself wrote on the previous round.
_CANONICAL_ITEM_FIELDS = frozenset({"role", "content"})
_FUNCTION_CALL_LINEAGE_FIELDS = _FUNCTION_CALL_FIELDS | _CANONICAL_ITEM_FIELDS
_FUNCTION_OUTPUT_LINEAGE_FIELDS = _FUNCTION_OUTPUT_FIELDS | _CANONICAL_ITEM_FIELDS
_FUNCTION_OUTPUT_CONTENT_FIELDS: Mapping[str, frozenset[str]] = {
    "input_text": frozenset({"type", "text"}),
    "input_image": frozenset({"type", "detail", "file_id", "image_url"}),
    "input_file": frozenset(
        {"type", "detail", "file_data", "file_id", "file_url", "filename"}
    ),
}
_SUPPORTED_TOP_LEVEL_FIELDS = frozenset(
    {
        "adapter_path",
        "backend",
        "cancel_on_disconnect",
        "draft_model",
        "draft_model_id",
        "frequency_penalty",
        "id",
        "input",
        "instructions",
        "max_output_tokens",
        "max_tokens",
        "metadata",
        "model",
        "model_revision",
        "owner_id",
        "parallel_tool_calls",
        "presence_penalty",
        "previous_response_id",
        "reasoning",
        "response_format",
        "response_id",
        "revision",
        "runtime_role",
        "seed",
        "stop",
        "store",
        "stream",
        "system_instruction",
        "temperature",
        "text",
        "tool_choice",
        "tools",
        "top_k",
        "top_p",
    }
)


class _FrozenDict(dict[str, Any]):
    """JSON-compatible mapping that rejects mutation after construction."""

    @staticmethod
    def _immutable(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise TypeError("canonical request mappings are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __ior__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable


class ResponsesMappingError(ValueError):
    """A request cannot be mapped without guessing or changing ownership."""

    def __init__(self, message: str, *, code: str, param: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.param = param


@dataclass(frozen=True, slots=True)
class ResolvedRuntime:
    """Trusted alias-to-role resolution returned by the target control plane."""

    runtime: RuntimeKey
    requested_model: str
    role: str


class RuntimeKeyResolver(Protocol):
    """Resolve one exact model/role selection into one runtime identity."""

    def __call__(
        self,
        *,
        model: str,
        role: str | None,
        revision: str | None,
        adapter_path: str | None,
        draft_model_id: str | None,
        backend: str | None,
    ) -> RuntimeKey | ResolvedRuntime: ...


class ProjectionFactory(Protocol):
    """Create isolated projection state without coupling to event constructors."""

    def __call__(self, prepared: PreparedResponse) -> ResponseProjection: ...


class CanonicalResponsesMapper:
    """Map normalized Responses input onto immutable runtime request boundaries."""

    def __init__(
        self,
        *,
        resolve_runtime: RuntimeKeyResolver,
        projection_factory: ProjectionFactory,
        compaction_codec: LocalCompactionCodec | None = None,
    ) -> None:
        if not callable(resolve_runtime):
            raise TypeError("resolve_runtime must be callable")
        if not callable(projection_factory):
            raise TypeError("projection_factory must be callable")
        self._resolve_runtime = resolve_runtime
        self._projection_factory = projection_factory
        self._compaction_codec = compaction_codec

    def prepare(
        self,
        payload: Mapping[str, Any],
        *,
        response_id: str,
        owner_id: str,
        parent_messages: Sequence[Mapping[str, Any]],
    ) -> PreparedResponse:
        if not isinstance(payload, Mapping):
            raise TypeError("response payload must be a mapping")
        response = _required_string(response_id, "response_id")
        owner = _required_string(owner_id, "owner_id")
        raw = _mutable_mapping(payload, "payload")
        _validate_ownership_claims(raw, response_id=response, owner_id=owner)
        raw["input"] = expand_compaction_input(
            raw.get("input"),
            owner_id=owner,
            codec=self._compaction_codec,
        )
        _validate_supported_fields(raw)
        _validate_raw_input(raw.get("input"))

        model = _required_string(raw.get("model"), "model")
        role = _runtime_role(raw)
        revision = _coalesced_optional_string(
            raw, "revision", "model_revision", param="revision"
        )
        adapter_path = _optional_string(raw.get("adapter_path"), "adapter_path")
        draft_model_id = _coalesced_optional_string(
            raw, "draft_model_id", "draft_model", param="draft_model_id"
        )
        backend_claim = _optional_string(raw.get("backend"), "backend")

        resolved = self._resolve_runtime(
            model=model,
            role=role,
            revision=revision,
            adapter_path=adapter_path,
            draft_model_id=draft_model_id,
            backend=backend_claim,
        )
        runtime, resolved_role = _validate_runtime_resolution(
            resolved,
            model=model,
            role=role,
            revision=revision,
            adapter_path=adapter_path,
            draft_model_id=draft_model_id,
            backend_claim=backend_claim,
        )

        normalised = normalise_responses_payload(raw)
        current = _current_messages(normalised, raw_input=raw.get("input"))
        parents = tuple(
            _canonical_lineage_item(item, "parent_messages") for item in parent_messages
        )
        _validate_call_lineage(parents, current)
        instructions = _instruction_messages(raw)
        lineage_messages = _instructions_first((*parents, *current))
        materialized = _instructions_first((*instructions, *lineage_messages))

        messages = tuple(_text_message(item) for item in materialized)
        media = _media_parts(materialized)
        tools = _tools(raw.get("tools"))
        sampling = _sampling(raw, has_tools=bool(tools))
        reasoning = _reasoning(raw.get("reasoning"))
        metadata = _metadata(
            raw,
            model=model,
            role=resolved_role,
            resolved_model=runtime.model_id,
        )
        store = _boolean(raw.get("store", True), "store")
        cancel_on_disconnect = _boolean(
            raw.get("cancel_on_disconnect", True), "cancel_on_disconnect"
        )

        request = GenerationRequest(
            response_id=response,
            runtime=runtime,
            messages=messages,
            media=media,
            tools=tools,
            sampling=sampling,
            reasoning=reasoning,
            lineage=parents,
            metadata=metadata,
        )
        return PreparedResponse(
            request=request,
            materialized_messages=materialized,
            lineage_messages=lineage_messages,
            store=store,
            cancel_on_disconnect=cancel_on_disconnect,
        )

    def start_projection(self, prepared: PreparedResponse) -> ResponseProjection:
        if not isinstance(prepared, PreparedResponse):
            raise TypeError("projection requires PreparedResponse")
        projection = self._projection_factory(prepared)
        if not isinstance(projection, ResponseProjection):
            raise TypeError("projection_factory must return ResponseProjection")
        return projection


def _validate_ownership_claims(
    payload: Mapping[str, Any], *, response_id: str, owner_id: str
) -> None:
    for field, expected in (("id", response_id), ("response_id", response_id)):
        claim = payload.get(field)
        if claim is not None and claim != expected:
            raise ResponsesMappingError(
                f"{field} does not belong to this response",
                code="response_ownership_mismatch",
                param=field,
            )
    owner_claim = payload.get("owner_id")
    if owner_claim is not None and owner_claim != owner_id:
        raise ResponsesMappingError(
            "owner_id does not match the authenticated owner",
            code="response_ownership_mismatch",
            param="owner_id",
        )
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        metadata_owner = metadata.get("owner_id")
        if metadata_owner is not None and metadata_owner != owner_id:
            raise ResponsesMappingError(
                "metadata.owner_id does not match the authenticated owner",
                code="response_ownership_mismatch",
                param="metadata.owner_id",
            )
        metadata_response = metadata.get("response_id")
        if metadata_response is not None and metadata_response != response_id:
            raise ResponsesMappingError(
                "metadata.response_id does not belong to this response",
                code="response_ownership_mismatch",
                param="metadata.response_id",
            )


def _validate_supported_fields(payload: Mapping[str, Any]) -> None:
    unsupported = sorted(set(payload) - _SUPPORTED_TOP_LEVEL_FIELDS)
    if unsupported:
        field = unsupported[0]
        raise ResponsesMappingError(
            f"unsupported Responses parameter: {field}",
            code="unsupported_parameter",
            param=field,
        )


def _validate_raw_input(raw_input: Any) -> None:
    if raw_input is None or isinstance(raw_input, str):
        return
    if isinstance(raw_input, Mapping):
        entries: Sequence[Any] = (raw_input,)
    elif _is_sequence(raw_input):
        entries = raw_input
    else:
        raise _invalid(
            "input must be text, a message, or a sequence of messages",
            "input",
        )
    function_call_ids: set[str] = set()
    for message_index, entry in enumerate(entries):
        param = f"input[{message_index}]"
        if isinstance(entry, str):
            continue
        if not isinstance(entry, Mapping):
            raise _invalid("input entries must be text or message mappings", param)
        item_type = entry.get("type")
        if item_type == _FUNCTION_CALL_ITEM_TYPE:
            _validate_function_call(entry, param)
            continue
        if item_type == _FUNCTION_OUTPUT_ITEM_TYPE:
            call_id = _validate_function_call_output(entry, param)
            if call_id in function_call_ids:
                raise _invalid(
                    "function_call_output call_id is duplicated",
                    f"{param}.call_id",
                )
            function_call_ids.add(call_id)
            continue
        if item_type not in (None, _MESSAGE_ITEM_TYPE):
            raise _invalid("input item type is unsupported", f"{param}.type")
        allowed = {"role", "content"}
        if item_type == _MESSAGE_ITEM_TYPE:
            allowed.add("type")
            if "role" not in entry:
                raise _invalid("message role is required", f"{param}.role")
            if "content" not in entry:
                raise _invalid("message content is required", f"{param}.content")
        unknown = set(entry) - allowed
        if unknown:
            raise _invalid("unsupported input item fields", param)
        role = entry.get("role", "user")
        if not isinstance(role, str) or role.strip().lower() not in _ROLES:
            raise _invalid("input role is unknown", f"{param}.role")
        _validate_raw_content(entry.get("content"), f"{param}.content")


def _validate_function_call(
    entry: Mapping[str, Any],
    param: str,
    *,
    allowed: frozenset[str] = _FUNCTION_CALL_FIELDS,
) -> str:
    """Validate one official `function_call` item against the installed SDK shape."""

    unknown = set(entry) - allowed
    if unknown:
        raise _invalid("function_call contains unsupported fields", param)
    call_id = _required_item_identity(entry, "call_id", "function_call", param)
    _required_item_identity(entry, "name", "function_call", param)
    if not isinstance(entry.get("arguments"), str):
        raise _invalid(
            "function_call arguments must be a JSON string",
            f"{param}.arguments",
        )
    _validate_optional_item_identity(entry, "id", "function_call", param)
    _validate_optional_item_identity(entry, "namespace", "function_call", param)
    _validate_item_status(entry, "function_call", param)
    return call_id


def _validate_function_call_output(
    entry: Mapping[str, Any],
    param: str,
    *,
    allowed: frozenset[str] = _FUNCTION_OUTPUT_FIELDS,
) -> str:
    unknown = set(entry) - allowed
    if unknown:
        raise _invalid(
            "function_call_output contains unsupported fields",
            param,
        )
    if "call_id" not in entry:
        raise _invalid(
            "function_call_output call_id is required",
            f"{param}.call_id",
        )
    if "output" not in entry:
        raise _invalid(
            "function_call_output output is required",
            f"{param}.output",
        )
    call_id = _required_item_identity(
        entry, "call_id", _FUNCTION_OUTPUT_ITEM_TYPE, param
    )
    _validate_optional_item_identity(entry, "id", _FUNCTION_OUTPUT_ITEM_TYPE, param)
    _validate_item_status(entry, _FUNCTION_OUTPUT_ITEM_TYPE, param)
    _validate_function_output_union(entry["output"], f"{param}.output")
    return call_id


def _validate_function_output_union(output: Any, param: str) -> None:
    """Accept the installed SDK `output` union: text or official content items."""

    if isinstance(output, str):
        return
    if not _is_sequence(output):
        raise _invalid(
            "function_call_output output must be text or official output content",
            param,
        )
    items = tuple(output)
    if not items:
        raise _invalid(
            "function_call_output output content must not be empty",
            param,
        )
    for index, item in enumerate(items):
        item_param = f"{param}[{index}]"
        if not isinstance(item, Mapping):
            raise _invalid(
                "function_call_output output content must be mappings",
                item_param,
            )
        kind = item.get("type")
        allowed = (
            _FUNCTION_OUTPUT_CONTENT_FIELDS.get(kind) if isinstance(kind, str) else None
        )
        if allowed is None:
            raise _invalid(
                "function_call_output output content type is unsupported",
                f"{item_param}.type",
            )
        if set(item) - allowed:
            raise _invalid(
                "function_call_output output content has unsupported fields",
                item_param,
            )
        if kind == "input_text":
            if not isinstance(item.get("text"), str):
                raise _invalid(
                    "function_call_output output text must be a string",
                    f"{item_param}.text",
                )
            continue
        sources = (
            ("file_id", "image_url")
            if kind == "input_image"
            else ("file_data", "file_id", "file_url")
        )
        if not any(
            isinstance(item.get(field), str) and item[field].strip()
            for field in sources
        ):
            raise _invalid(
                "function_call_output output content has no source",
                item_param,
            )


def _required_item_identity(
    entry: Mapping[str, Any], field: str, item_type: str, param: str
) -> str:
    value = entry.get(field)
    if field not in entry:
        raise _invalid(f"{item_type} {field} is required", f"{param}.{field}")
    if not isinstance(value, str) or not value or value != value.strip():
        raise _invalid(
            f"{item_type} {field} must be a non-blank string",
            f"{param}.{field}",
        )
    return value


def _validate_optional_item_identity(
    entry: Mapping[str, Any], field: str, item_type: str, param: str
) -> None:
    if field not in entry or entry[field] is None:
        return
    _required_item_identity(entry, field, item_type, param)


def _validate_item_status(entry: Mapping[str, Any], item_type: str, param: str) -> None:
    if "status" not in entry or entry["status"] is None:
        return
    if entry["status"] not in _ITEM_STATUSES:
        raise _invalid(f"{item_type} status is unknown", f"{param}.status")


def _validate_raw_content(content: Any, param: str) -> None:
    if isinstance(content, str):
        return
    if isinstance(content, Mapping):
        parts: Sequence[Any] = (content,)
    elif _is_sequence(content):
        parts = content
    else:
        raise _invalid("message content must be text or content parts", param)
    for index, part in enumerate(parts):
        part_param = f"{param}[{index}]"
        if isinstance(part, str):
            continue
        if not isinstance(part, Mapping):
            raise _invalid("content parts must be text or mappings", part_param)
        _validate_raw_part(part, part_param)


def _validate_raw_part(part: Mapping[str, Any], param: str) -> None:
    part_type = part.get("type")
    if part_type is None:
        families = _inferred_families(part)
        if len(families) != 1:
            raise _invalid("content part type is missing or ambiguous", f"{param}.type")
        part_type = next(iter(families))
    if not isinstance(part_type, str) or part_type not in _PART_TYPES:
        raise _invalid("content part type is unsupported", f"{param}.type")

    canonical_type = "input_text" if part_type in _TEXT_TYPES else part_type
    allowed = {
        "input_text": {"type", "text"},
        "input_image": {
            "type",
            "image_url",
            "image_base64",
            "url",
            "file_id",
            "detail",
        },
        "input_file": {
            "type",
            "file_id",
            "file_url",
            "file_data",
            "filename",
            "detail",
        },
        "input_audio": {"type", "audio_url", "file_id"},
        "input_video": {"type", "video_url", "file_id"},
    }[canonical_type]
    if set(part) - allowed:
        raise _invalid("content part contains unsupported fields", param)
    if canonical_type == "input_text":
        if not isinstance(part.get("text"), str):
            raise _invalid("text content must be a string", f"{param}.text")
        return

    source_fields = {
        "input_image": ("image_url", "image_base64", "url", "file_id"),
        "input_file": ("file_id", "file_url", "file_data"),
        "input_audio": ("audio_url", "file_id"),
        "input_video": ("video_url", "file_id"),
    }[canonical_type]
    sources = [field for field in source_fields if _has_source(part.get(field))]
    if len(sources) != 1:
        raise _invalid(
            f"{canonical_type} requires exactly one source",
            param,
        )
    detail = part.get("detail")
    if detail is not None and detail not in {"auto", "low", "high", "original"}:
        raise _invalid(
            "media detail must be auto, low, high, or original",
            f"{param}.detail",
        )
    filename = part.get("filename")
    if filename is not None and not isinstance(filename, str):
        raise _invalid("filename must be a string", f"{param}.filename")


def _inferred_families(part: Mapping[str, Any]) -> set[str]:
    families: set[str] = set()
    if "text" in part:
        families.add("input_text")
    if {"image_url", "image_base64"} & set(part):
        families.add("input_image")
    if {"file_url", "file_data", "filename"} & set(part):
        families.add("input_file")
    if "audio_url" in part:
        families.add("input_audio")
    if "video_url" in part:
        families.add("input_video")
    return families


def _has_source(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        if set(value) - {"data", "url", "file_id"}:
            return False
        fields = (value.get("data"), value.get("url"), value.get("file_id"))
        return sum(isinstance(item, str) and bool(item.strip()) for item in fields) == 1
    return False


def _runtime_role(payload: Mapping[str, Any]) -> str | None:
    metadata = payload.get("metadata")
    metadata_role = (
        metadata.get("runtime_role") if isinstance(metadata, Mapping) else None
    )
    top_role = payload.get("runtime_role")
    if top_role is not None and metadata_role is not None and top_role != metadata_role:
        raise ResponsesMappingError(
            "runtime role claims disagree",
            code="runtime_role_mismatch",
            param="runtime_role",
        )
    value = top_role if top_role is not None else metadata_role
    return _optional_string(value, "runtime_role")


def _validate_runtime_resolution(
    resolved: Any,
    *,
    model: str,
    role: str | None,
    revision: str | None,
    adapter_path: str | None,
    draft_model_id: str | None,
    backend_claim: Any,
) -> tuple[RuntimeKey, str | None]:
    alias_resolution = isinstance(resolved, ResolvedRuntime)
    if alias_resolution:
        runtime = resolved.runtime
        if resolved.requested_model != model:
            raise ResponsesMappingError(
                "resolver changed the requested model claim",
                code="runtime_ownership_mismatch",
                param="model",
            )
        resolved_role = _required_string(resolved.role, "runtime_role")
        if role is not None and role != resolved_role:
            raise ResponsesMappingError(
                "runtime role cannot override the resolved model alias",
                code="runtime_role_mismatch",
                param="runtime_role",
            )
    else:
        runtime = resolved
        resolved_role = role
    if not isinstance(runtime, RuntimeKey):
        raise TypeError("resolve_runtime must return RuntimeKey or ResolvedRuntime")
    expected = {}
    if not alias_resolution:
        expected = {
            "model": (runtime.model_id, model),
            "revision": (runtime.revision, revision),
            "adapter_path": (runtime.adapter_path, adapter_path),
            "draft_model_id": (runtime.draft_model_id, draft_model_id),
        }
    for field, (actual, requested) in expected.items():
        if actual != requested:
            raise ResponsesMappingError(
                f"resolved RuntimeKey changed requested {field}",
                code="runtime_ownership_mismatch",
                param=field,
            )
    if backend_claim is not None and runtime.backend.value != backend_claim:
        raise ResponsesMappingError(
            "resolved RuntimeKey changed requested backend",
            code="runtime_ownership_mismatch",
            param="backend",
        )
    return runtime, resolved_role


def _instruction_messages(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    system = _optional_string(payload.get("system_instruction"), "system_instruction")
    developer = _optional_string(payload.get("instructions"), "instructions")
    if system is not None and developer is not None and system != developer:
        raise ResponsesMappingError(
            "system_instruction and instructions disagree",
            code="ambiguous_instructions",
            param="instructions",
        )
    if developer is not None:
        return (_message("developer", developer),)
    if system is not None:
        return (_message("system", system),)
    return ()


def _instructions_first(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Keep Qwen instruction roles ahead of inherited conversation turns."""
    instructions: list[Mapping[str, Any]] = []
    conversation: list[Mapping[str, Any]] = []
    for message in messages:
        target = (
            instructions
            if message.get("role") in {"system", "developer"}
            else conversation
        )
        target.append(message)
    return (*instructions, *conversation)


def _current_messages(
    normalised: Mapping[str, Any], *, raw_input: Any
) -> tuple[Mapping[str, Any], ...]:
    value = normalised.get("input")
    if not _is_sequence(value):
        raise _invalid("normalizer returned invalid input", "input")
    normalised_items = tuple(value)
    if isinstance(raw_input, Mapping):
        raw_items: tuple[Any, ...] = (raw_input,)
    elif _is_sequence(raw_input):
        raw_items = tuple(raw_input)
    else:
        raw_items = ()
    if not raw_items:
        return tuple(
            _canonical_message(item, f"input[{index}]")
            for index, item in enumerate(normalised_items)
        )
    if len(raw_items) != len(normalised_items):
        raise _invalid("normalizer changed input item cardinality", "input")
    current: list[Mapping[str, Any]] = []
    for index, (raw_item, normalised_item) in enumerate(
        zip(raw_items, normalised_items, strict=True)
    ):
        param = f"input[{index}]"
        if (
            isinstance(raw_item, Mapping)
            and raw_item.get("type") == _FUNCTION_CALL_ITEM_TYPE
        ):
            current.append(_canonical_function_call(raw_item, param))
        elif (
            isinstance(raw_item, Mapping)
            and raw_item.get("type") == _FUNCTION_OUTPUT_ITEM_TYPE
        ):
            current.append(_canonical_function_call_output(raw_item, param))
        else:
            message = _canonical_message(normalised_item, param)
            if (
                isinstance(raw_item, Mapping)
                and raw_item.get("type") == _MESSAGE_ITEM_TYPE
            ):
                message = _FrozenDict({"type": _MESSAGE_ITEM_TYPE, **dict(message)})
            current.append(message)
    return tuple(current)


def _canonical_function_call_output(
    value: Mapping[str, Any],
    param: str,
    *,
    allowed: frozenset[str] = _FUNCTION_OUTPUT_FIELDS,
) -> Mapping[str, Any]:
    call_id = _validate_function_call_output(value, param, allowed=allowed)
    output = _deliverable_function_output(value["output"], f"{param}.output")
    content = (_FrozenDict({"type": "input_text", "text": output}),)
    canonical: dict[str, Any] = {
        "type": _FUNCTION_OUTPUT_ITEM_TYPE,
        "role": "tool",
        "call_id": call_id,
        "output": output,
        "content": content,
    }
    _carry_item_identity(canonical, value, ("id", "status"))
    return _FrozenDict(canonical)


def _deliverable_function_output(output: Any, param: str) -> str:
    """Reduce the official output union to the text the model renderers accept.

    The union is validated in full before this point, so nothing here can
    stringify or silently drop a variant. Image, file and multi-part text
    outputs keep their wire identity in the request and are refused with an
    explicit capability error at this narrowest semantic boundary, because no
    installed renderer can carry them on a `function_call_output` item.
    """

    if isinstance(output, str):
        return output
    items = tuple(output)
    if len(items) == 1 and items[0].get("type") == "input_text":
        return str(items[0]["text"])
    raise ResponsesMappingError(
        "function_call_output output content is not supported by this runtime",
        code="unsupported_capability",
        param=param,
    )


def _canonical_function_call(
    value: Mapping[str, Any],
    param: str,
    *,
    allowed: frozenset[str] = _FUNCTION_CALL_FIELDS,
) -> Mapping[str, Any]:
    """Preserve one official `function_call` losslessly in canonical history."""

    call_id = _validate_function_call(value, param, allowed=allowed)
    name = str(value["name"])
    arguments = str(value["arguments"])
    canonical: dict[str, Any] = {
        "type": _FUNCTION_CALL_ITEM_TYPE,
        "role": "assistant",
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
        "content": (
            _FrozenDict(
                {
                    "type": "input_text",
                    "text": _function_call_text(call_id, name, arguments),
                }
            ),
        ),
    }
    _carry_item_identity(canonical, value, ("id", "status", "namespace"))
    return _FrozenDict(canonical)


def _function_call_text(call_id: str, name: str, arguments: str) -> str:
    """Render the call exactly as the official item, deterministically."""

    return json.dumps(
        {
            "type": _FUNCTION_CALL_ITEM_TYPE,
            "call_id": call_id,
            "name": name,
            "arguments": arguments,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _carry_item_identity(
    canonical: dict[str, Any],
    value: Mapping[str, Any],
    fields: tuple[str, ...],
) -> None:
    for field in fields:
        carried = value.get(field)
        if isinstance(carried, str) and carried:
            canonical[field] = carried


def _canonical_lineage_item(value: Any, param: str) -> Mapping[str, Any]:
    """Canonicalize one inherited item without losing its official identity."""

    if isinstance(value, Mapping):
        item_type = value.get("type")
        if item_type == _FUNCTION_CALL_ITEM_TYPE:
            return _canonical_function_call(
                value, param, allowed=_FUNCTION_CALL_LINEAGE_FIELDS
            )
        if item_type == _FUNCTION_OUTPUT_ITEM_TYPE:
            return _canonical_function_call_output(
                value, param, allowed=_FUNCTION_OUTPUT_LINEAGE_FIELDS
            )
    return _canonical_message(value, param)


def _validate_call_lineage(
    parents: Sequence[Mapping[str, Any]],
    current: Sequence[Mapping[str, Any]],
) -> None:
    """Reject a tool result that no preceding call in the lineage can explain."""

    seen: set[str] = set()
    for index, item in enumerate((*parents, *current)):
        item_type = item.get("type")
        if item_type == _FUNCTION_CALL_ITEM_TYPE:
            seen.add(str(item["call_id"]))
        elif item_type == _FUNCTION_OUTPUT_ITEM_TYPE:
            call_id = str(item["call_id"])
            if call_id in seen:
                continue
            param = (
                f"input[{index - len(parents)}].call_id"
                if index >= len(parents)
                else "previous_response_id"
            )
            raise _invalid(
                "no function_call in this conversation matches "
                f"function_call_output call_id {call_id!r}",
                param,
            )


def _canonical_message(value: Any, param: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _invalid("materialized message must be a mapping", param)
    role_value = value.get("role", "user")
    if not isinstance(role_value, str) or role_value.strip().lower() not in _ROLES:
        raise _invalid("materialized message role is unknown", f"{param}.role")
    role = role_value.strip().lower()
    content = value.get("content", "")
    if isinstance(content, str):
        parts: Sequence[Any] = ({"type": "input_text", "text": content},)
    elif _is_sequence(content):
        parts = content
    else:
        raise _invalid("materialized content must be text or content parts", param)
    canonical_parts: list[Mapping[str, Any]] = []
    for part in parts:
        if not isinstance(part, Mapping):
            raise _invalid("materialized content part must be a mapping", param)
        kind = part.get("type")
        if kind not in {"input_text", *_MEDIA_TYPES}:
            raise _invalid("materialized content part is not canonical", param)
        canonical_parts.append(_freeze_mapping(part))
    return _FrozenDict({"role": role, "content": tuple(canonical_parts)})


def _message(role: str, text: str) -> Mapping[str, Any]:
    part = _FrozenDict({"type": "input_text", "text": text})
    return _FrozenDict({"role": role, "content": (part,)})


def _text_message(message: Mapping[str, Any]) -> Mapping[str, Any]:
    text_parts = tuple(
        part
        for part in message["content"]
        if isinstance(part, Mapping) and part.get("type") == "input_text"
    )
    canonical: dict[str, Any] = {
        "role": message["role"],
        "content": text_parts,
    }
    message_type = message.get("type")
    if message_type == _MESSAGE_ITEM_TYPE:
        canonical["type"] = _MESSAGE_ITEM_TYPE
    elif message_type == _FUNCTION_CALL_ITEM_TYPE:
        # No installed renderer seals a `function_call` item yet, so the call is
        # delivered as the assistant turn that produced it, carrying the exact
        # official item as its text. Canonical history keeps the typed item.
        canonical["type"] = _MESSAGE_ITEM_TYPE
    elif message_type == _FUNCTION_OUTPUT_ITEM_TYPE:
        canonical.update(
            {
                "type": _FUNCTION_OUTPUT_ITEM_TYPE,
                "call_id": message["call_id"],
                "output": message["output"],
            }
        )
    return _FrozenDict(canonical)


def _media_parts(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    media: list[Mapping[str, Any]] = []
    for message_index, message in enumerate(messages):
        for content_index, part in enumerate(message["content"]):
            if part.get("type") not in _MEDIA_TYPES:
                continue
            item = dict(part)
            item["_role"] = message["role"]
            item["_message_index"] = message_index
            item["_content_index"] = content_index
            media.append(_freeze_mapping(item))
    return tuple(media)


def _tools(value: Any) -> tuple[Mapping[str, Any], ...]:
    if value is None:
        return ()
    if not _is_sequence(value):
        raise _invalid("tools must be a sequence of mappings", "tools")
    tools: list[Mapping[str, Any]] = []
    for index, tool in enumerate(value):
        if not isinstance(tool, Mapping) or not tool:
            raise _invalid("each tool must be a non-empty mapping", f"tools[{index}]")
        param = f"tools[{index}]"
        tool_type = tool.get("type")
        if tool_type != "function":
            raise ResponsesMappingError(
                "only client-owned function tools are supported locally",
                code="unsupported_tool",
                param=f"{param}.type",
            )
        unknown = set(tool) - {
            "type",
            "name",
            "description",
            "parameters",
            "strict",
        }
        if unknown:
            raise ResponsesMappingError(
                "function tool contains unsupported fields",
                code="unsupported_parameter",
                param=f"{param}.{sorted(unknown)[0]}",
            )
        name = tool.get("name")
        if not isinstance(name, str) or not name.strip() or name != name.strip():
            raise _invalid("function tool name must be non-blank", f"{param}.name")
        parameters = tool.get("parameters")
        if parameters is not None and not isinstance(parameters, Mapping):
            raise _invalid(
                "function tool parameters must be a mapping",
                f"{param}.parameters",
            )
        strict = tool.get("strict")
        if strict is not None and not isinstance(strict, bool):
            raise _invalid("function tool strict must be a boolean", f"{param}.strict")
        tools.append(_freeze_mapping(tool))
    return tuple(tools)


def _sampling(payload: Mapping[str, Any], *, has_tools: bool) -> Mapping[str, Any]:
    sampling: dict[str, Any] = {}
    max_tokens = _coalesced_value(
        payload, "max_output_tokens", "max_tokens", param="max_output_tokens"
    )
    if max_tokens is not None:
        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens < 1
        ):
            raise _invalid(
                "max_output_tokens must be a positive integer",
                "max_output_tokens",
            )
        sampling["max_output_tokens"] = max_tokens
    for field in (
        "temperature",
        "top_p",
        "top_k",
        "presence_penalty",
        "frequency_penalty",
    ):
        value = payload.get(field)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise _invalid(f"{field} must be numeric", field)
        sampling[field] = value
    if "top_p" in sampling and not 0 <= sampling["top_p"] <= 1:
        raise _invalid("top_p must be between zero and one", "top_p")
    if "top_k" in sampling and (
        not isinstance(sampling["top_k"], int) or sampling["top_k"] < 1
    ):
        raise _invalid("top_k must be a positive integer", "top_k")
    stop = payload.get("stop")
    if stop is not None:
        if isinstance(stop, str):
            stops = (stop,)
        elif _is_sequence(stop) and all(isinstance(item, str) for item in stop):
            stops = tuple(stop)
        else:
            raise _invalid("stop must be text or a sequence of text", "stop")
        sampling["stop"] = stops
    seed = payload.get("seed")
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise _invalid("seed must be an integer", "seed")
        sampling["seed"] = seed
    parallel_tool_calls = payload.get("parallel_tool_calls")
    if parallel_tool_calls is not None:
        if not isinstance(parallel_tool_calls, bool):
            raise _invalid(
                "parallel_tool_calls must be a boolean",
                "parallel_tool_calls",
            )
        sampling["parallel_tool_calls"] = parallel_tool_calls
    tool_choice = payload.get("tool_choice")
    if tool_choice is not None:
        _validate_tool_choice(tool_choice, has_tools=has_tools)
        sampling["tool_choice"] = _freeze(tool_choice)
    for field in ("response_format", "text"):
        if field in payload and payload[field] is not None:
            if not isinstance(payload[field], Mapping):
                raise _invalid(f"{field} must be a mapping", field)
            sampling[field] = _freeze(payload[field])
    return _FrozenDict(sampling)


def _validate_tool_choice(tool_choice: Any, *, has_tools: bool) -> None:
    if not isinstance(tool_choice, str | Mapping):
        raise _invalid("tool_choice must be text or a mapping", "tool_choice")
    if isinstance(tool_choice, str) and tool_choice not in {
        "auto",
        "none",
        "required",
    }:
        raise ResponsesMappingError(
            "only auto, none, required, or function tool_choice is supported",
            code="unsupported_tool_choice",
            param="tool_choice",
        )
    if isinstance(tool_choice, Mapping):
        if set(tool_choice) - {"type", "name"}:
            raise ResponsesMappingError(
                "function tool_choice contains unsupported fields",
                code="unsupported_parameter",
                param="tool_choice",
            )
        if tool_choice.get("type") != "function":
            raise ResponsesMappingError(
                "only function tool_choice is supported locally",
                code="unsupported_tool_choice",
                param="tool_choice.type",
            )
        name = tool_choice.get("name")
        if not isinstance(name, str) or not name.strip():
            raise _invalid(
                "function tool_choice name must be non-blank",
                "tool_choice.name",
            )
    if not has_tools and tool_choice != "none":
        raise _invalid("tool_choice requires tools", "tool_choice")


def _metadata(
    payload: Mapping[str, Any],
    *,
    model: str,
    role: str | None,
    resolved_model: str,
) -> Mapping[str, Any]:
    raw = payload.get("metadata")
    if raw is None:
        metadata: dict[str, Any] = {}
    elif isinstance(raw, Mapping):
        metadata = dict(raw)
    else:
        raise _invalid("metadata must be a mapping", "metadata")
    requested_model = metadata.get("requested_model")
    if requested_model is not None and requested_model != model:
        raise ResponsesMappingError(
            "metadata.requested_model does not match model",
            code="runtime_ownership_mismatch",
            param="metadata.requested_model",
        )
    metadata["requested_model"] = model
    claimed_resolved_model = metadata.get("resolved_model")
    if claimed_resolved_model is not None and claimed_resolved_model != resolved_model:
        raise ResponsesMappingError(
            "metadata.resolved_model does not match runtime resolution",
            code="runtime_ownership_mismatch",
            param="metadata.resolved_model",
        )
    metadata["resolved_model"] = resolved_model
    if role is not None:
        metadata["runtime_role"] = role
    return _freeze_mapping(metadata)


def _mapping_or_empty(value: Any, param: str) -> Mapping[str, Any]:
    if value is None:
        return _FrozenDict()
    if not isinstance(value, Mapping):
        raise _invalid(f"{param} must be a mapping", param)
    return _freeze_mapping(value)


def _reasoning(value: Any) -> Mapping[str, Any]:
    reasoning = _mapping_or_empty(value, "reasoning")
    unsupported = sorted(set(reasoning) - {"enabled", "effort", "summary"})
    if unsupported:
        field = unsupported[0]
        raise ResponsesMappingError(
            f"unsupported Responses reasoning parameter: {field}",
            code="unsupported_parameter",
            param=f"reasoning.{field}",
        )
    enabled = reasoning.get("enabled")
    if enabled is not None and not isinstance(enabled, bool):
        raise _invalid("reasoning.enabled must be a boolean", "reasoning.enabled")
    effort = reasoning.get("effort")
    if effort is not None and effort not in {
        "none",
        "off",
        "low",
        "medium",
        "high",
        "xhigh",
    }:
        raise _invalid(
            "reasoning.effort must be one of none, off, low, medium, high, xhigh",
            "reasoning.effort",
        )
    summary = reasoning.get("summary")
    if summary is not None and summary != "auto":
        raise _invalid(
            "reasoning.summary currently supports only auto",
            "reasoning.summary",
        )
    return reasoning


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    frozen: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise _invalid("mapping keys must be strings", "mapping")
        frozen[key] = _freeze(item)
    return _FrozenDict(frozen)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if _is_sequence(value):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise _invalid("value must be JSON-compatible", "mapping")


def _mutable_mapping(value: Mapping[str, Any], param: str) -> dict[str, Any]:
    copied = _mutable_copy(value, param)
    if not isinstance(copied, dict):  # pragma: no cover - guarded by the type
        raise TypeError(f"{param} must be a mapping")
    return copied


def _mutable_copy(value: Any, param: str) -> Any:
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise _invalid("mapping keys must be strings", param)
            copied[key] = _mutable_copy(item, f"{param}.{key}")
        return copied
    if _is_sequence(value):
        return [_mutable_copy(item, param) for item in value]
    return value


def _coalesced_optional_string(
    payload: Mapping[str, Any], first: str, second: str, *, param: str
) -> str | None:
    value = _coalesced_value(payload, first, second, param=param)
    return _optional_string(value, param)


def _coalesced_value(
    payload: Mapping[str, Any], first: str, second: str, *, param: str
) -> Any:
    left = payload.get(first)
    right = payload.get(second)
    if left is not None and right is not None and left != right:
        raise ResponsesMappingError(
            f"{first} and {second} disagree",
            code="ambiguous_request_alias",
            param=param,
        )
    return left if left is not None else right


def _required_string(value: Any, param: str) -> str:
    result = _optional_string(value, param)
    if result is None:
        raise _invalid(f"{param} must be a non-empty string", param)
    return result


def _optional_string(value: Any, param: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise _invalid(f"{param} must be a non-empty string", param)
    return value.strip()


def _boolean(value: Any, param: str) -> bool:
    if not isinstance(value, bool):
        raise _invalid(f"{param} must be a boolean", param)
    return value


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, str | bytes | bytearray
    )


def _invalid(message: str, param: str) -> ResponsesMappingError:
    return ResponsesMappingError(message, code="invalid_responses_request", param=param)


__all__ = [
    "CanonicalResponsesMapper",
    "ProjectionFactory",
    "ResolvedRuntime",
    "ResponsesMappingError",
    "RuntimeKeyResolver",
]
