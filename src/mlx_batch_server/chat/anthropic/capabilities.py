"""One capability authority for the Anthropic Messages surface.

Schema recognition is not a capability claim. ``anthropic_schema`` names what
the official wire can carry; this module decides whether the *selected*
alias and canonical runtime role can honour it, and the existing mappers
execute only what is admitted here.

Three statuses exist, and every represented field carries exactly one:

``implemented``
    A named semantic owner executes the field.
``normalized``
    The field is accepted and deliberately reshaped, and the reshape is
    stated in the classification rather than hidden.
``unsupported``
    No owner exists yet. The request is refused before inference and, for
    ``stream=true``, before a single SSE byte exists.

A field is never admitted merely because a Pydantic model contains it: a key
absent from the profile table is a policy defect, reported as ``api_error``,
not silently allowed through.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Final

from mlx_batch_server.runtime.contracts import RoleName
from mlx_batch_server.vision.input import MediaSourceField

from .anthropic_schema import (
    AnthropicTool,
    InputMessage,
    MessagesRequest,
    RequestContainerUploadBlock,
    RequestDocumentBlock,
    RequestImageBlock,
    RequestRedactedThinkingBlock,
    RequestSearchResultBlock,
    RequestServerToolUseBlock,
    RequestTextBlock,
    RequestThinkingBlock,
    RequestToolResultBlock,
    RequestToolUseBlock,
    RequestWebSearchToolResultBlock,
    ThinkingConfigEnabled,
)
from .errors import AnthropicAPIError, UnsupportedCapabilityError

#: Role stamped on the profile when this process has composed no runtime
#: graph and the protocol surface is driven through the documented
#: substitution seam. It is a real, named role — not an absence — so a
#: profile is never inherited from whatever ran last.
DETACHED_ROLE: Final = "detached"

#: Capabilities credited to the substitution seam. Only the protocol batons
#: already admitted by W1/W2 are listed: text generation, client-defined
#: tools and the tool-result ABI.
_DETACHED_CAPABILITIES: Final[tuple[str, ...]] = ("text", "tools")

_CANONICAL_ROLES: Final[frozenset[str]] = frozenset(
    {role.value for role in RoleName} | {DETACHED_ROLE}
)


class CapabilityStatus(StrEnum):
    IMPLEMENTED = "implemented"
    NORMALIZED = "normalized"
    UNSUPPORTED = "unsupported"


class CapabilityPolicyError(AnthropicAPIError):
    """A represented field reached preflight with no classification.

    This is a defect in the profile table, not a client mistake, so it is
    reported as ``api_error`` rather than blamed on the request.
    """

    def __init__(self, field_key: str, wire_path: str) -> None:
        super().__init__(
            f"{wire_path} carries no capability classification for "
            f"{field_key!r}; the request is refused rather than answered "
            "with an unproven field.",
            error_type="api_error",
        )
        self.field_key = field_key


@dataclass(frozen=True, slots=True)
class FieldClassification:
    """What one official field means on one profile."""

    key: str
    status: CapabilityStatus
    owner: str
    detail: str = ""


@dataclass(frozen=True, slots=True)
class AnthropicCapabilityProfile:
    """Immutable capability authority for one alias on one runtime role."""

    alias: str
    role: str
    backend: str | None
    declared_capabilities: tuple[str, ...]
    fields: Mapping[str, FieldClassification]
    media_source_fields: frozenset[MediaSourceField] = field(default_factory=frozenset)

    def classification(self, key: str, wire_path: str) -> FieldClassification:
        entry = self.fields.get(key)
        if entry is None:
            raise CapabilityPolicyError(key, wire_path)
        return entry

    def supports(self, key: str) -> bool:
        entry = self.fields.get(key)
        return entry is not None and entry.status is not CapabilityStatus.UNSUPPORTED


@dataclass(frozen=True, slots=True)
class CapabilityAdmission:
    """The single classification result shared by both HTTP transports.

    Produced once per request, before model acquisition and before a
    ``StreamingResponse`` exists. The engine consumes this receipt instead of
    re-deciding, so streaming and non-streaming cannot drift apart.
    """

    profile: AnthropicCapabilityProfile
    admitted: tuple[FieldClassification, ...]

    @property
    def normalized(self) -> tuple[FieldClassification, ...]:
        return tuple(
            entry
            for entry in self.admitted
            if entry.status is CapabilityStatus.NORMALIZED
        )


@dataclass(frozen=True, slots=True)
class RuntimeRoleReceipt:
    """Read-only alias-to-role receipt.

    Producing it acquires nothing: it reads the alias map and role directory
    the composed runtime already published. It is not a second alias
    registry — it holds no state of its own beyond the snapshot it was built
    from.
    """

    aliases: Mapping[str, str]
    capabilities: Mapping[str, tuple[str, ...]]
    backends: Mapping[str, str]
    media_source_fields: Mapping[str, frozenset[MediaSourceField]]


def role_receipt(runtime: object | None) -> RuntimeRoleReceipt | None:
    """Read the alias-to-role receipt off a composed runtime, or ``None``.

    ``None`` means this process composed no runtime graph, not that a lookup
    failed. The caller resolves that into the named ``detached`` role.
    """

    if runtime is None:
        return None
    receipt = getattr(runtime, "responses", runtime)
    raw_aliases = getattr(receipt, "public_aliases", None)
    directory = getattr(receipt, "role_directory", None)
    if not isinstance(raw_aliases, Mapping) or directory is None:
        return None
    specs = getattr(directory, "specs", None)
    if not callable(specs):
        return None

    aliases: dict[str, str] = {}
    for alias, role in raw_aliases.items():
        if not isinstance(alias, str) or not alias.strip():
            continue
        aliases[alias.strip().casefold()] = _role_value(role)

    capabilities: dict[str, tuple[str, ...]] = {}
    backends: dict[str, str] = {}
    for spec in specs():
        role = _role_value(getattr(spec, "name", ""))
        capabilities[role] = tuple(
            str(item) for item in getattr(spec, "capabilities", ()) or ()
        )
        backend = getattr(spec, "backend", None)
        backends[role] = _role_value(backend) if backend is not None else ""
    media_source_fields = _media_source_receipt(
        getattr(receipt, "media_source_fields", None)
    )
    return RuntimeRoleReceipt(
        aliases=MappingProxyType(aliases),
        capabilities=MappingProxyType(capabilities),
        backends=MappingProxyType(backends),
        media_source_fields=MappingProxyType(media_source_fields),
    )


def _role_value(role: object) -> str:
    value = getattr(role, "value", role)
    return str(value).strip().lower()


def _media_source_receipt(
    raw_receipt: object,
) -> dict[str, frozenset[MediaSourceField]]:
    """Snapshot a typed composition receipt without probing runtime objects."""

    if not isinstance(raw_receipt, Mapping):
        return {}
    sources: dict[str, frozenset[MediaSourceField]] = {}
    for raw_role, raw_fields in raw_receipt.items():
        role = _role_value(raw_role)
        try:
            sources[role] = frozenset(MediaSourceField(field) for field in raw_fields)
        except (TypeError, ValueError) as error:
            raise AnthropicAPIError(
                f"runtime media-source receipt for role {role!r} is invalid",
                error_type="api_error",
            ) from error
    return sources


def resolve_capability_profile(
    alias: str,
    *,
    receipt: RuntimeRoleReceipt | None,
) -> AnthropicCapabilityProfile:
    """Resolve the one profile that governs this request.

    Fails closed: an alias this process does not publish, or a role outside
    the canonical directory, is refused here rather than carried into the
    runtime under someone else's capabilities.
    """

    requested = (alias or "").strip()
    if not requested:
        raise AnthropicAPIError(
            "model must name a configured alias",
            error_type="invalid_request_error",
        )

    if receipt is None:
        role = DETACHED_ROLE
        declared = _DETACHED_CAPABILITIES
        backend = None
        media_source_fields: frozenset[MediaSourceField] = frozenset()
    else:
        resolved = receipt.aliases.get(requested.casefold())
        if resolved is None:
            raise AnthropicAPIError(
                f"model alias {requested!r} is not configured on this runtime",
                error_type="invalid_request_error",
            )
        role = resolved
        declared = receipt.capabilities.get(resolved, ())
        backend = receipt.backends.get(resolved) or None
        media_source_fields = receipt.media_source_fields.get(resolved, frozenset())

    if role not in _CANONICAL_ROLES:
        raise AnthropicAPIError(
            f"model alias {requested!r} resolves to role {role!r}, which is "
            "not a canonical runtime role",
            error_type="invalid_request_error",
        )

    return AnthropicCapabilityProfile(
        alias=requested,
        role=role,
        backend=backend,
        declared_capabilities=tuple(declared),
        media_source_fields=media_source_fields,
        fields=_field_table(declared, media_source_fields),
    )


# ---------------------------------------------------------------------------
# The profile table
# ---------------------------------------------------------------------------


def _implemented(key: str, owner: str, detail: str = "") -> FieldClassification:
    return FieldClassification(key, CapabilityStatus.IMPLEMENTED, owner, detail)


def _normalized(key: str, owner: str, detail: str) -> FieldClassification:
    return FieldClassification(key, CapabilityStatus.NORMALIZED, owner, detail)


def _unsupported(key: str, owner: str, detail: str) -> FieldClassification:
    return FieldClassification(key, CapabilityStatus.UNSUPPORTED, owner, detail)


_MAPPER: Final = "request_mapper.build_turn"
_ROUTER: Final = "anthropic.router"
_W2_STOP: Final = "W2 stop-sequence owner"
_W2_TOOL_RESULT: Final = "W2 tool_result ABI owner"
_W3_AB: Final = "W3-AB extended-thinking tier"
_W3_AC: Final = "W3-AC rich-input form"

_NO_SEMANTIC_OWNER: Final = (
    "No semantic owner executes it on this runtime, so the request is "
    "refused instead of answered as though the control had applied."
)

_BASE_FIELDS: Final[tuple[FieldClassification, ...]] = (
    _implemented("model", "capabilities.resolve_capability_profile"),
    _implemented("messages", _MAPPER),
    _implemented("max_tokens", _MAPPER, "carried as sampling.max_output_tokens"),
    _implemented("stream", _ROUTER),
    _implemented("temperature", _MAPPER),
    _implemented("top_p", _MAPPER),
    _implemented("top_k", _MAPPER),
    _implemented("stop_sequences", _W2_STOP, "matched exactly, never truncated"),
    _normalized(
        "system",
        _MAPPER,
        "a list of system text blocks is joined with newlines into one system turn",
    ),
    _implemented("tool_choice", _MAPPER),
    _implemented("tools.custom", _MAPPER),
    _implemented(
        "tools.web_fetch_20250910",
        "hosted runtime plus request_mapper.build_turn",
        "the exact basic text-family web_fetch server tool",
    ),
    _normalized(
        "tools.web_fetch.max_uses",
        _MAPPER,
        "bounded to the admitted eight-round hosted runtime ceiling",
    ),
    _normalized(
        "tools.web_fetch.allowed_domains",
        _MAPPER,
        "IDNA-normalized, lower-cased, deduplicated and mutually exclusive",
    ),
    _normalized(
        "tools.web_fetch.blocked_domains",
        _MAPPER,
        "IDNA-normalized, lower-cased, deduplicated and mutually exclusive",
    ),
    _implemented("tools.web_fetch.citations", "runtime citation binder"),
    _normalized(
        "tools.web_fetch.max_content_tokens",
        _MAPPER,
        "bounded before inference and carried on the neutral hosted descriptor",
    ),
    _unsupported(
        "tools.web_fetch.allowed_callers",
        "hosted caller routing",
        _NO_SEMANTIC_OWNER,
    ),
    _unsupported(
        "tools.web_fetch.defer_loading",
        "deferred tool loading",
        _NO_SEMANTIC_OWNER,
    ),
    _unsupported(
        "tools.web_fetch.strict",
        "strict server-tool validation",
        _NO_SEMANTIC_OWNER,
    ),
    _implemented("content.text", _MAPPER),
    _implemented("content.tool_use", _MAPPER),
    _implemented("content.tool_result", _W2_TOOL_RESULT),
    _normalized(
        "metadata",
        _MAPPER,
        "carried as runtime metadata only; it never steers inference",
    ),
    _normalized(
        "metadata.user_id",
        _MAPPER,
        "carried as runtime metadata only; it never steers inference",
    ),
    _normalized(
        "service_tier",
        _MAPPER,
        "this runtime serves exactly one local capacity tier; both 'auto' "
        "and 'standard_only' resolve to it and the request is recorded, not "
        "routed",
    ),
    _normalized(
        "thinking",
        _MAPPER,
        "only the disabled form is honoured; see thinking.enabled",
    ),
    _normalized(
        "thinking.disabled",
        _MAPPER,
        "reasoning is switched off for the turn",
    ),
    _unsupported(
        "thinking.enabled",
        _W3_AB,
        "budget_tokens has no owner on this runtime. Accepting it would "
        "report a reasoning budget that nothing enforces.",
    ),
    _unsupported(
        "content.thinking",
        _W3_AB,
        "Thinking continuation carries a signature this runtime cannot "
        "verify or re-emit.",
    ),
    _unsupported(
        "content.redacted_thinking",
        _W3_AB,
        "Redacted thinking carries opaque data this runtime cannot restore.",
    ),
    _unsupported(
        "content.image",
        "canonical media-source receipt",
        "Image support is source-specific; this coarse key never grants all "
        "image source forms.",
    ),
    _unsupported(
        "content.document",
        "canonical media-source receipt",
        "Document support is source-specific; this coarse key never grants all "
        "document source forms.",
    ),
    _normalized(
        "content.search_result",
        _W3_AC,
        "caller-supplied provenance is delimited as untrusted input_text and "
        "its URL is never interpreted as media or fetch authorization",
    ),
    _unsupported(
        "content.server_tool_use",
        "hosted tool execution",
        "This runtime executes no Anthropic-hosted server tools.",
    ),
    _unsupported(
        "content.web_search_tool_result",
        "hosted tool execution",
        "This runtime executes no Anthropic-hosted server tools.",
    ),
    _unsupported(
        "content.container_upload",
        "container execution",
        "This runtime operates no Anthropic code-execution container.",
    ),
    _unsupported(
        "tools.hosted",
        "hosted tool execution",
        "Only client-defined (custom) tools execute here.",
    ),
    _unsupported("cache_control", "prompt-cache semantics", _NO_SEMANTIC_OWNER),
    _unsupported("citations", "citation semantics", _NO_SEMANTIC_OWNER),
    _unsupported("container", "container execution", _NO_SEMANTIC_OWNER),
    _unsupported("inference_geo", "inference geography", _NO_SEMANTIC_OWNER),
    _unsupported("output_config", "structured-output execution", _NO_SEMANTIC_OWNER),
    _unsupported(
        "output_config.format", "structured-output execution", _NO_SEMANTIC_OWNER
    ),
    _unsupported("output_config.effort", "effort scheduling", _NO_SEMANTIC_OWNER),
    _unsupported("effort", "effort scheduling", _NO_SEMANTIC_OWNER),
)

_TOOLS_WITHOUT_ROLE_SUPPORT: Final[tuple[FieldClassification, ...]] = (
    _unsupported(
        "tools",
        "runtime role capability directory",
        "The selected runtime role declares no tool capability.",
    ),
    _unsupported(
        "tool_choice",
        "runtime role capability directory",
        "The selected runtime role declares no tool capability.",
    ),
    _unsupported(
        "tools.custom",
        "runtime role capability directory",
        "The selected runtime role declares no tool capability.",
    ),
    _unsupported(
        "tools.web_fetch_20250910",
        "runtime role capability directory",
        "The selected runtime role declares no tool capability.",
    ),
    _unsupported(
        "content.tool_use",
        "runtime role capability directory",
        "The selected runtime role declares no tool capability.",
    ),
    _unsupported(
        "content.tool_result",
        "runtime role capability directory",
        "The selected runtime role declares no tool capability.",
    ),
)

_MEDIA_CLASSIFICATIONS: Final[tuple[tuple[str, MediaSourceField], ...]] = (
    ("content.image.image_url", MediaSourceField.IMAGE_URL),
    ("content.image.image_base64", MediaSourceField.IMAGE_BASE64),
    ("content.image.file_id", MediaSourceField.FILE_ID),
    ("content.document.file_url", MediaSourceField.FILE_URL),
    ("content.document.file_data", MediaSourceField.FILE_DATA),
    ("content.document.file_id", MediaSourceField.FILE_ID),
)


def _field_table(
    declared: Sequence[str],
    media_source_fields: frozenset[MediaSourceField],
) -> Mapping[str, FieldClassification]:
    """Build the immutable classification table for one role.

    The tool axis comes from the signed role manifest. Rich media varies only
    by the exact source fields published by the composition receipt. Nothing
    here reads runtime state or inspects a preparer.
    """

    table: dict[str, FieldClassification] = {entry.key: entry for entry in _BASE_FIELDS}
    table["tools"] = _implemented("tools", _MAPPER)
    if "tools" not in {str(item).strip().lower() for item in declared}:
        for entry in _TOOLS_WITHOUT_ROLE_SUPPORT:
            table[entry.key] = entry
    for key, source in _MEDIA_CLASSIFICATIONS:
        if source in media_source_fields:
            table[key] = _implemented(key, _W3_AC)
        else:
            table[key] = _unsupported(
                key,
                "canonical media-source receipt",
                f"The selected runtime role does not accept {source.value!r}.",
            )
    table["content.document.unsupported_source"] = _unsupported(
        "content.document.unsupported_source",
        _W3_AC,
        "This document source type has no exact canonical file ABI.",
    )
    return MappingProxyType(table)


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


def enforce_capabilities(
    request: MessagesRequest,
    profile: AnthropicCapabilityProfile,
) -> CapabilityAdmission:
    """Classify every represented field this request actually carries.

    Raises on the first unsupported field, naming its exact wire location.
    Returns the admission receipt both transports then share.
    """

    admitted: list[FieldClassification] = []
    for key, wire_path in _presented(request):
        entry = profile.classification(key, wire_path)
        if entry.status is CapabilityStatus.UNSUPPORTED:
            raise UnsupportedCapabilityError(
                f"Anthropic {entry.key} at {wire_path}",
                f"{entry.detail} Owner: {entry.owner}.",
            )
        admitted.append(entry)
    return CapabilityAdmission(profile=profile, admitted=tuple(admitted))


def _presented(request: MessagesRequest) -> Iterator[tuple[str, str]]:
    """Yield ``(classification key, wire path)`` for every field carried."""

    yield from _envelope(request)
    for index, message in enumerate(request.messages):
        yield from _message(message, f"messages.{index}")


def _envelope(request: MessagesRequest) -> Iterator[tuple[str, str]]:
    """Yield the top-level controls this request carries."""

    yield "model", "model"
    yield "messages", "messages"
    yield "max_tokens", "max_tokens"
    yield "stream", "stream"

    for name in (
        "temperature",
        "top_p",
        "top_k",
        "stop_sequences",
        "service_tier",
        "container",
        "inference_geo",
        "effort",
    ):
        if getattr(request, name) is not None:
            yield name, name

    yield from _system(request)
    yield from _metadata(request)
    yield from _output_config(request)
    yield from _thinking(request)

    if request.tool_choice is not None:
        yield "tool_choice", "tool_choice"
    if request.tools is not None:
        yield "tools", "tools"
        for index, tool in enumerate(request.tools):
            yield from _tool(tool, f"tools.{index}")


def _system(request: MessagesRequest) -> Iterator[tuple[str, str]]:
    if request.system is None:
        return
    yield "system", "system"
    if isinstance(request.system, str):
        return
    for index, block in enumerate(request.system):
        yield from _cache_control(block, f"system.{index}")


def _metadata(request: MessagesRequest) -> Iterator[tuple[str, str]]:
    if request.metadata is None:
        return
    yield "metadata", "metadata"
    if request.metadata.user_id is not None:
        yield "metadata.user_id", "metadata.user_id"


def _output_config(request: MessagesRequest) -> Iterator[tuple[str, str]]:
    config = request.output_config
    if config is None:
        return
    # The specific control is yielded before its container so the client is
    # told which knob has no owner, not merely that the envelope is refused.
    if config.format is not None:
        yield "output_config.format", "output_config.format"
    if config.effort is not None:
        yield "output_config.effort", "output_config.effort"
    yield "output_config", "output_config"


def _thinking(request: MessagesRequest) -> Iterator[tuple[str, str]]:
    if request.thinking is None:
        return
    yield "thinking", "thinking"
    if isinstance(request.thinking, ThinkingConfigEnabled):
        yield "thinking.enabled", "thinking.type"
    else:
        yield "thinking.disabled", "thinking.type"


def _tool(tool: AnthropicTool, path: str) -> Iterator[tuple[str, str]]:
    declared = (tool.type or "custom").strip()
    if declared == "custom":
        yield "tools.custom", f"{path}.type"
    elif declared == "web_fetch_20250910":
        yield "tools.web_fetch_20250910", f"{path}.type"
        for field_name in (
            "max_uses",
            "allowed_domains",
            "blocked_domains",
            "allowed_callers",
            "defer_loading",
            "max_content_tokens",
            "strict",
        ):
            if getattr(tool, field_name) is not None:
                yield f"tools.web_fetch.{field_name}", f"{path}.{field_name}"
        if tool.citations is not None:
            yield "tools.web_fetch.citations", f"{path}.citations.enabled"
    else:
        yield "tools.hosted", f"{path}.type"
    yield from _cache_control(tool, f"{path}")


def _message(message: InputMessage, path: str) -> Iterator[tuple[str, str]]:
    if isinstance(message.content, str):
        yield "content.text", f"{path}.content"
        return
    for index, block in enumerate(message.content):
        block_path = f"{path}.content.{index}"
        if isinstance(block, RequestImageBlock | RequestDocumentBlock):
            yield _direct_media_classification(block, block_path)
        else:
            yield _block_key(block), f"{block_path}.type"
        yield from _cache_control(block, block_path)
        if isinstance(block, RequestToolResultBlock):
            # The nested tool_result ABI keeps its W2 owner; preflight only
            # walks it for controls that owner does not decide.
            yield from _nested_tool_result(block, block_path)
        elif (
            isinstance(block, RequestDocumentBlock | RequestSearchResultBlock)
            and block.citations is not None
            and block.citations.enabled
        ):
            yield "citations", f"{block_path}.citations.enabled"


def _nested_tool_result(
    block: RequestToolResultBlock,
    path: str,
) -> Iterator[tuple[str, str]]:
    if isinstance(block.content, str):
        return
    for index, item in enumerate(block.content):
        yield from _cache_control(item, f"{path}.content.{index}")


def _cache_control(block: object, path: str) -> Iterator[tuple[str, str]]:
    if getattr(block, "cache_control", None) is not None:
        yield "cache_control", f"{path}.cache_control"


def _direct_media_classification(
    block: RequestImageBlock | RequestDocumentBlock,
    path: str,
) -> tuple[str, str]:
    """Name only the source field the caller supplied, at its wire path."""

    source_type = block.source.type
    if isinstance(block, RequestImageBlock):
        fields = {
            "url": ("content.image.image_url", "url"),
            "base64": ("content.image.image_base64", "data"),
            "file": ("content.image.file_id", "file_id"),
        }
        key, field = fields[source_type]
    else:
        fields = {
            "url": ("content.document.file_url", "url"),
            "base64": ("content.document.file_data", "data"),
            "file": ("content.document.file_id", "file_id"),
        }
        selected = fields.get(source_type)
        if selected is None:
            return "content.document.unsupported_source", f"{path}.source.type"
        key, field = selected
    return key, f"{path}.source.{field}"


#: Every represented request-content block maps to exactly one classification
#: key. A block type added to the schema without an entry here resolves to a
#: key absent from the profile table, which fails closed rather than passing
#: unclassified.
_BLOCK_KEYS: Final[Mapping[type, str]] = MappingProxyType(
    {
        RequestTextBlock: "content.text",
        RequestImageBlock: "content.image",
        RequestDocumentBlock: "content.document",
        RequestSearchResultBlock: "content.search_result",
        RequestThinkingBlock: "content.thinking",
        RequestRedactedThinkingBlock: "content.redacted_thinking",
        RequestToolUseBlock: "content.tool_use",
        RequestToolResultBlock: "content.tool_result",
        RequestServerToolUseBlock: "content.server_tool_use",
        RequestWebSearchToolResultBlock: "content.web_search_tool_result",
        RequestContainerUploadBlock: "content.container_upload",
    }
)


def _block_key(block: object) -> str:
    key = _BLOCK_KEYS.get(type(block))
    if key is not None:
        return key
    # An unrepresented block cannot be admitted by omission.
    return f"content.{getattr(block, 'type', type(block).__name__)}"


def detached_profile(alias: str) -> AnthropicCapabilityProfile:
    """The profile for a surface driven through the substitution seam."""

    return resolve_capability_profile(alias, receipt=None)


__all__ = [
    "DETACHED_ROLE",
    "AnthropicCapabilityProfile",
    "CapabilityAdmission",
    "CapabilityPolicyError",
    "CapabilityStatus",
    "FieldClassification",
    "RuntimeRoleReceipt",
    "detached_profile",
    "enforce_capabilities",
    "resolve_capability_profile",
    "role_receipt",
]
