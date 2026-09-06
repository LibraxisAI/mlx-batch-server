"""HTTP transport for the Anthropic Messages and Models surfaces.

The router does transport work only: it validates the request against the
strict schema, mints a request id, frames server-sent events, and projects
failures onto the Anthropic error envelope. Protocol shaping lives in the
projector; inference lives behind the turn-source seam.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import AsyncIterator
from typing import Any, Final

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Security
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials  # noqa: TC002
from pydantic import ValidationError

from mlx_batch_server.auth.dependency import (
    api_key_header,
    session_scheme,
    verify_auth,
)
from mlx_batch_server.utils.logger import logger

from .anthropic_schema import (
    AnthropicStreamEvent,
    MessagesRequest,
    MessagesResponse,
)
from .capabilities import (
    enforce_capabilities,
    resolve_capability_profile,
    role_receipt,
)
from .errors import (
    ERROR_TYPE_STATUS,
    REQUEST_ID_HEADER,
    AnthropicAPIError,
    AnthropicErrorEnvelope,
    attach_request_id,
    error_payload,
    new_request_id,
)
from .messages_engine import AnthropicMessagesEngine
from .models_service import AnthropicModelsService
from .request_mapper import build_turn
from .schema import AnthropicModelList

router = APIRouter(tags=["anthropic"])

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

# This adapter deliberately implements one stable Messages wire version. A
# newer date can carry changed fields or semantics, so accepting it without an
# implementation owner would be a compatibility lie.
SUPPORTED_ANTHROPIC_VERSIONS: Final = frozenset({"2023-06-01"})

# No beta header currently has a complete semantic owner in this bounded
# adapter. Keep the set explicit so adding support is a reviewed protocol
# decision rather than a permissive parser change.
IMPLEMENTED_ANTHROPIC_BETAS: Final = frozenset()

_AUTH_FAILURES: Final[dict[int, tuple[str, str]]] = {
    400: ("invalid_request_error", "authentication request is invalid"),
    401: ("authentication_error", "authentication failed"),
    402: ("billing_error", "authentication billing check failed"),
    403: ("permission_error", "authentication permission denied"),
    404: ("not_found_error", "authentication resource was not found"),
    413: ("request_too_large", "authentication request is too large"),
    429: ("rate_limit_error", "authentication rate limit exceeded"),
    500: ("api_error", "authentication service unavailable"),
    504: ("timeout_error", "authentication service timed out"),
    529: ("overloaded_error", "authentication service overloaded"),
}


class _ProtocolAuthError(AnthropicAPIError):
    """An auth failure with the canonical challenge/backoff headers intact."""

    def __init__(self, error: HTTPException) -> None:
        admitted = _AUTH_FAILURES.get(error.status_code)
        if admitted is None or ERROR_TYPE_STATUS.get(admitted[0]) != error.status_code:
            status_code = 500
            error_type, message = _AUTH_FAILURES[status_code]
        else:
            status_code = error.status_code
            error_type, message = admitted
        super().__init__(
            message,
            error_type=error_type,
            status_code=status_code,
        )
        # Keep an immutable-by-convention projection receipt independent of
        # Exception's mutable compatibility attributes.  The response path
        # below calls this override through ``payload``; a later accidental
        # assignment to ``self.message`` therefore cannot disclose the raw
        # HTTPException detail.
        self._public_projection = (error_type, message, status_code)
        self.headers = {
            key: value
            for key, value in (error.headers or {}).items()
            if key.lower() != REQUEST_ID_HEADER
        }

    def envelope(self, request_id: str | None = None) -> AnthropicErrorEnvelope:
        error_type, message, status_code = self._public_projection
        return AnthropicAPIError(
            message,
            error_type=error_type,
            status_code=status_code,
        ).envelope(request_id)


# Lazy initialization to avoid scanning cache during module import
_models_service: AnthropicModelsService | None = None


def get_models_service() -> AnthropicModelsService:
    """Get or create the anthropic models service singleton."""
    global _models_service
    if _models_service is None:
        _models_service = AnthropicModelsService()
    return _models_service


@router.get("/models", response_model=AnthropicModelList)
@router.get("/v1/models", response_model=AnthropicModelList)
async def list_anthropic_models(
    before_id: str | None = Query(
        default=None,
        title="Before Id",
        description="ID of the object to use as a cursor for pagination. When provided, returns the page of results immediately before this object.",
    ),
    after_id: str | None = Query(
        default=None,
        title="After Id",
        description="ID of the object to use as a cursor for pagination. When provided, returns the page of results immediately after this object.",
    ),
    limit: int = Query(
        default=20,
        ge=1,
        le=1000,
        title="Limit",
        description="Number of items to return per page. Defaults to 20. Ranges from 1 to 1000.",
    ),
    _auth: dict = Depends(verify_auth),
) -> AnthropicModelList:
    """List available models in Anthropic format."""
    return get_models_service().list_models(
        limit=limit, after_id=after_id, before_id=before_id
    )


@router.post("/messages", response_model=MessagesResponse)
@router.post("/v1/messages", response_model=MessagesResponse)
async def create_message(
    http_request: Request,
    api_key: str | None = Security(api_key_header),
    bearer_creds: HTTPAuthorizationCredentials | None = Security(session_scheme),
) -> JSONResponse | StreamingResponse:
    """Create an Anthropic Messages API completion.

    The body is read raw and validated against :class:`MessagesRequest` here
    rather than through a typed parameter, because a validation failure has to
    reach the client as an Anthropic error envelope with a request id — not as
    FastAPI's own 422 shape. ``MessagesRequest`` remains the single source of
    truth for what the body accepts.
    """

    request_id = new_request_id()
    try:
        # Keep the canonical auth implementation as the single authority, but
        # invoke it inside this protocol boundary so its HTTPException cannot
        # escape as FastAPI's generic {"detail": ...} body.
        await _verify_protocol_auth(http_request, api_key, bearer_creds)
        _validate_protocol_headers(http_request)
        request = await _parse_request(http_request)
        # Capability preflight runs exactly once, here: before any model is
        # acquired and — for stream=true — before the StreamingResponse
        # exists, so an unsupported control can never become a 200 carrying
        # an SSE error. The admission receipt it produces is the single
        # classification both transports below then execute against.
        profile = resolve_capability_profile(
            request.model,
            receipt=role_receipt(
                getattr(http_request.app.state, "responses_runtime", None)
            ),
        )
        admission = enforce_capabilities(request, profile)
        # Pre-flight the mapping so a malformed request fails with an HTTP
        # status too. Resolution of the inference owner deliberately stays
        # inside the engine: it is a substitution seam, and an unbound owner
        # is reported as an Anthropic overloaded_error on whichever transport
        # is in use.
        build_turn(request)
    except AnthropicAPIError as error:
        return _error_response(error, request_id)
    except Exception as error:
        logger.error(
            "Anthropic request preflight failed (%s)",
            type(error).__name__,
            exc_info=True,
        )
        return _error_response(
            AnthropicAPIError("request could not be processed"),
            request_id,
        )

    if not request.stream:
        try:
            engine = _create_request_engine(http_request, request.model)
            completion = await _maybe_await(
                engine.generate(request, admission=admission, trace_id=request_id)
            )
        except AnthropicAPIError as error:
            return _error_response(error, request_id)
        except Exception as error:
            logger.error(
                "Anthropic message failed (%s)",
                type(error).__name__,
                exc_info=True,
            )
            return _error_response(
                AnthropicAPIError("message generation failed"),
                request_id,
            )
        return JSONResponse(
            content=completion.model_dump(mode="json"),
            headers={REQUEST_ID_HEADER: request_id},
        )

    async def anthropic_event_generator() -> AsyncIterator[str]:
        try:
            engine = _create_request_engine(http_request, request.model)
            async for event in engine.generate_stream(
                request,
                admission=admission,
                trace_id=request_id,
            ):
                yield _encode_event(event)
        except AnthropicAPIError as error:
            yield _encode_error(error.error_type, error.message, request_id)
        except Exception as error:
            logger.error(
                "Anthropic stream failed (%s)",
                type(error).__name__,
                exc_info=True,
            )
            yield _encode_error("api_error", "message generation failed", request_id)

    return StreamingResponse(
        anthropic_event_generator(),
        media_type="text/event-stream",
        headers={**_SSE_HEADERS, REQUEST_ID_HEADER: request_id},
    )


async def _verify_protocol_auth(
    http_request: Request,
    api_key: str | None,
    bearer_creds: HTTPAuthorizationCredentials | None,
) -> dict[str, Any]:
    """Run canonical auth while retaining Anthropic error ownership."""

    try:
        return await verify_auth(
            http_request,
            api_key=api_key,
            bearer_creds=bearer_creds,
        )
    except HTTPException as error:
        raise _ProtocolAuthError(error) from error
    except Exception as error:
        logger.error(
            "Anthropic authentication authority failed (%s)",
            type(error).__name__,
            exc_info=True,
        )
        raise AnthropicAPIError("authentication service unavailable") from error


def _validate_protocol_headers(http_request: Request) -> None:
    """Require the exact admitted Messages version and reject unowned betas."""

    version_values = http_request.headers.getlist("anthropic-version")
    if len(version_values) != 1:
        raise AnthropicAPIError(
            "exactly one anthropic-version header is required; "
            "supported version: 2023-06-01",
            error_type="invalid_request_error",
        )
    normalized_version = version_values[0].strip()
    if (
        not normalized_version
        or "," in normalized_version
        or normalized_version not in SUPPORTED_ANTHROPIC_VERSIONS
    ):
        raise AnthropicAPIError(
            f"unsupported anthropic-version {normalized_version!r}; "
            "supported version: 2023-06-01",
            error_type="invalid_request_error",
        )

    beta_values = http_request.headers.getlist("anthropic-beta")
    if not beta_values:
        return
    tokens = [token.strip() for value in beta_values for token in value.split(",")]
    if any(not token for token in tokens):
        raise AnthropicAPIError(
            "anthropic-beta contains an empty beta token",
            error_type="invalid_request_error",
        )
    unsupported = sorted(set(tokens).difference(IMPLEMENTED_ANTHROPIC_BETAS))
    if unsupported:
        raise AnthropicAPIError(
            "unsupported or unimplemented anthropic-beta token(s): "
            + ", ".join(unsupported),
            error_type="invalid_request_error",
        )


async def _parse_request(http_request: Request) -> MessagesRequest:
    """Validate the body against the strict schema, Anthropic-style."""

    try:
        payload = await http_request.json()
    except (ValueError, UnicodeDecodeError) as error:
        raise AnthropicAPIError(
            "request body is not valid JSON",
            error_type="invalid_request_error",
        ) from error
    except Exception as error:
        logger.error(
            "Anthropic request body read failed (%s)",
            type(error).__name__,
            exc_info=True,
        )
        raise AnthropicAPIError("request body could not be read") from error
    if not isinstance(payload, dict):
        raise AnthropicAPIError(
            "request body must be a JSON object",
            error_type="invalid_request_error",
        )
    try:
        return MessagesRequest.model_validate(payload)
    except ValidationError as error:
        raise AnthropicAPIError(
            _validation_message(error),
            error_type="invalid_request_error",
        ) from error


def _validation_message(error: ValidationError) -> str:
    parts: list[str] = []
    for item in error.errors():
        location = ".".join(str(entry) for entry in item["loc"]) or "body"
        parts.append(f"{location}: {item['msg']}")
    return "; ".join(parts) or "request failed validation"


def _error_response(error: AnthropicAPIError, request_id: str) -> JSONResponse:
    preserved_headers = {
        key: value
        for key, value in getattr(error, "headers", {}).items()
        if key.lower() != REQUEST_ID_HEADER
    }
    return JSONResponse(
        status_code=error.status_code,
        content=error.payload(request_id),
        headers={**preserved_headers, REQUEST_ID_HEADER: request_id},
    )


def _encode_event(event: AnthropicStreamEvent) -> str:
    payload = json.dumps(event.model_dump(mode="json"), ensure_ascii=False)
    return f"event: {event.type}\ndata: {payload}\n\n"


def _encode_error(error_type: str, message: str, request_id: str) -> str:
    body = attach_request_id(
        error_payload(message, error_type=error_type),
        request_id,
    )
    payload = json.dumps(body, ensure_ascii=False)
    return f"event: error\ndata: {payload}\n\n"


async def _maybe_await(value: Any) -> Any:
    """Await ``value`` when the engine is asynchronous.

    ``_create_anthropic_model`` is an intentional substitution seam; callers
    outside this package replace it with synchronous doubles. Tolerating both
    shapes keeps that seam usable without a second code path.
    """

    if inspect.isawaitable(value):
        return await value
    return value


def _create_anthropic_model(
    model_id: str,
    adapter_path: str | None = None,
    draft_model: str | None = None,
) -> AnthropicMessagesEngine:
    """Build the Anthropic protocol engine for one turn.

    The engine holds no model. It resolves the process-bound typed inference
    owner, so this surface never becomes a second generation owner.
    """

    del model_id, adapter_path, draft_model
    return AnthropicMessagesEngine()


def _create_request_engine(
    http_request: Request,
    model_id: str,
) -> AnthropicMessagesEngine:
    """Bind canonical apps to their own receipt, never a process-global owner."""

    runtime = getattr(http_request.app.state, "responses_runtime", None)
    if runtime is None:
        return _create_anthropic_model(model_id)
    receipt = getattr(runtime, "responses", runtime)
    source = getattr(receipt, "anthropic_turn_source", None)
    if source is None:
        raise AnthropicAPIError(
            "the canonical runtime has no Anthropic turn source",
            error_type="overloaded_error",
        )
    return AnthropicMessagesEngine(turn_source=source)
