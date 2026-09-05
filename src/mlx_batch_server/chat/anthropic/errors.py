"""Typed Anthropic error envelope for the Messages protocol surface.

Anthropic clients read failures from one closed shape: an ``error`` object with
a documented ``type`` plus a top-level ``request_id``. This module owns that
projection so no handler hand-rolls a dict, and so unsupported hosted
capabilities can fail closed with an envelope the official SDK understands.
"""

from __future__ import annotations

import uuid
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict

REQUEST_ID_HEADER: Final = "request-id"

#: Top-level key that carries the request id on every Anthropic failure body,
#: including error frames delivered over SSE.
REQUEST_ID_FIELD: Final = "request_id"

#: Closed set of documented Anthropic error types. The server never invents a
#: new member: an unmapped internal failure projects onto ``api_error``.
ERROR_TYPE_STATUS: Final[dict[str, int]] = {
    "invalid_request_error": 400,
    "authentication_error": 401,
    "billing_error": 402,
    "permission_error": 403,
    "not_found_error": 404,
    "request_too_large": 413,
    "rate_limit_error": 429,
    "timeout_error": 504,
    "api_error": 500,
    "overloaded_error": 529,
}


def new_request_id() -> str:
    """Mint one request identifier in Anthropic's ``req_`` shape."""

    return f"req_{uuid.uuid4().hex[:24]}"


class AnthropicErrorBody(BaseModel):
    """The inner ``error`` object of an Anthropic failure."""

    model_config = ConfigDict(extra="forbid")

    type: str
    message: str


class AnthropicErrorEnvelope(BaseModel):
    """Canonical Anthropic failure payload, body and SSE alike."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["error"] = "error"
    error: AnthropicErrorBody
    request_id: str | None = None


class AnthropicAPIError(Exception):
    """One failure that already knows its Anthropic type and HTTP status."""

    def __init__(
        self,
        message: str,
        *,
        error_type: str = "api_error",
        status_code: int | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        if error_type not in ERROR_TYPE_STATUS:
            # Fail closed onto the documented generic type rather than leaking
            # an undocumented discriminator to the client.
            error_type = "api_error"
        self.message = message
        self.error_type = error_type
        self.status_code = status_code or ERROR_TYPE_STATUS[error_type]
        self.request_id = request_id

    def envelope(self, request_id: str | None = None) -> AnthropicErrorEnvelope:
        return AnthropicErrorEnvelope(
            error=AnthropicErrorBody(type=self.error_type, message=self.message),
            request_id=request_id or self.request_id,
        )

    def payload(self, request_id: str | None = None) -> dict[str, Any]:
        return self.envelope(request_id).model_dump(mode="json")


class UnsupportedCapabilityError(AnthropicAPIError):
    """A hosted Anthropic capability this runtime does not implement.

    Raised instead of silently dropping the field, so a client never believes
    a capability was honoured when it was ignored.
    """

    def __init__(self, capability: str, detail: str | None = None) -> None:
        suffix = f" {detail}" if detail else ""
        super().__init__(
            f"{capability} is not supported by this runtime.{suffix}",
            error_type="invalid_request_error",
        )
        self.capability = capability


class InferenceOwnerUnavailableError(AnthropicAPIError):
    """No typed inference owner is bound to the Anthropic protocol surface.

    Deliberately not named ``RuntimeUnavailableError``: that name already
    belongs to ``runtime.manager`` and means a model runtime could not be
    acquired. This one means the protocol surface has nothing to talk to.
    """

    def __init__(self, detail: str) -> None:
        super().__init__(detail, error_type="overloaded_error")


def error_payload(
    message: str,
    *,
    error_type: str = "api_error",
    request_id: str | None = None,
) -> dict[str, Any]:
    """Build one Anthropic error body without raising."""

    return AnthropicAPIError(
        message,
        error_type=error_type,
        request_id=request_id,
    ).payload()


def attach_request_id(
    payload: dict[str, Any], request_id: str | None
) -> dict[str, Any]:
    """Stamp the request id onto an already-built Anthropic failure body.

    Clients correlate a failure with server-side state through this id, so it
    travels on SSE error frames as well as on HTTP error bodies.
    """

    if request_id:
        payload[REQUEST_ID_FIELD] = request_id
    return payload


__all__ = [
    "ERROR_TYPE_STATUS",
    "REQUEST_ID_FIELD",
    "REQUEST_ID_HEADER",
    "AnthropicAPIError",
    "AnthropicErrorBody",
    "AnthropicErrorEnvelope",
    "InferenceOwnerUnavailableError",
    "UnsupportedCapabilityError",
    "attach_request_id",
    "error_payload",
    "new_request_id",
]
