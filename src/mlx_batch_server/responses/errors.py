"""Canonical HTTP and SSE errors for the OpenAI Responses surface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

from fastapi.responses import JSONResponse

_INTERNAL_ERROR_MESSAGE: Final = "Internal server error."


@dataclass(frozen=True, slots=True)
class OpenAIError:
    """One immutable failure value shared by Responses transports."""

    message: str
    type: str
    code: str
    param: str | None
    status_code: int
    sequence_number: int | None = None

    def __post_init__(self) -> None:
        for field_name in ("message", "type", "code"):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string")
            if not value.strip():
                raise ValueError(f"{field_name} must not be empty")
        if self.param is not None:
            if not isinstance(self.param, str):
                raise TypeError("param must be a string or None")
            if not self.param.strip():
                raise ValueError("param must not be empty")
        if type(self.status_code) is not int:
            raise TypeError("status_code must be an integer")
        if not 400 <= self.status_code <= 599:
            raise ValueError("status_code must be between 400 and 599")
        if self.sequence_number is not None:
            if type(self.sequence_number) is not int:
                raise TypeError("sequence_number must be an integer or None")
            if self.sequence_number < 0:
                raise ValueError("sequence_number must not be negative")

    @classmethod
    def from_exception(
        cls,
        exception: BaseException,
        *,
        status_code: int = 500,
        code: str = "internal_error",
        sequence_number: int | None = None,
    ) -> OpenAIError:
        """Classify an internal fault without exposing exception text."""

        if not isinstance(exception, BaseException):
            raise TypeError("exception must be an exception")
        if type(status_code) is not int:
            raise TypeError("status_code must be an integer")
        if not 500 <= status_code <= 599:
            raise ValueError("internal exception status_code must be a 5xx status")
        return cls(
            message=_INTERNAL_ERROR_MESSAGE,
            type="server_error",
            code=code,
            param=None,
            status_code=status_code,
            sequence_number=sequence_number,
        )


def render_http_error(error: OpenAIError) -> JSONResponse:
    """Render the exact nested OpenAI HTTP error envelope."""

    if not isinstance(error, OpenAIError):
        raise TypeError("error must be an OpenAIError")
    return JSONResponse(
        status_code=error.status_code,
        content={
            "error": {
                "message": error.message,
                "type": error.type,
                "param": error.param,
                "code": error.code,
            }
        },
    )


def render_sse_error(error: OpenAIError) -> dict[str, Any]:
    """Render one flat official Responses ``error`` streaming event."""

    if not isinstance(error, OpenAIError):
        raise TypeError("error must be an OpenAIError")
    if error.sequence_number is None:
        raise ValueError("sequence_number is required for an SSE error event")
    return {
        "type": "error",
        "code": error.code,
        "message": error.message,
        "param": error.param,
        "sequence_number": error.sequence_number,
    }


__all__ = ["OpenAIError", "render_http_error", "render_sse_error"]
