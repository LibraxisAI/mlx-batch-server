"""Authentication and authorization surface for mlx-batch-server.

All modules here are *opt-in*. With the default `SECURITY_LEVEL=0` the server
runs without authentication, and chuk-sessions / redis are not imported until
`SESSION_AUTH_ENABLED=true` or `RATE_LIMIT_ENABLED=true`.
"""

from .dependency import (
    build_open_auth_owner,
    is_auth_required,
    verify_api_key,
    verify_auth,
)

__all__ = [
    "build_open_auth_owner",
    "is_auth_required",
    "verify_api_key",
    "verify_auth",
]
