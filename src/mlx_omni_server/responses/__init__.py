"""
OpenAI Responses API implementation for MLX Omni Server.

Provides /v1/responses endpoint compatible with OpenAI's Responses API,
with intelligent routing to local models and external providers.

Features:
- Response storage and retrieval
- Multi-turn conversations via previous_response_id
- Context chain walking for conversation history
- In-memory storage with optional Redis backend

Created by M&K (c)2026 The LibraxisAI Team
"""

from .context_builder import (
    BuiltContext,
    build_context_from_previous_response,
    build_context_from_response_chain,
)
from .router import router
from .schema import ResponseRequest, ResponseResponse
from .store import (
    StoredResponse,
    delete_response,
    get_response,
    store_response,
)

__all__ = [
    "BuiltContext",
    "ResponseRequest",
    "ResponseResponse",
    "StoredResponse",
    "build_context_from_previous_response",
    "build_context_from_response_chain",
    "delete_response",
    "get_response",
    "router",
    "store_response",
]
