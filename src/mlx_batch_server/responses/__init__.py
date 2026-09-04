"""OpenAI Responses protocol package.

Legacy exports stay available lazily. Importing a canonical runtime submodule
must not initialize the legacy adapter, store, or router as a side effect.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_CONTEXT_EXPORTS = {
    "BuiltContext",
    "build_context_from_previous_response",
    "build_context_from_response_chain",
}
_SCHEMA_EXPORTS = {"ResponseRequest", "ResponseResponse"}
_STORE_EXPORTS = {
    "StoredResponse",
    "delete_response",
    "get_response",
    "store_response",
}
_LAZY_EXPORTS = _CONTEXT_EXPORTS | _SCHEMA_EXPORTS | _STORE_EXPORTS | {"router"}


def __getattr__(name: str) -> Any:
    if name in _CONTEXT_EXPORTS:
        module = import_module(".context_builder", __name__)
    elif name in _SCHEMA_EXPORTS:
        module = import_module(".schema", __name__)
    elif name in _STORE_EXPORTS:
        module = import_module(".store", __name__)
    elif name == "router":
        module = import_module(".router", __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_EXPORTS})


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
