"""The narrow seam between the Anthropic protocol surface and inference.

The Anthropic package projects protocol shapes; it does not own a generator,
a scheduler or a model. It consumes typed runtime events from whichever
component the process composes as the inference owner.

That owner is injected here. Until an integrator binds one, the protocol
surface fails closed with a documented Anthropic error rather than silently
falling back to a second, divergent generation path.
"""

from __future__ import annotations

import threading
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from mlx_batch_server.runtime.events import TurnEvent

from .errors import InferenceOwnerUnavailableError


@dataclass(frozen=True, slots=True)
class AnthropicTurn:
    """One protocol-neutral generation turn requested by an Anthropic client.

    Field names mirror ``mlx_batch_server.runtime.contracts.GenerationRequest``
    so the integrator's binding is a translation, not a reinterpretation.
    """

    model_alias: str
    messages: Sequence[Mapping[str, Any]]
    tools: Sequence[Mapping[str, Any]] = ()
    tool_choice: Mapping[str, Any] | None = None
    sampling: Mapping[str, Any] = field(default_factory=dict)
    reasoning: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class AnthropicTurnSource(Protocol):
    """A typed inference owner able to run one Anthropic turn."""

    def stream(self, turn: AnthropicTurn) -> AsyncIterator[TurnEvent]:
        """Yield the runtime events for ``turn`` in emission order."""
        ...


_lock = threading.Lock()
_turn_source: AnthropicTurnSource | None = None

_UNBOUND_DETAIL = (
    "The Anthropic Messages surface has no typed inference owner bound in "
    "this process. Register one with "
    "mlx_batch_server.chat.anthropic.turn_source.register_turn_source()."
)


def register_turn_source(source: AnthropicTurnSource) -> None:
    """Bind the process-wide typed inference owner for Anthropic turns."""

    if not isinstance(source, AnthropicTurnSource):
        raise TypeError("turn source must implement AnthropicTurnSource")
    global _turn_source
    with _lock:
        _turn_source = source


def clear_turn_source() -> None:
    """Unbind the inference owner. Used by tests and by shutdown paths."""

    global _turn_source
    with _lock:
        _turn_source = None


def current_turn_source() -> AnthropicTurnSource | None:
    """The bound inference owner, or ``None`` when the surface is unbound."""

    with _lock:
        return _turn_source


def require_turn_source() -> AnthropicTurnSource:
    """The bound inference owner, or a closed Anthropic failure."""

    source = current_turn_source()
    if source is None:
        raise InferenceOwnerUnavailableError(_UNBOUND_DETAIL)
    return source


__all__ = [
    "AnthropicTurn",
    "AnthropicTurnSource",
    "clear_turn_source",
    "current_turn_source",
    "register_turn_source",
    "require_turn_source",
]
