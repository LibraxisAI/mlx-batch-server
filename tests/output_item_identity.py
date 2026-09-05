"""One shared start/done identity invariant for every output-item producer.

Both the fused and the legacy adapter suites consume this module, so the two
paths are held to the same contract rather than to two similar-looking ones.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from mlx_batch_server.runtime.events import (
    OutputItemCompleted,
    OutputItemStarted,
)


def assert_output_item_identity_contract(events: Iterable[Any]) -> None:
    """Every tool item opens with its identity; nothing else may carry one."""

    started: dict[int, OutputItemStarted] = {}
    completed: dict[int, OutputItemCompleted] = {}
    for event in events:
        if isinstance(event, OutputItemStarted):
            assert event.index not in started, "output item started twice"
            started[event.index] = event
        elif isinstance(event, OutputItemCompleted):
            completed[event.index] = event

    assert started, "no output item was started"
    for index, start in started.items():
        if start.kind == "function_call":
            assert isinstance(start.call_id, str) and start.call_id.strip()
            assert isinstance(start.name, str) and start.name.strip()
        else:
            assert start.call_id is None
            assert start.name is None

        done = completed.get(index)
        if done is None:
            continue
        assert done.kind == start.kind, "output item kind changed before completion"
        assert done.item_id == start.item_id, "output item id changed"
        if start.kind == "function_call":
            assert done.call_id == start.call_id, "tool call_id changed after start"
            assert done.name == start.name, "tool name changed after start"
