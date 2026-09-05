"""RED contracts for delivery-ordered, protocol-neutral hosted evidence."""

from __future__ import annotations

import dataclasses
from types import MappingProxyType

import pytest

from mlx_batch_server.runtime.hosted_evidence import (
    HostedCallEvidence,
    HostedEvidenceRegistry,
    HostedRoundEvidence,
)


def _registry(*, capacity: int = 4) -> HostedEvidenceRegistry:
    return HostedEvidenceRegistry(capacity=capacity, ttl_s=900.0)


def _begin(registry: HostedEvidenceRegistry, suffix: str = "1") -> str:
    response_id = f"resp_{suffix}"
    registry.begin(
        response_id=response_id,
        trace_id=f"trace_{suffix}",
        protocol="openai_responses",
        profile="provider-absent",
    )
    registry.record_consumer_event(response_id, "waiting")
    return response_id


def _call(*, status: str = "failed") -> HostedCallEvidence:
    return HostedCallEvidence(  # type: ignore[arg-type]
        call_id="call_1",
        tool_name="web_search",
        action_kind="search",
        status=status,
        error_code="provider_unavailable" if status == "failed" else None,
        requested_url=None,
        final_url=None,
        result_digest=None,
        mime=None,
        started_monotonic_ns=10,
        ended_monotonic_ns=20,
        delivery_state="prepared",
        first_event_sequence=3,
        last_event_sequence=4,
    )


def test_prepared_call_is_invisible_until_all_wire_events_are_delivered() -> None:
    registry = _registry()
    response_id = _begin(registry)
    token = registry.prepare_call(response_id, _call())

    before = registry.by_response(response_id)
    assert before is not None and before.calls == ()

    registry.mark_call_delivered(response_id, token)
    registry.mark_call_delivered(response_id, token)
    after = registry.by_response(response_id)
    assert after is not None
    assert len(after.calls) == 1
    assert after.calls[0].delivery_state == "delivered"
    assert [event.state for event in after.consumer_events] == [
        "queued",
        "waiting",
        "tool_failed_but_continuing",
    ]


def test_delivery_failure_never_satisfies_delivered_acceptance() -> None:
    registry = _registry()
    response_id = _begin(registry)
    token = registry.prepare_call(response_id, _call(status="completed"))

    registry.mark_call_delivery_failed(response_id, token)
    registry.mark_call_delivery_failed(response_id, token)
    snapshot = registry.by_trace("trace_1")
    assert snapshot is not None
    assert snapshot.calls[0].delivery_state == "delivery_failed"
    assert all(
        event.state != "tool_failed_but_continuing"
        for event in snapshot.consumer_events
    )


def test_rounds_are_contiguous_and_usage_is_frozen() -> None:
    registry = _registry()
    response_id = _begin(registry)
    registry.record_round(
        response_id,
        HostedRoundEvidence(
            round_index=0,
            started_monotonic_ns=1,
            ended_monotonic_ns=2,
            terminal_kind="completed",
            usage=MappingProxyType(
                {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5}
            ),
            call_ids=("call_1",),
        ),
    )
    with pytest.raises(ValueError, match="contiguous"):
        registry.record_round(
            response_id,
            HostedRoundEvidence(2, 3, 4, "completed", None, ()),
        )
    snapshot = registry.by_response(response_id)
    assert snapshot is not None and snapshot.rounds[0].usage is not None
    with pytest.raises(TypeError):
        snapshot.rounds[0].usage["total_tokens"] = 9  # type: ignore[index]


def test_terminal_is_prepared_before_wire_and_mark_is_idempotent() -> None:
    registry = _registry()
    response_id = _begin(registry)
    token = registry.prepare_terminal(
        response_id,
        state="completed",
        terminal_usage={"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
    )
    before = registry.by_response(response_id)
    assert before is not None and before.state == "in_progress"

    registry.mark_terminal(response_id, token, delivered=True)
    registry.mark_terminal(response_id, token, delivered=True)
    after = registry.by_response(response_id)
    assert after is not None
    assert after.state == "completed"
    assert after.terminal_delivery_state == "delivered"
    assert after.consumer_events[-1].state == "terminal"
    with pytest.raises(dataclasses.FrozenInstanceError):
        after.state = "failed"  # type: ignore[misc]


def test_duplicate_identities_fail_and_capacity_evicts_only_sealed_records() -> None:
    registry = _registry(capacity=1)
    first = _begin(registry)
    with pytest.raises(ValueError, match="unique"):
        registry.begin(
            response_id=first,
            trace_id="another",
            protocol="openai_responses",
            profile="provider-absent",
        )
    with pytest.raises(RuntimeError, match="capacity"):
        _begin(registry, "2")

    token = registry.prepare_terminal(first, state="failed")
    registry.mark_terminal(first, token, delivered=True)
    second = _begin(registry, "2")
    assert registry.by_response(first) is None
    assert registry.by_response(second) is not None
