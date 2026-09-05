"""RED contracts for the target-owned response registry.

These tests are authored under Compile Embargo and must not be executed until
the integrator releases HOLD.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping

import pytest

from mlx_batch_server.responses.registry import (
    CancelDeliveryRejected,
    ResponseRegistry,
    ResponseRegistryError,
)

OWNER_A = "principal:a"
OWNER_B = "principal:b"


def _envelope(response_id: str, *, status: str = "completed") -> dict:
    return {
        "id": response_id,
        "object": "response",
        "status": status,
        "output": [{"type": "message", "content": "ok"}],
    }


def _commit(registry: ResponseRegistry, response_id: str, owner_id: str) -> None:
    registry.commit(
        response_id,
        _envelope(response_id),
        owner_id=owner_id,
        materialized_messages=[
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "ok"},
        ],
    )


def _without_ids(items: list[dict]) -> list[dict]:
    return [
        {key: value for key, value in item.items() if key != "id"} for item in items
    ]


def _live_snapshot(response_id: str, status: str) -> dict:
    return {
        "id": response_id,
        "object": "response",
        "status": status,
        "background": True,
        "model": "flash-next",
        "output": [],
        "metadata": {"phase": status},
    }


def test_registry_isolates_every_response_operation_by_owner() -> None:
    registry = ResponseRegistry()
    registry.begin(
        "resp_owned",
        owner_id=OWNER_A,
        store=True,
        materialized_messages=[{"role": "user", "content": "hello"}],
    )
    _commit(registry, "resp_owned", OWNER_A)

    for operation in (
        lambda: registry.get("resp_owned", owner_id=OWNER_B),
        lambda: registry.parent_messages("resp_owned", owner_id=OWNER_B),
        lambda: registry.request_cancel(
            "resp_owned", "foreign_cancel", owner_id=OWNER_B
        ),
        lambda: registry.delete("resp_owned", owner_id=OWNER_B),
        lambda: registry.wait_terminal("resp_owned", 0, owner_id=OWNER_B),
    ):
        with pytest.raises(ResponseRegistryError) as denied:
            operation()
        assert denied.value.code == "response_not_found"

    assert (
        registry.bind_cancel("resp_owned", lambda _reason: None, owner_id=OWNER_B)
        is False
    )
    assert registry.get("resp_owned", owner_id=OWNER_A)["status"] == "completed"


def test_parent_lineage_adds_assistant_output_without_changing_input_items() -> None:
    registry = ResponseRegistry()
    inputs = [{"role": "user", "content": "Reply with ROOT_OK"}]
    registry.begin(
        "resp_lineage",
        owner_id=OWNER_A,
        store=True,
        materialized_messages=inputs,
    )
    registry.commit(
        "resp_lineage",
        {
            "id": "resp_lineage",
            "object": "response",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "ROOT_OK"}],
                }
            ],
        },
        owner_id=OWNER_A,
        materialized_messages=inputs,
    )

    stored_inputs = registry.input_messages("resp_lineage", owner_id=OWNER_A)
    assert _without_ids(stored_inputs) == inputs
    assert registry.parent_messages("resp_lineage", owner_id=OWNER_A) == [
        *inputs,
        {
            "role": "assistant",
            "content": [{"type": "input_text", "text": "ROOT_OK"}],
        },
    ]


def test_cancel_before_bind_delivers_first_reason_once_and_waits_terminal() -> None:
    registry = ResponseRegistry()
    delivered: list[str] = []
    registry.begin(
        "resp_cancel",
        owner_id=OWNER_A,
        store=True,
        materialized_messages=[],
    )

    first = registry.request_cancel(
        "resp_cancel", "transport_disconnect", owner_id=OWNER_A
    )
    second = registry.request_cancel(
        "resp_cancel", "later_client_reason", owner_id=OWNER_A
    )
    assert first is second
    assert (
        registry.bind_cancel("resp_cancel", delivered.append, owner_id=OWNER_A) is True
    )
    assert delivered == ["transport_disconnect"]

    terminal = {
        **_envelope("resp_cancel", status="cancelled"),
        "error": {
            "code": "request_cancelled",
            "message": "transport_disconnect",
        },
    }
    registry.commit(
        "resp_cancel",
        terminal,
        owner_id=OWNER_A,
        materialized_messages=[],
    )
    assert registry.wait_terminal(first, 0, owner_id=OWNER_A) == terminal
    assert (
        registry.request_cancel("resp_cancel", "third_reason", owner_id=OWNER_A)
        is not first
    )
    assert delivered == ["transport_disconnect"]
    assert registry.stats()["cancel_requests_total"] == 1
    assert registry.stats()["cancel_settled_total"] == 1


def test_cancel_delivery_retries_failures_and_rejections_with_first_reason() -> None:
    registry = ResponseRegistry()
    attempts: list[str] = []
    outcomes: list[object] = [RuntimeError("backend busy"), False, True]

    def cancel(reason: str) -> bool:
        attempts.append(reason)
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        assert isinstance(outcome, bool)
        return outcome

    registry.begin(
        "resp_retry",
        owner_id=OWNER_A,
        store=True,
        materialized_messages=[],
        cancel=cancel,
    )

    with pytest.raises(RuntimeError, match="backend busy"):
        registry.request_cancel("resp_retry", "first reason", owner_id=OWNER_A)
    with pytest.raises(CancelDeliveryRejected):
        registry.request_cancel("resp_retry", "second reason", owner_id=OWNER_A)

    waiter = registry.request_cancel(
        "resp_retry",
        "third reason",
        owner_id=OWNER_A,
    )
    registry.request_cancel("resp_retry", "fourth reason", owner_id=OWNER_A)

    assert attempts == ["first reason", "first reason", "first reason"]
    assert waiter.cancel_delivered is True
    assert registry.stats()["cancel_requests_total"] == 1
    assert registry.stats()["cancel_delivery_failures_total"] == 2


def test_bind_cancel_failure_can_retry_the_pending_first_reason() -> None:
    registry = ResponseRegistry()
    attempts: list[str] = []

    def cancel(reason: str) -> None:
        attempts.append(reason)
        if len(attempts) == 1:
            raise RuntimeError("worker not ready")

    registry.begin(
        "resp_bind_retry",
        owner_id=OWNER_A,
        store=True,
        materialized_messages=[],
    )
    registry.request_cancel(
        "resp_bind_retry",
        "first reason",
        owner_id=OWNER_A,
    )

    with pytest.raises(RuntimeError, match="worker not ready"):
        registry.bind_cancel(
            "resp_bind_retry",
            cancel,
            owner_id=OWNER_A,
        )
    assert (
        registry.bind_cancel(
            "resp_bind_retry",
            cancel,
            owner_id=OWNER_A,
        )
        is True
    )
    assert attempts == ["first reason", "first reason"]


def test_concurrent_cancel_callers_share_one_delivery_attempt() -> None:
    entered = threading.Event()
    release = threading.Event()
    attempts: list[str] = []
    failures: list[BaseException] = []

    def cancel(reason: str) -> bool:
        attempts.append(reason)
        entered.set()
        assert release.wait(timeout=1)
        return True

    registry = ResponseRegistry()
    registry.begin(
        "resp_concurrent_cancel",
        owner_id=OWNER_A,
        store=True,
        materialized_messages=[],
        cancel=cancel,
    )

    def request(reason: str) -> None:
        try:
            registry.request_cancel(
                "resp_concurrent_cancel",
                reason,
                owner_id=OWNER_A,
            )
        except BaseException as exc:
            failures.append(exc)

    first = threading.Thread(target=request, args=("first reason",))
    second = threading.Thread(target=request, args=("second reason",))
    first.start()
    assert entered.wait(timeout=1)
    second.start()
    second.join(timeout=1)

    assert not second.is_alive()
    assert attempts == ["first reason"]
    release.set()
    first.join(timeout=1)

    assert not first.is_alive()
    assert failures == []
    registry.request_cancel(
        "resp_concurrent_cancel",
        "third reason",
        owner_id=OWNER_A,
    )
    assert attempts == ["first reason"]


def test_stale_cancel_ack_cannot_settle_a_reused_response_id() -> None:
    now = [100.0]
    entered = threading.Event()
    release = threading.Event()
    old_attempts: list[str] = []
    new_attempts: list[str] = []

    def old_cancel(reason: str) -> bool:
        old_attempts.append(reason)
        entered.set()
        assert release.wait(timeout=1)
        return True

    registry = ResponseRegistry(
        in_flight_ttl_s=1,
        idle_ttl_s=60,
        max_tombstones=1,
        clock=lambda: now[0],
    )
    registry.begin(
        "resp_reused",
        owner_id=OWNER_A,
        store=True,
        materialized_messages=[],
        cancel=old_cancel,
    )

    old_thread = threading.Thread(
        target=lambda: registry.request_cancel(
            "resp_reused",
            "old reason",
            owner_id=OWNER_A,
        )
    )
    old_thread.start()
    assert entered.wait(timeout=1)

    now[0] += 2
    assert registry.stats()["timed_out_total"] == 1
    registry.begin(
        "resp_pressure",
        owner_id=OWNER_A,
        store=True,
        materialized_messages=[],
    )
    registry.delete("resp_pressure", owner_id=OWNER_A)
    registry.begin(
        "resp_reused",
        owner_id=OWNER_A,
        store=True,
        materialized_messages=[],
    )
    registry.delete("resp_reused", owner_id=OWNER_A)

    release.set()
    old_thread.join(timeout=1)
    assert not old_thread.is_alive()
    assert old_attempts == ["old reason"]

    assert (
        registry.bind_cancel(
            "resp_reused",
            new_attempts.append,
            owner_id=OWNER_A,
        )
        is True
    )
    assert new_attempts == ["response_deleted"]


def test_delete_before_bind_cannot_resurrect_and_replays_delete_reason() -> None:
    registry = ResponseRegistry()
    delivered: list[str] = []
    registry.begin(
        "resp_deleted",
        owner_id=OWNER_A,
        store=True,
        materialized_messages=[],
    )

    assert registry.delete("resp_deleted", owner_id=OWNER_A)["deleted"] is True
    assert (
        registry.bind_cancel("resp_deleted", delivered.append, owner_id=OWNER_A) is True
    )
    assert delivered == ["response_deleted"]
    _commit(registry, "resp_deleted", OWNER_A)

    with pytest.raises(ResponseRegistryError) as deleted:
        registry.get("resp_deleted", owner_id=OWNER_A)
    assert deleted.value.code == "response_deleted"
    assert registry.delete("resp_deleted", owner_id=OWNER_A)["deleted"] is True


def test_registry_bounds_capacity_ttl_tombstones_and_json_state() -> None:
    now = [100.0]
    registry = ResponseRegistry(
        max_entries=1,
        max_in_flight=1,
        max_bytes=1024,
        max_entry_bytes=512,
        idle_ttl_s=5,
        in_flight_ttl_s=5,
        max_tombstones=2,
        clock=lambda: now[0],
    )
    registry.begin(
        "resp_live",
        owner_id=OWNER_A,
        store=True,
        materialized_messages=[],
    )
    with pytest.raises(ResponseRegistryError) as full:
        registry.begin(
            "resp_second",
            owner_id=OWNER_A,
            store=True,
            materialized_messages=[],
        )
    assert full.value.code == "response_store_capacity"

    now[0] += 6
    assert registry.stats()["timed_out_total"] == 1
    with pytest.raises(ResponseRegistryError) as timed_out:
        registry.get("resp_live", owner_id=OWNER_A)
    assert timed_out.value.code == "response_timeout"

    with pytest.raises(ResponseRegistryError) as invalid_json:
        registry.begin(
            "resp_python_object",
            owner_id=OWNER_A,
            store=True,
            materialized_messages=[{"not_json": object()}],
        )
    assert invalid_json.value.code == "response_invalid_state"
    json.dumps(registry.stats(), allow_nan=False)
    assert registry.stats()["tombstones"] <= 2


def test_registry_bounds_aggregate_in_flight_bytes() -> None:
    registry = ResponseRegistry(
        max_entries=4,
        max_in_flight=4,
        max_bytes=256,
        max_entry_bytes=256,
    )
    messages = [{"content": "x" * 20}]

    registry.begin(
        "resp_first",
        owner_id=OWNER_A,
        store=True,
        materialized_messages=messages,
    )
    with pytest.raises(ResponseRegistryError) as full:
        registry.begin(
            "resp_second",
            owner_id=OWNER_A,
            store=True,
            materialized_messages=messages,
        )

    assert full.value.code == "response_store_capacity"
    assert registry.stats()["in_flight"] == 1


def test_tombstone_accepts_only_the_first_terminal_envelope() -> None:
    registry = ResponseRegistry()
    registry.begin(
        "resp_terminal",
        owner_id=OWNER_A,
        store=False,
        materialized_messages=[],
    )
    first = _envelope("resp_terminal")

    assert (
        registry.commit(
            "resp_terminal",
            first,
            owner_id=OWNER_A,
            materialized_messages=[],
        )
        == first
    )
    assert (
        registry.commit(
            "resp_terminal",
            dict(first),
            owner_id=OWNER_A,
            materialized_messages=[],
        )
        == first
    )

    conflicting = {**first, "status": "failed"}
    with pytest.raises(ResponseRegistryError) as conflict:
        registry.commit(
            "resp_terminal",
            conflicting,
            owner_id=OWNER_A,
            materialized_messages=[],
        )
    assert conflict.value.code == "response_terminal_conflict"


def test_cancel_bind_commit_race_delivers_at_most_once() -> None:
    registry = ResponseRegistry()
    registry.begin(
        "resp_race",
        owner_id=OWNER_A,
        store=True,
        materialized_messages=[],
    )
    barrier = threading.Barrier(3)
    delivered: list[str] = []
    failures: list[ResponseRegistryError] = []

    def bind() -> None:
        barrier.wait()
        registry.bind_cancel("resp_race", delivered.append, owner_id=OWNER_A)

    def cancel() -> None:
        barrier.wait()
        try:
            registry.request_cancel("resp_race", "race_cancel", owner_id=OWNER_A)
        except ResponseRegistryError as exc:
            failures.append(exc)

    bind_thread = threading.Thread(target=bind)
    cancel_thread = threading.Thread(target=cancel)
    bind_thread.start()
    cancel_thread.start()
    barrier.wait()
    _commit(registry, "resp_race", OWNER_A)
    bind_thread.join(timeout=1)
    cancel_thread.join(timeout=1)

    assert not bind_thread.is_alive()
    assert not cancel_thread.is_alive()
    assert len(delivered) <= 1
    assert all(exc.code == "response_not_cancellable" for exc in failures)
    assert registry.stats()["in_flight"] == 0


def test_shutdown_request_runs_cancel_on_caller_and_wait_is_separate() -> None:
    registry = ResponseRegistry()
    response_id = registry.allocate_id()
    cancel_threads: list[int] = []

    registry.begin(
        response_id,
        owner_id=OWNER_A,
        store=True,
        materialized_messages=[],
        cancel=lambda _reason: cancel_threads.append(threading.get_ident()),
    )
    caller_thread = threading.get_ident()
    waiters = registry.request_shutdown()

    assert cancel_threads == [caller_thread]
    assert registry.wait_for_shutdown(waiters, 0.0) is False
    with pytest.raises(ResponseRegistryError) as closing:
        registry.allocate_id()
    assert closing.value.code == "response_registry_shutting_down"

    registry.commit(
        response_id,
        _envelope(response_id),
        owner_id=OWNER_A,
        materialized_messages=[],
    )
    assert registry.wait_for_shutdown(waiters, 0.0) is True


def test_parent_lineage_materializes_the_complete_output_item_sequence() -> None:
    """A stored function_call must reach round two, not only assistant text."""

    registry = ResponseRegistry()
    inputs = [{"role": "user", "content": "Jaka pogoda w Kielcach?"}]
    registry.begin(
        "resp_tool",
        owner_id=OWNER_A,
        store=True,
        materialized_messages=inputs,
    )
    registry.commit(
        "resp_tool",
        {
            "id": "resp_tool",
            "object": "response",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Sprawdzam."}],
                },
                {
                    "id": "fc_1",
                    "type": "function_call",
                    "status": "completed",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": '{"city":"Kielce"}',
                },
            ],
        },
        owner_id=OWNER_A,
        materialized_messages=inputs,
    )

    stored_inputs = registry.input_messages("resp_tool", owner_id=OWNER_A)
    assert _without_ids(stored_inputs) == inputs
    assert registry.parent_messages("resp_tool", owner_id=OWNER_A) == [
        *inputs,
        {
            "role": "assistant",
            "content": [{"type": "input_text", "text": "Sprawdzam."}],
        },
        {
            "type": "function_call",
            "role": "assistant",
            "call_id": "call_1",
            "name": "get_weather",
            "arguments": '{"city":"Kielce"}',
            "id": "fc_1",
            "status": "completed",
        },
    ]


def test_unusable_stored_function_call_cannot_anchor_a_continuation() -> None:
    """An identity-less call must fail closed, not hand the model an orphan."""

    registry = ResponseRegistry()
    inputs = [{"role": "user", "content": "hello"}]
    registry.begin(
        "resp_partial",
        owner_id=OWNER_A,
        store=True,
        materialized_messages=inputs,
    )
    registry.commit(
        "resp_partial",
        {
            "id": "resp_partial",
            "object": "response",
            "status": "incomplete",
            "output": [
                {"type": "function_call", "status": "incomplete", "arguments": "{"}
            ],
        },
        owner_id=OWNER_A,
        materialized_messages=inputs,
    )

    with pytest.raises(ResponseRegistryError) as error:
        registry.parent_messages("resp_partial", owner_id=OWNER_A)
    assert error.value.code == "response_invalid_lineage"
    assert error.value.status_code == 409
    assert error.value.param == "previous_response_id"
    # The stored response itself stays readable and owner-isolated.
    assert registry.get("resp_partial", owner_id=OWNER_A)["status"] == "incomplete"
    assert (
        _without_ids(registry.input_messages("resp_partial", owner_id=OWNER_A))
        == inputs
    )


def test_store_false_lineage_is_not_retained_for_tool_continuations() -> None:
    registry = ResponseRegistry()
    registry.begin(
        "resp_nostore",
        owner_id=OWNER_A,
        store=False,
        materialized_messages=[{"role": "user", "content": "hi"}],
    )
    registry.commit(
        "resp_nostore",
        {
            "id": "resp_nostore",
            "object": "response",
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "status": "completed",
                    "call_id": "call_1",
                    "name": "t",
                    "arguments": "{}",
                }
            ],
        },
        owner_id=OWNER_A,
        materialized_messages=[{"role": "user", "content": "hi"}],
    )

    with pytest.raises(ResponseRegistryError) as error:
        registry.parent_messages("resp_nostore", owner_id=OWNER_A)
    assert error.value.param == "previous_response_id"
    assert error.value.code != "response_invalid_lineage"


def test_begin_update_and_get_expose_deep_immutable_live_snapshots() -> None:
    registry = ResponseRegistry()
    queued = _live_snapshot("resp_background", "queued")

    begun = registry.begin(
        "resp_background",
        owner_id=OWNER_A,
        store=True,
        background=True,
        public_snapshot=queued,
        materialized_messages=[{"role": "user", "content": "hello"}],
    )
    queued["status"] = "failed"
    queued["metadata"]["phase"] = "corrupted"

    assert isinstance(begun, Mapping)
    assert begun["status"] == "queued"
    assert registry.get("resp_background", owner_id=OWNER_A) == begun
    with pytest.raises(TypeError, match="immutable"):
        begun["status"] = "failed"  # type: ignore[index]
    with pytest.raises(TypeError, match="immutable"):
        begun["output"].append({"type": "message"})
    with pytest.raises(TypeError, match="immutable"):
        begun["metadata"]["phase"] = "failed"

    in_progress = _live_snapshot("resp_background", "in_progress")
    in_progress["output"] = [
        {
            "id": "msg_live",
            "type": "message",
            "status": "in_progress",
            "content": [],
        }
    ]
    updated = registry.update(
        "resp_background",
        in_progress,
        owner_id=OWNER_A,
    )
    in_progress["output"][0]["status"] = "failed"

    assert updated["status"] == "in_progress"
    assert updated["output"][0]["status"] == "in_progress"
    assert registry.get("resp_background", owner_id=OWNER_A) == updated

    with pytest.raises(ResponseRegistryError) as foreign:
        registry.get("resp_background", owner_id=OWNER_B)
    assert foreign.value.code == "response_not_found"


def test_live_snapshot_transitions_cannot_replace_terminal_commit_authority() -> None:
    registry = ResponseRegistry()
    registry.begin(
        "resp_transitions",
        owner_id=OWNER_A,
        store=True,
        background=True,
        public_snapshot=_live_snapshot("resp_transitions", "queued"),
        materialized_messages=[],
    )
    registry.update(
        "resp_transitions",
        _live_snapshot("resp_transitions", "in_progress"),
        owner_id=OWNER_A,
    )

    for illegal_status in ("queued", "completed", "failed", "cancelled"):
        with pytest.raises(ResponseRegistryError) as illegal:
            registry.update(
                "resp_transitions",
                _live_snapshot("resp_transitions", illegal_status),
                owner_id=OWNER_A,
            )
        assert illegal.value.param == "status"

    terminal = _envelope("resp_transitions")
    assert (
        registry.commit(
            "resp_transitions",
            terminal,
            owner_id=OWNER_A,
            materialized_messages=[],
        )
        == terminal
    )
    assert (
        registry.commit(
            "resp_transitions",
            dict(terminal),
            owner_id=OWNER_A,
            materialized_messages=[],
        )
        == terminal
    )
    with pytest.raises(ResponseRegistryError) as conflict:
        registry.commit(
            "resp_transitions",
            {**terminal, "status": "failed"},
            owner_id=OWNER_A,
            materialized_messages=[],
        )
    assert conflict.value.code == "response_terminal_conflict"

    with pytest.raises(ResponseRegistryError) as terminal_update:
        registry.update(
            "resp_transitions",
            _live_snapshot("resp_transitions", "in_progress"),
            owner_id=OWNER_A,
        )
    assert terminal_update.value.code == "response_not_updatable"


def test_explicit_non_background_response_cannot_be_publicly_cancelled() -> None:
    registry = ResponseRegistry()
    registry.begin(
        "resp_foreground",
        owner_id=OWNER_A,
        store=True,
        background=False,
        public_snapshot={
            **_live_snapshot("resp_foreground", "in_progress"),
            "background": False,
        },
        materialized_messages=[],
    )

    with pytest.raises(ResponseRegistryError) as not_cancellable:
        registry.request_cancel(
            "resp_foreground",
            "client_cancelled",
            owner_id=OWNER_A,
        )
    assert not_cancellable.value.code == "response_not_cancellable"
    assert registry.stats()["cancel_requests_total"] == 0
    assert registry.get("resp_foreground", owner_id=OWNER_A)["status"] == "in_progress"


def test_stable_input_item_ids_are_persisted_and_included_in_byte_accounting() -> None:
    registry = ResponseRegistry(max_bytes=4096, max_entry_bytes=4096)
    raw_items = [
        {"role": "user", "content": "hello"},
        {
            "type": "function_call",
            "role": "assistant",
            "id": "fc_official",
            "call_id": "call_1",
            "name": "lookup",
            "arguments": "{}",
        },
        {
            "type": "function_call_output",
            "role": "tool",
            "call_id": "call_1",
            "output": "ok",
        },
    ]
    raw_size = len(
        json.dumps(
            raw_items,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    registry.begin(
        "resp_stable_items",
        owner_id=OWNER_A,
        store=True,
        materialized_messages=raw_items,
    )

    first = registry.input_items("resp_stable_items", owner_id=OWNER_A)
    second = registry.input_messages("resp_stable_items", owner_id=OWNER_A)
    assert first == second
    assert [item["id"] for item in first] == [
        first[0]["id"],
        "fc_official",
        first[2]["id"],
    ]
    assert first[0]["id"].startswith("input_")
    assert first[2]["id"].startswith("input_")
    assert len({item["id"] for item in first}) == len(first)
    assert registry.stats()["in_flight_bytes"] > raw_size

    first[0]["content"] = "changed"
    assert (
        registry.input_items("resp_stable_items", owner_id=OWNER_A)[0]["content"]
        == "hello"
    )
