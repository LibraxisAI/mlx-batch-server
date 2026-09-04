"""RED contracts for the canonical adapter to target-hosted tools."""

from __future__ import annotations

import json
from typing import Any

import pytest

from mlx_batch_server.tools.executor import RegistryToolExecutor
from mlx_batch_server.tools.parser import ParsedToolCall


def _call(arguments: str = '{"ticket":"LBRX-42"}') -> ParsedToolCall:
    return ParsedToolCall(
        index=0,
        call_id="call_lbrx_42",
        name="lookup_ticket",
        arguments=arguments,
    )


@pytest.mark.asyncio
async def test_registry_executor_preserves_call_id_and_arguments() -> None:
    seen: list[tuple[str, dict[str, Any]]] = []

    async def execute(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        seen.append((name, arguments))
        return {"success": True, "data": {"status": "open", "id": "LBRX-42"}}

    executor = RegistryToolExecutor(
        allowed_tools=frozenset({"lookup_ticket"}),
        execute=execute,
        is_hosted=lambda name: name == "lookup_ticket",
    )

    result = await executor.execute(_call())

    assert seen == [("lookup_ticket", {"ticket": "LBRX-42"})]
    assert result.call_id == "call_lbrx_42"
    assert result.ok is True
    assert json.loads(result.output) == {"id": "LBRX-42", "status": "open"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    ["[]", "null", '{"ticket":"first","ticket":"second"}', "{"],
)
async def test_registry_executor_rejects_non_object_or_ambiguous_json(
    arguments: str,
) -> None:
    called = False

    async def execute(_name: str, _arguments: dict[str, Any]) -> dict[str, Any]:
        nonlocal called
        called = True
        return {"success": True, "data": None}

    result = await RegistryToolExecutor(
        execute=execute,
        is_hosted=lambda _name: True,
    ).execute(_call(arguments))

    assert called is False
    assert result.ok is False
    assert result.metadata == {
        "tool_name": "lookup_ticket",
        "error_code": "invalid_tool_arguments",
    }


@pytest.mark.asyncio
async def test_client_tool_is_never_executed_server_side() -> None:
    async def unexpected(_name: str, _arguments: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("client tool must not execute")

    result = await RegistryToolExecutor(
        execute=unexpected,
        is_hosted=lambda _name: False,
    ).execute(_call())

    assert result.ok is False
    assert result.metadata["error_code"] == "client_tool_not_executable"


@pytest.mark.asyncio
async def test_allowlist_is_checked_before_registry_execution() -> None:
    async def unexpected(_name: str, _arguments: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("disallowed tool must not execute")

    result = await RegistryToolExecutor(
        allowed_tools=frozenset({"other_tool"}),
        execute=unexpected,
        is_hosted=lambda _name: True,
    ).execute(_call())

    assert result.ok is False
    assert result.metadata["error_code"] == "tool_not_allowed"


@pytest.mark.asyncio
async def test_registry_error_becomes_stable_failed_receipt() -> None:
    async def execute(_name: str, _arguments: dict[str, Any]) -> dict[str, Any]:
        return {
            "error": {
                "code": "ticket_unavailable",
                "message": "ticket service unavailable",
            }
        }

    result = await RegistryToolExecutor(
        execute=execute,
        is_hosted=lambda _name: True,
    ).execute(_call())

    assert result.call_id == "call_lbrx_42"
    assert result.ok is False
    assert result.error == "ticket service unavailable"
    assert result.metadata["error_code"] == "ticket_unavailable"


@pytest.mark.asyncio
async def test_registry_exception_becomes_stable_failed_receipt() -> None:
    async def execute(_name: str, _arguments: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("registry offline")

    result = await RegistryToolExecutor(
        execute=execute,
        is_hosted=lambda _name: True,
    ).execute(_call())

    assert result.call_id == "call_lbrx_42"
    assert result.ok is False
    assert result.error == "registry offline"
    assert result.metadata["error_code"] == "tool_execution_failed"


@pytest.mark.asyncio
async def test_non_json_registry_data_becomes_failed_receipt() -> None:
    async def execute(_name: str, _arguments: dict[str, Any]) -> dict[str, Any]:
        return {"success": True, "data": {"opaque": object()}}

    result = await RegistryToolExecutor(
        execute=execute,
        is_hosted=lambda _name: True,
    ).execute(_call())

    assert result.call_id == "call_lbrx_42"
    assert result.ok is False
    assert result.error == "tool registry data is not JSON-compatible"
    assert result.metadata["error_code"] == "invalid_tool_result"
