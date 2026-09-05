"""Contract tests for stable Responses input-item cursor pagination."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from mlx_batch_server.responses.input_items import (
    InputItemsPaginationError,
    paginate_input_items,
)
from mlx_batch_server.responses.registry import ResponseRegistry, ResponseRegistryError

OWNER_A = "principal:a"
OWNER_B = "principal:b"


def _items(count: int) -> list[dict]:
    return [
        {
            "id": f"item_{index:03d}",
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": str(index)}],
        }
        for index in range(count)
    ]


def _pages(items: list[dict], *, order: str, limit: int) -> Iterator[dict]:
    after = None
    while True:
        page = paginate_input_items(items, after=after, limit=limit, order=order)
        yield page
        if not page["has_more"]:
            return
        after = page["last_id"]


@pytest.mark.parametrize("order", ["asc", "desc"])
def test_pages_cover_every_item_once_in_requested_order(order: str) -> None:
    items = _items(47)
    pages = list(_pages(items, order=order, limit=7))
    seen = [item["id"] for page in pages for item in page["data"]]
    expected = [item["id"] for item in items]
    if order == "desc":
        expected.reverse()

    assert seen == expected
    assert len(seen) == len(set(seen))
    assert all(page["object"] == "list" for page in pages)
    assert all(page["first_id"] == page["data"][0]["id"] for page in pages)
    assert all(page["last_id"] == page["data"][-1]["id"] for page in pages)
    assert [page["has_more"] for page in pages] == [True] * 6 + [False]


@pytest.mark.parametrize(
    ("count", "after", "limit", "order", "expected_ids", "has_more"),
    [
        (0, None, 20, "desc", [], False),
        (1, None, 20, "desc", ["item_000"], False),
        (3, None, 2, "asc", ["item_000", "item_001"], True),
        (3, "item_001", 2, "asc", ["item_002"], False),
        (3, None, 2, "desc", ["item_002", "item_001"], True),
        (3, "item_001", 2, "desc", ["item_000"], False),
        (20, None, 20, "desc", [f"item_{i:03d}" for i in range(19, -1, -1)], False),
        (21, None, 20, "desc", [f"item_{i:03d}" for i in range(20, 0, -1)], True),
    ],
)
def test_cursor_truth_table(
    count: int,
    after: str | None,
    limit: int,
    order: str,
    expected_ids: list[str],
    has_more: bool,
) -> None:
    page = paginate_input_items(_items(count), after=after, limit=limit, order=order)

    assert [item["id"] for item in page["data"]] == expected_ids
    assert page == {
        "object": "list",
        "data": page["data"],
        "first_id": expected_ids[0] if expected_ids else None,
        "last_id": expected_ids[-1] if expected_ids else None,
        "has_more": has_more,
    }


def test_defaults_are_twenty_items_in_descending_order() -> None:
    page = paginate_input_items(_items(25))

    assert [item["id"] for item in page["data"]] == [
        f"item_{index:03d}" for index in range(24, 4, -1)
    ]
    assert page["has_more"] is True


@pytest.mark.parametrize("limit", [0, -1, 101, True, 1.5, "20"])
def test_invalid_limit_errors_name_only_limit(limit: object) -> None:
    with pytest.raises(InputItemsPaginationError) as invalid:
        paginate_input_items(_items(3), limit=limit)  # type: ignore[arg-type]
    assert invalid.value.param == "limit"
    assert invalid.value.code == "invalid_limit"


@pytest.mark.parametrize("order", ["ascending", "DESC", "", None, 7])
def test_invalid_order_errors_name_only_order(order: object) -> None:
    with pytest.raises(InputItemsPaginationError) as invalid:
        paginate_input_items(_items(3), order=order)  # type: ignore[arg-type]
    assert invalid.value.param == "order"
    assert invalid.value.code == "invalid_order"


@pytest.mark.parametrize("after", ["item_missing", "", "item_foreign"])
def test_unknown_foreign_and_stale_cursors_are_indistinguishable(after: str) -> None:
    with pytest.raises(InputItemsPaginationError) as invalid:
        paginate_input_items(_items(3), after=after)
    assert invalid.value.param == "after"
    assert invalid.value.code == "invalid_cursor"
    assert invalid.value.status_code == 400


def test_owner_registry_and_cursor_lookup_cannot_reveal_foreign_items() -> None:
    registry = ResponseRegistry()
    registry.begin(
        "resp_a",
        owner_id=OWNER_A,
        store=True,
        materialized_messages=[{"role": "user", "content": "secret-a"}],
    )
    registry.begin(
        "resp_b",
        owner_id=OWNER_B,
        store=True,
        materialized_messages=[{"role": "user", "content": "secret-b"}],
    )
    a_cursor = registry.input_items("resp_a", owner_id=OWNER_A)[0]["id"]
    b_items = registry.input_items("resp_b", owner_id=OWNER_B)

    with pytest.raises(ResponseRegistryError) as foreign_response:
        registry.input_items("resp_a", owner_id=OWNER_B)
    assert foreign_response.value.code == "response_not_found"

    errors = []
    for cursor in (a_cursor, "input_unknown"):
        with pytest.raises(InputItemsPaginationError) as invalid:
            paginate_input_items(b_items, after=cursor)
        errors.append((invalid.value.code, invalid.value.param, invalid.value.message))
    assert errors[0] == errors[1]


def test_registry_preserves_official_tool_items_and_ids_through_paging() -> None:
    canonical = []
    for index in range(9):
        canonical.extend(
            [
                {
                    "id": f"msg_{index}",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": f"turn {index}"}],
                },
                {
                    "id": f"fc_{index}",
                    "type": "function_call",
                    "role": "assistant",
                    "status": "completed",
                    "call_id": f"call_{index}",
                    "name": "lookup",
                    "arguments": f'{{"turn":{index}}}',
                },
                {
                    "id": f"fco_{index}",
                    "type": "function_call_output",
                    "role": "tool",
                    "status": "completed",
                    "call_id": f"call_{index}",
                    "output": f"result {index}",
                },
            ]
        )

    registry = ResponseRegistry()
    registry.begin(
        "resp_tools",
        owner_id=OWNER_A,
        store=True,
        materialized_messages=canonical,
    )
    registry.commit(
        "resp_tools",
        {
            "id": "resp_tools",
            "object": "response",
            "status": "completed",
            "output": [],
        },
        owner_id=OWNER_A,
        materialized_messages=canonical,
    )
    stored = registry.input_items("resp_tools", owner_id=OWNER_A)

    assert stored == canonical
    asc = [
        item for page in _pages(stored, order="asc", limit=5) for item in page["data"]
    ]
    desc = [
        item for page in _pages(stored, order="desc", limit=4) for item in page["data"]
    ]
    assert asc == canonical
    assert desc == list(reversed(canonical))
    assert registry.input_items("resp_tools", owner_id=OWNER_A) == canonical


def test_paginator_returns_copies_not_a_mutable_view_of_registry_truth() -> None:
    items = _items(2)
    page = paginate_input_items(items, order="asc")
    page["data"][0]["content"][0]["text"] = "changed"

    assert items[0]["content"][0]["text"] == "0"
