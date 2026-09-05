"""Pure cursor pagination for canonical Responses input items."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

DEFAULT_INPUT_ITEMS_LIMIT = 20
MAX_INPUT_ITEMS_LIMIT = 100
_ORDERS = frozenset(("asc", "desc"))


class InputItemsPaginationError(ValueError):
    """A field-specific input-items pagination contract error."""

    def __init__(self, message: str, *, code: str, param: str) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.param = param
        self.status_code = 400

    def payload(self) -> dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": "invalid_request_error",
                "param": self.param,
                "code": self.code,
            }
        }


def paginate_input_items(
    items: Sequence[Mapping[str, Any]],
    *,
    after: str | None = None,
    limit: int = DEFAULT_INPUT_ITEMS_LIMIT,
    order: str = "desc",
) -> dict[str, Any]:
    """Page one immutable canonical item sequence using item-ID cursors.

    ``after`` is interpreted after applying the requested order. A cursor is
    deliberately resolved only against the supplied owned sequence, so random,
    stale, and foreign cursors have one indistinguishable failure shape.
    """

    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= MAX_INPUT_ITEMS_LIMIT
    ):
        raise InputItemsPaginationError(
            "limit must be an integer between 1 and 100",
            code="invalid_limit",
            param="limit",
        )
    if not isinstance(order, str) or order not in _ORDERS:
        raise InputItemsPaginationError(
            "order must be 'asc' or 'desc'",
            code="invalid_order",
            param="order",
        )

    canonical = _json_clone(list(items))
    ordered = canonical if order == "asc" else list(reversed(canonical))
    start = 0
    if after is not None:
        if not isinstance(after, str) or not after:
            _raise_invalid_cursor()
        for index, item in enumerate(ordered):
            if item.get("id") == after:
                start = index + 1
                break
        else:
            _raise_invalid_cursor()

    data = ordered[start : start + limit]
    return {
        "object": "list",
        "data": data,
        "first_id": data[0]["id"] if data else None,
        "last_id": data[-1]["id"] if data else None,
        "has_more": start + len(data) < len(ordered),
    }


def _json_clone(value: Any) -> Any:
    try:
        cloned = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise InputItemsPaginationError(
            "input items must be JSON-compatible",
            code="invalid_input_items",
            param="after",
        ) from exc
    if not isinstance(cloned, list) or any(
        not isinstance(item, dict) for item in cloned
    ):
        raise InputItemsPaginationError(
            "input items must be mappings",
            code="invalid_input_items",
            param="after",
        )
    seen: set[str] = set()
    for item in cloned:
        item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id or item_id in seen:
            raise InputItemsPaginationError(
                "input items must have unique non-empty IDs",
                code="invalid_input_items",
                param="after",
            )
        seen.add(item_id)
    return cloned


def _raise_invalid_cursor() -> None:
    raise InputItemsPaginationError(
        "after cursor was not found for this response",
        code="invalid_cursor",
        param="after",
    )
