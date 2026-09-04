"""Small stateless helpers shared by cumulative tool dialects."""

from __future__ import annotations


def stable_call_id(dialect: str, index: int) -> str:
    """Return a deterministic identity that cannot change as output grows."""

    return f"call_{dialect}_{index}"


def hold_partial_marker(text: str, markers: tuple[str, ...]) -> str:
    """Hide a trailing marker prefix until it is proven to be normal text."""

    keep = 0
    for marker in markers:
        max_size = min(len(text), len(marker) - 1)
        for size in range(max_size, 0, -1):
            if text.endswith(marker[:size]):
                keep = max(keep, size)
                break
    return text[:-keep] if keep else text


def skip_ascii_space(text: str, index: int) -> int:
    while index < len(text) and text[index] in " \t\r\n":
        index += 1
    return index


def json_value_end(text: str, start: int) -> int | None:
    """Return the end of one complete JSON value without decoding it."""

    start = skip_ascii_space(text, start)
    if start >= len(text):
        return None

    opener = text[start]
    if opener not in "{[":
        return None
    closer = "}" if opener == "{" else "]"
    stack = [closer]
    quoted = False
    escaped = False

    for index in range(start + 1, len(text)):
        char = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue

        if char == '"':
            quoted = True
        elif char == "{":
            stack.append("}")
        elif char == "[":
            stack.append("]")
        elif char in "}]":
            if not stack or stack[-1] != char:
                return None
            stack.pop()
            if not stack:
                return index + 1
    return None


def json_value_prefix(text: str, start: int) -> str:
    """Return a complete JSON value or its cumulative unfinished prefix."""

    start = skip_ascii_space(text, start)
    end = json_value_end(text, start)
    return text[start:end] if end is not None else text[start:]
