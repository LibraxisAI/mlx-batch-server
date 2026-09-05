from __future__ import annotations

from mlx_batch_server.utils.streaming_detokenizer import (
    finalize_text,
    new_streaming_detokenizer,
    push_token,
)


class _ByteTokenizer:
    _tokens = {
        1: b"Cze\xc5",
        2: b"\x9b\xc4",
        3: b"\x87! ",
        4: b"\xf0\x9f",
        5: b"\x98\x8a",
    }

    @staticmethod
    def encode(text: str, *, add_special_tokens: bool = False) -> list[int]:
        del text, add_special_tokens
        return [0]

    def decode(self, tokens: list[int]) -> str:
        return b"".join(self._tokens.get(token, b"") for token in tokens).decode(
            "utf-8",
            "replace",
        )


def test_streaming_detokenizer_never_leaks_partial_utf8_replacements() -> None:
    detokenizer = new_streaming_detokenizer(_ByteTokenizer())

    deltas = [push_token(detokenizer, token) for token in (1, 2, 3, 4, 5)]
    deltas.append(finalize_text(detokenizer))

    assert "\ufffd" not in "".join(deltas)
    assert "".join(deltas) == "Cześć! 😊"
