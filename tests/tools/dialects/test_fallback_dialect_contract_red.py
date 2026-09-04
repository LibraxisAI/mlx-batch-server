from __future__ import annotations

from mlx_batch_server.tools.dialects.fallback import PlainTextDialect
from mlx_batch_server.tools.parser import IncrementalToolParser


def test_plain_text_fallback_preserves_output_and_emits_no_calls() -> None:
    parser = IncrementalToolParser(PlainTextDialect())

    first, deltas = parser.feed("ordinary ")
    second, calls = parser.feed("model output")
    final, completed = parser.finish()

    assert first + second + final == "ordinary model output"
    assert deltas == ()
    assert calls == ()
    assert completed == ()
