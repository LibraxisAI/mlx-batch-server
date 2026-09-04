from __future__ import annotations

import json

import pytest

from mlx_batch_server.tools.dialects.harmony import HarmonyToolDialect
from mlx_batch_server.tools.parser import IncrementalToolParser


def test_harmony_hides_analysis_and_streams_tool_arguments_once() -> None:
    parser = IncrementalToolParser(HarmonyToolDialect())
    prefix = (
        "<|start|>assistant<|channel|>analysis<|message|>private reasoning<|end|>"
        "<|start|>assistant<|channel|>analysis to=functions.lookup_ticket"
        "<|constrain|>json<|message|>"
    )

    visible, deltas = parser.feed(prefix + '{"ticket":"LBR')
    assert visible == ""
    assert len(deltas) == 1
    assert deltas[0].call_id == "call_harmony_0"
    assert deltas[0].name == "lookup_ticket"
    assert deltas[0].arguments_delta == '{"ticket":"LBR'

    visible, deltas = parser.feed('X-42"}<|ca')
    assert visible == ""
    assert deltas[0].call_id == "call_harmony_0"
    assert deltas[0].name is None
    assert deltas[0].arguments_delta == 'X-42"}'

    parser.feed("ll|><|start|>assistant<|channel|>final<|message|>Ticket found.<|end|>")
    visible, calls = parser.finish()

    assert visible == ""
    assert calls[0].call_id == "call_harmony_0"
    assert json.loads(calls[0].arguments) == {"ticket": "LBRX-42"}


def test_harmony_exposes_only_final_channel_text() -> None:
    dialect = HarmonyToolDialect()
    source = (
        "<|start|>assistant<|channel|>analysis<|message|>hidden<|end|>"
        "<|start|>assistant<|channel|>final<|message|>Visible answer<|end|>"
        "<|return|>"
    )

    snapshot = dialect.parse(source, final=True)

    assert snapshot.visible_text == "Visible answer"
    assert snapshot.calls == ()
    assert "<|" not in snapshot.visible_text


def test_harmony_partial_control_marker_never_becomes_visible() -> None:
    parser = IncrementalToolParser(HarmonyToolDialect())

    visible, _deltas = parser.feed("hello<|start|>assistant<|cha")

    assert visible == "hello"

    visible, _deltas = parser.feed("nnel|>final<|message|>world<|end|>")

    assert visible == "world"


@pytest.mark.parametrize(
    "split",
    range(1, len("<|start|>assistant<|channel|>")),
)
def test_harmony_every_split_of_assistant_marker_is_hidden(split: int) -> None:
    parser = IncrementalToolParser(HarmonyToolDialect())
    marker = "<|start|>assistant<|channel|>"
    source = f"hello{marker}final<|message|>world<|end|>"

    visible_before, _ = parser.feed(source[: len("hello") + split])
    visible_after, _ = parser.feed(source[len("hello") + split :])
    visible_final, calls = parser.finish()

    assert visible_before + visible_after + visible_final == "helloworld"
    assert calls == ()


def test_harmony_call_marker_inside_json_string_is_not_a_delimiter() -> None:
    parser = IncrementalToolParser(HarmonyToolDialect())
    visible_delta, _ = parser.feed(
        "<|start|>assistant<|channel|>analysis to=functions.echo"
        "<|constrain|>json<|message|>"
        '{"value":"literal <|call|> marker"}<|call|>'
        "<|start|>assistant<|channel|>final<|message|>done<|end|>"
    )

    visible, calls = parser.finish()

    assert visible_delta == "done"
    assert visible == ""
    assert len(calls) == 1
    assert json.loads(calls[0].arguments) == {"value": "literal <|call|> marker"}
