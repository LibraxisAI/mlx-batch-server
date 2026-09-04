from __future__ import annotations

import json

import pytest

from mlx_batch_server.tools.dialects.qwen import QwenDialectError, QwenToolDialect
from mlx_batch_server.tools.parser import IncrementalToolParser


def test_qwen_json_stream_has_stable_identity_and_no_marker_leakage() -> None:
    parser = IncrementalToolParser(QwenToolDialect())

    visible, deltas = parser.feed('Ready.<tool_call>{"name":"lookup","arg')
    assert visible == "Ready."
    assert len(deltas) == 1
    assert deltas[0].call_id == "call_qwen_0"
    assert deltas[0].name == "lookup"
    assert deltas[0].arguments_delta == ""

    visible, deltas = parser.feed('uments":{"ticket":"LBRX')
    assert visible == ""
    assert len(deltas) == 1
    assert deltas[0].call_id == "call_qwen_0"
    assert deltas[0].name is None
    assert deltas[0].arguments_delta == '{"ticket":"LBRX'

    visible, deltas = parser.feed('-42"}}</tool_call>')
    assert visible == ""
    assert len(deltas) == 1
    assert deltas[0].call_id == "call_qwen_0"
    assert deltas[0].name is None
    assert deltas[0].arguments_delta == '-42"}'

    visible, calls = parser.finish()
    assert visible == ""
    assert calls[0].call_id == "call_qwen_0"
    assert json.loads(calls[0].arguments) == {"ticket": "LBRX-42"}


def test_qwen_xml_function_normalizes_parameters() -> None:
    parser = IncrementalToolParser(QwenToolDialect())
    parser.feed(
        "<tool_call><function=record_lab_values>"
        '<parameter=wbc>12.5</parameter><parameter=patient>"Burek"</parameter>'
        "</function></tool_call>"
    )

    _visible, calls = parser.finish()

    assert calls[0].name == "record_lab_values"
    assert json.loads(calls[0].arguments) == {"wbc": 12.5, "patient": "Burek"}


def test_qwen_close_marker_inside_json_string_does_not_end_call() -> None:
    parser = IncrementalToolParser(QwenToolDialect())
    parser.feed(
        '<tool_call>{"name":"echo","arguments":{"value":"</tool_call>"}}</tool_call>'
    )

    visible, calls = parser.finish()

    assert visible == ""
    assert json.loads(calls[0].arguments) == {"value": "</tool_call>"}


def test_qwen_literal_parameter_close_inside_value_is_preserved() -> None:
    parser = IncrementalToolParser(QwenToolDialect())
    parser.feed(
        "<tool_call><function=echo>"
        "<parameter=value>literal </parameter> marker</parameter>"
        "</function></tool_call>"
    )

    _visible, calls = parser.finish()

    assert json.loads(calls[0].arguments) == {"value": "literal </parameter> marker"}


def test_qwen_unterminated_final_envelope_is_rejected_without_leaking() -> None:
    parser = IncrementalToolParser(QwenToolDialect())
    visible, _deltas = parser.feed("answer<tool_call>{")
    assert visible == "answer"

    with pytest.raises(QwenDialectError, match="unterminated"):
        parser.finish()


@pytest.mark.parametrize("split", [1, 17, 43, 71, 96])
def test_qwen_nested_fields_cannot_shadow_top_level_fields(split: int) -> None:
    parser = IncrementalToolParser(QwenToolDialect())
    source = (
        '<tool_call>{"metadata":{"name":"wrong","arguments":{"bad":1}},'
        '"name":"right","arguments":{"ok":2}}</tool_call>'
    )

    parser.feed(source[:split])
    parser.feed(source[split:])
    _visible, calls = parser.finish()

    assert len(calls) == 1
    assert calls[0].name == "right"
    assert json.loads(calls[0].arguments) == {"ok": 2}


@pytest.mark.parametrize("split", [1, 22, 48, 72])
def test_qwen_duplicate_top_level_keys_are_rejected_for_every_chunking(
    split: int,
) -> None:
    parser = IncrementalToolParser(QwenToolDialect())
    source = '<tool_call>{"name":"first","arguments":{},"name":"second"}</tool_call>'

    with pytest.raises(QwenDialectError, match="duplicate"):
        parser.feed(source[:split])
        parser.feed(source[split:])
        parser.finish()


def test_malformed_qwen_never_emits_closing_marker_as_arguments() -> None:
    parser = IncrementalToolParser(QwenToolDialect())

    visible, deltas = parser.feed(
        '<tool_call>{"name":"lookup","arguments":{"ticket":1]}</tool_call>'
    )

    assert visible == ""
    assert deltas
    assert all("</tool_call>" not in delta.arguments_delta for delta in deltas)
    with pytest.raises(QwenDialectError, match="unterminated"):
        parser.finish()
