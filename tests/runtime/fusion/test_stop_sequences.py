from __future__ import annotations

import pytest

from mlx_batch_server.runtime.fusion.stop_sequences import IncrementalStopMatcher


def _run(chunks: tuple[str, ...], stops: tuple[str, ...]) -> tuple[str, str | None]:
    matcher = IncrementalStopMatcher(stops)
    emitted: list[str] = []
    matched: str | None = None
    for chunk in chunks:
        observation = matcher.feed(chunk)
        emitted.append(observation.emitted)
        if observation.matched:
            matched = observation.stop_sequence
            break
    if matched is None:
        emitted.append(matcher.finish())
    return "".join(emitted), matched


@pytest.mark.parametrize("stop", ("END", "<stop>", "żółw", "🙂done"))
def test_every_stop_boundary_discards_match_and_same_chunk_tail(stop: str) -> None:
    source = f"before{stop}discarded"
    start = len("before")
    for split in range(start, start + len(stop) + 1):
        chunks = (source[:split], source[split:])
        assert _run(chunks, (stop,)) == ("before", stop)


def test_overlap_prefix_and_same_end_ties_are_deterministic() -> None:
    assert _run(("zabc-tail",), ("abc", "b")) == ("za", "b")
    assert _run(("zabc-tail",), ("bc", "abc")) == ("za", "bc")
    assert _run(("zabc-tail",), ("abc", "bc")) == ("z", "abc")
    assert _run(("xxabab-tail",), ("abab", "bab")) == ("xx", "abab")


def test_unmatched_terminal_flushes_buffer_exactly_once() -> None:
    matcher = IncrementalStopMatcher(("END", "ENDING"))
    observations = [matcher.feed("keep E"), matcher.feed("N")]

    assert "".join(item.emitted for item in observations) == "keep "
    assert matcher.pending == "EN"
    assert matcher.finish() == "EN"
    assert matcher.finish() == ""


def test_flush_breaks_a_channel_boundary_without_closing_matcher() -> None:
    matcher = IncrementalStopMatcher(("END",))

    assert matcher.feed("reason EN").emitted == "reason "
    assert matcher.flush() == "EN"
    assert matcher.feed("D visible").emitted == "D visible"
    assert matcher.finish() == ""


@pytest.mark.parametrize("stops", ((), ("",), ("ok", 1)))
def test_invalid_stop_configuration_fails_closed(stops: tuple[object, ...]) -> None:
    with pytest.raises((TypeError, ValueError)):
        IncrementalStopMatcher(stops)  # type: ignore[arg-type]
