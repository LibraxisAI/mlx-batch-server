"""Boundary-safe exact matching for generated text stop sequences."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StopMatch:
    """One incremental matcher observation."""

    emitted: str
    stop_sequence: str | None = None

    @property
    def matched(self) -> bool:
        return self.stop_sequence is not None


class IncrementalStopMatcher:
    """Emit text only after it can no longer participate in an exact stop.

    Matches are selected by the earliest completed end position. If multiple
    configured sequences complete at that same position, their request order
    is authoritative. Once matched, the sequence itself and all same-chunk
    tail text are discarded.
    """

    __slots__ = ("_finished", "_matched", "_pending", "_stop_sequences")

    def __init__(self, stop_sequences: Sequence[str]) -> None:
        if isinstance(stop_sequences, str):
            raise TypeError("stop_sequences must be a sequence of strings")
        sequences = tuple(stop_sequences)
        if not sequences:
            raise ValueError("stop_sequences must not be empty")
        for sequence in sequences:
            if not isinstance(sequence, str):
                raise TypeError("stop_sequences entries must be strings")
            if not sequence:
                raise ValueError("stop_sequences entries must not be empty")
        self._stop_sequences = sequences
        self._pending = ""
        self._matched: str | None = None
        self._finished = False

    @property
    def pending(self) -> str:
        return self._pending

    @property
    def matched_stop_sequence(self) -> str | None:
        return self._matched

    def feed(self, chunk: str) -> StopMatch:
        if not isinstance(chunk, str):
            raise TypeError("stop matcher chunks must be strings")
        if self._finished:
            raise RuntimeError("cannot feed a finished stop matcher")
        if self._matched is not None:
            raise RuntimeError("cannot feed a matched stop matcher")

        combined = self._pending + chunk
        winner: tuple[int, int, int, str] | None = None
        for order, sequence in enumerate(self._stop_sequences):
            start = combined.find(sequence)
            if start < 0:
                continue
            candidate = (start + len(sequence), order, start, sequence)
            if winner is None or candidate[:2] < winner[:2]:
                winner = candidate
        if winner is not None:
            _, _, start, sequence = winner
            self._pending = ""
            self._matched = sequence
            return StopMatch(emitted=combined[:start], stop_sequence=sequence)

        keep = 0
        max_prefix = min(
            len(combined),
            max(len(sequence) - 1 for sequence in self._stop_sequences),
        )
        for length in range(1, max_prefix + 1):
            suffix = combined[-length:]
            if any(sequence.startswith(suffix) for sequence in self._stop_sequences):
                keep = length
        if keep:
            emitted = combined[:-keep]
            self._pending = combined[-keep:]
        else:
            emitted = combined
            self._pending = ""
        return StopMatch(emitted=emitted)

    def flush(self) -> str:
        """Emit the currently buffered prefix candidate without closing."""

        if self._matched is not None:
            return ""
        pending = self._pending
        self._pending = ""
        return pending

    def finish(self) -> str:
        """Close an unmatched stream and emit its buffered text exactly once."""

        if self._finished:
            return ""
        self._finished = True
        return self.flush()


__all__ = ["IncrementalStopMatcher", "StopMatch"]
