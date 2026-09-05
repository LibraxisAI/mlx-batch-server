"""Bounded incremental citation filter for hosted continuation text.

``CitationStreamFilter`` implements design HOSTED_STREAMING_SNAPSHOT_RECOVERY
§4: a causal character-stream automaton (PASS → TAG → SPAN/TAIL) that sits
in front of every public message ``TextDelta`` of an armed continuation round.
Raw ``<cite url="URL">quoted text</cite>`` sentinel markup is only ever held
or stripped, never forwarded; a citation exists only when the URL byte-equals
a proven success identity and the quote span-matches the proven source
content. Malformed, oversize, nested, unclosed, or unproven sentinels degrade
to plain text with zero citation events.

Hold-back is bounded: at any instant the filter withholds at most
``MAX_CITE_TAG_CHARS + MAX_CITED_TEXT_CHARS + MAX_CLOSE_TAG_CHARS``
characters. Ordinary text forwards immediately. Source offsets are computed
by the filter from the NFC+whitespace-normalized proven content and mapped
back into the original content; the sentinel syntax carries no ranges, so
model-supplied offsets do not exist. All state dies with the turn.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .events import MAX_CITED_TEXT_CHARS
from .turn import MAX_CITATIONS_PER_ITEM

MAX_CITE_TAG_CHARS = 256
MAX_CLOSE_TAG_CHARS = 8

CITATION_PREPARATION = (
    "When you quote a successful tool result verbatim, wrap the exact quoted "
    'words as <cite url="SOURCE_URL">quoted text</cite> using the exact URL '
    "of the source you are quoting. Use this markup only for verbatim quotes "
    "from the tool results of this conversation; never invent URLs, quotes, "
    "or citations."
)

_OPEN_NAME = "<cite"
_OPEN_PREFIX = "<cite "
_CLOSE_TAG = "</cite>"
_OPEN_TAG_PATTERN = re.compile(r'\A<cite url="([^"<>]+)">\Z')

# Fail during import if a later token edit invalidates the automaton's bounds.
if len(_CLOSE_TAG) >= MAX_CLOSE_TAG_CHARS:
    raise RuntimeError("MAX_CLOSE_TAG_CHARS must exceed the closing token length")
if len(_OPEN_PREFIX) >= MAX_CITE_TAG_CHARS:
    raise RuntimeError("MAX_CITE_TAG_CHARS must exceed the opening prefix length")

_STATE_PASS = 0
_STATE_TAG = 1
_STATE_TAG_SKIP = 2
_STATE_SPAN = 3
_STATE_TAIL = 4


@dataclass(frozen=True, slots=True)
class CitationSource:
    """One proven quotable source: a success result identity plus its content."""

    call_id: str
    url: str
    content: str


@dataclass(frozen=True, slots=True)
class ProvenCitation:
    """One filter-proven citation with computed (never model-supplied) offsets."""

    source_call_id: str
    source_url: str
    cited_text: str
    source_start: int
    source_end: int
    output_start: int
    output_end: int


class ItemCitationBudget:
    """Shared per-item bound: further proven sentinels degrade to plain text."""

    __slots__ = ("_remaining",)

    def __init__(self, limit: int = MAX_CITATIONS_PER_ITEM) -> None:
        if limit < 0:
            raise ValueError("citation budget must be non-negative")
        self._remaining = limit

    @property
    def available(self) -> bool:
        return self._remaining > 0

    def take(self) -> bool:
        if self._remaining <= 0:
            return False
        self._remaining -= 1
        return True


def _normalize_with_spans(text: str) -> tuple[str, tuple[tuple[int, int], ...]]:
    """NFC + whitespace-collapse with a normalized→original span map.

    Whitespace runs collapse to one space; base+combining segments are
    NFC-composed. Each normalized character maps to the half-open original
    span that produced it, so proven matches yield exact original offsets.
    """

    normalized: list[str] = []
    spans: list[tuple[int, int]] = []
    i = 0
    length = len(text)
    while i < length:
        if text[i].isspace():
            start = i
            while i < length and text[i].isspace():
                i += 1
            normalized.append(" ")
            spans.append((start, i))
            continue
        start = i
        i += 1
        while i < length and unicodedata.combining(text[i]):
            i += 1
        for out_ch in unicodedata.normalize("NFC", text[start:i]):
            normalized.append(out_ch)
            spans.append((start, i))
    return "".join(normalized), tuple(spans)


def _normalize_quote(text: str) -> str:
    normalized, _ = _normalize_with_spans(text)
    return normalized.strip(" ")


class _PreparedSource:
    __slots__ = ("call_id", "normalized", "spans", "url")

    def __init__(self, source: CitationSource) -> None:
        self.call_id = source.call_id
        self.url = source.url
        self.normalized, self.spans = _normalize_with_spans(source.content)


class CitationStreamFilter:
    """One content part's causal sentinel automaton (design §4.3).

    ``feed``/``finish`` return the ordered public output: clean text chunks
    interleaved with the ``ProvenCitation`` records they ground. The
    concatenation of every returned text chunk always equals
    ``filtered_text``, so rewritten done events stay equal to their deltas.
    """

    def __init__(
        self,
        sources: tuple[CitationSource, ...] | list[CitationSource],
        *,
        budget: ItemCitationBudget | None = None,
    ) -> None:
        self._by_url: dict[str, list[_PreparedSource]] = {}
        for source in sources:
            if not isinstance(source, CitationSource):
                raise TypeError("sources must be CitationSource instances")
            self._by_url.setdefault(source.url, []).append(_PreparedSource(source))
        self._budget = budget if budget is not None else ItemCitationBudget()
        self._state = _STATE_PASS
        self._tag = ""
        self._span = ""
        self._url = ""
        self._hold = ""
        self._tail_closes_remaining = 0
        self._text: list[str] = []
        self._emitted = 0
        self._out: list[str | ProvenCitation] = []
        self._finished = False

    @property
    def filtered_text(self) -> str:
        return "".join(self._text)

    def feed(self, delta: str) -> list[str | ProvenCitation]:
        if self._finished:
            raise RuntimeError("citation filter already finished")
        self._out = []
        for ch in delta:
            self._step(ch)
        return self._out

    def finish(self) -> list[str | ProvenCitation]:
        if self._finished:
            raise RuntimeError("citation filter already finished")
        self._finished = True
        self._out = []
        if self._state == _STATE_PASS:
            # An unresolved open-prefix probe was never markup: verbatim.
            if self._tag != _OPEN_NAME:
                self._emit(self._tag)
        elif self._state == _STATE_SPAN:
            # Unclosed sentinel: opener stripped, inner text emitted plain.
            self._emit(self._span)
            if self._hold != _OPEN_NAME:
                self._emit(self._hold)
        elif self._state == _STATE_TAIL and self._hold != _OPEN_NAME:
            self._emit(self._hold)
        # A committed opener (_STATE_TAG/_STATE_TAG_SKIP) is markup: stripped.
        self._tag = ""
        self._span = ""
        self._hold = ""
        return self._out

    def _step(self, ch: str) -> None:
        handlers = {
            _STATE_PASS: self._step_pass,
            _STATE_TAG: self._step_tag,
            _STATE_TAG_SKIP: self._step_tag_skip,
            _STATE_SPAN: self._step_span,
            _STATE_TAIL: self._step_tail,
        }
        # A handler returning True re-dispatches the same character against
        # the state it just switched to (bounded: every switch sheds state).
        while handlers[self._state](ch):
            pass

    def _step_pass(self, ch: str) -> bool:
        if self._tag:
            if self._tag == _OPEN_NAME:
                return self._commit_pass_open(ch)
            candidate = self._tag + ch
            if _OPEN_NAME.startswith(candidate):
                self._tag = candidate
                return False
            self._emit(self._tag)
            self._tag = ""
            return True
        if ch == "<":
            self._tag = "<"
            return False
        self._emit(ch)
        return False

    def _commit_pass_open(self, ch: str) -> bool:
        self._tag = ""
        self._tail_closes_remaining = 1
        if ch == " ":
            self._tag = _OPEN_PREFIX
            self._state = _STATE_TAG
            return False
        if ch == ">":
            self._state = _STATE_TAIL
            return False
        self._state = _STATE_TAG_SKIP
        return True

    def _step_tag(self, ch: str) -> bool:
        if ch == "<":
            # Nested/malformed opener: strip it; _TAIL still strips the
            # orphaned close tag and probes fresh openers.
            self._tag = ""
            self._tail_closes_remaining = 1
            self._state = _STATE_TAIL
            return True
        self._tag += ch
        if ch == ">":
            match = _OPEN_TAG_PATTERN.match(self._tag)
            self._tag = ""
            if match is None:
                # Malformed/unmatched-quote opener: markup stripped, the
                # following text stays plain, orphan close stripped by _TAIL.
                self._tail_closes_remaining = 1
                self._state = _STATE_TAIL
                return False
            self._url = match.group(1)
            self._span = ""
            self._hold = ""
            self._state = _STATE_SPAN
            return False
        if len(self._tag) > MAX_CITE_TAG_CHARS:
            # Oversize opener: discard the rest of its attribute markup up
            # to ">" without holding anything.
            self._tag = ""
            self._tail_closes_remaining = 1
            self._state = _STATE_TAG_SKIP
        return False

    def _step_tag_skip(self, ch: str) -> bool:
        if ch == ">":
            self._state = _STATE_TAIL
        return False

    def _step_span(self, ch: str) -> bool:
        if self._hold:
            if self._hold == _OPEN_NAME:
                return self._degrade_nested_span(ch)
            candidate = self._hold + ch
            if _CLOSE_TAG.startswith(candidate):
                self._hold = candidate
                if candidate == _CLOSE_TAG:
                    self._hold = ""
                    self._resolve_span()
                    self._state = _STATE_PASS
                return False
            if _OPEN_NAME.startswith(candidate):
                self._hold = candidate
                return False
            held = self._hold
            self._hold = ""
            self._append_span(held)
            return True
        if ch == "<":
            self._hold = "<"
            return False
        self._append_span(ch)
        return False

    def _degrade_nested_span(self, ch: str) -> bool:
        # Any nested citation-shaped opener invalidates the outer sentinel.
        # Strip both control layers, preserve all human text, and emit no
        # citation even if the inner quote proves.
        self._emit(self._span)
        self._span = ""
        self._url = ""
        self._hold = ""
        self._tail_closes_remaining = 2
        if ch == ">":
            self._state = _STATE_TAIL
            return False
        self._state = _STATE_TAG_SKIP
        return ch != " "

    def _step_tail(self, ch: str) -> bool:
        # After a degraded sentinel: pass through human text while stripping
        # all control layers. Nested openers increase the number of closes to
        # consume and can never become a fresh citation candidate.
        if self._hold:
            if self._hold == _OPEN_NAME:
                return self._consume_nested_tail_opener(ch)
            candidate = self._hold + ch
            if _CLOSE_TAG.startswith(candidate):
                self._hold = candidate
                if candidate == _CLOSE_TAG:
                    self._hold = ""
                    self._tail_closes_remaining -= 1
                    if self._tail_closes_remaining <= 0:
                        self._state = _STATE_PASS
                return False
            if _OPEN_NAME.startswith(candidate):
                self._hold = candidate
                return False
            self._emit(self._hold)
            self._hold = ""
            return True
        if ch == "<":
            self._hold = "<"
            return False
        self._emit(ch)
        return False

    def _consume_nested_tail_opener(self, ch: str) -> bool:
        self._hold = ""
        self._tail_closes_remaining += 1
        if ch == ">":
            return False
        self._state = _STATE_TAG_SKIP
        return ch != " "

    def _append_span(self, text: str) -> None:
        self._span += text
        if len(self._span) > MAX_CITED_TEXT_CHARS:
            # Oversize quote: opener stripped, span flushes plain, the
            # orphaned close tag (if any) is stripped by _TAIL.
            self._emit(self._span)
            self._span = ""
            self._url = ""
            self._tail_closes_remaining = 1
            self._state = _STATE_TAIL

    def _resolve_span(self) -> None:
        span = self._span
        url = self._url
        self._span = ""
        self._url = ""
        if not span:
            return
        proof = self._prove(url, span) if self._budget.available else None
        start = self._emitted
        self._emit(span)
        if proof is None:
            return
        call_id, source_start, source_end = proof
        if not self._budget.take():  # pragma: no cover - availability checked
            return
        self._out.append(
            ProvenCitation(
                source_call_id=call_id,
                source_url=url,
                cited_text=span,
                source_start=source_start,
                source_end=source_end,
                output_start=start,
                output_end=start + len(span),
            )
        )

    def _prove(self, url: str, span: str) -> tuple[str, int, int] | None:
        candidates = self._by_url.get(url)
        if not candidates:
            return None
        quote = _normalize_quote(span)
        if not quote:
            return None
        for source in candidates:
            index = source.normalized.find(quote)
            if index < 0:
                continue
            source_start = source.spans[index][0]
            source_end = source.spans[index + len(quote) - 1][1]
            return source.call_id, source_start, source_end
        return None

    def _emit(self, text: str) -> None:
        if not text:
            return
        self._text.append(text)
        self._emitted += len(text)
        if self._out and isinstance(self._out[-1], str):
            self._out[-1] += text
        else:
            self._out.append(text)


__all__ = [
    "CITATION_PREPARATION",
    "MAX_CITATIONS_PER_ITEM",
    "MAX_CITED_TEXT_CHARS",
    "MAX_CITE_TAG_CHARS",
    "MAX_CLOSE_TAG_CHARS",
    "CitationSource",
    "CitationStreamFilter",
    "ItemCitationBudget",
    "ProvenCitation",
]
