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
from collections.abc import Iterator, Sequence
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


def _nfd_units_with_spans(text: str) -> list[tuple[str, int, int]]:
    """Return the full canonical decomposition with original-codepoint spans."""

    decomposed_units: list[tuple[str, int, int]] = []
    for index, char in enumerate(text):
        for decomposed in unicodedata.normalize("NFD", char):
            decomposed_units.append((decomposed, index, index + 1))

    # Canonical ordering is a stable sort of each non-starter run. It may cross
    # original-codepoint boundaries, so the source span travels with its unit.
    units: list[tuple[str, int, int]] = []
    nonstarters: list[tuple[str, int, int]] = []
    for unit in decomposed_units:
        if unicodedata.combining(unit[0]) == 0:
            units.extend(
                sorted(nonstarters, key=lambda item: unicodedata.combining(item[0]))
            )
            nonstarters.clear()
            units.append(unit)
        else:
            nonstarters.append(unit)
    units.extend(sorted(nonstarters, key=lambda item: unicodedata.combining(item[0])))
    if "".join(char for char, _, _ in units) != unicodedata.normalize("NFD", text):
        raise RuntimeError("canonical decomposition span alignment failed")
    return units


def _nfc_with_spans(text: str) -> tuple[str, tuple[tuple[int, int], ...]]:
    """Normalize the entire string to NFC and align output to original spans."""

    units = _nfd_units_with_spans(text)
    normalized = unicodedata.normalize("NFC", text)
    spans: list[tuple[int, int]] = []
    cursor = 0
    for char in normalized:
        decomposition = unicodedata.normalize("NFD", char)
        consumed = units[cursor : cursor + len(decomposition)]
        if "".join(unit[0] for unit in consumed) != decomposition:
            raise RuntimeError("canonical composition span alignment failed")
        spans.append(
            (
                min(unit[1] for unit in consumed),
                max(unit[2] for unit in consumed),
            )
        )
        cursor += len(decomposition)
    if cursor != len(units):
        raise RuntimeError("canonical composition left unaligned input")
    return normalized, tuple(spans)


def _normalize_with_spans(text: str) -> tuple[str, tuple[tuple[int, int], ...]]:
    """Full-string NFC plus whitespace-collapse and exact original spans."""

    nfc, nfc_spans = _nfc_with_spans(text)
    normalized: list[str] = []
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(nfc):
        if not nfc[index].isspace():
            normalized.append(nfc[index])
            spans.append(nfc_spans[index])
            index += 1
            continue
        end = index + 1
        while end < len(nfc) and nfc[end].isspace():
            end += 1
        whitespace_spans = nfc_spans[index:end]
        normalized.append(" ")
        spans.append(
            (
                min(span[0] for span in whitespace_spans),
                max(span[1] for span in whitespace_spans),
            )
        )
        index = end
    return "".join(normalized), tuple(spans)


def _normalize_quote(text: str) -> str:
    normalized, _ = _normalize_with_spans(text)
    return normalized


@dataclass(frozen=True, slots=True)
class _PreparedSource:
    call_id: str
    url: str
    normalized: str
    spans: tuple[tuple[int, int], ...]


def _prepare_source(source: CitationSource) -> _PreparedSource:
    if not isinstance(source, CitationSource):
        raise TypeError("sources must be CitationSource instances")
    normalized, spans = _normalize_with_spans(source.content)
    return _PreparedSource(source.call_id, source.url, normalized, spans)


@dataclass(frozen=True, slots=True)
class PreparedCitationCorpus:
    """Persistent per-turn corpus sharing each immutable prepared source once."""

    _parent: PreparedCitationCorpus | None = None
    _added: tuple[_PreparedSource, ...] = ()

    @classmethod
    def from_sources(
        cls,
        sources: Sequence[CitationSource],
    ) -> PreparedCitationCorpus:
        return cls().extend(sources)

    def extend(
        self,
        sources: Sequence[CitationSource],
    ) -> PreparedCitationCorpus:
        prepared = tuple(_prepare_source(source) for source in sources)
        if not prepared:
            return self
        return type(self)(_parent=self, _added=prepared)

    def candidates(self, url: str) -> Iterator[_PreparedSource]:
        generations: list[tuple[_PreparedSource, ...]] = []
        corpus: PreparedCitationCorpus | None = self
        while corpus is not None:
            generations.append(corpus._added)
            corpus = corpus._parent
        for generation in reversed(generations):
            for source in generation:
                if source.url == url:
                    yield source

    def __bool__(self) -> bool:
        corpus: PreparedCitationCorpus | None = self
        while corpus is not None:
            if corpus._added:
                return True
            corpus = corpus._parent
        return False


class CitationStreamFilter:
    """One content part's causal sentinel automaton (design §4.3).

    ``feed``/``finish`` return the ordered public output: clean text chunks
    interleaved with the ``ProvenCitation`` records they ground. The
    concatenation of every returned text chunk always equals
    ``filtered_text``, so rewritten done events stay equal to their deltas.
    """

    def __init__(
        self,
        corpus: PreparedCitationCorpus,
        *,
        budget: ItemCitationBudget | None = None,
    ) -> None:
        if not isinstance(corpus, PreparedCitationCorpus):
            raise TypeError("corpus must be a PreparedCitationCorpus")
        self._corpus = corpus
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
        quote = _normalize_quote(span)
        if not quote:
            return None
        for source in self._corpus.candidates(url):
            index = source.normalized.find(quote)
            if index < 0:
                continue
            matched_spans = source.spans[index : index + len(quote)]
            source_start = min(source_span[0] for source_span in matched_spans)
            source_end = max(source_span[1] for source_span in matched_spans)
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
    "PreparedCitationCorpus",
    "ProvenCitation",
]
