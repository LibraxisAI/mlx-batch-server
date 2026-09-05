"""Falsifiers for the bounded incremental citation filter (design §4, §8).

Every case proves byte causality: markup is only ever held or stripped, the
concatenated clean output equals ``filtered_text`` under every possible delta
boundary placement, and a citation exists only for a proven URL identity plus
a proven verbatim span with filter-computed offsets.
"""

from __future__ import annotations

import unicodedata

import pytest

from mlx_batch_server.runtime.citations import (
    MAX_CITATIONS_PER_ITEM,
    MAX_CITE_TAG_CHARS,
    MAX_CITED_TEXT_CHARS,
    MAX_CLOSE_TAG_CHARS,
    CitationSource,
    CitationStreamFilter,
    ItemCitationBudget,
    PreparedCitationCorpus,
    ProvenCitation,
)

_URL = "https://doc.example/loctree"
_CONTENT = (
    "Loctree gives structural sight before you touch anything.\n"
    "It maps  the blast radius."
)
_SOURCES = (
    CitationSource(call_id="call_a", url=_URL, content=_CONTENT),
    CitationSource(
        call_id="call_s",
        url="https://s.example/hit",
        content="A snippet about structural perception.",
    ),
)
_QUOTE = "It maps the blast radius."
_VALID = f'Before <cite url="{_URL}">{_QUOTE}</cite> after.'
_VALID_CLEAN = f"Before {_QUOTE} after."


def _filter(
    sources: tuple[CitationSource, ...] = _SOURCES,
    *,
    budget: ItemCitationBudget | None = None,
) -> CitationStreamFilter:
    return CitationStreamFilter(
        PreparedCitationCorpus.from_sources(sources),
        budget=budget,
    )


def _normalized(text: str) -> str:
    output: list[str] = []
    in_whitespace = False
    for char in unicodedata.normalize("NFC", text):
        if char.isspace():
            if not in_whitespace:
                output.append(" ")
            in_whitespace = True
        else:
            output.append(char)
            in_whitespace = False
    return "".join(output)


def _run(
    text: str, *, chunks: tuple[str, ...] | None = None
) -> tuple[
    str,
    list[ProvenCitation],
]:
    stream_filter = _filter()
    citations: list[ProvenCitation] = []
    output: list[str] = []
    for chunk in chunks if chunks is not None else (text,):
        for piece in stream_filter.feed(chunk):
            if isinstance(piece, str):
                output.append(piece)
            else:
                citations.append(piece)
    for piece in stream_filter.finish():
        if isinstance(piece, str):
            output.append(piece)
        else:
            citations.append(piece)
    joined = "".join(output)
    assert joined == stream_filter.filtered_text
    for citation in citations:
        assert (
            joined[citation.output_start : citation.output_end] == citation.cited_text
        )
        source = next(
            source
            for source in _SOURCES
            if source.call_id == citation.source_call_id
            and source.url == citation.source_url
        )
        source_slice = source.content[citation.source_start : citation.source_end]
        assert _normalized(source_slice) == _normalized(citation.cited_text)
    return joined, citations


def test_valid_sentinel_yields_clean_text_and_one_proven_citation() -> None:
    text, citations = _run(_VALID)
    assert text == _VALID_CLEAN
    assert "<cite" not in text
    assert len(citations) == 1
    citation = citations[0]
    assert citation.source_call_id == "call_a"
    assert citation.source_url == _URL
    assert citation.cited_text == _QUOTE
    # Output offsets index into the filtered text; source offsets map the
    # NFC+whitespace-normalized match back into the original content.
    assert text[citation.output_start : citation.output_end] == _QUOTE
    assert citation.source_start == _CONTENT.index("It maps")
    assert citation.source_end == len(_CONTENT)


def test_every_character_split_is_invisible() -> None:
    """Falsifier #7: boundary placement can never change the public output."""

    corpus = (
        _VALID,
        f'<cite url="{_URL}">{_QUOTE}</cite>',
        'x <cite url="https://unknown.example">not proven</cite> y',
        "plain < text with <c and <citrus> fruit",
        f'nested <cite url="{_URL}">a <cite url="{_URL}">{_QUOTE}</cite> b',
        f'unclosed <cite url="{_URL}">trailing words',
        "malformed <cite>inner</cite> tail",
        "malformed <citeX>inner</cite> tail",
        'malformed <cite href="x">inner</cite> tail',
        f"unmatched <cite url={_URL}>quote</cite> tail",
        "orphan close</cite> stays",
    )
    for sample in corpus:
        baseline = _run(sample)
        for split in range(1, len(sample)):
            observed = _run(sample, chunks=(sample[:split], sample[split:]))
            assert observed == baseline, f"{sample!r} diverged at split {split}"
        char_by_char = _run(sample, chunks=tuple(sample))
        assert char_by_char == baseline, f"{sample!r} diverged char-by-char"


def test_markup_never_escapes_for_true_sentinels() -> None:
    for sample in (
        _VALID,
        f'unclosed <cite url="{_URL}">tail text',
        'unknown <cite url="https://unknown.example">quoted</cite>.',
        f'nested <cite url="{_URL}">a <cite url="{_URL}">{_QUOTE}</cite>',
    ):
        text, _ = _run(sample)
        assert "<cite" not in text
        assert "</cite>" not in text


def test_unknown_url_fails_closed_keeping_inner_text() -> None:
    text, citations = _run(
        'See <cite url="https://unknown.example">quoted words</cite>!'
    )
    assert text == "See quoted words!"
    assert citations == []


def test_unproven_span_fails_closed() -> None:
    text, citations = _run(
        f'See <cite url="{_URL}">words the source never said</cite>!'
    )
    assert text == "See words the source never said!"
    assert citations == []


def test_whitespace_and_unicode_normalization_prove_the_span() -> None:
    source = CitationSource(
        call_id="call_u",
        url="https://u.example",
        content="Skróbany   tekst źródła.",
    )
    stream_filter = _filter((source,))
    pieces = stream_filter.feed(
        '<cite url="https://u.example">Skróbany tekst źródła.</cite>'
    )
    pieces += stream_filter.finish()
    citations = [p for p in pieces if isinstance(p, ProvenCitation)]
    assert len(citations) == 1
    assert citations[0].source_start == 0
    assert citations[0].source_end == len(source.content)


@pytest.mark.parametrize(
    ("source_text", "quote"),
    (
        ("\u1100\u1161", "\uac00"),
        ("\uac00", "\u1100\u1161"),
    ),
)
def test_full_string_nfc_proves_hangul_in_both_directions(
    source_text: str,
    quote: str,
) -> None:
    source = CitationSource(
        call_id="call_h",
        url="https://u.example/hangul",
        content=source_text,
    )
    stream_filter = _filter((source,))
    pieces = (
        stream_filter.feed(f'<cite url="{source.url}">{quote}</cite>')
        + stream_filter.finish()
    )

    text = "".join(piece for piece in pieces if isinstance(piece, str))
    citations = [piece for piece in pieces if isinstance(piece, ProvenCitation)]
    assert text == quote
    assert len(citations) == 1
    assert citations[0].cited_text == quote
    assert (citations[0].source_start, citations[0].source_end) == (
        0,
        len(source_text),
    )
    assert (citations[0].output_start, citations[0].output_end) == (0, len(quote))


def test_boundary_whitespace_is_proved_without_range_contradiction() -> None:
    source = CitationSource(
        call_id="call_w",
        url="https://u.example/whitespace",
        content="  Alpha beta  ",
    )
    quote = " Alpha beta "
    stream_filter = _filter((source,))
    pieces = (
        stream_filter.feed(f'<cite url="{source.url}">{quote}</cite>')
        + stream_filter.finish()
    )

    text = "".join(piece for piece in pieces if isinstance(piece, str))
    citations = [piece for piece in pieces if isinstance(piece, ProvenCitation)]
    assert text == quote
    assert len(citations) == 1
    citation = citations[0]
    assert citation.cited_text == quote
    assert text[citation.output_start : citation.output_end] == quote
    assert (citation.source_start, citation.source_end) == (0, len(source.content))
    assert source.content[citation.source_start : citation.source_end] == source.content


def test_malformed_attribute_strips_opener_and_keeps_text() -> None:
    text, citations = _run('a <cite href="x">inner words</cite> b')
    assert text == "a inner words b"
    assert citations == []


@pytest.mark.parametrize("opener", ("<cite>", "<citeX>"))
def test_malformed_citation_name_never_leaks_markup(opener: str) -> None:
    text, citations = _run(f"a {opener}inner words</cite> b")
    assert text == "a inner words b"
    assert "<cite" not in text
    assert "</cite>" not in text
    assert citations == []


def test_nested_proven_quote_preserves_text_without_false_citation() -> None:
    raw = f'a <cite url="{_URL}">outer <cite url="{_URL}">{_QUOTE}</cite> tail</cite> b'
    text, citations = _run(raw, chunks=tuple(raw))
    assert text == f"a outer {_QUOTE} tail b"
    assert "<cite" not in text
    assert "</cite>" not in text
    assert citations == []


def test_unmatched_quote_strips_opener() -> None:
    text, citations = _run(f"a <cite url={_URL}>inner</cite> b")
    assert text == "a inner b"
    assert citations == []


def test_oversize_tag_is_stripped_without_citation() -> None:
    url = "https://long.example/" + "a" * MAX_CITE_TAG_CHARS
    text, citations = _run(f'x <cite url="{url}">quote</cite> y')
    assert text == "x quote y"
    assert citations == []


def test_oversize_span_degrades_to_plain_and_strips_orphan_close() -> None:
    long_span = "z" * (MAX_CITED_TEXT_CHARS + 10)
    text, citations = _run(f'x <cite url="{_URL}">{long_span}</cite> y')
    assert text == f"x {long_span} y"
    assert citations == []
    assert "</cite>" not in text


def test_unclosed_sentinel_flushes_plain_at_finish() -> None:
    text, citations = _run(f'x <cite url="{_URL}">{_QUOTE}')
    assert text == f"x {_QUOTE}"
    assert citations == []


def test_partial_open_prefix_at_finish_is_verbatim() -> None:
    text, citations = _run("ends with <cit")
    assert text == "ends with <cit"
    assert citations == []


def test_nonmatching_probe_flushes_verbatim() -> None:
    # Design §4.3: bytes that stop matching the `<cite ` prefix were never
    # markup and flush verbatim ("<citrus>" is plain text, not a sentinel).
    text, citations = _run("a <citrus> b <x> c")
    assert text == "a <citrus> b <x> c"
    assert citations == []


def test_empty_quote_emits_nothing_and_no_citation() -> None:
    text, citations = _run(f'a <cite url="{_URL}"></cite>b')
    assert text == "a b"
    assert citations == []


def test_citations_per_item_budget_bounds_events() -> None:
    budget = ItemCitationBudget(2)
    stream_filter = _filter(budget=budget)
    sentinel = f'<cite url="{_URL}">{_QUOTE}</cite> '
    pieces = stream_filter.feed(sentinel * 4)
    pieces += stream_filter.finish()
    citations = [p for p in pieces if isinstance(p, ProvenCitation)]
    text = "".join(p for p in pieces if isinstance(p, str))
    assert len(citations) == 2  # further sentinels degrade to plain text
    assert text == (_QUOTE + " ") * 4
    assert MAX_CITATIONS_PER_ITEM == 64


def test_holdback_is_bounded_at_every_instant() -> None:
    """§4.3 latency law: withheld chars never exceed the named bound."""

    bound = MAX_CITE_TAG_CHARS + MAX_CITED_TEXT_CHARS + MAX_CLOSE_TAG_CHARS
    sample = f'pre <cite url="{_URL}">{_QUOTE}</cite> post ' + (
        f'x <cite url="{_URL}">{"y" * 100}</cite>'
    )
    stream_filter = _filter()
    emitted = 0
    for fed, ch in enumerate(sample, start=1):
        for piece in stream_filter.feed(ch):
            if isinstance(piece, str):
                emitted += len(piece)
        assert fed - emitted <= bound


def test_search_snippet_grounds_a_citation() -> None:
    stream_filter = _filter()
    pieces = stream_filter.feed(
        '<cite url="https://s.example/hit">structural perception</cite>'
    )
    pieces += stream_filter.finish()
    citations = [p for p in pieces if isinstance(p, ProvenCitation)]
    assert len(citations) == 1
    assert citations[0].source_call_id == "call_s"


def test_filter_refuses_reuse_after_finish() -> None:
    stream_filter = _filter()
    stream_filter.finish()
    with pytest.raises(RuntimeError):
        stream_filter.feed("x")
    with pytest.raises(RuntimeError):
        stream_filter.finish()
