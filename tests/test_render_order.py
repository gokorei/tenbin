"""The order of a section, asserted by index, because a reformat can break it silently.

Every assertion in this module compares ``output.index(a) < output.index(b)``. That is
the only kind of assertion that survives the failure it exists to catch. A test that
reads a rendered report and looks at it passes on a document that has been tidied into
the wrong order -- every word still present, every sentence still true, the caveats now
underneath the number -- because it is checking that the content exists rather than where
it is. An index comparison fails on that document, and fails on it silently enough to be
worth having: nothing about the text changed, only the sequence.

So the fixtures here are built to be unambiguous about it. The claim's own wording
contains none of the phrases a renderer prints for the order markers, because a stray
"falsify" inside a statement would be found by ``index`` before the label the assertion
is about, and the test would then pass for the wrong reason. That is a constraint on the
fixture rather than on the program: a real claim is free to use those words, and a reader
should know the constraint is the test's, not the design's.

One report, three sections, and one of each kind the ordering rule has to hold for: a
figure over a whole read, a figure over a truncated read, and a refusal. A rule checked
against one of the three is a rule that has only been checked once.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any

from tenbin.claims.mechanism import Granularity
from tenbin.claims.model import Claim, ClaimKind, Denominator
from tenbin.claims.refusals import Refusal
from tenbin.claims.registry import default_registry
from tenbin.corpus.snapshot import Completeness, CorpusSnapshot, Truncation
from tenbin.measures.base import Figure, Rate, RateFigure
from tenbin.report.base import Section
from tenbin.report.document import CorpusHeader, Report
from tenbin.report.json_ import render_json, report_as_dict
from tenbin.report.markdown import WHOLE_READ_SHORT, render_markdown
from tenbin.report.ordering import Part, ordered_parts

#: Fixed, so the rendered document is byte-identical between runs. This layer has no
#: clock and the fixture must not smuggle one in through the snapshot.
READ_AT = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)

#: The two marker phrases the ordering assertions search for, named rather than written
#: inline. They are labels the renderer prints, so an assertion that finds one is
#: asserting about the *order of the renderer's lines* rather than about the wording of
#: any particular claim -- which is the point, because the order has to hold for text
#: nobody has written yet.
DOES_NOT_MEAN = "does not mean"
FALSIFY = "falsify"

#: The refusal this report carries: one the catalogue blocks on a fact a ticket would
#: deliver, so it exercises all three of a refusal's lines at once. Rebased twice --
#: off ``review-verdict-rates-by-reviewer`` when ``RJNVJK1P`` landed, and off
#: ``unanswered-decision-requests-and-their-age`` when ``AZ8T5XYS`` landed and that one
#: was narrowed to a population no ticket will supply. It was a fixture throughout: what
#: it needs is a refusal naming a ticket, not any particular one.
REFUSED_SLUG = "coverage-of-development"


def whole_read() -> CorpusSnapshot:
    """A complete read of the collection, with nothing left out to caveat."""
    return CorpusSnapshot(
        records=(),
        collection="chronicler-real",
        read_at=READ_AT,
        store_total=500,
        enumerated=500,
        completeness=Completeness.COMPLETE,
    )


def truncated_read() -> CorpusSnapshot:
    """A read stopped by the store's offset ceiling, with the hole sized and located.

    ``records_missing`` is what tells a reader every count is an undercount and
    ``offset_reached`` is where an enumeration would resume; a fixture carrying only the
    first would let a report that reported the shortfall and not the position pass.
    """
    return CorpusSnapshot(
        records=(),
        collection="chronicler-real",
        read_at=READ_AT,
        store_total=25_000,
        enumerated=10_000,
        completeness=Completeness.OFFSET_CAP,
        truncation=Truncation(
            offset_reached=10_000,
            records_missing=15_000,
            boundary_repositories=MappingProxyType({"acme/widget": 500}),
            boundary_months=MappingProxyType({"2026-09": 500}),
        ),
    )


def a_claim(slug: str = "capture-rate-over-observed-changes", **overrides: Any) -> Claim:
    """A descriptive claim whose four fields avoid every phrase the tests search for.

    Named in the module docstring: the limit and the falsifier below contain neither
    marker's own words nor the other marker's label, so an index assertion finds the
    label the renderer printed rather than a word this fixture happened to use.
    """
    fields: dict[str, Any] = {
        "slug": slug,
        "statement": "Most changes enumerated by this read carry at least one knowledge record.",
        "does_not_mean": (
            "That the capture path ran on every change; a change with no capture and a "
            "change nobody examined are the same observation from outside this corpus."
        ),
        "falsifier": (
            "A census of changes that produced no capture at all, showing the same share "
            "carrying a record over all changes the forge observed."
        ),
        "denominator": Denominator("changes enumerated by this read", 500),
        "kind": ClaimKind.descriptive,
        "granularity": Granularity.corpus,
    }
    fields.update(overrides)
    return Claim(**fields)


def a_rate_figure(snapshot: CorpusSnapshot, **overrides: Any) -> RateFigure:
    """A rate over a named population, over whichever read it is handed.

    Real rather than stubbed: the completeness half of the figure part comes from
    :meth:`Figure.rate_text`, and a stub would let these tests pass against a report
    layer that printed some other sentence in its place.
    """
    claim = a_claim()
    fields: dict[str, Any] = {
        "claim": claim,
        "snapshot": snapshot,
        "title": "Changes carrying at least one knowledge record",
        "rate": Rate(numerator=412, denominator=claim.denominator),
    }
    fields.update(overrides)
    return RateFigure(**fields)


def a_report() -> tuple[Report, tuple[Figure, Figure]]:
    """One report holding all three section kinds, and the two figures it was built from.

    Returned together because the assertions are about *relative* position: a figure is
    what the test searches for in the output, and a test that reconstructed the expected
    sentence by hand would be asserting against its own guess rather than against what
    the figure says about its own read.

    The report is over the *truncated* read, so the header caveats every figure below it
    including the one taken over a whole read. That is deliberate: it is the case a
    reader is most likely to skim past, and a rule that holds when the caveat is
    convenient is not a rule.
    """
    clean = a_rate_figure(whole_read())
    # A distinct title per figure, because the assertions below address one section at a
    # time by its heading: two figures sharing a heading would make "the section called
    # this" ambiguous, and an assertion about an ambiguous section is an assertion about
    # whichever one happened to be rendered first.
    partial = a_rate_figure(
        truncated_read(),
        claim=a_claim(slug="capture-rate-over-prefix"),
        title="Changes carrying a record, over the prefix this read reached",
    )
    refusal = default_registry().get(REFUSED_SLUG)
    report = Report(
        corpus=CorpusHeader.from_snapshot(truncated_read()),
        sections=(
            Section(claim=clean.claim, figure=clean, refusal=None),
            Section(claim=partial.claim, figure=partial, refusal=None),
            Section(claim=None, figure=None, refusal=refusal),
        ),
    )
    return report, (clean, partial)


def refusal_of(report: Report) -> Refusal:
    """The one refusal this report holds, so the assertions read without unwrapping."""
    (refused,) = [section.refusal for section in report.sections if section.refusal is not None]
    return refused


def bullet_bodies(rendered: str) -> list[str]:
    """The text of every Markdown bullet, with the continuation indent taken back off.

    A part whose text spans lines -- a rate's payload beside its completeness sentence
    does -- is rendered with the continuation lines indented, because Markdown ends a
    list item at the first unindented line. Taking the indent back off is a reversal of
    that layout and not a change of content, which is what lets this suite compare the
    two renderings word for word rather than approximately.
    """
    bodies: list[str] = []
    current: list[str] | None = None
    for line in rendered.splitlines():
        if line.startswith("- **"):
            if current is not None:
                bodies.append("\n".join(current))
            current = [line.split(":** ", 1)[1]]
        elif current is not None and line.startswith("  "):
            current.append(line.removeprefix("  "))
    if current is not None:
        bodies.append("\n".join(current))
    return bodies


def section_block(rendered: str, heading: str) -> str:
    """The rendered lines of one section, from its heading up to the next heading.

    Scoping an index assertion to its own section is what makes it independent of the
    other sections: an assertion over the whole document can pass because some *other*
    section put a marker in the right place first, which is a false negative that only
    shows up when the section that actually broke is not the one being asserted on.
    """
    _, _, rest = rendered.partition(f"## {heading}\n")
    return rest.split("\n## ", 1)[0]


def value_at(pairs: list[tuple[str, Any]], key: str) -> Any:
    """The value stored under ``key``, read from a payload held as ordered pairs."""
    return next(value for name, value in pairs if name == key)


def json_pairs(text: str) -> list[tuple[str, Any]]:
    """A JSON payload as ``(key, value)`` pairs, so its key order is assertable.

    ``object_pairs_hook`` rather than a plain ``dict`` for one reason: the assertion in
    this module is *about* order, and reading the payload as pairs makes the order the
    subject of the test instead of something the container happens to preserve.
    """
    return json.loads(text, object_pairs_hook=list)


def test_the_falsifier_appears_above_the_figure_because_a_reader_who_stops_early_should_see_it() -> (
    None
):
    """The unconventional part, asserted by index against the figure's own sentence.

    Conventional practice puts the caveats in a footnote, which assumes a reader who
    reads to the end. The reader who stops after the number is the one this program
    exists for, so the falsifier sits immediately above the figure: by the time the
    number is on the page, the test of it has already been stated. The search string is
    :meth:`Figure.rate_text`'s own return value, so this fails if the sentence is moved
    above the caveats as well as if it is moved below the number.
    """
    report, (clean, _) = a_report()
    rendered = render_markdown(report)

    assert rendered.index(clean.claim.statement) < rendered.index(clean.claim.does_not_mean)
    # The whole-read figure's completeness sentence is the short pointer to the
    # header rather than the full paragraph; the position it holds is unchanged.
    assert WHOLE_READ_SHORT in rendered
    assert clean.rate_text() not in rendered
    assert (
        rendered.index(DOES_NOT_MEAN) < rendered.index(FALSIFY) < rendered.index(WHOLE_READ_SHORT)
    )


def test_the_limit_comes_before_the_falsifier_because_the_limit_is_what_a_reader_misses_first() -> (
    None
):
    """Statement, then limit, then falsifier -- the order a reader meets them in.

    What a claim is not decides whether the number should be quoted at all, so it comes
    first: somebody who reads two lines gets the limit, and somebody who reads three gets
    the limit and the falsifier. Asserted rather than assumed because a set of parts
    would satisfy a containment check on all five while rendering them in any order.
    """
    report, (clean, _) = a_report()
    rendered = render_markdown(report)

    assert (
        rendered.index(clean.claim.statement)
        < rendered.index(DOES_NOT_MEAN)
        < rendered.index(FALSIFY)
    )


def test_the_denominator_comes_last_because_it_is_the_part_a_reader_looks_up_when_they_doubt_the_rate() -> (
    None
):
    """The denominator is a population rather than a caveat, so it follows the figure.

    It is the part a reader has in hand when they start doubting a rate -- the one they
    go and check -- so putting it above the number would spend their attention on the
    part they did not come for, and would put the population before the thing it
    qualifies.
    """
    report, (clean, _) = a_report()
    rendered = render_markdown(report)
    block = section_block(rendered, clean.title)

    assert block.index(WHOLE_READ_SHORT) < block.index(clean.claim.denominator.render_text())


def test_the_corpus_header_comes_first_because_it_decides_whether_any_number_below_it_means_what_a_reader_would_assume() -> (
    None
):
    """The header is above the first statement, and it is a section rather than a preamble.

    A preamble is what a reader skips, and this is the part that says whether the corpus
    behind the numbers is the whole corpus. Both truncation numbers are asserted, because
    a header that reported "the read stopped" without saying how far it got would leave a
    reader unable to continue the enumeration rather than restart it.

    The header's own order is asserted too, not only its position above the sections. It
    reads collection, moment, how much, whether all of it arrived, and what was left --
    and the truncation line is last because it is the sentence that qualifies everything
    above it, so a reader who reads only the first two lines has still been stopped.
    """
    report, (clean, _) = a_report()
    rendered = render_markdown(report)

    assert rendered.index("**Corpus:**") < rendered.index("**Read at:**")
    assert rendered.index("**Completeness:**") < rendered.index("**Truncation:**")
    assert rendered.index("**Truncation:**") < rendered.index(clean.claim.statement)
    assert rendered.index("**Corpus:**") < rendered.index(WHOLE_READ_SHORT)
    assert "15,000 records were not reached" in rendered
    assert "offset 10,000" in rendered


def test_a_truncated_read_says_so_in_the_figures_own_rate_text_and_not_only_in_the_header() -> None:
    """The figure over the prefix carries the truncation, in its own words, below its caveats.

    The redundancy with the header is deliberate: the header is what a reader looks up
    before trusting a number, and this is what a reader who did not look up the header
    still sees. The sentence asserted is :meth:`Figure.rate_text`'s own return value
    rather than prose written here, so a report layer that rebuilt the verdict from the
    snapshot would have to match a second implementation of it to pass -- and the
    position of that same sentence is asserted, which is the half that is easy to get
    wrong while the content stays identical.
    """
    report, (_, partial) = a_report()
    rendered = render_markdown(report)
    block = section_block(rendered, partial.title)

    assert partial.rate_text() in rendered
    assert "not reached" in partial.rate_text()
    assert block.index(partial.claim.statement) < block.index(DOES_NOT_MEAN)
    assert block.index(FALSIFY) < block.index(partial.rate_text())
    assert block.index(partial.value_text()) < block.index(partial.claim.denominator.render_text())


def test_a_refusal_renders_as_a_refusal_rather_than_an_empty_result_because_a_stub_reads_as_a_result() -> (
    None
):
    """All three of the refusal's lines are there, and none of a claim section's are.

    The three lines are asserted individually rather than by searching for the word
    "refused", because a section that rendered "no data available" under a heading would
    satisfy a weaker test than this one. And the claim-section labels are asserted
    *absent* from the refusal's own block, which is the half that matters: a refusal
    rendering an empty figure or a denominator of zero would look like a measurement
    that found nothing.
    """
    report, _ = a_report()
    refusal = refusal_of(report)
    assert refusal.missing_fact is not None
    assert refusal.unblocked_by is not None

    rendered = render_markdown(report)
    block = rendered.split(f"## {refusal.title}", 1)[1]
    assert refusal.reason in block
    assert refusal.missing_fact in block
    assert refusal.unblocked_by in block
    assert rendered.index(refusal.title) < rendered.index(refusal.reason)
    for label in ("**Statement:**", "**Figure:**", "**Denominator:**"):
        assert label not in block


def test_the_json_keys_are_the_markdowns_parts_in_the_markdowns_order_because_a_consumer_reading_in_order_meets_the_caveats_first() -> (
    None
):
    """Key order in the payload is the reading order of the page.

    ``json.dumps`` preserves insertion order, so fields arriving from
    :func:`ordered_parts` mean a consumer that walks the object in order meets the
    statement, the limit and the falsifier before it meets the figure -- and a consumer
    indexing by name still finds all of them. The payload is read as a list of
    ``(key, value)`` pairs so that the key order is a value the assertion states rather
    than an incidental property of the container it came out of.
    """
    report, _ = a_report()
    sections = value_at(json_pairs(render_json(report)), "sections")
    (section,) = (
        entry
        for entry in sections
        if value_at(entry, "slug") == "capture-rate-over-observed-changes"
    )

    assert [name for name, _ in section] == [
        "slug",
        "kind",
        "title",
        *(part.kind.value for part in ordered_parts(report.sections[0])),
        "figure_data",
        "claim_kind",
        "granularity",
        "source",
    ]


def test_both_renderings_carry_every_claims_text_identically_because_one_renderer_agreeing_with_the_other_is_the_only_check_there_is() -> (
    None
):
    """Every part is in the Markdown verbatim, and is the JSON's value under its own key.

    The whole design rests on there being one ordering in two forms, so this is the test
    that catches a renderer inventing content: a summary in the Markdown the JSON does
    not carry, or a restructured payload that quietly dropped a caveat. It is asserted
    per part rather than per document so that a failure names the field that diverged.

    The one exception is the whole-read completeness sentence, which the Markdown
    renders as the short pointer (:data:`WHOLE_READ_SHORT`) while the JSON keeps the
    full sentence: same position, abbreviated wording, with the header carrying the
    verdict it points at.
    """
    report, _ = a_report()
    expected: list[str] = [part.text for part in report.corpus.parts()]
    for section in report.sections:
        for part in ordered_parts(section):
            if (
                section.figure is not None
                and section.figure.snapshot.completeness is Completeness.COMPLETE
                and part.kind.value == "figure"
            ):
                rest = part.text.splitlines()[1:]
                expected.append("\n".join([WHOLE_READ_SHORT, *rest]))
            else:
                expected.append(part.text)
    assert bullet_bodies(render_markdown(report)) == expected

    for section, entry in zip(report.sections, report_as_dict(report)["sections"], strict=True):
        for part in ordered_parts(section):
            assert entry[part.kind.value] == part.text


def test_no_part_of_a_figure_section_is_blank_because_an_empty_line_where_a_caveat_goes_reads_as_a_claim_with_no_caveat() -> (
    None
):
    """Every part in the report carries text, in both claim sections and the refusal.

    Checked here rather than only in the type's own test because this is the point at
    which the parts become a document: a renderer that dropped one, or a section built
    over a figure whose payload was empty, would leave a gap in the rendered output that
    no index assertion would notice.
    """
    report, _ = a_report()
    parts: list[Part] = [*report.corpus.parts()]
    for section in report.sections:
        parts.extend(ordered_parts(section))

    assert len(parts) == 5 + 2 * 5 + 3
    assert all(part.text.strip() for part in parts)
