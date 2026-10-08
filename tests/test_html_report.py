"""The HTML renderer: a chart cannot be separated from the prose that bounds it.

Everything in this module is arranged around one property, and most of it is
arranged around the *absence* of its negation: **no chart in the output appears without
the claim, the negative space, the falsifier and the denominator of its own section.**
A renderer that prints a number with its caveats in a tooltip, in a ``<details>``, in a
collapsible, or in a stylesheet that hides them at narrow widths has separated them,
and this program exists because that separation is where the discipline goes. So the
first test here walks the parsed document rather than searching the source, finds every
``<svg>``, and requires the four bound fields to be inside the same enclosing element
-- which is the only assertion here that would fail if the caveats were moved into
somewhere else on the page while still being present in the file.

**Assertions are made against the parsed document wherever the claim is structural.**
Substring search over the source is fine for "this sentence is on the page" and useless
for "this chart is inside this claim", because a substring cannot tell containment from
coincidence. So the charts are located as elements, the geometry is read off the
``<rect>`` and ``<line>`` attributes, and the containment is checked by walking parents.

**No assertion here reaches for a private constant of the module under test.** The
tests that need to know what a bar's length means work it out from the markup: the
axis of a rate is the distance between the 0% and 100% gridlines, and the two
distributions whose bars must be identical are made identical by construction rather
than by being compared against a number written down in the implementation. A test
that asserts against a private constant tests the constant; these assert the property.

**The fixtures use the real figure classes and the real
:func:`~tenbin.report.ordering.ordered_parts`, never a stub.** The completeness
sentence a chart carries is :meth:`~tenbin.measures.base.Figure.rate_text`'s own
return value, so a test that built the sentence itself would pass against a renderer
that mangled the real one.

The palettes are checked by computing WCAG contrast rather than by asserting that a
string is present. "There is a dark-mode block" is a claim about the file; "every
colour in both schemes clears the contrast threshold against both backgrounds" is a
claim about the legibility, and it is the second one that matters.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from html.parser import HTMLParser
from types import MappingProxyType
from typing import Any

from tenbin.claims.mechanism import Granularity
from tenbin.claims.model import Claim, ClaimKind, Denominator
from tenbin.claims.refusals import Refusal, refusal_for_measure
from tenbin.corpus.snapshot import Completeness, CorpusSnapshot, Truncation
from tenbin.measures.base import (
    CountFigure,
    DistributionFigure,
    Figure,
    FigureGroup,
    Finding,
    Rate,
    RateFigure,
)
from tenbin.report.base import Section
from tenbin.report.document import CorpusHeader, Report
from tenbin.report.html_ import (
    ABSENT_DISTRIBUTION,
    EXCLUDED_HEADING,
    FINDING_BANNER,
    REFUSAL_BANNER,
    render_html,
)
from tenbin.report.ordering import ordered_parts

#: A fixed moment, so a rendered document is byte-identical between runs. This layer
#: has no clock and a fixture that smuggled one in would make the determinism test
#: pass for the wrong reason.
READ_AT = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)

#: Elements that never have a closing tag, so the tree builder below knows not to wait
#: for one. Listed rather than special-cased one at a time because the alternative is
#: a stack that grows without bound and a tree in which everything after the ``<meta>``
#: is a child of the document head.
_VOID_ELEMENTS = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)

#: The smallest contrast ratio a colour carrying text has to clear against a
#: background, and the one a colour carrying a mark has to clear. The two numbers are
#: WCAG 2.1 AA and 1.4.11 respectively, and they are different because a bar is a
#: shape a reader compares and a sentence is a shape a reader reads. Using the text
#: threshold for bars as well would be stricter than the standard and would force a
#: palette this document has no reason to reject; using the mark threshold for text
#: would be looser than the standard and would let a caption be unreadable.
TEXT_CONTRAST = 4.5
MARK_CONTRAST = 3.0

#: The tokens that carry words rather than shapes, and therefore owe the text
#: threshold. Named here because the classification is a judgement and a test that
#: hides its judgements is a test that cannot be argued with: ``--ink`` and ``--muted``
#: are prose, and ``--bar``, ``--excluded``, ``--accent`` and ``--refusal`` are used as
#: a ``color:`` on a banner or a slug as well as a fill on a bar, so they are held to
#: the text threshold even where they are only a fill.
TEXT_TOKENS = ("--ink", "--muted", "--bar", "--excluded", "--accent", "--refusal")

#: The two tokens a foreground is measured against, and the two tokens that are not
#: themselves measured. A background compared against itself is a contrast ratio of
#: exactly 1.0 and would fail every threshold, so excluding them is not a loosening:
#: what is being checked is that every colour the renderer puts *in front of* something
#: can be read against the two things it is put in front of.
BACKGROUNDS = ("--bg", "--surface")

#: Tokens that are never drawn on a page background, and the one thing they are drawn
#: on. ``--stripe`` is the line inside the hatch pattern and the hatch is only ever
#: painted on top of an excluded bar, so measuring the stripe against ``--bg`` would be
#: measuring a colour against a surface it is never printed on -- and in the light
#: scheme it would fail, correctly, for a colour nobody can see. It is measured against
#: what it is drawn on by
#: :func:`test_the_hatch_texture_is_visible_against_the_colour_it_is_drawn_on`.
TEXTURE_TOKENS = ("--stripe",)


# --------------------------------------------------------------------------- fixtures


def whole_read(*, enumerated: int = 500) -> CorpusSnapshot:
    """A complete read of the collection, so a completeness caveat has nothing to add.

    ``records`` is empty on purpose: the report layer reads the header and the figures
    and never the records, so a fixture holding five hundred of them would be asserting
    nothing that a fixture holding none does not.
    """
    return CorpusSnapshot(
        records=(),
        collection="chronicler-real",
        read_at=READ_AT,
        store_total=enumerated,
        enumerated=enumerated,
        completeness=Completeness.COMPLETE,
    )


def truncated_read() -> CorpusSnapshot:
    """A read stopped by the store's offset ceiling, with the hole sized and located.

    Used for the rate figure, because a rate over a prefix and a rate over a whole
    corpus are the same picture with one sentence different, and that sentence is the
    one this renderer has to place above the bar rather than in a footnote.
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


def a_claim(
    slug: str,
    *,
    statement: str,
    does_not_mean: str,
    falsifier: str,
    population: str,
    size: int,
) -> Claim:
    """A descriptive claim, with all four fields distinct and none of them shared.

    Distinct per section on purpose. The containment test in this module is only worth
    running if a chart could not be satisfied by some *other* section's claim, and two
    sections carrying the same statement would let it be satisfied by either.
    """
    return Claim(
        slug=slug,
        statement=statement,
        does_not_mean=does_not_mean,
        falsifier=falsifier,
        denominator=Denominator(population, size),
        kind=ClaimKind.descriptive,
        granularity=Granularity.corpus,
    )


def model_claim() -> Claim:
    """The claim behind the distribution, and the one with exclusions in it."""
    return a_claim(
        "records-by-stated-model",
        statement="Records that stated which model answered, counted by that model.",
        does_not_mean=(
            "That the records naming no model are among the buckets; they are not, and "
            "they are counted separately below the bars."
        ),
        falsifier=(
            "A walk of the documents Kojutsu declined to write, showing a different "
            "share of records naming a model than the share below."
        ),
        population="records enumerated by this read",
        size=500,
    )


def model_distribution(
    *,
    claim: Claim | None = None,
    values: Any = None,
    excluded: Any = None,
    snapshot: CorpusSnapshot | None = None,
    title: str = "Records by stated model",
) -> DistributionFigure:
    """A distribution with three buckets and one named exclusion.

    Both mappings have defaults passed as ``None`` rather than as mutable objects in
    the signature, because a default argument that is a dict is a dict every call
    shares; the body is where the shape of a normal distribution lives.
    """
    return DistributionFigure(
        claim=model_claim() if claim is None else claim,
        snapshot=whole_read() if snapshot is None else snapshot,
        title=title,
        values={"opus": 212, "sonnet": 151, "gpt-5": 100} if values is None else values,
        excluded={"records_with_no_stated_model": 37} if excluded is None else excluded,
    )


def capture_rate_figure(
    *, numerator: int = 412, snapshot: CorpusSnapshot | None = None
) -> RateFigure:
    """A rate over the changes this read enumerated, over a prefix of the corpus."""
    claim = a_claim(
        "changes-carrying-a-record",
        statement="Most changes this read enumerated carry at least one knowledge record.",
        does_not_mean="That the capture path ran on every change the forge saw.",
        falsifier=(
            "A census of changes that produced no capture at all, holding a different "
            "share of recorded changes than the fraction above."
        ),
        population="changes enumerated by this read",
        size=500,
    )
    return RateFigure(
        claim=claim,
        snapshot=truncated_read() if snapshot is None else snapshot,
        title="Changes carrying a record",
        rate=Rate(numerator=numerator, denominator=claim.denominator),
    )


def clock_skew_finding() -> Finding:
    """A finding: something observed that does not look like an honest system produces."""
    claim = a_claim(
        "capture-timestamps-run-backwards",
        statement="Some records carry a created_at earlier than the change they describe.",
        does_not_mean="That the store lost them; they are present and they are out of order.",
        falsifier="A second read of the same collection with no such record in it.",
        population="records enumerated by this read",
        size=500,
    )
    return Finding(
        claim=claim,
        snapshot=whole_read(),
        title="Capture timestamps run backwards",
        what_was_observed="17 records carry a created_at earlier than the change they cite.",
        suspected_cause="A writer read the clock twice and stored the earlier reading.",
        where="records from acme/widget between 2026-08-01 and 2026-08-14",
        cannot_confirm="Tenbin reads the timestamps; it does not run the writer.",
    )


def coverage_refusal() -> Refusal:
    """A refusal naming a missing fact and a ticket, so all three of its lines exist."""
    return refusal_for_measure(
        "Coverage of development",
        reason=(
            "Kojutsu writes only on success, so a change nobody examined and a change "
            "that produced a capture are the same observation from outside this corpus."
        ),
        missing_fact="a census of changes that produced no capture",
        unblocked_by="EB6FE5ZP",
    )


def a_section(figure: Figure) -> Section:
    """The section a figure belongs in, which is the only way one can be rendered."""
    return Section(claim=figure.claim, figure=figure, refusal=None)


def a_refusal_section(refusal: Refusal | None = None) -> Section:
    """The section a refusal belongs in, which carries no claim and no figure."""
    return Section(
        claim=None, figure=None, refusal=coverage_refusal() if refusal is None else refusal
    )


def a_report(sections: tuple[Section, ...], *, snapshot: CorpusSnapshot | None = None) -> Report:
    """A report over a header built from ``snapshot``, holding the sections given.

    The header is copied out of the snapshot by
    :meth:`~tenbin.report.document.CorpusHeader.from_snapshot` rather than written
    out, for the reason that module gives: five numbers and a moment are five chances to
    transcribe one of them wrongly with no test watching.
    """
    read = truncated_read() if snapshot is None else snapshot
    return Report(corpus=CorpusHeader.from_snapshot(read), sections=sections)


def one_of_each() -> tuple[Report, tuple[DistributionFigure, RateFigure, Finding]]:
    """A report holding every shape the renderer has to get right, and its figures.

    All four section kinds in one document rather than one per test, because the
    properties under test are properties of the *document*: that a refusal's
    ``<section>`` contains no chart, that a finding's contains no chart, and that a
    distribution's contains one. Split across documents those are three facts; in one
    document they are also the fact that the renderer did not confuse them with each
    other.
    """
    distribution = model_distribution()
    rate = capture_rate_figure()
    finding = clock_skew_finding()
    report = a_report(
        (
            a_section(distribution),
            a_section(rate),
            a_section(finding),
            a_refusal_section(),
        )
    )
    return report, (distribution, rate, finding)


# ------------------------------------------------------------------ document parsing


@dataclass
class _Node:
    """One element or one run of text, with a parent, so containment is walkable."""

    tag: str
    attrs: dict[str, str] = field(default_factory=dict)
    children: list[_Node | str] = field(default_factory=list)
    parent: _Node | None = None

    def elements(self) -> list[_Node]:
        """Every descendant element, in document order, at any depth."""
        found: list[_Node] = []
        for child in self.children:
            if isinstance(child, _Node):
                found.append(child)
                found.extend(child.elements())
        return found

    def find_all(self, tag: str) -> list[_Node]:
        """Every descendant element with this tag, in document order."""
        return [node for node in self.elements() if node.tag == tag]

    def find(self, tag: str) -> _Node:
        """The first descendant with this tag. Named for a tag that is known to be there."""
        (found,) = self.find_all(tag)
        return found

    def classed(self, css_class: str) -> list[_Node]:
        """Every descendant whose ``class`` attribute holds ``css_class`` as a word.

        Split on whitespace rather than compared whole, because the renderer writes
        ``class="row row-excluded"`` and an equality test against one of those halves
        would silently find nothing.
        """
        return [node for node in self.elements() if css_class in node.attr("class").split()]

    def attr(self, name: str) -> str:
        """One attribute's value, looked up without regard to case, or ``""`` if absent.

        :mod:`html.parser` lowercases every attribute name it reads, so indexing
        ``attrs`` directly would make an assertion about the parser rather than about
        the document: ``viewBox`` is spelled in camel case in the file because SVG
        fixes that spelling, and it arrives here as ``viewbox``. Looking up without
        regard to case keeps the assertion about what the document says.

        Absent attributes read as the empty string rather than raising, because the
        searches here ask "which elements carry this class" and a ``<text>`` with no
        class at all is an answer, not a failure. Use :meth:`has_attr` when the
        presence of an attribute is the thing being asserted.
        """
        return self.attrs.get(name.casefold(), "")

    def has_attr(self, name: str) -> bool:
        """Whether this element carries an attribute at all."""
        return name.casefold() in self.attrs

    def text(self) -> str:
        """Every string in this subtree, whitespace-normalised, as one string.

        Normalised so that a claim field spanning two source lines is comparable with
        the same field rendered as two paragraphs, and because the source line breaks
        of the generated file carry no information a reader could be reading.
        """
        parts: list[str] = []
        for child in self.children:
            if isinstance(child, _Node):
                parts.append(child.text())
            else:
                parts.append(child)
        return " ".join(" ".join(parts).split())

    def enclosing(self, tag: str) -> _Node | None:
        """The nearest ancestor with this tag, or ``None`` at the top of the tree."""
        node = self.parent
        while node is not None:
            if node.tag == tag:
                return node
            node = node.parent
        return None

    def number(self, attribute: str) -> float:
        """One numeric attribute of this element, read as a number."""
        return float(self.attr(attribute))


class _Tree(HTMLParser):
    """Builds the element tree the assertions walk.

    ``convert_charrefs`` is on so that the text compared against a claim is the
    claim's own words rather than their escaped spelling. Whether the escaping happened
    is a separate question with its own test, and answering it here as well would make
    every containment assertion in this module also an escaping assertion -- which is
    the kind of coupling that makes a failure hard to read.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = _Node("[document]")
        self._stack: list[_Node] = [self.root]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        node = _Node(tag, {name: value or "" for name, value in attrs})
        node.parent = self._stack[-1]
        self._stack[-1].children.append(node)
        if tag not in _VOID_ELEMENTS:
            self._stack.append(node)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in _VOID_ELEMENTS:
            self._stack.pop()

    def handle_endtag(self, tag: str) -> None:
        for index in range(len(self._stack) - 1, 0, -1):
            if self._stack[index].tag == tag:
                del self._stack[index:]
                return

    def handle_data(self, data: str) -> None:
        self._stack[-1].children.append(data)


def parse(rendered: str) -> _Node:
    """The rendered document as a tree, so an assertion can ask about containment."""
    tree = _Tree()
    tree.feed(rendered)
    tree.close()
    return tree.root


def charts(document: _Node) -> list[_Node]:
    """Every chart on the page, as the ``<figure>`` that wraps each drawing."""
    return [node for node in document.elements() if node.tag == "figure" and node.find_all("svg")]


def sections(document: _Node) -> list[_Node]:
    """Every ``<section>``, header and claim and refusal alike."""
    return document.find_all("section")


def bound_fields(claim: Claim) -> tuple[str, ...]:
    """The four fields a chart is not allowed to appear without, in page order.

    Taken from the claim object rather than written out, because these are the strings
    :func:`~tenbin.report.ordering.ordered_parts` renders above and below the chart,
    and a test asserting against its own transcription of them would pass against a
    renderer that dropped the real ones.
    """
    return (
        claim.statement,
        claim.does_not_mean,
        claim.falsifier,
        claim.denominator.render_text(),
    )


# ------------------------------------------------------------------------- the tests


def test_a_figures_chart_and_its_four_bound_fields_are_all_in_the_output() -> None:
    """The basic contract, per section, before the harder assertions about it.

    Stated first and stated plainly because everything else in this module is a
    sharpening of it: a section carrying a figure has a chart, and the statement, the
    negative space, the falsifier and the denominator are all present. The four are
    taken from the claim the section was built with, so a renderer that rendered its own
    paraphrase of a caveat would not satisfy this.
    """
    report, (distribution, rate, finding) = one_of_each()
    document = parse(render_html(report))

    for figure in (distribution, rate, finding):
        section = _section_for(document, figure.claim.slug)
        assert section is not None, f"no section was rendered for {figure.claim.slug}"
        text = section.text()
        for field_text in bound_fields(figure.claim):
            assert field_text in text
    for figure in (distribution, rate):
        section = _section_for(document, figure.claim.slug)
        assert section is not None
        assert section.find_all("svg"), f"{figure.claim.slug} has no chart"


def test_no_chart_in_the_document_appears_without_the_claims_of_its_own_section() -> None:
    """The absence this whole renderer exists to make impossible.

    Every ``<svg>`` on the page is located as an element, its nearest enclosing
    ``<section>`` is found by walking parents, and the four bound fields of that
    section's claim are required to be inside that same element. A search over the
    whole file would pass on a document with the caveats moved to a footer, and a
    search for the sentence alone would pass on a document that put a different
    section's caveats next to this chart; neither of those is a chart whose bound
    travels with it, and neither would fail here.

    The count is asserted first so that a renderer which emitted no chart at all cannot
    pass the rest of the test vacuously -- an absence test that passes because the thing
    it forbids never appeared is a test that has stopped testing.
    """
    report, figures = one_of_each()
    document = parse(render_html(report))
    drawings = document.find_all("svg")

    assert len(drawings) == 2, "expected a chart for the distribution and one for the rate"
    for drawing in drawings:
        section = drawing.enclosing("section")
        assert section is not None, "a chart is not inside a section at all"
        slug = section.attr("data-slug")
        claim = next(figure.claim for figure in figures if figure.claim.slug == slug)
        text = section.text()
        for field_text in bound_fields(claim):
            assert field_text in text, f"the chart in {slug} lost {field_text[:40]!r}"


def test_a_chart_carries_the_bound_in_its_own_description_because_a_reader_may_only_meet_the_chart() -> (
    None
):
    """The ``<desc>`` repeats the four fields, so the bound survives losing the page.

    A reader who reaches a chart through an image list, or whose assistive technology
    announces the graphic and moves on, never sees the paragraphs beside it. A
    description carrying only the numbers would be a chart that had lost its bound in
    the one place where losing it is invisible -- so the description repeats the
    statement, the limit, the falsifier and the denominator, and the test asserts the
    whole of each rather than the presence of the element.
    """
    report, (distribution, _, _) = one_of_each()
    document = parse(render_html(report))
    section = _section_for(document, distribution.claim.slug)
    assert section is not None

    (chart,) = charts(section)
    description = chart.find("desc").text()
    for field_text in bound_fields(distribution.claim):
        assert field_text in description
    assert distribution.rate_text() in description


def test_the_denominator_comes_after_the_chart_and_the_falsifier_before_it() -> None:
    """The one ordering holds, with a picture in the middle of it.

    :mod:`tenbin.report.ordering` puts the falsifier above the figure because a reader
    who stops early should already have been told what would have made the number
    wrong. That argument is about a *sentence* above a number, and it is at least as
    strong about a bar chart, which is easier to take away than a number is. The
    ordering is inherited rather than reimplemented, so this test is really a check
    that the chart was emitted at the position the ordering gave it rather than
    wherever the layout would have put it.
    """
    report, (distribution, _, _) = one_of_each()
    document = parse(render_html(report))
    section = _section_for(document, distribution.claim.slug)
    assert section is not None
    parts = section.find_all("div")
    kinds = [node.attr("data-part") for node in parts if node.has_attr("data-part")]

    assert kinds == [
        "statement",
        "does_not_mean",
        "falsifier",
        "figure",
        "denominator",
    ]
    figure_part = next(
        node for node in parts if node.has_attr("data-part") and node.attr("data-part") == "figure"
    )
    assert figure_part.find_all("svg"), "the chart is not inside the figure part"


def test_an_excluded_bucket_is_rendered_apart_from_the_values_and_never_as_an_other_bucket() -> (
    None
):
    """The exclusions are their own group, with their own labels and their own counts.

    Rule one, and the reason the whole package exists: a bucket that did not occur and a
    bucket that was filtered out are the same integer in ``values`` and completely
    different facts. The assertion reads the chart's own row groups and compares them
    to the two mappings, so it fails on a renderer that dropped the exclusions, merged
    them into a bucket, or counted them inside a value -- and it compares them as two
    lists, so a renderer that happened to sum to the right total with the wrong labels
    also fails.
    """
    report, (distribution, _, _) = one_of_each()
    document = parse(render_html(report))
    section = _section_for(document, distribution.claim.slug)
    assert section is not None
    (chart,) = charts(section)

    counted = _rows(chart, "row-excluded", excluded=False)
    excluded = _rows(chart, "row-excluded", excluded=True)

    assert counted == [(label, count) for label, count in distribution.values.items()]
    assert excluded == [(label, count) for label, count in distribution.excluded.items()]
    assert [label for label, _ in excluded] not in [[label] for label, _ in counted]
    assert "other" not in chart.text().casefold()


def test_an_excluded_record_is_not_told_from_a_counted_one_by_colour_alone() -> None:
    """The excluded bars are hatched, and the hatching is named in words on the page.

    A second colour is not a distinction a reader can rely on: it fails in a monochrome
    print, on a bad projector, and for the roughly one reader in twelve who cannot
    separate the hues. So the excluded bars carry a texture *and* a heading that names
    the condition, and both are asserted -- the ``<pattern>`` for the texture and the
    heading text for the naming, because either alone would be a distinction made in one
    channel only.
    """
    report, (distribution, _, _) = one_of_each()
    document = parse(render_html(report))
    section = _section_for(document, distribution.claim.slug)
    assert section is not None
    (chart,) = charts(section)

    patterns = chart.find_all("pattern")
    hatched = [rect for rect in chart.find_all("rect") if rect.attr("class") == "hatch"]
    assert patterns, "an excluded bar is drawn without a texture to distinguish it"
    assert len(hatched) == len(distribution.excluded)
    for rect in hatched:
        assert rect.attr("fill").startswith("url(#")
        assert rect.attr("fill").removeprefix("url(#").removesuffix(")") in {
            pattern.attr("id") for pattern in patterns
        }
    assert EXCLUDED_HEADING in section.text()


def test_an_excluded_record_carries_its_own_count_and_the_two_totals_are_never_added_together() -> (
    None
):
    """The counted total and the excluded total are reported as two numbers.

    A chart that printed one total covering both would be asserting that the two add up
    to the population, and they only do if the measure is internally consistent -- which
    is a claim about the measure rather than something the renderer can assume. Both
    numbers are therefore printed, separately, in the caption that says what the bars
    are measured against.
    """
    report, (distribution, _, _) = one_of_each()
    document = parse(render_html(report))
    section = _section_for(document, distribution.claim.slug)
    assert section is not None
    (chart,) = charts(section)
    caption = chart.find("figcaption").text()

    assert f"Counted: {distribution.counted:,}." in caption
    assert f"Excluded: {distribution.excluded_total:,}." in caption
    assert sum(distribution.values.values()) + sum(distribution.excluded.values()) == (
        distribution.counted + distribution.excluded_total
    )


def test_a_distribution_says_on_the_page_what_its_bar_length_is_relative_to() -> None:
    """Rule three, in words, on the page rather than in the geometry.

    Bars proportional to the largest bucket and bars proportional to the denominator
    are the same picture to within a scale factor, so the reader cannot tell from the
    bars which one they are looking at. That is why the caption names the largest
    bucket and says explicitly that the length is not relative to the denominator: the
    sentence is the only part of the chart that can carry the difference, and a scale
    the reader has to assume is a scale the reader will get wrong.
    """
    report, (distribution, _, _) = one_of_each()
    document = parse(render_html(report))
    section = _section_for(document, distribution.claim.slug)
    assert section is not None
    (chart,) = charts(section)
    caption = chart.find("figcaption").text()

    assert "largest bucket" in caption
    assert "not to the denominator" in caption
    assert (
        f"{max(max(distribution.values.values()), max(distribution.excluded.values())):,}"
        in caption
    )


def test_two_distributions_with_the_same_buckets_and_different_populations_draw_identical_bars() -> (
    None
):
    """Rule three's geometry, tested by making the two scalings disagree on purpose.

    Both scalings are linear, so no single bar can tell them apart: a bar drawn at 40%
    of the track is the same bar whether the track is the largest bucket or the
    denominator. The only way to observe the difference is to change the denominator and
    leave the buckets alone, which is what these two figures do. If the bars move, they
    are scaled to the population -- which is the claim this program refuses to make,
    because the buckets and the exclusions are two different totals.
    """
    buckets = {"opus": 212, "sonnet": 151}
    exclusions = {"records_with_no_stated_model": 37}
    narrow = model_distribution(
        claim=model_claim(),
        values=buckets,
        excluded=exclusions,
        title="Records by stated model, over the records enumerated",
    )
    wide_claim = a_claim(
        "records-by-stated-model-wide",
        statement="Records that stated which model answered, over a population four times larger.",
        does_not_mean="That the records naming no model are among the buckets.",
        falsifier="A wider walk of the same collection holding a different share.",
        population="records enumerated by this read, and the ones the offset cap left behind",
        size=2_000,
    )
    wide = model_distribution(
        claim=wide_claim,
        values=buckets,
        excluded=exclusions,
        title="Records by stated model, over a wider population",
    )
    report = a_report((a_section(narrow), a_section(wide)))
    document = parse(render_html(report))

    narrow_chart = charts(_section_for(document, narrow.claim.slug) or document)[0]
    wide_chart = charts(_section_for(document, wide.claim.slug) or document)[0]

    assert _bar_lengths(narrow_chart) == _bar_lengths(wide_chart)


def test_a_distribution_with_every_bucket_at_zero_is_empty_because_that_is_what_a_reader_means() -> (
    None
):
    """An empty distribution is empty whether or not it has keys.

    :func:`tenbin.measures.base.duration_buckets` includes every rung of its ladder at
    zero so that a reader comparing two runs can see a rung was empty rather than infer
    it from a gap. So "no records" arrives as seven labelled buckets holding zero, not
    as an empty mapping, and a renderer that tested for the presence of keys would draw
    an axis for it. This is the case the emptiness check has to be written against the
    *counts* rather than against the shape of the mapping.
    """
    rungs = {f"under {n} hours": 0 for n in (1, 4, 24)}
    figure = model_distribution(values=rungs, excluded={}, title="Time to first review")
    report = a_report((a_section(figure),))
    document = parse(render_html(report))
    section = _section_for(document, figure.claim.slug)
    assert section is not None

    assert not section.find_all("svg"), "an empty distribution drew an axis"
    assert ABSENT_DISTRIBUTION in section.text()


def test_a_distribution_with_no_buckets_at_all_says_the_axis_is_absent_rather_than_drawing_a_zero() -> (
    None
):
    """Rule two, and the failure it names: a blank chart is a hole, this is a fact.

    Asserted in both directions. There is no ``<svg>`` at all, so there is no axis, no
    bar and nothing a reader could mistake for a measurement; and the sentence is
    present, so the section is not a heading with nothing under it either. A renderer
    that drew an empty axis here would satisfy a weaker version of this test, and an
    empty axis is exactly the shape that reads as a result.
    """
    figure = model_distribution(values={}, excluded={}, title="Time to first review")
    report = a_report((a_section(figure),))
    rendered = render_html(report)
    document = parse(rendered)
    section = _section_for(document, figure.claim.slug)
    assert section is not None

    assert not section.find_all("svg")
    assert not section.find_all("rect")
    assert not section.find_all("line")
    assert ABSENT_DISTRIBUTION in section.text()
    assert figure.claim.denominator.render_text() in section.text()


def test_a_rate_is_drawn_against_a_fixed_zero_to_one_hundred_axis_and_not_scaled_to_fit() -> None:
    """Rule four's geometry, read off the gridlines rather than off a constant.

    The bar's width is compared with the distance between the 0% and the 100% gridline,
    both of which are in the markup. So the assertion holds for a rate of 82.4% *and*
    for a rate of 1%, and a renderer that scaled every bar to fill the space it had
    would fail both: its 1% bar would be indistinguishable from its 82% one, which is the
    defect the fixed axis exists to prevent.
    """
    report = a_report((a_section(capture_rate_figure(numerator=412)),))
    document = parse(render_html(report))
    (chart,) = charts(document)
    assert _rate_is_drawn_on_a_fixed_axis(chart, 412 / 500)

    small = a_report((a_section(capture_rate_figure(numerator=5)),))
    (tiny_chart,) = charts(parse(render_html(small)))
    assert _rate_is_drawn_on_a_fixed_axis(tiny_chart, 5 / 500)
    assert _rate_is_drawn_on_a_fixed_axis(tiny_chart, 0.01)


def test_a_rate_prints_its_fraction_with_its_population_rather_than_a_bare_percentage() -> None:
    """The value on the page is a fraction over a named population.

    ``82.4%`` on its own reads as general, and a percentage with no population beside
    it is the exact shape of number this program exists to refuse. So the value is
    printed as ``412 of 500 = 82.4%`` inside the drawing *and* the population is named
    in the caption, and the fraction carries the same words
    :meth:`~tenbin.measures.base.Rate.rate_text` uses rather than a second phrasing.
    """
    figure = capture_rate_figure(numerator=412)
    report = a_report((a_section(figure),))
    document = parse(render_html(report))
    (chart,) = charts(document)
    section = document.find_all("section")[1]

    assert "412 of 500 = 82.4%" in chart.text()
    assert figure.rate.rate_text() in section.text()
    assert figure.rate.denominator.render_text() in chart.find("figcaption").text()


def test_a_finding_renders_as_a_callout_rather_than_a_chart() -> None:
    """Rule five: a claim that something is wrong is not a quantity to plot.

    A finding has no population for a bar to be a fraction of, and a bar drawn for one
    would be inventing a denominator to scale it to. So the section carries a callout,
    no drawing at all, and every one of the finding's own fields -- including the
    ``cannot_confirm`` sentence, which is the one that keeps an observation from
    becoming a diagnosis.
    """
    report, (_, _, finding) = one_of_each()
    document = parse(render_html(report))
    section = _section_for(document, finding.claim.slug)
    assert section is not None

    assert not section.find_all("svg")
    (callout,) = section.find_all("aside")
    assert callout.attr("role") == "note"
    callout_text = callout.text()
    assert FINDING_BANNER in callout_text
    for value in (
        finding.what_was_observed,
        finding.where,
        finding.suspected_cause,
        finding.cannot_confirm,
    ):
        assert value is not None
        assert value in callout_text


def test_a_refusal_renders_as_a_refusal_and_never_as_an_empty_chart() -> None:
    """Rule six, asserted in both directions: its three lines, and no chart.

    The absence matters more than the presence. A refusal rendered as an empty chart
    looks like a measure that found nothing, and rendered as a zero it is a number; both
    are the failure the ordering module's docstring refuses by name. So the section
    carries the reason, the missing fact and the unblocking ticket, and carries no
    ``<svg>``, no ``<rect>`` and none of a claim section's labels.
    """
    report, _ = one_of_each()
    document = parse(render_html(report))
    refusal = coverage_refusal()
    section = _section_for(document, refusal.claim_slug)
    assert section is not None
    text = section.text()

    assert REFUSAL_BANNER in text
    assert refusal.title in text
    assert refusal.reason in text
    assert refusal.missing_fact is not None
    assert refusal.missing_fact in text
    assert refusal.unblocked_by == "EB6FE5ZP"
    assert refusal.unblocked_by in text
    assert not section.find_all("svg")
    assert not section.find_all("rect")
    assert not section.find_all("line")
    for label in ("Statement", "What it does not mean", "Denominator", "Figure"):
        assert label not in text


def test_a_refusal_whose_absence_of_a_ticket_is_the_answer_says_so_rather_than_saying_nothing() -> (
    None
):
    """``unblocked_by is None`` is an answer, and it is rendered as one.

    A reader who cannot tell "nothing will unblock this" from "nobody recorded whether
    anything will" is a reader waiting on a ticket that was never going to be filed.
    The wording is not this renderer's to choose -- it is the constant
    :mod:`tenbin.claims.refusals` writes, and this test exists to catch a renderer
    that dropped the line.
    """
    from tenbin.claims.refusals import NO_TICKET_UNBLOCKS_THIS

    refusal = refusal_for_measure(
        "Whether agentic development is more effective",
        reason="The corpus holds no assignment mechanism, so the difference is uninterpretable.",
        missing_fact="a mechanism that decided who received which ticket",
        unblocked_by=None,
    )
    document = parse(render_html(a_report((a_refusal_section(refusal),))))
    section = _section_for(document, refusal.claim_slug)
    assert section is not None

    assert NO_TICKET_UNBLOCKS_THIS in section.text()
    assert not section.find_all("svg")


def test_a_count_is_left_as_a_number_because_a_single_bar_would_need_an_invented_denominator() -> (
    None
):
    """A count gets no chart, and that is a decision rather than an omission.

    One bar has nothing to be measured against. The only way to draw it honestly would
    be to scale it to something -- the store's count, the enumerated count -- and both of
    those would be a claim about what the count is a fraction of that the figure does
    not make. So the count renders as the number it is, inside the same figure part,
    with the completeness sentence above it, and the test asserts the absence of a
    drawing so that a later change to scale it has to argue with this.
    """
    claim = model_claim()
    figure = CountFigure(claim=claim, snapshot=whole_read(), title="Records enumerated", value=412)
    document = parse(render_html(a_report((a_section(figure),))))
    section = _section_for(document, claim.slug)
    assert section is not None

    assert not section.find_all("svg")
    assert figure.value_text() in section.text()
    assert figure.rate_text() in section.text()


def test_a_group_draws_a_chart_for_each_member_inside_the_one_claim_they_share() -> None:
    """A measure that answers in several parts still gets a chart per part.

    A ``FigureGroup`` shares one claim by construction, so a member's chart is bound by
    the same four parts as any other chart on the page. The test asserts the containment
    for every member rather than assuming it follows from the group case, because the
    group is the one path where a chart is emitted from something other than a
    ``Section``'s own figure and that is exactly where the binding could come apart.
    """
    distribution = model_distribution(title="Records by stated model")
    rate = capture_rate_figure(numerator=412)
    group = FigureGroup(
        claim=rate.claim,
        snapshot=truncated_read(),
        title="Coverage of review verdicts",
        figures=(distribution, rate),
    )
    document = parse(render_html(a_report((a_section(group),))))
    section = _section_for(document, group.claim.slug)
    assert section is not None

    assert len(section.find_all("svg")) == 2
    assert {node.text() for node in section.find_all("h3")} == {distribution.title, rate.title}
    for field_text in bound_fields(group.claim):
        assert field_text in section.text()


def test_a_bucket_holding_zero_is_shown_with_its_name_and_its_count_and_no_bar() -> None:
    """A rung that was empty is on the page, and that is not the same as a gap.

    :func:`tenbin.measures.base.duration_buckets` includes every rung of its ladder at
    zero precisely so a reader comparing two runs can see that a rung *was* empty rather
    than infer it from a missing row. So the row is drawn -- label and count, no bar --
    and the test asserts the row exists and that the bar does not. A renderer that
    dropped zero rows would look identical for a distribution with no zero in it, and
    would silently destroy the one thing the ladder is for.
    """
    figure = model_distribution(values={"opus": 212, "gpt-5": 0}, excluded={})
    document = parse(render_html(a_report((a_section(figure),))))
    section = _section_for(document, figure.claim.slug)
    assert section is not None
    (chart,) = charts(section)

    assert _rows(chart, "row-excluded", excluded=False) == [("opus", 212), ("gpt-5", 0)]
    row = next(group for group in chart.classed("row") if "gpt-5" in group.text())
    assert not row.find_all("rect"), "a bucket holding zero drew a bar"
    assert "gpt-5 0" in " ".join(chart.text().split())


def test_a_distribution_whose_buckets_are_all_zero_still_draws_its_exclusions() -> None:
    """Nothing counted and something excluded is a fact, and it is not an empty axis.

    The case that distinguishes rule one from rule two. Rule two covers a distribution
    with nothing in it at all; this one has records -- the ones the measure set aside --
    and they are exactly the records most worth showing. So the chart draws the excluded
    group, the counted rungs appear as rows with no bars, and the caption says in words
    that nothing fell into a bucket, because a chart of bars with nothing behind them
    and no sentence above them reads as a distribution of the exclusions.
    """
    figure = model_distribution(
        values={"under 1 hour": 0, "1 to 4 hours": 0},
        excluded={"records_with_no_stated_model": 5},
        title="Time to first review",
    )
    document = parse(render_html(a_report((a_section(figure),))))
    section = _section_for(document, figure.claim.slug)
    assert section is not None
    (chart,) = charts(section)

    assert _rows(chart, "row-excluded", excluded=False) == [
        ("under 1 hour", 0),
        ("1 to 4 hours", 0),
    ]
    assert _rows(chart, "row-excluded", excluded=True) == [("records_with_no_stated_model", 5)]
    caption = chart.find("figcaption").text()
    assert "No record fell into any counted bucket" in caption
    assert f"Counted: 0. Excluded: {figure.excluded_total:,}." in caption


def test_a_figure_with_no_payload_renders_its_completeness_sentence_and_nothing_else() -> None:
    """A measure with no number to state is a fact, and it renders as the sentence.

    :class:`~tenbin.measures.base.Figure` with an empty ``value_text`` is the shape a
    measure produces when its answer is that the number does not exist, and the part above
    it already says whether the read was whole -- which is the whole of what there is to
    say. So no block is emitted, and this test asserts the absence so that a later change
    to draw something here has to argue with the reason: there is no number to scale a
    bar to, and inventing a denominator to scale it to is the thing this package exists
    to refuse.
    """
    claim = model_claim()
    figure = Figure(
        claim=claim, snapshot=whole_read(), title="Latency, which this corpus cannot date"
    )
    document = parse(render_html(a_report((a_section(figure),))))
    section = _section_for(document, claim.slug)
    assert section is not None

    assert not section.find_all("svg")
    assert figure.rate_text() in section.text()
    assert figure.value_text() == ""


def test_a_group_member_with_nothing_to_draw_contributes_no_heading_either() -> None:
    """A member block is not emitted at all unless it has something in it.

    The alternative is a heading over nothing, which is the hole
    :mod:`tenbin.report.base` exists to prevent and which in a group would be easy to
    introduce: the member's own title is already in the payload text above, so a
    heading for a member that rendered nothing would be a second, empty appearance of
    the same sentence.
    """
    rate = capture_rate_figure()
    group = FigureGroup(
        claim=rate.claim,
        snapshot=truncated_read(),
        title="Two answers, one of them empty",
        figures=(
            capture_rate_figure(numerator=412),
            Figure(claim=rate.claim, snapshot=truncated_read(), title="A member with no number"),
        ),
    )
    document = parse(render_html(a_report((a_section(group),))))
    section = _section_for(document, group.claim.slug)
    assert section is not None

    assert [node.text() for node in section.find_all("h3")] == ["Changes carrying a record"]
    assert len(section.find_all("svg")) == 1


def test_prose_written_by_a_record_is_escaped_rather_than_executed() -> None:
    """Records carry prose written by people and by models, and it lands in a page.

    A bucket label is a string a measure counted, and a measure counts whatever was in
    the field -- so a label of ``<script>alert(1)</script>`` is a value, not an attack,
    and the renderer has to treat it as one. The assertions cover the three positions a
    label reaches: the payload prose above the chart, the ``<text>`` node in the
    drawing, and the ``<desc>``. The quote is there because a label containing one
    would break out of an attribute if any of those positions were an attribute.
    """
    hostile = '<script>alert("x")</script> & "quoted"'
    claim = a_claim(
        "hostile-bucket-labels",
        statement=f"Records whose model field says {hostile!r}.",
        does_not_mean="That the label is an instruction rather than a value.",
        falsifier="A walk finding no record whose model field contains a tag.",
        population="records enumerated by this read",
        size=500,
    )
    figure = model_distribution(
        claim=claim,
        values={hostile: 3, "opus": 212},
        excluded={"records_with_no_stated_model": 37},
        title="Records by stated model, from a store holding hostile labels",
    )
    rendered = render_html(a_report((a_section(figure),)))

    assert "<script" not in rendered
    assert "alert(&quot;x&quot;)" in rendered
    assert rendered.count("&lt;script&gt;") >= 3
    document = parse(rendered)
    section = _section_for(document, claim.slug)
    assert section is not None
    labels = [node.text() for node in section.find_all("text") if node.attr("class") == "row-label"]
    assert hostile in labels
    assert hostile in section.find("desc").text()
    assert hostile in section.text()


def test_two_renders_of_one_report_are_byte_identical_because_a_report_that_moves_is_undiffable() -> (
    None
):
    """No clock, no counter, no object identity: two renders are the same bytes.

    Rendered twice with a different report in between, so a renderer holding state
    between calls would be caught rather than a fresh interpreter. The ids are asserted
    as well, because a generated id is the most likely way for a chart to become
    non-deterministic -- ``id()`` values change every run and a diff of two reports would
    then be a diff of two integers.
    """
    report, _ = one_of_each()
    first = render_html(report)
    render_html(a_report((a_section(clock_skew_finding()),)))  # a different report in between
    second = render_html(report)

    assert first == second
    ids = re.findall(r'\sid="([^"]+)"', first)
    assert ids
    for value in ids:
        assert re.fullmatch(
            r"chart-[0-9-]+-(distribution|rate)-(title|desc)|hatch-[0-9-]+", value
        ), value
    assert len(ids) == len(set(ids)), "two elements share an id"


def test_the_document_holds_no_external_reference_because_a_report_that_needs_a_network_renders_without_its_caveats() -> (
    None
):
    """One file, no network, no script.

    A report that fetches its own caveats renders without them the moment the fetch
    fails, and the failure is silent: the page still opens, the bars are still there,
    and the sentence that said what the bars mean is the thing that did not arrive. So
    the check is a search for the four ways a self-contained file can quietly stop being
    one -- a URL, a script element, a stylesheet import, and a font that is not already
    on the machine.
    """
    rendered = render_html(one_of_each()[0])

    assert not re.search(r"https?://", rendered)
    assert "<script" not in rendered
    assert "@import" not in rendered
    assert not re.search(r"<link\b", rendered)
    assert "url(http" not in rendered
    document = parse(rendered)
    assert [node for node in document.elements() if node.tag in {"script", "link", "iframe"}] == []
    assert document.find("style").text().count("prefers-color-scheme") == 1


def test_both_colour_schemes_define_the_same_tokens_because_a_missing_one_falls_back_to_the_other() -> (
    None
):
    """The dark block has to name every colour the light one names.

    A token the dark scheme forgets is not a missing colour: CSS resolves it against
    the light value, so the page renders with a light-mode bar on a dark background at
    a contrast nobody chose. The failure is invisible in review and obvious in the
    report, and it is exactly the kind that gets introduced by adding a token in one
    block and forgetting the other.
    """
    rendered = render_html(one_of_each()[0])
    light = _tokens(_css_block(rendered, ":root", after="<style>"))
    dark = _tokens(_css_block(rendered, ":root", after="prefers-color-scheme: dark"))

    assert light
    assert set(light) == set(dark)


def test_every_colour_in_both_schemes_is_legible_against_both_backgrounds() -> None:
    """Contrast computed, not asserted as a string.

    "The stylesheet has a dark block" says nothing about whether anybody can read the
    dark one. So every token in both schemes is measured against both backgrounds -- the
    page and the card a section sits on -- at the threshold appropriate to what it
    carries: the text threshold for the tokens used as a ``color:``, and the mark
    threshold for the rest. A palette that passes this is legible; a palette that fails
    it is a report that renders its caveats in a colour nobody can read, which is the
    same failure as not rendering them.
    """
    rendered = render_html(one_of_each()[0])
    schemes = {
        "light": _tokens(_css_block(rendered, ":root", after="<style>")),
        "dark": _tokens(_css_block(rendered, ":root", after="prefers-color-scheme: dark")),
    }

    for name, tokens in schemes.items():
        for token, value in sorted(tokens.items()):
            if token in BACKGROUNDS or token in TEXTURE_TOKENS:
                continue
            for background in BACKGROUNDS:
                ratio = _contrast(value, tokens[background])
                floor = TEXT_CONTRAST if token in TEXT_TOKENS else MARK_CONTRAST
                assert ratio >= floor, (
                    f"{name} {token} on {background} is {ratio:.2f}:1, below {floor}:1"
                )


def test_a_banner_is_readable_in_both_schemes_because_a_banner_is_the_one_thing_that_reverses_the_page() -> (
    None
):
    """The refusal and finding banners are the one place the colours are swapped.

    Everywhere else a colour is a foreground on a background. A banner is a background
    with the page's own background colour as its text, which inverts the relationship in
    both schemes: white on a dark red in the light one, near-black on a light salmon in
    the dark one. That inversion is easy to get wrong in one scheme and invisible in the
    other, so it is measured as its own pairs rather than folded into the sweep above --
    and a sweep would miss it anyway, since ``--bg`` is excluded from the sweep as a
    background and here it is a foreground.
    """
    rendered = render_html(one_of_each()[0])
    for name, tokens in (
        ("light", _tokens(_css_block(rendered, ":root", after="<style>"))),
        ("dark", _tokens(_css_block(rendered, ":root", after="prefers-color-scheme: dark"))),
    ):
        for banner in ("--refusal", "--accent"):
            ratio = _contrast(tokens["--bg"], tokens[banner])
            assert ratio >= TEXT_CONTRAST, f"{name} banner text on {banner} is {ratio:.2f}:1"


def test_the_hatch_texture_is_visible_against_the_colour_it_is_drawn_on() -> None:
    """The stripe that makes an excluded bar look excluded has to be seen.

    The hatching is how a reader who cannot separate the two bar colours knows which
    is which, so a stripe that is nearly the same value as the bar under it would leave
    the texture carrying no information at all -- a distinction made in a channel the
    reader cannot use, which is the same defect as encoding a number in colour alone.
    Measured rather than eyeballed, and in both schemes, because a stripe that works on
    a light background is not thereby visible on a dark one.
    """
    rendered = render_html(one_of_each()[0])
    for scheme in (
        _tokens(_css_block(rendered, ":root", after="<style>")),
        _tokens(_css_block(rendered, ":root", after="prefers-color-scheme: dark")),
    ):
        assert _contrast(scheme["--stripe"], scheme["--excluded"]) >= TEXT_CONTRAST


def test_every_value_on_the_page_is_readable_as_text_and_not_only_as_a_shape() -> None:
    """Rule seven: no number is carried by a bar alone.

    Every bucket label and every count is printed in the payload prose above the chart
    as well as being drawn, so a reader with the picture switched off, a reader at 400%
    zoom, and a screen reader are all reading the same numbers off the same sentence.
    The assertion is that the text and the drawing agree, not merely that both exist: a
    renderer that drew one distribution and printed another would satisfy a weaker test
    than the one that would catch it.
    """
    report, (distribution, rate, _) = one_of_each()
    document = parse(render_html(report))
    section = _section_for(document, distribution.claim.slug)
    assert section is not None
    (chart,) = charts(section)

    printed = section.text()
    for label, count in distribution.values.items():
        assert label in printed
        assert f"{count:,}" in printed
    for label, count in distribution.excluded.items():
        assert label in printed
        assert f"{count:,}" in printed
    assert dict(_rows(chart, "row-excluded", excluded=False)) == dict(distribution.values)

    rate_section = _section_for(document, rate.claim.slug)
    assert rate_section is not None
    (rate_chart,) = charts(rate_section)
    assert f"{rate.rate.numerator:,} of {rate.rate.denominator.size:,} = {rate.rate.value:.1%}" in (
        rate_chart.text()
    )


def test_every_chart_is_named_for_a_screen_reader_and_points_at_its_own_description() -> None:
    """``role="img"``, a ``<title>`` and a ``<desc>`` inside the drawing, wired together.

    Asserted as a resolution rather than as a presence: the ids named in
    ``aria-labelledby`` have to exist *inside that same ``<svg>``, or a chart is
    announcing another chart's name. Two distributions on one page are what makes that
    worth testing, because they are the case in which ids derived from the kind of
    chart rather than from its position would collide -- and a colliding id is a
    duplicate id, which is invalid, and a chart pointing at the wrong one.

    Two claims rather than one claim twice, because :class:`~tenbin.report.document.Report`
    refuses two sections for one slug and is right to: a reader holding two charts for
    one claim cannot tell which is the program's answer.
    """
    second = a_claim(
        "records-by-stated-model-second-read",
        statement="The same distribution, read over a corpus one week later.",
        does_not_mean="That the two reads describe the same week.",
        falsifier="A second read whose shares match the first exactly.",
        population="records enumerated by the second read",
        size=500,
    )
    same_kind = a_report(
        (
            a_section(model_distribution(title="Records by stated model, first read")),
            a_section(
                model_distribution(claim=second, title="Records by stated model, second read")
            ),
        )
    )
    document = parse(render_html(same_kind))

    drawings = document.find_all("svg")
    assert len(drawings) == 2
    ids: set[str] = set()
    for drawing in drawings:
        assert drawing.attr("role") == "img"
        assert drawing.has_attr("viewBox")
        inside = {node.attr("id") for node in drawing.find_all("title") + drawing.find_all("desc")}
        named = drawing.attr("aria-labelledby").split()
        assert len(named) == 2
        assert set(named) <= inside
        assert not ids & set(named), "two charts share an accessible name"
        ids |= set(named)
        assert drawing.find("title").text().endswith("— distribution")
    titles = {drawing.find("title").text() for drawing in drawings}
    assert titles == {
        "Records by stated model, first read — distribution",
        "Records by stated model, second read — distribution",
    }


def test_the_html_carries_every_part_the_other_two_renderers_carry() -> None:
    """Three renderers, one ordering, no caveat dropped by the one that draws pictures.

    The design rests on there being one ordering in three forms, and this is the test
    that catches a renderer inventing content or quietly losing some: every line of
    every ordered part has to appear in the HTML. Asserted per part rather than per
    document so that a failure names the field that diverged, and per line because a
    part's text is a multi-line payload and a containment check over the whole string
    would pass for a renderer that dropped one bucket.
    """
    from tenbin.report.json_ import report_as_dict

    report, _ = one_of_each()
    rendered = render_html(report)
    document = parse(rendered)
    everything = document.text()

    parts = [part for section in report.sections for part in ordered_parts(section)]
    parts.extend(report.corpus.parts())
    for part in parts:
        for line in part.text.splitlines():
            assert _norm(line) in everything, f"{part.kind.value} lost the line {line[:40]!r}"

    payload = report_as_dict(report)
    for section, entry in zip(report.sections, payload["sections"], strict=True):
        for part in ordered_parts(section):
            for line in entry[part.kind.value].splitlines():
                assert _norm(line) in everything


def test_an_empty_report_is_a_document_rather_than_a_hole() -> None:
    """A report about an empty corpus is a fact, and it still gets a header.

    ``Report`` allows no sections, and a renderer that emitted an empty ``<body>`` here
    would have produced a file that opens to nothing. The corpus header is content in
    its own right and is what says whether the absence of numbers is about the corpus
    or about the program.
    """
    report = a_report(())
    document = parse(render_html(report))

    assert [node.attr("class") for node in document.find_all("section")] == ["corpus"]
    assert document.find("h1").text() == "Tenbin report"
    assert not document.find_all("svg")
    assert "10,000 of 25,000 documents" in document.text()


# --------------------------------------------------------------------------- helpers


def _section_for(document: _Node, slug: str) -> _Node | None:
    """The section rendered for a claim slug, or ``None``.

    Keyed on ``data-slug`` rather than on a heading, because a heading is a sentence a
    figure was given and two sections may share one; the slug is the identifier a report
    is keyed by everywhere else in this package.
    """
    for node in document.find_all("section"):
        if node.has_attr("data-slug") and node.attr("data-slug") == slug:
            return node
    return None


def _rows(chart: _Node, marker: str, *, excluded: bool) -> list[tuple[str, int]]:
    """The ``(label, count)`` pairs of the rows in one group of a chart.

    Read out of the drawing rather than out of the figure, so the assertion is about
    what the bars say and not about what the payload text above them says. The count is
    the right-hand text node of the row and the label the left-hand one, which is why
    the counts are parsed back out of their separators rather than compared as raw
    strings.
    """
    rows: list[tuple[str, int]] = []
    for group in chart.classed("row"):
        is_excluded = marker in group.attr("class").split()
        if is_excluded is not excluded:
            continue
        texts = [node.text() for node in group.find_all("text")]
        label = next(text for text in texts if not _is_count(text))
        count = int(next(text for text in texts if _is_count(text)).replace(",", ""))
        rows.append((label, count))
    return rows


def _is_count(text: str) -> bool:
    """Whether a text node is a formatted integer rather than a label."""
    return bool(re.fullmatch(r"[\d,]+", text))


def _norm(text: str) -> str:
    """Whitespace-normalised, the way :meth:`_Node.text` normalises a subtree.

    A part's own text carries indentation that a paragraph-per-line rendering does not
    -- :meth:`~tenbin.measures.base.DistributionFigure.value_text` indents its
    exclusions by two spaces to keep them readable as a list -- so a line compared
    verbatim against the page would fail on whitespace the reader cannot see. Comparing
    the normalised forms asks the question the test is about, which is whether the words
    are there.
    """
    return " ".join(text.split())


def _bar_lengths(chart: _Node) -> list[float]:
    """The width of every bar in a chart, in the order they are drawn."""
    return [
        rect.number("width")
        for rect in chart.find_all("rect")
        if rect.attr("class") in {"bar", "hatch"}
    ]


def _rate_is_drawn_on_a_fixed_axis(chart: _Node, expected: float) -> bool:
    """Whether a rate's bar is ``expected`` of the distance from the 0% to the 100% tick.

    The axis is measured from the markup rather than from a constant in the
    implementation, so the assertion is about the relationship between the bar and the
    axis it is drawn against. A tenth of a unit of slack absorbs the two-decimal
    rounding the coordinates are printed at and nothing else: the difference between a
    rate drawn on a fixed axis and one scaled to fit is not a rounding error.
    """
    grids = [line.number("x1") for line in chart.find_all("line") if line.attr("class") == "grid"]
    if not grids:
        return False
    axis = max(grids) - min(grids)
    bar = _bar_lengths(chart)[0]
    return abs(bar / axis - expected) < 1e-3


def _css_block(rendered: str, header: str, *, after: str) -> str:
    """The declarations inside the first block opening with ``header``, after ``after``.

    Brace-counted rather than matched with a pattern, because a stylesheet has nested
    blocks -- the dark scheme is a ``:root`` inside a ``@media`` -- and a pattern that
    stopped at the first ``}`` would return the media query rather than the variables
    in it.
    """
    start = rendered.index(after) + len(after)
    open_at = rendered.index("{", rendered.index(header, start))
    depth = 0
    for index in range(open_at, len(rendered)):
        if rendered[index] == "{":
            depth += 1
        elif rendered[index] == "}":
            depth -= 1
            if depth == 0:
                return rendered[open_at + 1 : index]
    raise AssertionError(f"unterminated block after {after!r}")


def _tokens(block: str) -> dict[str, str]:
    """The custom properties in a block of declarations, as ``name -> colour``."""
    return dict(re.findall(r"(--[a-z-]+):\s*(#[0-9a-fA-F]{6})\s*;", block))


def _contrast(one: str, other: str) -> float:
    """The WCAG contrast ratio between two hex colours, as a number.

    Written out rather than imported because there is no dependency to import it from,
    and adding one for a test would be a heavier change than the formula. The relative
    luminance is the WCAG definition applied to sRGB as written, which is what a browser
    does with these two values.
    """

    def luminance(colour: str) -> float:
        channels = [int(colour[index : index + 2], 16) / 255 for index in (1, 3, 5)]
        linear = [
            channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
            for channel in channels
        ]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    lighter, darker = sorted((luminance(one), luminance(other)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)
