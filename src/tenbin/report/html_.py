"""The report as one HTML file, where a chart and the prose that bounds it are one unit.

**A chart is the most effective device ever built for separating a number from its
bound.** Every other renderer in this package prints the caveats as text near the
figure and trusts the reader to keep them in mind while they look at the number. A
bar chart does not ask the reader to keep anything in mind: it puts a length on a
page, the eye goes to the length, and the sentence under it becomes a footnote to a
shape. So this renderer inverts the usual arrangement. The claim's statement, what it
does not mean and what would falsify it are emitted *before* the chart and inside the
same element, the figure's own completeness sentence is emitted *above* the chart, and
the denominator is emitted *below* it -- and the four are not in a tooltip, not in a
``<details>``, and not in a stylesheet that hides them.

**There is no code path that emits a chart without them, and the shape of this module
is the argument for that rather than the comment above it.** A chart is produced by
:func:`_distribution_chart` and :func:`_rate_chart`, and neither is reachable except
through :func:`_figure_block`, which is reachable only from :func:`_render_section`,
which has already rendered the section's ordered parts and hands the chart the very
:class:`~tenbin.claims.model.Claim` object those parts were rendered from. The
chart's ``<desc>`` repeats that claim verbatim, so the binding is not only structural
-- a chart whose description had drifted from the prose around it would be
noticeably wrong to a screen reader before it was wrong to anybody else. The test that
matters is :func:`test_no_chart_in_the_document_appears_without_the_claims_of_its_own_section`.

**The ordering is not this module's decision.** Prose comes from
:func:`~tenbin.report.ordering.ordered_parts` and geometry comes from
``section.figure`` -- the *same object* the prose describes. That is what stops the
two drifting: there is no second source for the numbers, and a distribution whose
buckets render one way in the text and another way in the bars would have to be two
implementations disagreeing rather than one implementation read twice.

**A distribution is drawn from its ``values`` *and* its ``excluded``, and the two are
never merged.** A bucket that did not occur and a bucket that was filtered out are the
same integer in ``values`` and completely different facts, and the reason this program
exists is the number of times that distinction has been lost. So the excluded records
get their own labelled group, their own count, their own hatched bars, and a caption
that says what the hatching means -- and they are never folded into an "other" bucket,
because "other" is a bucket that says nothing about the condition that produced it.

**Bar length is labelled as to what it is relative to, and for a distribution that is
the largest bucket rather than the denominator.** The buckets and the exclusions are
two different totals, and a reader who sees bars that appear to add up to the
population has been told something this program cannot support. A rate is the
opposite case and gets the opposite treatment: its bar spans a fixed 0-100% axis, so
a rate of 0.4% looks like 0.4% rather than being scaled to fill the page.

**An empty distribution is an absence, and it is rendered as one.** No axis, no zero,
no bars: a chart with an axis and no bars reads as a measurement that found nothing
*there*, which is a different and much stronger claim than a measure that has nothing
to show. The one thing this renderer will not produce is a hole that reads as a
result, in either direction -- which is also why a refusal renders as a refusal and
never as an empty chart, and why a ``Finding`` renders as a callout rather than as a
bar.

**This file is everything, and that is the requirement rather than a shortcut.** No
JavaScript, no external stylesheet, no CDN, no web font, no plotting library: a
report that needs a network to render its own caveats is a report that renders without
them the moment the network is not there, and the caveats are the load-bearing part.
The charts are inline SVG with a ``viewBox`` and no fixed width, so they scale with
the page and the file opens from ``file://`` on a machine that has never heard of
this repository. The two colour schemes are ``prefers-color-scheme`` blocks over CSS
custom properties, because the repository is read in both and a report that is legible
in one of them is legible in one of them only.

**What this module notably does not do.** It does not decide order, does not choose
what a claim says, and does not summarise a figure: it takes the prose from
:mod:`tenbin.report.ordering` and the numbers from the figure, and it adds words only
where a rule above requires words to exist on the page -- what a bar's length is
relative to, what the hatching means, that a rate's axis is fixed. A bare
:class:`~tenbin.measures.base.Figure` with an empty payload therefore gets no block
at all, because no rule asks for one and a sentence invented here would be a sentence
the JSON renderer does not carry. A :class:`~tenbin.measures.base.CountFigure` gets
no chart either, and that is a decision rather than an omission: a single bar has
nothing to compare itself against, so the only way to draw it honestly would be to
pick a denominator to scale it to, and this program does not invent a denominator.
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from dataclasses import dataclass

from tenbin.claims.model import Claim
from tenbin.measures.base import (
    CountFigure,
    DistributionFigure,
    Figure,
    FigureGroup,
    Finding,
    RateFigure,
)
from tenbin.report.base import Section
from tenbin.report.document import CorpusHeader, Report
from tenbin.report.ordering import Part, PartKind, ordered_parts

#: The document title. Its own constant rather than
#: :data:`tenbin.report.markdown.REPORT_TITLE`, because that one is the string
#: ``"# Tenbin report"`` -- a Markdown heading, not a word -- and importing it would
#: put a hash in an ``<h1>``. What the two renderers share is the decision, not the
#: characters, and the characters are presentation.
REPORT_TITLE = "Tenbin report"

#: The heading the corpus read is rendered under. Its own constant for the same
#: reason, and because a heading is not a claim about ordering: every renderer puts the
#: read first because :mod:`tenbin.report.build` hands them the corpus first.
CORPUS_HEADING = "Corpus"

#: The banner a refusal is rendered under, which is the one sentence a refusal needs
#: that its three ordered parts do not already say. The parts are the reason, the
#: missing fact and the ticket; what they do not say is that no number exists here,
#: and a section with no number under a heading is the failure
#: :mod:`tenbin.report.base` exists to make unconstructible -- so the refusal says so
#: in its own words rather than leaving the reader to notice the absence.
REFUSAL_BANNER = (
    "Refused: this measure is not rendered, and no number below is offered for it. "
    "The section is the refusal itself."
)

#: The banner a finding is rendered under. A finding is a claim that something is
#: wrong, not a measurement of how much of something there is, which is why it is a
#: callout and not a chart: there is no population for a bar to be a fraction of, and
#: a bar drawn for one would be inventing the denominator this program exists to
#: refuse to invent.
FINDING_BANNER = (
    "Finding, not a measurement: an observation that something is wrong. There is no "
    "population for this to be a fraction of, so it is stated rather than charted."
)

#: What an empty distribution is rendered as. Says what was counted and what was
#: excluded, and then says the axis is absent rather than drawing one -- because an
#: axis with no bars on it reads as a result rather than as the absence of one, and a
#: zero drawn in a reader's head is a number this program did not compute.
ABSENT_DISTRIBUTION = (
    "Empty distribution: no record fell into any bucket and none was excluded, so "
    "there is no axis to draw and none is drawn. This is the absence of a "
    "distribution rather than a result from one."
)

#: Prepended to a distribution whose buckets are all zero but whose exclusions are
#: not. The exclusions are still facts worth a chart, and saying the buckets were
#: empty is not the same as rendering an empty axis for them.
ABSENT_BUCKETS = (
    "No record fell into any counted bucket. The bars below are the excluded records, "
    "which are a different total and not a result."
)

#: The heading over the excluded group in a distribution chart. It names the
#: condition rather than the count, because "other" is a bucket that says nothing
#: about why the records were not counted and this heading is what a reader looks for
#: when they want to know. It is a separate group rather than a bar of its own colour
#: inside the buckets, and never an "other" bucket appended to them.
EXCLUDED_HEADING = "Excluded — filtered out, not a result"

#: The heading over the counted group, for the same reason: a reader who is looking
#: for where the exclusions went should find two named groups rather than one group
#: and a gap.
COUNTED_HEADING = "Counted buckets"

#: What a rate's caption says about its axis. Printed on the page rather than left to
#: the geometry, because a bar that has been scaled to fill the available space looks
#: identical whether the rate is 4% or 40% unless something says the axis is fixed.
RATE_CAPTION = (
    "The bar spans a fixed 0% to 100% axis rather than being scaled to fill the page, "
    "so a small rate looks small. The value is the fraction and its population below."
)

#: How many of a figure part's lines stay in the part's own body. One, because
#: :func:`tenbin.report.ordering._figure_text` puts the completeness sentence first and
#: every other line is the payload. The payload lines are then either left there as
#: visible text beside the chart, or moved inside the block -- which is what a finding's
#: callout does, re-emitting them from
#: :meth:`~tenbin.measures.base.Finding.value_text`. Either way each line is rendered
#: exactly once and none is dropped, which is the property that makes a chart a second
#: reading of a number rather than a replacement for it.
_COMPLETENESS_LINES = 1

#: Chart geometry, in SVG user units. The SVG carries a ``viewBox`` and no width, so
#: the whole drawing scales as one piece: text grows with the bars instead of
#: overflowing them, and no measurement of the label font is needed anywhere -- which
#: is the reason a bucket label is never truncated and never estimated into a column
#: that could clip it. The text size itself is in the stylesheet rather than here, so
#: that there is one place to change it and it is a place a browser already reads.
_CHAR_WIDTH = 7.2
_LABEL_PAD = 14
_TRACK_WIDTH = 300
_RATE_WIDTH = 400
_COUNT_COLUMN = 62
_ROW_HEIGHT = 24
_ROW_GAP = 6
_GROUP_GAP = 22
_BAR_HEIGHT = 16
_PAD_TOP = 6
_PAD_BOTTOM = 8
_RADIUS = 2

#: The shortest bar a non-zero count is drawn as, in SVG user units. A bucket holding
#: one record against a maximum of ten thousand would otherwise be a rectangle
#: narrower than a pixel and would read as a bucket that did not occur -- which is
#: precisely the confusion this renderer exists to prevent. One unit out of a
#: three-hundred-unit track is under half a percent of the width, well below the
#: precision any reader can take from a bar's length, and the count is printed beside
#: it in text regardless.
_MIN_BAR = 1.0

#: The prefix every generated element id in a chart carries. The ids exist only to
#: wire ``aria-labelledby`` to the ``<title>`` and ``<desc>`` inside the same drawing,
#: and they are suffixed with the chart's position in the document so that two charts
#: of the same kind on one page do not collide -- duplicate ids are invalid HTML and
#: a colliding one would leave a chart's accessible name pointing at another chart's.
_ID_PREFIX = "chart"


def render_html(report: Report) -> str:
    """Render the whole report as one self-contained HTML document.

    A pure function of ``report`` -- no clock, no store, no filename, nothing read
    from the environment and no randomness in any generated identifier -- so two
    renders of one report are byte-identical and a rendered document can be diffed,
    which is the only way a change to a chart is reviewable. The trailing newline is
    there for the same reason :func:`tenbin.report.markdown.render_markdown` puts
    one: every tool that appends to a text file appends it.

    The document holds no external reference of any kind, so it opens from ``file://``
    with the network off. A chart is inline SVG and the only stylesheet is a
    ``<style>`` element in the ``<head>``.
    """
    document: list[str] = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{_text(REPORT_TITLE)}</title>",
        "<style>",
        report_stylesheet(),
        "</style>",
        "</head>",
        "<body>",
        render_html_body(report),
        "</body>",
        "</html>",
        "",
    ]
    return "\n".join(document)


def render_html_body(report: Report) -> str:
    """The report's content as a ``<main>`` fragment, for a page that hosts it.

    **The fragment is not self-sufficient, and :func:`report_stylesheet` exists
    because of that.** Every class this emits is styled by a rule in that
    stylesheet, so a host that inlines the body without it gets a page of correct
    HTML that looks like plain text -- which is the worst outcome available here,
    because a reader would see the caveats and miss that the bars are missing.

    Splicing this into another document is therefore the supported way to embed a
    report, and it is what the web app does. The alternative -- a ``srcdoc`` frame
    around a whole document -- also works, and was what this replaced: it nests a
    document inside a page, which means the report's markup is escaped into an
    attribute and unreadable to anything inspecting the served HTML. A 200 KB page
    whose charts are invisible to ``curl`` is a page nobody can test.

    The fragment carries the same content in the same order as
    :func:`render_html`, because both call the same renderer for every section; a
    report cannot be reordered by choosing one entry point over the other.
    """
    body: list[str] = [
        "<main>",
        f"<h1>{_text(REPORT_TITLE)}</h1>",
        _render_header(report.corpus),
    ]
    for ordinal, section in enumerate(report.sections):
        body.append(_render_section(section, str(ordinal)))
    body.append("</main>")
    return "\n".join(body)


def report_stylesheet() -> str:
    """The rules every class in :func:`render_html_body` depends on.

    Public beside the body rather than folded into it, because a host embedding the
    fragment has to bring these along and there is no way to guess them. Inlining
    them into a ``<style>`` in the host's own head is what
    :mod:`tenbin.webapp.pages` does; the alternative -- copying the rules into the
    host's stylesheet -- is how two copies of one rule come to say two slightly
    different things.
    """
    return _STYLESHEET


def _text(value: object) -> str:
    """Escape a value for use as a text node or as an attribute value.

    One function for both positions and for numbers as well as prose, because the
    alternative is a call site that decides whether the string it is holding came from
    a person, a model or an f-string -- and records carry prose written by both of the
    first two, rendered into a page that a browser will happily execute. A single
    chokepoint means a new field cannot be added to a chart without passing through
    here, which is the same argument
    :meth:`tenbin.measures.base.Figure.value_text` makes one layer down.

    ``quote=True`` escapes the apostrophe and the double quote as well, so the same
    call is safe in an attribute whether or not the author remembered which quoting
    they were writing into.
    """
    return html.escape(str(value), quote=True)


def _render_header(header: CorpusHeader) -> str:
    """The read the numbers below are about, in its own section.

    A section rather than a preamble, for the reason
    :mod:`tenbin.report.document` gives: a preamble is what a reader skips, and this
    is the part that decides whether any number under it means what the reader would
    otherwise assume. The parts come from
    :meth:`~tenbin.report.document.CorpusHeader.parts` and this function decides
    nothing about them -- there is no ``figure`` part in a header, so no block is
    emitted and no chart is reachable from here.
    """
    return (
        '<section class="corpus" data-section="header">'
        f"<h2>{_text(CORPUS_HEADING)}</h2>"
        f"{_render_parts(header.parts())}"
        "</section>"
    )


def _render_section(section: Section, ordinal: str) -> str:
    """One measure's answer: its parts in the one order, and the chart inside them.

    The parts are taken from :func:`~tenbin.report.ordering.ordered_parts` and
    rendered in the sequence it returns, with one addition: the ``figure`` part also
    carries the block for the payload, so the chart sits in the slot the ordering gave
    it -- between the falsifier above it and the denominator below it. That position
    is the whole argument of :mod:`tenbin.report.ordering` applied to a picture, and
    it is inherited rather than reimplemented, which is why this renderer cannot be
    the thing that moves a caveat under a chart.

    A refusal takes the same walk with no block at all, so the absence of a chart in
    a refusal is structural: :func:`_figure_block` is not called on that path.
    """
    parts = ordered_parts(section)
    refusal = section.refusal
    if refusal is not None:
        return (
            f'<section class="section refusal" data-slug="{_text(section.slug)}">'
            f"<h2>{_text(section.heading)}</h2>"
            f'<p class="banner refusal-banner">{_text(REFUSAL_BANNER)}</p>'
            f'<p class="slug">Claim slug: <code>{_text(refusal.claim_slug)}</code></p>'
            f"{_render_parts(parts)}"
            "</section>"
        )
    claim = section.claim
    figure = section.figure
    if claim is None or figure is None:
        # Unreachable: ``Section`` refuses to be constructed in either shape, and it is
        # the same guard ``ordered_parts`` raises on. Repeated here because this
        # function reads ``section.claim`` and ``section.figure`` for a type checker,
        # and a report layer that rendered a heading with nothing under it is the
        # defect the whole package is built to make unconstructible.
        raise ValueError("a Section must hold a figure or a refusal; see Section.__post_init__")
    block, moves_payload = _figure_block(figure, claim, ordinal)
    return (
        f'<section class="section claim" data-slug="{_text(section.slug)}">'
        f"<h2>{_text(section.heading)}</h2>"
        f"{_render_parts(parts, block=block, moves_payload=moves_payload)}"
        "</section>"
    )


def _render_parts(
    parts: Sequence[Part],
    *,
    block: str = "",
    moves_payload: bool = False,
) -> str:
    """Render parts as labelled blocks, with the payload block inside the figure one.

    ``block`` is placed inside the ``figure`` part and nowhere else, so the chart is
    emitted at the position :mod:`tenbin.report.ordering` chose rather than
    wherever the layout would have put it. ``moves_payload`` is the finding's case: a
    finding's payload lines are rendered inside its callout rather than beside it, so
    that a claim that something is wrong reads as one block rather than as a
    paragraph with a badge under it. For a chart the payload lines stay in the body as
    visible text next to the bars, which is what makes the chart a second reading of
    the number rather than the only one.
    """
    return "".join(
        _figure_part(part, block=block, moves_payload=moves_payload)
        if part.kind is PartKind.figure
        else _part(part)
        for part in parts
    )


def _part(part: Part) -> str:
    """One part: the label its kind carries, then its text as its own paragraphs.

    The label comes from :attr:`tenbin.report.ordering.PartKind.label` rather than
    from a table in this file, so a part kind has one introduction across three
    renderers. A part whose text spans lines is rendered as one paragraph per line
    rather than with a line break, because a ``<p>`` is something a reader's
    assistive technology announces and a ``<br>`` is not always.
    """
    body = "".join(f"<p>{_text(line)}</p>" for line in part.text.splitlines())
    return (
        f'<div class="part" data-part="{_text(part.kind.value)}">'
        f'<p class="label">{_text(part.kind.label)}</p>'
        f'<div class="body">{body}</div>'
        "</div>"
    )


def _figure_part(part: Part, *, block: str, moves_payload: bool) -> str:
    """The figure part: completeness above, payload beside or inside the block.

    The completeness sentence is the first line of a figure part by construction --
    :func:`tenbin.report.ordering._figure_text` puts it there and nothing else can --
    so the first line stays in the part's own body and is never moved into a chart,
    a callout or a caption. That is the sentence that says whether the number is
    whole, and it is the one piece of prose a reader must see before the shape rather
    than after it.
    """
    lines = part.text.splitlines()
    head = lines[:_COMPLETENESS_LINES]
    body_lines = head if moves_payload else lines
    body = "".join(f"<p>{_text(line)}</p>" for line in body_lines)
    inner = f'<div class="body">{body}</div>'
    return (
        f'<div class="part part-figure" data-part="{_text(part.kind.value)}">'
        f'<p class="label">{_text(part.kind.label)}</p>'
        f"{inner}"
        f"{block}"
        "</div>"
    )


def _figure_block(figure: Figure, claim: Claim, ordinal: str) -> tuple[str, bool]:
    """The block for a figure, and whether the payload lines move inside it.

    The dispatch is on the payload type, and each branch is a different kind of
    visual answer to a different question:

    - a **distribution** draws its buckets and its exclusions, apart from each other;
    - a **rate** draws a fixed 0-100% bar with the fraction printed;
    - a **group** draws each member's own chart, under the member's own title, inside
      the one claim the whole group shares;
    - a **finding** is a callout, because there is no population to be a fraction of;
    - a **count** is left as its number, because a single bar has nothing to be
      measured against and choosing a denominator to scale it to would be inventing
      the one field this program refuses to publish without;
    - a bare figure with an empty payload gets nothing, because the completeness
      sentence already in the part above says the whole of what there is to say.

    ``claim`` is the section's own claim object -- the same one
    :func:`~tenbin.report.ordering.ordered_parts` rendered the four bound parts
    from -- and it is a required argument of every chart below rather than something a
    chart can reach for itself. A chart therefore cannot be constructed without the
    claim that bounds it, and the ``<desc>`` it carries is that claim rather than a
    restatement of it.
    """
    if isinstance(figure, Finding):
        return _callout(figure), True
    if isinstance(figure, FigureGroup):
        return _group_block(figure, claim, ordinal), False
    if isinstance(figure, DistributionFigure):
        return _distribution_chart(figure, claim, ordinal), False
    if isinstance(figure, RateFigure):
        return _rate_chart(figure, claim, ordinal), False
    if isinstance(figure, CountFigure):
        return "", False
    return "", False


def _group_block(group: FigureGroup, claim: Claim, ordinal: str) -> str:
    """Each member's own chart, under its own title, inside the group's one claim.

    A group is a measure that answers in several parts over the same population, so
    every member's chart is bound by the same four parts as any other chart on the
    page -- which is why drawing them is safe even though a member carries no claim of
    its own. The claim is shared by construction
    (:class:`~tenbin.measures.base.FigureGroup` refuses members that do not share
    it), and it is rendered once in the section above them all.

    A member with nothing to draw contributes nothing, and not even a heading: a
    heading over nothing is the hole :mod:`tenbin.report.base` exists to prevent,
    and the member's own title is already in the payload text above.
    """
    blocks: list[str] = []
    for index, member in enumerate(group.figures):
        block, _ = _figure_block(member, claim, f"{ordinal}-{index}")
        if not block:
            continue
        blocks.append(
            f'<div class="member"><h3 class="member-title">{_text(member.title)}</h3>{block}</div>'
        )
    return "".join(blocks)


def _callout(finding: Finding) -> str:
    """A finding, as a block that cannot be mistaken for a measurement.

    The payload is :meth:`~tenbin.measures.base.Finding.value_text` verbatim, one
    paragraph per line, rather than the four fields re-assembled into a description
    list. The reason is parity rather than taste: this is the third renderer, and the
    property the package holds is that all of them carry the same words. A callout
    built from the fields would carry the same *information* as the line the other two
    renderers print and not the same *string*, so a test asserting that the three agree
    would have to be weakened for this one shape -- and a weakened parity test is a
    parity test that has stopped being one.

    ``cannot_confirm`` lands last and stays visible in the callout, because
    :meth:`~tenbin.measures.base.Finding.value_text` puts it last and a reader who
    stops after the suspected cause has still been told what this is not.
    """
    body = "".join(f"<p>{_text(line)}</p>" for line in finding.value_text().splitlines())
    return (
        '<aside class="callout" role="note">'
        f'<p class="banner">{_text(FINDING_BANNER)}</p>'
        f"{body}"
        "</aside>"
    )


def _distribution_chart(figure: DistributionFigure, claim: Claim, ordinal: str) -> str:
    """A distribution's buckets, and beside them the records that were left out.

    Two groups on one axis, never one group of buckets and one "other": the excluded
    records are the ones this program has most reason to be honest about, and folding
    them into a bucket is how an ``unknown`` model became a body of records whose
    authors declared a model called ``unknown``. So they get a heading that names what
    they are, hatched bars so the distinction survives a monochrome print and a reader
    who cannot separate the hues, a count of their own, and a line in the caption
    totalling them separately from the buckets.

    **The scale is the largest bucket on the page and nothing else.** Not the
    denominator, because the buckets and the exclusions are two totals and a chart
    scaled to the population would imply they add up to it. The caption says so in
    words, because a bar chart with an unlabelled scale is a claim about proportion
    made by the absence of a sentence.

    **An empty distribution renders no chart.** See :data:`ABSENT_DISTRIBUTION`; the
    check is on ``counted`` and ``excluded_total`` together rather than on the
    presence of keys, because a histogram that includes every rung at zero is empty in
    the only sense a reader cares about.
    """
    if figure.counted == 0 and figure.excluded_total == 0:
        return _absent(ABSENT_DISTRIBUTION, "chart-absent")
    rows = [_Row(label, count, excluded=False) for label, count in figure.values.items()]
    excluded = [_Row(label, count, excluded=True) for label, count in figure.excluded.items()]
    # One scale for both groups, or an excluded bar would not be comparable with a
    # counted one and the comparison is the point of putting them on the same page.
    # Unreachable at zero, because this branch is only reached when one of the two
    # totals is above zero and every count is non-negative; the guard is there so a
    # future relaxation of that cannot divide by it.
    scale = max((row.count for row in (*rows, *excluded)), default=0)
    if scale < 1:
        return _absent(ABSENT_DISTRIBUTION, "chart-absent")
    gutter = _gutter(rows, excluded)
    width = gutter + _TRACK_WIDTH + _COUNT_COLUMN
    body: list[str] = []
    y = _PAD_TOP
    for heading, group in ((COUNTED_HEADING, rows), (EXCLUDED_HEADING, excluded)):
        if not group:
            continue
        # At x=0 rather than over the bars: the heading names the group, and the group
        # is the labels and the bars together. A heading sitting above the bar column
        # alone reads as a caption on the longest bar rather than as the name of what
        # the rows under it have in common.
        body.append(_text_node(0, y + _BAR_HEIGHT - 4, heading, css="group-label"))
        y += _GROUP_GAP
        for row in group:
            body.append(_row(row, gutter, y, scale, ordinal))
            y += _ROW_HEIGHT + _ROW_GAP
    height = y - _ROW_GAP + _PAD_BOTTOM
    caption = (
        f"Bar length is proportional to the largest bucket in this chart — "
        f"{scale:,} — and not to the denominator: the counted buckets and the "
        f"excluded records are two different totals, and a chart scaled to the "
        f"population would imply the bars add up to it. Counted: {figure.counted:,}. "
        f"Excluded: {figure.excluded_total:,}."
    )
    if figure.counted == 0:
        caption = f"{ABSENT_BUCKETS} {caption}"
    counted_line = "; ".join(f"{row.label} {row.count:,}" for row in rows)
    excluded_line = "; ".join(f"{row.label} {row.count:,}" for row in excluded)
    return _figure_element(
        chart_id=f"{_ID_PREFIX}-{ordinal}-distribution",
        title=f"{figure.title} — distribution",
        description=_description(
            figure,
            claim,
            [
                f"Counted buckets: {counted_line}." if rows else "No record fell in any bucket.",
                f"Excluded: {excluded_line}." if excluded else "Nothing was excluded.",
            ],
        ),
        width=width,
        height=height,
        body="".join([_hatch_defs(ordinal), *body]),
        caption=caption,
        css="chart-distribution",
    )


def _row(row: _Row, gutter: int, y: int, scale: int, ordinal: str) -> str:
    """One bucket: its name, a bar, and its count, all in the row.

    A bucket holding zero gets its name and its count and no bar, which is not the
    same as being absent from the chart. :func:`tenbin.measures.base.duration_buckets`
    includes every rung at zero precisely so that a reader comparing two runs can see
    that a rung was empty rather than infer it from a gap in the picture, and a gap in
    the picture is not a fact.
    """
    top = y + (_ROW_HEIGHT - _BAR_HEIGHT) / 2
    pieces = [_text_node(0, top + _BAR_HEIGHT - 4, row.label, css="row-label")]
    if row.count > 0:
        length = max(_TRACK_WIDTH * row.count / scale, _MIN_BAR)
        pieces.append(_bar(row.excluded, gutter, top, length, ordinal))
    pieces.append(
        _text_node(
            gutter + _TRACK_WIDTH + _COUNT_COLUMN,
            top + _BAR_HEIGHT - 4,
            f"{row.count:,}",
            anchor="end",
            css="row-count",
        )
    )
    css = "row row-excluded" if row.excluded else "row"
    return f'<g class="{css}">{"".join(pieces)}</g>'


def _bar(excluded: bool, x: float, y: float, length: float, ordinal: str) -> str:
    """The rectangle for one bucket, hatched when the record was filtered out.

    Two classes for one distinction rather than one class and a colour: a counted bar
    is a solid fill and an excluded one carries a texture as well as a different hue,
    so the reader who cannot separate the two colours -- or who printed the page in
    black -- is still being told which is which. The hatch is a ``<pattern>``
    referenced by id, and the id is derived from the chart's position in the document
    so that two renders of one report produce identical bytes.
    """
    if excluded:
        return (
            f'<rect class="hatch" x="{_n(x)}" y="{_n(y)}" width="{_n(length)}" '
            f'height="{_BAR_HEIGHT}" rx="{_RADIUS}" fill="url(#hatch-{ordinal})"/>'
        )
    return (
        f'<rect class="bar" x="{_n(x)}" y="{_n(y)}" width="{_n(length)}" '
        f'height="{_BAR_HEIGHT}" rx="{_RADIUS}"/>'
    )


def _rate_chart(figure: RateFigure, claim: Claim, ordinal: str) -> str:
    """A rate as a bar over a fixed 0-100% axis, with the fraction printed.

    The axis is the point. A bar scaled to its own value fills the space it is given
    and is indistinguishable from a bar scaled to the whole range, so a rate of 0.4%
    and a rate of 40% would be the same picture; here the 0, 50 and 100 labels are on
    the page and the bar is drawn against them. The value is printed as a fraction
    over a population rather than as a bare percentage, because a percentage on its
    own reads as general and this one is over a denominator stated two lines below.
    """
    rate = figure.rate
    length = max(_RATE_WIDTH * rate.value, _MIN_BAR)
    # Absolute coordinates throughout, with no translated groups. A ``<g>`` whose
    # transform offsets the axis and whose children then carry coordinates that already
    # include the offset puts two elements on the same baseline, and the symptom is a
    # value printed through a tick label rather than a compile error -- so the rows are
    # laid out here as named offsets from the top of the drawing, each computed once.
    bar_top = _PAD_TOP
    axis_y = bar_top + _BAR_HEIGHT + 6
    tick_y = axis_y + 12
    value_y = tick_y + 20
    marks: list[str] = [
        f'<rect class="bar" x="0" y="{_n(bar_top)}" width="{_n(length)}" '
        f'height="{_BAR_HEIGHT}" rx="{_RADIUS}"/>',
        f'<line class="axis" x1="0" y1="{_n(axis_y)}" x2="{_RATE_WIDTH}" y2="{_n(axis_y)}"/>',
    ]
    for percent in (0.0, 50.0, 100.0):
        x = _RATE_WIDTH * percent / 100
        anchor = "start" if percent == 0 else "end" if percent == 100 else "middle"
        marks.append(
            f'<line class="grid" x1="{_n(x)}" y1="{_n(axis_y)}" x2="{_n(x)}" '
            f'y2="{_n(axis_y + _PAD_BOTTOM)}"/>'
        )
        marks.append(_text_node(x, tick_y, f"{percent:.0f}%", anchor=anchor, css="tick"))
    marks.append(
        _text_node(
            0,
            value_y,
            f"{rate.numerator:,} of {rate.denominator.size:,} = {rate.value:.1%}",
            css="value",
        )
    )
    return _figure_element(
        chart_id=f"{_ID_PREFIX}-{ordinal}-rate",
        title=f"{figure.title} — rate",
        description=_description(
            figure,
            claim,
            [
                f"{rate.numerator:,} of {rate.denominator.size:,} = {rate.value:.1%}, "
                f"drawn against a fixed 0-100% axis."
            ],
        ),
        width=_RATE_WIDTH,
        height=value_y + _PAD_BOTTOM + 4,
        body="".join(marks),
        caption=f"{RATE_CAPTION} Population: {rate.denominator.render_text()}.",
        css="chart-rate",
    )


def _absent(message: str, css: str) -> str:
    """A payload with nothing to draw, rendered as the statement that it is empty.

    Deliberately not a ``<figure>`` holding an empty ``<svg>``: an SVG with a
    ``viewBox`` and nothing in it is a rectangle of background that a reader's eye
    treats as a plot, and a plot with no bars is a result rather than an absence.
    """
    return f'<figure class="chart {css}"><figcaption class="absent">{_text(message)}</figcaption></figure>'


def _figure_element(
    *,
    chart_id: str,
    title: str,
    description: str,
    width: int,
    height: int,
    body: str,
    caption: str,
    css: str,
) -> str:
    """Wrap chart markup as an accessible figure, and close the unit.

    ``role="img"`` with ``aria-labelledby`` pointing at a ``<title>`` and a ``<desc>``
    inside the drawing, because an SVG that a screen reader meets as an unlabelled
    graphic is a number with its bound removed for exactly the reader least able to
    infer it from the bars. The ids are suffixed with the chart's position in the
    document rather than with the kind of chart, so a page holding two rates has two
    sets of ids: an id that is a function of the kind is a document with duplicate
    ids, and a duplicate id is a chart whose accessible name points at a different
    chart's.

    The ``<figcaption>`` is the same content as visible text, which is what lets the
    value be read with the colours off, at any zoom, and by anybody who is not looking
    at the picture at all.

    No ``xmlns`` attribute. Inline SVG in an HTML document is placed in the SVG
    namespace by the parser, and writing the namespace out would put a URL in a file
    whose whole requirement is that it contains none.
    """
    title_id = f"{chart_id}-title"
    desc_id = f"{chart_id}-desc"
    return (
        f'<figure class="chart {css}">'
        f'<svg viewBox="0 0 {width} {height}" class="drawing" role="img" '
        f'aria-labelledby="{title_id} {desc_id}">'
        f'<title id="{title_id}">{_text(title)}</title>'
        f'<desc id="{desc_id}">{_text(description)}</desc>'
        f"{body}"
        "</svg>"
        f"<figcaption>{_text(caption)}</figcaption>"
        "</figure>"
    )


def _description(figure: Figure, claim: Claim, values: Sequence[str]) -> str:
    """What a chart says to a reader who only meets the chart.

    The claim's three caveats first, in the order
    :mod:`tenbin.report.ordering` puts them on the page, then the values, then the
    denominator, then whether the read behind it was whole. A ``<desc>`` that carried
    only the numbers would be a chart that had lost its bound in the one place a
    reader cannot see the bound beside it, and the binding being a required argument
    of every chart here is what stops that from being a matter of remembering.

    Sentence-terminated rather than space-joined, because a screen reader announces
    this as running text and four clauses separated by single spaces are one clause.
    """
    sentences = [
        f"{figure.title}.",
        f"Statement: {claim.statement}",
        f"What it does not mean: {claim.does_not_mean}",
        f"What would falsify it: {claim.falsifier}",
        *values,
        f"Denominator: {claim.denominator.render_text()}.",
        figure.rate_text(),
    ]
    return " ".join(sentences)


def _hatch_defs(ordinal: str) -> str:
    """The hatch an excluded bar is filled with, defined once per chart.

    A texture rather than a second colour, so the distinction between a counted
    record and a filtered one survives a monochrome print, a projector, and a reader
    who cannot separate the two hues. The id is derived from the chart's position in
    the document, which is a function of the report rather than of a counter or a
    clock, so two renders of one report produce the same ids and therefore the same
    bytes.
    """
    return (
        "<defs>"
        f'<pattern id="hatch-{ordinal}" width="7" height="7" patternUnits="userSpaceOnUse" '
        'patternTransform="rotate(45)">'
        '<rect class="hatch-bg" width="7" height="7"/>'
        '<line class="hatch-line" x1="0" y1="0" x2="0" y2="7"/>'
        "</pattern>"
        "</defs>"
    )


def _gutter(rows: Sequence[_Row], excluded: Sequence[_Row]) -> int:
    """How wide the label column is, sized to the longest label in the chart.

    A fixed gutter would either clip a long bucket label -- and a clipped label is a
    label a reader has to guess at -- or leave a column of white for short ones. The
    chart's ``viewBox`` grows instead, and because the drawing scales as a whole the
    text grows with it. The estimate is a character width rather than a measured one
    because no font is loaded and none can be: the file has to render identically on a
    machine with no network, so the label is never allowed to depend on a font this
    program does not ship. The full label is in the ``<desc>`` and in the payload text
    above the chart regardless of what the column turns out to be worth.
    """
    labels = [row.label for row in (*rows, *excluded)]
    longest = max((len(label) for label in labels), default=0)
    return int(longest * _CHAR_WIDTH) + _LABEL_PAD


def _text_node(
    x: float,
    y: float,
    value: str,
    *,
    anchor: str = "start",
    css: str = "row-label",
) -> str:
    """One piece of chart text, positioned and escaped.

    A ``<text>`` node rather than an HTML label overlaid on the drawing, so the label
    scales with the bars as the chart resizes and cannot end up describing a bar that
    has moved away from it.
    """
    return (
        f'<text class="{css}" x="{_n(x)}" y="{_n(y)}" text-anchor="{anchor}">{_text(value)}</text>'
    )


def _n(value: float) -> str:
    """A coordinate, at the precision a pixel can show.

    Two decimal places, and no more: a coordinate printed to full float precision is
    longer than the number it describes, and the shorter form is stable across runs
    because it is the same computation over the same inputs.
    """
    return f"{value:.2f}"


@dataclass(frozen=True)
class _Row:
    """One bar: what it is called, how many, and whether it was counted.

    ``excluded`` is carried on the row rather than inferred from which list the row
    came from, so that the hatch, the class and the text cannot disagree with each
    other -- the three are the same distinction rendered three ways, and a chart where
    a bar is hatched but labelled as counted is worse than one that dropped the
    exclusions.
    """

    label: str
    count: int
    excluded: bool


#: The stylesheet, inlined. Two blocks over one set of custom properties, so the two
#: colour schemes are two lists rather than two stylesheets: a rule that needs a
#: colour says ``var(--name)`` once and the light and dark values are both in the same
#: place. ``color-scheme`` is declared so that a form control or a scrollbar drawn by
#: the browser matches the page, and ``prefers-color-scheme`` is the mechanism rather
#: than a class a reader has to know about, because a report that has to be told which
#: scheme it is in is a report that will be read in the wrong one.
#:
#: No web font and no import. A system font stack is the only kind of font that exists
#: on a machine this file might be opened on, and a stylesheet that fetches a font is a
#: stylesheet whose caveats are behind a request that can fail.
_STYLESHEET = """
:root {
  color-scheme: light dark;
  --bg: #ffffff;
  --surface: #f4f4f1;
  --ink: #15151a;
  --muted: #4a4a52;
  --edge: #8b8b85;
  --bar: #1f4e79;
  --excluded: #7a4a12;
  --accent: #6a2c70;
  --refusal: #8a2020;
  --track: #767670;
  --stripe: #f4f4f1;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #141417;
    --surface: #1e1e23;
    --ink: #f2f2f0;
    --muted: #b9b9c2;
    --edge: #7e7e88;
    --bar: #7fb3e0;
    --excluded: #e0a860;
    --accent: #d9a2dd;
    --refusal: #f08a8a;
    --track: #6b6b75;
    --stripe: #1e1e23;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 0 1rem 4rem;
  background: var(--bg);
  color: var(--ink);
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  font-size: 16px;
  line-height: 1.5;
}
main { max-width: 62rem; margin: 0 auto; }
h1 { font-size: 1.6rem; margin: 2rem 0 0.5rem; }
h2 { font-size: 1.2rem; margin: 2.5rem 0 0.5rem; }
h3 { font-size: 1rem; margin: 1.25rem 0 0.35rem; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.95em; }
.corpus, .section {
  border: 1px solid var(--edge);
  border-radius: 6px;
  background: var(--surface);
  padding: 0.75rem 1rem 1rem;
  margin: 1.25rem 0;
}
.section.refusal { border-left: 6px solid var(--refusal); }
.section.claim { border-left: 6px solid var(--bar); }
.part { margin: 0.65rem 0; }
.part .label {
  margin: 0 0 0.15rem;
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--muted);
}
.part .body p { margin: 0.15rem 0; }
.banner { margin: 0.25rem 0 0.75rem; padding: 0.5rem 0.65rem; border-radius: 4px; }
.refusal-banner { background: var(--refusal); color: var(--bg); font-weight: 600; }
.callout .banner { background: var(--accent); color: var(--bg); font-weight: 600; }
.slug { margin: 0 0 0.5rem; color: var(--muted); font-size: 0.9rem; }
.callout { border: 2px solid var(--accent); border-radius: 6px; padding: 0.25rem 0.75rem 0.75rem; }
.callout p { margin: 0.35rem 0; }
.chart { margin: 0.85rem 0 0; padding: 0.6rem 0.7rem; background: var(--bg); border-radius: 4px; }
.chart .drawing { display: block; width: 100%; height: auto; }
.chart figcaption { margin-top: 0.55rem; color: var(--muted); font-size: 0.88rem; }
.chart-absent figcaption { color: var(--ink); font-weight: 600; }
.drawing .row-label, .drawing .group-label, .drawing .value { fill: var(--ink); font-size: 13px; }
.drawing .row-count, .drawing .tick { fill: var(--muted); font-size: 12px; }
.drawing .group-label { font-weight: 700; }
.drawing .value { font-weight: 700; }
.drawing .bar { fill: var(--bar); }
.drawing .hatch-bg { fill: var(--excluded); }
.drawing .hatch-line { stroke: var(--stripe); stroke-width: 2.5; }
.drawing .axis { stroke: var(--track); stroke-width: 1.5; }
.drawing .grid { stroke: var(--edge); stroke-width: 1; }
@media print {
  body { padding: 0; }
  .corpus, .section { break-inside: avoid; }
}
"""
