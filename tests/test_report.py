"""The report layer's claim: a broken report cannot be built, and a refusal renders.

Two properties are tested here and they are different in kind. The first is about the
*types*: a section with neither a figure nor a refusal, a figure without its claim, a
refusal carrying the claim it refuses, a header whose arithmetic contradicts the walk
it describes. Each of those is a document that renders wrong in a way nobody can see
in the document -- a blank heading, a number with no statement, a rejection printed
next to the argument it rejected -- so each has to fail at construction instead.

The second is about *assembly*: what :func:`build_report` does with the measures it is
given. The interesting assertion in this module is not that a figure appears but that
a refusal does, as content. A report layer that raised on an unsupported claim would be
a program that can describe this corpus only when asked the questions it can answer,
and the measures this corpus forecloses are the ones somebody came for.

Figures here are the real classes from :mod:`tenbin.measures.base` rather than stubs.
A fake figure would let this module pass against a report layer that mangled the
completeness sentence, because the fake would carry whatever text the test gave it --
so the completeness assertions are made against the sentence the figure really renders,
and they are the reason the truncated read in this module is truncated rather than
merely described as such.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any

import pytest

from tenbin.claims.gate import ClaimGate
from tenbin.claims.mechanism import Granularity
from tenbin.claims.model import Claim, ClaimKind, Denominator
from tenbin.claims.refusals import NO_TICKET_UNBLOCKS_THIS
from tenbin.claims.registry import RefusalRegistry, default_registry
from tenbin.corpus.snapshot import Completeness, CorpusSnapshot, Truncation
from tenbin.measures.base import CountFigure, Figure, Finding, Rate, RateFigure
from tenbin.measures.completeness import READ_COMPLETENESS_SLUG, CompletenessMeasure
from tenbin.report.base import Section
from tenbin.report.build import build_report
from tenbin.report.document import CorpusHeader, Report
from tenbin.report.json_ import render_json, report_as_dict
from tenbin.report.markdown import render_markdown
from tenbin.report.ordering import Part, PartKind, ordered_parts

#: A fixed moment, so a rendered report is byte-identical between runs and between
#: machines. The report layer has no clock and this fixture must not smuggle one in.
READ_AT = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)

#: The corpus every fixture in this module reads from. A collection name is not
#: decoration: a figure over the wrong collection is a real number about a corpus
#: nobody asked for, and the header is the only thing that says which one this is.
COLLECTION = "chronicler-real"


def whole_read(*, enumerated: int = 500) -> CorpusSnapshot:
    """A complete read of a corpus, built by hand rather than walked.

    ``records`` is empty on purpose: the report layer reads the header and the figures
    and never the records, so a fixture that held five hundred of them would be
    asserting nothing a fixture holding none does not. That is also why these tests do
    not need a store -- there is no seam to fake, because nothing here opens one.
    """
    return CorpusSnapshot(
        records=(),
        collection=COLLECTION,
        read_at=READ_AT,
        store_total=enumerated,
        enumerated=enumerated,
        completeness=Completeness.COMPLETE,
    )


def truncated_read(*, enumerated: int = 10_000, store_total: int = 25_000) -> CorpusSnapshot:
    """A read stopped by the store's offset ceiling, with the hole sized and located.

    Both numbers matter to the assertions below: ``records_missing`` is what tells a
    reader every count is an undercount, and ``offset_reached`` is what tells them
    where the enumeration would resume. A fixture with only the first would let a
    header that reported the shortfall and not the position pass.
    """
    return CorpusSnapshot(
        records=(),
        collection=COLLECTION,
        read_at=READ_AT,
        store_total=store_total,
        enumerated=enumerated,
        completeness=Completeness.OFFSET_CAP,
        truncation=Truncation(
            offset_reached=enumerated,
            records_missing=store_total - enumerated,
            boundary_repositories=MappingProxyType({"acme/widget": 500}),
            boundary_months=MappingProxyType({"2026-09": 500}),
        ),
    )


def a_claim(**overrides: Any) -> Claim:
    """A descriptive claim whose four fields say nothing about ordering.

    The wording is constrained on purpose. ``does_not_mean`` and ``falsifier`` below
    contain neither the phrase a renderer prints for the other field nor the other's
    label, because the ordering assertions in this suite are made by index and an
    earlier occurrence of a marker phrase inside a claim would be found first. A real
    claim is free to contain those words; the fixture cannot be, or it would be
    asserting something about its own prose.
    """
    fields: dict[str, Any] = {
        "slug": "capture-rate-over-observed-changes",
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

    Built from the real class so that ``rate_text`` is the sentence this package
    actually renders: the completeness half of the figure part comes from there, and a
    stub would have made that half whatever the test felt like writing.
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


def a_finding(snapshot: CorpusSnapshot) -> Finding:
    """A claim that the corpus does not look like an honest system produced.

    The real :class:`Finding`, including the required ``cannot_confirm``: a finding
    whose author cannot say what they could not confirm has either diagnosed something
    or decided to imply that they had, and those are the same sentence to a reader.
    """
    return Finding(
        claim=a_claim(slug="corpus.lifecycle_opens_without_closes"),
        snapshot=snapshot,
        title="Pull-request transitions open changes and record no closes",
        what_was_observed="40 transitions opened a change and none closed one.",
        suspected_cause="A capture path that stops before the close is written.",
        where="all 40 opens are in `acme/widget`",
        cannot_confirm="Tenbin read a store and did not execute the writer.",
    )


@dataclass(frozen=True)
class StubMeasure:
    """A measure that returns the figure the test built, so the test states the figure.

    A real measure would have to compute something to be worth calling, and every
    assertion in this module is about what the report layer does with a figure it was
    given. The protocol is satisfied honestly -- ``slug``, ``claim`` and ``compute`` --
    rather than by a stand-in that only happens to have the right methods.
    """

    slug: str
    figure: Figure

    def claim(self, snapshot: CorpusSnapshot) -> Claim:
        """The claim the stubbed figure carries, so the protocol is genuinely met."""
        return self.figure.claim

    def compute(self, snapshot: CorpusSnapshot) -> Figure:
        """The figure this measure was built with. The snapshot is not consulted."""
        return self.figure


def a_count_figure_over(slug: str, snapshot: CorpusSnapshot) -> CountFigure:
    """A count for an arbitrary slug, so a test can name a measure the catalogue refuses."""
    return CountFigure(
        claim=a_claim(slug=slug),
        snapshot=snapshot,
        title="Knowledge records with a single custodian",
        value=1,
    )


def a_report(sections: tuple[Section, ...], snapshot: CorpusSnapshot | None = None) -> Report:
    """A report over ``sections``, with a whole read unless another one is given."""
    return Report(
        corpus=CorpusHeader.from_snapshot(snapshot or whole_read()),
        sections=sections,
    )


def bullet_bodies(rendered: str) -> list[str]:
    """The text of every Markdown bullet, with the continuation indent taken back off.

    A part's text may span several lines -- a distribution's payload does -- and the
    renderer indents the continuation lines because Markdown ends a list item at the
    first unindented line. That is a layout requirement, not a change of content, so
    the comparison the two renderers get is made against the un-indented text: it is
    what lets a test say the two carry *the same* words rather than words that differ
    by two spaces.
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


def test_a_section_with_neither_a_figure_nor_a_refusal_cannot_be_built_because_a_blank_heading_is_a_hole_in_a_report() -> (
    None
):
    """The zero case is refused, and the message says what is missing.

    A section with nothing in it renders as a heading with nothing under it, which reads
    as an absence of findings rather than as a measure that never ran. Those are the two
    most different things a report can say and they look the same, which is the reason
    the constructor is where this is caught.
    """
    with pytest.raises(ValueError, match="needs a figure or a refusal"):
        Section(claim=a_claim(), figure=None, refusal=None)


def test_a_section_cannot_hold_a_figure_and_the_refusal_that_would_replace_it_because_a_reader_cannot_guess_which_one_was_meant() -> (
    None
):
    """The two case is refused, for the same reason as the zero case.

    A figure beside the refusal that would have replaced it is a number and an argument
    against that number in one section. A reader has no way to know which of them this
    program is standing behind, and the temptation is to average them.
    """
    refusal = default_registry().all()[0]
    with pytest.raises(ValueError, match="cannot hold a figure and the refusal"):
        Section(claim=a_claim(), figure=a_rate_figure(whole_read()), refusal=refusal)


def test_a_section_holding_a_figure_must_hold_its_claim_because_a_number_without_a_claim_attached_is_a_claim() -> (
    None
):
    """The figure cannot be separated from the thing that says what it means.

    This is the project's central claim, applied to the document rather than to the
    claim object: if a section can be built with a figure and no caveats, then the
    number is one constructor call away from a report and every other constructor in
    the package stops mattering.
    """
    with pytest.raises(ValueError, match="must hold its claim"):
        Section(claim=None, figure=a_rate_figure(whole_read()), refusal=None)


def test_a_section_holding_a_refusal_must_not_hold_the_claim_it_refuses_because_that_would_print_the_rejected_argument() -> (
    None
):
    """A refusal renders the refusal, and the claim stays out of it.

    The gate refused the claim, so printing the claim in the same section would put
    back the argument the gate took out -- and a reader who saw the claim and the
    refusal together would have to decide which one the program meant, which is the
    ambiguity the gate exists to end.
    """
    refusal = default_registry().all()[0]
    with pytest.raises(ValueError, match="must not hold the claim it refuses"):
        Section(claim=a_claim(), figure=None, refusal=refusal)


def test_the_corpus_header_is_copied_out_of_the_snapshot_because_a_header_written_by_hand_is_a_header_that_can_disagree_with_the_walk() -> (
    None
):
    """Every value in the header is the snapshot's, in both directions.

    Asserted against the snapshot rather than against literals, because a test that
    hard-codes the numbers would keep passing if the factory copied the wrong field --
    ``enumerated`` and ``store_total`` are adjacent and both integers.
    """
    snapshot = truncated_read()
    header = CorpusHeader.from_snapshot(snapshot)

    assert header.collection == snapshot.collection
    assert header.read_at == snapshot.read_at
    assert header.enumerated == snapshot.enumerated
    assert header.store_total == snapshot.store_total
    assert header.completeness is snapshot.completeness
    assert header.truncation == snapshot.truncation


def test_a_header_whose_truncation_contradicts_its_own_counts_cannot_be_built_because_a_transcribed_offset_is_a_number_nobody_read() -> (
    None
):
    """The one combination that cannot come out of a real snapshot is refused.

    A truncation is derived from the store's count and what the walk enumerated, so a
    truncation whose ``records_missing`` disagrees with those two was typed rather than
    observed. It would invalidate every count in the report below it rather than just
    its own line, which is why it is refused at construction instead of rendered.
    """
    with pytest.raises(ValueError, match="nobody read"):
        CorpusHeader(
            collection=COLLECTION,
            read_at=READ_AT,
            enumerated=10_000,
            store_total=25_000,
            completeness=Completeness.OFFSET_CAP,
            truncation=Truncation(
                offset_reached=10_000,
                records_missing=999,  # the counts leave 15,000
                boundary_repositories=MappingProxyType({}),
                boundary_months=MappingProxyType({}),
            ),
        )


def test_a_truncated_read_names_what_it_missed_and_where_it_stopped_because_a_prefix_read_like_a_whole_one_is_the_most_confident_wrong_number_available() -> (
    None
):
    """The header says the size of the hole and the offset the walk would resume from.

    Both, and the fixture asserts both: a header that reported the shortfall without the
    offset would leave a reader unable to continue the enumeration rather than restart
    it, and one that reported the offset without the shortfall would not have told them
    the counts are undercounts.
    """
    rendered = render_markdown(a_report((), truncated_read()))

    assert "15,000 records were not reached" in rendered
    assert "offset 10,000" in rendered
    assert Completeness.OFFSET_CAP.value in rendered
    assert "prefix of the corpus" in rendered


def test_a_part_may_not_be_blank_because_a_blank_line_where_a_caveat_goes_is_the_defect_this_package_exists_to_stop() -> (
    None
):
    """A part is checked for blankness, so the defect cannot reach a renderer.

    A blank caveat under a number looks like a rendering bug and reads like a claim with
    no caveat, which is the confusion this package exists to prevent. The check is the
    package's own validator rather than a local ``if not text``, so a part and a claim
    cannot disagree about what counts as blank.
    """
    with pytest.raises(ValueError, match="Part.text must not be blank"):
        Part(kind=PartKind.falsifier, text="   ")


def test_a_figure_reaches_the_report_through_its_own_rate_text_because_the_report_layer_must_not_reconstruct_what_the_figure_knows_about_its_read() -> (
    None
):
    """The completeness sentence in the report is the figure's, verbatim.

    The report layer holds a snapshot and a figure and could in principle summarise the
    read itself. It does not, and this is the test that says so: the sentence asserted
    is the one :meth:`Figure.rate_text` returns for this exact snapshot, so a report
    layer that rebuilt the verdict would have to match a second implementation of it to
    pass -- and a divergence between the two would show up here as a different string.
    """
    snapshot = truncated_read()
    figure = a_rate_figure(snapshot)
    rendered = render_markdown(
        a_report((Section(claim=figure.claim, figure=figure, refusal=None),))
    )

    assert figure.rate_text() in rendered
    assert figure.value_text() in rendered


def test_build_report_renders_a_refused_claim_instead_of_raising_because_a_refusal_is_content_and_not_an_exception() -> (
    None
):
    """A comparative claim reaches the report as a refusal, and its figure does not.

    The corpus holds no assignment mechanism, so this claim cannot be rendered -- and
    the report still builds. The figure is dropped rather than printed beside the
    refusal because the figure is the thing being refused: a report that showed the
    number and the objection would be asking the reader to prefer one.
    """
    comparative = a_claim(
        slug="agent-adoption-rate-gap",
        kind=ClaimKind.comparative,
        granularity=Granularity.team,
    )
    figure = a_rate_figure(whole_read(), claim=comparative)
    report = build_report(
        whole_read(),
        gate=ClaimGate(),
        registry=RefusalRegistry(),
        measures=(StubMeasure(slug=comparative.slug, figure=figure),),
    )

    (section,) = report.sections
    assert section.figure is None
    assert section.refusal is not None
    assert section.slug == "agent-adoption-rate-gap"
    assert figure.rate_text() not in render_markdown(report)
    assert "assignment mechanism" in render_markdown(report)


def test_build_report_renders_every_registered_refusal_because_a_report_that_stopped_at_the_first_would_read_as_though_the_rest_were_permitted() -> (
    None
):
    """All nine catalogue entries appear, and a permissive measure beside them.

    Rendering one refusal and stopping is the failure mode the gate's split between
    ``check`` and ``require`` exists to prevent: the reader sees one refusal and no
    statement about the other eight, which reads as permission. The figures are asserted
    present too, because a report of nothing but refusals would satisfy this test.
    """
    registry = default_registry()
    report = build_report(
        whole_read(),
        gate=ClaimGate(),
        registry=registry,
        measures=(CompletenessMeasure(),),
    )

    refusals = [section for section in report.sections if section.refusal is not None]
    assert [section.slug for section in refusals] == [r.claim_slug for r in registry.all()]
    assert len(report.sections) == len(registry.all()) + 1


def test_build_report_puts_a_finding_before_the_numbers_computed_from_the_corpus_it_questions() -> (
    None
):
    """The finding is first even though the registry listed the number first.

    A finding says the corpus is not what an honest system produces, and every other
    number is computed from that corpus. The measure order is deliberately the wrong way
    round in this test: if the finding followed the registry rather than leading it, the
    report would hand a reader a distribution before telling them the corpus behind it
    is suspect.
    """
    snapshot = whole_read()
    number = a_rate_figure(snapshot)
    report = build_report(
        snapshot,
        gate=ClaimGate(),
        registry=RefusalRegistry(),
        measures=(
            StubMeasure(slug="capture-rate-over-observed-changes", figure=number),
            StubMeasure(slug="corpus.lifecycle_opens_without_closes", figure=a_finding(snapshot)),
        ),
    )

    assert [section.slug for section in report.sections] == [
        "corpus.lifecycle_opens_without_closes",
        "capture-rate-over-observed-changes",
    ]


def test_build_report_puts_the_completeness_figure_before_every_other_number_because_the_others_assume_it() -> (
    None
):
    """Completeness is second, by slug, whatever the registry says.

    Every other figure is conditional on this one and the condition is invisible in the
    numbers -- a distribution over four hundred documents looks like one over four
    thousand. Identifying it by the slug the measures package publishes rather than by
    position is what makes the rule survive somebody prepending a measure.
    """
    snapshot = whole_read(enumerated=500)
    report = build_report(
        snapshot,
        gate=ClaimGate(),
        registry=RefusalRegistry(),
        measures=(
            StubMeasure(slug="capture-rate-over-observed-changes", figure=a_rate_figure(snapshot)),
            CompletenessMeasure(),
        ),
    )

    slugs = [section.slug for section in report.sections]
    assert slugs.index(READ_COMPLETENESS_SLUG) == 0
    assert slugs == [READ_COMPLETENESS_SLUG, "capture-rate-over-observed-changes"]


def test_a_refusal_renders_the_reason_and_no_figure_or_denominator_because_a_stub_reads_as_a_result() -> (
    None
):
    """A refused section carries the refusal and none of a claim section's fields.

    The three claims made here are the ones a stub would get wrong: that a refusal
    section has no figure and no denominator, and that the reason is rendered verbatim
    rather than summarised. A refusal that rendered "no data available" in place of its
    reason would pass a test that only checked for the word "refused".
    """
    refusal = default_registry().get("coverage-of-development")
    rendered = render_markdown(
        a_report((Section(claim=None, figure=None, refusal=refusal),)),
    )

    assert refusal.reason in rendered
    assert refusal.missing_fact in rendered  # type: ignore[operator]
    for label in ("**Statement:**", "**Figure:**", "**Denominator:**"):
        assert label not in rendered


def test_a_refusal_with_no_ticket_says_so_in_words_because_silence_looks_like_nobody_having_looked() -> (
    None
):
    """``None`` renders as the sentence, and the JSON keeps the ``null``.

    The two renderings answer the same question differently on purpose. The sentence is
    what a reader needs; the ``null`` is what a machine can act on, and it is the only
    way to tell "nothing unblocks this and that is correct" from "nobody recorded what
    unblocks this".
    """
    refusal = default_registry().get("whether-agentic-development-is-more-effective")
    assert refusal.unblocked_by is None

    report = a_report((Section(claim=None, figure=None, refusal=refusal),))
    (section,) = report_as_dict(report)["sections"]

    assert NO_TICKET_UNBLOCKS_THIS in render_markdown(report)
    assert section["unblocking"] == NO_TICKET_UNBLOCKS_THIS
    assert section["unblocked_by"] is None
    assert section["missing_fact"] is not None


def test_a_refusal_that_a_ticket_unblocks_names_the_ticket_in_both_renderings_because_a_reader_wants_to_file_it() -> (
    None
):
    """The unblocking ticket is the part somebody acts on, so it is in both forms.

    The contrast with the refusal above is the point: ``None`` and a ticket id are
    different facts and both render, so a reader who wants to unblock one measure and a
    reader who wants to stop waiting for a ticket are both answered.
    """
    # ``AZ8T5XYS`` landed and that refusal was narrowed to the outstanding population
    # the projection deliberately omits, so it no longer names a ticket. The fixture is
    # therefore whichever refusal still does, because a ticket is the shape under test.
    refusal = default_registry().get("coverage-of-development")
    assert refusal.unblocked_by == "EB6FE5ZP"

    report = a_report((Section(claim=None, figure=None, refusal=refusal),))
    (section,) = report_as_dict(report)["sections"]

    assert section["unblocking"] == "EB6FE5ZP"
    assert section["unblocked_by"] == "EB6FE5ZP"
    assert "EB6FE5ZP" in render_markdown(report)


def test_the_json_carries_the_same_four_claim_fields_as_the_markdown_because_a_machine_given_only_a_number_has_been_handed_the_output_this_program_refuses() -> (
    None
):
    """The field set is exactly the parts plus the typed figure, in both renderings.

    A consumer given ``{"figure": "..."}`` puts that string on a wallboard and drops the
    rest, so the payload has to make the caveats the cheap thing to keep. The field set
    is asserted as an equality rather than as a containment, so a field added for
    convenience and a field dropped by a refactor both fail here. ``figure_data``
    carries the same numbers as the ``figure`` text, keyed so a script need not
    string-parse prose.
    """
    snapshot = truncated_read()
    figure = a_rate_figure(snapshot)
    section = Section(claim=figure.claim, figure=figure, refusal=None)
    report = a_report((section,))

    (payload,) = report_as_dict(report)["sections"]
    assert set(payload) == {
        "slug",
        "kind",
        "title",
        "claim_kind",
        "granularity",
        "source",
        "figure_data",
        *(part.kind.value for part in ordered_parts(section)),
    }
    assert payload["figure_data"] == figure.value_data()
    assert payload["figure_data"]["type"] == "rate"
    assert payload["figure_data"]["numerator"] == 412
    assert payload["figure_data"]["denominator_size"] == 500
    rendered = render_markdown(report)
    expected = [part.text for part in report.corpus.parts()]
    expected.extend(part.text for part in ordered_parts(section))
    assert bullet_bodies(rendered) == expected
    assert payload["statement"] == figure.claim.statement


def test_both_renderings_are_pure_functions_of_a_report_because_a_report_that_changed_between_renders_could_not_be_checked() -> (
    None
):
    """Two renders of one report are byte-identical, and so are two reports of one read.

    Nothing in this layer opens a socket or reads a clock, so the only way a second
    render could differ is if a renderer had state. The moment in the header comes from
    the snapshot rather than from ``now()``, which is the version of this bug that would
    otherwise pass unnoticed: a report that says "today" cannot be reproduced and a
    reader cannot tell which read it describes.
    """
    snapshot = truncated_read()
    first = build_report(
        snapshot, gate=ClaimGate(), registry=default_registry(), measures=(CompletenessMeasure(),)
    )
    second = build_report(
        snapshot, gate=ClaimGate(), registry=default_registry(), measures=(CompletenessMeasure(),)
    )

    assert first == second
    assert render_markdown(first) == render_markdown(second)
    assert render_json(first) == render_json(second)
    assert json.loads(render_json(first)) == report_as_dict(first)
    assert READ_AT.isoformat() in render_markdown(first)


def test_a_report_refuses_two_sections_for_one_claim_because_a_claim_with_two_figures_is_a_contradiction() -> (
    None
):
    """One claim, one section, even when two figures were handed in.

    Two sections for one claim is a summary with repetition if the figures agree and a
    contradiction if they do not, and a reader has no way to tell which case they are
    in. The refusal is at construction because the alternative is a report that renders
    both and looks like an answer.
    """
    snapshot = whole_read()
    figure = a_rate_figure(snapshot)
    section = Section(claim=figure.claim, figure=figure, refusal=None)

    with pytest.raises(ValueError, match="two sections for"):
        Report(
            corpus=CorpusHeader.from_snapshot(snapshot),
            sections=(section, Section(claim=figure.claim, figure=figure, refusal=None)),
        )


def test_a_measure_computing_for_a_slug_the_catalogue_refuses_renders_that_refusal_because_an_appendix_cannot_stop_the_number_above_it() -> (
    None
):
    """The registry is load-bearing for the figures, not only for the appendix.

    ``docs/seam.md`` records the failure this prevents: a report claiming a weaker
    measure "while wearing the same name" as one the corpus forecloses. A measure that
    computed something for a refused slug is rendering that weaker measure, so the
    catalogue entry has to win at the figure rather than appearing further down the
    report as an objection to a number already printed.
    """
    refusal = default_registry().get("coverage-of-development")
    figure = a_count_figure_over(refusal.claim_slug, whole_read())
    report = build_report(
        whole_read(),
        gate=ClaimGate(),
        registry=default_registry(),
        measures=(StubMeasure(slug=refusal.claim_slug, figure=figure),),
    )

    rendered = render_markdown(report)
    assert refusal.reason in rendered
    assert figure.title not in rendered, (
        "a figure the catalogue refuses must not be printed at all, not printed and then "
        "objected to"
    )
    assert [section.slug for section in report.sections].count(refusal.claim_slug) == 1
    assert all(section.figure is None for section in report.sections), (
        "the only measure was the refused one, so the whole report should be refusals"
    )


def test_the_markdown_summary_lists_every_section_in_order_because_a_reader_skims_before_they_read() -> (
    None
):
    """The summary is one table row per section, in report order, ahead of the sections.

    A thirty-section report with no entry point is read linearly or not at all.
    The table carries the slug (the identifier to cite), the heading, the first
    payload line, and the read's own completeness word -- and a refusal reads as
    one, so a refused row cannot be mistaken for a result at skim speed.
    """
    snapshot = whole_read()
    figure = a_rate_figure(snapshot)
    refusal = default_registry().get("coverage-of-development")
    report = a_report(
        (
            Section(claim=figure.claim, figure=figure, refusal=None),
            Section(claim=None, figure=None, refusal=refusal),
        )
    )
    rendered = render_markdown(report)

    assert "## Summary" in rendered
    assert rendered.index("## Summary") < rendered.index(f"## {figure.title}")
    rows = [line for line in rendered.splitlines() if line.startswith("| `")]
    assert [row.split("`")[1] for row in rows] == [figure.claim.slug, refusal.claim_slug]
    assert "Refused" in rows[1]
    assert Completeness.COMPLETE.value in rows[0]
    assert figure.value_text().splitlines()[0] in rows[0]


def test_a_whole_read_keeps_a_short_pointer_in_markdown_and_the_full_sentence_in_json() -> (
    None
):
    """The repeated paragraph is shortened in exactly one renderer.

    Thirty identical completeness paragraphs teach a reader to skip them, which
    is where the truncated read that matters stops being seen. The Markdown
    figure part keeps a short pointer to the header that already states the
    verdict; the JSON keeps the full sentence, so a machine matching on words
    sees no change.
    """
    from tenbin.report.markdown import WHOLE_READ_SHORT

    snapshot = whole_read()
    figure = a_rate_figure(snapshot)
    report = a_report((Section(claim=figure.claim, figure=figure, refusal=None),))
    rendered = render_markdown(report)
    (payload,) = report_as_dict(report)["sections"]

    assert WHOLE_READ_SHORT in rendered
    assert figure.rate_text() not in rendered
    assert figure.value_text() in rendered
    assert payload["figure"].splitlines()[0] == figure.rate_text()


def test_a_distribution_states_its_totals_because_buckets_without_a_total_are_mental_arithmetic() -> (
    None
):
    """The totals line rides with the buckets, in prose and in values.

    A reader handed twelve bucket counts and no total does the addition
    themselves or does not do it; a script handed only the sentence parses it.
    Both get the numbers: the last prose line totals counted and excluded, and
    the typed payload carries the mappings with the same two totals.
    """
    from tenbin.measures.base import DistributionFigure

    snapshot = whole_read()
    figure = DistributionFigure(
        claim=a_claim(),
        snapshot=snapshot,
        title="A distribution with exclusions",
        values={"b": 2, "a": 1},
        excluded={"no timestamp on the record": 4},
    )
    lines = figure.value_text().splitlines()

    assert lines[-1] == "Counted: 3. Excluded: 4."
    data = figure.value_data()
    assert data["type"] == "distribution"
    assert data["values"] == {"b": 2, "a": 1}
    assert data["excluded"] == {"no timestamp on the record": 4}
    assert data["counted"] == 3
    assert data["excluded_total"] == 4
    assert data["counted"] == sum(data["values"].values())

    report = a_report((Section(claim=figure.claim, figure=figure, refusal=None),))
    (payload,) = report_as_dict(report)["sections"]
    assert payload["figure_data"] == data
    assert "Counted: 3. Excluded: 4." in payload["figure"]


def test_a_refusal_carries_no_figure_data_because_there_is_no_figure_to_type() -> None:
    """Typed payloads exist for claim sections only.

    A refusal with a ``figure_data`` of zeros would be the stub this program
    refuses to emit wearing a new key. The key is absent rather than null, so
    a consumer can tell "no figure" from "a figure nobody typed".
    """
    refusal = default_registry().get("coverage-of-development")
    report = a_report((Section(claim=None, figure=None, refusal=refusal),))
    (payload,) = report_as_dict(report)["sections"]

    assert payload["kind"] == "refusal"
    assert "figure_data" not in payload
