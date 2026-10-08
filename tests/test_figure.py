"""That a figure cannot be built without the two objects that make it readable.

A number is not the deliverable. The deliverable is a number with the claim that
says what it means and the read that says whether it is whole, and the types exist
so that holding one without the other is not a mistake anybody has to remember not
to make. So most of this module is about the *refusals*: a figure with no claim, a
rate over no population, a rate above 1, a distribution that cannot say what it
excluded, a finding that cannot say what it cannot confirm.

:meth:`~tenbin.measures.base.Figure.rate_text` gets the most attention because it
is the method the whole design exists to make unavoidable. It takes no
caller-supplied wording, and it reads the snapshot rather than a boolean -- so a
renderer cannot print a number without the object that says whether the number is
whole, and cannot print it without first having been told whether the walk
finished. The four completeness outcomes are pinned separately, because the two that
look alike -- a walk stopped by the offset ceiling and a walk that failed -- are
the two a reader would most want conflated.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from tenbin.claims.gate import ClaimGate
from tenbin.claims.mechanism import Granularity
from tenbin.claims.model import Claim, ClaimKind, Denominator
from tenbin.corpus.snapshot import Completeness, Truncation
from tenbin.measures import (
    CountFigure,
    DistributionFigure,
    Figure,
    FigureGroup,
    Finding,
    Measure,
    Rate,
    RateFigure,
    duration_bucket,
    duration_buckets,
    histogram,
)
from tenbin.measures.completeness import CompletenessMeasure
from tenbin.store.client import StoreUnavailableError
from tests.fixtures import build_record, build_snapshot, truncating_snapshot

#: A claim that is valid in every field, so a test can invalidate exactly one thing
#: and be certain the ``ValueError`` came from the field under test rather than from
#: a blank one the author forgot.
_CLAIM = Claim(
    slug="a-claim-under-test",
    statement="The records enumerated on this read are distributed as the figure below shows.",
    does_not_mean="That they are the records anyone cared to enumerate.",
    falsifier="A record in the corpus that the figure does not account for.",
    denominator=Denominator("records in this read", 3),
    kind=ClaimKind.descriptive,
    granularity=Granularity.corpus,
)

_SNAPSHOT = build_snapshot([build_record(pr=index + 1) for index in range(3)])


def _rate_figure(*, numerator: int, size: int, snapshot=None) -> RateFigure:
    """A rate figure over a denominator of a given size, valid in every other field."""
    return RateFigure(
        claim=_CLAIM,
        snapshot=_SNAPSHOT if snapshot is None else snapshot,
        title="A rate under test",
        rate=Rate(numerator=numerator, denominator=Denominator("records under test", size)),
    )


# -- the figure's own guards --------------------------------------------------


def test_a_figure_cannot_be_built_without_a_claim_because_a_number_with_no_claim_is_a_claim() -> (
    None
):
    """The refusal this whole program exists to make, stated at the level of the type.

    ``None`` is deliberately past the annotation. A caller assembling a figure from
    a computed value and a lookup will do it, and the answer has to name the field
    rather than raise something from inside the check that reads as a defect in the
    checker.
    """
    with pytest.raises(ValueError, match="Figure.claim is required"):
        Figure(claim=None, snapshot=_SNAPSHOT, title="A figure with no claim")  # type: ignore[arg-type]


def test_a_figure_cannot_have_a_blank_title_because_a_titleless_figure_is_labelled_by_whoever_renders_it() -> (
    None
):
    """A blank title is not a formatting detail.

    A distribution of review verdicts rendered under no title is read as a
    distribution of review *quality*, and the reader has no way to notice the
    substitution. ``""`` satisfies a required argument in the sense that it was
    passed, and it is a label that says nothing.
    """
    with pytest.raises(ValueError, match="Figure.title must not be blank"):
        Figure(claim=_CLAIM, snapshot=_SNAPSHOT, title="   ")


def test_a_figure_cannot_be_built_without_a_snapshot_because_a_number_needs_to_know_whether_it_is_whole() -> (
    None
):
    """The snapshot is not a convenience field and the check says so.

    :meth:`~tenbin.measures.base.Figure.rate_text` reads it, so a figure whose
    snapshot is ``None`` fails deep inside the one method that makes the figure
    safe -- which is the worst place to fail, because the caller has usually already
    printed the number by then.
    """
    with pytest.raises(ValueError, match="Figure.snapshot is required"):
        Figure(claim=_CLAIM, snapshot=None, title="A figure with no read")  # type: ignore[arg-type]


# -- the method no caller can forget -------------------------------------------


def test_a_whole_corpus_says_so_because_a_reader_assuming_coverage_is_the_default_mistake() -> None:
    """Completeness is stated even when there is nothing wrong, and that is the point.

    A caveat that only appears when something is broken is a caveat readers learn
    to skim, because most figures have nothing wrong. So a figure over a whole
    corpus carries a sentence saying it is a whole corpus -- and the sentence is
    the same one every time, from a named constant, so no renderer can soften it.
    """
    assert _rate_figure(numerator=3, size=3).rate_text() == (
        "Whole read: this figure covers every record example-collection enumerated on this "
        "read, and its own count agreed with the walk."
    )


def test_a_truncated_read_names_the_shortfall_because_an_undercount_looks_exactly_like_a_count() -> (
    None
):
    """The truncation sentence says how many records were never reached.

    A histogram over four hundred documents and one over four thousand are the same
    figure to a reader who is not told otherwise, and the number never reached is
    the part that decides which of the two it is. Naming the offset too gives a
    reader the number they would need to continue the enumeration rather than start
    it again.
    """
    figure = DistributionFigure(
        claim=_CLAIM,
        snapshot=truncating_snapshot([build_record(pr=1)], missing=1234, offset=10_000),
        title="A distribution over a prefix",
        values={"acme/widget": 1},
        excluded={},
    )
    rendered = figure.rate_text()
    assert "Truncated read" in rendered
    assert "1,234 records were not reached" in rendered
    assert "offset 10,000" in rendered


def test_an_offset_cap_with_no_truncation_detail_does_not_invent_a_hole_size_because_a_guess_is_a_finding() -> (
    None
):
    """The contradiction is rendered as the contradiction, not as a number.

    An ``OFFSET_CAP`` snapshot with no truncation detail is a state the walk never
    produces, so a measure reaching it has a caller bug or a fixture bug. The
    completeness statement still has to be true, and it is -- what it must not do
    is fabricate a count so the sentence reads smoothly.
    """
    figure = DistributionFigure(
        claim=_CLAIM,
        snapshot=build_snapshot(
            [build_record(pr=1)],
            completeness=Completeness.OFFSET_CAP,
        ),
        title="A distribution over a prefix with no detail",
        values={},
        excluded={},
    )
    assert "Truncated read" in figure.rate_text()
    assert "not recorded on this snapshot" in figure.rate_text()


def test_a_walk_that_failed_says_so_and_carries_the_exception_because_the_reason_is_the_actionable_part() -> (
    None
):
    """A failed walk is a prefix with a cause attached, and it renders differently.

    A walk stopped by the offset ceiling has a known length; a walk that failed has
    an exception. One sentence for both would tell a reader the shortfall is
    quantified in the case where it is not, and the difference between "we read 400
    and there are 4,000" and "we read 400 and then the store died" is the difference
    between a size problem and an outage.

    The exception is carried on the snapshot rather than logged, so the sentence
    quotes it: a report listing every measure it hit needs the reason in the figure
    rather than in a log somebody has to find.
    """
    without_reason = CountFigure(
        claim=_CLAIM,
        snapshot=build_snapshot(
            [build_record(pr=1)],
            completeness=Completeness.STORE_ERROR,
        ),
        title="A count taken from a walk that failed",
        value=1,
    )
    assert "did not finish" in without_reason.rate_text()
    assert "The snapshot does not carry the failure" in without_reason.rate_text()

    with_reason = CountFigure(
        claim=_CLAIM,
        snapshot=build_snapshot(
            [build_record(pr=1)],
            completeness=Completeness.STORE_ERROR,
            error=StoreUnavailableError("the store went away mid-walk"),
        ),
        title="A count taken from a walk that failed with a reason",
        value=1,
    )
    assert "the store went away mid-walk" in with_reason.rate_text()


def test_a_count_that_contradicts_its_own_listing_says_the_corpus_has_no_agreed_size() -> None:
    """``STORE_TOTAL_EXCEEDED`` is a statement about the count, and it is the loudest one.

    A count that disagrees with its own listing is not a count of this corpus, and
    that includes the number saying how far the walk got. Every figure computed from
    such a read inherits the problem, so the sentence is unconditional and names
    the consequence rather than the enum name.
    """
    figure = CountFigure(
        claim=_CLAIM,
        snapshot=build_snapshot(
            [build_record(pr=1)],
            store_total=0,
            completeness=Completeness.STORE_TOTAL_EXCEEDED,
        ),
        title="A count over a read whose count is not a count",
        value=1,
    )
    assert "no agreed size" in figure.rate_text()


# -- rates --------------------------------------------------------------------


def test_a_rate_over_zero_records_is_refused_because_an_empty_population_is_a_count_not_a_division() -> (
    None
):
    """The zero-over-zero case stops here rather than rendering a percentage.

    ``Denominator`` refuses to be built at size 0, so this guard is unreachable
    through the type and the test has to defeat the inner guard to reach the outer
    one -- which is the point of writing it. A future refactor that made the
    denominator a plain integer would take the inner guard with it, and without this
    check the same defect would surface as a ``ZeroDivisionError`` inside
    :meth:`~tenbin.measures.base.Rate.rate_text`, at render time, in a report
    rather than at construction.
    """
    empty = Denominator("records under test", 1)
    object.__setattr__(empty, "size", 0)
    with pytest.raises(ValueError, match="RateFigure.rate.denominator.size must be at least 1"):
        RateFigure(
            claim=_CLAIM,
            snapshot=_SNAPSHOT,
            title="A rate over nothing",
            rate=Rate(numerator=0, denominator=empty),
        )


def test_a_rate_above_one_is_refused_because_it_is_either_a_unit_error_or_a_bug() -> None:
    """A numerator above its denominator never reaches a report.

    ``183.3%`` renders as a finding and is read as one, and the two ways to get
    there -- counting reviewers over a count of changes, and a snapshot whose
    ``STORE_TOTAL_EXCEEDED`` made the store's own count smaller than what it served
    -- are both defects that look like results.
    """
    with pytest.raises(ValueError, match="exceeds its denominator"):
        _rate_figure(numerator=5, size=3)


def test_a_rate_renders_its_population_in_the_same_string_because_a_rate_without_its_denominator_reads_as_general() -> (
    None
):
    """The fraction and the population are one line, and a bare percentage is not.

    ``62.5%`` is a claim about everything; ``5 of 8 records in this read that state
    a model`` is a claim about eight records. The rate type has no percentage field
    and no rounding flag precisely so a caller cannot print the first without the
    second, and the value is rendered as a fraction of the same population rather
    than as a percentage a reader might carry into a sentence.
    """
    rendered = _rate_figure(numerator=2, size=4).value_text()
    assert rendered == "2 of records under test (4 records) = 50.0%"


# -- distributions --------------------------------------------------------------


def test_a_distribution_can_state_what_it_excluded_because_a_bucket_that_did_not_occur_is_not_a_bucket_that_was_filtered() -> (
    None
):
    """The distinction the whole package turns on, asserted on the type that carries it.

    The excluded mapping is not optional and neither is any of its keys, so a
    measure that filters records has nowhere to put the count except beside the
    buckets it did count. There is no version of this figure in which "0 excluded"
    is a thing a caller has to remember to mean.
    """
    figure = DistributionFigure(
        claim=_CLAIM,
        snapshot=_SNAPSHOT,
        title="A distribution with exclusions",
        values={"acme/widget": 2, "acme/gadget": 1},
        excluded={"no repository on the record": 1, "no timestamp on the record": 2},
    )
    assert figure.counted == 3
    assert figure.excluded_total == 3
    rendered = figure.value_text()
    assert "acme/widget: 2" in rendered
    assert "excluded:" in rendered
    assert "  no repository on the record: 1" in rendered


def test_a_distribution_refuses_a_negative_count_because_that_many_things_did_not_happen() -> None:
    """A negative bucket is arithmetically impossible and renders as a number.

    Nothing stops a measure computing ``counted - retained`` and getting the
    argument order wrong, and the result would appear in a report next to buckets
    that are fine. The check is here rather than in each measure so it runs once.
    """
    with pytest.raises(ValueError, match="must not be negative"):
        DistributionFigure(
            claim=_CLAIM,
            snapshot=_SNAPSHOT,
            title="A distribution with a negative bucket",
            values={"acme/widget": -1},
            excluded={},
        )


def test_a_distribution_is_frozen_because_a_figure_is_a_statement_about_a_read_and_not_a_workspace() -> (
    None
):
    """Editing a figure's counts after the fact would be reporting a different read.

    The snapshot is immutable for the same reason and the rule is the same: a
    figure is a statement about a moment, and a caller holding one has to be able to
    treat it as the statement rather than as a container.
    """
    figure = DistributionFigure(
        claim=_CLAIM,
        snapshot=_SNAPSHOT,
        title="A frozen distribution",
        values={"a": 1},
        excluded={},
    )
    with pytest.raises(TypeError):
        figure.values["b"] = 2  # type: ignore[index]


# -- findings -------------------------------------------------------------------


def test_a_finding_cannot_have_a_blank_cannot_confirm_because_that_is_what_stops_it_being_a_confident_wrong_answer() -> (
    None
):
    """The required field is the one that makes a finding an observation.

    A detector that reads a store and does not run the code that writes it has
    candidates, not a diagnosis. If the author of a finding cannot write what they
    cannot confirm, they have either confirmed something or decided to imply it --
    and to a reader those are the same sentence.
    """
    for blank in ("", "   "):
        with pytest.raises(ValueError, match="Finding.cannot_confirm must not be blank"):
            Finding(
                claim=_CLAIM,
                snapshot=_SNAPSHOT,
                title="A finding with nothing to say about its own limits",
                what_was_observed="Something did not happen.",
                suspected_cause="Something caused it.",
                where="acme/widget",
                cannot_confirm=blank,
            )


def test_a_finding_renders_what_it_cannot_confirm_last_because_it_is_the_easiest_line_to_skip() -> (
    None
):
    """Ordering is a decision, and here it puts the limit where a hurried reader hits it.

    The observed shape comes first because it is the finding, then where, then the
    suspected cause, then the limit. A limit rendered first is one nobody reads; one
    rendered last under a heading of its own is the final thing before the reader
    decides what to do.
    """
    finding = Finding(
        claim=_CLAIM,
        snapshot=_SNAPSHOT,
        title="A finding under test",
        what_was_observed="Ten transitions opened a change and none closed one.",
        suspected_cause="A cause, named as a hypothesis.",
        where="acme/widget",
        cannot_confirm="Tenbin read a store and did not run Kojutsu's code.",
    )
    lines = finding.value_text().splitlines()
    assert lines[0].startswith("Observed: ")
    assert lines[-1].startswith("Cannot confirm: ")
    assert "Where: acme/widget" in lines


def test_a_finding_with_no_cause_and_no_location_renders_without_empty_headings() -> None:
    """``None`` says "none"; ``""`` would render a paragraph with a blank heading.

    A finding sometimes has no suspected cause and no single place -- a corpus
    spread evenly across thirty repositories has no where -- and the rendering has
    to survive that without a line reading ``Suspected cause:`` followed by
    nothing, which reads as a cause that was withheld rather than one that does not
    exist.
    """
    finding = Finding(
        claim=_CLAIM,
        snapshot=_SNAPSHOT,
        title="A finding with nowhere to point",
        what_was_observed="Ten opens and no closes.",
        suspected_cause=None,
        where=None,
        cannot_confirm="Tenbin read a store and did not run Kojutsu's code.",
    )
    assert "Suspected cause" not in finding.value_text()
    assert "Where" not in finding.value_text()


# -- groups and the protocol ------------------------------------------------------


def test_an_empty_figure_group_is_refused_because_a_group_with_no_figures_is_a_bare_figure() -> (
    None
):
    """A measure that answers in three parts must answer in three parts.

    A ``FigureGroup`` exists so a report iterating measures cannot silently drop
    the last two of them. Allowing an empty one would put that back: the type would
    be constructible with nothing in it, and the code that flattens it would have
    nothing to flatten rather than an error.
    """
    with pytest.raises(ValueError, match="must not be empty"):
        FigureGroup(claim=_CLAIM, snapshot=_SNAPSHOT, title="An empty group", figures=())


def test_a_figure_group_holds_only_figures_because_a_payload_a_renderer_cannot_recognise_is_one_it_will_not_render() -> (
    None
):
    """The containment check is about the report, not about the type system.

    ``tuple`` is covariant, so a ``tuple[Figure, ...]`` annotation is satisfied by a
    tuple of anything at all, and a group holding a string would reach a renderer
    that branches on the four payload kinds and has no branch for it. An explicit
    check turns that into a sentence at construction.
    """
    with pytest.raises(ValueError, match="must hold figures"):
        FigureGroup(
            claim=_CLAIM,
            snapshot=_SNAPSHOT,
            title="A group holding something else",
            figures=("not a figure",),  # type: ignore[arg-type]
        )


def test_the_measure_protocol_is_structural_so_a_measure_can_be_a_plain_function_with_metadata() -> (
    None
):
    """A measure is anything with the three members; no class to inherit from.

    Two reasons that are one reason. A measure written for a single run should not
    have to be a subclass, and a measure the corpus cannot support should live in
    the refusal catalogue as a value with a slug rather than as a subclass whose
    constructor raises -- which is a thing a report has to catch and a thing nobody
    remembers to catch. A test that binds a plain object to the protocol is the only
    way to know the protocol was not quietly grown a base class.
    """

    class FunctionMeasure:
        """A measure built out of a closure, the shape the protocol exists to allow."""

        def __init__(self) -> None:
            self.slug = "a-closure-measure"
            self._claim = _CLAIM

        def claim(self, snapshot) -> Claim:
            return self._claim

        def compute(self, snapshot):
            return CountFigure(
                claim=self._claim, snapshot=snapshot, title="A closure's figure", value=1
            )

    measure: Measure = FunctionMeasure()
    figure = measure.compute(_SNAPSHOT)
    assert measure.slug == "a-closure-measure"
    assert isinstance(figure, CountFigure)


def test_a_measure_over_a_read_sends_its_figures_through_the_gate_on_the_way_out() -> None:
    """The figures here are descriptive, and the gate is what keeps them that way.

    Every claim in this package is built with ``ClaimKind.descriptive`` and at or
    above the ``team`` floor, so the gate permits all of them. The test is on that
    property rather than on any one claim: a measure that reached for
    ``ClaimKind.comparative`` to make a sentence sound stronger would render
    something this package refuses, and the gate is the thing that says no.
    """
    figure = CompletenessMeasure().compute(_SNAPSHOT)
    refusal = ClaimGate().check(figure.claim)
    assert refusal is None
    assert figure.claim.granularity is Granularity.corpus


# -- the shared distribution helpers -------------------------------------------------


def test_a_histogram_sorts_by_count_then_by_key_so_two_histograms_over_equal_data_are_equal() -> (
    None
):
    """A test that can assert on a histogram is a test that does not eyeball an order.

    Two buckets with the same count arriving in different orders render differently
    and compare unequal, which is a flaky test when the order comes from a set and a
    spurious diff when it comes from a report. The secondary sort by key is what
    makes two equal histograms equal.
    """
    one = histogram(["b", "a", "a", "c", "c", "c"])
    two = histogram(["c", "c", "c", "a", "a", "b"])
    assert one == two
    assert list(one) == ["c", "a", "b"]


def test_a_histogram_knows_nothing_about_the_values_it_counts_so_it_cannot_rank_them() -> None:
    """Buckets come out in count order and never in the order the vocabulary implies.

    An ordinal axis -- independence, most obviously -- invites exactly the ranking
    this program refuses, and the temptation is strongest precisely where a rank
    exists in the data. The helper that builds every distribution here has no
    knowledge of its values, so the ordering cannot be the vocabulary's, and the
    test says so on the helper rather than on any one caller.
    """
    values = ["independent", "self_certified", "model_separated", "independent"]
    assert list(histogram(values)) == ["independent", "model_separated", "self_certified"]


def test_a_duration_buckets_into_a_ladder_a_reader_already_has_words_for() -> None:
    """The edges are legible rather than statistical, and they are a fixed ladder.

    A finer ladder invites a comparison between two buckets that differ by a bucket
    width, and a bespoke ladder per measure means two measures cannot be compared
    at all. Every duration in this program goes through these boundaries, so a day
    is the same length in both the revision figure and the time-to-review figure.
    """
    assert duration_bucket(0.5) == "under 1 hour"
    assert duration_bucket(1.0) == "1h to 4h"
    assert duration_bucket(24.0) == "1d to 3d"
    assert duration_bucket(10_000.0) == "over 30 days"


def test_a_negative_interval_gets_its_own_bucket_because_clock_skew_is_not_a_very_short_interval() -> (
    None
):
    """The impossible interval is reported as the data defect it is.

    Two writers with their clocks out of order produce a negative gap, and folding
    it into the smallest bucket would report a broken clock as the fastest revision
    in the corpus. It gets a bucket of its own, worded so a report rendering it
    tells the reader which of the two things happened.
    """
    assert duration_bucket(-1.0) == "negative interval (clocks out of order)"


def test_every_duration_bucket_is_present_even_when_nothing_landed_in_it_because_an_absent_rung_is_not_an_empty_one() -> (
    None
):
    """A stable shape is what makes two reads comparable at all.

    The same argument :mod:`tenbin.corpus.provenance` makes about anomaly kinds: a
    distribution whose rungs come and go with the corpus is one every report has to
    guard against, and a reader comparing two runs needs the empty rungs to be
    *there* so they can see they were empty rather than infer it from an absence.
    """
    counts = duration_buckets([0.5])
    assert counts["over 30 days"] == 0
    assert counts["under 1 hour"] == 1
    assert sum(counts.values()) == 1


def test_a_truncation_object_the_walk_produced_is_carried_verbatim_into_the_figure_text() -> None:
    """The figure renders the walk's own numbers rather than recomputing them.

    A figure that derived the missing count from ``store_total - enumerated`` would
    give a different answer from the walk on exactly the snapshots where they
    disagree -- which is the case a reader most needs the walk's own number. So the
    truncation object is read, not recomputed, and this test holds a hand-built one
    to prove it.
    """
    from tenbin.corpus.snapshot import CorpusSnapshot

    snapshot = CorpusSnapshot(
        records=(_SNAPSHOT.records[0],),
        collection="chronicler-real",
        read_at=datetime(2026, 9, 30, tzinfo=UTC),
        store_total=500,
        enumerated=1,
        completeness=Completeness.OFFSET_CAP,
        truncation=Truncation(
            offset_reached=10_000,
            records_missing=499,
            boundary_repositories={"acme/widget": 1},
            boundary_months={"2026-09": 1},
        ),
    )
    figure = CountFigure(claim=_CLAIM, snapshot=snapshot, title="A count over a prefix", value=1)
    assert "499 records were not reached" in figure.rate_text()


# -- the smaller surfaces a renderer touches --------------------------------------


def test_a_bare_figure_renders_its_title_and_nothing_else_because_a_claim_with_no_number_is_a_real_thing_to_want() -> (
    None
):
    """The base class is constructible and renders, rather than being abstract.

    A measure whose answer is that the number does not exist still has a claim to
    print, and forcing it to invent a number to satisfy an abstract method would be
    the wrong pressure. So the base ``value_text`` is empty and honest, and the
    completeness sentence is the only thing ``rate_text`` adds.
    """
    figure = Figure(claim=_CLAIM, snapshot=_SNAPSHOT, title="A claim with no number")
    assert figure.value_text() == ""
    assert figure.rate_text().startswith("Whole read")
    assert figure.payload == "figure"


def test_a_figure_is_hashable_and_two_figures_over_the_same_read_are_the_same_figure() -> None:
    """A report putting figures in a set has to be able to, and to get one answer.

    A frozen dataclass holding a mapping claims to be hashable and is not, so the
    hash is defined on the claim, the read and the payload kind -- the three things
    that make a figure the figure it is. Two distributions computed the same way
    over the same read compare and hash alike, which is what lets a test assert
    purity with ``==`` at all.
    """
    one = DistributionFigure(
        claim=_CLAIM, snapshot=_SNAPSHOT, title="A distribution", values={"a": 1}, excluded={}
    )
    two = DistributionFigure(
        claim=_CLAIM, snapshot=_SNAPSHOT, title="A distribution", values={"a": 1}, excluded={}
    )
    other = DistributionFigure(
        claim=_CLAIM, snapshot=_SNAPSHOT, title="A distribution", values={"a": 2}, excluded={}
    )
    assert hash(one) == hash(two)
    assert len({one, two, other}) == 2

    # Every payload kind, because each dataclass writes its own generated hash and
    # the base's is shadowed in all of them. One kind tested is three untested.
    for figure in (
        Figure(claim=_CLAIM, snapshot=_SNAPSHOT, title="A bare figure"),
        one,
        CountFigure(claim=_CLAIM, snapshot=_SNAPSHOT, title="c", value=1),
        _rate_figure(numerator=1, size=2),
        Finding(
            claim=_CLAIM,
            snapshot=_SNAPSHOT,
            title="f",
            what_was_observed="o",
            suspected_cause=None,
            where=None,
            cannot_confirm="Tenbin read a store.",
        ),
        FigureGroup(
            claim=_CLAIM,
            snapshot=_SNAPSHOT,
            title="g",
            figures=(CountFigure(claim=_CLAIM, snapshot=_SNAPSHOT, title="c", value=1),),
        ),
    ):
        assert isinstance(hash(figure), int)


def test_a_rate_figure_cannot_carry_no_rate_because_a_figure_labelled_a_rate_with_no_rate_in_it_has_no_population() -> (
    None
):
    """The payload is checked as well as the three shared fields.

    ``rate`` is annotated, so this is past the type -- and that is the point of the
    test. A measure assembling a figure from a computed value and a lookup will do
    it, and the answer has to name the field rather than raise
    ``AttributeError`` from inside :meth:`RateFigure.value_text` at render time.
    """
    with pytest.raises(ValueError, match="RateFigure.rate is required"):
        RateFigure(claim=_CLAIM, snapshot=_SNAPSHOT, title="A rate with no rate", rate=None)  # type: ignore[arg-type]


def test_a_distribution_refuses_a_non_integer_count_because_a_fractional_number_of_records_is_a_unit_error() -> (
    None
):
    """Counts are integers, and a float in a bucket renders as a number in a report.

    The check runs once, here, rather than in each of the ten measures that build a
    figure -- and a measure that computed ``counted / total`` into a bucket would
    otherwise put ``0.75`` beside three integers and look like a proportion.
    """
    with pytest.raises(ValueError, match="must be an integer count"):
        DistributionFigure(
            claim=_CLAIM,
            snapshot=_SNAPSHOT,
            title="A distribution of fractions",
            values={"a": 0.5},
            excluded={},  # type: ignore[dict-item]
        )
    with pytest.raises(ValueError, match="must be an integer count"):
        DistributionFigure(
            claim=_CLAIM,
            snapshot=_SNAPSHOT,
            title="A distribution of booleans",
            values={"a": True},  # type: ignore[dict-item]
            excluded={},
        )


def test_a_count_figure_renders_with_separators_because_a_bare_four_digit_number_reads_as_small() -> (
    None
):
    """Formatting, and the reason for it.

    ``1240`` read against a description of unknown scope invites a reader to treat
    it as small, and the separator is the part of the rendering that says it is
    four figures long. Same reasoning as the denominator's own rendering.
    """
    figure = CountFigure(claim=_CLAIM, snapshot=_SNAPSHOT, title="A count", value=1_240_500)
    assert figure.value_text() == "1,240,500"


def test_a_figure_group_renders_one_block_per_member_because_a_report_showing_the_group_shows_all_of_it() -> (
    None
):
    """The flattening, so a report iterating measures does not have to know the shape.

    A group exists so the second and third figures of a measure cannot be dropped
    by a caller that renders whatever ``compute`` returned. Rendering them as one
    block each -- title, then payload -- is what makes that containment visible in
    the output rather than only in the type.
    """
    group = FigureGroup(
        claim=_CLAIM,
        snapshot=_SNAPSHOT,
        title="Two parts",
        figures=(
            CountFigure(claim=_CLAIM, snapshot=_SNAPSHOT, title="First", value=1),
            CountFigure(claim=_CLAIM, snapshot=_SNAPSHOT, title="Second", value=2),
        ),
    )
    assert group.value_text() == "First\n1\nSecond\n2"
    assert group.payload == "group"
