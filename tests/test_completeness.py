"""That this report says how much of the store it read, before anything else does.

This is the figure every other figure is conditional on, and the condition is
invisible in the numbers themselves: a distribution over four hundred documents
looks exactly like a distribution over four thousand. So the measure is tested on
the two things a reader cannot see in a percentage, and on the two cases where a
percentage is not available at all.

The load-bearing one is the denominator. A rate of enumerated over enumerated is
always exactly 1, renders as ``100.0%``, and says nothing -- and it is one line of
code away at all times, which is exactly why it is worth a test that says so. The
other is the empty store: a reachable store holding nothing is a *fact* and an
unreachable store is a *failure*, and the two must never produce the same output.
The second is guaranteed upstream by the walk raising, and the tests below pin the
half of it that lives here.
"""

from __future__ import annotations

import pytest

from tenbin.claims.model import ClaimKind, Denominator
from tenbin.corpus.snapshot import Completeness
from tenbin.measures.base import CountFigure, RateFigure
from tenbin.measures.completeness import (
    READ_COMPLETENESS_SLUG,
    STORE_POPULATION,
    CompletenessMeasure,
)
from tests.fixtures import build_record, build_snapshot, truncating_snapshot

_MEASURE = CompletenessMeasure()


def test_a_rate_over_the_enumerated_count_is_always_one_and_says_nothing() -> None:
    """The tautology this measure exists not to publish, asserted on the population.

    ``enumerated / enumerated`` is 1 for every corpus that has ever been read,
    including one that was truncated at the first page, and it renders as
    ``100.0%`` in the same typeface as a real finding. The only thing that
    separates them is the denominator, so the denominator is the whole measure: it
    is the store's own count, which is the one number in the snapshot that this
    program did not produce and can therefore be caught disagreeing with.
    """
    figure = _MEASURE.compute(build_snapshot([build_record(pr=1)]))
    assert isinstance(figure, RateFigure)
    assert figure.rate.denominator.description == STORE_POPULATION
    assert figure.rate.denominator.size == 1

    partial = _MEASURE.compute(
        truncating_snapshot([build_record(pr=1)], missing=999, offset=10_000)
    )
    assert isinstance(partial, RateFigure)
    # A read that reached one of a thousand renders as 0.1%, not 100%.
    assert partial.rate.value == pytest.approx(0.001)
    assert partial.rate.value != 1.0


def test_the_whole_figure_is_the_store_s_own_count_and_nothing_else() -> None:
    """The numerator is what arrived and the denominator is what the store said.

    Both numbers are on the snapshot because they are the two independent
    statements a reader checks against each other, and a completeness measure that
    compared anything else -- records against records, or records against a
    constant -- would be the tautology above wearing a different denominator.
    """
    snapshot = build_snapshot([build_record(pr=index + 1) for index in range(7)])
    figure = _MEASURE.compute(snapshot)
    assert isinstance(figure, RateFigure)
    assert figure.rate.numerator == 7
    assert figure.rate.denominator.size == 7
    assert figure.rate.denominator.render_text() == f"{STORE_POPULATION} (7 records)"


def test_a_reachable_but_empty_store_is_a_count_saying_so_because_a_rate_needs_a_population() -> (
    None
):
    """Zero over zero is not a rate, and rendering it as one would be a division.

    The corpus being empty is a finding an operator can act on -- the collection
    name is wrong, or the capture path wrote somewhere else, or the deployment is
    new. Rendering it as ``0/0`` gives a reader a number to interpret; rendering it
    as a count whose title says the store reports no documents gives them a
    sentence to act on. The distinction the package cares about most is between
    this and an unreachable store, and an unreachable store never reaches here: the
    walk raises rather than returning a snapshot with no records.
    """
    figure = _MEASURE.compute(build_snapshot([]))
    assert isinstance(figure, CountFigure)
    assert figure.value == 0
    assert "no documents" in figure.title
    # And no rate is hiding behind it: the type is the difference.
    assert not isinstance(figure, RateFigure)


def test_a_store_that_served_more_than_it_counted_is_a_count_too_because_a_rate_above_one_is_not_a_rate() -> (
    None
):
    """``STORE_TOTAL_EXCEEDED`` makes the rate impossible, so the figure is a count.

    A count that disagrees with its own listing is not a count of this corpus, and
    the numerator here is larger than the denominator. ``RateFigure`` would refuse
    it -- which is the right behaviour and would make this measure raise on a
    reachable and interesting corpus, so the case is handled rather than guarded
    against. The completeness sentence still fires and is the alarm: a reader is
    told the corpus has no agreed size rather than handed a percentage over one.
    """
    figure = _MEASURE.compute(
        build_snapshot(
            [build_record(pr=index + 1) for index in range(3)],
            store_total=1,
            completeness=Completeness.STORE_TOTAL_EXCEEDED,
        )
    )
    assert isinstance(figure, CountFigure)
    assert figure.value == 3
    assert "no rate to compute" in figure.title
    assert "no agreed size" in figure.rate_text()


def test_the_claim_names_representativeness_as_what_it_does_not_mean_because_completeness_is_not_coverage() -> (
    None
):
    """A 100% complete read says nothing about whether the corpus is representative.

    This is the sentence that keeps the figure from being used for the one thing a
    reader wants it for. The documents enumerated here are the documents somebody
    captured: self-selected, and censored by capture paths that fail silently. A
    repository nobody answers on is byte-identical to a repository with no knowledge
    debt, and there is no backfill path for either -- so a whole corpus can be a
    complete description of the smallest part of development.
    """
    claim = _MEASURE.claim(build_snapshot([build_record(pr=1)]))
    rendered = claim.does_not_mean
    assert "representative of development" in rendered
    assert "self-selected" in rendered
    assert "byte-identical" in rendered
    assert "no backfill path" in rendered
    assert claim.kind is ClaimKind.descriptive


def test_the_falsifier_is_a_change_in_the_store_s_own_total_because_that_is_the_thing_the_snapshot_detects() -> (
    None
):
    """A falsifier has to name an observation, not a sentiment.

    The snapshot detects a change in the store's total between the count and the
    enumeration and reports it as ``STORE_TOTAL_EXCEEDED``, so the claim names that
    same condition. A falsifier phrased as "if the corpus were incomplete" would be
    unfalsifiable -- it restates the claim's own subject -- and the reader would
    have no way to check it.
    """
    claim = _MEASURE.claim(build_snapshot([build_record(pr=1)]))
    assert "store's total" in claim.falsifier
    assert "store_total_exceeded" in claim.falsifier


def test_the_claim_is_built_per_read_because_its_denominator_is_a_fact_about_a_read() -> None:
    """A claim carrying a placeholder population is a rate over a number nobody wrote down.

    ``Denominator`` refuses a size below one, which is what makes the empty-store
    case interesting: the claim still has to be constructible for a store holding
    nothing, so it is built with the count floored at one and the *figure* is the
    count. The point of the test is that the denominator is the store's number and
    not a constant -- two reads of two collections produce two different claims.
    """
    small = _MEASURE.claim(build_snapshot([build_record(pr=1)], store_total=4))
    large = _MEASURE.claim(build_snapshot([build_record(pr=1)], store_total=4000))
    assert isinstance(small.denominator, Denominator)
    assert (small.denominator.size, large.denominator.size) == (4, 4000)
    assert small.slug == large.slug == READ_COMPLETENESS_SLUG


def test_a_truncated_read_still_gets_a_rate_because_the_shortfall_belongs_in_the_completeness_sentence() -> (
    None
):
    """A prefix is still measurable, and the number that says so is not optional.

    Refusing to compute anything over a partial walk would throw away a usable
    figure; publishing it without the truncation would overstate it. Both problems
    are solved by computing the rate and letting
    :meth:`~tenbin.measures.base.Figure.rate_text` name the hole, so the test is
    that the rate is there *and* the sentence is there.
    """
    figure = _MEASURE.compute(truncating_snapshot([build_record(pr=1)], missing=3, offset=500))
    assert isinstance(figure, RateFigure)
    assert figure.rate.value == pytest.approx(0.25)
    assert "Truncated read" in figure.rate_text()
    assert "3 records were not reached" in figure.rate_text()


def test_the_measure_is_pure_so_a_test_needs_no_store_and_two_reads_agree() -> None:
    """No I/O, no clock, no randomness: the same snapshot gives the same figure.

    A measure that read a clock would make its own output untestable, and one that
    reached a store would make the tests depend on a daemon. Both are the reason
    every measure here is a function of a snapshot and the reason the fixtures can
    build one without a fake.
    """
    snapshot = build_snapshot([build_record(pr=index + 1) for index in range(3)])
    assert _MEASURE.compute(snapshot) == _MEASURE.compute(snapshot)
    assert _MEASURE.claim(snapshot) == _MEASURE.claim(snapshot)
