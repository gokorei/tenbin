"""The two write paths, joined: how many answers survived, and the ones that did not.

Kojutsu writes the question registry into Tanseki and Kojutsu writes the answers
into Tanseki, and those are two different code paths. A request the registry places at
``answered`` and a capture naming the same ``question_id`` are two statements about one
fact, made by two writers, and until this ticket nothing in Tenbin read both of them.
The join is the measure the corpus inventory called the richest dataset in the project,
and half of it was already being computed in isolation: Tenbin could say how long a
question sat and how many reached each terminal state, and could not say how many of
them produced a record at all -- which is the number a reader actually came for.

**The disagreement is the finding and the rate is the smaller half of it.** The three
ways the two sides fail to correspond are kept apart rather than totalled, because they
have three causes and three remedies and one count is none of them: a registry row
saying ``answered`` with no document behind it, a capture for a question the registry
placed at some other terminal state, and a capture naming a question this read holds no
request for. A measure that reported the ratio alone would have turned a lost answer --
the only copy of a human's answer, by Kojutsu's own phrase for a queued outbox row --
into a percentage. That is the transformation this program exists to refuse, and the
finding is what is left when it is refused.

**The rule the join has to hold is that it joins *between* two populations and never
merges them.** Decision 001 says a question document is never an answer and a question
is never evidence: it sits in its own namespace, carries no independence level, and is
counted apart from captures. So the denominator of every figure here is requests, the
captures are the other side of the join and never a member of the population being
measured, and the exclusions name which population each count belongs to -- because
reconciling a capture count against a request count would be adding two things that are
not the same kind of thing.

**The leak test is the one that cannot be argued with.** Every other test here says a
measure does the right thing; that one says a measure is *unchanged* when question
documents are added, which is the only statement that catches a helper quietly widening
a population. It runs over :func:`tenbin.measures.default_measures` rather than over a
list written here, because a test that passes its own list proves the list it passed --
which is how the lifecycle-shape detector stayed correct, tested, and uncalled for a whole
release. The measures that are *about* questions are exempt by class rather than by
position, so adding a measure to the registry cannot quietly exempt a leaky one.
"""

from __future__ import annotations

from typing import Final

import pytest

from tenbin.claims.gate import ClaimGate
from tenbin.claims.model import ClaimKind
from tenbin.claims.registry import default_registry
from tenbin.corpus.record import Record, RecordKind
from tenbin.corpus.snapshot import CorpusSnapshot
from tenbin.measures import default_measures
from tenbin.measures.base import (
    DistributionFigure,
    Figure,
    FigureGroup,
    Finding,
    Measure,
    RateFigure,
)
from tenbin.measures.completeness import CompletenessMeasure
from tenbin.measures.decision_requests import (
    ANSWERED_CAPTURE_POPULATION,
    ANSWERED_CAPTURE_SLUG,
    NO_CAPTURE_QUESTION_ID_KEY,
    NO_REQUEST_QUESTION_ID_KEY,
    NO_STATUS_KEY,
    NO_TERMINAL_REQUEST_KEY,
    UNBLOCKED_BY,
    UNRECOGNISED_STATUS_PREFIX,
    DecisionRequestLifecycleMeasure,
    DecisionRequestOutcomeMeasure,
    QuestionAnsweredRateMeasure,
    SupersededIntervalMeasure,
    join_bucket_label,
    join_questions_and_captures,
)
from tenbin.measures.filtering import CAPTURE_KINDS, capture_population
from tenbin.measures.projection_safety import (
    QUESTION_CAPTURE_AGREEMENT_CLEAN_SLUG,
    QUESTION_CAPTURE_AGREEMENT_SLUG,
    ProjectionSafetyMeasure,
    QuestionCaptureAgreementMeasure,
    scan_question_capture_agreement,
)
from tests.fixtures import build_record, build_snapshot, decision_request

_RATE = QuestionAnsweredRateMeasure()
_AGREEMENT = QuestionCaptureAgreementMeasure()

#: The measures that are *about* projected questions, so they are supposed to move when
#: question documents appear, plus the one that divides by the store rather than by a
#: population this program chose. Exempt by class and not by name or position, so a
#: registry change cannot quietly widen the exemption list. The two lifecycle measures
#: read questions by design and predate this ticket; they are here for the same reason and
#: are named rather than exempted by "everything with a question in its title".
_ABOUT_QUESTIONS: Final[frozenset[type]] = frozenset(
    {
        QuestionAnsweredRateMeasure,
        QuestionCaptureAgreementMeasure,
        DecisionRequestLifecycleMeasure,
        DecisionRequestOutcomeMeasure,
        SupersededIntervalMeasure,
        ProjectionSafetyMeasure,
        CompletenessMeasure,
    }
)

#: How many measures must produce something over the captures alone for the leak test to
#: have proved anything. A distribution that is empty passes whether the measures filtered
#: or not, so the guard on the guard is a floor rather than a per-measure assertion.
MIN_POPULATED_MEASURES: Final[int] = 8


def _captures() -> tuple[Record, ...]:
    """A corpus with one of every capture kind, so no measure is handed nothing.

    Every field the pre-existing measures read is stated on at least one of these: an
    independence level, a capture source, a commenting account, a declared model, a
    change author, a merge outcome, a review id, a rationale chain, and both a lifecycle
    open and a close. A leak test run over records that stated none of them would pass
    whether the measures filtered or not, because the distributions would be empty either
    way.
    """
    return (
        build_record(
            entry_id="answer-0001",
            pr=1,
            tags=["agent_authored"],
            capture_source="webhook",
            independence="self_certified",
            independence_reason="the author answered their own question",
            comment_author="dev",
            answered_by_agent="answerer",
            answered_by_model="model-a",
            answered_at="2026-01-02T09:00:00+00:00",
            captured_at="2026-01-02T09:00:00+00:00",
            delivery_id="d1",
            change_author_account="dev",
            pr_opened_at="2026-01-01T00:00:00+00:00",
            pr_outcome="merged",
            pr_merged_at="2026-01-03T00:00:00+00:00",
        ),
        build_record(
            entry_id="review-2",
            pr=2,
            record_kind=RecordKind.REVIEW_VERDICT,
            tags=["review", "review_state_changes_requested"],
            capture_source="collect",
            independence="independent",
            comment_author="dana",
            review_id=4242,
            answered_at="2026-02-02T09:00:00+00:00",
            captured_at="2026-02-02T09:00:00+00:00",
            github_comment_id=77,
            change_author_account="someone-else",
            pr_opened_at="2026-02-01T00:00:00+00:00",
            pr_outcome="closed_unmerged",
        ),
        build_record(
            entry_id="review-2-inline",
            pr=2,
            record_kind=RecordKind.INLINE_REVIEW_COMMENT,
            tags=["review", "inline_comment"],
            comment_author="dana",
            review_id=4242,
            answered_at="2026-02-02T09:10:00+00:00",
        ),
        build_record(
            entry_id="pr-event-3",
            pr=3,
            record_kind=RecordKind.PR_LIFECYCLE,
            tags=["pr_state_change", "action_opened"],
            capture_source="webhook",
            captured_at="2026-03-01T00:00:00+00:00",
            delivery_id="d3",
            pr_opened_at="2026-03-01T00:00:00+00:00",
        ),
        build_record(
            entry_id="pr-event-3-closed",
            pr=3,
            record_kind=RecordKind.PR_LIFECYCLE,
            tags=["pr_state_change", "action_closed"],
            capture_source="webhook",
            captured_at="2026-03-09T00:00:00+00:00",
            pr_opened_at="2026-03-01T00:00:00+00:00",
        ),
        build_record(
            entry_id="rationale-v1-0",
            pr=4,
            record_kind=RecordKind.RATIONALE,
            tags=["rationale", "rationale_declared"],
            rationale=True,
            capture_source="asserted",
            declared_by="dev",
            declared_at="2026-04-01T00:00:00+00:00",
            rationale_source="declared",
            rationale_revision=1,
        ),
    )


def _answering(question_id: str, pr: int) -> Record:
    """A capture that names the question it answers, in the shape Kojutsu writes one."""
    return build_record(
        entry_id=f"answer-{pr:04d}",
        pr=pr,
        tags=["agent_authored"],
        capture_source="webhook",
        independence="independent",
        question_id=question_id,
        comment_author="dev",
        answered_at="2026-09-02T00:00:00+00:00",
        captured_at="2026-09-02T00:00:00+00:00",
    )


def _members(figure: Figure) -> tuple[Figure, ...]:
    """The figure and everything under it, so a group's members are compared too."""
    if isinstance(figure, FigureGroup):
        return (figure, *(member for child in figure.figures for member in _members(child)))
    return (figure,)


def _rendered(measure: Measure, snapshot: CorpusSnapshot) -> tuple[tuple[str, ...], ...]:
    """Every number and every population in a measure's output, as comparable text.

    The denominator is rendered rather than read as an integer, because the size alone
    would pass while the *description* beside it drifted -- and a denominator whose
    description has stopped matching its population is the failure this whole program
    exists to make unconstructible.
    """
    return tuple(
        (
            member.payload,
            member.title,
            member.claim.slug,
            member.claim.denominator.render_text(),
            member.value_text(),
        )
        for member in _members(measure.compute(snapshot))
    )


def _rate_and_distribution(figure: Figure) -> tuple[RateFigure, DistributionFigure]:
    """The rate and the bucket distribution of the join, asserting both types on the way out."""
    assert isinstance(figure, FigureGroup)
    distribution, rate = figure.figures
    assert isinstance(distribution, DistributionFigure)
    assert isinstance(rate, RateFigure)
    return rate, distribution


def _agreeing_corpus() -> tuple[Record, ...]:
    """Four terminal requests, every answer that should exist present.

    Two answered and kept, one failed and correctly not kept -- a ``failed`` request is
    Kojutsu reaching its attempt ceiling and there is no answer to keep -- and one
    superseded, which is the same shape for a different reason. Nothing here disagrees,
    which is the state the finding has to be silent in.
    """
    return (
        decision_request("q-1", status="answered", pr=1),
        decision_request("q-2", status="answered", pr=2),
        decision_request("q-3", status="failed", pr=3),
        decision_request("q-4", status="superseded", pr=4, answered_at=None),
        _answering("q-1", 1),
        _answering("q-2", 2),
    )


def _request_without_id(question_id: str, pr: int, *, status: str | None) -> Record:
    """A projected request whose frontmatter carries no ``question_id``.

    Built through :func:`build_record` with a hand-written document id, because the shape
    is the one document the projection cannot produce and a fixture that supplied the key
    would be testing a document the writer never writes. The path segment still says
    ``question`` -- only the frontmatter key is missing -- so it classifies as a request,
    which is the point: it reaches the denominators and is then named as unjoinable.
    """
    return _projected_request(question_id, pr, status=status, writes_question_id=False)


def _projected_request(
    question_id: str,
    pr: int,
    *,
    status: str | None,
    writes_question_id: bool = True,
) -> Record:
    """A projected request with whichever keys the test is about, and none of the others.

    ``writes_question_id=False`` builds the document whose ``question_id`` key is absent
    while its id namespace still says ``question``; ``status=None`` with the id written is
    the other half, a request whose terminal state never reached the store. Both are
    documents Kojutsu's projection cannot produce, and both are here because the join
    has to say which condition each one hit rather than folding them together.
    """
    return build_record(
        entry_id=question_id,
        pr=pr,
        record_kind=None,
        tags=["question"] if status is None else ["question", f"question_{status}"],
        doc_id=f"acme/widget/pr-{pr}/question/{question_id}",
        question_id=question_id if writes_question_id else None,
        question_status=status,
    )


# -- the rate, over questions --------------------------------------------------------


def test_the_share_is_over_questions_that_reached_a_terminal_state_and_never_over_capture_records_because_a_question_is_never_an_answer() -> (
    None
):
    """The denominator is the population the claim names, and it is not the record count.

    The snapshot holds six records: four requests and two captures. A denominator of six
    would be a rate over a population that includes the captures, which is the one
    population a request may not be counted inside -- and a numerator of two over the two
    captures, which is the count the tempting implementation would have produced, renders
    as 100% on a corpus where an answer was lost.
    """
    snapshot = build_snapshot(_agreeing_corpus())

    join = join_questions_and_captures(snapshot.records)
    rate, distribution = _rate_and_distribution(_RATE.compute(snapshot))

    assert join.terminal_questions == 4
    assert join.answered_with_capture == 2
    assert rate.rate.numerator == 2
    assert rate.rate.denominator.size == 4
    assert rate.rate.denominator.description == ANSWERED_CAPTURE_POPULATION
    assert "capture records" not in rate.rate.denominator.description
    assert rate.claim.slug == ANSWERED_CAPTURE_SLUG
    assert rate.claim.unblocked_by == UNBLOCKED_BY == "AZ8T5XYS"
    assert rate.claim.kind is ClaimKind.descriptive
    assert ClaimGate().check(rate.claim) is None
    # Every terminal request is in exactly one bucket, so a reader can add them up.
    assert distribution.counted == rate.rate.denominator.size
    assert distribution.excluded == {}


def test_the_share_is_not_the_share_of_questions_that_were_answered_because_the_two_differ_exactly_by_the_answers_that_were_lost() -> (
    None
):
    """The negative space, checked as arithmetic rather than as prose.

    Two of three requests reached ``answered`` and one of those two has no document. So
    the share that was answered is 2/3 and the share whose answer was kept is 1/3, and the
    gap between the two is precisely the one lost answer. A measure that reported the ratio
    alone would have rendered the loss as 66.7% and called the corpus healthy, which is
    the transformation the finding exists to prevent.
    """
    snapshot = build_snapshot(
        (
            decision_request("q-1", status="answered", pr=1),
            decision_request("q-2", status="answered", pr=2),
            decision_request("q-3", status="superseded", pr=3, answered_at=None),
            _answering("q-1", 1),
        )
    )

    join = join_questions_and_captures(snapshot.records)
    rate, _ = _rate_and_distribution(_RATE.compute(snapshot))

    assert join.answered_with_capture == 1
    assert join.terminal_questions == 3
    assert len(join.answered_without_capture) == 1
    assert rate.rate.value == pytest.approx(1 / 3)
    assert "The share of questions that were answered" in rate.claim.does_not_mean
    assert "QuestionCaptureAgreementMeasure" in rate.claim.does_not_mean


def test_every_terminal_state_appears_at_both_sides_of_the_join_because_a_reader_comparing_two_runs_needs_to_see_that_a_bucket_was_empty() -> (
    None
):
    """All six buckets, always, so the shape does not depend on the corpus.

    Three statuses by two sides, frozen at zero the way
    :func:`~tenbin.measures.base.duration_buckets` freezes its ladder. A distribution
    whose buckets come and go is one every report has to guard against, and ``superseded
    with a capture naming it`` is the bucket a reader most needs to see at zero -- its
    non-zero form is a finding.
    """
    _, distribution = _rate_and_distribution(_RATE.compute(build_snapshot(_agreeing_corpus())))

    frozen = {
        join_bucket_label(status, captured=captured): 0
        for status in ("answered", "failed", "superseded")
        for captured in (True, False)
    }
    assert dict(distribution.values) == {
        **frozen,
        join_bucket_label("answered", captured=True): 2,
        join_bucket_label("failed", captured=False): 1,
        join_bucket_label("superseded", captured=False): 1,
    }


def test_a_capture_naming_an_answered_question_twice_is_still_one_answer_because_the_figure_is_about_questions_and_not_about_capture_records() -> (
    None
):
    """Two knowledge entries can answer one question, and a rate over captures is the bug.

    The numerator is a count of questions, so the set is taken before the count. Counting
    captures would report 2/2 on a corpus where both answers were kept and 1/1 on a corpus
    where one was lost -- two different situations rendering as the same number, which is
    the ``unknown``-bucket defect in a rate.
    """
    snapshot = build_snapshot(
        (
            decision_request("q-1", status="answered", pr=1),
            _answering("q-1", 1),
            build_record(
                entry_id="answer-0002",
                pr=1,
                tags=["agent_authored"],
                capture_source="webhook",
                question_id="q-1",
                comment_author="dana",
            ),
        )
    )

    join = join_questions_and_captures(snapshot.records)

    assert join.answered_with_capture == 1
    assert join.capture_without_terminal == ()
    assert join.unmatched_captures == 0


# -- the three disagreements --------------------------------------------------------


def test_a_registry_row_saying_answered_with_no_document_is_a_finding_because_a_lost_outbox_row_is_the_only_copy_of_a_human_s_answer() -> (
    None
):
    """The expensive kind, and the one a bare count would have turned into a percentage.

    Kojutsu calls a queued row the only copy of a human's answer, so a row recorded and
    never delivered is an answer that exists nowhere else -- and no writer reports it,
    because the writer that lost it is the thing that is broken. The finding names the
    question id so the check is something somebody can go and run, and it says which side
    is short rather than only that the two sides differ.
    """
    snapshot = build_snapshot((decision_request("q-42", status="answered", pr=4),))

    join = join_questions_and_captures(snapshot.records)
    finding = scan_question_capture_agreement(snapshot.records, snapshot=snapshot)

    assert join.answered_without_capture == ("q-42",)
    assert join.answered_with_capture == 0
    assert isinstance(finding, Finding)
    assert finding.claim.slug == QUESTION_CAPTURE_AGREEMENT_SLUG
    assert finding.claim.denominator.description == ANSWERED_CAPTURE_POPULATION
    assert finding.claim.denominator.size == 1
    assert "q-42" in finding.what_was_observed
    assert finding.where is not None
    assert "q-42" in finding.where
    assert "the registry says answered and the corpus has no document for it" in (
        finding.what_was_observed
    )
    assert "only copy of a human's answer" in finding.what_was_observed
    assert "absence of a document" in finding.cannot_confirm
    assert "not from an error report" in finding.cannot_confirm


def test_a_capture_for_a_question_the_registry_failed_or_superseded_is_a_finding_because_the_two_write_paths_disagreeing_is_the_finding() -> (
    None
):
    """The second kind, for both statuses, and the rate is explicitly not the finding.

    A ``failed`` request is Kojutsu reaching its attempt ceiling and a ``superseded``
    one is the code moving first; in both cases the registry says the request resolved
    without an answer and the corpus says an answer exists. ``failed`` is the more
    interesting of the two to a reader watching the unattended loop, because it is one of
    the few places that loop's own behaviour is observable at all. The second half of the
    test is the point: on this corpus the rate measure reports a number and says nothing,
    and a rate of 0/1 renders as 0.0% -- an entirely ordinary-looking corpus.
    """
    for status in ("failed", "superseded"):
        snapshot = build_snapshot(
            (
                decision_request("q-9", status=status, pr=5),
                _answering("q-9", 5),
            )
        )

        join = join_questions_and_captures(snapshot.records)
        finding = scan_question_capture_agreement(snapshot.records, snapshot=snapshot)
        assert join.capture_without_terminal == ("q-9",), status
        assert join.answered_without_capture == (), status
        assert isinstance(finding, Finding), status
        assert "failed or superseded" in finding.what_was_observed, status
        assert finding.where is not None, status
        assert "q-9" in finding.where, status

        # The descriptive half must not answer as a finding: it reports the shape of the
        # corpus and the integrity half reports the shape of the pipeline. Merging them
        # would put a number where the reader needs a cause.
        assert not any(isinstance(member, Finding) for member in _members(_RATE.compute(snapshot)))


def test_a_capture_naming_no_request_in_this_read_is_counted_rather_than_dropped_because_a_bucket_that_was_filtered_out_and_a_bucket_that_did_not_occur_are_different_facts() -> (
    None
):
    """The third kind, counted, reported, and *not* filed as an exclusion.

    A capture naming a question this read holds no request for is a fact about the two
    write paths and the least alarming of the three: decision 001 calls the projection
    eventually consistent, so a read taken while an outbox drains looks exactly like this.
    It has no denominator -- the capture is not a question and the question is not here --
    so it is reported as a count inside the finding and deliberately kept out of the
    figure's ``excluded``, where it would render as a record this program declined to
    count rather than a disagreement that occurred.
    """
    snapshot = build_snapshot((*_agreeing_corpus(), _answering("q-not-here", 99)))

    join = join_questions_and_captures(snapshot.records)
    _, distribution = _rate_and_distribution(_RATE.compute(snapshot))
    finding = scan_question_capture_agreement(snapshot.records, snapshot=snapshot)

    assert join.unmatched_captures == 1
    assert join.disagrees
    assert join.answered_without_capture == ()
    assert isinstance(finding, Finding)
    assert "1 capture record(s) name a question id" in finding.what_was_observed
    # Counted in a report rather than excluded from one, and asserted as an exact mapping
    # so a reworded key cannot quietly absorb it.
    assert dict(distribution.excluded) == {}
    assert distribution.counted == join.terminal_questions == 4


def test_a_clean_join_says_the_check_ran_because_a_reader_who_cannot_tell_checked_from_unchecked_has_an_absence_where_a_fact_belongs() -> (
    None
):
    """Silence on the finding, a count of zero under a different slug on the measure.

    Both halves are needed and they are the split :class:`ProjectionSafetyMeasure` makes.
    The detector returning ``None`` on a clean corpus is what stops the check training its
    readers to ignore it; the measure rendering zero under a slug of its own is what stops
    an unwired check from looking identical to a clean one. Sharing a slug would let a
    reader match the two up and conclude the clean one refutes the finding.
    """
    snapshot = build_snapshot(_agreeing_corpus())

    assert scan_question_capture_agreement(snapshot.records, snapshot=snapshot) is None

    figure = _AGREEMENT.compute(snapshot)
    assert figure.payload == "count"
    assert figure.value_text() == "0"
    assert _AGREEMENT.claim(snapshot).slug == QUESTION_CAPTURE_AGREEMENT_CLEAN_SLUG
    assert QUESTION_CAPTURE_AGREEMENT_CLEAN_SLUG != QUESTION_CAPTURE_AGREEMENT_SLUG


def test_a_disagreement_reaches_a_report_as_a_figure_group_because_a_measure_returning_a_tuple_could_have_its_finding_dropped() -> (
    None
):
    """The shape ``ProjectionSafetyMeasure`` already uses, for the same reason.

    :func:`tenbin.report.build` partitions on figure type to put findings ahead of the
    numbers, so a measure that returned a bare tuple of figures could have the finding
    silently dropped by a report iterating its output. The group keeps it reachable rather
    than merely computed, and the group's claim is the finding's claim so the section that
    renders carries the caveats.
    """
    snapshot = build_snapshot((decision_request("q-42", status="answered", pr=4),))

    figure = _AGREEMENT.compute(snapshot)

    assert isinstance(figure, FigureGroup)
    assert isinstance(figure.figures[0], Finding)
    assert figure.claim.slug == QUESTION_CAPTURE_AGREEMENT_SLUG
    assert figure.claim.denominator.size == 1
    # The claim a report would look up has to be the finding's claim on the firing path
    # and the clean one otherwise, so a section is never headed by a claim that does not
    # describe the figure under it.
    assert _AGREEMENT.claim(snapshot).slug == QUESTION_CAPTURE_AGREEMENT_SLUG


def test_a_corpus_that_lost_a_whole_batch_of_answers_names_ten_of_them_and_counts_the_rest_because_a_censor_is_worse_than_a_list() -> (
    None
):
    """The bound on naming, and the count of what was not named.

    ``MAX_DOCUMENTS_NAMED`` is ten, because a reader has to be able to go and check the
    thing and every one of the rest is findable by re-running the scan. What it must not
    do is drop them: a finding that named ten of twenty-two and said nothing about the
    other twelve has turned a lost batch into a lost batch of ten, which is the same
    under-reporting this whole module exists to avoid, one tier up.
    """
    lost = tuple(
        decision_request(f"q-{index:03d}", status="answered", pr=index + 1) for index in range(22)
    )
    snapshot = build_snapshot(lost)

    finding = scan_question_capture_agreement(snapshot.records, snapshot=snapshot)

    assert isinstance(finding, Finding)
    assert finding.where is not None
    assert "12 more this finding did not name" in finding.where
    assert "22 request(s) reached answered" in finding.what_was_observed


# -- what cannot be joined, and why that is a normal state --------------------------


def test_a_capture_with_no_question_id_is_excluded_under_a_key_naming_its_condition_because_a_collect_capture_has_no_question_to_answer() -> (
    None
):
    """The large, ordinary exclusion, and the key that keeps it from becoming a bucket.

    Kojutsu writes no question id on a capture taken by ``collect``, because there is no
    question behind it, so a corpus captured rather than answered excludes most of itself
    here. That is the corpus telling the truth about itself, so it belongs in ``excluded``
    under a condition phrase and fires no finding. Dropping it would make the rate a
    statement about the captures that happen to name a question, which is a selected
    population wearing a plain sentence.
    """
    snapshot = build_snapshot(
        (
            decision_request("q-1", status="answered", pr=1),
            _answering("q-1", 1),
            build_record(entry_id="answer-0002", pr=2, capture_source="collect"),
            build_record(entry_id="answer-0003", pr=3, capture_source="collect"),
        )
    )

    join = join_questions_and_captures(snapshot.records)
    _, distribution = _rate_and_distribution(_RATE.compute(snapshot))

    assert distribution.excluded[NO_CAPTURE_QUESTION_ID_KEY] == 2
    assert NO_CAPTURE_QUESTION_ID_KEY not in distribution.values
    assert not join.disagrees, "a capture with nothing to join to is not a disagreement"
    assert scan_question_capture_agreement(snapshot.records, snapshot=snapshot) is None
    # The key says which population the count belongs to, because this one is *not*
    # addable to the denominator and a reader who assumed it was would be reconciling two
    # different kinds of record.
    assert "captures" in NO_CAPTURE_QUESTION_ID_KEY


def test_a_request_with_no_question_id_is_excluded_under_its_own_key_because_a_request_we_cannot_join_is_not_a_request_we_counted() -> (
    None
):
    """The other half of the same absence, and this one *is* addable to the denominator.

    Two different populations, so two different keys, each naming which it is. A capture
    cannot be joined because nothing names the question it answers; a request cannot be
    joined because the projection did not carry its identifier. One key would send a reader
    looking in the wrong half of the schema, and it would also quietly make the figure's
    exclusions add up when only one of the two is a request.
    """
    record = _request_without_id("q-77", 7, status="answered")
    assert record.question_id is None
    assert record.is_question

    snapshot = build_snapshot((record,))
    join = join_questions_and_captures(snapshot.records)
    _, distribution = _rate_and_distribution(_RATE.compute(snapshot))

    assert dict(distribution.excluded) == {NO_REQUEST_QUESTION_ID_KEY: 1}
    assert distribution.counted == 0
    assert "requests" in NO_REQUEST_QUESTION_ID_KEY
    assert not join.disagrees


def test_a_capture_naming_a_request_with_no_terminal_status_is_excluded_because_there_are_no_two_sides_to_compare() -> (
    None
):
    """A request whose status never reached the store cannot agree or disagree with a capture.

    It is neither a join nor a disagreement, and it is a *different* gap from a capture
    naming no request: this one has both documents and no status to compare. Naming it in
    ``excluded`` rather than in the finding is what keeps "the two write paths disagree"
    meaning what it says -- a reader seeing it in the finding would go looking for a lost
    row where the only thing missing is a field.
    """
    snapshot = build_snapshot(
        (
            _projected_request("q-88", 8, status=None),
            _answering("q-88", 8),
        )
    )

    join = join_questions_and_captures(snapshot.records)
    _, distribution = _rate_and_distribution(_RATE.compute(snapshot))

    assert dict(distribution.excluded) == {NO_STATUS_KEY: 1, NO_TERMINAL_REQUEST_KEY: 1}
    assert join.terminal_questions == 0
    assert not join.disagrees
    assert scan_question_capture_agreement(snapshot.records, snapshot=snapshot) is None


def test_a_request_with_a_status_outside_kojutsus_three_is_named_under_the_key_the_outcome_measure_names_it_under_because_one_absence_deserves_one_vocabulary() -> (
    None
):
    """A fourth status, carried through as itself and kept out of both sides of the join.

    Folding it into ``failed`` would report an outcome Kojutsu never wrote, and it
    would do it twice: once in the terminal-state distribution and once here. So it is
    excluded under the module's existing ``terminal status outside Kojutsu's three``
    key, which is what lets a reader take the two measures as one account of the same
    corpus rather than as two vocabularies that happen to describe it.
    """
    snapshot = build_snapshot((decision_request("q-9", status="withdrawn", pr=9),))

    join = join_questions_and_captures(snapshot.records)
    _, distribution = _rate_and_distribution(_RATE.compute(snapshot))

    assert dict(distribution.excluded) == {f"{UNRECOGNISED_STATUS_PREFIX}withdrawn": 1}
    assert distribution.counted == 0
    assert join.terminal_questions == 0


def test_two_joins_over_the_same_records_hash_alike_because_their_count_mappings_are_read_off_in_order() -> (
    None
):
    """The generated hash would either raise or compare unequal, and both are silent.

    ``QuestionCaptureJoin`` carries two ``MappingProxyType`` fields, and a dataclass that
    generates ``__hash__`` over them raises at runtime rather than at construction -- the
    same trap :class:`tenbin.corpus.record.Record` documents for itself. The worse of the
    two failures is not the exception: two joins computed the same way over the same
    records must be equal *and* hash alike, or a caller putting them in a set or using
    one as a cache key gets two answers for one fact.
    """
    records = _agreeing_corpus()
    once = join_questions_and_captures(records)
    twice = join_questions_and_captures(records)
    other = join_questions_and_captures((*records, _answering("q-3", 3)))

    assert once == twice
    assert hash(once) == hash(twice)
    assert hash(once) != hash(other), "a different corpus is a different join, not a collision"
    assert len({once, twice, other}) == 2


# -- the rule that has to hold ------------------------------------------------------


def test_a_question_document_is_never_treated_as_independent_evidence_because_nobody_stated_anything_in_one() -> (
    None
):
    """Decision 001's prohibition, asserted as a mechanism and not as a promise.

    The mechanisms it names are a namespace of the request's own and the absence of the
    evidence axis, so the assertion is that both absences are real: the fields are
    ``None``, the kind is not in :data:`~tenbin.measures.filtering.CAPTURE_KINDS`, and
    the measure declares no certainty requirement that would act as a substitute
    threshold. The claims are descriptive over the request population, which is the
    strongest form the requirement can take -- there is no evidence axis on either measure
    for a ``min_independence`` filter to read at all.
    """
    requests = (
        decision_request("q-1", status="answered", pr=1),
        decision_request("q-2", status="failed", pr=2),
    )
    snapshot = build_snapshot((_answering("q-1", 1), *requests))

    assert all(record.independence is None for record in requests)
    assert all(record.capture_source is None for record in requests)
    assert RecordKind.QUESTION not in CAPTURE_KINDS
    assert not any(record in capture_population(requests) for record in requests)
    for measure in (_RATE, _AGREEMENT):
        assert measure.requires_certainty is None
        claim = measure.claim(snapshot)
        assert claim.kind is ClaimKind.descriptive
        assert claim.denominator.description == ANSWERED_CAPTURE_POPULATION


def test_a_question_document_cannot_reach_the_capture_side_of_the_join_because_the_join_reads_across_two_populations_and_never_merges_them() -> (
    None
):
    """The rule as an arithmetic property: adding questions changes nothing on the other side.

    A helper that widened the capture population to include requests would not change the
    rate -- the denominator is requests either way -- but it would change
    ``unmatched_captures``, ``answered_with_capture`` and the exclusion count, and those
    are the numbers a reader acts on. Five request documents added to a corpus of five
    unjoinable captures must leave every capture-side count exactly where it was.
    """
    captures = tuple(
        build_record(entry_id=f"answer-{index:04d}", pr=index + 1, capture_source="collect")
        for index in range(5)
    )
    requests = tuple(decision_request(f"q-{index}", pr=index + 1) for index in range(5))

    without = join_questions_and_captures(captures)
    with_requests = join_questions_and_captures((*captures, *requests))

    assert without.unmatched_captures == 0
    assert without.answered_with_capture == 0
    assert with_requests.unmatched_captures == 0
    assert with_requests.answered_with_capture == 0
    assert with_requests.excluded[NO_CAPTURE_QUESTION_ID_KEY] == 5
    assert with_requests.terminal_questions == 5


def test_no_pre_existing_measure_moves_when_question_documents_are_present_because_the_join_reads_across_the_two_populations_without_merging_them() -> (
    None
):
    """The defect decision 001 exists to prevent, asserted as an absence, over the registry.

    Every other test here says a measure does the right thing. This one says a measure is
    *unchanged*, and unchanged is the only property that catches a helper quietly widening
    a population: a projected request carries a repository and a document update time, so
    without the exclusions it would join the by-repository histogram as a record somebody
    examined and the by-month histogram filed under the day it was written rather than the
    day it was asked.

    Runs over :func:`~tenbin.measures.default_measures` because a list written here would
    only prove the list. The measures that read questions and the read-completeness figure
    are exempt by class, and the class exemption is what stops a sixth measure from
    joining the list quietly.
    """
    captures = _captures()
    requests = (
        decision_request("q-1", status="answered", pr=1),
        decision_request("q-2", status="superseded", pr=3, answered_at=None),
        decision_request("q-3", status="failed", pr=2),
    )
    without = build_snapshot(captures)
    with_requests = build_snapshot((*captures, *requests))

    pre_existing = [
        measure for measure in default_measures() if type(measure) not in _ABOUT_QUESTIONS
    ]
    assert len(pre_existing) >= MIN_POPULATED_MEASURES, (
        "the leak test proved nothing if the registry held no other measure: "
        f"{sorted(measure.slug for measure in pre_existing)}"
    )

    moved = [
        f"{type(measure).__name__} ({measure.slug})"
        for measure in pre_existing
        if _rendered(measure, without) != _rendered(measure, with_requests)
    ]

    assert not moved, (
        "these figures moved when projected decision requests were added to the corpus, "
        "and decision 001 requires them not to: a question document is never an answer, a "
        "question is never evidence, and the join reads across the two populations without "
        "merging them.\n  " + "\n  ".join(moved)
    )


def test_the_corpus_the_leak_test_uses_is_one_where_a_leak_would_be_visible_because_an_empty_distribution_passes_either_way() -> (
    None
):
    """The guard on the guard: the captures state every field the measures read.

    A leak test over records with no independence level, no capture source, no commenting
    account and no declared model would pass whether the measures filtered or not,
    because every distribution would be empty in both runs. This asserts the figures are
    non-empty *without* the requests, so a pass cannot have come from nothing to count.
    """
    without = build_snapshot(_captures())

    populated = {
        measure.slug
        for measure in default_measures()
        if type(measure) not in _ABOUT_QUESTIONS and _rendered(measure, without)
    }

    assert len(populated) >= MIN_POPULATED_MEASURES, f"only {sorted(populated)} produced anything"


# -- slugs and wiring ----------------------------------------------------------------


def test_the_two_measures_publish_distinct_dotted_slugs_and_neither_collides_with_a_refusal_because_a_collision_would_refuse_the_wrong_measure() -> (
    None
):
    """The binding between a measure, its claim and the refusal catalogue.

    A report asks the registry whether a measure is blocked, using the measure's slug. A
    slug colliding with a slugified entry in ``docs/seam.md`` would be refused by a refusal
    filed about something else, and the collision to guard against here is a real one
    rather than a hypothetical: the catalogue's first entry is a refusal about decision
    requests, and it is the entry that would swallow this figure. Both are dotted, like
    every other slug in the package, and each measure's own claim carries its slug so a
    report cannot render one under the other's heading.
    """
    snapshot = build_snapshot(_agreeing_corpus())
    slugs = (_RATE.slug, _AGREEMENT.slug)

    assert len(set(slugs)) == 2
    assert all("." in slug for slug in slugs)
    registry = default_registry()
    for slug in slugs:
        assert slug not in registry, f"{slug} collides with a refusal entry"
    assert _RATE.claim(snapshot).slug == _RATE.slug
    assert _RATE.slug != QUESTION_CAPTURE_AGREEMENT_SLUG


def test_only_the_agreement_measure_is_about_the_capture_system_because_a_description_is_not_an_integrity_statement() -> (
    None
):
    """Opt in, do not default in, and opt in for exactly one of the two.

    A registry row whose answer is not in the store is a statement about the capture path
    rather than about the work it captured, which is the question ``tenbin
    --integrity-only`` exists to answer -- and the flag is opt-in, so a measure that does
    not declare it is treated as being about the work. The rate is a description of the
    corpus; declaring it would put a number in the view whose whole purpose is that the
    reader cannot yet believe the numbers.
    """
    assert _AGREEMENT.about_the_capture_system is True
    assert getattr(_RATE, "about_the_capture_system", False) is False
