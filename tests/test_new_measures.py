"""The four measures Kojutsu stopped foreclosing, and the one they do not unblock.

``3QRPK52A``, ``9PTQE45Y``, ``QEVSTMYW`` and ``RJNVJK1P`` landed, and four measures
exist because of them. Each is small; the interesting part of each is the population
it refuses. A verdict with no ``review_id`` cannot be paired with the comments of
its own review. A change with captures but no review has an *absent* interval, and
an absent interval written as zero would be the fastest possible review. A login is
not a person. An outcome is what GitHub reported, not whether the change was good.

The one test here that is not about a measure is about decision 002.
``change_author_account`` is a new field, four of these measures are new, and the
refusal of the agentic-effectiveness comparison is the single easiest thing in this
package to lose: a ticket lands, a field appears, and somebody concludes the
question is now answerable. It is not, and the gate still refuses -- asserted here
on a comparative claim built on the new field, so the refusal is tested against the
thing that would tempt somebody to retire it.

The classification tests are here too because ``record_kind`` is what made the
verdict filter possible, and a test of the new field belongs beside the first measure
that depends on it.
"""

from __future__ import annotations

import pytest

from tenbin.claims.gate import ClaimGate
from tenbin.claims.mechanism import Granularity
from tenbin.claims.model import Claim, ClaimKind, Denominator
from tenbin.claims.refusals import ComparativeRefusalError
from tenbin.claims.registry import default_registry
from tenbin.corpus.record import Certainty, Record, RecordKind, classify, record_kind_from
from tenbin.measures.authorship import (
    AUTHORSHIP_SLUG,
    NO_CHANGE_AUTHOR_KEY,
    ChangeAuthorshipMeasure,
)
from tenbin.measures.base import DistributionFigure, Figure, FigureGroup, RateFigure
from tenbin.measures.outcome import (
    NO_OUTCOME_KEY,
    OUTCOME_SLUG,
    PR_OUTCOME_CLOSED_UNMERGED,
    PR_OUTCOME_MERGED,
    UNRECOGNISED_OUTCOME_PREFIX,
    ChangeOutcomeMeasure,
)
from tenbin.measures.review_verdicts import (
    INLINE_COMMENT_KEY,
    NO_REVIEW_ID_KEY,
    NO_REVIEWER_KEY,
    REVIEW_VERDICTS_SLUG,
    ReviewVerdictMeasure,
)
from tenbin.measures.time_to_review import (
    NO_OPEN_TIME_KEY,
    NO_REVIEW_KEY,
    TIME_TO_REVIEW_SLUG,
    TimeToReviewMeasure,
)
from tests.fixtures import build_record, build_snapshot

_VERDICTS = ReviewVerdictMeasure()
_WAITS = TimeToReviewMeasure()
_AUTHORS = ChangeAuthorshipMeasure()
_OUTCOMES = ChangeOutcomeMeasure()


def _verdict(
    *,
    pr: int = 1,
    state: str = "approved",
    review_id: int | None = 4242,
    reviewer: str | None = "dana",
    legacy: bool = False,
    record_kind: str | RecordKind | None = RecordKind.REVIEW_VERDICT,
    answered_at: str | None = "2026-01-02T09:00:00+00:00",
    pr_opened_at: str | None = None,
) -> Record:
    """A review verdict, in the shape Kojutsu writes one since ``RJNVJK1P``."""
    return build_record(
        entry_id=f"review-{pr}",
        pr=pr,
        record_kind=None if legacy else record_kind,
        tags=["review", f"review_state_{state}"],
        review_id=review_id,
        comment_author=reviewer,
        answered_at=answered_at,
        pr_opened_at=pr_opened_at,
    )


def _inline(pr: int = 1, *, review_id: int | None = 4242) -> Record:
    """An inline comment on a line of code, which is not a verdict."""
    return build_record(
        entry_id=f"review-{pr}-inline",
        pr=pr,
        record_kind=RecordKind.INLINE_REVIEW_COMMENT,
        tags=["review", "inline_comment"],
        review_id=review_id,
        comment_author="dana",
        answered_at="2026-01-02T09:00:00+00:00",
    )


def _distribution(figure: Figure, index: int = 0) -> DistributionFigure:
    """The distribution at a position in a group, asserting the type on the way out."""
    assert isinstance(figure, FigureGroup)
    member = figure.figures[index]
    assert isinstance(member, DistributionFigure)
    return member


# -- record_kind, which is what unblocked the verdict filter ---------------------


def test_a_record_carrying_record_kind_is_determined_by_it_because_the_writer_states_the_kind_twice() -> (
    None
):
    """The key is authoritative and the tags are a projection of it, so the key wins.

    Kojutsu derives the tag set from the kind it already holds, so a record
    carrying ``record_kind`` states its kind twice and the two cannot drift. A reader
    that preferred the tags would be preferring a projection over the thing it was
    derived from, and the evidence has to name the key so a reader can see which
    reading was used.
    """
    reading = classify(["agent_authored"], record_kind="review_verdict")
    assert reading.kind is RecordKind.REVIEW_VERDICT
    assert reading.certainty is Certainty.DETERMINED
    assert reading.evidence == ("record_kind", "review_verdict")
    assert "states 'review_verdict'" in reading.detail


def test_a_legacy_record_without_the_key_is_classified_from_its_tags_because_that_is_all_it_has() -> (
    None
):
    """The fallback is a weaker statement and stays available for every old document.

    Documents written before the key existed, and every rationale -- which is
    dispatched through a different payload and never carried it -- have tags and no
    key. The tag reading is unchanged by the new path, which is what makes this
    additive rather than a change of behaviour: a legacy review record still reads
    as a determined review verdict from its ``review`` tag.
    """
    reading = classify(["review", "review_state_approved"])
    assert reading.kind is RecordKind.REVIEW_VERDICT
    assert reading.certainty is Certainty.DETERMINED
    assert reading.evidence == ("review", "review_state_approved")
    assert record_kind_from(None) is None


def test_an_answer_with_no_tags_is_inferred_because_nothing_wrote_its_kind_down() -> None:
    """The one case that rests on an absence, and it is still the answer case.

    A record with no ``record_kind`` and no tags is an answer written by somebody
    who named no agent -- or a record of any other kind that lost its tags. Calling
    that determined would put a conclusion about a corpus in the corpus's own
    mouth, which is exactly what the key was added to stop.
    """
    reading = classify(None)
    assert reading.kind is RecordKind.ANSWER
    assert reading.certainty is Certainty.INFERRED
    assert reading.evidence == ("no tags",)


def test_an_unrecognised_record_kind_names_itself_in_the_evidence_because_a_misspelled_key_falls_through_silently() -> (
    None
):
    """A writer that has invented a fifth kind is visible rather than invisible.

    The record still classifies -- from its tags -- and still reaches the
    denominators, because dropping it would make every rate a rate over a population
    this reader happened to understand. But the evidence carries the value nobody
    could read, so a corpus where every record carries a misspelled kind does not
    look like a clean corpus.
    """
    reading = classify(["review", "review_state_approved"], record_kind="review_verdictt")
    assert reading.kind is RecordKind.REVIEW_VERDICT
    assert reading.certainty is Certainty.INFERRED
    assert reading.evidence[0] == "record_kind='review_verdictt'"
    assert record_kind_from("review_verdictt") is None


# -- review verdicts ----------------------------------------------------------------


def test_the_verdict_distribution_refuses_a_kind_the_writer_stated_and_nobody_can_read_because_a_reading_would_be_published_beside_a_name() -> (
    None
):
    """The filter, exercised on the case it exists for.

    A record whose ``record_kind`` is a misspelling of a real one still reads as a
    review verdict from its tags -- the fallback works, and the record is not lost.
    But that reading is *inferred*, and a verdict rate is one of two figures here
    where an inferred kind would end up published next to a reviewer's name. So the
    measure asks for ``DETERMINED``, and this record is excluded: the count would
    otherwise say the writer stated a kind, which is not what happened.
    """
    assert ReviewVerdictMeasure.requires_certainty is Certainty.DETERMINED
    misspelt = _verdict(pr=1, record_kind="review_verdictt")
    assert misspelt.classification.certainty is Certainty.INFERRED
    assert misspelt.classification.kind is RecordKind.REVIEW_VERDICT

    whole = _distribution(_VERDICTS.compute(build_snapshot([misspelt])), 0)
    assert dict(whole.values) == {}
    # It never reaches the pairing, so it is not in the no-review-id exclusion
    # either: the filter is upstream, and a record filtered out is not "a verdict we
    # could not pair".
    assert whole.excluded == {}


def test_a_legacy_verdict_read_from_its_tags_is_still_counted_because_that_reading_is_a_determination() -> (
    None
):
    """The other side of the filter, and the reason it is not stricter.

    Kojutsu derives the tag set from the kind it holds, so a record carrying
    ``review`` and a review-state tag is stating its kind -- through a projection of
    it, but stating it. ``classify`` calls that determined, and the measure counts
    it. The record in the test above carries a key nobody can read, which is a
    different thing from a key that was never written, and the difference is the
    whole of what ``DETERMINED`` means.
    """
    legacy = _verdict(pr=1, legacy=True)
    assert legacy.record_kind is None
    assert legacy.classification.certainty is Certainty.DETERMINED
    whole = _distribution(_VERDICTS.compute(build_snapshot([legacy])), 0)
    assert dict(whole.values) == {"approved": 1}


def test_a_verdict_with_no_review_id_is_excluded_because_it_cannot_be_paired_with_its_own_comments() -> (
    None
):
    """The first of the two exclusions, and the reason it is not a filter argument.

    A review arrives as a verdict plus however many inline comments share its
    ``review_id``. Without the id, a verdict cannot be matched to its own comments,
    and counting it would inflate a reviewer's verdict count with records whose
    relationship to that review is unknown. So it is counted in ``excluded`` under a
    key that says the condition -- a reader can see how much of the review record
    the figure covers.
    """
    records = (_verdict(pr=1), _verdict(pr=2, review_id=None))
    whole = _distribution(_VERDICTS.compute(build_snapshot(records)), 0)
    assert dict(whole.values) == {"approved": 1}
    assert dict(whole.excluded) == {NO_REVIEW_ID_KEY: 1}


def test_inline_comments_are_excluded_because_a_note_about_a_line_of_code_is_not_a_verdict() -> (
    None
):
    """The second exclusion, and it is a vocabulary decision rather than a technicality.

    Kojutsu's review-state vocabulary deliberately leaves ``commented`` out,
    because leaving a note with no verdict is feedback rather than an approval.
    Folding inline comments into a verdict distribution would inflate an approval
    rate with records where nobody approved anything.
    """
    records = (_verdict(pr=1), _inline(pr=1))
    whole = _distribution(_VERDICTS.compute(build_snapshot(records)), 0)
    assert dict(whole.values) == {"approved": 1}
    assert dict(whole.excluded) == {INLINE_COMMENT_KEY: 1}
    assert "not a verdict" in INLINE_COMMENT_KEY


def test_the_verdict_keys_name_the_reviewer_and_the_state_because_a_bare_account_bucket_hides_the_mix() -> (
    None
):
    """Per reviewer, keyed by account and state together.

    A reviewer's mix is the interesting quantity, and a bucket per account would
    make the reader sum the buckets to recover it. The separator is a character that
    cannot appear in a login, so a reader can split a key back into its two parts
    without the code having to parse anything.
    """
    records = (
        _verdict(pr=1, state="approved", reviewer="dana"),
        _verdict(pr=2, state="approved", reviewer="dana"),
        _verdict(pr=3, state="changes_requested", reviewer="sam"),
    )
    per_reviewer = _distribution(_VERDICTS.compute(build_snapshot(records)), 1)
    assert dict(per_reviewer.values) == {
        "dana — approved": 2,
        "sam — changes_requested": 1,
    }


def test_the_verdict_claim_names_its_ticket_and_says_neither_that_a_verdict_was_right_nor_that_a_rate_is_a_person() -> (
    None
):
    """Two negative spaces, and the second is the one a report is more likely to drop.

    Nothing in the corpus can show a verdict was correct. And a reviewer who
    approved more may simply have been assigned the changes that were going to be
    approved, because who was assigned which change is not in the corpus -- so a
    rate here is a count of what reviewers recorded, and not a property of a
    reviewer. The ticket is on the claim because the field is what made the measure
    possible and a reader asking "what changed" deserves one reference.
    """
    claim = _VERDICTS.claim(build_snapshot([_verdict(pr=1)]))
    assert claim.unblocked_by == "RJNVJK1P"
    assert claim.slug == REVIEW_VERDICTS_SLUG
    assert "no verdict was correct" in claim.does_not_mean or "verdict was correct" in (
        claim.does_not_mean
    )
    assert "property of the reviewer" in claim.does_not_mean
    assert ClaimGate().check(claim) is None


# -- time to review -----------------------------------------------------------------


def test_an_absent_interval_is_not_a_zero_interval_because_a_zero_would_mean_reviewed_instantly() -> (
    None
):
    """The substitution this measure exists to refuse, asserted in both directions.

    A change with captures and no review contributes *nothing* to the distribution,
    and is counted under a key naming the condition. Writing it as zero would be the
    most destructive available mistake: zero is the fastest possible review, so a
    corpus where most changes were never reviewed would render as a corpus where
    most changes were reviewed instantly. The two are not adjacent on the scale,
    they are opposite meanings.
    """
    reviewed = _verdict(
        pr=1,
        review_id=1,
        answered_at="2026-01-01T00:30:00+00:00",
        pr_opened_at="2026-01-01T00:00:00+00:00",
    )
    unreviewed = build_record(pr=2, pr_opened_at="2026-01-01T00:00:00+00:00")
    figure = _WAITS.compute(build_snapshot([reviewed, unreviewed]))
    assert isinstance(figure, DistributionFigure)
    assert figure.values["under 1 hour"] == 1
    assert dict(figure.excluded) == {NO_REVIEW_KEY: 1}
    assert sum(figure.values.values()) == 1


def test_the_wait_is_measured_from_the_change_opening_to_the_earliest_verdict_and_not_to_the_mean_one() -> (
    None
):
    """Earliest, because a change reviewed three times has three waits and one answer.

    The other two are facts about review practice rather than about how long the
    change waited, and averaging them would produce a number describing neither. The
    test builds two verdicts on one change -- hours and two days -- and asserts the
    figure reports the short one.
    """
    opened = "2026-01-01T00:00:00+00:00"
    early = _verdict(
        pr=1, review_id=1, answered_at="2026-01-01T02:00:00+00:00", pr_opened_at=opened
    )
    later = _verdict(
        pr=1, review_id=2, answered_at="2026-01-05T00:00:00+00:00", pr_opened_at=opened
    )
    figure = _WAITS.compute(build_snapshot([early, later]))
    assert isinstance(figure, DistributionFigure)
    # Two verdicts on one change, so one interval: the two-hour one. The four-day
    # one is a fact about the second review rather than about how long the change
    # waited, and it is not in the figure at all.
    assert figure.values["1h to 4h"] == 1
    assert figure.values["3d to 7d"] == 0
    assert sum(figure.values.values()) == 1
    assert figure.claim.denominator.size == 1
    assert figure.excluded == {}


def test_a_change_with_a_review_and_no_open_time_is_excluded_under_its_own_key_because_it_is_a_different_gap() -> (
    None
):
    """One end or the other, and a reader fixing the corpus needs to know which.

    A change with no ``pr_opened_at`` is a document written before the key existed,
    or a payload that carried no creation time. A change with no review is a change
    nobody looked at. Both are counted, separately, because the two have different
    fixes and one key would send somebody to the wrong field.
    """
    records = (_verdict(pr=1, review_id=1),)
    figure = _WAITS.compute(build_snapshot(records))
    assert isinstance(figure, DistributionFigure)
    assert dict(figure.excluded) == {NO_OPEN_TIME_KEY: 1}
    assert figure.claim.denominator.size == 1


def test_the_wait_claim_names_its_ticket_and_says_the_gap_contains_delivery_latency_and_a_human_reading() -> (
    None
):
    """What the interval is not, and the ticket that made it possible to say.

    Between the change opening and a verdict being captured there is the bot's
    delivery, a reviewer reading the change, and a reviewer not looking at all --
    three things, none of them separable here. Before ``9PTQE45Y`` the measure
    could only be computed against ``answered_at``, which is a statement about the
    bot's latency, and the two quantities differ by roughly the bot's own
    turnaround with nothing in the figure to show it.
    """
    claim = _WAITS.claim(build_snapshot([_verdict(pr=1, review_id=1)]))
    assert claim.unblocked_by == "9PTQE45Y"
    assert claim.slug == TIME_TO_REVIEW_SLUG
    assert "Delivery latency" in claim.does_not_mean
    assert "none of them is separable" in claim.does_not_mean


# -- authorship -----------------------------------------------------------------------


def test_records_are_grouped_by_the_login_that_opened_the_change_and_the_absent_ones_are_counted_separately() -> (
    None
):
    """A login, counted, with the records the forge reported no user for named.

    The key is a sentence rather than a placeholder bucket, because a placeholder
    rendered in the same column as the real logins would read as a principal. The
    field is absent when the payload carried no user, and that is the whole fact.
    """
    records = (
        build_record(pr=1, change_author_account="dana"),
        build_record(pr=2, change_author_account="dana"),
        build_record(pr=3, change_author_account="agent-service"),
        build_record(pr=4),
    )
    figure = _AUTHORS.compute(build_snapshot(records))
    assert isinstance(figure, DistributionFigure)
    assert dict(figure.values) == {"dana": 2, "agent-service": 1}
    assert sum(figure.excluded.values()) == 1
    assert NO_CHANGE_AUTHOR_KEY in figure.excluded
    assert NO_CHANGE_AUTHOR_KEY not in figure.values


def test_the_authorship_claim_says_a_login_is_not_a_person_a_bot_or_a_statement_about_what_they_were_doing() -> (
    None
):
    """Kojutsu's own framing, in the three terms they used.

    A paraphrase drops one of the three things it is not, and the dropped one is
    usually the third -- which is the one a reader actually needs, because it is the
    inference they were about to make.
    """
    claim = _AUTHORS.claim(build_snapshot([build_record(pr=1, change_author_account="dana")]))
    assert claim.slug == AUTHORSHIP_SLUG
    assert claim.unblocked_by == "3QRPK52A"
    assert "not a person, not a bot" in claim.does_not_mean
    assert "not a statement about what they were doing" in claim.does_not_mean
    assert ClaimGate().check(claim) is None


def test_a_new_author_field_does_not_unblock_the_agentic_effectiveness_comparison_because_a_login_is_not_an_assignment_mechanism() -> (
    None
):
    """Decision 002, tested against the thing that would tempt somebody to retire it.

    Four new measures and a new author field have landed, and the project's central
    question is still refused. The refusal is for want of an *assignment mechanism*:
    the principal that happened to write a change was not assigned to write it,
    agents get the well-specified tickets and humans the rest, and that selection is
    the entire mechanism by which two groups differ. A login says who wrote a change
    and nothing about what they were given, so it is not that mechanism. The test
    builds the comparative claim somebody would build on the new field and asserts
    the gate still refuses it.
    """
    comparative = Claim(
        slug="agentic-change-author-comparison",
        statement="Changes opened by agent accounts were merged more often than changes opened by people.",
        does_not_mean="That the accounts are the agents or the people.",
        falsifier="A difference that survived conditioning on ticket specification.",
        denominator=Denominator("captured changes", 40),
        kind=ClaimKind.comparative,
        granularity=Granularity.team,
        unblocked_by="3QRPK52A",
    )
    with pytest.raises(ComparativeRefusalError):
        ClaimGate().require(comparative)

    refusal = default_registry().get("whether-agentic-development-is-more-effective")
    assert refusal.unblocked_by is None
    assert "assignment mechanism" in (refusal.missing_fact or "")


# -- outcome ----------------------------------------------------------------------------


def test_the_merged_share_is_over_captured_changes_and_says_so_plainly_because_the_census_is_still_missing() -> (
    None
):
    """The denominator says *captured*, twice, and the claim says why that matters.

    Kojutsu writes only on success, so a change it never wrote about is
    indistinguishable from one it examined and found unremarkable. A merge rate over
    the captured changes is a rate over a selected population, and a reader who
    quotes it as "the merge rate" has quoted a statement about development while
    believing they have quoted one about a store.
    """
    records = (
        _lifecycle(pr=1, outcome=PR_OUTCOME_MERGED),
        _lifecycle(pr=2, outcome=PR_OUTCOME_MERGED),
        _lifecycle(pr=3, outcome=PR_OUTCOME_CLOSED_UNMERGED),
    )
    figure = _OUTCOMES.compute(build_snapshot(records))
    assert isinstance(figure, FigureGroup)
    rate = figure.figures[1]
    assert isinstance(rate, RateFigure)
    assert rate.rate.numerator == 2
    assert "captured changes" in rate.rate.denominator.description
    assert "captured" in rate.title
    assert "changes that produced a capture, not the changes that were made" in (
        rate.claim.does_not_mean
    )


def test_an_unrecognised_outcome_is_neither_a_merge_nor_an_abandonment_because_a_broken_writer_is_not_a_rejected_change() -> (
    None
):
    """Counted, named with the value it held, and kept out of both sides of the rate.

    Folding it into ``closed_unmerged`` would report a writer that has changed as a
    change somebody rejected. Folding it into ``merged`` would be worse and much
    harder to notice. So it is excluded, and the key carries the value it actually
    held, because "something else happened" is not actionable and the value is.
    """
    records = (
        _lifecycle(pr=1, outcome=PR_OUTCOME_MERGED),
        _lifecycle(pr=2, outcome="squashed_by_a_broken_writer"),
    )
    figure = _OUTCOMES.compute(build_snapshot(records))
    assert isinstance(figure, FigureGroup)
    rate = figure.figures[1]
    assert isinstance(rate, RateFigure)
    assert rate.rate.numerator == 1
    assert rate.rate.denominator.size == 1
    distribution = figure.figures[0]
    assert isinstance(distribution, DistributionFigure)
    assert f"{UNRECOGNISED_OUTCOME_PREFIX}squashed_by_a_broken_writer" in distribution.excluded


def test_a_captured_change_with_no_outcome_is_excluded_because_it_is_still_open_and_the_forge_reported_nothing() -> (
    None
):
    """An open change has no outcome, and asserting either one would invent it.

    ``_close_outcome`` in Kojutsu returns nothing for a change that was never
    closed, and the key is then absent rather than written as a third value. Filing
    an open change under ``closed_unmerged`` would say a change was abandoned when it
    is merely still going, which is the mirror of the merge-folding mistake.
    """
    records = (
        _lifecycle(pr=1, outcome=PR_OUTCOME_MERGED),
        _lifecycle(pr=2, outcome=None),
    )
    figure = _OUTCOMES.compute(build_snapshot(records))
    assert isinstance(figure, FigureGroup)
    distribution = figure.figures[0]
    assert isinstance(distribution, DistributionFigure)
    assert dict(distribution.values) == {PR_OUTCOME_MERGED: 1}
    assert dict(distribution.excluded) == {NO_OUTCOME_KEY: 1}
    rate = figure.figures[1]
    assert isinstance(rate, RateFigure)
    assert rate.rate.denominator.size == 1


def test_a_record_claiming_no_capture_is_out_of_the_outcome_measure_because_a_person_typed_it_in() -> (
    None
):
    """``asserted`` records have nothing behind them, so they have no outcome.

    A person typed the record in, so there is no delivery and no forge payload and
    nothing that can have been merged or abandoned. And a record with no
    ``capture_source`` at all is not the same thing: silence about provenance is not
    a claim of assertion, and a distribution of outcomes over records that may have
    been typed in by hand describes something this program cannot name.
    """
    records = (
        _lifecycle(pr=1, outcome=PR_OUTCOME_MERGED),
        _lifecycle(pr=2, outcome=PR_OUTCOME_MERGED, source="asserted"),
        _lifecycle(pr=3, outcome=PR_OUTCOME_MERGED, source=None),
    )
    figure = _OUTCOMES.compute(build_snapshot(records))
    assert isinstance(figure, FigureGroup)
    rate = figure.figures[1]
    assert isinstance(rate, RateFigure)
    assert rate.rate.numerator == 1
    assert rate.rate.denominator.size == 1


def test_the_outcome_claim_names_its_tickets_and_says_a_merge_is_not_a_judgement_because_a_report_will_put_it_next_to_a_review_rate() -> (
    None
):
    """The two tickets, and the sentence that stops the two rates being composed.

    ``QEVSTMYW`` delivered the merge fact and ``EB6FE5ZP`` is the census that would
    let the population be every change rather than the captured ones. Both are on
    the claim because the second is the one that decides whether the first was
    enough. And a merge is what GitHub reported happening, not a judgement that the
    change was good -- which is the inference a reader makes when a merge rate sits
    beside a review-verdict rate in the same report.
    """
    claim = _OUTCOMES.claim(build_snapshot([_lifecycle(pr=1, outcome=PR_OUTCOME_MERGED)]))
    assert claim.slug == OUTCOME_SLUG
    assert claim.unblocked_by == "QEVSTMYW, EB6FE5ZP"
    assert "not a judgement that the change was good" in claim.does_not_mean
    assert ClaimGate().check(claim) is None


# -- helpers -----------------------------------------------------------------------------


def _lifecycle(*, pr: int, outcome: str | None, source: str | None = "webhook") -> Record:
    """A captured lifecycle record for a change, with the outcome GitHub reported."""
    return build_record(
        entry_id=f"pr-event-{pr}",
        pr=pr,
        record_kind=RecordKind.PR_LIFECYCLE,
        tags=["pr_state_change", "action_closed"],
        capture_source=source,
        pr_outcome=outcome,
        pr_merged_at="2026-01-05T00:00:00+00:00" if outcome == PR_OUTCOME_MERGED else None,
        pr_opened_at="2026-01-01T00:00:00+00:00",
    )


# -- slugs, which the report layer looks up ------------------------------------------


def test_every_measure_has_a_distinct_slug_and_none_of_them_is_refused_because_a_collision_would_refuse_the_wrong_measure() -> (
    None
):
    """The binding between a measure, its claim and the refusal catalogue.

    A report asks the registry whether a measure is blocked, using the measure's
    slug. Two measures sharing a slug would have one of them refused on the other's
    grounds; a measure whose slug collided with a slugified entry in
    ``docs/seam.md`` would be refused by a refusal filed about something else. So
    the slugs are dotted -- a namespace the registry does not use -- and the test
    holds all of them at once, because uniqueness is a property of the set and
    cannot be checked one measure at a time.
    """
    from tenbin.measures import (
        AuthorCoverageMeasure,
        CompletenessMeasure,
        DeclaredModelCoverageMeasure,
        MonthlyCoverageMeasure,
        RepositoryCoverageMeasure,
        RevisionIntervalMeasure,
        RevisionPressureMeasure,
        TrustProfileMeasure,
    )

    measures = [
        _VERDICTS,
        _WAITS,
        _AUTHORS,
        _OUTCOMES,
        CompletenessMeasure(),
        TrustProfileMeasure(),
        RepositoryCoverageMeasure(),
        MonthlyCoverageMeasure(),
        AuthorCoverageMeasure(),
        DeclaredModelCoverageMeasure(),
        RevisionPressureMeasure(),
        RevisionIntervalMeasure(),
    ]
    slugs = [measure.slug for measure in measures]
    assert len(slugs) == len(set(slugs)) == len(measures)
    assert all("." in slug for slug in slugs), "slugs are dotted, so they cannot collide"
    registry = default_registry()
    for slug in slugs:
        assert slug not in registry, f"{slug} collides with a refusal entry"
    for measure in measures:
        assert measure.claim(build_snapshot([build_record(pr=1)])).slug == measure.slug


# -- the branches a real corpus reaches ----------------------------------------------


def test_a_verdict_with_no_reviewing_account_is_excluded_because_it_has_nobody_to_attribute() -> (
    None
):
    """A verdict the forge reported no user for, counted rather than filed under a name.

    Putting it under a placeholder would make the absence look like a principal,
    and a reader who saw it in a reviewer's column would be reading a fact about a
    person that the record does not contain. It is the same rule as the model and
    the author axes, on a field where the temptation is sharpest because the
    neighbouring column is made of names.
    """
    records = (_verdict(pr=1, reviewer=None),)
    whole = _distribution(_VERDICTS.compute(build_snapshot(records)), 0)
    assert dict(whole.values) == {"approved": 1}
    assert dict(whole.excluded) == {NO_REVIEWER_KEY: 1}
    per_reviewer = _distribution(_VERDICTS.compute(build_snapshot(records)), 1)
    assert "(no reviewing account on the record) — approved" in per_reviewer.values


def test_a_verdict_with_no_verdict_state_is_counted_under_an_explicit_key_because_an_unknown_state_is_not_an_approval() -> (
    None
):
    """A ``review`` tag with no state this reader knows, counted as itself.

    Kojutsu's vocabulary has two states and deliberately no ``commented``,
    because leaving a note with no verdict is feedback rather than an approval. A
    record tagged ``review`` with a state nobody knows is either a new state or a
    hand-edited document, and neither is ``approved`` -- so it is counted under a
    key that says the record carries no state this reader can name.
    """
    record = build_record(
        entry_id="review-x",
        pr=1,
        record_kind=RecordKind.REVIEW_VERDICT,
        tags=["review"],
        review_id=1,
        comment_author="dana",
    )
    whole = _distribution(_VERDICTS.compute(build_snapshot([record])), 0)
    assert dict(whole.values) == {"(no verdict state on the record)": 1}


def test_a_verdict_with_no_timestamp_contributes_no_interval_because_an_interval_needs_both_ends() -> (
    None
):
    """The other end of the interval, and the same key as a missing open time.

    A verdict Kojutsu captured with no ``answered_at`` has no time to measure
    from. It is counted under the open-time key rather than a fourth one, because
    the two are the same gap from a reader's point of view -- this change has one
    end of its wait and not the other -- and a reader checking the timestamps is
    checking both.
    """
    records = (
        _verdict(
            pr=1,
            review_id=1,
            answered_at=None,
            pr_opened_at="2026-01-01T00:00:00+00:00",
        ),
    )
    figure = _WAITS.compute(build_snapshot(records))
    assert isinstance(figure, DistributionFigure)
    assert dict(figure.excluded) == {NO_OPEN_TIME_KEY: 1}
    assert sum(figure.values.values()) == 0


def test_a_verdict_captured_before_the_change_opened_is_excluded_because_the_wrong_end_is_the_one_to_check_first() -> (
    None
):
    """A negative wait, and which of the two timestamps the key points at.

    A verdict captured before the change opened is either clock skew or an open time
    read off the wrong event. It is excluded rather than bucketed, and it goes under
    the open-time key because that is the field a reader would go and check first
    and the wrong end is the more likely of the two.
    """
    records = (
        _verdict(
            pr=1,
            review_id=1,
            answered_at="2025-12-31T00:00:00+00:00",
            pr_opened_at="2026-01-01T00:00:00+00:00",
        ),
    )
    figure = _WAITS.compute(build_snapshot(records))
    assert isinstance(figure, DistributionFigure)
    assert dict(figure.excluded) == {NO_OPEN_TIME_KEY: 1}
    assert figure.values["under 1 hour"] == 0


def test_a_captured_record_with_no_pull_request_is_out_of_the_outcome_measure_because_half_a_key_is_not_a_key() -> (
    None
):
    """A record with a repository and no pull request number cannot be a change.

    :attr:`Record.pr_key` refuses a half key on purpose, and this is where that
    refusal pays: a record with no change to be an outcome of is skipped rather
    than joined to every repository that happens to hold a document with no number.
    """
    records = (
        _lifecycle(pr=1, outcome=PR_OUTCOME_MERGED),
        build_record(
            entry_id="pr-event-0",
            pr=None,
            record_kind=RecordKind.PR_LIFECYCLE,
            tags=["pr_state_change", "action_closed"],
            capture_source="webhook",
            pr_outcome=PR_OUTCOME_MERGED,
        ),
    )
    figure = _OUTCOMES.compute(build_snapshot(records))
    assert isinstance(figure, FigureGroup)
    rate = figure.figures[1]
    assert isinstance(rate, RateFigure)
    assert rate.rate.denominator.size == 1


def test_a_review_with_no_pull_request_is_out_of_the_wait_measure_because_half_a_key_is_not_a_key() -> (
    None
):
    """A verdict with no change attached to it has no wait to measure.

    :attr:`Record.pr_key` refuses a half key on purpose, and this is where that
    refusal pays: a verdict whose document has no ``pr-<n>`` segment is not joined
    to every repository that happens to hold a document with no number, and it is
    not measured against an open time that belongs to some other change.
    """
    records = (
        build_record(
            entry_id="review-orphan",
            pr=None,
            record_kind=RecordKind.REVIEW_VERDICT,
            tags=["review", "review_state_approved"],
            review_id=1,
            comment_author="dana",
            answered_at="2026-01-02T09:00:00+00:00",
            pr_opened_at="2026-01-01T00:00:00+00:00",
        ),
    )
    figure = _WAITS.compute(build_snapshot(records))
    assert isinstance(figure, DistributionFigure)
    assert figure.excluded == {}
    assert sum(figure.values.values()) == 0
    assert figure.claim.denominator.size == 1
