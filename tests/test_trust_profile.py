"""That the trust profile describes review structure and refuses to score it.

Kojutsu's independence axis is the strongest thing the corpus has, and it is
also the axis most likely to be turned into a ranking by a reader who has not read
``models.py:116``. So this module pins three things: that the levels are counted as
levels, that an absent level is counted as an absence rather than filed under a
bucket, and that nothing in the rendered output presents the axis as a scale of
quality. The last one is a test on the *text* rather than on the sort order,
because a title saying "strongest independent evidence" is a ranking with no
ordering in it at all.

The capture-source half is about the split. ``asserted`` is a permanent constant on
a rationale, so a single blended percentage moves whenever the mix of record kinds
moves, and a reader would see a trust change where nothing about trust changed.
Both halves are checked, and the test that would catch a regression is the one that
moves the *mix* rather than the *levels*: if the figures do not change when a
rationale is added, the split is not doing its job.
"""

from __future__ import annotations

from tenbin.claims.model import ClaimKind
from tenbin.corpus.record import Record, RecordKind
from tenbin.measures.base import DistributionFigure, Figure, FigureGroup
from tenbin.measures.trust import (
    ENTRY_CAPTURE_TITLE,
    INDEPENDENCE_RANK,
    INDEPENDENCE_TITLE,
    RATIONALE_CAPTURE_TITLE,
    TRUST_PROFILE_SLUG,
    UNSTATED_CAPTURE_SOURCE,
    UNSTATED_INDEPENDENCE,
    CaptureSource,
    Independence,
    TrustProfileMeasure,
)
from tests.fixtures import build_record, build_snapshot

_MEASURE = TrustProfileMeasure()

#: Words that would turn the axis into a scale. Checked against the rendered text
#: rather than the ordering because a ranking does not need an ordering: "the
#: strongest evidence in the corpus" ranks just as firmly as a sorted bar chart,
#: and a test on the sort would pass straight over it.
_RANKING_WORDS = (
    "strongest",
    "weakest",
    "best",
    "worst",
    "highest",
    "lowest",
    "top",
    "ranking",
    "score",
)


def _independence(figure: Figure) -> DistributionFigure:
    """The independence figure out of the group, found by title rather than position.

    By title because the group's order is a presentation decision and a test that
    indexed it would break the moment somebody decided the capture figures read
    better first -- without anything about the measure having changed.
    """
    assert isinstance(figure, FigureGroup)
    for member in figure.figures:
        if member.title == INDEPENDENCE_TITLE:
            assert isinstance(member, DistributionFigure)
            return member
    raise AssertionError("the independence figure is missing from the group")


def _capture(figure: Figure, title: str) -> DistributionFigure:
    """The capture figure with a given title, found the same way."""
    assert isinstance(figure, FigureGroup)
    for member in figure.figures:
        if member.title == title:
            assert isinstance(member, DistributionFigure)
            return member
    raise AssertionError(f"no capture figure titled {title!r}")


def _rationale(
    *,
    entry_id: str = "rationale-v1-0",
    pr: int = 1,
    independence: str | None = None,
) -> Record:
    """A stated rationale, which is permanently ``asserted``.

    A rationale record rather than an answer because the split is by kind: an
    answer carrying ``capture_source="asserted"`` is possible -- a person can type
    an answer in -- and folding it in with the rationales would make the split a
    guess about the writer's intent rather than a fact about the kind.
    """
    return build_record(
        entry_id=entry_id,
        pr=pr,
        rationale=True,
        record_kind=None,
        tags=["rationale", "rationale_declared"],
        capture_source=CaptureSource.ASSERTED.value,
        independence=independence,
        rationale_source="declared",
    )


def test_the_independence_levels_are_counted_as_levels_and_nothing_is_weighted() -> None:
    """A distribution of levels, with no level turned into a number to add up.

    Independence says who was positioned to disagree. Counting the levels is a
    description of review structure; the moment the levels are given a value and
    the values are added, the figure becomes a claim that some records are worth
    more than others, which is the claim this package refuses for want of anything
    to compare them with.
    """
    records = (
        build_record(pr=1, independence=Independence.INDEPENDENT.value),
        build_record(pr=2, independence=Independence.INDEPENDENT.value),
        build_record(pr=3, independence=Independence.MODEL_SEPARATED.value),
    )
    values = _independence(_MEASURE.compute(build_snapshot(records))).values
    assert dict(values) == {"independent": 2, "model_separated": 1}


def test_no_rendered_output_presents_the_independence_axis_as_a_ranking_because_independence_is_not_quality() -> (
    None
):
    """The check is on the text, because a ranking does not need a sort order.

    Kojutsu ranks its three levels and the rank is in the data, which is exactly
    what makes the temptation here stronger than elsewhere: a reader who concludes
    that ``independent`` records are the interesting ones is not being
    unreasonable, and is still wrong. A corpus full of independent records is a
    corpus in which people disagreed -- not a corpus of correct answers. So the
    a scale, and a test on the sort would pass straight over it. The words are
    searched for in the title and the bucket labels; the claim is checked separately
    for the denial, because the claim has to *say* there is no ranking and a word
    search over a sentence denying one finds the denial and calls it a violation.
    """
    records = tuple(
        build_record(pr=index + 1, independence=level.value)
        for index, level in enumerate(Independence)
    )
    figure = _independence(_MEASURE.compute(build_snapshot(records)))
    rendered = f"{figure.title}\n{figure.value_text()}".casefold()
    for word in _RANKING_WORDS:
        assert word not in rendered, f"{word!r} presents the axis as a scale"
    assert "no score to rank them with" in figure.claim.does_not_mean
    # And the order is by count, not by the vocabulary's own order. One of each
    # level ties on count, so the tie-break by key decides and that is visible here;
    # the uneven corpus below is the one that would catch a rank order.
    assert list(figure.values) == [
        Independence.INDEPENDENT.value,
        Independence.MODEL_SEPARATED.value,
        Independence.SELF_CERTIFIED.value,
    ]
    uneven = _independence(
        _MEASURE.compute(
            build_snapshot(
                tuple(
                    build_record(pr=index + 1, independence=Independence.SELF_CERTIFIED.value)
                    for index in range(5)
                )
                + (build_record(pr=9, independence=Independence.INDEPENDENT.value),)
            )
        )
    )
    # Count order puts the five first; a rank order would put the single
    # ``independent`` first, which is the reading the whole test is about.
    assert list(uneven.values) == [
        Independence.SELF_CERTIFIED.value,
        Independence.INDEPENDENT.value,
    ]


def test_the_rank_is_written_down_because_a_rule_that_says_weakest_to_strongest_needs_an_order_somewhere() -> (
    None
):
    """The order is a vocabulary fact, and it is available without being applied.

    An enum is unordered by construction, so a reader is entitled to know the
    sequence Kojutsu documents. It is held as a mapping rather than an enum
    property so that iterating the members cannot accidentally be the way the order
    gets applied -- which is the failure the module docstring is about.
    """
    assert dict(INDEPENDENCE_RANK) == {
        Independence.SELF_CERTIFIED.value: 0,
        Independence.MODEL_SEPARATED.value: 1,
        Independence.INDEPENDENT.value: 2,
    }
    assert list(INDEPENDENCE_RANK) == [
        Independence.SELF_CERTIFIED.value,
        Independence.MODEL_SEPARATED.value,
        Independence.INDEPENDENT.value,
    ]


def test_an_absent_independence_is_counted_as_an_absence_because_nobody_made_a_claim_about_who_could_disagree() -> (
    None
):
    """A record with no level is not in the ``self_certified`` bucket.

    Kojutsu omits the key when no level was derived, so a record in that state
    was never assigned one, and giving it a bucket would put an absence in the same
    column as three levels with the same formatting. That is the ``unknown``-model
    defect in a second costume, and it is the single most expensive shape a
    distribution can have here. So the absence is counted under a key that names it.
    """
    records = (
        build_record(pr=1, independence=Independence.INDEPENDENT.value),
        build_record(pr=2),
        build_record(pr=3, independence="   "),
    )
    figure = _independence(_MEASURE.compute(build_snapshot(records)))
    assert dict(figure.values) == {"independent": 1}
    assert dict(figure.excluded) == {UNSTATED_INDEPENDENCE: 2}
    assert UNSTATED_INDEPENDENCE not in figure.values


def test_an_unrecognised_level_is_counted_as_itself_because_the_writer_stated_something() -> None:
    """A level this reader does not know is a fact about the writer, not an absence.

    The opposite case to the one above, and the reason the two cannot share a
    treatment: a record with a blank level is missing a statement, while a record
    with a level outside the vocabulary has made one. Folding the second into a
    neighbour would report a writer that has changed as one that said
    ``self_certified``, and dropping it would be invisible.
    """
    records = (
        build_record(pr=1, independence="four_party_review"),
        build_record(pr=2, independence=Independence.INDEPENDENT.value),
    )
    values = _independence(_MEASURE.compute(build_snapshot(records))).values
    assert dict(values) == {"four_party_review": 1, "independent": 1}


def test_the_capture_figure_splits_rationales_out_before_aggregating_because_asserted_is_a_constant_there() -> (
    None
):
    """The composition change that must not read as a trust change, and the test that catches it.

    A rationale is permanently ``asserted``, so adding one to the corpus moves a
    blended capture percentage while every captured record keeps the source it had.
    A reader comparing the two runs would see trust change. On a split figure the
    entry distribution does not move at all, which is the only correct answer and is
    what this asserts.
    """
    entries = (
        build_record(pr=1, capture_source=CaptureSource.WEBHOOK.value),
        build_record(pr=2, capture_source=CaptureSource.WEBHOOK.value),
        build_record(pr=3, capture_source=CaptureSource.COLLECT.value),
    )
    before = _capture(_MEASURE.compute(build_snapshot(entries)), ENTRY_CAPTURE_TITLE)
    after = _capture(
        _MEASURE.compute(build_snapshot(entries + (_rationale(pr=4),))), ENTRY_CAPTURE_TITLE
    )
    assert before.values == after.values
    assert dict(after.values) == {"webhook": 2, "collect": 1}

    rationales = _capture(
        _MEASURE.compute(build_snapshot(entries + (_rationale(pr=4),))), RATIONALE_CAPTURE_TITLE
    )
    assert dict(rationales.values) == {"asserted": 1}


def test_the_two_capture_titles_each_name_their_population_because_a_reader_must_not_have_to_guess_which_is_which() -> (
    None
):
    """The population is in the title, and the titles say the two are not comparable.

    The figure takes its population as an argument, which the package's own rule
    says is a call-site decision rather than a property of the figure. It is safe
    here for one reason and one reason only: the title names the population, and a
    report cannot leave a title out. Each title also carries the reason the two are
    separate -- one says ``asserted`` is a constant, the other says the mix is
    something that happened -- so the reader is told why not to compare them rather
    than only being given two numbers.
    """
    figure = _MEASURE.compute(build_snapshot([_rationale()]))
    assert "constant" in RATIONALE_CAPTURE_TITLE
    assert "happened" in ENTRY_CAPTURE_TITLE
    titles = {member.title for member in figure.figures}  # type: ignore[union-attr]
    assert titles == {INDEPENDENCE_TITLE, ENTRY_CAPTURE_TITLE, RATIONALE_CAPTURE_TITLE}


def test_a_rationale_with_no_independence_level_is_excluded_rather_than_rated_because_the_writer_stated_none() -> (
    None
):
    """Kojutsu does not derive independence for a rationale, and this shows it.

    ``asserted`` records have no level at all -- a declared rationale never raises
    independence, because there is no second party to raise it against. So a corpus
    with rationales in it has records the independence figure cannot count, and the
    honest handling is the exclusion rather than an assumption that a rationale is
    self-certified.
    """
    records = (
        build_record(pr=1, independence=Independence.INDEPENDENT.value),
        _rationale(pr=2),
    )
    figure = _independence(_MEASURE.compute(build_snapshot(records)))
    assert dict(figure.values) == {"independent": 1}
    assert dict(figure.excluded) == {UNSTATED_INDEPENDENCE: 1}


def test_a_record_with_no_capture_source_is_excluded_because_silence_about_provenance_is_not_assertion() -> (
    None
):
    """The missing key is not a claim that somebody typed the record in.

    Kojutsu defaults nothing, so a document with no ``capture_source`` is a
    document that says nothing about its own provenance. Calling it ``asserted``
    would make a hand-written document pass as a documented claim about intent,
    and would make the anomaly invisible in the corpus where it is the interesting
    thing.
    """
    figure = _capture(_MEASURE.compute(build_snapshot([build_record(pr=1)])), ENTRY_CAPTURE_TITLE)
    assert dict(figure.values) == {}
    assert dict(figure.excluded) == {UNSTATED_CAPTURE_SOURCE: 1}


def test_a_capture_source_outside_the_vocabulary_is_excluded_rather_than_folded_into_asserted() -> (
    None
):
    """A writer that has invented a fourth source is not a person typing something in.

    Folding it into ``asserted`` would report a broken writer as a documented claim
    about intent, which is the one reading the axis exists to distinguish and would
    be indistinguishable from a real one in the rendered figure.
    """
    figure = _capture(
        _MEASURE.compute(build_snapshot([build_record(pr=1, capture_source="telepathy")])),
        ENTRY_CAPTURE_TITLE,
    )
    assert dict(figure.values) == {}
    assert sum(figure.excluded.values()) == 1


def test_the_claim_names_who_was_positioned_to_disagree_and_not_whether_anything_is_correct() -> (
    None
):
    """The negative space, in the terms Kojutsu uses for the axis.

    Two clauses, and the second is the one a report is most likely to leave out.
    The axis does not say anything is true -- and it does not say a record with a
    higher level is a better record either, which is the reading an ordinal
    vocabulary invites and the one this figure must not enable.
    """
    claim = _MEASURE.claim(build_snapshot([build_record(pr=1)]))
    assert claim.kind is ClaimKind.descriptive
    assert "positioned to disagree" in claim.does_not_mean
    assert "not whether anything is correct" in claim.does_not_mean
    assert "ranks records or principals" in claim.does_not_mean
    assert claim.slug == TRUST_PROFILE_SLUG


def test_the_falsifier_is_a_correlation_with_distinct_accounts_because_that_is_what_would_disprove_the_axis() -> (
    None
):
    """A falsifier that could actually be checked, against data that exists.

    The level is a claim about how many parties were in a position to object. If a
    repository with more independent records has no more distinct accounts
    commenting on it, the level is not describing that, and the axis means nothing.
    Both halves of the falsifier are observable in the corpus, which is what makes
    it a falsifier rather than a sentiment.
    """
    claim = _MEASURE.claim(build_snapshot([build_record(pr=1)]))
    assert "distinct commenting accounts" in claim.falsifier
    assert "uncorrelated" in claim.falsifier


def test_every_trust_figure_carries_the_same_claim_and_therefore_the_same_population() -> None:
    """A group shares one claim, which is a constraint rather than a convenience.

    The claim's denominator names the records in the read, and all three figures
    are over exactly those. A figure carrying a claim whose denominator does not
    describe it is the defect this package is built to stop, so the assertion is
    that the three really are the same population -- and a measure whose parts
    genuinely differed in population would have to be two measures instead.
    """
    records = (build_record(pr=1, independence=Independence.INDEPENDENT.value), _rationale(pr=2))
    snapshot = build_snapshot(records)
    figure = _MEASURE.compute(snapshot)
    assert isinstance(figure, FigureGroup)
    assert len(figure.figures) == 3
    for member in figure.figures:
        assert member.claim == figure.claim
        assert member.claim.denominator.size == len(records)
        assert member.payload != FigureGroup.payload


def test_a_rationale_record_is_addressed_in_the_rationale_namespace_because_the_split_is_by_kind() -> (
    None
):
    """The fixture is a real rationale document, not an answer carrying a tag.

    The split between the two capture figures is by ``is_rationale``, which is the
    union of the id path and the tag. A fixture that built a rationale's *frontmatter*
    under an answer's id would still classify as a rationale on the tag, which is
    the honest union -- but a test that then asserted on the id would be asserting on
    a document Kojutsu does not write. So the id is built the way it builds it.
    """
    record = _rationale()
    assert record.is_rationale
    assert record.rationale_path
    assert record.classification.kind is RecordKind.RATIONALE
