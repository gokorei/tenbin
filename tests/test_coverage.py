"""Where the corpus reaches, and the four ways a distribution of coverage can lie.

Coverage is the question a self-selected corpus answers worst and most
confidently, so this module spends most of its effort on the ways a coverage
figure misleads rather than on the counting. Three of the four measures can
produce a number that looks like knowledge debt and is not: a repository nobody
answers on is byte-identical to a repository with nothing to know, a model bucket
named for the absence of a model is a group that never existed, and an ``author``
fallback puts an agent id and a person's login in the same bucket.

The fourth is the one this package is named after. ``author`` is the answer's
author, which Kojutsu fills with ``"unknown"`` for an unattributed answer and
with an agent id for an agent-authored one -- two different things sharing one
string. A fallback to it would produce a distribution of *principals* that is
really a distribution of provenance, and the test below builds both halves of the
collision on purpose so the failure has something to be a failure *of*.
"""

from __future__ import annotations

from tenbin.claims.gate import ClaimGate
from tenbin.claims.mechanism import Granularity
from tenbin.claims.model import ClaimKind
from tenbin.corpus.record import RecordKind
from tenbin.measures.base import DistributionFigure
from tenbin.measures.coverage import (
    NO_COMMENT_AUTHOR_KEY,
    NO_MONTH_KEY,
    NO_REPOSITORY_KEY,
    NO_STATED_MODEL_KEY,
    AuthorCoverageMeasure,
    DeclaredModelCoverageMeasure,
    MonthlyCoverageMeasure,
    RepositoryCoverageMeasure,
)
from tests.fixtures import build_record, build_snapshot

_REPOSITORIES = RepositoryCoverageMeasure()
_MONTHS = MonthlyCoverageMeasure()
_AUTHORS = AuthorCoverageMeasure()
_MODELS = DeclaredModelCoverageMeasure()


def _figure(measure, records) -> DistributionFigure:
    """Compute a measure and assert the one type it is supposed to return."""
    figure = measure.compute(build_snapshot(records))
    assert isinstance(figure, DistributionFigure)
    return figure


# -- repositories ---------------------------------------------------------------


def test_records_are_grouped_by_repository_because_a_repository_is_the_corpus_s_only_real_unit() -> (
    None
):
    """The ordinary case, and the counting itself is not the interesting part.

    A histogram of repositories is a description of the shape of the corpus. What
    makes it safe is the claim attached to it, and that is what the other tests in
    this module are about.
    """
    records = (
        build_record(pr=1, repo="acme/widget"),
        build_record(pr=2, repo="acme/widget"),
        build_record(pr=3, repo="acme/gadget"),
    )
    assert dict(_figure(_REPOSITORIES, records).values) == {"acme/widget": 2, "acme/gadget": 1}


def test_a_record_with_no_repository_is_counted_separately_because_absent_is_not_a_repository() -> (
    None
):
    """A record that names no repository is an absence, and the key says so.

    Kojutsu writes the literal ``"unknown"`` where no repository was stated, and
    a bucket labelled with that would be a claim about a repository that nobody
    named. The excluded key is a sentence rather than a bucket label, because this
    is a *reason for exclusion* in a distribution of all records and not a key in a
    histogram of anomalous ones -- reusing one string for both would give a reader
    who has seen the provenance profile a bucket they think they understand.
    """
    records = (build_record(pr=1, repo="acme/widget"), build_record(pr=2, repo=None))
    figure = _figure(_REPOSITORIES, records)
    assert dict(figure.values) == {"acme/widget": 1}
    assert dict(figure.excluded) == {NO_REPOSITORY_KEY: 1}
    assert NO_REPOSITORY_KEY not in figure.values


def test_the_repository_claim_says_an_absent_repository_is_not_an_absent_debt_because_they_are_byte_identical() -> (
    None
):
    """The negative space, written into every one of these claims.

    This is the sentence that keeps a coverage figure from being read as a map of
    the organisation. Four hundred documents over nine repositories reads like one,
    and it is a map of the paths somebody happened to walk: a repository nobody
    answers on is byte-identical to a repository with no knowledge debt, and no
    backfill path exists for either.
    """
    claim = _REPOSITORIES.claim(build_snapshot([build_record(pr=1)]))
    assert "shape of development. It is not" in claim.does_not_mean
    assert "absent from it" in claim.does_not_mean
    assert "byte-identical" in claim.does_not_mean
    assert "no backfill path" in claim.does_not_mean
    assert "self-selected" in claim.does_not_mean


def test_every_coverage_falsifier_is_the_census_because_the_census_is_what_would_change_the_population() -> (
    None
):
    """One falsifier for four measures, because the fact that settles them is one fact.

    ``EB6FE5ZP`` is a census of changes that produced no capture. A census naming
    materially more repositories than the distribution shows would mean the figure is
    measuring capture rather than knowledge debt -- which is a different claim, and a
    much less flattering one. The census has not landed, so every one of these
    claims names it and the population stays what it is.
    """
    for measure in (_REPOSITORIES, _MONTHS, _AUTHORS, _MODELS):
        claim = measure.claim(build_snapshot([build_record(pr=1)]))
        assert "EB6FE5ZP" in claim.falsifier
        assert "materially" in claim.falsifier


# -- months ----------------------------------------------------------------------


def test_records_are_grouped_by_the_month_of_the_event_they_are_about() -> None:
    """The event first, the write second, and the mixture is named rather than hidden.

    ``Record.month`` already encodes the preference order -- ``answered_at``, then
    ``captured_at``, then ``declared_at``, then the document's own update -- so a
    measure that restated it would be a second definition to keep in step. What the
    claim has to add is that a corpus with gaps in any of those clocks is being
    grouped over a mixture, and that the gaps are not absences of knowledge.
    """
    records = (
        build_record(pr=1, answered_at="2026-01-15T09:00:00+00:00"),
        build_record(pr=2, answered_at="2026-01-20T09:00:00+00:00"),
        build_record(pr=3, answered_at="2026-02-02T09:00:00+00:00"),
    )
    assert dict(_figure(_MONTHS, records).values) == {"2026-01": 2, "2026-02": 1}
    claim = _MONTHS.claim(build_snapshot(records))
    assert "mixture" in claim.does_not_mean
    assert "not absences of knowledge" in claim.does_not_mean


def test_a_record_with_no_timestamp_at_all_is_counted_separately_because_undated_is_not_a_month() -> (
    None
):
    """``None`` is a real answer from ``Record.month`` and has to be honoured.

    A record with no timestamp of any kind is not in January. Filing it under the
    oldest month, or under the read's own month, would put a record in a period it
    has no relationship to, and a month histogram is exactly the place where a
    fabricated period is least visible.
    """
    dated = build_record(pr=1, answered_at="2026-01-15T09:00:00+00:00")
    undated = build_record(pr=2, updated_at=None, answered_at=None, captured_at=None)
    assert dated.month == "2026-01"
    assert undated.month is None
    figure = _figure(_MONTHS, [dated, undated])
    assert dict(figure.values) == {"2026-01": 1}
    assert dict(figure.excluded) == {NO_MONTH_KEY: 1}


# -- authors ------------------------------------------------------------------------


def test_the_author_grouping_uses_comment_author_and_never_falls_back_to_author() -> None:
    """The sentinel problem in a second costume, built on both sides so it can bite.

    Three records: one posted by a person, one posted by nobody the forge reported,
    one written by an agent. Kojutsu fills ``author`` with the account for the
    first, the literal ``"unknown"`` for the second, and the agent id for the third
    -- so a fallback to ``author`` puts a person, a placeholder and a bot in one
    bucket and renders the result as a distribution of people. The fixture states
    all three explicitly, because the failure needs the collision to be real.
    """
    records = (
        build_record(pr=1, comment_author="dana", author="dana"),
        build_record(pr=2, comment_author=None, author="unknown"),
        build_record(pr=3, comment_author=None, author="agent-42", answered_by_agent="agent-42"),
    )
    figure = _figure(_AUTHORS, records)
    assert dict(figure.values) == {"dana": 1}
    assert dict(figure.excluded) == {NO_COMMENT_AUTHOR_KEY: 2}
    # The two excluded records are the ones a fallback would have filed, and neither
    # ``unknown`` nor ``agent-42`` appears anywhere in the figure.
    assert "unknown" not in figure.values
    assert "agent-42" not in figure.values


def test_the_author_claim_says_a_count_here_is_not_a_property_of_a_person_because_the_floor_only_refuses_the_claim_about_one() -> (
    None
):
    """Why a login in a bucket key is not a per-principal claim.

    The granularity floor refuses a claim *about* a principal, not a figure that
    happens to have a login in one of its keys. A histogram of records per account
    compares no account with another and attaches no count to anybody as a property
    of them; who was assigned the work is not in the corpus, so an account absent
    from the figure has not been shown to have had nothing to say. Both halves are
    asserted because either alone is half the argument.
    """
    claim = _AUTHORS.claim(build_snapshot([build_record(pr=1)]))
    assert claim.granularity is Granularity.team
    assert "property of a person" in claim.does_not_mean
    assert "not been shown to have had nothing to say" in claim.does_not_mean
    assert ClaimGate().check(claim) is None


# -- stated models --------------------------------------------------------------------


def test_the_model_distribution_is_built_only_from_stated_models_because_an_absence_is_not_a_group() -> (
    None
):
    """The measure, asserted directly: unstated records are skipped and counted.

    ``answered_by_model`` is parsed into ``Stated | Unstated``, the second is
    skipped, and the skips land in ``excluded`` under a key naming the condition. A
    record that states no model is not evidence that nobody used one, and a bucket
    named for the absence would report that absence as a characteristic of a
    principal -- the specific failure this package exists to have stopped.
    """
    records = (
        build_record(pr=1, answered_by_model="claude-opus-5"),
        build_record(pr=2, answered_by_model="claude-opus-5"),
        build_record(pr=3, answered_by_model="gpt-5"),
        build_record(pr=4),
    )
    figure = _figure(_MODELS, records)
    assert dict(figure.values) == {"claude-opus-5": 2, "gpt-5": 1}
    assert dict(figure.excluded) == {NO_STATED_MODEL_KEY: 1}
    assert "unknown" not in figure.values


def test_a_legacy_record_writing_the_literal_unknown_is_excluded_not_bucketed_because_the_two_are_not_separable() -> (
    None
):
    """Kojutsu stopped writing the literal, and for the records that still carry it the two collide.

    ``parse_stated`` maps ``"unknown"`` to ``Unstated``, because a value a principal
    stated and Kojutsu's placeholder are the same string in a stored document and
    nothing in the document separates them. The exclusion is therefore stated as a
    *condition* -- a record that names no model -- rather than as a clean split, and
    the reason is that the ambiguity is confined to records written before Kojutsu
    stopped writing the literal, so the excluded count is a legacy count and is
    shrinking. A reader who wants to know whether a particular bucket is a group or
    an absence is told, in the key, that the answer is not available.
    """
    records = (
        build_record(pr=1, answered_by_model="unknown"),
        build_record(pr=2, answered_by_model="claude-opus-5"),
    )
    figure = _figure(_MODELS, records)
    assert dict(figure.values) == {"claude-opus-5": 1}
    assert dict(figure.excluded) == {NO_STATED_MODEL_KEY: 1}
    assert "no model at all" in NO_STATED_MODEL_KEY


def test_the_model_claim_says_a_stated_model_is_a_self_assertion_because_nothing_can_verify_it() -> (
    None
):
    """The forge proves who posted a comment and nothing more.

    Kojutsu's own seam document is explicit that the platform cannot prove which
    model drafted the text, so a stated model is trustworthy exactly as far as the
    account making it. A distribution of those statements is a distribution of
    assertions, and a reader who does not know that will read it as a distribution
    of models that were used.
    """
    claim = _MODELS.claim(build_snapshot([build_record(pr=1)]))
    assert "self-assertion" in claim.does_not_mean
    assert "can verify" in claim.does_not_mean
    assert "the models that were used" in claim.does_not_mean


def test_every_coverage_figure_is_over_the_same_corpus_so_two_of_them_can_be_compared() -> None:
    """Four measures, one read, one population.

    Each claim's denominator is every record in the read rather than only the
    records the particular measure counted, so a reader comparing the by-repository
    figure with the model figure is comparing the same corpus and the difference
    between them is the exclusions each one names. Making the denominator the
    counted subset would break exactly that comparison, and the model measure is the
    one where it would be most tempting.
    """
    records = (
        build_record(pr=1, repo="acme/widget", answered_by_model="claude-opus-5"),
        build_record(pr=2, repo="acme/widget"),
    )
    snapshot = build_snapshot(records)
    for measure in (_REPOSITORIES, _MONTHS, _AUTHORS, _MODELS):
        claim = measure.claim(snapshot)
        assert claim.denominator.size == 2
        assert claim.kind is ClaimKind.descriptive
        assert ClaimGate().check(claim) is None


def test_a_coverage_measure_refuses_nothing_because_its_records_are_whatever_the_corpus_holds() -> (
    None
):
    """No certainty requirement, because a coverage figure is not a reading.

    Requiring ``DETERMINED`` here would discard every legacy record over something
    these measures never use: which repository a record is about, what month it is
    in, who posted it and what model it states are all fields on the document, and
    none of them is a consequence of how the record's kind was recovered. The
    classification filter exists for measures that are about *kinds*, and a measure
    that filters on nothing says so by defaulting.
    """
    from tenbin.measures.filtering import requires_certainty

    assert _REPOSITORIES.requires_certainty is requires_certainty
    assert requires_certainty is None
    legacy = build_record(pr=1, repo="acme/widget", record_kind=None)
    assert legacy.classification.certainty.value == "inferred"
    assert legacy.classification.kind is RecordKind.ANSWER
    assert dict(_figure(_REPOSITORIES, [legacy]).values) == {"acme/widget": 1}


def test_a_repository_is_named_only_when_it_is_one_because_the_placeholder_is_a_string_and_not_an_absence() -> (
    None
):
    """The rule, with all three cases, because a truthiness test gets one of them wrong.

    A record with no repository is filed at ``unknown/pr-<n>/<entry>``, so
    ``Record.repo`` is the string ``"unknown"`` rather than ``None`` -- which means
    ``if record.repo:`` counts the placeholder as a repository and every
    by-repository figure grows a bucket named after an absence. The function is
    public and used by two measures, so the three cases are asserted directly
    rather than only through whichever figure happened to reach them.
    """
    from tenbin.measures.filtering import named_repository

    assert named_repository("acme/widget") is True
    assert named_repository("  acme/widget  ") is True
    assert named_repository("unknown") is False
    assert named_repository("UNKNOWN") is False
    assert named_repository("   ") is False
    assert named_repository(None) is False
    # And a repository genuinely named something else is still a repository.
    assert named_repository("unknown-service/widget") is True
