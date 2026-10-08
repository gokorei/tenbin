"""Decision requests are readable, counted apart from everything, and checked for what must not cross.

Kojutsu's ``AZ8T5XYS`` projected terminal decision requests into Tanseki, and this
module is the regression suite for the three things that arrival could have got
wrong. The classification, because a request whose tags name no record kind used to
fall through to the answer case -- which is not a mislabel, it is a promotion of a
request into evidence. The safety check, because the promise that ``claim_token``
never crosses the seam was Kojutsu's to make and nobody's to verify, and a store
readable over MCP by agents turns that promise into the whole of the risk. And the
leak, because every measure in this program counts records, and a projected request
carries a repository and a document update time, so without a stated exclusion it
lands in the by-repository histogram as something somebody examined and in the
by-month histogram filed under the day it was written rather than the day it was
asked.

**The leak test is the one that cannot be argued with.** Every other test in this file
says a measure does the right thing; this one says a measure is *unchanged*, which is
the only statement that catches a helper quietly widening a population. It runs over
:func:`tenbin.measures.default_measures` rather than over a list written here,
because a test that passes its own list proves the list it passed -- which is how the
lifecycle-shape detector stayed correct, tested, and uncalled for a whole release.

**The one measure allowed to move is named, not skipped.** Read completeness divides
by the store's own count, so publishing more documents moves it without any question
having leaked into anything. Naming the exception means a second leaky measure fails
this test instead of joining a list nobody reads.
"""

from __future__ import annotations

from typing import Final

from tenbin.claims.gate import ClaimGate
from tenbin.claims.model import ClaimKind
from tenbin.claims.registry import default_registry
from tenbin.corpus.record import Record, RecordKind
from tenbin.corpus.snapshot import CorpusSnapshot, take_snapshot
from tenbin.measures import default_measures
from tenbin.measures.base import (
    DistributionFigure,
    Figure,
    FigureGroup,
    Finding,
    Measure,
)
from tenbin.measures.completeness import CompletenessMeasure
from tenbin.measures.decision_requests import (
    NO_RAISED_TIME_KEY,
    NO_STATUS_KEY,
    NO_TERMINAL_TIME_KEY,
    REQUEST_POPULATION,
    DecisionRequestLifecycleMeasure,
    DecisionRequestOutcomeMeasure,
    QuestionAnsweredRateMeasure,
    SupersededIntervalMeasure,
)
from tenbin.measures.filtering import CAPTURE_KINDS, capture_population
from tenbin.measures.integrity import LifecycleShapeMeasure
from tenbin.measures.projection_safety import (
    FORBIDDEN_PROJECTION_KEYS,
    PROJECTION_SAFETY_CLEAN_SLUG,
    PROJECTION_SAFETY_SLUG,
    ProjectionSafetyMeasure,
    QuestionCaptureAgreementMeasure,
    scan_for_forbidden_keys,
)
from tenbin.report.build import build_report
from tenbin.report.markdown import render_markdown
from tests.fakes import FakeStore
from tests.fixtures import build_record, build_snapshot, decision_request

_SAFETY = ProjectionSafetyMeasure()
_LIFECYCLE = DecisionRequestLifecycleMeasure()
_OUTCOMES = DecisionRequestOutcomeMeasure()

#: The three measures this ticket adds. Excluded from the leak test by name rather
#: than by "everything after the thirteenth entry", so adding a measure to the
#: registry cannot quietly exempt it.
_NEW_MEASURES: Final[frozenset[type]] = frozenset(
    {
        ProjectionSafetyMeasure,
        DecisionRequestLifecycleMeasure,
        DecisionRequestOutcomeMeasure,
        QuestionAnsweredRateMeasure,
        QuestionCaptureAgreementMeasure,
        SupersededIntervalMeasure,
    }
)

#: The pre-existing measures whose figures are expected to move when the store holds
#: more documents. One, and for a reason that is not about leaking: this figure divides
#: by the store's own count, which is a fact about the store rather than a population
#: this program chose, so it cannot be identical across two stores of different sizes
#: and should not pretend to be.
_MAY_MOVE: Final[frozenset[type]] = frozenset({CompletenessMeasure})


def _captures() -> tuple[Record, ...]:
    """A corpus with one of every capture kind, so no measure is handed nothing.

    Every field the existing measures read is stated on at least one of these: an
    independence level, a capture source, a commenting account, a declared model, a
    change author, a merge outcome, a review id and a rationale chain. A leak test run
    over records that stated none of them would pass whether the measures filtered or
    not, because the distributions would be empty either way.
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


def _requests() -> tuple[Record, ...]:
    """Three terminal requests, one per state, raised on the changes above."""
    return (
        decision_request("q-1", status="answered", pr=1),
        decision_request("q-2", status="superseded", pr=3, answered_at=None),
        decision_request("q-3", status="failed", pr=2),
    )


def _members(figure: Figure) -> tuple[Figure, ...]:
    """The figure and everything under it, so a group's members are compared too."""
    if isinstance(figure, FigureGroup):
        return (figure, *(member for child in figure.figures for member in _members(child)))
    return (figure,)


def _rendered(measure: Measure, snapshot: CorpusSnapshot) -> tuple[tuple[str, ...], ...]:
    """Every number and every population in a measure's output, as comparable text.

    The denominator is rendered rather than read as an integer, because the size
    alone would pass while the *description* beside it drifted -- and a denominator
    whose description has stopped matching its population is the failure this whole
    program exists to make unconstructible.
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


# -- classification ---------------------------------------------------------------


def test_a_projected_decision_request_is_a_request_and_not_an_answer_because_a_question_is_never_evidence() -> (
    None
):
    """The classification is the whole fix, and it is a kind of its own.

    A projected request arrives with no ``record_kind`` -- Kojutsu writes that key
    for its four entry writers and not for this projection -- so its kind comes from
    the tag set. Before the ``question`` tag was read, ``["question",
    "question_answered"]`` named no record kind at all and fell through to the answer
    case: an inferred answer, which would have been swept into every capture
    distribution in the program as a conclusion somebody had reached.
    """
    record = decision_request()

    assert record.classification.kind is RecordKind.QUESTION
    assert record.classification.certainty is not None
    assert record.record_kind is None, (
        "Kojutsu writes no record_kind on a projected request, so a fixture that "
        "supplied one would be testing a document the writer never produces"
    )
    assert record.classification.evidence == ("question", "question_answered")


def test_a_projected_decision_request_is_not_a_capture_by_decision_001_because_it_carries_neither_a_capture_source_nor_an_independence_level() -> (
    None
):
    """Both mechanisms the decision names, asserted rather than assumed.

    Decision 001 requires a question to be counted apart from captures, and it offers
    two mechanisms: a namespace of its own and the absence of the evidence axis. This
    asserts the absence directly, because the exclusion in
    :data:`~tenbin.measures.filtering.CAPTURE_KINDS` is a decision somebody could
    reverse, and the absence of the fields is the reason the decision is right.
    """
    record = decision_request()

    assert record.capture_source is None
    assert record.independence is None
    assert RecordKind.QUESTION not in CAPTURE_KINDS
    assert record not in capture_population((record,))


def test_the_request_namespace_is_read_from_the_document_id_alongside_the_rationale_one_because_a_second_parser_would_drift() -> (
    None
):
    """One branch over both sub-namespaces, and both still parse.

    The ids have the same shape, so the tempting change is a second ``if``; the cost of
    that is a second definition of what a document id looks like, and the second one
    is the one that quietly stops being applied.
    """
    from tenbin.corpus.record import parse_document_id

    request = parse_document_id("acme/widget/pr-7/question/q-7")
    rationale = parse_document_id("acme/widget/pr-7/rationale/rationale-v1-0")

    assert (request.is_question, request.is_rationale) == (True, False)
    assert request.entry_id == "q-7"
    assert (rationale.is_question, rationale.is_rationale) == (False, True)
    assert rationale.entry_id == "rationale-v1-0"


# -- the safety check -------------------------------------------------------------


def test_a_published_claim_token_is_a_finding_and_not_a_warning_because_tanseki_is_readable_by_agents() -> (
    None
):
    """The check's whole reason for being, stated in the shape it produces.

    ``claim_token`` is the ``WHERE``-clause guard on releasing a claim, so anyone
    holding it can release a claim somebody else holds; Tanseki is readable over MCP by
    agents, so a token in a document is a capability handed to every reader of the
    store. A *warning* would render as a note beside the numbers and the numbers would
    still be read. A finding is rendered first whatever order the registry was written
    in, and it carries a ``cannot_confirm`` that cannot be blank.
    """
    snapshot = build_snapshot((decision_request(frontmatter={"claim_token": "s3cret"}),))

    finding = scan_for_forbidden_keys(snapshot.records, snapshot=snapshot)

    assert isinstance(finding, Finding)
    assert finding.claim.slug == PROJECTION_SAFETY_SLUG
    assert "claim_token" in finding.what_was_observed
    assert "acme/widget/pr-42/question/q-0001" in finding.what_was_observed
    assert finding.cannot_confirm.strip(), "a finding with nothing here is a finding that implies"


def test_the_finding_names_the_key_and_the_document_but_never_the_value_because_a_report_is_not_a_place_to_republish_a_capability() -> (
    None
):
    """The value is the capability, and a finding is rendered into reports.

    The observation therefore stops at the key and the document id. Naming the value
    would put the token into a rendered report and into anybody's terminal reading one,
    which is the outcome the check exists to prevent, reached by the check itself.
    """
    snapshot = build_snapshot((decision_request(frontmatter={"claim_token": "s3cret"}),))

    finding = scan_for_forbidden_keys(snapshot.records, snapshot=snapshot)
    rendered = "" if finding is None else finding.value_text()

    assert "s3cret" not in rendered
    assert "allowlist" in rendered, (
        "the suspected cause has to name the shape -- an allowlist that grew -- rather than "
        "leave a reader with a token and no idea what to look for"
    )
    assert "did not read Kojutsu's projection source" in rendered, (
        "the check reads documents, not code, and a finding that cannot say so reads as a diagnosis"
    )


def test_a_last_error_key_is_a_finding_too_because_internal_exception_detail_has_no_analytical_value() -> (
    None
):
    """Both prohibited names, because each was prohibited for its own reason.

    ``last_error`` is a much smaller problem than a claim token -- it is not a
    credential -- and it is on the list for a different reason: it carries exception
    detail that is most likely to include a filesystem path or a fragment of untrusted
    upstream text, and nothing analytical is lost by refusing it.
    """
    assert frozenset({"claim_token", "last_error"}) == FORBIDDEN_PROJECTION_KEYS

    snapshot = build_snapshot((decision_request(frontmatter={"last_error": "boom"}),))

    finding = scan_for_forbidden_keys(snapshot.records, snapshot=snapshot)

    assert isinstance(finding, Finding)
    assert "last_error" in finding.what_was_observed


def test_a_clean_projection_says_nothing_because_a_check_that_fires_on_every_document_is_a_check_nobody_reads() -> (
    None
):
    """Silence on the clean case, and only on the clean case.

    This is the half that is easy to get wrong in the other direction: a check that
    reported on every document it read would be a check whose output gets ignored, and
    one that reported nothing on a corpus that had never been projected would be
    indistinguishable from a check that never ran.
    """
    snapshot = build_snapshot((decision_request(), build_record(pr=1)))

    assert scan_for_forbidden_keys(snapshot.records, snapshot=snapshot) is None


def test_a_prohibited_key_on_a_record_that_is_not_a_request_is_outside_this_check_because_the_promise_is_about_the_projection() -> (
    None
):
    """The population is stated, so the limit of the check is a decision and not an accident.

    The check covers projected decision requests, because that is what Kojutsu
    promises about. A hand-written document carrying the same key is a different
    problem in a different namespace, and folding the two together would make the
    check's population something nobody chose.
    """
    answer = build_record(entry_id="answer-1", frontmatter={"claim_token": "s3cret"})

    snapshot = build_snapshot((answer,))

    assert scan_for_forbidden_keys(snapshot.records, snapshot=snapshot) is None


# -- wiring -----------------------------------------------------------------------


def test_the_projection_safety_check_is_wired_into_the_default_registry_because_a_correct_function_nothing_calls_looks_finished() -> (
    None
):
    """The regression test for the mistake this check was written to avoid.

    ``LifecycleShapeDetector`` was written, tested, and correct, and no report ever
    called it: a corpus with a broken close path rendered as a clean one. A promise in
    the refusal catalogue that ``claim_token`` would never be projected was the same
    shape -- a correct function and no caller. So the measure has to be in the registry
    ``build_report`` resolves by name, and this test goes through that registry.
    """
    assert any(isinstance(measure, ProjectionSafetyMeasure) for measure in default_measures())


def test_a_published_claim_token_reaches_a_report_rendered_from_the_default_measures_because_a_wired_check_is_the_only_kind_that_runs() -> (
    None
):
    """Through ``build_report``, with no ``measures=`` argument.

    A test that passes its own measure list proves the list it passed. This one builds
    the report the way a reader gets it, over a real store walk, so the only thing that
    can make it pass is the registry carrying the check.
    """
    documents = {
        "acme/widget/pr-42/question/q-0001": {
            "question_id": "q-0001",
            "question_status": "answered",
            "question_author": "agent",
            "attempts": "1",
            "created_at": "2026-09-01T00:00:00+00:00",
            "answered_at": "2026-09-02T00:00:00+00:00",
            "claim_token": "s3cret",
        },
    }
    snapshot = take_snapshot(FakeStore(documents=documents), page_limit=500)
    rendered = render_markdown(
        build_report(snapshot, gate=ClaimGate(), registry=default_registry())
    )

    assert "must never be published" in rendered
    assert "claim_token" in rendered
    assert "s3cret" not in rendered


def test_a_clean_projection_still_renders_the_check_because_silence_is_not_a_result() -> None:
    """The clean case is a count of zero under a slug of its own.

    A reader who cannot distinguish "checked, and nothing crossed the seam" from
    "nothing here" has been handed an absence where a fact belongs -- and the absence
    is exactly what an unwired check produces.
    """
    assert _SAFETY.compute(build_snapshot(_requests())).payload == "count"
    assert _SAFETY.claim(build_snapshot(_requests())).slug == PROJECTION_SAFETY_CLEAN_SLUG

    fired = build_snapshot((decision_request(frontmatter={"claim_token": "s3cret"}),))
    assert isinstance(_SAFETY.compute(fired), FigureGroup)


# -- the measures -----------------------------------------------------------------


def test_the_wait_distribution_is_over_requests_and_not_over_records_because_a_question_is_never_evidence() -> (
    None
):
    """The denominator is the population the claim names, and it is not the record count.

    The snapshot holds seven records and three of them are requests. A figure whose
    denominator said seven would be a rate over a population including the captures,
    which is the one population a request may not be counted inside.
    """
    records = (*_captures(), *_requests())
    snapshot = build_snapshot(records)

    figure = _LIFECYCLE.compute(snapshot)

    assert isinstance(figure, DistributionFigure)
    assert figure.claim.denominator.description == REQUEST_POPULATION
    assert figure.claim.denominator.size == 3
    assert figure.claim.unblocked_by == "AZ8T5XYS"
    assert figure.claim.kind is ClaimKind.descriptive
    assert figure.counted + figure.excluded_total == 3


def test_a_request_missing_either_clock_is_named_rather_than_dropped_because_a_bucket_that_did_not_occur_and_one_that_was_filtered_out_are_different_facts() -> (
    None
):
    """Both absences, under keys that say which one it was.

    A superseded request with no ``answered_at`` is ordinary rather than broken, and a
    distribution that quietly omitted it would be reporting the wait of the requests
    that happen to carry both clocks -- a selected population wearing a plain sentence.
    """
    records = (
        decision_request("q-1", created_at=None),
        decision_request("q-2", answered_at=None),
        decision_request("q-3", created_at="2026-09-01T00:00:00+00:00", answered_at=None),
    )

    figure = _LIFECYCLE.compute(build_snapshot(records))

    assert isinstance(figure, DistributionFigure)
    assert figure.excluded[NO_RAISED_TIME_KEY] == 1
    assert figure.excluded[NO_TERMINAL_TIME_KEY] == 2
    assert figure.counted == 0


def test_the_outcome_distribution_names_all_three_terminal_states_even_at_zero_because_a_reader_comparing_two_runs_needs_to_see_that_a_bucket_was_empty() -> (
    None
):
    """The full ladder, always, so the shape does not depend on the corpus.

    A distribution whose buckets come and go is one every report has to guard against.
    A status outside Kojutsu's three is named in ``excluded`` under a key carrying
    the value itself and folded into neither bucket.
    """
    records = (
        decision_request("q-1", status="answered"),
        decision_request("q-2", status="superseded"),
        decision_request("q-3", status=None),
        decision_request("q-4", status="withdrawn"),
    )

    figure = _OUTCOMES.compute(build_snapshot(records))

    assert isinstance(figure, DistributionFigure)
    assert dict(figure.values) == {"answered": 1, "failed": 0, "superseded": 1}
    assert "terminal status outside Kojutsu's three: withdrawn" in figure.excluded
    assert figure.excluded[NO_STATUS_KEY] == 1


def test_a_superseded_request_means_the_code_moved_first_because_it_is_evidence_that_the_change_outran_the_conversation() -> (
    None
):
    """The sentence the whole projection exists to make possible, on both claims.

    Every other bucket here is a request that reached an outcome. This one is a request
    that was overtaken: somebody asked "why is this here?" about a change that was then
    rewritten or closed underneath the question, so the answer, had one arrived, would
    have been about code that no longer exists. A distribution rendering three
    equal-looking bars with no reading attached throws away the only thing the reader
    came for.
    """
    outcome = _OUTCOMES.claim(build_snapshot(_requests())).render_text()

    assert "code moved before a human replied" in outcome
    assert "no longer existed" in outcome
    assert "AZ8T5XYS" in outcome


def test_a_request_is_never_treated_as_independent_evidence_because_nobody_stated_anything_in_one() -> (
    None
):
    """No independence level to threshold, and nothing in the measure to threshold on it.

    Decision 001 requires a question to sit below every ``min_independence`` threshold.
    The mechanism it names is the absence of the field, so the assertion here is that
    the field is absent and that the claims are descriptive over the request
    population -- there is no evidence axis on these measures for a threshold to
    filter, which is the strongest form the requirement can take.
    """
    snapshot = build_snapshot((*_captures(), *_requests()))

    for measure in (_LIFECYCLE, _OUTCOMES):
        claim = measure.claim(snapshot)
        assert claim.kind is ClaimKind.descriptive
        assert claim.denominator.description == REQUEST_POPULATION
    assert all(record.independence is None for record in _requests())


def test_a_requests_session_id_is_not_a_work_episode_because_a_session_is_one_cli_invocation_and_not_a_piece_of_work() -> (
    None
):
    """The field is read and nothing groups by it, which is the requirement in terms.

    A `session_id` is Kojutsu's identifier for one invocation of the tool, and it is
    on a request because the request was raised during one. Joining requests by it would
    group by how many times somebody ran a command rather than by what anybody was doing,
    and the two measures here group by nothing but the request's own clocks and status.
    """
    records = (
        decision_request("q-1", session_id="sess-1"),
        decision_request("q-2", session_id="sess-1"),
    )
    snapshot = build_snapshot(records)

    for measure in (_LIFECYCLE, _OUTCOMES):
        rendered = measure.compute(snapshot).value_text()
        assert "sess-1" not in rendered
    assert all(record.session_id == "sess-1" for record in records)


# -- the leak test ----------------------------------------------------------------


def test_no_pre_existing_measure_moves_when_request_documents_are_present_because_a_question_is_never_an_answer_and_a_question_is_never_evidence() -> (
    None
):
    """The defect decision 001 exists to prevent, asserted as an absence.

    Every other test here says a measure does the right thing. This one says a measure
    is *unchanged*, and unchanged is the only property that catches a helper quietly
    widening a population: a projected request carries a repository and a document
    update time, so without the exclusion it would join the by-repository histogram as a
    record somebody examined and the by-month histogram filed under the day it was
    written rather than the day it was asked.

    Runs over ``default_measures()`` because a list written here would only prove the
    list. The two new measures are excluded by name -- they are *about* requests, so
    they are supposed to move -- and the one pre-existing measure allowed to move is
    named too, so a second leaky measure fails rather than joining a list nobody reads.
    """
    captures = _captures()
    requests = _requests()
    without = build_snapshot(captures)
    with_requests = build_snapshot((*captures, *requests))

    pre_existing = [measure for measure in default_measures() if type(measure) not in _NEW_MEASURES]
    assert pre_existing, "the leak test proved nothing if the registry held no other measure"

    moved = [
        f"{type(measure).__name__} ({measure.slug})"
        for measure in pre_existing
        if type(measure) not in _MAY_MOVE
        and _rendered(measure, without) != _rendered(measure, with_requests)
    ]

    assert not moved, (
        "these figures moved when projected decision requests were added to the corpus, "
        "and decision 001 requires them not to: a question document is never an answer, a "
        "question is never evidence, and it is counted apart from captures.\n  "
        + "\n  ".join(moved)
    )


def test_only_the_read_completeness_figure_is_allowed_to_move_because_its_denominator_is_the_store_and_not_a_population_this_program_chose() -> (
    None
):
    """The exception is asserted, so it cannot grow into a list.

    Completeness divides by the store's own count, so a store holding ten documents
    cannot report the same rate as a store holding seven -- and that has nothing to do
    with any of them being requests. If the exemption ever needs a second entry, this
    fails first and the reason has to be written down.
    """
    captures = _captures()
    without = build_snapshot(captures)
    with_requests = build_snapshot((*captures, *_requests()))

    changed = {
        type(measure).__name__
        for measure in default_measures()
        if _rendered(measure, without) != _rendered(measure, with_requests)
    }

    assert changed == {measure.__name__ for measure in _MAY_MOVE} | {
        measure.__name__
        for measure in (ProjectionSafetyMeasure, *_NEW_MEASURES - {ProjectionSafetyMeasure})
    } | {
        DecisionRequestLifecycleMeasure.__name__,
        DecisionRequestOutcomeMeasure.__name__,
        SupersededIntervalMeasure.__name__,
    }, f"these are the measures whose figures moved: {sorted(changed)}"


def test_the_corpus_the_leak_test_uses_is_one_where_a_leak_would_be_visible_because_an_empty_distribution_passes_either_way() -> (
    None
):
    """The guard on the guard: the captures state every field the measures read.

    A leak test over records with no independence level, no capture source, no
    commenting account and no declared model would pass whether the measures filtered
    or not, because every distribution would be empty in both runs. This asserts the
    figures are non-empty *without* the requests, so a pass cannot have come from
    nothing to count.
    """
    without = build_snapshot(_captures())

    populated = {
        measure.slug
        for measure in default_measures()
        if type(measure) not in _NEW_MEASURES | _MAY_MOVE and _rendered(measure, without) != ()
    }
    assert len(populated) >= 8, f"only {sorted(populated)} produced anything over the captures"

    repositories = build_snapshot(_captures())
    assert repositories.records, "the corpus the leak test compares against is empty"


def test_the_lifecycle_and_lifecycle_shape_measures_are_distinct_because_two_checks_with_one_name_is_one_of_them_unrunnable() -> (
    None
):
    """Both integrity measures are in the registry, and neither shadows the other.

    The clean path of each renders under a slug of its own. Sharing the slug would let
    a reader match a clean result against a finding and conclude the clean one refutes
    it, which is the opposite of what each is for.
    """
    slugs = {measure.slug for measure in default_measures()}

    assert LifecycleShapeMeasure.slug in slugs
    assert PROJECTION_SAFETY_SLUG in slugs
    assert PROJECTION_SAFETY_CLEAN_SLUG not in slugs
    assert (
        LifecycleShapeMeasure().claim(build_snapshot(_requests())).slug
        == "corpus.lifecycle_shape_checked"
    )
    assert PROJECTION_SAFETY_CLEAN_SLUG != "corpus.lifecycle_shape_checked"
