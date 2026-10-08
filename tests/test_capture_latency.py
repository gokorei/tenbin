"""How long the corpus took to hold an answer, and how long a superseded question waited.

Two intervals, both already in the corpus and both unmeasured, and the argument for
each is that a reader would otherwise read a different system than the one the number
is about.

**Capture latency measures the path, and the path starts at the webhook.** Kojutsu
stamps ``captured_at`` when it builds the entry and before it enqueues anything, so the
interval from ``answered_at`` to ``captured_at`` is the worker's own turnaround and
whatever the build had to read -- and *not* the store's write, which happens after the
stamp, and not the forge's delivery, which finished before the interval started. A
reader who gets that wrong reads a delivery spike as the forge being slow, so the
figure has to say which system it is about rather than let the name do it.

**It is a distribution and never a summary, because the distribution is bimodal.** The
same tick that makes the interval honest is a tick that retries, so the bulk of a corpus
is the ordinary path and a tail of rows sat behind a store outage. One number over two
modes describes neither, and the tests here assert on the shape rather than on a
statistic precisely because the shape is the finding.

**A record that cannot be measured is named, not dated and not dropped.** Three different
absences get three different keys -- a record with no start, one with no end, and one
whose two clocks disagree -- because a reader going to fix one of them needs to know
which they are holding. The backwards clock is the one worth a test of its own: a
negative interval rendered as a fast capture reports a data defect as an achievement.

**The superseded interval measures code movement and non-response at once.** That is
the sentence the claim exists to carry, and the test asserts it in the claim's own words
because a paraphrase is where the caveat goes to die.

**The last test is a prohibition, and it is the one this ticket could most easily have
broken.** The plan for this work said not to combine a latency with an outcome, and the
easiest way to comply is to not do it; the test is here so that a later measure cannot do
it either. It walks :func:`tenbin.measures.default_measures` -- read-only, imported and
not edited -- plus the two measures this ticket adds, because a prohibition that only
covered registered measures would not cover a measure that is one import away.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Final

from tenbin.claims.gate import ClaimGate
from tenbin.claims.model import ClaimKind
from tenbin.corpus.record import Record, RecordKind
from tenbin.corpus.snapshot import CorpusSnapshot
from tenbin.measures import default_measures
from tenbin.measures.base import (
    NEGATIVE_INTERVAL,
    DistributionFigure,
    Figure,
    FigureGroup,
    Measure,
    bucket_labels,
)
from tenbin.measures.capture_latency import (
    CAPTURE_LATENCY_POPULATION,
    CAPTURE_LATENCY_SLUG,
    NO_ANSWERED_AT_KEY,
    NO_CAPTURE_SOURCE_KEY,
    NO_CAPTURED_AT_KEY,
    TYPED_IN_KEY,
    CaptureLatencyMeasure,
)
from tenbin.measures.decision_requests import (
    NO_TERMINAL_TIME_KEY,
    SUPERSEDED_INTERVAL_SLUG,
    SUPERSEDED_POPULATION,
    DecisionRequestLifecycleMeasure,
    SupersededIntervalMeasure,
)
from tests.fixtures import build_record, build_snapshot, decision_request, truncating_snapshot

_LATENCY = CaptureLatencyMeasure()
_SUPERSEDED = SupersededIntervalMeasure()
_LIFECYCLE = DecisionRequestLifecycleMeasure()

#: The vocabulary a duration axis is written in, and the vocabulary an outcome axis is
#: written in. Both named here rather than imported, because the whole point of the last
#: test is that the two cannot be told apart once they are in the same figure -- and a
#: check that read the buckets out of the same module it was checking would be a check
#: that could be satisfied by renaming.
_DURATION_TERMS: Final[frozenset[str]] = frozenset(bucket_labels())
_OUTCOME_TERMS: Final[frozenset[str]] = frozenset(
    {"answered", "failed", "superseded", "merged", "closed_unmerged", "approved"}
)

#: The record fields that carry an outcome rather than a clock. Named separately from
#: :data:`_OUTCOME_TERMS` because the two checks ask different questions: the first asks
#: what a rendered bucket says, and the second asks what a source file is willing to read.
#: A measure may legitimately read one of these to *select* a population -- the superseded
#: measure reads ``question_status`` -- and the check that forbids it is scoped to the module
#: where no such reading has any business existing.
_OUTCOME_FIELDS: Final[frozenset[str]] = frozenset(
    {"pr_outcome", "pr_merged_at", "check_conclusion", "check_name", "question_status"}
)


def _capture(
    *,
    pr: int,
    answered_at: str | None = "2026-01-02T09:00:00+00:00",
    captured_at: str | None = "2026-01-02T09:02:00+00:00",
    capture_source: str | None = "webhook",
) -> Record:
    """One capture, with the two clocks the interval is built from.

    The default pair is a two-minute delivery: fast enough to land in the first rung, and
    a real number rather than zero, because a zero interval is indistinguishable from a
    record written and stamped in the same instant and would make a test of the *shape*
    unable to tell the ordinary path from a degenerate corpus.
    """
    return build_record(
        entry_id=f"answer-{pr:04d}",
        pr=pr,
        record_kind=RecordKind.ANSWER,
        capture_source=capture_source,
        answered_at=answered_at,
        captured_at=captured_at,
    )


def _bucket(figure: DistributionFigure, label: str) -> int:
    """The count in one named rung, failing loudly on a rung the ladder does not have.

    A subscript would do the same thing, so this is here for the sentence rather than the
    behaviour: a test that says "these many records took this long" reads as the claim it is
    checking, where ``figure.values[...]`` reads as a lookup into a mapping whose shape is
    the thing under test. A ``KeyError`` naming the missing rung is also a better failure
    than a silent zero, which reads as "nothing landed there".
    """
    return figure.values[label]


def _members(figure: Figure) -> tuple[Figure, ...]:
    """The figure and everything under it, so a group's members are compared too."""
    if isinstance(figure, FigureGroup):
        return (figure, *(member for child in figure.figures for member in _members(child)))
    return (figure,)


def _captures() -> tuple[Record, ...]:
    """A corpus where a delivery spike is visible, because nothing here is slow by default.

    Every record states both clocks and says a delivery path built it, so the figure is
    non-empty and the ordinary path is the thing a reader is comparing the tail against.
    """
    return tuple(_capture(pr=pr) for pr in range(1, 9))


def _latency_figure(records: tuple[Record, ...]) -> DistributionFigure:
    """The capture-latency distribution over a corpus, asserting the type on the way out."""
    figure = _LATENCY.compute(build_snapshot(records))
    assert isinstance(figure, DistributionFigure)
    return figure


# -- a distribution, over the records that carry both clocks ------------------------------


def test_each_measure_is_cited_by_the_slug_its_own_figure_carries_because_the_catalogue_keys_on_one_and_the_report_on_the_other() -> (
    None
):
    """``slug`` and ``claim().slug`` are two ways of naming the same measure, and they agree.

    A refusal is keyed by ``slugify(measure_name)`` and a report indexes figures by the
    claim's own slug, so a measure whose two spellings drifted would be citable by a name
    that never renders and would render under a name the catalogue does not know. Dotted, for
    the same reason every other slug in the package is: a slug that could collide with a
    row in ``docs/seam.md`` is a slug that will.
    """
    snapshot = build_snapshot(_captures())

    for measure in (_LATENCY, _SUPERSEDED):
        slug = measure.slug
        assert slug == measure.claim(snapshot).slug
        assert slug.count(".") == 1, f"{slug} is not dotted the way the package's slugs are"


def test_the_latency_is_bucketed_over_the_records_carrying_both_clocks_and_the_denominator_is_those_records() -> (
    None
):
    """The ordinary case: a distribution, and a denominator describing it.

    The figure is over the records that name both moments and not over the read, because
    the read holds documents this interval says nothing about -- a check run, a projected
    request -- and a denominator naming those would be describing a population the figure
    does not show.
    """
    records = _captures()

    figure = _latency_figure(records)

    assert figure.claim.slug == CAPTURE_LATENCY_SLUG
    assert figure.claim.denominator.description == CAPTURE_LATENCY_POPULATION
    assert figure.claim.denominator.size == len(records)
    assert figure.counted == len(records)
    assert figure.excluded == {}
    assert figure.claim.kind is ClaimKind.descriptive
    assert ClaimGate().check(figure.claim) is None


def test_a_normal_corpus_puts_its_whole_mass_in_the_fastest_bucket_because_two_minutes_is_the_ordinary_path() -> (
    None
):
    """The baseline the tail is read against.

    A test of the tail is worthless without a test of the shape underneath it: if the
    ordinary path drifted into the second rung, the long-tail test would still pass while
    the figure had stopped describing a corpus where delivery is quick.
    """
    figure = _latency_figure(_captures())

    assert _bucket(figure, "under 1 hour") == 8
    assert figure.counted == 8


def test_a_truncated_read_renders_the_truncation_sentence_because_a_figure_that_cannot_be_asked_whether_it_is_whole_is_a_number_with_nothing_behind_it() -> (
    None
):
    """The snapshot travels with the figure, so a prefix cannot render as a whole corpus.

    Eight records on a store holding eleven is the most confident and most wrong number
    available, and the sentence beside it is the only thing standing between it and a
    reader who believes it.
    """
    figure = _LATENCY.compute(truncating_snapshot(_captures(), missing=3, offset=500))

    assert "prefix" in figure.rate_text()
    assert "3 records were not reached" in figure.rate_text()


# -- a store outage is a tail, not a summary -------------------------------------------------


def test_a_store_outage_shows_up_as_a_tail_of_records_rather_than_as_one_number() -> None:
    """The bimodality, scripted: an ordinary corpus with a cluster that sat behind an outage.

    The point is not that the outliers exist -- any corpus has those -- but that the figure
    can hold both modes at once without one of them summarising the other. A mean here
    would move by hours when a single row shifted, and a percentile would report whichever
    mode it happened to land in without saying so.
    """
    ordinary = tuple(_capture(pr=pr) for pr in range(1, 8))
    behind_an_outage = tuple(
        _capture(
            pr=pr,
            answered_at="2026-01-02T09:00:00+00:00",
            captured_at=f"2026-01-{day:02d}T09:00:00+00:00",
        )
        for pr, day in ((8, 4), (9, 6))
    )

    figure = _latency_figure((*ordinary, *behind_an_outage))

    assert _bucket(figure, "under 1 hour") == 7, "the ordinary path is its own mode, not noise"
    assert _bucket(figure, "1d to 3d") == 1
    assert _bucket(figure, "3d to 7d") == 1
    assert figure.counted == 9


def test_the_duration_ladder_is_rendered_in_full_so_the_shape_does_not_depend_on_the_corpus() -> (
    None
):
    """Every rung present at zero, for the reason the ladder is shared.

    A distribution whose buckets come and go is one every report has to guard against, and
    a reader comparing two reads needs the empty rungs to be there so they can see they
    were empty rather than infer it from their absence.
    """
    figure = _latency_figure(_captures())

    assert tuple(figure.values) == bucket_labels()
    assert set(figure.values) & _DURATION_TERMS == _DURATION_TERMS


def test_the_claim_refuses_a_mean_and_a_percentile_because_the_tail_is_an_outage_and_the_bulk_is_the_ordinary_path() -> (
    None
):
    """The two sentences that stop the number being quoted, asserted in the claim's words.

    A paraphrase here is where the caveat goes: "it does not report averages" drops that the
    distribution is *bimodal*, and the bimodality is the whole reason. So the claim says
    both, and this test holds it to both.
    """
    claim = _LATENCY.claim(build_snapshot(_captures()))

    assert "bimodal" in claim.does_not_mean
    assert "the tail is rows that sat behind a store outage" in claim.does_not_mean
    assert "not a mean or a percentile" in claim.does_not_mean


# -- the interval is not the forge's latency and not the store's ----------------------------


def test_the_claim_says_the_interval_runs_from_the_webhook_to_the_stamp_because_a_reader_who_assumes_otherwise_chases_the_wrong_system() -> (
    None
):
    """The load-bearing sentence, and the reason it is there.

    ``captured_at`` is stamped when Kojutsu builds the entry, before the enqueue. So the
    store's write is downstream of the figure and the forge's delivery is upstream of it,
    and a reader who reads a spike as either one goes looking in a system the number cannot
    speak about. The claim names the stamp, the enqueue, and both wrong systems.
    """
    rendered = _LATENCY.claim(build_snapshot(_captures())).render_text()

    assert "when the webhook fires" in rendered
    assert "at build time, before the enqueue" in rendered
    assert "How slowly the forge delivers, and not how slowly the store writes" in rendered
    assert "The store's write happens after the stamp" in rendered


# -- every unmeasurable record is named, and a backwards clock is named separately -----------


def test_a_record_missing_the_answer_time_is_excluded_under_a_key_naming_which_clock_it_lacks() -> (
    None
):
    """The two one-sided records, under two keys.

    A record with no ``answered_at`` has no start and one with no ``captured_at`` has no
    end. Folding them into one key would send a reader to look for the wrong missing field,
    and the two absences are different defects in different writers.
    """
    records = (
        _capture(pr=1, answered_at=None),
        _capture(pr=2, captured_at=None),
        _capture(pr=3),
    )

    figure = _latency_figure(records)

    assert figure.excluded[NO_ANSWERED_AT_KEY] == 1
    assert figure.excluded[NO_CAPTURED_AT_KEY] == 1
    assert figure.counted == 1, "the record with both clocks is still measured"
    assert figure.counted + figure.excluded_total == len(records)


def test_a_record_somebody_typed_in_is_excluded_because_no_delivery_path_was_ever_behind_it() -> (
    None
):
    """``asserted`` is not a fast capture, it is not a capture at all.

    A person typed it in, so its two timestamps are one person writing twice and the
    interval between them measures nothing about the pipeline. Folding it into the fast
    bucket would make a hand-written record look like the pipeline's best work.
    """
    records = (_capture(pr=1, capture_source="asserted"), _capture(pr=2))

    figure = _latency_figure(records)

    assert figure.excluded[TYPED_IN_KEY] == 1
    assert figure.counted == 1


def test_a_record_saying_nothing_about_its_provenance_is_excluded_because_silence_is_not_a_claim_of_assertion() -> (
    None
):
    """A fourth condition, kept apart from a typed-in record.

    A record with no ``capture_source`` and a record somebody typed in are both outside the
    interval, and they are different facts: one says a person typed it, the other says
    nothing at all about how it arrived. Merging the two counts would report a broken writer
    as a quiet corpus.
    """
    records = (_capture(pr=1, capture_source=None), _capture(pr=2))

    figure = _latency_figure(records)

    assert figure.excluded[NO_CAPTURE_SOURCE_KEY] == 1
    assert TYPED_IN_KEY not in figure.excluded
    assert figure.counted == 1


def test_a_capture_before_the_answer_is_excluded_rather_than_bucketed_because_a_clock_that_disagrees_with_itself_is_not_a_fast_capture() -> (
    None
):
    """A negative interval is a fact about the corpus and never the fastest bucket.

    Folding it into ``under 1 hour`` would report skew between two writers as the
    pipeline's best performance, which is the specific inversion the exclusion exists to
    stop. The key is the ladder's own label rather than a new string, so a reader meets one
    phrase for this condition wherever they meet it.
    """
    records = (
        _capture(
            pr=1,
            answered_at="2026-01-02T09:05:00+00:00",
            captured_at="2026-01-02T09:00:00+00:00",
        ),
        _capture(pr=2),
    )

    figure = _latency_figure(records)

    assert figure.excluded[NEGATIVE_INTERVAL] == 1
    assert _bucket(figure, "under 1 hour") == 1, "the backwards clock did not become a fast capture"
    assert NEGATIVE_INTERVAL not in figure.values
    assert figure.counted + figure.excluded_total == len(records)


# -- how long a superseded question waited ---------------------------------------------------


def _superseded(
    *,
    question_id: str,
    created_at: str | None = "2026-09-01T00:00:00+00:00",
    answered_at: str | None = "2026-09-02T00:00:00+00:00",
) -> Record:
    """One superseded request, over the two clocks the interval is built from."""
    return decision_request(
        question_id,
        status="superseded",
        created_at=created_at,
        answered_at=answered_at,
    )


def test_the_superseded_interval_is_measured_over_superseded_requests_and_not_over_the_other_terminal_states() -> (
    None
):
    """One state, and the arithmetic that proves it.

    The buckets plus the exclusions add up to the denominator, and the denominator counts
    requests and not records. A corpus holding an answered request and a failed one as well
    contributes neither a bucket nor an exclusion here: they are outside a population
    defined by the state, and mixing them in would put answers arriving and questions being
    overtaken in the same distribution.
    """
    records = (
        decision_request("q-answered", status="answered", answered_at="2026-09-01T06:00:00+00:00"),
        decision_request("q-failed", status="failed", answered_at="2026-09-01T18:00:00+00:00"),
        _superseded(question_id="q-1"),
        _superseded(
            question_id="q-2",
            created_at="2026-09-01T00:00:00+00:00",
            answered_at="2026-09-05T00:00:00+00:00",
        ),
    )

    figure = _SUPERSEDED.compute(build_snapshot(records))
    assert isinstance(figure, DistributionFigure)
    assert figure.claim.slug == SUPERSEDED_INTERVAL_SLUG
    assert figure.claim.denominator.description == SUPERSEDED_POPULATION
    assert figure.claim.denominator.size == 2
    assert figure.counted + figure.excluded_total == figure.claim.denominator.size
    assert _bucket(figure, "1d to 3d") == 1
    assert _bucket(figure, "3d to 7d") == 1
    assert figure.counted == 2, "the answered and failed requests contributed nothing"


def test_a_superseded_request_with_no_terminal_clock_is_named_because_a_projection_that_does_not_record_the_moment_leaves_a_hole_and_not_a_zero() -> (
    None
):
    """The absence is counted, under the key its sibling measure already uses.

    A superseded request whose projection carries no terminal time is a real state, and it
    is the normal case where the registry never recorded when the code moved. Writing it as
    zero would make the corpus look like it overtakes questions the instant they are asked,
    so it is counted under one vocabulary -- the same key the lifecycle figure uses -- and
    the figure reports its own emptiness rather than a guess.
    """
    records = (
        _superseded(question_id="q-1"),
        _superseded(question_id="q-2", answered_at=None),
    )

    figure = _SUPERSEDED.compute(build_snapshot(records))
    assert isinstance(figure, DistributionFigure)
    assert figure.excluded[NO_TERMINAL_TIME_KEY] == 1
    assert figure.counted == 1
    assert figure.claim.denominator.size == 2


def test_the_superseded_claim_says_the_interval_is_code_movement_and_non_response_at_once_because_nothing_records_why_a_question_went_unasked() -> (
    None
):
    """The ambiguity is the claim, asserted in the words the claim uses.

    A superseded question is not a wasted one: Kojutsu supersedes when the code moves, so
    the same interval is a statement about how fast the codebase moves and about how fast
    people answer, and the corpus holds nothing that says which produced a particular wait.
    The figure is only actionable if the reader is told that before they act on it.
    """
    rendered = _SUPERSEDED.claim(build_snapshot((_superseded(question_id="q-1"),))).render_text()

    assert "code moved before a human replied" in rendered, (
        "the interval claim carries the sentence the outcome figure already carries, so a "
        "reader meeting one of the two figures is not told two subtly different things"
    )
    assert "not a wasted question" in rendered
    assert "the rate of code movement and the rate of non-response" in rendered
    assert "cannot separate them" in rendered
    assert "records why a question went unasked" in rendered


def test_the_superseded_interval_reads_the_same_terminal_clock_as_the_lifecycle_figure_because_a_reader_comparing_two_distributions_has_to_be_comparing_two_populations() -> (
    None
):
    """The two figures agree on what "when the request ended" means, and it is asserted.

    ``status_as_of`` is when the *projection* noticed, so timing against it would measure
    the outbox a second time. Both distributions therefore read ``answered_at``, and this
    gives one superseded request a terminal clock seven days before its status was observed:
    the lifecycle figure and this one must put it in the same rung, and a measure that had
    reached for ``status_as_of`` would not.
    """
    records = (
        decision_request(
            "q-1",
            status="superseded",
            created_at="2026-09-01T00:00:00+00:00",
            answered_at="2026-09-02T00:00:00+00:00",
            status_as_of="2026-09-08T00:00:00+00:00",
        ),
    )
    snapshot = build_snapshot(records)

    superseded = _SUPERSEDED.compute(snapshot)
    lifecycle = _LIFECYCLE.compute(snapshot)
    assert isinstance(superseded, DistributionFigure)
    assert isinstance(lifecycle, DistributionFigure)

    assert _bucket(superseded, "1d to 3d") == 1
    assert _bucket(lifecycle, "1d to 3d") == 1
    assert _bucket(superseded, "3d to 7d") == 0, "the projection noticing is not the request ending"


# -- the prohibition this ticket must not break -----------------------------------------------


def _module_of(measure: object) -> Path:
    """The file a measure is written in, so the prohibition can be read over its source."""
    source = inspect.getsourcefile(type(measure))
    assert source is not None
    return Path(source)


def _members_of(measure: Measure, snapshot: CorpusSnapshot) -> tuple[Figure, ...]:
    """Every figure a measure produces, and every figure under it."""
    return tuple(member for member in _members(measure.compute(snapshot)))


def _measures_under_the_prohibition() -> tuple[Measure, ...]:
    """The registry, plus the two measures this ticket adds.

    Read-only over :func:`tenbin.measures.default_measures`: a prohibition that only
    covered the registered measures would not cover a measure one import away, which is the
    same gap the whole-package scan in ``tests/test_anchor_measures.py`` exists to close.
    """
    return (*default_measures(), CaptureLatencyMeasure(), SupersededIntervalMeasure())


def _mixed(records: tuple[Record, ...]) -> list[str]:
    """The measures whose figures put a duration and an outcome in the same figure.

    Read over rendered output rather than over source, because the thing being forbidden is
    a *number* holding both -- and a source-level check would fire on a module that
    *discusses* an outcome in a docstring, which is a rule that gets switched off.
    """
    snapshot = build_snapshot(records)
    offenders: list[str] = []
    for measure in _measures_under_the_prohibition():
        for figure in _members_of(measure, snapshot):
            if not isinstance(figure, DistributionFigure):
                continue
            keys = frozenset(figure.values)
            if keys & _DURATION_TERMS and keys & _OUTCOME_TERMS:
                offenders.append(f"{type(measure).__name__} ({figure.payload})")
    return offenders


def test_no_measure_puts_a_duration_and_an_outcome_in_one_figure_because_a_latency_scored_against_an_outcome_is_a_causal_claim_with_no_assignment_mechanism() -> (
    None
):
    """The composite prohibition, applied to the pair this ticket put next to each other.

    A merge outcome multiplied by a check conclusion is the composite decision 002 refuses,
    and a capture latency tabulated against what happened to the change is the same
    transformation with a delivery costume: two descriptive facts joined into one number
    that reads as a cause. The measure must stay the latency, and the outcome must stay its
    own figure with its own claim.
    """
    corpus = (
        *_captures(),
        _capture(pr=9, captured_at="2026-01-04T09:00:00+00:00"),
        build_record(
            entry_id="pr-event-0010",
            pr=10,
            record_kind=RecordKind.PR_LIFECYCLE,
            tags=["pr_state_change", "action_closed"],
            capture_source="webhook",
            pr_outcome="merged",
        ),
        decision_request("q-1", status="superseded", answered_at=None),
    )

    offenders = _mixed(corpus)

    assert not offenders, (
        "these figures put a duration bucket and an outcome in one distribution, which is the "
        "composite score decision 002 refuses: " + ", ".join(offenders) + ". Latency and "
        "outcome stay separate figures with separate claims."
    )


def test_the_corpus_the_prohibition_is_checked_on_is_one_where_a_composite_would_be_visible_because_an_empty_distribution_passes_either_way() -> (
    None
):
    """The guard on the guard, mirroring the leak test in ``tests/test_decision_requests.py``.

    A prohibition checked over a corpus that produces no durations and no outcomes proves
    nothing: the check would pass because there was nothing to mix. So the same corpus is
    asserted to produce both vocabularies, and the measure under test is asserted to be one
    of the producers.
    """
    corpus = (
        *_captures(),
        build_record(
            entry_id="pr-event-0010",
            pr=10,
            record_kind=RecordKind.PR_LIFECYCLE,
            tags=["pr_state_change", "action_closed"],
            capture_source="webhook",
            pr_outcome="merged",
        ),
    )
    snapshot = build_snapshot(corpus)

    vocabularies = set()
    for measure in _measures_under_the_prohibition():
        for figure in _members_of(measure, snapshot):
            if not isinstance(figure, DistributionFigure):
                continue
            keys = frozenset(figure.values)
            if keys & _DURATION_TERMS:
                vocabularies.add("duration")
            if keys & _OUTCOME_TERMS:
                vocabularies.add("outcome")

    assert vocabularies == {"duration", "outcome"}, (
        f"the corpus produces {sorted(vocabularies)}, so a composite would not have been visible"
    )
    assert tuple(_latency_figure(corpus).values) == bucket_labels()


def _fields_read(path: Path) -> set[str]:
    """The corpus fields a source file *reads*, as opposed to the ones it discusses.

    Parsed rather than grepped, for the reason
    ``tests/test_anchor_measures.py`` parses rather than greps: a substring search would
    fail on the module that argues the prohibition, and a rule that punishes writing the
    prohibition down is a rule that gets switched off. Docstrings are dropped and what
    remains is the code -- an attribute read, or a string where a field name would be looked
    up. Either is a read; prose about one is not.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }
    read: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            read.add(node.attr)
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
        ):
            read.add(node.value)
    return read


def test_the_capture_latency_module_reads_no_outcome_field_because_a_latency_bucketed_by_what_happened_to_the_change_is_the_composite_decision_002_refuses() -> (
    None
):
    """The source half of the prohibition, over the one module this ticket added.

    The rendered-output check above catches a figure that has already been built with both
    vocabularies in it. This catches the earlier and quieter version: a measure whose
    *population* is selected by an outcome -- every merged change's latency, say -- which
    renders one vocabulary and reads the other. That is the shape a duration-plus-outcome
    composite takes before anybody multiplies anything, so the module is read directly and
    the offender named.
    """
    read = _fields_read(_module_of(CaptureLatencyMeasure()))

    assert not read & _OUTCOME_FIELDS, (
        "capture_latency.py reads "
        + ", ".join(sorted(read & _OUTCOME_FIELDS))
        + ". Latency is how long the pipeline took and an outcome is what happened to the "
        "change; selecting a latency population by an outcome is the first step towards the "
        "composite score, and the two stay separate measures with separate claims."
    )
