"""Decision requests at terminal state: how long they sat, and how they ended.

Kojutsu's ``AZ8T5XYS`` projected the question registry into Tanseki, and the corpus
inventory's verdict on the result was that it is the closest thing this corpus has to
a decision-debt dataset. This module is the measure that reads it, and it is two
figures rather than one because the two questions are different: *how long did a
request sit* is about the wait, and *how did it end* is about whether anybody answered.

**The denominator is requests, never records.** That is decision 001 stated in terms
and it is the reason these measures exist separately rather than as a grouping inside
one of the capture figures. A projected request carries no ``capture_source`` and no
``independence`` -- the omission is the mechanism, not an oversight -- so it is not
evidence and cannot be counted as any. A reader who sees a rate here and cannot tell
what it is over has been handed the one number this package exists to make
unconstructible, so every denominator here names the population as projected decision
requests and counts them.

**The ``superseded`` bucket is the interesting one, and it deserves a sentence a
reader cannot skim.** A superseded request means the code moved before a human
replied: somebody asked "why is this here?" about a change that was then rewritten or
closed underneath the question, so the question was asked about something that no
longer existed and its answer, had one arrived, would have been about code that is
gone. Every other bucket here is a request that reached an outcome. This one is a
request that was overtaken, and it is the only one of the three that is evidence about
the relationship between the code and the questions being asked about it rather than
about the questions alone.

**The superseded bucket gets an interval of its own, and the reason is the same
sentence.** :class:`SupersededIntervalMeasure` reports how long those requests
waited before the code moved, over the ``superseded`` state alone rather than over
all three, because a distribution mixing the three would put answers arriving and
questions being overtaken in the same buckets. And it carries the caveat the bucket
alone cannot: the interval reads as the rate of code movement and as the rate of
non-response at the same time, and nothing in a projected request records which of
the two produced a particular wait. The other two states do not have that
ambiguity, which is why the interval figure is not a cut of the lifecycle figure but
a separate measure over a different population.

**The join is between two populations and never merges them.** ``question_id`` is
written on both sides -- on the projected request and on the knowledge entry that
answers it -- and until this ticket nothing in this program read it.
:func:`join_questions_and_captures` is that join, and it is the only place the two are
read together: the registry says a question was answered, the corpus says whether an
answer was kept, and those are two independent write paths whose ratio is a fact about
the pipeline rather than about anybody.

**The disagreement is the finding and the count is the smaller half of it.** The join
therefore splits the three ways the two sides can fail to correspond rather than
totalling them, because the three have three different causes and three different
remedies and one number is none of them: a registry row saying ``answered`` with no
document behind it, a capture for a question the registry placed at some other terminal
state, and a capture naming a question this read holds no request for at all. A measure
that reported only the ratio would turn a lost answer into a percentage, which is the
exact transformation this program exists to refuse. The descriptive half is
:class:`QuestionAnsweredRateMeasure` and the integrity half is
:class:`tenbin.measures.projection_safety.QuestionCaptureAgreementMeasure`.

**What this module notably does not do:** it does not measure the outstanding queue,
and the difference is not a gap. A ``pending`` or ``claimed`` request is operational
state -- rewritten on every lease -- and Kojutsu deliberately excludes it, because
the delivery path is built for immutable records and a live queue in a knowledge store
is a queue pretending to be knowledge. So "how long has the queue been" is refused
rather than approximated here, and the age below is the age of requests that *reached*
a terminal state, which is a different population and says so. It also does not treat
a request's ``session_id`` as a work-episode identity: a session is one CLI invocation
and not a piece of work, so joining requests by it would group by how many times
somebody ran a command rather than by what anybody was doing. And the join below adds
nothing to that list of things it will not do: a question document is never an answer
and a question is never evidence, so the two sides are read *across* each other and
neither is ever counted inside the other's denominator.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Final

from tenbin.claims.mechanism import Granularity
from tenbin.claims.model import Claim, ClaimKind, Denominator
from tenbin.corpus.record import Record, RecordKind
from tenbin.corpus.snapshot import CorpusSnapshot
from tenbin.measures.base import (
    DistributionFigure,
    Figure,
    FigureGroup,
    Rate,
    RateFigure,
    duration_buckets,
    freeze_counts,
    histogram,
)
from tenbin.measures.filtering import capture_population, select

#: The three states Kojutsu projects a decision request in, and the only three.
#: Spelled here rather than imported, for the reason ``tenbin.measures.outcome``
#: spells its two: the vocabulary is the writing project's, and a reader changing
#: these should be reading Kojutsu's source. It is a tuple rather than a set
#: because the rendered distribution walks it in this order, and a histogram whose
#: bucket order depended on set iteration would render differently between runs.
TERMINAL_QUESTION_STATUSES: Final[tuple[str, ...]] = ("answered", "failed", "superseded")

#: The ticket that made these measures possible, named on both claims. A claim that
#: does not say what unblocked it leaves a reader unable to tell a measure nobody has
#: thought about from one nobody had the data for.
UNBLOCKED_BY: Final[str] = "AZ8T5XYS"

#: What the population is, in the words the denominators render. Says *projected
#: decision requests* rather than *records*, because those are not the same population
#: and the word carries the exclusion.
REQUEST_POPULATION: Final[str] = "projected decision requests in this read"

#: The exclusion keys. Both name a condition rather than a code, because a report
#: renders them verbatim and ``"no_created_at"`` is not a sentence an operator can act
#: on. Both are also genuinely different absences: one is a request whose raising time
#: did not survive the projection, the other is a request that reached a terminal state
#: at a moment nobody recorded -- which is the normal case for a request that was
#: superseded rather than answered.
NO_RAISED_TIME_KEY: Final[str] = "request carries no time for when it was raised"
NO_TERMINAL_TIME_KEY: Final[str] = "request reached a terminal state with no time recorded for it"
NO_STATUS_KEY: Final[str] = "request carries no terminal status at all"
UNRECOGNISED_STATUS_PREFIX: Final[str] = "terminal status outside Kojutsu's three: "

#: The one sentence about ``superseded`` that both claims carry, because the bucket is
#: the reason this dataset was worth projecting and a distribution that renders three
#: equal-looking bars with no reading attached throws away the only thing the reader
#: came for. Written once so the lifecycle claim and the outcome claim cannot say
#: subtly different things about the same bucket.
SUPERSEDED_MEANS: Final[str] = (
    "A superseded request means the code moved before a human replied, so the question was "
    "asked about something that no longer existed: the request is the one record in this "
    "corpus that says the change outran the conversation about it, and it is evidence about "
    "that relationship rather than about the request."
)

#: The one terminal status that means a human said something, named here because the
#: join branches on it and a literal spelled at the branch would be the second copy of
#: Kojutsu's vocabulary. It is :data:`TERMINAL_QUESTION_STATUSES` that is the
#: vocabulary; this is a pointer to one of its members, which is a different kind of
#: statement and is checked against it rather than trusted.
ANSWERED_STATUS: Final[str] = "answered"

#: The terminal status the interval figure is over, named the same way and for the same
#: reason: it is a pointer to a member of :data:`TERMINAL_QUESTION_STATUSES` rather than a
#: second copy of Kojutsu's vocabulary, and a spelling of ``superseded`` at a filter
#: would be the one place in this module where the closed set is written out.
SUPERSEDED_STATUS: Final[str] = "superseded"

#: The slug the answered-and-captured rate is cited by. Dotted like every other slug
#: in this package, so it cannot collide with a slugified entry in ``docs/seam.md``.
ANSWERED_CAPTURE_SLUG: Final[str] = "decision_request.answered_with_a_capture"

#: What that rate is a rate over: the projected requests that *reached* a terminal
#: state, counted as requests. Not :data:`REQUEST_POPULATION`, because the numerator
#: is only defined over a request that resolved, and not ``capture records``, because
#: the numerator is a count of requests and a denominator of records beside it would
#: be a rate over a population the claim does not describe.
ANSWERED_CAPTURE_POPULATION: Final[str] = (
    "projected decision requests that reached a terminal state in this read"
)

#: The slug the superseded wait is cited by, dotted like every other slug here so it
#: cannot collide with a slugified row in ``docs/seam.md``.
SUPERSEDED_INTERVAL_SLUG: Final[str] = "decision_request.wait_before_supersession"

#: What the superseded figure is over: projected requests that reached one state, counted
#: as requests. Not :data:`REQUEST_POPULATION`, because a denominator naming every request
#: beside a figure counting only the overtaken ones would describe a population the figure
#: does not show, and not ``capture records``, because a request is not a capture.
SUPERSEDED_POPULATION: Final[str] = (
    "projected decision requests in this read that reached the superseded state"
)

#: The one sentence that makes the interval uninterpretable on its own, written once so the
#: outcome claim and the interval claim cannot say subtly different things about it. A
#: superseded request is not a wasted question: Kojutsu supersedes when the code moves, so
#: the interval reads as the rate of code movement and as the rate of non-response at the same
#: time, and the two cannot be told apart because nothing in a projected request records why a
#: question went unasked. This is the sentence the whole interval figure exists to carry.
SUPERSEDED_IS_AMBIGUOUS: Final[str] = (
    "A superseded question is not a wasted question. The interval is the rate of code movement "
    "and the rate of non-response at the same time, and this corpus cannot separate them, because "
    "nothing in a projected request records why a question went unasked: a request overtaken an "
    "hour after it was raised says the change moved inside that hour, and one overtaken after a "
    "month says the same about a month. So a large number here is a codebase outrunning its "
    "questions, and it is equally a conversation that stopped, and no figure can tell a reader "
    "which they are holding."
)

#: The two bucket suffixes the join's distribution is built from. They are constants
#: rather than inline f-strings so that a test asserting on a bucket label is asserting
#: on the same string the measure renders, and so that the "with a capture" half can be
#: selected without re-parsing a rendered sentence.
WITH_CAPTURE_SUFFIX: Final[str] = "with a capture naming it"
WITHOUT_CAPTURE_SUFFIX: Final[str] = "with no capture naming it"

#: The exclusion keys the join adds. Each names its condition *and* the population the
#: count belongs to, because the first two are not addable to the figure's denominator
#: and a reader who assumed they were would be reconciling a question count against a
#: capture count. A large ``NO_CAPTURE_QUESTION_ID_KEY`` is a normal state rather than a
#: finding: Kojutsu writes no question id on a capture taken by ``collect``, because
#: there is no question behind it, so a corpus captured rather than answered excludes
#: most of itself here and that is the corpus telling the truth about itself.
NO_REQUEST_QUESTION_ID_KEY: Final[str] = (
    "projected request stating no question id, so it cannot be joined "
    "(requests, so this adds up to the figure's denominator)"
)
NO_CAPTURE_QUESTION_ID_KEY: Final[str] = (
    "capture record stating no question id, as every capture taken by collect does, so it "
    "cannot be joined (captures, so this does not add up to the figure's denominator)"
)
NO_TERMINAL_REQUEST_KEY: Final[str] = (
    "capture naming a request this read holds with no terminal status, so the two sides "
    "cannot be compared at all"
)


def lifecycle_claim(total: int) -> Claim:
    """The claim behind the wait distribution, over the requests this read enumerated."""
    return Claim(
        slug="decision_request.wait_before_terminal_state",
        # Both measures exist because Kojutsu shipped the registry projection on
        # ``AZ8T5XYS``, so the unblocking ticket is named here rather than living only in
        # a merge commit. Lifting a refusal is a deliberate act and the record of why it
        # was lifted belongs in the code, next to the claim it lifted it for.
        source="Kojutsu AZ8T5XYS, the question-registry projection",
        statement=(
            "The decision requests in this corpus took the spread of times below to reach a "
            "terminal state, over the projected requests rather than over the records."
        ),
        does_not_mean=(
            "How long the queue is, or how long a decision takes. A pending or claimed request "
            "is operational state that the projection deliberately omits, so what is timed here "
            "is the wait of requests that already resolved: a reader who quotes this as a "
            "response time is quoting the age of finished requests and calling it a current "
            "one. It is also not a measure of anybody's diligence -- the wait ends when a human "
            "or an attempt ceiling ends it, and the corpus cannot say which. " + SUPERSEDED_MEANS
        ),
        falsifier=(
            "A read in which the distribution is dominated by requests whose interval is hours "
            "where this one is dominated by days or weeks. The buckets are the store's own "
            "duration ladder, so a corpus that shifted buckets wholesale would be describing a "
            "different population rather than the same one moving."
        ),
        denominator=Denominator(REQUEST_POPULATION, max(1, total)),
        kind=ClaimKind.descriptive,
        granularity=Granularity.corpus,
        unblocked_by=UNBLOCKED_BY,
    )


def outcome_claim(total: int) -> Claim:
    """The claim behind the terminal-state distribution, over the requests enumerated."""
    return Claim(
        slug="decision_request.terminal_state",
        # Both measures exist because Kojutsu shipped the registry projection on
        # ``AZ8T5XYS``, so the unblocking ticket is named here rather than living only in
        # a merge commit. Lifting a refusal is a deliberate act and the record of why it
        # was lifted belongs in the code, next to the claim it lifted it for.
        source="Kojutsu AZ8T5XYS, the question-registry projection",
        statement=(
            "The projected decision requests in this corpus reached the terminal states the "
            "figure below names, over the requests and not over the records they sit beside."
        ),
        does_not_mean=(
            "A rate of answering, or a measure of whether the questions were good. A request "
            "that was superseded and one that hit the attempt ceiling are both 'not answered', "
            "and they are different facts: one was overtaken by the code and one ran out of "
            "attempts, and only the first says anything about the change. " + SUPERSEDED_MEANS
        ),
        falsifier=(
            "A read whose projection writes a fourth terminal state, or one in which requests "
            "leave the `superseded` bucket entirely while the code's churn rate is unchanged. "
            "Either would mean the three buckets are not the closed set this figure is drawn "
            "over, and a reader would be comparing two different vocabularies."
        ),
        denominator=Denominator(REQUEST_POPULATION, max(1, total)),
        kind=ClaimKind.descriptive,
        granularity=Granularity.corpus,
        unblocked_by=UNBLOCKED_BY,
    )


def _requests(records: tuple[Record, ...]) -> tuple[Record, ...]:
    """The projected decision requests, and nothing else.

    Filtered on the *determined* kind rather than on
    :attr:`~tenbin.corpus.record.Record.is_question`, because a measure needs the
    writer's own kind to have been established while a safety check is happy to widen
    to the union. A request read from its tags and one read from its address are the
    same request, but only one of them is a kind Kojutsu stated, and a distribution
    of what the projection wrote should be over documents whose kind the projection
    projected.
    """
    return select(records, kinds=(RecordKind.QUESTION,))


@dataclass(frozen=True)
class DecisionRequestLifecycleMeasure:
    """How long each projected request sat before it reached a terminal state.

    Bucketed rather than summarised: a distribution of waits is skewed by construction
    -- most requests resolve quickly or not at all -- so a mean would be a point no
    observation occupies. The buckets are
    :func:`~tenbin.measures.base.duration_buckets`, the same ladder every other
    duration in this program uses, because two measures that disagreed about what a
    bucket is would be two definitions of the same axis.

    A request missing either clock is counted in ``excluded`` under a key naming which
    one it was missing, and never in a bucket. Dropping it silently would make the
    distribution a statement about the requests that happen to carry both timestamps,
    which is a selected population wearing a plain sentence.
    """

    #: Every projected request counts. The kind comes from Kojutsu's own tag
    #: projection, so requiring ``DETERMINED`` here would exclude nothing that the
    #: filter does not already exclude and would make the exclusion rules harder to
    #: read than the thing they exclude.
    requires_certainty = None

    @property
    def slug(self) -> str:
        """The identifier this measure is cited by."""
        return "decision_request.wait_before_terminal_state"

    def claim(self, snapshot: CorpusSnapshot) -> Claim:
        """The claim, over the projected requests this read enumerated."""
        return lifecycle_claim(len(_requests(snapshot.records)))

    def compute(self, snapshot: CorpusSnapshot) -> Figure:
        """The bucket distribution of the wait, with both absences named."""
        measured: list[float] = []
        missing_raised = 0
        missing_terminal = 0
        for record in _requests(snapshot.records):
            raised = record.created_at
            terminal = record.answered_at
            if raised is None:
                missing_raised += 1
                continue
            if terminal is None:
                missing_terminal += 1
                continue
            measured.append((terminal - raised).total_seconds() / 3600.0)

        excluded: dict[str, int] = {}
        if missing_raised:
            excluded[NO_RAISED_TIME_KEY] = missing_raised
        if missing_terminal:
            excluded[NO_TERMINAL_TIME_KEY] = missing_terminal
        return DistributionFigure(
            claim=self.claim(snapshot),
            snapshot=snapshot,
            title="How long a decision request sat before it reached a terminal state",
            values=duration_buckets(measured),
            excluded=excluded,
        )


@dataclass(frozen=True)
class DecisionRequestOutcomeMeasure:
    """Which terminal state each projected request reached, all three always present.

    Every rung of :data:`TERMINAL_QUESTION_STATUSES` is rendered even when nothing
    landed in it, for the reason :func:`~tenbin.measures.base.duration_buckets` does
    the same: a distribution whose shape depends on the corpus is one every report has
    to guard against, and a reader comparing two runs needs the empty buckets to be
    there so they can see they were empty rather than infer it from their absence.

    A status outside Kojutsu's three is counted in ``excluded`` under a key carrying
    the value itself, and is kept out of both the numerator and the denominator of the
    reading. Folding it into ``failed`` would report a vocabulary this reader does not
    recognise as an outcome somebody reached.
    """

    requires_certainty = None

    @property
    def slug(self) -> str:
        """The identifier this measure is cited by."""
        return "decision_request.terminal_state"

    def claim(self, snapshot: CorpusSnapshot) -> Claim:
        """The claim, over the projected requests this read enumerated."""
        return outcome_claim(len(_requests(snapshot.records)))

    def compute(self, snapshot: CorpusSnapshot) -> Figure:
        """The distribution across the three terminal states, plus whatever else arrived."""
        requests = _requests(snapshot.records)
        statuses = [record.question_status for record in requests]
        stated = [status for status in statuses if status is not None]
        counted = histogram(status for status in stated if status in TERMINAL_QUESTION_STATUSES)
        unrecognised = histogram(
            status for status in stated if status not in TERMINAL_QUESTION_STATUSES
        )
        excluded = {
            f"{UNRECOGNISED_STATUS_PREFIX}{status}": count for status, count in unrecognised.items()
        }
        missing = len(statuses) - len(stated)
        if missing:
            # Its own key rather than a bucket: a request whose status never reached
            # the store is not a request in an unknown state, it is a request nobody
            # could say anything about, and a fourth bar for it would invite a reader
            # to read it as a state Kojutsu writes.
            excluded[NO_STATUS_KEY] = missing
        return DistributionFigure(
            claim=self.claim(snapshot),
            snapshot=snapshot,
            title="Terminal state each projected decision request reached",
            values=freeze_counts(
                {status: counted.get(status, 0) for status in TERMINAL_QUESTION_STATUSES},
                "DecisionRequestOutcomeMeasure.values",
            ),
            excluded=excluded,
        )


def superseded_interval_claim(total: int) -> Claim:
    """The claim behind the superseded wait, over the requests that were overtaken.

    A function rather than a constant, for the reason
    :func:`~tenbin.measures.decision_requests.answered_capture_claim` is one: the
    denominator is how many projected requests reached ``superseded`` on *this* read, which
    is a fact about the read and not about the measure. Note what it counts -- every
    superseded request rather than only the measurable ones -- so the buckets and the
    exclusions add up to it, and a reader can see how many were overtaken as well as how
    long they took to be.
    """
    return Claim(
        slug=SUPERSEDED_INTERVAL_SLUG,
        source="Kojutsu AZ8T5XYS, the question-registry projection",
        statement=(
            "The decision requests in this corpus that reached the superseded state took the "
            "spread of times below to be overtaken, over the requests and not over the records "
            "they sit beside."
        ),
        does_not_mean=(
            "How long people take to answer, and not how fast the code moves. "
            + SUPERSEDED_MEANS
            + " "
            + SUPERSEDED_IS_AMBIGUOUS
            + " It is also not the wait of the requests that were answered or that ran out of "
            "attempts: those are in the lifecycle figure beside it, which is over every terminal "
            "state, and a superseded request is the one state that means neither an answer "
            "arriving nor a ceiling being reached. No mean is reported, for the reason every "
            "other duration in this program gives: the intervals are skewed, and a mean of a "
            "skewed distribution is a point no observation occupies."
        ),
        falsifier=(
            "A read whose superseded requests land an order of magnitude earlier while the churn "
            "behind them is unchanged, or a projection that starts writing a supersession moment "
            "of its own beside the terminal one. Either would mean the two figures were not "
            "reading the same field: the interval here is the same terminal clock the lifecycle "
            "figure reads, and a reader comparing the two distributions has to be comparing two "
            "populations rather than two different notions of when a request ended."
        ),
        denominator=Denominator(SUPERSEDED_POPULATION, max(1, total), size_noun="requests"),
        kind=ClaimKind.descriptive,
        granularity=Granularity.corpus,
        unblocked_by=UNBLOCKED_BY,
    )


@dataclass(frozen=True)
class SupersededIntervalMeasure:
    """How long a superseded request sat before the code moved underneath it.

    **Over the ``superseded`` state alone, and separately from the other two terminal
    states.** ``answered`` and ``failed`` are requests that reached an outcome; this is
    the one state that means the code moved instead, so a distribution mixing the three
    would describe a population whose fastest rungs are answers arriving and whose slowest
    are questions being overtaken, and a reader comparing two runs would be comparing two
    different questions. :class:`DecisionRequestLifecycleMeasure` remains the figure over
    every terminal state; this one is the interval behind the bucket the outcome figure
    says is the interesting one.

    **The terminal clock is the same field the lifecycle figure reads.** It is
    ``answered_at`` on the projected document, named for the state Kojutsu settled it
    in rather than for the one it settled into, and using it here rather than
    ``status_as_of`` is deliberate: ``status_as_of`` is when the *projection* noticed, so
    timing against it would measure the outbox again and this figure is already that
    figure's neighbour. A superseded request carrying no terminal clock is counted under
    :data:`NO_TERMINAL_TIME_KEY` and not dated approximately, so a corpus whose projection
    does not record when a question was overtaken reports its own emptiness rather than a
    guess.

    **A backwards interval gets the ladder's own bucket rather than an exclusion key.** The
    two are not reconciled here because they are not reconciled in
    :class:`DecisionRequestLifecycleMeasure` beside it, which hands the same
    :func:`~tenbin.measures.base.duration_buckets` output straight into ``values`` -- and a
    reader comparing two distributions is not asked to match a bucket against an
    exclusion. What the ladder guarantees either way is the part that matters: a clock
    that disagrees with itself can never be filed as the fastest wait there was.
    """

    #: Every projected request counts, for the reason its two siblings in this module do:
    #: the kind comes from Kojutsu's own tag projection, so requiring ``DETERMINED``
    #: would exclude nothing the status filter does not already exclude and would make the
    #: exclusion rules harder to read than the thing they exclude.
    requires_certainty = None

    @property
    def slug(self) -> str:
        """The identifier this measure is cited by."""
        return SUPERSEDED_INTERVAL_SLUG

    def claim(self, snapshot: CorpusSnapshot) -> Claim:
        """The claim, over the superseded requests this read enumerated."""
        return superseded_interval_claim(len(_superseded(snapshot.records)))

    def compute(self, snapshot: CorpusSnapshot) -> Figure:
        """The bucket distribution of the wait, with both absences named."""
        superseded = _superseded(snapshot.records)
        measured: list[float] = []
        excluded: dict[str, int] = {}
        for record in superseded:
            raised = record.created_at
            terminal = record.answered_at
            if raised is None:
                _bump(excluded, NO_RAISED_TIME_KEY)
                continue
            if terminal is None:
                _bump(excluded, NO_TERMINAL_TIME_KEY)
                continue
            measured.append((terminal - raised).total_seconds() / 3600.0)
        return DistributionFigure(
            claim=self.claim(snapshot),
            snapshot=snapshot,
            title="How long a superseded decision request sat before the code moved",
            values=duration_buckets(measured),
            excluded=excluded,
        )


def _superseded(records: tuple[Record, ...]) -> tuple[Record, ...]:
    """The projected requests the registry placed at ``superseded``, and nothing else.

    Filtered on the terminal status rather than by anything else the request carries, so
    the population is the bucket :class:`DecisionRequestOutcomeMeasure` already renders and
    the two figures are describing one state between them. The status is compared as a
    string because it is Kojutsu's vocabulary and not this reader's; a status this
    reader does not recognise is :class:`DecisionRequestOutcomeMeasure`'s to count, and a
    request that is not superseded is not an exclusion *here* -- it is outside a
    population defined by the state.
    """
    return tuple(
        record for record in _requests(records) if record.question_status == SUPERSEDED_STATUS
    )


def join_bucket_label(status: str, *, captured: bool) -> str:
    """The distribution's label for one terminal status, split by whether a capture names it.

    A sentence rather than a code, for the reason every exclusion key in this package is
    a sentence: this label is rendered verbatim into a report beside a reader's
    ``failed`` and ``superseded``, and a bucket called ``answered_captured`` invites the
    reader to treat "captured" as a fourth terminal state Kojutsu writes. It does not
    -- it is a fact about the corpus and not about the registry.

    Exported because the measure and the test that checks the rendering have to name the
    same string, and a test that rebuilt the label itself would be testing its own
    f-string.
    """
    suffix = WITH_CAPTURE_SUFFIX if captured else WITHOUT_CAPTURE_SUFFIX
    return f"{status} {suffix}"


@dataclass(frozen=True)
class QuestionCaptureJoin:
    """What the registry and the corpus each say about the same question, side by side.

    **The three disagreement fields are separate because one count would be none of the
    three causes.** ``answered_without_capture`` is a row lost between the registry and
    the store, and Kojutsu calls a queued row the only copy of a human's answer, so
    this is the expensive one. ``capture_without_terminal`` -- a capture naming a request
    the registry placed at ``failed`` or ``superseded`` -- is two write paths recording
    different outcomes for one question. ``unmatched_captures`` is a capture naming a
    question this read holds no request for, which is what an eventually-consistent
    projection and a truncated read both look like. Three remedies, so three fields, and
    ``answered_with_capture`` beside them is the one count that is a rate's numerator
    rather than a fault.

    ``values`` is the exhaustive bucketing of the same facts over the request
    population: six buckets, three statuses by two sides, every terminal request in
    exactly one of them, so the buckets add up to the denominator rather than to a
    subset of it. ``excluded`` holds the records that could not be joined at all, keyed
    by condition, and -- unlike every other exclusion in this package -- some of those
    counts are *not* addable to the denominator, because they are captures rather than
    requests. Each such key says which population it belongs to, and that clause is the
    honest rendering rather than a footnote to it.

    The hash is written out rather than generated because two fields are mappings: a
    generated hash would hash a ``MappingProxyType`` and raise, and the second of the
    two ways that happens is worse -- two joins computed the same way over the same
    records comparing unequal because a count mapping was built in a different order.
    The base class makes the same argument about every figure it carries.
    """

    values: Mapping[str, int]
    answered_with_capture: int
    answered_without_capture: tuple[str, ...]
    capture_without_terminal: tuple[str, ...]
    unmatched_captures: int
    excluded: Mapping[str, int]

    def __hash__(self) -> int:
        """Identity is every field, read off the mappings in order."""
        return hash(
            (
                tuple(self.values.items()),
                self.answered_with_capture,
                self.answered_without_capture,
                self.capture_without_terminal,
                self.unmatched_captures,
                tuple(self.excluded.items()),
            )
        )

    @property
    def terminal_questions(self) -> int:
        """How many projected requests reached a terminal state: the rate's denominator.

        A property over ``values`` rather than a field of its own so the two cannot
        disagree. A denominator that disagreed with its own numerator's population by
        one is the kind of defect this package exists to make unconstructible, and the
        cheapest way to keep it unconstructible is to refuse to store the number twice.
        """
        return sum(self.values.values())

    @property
    def disagrees(self) -> bool:
        """Whether either side of the join is missing something the other has.

        Three ways to disagree and none of them is a rate, so this is a boolean and the
        *kind* is carried by the fields. A finding that said "the two disagree" would
        leave a reader with nothing to go and check, and the whole point of the split is
        that the check differs per kind.
        """
        return bool(
            self.answered_without_capture
            or self.capture_without_terminal
            or self.unmatched_captures
        )


def join_questions_and_captures(records: Iterable[Record]) -> QuestionCaptureJoin:
    """Join the projected requests to the captures naming them, and split the misses.

    The two sides are read separately and joined on ``question_id`` alone. On the
    request side that is the registry's own identifier, projected as frontmatter and
    matching the document id; on the capture side it is the question a knowledge entry
    answers, written by Kojutsu and absent on any capture that answered nothing. The
    captures come from :func:`~tenbin.measures.filtering.capture_population` and the
    requests from :func:`_requests`, so a projected request can never be counted as a
    capture and a capture can never be counted as a request -- which is decision 001
    stated in terms of the code rather than of the documentation.

    **A capture naming an answered request is counted once, not once per capture.** Two
    knowledge entries can answer one question, and the figure is about questions, so the
    set is taken before the count. Counting captures would have made a question answered
    twice look like the reason the rate is high.

    A request whose status is not one of Kojutsu's three is neither a join nor a
    disagreement: it goes into ``excluded`` under the same key
    :class:`DecisionRequestOutcomeMeasure` names it under, because one vocabulary for one
    absence is worth more than a second key that says the same thing in other words.

    Pure: the same records in give the same join out. No clock, no store, no state.
    """
    materialized = tuple(records)
    requests = _requests(materialized)
    captures = capture_population(materialized)

    known_ids: set[str] = set()
    terminal_status: dict[str, str] = {}
    requests_without_id = 0
    missing_status = 0
    foreign_status: dict[str, int] = {}
    for record in requests:
        question_id = record.question_id
        if question_id is None:
            requests_without_id += 1
            continue
        known_ids.add(question_id)
        status = record.question_status
        if status in TERMINAL_QUESTION_STATUSES:
            # First writer wins on a duplicate id, for the reason ``_census`` takes the
            # first outcome per change: the store has said two things about one question
            # and this reader does not adjudicate between them.
            terminal_status.setdefault(question_id, str(status))
        elif status is None:
            missing_status += 1
        else:
            foreign_status[status] = foreign_status.get(status, 0) + 1

    named_by_a_capture: set[str] = set()
    naming_an_unterminal_request: set[str] = set()
    captures_without_id = 0
    unmatched_captures = 0
    for record in captures:
        question_id = record.question_id
        if question_id is None:
            captures_without_id += 1
        elif question_id not in known_ids:
            unmatched_captures += 1
        else:
            named_by_a_capture.add(question_id)
            if question_id not in terminal_status:
                naming_an_unterminal_request.add(question_id)

    counts = {
        join_bucket_label(status, captured=captured): 0
        for status in TERMINAL_QUESTION_STATUSES
        for captured in (True, False)
    }
    answered_without_capture: list[str] = []
    capture_without_terminal: list[str] = []
    answered_with_capture = 0
    for question_id, status in terminal_status.items():
        captured = question_id in named_by_a_capture
        counts[join_bucket_label(status, captured=captured)] += 1
        if status != ANSWERED_STATUS:
            if captured:
                capture_without_terminal.append(question_id)
            continue
        if captured:
            answered_with_capture += 1
        else:
            answered_without_capture.append(question_id)

    excluded: dict[str, int] = {}
    if requests_without_id:
        excluded[NO_REQUEST_QUESTION_ID_KEY] = requests_without_id
    if missing_status:
        excluded[NO_STATUS_KEY] = missing_status
    for status, count in sorted(foreign_status.items()):
        excluded[f"{UNRECOGNISED_STATUS_PREFIX}{status}"] = count
    if captures_without_id:
        excluded[NO_CAPTURE_QUESTION_ID_KEY] = captures_without_id
    if naming_an_unterminal_request:
        excluded[NO_TERMINAL_REQUEST_KEY] = len(naming_an_unterminal_request)

    return QuestionCaptureJoin(
        values=freeze_counts(counts, "QuestionCaptureJoin.values"),
        answered_with_capture=answered_with_capture,
        answered_without_capture=tuple(sorted(answered_without_capture)),
        capture_without_terminal=tuple(sorted(capture_without_terminal)),
        unmatched_captures=unmatched_captures,
        excluded=freeze_counts(excluded, "QuestionCaptureJoin.excluded"),
    )


def answered_capture_claim(total: int) -> Claim:
    """The claim behind the answered-and-captured rate, over the terminal requests.

    A function rather than a constant because the denominator is how many projected
    requests reached a terminal state on *this* read, which is a fact about the read and
    not about the measure. A claim carrying a placeholder population would be a claim
    about a corpus nobody described, and the numerator here is a count of requests, so
    the denominator has to be a count of requests too.
    """
    return Claim(
        slug=ANSWERED_CAPTURE_SLUG,
        source="Kojutsu AZ8T5XYS, the question-registry projection",
        statement=(
            "Of the decision requests in this corpus that reached a terminal state, the share "
            "whose answer was kept as a capture naming the same question, over the requests "
            "and never over the records they sit beside."
        ),
        does_not_mean=(
            "The share of questions that were answered. That is a different number over the "
            "same population and the two differ exactly by the disagreements "
            "`tenbin.measures.projection_safety.QuestionCaptureAgreementMeasure` reports: a "
            "request the registry places at answered and the corpus holds no document for is "
            "an answer recorded and not stored, and rendering it as a percentage is the one "
            "transformation this program exists to refuse. It is also not a measure of "
            "whether anybody tried, of delivery latency, or of anything about the answers "
            "themselves -- only of whether a document survived. " + SUPERSEDED_MEANS
        ),
        falsifier=(
            "A read in which the share rises while the registry's own answered count does not, "
            "which would mean the figure was measuring a change in what Kojutsu writes on "
            "captures rather than a change in what people did. The other falsifier is a "
            "delivery that is still catching up: decision 001 calls the projection eventually "
            "consistent, so a read taken while an outbox is draining will show answers stored "
            "and questions not yet re-projected, and the figure should be read again once it "
            "has settled."
        ),
        denominator=Denominator(ANSWERED_CAPTURE_POPULATION, max(1, total)),
        kind=ClaimKind.descriptive,
        granularity=Granularity.corpus,
        unblocked_by=UNBLOCKED_BY,
    )


@dataclass(frozen=True)
class QuestionAnsweredRateMeasure:
    """Of the requests that resolved, the share whose answer was kept -- as two figures.

    A distribution and a rate under one claim, following
    :class:`~tenbin.measures.outcome.ChangeOutcomeMeasure`: the buckets say which
    terminal state each request reached and whether a capture names it, and the rate is
    the one bar of that distribution a reader usually wants as a share. The distribution
    is what makes the rate checkable -- it is where the reader sees that the denominator
    really was requests at terminal state, and it carries the ``excluded`` counts, which
    a :class:`~tenbin.measures.base.RateFigure` has nowhere to put.

    **Those exclusions are not all addable to the denominator, and the keys say so.** A
    capture with no question id is a capture, so reconciling it against a request count
    would be adding two populations; a request with no question id is a request, so
    that one does add up. Both are named with their population in the key rather than in
    a footnote, because the alternative is a reader who adds the two columns together and
    gets a number this program never published.

    The denominator is requests, never records, and the denominator is computed from the
    same join as the numerator rather than by a second walk. A rate whose numerator and
    denominator came from two passes over the corpus could disagree by one on a store
    that changed between them, and it would render.
    """

    requires_certainty = None

    @property
    def slug(self) -> str:
        """The identifier this measure is cited by."""
        return ANSWERED_CAPTURE_SLUG

    def claim(self, snapshot: CorpusSnapshot) -> Claim:
        """The claim, over the requests this read held at a terminal state."""
        join = join_questions_and_captures(snapshot.records)
        return answered_capture_claim(join.terminal_questions)

    def compute(self, snapshot: CorpusSnapshot) -> Figure:
        """The join bucketed by status and correspondence, and the answered share of it."""
        join = join_questions_and_captures(snapshot.records)
        claim = answered_capture_claim(join.terminal_questions)
        return FigureGroup(
            claim=claim,
            snapshot=snapshot,
            title="Terminal decision requests, and the answers the corpus kept",
            figures=(
                DistributionFigure(
                    claim=claim,
                    snapshot=snapshot,
                    title=(
                        "Every terminal decision request, by registry status and whether a "
                        "capture names it"
                    ),
                    values=join.values,
                    excluded=join.excluded,
                ),
                RateFigure(
                    claim=claim,
                    snapshot=snapshot,
                    title="Terminal decision requests whose answer was kept as a capture",
                    rate=Rate(
                        numerator=join.answered_with_capture,
                        denominator=Denominator(
                            ANSWERED_CAPTURE_POPULATION, max(1, join.terminal_questions)
                        ),
                    ),
                ),
            ),
        )


def _bump(counts: dict[str, int], key: str) -> None:
    """Count one more request under a condition, for the ``excluded`` mapping.

    A function rather than an open-coded ``counts[key] = counts.get(key, 0) + 1`` at each
    branch of the interval loop, because that is an expression to be transcribed slightly
    differently in the second one, and a mistyped exclusion key is a figure that reports a
    condition nobody has.
    """
    counts[key] = counts.get(key, 0) + 1
