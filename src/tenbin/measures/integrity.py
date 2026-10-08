"""One shape in the corpus that should not occur, named as a finding and not a cause.

Pull-request transitions are the write path reporting on itself: the change
opened, the change closed, the change reopened. A store that holds ten or more
``opened`` transitions and not one ``closed`` is not a plausible history of a busy
repository. It is a shape, and the shape is real regardless of why it is there --
so this detector fires on the shape and says so.

**The cause is a hypothesis with alternatives, because the obvious one has been
fixed.** Kojutsu's ``semantic_pr_event_id`` built its identity preimage with
``json.dumps`` and folded in ``datetime`` objects, which it cannot encode: every
close and reopen carrying a timestamp raised ``TypeError`` and was never written.
A close is the event that says a change was adopted or abandoned, so for as long as
that defect stood the store could not record an outcome at all, and "opens with no
closes" was the signature of it. **That defect is fixed.** Markers are now
normalised through ``_transition_marker`` to ISO-8601 before they reach the JSON,
and Kojutsu's own docstring names the bug it replaced.

So a detector that hardcoded the old cause would now emit a confident wrong
answer: it would find the shape, attach the one cause everybody remembers, and
present a historical bug as a live diagnosis. The correction is to fire on the
shape, name the historical cause *as one candidate among several*, say that it has
been fixed upstream, and list the causes that would still be live. A store that
genuinely merged nothing in the period; a capture path broken for a different
reason; a collection filtered to opened events only. Each of those produces the
same shape and none of them is the one from the commit log.

**A detector that fires on noise gets switched off.** That is why
:data:`MIN_OPENS_FOR_INFERENCE` is ten and why the closed count must be exactly
zero rather than merely small: a small repository with two opens and no closes is
an ordinary quiet week, and a detector that reported it would be training its
readers to ignore it. The threshold is a named constant so the next person can
argue with the number rather than with the code.

**The second detector in this module watches the other dimension of the same
observation: a path that is reachable and producing nothing.** The event-shape
detector asks whether a *busy* corpus recorded the outcomes it should have; this
one asks whether a corpus full of documents has recorded anything *recently*. A
store that is up, holds thousands of documents, and has captured nothing for a
quarter is a capture path that stopped, and completeness does not catch it --
completeness is a statement about the read, and this read was complete. **The
bound is :data:`MAX_CORPUS_SILENCE` and not "yesterday"** for the reason the
threshold above is ten: a repository nobody has touched for a month is ordinary,
and a detector that fires on that trains its readers to ignore it within a quarter.
The bound is a named constant so the next person argues with the number.

**An unknown window is silence, and this check stays quiet about it.** The window
is the earliest and latest moment this read's records could be placed at, and a
corpus whose every timestamp would not parse has no window. "The newest record is
old" is a statement about a record that can be placed in time, and inferring it
from a corpus that cannot be placed in time would be a finding with no observation
behind it. So an unknown window does not fire this check, and the clean claim
says the check could not run rather than that it found nothing.

**What this module notably does not do:** it does not diagnose. Tenbin reads a
store; it does not execute Kojutsu's code, does not see the forge's payloads,
and cannot tell a capture path that raised from one that was never called. So
``cannot_confirm`` is required and cannot be blank -- a finding with nothing in
that field is a finding whose author either confirmed something or decided to
imply it had, and those are the same sentence to a reader.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from datetime import datetime, timedelta
from typing import Final

from tenbin.claims.mechanism import Granularity
from tenbin.claims.model import Claim, ClaimKind, Denominator
from tenbin.corpus.record import LifecycleAction, Record, RecordKind
from tenbin.corpus.snapshot import CorpusSnapshot
from tenbin.corpus.window import CorpusWindow, record_moment, render_duration
from tenbin.measures.base import CountFigure, Figure, FigureGroup, Finding
from tenbin.measures.filtering import capture_population, named_repository, select

#: How many ``opened`` transitions before the shape is worth a finding. Ten is a
#: judgement about noise, not about statistics: below it, "a busy week with no
#: merges" is an ordinary event in a small repository, and a detector that fires
#: there gets ignored by the third false alarm. Above it, a repository that has
#: opened ten changes and recorded no close is telling us something.
MIN_OPENS_FOR_INFERENCE: Final[int] = 10

#: The finding's slug. A finding is a claim about the corpus like any other, so it
#: is cited like one, and a report that hits it prints a caveat rather than a bare
#: sentence.
LIFECYCLE_SHAPE_SLUG = "corpus.lifecycle_opens_without_closes"

#: The population the finding is about: the lifecycle records this read enumerated.
#: Not the whole corpus -- a finding about one shape of one kind of record does not
#: get to borrow the population of every other kind.
LIFECYCLE_POPULATION = "pull-request lifecycle records in this read"

#: The historical cause, named in full, with the fix named beside it. Kept as one
#: constant because a paraphrase of it is a second account of a bug in another
#: repository, and this account is the one a reader will check.
HISTORICAL_CAUSE: Final[str] = (
    "Kojutsu's semantic_pr_event_id built its identity preimage with json.dumps and folded "
    "in the payload's datetime values, which json cannot encode, so it raised TypeError on every "
    "close and reopen that carried a timestamp and the record was never written. That defect is "
    "fixed: markers are now normalised to ISO-8601 through _transition_marker before they reach "
    "the JSON, and Kojutsu's own docstring names the bug, so the shape observed now cannot be "
    "explained by it. Treat it as the reason this corpus was shaped the way it was before the fix, "
    "not as a live cause."
)

#: The causes that are still live, listed so that a reader has somewhere to look
#: that is not the commit log. Each produces exactly this shape.
ALTERNATIVE_CAUSES: Final[str] = (
    "Other causes produce the same shape and are not excluded by anything here: the store "
    "genuinely merged and abandoned nothing in this period; a capture path is broken for a reason "
    "other than the one above, so the close never reached the writer rather than the writer "
    "refusing it; or the collection this read walked has been filtered to opened transitions."
)

#: What Tenbin is and is not in a position to say. Required, non-blank, and the
#: reason this is a finding rather than a diagnosis.
CANNOT_CONFIRM: Final[str] = (
    "Tenbin read a store and did not execute Kojutsu's code, see the forge's payloads, or "
    "observe any capture path. This is an observation with candidate causes, not a diagnosis: "
    "nothing in the store distinguishes a writer that raised from a writer that was never called, "
    "and the absence of closes is consistent with all of the causes above."
)


def event_shape_claim(total: int) -> Claim:
    """The claim a lifecycle-shape finding is attached to, over the lifecycle records seen.

    Built by a function rather than held as a constant because the denominator is
    the number of lifecycle records this read enumerated, which is a fact about the
    read. A claim carrying a placeholder population would be a claim about a corpus
    nobody described.
    """
    return Claim(
        slug=LIFECYCLE_SHAPE_SLUG,
        statement=(
            "This read enumerated pull-request transitions that open changes and none that close "
            "them, which is not a shape a working repository produces."
        ),
        does_not_mean=(
            "That no change was merged. The absence of a close record is a statement about what "
            "this corpus holds and not about what happened in the repository: a change that was "
            "merged is recorded by a close transition, and this read holds none."
        ),
        falsifier=(
            "A single closed transition appearing in a later read of the same collection. One "
            "close makes this shape a quiet period rather than a broken path, and the finding "
            "should stop firing on the threshold alone."
        ),
        denominator=Denominator(LIFECYCLE_POPULATION, max(1, total)),
        kind=ClaimKind.descriptive,
        granularity=Granularity.corpus,
    )


class LifecycleShapeDetector:
    """Detect opens with no closes, and say what that might mean without settling it.

    Stateless and total: the same records in give the same finding or the same
    ``None`` out, with no clock and no state. It takes records rather than a
    snapshot because a shape is a shape of records -- and because the finding it
    produces carries the snapshot for the completeness sentence, while the
    detector itself has no opinion about whether the read was whole.
    """

    __slots__ = ()

    #: Lifecycle transitions are read from the tag projection, which Kojutsu
    #: derives from the kind it holds, so a record read either way names its
    #: action. Certainty is not required: a lifecycle record whose kind came from
    #: the ``pr_state_change`` tag is exactly as reliable as one that carried the
    #: key, and requiring ``DETERMINED`` would discard every legacy transition
    #: while catching nothing -- the shape does not depend on how the kind was
    #: recovered.
    requires_certainty = None

    def detect(
        self,
        records: Iterable[Record],
        *,
        snapshot: CorpusSnapshot,
    ) -> Finding | None:
        """The finding if this corpus has the shape, or ``None`` if it does not.

        ``snapshot`` is required and keyword-only even though the shape does not
        need it, because the *finding* does. A :class:`Finding` carries the read it
        was observed over -- that is what makes
        :meth:`~tenbin.measures.base.Figure.rate_text` available on it -- and a
        detector that could be called with records alone would return a finding
        with no completeness statement attached, which is the exact thing
        :class:`~tenbin.measures.base.Figure` is shaped to prevent.

        The same records in give the same answer out: no clock, no store, no state.
        """
        materialized = tuple(records)
        actions = Counter(
            record.lifecycle_action
            for record in select(materialized, kinds=(RecordKind.PR_LIFECYCLE,))
            if record.lifecycle_action is not None
        )
        opened = actions.get(LifecycleAction.OPENED, 0)
        closed = actions.get(LifecycleAction.CLOSED, 0)
        if opened < MIN_OPENS_FOR_INFERENCE or closed != 0:
            return None
        return self._finding(materialized, opened=opened, snapshot=snapshot)

    def _finding(
        self,
        records: tuple[Record, ...],
        *,
        opened: int,
        snapshot: CorpusSnapshot,
    ) -> Finding:
        """Assemble the finding, including where the opens were concentrated.

        ``where`` names the repository holding the most of the opens rather than
        listing them, because a finding with no location is a finding about a
        collection in the abstract and a reader's next question is always "which
        repository". It is optional -- a corpus spread evenly across repositories
        has no such place, and ``None`` says that better than naming the largest of
        thirty near-identical buckets as though it were the story.
        """
        lifecycle = select(records, kinds=(RecordKind.PR_LIFECYCLE,))
        # The placeholder check matters here for the same reason it does in the
        # coverage measures: a record with no repository has the string "unknown"
        # for one, and "all 10 opens are in `unknown`" is a location that does not
        # exist.
        opened_repos = Counter(
            record.repo
            for record in lifecycle
            if record.lifecycle_action is LifecycleAction.OPENED
            and named_repository(record.repo)
            and record.repo is not None
        )
        where = None
        if opened_repos:
            top_repo, top_count = opened_repos.most_common(1)[0]
            where = (
                f"{top_count} of {opened} opens are in `{top_repo}`"
                if top_count < opened
                else f"all {opened} opens are in `{top_repo}`"
            )
        observed = (
            f"{opened} pull-request transitions opened a change and none closed one, across this "
            f"read's {len(lifecycle)} lifecycle records."
        )
        return Finding(
            claim=event_shape_claim(len(lifecycle)),
            snapshot=snapshot,
            title="Pull-request transitions open changes and record no closes",
            what_was_observed=observed,
            suspected_cause=f"{HISTORICAL_CAUSE} {ALTERNATIVE_CAUSES}",
            where=where,
            cannot_confirm=CANNOT_CONFIRM,
        )


#: The slug a clean lifecycle-shape check reports under. It is a slug of its own
#: rather than a reused heading, because "the check ran and found nothing" and "the
#: check did not run" must not render as the same absence.
LIFECYCLE_SHAPE_CLEAN_SLUG: Final[str] = "corpus.lifecycle_shape_checked"


class LifecycleShapeMeasure:
    """Run :class:`LifecycleShapeDetector` as a measure, so a report cannot skip it.

    **This class exists because the detector was not being called.** A detector that
    nothing invokes is a function that looks finished, which is the worse of the two
    states a finished module can be in -- the same argument ``kojutsu.compare``
    makes for putting a reachable module behind a command. The check that a report's
    own integrity group exists to perform was absent from every report, so a corpus
    with a broken close path rendered as a clean one.

    A clean result is a :class:`~tenbin.measures.base.CountFigure` of zero and not
    an empty section, because a reader who cannot distinguish "checked, and the shape
    is ordinary" from "nothing here" has been handed an absence where a fact belongs.
    The count is zero, the claim says the check ran, and the falsifier names the shape
    that would have produced a finding instead.
    """

    __slots__ = ()

    slug = LIFECYCLE_SHAPE_SLUG
    requires_certainty = None
    #: A reader asking about this measure is asking whether the numbers below it can
    #: be believed, so it belongs to the capture-system view of the report.
    about_the_capture_system = True
    detector: LifecycleShapeDetector = LifecycleShapeDetector()

    def claim(self, snapshot: CorpusSnapshot) -> Claim:
        """The claim this measure answers, over the lifecycle records the read held.

        The denominator is the number of lifecycle records enumerated, so a corpus
        with no transitions at all reports over zero and the claim says the check had
        nothing to look at -- a different sentence from "the check found nothing".
        """
        finding = self.detector.detect(snapshot.records, snapshot=snapshot)
        if finding is not None:
            return finding.claim
        return _clean_event_shape_claim(self._lifecycle_count(snapshot))

    def compute(self, snapshot: CorpusSnapshot) -> Figure:
        """The finding, or a count of zero with a claim that says the check ran."""
        finding = self.detector.detect(snapshot.records, snapshot=snapshot)
        if finding is not None:
            return FigureGroup(
                claim=finding.claim,
                snapshot=snapshot,
                title=finding.title,
                figures=(finding,),
            )
        total = self._lifecycle_count(snapshot)
        return CountFigure(
            claim=_clean_event_shape_claim(total),
            snapshot=snapshot,
            title="Pull-request transition shape",
            value=0,
        )

    @staticmethod
    def _lifecycle_count(snapshot: CorpusSnapshot) -> int:
        return sum(
            1
            for record in snapshot.records
            if record.classification.kind is RecordKind.PR_LIFECYCLE
        )


def _clean_event_shape_claim(total: int) -> Claim:
    """The claim attached to a lifecycle-shape check that found nothing.

    A distinct claim from the finding's, because the two say opposite things and a
    report that reused one slug for both would let a reader match them up and conclude
    the clean one refutes the finding. It is still a claim: a check that ran and found
    an ordinary shape is a statement about the corpus, and it is the statement whose
    falsifier is the shape the detector looks for.
    """
    return Claim(
        slug=LIFECYCLE_SHAPE_CLEAN_SLUG,
        statement=(
            "This read's pull-request transitions have the shape a working repository produces: "
            "changes that open are matched by changes that close."
        ),
        does_not_mean=(
            "That the transition path is healthy in general. This is the absence of one specific "
            "shape -- opens with no closes -- in one collection on one read, and a path that broke "
            "after this read, or in a collection nobody read, would pass it."
        ),
        falsifier=(
            "A read of the same collection in which transitions open changes and none close them. "
            "That shape is what the detector looks for, and seeing it would make this statement "
            "false for the period it appeared in."
        ),
        denominator=Denominator(LIFECYCLE_POPULATION, max(1, total)),
        kind=ClaimKind.descriptive,
        granularity=Granularity.corpus,
    )


# -- the other dimension of the same observation: a path that is up and silent -------


#: The finding's slug. A finding is a claim about the corpus like any other, so it is
#: cited like one, and a report that hits it prints a caveat rather than a bare
#: sentence.
CAPTURE_FRESHNESS_SLUG = "corpus.capture_has_gone_quiet"

#: The slug a clean capture-freshness check reports under. A slug of its own rather
#: than a reused heading, for the reason the lifecycle-shape check has one: "the check
#: ran and found nothing" and "the check could not run" are different facts and must
#: not render as the same absence.
CAPTURE_FRESHNESS_CLEAN_SLUG: Final[str] = "corpus.capture_freshness_checked"

#: How long the corpus may go without capturing anything before the silence is a
#: finding. Ninety days rather than a week, and the reasoning is noise in both
#: directions.
#:
#: Below it, a finding fires on ordinary life: a repository nobody has opened since
#: the summer, a team between projects, a holiday. A detector that reports those gets
#: ignored by the third one, and a detector that gets ignored is worse than no
#: detector, because it is still in the report.
#:
#: Above it, a corpus fed by webhook events has stopped for a whole quarter. The
#: threshold is a named constant and not a value inlined at the comparison, so the
#: next person argues with the number rather than with the code -- the same discipline
#: :data:`MIN_OPENS_FOR_INFERENCE` above is under.
MAX_CORPUS_SILENCE: Final[timedelta] = timedelta(days=90)

#: The population the check is about: the records in this read that carried a moment
#: this reader could place. Not the whole corpus, because a record with no readable
#: timestamp cannot have an age, and a denominator that counted it would be describing
#: a population the check did not look at.
FRESHNESS_POPULATION: Final[str] = (
    "records in this read that carried a moment this reader could place"
)

#: What a quiet capture path is, and what it is not. Two sentences because they are two
#: misreadings, and the second is the one the finding is most likely to produce on its
#: own: a reader who sees "nothing captured for three months" will very reasonably
#: conclude the corpus is fine because the read was complete, and completeness is a
#: statement about the read.
QUIET_IS_NOT_AN_OUTAGE: Final[str] = (
    "That the capture path has failed. A collection can be quiet because the work "
    "stopped, because the repositories in it were archived, or because somebody "
    "installed the webhook three months ago and there has been little to capture "
    "since -- and nothing in this store distinguishes those."
)
COMPLETE_IS_NOT_CAPTURING: Final[str] = (
    "That the corpus is damaged or incomplete. A read that found every document the store "
    "holds is a complete read whatever the documents are dated, and this finding is about "
    "the dates and not about the count: completeness is a statement about the read, and a "
    "read of a store that has stopped producing records is the most complete read "
    "available."
)

#: The causes, named with their alternatives rather than settled on, for the reason
#: :data:`HISTORICAL_CAUSE` above is: a detector that reads a store and did not run the
#: code that writes it has candidates, not a diagnosis.
SILENCE_CAUSES: Final[str] = (
    "The likeliest explanation is a capture path that stopped: a webhook whose secret "
    "expired, a subscription deleted, a writer deployed against a different collection, or a "
    "token that no longer authenticates. Other causes produce exactly the same shape and are "
    "not excluded by anything here: the work in this collection genuinely stopped; the "
    "repositories in it were archived and nothing has been written since; the corpus was "
    "seeded from an import rather than from webhook events, so it was never going to keep "
    "up; or this collection is a snapshot somebody copied rather than a live view of one."
)

#: What Tenbin is and is not in a position to say about the silence. Required and
#: non-blank, and the reason this is a finding rather than a diagnosis.
CANNOT_CONFIRM_SILENCE: Final[str] = (
    "Tenbin read a store and did not execute Kojutsu's code, observe any webhook delivery, "
    "or inspect the forge's subscription list, so this is an observation with candidate causes "
    "and not a diagnosis. The store cannot distinguish a writer that raised on every event from "
    "a writer that was never called, and it cannot tell a collection nobody writes to from a "
    "collection whose writes are going somewhere else."
)


def capture_silence_claim(placed: int, *, silence: timedelta) -> Claim:
    """The claim a capture-freshness finding is attached to, over the records it dated.

    Built by a function rather than held as a constant because two of its inputs are
    facts about the read: how many records carried a moment, and how long the silence
    was. A claim carrying a placeholder for either would be a claim about a corpus
    nobody described, and a finding whose claim does not name the silence it is about is
    a finding a reader has to reconstruct from its title.
    """
    return Claim(
        slug=CAPTURE_FRESHNESS_SLUG,
        statement=(
            f"The newest capture in this read is {render_duration(silence)} older than the read "
            "itself, so a collection that is reachable and holds documents has captured nothing "
            "for longer than the bound this check applies."
        ),
        does_not_mean=QUIET_IS_NOT_AN_OUTAGE + " " + COMPLETE_IS_NOT_CAPTURING,
        falsifier=(
            "A later read of the same collection in which some record carries a moment less "
            f"than {render_duration(MAX_CORPUS_SILENCE)} before the read. One captured record "
            "inside the bound makes this a quiet period rather than a stopped path, and the "
            "finding should stop firing on the threshold alone."
        ),
        denominator=Denominator(FRESHNESS_POPULATION, max(1, placed)),
        kind=ClaimKind.descriptive,
        granularity=Granularity.corpus,
    )


def detect_capture_silence(snapshot: CorpusSnapshot) -> Finding | None:
    """The finding if this corpus has gone quiet, or ``None`` if it has not.

    **Two conditions, and the second is the interesting one.** The window must be
    known -- that is, some record in this read must have carried a moment this reader
    could place -- and the newest of those must be further back than
    :data:`MAX_CORPUS_SILENCE` from the read. An unknown window does not fire, because
    "the newest record is old" is a statement about a record that can be placed in
    time, and a corpus whose every timestamp would not parse supports no such
    statement. Silence about an unknown window is not a stopped path; it is a reader
    that could not see.

    The comparison is against ``read_at`` and not against a clock, so the same snapshot
    gives the same answer today and next year, and a figure computed from a stored read
    model is the age of the read that produced it rather than the age of the machine
    rendering it.

    **The window is the captures' own, not the read's.** The subject of this finding is
    the capture path, so the newest thing that can vouch for the capture path is the
    newest *capture*. A projected decision request is projected from the registry, not
    captured from the forge, and decision 001 counts the two populations apart; letting
    a question document's timestamp satisfy this check would let a store whose captures
    stopped years ago report itself healthy because the registry was read recently.
    The corpus header keeps the whole read's window, because a header describes a read;
    this check describes a path, and those are different subjects.

    **A record dated after the read does not fire this either.** Its age is negative,
    which is clock skew between two writers -- the same fact
    :data:`~tenbin.measures.base.NEGATIVE_INTERVAL` has a bucket for -- and reporting a
    large positive age for it would turn a data defect into a capture path that stopped.
    """
    captures = capture_population(snapshot.records)
    window = CorpusWindow.from_records(captures)
    if not window.is_known:
        return None
    age = window.age_at(snapshot.read_at)
    if age is None or age <= MAX_CORPUS_SILENCE:
        return None
    return _silence_finding(snapshot, window=window, silence=age)


def _silence_finding(
    snapshot: CorpusSnapshot, *, window: CorpusWindow, silence: timedelta
) -> Finding:
    """Assemble the finding, naming the silence, the newest record and where it sits.

    ``where`` names the repository holding the newest record rather than the
    distribution of them, because a finding about a stopped capture path has an obvious
    next question and it is "which repository is still live". It is optional, and
    ``None`` says so better than a placeholder: a record whose id filed it under
    ``unknown`` has no repository, and naming that as a location would be a location
    that does not exist -- the defect :func:`~tenbin.measures.filtering.named_repository`
    exists to stop.
    """
    assert window.latest is not None
    captures = capture_population(snapshot.records)
    placed = len(captures) - window.unreadable_timestamps
    newest = _newest_record(captures)
    where = None
    if newest is not None and named_repository(newest.repo):
        where = (
            f"the newest capture in this read is in `{newest.repo}`, last captured "
            f"{render_duration(silence)} before it"
        )
    observed = (
        f"The newest capture in this read carries {window.latest.isoformat()} and the read "
        f"itself began at {snapshot.read_at.isoformat()}, leaving {render_duration(silence)} of "
        f"silence across {placed:,} capture(s) this reader could place in time and "
        f"{window.unreadable_timestamps:,} that carried no moment at all."
    )
    return Finding(
        claim=capture_silence_claim(placed, silence=silence),
        snapshot=snapshot,
        title="Nothing has been captured in this collection for over a quarter",
        what_was_observed=observed,
        suspected_cause=SILENCE_CAUSES,
        where=where,
        cannot_confirm=CANNOT_CONFIRM_SILENCE,
    )


def _newest_record(records: Iterable[Record]) -> Record | None:
    """The record in this read carrying the latest moment, or ``None`` if none does.

    A single walk rather than a ``max`` over a generator of pairs, because the
    generator version evaluates ``record_moment`` twice per record and the difference
    is a read of the same field twice, which is exactly the kind of duplication that
    survives a refactor and starts disagreeing with itself.
    """
    newest: Record | None = None
    latest: datetime | None = None
    for record in records:
        moment = record_moment(record)
        if moment is None:
            continue
        if latest is None or moment > latest:
            newest, latest = record, moment
    return newest


def _clean_capture_freshness_claim(placed: int) -> Claim:
    """The claim attached to a capture-freshness check that found nothing.

    **Two states are folded into it, and the statement is a disjunction because it has
    to be.** A corpus with a known window and a recent record has been checked and is
    not quiet. A corpus whose window is *unknown* has not been checked at all -- this
    module cannot tell the two apart from the population alone -- so the statement
    claims the weaker of the two things that are true in each case rather than the
    stronger, which would be false for every corpus that is simply healthy. Written as
    a conjunction it would be the most confident sentence in the module: a claim that
    is false for most of the corpora it is rendered on, and false in the direction of
    reassuring. Its falsifier and its ``does_not_mean`` are about the check rather than
    about the capture path, which is what keeps a clean result from reading as a health
    certificate. A second slug for the unchecked case would be a fourth slug in this
    module, and the reader would learn the difference only from the word "unknown" in a
    sentence they had to notice.
    """
    return Claim(
        slug=CAPTURE_FRESHNESS_CLEAN_SLUG,
        statement=(
            "This read found no silence: either no record in it carries a moment further back "
            f"than the {render_duration(MAX_CORPUS_SILENCE)} this check applies, or no record in "
            "it carries a moment this reader could place at all."
        ),
        does_not_mean=(
            "That the capture path is healthy, or that anything was captured recently. This is "
            "the absence of one specific condition -- a corpus whose newest record is older than "
            "the bound -- on one read, and a path that broke after this read, in a collection "
            "nobody read, or on a record whose timestamp would not parse, all pass it."
        ),
        falsifier=(
            "A read of the same collection in which the newest record is more than "
            f"{render_duration(MAX_CORPUS_SILENCE)} older than the read. That is the condition "
            "the check looks for, and seeing it would make this statement false for the period "
            "it was true in."
        ),
        denominator=Denominator(FRESHNESS_POPULATION, max(1, placed)),
        kind=ClaimKind.descriptive,
        granularity=Granularity.corpus,
    )


class CaptureFreshnessMeasure:
    """Run :func:`detect_capture_silence` as a measure, so a report cannot skip it.

    **This class exists for the same reason
    :class:`~tenbin.measures.projection_safety.ProjectionSafetyMeasure` does.** A
    check that nothing invokes is a function that looks finished, which is the worse of
    the two states a finished module can be in. This one is the more alarming of the
    pair when it is absent, because the condition it looks for is the one completeness
    structurally cannot catch: a store that is reachable, holds thousands of documents,
    and has captured nothing for a quarter is a *complete* read of a collection that has
    stopped growing, and every figure below it would be computed over a corpus whose own
    header is the evidence and whose figures say nothing.

    A clean result is a :class:`~tenbin.measures.base.CountFigure` of zero and not an
    empty section, because a reader who cannot distinguish "checked, and the capture
    path is not quiet" from "nothing here" has been handed an absence where a fact
    belongs. The count is zero, the claim says the check ran, and the falsifier names
    the silence that would have produced a finding instead.
    """

    __slots__ = ()

    slug = CAPTURE_FRESHNESS_SLUG
    #: Same reason as ``LifecycleShapeMeasure``: a corpus that has stopped capturing is
    #: not a corpus whose remaining figures mean what they appear to mean.
    about_the_capture_system = True
    #: Every record counts, whatever certainty its classification carries. The check
    #: reads timestamps rather than kinds, so requiring ``DETERMINED`` here would
    #: discard exactly the legacy documents whose silence is most worth noticing.
    requires_certainty = None

    def claim(self, snapshot: CorpusSnapshot) -> Claim:
        """The claim this measure answers, over the records this read could date."""
        finding = detect_capture_silence(snapshot)
        if finding is not None:
            return finding.claim
        return _clean_capture_freshness_claim(self._placed_count(snapshot))

    def compute(self, snapshot: CorpusSnapshot) -> Figure:
        """The finding, or a count of zero with a claim that says the check ran."""
        finding = detect_capture_silence(snapshot)
        if finding is not None:
            return FigureGroup(
                claim=finding.claim,
                snapshot=snapshot,
                title=finding.title,
                figures=(finding,),
            )
        return CountFigure(
            claim=_clean_capture_freshness_claim(self._placed_count(snapshot)),
            snapshot=snapshot,
            title="How long this collection has gone without capturing anything",
            value=0,
        )

    @staticmethod
    def _placed_count(snapshot: CorpusSnapshot) -> int:
        """The records whose age this check could have computed, from the window's own count.

        Derived rather than recounted, and for a reason that is about arithmetic rather
        than speed: the check's population is the records that named a moment, so the
        number has to be the same number the window arrived at. A second walk counting
        ``record_moment(record) is not None`` would agree today.
        """
        return len(snapshot.records) - snapshot.window.unreadable_timestamps
