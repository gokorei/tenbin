"""A measure against its own past value -- permitted, and not the comparison decision 002 refuses.

**The distinction this module exists to keep is between two populations and one population
twice.** ``docs/decisions/002-no-causal-claims.md`` refuses a comparison *between principals*:
agent against human, model against model. It refuses because the corpus holds no assignment
mechanism, and because the selection of well-specified tickets onto the agent path is the
entire mechanism by which the two groups differ -- so a difference in outcome between them is
uninterpretable rather than merely uncertain. **A measure against its own past value is a
different object.** It is one population, read twice. There is no treatment, no arm, no
selection between groups, and therefore no counterfactual anybody is being asked to supply.
The corpus changing between last week and this week is not an intervention; it is the same
population at a later moment, and nothing about it needs randomising.

So a drift claim is :attr:`tenbin.claims.model.ClaimKind.temporal`, which the gate does not
hold to the mechanism requirement. That is the whole content of the amendment, and it is
one line in ``Claim.requires_assignment_mechanism`` -- which is why it is stated here in prose
as well, because a one-line derivation with no argument next to it is how the decision gets
re-read the other way by the next person who opens decision 002 and wants to build the most
useful thing this corpus can support.

**A drift without the archive is a snapshot repeated, and this module holds the readings the
archive stores.** :class:`Reading` is one rendered figure, stamped with the read that produced
it: the moment, how many documents that read enumerated, and the corpus window it covered. It
holds the value *as rendered* and never parses it back out, for the reason
:class:`~tenbin.corpus.window.CorpusWindow` makes about a half-read window -- a reading of
``"5 of 40 records = 12.5%"`` reconstructed from the string is a transcription, and
transcription is exactly the hazard :class:`~tenbin.report.document.CorpusHeader` exists to
remove. The archive stores what was published; this module reports both readings side by side
and lets the reader weigh them, which is honest for *any* measure including the distributions
whose value has no percentage in it at all.

:class:`Reading` is here and :class:`~tenbin.report.archive.Archive` is over there, and the
split is the dependency direction rather than convenience: this package knows about claims and
corpus and the report layer calls it, so the archive may import a reading from here while a
reading defined in the archive would have to be imported *out* of it. The archive owns the file;
this owns the thing the file is made of.

**Two refusals, and both are about the corpus rather than about the arithmetic.** A pair of
windows of very different lengths is two different claims -- a figure that moved over three
weeks and one that moved over three years -- and there is no honest way to average them, so
:meth:`DriftMeasure.compute` refuses rather than renders. And two reads separated by a
documented corpus event -- a migration, a webhook reinstall -- did not differ because time
passed, so the figure says nothing about drift and is refused for that. **The event is a
parameter and never a detection, because Tenbin cannot observe a webhook reinstall** and
guessing would be worse than asking: a program that inferred "something happened here" from a
jump in a number would be inventing the finding that is the only thing standing between a
change and a cause.

**What this module notably does not do:** it does not read the store, and it holds no client.
It does not compare two *collections*, two *measures*, or two *populations*, and it refuses
all three rather than quietly measuring whichever pairing it was handed -- a drift is within
one population over one measure, and the constructor enforces that. It does not compute a
percentage-point delta, because the two rendered values are the programme's answer and a delta
computed from them would be a third rendering to keep in step. And it does not decide whether
a figure moved: it reports the two readings, and whether they differ is
:attr:`DriftFigure.moved`, which is a fact about the archive rather than a judgement about
whether the change was interesting.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar, Final

from tenbin.claims.mechanism import Granularity, require_text
from tenbin.claims.model import Claim, ClaimKind, Denominator
from tenbin.claims.refusals import ComparativeRefusalError, Refusal
from tenbin.corpus.snapshot import CorpusSnapshot
from tenbin.corpus.window import CorpusWindow
from tenbin.measures.base import Figure

#: How far apart two windows may be before the pair is refused rather than reported. Coarse on
#: purpose: this is a refusal threshold and not a statistic, so it is a factor rather than a
#: test, and two to one is the point at which "these two reads are the same corpus observed a
#: little later" stops being a sentence anybody would write. Exported rather than inlined
#: because a reader who wants to know what the rule is should not have to reverse the
#: comparison out of the refusal text.
WINDOW_LENGTH_RATIO: Final[float] = 2.0

#: The suffix that turns a measure's slug into a drift over it, so a report can hold the
#: reading and the movement of the same measure without the two sections colliding on a slug.
DRIFT_SLUG_SUFFIX: Final[str] = ".drift"

#: What a reader must not conclude from a figure of this shape, and what decision 002 actually
#: refuses. A drift is the most useful thing this corpus can support after a plain count, and a
#: gate or a decision record that read as a blanket ban would cost the programme the ability to
#: answer "is this getting better" -- the question the whole archive exists for. Stated here in
#: the claim's own ``does_not_mean`` and, in the gate's words, in
#: :data:`tenbin.claims.gate.COMPARATIVE_ABSENCE_IS_AN_ASSIGNMENT_MECHANISM`.
DRIFT_IS_NOT_THE_REFUSED_COMPARISON: Final[str] = (
    "This is one population read twice rather than two populations compared, so it is not the "
    "comparison decision 002 refuses: that refusal is about the absence of an assignment "
    "mechanism between principals, and there is no mechanism here to be missing because no "
    "principal was assigned to anything."
)


@dataclass(frozen=True)
class Reading:
    """One archived figure: what it rendered as, and the read that produced it.

    **This lives here rather than in :mod:`tenbin.report.archive` because of the dependency
    direction.** :mod:`tenbin.measures` knows about claims and corpus and the report layer
    calls it; ``report.archive`` may therefore import a reading from this module, while a
    reading defined in the archive would have to be imported *from* it, and the drift measure
    -- which is the only thing that consumes one -- would then be reaching upwards into the
    layer above it to get at a four-field dataclass. The archive owns the file; this owns the
    thing the file is made of.

    Six fields and every one of them is required. ``value`` is the figure's rendered payload,
    stored verbatim and never parsed back into numbers: this is what makes the archive work
    for *any* measure, including the distributions whose value is a set of bucket counts and
    not a figure at all, and it is why a change of measure shows up as a changed reading
    rather than as a changed definition of what a reading means. ``window`` is the other
    load-bearing one -- it is what lets :class:`DriftMeasure` refuse a pair of reads whose
    windows are not comparable rather than reporting a change over three weeks as though it
    were one over three years.

    ``enumerated`` is the read's own count and defaults to nothing rather than to a length: a
    reading with no denominator beside it is a number a reader cannot weigh, and the drift's
    denominator is *both* reads precisely so that forty records and twelve hundred cannot be
    read as the same fact.
    """

    slug: str
    value: str
    collection: str
    read_at: datetime
    enumerated: int
    window: CorpusWindow

    def __post_init__(self) -> None:
        for field in ("slug", "value", "collection"):
            require_text("Reading", field, getattr(self, field))
        if not isinstance(self.window, CorpusWindow):
            raise ValueError(
                f"Reading.window must be a CorpusWindow, not {self.window!r}; a reading whose "
                "span cannot be read cannot be compared against another reading's span, and "
                "that is the comparison the archive exists to make possible."
            )
        if isinstance(self.enumerated, bool) or not isinstance(self.enumerated, int):
            raise ValueError(
                f"Reading.enumerated must be an integer count, not {self.enumerated!r}"
            )
        if self.enumerated < 0:
            raise ValueError(
                f"Reading.enumerated must not be negative; {self.enumerated} documents were not "
                "read fewer than none"
            )
        if not isinstance(self.read_at, datetime) or self.read_at.tzinfo is None:
            raise ValueError(
                f"Reading.read_at must be an aware datetime, not {self.read_at!r}; a reading "
                "nobody can place in time is a reading that cannot be ordered against another."
            )

    def summary(self) -> str:
        """The reading's first line, for a sentence that has to quote a value.

        A distribution's rendered value is several lines of buckets, and a claim is one
        sentence; quoting the whole thing would put a block of numbers in the middle of a
        caveat and the reader would stop reading. The full value is in :attr:`value` and in
        :meth:`tenbin.measures.drift.DriftFigure.value_text`, so nothing is elided from the
        figure -- only from the one sentence that has to be short.
        """
        for line in self.value.splitlines():
            if line.strip():
                return line.strip()
        return self.value


@dataclass(frozen=True)
class CorpusEvent:
    """A documented change to the corpus itself, and how somebody knows it happened.

    Two required fields and no detection anywhere. ``what`` names the event -- a webhook
    reinstall, a migration, a change to what the store enumerates -- and ``documented_by`` names
    the artefact that records it, so the assertion that an event separates two reads carries
    its own evidence rather than resting on whoever passed it.

    **Nothing here checks that the event falls between the two reads being compared.** Tenbin
    cannot observe a webhook reinstall: the store does not record it, the webhook does not
    write to Tanseki, and an inference from a jump in a figure would be the programme inventing
    the finding that is the only thing separating a change from a cause. So the caller says so
    and the drift refuses, and a caller who guesses wrong gets a refusal they did not want,
    which is the right direction to be wrong in.
    """

    what: str
    documented_by: str

    def __post_init__(self) -> None:
        require_text("CorpusEvent", "what", self.what)
        require_text("CorpusEvent", "documented_by", self.documented_by)

    def render_text(self) -> str:
        """The event as one line, naming its evidence, for the refusal that cites it."""
        return f"{self.what} (documented in {self.documented_by})"


def window_divergence(earlier: CorpusWindow, later: CorpusWindow) -> float | None:
    """How much wider the wider of two windows is, or ``None`` if they cannot be weighed.

    The larger span over the smaller, so an unknown window and a window twice the other's width
    are both "not comparable" while remaining two different facts. ``None`` covers the two ways
    a pair cannot be weighed at all: a window whose ends could not be read, and a window with no
    length on either side -- an empty read and a read whose records all share one instant are
    not the same finding, and :func:`window_comparability_problem` says which one it hit.

    Two spans of the same width return ``1.0`` rather than a special case, so a caller asking
    the question always gets a number or nothing.
    """
    earlier_span = earlier.span
    later_span = later.span
    if earlier_span is None or later_span is None:
        return None
    shorter = earlier_span.total_seconds()
    longer = later_span.total_seconds()
    if shorter == 0.0 and longer == 0.0:
        return 1.0
    if shorter == 0.0 or longer == 0.0:
        return None
    return max(shorter, longer) / min(shorter, longer)


def window_comparability_problem(earlier: CorpusWindow, later: CorpusWindow) -> str | None:
    """Why these two windows cannot be compared, or ``None`` when they can.

    A sentence rather than a boolean, because the two ways a pair fails are different facts for
    a reader: a window that could not be read at all is a corpus whose timestamps do not parse,
    and a pair that differ by a factor of three is a corpus that grew. Both are refused, and a
    single ``False`` would render both as one unexplained silence.

    The threshold is :data:`WINDOW_LENGTH_RATIO`, and it is deliberately coarse. A figure that
    moved over three weeks and one that moved over three years are different claims, and the
    honest answer to "which of those moved" is neither of them without saying which window it
    was.
    """
    if not earlier.is_known:
        return (
            f"The window of the earlier read could not be read: {earlier.render_text()} A drift "
            "cannot be dated against a window of unknown length, and reporting one anyway "
            "would be a duration nobody measured."
        )
    if not later.is_known:
        return (
            f"The window of the later read could not be read: {later.render_text()} A drift "
            "cannot be dated against a window of unknown length, and reporting one anyway "
            "would be a duration nobody measured."
        )
    divergence = window_divergence(earlier, later)
    if divergence is None:
        return (
            f"One of these windows spans no time at all ({earlier.render_text()}) "
            f"({later.render_text()}), so the two cannot be placed on the same scale and a "
            "change between them is a change between a point and a length."
        )
    if divergence > WINDOW_LENGTH_RATIO:
        return (
            f"These two windows are not comparable: {earlier.render_text()} "
            f"{later.render_text()} The wider is {divergence:.1f} times the narrower, past the "
            f"{WINDOW_LENGTH_RATIO:g} to 1 this package will compare, because a measure that "
            "moved over three weeks and one that moved over three years are different claims "
            "about the same corpus."
        )
    return None


@dataclass(frozen=True)
class DriftFigure(Figure):
    """Two readings of one measure, put side by side, with the change stated as an observation.

    A fifth payload kind rather than a distribution or a finding, and both refusals are
    deliberate. A :class:`~tenbin.measures.base.DistributionFigure` would want bucket *counts*,
    and the honest numbers here are two rendered values that are not counts of anything.
    A :class:`~tenbin.measures.base.Finding` would be the wrong shape for a different reason:
    :func:`tenbin.report.build.build_report` hoists every finding to the front of a report as
    a matter of discipline, and a drift is not a finding about whether the corpus is sound -- it
    is the closing observation, and it would jump above the numbers it is about if it were
    allowed to masquerade as one.

    ``moved`` is the whole of the arithmetic. There is no percentage-point delta because the two
    values are the programme's own renderings and a third number computed from them would be a
    third thing to keep in step with the first two.
    """

    earlier: Reading
    later: Reading
    payload: ClassVar[str] = "drift"

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("earlier", "later"):
            if not isinstance(getattr(self, name), Reading):
                raise ValueError(
                    f"DriftFigure.{name} must be a Reading, not {getattr(self, name)!r}; a drift "
                    "over something nobody archived is a comparison with no earlier observation."
                )

    def __hash__(self) -> int:
        # Repeated from the base rather than inherited, because ``@dataclass`` writes a
        # generated ``__hash__`` onto every class it decorates and shadows the base's. The
        # identity is what the figure is *about* -- the claim, the read it renders over, the
        # title and the kind of payload -- for the reason
        # ``tenbin.measures.base._payload_hash`` gives: two figures computed the same way
        # over the same read hash alike.
        return hash((self.claim.slug, self.snapshot, self.title, self.payload))

    @property
    def moved(self) -> bool:
        """Whether the two archived readings render differently from one another.

        A comparison of two strings rather than of two numbers, and that is deliberate: the
        archive stores what was published, so "did it move" is a question about the published
        text and no amount of arithmetic on the other side of it could answer for it. Two
        readings that differ by a rounding of a rendered percentage *did* move, and a reader
        who thinks otherwise should be comparing the measures rather than the archive.
        """
        return self.earlier.value != self.later.value

    def value_text(self) -> str:
        """Both readings, the verdict on whether they differ, and each read's window.

        Both windows rather than the pair's, because each belongs to its own read and a single
        combined sentence would have to pick one of them to be the anchor -- and the anchor is
        the choice a reader is most likely to get wrong when one window is a prefix of the
        other, which is the ordinary case for an archive.
        """
        verdict = (
            "The figure moved between these two reads."
            if self.moved
            else "The figure read the same at both of these reads."
        )
        return "\n".join(
            (
                f"At {self.earlier.read_at.isoformat()}: {self.earlier.value}",
                f"At {self.later.read_at.isoformat()}: {self.later.value}",
                verdict,
                f"Window of the earlier read: {self.earlier.window.render_text()}",
                f"Window of the later read: {self.later.window.render_text()}",
            )
        )


def drift_slug(slug: str) -> str:
    """The claim slug for the drift over ``slug``, distinct from the slug itself.

    Distinct because a report holds one section per slug and a measure's reading and its
    movement are two different claims about two different populations: the reading is over one
    read, the drift is over both. A shared slug would make the second one a duplicate of the
    first, which :class:`~tenbin.report.document.Report` refuses at construction.
    """
    return f"{slug}{DRIFT_SLUG_SUFFIX}"


def drift_claim(earlier: Reading, later: Reading) -> Claim:
    """The claim behind a drift, over both reads and stated as one population twice.

    **The denominator is both reads and names both counts, because the two facts a reader needs
    are different facts.** A change from 12% to 14% over forty records and the same two
    percentages over twelve hundred are the same percentage points and completely different
    events: the first may be one record moving, the second may be two hundred. A drift whose
    denominator held only the later read -- or only the earlier one -- would make those two
    indistinguishable, and the size here is the sum so that a reader dividing the figure by it
    lands on the total number of records observed across both moments.

    The floor of one follows the convention every other denominator here uses: two empty reads
    are a real answer rather than a division, and ``Denominator`` refuses a size below one, so
    the honest rendering is over one with the counts in the description saying what they were.
    """
    return Claim(
        slug=drift_slug(earlier.slug),
        statement=(
            f"Read twice over {earlier.collection}, {earlier.slug} read "
            f"“{earlier.summary()}” at the read of {earlier.read_at.date()} and "
            f"“{later.summary()}” at the read of {later.read_at.date()}."
        ),
        does_not_mean=(
            "That anything changed about the work rather than about the measure, and not that "
            "anything caused the change. "
            + DRIFT_IS_NOT_THE_REFUSED_COMPARISON
            + " The corpus moving between two moments is not a treatment: there is no arm to "
            "assign anyone to and no counterfactual to supply. It also does not mean the two "
            "reads are the same size — the earlier enumerated "
            f"{earlier.enumerated:,} records and the later {later.enumerated:,}, and the same "
            "percentage points over two different populations is a different fact."
        ),
        falsifier=(
            "Two archived reads separated by a documented corpus event rather than by the "
            "passage of time — a migration, a webhook reinstall, a change to what the store "
            "enumerates — in which the figure moved. That is not drift. Nothing in the corpus "
            "marks such an event, so the caller supplies it and this figure is refused over any "
            "pair with one in it rather than reported over it. The second falsifier is a pair of "
            "windows that differ in length by more than "
            f"{WINDOW_LENGTH_RATIO:g} to one, which is two claims about the same corpus rather "
            "than one claim over time."
        ),
        denominator=Denominator(
            description=(
                f"both archived reads of {earlier.collection}: {earlier.enumerated:,} records at "
                f"the read of {earlier.read_at.date()} and {later.enumerated:,} at the read of "
                f"{later.read_at.date()}"
            ),
            size=max(1, earlier.enumerated + later.enumerated),
        ),
        kind=ClaimKind.temporal,
        granularity=Granularity.corpus,
    )


@dataclass(frozen=True)
class DriftMeasure:
    """One measure compared against its own past value, over two archived readings.

    Both readings are required and neither is derived from the other, so a drift cannot be
    computed from a single read and then described as a comparison: there is no default "the
    previous one" to fall back on, and an archive with one reading in it is not a program that
    can drift. The constructor refuses the three pairings that are not one population observed
    twice -- two measures, two collections, and two reads that are not in order -- because a
    caller who means one of those has a different figure and should have to name it.

    ``corpus_event`` is supplied by the caller and defaults to none. See :class:`CorpusEvent`:
    Tenbin cannot observe a migration or a webhook reinstall, so the honest handling of a
    pair that somebody knows was separated by one is to be told about it and refuse.

    The ``snapshot`` passed to :meth:`compute` and :meth:`claim` is the read that backs the
    figure, and the completeness sentence :meth:`~tenbin.measures.base.Figure.rate_text`
    renders comes from it. Pass the *later* read: the figure's own backing read is the one a
    reader is about to be asked to trust, and a drift whose later reading came from a truncated
    walk should say so beside the number rather than two reports away.
    """

    earlier: Reading
    later: Reading
    corpus_event: CorpusEvent | None = None

    def __post_init__(self) -> None:
        for name in ("earlier", "later"):
            if not isinstance(getattr(self, name), Reading):
                raise ValueError(
                    f"DriftMeasure.{name} must be a Reading, not {getattr(self, name)!r}; a "
                    "measure compared against its own past value needs the past to have been "
                    "archived, and there is nothing here to substitute for it."
                )
        if self.earlier.slug != self.later.slug:
            raise ValueError(
                f"DriftMeasure compares one measure against its own past value, and these two "
                f"readings are of {self.earlier.slug!r} and {self.later.slug!r}. A difference "
                "between two measures is not drift in either one; it is two measures."
            )
        if self.earlier.collection != self.later.collection:
            raise ValueError(
                f"DriftMeasure compares one collection against its own past, and these two "
                f"readings are of {self.earlier.collection!r} and "
                f"{self.later.collection!r}. Nothing here can place a movement between two "
                "populations onto one population, and a collection is a population."
            )
        if self.earlier.read_at >= self.later.read_at:
            raise ValueError(
                f"DriftMeasure.earlier was read at {self.earlier.read_at.isoformat()} and "
                f"DriftMeasure.later at {self.later.read_at.isoformat()}, which is not a drift. "
                "Two readings of one read are refused by the archive, and two reads in the other "
                "order are a caller who passed them the wrong way round."
            )

    @property
    def slug(self) -> str:
        """The identifier this measure is cited by, and it is the drift's own slug.

        Derived from the readings rather than stored, because a stored slug here is a fourth
        field that can disagree with the two it describes and a drift under the wrong slug is a
        section a reader cannot look up against the measure it is about.
        """
        return drift_slug(self.earlier.slug)

    def claim(self, snapshot: CorpusSnapshot) -> Claim:
        """The claim, over both reads.

        The snapshot is not consulted and is taken only to satisfy the measure protocol: a
        claim's denominator is a population, and a drift's population is *both* archived reads,
        which are already in hand. A caller who expects the snapshot's counts here would get
        this week's figures substituted into a claim about two moments, which is the one thing
        the denominator is for.
        """
        return drift_claim(self.earlier, self.later)

    def compute(self, snapshot: CorpusSnapshot) -> Figure:
        """The drift figure, or a refusal naming what the corpus will not support.

        Two refusals and they are checked in the order that puts the unfixable first. A
        documented corpus event cannot be undone by re-reading anything, so a pair with one in
        it is refused before anything is computed at all; a pair of incomparable windows is
        refused next, because the remedy there is a *different* pair rather than a fix to this
        one.

        Both raise :class:`~tenbin.claims.refusals.ComparativeRefusalError` rather than
        returning a figure with a caveat attached, and the choice is the point: a drift over a
        pair that did not drift is not a finding with a limitation, it is a number about
        something other than what it says, and the one thing a drift figure must never be is
        that.
        """
        event = self.corpus_event
        if event is not None:
            raise ComparativeRefusalError(self._refusal(_confounded_reason(event)))
        problem = window_comparability_problem(self.earlier.window, self.later.window)
        if problem is not None:
            raise ComparativeRefusalError(self._refusal(problem))
        return DriftFigure(
            claim=self.claim(snapshot),
            snapshot=snapshot,
            title=(
                f"How {self.earlier.slug} read at the earlier of two archived reads and at the "
                "later one"
            ),
            earlier=self.earlier,
            later=self.later,
        )

    def _refusal(self, reason: str) -> Refusal:
        """The refusal this measure produces, over the slug the two readings share.

        ``unblocked_by`` is ``None`` and no ticket is offered, for the reason every other
        refusal without a blocker gives: no fact arriving from Kojutsu would make a pair of
        reads separated by a migration into a pair of reads that drifted, so an empty ticket
        list there would be a false promise of work that cannot help.
        """
        return Refusal(
            claim_slug=self.slug,
            title="Drift between two archived reads",
            reason=reason,
            missing_fact=None,
            unblocked_by=None,
        )


def _confounded_reason(event: CorpusEvent) -> str:
    """Why a pair separated by a documented event is not drift, stated as a refusal.

    The word "confounded" is used deliberately and the sentence around it does the work. These
    two reads did not differ because time passed; they differed because the corpus did, for a
    reason somebody recorded. Any movement between them is that reason's effect until somebody
    shows otherwise, and a figure that rendered anyway would be the programme's most confident
    number with nothing behind it.
    """
    return (
        f"These two reads are separated by a documented corpus event -- {event.render_text()} "
        "-- rather than by the passage of time, so the pair is confounded and the difference "
        "between them is not drift. A migration, a reinstalled webhook or a change to what the "
        "store enumerates moves the figure by itself, and nothing in the corpus distinguishes "
        "that from a process that changed, which is why the event is supplied by the caller and "
        "never inferred. This pair renders no drift until somebody supplies two reads with no "
        "such event between them, and no Kojutsu ticket delivers those: the remedy is a "
        "different pair of reads, not a different corpus."
    )
