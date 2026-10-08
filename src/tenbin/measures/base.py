"""A figure is a claim, a snapshot, and a payload -- and the snapshot is the point.

Every number this program prints is attached to a :class:`~tenbin.claims.model.Claim`
that says what it means, what it does not mean, and what would falsify it. That
half is :mod:`tenbin.claims`' argument and this module does not restate it. What
this module adds is the reason a claim is not enough on its own: a claim says how
to read a number, and it says nothing about whether the number is *whole*. So a
figure carries the :class:`~tenbin.corpus.snapshot.CorpusSnapshot` it was
computed over, and :meth:`Figure.rate_text` -- which takes no caller-supplied
wording at all, because caller-supplied wording is a caveat a caller can forget --
renders whether the corpus behind the figure was read to its end or is a prefix
of one. A renderer physically cannot print a number without holding the object
that says whether the number is whole, because the number and the object are the
same value.

**A distribution has to be able to say what it left out.** :class:`DistributionFigure`
carries ``excluded`` beside ``values`` and the two are not optional, which is the
difference between this and the ``unknown``-model bug the whole package exists to
stop. A bucket that did not occur and a bucket that was filtered out are the same
number in ``values`` and completely different facts; a figure that cannot say
which it is showing is a figure whose silence is indistinguishable from a result.
So :func:`histogram` and :func:`duration_buckets` live here rather than in each
measure: two measures that disagreed about what a bucket is would be two
definitions of the same axis.

**A population is a span and a count, and a whole-corpus denominator says only the
count.** A rate's denominator is where its population is stated, and a population
stated in records alone is still unstated if nobody says the records span three
weeks or three years. The window therefore lives in the corpus header -- the first
content of every report, holding it once for every figure below it -- and *not* in
each denominator, because twenty measures repeating "over a corpus spanning X to Y"
is a report nobody reads, and the header carries it once, correctly, for all of
them. :func:`subset_denominator` is the exception and the reason one is needed: a
measure whose population is narrower than the corpus -- the custody measure's files,
the staleness measure's rationales -- has a span the corpus header does not
describe, so that one states its own.

**What this module notably does not do:** it does not decide what is worth
measuring, and it holds no opinion about any record. A figure is a value plus the
two objects that make it interpretable; the refusal to measure something is
:mod:`tenbin.claims.registry`'s job and the judgement that a figure is *about*
the wrong population is the gate's. Nothing here renders, formats or writes --
a report builds its own layout from :meth:`Figure.value_text` and
:meth:`Figure.rate_text`, and the reason this module does not do it is that a
renderer in this package could not be prevented from choosing to print a count
without the completeness sentence beside it.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar, Final, Protocol

from tenbin.claims.mechanism import require_text
from tenbin.claims.model import Claim, Denominator
from tenbin.corpus.record import Record
from tenbin.corpus.snapshot import Completeness, Snapshot, Truncation
from tenbin.corpus.window import CorpusWindow

#: A whole corpus, in the words a reader needs. Named so that the four
#: completeness outcomes cannot drift into four different claims about the same
#: read, and because this sentence is the *only* thing standing between a number
#: and a reader who assumes it covers everything the store holds.
#: ``{collection}`` is substituted from the snapshot at render time. It was a
#: constant naming "document" and "store", which were true of the one corpus this
#: package originally read and are false of every other one: a ticket transition
#: is not a document and a ticket source is not the store. A rate sentence that
#: names the wrong unit is worse than one that names none, because it tells a
#: reader the figure covers documents when what it covers is transitions.
WHOLE_CORPUS: Final[str] = (
    "Whole read: this figure covers every record {collection} enumerated on this "
    "read, and its own count agreed with the walk."
)

#: A prefix, so the number is an undercount and the size of the hole is the first
#: thing a caveat should say.
TRUNCATED_BY_OFFSET: Final[str] = (
    "Truncated read: this figure covers a prefix of {collection} and not all of it, "
    "so every count here is an undercount."
)

#: The walk failed, or a document that was listed was not there to be fetched. The
#: exception is on the snapshot, so this sentence points at it rather than
#: restating a cause nobody can act on.
WALK_FAILED: Final[str] = (
    "The walk did not finish: this figure covers whatever had been read when "
    "{collection} failed or a listed record was missing, so every count here is a "
    "lower bound."
)

#: The store's own count disagreed with its own listing, in either direction. The
#: number below is not a count of anything, which includes the number that says
#: how far the walk got.
COUNT_CONTRADICTED: Final[str] = (
    "{collection}'s own count disagreed with what it served, so the read has no "
    "agreed size: nothing that depends on the count, including this figure, can be "
    "trusted."
)

#: What a rate is a rate over, as one line. Used by :meth:`Rate.rate_text` and by
#: every figure that renders a rate, so the population is stated in the same words
#: everywhere.
POPULATION_PREFIX: Final[str] = "of"


@dataclass(frozen=True)
class Rate:
    """A numerator over a named, counted population, and nothing more.

    The two fields are the whole of it. There is no percentage field, no
    ``is_percentage`` flag and no rounding, because each of those is a decision
    about how a number will be *read* and every one of them loses information
    that the denominator's description was written to preserve. A rate is a
    fraction; what fraction, over what, is the caller's claim to make and this
    type's job is to refuse to let the two drift apart.
    """

    numerator: int
    denominator: Denominator

    @property
    def value(self) -> float:
        """The rate as a fraction, which is the only form this package renders."""
        return self.numerator / self.denominator.size

    def rate_text(self) -> str:
        """Render the rate and its population as one line, so neither can be dropped.

        Both halves are required in a single string. A rate printed without its
        denominator reads as general, and a denominator printed without its rate
        is a number nobody asked for.
        """
        return (
            f"{self.numerator:,} {POPULATION_PREFIX} {self.denominator.render_text()} "
            f"= {self.value:.1%}"
        )


def subset_window(records: Iterable[Record]) -> CorpusWindow:
    """The window of a population narrower than the corpus, over those records only.

    A function rather than an open-coded ``CorpusWindow.from_records`` at each call
    site, because the failure it exists to prevent is a subset measure borrowing the
    corpus's window: the header already states the whole read's span, so a per-file
    denominator that repeated it would look like it had described its own population
    and would be describing somebody else's. The two windows differ whenever the
    subset is a fraction of the corpus, and for the measures this exists for they
    nearly always are -- files are touched by a minority of records, and rationales
    are a minority of everything.
    """
    return CorpusWindow.from_records(records)


def subset_denominator(
    description: str,
    size: int,
    records: Iterable[Record],
) -> Denominator:
    """A denominator for a subset, carrying that subset's own window in its description.

    **This is the one place a denominator states a window, and the exception is the
    point.** Every other denominator in this program is a whole-corpus population, and
    the corpus header already carries the read's span -- so repeating it per measure
    would put the same sentence twenty times in one report, which is a document nobody
    reads, and a reader who sees it twenty times stops reading it. A subset's span is
    not that sentence: the header describes the whole read, and a population of one
    repository's files inside a read of thirty repositories has a different and much
    shorter window, which no header line states.

    The window rides in the ``description`` rather than in a field of
    :class:`~tenbin.claims.model.Denominator` because that type's two required fields
    are a name and a count, and a name is where a reader looks for the scope. A field
    would need rendering in :meth:`Denominator.render_text` to be visible at all, and
    the field is a change to a package this ticket does not hold; the exact change is
    written up with the rest of them rather than smuggled in here as a helper.

    ``size`` is the caller's to floor, and the floor is the same one every other
    denominator in this program applies: a subset with no records in it is reported
    over one, with the claim saying so, rather than raising -- see
    :func:`tenbin.measures.staleness.staleness_claim` for the convention.
    """
    return Denominator(f"{description}; {subset_window(records).render_text()}", size)


@dataclass(frozen=True)
class Figure:
    """A claim, the read it was computed over, and a title for the reader.

    Three fields, all required, and :meth:`__post_init__` refuses a missing claim
    and a blank title. A figure with no claim is the thing this whole program
    exists to make unconstructible, and a figure with a blank title is one that
    reaches a report labelled by nothing -- which is how a distribution of review
    verdicts becomes a bar chart somebody reads as a distribution of review
    *quality*.

    ``snapshot`` is required for the same kind of reason and is checked as well.
    :meth:`rate_text` is the method that makes it load-bearing, and a renderer
    holding a figure is holding the evidence about whether the figure is whole --
    not a boolean somebody remembered to pass, and not a field a report could
    choose to read or not read.

    Subclasses add the payload. Each overrides :meth:`value_text` so that a
    renderer has one place to get the number, which is the same chokepoint
    argument one level down: there is no method that renders a figure's payload
    and stops short of the object that says whether the payload is complete.
    """

    claim: Claim
    snapshot: Snapshot
    title: str

    #: The four payload kinds, named so a report can branch on them without
    #: importing four classes or, worse, guessing from the presence of an
    #: attribute.
    payload: ClassVar[str] = "figure"

    def __post_init__(self) -> None:
        if not isinstance(self.claim, Claim):
            raise ValueError(
                "Figure.claim is required; a figure without a claim is a claim, which "
                "is the thing this package exists to make unconstructible."
            )
        if not isinstance(self.snapshot, Snapshot):
            raise ValueError(
                "Figure.snapshot is required; a figure that cannot be asked whether the "
                "corpus behind it is whole is a number with nothing behind it."
            )
        require_text("Figure", "title", self.title)

    def value_text(self) -> str:
        """Render the payload. Empty on a bare figure, which has no payload.

        Not abstract, because a figure carrying only a claim and a title is a real
        thing to want -- the shape a measure produces when its answer is that the
        number does not exist -- and forcing it to invent a number to satisfy an
        abstract method would be the wrong pressure.
        """
        return ""

    def rate_text(self) -> str:
        """State whether the corpus behind this figure is whole, in no caller's words.

        This is the method the whole design exists to make unavoidable. It reads
        :attr:`~tenbin.corpus.snapshot.CorpusSnapshot.completeness` and names the
        truncation when there is one -- including how many records were never
        reached -- and states completeness when there is not. There is no
        parameter and no override: a caller cannot supply a sentence, cannot
        supply ``None`` to mean "no caveat needed", and cannot reach a renderer
        that prints a number without this having run.

        The name is about the figure's *representativeness* rather than about a
        numeric rate, and it is the same name on every payload kind for that
        reason: a count, a distribution and a finding are all "how much of the
        store did this come from", and a caller that had to ask which figure kinds
        carry the sentence would eventually ask one that did not.

        The two truncating outcomes get different sentences on purpose. A walk
        stopped by the store's offset ceiling is a prefix with a known length, and
        a walk that failed is a prefix with an exception attached; rendering them
        with one sentence would be to tell a reader that the shortfall is known in
        both cases, which it is not.
        """
        completeness = self.snapshot.completeness
        collection = self.snapshot.collection
        if completeness is Completeness.COMPLETE:
            return WHOLE_CORPUS.format(collection=collection)
        if completeness is Completeness.OFFSET_CAP:
            return f"{TRUNCATED_BY_OFFSET.format(collection=collection)} {self._missing_clause()}"
        if completeness is Completeness.STORE_TOTAL_EXCEEDED:
            return COUNT_CONTRADICTED.format(collection=collection)
        return f"{WALK_FAILED.format(collection=collection)} {self._error_clause()}"

    def _missing_clause(self) -> str:
        """How many documents the walk never reached, when the ceiling stopped it."""
        truncation = self.snapshot.truncation
        if not isinstance(truncation, Truncation):
            # An offset-cap snapshot with no truncation detail is a contradiction,
            # and the honest rendering of one is the sentence that does not
            # invent a count. The completeness statement above still stands.
            return "How many records were not reached is not recorded on this snapshot."
        missing = truncation.records_missing
        noun = "record was" if missing == 1 else "records were"
        return f"{missing:,} {noun} not reached, the walk having stopped at offset {truncation.offset_reached:,}."

    def _error_clause(self) -> str:
        """The exception that stopped the walk, when the snapshot carries one."""
        error = self.snapshot.error
        if error is None:
            return "The snapshot does not carry the failure that stopped it."
        return f"It failed with: {error}"

    def __hash__(self) -> int:
        return _payload_hash(self)


@dataclass(frozen=True)
class CountFigure(Figure):
    """How many, with no population behind it beyond the read it was taken from.

    A count is the honest shape for a figure with no rate to compute, and it is
    the right answer in the one case a rate gets wrong: a store that was reached
    and holds nothing. There is no population to be a fraction of, so a rate over
    an empty corpus would be a division whose result nobody can act on, and
    ``0/0`` is not one.
    """

    value: int
    payload: ClassVar[str] = "count"

    def __hash__(self) -> int:
        return _payload_hash(self)

    def value_text(self) -> str:
        return f"{self.value:,}"


@dataclass(frozen=True)
class RateFigure(Figure):
    """A fraction over a named population, refused when the fraction is impossible.

    Two refusals, and both are here rather than in :class:`Rate` because a rate is
    also built by things that are not a figure. The denominator of zero is
    unreachable while :class:`~tenbin.claims.model.Denominator` refuses to be
    built at that size, and it is checked anyway so that the two conditions
    cannot be separated: a refactor that made the denominator a plain integer
    would get a division by zero from a renderer instead of a sentence naming the
    field.

    The numerator check is the live one. A numerator above its denominator is a
    rate above 1, which is either a unit error -- a count of reviewers over a
    count of changes -- or a bug in a numerator, and both are defects that would
    otherwise render as ``183.3%`` and be read as a finding. The snapshot layer
    produces such a case for real: ``STORE_TOTAL_EXCEEDED`` means the store's own
    count is smaller than what it served.
    """

    rate: Rate
    payload: ClassVar[str] = "rate"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.rate, Rate):
            raise ValueError(
                "RateFigure.rate is required; a figure labelled a rate with no rate in it "
                "is a number whose population was never written down."
            )
        if self.rate.denominator.size < 1:
            raise ValueError(
                "RateFigure.rate.denominator.size must be at least 1; a rate over zero "
                "records is not a rate, and an empty population is a CountFigure."
            )
        if self.rate.numerator > self.rate.denominator.size:
            raise ValueError(
                f"RateFigure.rate.numerator ({self.rate.numerator:,}) exceeds its "
                f"denominator ({self.rate.denominator.size:,}); a rate above 1 is either a "
                "unit error or a bug, and neither should reach a report."
            )

    def __hash__(self) -> int:
        return _payload_hash(self)

    def value_text(self) -> str:
        return self.rate.rate_text()


@dataclass(frozen=True)
class DistributionFigure(Figure):
    """Buckets, and -- the load-bearing half -- what was left out of them.

    ``values`` is what the measure counted. ``excluded`` is everything the measure
    declined to count *and why*, keyed by a phrase that names the condition rather
    than by a code, because a report renders these keys verbatim and
    ``"records_with_no_review_id"`` is not a sentence a reader can act on.

    The distinction is not pedantry. A bucket that did not occur and a bucket that
    was filtered out are the same integer in ``values`` and completely different
    facts, and the way they get confused is by dropping the exclusions and
    letting the absence read as a result -- which is precisely how a corpus's
    ``unknown`` model bucket becomes a group of records whose authors declared a
    model called ``unknown``.
    """

    values: Mapping[str, int]
    excluded: Mapping[str, int]
    payload: ClassVar[str] = "distribution"

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "values", freeze_counts(self.values, "DistributionFigure.values"))
        object.__setattr__(
            self, "excluded", freeze_counts(self.excluded, "DistributionFigure.excluded")
        )

    def __hash__(self) -> int:
        return _payload_hash(self)

    @property
    def counted(self) -> int:
        """How many records the buckets account for."""
        return sum(self.values.values())

    @property
    def excluded_total(self) -> int:
        """How many records the measure set aside, across every named condition."""
        return sum(self.excluded.values())

    def value_text(self) -> str:
        parts = [f"{label}: {count:,}" for label, count in self.values.items()]
        if self.excluded:
            parts.append("excluded:")
            parts.extend(f"  {label}: {count:,}" for label, count in self.excluded.items())
        return "\n".join(parts)


@dataclass(frozen=True)
class Finding(Figure):
    """A claim that something is wrong, carrying the same four claim fields as any figure.

    A finding is not a measurement, and it is not a softer measurement. It asserts
    that an observation does not look like what an honest system produces, and it
    carries the whole apparatus of a claim because a finding published without
    its limits is the most confident and most wrong sentence a measurement program
    can print. Everything in :class:`tenbin.claims.model.Claim` applies here
    unchanged, including the requirement to name what would falsify it.

    ``suspected_cause`` and ``where`` are optional because a finding sometimes has
    neither, and a blank string is not the way to say so: ``""`` renders as a
    paragraph with an empty heading in it. ``cannot_confirm`` is required, cannot be
    blank, and is the field that keeps a finding an observation. A detector that
    read a store and did not run the code that writes it has candidates, not a
    diagnosis, and a finding whose author could not write that sentence has either
    confirmed something or decided to imply it had.
    """

    what_was_observed: str
    suspected_cause: str | None
    where: str | None
    cannot_confirm: str
    payload: ClassVar[str] = "finding"

    def __post_init__(self) -> None:
        super().__post_init__()
        require_text("Finding", "what_was_observed", self.what_was_observed)
        require_text("Finding", "cannot_confirm", self.cannot_confirm)
        for field in ("suspected_cause", "where"):
            value = getattr(self, field)
            if value is not None:
                require_text("Finding", field, value)

    def __hash__(self) -> int:
        return _payload_hash(self)

    def value_text(self) -> str:
        lines = [f"Observed: {self.what_was_observed}"]
        if self.where is not None:
            lines.append(f"Where: {self.where}")
        if self.suspected_cause is not None:
            lines.append(f"Suspected cause: {self.suspected_cause}")
        lines.append(f"Cannot confirm: {self.cannot_confirm}")
        return "\n".join(lines)


@dataclass(frozen=True)
class FigureGroup(Figure):
    """Several figures sharing one claim, because they are one measure's answer.

    A ``Figure`` rather than a bare tuple, so a measure that answers in three
    parts still satisfies the :class:`Measure` protocol and a report iterating
    measures cannot silently drop the last two. That is the whole reason for the
    class: a report that renders ``measure.compute(snapshot)`` and reaches a
    ``Figure`` knows there may be more underneath, and a measure that returned a
    tuple would make the difference invisible until somebody's figure was missing.

    Sharing one claim is a constraint, not a convenience. Every member is a
    statement over the same population, because the claim names the denominator
    and a figure carrying a claim whose denominator does not describe it is the
    exact defect this package is built to stop. A measure whose parts genuinely
    differ in population is two measures, and the way to tell is to try to write
    the denominator: if no single population fits, the group is the mistake.
    """

    figures: tuple[Figure, ...]
    payload: ClassVar[str] = "group"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.figures:
            raise ValueError(
                "FigureGroup.figures must not be empty; a group with no figures is a "
                "title and a claim, which is a Figure."
            )
        for figure in self.figures:
            if not isinstance(figure, Figure):
                raise ValueError(
                    "FigureGroup.figures must hold figures; a payload a renderer cannot "
                    "recognise is a payload nobody will render."
                )

    def __hash__(self) -> int:
        return _payload_hash(self)

    def value_text(self) -> str:
        return "\n".join(f"{figure.title}\n{figure.value_text()}" for figure in self.figures)


class Measure(Protocol):
    """Something that turns one read of the corpus into a figure, and is nothing else.

    A ``Protocol`` rather than a base class, for two reasons that are really one.
    A measure can be a plain callable with metadata hung off it, so a report can
    carry one that was written for a single run without this package having to
    invent a class hierarchy for it. And a measure that *cannot* be constructed --
    one the corpus cannot support -- belongs in the refusal catalogue, where it is
    a first-class value with a slug, rather than in a subclass whose constructor
    raises, which is a thing a report has to catch and a thing nobody remembers to
    catch.

    ``claim`` takes the snapshot because a claim's denominator is a population and
    a population is a fact about a read rather than about a measure. A measure over
    a fixed population -- a team, a repository -- ignores the argument; the
    completeness measure cannot, because its denominator is the store's own count
    and a placeholder size would be a rate over a number nobody wrote down. The
    returned claim is the one the figure carries, so the two cannot disagree.

    **The parameter is :class:`Snapshot`, not ``CorpusSnapshot``, and that is what lets a
    read over something other than documents become a report.** A measure that needs
    records may still narrow it back to ``CorpusSnapshot`` -- several do -- and nothing
    stops it: the classes here are plain, not declared as ``Measure``, and **mypy does
    not catch the narrowing.** Verified, not assumed: assigning a measure whose
    ``compute`` takes ``CorpusSnapshot`` to a ``Measure``-typed variable passes mypy
    clean, because mypy compares protocol method parameters bivariantly.

    So this protocol is a promise about what a measure may be *given*, not a guarantee
    about what every measure can *use*. The runtime guard is in
    :func:`tenbin.report.build.build_report`, which refuses to choose the project
    measures for a read that carries no records -- and the guard is a test, in
    ``tests/test_report_over_a_second_source.py``. Treat this signature as the shape of
    the seam, and the refusal as the thing actually holding it up.
    """

    @property
    def slug(self) -> str:
        """The identifier this measure is cited by, and the one a refusal would use."""
        ...

    def claim(self, snapshot: Snapshot) -> Claim:
        """The claim every figure this measure produces is attached to."""
        ...

    def compute(self, snapshot: Snapshot) -> Figure:
        """Compute the figure, as a pure function of the snapshot given."""
        ...


def _payload_hash(figure: Figure) -> int:
    """The hash every figure shares: what it is about, not what it holds.

    A frozen dataclass generates a hash over *all* its fields, and two of the
    payloads are mappings -- so a generated hash would hash a ``MappingProxyType``
    and raise, and ``eq=False`` on the payloads would make two distributions with
    different counts compare equal. The hash is therefore stated once, here, and
    each subclass delegates to it: identity is the claim, the title and the kind of
    payload, which is what a figure is *about*. Two figures computed the same way
    over the same read hash alike, so a report that put them in a set gets one
    answer rather than an exception.

    The delegation is repeated in each subclass rather than inherited because
    ``@dataclass`` writes a generated ``__hash__`` onto every class it decorates,
    shadowing the base's. That is why a base-class hash on a dataclass hierarchy
    is decoration and the one-liners below are the actual mechanism.
    """
    return hash((figure.claim.slug, figure.snapshot, figure.title, figure.payload))


def histogram(values: Iterable[str]) -> Mapping[str, int]:
    """Count text values, most frequent first, frozen against later mutation.

    Sorted by count descending and then by key, so two histograms over equal data
    are *equal* and a test can assert on one rather than eyeballing the order. The
    secondary sort is what makes that true: two buckets with the same count in
    different insertion orders would otherwise render differently and compare
    unequal, and the first of those is a flaky test and the second is a diff.

    The ordering is count, never a value's own scale. A histogram over a *ranked*
    vocabulary that came out in rank order would be an argument about which bucket
    is stronger, and this function has no way to express that on purpose: it knows
    nothing about the values it is counting.
    """
    counts = Counter(values)
    return freeze_counts(
        dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))), "histogram"
    )


def freeze_counts(counts: Mapping[str, int], owner: str) -> Mapping[str, int]:
    """Copy a count mapping and refuse anything that is not a non-negative count.

    A count is a fact about how many things were observed, so a negative one is a
    defect and a non-integer one is a unit error, and both would render as a
    number in a report. Refusing them here means the check runs once rather than
    in every measure that builds a figure, and the frozen copy is what makes a
    figure hashable -- see :meth:`Figure.__hash__`.
    """
    frozen: dict[str, int] = {}
    for key, count in counts.items():
        if isinstance(count, bool) or not isinstance(count, int):
            raise ValueError(f"{owner}[{key!r}] must be an integer count, not {count!r}")
        if count < 0:
            raise ValueError(
                f"{owner}[{key!r}] must not be negative; {count} things did not happen"
            )
        frozen[str(key)] = count
    return MappingProxyType(frozen)


#: The upper edges of a duration bucket, in hours, shortest first. Chosen to be
#: legible rather than statistical: the edges a reader already has a word for are
#: the ones they can act on, and a finer ladder would invite a comparison between
#: two buckets that differ by a bucket width. The last bucket is open-ended, so a
#: duration is always in exactly one of them.
HOUR_BUCKET_EDGES: Final[tuple[float, ...]] = (1.0, 4.0, 24.0, 72.0, 168.0, 720.0)

#: The label for the open-ended top bucket, and for the whole ladder's smallest
#: one. A bucket with no lower edge and a bucket with no upper edge are the two
#: that get mislabelled most, because both are written "else".
LESS_THAN_ONE_HOUR: Final[str] = "under 1 hour"
MORE_THAN_720_HOURS: Final[str] = "over 30 days"

#: A duration in the wrong direction, which is clock skew between two writers
#: rather than a fast revision. Its own bucket rather than the smallest one,
#: because a negative interval is not a very short one and folding it into the
#: fast bucket would report a data defect as an achievement.
NEGATIVE_INTERVAL: Final[str] = "negative interval (clocks out of order)"


def duration_bucket(hours: float) -> str:
    """The bucket label for an elapsed duration in hours.

    Total: a distribution of skewed durations summarised by a mean is a number
    nobody can act on, because a mean of a skewed distribution is a point no
    observation occupies. Every measure here that has a duration to report reports
    the buckets and no mean, and this function is the one place that decides what
    a bucket is so that two measures cannot disagree.
    """
    if hours < 0:
        return NEGATIVE_INTERVAL
    if hours < HOUR_BUCKET_EDGES[0]:
        return LESS_THAN_ONE_HOUR
    lower = HOUR_BUCKET_EDGES[0]
    for edge in HOUR_BUCKET_EDGES[1:]:
        if hours < edge:
            return f"{_hours(lower)} to {_hours(edge)}"
        lower = edge
    return MORE_THAN_720_HOURS


def duration_buckets(interval_hours: Iterable[float]) -> Mapping[str, int]:
    """Bucket a set of durations, every bucket present at zero.

    Every rung appears even when nothing landed in it, for the same reason
    :mod:`tenbin.corpus.provenance` includes every anomaly kind: a distribution
    whose shape depends on the corpus is one every report has to guard against, and
    a reader comparing two runs needs the empty rungs to be there to see they were
    empty rather than to infer it from their absence.
    """
    counts = Counter(duration_bucket(hours) for hours in interval_hours)
    return freeze_counts(
        {label: counts.get(label, 0) for label in bucket_labels()},
        "duration_buckets",
    )


def bucket_labels() -> tuple[str, ...]:
    """Every duration bucket label, shortest interval first.

    A function rather than a constant so the ladder and the labels cannot drift
    apart, which is the same reason :data:`HOUR_BUCKET_EDGES` is a list of numbers
    and not a list of strings.
    """
    labels = [LESS_THAN_ONE_HOUR]
    lower = HOUR_BUCKET_EDGES[0]
    for edge in HOUR_BUCKET_EDGES[1:]:
        labels.append(f"{_hours(lower)} to {_hours(edge)}")
        lower = edge
    labels.append(MORE_THAN_720_HOURS)
    return tuple(labels)


def _hours(hours: float) -> str:
    """Render a bucket edge the way a reader would say it out loud."""
    if hours < 24:
        return f"{int(hours)}h"
    days = hours / 24
    return f"{int(days)}d"
