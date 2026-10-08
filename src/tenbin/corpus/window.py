"""The corpus's window in time, because a rate over an unknown span is not a rate.

Every rate this program publishes states its population in records. Not one states
it in time. A corpus three weeks old reports how often a stated reason was revised as
a rate over three chains and says nothing about the three weeks, so a reader cannot
tell *no reasons were revised* from *this corpus was created last Tuesday* -- and a
corpus installed last Tuesday is a completely ordinary state for a knowledge store,
because Kojutsu writes on webhook events and the corpus therefore begins whenever
the webhook was installed. A population stated as a count of records is still
unstated if nobody says the records span three weeks or three years.

**The two ends, the span, and the age of the newest record.** :class:`CorpusWindow`
holds the earliest and the latest moment any record in a read could be placed at, the
span between them, and the count of records that carried no moment at all. The age is
not stored, because a stored age is a claim about *now* held by an object that is a
statement about a past read: :meth:`CorpusWindow.age_at` takes the moment to measure
against, so a caller has to say which moment it means and a read model restores the
age of the read it was built from rather than of the machine rendering it.

**An unknown window and a zero-length window are different facts, and both are
expressible.** Every field is optional-with-``None`` rather than defaulted, so a
window that could not be read and a window whose two ends coincide are not the same
object. That distinction is load-bearing twice over: a rate over an unknown span
cannot be checked against a duration at all, while a rate over a window of no length
can be checked and is almost certainly wrong. Collapsing the two -- by defaulting
``earliest`` and ``latest`` to the read's moment, or by reporting a span of zero for a
corpus whose timestamps would not parse -- is how a report ends up stating a duration
nobody measured.

**A record with no readable moment is counted, not dropped.** A timestamp that will
not parse yields ``None`` in the reader, as everywhere else in this package, and one
record with a bad field is still a record. What is not permitted is to let it vanish
from the denominator of the *window* without appearing in
:attr:`CorpusWindow.unreadable_timestamps`, because a reader who is told the corpus
spans nine months and is not told that four records could not be placed in it has been
given half the evidence and no indication that it is half.

**This module notably does not read a clock.** Nothing here calls ``now()``: a window
is a fact about a read, and the moment it is measured against is always an argument.
That is what makes a figure computed from a read model report the window of the read
that produced the model rather than the window of the machine rendering it, and it is
the same rule :mod:`tenbin.readmodel` follows about everything else it restores.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Final

from tenbin.corpus.record import Record

#: The units a duration is rendered in, largest first. At most two are used, and the
#: reason is legibility rather than precision: "29 days, 2 hours" is a sentence a
#: reader can weigh against what they expected, where "2,517,600 seconds" is a number
#: they have to convert, and a reader who has to convert is a reader who skips.
DURATION_UNITS: Final[tuple[tuple[str, int], ...]] = (
    ("day", 86_400),
    ("hour", 3_600),
    ("minute", 60),
    ("second", 1),
)

#: A duration of nothing at all, which is a real answer and not a rounding of one. It
#: is worded rather than rendered as ``0s`` because a zero in a report reads as a
#: count of something that did not happen, and the fact here is that no time passed
#: between the two ends of a window.
NO_TIME_AT_ALL: Final[str] = "no time at all"

#: A duration shorter than the renderer's resolution, stated rather than rounded away.
UNDER_A_SECOND: Final[str] = "under a second"


def record_moment(record: Record) -> datetime | None:
    """The instant a record can be placed at, in UTC, or ``None`` if it names none.

    The preference is the one :attr:`tenbin.corpus.record.Record.month` uses and for
    the same reason: the event the record is *about* first -- a review's submission,
    a comment's creation, a close -- then the capture, then a rationale's
    declaration, and the document's own update time last, because a document's update
    time moves on every push and is a fact about the write rather than about what
    happened.

    **The chain is written out here rather than read off the record, and that
    duplication is deliberate and has a home.** :attr:`Record.month` holds the same
    preference, and neither of the two should own it: a shared accessor belongs on
    :class:`~tenbin.corpus.record.Record`, which is a package this ticket does not
    hold, so until that edit is made this is one definition here and one there, both
    written down and both naming the other. A refactor that changes one of them has to
    change both, which is a louder failure than a silent divergence -- the same
    argument :func:`tenbin.measures.filtering.named_repository` makes about the
    private twin it deliberately does not reach for.

    Returned in UTC even when the document carried an offset, so that two ends
    rendered beside each other are in the same zone and a reader is not asked to
    compare two clocks.
    """
    moment = record.answered_at or record.captured_at or record.declared_at or record.updated_at
    return None if moment is None else moment.astimezone(UTC)


@dataclass(frozen=True)
class CorpusWindow:
    """The two ends of a read's history, how far apart they are, and what could not be read.

    Frozen for the reason a :class:`~tenbin.corpus.snapshot.CorpusSnapshot` is: this
    is a statement about a past read, and a window that could be edited afterwards
    would be a statement about something that never happened. Every field is required
    and every one is optional in its *value* -- the constructor refuses a half-read
    window, an inverted one, a naive moment and a negative count, because each of those
    is a contradiction rather than an absence and the two must not be spelled the same
    way in the type.

    :attr:`span` is derived rather than stored, so a caller cannot hold a window whose
    span disagrees with its two ends -- the same transcription hazard
    :class:`~tenbin.report.document.CorpusHeader` refuses a truncation for.
    """

    #: The earliest moment any record in the read could be placed at, or ``None`` when
    #: not one could be. The value is ``None`` for a read that enumerated no records and
    #: for a read whose every timestamp would not parse, and the two render differently.
    earliest: datetime | None
    #: The latest moment any record in the read could be placed at, on the same terms as
    #: :attr:`earliest`. Both ends or neither: a window with one end is a half-read
    #: object, and the honest handling of one is to refuse it rather than to guess the
    #: missing end from the read's own moment.
    latest: datetime | None
    #: How many records in the read named no moment this reader could place. A count
    #: and not a proportion, because the count is what a reader can add up against a
    #: denominator, and a proportion here would be a second rate to distrust.
    unreadable_timestamps: int

    def __post_init__(self) -> None:
        if isinstance(self.unreadable_timestamps, bool) or not isinstance(
            self.unreadable_timestamps, int
        ):
            raise ValueError(
                f"CorpusWindow.unreadable_timestamps must be a count, not {self.unreadable_timestamps!r}"
            )
        if self.unreadable_timestamps < 0:
            raise ValueError(
                "CorpusWindow.unreadable_timestamps must not be negative; "
                f"{self.unreadable_timestamps} records did not fail to be read"
            )
        if (self.earliest is None) != (self.latest is None):
            raise ValueError(
                "CorpusWindow.earliest and CorpusWindow.latest must both be readable or "
                "neither; a window with one end is a half-read object, and guessing the "
                "other from the read's moment would state a duration nobody measured."
            )
        for name, moment in (("earliest", self.earliest), ("latest", self.latest)):
            if moment is None:
                continue
            if not isinstance(moment, datetime) or moment.tzinfo is None:
                raise ValueError(
                    f"CorpusWindow.{name} must be an aware datetime, not {moment!r}; a moment "
                    "that cannot be placed in time cannot bound a window."
                )
        if self.earliest is not None and self.latest is not None and self.earliest > self.latest:
            raise ValueError(
                f"CorpusWindow.earliest ({self.earliest.isoformat()}) is after its own "
                f"latest ({self.latest.isoformat()}); a window that ends before it begins "
                "is not a shorter window, it is a contradiction."
            )

    @classmethod
    def from_records(cls, records: Iterable[Record]) -> CorpusWindow:
        """The window of whatever these records place themselves in, and no more.

        One pass, and the reason it is one pass is that the answer is a pair of
        extrema: the earliest moment anybody stated and the latest, plus the number of
        records that stated none. Building it here rather than in each measure is the
        same argument :meth:`tenbin.corpus.snapshot.CorpusSnapshot.window` makes -- a
        window twenty measures computed independently is twenty windows that can
        disagree, and a corpus that is reported as spanning nine months in one section
        and eleven in another is a report about nothing.

        A record that names no moment is counted in
        :attr:`unreadable_timestamps` rather than skipped silently, because a record
        skipped silently is a record that left the denominator without appearing in an
        exclusion -- the exact defect
        :class:`~tenbin.measures.base.DistributionFigure` exists to make impossible,
        reached here in a different costume.
        """
        earliest: datetime | None = None
        latest: datetime | None = None
        unreadable = 0
        for record in records:
            moment = record_moment(record)
            if moment is None:
                unreadable += 1
                continue
            if earliest is None or moment < earliest:
                earliest = moment
            if latest is None or moment > latest:
                latest = moment
        return cls(earliest=earliest, latest=latest, unreadable_timestamps=unreadable)

    @property
    def is_known(self) -> bool:
        """Whether both ends of the window could be read at all."""
        return self.earliest is not None and self.latest is not None

    @property
    def span(self) -> timedelta | None:
        """The distance between the two ends, or ``None`` when the window is unknown.

        ``None`` rather than a zero of :class:`~datetime.timedelta`, for the reason the
        whole type exists: a corpus with no readable timestamps does not span no time, it
        spans an unknown amount of it. A rate over a zero-length window can at least be
        checked against a duration -- and found wanting, which is a finding -- whereas a
        rate over an unknown one cannot be checked at all, and reporting it as a rate over
        nothing is how a number nobody measured gets published.
        """
        if not self.is_known:
            return None
        assert self.earliest is not None and self.latest is not None
        return self.latest - self.earliest

    def age_at(self, moment: datetime) -> timedelta | None:
        """How old the newest record was at ``moment``, or ``None`` if nothing is placed.

        An argument rather than a clock for the reason this module holds no clock: the
        same window is old against one read and young against another, and a stored age
        would be whichever of the two the writer happened to run. A caller holding a
        read model passes the model's own ``read_at`` and gets the age of the read that
        produced it.

        The result is signed, and a negative one is not silenced: a record dated after
        the read that found it is clock skew between two writers, and rendering it as a
        large positive age would report a data defect as a capture path that stopped.
        """
        if self.latest is None:
            return None
        return moment - self.latest

    def render_text(self) -> str:
        """The window as a sentence: where it starts, where it ends, how wide it is.

        Used by the corpus header and by a subset measure's own denominator, and
        written once so the two cannot describe the same window in different words. The
        unknown cases are stated rather than rendered as an absence, because a reader
        who is handed no window at all cannot tell that the program asked for one, and a
        header line that is simply missing is a header line that looks like a decision
        rather than a limitation.
        """
        if not self.is_known:
            return self._render_unknown()
        assert self.earliest is not None and self.latest is not None
        span = self.span
        assert span is not None
        text = (
            f"Records span {self.earliest.isoformat()} to {self.latest.isoformat()}, "
            f"a window of {render_duration(span)}"
        )
        unreadable = self.unreadable_timestamps
        if unreadable:
            noun = "record carried" if unreadable == 1 else "records carried"
            text += (
                f"; {unreadable:,} {noun} no moment this reader could place, so these are "
                "the ends of the ones that could be read"
            )
        return f"{text}."

    def age_text(self, moment: datetime) -> str:
        """The age of the newest record against ``moment``, or nothing when unknown.

        Empty on a window that could not be read, and that is a documented omission
        rather than a missing part: the unknown-window sentence has already said the
        one thing that is true, and a second sentence about the age of a record that
        cannot be placed in time would be a caveat about a caveat.
        """
        age = self.age_at(moment)
        if age is None:
            return ""
        if age.total_seconds() < 0:
            return (
                "The newest of them is dated "
                f"{render_duration(-age)} after this read, which is clock skew between two "
                "writers rather than a corpus running ahead of its own read."
            )
        if age.total_seconds() == 0:
            return "The newest of them is stamped at the same instant as this read."
        return f"The newest of them is {render_duration(age)} before this read."

    def _render_unknown(self) -> str:
        """The window that could not be read, named as the condition it is.

        The empty-read case is separated from the unreadable-timestamps case because
        they are different facts with different readers: one is a store that was
        reached and holds nothing, the other is a store whose documents arrived without
        a moment this reader could place. Reporting the first as the second would send
        an operator to look for a broken timestamp reader on a collection that is simply
        empty.
        """
        unreadable = self.unreadable_timestamps
        if unreadable == 0:
            return (
                "Window unknown: this read enumerated no records, so it covers no span at "
                "all and no rate over it can be dated."
            )
        noun = "record" if unreadable == 1 else "records"
        return (
            f"Window unknown: none of the {unreadable:,} {noun} in this read carried a moment "
            "this reader could place, so the span it covers is not established and a rate over "
            "it is a rate over an unknown duration."
        )


def render_duration(span: timedelta) -> str:
    """A duration in the two largest units somebody would say out loud.

    Total, and a number rather than a rounded one, because the whole point of stating
    a window is that a reader can weigh it against what they expected -- "29 days,
    2 hours" and "29 days" are different claims about the same corpus and only the
    first of them is true. At most two units, for the reason
    :data:`DURATION_UNITS` is a ladder rather than a single scale: a duration rendered
    to the second invites a reader to compare two windows that differ by a bucket of
    noise.
    """
    seconds = abs(span).total_seconds()
    if seconds == 0:
        return NO_TIME_AT_ALL
    if seconds < 1:
        return UNDER_A_SECOND
    remaining = int(seconds)
    parts: list[str] = []
    for name, size in DURATION_UNITS:
        count, remaining = divmod(remaining, size)
        if count == 0:
            continue
        parts.append(f"{count:,} {name if count == 1 else name + 's'}")
        if len(parts) == 2:
            break
    return ", ".join(parts)
