"""The corpus the numbers below are about, and the report that says so before them.

A report's first content is not a summary of what was found; it is the read the
findings came from. A number computed over four hundred documents looks exactly like
a number computed over four thousand, and the difference is invisible in the figure
itself -- which is why the header exists rather than the figure having a footnote. So
:class:`CorpusHeader` carries the collection, the moment, the two independent counts
and the snapshot's own verdict, and every renderer puts it first. A reader who has
read the header knows whether any number below it means what they would otherwise
assume.

**The header is copied out of the snapshot, never transcribed into one.**
:meth:`CorpusHeader.from_snapshot` is the way a header is built in this program, and
that is not a convenience: a header is five numbers and a moment, and a caller who
assembled it by hand would be copying ``store_total`` into ``enumerated`` on a
tired afternoon with no test telling them. The factory removes the transcription
rather than documenting it. What remains possible -- a header written by hand -- is
not thereby untrustworthy: :meth:`CorpusHeader.__post_init__` refuses the one
combination that cannot come out of a real snapshot, a truncation whose arithmetic
disagrees with the counts it arrived with.

**A truncated read states what it missed and where it stopped.** Both, because the
first without the second is a warning and the second without the first is a
footnote. "15,000 records were not reached" is the size of the hole and is what makes
every count below an undercount; "the walk having stopped at offset 10,000" is the
number somebody needs in order to *continue* the enumeration rather than start again.
The corpus header repeats what each figure says about its own read, and that
redundancy is deliberate rather than accidental: the header is what a reader looks up
before trusting a number, and the sentence under the number is what a reader who did
not look up the header still sees.

**The header states the read's window in time as well as its size in records,
because those are not the same fact.** Four thousand documents reads identically over
three weeks and over three years, and a rate computed over them reads identically
too. A corpus that began when the webhook was installed three weeks ago is not a
clean history, it is a three-week history, and the two produce the same figures. The
window therefore sits on the enumeration line rather than on a line of its own: it
answers the same question -- over what, exactly, was this read taken -- and the
age of the newest record rides with it because that is the half a reader cannot
derive from the count.

``window`` is optional rather than required for the reason ``truncation`` is: a header
written by hand is possible, and a hand-written header has no window to state, so the
rendering says the window is unknown rather than inventing one. Every header this
program builds comes from :meth:`from_snapshot` and therefore has one.

**``period`` says which declared window a scoped read was cut to, and it is the field
that makes a retrospective's figures add up.** A count of four hundred transitions means
nothing without the fortnight it covers, and there is a specific trap here that the
``window`` field does not catch: a period-scoped read derives its window from the
transitions it happened to contain, so a quiet period's window is *narrower* than the
period it was cut to. Two retrospectives side by side would then appear to describe
populations of different lengths when they describe windows of equal length and records
of unequal count -- so the declared period rides beside the observed window rather than
replacing it, and the two are different facts about the same read.

**What this module notably does not do:** it does not recompute completeness. The
header reports the snapshot's own verdict and adds a plain-language reading of it; it
never derives one, because a layer that second-guessed the walk would be a second
implementation of the rule that made the walk safe. It does not recompute the window
either, for the same reason, and it does not hold the records, and it does not hold
the figures -- a header over ten thousand documents is six fields, and a report that
kept the corpus would be a second copy of the snapshot with a different lifetime. Nor
does it decide the order of sections: :class:`Report` is the container, and
:mod:`tenbin.report.build` decides what goes in it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from tenbin.claims.mechanism import require_text
from tenbin.corpus.period import Period
from tenbin.corpus.snapshot import Completeness, Snapshot, Truncation
from tenbin.corpus.window import CorpusWindow
from tenbin.report.base import Section
from tenbin.report.ordering import Part, PartKind

#: How each of the snapshot's four verdicts reads in a report, in prose rather than
#: as an enum name. A machine can match the enum; a reader has to be told what
#: ``offset_cap`` means, and the one thing standing between a number and a reader who
#: assumes it covers everything is that sentence existing at all. Keyed by the
#: snapshot's own vocabulary so that adding a completeness value without a reading
#: here is a visible omission.
_COMPLETENESS_READINGS: dict[Completeness, str] = {
    Completeness.COMPLETE: "the store was walked to its end and its own count agreed.",
    Completeness.OFFSET_CAP: (
        "the store's document offset ceiling was reached, so what follows is a prefix "
        "of the corpus and every count in this report is an undercount."
    ),
    Completeness.STORE_TOTAL_EXCEEDED: (
        "the store's own count disagreed with what it served, so the corpus has no "
        "agreed size and nothing that depends on the count can be trusted."
    ),
    Completeness.STORE_ERROR: (
        "the walk did not finish, so this report covers whatever had been read when it "
        "stopped and every count in it is a lower bound."
    ),
}


@dataclass(frozen=True)
class CorpusHeader:
    """The read a report is about: where it came from, how much arrived, over what span, and whether it is whole.

    Built by :meth:`from_snapshot` in every case that matters, and refused when a
    truncation contradicts the counts it came with -- because a hand-written header is
    the one way these values can come to disagree with the walk they describe, and a
    disagreement there invalidates every number underneath rather than just the
    header.

    ``window`` defaults to ``None`` for the same reason ``truncation`` does, and the
    rendering of a missing window is the unknown-window sentence rather than an absent
    line: a header that says nothing about its span reads like a header that was asked
    and had nothing to say, which is not the same as one that was never asked.
    """

    collection: str
    read_at: datetime
    enumerated: int
    store_total: int
    completeness: Completeness
    truncation: Truncation | None = None
    window: CorpusWindow | None = None
    #: The declared window this read was cut to, when it was cut to one.
    #:
    #: **Optional, and last, for the same reason ``truncation`` is optional.** A read of a
    #: whole collection has no period, and a header that demanded one would force every
    #: caller to invent an edge for a read that was not scoped to an edge. What is *not*
    #: optional is that a header which has one says so: the part is emitted whenever
    #: ``period`` is set, so a retrospective's boundaries travel in the header rather than
    #: being a fact some reader has to notice is missing from the figure they are reading.
    #:
    #: It sits between the read moment and the enumeration because those are the three
    #: questions in order -- what was read, over what window, how much of it arrived -- and
    #: putting the period after the count would let a reader weigh four hundred
    #: transitions as a population before learning which fortnight they are a population of.
    period: Period | None = None

    def __post_init__(self) -> None:
        require_text("CorpusHeader", "collection", self.collection)
        if not isinstance(self.completeness, Completeness):
            raise ValueError(
                f"CorpusHeader.completeness must be a Completeness, not {self.completeness!r}; "
                "a bare string would render the same as a verdict the walk actually reached."
            )
        truncation = self.truncation
        if truncation is not None:
            missing = max(0, self.store_total - self.enumerated)
            if truncation.records_missing != missing:
                raise ValueError(
                    f"CorpusHeader.truncation says {truncation.records_missing:,} records were "
                    f"missing while the counts it arrived with leave {missing:,}; a truncation "
                    "that does not follow from the two counts is a number nobody read."
                )

    @classmethod
    def from_snapshot(cls, snapshot: Snapshot, *, period: Period | None = None) -> CorpusHeader:
        """Copy the read out of a snapshot, so no value has to be written down twice.

        Takes the :class:`~tenbin.corpus.snapshot.Snapshot` **protocol** rather than
        :class:`~tenbin.corpus.snapshot.CorpusSnapshot`, and that is what lets a second
        source become a report at all. Every attribute read below -- ``collection``,
        ``read_at``, ``enumerated``, ``store_total``, ``completeness``, ``truncation``,
        ``window`` -- is a member of the protocol, and the concrete snapshot type was
        never consulted for any of them. It is a read-only header: it holds no records, so
        there is nothing here that would have to be the knowledge store's own shape.

        The counts are copied as they are rather than reconciled: a snapshot whose
        ``store_total`` disagrees with what it enumerated is a finding the snapshot
        layer already made, and a header that quietly fixed the disagreement would
        report a corpus rather than the one that was walked. The window is copied for
        the same reason and is *not* recomputed here -- it is the snapshot's own,
        computed once from the records it holds, so a header cannot describe a span
        that disagrees with the read it was built from.

        ``period`` is the one field that is not on the snapshot, and it is a keyword
        argument rather than something a caller assigns afterwards for that reason: a
        header built here is the header, and a period attached by a later line of code
        is a second statement about the same document that nothing checks against the
        one above it. ``None`` is the default because most reads are not cut to a period.
        """
        return cls(
            collection=snapshot.collection,
            read_at=snapshot.read_at,
            enumerated=snapshot.enumerated,
            store_total=snapshot.store_total,
            completeness=snapshot.completeness,
            truncation=snapshot.truncation,
            window=snapshot.window,
            period=period,
        )

    def parts(self) -> tuple[Part, ...]:
        """This header as ordered parts, in the order a reader needs them.

        The same type :func:`~tenbin.report.ordering.ordered_parts` returns for a
        section, and for the same reason: the header is content rather than a
        preamble, so it is held to the same rule that the caveats in a section cannot
        be reordered away. Collection first, then when, then how much and over what
        span, then whether all of it arrived -- and the truncation last, because it is
        the sentence that qualifies everything above it and a reader who reads only
        the first line of a report should still be stopped by it.
        """
        return (
            Part(PartKind.collection, self.collection),
            Part(PartKind.read_at, self.read_at.isoformat()),
            *self._period_parts(),
            Part(PartKind.enumeration, self._enumeration_text()),
            Part(
                PartKind.completeness,
                f"{self.completeness.value} — {_COMPLETENESS_READINGS[self.completeness]}",
            ),
            *self._truncation_parts(),
        )

    def _period_parts(self) -> tuple[Part, ...]:
        """The declared window, or nothing at all when this read was not cut to one.

        An empty tuple rather than a "not applicable" line, because unlike the truncation
        this is not a qualification of the numbers -- it is the scope they were taken over,
        and a whole-collection read genuinely has none. Emitting a placeholder would put a
        sentence about the absence of a boundary in the place a reader looks for the
        boundary.
        """
        if self.period is None:
            return ()
        return (Part(PartKind.period, self.period.render_declaration()),)

    def _enumeration_text(self) -> str:
        """How much was read, and over what window -- the two halves of one question.

        On one part rather than two, and the reason is the report rather than the
        header. A count and a span describe the same population from two directions,
        so a reader who reads one and skips the other has half of what they need; a
        separate line is a separate thing to skip. The window goes second, after the
        count and before the verdict, because it qualifies the count rather than
        qualifying the verdict.

        A header with no window renders the unknown-window sentence with a read it
        did not have rather than nothing at all, for the reason
        :attr:`window` is optional at all.
        """
        counted = f"{self.enumerated:,} of {self.store_total:,} documents"
        window = self.window
        if window is None:
            window = CorpusWindow(earliest=None, latest=None, unreadable_timestamps=0)
            return f"{counted}\n{window.render_text()}"
        age = window.age_text(self.read_at)
        span = f"{counted}\n{window.render_text()}"
        return f"{span} {age}" if age else span

    def _truncation_parts(self) -> tuple[Part, ...]:
        """The truncation sentence, or nothing at all when the read was whole.

        A header with no truncation part is not a header with a missing one: there is
        a completeness part saying the read was whole, and a second part saying the
        read stopped would be the contradiction.
        """
        truncation = self.truncation
        if truncation is None:
            return ()
        return (
            Part(
                PartKind.truncation,
                f"{truncation.records_missing:,} records were not reached; enumeration "
                f"stopped at offset {truncation.offset_reached:,} and would resume there",
            ),
        )


@dataclass(frozen=True)
class Report:
    """One read of a corpus, and everything this program will say about it.

    Sections are held in the order they should be read, and the constructor refuses a
    report that says the same thing twice. Two sections for one claim is not a
    summary with repetition -- it is two figures for one claim, and a reader has no
    way to tell which one this program meant, which is the position the gate exists to
    put a reader in never. A report may legitimately hold no sections: a registry of
    nothing and a measure set of nothing is a report about an empty corpus, and that is
    a fact rather than a defect.
    """

    corpus: CorpusHeader
    sections: tuple[Section, ...]

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for section in self.sections:
            if section.slug in seen:
                raise ValueError(
                    f"a report holds two sections for {section.slug!r}; one claim rendered "
                    "twice is a contradiction, and a reader cannot tell which figure is the "
                    "program's answer."
                )
            seen.add(section.slug)
