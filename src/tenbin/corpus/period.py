"""A declared period, and the boundaries it produces. There is no default for either.

**Nothing in either system declares a cadence, so this module refuses to invent one.**
No ticket source Tenbin reads declares a sprint length: a month is not a sprint, and a
figure computed over calendar months reads exactly like a figure computed
over sprints while describing something else. So :class:`PeriodConfig` has two required fields and no defaults at
all, and :meth:`PeriodConfig.from_text` refuses an unparseable one rather than guessing
what ``"2w"`` was meant to be. A default cadence would be the worst possible default: it
would be invisible in every rendered report, it would imply a sprint length that nobody
chose, and a retrospective is precisely the artefact whose whole claim is that its
boundaries were chosen rather than inherited.

**The anchor is a boundary, not an origin.** It is the start of one period, and every
other boundary is that one offset by whole cadences. The alternative -- anchoring at the
epoch, or at the first record in the corpus -- makes the boundaries a function of the
data, which is the defect this module exists to prevent twice over: the periods would
move every time a record arrived, so two retrospectives of the same period would not be
comparable, and a period boundary would be indistinguishable from a gap in recording. An
anchor that is declared once and stated in every output makes the boundaries a property
of the convention rather than of the corpus.

**Periods are half-open, ``[start, end)``, and that is what makes them a partition.** A
transition recorded at exactly the moment a period ends belongs to the next period and to
only one of them. With both ends closed, every boundary transition would be counted twice
and a retrospective over two consecutive periods would report more completed stays than
the corpus holds; with both ends open, each would be counted zero times. Half-open is the
only convention under which the periods tile without a seam, which is what lets a reader
add two retrospectives together and get the truth.

**A period knows its own index, and the index is what makes it reproducible.** The same
configuration and the same moment give the same :class:`Period` on any machine in any
year, because the index is a floor division of the distance from the anchor rather than
a count of periods since something happened to be first. So ``period.previous()`` is
exact rather than approximate, and a retrospective can name the period it is comparing
against without either read having to have run.

**What this module notably does not do:** it does not read anything, hold a client, or
know that ticket transitions exist. It is arithmetic over two declared values, which is
why it is testable without a backend and why a period boundary can be asserted in a test
rather than inferred from a rendered report. It does not decide whether a period contained
any activity -- that is a finding about a corpus, and it belongs to a measure, not to the
calendar. And it notably has no notion of a "current" period: :meth:`PeriodConfig.period_at`
takes an index and :meth:`period_containing` takes a moment, so choosing which period to
describe stays with the caller, who has to state it in the output either way.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

from tenbin.claims.mechanism import require_text

#: The duration grammar, as data. A tiny one on purpose: a period cadence is a team
#: convention somebody has to be able to state in a sentence ("two weeks"), so the
#: spellings accepted are the two that are unambiguous in English -- ``7d``, ``336h`` --
#: and nothing else. A parser that accepted ISO-8601 durations or ``"fortnight"`` would
#: be accepting more ways to be wrong about the one number every figure in a retrospective
#: is scoped by.
_DURATION = re.compile(r"\A(?P<count>[1-9][0-9]*)(?P<unit>[dh])\Z")

#: The units the grammar accepts, and what each one is worth. Keyed by the letter rather
#: than branched on inline so that a third unit is one entry here and a test, rather than
#: a second ``if`` somebody has to remember to update.
_UNIT_SECONDS: dict[str, int] = {"d": 86_400, "h": 3_600}


class PeriodActivity(StrEnum):
    """What a period contains, which is not the same question as whether it is empty.

    **The three-way split is the whole point, and a boolean cannot carry it.** A period
    with no ticket transitions in it is one of two entirely different facts, and they call
    for opposite responses from a reader. Either the recorder was running and nothing
    happened -- which is a finding about the work -- or there was no recorder yet, so
    there was nothing that *could* have happened -- which is a finding about the
    installation and says nothing at all about the work. Rendering both as "no activity"
    is how a retro comes to report a team's quietest sprint as a period where nothing was
    measured, which is the one confusion this whole program exists to prevent.

    ``RECORDED`` is the ordinary case and is stated as a count rather than as an absence
    of a finding, because a period with transitions in it needs no verdict -- it has a
    population, and the figures below it are about that population.
    """

    #: Transitions were recorded inside the period. The period has a population.
    RECORDED = "recorded"
    #: Transitions exist before the period began, and none fall inside it. The recorder
    #: was running; nothing moved.
    QUIET = "quiet"
    #: No transition exists at or before the period's start. There was nothing recording
    #: yet, so the period cannot be a statement about the work.
    UNRECORDED = "unrecorded"


@dataclass(frozen=True)
class Period:
    """One half-open window ``[start, end)``, and the convention that produced it.

    Holds the cadence and the anchor beside its own boundaries rather than only its index,
    so a rendered period can state *how* its boundaries were arrived at rather than only
    where they fell. That is not redundancy: "2026-09-14 to 2026-09-28" is a range, and a
    range with no cadence beside it cannot be told apart from a hand-picked window that
    happens to have those endpoints -- which is exactly the ambiguity a retrospective has
    to refuse, since a reader who cannot reproduce the boundary cannot trust the figure
    scoped to it.

    ``__post_init__`` checks that ``start`` really is the anchor plus whole cadences, which
    is what makes a ``Period`` impossible to construct by arithmetic that disagrees with the
    configuration that claims to have produced it. A hand-built period with plausible dates
    is the failure this catches: it would render identically to a derived one and would not
    be the period anybody declared.
    """

    index: int
    start: datetime
    end: datetime
    cadence: timedelta
    anchor: datetime

    def __post_init__(self) -> None:
        require_text("Period", "index", str(self.index))
        for name in ("start", "end", "anchor"):
            moment = getattr(self, name)
            if not isinstance(moment, datetime):
                raise ValueError(f"Period.{name} must be a datetime, not {moment!r}")
            if moment.tzinfo is None:
                raise ValueError(
                    f"Period.{name} must be an aware datetime, not {moment.isoformat()!r}; a "
                    "boundary nobody can place in a zone is a boundary that moves with the "
                    "machine reading it, and every figure scoped to it moves too."
                )
        if self.cadence <= timedelta(0):
            raise ValueError(
                f"Period.cadence must be positive, not {self.cadence}; a period of no length "
                "would contain no transition and would report an empty population as though "
                "the team had done nothing."
            )
        if self.end <= self.start:
            raise ValueError(
                f"Period ends at {self.end.isoformat()}, which is not after its start at "
                f"{self.start.isoformat()}; a period is half-open and must have length."
            )
        offset = self.start - self.anchor
        if offset % self.cadence != timedelta(0):
            raise ValueError(
                f"Period.start {self.start.isoformat()} is not a whole number of "
                f"{self.cadence} cadences from the anchor {self.anchor.isoformat()}, so this "
                "period was not produced by the convention it carries. A hand-picked window "
                "renders exactly like a declared one and is not one."
            )

    def contains(self, moment: datetime) -> bool:
        """Whether ``moment`` falls in this period. Half-open, so the end belongs elsewhere.

        A naive moment is refused rather than assumed to be UTC, for the reason
        :meth:`__post_init__` refuses a naive boundary: the comparison would silently
        resolve in the machine's zone and place a transition on the wrong side of a
        boundary, which for a period means counting it in the wrong retrospective.
        """
        if moment.tzinfo is None:
            raise ValueError(
                f"Period.contains needs an aware datetime, not {moment.isoformat()!r}; a naive "
                "moment compared against an aware boundary resolves in this machine's zone, "
                "which would put a transition on the wrong side of a period edge."
            )
        return self.start <= moment < self.end

    @property
    def hours(self) -> float:
        """The period's own length in hours, which is the figure every rate is over."""
        return (self.end - self.start).total_seconds() / 3600.0

    def previous(self) -> Period:
        """The period immediately before this one, from the same configuration.

        Exact rather than approximate because the index is a floor division of a distance,
        so ``index - 1`` is the adjacent period by construction and not by counting periods
        that happened to be recorded.
        """
        return Period(
            index=self.index - 1,
            start=self.start - self.cadence,
            end=self.start,
            cadence=self.cadence,
            anchor=self.anchor,
        )

    def render_text(self) -> str:
        """The window, as the short form a figure may repeat.

        **Short on purpose, and the shortness is load-bearing.** This string is embedded in
        every title, statement and denominator of a retrospective -- ten times in a report
        with three figures -- so anything beyond the boundaries and the index turns the
        document into a wall of identical text a reader learns to skip, which is the fate
        every caveat in this program exists to avoid. The cadence and the anchor belong in
        :meth:`render_declaration`, which the header prints **once**.

        The end is stated as a boundary rather than an inclusive date so the half-open
        convention is visible in the output instead of being a property only this module
        knows about. A reader who takes "the 5th to the 19th" to include both ends would
        double-count every transition on the 19th.
        """
        return f"period {self.index}, {self.start.isoformat()} to {self.end.isoformat()} (end exclusive)"

    def render_declaration(self) -> str:
        """The window *and* the convention that produced it, for the one place both belong.

        The header. A reader who wants to know how the edges were arrived at gets it once,
        above every figure scoped to them, and a figure's own text stays short enough to be
        read. The two forms are separate methods rather than a flag because a report that
        can print either in either place will eventually print the long one in a title.
        """
        return (
            f"{self.render_text()}, one of a {self.cadence} cadence anchored at "
            f"{self.anchor.isoformat()}"
        )


@dataclass(frozen=True)
class PeriodConfig:
    """A cadence and an anchor: the whole of a declared period, with nothing defaulted.

    Two required fields. The refusal to supply a third is the design, and it is worth being
    explicit about what a default would have cost: a program that guessed a fortnight would
    produce a retrospective whose every figure was scoped correctly to *something*, whose
    boundary nobody had chosen, and whose output would not say so. The failure is invisible
    in the rendered report, which is the definition of the failure this project is built to
    make impossible.

    The anchor does not have to be in the past and does not have to be a round date. It has
    to be a moment that is the start of a period, because that is the only thing a boundary
    derived from it can honestly be.
    """

    cadence: timedelta
    anchor: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.cadence, timedelta):
            raise ValueError(
                f"PeriodConfig.cadence must be a timedelta, not {self.cadence!r}; parse it "
                "with PeriodConfig.from_text so the spelling somebody declared is the one "
                "that is used."
            )
        if self.cadence <= timedelta(0):
            raise ValueError(
                f"PeriodConfig.cadence must be positive, not {self.cadence}; see the class "
                "docstring for what a zero-length period would report."
            )
        if not isinstance(self.anchor, datetime):
            raise ValueError(f"PeriodConfig.anchor must be a datetime, not {self.anchor!r}")
        if self.anchor.tzinfo is None:
            raise ValueError(
                f"PeriodConfig.anchor must be an aware datetime, not {self.anchor.isoformat()!r}; "
                "an anchor in no zone makes every boundary derived from it a function of the "
                "machine doing the reading."
            )

    @classmethod
    def from_text(cls, cadence: str, anchor: str) -> PeriodConfig:
        """Build a configuration from the two strings a person or an env var supplies.

        Both are required and neither is defaulted, so a caller cannot reach a period
        without having stated a cadence. The refusals name what was accepted, because a
        message saying only "invalid duration" leaves the reader to guess whether the
        problem was the number, the unit, or a unit this program does not implement.
        """
        require_text("PeriodConfig.from_text", "cadence", cadence)
        require_text("PeriodConfig.from_text", "anchor", anchor)

        match = _DURATION.match(cadence.strip())
        if match is None:
            raise ValueError(
                f"a period cadence is written as a whole number of days or hours -- 7d, 336h -- "
                f"and {cadence.strip()!r} is not one of those. Nothing here is defaulted, "
                "because a guessed cadence produces a retrospective whose boundaries nobody "
                "chose and whose output does not say so."
            )
        seconds = int(match.group("count")) * _UNIT_SECONDS[match.group("unit")]

        moment_text = anchor.strip()
        try:
            moment = datetime.fromisoformat(moment_text)
        except ValueError as exc:
            raise ValueError(
                f"a period anchor is an ISO-8601 moment with a zone -- 2026-01-05T00:00:00Z -- "
                f"and {moment_text!r} is not one ({exc})."
            ) from exc
        if moment.tzinfo is None:
            raise ValueError(
                f"a period anchor must carry a zone, and {moment_text!r} does not. An anchor in "
                "no zone would put every boundary in the machine's zone at read time, so two "
                "operators would compute different periods from the same declaration."
            )
        return cls(cadence=timedelta(seconds=seconds), anchor=moment)

    def index_of(self, moment: datetime) -> int:
        """Which period ``moment`` falls in, counted from the anchor.

        Floor division rather than truncation, so a moment before the anchor lands in a
        negative-indexed period instead of period zero. That matters because the anchor is
        a boundary and not necessarily the first thing ever recorded: a retrospective of
        the period before it is a legitimate thing to want, and truncating would silently
        answer with the wrong period instead of a negative index.
        """
        if moment.tzinfo is None:
            raise ValueError(
                f"PeriodConfig.index_of needs an aware datetime, not {moment.isoformat()!r}"
            )
        return (moment - self.anchor) // self.cadence

    def period_at(self, index: int) -> Period:
        """The period with this index, exactly."""
        return Period(
            index=index,
            start=self.anchor + index * self.cadence,
            end=self.anchor + (index + 1) * self.cadence,
            cadence=self.cadence,
            anchor=self.anchor,
        )

    def period_containing(self, moment: datetime) -> Period:
        """The period ``moment`` falls in.

        The convenience over :meth:`index_of` and :meth:`period_at`, and the one a caller
        wants when the question is "the period we are in". It still requires the moment to
        be supplied rather than reading a clock, so a report is reproducible: the caller
        decides what "now" means for this run and states it in the output.
        """
        return self.period_at(self.index_of(moment))

    def render_text(self) -> str:
        """The convention as one line, for a report that has to state where its edges came from."""
        return f"{self.cadence} cadence anchored at {self.anchor.isoformat()}"
