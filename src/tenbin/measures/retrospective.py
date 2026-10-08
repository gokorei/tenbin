"""What a declared period contains, and the two figures a retrospective is allowed to lead with.

A retrospective is a report over a *window somebody chose*, and that makes it different
from every other report in this program in one specific way: its population is bounded on
both sides by a convention rather than by the extent of the corpus. Three consequences run
through this module, and each one is a refusal rather than a number.

**A period with nothing in it is two different findings, and this module will not render
them as one.** :class:`PeriodActivityMeasure` asks the backend whether any transition
exists *at or before* the period's start, and the answer is what separates "the recorder
was running and no ticket moved" from "there was no recorder yet, so nothing could have
moved". The second is not a fact about the work at all, and a retrospective that rendered
it as a quiet sprint would be reporting an installation date as a team outcome. The
distinction is a property of the corpus rather than of the arithmetic, so it is a
:class:`~tenbin.measures.base.Finding`, which :mod:`tenbin.report.build` hoists to the
front of a report as a matter of discipline -- a reader should meet "this period was never
recorded" before any figure that might otherwise be read as describing it.

**Time-to-terminal is a distribution and a lower bound, and it is not lead time.** A
transition carries four fields and no creation moment, so the
interval this module measures cannot begin where lead time begins: it begins at the
earliest transition *this source* holds for a ticket, which is a lower bound on when that
ticket entered the world. Two biases follow and both are stated in the claim rather than
left for the reader to derive. The intervals are too short, because everything before the
first recorded transition is missing. And the population is tickets whose first recorded
transition falls inside the period, so every ticket already in flight when the period
opened is absent -- which is a one-directional bias, and a one-directional bias has to be
named or it reads as a level.

**What this module notably does not do, and the two absences are load-bearing.** It does
not compute committed-versus-delivered, and it does not compute anything per principal.
Both are refusals this module publishes as values rather than as omissions:
:func:`committed_versus_delivered_refusal` because no supported source validates
estimates against delivery, so a comparison built over them would be a
comparison over nothing rendered as a shortfall;
:func:`who_carried_what_refusal` because a transition records no principal -- the
adapters drop whatever principal the wire carried -- so "who
carried what" is not a figure this source cannot support -- it is one it has no field for.
That is a stronger statement than a gate refusal, and it is still rendered as a refusal
rather than dropped, because a reader who asked for it and got nothing has to be able to
tell that from a reader who asked for it and got a number.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final

from tenbin.claims.mechanism import Granularity
from tenbin.claims.model import Claim, ClaimKind, Denominator
from tenbin.claims.refusals import ComparativeRefusalError, Refusal
from tenbin.corpus.period import Period, PeriodActivity
from tenbin.corpus.snapshot import Snapshot
from tenbin.corpus.transition import Transition
from tenbin.measures.base import (
    CountFigure,
    DistributionFigure,
    Figure,
    Finding,
    Rate,
    RateFigure,
    duration_buckets,
)

#: The slug of the figure that says what the period contains, and the three refusals'
#: slugs. Named rather than written inline so the report, the tests and a reader looking
#: up a section all key on the same string.
PERIOD_ACTIVITY_SLUG: Final[str] = "period-activity"
THROUGHPUT_SLUG: Final[str] = "period-throughput"
TIME_TO_TERMINAL_SLUG: Final[str] = "time-to-terminal"
CARRIED_BY_SLUG: Final[str] = "who-carried-what"
COMMITTED_VERSUS_DELIVERED_SLUG: Final[str] = "committed-versus-delivered"

#: The terminal state a project is assumed to have when the caller names none. A default,
#: and the only one in this module, so it is named, stated in the claim of every figure
#: that depends on it, and overridable: a ticket system's lifecycle is per-project
#: configuration, so the honest default is the one state every lifecycle in this system
#: has rather than a guess at which states a particular team considers finished.
DEFAULT_TERMINAL_STATE: Final[str] = "done"

#: A ticket whose only recorded transition is the one that finished it has no interval to
#: measure. Excluded under its own key rather than rendered as zero: a zero-hour ticket is
#: a claim that work took no time, and what actually happened is that this source never saw
#: the ticket before it was finished.
NO_EARLIER_TRANSITION: Final[str] = "only the finishing transition was ever recorded"

#: A ticket seen inside the period that never reached a terminal state inside it. Counted
#: and not bucketed, for the reason :data:`tenbin.measures.dwell.OPEN_ENDED` gives: a stay
#: with no end is not a short one, and folding it into a top bucket would report that work
#: piles up at the end of a period, which is a statement about the measurement and not
#: about the work.
STILL_OPEN: Final[str] = "seen in this period, not finished inside it"

#: Why lead time is absent, said once and reused, because a retrospective that omits a
#: figure without saying so leaves a reader to assume the corpus could not support it --
#: which is a different and wrong conclusion.
NO_CREATION_MOMENT: Final[str] = (
    "Lead time cannot be measured from this source: a status transition records that a "
    "ticket's status changed and when, and the backend holds no moment at which the ticket "
    "was created. The interval below starts at the earliest transition this source holds "
    "for the ticket, which is a lower bound on when it entered the world rather than the "
    "moment it entered it."
)


def terminal_states(declared: Iterable[str] | None) -> tuple[str, ...]:
    """The states counted as finished, from what the caller declared.

    Deduplicated and sorted rather than kept in the order given, so two runs declaring the
    same set in a different order produce byte-identical claims -- which is what lets an
    archived retrospective be compared against a later one without the comparison turning
    on argument order. An empty declaration falls back to
    :data:`DEFAULT_TERMINAL_STATE` rather than raising, because "the caller named no
    terminal state" and "the caller named a set with nothing in it" are the same request
    and the fallback states which state it assumed in every figure's claim.
    """
    if declared is None:
        return (DEFAULT_TERMINAL_STATE,)
    named = tuple(sorted({state.strip() for state in declared if state.strip()}))
    return named or (DEFAULT_TERMINAL_STATE,)


def _is_terminal(transition: Transition, terminal: Sequence[str]) -> bool:
    return transition.to_state in terminal


@dataclass(frozen=True)
class PeriodActivityMeasure:
    """What this period contains, distinguishing a quiet period from an unrecorded one.

    ``activity`` and ``recorded_before`` are both the caller's, and both are required,
    because neither can be derived from the period-scoped read alone: an empty scoped read
    is the ambiguous case this measure exists to resolve, and resolving it needs a second
    fact about the source outside the window. ``recorded_before`` is a count rather than a
    boolean because a reader who wants to know how much earlier history exists is asking a
    different and answerable question, and the number is already in hand.

    A ``RECORDED`` period renders a count rather than a finding, because a period with a
    population needs no verdict -- it has figures, and they are about those figures. The
    two empty cases render findings, and findings are hoisted to the front of a report, so
    a reader meets "this period was never recorded" before anything that could be misread
    as describing it.
    """

    period: Period
    activity: PeriodActivity
    recorded_before: int

    slug = PERIOD_ACTIVITY_SLUG

    def compute(self, snapshot: Snapshot) -> Figure:
        observed = _in_period(snapshot, self.period)
        if self.activity is PeriodActivity.RECORDED:
            return CountFigure(
                claim=self.claim(snapshot),
                snapshot=snapshot,
                title=f"Status transitions recorded in {self.period.render_text()}",
                value=len(observed),
            )
        if self.activity is PeriodActivity.UNRECORDED:
            return self._unrecorded(snapshot, observed)
        return self._quiet(snapshot)

    def claim(self, snapshot: Snapshot) -> Claim:
        observed = _in_period(snapshot, self.period)
        return Claim(
            slug=self.slug,
            statement=(
                f"{self.period.render_text()} contains {len(observed):,} status transitions"
                + (
                    f", and {self.recorded_before:,} were recorded before it began."
                    if self.recorded_before
                    else ", and no transition at all was recorded before it began."
                )
            ),
            does_not_mean=(
                "That the period's boundaries were chosen by this program. They come from a "
                "declared cadence and anchor, which is a team convention rather than a fact "
                "about the work, and a figure scoped to a convention nobody chose would read "
                "exactly like this one. Nor does the count say anything was completed: it "
                "counts status changes, and a ticket can change status six times without "
                "finishing."
            ),
            falsifier=(
                "A read in which the backend reports transitions inside this window that this "
                "walk did not receive, or reports none at or before it that it does hold -- "
                "either would mean the period's own edges and the source's view of them "
                "disagree, and every figure scoped to this window would be scoped wrongly."
            ),
            denominator=Denominator(
                description=(
                    f"status transitions read from {snapshot.collection} for "
                    f"{self.period.render_text()}"
                ),
                size=max(1, len(observed)),
                size_noun="transitions",
            ),
            kind=ClaimKind.descriptive,
            granularity=Granularity.corpus,
        )

    def _unrecorded(self, snapshot: Snapshot, observed: Sequence[Transition]) -> Finding:
        """A period with no transitions in it and none before it: nothing was recording.

        The ``cannot_confirm`` is the sentence that matters, and it is deliberately not
        "whether work happened". It cannot: this source holds no observation of the period
        at all, so there is nothing here to reason about whether the work happened, and a
        finding that appeared to raise that question would be inviting the reader to answer
        it from the rest of the report, which describes a different population entirely.
        """
        return Finding(
            claim=self.claim(snapshot),
            snapshot=snapshot,
            title="This period was never recorded",
            what_was_observed=(
                f"{len(observed)} status transitions fall inside {self.period.render_text()}, "
                "and the backend holds no transition dated at or before the period's start."
            ),
            suspected_cause=(
                "The transition log had not been recording yet — a backend installed during "
                "or after this period, or a project that did not exist in it."
            ),
            where=f"{snapshot.collection}",
            cannot_confirm=(
                "Whether any work happened in this period. This source holds no observation "
                "of it, so the period is absent rather than empty, and every figure a "
                "retrospective would place here would be a figure about no population. The "
                "absence is about when the recorder started, not about the team."
            ),
        )

    def _quiet(self, snapshot: Snapshot) -> Finding:
        """A period with no transitions in it, inside a source that was recording before it.

        The opposite conclusion to :meth:`_unrecorded` from the same empty read, which is
        why this cannot be a count of zero: the recorder was running, so the emptiness is a
        fact about the period rather than about the installation, and it is the only one of
        the two a reader should be invited to draw a conclusion from.
        """
        return Finding(
            claim=self.claim(snapshot),
            snapshot=snapshot,
            title="Nothing moved in this period, and the recorder was running",
            what_was_observed=(
                f"No status transition falls inside {self.period.render_text()}, while "
                f"{self.recorded_before:,} transition(s) are recorded before it began."
            ),
            suspected_cause=(
                "Either no ticket was worked on in this window, or tickets were worked on "
                "without their status changing — work that leaves no trace in this source is "
                "invisible here rather than absent from it."
            ),
            where=f"{snapshot.collection}",
            cannot_confirm=(
                "That nothing happened. A status transition is the only thing this source "
                "records, so a period of design, discussion or review that moved no ticket "
                "looks exactly like a period in which nobody worked. Distinguishing them "
                "needs a record of the work rather than of its status changes, and this "
                "source is not one."
            ),
        )


@dataclass(frozen=True)
class PeriodThroughputMeasure:
    """How much of the period's population finished inside it, as a rate over named states.

    The terminal states are the caller's because a ticket system's lifecycle is
    per-project configuration; see :func:`terminal_states` for what happens when none are
    named. They appear in the claim's own text, so a reader who disagrees with the choice
    can see it rather than having to infer it from a number.
    """

    period: Period
    terminal: tuple[str, ...] = (DEFAULT_TERMINAL_STATE,)

    slug = THROUGHPUT_SLUG

    def compute(self, snapshot: Snapshot) -> Figure:
        observed = _in_period(snapshot, self.period)
        tickets = {transition.ticket_id for transition in observed}
        if not tickets:
            raise ComparativeRefusalError(self._empty_refusal())
        finished = {
            transition.ticket_id
            for transition in observed
            if _is_terminal(transition, self.terminal)
        }
        return RateFigure(
            claim=self.claim(snapshot),
            snapshot=snapshot,
            title=(f"Tickets reaching {self._terminal_text()} inside {self.period.render_text()}"),
            rate=Rate(
                numerator=len(finished),
                denominator=Denominator(
                    description=self._denominator_text(snapshot),
                    size=len(tickets),
                    size_noun="tickets",
                ),
            ),
        )

    def claim(self, snapshot: Snapshot) -> Claim:
        observed = _in_period(snapshot, self.period)
        tickets = {transition.ticket_id for transition in observed}
        finished = {
            transition.ticket_id
            for transition in observed
            if _is_terminal(transition, self.terminal)
        }
        return Claim(
            slug=self.slug,
            statement=(
                f"{len(finished):,} of {len(tickets):,} tickets seen in "
                f"{self.period.render_text()} reached {self._terminal_text()} inside it."
            ),
            does_not_mean=(
                "That the period's work was completed, and not that the team finished what it "
                "set out to do: the denominator is tickets *seen* in the window, which is not "
                "the same population as tickets *planned* for it, and no planned side exists "
                f"in this source — see the {COMMITTED_VERSUS_DELIVERED_SLUG} refusal. Nor does "
                "it say how long anything took, or who did it: a status transition names no "
                "principal. A ticket can also reach a terminal state and be reopened, and this "
                "figure counts the ticket once however many times it got there."
            ),
            falsifier=(
                "A read in which the same window yields a materially different share, or in "
                "which the backend's lifecycle places a state outside "
                f"{self._terminal_text()} in the done category — the second would mean the "
                "terminal set was the wrong declared choice rather than that the work changed."
            ),
            denominator=Denominator(
                description=self._denominator_text(snapshot),
                size=max(1, len(tickets)),
                size_noun="tickets",
            ),
            kind=ClaimKind.descriptive,
            granularity=Granularity.corpus,
        )

    def _denominator_text(self, snapshot: Snapshot) -> str:
        """The population, and the window in time beside the count in records.

        **The window is in this string because a rate over records is not a rate over
        time.** Two hundred tickets in a fortnight and two hundred in a year are the same
        figure and different events, and a denominator that named only the population would
        let a reader take the first as evidence of throughput. So the declared period is
        part of the denominator's own description rather than a note above it, which is what
        makes it impossible to quote this rate without quoting its window.
        """
        return (
            f"tickets seen in {self.period.render_text()}, counted as reaching "
            f"{self._terminal_text()}"
        )

    def _terminal_text(self) -> str:
        return ", ".join(f"`{state}`" for state in self.terminal)

    def _empty_refusal(self) -> Refusal:
        """No ticket was seen in the window at all, so there is no population to be a fraction of.

        A rate over zero tickets is not a rate of zero throughput; it is a division whose
        result nobody can act on, which is what :class:`~tenbin.claims.model.Denominator`
        refuses to be built at the bottom of. The honest result is this refusal, and it
        points at the activity figure rather than at a missing fact -- the period-activity
        finding above it already says whether this is a quiet period or an unrecorded one,
        and duplicating that here would give a reader two answers to one question.
        """
        return Refusal(
            claim_slug=self.slug,
            title="No ticket was seen in this period",
            reason=(
                f"{self.period.render_text()} contains no ticket this read observed, so there "
                "is no population for a share to be taken over. A rate of zero would say the "
                "team finished nothing, which is a different claim from there being nothing "
                "to finish. The period-activity section above says which of the two this is."
            ),
            missing_fact=None,
            unblocked_by=None,
        )


@dataclass(frozen=True)
class TimeToTerminalMeasure:
    """How long tickets took to reach a terminal state, as a distribution of lower bounds.

    **The left edge is this source's first observation of a ticket, not its creation**, and
    that single fact governs everything else here. The backend records no creation moment
    (:data:`NO_CREATION_MOMENT`), so the interval is short by everything that happened
    before the earliest recorded transition, and it is undefined for a ticket whose only
    recorded transition is the one that finished it -- counted under
    :data:`NO_EARLIER_TRANSITION` rather than rendered as a zero, because a zero-hour ticket
    is a claim about work rather than about what was recorded.

    **The population is left-censored, and the bias has one direction.** Only tickets whose
    first recorded transition falls inside the period are in it, because a period-scoped
    read has nothing to say about a ticket that was already in flight when the window
    opened. So this distribution is missing the long stays preferentially, which makes it
    read faster than the period was. A one-directional bias that is merely mentioned in a
    docstring is a bias a report will publish as a level, so it is in the claim.

    A distribution and never a mean, for the reason :func:`tenbin.measures.base.duration_buckets`
    exists: the mean of this population is a number no ticket occupies, and it would move
    with the bucket edges rather than with the work.
    """

    period: Period
    terminal: tuple[str, ...] = (DEFAULT_TERMINAL_STATE,)

    slug = TIME_TO_TERMINAL_SLUG

    def compute(self, snapshot: Snapshot) -> Figure:
        walk = self._intervals(snapshot)
        if not walk.measured:
            raise ComparativeRefusalError(
                Refusal(
                    claim_slug=self.slug,
                    title="No ticket finished inside this period with an earlier transition",
                    reason=(
                        f"{self.period.render_text()} contains no ticket that both reached "
                        f"{self._terminal_text()} and had an earlier transition recorded "
                        "here, so no interval can be measured. An empty distribution would "
                        "render as a shape, and a shape reads as a finding."
                    ),
                    missing_fact=(
                        f"a ticket reaching {self._terminal_text()} inside the period with at "
                        "least one transition recorded before it"
                    ),
                    unblocked_by=None,
                )
            )
        excluded: dict[str, int] = {}
        if walk.no_earlier:
            excluded[NO_EARLIER_TRANSITION] = walk.no_earlier
        if walk.still_open:
            excluded[STILL_OPEN] = walk.still_open
        return DistributionFigure(
            claim=self.claim(snapshot),
            snapshot=snapshot,
            title=(
                f"Time from a ticket's first recorded transition to "
                f"{self._terminal_text()}, inside {self.period.render_text()}"
            ),
            values=duration_buckets(walk.measured),
            excluded=excluded,
        )

    def claim(self, snapshot: Snapshot) -> Claim:
        walk = self._intervals(snapshot)
        return Claim(
            slug=self.slug,
            statement=(
                f"{len(walk.measured):,} tickets reached {self._terminal_text()} inside "
                f"{self.period.render_text()} with an earlier transition recorded in the "
                "same read, distributed by the hours between the two."
            ),
            does_not_mean=(
                "How long the work took. " + NO_CREATION_MOMENT + " The distribution is also "
                "left-censored: it holds only tickets whose first recorded transition falls "
                "inside this window, so a ticket already in flight when the period opened is "
                "absent, and the stays that are missing are preferentially the long ones. "
                "Read it as a floor on duration and as fast by an unstated amount, not as a "
                "level."
            ),
            falsifier=(
                "A read over the same period that includes tickets whose first transition "
                "precedes the window and yields materially longer intervals — that would "
                "confirm the censoring is doing the work rather than the work being fast. The "
                "second falsifier is a backend that begins recording creation moments, which "
                "would make this interval measurable properly and this claim wrong."
            ),
            denominator=Denominator(
                description=(
                    f"tickets reaching {self._terminal_text()} inside "
                    f"{self.period.render_text()} with an earlier transition in the same read"
                ),
                size=max(1, len(walk.measured)),
                size_noun="tickets",
            ),
            kind=ClaimKind.descriptive,
            granularity=Granularity.corpus,
        )

    def _terminal_text(self) -> str:
        return ", ".join(f"`{state}`" for state in self.terminal)

    def _intervals(self, snapshot: Snapshot) -> _IntervalWalk:
        """The measured hours, and each population the measure declined to measure.

        Grouped through :func:`tenbin.store.transitions.transitions_by_ticket` so the
        ordering of a ticket's transitions is the sorted one rather than the backend's
        pagination order -- an interval computed from two out-of-order rows would be
        negative, and a negative duration would land in the bottom bucket and read as a
        ticket finished impossibly fast.
        """
        from tenbin.store.transitions import transitions_by_ticket

        grouped = transitions_by_ticket(_in_period(snapshot, self.period))
        measured: list[float] = []
        no_earlier = 0
        still_open = 0
        for rows in grouped.values():
            finished = next((row for row in rows if _is_terminal(row, self.terminal)), None)
            if finished is None:
                still_open += 1
                continue
            first = rows[0]
            if len(rows) < 2 or first.at == finished.at:
                no_earlier += 1
                continue
            measured.append((finished.at - first.at).total_seconds() / 3600.0)
        return _IntervalWalk(
            measured=tuple(measured),
            no_earlier=no_earlier,
            still_open=still_open,
        )


@dataclass(frozen=True)
class _IntervalWalk:
    """The measured hours and the two populations set aside, counted rather than dropped."""

    measured: tuple[float, ...]
    no_earlier: int
    still_open: int


def _in_period(snapshot: Snapshot, period: Period) -> tuple[Transition, ...]:
    """The transitions inside ``period``, half-open, from whatever the read holds.

    Filters here rather than trusting the read's own window, because the snapshot's window
    is derived from the transitions it contains and a period-scoped read therefore has a
    window *narrower* than the period it was cut to. Filtering by the declared edges is
    what makes the two different facts stay different: the header states the period, and
    this states which rows it covers.
    """
    return tuple(row for row in getattr(snapshot, "transitions", ()) if period.contains(row.at))


def who_carried_what_refusal() -> Refusal:
    """ "Who carried what", refused, and the reason is stronger than a missing fact.

    This is the slide a retrospective audience asks for first, and it is refused here for a
    reason the rest of this program does not have: a transition carries a
    ticket, a from-state, a to-state and a moment, and no principal. The wires
    do carry one -- Jira changelog entries have an author, Linear history
    entries have an actor -- and the adapters drop it at their own seam, because
    per-principal output is a different product with different consent,
    retention and access requirements. There is therefore no field to
    aggregate and no fact to withhold, so this is not the granularity floor doing its job
    — the floor would be the *second* line of defence here, behind an absence that no
    amount of configuration reaches.

    It is still rendered as a refusal rather than omitted, because a reader who asked for
    it and received nothing has to be able to tell that from a reader who asked and got a
    number. ``unblocked_by`` is ``None`` with no ticket offered: no change to a ticket
    source's read path would put a principal on a row this program already holds.
    """
    return Refusal(
        claim_slug=CARRIED_BY_SLUG,
        title="Who carried what",
        reason=(
            "A ticket status transition records a ticket, the state it moved from, "
            "the state it moved to, and when. It records no principal -- the wires "
            "do carry one, and the adapters drop it at their own seam -- so there "
            "is nothing here to group by: this is not the granularity floor refusing "
            "a narrow population, it is the absence of the field a grouping would need. "
            "Per-principal output would in any case be a different product with different "
            "consent, retention and access requirements rather than this report with a "
            "finer filter — but that argument is not what is stopping it here, and "
            "claiming it would misattribute the refusal."
        ),
        missing_fact=None,
        unblocked_by=None,
    )


def committed_versus_delivered_refusal() -> Refusal:
    """Committed versus delivered, refused, because the committed side does not exist.

    A retrospective's most quotable slide is a comparison against what was planned, and no
    supported ticket source validates estimates against delivery -- story points and
    estimates are fields a team may fill in and nothing in any of the systems would
    notice them filled in inconsistently. A comparison built over them would therefore
    compare two fields against each other and render the result as a shortfall in
    delivery, which is the most confident wrong sentence this program could print
    about a team.

    Populating those fields is a process change and not a code change, and it is not this
    ticket's to make. So the comparison is omitted and the omission is a rendered refusal
    naming what is missing — which is the difference between a retro that says it cannot
    say this and one that silently omits a slide everybody expected.
    """
    return Refusal(
        claim_slug=COMMITTED_VERSUS_DELIVERED_SLUG,
        title="Committed versus delivered",
        reason=(
            "The planned side of this comparison does not exist in the source. Ticket "
            "systems record estimates they never validate against delivery, so they are "
            "optional and empty in live data, and a comparison over them would measure "
            "the difference between two unpopulated fields and render it as a shortfall "
            "in what was delivered. Populating them is a process change rather than a "
            "code change, and until somebody does it the honest retrospective has no "
            "committed-versus-delivered figure at all."
        ),
        missing_fact=(
            "story points or estimates actually recorded on tickets, under a lifecycle that "
            "validates them rather than merely defining them"
        ),
        unblocked_by=None,
    )
