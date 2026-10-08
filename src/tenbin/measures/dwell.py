"""How long tickets sat in each state before moving on, over ticket status transitions.

The first measure in this package to read a source that is not the Kojutsu
corpus. It exists to be read as much as to be run: it is the proof that the
:class:`tenbin.corpus.snapshot.Snapshot` protocol actually admits a second source,
that the claim discipline reaches across both seams, and that a measure over ticket
flow needs none of Kojutsu's vocabulary to say honestly what it found.

**The interval is between two transitions of the same ticket, so the last one has
none, and that is the whole subtlety here.** A ticket that moved ``open →
in_progress → done`` yields two dwell intervals: in ``open`` and in ``in_progress``.
It does not yield one in ``done``, because nothing followed to bound it. The
alternative -- treating that final dwell as lasting until now -- would silently
add "however long it has been since" to a distribution of completed dwells, and
every such ticket would land in the top bucket, so the measure would report that
work accumulates in the final state when it would actually be reporting that the
distribution has no upper bound. It is therefore **excluded and counted**, under
its own key, never folded into a bucket and never rendered as zero. A ticket
sitting in ``done`` for a year is not a short interval and it is not an absent
one; it is an interval this corpus cannot bound, and the figure says so.

**The denominator is intervals, never transitions.** They are different
populations and the difference is exactly the exclusions: 1,848 transitions over
this backend yield fewer dwell intervals than that, by exactly the number of
tickets whose last transition has no successor. A figure whose denominator was
transitions while its buckets counted intervals would be a rate over a population
the buckets do not cover.

**What this cannot say, and says.** It cannot say who moved a ticket: a
transition records four fields and no principal -- the adapters drop whatever
principal the wire carried -- so there is nothing to refuse here, which is a
stronger statement than a refusal. It cannot say a dwell was long because
somebody was slow -- a state can be long because work in it is long, because
it was blocked, or because the ticket sat untouched. And the moment
on each transition is when the *source observed* the change, which is not
necessarily when a person made it.

**Not a mean, and not a count of "slow" tickets.** ``duration_buckets/1`` puts
every rung on the ladder in the output even where nothing landed, so a reader
comparing two periods sees the empty rungs rather than inferring them. A mean of
this distribution is a number no observation occupies.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Final

from tenbin.claims.model import Claim, ClaimKind, Denominator, Granularity
from tenbin.claims.refusals import ComparativeRefusalError, Refusal
from tenbin.corpus.snapshot import Snapshot
from tenbin.corpus.transition import Transition
from tenbin.measures.base import DistributionFigure, Figure, duration_buckets

#: The key the unbounded dwell is excluded under. Named rather than written inline
#: so the rendering, the tests, and a reader comparing two periods all name the
#: same condition with the same words.
OPEN_ENDED: Final[str] = "still in this state at read time, so no end to measure"

#: The exclusion key for a ticket whose transitions could not be ordered.
UNORDERABLE: Final[str] = "transitions for one ticket at an identical moment"


@dataclass(frozen=True)
class DwellInterval:
    """One completed stay in one state, bounded by the transition that ended it."""

    ticket_id: str
    state: str
    #: The transition that *entered* this state.
    entered: Transition
    #: The transition that left it. Always present -- this is what makes the
    #: interval closed, and its absence is what :data:`OPEN_ENDED` counts.
    left: Transition

    @property
    def duration(self) -> timedelta:
        return self.left.at - self.entered.at

    @property
    def hours(self) -> float:
        return self.duration.total_seconds() / 3600.0


@dataclass(frozen=True)
class DwellWalk:
    """The closed intervals a snapshot yields, and what it set aside."""

    intervals: tuple[DwellInterval, ...]
    #: The state each ticket was last seen in, once per ticket. Carried as the
    #: states rather than only as a count because a caller asking "how long do
    #: tickets sit in `blocked`" needs to know how many of the unbounded stays were
    #: in `blocked`, and counting every transition *into* a state instead would
    #: include the ones that were properly bounded by the next transition.
    open_ended_states: tuple[str, ...]
    unorderable: int

    @property
    def open_ended(self) -> int:
        return len(self.open_ended_states)

    @property
    def transitions(self) -> int:
        return len(self.intervals) + self.open_ended + self.unorderable


def dwell_intervals(snapshot: Snapshot) -> DwellWalk:
    """Walk a snapshot's transitions into closed dwell intervals.

    Takes the :class:`~tenbin.corpus.snapshot.Snapshot` protocol rather than a
    concrete snapshot type, because it reads ``transitions`` off the argument and
    that is the one thing the protocol does not guarantee. It is called with a
    :class:`~tenbin.store.transitions.TransitionSnapshot` and says so in its type
    error, so a caller passing a Kojutsu snapshot gets a sentence rather than an
    ``AttributeError`` from somewhere further from the cause.
    """
    transitions = getattr(snapshot, "transitions", None)
    if transitions is None:
        raise TypeError(
            "dwell_intervals/1 needs a transition snapshot, which carries "
            "'transitions'; this one carries "
            f"{type(snapshot).__name__}, which is a {snapshot.collection} read."
        )

    by_ticket: dict[str, list[Transition]] = {}
    for transition in transitions:
        by_ticket.setdefault(transition.ticket_id, []).append(transition)

    intervals: list[DwellInterval] = []
    open_ended_states: list[str] = []
    unorderable = 0

    for ticket_id, rows in by_ticket.items():
        # Sorted by moment so an interval cannot be computed from two rows the
        # backend served out of order. Ties break on the destination state, which
        # is arbitrary but deterministic -- which is why a tie is counted as
        # unorderable rather than silently resolved into a zero-length stay.
        ordered = sorted(rows, key=lambda row: (row.at, row.to_state))
        for earlier, later in zip(ordered, ordered[1:], strict=False):
            if earlier.at == later.at:
                unorderable += 1
                continue
            intervals.append(
                DwellInterval(
                    ticket_id=ticket_id,
                    state=earlier.to_state,
                    entered=earlier,
                    left=later,
                )
            )
        open_ended_states.append(ordered[-1].to_state)

    return DwellWalk(
        intervals=tuple(intervals),
        open_ended_states=tuple(sorted(open_ended_states)),
        unorderable=unorderable,
    )


class TimeInStateMeasure:
    """How long completed stays in each state were, as a distribution.

    One figure per state rather than one figure over every state pooled, because
    the states are not comparable quantities: a ticket in ``blocked`` for a week
    and a ticket in ``in_progress`` for a week have been in genuinely different
    situations, and pooling them produces a distribution whose shape is a property
    of the mix of states rather than of any of them. The alternative -- a single
    pooled figure with the state as a second axis -- was rejected because the
    reader who wants "how long is in_progress" would have to subtract.

    Denominator: completed intervals, never transitions.
    """

    slug = "time-in-state"

    def __init__(self, state: str) -> None:
        if not state.strip():
            raise ValueError("a state is named, not blank")
        self._state = state.strip()

    @property
    def state(self) -> str:
        return self._state

    def compute(self, snapshot: Snapshot) -> Figure:
        walk = dwell_intervals(snapshot)

        in_state = [interval for interval in walk.intervals if interval.state == self._state]

        excluded: dict[str, int] = {}
        # Counted over the walk's own open-ended states rather than over every
        # transition into this state: a transition into `blocked` that was followed
        # by another transition is a bounded stay and is already a bucket.
        open_in_state = walk.open_ended_states.count(self._state)
        if open_in_state:
            excluded[OPEN_ENDED] = open_in_state
        if walk.unorderable:
            excluded[UNORDERABLE] = walk.unorderable

        if not in_state:
            # An empty population has no claim, and that is enforced rather than
            # worked around: `Denominator` refuses a size below one because a rate
            # over zero records is not a rate. So there is nothing to attach a
            # figure to, and the honest result is the refusal -- carrying the
            # reason, as `drift.py` does, so a report listing what it could not
            # say gets a sentence rather than an absent section.
            raise ComparativeRefusalError(
                Refusal(
                    claim_slug=f"{self.slug}-{_slug(self._state)}",
                    title=f"Time in {_readable(self._state)}",
                    reason=(
                        f"No ticket left `{self._state}` in this read, so no interval "
                        "can be measured and no rate over one can be stated. An "
                        "empty distribution would render as a shape, which reads as "
                        "a finding."
                    ),
                    missing_fact=(
                        f"at least one transition out of `{self._state}` in the period read"
                    ),
                    unblocked_by=None,
                )
            )

        return DistributionFigure(
            claim=self.claim(snapshot),
            snapshot=snapshot,
            title=f"Hours a ticket spent in {_readable(self._state)}",
            values=duration_buckets(interval.hours for interval in in_state),
            excluded=excluded,
        )

    def claim(self, snapshot: Snapshot) -> Claim:
        walk = dwell_intervals(snapshot)
        in_state = sum(1 for interval in walk.intervals if interval.state == self._state)
        total_transitions = len(getattr(snapshot, "transitions", ()))

        return Claim(
            slug=f"{self.slug}-{_slug(self._state)}",
            statement=(
                f"{in_state:,} completed stays in `{self._state}` were observed, "
                "distributed by how long they lasted."
            ),
            does_not_mean=(
                "That the time was spent badly, or by whom. A transition records no "
                "principal, and a long stay is equally consistent with work that "
                "takes a long time, work that was blocked, and work nobody returned to."
            ),
            falsifier=(
                "A read in which the same states yield materially different intervals, "
                "or in which the backend's own transition ordering disagrees with the "
                "one reconstructed here."
            ),
            denominator=Denominator(
                description=(
                    f"completed stays in `{self._state}`, of "
                    f"{total_transitions:,} transitions read from "
                    f"{snapshot.collection}"
                ),
                size=in_state,
                size_noun="intervals",
            ),
            kind=ClaimKind.descriptive,
            granularity=Granularity.team,
        )


def states_in(snapshot: Snapshot) -> tuple[str, ...]:
    """Every state this read observed, in a stable order.

    Sorted rather than first-seen so a report listing states does not reorder
    itself between two runs of the same corpus, which would make the two reports
    harder to read side by side than they need to be.
    """
    return tuple(sorted({row.to_state for row in getattr(snapshot, "transitions", ())}))


def _readable(state: str) -> str:
    return state.replace("_", " ")


def _slug(state: str) -> str:
    return state.replace("_", "-").lower()
