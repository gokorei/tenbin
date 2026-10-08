"""Dwell intervals over ticket transitions, and the exclusion that carries the measure.

The open-ended stay is the whole point of this file. A ticket's final transition
has no successor, so nothing bounds how long it sat there. Every wrong answer to
that is available and each is tested here: treat it as zero and the distribution
gains a pile at the bottom; treat it as "until now" and every ticket piles into the
top bucket, so the measure reports accumulation where it should report an unbounded
range; drop it silently and the figure's denominator stops matching its buckets.

The tests below therefore assert the *count* of the exclusion, not merely that
something was excluded, because a figure that said "excluded: 1" for the wrong
reason would pass a looser test.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from tenbin.claims.gate import ClaimGate
from tenbin.claims.refusals import ComparativeRefusalError
from tenbin.corpus.snapshot import Completeness, CorpusWindow
from tenbin.corpus.transition import Transition
from tenbin.measures.dwell import (
    OPEN_ENDED,
    DwellWalk,
    TimeInStateMeasure,
    dwell_intervals,
    states_in,
)
from tenbin.store.transitions import TransitionSnapshot


def _t(
    ticket_id: str = "T1",
    from_state: str | None = "open",
    to_state: str = "in_progress",
    at: datetime | None = None,
) -> Transition:
    return Transition(
        ticket_id=ticket_id,
        from_state=from_state,
        to_state=to_state,
        at=at or datetime(2026, 9, 1, tzinfo=UTC),
    )


def _snapshot(rows: list[Transition], **overrides) -> TransitionSnapshot:
    defaults = {
        "transitions": tuple(rows),
        "read_at": datetime(2026, 9, 30, tzinfo=UTC),
        "store_total": len(rows),
        "enumerated": len(rows),
        "completeness": Completeness.COMPLETE,
    }
    defaults.update(overrides)
    return TransitionSnapshot(**defaults)  # type: ignore[arg-type]


class TestBuildingIntervals:
    def test_a_two_step_ticket_yields_one_closed_interval_and_one_open_stay(self) -> None:
        """open -> in_progress -> done: the interval in `open` is closed; the one in
        `in_progress` has no successor and must not be counted as an interval."""
        snapshot = _snapshot(
            [
                _t(to_state="in_progress", at=datetime(2026, 9, 1, tzinfo=UTC)),
                _t(to_state="done", at=datetime(2026, 9, 3, tzinfo=UTC)),
            ]
        )
        walk = dwell_intervals(snapshot)

        assert len(walk.intervals) == 1
        assert walk.intervals[0].state == "in_progress"
        assert walk.intervals[0].duration == timedelta(days=2)
        # The `done` stay is open: it is the ticket's last transition.
        assert walk.open_ended == 1
        assert walk.unorderable == 0

    def test_rows_are_ordered_by_moment_not_by_arrival(self) -> None:
        later = _t(to_state="done", at=datetime(2026, 9, 5, tzinfo=UTC))
        earlier = _t(to_state="in_progress", at=datetime(2026, 9, 1, tzinfo=UTC))
        walk = dwell_intervals(_snapshot([later, earlier]))

        assert walk.intervals[0].entered is earlier
        assert walk.intervals[0].left is later
        assert walk.intervals[0].duration > timedelta(0), "a negative dwell is impossible"

    def test_a_single_transition_yields_no_interval_and_one_open_stay(self) -> None:
        walk = dwell_intervals(_snapshot([_t()]))
        assert walk.intervals == ()
        assert walk.open_ended == 1

    def test_two_transitions_at_the_same_moment_are_unorderable_not_zero(self) -> None:
        """A tie cannot be resolved without inventing which came first."""
        moment = datetime(2026, 9, 1, tzinfo=UTC)
        walk = dwell_intervals(
            _snapshot([_t(to_state="in_progress", at=moment), _t(to_state="done", at=moment)])
        )
        assert walk.intervals == ()
        assert walk.unorderable == 1
        assert walk.open_ended == 1

    def test_the_counts_account_for_every_transition(self) -> None:
        """A walk that loses a row is a walk whose denominator is a lie."""
        rows = [
            _t(ticket_id="A", to_state="in_progress", at=datetime(2026, 9, 1, tzinfo=UTC)),
            _t(ticket_id="A", to_state="done", at=datetime(2026, 9, 2, tzinfo=UTC)),
            _t(ticket_id="B", to_state="blocked", at=datetime(2026, 9, 3, tzinfo=UTC)),
        ]
        walk = dwell_intervals(_snapshot(rows))
        assert walk.transitions == len(rows)

    def test_a_kojutsu_snapshot_is_refused_with_a_sentence_not_an_attribute_error(self) -> None:
        from tenbin.corpus.record import Classification, RecordKind

        kojutsu = type(
            "_KojutsuLike",
            (),
            {
                "collection": "kojutsu",
                "records": (),
                "read_at": datetime(2026, 9, 30, tzinfo=UTC),
                "store_total": 0,
                "enumerated": 0,
                "completeness": Completeness.COMPLETE,
                "truncation": None,
                "error": None,
                "window": CorpusWindow(earliest=None, latest=None, unreadable_timestamps=0),
            },
        )()

        with pytest.raises(TypeError, match="transition snapshot"):
            dwell_intervals(kojutsu)  # type: ignore[arg-type]
        # The enum import above keeps this test honest about what it is not building.
        assert Classification is not None and RecordKind is not None


class TestTheFigure:
    def _two_step(self) -> TransitionSnapshot:
        return _snapshot(
            [
                _t(to_state="in_progress", at=datetime(2026, 9, 1, tzinfo=UTC)),
                _t(to_state="done", at=datetime(2026, 9, 3, tzinfo=UTC)),
            ]
        )

    def test_it_buckets_the_closed_interval(self) -> None:
        figure = TimeInStateMeasure("in_progress").compute(self._two_step())
        assert figure.counted == 1
        assert sum(figure.values.values()) == 1

    def test_the_open_stay_is_excluded_under_its_own_key_and_not_bucketed(self) -> None:
        """The measure's own state is not open-ended here, so nothing is excluded --
        which is the case that would otherwise be silently wrong in the other
        direction. Adding the open stay to this state must surface it."""
        rows = [
            _t(ticket_id="A", to_state="in_progress", at=datetime(2026, 9, 1, tzinfo=UTC)),
            _t(ticket_id="A", to_state="done", at=datetime(2026, 9, 3, tzinfo=UTC)),
            # A second ticket that entered in_progress and never left.
            _t(ticket_id="B", to_state="in_progress", at=datetime(2026, 9, 5, tzinfo=UTC)),
        ]
        figure = TimeInStateMeasure("in_progress").compute(_snapshot(rows))

        assert figure.counted == 1, "only the bounded stay is an interval"
        assert figure.excluded[OPEN_ENDED] == 1, (
            "the unbounded stay must be counted, not bucketed as zero"
        )

    def test_the_exclusion_key_reads_as_a_condition_not_a_number(self) -> None:
        # Two tickets, because one ticket cannot both leave `blocked` and still be
        # in it. A leaves (a closed interval, so the figure exists at all); B is
        # last seen in `blocked` (an unbounded stay, which is the exclusion).
        rows = [
            _t(ticket_id="A", to_state="blocked", at=datetime(2026, 9, 1, tzinfo=UTC)),
            _t(ticket_id="A", to_state="in_progress", at=datetime(2026, 9, 2, tzinfo=UTC)),
            _t(ticket_id="B", to_state="blocked", at=datetime(2026, 9, 3, tzinfo=UTC)),
        ]
        figure = TimeInStateMeasure("blocked").compute(_snapshot(rows))
        assert figure.counted == 1
        assert OPEN_ENDED in figure.excluded
        assert "state" in OPEN_ENDED and "read time" in OPEN_ENDED

    def test_an_unobserved_state_refuses_rather_than_rendering_an_empty_shape(self) -> None:
        """A distribution of nothing renders as a shape, which reads as a finding.

        The refusal is raised rather than returned because `Denominator` refuses a
        size below one, so there is no valid claim to attach a figure to. Following
        `drift.py`, the exception carries the whole `Refusal` so a report listing
        what it could not say gets the reason and not just an absent section.
        """
        with pytest.raises(ComparativeRefusalError) as excinfo:
            TimeInStateMeasure("nonexistent_state").compute(self._two_step())
        refusal = excinfo.value.refusal
        assert refusal.claim_slug == "time-in-state-nonexistent-state"
        assert "No ticket left" in refusal.reason
        assert refusal.missing_fact
        assert refusal.unblocked_by is None

    def test_the_denominator_counts_intervals_not_transitions(self) -> None:
        figure = TimeInStateMeasure("in_progress").compute(self._two_step())
        assert figure.claim.denominator.size == figure.counted
        assert figure.claim.denominator.size == 1
        assert "2 transitions" in figure.claim.denominator.description

    def test_the_claim_names_the_state_and_refuses_the_reading_it_cannot_support(self) -> None:
        figure = TimeInStateMeasure("in_progress").compute(self._two_step())
        assert "`in_progress`" in figure.claim.statement
        assert figure.claim.falsifier.strip()
        assert "principal" in figure.claim.does_not_mean

    def test_it_passes_the_claim_gate(self) -> None:
        """A measure that cannot clear the gate is a measure whose claims are decorative."""
        figure = TimeInStateMeasure("in_progress").compute(self._two_step())
        ClaimGate().check(figure.claim)

    def test_rate_text_reports_the_snapshots_own_completeness(self) -> None:
        rows = [
            _t(to_state="in_progress", at=datetime(2026, 9, 1, tzinfo=UTC)),
            _t(to_state="done", at=datetime(2026, 9, 3, tzinfo=UTC)),
        ]
        partial = _snapshot(
            rows, store_total=10, enumerated=2, completeness=Completeness.OFFSET_CAP
        )
        figure = TimeInStateMeasure("in_progress").compute(partial)
        text = figure.rate_text()
        assert "whole corpus" not in text.lower()
        assert "not reached" in text.lower()

    def test_a_blank_state_is_refused(self) -> None:
        with pytest.raises(ValueError, match="named, not blank"):
            TimeInStateMeasure("   ")

    def test_the_two_states_get_two_figures_rather_than_one_pooled_one(self) -> None:
        """Pooling is refused by construction: one figure per state, chosen by the caller."""
        rows = [
            _t(ticket_id="A", to_state="blocked", at=datetime(2026, 9, 1, tzinfo=UTC)),
            _t(ticket_id="A", to_state="in_progress", at=datetime(2026, 9, 2, tzinfo=UTC)),
            _t(ticket_id="B", to_state="blocked", at=datetime(2026, 9, 1, tzinfo=UTC)),
            _t(ticket_id="B", to_state="done", at=datetime(2026, 9, 2, tzinfo=UTC)),
        ]
        snapshot = _snapshot(rows)
        blocked = TimeInStateMeasure("blocked").compute(snapshot)
        assert blocked.counted == 2
        # Nobody left in_progress, so that measure refuses rather than reporting a
        # distribution of nothing. Two states, two verdicts, no pooling either way.
        with pytest.raises(ComparativeRefusalError):
            TimeInStateMeasure("in_progress").compute(snapshot)
        assert states_in(snapshot) == ("blocked", "done", "in_progress")


class TestStatesIn:
    def test_it_is_sorted_so_two_runs_render_in_the_same_order(self) -> None:
        rows = [
            _t(ticket_id="A", to_state="done"),
            _t(ticket_id="B", to_state="blocked"),
            _t(ticket_id="C", to_state="archived"),
        ]
        assert states_in(_snapshot(rows)) == ("archived", "blocked", "done")

    def test_an_empty_read_yields_no_states_rather_than_a_placeholder(self) -> None:
        assert states_in(_snapshot([])) == ()


def test_the_walk_type_carries_its_own_accounting() -> None:
    """A reader must be able to check the walk kept its books without recomputing it."""
    walk = DwellWalk(intervals=(), open_ended_states=("done", "blocked", "done"), unorderable=2)
    assert walk.open_ended == 3
    assert walk.transitions == 5
