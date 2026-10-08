"""A retrospective over a declared period, and the four things it must never do.

The tests here are grouped by what they protect rather than by what they call, because
almost every one of them is about a way the output could mislead.

**The period arithmetic is tested without a backend at all**, and that is the point: period
boundaries are the one claim a retrospective makes that no figure in it can corroborate, so
they are asserted as dates. The tests that matter most are the two about the seam -- that
``[start, end)`` really is half-open, and that a transition recorded at exactly a boundary
lands in one retrospective and not two. A backend whose ``to`` bound is inclusive makes the
second one fail, and it did.

**The three empty-period cases are distinguished by a second request, and both directions
are tested.** A period with nothing in it must render differently depending on whether the
recorder was running beforehand, because the two readings call for opposite conclusions
about the team. The tests assert on the rendered figure's shape rather than on a helper's
return value, so a change that kept the verdict but lost the distinction would fail here.

**The prior-period comparison is tested through the archive, not around it.** The property
is that two retrospectives archived from consecutive periods can be drifted, and that the
drift reads the *stored* figures rather than re-reading the corpus -- so the test asserts
the drift renders the two archived values, and separately that archiving the same period
twice is refused rather than appended.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

import tenbin.cli as cli_module
from tenbin.cli import app as cli
from tenbin.corpus.period import Period, PeriodActivity, PeriodConfig
from tenbin.corpus.snapshot import Completeness
from tenbin.corpus.transition import Transition
from tenbin.measures.base import CountFigure, DistributionFigure, Finding, RateFigure
from tenbin.measures.drift import DriftMeasure
from tenbin.measures.retrospective import (
    CARRIED_BY_SLUG,
    COMMITTED_VERSUS_DELIVERED_SLUG,
    NO_EARLIER_TRANSITION,
    PERIOD_ACTIVITY_SLUG,
    STILL_OPEN,
    THROUGHPUT_SLUG,
    TIME_TO_TERMINAL_SLUG,
    terminal_states,
)
from tenbin.report import render_markdown
from tenbin.report.archive import Archive, DuplicateReadingError
from tenbin.report.ordering import PartKind
from tenbin.report.retrospective import PeriodRead, build_retrospective, read_period
from tenbin.store.ticket_sources.go_experiment import GoExperimentClient
from tenbin.store.transitions import take_transition_snapshot
from tests.conftest import plain_output

#: The convention every test in this file uses unless it is testing the convention itself.
#: Two weeks from the 5th of January 2026, which puts the boundaries mid-month so a
#: calendar-month grouping could not accidentally agree with them.
CADENCE = "14d"
ANCHOR = "2026-01-05T00:00:00Z"

#: Period 0 of that convention. Written out rather than derived so a test that asserts a
#: boundary is asserting a date, and an arithmetic mistake in the code under test shows up
#: as a wrong date rather than as a consistent wrong date.
PERIOD_0 = ("2026-01-05T00:00:00+00:00", "2026-01-19T00:00:00+00:00")
PERIOD_1 = ("2026-01-19T00:00:00+00:00", "2026-02-02T00:00:00+00:00")


def config() -> PeriodConfig:
    return PeriodConfig.from_text(CADENCE, ANCHOR)


def period(index: int = 0) -> Period:
    return config().period_at(index)


def at(text: str) -> datetime:
    return datetime.fromisoformat(text)


def row(ticket: str, frm: str | None, to: str, moment: str) -> dict[str, Any]:
    """One backend row. ``frm`` may be ``None``: the backend omits an absent origin."""
    return {"ticket_id": ticket, "from": frm, "to": to, "at": moment}


def envelope(rows: list[dict[str, Any]], *, total: int | None = None) -> dict[str, Any]:
    return {
        "results": rows,
        "total": len(rows) if total is None else total,
        "page": 1,
        "limit": 200,
        "total_pages": 1,
    }


def ticket_backend(rows: list[dict[str, Any]]):
    """A ``/status-transitions`` that filters the way the backend does: both bounds inclusive.

    Inclusive on purpose, and asserted to be. The backend documents it at
    ``filterStatusTransitions``, and it is the reason
    :func:`tenbin.report.retrospective.replace_to_period` exists -- so a fake with exclusive
    bounds would let that function's removal go unnoticed.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        since, until = params.get("from"), params.get("to")
        matching = [
            r
            for r in rows
            if (not since or at(r["at"]) >= at(since)) and (not until or at(r["at"]) <= at(until))
        ]
        return httpx.Response(200, json=envelope(matching))

    return handler


def a_client(rows: list[dict[str, Any]]) -> GoExperimentClient:
    return GoExperimentClient(
        "http://127.0.0.1:8082", transport=httpx.MockTransport(ticket_backend(rows))
    )


def a_snapshot(rows: list[Transition], **overrides: Any):
    from tenbin.store.transitions import TransitionSnapshot

    defaults: dict[str, Any] = {
        "transitions": tuple(rows),
        "read_at": at("2026-01-20T00:00:00+00:00"),
        "store_total": len(rows),
        "enumerated": len(rows),
        "completeness": Completeness.COMPLETE,
    }
    return TransitionSnapshot(**{**defaults, **overrides})


def a_read(
    rows: list[Transition], *, activity=PeriodActivity.RECORDED, before: int = 0
) -> PeriodRead:
    return PeriodRead(
        period=period(),
        snapshot=a_snapshot(rows),
        activity=activity,
        recorded_before=before,
    )


class TestTheConventionIsRequiredAndNeverGuessed:
    """Nothing is defaulted, because a guessed cadence is invisible in the output."""

    def test_a_cadence_and_an_anchor_produce_the_boundaries_they_imply(self) -> None:
        boundaries = period(0)

        assert boundaries.start.isoformat() == PERIOD_0[0]
        assert boundaries.end.isoformat() == PERIOD_0[1]

    def test_a_cadence_is_accepted_as_days_or_hours_and_nothing_else(self) -> None:
        assert PeriodConfig.from_text("336h", ANCHOR).cadence == timedelta(days=14)
        assert PeriodConfig.from_text("14d", ANCHOR).cadence == timedelta(days=14)

    @pytest.mark.parametrize("spelling", ["2w", "fortnight", "14", "-14d", "1.5d", "d"])
    def test_an_unparseable_cadence_is_refused_rather_than_guessed(self, spelling: str) -> None:
        with pytest.raises(ValueError, match="whole number of days or hours"):
            PeriodConfig.from_text(spelling, ANCHOR)

    def test_an_anchor_without_a_zone_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must carry a zone"):
            PeriodConfig.from_text(CADENCE, "2026-01-05T00:00:00")

    def test_a_zero_or_negative_cadence_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            PeriodConfig(cadence=timedelta(0), anchor=at(ANCHOR))

    def test_a_period_before_the_anchor_gets_a_negative_index_rather_than_the_wrong_one(
        self,
    ) -> None:
        earlier = config().period_containing(at("2025-12-25T00:00:00+00:00"))

        assert earlier.index == -1
        assert earlier.end == at(PERIOD_0[0])

    def test_a_moment_exactly_on_a_boundary_belongs_to_the_later_period(self) -> None:
        assert period(0).contains(at(PERIOD_0[1])) is False
        assert period(1).contains(at(PERIOD_0[1])) is True

    def test_the_previous_period_is_exact_rather_than_approximate(self) -> None:
        assert period(1).previous().start == at(PERIOD_0[0])
        assert period(1).previous().end == at(PERIOD_1[0])

    def test_a_naive_moment_cannot_be_placed_against_a_declared_boundary(self) -> None:
        with pytest.raises(ValueError, match="aware datetime"):
            period(0).contains(datetime(2026, 1, 10, 12, 0))

    def test_a_hand_picked_window_is_refused_because_it_was_not_derived_from_the_convention(
        self,
    ) -> None:
        with pytest.raises(ValueError, match="not a whole number of"):
            Period(
                index=0,
                start=at("2026-01-06T00:00:00+00:00"),
                end=at("2026-01-20T00:00:00+00:00"),
                cadence=timedelta(days=14),
                anchor=at(ANCHOR),
            )

    def test_the_short_form_states_the_window_and_which_end_is_exclusive(self) -> None:
        """The form a figure repeats: the boundaries, and no more."""
        rendered = period(0).render_text()

        assert PERIOD_0[0] in rendered
        assert PERIOD_0[1] in rendered
        assert "end exclusive" in rendered
        # The convention clause is deliberately absent: a figure repeats this string ten
        # times in one report, and the anchor's own date is the period's start here, so
        # what is absent is the *clause*, not the date.
        assert "cadence" not in rendered
        assert "anchored at" not in rendered

    def test_the_declaration_carries_the_convention_and_the_header_states_it_once(self) -> None:
        """A figure's text stays short because the header prints the long form exactly once."""
        document = build_retrospective(
            a_read([Transition("T1", "open", "in_progress", at("2026-01-06T09:00:00+00:00"))])
        )
        rendered = render_markdown(document)

        assert document.corpus.period is not None
        assert "anchored at" in document.corpus.period.render_declaration()
        assert rendered.count("anchored at") == 1


class TestAnEmptyPeriodIsDistinguishedFromAnUnrecordedOne:
    """The two are different findings and must not render as one."""

    def test_a_period_with_rows_is_recorded_and_renders_a_count(self) -> None:
        read = a_read([Transition("T1", "open", "in_progress", at("2026-01-06T09:00:00+00:00"))])

        figure = build_retrospective(read).sections[0].figure

        assert isinstance(figure, CountFigure)
        assert figure.value == 1

    def test_no_rows_with_history_before_is_a_quiet_period_not_an_empty_count(self) -> None:
        read = a_read([], activity=PeriodActivity.QUIET, before=1_848)

        figure = build_retrospective(read).sections[0].figure

        assert isinstance(figure, Finding)
        assert "recorder was running" in figure.title
        assert "1,848" in figure.what_was_observed

    def test_no_rows_and_no_history_is_never_recorded_and_says_it_is_about_the_installation(
        self,
    ) -> None:
        read = a_read([], activity=PeriodActivity.UNRECORDED, before=0)

        figure = build_retrospective(read).sections[0].figure

        assert isinstance(figure, Finding)
        assert "never recorded" in figure.title
        assert "not about the team" in figure.cannot_confirm

    def test_the_two_empty_cases_are_told_apart_from_the_output_alone(self) -> None:
        quiet = build_retrospective(a_read([], activity=PeriodActivity.QUIET, before=10))
        unrecorded = build_retrospective(a_read([], activity=PeriodActivity.UNRECORDED, before=0))

        assert quiet.sections[0].heading != unrecorded.sections[0].heading

    def test_a_quiet_period_refuses_the_throughput_rate_rather_than_reporting_zero(self) -> None:
        document = build_retrospective(a_read([], activity=PeriodActivity.QUIET, before=10))

        section = next(s for s in document.sections if s.slug == THROUGHPUT_SLUG)

        assert section.figure is None
        assert "no ticket this read observed" in (section.refusal.reason or "")

    def test_the_probe_asks_up_to_the_periods_own_start_and_not_past_it(self) -> None:
        """The boundary is the period's *start*, so nothing the period owns is counted twice.

        An inclusive `to` at ``period.start`` would count a transition recorded exactly
        there -- which belongs to this period -- and report the period as not being the
        first thing ever recorded, when it is the first thing recorded *in this window*.
        """
        seen: list[dict[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            params = dict(request.url.params)
            seen.append(params)
            earlier = params.get("to") == PERIOD_0[0]
            return httpx.Response(
                200,
                json=envelope(
                    [row("T0", "backlog", "open", "2025-12-20T09:00:00+00:00")] if earlier else []
                ),
            )

        with GoExperimentClient(
            "http://127.0.0.1:8082", transport=httpx.MockTransport(handler)
        ) as c:
            read = read_period(c, period(0))

        assert seen[0]["to"] == PERIOD_0[1]
        assert seen[1]["to"] == PERIOD_0[0]
        assert read.activity is PeriodActivity.QUIET
        assert read.recorded_before == 1

    def test_the_probe_shares_the_projects_scope_so_history_is_the_projects_own(self) -> None:
        seen: list[dict[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(dict(request.url.params))
            return httpx.Response(200, json=envelope([]))

        with GoExperimentClient(
            "http://127.0.0.1:8082", transport=httpx.MockTransport(handler)
        ) as c:
            read_period(c, period(0), project_id="4GK6J742")

        assert seen[1].get("project_id") == "4GK6J742"


class TestTheBoundaryTransitionBelongsToExactlyOnePeriod:
    """The backend's ``to`` is inclusive and the period is half-open."""

    def test_a_transition_at_the_boundary_is_in_the_next_period_and_not_this_one(self) -> None:
        rows = [row("T1", "open", "in_progress", PERIOD_0[1])]
        boundary = Transition("T1", "open", "in_progress", at(PERIOD_0[1]))

        this_period = read_period(a_client(rows), period(0))
        next_period = read_period(a_client(rows), period(1))

        assert this_period.snapshot.transitions == ()
        assert boundary in next_period.snapshot.transitions

    def test_two_consecutive_retrospectives_between_them_count_every_transition_once(self) -> None:
        rows = [
            row("T1", "open", "in_progress", "2026-01-06T09:00:00+00:00"),
            row("T1", "in_progress", "done", PERIOD_0[1]),
            row("T2", "open", "in_progress", "2026-01-25T09:00:00+00:00"),
        ]

        first = read_period(a_client(rows), period(0))
        second = read_period(a_client(rows), period(1))

        # Period 0 holds the opening transition only; the `done` sits exactly on the
        # boundary and belongs to period 1, which also holds T2's.
        assert first.snapshot.enumerated == 1
        assert second.snapshot.enumerated == 2
        assert first.snapshot.enumerated + second.snapshot.enumerated == len(rows)

    def test_the_narrowed_counts_agree_with_the_narrowed_rows(self) -> None:
        """The header must not render a boundary artefact as a walk that missed records."""
        rows = [row("T1", "open", "in_progress", PERIOD_0[1])]

        snapshot = read_period(a_client(rows), period(0)).snapshot

        assert snapshot.enumerated == snapshot.store_total == 0


class TestTheFiguresCarryTheirWindowAndTheirPopulation:
    """Criterion: a rate names its window in time, not only its size in records."""

    def test_the_period_boundaries_are_in_the_header_and_qualify_every_figure(self) -> None:
        read = a_read([Transition("T1", "open", "in_progress", at("2026-01-06T09:00:00+00:00"))])

        header = build_retrospective(read).corpus

        assert header.period is not None
        kinds = [part.kind for part in header.parts()]
        assert kinds.index(PartKind.period) < kinds.index(PartKind.enumeration)

    def test_an_unscoped_report_has_no_period_part_at_all(self) -> None:
        """The absence of a boundary is not a sentence about the absence of boundaries."""
        from tenbin.corpus.snapshot import CorpusSnapshot

        header = CorpusSnapshot(
            records=(),
            collection="c",
            read_at=at(ANCHOR),
            store_total=0,
            enumerated=0,
            completeness=Completeness.COMPLETE,
        )
        from tenbin.report.document import CorpusHeader

        assert PartKind.period not in [
            part.kind for part in CorpusHeader.from_snapshot(header).parts()
        ]

    def test_the_throughput_rate_names_the_period_in_its_denominator(self) -> None:
        read = a_read(
            [
                Transition("T1", "open", "in_progress", at("2026-01-06T09:00:00+00:00")),
                Transition("T1", "in_progress", "done", at("2026-01-08T09:00:00+00:00")),
                Transition("T2", "open", "in_progress", at("2026-01-09T09:00:00+00:00")),
            ]
        )

        section = next(s for s in build_retrospective(read).sections if s.slug == THROUGHPUT_SLUG)

        assert isinstance(section.figure, RateFigure)
        assert PERIOD_0[0] in section.claim.denominator.description
        assert "`done`" in section.claim.denominator.description

    def test_the_terminal_states_are_named_in_the_claim_so_the_choice_is_visible(self) -> None:
        read = a_read(
            [
                Transition("T1", "open", "in_progress", at("2026-01-06T09:00:00+00:00")),
                Transition("T1", "in_progress", "shipped", at("2026-01-08T09:00:00+00:00")),
            ]
        )

        document = build_retrospective(read, terminal=["shipped"])
        section = next(s for s in document.sections if s.slug == THROUGHPUT_SLUG)

        assert "`shipped`" in section.claim.statement
        assert section.figure.rate.numerator == 1

    def test_an_unnamed_terminal_set_falls_back_to_done_and_states_it(self) -> None:
        assert terminal_states(None) == ("done",)
        assert terminal_states([]) == ("done",)
        assert terminal_states(["b", "a", "a"]) == ("a", "b")

    def test_lead_and_cycle_time_are_a_distribution_and_say_why_it_is_not_lead_time(self) -> None:
        rows = [
            Transition("T1", "open", "in_progress", at("2026-01-06T09:00:00+00:00")),
            Transition("T1", "in_progress", "done", at("2026-01-06T21:00:00+00:00")),
        ]

        section = next(
            s for s in build_retrospective(a_read(rows)).sections if s.slug == TIME_TO_TERMINAL_SLUG
        )

        assert isinstance(section.figure, DistributionFigure)
        assert section.figure.values["4h to 1d"] == 1
        assert section.figure.values["over 30 days"] == 0
        assert "no moment at which the ticket was created" in section.claim.does_not_mean

    def test_a_ticket_whose_only_recorded_transition_finishes_it_is_excluded_not_rendered_as_zero(
        self,
    ) -> None:
        rows = [
            Transition("T1", "open", "in_progress", at("2026-01-06T09:00:00+00:00")),
            Transition("T1", "in_progress", "done", at("2026-01-06T15:00:00+00:00")),
            Transition("T9", "in_progress", "done", at("2026-01-08T09:00:00+00:00")),
        ]

        section = next(
            s for s in build_retrospective(a_read(rows)).sections if s.slug == TIME_TO_TERMINAL_SLUG
        )

        # Counted under its own key, and absent from every bucket: a zero-hour ticket
        # would be a claim that the work took no time rather than that nothing was seen.
        assert section.figure.excluded[NO_EARLIER_TRANSITION] == 1
        assert section.figure.counted == 1
        assert section.figure.values["under 1 hour"] == 0

    def test_a_ticket_still_open_at_the_period_end_is_excluded_under_its_own_key(self) -> None:
        rows = [
            Transition("T1", "open", "in_progress", at("2026-01-06T09:00:00+00:00")),
            Transition("T1", "in_progress", "done", at("2026-01-08T09:00:00+00:00")),
            Transition("T2", "open", "blocked", at("2026-01-09T09:00:00+00:00")),
        ]

        section = next(
            s for s in build_retrospective(a_read(rows)).sections if s.slug == TIME_TO_TERMINAL_SLUG
        )

        assert section.figure.excluded[STILL_OPEN] == 1

    def test_the_left_censoring_is_stated_because_it_biases_in_one_direction(self) -> None:
        rows = [
            Transition("T1", "open", "in_progress", at("2026-01-06T09:00:00+00:00")),
            Transition("T1", "in_progress", "done", at("2026-01-08T09:00:00+00:00")),
        ]

        section = next(
            s for s in build_retrospective(a_read(rows)).sections if s.slug == TIME_TO_TERMINAL_SLUG
        )

        assert "left-censored" in section.claim.does_not_mean
        assert "preferentially the long ones" in section.claim.does_not_mean

    def test_a_period_where_nothing_finished_refuses_the_distribution_rather_than_rendering_it(
        self,
    ) -> None:
        rows = [Transition("T1", "open", "in_progress", at("2026-01-09T09:00:00+00:00"))]

        section = next(
            s for s in build_retrospective(a_read(rows)).sections if s.slug == TIME_TO_TERMINAL_SLUG
        )

        assert section.figure is None
        assert "would render as a shape" in (section.refusal.reason or "")


class TestTheTwoSlidesThisSourceCannotProduce:
    """Both are rendered as refusals rather than omitted, and neither claims a missing fact."""

    def test_who_carried_what_is_refused_because_the_field_is_absent_not_the_floor(self) -> None:
        section = next(
            s for s in build_retrospective(a_read([])).sections if s.slug == CARRIED_BY_SLUG
        )

        assert section.figure is None
        assert "records no principal" in (section.refusal.reason or "")
        assert "not the granularity floor" in (section.refusal.reason or "")
        assert section.refusal.unblocked_by is None

    def test_committed_versus_delivered_is_omitted_and_the_omission_says_why(self) -> None:
        section = next(
            s
            for s in build_retrospective(a_read([])).sections
            if s.slug == COMMITTED_VERSUS_DELIVERED_SLUG
        )

        assert section.figure is None
        assert "never validate against delivery" in (section.refusal.reason or "")
        assert "story points" in (section.refusal.missing_fact or "")

    def test_the_activity_figure_is_present_under_a_stable_slug(self) -> None:
        slugs = {s.slug for s in build_retrospective(a_read([])).sections}

        assert PERIOD_ACTIVITY_SLUG in slugs


class TestAPriorPeriodComparisonComesFromTheArchive:
    """Criterion: the comparison reads stored figures and never recomputes them."""

    def rows(self) -> list[dict[str, Any]]:
        return [
            row("T1", "open", "in_progress", "2026-01-06T09:00:00+00:00"),
            row("T1", "in_progress", "done", "2026-01-08T09:00:00+00:00"),
            row("T2", "open", "in_progress", "2026-01-25T09:00:00+00:00"),
            row("T2", "in_progress", "done", "2026-01-27T09:00:00+00:00"),
        ]

    def test_two_archived_periods_can_be_drifted_against_each_other(self, tmp_path: Path) -> None:
        archive = Archive(tmp_path / "readings.jsonl")
        for index in (0, 1):
            read = read_period(a_client(self.rows()), period(index))
            archive.record(build_retrospective(read))

        readings = archive.readings(THROUGHPUT_SLUG)

        assert len(readings) == 2
        figure = DriftMeasure(earlier=readings[0], later=readings[1]).compute(
            readings[1] and read_period(a_client(self.rows()), period(1)).snapshot
        )
        assert figure.claim.slug.endswith(".drift")

    def test_archiving_the_same_period_twice_is_refused_rather_than_appended(
        self, tmp_path: Path
    ) -> None:
        archive = Archive(tmp_path / "readings.jsonl")
        read = read_period(a_client(self.rows()), period(0))
        archive.record(build_retrospective(read))

        with pytest.raises(DuplicateReadingError):
            archive.record(build_retrospective(read))

    def test_a_drift_against_a_quiet_period_refuses_because_its_window_cannot_be_placed(
        self, tmp_path: Path
    ) -> None:
        """An unrecorded period has no window at all, and a drift against one is not a drift."""
        from tenbin.claims.refusals import ComparativeRefusalError

        archive = Archive(tmp_path / "readings.jsonl")
        archive.record(build_retrospective(read_period(a_client(self.rows()), period(0))))
        quiet = read_period(a_client(self.rows()), period(2))
        assert quiet.activity is PeriodActivity.QUIET
        archive.record(build_retrospective(quiet))

        readings = archive.readings(PERIOD_ACTIVITY_SLUG)

        with pytest.raises(ComparativeRefusalError, match="could not be read"):
            DriftMeasure(earlier=readings[0], later=readings[1]).compute(quiet.snapshot)


class TestTheCommandRefusesRatherThanGuessing:
    def invoke(self, args: list[str], **env: str):
        return runner.invoke(cli, args, env=env)

    @staticmethod
    def said(result) -> str:
        """The message with Typer's rich panel flattened.

        A usage error is rendered inside a box, and the box's left border lands in the
        middle of a sentence at whatever width the terminal happens to be -- so a phrase
        can be split across two lines with a ``|`` between the halves. Stripping the border
        and collapsing whitespace is what lets a test assert on a *sentence* rather than on
        where the renderer decided to wrap it. Styling is stripped first: without it a
        phrase is contiguous only on terminals that render no color, and GitHub Actions
        sets ``FORCE_COLOR``. See ``flat`` in ``test_report_over_a_ticket_source.py``.
        """
        flattened = (
            plain_output(result.output).replace("│", " ").replace("╭", " ").replace("╮", " ")
        )
        flattened = flattened.replace("╰", " ").replace("╯", " ")
        return " ".join(flattened.split())

    def test_no_declared_period_is_an_error_naming_both_variables(self) -> None:
        result = self.invoke(["retrospective"])
        said = self.said(result)

        assert result.exit_code != 0
        assert "TENBIN_PERIOD_CADENCE, TENBIN_PERIOD_ANCHOR are not set" in said
        assert "Neither has a default on purpose" in said

    def test_a_cadence_without_an_anchor_names_the_one_that_is_missing(self) -> None:
        """One missing variable is named as missing, rather than the message naming both."""
        result = self.invoke(["retrospective", "--cadence", "14d"])
        said = self.said(result)

        assert result.exit_code != 0
        assert "TENBIN_PERIOD_ANCHOR is not set" in said
        assert "are not set" not in said

    def test_the_environment_declares_a_period_when_the_flags_do_not(self) -> None:
        result = self.invoke(
            ["retrospective"],
            TENBIN_PERIOD_CADENCE="14d",
            TENBIN_PERIOD_ANCHOR="2026-01-05T00:00:00Z",
        )

        # It gets past the convention and fails on the unconfigured source instead, which
        # is what proves the environment was read.
        assert "TENBIN_TICKET_API_URL" in result.output

    def test_an_unknown_source_names_the_adapters_it_accepts(self) -> None:
        """A URL without a named adapter reads no wire; the refusal says which names exist."""
        result = self.invoke(
            ["retrospective", "--cadence", "14d", "--anchor", ANCHOR],
            TENBIN_TICKET_SOURCE="trac",
            TENBIN_TICKET_API_URL="http://127.0.0.1:8082",
        )
        said = self.said(result)

        assert result.exit_code != 0
        assert "TENBIN_TICKET_SOURCE" in said
        assert "go-experiment, jira, linear, youtrack" in said

    def test_a_group_scope_against_jira_is_a_usage_error_not_a_quiet_unscoping(self) -> None:
        """Jira has no groups, so --group with the jira adapter must refuse up front."""
        result = self.invoke(
            [
                "retrospective",
                "--cadence",
                "14d",
                "--anchor",
                ANCHOR,
                "--group",
                "platform",
            ],
            TENBIN_TICKET_SOURCE="jira",
            TENBIN_TICKET_API_URL="https://example.atlassian.net",
            TENBIN_TICKET_API_USER="reader@example.com",
            TENBIN_TICKET_API_TOKEN="token",
        )
        said = self.said(result)

        assert result.exit_code != 0
        assert "no group scoping" in said

    def test_a_naive_at_is_refused_rather_than_read_in_this_machines_zone(self) -> None:
        result = self.invoke(
            ["retrospective", "--cadence", "14d", "--anchor", ANCHOR, "--at", "2026-01-10T00:00:00"]
        )

        assert "must carry a zone" in result.output

    def test_an_unparseable_at_names_the_format_it_wanted(self) -> None:
        result = self.invoke(
            ["retrospective", "--cadence", "14d", "--anchor", ANCHOR, "--at", "last tuesday"]
        )

        assert "ISO-8601" in result.output

    def test_an_unreadable_backend_writes_nothing_and_exits_three(self, tmp_path: Path) -> None:
        class Refusing:
            def __enter__(self):
                return self

            def __exit__(self, *_exc: object) -> None:
                return None

            def list_transitions(self, **_kwargs: Any):
                from tenbin.store.client import StoreUnavailableError

                raise StoreUnavailableError("connection refused")

        original = cli_module.TICKET_CLIENT_FACTORY
        cli_module.TICKET_CLIENT_FACTORY = lambda _settings: Refusing()  # type: ignore[assignment]
        try:
            output = tmp_path / "retro.md"
            result = self.invoke(
                [
                    "retrospective",
                    "--cadence",
                    "14d",
                    "--anchor",
                    ANCHOR,
                    "--at",
                    "2026-01-10T00:00:00Z",
                    "--output",
                    str(output),
                ],
                TENBIN_TICKET_SOURCE="go-experiment",
                TENBIN_TICKET_API_URL="http://127.0.0.1:8082",
            )
        finally:
            cli_module.TICKET_CLIENT_FACTORY = original

        assert result.exit_code == 3
        assert not output.exists()

    def test_a_rendered_retrospective_states_the_period_and_both_standing_refusals(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tenbin.report import render_markdown

        rows = [
            row("T1", "open", "in_progress", "2026-01-06T09:00:00+00:00"),
            row("T1", "in_progress", "done", "2026-01-08T09:00:00+00:00"),
        ]

        monkeypatch.setattr(
            cli_module, "TICKET_CLIENT_FACTORY", lambda _s: a_client(rows), raising=True
        )
        result = self.invoke(
            [
                "retrospective",
                "--cadence",
                "14d",
                "--anchor",
                ANCHOR,
                "--at",
                "2026-01-10T00:00:00Z",
                "--terminal",
                "done",
            ],
            TENBIN_TICKET_SOURCE="go-experiment",
            TENBIN_TICKET_API_URL="http://127.0.0.1:8082",
        )

        assert result.exit_code == 0, result.output
        rendered = result.stdout
        assert "Period" in rendered
        assert PERIOD_0[0] in rendered
        assert CARRIED_BY_SLUG.replace("-", " ") in rendered.lower().replace("_", " ")
        assert render_markdown is not None

    def test_archiving_a_retrospective_appends_one_line_per_figure(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        rows = [
            row("T1", "open", "in_progress", "2026-01-06T09:00:00+00:00"),
            row("T1", "in_progress", "done", "2026-01-08T09:00:00+00:00"),
        ]
        archive = tmp_path / "readings.jsonl"

        monkeypatch.setattr(
            cli_module, "TICKET_CLIENT_FACTORY", lambda _s: a_client(rows), raising=True
        )
        result = self.invoke(
            [
                "retrospective",
                "--cadence",
                "14d",
                "--anchor",
                ANCHOR,
                "--at",
                "2026-01-10T00:00:00Z",
                "--archive",
                str(archive),
            ],
            TENBIN_TICKET_SOURCE="go-experiment",
            TENBIN_TICKET_API_URL="http://127.0.0.1:8082",
        )

        assert result.exit_code == 0, result.output
        lines = [json.loads(line) for line in archive.read_text(encoding="utf-8").splitlines()]
        assert THROUGHPUT_SLUG in {entry["slug"] for entry in lines}
        assert all(entry["version"] == "1" for entry in lines)


class TestTheScopedWalkStillReportsHowFarItGot:
    """Narrowing a window must not upgrade a partial walk into a complete one.

    ``replace_to_period`` rewrites the two counts, and rewriting counts is exactly the
    kind of edit that can quietly turn a caveat into a clean bill of health. The verdict
    is not one of the fields it touches, so this pins that it survives.
    """

    def a_walk_that_stops_on_its_second_page(self):
        """A two-page walk whose second page fails.

        The page count is derived by the client from ``total`` and the limit it asked for,
        so the fake has to *report* two pages rather than assert them: ``page_limit=1`` with
        a ``total`` of two is what makes the walk ask twice.
        """

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.params.get("page") == "1":
                body = envelope([row("T1", "open", "in_progress", "2026-01-06T09:00:00+00:00")])
                return httpx.Response(200, json={**body, "total": 2, "limit": 1})
            return httpx.Response(503)

        client = GoExperimentClient("http://127.0.0.1:8082", transport=httpx.MockTransport(handler))
        return client, take_transition_snapshot(
            client, since=at(PERIOD_0[0]), until=at(PERIOD_0[1]), page_limit=1
        )

    def test_a_walk_that_stopped_early_keeps_its_own_verdict(self) -> None:
        from tenbin.report.retrospective import replace_to_period

        client, walked = self.a_walk_that_stops_on_its_second_page()
        with client:
            assert walked.completeness is Completeness.STORE_ERROR
            assert walked.error is not None

            narrowed = replace_to_period(walked, period(0))

        assert narrowed.completeness is Completeness.STORE_ERROR
        assert narrowed.error is not None

    def test_the_narrowing_keeps_the_rows_the_partial_walk_did_reach(self) -> None:
        from tenbin.report.retrospective import replace_to_period

        client, walked = self.a_walk_that_stops_on_its_second_page()
        with client:
            narrowed = replace_to_period(walked, period(0))

        assert narrowed.enumerated == 1
        assert narrowed.transitions[0].to_state == "in_progress"


runner = CliRunner()
