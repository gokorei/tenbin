"""``tenbin report --source ticket``: the general report over a declared period of flow.

``tenbin retrospective`` renders a period of ticket flow through a bespoke builder. This
file pins the *general* report over the same read, and the four things that had to be true
for that to be safe rather than merely possible.

**The period is declared, never defaulted.** The command has no "since the beginning" and
no inferred cadence, because a figure over an unstated window describes a boundary nobody
chose -- and the refusal happens at the argument parser, where the operator is still
reading.

**The measures are the ticket ones.** Not :func:`default_measures`, whose claims and
denominators are written about a document corpus. ``test_the_measures_are_ticket_shaped``
is the test for the substitution this most invites.

**A refusal that a measure raises survives as a section.** ``TimeInStateMeasure.compute``
raises when a state has no completed interval, and ``build_report`` does not catch it --
so the first draft of this command aborted with a traceback on any period whose states
happened to have no exits. ``test_a_state_with_no_interval_becomes_a_section_not_a_crash``
is that crash, pinned.

**The declared period travels in the header, in both formats.** ``--format json`` used to
drop it entirely, which left a JSON reader holding a span and no way to know it was not
the whole of what was asked for.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from typer.testing import CliRunner

from tenbin import cli
from tenbin.corpus.period import Period, PeriodActivity
from tenbin.corpus.snapshot import Completeness, CorpusSnapshot
from tenbin.corpus.transition import Transition
from tenbin.measures import default_measures
from tenbin.measures.dwell import TimeInStateMeasure
from tenbin.measures.retrospective import PeriodActivityMeasure
from tenbin.report import CorpusHeader, Report
from tenbin.report.json_ import render_json
from tenbin.report.retrospective import PeriodRead
from tenbin.store.transitions import TransitionSnapshot
from tests.conftest import plain_output

runner = CliRunner()

ANCHOR = datetime(2026, 1, 5, tzinfo=UTC)
CADENCE = timedelta(days=7)
READ_AT = datetime(2026, 1, 20, tzinfo=UTC)

PERIOD_ENV = (
    "TENBIN_PERIOD_CADENCE",
    "TENBIN_PERIOD_ANCHOR",
    "TENBIN_TICKET_SOURCE",
    "TENBIN_TICKET_API_URL",
    "TENBIN_TICKET_API_TOKEN",
    "TENBIN_TICKET_API_USER",
)


@pytest.fixture(autouse=True)
def _no_declared_convention(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts from nothing declared and no backend, so refusals are the ones under test."""
    for name in PERIOD_ENV:
        monkeypatch.delenv(name, raising=False)


def flat(text: str) -> str:
    """The output with Typer's rich panel flattened.

    A usage error is rendered inside a box, and the box's left border lands in the middle
    of a sentence at whatever width the terminal happens to be -- so a phrase can be split
    across two lines with a ``|`` between the halves. Stripping the border and collapsing
    whitespace is what lets a test assert on a *sentence* rather than on where the renderer
    decided to wrap it. Styling is stripped first, for the same reason one layer down:
    click highlights ``--options`` inside error messages as two separately-styled spans,
    and Rich reopens the surrounding style after every wrapped line, so without this a
    phrase is contiguous only on terminals that render no color.
    """
    flattened = plain_output(text)
    for border in ("│", "╭", "╮", "╰", "╯"):
        flattened = flattened.replace(border, " ")
    return " ".join(flattened.split())


def period(index: int = 39) -> Period:
    return Period(
        index=index,
        start=ANCHOR + index * CADENCE,
        end=ANCHOR + (index + 1) * CADENCE,
        cadence=CADENCE,
        anchor=ANCHOR,
    )


def a_snapshot(rows: tuple[Transition, ...], **overrides: Any) -> TransitionSnapshot:
    defaults: dict[str, Any] = {
        "transitions": rows,
        "read_at": READ_AT,
        "store_total": len(rows),
        "enumerated": len(rows),
        "completeness": Completeness.COMPLETE,
    }
    return TransitionSnapshot(**{**defaults, **overrides})


def into(state: str, count: int = 2) -> tuple[Transition, ...]:
    """Transitions that all *arrive* in ``state``, so no ticket ever leaves it."""
    return tuple(
        Transition(
            ticket_id=f"T{index}",
            from_state="open",
            to_state=state,
            at=READ_AT - timedelta(days=count - index),
        )
        for index in range(count)
    )


def a_read(
    rows: tuple[Transition, ...],
    *,
    activity: PeriodActivity = PeriodActivity.RECORDED,
    before: int = 0,
) -> PeriodRead:
    return PeriodRead(
        period=period(),
        snapshot=a_snapshot(rows),
        activity=activity,
        recorded_before=before,
    )


class TestThePeriodIsDeclaredAndNeverGuessed:
    """Nothing is defaulted, because a guessed cadence is invisible in the output."""

    def test_a_ticket_source_without_a_cadence_refuses(self) -> None:
        result = runner.invoke(cli.app, ["report", "--source", "ticket"])

        assert result.exit_code != 0
        assert "--cadence" in flat(result.output)
        assert "--anchor" in flat(result.output)

    def test_the_refusal_says_why_there_is_no_default(self) -> None:
        result = runner.invoke(cli.app, ["report", "--source", "ticket"])

        assert "Neither has a default on purpose" in flat(result.output)
        assert "boundary nobody chose" in flat(result.output)


class TestIncoherentCombinationsAreRefusedRatherThanResolved:
    """Each of these is two requests that cannot both be honoured, so neither is honoured."""

    def test_a_kept_read_and_a_ticket_source_are_two_corpora(self, tmp_path: Any) -> None:
        kept = tmp_path / "read.json"
        kept.write_text("{}")

        result = runner.invoke(
            cli.app,
            [
                "report",
                "--source",
                "ticket",
                "--read-model",
                str(kept),
                "--cadence",
                "7d",
                "--anchor",
                ANCHOR.isoformat(),
            ],
        )

        assert result.exit_code != 0
        assert "--read-model" in flat(result.output)
        assert "two different corpora" in flat(result.output)

    def test_integrity_measures_have_no_capture_system_to_describe(self) -> None:
        result = runner.invoke(
            cli.app,
            [
                "report",
                "--source",
                "ticket",
                "--integrity-only",
                "--cadence",
                "7d",
                "--anchor",
                ANCHOR.isoformat(),
            ],
        )

        assert result.exit_code != 0
        assert "capture system" in flat(result.output)

    def test_backend_filters_without_a_backend_are_refused(self) -> None:
        result = runner.invoke(cli.app, ["report", "--project", "some-project"])

        assert result.exit_code != 0
        assert "--project" in flat(result.output)
        assert "no backend to filter against" in flat(result.output)


class TestTheMeasuresAreTheTicketOnes:
    """Per-state dwell, over the states this read saw -- not the Kojutsu set."""

    def test_the_activity_measure_leads_and_names_every_state_observed(self) -> None:
        read = a_read(into("done") + into("in_progress"))

        measures = cli._ticket_measures(read)

        assert isinstance(measures[0], PeriodActivityMeasure)
        assert measures[0].recorded_before == read.recorded_before
        assert [m.state for m in measures[1:]] == ["done", "in_progress"]
        assert all(isinstance(measure, TimeInStateMeasure) for measure in measures[1:])

    def test_no_kojutsu_measure_is_run(self) -> None:
        """A default measure's claim is written about documents, and is false over transitions."""
        measures = cli._ticket_measures(a_read(into("done") + into("in_progress")))

        assert {type(measure).__name__ for measure in measures} == {
            "PeriodActivityMeasure",
            "TimeInStateMeasure",
        }
        assert not any(isinstance(measure, type(default_measures()[0])) for measure in measures)

    def test_an_unrecorded_period_still_reports_what_it_could_not_see(self) -> None:
        read = a_read((), activity=PeriodActivity.UNRECORDED, before=0)

        measures = cli._ticket_measures(read)

        assert len(measures) == 1
        assert isinstance(measures[0], PeriodActivityMeasure)


class TestARefusalStaysASectionRatherThanBecomingACrash:
    """The defect this command shipped with, pinned."""

    def test_a_state_with_no_completed_interval_becomes_a_section(self) -> None:
        read = a_read(into("done"))

        document = cli._ticket_report(read.period, read.snapshot, cli._ticket_measures(read))

        activity, dwell = document.sections
        assert activity.figure is not None
        assert dwell.claim is None
        assert dwell.figure is None
        assert dwell.refusal is not None
        assert "No ticket left `done`" in dwell.refusal.reason
        assert dwell.refusal.missing_fact is not None

    def test_a_measureable_state_is_still_reported_alongside_an_unmeasurable_one(self) -> None:
        rows = (
            into("done")[0],
            Transition(
                ticket_id="moving",
                from_state="in_progress",
                to_state="done",
                at=READ_AT - timedelta(hours=1),
            ),
            Transition(
                ticket_id="moving",
                from_state="done",
                to_state="closed",
                at=READ_AT - timedelta(hours=2),
            ),
        )

        document = cli._ticket_report(
            period(), a_snapshot(rows), cli._ticket_measures(a_read(rows))
        )

        refusals = [section for section in document.sections if section.refusal is not None]
        figures = [section for section in document.sections if section.figure is not None]
        assert refusals, "the state with no exits must still say so"
        assert figures, "and the state that does have intervals must still be reported"


class TestTheDeclaredPeriodTravelsInTheHeader:
    """In both formats, because a JSON reader has no other place to learn the window."""

    def test_the_header_carries_the_period_it_measured(self) -> None:
        read = a_read(into("done"))

        document = cli._ticket_report(read.period, read.snapshot, cli._ticket_measures(read))

        assert document.corpus.period == read.period

    def test_json_carries_the_period_rather_than_dropping_it(self) -> None:
        read = a_read(into("done"))
        document = cli._ticket_report(read.period, read.snapshot, cli._ticket_measures(read))

        corpus = render_json(document)

        assert '"period": {' in corpus
        assert f'"index": {read.period.index}' in corpus
        assert f'"start": "{read.period.start.isoformat()}"' in corpus
        assert f'"end": "{read.period.end.isoformat()}"' in corpus

    def test_a_store_walk_is_not_given_an_invented_period(self) -> None:
        """The field is optional because a whole-collection read has no declared window."""
        header = CorpusHeader.from_snapshot(
            CorpusSnapshot(
                collection="documents",
                records=(),
                read_at=READ_AT,
                store_total=0,
                enumerated=0,
                completeness=Completeness.COMPLETE,
            )
        )

        assert header.period is None
        assert '"period": null' in render_json(Report(corpus=header, sections=()))
