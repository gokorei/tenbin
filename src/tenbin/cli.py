"""The commands, and the flags that deliberately do not exist.

Tenbin is a program people run, and a command line is the easiest place for this
project's discipline to rot. A measure's caveats are in the type; the caveats'
*order* is in one function; the completeness sentence is on the figure. None of
that survives being re-typed into a flag, so this module holds the flags to the
same standard the rest of the package holds itself to: no options that choose a
population this corpus cannot support, and no options that turn a description of one
read into a comparison between two. **The command count is pinned by a test rather
than asserted here**, because a number in a docstring is a number that drifts: the
surface was four commands for most of this program's life and is five now, and the
test in ``tests/test_cli.py`` is what makes adding a sixth a reviewable decision
rather than a diff.

``retrospective`` was the fifth, and it is the first over the second source. Its
flags are the interesting part: ``--cadence`` and ``--anchor`` *declare* a period
that nothing in either system declares, which is why neither has a default and why
neither is a comparison.

**The failure behaviour is the part that matters more than the success
behaviour.** A store that cannot be reached and a store that is empty produce the
same zero figures, and the first of those is a failure while the second is a
finding. So :func:`_read` is the only door to the store, it raises before anything
is rendered, and a failure there exits non-zero with nothing written: a command
that prints an error and *also* leaves an empty report behind has produced the most
dangerous output this program can emit, because an empty report is a document that
looks like an answer. A reachable store holding nothing is a different sentence --
a report that says the corpus is empty, and exits zero.

**There is no comparison flag, and there is no ``--granularity`` offering
``individual``.** Both absences are load-bearing rather than tidy. A comparison
between two populations needs an assignment mechanism, and this corpus has none
(:mod:`tenbin.claims.mechanism`), so a flag shaped like a comparison is a flag
that will be passed by somebody in a hurry and refused by the gate at the far end
-- or worse, not refused, if it is wired to something that quietly reads one group
as the baseline. ``individual`` is refused in the type itself, below the
granularity floor, and a flag that offered it would put an illegal value one
keystroke away from a reader. ``tests/test_cli.py`` walks the command tree and
asserts both, so a flag added later in a hurry is caught by the suite rather than
by a report somebody trusted.

**Two of the commands need no store and no configuration at all.**
:func:`claims` and :func:`refusals` exist so that this program's reasoning can be
inspected without the thing it measures: if you cannot see what it claims and
what it refuses without a running Tanseki and a populated collection, then the
refusal catalogue is a property of a deployment rather than of the program.
``--require-all`` is the one option on ``claims`` and it exists so a refusal is
usable in a check rather than display-only; it exits non-zero whenever the
catalogue is non-empty, which is to say always today, and that is the point -- a
pipeline can watch the count, and the day it drops to zero is a real event.

**What this module notably does not do:** it does not measure, gate, or decide
what may be said -- :mod:`tenbin.report.build` and :mod:`tenbin.claims` own all
three, and this module passes their arguments. It does not write a read model
except through ``read-model build``, because a model that a report refreshes on the
way past is a maintained model, and a maintained model is a source of truth nobody
reconciles (:mod:`tenbin.readmodel`). It does not cache a report, resume an
interrupted one, or retry a failed store read beyond what
:class:`~tenbin.store.client.StoreClient` already does. And it notably does not
run :class:`~tenbin.measures.integrity.LifecycleShapeDetector`: the report layer's
measure protocol requires a figure from every measure, and that detector's honest
answer when a corpus looks sound is ``None``. Adapting it here would mean inventing
an empty figure -- a blank section that reads as an absence of findings -- or
changing the protocol, which is a decision about what a report may contain rather
than about what a command runs.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Final, NoReturn

import typer
from pydantic import ValidationError

from tenbin.claims.gate import ClaimGate
from tenbin.claims.model import Claim
from tenbin.claims.refusals import ComparativeRefusalError, Refusal
from tenbin.claims.registry import default_registry
from tenbin.config import Settings, TicketSettings, settings_from_env, ticket_settings_from_env
from tenbin.corpus.period import Period, PeriodConfig
from tenbin.corpus.snapshot import Completeness, CorpusSnapshot, Snapshot, take_snapshot
from tenbin.measures import (
    Measure,
    default_measures,
    measures_about_the_capture_system,
)
from tenbin.measures.drift import CorpusEvent, DriftMeasure
from tenbin.measures.dwell import TimeInStateMeasure, states_in
from tenbin.measures.retrospective import PeriodActivityMeasure
from tenbin.readmodel import ReadModel, UnknownSchemaVersionError, to_snapshot
from tenbin.readmodel import build as build_read_model
from tenbin.readmodel import load as load_read_model
from tenbin.readmodel import write as write_read_model
from tenbin.readmodel.resume import build_across_windows
from tenbin.report import (
    CorpusHeader,
    Report,
    Section,
    build_report,
    render_json,
    render_markdown,
)
from tenbin.report.archive import Archive
from tenbin.report.html_ import render_html
from tenbin.report.retrospective import PeriodRead, build_retrospective, read_period
from tenbin.store.client import (
    StoreAuthenticationError,
    StoreClient,
    StoreConfigurationError,
    StoreError,
    StoreUnavailableError,
)
from tenbin.store.ticket_sources import (
    TicketConfigurationError,
    TransitionSource,
    ticket_source_client,
)


class ReportSource(StrEnum):
    """Where a report's documents come from, as a closed set.

    An enum rather than a free string for the reason :class:`ReportFormat` is one: a
    source this program does not implement has to be refused by the argument parser
    rather than discovered after the corpus has been walked.

    ``kojutsu`` is the default because it is the store tenbin was built to read, and a
    report with no ``--source`` should describe the corpus the operator was already
    pointing at.
    """

    kojutsu = "kojutsu"
    ticket = "ticket"


class ReportFormat(StrEnum):
    """The ways a rendered report can be delivered, as a closed set.

    An enum rather than a free string because Typer renders an enum as a choice, and
    a format this program does not implement has to be refused by the argument parser
    rather than discovered after the corpus has been walked.
    """

    markdown = "markdown"
    json = "json"
    #: A self-contained page with inline SVG charts. **Not an alternative to the
    #: prose -- it is the prose with the numbers drawn.** Every chart carries the claim,
    #: the negative space, the falsifier and the denominator of the figure it depicts,
    #: because a chart is the most effective device there is for separating a number
    #: from its bound, and a report whose charts can be lifted out of their caveats is
    #: the output this program exists to refuse. No JavaScript and no external asset,
    #: so the file renders its own caveats on a machine with no network.
    html = "html"


#: The command succeeded. Stated as a constant because a bare ``0`` in five
#: ``raise`` sites is five places to change the meaning of success.
EXIT_OK: Final[int] = 0

#: A check that was asked for and is not satisfied. Distinct from a failure, and
#: separate on purpose: ``tenbin claims --require-all`` exits with this *because
#: the program is working*, and a pipeline reading it as a crash would be told to
#: page somebody about a refusal catalogue that is behaving exactly as documented.
EXIT_REFUSED: Final[int] = 1

#: The corpus could not be read at all. One code for unreachable, refused credential
#: and misconfigured URL, because the operator's next step differs and the message
#: says which -- and because all three share the property this module exists to
#: guarantee: nothing was written.
EXIT_STORE_UNREADABLE: Final[int] = 3

#: The environment variable naming the store, quoted in the failure message because
#: "it did not work" is not an action. The value itself is never echoed: the URL is
#: validated precisely where a credential may have been embedded in it.
STORE_URL_VARIABLE: Final[str] = "TENBIN_TANSEKI_URL"

#: Likewise for the credential. A refused key is not retried and not worked around;
#: an operator changes this one line.
STORE_KEY_VARIABLE: Final[str] = "TENBIN_TANSEKI_API_KEY"

#: The ticket source's credential, quoted for the same reason as the two above: a message
#: that says "it did not work" is not an action, and this one is a different variable from
#: the store's, which is the mistake an operator makes when both are misconfigured at once.
TICKET_TOKEN_VARIABLE: Final[str] = "TENBIN_TICKET_API_TOKEN"

#: The two variables that declare a period. Quoted together in one refusal rather than one
#: at a time, because a retrospective needs both and a caller who is told about only the
#: first meets a second error to learn about the second.
PERIOD_CADENCE_VARIABLE: Final[str] = "TENBIN_PERIOD_CADENCE"
PERIOD_ANCHOR_VARIABLE: Final[str] = "TENBIN_PERIOD_ANCHOR"

#: The read that never happened, used only so a measure will hand over its claim
#: without a store. Every denominator a measure states is a count over *this* read,
#: so its size is never rendered from it -- :func:`_denominator_text` prints the
#: description and says the size comes from the read instead. The moment is the
#: epoch, not today, because no figure is ever computed over this snapshot and a
#: plausible timestamp would be a lie in the one field whose whole job is to say
#: when something was read.
NO_READ: Final[CorpusSnapshot] = CorpusSnapshot(
    records=(),
    collection="",
    read_at=datetime(1970, 1, 1, tzinfo=UTC),
    store_total=0,
    enumerated=0,
    completeness=Completeness.COMPLETE,
)

#: What the denominator line says when no read was supplied. A sentence rather than
#: a blank, for the reason every other line in this program is a sentence: a reader
#: who sees an empty field reads it as "the program forgot", and the honest state --
#: "the number is whatever your read turned up" -- is a thing they can act on.
DENOMINATOR_FROM_READ: Final[str] = "the size comes from the read this report is built from"

#: How the command reaches the store, as a module-level name rather than a call to the
#: constructor inside the command. That is the seam a test uses to hand the command a
#: store which answers in process, and it is the reason every test of this file runs
#: with the suite's socket denial still in force: a command that called
#: :class:`StoreClient` itself could only be tested against a real store, and a test
#: that cannot run offline is a test that stops being run. Not ``Final`` for the same
#: reason -- the production value is fixed and the test value is not.
CLIENT_FACTORY: Callable[[Settings], StoreClient] = StoreClient

#: The same seam for the ticket source, and the same reasoning. ``tenbin retrospective``
#: reaches the second source through this name, so a test hands the command a client that
#: answers in process rather than one that opens a socket -- which is the only reason a
#: test of the command can run at all, given that every test in this suite runs with
#: ``socket.connect`` denied. Bound immediately below rather than beside this comment,
#: because a module-level assignment evaluates at import and a function defined further
#: down the file has not been bound yet.
TICKET_CLIENT_FACTORY: Callable[[TicketSettings], TransitionSource]


def _ticket_client_from_settings(settings: TicketSettings) -> TransitionSource:
    """A ticket-source client built from configuration.

    A named function rather than a lambda so the seam above reads as the swap it is, and so
    the settings that reach the second source are visible in one place: the sources fail
    differently and are configured differently, and a client assembled inline
    at the call site would be a second place to look for which variables reach it.
    Which adapter is built is the source's own choice, named by TENBIN_TICKET_SOURCE.
    """
    return ticket_source_client(
        source=settings.ticket_source,
        api_url=settings.ticket_api_url,
        token=settings.ticket_api_token,
        user=settings.ticket_api_user,
        timeout_seconds=settings.ticket_timeout_seconds,
    )


TICKET_CLIENT_FACTORY = _ticket_client_from_settings


#: The measures whose subject is the capture system rather than the work: whether
#: the read reached the end of the store, and what the records say about who was
#: positioned to disagree. ``--integrity-only`` runs these and nothing else, which
#: is the report a reader wants when the question is "can I believe the rest of
#: this?" rather than "what is in the corpus?".
#: The measures whose subject is the capture system rather than the work it captured.
#:
#: A hand-picked subset, and deliberately a *view* of the one registry rather than a
#: second one. This file used to declare its own complete list, which is the shape of
#: defect this project exists to report: two lists of the same thing that nobody
#: reconciles, so a measure added to the registry and not here would be written, tested,
#: and listed by ``tenbin claims`` while never appearing in a report. Deriving both from
#: :func:`tenbin.measures.default_measures` makes that unrepresentable.
#: Published by :mod:`tenbin.measures` and re-exported here, because two surfaces now
#: answer "which measures are about the capture system" -- this command line and the MCP
#: server -- and a second copy of the answer is a second list that can disagree with the
#: first. The selection is made once, over the registry, by both.
_measures_about_the_capture_system = measures_about_the_capture_system


#: There is deliberately no slug set for the above. A check that fires and a check that
#: does not report under different slugs, so any list of slugs would silently drop half
#: the checks depending on which way the corpus came out that day -- a filter whose
#: membership changes with the data is not a filter.


app = typer.Typer(
    name="tenbin",
    help="Measure a Kojutsu corpus, and refuse what it cannot support.",
    no_args_is_help=True,
    add_completion=False,
)

# A sub-application rather than a second top-level command, so the read model has
# exactly one verb -- ``build`` -- and adding one later would be visible in the help as
# an addition rather than as a quietly re-scoped command that sometimes updates.
read_model_app = typer.Typer(
    name="read-model",
    help="Keep a read of the collection, or rebuild it.",
    no_args_is_help=True,
)
app.add_typer(read_model_app, name="read-model")

#: ``--read-model`` is explicit and never implicit. A report that silently preferred a
#: kept read over the store would be describing a corpus from whenever that read
#: happened, and the operator would have no way to know which they were given beyond
#: reading the header's date. So the model is a flag, and the choice sits on the command
#: line where a reader of the invocation can see it.
#:
#: The reasoning lives here rather than in the command's docstring because a docstring
#: here is ``--help`` text, and ``--help`` rewraps a paragraph to a terminal width. A
#: paragraph written to be read at eighty columns comes out of ``--help`` as a staircase,
#: which is a poor way to introduce a program whose entire argument is about legibility.


@app.command()
def report(
    output_format: ReportFormat = typer.Option(
        ReportFormat.markdown,
        "--format",
        help="How to render. Both carry every caveat; JSON keeps them as named keys.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Write here instead of to stdout. Nothing is written when the read fails.",
    ),
    read_model: Path | None = typer.Option(
        None,
        "--read-model",
        help="Measure this kept read instead of walking the store.",
    ),
    integrity_only: bool = typer.Option(
        False,
        "--integrity-only",
        help="Run only the measures about the capture system, not the work it captured.",
    ),
    archive: Path | None = typer.Option(
        None,
        "--archive",
        help=(
            "Append this report's figures to a readings log at this path, one line each. "
            "Append-only: a measure that changes shows up as two readings, never as one "
            "corrected reading."
        ),
    ),
    source: ReportSource = typer.Option(
        ReportSource.kojutsu,
        "--source",
        help="Where the documents come from. 'ticket' reads a declared period of ticket flow.",
    ),
    cadence: str | None = typer.Option(
        None,
        "--cadence",
        help="Period length, e.g. 7d. Required with --source ticket; see --anchor.",
    ),
    anchor: str | None = typer.Option(
        None,
        "--anchor",
        help="When the cadence starts. Required with --source ticket; see --cadence.",
    ),
    at: str | None = typer.Option(
        None,
        "--at",
        help="Which period to measure, as an instant. Defaults to now.",
    ),
    period_index: int | None = typer.Option(
        None,
        "--period",
        help="Which period from the anchor, counting from zero. Overrides --at.",
    ),
    project: str | None = typer.Option(
        None,
        "--project",
        help="Restrict to this project, by name or id. Only with --source ticket.",
    ),
    group: str | None = typer.Option(
        None,
        "--group",
        help="Restrict to this group. Only with --source ticket, and only the go-experiment source.",
    ),
) -> None:
    """Take a read, measure it, and render it with the refusals in place of numbers.

    **``--source ticket`` measures a declared period of ticket status flow, and it
    requires that period to be declared.** ``--cadence`` and ``--anchor`` together name
    the window, or ``TENBIN_PERIOD_CADENCE`` and ``TENBIN_PERIOD_ANCHOR`` do. There is no
    default window and no "since the beginning": the refusal in :func:`_period_config`
    is the same one ``tenbin retrospective`` makes, and a figure over an undeclared
    window describes a boundary nobody chose.

    What this does *not* claim is anything about planned work. A period-scoped read
    enumerates the transitions that happened inside it, and a transition says a ticket
    changed status, not that anything was finished. ``tenbin retrospective`` carries that
    distinction in a renderer built for it; this command renders the general measure set,
    so the denominators it prints are counts of records it enumerated, and each figure
    carries its own ``What it does not mean``.

    The measures are per-state dwell times -- one distribution per observed state -- and
    deliberately not :func:`default_measures`, whose claims and distributions are
    Kojutsu-shaped. A completion rate against a ticket denominator would be a rate
    between two unrelated populations.
    """
    snapshot: Snapshot
    if source is ReportSource.ticket:
        if read_model is not None:
            raise typer.BadParameter(
                "--read-model names a kept read of the store, and --source ticket reads a "
                "period of ticket flow instead. They are two different corpora, and "
                "measuring the one while reporting the other would print a header that "
                "describes neither. Nothing was measured and nothing was written."
            )
        if integrity_only:
            raise typer.BadParameter(
                "--integrity-only runs the measures about the capture system, and a ticket "
                "source has no capture system of its own to be wrong about. Nothing was "
                "measured and nothing was written."
            )
        period = _declared_period(cadence, anchor, at, period_index)
        read = _read_the_period(period, project=project, group=group)
        document = _ticket_report(read.period, read.snapshot, _ticket_measures(read))
    else:
        if project is not None or group is not None:
            raise typer.BadParameter(
                "--project and --group are ticket-backend filters, and a store walk has no "
                "backend to filter against. Pass --source ticket, or drop them. Nothing was "
                "measured and nothing was written."
            )
        snapshot, _from_a_read = _snapshot_for(read_model, require_store=read_model is None)
        measures = _measures_about_the_capture_system() if integrity_only else default_measures()
        document = build_report(
            snapshot, gate=ClaimGate(), registry=default_registry(), measures=measures
        )
    _archive_if_asked(document, archive)
    _emit(_render(document, output_format=output_format), output)


@app.command()
def drift(
    slug: str = typer.Argument(..., help="The measure whose two archived readings to compare."),
    archive: Path = typer.Option(
        ...,
        "--archive",
        help="The readings log. There is no default: an archive nobody can name is not one.",
    ),
    corpus_event: str | None = typer.Option(
        None,
        "--corpus-event",
        help=(
            "A documented change to the corpus separating these two reads -- a migration, a "
            "reinstalled webhook. Supplying one refuses the drift, because two reads that "
            "differ because the corpus did are not drift."
        ),
    ),
    documented_by: str | None = typer.Option(
        None,
        "--documented-by",
        help="Where that event is recorded. Required with --corpus-event.",
    ),
    read_model: Path | None = typer.Option(
        None,
        "--read-model",
        help="Back the figure with this kept read instead of walking the store.",
    ),
    output_format: ReportFormat = typer.Option(
        ReportFormat.markdown, "--format", help="How to render. Both carry every caveat."
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Write here instead of to stdout. Nothing is written when the read fails.",
    ),
) -> None:
    """Render one measure against its own past value, from two archived readings.

    **This command reads the archive and never answers for today.** The archive records
    what was true then; a figure for the present comes from a read of the corpus, which is
    why this takes ``--read-model`` or walks the store rather than reading the log for the
    current value. That read backs the figure and supplies the completeness sentence above
    it, so if the newest reading in the log is from last month the command renders last
    month's drift and says so in the header -- because the header is the read, and the
    drift is not the read.

    **Fewer than two archived readings is a sentence and exit code 1, not a traceback.**
    Run ``tenbin report --archive`` until the log holds two reads of this measure. Nothing
    in the archive can stand in for a read of the corpus, which is the point of it.

    A refusal renders as a refusal and exits 0, like every other refusal in this program:
    incomparable windows, or a pair you have documented as separated by a migration,
    produce the reason and the statement that nothing unblocks it. The programme did what
    it was asked. ``--corpus-event`` is a flag for *refusing* a drift, not for producing
    one: Tenbin cannot observe a webhook reinstall, so it asks rather than infers, and a
    caller who guesses wrong gets the refusal they did not want, which is the right
    direction to be wrong in.
    """
    log = Archive(archive)
    readings = log.readings(slug)
    if len(readings) < 2:
        _fail(
            f"{archive} holds {len(readings)} archived reading(s) of {slug}, and a drift needs "
            "two. Run `tenbin report --archive` until it has held two reads of this measure: "
            "the archive records what was true then and never what is true now, so nothing in "
            "it can stand in for a read of the corpus. Nothing was rendered and nothing was "
            "written.",
            code=EXIT_REFUSED,
        )
    event: CorpusEvent | None = None
    if corpus_event is not None:
        if documented_by is None:
            raise typer.BadParameter(
                "--corpus-event also needs --documented-by; an event nobody can point at is a "
                "guess, and Tenbin cannot observe a webhook reinstall."
            )
        event = CorpusEvent(what=corpus_event, documented_by=documented_by)

    snapshot, _from_a_read = _snapshot_for(read_model, require_store=read_model is None)
    earlier, later = readings[-2], readings[-1]
    try:
        figure = DriftMeasure(earlier=earlier, later=later, corpus_event=event).compute(snapshot)
    except ComparativeRefusalError as exc:
        section = Section(claim=None, figure=None, refusal=exc.refusal)
    else:
        refusal = ClaimGate().check(figure.claim)
        section = (
            Section(claim=None, figure=None, refusal=refusal)
            if refusal is not None
            else Section(claim=figure.claim, figure=figure, refusal=None)
        )
    _emit(
        _render(
            Report(corpus=CorpusHeader.from_snapshot(snapshot), sections=(section,)),
            output_format=output_format,
        ),
        output,
    )


#: There is no incremental option, no ``--since`` and no merge, and the reason is in
#: :mod:`tenbin.readmodel`: an update path is where a derived model becomes a maintained
#: one. The whole model is replaced or nothing is, so a build that fails leaves the
#: previous read exactly as it was and the next run starts from the store again.


@app.command()
def retrospective(
    cadence: str | None = typer.Option(
        None,
        "--cadence",
        help="How long a period is: 7d or 336h. Falls back to TENBIN_PERIOD_CADENCE.",
    ),
    anchor: str | None = typer.Option(
        None,
        "--anchor",
        help=(
            "An ISO-8601 moment with a zone that starts one period: 2026-01-05T00:00:00Z. "
            "Falls back to TENBIN_PERIOD_ANCHOR."
        ),
    ),
    at: str | None = typer.Option(
        None,
        "--at",
        help=(
            "Which moment to find the period containing. An ISO-8601 moment with a zone. "
            "Defaults to now, and the moment used is printed in the header either way."
        ),
    ),
    period_index: int | None = typer.Option(
        None,
        "--period",
        help="Which period, by index from the anchor. Overrides --at. Negative for a period before it.",
    ),
    terminal: list[str] = typer.Option(
        [],
        "--terminal",
        help=(
            "A state that counts as finished. Repeatable. Defaults to `done`, and whichever "
            "set was used is named in the claim of every figure that depends on it."
        ),
    ),
    project: str | None = typer.Option(
        None,
        "--project",
        help="Only this project's transitions. Sent to the source, not filtered here.",
    ),
    group: str | None = typer.Option(
        None,
        "--group",
        help="Only this group's transitions. Go-experiment only; refused by other sources.",
    ),
    output_format: ReportFormat = typer.Option(
        ReportFormat.markdown, "--format", help="How to render. All three carry every caveat."
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Write here instead of to stdout. Nothing is written when the read fails.",
    ),
    archive: Path | None = typer.Option(
        None,
        "--archive",
        help=(
            "Append this period's figures to a readings log at this path, so `tenbin drift` "
            "can compare this period against the one before it. Append-only."
        ),
    ),
) -> None:
    """Render one declared period of ticket flow, with its boundaries stated in the output.

    **A cadence and an anchor are required and neither is defaulted, because nothing in
    either system declares a sprint.** No ticket source Tenbin reads declares a sprint
    length, and a figure computed over calendar months reads exactly like a figure
    computed over sprints while describing something else. A default here would be invisible in every rendered report,
    would imply a sprint length nobody chose, and is precisely the failure this command
    exists to make impossible. Set them with the flags or with ``TENBIN_PERIOD_CADENCE``
    and ``TENBIN_PERIOD_ANCHOR``; undeclared is an error naming both variables.

    **Which period is a parameter, not a default in the same sense.** ``--at`` defaults to
    now and ``--period`` names an index outright, and either way the moment that was used
    is printed in the header -- so the choice is stated rather than silent, which is the
    only thing "no defaults" has to mean for a selector.

    **An empty period says which kind of empty it is.** No transitions in a window whose
    recorder was already running means the work did not move tickets; no transitions and
    none before the window either means the log had not started recording, and says nothing
    about the team. Those are different findings and they render as different ones.

    ``--archive`` is what makes this a retrospective rather than a snapshot: run it once
    per period and ``tenbin drift`` compares two of them, reading the prior period's
    rendered figures out of the append-only log rather than recomputing them.
    """
    period = _declared_period(cadence, anchor, at, period_index)
    read = _read_the_period(period, project=project, group=group)

    document = build_retrospective(read, terminal=terminal or None)
    _archive_if_asked(document, archive)
    _emit(_render(document, output_format=output_format), output)


def _declared_period(
    cadence: str | None, anchor: str | None, at: str | None, period_index: int | None
) -> Period:
    """The period a ticket-source command should measure, or a refusal naming what is missing.

    **Shared, because a period chosen two ways is a population that can disagree with
    itself.** ``tenbin retrospective`` and ``tenbin report --source ticket`` both measure a
    declared window of ticket flow, and the whole reason the period is declared rather than
    defaulted is that a figure over an unstated window describes a boundary nobody chose.
    Two copies of this resolution would be two places for the two commands' populations to
    drift apart, which is the shape of defect this project keeps finding in its own
    registries.
    """
    settings = _ticket_settings()
    config = _period_config(cadence, anchor, settings)
    moment = _moment_for(at)
    return (
        config.period_at(period_index)
        if period_index is not None
        else config.period_containing(moment)
    )


def _read_the_period(period: Period, *, project: str | None, group: str | None) -> PeriodRead:
    """Hold a ticket client for one declared period, and turn any failure into an exit.

    **The one place in the command line that opens a ticket connection.** It was inline in
    ``retrospective`` until a second command needed the same read, and the failure wording
    is the part that must not fork: an operator whose backend is down should be told the
    same thing whichever command they ran, and a second copy of these four branches is four
    more sentences that can drift.

    Exits rather than returning a failure, matching :func:`_walk`: nothing downstream has
    run by the time it exits, so "nothing was measured and nothing was written" is a
    structural property rather than a promise.
    """
    settings = _ticket_settings()
    if not settings.ticket_source_configured:
        raise typer.BadParameter(
            "no ticket source is configured, so there are no status transitions to measure. "
            "Set TENBIN_TICKET_SOURCE to one of go-experiment, jira, linear, youtrack, and "
            "TENBIN_TICKET_API_URL to the source's address -- for example "
            "http://localhost:8082 for a local go-experiment backend. Nothing was "
            "measured and nothing was written."
        )
    try:
        with TICKET_CLIENT_FACTORY(settings) as client:
            return read_period(client, period, project_id=project, group_id=group)
    except TicketConfigurationError as exc:
        raise typer.BadParameter(f"{exc} Nothing was measured and nothing was written.") from exc
    except StoreAuthenticationError as exc:
        _fail(
            f"the ticket source refused the credential ({exc}). Set "
            f"{TICKET_TOKEN_VARIABLE}; this is a configuration problem and retrying will not "
            f"help. Nothing was measured and nothing was written.",
            code=EXIT_STORE_UNREADABLE,
        )
    except StoreUnavailableError as exc:
        _fail(
            f"the ticket source could not be reached ({exc}). The period is not empty -- it "
            f"is unreadable, and a report of zeroes would say otherwise. Nothing was measured "
            f"and nothing was written.",
            code=EXIT_STORE_UNREADABLE,
        )
    except StoreError as exc:
        _fail(
            f"the ticket source could not be read ({exc}). Nothing was measured and nothing "
            f"was written.",
            code=EXIT_STORE_UNREADABLE,
        )


def _ticket_report(period: Period, snapshot: Snapshot, measures: tuple[Measure, ...]) -> Report:
    """Render the ticket measures, keeping a refusal that a measure raises as a section.

    **Not :func:`build_report`, because a ticket measure refuses by raising.**
    ``TimeInStateMeasure.compute`` raises :class:`ComparativeRefusalError` when a state
    has no completed interval in the read -- correctly, since an empty distribution
    rendered as a shape reads as a finding. ``build_report`` calls ``compute`` and expects
    a figure, so a period whose first state happened to have no exits would abort the whole
    command with a traceback.

    The same shape :func:`drift` uses for its single measure, applied to each in turn, so a
    report over a quiet period lists what it could not say and says why, instead of
    dropping the unmeasurable states silently or refusing the whole read. That is the
    distinction that matters here: one state having no intervals is not a reason to withhold
    the states that do have some.
    """
    gate = ClaimGate()
    sections: list[Section] = []
    for measure in measures:
        try:
            figure = measure.compute(snapshot)
        except ComparativeRefusalError as exc:
            sections.append(Section(claim=None, figure=None, refusal=exc.refusal))
        else:
            refusal = gate.check(figure.claim)
            sections.append(
                Section(claim=None, figure=None, refusal=refusal)
                if refusal is not None
                else Section(claim=figure.claim, figure=figure, refusal=None)
            )
    return Report(
        corpus=CorpusHeader.from_snapshot(snapshot, period=period), sections=tuple(sections)
    )


def _ticket_measures(read: PeriodRead) -> tuple[Measure, ...]:
    """The measures a period-scoped ticket read supports, in the order they should run.

    **One dwell distribution per observed state, and nothing else.** Not
    :func:`default_measures`, because those are Kojutsu-shaped: their claims, their
    denominators and their "what would falsify this" sentences are all written about a
    corpus of documents with authors and dates, and running them over a window of status
    changes would print sentences that are individually well-formed and jointly false.

    The state list comes from :func:`states_in` rather than from the backend's lifecycle,
    so a report names the states this read actually saw. A state the lifecycle declares
    but no ticket reached is absent, and its absence is the honest rendering: the report
    does not claim to know how long tickets spend in a state it did not observe.

    **An empty read renders as an empty report, with its own header saying so.** When no
    transition falls inside the period, ``states_in`` is empty and so is this tuple, and
    the render carries the period and its enumeration count with no figures under it. That
    is the same decision ``tenbin retrospective`` makes for a state with no completed
    interval, and for the same reason: an empty distribution rendered as a shape reads as a
    finding, and "nothing happened inside this window" is not a finding about the work, it
    is an absence of evidence about it. The header still states the window, so a reader can
    see the period was real and the emptiness is not a rendering failure.
    """
    return (
        PeriodActivityMeasure(
            period=read.period, activity=read.activity, recorded_before=read.recorded_before
        ),
        *(TimeInStateMeasure(state) for state in states_in(read.snapshot)),
    )


def _period_config(
    cadence: str | None, anchor: str | None, settings: TicketSettings
) -> PeriodConfig:
    """The declared convention, from the flags or the environment, or a refusal naming both.

    Flags first and the environment second, in that order, so an operator overriding a
    configured convention for one run does it the way they would override anything else.

    The refusal names **both** variables rather than the first missing one. A half-named
    error sends somebody to set one variable, run again, and meet a second error naming the
    other -- two round trips to learn a thing the message could have said once, at seven in
    the morning, which is the hour this project cares about.
    """
    cadence_text = (cadence or settings.period_cadence).strip()
    anchor_text = (anchor or settings.period_anchor).strip()
    if not cadence_text or not anchor_text:
        missing = [
            name
            for name, value in (
                (PERIOD_CADENCE_VARIABLE, cadence_text),
                (PERIOD_ANCHOR_VARIABLE, anchor_text),
            )
            if not value
        ]
        raise typer.BadParameter(
            f"a report over ticket flow needs a declared period, and {', '.join(missing)} "
            f"{'is' if len(missing) == 1 else 'are'} not set. A cadence is a whole number of "
            f"days or hours (7d, 336h) and an anchor is an ISO-8601 moment with a zone that "
            f"starts one period (2026-01-05T00:00:00Z). Neither has a default on purpose: "
            f"nothing this program reads declares a sprint length, and a guessed "
            f"cadence would scope every figure to a boundary nobody chose while the rendered "
            f"report said nothing about it. Pass --cadence and --anchor, or set "
            f"{PERIOD_CADENCE_VARIABLE} and {PERIOD_ANCHOR_VARIABLE}."
        )
    try:
        return PeriodConfig.from_text(cadence_text, anchor_text)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _moment_for(at: str | None) -> datetime:
    """The moment to find a period in: the one supplied, or now.

    **The only clock this program reads outside a snapshot's own ``read_at``**, and it is
    here because choosing which period to describe is the one question that cannot be
    answered from a corpus. The moment is printed in the header beside the boundaries it
    produced, so "the current period" is a stated fact in the output rather than an
    assumption -- which is what makes a retrospective reproducible: the same flags with the
    same ``--at`` give byte-identical output, and the default is the only part a reader has
    to be told about.
    """
    if at is None:
        return datetime.now(UTC)
    text = at.strip()
    try:
        moment = datetime.fromisoformat(text)
    except ValueError as exc:
        raise typer.BadParameter(
            f"--at is an ISO-8601 moment with a zone, like 2026-10-01T12:00:00Z, and {text!r} "
            f"is not one ({exc})."
        ) from exc
    if moment.tzinfo is None:
        raise typer.BadParameter(
            f"--at must carry a zone, and {text!r} does not; a moment in no zone would put "
            f"the period's edges in this machine's zone."
        )
    return moment


@read_model_app.command("build")
def build_read_model_command(
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Where to keep the read. Defaults to TENBIN_READ_MODEL_PATH.",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Continue the read already kept at this path instead of starting it again.",
    ),
) -> None:
    """Walk the store, keep the read, and replace the model that was there.

    ``--resume`` is opt-in and the default is a full rebuild, because a full rebuild
    is the thing that is obviously correct and a tool that resumes by default is a
    tool whose output nobody has reasoned about. **Resuming is not updating:** the
    resumed build still replaces the model whole, so a resume that fails partway
    leaves the previous read exactly as a plain build that fails would.
    """
    settings = _settings()
    destination = output if output is not None else _configured_path(settings)
    if destination is None:
        raise typer.BadParameter(
            "no destination for the read model: pass --output or set TENBIN_READ_MODEL_PATH, "
            "because a read model kept somewhere nobody can name is not a cache"
        )
    if resume:
        _resume_read_model(settings, destination)
        return
    snapshot = _walk(settings)
    kept = write_read_model(destination, build_read_model(snapshot))
    # A status line, not a report: one sentence saying what was kept and when it was
    # read, so a rebuild that silently wrote an empty model over a good one is visible
    # in the terminal even when nobody is reading the file.
    _emit_text(
        f"kept {len(snapshot.records):,} records read at {snapshot.read_at.isoformat()} "
        f"({snapshot.completeness.value}) in {kept}",
        output=None,
    )


#: The three prose fields of each claim are the same strings a report renders above each
#: figure, so this command is the program's reasoning rather than a summary of it, and it
#: needs no store. The denominator is the fourth field and the awkward one: its *size* is a
#: fact about a read, so without ``--read-model`` the description is printed and the size
#: is declared read-dependent rather than invented from an empty corpus.


@app.command()
def claims(
    read_model: Path | None = typer.Option(
        None,
        "--read-model",
        help="Take the denominator sizes from this kept read instead of leaving them to the reader.",
    ),
    require_all: bool = typer.Option(
        False,
        "--require-all",
        help="Exit non-zero when anything is refused, so a refusal is usable in a check.",
    ),
) -> None:
    """List every claim this program makes and every measure it refuses."""
    registry = default_registry()
    snapshot, from_a_read = _snapshot_for(read_model, require_store=False)
    lines: list[str] = ["CLAIMS", ""]
    for measure in default_measures():
        claim = measure.claim(snapshot)
        lines.extend(_claim_text(claim, sized=from_a_read))
        lines.append("")
    lines.extend(["REFUSALS", ""])
    lines.extend(_refusal_text(registry.all()))
    _emit_text("\n".join(lines).rstrip("\n"), output=None)
    if require_all and len(registry):
        raise typer.Exit(EXIT_REFUSED)


#: The four fields are the whole of a refusal -- what is missing is a sentence, and the
#: unblocking is either a ticket or the explicit statement that nothing will. Two entries
#: say the latter, including the project's central question, which is refused because no
#: ticket in the Kojutsu project can deliver it. That is the correct outcome rather
#: than a gap in the plan, and it is rendered as a sentence for exactly the reason an empty
#: field is not: a reader who cannot tell "nothing will" from "nobody looked" waits for a
#: ticket that was never going to be filed.


@app.command()
def refusals() -> None:
    """List the refusal registry: what is not measurable, why, and what would change it."""
    lines = _refusal_text(default_registry().all())
    lines.extend(("", "WHAT THESE DO NOT REFUSE", ""))
    lines.extend(
        f"  {line}" for line in ClaimGate().comparison_permitted_without_a_mechanism.splitlines()
    )
    _emit_text("\n".join(lines).rstrip("\n"), output=None)


def _archive_if_asked(document: Report, archive: Path | None) -> None:
    """Append a report's figures to the readings log, if one was named.

    **Archived before anything is emitted, and confirmed on stderr.** The order matters
    because the archive is where the failure should land: a duplicate reading raises after
    the figures exist and before a byte has been rendered, so a run that would have
    published a drift that never happened publishes nothing. And the confirmation goes to
    stderr because this is the one command whose stdout somebody pipes into a file, and a
    status line in front of the report would be its first line.

    Re-running the same read against the same archive is refused rather than appended, and
    that is the design and not an annoyance. A read model has a fixed ``read_at``, so
    ``tenbin report --read-model m --archive a`` run twice exits non-zero and names the
    slug it already holds. That message is the operator's clue that the model, not the
    archive, needs a rebuild.
    """
    if archive is None:
        return
    log = Archive(archive)
    recorded = log.record(document)
    if recorded:
        typer.echo(
            f"archived {len(recorded)} readings at {archive} "
            f"({len(log.slugs())} measure(s) in the log)",
            err=True,
        )


def _refusal_text(refusals: Sequence[Refusal]) -> list[str]:
    """The refusal registry as lines, one blank line between entries.

    The same rendering in both listings, by calling
    :meth:`~tenbin.claims.refusals.Refusal.render_text` rather than restating its
    wording: the sentence a refusal makes in a terminal and the sentence it makes in a
    report have to be the same sentence, or a reader comparing them is comparing two
    arguments rather than one.
    """
    lines: list[str] = []
    for refusal in refusals:
        lines.extend(f"  {line}" for line in refusal.render_text().splitlines())
        lines.append("")
    return lines


def _claim_text(claim: Claim, *, sized: bool) -> list[str]:
    """One claim as a reader would need it, indented under its slug.

    The claim's own four fields, in the order
    :meth:`tenbin.claims.model.Claim.render_text` uses, so that this listing and a
    report are read the same way round. Identity -- slug, kind, granularity -- comes
    first because a reader scanning a list of a dozen claims is looking up one and
    needs to recognise it before reading its prose.
    """
    return [
        f"  {claim.slug}",
        f"    kind: {claim.kind.value}",
        f"    granularity: {claim.granularity.value}",
        f"    statement: {claim.statement}",
        f"    what it does not mean: {claim.does_not_mean}",
        f"    what would falsify it: {claim.falsifier}",
        f"    denominator: {_denominator_text(claim, sized=sized)}",
    ]


def _denominator_text(claim: Claim, *, sized: bool) -> str:
    """The population, with its size only when a read supplied one.

    A size is a count over a read, and :data:`NO_READ` exists to make claims
    available without one. Printing ``(1 records)`` for that would be a fabricated
    number in the one field a reader uses to doubt the rest, so the honest rendering
    names the description and says where the size comes from.
    """
    if sized:
        return claim.denominator.render_text()
    return f"{claim.denominator.description} ({DENOMINATOR_FROM_READ})"


def _snapshot_for(read_model: Path | None, *, require_store: bool) -> tuple[Snapshot, bool]:
    """The read a command should measure, and whether a read supplied its sizes.

    A model that is absent or unreadable falls back to the store -- except that a
    model at a version this program does not implement is *refused* rather than
    ignored, because a version skew is a fact about the world that a human has to
    resolve and a quiet fallback would hide it.

    The second value is the answer to "may a denominator be printed with its size?",
    and it is returned rather than inferred by the caller from the flag it passed: a
    ``--read-model`` naming a file that is not there is a flag with no read behind it,
    and a caller that trusted the flag would print a count over :data:`NO_READ`.
    """
    if read_model is not None:
        model = _load_model(read_model)
        if model is not None:
            return to_snapshot(model), True
    if not require_store:
        return NO_READ, False
    return _walk(_settings()), True


def _resume_read_model(settings: Settings, destination: Path) -> None:
    """Continue the read kept at ``destination``, and say what the merge found.

    **The findings go to the terminal unconditionally and before the write.** A build
    that merged two windows and noticed a document had moved has learned something the
    model cannot carry -- the model keeps the windows and the counts, and loses *which*
    documents shifted -- so a caller that dropped them would keep a model and lose the
    only record of how it was assembled. Printing them is the only place that record
    exists, which is why it is not behind a flag.

    The exit code does not change. A finding is a report, not a failure, and this
    command's job is to keep a read; a resume that found a hole and still kept the
    read has done its job and said so.
    """
    existing = _load_model(destination)
    if existing is None:
        raise typer.BadParameter(
            f"--resume was passed but {destination} holds no read model this program can "
            "continue; run without it to walk the collection from the top."
        )
    try:
        with CLIENT_FACTORY(settings) as client:
            outcome = build_across_windows(client, existing=existing)
    except StoreAuthenticationError as exc:
        _fail(
            f"the store refused the credential ({exc}). This is a configuration problem and "
            f"retrying will not help: set {STORE_KEY_VARIABLE}, or {STORE_URL_VARIABLE} if the "
            "store is elsewhere. Nothing was measured and nothing was written.",
            code=EXIT_STORE_UNREADABLE,
        )
    _emit_text(outcome.render_findings(), output=None)
    kept = write_read_model(destination, outcome.model)
    _emit_text(
        f"resumed from offset {outcome.resumed_from:,} across "
        f"{len(outcome.model.windows)} window(s); kept {outcome.model.enumerated:,} records "
        f"read at {outcome.model.read_at.isoformat()} "
        f"({outcome.model.completeness.value}) in {kept}",
        output=None,
    )


def _load_model(read_model: Path) -> ReadModel | None:
    """A kept read, or ``None`` when there is nothing usable there.

    A malformed model is treated as absent rather than as half a corpus, so the
    fallback is a walk of the store: a report computed from a fresh read is correct
    where a report computed from a partly-read model would have carried a
    completeness sentence saying the walk reached the end of the store.
    """
    try:
        return load_read_model(read_model)
    except UnknownSchemaVersionError as exc:
        _fail(str(exc), code=EXIT_STORE_UNREADABLE)


def _ticket_settings() -> TicketSettings:
    """The ticket-source configuration, which unlike the store's loads with nothing set.

    **No exit code here, and that is the difference from :func:`_settings`.** An
    unconfigured store means this program has nothing to say and must say so by failing; an
    unconfigured *source* means one family of measures is unavailable and the rest are
    unaffected, so the commands that need it check
    :attr:`~tenbin.config.TicketSettings.ticket_source_configured` and refuse in their own
    words. Folding that into a loader would produce a message about a missing variable for
    a condition that is not a misconfiguration.

    A :class:`~pydantic.ValidationError` can still arrive -- an unusable ticket URL, or a
    timeout outside its bounds -- and it is turned into the same operator-shaped exit as the
    store's, because at that point the value *is* wrong rather than absent.
    """
    try:
        return ticket_settings_from_env()
    except ValidationError as exc:
        _fail(
            f"the ticket source is not configured correctly: {_first_line(exc)}. Set "
            f"TENBIN_TICKET_SOURCE to one of go-experiment, jira, linear, youtrack, and "
            f"TENBIN_TICKET_API_URL to the source's address, over TLS unless it targets "
            f"loopback. Nothing was measured and nothing was written.",
            code=EXIT_STORE_UNREADABLE,
        )


def _settings() -> Settings:
    """The configuration, or an exit naming the variable that is missing.

    Unconfigured is a failure here rather than a default, and the message says which
    variable to set. A measurement program with a default store is a measurement
    program that will happily report a real number about the wrong corpus.
    """
    try:
        return settings_from_env()
    except ValidationError as exc:
        _fail(
            f"tenbin is not configured: no store URL. Set {STORE_URL_VARIABLE} to the Tanseki "
            f"endpoint, for example http://localhost:8088/v1. Nothing was measured and "
            f"nothing was written. ({exc.error_count()} configuration problem(s).)",
            code=EXIT_STORE_UNREADABLE,
        )


def _configured_path(settings: Settings) -> Path | None:
    """Where a read model is kept, or ``None`` when the operator named nowhere."""
    configured = settings.read_model_path.strip()
    return Path(configured) if configured else None


def _walk(settings: Settings) -> CorpusSnapshot:
    """Read the whole collection, or exit non-zero having written nothing.

    **The only door to the store, and the reason a failure cannot produce a
    report.** A store that cannot be reached raises rather than returning an empty
    snapshot, and a store that refused the credential is not retried -- so the two
    cases are told apart here, before a :class:`~tenbin.report.document.Report`
    exists. Nothing downstream of this function has run by the time it exits, which
    is why "nothing was written" is a structural property rather than a promise.

    A store that *is* reached and holds nothing is not a failure: it returns an
    empty snapshot, the report says the corpus is empty, and the exit code is zero.
    That is the difference between a finding and a failure, and it is the whole
    reason this function is shaped as it is.
    """
    try:
        with CLIENT_FACTORY(settings) as client:
            return take_snapshot(client)
    except StoreAuthenticationError as exc:
        _fail(
            f"the store refused the credential ({exc}). This is a configuration problem and "
            f"retrying will not help: set {STORE_KEY_VARIABLE}, or {STORE_URL_VARIABLE} if the "
            "store is elsewhere. Nothing was measured and nothing was written.",
            code=EXIT_STORE_UNREADABLE,
        )
    except StoreUnavailableError as exc:
        _fail(
            f"the store could not be reached ({exc}). The corpus is not empty -- it is "
            "unreachable, and a report of zeroes would say otherwise. Nothing was measured "
            "and nothing was written.",
            code=EXIT_STORE_UNREADABLE,
        )
    except (ValidationError, StoreConfigurationError) as exc:
        # Both refusals of the same rule, and both are caught together because the two
        # layers that enforce it would otherwise produce two different failure messages
        # for one mistake: the settings model rejects an unsafe URL on the way in, and
        # the client rejects it again on the way to the socket. The first one is what
        # an operator meets, and a raw validation error is not a message anybody can act
        # on at seven in the morning.
        _fail(
            f"the configured store URL is unusable ({_first_line(exc)}). Set "
            f"{STORE_URL_VARIABLE} to an HTTP(S) URL, over TLS unless it targets loopback. "
            "Nothing was measured and nothing was written.",
            code=EXIT_STORE_UNREADABLE,
        )
    except StoreError as exc:
        # The two classes above are the ones an operator can act on. This one is a
        # walk that stopped for a reason with no better remedy than looking at it, and
        # it is named as its own case rather than folded into "could not be reached".
        _fail(
            f"the store could not be read ({exc}). Nothing was measured and nothing was written.",
            code=EXIT_STORE_UNREADABLE,
        )


def _render(report: Report, *, output_format: ReportFormat) -> str:
    """The rendered report, from the renderers that own their own formatting.

    Two renderers rather than one with a switch, because the JSON payload is not the
    Markdown with the formatting taken out: it keeps the caveats as named keys so
    that the cheapest thing a consumer can do with it is the responsible thing.
    """
    if output_format is ReportFormat.json:
        return render_json(report)
    if output_format is ReportFormat.html:
        return render_html(report)
    return render_markdown(report)


def _emit(rendered: str, output: Path | None) -> None:
    """Deliver a rendered report: to a file if one was named, else to stdout.

    Called only after a read has succeeded, so there is no path through this function
    that writes a file about a corpus that was never read.
    """
    _emit_text(rendered, output=output)


def _emit_text(text: str, *, output: Path | None) -> None:
    """Write text where it was asked for, or print it exactly as rendered.

    ``nl=False`` because the renderers own their own trailing newline: a Markdown
    report ends in one and adding another leaves a blank line in a file, while a JSON
    payload does not and adding one changes a byte a diff would show.
    """
    if output is None:
        typer.echo(text, nl=False)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def _first_line(exc: Exception) -> str:
    """The first line of an exception's message, for a one-line failure report.

    A pydantic ``ValidationError`` prints every field, every constraint and a link to
    its documentation, which is the right thing in a traceback and the wrong thing in a
    message an operator reads once. Nothing here needs more than the sentence the
    validator wrote, and that sentence never contains the value that failed -- which is
    the point of the settings model hiding its input.
    """
    if isinstance(exc, ValidationError):
        details = exc.errors()
        if details:
            sentence = str(details[0].get("msg", "")).removeprefix("Value error, ")
            if sentence:
                return sentence
    return str(exc).splitlines()[0] if str(exc) else type(exc).__name__


def _fail(message: str, *, code: int) -> NoReturn:
    """Report a failure on stderr and leave, having produced nothing.

    A single exit point for every store failure so that the guarantee is one function
    rather than a habit: :func:`_fail` is the only place this program ends a run
    early, and it is only ever called before anything has been rendered or written.
    """
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(code)
