"""A retrospective: one declared period, read, measured, and rendered with its edges stated.

**This does not call :func:`tenbin.report.build.build_report`, and the reason is the
catalogue.** ``build_report`` ends every report with the whole refusal registry -- nine
entries about what the *Kojutsu corpus* cannot support, cross-checked against
``docs/seam.md`` by a test. Those refusals are statements about a different source, and
rendering them inside a retrospective over ticket transitions would put nine sentences
about knowledge capture between a reader and the numbers they asked for, none of which
would be about the period. So the assembly is here, and it reuses what is genuinely
shared: :class:`~tenbin.report.base.Section` for the one-of-two invariant,
:func:`~tenbin.claims.gate.ClaimGate` for what may be rendered, and the same
findings-first discipline :mod:`tenbin.report.build` argues for.

**A refused measure becomes a section here, which ``build_report`` deliberately does not
do.** That module lets a raising measure propagate, on the grounds that a measure which
raises has a defect and a report that hid it would publish a description with a silent
hole in it. That reasoning is right for a whole corpus and wrong for a *window*: a
fourteen-day period in which no ticket left ``in_review`` is not a defect, it is the fact
the reader came for, and :class:`~tenbin.measures.dwell.TimeInStateMeasure` refuses it for
exactly that reason -- an empty distribution renders as a shape and a shape reads as a
finding. So :exc:`~tenbin.claims.refusals.ComparativeRefusalError` is caught here and only
that: it is this program's own deliberate refusal type, so catching it converts a refusal
into content without also converting a ``TypeError`` or an ``AttributeError`` into a
sentence about the corpus.

**The period's boundaries are in the header, not in a section.** They qualify every figure
in the document at once, and :class:`~tenbin.report.document.CorpusHeader` is the one place
a reader looks before trusting any of them. Putting them in a section would make them
something to scroll past, and a retrospective whose edges are below its numbers is a
retrospective whose numbers can be quoted without them.

**What this module notably does not do:** it does not choose the period. It is handed one,
which came from a :class:`~tenbin.corpus.period.PeriodConfig` somebody declared, and the
cadence and anchor are rendered beside the boundaries so the reader can reproduce them. It
does not read a clock, so a retrospective is a pure function of its inputs and the same
period with the same read renders identically -- which is what makes two archived
retrospectives comparable at all. And it does not compare periods: that is
:mod:`tenbin.measures.drift` over two archived readings, and this module's job is to
produce readings worth drifting.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Final

from tenbin.claims.gate import ClaimGate
from tenbin.claims.refusals import ComparativeRefusalError, Refusal
from tenbin.corpus.period import Period, PeriodActivity
from tenbin.corpus.snapshot import Snapshot
from tenbin.measures.base import Finding, Measure
from tenbin.measures.retrospective import (
    PeriodActivityMeasure,
    PeriodThroughputMeasure,
    TimeToTerminalMeasure,
    committed_versus_delivered_refusal,
    terminal_states,
    who_carried_what_refusal,
)
from tenbin.report.base import Section
from tenbin.report.document import CorpusHeader, Report
from tenbin.store.transitions import TransitionSource, take_transition_snapshot

#: The standing refusals a retrospective ends with, in a fixed order. Not the registry --
#: these two are specific to a retrospective over ticket transitions, and the order is
#: "who carried what" before "committed versus delivered" because the first is the slide
#: the audience asks for first and answering it first is what stops the reader assuming
#: the second was merely forgotten.
STANDING_REFUSALS: tuple[Callable[[], Refusal], ...] = (
    who_carried_what_refusal,
    committed_versus_delivered_refusal,
)

#: How many rows the "was anything recorded before this period" probe asks for. One,
#: because the answer is used as a count of *existence* rather than as a population: the
#: figure rendered above states how much earlier history exists from the backend's own
#: ``total``, which the probe's envelope carries without paging. Asking for a page of two
#: hundred to learn a boolean would be the shape of a walk that does not know what it wants.
_RECORDED_BEFORE_PROBE_LIMIT: Final[int] = 1


@dataclass(frozen=True)
class PeriodRead:
    """One period's scoped read, and the verdict on whether it was ever recorded.

    ``snapshot`` is cut to the period by the source rather than filtered afterwards, so
    its ``total`` describes the population actually looked at -- the reason
    every adapter sends the window instead of applying it locally. A period-scoped
    snapshot's ``window`` is derived from the transitions it received, so it is *narrower*
    than the period on a quiet one; the period travels separately on the header for
    exactly that reason.

    ``recorded_before`` is the count that resolves the ambiguity an empty period has. It
    is a count rather than a boolean because the number is free: the probe asks for one
    row, and a reader who wants to know how much history predates the period is asking an
    answerable question.
    """

    period: Period
    snapshot: Snapshot
    activity: PeriodActivity
    recorded_before: int


def read_period(
    client: TransitionSource,
    period: Period,
    *,
    project_id: str | None = None,
    group_id: str | None = None,
) -> PeriodRead:
    """Take the period's read, and settle whether an empty period was quiet or unrecorded.

    **The one function in this module that holds a client, and it is here rather than in
    :func:`build_retrospective` so that the renderer stays a pure function of a read.** The
    split is the same one :func:`tenbin.store.transitions.take_transition_snapshot` makes
    between the walk and the client: the walk takes the client as an argument rather than
    constructing one, which is why every test of it runs with the suite's socket denial
    still in force. A retrospective that opened its own connection could only be tested
    against a live backend, and a test that cannot run offline is a test that stops being
    run.

    **The probe is one extra request and it is what makes an empty period legible.** The
    scoped walk cannot distinguish "nothing happened" from "nothing was recording", because
    both arrive as zero rows, so this asks the backend for anything at or before the
    period's start. Rows there mean the recorder was running and the period was quiet; no
    rows there mean the period predates the log, and every figure a retrospective would
    place in it would be a figure about no population. The probe is asked for *before* the
    period rather than after, because a transition recorded exactly at the boundary belongs
    to the next period -- asking afterwards would make a period that begins the log look
    like one that was recorded throughout.

    The probe shares the period's scoping filters, deliberately: a project's transitions say
    nothing about whether *that project's* recorder was running, and a global history would
    make every project look recorded from the day the backend was installed.

    A failure of the **probe** raises rather than degrading into "unrecorded". That is the
    asymmetry that matters: the scoped walk may return a partial snapshot carrying its
    exception, but a verdict of :attr:`~tenbin.corpus.period.PeriodActivity.UNRECORDED`
    derived from a request that failed would state as a fact about the installation the one
    thing this program knows it does not know.

    **The read is narrowed to the half-open window, because a source's ``until``
    is not necessarily half-open.** The go-experiment backend answers an
    inclusive ``to`` -- a request for ``[start, end]`` returns a transition
    recorded at exactly ``end`` -- which belongs to the *next* period under
    :attr:`~tenbin.corpus.period.Period.contains`. Left alone, that row would sit in two
    consecutive retrospectives at once, each of which would then count a transition the
    other also counted. Jira and Linear narrow to the half-open window inside
    their adapters already; this narrowing applies uniformly anyway, so the
    population a retrospective reports never depends on which adapter read it.

    So the rows are re-filtered to the half-open window and the snapshot's two counts are
    set to the narrowed population. **That is a deliberate claim, not a reconciliation of a
    discrepancy**: the population this report is about is the transitions inside a
    half-open window, it was enumerated in full, and its size is what is stated. The
    backend's count for the inclusive window it was asked about is larger by exactly the
    boundary rows, and that difference is an artefact of the bound's semantics rather than
    anything unread -- which is why it is corrected here, with the reason written down,
    rather than left to render as a shortfall in the header. Nothing is dropped silently:
    the rows removed are on the far side of a declared boundary and belong to the period
    that declares them.
    """
    walked = take_transition_snapshot(
        client,
        project_id=project_id,
        group_id=group_id,
        since=period.start,
        until=period.end,
    )
    snapshot = replace_to_period(walked, period)
    recorded_before = client.list_transitions(
        project_id=project_id,
        group_id=group_id,
        until=period.start,
        page=1,
        limit=_RECORDED_BEFORE_PROBE_LIMIT,
    ).total

    activity = _activity_for(len(snapshot.transitions), recorded_before)
    return PeriodRead(
        period=period,
        snapshot=snapshot,
        activity=activity,
        recorded_before=recorded_before,
    )


def replace_to_period(walked: Snapshot, period: Period) -> Snapshot:
    """The walked read, narrowed to the half-open window the period declares.

    Named for what it returns rather than hidden as a line inside :func:`read_period`,
    because it is a **claim about the population** and deserves to be checkable on its
    own: a caller can hand this function a walk and a period and get back a read whose rows
    and counts agree with each other, which is the property the header depends on.

    The counts are set to the narrowed rows rather than left as the backend reported them,
    and the reasoning is in :func:`read_period`: the backend's ``to`` is inclusive, so its
    count describes a slightly larger window than the one this report is about, and
    carrying that count forward would render a known boundary artefact as though the walk
    had missed records. Both counts move together, so the header reads "N of N" and means
    it -- the enumeration was complete for the window it names.

    ``window`` is left alone rather than recomputed here. The snapshot derives it in its own
    ``__post_init__``, and a second derivation in this function would be a second place for
    the span and the rows to disagree.
    """
    kept = tuple(row for row in getattr(walked, "transitions", ()) if period.contains(row.at))
    return replace(walked, transitions=kept, enumerated=len(kept), store_total=len(kept))


def _activity_for(observed: int, recorded_before: int) -> PeriodActivity:
    """Which of the three states a period with this many rows and this much history is in.

    A period with rows is :attr:`PeriodActivity.RECORDED` whatever came before it --
    including a period that *begins* the log, which is recorded from its first row even
    though nothing precedes it. The other two are distinguished by history alone, because
    an empty read is the ambiguous case and the history is the only thing that resolves it.
    """
    if observed:
        return PeriodActivity.RECORDED
    if recorded_before:
        return PeriodActivity.QUIET
    return PeriodActivity.UNRECORDED


def retrospective_measures(
    read: PeriodRead,
    *,
    terminal: Sequence[str] | None = None,
) -> tuple[Measure, ...]:
    """The measures a retrospective runs, in the order it wants them considered.

    The activity measure leads because its ``RECORDED`` case is a count the others are
    populations of, and its other two cases are findings that
    :func:`~tenbin.report.retrospective.build_retrospective` hoists to the front
    regardless. The two headline figures follow, and the per-state dwell measures come
    last because they are the ones a reader comes back to rather than the ones they open
    with.

    Dwell measures are built for exactly the states this read observed, via
    :func:`tenbin.measures.dwell.states_in`. One per observed state rather than a pooled
    figure, for the reason that module gives: a ticket blocked for a week and a ticket in
    progress for a week have been in different situations, and pooling them produces a
    distribution whose shape is a property of the mix.
    """
    from tenbin.measures.dwell import TimeInStateMeasure, states_in

    states = terminal_states(terminal)
    return (
        PeriodActivityMeasure(
            period=read.period,
            activity=read.activity,
            recorded_before=read.recorded_before,
        ),
        PeriodThroughputMeasure(period=read.period, terminal=states),
        TimeToTerminalMeasure(period=read.period, terminal=states),
        *(TimeInStateMeasure(state) for state in states_in(read.snapshot)),
    )


def build_retrospective(
    read: PeriodRead,
    *,
    gate: ClaimGate | None = None,
    terminal: Sequence[str] | None = None,
) -> Report:
    """Render one period: its edges in the header, its findings first, its refusals last.

    Purity for the reason :func:`tenbin.report.build.build_report` is pure: no clock, no
    client, no environment. The period arrived already chosen and the read already taken,
    so the same three inputs render byte-identically every time, which is what lets an
    archived retrospective be compared against a later one rather than merely re-read.

    ``gate`` defaults to a gate with no assignment mechanism, which refuses every
    comparative claim. That is the same default the rest of the program uses and for the
    same reason: a caller who wanted a comparison to render would have to construct the
    mechanism deliberately, and a retrospective over ticket transitions has no mechanism to
    construct.
    """
    the_gate = gate if gate is not None else ClaimGate()
    figures = [
        _figure_or_refusal(measure, read.snapshot, gate=the_gate)
        for measure in retrospective_measures(read, terminal=terminal)
    ]
    sections = _findings_first(figures)
    # Built once and filtered by slug, because `Report` refuses two sections for one claim
    # and a measure that ever produced one of these slugs would otherwise collide with the
    # standing refusal rather than replace it.
    standing = [builder() for builder in STANDING_REFUSALS]
    rendered = {section.slug for section in sections}
    sections.extend(
        _refusal_section(refusal) for refusal in standing if refusal.claim_slug not in rendered
    )
    header = CorpusHeader.from_snapshot(read.snapshot, period=read.period)
    return Report(corpus=header, sections=tuple(sections))


def _figure_or_refusal(measure: Measure, snapshot: Snapshot, *, gate: ClaimGate) -> Section:
    """One measure becomes one section: its figure, or the refusal that replaces it.

    Three exits and they are tried in the order a reader would want them. A measure that
    refuses becomes the refusal, because in a period-scoped read that is a fact about the
    window rather than a defect (see the module docstring). Then the gate, which refuses a
    claim below the granularity floor -- unreachable for the measures here, since they all
    declare ``corpus``, and kept anyway so that a measure added to this report later is
    gated without anyone remembering to. Then the figure.
    """
    try:
        figure = measure.compute(snapshot)
    except ComparativeRefusalError as exc:
        return Section(claim=None, figure=None, refusal=exc.refusal)
    refusal = gate.check(figure.claim)
    if refusal is not None:
        return Section(claim=None, figure=None, refusal=refusal)
    return Section(claim=figure.claim, figure=figure, refusal=None)


def _findings_first(sections: Sequence[Section]) -> list[Section]:
    """Findings to the front, everything else in the order the measures were declared.

    The same rule :func:`tenbin.report.build._integrity_first` applies, for the same
    reason: "this period was never recorded" is a statement about whether the population
    exists, and every number below it is computed from that population.

    Partitioned by identity rather than by ``in``, because :class:`Section` compares by
    value and two measures that legitimately produced equal sections would then be filed
    as one -- which would move the second below the findings and quietly drop it from the
    count of rendered sections.
    """
    findings = [
        section
        for section in sections
        if section.figure is not None and isinstance(section.figure, Finding)
    ]
    hoisted = {id(section) for section in findings}
    return findings + [section for section in sections if id(section) not in hoisted]


def _refusal_section(refusal: Refusal) -> Section:
    """A refusal as a section: no claim beside it, because the claim is what was refused."""
    return Section(claim=None, figure=None, refusal=refusal)
