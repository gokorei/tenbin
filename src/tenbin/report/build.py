"""Assembling a report: run the measures, gate every claim, and refuse in place of raising.

Three decisions are made here and nowhere else, and each of them is about order.

**A finding comes before the numbers computed from the corpus it is about.** A
finding says that something in the corpus does not look like what an honest system
produces, and every number below it is computed from that corpus. So a
:class:`~tenbin.measures.base.Finding` is rendered first whatever order the measure
registry was written in, and this is not a preference: a report that leads with a
distribution and mentions the broken shape in the appendix has given the reader a
number to carry away before telling them the number is derived from something suspect.

**The completeness figure comes next, before every other number.** Every other figure
in this program is conditional on it, and the condition is invisible in the numbers
themselves -- a distribution over four hundred documents looks exactly like a
distribution over four thousand. So it is second, identified by the slug the measures
package publishes for it rather than by position in a list, which means the rule
survives somebody adding a measure to the front of the registry.

**The refusals come last, and they come last in the catalogue's own order.** Not first:
a reader who came for a number should not have to read nine refusals to reach it, and
the header has already told them what corpus they are looking at. And because the
catalogue's last entry is this project's central question, ending the report with the
refusal of the question the corpus cannot answer is the honest ending rather than an
accident of placement.

**A refusal is content, so it is a section rather than an exception.** Every claim a
measure produced goes through :meth:`~tenbin.claims.gate.ClaimGate.check`, and a
claim that comes back refused becomes a rendered section: the figure is dropped, since
the figure is the thing being refused, and the refusal takes its place. A report that
failed to build because one claim was refused would be a program that can describe a
corpus only when the corpus supports everything asked of it -- and the measures this
corpus cannot support are most of the interesting ones. The catalogue is rendered the
same way, once per entry, because a report that listed the first refusal and stopped
would read as though the other eight were permitted.

**What this module notably does not do:** it does not render, and it holds no opinion
about how a claim reads -- :mod:`tenbin.report.ordering` owns both. It does not read
records, decide what is worth measuring, or compute anything itself: a measure is
called and its figure is taken as it came. It does not catch an exception from a
measure, because a measure that raises has a defect and a report that hid it would
publish a corpus description with a silent hole in it. Nor does it repair a refusal:
a claim slug the registry refuses is rendered as *that* refusal, whatever the measure
thought it was computing, because a catalogue entry that only appears in an appendix
cannot stop the number printed above it.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence

from tenbin.claims.gate import ClaimGate
from tenbin.claims.registry import RefusalRegistry
from tenbin.corpus.snapshot import Snapshot
from tenbin.measures.base import Figure, Finding, Measure
from tenbin.measures.completeness import READ_COMPLETENESS_SLUG
from tenbin.report.base import Section
from tenbin.report.document import CorpusHeader, Report

#: The entry point this layer expects :mod:`tenbin.measures` to publish, and the only
#: place the report layer names that package. Resolved by attribute rather than by a
#: module-scope ``import`` for two reasons that are really one: the dependency runs
#: one way (measures know about claims and corpus, the report layer calls them and
#: never the reverse), and a renderer asked to print a report about refusals should not
#: have to import every measure in order to do it. If the entry point moves, this is the
#: one line to change and the failure is an ``AttributeError`` naming it.
PROJECT_MEASURES = "default_measures"


def build_report(
    snapshot: Snapshot,
    *,
    gate: ClaimGate,
    registry: RefusalRegistry,
    measures: Sequence[Measure] | None = None,
) -> Report:
    """Run the measures over one read, gate what they produced, and render the refusals.

    Pure in everything but its inputs: no clock, no store, no environment, and the same
    snapshot with the same measures gives the same report. That is what lets a rendered
    report be checked against the value it was rendered from, and it is why the moment
    in the header comes from the snapshot rather than from ``now()`` -- a report whose
    header said "today" could not be reproduced and a reader could not tell which read
    it was describing.

    ``measures`` defaults to this project's own measures and exists as a parameter so
    that a caller -- or a test -- can build a report over a set of measures it chose,
    including none. That is how the report layer is exercised without a corpus: a test
    passes its own measures and gets a report out of a snapshot it built by hand.
    """
    figures = [measure.compute(snapshot) for measure in _measures(measures, snapshot)]
    sections: list[Section] = [
        _section_for(figure, gate=gate, registry=registry) for figure in _integrity_first(figures)
    ]
    rendered = {section.slug for section in sections}
    sections.extend(
        Section(claim=None, figure=None, refusal=refusal)
        for refusal in registry.all()
        if refusal.claim_slug not in rendered
    )
    return Report(corpus=CorpusHeader.from_snapshot(snapshot), sections=tuple(sections))


def _measures(measures: Sequence[Measure] | None, snapshot: Snapshot) -> Sequence[Measure]:
    """The measures to run: the caller's, or this project's own.

    See :data:`PROJECT_MEASURES` for why the lookup is done here rather than at module
    scope. The result is materialised into a tuple so that a caller passing a generator
    gets it run exactly once, whether or not a second pass is ever needed.

    **The snapshot arrives rather than its records, and the difference is not
    cosmetic.** This used to take ``records`` as an argument, which meant the caller
    wrote ``snapshot.records`` at the call site and Python evaluated it before this
    function ran -- so a caller who passed their own measures, and needed no records
    at all, still raised ``AttributeError`` on any read that had none. A read over
    ticket status transitions has no records and cannot have any: ``RecordKind`` is a
    closed five-value vocabulary and a ticket moving from ``ready_for_review`` to
    ``done`` fits none of its members. Reaching for records here, on the one branch
    that wants them, is what lets a second source become a report.

    **A read with no records and no caller-supplied measures refuses, rather than
    rendering an empty report.** The project measures are Kojutsu-shaped: they
    filter on record kind, so with nothing to filter they have no honest default. An
    empty report would render as a finding, and a silent one at that.
    """
    if measures is not None:
        return tuple(measures)
    records = getattr(snapshot, "records", None)
    if records is None:
        raise ValueError(
            "This read carries no records, so the project measures cannot be chosen: "
            "they filter on record kind and there is nothing here to filter. Pass "
            f"`measures=` explicitly for a {snapshot.collection} read."
        )
    module = importlib.import_module("tenbin.measures")
    return tuple(getattr(module, PROJECT_MEASURES)(records))


def _integrity_first(figures: Sequence[Figure]) -> tuple[Figure, ...]:
    """Findings first, then the completeness figure, then everything else as declared.

    A stable partition rather than a sort by key, so that within each group the order
    the registry was written in survives untouched. A sort would be tidier and would
    quietly make a measure's position in the registry meaningless, which is a decision
    nobody would then be able to point at.

    The completeness figure is identified by the slug :mod:`tenbin.measures`
    publishes for it rather than by its position, because the position is not stable:
    it is whatever the registry happens to say today.
    """
    findings = tuple(figure for figure in figures if isinstance(figure, Finding))
    completeness = tuple(
        figure for figure in figures if figure.claim.slug == READ_COMPLETENESS_SLUG
    )
    rest = tuple(
        figure
        for figure in figures
        if not isinstance(figure, Finding) and figure.claim.slug != READ_COMPLETENESS_SLUG
    )
    return findings + completeness + rest


def _section_for(
    figure: Figure,
    *,
    gate: ClaimGate,
    registry: RefusalRegistry,
) -> Section:
    """One figure becomes one section, or the refusal that replaces it.

    Three ways to end up holding a refusal, and they are tried in the order a reader
    would want to hear them. The catalogue first: a claim whose slug the registry
    refuses is rendered as that refusal, because a measure that computed something for
    a measure the corpus forecloses is the precise defect the catalogue exists to
    prevent, and a report that printed the number and the refusal side by side would
    have told the reader to average them. Then the gate, which is what refuses a
    comparative claim or one below the granularity floor. Then the figure itself,
    which is the only outcome that is not a refusal.
    """
    claim = figure.claim
    if claim.slug in registry:
        return Section(claim=None, figure=None, refusal=registry.get(claim.slug))
    refusal = gate.check(claim)
    if refusal is not None:
        return Section(claim=None, figure=None, refusal=refusal)
    return Section(claim=claim, figure=figure, refusal=None)
