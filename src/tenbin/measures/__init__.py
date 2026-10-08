"""Figures and measures: the layer where a claim and a number become one value.

Everything below this package describes something. Everything above it renders.
This package is the join, and the join is where the arguments of the two packages
below meet: :mod:`tenbin.corpus` supplies a read of the store and says how far it
got, and :mod:`tenbin.claims` supplies the statement, the limits, the falsifier
and the population. A figure here carries both, and the type makes it impossible to
hold one without the other.

**The load-bearing method is :meth:`~tenbin.measures.base.Figure.rate_text`.** It
reads the snapshot a figure was computed over and renders whether the corpus behind
the number is whole, a prefix with a known length, or a walk that failed -- with no
caller-supplied wording, because caller-supplied wording is a caveat a caller can
forget. That is why a figure carries a :class:`~tenbin.corpus.snapshot.CorpusSnapshot`
rather than a boolean somebody remembered to pass: a renderer physically cannot
print a number without holding the object that says whether the number is whole.

**The second load-bearing type is
:class:`~tenbin.measures.base.DistributionFigure`,** whose ``excluded`` mapping is
not optional. A bucket that did not occur and a bucket that was filtered out are the
same integer in ``values`` and completely different facts, and telling a reader
which is which is the difference between a correct report and the ``unknown``-model
bug this whole program exists to have stopped.

**What this package notably does not do:** it does not render, and it does not
decide what to measure. There is no template, no layout and no report here, because
a renderer inside this package could not be prevented from printing a count without
the completeness sentence beside it -- and the absence of a renderer is what makes
:meth:`~tenbin.measures.base.Figure.rate_text` something every renderer must call
rather than something one of them calls. Nor does it judge a record: a measure that
stopped reporting something because the value looked wrong would be grading its own
input, and a corpus layer that did the same would be measuring its own preferences.
The judgement is :mod:`tenbin.claims.gate`'s, and the refusals are
:mod:`tenbin.claims.refusals`'.

Every measure is a pure function of a snapshot: no store, no clock, no randomness.
A test builds a snapshot from a fixture and asserts on the figure, and the absence
of any other way to test one is what makes a measure's correctness checkable at
all.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final

from tenbin.corpus.record import Record
from tenbin.measures.authorship import ChangeAuthorshipMeasure
from tenbin.measures.base import (
    CountFigure,
    DistributionFigure,
    Figure,
    FigureGroup,
    Finding,
    Measure,
    Rate,
    RateFigure,
    duration_bucket,
    duration_buckets,
    histogram,
)
from tenbin.measures.capture_latency import CaptureLatencyMeasure
from tenbin.measures.category import CategoryMeasure
from tenbin.measures.check_gates import CheckGateMeasure, check_gate_measures
from tenbin.measures.check_outcome import CheckOutcomeMeasure
from tenbin.measures.completeness import CompletenessMeasure
from tenbin.measures.coverage import (
    AuthorCoverageMeasure,
    DeclaredModelCoverageMeasure,
    MonthlyCoverageMeasure,
    RepositoryCoverageMeasure,
)
from tenbin.measures.custody import SingleCustodianMeasure
from tenbin.measures.decision_requests import (
    DecisionRequestLifecycleMeasure,
    DecisionRequestOutcomeMeasure,
    QuestionAnsweredRateMeasure,
    SupersededIntervalMeasure,
)
from tenbin.measures.drift import DriftMeasure
from tenbin.measures.filtering import requires_certainty, select
from tenbin.measures.independence_reason import IndependenceReasonMeasure
from tenbin.measures.integrity import (
    CAPTURE_FRESHNESS_CLEAN_SLUG,
    CAPTURE_FRESHNESS_SLUG,
    LIFECYCLE_SHAPE_CLEAN_SLUG,
    LIFECYCLE_SHAPE_SLUG,
    CaptureFreshnessMeasure,
    LifecycleShapeDetector,
    LifecycleShapeMeasure,
)
from tenbin.measures.outcome import ChangeOutcomeMeasure
from tenbin.measures.projection_safety import (
    FORBIDDEN_PROJECTION_KEYS,
    PROJECTION_SAFETY_CLEAN_SLUG,
    PROJECTION_SAFETY_SLUG,
    ProjectionSafetyMeasure,
    QuestionCaptureAgreementMeasure,
)
from tenbin.measures.rationale_comparison import RationaleRelationshipMeasure
from tenbin.measures.review_verdicts import ReviewVerdictMeasure
from tenbin.measures.revision import RevisionIntervalMeasure, RevisionPressureMeasure
from tenbin.measures.staleness import RationaleStalenessMeasure
from tenbin.measures.standing import AuthorAssociationMeasure
from tenbin.measures.tickets import JiraLinkageMeasure
from tenbin.measures.time_to_review import TimeToReviewMeasure
from tenbin.measures.trust import CaptureSource, Independence, TrustProfileMeasure

__all__ = (
    "CAPTURE_FRESHNESS_CLEAN_SLUG",
    "CAPTURE_FRESHNESS_SLUG",
    "LIFECYCLE_SHAPE_CLEAN_SLUG",
    "LIFECYCLE_SHAPE_SLUG",
    "LifecycleShapeMeasure",
    "PROJECTION_SAFETY_CLEAN_SLUG",
    "PROJECTION_SAFETY_SLUG",
    "FORBIDDEN_PROJECTION_KEYS",
    "ProjectionSafetyMeasure",
    "READ_COMPLETENESS_SLUG",
    "default_measures",
    "measures_about_the_capture_system",
    "AuthorAssociationMeasure",
    "CheckGateMeasure",
    "AuthorCoverageMeasure",
    "CaptureFreshnessMeasure",
    "CaptureLatencyMeasure",
    "CaptureSource",
    "CategoryMeasure",
    "ChangeAuthorshipMeasure",
    "ChangeOutcomeMeasure",
    "CheckOutcomeMeasure",
    "CompletenessMeasure",
    "CountFigure",
    "DecisionRequestLifecycleMeasure",
    "DecisionRequestOutcomeMeasure",
    "DriftMeasure",
    "QuestionCaptureAgreementMeasure",
    "DeclaredModelCoverageMeasure",
    "DistributionFigure",
    "Figure",
    "FigureGroup",
    "Finding",
    "Independence",
    "IndependenceReasonMeasure",
    "JiraLinkageMeasure",
    "LifecycleShapeDetector",
    "Measure",
    "MonthlyCoverageMeasure",
    "FORBIDDEN_PROJECTION_KEYS",
    "ProjectionSafetyMeasure",
    "RationaleRelationshipMeasure",
    "RationaleStalenessMeasure",
    "Rate",
    "RateFigure",
    "RepositoryCoverageMeasure",
    "ReviewVerdictMeasure",
    "RevisionIntervalMeasure",
    "RevisionPressureMeasure",
    "SingleCustodianMeasure",
    "SupersededIntervalMeasure",
    "TimeToReviewMeasure",
    "TrustProfileMeasure",
    "duration_bucket",
    "duration_buckets",
    "histogram",
    "requires_certainty",
    "select",
)


#: The slug the read-completeness measure publishes, duplicated here rather than
#: imported from :mod:`tenbin.report` so the dependency keeps running one way.
READ_COMPLETENESS_SLUG: Final[str] = "corpus.read_completeness"


def default_measures(
    records: Iterable[Record] = (), drift: DriftMeasure | None = None
) -> tuple[Measure, ...]:
    """This project's measures, in the order a report should run them.

    **Integrity first, and that ordering is the argument rather than a convention.**
    :func:`tenbin.report.build.build_report` partitions on figure type, so the
    partition is belt and braces -- but a registry written in the wrong order would
    still produce a correct report today and a confusing one the day the partition
    changed. A corpus that opens changes and records no closes is a corpus whose
    remaining figures are describing something other than what a reader will assume,
    and that has to arrive before the numbers rather than after them.

    This function is the entry point :mod:`tenbin.report` resolves by name
    (:data:`tenbin.report.build.PROJECT_MEASURES`). It exists rather than a module
    constant because a registry assembled at import time cannot be extended by a
    caller, and a program whose measure list cannot be extended without editing a
    list in a module docstring's vicinity will grow forks instead.

    **Two of this project's measures are arguments rather than members, and the
    reason is that neither can be constructed from nothing.**

    ``records`` exists because the per-gate check figures are *discovered* from the
    corpus: which gates ran is a fact about the data, so a static list could only be
    a list of the gates somebody already knew about, and a report that omitted an
    unrecognised gate is the failure the check-gate work exists to prevent.
    :func:`check_gate_measures` is called with whatever the read actually holds, and
    with nothing it holds no gates and contributes no figures.

    ``drift`` exists because a drift needs two archived readings, which exist
    outside any read. A registry that built one from nothing would either refuse
    every report or render a figure with no earlier reading to compare against, and
    both are worse than its absence. It is the closing observation when supplied,
    because every other measure here describes one read and this is the only one
    that puts two beside each other.
    """
    measures: tuple[Measure, ...] = (
        LifecycleShapeMeasure(),
        CaptureFreshnessMeasure(),
        CompletenessMeasure(),
        TrustProfileMeasure(),
        AuthorAssociationMeasure(),
        IndependenceReasonMeasure(),
        ReviewVerdictMeasure(),
        TimeToReviewMeasure(),
        ChangeAuthorshipMeasure(),
        ChangeOutcomeMeasure(),
        RepositoryCoverageMeasure(),
        MonthlyCoverageMeasure(),
        AuthorCoverageMeasure(),
        DeclaredModelCoverageMeasure(),
        CategoryMeasure(),
        RevisionPressureMeasure(),
        RevisionIntervalMeasure(),
        SingleCustodianMeasure(),
        RationaleStalenessMeasure(),
        CheckOutcomeMeasure(),
        *check_gate_measures(records),
        CaptureLatencyMeasure(),
        ProjectionSafetyMeasure(),
        QuestionCaptureAgreementMeasure(),
        DecisionRequestLifecycleMeasure(),
        DecisionRequestOutcomeMeasure(),
        SupersededIntervalMeasure(),
        QuestionAnsweredRateMeasure(),
        JiraLinkageMeasure(),
        RationaleRelationshipMeasure(),
    )
    return measures if drift is None else (*measures, drift)


def measures_about_the_capture_system() -> tuple[Measure, ...]:
    """The measures about whether the corpus is what it claims to be.

    Selected by an opt-in flag on the measure rather than by slug, because a check that
    fires and a check that does not report under different claim slugs -- a finding and
    the clean statement it replaced are different claims, and that is the point of them --
    so a slug is not a stable identity for the measure. The flag is, and it is opt-in:
    a measure that does not declare it is about the work rather than the capture system,
    which is the right default and the reason this is a ``getattr`` rather than a member
    of the protocol every measure would then have to remember to set.

    **It lives here rather than in the command line because two surfaces answer the
    question.** ``--integrity-only`` and the MCP report tool both want "the checks, and
    not the descriptions of the work", and a second copy of the selection in either
    place is a second list that can disagree with the first -- silently, because the
    only symptom is a report that quietly stopped running a check.
    """
    return tuple(
        measure
        for measure in default_measures()
        if getattr(measure, "about_the_capture_system", False)
    )
