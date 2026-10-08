"""The report layer over a read that is not a document corpus.

A second read seam — ticket status transitions — carries no records, because
``RecordKind`` is a closed five-value vocabulary and a ticket moving from
``ready_for_review`` to ``done`` fits none of its members. These tests pin the seam
that lets it become a report anyway, and the three things that had to be refused
rather than worked around for it to be safe.

The failure this file exists to prevent: ``build_report`` evaluating
``snapshot.records`` eagerly, so a caller who passed their own measures and needed
no records at all still crashed on a read that had none. That was not a hypothetical —
it is what the code did, and the fix is four lines. The test for it is
``test_a_second_source_report_needs_no_records_at_all``.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from tenbin.claims.gate import ClaimGate
from tenbin.claims.refusals import ComparativeRefusalError
from tenbin.claims.registry import RefusalRegistry
from tenbin.corpus.snapshot import Completeness
from tenbin.corpus.transition import Transition
from tenbin.measures.drift import DriftMeasure
from tenbin.measures.dwell import TimeInStateMeasure
from tenbin.report.archive import Archive, DuplicateReadingError
from tenbin.report.build import build_report
from tenbin.store.transitions import TransitionSnapshot

READ_AT = datetime(2026, 9, 30, tzinfo=UTC)


def _rows() -> tuple[Transition, ...]:
    return (
        Transition(
            ticket_id="A",
            from_state="open",
            to_state="in_progress",
            at=datetime(2026, 9, 1, tzinfo=UTC),
        ),
        Transition(
            ticket_id="A",
            from_state="in_progress",
            to_state="done",
            at=datetime(2026, 9, 2, tzinfo=UTC),
        ),
    )


def _snapshot(
    rows: tuple[Transition, ...] | None = None,
    read_at: datetime = READ_AT,
) -> TransitionSnapshot:
    rows = _rows() if rows is None else rows
    return TransitionSnapshot(
        transitions=tuple(rows),
        read_at=read_at,
        store_total=len(rows),
        enumerated=len(rows),
        completeness=Completeness.COMPLETE,
    )


def _report(snapshot: TransitionSnapshot, measures=None):
    return build_report(
        snapshot,
        gate=ClaimGate(),
        registry=RefusalRegistry(),
        measures=measures if measures is not None else [TimeInStateMeasure("in_progress")],
    )


class TestASecondSourceBecomesAReport:
    def test_a_second_source_report_needs_no_records_at_all(self) -> None:
        """The eager-evaluation defect, pinned.

        `snapshot.records` used to be written at the call site, so Python evaluated
        it before `_measures` could see that the caller had passed their own
        measures. A read with no records therefore raised `AttributeError` for a
        caller that needed no records. `TransitionSnapshot` has no `records`
        attribute at all, so this test fails against the old code.
        """
        assert not hasattr(_snapshot(), "records")
        report = _report(_snapshot())
        assert report.sections

    def test_the_header_renders_from_the_protocol_alone(self) -> None:
        """Every attribute `CorpusHeader.from_snapshot` reads is protocol-wide."""
        report = _report(_snapshot())
        assert report.corpus.collection == "status-transitions"
        assert report.corpus.enumerated == 2
        assert report.corpus.completeness is Completeness.COMPLETE
        assert report.corpus.window.earliest == datetime(2026, 9, 1, tzinfo=UTC)

    def test_the_figure_carries_its_claim_and_its_completeness_sentence(self) -> None:
        report = _report(_snapshot())
        figure = report.sections[0].figure
        assert figure is not None
        assert "in progress" in figure.title
        assert "Whole read" in figure.rate_text()
        assert "status-transitions" in figure.claim.denominator.description

    def test_a_measure_that_cannot_support_its_claim_refuses_rather_than_rendering_empty(
        self,
    ) -> None:
        """A refusal the measure *discovers* propagates; it does not become a section.

        Only refusals the catalogue already knows become sections. A measure that runs
        and discovers it has no intervals raises, because an empty distribution renders
        as a shape and a shape reads as a finding. The distinction matters across a
        second source: ``blocked`` is absent from two rows for an ordinary reason, but a
        read that enumerated nothing at all would refuse it for the same reason, and only
        the completeness sentence distinguishes those.
        """
        with pytest.raises(ComparativeRefusalError) as excinfo:
            _report(_snapshot(), measures=[TimeInStateMeasure("blocked")])
        message = str(excinfo.value)
        assert "blocked" in message
        assert "render as a shape" in message


class TestProjectMeasuresRefuseASecondSource:
    def test_choosing_the_project_measures_refuses_with_a_sentence_naming_the_fix(self) -> None:
        """They filter on record kind, so with nothing to filter they have no default.

        An empty report here would render as a finding, and a silent one.
        """
        with pytest.raises(ValueError) as excinfo:
            build_report(_snapshot(), gate=ClaimGate(), registry=RefusalRegistry())
        message = str(excinfo.value)
        assert "no records" in message
        assert "measures=" in message
        assert "status-transitions" in message

    def test_the_refusal_does_not_mention_documents(self) -> None:
        """A refusal about a second source must not describe the problem in the first one's terms."""
        with pytest.raises(ValueError) as excinfo:
            build_report(_snapshot(), gate=ClaimGate(), registry=RefusalRegistry())
        assert "document" not in str(excinfo.value).lower()


class TestArchivingASecondSource:
    def test_a_second_source_report_is_archivable(self, tmp_path) -> None:
        archive = Archive(tmp_path / "a.jsonl")
        written = archive.record(_report(_snapshot()))
        assert written
        assert {reading.collection for reading in written} == {"status-transitions"}

    def test_readings_of_a_slug_in_one_collection_return_them(self, tmp_path) -> None:
        archive = Archive(tmp_path / "a.jsonl")
        archive.record(_report(_snapshot()))
        found = archive.readings("time-in-state-in-progress", collection="status-transitions")
        assert len(found) == 1

    def test_the_same_slug_in_two_collections_does_not_collide(self, tmp_path) -> None:
        """Two sources, one slug, one instant: two readings, not a duplicate.

        `record`'s key used to be `(slug, read_at)`, so a second source's first
        reading at the same moment would be refused as a re-run of the first. That
        refusal is correct for one source and wrong here.
        """
        report = _report(_snapshot())
        other = _report(_snapshot())
        # Same figures, but stamped from a different read.
        object.__setattr__(other.corpus, "collection", "other-source")

        archive = Archive(tmp_path / "a.jsonl")
        first = archive.record(report)
        second = archive.record(other)
        assert len(first) == len(second)
        assert {r.collection for r in first} == {"status-transitions"}
        assert {r.collection for r in second} == {"other-source"}

    def test_a_duplicate_within_one_collection_is_still_refused(self, tmp_path) -> None:
        """Widening the key must not have weakened it."""
        report = _report(_snapshot())
        archive = Archive(tmp_path / "a.jsonl")
        archive.record(report)
        with pytest.raises(DuplicateReadingError):
            archive.record(report)

    def test_the_duplicate_error_names_the_collection(self, tmp_path) -> None:
        report = _report(_snapshot())
        archive = Archive(tmp_path / "a.jsonl")
        archive.record(report)
        with pytest.raises(DuplicateReadingError) as excinfo:
            archive.record(report)
        assert "status-transitions" in str(excinfo.value)

    def test_readings_refuse_to_mix_two_collections_rather_than_returning_the_union(
        self, tmp_path
    ) -> None:
        """A drift over the union would compare two different things.

        The figures are stored as rendered text and never parsed back, so nothing
        downstream can detect the category error. Scoping has to happen where the
        collection is still known.
        """
        report = _report(_snapshot())
        other = _report(_snapshot())
        object.__setattr__(other.corpus, "collection", "other-source")

        archive = Archive(tmp_path / "a.jsonl")
        archive.record(report)
        archive.record(other)

        with pytest.raises(ValueError) as excinfo:
            archive.readings("time-in-state-in-progress")
        message = str(excinfo.value)
        assert "more than one collection" in message
        assert "status-transitions" in message and "other-source" in message

    def test_scoping_resolves_the_single_collection_case_without_being_asked(
        self, tmp_path
    ) -> None:
        """One collection is the whole of the single-source case, and must keep working."""
        archive = Archive(tmp_path / "a.jsonl")
        archive.record(_report(_snapshot()))
        assert archive.readings("time-in-state-in-progress")

    def test_a_slug_naming_no_collection_reads_as_empty_rather_than_raising(self, tmp_path) -> None:
        archive = Archive(tmp_path / "a.jsonl")
        assert archive.readings("no-such-measure") == ()


class TestTheRetrospectivesComparison:
    """The reason this seam exists: "compared to last sprint".

    ``Y29FMFPX`` called the cross-period comparison the deep part of a retrospective,
    and ``tenbin drift`` supplies it over two archived readings. This is the path that
    only becomes walkable once a second source can be a report.
    """

    def test_two_archived_reads_of_a_second_source_drift_against_each_other(self, tmp_path) -> None:
        first = _snapshot(read_at=datetime(2026, 9, 15, tzinfo=UTC))
        second = _snapshot(
            read_at=datetime(2026, 9, 30, tzinfo=UTC),
            rows=(
                Transition(
                    ticket_id="A",
                    from_state="open",
                    to_state="in_progress",
                    at=datetime(2026, 9, 16, tzinfo=UTC),
                ),
                Transition(
                    ticket_id="A",
                    from_state="in_progress",
                    to_state="done",
                    at=datetime(2026, 9, 18, tzinfo=UTC),
                ),
            ),
        )
        archive = Archive(tmp_path / "a.jsonl")
        archive.record(_report(first))
        archive.record(_report(second))

        found = archive.readings("time-in-state-in-progress")
        assert len(found) == 2
        figure = DriftMeasure(earlier=found[0], later=found[1]).compute(second)

        assert figure.claim.slug == "time-in-state-in-progress.drift"
        assert "status-transitions" in figure.claim.denominator.description
        assert "2 records" in figure.claim.denominator.description

    def test_reads_of_uncomparable_windows_refuse_rather_than_report_a_change(
        self, tmp_path
    ) -> None:
        """A two-day sprint against a twelve-day one is not a trend.

        The drift measure bounds window comparability at 2:1 and refuses beyond it. This
        is here because it is the refusal most likely to be mistaken for the seam failing:
        it raises ``ComparativeRefusalError``, which is a refusal, not an error.
        """
        wide = _snapshot(
            read_at=datetime(2026, 9, 30, tzinfo=UTC),
            rows=(
                Transition(
                    ticket_id="A",
                    from_state="open",
                    to_state="in_progress",
                    at=datetime(2026, 9, 1, tzinfo=UTC),
                ),
                Transition(
                    ticket_id="A",
                    from_state="in_progress",
                    to_state="done",
                    at=datetime(2026, 9, 13, tzinfo=UTC),
                ),
            ),
        )
        archive = Archive(tmp_path / "a.jsonl")
        # The first read needs a moment of its own. `wide` is read at the module's
        # `READ_AT`, which is also `_snapshot()`'s default, so archiving both as
        # written put two readings of one slug at one instant -- and `record`
        # refused that, correctly, before the comparison under test ever ran. The
        # window is derived from the transitions rather than from `read_at`, so this
        # changes only which moment each read belongs to and leaves the 1-day-against-
        # 12-day span that makes the comparison refuse in the first place.
        archive.record(_report(_snapshot(read_at=datetime(2026, 9, 15, tzinfo=UTC))))
        archive.record(_report(wide))

        found = archive.readings("time-in-state-in-progress")
        with pytest.raises(ComparativeRefusalError) as excinfo:
            DriftMeasure(earlier=found[0], later=found[1]).compute(wide)
        assert "not comparable" in str(excinfo.value)
