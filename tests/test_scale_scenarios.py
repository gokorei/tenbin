"""Corpora that are large in entries and wide in datapoints, and still read whole.

Small fixtures prove the rules; they do not prove the rules survive growth. A
codebase that gains entries multiplies the walk, the parse and every
measure's scan. A codebase that gains datapoints per entry — files touched,
gates run, repositories and months — multiplies the grouping inside those
scans. These scenarios pin the growth axes separately so a regression in one
does not read as a failure of the other.
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from tenbin.claims.gate import ClaimGate
from tenbin.claims.registry import default_registry
from tenbin.corpus.period import PeriodConfig
from tenbin.corpus.record import RecordKind
from tenbin.corpus.snapshot import Completeness, take_snapshot
from tenbin.measures import CompletenessMeasure, RepositoryCoverageMeasure
from tenbin.measures.check_gates import check_gate_measures, check_runs
from tenbin.readmodel import build, load, to_snapshot, write
from tenbin.readmodel.resume import (
    MergeFindingKind,
    build_across_windows,
    reachable_ceiling,
)
from tenbin.report.build import build_report
from tenbin.report.retrospective import build_retrospective, read_period
from tenbin.store.ticket_sources.go_experiment import GoExperimentClient
from tenbin.store.transitions import take_transition_snapshot
from tests.fakes import (
    FakeStore,
    closing,
    document_id,
    documents_for,
    kojutsu_frontmatter,
)
from tests.fixtures import build_answers, build_record, build_snapshot

#: Entries axis: enough pages to exercise the walk more than once, small
#: enough that the suite stays fast. Page 500 below means four listings plus
#: their batch queries.
LARGE_ENTRY_COUNT = 2_000

#: Datapoints-per-entry axis: files named on one record. Forty paths is a
#: large change without being a vendored one.
WIDE_FILE_COUNT = 40
WIDE_RECORD_COUNT = 200

#: Distinct-values axis: one measure per gate, so gates are a multiplier on
#: the registry itself rather than on one distribution.
GATE_COUNT = 60
RUNS_PER_GATE = 3


def test_a_walk_over_two_thousand_entries_is_complete_and_counts_each_document_once_because_growth_in_entries_is_growth_in_pages() -> (
    None
):
    """Two thousand documents walk whole with no hole and no double count."""
    with closing(FakeStore(documents_for(LARGE_ENTRY_COUNT))) as store:
        snapshot = take_snapshot(store, page_limit=500)

    assert snapshot.completeness is Completeness.COMPLETE
    assert (snapshot.store_total, snapshot.enumerated) == (
        LARGE_ENTRY_COUNT,
        LARGE_ENTRY_COUNT,
    )
    assert len(snapshot.records) == LARGE_ENTRY_COUNT
    assert len({record.doc_id for record in snapshot.records}) == LARGE_ENTRY_COUNT
    assert snapshot.truncation is None
    assert snapshot.error is None


def test_a_corpus_with_many_files_per_record_groups_by_file_without_double_counting_because_growth_in_datapoints_is_growth_inside_each_scan() -> (
    None
):
    """Two hundred records naming forty files each stay a set per record."""
    records = tuple(
        build_record(
            entry_id=f"answer-{index:04d}",
            pr=index + 1,
            record_kind=RecordKind.ANSWER,
            frontmatter={
                "files": [f"src/mod_{file_index}.py" for file_index in range(WIDE_FILE_COUNT)]
            },
        )
        for index in range(WIDE_RECORD_COUNT)
    )
    snapshot = build_snapshot(records)

    figure = RepositoryCoverageMeasure().compute(snapshot)

    assert figure.counted + figure.excluded_total == WIDE_RECORD_COUNT
    assert len(snapshot.records) == WIDE_RECORD_COUNT
    assert all(len(record.files) == WIDE_FILE_COUNT for record in snapshot.records)


def test_a_corpus_with_sixty_gates_discovers_one_measure_per_gate_because_distinct_values_multiply_the_registry() -> (
    None
):
    """Sixty gates yield sixty measures, each over exactly its own runs."""
    records = tuple(
        build_record(
            entry_id=f"check-gate-{gate_index:03d}-run-{run_index}",
            pr=gate_index + 1,
            record_kind=RecordKind.CHECK_RUN,
            tags=["check", "check_state_success"],
            head_sha="9f2c1ab",
            capture_source="webhook",
            check_name=f"gate-{gate_index:03d}",
            frontmatter={"check_conclusion": "success"},
        )
        for gate_index in range(GATE_COUNT)
        for run_index in range(RUNS_PER_GATE)
    )
    snapshot = build_snapshot(records)

    assert len(check_runs(snapshot.records)) == GATE_COUNT * RUNS_PER_GATE
    measures = check_gate_measures(snapshot.records)
    assert len(measures) == GATE_COUNT
    for measure in measures:
        figure = measure.compute(snapshot)
        assert figure.counted == RUNS_PER_GATE


def test_a_large_read_model_round_trip_preserves_figures_because_a_kept_read_is_not_a_second_source_of_truth() -> (
    None
):
    """Fifteen hundred records survive the file with both figures unmoved."""
    snapshot = build_snapshot(build_answers(1500))
    model = build(snapshot)
    path = Path(tempfile.mkdtemp()) / "read-model.json"
    write(path, model)

    restored = load(path)
    assert restored is not None
    assert len(restored.records) == 1500
    rebuilt = to_snapshot(restored)

    for measure in (RepositoryCoverageMeasure(), CompletenessMeasure()):
        assert measure.compute(rebuilt).value_text() == measure.compute(snapshot).value_text()
        assert measure.compute(rebuilt).rate_text() == measure.compute(snapshot).rate_text()


def test_a_full_report_over_a_large_mixed_corpus_renders_because_entries_times_measures_is_the_hot_loop() -> (
    None
):
    """Eight hundred mixed records run every default measure without raising."""
    snapshot = build_snapshot(
        (
            *build_answers(600),
            *(
                build_record(
                    entry_id=f"review-{index:04d}",
                    pr=10_000 + index,
                    record_kind=RecordKind.REVIEW_VERDICT,
                    tags=["review", "review_state_approved"],
                    repo="acme/gadget",
                )
                for index in range(200)
            ),
        )
    )
    report = build_report(snapshot, gate=ClaimGate(), registry=default_registry())

    assert report.sections, "a large corpus must still produce sections"
    assert any(section.figure is not None for section in report.sections)


#: A heavy year on the document side: more than the seam can address with a
#: 500-document page, so no single walk can read it whole. One hundred past
#: the addressable range keeps the hole exact and the walk affordable.
HEAVY_YEAR_DOC_COUNT = reachable_ceiling(500) + 100

#: A heavy year on the ticket side: twenty-five hundred tickets, two
#: transitions each, spread across a calendar year.
HEAVY_YEAR_TICKETS = 2_500


def test_a_heavy_year_of_documents_reports_a_prefix_with_a_known_hole_because_a_year_does_not_fit_in_one_walk() -> (
    None
):
    """Ten thousand six hundred documents walk into an offset cap, not a corpus.

    The resumed read holds the 10,500 addressable documents exactly once, names
    the 100 beyond the seam, and says so in the completeness sentence -- so a
    report over a heavy year renders a prefix with a stated shortfall rather
    than a whole corpus or a second source of truth.
    """
    documents = {
        document_id("acme/widget", index, f"answer-{index:05d}"): kojutsu_frontmatter(
            repo="acme/widget", pr=index
        )
        for index in range(1, HEAVY_YEAR_DOC_COUNT + 1)
    }
    with closing(FakeStore(documents)) as store:
        outcome = build_across_windows(store, page_limit=500)

    model = outcome.model
    assert model.completeness is Completeness.OFFSET_CAP
    assert model.enumerated == len(model.records) == reachable_ceiling(500)
    assert len({record.doc_id for record in model.records}) == len(model.records)
    assert model.truncation is not None
    assert model.truncation.records_missing == 100
    assert any(finding.kind is MergeFindingKind.SEAM_EXHAUSTED for finding in outcome.findings)
    assert "were not reached" in CompletenessMeasure().compute(to_snapshot(model)).rate_text()

    report = build_report(to_snapshot(model), gate=ClaimGate(), registry=default_registry())
    assert report.sections, "a truncated year must still produce sections"


def _heavy_year_ticket_rows() -> list[dict[str, Any]]:
    """Five thousand backend rows spread across a calendar year, two per ticket."""
    base = datetime(2026, 1, 5, tzinfo=UTC)
    rows: list[dict[str, Any]] = []
    for index in range(HEAVY_YEAR_TICKETS):
        opened = base + timedelta(days=index * 365 // HEAVY_YEAR_TICKETS, hours=9)
        closed = opened + timedelta(days=1)
        rows.append(
            {
                "ticket_id": f"T{index:05d}",
                "from": "open",
                "to": "in_progress",
                "at": opened.isoformat(),
            }
        )
        rows.append(
            {
                "ticket_id": f"T{index:05d}",
                "from": "in_progress",
                "to": "done",
                "at": closed.isoformat(),
            }
        )
    return rows


def _heavy_year_ticket_client(rows: list[dict[str, Any]]) -> GoExperimentClient:
    """A backend holding a year of flow, filtering the way the backend does."""

    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        page = int(params.get("page", 1))
        limit = int(params.get("limit", 200))
        since, until = params.get("from"), params.get("to")
        matching = [
            row
            for row in rows
            if (not since or row["at"] >= since) and (not until or row["at"] <= until)
        ]
        total_pages = max(1, -(-len(matching) // limit))
        chunk = matching[(page - 1) * limit : page * limit]
        return httpx.Response(
            200,
            json={
                "results": chunk,
                "total": len(matching),
                "page": page,
                "limit": limit,
                "total_pages": total_pages,
            },
        )

    return GoExperimentClient("http://127.0.0.1:8082", transport=httpx.MockTransport(handler))


def test_a_heavy_year_of_ticket_flow_is_read_one_period_at_a_time_because_a_year_is_not_a_population() -> (
    None
):
    """Five thousand transitions walk whole, and one fortnight renders on its own.

    The full-year walk proves the volume is readable; the retrospective proves
    the year is never the denominator -- one declared period's tickets are, so
    consecutive retrospectives add up to the corpus rather than to its seams.
    """
    rows = _heavy_year_ticket_rows()
    period = PeriodConfig.from_text("14d", "2026-01-05T00:00:00Z").period_at(0)
    with _heavy_year_ticket_client(rows) as client:
        whole = take_transition_snapshot(client)
        assert whole.completeness is Completeness.COMPLETE
        assert whole.enumerated == whole.store_total == 2 * HEAVY_YEAR_TICKETS

        read = read_period(client, period)
        document = build_retrospective(read)

    assert 0 < read.snapshot.enumerated < 2 * HEAVY_YEAR_TICKETS
    assert read.snapshot.enumerated == read.snapshot.store_total
    tickets_in_window = {row.ticket_id for row in read.snapshot.transitions}
    assert 0 < len(tickets_in_window) < HEAVY_YEAR_TICKETS
    assert document.sections, "one period of a heavy year must still render"
