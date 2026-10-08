"""What the Snapshot protocol rejects, and what it admits.

This exists because widening ``Figure.snapshot`` from ``CorpusSnapshot`` to a
protocol is a real weakening, and the house rule is that a limitation living in a
docstring is a comment. So the limitation is pinned here as behaviour: these are
the objects that must still fail, and this one that now passes.

The weakening, stated plainly. ``isinstance(x, CorpusSnapshot)`` was nominal --
one class, unforgeable. ``isinstance(x, Snapshot)`` is structural, because a
runtime-checkable protocol can only ask whether the attributes are present. So an
object carrying the eight members with arbitrary values satisfies it, where before
it could not have. That is a deliberate trade: the alternative was a base class
that a second source would have to inherit from a Kojutsu-specific parent, and
inheriting ``CorpusSnapshot`` to describe status transitions would have been the
category error the placement decision warned about.

What the trade does *not* give up is the property the design exists for. A
figure still cannot be constructed without a snapshot that can be asked whether
the corpus behind it is whole. The values are not validated, but the questions
are answerable, and ``rate_text`` is the one method that reads them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pytest

from tenbin.claims import Claim, ClaimKind, Denominator, Granularity
from tenbin.corpus.snapshot import Completeness, CorpusSnapshot, Snapshot, Truncation
from tenbin.corpus.window import CorpusWindow
from tenbin.measures.base import CountFigure, Figure


def _claim(size: int = 1847) -> Claim:
    return Claim(
        slug="test-claim",
        statement="Transitions were recorded this often.",
        does_not_mean="That anybody responded to any of them.",
        falsifier="A read showing a materially different rate.",
        denominator=Denominator(
            description="status transitions read from the ticket backend",
            size=size,
            size_noun="transitions",
        ),
        kind=ClaimKind.descriptive,
        granularity=Granularity.team,
    )


def _kojutsu_snapshot() -> CorpusSnapshot:
    return CorpusSnapshot(
        records=(),
        collection="kojutsu",
        read_at=datetime(2026, 9, 30, tzinfo=UTC),
        store_total=0,
        enumerated=0,
        completeness=Completeness.COMPLETE,
    )


@dataclass(frozen=True)
class _SecondSourceSnapshot:
    """A different source's snapshot: no records, and it answers the questions."""

    read_at: datetime
    store_total: int
    enumerated: int
    completeness: Completeness
    truncation: Truncation | None
    error: None
    window: CorpusWindow
    collection: str


def _second_source() -> _SecondSourceSnapshot:
    return _SecondSourceSnapshot(
        read_at=datetime(2026, 9, 30, tzinfo=UTC),
        store_total=1847,
        enumerated=1847,
        completeness=Completeness.COMPLETE,
        truncation=None,
        error=None,
        window=CorpusWindow(earliest=None, latest=None, unreadable_timestamps=0),
        collection="status-transitions",
    )


class _HalfASnapshot:
    completeness = Completeness.COMPLETE
    truncation = None
    error = None


class _Bare:
    pass


class TestCorpusSnapshotStillSatisfiesIt:
    def test_the_existing_snapshot_is_a_snapshot(self) -> None:
        assert isinstance(_kojutsu_snapshot(), Snapshot)

    def test_a_figure_over_it_is_unaffected(self) -> None:
        # A bare Figure, because a CountFigure needs a Denominator and
        # Denominator.size < 1 is refused -- an empty corpus is a finding, not a
        # rate, and the refusal is the design working rather than a limit on this
        # test.
        figure = Figure(claim=_claim(), snapshot=_kojutsu_snapshot(), title="Records")
        assert figure.snapshot.collection == "kojutsu"


class TestWhatIsStillRefused:
    @pytest.mark.parametrize(
        "candidate",
        [None, 0, "", [], {}, _Bare(), _HalfASnapshot()],
        ids=["none", "int", "str", "list", "dict", "bare-object", "partial"],
    )
    def test_figure_refuses_a_non_snapshot(self, candidate: Any) -> None:
        """The claim is valid, so the only thing left to fail on is the snapshot."""
        with pytest.raises(ValueError, match="Figure.snapshot is required"):
            Figure(claim=_claim(), snapshot=candidate, title="t")  # type: ignore[arg-type]

    def test_refusal_names_the_reason_rather_than_raising_deeper_later(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            Figure(claim=_claim(), snapshot={}, title="t")  # type: ignore[arg-type]
        assert "cannot be asked whether" in str(excinfo.value)

    def test_a_missing_claim_is_still_refused_first(self) -> None:
        """Order matters: claim before snapshot, so the first error is the design's."""
        with pytest.raises(ValueError, match="Figure.claim is required"):
            Figure(claim=None, snapshot=_second_source(), title="t")  # type: ignore[arg-type]


class TestASecondSourceIsAdmitted:
    def test_a_different_source_satisfies_the_protocol(self) -> None:
        assert isinstance(_second_source(), Snapshot)

    def test_a_figure_can_be_built_over_it(self) -> None:
        figure = CountFigure(
            claim=_claim(), snapshot=_second_source(), value=1847, title="Transitions read"
        )
        assert figure.snapshot.collection == "status-transitions"

    def test_rate_text_reads_integrity_from_the_second_source(self) -> None:
        """The point of the protocol: rate_text reads integrity, not provenance."""
        figure = CountFigure(
            claim=_claim(), snapshot=_second_source(), value=1847, title="Transitions read"
        )
        text = figure.rate_text()
        assert "whole" in text.lower()
        assert "kojutsu" not in text

    def test_a_truncated_second_source_names_its_shortfall(self) -> None:
        """Completeness is not decoration: a prefix must render as a prefix."""
        source = _second_source()
        truncated_source = _SecondSourceSnapshot(
            read_at=source.read_at,
            store_total=1847,
            enumerated=100,
            completeness=Completeness.OFFSET_CAP,
            truncation=Truncation(
                offset_reached=100,
                records_missing=1747,
                boundary_repositories={},
                boundary_months={},
            ),
            error=None,
            window=source.window,
            collection=source.collection,
        )
        figure = CountFigure(
            claim=_claim(size=1847), snapshot=truncated_source, value=100, title="Transitions read"
        )
        text = figure.rate_text()
        assert "whole corpus" not in text.lower()
        assert "not reached" in text.lower()
