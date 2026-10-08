"""Drift over two archived reads: the gate's amendment, the archive, and what a refusal means.

Three subjects, and the order is the argument. **First the gate**, because the whole ticket
rests on a distinction nobody has written down: a comparison *between principals* is refused
and a comparison of one population *against its own past value* is not. Those two look like
the same thing from a distance, and the reader who concludes the second is also refused
declines to build the most useful figure this corpus supports and nobody finds out for a year.
**Then the archive**, because a drift needs two readings and the properties that make those
readings worth keeping -- append-only, one entry per read, the window stored beside the value
-- are the reason a stored figure can be trusted rather than merely believed. **Then the
refusals**, because a drift that reports over incomparable windows or over a pair separated by
a migration is the failure mode this whole design exists to prevent, and a refusal that only
the test suite knows about is a comment.

Every test here is named for what it protects rather than for what it does, per
``docs/measure-protocol.md``. The two gate tests are named for the *distinction* they pin --
the whole content of the amendment is which of two kinds is refused, and a test called
``test_temporal_claim_renders`` would have to be rewritten the day somebody added a third kind.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import pytest

from tenbin.claims.gate import (
    COMPARATIVE_ABSENCE_IS_AN_ASSIGNMENT_MECHANISM,
    ClaimGate,
    compares_principals,
)
from tenbin.claims.model import Claim, ClaimKind, Denominator, Granularity
from tenbin.claims.refusals import ComparativeRefusalError
from tenbin.corpus.record import Record
from tenbin.corpus.snapshot import CorpusSnapshot
from tenbin.corpus.window import CorpusWindow
from tenbin.measures.base import CountFigure, Figure
from tenbin.measures.drift import (
    WINDOW_LENGTH_RATIO,
    CorpusEvent,
    DriftFigure,
    DriftMeasure,
    Reading,
)
from tenbin.report.archive import Archive, CorruptArchiveError, DuplicateReadingError
from tenbin.report.base import Section
from tenbin.report.document import CorpusHeader, Report
from tests.fixtures import SnapshotBuilder, build_record

#: The slug the census measure publishes, stated once so a test can look a reading up without
#: repeating a string that would then be allowed to drift from the measure's own.
CENSUS_SLUG: Final[str] = "corpus.census"

#: Two reads a week apart, so the ordinary case needs no dates to make it comparable.
EARLIER_READ_AT: Final[datetime] = datetime(2026, 9, 21, 12, 0, 0, tzinfo=UTC)
LATER_READ_AT: Final[datetime] = datetime(2026, 9, 28, 12, 0, 0, tzinfo=UTC)

#: The ends of the two comparable windows: seven days, then twelve. A factor of 1.7, comfortably
#: inside :data:`WINDOW_LENGTH_RATIO`, so the test that expects a figure is not one boundary
#: away from the test that expects a refusal.
EARLIEST: Final[str] = "2026-09-14T00:00:00+00:00"
EARLIER_NEWEST: Final[str] = "2026-09-21T00:00:00+00:00"
LATER_NEWEST: Final[str] = "2026-09-26T00:00:00+00:00"

#: A later read whose window is over two years wide rather than twelve days. Same measure,
#: same collection, same shape of event -- and a different claim, which is the whole point of
#: storing the window beside the figure.
ANCIENT_EARLIEST: Final[str] = "2024-01-01T00:00:00+00:00"


@dataclass(frozen=True)
class CensusMeasure:
    """A measure that counts documents filed under one id segment, and nothing else.

    Its slug is fixed while its answer moves with ``bucket``, which is how a test gets a *measure
    that changed* without a second slug to look it up by: the two instances answer the same
    question about the same corpus and disagree, exactly as two versions of one measure would.

    It is a count rather than a rate on purpose. ``Archive.record`` stores whatever the figure
    rendered, so the archive has to be shown working for the shape that carries no percentage in
    it at all -- a drift over a distribution is a drift over a multi-line value, and the whole
    "any measure" claim depends on the archive not having been built for rates.
    """

    bucket: str

    @property
    def slug(self) -> str:
        """The identifier this measure is cited by, and it does not vary with the bucket."""
        return CENSUS_SLUG

    def claim(self, snapshot: CorpusSnapshot) -> Claim:
        """The claim, over the read that produced it.

        A plain descriptive claim, because the measure is one: the temporal kind is what a drift
        over it gets, and a figure that claimed to be about time while being computed from one
        read would be the exact confusion this ticket exists to undo.
        """
        return Claim(
            slug=CENSUS_SLUG,
            statement=f"Documents filed under {self.bucket} in this read.",
            does_not_mean=(
                "That any of them were written by a principal of any particular kind; the id "
                "segment says where Kojutsu filed the document and nothing about who wrote it."
            ),
            falsifier=(
                f"A document outside {self.bucket} that this read filed there, or one inside it "
                "that this read filed elsewhere."
            ),
            denominator=Denominator("documents this read enumerated", max(1, snapshot.enumerated)),
            kind=ClaimKind.descriptive,
            granularity=Granularity.corpus,
        )

    def compute(self, snapshot: CorpusSnapshot) -> Figure:
        """How many of the read's documents carry the segment, as a count."""
        counted = sum(1 for record in snapshot.records if self.bucket in record.doc_id)
        return CountFigure(
            claim=self.claim(snapshot),
            snapshot=snapshot,
            title=f"Documents filed under {self.bucket}",
            value=counted,
        )


def _snapshot(
    *,
    read_at: datetime,
    answers: int,
    rationales: int,
    earliest: str = EARLIEST,
    newest: str = EARLIER_NEWEST,
) -> CorpusSnapshot:
    """A read with a stated window, so the archive has a span to store beside the figure.

    Two id segments with different ``updated_at`` values, which is how the window is set without
    reaching into :class:`~tenbin.corpus.window.CorpusWindow` to hand the archive one: the
    snapshot derives its own window from the records, exactly as a walk would, so a test that
    archived a window no read could have produced would not be caught here.
    """
    records: list[Record] = [
        build_record(entry_id=f"answer-{index:04d}", pr=index + 1, updated_at=earliest)
        for index in range(answers)
    ]
    records.extend(
        build_record(
            entry_id=f"r-{index:04d}",
            pr=100 + index,
            rationale=True,
            updated_at=newest,
            tags=["rationale"],
        )
        for index in range(rationales)
    )
    return SnapshotBuilder(records=tuple(records), read_at=read_at).build()


def _report(snapshot: CorpusSnapshot, measure: CensusMeasure) -> Report:
    """A report holding one figure, which is the unit the archive appends."""
    figure = measure.compute(snapshot)
    assert figure.claim is not None
    return Report(
        corpus=CorpusHeader.from_snapshot(snapshot),
        sections=(Section(claim=figure.claim, figure=figure, refusal=None),),
    )


def _archive(tmp_path: Path) -> Archive:
    """An archive over a path nothing has touched, in a temporary directory."""
    return Archive(tmp_path / "readings.jsonl")


def _two_readings(tmp_path: Path) -> tuple[Reading, Reading]:
    """Two comparable archived readings of one measure, five documents apart at first.

    Both come out of the archive rather than being constructed here, because the drift's subject
    is what the archive *kept* and a :class:`Reading` built by hand would let a test pass against
    a shape the file never round-trips.
    """
    archive = _archive(tmp_path)
    archive.record(
        _report(
            _snapshot(read_at=EARLIER_READ_AT, answers=2, rationales=1), CensusMeasure("answer")
        )
    )
    archive.record(
        _report(
            _snapshot(read_at=LATER_READ_AT, answers=5, rationales=3, newest=LATER_NEWEST),
            CensusMeasure("answer"),
        )
    )
    earlier, later = archive.readings(CENSUS_SLUG)
    return earlier, later


# ---------------------------------------------------------------------------------------------
# The gate amendment: two populations are refused, one population read twice is not.
# ---------------------------------------------------------------------------------------------


def _claim(kind: ClaimKind) -> Claim:
    """A valid claim of the given kind, so a test can vary exactly one thing."""
    return Claim(
        slug="example-claim",
        statement="Two readings of one corpus, a week apart.",
        does_not_mean="That anything caused the difference between them.",
        falsifier="Two reads separated by a documented migration rather than by the passage of time.",
        denominator=Denominator("both archived reads", 80),
        kind=kind,
        granularity=Granularity.corpus,
    )


def test_a_temporal_claim_renders_and_a_comparative_one_does_not_because_only_the_second_compares_two_principals() -> (
    None
):
    """The distinction the amendment makes, pinned in one test rather than two.

    The gate is built with the mechanism unset, which is the state the corpus is actually in, so
    the only thing separating the two claims is the kind. **A temporal claim is one population
    read twice: no treatment, no arm, and no selection between groups**, so there is no counterfactual
    to supply and nothing decision 002 refuses. A comparative claim is two populations, and the
    corpus holds no assignment mechanism, so it is refused -- which is unchanged, and is asserted
    here rather than left to ``tests/test_refusal.py`` so that a future change to the gate cannot
    quietly widen or narrow the space between the two.

    Rendered, not merely permitted: ``ClaimGate.render`` is the path a report takes, and a claim
    that passes ``check`` while failing to render would be a capability nobody can use.
    """
    gate = ClaimGate()
    temporal, comparative = _claim(ClaimKind.temporal), _claim(ClaimKind.comparative)

    rendered = gate.render(temporal)
    assert "Two readings of one corpus" in rendered
    assert "Assignment mechanism" not in rendered

    with pytest.raises(ComparativeRefusalError) as raised:
        gate.render(comparative)
    assert raised.value.refusal.claim_slug == comparative.slug


def test_the_mechanism_rule_keys_on_the_declared_kind_and_nothing_else_because_a_rule_that_reads_a_caller_controlled_value_moves_when_the_call_site_does() -> (
    None
):
    """:func:`compares_principals` is a membership test on the kind, and every kind is asserted.

    A gate whose verdict depended on the denominator's size, the granularity, or how much the
    two numbers differed would be a guard that fires on messy data and stays quiet on a clean
    fixture -- the shape ``tests/test_refusal.py`` exists to argue against. So the rule is stated
    once, under a name that says what it keys on, and this test walks all four kinds rather than
    asserting only the pair that happens to matter today: a third kind added later must make a
    decision here rather than inherit one.
    """
    assert compares_principals(_claim(ClaimKind.comparative)) is True
    assert compares_principals(_claim(ClaimKind.assignment_mechanism)) is True
    assert compares_principals(_claim(ClaimKind.descriptive)) is False
    assert compares_principals(_claim(ClaimKind.temporal)) is False


def test_the_refusal_names_an_absent_assignment_mechanism_and_says_what_is_still_permitted_because_a_silent_gate_reads_as_a_refusal_of_all_comparison() -> (
    None
):
    """The reason a reader who was just refused is owed, and the sentence that goes with it.

    The refusal's own reason argues about the corpus and names the assignment mechanism as the
    missing fact -- that half is already there and is asserted here so a reworded
    :data:`~tenbin.claims.mechanism.MECHANISM_ABSENT_REASON` cannot quietly stop saying it. What
    the refusal cannot do on its own is say what *survives* it, and the asymmetry is the whole
    reason this gate carries a note: a reader who concludes that no comparison of any kind is
    available declines to build the most useful figure this corpus supports, and the omission is
    invisible for a year. So the note is required to name both the absent thing and the permitted
    one, and this test fails if either goes quiet.
    """
    refusal = ClaimGate().check(_claim(ClaimKind.comparative))
    assert refusal is not None
    assert "assigned to write it" in refusal.reason
    assert refusal.missing_fact is not None and "assignment mechanism" in refusal.missing_fact
    assert refusal.title == "Comparative claim with no assignment mechanism"

    note = ClaimGate().comparison_permitted_without_a_mechanism
    assert note == COMPARATIVE_ABSENCE_IS_AN_ASSIGNMENT_MECHANISM
    assert "assignment mechanism between principals" in note
    assert "read twice" in note and "temporal" in note


# ---------------------------------------------------------------------------------------------
# The archive: append-only, one entry per read, the window stored beside the value.
# ---------------------------------------------------------------------------------------------


def test_a_changed_measure_appends_a_second_reading_because_a_drift_programme_that_rewrites_its_own_past_has_no_drift_to_report(
    tmp_path: Path,
) -> None:
    """The load-bearing property, and it is about the first line of the file, not the last.

    The census measure answers "5" at the first read. The measure is then changed -- same slug,
    same corpus, a different subject, so it answers "3" -- and the second read is archived. The
    archive must hold **both**, with the first line untouched, and a drift over the pair reports
    a figure that moved for a reason nobody can name. An archive that recomputed or overwrote the
    first entry would report no movement at all, which is indistinguishable from a process that
    did not change, and that is the failure this whole file exists to prevent.

    The assertion is on the raw lines rather than on :meth:`Archive.readings` alone, because a
    reader who opens a log has to be able to see that nothing was rewritten.
    """
    archive = _archive(tmp_path)
    earlier_snapshot = _snapshot(read_at=EARLIER_READ_AT, answers=2, rationales=1)
    archive.record(_report(earlier_snapshot, CensusMeasure("answer")))
    first_lines = archive.path.read_text(encoding="utf-8").splitlines()

    later_snapshot = _snapshot(read_at=LATER_READ_AT, answers=0, rationales=3, newest=LATER_NEWEST)
    archive.record(_report(later_snapshot, CensusMeasure("rationale")))

    assert archive.path.read_text(encoding="utf-8").splitlines()[:1] == first_lines, (
        "the earlier reading must survive verbatim; a corrected reading is a rewritten past"
    )
    earlier, later = archive.readings(CENSUS_SLUG)
    assert (earlier.value, later.value) == ("2", "3")
    assert (earlier.read_at, later.read_at) == (EARLIER_READ_AT, LATER_READ_AT)
    assert DriftMeasure(earlier=earlier, later=later).slug == f"{CENSUS_SLUG}.drift"


def test_a_second_reading_of_the_same_measure_at_the_same_moment_is_refused_because_two_entries_for_one_read_is_a_drift_that_never_happened(
    tmp_path: Path,
) -> None:
    """Re-running the same report against the same archive is refused, not appended.

    Two entries for one ``(slug, read_at)`` would be two values for one observation, and every
    drift computed across them would report a change that no read ever saw -- which is the shape
    of defect this project exists to report, arriving from the measuring apparatus itself. The
    whole file must be untouched by the refused attempt, so the count is asserted rather than
    only the exception.

    ``RuntimeError`` rather than ``ValueError`` and the assertion is on the type, because the
    distinction matters to a caller: nothing about either reading is malformed, and a
    ``ValueError`` would invite somebody to go and fix one of them.
    """
    archive = _archive(tmp_path)
    report = _report(
        _snapshot(read_at=EARLIER_READ_AT, answers=2, rationales=1), CensusMeasure("answer")
    )
    archive.record(report)

    with pytest.raises(DuplicateReadingError) as raised:
        archive.record(report)
    assert raised.value.slug == CENSUS_SLUG
    assert raised.value.read_at == EARLIER_READ_AT
    assert len(archive.path.read_text(encoding="utf-8").splitlines()) == 1


def test_a_reading_carries_the_window_of_its_own_read_because_a_drift_that_cannot_see_the_window_cannot_refuse_an_incomparable_pair(
    tmp_path: Path,
) -> None:
    """Each reading arrives with the span of the read that produced it, and it round-trips.

    This is the dependency the ticket names: a figure that moved over three weeks and one that
    moved over three years are different claims, and nothing in the rendered figure says which it
    is. Without the window beside the value there is no way to tell, and the drift would have to
    take the reader's word for it.

    The window is read back *out of the file* rather than compared to the snapshot, because a
    reading that serialises a window and restores a different one would be a silent hole: the
    round trip is the whole of the claim, and asserting on the in-memory object would prove
    nothing about the archive.
    """
    archive = _archive(tmp_path)
    snapshot = _snapshot(read_at=LATER_READ_AT, answers=2, rationales=1, newest=LATER_NEWEST)
    archive.record(_report(snapshot, CensusMeasure("answer")))
    (reading,) = archive.readings(CENSUS_SLUG)

    assert reading.window == snapshot.window
    assert reading.enumerated == snapshot.enumerated
    assert reading.collection == snapshot.collection
    payload = json.loads(archive.path.read_text(encoding="utf-8").strip())
    assert payload["window"]["latest"] == snapshot.window.latest.isoformat()
    assert payload["window"]["earliest"] == snapshot.window.earliest.isoformat()


def test_an_unreadable_line_is_refused_rather_than_skipped_because_a_skipped_reading_becomes_the_change_it_bridged(
    tmp_path: Path,
) -> None:
    """A half-written line aborts the read instead of becoming a gap.

    If the line this program cannot parse were dropped, the two readings either side of it would
    be compared as adjacent and the gap would be published as the drift. That is the archive
    manufacturing the finding it exists to record, so a truncated final line -- the one shape a
    crash mid-append actually produces -- raises and names the line number.
    """
    archive = _archive(tmp_path)
    archive.record(
        _report(
            _snapshot(read_at=EARLIER_READ_AT, answers=2, rationales=1), CensusMeasure("answer")
        )
    )
    with archive.path.open("a", encoding="utf-8") as stream:
        stream.write('{"slug": "corpus.census", "value": "9"\n')

    with pytest.raises(CorruptArchiveError, match="line 2"):
        archive.readings(CENSUS_SLUG)


def test_the_archive_offers_no_verb_that_could_rewrite_a_reading_because_the_temptation_is_exactly_what_it_refuses(
    tmp_path: Path,
) -> None:
    """The class surface is the property, so it is asserted rather than trusted.

    Four verbs and none of them updates or removes a line. A caller who wants history rewritten
    has to open the file in an editor, which is the loudest version of that decision and the one
    a reviewer can see. This test exists because the next person who finds a use case for a
    "corrected" reading will otherwise add ``update`` rather than argue for it here.
    """
    verbs = {
        name for name in vars(Archive) if not name.startswith("_") and callable(vars(Archive)[name])
    }
    assert verbs == {"record", "readings", "slugs"}


# ---------------------------------------------------------------------------------------------
# The drift: over any measure, over both reads, and the two refusals.
# ---------------------------------------------------------------------------------------------


def test_a_drift_between_two_archived_reads_is_produced_for_any_measure_because_a_programme_with_no_memory_is_a_snapshot_taken_repeatedly(
    tmp_path: Path,
) -> None:
    """A drift renders for a measure whose value is a bare count, with no percentage in it.

    Criterion one of the ticket and the reason the archive stores rendered text rather than a
    parsed numerator: "any measure" has to include the ones whose figure is a set of bucket
    counts, and an archive built around a rate would answer "is the merge rate falling" and
    nothing else. The figure reports both readings, says which is which by the moment it was
    read, and states plainly whether they differ -- and it carries the completeness sentence,
    because a drift is a figure like any other and has no business escaping it.
    """
    earlier, later = _two_readings(tmp_path)
    later_snapshot = _snapshot(read_at=LATER_READ_AT, answers=5, rationales=3, newest=LATER_NEWEST)
    figure = DriftMeasure(earlier=earlier, later=later).compute(later_snapshot)

    assert isinstance(figure, DriftFigure)
    assert figure.claim.slug == f"{CENSUS_SLUG}.drift"
    rendered = figure.value_text()
    assert f"At {EARLIER_READ_AT.isoformat()}: 2" in rendered
    assert f"At {LATER_READ_AT.isoformat()}: 5" in rendered
    assert "moved between these two reads" in rendered
    assert figure.moved is True
    assert "Whole read" in figure.rate_text()


def test_the_drift_denominator_counts_both_reads_and_states_both_because_the_same_percentage_points_over_forty_and_twelve_hundred_are_different_facts(
    tmp_path: Path,
) -> None:
    """Three records then eight: the denominator is eleven and it says which was which.

    A drift whose denominator held only the later read would make a change over three documents
    look like a change over a corpus; one that held only the earlier would be a rate whose
    population no longer exists. So the size is the sum and the description names both counts
    *and both dates*, because "40 then 1200" without the moment of each is a pair of numbers
    with no order to them.

    The size is also asserted as the sum rather than trusted, because the failure this guards is
    a denominator that quietly became one of the two rather than both.
    """
    earlier, later = _two_readings(tmp_path)
    denominator = (
        DriftMeasure(earlier=earlier, later=later)
        .claim(_snapshot(read_at=LATER_READ_AT, answers=5, rationales=3, newest=LATER_NEWEST))
        .denominator
    )

    assert denominator.size == earlier.enumerated + later.enumerated == 11
    rendered = denominator.render_text()
    assert "both archived reads" in rendered
    assert "3 records at the read of 2026-09-21" in rendered
    assert "8 at the read of 2026-09-28" in rendered
    assert "11 records" in rendered


def test_a_drift_between_windows_of_very_different_lengths_is_refused_because_three_weeks_and_three_years_are_different_claims(
    tmp_path: Path,
) -> None:
    """Same measure, same collection, windows seven days and two years wide: no figure.

    Both windows are stated in the refusal rather than one of them being chosen as the anchor,
    because the anchor is the choice a reader gets wrong exactly when one window is a prefix of
    the other -- which is the ordinary case for an archive, since every read's window contains
    the one before it.

    The two figures are byte-identical apart from their dates, so nothing about the *data* says
    these cannot be compared; only the windows do, and that is the point of storing them.
    """
    archive = _archive(tmp_path)
    archive.record(
        _report(
            _snapshot(read_at=EARLIER_READ_AT, answers=2, rationales=1), CensusMeasure("answer")
        )
    )
    later_snapshot = _snapshot(
        read_at=LATER_READ_AT,
        answers=2,
        rationales=1,
        earliest=ANCIENT_EARLIEST,
        newest=LATER_NEWEST,
    )
    archive.record(_report(later_snapshot, CensusMeasure("answer")))
    earlier, later = archive.readings(CENSUS_SLUG)

    with pytest.raises(ComparativeRefusalError) as raised:
        DriftMeasure(earlier=earlier, later=later).compute(later_snapshot)

    reason = raised.value.refusal.reason
    assert "not comparable" in reason
    assert f"{WINDOW_LENGTH_RATIO:g} to 1" in reason
    assert raised.value.refusal.missing_fact is None, (
        "no fact arriving from Kojutsu makes two windows one span; the remedy is a "
        "different pair of reads and a refusal must not promise a ticket"
    )


def test_two_reads_separated_by_a_documented_corpus_event_are_refused_as_confounded_because_a_migration_is_not_drift(
    tmp_path: Path,
) -> None:
    """A webhook reinstall between the two reads, named by the caller, kills the drift.

    The event is a parameter and never a detection, because Tenbin cannot observe a reinstall:
    the store does not record it and the webhook does not write to Tanseki. Guessing would be worse
    than asking -- an inferred "something happened here" is the programme inventing the only
    finding separating a change from a cause. So the caller says so and the figure refuses, and
    the refusal quotes the event *and its evidence*, so the claim that the pair is unusable is
    itself checkable.

    The pair here is otherwise clean: same measure, same collection, comparable windows. The
    event alone is enough, which is the whole point.
    """
    earlier, later = _two_readings(tmp_path)
    event = CorpusEvent(
        what="the Kojutsu webhook was reinstalled, so captures resumed at a new address",
        documented_by="operations runbook entry 2026-09-24",
    )
    measure = DriftMeasure(earlier=earlier, later=later, corpus_event=event)

    with pytest.raises(ComparativeRefusalError) as raised:
        measure.compute(
            _snapshot(read_at=LATER_READ_AT, answers=5, rationales=3, newest=LATER_NEWEST)
        )

    reason = raised.value.refusal.reason
    assert "confounded" in reason
    assert "webhook was reinstalled" in reason
    assert "operations runbook entry 2026-09-24" in reason
    assert "never inferred" in reason
    assert raised.value.refusal.claim_slug == f"{CENSUS_SLUG}.drift"


def test_a_corpus_event_does_not_silence_the_claim_because_the_falsifier_has_to_name_what_would_make_the_drift_wrong(
    tmp_path: Path,
) -> None:
    """The falsifier names the documented event, not "if the figure were wrong".

    A required falsifier is only worth anything if somebody could act on it, and for a drift the
    actionable observation is specific: two archived reads whose corpora differ by a migration or
    a reinstall rather than by the passage of time. The claim says which pair would kill it, and
    says that the caller supplies the event because nothing in the corpus marks one.

    The second falsifier is the window bound, which is the other thing that would make a drift
    over this pair the wrong claim.
    """
    earlier, later = _two_readings(tmp_path)
    claim = DriftMeasure(earlier=earlier, later=later).claim(
        _snapshot(read_at=LATER_READ_AT, answers=5, rationales=3, newest=LATER_NEWEST)
    )

    assert claim.kind is ClaimKind.temporal
    assert "webhook reinstall" in claim.falsifier
    assert "migration" in claim.falsifier
    assert f"{WINDOW_LENGTH_RATIO:g} to one" in claim.falsifier
    assert "nothing here says the process changed" not in claim.does_not_mean.casefold()


def test_a_drift_refuses_two_measures_two_collections_and_the_wrong_order_because_none_of_those_is_this_figure(
    tmp_path: Path,
) -> None:
    """The three pairings that are not one population observed twice, refused at construction.

    There is no default "the previous reading", so every drift has to name both. What it can do
    is refuse the pairs that mean something else -- two measures (two figures, not a drift in
    either), two collections (two populations, which is the refusal decision 002 exists for and
    which a length field would have let through), and two reads the wrong way round. The refusal
    is a ``ValueError`` in each case because the call is malformed, not because the corpus cannot
    support the figure.
    """
    earlier, later = _two_readings(tmp_path)
    other_slug = Reading(
        slug="corpus.other",
        value="1",
        collection=earlier.collection,
        read_at=LATER_READ_AT,
        enumerated=1,
        window=CorpusWindow(
            earliest=datetime(2026, 9, 14, tzinfo=UTC),
            latest=datetime(2026, 9, 21, tzinfo=UTC),
            unreadable_timestamps=0,
        ),
    )
    other_collection = Reading(
        slug=earlier.slug,
        value="1",
        collection="kojutsu-other",
        read_at=LATER_READ_AT,
        enumerated=1,
        window=earlier.window,
    )

    with pytest.raises(ValueError, match="its own past value"):
        DriftMeasure(earlier=earlier, later=other_slug)
    with pytest.raises(ValueError, match="one collection against its own past"):
        DriftMeasure(earlier=earlier, later=other_collection)
    with pytest.raises(ValueError, match="not a drift"):
        DriftMeasure(earlier=later, later=earlier)


def test_todays_figure_is_computed_from_the_corpus_because_the_archive_answers_what_was_true_then_and_never_what_is_true_now(
    tmp_path: Path,
) -> None:
    """A report run today is measured from today's corpus and does not consult the archive.

    The archive holds "5" for the census at the last read. The corpus today holds nine answers
    and no rationales, and the figure today's measure computes is "9" -- and the archived
    reading is still "5" afterwards, because a reader who asked for today's number is owed the
    number today's corpus supports and the archive has no vote in it.

    The archive is a log, not a cache and not a second source of truth: if it could answer for
    the present then a stale reading would be reported as a current figure, and the two would
    disagree for reasons no report could state. That is the failure
    :mod:`tenbin.readmodel` refuses to have and this file inherits the same argument for a
    different subject.
    """
    archive = _archive(tmp_path)
    measure = CensusMeasure("answer")
    archive.record(
        _report(
            _snapshot(read_at=LATER_READ_AT, answers=5, rationales=3, newest=LATER_NEWEST), measure
        )
    )

    today = _snapshot(read_at=LATER_READ_AT, answers=9, rationales=0, newest=LATER_NEWEST)
    figure = measure.compute(today)

    assert isinstance(figure, CountFigure)
    assert figure.value == 9
    assert figure.value_text() != archive.readings(CENSUS_SLUG)[-1].value
    assert archive.readings(CENSUS_SLUG)[-1].value == "5", (
        "asking for today's figure must not rewrite what the archive said was true then"
    )


def test_a_reading_whose_window_could_not_be_read_is_refused_rather_than_reported_because_an_unknown_span_is_not_a_short_one(
    tmp_path: Path,
) -> (  # noqa: E501
    None
):
    """A read whose records carry no moment yields an *unknown* window, and the drift refuses.

    The other half of storing the window beside the value. A corpus whose timestamps will not
    parse spans an unknown amount of time, which is not the same as spanning none of it, and the
    honest answer to "how long a window did this drift cross" for a pair of such reads is that
    nobody knows. Defaulting the span to the read's own moment would state a duration nobody
    measured, and defaulting it to zero would report an empty read as a fast one.

    Asserted on the rendered reason as well as on the refusal, because "could not be read" and
    "spans no time at all" are two different operator problems and a single ``False`` would
    render both as one unexplained silence.
    """
    undated = SnapshotBuilder(
        records=(
            build_record(entry_id="answer-0000", pr=1, updated_at=None),
            build_record(entry_id="r-0000", pr=2, rationale=True, updated_at=None),
        ),
        read_at=LATER_READ_AT,
    ).build()
    archive = _archive(tmp_path)
    archive.record(_report(undated, CensusMeasure("answer")))
    (later,) = archive.readings(CENSUS_SLUG)
    assert later.window.is_known is False

    earlier = Reading(
        slug=later.slug,
        value="2",
        collection=later.collection,
        read_at=EARLIER_READ_AT,
        enumerated=3,
        window=later.window,
    )
    with pytest.raises(ComparativeRefusalError) as raised:
        DriftMeasure(earlier=earlier, later=later).compute(undated)
    assert "could not be read" in raised.value.refusal.reason
    assert "duration nobody measured" in raised.value.refusal.reason


def test_a_reading_that_names_no_span_no_count_and_no_moment_is_refused_because_each_of_those_is_what_the_archive_is_for(  # noqa: E501
    tmp_path: Path,
) -> None:
    """The three fields a reading cannot be built without, each refused by name.

    A reading with no enumerated count has a denominator nobody can weigh, one with a naive
    moment cannot be ordered against another reading, and one with no window cannot be compared
    at all -- which is the comparison the archive exists to make possible, so all three are
    refusals at construction rather than at the point of use, where the defect would be a wrong
    figure rather than a missing one.

    The window case is the one worth naming: a caller holding a value and a date could perfectly
    well leave the span out, and the drift would then compare two readings over durations nobody
    recorded.
    """
    valid = _two_readings(tmp_path)[0]
    fields: dict[str, Any] = {
        "slug": valid.slug,
        "value": valid.value,
        "collection": valid.collection,
        "read_at": valid.read_at,
        "enumerated": valid.enumerated,
        "window": valid.window,
    }
    for field, bad in (
        ("window", "seven days"),
        ("enumerated", -1),
        ("read_at", datetime(2026, 9, 21, 12, 0, 0)),
    ):
        with pytest.raises(ValueError, match=rf"Reading\.{field}"):
            Reading(**{**fields, field: bad})
    with pytest.raises(ValueError, match=r"Reading\.slug must not be blank"):
        Reading(**{**fields, "slug": "   "})


def test_an_archive_holds_nothing_and_says_so_because_no_reading_is_an_answer_rather_than_a_failure(
    tmp_path: Path,
) -> None:
    """A path that does not exist yet, and a slug with nothing in it, both answer.

    A drift command pointed at an empty archive should say "there is nothing archived" and not
    raise, because the two are different situations and the caller has a remedy for one of them.
    Asserted on both halves -- the file is not created by asking, and the question answers
    whether or not it was -- because a constructor that created an empty log would turn a read
    into a write.
    """
    archive = _archive(tmp_path)
    assert not archive.path.exists()
    assert archive.readings(CENSUS_SLUG) == ()
    assert archive.slugs() == ()
    assert not archive.path.exists()
