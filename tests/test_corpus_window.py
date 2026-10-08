"""The corpus's window in time, because a rate over an unknown span is not a rate.

Every rate this program publishes states its population in records, and not one of
them stated it in time -- so a corpus three weeks old and a corpus three years old
produced identical figures and a reader could not tell them apart. A corpus installed
last Tuesday is an ordinary state for a knowledge store, because Kojutsu writes on
webhook events and the corpus begins whenever the webhook was installed. This file is
the set of shapes that fix, each one named for what it protects rather than for what
it does.

**The two halves of the window are the whole of it: how wide the read is, and how long
ago it ended.** The width is what makes a rate interpretable -- 3 of 12 is a different
claim over three weeks than over three years -- and the age is what makes a *complete*
read suspicious, because a store that is reachable and has captured nothing for a
quarter is a capture path that stopped, and completeness is structurally blind to it:
the read did reach the end of the store. That is the same observation as the
event-shape detector's in a different dimension, so it is the shape that already
existed with the dimension missing, not a new idea.

**The unknown window is a distinct object, not a zero-length one.** A corpus whose
every timestamp would not parse spans an unknown amount of time, and a reader told
"span: 0" has been handed a duration nobody measured. The type keeps the two apart by
refusing to express one as the other, and the tests below assert the refusal as well
as the rendering.

**The read-model test is the one that cannot be got right by accident.** A read model
is a *past* read, and a figure computed from one has to report the window of the read
that produced it rather than the window of the machine rendering it. That is not a
statement about a clock -- :mod:`tenbin.corpus.window` holds none -- and it is the
only path in the program where the moment of the read and the moment of the render can
differ by months. Every fixture here is dated far enough in the past that a ``now()``
would be visibly wrong, because a test that passes today and fails in ninety days is a
test that was never asserting the thing.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from tenbin.corpus.record import Record
from tenbin.corpus.snapshot import Completeness, CorpusSnapshot
from tenbin.corpus.window import (
    NO_TIME_AT_ALL,
    CorpusWindow,
    record_moment,
    render_duration,
)
from tenbin.measures.base import (
    CountFigure,
    FigureGroup,
    subset_denominator,
    subset_window,
)
from tenbin.measures.completeness import CompletenessMeasure
from tenbin.measures.integrity import (
    CAPTURE_FRESHNESS_CLEAN_SLUG,
    CAPTURE_FRESHNESS_SLUG,
    MAX_CORPUS_SILENCE,
    CaptureFreshnessMeasure,
    detect_capture_silence,
)
from tenbin.measures.trust import TrustProfileMeasure
from tenbin.readmodel import build, to_snapshot
from tenbin.report.document import CorpusHeader, Report
from tenbin.report.markdown import render_markdown
from tenbin.report.ordering import PartKind
from tests.fixtures import DEFAULT_UPDATED_AT, FIXED_READ_AT, build_record, build_snapshot

_FRESHNESS = CaptureFreshnessMeasure()

#: A moment inside the corpus's own history, for the tests that want a timestamp they
#: can reason about without holding a snapshot. A constant rather than
#: ``now() - something``, because a fixture that computes its own moments is a fixture
#: whose figures change as the clock does.
INSIDE_HISTORY = "2026-09-20T09:00:00+00:00"

#: A value that is not a timestamp, for the corpus whose window cannot be read. Real
#: prose rather than an empty string, because a writer who typed a date in words is a
#: different failure from a writer who left the key out, and both arrive as ``None``.
NOT_A_TIMESTAMP = "the day before yesterday, roughly"

#: Where the three-week corpus starts: twenty-one days before the read, less the three
#: hours the newest record sits behind it. Named so the arithmetic in
#: :func:`_three_week_history` reads as the fact it is rather than as a subtraction.
THREE_WEEKS_AGO = FIXED_READ_AT - timedelta(days=21, hours=3)


def _answer(index: int, *, answered_at: str | None = INSIDE_HISTORY, **extras: Any) -> Record:
    """One dated record, addressed as its own change so no two ids collide."""
    return build_record(
        entry_id=f"answer-{index:04d}",
        pr=index + 1,
        record_kind=None,
        answered_at=answered_at,
        **extras,
    )


def _undated(index: int) -> Record:
    """One record whose only timestamp is not a timestamp at all."""
    return build_record(
        entry_id=f"answer-{index:04d}",
        pr=index + 1,
        record_kind=None,
        updated_at=NOT_A_TIMESTAMP,
    )


def _naming_a_file(when: str, *, index: int, path: str = "src/queue.py") -> Record:
    """One record that names a file, which is what makes it part of a *subset*.

    The file list goes through ``frontmatter`` rather than through an extra, because an
    extra is stringified the way the store stringifies it -- and ``"['src/queue.py']"``
    is a string, so a fixture that used one would be building a record with no files and
    would not know it.
    """
    return build_record(
        entry_id=f"answer-{index:04d}",
        pr=index + 1,
        record_kind=None,
        answered_at=when,
        frontmatter={"files": [path]},
    )


def _three_week_history() -> CorpusSnapshot:
    """A corpus that began three weeks before the read, which is what a webhook gives you.

    Twenty-two records spanning exactly twenty-one days and ending three hours before the
    read. This is the ordinary state of a knowledge store installed recently, and it is the
    state in which "a stated reason was never revised" and "there has been no time in which
    to revise one" produce the same figure.
    """
    records = tuple(
        _answer(index, answered_at=(THREE_WEEKS_AGO + timedelta(days=index)).isoformat())
        for index in range(22)
    )
    return build_snapshot(records)


# -- the window is stated, and the header is where it is stated ----------------------


def test_a_rate_over_an_unknown_window_is_not_a_rate_because_the_header_states_the_span_the_records_cover() -> (
    None
):
    """Criterion 1: the header carries both ends and the distance between them.

    Asserted on the rendered report rather than on the window object, because the thing
    being protected is that a *reader* meets the span. A window computed correctly and
    never rendered is the same defect in a different costume.
    """
    snapshot = build_snapshot(
        (
            _answer(1, answered_at="2026-09-01T00:00:00+00:00"),
            _answer(2, answered_at="2026-09-18T06:00:00+00:00"),
        )
    )

    rendered = render_markdown(Report(corpus=CorpusHeader.from_snapshot(snapshot), sections=()))

    assert "2026-09-01T00:00:00+00:00" in rendered
    assert "2026-09-18T06:00:00+00:00" in rendered
    assert "a window of 17 days, 6 hours" in rendered


def test_the_header_states_the_window_on_the_enumeration_line_because_a_count_and_a_span_are_one_population() -> (
    None
):
    """The window sits beside the count: on one part, count first and window second.

    Asserted by kind *and* by position, because the failure being guarded against is a
    header that carries the window somewhere harmless -- a reader looking for it would
    find it, and a test that only looked for the string would pass.
    """
    snapshot = _three_week_history()

    parts = CorpusHeader.from_snapshot(snapshot).parts()

    assert [part.kind for part in parts].index(PartKind.enumeration) == 2
    enumeration = parts[2]
    assert enumeration.text.splitlines()[0] == "22 of 22 documents"
    assert "Records span" in enumeration.text.splitlines()[1]
    assert "a window of 21 days" in enumeration.text


def test_a_corpus_with_no_history_is_a_different_claim_from_a_corpus_with_a_clean_one_because_three_weeks_are_visible() -> (
    None
):
    """Criterion 3: a corpus that began when the webhook was installed says so.

    Asserted on the *duration* rather than on the two timestamps alone, because a header
    naming two ends a reader has to subtract is a header that requires arithmetic, and
    the reader who does not do the arithmetic is the reader this program exists for. The
    age is stated in the same line for the same reason: "spans 21 days" and "newest
    record is 3 hours old" are the two halves of "this store is young".
    """
    snapshot = _three_week_history()

    text = CorpusHeader.from_snapshot(snapshot).parts()[2].text

    assert "a window of 21 days" in text
    assert "The newest of them is 3 hours before this read." in text


def test_a_whole_corpus_rate_does_not_repeat_the_window_because_the_header_carries_it_once_for_all_of_them() -> (
    None
):
    """Criterion 5: twenty measures repeating it is a report nobody reads.

    Checked against real whole-corpus measures rather than against a literal, because
    the failure is a measure author *choosing* to add it and a test over a string would
    not notice the choice.
    """
    snapshot = _three_week_history()

    denominators = [
        TrustProfileMeasure().claim(snapshot).denominator.render_text(),
        CompletenessMeasure().claim(snapshot).denominator.render_text(),
        _FRESHNESS.claim(snapshot).denominator.render_text(),
    ]

    assert snapshot.window.render_text() not in denominators
    for rendered in denominators:
        assert "Records span" not in rendered
        # And no date leaks in by another route, which is the shape the same sentence
        # would take if a measure pasted the window into its description by hand.
        assert "2026-" not in rendered
    # The information is not lost by not repeating it: the header carries it once.
    assert snapshot.window.render_text() in CorpusHeader.from_snapshot(snapshot).parts()[2].text


# -- a subset states its own window --------------------------------------------------


def test_a_population_narrower_than_the_corpus_states_its_own_window_because_a_corpus_header_does_not_describe_a_subset() -> (
    None
):
    """Criterion 4, over a per-file population: the custody measure's shape.

    Built from real records rather than from a hand-written denominator, because the
    failure is a measure that *forgot* to say its window and a test inspecting a string
    it assembled itself would only be testing the assembly.
    """
    january = _naming_a_file("2026-01-04T00:00:00+00:00", index=1)
    january_later = _naming_a_file("2026-01-20T00:00:00+00:00", index=2)
    september = _answer(3, answered_at="2026-09-25T00:00:00+00:00")
    snapshot = build_snapshot((january, january_later, september))
    files = [record for record in snapshot.records if record.files]

    denominator = subset_denominator("files named by records in this read", len(files), files)

    # The subset's own span, sixteen days, and not the corpus's two hundred and sixty
    # four: the corpus reaches back to January and the subset stops in the middle of it.
    assert subset_window(files).span == timedelta(days=16)
    assert "a window of 16 days" in denominator.render_text()
    assert "2026-09-25" not in denominator.render_text()
    assert snapshot.window.span == timedelta(days=264)
    assert "2026-09-25" in snapshot.window.render_text()


def test_a_subset_window_that_cannot_be_read_says_so_rather_than_reporting_a_span_of_zero() -> None:
    """An unmeasurable subset is not a subset that spans no time.

    The same rule as the corpus's own window, applied one level down, and the reason the
    helper is a function: a subset measure reaching for ``timedelta(0)`` when it has
    nothing to measure produces a confident wrong number about a population it never
    placed in time.
    """
    undated = (_undated(1), _undated(2))

    window = subset_window(undated)

    assert window.is_known is False
    assert window.span is None
    rendered = subset_denominator("records in this subset", 2, undated).render_text()
    assert "Window unknown" in rendered
    assert "no time at all" not in rendered


# -- a corpus that is up and silent --------------------------------------------------


def test_a_store_that_is_reachable_and_capturing_nothing_is_a_finding_because_completeness_cannot_see_it() -> (
    None
):
    """Criterion 2, and the case the whole detector exists for.

    The snapshot is deliberately ``COMPLETE`` with agreeing counts, because that is what
    makes the finding necessary rather than redundant: a walk that reached the end of
    the store is the most complete read available and says nothing whatever about
    whether anything is still being written into it.
    """
    snapshot = build_snapshot(
        (
            _answer(1, answered_at="2026-03-01T00:00:00+00:00"),
            _answer(2, answered_at="2026-02-01T00:00:00+00:00"),
        )
    )
    assert snapshot.is_complete

    finding = detect_capture_silence(snapshot)

    assert finding is not None
    assert finding.claim.slug == CAPTURE_FRESHNESS_SLUG
    assert "213 days, 12 hours of silence" in finding.what_was_observed
    assert finding.cannot_confirm.strip()
    assert finding.snapshot is snapshot
    assert finding.where is not None and "acme/widget" in finding.where


def test_the_freshness_check_is_a_measure_and_not_only_a_function_because_a_check_nothing_invokes_looks_finished() -> (
    None
):
    """The finding has to reach a report, which is the reason the measure class exists.

    The same argument as the lifecycle-shape and projection-safety measures: a detector
    with no registry entry is a function that looks finished, and a report that never ran
    the check renders a quiet corpus exactly like a report that ran it.
    """
    snapshot = build_snapshot((_answer(1, answered_at="2026-03-01T00:00:00+00:00"),))

    figure = _FRESHNESS.compute(snapshot)

    assert isinstance(figure, FigureGroup)
    assert figure.claim.slug == CAPTURE_FRESHNESS_SLUG
    assert _FRESHNESS.claim(snapshot).slug == CAPTURE_FRESHNESS_SLUG
    assert _FRESHNESS.about_the_capture_system is True


def test_a_corpus_whose_newest_record_is_thirty_seconds_old_does_not_fire_because_a_quiet_repository_is_ordinary() -> (
    None
):
    """The bound is a judgement about noise and this is the case it exists to exclude.

    Thirty seconds of silence and ninety days of silence produce the same sentence if the
    detector has no threshold, and a detector with no threshold gets switched off -- which
    leaves the reader with no detector at all.
    """
    snapshot = build_snapshot(
        (
            _answer(1, answered_at=(FIXED_READ_AT - timedelta(seconds=30)).isoformat()),
            _answer(2, answered_at="2026-09-01T00:00:00+00:00"),
        )
    )

    assert detect_capture_silence(snapshot) is None
    assert _FRESHNESS.claim(snapshot).slug == CAPTURE_FRESHNESS_CLEAN_SLUG


def test_a_corpus_whose_window_cannot_be_read_does_not_fire_because_an_unknown_window_is_not_a_stopped_path() -> (
    None
):
    """Criterion 6, and the reason the window is ``None``-able rather than zeroed.

    Every record here would be inside the bound if the one timestamp that does not parse
    were believed, and not one of them can be placed at all. A detector that inferred
    "nothing has been captured for months" from that would be reporting an observation it
    did not make -- and the clean claim it returns says the check could not run rather
    than that the path is healthy.
    """
    snapshot = build_snapshot((_undated(1), _undated(2), _undated(3)))

    assert snapshot.window.is_known is False
    assert detect_capture_silence(snapshot) is None
    claim = _FRESHNESS.claim(snapshot)
    assert claim.slug == CAPTURE_FRESHNESS_CLEAN_SLUG
    assert "no record in it carries a moment this reader could place at all" in claim.statement
    assert "healthy" in claim.does_not_mean


def test_a_clean_freshness_check_still_renders_a_figure_because_checked_and_found_nothing_is_a_fact() -> (
    None
):
    """The clean case is a count of zero with a claim, never an empty section.

    The same rule the other two detectors in this package are held to: a reader who
    cannot tell "the check ran and the path is not quiet" from "there is nothing here" has
    been handed an absence where a fact belongs, and the absence is the one that reads
    like a clean corpus.
    """
    snapshot = build_snapshot((_answer(1),))

    figure = _FRESHNESS.compute(snapshot)

    assert isinstance(figure, CountFigure)
    assert figure.value == 0
    assert figure.claim.slug == CAPTURE_FRESHNESS_CLEAN_SLUG
    assert figure.claim.denominator.size == 1


def test_the_freshness_denominator_counts_only_the_records_whose_age_the_check_could_compute() -> (
    None
):
    """A record with no readable moment has no age, so it is not in the population.

    The exclusion is a denominator and not a bucket, for the reason the whole package
    holds: a record dropped from the population without appearing anywhere is a filter
    that rendered as a result.
    """
    snapshot = build_snapshot((_answer(1), _answer(2), _undated(3), _undated(4)))

    claim = _FRESHNESS.claim(snapshot)

    assert claim.denominator.size == 2
    assert claim.denominator.description == (
        "records in this read that carried a moment this reader could place"
    )


def test_a_quiet_corpus_that_also_holds_undated_records_names_because_the_finding_counts_what_it_could_not_date() -> (
    None
):
    """The exclusion is in the observation, not merely in the denominator.

    A corpus where half the documents carry no readable moment and the other half stopped
    six months ago is the ordinary messy case, and a finding that reported the silence
    without saying how many records it could not weigh would be reporting a rate over a
    population it never described.
    """
    snapshot = build_snapshot(
        (
            _answer(1, answered_at="2026-03-01T00:00:00+00:00"),
            _undated(2),
            _undated(3),
        )
    )

    finding = detect_capture_silence(snapshot)

    assert finding is not None
    assert finding.claim.denominator.size == 1
    assert "1 capture(s) this reader could place in time" in finding.what_was_observed
    assert "2 that carried no moment at all" in finding.what_was_observed


def test_a_quiet_corpus_whose_newest_record_names_no_repository_does_not_invent_a_location_for_it() -> (
    None
):
    """``where`` is optional, and Kojutsu's ``unknown`` placeholder is not a place.

    The same defect the ``unknown``-model bug wore in its third costume: a record filed at
    ``unknown/pr-1/...`` has the *string* ``"unknown"`` for a repository, and rendering it
    as one would hand a reader a location to go to that does not exist.
    """
    snapshot = build_snapshot(
        (
            _answer(1, repo=None, answered_at="2026-03-01T00:00:00+00:00"),
            _answer(2, answered_at="2026-01-01T00:00:00+00:00"),
        )
    )

    finding = detect_capture_silence(snapshot)

    assert finding is not None
    assert finding.where is None


def test_a_record_dated_after_the_read_does_not_fire_because_clock_skew_is_not_a_stopped_path() -> (
    None
):
    """A negative age is a data defect, and rendering it as a long silence inverts it."""
    snapshot = build_snapshot(
        (
            _answer(1, answered_at="2026-01-01T00:00:00+00:00"),
            _answer(2, answered_at=(FIXED_READ_AT + timedelta(days=30)).isoformat()),
        )
    )

    assert snapshot.window.age_at(snapshot.read_at) == timedelta(days=-30)
    assert detect_capture_silence(snapshot) is None
    assert "after this read" in snapshot.window.age_text(snapshot.read_at)


def test_the_bound_is_ninety_days_because_it_is_a_judgement_about_noise_and_naming_it_is_how_it_gets_argued_with() -> (
    None
):
    """The constant is pinned by a test, so changing it is a deliberate and visible act.

    A threshold inlined at the comparison cannot be argued with: the next person edits a
    number inside a condition and the corpus that used to report a finding stops reporting
    one, with no diff anywhere that says the reporting rule changed.
    """
    assert timedelta(days=90) == MAX_CORPUS_SILENCE


# -- unknown is not zero --------------------------------------------------------------


def test_a_corpus_with_no_readable_timestamps_reports_an_unknown_window_and_not_a_span_of_zero() -> (
    None
):
    """Criterion 6 in the type: ``None`` is not ``timedelta(0)``.

    Asserted on both halves, because either alone would pass a weaker test: the rendering
    could say "unknown" while ``span`` still returned zero to a measure that never looks
    at the prose, and a measure that bucketed a duration of zero is the bug this shape was
    introduced to prevent.
    """
    snapshot = build_snapshot((_undated(1), _undated(2), _undated(3)))

    window = snapshot.window

    assert window.earliest is None
    assert window.latest is None
    assert window.span is None
    assert window.unreadable_timestamps == 3
    assert "Window unknown" in window.render_text()
    assert "0 seconds" not in window.render_text()
    assert NO_TIME_AT_ALL not in window.render_text()
    assert window.age_text(FIXED_READ_AT) == ""


def test_a_corpus_whose_every_record_shares_one_instant_spans_no_time_at_all_and_says_so() -> None:
    """The zero-length window is a real object, and it is not the unknown one.

    A store holding a thousand documents all written in the same second is possible, and
    it is a different fact from a store whose timestamps would not parse. Rendering both
    as "no window" would lose the difference, and the difference is the difference between
    a corpus that spans no time and a corpus nobody could measure.
    """
    same = "2026-09-20T09:00:00+00:00"
    snapshot = build_snapshot((_answer(1, answered_at=same), _answer(2, answered_at=same)))

    window = snapshot.window

    assert window.is_known is True
    assert window.span == timedelta(0)
    span = window.span
    assert span is not None
    assert render_duration(span) == NO_TIME_AT_ALL
    assert NO_TIME_AT_ALL in window.render_text()
    assert "Window unknown" not in window.render_text()
    # And it is still a real, dated window: the age is the read's, not nothing's.
    assert window.age_at(FIXED_READ_AT) == timedelta(days=10, hours=3)


def test_a_window_with_one_end_cannot_be_built_because_a_half_read_window_would_have_its_other_end_guessed() -> (
    None
):
    """Both ends or neither, because the missing one is not recoverable.

    Defaulting ``latest`` to the read's own moment would produce a plausible span for
    every corpus in the program and a true one for none of them.
    """
    with pytest.raises(ValueError, match="both be readable or neither"):
        CorpusWindow(earliest=FIXED_READ_AT, latest=None, unreadable_timestamps=0)


def test_a_window_that_ends_before_it_begins_cannot_be_built_because_that_is_a_contradiction() -> (
    None
):
    with pytest.raises(ValueError, match="contradiction"):
        CorpusWindow(
            earliest=FIXED_READ_AT,
            latest=FIXED_READ_AT - timedelta(days=1),
            unreadable_timestamps=0,
        )


def test_a_window_holding_a_naive_moment_cannot_be_built_because_a_moment_that_cannot_be_placed_cannot_bound_anything() -> (
    None
):
    """A naive moment is a writer that forgot an offset, and guessing UTC is a guess."""
    with pytest.raises(ValueError, match="aware datetime"):
        CorpusWindow(
            earliest=datetime(2026, 9, 1),
            latest=datetime(2026, 9, 2),
            unreadable_timestamps=0,
        )


def test_a_negative_count_of_unreadable_timestamps_cannot_be_built_because_no_record_failed_to_be_read() -> (
    None
):
    with pytest.raises(ValueError, match="must not be negative"):
        CorpusWindow(earliest=None, latest=None, unreadable_timestamps=-1)


def test_a_count_of_unreadable_timestamps_written_as_words_cannot_be_built_because_it_is_a_denominator() -> (
    None
):
    """The field is a count, and a boolean is not one however it arrives.

    ``True`` is an ``int`` in Python, and a window claiming one unreadable timestamp
    because somebody passed a flag would render a number nobody counted -- in a program
    whose whole argument is that a number with no condition attached is how an exclusion
    becomes a result.
    """
    with pytest.raises(ValueError, match="must be a count"):
        CorpusWindow(  # type: ignore[arg-type]
            earliest=None, latest=None, unreadable_timestamps="three"
        )
    with pytest.raises(ValueError, match="must be a count"):
        CorpusWindow(  # type: ignore[arg-type]
            earliest=None, latest=None, unreadable_timestamps=True
        )


def test_a_record_stamped_at_the_same_instant_as_the_read_is_said_to_be_so_rather_than_being_rounded_to_zero() -> (
    None
):
    """Zero seconds of age is an observation, and it is not the same as no age at all.

    The same instant is a real fact about a corpus that was written to during the walk,
    and saying "0 seconds before this read" would read as a rounding of something rather
    than as the thing it is.
    """
    window = CorpusWindow.from_records((_answer(1, answered_at=FIXED_READ_AT.isoformat()),))

    assert window.age_text(FIXED_READ_AT) == (
        "The newest of them is stamped at the same instant as this read."
    )


def test_records_with_no_readable_moment_are_counted_rather_than_dropped_because_a_filter_must_leave_a_trace() -> (
    None
):
    """A record that left the window's population appears as a number, not as an absence.

    This is the ``unknown``-model defect in a new costume: two records whose timestamps
    would not parse and one whose would, reported as a window that never mentions the two,
    is a rate over a population smaller than the one anybody is reading about.
    """
    snapshot = build_snapshot(
        (
            _answer(1, answered_at="2026-09-01T00:00:00+00:00"),
            _undated(2),
            _answer(3, answered_at="2026-09-18T00:00:00+00:00"),
        )
    )

    window = snapshot.window

    assert window.unreadable_timestamps == 1
    assert "1 record carried no moment" in window.render_text()
    assert window.span == timedelta(days=17)


def test_a_read_that_enumerated_no_records_says_its_window_is_unknown_because_an_empty_read_spans_nothing_at_all() -> (
    None
):
    """Zero records and zero readable timestamps are different facts with different readers.

    One is a store that was reached and holds nothing; the other is a store whose documents
    arrived without a moment. Sending an operator to look for a broken timestamp reader on
    an empty collection is how a check gets ignored.
    """
    snapshot = build_snapshot(())

    assert snapshot.window.unreadable_timestamps == 0
    assert "enumerated no records" in snapshot.window.render_text()
    assert "none of the" not in snapshot.window.render_text()


def test_a_header_written_by_hand_still_states_a_window_rather_than_saying_nothing_because_a_silent_line_reads_like_a_decision() -> (
    None
):
    """A header with no window is possible, and it says the window is unknown.

    The alternative -- omitting the line -- renders a report that looks like a header that
    was asked and had nothing to say, which is not the same as one that was never asked.
    """
    from tenbin.corpus.snapshot import Completeness as _Completeness

    header = CorpusHeader(
        collection="chronicler-real",
        read_at=FIXED_READ_AT,
        enumerated=3,
        store_total=3,
        completeness=_Completeness.COMPLETE,
    )

    enumeration = header.parts()[2]
    assert enumeration.text.splitlines()[0] == "3 of 3 documents"
    assert "Window unknown" in enumeration.text


# -- a record's place in time --------------------------------------------------------


def test_a_record_is_placed_by_the_event_it_is_about_rather_than_by_its_own_update_time() -> None:
    """The same preference :attr:`~tenbin.corpus.record.Record.month` uses, and for the same reason.

    ``updated_at`` moves on every push, so a corpus measured by it is measured by its write
    activity. The capture and the declaration are weaker facts about other clocks, in that
    order, and this is the one place in the program where the order is written down for the
    window.
    """
    record = build_record(
        entry_id="answer-0001",
        record_kind=None,
        answered_at="2026-05-05T00:00:00+00:00",
        updated_at=DEFAULT_UPDATED_AT,
    )

    assert record_moment(record) == datetime(2026, 5, 5, tzinfo=UTC)


def test_a_record_naming_no_moment_at_all_is_placed_nowhere_rather_than_at_the_read() -> None:
    assert record_moment(_undated(1)) is None


# -- a window is computed once, so measures cannot disagree about it ------------------


def test_twenty_measures_get_the_same_window_because_it_is_computed_once_and_not_per_measure() -> (
    None
):
    """A window twenty measures computed independently is twenty windows that can disagree.

    Asserted on object identity rather than on equality, because equality would also hold
    for two separately computed windows and identity cannot: the snapshot derives it once
    and every reader receives that one object.
    """
    snapshot = _three_week_history()

    assert snapshot.window is snapshot.window
    assert snapshot.window is CorpusHeader.from_snapshot(snapshot).window
    assert snapshot.window is _FRESHNESS.compute(snapshot).snapshot.window


def test_a_window_cannot_be_supplied_by_a_caller_because_two_accounts_of_one_read_would_be_one_too_many() -> (
    None
):
    """``init=False``: a snapshot whose window disagreed with its records is a contradiction.

    The disagreement would be invisible from outside -- every header would render from the
    window while every figure counted the records -- and it is the same reason
    :mod:`tenbin.readmodel` derives its indexes rather than accepting them.
    """
    with pytest.raises(TypeError):
        CorpusSnapshot(
            records=(),
            collection="chronicler-real",
            read_at=FIXED_READ_AT,
            store_total=0,
            enumerated=0,
            completeness=Completeness.COMPLETE,
            window=CorpusWindow(  # type: ignore[call-arg]  # deliberately not a field
                earliest=None, latest=None, unreadable_timestamps=0
            ),
        )


# -- a read model reports the window of the read, not of the render ------------------


def test_a_figure_from_a_stored_read_reports_that_read_s_window_because_now_is_not_a_fact_about_a_read() -> (
    None
):
    """Criterion 7: the read model is six months old and its figures must not notice.

    The model is built from a read whose newest record is 194 days old and is then handed
    back to the measures, and every figure it produces still describes that read. The
    finding still fires too, because what it reports is the read's silence and not the
    render's -- which is the direction a shortcut would have got wrong, in the one case
    where getting it wrong is invisible until somebody reads a stale report.
    """
    read = build_snapshot(
        (
            _answer(1, answered_at="2026-03-01T00:00:00+00:00"),
            _answer(2, answered_at="2026-03-20T00:00:00+00:00"),
        )
    )
    restored = to_snapshot(build(read))

    assert restored.read_at == read.read_at
    assert restored.window == read.window
    assert restored.window.span == timedelta(days=19)
    assert restored.window.age_at(restored.read_at) == timedelta(days=194, hours=12)
    assert detect_capture_silence(restored) is not None
    assert "2026-03-20T00:00:00+00:00" in CorpusHeader.from_snapshot(restored).parts()[2].text


def test_a_read_model_that_carried_no_window_still_restores_one_because_a_window_is_what_the_records_say() -> (
    None
):
    """The window follows the records through the file, and the schema did not change.

    ``tenbin/readmodel/**`` is a package this ticket does not hold and it did not have to
    change: the model stores the records, a window is derived from the records, and a
    snapshot rebuilds it from exactly those. The alternative -- a stored window -- would be
    a second account of the same records in a file, and a file holding two accounts of one
    fact is a file somebody has to reconcile.
    """
    read = build_snapshot((_answer(1, answered_at="2026-04-01T00:00:00+00:00"),))
    model = build(read)

    assert not hasattr(model, "window")
    assert to_snapshot(model).window == read.window
    assert to_snapshot(model).window.unreadable_timestamps == 0


def test_two_reads_of_one_corpus_report_different_ages_because_an_age_is_measured_and_not_stored() -> (
    None
):
    """The window is stored on the snapshot; the age is an argument, so it moves.

    A stored age would be whichever of the two the writer happened to run, and the second
    of these two snapshots is exactly the case that would render it as a claim about the
    machine rather than about the read.
    """
    records = (_answer(1, answered_at="2026-05-01T00:00:00+00:00"),)
    earlier = build_snapshot(records)
    later_snapshot = CorpusSnapshot(
        records=records,
        collection=earlier.collection,
        read_at=earlier.read_at + timedelta(days=10),
        store_total=earlier.store_total,
        enumerated=earlier.enumerated,
        completeness=earlier.completeness,
    )

    assert earlier.window == later_snapshot.window
    assert earlier.window.age_at(earlier.read_at) == timedelta(days=152, hours=12)
    assert later_snapshot.window.age_at(later_snapshot.read_at) == timedelta(days=162, hours=12)
    # And the freshness check follows the read, so a corpus that was quiet a fortnight ago
    # and is quiet now fires from both -- the property the measure loses the moment it
    # reaches for a clock, and the one a stale report would have been built on.
    assert detect_capture_silence(earlier) is not None
    assert detect_capture_silence(later_snapshot) is not None


# -- the rendering ------------------------------------------------------------------


def test_a_duration_is_rendered_in_the_two_largest_units_somebody_would_say_aloud() -> None:
    """Legibility over precision, and a number rather than a rounded one.

    "29 days" and "29 days, 2 hours" are different claims about the same corpus and only the
    second is true; a duration rendered to the second invites a comparison between two
    windows that differ by noise.
    """
    assert render_duration(timedelta(days=3)) == "3 days"
    assert render_duration(timedelta(days=1, hours=2)) == "1 day, 2 hours"
    assert render_duration(timedelta(hours=1)) == "1 hour"
    assert render_duration(timedelta(minutes=90)) == "1 hour, 30 minutes"
    assert render_duration(timedelta(seconds=45)) == "45 seconds"
    assert render_duration(timedelta(milliseconds=400)) == "under a second"
    assert render_duration(timedelta(0)) == NO_TIME_AT_ALL
