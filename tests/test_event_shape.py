"""That the event-shape detector finds the shape and refuses to name the cause.

This module exists because of a correction. The shape is real: ten or more
``opened`` transitions and not one ``closed`` is not a history a working repository
produces. The cause that was originally hardcoded into the detector is not -- or
rather, it *was* and has been fixed. Kojutsu's ``semantic_pr_event_id`` built its
identity preimage with ``json.dumps`` and folded in ``datetime`` values it cannot
encode, so every timestamped close and reopen raised and was never written, and a
close is the event that says a change was adopted or abandoned. Markers are now
normalised through ``_transition_marker`` first, and Kojutsu's own docstring
names the bug.

A detector that hardcoded that cause would now fire and be confidently wrong. So
these tests check three things separately: that the shape is detected, that the
historical cause is named *as a hypothesis that has been fixed* rather than as the
answer, and that ``cannot_confirm`` is non-empty. The last is the one that stops the
whole failure mode, and it is asserted on its own rather than as part of a larger
check, because a finding whose limits are optional is a finding whose author could
not say what they did not know.
"""

from __future__ import annotations

from tenbin.corpus.record import LifecycleAction, Record, RecordKind
from tenbin.measures.integrity import (
    MIN_OPENS_FOR_INFERENCE,
    LifecycleShapeDetector,
)
from tests.fixtures import build_record, build_snapshot

_DETECTOR = LifecycleShapeDetector()


def _opens(count: int, *, repo: str = "acme/widget") -> tuple[Record, ...]:
    """``count`` lifecycle records that opened a change, with distinct pull requests.

    Distinct changes rather than one change opened many times, because a change
    opened and closed ten times is a different shape from ten changes opened once --
    and the second is the one that should not happen.
    """
    return tuple(
        build_record(
            entry_id=f"pr-event-{index:04d}",
            pr=1000 + index,
            repo=repo,
            record_kind=RecordKind.PR_LIFECYCLE,
            tags=["pr_state_change", f"action_{LifecycleAction.OPENED.value}"],
            pr_opened_at="2026-09-01T09:00:00+00:00",
        )
        for index in range(count)
    )


def _closes(count: int = 1, *, repo: str = "acme/widget") -> tuple[Record, ...]:
    """``count`` lifecycle records that closed a change."""
    return tuple(
        build_record(
            entry_id=f"pr-event-closed-{index:04d}",
            pr=9000 + index,
            repo=repo,
            record_kind=RecordKind.PR_LIFECYCLE,
            tags=["pr_state_change", f"action_{LifecycleAction.CLOSED.value}"],
            pr_outcome="merged",
        )
        for index in range(count)
    )


def test_the_detector_fires_on_opens_with_no_closes_because_that_shape_is_not_a_working_repository() -> (
    None
):
    """The shape, on a corpus of changes that opened and never closed.

    The threshold and the closed count are both asserted, because the alternative
    implementations are the plausible ones: a detector that fires below the
    threshold trains its readers to ignore it, and one that tolerates a few closes
    would keep firing on a repository where close captures are merely patchy.
    """
    records = _opens(MIN_OPENS_FOR_INFERENCE)
    finding = _DETECTOR.detect(records, snapshot=build_snapshot(records))
    assert finding is not None
    assert f"{MIN_OPENS_FOR_INFERENCE} pull-request transitions" in finding.what_was_observed
    assert "none closed one" in finding.what_was_observed


def test_a_detector_that_fires_on_noise_gets_switched_off_because_that_is_what_a_quiet_week_looks_like() -> (
    None
):
    """One short of the threshold is silent, and the threshold is a named constant.

    A busy week with no merges in a small repository is an ordinary event, and a
    detector that reported it would reach its third false alarm and be muted. So the
    bar is ten and the constant is named and commented, so the next person argues
    with the number rather than with the code.
    """
    assert MIN_OPENS_FOR_INFERENCE == 10
    quiet = _opens(MIN_OPENS_FOR_INFERENCE - 1)
    assert _DETECTOR.detect(quiet, snapshot=build_snapshot(quiet)) is None


def test_a_single_close_silences_the_detector_because_a_close_is_the_thing_whose_absence_is_the_finding() -> (
    None
):
    """Not "few closes" -- no closes at all.

    A partially working capture path produces some closes and some not, and that
    corpus is a different finding: the write path is running and missing some
    events. This detector does not make that finding, and it must not make this one
    either, or the two would be indistinguishable in a report.
    """
    records = _opens(MIN_OPENS_FOR_INFERENCE) + _closes(1)
    assert _DETECTOR.detect(records, snapshot=build_snapshot(records)) is None


def test_the_suspected_cause_names_the_historical_defect_and_says_it_has_been_fixed_because_a_hardcoded_cause_is_now_a_wrong_answer() -> (
    None
):
    """The correction, asserted on the words.

    Three things have to be present or the sentence is wrong in one direction or
    the other: the historical cause, the fix, and the fact that the fix means this
    corpus *cannot* be explained by it. A cause stated without the fix reads as a
    live diagnosis; a fix stated without the cause throws away the one thing a
    reader can actually check.
    """
    records = _opens(MIN_OPENS_FOR_INFERENCE)
    finding = _DETECTOR.detect(records, snapshot=build_snapshot(records))
    assert finding is not None
    cause = finding.suspected_cause
    assert cause is not None
    assert "json.dumps" in cause
    assert "TypeError" in cause
    assert "_transition_marker" in cause
    assert "is fixed" in cause
    assert "cannot be explained by it" in cause


def test_the_suspected_cause_lists_the_causes_that_are_still_live_because_one_candidate_is_not_an_explanation() -> (
    None
):
    """Four candidates, and three of them are not in the commit log.

    A store that genuinely merged nothing in the period; a capture path broken for a
    different reason, so the close never reached the writer rather than the writer
    refusing it; a collection filtered to opened transitions. Each produces exactly
    this shape, and a finding that named only the memorable one would send a reader
    to a bug that has been fixed for a while.
    """
    records = _opens(MIN_OPENS_FOR_INFERENCE)
    finding = _DETECTOR.detect(records, snapshot=build_snapshot(records))
    assert finding is not None
    cause = finding.suspected_cause or ""
    assert "merged and abandoned nothing" in cause
    assert "capture path is broken" in cause
    assert "filtered to opened transitions" in cause
    assert "are not excluded by anything here" in cause


def test_cannot_confirm_is_non_empty_because_tenbin_read_a_store_and_did_not_run_kojutsus_code() -> (
    None
):
    """The field that stops the finding being a diagnosis.

    Tenbin reads Tanseki. It does not execute Kojutsu's code, does not see the
    forge's payloads, and cannot tell a writer that raised from a writer that was
    never called. A finding whose author cannot write that sentence has either
    confirmed something or decided to imply it had, and to a reader those are the
    same sentence. So the field is required, cannot be blank, and is asserted
    non-empty here in its own right rather than as part of a larger check.
    """
    records = _opens(MIN_OPENS_FOR_INFERENCE)
    finding = _DETECTOR.detect(records, snapshot=build_snapshot(records))
    assert finding is not None
    assert finding.cannot_confirm.strip()
    assert "did not execute Kojutsu's code" in finding.cannot_confirm
    assert "not a diagnosis" in finding.cannot_confirm


def test_the_finding_names_where_the_opens_were_because_a_finding_with_no_location_is_about_a_collection_in_the_abstraction() -> (
    None
):
    """A reader's next question is always "which repository", and it is answered here.

    The largest bucket rather than a list, because a list of ten repositories in a
    finding is a table the reader has to reduce themselves. ``None`` when the opens
    are spread evenly, which is the honest answer for a corpus with no such place
    and better than naming the largest of thirty near-identical buckets as though it
    were the story.
    """
    concentrated = _opens(7, repo="acme/widget") + _opens(3, repo="acme/gadget")
    finding = _DETECTOR.detect(concentrated, snapshot=build_snapshot(concentrated))
    assert finding is not None
    assert finding.where == "7 of 10 opens are in `acme/widget`"

    spread = tuple(
        build_record(
            entry_id=f"pr-event-{index:04d}",
            pr=1000 + index,
            repo=f"acme/repo-{index}",
            record_kind=RecordKind.PR_LIFECYCLE,
            tags=["pr_state_change", "action_opened"],
        )
        for index in range(MIN_OPENS_FOR_INFERENCE)
    )
    even = _DETECTOR.detect(spread, snapshot=build_snapshot(spread))
    assert even is not None
    assert even.where == "1 of 10 opens are in `acme/repo-0`"


def test_the_finding_carries_the_read_so_it_can_say_whether_the_corpus_behind_the_shape_is_whole() -> (
    None
):
    """The snapshot is required, and a shape found in a prefix of a corpus is a different finding.

    A walk stopped by the offset ceiling would find ten opens and no closes in the
    first thousand documents whether or not the corpus goes on to record a close.
    That is not a weaker version of this finding, it is a different one, and the
    completeness sentence is what tells the two apart. So the snapshot is a required
    argument rather than something the detector can be called without.
    """
    records = _opens(MIN_OPENS_FOR_INFERENCE)
    finding = _DETECTOR.detect(records, snapshot=build_snapshot(records))
    assert finding is not None
    assert "Whole read" in finding.rate_text()


def test_the_finding_carries_a_claim_like_any_other_figure_because_a_finding_published_without_limits_is_the_most_wrong_sentence_here() -> (
    None
):
    """Four claim fields, a denominator over the lifecycle records, and a falsifier.

    A finding is a claim that something is wrong, and it is held to the whole
    apparatus. The denominator is the lifecycle records rather than the whole
    corpus: a finding about one shape of one kind of record does not get to borrow
    the population of every other kind, and the falsifier is a single close
    appearing later, because one close makes this shape a quiet period rather than a
    broken path.
    """
    records = _opens(MIN_OPENS_FOR_INFERENCE)
    finding = _DETECTOR.detect(records, snapshot=build_snapshot(records))
    assert finding is not None
    assert finding.claim.slug == "corpus.lifecycle_opens_without_closes"
    assert finding.claim.denominator.size == MIN_OPENS_FOR_INFERENCE
    assert "no change was merged" in finding.claim.does_not_mean
    assert "A single closed transition" in finding.claim.falsifier


def test_a_reopen_does_not_count_as_a_close_because_a_reopen_is_not_the_end_of_anything() -> None:
    """Three transitions are stored and only one of them ends a change.

    A corpus with ten opens and ten reopens has nothing to report: a reopen says a
    change came back, which is a fact about the start rather than the end. Counting
    reopens towards "closes" would silence the detector on exactly the corpus it
    exists for.
    """
    reopens = tuple(
        build_record(
            entry_id=f"pr-event-reopen-{index:04d}",
            pr=8000 + index,
            record_kind=RecordKind.PR_LIFECYCLE,
            tags=["pr_state_change", f"action_{LifecycleAction.REOPENED.value}"],
        )
        for index in range(3)
    )
    records = _opens(MIN_OPENS_FOR_INFERENCE) + reopens
    assert _DETECTOR.detect(records, snapshot=build_snapshot(records)) is not None


def test_records_that_are_not_lifecycle_records_are_ignored_because_the_shape_is_about_transitions() -> (
    None
):
    """Ten answers are not ten opens.

    The detector counts by transition action, and a record whose kind is something
    else cannot carry one. So a corpus of ten answers produces no finding, which is
    the negative case a detector that counted records rather than transitions would
    get wrong in the loudest way available.
    """
    answers = tuple(
        build_record(pr=index + 1, repo="acme/widget") for index in range(MIN_OPENS_FOR_INFERENCE)
    )
    assert _DETECTOR.detect(answers, snapshot=build_snapshot(answers)) is None


def test_a_legacy_transition_counted_from_its_tags_because_the_shape_does_not_depend_on_how_the_kind_was_recovered() -> (
    None
):
    """A record read as a lifecycle transition from its tag projection is still one.

    The other two detectors in this package require ``DETERMINED`` kinds, and here
    they must not: a transition recovered from the ``pr_state_change`` tag is
    exactly as reliable as one that carried the key, and the shape this detector
    looks for does not depend on how the kind was recovered at all. Requiring
    ``DETERMINED`` would discard every transition written before the key existed
    while catching nothing.
    """
    legacy = tuple(
        build_record(
            entry_id=f"pr-event-legacy-{index:04d}",
            pr=7000 + index,
            record_kind=None,
            tags=["pr_state_change", "action_opened"],
        )
        for index in range(MIN_OPENS_FOR_INFERENCE)
    )
    assert all(record.classification.certainty.value == "determined" for record in legacy)
    found = _DETECTOR.detect(legacy, snapshot=build_snapshot(legacy))
    assert found is not None
    assert LifecycleShapeDetector.requires_certainty is None


def test_a_finding_with_no_repository_to_point_at_says_none_rather_than_naming_a_bucket_because_thirty_even_buckets_are_not_a_place() -> (
    None
):
    """``where`` is optional, and a corpus spread over thirty repositories has none.

    Ten opens, one per repository: ``most_common`` would return the first, and a
    reader would take "1 of 10 opens are in ``acme/repo-0``" as the story. Naming a
    repository the opens are *not* concentrated in is worse than naming none, and
    ``None`` says the honest thing: this shape is not localised.
    """
    spread = tuple(
        build_record(
            entry_id=f"pr-event-{index:04d}",
            pr=1000 + index,
            repo=None,
            record_kind=RecordKind.PR_LIFECYCLE,
            tags=["pr_state_change", "action_opened"],
        )
        for index in range(MIN_OPENS_FOR_INFERENCE)
    )
    finding = _DETECTOR.detect(spread, snapshot=build_snapshot(spread))
    assert finding is not None
    assert finding.where is None
    assert "Where" not in finding.value_text()
