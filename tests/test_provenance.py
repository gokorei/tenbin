"""How far a record's claim to be evidence can be taken, counted rather than judged.

Kojutsu writes a ``capture_source`` on every record and re-checks the anchors
on the read path, precisely so a reader can tell a capture from a claim. These
tests pin the two halves of that: the rules, which mirror Kojutsu's own
``capture_anchor_gaps`` so both sides of the seam share one definition of
"captured means checkable"; and the counting, which is the part a measurement
program is actually made of.

The counting half is where the interesting failures live. Anomalous records are
counted, never filtered, because dropping them would make every rate here a rate
over a population nobody had described. And the numeric leniency is not
tolerance for its own sake: the store stringifies Kojutsu's extras, so a
strict validator would flag every pull request in the corpus, and a validator
that reports everything reports nothing -- which is worse than no validator,
because its output looks like a finding.
"""

from __future__ import annotations

from typing import Any

import pytest

from tenbin.corpus.provenance import (
    CAPTURE_ASSERTED,
    CAPTURE_COLLECT,
    CAPTURE_WEBHOOK,
    NO_MONTH,
    NO_REPOSITORY,
    AnomalyKind,
    anchor_gaps,
    provenance_profile,
    validate,
)
from tenbin.corpus.record import Record
from tests.fakes import document_id, kojutsu_frontmatter, record_from_frontmatter


def _webhook_record(**overrides: Any) -> Record:
    """A fully anchored webhook capture -- the clean case, and the baseline."""
    frontmatter = kojutsu_frontmatter(
        tags=["agent_authored"],
        capture_source=CAPTURE_WEBHOOK,
        delivery_id="delivery-1",
        captured_at="2026-09-28T00:00:00+00:00",
        answered_at="2026-09-28T00:00:00+00:00",
        independence="independent",
    )
    frontmatter.update(overrides)
    return record_from_frontmatter(frontmatter)


# -- validation -------------------------------------------------------------


def test_a_fully_anchored_webhook_capture_validates_clean() -> None:
    """The baseline: a capture that can be re-checked has nothing to report.

    Worth a test of its own because a validator that flags everything is
    indistinguishable from one that works, and this is the case it must not
    touch.
    """
    assert validate(_webhook_record()) == frozenset()


def test_a_record_with_no_capture_source_is_reported_as_unknown_not_as_asserted() -> None:
    """Silence about how a record came to exist is not a claim that it was typed in.

    Defaulting the missing key to ``asserted`` would let a document with its
    provenance key deleted pass as a documented claim about intent, and would
    make the anomaly invisible in a corpus where it is the interesting thing.
    """
    found = validate(record_from_frontmatter(kojutsu_frontmatter(capture_source=None)))
    assert AnomalyKind.CAPTURE_SOURCE_UNKNOWN in found


def test_a_blank_capture_source_counts_as_missing_because_a_placeholder_is_not_a_claim() -> None:
    """Kojutsu's writer skips empty values, so a blank key means a hand-written document.

    Which is exactly the record where a validator earns its keep: it is the one
    that must not be read as review evidence.
    """
    found = validate(record_from_frontmatter(kojutsu_frontmatter(capture_source="   ")))
    assert found == frozenset({AnomalyKind.CAPTURE_SOURCE_UNKNOWN})


def test_a_capture_source_outside_the_vocabulary_is_invalid_and_not_unknown() -> None:
    """A fourth source is a different problem from a missing one.

    ``unknown`` says nothing was written; ``invalid`` says something was written
    that this reader cannot interpret, and only the second one is worth going and
    looking at.
    """
    found = validate(record_from_frontmatter(kojutsu_frontmatter(capture_source="telepathy")))
    assert AnomalyKind.CAPTURE_SOURCE_INVALID in found
    assert AnomalyKind.CAPTURE_SOURCE_UNKNOWN not in found


def test_a_webhook_capture_missing_its_delivery_id_is_unanchored_and_names_the_gap() -> None:
    """The delivery id is the one thing that makes a signed delivery re-fetchable.

    Without it the record claims a capture that nothing can check, and the gap is
    named as well as counted so a report can say which key was empty.
    """
    record = record_from_frontmatter(
        kojutsu_frontmatter(
            tags=["agent_authored"],
            capture_source=CAPTURE_WEBHOOK,
            captured_at="2026-09-28T00:00:00+00:00",
        )
    )
    assert "delivery_id" in anchor_gaps(record)
    assert AnomalyKind.CAPTURE_ANCHOR_MISSING in validate(record)


def test_a_collect_capture_is_not_required_to_carry_a_delivery_id() -> None:
    """The ``collect`` path is an authenticated API read, not a provider delivery.

    There is no delivery to record, so demanding one would flag every honestly
    collected record -- and a validator that flags the honest records is a
    validator whose output nobody will read.
    """
    record = record_from_frontmatter(
        kojutsu_frontmatter(
            tags=["agent_authored"],
            capture_source=CAPTURE_COLLECT,
            captured_at="2026-09-28T00:00:00+00:00",
            github_comment_id="2400123",
        )
    )
    assert validate(record) == frozenset()
    assert anchor_gaps(record) == ()


def test_a_collect_capture_without_its_comment_id_is_unanchored() -> None:
    """The comment id is what makes a collected answer checkable, by re-fetching it."""
    record = record_from_frontmatter(
        kojutsu_frontmatter(
            tags=["agent_authored"],
            capture_source=CAPTURE_COLLECT,
            captured_at="2026-09-28T00:00:00+00:00",
        )
    )
    assert "github_comment_id" in anchor_gaps(record)
    assert AnomalyKind.CAPTURE_ANCHOR_MISSING in validate(record)


def test_every_asserted_record_requires_no_anchor_at_all() -> None:
    """``asserted`` claims no capture, so there is nothing for it to anchor.

    It is the default Kojutsu gives a record, and a rationale is permanently
    ``asserted``. Requiring anchors here would make every rationale anomalous and
    the anomaly meaningless.
    """
    record = record_from_frontmatter(
        kojutsu_frontmatter(
            tags=["rationale", "rationale_declared"],
            capture_source=CAPTURE_ASSERTED,
        ),
        doc_id=document_id("acme/widget", 42, "rationale-v1-abc", rationale=True),
    )
    assert anchor_gaps(record) == ()
    assert AnomalyKind.CAPTURE_ANCHOR_MISSING not in validate(record)


def test_a_capture_with_no_pull_request_is_unanchored_because_there_is_nothing_to_re_check() -> (
    None
):
    """A capture with no change attached cannot be traced to anything the provider knows."""
    record = record_from_frontmatter(
        kojutsu_frontmatter(
            tags=["agent_authored"],
            capture_source=CAPTURE_WEBHOOK,
            delivery_id="delivery-1",
            captured_at="2026-09-28T00:00:00+00:00",
            pr=0,
        )
    )
    assert "pr" in anchor_gaps(record)


def test_a_capture_with_no_capture_time_is_unanchored() -> None:
    """When the capture was seen is part of what makes it checkable."""
    record = record_from_frontmatter(
        kojutsu_frontmatter(
            tags=["agent_authored"],
            capture_source=CAPTURE_WEBHOOK,
            delivery_id="delivery-1",
        )
    )
    assert "captured_at" in anchor_gaps(record)


def test_a_pull_request_number_is_read_in_both_spellings_because_the_store_stringifies() -> None:
    """An int-only reader would flag every stored change, and so would report nothing.

    The two spellings are the same value: Kojutsu stringifies every frontmatter
    extra on the way into the store, so a document read back carries ``"42"``.
    """
    as_string = record_from_frontmatter(
        kojutsu_frontmatter(
            tags=["agent_authored"],
            capture_source=CAPTURE_WEBHOOK,
            delivery_id="delivery-1",
            captured_at="2026-09-28T00:00:00+00:00",
            pr="42",
        )
    )
    assert validate(as_string) == frozenset()

    as_number = record_from_frontmatter(
        kojutsu_frontmatter(
            tags=["agent_authored"],
            capture_source=CAPTURE_WEBHOOK,
            delivery_id="delivery-1",
            captured_at="2026-09-28T00:00:00+00:00",
            pr=42,
        )
    )
    assert as_number.frontmatter["pr"] == "42"
    assert validate(as_number) == frozenset()


def test_a_comment_id_that_is_not_a_number_counts_as_absent_rather_than_being_coerced() -> None:
    """Zero is a placeholder and ``True`` is a boolean, so neither is comment 1."""
    for value in ("0", "not-a-comment", True):
        record = record_from_frontmatter(
            kojutsu_frontmatter(
                tags=["agent_authored"],
                capture_source=CAPTURE_COLLECT,
                captured_at="2026-09-28T00:00:00+00:00",
                github_comment_id=value,
            )
        )
        assert "github_comment_id" in anchor_gaps(record), value


def test_a_blank_delivery_id_counts_as_missing_because_a_placeholder_is_not_an_anchor() -> None:
    """The key is present, so a naive reader would accept the record as anchored."""
    record = record_from_frontmatter(
        kojutsu_frontmatter(
            tags=["agent_authored"],
            capture_source=CAPTURE_WEBHOOK,
            delivery_id="  ",
            captured_at="2026-09-28T00:00:00+00:00",
        )
    )
    assert "delivery_id" in anchor_gaps(record)


def test_an_independence_level_on_an_asserted_record_is_a_claim_nothing_derived() -> None:
    """Independence is computed by the capture pipeline from the posting accounts.

    A level on a record claiming no capture was not computed at all -- somebody
    typed it -- and that is the one thing the axis exists to distinguish. A
    reader that reported it as a captured independence level would be reporting
    an assertion as a derivation.
    """
    record = record_from_frontmatter(
        kojutsu_frontmatter(
            tags=["agent_authored"],
            capture_source=CAPTURE_ASSERTED,
            independence="independent",
            independence_reason="a second party looked at it",
        )
    )
    assert AnomalyKind.INDEPENDENCE_WITHOUT_CAPTURE in validate(record)


def test_a_blank_independence_level_is_an_absence_and_not_a_contradiction() -> None:
    """An empty key is nobody having claimed a level, which is not a claim about one.

    Flagging it would report every hand-written record as contradictory, and the
    axis is only useful while it stays rare enough to mean something.
    """
    record = record_from_frontmatter(
        kojutsu_frontmatter(
            tags=["agent_authored"], capture_source=CAPTURE_ASSERTED, independence="  "
        )
    )
    assert AnomalyKind.INDEPENDENCE_WITHOUT_CAPTURE not in validate(record)


def test_an_independence_level_with_no_capture_source_at_all_is_still_a_contradiction() -> None:
    """The capture source is unknown, so the independence level has nothing behind it."""
    found = validate(
        record_from_frontmatter(
            kojutsu_frontmatter(capture_source=None, independence="independent")
        )
    )
    assert AnomalyKind.INDEPENDENCE_WITHOUT_CAPTURE in found
    assert AnomalyKind.CAPTURE_SOURCE_UNKNOWN in found


def test_the_unknown_repository_placeholder_counts_as_no_repository() -> None:
    """Kojutsu writes the literal where it had no repository to write.

    A bucket labelled ``unknown`` is a claim about a principal that no writer
    made, and a record that cannot be joined to a change cannot be checked against
    anything, so it belongs in the anomaly count.
    """
    found = validate(
        record_from_frontmatter(kojutsu_frontmatter(tags=["agent_authored"], repo="unknown", pr=42))
    )
    assert AnomalyKind.REPO_MISSING in found


def test_a_record_whose_id_carries_no_repository_segment_is_reported_as_missing_one() -> None:
    """A document written outside Kojutsu has no repository, and saying so is the point."""
    found = validate(
        record_from_frontmatter(
            kojutsu_frontmatter(tags=["agent_authored"]),
            doc_id="scratch/note.md",
        )
    )
    assert AnomalyKind.REPO_MISSING in found


def test_a_record_may_carry_several_anomalies_at_once_and_gets_all_of_them() -> None:
    """Collapsing a record's problems into one label hides the ones that need different fixes.

    A hand-written document with no capture source, no repository and a
    self-asserted independence level is three findings, and a report that named
    only the first would send somebody to look in the wrong place.
    """
    found = validate(
        record_from_frontmatter(
            kojutsu_frontmatter(capture_source=None, repo=None, independence="independent"),
            doc_id="scratch/note.md",
        )
    )
    assert AnomalyKind.CAPTURE_SOURCE_UNKNOWN in found
    assert AnomalyKind.REPO_MISSING in found
    assert AnomalyKind.INDEPENDENCE_WITHOUT_CAPTURE in found


def test_validation_is_a_pure_function_so_the_same_record_always_reads_the_same_way() -> None:
    """A validator whose answer depends on when it was called cannot be checked.

    The whole point of this layer is that a report can state which records were
    anomalous, and that statement is worthless if re-running the validation on
    the same records gave a different answer.
    """
    record = _webhook_record()
    assert validate(record) == validate(record) == frozenset()
    assert anchor_gaps(record) == anchor_gaps(record) == ()


# -- the profile ------------------------------------------------------------


def test_anomalous_records_are_counted_rather_than_filtered_out() -> None:
    """Dropping them would make every rate here a rate over an undescribed population.

    The clean count and the total are on the same object for exactly that reason:
    a caller who quotes one has the other in hand.
    """
    records = [
        _webhook_record(),
        record_from_frontmatter(kojutsu_frontmatter(capture_source=None)),
        record_from_frontmatter(kojutsu_frontmatter(capture_source="telepathy")),
    ]
    profile = provenance_profile(records)
    assert (profile.total, profile.clean, profile.anomalous) == (3, 1, 2)


def test_every_anomaly_kind_is_present_in_the_profile_even_at_zero() -> None:
    """A shape that depends on the corpus is a shape every report has to guard against.

    A stable mapping means a row for every kind means the same thing on a clean
    corpus as on a broken one, so a report can print all five without special
    casing the empty case.
    """
    profile = provenance_profile([])
    assert set(profile.anomalies_by_kind) == set(AnomalyKind)
    assert all(count == 0 for count in profile.anomalies_by_kind.values())
    assert profile.total == profile.clean == 0


def test_the_anomalous_records_are_kept_so_a_histogram_bucket_can_be_opened() -> None:
    """ "Eight records in ``core`` have no repository" is a finding; the eight are the work.

    Keeping them is what makes the bucket actionable, and it is why
    :func:`provenance_profile` consumes an iterable once rather than re-deriving
    the anomalies from a second pass over something that may have changed.
    """
    broken = record_from_frontmatter(kojutsu_frontmatter(capture_source=None))
    profile = provenance_profile([broken])
    assert profile.records == (broken,)


def test_the_repository_histogram_counts_the_anomalous_records_and_nothing_else() -> None:
    """The population is named in the field, because ``by_repository`` invites the other reading.

    A repository with three anomalous records is three times the problem of one
    with a single one, and a concentration of broken provenance in one repository
    is a fact about that repository.
    """
    records = [
        _webhook_record(),
        _webhook_record(),
        record_from_frontmatter(kojutsu_frontmatter(capture_source=None, repo="broken/one", pr=1)),
        record_from_frontmatter(kojutsu_frontmatter(capture_source=None, repo="broken/one", pr=2)),
        record_from_frontmatter(kojutsu_frontmatter(capture_source=None, repo="broken/two", pr=3)),
    ]
    profile = provenance_profile(records)
    assert profile.anomalous_by_repository == {"broken/one": 2, "broken/two": 1}


def test_a_record_with_no_repository_lands_in_an_explicit_bucket_rather_than_an_empty_one() -> None:
    """A missing repository is not an empty string, and the two must not share a key.

    Collapsing them would let a report present "records with no repository" as a
    repository, which is precisely the sort of thing this project exists to catch
    in somebody else's programme.
    """
    profile = provenance_profile(
        [record_from_frontmatter(kojutsu_frontmatter(capture_source=None), doc_id="x")]
    )
    assert profile.anomalous_by_repository == {NO_REPOSITORY: 1}


def test_a_record_with_no_timestamp_lands_in_an_explicit_month_bucket() -> None:
    """An undated record is not a record from January 1970, and not from this month."""
    profile = provenance_profile(
        [
            record_from_frontmatter(
                kojutsu_frontmatter(capture_source=None, updated_at=None, repo=None, pr=None),
                doc_id="x",
            )
        ]
    )
    assert profile.anomalous_by_month == {NO_MONTH: 1}


def test_the_month_histogram_is_ordered_by_frequency_so_the_worst_bucket_is_first() -> None:
    """A histogram whose first row is arbitrary is a histogram a reader has to sort.

    Ordering is by count and then by key, so two profiles over equal data are
    equal -- which is what makes a profile comparable in a test rather than
    merely comparable by eye.
    """
    records = [
        record_from_frontmatter(
            kojutsu_frontmatter(
                capture_source=None,
                repo="a/b",
                pr=1,
                answered_at="2026-01-05T00:00:00+00:00",
            )
        ),
        record_from_frontmatter(
            kojutsu_frontmatter(
                capture_source=None,
                repo="a/b",
                pr=2,
                answered_at="2026-02-05T00:00:00+00:00",
            )
        ),
        record_from_frontmatter(
            kojutsu_frontmatter(
                capture_source=None,
                repo="a/b",
                pr=3,
                answered_at="2026-02-06T00:00:00+00:00",
            )
        ),
    ]
    assert list(provenance_profile(records).anomalous_by_month) == ["2026-02", "2026-01"]


def test_the_profile_is_frozen_against_a_later_mutation_of_its_own_counters() -> None:
    """A profile is a statement about a corpus at a moment, and it must not drift after it.

    The histograms are handed out as read-only mappings precisely so that a
    caller cannot quietly adjust the denominator it is about to divide by.
    """
    profile = provenance_profile(
        [record_from_frontmatter(kojutsu_frontmatter(capture_source=None))]
    )
    with pytest.raises(TypeError):
        profile.anomalous_by_repository[NO_REPOSITORY] = 99  # type: ignore[index]


def test_a_profile_accepts_any_iterable_and_consumes_it_once() -> None:
    """A generator is the natural thing to hand a function like this, and it has to work.

    Consuming twice would double-count or raise on the second pass, and either
    way the caller would get a number that depends on the container they happened
    to use.
    """
    records = (_webhook_record() for _ in range(3))
    profile = provenance_profile(records)
    assert (profile.total, profile.clean) == (3, 3)


def test_a_profile_is_hashable_because_a_frozen_dataclass_claims_to_be_and_is_not() -> None:
    """A profile holds read-only mappings, so the generated hash would raise at runtime.

    Defining it on the population and the shape is better than letting a set of
    profiles blow up two modules away from here. Two profiles over equal data hash
    alike because equality already implies equal counts.
    """
    from tenbin.corpus.provenance import provenance_profile

    records = [record_from_frontmatter(kojutsu_frontmatter(capture_source=None))]
    assert len({provenance_profile(records), provenance_profile(records)}) == 1
