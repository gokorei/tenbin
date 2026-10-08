"""Recovering what a record *is* from what the writer happened to store.

Kojutsu writes a ``record_kind`` on every record and throws it away at the
storage boundary, so a reader holding a document is holding a tag set and a
guess. Most of this module is about keeping that guess visible: the tags are
Kojutsu's own projection of a kind it already held, so recovering one is
reading rather than inferring -- except for the answer, whose tag projection is
empty, and where "no tags" could mean an answer written by someone who named no
agent or any other record that lost its tags. That one case is the reason
:class:`Certainty` exists, and a test below fails the day it is quietly promoted
to a determination.

The other half is leniency. The store stringifies every frontmatter extra, so a
pull request number arrives as ``"42"`` and a timestamp as an ISO string with a
``+00:00`` offset, and a reader that insists on the writer's original types would
call the whole corpus malformed. A malformed field therefore yields ``None`` and
the record survives: one bad field must cost a field, never a denominator.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from tenbin.corpus.record import (
    Certainty,
    DocumentPath,
    LifecycleAction,
    Record,
    RecordKind,
    ReviewState,
    classify,
    entry_id_prefix,
    lifecycle_action_from_tags,
    parse_document_id,
    rationale_source_from_tags,
    review_state_from_tags,
)
from tenbin.corpus.values import Stated, Unstated, parse_stated
from tenbin.store.client import StoreDocument
from tests.fakes import document_id, kojutsu_frontmatter, record_from_frontmatter

# -- classification ---------------------------------------------------------


def test_an_answer_with_no_tags_is_inferred_because_nothing_wrote_the_kind_down() -> None:
    """The one kind whose tag projection is empty is the one that cannot be read.

    An answer written by somebody who named no agent carries no tags at all, and
    so does a record of any other kind that lost them. Calling that ``DETERMINED``
    would put a conclusion about a corpus in the corpus's own mouth, which is the
    failure ``docs/corpus-inventory.md`` §1 records as the ambiguity the answer
    case creates.
    """
    result = classify(None)
    assert result.kind is RecordKind.ANSWER
    assert result.certainty is Certainty.INFERRED
    assert result.evidence == ("no tags",)


def test_the_agent_authored_tag_states_the_kind_so_the_reading_is_determined() -> None:
    """A tag Kojutsu projected from a kind it held is a determination, not a guess.

    ``agent_authored`` answers "who wrote this", which is orthogonal to the kind,
    and Kojutsu adds it only to answers -- so a record carrying it is stating
    that it is one.
    """
    result = classify(["agent_authored"])
    assert result.kind is RecordKind.ANSWER
    assert result.certainty is Certainty.DETERMINED
    assert result.evidence == ("agent_authored",)


def test_a_rationale_is_named_by_its_own_tag_and_carries_its_source_as_evidence() -> None:
    """The evidence has to name the tag, or a reader cannot check the reading.

    ``rationale_declared`` is the refinement the writer adds, and dropping it
    from the evidence would leave a classification that could not be traced back
    to the document without re-deriving the tag convention.
    """
    result = classify(["rationale", "rationale_declared"])
    assert result.kind is RecordKind.RATIONALE
    assert result.evidence == ("rationale", "rationale_declared")
    assert "declared" in result.detail


@pytest.mark.parametrize(
    ("tags", "action"),
    [
        (["pr_state_change", "action_opened"], LifecycleAction.OPENED),
        (["pr_state_change", "action_closed"], LifecycleAction.CLOSED),
        (["pr_state_change", "action_reopened"], LifecycleAction.REOPENED),
    ],
)
def test_a_lifecycle_record_is_named_by_its_transition_tag(
    tags: list[str], action: LifecycleAction
) -> None:
    """Three transitions and no others, because those are the three that are stored."""
    result = classify(tags)
    assert result.kind is RecordKind.PR_LIFECYCLE
    assert result.certainty is Certainty.DETERMINED
    assert lifecycle_action_from_tags(tags) is action


def test_an_inline_comment_outranks_a_verdict_because_it_is_the_more_specific_tag() -> None:
    """Both carry ``review``; only one of them is about a line of code.

    A record carrying both tags would otherwise be classified by whichever branch
    ran first, and the more specific statement is the one the writer bothered to
    make.
    """
    result = classify(["review", "inline_comment", "review_state_approved"])
    assert result.kind is RecordKind.INLINE_REVIEW_COMMENT


def test_a_review_verdict_carries_the_state_it_stated() -> None:
    """A verdict without its state is a record of feedback, and must not read as approval."""
    result = classify(["review", "review_state_changes_requested"])
    assert result.kind is RecordKind.REVIEW_VERDICT
    assert result.evidence == ("review", "review_state_changes_requested")
    assert review_state_from_tags(["review", "review_state_changes_requested"]) is (
        ReviewState.CHANGES_REQUESTED
    )


def test_a_review_tag_with_no_state_this_reader_knows_is_still_a_review_record() -> None:
    """An unfamiliar state is unknown, not absent, and not "commented".

    Filing it under a state this reader happens not to have would be a guess
    about somebody's verdict, and a verdict is exactly the thing a measurement
    program must not guess at.
    """
    result = classify(["review", "review_state_dismissed"])
    assert result.kind is RecordKind.REVIEW_VERDICT
    assert result.certainty is Certainty.DETERMINED
    assert review_state_from_tags(["review", "review_state_dismissed"]) is None
    assert "dismissed" not in result.detail or "no verdict state" in result.detail


def test_a_review_tag_alone_is_a_verdict_and_not_an_answer() -> None:
    """``review`` is a statement about the kind, and the kind is not an answer."""
    assert classify(["review"]).kind is RecordKind.REVIEW_VERDICT


def test_precedence_runs_rationale_then_lifecycle_then_review_then_answer() -> None:
    """The most specific tag wins, so a mixed record is read as the thing it is.

    A record carrying every kind's tags is a contradiction, and the only stable
    answer to a contradiction is a documented order rather than whatever the
    branches happen to be written in.
    """
    every_tag = [
        "rationale",
        "pr_state_change",
        "review",
        "inline_comment",
        "agent_authored",
    ]
    assert classify(every_tag).kind is RecordKind.RATIONALE
    assert classify([tag for tag in every_tag if tag != "rationale"]).kind is (
        RecordKind.PR_LIFECYCLE
    )
    assert classify(["review", "inline_comment", "agent_authored"]).kind is (
        RecordKind.INLINE_REVIEW_COMMENT
    )
    assert classify(["review", "agent_authored"]).kind is RecordKind.REVIEW_VERDICT
    assert classify(["agent_authored"]).kind is RecordKind.ANSWER


def test_tags_that_name_no_kind_are_carried_as_evidence_rather_than_discarded() -> None:
    """An unrecognised tag is an anomaly, and the kind is still an answer.

    The evidence field exists so a reader can see that something was written and
    that this program could not read it -- an empty evidence tuple would make the
    inference indistinguishable from a record with no tags at all.
    """
    result = classify(["retention_policy_v2"])
    assert result.kind is RecordKind.ANSWER
    assert result.certainty is Certainty.INFERRED
    assert result.evidence == ("retention_policy_v2",)


def test_unusable_tag_entries_are_dropped_so_an_unreadable_record_reads_as_unreadable() -> None:
    """A tag that is not text was not written by Kojutsu's tag projection.

    Coercing it would invent a reading, and a record whose tags are *all*
    unusable is therefore in the same position as one with no tags: the kind
    cannot be recovered, and the classification says so.
    """
    result = classify(["", "   ", None, 7])  # type: ignore[list-item]
    assert result.certainty is Certainty.INFERRED
    assert result.evidence == ("no tags",)


def test_a_repeated_tag_is_listed_once_so_evidence_cannot_inflate() -> None:
    """Two copies of one tag is one piece of evidence, and a report would double-count it."""
    assert classify(["agent_authored", "agent_authored"]).evidence == ("agent_authored",)


@pytest.mark.parametrize(
    ("tags", "expected"),
    [
        (["rationale", "rationale_reconstructed"], "reconstructed"),
        (["rationale", "rationale_unknown"], "unknown"),
        (["rationale"], None),
    ],
)
def test_a_rationale_source_is_recovered_from_its_tag_and_is_allowed_to_be_unknown(
    tags: list[str], expected: str | None
) -> None:
    """``rationale_unknown`` is a real stated source, not an absence.

    A rationale written before the axis existed says so explicitly, and that is a
    weaker claim than having said nothing -- which is why it comes back as the
    string ``"unknown"`` here rather than as :class:`Unstated`.
    """
    assert rationale_source_from_tags(tags) == expected


# -- document id parsing ----------------------------------------------------


def test_the_namespace_regex_does_not_assume_a_fixed_segment_count() -> None:
    """A repository is ``owner/name``, so a real id has five segments, not four.

    A regex written against four would reject every document in a real corpus,
    and the failure would be indistinguishable from an empty one.
    """
    parsed = parse_document_id("acme/widget/pr-42/answer-ab12cd")
    assert parsed == DocumentPath(repo="acme/widget", pr=42, entry_id="answer-ab12cd")


def test_a_pr_zero_placeholder_parses_to_no_pull_request_rather_than_to_zero() -> None:
    """``pr-0`` is how Kojutsu writes "no pull request", and pull request zero does not exist.

    Reading it as the number zero would invent a change that is not there and put
    it at the bottom of every per-change histogram.
    """
    assert parse_document_id("acme/widget/pr-0/answer-ab12cd").pr is None
    assert parse_document_id("acme/widget/pr-0/answer-ab12cd").repo == "acme/widget"


def test_a_rationale_lives_under_its_own_segment_so_its_kind_is_readable_from_the_id() -> None:
    """The segment is the only thing in the id that says "reason" rather than "answer"."""
    parsed = parse_document_id("acme/widget/pr-42/rationale/rationale-v1-abc")
    assert parsed.is_rationale is True
    assert parsed.entry_id == "rationale-v1-abc"
    assert parsed.pr == 42


def test_a_repository_named_like_a_pull_request_does_not_confuse_the_parser() -> None:
    """The ``pr-`` segment is found by pattern, and the *last* match wins.

    A repository path could contain a segment spelled ``pr-1``; taking the first
    match would read the repository as the change.
    """
    parsed = parse_document_id("org/pr-1/widget/pr-42/answer-ab12cd")
    assert parsed.repo == "org/pr-1/widget"
    assert parsed.pr == 42


def test_an_id_with_no_change_segment_still_yields_a_record() -> None:
    """A document written outside Kojutsu has no such shape and is still a document.

    The entry id is recovered so the record is addressable, and the repository
    and pull request are honestly absent rather than invented.
    """
    parsed = parse_document_id("scratch/notes.md")
    assert parsed.repo is None
    assert parsed.pr is None
    assert parsed.entry_id == "notes.md"


def test_an_empty_id_parses_to_nothing_rather_than_raising() -> None:
    """Nothing here is allowed to raise: a record with an unreadable id is still countable."""
    assert parse_document_id("") == DocumentPath()


def test_an_id_with_a_trailing_segment_after_the_entry_is_not_misread() -> None:
    """A malformed tail must not be turned into a plausible entry id for a rationale."""
    parsed = parse_document_id("acme/widget/pr-42/extra/rationale/deeper/answer-x")
    assert parsed.is_rationale is False
    assert parsed.entry_id == "answer-x"


@pytest.mark.parametrize(
    ("entry_id", "prefix"),
    [
        ("answer-ab12", "answer-"),
        ("review-ab12", "review-"),
        ("pr-event-legacy", "pr-event-"),
        ("rationale-v1-ab12", "rationale-v1-"),
        ("mystery-1", None),
        ("", None),
        (None, None),
    ],
)
def test_the_entry_id_prefix_names_the_writer_when_it_is_one_we_know(
    entry_id: str | None, prefix: str | None
) -> None:
    """The id's namespace is a second, independent statement about the record's kind.

    A record whose prefix and tags disagree is a contradiction between two
    functions that both knew the kind; nothing here resolves it, and the prefix
    exists so a reader can see it.
    """
    assert entry_id_prefix(entry_id) == prefix


# -- values: stated versus absent -------------------------------------------


def test_the_literal_unknown_a_writer_wrote_is_read_as_an_absence_not_as_a_model() -> None:
    """Kojutsu fills the field with that string wherever nothing was stated.

    Read as a value, it becomes a group of records whose authors declared a model
    called ``"unknown"`` -- a claim about a principal that nobody made, and the
    exact mistake ``docs/corpus-inventory.md`` §5 names.
    """
    assert parse_stated("unknown") is Unstated()


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_an_absent_or_blank_value_is_an_absence_because_a_blank_name_is_not_a_name(
    raw: object,
) -> None:
    """All three arrive in real documents: the key skipped, written empty, or padded."""
    assert parse_stated(raw) is Unstated()


def test_a_non_string_where_a_model_name_belongs_states_no_model() -> None:
    """Stringifying a number would manufacture a principal called ``"42"``."""
    assert parse_stated(42) is Unstated()
    assert parse_stated(["gpt"]) is Unstated()


def test_a_stated_model_survives_as_a_wrapped_value_and_is_falsy_only_when_unstated() -> None:
    """The two cases have to be told apart at the point of use, not reconstructed there."""
    stated = parse_stated(" claude-opus ")
    assert isinstance(stated, Stated)
    assert stated.value == "claude-opus"
    assert bool(stated) is True
    assert bool(Unstated()) is False


def test_unstated_is_a_singleton_so_two_absences_are_the_same_value() -> None:
    """Two sentinels for one absence is the defect this type exists to prevent.

    A copy is a copy, not a new sentinel: ``copy`` and ``deepcopy`` both hand back
    the same object, so a record deep-copied for a report still compares equal by
    identity to one that was not.
    """
    import copy

    assert Unstated() is Unstated()
    assert copy.copy(Unstated()) is Unstated()
    assert copy.deepcopy(Unstated()) is Unstated()


def test_unstated_does_not_equal_none_so_an_absence_of_a_field_is_not_an_absence_of_a_model() -> (
    None
):
    """``None`` is a fact about a field; ``Unstated`` is a fact about the record.

    Collapsing them is how a program ends up asserting a model was unknown when
    the truth is that no key was ever written.
    """
    assert Unstated() != None  # noqa: E711 - the comparison is the assertion
    assert Unstated() != ""


def test_a_stated_value_is_orderable_and_hashable_so_it_can_be_counted_and_sorted() -> None:
    """A report groups and sorts by model, and both need more than a bare string."""
    values = [parse_stated("b"), parse_stated("a")]
    ordered = sorted(value for value in values if isinstance(value, Stated))
    assert [value.value for value in ordered] == ["a", "b"]
    assert len({Stated("a"), Stated("a"), Unstated(), Unstated()}) == 2


# -- record parsing ---------------------------------------------------------


def test_a_webhook_capture_reads_its_anchors_from_the_stringified_frontmatter() -> None:
    """The store stringifies its extras, so a pull request arrives as ``"42"``.

    A reader that required an ``int`` would call every stored change unanchored,
    and a validator that reports everything reports nothing.
    """
    record = record_from_frontmatter(
        kojutsu_frontmatter(
            tags=["agent_authored"],
            delivery_id="d-1",
            captured_at="2026-09-28T00:00:00+00:00",
            answered_at="2026-09-28T00:00:00+00:00",
            answered_by_model="claude-opus",
        )
    )
    assert record.pr == 42
    assert record.delivery_id == "d-1"
    assert record.captured_at == datetime(2026, 9, 28, tzinfo=UTC)
    assert record.answered_by_model == Stated("claude-opus")


def test_a_malformed_timestamp_costs_a_field_and_never_the_record() -> None:
    """One unreadable field must not remove a document from every denominator."""
    record = record_from_frontmatter(
        kojutsu_frontmatter(tags=["agent_authored"], captured_at="last tuesday")
    )
    assert record.captured_at is None
    assert record.doc_id.endswith("answer-0")
    assert record.classification.kind is RecordKind.ANSWER


def test_a_missing_timestamp_is_absent_rather_than_the_epoch() -> None:
    """Defaulting an absent time to ``1970`` would put the record in a month it never happened."""
    record = record_from_frontmatter(kojutsu_frontmatter(tags=["agent_authored"], updated_at=None))
    assert record.updated_at is None
    assert record.month is None


def test_a_naive_timestamp_is_read_as_utc_because_every_kojutsu_writer_used_utc() -> None:
    """A naive value is a writer that forgot an offset, not an event in another zone."""
    record = record_from_frontmatter(
        kojutsu_frontmatter(tags=["agent_authored"], answered_at="2026-09-28T00:00:00")
    )
    assert record.answered_at == datetime(2026, 9, 28, tzinfo=UTC)


def test_a_z_suffixed_timestamp_is_accepted_because_the_instant_is_still_the_instant() -> None:
    """Kojutsu writes ``+00:00``, so a ``Z`` means another writer; refusing it loses a record."""
    record = record_from_frontmatter(
        kojutsu_frontmatter(tags=["agent_authored"], answered_at="2026-09-28T00:00:00Z")
    )
    assert record.answered_at == datetime(2026, 9, 28, tzinfo=UTC)


def test_a_non_positive_pull_request_number_reads_as_absent_because_zero_is_a_placeholder() -> None:
    """Pull request zero does not exist, and a boolean is not a number either."""
    assert record_from_frontmatter(kojutsu_frontmatter(tags=["agent_authored"], pr=0)).pr is None
    assert (
        record_from_frontmatter(kojutsu_frontmatter(tags=["agent_authored"], pr="not-a-number")).pr
        is None
    )


def test_the_document_id_is_the_source_of_truth_for_the_repository_and_the_change() -> None:
    """The store keys on the id, so a frontmatter that disagrees does not move the record.

    A writer that put one repository in its path and another in its frontmatter
    has made a contradiction, and silently preferring the frontmatter would make
    the corpus's self-description depend on which field a reader happened to
    trust.
    """
    record = record_from_frontmatter(
        kojutsu_frontmatter(tags=["agent_authored"], repo="other/place", pr=7),
        doc_id=document_id("acme/widget", 42, "answer-1"),
    )
    assert record.repo == "acme/widget"
    assert record.pr == 42
    assert record.frontmatter["repo"] == "other/place"


def test_a_legacy_document_with_no_repository_frontmatter_still_names_its_repository() -> None:
    """The key was added after some documents were written, and they are still countable."""
    frontmatter = kojutsu_frontmatter(tags=["agent_authored"])
    del frontmatter["repo"]
    record = record_from_frontmatter(frontmatter, doc_id=document_id("acme/widget", 42, "answer-1"))
    assert record.repo == "acme/widget"


def test_a_tag_written_as_a_bare_string_is_read_as_one_tag_not_split() -> None:
    """Guessing a separator invents tags, and every invented tag can drive a classification."""
    record = record_from_frontmatter(
        kojutsu_frontmatter(tags="review"),
    )
    assert record.classification.kind is RecordKind.REVIEW_VERDICT


def test_the_frontmatter_of_a_record_cannot_be_edited_through_it() -> None:
    """The frontmatter is the evidence a classification was read from.

    A record that could have its own evidence rewritten would let a caller change
    a kind without touching a line of this program, which is the shape of bug a
    frozen dataclass is supposed to make impossible.
    """
    record = record_from_frontmatter(kojutsu_frontmatter(tags=["review"]))
    with pytest.raises(TypeError):
        record.frontmatter["tags"] = []  # type: ignore[index]


def test_a_record_is_hashed_by_its_document_id_so_two_snapshots_agree() -> None:
    """A frozen dataclass holding a mapping claims to be hashable and is not.

    Identity here is the store's own key, so the same document read twice is the
    same record -- and defining that explicitly is better than discovering the
    problem two modules away from the dataclass.
    """
    frontmatter = kojutsu_frontmatter(tags=["agent_authored"])
    first = record_from_frontmatter(frontmatter)
    second = record_from_frontmatter(frontmatter)
    assert len({first, second}) == 1
    assert hash(first) == hash(second)


def test_a_rationale_document_is_recognised_from_its_path_even_without_tags() -> None:
    """A rationale whose tags were lost is still in the rationale namespace.

    The two signals are independent: the tag decides the kind, the path decides
    where the document lives, and a record satisfying either is a rationale.
    """
    frontmatter = kojutsu_frontmatter(
        tags=[],
        rationale_source="declared",
        declared_by="an-agent",
        declared_at="2026-09-28T00:00:00+00:00",
    )
    record = record_from_frontmatter(
        frontmatter,
        doc_id=document_id("acme/widget", 42, "rationale-v1-abc", rationale=True),
    )
    assert record.is_rationale is True
    assert record.classification.kind is RecordKind.ANSWER
    assert record.rationale_source == "declared"
    assert record.declared_by == "an-agent"


def test_a_rationale_declares_its_model_as_unknown_when_none_was_stated() -> None:
    """The literal stands in for an absence, and this reader refuses to read it as a value."""
    record = record_from_frontmatter(
        kojutsu_frontmatter(
            tags=["rationale", "rationale_declared"],
            declared_by_model="unknown",
        ),
        doc_id=document_id("acme/widget", 42, "rationale-v1-abc", rationale=True),
    )
    assert record.declared_by_model is Unstated()


def test_a_change_key_needs_both_halves_because_one_alone_joins_to_everything() -> None:
    """A record with a number and no repository cannot be joined to anything.

    Returning a partial key would let it pair with every repository holding the
    same pull request number, which is a fabricated correspondence.
    """
    record = Record(
        doc_id="a",
        collection="c",
        content="",
        frontmatter={},
        classification=classify(None),
        repo="acme/widget",
        pr=None,
    )
    assert record.pr_key is None
    assert Record(
        doc_id="a",
        collection="c",
        content="",
        frontmatter={},
        classification=classify(None),
        repo="acme/widget",
        pr=42,
    ).pr_key == ("acme/widget", 42)


def test_a_month_comes_from_the_event_the_record_is_about_and_not_from_the_write() -> None:
    """``answered_at`` is the event; ``captured_at`` and ``updated_at`` are clocks of other things.

    Grouping by the write time would answer "when did we run" and label it
    "when did this happen", which is the substitution that makes a rate over time
    mean something other than what it says.
    """
    record = record_from_frontmatter(
        kojutsu_frontmatter(
            tags=["agent_authored"],
            answered_at="2026-09-28T12:00:00+00:00",
            captured_at="2026-10-01T00:00:00+00:00",
        )
    )
    assert record.month == "2026-09"


def test_a_month_falls_back_through_the_weaker_clocks_when_the_event_time_is_absent() -> None:
    """A record with no event time is still placeable, and the fallback is stated in order."""
    record = record_from_frontmatter(
        kojutsu_frontmatter(tags=["agent_authored"], captured_at="2026-10-01T00:00:00+00:00")
    )
    assert record.month == "2026-10"


def test_a_session_id_is_carried_opaquely_because_it_is_not_a_work_episode() -> None:
    """Kojutsu sets it once per CLI invocation, from a value derived from the pull request.

    So it is a stable *per-change* grouping key and emphatically not a session: a
    change touched over three weeks is one id, and a piece of work spanning three
    pull requests is three. Any duration computed from it is a statement about the
    pull request, not about the work -- so the record carries the string and
    derives nothing from it, and the assertion is that there is no helper here
    that could.
    """
    record = record_from_frontmatter(
        kojutsu_frontmatter(tags=["agent_authored"], session_id="6f1c-uuid")
    )
    assert record.session_id == "6f1c-uuid"
    derived = [name for name in dir(Record) if "session" in name and name != "session_id"]
    assert derived == []


def test_a_document_without_a_title_still_reads_rather_than_inventing_one() -> None:
    """A missing field is ``None``; a made-up one would be a claim about the document."""
    record = StoreDocument(id="a/b/pr-1/answer-1", collection="c")
    parsed = Record.from_document(record)
    assert parsed.title is None
    assert parsed.author is None
    assert parsed.classification.certainty is Certainty.INFERRED


def test_the_record_reads_the_capture_source_raw_so_an_invalid_one_stays_visible() -> None:
    """A fourth capture source is an anomaly to report, not a value to normalise away.

    Parsing it into a closed enum here would move the anomaly out of
    :mod:`tenbin.corpus.provenance` and into a crash, and a crash is not a
    report.
    """
    record = record_from_frontmatter(
        kojutsu_frontmatter(tags=["agent_authored"], capture_source="telepathy")
    )
    assert record.capture_source == "telepathy"


def test_a_review_state_and_a_lifecycle_action_are_recovered_from_the_tags_not_guessed() -> None:
    """Both are refinements the writer projected, and neither is inferred from absence."""
    verdict = record_from_frontmatter(kojutsu_frontmatter(tags=["review", "review_state_approved"]))
    assert verdict.review_state is ReviewState.APPROVED
    assert verdict.lifecycle_action is None

    lifecycle = record_from_frontmatter(
        kojutsu_frontmatter(tags=["pr_state_change", "action_closed"])
    )
    assert lifecycle.lifecycle_action is LifecycleAction.CLOSED
    assert lifecycle.review_state is None


def test_a_record_survives_a_document_with_nothing_readable_in_it() -> None:
    """The floor case: a document with no frontmatter at all is still a record.

    It is classified as an inferred answer with no repository, which is a
    description rather than a failure -- and it is the description a report
    needs in order to notice that it is holding one.
    """
    record = Record.from_document(StoreDocument(id="orphan", collection="c"))
    assert record.classification.certainty is Certainty.INFERRED
    assert record.repo is None
    assert record.pr_key is None
    assert record.month is None
    assert record.entry_prefix is None


# -- the edges of the sentinel and the lenient readers ----------------------


def test_the_sentinel_refuses_to_be_subclassed_because_two_sentinels_would_not_be_one() -> None:
    """A subclass would share the cached instance and become a second spelling of it.

    ``Unstated() is Unstated()`` is the guarantee every comparison in this program
    relies on, and a subclass that satisfied it for a *different* class would
    make an ``isinstance`` check pass for the wrong reason.
    """
    with pytest.raises(TypeError, match="not subclassable"):

        class AlsoUnstated(Unstated):
            pass

        AlsoUnstated()


def test_the_sentinel_renders_as_a_word_because_a_report_will_print_it() -> None:
    """``<Unstated object at 0x...>`` in a rendered table is not a readable report."""
    assert repr(Unstated()) == "UNSTATED"


def test_a_stated_value_renders_as_itself_so_a_formatted_report_shows_the_name() -> None:
    """The wrapper is a type distinction, not an extra layer of punctuation on the way out."""
    assert str(Stated("claude-opus")) == "claude-opus"


def test_a_scalar_of_the_wrong_type_is_read_as_text_because_the_store_stringifies() -> None:
    """A store that round-trips through a YAML serialiser hands back bare scalars.

    Refusing them would call every stored title and author missing, and a reader
    that reports the whole corpus as untitled is reporting its own strictness.
    """
    record = Record.from_document(
        StoreDocument(
            id="a/b/pr-1/answer-1",
            collection="c",
            frontmatter={"title": 42, "author": True, "pr_url": 7},
        )
    )
    assert record.title == "42"
    assert record.author == "True"
    assert record.pr_url == "7"


def test_a_container_where_a_name_belongs_reads_as_absent_because_there_is_no_safe_reading() -> (
    None
):
    """Joining a list would invent a principal whose name contains punctuation nobody chose."""
    record = Record.from_document(
        StoreDocument(
            id="a/b/pr-1/answer-1",
            collection="c",
            frontmatter={"title": ["a", "b"], "author": {"login": "x"}},
        )
    )
    assert record.title is None
    assert record.author is None


def test_a_zero_or_negative_number_is_not_a_positive_number() -> None:
    """Zero is Kojutsu's "no pull request" placeholder, and negatives do not exist."""
    record = Record.from_document(
        StoreDocument(
            id="a/b/pr-1/answer-1",
            collection="c",
            frontmatter={"github_comment_id": 0, "rationale_revision": -1},
        )
    )
    assert record.github_comment_id is None
    assert record.rationale_revision is None


def test_a_tags_key_holding_something_unreadable_reads_as_no_tags() -> None:
    """An unreadable tag list is an absence, and the classification says ``inferred``.

    Falling back to a determination here would be the one classification this
    program must never make without evidence.
    """
    record = Record.from_document(
        StoreDocument(id="a/b/pr-1/answer-1", collection="c", frontmatter={"tags": 7})
    )
    assert record.classification.certainty is Certainty.INFERRED
    assert record.classification.evidence == ("no tags",)


def test_a_number_that_is_neither_digits_nor_an_int_reads_as_absent() -> None:
    """``"1.5"``, ``" 7a"`` and a list are not counts, and coercing any would invent one."""
    record = Record.from_document(
        StoreDocument(
            id="a/b/pr-1/answer-1",
            collection="c",
            frontmatter={
                "github_comment_id": "1.5",
                "rationale_revision": " 7a",
                "delivery_id": ["d-1"],
            },
        )
    )
    assert record.github_comment_id is None
    assert record.rationale_revision is None
    assert record.delivery_id is None


def test_a_boolean_is_not_a_number_because_true_is_one_in_python() -> None:
    """A boolean reaching a numeric field is a writer bug, and reading it as 1 invents a record.

    Hand-built here rather than through the fake, because the fake stringifies its
    extras exactly as the store does -- so a real boolean can only arrive from a
    store that did not stringify, which is the case the guard is for.
    """
    record = Record.from_document(
        StoreDocument(
            id="a/b/pr-1/answer-1",
            collection="c",
            frontmatter={"github_comment_id": True, "rationale_revision": False},
        )
    )
    assert record.github_comment_id is None
    assert record.rationale_revision is None
