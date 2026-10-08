"""The measures the file anchor and the check runs unblocked, and the one they refuse.

``MWF7Z1EN`` wrote the file anchor and ``RSF1DK1S`` wrote check runs, and two refusals
that named the first as their blocker came out of :mod:`tenbin.claims.refusals` in the
same change that built the measures they were refusing. Each measure here is small; the
interesting part of each is the population it refuses and the sentence in the claim that
says so.

Four refusals are asserted across this file, and they are the reason it exists rather
than an appendix to it. A single-custodian figure that is about the corpus is not a
figure about a team, and one that is about a team is not a figure about a person.
A rationale's staleness is movement rather than error, and it generalises to exactly one
file. A check concluding ``success`` is a fact about a build, and a change that was
merged and then reverted is one this corpus cannot tell from a change that stuck. And
nothing multiplies a merge outcome by a check conclusion into a quality score, because
two descriptive facts composed into a judgement is the causal-claim machinery in a
different costume -- which is why that last one is enforced over the *sources* of every
registered measure rather than trusted to a reviewer's eye.

The last group is about a thing this suite cannot see: a file past Kojutsu's
fifty-path bound. The writer states how many paths it left out and does not name them,
so a reader can report the bound and can never recover what is beyond it. Every
assertion here that could otherwise read as "unaffected" is checked against that.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from tenbin.claims import default_registry, slugify
from tenbin.claims.gate import ClaimGate
from tenbin.claims.mechanism import MINIMUM_GRANULARITY, Granularity, is_narrower_than
from tenbin.corpus.record import (
    Certainty,
    Record,
    RecordKind,
    check_conclusion_from_tags,
    classify,
)
from tenbin.measures import default_measures
from tenbin.measures.base import (
    DistributionFigure,
    Figure,
    FigureGroup,
    bucket_labels,
)
from tenbin.measures.check_outcome import (
    CHECK_OUTCOME_SLUG,
    KNOWN_CONCLUSIONS,
    NO_CONCLUSION_KEY,
    CheckOutcomeMeasure,
)
from tenbin.measures.custody import (
    CUSTODY_SLUG,
    MANY_CONTRIBUTORS_LABEL,
    NO_AREA_KEY,
    SINGLE_CUSTODIAN_LABEL,
    UNSTATED_ACCOUNT_KEY,
    SingleCustodianMeasure,
)
from tenbin.measures.staleness import (
    NO_FILES_KEY,
    NO_MERGE_KEY,
    NO_MOVEMENT_KEY,
    STALENESS_SLUG,
    TRUNCATED_FILE_LIST_KEY,
    RationaleStalenessMeasure,
)
from tests.fixtures import build_record, build_snapshot

_CUSTODY = SingleCustodianMeasure()
_STALENESS = RationaleStalenessMeasure()
_CHECKS = CheckOutcomeMeasure()

#: A moment and its neighbours, fixed so a bucketed gap is the same string today and
#: next week. ``datetime.now()`` in a fixture makes every assertion that touches a
#: timestamp a test that can flake for reasons nobody can see.
_MERGED = "2026-01-10T00:00:00+00:00"
_ONE_DAY_AFTER = "2026-01-11T00:00:00+00:00"
_TWO_DAYS_AFTER = "2026-01-12T00:00:00+00:00"


def _named_file(
    path: str,
    *,
    repo: str | None = "acme/widget",
    pr: int = 1,
    entry_id: str = "review-1",
    rationale: bool = False,
    **extras: str,
) -> Record:
    """A record anchored to one file, as Kojutsu writes a review capture."""
    return build_record(
        entry_id=entry_id,
        repo=repo,
        pr=pr,
        rationale=rationale,
        frontmatter={"files": [path]},
        **extras,
    )


def _distribution(figure: Figure, index: int = 0) -> DistributionFigure:
    """The distribution at a position in a group, asserting the type on the way out."""
    assert isinstance(figure, FigureGroup)
    member = figure.figures[index]
    assert isinstance(member, DistributionFigure)
    return member


# -- the file anchor ---------------------------------------------------------------


def test_a_bounded_file_list_does_not_make_the_marker_look_like_a_path() -> None:
    """Kojutsu writes its overflow into the list, and a reader must not count it.

    A change touching sixty files writes fifty paths and then the sentence ``"10 more
    files not listed"``. That sentence sits exactly where a path would sit, and a reader
    that kept it would report a file called ``"10 more files not listed"`` as a file this
    corpus holds evidence about -- a number in a report that nobody could look up and
    that a reader would have no way to tell was wrong.
    """
    record = build_record(
        frontmatter={"files": ["src/a.py", "src/b.py", "10 more files not listed"]},
    )

    assert record.files == ("src/a.py", "src/b.py")
    assert "not listed" not in " ".join(record.files)
    assert record.files_truncated is True


def test_the_marker_is_matched_as_a_shape_because_the_count_inside_it_is_written() -> None:
    """Three and ten are the same condition, and only one of them was the example.

    A reader that recognised the marker as one string would start counting ``"3 more
    files not listed"`` as a path the moment Kojutsu wrote ``"17"``, and nothing
    would fail -- the file would simply appear. So the test uses a count that is not
    the one the writer's own docstring shows.
    """
    record = build_record(frontmatter={"files": ["src/a.py", "17 more files not listed"]})

    assert record.files == ("src/a.py",)
    assert record.files_truncated is True


def test_a_path_that_merely_mentions_the_marker_is_kept_because_the_pattern_is_anchored() -> None:
    """A real path containing the phrase is a real path, and discarding it loses code.

    The pattern is anchored at both ends precisely so that this case survives: a
    repository can contain a document about a truncation marker, and a reader that
    dropped it would make a file this corpus has evidence about silently absent --
    which is the same failure as counting the marker, reached from the other direction.
    """
    record = build_record(frontmatter={"files": ["docs/3 more files not listed.md"]})

    assert record.files == ("docs/3 more files not listed.md",)
    assert record.files_truncated is False


def test_an_unbounded_file_list_reports_no_truncation_because_zero_is_not_a_bound() -> None:
    """The flag answers "did the writer leave anything out", and nothing else.

    A record whose change touched four files is not a truncated record, and a record
    written before the key existed is not one either. Both carry the same empty file
    list, and a flag that reported either as bounded would put a bound in a figure for a
    document that never had a long list to bound.
    """
    assert _named_file("src/a.py").files_truncated is False
    assert build_record().files == ()
    assert build_record().files_truncated is False


def test_head_sha_is_anchored_and_never_a_verification_because_no_field_claims_one() -> None:
    """The commit a capture was taken at, and nothing that could be mistaken for more.

    Kojutsu named the key ``head_sha`` and a field called ``verified_at_commit`` would
    be a claim the store cannot back: it says which commit this record is about, not that
    the record is still true there, that the change was reviewed, or that the two
    correspond to the same code. A reader who took it for a verification would be
    inferring agreement between two documents from a shared string, so the model carries
    no field whose name would license the inference.
    """
    record = build_record(head_sha="9f2c1ab")

    assert record.head_sha == "9f2c1ab"
    assert not [name for name in Record.__dataclass_fields__ if "verif" in name], (
        "a field named for a verification is a claim the store cannot back, whatever its value"
    )


def test_a_check_run_is_a_kind_of_its_own_because_a_build_conclusion_never_lands_in_the_review_namespace() -> (
    None
):
    """Its own kind, in the writer's own tag, and never read as somebody's judgement.

    Kojutsu projects ``check`` onto the kind and keeps it there, and the reason is
    stated there: a check conclusion rendered beside a review verdict is an inference
    nobody made. So a record carrying ``check`` reads as a check run whatever else it
    carries, and a check record never reads as a review.
    """
    from_tags = classify(["check", "check_state_success"])
    from_key = classify(["check", "check_state_success"], record_kind="check_run")

    assert from_tags.kind is RecordKind.CHECK_RUN
    assert from_tags.certainty is Certainty.DETERMINED
    assert from_tags.evidence == ("check", "check_state_success")
    assert from_key.evidence == ("record_kind", "check_run")
    assert classify(["check"]).kind is RecordKind.CHECK_RUN
    assert classify(["review", "review_state_approved"]).kind is RecordKind.REVIEW_VERDICT


def test_a_check_tag_beats_a_review_tag_because_it_names_one_kind_where_review_names_two() -> None:
    """The precedence, stated as the rule it follows rather than as a position.

    The reader's rule is that the more specific tag wins, and ``check`` is a whole tag
    projection naming one kind while ``review`` is shared with two kinds and needs the
    ``inline_comment`` refinement before it can be read at all. Kojutsu writes a check
    run without ``review``, so this settles a contradiction the corpus should not contain
    rather than a case it produces -- which is exactly why the evidence has to name the
    tag it read.
    """
    reading = classify(["review", "check", "check_state_failure", "inline_comment"])

    assert reading.kind is RecordKind.CHECK_RUN
    assert reading.evidence == ("check", "check_state_failure")


def test_a_conclusion_is_recovered_from_its_tag_because_the_forge_owns_the_vocabulary() -> None:
    """Prefix match, and a bare prefix is no conclusion rather than an empty one.

    The forge's nine conclusions are Kojutsu's to close, not this reader's, and a
    local copy of the list would go stale the first time a forge added one. So whatever
    follows ``check_state_`` is carried through as itself. A tag that is the prefix and
    nothing else yields ``None``, because an empty conclusion would bucket as a real one.
    """
    assert check_conclusion_from_tags(["check", "check_state_startup_failure"]) == "startup_failure"
    assert check_conclusion_from_tags(["check", "check_state_"]) is None
    assert check_conclusion_from_tags(["check"]) is None
    assert set(KNOWN_CONCLUSIONS) >= {"success", "failure", "startup_failure"}


# -- single-custodian knowledge per area -------------------------------------------


def test_the_single_custodian_figure_is_the_answer_to_the_question_the_measure_was_written_for() -> (
    None
):
    """One account, one file, one bucket -- and the bucket is present even at zero.

    The corpus holds two records about one file, both from ``dana``, so the file has one
    account behind it and belongs in the rung the measure is named for. The other rungs
    are materialised at zero rather than left out, because ``"exactly 1 account: 0"`` is
    an answer to a reader's question and an absent bucket reads as "not computed".
    """
    whole = _distribution(
        _CUSTODY.compute(build_snapshot([_named_file("src/queue.py", comment_author="dana")]))
    )

    assert whole.values[SINGLE_CUSTODIAN_LABEL] == 1
    assert set(whole.values) == {
        SINGLE_CUSTODIAN_LABEL,
        *(f"{n} accounts" for n in range(2, 5)),
        MANY_CONTRIBUTORS_LABEL,
    }
    assert whole.values["2 accounts"] == 0
    assert whole.claim.denominator.size == 1
    assert whole.claim.slug == CUSTODY_SLUG


def test_two_accounts_on_one_file_move_it_out_of_the_single_custodian_bucket() -> None:
    """The three account fields are one set, so two of them is two accounts.

    ``comment_author``, ``declared_by`` and ``change_author_account`` are the only three
    fields a name is drawn from, and they describe three different relationships to a
    change. Reading them as alternatives rather than as contributors would file a file
    two people wrote about as a file one person knew.
    """
    records = (
        _named_file("src/queue.py", comment_author="dana"),
        _named_file(
            "src/queue.py", pr=2, entry_id="rationale-v1-1", rationale=True, declared_by="raj"
        ),
        _named_file("src/queue.py", pr=3, entry_id="pr-event-3", change_author_account="kai"),
    )

    whole = _distribution(_CUSTODY.compute(build_snapshot(records)))

    assert whole.values[SINGLE_CUSTODIAN_LABEL] == 0
    assert whole.values["3 accounts"] == 1


def test_a_file_nobody_named_is_absent_rather_than_shown_to_have_no_custodian() -> None:
    """A file no record mentions is not in the distribution, and the claim says why.

    A file nobody commented on and a file everybody passed over are the same observation
    from inside this seam, and a bucket of zero for one of them would be reporting that
    equivalence as a finding. So the file is absent -- the totals are over the files this
    read could resolve -- and the sentence a reader is most likely to need is in the
    claim rather than in a bucket.
    """
    corpus = [_named_file("src/queue.py", comment_author="dana")]

    whole = _distribution(_CUSTODY.compute(build_snapshot(corpus)))

    assert sum(whole.values.values()) == 1, "only the file a record named may be counted"
    assert whole.claim.denominator.size == 1
    assert "no captured record names is absent" in whole.claim.does_not_mean
    assert "five people looking at it and one person writing" in whole.claim.does_not_mean


def test_an_unstated_account_goes_to_excluded_and_never_into_a_name() -> None:
    """The literal ``"unknown"`` is an absence, and an absence is not a contributor.

    Kojutsu fills several fields with the literal where nothing was stated, and a
    truthiness check would pass it: the string is non-empty. So a file whose only record
    names nobody would be counted as a file one account -- a contributor called
    ``"unknown"`` -- which is the same defect as a model bucket of ``"unknown"`` in a
    second costume. It is counted under a key that names the condition instead.
    """
    record = _named_file("src/queue.py", comment_author="unknown", change_author_account="unknown")

    whole = _distribution(_CUSTODY.compute(build_snapshot([record])))

    assert sum(whole.values.values()) == 0, (
        "a file no account was named for is not a custody figure"
    )
    assert whole.excluded == {UNSTATED_ACCOUNT_KEY: 1}


def test_a_file_named_without_a_repository_is_set_aside_because_the_path_is_relative() -> None:
    """Two repositories' ``src/queue.py`` are two files, and only the repository says which.

    Kojutsu writes repository-relative paths, so a bare path is ambiguous across areas
    and merging them would report a shared custodian for two files that each have one.
    A record filed under the ``unknown`` placeholder is therefore counted separately
    rather than resolved against a repository the read happened not to hold.
    """
    records = (
        _named_file("src/queue.py", repo="unknown", comment_author="dana"),
        _named_file("src/queue.py", repo="acme/other", comment_author="dana"),
    )

    whole = _distribution(_CUSTODY.compute(build_snapshot(records)))

    assert whole.excluded == {NO_AREA_KEY: 1}
    assert whole.values[SINGLE_CUSTODIAN_LABEL] == 1, (
        "the two repositories' files must not be merged into one with a shared custodian"
    )
    assert _distribution(_CUSTODY.compute(build_snapshot(records)), 1).values == {"acme/other": 1}


def test_no_figure_here_names_a_person_as_a_sole_custodian_because_that_is_a_different_product() -> (
    None
):
    """The output is a count over an area, and the floor says so structurally.

    Per-principal output is the product decision 002 refuses -- different consent,
    retention and access requirements, not this report with a filter applied. So the
    measure is checked three ways: the claim sits at or above the granularity floor, the
    gate permits it, and no login appears anywhere in the rendered payload. A figure that
    satisfied only the first two would still be able to print a name beside a count.
    """
    records = (_named_file("src/queue.py", comment_author="dana"),)
    figure = _CUSTODY.compute(build_snapshot(records))
    whole = _distribution(figure)

    assert not is_narrower_than(whole.claim.granularity, MINIMUM_GRANULARITY)
    assert whole.claim.granularity is Granularity.repository
    assert ClaimGate().check(whole.claim) is None
    assert "dana" not in figure.value_text()
    assert "dana" not in whole.claim.render_text()
    assert "never names a person" in whole.claim.does_not_mean


def test_the_falsifier_is_a_file_several_accounts_also_wrote_about_because_that_is_what_would_expose_it() -> (
    None
):
    """The observation that would show the measure tracking capture rather than knowledge.

    A file with exactly one captured record whose change has several distinct accounts
    commenting on it is a direct observation of several people having written about one
    file. If that file still read as a single-custodian file, the figure would be about
    who captured rather than who knows -- so the observation belongs in the falsifier
    rather than only in this test file, where a reader of a rendered claim will not
    come across it.
    """
    whole = _distribution(_CUSTODY.compute(build_snapshot([_named_file("src/queue.py")])))

    assert "several distinct accounts" in whole.claim.falsifier
    assert "tracking who captured rather than who knows" in whole.claim.falsifier


# -- rationale staleness against code movement ---------------------------------------


def _rationale(
    *,
    repo: str = "acme/widget",
    pr: int = 1,
    files: list[str] | None = None,
    declared_at: str = "2026-01-07T00:00:00+00:00",
    tags: list[str] | None = None,
) -> Record:
    """A stated reason in the rationale namespace, with the files it is about."""
    return build_record(
        entry_id="rationale-v1-1",
        repo=repo,
        pr=pr,
        rationale=True,
        record_kind=None,
        tags=tags if tags is not None else ["rationale", "rationale_declared"],
        declared_at=declared_at,
        captured_at=declared_at,
        frontmatter={"files": list(files or ["src/queue.py"])},
    )


def _merge(*, repo: str = "acme/widget", pr: int = 1, merged_at: str = _MERGED) -> Record:
    """A lifecycle record carrying the change's own merge clock."""
    return build_record(
        entry_id="pr-event-1",
        repo=repo,
        pr=pr,
        record_kind=RecordKind.PR_LIFECYCLE,
        tags=["pr_state_change", "action_closed"],
        pr_merged_at=merged_at,
        pr_outcome="merged",
    )


def test_the_gap_runs_from_the_merge_because_a_reason_is_stated_while_the_change_is_open() -> None:
    """From the merge, not from the declaration, and the two are three days apart here.

    A rationale is usually written while the change is still under review. Measuring from
    ``declared_at`` would fold the review period into a measure of code movement, so a
    change that merged yesterday and had one capture a day before that would read as a
    week of staleness. The merge is the first instant at which the code exists in the form
    the reason is about.
    """
    corpus = (
        _rationale(),
        _merge(),
        _named_file("src/queue.py", pr=9, entry_id="review-9", captured_at=_TWO_DAYS_AFTER),
    )

    figure = _STALENESS.compute(build_snapshot(corpus))
    assert isinstance(figure, DistributionFigure)
    assert figure.values["1d to 3d"] == 1, (
        "the gap is measured from the merge, not from the declaration three days earlier"
    )
    assert figure.excluded == {}
    assert figure.claim.slug == STALENESS_SLUG
    assert figure.claim.denominator.size == 1
    assert "rationale-file pairs" in figure.claim.denominator.description


def test_a_file_nothing_was_captured_against_after_the_merge_is_not_the_freshest_bucket() -> None:
    """No interval is not a zero interval, and the difference is the whole reading.

    Writing an absent gap as zero would file a rationale nobody revisited next to one
    whose file was confirmed the moment it merged -- the most confident and most wrong
    number available, and it is the *opposite* meaning rather than an adjacent one. So
    the pair is counted under a key naming the condition and does not appear in any rung.
    """
    corpus = (_rationale(), _merge())

    figure = _STALENESS.compute(build_snapshot(corpus))
    assert isinstance(figure, DistributionFigure)
    assert figure.excluded == {NO_MOVEMENT_KEY: 1}
    assert figure.values[bucket_labels()[0]] == 0, (
        "the freshest rung is present and empty, which is the same as saying the gap was "
        "not measured -- a pair filed there would read as a capture the instant it merged"
    )
    assert sum(figure.values.values()) == 0


def test_a_bounded_file_list_sets_the_rationale_aside_because_the_unlisted_files_can_only_add_staleness() -> (
    None
):
    """Truncation biases this measure in one direction, so it is an exclusion here.

    Kojutsu caps the list at fifty and writes ``"N more files not listed"``. The paths
    it left out are real paths on a change that moved them, so a partial list can only
    *understate* how stale a rationale is -- the bias runs one way and a reader would have
    no way to see it. The custody measure can keep its truncated records for the opposite
    reason, and the difference is the point: there a bound shrinks a population, here it
    would flatter one.
    """
    bounded = build_record(
        entry_id="rationale-v1-1",
        pr=1,
        rationale=True,
        record_kind=None,
        tags=["rationale", "rationale_declared"],
        frontmatter={"files": ["src/queue.py", "12 more files not listed"]},
    )
    corpus = (
        bounded,
        _merge(),
        _named_file("src/queue.py", pr=9, entry_id="review-9", captured_at=_TWO_DAYS_AFTER),
    )

    figure = _STALENESS.compute(build_snapshot(corpus))
    assert isinstance(figure, DistributionFigure)
    assert figure.excluded == {TRUNCATED_FILE_LIST_KEY: 1}
    assert sum(figure.values.values()) == 0


def test_a_rationale_about_one_file_says_nothing_about_the_rest_of_its_change() -> None:
    """The sibling file moved and does not appear, because nobody named it.

    A change can rewrite the module a file depends on while leaving that file alone, and
    a rationale that said nothing about the module is not a statement about it. The
    distribution therefore holds the pair the rationale named and not the change's whole
    file set, and the claim says so in the words a reader arrives with.
    """
    sibling = build_record(
        entry_id="review-8",
        pr=8,
        comment_author="kai",
        captured_at=_TWO_DAYS_AFTER,
        frontmatter={"files": ["src/worker.py"]},
    )
    corpus = (
        _rationale(),
        _merge(),
        _named_file("src/queue.py", pr=9, entry_id="review-9", captured_at=_ONE_DAY_AFTER),
        sibling,
    )

    figure = _STALENESS.compute(build_snapshot(corpus))
    assert isinstance(figure, DistributionFigure)
    assert figure.claim.denominator.size == 1, "the sibling file is not a pair this read can date"
    assert sum(figure.values.values()) == 1
    assert "the other files the same change touched" in figure.claim.does_not_mean
    assert "not in this distribution at all" in figure.claim.does_not_mean


def test_staleness_is_movement_and_not_error_because_nobody_needed_to_revisit_the_reason() -> None:
    """The clause exists because the misreading is a moral one rather than a technical one.

    A reason not revisited in a year may be one nobody needed to revisit, and a reason
    that was revisited may have been confirmed rather than corrected. The corpus holds
    neither the correction nor the confirmation, so the sentence is in the claim where a
    reader sees it before the number.
    """
    figure = _STALENESS.compute(build_snapshot((_rationale(), _merge())))
    assert isinstance(figure, DistributionFigure)

    assert "Staleness is movement, not error" in figure.claim.does_not_mean
    assert "may be one nobody needed to revisit" in figure.claim.does_not_mean
    assert figure.claim.unblocked_by is not None
    assert "MWF7Z1EN" in figure.claim.unblocked_by


def test_a_rationale_with_no_clock_and_one_with_no_file_are_two_different_gaps() -> None:
    """Both set aside, counted separately, because a reader fixing one needs the other intact.

    A rationale whose change never merged has no end to measure from and a rationale that
    names no file has no code to measure against. Collapsing them into one key would tell
    an operator to go and look for a merge time when the real gap is a missing file list.
    """
    no_files = build_record(
        entry_id="rationale-v1-1",
        pr=2,
        rationale=True,
        record_kind=None,
        tags=["rationale", "rationale_declared"],
        frontmatter={"files": []},
    )
    no_merge = _rationale(pr=3)

    figure = _STALENESS.compute(build_snapshot((no_files, no_merge, _merge())))
    assert isinstance(figure, DistributionFigure)
    assert figure.excluded == {NO_FILES_KEY: 1, NO_MERGE_KEY: 1}


# -- what a check concluded ----------------------------------------------------------


def _check(conclusion: str | None, *, pr: int = 1, entry_id: str = "check-abc") -> Record:
    """A check run, in the kind and tag set Kojutsu projects for it."""
    tags = ["check"] + ([f"check_state_{conclusion}"] if conclusion else [])
    return build_record(
        entry_id=entry_id,
        pr=pr,
        record_kind=RecordKind.CHECK_RUN,
        tags=tags,
        head_sha="9f2c1ab",
        frontmatter={"check_conclusion": conclusion} if conclusion else {},
    )


def test_the_conclusions_are_distributed_over_check_runs_and_not_over_commits() -> None:
    """Three checks on one commit are three reports, and collapsing them invents a verdict.

    There is no rule in the corpus for deciding which of a commit's checks is the
    commit's outcome, and picking one -- or multiplying them -- is a composite. So the
    unit is the check run, the population says so, and a commit that both linted and
    type-checked contributes two buckets rather than one ambiguous number.
    """
    corpus = (
        _check("success", pr=1, entry_id="check-a"),
        _check("success", pr=1, entry_id="check-b"),
        _check("failure", pr=1, entry_id="check-c"),
        build_record(entry_id="answer-0", pr=2, comment_author="dana"),
    )

    figure = _CHECKS.compute(build_snapshot(corpus))
    assert isinstance(figure, DistributionFigure)
    assert dict(figure.values) == {"success": 2, "failure": 1}
    assert figure.claim.slug == CHECK_OUTCOME_SLUG
    assert "check runs" in figure.claim.denominator.description
    assert "not over commits" in figure.claim.does_not_mean


def test_a_passing_check_is_a_fact_about_the_build_and_not_about_the_change() -> None:
    """``success`` is not ``correct``, and the difference is the whole claim.

    A green pipeline on a change that broke production passed its pipeline: a pipeline
    that never exercises the failure has never reported one. So the figure describes what
    builds said, and a reader composing it into a judgement about a change is making an
    inference the store never licensed.
    """
    figure = _CHECKS.compute(build_snapshot((_check("success"),)))

    assert isinstance(figure, DistributionFigure)
    assert "broke production passed its pipeline" in figure.claim.does_not_mean
    assert "the forge's own wording" in figure.claim.does_not_mean


def test_a_merged_then_reverted_change_is_indistinguishable_from_one_that_stuck() -> None:
    """The limit a reader cannot see from the figure, so the claim names it by name.

    ``RSF1DK1S`` covers check runs; reversion is a separate event the store does not
    record. So a change that merged and was then reverted contributes exactly what one
    that stuck contributes, and nothing in this figure counts survivors or walks a
    conclusion back. Stated explicitly because the figure would otherwise read as a
    distribution of changes that held.
    """
    figure = _CHECKS.compute(build_snapshot((_check("success"),)))

    assert isinstance(figure, DistributionFigure)
    assert "merged and was later reverted" in figure.claim.does_not_mean
    assert "indistinguishable in this figure from one that stuck" in figure.claim.does_not_mean
    assert "nothing below counts survivors" in figure.claim.does_not_mean


def test_a_check_record_with_no_conclusion_is_excluded_because_the_build_reported_nothing() -> None:
    """A check run with no conclusion is a document that says the check did not report.

    Kojutsu refuses to store an unconcluded run, so a record in this state is
    hand-edited or written by something else. Counting it under a key that names the
    condition keeps the corpus's shape visible without filing an empty conclusion as a
    bucket a reader would then have to interpret.
    """
    figure = _CHECKS.compute(build_snapshot((_check(None), _check("success"))))

    assert isinstance(figure, DistributionFigure)
    assert figure.excluded == {NO_CONCLUSION_KEY: 1}
    assert dict(figure.values) == {"success": 1}


def test_an_unrecognised_conclusion_is_kept_as_itself_because_the_forge_owns_the_vocabulary() -> (
    None
):
    """A tenth conclusion is a bucket, not a crash and not a silent loss.

    The writer stores whatever the forge reported, lowercased. A reader that closed the
    vocabulary would either raise on a value Kojutsu had no reason to reject, or fold
    it into a neighbour -- and the second is worse, because a figure would quietly shrink
    while every number in it still looked like a count of builds.
    """
    figure = _CHECKS.compute(build_snapshot((_check("flaky"), _check("success"))))

    assert isinstance(figure, DistributionFigure)
    assert dict(figure.values) == {"flaky": 1, "success": 1}
    assert figure.excluded == {}


# -- the prohibition this ticket is not allowed to break ---------------------------------


def _module_of(measure: object) -> Path:
    """The file a measure is written in, so the prohibition can be read over its source.

    Source rather than behaviour, and deliberately so: a runtime check would have to
    decide whether two numbers were multiplied, and a measure that composes them inside a
    helper would pass a behavioural check while the figure it renders is exactly the
    score the prohibition exists to refuse.
    """
    source = inspect.getsourcefile(type(measure))
    assert source is not None
    return Path(source)


def _fields_read(path: Path) -> set[str]:
    """The corpus fields a source file *reads*, as opposed to the ones it discusses.

    Parsed rather than grepped, and that is the whole design of this check. A substring
    search would fail on the module that argues the prohibition -- ``check_outcome.py``
    names ``pr_outcome`` precisely to say it does not read it -- and a rule that punishes
    writing down a prohibition is a rule that gets switched off, because the cheapest fix
    is to stop saying the thing. So comments and docstrings are dropped and what remains
    is the code: an attribute read, or a string literal where a field name is looked up.
    Either of those is a read; prose about one is not.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }
    read: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            read.add(node.attr)
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
        ):
            read.add(node.value)
    return read


def test_no_measure_reads_a_merge_outcome_and_a_check_conclusion_together() -> None:
    """Two descriptive facts multiplied into one number is a judgement nobody licensed.

    A merge outcome says what the forge reported happening to a change; a check conclusion
    says what a build reported about a commit. Multiplying them, scoring them, or even
    tabulating them against each other produces a "quality" number that no assignment
    mechanism and no counterfactual supports -- the causal-claim machinery in a different
    costume, and the easiest thing in this package to add by accident once both fields
    exist. The check walks the code of every registered measure and names the offender.
    """
    offenders = [
        f"{type(measure).__name__} in {_module_of(measure).name}"
        for measure in default_measures()
        if {"pr_outcome", "check_conclusion"} <= _fields_read(_module_of(measure))
    ]

    assert not offenders, (
        "these measures read both a merge outcome and a check conclusion, which is how a "
        "composite quality score gets built: "
        + ", ".join(offenders)
        + ". Keep the two measures separate, each with its own claim. Nothing joins them."
    )


def test_no_module_in_the_measures_package_can_compose_the_two_fields() -> None:
    """The per-measure check only reads where a measure is written, and that is a real gap.

    A composite could sit in a helper module the check-outcome measure calls, or a future
    measure could import a builder from elsewhere in the package, and the per-measure test
    would pass while the figure it renders is the score the prohibition exists to refuse.
    So the whole package is scanned, which catches a shared builder the per-measure walk
    cannot reach.
    """
    package = Path(inspect.getsourcefile(default_measures) or "").parent
    offenders = [
        path.name
        for path in sorted(package.glob("*.py"))
        if {"pr_outcome", "check_conclusion"} <= _fields_read(path)
    ]

    assert not offenders, (
        "these modules read both a merge outcome and a check conclusion, so a composite is "
        "one import away: "
        + ", ".join(offenders)
        + ". The two figures stay separate, each with its own claim."
    )


# -- two refusals retired -------------------------------------------------------------


def test_the_file_anchor_retired_two_refusals_and_both_replacement_measures_are_permitted() -> None:
    """Retirement is not a deletion, and the replacement has to render.

    A refusal removed before its measure is written tells a reader the impossible is
    impossible while leaving the thing unwritten, and the completeness test in
    ``tests/test_refusal_registry`` cannot see that from either side. So both directions
    are checked here: the slugs are out of the registry, and the measures that replaced
    them exist, are registered, and pass the gate.
    """
    registry = default_registry()
    for measure in (
        "Single-custodian knowledge per area",
        "Rationale staleness against code movement",
    ):
        assert slugify(measure) not in registry, (
            f"{measure!r} is measured now, so its refusal outlived its blocker and must go"
        )

    for measure in (CUSTODY_SLUG, STALENESS_SLUG):
        assert measure in {registered.slug for registered in default_measures()}, (
            f"{measure!r} is built but not registered, so no report would ever run it"
        )
    for slug, built in ((CUSTODY_SLUG, _CUSTODY), (STALENESS_SLUG, _STALENESS)):
        claim = built.claim(build_snapshot((_rationale(), _merge())))
        assert slug not in registry
        assert ClaimGate().check(claim) is None
