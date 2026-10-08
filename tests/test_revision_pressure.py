"""That a revision chain is a chain, and that a revision is not a retraction.

The denominator of the whole measure is the thing most likely to be got wrong, so
it is the first test: a rationale revised four times is one stated reason that
changed four times, and a denominator of records would report it as four separate
acts of revision by four separate authors. Every number this module asserts is
therefore about the number of *reasons* that changed.

The second thing likely to be got wrong is the moral reading. This is the one
figure here whose most natural misreading is not a technical error, and the corpus
inventory says directly that whether the first reason was better is not supportable
from these records. So the claim's negative space is asserted on the words, and the
sentence is checked for the specific failure it exists to prevent: a claim that a
chain records somebody changing their mind.

The third is the dangling target. A record whose ``rationale_revises`` names an
entry this corpus does not hold is neither a root nor a deletion, and both of the
obvious alternatives are wrong in ways a reader would not detect from the figure.
"""

from __future__ import annotations

from tenbin.claims.gate import ClaimGate
from tenbin.corpus.record import Record, RecordKind
from tenbin.measures.base import NEGATIVE_INTERVAL, DistributionFigure, FigureGroup
from tenbin.measures.revision import (
    DANGLING_REVISION_KEY,
    NO_DECLARATION_TIME_KEY,
    UNLINKABLE_KEY,
    RevisionIntervalMeasure,
    RevisionPressureMeasure,
    chain_lengths,
    walk_chains,
)
from tests.fixtures import build_record, build_snapshot

_PRESSURE = RevisionPressureMeasure()
_INTERVAL = RevisionIntervalMeasure()

_T0 = "2026-01-01T00:00:00+00:00"


def _rationale(
    entry_id: str,
    *,
    pr: int = 1,
    revises: str | None = None,
    source: str | None = "declared",
    declared_at: str | None = _T0,
) -> Record:
    """A rationale in the rationale namespace, with an optional supersession."""
    return build_record(
        entry_id=entry_id,
        pr=pr,
        rationale=True,
        record_kind=None,
        tags=["rationale", f"rationale_{source or 'unknown'}"],
        rationale_source=source,
        rationale_revises=revises,
        declared_at=declared_at,
        capture_source="asserted",
    )


def _pressures(records) -> FigureGroup:
    """The chain-length figures, asserting the group shape the measure is contracted to return."""
    figure = _PRESSURE.compute(build_snapshot(records))
    assert isinstance(figure, FigureGroup)
    return figure


def _lengths(records) -> dict[str, int]:
    """The whole-corpus chain-length distribution, as a plain dict for reading."""
    chains, _ = walk_chains(records)
    return dict(chain_lengths(chains))


# -- the denominator -------------------------------------------------------------


def test_the_denominator_is_chains_and_not_records_because_one_reason_revised_four_times_is_one_reason() -> (
    None
):
    """The single most important detail in the measure, asserted on the denominator itself.

    Five records -- one root and four revisions -- are one stated reason that
    changed four times. A denominator of records would make it a rate of four
    revisions over five records and the claim would be reporting a corpus of five
    stated reasons when it holds one. So the claim's denominator is the chain count,
    and the bucket lengths are lengths rather than counts of records.
    """
    records = [
        _rationale("rationale-v1-0"),
        _rationale("rationale-v2-0", revises="rationale-v1-0"),
        _rationale("rationale-v3-0", revises="rationale-v2-0"),
        _rationale("rationale-v4-0", revises="rationale-v3-0"),
        _rationale("rationale-v5-0", revises="rationale-v4-0"),
    ]
    chains, _ = walk_chains(records)
    assert len(records) == 5
    assert len(chains) == 1
    assert chains[0].length == 5

    figure = _PRESSURE.claim(build_snapshot(records))
    assert figure.denominator.size == 1
    assert "counted as chains" in figure.denominator.description
    assert _lengths(records) == {"chain of 5 (4 revision(s))": 1}


def test_a_chain_of_one_is_a_stated_reason_nobody_revised_and_is_the_largest_bucket() -> None:
    """A length of one is named in words because a reader skims the largest bucket.

    Calling it ``1`` is how "never revised" gets read as "one thing happened". The
    bucket label says what the length means, and the bucket is the one a reader is
    most likely to pass over, so it is the one that most needs words.
    """
    records = [_rationale("rationale-v1-0"), _rationale("rationale-v1-1", pr=2)]
    assert _lengths(records) == {"chain of 1 (never revised)": 2}


def test_a_revision_is_not_a_retraction_and_the_claim_says_it_in_a_sentence_that_cannot_be_skimmed() -> (
    None
):
    """The negative space, asserted on the words rather than on its presence.

    This is the one figure whose most natural misreading is moral rather than
    technical: a chain of four reads as somebody changing their mind four times,
    which is a claim about the person and about the quality of the earlier reason,
    and the corpus does not support either. So the sentence names the specific wrong
    reading rather than only stating the right one, and the test checks for the
    wrong reading's own vocabulary.
    """
    claim = _PRESSURE.claim(build_snapshot([_rationale("rationale-v1-0")]))
    rendered = claim.does_not_mean
    assert "A revision is not a retraction" in rendered
    assert "not supportable from these records at all" in rendered
    assert "moral reading" in rendered
    for word in ("better", "worse", "better than"):
        assert f"the first reason was {word}" not in rendered


def test_the_falsifier_is_a_revision_that_supersedes_nothing_because_that_is_what_would_break_the_walk() -> (
    None
):
    """A falsifier that could be looked for, in a corpus that holds the fact.

    A rationale that supersedes nothing is a new stated reason rather than a
    revision, and finding one would mean the chain walk is reading a supersession
    the record does not claim. Both records are in the corpus, so the reader has
    something to check rather than a sentiment to accept.
    """
    claim = _PRESSURE.claim(build_snapshot([_rationale("rationale-v1-0")]))
    assert "supersedes nothing" in claim.falsifier
    assert ClaimGate().check(claim) is None


# -- the walk ---------------------------------------------------------------------


def test_a_dangling_supersession_is_excluded_and_not_treated_as_a_root_because_a_root_is_a_reason_nothing_superseded() -> (
    None
):
    """The two obvious alternatives are both wrong, so the record is named and counted.

    Counting it as a root would say a reason nobody superseded, which inverts what
    the record states. Dropping it would make a chain-length distribution over the
    followable chains with no account of the others -- the silent-filtering failure
    this package exists to report. And calling it a chain of one would report an
    interval it cannot measure. So it is excluded, under a key that says the chain
    cannot be followed.
    """
    records = [
        _rationale("rationale-v1-0"),
        _rationale("rationale-v2-0", revises="rationale-v1-0", pr=2),
        _rationale("rationale-v9-0", revises="rationale-v8-0", pr=3),
    ]
    chains, excluded = walk_chains(records)
    assert [chain.length for chain in chains] == [2]
    assert dict(excluded) == {DANGLING_REVISION_KEY: 1}
    assert "not in this corpus" in DANGLING_REVISION_KEY


def test_a_record_with_no_entry_id_is_excluded_under_its_own_condition_because_the_two_absences_are_different() -> (
    None
):
    """A record that cannot be a node and a record whose parent is missing are different gaps.

    Both are absent from the chains and both are counted, but a reader fixing one of
    them needs to know which they have: one needs an entry id written and the other
    needs the earlier revision to be in the corpus. One key for both would send
    somebody to look in the wrong place.
    """
    unlinkable = build_record(
        entry_id="rationale-v1-x",
        pr=1,
        rationale=True,
        record_kind=None,
        tags=["rationale", "rationale_declared"],
        rationale_source="declared",
    )
    object.__setattr__(unlinkable, "entry_id", "")
    chains, excluded = walk_chains([unlinkable])
    assert chains == ()
    assert dict(excluded) == {UNLINKABLE_KEY: 1}
    assert "cannot be a node" in UNLINKABLE_KEY


def test_a_chain_split_by_the_source_of_its_root_because_a_reconstruction_being_revised_is_a_different_phenomenon() -> (
    None
):
    """The split is on the root, and the parts sum to the whole.

    Two reasons: the source describes how the chain came to exist and the root is
    where it did, so a revision of a declared reason is still a declared line of
    reasoning; and a per-source figure that silently omitted a source would be a
    decomposition whose parts do not sum to the total, which is the one arithmetic
    error a reader notices and cannot explain. So the sources are read from the data
    -- a source outside the three Kojutsu writes gets a bucket of its own rather
    than being dropped.
    """
    records = [
        _rationale("rationale-v1-0", source="declared"),
        _rationale("rationale-v2-0", pr=2, revises="rationale-v1-0", source="declared"),
        _rationale("rationale-v1-1", pr=3, source="reconstructed"),
        _rationale("rationale-v1-2", pr=4, source="inferred_by_a_guess"),
    ]
    figure = _pressures(records)
    per_source = {
        member.title.split("whose root is ")[1].removesuffix(
            ", by how many times they were revised"
        ): member
        for member in figure.figures[1:]
        if isinstance(member, DistributionFigure)
    }
    assert set(per_source) == {"declared", "reconstructed", "inferred_by_a_guess"}
    assert dict(per_source["declared"].values) == {
        "chain of 2 (1 revision(s))": 1,
    }
    assert dict(per_source["reconstructed"].values) == {"chain of 1 (never revised)": 1}
    # Every chain appears in exactly one source bucket, and the total is unchanged.
    assert all(isinstance(member, DistributionFigure) for member in figure.figures)
    total = sum(
        member.counted for member in figure.figures[1:] if isinstance(member, DistributionFigure)
    )
    assert total == len(walk_chains(records)[0])


def test_every_chain_figure_carries_the_same_claim_because_the_populations_are_parts_of_one_whole() -> (
    None
):
    """A decomposition, not four samples.

    The chain-length figure and the three per-source figures are over the same
    chains -- the per-source ones are a partition of it -- so they share the claim
    and the denominator. The *interval* figure is a different population over
    different objects, which is why it is a separate measure with its own claim
    rather than a fifth member of this group.
    """
    records = [
        _rationale("rationale-v1-0", source="declared"),
        _rationale("rationale-v1-1", pr=2, source="reconstructed"),
    ]
    figure = _pressures(records)
    for member in figure.figures:
        assert member.claim == figure.claim
        assert member.claim.denominator.size == 2


# -- intervals ----------------------------------------------------------------------


def test_the_elapsed_time_is_a_distribution_and_never_a_mean_because_a_mean_of_a_skewed_distribution_is_a_point_nobody_occupies() -> (
    None
):
    """Buckets, and no summary statistic anywhere in the rendered output.

    The intervals are skewed by construction: most are short, a few are days. A
    mean of that is a point no observation occupies -- a number nobody can act on
    and everybody can quote. So the figure is buckets, and the claim says why a mean
    is not reported rather than leaving the absence for a reader to notice.
    """
    records = [
        _rationale("rationale-v1-0", declared_at="2026-01-01T00:00:00+00:00"),
        _rationale(
            "rationale-v2-0", revises="rationale-v1-0", declared_at="2026-01-01T00:30:00+00:00"
        ),
        _rationale("rationale-v1-1", pr=2, declared_at="2026-01-01T00:00:00+00:00"),
        _rationale(
            "rationale-v2-1",
            pr=2,
            revises="rationale-v1-1",
            declared_at="2026-01-05T00:00:00+00:00",
        ),
    ]
    figure = _INTERVAL.compute(build_snapshot(records))
    assert isinstance(figure, DistributionFigure)
    assert dict(figure.values)["under 1 hour"] == 1
    assert dict(figure.values)["3d to 7d"] == 1
    assert "mean" in figure.claim.does_not_mean
    assert "no observation occupies" in figure.claim.does_not_mean
    assert "mean" not in figure.title.casefold()


def test_an_interval_with_a_missing_timestamp_is_excluded_because_an_absent_interval_is_not_a_fast_one() -> (
    None
):
    """The other half of the absent-versus-zero argument, on the interval measure.

    A revision with no ``declared_at`` has no interval. Writing one as zero would
    put it in the smallest bucket and report a record nobody can place as a record
    revised instantly, which is the same substitution the time-to-review measure
    refuses and for the same reason.
    """
    records = [
        _rationale("rationale-v1-0", declared_at="2026-01-01T00:00:00+00:00"),
        _rationale("rationale-v2-0", revises="rationale-v1-0", declared_at=None),
    ]
    figure = _INTERVAL.compute(build_snapshot(records))
    assert isinstance(figure, DistributionFigure)
    assert dict(figure.values)["under 1 hour"] == 0
    assert dict(figure.excluded) == {NO_DECLARATION_TIME_KEY: 1}


def test_a_negative_interval_gets_its_own_key_because_clock_skew_is_a_data_defect_and_not_a_very_short_revision() -> (
    None
):
    """Two writers with their clocks out of order, reported as what they are.

    A negative gap folded into the smallest bucket would report a broken clock as
    the fastest revision in the corpus. It is excluded under a key that says which
    of the two things happened, so a reader goes and looks at the timestamps rather
    than at the distribution.
    """
    records = [
        _rationale("rationale-v1-0", declared_at="2026-01-05T00:00:00+00:00"),
        _rationale(
            "rationale-v2-0", revises="rationale-v1-0", declared_at="2026-01-01T00:00:00+00:00"
        ),
    ]
    figure = _INTERVAL.compute(build_snapshot(records))
    assert isinstance(figure, DistributionFigure)
    assert figure.values["under 1 hour"] == 0
    assert sum(figure.excluded.values()) == 1
    assert "clocks out of order" in NEGATIVE_INTERVAL


def test_the_interval_denominator_counts_the_pairs_the_figure_can_actually_place_in_a_bucket() -> (
    None
):
    """A denominator that counted an unplaceable pair would describe a different figure.

    Two chains, one of which has an interval with a missing endpoint: the figure
    shows one interval and the claim has to be a rate over one, not over two. This
    is the only thing that stops the two disagreeing, and it is why the count is
    computed rather than assumed to equal the number of revisions.
    """
    records = [
        _rationale("rationale-v1-0"),
        _rationale("rationale-v2-0", revises="rationale-v1-0", declared_at=None),
        _rationale("rationale-v1-1", pr=2),
        _rationale(
            "rationale-v2-1",
            pr=2,
            revises="rationale-v1-1",
            declared_at="2026-01-01T01:00:00+00:00",
        ),
    ]
    figure = _INTERVAL.compute(build_snapshot(records))
    assert isinstance(figure, DistributionFigure)
    assert figure.claim.denominator.size == 1
    assert "consecutive revision pairs" in figure.claim.denominator.description
    assert sum(figure.values.values()) == 1


# -- purity and shape ---------------------------------------------------------------


def test_two_reads_of_the_same_corpus_give_the_same_chains_because_nothing_here_has_a_clock() -> (
    None
):
    """A measure that read a clock would make its own output untestable.

    Both halves are asserted: the same records give equal figures twice, and the
    walk gives chains in root order regardless of the order the records arrived in.
    The second is what makes a chain a chain rather than a set, and a walk over a
    set-ordered collection would produce a different chain on every run.
    """
    records = [
        _rationale("rationale-v1-0"),
        _rationale("rationale-v2-0", revises="rationale-v1-0"),
        _rationale("rationale-v1-1", pr=2),
    ]
    snapshot = build_snapshot(records)
    assert _PRESSURE.compute(snapshot) == _PRESSURE.compute(snapshot)
    forward, _ = walk_chains(records)
    backward, _ = walk_chains(tuple(reversed(records)))
    assert [chain.records for chain in forward] == [chain.records for chain in backward]
    assert [record.entry_id for record in forward[0].records] == [
        "rationale-v1-0",
        "rationale-v2-0",
    ]


def test_records_that_are_not_rationales_are_not_part_of_a_chain_because_a_chain_is_about_a_stated_reason() -> (
    None
):
    """An answer cannot be a node in a chain, and cannot be counted as one.

    Rationales are dispatched through a different payload from the four entry
    writers, so they arrive in the rationale namespace and carry the ``rationale``
    tag. An answer carries neither. A walk that accepted any record with a
    ``rationale_revises`` would build chains out of records that are not stated
    reasons, and the whole denominator would be wrong.
    """
    answers = (
        build_record(pr=1, rationale_revises="rationale-v1-0"),
        build_record(pr=2, rationale_revises="rationale-v2-0"),
    )
    chains, excluded = walk_chains(answers)
    assert chains == ()
    assert excluded == {}
    assert all(record.classification.kind is RecordKind.ANSWER for record in answers)


def test_a_corpus_with_no_rationales_produces_an_empty_distribution_and_not_an_error_because_zero_chains_is_a_finding() -> (
    None
):
    """The empty case, which is the common one before anybody writes a rationale.

    ``Denominator`` refuses a size below one, so the claim is built with the count
    floored and the figure is a distribution with no buckets rather than a
    ``ValueError``. A measure that raised here would make "nobody has stated a
    reason yet" indistinguishable from a bug, and it is neither.
    """
    records = (build_record(pr=1), build_record(pr=2, repo="acme/gadget"))
    figure = _PRESSURE.compute(build_snapshot(records))
    assert isinstance(figure, FigureGroup)
    whole = figure.figures[0]
    assert isinstance(whole, DistributionFigure)
    assert whole.values == {}
    assert whole.claim.denominator.size == 1
    intervals = _INTERVAL.compute(build_snapshot(records))
    assert isinstance(intervals, DistributionFigure)
    # Every rung is present at zero rather than absent, so two reads of a corpus
    # with no revisions have the same shape and not two different ones.
    assert set(intervals.values.values()) == {0}
