"""How often a stated reason changed after it was stated, counted as chains.

Kojutsu appends rationale revisions and never overwrites: ``rationale_revision``
advances and ``rationale_revises`` names the entry being superseded. That is the
one event stream in the store, and it is the closest thing here to a measurement
of a decision changing its mind.

**The denominator is chains, not records, and that is the whole measure.** A
rationale revised four times is one stated reason that changed four times, and a
denominator of records would report it as four separate acts of revision by four
separate authors -- four times the moral weight, for the same event. So a chain is
a root plus everything reachable from it by following ``rationale_revises``, and a
chain of length one is a stated reason nobody revised. The claim's denominator is
the number of chains; the figure's buckets are chain lengths. Every number here is
about the number of *reasons* that changed, which is the quantity the reader is
actually asking about.

**A dangling target is an exclusion, not a root and not a deletion.** A record
whose ``rationale_revises`` names an entry absent from this corpus is a chain that
cannot be followed -- the earlier revision exists somewhere Tenbin cannot see, or
no longer exists. It is counted in ``excluded`` under a key naming that condition.
It is deliberately *not* treated as a root, because a root is a stated reason that
nothing superseded, and saying so about a record that explicitly names what it
supersedes would be a statement inverting the record. It is not dropped either:
a chain-length distribution over the followable chains and no mention of the
others is exactly the silent-filtering failure this package exists to report.

**The elapsed time between revisions is a distribution, never a mean.** The
intervals are skewed by construction: most are short, a few are days, and a mean
of a skewed distribution is a point no observation occupies -- a number nobody can
act on and everybody can quote.

**A revision is not a retraction.** This is the one figure here whose most natural
misreading is not a technical error but a moral one, and the corpus inventory says
directly that whether the first reason was better is not supportable. So the claim
says it in a sentence built not to be skimmed past, and the split by
``rationale_source`` is there for the same reason: a reconstructed rationale was
assembled after the fact, and one being revised is a different phenomenon from a
declared one being revised.

**What this module notably does not do:** it does not compare the text of a
revision with the text it supersedes, and it does not say a reason got better or
worse. The store holds both texts and this measure reads neither, because a
diff that showed the newer reason to be more thorough would be a finding about two
prose fragments written by different processes at different times, and the reader
who wanted to know whether the original was right would take it as one.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from tenbin.claims.mechanism import Granularity
from tenbin.claims.model import Claim, ClaimKind, Denominator
from tenbin.corpus.record import Record, RecordKind
from tenbin.corpus.snapshot import CorpusSnapshot
from tenbin.measures.base import (
    DistributionFigure,
    Figure,
    FigureGroup,
    duration_buckets,
    freeze_counts,
)
from tenbin.measures.filtering import requires_certainty, select

#: The two slugs. Separate measures because the populations differ: one is a rate
#: over chains and the other is a distribution over revision intervals, and a
#: figure carrying a claim whose denominator does not describe it is the defect
#: this package is built to stop.
REVISION_PRESSURE_SLUG: Final[str] = "corpus.rationale_revision_pressure"
REVISION_INTERVAL_SLUG: Final[str] = "corpus.rationale_revision_interval"

#: Exclusion keys, each naming the condition rather than a code.
DANGLING_REVISION_KEY: Final[str] = "revises an entry that is not in this corpus"
UNLINKABLE_KEY: Final[str] = "carries no entry id, so it cannot be a node in a chain"
NO_DECLARATION_TIME_KEY: Final[str] = "one of the two revisions has no declared_at timestamp"
BACKWARDS_INTERVAL_KEY: Final[str] = "the later revision is timestamped before the one it revises"

#: The key for a chain-length bucket. ``"one revision, never revised"`` rather than
#: ``1``: the bucket is a fact about a stated reason, and rendering it as an integer
#: invites reading the count of chains rather than the length of each.
CHAIN_LENGTH_LABEL: Final[str] = "chain of {length} ({statement})"

#: The sources Kojutsu writes, in the order a reader wants them. ``unknown`` is a
#: real value for a rationale written before this axis existed -- the writer stated
#: that it did not know where the reasoning came from, which is a weaker claim than
#: having said nothing and is worth its own bucket rather than an absence.
RATIONALE_SOURCES: Final[tuple[str, ...]] = ("declared", "reconstructed", "unknown")

#: What a chain-length distribution is a rate over, in the words the denominator
#: renders. "Reasons" and not "records" is the entire point; see the module
#: docstring.
CHAIN_POPULATION: Final[str] = "stated reasons in this corpus, counted as chains"
INTERVAL_POPULATION: Final[str] = "consecutive revision pairs within those chains"

#: The negative space, written once and rendered into both claims. This is the one
#: figure whose most natural misreading is moral rather than technical, so the
#: sentence is built to be hard to skim and says what is not supportable rather than
#: only what is.
REVISION_IS_NOT_RETRACTION: Final[str] = (
    "A revision is not a retraction, and nothing here says the first reason was wrong. A "
    "rationale is revised because work moved on and the reason that was stated no longer "
    "describes the work; the later reason may be more thorough, more careless, or both, and the "
    "corpus does not support the question of which. Reading a chain as a record of somebody "
    "changing their mind is a moral reading of a corpus that records that a statement was "
    "superseded, and it is the wrong one: whether the superseded reason was better is not "
    "supportable from these records at all."
)

#: Why the split by source exists, as its own sentence so it is not lost in the
#: caveat.
SOURCE_SPLIT_NOTE: Final[str] = (
    "Chains are split by the source of their root: a reconstructed rationale was assembled after "
    "the fact, and one being revised is a different phenomenon from a declared reason being "
    "revised."
)


def revision_claim(chains: int) -> Claim:
    """The claim behind the chain-length figures, over the chains themselves."""
    return Claim(
        slug=REVISION_PRESSURE_SLUG,
        statement=(
            "The stated reasons in this corpus changed as often as the chain-length distribution "
            "below shows, counting each stated reason once however many times it was revised."
        ),
        does_not_mean=REVISION_IS_NOT_RETRACTION
        + " "
        + SOURCE_SPLIT_NOTE
        + " It also does not mean the corpus is under review: a chain is a record that a "
        "statement was superseded, and the superseding statement is the one this corpus holds as "
        "current.",
        falsifier=(
            "A rationale whose `rationale_revises` names a chain of length one. A reason that "
            "supersedes nothing is a new stated reason rather than a revision, and finding one "
            "would mean the chain-walk is reading a supersession that the record does not claim."
        ),
        denominator=Denominator(CHAIN_POPULATION, max(1, chains)),
        kind=ClaimKind.descriptive,
        granularity=Granularity.corpus,
    )


def revision_interval_claim(intervals: int) -> Claim:
    """The claim behind the elapsed-time distribution, over the revision intervals."""
    return Claim(
        slug=REVISION_INTERVAL_SLUG,
        statement=(
            "Where a stated reason was revised, the time between one revision and the next is "
            "distributed as the buckets below show."
        ),
        does_not_mean=(
            "How long it takes to revise a reason. The interval is between two writes about the "
            "same stated reason, and what happened in between -- more code, a meeting, a "
            "reversal -- is not in the corpus. No mean is reported: the intervals are skewed and "
            "a mean of a skewed distribution is a point no observation occupies. "
            + REVISION_IS_NOT_RETRACTION
        ),
        falsifier=(
            "A pair of revisions whose `declared_at` values agree with the buckets for a corpus "
            "that is otherwise unchanged. Buckets are a fixed ladder in hours; if the boundaries "
            "moved between two reads of the same corpus, the comparison between the two reads is "
            "not a comparison."
        ),
        denominator=Denominator(INTERVAL_POPULATION, max(1, intervals)),
        kind=ClaimKind.descriptive,
        granularity=Granularity.corpus,
    )


@dataclass(frozen=True)
class RationaleChain:
    """One stated reason and every revision of it, in the order they were written.

    ``records`` is ordered root-first, which the walk guarantees by construction:
    each step follows a ``rationale_revises`` edge, so a record can only be reached
    after the entry it supersedes. ``source`` is the *root's* source, because the
    source describes how the chain came to exist and the root is where it did -- a
    revision of a declared reason is still a declared line of reasoning, and a
    reconstruction being revised is a different phenomenon whichever of its later
    entries says otherwise.
    """

    records: tuple[Record, ...]
    source: str | None

    @property
    def length(self) -> int:
        """How many statements this stated reason has had, the root included."""
        return len(self.records)

    def intervals(self) -> tuple[float, ...]:
        """Hours between each revision and the one before it, in chain order.

        A pair where either timestamp is missing yields nothing rather than zero:
        an absent interval is not a fast one, and a zero would be indistinguishable
        from two revisions written in the same second. A negative interval is
        returned as it is so the caller can exclude it under a key naming the
        condition -- clock skew between two writers is a data defect, and folding
        it into the smallest bucket would report it as the fastest revision there
        was.
        """
        gaps: list[float] = []
        for earlier, later in zip(self.records, self.records[1:], strict=False):
            if earlier.declared_at is None or later.declared_at is None:
                gaps.append(_missing_interval())
                continue
            gaps.append((later.declared_at - earlier.declared_at).total_seconds() / 3600.0)
        return tuple(gaps)


@dataclass(frozen=True)
class RevisionPressureMeasure:
    """How many times a stated reason changed, over the stated reasons rather than
    the records."""

    requires_certainty = requires_certainty

    @property
    def slug(self) -> str:
        """The identifier this measure is cited by."""
        return REVISION_PRESSURE_SLUG

    def claim(self, snapshot: CorpusSnapshot) -> Claim:
        """The claim, over the number of followable chains in this read."""
        chains, _ = walk_chains(snapshot.records)
        return revision_claim(len(chains))

    def compute(self, snapshot: CorpusSnapshot) -> Figure:
        """The chain-length distribution, and the same split by the root's source.

        Four figures under one claim because all four are over the same population
        -- the chains -- and the per-source ones are a decomposition of it rather
        than a different sample. The intervals are a *different* population and
        therefore a different measure, with its own claim; see
        :class:`RevisionIntervalMeasure`.
        """
        chains, excluded = walk_chains(snapshot.records)
        claim = self.claim(snapshot)
        figures = [
            DistributionFigure(
                claim=claim,
                snapshot=snapshot,
                title="Stated reasons by how many times they were revised",
                values=chain_lengths(chains),
                excluded=excluded,
            )
        ]
        for source in sources_present(chains):
            subset = [chain for chain in chains if chain.source == source]
            figures.append(
                DistributionFigure(
                    claim=claim,
                    snapshot=snapshot,
                    title=(
                        f"Stated reasons whose root is {source}, by how many times they were "
                        "revised"
                    ),
                    values=chain_lengths(subset),
                    excluded={},
                )
            )
        return FigureGroup(
            claim=claim,
            snapshot=snapshot,
            title="How often a stated reason changed after it was stated",
            figures=tuple(figures),
        )


@dataclass(frozen=True)
class RevisionIntervalMeasure:
    """Elapsed time between consecutive revisions, as buckets and never as a mean."""

    requires_certainty = requires_certainty

    @property
    def slug(self) -> str:
        """The identifier this measure is cited by."""
        return REVISION_INTERVAL_SLUG

    def claim(self, snapshot: CorpusSnapshot) -> Claim:
        """The claim, over the number of measurable revision pairs in this read."""
        chains, _ = walk_chains(snapshot.records)
        return revision_interval_claim(count_measurable_intervals(chains))

    def compute(self, snapshot: CorpusSnapshot) -> Figure:
        """The bucketed distribution, with the unmappable intervals named and counted."""
        chains, _ = walk_chains(snapshot.records)
        measured: list[float] = []
        missing = 0
        backwards = 0
        for chain in chains:
            for hours in chain.intervals():
                if hours != hours:  # NaN stands for "a timestamp was missing".
                    missing += 1
                elif hours < 0:
                    backwards += 1
                else:
                    measured.append(hours)
        excluded: dict[str, int] = {}
        if missing:
            excluded[NO_DECLARATION_TIME_KEY] = missing
        if backwards:
            excluded[BACKWARDS_INTERVAL_KEY] = backwards
        return DistributionFigure(
            claim=self.claim(snapshot),
            snapshot=snapshot,
            title="Time between one revision of a stated reason and the next",
            values=duration_buckets(measured),
            excluded=excluded,
        )


def walk_chains(records: Iterable[Record]) -> tuple[tuple[RationaleChain, ...], Mapping[str, int]]:
    """Follow ``rationale_revises`` from every root, and name what could not be followed.

    Two collections, and the second is the reason this function returns a pair: a
    distribution over the chains that could be walked, with no account of the ones
    that could not, is a distribution over a population nobody described.

    A root is a rationale record that supersedes nothing. A record that supersedes
    something absent from this corpus is neither a root nor dropped -- it is
    counted in ``excluded`` and reached by nobody, which is the same fact as saying
    the chain cannot be followed and is different from saying the reason was never
    revised. A record with no entry id cannot be a node at all, and is counted
    under its own condition so the two absences stay distinguishable.
    """
    rationales = select(records, kinds=(RecordKind.RATIONALE,))
    by_entry = {record.entry_id: record for record in rationales if record.entry_id}
    superseded_by: dict[str, list[Record]] = {}
    excluded: dict[str, int] = {}
    for record in rationales:
        if not record.entry_id:
            excluded[UNLINKABLE_KEY] = excluded.get(UNLINKABLE_KEY, 0) + 1
            continue
        target = record.rationale_revises
        if not target:
            continue
        if target not in by_entry:
            excluded[DANGLING_REVISION_KEY] = excluded.get(DANGLING_REVISION_KEY, 0) + 1
            continue
        superseded_by.setdefault(target, []).append(record)

    chains: list[RationaleChain] = []
    roots = sorted(
        (record for record in rationales if record.entry_id and not record.rationale_revises),
        key=lambda record: record.entry_id,
    )
    for record in roots:
        # Roots are walked in entry-id order rather than arrival order, so the
        # list of chains is a function of the corpus and not of the order the
        # store happened to list it in. A figure whose buckets are in one order
        # today and another tomorrow is a figure that cannot be diffed between two
        # runs, and a chain walk is exactly where that would go unnoticed.
        chain: list[Record] = [record]
        seen = {record.entry_id}
        cursor = record.entry_id
        while True:
            successors = [
                child for child in superseded_by.get(cursor, []) if child.entry_id not in seen
            ]
            if not successors:
                break
            # One successor at a time, lowest entry id first. A record superseded
            # twice is a fork and a fork is not a chain, so the choice cannot be
            # *right*; taking the lowest id makes it the same wrong choice on every
            # read, which is the property that lets a test assert on the figure.
            successor = min(successors, key=lambda item: item.entry_id)
            chain.append(successor)
            seen.add(successor.entry_id)
            cursor = successor.entry_id
        chains.append(RationaleChain(records=tuple(chain), source=record.rationale_source))
    return tuple(chains), freeze_counts(excluded, "walk_chains.excluded")


def sources_present(chains: Sequence[RationaleChain]) -> tuple[str, ...]:
    """The root sources actually in the corpus, the known vocabulary first.

    Built from the data rather than iterated as :data:`RATIONALE_SOURCES`, because
    a loop over the three known values would silently drop a chain whose root says
    something else -- and a per-source figure that quietly omits a source is a
    decomposition whose parts do not sum to the whole, which is the one arithmetic
    error a reader is most likely to notice and least able to explain.
    """
    present = {chain.source for chain in chains}
    known = tuple(source for source in RATIONALE_SOURCES if source in present)
    other = tuple(
        sorted(source for source in present if source is not None and source not in known)
    )
    unknown = ("(no source stated)",) if None in present else ()
    return known + other + unknown


def chain_lengths(chains: Sequence[RationaleChain]) -> Mapping[str, int]:
    """Bucket chains by how many statements they have, root included.

    Every length present in the corpus gets a bucket, and the label says in words
    what the length means. A length of one is "never revised", which is the largest
    and least alarming bucket in the distribution and the one a reader is most
    likely to skim; calling it ``1`` is how that happens.
    """
    lengths = Counter(chain.length for chain in chains)
    return freeze_counts(
        {
            CHAIN_LENGTH_LABEL.format(
                length=length,
                statement="never revised" if length == 1 else f"{length - 1} revision(s)",
            ): count
            for length, count in sorted(lengths.items())
        },
        "chain_lengths",
    )


def count_measurable_intervals(chains: Sequence[RationaleChain]) -> int:
    """How many revision pairs have both timestamps, for the claim's denominator.

    Counted here rather than assumed to equal the number of revisions, because a
    chain of length four has three intervals and one of them may have an endpoint
    without a timestamp -- and a denominator that counted a pair the figure cannot
    place in a bucket is a denominator for a different number than the figure shows.
    """
    total = 0
    for chain in chains:
        for hours in chain.intervals():
            if hours == hours and hours >= 0:
                total += 1
    return total


def _missing_interval() -> float:
    """The sentinel for a revision pair with an endpoint that has no timestamp.

    ``NaN`` rather than ``None``, because a sentinel that compares equal to nothing
    invites a caller to treat it as a duration and put it in a bucket. ``NaN``
    survives arithmetic as ``NaN``, so the only correct handling of it -- skip it,
    and say so -- is the one that works by accident rather than by a check somebody
    has to remember. Every comparison against it is false, including ``==``, which
    is exactly the test used to recognise it.
    """
    return float("nan")
