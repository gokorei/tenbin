"""How long the corpus took to hold something that already existed.

A capture record carries two clocks about one event: ``answered_at`` is the moment a
human or an agent wrote the answer, and ``captured_at`` is the moment Kojutsu built
the entry out of it. **The gap between the two is the capture pipeline's own latency,
and it is the only figure in this program that measures the pipeline rather than the
work it was carrying.**

**The interval begins when the webhook fires, and it is neither the forge's latency
nor the store's.** ``captured_at`` is stamped when the entry is *built*, before the
enqueue, so everything downstream of that stamp -- the write, the retry, the drain --
is outside this measurement and cannot move a number here. What is inside is the work
between a human writing an answer and Kojutsu assembling the document out of it:
the worker's own turnaround, and whatever the build has to read to do the work.
Kojutsu calls a queued row the only copy of a human's answer, and this interval is
how long that row existed with nobody holding it. **A reader who takes this for
delivery latency will read a delivery spike as the forge being slow,** and the forge is
then the wrong system, because it answered long before the interval started.

**A distribution, and never a mean or a percentile, because the distribution is bimodal
by construction.** The same tick that makes the interval honest is a tick that
retries: most rows take the ordinary path and a few sit behind a store outage for
hours or days. One summary number over two modes describes neither of them, and a
percentile is the worse of the two summaries here because it silently reports which
mode it landed in. The buckets are
:func:`~tenbin.measures.base.duration_buckets` -- the same fixed ladder every other
duration in this program uses, because two measures that disagreed about what a day is
would be two definitions of the same axis.

**A record carrying only one of the two moments is counted and named, never dated
approximately.** A record with no ``answered_at`` has no start, and one with no
``captured_at`` has no end; both are absences rather than zeroes, and a zero interval
is the fastest possible capture, so a corpus where most records predate a field would
render as a corpus that captures instantly. The two absences get two keys, because a
reader going to fix one of them needs to know which one they have.

**A clock that disagrees with itself is a fact about the corpus and not a fast
capture.** A ``captured_at`` earlier than its own ``answered_at`` is skew between two
writes, and it is counted in ``excluded`` under
:data:`~tenbin.measures.base.NEGATIVE_INTERVAL` -- the label the duration ladder
already gives that condition, reused rather than re-spelled, so a reader meets one
phrase for it wherever they meet it.

**What this module notably does not do:** it does not measure the latency of anything
besides the path between the answer and the entry, and it does not publish its figure
beside an outcome. A duration says how long the pipeline took; an outcome says what
happened to the change. **No figure in this program puts a duration and an outcome in
one number**, which is the same prohibition that keeps a merge outcome from being
multiplied by a check conclusion: the composite is where a causal reading gets in
without an assignment mechanism, and a latency scored against an outcome is that
composite wearing a delivery costume.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from tenbin.claims.mechanism import Granularity
from tenbin.claims.model import Claim, ClaimKind, Denominator
from tenbin.corpus.provenance import CAPTURED_SOURCES
from tenbin.corpus.record import Record
from tenbin.corpus.snapshot import CorpusSnapshot
from tenbin.measures.base import (
    NEGATIVE_INTERVAL,
    DistributionFigure,
    Figure,
    duration_buckets,
)
from tenbin.measures.filtering import capture_population

#: The slug this measure is cited by. Dotted like every other slug in the package, so it
#: cannot collide with a slugified row in ``docs/seam.md``.
CAPTURE_LATENCY_SLUG: Final[str] = "corpus.capture_latency"

#: What the population is, in the words the denominator renders. "Capture records" is
#: :data:`~tenbin.measures.filtering.CAPTURE_POPULATION_WORD`, and using the same words
#: is what lets a reader add this figure's size to another capture figure's without first
#: working out whether the two counted the same documents. The trailing clause is the
#: filter: only a record whose provenance says a delivery path built it has an interval
#: between writing and stamping, and the counts of the others are in ``excluded`` under
#: their own keys rather than in a bucket.
CAPTURE_LATENCY_POPULATION: Final[str] = (
    "capture records in this read that a delivery path built, naming both the answer's own "
    "time and the corpus's own"
)

#: The four exclusion keys, each naming its condition rather than a code. A report
#: renders them verbatim, so each is a sentence an operator can act on, and the pairs
#: that could be confused with one another are kept apart: a missing start is not a
#: missing end, a record somebody typed is not a record with no provenance, and a
#: backwards clock is neither of them.
NO_ANSWERED_AT_KEY: Final[str] = (
    "record names no time for when the answer was written, so the wait has no start"
)
NO_CAPTURED_AT_KEY: Final[str] = (
    "record names no time for when Kojutsu stamped the entry, so the wait has no end"
)
TYPED_IN_KEY: Final[str] = (
    "a record somebody typed in, so no delivery path was ever behind it and its two "
    "timestamps are one person writing twice"
)
NO_CAPTURE_SOURCE_KEY: Final[str] = (
    "record states no capture source, so nothing says whether a delivery path built it"
)


def capture_latency_claim(total: int) -> Claim:
    """The claim behind the interval distribution, over the records carrying both moments.

    A function rather than a constant, for the reason every claim in this package is one:
    the denominator is how many records on *this* read have both moments, which is a fact
    about the read and not about the measure. A claim carrying a placeholder population
    would be a claim about a corpus nobody described.

    The population is stated as records that a delivery path built *and* that carry both
    clocks, so the two exclusions that remove records -- a typed-in record and a record
    with no provenance -- are visibly outside it rather than quietly inside a number a
    reader is supposed to add up.
    """
    return Claim(
        slug=CAPTURE_LATENCY_SLUG,
        statement=(
            "The time between an answer being written and Kojutsu stamping the entry that "
            "holds it is distributed as the buckets below show, over the records that name both "
            "moments."
        ),
        does_not_mean=(
            "How slowly the forge delivers, and not how slowly the store writes. "
            "The interval begins when the webhook fires and ends when the entry is stamped, and "
            "the stamp is taken from Kojutsu's own clock at build time, before the enqueue: "
            "so what is measured is the work between an answer existing and the document "
            "holding it -- the worker's own turnaround and whatever the build has to read to "
            "assemble it. The store's write happens after the stamp and is not inside this "
            "interval at all, which is the sentence that stops a reader chasing an outage in "
            "the wrong system, and the forge answered long before the interval started. It is "
            "also not a mean or a percentile, because the distribution is bimodal by "
            "construction: the bulk is the ordinary path and the tail is rows that sat behind a "
            "store outage, and one summary number over two modes describes neither of them. A "
            "percentile is the worse of the two here, because it reports which mode it landed "
            "in without saying so. The shape is the finding, and a single number for it would "
            "be the finding's loss."
        ),
        falsifier=(
            "A read in which the same answers, written at the same times, land in different "
            "buckets. The stamp is taken from Kojutsu's own clock at build time, so the "
            "interval can only move if the build got later or the path between the two did; a "
            "figure that tracked the store's reported write latency, or the size of what was "
            "answered, would mean this interval was measuring something downstream of the thing "
            "it names. The other falsifier is a store that stamps the entry after the enqueue: "
            "the buckets would then hold the store's latency and nothing in the corpus would "
            "say so."
        ),
        denominator=Denominator(CAPTURE_LATENCY_POPULATION, max(1, total)),
        kind=ClaimKind.descriptive,
        # Repository rather than corpus: a webhook is per repository, so this is a
        # statement about how long one repository's answers took to reach the store.
        # Nothing here ranks anybody, so the per-principal floor does not apply.
        granularity=Granularity.repository,
    )


@dataclass(frozen=True)
class CaptureLatencyMeasure:
    """The gap between an answer existing and the corpus holding it, in buckets.

    Bucketed rather than summarised, for the reason the claim gives: the distribution is
    bimodal -- the ordinary path and the outage path -- and a mean or a percentile over it
    describes one mode while claiming to describe the other. The ladder is
    :func:`~tenbin.measures.base.duration_buckets`, the same one every other duration in
    this program uses, so a reader comparing this figure with a review wait or a supersession
    wait is comparing distributions and not comparing definitions of a bucket.

    A record that cannot be measured is counted in ``excluded`` under a key naming why, and
    never in a bucket. Dropping them silently would make this a distribution over the
    records that happen to carry both timestamps, which is a selected population wearing a
    plain sentence.
    """

    #: Every capture kind counts, whatever certainty its classification carries. The kind is
    #: not what this measures, and the only kind that can be inferred is the answer -- so
    #: requiring ``DETERMINED`` would drop every legacy answer, and a legacy answer is
    #: exactly the record whose delivery path this figure is about.
    requires_certainty = None

    @property
    def slug(self) -> str:
        """The identifier this measure is cited by."""
        return CAPTURE_LATENCY_SLUG

    def claim(self, snapshot: CorpusSnapshot) -> Claim:
        """The claim, over the records in this read that name both moments."""
        intervals, _ = self._intervals(snapshot.records)
        return capture_latency_claim(len(intervals))

    def compute(self, snapshot: CorpusSnapshot) -> Figure:
        """The bucketed distribution, with every unmeasurable record named and counted."""
        intervals, excluded = self._intervals(snapshot.records)
        return DistributionFigure(
            claim=self.claim(snapshot),
            snapshot=snapshot,
            title="How long the corpus took to hold an answer that already existed",
            values=duration_buckets(intervals),
            excluded=excluded,
        )

    def _intervals(self, records: Sequence[Record]) -> tuple[list[float], dict[str, int]]:
        """Every measurable interval, and what made the other records unmeasurable.

        **The provenance check comes first, because it is the deeper reason.** A record
        somebody typed in has two timestamps written by one person at one moment, and
        asking whether they are in the right order is asking a question about a document
        rather than about a path. It is excluded under its own key, beside the key for a
        record that states no provenance at all -- silence about provenance is a third
        condition, not a weaker version of assertion, and a reader deciding whether to
        trust the corpus needs to know which of the two they have.

        **A negative interval is excluded rather than bucketed, and the key is the ladder's
        own label.** :func:`~tenbin.measures.base.duration_buckets` does give a backwards
        interval a bucket of its own rather than folding it into the fastest one, so it
        cannot be misread as a quick capture either way; it is excluded here because a
        clock that disagrees with itself is countable on its own, and a reader who has
        three of them wants the number and not a rung.
        """
        measured: list[float] = []
        excluded: dict[str, int] = {}
        for record in capture_population(records):
            source = record.capture_source
            if source is None:
                _bump(excluded, NO_CAPTURE_SOURCE_KEY)
                continue
            if source not in CAPTURED_SOURCES:
                _bump(excluded, TYPED_IN_KEY)
                continue
            if record.answered_at is None:
                _bump(excluded, NO_ANSWERED_AT_KEY)
                continue
            if record.captured_at is None:
                _bump(excluded, NO_CAPTURED_AT_KEY)
                continue
            hours = (record.captured_at - record.answered_at).total_seconds() / 3600.0
            if hours < 0:
                _bump(excluded, NEGATIVE_INTERVAL)
                continue
            measured.append(hours)
        return measured, excluded


def _bump(counts: dict[str, int], key: str) -> None:
    """Count one more record under a condition, for the ``excluded`` mapping.

    A function rather than an open-coded ``counts[key] = counts.get(key, 0) + 1`` at each
    branch, because there are five branches and the expression is the kind that gets
    transcribed slightly differently in the fifth one.
    """
    counts[key] = counts.get(key, 0) + 1
