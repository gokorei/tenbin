"""How long a change waited for its first review, over the changes that got one.

Kojutsu's ticket ``9PTQE45Y`` stopped discarding the pull request's open time.
Before it, ``answered_at`` recorded when Kojutsu captured an answer -- a
statement about the bot's latency, not about review latency -- and the difference
between the two was roughly the bot's own turnaround. There was no way to detect
that from a rendered figure, which is what makes the substitution so easy: a number
that looks like review latency and is a mixture of review latency and a delivery
pipeline's. With ``pr_opened_at`` stored, the interval is the interval.

**From the change opening to the earliest verdict on it.** The earliest, not the
mean and not the first one the store returned: a change reviewed three times has
three intervals, and only one of them answers "how long did this change wait". The
rest are about review practice, which is a different question. The
``pr_opened_at`` used is the earliest one stated among the change's records, which
is GitHub's own creation time for the change -- every writer of a record about the
same change writes the same value, so the earliest is only a tie-break for a corpus
where one of them is wrong.

**An absent interval is not a zero interval.** A change with captures but no
review contributes *nothing* to this distribution and is counted in ``excluded``
under a key naming that. Writing it as zero would be the most destructive
available mistake: a zero interval is the fastest possible review, so a corpus
where most changes were never reviewed would render as a corpus where most changes
were reviewed instantly. The two are not adjacent on the scale; they are opposite
meanings, and the difference is precisely the thing a reader is here to learn.

**What this module notably does not do:** it does not report a mean. The intervals
are skewed -- most are short, a few are days -- and a mean of a skewed
distribution is a point no observation occupies. It also does not report a
"reviewed within N" percentage, which would require choosing N, and every choice of
N makes the threshold the finding. The buckets are the same fixed ladder every
duration in this program uses, so two measures cannot disagree about what a day is.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from tenbin.claims.mechanism import Granularity
from tenbin.claims.model import Claim, ClaimKind, Denominator
from tenbin.corpus.record import Certainty, Record, RecordKind
from tenbin.corpus.snapshot import CorpusSnapshot
from tenbin.measures.base import DistributionFigure, Figure, duration_buckets
from tenbin.measures.filtering import select

#: The slug this measure is cited by, and the ticket that unblocked it.
TIME_TO_REVIEW_SLUG: Final[str] = "corpus.time_to_first_review"

#: The ticket named on the claim, in the field whose whole job is to say what would
#: change the answer.
UNBLOCKED_BY: Final[str] = "9PTQE45Y"

#: Exclusion keys. Both name a condition, and both exist because the alternative is
#: a zero or a dropped change.
NO_REVIEW_KEY: Final[str] = "change has captures but no review verdict to measure a wait from"
NO_OPEN_TIME_KEY: Final[str] = "change has a review but no pr_opened_at to measure the wait from"

#: What the population is, in the words the denominator renders.
INTERVAL_POPULATION: Final[str] = "changes in this read with a review and a recorded open time"


def time_to_review_claim(total: int) -> Claim:
    """The claim behind the interval distribution, over the changes that have both ends."""
    return Claim(
        slug=TIME_TO_REVIEW_SLUG,
        statement=(
            "The time from a change opening to the earliest review verdict recorded against it is "
            "distributed as the buckets below show."
        ),
        does_not_mean=(
            "How long review takes in this organisation, or how long a change waited to be "
            "looked at by a person. What is measured is the gap between two writes about the "
            "same change: when GitHub reported it opening, and when Kojutsu captured a verdict "
            "on it. Delivery latency, a reviewer reading the change before submitting, and a "
            "reviewer not looking at all are all inside that gap and none of them is separable "
            "from the others here. No mean is reported: the intervals are skewed, and a mean of "
            "a skewed distribution is a point no observation occupies."
        ),
        falsifier=(
            "Two records for the same change whose `pr_opened_at` values differ. Every writer of "
            "a record about one change should write GitHub's creation time for it, so a "
            "disagreement would mean the interval is being computed from more than one open time "
            "and the buckets would be comparing two different questions."
        ),
        denominator=Denominator(INTERVAL_POPULATION, max(1, total)),
        kind=ClaimKind.descriptive,
        granularity=Granularity.repository,
        unblocked_by=UNBLOCKED_BY,
    )


@dataclass(frozen=True)
class TimeToReviewMeasure:
    """Bucketed waits from a change opening to its earliest recorded verdict."""

    #: Determined, because a record read as a review verdict from its tags is a
    #: reading, and "the earliest verdict on this change" is only the earliest
    #: verdict if the records really are verdicts.
    requires_certainty = Certainty.DETERMINED

    @property
    def slug(self) -> str:
        """The identifier this measure is cited by."""
        return TIME_TO_REVIEW_SLUG

    def claim(self, snapshot: CorpusSnapshot) -> Claim:
        """The claim, over the changes in this read that have both ends of the interval."""
        intervals, _ = self._intervals(snapshot.records)
        return time_to_review_claim(len(intervals))

    def compute(self, snapshot: CorpusSnapshot) -> Figure:
        """The bucketed distribution, with the unmeasurable changes named and counted."""
        intervals, excluded = self._intervals(snapshot.records)
        return DistributionFigure(
            claim=self.claim(snapshot),
            snapshot=snapshot,
            title="Time from a change opening to the earliest review verdict on it",
            values=duration_buckets(intervals),
            excluded=excluded,
        )

    def _intervals(self, records: Sequence[Record]) -> tuple[list[float], dict[str, int]]:
        """Every measurable wait, and what made the others unmeasurable.

        A change is a ``(repo, pr)`` pair, or nothing: :attr:`Record.pr_key` refuses
        a half key, and a record with a pull request number and no repository must
        not join to every repository holding the same number. A change with
        captures but no verdict has no end to its interval; a change with a verdict
        but no ``pr_opened_at`` has no start. Both are counted, separately, because
        they are different gaps in the corpus and a reader fixing one of them needs
        to know which they have.
        """
        by_change: dict[tuple[str, int], list[Record]] = defaultdict(list)
        for record in records:
            key = record.pr_key
            if key is not None:
                by_change[key].append(record)
        intervals: list[float] = []
        excluded: dict[str, int] = {}
        for group in by_change.values():
            verdicts = select(
                group, certainty=self.requires_certainty, kinds=(RecordKind.REVIEW_VERDICT,)
            )
            if not verdicts:
                excluded[NO_REVIEW_KEY] = excluded.get(NO_REVIEW_KEY, 0) + 1
                continue
            opened = next(
                (record.pr_opened_at for record in group if record.pr_opened_at is not None), None
            )
            if opened is None:
                excluded[NO_OPEN_TIME_KEY] = excluded.get(NO_OPEN_TIME_KEY, 0) + 1
                continue
            earliest = min(
                (record.answered_at for record in verdicts if record.answered_at is not None),
                default=None,
            )
            if earliest is None:
                excluded[NO_OPEN_TIME_KEY] = excluded.get(NO_OPEN_TIME_KEY, 0) + 1
                continue
            hours = (earliest - opened).total_seconds() / 3600.0
            if hours < 0:
                # A review captured before the change opened, which is either clock
                # skew or an open time recorded from the wrong event. Excluded under
                # the open-time key because that is the field a reader would go and
                # check first, and it is the wrong end that is more likely.
                excluded[NO_OPEN_TIME_KEY] = excluded.get(NO_OPEN_TIME_KEY, 0) + 1
                continue
            intervals.append(hours)
        return intervals, excluded
