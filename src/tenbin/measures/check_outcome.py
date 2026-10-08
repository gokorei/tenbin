"""What a build concluded about a commit, over the builds this corpus holds a report for.

Kojutsu's ``RSF1DK1S`` subscribed to check runs and stores each conclusion as a
record in its own kind. Before it, the corpus's only outcome for a change was whether
GitHub reported a merge -- and that is a fact about the forge, not a result. A check
conclusion is the nearest thing this store has ever held to a machine saying "this works",
and it is the closest the corpus gets to an outcome that is not a merge.

**A check concluding ``success`` is a fact about the build and not about the change.**
That is the whole negative space and it is the reason the wording is Kojutsu's:
``success`` is not ``correct``, and a green pipeline on a change that broke production
passed its pipeline. Nothing in this corpus can tell the two apart, because the only
thing a build observes is the change as the pipeline was configured to observe it -- and
a pipeline that does not exercise the failure is a pipeline that has never reported one.
So this figure describes what builds said, and a reader composing it into a judgement
about a change is making an inference the store never licensed.

**Revert detection is out of reach and the claim says so by name.** ``RSF1DK1S`` covers
check runs. Reverting a change is a separate event and the store does not record it, so
a change that merged and was then reverted is *indistinguishable here from one that
stuck* -- it contributes a ``merged`` outcome and whatever its builds concluded, and
nothing in this figure walks either back. That is not a caveat bolted on: it is the
reason a distribution of check conclusions cannot be read as a distribution of
survivors, and a reader who does not know it will read a green build after a revert as
evidence the change held.

**This measure is not composable with the merge outcome, and that is deliberate.** Two
descriptive facts multiplied together and rendered as one number are the causal-claim
machinery in a different costume, and this package refuses it structurally rather than by
convention: the two measures sit in separate modules, neither reads the other's field, and
a test walks the sources of every registered measure to make sure. Nothing here reads
``pr_outcome`` at all, and the reason is in this docstring rather than only in the test.

**What this module notably does not do:** it does not collapse a commit's checks into
one verdict. A commit with a passing lint and a failing type-check is two reports, and
there is no rule in the corpus for deciding which of them is the commit's outcome --
inventing one would be a composite, which is the thing refused above. So the unit is the
check run, and the population says so in the words the denominator renders.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from tenbin.claims.mechanism import Granularity
from tenbin.claims.model import Claim, ClaimKind, Denominator
from tenbin.corpus.record import Record, RecordKind
from tenbin.corpus.snapshot import CorpusSnapshot
from tenbin.measures.base import CountFigure, DistributionFigure, Figure, histogram
from tenbin.measures.filtering import requires_certainty, select

#: The slug this measure is cited by, and the ticket that delivered the keys.
CHECK_OUTCOME_SLUG: Final[str] = "corpus.check_outcome"

#: The ticket named on the claim.
UNBLOCKED_BY: Final[str] = "RSF1DK1S"

#: The conclusions the forge can report, in the words Kojutsu documents
#: (``integrations/github_models.py``). Spelled here rather than imported because the
#: vocabulary belongs to the forge and to Kojutsu's model, and *not* used as a filter:
#: an unrecognised conclusion is counted in ``excluded`` under a key carrying the value
#: itself rather than dropped, because a forge that added a tenth would otherwise make
#: this figure quietly shrink. Kept as documentation and as the vocabulary a test
#: exercises -- a closed list in this module is a maintenance burden, not a guarantee.
KNOWN_CONCLUSIONS: Final[tuple[str, ...]] = (
    "success",
    "failure",
    "neutral",
    "cancelled",
    "timed_out",
    "action_required",
    "skipped",
    "stale",
    "startup_failure",
)

#: Exclusion key for the one condition that leaves a check record with nothing to say.
NO_CONCLUSION_KEY: Final[str] = (
    "a record stating a check run that carries no conclusion, so there is nothing the build "
    "reported"
)

#: What the population is, in the words the denominator renders. "Check runs" rather than
#: "commits" is the load-bearing word: a commit with three checks contributes three
#: conclusions, and calling this a distribution over commits would invite a reader to add
#: them up into a per-commit verdict that nothing here computed.
CHECK_POPULATION: Final[str] = "check runs recorded in this read, one conclusion each"

#: A build's conclusion is about a build, stated as its own sentence because it is the
#: half of the claim a reader in a hurry will drop.
SUCCESS_IS_NOT_CORRECTNESS: Final[str] = (
    "That a change was correct, or that a failing check means the change was wrong. A check "
    "reports on a commit through the pipeline as that pipeline is configured, and a pipeline "
    "that never exercises a failure has never reported one: a green pipeline on a change that "
    "broke production passed its pipeline. The conclusion below is the forge's own wording, "
    "stored verbatim, and it is a fact about a build."
)

#: The limit a reader cannot see from the figure at all, named because its absence would
#: make the figure look like a distribution of survivors.
REVERTS_ARE_OUT_OF_REACH: Final[str] = (
    "Whether a change that merged and was later reverted is in this corpus or not -- it is not, "
    "and that is the one thing here a reader is most likely to assume. `RSF1DK1S` covers check "
    "runs; reversion is a separate event the store does not record. A change that merged and was "
    "then reverted is therefore indistinguishable in this figure from one that stuck, so nothing "
    "below counts survivors and nothing walks a conclusion back."
)


#: The slug a *checked and absent* axis reports under, kept distinct from
#: :data:`CHECK_OUTCOME_SLUG` for the same reason the other clean slugs in this
#: package are: a check that ran and found nothing and a check that could not look
#: are different facts, and one slug would let a reader take the second for the
#: first.
CHECK_OUTCOME_ABSENT_SLUG: Final[str] = "corpus.check_runs_absent"


def _axis_absent_claim() -> Claim:
    """The claim for a read that holds no check runs at all.

    Its falsifier is the one that matters here and is deliberately narrow: a check
    run appearing in a later read of this collection. Anything weaker -- another
    conclusion, another gate -- would be satisfied by a corpus that had started
    capturing and would say nothing about whether this read saw a gap.
    """
    return Claim(
        slug=CHECK_OUTCOME_ABSENT_SLUG,
        statement=(
            "This read holds no check runs, so it can say nothing about which gates ran or "
            "how often they concluded. Check runs are an opt-in subscription rather than a "
            "capture path, so their absence is a fact about this collection's configuration "
            "and not a defect in it."
        ),
        does_not_mean=(
            "That no gate ran on these changes, and not that no gate failed. A build that "
            "concluded something and was never captured left no record here at all, which is "
            "the same hole this figure reports as an absence."
        ),
        falsifier=(
            "A check run appearing in a later read of this collection would show the "
            "subscription was switched on, or turned on and off again around this read, and "
            "either way that this read's silence was about configuration rather than about "
            "the work."
        ),
        denominator=Denominator("capture records in this read", 1),
        kind=ClaimKind.descriptive,
        granularity=Granularity.corpus,
    )


def check_outcome_claim(total: int) -> Claim:
    """The claim behind the conclusion distribution, over the check runs in this read."""
    return Claim(
        slug=CHECK_OUTCOME_SLUG,
        statement=(
            "The conclusions the builds in this corpus reported are distributed as the figure "
            "below shows, over the check runs Kojutsu stored."
        ),
        does_not_mean=(
            SUCCESS_IS_NOT_CORRECTNESS
            + " "
            + REVERTS_ARE_OUT_OF_REACH
            + " It is also a distribution over check runs and not over commits: a commit with "
            "several checks contributes several conclusions, and nothing here collapses them "
            "into a verdict, because no rule in the corpus says which of a commit's checks is "
            "the commit's outcome."
        ),
        falsifier=(
            "A check-run record whose conclusion is not one the forge reports, reported here as "
            "an ordinary bucket rather than as an exclusion. The writer stores the conclusion "
            "verbatim and lowercased, so a value outside the forge's vocabulary would mean "
            "something other than a build is writing this key, and the buckets below would be "
            "describing that something."
        ),
        denominator=Denominator(CHECK_POPULATION, max(1, total)),
        kind=ClaimKind.descriptive,
        granularity=Granularity.repository,
        unblocked_by=UNBLOCKED_BY,
    )


@dataclass(frozen=True)
class CheckOutcomeMeasure:
    """The distribution of build conclusions, over the check runs this read enumerated.

    Filtered to :data:`~tenbin.corpus.record.RecordKind.CHECK_RUN` rather than to "any
    record carrying a conclusion", because the kind is what Kojutsu states and a
    record that happens to have the key without being a check report is a document this
    program has no story for. Records that are not check runs are not an exclusion here
    any more than a non-review record is one for the verdict measure: the population is
    the check runs, and the denominator says so.
    """

    requires_certainty = requires_certainty

    @property
    def slug(self) -> str:
        """The identifier this measure is cited by."""
        return CHECK_OUTCOME_SLUG

    def claim(self, snapshot: CorpusSnapshot) -> Claim:
        """The claim, over the check runs in this read that state a conclusion."""
        conclusions, _ = self._conclusions(snapshot.records)
        return check_outcome_claim(len(conclusions))

    def compute(self, snapshot: CorpusSnapshot) -> Figure:
        """The histogram of conclusions, with the silent ones named and counted.

        **A corpus with no check runs is a count of zero and not an empty
        histogram.** An empty ``DistributionFigure`` renders as its completeness
        sentence and nothing else, which is a claim with a caveat and no number
        under it -- and a reader cannot tell that from a measure which found
        nothing. The zero says the check: this corpus holds no check runs, which is
        an ordinary state for a store whose check-run subscription is off, and it
        is a fact about the corpus rather than an absence of one.
        """
        conclusions, excluded = self._conclusions(snapshot.records)
        if not conclusions and not excluded:
            return CountFigure(
                claim=_axis_absent_claim(),
                snapshot=snapshot,
                title="Check runs this corpus holds, of which it holds none",
                value=0,
            )
        return DistributionFigure(
            claim=self.claim(snapshot),
            snapshot=snapshot,
            title="Check conclusions this corpus holds a report for",
            values=histogram(conclusions),
            excluded=excluded,
        )

    def _conclusions(self, records: Sequence[Record]) -> tuple[list[str], Mapping[str, int]]:
        """Every stated conclusion, and how many check records state none.

        One walk, one filter, one exclusion. The conclusion is carried through as
        itself rather than mapped onto a local enum, because the forge owns the
        vocabulary and a local copy would either raise on a new value or fold it into a
        neighbour -- and both would lose the fact that something unrecognised was
        written, which is the whole content of a check record.
        """
        checks = select(records, certainty=self.requires_certainty, kinds=(RecordKind.CHECK_RUN,))
        conclusions = [
            record.check_conclusion for record in checks if record.check_conclusion is not None
        ]
        silent = len(checks) - len(conclusions)
        return conclusions, {NO_CONCLUSION_KEY: silent} if silent else {}
