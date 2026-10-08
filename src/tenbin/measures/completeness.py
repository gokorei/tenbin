"""How much of the store this report actually read, which is the first thing to say.

Every other figure in this program is conditional on this one, and the condition
is invisible in the number itself: a distribution over four hundred documents
looks exactly like a distribution over four thousand. So the completeness measure
comes first, and it is the only measure whose denominator is *the store's own
count* rather than a population this program chose. That choice is the whole
design. A rate of enumerated over enumerated is always exactly 1, renders as
``100.0%``, and says nothing at all -- it is the arithmetic of a tautology
dressed as a finding, and it is available here in one line of code, which is
exactly why it is not the line of code that got written.

**The empty corpus and the unreachable store must not look alike.** A reachable
store holding nothing is a fact and it gets a :class:`~tenbin.measures.base.CountFigure`
saying so. An unreachable store never produces a snapshot at all: the walk raises,
because a snapshot with no records from a store that was never reached is
indistinguishable from a corpus of nothing, and that distinction is the whole
difference between a finding and a failure. So the zero-over-zero case here is
reachable-but-empty, and a measure that rendered it as a rate would be reporting a
division it could not perform.

**A count that contradicts its own listing is not a count.** ``STORE_TOTAL_EXCEEDED``
means the store served more documents than it said it held, so the numerator is
larger than the denominator and the rate is above 1 -- which
:class:`~tenbin.measures.base.RateFigure` refuses. This measure therefore returns
a count for that case too, and lets :meth:`~tenbin.measures.base.Figure.rate_text`
carry the alarm. The claim's falsifier names the same condition, so a reader has
two independent statements about it.

**What this module notably does not do:** it does not say the corpus is
representative, and cannot. The documents enumerated here are the documents
somebody captured: self-selected, and censored by the capture paths that fail
silently. A repository nobody answers on is byte-identical to a repository with no
knowledge debt, no backfill path exists, and a completeness of 100% is a statement
about the store's count matching this read -- not about development. Saying
otherwise is the one thing this figure must never be used to do, which is why the
claim's negative space is about representativeness and not about arithmetic.
"""

from __future__ import annotations

from tenbin.claims.mechanism import Granularity
from tenbin.claims.model import Claim, ClaimKind, Denominator
from tenbin.corpus.snapshot import CorpusSnapshot
from tenbin.measures.base import CountFigure, Figure, Rate, RateFigure

#: The slug this measure is cited by, and the one a report looks up in the refusal
#: registry. Dotted rather than dashed so it can never collide with a
#: :func:`~tenbin.claims.refusals.slugify`-ed catalogue entry: the registry is
#: keyed by the measure *names* in ``docs/seam.md``, and a measure that shared
#: their spelling space would be one refactor away from being refused by a refusal
#: filed about something else.
READ_COMPLETENESS_SLUG = "corpus.read_completeness"

#: What the rate is a rate over, in the words the reader needs. The store's count
#: is the only independent check on the walk available, which is why this figure
#: and not any other one names it.
STORE_POPULATION = "documents the store says the collection holds"


class CompletenessMeasure:
    """Whether this read saw the whole store, as a rate over the store's own count.

    One measure, one figure, and the figure is a :class:`RateFigure` in the
    ordinary case. The two exceptions are both counts and both are argued in the
    module docstring: there is no population to be a fraction of when the store
    holds nothing, and there is no rate at all when the store's count is smaller
    than what it served.
    """

    __slots__ = ()

    #: A completeness figure answers "did this read see everything", which is a
    #: question about the read rather than about the work, so it belongs to the
    #: capture-system view of the report.
    about_the_capture_system = True

    @property
    def slug(self) -> str:
        """The identifier this measure is cited by."""
        return READ_COMPLETENESS_SLUG

    def claim(self, snapshot: CorpusSnapshot) -> Claim:
        """The claim, over the store's own count for *this* read.

        Built per snapshot because the denominator is a fact about the read rather
        than about the measure. A claim carrying a placeholder population would be
        a rate over a number nobody wrote down, which is the one thing
        :class:`~tenbin.claims.model.Denominator` exists to prevent.
        """
        return Claim(
            slug=READ_COMPLETENESS_SLUG,
            statement=(
                "This report covers the documents enumerated on this read of this collection, "
                "at this moment, and the store's own count of the collection agrees that the "
                "read reached the end of it."
            ),
            does_not_mean=(
                "That those documents are representative of development. They are not and "
                "cannot be: the corpus is self-selected, because Kojutsu captures only what "
                "it is present for, and self-censored, because a capture path that fails writes "
                "nothing and is indistinguishable from a change nobody examined. A repository "
                "nobody answers on is byte-identical to a repository with no knowledge debt, and "
                "no backfill path exists for either."
            ),
            falsifier=(
                "A change in the store's total between the count and the enumeration. The "
                "snapshot detects that as store_total_exceeded, and a count that no longer "
                "agrees with its own listing is not a count of this corpus at all."
            ),
            denominator=Denominator(STORE_POPULATION, max(1, snapshot.store_total)),
            kind=ClaimKind.descriptive,
            granularity=Granularity.corpus,
        )

    def compute(self, snapshot: CorpusSnapshot) -> Figure:
        """The fraction of the store's own count that this read enumerated.

        Pure: the same snapshot gives the same figure, and there is no clock, no
        store and no randomness in here. A test builds a snapshot from a fixture
        and asserts on the figure, which is the property that makes a measure
        testable at all.
        """
        if snapshot.store_total < 1:
            return CountFigure(
                claim=self.claim(snapshot),
                snapshot=snapshot,
                title=f"The store reports no documents in `{snapshot.collection}`",
                value=snapshot.enumerated,
            )
        if snapshot.enumerated > snapshot.store_total:
            # Rendered as a count because the rate cannot be computed and a
            # ``RateFigure`` refuses a numerator above its denominator. The
            # completeness sentence still fires -- ``rate_text`` reads
            # ``store_total_exceeded`` off the snapshot -- so the reader is told
            # the corpus has no agreed size rather than handed a percentage.
            return CountFigure(
                claim=self.claim(snapshot),
                snapshot=snapshot,
                title=(
                    "The store served more documents than its own count reported, so there is "
                    "no rate to compute"
                ),
                value=snapshot.enumerated,
            )
        return RateFigure(
            claim=self.claim(snapshot),
            snapshot=snapshot,
            title="Documents this read enumerated, over what the store says it holds",
            rate=Rate(
                numerator=snapshot.enumerated,
                denominator=Denominator(STORE_POPULATION, snapshot.store_total),
            ),
        )
