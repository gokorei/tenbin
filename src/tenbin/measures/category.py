"""What kind of thing gets captured, over a six-value vocabulary with no review step.

``category`` is the closest the corpus comes to saying what it is for. Kojutsu
assigns one of six values at capture -- ``design_decision``, ``trade_off``,
``domain_knowledge``, ``edge_case``, ``dependency``, ``system_event`` -- and
assigns it *retrospectively*, on Kojutsu's own reasoning: a rationale states a
reason before anybody asks, so a forward-looking category would let a claim carry
the label of a decision already taken. That makes the mix a real distribution
rather than a plan, and it is the only field in the corpus that describes the
subject matter of what was written down.

**A model chose every one of these labels and nobody checked them afterwards.**
The categoriser is a model and the store holds its output without a review step,
so the distribution describes the categoriser's behaviour over this corpus and
nothing more. It is not a measure of importance and not a measure of what the
corpus is *for*: a rare category is not a rare kind of knowledge, and a common one
is not important. The claim says so in its own negative space rather than leaving
it to a reader who has not read Kojutsu's prompt.

**The figures are split by record kind, and the split is the measure.** Two of the
six values are constants rather than observations. ``system_event`` is what a
lifecycle record carries and ``design_decision`` is what a verdict carries, so a
single distribution over the whole corpus would move every time the *mix of record
kinds* moved -- a reader comparing two runs would see a change in what gets
captured where nothing about capture had changed. The three figures below partition
the capture population, so a reader holding the whole thing can add them up, and no
one of them is a blend whose movement has two causes.

**Every rung is present at zero, in Kojutsu's declaration order.** The six values
are a closed set rather than a ranked one, so a distribution over them has a shape
that is the same shape on every run, and a reader comparing two runs needs the empty
rungs to be *there* to see they were empty rather than to infer it from an absence
-- the same argument :func:`tenbin.measures.base.duration_buckets` makes, and the
same reason the ordering here is the declaration order rather than the count order
:func:`tenbin.measures.base.histogram` uses. Declaration order is not a ranking
because Kojutsu does not rank these: there is no weakest category in this
vocabulary, and a figure that came out in one would be an argument nobody made.

**A category outside the six, and a record with none at all, are two different
facts.** A record Kojutsu assigned nothing to says nothing about what it is; a
record carrying a seventh value is a writer that has changed, and the value it
carried is the interesting part. So the absence goes to ``excluded`` under a key
naming the condition and the seventh value goes to ``excluded`` under a key naming
itself, exactly as :mod:`tenbin.measures.outcome` handles a third merge outcome.
Neither is folded into a neighbouring category, because both of the neighbours are
things somebody was categorised *as*.

**What this module notably does not do:** it does not report what a category is
worth, it does not aggregate the three populations into a single mix, and it does
not compare a category against an outcome, an independence level or a change that
merged. Each of those is a small transformation of a description of what was
written down into a statement about whether it was any good, which is the claim this
package refuses everywhere else and which this corpus has nothing to support.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from tenbin.claims.mechanism import Granularity
from tenbin.claims.model import Claim, ClaimKind, Denominator
from tenbin.corpus.record import Record, RecordKind
from tenbin.corpus.snapshot import CorpusSnapshot
from tenbin.corpus.values import Unstated, parse_stated
from tenbin.measures.base import (
    DistributionFigure,
    Figure,
    FigureGroup,
    freeze_counts,
    histogram,
)
from tenbin.measures.filtering import capture_population, requires_certainty, select

#: The slug this measure is cited by. Dotted rather than dashed so it can never
#: collide with a :func:`~tenbin.claims.refusals.slugify`-ed catalogue entry: the
#: registry is keyed by the measure *names* in ``docs/seam.md``, and a measure
#: sharing their spelling space would be one edit away from being refused by a
#: refusal filed about something else entirely.
CATEGORY_SLUG: Final[str] = "corpus.capture_category"

#: What the population is, in the words the denominator renders. The partition is
#: named in the description rather than left to the figure titles, because the
#: denominator is the whole capture population and a reader has to be able to check
#: the three figures against it.
CATEGORY_POPULATION: Final[str] = "capture records in this read, partitioned by record kind"

#: The title of the group, and the three figure titles. Each names the population
#: it covers, because a figure whose population is only in the module that built it
#: is a number a reader cannot check, and because the first two carry the reason they
#: are separate figures: their category is a constant of the record kind rather than
#: an observation about it.
CATEGORY_GROUP_TITLE: Final[str] = "What kind of thing gets captured"
LIFECYCLE_CATEGORY_TITLE: Final[str] = (
    "Category of captured lifecycle records, where the label is a constant of the record kind"
)
VERDICT_CATEGORY_TITLE: Final[str] = (
    "Category of review verdicts, where the label is a constant of the record kind"
)
OTHER_CATEGORY_TITLE: Final[str] = (
    "Category of the captures that are neither lifecycle records nor verdicts"
)

#: Exclusion keys, each a sentence a report renders verbatim. The first names an
#: absence; the second is a prefix, because the value it carries is the part a reader
#: needs -- a seventh category is a fact about a writer that has changed, and which
#: seventh category it is a fact worth having.
UNSTATED_CATEGORY_KEY: Final[str] = "no category stated (the record carries no category at all)"
UNRECOGNISED_CATEGORY_PREFIX: Final[str] = "category outside the six Kojutsu writes: "

#: The half of the negative space that is about the categoriser, written once
#: because it is the sentence a reader who has only skimmed the figure will not
#: otherwise reach. It is a claim about a model and a storage path, not about the
#: corpus: nothing in this program can re-categorise a record and compare, so the
#: figure is the only evidence there is about how well the labels describe the thing
#: they are on.
LABELS_ARE_UNREVIEWED: Final[str] = (
    "A model assigned every one of these categories and the store holds its output "
    "without a review step, so the distribution describes the categoriser's behaviour "
    "over this corpus and not the importance of what was captured. A rare category is "
    "not a rare kind of knowledge, a common one is not important, and the mix is not a "
    "statement about what the corpus is for. Nothing in this program can re-categorise "
    "a record and compare, so nothing here can say the labels were right -- only that "
    "they are what Kojutsu wrote."
)

#: The half of the negative space that is about the split, written once for the same
#: reason. Without it a reader holds three figures and a habit of adding them, and
#: the sum is a distribution whose movement has two causes.
SPLIT_BEFORE_AGGREGATING: Final[str] = (
    "Nor is the sum of the three figures a figure about anything. `system_event` is "
    "constant on lifecycle records and `design_decision` is constant on review verdicts, "
    "so one distribution over the whole corpus would move every time the mix of record "
    "kinds moved. The three are separate figures for that reason, and they partition the "
    "capture population between them: their counts and their exclusions add up to the "
    "denominator, and no one of them is a blend."
)


class Category(StrEnum):
    """The closed six-value vocabulary Kojutsu assigns at capture.

    The values are Kojutsu's own strings, copied rather than imported because
    this program reads only Tanseki and has no Kojutsu on its path: a reader changing
    one of these should go and read ``kojutsu.models.QuestionCategory`` first, and
    a sixth-to-seventh change is a change to the writer rather than to this
    vocabulary's spelling.

    ``StrEnum`` so a bucket serialises into a report and a JSON fixture with the same
    word, and so two fixtures cannot disagree about what ``"trade_off"`` means by
    spelling it differently in each. Declaration order is
    :func:`category_labels`, and it is not a ranking -- see the module docstring.
    """

    #: A choice between ways of doing something, with the choice stated.
    DESIGN_DECISION = "design_decision"
    #: What was given up to get something, stated as the giving-up.
    TRADE_OFF = "trade_off"
    #: How the subject domain works, as opposed to what was decided about it.
    DOMAIN_KNOWLEDGE = "domain_knowledge"
    #: A case the general rule does not cover.
    EDGE_CASE = "edge_case"
    #: What this rests on, which is also what breaks when it moves.
    DEPENDENCY = "dependency"
    #: Something that happened to the change rather than a reason for it. Constant on
    #: lifecycle records, which is one of the two constants the split exists for.
    SYSTEM_EVENT = "system_event"


#: The six labels in Kojutsu's declaration order, in one place a test can check
#: against the enum rather than against a rendering of it.
CATEGORY_LABELS: Final[tuple[str, ...]] = tuple(member.value for member in Category)


def category_labels() -> tuple[str, ...]:
    """Every category label, in Kojutsu's declaration order.

    A function rather than a constant, for the reason
    :func:`tenbin.measures.base.bucket_labels` is one: the ladder and the labels
    cannot drift apart. Iterating the enum is the definition in both places, and an
    enum is ordered by declaration, which is what makes the figure come out the same
    shape on every run.
    """
    return CATEGORY_LABELS


def category_claim(total: int) -> Claim:
    """The claim behind all three category figures, over the capture population.

    One claim for the group rather than three, because a figure carrying a claim
    whose denominator does not describe it is the defect this package is built to
    stop -- and the three figures are three views of one partition of one
    population, so a single denominator describes all of them.

    ``corpus`` granularity, and the reason is the one the trust profile gives: the
    corpus holds no team, area or repository dimension on this field, so the
    population genuinely *is* the whole corpus and a narrower description would be
    a promise about a field that does not exist rather than a tighter caveat. It is
    well above the floor in :data:`tenbin.claims.mechanism.MINIMUM_GRANULARITY`, so
    it is not a per-principal claim, and the split below is by record kind rather
    than by any organisational boundary the corpus does not have.
    """
    return Claim(
        slug=CATEGORY_SLUG,
        statement=(
            "The categories Kojutsu assigned at capture are distributed as the figures "
            "below show, over the capture records in this read partitioned by record kind. "
            "`category` is a closed six-value vocabulary, so every rung of it appears "
            "whether or not anything was captured into it."
        ),
        does_not_mean=LABELS_ARE_UNREVIEWED + " " + SPLIT_BEFORE_AGGREGATING,
        falsifier=(
            "A sampled re-categorisation in which a model asked to label the same records "
            "again disagrees with the stored labels in a proportion large enough to change "
            "the shape of the distribution. The corpus holds no review of any label, so "
            "that observation is the only thing that could check the categoriser, and it "
            "does not exist yet; a second falsifier is a record kind whose category is not "
            "the constant this measure's split assumes, which would mean the split is "
            "hiding a change in the writer rather than reporting one."
        ),
        denominator=Denominator(CATEGORY_POPULATION, max(1, total)),
        kind=ClaimKind.descriptive,
        granularity=Granularity.corpus,
    )


#: The enum's own values as a set, because ``value in TheEnum`` only answers the
#: question from Python 3.12. On 3.11 -- which ``requires-python`` still admits and
#: which mypy is configured to check against -- ``in`` on an enum raises TypeError
#: for anything that is not a member, so the unrecognised-value branch in the measure
#: below would be unreachable and it would crash on the one input it exists to catch.
#: A set of values is the same question and answers it the same way on both.
CATEGORY_VALUES: Final[frozenset[str]] = frozenset(m.value for m in Category)


@dataclass(frozen=True)
class CategoryMeasure:
    """What kind of thing was captured, in three figures over three record kinds.

    Stateless and total, like every measure here: ``compute`` is a pure function of
    the snapshot, so a test can build a read from a fixture and assert on the
    figures without a store, a clock or an ordering that is not in the data.

    The three figures are computed in one partition of the capture population rather
    than by three independent filters, and that is the difference between a
    partition and an overlap: the remainder is whatever the two named kinds did not
    take, so the three are exhaustive by construction and a record cannot appear in
    two of them or in none.
    """

    #: Every record counts, whatever certainty its classification carries. A record's
    #: category is a field on the document, not a consequence of how its kind was
    #: recovered, so requiring ``DETERMINED`` here would discard every legacy record
    #: over something this measure never reads. The split is by *kind*, which is
    #: read from whatever the record has, and the claim says the partition is by kind
    #: rather than by a certainty the corpus does not always state.
    requires_certainty = requires_certainty

    @property
    def slug(self) -> str:
        """The identifier this measure is cited by."""
        return CATEGORY_SLUG

    def claim(self, snapshot: CorpusSnapshot) -> Claim:
        """The claim, over the capture records this read enumerated."""
        return category_claim(len(capture_population(snapshot.records)))

    def compute(self, snapshot: CorpusSnapshot) -> Figure:
        """The category distribution, once per record kind, over one partition.

        Lifecycle records and verdicts first, because those are the two kinds whose
        category is constant, and the remainder last because it is whatever they did
        not take. The population arrives from :func:`capture_population` so a
        projected decision request is outside all three, which is decision 001
        rather than a decision this measure makes.
        """
        claim = self.claim(snapshot)
        population = capture_population(snapshot.records)
        lifecycle = select(
            population,
            certainty=self.requires_certainty,
            kinds=(RecordKind.PR_LIFECYCLE,),
        )
        verdicts = select(
            population,
            certainty=self.requires_certainty,
            kinds=(RecordKind.REVIEW_VERDICT,),
        )
        named = {record.doc_id for record in (*lifecycle, *verdicts)}
        remainder = tuple(record for record in population if record.doc_id not in named)
        return FigureGroup(
            claim=claim,
            snapshot=snapshot,
            title=CATEGORY_GROUP_TITLE,
            figures=(
                category_figure(claim, snapshot, lifecycle, title=LIFECYCLE_CATEGORY_TITLE),
                category_figure(claim, snapshot, verdicts, title=VERDICT_CATEGORY_TITLE),
                category_figure(claim, snapshot, remainder, title=OTHER_CATEGORY_TITLE),
            ),
        )


def category_figure(
    claim: Claim,
    snapshot: CorpusSnapshot,
    records: Sequence[Record],
    *,
    title: str,
) -> DistributionFigure:
    """The category distribution over one already-partitioned population.

    Two decisions live here rather than at the call sites, so that no caller can
    make one of them differently.

    **The absence and the seventh value are separated before counting.**
    :func:`tenbin.corpus.values.parse_stated` returns
    :class:`~tenbin.corpus.values.Unstated` for a record Kojutsu assigned no
    category to, and that record is counted in ``excluded`` rather than given a
    bucket -- a bucket named after the absence would put it back into the
    distribution it is not part of, which is the ``unknown``-model defect in a
    costume specific to this field. The one consequence of the shared reader is
    worth stating rather than discovering: Kojutsu does not write the literal
    ``"unknown"`` into ``category``, so the placeholder collision the wrapper exists
    for does not arise here, and a category of the literal ``"unknown"`` would be
    reported as an absence rather than as a seventh value. That is a deliberate
    reading of a field where the placeholder is not Kojutsu's, and it is pinned
    by a test so it cannot change by accident.

    **The population arrives as an argument, which the module docstring of
    :mod:`tenbin.measures.filtering` calls wrong, and the reason it is safe here is
    that the population is also in the title.** A figure handed to a report without
    its title is a number with no population, and that is the defect the split
    exists to prevent rather than something the split introduces.

    Every rung is present at zero, in the vocabulary's declaration order, and the
    order is deliberate: these six values are a closed set and not a scale, so the
    shape of the figure is the same on every run, and a reader comparing two runs
    needs the empty rungs to be visible to see they were empty.
    """
    counted: list[str] = []
    unstated = 0
    unrecognised: list[str] = []
    for record in records:
        stated = parse_stated(record.category)
        if isinstance(stated, Unstated):
            unstated += 1
        elif stated.value in CATEGORY_VALUES:
            counted.append(stated.value)
        else:
            unrecognised.append(stated.value)
    excluded: dict[str, int] = {}
    if unstated:
        excluded[UNSTATED_CATEGORY_KEY] = unstated
    # One key per distinct unexpected value, sorted so two reads of the same corpus
    # render the same figure. The count is bounded by the corpus rather than by a
    # constant, which is the price of naming the value instead of guessing at it.
    for value in sorted(set(unrecognised)):
        excluded[f"{UNRECOGNISED_CATEGORY_PREFIX}{value}"] = unrecognised.count(value)
    counts = histogram(counted)
    return DistributionFigure(
        claim=claim,
        snapshot=snapshot,
        title=title,
        values={label: counts.get(label, 0) for label in category_labels()},
        excluded=freeze_counts(excluded, "category.excluded"),
    )
