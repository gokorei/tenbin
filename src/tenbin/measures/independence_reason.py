"""The reason behind each independence level, cross-tabulated rather than counted twice.

``independence`` is the strongest axis the corpus has and the one most likely to be
believed too readily. :mod:`tenbin.measures.trust` reports the levels and refuses to
rank them; this module reports the phrase Kojutsu wrote behind each one, and it is
a direct test of whether the level means anything.

**The question is whether one level is doing the work of several.** Kojutsu
derives the level from four accounts and models on the record, and it writes down the
phrase it used. If the same level arises from two different reasons in different
proportions, the level is collapsing distinct situations into one bucket, and a reader
holding only the level cannot see it. That is invisible in a distribution of levels
and obvious in a distribution of *pairings*, which is why this is one figure with
composite keys rather than two distributions sitting next to each other: two
independent distributions leave the pairing in the reader's head, and a reader holding
a level in one hand and a reason in the other is holding the question rather than the
answer.

**A key is ``level/reason``, and the composite is the point.** The two halves are
joined in one bucket name so that the figure can be read without the reader holding
the pairing in mind, which is the same argument
:mod:`tenbin.measures.review_verdicts` makes for its ``account — verdict`` keys. The
separator is a character neither half can contain: an independence level is one of
three underscore-joined words and a Kojutsu reason phrase is prose, so a key splits
back into its two parts by anybody reading it. The cost is accepted and stated: a
future phrase containing a slash would make one key ambiguous, and the price of that
is a rung a reader has to look at twice rather than a silent miscount.

**The four known reasons are documented and used to fill empty pairings, and are
never used to filter.** Kojutsu writes the phrase as a return value in
``compute_independence`` rather than as a member of an enum, so a fifth phrase is a
code change upstream rather than a vocabulary this reader gets to extend -- and a
reason outside the four is *information about a corpus that has changed*, which is
exactly what a filter would destroy. So :data:`KNOWN_INDEPENDENCE_REASONS` is read
here for two things and never for a third: it names the level each phrase is expected
at, which is what fills the four pairings at zero so a reader can see that a level
arose from one situation rather than several, and it is the fixture a test builds its
reasons from. An unrecognised reason is counted under a key carrying itself, exactly
as an unrecognised association is in :mod:`tenbin.measures.standing`.

**A record with a level and no reason is not a pairing.** The two absences are
counted separately, because they are two different gaps: a record with no level at all
was never assigned one, and a record with a level and no phrase behind it is a writer
that stored the conclusion and not the derivation. Neither is folded into a bucket, and
neither is silently dropped -- the excluded count is the only place either shows up.

**What this module notably does not do:** it does not turn a reason into a level, and
it does not read the reason to decide the level. The level is Kojutsu's field and
this program reports it; inferring the level from the phrase would be a reader
deciding which of two statements on one record is the real one, and the corpus layer
takes the same posture towards a tag and an id that disagree -- it reports both and
adjudicates neither. Where a phrase pairs with a level it is not expected at, the
figure says so under its own key, because that contradiction is the finding.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from tenbin.claims.mechanism import Granularity
from tenbin.claims.model import Claim, ClaimKind, Denominator
from tenbin.corpus.record import Record
from tenbin.corpus.snapshot import CorpusSnapshot
from tenbin.corpus.values import Unstated, parse_stated
from tenbin.measures.base import DistributionFigure, Figure, freeze_counts, histogram
from tenbin.measures.filtering import capture_population, requires_certainty
from tenbin.measures.trust import Independence

#: The slug this measure is cited by, and the one the trust profile's falsifier names
#: when it says the sharper version of the correlation check is computed here. Dotted
#: rather than dashed so it can never collide with a
#: :func:`~tenbin.claims.refusals.slugify`-ed catalogue entry.
INDEPENDENCE_REASON_SLUG: Final[str] = "corpus.independence_reason"

#: The separator between a level and the phrase behind it. A character neither half
#: can contain, so the key splits back into its two parts by anybody reading it. See
#: the module docstring for the one case that would make it ambiguous and for the
#: reason the cost is accepted.
PAIRING_SEPARATOR: Final[str] = "/"

#: The four phrases Kojutsu writes, each with the level it arises at. Read from
#: ``kojutsu.models.compute_independence``, which returns them as literals beside
#: the level, so this mapping is a statement about one function rather than about a
#: vocabulary somewhere else -- and the ``self_certified`` level has *two* of them,
#: which is the reason a mapping and not a list.
#:
#: **Nothing filters on this.** The value of holding it is that the four pairings are
#: materialised at zero, so a reader can see a level that arose from one situation
#: rather than from several without having to notice an absence; and the value of
#: publishing it is that a test can build its reasons from the same list this module
#: reads. A reason outside these four is counted under a key carrying itself, because
#: Kojutsu writes the phrase as a return value rather than as a member of an enum
#: and a fifth one is a fact about a corpus that has changed.
KNOWN_INDEPENDENCE_REASONS: Final[Mapping[str, str]] = MappingProxyType(
    {
        # A different posting account, whatever the models were: two parties.
        "different posting accounts": Independence.INDEPENDENT.value,
        # The same account, a different model: a second mind rather than a second party.
        "same account, different models": Independence.MODEL_SEPARATED.value,
        # The same account, the same model: the record restates what produced it.
        "same account, same model": Independence.SELF_CERTIFIED.value,
        # The same account with a model unstated on one side. A gap, reported as the
        # weakest level rather than assumed to be a match.
        "same account; model not stated by both parties": Independence.SELF_CERTIFIED.value,
    }
)

#: The title, and it says what a rung is: a pairing, not a level and not a reason.
REASON_PAIRING_TITLE: Final[str] = (
    "Independence level paired with the reason behind it, as `level/reason`"
)

#: What the population is, in the words the denominator renders. The whole capture
#: population rather than the records that state both fields, so the excluded count is
#: inside the population the claim describes instead of beside it.
REASON_POPULATION: Final[str] = "capture records in this read"

#: The two exclusions, and they are two different gaps. The first is a record nobody
#: ever assigned a level to; the second is a record whose level arrived without the
#: phrase behind it, which is a writer that stored the conclusion and not the
#: derivation. Neither is folded into a bucket.
UNSTATED_LEVEL_KEY: Final[str] = (
    "no independence level stated, so there is no level to pair a reason with"
)
UNSTATED_REASON_KEY: Final[str] = "an independence level with no reason written behind it"


def reason_claim(total: int) -> Claim:
    """The claim behind the level-by-reason pairing, over the capture population.

    ``corpus`` granularity, and for the reason the trust profile gives: the corpus
    holds no team, area or repository dimension on either of these fields, so the
    population genuinely is the whole corpus and anything narrower would promise a
    field that does not exist. It is well above the floor in
    :data:`tenbin.claims.mechanism.MINIMUM_GRANULARITY`, and the claim is about the
    derivation of a level rather than about any record or principal.
    """
    return Claim(
        slug=INDEPENDENCE_REASON_SLUG,
        statement=(
            "The reasons Kojutsu wrote behind each independence level are distributed "
            "against those levels as the figure below shows, one bucket per pairing. A level "
            "that arises from a single reason is a name for one situation; a level that "
            "arises from two is a level, and a reader can see which it is from the figure."
        ),
        does_not_mean=(
            "That a reason is a derivation anybody checked, or that a level is a grade. The "
            "phrase is Kojutsu's own account of how it reached the level, written at the "
            "same moment from the same four fields, and reading it is reading a writer's "
            "explanation rather than an audit of one. Nothing here ranks the pairings, and a "
            "level is not better for having a reason: a record whose level came from two "
            "different accounts is not more trustworthy than one whose level came from a "
            "different model, only differently derived. Nor is the pairing a second "
            "independence figure to be read alongside the first -- it is a check on whether "
            "the level above means what it says."
        ),
        falsifier=(
            "One independence level arising from every reason behind it in the same "
            "proportion in every repository and every month, or a reason that pairs with two "
            "different levels. Either would mean the level is either a name for a single "
            "situation -- in which case a corpus of `self_certified` records is a corpus in "
            "which nobody was ever in a position to disagree, which is a different and much "
            "stronger claim than the level makes -- or the phrase is not describing the level "
            "at all, in which case the axis above it is one thing wearing three names."
        ),
        denominator=Denominator(REASON_POPULATION, max(1, total)),
        kind=ClaimKind.descriptive,
        granularity=Granularity.corpus,
    )


def pairing_key(level: str, reason: str) -> str:
    """The bucket key for one pairing: the level, the separator, the reason.

    A function rather than an f-string at each call site, so that the *form* of a key
    is defined once. The form is the interface a reader parses by eye, and a reader who
    has to work out the convention from one example and then apply it to a rung written
    by somebody else is doing the work this figure exists to do for them.
    """
    return f"{level}{PAIRING_SEPARATOR}{reason}"


def known_pairings() -> tuple[str, ...]:
    """Every pairing Kojutsu's four phrases produce, in a fixed order.

    Materialised at zero in the figure so that an empty pairing is *visible* rather
    than absent: the reader's question is whether one level came from two situations,
    and the answer is a thing they can read off a rung showing zero next to a rung
    showing four. An absence in a histogram is indistinguishable from a bucket that
    was filtered out, and :class:`~tenbin.measures.base.DistributionFigure` exists
    precisely because that ambiguity is the defect this package is built to stop.

    The order is the declaration order of :data:`KNOWN_INDEPENDENCE_REASONS`, which is
    a mapping, so it is stated here rather than left to be implied. A frozen order
    means two reads of the same corpus render the same figure.
    """
    return tuple(pairing_key(level, reason) for reason, level in KNOWN_INDEPENDENCE_REASONS.items())


@dataclass(frozen=True)
class IndependenceReasonMeasure:
    """The reasons behind the levels, one bucket per pairing, over the capture records.

    Stateless and total: ``compute`` is a pure function of the snapshot, so a test
    builds a read from a fixture and asserts on the figure.
    """

    #: Every record counts, whatever certainty its classification carries. Both fields
    #: are on the document rather than consequences of how its kind was recovered, so
    #: requiring ``DETERMINED`` would discard every legacy record over something this
    #: measure never reads.
    requires_certainty = requires_certainty

    @property
    def slug(self) -> str:
        """The identifier this measure is cited by."""
        return INDEPENDENCE_REASON_SLUG

    def claim(self, snapshot: CorpusSnapshot) -> Claim:
        """The claim, over the capture records this read enumerated."""
        return reason_claim(len(capture_population(snapshot.records)))

    def compute(self, snapshot: CorpusSnapshot) -> Figure:
        """The level-by-reason distribution, with both absences counted and named."""
        return reason_pairing_figure(
            self.claim(snapshot), snapshot, capture_population(snapshot.records)
        )


def reason_pairing_figure(
    claim: Claim,
    snapshot: CorpusSnapshot,
    records: Sequence[Record],
) -> DistributionFigure:
    """Cross-tabulate each stated level against the reason stated behind it.

    A record contributes to a bucket only when it states *both* halves, because a
    pairing is what this figure counts and a level on its own is the figure
    :mod:`tenbin.measures.trust` already publishes. The two gaps are counted
    separately so that neither is silent: no level at all is a record nobody assigned
    one to, and a level with no phrase is a record whose derivation was stored as its
    conclusion.

    A reason outside :data:`KNOWN_INDEPENDENCE_REASONS` is counted under a key
    carrying itself rather than being folded into a neighbour, and a reason that pairs
    with a level it is not expected at gets its own bucket for the same reason: the
    contradiction between two statements on one record is the finding, and a reader
    who could not see it would be reading a distribution that quietly smoothed it over.
    """
    pairings: list[str] = []
    unstated_levels = 0
    unstated_reasons = 0
    for record in records:
        level = parse_stated(record.independence)
        if isinstance(level, Unstated):
            unstated_levels += 1
            continue
        reason = parse_stated(record.independence_reason)
        if isinstance(reason, Unstated):
            unstated_reasons += 1
            continue
        pairings.append(pairing_key(level.value, reason.value))
    counts = histogram(pairings)
    known = known_pairings()
    excluded: dict[str, int] = {}
    if unstated_levels:
        excluded[UNSTATED_LEVEL_KEY] = unstated_levels
    if unstated_reasons:
        excluded[UNSTATED_REASON_KEY] = unstated_reasons
    return DistributionFigure(
        claim=claim,
        snapshot=snapshot,
        title=REASON_PAIRING_TITLE,
        # The four known pairings first, in their fixed order, so an empty one is
        # visible; then whatever else arrived, in count order from ``histogram``. The
        # frozen copy is what lets a report print the mapping without a copy of it.
        values=freeze_counts(
            {key: counts.get(key, 0) for key in known}
            | {key: count for key, count in counts.items() if key not in known},
            "independence_reason.values",
        ),
        excluded=freeze_counts(excluded, "independence_reason.excluded"),
    )
