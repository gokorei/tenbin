"""The corpus's review structure, described and not scored.

Kojutsu writes ``independence`` and ``capture_source`` on every record
precisely so a reader can tell a capture from a claim, and re-validates the
anchors on the read path because the store is a separate service that never ran the
constructor check. This measure reports what those two fields say and stops there.
Independence says who was positioned to disagree; it does not say whether anything
is correct, and ``docs/corpus-inventory.md`` says so in Kojutsu's own words at
``models.py:116``.

**No ranking, on this axis or any other.** An ordinal vocabulary over a
self-selected corpus is an invitation to the one comparison this program refuses
everywhere else, and it is the more tempting one here because the axis *is*
ordered -- Kojutsu ranks its three levels, and the rank is right there in the
data. So :class:`Independence` carries its rank because a level this reader does
not know has to be reportable as unrecognised, and a reader is entitled to know
the order Kojutsu documents. Nothing in this module uses it to order anything:
the figures come out in count order, ``histogram`` knows nothing about the values
it counts, and a test asserts that the rendered text carries no language of
strength, quality or rank. A report that wanted to sort by it would have to sort
the mapping itself, which is the point.

**The capture figure splits before it aggregates, because ``asserted`` is a
constant on one kind of record.** A rationale is permanently ``asserted`` --
Kojutsu's own framing is that folding it in with the others would give it a
field claiming it can be evidence -- so a single capture-source percentage over the
whole corpus moves whenever the *mix of record kinds* moves, and a reader would see
a trust change where nothing about trust changed. The figure is therefore emitted
twice, over two populations, each titled so the difference cannot be skimmed past.
A composition change and a trust change produce identical numbers on a blended
figure and completely different numbers on a split one; that is the whole reason
for the split, and the reason it is not a parameter.

**A blank independence is an absence and is counted as one.** Kojutsu omits the
key rather than writing a placeholder, so a record that omits it is not in the
``self_certified`` bucket -- a level would be a claim about who was positioned to
disagree, and nobody made one. So the absence goes in ``excluded`` under a key that
names it, which is how a reader tells a bucket that did not occur from a bucket
that was filtered out. A level *outside* the three is the opposite case and is
counted as itself: the writer stated something, and it is not this reader's place
to decide it was not a level.

**The trust axis has two halves and this claim says so.** :data:`TRUST_HAS_TWO_AXES`
is the argument that the axis is not only what :class:`Independence` measures, and it
is rendered into the claim here rather than left to a module that reports the other
half. The reason is a misreading rather than a gap: a distribution of independence
levels read on its own invites the conclusion that independence is the whole of
trustworthiness, and the other half -- who was entitled to speak at all -- is not even
self-asserted, so a reader who never sees it has no way to know it exists.
:mod:`tenbin.measures.standing` reports it, and the claim here is what tells a reader
that the figure below is one of two axes rather than the axis.

**What this module notably does not do:** it does not aggregate the two axes into a
score, it does not weight a record by its level, and it does not compare two
records' independence. Every one of those is a small transformation, and each turns
a description of review structure into a claim that some records are better -- the
claim the whole package refuses, for want of anything to compare them with.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Final

from tenbin.claims.mechanism import Granularity
from tenbin.claims.model import Claim, ClaimKind, Denominator
from tenbin.corpus.provenance import CAPTURE_ASSERTED, CAPTURE_COLLECT, CAPTURE_WEBHOOK
from tenbin.corpus.record import Record
from tenbin.corpus.snapshot import CorpusSnapshot
from tenbin.corpus.values import Unstated, parse_stated
from tenbin.measures.base import DistributionFigure, Figure, FigureGroup, histogram
from tenbin.measures.filtering import capture_population, requires_certainty

#: The slug both trust figures are cited by. Dotted rather than dashed so it can
#: never collide with a :func:`~tenbin.claims.refusals.slugify`-ed catalogue
#: entry: the registry is keyed by the measure *names* in ``docs/seam.md``, and a
#: measure sharing their spelling space would be one edit away from being refused
#: by a refusal filed about something else entirely.
TRUST_PROFILE_SLUG: Final[str] = "corpus.trust_profile"

#: The key a record with no independence level at all is counted under. Not a
#: member of the distribution: an absent level is an absence, and a bucket named
#: after the absence would put it back into the distribution it is not part of.
UNSTATED_INDEPENDENCE: Final[str] = "no independence level stated"

#: The key for a record that states no capture source. Same reasoning as above.
UNSTATED_CAPTURE_SOURCE: Final[str] = "no capture source stated"

#: The key for a capture source outside the three-value vocabulary. Counted here
#: rather than folded into ``asserted``, because folding reports a broken writer as
#: a person typing something in -- the one reading the axis exists to prevent.
UNRECOGNISED_CAPTURE_SOURCE: Final[str] = "a capture source outside the three Kojutsu writes"

#: The two capture titles. Each names its population, because a title that said
#: "capture source" and nothing else would render two numbers a reader compares,
#: and on a blended figure that comparison is a composition change read as a trust
#: change.
RATIONALE_CAPTURE_TITLE: Final[str] = (
    "Capture source of stated rationales, where `asserted` is a constant and not an observation"
)
ENTRY_CAPTURE_TITLE: Final[str] = (
    "Capture source of captured entry records, where the mix is something that happened"
)

#: The independence title, saying what the axis describes rather than what it is
#: worth.
INDEPENDENCE_TITLE: Final[str] = (
    "Independence level stated on records, describing who could disagree"
)

#: The population both trust figures are read over. *Capture* records, and the
#: word is load-bearing: a projected decision request carries no independence level and no
#: capture source, so counting it here would move the ``unstated`` count and the entry mix
#: every time one was projected -- a change in the corpus's shape wearing the appearance of
#: a change in its review structure.
TRUST_POPULATION: Final[str] = "capture records in this read"

#: The second half of the trust axis, written once and rendered into the trust
#: claim and into the standing measure's own claim.
#:
#: It is here rather than in :mod:`tenbin.measures.standing` because the claim that
#: has to *name* the second axis is this one, and a sentence restated by two
#: modules is a sentence that will one day say two slightly different things. The
#: independence half is computed from fields the record asserts about itself, so
#: it is a claim about derivation that nothing outside the record can check; the
#: association half is a fact about position, because Kojutsu gates capture on
#: it, so it is not anybody's self-assertion at all. Two records at the same
#: independence level and different associations are not equally weighted, and the
#: corpus holds the fact that tells them apart.
TRUST_HAS_TWO_AXES: Final[str] = (
    "The trust axis has two halves, and this figure is one of them. Independence is "
    "derived from fields a record asserts about itself, so it is a claim about derivation "
    "that is self-asserted and unverifiable; author association is a fact about position, "
    "because Kojutsu gates capture on it, so it says who was entitled to speak at all and "
    "is nobody's self-assertion. A record asserted by an OWNER and one asserted by a "
    "COLLABORATOR at the same independence level are not equally weighted, and the corpus "
    "holds the fact that tells them apart. Reading an independence distribution on its own "
    "invites the misreading that independence is the whole of trustworthiness, and it is "
    "one of two axes: `corpus.author_standing` reports the other."
)


class CaptureSource(StrEnum):
    """How a record came to exist, in the three ways Kojutsu distinguishes.

    The values are the constants :mod:`tenbin.corpus.provenance` already pins, so
    the words have exactly one definition in this program: a fourth source has to
    be added there, deliberately, rather than appearing here as a member nobody
    else has agreed to. ``StrEnum`` so a bucket serialises into a report and a JSON
    fixture with the same word, and two fixtures cannot disagree about what
    ``"asserted"`` means by spelling it differently in each.
    """

    #: A signed provider delivery. Re-fetchable, and anchored to a delivery id.
    WEBHOOK = CAPTURE_WEBHOOK
    #: An authenticated read of the forge. Real, and anchored to a comment id.
    COLLECT = CAPTURE_COLLECT
    #: A person typed it in. Claims no capture, and is the only value that promises
    #: nothing to check against.
    ASSERTED = CAPTURE_ASSERTED


class Independence(StrEnum):
    """How far a captured record is from being a check on itself.

    The three levels and their meanings are Kojutsu's: the same account
    answering with the same model restates rather than checks; the same account
    with a different model is a second opinion from a second mind; a different
    account is a second party. None of them is a claim that the reasoning is
    correct, which is the whole reason this module does not rank anything by them.

    The order is written down in :data:`INDEPENDENCE_RANK` rather than read off the
    members, because an enum is unordered by construction and a rule that says
    "weakest to strongest" needs the order to be somewhere a reader can look.
    """

    SELF_CERTIFIED = "self_certified"
    MODEL_SEPARATED = "model_separated"
    INDEPENDENT = "independent"


#: The strength order, weakest to strongest, and the membership test for a level
#: this reader recognises. A mapping rather than an enum property so that
#: iterating the members cannot accidentally be the way an order is applied -- see
#: the module docstring on why nothing here uses it.
#: The enum's own values as a set, because ``value in TheEnum`` only answers the
#: question from Python 3.12. On 3.11 -- which ``requires-python`` still admits and
#: which mypy is configured to check against -- ``in`` on an enum raises TypeError
#: for anything that is not a member, so the unrecognised-value branch below would
#: be unreachable and the measure would crash on the one input it exists to catch.
#: A set of values is the same question and answers it the same way on both.
CAPTURE_SOURCE_VALUES: Final[frozenset[str]] = frozenset(m.value for m in CaptureSource)


INDEPENDENCE_RANK: Final[Mapping[str, int]] = MappingProxyType(
    {
        Independence.SELF_CERTIFIED.value: 0,
        Independence.MODEL_SEPARATED.value: 1,
        Independence.INDEPENDENT.value: 2,
    }
)


def trust_claim(total: int) -> Claim:
    """The claim all three trust figures are attached to, over the records examined.

    Over the whole read rather than over either sub-population, because the claim
    is about the corpus's review structure and the figures are three views of that
    one structure. A figure over a subset would need a claim of its own, and the
    split between rationales and entries is a fact about how ``asserted`` is used
    rather than a different question.
    """
    return Claim(
        slug=TRUST_PROFILE_SLUG,
        statement=(
            "The review structure of this corpus is distributed as the figures below describe: "
            "records differ in how far they are from being a check on themselves, and in "
            "whether they were captured at all or typed in. A second axis is reported "
            "alongside these under `corpus.author_standing`, and the two are different kinds "
            "of fact rather than two readings of the same one."
        ),
        does_not_mean=(
            "That anything here is true, and not that a record with a higher independence level "
            "is a better record. Independence describes who was positioned to disagree, not "
            "whether anything is correct -- it is a statement about how many parties were in a "
            "position to object, and a corpus full of `independent` records is a corpus in which "
            "people disagreed, not a corpus of correct answers. Nothing here ranks records or "
            "principals by level, and there is no score to rank them with. " + TRUST_HAS_TWO_AXES
        ),
        falsifier=(
            "Independence level uncorrelated with the number of distinct commenting accounts in "
            "any repository in the corpus. If a repository holding more `independent` records has "
            "no more distinct accounts commenting on it, the level is not describing who was in "
            "a position to disagree, and the axis means nothing. And the sharper form of the same "
            "check, which `corpus.independence_reason` computes: one level arising from every "
            "reason behind it in the same proportion, or a reason pairing with two different "
            "levels. Either would mean the level is a name for a single situation rather than a "
            "description of several, which is the failure this axis is most likely to have and "
            "the one a level on its own cannot show."
        ),
        denominator=Denominator(TRUST_POPULATION, max(1, total)),
        kind=ClaimKind.descriptive,
        granularity=Granularity.corpus,
    )


@dataclass(frozen=True)
class TrustProfileMeasure:
    """Two axes of the corpus's review structure, over the read it was given.

    Stateless and total. ``compute`` is a pure function of the snapshot: no clock,
    no store, and no ordering that depends on anything but the data, so a test can
    build a snapshot from a fixture and assert on the figures.
    """

    #: Every record counts, whatever certainty its classification carries. A
    #: record's independence level and capture source are fields on the document,
    #: not consequences of how its kind was recovered, so requiring ``DETERMINED``
    #: here would discard every legacy record over something this measure never
    #: reads.
    requires_certainty = requires_certainty

    #: Independence is a statement about how a record was captured and who was in a
    #: position to object, so it answers "can this corpus's evidence be believed"
    #: rather than "what happened to the work". That puts it beside the integrity
    #: checks in a reader asking whether to believe the figures below it.
    about_the_capture_system = True

    @property
    def slug(self) -> str:
        """The identifier this measure is cited by."""
        return TRUST_PROFILE_SLUG

    def claim(self, snapshot: CorpusSnapshot) -> Claim:
        """The claim, over the number of capture records this read enumerated."""
        return trust_claim(len(capture_population(snapshot.records)))

    def compute(self, snapshot: CorpusSnapshot) -> Figure:
        """The independence distribution and the two split capture distributions.

        Three figures under one claim, because they answer one question. Each
        carries the same claim and therefore the same denominator, and each says
        its own population in its title -- which together are what stops a
        composition change from reading as a trust change.
        """
        claim = self.claim(snapshot)
        records = capture_population(snapshot.records)
        return FigureGroup(
            claim=claim,
            snapshot=snapshot,
            title="The corpus's review structure",
            figures=(
                independence_figure(claim, snapshot, records),
                capture_figure(
                    claim,
                    snapshot,
                    [record for record in records if not record.is_rationale],
                    title=ENTRY_CAPTURE_TITLE,
                ),
                capture_figure(
                    claim,
                    snapshot,
                    [record for record in records if record.is_rationale],
                    title=RATIONALE_CAPTURE_TITLE,
                ),
            ),
        )


def independence_figure(
    claim: Claim,
    snapshot: CorpusSnapshot,
    records: Sequence[Record] | Iterable[Record],
) -> DistributionFigure:
    """Count the independence levels stated, and name the records that stated none.

    ``Unstated`` is separated before counting rather than given a bucket.
    Kojutsu omits the key when no level was derived, so a record in that state
    was never assigned one, and inventing a bucket for it is the ``unknown``-model
    defect in a second costume: a distribution in which "nobody said" sits beside
    the levels, looking like a group of records whose authors claimed
    ``self_certified``.
    """
    levels: list[str] = []
    unstated = 0
    for record in records:
        stated = parse_stated(record.independence)
        if isinstance(stated, Unstated):
            unstated += 1
        else:
            levels.append(stated.value)
    return DistributionFigure(
        claim=claim,
        snapshot=snapshot,
        title=INDEPENDENCE_TITLE,
        values=histogram(levels),
        excluded={UNSTATED_INDEPENDENCE: unstated} if unstated else {},
    )


def capture_figure(
    claim: Claim,
    snapshot: CorpusSnapshot,
    subset: Sequence[Record],
    *,
    title: str,
) -> DistributionFigure:
    """Count capture sources over one population, which the caller has already split.

    The population arrives as an argument, which is exactly what the module
    docstring says is wrong elsewhere, and the reason it is safe here is that the
    population is *also* in the title -- the one part of a figure a report cannot
    choose to leave out. A reader who is handed this figure and not its title has a
    number with no population, and that is the defect the split exists to prevent,
    not something the split introduces.
    """
    sources: list[str] = []
    unstated = 0
    unrecognised = 0
    for record in subset:
        stated = parse_stated(record.capture_source)
        if isinstance(stated, Unstated):
            unstated += 1
        elif stated.value in CAPTURE_SOURCE_VALUES:
            sources.append(stated.value)
        else:
            unrecognised += 1
    excluded: dict[str, int] = {}
    if unstated:
        excluded[UNSTATED_CAPTURE_SOURCE] = unstated
    if unrecognised:
        excluded[UNRECOGNISED_CAPTURE_SOURCE] = unrecognised
    return DistributionFigure(
        claim=claim,
        snapshot=snapshot,
        title=title,
        values=histogram(sources),
        excluded=excluded,
    )
