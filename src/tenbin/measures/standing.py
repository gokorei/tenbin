"""Who was entitled to speak at all, which is not the same question as who checked.

``github_author_association`` is the half of the corpus's trust axis that is a fact
about position rather than a claim about derivation. Kojutsu gates capture on it --
an unauthorised association is refused at the write path -- so the field says who the
forge *considered entitled to make the record*, and it is not anybody's self-assertion
the way the fields behind :class:`~tenbin.measures.trust.Independence` are.

Reporting it beside independence is the point, and the reason is a misreading rather
than a gap. An independence distribution on its own invites the conclusion that
independence is the whole of trustworthiness: it is the strongest axis the corpus
has, it is ordinal, and it is the one Kojutsu documents in ranked prose. It is not
the whole. A record asserted by an OWNER and a record asserted by a COLLABORATOR at
the same independence level are not equally weighted, and the corpus holds the fact
that tells them apart. The argument is written once, in
:data:`tenbin.measures.trust.TRUST_HAS_TWO_AXES`, and rendered into both claims so
that the two figures cannot drift into saying two different things about one axis.

**No principal is identified by standing, and that is a refusal rather than an
omission.** A report of who is an OWNER and who is a MEMBER, attached to a login, is
per-principal output: a different product with different consent, retention and access
requirements, and the reason :data:`tenbin.claims.mechanism.MINIMUM_GRANULARITY` is
where it is. So the figure groups by association alone. No bucket key contains an
account, no title names one, and the claim says that the absence is deliberate --
because a figure that simply does not mention a person is indistinguishable from one
where nobody thought about it, and the difference is the whole point.

**An association outside the three is reported, not folded.** GitHub's own vocabulary
for this field is wider than Kojutsu's gate: it has ``CONTRIBUTOR`` and ``NONE``
among others, and a record carrying one of those is a fact about a corpus that has
changed rather than a record to be tidied into ``COLLABORATOR``. So it goes to
``excluded`` under a key naming the value, and a record with no association at all
goes under a key naming the absence -- two different facts, and the second is a
capture path that recorded nothing.

**What this module notably does not do:** it does not join association to
independence. The cross-tabulation of the two axes would be a small transformation of
two descriptions into one score, and it is exactly the "trust index" this package
refuses: there is no weight to put on an OWNER's assertion that any part of the
corpus can justify, and a number that combined the two would be the more legible of
the two figures and the less supported.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from tenbin.claims.mechanism import Granularity
from tenbin.claims.model import Claim, ClaimKind, Denominator
from tenbin.corpus.record import Record
from tenbin.corpus.snapshot import CorpusSnapshot
from tenbin.corpus.values import Unstated, parse_stated
from tenbin.measures.base import DistributionFigure, Figure, freeze_counts, histogram
from tenbin.measures.filtering import capture_population, requires_certainty
from tenbin.measures.trust import TRUST_HAS_TWO_AXES

#: The slug this measure is cited by, and the one the trust profile's claim names
#: when it says the axis has a second half. Dotted rather than dashed so it can never
#: collide with a :func:`~tenbin.claims.refusals.slugify`-ed catalogue entry: the
#: registry is keyed by the measure *names* in ``docs/seam.md``.
AUTHOR_STANDING_SLUG: Final[str] = "corpus.author_standing"

#: The title, and it says the thing the figure refuses to do. A title is the one part
#: of a figure a report cannot leave out, so the reason there is no login in it
#: belongs here rather than only in the claim.
ASSOCIATION_TITLE: Final[str] = (
    "Author association of captured records, with no account named: who was entitled "
    "to speak, counted without saying who"
)

#: What the population is, in the words the denominator renders. "Capture records" is
#: the word carrying the whole caveat: a projected decision request is not a capture
#: and has no author to have a standing.
STANDING_POPULATION: Final[str] = "capture records in this read"

#: Exclusion keys, each a sentence a report renders verbatim. The second is a prefix
#: because the value it carries is the fact -- an association Kojutsu does not gate
#: on is a fact about a writer that has changed, and which one it is is worth having.
UNSTATED_ASSOCIATION_KEY: Final[str] = (
    "no author association stated (the record carries none, so nobody's standing was recorded)"
)
UNRECOGNISED_ASSOCIATION_PREFIX: Final[str] = (
    "author association outside the three Kojutsu gates capture on: "
)

#: The half of the negative space about the two axes being different kinds of fact.
#: It is :data:`tenbin.measures.trust.TRUST_HAS_TWO_AXES` rather than a paraphrase
#: of it, and the reason is the whole argument of this module: a sentence restated by
#: two modules is a sentence that will one day say two slightly different things, and
#: a reader comparing the two figures would have no way to tell which had drifted.
AXES_ARE_DIFFERENT_FACTS: Final[str] = TRUST_HAS_TWO_AXES

#: The half about standing not identifying a person, which is a refusal and not a
#: property of the data. Named so the reason travels with the claim rather than being
#: a reader's inference from a figure that happens to lack names.
NO_PRINCIPAL_IS_NAMED: Final[str] = (
    "That this says anything about any account, and no principal is identified by it. "
    "The buckets are three words describing a position, the figure groups by position "
    "alone, and no login appears anywhere in it. A report of who is an OWNER and who is "
    "a MEMBER is per-principal output -- a different product, with different consent, "
    "retention and access requirements -- and a distribution of standing is the part of "
    "that which is not one."
)


class Association(StrEnum):
    """The three associations Kojutsu gates capture on, spelled as GitHub spells them.

    Upper case because that is the form the forge reports and the form Kojutsu
    stores, and a lowercased bucket would read as a normalising this program had no
    licence to do. The consequence is stated rather than fixed: a record spelling the
    same association in another case is *unrecognised*, and is counted in
    ``excluded`` under a key naming what it actually said, because a writer that has
    changed its casing is a fact about the writer and folding it into a neighbour
    would lose it.

    The three values are Kojutsu's gate rather than GitHub's vocabulary, which is
    wider -- ``CONTRIBUTOR`` and ``NONE`` are real values the forge reports and are
    the reason the unrecognised path above is a live case rather than a formality.
    """

    #: The account that owns the repository. Standing nobody else on the forge has.
    OWNER = "OWNER"
    #: An account with write access to the repository, by the forge's reckoning.
    MEMBER = "MEMBER"
    #: Somebody outside the repository who has been invited to contribute to it.
    COLLABORATOR = "COLLABORATOR"


def standing_claim(total: int) -> Claim:
    """The claim behind the association distribution, over the capture population.

    One figure, so one claim. It is its own measure with its own slug rather than a
    fourth member of the trust profile's figure group, for two reasons that point the
    same way: a slug is a report's key and two measures under one would be one
    identifier with two figures, and the populations are not quite the same -- the
    association is stated on captures, while the trust profile's group also holds
    figures over the rationale and entry sub-populations. Sharing the sentence that
    describes the two axes is what makes them one axis; sharing a slug would make
    them one measure, which they are not.

    ``corpus`` granularity, and the reason is the same one the trust profile gives:
    the corpus holds no team, area or repository dimension on this field, so the
    population genuinely is the whole corpus and anything narrower would be a promise
    about a field that does not exist. It is well above the floor in
    :data:`tenbin.claims.mechanism.MINIMUM_GRANULARITY`, and the claim is not about a
    principal in any case -- see :data:`NO_PRINCIPAL_IS_NAMED`.
    """
    return Claim(
        slug=AUTHOR_STANDING_SLUG,
        statement=(
            "The author associations Kojutsu accepted capture from are distributed as the "
            "figure below shows, over the capture records in this read. This is the second "
            "half of the trust axis `corpus.trust_profile` reports the first half of."
        ),
        does_not_mean=AXES_ARE_DIFFERENT_FACTS + " " + NO_PRINCIPAL_IS_NAMED,
        falsifier=(
            "A capture record whose association is outside the three Kojutsu gates on, in "
            "a corpus that holds no such record today. Either the gate has been widened, in "
            "which case this figure is a distribution over a different population than the "
            "one the trust profile read, or the record was written outside the gate, in "
            "which case a record exists in the corpus that the capture policy says cannot "
            "and the exclusion count above is the only place that shows it."
        ),
        denominator=Denominator(STANDING_POPULATION, max(1, total)),
        kind=ClaimKind.descriptive,
        granularity=Granularity.corpus,
    )


@dataclass(frozen=True)
class AuthorAssociationMeasure:
    """Who was entitled to speak, counted by standing and by nothing else.

    Stateless and total: ``compute`` is a pure function of the snapshot, so a test
    builds a read from a fixture and asserts on the figure.
    """

    #: Every record counts, whatever certainty its classification carries. An
    #: association is a field on the document and not a consequence of how its kind
    #: was recovered, so requiring ``DETERMINED`` would discard every legacy record
    #: over something this measure never reads. It matters here for the same reason
    #: it matters on the independence figure: this axis is about who was entitled to
    #: speak, and a record whose *kind* had to be guessed is not thereby a record
    #: whose author is unknown.
    requires_certainty = requires_certainty

    @property
    def slug(self) -> str:
        """The identifier this measure is cited by."""
        return AUTHOR_STANDING_SLUG

    def claim(self, snapshot: CorpusSnapshot) -> Claim:
        """The claim, over the capture records this read enumerated."""
        return standing_claim(len(capture_population(snapshot.records)))

    def compute(self, snapshot: CorpusSnapshot) -> Figure:
        """The distribution of association, with absences and outsiders named.

        A single :class:`~tenbin.measures.base.DistributionFigure` rather than a
        group: one axis, one vocabulary, one population, and a group here would be
        two ways of publishing the same three buckets.
        """
        return association_figure(
            self.claim(snapshot), snapshot, capture_population(snapshot.records)
        )


def association_figure(
    claim: Claim,
    snapshot: CorpusSnapshot,
    records: Sequence[Record],
) -> DistributionFigure:
    """Count the associations stated, and name the records that stated none.

    The absence is separated before counting rather than given a bucket, for the reason
    every axis in this package does it: a record with no association stated is not in
    the ``OWNER`` group, and a bucket named after the absence would put it in the same
    column as three positions with the same formatting. A record whose association is
    *outside* the three is the opposite case and is counted under a key carrying the
    value, for the reason :mod:`tenbin.measures.outcome` gives for a third merge
    outcome: the writer stated something, and it is not this reader's place to decide
    it was not a position.

    The counts come out in count order rather than in the order of the vocabulary, so
    the figure is a distribution and not a ranking of standing. Association is not
    ranked by the forge -- ``OWNER`` is a fact about repository access rather than a
    grade -- and nothing in this module treats it as one.
    """
    associations: list[str] = []
    unstated = 0
    unrecognised: list[str] = []
    for record in records:
        stated = parse_stated(record.github_author_association)
        if isinstance(stated, Unstated):
            unstated += 1
        elif stated.value in ASSOCIATION_VALUES:
            associations.append(stated.value)
        else:
            unrecognised.append(stated.value)
    excluded: dict[str, int] = {}
    if unstated:
        excluded[UNSTATED_ASSOCIATION_KEY] = unstated
    # One key per distinct unexpected value, sorted so two reads of the same corpus
    # render the same figure, and bounded by the corpus rather than by a constant.
    for value in sorted(set(unrecognised)):
        excluded[f"{UNRECOGNISED_ASSOCIATION_PREFIX}{value}"] = unrecognised.count(value)
    return DistributionFigure(
        claim=claim,
        snapshot=snapshot,
        title=ASSOCIATION_TITLE,
        values=histogram(associations),
        excluded=freeze_counts(excluded, "standing.excluded"),
    )


#: The enum's own values as a set, because ``value in TheEnum`` only answers the
#: question from Python 3.12. On 3.11 -- which ``requires-python`` still admits and
#: which mypy is configured to check against -- ``in`` on an enum raises TypeError
#: for anything that is not a member, so the unrecognised-value branch below would
#: be unreachable and the measure would crash on the one input it exists to catch.
#: A set of values is the same question and answers it the same way on both.
ASSOCIATION_VALUES: Final[frozenset[str]] = frozenset(m.value for m in Association)
