"""Which captures name a planning ticket, and what an absent key does not mean.

``jira`` is the only field that crosses from the forge into the planning system.
Kojutsu reads a ticket key off the *branch name* -- ``tanseki_mapping.py:136``
projects ``metadata["jira_ticket_key"]`` onto the ``jira`` key, and the key is
filled from a branch-name pattern -- so a capture naming ``ABC-1`` is a change
somebody opened against ``ABC-1``. Linking captures back to tickets is how a
reader connects *what was said* to *what was asked for*, and this is the one join
in the corpus where that sentence is true.

**The join is one-directional, lossy, and its absence means not supplied.** The
forge is read and the ticket is never read back: nothing here knows what ``ABC-1``
asked for, whether it is closed, or who is waiting on it, so a capture naming a
ticket says the change was opened against one and stops there. And because the
key comes from a branch-name pattern, a branch that did not follow the convention
carries no key -- which is a fact about the branch, not about the planning system.
A reader who misses that will count the missing keys as work nobody planned, so
the sentence is in the claim rather than in a docstring: **absence means not
supplied, not none.**

**A stated reason can never appear in the numerator, and that is not a gap in the
corpus.** Kojutsu's rationale writer emits ``repo`` and ``pr`` and does not
emit ``jira`` (``tanseki_mapping.py:224``), so every stated reason is a capture that
structurally cannot name a ticket. Those records are in the population and out of
the numerator, and the exclusion names that condition rather than folding them in
with the captures whose branch simply did not carry a key -- two different gaps,
and a reader fixing one of them does not fix the other.

**Both figures are over capture records, and the ladder answers "over what
tickets" without a bucket per ticket.** The share of captures naming a ticket is
one figure. The second buckets each ticket-naming capture by how many captures in
this read name the same ticket, which is how crowded each ticket is -- one rung for
a ticket exactly one capture mentions, up to an open-ended top rung. A
distribution keyed by ticket key would answer the same question with one bucket per
ticket, so a corpus with five thousand tickets would produce a five-thousand-rung
figure and the report would become a list. Every rung is materialised at zero for
the reason :func:`~tenbin.measures.base.duration_buckets` gives: a distribution
whose shape depends on the corpus is one every report has to guard against, and
``"exactly 1 capture: 0"`` is this figure's answer to the question it was written
for.

**A ticket nobody else referenced is counted, and the first rung is where it
lands.** A ticket named by a single capture is a fact about the corpus and not a
thin result, so it has a rung of its own rather than being folded into a bucket of
tickets that recurred. Dropping it would report a corpus whose tickets all
recurred, which is a corpus none of us have.

**What this module notably does not do:** it does not read Jira, and it does not
say a ticket was well specified. Tenbin reads one store and this key is the whole
of what it has about planning: the branching decisions a ticket constrained, the
scope nobody wrote down, and the fact that a change named a ticket and then
disagreed with it are all outside the seam. Nothing here is evidence that
capturing knowledge helped, and the figure is deliberately not a join to any
outcome measure -- a change naming a ticket is a change somebody had already been
asked something about, and comparing the two groups is decision 002's refusal.
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
from tenbin.measures.base import (
    DistributionFigure,
    Figure,
    FigureGroup,
    Rate,
    RateFigure,
    freeze_counts,
)
from tenbin.measures.filtering import (
    CAPTURE_POPULATION_WORD,
    capture_population,
    requires_certainty,
)

#: The slug this measure is cited by. Dotted rather than dashed so it can never
#: collide with a :func:`~tenbin.claims.refusals.slugify`-ed catalogue entry.
JIRA_LINKAGE_SLUG: Final[str] = "corpus.jira_linkage"

#: What the population is, in the words the denominator renders. Built from the
#: shared word rather than spelled out, because the size beside it changed the
#: moment projected decision requests left the capture population and a
#: description that no longer says what the count holds is a denominator
#: describing a population nobody measured.
JIRA_POPULATION: Final[str] = f"{CAPTURE_POPULATION_WORD} in this read"

#: The two exclusions, and they are two different gaps. The first is a capture
#: whose branch name carried no key, which is an absence of information rather
#: than an absence of tickets. The second is a stated reason, which the writer
#: cannot give a key to at all.
NO_TICKET_KEY: Final[str] = (
    "a capture whose branch name carried no ticket key, so no ticket was supplied rather than "
    "none existing"
)
RATIONALE_NO_TICKET_KEY: Final[str] = (
    "a stated reason, which Kojutsu's rationale writer never gives a ticket key: it writes "
    "`repo` and `pr` and not `jira`, so a stated reason can never name a ticket"
)

#: The ladder, and the reason for its top. The question is legible at one capture
#: and stops being one at five: a ticket six captures name is not a linkage
#: question, and a ladder with no top would give every heavily-referenced ticket a
#: bucket of its own so the shape of the figure would depend on the corpus. Matches
#: the custody measure's contributor ladder, for the same reason and with the same
#: top.
MAX_NAMED_TICKET_CAPTURES: Final[int] = 5
MANY_CAPTURE_LABEL: Final[str] = f"a ticket named by {MAX_NAMED_TICKET_CAPTURES} or more captures"

#: The three titles. Each names its own population, because the two figures answer
#: different questions over the same records and a reader who was handed one of
#: them without its title would be holding a number with no population.
LINKAGE_RATE_TITLE: Final[str] = "Captures naming a planning ticket, of every capture in this read"
TICKET_CROWDING_TITLE: Final[str] = (
    "Captures by how many captures name the same ticket, which is how crowded each ticket is"
)
LINKAGE_GROUP_TITLE: Final[str] = "Planning-ticket linkage, and what an absent key does not say"

#: The negative space, in the words the reader needs, and the sentence this whole
#: measure exists to make unavoidable. The one-directional join and the
#: not-supplied absence are separate sentences because they are separate
#: misreadings: one makes the figure a traceability claim, the other makes the
#: missing keys a finding about planning.
ABSENCE_IS_NOT_NONE: Final[str] = (
    "That a capture naming no ticket is a capture with no ticket behind it. A key is present "
    "when the branch name carried one, and Kojutsu reads it from a branch-name pattern rather "
    "than from the forge, so a branch that did not follow the convention says nothing at all "
    "about whether a ticket existed: absence here means not supplied, not none."
)
JOIN_IS_ONE_DIRECTIONAL: Final[str] = (
    "A measure of traceability, of planning, or of whether the work was specified. The ticket is "
    "never read back -- Tenbin reads one store, and this key is the whole of what it has about "
    "planning -- so a capture naming ABC-1 says the change was opened against ABC-1 and nothing "
    "about what ABC-1 asked for, whether it is closed, or who is waiting on it. A ticket named by "
    "many captures is not a ticket that took longer, and one named by a single capture is not a "
    "ticket nobody returned to."
)


def jira_linkage_claim(total: int) -> Claim:
    """The claim behind both linkage figures, over the capture records in this read."""
    return Claim(
        slug=JIRA_LINKAGE_SLUG,
        statement=(
            "Of the capture records in this read, the figures below report how many name a "
            "planning ticket, and how those captures spread over the tickets they name. A ticket "
            "is a piece of work somebody committed to before the change existed, and `jira` is the "
            "only field in this corpus that reaches from the forge into the planning system."
        ),
        does_not_mean=ABSENCE_IS_NOT_NONE + " " + JOIN_IS_ONE_DIRECTIONAL,
        falsifier=(
            "Two records about the same change carrying different `jira` keys, or one carrying a "
            "key where another does not. Every writer of a record about one change reads the same "
            "branch, so a disagreement would mean the key is a property of the branch a capture "
            "happened to be taken from rather than of the change itself -- and the figure would "
            "be counting branches rather than planning."
        ),
        denominator=Denominator(JIRA_POPULATION, max(1, total), size_noun=CAPTURE_POPULATION_WORD),
        kind=ClaimKind.descriptive,
        # The floor, and never below it. A figure about which tickets captures name
        # has no per-person reading available -- a ticket is not a person -- but the
        # population it aggregates over is captures, and a narrower population than a
        # team is a ranking of principals wearing a different name.
        granularity=Granularity.team,
    )


@dataclass(frozen=True)
class JiraLinkageMeasure:
    """How many captures name a ticket, and how crowded each ticket is.

    Two figures under one claim and therefore one denominator, because both are
    statements over the same set of capture records: the first is a share of them,
    the second is those same records bucketed by how many of the others name the
    same ticket. Nothing here is keyed by ticket, because a figure with one bucket
    per ticket is a list rather than a distribution.

    Stateless and total. ``compute`` is a pure function of the snapshot, so a test
    builds one from a fixture and asserts on the figures.
    """

    #: Every capture counts, whatever certainty its classification carries. A
    #: ticket key is a field on the document, not a consequence of how the record's
    #: kind was recovered, so requiring ``DETERMINED`` here would discard every
    #: legacy record over something this measure never reads.
    requires_certainty = requires_certainty

    @property
    def slug(self) -> str:
        """The identifier this measure is cited by."""
        return JIRA_LINKAGE_SLUG

    def claim(self, snapshot: CorpusSnapshot) -> Claim:
        """The claim, over the capture records this read enumerated."""
        return jira_linkage_claim(len(capture_population(snapshot.records)))

    def compute(self, snapshot: CorpusSnapshot) -> Figure:
        """The share naming a ticket, and the crowding of the tickets they name."""
        records = capture_population(snapshot.records)
        per_ticket, excluded = _captures_per_ticket(records)
        named = sum(per_ticket.values())
        claim = self.claim(snapshot)
        return FigureGroup(
            claim=claim,
            snapshot=snapshot,
            title=LINKAGE_GROUP_TITLE,
            figures=(
                RateFigure(
                    claim=claim,
                    snapshot=snapshot,
                    title=LINKAGE_RATE_TITLE,
                    rate=Rate(numerator=named, denominator=claim.denominator),
                ),
                DistributionFigure(
                    claim=claim,
                    snapshot=snapshot,
                    title=TICKET_CROWDING_TITLE,
                    values=_crowding_buckets(per_ticket),
                    # The same exclusions as the figure above, because both are over
                    # the same records: the numerator of the rate is exactly the sum of
                    # this figure's buckets, so the two cannot disagree about how many
                    # captures named a ticket.
                    excluded=excluded,
                ),
            ),
        )


def _captures_per_ticket(records: Sequence[Record]) -> tuple[Mapping[str, int], Mapping[str, int]]:
    """How many captures name each ticket, and the two conditions that named none.

    Two passes over one list and the order is the reason: the first counts, so
    that a capture naming a ticket shared by forty others lands in the open-ended
    rung rather than in the single-capture one; the second is the exclusion split,
    which needs to know whether the record was a stated reason at all.

    A stated reason is separated from a capture with no key because it is not a
    smaller version of it. Kojutsu's rationale writer does not emit ``jira``,
    so every stated reason in this corpus is a capture that could not have named a
    ticket however many people worked on it, and folding the two conditions into
    one key would report a gap in branch naming where there is a gap in the writer.
    """
    counts: dict[str, int] = {}
    no_key = 0
    rationale_no_key = 0
    for record in records:
        ticket = record.jira
        if not ticket:
            if record.is_rationale:
                rationale_no_key += 1
            else:
                no_key += 1
            continue
        counts[ticket] = counts.get(ticket, 0) + 1
    excluded: dict[str, int] = {}
    if no_key:
        excluded[NO_TICKET_KEY] = no_key
    if rationale_no_key:
        excluded[RATIONALE_NO_TICKET_KEY] = rationale_no_key
    return MappingProxyType(counts), freeze_counts(excluded, "tickets.excluded")


def _crowding_buckets(per_ticket: Mapping[str, int]) -> Mapping[str, int]:
    """Bucket each ticket-naming capture by how crowded its ticket is, rungs at zero.

    Weighted by the ticket's own count rather than counting tickets, because the
    figure's unit is the capture -- that is what the claim's denominator counts --
    and a bucket of *tickets* under a denominator of *captures* would not add up to
    it. The whole ladder is materialised for the reason
    :func:`~tenbin.measures.base.duration_buckets` gives.
    """
    labels = _crowding_labels()
    buckets: dict[str, int] = dict.fromkeys(labels, 0)
    for count in per_ticket.values():
        # Clamped before indexing rather than after, so the open-ended top rung is
        # counted rather than raising on a ticket forty captures name.
        size = min(count, MAX_NAMED_TICKET_CAPTURES)
        buckets[labels[size - 1]] += count
    return freeze_counts(buckets, "tickets.crowding_buckets")


def _crowding_labels() -> tuple[str, ...]:
    """The ladder's labels, from the single-capture rung upwards.

    A function rather than a constant for the reason
    :func:`~tenbin.measures.base.bucket_labels` is one: the rung count and the
    labels have to come from the same number. The top rung is the open one, so the
    named rungs stop one short of it -- a rung labelled ``"5 captures"`` beside
    another labelled ``"5 or more captures"`` would be a bucket nothing could ever
    land in.
    """
    labels = [
        f"a ticket named by exactly {size} capture{'' if size == 1 else 's'}"
        for size in range(1, MAX_NAMED_TICKET_CAPTURES)
    ]
    labels.append(MANY_CAPTURE_LABEL)
    return tuple(labels)
