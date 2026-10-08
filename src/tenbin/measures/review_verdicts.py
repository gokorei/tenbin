"""Review verdicts, counted per reviewer, with the reviews that cannot be counted named.

Kojutsu's ticket ``RJNVJK1P`` stopped dropping ``review_id`` and ``record_kind``
at the storage boundary, and this measure exists because of it. Before that, a
review verdict was not identifiable in the stored record: the kind could only be
inferred from tags, and the answer case made that inference genuinely ambiguous,
so a rate over it was a rate over a reader's guess -- published under a reviewer's
name, which attributes the guess to a person. Now the writer says what kind each
record is, and a review carries the forge's own identifier for it.

**Filtered to determined verdicts, because the filter is the measure.** A record
whose kind rests on the tag projection is a *reading*, and a verdict rate is the
one figure here where a reading would be published beside a person's name. So the
population is records Kojutsu stated were review verdicts, and the rate a reader
sees is over statements rather than over this program's inferences.

**Two exclusions, both reportable.** A verdict with no ``review_id`` cannot be
paired with the inline comments of the same review, so counting it would inflate a
reviewer's verdict count with records whose relationship to it is unknown. An
inline comment is not a verdict at all -- it is a note about a line of code, and
Kojutsu's own review-state vocabulary deliberately leaves ``commented`` out
because feedback is not an approval. Both are counted in ``excluded`` under keys
that name the condition, because a verdict distribution that silently drops a third
of the review records is a distribution over the ones that parsed.

**What this does not mean, and the second half matters more.** Not that any verdict
was *correct* -- nothing in the corpus can show that, and the merge outcome of a
change is not a review's assessment of it. And not that a reviewer's rate is a
property of the reviewer: the corpus holds no record of who was assigned which
change, so a reviewer with a high approval rate may simply have been reviewing
changes that were going to be approved. The figure counts what reviewers recorded.

**What this module notably does not do:** it does not produce a score, a ranking, or
a threshold. There is no "approval rate above which a reviewer is good", because
every input to that judgement is either absent from the corpus or about a change
rather than about a person.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from tenbin.claims.mechanism import Granularity
from tenbin.claims.model import Claim, ClaimKind, Denominator
from tenbin.corpus.record import Certainty, Record, RecordKind
from tenbin.corpus.snapshot import CorpusSnapshot
from tenbin.measures.base import DistributionFigure, Figure, FigureGroup, histogram
from tenbin.measures.filtering import select

#: The slug this measure is cited by, and the ticket that unblocked it. Dotted so
#: it can never collide with a slugified refusal entry for the measure ``docs/seam.md``
#: once listed under the same words.
REVIEW_VERDICTS_SLUG: Final[str] = "corpus.review_verdicts"

#: The ticket named on the claim. ``unblocked_by`` and not only prose, because a
#: reader who wants to know why this measure exists and a reader who wants to know
#: what would make it different are asking two questions and the field answers the
#: second one in one place.
UNBLOCKED_BY: Final[str] = "RJNVJK1P"

#: Exclusion keys, each a sentence a report renders verbatim.
NO_REVIEW_ID_KEY: Final[str] = (
    "review verdict with no review_id, so it cannot be paired with its comments"
)
INLINE_COMMENT_KEY: Final[str] = "inline review comment, which records a note and not a verdict"
NO_REVIEWER_KEY: Final[str] = "review verdict with no commenting account on the record"

#: What the population is, in the words the denominator renders. "Determined" is in
#: the description rather than only in the code because the claim is rendered and
#: the code is not.
VERDICT_POPULATION: Final[str] = (
    "records Kojutsu stated were review verdicts and that carry a review id"
)

#: The separator between a reviewer's account and a verdict in a bucket key. A
#: character that cannot appear in a GitHub login, so a key can be split back into
#: its two parts by anybody reading it.
KEY_SEPARATOR: Final[str] = " — "


def review_verdict_claim(total: int) -> Claim:
    """The claim behind the verdict figures, over the verdicts that carry a review id."""
    return Claim(
        slug=REVIEW_VERDICTS_SLUG,
        statement=(
            "Review verdicts in this corpus are distributed as the figures below show, over the "
            "reviews Kojutsu stated were verdicts and identified with a review id."
        ),
        does_not_mean=(
            "That any verdict was correct, and not that a reviewer's rate is a property of the "
            "reviewer. Nothing in the corpus can show a verdict was right; the only outcome it "
            "records is whether the change merged later, which is a fact about the change rather "
            "than an assessment of the review. And a reviewer who approved more may simply have "
            "been assigned the changes that were going to be approved, because who was assigned "
            "which change is not in the corpus. The figures count what reviewers recorded."
        ),
        falsifier=(
            "Two records sharing a `review_id` where one is a verdict and the other is not, or "
            "two verdicts sharing a review id with different states. Either would mean the "
            "pairing this measure relies on cannot be relied on, and the verdict count per "
            "account would not be a count of reviews."
        ),
        denominator=Denominator(VERDICT_POPULATION, max(1, total)),
        kind=ClaimKind.descriptive,
        granularity=Granularity.team,
        unblocked_by=UNBLOCKED_BY,
    )


@dataclass(frozen=True)
class ReviewVerdictMeasure:
    """The verdicts, overall and per reviewer, over the reviews that can be identified."""

    #: Determined, and the reason is in the claim: a verdict rate is one of the two
    #: figures here where a kind recovered from a tag projection would be published
    #: beside a person's name. The other is the review pairing, which the
    #: ``review_id`` filter handles.
    requires_certainty = Certainty.DETERMINED

    @property
    def slug(self) -> str:
        """The identifier this measure is cited by."""
        return REVIEW_VERDICTS_SLUG

    def claim(self, snapshot: CorpusSnapshot) -> Claim:
        """The claim, over the identifiable verdicts in this read."""
        verdicts, _ = self._partition(snapshot.records)
        return review_verdict_claim(len(verdicts))

    def compute(self, snapshot: CorpusSnapshot) -> Figure:
        """The verdict mix overall, and the same verdicts per reviewer.

        Two figures under one claim, both over the same population, and so both
        carrying the same exclusions -- a report showing either one standalone is
        then correct on its own terms. The per-reviewer keys are ``account —
        verdict`` rather than accounts alone, because a reviewer's mix is the
        interesting quantity and a bare account bucket would make the reader sum
        the buckets to recover it.
        """
        verdicts, excluded = self._partition(snapshot.records)
        claim = self.claim(snapshot)
        return FigureGroup(
            claim=claim,
            snapshot=snapshot,
            title="Review verdicts",
            figures=(
                DistributionFigure(
                    claim=claim,
                    snapshot=snapshot,
                    title="Review verdicts by state",
                    values=histogram(_states(verdicts)),
                    excluded=excluded,
                ),
                DistributionFigure(
                    claim=claim,
                    snapshot=snapshot,
                    title="Review verdicts by reviewing account and state",
                    values=histogram(_by_reviewer(verdicts)),
                    excluded=excluded,
                ),
            ),
        )

    def _partition(
        self,
        records: Sequence[Record],
    ) -> tuple[tuple[Record, ...], dict[str, int]]:
        """The verdicts this measure counts, and the exclusions, from one pass.

        Returned together rather than as two calls because one of the two numbers
        is the claim's denominator: a figure built from a population the claim does
        not describe is the defect this package is built to prevent, and having both
        come out of one partition is what makes that impossible rather than merely
        unlikely.
        """
        verdicts = select(
            records,
            certainty=self.requires_certainty,
            kinds=(RecordKind.REVIEW_VERDICT,),
        )
        excluded: dict[str, int] = {}
        counted: list[Record] = []
        for record in verdicts:
            if not record.review_id:
                excluded[NO_REVIEW_ID_KEY] = excluded.get(NO_REVIEW_ID_KEY, 0) + 1
                continue
            counted.append(record)
        inline = len(
            select(
                records,
                certainty=self.requires_certainty,
                kinds=(RecordKind.INLINE_REVIEW_COMMENT,),
            )
        )
        if inline:
            excluded[INLINE_COMMENT_KEY] = inline
        unattributed = sum(1 for record in counted if not record.comment_author)
        if unattributed:
            excluded[NO_REVIEWER_KEY] = unattributed
        return tuple(counted), excluded


def _states(verdicts: Sequence[Record]) -> list[str]:
    """The verdict state of each record, as written.

    A verdict whose state tag names a state this reader does not know is counted
    as the literal tag rather than folded into a neighbour, and is *not* an
    exclusion: the writer stated a state, and a reader who cannot name it is the
    one with the gap. Kojutsu's vocabulary has two states and deliberately no
    ``commented``, so an unfamiliar tag is either a new state or a hand-edited
    document, and neither is ``approved``.
    """
    states: list[str] = []
    for record in verdicts:
        state = record.review_state
        states.append(state.value if state is not None else "(no verdict state on the record)")
    return states


def _by_reviewer(verdicts: Sequence[Record]) -> list[str]:
    """One bucket key per reviewing account and verdict, split on an unambiguous separator.

    A verdict with no ``comment_author`` is bucketed under an explicit key rather
    than under an account name, because it has nobody to attribute and a reader
    who saw it under a login would be reading a fact about a person that the
    record does not contain.
    """
    keys: list[str] = []
    for record in verdicts:
        state = record.review_state
        verdict = state.value if state is not None else "(no verdict state on the record)"
        account = record.comment_author or "(no reviewing account on the record)"
        keys.append(f"{account}{KEY_SEPARATOR}{verdict}")
    return keys
