"""How captured changes ended, over the changes that were captured.

Kojutsu's ticket ``QEVSTMYW`` stopped folding the merge fact into a dedupe hash
and discarding it. Before it, a merged change and an abandoned change were
indistinguishable in the record: both reported ``state == "closed"``, and the only
difference lived in a derivation a consumer would have to know and reimplement.
This is the measure that derivation was never built for, and it is the closest this
program comes to an outcome.

**The denominator is captured changes, and it says so, because the census is
missing.** A change that produced no capture is not in the corpus at all -- Kojutsu
writes only on success, so "examined and unremarkable" and "never seen" are the same
observation from outside. So the merge rate below is a rate over the changes
something wrote about, and a reader who quotes it as "the merge rate" has quoted a
statement about a selected population while believing they have quoted one about
development. ``EB6FE5ZP`` is the census that would settle it and it has not landed,
so the claim names the gap rather than rounding the population silently.

**An unrecognised outcome is not a merge and not an abandonment.** ``pr_outcome`` is
the two strings GitHub can be read out of a close for, and a third is a writer that
has changed or a hand-edited document. Either way it is counted in ``excluded``
under a key that names the value it actually held, and it is kept out of both the
numerator and the denominator. Folding it into ``closed_unmerged`` would report a
broken writer as a rejected change; folding it into ``merged`` would be worse and
harder to notice.

**Whether GitHub reported a merge is not whether the change was any good.** That is
Kojutsu's own framing of the field and the claim repeats it, because a merge rate
read next to a review-verdict rate invites a reader to compose them into a claim
about quality that neither supports and that this corpus cannot make at all.

**What this module notably does not do:** it does not read the *content* of a
change, and it does not report an outcome for a change whose outcome GitHub has not
reported. A change still open has no outcome, and asserting ``closed_unmerged``
about one would be asserting something the forge never said; those records are
counted as changes with no outcome stated rather than being filed on either side.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from tenbin.claims.mechanism import Granularity
from tenbin.claims.model import Claim, ClaimKind, Denominator
from tenbin.corpus.provenance import CAPTURED_SOURCES
from tenbin.corpus.record import Record
from tenbin.corpus.snapshot import CorpusSnapshot
from tenbin.measures.base import (
    DistributionFigure,
    Figure,
    FigureGroup,
    Rate,
    RateFigure,
    freeze_counts,
    histogram,
)
from tenbin.measures.filtering import requires_certainty

#: The two values Kojutsu writes, read from GitHub's two ways of saying a change
#: is closed. Spelled here rather than imported because the store's vocabulary is
#: Kojutsu's and a reader changing these should be reading Kojutsu's source
#: -- which is where the values come from, and which documents that anything else
#: is not a third outcome but a writer that has invented one.
PR_OUTCOME_MERGED: Final[str] = "merged"
PR_OUTCOME_CLOSED_UNMERGED: Final[str] = "closed_unmerged"

#: The slug this measure is cited by, and the tickets on the claim. ``QEVSTMYW``
#: delivered the merge fact; ``EB6FE5ZP`` is the census that would let the
#: population be every change rather than the captured ones. Both are named because
#: the second is the one that decides whether the first was enough.
OUTCOME_SLUG: Final[str] = "corpus.change_outcome"

#: The two exclusion keys. The first names a population this program cannot see; the
#: second names a vocabulary it refuses to extend on a writer's behalf.
NO_OUTCOME_KEY: Final[str] = (
    "captured change with no outcome stated (still open, or the payload carried none)"
)
UNRECOGNISED_OUTCOME_PREFIX: Final[str] = "pr_outcome outside GitHub's two values: "

#: What the two populations are, in the words their denominators render. Both say
#: *captured*, because that is the word carrying the whole caveat and a reader who
## misses it in the claim misses it in the one place it was repeated.
OUTCOME_POPULATION: Final[str] = "captured changes whose close GitHub reported an outcome for"
SHARE_POPULATION: Final[str] = "captured changes whose close GitHub reported a recognised outcome"

#: The gap the census would close, in the words a reader needs before the number.
CENSIS_MISSING: Final[str] = (
    "The denominator is the changes that produced a capture, not the changes that were made. A "
    "change Kojutsu never wrote about is indistinguishable, from outside, from one it "
    "examined and found unremarkable, and no backfill path exists. The census of uncaptured "
    "changes (`EB6FE5ZP`) is the fact that would decide whether this rate describes development "
    "or only the part of development that produced a record."
)

#: Kojutsu's framing of the field, because a merge rate sits next to a review rate
#: in a report and the two invite a composition the corpus cannot support.
OUTCOME_IS_NOT_QUALITY: Final[str] = (
    "A merge is what GitHub reported happening to a change, not a judgement that the change was "
    "good. It does not mean the change was correct, that it was reviewed properly, or that it "
    "would survive contact with production -- none of which this corpus observes. It is also not "
    "a performance measure of whoever opened it: the change that was written and the change that "
    "was abandoned were not assigned at random, and nothing here says which of them anybody was "
    "given."
)


def outcome_claim(total: int) -> Claim:
    """The claim behind both outcome figures, over the recognised captured outcomes."""
    return Claim(
        slug=OUTCOME_SLUG,
        statement=(
            "The captured changes in this corpus ended as the figures below show, over the changes "
            "GitHub reported the outcome of."
        ),
        does_not_mean=CENSIS_MISSING
        + " "
        + OUTCOME_IS_NOT_QUALITY
        + " It also does not cover changes that are still open: an open change has no outcome, and "
        "counting it as anything would be asserting something the forge did not report.",
        falsifier=(
            "A census of changes that produced no capture (`EB6FE5ZP`) showing a materially "
            "different merge rate. A census finding a substantially lower merge rate among "
            "uncaptured changes would mean this figure is a rate over the changes that made a "
            "record, which is a different and much less flattering claim."
        ),
        denominator=Denominator(SHARE_POPULATION, max(1, total)),
        kind=ClaimKind.descriptive,
        granularity=Granularity.repository,
        unblocked_by="QEVSTMYW, EB6FE5ZP",
    )


@dataclass(frozen=True)
class ChangeOutcomeMeasure:
    """How captured changes ended, and the merged share of those GitHub reported on.

    Two figures under one claim. The distribution is over every captured change
    that states an outcome -- including one whose outcome this reader does not
    recognise, which appears in ``excluded`` under a key naming the value it held.
    The rate is over the *recognised* subset, which is a strictly smaller
    population than the distribution's, and the difference is exactly what the
    unrecognised bucket holds. So the rate's denominator description says
    "recognised" and its title says the same, and a reader who wants the wider
    figure has the distribution beside it.
    """

    requires_certainty = requires_certainty

    @property
    def slug(self) -> str:
        """The identifier this measure is cited by."""
        return OUTCOME_SLUG

    def claim(self, snapshot: CorpusSnapshot) -> Claim:
        """The claim, over the recognised outcomes in this read."""
        return outcome_claim(_census(snapshot.records).recognised)

    def compute(self, snapshot: CorpusSnapshot) -> Figure:
        """The distribution of outcomes, and the merged share of the recognised ones."""
        census = _census(snapshot.records)
        claim = self.claim(snapshot)
        return FigureGroup(
            claim=claim,
            snapshot=snapshot,
            title="How captured changes ended",
            figures=(
                DistributionFigure(
                    claim=claim,
                    snapshot=snapshot,
                    title="How captured changes ended, as GitHub reported it",
                    values=census.values,
                    excluded=census.excluded,
                ),
                RateFigure(
                    claim=claim,
                    snapshot=snapshot,
                    title="Merged share of captured changes whose outcome GitHub reported",
                    rate=Rate(
                        numerator=census.merged,
                        denominator=Denominator(SHARE_POPULATION, max(1, census.recognised)),
                    ),
                ),
            ),
        )


@dataclass(frozen=True)
class _OutcomeCensus:
    """What GitHub reported about how the captured changes ended.

    Two populations in one object because the two figures need different ones and
    they must come from a single pass. The distribution is over every captured
    change that states an outcome; the rate is over the recognised subset of that.
    Keeping the counts together is what makes the subset visible instead of
    implicit -- a rate whose numerator and denominator are computed by two walks
    over the same records is a rate that can disagree with itself.
    """

    values: Mapping[str, int]
    excluded: Mapping[str, int]
    merged: int
    recognised: int


def _census(records: Sequence[Record]) -> _OutcomeCensus:
    """Walk the captured changes once and describe how they ended.

    A *change* is a ``(repo, pr)`` pair and the outcome is taken from the first
    record of that change that states one. A change with three lifecycle records
    saying three different things is a contradiction the store cannot explain, and
    this measure reports the first without adjudicating -- the same posture the
    corpus layer takes towards a tag and an id that disagree, and the same reason:
    this program reads what was written and does not decide which writer was right.

    An unrecognised value is counted in ``excluded`` under a key carrying the value
    itself, so a writer that has invented a third outcome is visible with the
    outcome it invented rather than as a number that has to be guessed at. One key
    per distinct value is the price of that visibility, and it is bounded by the
    corpus rather than by a constant.
    """
    by_change: dict[tuple[str, int], str] = {}
    captured_changes: set[tuple[str, int]] = set()
    for record in records:
        if not _is_captured(record):
            continue
        key = record.pr_key
        if key is None:
            continue
        captured_changes.add(key)
        if record.pr_outcome is not None:
            by_change.setdefault(key, record.pr_outcome)
    outcomes = list(by_change.values())
    recognised_values = [
        outcome
        for outcome in outcomes
        if outcome in (PR_OUTCOME_MERGED, PR_OUTCOME_CLOSED_UNMERGED)
    ]
    unrecognised = [outcome for outcome in outcomes if outcome not in recognised_values]
    excluded: dict[str, int] = {
        f"{UNRECOGNISED_OUTCOME_PREFIX}{outcome}": unrecognised.count(outcome)
        for outcome in sorted(set(unrecognised))
    }
    # Counted per *change*, so a change with five captured records and no outcome is
    # one absence in the figure rather than five. The alternative makes a busy
    # repository look like a repository whose changes are unaccounted for.
    with_no_outcome = len(captured_changes - by_change.keys())
    if with_no_outcome:
        excluded[NO_OUTCOME_KEY] = with_no_outcome
    return _OutcomeCensus(
        values=histogram(recognised_values),
        excluded=freeze_counts(excluded, "outcome.excluded"),
        merged=recognised_values.count(PR_OUTCOME_MERGED),
        recognised=len(recognised_values),
    )


def _is_captured(record: Record) -> bool:
    """Whether this record claims a real capture, per Kojutsu's own two sources.

    ``asserted`` is not one: a person typed it in, so there is nothing behind it and
    nothing that can have been merged or abandoned. A record with no
    ``capture_source`` at all is not one either, and for a sharper reason -- silence
    about provenance is not a claim of assertion, and a distribution of outcomes
    over records that may have been typed in by hand describes something this
    program cannot name.
    """
    return record.capture_source is not None and record.capture_source in CAPTURED_SOURCES
