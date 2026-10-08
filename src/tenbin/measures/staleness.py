"""How long since anything was captured against the files a stated reason is about.

Two facts had to land before this was measurable, and they landed in that order.
``QEVSTMYW`` put ``pr_merged_at`` on the record, which is the clock a rationale has to
be dated against -- a change still open has no end, and dating a reason against a
repository-wide average would be a measure wearing a name it had not earned. Then
``MWF7Z1EN`` wrote the file anchor, which is the only thing that says *which* code a
reason is about. The refusal that used to sit in the catalogue quoted the first ticket
and named the second, and the second is why this module exists.

**A rationale is joined to its change by ``pr_key``, and a half key joins to nothing.**
``(repo, pr)`` or nothing: a record with a pull-request number and no repository must
not join to every repository holding that number, because that is a fabricated
correspondence and the whole point of the key is that it is not. File paths are
repository-relative for the same reason -- ``src/queue.py`` exists once per repository,
so both sides of the lookup carry the area.

**The gap runs from the merge to the newest capture naming the file, and not from the
declaration.** A rationale is usually stated while the change is still open, so
measuring from ``declared_at`` would fold the review period into a measure of code
movement and report a week of staleness for a change that merged an hour ago. The merge
is the first instant at which the code exists in the form the reason is about.

**An absent interval is not a zero interval.** A file the reason names that nothing has
been captured against since the merge is *not* the freshest bucket: it is a file this
corpus cannot say anything about, and writing it as zero would make a rationale nobody
revisited look like the most recently confirmed thing in the repository. Those pairs
are counted in ``excluded`` under a key naming the condition, separately from a
rationale with no files and from one whose change never merged, because those are three
different gaps and a reader fixing one of them needs to know which they have.

**A bounded file list is an exclusion here and only a caveat in the custody measure,
and the asymmetry is the point.** Kojutsu caps ``files`` at fifty and writes
``"N more files not listed"`` in place of the rest. For the file-set count in
:mod:`tenbin.measures.custody` a bounded list merely makes the population smaller,
which an undercount already says. Here it is worse than smaller: the unlisted files can
only have *more* movement behind them than the listed ones, so including a partial list
biases the distribution towards "not stale" and the bias runs one way. So a truncated
rationale is set aside and counted.

**What this module notably does not do:** it does not say a stale rationale is a wrong
one. A reason that has not been revisited in a year may be one nobody needed to revisit,
and a reason that was revisited may have been confirmed rather than corrected -- neither
of which is in the corpus. Nor does it generalise past the file: a rationale about one
file says nothing about the *other* files the same change touched, and nothing at all
about a file nobody mentioned.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Final

from tenbin.claims.mechanism import Granularity
from tenbin.claims.model import Claim, ClaimKind, Denominator
from tenbin.corpus.record import Record
from tenbin.corpus.snapshot import CorpusSnapshot
from tenbin.measures.base import DistributionFigure, Figure, duration_buckets, freeze_counts
from tenbin.measures.filtering import named_repository, requires_certainty

#: The slug this measure is cited by, and both tickets it waits on. ``MWF7Z1EN``
#: delivered the file anchor and is the one the retired refusal named; ``QEVSTMYW``
#: delivered the merge fact and was already in the store when that refusal was
#: written, which is why quoting it there would have sent a reader to a ticket whose
#: work was done.
STALENESS_SLUG: Final[str] = "corpus.rationale_staleness"

#: The tickets named on the claim, in the order the facts arrived.
UNBLOCKED_BY: Final[str] = "MWF7Z1EN, QEVSTMYW"

#: Exclusion keys, each a sentence a report renders verbatim. Three are per rationale
#: and one is per file, and the units differ because the conditions differ: a rationale
#: either has a clock and a file set or it does not, while a file either has movement
#: after the merge or it does not.
NO_FILES_KEY: Final[str] = "a rationale naming no file, so there is no code to be stale against"
NO_MERGE_KEY: Final[str] = (
    "a rationale whose change has no pr_merged_at, so there is no clock to measure movement from"
)
TRUNCATED_FILE_LIST_KEY: Final[str] = (
    "a rationale whose file list Kojutsu bounded, so the files outside the bound are absent "
    "rather than undisturbed"
)
NO_MOVEMENT_KEY: Final[str] = (
    "a file the rationale names that no record was captured against after the change merged, so "
    "there is no interval to report"
)

#: What the population is, in the words the denominator renders. "Rationale-file pairs"
#: because that is the unit: one rationale about three files is three statements about
#: staleness, and a denominator of rationales would make a reason that names more code
#: look like a heavier claim than one that names a single file.
STALENESS_POPULATION: Final[str] = (
    "rationale-file pairs in this read whose change merged and whose file was captured against "
    "afterwards"
)

#: The negative space, in the words the reader needs. Three clauses and each is a
#: different misreading, so they are separate sentences rather than one long one a
#: reader skims.
STALENESS_IS_NOT_ERROR: Final[str] = (
    "That a stale rationale is a wrong rationale, or a bad decision. Staleness is movement, not "
    "error: a reason that has not been revisited in a year may be one nobody needed to revisit, "
    "and a reason that was revisited may have been confirmed rather than corrected. Nothing in "
    "this corpus says which."
)
ONE_FILE_IS_NOT_THE_CHANGE: Final[str] = (
    "A statement about the change as a whole, or about a file nobody mentioned. A rationale about "
    "one file says nothing about the other files the same change touched -- a change can move one "
    "file the reason covers and rewrite the module it depends on -- and a file no rationale names "
    "is not in this distribution at all rather than shown as fresh."
)
CAPTURE_IS_NOT_MOVEMENT: Final[str] = (
    "A measurement of code movement. What is dated is a *capture* against the file, which is a "
    "proxy somebody left something in the store, and the two are not the same: a stated reason "
    "revisited after the merge counts here even where the code has not changed a line."
)


def staleness_claim(total: int) -> Claim:
    """The claim behind the gap distribution, over the pairs that have both ends."""
    return Claim(
        slug=STALENESS_SLUG,
        statement=(
            "The time between a change merging and the last capture against a file its stated "
            "reason names is distributed as the buckets below show."
        ),
        does_not_mean=(
            STALENESS_IS_NOT_ERROR
            + " "
            + ONE_FILE_IS_NOT_THE_CHANGE
            + " "
            + CAPTURE_IS_NOT_MOVEMENT
        ),
        falsifier=(
            "Two records about the same change whose `pr_merged_at` values differ. Every writer "
            "of a record about one change should write the forge's merge time for it, so a "
            "disagreement would mean the gap is being computed between two clocks that are not "
            "the clock the claim names, and the buckets would be comparing two different "
            "questions. This measure takes the first stated value and does not adjudicate, which "
            "is the same posture the corpus layer takes towards a tag and an id that disagree."
        ),
        denominator=Denominator(STALENESS_POPULATION, max(1, total)),
        kind=ClaimKind.descriptive,
        granularity=Granularity.repository,
        unblocked_by=UNBLOCKED_BY,
    )


@dataclass(frozen=True)
class RationaleStalenessMeasure:
    """Bucketed gaps from a change's merge to the last capture on a file it reasoned about."""

    requires_certainty = requires_certainty

    @property
    def slug(self) -> str:
        """The identifier this measure is cited by."""
        return STALENESS_SLUG

    def claim(self, snapshot: CorpusSnapshot) -> Claim:
        """The claim, over the rationale-file pairs in this read that have both ends."""
        gaps, _ = self._gaps(snapshot.records)
        return staleness_claim(len(gaps))

    def compute(self, snapshot: CorpusSnapshot) -> Figure:
        """The bucketed distribution, with every unmeasurable pair named and counted."""
        gaps, excluded = self._gaps(snapshot.records)
        return DistributionFigure(
            claim=self.claim(snapshot),
            snapshot=snapshot,
            title=(
                "Time from a change merging to the last capture against a file its stated reason "
                "names"
            ),
            values=duration_buckets(gaps),
            excluded=excluded,
        )

    def _gaps(self, records: Sequence[Record]) -> tuple[list[float], Mapping[str, int]]:
        """Every measurable gap, and the conditions that made the rest unmeasurable.

        Two passes over one list, and the order is the reason. The first builds the
        index of the newest capture per file, because "the last capture against this
        file" is a question about the whole corpus and answering it while walking the
        rationales would answer it several times over, once per rationale, with a
        different answer each time. The second walks the rationales against that index.

        A file's key carries the repository, and so does the rationale's, because the
        paths Kojutsu writes are repository-relative: a bare path would let
        ``src/queue.py`` in one area be dated by a capture in another.
        """
        newest: dict[tuple[str, str], datetime] = {}
        for record in records:
            if not named_repository(record.repo):
                continue
            assert record.repo is not None
            if record.captured_at is None:
                continue
            for path in record.files:
                key = (record.repo, path)
                current = newest.get(key)
                if current is None or record.captured_at > current:
                    newest[key] = record.captured_at

        merged: dict[tuple[str, int], datetime] = {}
        for record in records:
            key = record.pr_key
            if key is None or record.pr_merged_at is None:
                continue
            merged.setdefault(key, record.pr_merged_at)

        gaps: list[float] = []
        excluded: dict[str, int] = {}
        for record in records:
            if not record.is_rationale:
                continue
            if not record.files:
                excluded[NO_FILES_KEY] = excluded.get(NO_FILES_KEY, 0) + 1
                continue
            if record.files_truncated:
                excluded[TRUNCATED_FILE_LIST_KEY] = excluded.get(TRUNCATED_FILE_LIST_KEY, 0) + 1
                continue
            key = record.pr_key
            merge_time = None if key is None else merged.get(key)
            if merge_time is None:
                excluded[NO_MERGE_KEY] = excluded.get(NO_MERGE_KEY, 0) + 1
                continue
            assert record.repo is not None
            for path in record.files:
                touched = newest.get((record.repo, path))
                # ``<=`` rather than ``<``, and the reason the two are not the same
                # test: a capture at the merge instant leaves a gap of zero, which is
                # the same number a never-moved file would produce and means the
                # opposite thing. Both go to the exclusion, so the gap reported is
                # always strictly positive and the duration ladder's negative rung is
                # never reachable from here.
                if touched is None or touched <= merge_time:
                    excluded[NO_MOVEMENT_KEY] = excluded.get(NO_MOVEMENT_KEY, 0) + 1
                    continue
                gaps.append((touched - merge_time).total_seconds() / 3600.0)
        return gaps, freeze_counts(excluded, "staleness.excluded")
