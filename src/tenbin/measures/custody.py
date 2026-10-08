"""How many files in this corpus have exactly one account behind them, and where.

Kojutsu's ``MWF7Z1EN`` wrote the file anchor: ``files`` on a record, the paths the
change touched, which is the edge Tanseki's deriver reads. That is the fact two refused
measures were waiting on, and this is the one a reader is most likely to want, because
"only one person ever said anything about this module" is actionable and "the corpus is
forty percent one repository" is not.

**It is a statement about the corpus, and not about the team.** The unit is the
*distinct account that has captured something about a file*, drawn from the three fields
that carry one -- ``comment_author``, ``declared_by`` and ``change_author_account`` -- and
that number is a count of what reached the store. A file nobody commented on is absent
from the distribution entirely, and a file with one captured record may have had five
people reading it and one person writing. Neither of those is recoverable from inside
this seam: the corpus holds what somebody chose to say, and silence in it is not evidence
of absence. So the claim says so where a reader will see it before the number, and the
figure is keyed by an area rather than by a person.

**Two things this measure is built not to become, and both are in the claim's negative
space.** It never names an individual, because per-principal output is a different
product with different consent and access requirements, and the answer is an
*area-level count* rather than a list of sole custodians. And it never multiplies a count
of accounts by a judgement about the file, for the reason every measure here refuses
that: there is no field in the corpus that says whether a file is well understood.

**A file is identified by its area as well as its path.** Kojutsu writes the paths
repository-relative, so ``src/queue.py`` exists once per repository and a key of the path
alone would merge two repositories' files into one and report a shared custodian for a
file that has two. A record with no named repository is therefore excluded rather than
filed under the placeholder, for the same reason
:func:`~tenbin.measures.filtering.named_repository` exists.

**A bounded file list loses files here, not accounts, so it is not an exclusion.**
Kojutsu caps ``files`` at fifty and writes ``"N more files not listed"`` into the
list. For this measure that means a file beyond the bound is absent from the file set --
the file is the unit, and the bound dropped whole files -- while every file that *is*
named still carries the full set of accounts that touched it. Excluding the record would
discard correct data to fix nothing, and the undercount is stated in the claim instead.

**What this module notably does not do:** it does not count a person. An account is
counted where it is stated, once per file, and a file nobody named is not in the
distribution at all. It also does not join this measure to a merge outcome or a check
conclusion; the two axes exist in this corpus and combining them into a score is what
:mod:`tenbin.measures.check_outcome` exists to refuse.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from tenbin.claims.mechanism import Granularity
from tenbin.claims.model import Claim, ClaimKind, Denominator
from tenbin.corpus.record import Record
from tenbin.corpus.snapshot import CorpusSnapshot
from tenbin.corpus.values import Stated, parse_stated
from tenbin.measures.base import DistributionFigure, Figure, FigureGroup, freeze_counts
from tenbin.measures.filtering import named_repository, requires_certainty

#: The slug this measure is cited by, and the ticket that unblocked it.
CUSTODY_SLUG: Final[str] = "corpus.single_custodian_per_area"

#: The ticket named on the claim, in the field whose whole job is to say what would
#: change the answer.
UNBLOCKED_BY: Final[str] = "MWF7Z1EN"

#: The bucket holding the answer to the question this measure exists for. Named rather
#: than written at the call sites so that a report, a test and a second measure cannot
#: disagree about which bucket is the single-custodian one.
SINGLE_CUSTODIAN_LABEL: Final[str] = "exactly 1 account"

#: The top rung of the contributor ladder, and the label for everything at or above it.
#: The question is legible at the low end and stops being one above five: a file six
#: accounts have said something about is not a custody question, and a ladder with no
#: top would make every high-cardinality file a bucket of its own and the shape of the
#: distribution would depend on the corpus.
MAX_NAMED_CONTRIBUTORS: Final[int] = 5
MANY_CONTRIBUTORS_LABEL: Final[str] = f"{MAX_NAMED_CONTRIBUTORS} or more accounts"

#: Exclusion keys, each naming a condition rather than a code. A report renders them
#: verbatim and an operator acts on them, so all three are sentences.
NO_AREA_KEY: Final[str] = (
    "a record that names a file but no repository (the id carries Kojutsu's placeholder), "
    "so the path cannot be resolved against one"
)
UNSTATED_ACCOUNT_KEY: Final[str] = (
    "a record that names a file and states no account on any of the three fields an account is "
    "drawn from, so nothing was captured under a name"
)

#: What the population is, in the words the denominator renders. "Files" rather than
#: "records" because the record is the raw material and the file is the unit, and a
#: reader who saw "records" in the denominator of a per-file figure would add the wrong
#: column up.
CUSTODY_POPULATION: Final[str] = (
    "distinct files named by the records in this read, within a named repository"
)

#: The half of the argument that is a refusal rather than a caveat: the figure is an
#: area-level count precisely because per-principal output is decision 002's refusal and
#: not this report with a different filter.
NO_SOLE_CUSTODIAN_NAMED: Final[str] = (
    "This is a statement about the corpus, not about the team, and it never names a person. A "
    "file no captured record names is absent from the distribution rather than shown as having "
    "no custodian, because nobody commenting on a file is not the same observation as nobody "
    "reading it. A file with one captured record may have had five people looking at it and one "
    "person writing, and nothing in this corpus can tell those two apart. The output is an "
    "area-level count because a per-principal figure is a different product with different "
    "consent and access requirements, which is the floor decision 002 draws."
)

#: The second refusal, about the file rather than the corpus: there is no field here that
#: says a file is understood, and a count of accounts is not a proxy for it.
ACCOUNTS_IS_NOT_KNOWLEDGE: Final[str] = (
    "It does not mean the file is known, or well documented, or risky to change. The count is of "
    "accounts that left something in the store, and `declared_by` in particular comes from a "
    "stated reason, which is a claim about intent and never evidence about code."
)


def custody_claim(total: int) -> Claim:
    """The claim behind both custody figures, over the files this read can resolve."""
    return Claim(
        slug=CUSTODY_SLUG,
        statement=(
            f"The files in this corpus are distributed across the number of accounts that stated "
            f"something about them as the figures below show, and the files with "
            f"{SINGLE_CUSTODIAN_LABEL} are counted per area."
        ),
        does_not_mean=NO_SOLE_CUSTODIAN_NAMED + " " + ACCOUNTS_IS_NOT_KNOWLEDGE,
        falsifier=(
            "A file with exactly one captured record whose change has several distinct accounts "
            "commenting on it. That is a direct observation of several people having written "
            "about one file, and it would mean the figure is tracking who captured rather than "
            "who knows -- so it is the observation that distinguishes this measure from the "
            "claim it must not be read as."
        ),
        denominator=Denominator(CUSTODY_POPULATION, max(1, total)),
        kind=ClaimKind.descriptive,
        # Coarser than a team and never about a person. The unit is a repository because
        # nothing in a record names a team, and a repository is the widest population
        # this corpus can name without inventing an org chart it does not hold.
        granularity=Granularity.repository,
        unblocked_by=UNBLOCKED_BY,
    )


@dataclass(frozen=True)
class SingleCustodianMeasure:
    """Files per contributing account, and the single-custodian ones per area.

    Two figures under one claim and therefore one denominator, because both are
    statements over the same set of files: the first buckets the files by how many
    accounts named them, the second counts the files in the one bucket that matters per
    area. A report showing either alone is correct on its own terms.
    """

    requires_certainty = requires_certainty

    @property
    def slug(self) -> str:
        """The identifier this measure is cited by."""
        return CUSTODY_SLUG

    def claim(self, snapshot: CorpusSnapshot) -> Claim:
        """The claim, over the files this read could resolve to a named repository."""
        accounts, _ = self._custody(snapshot.records)
        return custody_claim(len(accounts))

    def compute(self, snapshot: CorpusSnapshot) -> Figure:
        """The two distributions, with everything left out named and counted."""
        accounts, excluded = self._custody(snapshot.records)
        claim = self.claim(snapshot)
        return FigureGroup(
            claim=claim,
            snapshot=snapshot,
            title="Files per number of accounts that stated something about them",
            figures=(
                DistributionFigure(
                    claim=claim,
                    snapshot=snapshot,
                    title="Files per number of accounts that stated something about them",
                    values=_contributor_buckets(accounts),
                    excluded=excluded,
                ),
                DistributionFigure(
                    claim=claim,
                    snapshot=snapshot,
                    title="Files with exactly one contributing account, per area",
                    values=_single_custodian_by_area(accounts),
                    # The same exclusions as the figure above, because both figures are
                    # over the same file set: a record kept out of one is kept out of
                    # the other, and the two share a denominator for that reason. The
                    # columns will not add up to the same number -- only the
                    # single-custodian files are counted here -- and that difference is
                    # the rung above rather than an unexplained remainder.
                    excluded=excluded,
                ),
            ),
        )

    def _custody(
        self, records: Sequence[Record]
    ) -> tuple[Mapping[tuple[str, str], frozenset[str]], Mapping[str, int]]:
        """Every resolvable file, and the accounts that stated something about it.

        Two exclusions rather than one, because they are different gaps and a reader
        fixing one of them needs to know which they have. A record naming no repository
        cannot have its path resolved, and a record naming a file and no account has
        nothing to count. The second is counted per record and there is deliberately no
        matching file-level key: a file reaches the mapping only through a record that
        stated an account, so a file with no accounts never comes into existence, and a
        key that could never fire would be a sentence a report could render about a
        condition that does not occur.

        Accounts are drawn from three fields and from nothing else. ``author`` is
        deliberately absent: Kojutsu fills it with an agent id and with the literal
        ``"unknown"``, so a fallback there would put a machine and a person in one set
        and call the result a count of contributors. Each field goes through
        :func:`~tenbin.corpus.values.parse_stated`, so a literal ``"unknown"`` is an
        absence and never becomes a contributor.
        """
        by_file: dict[tuple[str, str], set[str]] = defaultdict(set)
        unlocated = 0
        unattributed = 0
        for record in records:
            if not record.files:
                continue
            if not named_repository(record.repo):
                # The path is repository-relative, so without a repository this file
                # cannot be told apart from a same-named file in a repository the read
                # did hold. Counting it would merge two areas and invent a shared
                # custodian across them.
                unlocated += 1
                continue
            assert record.repo is not None
            accounts = {
                parsed.value
                for parsed in (
                    parse_stated(value)
                    for value in (
                        record.comment_author,
                        record.declared_by,
                        record.change_author_account,
                    )
                )
                if isinstance(parsed, Stated)
            }
            if not accounts:
                unattributed += 1
                continue
            for path in record.files:
                by_file[(record.repo, path)].update(accounts)
        excluded: dict[str, int] = {}
        if unlocated:
            excluded[NO_AREA_KEY] = unlocated
        if unattributed:
            excluded[UNSTATED_ACCOUNT_KEY] = unattributed
        return (
            {key: frozenset(accounts) for key, accounts in by_file.items()},
            freeze_counts(excluded, "custody.excluded"),
        )


def _contributor_buckets(accounts: Mapping[tuple[str, str], frozenset[str]]) -> Mapping[str, int]:
    """Bucket the files by how many accounts named them, every rung present at zero.

    Materialising the whole ladder is the same call
    :func:`~tenbin.measures.base.duration_buckets` makes, for the same reason: a
    distribution whose shape depends on the corpus is one every report has to guard
    against, and ``"exactly 1 account: 0"`` is this measure's answer to the question it
    was written for. An absent rung would read as "not computed".
    """
    labels = _contributor_labels()
    buckets: dict[str, int] = dict.fromkeys(labels, 0)
    for names in accounts.values():
        # Clamped before indexing rather than after, so the open-ended top rung is
        # counted rather than raising on a file six accounts have touched.
        size = min(len(names), MAX_NAMED_CONTRIBUTORS)
        buckets[labels[size - 1]] += 1
    return freeze_counts(buckets, "custody.contributor_buckets")


def _contributor_labels() -> tuple[str, ...]:
    """The ladder's labels, from the single-custodian rung upwards.

    A function rather than a constant for the reason
    :func:`~tenbin.measures.base.bucket_labels` is one: the rung count and the labels
    have to come from the same number. The top rung is the open one, so the named rungs
    stop one short of it -- a rung labelled ``"5 accounts"`` beside another labelled
    ``"5 or more accounts"`` would be a bucket nothing could ever land in.
    """
    labels = [SINGLE_CUSTODIAN_LABEL]
    labels.extend(f"{size} accounts" for size in range(2, MAX_NAMED_CONTRIBUTORS))
    labels.append(MANY_CONTRIBUTORS_LABEL)
    return tuple(labels)


def _single_custodian_by_area(
    accounts: Mapping[tuple[str, str], frozenset[str]],
) -> Mapping[str, int]:
    """How many single-custodian files each named area has.

    Sorted by count descending and then by area, which is the ordering
    :func:`~tenbin.measures.base.histogram` uses and the reason it is reproduced
    rather than inherited: ``histogram`` counts one value per item, and this is the same
    shape over a pre-counted mapping. The secondary sort is what makes two reads of the
    same corpus render identically, which is what a test can then assert on.
    """
    by_area: Counter[str] = Counter()
    for (area, _path), names in accounts.items():
        if len(names) == 1:
            by_area[area] += 1
    return freeze_counts(
        dict(sorted(by_area.items(), key=lambda item: (-item[1], item[0]))),
        "custody.by_area",
    )
