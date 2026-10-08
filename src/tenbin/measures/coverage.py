"""Where this corpus reaches and where it is silent, in four distributions.

Coverage is the question a self-selected corpus answers worst and most
confidently. Four hundred documents over nine repositories reads like a map of the
organisation; it is a map of the paths somebody happened to walk. So the figures
here are distributions of *records*, and each one's claim names the same negative
space: a repository nobody answers on is byte-identical to a repository with no
knowledge debt, and no backfill path exists for either. The census
(``EB6FE5ZP``) is the fact that would change that, and it is named as the
falsifier of every one of them -- a census naming materially more repositories
than these distributions show is a measurement error, and a census of *uncaptured*
changes is a coverage error this program currently cannot see.

**The model distribution is built only from stated values.** ``answered_by_model``
is parsed into :class:`~tenbin.corpus.values.Stated` or
:class:`~tenbin.corpus.values.Unstated`, the second is skipped, and the skips are
counted in ``excluded`` under a key naming the condition. A record that states no
model is not evidence that nobody used one, and putting it in a bucket named
``unknown`` would report the absence of a claim as a characteristic of a
principal. Kojutsu has now stopped writing the literal: an unstated model
leaves its key absent, which is the one mechanism in the writing project that
expresses absence honestly. Every record written before that change still carries
the literal, and for those the two are not separable -- which is why the exclusion
is stated as a condition rather than presented as a clean split.

**The author grouping does not fall back to ``author``.** ``comment_author`` is
the account that posted; ``author`` is the answer's author, which Kojutsu fills
with ``"unknown"`` for an unattributed answer and with an agent id for an
agent-authored one. Two different things sharing one string, so a fallback would
put an agent id and a person in the same bucket and call it a distribution of
people. A record with no ``comment_author`` has nobody to attribute and is counted
separately; the sentinel problem does not get a second costume here.

**Every claim is at ``team`` granularity, and none of them is a ranking.** A
histogram over repositories is a statement about the shape of the corpus, and a
histogram over accounts is a count of records per account with no account compared
to another and none of the numbers attached to anybody as a property. That is the
distinction the granularity floor draws: it refuses a claim *about* a principal,
not a figure that happens to have a login in a bucket key.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from tenbin.claims.mechanism import Granularity
from tenbin.claims.model import Claim, ClaimKind, Denominator
from tenbin.corpus.snapshot import CorpusSnapshot
from tenbin.corpus.values import Stated
from tenbin.measures.base import DistributionFigure, Figure, histogram
from tenbin.measures.filtering import (
    capture_population,
    named_repository,
    requires_certainty,
)

#: The four slugs, one per figure, all dotted so they cannot collide with a
#: slugified refusal entry from ``docs/seam.md``.
REPOSITORY_COVERAGE_SLUG: Final[str] = "corpus.repository_coverage"
MONTH_COVERAGE_SLUG: Final[str] = "corpus.monthly_coverage"
AUTHOR_COVERAGE_SLUG: Final[str] = "corpus.author_coverage"
MODEL_COVERAGE_SLUG: Final[str] = "corpus.declared_model_coverage"

#: Exclusion keys. Every one is a sentence a report can render verbatim, because a
#: report renders these keys and a reader acts on them; a machine-readable key like
#: ``"no_model"`` is a code an operator has to translate, and translating it wrong
#: is how an exclusion becomes a result.
NO_REPOSITORY_KEY: Final[str] = "no repository on the record (the id carries Kojutsu's placeholder)"
NO_MONTH_KEY: Final[str] = "no timestamp on the record"
NO_COMMENT_AUTHOR_KEY: Final[str] = "no commenting account on the record"
NO_STATED_MODEL_KEY: Final[str] = "no model stated (the record names no model at all)"


#: What each population is, in the words the claim's denominator renders.
REPOSITORY_POPULATION: Final[str] = "capture records in this read, grouped by repository"
MONTH_POPULATION: Final[str] = "capture records in this read, grouped by the month of their event"
AUTHOR_POPULATION: Final[str] = "capture records in this read, grouped by the account that posted"
MODEL_POPULATION: Final[str] = (
    "capture records in this read that state a model, grouped by that model"
)

#: The negative space every one of these claims shares. Written once and rendered
#: into each of them, because four paraphrases of one bias is four chances to lose
#: the half that matters.
NOT_COVERAGE_OF_DEVELOPMENT: Final[str] = (
    "That this is the shape of development. It is not: the corpus is self-selected, because "
    "Kojutsu captures only what it is present for, and self-censored, because a capture path "
    "that fails writes nothing and is indistinguishable from a change nobody examined. A "
    "repository nobody answers on is byte-identical to a repository with no knowledge debt, and "
    "no backfill path exists for either."
)

#: The falsifier every one of these claims shares, for the same reason.
COVERAGE_FALSIFIER: Final[str] = (
    "A census of changes that produced no capture (`EB6FE5ZP`) naming materially more "
    "repositories than the distribution below shows. More repositories in the census than in the "
    "corpus would mean the distribution is measuring capture, not knowledge debt."
)


def _coverage_claim(
    *, slug: str, statement: str, not_mean: str, population: str, total: int
) -> Claim:
    """Build one of the four coverage claims, over a population this caller chose.

    Every one is ``descriptive`` and at ``team`` granularity. Neither is decoration:
    ``descriptive`` because no assignment mechanism exists and one is not going to
    arrive from a ticket, and ``team`` because the floor refuses a claim narrower
    than a team and a histogram that has a login in a bucket key is not a claim
    *about* that login -- no account is compared to another here, and no count is
    attached to anybody as a property of them.
    """
    return Claim(
        slug=slug,
        statement=statement,
        does_not_mean=not_mean,
        falsifier=COVERAGE_FALSIFIER,
        denominator=Denominator(population, max(1, total)),
        kind=ClaimKind.descriptive,
        granularity=Granularity.team,
    )


def repository_claim(total: int) -> Claim:
    """The claim behind the by-repository distribution."""
    return _coverage_claim(
        slug=REPOSITORY_COVERAGE_SLUG,
        statement="This corpus's knowledge is distributed across repositories as the figure below shows.",
        not_mean=(
            "That the repositories not in the figure have no knowledge debt. They are absent "
            "from it. " + NOT_COVERAGE_OF_DEVELOPMENT
        ),
        population=REPOSITORY_POPULATION,
        total=total,
    )


def month_claim(total: int) -> Claim:
    """The claim behind the by-month distribution."""
    return _coverage_claim(
        slug=MONTH_COVERAGE_SLUG,
        statement="This corpus's knowledge is distributed over time as the figure below shows.",
        not_mean=(
            "That months with no records are months with no knowledge, or that activity is "
            "continuous. A month is read from the event a record is about -- its answer time, "
            "then its capture time, then its declaration -- so a corpus with gaps in any of "
            "those clocks is grouped over a mixture, and the gaps are not absences of knowledge. "
            + NOT_COVERAGE_OF_DEVELOPMENT
        ),
        population=MONTH_POPULATION,
        total=total,
    )


def author_claim(total: int) -> Claim:
    """The claim behind the by-account distribution."""
    return _coverage_claim(
        slug=AUTHOR_COVERAGE_SLUG,
        statement=(
            "This corpus's knowledge is distributed across commenting accounts as the figure below "
            "shows."
        ),
        not_mean=(
            "That any account contributed more knowledge than another, or that a count here is a "
            "property of a person. The figure counts records per account and compares no account "
            "with another; who was assigned the work is not in the corpus, and an account absent "
            "from the figure has not been shown to have had nothing to say. "
            + NOT_COVERAGE_OF_DEVELOPMENT
        ),
        population=AUTHOR_POPULATION,
        total=total,
    )


def model_claim(total: int) -> Claim:
    """The claim behind the stated-model distribution."""
    return _coverage_claim(
        slug=MODEL_COVERAGE_SLUG,
        statement=(
            "The models principals state for themselves in this corpus are distributed as the "
            "figure below shows."
        ),
        not_mean=(
            "That these are the models that were used, or that the records excluded from the "
            "figure used no model. A stated model is a self-assertion by the account posting it, "
            "which nothing in the system can verify, and a record that states none is excluded "
            "rather than filed under an absence. " + NOT_COVERAGE_OF_DEVELOPMENT
        ),
        population=MODEL_POPULATION,
        total=total,
    )


@dataclass(frozen=True)
class RepositoryCoverageMeasure:
    """Records per repository, with a record that names none counted separately.

    The exclusion key here is a sentence rather than the provenance layer's
    ``(no repository)`` bucket label, and the difference is not cosmetic. The two
    live in different places: ``(no repository)`` is a *key in a histogram* of
    anomalous records, and this is a *reason for exclusion* in a distribution of
    all records. Reusing one string for both would give a reader who has seen the
    provenance profile a bucket they think they already understand, in a figure
    that counts something else entirely.
    """

    requires_certainty = requires_certainty

    @property
    def slug(self) -> str:
        """The identifier this measure is cited by."""
        return REPOSITORY_COVERAGE_SLUG

    def claim(self, snapshot: CorpusSnapshot) -> Claim:
        """The claim, over the capture records this read enumerated."""
        return repository_claim(len(capture_population(snapshot.records)))

    def compute(self, snapshot: CorpusSnapshot) -> Figure:
        """The histogram of repositories, with the placeholder counted and named.

        A record with no repository is filed at ``unknown/pr-<n>/<entry>``, so
        ``Record.repo`` is the string ``"unknown"`` rather than ``None`` -- the
        truthiness check is not enough and the placeholder has to be named as one.
        """
        unnameable = 0
        repos: list[str] = []
        for record in capture_population(snapshot.records):
            if named_repository(record.repo):
                assert record.repo is not None
                repos.append(record.repo)
            else:
                unnameable += 1
        return DistributionFigure(
            claim=self.claim(snapshot),
            snapshot=snapshot,
            title="Records per repository",
            values=histogram(repos),
            excluded={NO_REPOSITORY_KEY: unnameable} if unnameable else {},
        )


@dataclass(frozen=True)
class MonthlyCoverageMeasure:
    """Records per month of the event each one is about, with undated records excluded.

    ``Record.month`` already encodes the preference order over clocks --
    ``answered_at`` then ``captured_at`` then ``declared_at`` then the document's
    own update -- so this measure does not restate it. What it does is say, in the
    claim, that a month is read from a mixture of clocks when a corpus has gaps in
    any of them, because a reader comparing two months of a corpus with a broken
    clock is comparing two different questions.
    """

    requires_certainty = requires_certainty

    @property
    def slug(self) -> str:
        """The identifier this measure is cited by."""
        return MONTH_COVERAGE_SLUG

    def claim(self, snapshot: CorpusSnapshot) -> Claim:
        """The claim, over the capture records this read enumerated."""
        return month_claim(len(capture_population(snapshot.records)))

    def compute(self, snapshot: CorpusSnapshot) -> Figure:
        """The histogram of months, with undated records excluded and named."""
        undated = 0
        months: list[str] = []
        for record in capture_population(snapshot.records):
            month = record.month
            if month is None:
                undated += 1
            else:
                months.append(month)
        return DistributionFigure(
            claim=self.claim(snapshot),
            snapshot=snapshot,
            title="Records per month of the event each record is about",
            values=histogram(months),
            excluded={NO_MONTH_KEY: undated} if undated else {},
        )


@dataclass(frozen=True)
class AuthorCoverageMeasure:
    """Records per commenting account, with unattributed records excluded and named.

    The grouping is ``comment_author`` and only ``comment_author``. ``author`` is
    the answer's author, which Kojutsu fills with the literal ``"unknown"`` for
    an unattributed answer and with an agent id for an agent-authored one -- so a
    fallback would put an agent and a person in one bucket and render the result as
    a distribution of people. A record with no ``comment_author`` has nobody to
    attribute and is counted in ``excluded``; it is not filed under the string
    ``"unknown"``, which is a value somebody could have been named.
    """

    requires_certainty = requires_certainty

    @property
    def slug(self) -> str:
        """The identifier this measure is cited by."""
        return AUTHOR_COVERAGE_SLUG

    def claim(self, snapshot: CorpusSnapshot) -> Claim:
        """The claim, over the capture records this read enumerated."""
        return author_claim(len(capture_population(snapshot.records)))

    def compute(self, snapshot: CorpusSnapshot) -> Figure:
        """The histogram of commenting accounts, with unattributed ones excluded."""
        unattributed = 0
        accounts: list[str] = []
        for record in capture_population(snapshot.records):
            if record.comment_author:
                accounts.append(record.comment_author)
            else:
                unattributed += 1
        return DistributionFigure(
            claim=self.claim(snapshot),
            snapshot=snapshot,
            title="Records per commenting account",
            values=histogram(accounts),
            excluded={NO_COMMENT_AUTHOR_KEY: unattributed} if unattributed else {},
        )


@dataclass(frozen=True)
class DeclaredModelCoverageMeasure:
    """The models principals stated, built only from the records that state one.

    The iteration is over the parsed ``Stated | Unstated`` rather than over a
    string, and ``Unstated`` is skipped and counted rather than turned into a
    bucket. That is the whole measure: a distribution in which "nobody said" sits
    beside the models, in the same column and with the same formatting, is a
    distribution in which a reader will conclude that a group of records declared a
    model called ``unknown``. Kojutsu used to write exactly that literal and has
    stopped, so the excluded count is a legacy record count -- which is worth
    knowing and is why the exclusion key names the condition rather than the
    remedy.
    """

    requires_certainty = requires_certainty

    @property
    def slug(self) -> str:
        """The identifier this measure is cited by."""
        return MODEL_COVERAGE_SLUG

    def claim(self, snapshot: CorpusSnapshot) -> Claim:
        """The claim, over the capture records this read enumerated.

        The denominator population is every capture, not only the ones that state a
        model, so the excluded count is inside the population the claim describes
        rather than beside it -- a reader comparing this figure with the by-repository
        one is then comparing the same corpus.
        """
        return model_claim(len(capture_population(snapshot.records)))

    def compute(self, snapshot: CorpusSnapshot) -> Figure:
        """The histogram of stated models, with unstated ones excluded and named."""
        unstated = 0
        models: list[str] = []
        for record in capture_population(snapshot.records):
            stated = record.answered_by_model
            if isinstance(stated, Stated):
                models.append(stated.value)
            else:
                # ``Unstated`` is the only other member, so the else is the
                # exclusion rather than a third case nobody thought of.
                unstated += 1
        return DistributionFigure(
            claim=self.claim(snapshot),
            snapshot=snapshot,
            title="Models principals stated for themselves",
            values=histogram(models),
            excluded={NO_STATED_MODEL_KEY: unstated} if unstated else {},
        )
