"""Whether two statements about one change agree -- and the three questions that is not.

A single implementation produces two rationales for a change: a ``declared`` one stated
by whoever did the work, and a ``reconstructed`` one inferred from the diff by a
reviewer. Kojutsu classifies the relationship between them, and Tenbin was
publishing a corpus inventory in which that classification did not appear.

**The vocabulary is Kojutsu's, and a second one would be two truths about one
relationship.** :class:`ComparisonOutcome` below is a verbatim copy of the enum at
``kojutsu/src/kojutsu/core/rationale_link.py:37-56``, values and all, and so is
:data:`OUTCOME_NOTES` from ``:61-83``. Both copies name their source in the module and
in :data:`KOJUTSU_VOCABULARY_SOURCE`, and
:func:`tenbin.measures.rationale_comparison.outcome_values_are_kojutsus` asserts
the value sets are equal against Kojutsu's source rather than against a comment.
The failure mode this prevents is quiet: two enumerations would each be right about
most pairs, so nothing would fail and the two would simply disagree in a report.

The same discipline applies one level down to independence. :func:`compute_independence`
reproduces ``kojutsu.models.compute_independence`` and returns
:class:`~tenbin.measures.trust.Independence`, the enum this package already holds, so
that the axis is shared rather than re-spelled here in a third place.

**Agreement is not corroboration, and the vocabulary is arranged to make that
unreadable past.** ``RESTATEMENT`` is a member with the same weight as
``DIVERGENT``, and it is what two rationales from the same account and model produce
whether they match or not: the same mind agreeing with itself. Kojutsu's docstring
is explicit that arranging any output to invite "the reviewer disagreed with the
implementer, therefore ..." is the failure, which is why the caveat sits in
:attr:`Claim.does_not_mean` rather than in a footnote a reader can skip -- and why the
second figure in the group counts the changes a comparison was *actually* made on.

**A change where nobody reconstructed anything is reported, not dropped.** It lands in
``DECLARED_ONLY``, which is a fact about the corpus: nothing about the reasoning was
reconstructed. Excluding it would make the denominator "changes somebody checked", and
a distribution over changes somebody checked cannot report how often nobody reconstructed
anything at all.

**The unit of every figure here is the change, and the exclusions are counted in
changes too.** Buckets, exclusions and denominator are all one unit so that they add
up, which is the invariant the rest of this package's distributions hold. A rationale
document naming no repository-and-pull-request pair is therefore *outside* the
population rather than an exclusion -- it is not a change -- and the population
description says so, because a restriction the reader cannot see is the defect
:class:`~tenbin.measures.base.DistributionFigure` exists to prevent.

**What this module notably does not do:** it does not rank, score, or resolve a
divergence, and it does not say which rationale is right. A divergence is the finding
and not a defect in either record -- Kojutsu's own note says so, and it is carried
here verbatim. Whether a particular divergence mattered is a question about one change
and one person's job, and this corpus holds nothing that would answer it.
"""

from __future__ import annotations

import pathlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Final

from tenbin.claims.mechanism import Granularity
from tenbin.claims.model import Claim, ClaimKind, Denominator
from tenbin.corpus.record import Record
from tenbin.corpus.snapshot import CorpusSnapshot
from tenbin.corpus.values import Stated, Unstated
from tenbin.measures.base import (
    CountFigure,
    DistributionFigure,
    Figure,
    FigureGroup,
    freeze_counts,
)
from tenbin.measures.filtering import named_repository, requires_certainty
from tenbin.measures.trust import Independence

#: Where the vocabulary below comes from, as a file and a line so that the copy can be
#: checked rather than trusted. Kojutsu owns this classification; Tenbin reads it.
#: If the two enums ever disagree, Kojutsu's file is the truth and this copy is a
#: defect -- and :func:`outcome_values_are_kojutsus` is what notices.
KOJUTSU_VOCABULARY_SOURCE: Final[str] = (
    "kojutsu/src/kojutsu/core/rationale_link.py:37-56 (ComparisonOutcome), "
    ":61-83 (OUTCOME_NOTES), :132-199 (compare_rationales), "
    "kojutsu/src/kojutsu/models.py:205-233 (compute_independence)"
)

#: The slug this measure is cited by. Dotted rather than dashed so it can never collide
#: with a ``refusals.slugify``-ed catalogue entry.
RATIONALE_RELATIONSHIP_SLUG: Final[str] = "corpus.rationale_relationship"

#: The Kojutsu ticket that made the relationship reachable, recorded on the claim so
#: a report says why the figure exists rather than leaving it to a merge commit.
RELATIONSHIP_SOURCE: Final[str] = "XMHKJJ9C"


class ComparisonOutcome(StrEnum):
    """What a comparison of two rationales actually established.

    **Copied verbatim from ``kojutsu/src/kojutsu/core/rationale_link.py:37-56``,
    values included.** Not "adapted", not "re-expressed": the members are the same six
    with the same six strings, and the ordering of the reasons below is Kojutsu's
    ordering and his reasoning.

    A second vocabulary for one relationship is two truths about it, and the failure
    would be silent -- both would be correct about most pairs, so no assertion would
    fail and the two would simply disagree in a report about a corpus whose author had
    already computed the answer. ``RESTATEMENT`` in particular is a member Kojutsu's
    module exists to make hard to skip, and renaming it to something like
    ``"agreeing_same_source"`` would remove exactly the word that says the agreement
    checks nothing.

    ``StrEnum`` for the reason every other vocabulary in this package is: the value is
    the string a report groups by, so a member needing ``.value`` to be printable is a
    member that gets printed wrong.
    """

    #: Both present and they disagree. The intent is not visible in the change.
    DIVERGENT = "divergent"
    #: Both present and consistent. Informative only across separate principals.
    CONCURRENT = "concurrent"
    #: Both present, same principal and model. Agreement carries no information.
    RESTATEMENT = "restatement"
    #: Only a declaration exists. Absence is reported, not treated as agreement.
    DECLARED_ONLY = "declared_only"
    #: Only a reconstruction exists.
    RECONSTRUCTED_ONLY = "reconstructed_only"
    #: Neither exists.
    NEITHER = "neither"


#: The phrases a comparison reports, **copied verbatim from
#: ``rationale_link.py:61-83``** for the same reason the enum is. Each names the thing
#: that is *not* being claimed, which is what makes the bucket a caveat rather than a
#: label. A paraphrase would be a second copy to keep true, and this one is the sentence
#: a reader meets before either rationale's prose -- see
#: :func:`tenbin.corpus.render.render` for where the ordering is enforced.
OUTCOME_NOTES: Final[Mapping[ComparisonOutcome, str]] = MappingProxyType(
    {
        ComparisonOutcome.DIVERGENT: (
            "The stated reason and the inferred one disagree, so the intent is not visible "
            "in the change. This is the finding, not a defect in either record."
        ),
        ComparisonOutcome.CONCURRENT: (
            "Both rationales agree, across separate principals, so this is a genuine second "
            "opinion rather than a restatement."
        ),
        ComparisonOutcome.RESTATEMENT: (
            "Both rationales come from the same principal and model. Agreement between them "
            "carries NO information: this is a restatement, not corroboration."
        ),
        ComparisonOutcome.DECLARED_ONLY: (
            "Only a stated rationale exists for this change. No reviewer rationalised it, "
            "which is an absence of evidence and not agreement."
        ),
        ComparisonOutcome.RECONSTRUCTED_ONLY: (
            "Only an inferred rationale exists for this change. Nothing stated what the "
            "author intended, so this is a guess about intent and is labelled as one."
        ),
        ComparisonOutcome.NEITHER: ("No rationale of either kind exists for this change."),
    }
)


#: Kojutsu's checkout, resolved **once, at import**. Binding it here rather than
#: inside the check is deliberate: ``tests/conftest.py`` repoints ``HOME`` at a temporary
#: directory for every test, so a lookup performed at call time would answer "I cannot
#: tell" from inside the very suite that exists to prove the two vocabularies agree --
#: and a check that silently degrades to a skip under test is not a check. ``Path.home``
#: reads the environment and touches nothing, so this costs no I/O at import.
KOJUTSU_CHECKOUT: Final[pathlib.Path] = pathlib.Path.home() / "Documents" / "kojutsu"


def outcome_values_are_kojutsus() -> bool:
    """Whether this copy of the vocabulary still equals Kojutsu's, as a set of values.

    **A function rather than an assertion in a test so that the check is callable from
    the CLI and from a report.** A test asserting against a hardcoded list of six
    strings would prove this enum equals a list somebody typed; it would not notice
    Kojutsu adding a seventh. This reads Kojutsu's own source and compares the two
    member sets, so upstream adding an outcome is a change this program is *told* about
    rather than one it drifts across.

    Returns ``False`` rather than raising when Kojutsu is not readable from this
    checkout -- the caller asked a yes/no question and "I cannot tell" is an honest
    answer to it, and a raise would make a documentation check into an outage for a
    program whose read seam is otherwise a pure function of a snapshot. A caller that
    must not conflate "drifted" with "absent" should check
    :data:`KOJUTSU_CHECKOUT` itself before asking.
    """
    source = _kojutsu_source()
    if source is None:
        return False
    return _kojutsu_outcomes(source) == {outcome.value for outcome in ComparisonOutcome}


def _kojutsu_source() -> str | None:
    """Kojutsu's rationale-link module as text, or ``None`` if it is not here.

    Read from the checkout this program was developed against rather than vendored, so
    the equality check compares against the real file rather than against a copy that
    could be edited to agree with itself. A repository that does not have Kojutsu
    alongside it has no way to answer the question and says so.
    """
    candidate = KOJUTSU_CHECKOUT / "src" / "kojutsu" / "core" / "rationale_link.py"
    try:
        return candidate.read_text(encoding="utf-8")
    except OSError:
        return None


def _kojutsu_outcomes(source: str) -> set[str]:
    """The values Kojutsu's enum holds, read out of its source text.

    Parsed rather than imported for one reason: importing would make Tenbin depend on
    Kojutsu being importable, and the read seam says the opposite in both directions.
    The pattern is anchored on the member assignments inside the class body, so a
    string that merely mentions an outcome elsewhere in the file is not mistaken for a
    member of it.
    """
    members = re.findall(r'^\s{4}([A-Z_]+)\s*=\s*"([a-z_]+)"\s*$', source, flags=re.MULTILINE)
    return {value for _, value in members}


#: The two sources a rationale can carry, as the store spells them. Kojutsu writes
#: them into the ``rationale_source`` key and projects the same word into the
#: ``rationale_declared`` / ``rationale_reconstructed`` tags
#: (``record.rationale_source_from_tags``). ``unknown`` is a third real value for a
#: rationale written before the axis existed, and it is neither of these.
DECLARED_SOURCE: Final[str] = "declared"
RECONSTRUCTED_SOURCE: Final[str] = "reconstructed"

#: The heading Kojutsu writes the reason under, and the marker of the attribution
#: block that follows it. Read rather than compared whole because the body is a
#: *rendering*: it repeats the principal and the model in prose, so comparing two bodies
#: in full would report a divergence for every pair of rationales that happened to be
#: written by different people -- which is every pair the figure is about. Verified
#: against ``kojutsu/src/kojutsu/core/tanseki_mapping.py:264-284``.
REASON_HEADING: Final[str] = "## Reason"
_NEXT_HEADING: Final[re.Pattern[str]] = re.compile(r"^##\s", flags=re.MULTILINE)


@dataclass(frozen=True)
class RationaleSummary:
    """One stored rationale, as the comparison needs to see it.

    The fields are the ones ``rationale_link.py:86-95`` uses, named as this package
    names them: ``declared_by`` rather than Kojutsu's ``RationaleSummary.declared_by``
    of the same name, ``model`` rather than the ``Stated``/``Unstated`` pair the record
    holds, because by the time a comparison reads it the distinction has already been
    made and an unstated model has to stay an unstated model rather than becoming
    ``None`` -- which ``compute_independence`` would read as a model named ``""``.
    """

    doc_id: str
    source: str
    text: str | None
    declared_by: str | None
    model: str | None
    revision: int


@dataclass(frozen=True)
class RationaleComparison:
    """The relationship between two rationales for one change.

    Both records rather than a verdict alone, for Kojutsu's reason: a reader holding
    only ``divergent`` has learned a conclusion and not the reasons, and holding both
    keeps the two sources visible, which is the whole point -- a merged view would let a
    reader be unable to tell what was stated from what was inferred.

    :attr:`not_reached` is the fields a response budget or a stored projection did not
    reach, named rather than dropped. A comparison that reported "no divergences found"
    after reaching two of five would be worse than one naming the three it did not.
    """

    outcome: ComparisonOutcome
    independence: Independence
    independence_reason: str
    note: str
    declared: RationaleSummary | None = None
    reconstructed: RationaleSummary | None = None
    not_reached: tuple[str, ...] = ()

    @property
    def is_informative(self) -> bool:
        """False when the outcome is a restatement or a bare absence.

        Kojutsu's own test at ``rationale_link.py:117-125``, kept because a caller
        filtering on it is asking a real question: "did this teach me anything the
        records did not already say independently?", and a restatement is the answer no.
        """
        return self.outcome not in (ComparisonOutcome.RESTATEMENT, ComparisonOutcome.NEITHER)


def compare_rationales(
    declared: RationaleSummary | None,
    reconstructed: RationaleSummary | None,
    *,
    not_reached: tuple[str, ...] = (),
) -> RationaleComparison:
    """Classify the relationship between a stated and an inferred rationale.

    **The four-way structure is Kojutsu's** (``rationale_link.py:132-199``) and is
    reproduced rather than simplified, because each branch is a different answer to a
    different question. A reader asking "did they agree?" must not be handed a
    ``NEITHER`` without being told there was nothing to agree about, and the two
    only-one cases are the two directions of the same absence and are not
    interchangeable: nobody reconstructed anything is not the same fact as nobody stated
    anything.

    The independence level is computed by :func:`compute_independence`, which mirrors
    Kojutsu's own function rather than inventing a scale. A comparison that implied
    more separation than the provenance supports would be inventing a vaguer version of
    an axis the codebase already has, and two axes disagreeing about the same pair of
    records is worse than one axis being conservative.
    """
    if declared is None and reconstructed is None:
        return RationaleComparison(
            outcome=ComparisonOutcome.NEITHER,
            independence=Independence.SELF_CERTIFIED,
            independence_reason="no rationale of either kind exists",
            note=OUTCOME_NOTES[ComparisonOutcome.NEITHER],
            not_reached=not_reached,
        )
    if declared is None:
        return RationaleComparison(
            outcome=ComparisonOutcome.RECONSTRUCTED_ONLY,
            independence=Independence.SELF_CERTIFIED,
            independence_reason="only an inferred rationale exists",
            note=OUTCOME_NOTES[ComparisonOutcome.RECONSTRUCTED_ONLY],
            reconstructed=reconstructed,
            not_reached=not_reached,
        )
    if reconstructed is None:
        return RationaleComparison(
            outcome=ComparisonOutcome.DECLARED_ONLY,
            independence=Independence.SELF_CERTIFIED,
            independence_reason="only a stated rationale exists",
            note=OUTCOME_NOTES[ComparisonOutcome.DECLARED_ONLY],
            declared=declared,
            not_reached=not_reached,
        )

    independence, reason = compute_independence(
        asker_account=declared.declared_by,
        asker_model=declared.model,
        answerer_account=reconstructed.declared_by,
        answerer_model=reconstructed.model,
    )
    agrees = _normalise(declared.text) == _normalise(reconstructed.text)
    if independence is Independence.SELF_CERTIFIED:
        # Agreement is uninformative and disagreement is the same mind reaching
        # different words. Either way these are not two accounts, and saying so is what
        # keeps a restatement from reading as a second opinion.
        outcome = ComparisonOutcome.RESTATEMENT
    elif agrees:
        outcome = ComparisonOutcome.CONCURRENT
    else:
        outcome = ComparisonOutcome.DIVERGENT

    return RationaleComparison(
        outcome=outcome,
        independence=independence,
        independence_reason=reason,
        note=OUTCOME_NOTES[outcome],
        declared=declared,
        reconstructed=reconstructed,
        not_reached=not_reached,
    )


def compute_independence(
    *,
    asker_account: str | None,
    asker_model: str | None,
    answerer_account: str | None,
    answerer_model: str | None,
) -> tuple[Independence, str]:
    """How far the two principals were from disagreeing, in Kojutsu's own terms.

    **Reproduces ``kojutsu.models.compute_independence`` (``models.py:205-233``)**
    against this package's :class:`~tenbin.measures.trust.Independence`, whose three
    members carry Kojutsu's names and values. The rule: a different posting account is
    ``INDEPENDENT`` regardless of model, because two parties are stronger evidence than
    two models; where the account is shared the comparison is on model; and an unstated
    model on either side is *not* treated as a match, because assuming two unknowns are
    the same manufactures a worse label than admitting the gap.

    The reason it is reproduced at all is the one this whole module is about. A second
    spelling of this axis would be a second truth about the same pair of records, and
    the drift would be invisible -- a comparison that reported ``MODEL_SEPARATED`` where
    Kojutsu reported ``SELF_CERTIFIED`` is a restatement misfiled as a second
    opinion, which is the exact misreading :attr:`RationaleComparison.is_informative`
    exists to prevent.
    """
    asker = (asker_account or "").strip().casefold()
    answerer = (answerer_account or "").strip().casefold()
    if asker and answerer and asker != answerer:
        return Independence.INDEPENDENT, "different posting accounts"
    asked_by = (asker_model or "").strip().casefold()
    answered_by = (answerer_model or "").strip().casefold()
    if not asked_by or not answered_by:
        return Independence.SELF_CERTIFIED, "same account; model not stated by both parties"
    if asked_by != answered_by:
        return Independence.MODEL_SEPARATED, "same account, different models"
    return Independence.SELF_CERTIFIED, "same account, same model"


def _normalise(text: str | None) -> str:
    """The comparison Kojutsu applies: case-folded with runs of whitespace collapsed.

    ``rationale_link.py:128-129``. Reproduced rather than using ``str.split`` directly so
    that a future change to how Kojutsu normalises has one place to be mirrored, and
    so the two implementations cannot disagree about what "the same words" means.
    """
    return " ".join((text or "").casefold().split())


#: The population, in the words the denominator renders. **The restriction is part of the
#: definition and is written into it** rather than left to an exclusion count, because a
#: rationale naming no repository-and-pull-request pair is not a change: there is no
#: change to exclude it from, and counting it in a change-shaped figure in any other unit
#: would break the invariant that buckets, exclusions and denominator add up. Verified
#: against ``kojutsu/src/kojutsu/core/tanseki_mapping.py:212-214``, where a rationale
#: with no pull request is addressed at ``pr-0``.
RATIONALE_POPULATION: Final[str] = (
    "changes in this read that carry at least one rationale naming both a repository and a "
    "pull request"
)

#: The two exclusions, both counted in changes. Each is a condition under which no
#: comparison could be made, and each is a different gap: one is a rationale written
#: before the source axis existed, the other is a stored document whose reason section
#: this reader could not find.
SOURCE_NOT_READABLE: Final[str] = (
    "a change whose rationales name neither a stated nor an inferred source, so neither side "
    "of the comparison exists: Kojutsu writes `unknown` for a rationale written before "
    "the source axis did"
)
REASON_NOT_READABLE: Final[str] = (
    "a change carrying both a stated and an inferred rationale where the reason text could "
    "not be read from either, so the two texts cannot be compared and the change is neither "
    "a divergence nor an agreement"
)

#: The two titles, each naming its own unit, because the group answers two questions over
#: one population and a reader handed either without its title holds a number with no
#: population.
RELATIONSHIP_TITLE: Final[str] = (
    "Changes by how their stated and inferred rationales relate, in Kojutsu's vocabulary"
)
COMPARED_TITLE: Final[str] = (
    "Changes carrying both a stated and an inferred rationale, which is how many comparisons "
    "the figure above actually made"
)
RELATIONSHIP_GROUP_TITLE: Final[str] = (
    "Declared versus reconstructed rationales, and the three questions that comparison does "
    "not answer"
)

#: The refusal, in the words the reader needs, and the sentence this whole measure exists
#: to make unavoidable. It is one string rather than three because the three are one
#: misreading arriving in three steps, and a reader who meets only the first of them has
#: already drawn the conclusion the figure exists to refuse.
COMPARISON_IS_NOT_A_VERDICT: Final[str] = (
    "Whether two people agreed, who was right, or what should be done about it. The "
    "comparison says whether two statements about a change agree, disagree, or come from the "
    "same mind. It does not say which is correct, whether the disagreement mattered, or what "
    "anyone should do about it. A divergence is the finding and not a defect in either "
    "record, and agreement between two rationales from the same principal and model is a "
    "restatement that carries no information at all rather than a second opinion."
)

#: The second misreading, which is separate from the first and is what a distribution of
#: these six values invites most readily. A reader summing the ``DIVERGENT`` rung into a
#: rate has produced a number about a corpus that this measure does not license.
NOT_A_DISAGREEMENT_RATE: Final[str] = (
    "A disagreement rate, a measure of review quality, or a measure of anybody's "
    "effectiveness. The buckets count changes, so a corpus with many divergences and a "
    "corpus with few are not comparable until you know how many changes in each could have "
    "been compared at all -- which is what the second figure reports. Whether a divergence "
    "on one particular change mattered is a question about that change and about one "
    "person's job, and nothing in this corpus would answer it."
)

#: The third misreading: that the figure is about the two texts rather than about the
#: store's rendering of them. A rationale's reason is read out of a projected Markdown
#: body, and a body Kojutsu re-renders differently tomorrow is the same finding.
THIS_IS_NOT_THE_UPSTREAM_COMPARISON: Final[str] = (
    "A reproduction of what Kojutsu computed. The reason text is read out of a projected "
    "document body rather than out of the field Kojutsu compares, so the two agree on most "
    "pairs and can disagree on any pair; where the reason section cannot be read the change "
    "is counted as an exclusion rather than compared on the attribution prose around it."
)


def rationale_relationship_claim(total: int, unreachable: int = 0) -> Claim:
    """The claim behind both figures, over the changes that carry a rationale.

    ``total`` is the whole population -- the changes that were bucketed *plus* the
    changes that could not be -- so that the denominator, the buckets and the exclusions
    are one number a reviewer can add up. Floor it at one rather than refusing zero,
    for the convention every subset denominator here follows: an empty read is an
    answer, and ``0/0`` is not one.

    ``unreachable`` is the number of rationale documents in this read that named no
    repository-and-pull-request pair. It is carried into the *denominator description*
    rather than into an exclusion count, because the figure's unit is the change and a
    document is not a change -- and because a restriction the reader cannot see is the
    silent-filter defect :class:`~tenbin.measures.base.DistributionFigure` exists to
    prevent. Naming it in the scope of the population is the sanctioned way to say it
    without changing the unit three figures have to agree on.
    """
    description = RATIONALE_POPULATION
    if unreachable:
        description += (
            f"; {unreachable} rationale "
            f"{'document names' if unreachable == 1 else 'documents name'} no repository and "
            "pull request pair and is in no figure here, which is a fact about this reader's "
            "ability to place it rather than about the corpus"
        )
    return Claim(
        slug=RATIONALE_RELATIONSHIP_SLUG,
        statement=(
            "A single implementation produces two rationales for a change: one stated by "
            "whoever did the work, and one inferred from the diff by a reviewer. Where this "
            "read holds both, the figures below report how the two relate, in the vocabulary "
            "Kojutsu defines for exactly that question."
        ),
        does_not_mean=(
            COMPARISON_IS_NOT_A_VERDICT
            + " "
            + NOT_A_DISAGREEMENT_RATE
            + " "
            + THIS_IS_NOT_THE_UPSTREAM_COMPARISON
        ),
        falsifier=(
            "Two rationale documents about the same change that agree on the text but carry "
            "different `rationale_source` values, or one change whose declared and "
            "reconstructed rationales are byte-identical bodies with different "
            "`rationale_revision` numbers. Either would mean the source axis and the revision "
            "chain describe different documents than the bodies do, and every bucket here "
            "would be computed over a pairing the store does not actually assert. This "
            "measure takes the highest revision per source per change and does not "
            "adjudicate a disagreement between them, which is the same posture the corpus "
            "layer takes towards a tag and an id that disagree."
        ),
        denominator=Denominator(description, max(1, total), size_noun="changes"),
        # Descriptive, and the temptation is real: a figure comparing a declared rationale
        # with a reconstructed one sounds like a comparison, and `comparative` would be
        # refused by the gate for want of an assignment mechanism. It is not one. The two
        # things being related are two *documents about one change*, not two populations over
        # an outcome, and no work was allocated between them -- there is nothing for a
        # mechanism to name. `comparative` would also invite the reader to look for a
        # counterfactual here, which is the confusion decision 002 exists to prevent.
        kind=ClaimKind.descriptive,
        # The floor, and never below it: a ticket is not a person and a change is not
        # either, so nothing in this figure can be computed per principal -- but the
        # population it aggregates over is people, and a narrower population than a team
        # is a ranking of principals wearing a different name.
        granularity=Granularity.team,
        source=f"Kojutsu {RELATIONSHIP_SOURCE}, which made the comparison reachable",
    )


@dataclass(frozen=True)
class RationaleRelationshipMeasure:
    """Changes by how their two rationales relate, and how many were actually compared.

    One outcome per change, taken from the highest-revision declared and reconstructed
    rationale that change holds. Two figures under one claim and therefore one
    denominator, because the second is a count over the same population rather than a
    separate question: a distribution of six buckets where the informative rungs can be
    small is not readable without knowing how many changes could have been compared at
    all, and that number is a fact about this read rather than about the corpus.

    Stateless and total. ``compute`` is a pure function of the snapshot, so a test builds
    one from a fixture and asserts on the figures.
    """

    #: Every rationale counts, whatever certainty its classification carries. A rationale
    #: carries no `record_kind` -- Kojutsu dispatches it through a different payload and
    #: never writes the key -- so requiring DETERMINED here would be a filter against a
    #: field this measure never reads, and a rationale whose tags were lost would drop out
    #: of the corpus for a reason that has nothing to do with why it is here.
    requires_certainty = requires_certainty

    @property
    def slug(self) -> str:
        """The identifier this measure is cited by."""
        return RATIONALE_RELATIONSHIP_SLUG

    def claim(self, snapshot: CorpusSnapshot) -> Claim:
        """The claim, over the changes this read placed a rationale on."""
        reading = _classify_changes(snapshot.records)
        return rationale_relationship_claim(reading.changes, reading.unplaceable)

    def compute(self, snapshot: CorpusSnapshot) -> Figure:
        """The bucketed relationship, and the count of changes a comparison was made on."""
        reading = _classify_changes(snapshot.records)
        claim = self.claim(snapshot)
        return FigureGroup(
            claim=claim,
            snapshot=snapshot,
            title=RELATIONSHIP_GROUP_TITLE,
            figures=(
                DistributionFigure(
                    claim=claim,
                    snapshot=snapshot,
                    title=RELATIONSHIP_TITLE,
                    values=reading.buckets,
                    excluded=reading.excluded,
                ),
                CountFigure(
                    claim=claim,
                    snapshot=snapshot,
                    title=COMPARED_TITLE,
                    value=reading.compared,
                ),
            ),
        )


@dataclass(frozen=True)
class _Reading:
    """One pass of a read, in the three shapes the measure needs.

    ``buckets``, ``excluded`` and :attr:`changes` are all counted in **changes**, which
    is what lets a reviewer add them up and is the invariant every other distribution in
    this package holds. :attr:`compared` is the three buckets a comparison was actually
    made on -- divergent, concurrent and restatement -- so the second figure can state
    the denominator the informative rungs are a fraction of.

    :attr:`unplaceable` counts *rationale documents* rather than changes and is therefore
    the one number here in a different unit. It is named in the claim's denominator
    description rather than in :attr:`excluded`, because a document is not a change and
    mixing the two units into one mapping would break the add-up invariant silently.
    """

    buckets: Mapping[str, int]
    excluded: Mapping[str, int]
    changes: int
    compared: int
    unplaceable: int


#: The outcomes a comparison was actually made on, as opposed to the three that report
#: an absence. Kept as one tuple so the second figure and the bucket keys cannot drift
#: apart -- a figure counting one set of rungs beside a distribution keyed on a different
#: one is two answers to "how many comparisons".
COMPARED_OUTCOMES: Final[tuple[ComparisonOutcome, ...]] = (
    ComparisonOutcome.DIVERGENT,
    ComparisonOutcome.CONCURRENT,
    ComparisonOutcome.RESTATEMENT,
)


def _classify_changes(records: Sequence[Record]) -> _Reading:
    """Every change that carries a rationale, bucketed by the relationship it holds.

    Two passes, and the order is load-bearing. The first indexes the highest revision
    per source per change, because a rationale is revised -- ``rationale_revises`` names
    what it supersedes -- and comparing a superseded draft against the current inference
    would manufacture divergences out of editorial history. The second walks the changes
    and compares what survived, so a change with a lone rationale lands in its only-one
    case rather than being dropped (see the module docstring).

    A change whose rationales all carry an unreadable ``rationale_source`` is counted as
    an exclusion rather than quietly absent from the denominator: it is in the
    population by the definition, and a change that cannot be bucketed has to say so.
    """
    held: dict[tuple[str, int], dict[str, RationaleSummary]] = {}
    placed: set[tuple[str, int]] = set()
    unplaceable = 0
    for record in records:
        if not record.is_rationale:
            continue
        repo, pr = record.repo, record.pr
        if pr is None or not named_repository(repo):
            # A half key joins to nothing, and Kojutsu's placeholder repository would
            # join every unstated repository in this read into one change. Neither is a
            # comparison, so neither is a bucket; both are counted so the claim can name
            # them rather than have the gap read as a change that reconstructed nothing.
            unplaceable += 1
            continue
        assert repo is not None
        key = (repo.strip(), pr)
        placed.add(key)
        source = record.rationale_source
        if source not in (DECLARED_SOURCE, RECONSTRUCTED_SOURCE):
            continue
        incumbent = held.setdefault(key, {}).get(source)
        revision = record.rationale_revision or 0
        if incumbent is None or revision > incumbent.revision:
            held[key][source] = _summary(record, source)

    buckets: dict[str, int] = {outcome.value: 0 for outcome in ComparisonOutcome}
    excluded: dict[str, int] = {}
    source_gap = sum(1 for key in placed if not held.get(key))
    text_gap = 0
    for key in placed:
        summaries = held.get(key)
        if not summaries:
            # Already counted as a source gap. Bucketing it as ``NEITHER`` would say the
            # change has no rationale at all, which is false -- it has one this reader
            # cannot place on either side of the comparison, and the two sentences are
            # the difference between a corpus gap and a reader's.
            continue
        comparison = compare_rationales(
            summaries.get(DECLARED_SOURCE), summaries.get(RECONSTRUCTED_SOURCE)
        )
        if comparison.outcome in COMPARED_OUTCOMES:
            sides = (comparison.declared, comparison.reconstructed)
            if any(side is None or side.text is None for side in sides):
                # Both sides exist and one carries no readable reason, so the single
                # question a comparison asks -- do these texts agree -- has no answer.
                # Bucketing it as divergent would report an absence as a finding, which
                # is the misreading this module exists to prevent wearing our own tool.
                text_gap += 1
                continue
        buckets[comparison.outcome.value] += 1

    if source_gap:
        excluded[SOURCE_NOT_READABLE] = source_gap
    if text_gap:
        excluded[REASON_NOT_READABLE] = text_gap
    frozen_buckets = freeze_counts(buckets, "rationale.buckets")
    return _Reading(
        buckets=frozen_buckets,
        excluded=freeze_counts(excluded, "rationale.excluded"),
        # The population, which is the buckets plus the two gaps: every placed change is
        # in exactly one of them, so `counted + excluded_total == denominator.size` holds
        # in this figure's own unit rather than only in the abstract.
        changes=len(placed),
        compared=sum(frozen_buckets[outcome.value] for outcome in COMPARED_OUTCOMES),
        unplaceable=unplaceable,
    )


def _summary(record: Record, source: str) -> RationaleSummary:
    """One stored rationale as the comparison sees it, with its reason read out."""
    return RationaleSummary(
        doc_id=record.doc_id,
        source=source,
        text=reason_text(record.content),
        declared_by=record.declared_by,
        model=_stated_model(record.declared_by_model),
        revision=record.rationale_revision or 0,
    )


def _stated_model(model: Stated[str] | Unstated) -> str | None:
    """The model a rationale states, or ``None`` when it states none.

    An unstated model stays ``None`` rather than becoming ``""`` because
    :func:`compute_independence` reads a blank model as *unstated* and a model named
    ``""`` as *named* -- and the second is a model nobody declared.
    """
    return model.value if isinstance(model, Stated) else None


def reason_text(body: str) -> str | None:
    """The reason a projected rationale body carries, or ``None`` if it carries none.

    **Read rather than compared whole, and the reason is the body.** Kojutsu renders a
    rationale as a ``## Reason`` section followed by an ``## Attribution`` block
    repeating the principal, the model and the source
    (``tanseki_mapping.py:264-284``). Comparing two bodies in full would therefore report a
    divergence for every pair of rationales written by different people -- which is
    precisely the population this figure is about -- and would have turned the measure
    into a restatement detector wearing the name of a disagreement detector.

    ``None`` rather than the whole body when the heading is absent, because the fallback
    would be the comparison this module refuses to make, and the caller turns a ``None``
    into a counted exclusion instead. The section is found by its next ``##`` heading
    rather than by a fixed line count, so a writer that adds a field to the attribution
    block does not change what is compared.
    """
    lines = body.splitlines()
    for index, line in enumerate(lines):
        if line.strip().casefold() != REASON_HEADING.casefold():
            continue
        collected: list[str] = []
        for follower in lines[index + 1 :]:
            if _NEXT_HEADING.match(follower):
                break
            collected.append(follower)
        reason = "\n".join(collected).strip()
        return reason or None
    return None
