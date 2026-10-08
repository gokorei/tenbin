"""The seam's two obligations: what must never cross it, and what must not be lost on it.

**This module checks the join, not only what crossed it.** Kojutsu projects the
question registry into Tanseki so a downstream program can read decision-request lifecycle
over the same seam it uses for everything else
(``docs/decisions/001-registry-through-tanseki.md``). The projection is an allowlist of
columns, and two are deliberately excluded because publishing either would be a mistake
that no later reader could undo.

``claim_token`` is the more dangerous of the two and the reason this module exists. It is
``secrets.token_urlsafe(32)`` and it is the ``WHERE``-clause guard on releasing a claim,
so anyone holding it can release a claim somebody else holds. Tanseki is readable over MCP
by agents, so a projected token is not a field, it is a capability handed to every reader
of the store -- including the unattended workers whose claims it would let them release.
``last_error`` carries internal exception detail and has no analytical value.

**The second check is the same seam read from the other direction.** The registry and the
corpus are written by two different code paths, and they can disagree: a request the
registry places at ``answered`` with no document behind it, or a capture naming a request
the registry placed somewhere else entirely. That check lives beside the key check rather
than in a module of its own because it is the same question asked of the same crossing --
*is the store what the two writers agree it is?* -- and because a reader who has just been
told what this module is about, which documents an agent may read, is the reader who needs
to hear that the store also silently failed to keep one.

**The check is applied to what arrived, not to how it was written.** Tenbin does not
read Kojutsu's source and does not model the projection; it knows which names must
never appear in a document an agent can read, and it looks for those, and it knows that a
request and an answer are supposed to name each other. That is the whole of each check,
and it is why they still work if the projection is reimplemented: both prohibitions are
properties of the seam rather than of one writer.

**What this module notably does not do.** It does not refuse to read a corpus that
contains one of either shape. A published token is a fact about the store, and a registry
row whose answer is not here is a fact about the store, and hiding the corpus because one
is present would replace a finding a reader can act on with an absence they cannot. It
produces a :class:`~tenbin.measures.base.Finding` naming the key and the document, or
naming the question id and which side of the join is short, which is what the corpus layer
already uses for a corpus whose trustworthiness is in question. It also does not attempt
to check Kojutsu's projection source: Tenbin reads a store, and a check that required
reading the writer would be a check that stopped working the day the writer moved. And it
does not turn the disagreement into a rate, which is the specific thing it exists to
prevent -- a lost answer rendered as a percentage of a corpus is the one transformation
that would make this check worse than silence.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Final

from tenbin.claims.mechanism import Granularity
from tenbin.claims.model import Claim, ClaimKind, Denominator
from tenbin.corpus.record import Record
from tenbin.corpus.snapshot import CorpusSnapshot
from tenbin.measures.base import CountFigure, Figure, FigureGroup, Finding
from tenbin.measures.decision_requests import (
    ANSWERED_CAPTURE_POPULATION,
    join_questions_and_captures,
)

#: The frontmatter keys that must never appear in a projected decision request, and
#: why each one is prohibited rather than merely discouraged.
#:
#: ``claim_token`` is the more dangerous of the two and the reason this module exists.
#: It is a ``secrets.token_urlsafe(32)`` value and it is the ``WHERE``-clause guard on
#: releasing a claim, so anyone holding it can release a claim that somebody else
#: holds -- which is not a leak of information but a loss of exclusivity over work in
#: flight. Tanseki is readable over MCP by agents, so a token in a document is a
#: capability handed to every reader of the store, and the store is the wrong place to
#: keep a capability that is also a credential.
#:
#: ``last_error`` is prohibited for a smaller and separate reason: it carries internal
#: exception detail, and a question row's error text is the field most likely to carry
#: a filesystem path or a fragment of untrusted upstream text. It has no analytical
#: value whatever, so nothing is lost by refusing it and something is lost by keeping it.
FORBIDDEN_PROJECTION_KEYS: Final[frozenset[str]] = frozenset({"claim_token", "last_error"})

#: The finding's slug. A finding is a claim about the corpus like any other, so it is
#: cited like one and a report that hits it prints a caveat rather than a bare sentence.
PROJECTION_SAFETY_SLUG: Final[str] = "corpus.projection_carries_a_prohibited_key"

#: The slug a clean projection-safety check reports under. A slug of its own rather than
#: a reused heading, for the reason the lifecycle-shape check has one: "the check ran and
#: found nothing" and "the check did not run" must not render as the same absence.
PROJECTION_SAFETY_CLEAN_SLUG: Final[str] = "corpus.projection_safety_checked"

#: The population the finding is about: the projected decision requests this read
#: enumerated. Not the whole corpus -- a finding about one namespace does not get to
#: borrow the population of every other kind of record.
PROJECTION_POPULATION: Final[str] = "projected decision requests in this read"

#: How many document ids the finding names before it stops naming them. An operator has
#: to be able to go and revoke the thing, and every offending document is findable by
#: re-running the scan, so a corpus with thousands of them should read as a count rather
#: than as a paragraph. The count of the rest is stated rather than dropped, which is the
#: difference between a bounded list and a censored one.
MAX_DOCUMENTS_NAMED: Final[int] = 10

#: The cause, named with the alternatives rather than settled on, because this is a
#: corpus and not a debugger. An allowlist is a list somebody maintains; a maintained
#: list that has grown is the likeliest explanation and it is still a hypothesis.
SUSPECTED_CAUSE: Final[str] = (
    "Kojutsu's projection is an allowlist rather than a deny-list, which is the right shape "
    "and fails closed only for as long as the code naming it is not edited. The likeliest "
    "explanation is therefore that the allowlist grew: a migration adding a column to the "
    "projected set, or a writer changed to splat the registry row instead of naming its "
    "columns. Other causes produce the same document and are not excluded by anything here: "
    "a document hand-edited in Tanseki, or a projection written by a different version of "
    "Kojutsu than the one that wrote the rest of this collection. Nothing in the store "
    "distinguishes them."
)

#: What Tenbin is and is not in a position to say. Required, non-blank, and the reason
#: this is a finding rather than a diagnosis.
CANNOT_CONFIRM: Final[str] = (
    "Tenbin read documents out of a store and did not read Kojutsu's projection source at "
    "read time, run its code, or inspect the registry, so this is an observation about what "
    "arrived rather than a diagnosis of how it got there. The scan sees only the documents this "
    "read enumerated and only the keys their frontmatter carried: a document the walk never "
    "reached, a key renamed before it was written, and a value held anywhere other than the "
    "frontmatter are each invisible to it, and a clean result on a truncated read is a clean "
    "result over the part that was read."
)


def forbidden_keys_claim(scanned: int) -> Claim:
    """The claim a projection-safety finding is attached to, over the requests scanned.

    Built by a function rather than held as a constant because the denominator is the
    number of projected decision requests this read enumerated, which is a fact about
    the read rather than about the check. A claim carrying a placeholder population
    would be a claim about a corpus nobody described.
    """
    return Claim(
        slug=PROJECTION_SAFETY_SLUG,
        statement=(
            "This read found a projected decision request carrying a key that must never be "
            "published, so the store holds a capability in a document any agent can read."
        ),
        does_not_mean=(
            "That the claim was used, or that anything has gone wrong yet. It means the "
            "capability is readable, which is the condition under which it could be used by "
            "anyone with read access to the collection -- including every agent on the other "
            "side of this seam."
        ),
        falsifier=(
            "A later read of the same collection in which no projected decision request carries "
            "the key. That would make this a statement about what was in the store when it was "
            "read rather than about the store, and the underlying capability would still have "
            "existed until the document was rewritten."
        ),
        denominator=Denominator(PROJECTION_POPULATION, max(1, scanned)),
        kind=ClaimKind.descriptive,
        granularity=Granularity.corpus,
    )


def scan_for_forbidden_keys(
    records: Iterable[Record],
    *,
    snapshot: CorpusSnapshot,
) -> Finding | None:
    """The finding if a projected decision request carries a prohibited key, else ``None``.

    Checks **what arrived**, not what ought to have arrived. The population is every
    record in this read that is a projected decision request -- taken as the union of
    the id namespace and the tag projection, the way
    :attr:`~tenbin.corpus.record.Record.is_question` states it -- and the check is
    whether any of their frontmatter keys is in
    :data:`FORBIDDEN_PROJECTION_KEYS`. There is no second copy of Kojutsu's column
    list here to compare against, because a second copy is precisely the thing that
    would drift.

    Silent on a clean corpus, which is the whole design: a check that reports on every
    document it read would be a check nobody reads, and one that reported on a corpus
    without a prohibited key would be indistinguishable from a check that never ran.

    ``snapshot`` is required and keyword-only even though the scan needs nothing else
    from it, because the *finding* does: a :class:`~tenbin.measures.base.Finding`
    carries the read it was observed over, which is what makes its completeness
    sentence renderable. A detector callable with records alone could return a finding
    with no statement about whether the corpus behind it was whole.

    Pure: the same records in give the same finding or the same ``None`` out. No
    clock, no store, no state.
    """
    offenders: dict[str, list[str]] = defaultdict(list)
    scanned = 0
    for record in records:
        if not record.is_question:
            continue
        scanned += 1
        for key in FORBIDDEN_PROJECTION_KEYS:
            if key in record.frontmatter:
                offenders[key].append(record.doc_id)
    if not offenders:
        return None
    return _finding(
        snapshot,
        offenders={key: tuple(sorted(ids)) for key, ids in offenders.items()},
        scanned=scanned,
    )


def _finding(
    snapshot: CorpusSnapshot,
    *,
    offenders: dict[str, tuple[str, ...]],
    scanned: int,
) -> Finding:
    """Assemble the finding, naming the keys and the documents without naming the values.

    The values are the capability, so the observation names the key and the document
    and stops there. A finding is rendered into a report, and a report is somewhere a
    token ends up being copied from.
    """
    observed = "; ".join(
        f"`{key}` is carried by {len(documents)} document(s): {_named(documents)}"
        for key, documents in sorted(offenders.items())
    )
    return Finding(
        claim=forbidden_keys_claim(scanned),
        snapshot=snapshot,
        title="A projected decision request carries a key that must never be published",
        what_was_observed=(
            f"{observed}. These keys are among {', '.join(sorted(FORBIDDEN_PROJECTION_KEYS))}, "
            "which may not appear "
            f"in a document any agent can read; {scanned} projected decision request(s) were "
            "checked and the rest of this read carries no projected request."
        ),
        suspected_cause=SUSPECTED_CAUSE,
        where=None,
        cannot_confirm=CANNOT_CONFIRM,
    )


def _named(documents: tuple[str, ...]) -> str:
    """The document ids, bounded, with the number not named rather than the number dropped."""
    named = ", ".join(documents[:MAX_DOCUMENTS_NAMED])
    remaining = len(documents) - MAX_DOCUMENTS_NAMED
    if remaining <= 0:
        return named
    return f"{named}, and {remaining} more this finding did not name"


def _clean_projection_safety_claim(scanned: int) -> Claim:
    """The claim attached to a projection-safety check that found nothing.

    A distinct claim from the finding's, for the reason
    :func:`tenbin.measures.integrity._clean_event_shape_claim` is a distinct one: the
    two say opposite things, and a report that reused one slug for both would let a
    reader match them up and conclude the clean one refutes the finding. It is still a
    claim -- a check that ran and found nothing is a statement about the corpus -- and
    it is still the claim whose falsifier is the document that would have produced a
    finding instead.
    """
    return Claim(
        slug=PROJECTION_SAFETY_CLEAN_SLUG,
        statement=(
            "No projected decision request in this read carries a key that must never be "
            "published, so the store holds no claim token and no internal error text in a "
            "document any agent can read."
        ),
        does_not_mean=(
            "That the projection is correct, or that it is an allowlist, or that a document "
            "outside this read is safe. This is the absence of one specific shape -- a "
            "prohibited key in a projected decision request -- in one namespace on one read, "
            "and a projection that changed afterwards, or a collection nobody read, would pass "
            "it."
        ),
        falsifier=(
            "A read of the same collection in which a projected decision request carries "
            "`claim_token` or `last_error`. That document is what the check looks for, and "
            "seeing it would make this statement false for the period it was in the store."
        ),
        denominator=Denominator(PROJECTION_POPULATION, max(1, scanned)),
        kind=ClaimKind.descriptive,
        granularity=Granularity.corpus,
    )


class ProjectionSafetyMeasure:
    """Run :func:`scan_for_forbidden_keys` as a measure, so a report cannot skip it.

    **This class exists for the same reason
    :class:`~tenbin.measures.integrity.LifecycleShapeMeasure` does.** A check that
    nothing invokes is a function that looks finished, which is the worse of the two
    states a finished module can be in. The refusal catalogue used to *promise* that
    ``claim_token`` would never be projected and nothing checked it, so a projection
    that published one would have rendered as a clean corpus with a footnote
    reassuring the reader that it could not happen. It is wired here, into
    :func:`tenbin.measures.default_measures`, for the same reason the lifecycle-shape
    check is.

    A clean result is a :class:`~tenbin.measures.base.CountFigure` of zero and not an
    empty section, because a reader who cannot distinguish "checked, and nothing
    crossed" from "nothing here" has been handed an absence where a fact belongs. The
    count is zero, the claim says the check ran, and the falsifier names the document
    that would have produced a finding instead.
    """

    __slots__ = ()

    slug = PROJECTION_SAFETY_SLUG
    #: Same reason as ``LifecycleShapeMeasure``: a corpus carrying a prohibited key is
    #: not a corpus whose other figures mean what they appear to mean.
    about_the_capture_system = True
    #: Every projected decision request counts, whatever certainty its classification
    #: carries. A request whose kind had to be recovered rather than read is *more*
    #: interesting to this check than one whose kind was stated, not less, so requiring
    #: ``DETERMINED`` here would narrow exactly the population a hand-edited document
    #: would land in.
    requires_certainty = None

    def claim(self, snapshot: CorpusSnapshot) -> Claim:
        """The claim this measure answers, over the projected requests the read held.

        The denominator is the number of projected decision requests enumerated, so a
        corpus holding none reports over zero and the claim says the check had nothing
        to look at -- a different sentence from "the check found nothing", and the one
        that is true when the projection has not been run yet.
        """
        finding = scan_for_forbidden_keys(snapshot.records, snapshot=snapshot)
        if finding is not None:
            return finding.claim
        return _clean_projection_safety_claim(self._question_count(snapshot))

    def compute(self, snapshot: CorpusSnapshot) -> Figure:
        """The finding, or a count of zero with a claim that says the check ran."""
        finding = scan_for_forbidden_keys(snapshot.records, snapshot=snapshot)
        if finding is not None:
            return FigureGroup(
                claim=finding.claim,
                snapshot=snapshot,
                title=finding.title,
                figures=(finding,),
            )
        return CountFigure(
            claim=_clean_projection_safety_claim(self._question_count(snapshot)),
            snapshot=snapshot,
            title="Projected decision requests carrying a key that must never be published",
            value=0,
        )

    @staticmethod
    def _question_count(snapshot: CorpusSnapshot) -> int:
        return sum(1 for record in snapshot.records if record.is_question)


#: The slug a question-and-capture disagreement is published under, and the slug a
#: clean check of the same thing publishes under. Two slugs for the reasons
#: :data:`PROJECTION_SAFETY_CLEAN_SLUG` states: a reader must be able to tell "the check
#: ran and the two writers agree" from "the check did not run", and matching a clean
#: result against a finding would let them conclude the clean one refutes it.
QUESTION_CAPTURE_AGREEMENT_SLUG: Final[str] = "corpus.question_and_capture_disagree"
QUESTION_CAPTURE_AGREEMENT_CLEAN_SLUG: Final[str] = "corpus.question_capture_agreement_checked"

#: The three observations, one sentence each, written so the reader is told *which side
#: of the join is missing what* rather than only that the two sides differ. "They
#: disagree" is not actionable; "the registry says answered and the corpus holds no
#: document for it" names a check somebody can go and run. Kept as constants so the
#: findings and the tests that assert on them cite one string, and so a reword that
#: dropped the actionable half would fail a test rather than a reader.
ANSWERED_WITHOUT_CAPTURE: Final[str] = (
    "The registry says {count} request(s) reached answered and this read holds no capture "
    "naming {them}: the registry says answered and the corpus has no document for it. "
    "Kojutsu calls a queued row the only copy of a human's answer, so a row lost "
    "between the registry and the store is a copy of an answer that exists nowhere else."
)
CAPTURE_WITHOUT_ANSWERED: Final[str] = (
    "For {count} request(s) the registry says failed or superseded and this read holds a "
    "capture naming {them} anyway: the corpus holds a record for a question the registry "
    "says reached a terminal state elsewhere, and the two write paths disagree about what "
    "happened to it."
)
UNMATCHED_CAPTURE_OBSERVED: Final[str] = (
    "{count} capture record(s) name a question id this read holds no projected request for: "
    "the corpus holds answers the registry projection has not delivered."
)

#: The causes, named with their alternatives rather than settled on, because every one of
#: the three observations is consistent with more than one. A projection that is merely
#: behind is the least alarming reading of the third and is a documented property of the
#: seam rather than a defect, and saying so is the difference between naming candidates
#: and naming a cause.
DISAGREEMENT_CAUSES: Final[str] = (
    "The three observations have three causes and none of them can be chosen between from "
    "inside the store. A registry row with no document behind it is either a row the outbox "
    "lost, a document deleted after it was written, or a document this read never reached. A "
    "capture for a request the registry failed or superseded is either an answer delivered "
    "after the request was closed out, a status the projection wrote late, or a question id "
    "that was reused. A capture naming no projected request is the least alarming of the "
    "three: decision 001 calls the projection eventually consistent, so a read taken while "
    "an outbox is draining looks exactly like this, and so does a walk this program stopped "
    "before the question arrived. Read the completeness sentence on the figure before "
    "treating any of them as a loss."
)

#: What Tenbin is and is not in a position to say, and the specific way this check is
#: weaker than the key scan beside it. Every disagreement here was inferred from the
#: *absence* of a document: no writer said it failed, and a store says nothing about what
#: it was never asked for. That is a statement about what arrived and not a diagnosis of
#: how it got there, which is exactly what a blank ``cannot_confirm`` would have hidden.
AGREEMENT_CANNOT_CONFIRM: Final[str] = (
    "Tenbin read documents out of a store and did not read Kojutsu's outbox, run either "
    "writer, or observe the registry, so this is an observation about what arrived rather "
    "than a diagnosis of how it got there. Every line of it was inferred from the absence of "
    "a document and not from an error report: no writer reported a failure, the store says "
    "nothing about what it was not asked for, and a projection that is simply behind or a "
    "walk that stopped early produces the same picture as one that lost a row. On a "
    "truncated read this finding is a finding about the part that was read."
)


def question_capture_agreement_claim(terminal: int) -> Claim:
    """The claim a disagreement finding is attached to, over the terminal requests.

    Built by a function for the reason :func:`forbidden_keys_claim` is: the denominator
    is a fact about the read rather than about the check, and a claim carrying a
    placeholder population would be a claim about a corpus nobody described. The
    population is the projected requests that reached a terminal state, because that is
    the population both sides of the join can be compared over -- a capture is not in it,
    and the whole of decision 001 is that it must not be.
    """
    return Claim(
        slug=QUESTION_CAPTURE_AGREEMENT_SLUG,
        statement=(
            "This read found the question registry and the captured corpus disagreeing about "
            "the same questions, so at least one of the two write paths lost something and "
            "the store is not the union of what the writers believe they wrote."
        ),
        does_not_mean=(
            "That any particular answer is missing rather than merely unread from this "
            "collection, and not that the projection is broken. A capture can be absent "
            "because the walk did not reach it or because the answer has not been delivered "
            "yet, and a capture can exist for a question the registry closed out early. What "
            "is established is that the two sides do not correspond in this read, which is "
            "the fact a reader has to resolve before trusting either side alone. It says "
            "nothing whatever about the quality of any answer on either side."
        ),
        falsifier=(
            "A later read of the same collection in which every request the registry places "
            "at answered has a capture naming it and every capture names a request this "
            "read holds. That would make this a statement about the moment of the earlier "
            "read rather than about the corpus, and the outstanding question is whether the "
            "two write paths ever converge -- a question only a second read can answer."
        ),
        denominator=Denominator(ANSWERED_CAPTURE_POPULATION, max(1, terminal)),
        kind=ClaimKind.descriptive,
        granularity=Granularity.corpus,
    )


def scan_question_capture_agreement(
    records: Iterable[Record],
    *,
    snapshot: CorpusSnapshot,
) -> Finding | None:
    """The finding if the registry and the corpus disagree, or ``None`` if they do not.

    Silent on a corpus where they correspond, for the reason
    :func:`scan_for_forbidden_keys` is: a check that reports on every read is a check
    nobody reads, and one that reported on a corpus holding no projected requests at all
    would be indistinguishable from a check that never ran.

    Pure, and the join it reads is pure: the same records in give the same finding or the
    same ``None`` out. No clock, no store, no state.
    """
    join = join_questions_and_captures(records)
    if not join.disagrees:
        return None
    observed: list[str] = []
    if join.answered_without_capture:
        observed.append(
            ANSWERED_WITHOUT_CAPTURE.format(
                count=len(join.answered_without_capture),
                them=_named(join.answered_without_capture),
            )
        )
    if join.capture_without_terminal:
        observed.append(
            CAPTURE_WITHOUT_ANSWERED.format(
                count=len(join.capture_without_terminal),
                them=_named(join.capture_without_terminal),
            )
        )
    if join.unmatched_captures:
        observed.append(UNMATCHED_CAPTURE_OBSERVED.format(count=join.unmatched_captures))
    ids = tuple(sorted((*join.answered_without_capture, *join.capture_without_terminal)))
    return Finding(
        claim=question_capture_agreement_claim(join.terminal_questions),
        snapshot=snapshot,
        title="The question registry and the captured corpus disagree about the same questions",
        what_was_observed=" ".join(observed),
        suspected_cause=DISAGREEMENT_CAUSES,
        where=f"question ids: {_named(ids)}" if ids else None,
        cannot_confirm=AGREEMENT_CANNOT_CONFIRM,
    )


def _clean_agreement_claim(terminal: int) -> Claim:
    """The claim attached to an agreement check that found nothing.

    A distinct claim from the finding's, for the reason
    :func:`_clean_projection_safety_claim` is one: the two say opposite things and a
    reader who could match them up would conclude the clean one refutes the finding. It
    is still a claim, because a check that ran and found the two write paths agreeing is
    a fact about the corpus, and it is still a claim whose falsifier is the observation
    that would have produced a finding instead.
    """
    return Claim(
        slug=QUESTION_CAPTURE_AGREEMENT_CLEAN_SLUG,
        statement=(
            "In this read the question registry and the captured corpus agree: every request "
            "the registry placed at answered has a capture naming it, and every capture names "
            "a request this read holds."
        ),
        does_not_mean=(
            "That the delivery path is healthy in general, or that no answer has ever been "
            "lost. This is the absence of one specific disagreement in one collection on one "
            "read, and the projection is eventually consistent by design, so a read taken "
            "while an outbox drains will look clean and a read taken mid-drain will not."
        ),
        falsifier=(
            "A read of the same collection in which a request the registry places at answered "
            "has no capture naming it, or a capture names a request the read does not hold. "
            "Either observation is what this check looks for, and seeing it would make this "
            "statement false for the period it was true."
        ),
        denominator=Denominator(ANSWERED_CAPTURE_POPULATION, max(1, terminal)),
        kind=ClaimKind.descriptive,
        granularity=Granularity.corpus,
    )


class QuestionCaptureAgreementMeasure:
    """Run :func:`scan_question_capture_agreement` as a measure, so a report cannot skip it.

    **This class exists for the reason
    :class:`~tenbin.measures.integrity.LifecycleShapeMeasure` does.** A correct function
    nothing invokes is a function that looks finished, which is the worse of the two
    states a finished module can be in, and the failure this check guards against is
    silent in a way almost nothing else here is: a lost outbox row renders as a corpus
    that is merely slightly smaller, with no error, no gap in the walk and no complaint
    from any writer. Wire it, or the whole seam check is decoration.

    A clean result is a :class:`~tenbin.measures.base.CountFigure` of zero under a slug
    of its own, for the same reason the key scan's is: a reader who cannot distinguish
    "the two write paths agreed" from "nobody checked" has been handed an absence where a
    fact belongs. Silence on the finding and a fact on the clean path are not in tension
    -- the detector returns ``None`` and the measure renders the count, which is exactly
    the split :class:`ProjectionSafetyMeasure` makes between ``scan_for_forbidden_keys``
    and its own ``compute``.
    """

    __slots__ = ()

    slug = QUESTION_CAPTURE_AGREEMENT_SLUG
    #: A registry row whose answer is not in the store is a statement about the capture
    #: path rather than about the work it captured, which is the question
    #: ``tenbin --integrity-only`` exists to answer. Same reason as
    #: :class:`ProjectionSafetyMeasure` and :class:`~tenbin.measures.integrity.LifecycleShapeMeasure`.
    about_the_capture_system = True
    #: Every request counts, whatever certainty its classification carries. A request read
    #: from its tags and a request read from Kojutsu's own projection are the same
    #: request, and the check is about whether the two writers correspond -- which a
    #: stricter filter would only narrow, by discarding the hand-edited document that is
    #: exactly what this check is looking for.
    requires_certainty = None

    def claim(self, snapshot: CorpusSnapshot) -> Claim:
        """The claim this measure answers, over the terminal requests the read held.

        The denominator is the projected requests that reached a terminal state, so a
        corpus holding none reports over zero and the claim says the check had nothing to
        compare -- a different sentence from "the two sides agree".
        """
        finding = scan_question_capture_agreement(snapshot.records, snapshot=snapshot)
        if finding is not None:
            return finding.claim
        return _clean_agreement_claim(
            join_questions_and_captures(snapshot.records).terminal_questions
        )

    def compute(self, snapshot: CorpusSnapshot) -> Figure:
        """The finding, or a count of zero with a claim that says the check ran."""
        finding = scan_question_capture_agreement(snapshot.records, snapshot=snapshot)
        if finding is not None:
            return FigureGroup(
                claim=finding.claim,
                snapshot=snapshot,
                title=finding.title,
                figures=(finding,),
            )
        return CountFigure(
            claim=_clean_agreement_claim(
                join_questions_and_captures(snapshot.records).terminal_questions
            ),
            snapshot=snapshot,
            title="Questions the registry and the captured corpus disagree about",
            value=0,
        )
