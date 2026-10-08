"""Reading a stored document into a record, and recovering the kind an old one lost.

Kojutsu writes a ``record_kind`` on every record and derives the tag set from
it, so the writer states the kind twice and the two cannot drift. When the key is
present :func:`classify` reads it and the kind is determined; when it is absent --
a document written before the key existed, or a rationale, which is dispatched
through a different payload and never carried it -- the kind is recovered from the
tag set by :func:`classify_by_tags`, which is a weaker statement about the same
record. :meth:`Record.from_document` is the single place a document becomes a
:class:`Record`, and :attr:`Record.record_kind` records which of the two happened.

**The tag set is a projection of the kind, and one case is a guess.** Kojutsu
derives tags from the kind it already holds, so ``["review", "inline_comment"]``
is the writer stating what it is, and the reader recovering that is not inference
in any interesting sense. The exception is the answer: a record with no tags is a
record whose kind the writer never projected, so "no tags" could be an answer
written by someone who named no agent, or any other record whose tags were lost.
:func:`classify_by_tags` returns that case with :data:`Certainty.INFERRED` and the
evidence ``("no tags",)`` rather than reporting a determination, because the
difference between "the store says this is an answer" and "nothing here said so"
is the difference between a fact and a conclusion drawn from a gap.

**Six keys describe the change a record is about, and are read as events.**
``record_kind``, ``review_id``, ``change_author_account``, ``pr_opened_at``,
``pr_outcome`` and ``pr_merged_at``. Every one is absent rather than defaulted
where the payload carried no value, so a reader cannot mistake a placeholder for
a value somebody stated. Two of them carry Kojutsu's own caution in their
names and it is repeated on the fields: a ``change_author_account`` is the login
GitHub reports opened the pull request -- not a person, not a bot, not a
statement about what they were doing -- and a ``pr_outcome`` is whether GitHub
reported a merge, not whether the change was any good.

**``files`` is a bounded list whose overflow is written into the list as a marker.**
Kojutsu caps the list at fifty paths and, where a change touched more, appends a
literal sentence such as ``"3 more files not listed"`` rather than dropping the rest in
silence. That is the right call -- a quietly shortened list reads as a complete
description of a change -- and it has a consequence for a reader: the list is not a list
of paths. The marker sits exactly where a path would sit, and a reader that counted it
would report a file called ``"3 more files not listed"`` as a file this corpus holds
evidence about. So :attr:`Record.files` drops it and :attr:`Record.files_truncated` says
it happened. What no reader can do is recover the paths beyond the bound: a file outside
it is *absent* rather than unreferenced, and those two are indistinguishable from inside
this seam.

**A question is a record, and it is the one kind nobody stated anything in.** A
projected decision request arrives with no ``record_kind`` -- Kojutsu writes the
key for the four entry writers and not for this projection -- so its kind comes from
the tag set, where ``question`` names it. That is the same tag the id namespace uses,
and the reason it has to be: before this branch existed, ``question`` and
``question_answered`` fell through to the answer case, and a request was counted as a
conclusion by every measure that groups by kind. A question carries no
``capture_source`` and no ``independence`` by construction, which is what keeps it
out of an evidence-only query -- so a reader that files it among the answers has not
mis-labelled a record, it has promoted a request into evidence. The projected fields
are read here and nowhere interpreted: what a request means, and how long it waited,
belongs to :mod:`tenbin.measures.decision_requests`.

**Nothing here raises.** A document is a fact about the store, and a fact with a
malformed field in it is still a fact: a timestamp that will not parse yields
``None``, a number that arrives as a non-number yields ``None``, and the record
stays countable. The strict alternative -- raise and lose the record -- would
make one bad field remove a document from every denominator, which is the
opposite of what a measurement program is for. What the gap means is not
discarded either: it is visible as ``None`` in a field the caller had to ask
about, and named in :mod:`tenbin.corpus.provenance` when it matters to a claim.

**The document id is the structure.** Ids are path-derived and the repository
segment is ``owner/name``, so an id contains more slashes than a regex written
by someone thinking of four segments would allow. ``pr-0`` is Kojutsu's
placeholder for "no pull request" and parses to ``None``, not to zero: a
zero-numbered pull request does not exist, and treating the placeholder as a
number would put an invented change at the bottom of every per-change histogram.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Final

from tenbin.corpus.values import Stated, Unstated, parse_stated
from tenbin.store.client import StoreDocument

#: The path segment a rationale document carries, and the only thing separating
#: "a stated reason" from "an answer" in the id namespace. Kojutsu writes it
#: deliberately: a rationale landing at ``<repo>/pr-<n>/<entry_id>`` would be
#: indistinguishable from an answer by its address alone.
RATIONALE_SEGMENT: Final[str] = "rationale"

#: The path segment a projected decision request carries, and the only thing
#: separating "a request for a reason" from "a record of one" in the id namespace.
#: Kojutsu writes it for the same reason it writes ``rationale/``: a question
#: that was asked and never answered is a request, not a conclusion, and a store
#: that cannot tell them apart is lying about its own contents. It is also the tag
#: Kojutsu projects the kind from, so the same word addresses the document and
#: classifies it.
QUESTION_SEGMENT: Final[str] = "question"

#: The two sub-namespaces a document id may carry between the pull-request segment
#: and the entry id, and each exists for one reason: the thing at that address is
#: not the thing a reader of the address would assume it is. A second sub-kind
#: gets a second branch over this set rather than a second parser, because two
#: definitions of "the shape of an id" would drift and a reader would be left
#: guessing which one a given document had been through.
NAMESPACE_SEGMENTS: Final[frozenset[str]] = frozenset({RATIONALE_SEGMENT, QUESTION_SEGMENT})

#: The literal Kojutsu writes where a value was never stated. It is carried
#: verbatim into a record rather than normalised to ``None`` so that a report
#: grouping by repository shows a bucket called ``unknown`` rather than one
#: labelled with the absence of a label.
PLACEHOLDER: Final[str] = "unknown"

#: The frontmatter key carrying the kind of record a document is. Present on
#: every record Kojutsu's four entry writers produce, and absent from every
#: document written before it existed and from every rationale, which is
#: dispatched through a different payload. Named as a constant because the key is
#: quoted twice -- once where it is read and once in the evidence of every
#: classification that rests on it -- and two spellings of one key would make
#: half the corpus's classifications cite something no document contains.
RECORD_KIND_KEY: Final[str] = "record_kind"

#: The keys describing which code a capture is anchored to. ``files`` is the edge
#: Tanseki's deriver recognises, so a change is reachable from the paths it touched;
#: ``head_sha`` is the commit the capture was taken against, which is an anchor and
#: never a verification. Named here for the reason :data:`RECORD_KIND_KEY` is: the key
#: is quoted where it is read and again in every exclusion that refers to it, and two
#: spellings of one key would make half the corpus's caveats cite something no document
#: contains.
FILES_KEY: Final[str] = "files"
HEAD_SHA_KEY: Final[str] = "head_sha"

#: The key carrying the conclusion a build reported about a commit, verbatim. Not
#: named ``check_outcome``: the field is the forge's own wording with a meaning this
#: package does not get to extend, and a name that reads as a result would be one
#: somebody would eventually normalise.
CHECK_CONCLUSION_KEY: Final[str] = "check_conclusion"

#: The forge's own identifier for the check that ran, and its label for that check.
#: Read together with :data:`CHECK_CONCLUSION_KEY` and never without it: a conclusion
#: with no check name is the least actionable number in the package, because it says
#: *something* failed at a rate nobody can attribute to a gate. Kojutsu writes all
#: three under ``build_frontmatter`` from the same webhook payload
#: (``answer_collector.py:954-958``) and projects them verbatim
#: (``tanseki_mapping.py:193-196``), so a corpus either has all three or has none.
CHECK_ID_KEY: Final[str] = "check_id"
CHECK_NAME_KEY: Final[str] = "check_name"

#: The planning ticket a change was opened against, if the branch name carried one.
#: Kojutsu derives it from a branch-name pattern rather than from the forge
#: (``tanseki_mapping.py:136`` reads ``jira_ticket_key``), so its absence means *not
#: supplied* and not *none* -- a distinction this field's docstring carries because
#: a reader who misses it will count the gap as an absence of tickets.
JIRA_KEY: Final[str] = "jira"

#: The tag Kojutsu projects onto a check run, and the prefix of the tag that
#: refines it. Kojutsu derives tags from the kind it already holds, so a check
#: record carries ``"check"`` and, when the forge stated a conclusion,
#: ``"check_state_<conclusion>"``.
CHECK_TAG: Final[str] = "check"
CHECK_STATE_PREFIX: Final[str] = "check_state_"

#: The shape of the marker Kojutsu appends to a bounded file list, matched as a
#: shape rather than as one string because the count is part of the text. A reader
#: that knew only the example would start counting ``"3 more files not listed"`` as a
#: path the moment Kojutsu wrote ``"17"``, and the failure would be a file in a
#: report that nobody can look up. Anchored, because a path that merely mentioned the
#: phrase would be a real path this reader must not discard.
TRUNCATION_MARKER: Final[re.Pattern[str]] = re.compile(r"^\d+ more files not listed$")

#: Entry-id prefixes, one per writer, recoverable from the id without reading
#: anything else. Kept as a separate signal from the tags on purpose: it is the
#: writer's own namespace, and a record whose id prefix and tag set disagree is
#: a contradiction between two independent statements about the same record.
#: Nothing in this program resolves that contradiction -- the tags win, because
#: they are what Kojutsu projects from the kind it holds, and the id is a
#: rendering of the same decision made by a different function.
ENTRY_ID_PREFIXES: Final[tuple[str, ...]] = (
    "answer-",
    "review-",
    "pr-event-",
    "rationale-v1-",
)

#: The one segment of an id that names a pull request. Anchored, so a repository
#: named ``pr-notes`` cannot be mistaken for one, and matched with a digit group
#: so the placeholder ``pr-0`` is recognisable as a number-shaped token rather
#: than as a pull request.
_PR_SEGMENT = re.compile(r"^pr-(\d+)$")


class RecordKind(StrEnum):
    """What a record is, in the store's own vocabulary.

    ``StrEnum`` rather than ``Enum`` because these values are the strings
    Kojutsu writes and the strings a report groups by; a member that needed
    ``.value`` to be printable would be a member that gets printed wrong.
    """

    #: A conclusion someone reached in response to a question. The only kind that
    #: is inferred rather than stated, because it is the only kind whose tag
    #: projection is empty.
    ANSWER = "answer"
    #: A decision about somebody else's change, carrying a review state.
    REVIEW_VERDICT = "review_verdict"
    #: A note about a line of code, which may since have moved.
    INLINE_REVIEW_COMMENT = "inline_review_comment"
    #: The write path reporting on itself: the change opened, closed, reopened.
    PR_LIFECYCLE = "pr_lifecycle"
    #: A stated reason for a decision, with no claim to have been verified.
    RATIONALE = "rationale"
    #: A request for a reason, projected out of Kojutsu's question registry once
    #: it reached a terminal state. It is the one kind nobody stated anything in, so
    #: it carries no independence level and no capture source by design, and it must
    #: sit below every evidence threshold in the program. The value Kojutsu writes
    #: is both its tag and its path segment, which is why it needs no reader of its own.
    QUESTION = "question"
    #: A build's own report about a commit, and nothing more. Its own kind so that it
    #: can never be rendered where a reader would take it for a person's judgement --
    #: Kojutsu's stated reason for the split, kept in the same words here because a
    #: check conclusion rendered beside a review verdict is an inference nobody made.
    CHECK_RUN = "check_run"


class ReviewState(StrEnum):
    """The review states that represent a decision about the code.

    ``commented`` is deliberately absent. Leaving a note with no verdict is
    feedback, not an approval, and folding the two together would inflate an
    approval rate with records where nobody approved anything.
    """

    APPROVED = "approved"
    CHANGES_REQUESTED = "changes_requested"


class LifecycleAction(StrEnum):
    """The pull-request transitions Kojutsu stores, and the only three it stores.

    ``synchronize`` is accepted by the webhook and produces no record, so it is
    not here: an enum member no record can ever carry is vocabulary that invites
    a reader to expect a population that does not exist.
    """

    OPENED = "opened"
    CLOSED = "closed"
    REOPENED = "reopened"


class Certainty(StrEnum):
    """Whether a classification was written down or inferred from an absence."""

    #: A tag stated the kind. The writer's own projection, recovered.
    DETERMINED = "determined"
    #: No tag stated the kind; it is the best reading of what is missing. Only
    #: ever the answer case, and the reason it is a separate value at all.
    INFERRED = "inferred"


#: Tag to review state, as Kojutsu projects it. The key is the whole tag, so a
#: record's evidence can name the tag a reader would have to go and look for.
_REVIEW_STATE_TAGS: Final[Mapping[str, ReviewState]] = MappingProxyType(
    {
        "review_state_approved": ReviewState.APPROVED,
        "review_state_changes_requested": ReviewState.CHANGES_REQUESTED,
    }
)

#: Tag to lifecycle action, same projection and same reason for being a map.
_LIFECYCLE_ACTION_TAGS: Final[Mapping[str, LifecycleAction]] = MappingProxyType(
    {
        "action_opened": LifecycleAction.OPENED,
        "action_closed": LifecycleAction.CLOSED,
        "action_reopened": LifecycleAction.REOPENED,
    }
)

#: Tag to rationale source. ``rationale_unknown`` is a real value Kojutsu
#: writes for a rationale written before this axis existed, and it is carried
#: through as a source rather than as an absence: the writer did state that it
#: did not know where the reasoning came from, which is a weaker claim than
#: having said nothing at all, and the difference is worth the three lines.
_RATIONALE_SOURCE_TAGS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "rationale_declared": "declared",
        "rationale_reconstructed": "reconstructed",
        "rationale_unknown": "unknown",
    }
)

#: The prefix Kojutsu puts on the tag carrying a request's terminal state, as
#: ``question_<status>``. Derived from :data:`QUESTION_SEGMENT` rather than spelled
#: out, because it is the same word twice -- once in the path and once in the tag --
#: and the two spellings are one namespace rather than two coincidences.
_QUESTION_TAG_PREFIX: Final[str] = f"{QUESTION_SEGMENT}_"


@dataclass(frozen=True)
class Classification:
    """What kind of record this is, and what that determination rests on.

    ``evidence`` holds the tags that drove the reading, so a reader can check
    the classification against the document without re-deriving it, and
    ``certainty`` is the honest label for the one case where there was nothing to
    point at. ``detail`` is the short auditable phrase a report can quote, and
    deliberately says what was *read* rather than what it concluded.
    """

    kind: RecordKind
    certainty: Certainty
    evidence: tuple[str, ...]
    detail: str


def review_state_from_tags(tags: Sequence[str]) -> ReviewState | None:
    """Recover the review state a verdict carries, or ``None`` if it names none.

    First match wins, so a record whose tags somehow name two states resolves
    deterministically rather than depending on set ordering. A record tagged
    ``review`` with no state is a real possibility -- a state this reader does
    not know about -- and is reported as an unknown state rather than being
    quietly filed under ``commented``.
    """
    for tag in tags:
        state = _REVIEW_STATE_TAGS.get(tag)
        if state is not None:
            return state
    return None


def lifecycle_action_from_tags(tags: Sequence[str]) -> LifecycleAction | None:
    """Recover the transition a lifecycle record carries, or ``None``."""
    for tag in tags:
        action = _LIFECYCLE_ACTION_TAGS.get(tag)
        if action is not None:
            return action
    return None


def rationale_source_from_tags(tags: Sequence[str]) -> str | None:
    """Recover the source a rationale's tags carry, or ``None``.

    Kept as a plain string rather than an enum because this module does not own
    the vocabulary -- Kojutsu does, in ``models.RationaleSource`` -- and a
    reader that pinned it to a local enum would have to be widened every time a
    fourth source appeared.
    """
    for tag in tags:
        source = _RATIONALE_SOURCE_TAGS.get(tag)
        if source is not None:
            return source
    return None


def question_status_from_tags(tags: Sequence[str]) -> str | None:
    """Recover the terminal state a decision request's tags carry, or ``None``.

    Read by prefix rather than through a map of the three known statuses, for the
    same reason :func:`rationale_source_from_tags` avoids an enum: the closed set
    is Kojutsu's (``answered``, ``failed``, ``superseded``) and a fourth value
    arriving has to survive as itself, because a reader that could not name it
    would be pushed towards folding it into a neighbour -- and a request folded
    into ``failed`` is a request whose outcome this store never reported.
    """
    for tag in tags:
        if tag.startswith(_QUESTION_TAG_PREFIX):
            status = tag[len(_QUESTION_TAG_PREFIX) :].strip()
            if status:
                return status
    return None


def check_conclusion_from_tags(tags: Sequence[str]) -> str | None:
    """Recover the conclusion a check record's tags carry, or ``None``.

    A prefix match rather than a lookup in a map, unlike the review state and the
    lifecycle action beside it, because the forge's vocabulary of conclusions is not
    closed and a local copy of it would go stale the first time a forge added one. So
    the tag is split and whatever follows the prefix is carried through as itself, which
    is the same posture :func:`classify` takes on an unreadable ``record_kind``. A bare
    ``check_state_`` names no conclusion and yields ``None`` rather than the empty
    string, because an empty conclusion would bucket as a real one.
    """
    for tag in tags:
        if tag.startswith(CHECK_STATE_PREFIX):
            conclusion = tag[len(CHECK_STATE_PREFIX) :].strip()
            if conclusion:
                return conclusion
    return None


def record_kind_from(raw: str | None) -> RecordKind | None:
    """The kind a writer *stated*, or ``None`` when it stated none we can read.

    Separate from :func:`classify` because the two answer different questions
    and a reader checking a figure needs to tell them apart: this one asks "did
    the document say what it is", and ``classify`` asks "what is it". A document
    written before Kojutsu wrote ``record_kind`` -- and every rationale, which
    is dispatched through a different payload and never carried the key at all --
    answers ``None`` here, and is classified from its tags instead.

    A value outside the vocabulary is also ``None``, deliberately. Refusing it
    would drop the record, and reading it as a kind nobody defined would invent
    one; the honest handling is to classify from the tags and carry the value as
    evidence, which is what :func:`classify` does with it.
    """
    if raw is None:
        return None
    try:
        return RecordKind(raw)
    except ValueError:
        return None


def classify(tags: Sequence[str] | None, *, record_kind: str | None = None) -> Classification:
    """Recover a record's kind, preferring what the writer stated to what we read.

    ``record_kind`` is Kojutsu's own key, and when it is present and readable
    it *is* the kind -- so the reading is :data:`Certainty.DETERMINED` and the
    evidence names the key rather than a tag. A document with no such key is read
    from its tag projection by :func:`classify_by_tags`, which is the weaker
    statement and stays available for every record written before the key existed.

    An unrecognised value falls through to the tag reading *and names itself in
    the evidence*. Folding it into a kind would be inventing a category, and
    dropping it would be invisible: a corpus where every record carries a
    misspelled kind would otherwise classify cleanly from tags, with nothing
    anywhere recording that the key nobody could read was sitting right there.
    """
    stated = record_kind_from(record_kind)
    if stated is not None:
        return Classification(
            kind=stated,
            certainty=Certainty.DETERMINED,
            evidence=(RECORD_KIND_KEY, stated.value),
            detail=f"frontmatter {RECORD_KIND_KEY!r} states {stated.value!r}",
        )
    reading = classify_by_tags(tags)
    if record_kind is None:
        return reading
    return Classification(
        kind=reading.kind,
        certainty=Certainty.INFERRED,
        evidence=(f"{RECORD_KIND_KEY}={record_kind!r}", *reading.evidence),
        detail=(
            f"frontmatter {RECORD_KIND_KEY!r} holds {record_kind!r}, which is not in the "
            f"store's vocabulary; {reading.detail}"
        ),
    )


def classify_by_tags(tags: Sequence[str] | None) -> Classification:
    """Recover a record's kind from the tag set Kojutsu projects from it.

    Precedence is rationale, then ``question``, then ``pr_state_change``, then ``check``,
    then ``review``
    (inline before verdict), then answer, and the order is not arbitrary: the more
    specific tag wins, so a record carrying both ``review`` and an unrelated tag is still
    read as the review it is. ``question`` sits with the two namespace tags ahead of
    everything because it is the tag Kojutsu projects for a projected decision
    request, and a request that fell through to the answer case would be a request
    counted as a conclusion. ``check`` sits beside ``pr_state_change`` and ahead of
    ``review`` for that same reason -- it is a whole tag projection naming one kind,
    exactly as ``pr_state_change`` is, whereas ``review`` is shared with two kinds and
    needs the ``inline_comment`` refinement before it can be read at all. Kojutsu
    writes a check run with ``check`` and without ``review``, so the ordering settles a
    contradiction this corpus should not contain rather than a case it produces.
    Only the answer case can be :data:`Certainty.INFERRED`, because it is the only kind
    with an empty tag projection.

    Named rather than inlined into :func:`classify` because the two are different
    statements about the same record and a report has to be able to say which one
    it is resting on. A reader who is handed a determination and cannot see that
    the determination came from a tag projection is being asked to trust a
    stronger statement than was made.

    Tags that name no kind at all are an anomaly rather than an answer, but the
    kind is still answer -- there is no fifth possibility -- and the evidence
    carries those tags so the anomaly is visible at the point of reading.
    """
    present = _clean_tags(tags)
    if not present:
        return Classification(
            kind=RecordKind.ANSWER,
            certainty=Certainty.INFERRED,
            evidence=("no tags",),
            detail="no tags at all; the writer projected nothing, so the kind is inferred",
        )

    if RATIONALE_SEGMENT in present:
        source = rationale_source_from_tags(present)
        evidence = (
            (RATIONALE_SEGMENT,) if source is None else (RATIONALE_SEGMENT, _source_tag(source))
        )
        detail = (
            f"tag {RATIONALE_SEGMENT!r} names a stated reason"
            if source is None
            else f"tag {RATIONALE_SEGMENT!r} with source {source!r}"
        )
        return Classification(RecordKind.RATIONALE, Certainty.DETERMINED, evidence, detail)

    if QUESTION_SEGMENT in present:
        status = question_status_from_tags(present)
        evidence = (
            (QUESTION_SEGMENT,) if status is None else (QUESTION_SEGMENT, f"question_{status}")
        )
        detail = (
            f"tag {QUESTION_SEGMENT!r} names a request for a reason"
            if status is None
            else f"tag {QUESTION_SEGMENT!r} with status {status!r}"
        )
        return Classification(RecordKind.QUESTION, Certainty.DETERMINED, evidence, detail)

    if "pr_state_change" in present:
        action = lifecycle_action_from_tags(present)
        evidence = (
            ("pr_state_change",) if action is None else ("pr_state_change", _action_tag(action))
        )
        detail = (
            "tag 'pr_state_change' with no action this reader knows"
            if action is None
            else f"tag 'pr_state_change' with action {action.value!r}"
        )
        return Classification(RecordKind.PR_LIFECYCLE, Certainty.DETERMINED, evidence, detail)

    if CHECK_TAG in present:
        conclusion = check_conclusion_from_tags(present)
        evidence = (
            (CHECK_TAG,) if conclusion is None else (CHECK_TAG, f"{CHECK_STATE_PREFIX}{conclusion}")
        )
        detail = (
            "tag 'check' with no conclusion this reader can read"
            if conclusion is None
            else f"tag 'check' with conclusion {conclusion!r}"
        )
        return Classification(RecordKind.CHECK_RUN, Certainty.DETERMINED, evidence, detail)

    if "review" in present:
        if "inline_comment" in present:
            return Classification(
                RecordKind.INLINE_REVIEW_COMMENT,
                Certainty.DETERMINED,
                ("review", "inline_comment"),
                "tags 'review' and 'inline_comment'",
            )
        state = review_state_from_tags(present)
        state_tag = None if state is None else _state_tag(state)
        evidence = ("review",) if state_tag is None else ("review", state_tag)
        detail = (
            "tag 'review' with no verdict state this reader knows"
            if state_tag is None
            else f"tags 'review' and {state_tag!r}"
        )
        return Classification(RecordKind.REVIEW_VERDICT, Certainty.DETERMINED, evidence, detail)

    if "agent_authored" in present:
        return Classification(
            RecordKind.ANSWER,
            Certainty.DETERMINED,
            ("agent_authored",),
            "tag 'agent_authored' names a written answer",
        )

    return Classification(
        RecordKind.ANSWER,
        Certainty.INFERRED,
        present,
        "tags present but none names a record kind; the kind is inferred",
    )


def _source_tag(source: str) -> str:
    return f"{RATIONALE_SEGMENT}_{source}"


def _action_tag(action: LifecycleAction) -> str:
    return f"action_{action.value}"


def _state_tag(state: ReviewState) -> str:
    return f"review_state_{state.value}"


def _clean_tags(tags: Sequence[str] | None) -> tuple[str, ...]:
    """Normalise a tag list, dropping order duplicates and unusable entries.

    A tag that is not a non-empty string is dropped rather than coerced: it was
    not written by Kojutsu's tag projection, so it says nothing about the
    kind, and inventing a reading for it would be exactly the kind of silent
    recovery this module is refusing to do. A record whose tags are *all* unusable
    is therefore classified the same way as a record with no tags -- inferred, on
    the evidence of the tags that were not readable.
    """
    if not tags:
        return ()
    return tuple(dict.fromkeys(tag.strip() for tag in tags if isinstance(tag, str) and tag.strip()))


@dataclass(frozen=True)
class DocumentPath:
    """The structure of a document id, as far as it could be read.

    Every field is optional, including :attr:`entry_id`, because a document
    written outside Kojutsu has no such shape and is still a document. The
    point of parsing leniently is that a record with an unparseable id still
    reaches the denominators.
    """

    repo: str | None = None
    pr: int | None = None
    entry_id: str | None = None
    is_rationale: bool = False
    #: True for a projected decision request, on the same terms as
    #: :attr:`is_rationale`: the segment is the writer's own statement about what
    #: the document is, recovered from the address rather than from the frontmatter.
    is_question: bool = False


def parse_document_id(doc_id: str) -> DocumentPath:
    """Split a path-derived document id into repository, pull request and entry.

    No fixed segment count, because ``repo`` is ``owner/name`` and contributes a
    slash of its own; a regex written against four segments would reject every
    document in a real corpus. The ``pr-<n>`` segment is located by pattern and
    taken as the *last* match, so a repository whose own name looks like
    ``pr-1`` cannot be mistaken for the change.

    ``pr-0`` parses to ``pr=None``: it is the placeholder Kojutsu writes for a
    record with no pull request, and a pull request numbered zero does not exist.

    A sub-kind between the change and the entry is one branch over
    :data:`NAMESPACE_SEGMENTS` rather than one branch per kind. The shapes are the
    same -- ``<repo>/pr-<n>/<sub-kind>/<entry_id>`` -- and a parser that recognised
    only the first of them would quietly file a projected decision request under an
    answer-shaped entry id, which is the one thing decision 001 says must not happen.
    """
    segments = [segment for segment in doc_id.split("/") if segment]
    if not segments:
        return DocumentPath()

    match_index: int | None = None
    pr_number: int | None = None
    for index, segment in enumerate(segments):
        match = _PR_SEGMENT.match(segment)
        if match is not None:
            match_index, pr_number = index, int(match.group(1))

    if match_index is None:
        return DocumentPath(entry_id=segments[-1])

    repo = "/".join(segments[:match_index]) or None
    tail = segments[match_index + 1 :]
    if len(tail) == 2 and tail[0] in NAMESPACE_SEGMENTS:
        sub_kind, entry_id = tail
        return DocumentPath(
            repo=repo,
            pr=pr_number if pr_number else None,
            entry_id=entry_id,
            is_rationale=sub_kind == RATIONALE_SEGMENT,
            is_question=sub_kind == QUESTION_SEGMENT,
        )
    return DocumentPath(
        repo=repo,
        pr=pr_number if pr_number else None,
        entry_id=tail[-1] if tail else None,
        is_rationale=False,
        is_question=False,
    )


def entry_id_prefix(entry_id: str | None) -> str | None:
    """Return the writer's namespace prefix on an entry id, if it is one we know."""
    if not entry_id:
        return None
    return next((prefix for prefix in ENTRY_ID_PREFIXES if entry_id.startswith(prefix)), None)


def _text(raw: Any) -> str | None:
    """Read a frontmatter value as text, or ``None`` if there is none to read.

    Numbers and booleans are stringified because Kojutsu stringifies them on
    the way in, so a store that hands back a bare ``42`` has round-tripped
    through a serialiser that did not. Lists and mappings yield ``None``: there
    is no defensible one-line reading of a list where a name was expected, and
    joining it would invent a principal whose name contains punctuation the
    writer never chose.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        stripped = raw.strip()
        return stripped or None
    if isinstance(raw, bool | int | float):
        return str(raw)
    return None


def _positive_int(raw: Any) -> int | None:
    """Read a number that Kojutsu stringified, or ``None``.

    Both spellings are accepted because the store stringifies extras: ``42`` and
    ``"42"`` are the same value and an int-only reader would call every stored
    pull-request number missing. Booleans are rejected explicitly -- ``bool`` is
    an ``int`` in Python and ``True`` would otherwise be pull request number 1.
    """
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw if raw > 0 else None
    if isinstance(raw, str):
        candidate = raw.strip()
        if candidate.isdigit() and int(candidate) > 0:
            return int(candidate)
    return None


def _non_negative_int(raw: Any) -> int | None:
    """Read a count that is allowed to be zero, or ``None``.

    A separate reader from :func:`_positive_int` because the two answer different
    questions and folding them together would make one of them lie. Every other
    number this module reads is an identifier, where zero is Kojutsu's
    placeholder for "none" and must read as absent; an attempt count of zero is a
    real observation -- nobody was asked -- and reading it as ``None`` would report
    it as a count nobody recorded.
    """
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw if raw >= 0 else None
    if isinstance(raw, str):
        candidate = raw.strip()
        if candidate.isdigit():
            return int(candidate)
    return None


def _timestamp(raw: Any) -> datetime | None:
    """Parse a Kojutsu timestamp, or ``None`` if it is not one.

    Kojutsu writes ``datetime.isoformat()`` with a ``+00:00`` offset, so a
    ``Z`` suffix means the document came from something else; it is accepted
    rather than refused, because a timestamp in the right instant and format is
    still a timestamp and refusing it would lose the record's place in time.
    A naive timestamp is taken as UTC, which is what every Kojutsu writer
    used: a naive value here is a writer that forgot an offset, not a record of
    an event in a different zone.
    """
    text = _text(raw)
    if text is None:
        return None
    candidate = f"{text[:-1]}+00:00" if text.endswith(("Z", "z")) else text
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed


def _tag_list(raw: Any) -> tuple[str, ...]:
    """Read the ``tags`` key as a sequence of strings.

    A bare string is accepted as a single tag rather than being split on
    whitespace or commas: guessing a separator invents tags the writer never
    wrote, and every one of them would then be eligible to drive a
    classification.
    """
    if isinstance(raw, str):
        return (raw.strip(),) if raw.strip() else ()
    if isinstance(raw, list | tuple):
        return tuple(tag for tag in raw if isinstance(tag, str))
    return ()


def is_truncation_marker(value: str) -> bool:
    """Whether this entry is Kojutsu's overflow marker rather than a path.

    Exposed because the marker is a fact about a document rather than a detail of
    parsing: a measure has to be able to say *the list was bounded* without holding the
    raw frontmatter, and the only way to know that is to recognise the sentence the
    writer put in place of the paths.
    """
    return TRUNCATION_MARKER.match(value.strip()) is not None


def _file_list(raw: Any) -> tuple[str, ...]:
    """Read ``files`` as repository-relative paths, with the overflow marker removed.

    Four things are dropped, and each is dropped for a stated reason rather than for
    tidiness. Kojutsu's marker, because it is a sentence where a path goes (see
    :attr:`Record.files_truncated`). A non-string, because a document can only be
    malformed rather than informative that way. A blank, because a path nobody wrote is
    not a path. And a repeat, because a record names the files it touched as a set, and
    counting one twice would inflate the per-file distribution above it. Order is
    preserved, so the list stays a reading of the document rather than of a set.
    """
    if not isinstance(raw, list | tuple):
        return ()
    paths: list[str] = []
    for entry in raw:
        if not isinstance(entry, str):
            continue
        candidate = entry.strip()
        if not candidate or is_truncation_marker(candidate) or candidate in paths:
            continue
        paths.append(candidate)
    return tuple(paths)


@dataclass(frozen=True)
class Record:
    """One stored document, read into the shape a measurement program can use.

    Built only by :meth:`from_document`, so every record in this program has been
    through the same lenient parse and there is no second, more forgiving way to
    make one. The frontmatter is held as a read-only mapping: a record is a
    statement about a document, and a caller that could edit the evidence out of
    it would be able to report a corpus that does not exist.

    Two identity rules hold throughout. **The document id is the structure** --
    repository, pull request and entry id are parsed from the path, because that
    is the store's own key and a writer can disagree with itself between the
    path and the frontmatter. **A malformed value is ``None``**, never an
    exception, so one bad field costs a field rather than a record.

    ``answered_by_model`` and ``declared_by_model`` are :class:`Stated` or
    :class:`Unstated` rather than strings, because Kojutsu writes the literal
    ``"unknown"`` into both where nothing was stated, and a reader holding a
    plain string cannot tell that bucket from a group. See
    :mod:`tenbin.corpus.values`.
    """

    doc_id: str
    collection: str
    content: str
    frontmatter: Mapping[str, Any]
    classification: Classification
    path: str = ""
    content_hash: str | None = None
    revision: str | None = None
    updated_at: datetime | None = None
    entry_id: str = ""
    repo: str | None = None
    pr: int | None = None
    pr_url: str | None = None
    branch: str | None = None
    title: str | None = None
    author: str | None = None
    comment_author: str | None = None
    answered_by_agent: str | None = None
    answered_by_model: Stated[str] | Unstated = Unstated()
    github_author_association: str | None = None
    declared_by: str | None = None
    declared_by_model: Stated[str] | Unstated = Unstated()
    category: str | None = None
    capture_source: str | None = None
    independence: str | None = None
    independence_reason: str | None = None
    delivery_id: str | None = None
    github_comment_id: int | None = None
    session_id: str | None = None
    question_id: str | None = None
    #: The request a projected decision request is about, verbatim. Free text rather
    #: than an id because a question is the whole of what the document holds: there is
    #: no answer in it, so the only thing a reader can hold is the question asked.
    question_text: str | None = None
    #: The account that raised the request. Not the person who answered it and not
    #: the account that opened the change: a decision request is asked *at* a change,
    #: and the three are different questions about a different repository.
    question_author: str | None = None
    #: The terminal state Kojutsu observed, as ``answered``, ``failed`` or
    #: ``superseded``. A plain string rather than an enum because this reader does not
    #: own the vocabulary, and a fourth value has to survive as itself rather than be
    #: refused -- see :func:`question_status_from_tags` for the same argument.
    question_status: str | None = None
    #: How many times the request was put to somebody before it reached that state.
    #: ``None`` on every record that is not a request, and never defaulted to zero: an
    #: absent count and a count of nothing are different facts about different kinds.
    attempts: int | None = None
    #: The comment a human answered the request in, when one did. It is a pointer to
    #: the answer, not the answer, and reading it as one is the mistake this program
    #: makes elsewhere about every pointer it holds.
    answer_comment_id: int | None = None
    #: When the request was raised. Not the document's creation time: a request is
    #: projected to the store long after somebody asked it, and the wait is the whole
    #: of what the decision-request lifecycle measures.
    created_at: datetime | None = None
    #: When the projected status was observed, written by Kojutsu for the same
    #: reason the projection is eventually consistent. A status with no age attached
    #: is a claim about *now* made by a document about the past.
    status_as_of: datetime | None = None
    rationale_source: str | None = None
    rationale_revision: int | None = None
    rationale_revises: str | None = None
    rationale_path: bool = False
    question_path: bool = False
    answered_at: datetime | None = None
    captured_at: datetime | None = None
    declared_at: datetime | None = None
    #: The kind the *writer* stated, which is not the same question as
    #: ``classification.kind``. ``None`` means the document carried no readable
    #: ``record_kind`` -- a legacy record, or a rationale, which Kojutsu never
    #: gave the key -- and the kind on the classification was then recovered from
    #: tags, a weaker statement about the same record. A report that has to say
    #: which of the two it is resting on asks this field.
    record_kind: RecordKind | None = None
    #: The forge's identifier for the review a record came from. Its presence or
    #: absence is itself reportable: with it, a verdict and its inline comments
    #: can be put back together, and a review that was later edited can be
    #: noticed. ``None`` on every record written before the key existed.
    review_id: str | None = None
    #: The login GitHub reports opened the pull request. Not a person, not a bot,
    #: and not a statement about what they were doing with it -- Kojutsu's own
    #: framing of the field, repeated here because a reader who treats it as any
    #: of those three will draw a conclusion the store never licensed.
    change_author_account: str | None = None
    #: When the change opened, as GitHub reports it. Distinct from
    #: ``updated_at``, which moves on every push and comment, and from
    #: ``answered_at``, which is when Kojutsu captured *this record's* event.
    pr_opened_at: datetime | None = None
    #: How the change ended, in the two strings GitHub can be read out of a close
    #: for: ``merged`` and ``closed_unmerged``. Deliberately a plain string, not
    #: an enum, because a value outside that pair has to survive as itself -- a
    #: closed vocabulary here would either raise on a writer's new value or fold
    #: it into a neighbour, and both of those lose the fact that something
    #: unrecognised was written.
    pr_outcome: str | None = None
    #: When the change merged, which is a different event from the close this
    #: record is about: one pull request can be closed and merged at different
    #: times, and ``pr_outcome`` is a claim where this is a clock.
    pr_merged_at: datetime | None = None
    #: The files this record is about, as repository-relative paths, which is the
    #: anchor Tanseki's edge deriver reads and the first fact that connects a record to
    #: code. Kojutsu bounds the list and says so in the list; a file beyond the
    #: bound is absent rather than unreferenced, which no reader inside this seam can
    #: tell apart -- see :attr:`files_truncated`. Empty on every record written before
    #: the key existed, and empty on a record whose change touched nothing.
    files: tuple[str, ...] = ()
    #: The commit the capture was taken against. An anchor and never a verification:
    #: it says which commit this record is about, not that the record is still true
    #: there, that the change was reviewed, or that the two correspond to the same
    #: code. A field named ``verified_at_commit`` would be a claim the store cannot
    #: back, which is why Kojutsu did not write one.
    head_sha: str | None = None
    #: What a build concluded about the commit ``head_sha`` names, in the forge's own
    #: words and lowercased by the writer. Deliberately a plain string and not an enum,
    #: for the reason ``pr_outcome`` is one: the vocabulary belongs to the forge, a
    #: reader that closed it would either raise on a new value or fold it into a
    #: neighbour, and both lose the fact that something unrecognised was written.
    check_conclusion: str | None = None
    #: The forge's identifier for the check that ran. The join key for a check-run
    #: record, and the fallback identity for a check whose name the forge did not
    #: give: an unnamed gate cannot be acted on, which is why this is read rather
    #: than synthesised away.
    check_id: str | None = None
    #: The forge's label for the check that ran. Free text and not a closed
    #: vocabulary, deliberately: it is the forge's wording, and a reader that
    #: normalised it would fold a newly added gate into a bucket and make a new
    #: gate look like a long-standing one.
    check_name: str | None = None
    #: The planning ticket this change was opened against, when the branch name
    #: carried one. **Absence means not supplied rather than none**, because the
    #: key is derived from a branch-name pattern and a branch that did not follow
    #: the convention says nothing at all about whether a ticket existed.
    jira: str | None = None

    def __post_init__(self) -> None:
        # Frozen dataclasses with a mapping field are hashable *by declaration*
        # and raise at runtime, which is the kind of latent trap that shows up
        # in a set two modules away from the dataclass. Identity here is the
        # document id -- the store's own key, and the only field guaranteed
        # present -- so it is what the hash is defined on, and two records read
        # from two snapshots of the same document compare and hash alike.
        object.__setattr__(self, "frontmatter", MappingProxyType(dict(self.frontmatter)))

    def __hash__(self) -> int:
        return hash(self.doc_id)

    @classmethod
    def from_document(cls, document: StoreDocument) -> Record:
        """The single point where a document becomes a record.

        Everything this module knows how to read is read here, and nothing
        raises. A field that cannot be read becomes ``None`` and the record
        survives, because a corpus measurement that silently drops the records it
        could not parse is a measurement of the records it happened to
        understand.
        """
        frontmatter = document.frontmatter
        tags = _tag_list(frontmatter.get("tags"))
        parsed = parse_document_id(document.id)
        stated_kind = _text(frontmatter.get(RECORD_KIND_KEY))
        return cls(
            doc_id=document.id,
            collection=document.collection,
            content=document.content,
            frontmatter=frontmatter,
            classification=classify(tags, record_kind=stated_kind),
            path=document.path,
            content_hash=document.content_hash,
            revision=document.revision,
            updated_at=_timestamp(document.updated_at),
            entry_id=parsed.entry_id or "",
            repo=parsed.repo,
            pr=parsed.pr,
            pr_url=_text(frontmatter.get("pr_url")),
            branch=_text(frontmatter.get("branch")),
            title=_text(frontmatter.get("title")),
            author=_text(frontmatter.get("author")),
            comment_author=_text(frontmatter.get("comment_author")),
            answered_by_agent=_text(frontmatter.get("answered_by_agent")),
            answered_by_model=parse_stated(frontmatter.get("answered_by_model")),
            github_author_association=_text(frontmatter.get("github_author_association")),
            declared_by=_text(frontmatter.get("declared_by")),
            declared_by_model=parse_stated(frontmatter.get("declared_by_model")),
            category=_text(frontmatter.get("category")),
            capture_source=_text(frontmatter.get("capture_source")),
            independence=_text(frontmatter.get("independence")),
            independence_reason=_text(frontmatter.get("independence_reason")),
            delivery_id=_text(frontmatter.get("delivery_id")),
            github_comment_id=_positive_int(frontmatter.get("github_comment_id")),
            session_id=_text(frontmatter.get("session_id")),
            question_id=_text(frontmatter.get("question_id")),
            question_text=_text(frontmatter.get("question_text")),
            question_author=_text(frontmatter.get("question_author")),
            question_status=_text(frontmatter.get("question_status")),
            attempts=_non_negative_int(frontmatter.get("attempts")),
            answer_comment_id=_positive_int(frontmatter.get("answer_comment_id")),
            created_at=_timestamp(frontmatter.get("created_at")),
            status_as_of=_timestamp(frontmatter.get("status_as_of")),
            rationale_source=_text(frontmatter.get("rationale_source")),
            rationale_revision=_positive_int(frontmatter.get("rationale_revision")),
            rationale_revises=_text(frontmatter.get("rationale_revises")),
            rationale_path=parsed.is_rationale,
            question_path=parsed.is_question,
            answered_at=_timestamp(frontmatter.get("answered_at")),
            captured_at=_timestamp(frontmatter.get("captured_at")),
            declared_at=_timestamp(frontmatter.get("declared_at")),
            record_kind=record_kind_from(stated_kind),
            review_id=_text(frontmatter.get("review_id")),
            change_author_account=_text(frontmatter.get("change_author_account")),
            pr_opened_at=_timestamp(frontmatter.get("pr_opened_at")),
            pr_outcome=_text(frontmatter.get("pr_outcome")),
            pr_merged_at=_timestamp(frontmatter.get("pr_merged_at")),
            files=_file_list(frontmatter.get(FILES_KEY)),
            head_sha=_text(frontmatter.get(HEAD_SHA_KEY)),
            check_conclusion=_text(frontmatter.get(CHECK_CONCLUSION_KEY)),
            check_id=_text(frontmatter.get(CHECK_ID_KEY)),
            check_name=_text(frontmatter.get(CHECK_NAME_KEY)),
            jira=_text(frontmatter.get(JIRA_KEY)),
        )

    @property
    def pr_key(self) -> tuple[str, int] | None:
        """``(repo, pr)`` when both are readable, else ``None``.

        Either half alone is not a key. A record with a pull request number and
        no repository cannot be joined to anything, and returning a partial key
        would let it join to every repository that happens to hold the same
        number -- which is a fabricated correspondence, and exactly the sort of
        thing a measurement program must not do quietly.
        """
        if not self.repo or self.pr is None:
            return None
        return (self.repo, self.pr)

    @property
    def month(self) -> str | None:
        """The record's ``YYYY-MM`` in UTC, from the event it is about.

        ``answered_at`` first, because that is the event: a review's submission,
        a comment's creation, a close. Then the capture time, then a rationale's
        declaration, then the document's own update. The order is a preference
        for describing the record over describing the write, and every fallback
        is a weaker fact about a different clock, so a caller grouping by month
        over a corpus with gaps is grouping over a mixture and should say so.
        """
        moment = self.answered_at or self.captured_at or self.declared_at or self.updated_at
        if moment is None:
            return None
        return moment.astimezone(UTC).strftime("%Y-%m")

    @property
    def is_rationale(self) -> bool:
        """True when the id says rationale, or the tags do.

        The two are independent statements, and either is enough: a rationale
        whose tags were lost is still in the rationale namespace, and a record
        tagged ``rationale`` that is filed under an answer-shaped id is still
        describing a stated reason. The tag wins for classification, the path
        wins for addressing, and this property is the honest union of both.
        """
        return self.rationale_path or self.classification.kind is RecordKind.RATIONALE

    @property
    def is_question(self) -> bool:
        """True when the id says decision request, or the tags do.

        The union, and for the reason :attr:`is_rationale` is: a request is a
        request whichever of the two statements survives, and a safety check that
        only looked at one of them would be looking at whichever the store happened
        to preserve. This is deliberately wider than the classification on its own.
        A check for a capability that must never be published wants the wider
        population; a measure wants the narrower one, because only there is the
        writer's own kind actually established.
        """
        return self.question_path or self.classification.kind is RecordKind.QUESTION

    @property
    def review_state(self) -> ReviewState | None:
        """The review state a verdict carries, recovered from its tags."""
        return review_state_from_tags(_tag_list(self.frontmatter.get("tags")))

    @property
    def lifecycle_action(self) -> LifecycleAction | None:
        """The transition a lifecycle record carries, recovered from its tags."""
        return lifecycle_action_from_tags(_tag_list(self.frontmatter.get("tags")))

    @property
    def entry_prefix(self) -> str | None:
        """The writer's namespace on this entry id, if it is one we know."""
        return entry_id_prefix(self.entry_id)

    @property
    def files_truncated(self) -> bool:
        """Whether Kojutsu bounded the file list and wrote the overflow into it.

        Read from the retained frontmatter rather than from :attr:`files`, because
        :attr:`files` has already discarded the marker and this is the only place the
        marker survives. A property rather than a third field, for the same reason
        :attr:`review_state` is one: this is a reading of the document, not a value the
        writer stated, and a stored flag would be a second copy of the reading to keep
        in step with the parse.

        A ``True`` here is a statement about what this record *does not* name, and it is
        the only one available: Kojutsu states how many paths it left out in the
        marker and does not name them, so a reader can report the bound and can never
        recover what is beyond it.
        """
        raw = self.frontmatter.get(FILES_KEY)
        if not isinstance(raw, list | tuple):
            return False
        return any(isinstance(entry, str) and is_truncation_marker(entry) for entry in raw)
