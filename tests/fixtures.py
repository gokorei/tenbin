"""Fixtures for the measures layer: a snapshot a test can state in a few lines.

A measure is a pure function of a snapshot, so testing one needs a snapshot and
nothing else. Building one the way the production walk does -- a fake store, a
paged enumeration, an assertion on the walk -- would put the walk between the test
and the thing the test is about, and every measure test would be three tests: one
for the walk, one for the measure, and one for the interaction. So the corpus layer
has its own fixtures in :mod:`tests.fakes` and this module has its own, and the two
do not overlap: that one fakes a *store*, this one builds a *read*.

**The builder is frozen and every knob is settable, including the awkward ones.**
``store_total`` disagreeing with ``enumerated``, ``completeness`` being anything but
``COMPLETE``, a truncation with a size: these are the cases a measure's
:meth:`~tenbin.corpus.snapshot.CorpusSnapshot`-derived text exists to handle, and a
fixture that could only build a whole corpus could not test any of them. The
``read_at`` default is fixed rather than ``datetime.now()`` so a figure is
comparable across runs, which is what lets a test assert on rendered text.

**The record helper goes through the real parser.** :func:`build_record` builds
frontmatter the way Kojutsu writes it -- typed values stringified, keys absent
where the payload carried nothing -- and hands it to
:meth:`tenbin.corpus.record.Record.from_document`, so a test that wants a record
with a stated ``record_kind`` is exercising the same read path a real snapshot uses.
The alternative, constructing a :class:`~tenbin.corpus.record.Record` directly, can
produce a record that no document could ever produce, and a test asserting on it
would be asserting on a fiction.

The defaults are the new reality rather than the old one: a record states its
``record_kind`` and carries no ``capture_source``, because a document that says
nothing about its own provenance is a real and common document and a fixture that
quietly supplied ``webhook`` would hide the handling of it. A test that wants a
capture says ``capture_source="webhook"``, and a reader can see it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final

from tenbin.corpus.record import Record, RecordKind
from tenbin.corpus.snapshot import Completeness, CorpusSnapshot, Truncation
from tenbin.store.client import StoreDocument, StoreError
from tests.fakes import document_id, fake_document, kojutsu_frontmatter

#: A fixed moment, so a figure rendered from a built snapshot is the same string
#: today and next week. ``datetime.now()`` in a fixture makes every assertion that
#: touches a timestamp a test that can flake for reasons nobody can see.
FIXED_READ_AT: Final[datetime] = datetime(2026, 9, 30, 12, 0, 0, tzinfo=UTC)

#: The collection the builder reads by default. Deliberately not the shipped
#: ``DEFAULT_COLLECTION``: that one is a placeholder an operator must replace, and a
#: fixture that carried it would let a test assert against the placeholder instead
#: of against a name. Nothing reads this through the environment.
DEFAULT_COLLECTION: Final[str] = "example-collection"

#: The default repository and pull request. Chosen to be unremarkable and to parse
#: the way a real id does: a repository is ``owner/name`` and contributes a slash of
#: its own, so an id built here has more segments than a naive reader allows.
DEFAULT_REPO: Final[str] = "acme/widget"
DEFAULT_PR: Final[int] = 42

#: The entry id a record gets when the caller does not name one. An answer-shaped
#: id by default, matching the default ``record_kind``, so a fixture that changes
#: neither is a coherent document rather than an answer id on a review.
DEFAULT_ENTRY_ID: Final[str] = "answer-0"


#: The document's own update time, written by default. Forwarded explicitly rather
#: than left to :func:`tests.fakes.kojutsu_frontmatter`'s default so a test can
#: write ``updated_at=None`` and get a document with no timestamp of any kind --
#: which is the only way to reach ``Record.month is None``, and therefore the only
#: way to test a measure's handling of an undated record.
DEFAULT_UPDATED_AT: Final[str] = "2026-09-28T00:00:00+00:00"

#: The id a projected decision request gets when the caller does not name one. Not
#: an entry id of its own shape: a request's id *is* its registry ``question_id``, and
#: a fixture that invented a second numbering would hide the fact that a request and an
#: answer are addressed differently.
DEFAULT_QUESTION_ID: Final[str] = "q-0001"

#: The segment a projected decision request's id carries, restated here rather than
#: imported so a fixture cannot quietly change shape if the reader's constant moves.
QUESTION_SEGMENT: Final[str] = "question"


def _pr_number(pr: int | str | None) -> int | None:
    """The pull request number an id is built from, or ``None`` for the placeholder.

    A fixture may be handed the string a stored document carries, because that is
    what Kojutsu writes and what a test copying a real frontmatter will have. A
    number that is not one becomes ``None``, which builds the ``pr-0`` placeholder
    id -- the right answer for a record with no pull request, and the reason
    ``parse_document_id`` refuses to read ``pr-0`` as number zero.
    """
    if isinstance(pr, str):
        return int(pr) if pr.isdigit() else None
    return pr


def build_record(
    *,
    entry_id: str = DEFAULT_ENTRY_ID,
    repo: str | None = DEFAULT_REPO,
    pr: int | str | None = DEFAULT_PR,
    record_kind: str | RecordKind | None = RecordKind.ANSWER,
    tags: Sequence[str] | str | None = None,
    rationale: bool = False,
    collection: str = DEFAULT_COLLECTION,
    content: str = "# body\n",
    doc_id: str | None = None,
    frontmatter: Mapping[str, Any] | None = None,
    **extras: Any,
) -> Record:
    """One record, built the way Kojutsu writes one and read back the real way.

    ``record_kind`` is written to the ``record_kind`` key, which is what makes the
    classification determined; pass ``None`` to produce a legacy document with no
    kind at all, and the classification falls back to the tag projection. ``tags``
    defaults to nothing, so a record with a stated kind and no tags is the ordinary
    case in the new reality -- an answer written by somebody who named no agent.

    Every extra is stringified, because that is what the storage boundary does and
    a fixture that handed the reader a real ``int`` would be testing a leniency the
    store does not have. An extra passed as ``None`` is left out entirely rather
    than written as the string ``"None"``, because the absence *is* the fact these
    keys exist to express.

    ``rationale=True`` puts the record in the rationale namespace, which is where
    ``asserted`` capture sources are constant and where the revision chain lives --
    a test about either needs a record addressed that way and cannot make one out
    of an answer id.
    """
    resolved_id = doc_id or document_id(repo, _pr_number(pr), entry_id, rationale=rationale)
    frontmatter_dict = dict(
        kojutsu_frontmatter(
            tags=tags if tags is not None else [],
            repo=repo,
            pr=pr,
            capture_source=None,
            updated_at=extras.get("updated_at", DEFAULT_UPDATED_AT),
        )
    )
    if record_kind is not None:
        frontmatter_dict["record_kind"] = (
            record_kind.value if isinstance(record_kind, RecordKind) else str(record_kind)
        )
    for key, value in extras.items():
        if value is not None:
            frontmatter_dict[key] = str(value)
    if frontmatter is not None:
        frontmatter_dict.update(frontmatter)
    document = fake_document(resolved_id, frontmatter_dict, collection=collection, content=content)
    return Record.from_document(
        StoreDocument(
            id=document["id"],
            collection=document["collection"],
            path=document["path"],
            content=document["content"],
            content_hash=document["contentHash"],
            updated_at=document["updatedAt"],
            revision=document["revision"],
            frontmatter=document["frontmatter"],
        )
    )


def legacy_record(**overrides: Any) -> Record:
    """A record from before ``record_kind`` was written, classified from its tags.

    Exists as a named function rather than as ``build_record(record_kind=None)`` at
    each call site because the distinction is the *subject* of those tests and a
    call site that says ``record_kind=None`` reads as an omission rather than as a
    deliberate legacy document. The wrapper is the documentation.
    """
    return build_record(record_kind=None, **overrides)


def build_answers(
    count: int,
    *,
    entry_prefix: str = "answer",
    record_kind: str | RecordKind | None = RecordKind.ANSWER,
    **overrides: Any,
) -> tuple[Record, ...]:
    """``count`` records with distinct entry ids, for a test about a total.

    The pull request number advances with the index so the records are separate
    *changes*, not five records about one change -- which matters for every measure
    here that groups by change, and is the difference between a measure being tested
    and a measure being fed a corpus it will never see.
    """
    return tuple(
        build_record(
            entry_id=f"{entry_prefix}-{index:04d}",
            pr=index + 1,
            record_kind=record_kind,
            **overrides,
        )
        for index in range(count)
    )


def decision_request(
    question_id: str = DEFAULT_QUESTION_ID,
    *,
    status: str | None = "answered",
    repo: str | None = DEFAULT_REPO,
    pr: int | str | None = DEFAULT_PR,
    created_at: str | None = "2026-09-01T00:00:00+00:00",
    answered_at: str | None = "2026-09-02T00:00:00+00:00",
    status_as_of: str | None = None,
    question_author: str | None = "agent",
    attempts: int | None = 1,
    answer_comment_id: int | None = None,
    session_id: str | None = None,
    frontmatter: Mapping[str, Any] | None = None,
) -> Record:
    """One projected decision request, in the shape Kojutsu's projection writes.

    A named function rather than ``build_record`` with arguments spelled out at each
    call site, because the *absence* is the subject of most of the tests that use it:
    a request carries no ``record_kind`` (Kojutsu writes the key for its four entry
    writers and not for this projection), no ``capture_source`` and no
    ``independence``, and it lives under ``question/`` rather than at the change's own
    address. A call site that said ``record_kind=None`` reads as an omission, and
    :func:`legacy_record` exists for exactly that reason.

    ``frontmatter`` writes anything else onto the document, and it is how a test
    builds the one document this fixture cannot build honestly: a request carrying
    ``claim_token``. Kojutsu's projection cannot produce that -- its model has no
    such field -- so a test that needs it is testing the reader's response to a
    document the writer should never have made.
    """
    resolved_id = f"{repo or 'unknown'}/pr-{_pr_number(pr) or 0}/{QUESTION_SEGMENT}/{question_id}"
    extras: dict[str, Any] = {
        "question_id": question_id,
        "question_status": status,
        "created_at": created_at,
        "answered_at": answered_at,
        "question_author": question_author,
        "attempts": attempts,
        "answer_comment_id": answer_comment_id,
        "session_id": session_id,
        "status_as_of": status_as_of,
    }
    return build_record(
        entry_id=question_id,
        repo=repo,
        pr=pr,
        record_kind=None,
        tags=["question", f"question_{status}"] if status else [QUESTION_SEGMENT],
        doc_id=resolved_id,
        frontmatter=frontmatter,
        **extras,
    )


@dataclass(frozen=True, kw_only=True)
class SnapshotBuilder:
    """A :class:`~tenbin.corpus.snapshot.CorpusSnapshot` a test states rather than walks.

    Frozen for the same reason a real snapshot is: it is a statement about a moment,
    and a builder a test could keep mutating would let one test's setup leak into
    the next one's figure. Every field has a default, so the ordinary case is
    :meth:`build` over some records and the awkward cases name the one field they
    are about.

    ``store_total`` defaults to the number of records, which is what makes a
    complete snapshot complete. Setting it to something else is how a test produces
    a ``STORE_TOTAL_EXCEEDED`` read, and setting ``enumerated`` apart from
    ``store_total`` is how a test produces a walk that lost a document -- the case
    that yields the most confident and most wrong number available, and the reason
    the completeness field exists at all.
    """

    records: tuple[Record, ...] = ()
    collection: str = DEFAULT_COLLECTION
    read_at: datetime = FIXED_READ_AT
    store_total: int | None = None
    enumerated: int | None = None
    completeness: Completeness = Completeness.COMPLETE
    truncation: Truncation | None = None
    error: StoreError | None = None

    def build(self) -> CorpusSnapshot:
        """The snapshot, with the counts worked out from the records where not given.

        Two derived numbers, both of which have to agree for a snapshot to claim
        completeness: ``enumerated`` is what the listing returned and ``store_total``
        is what the store said it held. A test that wants them to disagree sets one
        of them; a test that wants them to agree says nothing.
        """
        records = self.records
        return CorpusSnapshot(
            records=records,
            collection=self.collection,
            read_at=self.read_at,
            store_total=len(records) if self.store_total is None else self.store_total,
            enumerated=len(records) if self.enumerated is None else self.enumerated,
            completeness=self.completeness,
            truncation=self.truncation,
            error=self.error,
        )


def build_snapshot(
    records: Sequence[Record] = (),
    *,
    collection: str = DEFAULT_COLLECTION,
    store_total: int | None = None,
    enumerated: int | None = None,
    completeness: Completeness = Completeness.COMPLETE,
    truncation: Truncation | None = None,
    error: StoreError | None = None,
) -> CorpusSnapshot:
    """The one-call form, for a test whose subject is the figure and not the read.

    Nine lines of builder behind one function call, and the call reads as the thing
    it is: "here is a corpus, here is how far the read got". A test that needs more
    than this -- a specific ``read_at``, an error object -- constructs the builder.
    """
    return SnapshotBuilder(
        records=tuple(records),
        collection=collection,
        store_total=store_total,
        enumerated=enumerated,
        completeness=completeness,
        truncation=truncation,
        error=error,
    ).build()


def truncating_snapshot(
    records: Sequence[Record] = (),
    *,
    missing: int,
    offset: int,
    store_total: int | None = None,
) -> CorpusSnapshot:
    """A snapshot stopped by the store's offset ceiling, with the shortfall named.

    Built in one place because getting it right needs three numbers to agree: the
    store's total, what was enumerated, and the truncation's ``records_missing``.
    Three tests that each set two of them would be three tests of a fixture nobody
    fully constructed, and one of them would be asserting on a hole the snapshot
    claims is a different size.
    """
    materialized = tuple(records)
    return SnapshotBuilder(
        records=materialized,
        store_total=len(materialized) + missing if store_total is None else store_total,
        enumerated=len(materialized),
        completeness=Completeness.OFFSET_CAP,
        truncation=Truncation(
            offset_reached=offset,
            records_missing=missing,
            boundary_repositories={},
            boundary_months={},
        ),
    ).build()
