"""Walking the store into a snapshot that is honest about how far it got.

There is no aggregation in Tanseki, so every measure in this program is a
full-enumeration computation over documents fetched by hand. This module does
that walk once and returns a :class:`CorpusSnapshot` that carries, alongside the
records, the evidence a reader needs to know whether the corpus in front of them
is the corpus that exists.

**The walk never returns quietly on a failure.** A snapshot that stopped early is
returned -- a partial corpus is still data, and a caller who wants to work with
what arrived should not have to re-implement the walk -- but it is returned with
``completeness=STORE_ERROR`` carrying the exception that stopped it, and with
``enumerated`` larger than the number of records present. So a caller who ignores
``completeness`` gets a visibly incomplete object rather than a plausible one.
The one exception is the first request: an unreachable store **raises**, because
an empty snapshot from a store that was never reached is indistinguishable from a
corpus of nothing, and that distinction is the whole difference between a finding
and a failure.

**The four ways to stop are four different findings.** A store that has been
walked to the end is ``COMPLETE``. A walk that hit Tanseki's ten-thousand document
offset ceiling is ``OFFSET_CAP``, and a corpus larger than that cannot be walked
by anybody -- so the shortfall is reported with a count of what is missing and a
histogram of where the cut fell, rather than as a corpus that quietly ends. A
store whose own ``total`` disagrees with what it served is
``STORE_TOTAL_EXCEEDED`` in either direction, because a count that does not match
its own listing is not a count of anything. Everything else is an error, whether
it arrived as one or as a document that vanished.

**A hole in the middle downgrades a complete walk.** A document listed and then
absent from the ``:query`` response is a record that no longer exists, and a
snapshot missing a record is not a whole snapshot no matter how clean the walk
around it was. This is the case a strict reader is most likely to assume away,
and it is the one that produces the most confident and most wrong number: a
denominator that quietly lost its own numerator.

**The read's size is in records and its window is in time, and both are stated.**
:attr:`CorpusSnapshot.window` is derived from the records once, at construction,
so a report can say over what span these records were written rather than only how
many of them there are. It is derived rather than passed for two reasons: a caller
that could supply it could supply one that disagrees with the records, and a
snapshot rebuilt from a read model has to arrive at the same window the original
read did. A rate over three records is a rate over three records whether they
cover a decade or an afternoon, and only one of those two is what a reader needs
to know. See :mod:`tenbin.corpus.window`.

This module does not interpret records, judge provenance, or form claims about
the corpus; it moves documents and reports what happened while it did.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from tenbin.corpus.provenance import NO_MONTH, NO_REPOSITORY
from tenbin.corpus.record import Record
from tenbin.corpus.window import CorpusWindow
from tenbin.store.client import (
    MAX_PAGE_LIMIT,
    MAX_PAGE_OFFSET,
    StoreClient,
    StoreError,
    StoreResponseError,
)


class Completeness(StrEnum):
    """Why the walk stopped, which is part of every figure computed from it."""

    #: The store said there was nothing more, and its count agreed.
    COMPLETE = "complete"

    #: Tanseki's offset ceiling was reached. The corpus is larger than one pass can
    #: read, so what follows is a prefix and the truncation says how long a prefix.
    OFFSET_CAP = "offset_cap"

    #: The store's own ``total`` disagreed with the number of documents it served,
    #: in either direction. The count is not a count of this corpus, so nothing
    #: that depends on the count can be trusted -- including the decision to stop.
    STORE_TOTAL_EXCEEDED = "store_total_exceeded"

    #: The walk failed, or a listed document was not there to be fetched. The
    #: exception is carried on the snapshot so the reason survives the report.
    STORE_ERROR = "store_error"


@dataclass(frozen=True)
class Truncation:
    """What was left behind, and where the cut fell.

    ``offset_reached`` is the document offset the walk would have resumed from,
    so it is the number a reader needs in order to continue the enumeration
    rather than start again. ``records_missing`` is the size of the hole,
    computed from the store's own count, and it is the first thing a caveat
    should say. The two histograms
    describe **the last page the walk read**, not the records it never reached:
    the tail cannot be described without reading it, and pretending otherwise
    would make these look like a census of what is missing when they are evidence
    of only where the walk stopped. That is still worth having, because a cut in
    the middle of one busy repository and a cut between two are different problems
    with different fixes.
    """

    offset_reached: int
    records_missing: int
    boundary_repositories: Mapping[str, int]
    boundary_months: Mapping[str, int]


@dataclass(frozen=True)
class WalkProgress:
    """How far the walk has got, for a caller that wants to show or log it.

    A frozen object rather than positional arguments because a callback that has
    to remember which integer meant what is a callback that will be read wrongly,
    and a progress line that misreports is worse than no progress line.
    """

    enumerated: int
    store_total: int
    offset: int


@dataclass(frozen=True)
class CorpusSnapshot:
    """A read of the collection, and everything needed to doubt it.

    Immutable because a snapshot is a statement about a moment -- ``read_at`` --
    and a record that could be appended to afterwards would no longer be the
    corpus anybody read. ``read_at`` is when the walk *began*, which makes it a
    lower bound on the age of every record here and the honest reading for a
    ``STORE_ERROR`` snapshot, where some of the documents it names may not exist
    at all. The interesting field is ``completeness``: a snapshot is not a
    corpus, it is a corpus *plus* a statement about whether it is whole, and a
    type that could be constructed without that statement would let the second
    half be dropped.

    **``window`` is the corpus's span in time, and it is computed once here rather
    than per measure.** A population is a span and a count, and the two are not
    separable: a rate over three records says nothing about whether those three
    records are a decade of quiet or a single afternoon, and twenty measures
    computing the window independently is twenty windows that can disagree. It is
    ``init=False`` and derived in :meth:`__post_init__` for the same reason
    :mod:`tenbin.readmodel` derives its indexes rather than accepting them: a
    caller that could pass its own would be able to pass one that disagrees with
    ``records``, and the disagreement would be invisible -- every header would
    render from the window while every figure counted the records. Deriving it also
    means the read model needs no change to carry it: :func:`tenbin.readmodel.convert.to_snapshot`
    restores the records, and a snapshot rebuilds its window from exactly those,
    which is what makes a figure computed from a stored read report the window of
    the read that produced it rather than the window of the machine rendering it.
    """

    records: tuple[Record, ...]
    collection: str
    read_at: datetime
    store_total: int
    enumerated: int
    completeness: Completeness
    truncation: Truncation | None = None
    #: The failure that stopped the walk, when ``completeness`` is
    #: ``STORE_ERROR``. Carried rather than logged, because a snapshot that
    #: explains itself is the only one whose completeness claim can be checked.
    error: StoreError | None = None
    #: The earliest and latest moments this read's records could be placed at, the
    #: span between them, and how many of them named no moment at all. Derived from
    #: ``records`` and not supplied, so it cannot be a second account of them.
    window: CorpusWindow = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "window", CorpusWindow.from_records(self.records))

    def __hash__(self) -> int:
        # Frozen with a mapping-free payload, so the generated hash is fine; this
        # is here only to keep the error field out of the identity. A snapshot
        # read twice is the same snapshot; a snapshot that failed is a different
        # object with the same records, and hashing them alike would let a set
        # quietly collapse them.
        return hash(
            (
                self.collection,
                self.read_at,
                self.enumerated,
                self.completeness,
                self.truncation.offset_reached if self.truncation else None,
            )
        )

    @property
    def is_complete(self) -> bool:
        """Whether the walk read everything the store said it held."""
        return self.completeness is Completeness.COMPLETE


def take_snapshot(
    client: StoreClient,
    *,
    page_limit: int = MAX_PAGE_LIMIT,
    on_progress: Callable[[WalkProgress], None] | None = None,
) -> CorpusSnapshot:
    """Read the collection page by page and return what came, with how it ended.

    The first page is read outside the guarded region on purpose: if the store
    cannot be reached at all, the caller gets a :exc:`StoreError` and not a
    snapshot with zero records, because the second of those is the output this
    program most needs never to produce. Every failure after that is captured
    onto the snapshot instead, so a walk that got three quarters of the way
    through is still usable by a caller who checks ``completeness``.

    The step is the number of ids the previous page returned rather than the page
    size requested, because the offset Tanseki applies is an offset into documents
    and a short page means the store served fewer documents than asked for. A
    walk that stepped by the requested size would skip whatever the short page
    withheld.
    """
    if isinstance(page_limit, bool) or not isinstance(page_limit, int):
        raise ValueError("Page limit must be an integer.")
    if not 1 <= page_limit <= MAX_PAGE_LIMIT:
        raise ValueError(f"Page limit must be between 1 and {MAX_PAGE_LIMIT}.")

    read_at = datetime.now(UTC)
    first = client.list_documents(limit=page_limit, offset=0)
    store_total = first.total
    if store_total is None:
        # ``list_documents`` tolerates a listing with no total, because an
        # envelope shape that has varied is not a broken document. A walk cannot
        # be that tolerant, because completeness here *is* a claim about a count:
        # without the number there is nothing to compare the enumeration against,
        # and guessing would make "complete" a word this program had no basis for.
        raise StoreResponseError(
            "list_documents returned no total, so the walk cannot be checked for completeness"
        )

    records: list[Record] = []
    enumerated = 0
    store_error: StoreError | None = None
    completeness = Completeness.COMPLETE
    boundary: list[Record] = []
    offset = 0
    reached = 0
    page = first

    try:
        while True:
            if not page.ids:
                # An empty page with ``hasMore`` set is a store contradicting
                # itself, and calling that ``complete`` would report a whole
                # corpus from a response that says there is more.
                if page.has_more:
                    store_error = StoreResponseError(
                        "list_documents returned no documents while reporting more"
                    )
                    completeness = Completeness.STORE_ERROR
                break

            resolved = client.get_documents(page.ids)
            fetched = [document for document in resolved if document is not None]
            # The enumeration counts what was *listed*, and is incremented before
            # the hole check, so a document that vanished is still visible as a
            # gap between ``enumerated`` and the records present. Counting only
            # what arrived would make a snapshot with a hole in it look exactly
            # like a snapshot of a smaller corpus.
            enumerated += len(page.ids)
            boundary = [Record.from_document(document) for document in fetched]
            records.extend(boundary)

            if on_progress is not None:
                on_progress(WalkProgress(enumerated, store_total, offset))

            if len(fetched) != len(resolved):
                store_error = StoreResponseError(
                    f"{len(resolved) - len(fetched)} listed documents were not returned by the store"
                )
                completeness = Completeness.STORE_ERROR
                break

            if page.has_more is False:
                break
            # A store that does not say whether there is more is trusted up to its
            # own count and no further: that is the only other number on offer,
            # and a walk that ignored it would ask for page after page of nothing
            # until the offset ceiling.
            if page.has_more is not True and enumerated >= store_total:
                break
            next_offset = offset + len(page.ids)
            reached = next_offset
            if next_offset > MAX_PAGE_OFFSET:
                completeness = Completeness.OFFSET_CAP
                break
            offset = next_offset
            page = client.list_documents(limit=page_limit, offset=offset)
    except StoreError as exc:
        store_error = exc
        completeness = Completeness.STORE_ERROR

    # The walk ended without an error but the store's own count disagrees with
    # what it served, in either direction. Both directions are the same finding: a
    # count that does not match its own listing cannot be used to say anything
    # about completeness, including the claim just made here.
    if (
        store_error is None
        and store_total != enumerated
        and completeness is not Completeness.OFFSET_CAP
    ):
        completeness = Completeness.STORE_TOTAL_EXCEEDED

    truncation: Truncation | None = None
    if completeness is Completeness.OFFSET_CAP:
        truncation = Truncation(
            offset_reached=reached,
            records_missing=max(0, store_total - enumerated),
            boundary_repositories=_histogram(record.repo or NO_REPOSITORY for record in boundary),
            boundary_months=_histogram(record.month or NO_MONTH for record in boundary),
        )

    return CorpusSnapshot(
        records=tuple(records),
        collection=client.collection,
        read_at=read_at,
        store_total=store_total,
        enumerated=enumerated,
        completeness=completeness,
        truncation=truncation,
        error=store_error,
    )


def _histogram(values: Iterable[str]) -> Mapping[str, int]:
    """Count text values, most frequent first, frozen against later mutation."""
    counts = Counter(values)
    return MappingProxyType(dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))))


@runtime_checkable
class Snapshot(Protocol):
    """What a figure needs in order to be doubted.

    This is the seam the whole program is built around, expressed as a type. A
    figure cannot be constructed without a snapshot, and the snapshot it is checked
    against is *this* protocol rather than :class:`CorpusSnapshot` specifically,
    because the question "was this corpus whole?" is not a question about Tanseki.
    It is a question about any read, and a second read that answered it with a
    constant would have been a bug, not a convenience.

    The members are exactly the integrity surface -- what was read, when, how many
    the source said there were, how far the walk got, and what stopped it. They
    are deliberately *not* the records. A snapshot's records are Kojutsu
    documents, and a second source's are not documents at all; a protocol
    demanding ``records`` would either be wrong about the second source or would
    quietly require it to flatten its facts into Kojutsu's shape. Which is the
    mistake the placement decision for this work named in advance: ``record_kind``
    is a closed five-value vocabulary, and patch pushes and status transitions do
    not fit inside it. They get their own record type and share this.

    ``records`` is also absent because it is the one member that decides what the
    snapshot *is*. Everything here is what can be known about it without opening
    it, which is exactly the part a renderer is entitled to rely on.

    ``window`` is required, not merely available. A population is a span and a
    count and the two are not separable, so a source that cannot state its span
    in time has not finished describing its own population.
    """

    #: The moment the read began -- a lower bound on the age of everything in it.
    read_at: datetime
    #: What the source said it held, before the walk.
    store_total: int
    #: What the walk actually reached.
    enumerated: int
    #: Why the walk stopped. ``COMPLETE`` means nothing stopped it.
    completeness: Completeness
    #: How long a prefix this is, when ``completeness`` says it is a prefix.
    truncation: Truncation | None
    #: The failure that stopped the walk, when one did.
    error: StoreError | None
    #: The earliest and latest moments anything here could be placed at.
    window: CorpusWindow
    #: Which collection was read, for a reader who has more than one.
    collection: str
