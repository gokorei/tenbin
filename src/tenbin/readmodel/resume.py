"""Finishing a read that the offset ceiling cut short, in more than one window.

**The read model is the artefact that was supposed to remove the ceiling, and it
inherited it.** Tanseki caps a listing ``offset`` at :data:`MAX_PAGE_OFFSET`, so
:func:`tenbin.corpus.snapshot.take_snapshot` stops at the ceiling and
:func:`tenbin.readmodel.model.build` keeps what it reached. At 10,001 documents
``tenbin read-model build`` therefore wrote a model of the first 10,000 and called
it a read of the collection. The model says ``offset_cap``, so a careful reader
catches it once -- and a read model exists to be read *repeatedly without re-walking
the store*, so the second reader, and the hundredth, do not. The failure is not a
wrong number on the first run. It is a number that is right once and then invisible.

**The fix is more than one walk, and the hard part of it is not the walking.** A
walk that stops at the ceiling can be continued: the model records the offset it
reached, and the next window starts inside the range the last one covered rather
than past it. But ``offset`` is **not a cursor**. Tanseki does not document the order of
its listing as stable, and a document written between two windows shifts everything
behind it, so a document can be listed twice or not at all, and neither is detectable
from the offsets alone. **The document id is the only thing here that does not move,
so the merge is keyed on it and the offset is treated as a place to look rather than
a place to resume from** -- a window deliberately re-lists the last page of the
window before it, so a document that moved cannot fall between two windows without
being listed twice, and being listed twice is free because the id is recognised.

**A document listed twice is counted once, and one that vanished is a finding rather
than a deduplication success.** A re-seen document is folded into the record it
already has, so it cannot reach a denominator twice -- the model itself refuses two
records under one id, and the windows record how many ids were listed against how
many documents were added, so the reconciliation is checkable rather than promised.
But a document listed in an earlier window and *absent* from the later listing of the
same range is a different fact: something either reindexed it or rewrote it, the
corpus does not say which, and a read that quietly dropped it would be reporting
coverage it never established. That is a :class:`MergeFinding`, and it downgrades the
model's completeness so that a caller who ignores the findings still gets a visibly
incomplete read rather than a clean one.

**Resuming is not updating, and the difference is the whole design.** A resumed build
still produces a *complete* model -- it just gets there in more than one walk -- and
:func:`tenbin.readmodel.model.write` still replaces the whole file, so a build that
fails leaves the previous model exactly as a whole build that fails leaves it. There
is no merge of two models, no refresh in place, and no way for this module to leave
half a read on disk: it holds no path and writes nothing. A maintained model is a
second source of truth nobody reconciles, and resuming is how a read that was cut
short gets finished rather than how a model gets kept up to date.

**What a resumable walk cannot do, stated here because the ticket's premise
overstated it.** The seam will not serve an ``offset`` above
:data:`MAX_PAGE_OFFSET` at all, so the documents *addressable* by any sequence of
legal requests are the first :func:`reachable_ceiling` of them -- the ceiling plus
one page, and one more page buys nothing because the last page already sits at the
ceiling. A collection larger than that is unreadable in whole by anybody, however
many times the walk is repeated; what repeating it buys is the documents that *moved
into* the addressable range, and the exact size of the hole that is left. This
module therefore reports that hole rather than closing it with a walk, because a
read that counted 10,500 of 10,600 documents and said so is a finding and one that
said ``complete`` would be the defect this program exists to stop.

**What this module notably does not do:** it does not write anything, hold a path, or
decide whether a resumed model should be kept -- the caller does, through
:func:`tenbin.readmodel.model.write`, which replaces. It does not retry a failed
window beyond what the client already does, and it does not re-read a document it
already holds: a re-listed document is *compared*, not re-fetched, so a document
rewritten in place between two windows keeps the version the earlier window read.
That is a choice, and the wrong one to hide -- an in-place rewrite is invisible to
this walk, because the id and its position are both unchanged -- but the alternative,
overwriting a record because it was read twice, is an update, and the differences it
would make to a stored read depend on which window happened to run last. The store's
own count is the only other number on offer, and it is recorded per window so a
collection that moves under a read is visible.
"""

from __future__ import annotations

from collections.abc import Iterator, MutableMapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Final

from tenbin.claims.mechanism import require_text
from tenbin.corpus.provenance import NO_MONTH, NO_REPOSITORY
from tenbin.corpus.record import Record
from tenbin.corpus.snapshot import Completeness, Truncation

# ``histogram`` is imported from the measures layer rather than reimplemented, and the
# dependency is deliberate: it lives there precisely so that two callers cannot define a
# bucket differently, and a second copy of "count values, most frequent first" in this
# module is the private twin that :func:`tenbin.measures.filtering.named_repository`
# already argues about. The read model importing one counting helper from below it costs
# less than a read that describes where a truncation fell in words nothing else uses.
from tenbin.measures.base import histogram
from tenbin.readmodel.model import SCHEMA_VERSION, ReadModel, StoredFailure, WalkWindow
from tenbin.store.client import (
    MAX_PAGE_LIMIT,
    MAX_PAGE_OFFSET,
    DocumentPage,
    StoreClient,
    StoreError,
    StoreResponseError,
)

#: How many windows one build will walk before it gives up on a store that is moving
#: faster than it can be read.
#:
#: A build that is going well takes three or four: the window that walks the listing
#: from the top, the window that finishes past the ceiling or re-tries a failed page,
#: and one confirmation pass over the seam that added nothing. **Sixteen is therefore
#: roughly four times what a healthy build needs**, and reaching it means every window
#: since the second added a document -- which is a store being written to as fast as
#: it is read, not a walk that needs longer. The bound is on the windows walked *in
#: this build* rather than on the model's total, so continuing a model that already
#: has a dozen windows is not itself a reason to stop.
MAX_RESUME_WINDOWS: Final[int] = 16

#: How many affected document ids a finding names when it renders. Enough to go and
#: look at one, not so many that the sentence stops being a sentence: a reindex can
#: move thousands, and a list of thousands is not something anybody reads.
SEAM_IDS_SHOWN: Final[int] = 5

#: What a build with nothing to report says, so a caller never has to invent the
#: absence itself. A finding list that renders as an empty string is a build whose
#: output nobody can tell from one that forgot to look.
NO_FINDINGS: Final[str] = (
    "No findings: the windows this read walked agree with the store's own count and with "
    "each other, and no document listed in one window was missing from the next."
)


class MergeFindingKind(StrEnum):
    """What a resumed build noticed that a single walk could not have.

    A closed set because each member names a different thing for an operator to go
    and do, and a finding whose remedy is "look at it" is a finding that has not
    decided anything.
    """

    #: A document the previous window listed was not listed again when that range was
    #: re-listed. Something reindexed it or rewrote it, the corpus does not say which,
    #: and the read is not confirmed to hold what the store holds.
    DOCUMENT_MOVED = "document_moved"

    #: The store's own count was not the same in every window. The collection changed
    #: while it was being read, which is the condition
    #: :class:`~tenbin.corpus.snapshot.Completeness` already names as
    #: ``STORE_TOTAL_EXCEEDED`` -- a count that does not match its own listing.
    STORE_TOTAL_CHANGED = "store_total_changed"

    #: The walk stopped because re-listing the seam added no document, while the store
    #: was still claiming there were more. The collection is larger than the seam can
    #: address, and no number of walks fixes that.
    SEAM_EXHAUSTED = "seam_exhausted"

    #: The walk was cut off by :data:`MAX_RESUME_WINDOWS` with every window still
    #: adding documents: the store is being written to at least as fast as it is read.
    WINDOW_LIMIT_REACHED = "window_limit_reached"


@dataclass(frozen=True)
class MergeFinding:
    """One thing the merge noticed, stated so somebody can act on it.

    A finding rather than a log line or a number, because none of these can be
    repaired by looking harder at the model: a document that moved and a collection
    that grew are facts about the store, and the only honest response to either is to
    say so where a report will carry it. ``doc_ids`` is empty for the three kinds that
    are about counts and seams rather than about documents, and is required rather
    than optional for the fourth -- an empty tuple is the honest "the seam is
    exhausted", and a missing field would be a finding that could not say what it
    found.
    """

    kind: MergeFindingKind
    doc_ids: tuple[str, ...]
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, MergeFindingKind):
            raise ValueError(
                f"MergeFinding.kind must be a MergeFindingKind, not {self.kind!r}; a bare string "
                "would restore as a finding this program never reached."
            )
        if not isinstance(self.doc_ids, tuple) or not all(
            isinstance(doc_id, str) and doc_id.strip() for doc_id in self.doc_ids
        ):
            raise ValueError("MergeFinding.doc_ids must be a tuple of document ids.")
        # The project's own blank-string rule rather than a local one, so a finding
        # cannot render as a heading with an empty paragraph under it.
        require_text("MergeFinding", "detail", self.detail)

    def render_text(self) -> str:
        """The finding as two lines: what it is, and which documents it is about.

        The ids are truncated rather than listed, and the count is exact, because the
        count is what somebody needs to decide whether to go and look while the first
        few are what they need once they have decided to.
        """
        lines = [f"{self.kind.value}: {self.detail}"]
        if self.doc_ids:
            shown = ", ".join(self.doc_ids[:SEAM_IDS_SHOWN])
            noun = "document" if len(self.doc_ids) == 1 else "documents"
            remainder = len(self.doc_ids) - SEAM_IDS_SHOWN
            suffix = "" if remainder <= 0 else f", and {remainder:,} more"
            lines.append(f"  {len(self.doc_ids):,} {noun} affected, including: {shown}{suffix}")
        return "\n".join(lines)


@dataclass(frozen=True)
class ResumeOutcome:
    """A resumed build's model, and the findings the merge produced on the way to it.

    **The findings are a return value and not a field of the model, and the model's
    completeness is the safety net for a caller who drops them.** They belong here
    because they are about the merge rather than about the read: which documents moved
    between two windows is a fact about how this read was assembled, and a read model
    stores the read. A caller that persists a model without printing its findings loses
    the detail, and the completeness verdict is what stops that from becoming a clean
    number: every finding here downgrades the verdict to one this package already
    renders a sentence for.
    """

    model: ReadModel
    findings: tuple[MergeFinding, ...]
    #: The offset this build started at, or ``None`` when it walked from the top. The
    #: one number that says whether a model was rebuilt or continued, which is the
    #: difference an operator is asked to reason about when they pass ``--resume``.
    resumed_from: int | None

    @property
    def resumed(self) -> bool:
        """Whether this build continued an existing model rather than starting a walk."""
        return self.resumed_from is not None

    @property
    def windows(self) -> tuple[WalkWindow, ...]:
        """The windows the read was walked in, which is the model's own record of them."""
        return self.model.windows

    def render_findings(self) -> str:
        """Every finding as text, or the sentence that says there were none.

        The absence is stated rather than rendered as an empty string, for the reason
        the rest of this program states absences: a caller cannot tell a build that
        found nothing from a build that never looked.
        """
        if not self.findings:
            return NO_FINDINGS
        return "\n".join(finding.render_text() for finding in self.findings)


def reachable_ceiling(page_limit: int = MAX_PAGE_LIMIT) -> int:
    """How many documents any walk can ever address, however many times it is repeated.

    **The seam's ceiling is on the offset, so repeating a walk cannot get past it.**
    The largest offset the store will serve is :data:`MAX_PAGE_OFFSET`, and the last
    page read at that offset returns at most ``page_limit`` documents, so the whole
    addressable corpus is the ceiling plus one page. Walking the seam again re-lists
    that same final page: it finds documents the listing *moved into* the addressable
    range, and it cannot find one that sits beyond it.

    Named as a function rather than left as arithmetic in prose because the number is
    the honest answer to "how large a corpus can this program read", and a limit that
    lives only in a docstring is a limit somebody will re-derive wrongly.
    """
    _require_page_limit(page_limit)
    return MAX_PAGE_OFFSET + page_limit


def build_across_windows(
    client: StoreClient,
    *,
    page_limit: int = MAX_PAGE_LIMIT,
    existing: ReadModel | None = None,
) -> ResumeOutcome:
    """Walk the collection in as many windows as it takes, and keep the whole read.

    With no ``existing`` model this is a walk that behaves like
    :func:`~tenbin.corpus.snapshot.take_snapshot` -- same order of requests, same
    verdict, same first-request-raises behaviour -- and it exists as its own function
    rather than as a call to that one because it also has to start at an offset other
    than zero, and :func:`~tenbin.corpus.snapshot.take_snapshot` cannot be asked to.

    **The first request of a build is outside the guarded region, exactly as it is in
    the whole walk.** A store that cannot be reached raises rather than returning a
    model, because a model of nothing is indistinguishable from a corpus of nothing --
    and when ``existing`` was given, the previous model is already on disk for
    whoever is running the build to keep reading. Every later failure is carried on
    the model instead, with its class and message, so a walk that got most of the way
    is still usable by a caller that checks ``completeness``.

    **A collection is walked in more than one window only when it has to be.** A window
    the store says has nothing more behind ends the build, which is the ordinary case
    and the one :mod:`tenbin.readmodel.convert`'s equality depends on. A window stopped
    by the ceiling is followed by one that re-lists from inside the range it covered,
    and the build ends when re-listing the same range adds nothing -- at which point the
    store's own count is all there is to say whether what was read is the whole
    collection. **A build cannot walk past :func:`reachable_ceiling` documents however
    many windows it takes**, because the seam refuses the offsets that would reach them,
    and a collection larger than that is reported with the size of the hole rather than
    described as read.
    """
    _require_page_limit(page_limit)
    if existing is not None and existing.collection != client.collection:
        # Refused rather than merged: two collections in one read is the thing the
        # corpus layer refuses to call a corpus, and a resume is exactly the moment
        # somebody would pass the wrong file.
        raise ValueError(
            f"cannot resume a read of {existing.collection!r} against a store serving "
            f"{client.collection!r}; a read model names one collection and a resumed read is "
            "the same read continued, not two of them joined."
        )

    held: dict[str, Record] = dict(existing.by_id) if existing is not None else {}
    windows: list[WalkWindow] = list(existing.windows) if existing is not None else []
    findings: list[MergeFinding] = []
    start = _relist_offset(
        offset_end=windows[-1].offset_end if windows else 0, page_limit=page_limit
    )
    resumed_from = None if existing is None else start
    # The read's moment is the moment the *read* began, which for a resumed read is
    # the earlier one. Every record's age is measured from it, so taking the later
    # moment would make documents read ten minutes ago look as old as the model file.
    read_at = existing.read_at if existing is not None else datetime.now(UTC)

    previous: _WindowWalk | None = None
    last: _WindowWalk
    walked_here = 0
    while True:
        current = _walk_window(
            client, start=start, page_limit=page_limit, held=held, first_window=walked_here == 0
        )
        walked_here += 1
        if previous is not None:
            findings.extend(_seam_findings(previous, current))
        windows.append(current.window)
        previous = current
        if current.window.completeness is Completeness.COMPLETE:
            break
        if current.window.completeness is Completeness.STORE_ERROR:
            break
        # The ceiling stopped this window and the store may still hold documents, so
        # the next one starts inside the range this one covered rather than past it.
        next_start = _relist_offset(offset_end=current.window.offset_end, page_limit=page_limit)
        if walked_here >= MAX_RESUME_WINDOWS:
            findings.append(_window_limit_finding(current.window))
            break
        if next_start <= start and current.window.documents == 0:
            findings.append(
                _seam_exhausted_finding(current.window, page_limit=page_limit, held=len(held))
            )
            break
        start = next_start

    last = previous
    assert last is not None  # the loop above walks at least one window, and assigns every pass
    # The store's most recent account of the collection, which is the last window that
    # got one: a window whose very first request failed has no account to give, and
    # inventing it from the window before would be a count this read never heard.
    store_total = next(
        (window.store_total for window in reversed(windows) if window.store_total is not None),
        None,
    )
    if store_total is None:
        # Unreachable: the first window reads its first page outside the guarded region
        # and a listing with no total is refused there, so a build either has a count
        # or has already raised. Asserted rather than handled, so a change to that rule
        # fails here instead of quietly building a model whose count came from nowhere.
        raise StoreResponseError("a walk window recorded no store total, so it cannot be compared")
    findings.extend(_store_total_findings(windows))
    completeness, truncation = _verdict(
        last, findings=findings, held=len(held), store_total=store_total
    )
    return ResumeOutcome(
        model=ReadModel(
            schema_version=SCHEMA_VERSION,
            collection=client.collection,
            read_at=read_at,
            # The number of *distinct* documents this read holds, which is the whole of
            # what a denominator may be computed over. The count of ids the windows
            # listed is deliberately not used here: it is larger by exactly the
            # re-listings this walk made on purpose, and the completeness measure's
            # denominator is the store's own count, so a numerator that exceeded it
            # would render as "the store served more documents than its own count
            # reported" -- an incoherence this walk would have caused and would then
            # have to explain. Nothing is lost: the listing count is the sum of
            # ``window.enumerated``, and the sum of ``window.documents`` is this
            # number, so their difference is every document listed more than once.
            enumerated=len(held),
            store_total=store_total,
            completeness=completeness,
            windows=tuple(windows),
            records=tuple(held.values()),
            truncation=truncation,
            failure=StoredFailure.from_error(last.error),
        ),
        findings=tuple(findings),
        resumed_from=resumed_from,
    )


@dataclass(frozen=True)
class _WindowWalk:
    """One window's results, plus the two things only the caller can reconcile.

    ``listed`` and ``listed_last_page`` are the window's own record of what the store
    showed it, kept for the comparison with the *next* window: the last page is the
    seam that the next window re-lists, and the whole listing is what a document from
    the seam is checked against. ``boundary`` is the last page's records, which is
    where a truncation's histograms come from -- the same page
    :func:`~tenbin.corpus.snapshot.take_snapshot` describes, and for the same reason:
    the tail cannot be described without reading it.
    """

    window: WalkWindow
    listed: tuple[str, ...]
    listed_last_page: tuple[str, ...]
    boundary: tuple[Record, ...]
    error: StoreError | None


def _walk_window(
    client: StoreClient,
    *,
    start: int,
    page_limit: int,
    held: MutableMapping[str, Record],
    first_window: bool,
) -> _WindowWalk:
    """One window: list from ``start`` until the store, the ceiling or a failure ends it.

    **The page loop is :func:`~tenbin.corpus.snapshot.take_snapshot`'s, rule for
    rule**, and it has to be: a resumed read and a whole read of the same collection
    must produce the same records, the same counts and the same verdict, or the
    equality :mod:`tenbin.readmodel.convert` is built on is a claim rather than a
    fact. The step is the number of ids the previous page returned rather than the
    page size asked for, the ``has_more`` handling is the same, and the ceiling is
    compared against the *absolute* offset rather than one relative to ``start`` --
    which is the one line that has to differ, because the ceiling is a property of the
    store and not of the window.

    ``first_window`` decides where the guarded region starts, and the distinction is
    the one :func:`~tenbin.corpus.snapshot.take_snapshot` draws. The first request of
    a *build* raises, because a model of nothing is indistinguishable from a corpus of
    nothing. Every request after that is captured onto the window instead, **including
    the first request of the second and later windows**: a build that has read ten
    thousand documents and then meets a store that has gone away has a partial read,
    which is still data, and raising would throw it away and leave the caller with the
    model that was already on disk.

    **Only ids this read does not already hold are fetched.** A re-listed document is
    compared, not re-read, which is the point of keying the merge on ids: the round
    trip is not spent and the record cannot be replaced by a later window. The
    documents that were known are still counted in the window's ``enumerated``,
    because that is what the store listed.
    """
    # A page carrying no count cannot be checked for completeness, and this walk claims
    # completeness. Refused in the one place both the whole walk and this one refuse it,
    # so a later window's missing count is a failure of that window rather than a
    # reason to throw away the windows that came before it.
    page: DocumentPage | None = (
        _listed(client, offset=start, page_limit=page_limit) if first_window else None
    )
    offset = start
    enumerated = 0
    documents = 0
    completeness = Completeness.COMPLETE
    error: StoreError | None = None
    listed: list[str] = []
    last_page: tuple[str, ...] = ()
    boundary: tuple[Record, ...] = ()
    store_total: int | None = None
    try:
        while True:
            if page is None:
                page = _listed(client, offset=offset, page_limit=page_limit)
            current = page
            if current.total is None:
                # Unreachable through :func:`_listed`, which refuses a page with no
                # count. Checked rather than asserted so the type carries the invariant
                # and a future change to that helper cannot turn into a comparison
                # against ``None``.
                raise StoreResponseError(
                    "list_documents returned no total, so the walk cannot be checked for "
                    "completeness"
                )
            store_total = current.total
            if not current.ids:
                # An empty page with ``has_more`` set is a store contradicting itself,
                # and calling that complete would report a whole corpus from a
                # response that says there is more.
                if current.has_more:
                    error = StoreResponseError(
                        "list_documents returned no documents while reporting more"
                    )
                    completeness = Completeness.STORE_ERROR
                break

            # ``dict.fromkeys`` because a store that lists one id twice in a page would
            # otherwise be fetched twice and counted twice, and the whole of this
            # module's argument is that an id is what identifies a document.
            unknown = tuple(dict.fromkeys(doc_id for doc_id in current.ids if doc_id not in held))
            resolved = client.get_documents(unknown)
            present = tuple(document for document in resolved if document is not None)
            if len(present) != len(unknown):
                error = StoreResponseError(
                    f"{len(unknown) - len(present)} listed documents were not returned by the store"
                )
                completeness = Completeness.STORE_ERROR
            for document in present:
                # First seen wins. A document listed again is already in ``held`` and is
                # not fetched at all, so this only decides between two fetches of the
                # same id inside one page.
                held.setdefault(document.id, Record.from_document(document))
            enumerated += len(current.ids)
            documents += len(present)
            listed.extend(current.ids)
            last_page = current.ids
            boundary = tuple(held[doc_id] for doc_id in current.ids if doc_id in held)
            next_offset = offset + len(current.ids)

            if completeness is Completeness.STORE_ERROR:
                break
            if current.has_more is False:
                break
            # A store that does not say whether there is more is trusted up to its own
            # count and no further. The count compared here is the absolute offset,
            # because that is how many documents the store has served this read across
            # every window -- a relative one would let a resumed window walk past the
            # store's own total on the strength of a page limit.
            if current.has_more is not True and next_offset >= store_total:
                break
            if next_offset > MAX_PAGE_OFFSET:
                completeness = Completeness.OFFSET_CAP
                break
            offset = next_offset
            page = None
    except StoreError as exc:
        error = exc
        completeness = Completeness.STORE_ERROR

    return _WindowWalk(
        window=WalkWindow(
            offset_start=start,
            enumerated=enumerated,
            documents=documents,
            store_total=store_total,
            completeness=completeness,
        ),
        listed=tuple(listed),
        listed_last_page=last_page,
        boundary=boundary,
        error=error,
    )


def _listed(client: StoreClient, *, offset: int, page_limit: int) -> DocumentPage:
    """One page of the listing, with the count that makes completeness checkable.

    The refusal is :class:`~tenbin.store.client.StoreResponseError` for the reason the
    whole walk gives it: a listing with no ``total`` is a shape that has varied, and
    ``list_documents`` tolerates it, but a walk cannot -- completeness here *is* a claim
    about a count, and without the number there is nothing to compare the enumeration
    against.
    """
    page = client.list_documents(limit=page_limit, offset=offset)
    if page.total is None:
        raise StoreResponseError(
            "list_documents returned no total, so the walk cannot be checked for completeness"
        )
    return page


def _relist_offset(*, offset_end: int, page_limit: int) -> int:
    """Where a window starts when it is continuing the one before it.

    **One page back from where the previous window's range ended**, which is the
    offset that page started at, and the reason is the one this module is built on: a
    window that started *past* the previous range would trust ``offset`` as a cursor,
    and it is not one. Re-listing the last page means a document the listing moved
    between two windows is listed twice -- recognised by id, free -- rather than
    listed not at all, which nothing would notice.

    It has a second consequence worth stating because it is not the reason and is more
    useful: a window that stopped *on* a failing page is retried rather than skipped.
    A walk that died at offset 6,000 resumes at 5,900.

    Clamped twice, and both clamps are the seam rather than taste. The store refuses an
    offset above :data:`MAX_PAGE_OFFSET`, and a resumed walk that asked for one would
    raise out of the client; a negative offset is not an offset, which happens when the
    previous window was shorter than a page.

    **The overlap is one page wide, and that is a bound rather than a guarantee.** A
    document that moves further than a page between two windows is not covered by the
    re-listing, and the only other number available to catch it is the store's own
    count -- which is why each window records the total it saw.
    """
    return max(0, min(MAX_PAGE_OFFSET, offset_end - page_limit))


def _seam_findings(previous: _WindowWalk, walked: _WindowWalk) -> Iterator[MergeFinding]:
    """The documents the previous window's seam listed and this one did not list again.

    A finding and not a deduplication success, and the difference is the whole point
    of comparing at all: a document that was re-listed is uninteresting, while one
    that *disappeared* from a range that was just re-listed was either reindexed or
    rewritten, the corpus does not say which, and the read cannot be confirmed to hold
    what the store holds. Only the previous window's **last page** is checked, because
    that is the only part of the previous read this window re-listed; movement
    elsewhere in the read is not detectable from ids, and saying otherwise would be
    claiming a check this walk does not perform.
    """
    listed_again = set(walked.listed)
    moved = tuple(doc_id for doc_id in previous.listed_last_page if doc_id not in listed_again)
    if not moved:
        return
    noun = "document was" if len(moved) == 1 else "documents were"
    yield MergeFinding(
        kind=MergeFindingKind.DOCUMENT_MOVED,
        doc_ids=moved,
        detail=(
            f"{len(moved):,} {noun} listed in the seam of the window ending at offset "
            f"{previous.window.offset_end:,} and not listed again when offsets "
            f"[{walked.window.offset_start}, {walked.window.offset_end}) were re-read. Something "
            "reindexed or rewrote them and the corpus does not say which, so this read is not "
            "confirmed to hold what the store holds."
        ),
    )


def _store_total_findings(windows: Sequence[WalkWindow]) -> Iterator[MergeFinding]:
    """One finding per pair of windows whose store counts disagree.

    A window's ``store_total`` is the store's own account of the collection *while
    that window ran*, and two windows reporting different numbers is the store
    disagreeing with itself inside one read. That is the finding rather than a
    tolerance, because a corpus that grows under a walk was not read whole, and a
    corpus that shrinks under one leaves the model holding documents the store no
    longer lists -- which is why the records already read are kept rather than
    dropped, and why the verdict cannot be ``complete``.
    """
    for earlier, later in zip(windows, windows[1:], strict=False):
        if earlier.store_total is None or later.store_total is None:
            continue
        if earlier.store_total == later.store_total:
            continue
        yield MergeFinding(
            kind=MergeFindingKind.STORE_TOTAL_CHANGED,
            doc_ids=(),
            detail=(
                f"the store reported {earlier.store_total:,} documents for offsets "
                f"[{earlier.offset_start}, {earlier.offset_end}) and "
                f"{later.store_total:,} for offsets [{later.offset_start}, {later.offset_end}). The "
                "collection changed while it was being read, and a corpus that moves under a "
                "read is not a corpus that was read whole."
            ),
        )


def _seam_exhausted_finding(window: WalkWindow, *, page_limit: int, held: int) -> MergeFinding:
    """The walk ran out of addressable offsets with the store still claiming more.

    The counts in the sentence are the store's own total and the number of distinct
    documents this read holds, because the difference between them *is* the finding: it
    is the size of the hole, stated by the one pair of numbers a reader can check. The
    window's own ``documents`` is not in it, because that is how many the last window
    added, which is zero by the only condition under which this finding is raised.
    """
    return MergeFinding(
        kind=MergeFindingKind.SEAM_EXHAUSTED,
        doc_ids=(),
        detail=(
            f"re-listing the seam at offset {window.offset_start:,} added no document, while the "
            f"store still reported {window.store_total or 0:,} documents against the {held:,} this "
            f"read holds. The seam refuses a listing offset above {MAX_PAGE_OFFSET:,}, so no more "
            f"walks can reach the rest: the collection is larger than the "
            f"{reachable_ceiling(page_limit):,} documents one read can address."
        ),
    )


def _window_limit_finding(window: WalkWindow) -> MergeFinding:
    """The build ran out of windows while the store was still producing documents."""
    return MergeFinding(
        kind=MergeFindingKind.WINDOW_LIMIT_REACHED,
        doc_ids=(),
        detail=(
            f"{MAX_RESUME_WINDOWS} windows were walked and every one of them after the first "
            f"added documents, the last ending at offset {window.offset_end:,}. The store is "
            "being written to at least as fast as it is read, so there is no point at which a "
            "further walk would finish."
        ),
    )


def _verdict(
    walked: _WindowWalk,
    *,
    findings: Sequence[MergeFinding],
    held: int,
    store_total: int,
) -> tuple[Completeness, Truncation | None]:
    """The verdict on the read as a whole, and the truncation that goes with it.

    **The order of the four cases is the argument, and it is not alphabetical.** A
    walk that failed is a failure whatever else is true of the counts, so
    ``STORE_ERROR`` comes first and a caller is told why rather than how much. A walk
    that stopped at the ceiling is a prefix with a *known* size, which is more useful
    than a count contradiction and is reported as one -- the shortfall is a real
    number here, because the store's count and the number of documents this read holds
    are two counts of the same collection taken moments apart. Only then does a
    disagreement count for what it is: the store's own count does not match this read,
    which is :class:`~tenbin.corpus.snapshot.Completeness`'s existing verdict for a
    corpus with no agreed size, and it is what a moved document or a store whose count
    moved between windows produces.

    ``records_missing`` is computed against the documents this read *holds* rather
    than against the sum of what the windows listed. Those differ by the re-listings
    the walk made on purpose, and subtracting the larger number from the store's count
    would under-report the hole -- in the ordinary case by a page per window, which is
    a hole reported as smaller than it is, on the one sentence a reader is given about
    how much of the corpus is missing.
    """
    if walked.error is not None or walked.window.completeness is Completeness.STORE_ERROR:
        return Completeness.STORE_ERROR, None
    if walked.window.completeness is Completeness.OFFSET_CAP:
        return Completeness.OFFSET_CAP, Truncation(
            offset_reached=walked.window.offset_end,
            records_missing=max(0, store_total - held),
            boundary_repositories=histogram(
                record.repo or NO_REPOSITORY for record in walked.boundary
            ),
            boundary_months=histogram(record.month or NO_MONTH for record in walked.boundary),
        )
    moved = any(
        finding.kind in (MergeFindingKind.DOCUMENT_MOVED, MergeFindingKind.STORE_TOTAL_CHANGED)
        for finding in findings
    )
    if moved or held != store_total:
        return Completeness.STORE_TOTAL_EXCEEDED, None
    return Completeness.COMPLETE, None


def _require_page_limit(page_limit: int) -> None:
    """Refuse a page size the store would answer with a 400, before the first request.

    The same bounds and the same messages as
    :func:`~tenbin.corpus.snapshot.take_snapshot`, restated rather than imported
    because that check is inline in a function this module cannot call and a second
    copy of a bound is better than a walk that discovers the store's limits by
    failing.
    """
    if isinstance(page_limit, bool) or not isinstance(page_limit, int):
        raise ValueError("Page limit must be an integer.")
    if not 1 <= page_limit <= MAX_PAGE_LIMIT:
        raise ValueError(f"Page limit must be between 1 and {MAX_PAGE_LIMIT}.")
