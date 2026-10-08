"""A read that reached the ceiling, and what it takes to finish it.

The read model was the answer to a cost problem -- one walk instead of nine -- and it
inherited the seam's limit without anybody deciding to. A listing offset above
:data:`~tenbin.store.client.MAX_PAGE_OFFSET` is refused, so a walk stops there, so a
model built from a walk is a model of a prefix, and the completeness field says
``offset_cap``. That field is not the mitigation it looks like: a read model exists so
a consumer can read it *repeatedly without walking the store again*, so a caveat
noticed on the first run is read once and never again. **The failure these tests are
about is a number that is right once and then invisible**, and each one fails if a
resumed read produces a clean-looking model it cannot justify.

Three things have to hold at once for a resumed read to be honest, and they are
different in kind. **It gets past the ceiling**, which the seam permits in exactly one
way and not by continuing the offset. **A document it saw twice is counted once**,
which is a property of keying the merge on the document id rather than on the offset
-- and the offset is not a cursor, because the listing order is not documented as
stable and a document written mid-walk shifts everything behind it. **And it says so
when the corpus moved under it**, which is the case offsets cannot detect and ids can:
a document listed in one window and absent from the next is a finding and never a
deduplication success, because something either reindexed it or rewrote it and the
corpus does not say which.

**What a resumable walk cannot do, asserted rather than described.** The seam refuses
a listing offset above 10,000, and the last page read *at* that offset returns at most
one page of documents, so the addressable corpus is ``10,000 + page_limit`` documents
and no sequence of legal requests can ever see one more. The tests below use a
100-document page, which puts that ceiling at 10,100 and the smallest corpus that
cannot be read whole at 10,101, and
:func:`tenbin.readmodel.resume.reachable_ceiling` states the bound where a reader can
check it rather than leaving it in a docstring. **So the second window does not make a
large corpus readable.** What it does is read the whole addressable range, re-list the
seam to find out whether the listing moved anything into it, and report the exact size
of the hole rather than leaving a reader to infer one from a prefix.

**The store lives in this file because these cases cannot be scripted with the shared
one.** :class:`tests.fakes.FakeStore` serves one listing of a fixed length, and a
resumable walk needs a listing longer than the seam's ceiling, a listing that *changes*
between two windows, and a store that goes away mid-walk. All three are one object
here, over the real :class:`~tenbin.store.client.StoreClient`, so the envelope
parsing, the batch alignment and the offset validation under test are the production
ones -- a stand-in would test the stand-in. The ceiling tests are the slow ones, each
walking ten thousand documents, so the resumed read over that corpus is walked once by
:func:`_resumed_past_the_ceiling` and shared.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl

import httpx
import pytest

from tenbin.corpus.snapshot import Completeness, take_snapshot
from tenbin.measures import CompletenessMeasure, RepositoryCoverageMeasure
from tenbin.readmodel import build, load, to_snapshot, write
from tenbin.readmodel.model import ReadModel, WalkWindow
from tenbin.readmodel.resume import (
    MAX_RESUME_WINDOWS,
    MergeFinding,
    MergeFindingKind,
    ResumeOutcome,
    build_across_windows,
    reachable_ceiling,
)
from tenbin.store.client import (
    MAX_PAGE_LIMIT,
    MAX_PAGE_OFFSET,
    StoreClient,
    StoreResponseError,
    StoreUnavailableError,
)
from tests.fakes import FakeClock, fake_document, fake_settings, kojutsu_frontmatter

#: The page size every cap-crossing test here walks with, chosen so the smallest
#: corpus that cannot be read whole is ten thousand and one rather than ten thousand
#: five hundred -- and so the arithmetic in
#: :func:`tenbin.readmodel.resume.reachable_ceiling` is the number these fixtures are
#: built around.
PAGE: int = 100

#: The most documents any sequence of legal requests can address, and therefore the
#: most a whole corpus can be. Read from the function that states the bound rather than
#: written out again, so a test cannot quietly disagree with the module about it.
REACHABLE: int = reachable_ceiling(PAGE)

#: One document past that: the smallest corpus no walk can finish, so the smallest
#: corpus that forces a second window and the whole of what this ticket is about.
PAST_THE_CEILING: int = REACHABLE + 1

#: A hundred documents past it, which leaves a hole big enough to be worth reading.
BEYOND_ANY_WINDOW: int = REACHABLE + 100

#: The measures the "a resumed read is the same read" property is asserted with. Two of
#: different kinds, for the reason ``tests/test_read_model.py`` gives: one is a
#: distribution over the records, so it fails if a document was lost *or counted
#: twice*, and one is a rate over the store's own count, so it fails if the model's
#: header drifted from what it holds. One measure could pass with a model that kept its
#: records and lost its counts.
FIGURES = (RepositoryCoverageMeasure(), CompletenessMeasure())


# -- the store these tests script ------------------------------------------------------


class Listing:
    """One pass of the listing: its order, what the store claims to hold, and its lies.

    Every knob rather than a behaviour toggle, so a test's own source says which
    misbehaviour it is scripting. ``total`` overriding what the listing actually
    contains is how a store that lies about its size is expressed; ``omit`` is a
    document the store will not return from a batch read, which is the hole case;
    ``fail_at`` is a listing offset at which the transport dies, which is a walk that
    stopped partway rather than a store that was never there; ``omit_total`` and
    ``omit_has_more`` are the two envelope fields Tanseki's shapes have varied on, and a
    walk has to refuse the first and decide on the second.
    """

    def __init__(
        self,
        ids: tuple[str, ...],
        *,
        total: int | None = None,
        omit: frozenset[str] = frozenset(),
        fail_at: frozenset[int] = frozenset(),
        has_more: bool | None = None,
        omit_total: bool = False,
        omit_has_more: bool = False,
    ) -> None:
        self.ids = ids
        self.total = total
        self.omit = omit
        self.fail_at = fail_at
        self.has_more = has_more
        self.omit_total = omit_total
        self.omit_has_more = omit_has_more

    def reported_total(self) -> int:
        """The number this pass puts in the envelope, which may be a lie."""
        return len(self.ids) if self.total is None else self.total


class ScriptedStore(StoreClient):
    """A store whose listing this test dictates, and which can change under a walk.

    Subclasses the real client for the reason :class:`tests.fakes.FakeStore` gives:
    the offset validation, the envelope parsing and the batch alignment are the
    production ones, and the only thing scripted here is what the store *holds*.

    **A pass advances when a listing request arrives at an offset this pass has already
    served**, which is the only thing a store can observe that distinguishes a window
    that re-lists a seam from a walk that is still advancing. The last pass is held
    rather than wrapped: re-walking a range must serve the same documents, or every
    re-read would look like a reindex and no test here could tell a moved document from
    a stable one.

    **A repeated request at the same offset is the client retrying, and it does not
    advance the pass.** The client retries a transport failure two times before
    raising, and it resends the identical request; if that counted as a walk starting
    over then a scripted failure would be answered on the second attempt by a store
    that had changed its mind, and every test below that scripts a failure would be
    testing the retry budget instead. The discriminator is the request immediately
    before: a retry follows its own listing request, and a re-listing follows a batch
    read.
    """

    def __init__(self, listings: Sequence[Listing]) -> None:
        self._listings = list(listings)
        self._pass = 0
        #: The highest offset this pass has been asked for, and the evidence the next
        #: request is weighed against.
        self._furthest = -1
        #: What the store was asked immediately before, as a kind and an offset. The
        #: listing kind and its offset are what tell a re-listing from a retry.
        self._previous: tuple[str, int] | None = None
        self.documents: dict[str, dict[str, Any]] = {}
        self.clock = FakeClock()
        self.requests: list[httpx.Request] = []
        settings = fake_settings()
        transport = httpx.MockTransport(self._handle)
        super().__init__(
            settings,
            client=httpx.Client(transport=transport, base_url=settings.tanseki_url),
            max_retries=2,
            sleep=self.clock,
        )

    @property
    def passes_served(self) -> int:
        """How many times the store changed what it was serving."""
        return self._pass

    def queried_ids(self) -> list[str]:
        """Every document id the store was asked to return, in order.

        Read out of the request log rather than counted by the test, so "was this
        document read twice?" is answered by what went over the wire.
        """
        return [
            doc_id
            for request in self.requests
            if request.url.path.endswith("/documents:query")
            for doc_id in json.loads(request.content.decode())["ids"]
        ]

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path.endswith("/documents"):
            # ``_list`` records this request as the previous one itself, after it has
            # weighed the request before it against this one.
            return self._list(request)
        if path.endswith("/documents:query"):
            self._previous = ("query", 0)
            return self._query(request)
        return httpx.Response(404, json={"error": "no such route"})

    def _list(self, request: httpx.Request) -> httpx.Response:
        params = dict(parse_qsl(request.url.query.decode()))
        offset = int(params.get("offset", 0))
        limit = int(params.get("limit", 1))
        if offset <= self._furthest and self._previous != ("list", offset):
            self._pass = min(self._pass + 1, len(self._listings) - 1)
            self._furthest = -1
        self._previous = ("list", offset)
        self._furthest = max(self._furthest, offset)
        listing = self._listings[self._pass]
        if offset in listing.fail_at:
            raise httpx.ConnectError("scripted transport failure", request=request)
        window = listing.ids[offset : offset + limit]
        has_more = (
            listing.has_more
            if listing.has_more is not None
            else offset + len(window) < len(listing.ids)
        )
        body: dict[str, Any] = {"documents": [{"id": doc_id} for doc_id in window]}
        if not listing.omit_has_more:
            body["hasMore"] = has_more
        if not listing.omit_total:
            body["total"] = listing.reported_total()
        return httpx.Response(200, json=body)

    def _query(self, request: httpx.Request) -> httpx.Response:
        listing = self._listings[self._pass]
        payload = json.loads(request.content.decode())
        ids = payload.get("ids") or []
        documents = [
            self.documents[doc_id]
            for doc_id in ids
            if doc_id in self.documents and doc_id not in listing.omit
        ]
        return httpx.Response(200, json={"documents": documents})


def _ids(count: int, *, start: int = 0) -> tuple[str, ...]:
    """``count`` document ids of Kojutsu's shape, numbered from ``start``.

    Ids rather than integers because the merge is keyed on them, and a test using bare
    integers would not exercise the thing the merge is built on. Each is a separate pull
    request, so a corpus of ten thousand is ten thousand *changes* rather than one
    change ten thousand times over.
    """
    return tuple(
        f"acme/widget/pr-{index}/answer-{index:04d}" for index in range(start, start + count)
    )


def _store_with(*listings: Listing) -> ScriptedStore:
    """A store serving ``listings`` and able to return every id any of them names.

    The bodies are the union across passes, because a reindex can move an id and a
    document the store listed in one pass has to still be there in the next.
    """
    store = ScriptedStore(listings)
    for doc_id in sorted({doc_id for listing in listings for doc_id in listing.ids}):
        owner, name, *_ = doc_id.split("/")
        store.documents[doc_id] = fake_document(doc_id, kojutsu_frontmatter(repo=f"{owner}/{name}"))
    return store


@contextmanager
def _serving(*listings: Listing) -> Iterator[ScriptedStore]:
    """A scripted store, closed however the block ends.

    A local context manager rather than ``tests.fakes.closing`` because that one is
    typed to hand back a :class:`~tenbin.store.client.StoreClient`, and everything
    these tests read off the store afterwards is this file's own.
    """
    store = _store_with(*listings)
    try:
        yield store
    finally:
        store.close()


def _of_kind(outcome: ResumeOutcome, kind: MergeFindingKind) -> list[MergeFinding]:
    """The findings of one kind, which is how every assertion about them is phrased."""
    return [finding for finding in outcome.findings if finding.kind is kind]


@pytest.fixture(scope="module")
def _past_the_ceiling() -> Iterator[ScriptedStore]:
    """One store holding a corpus one document past the ceiling, for the cap tests.

    Module-scoped because four tests walk it and each walk reads ten thousand
    documents; the resumed read over it is built once by
    :func:`_resumed_past_the_ceiling`, which is the same trade every other expensive
    fixture in this suite makes.
    """
    with _serving(Listing(_ids(PAST_THE_CEILING))) as store:
        yield store


@pytest.fixture(scope="module")
def _resumed_past_the_ceiling(_past_the_ceiling: ScriptedStore) -> ResumeOutcome:
    """A resumed read of a corpus one document past the ceiling, walked once."""
    return build_across_windows(_past_the_ceiling, page_limit=PAGE)


# -- the properties under test ----------------------------------------------------------


def test_a_resumed_walk_reads_the_whole_addressable_collection_and_says_how_much_is_beyond_it_because_a_read_that_stopped_at_the_ceiling_cannot(
    _resumed_past_the_ceiling: ResumeOutcome,
) -> None:
    """Every document a legal offset can address, and the exact size of the hole.

    The first window walks offsets 0 through 10,000 and is cut off by the ceiling
    having listed 10,100 documents of a 10,101-document collection. **The second window
    starts at offset 10,000 rather than at 10,100**, which is the only way past the
    ceiling the seam allows: it re-lists the last page, recognises its hundred documents
    by id, and finds nothing new among them.

    So the resume does not make a large corpus readable, and this test says so: what it
    does is convert "we stopped and here is a prefix" into "we re-listed the seam and it
    added nothing, so the hole is exactly one document". The second window is a
    confirmation, and a confirmation is the difference between a number and an
    assumption.
    """
    model = _resumed_past_the_ceiling.model
    first, second = model.windows

    assert (first.offset_start, first.offset_end) == (0, REACHABLE)
    assert first.completeness is Completeness.OFFSET_CAP
    assert first.documents == REACHABLE
    assert (second.offset_start, second.offset_end) == (MAX_PAGE_OFFSET, REACHABLE)
    assert second.enumerated == PAGE, "the seam page is listed again, in full"
    assert second.documents == 0, "and every document in it was already held"

    assert model.enumerated == REACHABLE == len(model.records)
    assert model.store_total == PAST_THE_CEILING
    assert model.completeness is Completeness.OFFSET_CAP, (
        "a collection larger than the seam can address was not read whole, however many "
        "windows walked it"
    )
    assert model.truncation is not None
    assert model.truncation.records_missing == 1
    assert _of_kind(_resumed_past_the_ceiling, MergeFindingKind.SEAM_EXHAUSTED)


def test_a_figure_computed_from_a_resumed_read_renders_the_numbers_of_one_computed_from_the_single_whole_walk_because_the_seams_are_a_property_of_how_the_read_happened(
    _resumed_past_the_ceiling: ResumeOutcome, _past_the_ceiling: ScriptedStore
) -> None:
    """Same corpus, walked by two different walks, and every rendered number matches.

    The seams are the only difference between the two reads, so this is the test that a
    window is bookkeeping rather than a difference in what was read. **What is compared
    is what a reader sees -- ``value_text`` and ``rate_text`` -- and not the figure
    objects**, because a figure carries the snapshot it was computed over and the two
    walks happened at different moments: the two objects cannot be equal without the
    reads having happened together, and asserting that they were would be asserting
    something about the clock rather than about the seams. The completeness sentence is
    included in the comparison, so a resumed read that quietly rendered a different
    caveat would fail here.
    """
    resumed = to_snapshot(_resumed_past_the_ceiling.model)
    whole = to_snapshot(build(take_snapshot(_past_the_ceiling, page_limit=PAGE)))

    assert len(_resumed_past_the_ceiling.model.windows) == 2, (
        "this test is about a model read in more than one window"
    )
    for measure in FIGURES:
        assert measure.compute(resumed).value_text() == measure.compute(whole).value_text(), (
            measure.slug
        )
        assert measure.compute(resumed).rate_text() == measure.compute(whole).rate_text(), (
            measure.slug
        )
    assert resumed.records == whole.records
    assert resumed.truncation == whole.truncation
    assert resumed.completeness is whole.completeness is Completeness.OFFSET_CAP


def test_a_document_the_two_windows_both_listed_is_counted_once_because_a_denominator_that_double_counted_a_document_would_be_a_rate_over_nothing(
    _resumed_past_the_ceiling: ResumeOutcome,
) -> None:
    """A hundred documents listed twice, and a read that still holds 10,100 of them.

    Two numbers, and the second is the one that matters. The listing count is 10,200:
    the hundred documents at the seam were listed twice, on purpose, because that is
    what makes a moved document detectable. The model's count is 10,100, because a
    document recognised by id is counted once -- **and ``model.enumerated`` is what a
    denominator is computed over, so the difference between the two numbers is the
    difference between a rate over the collection and a rate over the collection plus
    its own overlap.**

    The third assertion is the structural one: the windows' ``documents`` add up to
    exactly the records the model holds, so the deduplication is checkable from the
    model's own record rather than promised by it.
    """
    model = _resumed_past_the_ceiling.model
    listed = sum(window.enumerated for window in model.windows)
    contributed = sum(window.documents for window in model.windows)

    assert listed == REACHABLE + PAGE
    assert contributed == model.enumerated == len(model.records) == REACHABLE
    assert len({record.doc_id for record in model.records}) == len(model.records)

    figure = RepositoryCoverageMeasure().compute(to_snapshot(model))
    assert figure.counted + figure.excluded_total == len(model.records)
    assert sum(figure.values.values()) == REACHABLE
    assert (
        CompletenessMeasure().compute(to_snapshot(model)).value_text().startswith(f"{REACHABLE:,}")
    ), "the rate is over the distinct documents, so it is not above one"


def test_a_document_re_seen_at_the_seam_is_not_fetched_a_second_time_because_a_second_read_is_an_update_with_a_better_justification() -> (
    None
):
    """One batch read per document across the whole build, read out of the request log.

    The seam re-lists a hundred ids the read already holds and the resume does not
    fetch them: the ids are compared, not re-read. That is the reason the merge is keyed
    on ids rather than offsets -- a re-fetched document could be *replaced* by a later
    window, and a model whose contents depend on which window happened to run last is a
    model being maintained by a walk.
    """
    with _serving(Listing(_ids(PAST_THE_CEILING))) as store:
        outcome = build_across_windows(store, page_limit=PAGE)
        queried = store.queried_ids()

    assert len(queried) == len(set(queried)), "a document was fetched twice, so the seam re-read it"
    assert len(queried) == outcome.model.enumerated
    assert set(queried) == set(outcome.model.by_id)


def test_a_model_records_every_window_it_walked_and_its_range_because_a_reader_has_to_see_the_seams_rather_than_infer_a_single_walk_that_never_happened(
    _resumed_past_the_ceiling: ResumeOutcome,
) -> None:
    """The windows survive the file, with their ranges, and the file says how many there are.

    Two windows that overlap -- ``[0, 10100)`` and ``[10000, 10100)`` -- because the
    overlap is what the second walk is *for*, and a reader who cannot see it has been
    shown two counts and no reason for them. The round trip through the file is the
    point: a record that is written and not read back would let a model report one walk
    while the file holds two, which is the drift the schema version exists to prevent.
    """
    model = _resumed_past_the_ceiling.model
    path = Path(tempfile.mkdtemp()) / "read-model.json"
    write(path, model)

    restored = load(path)
    assert restored is not None
    assert restored.windows == model.windows
    assert [(window.offset_start, window.offset_end) for window in restored.windows] == [
        (0, REACHABLE),
        (MAX_PAGE_OFFSET, REACHABLE),
    ]

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert [entry["offset_start"] for entry in payload["windows"]] == [0, MAX_PAGE_OFFSET]
    assert payload["windows"][1] == {
        "offset_start": MAX_PAGE_OFFSET,
        "enumerated": PAGE,
        "documents": 0,
        "store_total": PAST_THE_CEILING,
        "completeness": Completeness.OFFSET_CAP.value,
    }


def test_a_model_whose_windows_leave_a_gap_is_refused_because_a_read_with_a_gap_in_its_own_offsets_is_claiming_documents_nobody_asked_the_store_for(
    _resumed_past_the_ceiling: ResumeOutcome,
) -> None:
    """Four ways a window list can be a contradiction rather than a read.

    No windows at all, a first window that starts partway down the listing, a gap
    between two windows, and a window claiming to have added more documents than it was
    shown. None of them can come out of a walk, so a model holding one was built by
    something else -- and every figure over it would be a rate over a coverage the read
    never established.
    """
    model = _resumed_past_the_ceiling.model
    first, second = model.windows

    with pytest.raises(ValueError, match="must not be empty"):
        replace(model, windows=())
    with pytest.raises(ValueError, match="starts at offset"):
        replace(model, windows=(replace(second, offset_start=50, enumerated=50),))
    with pytest.raises(ValueError, match="unwalked"):
        replace(model, windows=(first, replace(second, offset_start=first.offset_end + 1)))
    with pytest.raises(ValueError, match="cannot add a document it was not shown"):
        WalkWindow(
            offset_start=0,
            enumerated=10,
            documents=11,
            store_total=10,
            completeness=Completeness.COMPLETE,
        )


def test_a_document_listed_in_one_window_and_absent_from_the_next_is_a_finding_because_a_document_that_moved_is_either_a_reindex_or_a_rewrite() -> (
    None
):
    """A reindex between two windows of one build, and what it costs the reader to know.

    The store drops its first hundred documents between the two windows, which is the
    shape a reindex or a rewrite takes, and the consequences are three different facts.
    **A document came into reach that no legal offset could address before**: the
    hundred-and-first document of the first listing now sits at the last legal offset, so
    the second window reads something the first could not have seen. **A hundred
    documents the first window read are no longer listed at all**, which is a finding
    and never a deduplication success. And **the store's own count changed**, from
    10,101 to 10,001, which is the one independent number available and does not agree
    with itself.

    The model holds 10,101 documents against a store claiming 10,001, so the counts
    disagree in the other direction from a whole walk's truncated read and the verdict
    is the corpus-has-no-agreed-size one. Nothing here is a deduplication success: the
    merge worked, and the corpus changed.
    """
    before = _ids(PAST_THE_CEILING)
    unreachable_to_the_first_window = before[PAST_THE_CEILING - 1]
    with _serving(Listing(before), Listing(before[PAGE:])) as store:
        resumed = build_across_windows(store, page_limit=PAGE)

    assert store.passes_served == 1, (
        "the second window had to be served a different listing, or this test is not "
        "about a reindex"
    )
    assert unreachable_to_the_first_window in resumed.model.by_id, (
        "the document the reindex pulled into the addressable range must be read, or the "
        "second window went back to the seam for no reason at all"
    )
    assert resumed.model.enumerated == PAST_THE_CEILING
    assert resumed.model.store_total == PAST_THE_CEILING - PAGE
    assert resumed.model.completeness is Completeness.STORE_TOTAL_EXCEEDED
    assert "no agreed size" in CompletenessMeasure().compute(to_snapshot(resumed.model)).rate_text()

    moved = _of_kind(resumed, MergeFindingKind.DOCUMENT_MOVED)
    assert len(moved) == 1
    assert set(moved[0].doc_ids) == set(before[MAX_PAGE_OFFSET : MAX_PAGE_OFFSET + PAGE])
    assert len(_of_kind(resumed, MergeFindingKind.STORE_TOTAL_CHANGED)) == 1
    rendered = resumed.render_findings()
    assert "document_moved:" in rendered
    assert "100 documents affected" in rendered
    assert "store_total_changed:" in rendered


def test_a_corpus_that_grew_between_two_windows_is_not_a_corpus_that_was_read_whole_because_the_store_reported_a_count_it_itself_changed() -> (
    None
):
    """A whole read of 300 documents, then a store holding 500, and every count agrees.

    The window after the first one is walked anyway -- continuing a read is how the new
    documents are found at all -- and this one ends at the end of the listing holding
    500 of 500. **The collection changed while it was being read, so nothing
    established that what was read is what the store holds**, and the verdict says so
    while the counts say nothing.

    This is the case a per-window ``store_total`` exists for: a model recording only the
    final count would see 500 of 500 and call it whole.
    """
    before = _ids(300)
    after = (*before, *_ids(200, start=300))
    with _serving(Listing(before), Listing(after)) as store:
        first = build_across_windows(store, page_limit=PAGE)
        assert first.model.completeness is Completeness.COMPLETE
        assert first.findings == (), "a single window has no seam to have moved"

        resumed = build_across_windows(store, page_limit=PAGE, existing=first.model)

    changed = _of_kind(resumed, MergeFindingKind.STORE_TOTAL_CHANGED)
    assert len(changed) == 1
    assert "300" in changed[0].detail and "500" in changed[0].detail
    assert resumed.resumed_from == 200
    assert resumed.model.enumerated == resumed.model.store_total == 500
    assert resumed.model.completeness is Completeness.STORE_TOTAL_EXCEEDED
    assert "store_total_changed" in resumed.render_findings()


def test_a_store_that_shrank_keeps_the_documents_it_read_and_says_the_counts_disagree_because_a_document_the_store_no_longer_lists_is_not_deduplication() -> (
    None
):
    """A store that lost fifty documents between builds, and a read that holds them still.

    The tempting response is to drop them, so that the read agrees with the store. That
    is an update, it is the one operation this package refuses, and it would answer
    "what changed since the last read" by rewriting the evidence of the change. So the
    read keeps what it read and the verdict says the two counts disagree, which is the
    sentence :class:`~tenbin.corpus.snapshot.Completeness` already has for a corpus
    with no agreed size.

    There is no ``document_moved`` finding here and that is not an oversight: the store
    lost documents from the *middle* of the listing, which this walk does not re-read,
    so only the count shows it. A walk that claimed otherwise would be claiming a check
    it does not perform.
    """
    before = _ids(500)
    after = before[:450]
    with _serving(Listing(before), Listing(after)) as store:
        first = build_across_windows(store, page_limit=PAGE)
        resumed = build_across_windows(store, page_limit=PAGE, existing=first.model)

    assert resumed.model.enumerated == 500
    assert resumed.model.store_total == 450
    assert resumed.model.completeness is Completeness.STORE_TOTAL_EXCEEDED
    assert all(doc_id in resumed.model.by_id for doc_id in after[400:])
    assert _of_kind(resumed, MergeFindingKind.DOCUMENT_MOVED) == []


def test_a_walk_that_failed_is_finished_by_a_later_window_because_a_resume_is_how_a_read_that_was_cut_short_gets_ended() -> (
    None
):
    """A walk that died at offset 100 of 300, resumed into a read of all 300.

    This is the case an operator meets and the reason the flag is opt-in rather than
    automatic: a build that fails leaves the model it had, and running it again with the
    model passed in continues the walk rather than starting it. The resumed read reaches
    the end of the listing and the model is whole.

    The first window's ``store_error`` verdict is **kept**, because it is what happened
    and a reader of the model is entitled to see that one of its windows died. It does
    not make the read partial, because the later window covered the range the failed one
    did not: a window keeps the verdict it reached when it was walked, and the model's
    verdict is about the read as a whole.
    """
    ids = _ids(300)
    with _serving(Listing(ids, fail_at=frozenset({PAGE}))) as store:
        failed = build_across_windows(store, page_limit=PAGE)

    assert failed.model.completeness is Completeness.STORE_ERROR
    assert failed.model.enumerated == PAGE
    assert failed.model.failure is not None
    assert failed.model.windows[0].completeness is Completeness.STORE_ERROR

    with _serving(Listing(ids)) as store:
        resumed = build_across_windows(store, page_limit=PAGE, existing=failed.model)

    assert resumed.model.completeness is Completeness.COMPLETE
    assert resumed.model.enumerated == resumed.model.store_total == 300
    assert resumed.model.failure is None, (
        "the failure that stopped the earlier walk is not this one's"
    )
    assert [window.completeness for window in resumed.model.windows] == [
        Completeness.STORE_ERROR,
        Completeness.COMPLETE,
    ]
    assert resumed.model.windows[1].offset_start == 0, (
        "the resume re-listed from before the page that failed, so the page that killed the "
        "walk was tried again rather than skipped"
    )
    assert resumed.resumed is True
    assert resumed.resumed_from == 0, (
        "resuming from offset zero is a resume, which is why the property tests for None "
        "rather than for truthiness"
    )
    assert resumed.findings == ()


def test_a_window_that_stops_partway_keeps_the_documents_the_windows_before_it_read_because_a_partial_read_is_still_data() -> (
    None
):
    """The third window's store goes away, and the model keeps 10,101 documents.

    Only the *build's* first request raises, and this is the case that shows why: a walk
    that has read ten thousand documents and then meets a store that has gone away has a
    partial read, which is data. Raising would throw it away and leave the caller with
    whatever model was already on disk -- and the reindexed listing this store is
    serving is how the second window came to hold a document the first could not reach,
    which is the whole reason to keep walking.

    The window that failed has no store count of its own, because it never read one, so
    the model's count is the last one the walk actually heard rather than a number
    invented for the window that stopped.
    """
    ids = _ids(PAST_THE_CEILING)
    rotated = (*ids[1:], ids[0])
    with _serving(
        Listing(ids),
        Listing(rotated),
        Listing(rotated, fail_at=frozenset({MAX_PAGE_OFFSET})),
    ) as store:
        outcome = build_across_windows(store, page_limit=PAGE)

    assert outcome.model.completeness is Completeness.STORE_ERROR
    assert outcome.model.failure is not None
    assert isinstance(outcome.model.failure.restore(), StoreUnavailableError)
    assert len(outcome.model.records) == outcome.model.enumerated == PAST_THE_CEILING
    assert ids[PAST_THE_CEILING - 1] in outcome.model.by_id, (
        "the second window reached a document the first could not, so the third window's "
        "failure cost the read a walk and not a corpus"
    )
    assert outcome.windows[1].documents == 1
    assert outcome.windows[2].store_total is None, "the window that failed never read a count"
    assert outcome.model.store_total == PAST_THE_CEILING, (
        "the count is the last one the walk heard, not a number for the window that stopped"
    )


def test_a_store_that_says_there_is_more_and_lists_nothing_is_a_failure_because_an_empty_page_under_a_has_more_flag_is_a_store_contradicting_itself() -> (
    None
):
    """An empty listing with ``hasMore`` set, and the walk refuses to call it complete.

    Reporting a whole corpus from a response that says there is more is the failure this
    branch exists to prevent, so it is pinned here rather than left to the whole walk's
    copy of the rule.
    """
    with _serving(Listing((), has_more=True)) as store:
        outcome = build_across_windows(store, page_limit=PAGE)

    assert outcome.model.completeness is Completeness.STORE_ERROR
    assert outcome.model.failure is not None
    assert "no documents while reporting more" in outcome.model.failure.message
    assert outcome.model.enumerated == 0
    assert "did not finish" in CompletenessMeasure().compute(to_snapshot(outcome.model)).rate_text()


def test_a_store_that_says_nothing_about_has_more_is_trusted_up_to_its_own_count_because_the_only_other_number_on_offer_is_the_store_s() -> (
    None
):
    """A listing with no ``hasMore`` at all, walked by its count and no further.

    The walk cannot ask page after page of nothing to find out, so it stops when the
    count it was already given has been enumerated. Here that is the whole collection,
    so the read is complete -- and a store that had more would leave the shortfall
    visible as a count that was not reached.
    """
    with _serving(Listing(_ids(40), omit_has_more=True)) as store:
        outcome = build_across_windows(store, page_limit=PAGE)

    assert outcome.model.completeness is Completeness.COMPLETE
    assert outcome.model.enumerated == outcome.model.store_total == 40
    assert len(outcome.model.windows) == 1


def test_a_listing_with_no_total_is_refused_before_any_claim_about_completeness_because_completeness_here_is_a_claim_about_a_count() -> (
    None
):
    """A listing envelope with no ``total``, which a walk cannot check and must not guess.

    ``list_documents`` tolerates the shape because Tanseki's envelopes have varied, and a
    walk cannot: "complete" is a claim about a count, and with no count there is nothing
    to compare the enumeration against. The refusal is raised rather than returned
    because it is the build's first request, so the caller gets a store error and writes
    nothing.
    """
    with _serving(Listing(_ids(40), omit_total=True)) as store:
        with pytest.raises(StoreResponseError, match="no total"):
            build_across_windows(store, page_limit=PAGE)
        assert store.requests == [store.requests[0]], "one request, and nothing was read"


def test_a_resume_that_fails_partway_leaves_the_previous_model_exactly_as_it_was_because_resuming_is_not_updating() -> (
    None
):
    """A store that goes away on the resume's first request, and an untouched file.

    Three assertions, and the third is the one that would survive a careless
    implementation. The build did not raise -- it had documents already, so the failure
    is data and is carried on the model -- and the previous model is on disk unchanged,
    because this walk writes nothing at all. And **if that failed model had been
    written, it would have replaced the previous one rather than merged with it**: the
    file would hold the read that failed and none of the read that succeeded, which is
    the whole distinction between resuming and updating. A half-merged model is worse
    than the one it was going to replace.
    """
    path = Path(tempfile.mkdtemp()) / "read-model.json"
    kept = _small_read(count=100)
    write(path, kept)
    before = path.read_text(encoding="utf-8")

    with _serving(
        Listing(_ids(300), fail_at=frozenset({PAGE})),
        Listing(_ids(300), fail_at=frozenset({0})),
    ) as store:
        cut_short = build_across_windows(store, page_limit=PAGE)
        assert cut_short.model.enumerated == PAGE
        with pytest.raises(StoreUnavailableError):
            # The resume's first request is the build's first request, so a store that
            # cannot be reached raises rather than producing a model of nothing.
            build_across_windows(store, page_limit=PAGE, existing=cut_short.model)
        assert store.clock.delays, "the client retried a transport failure before raising"

    assert path.read_text(encoding="utf-8") == before
    assert load(path) == kept

    with _serving(Listing(_ids(300), fail_at=frozenset({200}))) as store:
        failed_read = build_across_windows(store, page_limit=PAGE)
    assert failed_read.model.completeness is Completeness.STORE_ERROR
    assert failed_read.model.enumerated == 200

    write(path, failed_read.model)
    replaced = load(path)
    assert replaced is not None
    assert replaced == failed_read.model
    assert len(replaced.records) == 200 != len(kept.records), (
        "the failed read replaced the model rather than merging with it, which is the only "
        "behaviour that keeps a model a read instead of a running total"
    )


def test_a_collection_larger_than_the_seam_can_address_is_reported_with_its_shortfall_because_a_walk_that_runs_out_of_offsets_is_a_finding() -> (
    None
):
    """A corpus of 10,200, where the most any sequence of legal requests can reach is 10,100.

    The finding names the bound in the number a reader can check against the store,
    because "there is more" is not actionable and "there are a hundred documents past
    what this seam can address" is. The verdict is ``offset_cap`` rather than a count
    contradiction, and the difference is which of the two sentences a reader is given: a
    prefix with a known length, against a corpus that has no agreed size. The first is
    true here, and it is the more useful of the two.
    """
    with _serving(Listing(_ids(BEYOND_ANY_WINDOW))) as store:
        resumed = build_across_windows(store, page_limit=PAGE)

    assert resumed.model.enumerated == REACHABLE
    assert resumed.model.completeness is Completeness.OFFSET_CAP
    assert resumed.model.truncation is not None
    assert resumed.model.truncation.records_missing == 100
    assert resumed.model.truncation.offset_reached == REACHABLE
    assert resumed.model.truncation.boundary_repositories, (
        "the truncation describes where the cut fell, which is a page the walk did read"
    )
    exhausted = _of_kind(resumed, MergeFindingKind.SEAM_EXHAUSTED)
    assert len(exhausted) == 1
    assert f"{REACHABLE:,}" in exhausted[0].detail
    assert (
        "100 records were not reached"
        in CompletenessMeasure().compute(to_snapshot(resumed.model)).rate_text()
    )
    assert "seam_exhausted" in resumed.render_findings()


def test_a_store_being_written_to_faster_than_it_is_read_stops_at_the_window_bound_because_there_is_no_point_at_which_another_walk_would_finish() -> (
    None
):
    """Every window adds a document, so the walk never runs out of work -- and stops anyway.

    Sixteen windows is the bound, and reaching it means the store's listing is losing a
    document at the front on every pass, which pulls new documents into the addressable
    range faster than the seam can be exhausted. There is no offset at which this walk
    would have finished, so the honest ending is a bound and a finding rather than a
    number that stopped being right at some point nobody logged.
    """
    ids = _ids(BEYOND_ANY_WINDOW)
    with _serving(*(Listing(ids[shift:]) for shift in range(MAX_RESUME_WINDOWS + 1))) as store:
        resumed = build_across_windows(store, page_limit=PAGE)

    assert len(resumed.windows) == MAX_RESUME_WINDOWS
    assert all(window.documents > 0 for window in resumed.windows[1:])
    assert len(_of_kind(resumed, MergeFindingKind.WINDOW_LIMIT_REACHED)) == 1
    assert "window_limit_reached" in resumed.render_findings()


def test_a_document_the_store_will_not_return_is_a_hole_in_a_resumed_read_too_because_a_denominator_that_lost_its_own_numerator_is_the_worst_number_available() -> (
    None
):
    """A document listed and not returned ends the window as a store error, not a merge.

    The hole is in the last page of the first window, which is the page the second
    window would have re-listed, and that is the interesting place for it: **a hole in a
    resumed read is a hole in a model that already holds ten thousand good documents**,
    and the temptation to carry on and count what did arrive is exactly the temptation a
    whole walk resists. It is reported as a failure, the failure is carried as a
    failure, and no record is invented to fill the gap.
    """
    ids = _ids(PAST_THE_CEILING)
    with _serving(Listing(ids, omit=frozenset({ids[MAX_PAGE_OFFSET]}))) as store:
        resumed = build_across_windows(store, page_limit=PAGE)

    assert resumed.model.completeness is Completeness.STORE_ERROR
    assert resumed.model.failure is not None
    assert "not returned by the store" in resumed.model.failure.message
    assert ids[MAX_PAGE_OFFSET] not in resumed.model.by_id
    assert len(resumed.model.records) == resumed.model.enumerated == REACHABLE - 1
    assert resumed.model.windows[0].enumerated == REACHABLE, (
        "the window still records what the store listed, so the hole is visible as the "
        "difference between the two counts"
    )
    assert "did not finish" in CompletenessMeasure().compute(to_snapshot(resumed.model)).rate_text()


def test_a_clean_build_says_it_found_nothing_rather_than_rendering_nothing() -> None:
    """The absence of a finding is a sentence, because an empty list reads as no look.

    And the other direction: a finding renders as its kind, its sentence, and a bounded
    number of the documents it is about -- a reindex can move thousands, and a list of
    thousands is not something anybody reads.
    """
    with _serving(Listing(_ids(40))) as store:
        clean = build_across_windows(store, page_limit=PAGE)

    assert clean.findings == ()
    assert clean.render_findings() == (
        "No findings: the windows this read walked agree with the store's own count and with "
        "each other, and no document listed in one window was missing from the next."
    )

    moved = MergeFinding(
        kind=MergeFindingKind.DOCUMENT_MOVED,
        doc_ids=tuple(f"acme/widget/pr-{index}/answer-{index}" for index in range(12)),
        detail="twelve documents were listed and not listed again",
    ).render_text()
    assert (
        moved.splitlines()[0] == "document_moved: twelve documents were listed and not listed again"
    )
    assert "12 documents affected, including:" in moved
    assert "and 7 more" in moved

    one = MergeFinding(
        kind=MergeFindingKind.SEAM_EXHAUSTED,
        doc_ids=(),
        detail="the seam is exhausted",
    ).render_text()
    assert one == "seam_exhausted: the seam is exhausted", (
        "a finding about counts and seams names no documents, and renders as one line"
    )


def test_a_finding_with_nothing_in_it_is_refused_because_a_heading_over_an_empty_paragraph_is_a_sentence_nobody_wrote() -> (
    None
):
    """A blank detail, a bare-string kind, and an id that is not an id."""
    with pytest.raises(ValueError, match="detail"):
        MergeFinding(kind=MergeFindingKind.SEAM_EXHAUSTED, doc_ids=(), detail="   ")
    with pytest.raises(ValueError, match="MergeFindingKind"):
        MergeFinding(kind="seam_exhausted", doc_ids=(), detail="a bare string")
    with pytest.raises(ValueError, match="document ids"):
        MergeFinding(kind=MergeFindingKind.DOCUMENT_MOVED, doc_ids=("",), detail="a blank id")


def test_a_resume_of_a_read_from_another_collection_is_refused_because_two_collections_in_one_read_is_not_a_corpus() -> (
    None
):
    """A model from a different collection, and a refusal rather than a merge."""
    kept = replace(_small_read(count=5), collection="some-other-collection")
    with (
        _serving(Listing(_ids(5))) as store,
        pytest.raises(ValueError, match="a read model names one collection"),
    ):
        build_across_windows(store, page_limit=PAGE, existing=kept)


def test_a_page_limit_the_store_would_refuse_is_refused_before_the_first_request() -> None:
    """Four page limits the seam answers with a 400, and no request to find out."""
    with _serving(Listing(_ids(5))) as store:
        for bad in (0, MAX_PAGE_LIMIT + 1, -1, True):
            with pytest.raises(ValueError, match="Page limit"):
                build_across_windows(store, page_limit=bad)
        assert store.requests == [], "a refused page size must not reach the store"


# -- helpers for the tests above ------------------------------------------------------


def _small_read(*, count: int) -> ReadModel:
    """A whole read of ``count`` documents, walked rather than hand-made, for a file to hold.

    100 documents is enough to tell a replaced file from a merged one by its length,
    and building it by walking keeps the fixture honest: it is what a build produces,
    not a model shaped to make an assertion pass.
    """
    with _serving(Listing(_ids(count))) as store:
        return build_across_windows(store, page_limit=PAGE).model
