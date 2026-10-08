"""That a snapshot says how far it got, and never says it got further than it did.

Tanseki has no aggregation, so every measure in this program is a full enumeration
walking pages by hand, and the snapshot is the object that walk returns. Its
whole job is to be checkable: a caller holding one must be able to tell a whole
corpus from a prefix of one without reading a log, and must not be able to mistake
the two by ignoring a field.

The four ways a walk can end are four different findings, and this module pins
each of them separately: the store said there was no more; the offset ceiling was
reached; the store's own count contradicted its own listing; or something failed
partway through. The case that most needs the test is the one a reader is most
likely to assume away -- a document that was listed and then was not there -- since
a denominator that quietly loses its own numerator produces the most confident and
most wrong number available.

The one thing a snapshot must never do is come back empty because the store was
unreachable. That case raises, and the test says so.
"""

from __future__ import annotations

import json
from urllib.parse import parse_qsl

import httpx
import pytest

from tenbin.corpus.record import RecordKind
from tenbin.corpus.snapshot import (
    Completeness,
    CorpusSnapshot,
    Truncation,
    WalkProgress,
    take_snapshot,
)
from tenbin.store.client import (
    MAX_PAGE_LIMIT,
    StoreClient,
    StoreError,
    StoreResponseError,
    StoreUnavailableError,
)
from tests.fakes import (
    FakeOptions,
    FakeStore,
    closing,
    documents_for,
    fake_settings,
    json_response,
    store_over,
)


def test_a_walk_that_reaches_the_end_of_the_store_is_complete_and_carries_its_records() -> None:
    """The ordinary case, stated so that every other case is a departure from it.

    ``store_total`` and ``enumerated`` are both on the object because they are
    the two independent numbers a reader checks against each other: the store's
    count, and what actually arrived.
    """
    with closing(FakeStore(documents_for(5))) as store:
        snapshot = take_snapshot(store, page_limit=2)
    assert snapshot.completeness is Completeness.COMPLETE
    assert snapshot.is_complete is True
    assert (snapshot.store_total, snapshot.enumerated) == (5, 5)
    assert len(snapshot.records) == 5
    assert snapshot.truncation is None
    assert snapshot.error is None


def test_the_snapshot_names_the_collection_and_the_moment_it_was_read() -> None:
    """A snapshot is a statement about a moment, and the moment has to be on it.

    The collection is named for the same reason: a figure over the wrong
    collection is a real number about a corpus nobody asked for, and the only
    thing distinguishing the two is what this object says it is about.
    """
    with closing(FakeStore(documents_for(1))) as store:
        snapshot = take_snapshot(store)
    assert snapshot.collection == "chronicler-real"
    assert snapshot.read_at.tzinfo is not None
    assert snapshot.records[0].classification.kind is RecordKind.ANSWER


def test_the_walk_steps_the_offset_by_the_documents_it_actually_got() -> None:
    """A short page means the store served fewer documents than were asked for.

    Stepping by the requested page size would skip whatever the short page
    withheld, and the walk would finish with fewer records than the store holds
    and report itself as complete.
    """
    offsets: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(parse_qsl(request.url.query.decode()))
        offset = int(params.get("offset", 0))
        offsets.append(str(offset))
        if offset == 0:
            return json_response(
                {"documents": [{"id": "a"}, {"id": "b"}], "total": 3, "hasMore": True}
            )
        return json_response({"documents": [{"id": "c"}], "total": 3, "hasMore": False})

    def documents(request: httpx.Request) -> httpx.Response:
        ids = json.loads(request.content.decode())["ids"]
        return json_response(
            {"documents": [{"id": doc_id, "collection": "chronicler-real"} for doc_id in ids]}
        )

    def route(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/documents:query"):
            return documents(request)
        return handler(request)

    with closing(store_over(route)) as client:
        snapshot = take_snapshot(client, page_limit=5)
    assert offsets == ["0", "2"], "stepped by the number of ids returned, not the limit"
    assert snapshot.enumerated == 3
    assert snapshot.completeness is Completeness.COMPLETE


def test_an_unreachable_store_raises_rather_than_returning_an_empty_snapshot() -> None:
    """An empty snapshot from a store that was never reached is a corpus of nothing.

    The two are indistinguishable downstream, and the second is the most
    dangerous output this program can produce: a real-looking number, with no
    claim attached, meaning nothing at all. So the first page is read outside the
    guarded region and its failure propagates.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host", request=request)

    with (
        closing(store_over(handler, max_retries=0)) as client,
        pytest.raises(StoreUnavailableError),
    ):
        take_snapshot(client)


def test_a_mid_walk_failure_is_carried_on_the_snapshot_instead_of_raising() -> None:
    """Three quarters of a corpus is still data, and the exception explains the quarter that is not.

    Raising would discard the records already read; returning without the
    exception would hand a caller a short corpus indistinguishable from a whole
    one. The snapshot carries both the partial records and the reason.
    """
    with closing(
        FakeStore(documents_for(6), options=FakeOptions(fail_at_offsets=frozenset({4})))
    ) as store:
        snapshot = take_snapshot(store, page_limit=2)

    assert snapshot.completeness is Completeness.STORE_ERROR
    assert isinstance(snapshot.error, StoreError)
    assert snapshot.enumerated == 4
    assert len(snapshot.records) == 4


def test_a_snapshot_that_failed_still_says_how_many_documents_it_never_saw() -> None:
    """``enumerated`` counts what was listed, so a caller can see the size of the gap.

    After a mid-walk failure the two diverge, and that divergence is the visible
    proof that the records in hand are not the corpus: a caller ignoring
    ``completeness`` gets a short list and a number that does not match it.
    """
    with closing(
        FakeStore(documents_for(6), options=FakeOptions(fail_at_offsets=frozenset({4})))
    ) as store:
        snapshot = take_snapshot(store, page_limit=2)
    assert snapshot.store_total == 6
    assert snapshot.enumerated < snapshot.store_total


def test_a_document_that_vanishes_between_listing_and_fetch_downgrades_the_walk() -> None:
    """:query omits ids it does not hold, and a snapshot missing a record is not whole.

    This is the case a strict reader is most likely to assume away. The
    enumeration succeeded, every page arrived, and the resulting records are all
    internally consistent -- the denominator has simply lost part of itself, and
    the only thing that distinguishes it from a whole corpus is this flag.
    """
    documents = documents_for(4)
    first_id = next(iter(documents))
    with closing(
        FakeStore(documents, options=FakeOptions(omit_ids=frozenset({first_id})))
    ) as store:
        snapshot = take_snapshot(store, page_limit=2)

    assert snapshot.completeness is Completeness.STORE_ERROR
    assert isinstance(snapshot.error, StoreResponseError)
    assert "not returned by the store" in str(snapshot.error)
    assert len(snapshot.records) < snapshot.enumerated


def test_a_hole_stops_the_walk_rather_than_producing_a_later_page_of_shifted_records() -> None:
    """Carrying on after a hole would mean every subsequent record is attached to the wrong place.

    The batch read keeps alignment, so a hole cannot shift anything -- but a walk
    that continued past one would be reporting a corpus with a record silently
    missing from the middle, which is the whole failure this flag exists for.
    """
    documents = documents_for(6)
    first_id = next(iter(documents))
    with closing(
        FakeStore(documents, options=FakeOptions(omit_ids=frozenset({first_id})))
    ) as store:
        snapshot = take_snapshot(store, page_limit=2)
    assert snapshot.enumerated == 2, "the listed id is counted even though it did not arrive"
    assert len(snapshot.records) == 1


def test_a_store_that_says_there_is_more_and_then_serves_nothing_is_an_error() -> None:
    """A contradictory envelope is not a finished enumeration.

    Reporting ``complete`` here would mean a whole-corpus claim built on a
    response that says there is more, and the caller has no way to see the
    contradiction other than this flag.
    """
    with closing(FakeStore(documents_for(3), options=FakeOptions(has_more=True))) as store:
        snapshot = take_snapshot(store, page_limit=2)
    assert snapshot.completeness is Completeness.STORE_ERROR
    assert isinstance(snapshot.error, StoreResponseError)
    assert "while reporting more" in str(snapshot.error)


def test_a_store_whose_count_is_smaller_than_what_it_served_is_not_called_complete() -> None:
    """A count smaller than the enumeration it is supposed to count is not a count.

    The walk finished, the store said there was no more, and the only thing wrong
    is the number the whole completeness story depends on -- so it is reported as
    its own outcome rather than folded into ``complete``, which would have told a
    reader that four records were a whole corpus of two.
    """
    with closing(
        FakeStore(documents_for(4), options=FakeOptions(total=2, has_more=False))
    ) as store:
        snapshot = take_snapshot(store, page_limit=5)
    assert snapshot.completeness is Completeness.STORE_TOTAL_EXCEEDED
    assert snapshot.store_total == 2
    assert snapshot.enumerated == 4


def test_a_store_whose_count_is_larger_than_what_it_served_is_also_not_called_complete() -> None:
    """The other direction is the same finding: the count does not match the listing.

    The store claims ninety-nine documents and then says there is no more.
    Accepting the count would leave a report stating a denominator thirty times
    the records in hand, and every rate over it would be understated by a factor
    the reader cannot see.
    """
    with closing(
        FakeStore(documents_for(3), options=FakeOptions(total=99, has_more=False))
    ) as store:
        snapshot = take_snapshot(store, page_limit=3)
    assert snapshot.completeness is Completeness.STORE_TOTAL_EXCEEDED
    assert snapshot.store_total == 99
    assert snapshot.enumerated == 3


def test_a_total_that_agrees_with_the_enumeration_is_what_complete_means() -> None:
    """The negative case for the rule above, so it cannot be satisfied by accident."""
    with closing(FakeStore(documents_for(3))) as store:
        snapshot = take_snapshot(store, page_limit=2)
    assert snapshot.store_total == snapshot.enumerated == 3
    assert snapshot.completeness is Completeness.COMPLETE


def test_the_offset_ceiling_is_named_as_its_own_outcome_and_reports_the_hole() -> None:
    """A corpus past ten thousand documents cannot be walked by anybody, and that is a fact.

    The shortfall is a count of what was missed and a histogram of where the cut
    fell, so a report can say "the first 10 000 of about 14 000" rather than
    presenting a prefix as though it were the corpus.
    """
    documents = documents_for(4)
    with closing(FakeStore(documents, options=FakeOptions(total=9, has_more=True))) as store:
        # Force the cap rather than reading ten thousand documents: the rule under
        # test is which outcome is chosen at the boundary, not the paging itself.
        snapshot = _walk_to_the_cap(store)
    assert snapshot.completeness is Completeness.OFFSET_CAP
    assert snapshot.truncation is not None
    assert snapshot.truncation.records_missing == 5
    assert snapshot.truncation.offset_reached == 4


def _walk_to_the_cap(store: StoreClient) -> CorpusSnapshot:
    """Run a walk against a store positioned at the offset ceiling.

    Expressed as a helper rather than a monkeypatched constant so the test that
    cares about the *choice of outcome* does not have to also own the arithmetic
    of reaching ten thousand documents.
    """
    from tenbin.corpus import snapshot as snapshot_module

    original = snapshot_module.MAX_PAGE_OFFSET
    try:
        object.__setattr__(snapshot_module, "MAX_PAGE_OFFSET", 2)
        return take_snapshot(store, page_limit=2)
    finally:
        object.__setattr__(snapshot_module, "MAX_PAGE_OFFSET", original)


def test_the_truncation_describes_the_boundary_page_and_not_the_records_it_never_read() -> None:
    """The tail cannot be described without reading it, and pretending otherwise would mislead.

    What these histograms are: the composition of the last page the walk read,
    which is the only evidence available about where the cut fell. A cut inside
    one busy repository and a cut between two are different problems with
    different fixes, so the boundary is worth reporting -- as long as it is not
    dressed up as a census of what is missing.
    """
    documents = documents_for(4)
    with closing(FakeStore(documents, options=FakeOptions(total=9, has_more=True))) as store:
        snapshot = _walk_to_the_cap(store)
    truncation = snapshot.truncation
    assert truncation is not None
    assert truncation.boundary_repositories == {"acme/widget": 2}
    assert truncation.boundary_months == {"2026-09": 2}


def test_a_complete_walk_carries_no_truncation_because_there_is_no_tail() -> None:
    """A truncation on a whole corpus would be a caveat about nothing, which is noise."""
    with closing(FakeStore(documents_for(3))) as store:
        assert take_snapshot(store, page_limit=2).truncation is None


def test_a_listing_without_a_total_cannot_be_walked_because_completeness_is_about_a_count() -> None:
    """The listing is lenient about a missing total; a walk cannot be.

    ``completeness`` is a claim that the enumeration matched what the store holds,
    and that claim needs the number to be about. Guessing -- treating the walked
    count as its own total -- would make "complete" a word this program had no
    basis for, which is the failure the whole field exists to prevent.
    """
    with (
        closing(FakeStore(documents_for(3), options=FakeOptions(omit_total=True))) as store,
        pytest.raises(StoreResponseError, match="no total"),
    ):
        take_snapshot(store, page_limit=2)


def test_a_store_that_cannot_even_be_counted_raises_on_the_first_page() -> None:
    """A count is a number, and a response without one cannot be walked against."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/documents"):
            return json_response({"documents": [{"id": "a"}], "hasMore": False})
        return json_response({"documents": [{"id": "a", "collection": "chronicler-real"}]})

    with closing(store_over(handler)) as client, pytest.raises(StoreResponseError):
        take_snapshot(client)


def test_progress_is_reported_after_every_page_so_a_long_walk_is_not_silent() -> None:
    """A corpus of ten thousand documents takes long enough that silence is its own bug.

    The callback gets a named object rather than positional integers because a
    progress line that misreports which number was which is worse than no
    progress line at all.
    """
    seen: list[WalkProgress] = []
    with closing(FakeStore(documents_for(5))) as store:
        take_snapshot(store, page_limit=2, on_progress=seen.append)
    assert [(step.enumerated, step.store_total, step.offset) for step in seen] == [
        (2, 5, 0),
        (4, 5, 2),
        (5, 5, 4),
    ]


def test_a_page_limit_past_what_the_store_permits_is_refused_before_any_request() -> None:
    """Tanseki caps a page at five hundred, and a walk that asked for more would be a 400.

    Refused rather than clamped, because a clamp would silently shorten the walk
    and the caller would never learn that its page size was not the one it set.
    """
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return json_response({})

    with closing(store_over(handler)) as client, pytest.raises(ValueError, match="Page limit"):
        take_snapshot(client, page_limit=MAX_PAGE_LIMIT + 1)
    assert calls == []


@pytest.mark.parametrize("page_limit", [0, -1, True, "10"])
def test_a_page_limit_that_is_not_a_usable_number_is_refused(page_limit: object) -> None:
    """``True`` is an ``int`` in Python, and page size one is not what anyone meant."""
    with (
        closing(store_over(lambda request: json_response({}))) as client,
        pytest.raises(ValueError, match="Page limit"),
    ):
        take_snapshot(client, page_limit=page_limit)  # type: ignore[arg-type]


def test_two_reads_of_one_corpus_are_two_statements_because_a_snapshot_is_about_a_moment() -> None:
    """``read_at`` is part of the identity, so a set of snapshots does not collapse them.

    The same documents read twice are two observations, and the moment is what
    tells them apart -- which is also why the error field is deliberately left
    out of the hash, so a set cannot quietly merge a failed walk into a good one.
    """
    with closing(FakeStore(documents_for(2))) as store:
        first = take_snapshot(store, page_limit=2)
    with closing(FakeStore(documents_for(2))) as store:
        second = take_snapshot(store, page_limit=2)
    assert hash(first) == hash(first)
    assert first.read_at <= second.read_at
    assert len({first, second}) == 2


def test_the_truncation_type_states_the_offset_reached_and_the_size_of_the_hole() -> None:
    """The two numbers a caveat needs, and no third that would have to be explained.

    ``records_missing`` comes from the store's own count, so it is the store's
    estimate of the hole rather than a measurement of it -- which is why the
    boundary histograms are attached to it.
    """
    truncation = Truncation(
        offset_reached=10_000,
        records_missing=412,
        boundary_repositories={"a/b": 500},
        boundary_months={"2026-09": 500},
    )
    assert (truncation.offset_reached, truncation.records_missing) == (10_000, 412)


def test_the_snapshot_reads_a_store_that_serves_its_documents_under_an_unexpected_envelope_key() -> (
    None
):
    """The whole protocol is exercised through the real client, envelope key and all.

    If the snapshot had a fake standing in for the client, this would be a test
    of the fake's idea of the protocol, and the parsing that actually matters
    would be untested.
    """
    with closing(
        FakeStore(documents_for(2), options=FakeOptions(array_key="items", bare_ids=True))
    ) as store:
        snapshot = take_snapshot(store, page_limit=2)
    assert snapshot.completeness is Completeness.COMPLETE
    assert len(snapshot.records) == 2


def test_a_snapshot_over_an_empty_store_is_complete_because_zero_is_a_real_population() -> None:
    """A store that answered and held nothing is a finding; a store that did not answer is an error.

    The distinction is the whole reason an unreachable store raises, and this is
    the other half of it: both produce zero records, and only one of them is a
    corpus of nothing.
    """
    with closing(FakeStore({})) as store:
        snapshot = take_snapshot(store)
    assert snapshot.completeness is Completeness.COMPLETE
    assert (snapshot.store_total, snapshot.enumerated) == (0, 0)
    assert snapshot.records == ()


def test_the_snapshot_uses_the_clients_collection_rather_than_a_name_of_its_own() -> None:
    """Two clients pointed at two collections must not produce indistinguishable snapshots.

    The collection is read off the client because the client is what was
    configured; a second source of truth here is a second way to be wrong about
    which corpus a figure describes.
    """
    settings = fake_settings(tanseki_collection="kojutsu-staging")
    assert settings.tanseki_collection == "kojutsu-staging"
    with closing(FakeStore(documents_for(1), options=FakeOptions(collection="other"))) as store:
        snapshot = take_snapshot(store)
    assert snapshot.collection == store.collection == "other"
    assert snapshot.records[0].collection == "other"


def test_a_store_that_says_nothing_about_paging_is_walked_to_its_own_count() -> None:
    """A missing ``hasMore`` is not a missing total: the enumeration still has a number.

    The walk trusts the store's count up to that number and no further, which is
    what stops it asking for page after page of nothing against a store whose
    envelope predates its ``hasMore``.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/documents:query"):
            ids = json.loads(request.content.decode())["ids"]
            return json_response(
                {"documents": [{"id": doc_id, "collection": "chronicler-real"} for doc_id in ids]}
            )
        offset = int(dict(parse_qsl(request.url.query.decode())).get("offset", 0))
        if offset == 0:
            return json_response({"documents": [{"id": "a"}, {"id": "b"}], "total": 2})
        return json_response({"documents": [], "total": 2})

    with closing(store_over(handler)) as client:
        snapshot = take_snapshot(client, page_limit=2)
    assert snapshot.completeness is Completeness.COMPLETE
    assert snapshot.enumerated == 2
    assert snapshot.store_total == 2


def test_a_walk_over_documents_with_nothing_readable_still_returns_every_one_of_them() -> None:
    """Every field being unreadable is a corpus of nothing, and that is a countable corpus.

    The records come back with no repository, no month and an inferred kind, which
    is a *description* of the corpus rather than a failure to read it -- and the
    alternative, dropping them, would have made a report over an unreadable corpus
    indistinguishable from a report over a small one.
    """
    documents = {
        "acme/widget/pr-1/answer-0": {"tags": ["agent_authored"]},
        "acme/widget/pr-2/answer-1": {"tags": ["agent_authored"]},
    }
    with closing(FakeStore(documents)) as store:
        snapshot = take_snapshot(store, page_limit=5)
    assert snapshot.completeness is Completeness.COMPLETE
    assert [record.month for record in snapshot.records] == [None, None]
    assert all(record.repo == "acme/widget" for record in snapshot.records)
