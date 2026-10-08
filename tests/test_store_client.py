"""What the read seam refuses, and why, is the whole content of this module.

Every test here is about a boundary: a status code, an envelope shape, a limit
the store publishes, a URL it must not be handed. The client is the one place in
this program that touches another system's data, and the failure mode it has to
prevent is not a crash -- it is a *plausible wrong answer*. An unreachable store
reported as a corpus of zero, a listing envelope with an unrecognised key read
as an empty collection, a batch read whose holes silently shift its results onto
the wrong ids: each of those produces a real-looking number about a corpus that
was never read, which is the output this project is built to never produce.

The tests are named for the reason they exist rather than the method they call,
so a reader who disagrees with a decision can find the argument and disagree
with that.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Final
from urllib.parse import parse_qsl

import httpx
import pytest

from tenbin.config import MAX_DOCUMENT_ID_LENGTH, Settings, settings_from_env
from tenbin.store.client import (
    API_KEY_HEADER,
    MAX_BATCH_IDS,
    MAX_PAGE_LIMIT,
    MAX_PAGE_OFFSET,
    MAX_SEARCH_LIMIT,
    StoreAuthenticationError,
    StoreClient,
    StoreConfigurationError,
    StoreDocument,
    StoreError,
    StoreNotFoundError,
    StoreResponseError,
    StoreUnavailableError,
)
from tests.fakes import (
    API_KEY_HEADER as FAKE_API_KEY_HEADER,
)
from tests.fakes import (
    FakeClock,
    FakeOptions,
    FakeStore,
    closing,
    document_id,
    documents_for,
    fake_settings,
    json_response,
    kojutsu_frontmatter,
    store_over,
)


def settings_bypassing_validation(tanseki_url: str) -> Settings:
    """Settings carrying a URL the validator would have refused.

    ``model_copy`` replaces a value without re-running the validator, which is
    exactly what an attacker -- or a future refactor that relaxes the settings
    rule -- would produce. The client's own guard is the backstop, and testing it
    needs a way to reach it.
    """
    return settings_from_env(tanseki_url="https://store.test/v1").model_copy(
        update={"tanseki_url": tanseki_url}
    )


# -- URL validation ---------------------------------------------------------


def test_a_credential_in_the_store_url_is_never_echoed_into_the_error() -> None:
    """A message that repeats the offending value hands the key to every log.

    The settings layer already refuses such a URL, so this constructs a
    ``Settings`` past that validator on purpose: the client's own guard is the
    backstop, and a backstop that leaks the secret is not a backstop. The
    assertion is on the *message*, not merely on the refusal, because the
    message is the part that gets logged.
    """
    bypassed = settings_bypassing_validation("https://someone:hunter2@store.test/v1")
    with (
        pytest.raises(StoreConfigurationError) as raised,
        closing(StoreClient(bypassed, client=httpx.Client())),
    ):
        pass
    assert "hunter2" not in str(raised.value)
    assert "someone" not in str(raised.value)


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "127.0.0.53", "[::1]"])
def test_a_loopback_store_may_be_reached_over_plain_http(host: str) -> None:
    """A local daemon is the common case and is not a network boundary.

    The whole 127/8 range and the whole IPv6 loopback range count, not just the two
    addresses somebody remembered: a refusal that only recognised
    ``127.0.0.1`` would push a developer into a certificate for their own machine.
    """
    from tenbin.config import settings_from_env

    url = f"http://{host}:8088/v1"
    bypassed = settings_from_env(tanseki_url="https://store.test/v1").model_copy(
        update={"tanseki_url": url}
    )
    assert StoreClient(bypassed, client=httpx.Client()).base_url == url


def test_plain_http_to_a_remote_host_is_refused_because_the_key_would_travel_in_the_clear() -> None:
    """The API key and the corpus are not secrets only to their owner."""
    bypassed = settings_bypassing_validation("http://store.example.com/v1")
    with pytest.raises(StoreConfigurationError, match="HTTPS"):
        StoreClient(bypassed, client=httpx.Client())


def test_a_non_http_scheme_is_refused_because_a_file_url_is_not_a_store() -> None:
    """Anything but HTTP(S) is a URL this client has no contract with."""
    bypassed = settings_bypassing_validation("file:///etc/passwd")
    with pytest.raises(StoreConfigurationError, match="HTTP"):
        StoreClient(bypassed, client=httpx.Client())


def test_the_client_does_not_close_a_client_it_was_given() -> None:
    """An injected client belongs to whoever made it, and may be reused.

    A client that closed something it did not open would break the next
    component that borrowed it, and would do so silently -- the borrower's next
    request would fail against a closed pool.
    """
    transport = httpx.MockTransport(lambda request: json_response({"ok": True}))
    borrowed = httpx.Client(transport=transport)
    with closing(StoreClient(fake_settings(), client=borrowed)):
        pass
    assert not borrowed.is_closed


# -- status mapping ---------------------------------------------------------


@pytest.mark.parametrize("status", [408, 425, 429, 500, 502, 503, 504])
def test_a_transient_status_is_retried_and_then_reported_as_unavailable(status: int) -> None:
    """These statuses say "not now", so they are retried and never mistaken for an answer.

    Reported as :exc:`StoreUnavailableError` rather than as a response error
    because the remedy differs: the caller can wait, and a caller told the
    response was incoherent would be right to give up on a store that is merely
    busy.
    """
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return json_response({"error": "busy"}, status=status)

    with (
        closing(store_over(handler, max_retries=1)) as client,
        pytest.raises(StoreUnavailableError),
    ):
        client.count()
    assert len(calls) == 2, "one attempt plus one retry"


def test_a_transport_failure_is_retried_and_surfaces_as_unavailable() -> None:
    """A connection refused and a 503 are the same event from a caller's side.

    Neither got a real answer, so neither is a fact about the corpus. Reporting
    a transport error as anything else would let a snapshot proceed as though it
    had read something.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host", request=request)

    with (
        closing(store_over(handler, max_retries=1)) as client,
        pytest.raises(StoreUnavailableError),
    ):
        client.count()


@pytest.mark.parametrize("status", [401, 403])
def test_a_rejected_credential_says_a_human_has_to_act(status: int) -> None:
    """Retrying an API key does nothing but burn the budget; the flag says stop.

    ``operator_action_required`` is the only thing separating "wait" from "fix
    the configuration", and a caller that cannot tell them apart either retries a
    key forever or gives up on a store that was merely slow.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response({"error": "nope"}, status=status)

    with closing(store_over(handler)) as client, pytest.raises(StoreAuthenticationError) as raised:
        client.count()
    assert raised.value.operator_action_required is True
    assert isinstance(raised.value, StoreError)


def test_a_rejected_credential_is_not_retried_because_the_answer_cannot_change() -> None:
    """One attempt, not three: the key is the same on every one of them."""
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return json_response({"error": "nope"}, status=403)

    with (
        closing(store_over(handler, max_retries=2)) as client,
        pytest.raises(StoreAuthenticationError),
    ):
        client.count()
    assert len(calls) == 1


def test_a_retry_after_header_is_honoured_rather_than_guessed_at() -> None:
    """The store said how long to wait, and the store knows things we do not.

    The recorded delay is the assertion, not the retry: a backoff that doubled
    past a 429 would turn a rate limit into a timeout.
    """
    clock = FakeClock()

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response({"error": "slow down"}, status=429, **{"Retry-After": "7"})

    with (
        closing(store_over(handler, clock=clock, max_retries=1)) as client,
        pytest.raises(StoreUnavailableError),
    ):
        client.count()
    assert clock.delays == [7.0]


def test_a_retry_after_the_client_would_not_wait_out_is_capped() -> None:
    """A store asking for an hour must not turn a snapshot into a hang.

    The cap is this program's own: an outage lasting longer than a measurement
    run is a failure to report, not a wait to sit through.
    """
    clock = FakeClock()

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response({"error": "down"}, status=503, **{"Retry-After": "86400"})

    with (
        closing(store_over(handler, clock=clock, max_retries=1)) as client,
        pytest.raises(StoreUnavailableError),
    ):
        client.count()
    assert clock.delays == [300.0]


def test_an_unparseable_retry_after_falls_back_to_the_backoff_schedule() -> None:
    """A header the store got wrong should not crash the client that read it."""
    clock = FakeClock()

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response({"error": "down"}, status=503, **{"Retry-After": "soon"})

    with (
        closing(store_over(handler, clock=clock, max_retries=1)) as client,
        pytest.raises(StoreUnavailableError),
    ):
        client.count()
    assert clock.delays == [0.2]


def test_an_unexpected_error_status_is_a_response_error_not_an_unavailable_one() -> None:
    """A 400 will be a 400 on the next attempt too, so waiting cannot help."""

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response({"error": "bad request"}, status=400)

    with closing(store_over(handler)) as client, pytest.raises(StoreResponseError):
        client.count()


def test_a_body_that_is_not_json_is_a_response_error_rather_than_a_crash() -> None:
    """A proxy's HTML error page is a broken contract, not an exception to leak."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>gateway</html>")

    with (
        closing(store_over(handler)) as client,
        pytest.raises(StoreResponseError, match="invalid JSON"),
    ):
        client.count()


def test_a_json_body_that_is_not_an_object_is_refused() -> None:
    """An envelope is an object; an array or a bare number is a different API."""

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response([1, 2, 3])

    with (
        closing(store_over(handler)) as client,
        pytest.raises(StoreResponseError, match="invalid response"),
    ):
        client.count()


# -- count ------------------------------------------------------------------


def test_a_count_request_with_no_total_is_refused_rather_than_reported_as_zero() -> None:
    """The operation *is* a count, so a response without one is a broken count.

    ``list_documents`` tolerates a missing total for a different reason -- an
    envelope shape that has varied is not a broken document -- but borrowing that
    leniency here would let a store answer "how many records?" with "zero", and
    zero is the most dangerous number in this program.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response({"documents": [], "hasMore": False})

    with (
        closing(store_over(handler)) as client,
        pytest.raises(StoreResponseError, match="no total"),
    ):
        client.count()


def test_a_negative_total_is_refused_because_no_collection_holds_a_negative_number() -> None:
    """A ``bool`` is an ``int`` in Python, so ``True`` would otherwise be a count."""

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response({"total": -1})

    with (
        closing(store_over(handler)) as client,
        pytest.raises(StoreResponseError, match="invalid total"),
    ):
        client.count()

    def boolean_handler(request: httpx.Request) -> httpx.Response:
        return json_response({"total": True})

    with (
        closing(store_over(boolean_handler)) as client,
        pytest.raises(StoreResponseError, match="invalid total"),
    ):
        client.count()


def test_counting_asks_for_one_document_because_the_answer_is_in_the_envelope() -> None:
    """The count must not cost a page of bodies to be read."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return json_response({"total": 12_345, "documents": [], "hasMore": False})

    with closing(store_over(handler)) as client:
        assert client.count() == 12_345
    assert dict(_query_params(seen[0]))["limit"] == "1"


# -- listing ----------------------------------------------------------------


def test_a_listing_with_no_document_array_is_an_error_and_never_an_empty_collection() -> None:
    """ "I could not find your documents" and "you have no documents" are opposites.

    Returning an empty list here would be the single most dangerous thing this
    client could do: a caller with nothing to distinguish them would report a
    corpus of zero, over a store it never successfully read.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response({"total": 0, "hasMore": False})

    with (
        closing(store_over(handler)) as client,
        pytest.raises(StoreResponseError, match="no document array"),
    ):
        client.list_documents()


@pytest.mark.parametrize("array_key", ["documents", "items", "results", "hits"])
def test_the_document_array_is_found_under_any_key_tanseki_has_shipped(array_key: str) -> None:
    """The envelope key has changed across Tanseki versions, and the reader has not.

    Pinning one key would turn a serialiser's rename into an empty corpus, which
    is the failure a defensive read exists to prevent.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response({array_key: ["a/b/pr-1/answer-1"], "total": 1, "hasMore": False})

    with closing(store_over(handler)) as client:
        page = client.list_documents()
    assert page.ids == ("a/b/pr-1/answer-1",)
    assert page.total == 1
    assert page.has_more is False


def test_listing_entries_may_be_bare_ids_or_objects_carrying_one() -> None:
    """Both shapes have shipped; neither is more correct than the other."""

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(
            {"documents": ["bare", {"id": "wrapped"}], "total": 2, "hasMore": False}
        )

    with closing(store_over(handler)) as client:
        assert client.list_documents().ids == ("bare", "wrapped")


def test_a_listing_entry_with_no_usable_id_is_dropped_rather_than_invented() -> None:
    """A hole in a page is visible; a fabricated id in its place is not.

    Dropping the entry leaves the enumeration short, which the snapshot
    compares against the store's own total. Inventing a placeholder would make
    the enumeration look complete while pointing at a document that does not
    exist.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(
            {"documents": [{"id": "real"}, {"path": "x.md"}, 7], "total": 3, "hasMore": False}
        )

    with closing(store_over(handler)) as client:
        assert client.list_documents().ids == ("real",)


def test_a_listing_that_omits_its_total_is_tolerated_because_the_envelope_varies() -> None:
    """Tanseki's listing has carried ``total`` and the reader must not depend on it.

    The snapshot falls back to an explicit count, so the tolerance costs one
    request rather than a figure.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response({"documents": [], "hasMore": False})

    with closing(store_over(handler)) as client:
        page = client.list_documents()
    assert page.total is None
    assert page.ids == ()


def test_a_malformed_has_more_is_refused_rather_than_read_as_its_default() -> None:
    """A paging flag that is neither true nor false is an answer this client cannot use."""

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response({"documents": [], "hasMore": "yes", "total": 0})

    with closing(store_over(handler)) as client, pytest.raises(StoreResponseError, match="hasMore"):
        client.list_documents()


def test_the_listing_reports_the_collections_own_total_so_a_walk_can_be_checked() -> None:
    """The store's count is the only independent check on whether a walk finished."""

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response({"documents": [{"id": "a"}], "total": 900, "hasMore": True})

    with closing(store_over(handler)) as client:
        page = client.list_documents()
    assert (page.total, page.has_more) == (900, True)


@pytest.mark.parametrize("limit", [0, -1, MAX_PAGE_LIMIT + 1, True, "10"])
def test_a_page_request_tanseki_would_refuse_is_refused_here_instead(limit: object) -> None:
    """Clamping would skip documents and report the walk as complete.

    A caller asking for more than the store permits has a bug, and the bug's
    symptom -- a corpus that quietly ends early -- is invisible in the output.
    """
    with (
        closing(store_over(lambda request: json_response({}))) as client,
        pytest.raises(ValueError, match="Page limit"),
    ):
        client.list_documents(limit=limit)  # type: ignore[arg-type]


@pytest.mark.parametrize("offset", [-1, MAX_PAGE_OFFSET + 1, False])
def test_an_offset_past_the_store_ceiling_is_refused_rather_than_clamped(offset: int) -> None:
    """Past ten thousand documents Tanseki has nothing to serve, and neither has a reader."""
    with (
        closing(store_over(lambda request: json_response({}))) as client,
        pytest.raises(ValueError, match="offset"),
    ):
        client.list_documents(offset=offset)


def test_an_offset_exactly_at_the_ceiling_is_allowed_because_the_store_allows_it() -> None:
    """The boundary is inclusive on the store's side, so it has to be here too.

    Off by one is the difference between reporting a truncation and losing a
    record, and a walk that stops at 9 999 when 10 000 is readable has lied about
    the size of its own hole.
    """
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return json_response({"documents": [], "total": 0, "hasMore": False})

    with closing(store_over(handler)) as client:
        client.list_documents(limit=1, offset=MAX_PAGE_OFFSET)
    assert dict(_query_params(seen[0]))["offset"] == str(MAX_PAGE_OFFSET)


# -- single document read ---------------------------------------------------


def test_a_single_document_id_travels_in_the_body_because_ids_contain_slashes() -> None:
    """A repository is ``owner/name``, so a path-addressed read is a different URL.

    Putting the id in the path would have the store looking for
    ``acme/widget/pr-42/answer-1`` as a five-segment path, which is not a
    document id at all.
    """
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return json_response({"id": "acme/widget/pr-42/answer-1", "collection": "chronicler-real"})

    with closing(store_over(handler)) as client:
        document = client.get_document("acme/widget/pr-42/answer-1")
    assert seen[0].url.path.endswith("/documents:get")
    assert b"acme/widget/pr-42/answer-1" in seen[0].content
    assert document.id == "acme/widget/pr-42/answer-1"


def test_a_missing_single_document_is_not_found_rather_than_a_value() -> None:
    """A single read has one right answer, so an absence is an exception.

    There is no alignment to preserve here, so there is no reason to turn a 404
    into ``None`` -- and a ``None`` in this position is one more thing a caller
    has to remember to check.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response({"error": "not found"}, status=404)

    with closing(store_over(handler)) as client, pytest.raises(StoreNotFoundError):
        client.get_document("acme/widget/pr-42/answer-1")


def test_a_single_read_that_returns_a_different_document_is_refused() -> None:
    """Being handed someone else's document is worse than being handed none.

    The store is keyed by id, so a mismatched identity means the response is not
    about the request, and a program that accepted it would attribute one
    record's provenance to another.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response({"id": "acme/widget/pr-42/answer-2", "collection": "chronicler-real"})

    with (
        closing(store_over(handler)) as client,
        pytest.raises(StoreResponseError, match="unexpected document identity"),
    ):
        client.get_document("acme/widget/pr-42/answer-1")


def test_a_document_from_another_collection_is_refused_because_it_is_another_corpus() -> None:
    """A reader must not silently widen its own population mid-walk."""

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(
            {
                "id": "acme/widget/pr-42/answer-1",
                "collection": "kojutsu-staging",
            }
        )

    with (
        closing(store_over(handler)) as client,
        pytest.raises(StoreResponseError, match="unexpected collection"),
    ):
        client.get_document("acme/widget/pr-42/answer-1")


def test_a_document_without_a_body_is_still_a_document() -> None:
    """A batch projection may omit the prose, and Tenbin measures frontmatter.

    Refusing a document over the one field this program does not use would remove
    a countable record from every denominator, which is the wrong trade.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(
            {
                "id": "acme/widget/pr-42/answer-1",
                "collection": "chronicler-real",
                "frontmatter": {"title": "t"},
            }
        )

    with closing(store_over(handler)) as client:
        document = client.get_document("acme/widget/pr-42/answer-1")
    assert document.content == ""
    assert document.frontmatter["title"] == "t"


def test_a_document_with_the_wrong_frontmatter_type_is_refused() -> None:
    """Frontmatter that is a list is not frontmatter, and cannot be read as one."""

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(
            {
                "id": "acme/widget/pr-42/answer-1",
                "collection": "chronicler-real",
                "frontmatter": ["title"],
            }
        )

    with (
        closing(store_over(handler)) as client,
        pytest.raises(StoreResponseError, match="frontmatter"),
    ):
        client.get_document("acme/widget/pr-42/answer-1")


@pytest.mark.parametrize("spelling", ["contentHash", "content_hash"])
def test_both_spellings_of_the_content_hash_are_accepted(spelling: str) -> None:
    """Tanseki's serialisers have shipped both, and rejecting either loses real documents.

    The tolerance is for the spelling, never for the type: a field that is
    present and is not a string is still a broken response.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(
            {
                "id": "acme/widget/pr-42/answer-1",
                "collection": "chronicler-real",
                spelling: "sha256:abc",
            }
        )

    with closing(store_over(handler)) as client:
        assert client.get_document("acme/widget/pr-42/answer-1").content_hash == "sha256:abc"


@pytest.mark.parametrize("spelling", ["updatedAt", "updated_at"])
def test_both_spellings_of_the_update_time_are_accepted(spelling: str) -> None:
    """Same reason as the hash, and the same limit on the tolerance."""

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(
            {
                "id": "acme/widget/pr-42/answer-1",
                "collection": "chronicler-real",
                spelling: "2026-09-28T00:00:00+00:00",
            }
        )

    with closing(store_over(handler)) as client:
        document = client.get_document("acme/widget/pr-42/answer-1")
    assert document.updated_at == "2026-09-28T00:00:00+00:00"


def test_an_empty_document_id_is_refused_before_it_reaches_the_store() -> None:
    """A request for the empty document is a bug, and the store is not the place to find out."""
    with (
        closing(store_over(lambda request: json_response({}))) as client,
        pytest.raises(ValueError, match="non-empty"),
    ):
        client.get_document("  ")


def test_an_unbounded_document_id_is_refused_because_ids_reach_error_messages() -> None:
    """An id is quoted in ``StoreNotFoundError``, so an unbounded one is an unbounded log line."""
    with (
        closing(store_over(lambda request: json_response({}))) as client,
        pytest.raises(ValueError, match=str(MAX_DOCUMENT_ID_LENGTH)),
    ):
        client.get_document("a" * (MAX_DOCUMENT_ID_LENGTH + 1))


def test_the_api_key_travels_in_the_header_tanseki_reads() -> None:
    """The header name is a contract, and a second spelling authenticates as nobody."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return json_response({"total": 0})

    with closing(store_over(handler, tanseki_api_key="s3cret")) as client:
        client.count()
    assert seen[0].headers[FAKE_API_KEY_HEADER] == "s3cret"
    assert API_KEY_HEADER == "X-API-Key"


def test_no_api_key_is_sent_when_none_is_configured() -> None:
    """An empty header is a header the store will reject, and a confusing way to find out."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return json_response({"total": 0})

    with closing(store_over(handler, tanseki_api_key="")) as client:
        client.count()
    assert FAKE_API_KEY_HEADER not in seen[0].headers


# -- batch read -------------------------------------------------------------


def test_a_batch_read_stays_positionally_aligned_with_the_ids_it_was_asked_for() -> None:
    """The store may return documents in any order, and a shift is undetectable downstream.

    A caller zipping results against its own list has no way to notice a
    re-ordering, so alignment is a property of this method rather than something
    the caller is trusted to do.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(
            {
                "documents": [
                    {"id": "b", "collection": "chronicler-real"},
                    {"id": "a", "collection": "chronicler-real"},
                ]
            }
        )

    with closing(store_over(handler)) as client:
        fetched = client.get_documents(["a", "b"])
    assert [None if d is None else d.id for d in fetched] == ["a", "b"]


def test_an_id_the_store_omits_becomes_none_rather_than_shifting_the_rest() -> None:
    """:query omits ids it does not hold, and a hole must stay where it is.

    Dropping the hole would re-attach every later document to the wrong id, and
    the resulting records would be internally consistent and entirely wrong --
    the most expensive kind of bug a measurement program can have.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(
            {
                "documents": [
                    {"id": "a", "collection": "chronicler-real"},
                    {"id": "c", "collection": "chronicler-real"},
                ]
            }
        )

    with closing(store_over(handler)) as client:
        fetched = client.get_documents(["a", "b", "c"])
    assert [None if d is None else d.id for d in fetched] == ["a", None, "c"]


def test_a_batch_larger_than_the_store_accepts_is_chunked_rather_than_refused() -> None:
    """Tanseki caps ``:query`` at a hundred ids, and a corpus walk is always larger.

    Walking a real corpus in one request would be a 400 on the first page, so
    the chunking is not a convenience -- it is the difference between a snapshot
    and a stack trace.
    """
    sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        ids = json.loads(request.content.decode())["ids"]
        sizes.append(len(ids))
        return json_response(
            {"documents": [{"id": doc_id, "collection": "chronicler-real"} for doc_id in ids]}
        )

    with closing(store_over(handler)) as client:
        fetched = client.get_documents([f"id-{index}" for index in range(250)])
    assert sizes == [MAX_BATCH_IDS, MAX_BATCH_IDS, 50]
    assert len(fetched) == 250


def test_an_empty_id_list_asks_the_store_nothing() -> None:
    """A request with no ids is a round trip that can only fail."""
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return json_response({"documents": []})

    with closing(store_over(handler)) as client:
        assert client.get_documents([]) == []
    assert calls == []


def test_a_batch_response_without_a_document_array_is_refused() -> None:
    """A missing array is not an empty batch, and the difference is the whole corpus."""

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response({"hits": []})

    with (
        closing(store_over(handler)) as client,
        pytest.raises(StoreResponseError, match="no document array"),
    ):
        client.get_documents(["a"])


# -- search -----------------------------------------------------------------


def test_a_search_with_no_total_is_accepted_because_tanseki_never_sends_one() -> None:
    """Demanding a field the store does not have would make search unrefusable.

    Tanseki's search response is a pagination envelope, ``{hits, limit, offset,
    hasMore}``, and has never carried a ``total``. A reader that required one
    would reject every correct response.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(
            {"hits": [{"id": "a", "score": 1.0}], "limit": 10, "offset": 0, "hasMore": False}
        )

    with closing(store_over(handler)) as client:
        result = client.search("queue")
    assert result.total is None
    assert [hit.id for hit in result.hits] == ["a"]


def test_a_search_total_smaller_than_the_hits_returned_is_refused() -> None:
    """A present total that contradicts its own page is not a missing field.

    Accepting it would mean a report could state "3 results" over five hits, and
    the discrepancy would be invisible because both numbers came from the same
    response.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(
            {"hits": [{"id": "a", "score": 1.0}, {"id": "b", "score": 0.5}], "total": 1}
        )

    with (
        closing(store_over(handler)) as client,
        pytest.raises(StoreResponseError, match="fewer results than its total"),
    ):
        client.search("queue")


def test_search_sends_tag_and_frontmatter_filters_as_repeated_query_parameters() -> None:
    """A ``GET`` has no body, so the filters have to be in the query string.

    And they have to be *repeated* rather than joined: Tanseki reads each ``tags``
    and each ``fm=`` as a separate filter, and one comma-joined value is a filter
    matching nothing.
    """
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return json_response({"hits": [], "hasMore": False})

    with closing(store_over(handler)) as client:
        client.search("queue", tags=["review", "agent_authored"], frontmatter={"repo": "a/b"})
    pairs = list(_query_params(seen[0], keep_blank=True))
    assert ("tags", "review") in pairs
    assert ("tags", "agent_authored") in pairs
    assert ("fm", "repo=a/b") in pairs


def test_a_search_hit_without_a_finite_score_is_refused() -> None:
    """A NaN score has no ordering, so a ranked result carrying one is not ranked."""

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response({"hits": [{"id": "a", "score": "high"}]})

    with closing(store_over(handler)) as client, pytest.raises(StoreResponseError, match="score"):
        client.search("queue")


def test_a_search_limit_past_this_programs_bound_is_refused() -> None:
    """The bound is ours, not the store's, and a mistyped limit must not become a huge request."""
    with (
        closing(store_over(lambda request: json_response({"hits": []}))) as client,
        pytest.raises(ValueError, match="Search limit"),
    ):
        client.search("queue", limit=MAX_SEARCH_LIMIT + 1)


def test_search_results_are_truncated_to_the_limit_the_caller_asked_for() -> None:
    """A store that returns more than was asked for must not widen the caller's window."""

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response({"hits": [{"id": str(index), "score": 1.0} for index in range(10)]})

    with closing(store_over(handler)) as client:
        result = client.search("queue", limit=3)
    assert len(result.hits) == 3


# -- health -----------------------------------------------------------------


def test_health_reports_false_rather_than_raising_when_the_store_is_down() -> None:
    """A probe that raises is a probe nobody can use in a status line."""

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response({"error": "no"}, status=503)

    with closing(store_over(handler, max_retries=0)) as client:
        assert client.health() is False


def test_health_reports_false_on_a_refused_credential_too() -> None:
    """A store that will not let us in is not healthy, however much it is up."""

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response({"error": "no"}, status=401)

    with closing(store_over(handler)) as client:
        assert client.health() is False


def test_health_reports_true_only_for_a_two_hundred_something() -> None:
    """A 404 from a store with no health route is an answer of no, not a crash."""

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response({"error": "no"}, status=404)

    with closing(store_over(handler)) as client:
        assert client.health() is False

    def ok(request: httpx.Request) -> httpx.Response:
        return httpx.Response(204)

    with closing(store_over(ok)) as client:
        assert client.health() is True


# -- the absence of a writer ------------------------------------------------


#: The requests this program is allowed to make. Every entry is a *read*, and the
#: colon form is Tanseki's: a ``:suffix`` on a ``POST`` names a query rather than a
#: mutation, which is the one thing that makes ``POST`` safe to reason about here
#: and the reason the test below keys on (method, path) pairs rather than on verbs.
READ_ENDPOINTS: Final[frozenset[tuple[str, str]]] = frozenset(
    {
        ("GET", "/v1/health"),
        ("GET", "/v1/documents"),
        ("GET", "/v1/search"),
        ("POST", "/v1/documents:get"),
        ("POST", "/v1/documents:query"),
        ("POST", "/v1/documents:traverse"),
    }
)

#: Enough of an answer for every read method to finish without raising, keyed by the
#: path each one asks for. A single read answers with the document itself at the top
#: level and a batch answers with an array; the client's own tests pin both shapes,
#: and a fixture that matched only one would be testing the envelope instead.
ANSWERS: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "/v1/health": {},
        "/v1/documents": {"documents": [], "total": 0},
        "/v1/search": {"hits": [], "limit": 1, "offset": 0, "hasMore": False},
        "/v1/documents:get": {"id": "a-document", "frontmatter": {}, "content": ""},
        "/v1/documents:query": {"documents": [{"id": "a-document", "frontmatter": {}}]},
        "/v1/documents:traverse": {"ids": []},
    }
)

#: How to call each public read, by method name. Keyword arguments are kept
#: separate because some of them are keyword-only on the client, and a fixture
#: that had to flatten them would be calling the method wrongly rather than
#: testing it. A method with no entry here is unexercised, and the test says so by
#: name rather than skipping it.
CALLS: Final[Mapping[str, tuple[tuple[Any, ...], Mapping[str, Any]]]] = MappingProxyType(
    {
        "health": ((), {}),
        "count": ((), {}),
        "list_documents": ((), {}),
        "get_document": (("a-document",), {}),
        "get_documents": (("a-document",), {}),
        "search": (("a query",), {}),
        "traverse": (("a-document", "repo"), {"depth": 1}),
    }
)


def test_the_client_issues_no_request_that_writes_to_the_store() -> None:
    """The read seam is a guarantee, so it is asserted against requests and not names.

    A measurement program that can modify the corpus it measures is measuring a
    corpus it may have edited, and the edit appears in no figure it later
    reports.

    **Every public method is called and the requests it issues are inspected.**
    An earlier version of this test scanned method *names* for write-shaped
    substrings, which was wrong in both directions at once. It could not pass a
    reader called ``traverse`` -- a read -- and it would have failed a future
    ``compute_counts`` because ``compute`` contains ``put``. A name is not a
    capability. What this test asserts instead is the thing it actually means:
    the client issues no ``PUT``, ``PATCH`` or ``DELETE``, and no ``POST`` outside
    the fixed set of query endpoints above. A writer called ``archive`` or
    ``record`` fails this; a reader called anything fails nothing.
    """
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        return json_response(ANSWERS[request.url.path])

    public = [name for name in dir(StoreClient) if not name.startswith("_")]
    unexercised = sorted(set(public) - set(CALLS) - {"close"})

    with closing(store_over(handler)) as client:
        for name, (args, kwargs) in CALLS.items():
            seen.clear()
            getattr(client, name)(*args, **kwargs)
            assert seen, f"{name}() issued no request, so this test checked nothing for it"
            offenders = [pair for pair in seen if pair not in READ_ENDPOINTS]
            assert not offenders, (
                f"{name}() issued {offenders}, which is outside the read endpoints "
                f"{sorted(READ_ENDPOINTS)}. A measurement program that can write to the "
                "corpus it measures is measuring a corpus it may have edited."
            )

    assert not unexercised, (
        f"{unexercised} are public and unexercised: add them to CALLS with an answer, "
        "because a method this test cannot call is a method whose requests it cannot "
        "check. A new read belongs in READ_ENDPOINTS too."
    )


def test_the_clients_public_surface_is_closed_because_a_deliberate_read_belongs_in_a_review() -> (
    None
):
    """Completeness, not safety: a new read is somebody's decision, not a diff's.

    The test above is the one that proves the seam is read-only, and it does so
    against requests. This one exists for a different reason and does not replace
    that one: a measurement program publishes a fixed set of things about a
    corpus, and a new read is a new claim about that corpus, so it belongs in a
    review of this file with its reasons written down rather than in a diff that
    only shows the new name. ``traverse`` is here because the graph arrived with
    Kojutsu's edge derivation and reading it is a deliberate read.
    """
    assert {name for name in dir(StoreClient) if not name.startswith("_")} == {
        "close",
        "count",
        "get_document",
        "get_documents",
        "health",
        "list_documents",
        "search",
        "traverse",
    }


def test_a_document_value_cannot_be_edited_through_the_record_that_quotes_it() -> None:
    """A frozen dataclass holding a dict is not frozen, and a report would be editable.

    The frontmatter is the evidence a record is built from, so a caller who
    could change it could change a classification without touching a single line
    of this program.
    """
    document = StoreDocument(id="a", collection="c", frontmatter={"tags": ["review"]})
    with pytest.raises(TypeError):
        document.frontmatter["tags"] = []  # type: ignore[index]


# -- against the in-memory protocol fake ------------------------------------


def test_the_fake_serves_a_document_through_the_same_read_path_as_a_real_store() -> None:
    """The fake is a transport, not a stand-in for the client.

    If it were a stand-in, every test using it would be testing the stand-in's
    idea of the protocol, and the client under it would never be exercised at
    all.
    """
    store = FakeStore(
        {document_id("acme/widget", 42, "answer-1"): kojutsu_frontmatter(tags=["agent_authored"])}
    )
    with closing(store) as client:
        assert client.count() == 1
        assert client.get_document("acme/widget/pr-42/answer-1").id == (
            "acme/widget/pr-42/answer-1"
        )


def test_the_fake_can_make_the_stores_own_count_contradict_its_listing() -> None:
    """The snapshot's ``store_total_exceeded`` case needs a store that lies about itself.

    Offered here as a capability of the fake rather than as a test of the
    snapshot, so the lying store exists in one place for both to use.
    """
    store = FakeStore({}, options=FakeOptions(total=99))
    with closing(store) as client:
        assert client.count() == 99
        assert client.list_documents().total == 99


def _query_params(request: httpx.Request, *, keep_blank: bool = False) -> list[tuple[str, str]]:
    return parse_qsl(request.url.query.decode(), keep_blank_values=keep_blank)


# -- document shape ---------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("id", 7, "invalid id"),
        ("id", "  ", "invalid id"),
        ("content", ["text"], "invalid content"),
        ("path", 7, "invalid path"),
        ("frontmatter", ["title"], "invalid frontmatter"),
        ("revision", 7, "invalid revision"),
        ("contentHash", 7, "invalid contentHash"),
        ("updatedAt", 7, "invalid updatedAt"),
    ],
)
def test_a_document_field_that_is_present_and_of_the_wrong_type_is_refused(
    field: str, value: object, message: str
) -> None:
    """The tolerance in this client is for a *spelling* varying, never for a type.

    A number where a document id belongs is not a document the store sent; it is a
    response that is not about this request, and a program that read one anyway
    would be counting something it has no description of.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        body: dict[str, object] = {"id": "a/b/pr-1/answer-1", "collection": "chronicler-real"}
        body[field] = value
        return json_response(body)

    with closing(store_over(handler)) as client, pytest.raises(StoreResponseError, match=message):
        client.get_document("a/b/pr-1/answer-1")


def test_a_batch_entry_that_is_not_an_object_is_refused_rather_than_skipped() -> None:
    """A list where a document belongs is a different endpoint's answer.

    Skipping it would be the quiet version of the hole problem: a batch read that
    dropped what it could not parse would return a shorter list than it was asked
    for, and a caller zipping against its ids would see ``None`` for a document the
    store *did* hold.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response({"documents": ["a/b/pr-1/answer-1"]})

    with (
        closing(store_over(handler)) as client,
        pytest.raises(StoreResponseError, match="invalid document"),
    ):
        client.get_documents(["a/b/pr-1/answer-1"])


def test_a_single_read_whose_body_is_an_array_is_refused_as_an_invalid_response() -> None:
    """The envelope is checked before the document is, so the message names the envelope."""

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response([{"id": "a/b/pr-1/answer-1"}])

    with (
        closing(store_over(handler)) as client,
        pytest.raises(StoreResponseError, match="invalid response"),
    ):
        client.get_document("a/b/pr-1/answer-1")


def test_a_store_returning_the_same_id_twice_resolves_to_the_first_copy() -> None:
    """A duplicate in a batch is a store quirk, not a reason to fail the whole read.

    First-wins rather than last-wins so the result does not depend on the order
    the store happened to serialise, which is not a thing a reader can check.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(
            {
                "documents": [
                    {"id": "a", "collection": "chronicler-real", "revision": "first"},
                    {"id": "a", "collection": "chronicler-real", "revision": "second"},
                ]
            }
        )

    with closing(store_over(handler)) as client:
        fetched = client.get_documents(["a"])
    assert [None if d is None else d.revision for d in fetched] == ["first"]


def test_a_document_handed_to_the_reader_cannot_be_edited_through_the_callers_dict() -> None:
    """The mapping is copied on the way in, so the caller's dict is not the record's evidence.

    Without the copy, a caller that reused one document dict across two records
    would find both of them changing together, and neither would be a frozen
    dataclass any more.
    """
    source = {"title": "one"}
    document = StoreDocument(id="a", collection="c", frontmatter=source)
    source["title"] = "two"
    assert document.frontmatter["title"] == "one"


# -- search shape -----------------------------------------------------------


def test_a_search_hit_that_is_not_an_object_is_refused() -> None:
    """A bare string in the hits array is not a hit, and no score can be read from it."""

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response({"hits": ["a/b/pr-1/answer-1"]})

    with (
        closing(store_over(handler)) as client,
        pytest.raises(StoreResponseError, match="invalid hit"),
    ):
        client.search("queue")


def test_a_search_hit_without_a_score_is_refused_because_a_ranking_needs_one() -> None:
    """``score`` is the only thing that makes the list ordered, and absence is not zero."""

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response({"hits": [{"id": "a/b/pr-1/answer-1"}]})

    with (
        closing(store_over(handler)) as client,
        pytest.raises(StoreResponseError, match="invalid score"),
    ):
        client.search("queue")


def test_a_search_hit_with_a_non_string_snippet_is_refused() -> None:
    """A snippet is text, and a list of strings is some other field's value."""

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response({"hits": [{"id": "a", "score": 1.0, "snippet": ["x"]}]})

    with (
        closing(store_over(handler)) as client,
        pytest.raises(StoreResponseError, match="invalid snippet"),
    ):
        client.search("queue")


def test_a_search_response_with_no_hits_array_is_refused() -> None:
    """A search that found nothing is a well-formed response, so its absence is a broken one."""

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response({"results": []})

    with (
        closing(store_over(handler)) as client,
        pytest.raises(StoreResponseError, match="invalid response"),
    ):
        client.search("queue")


def test_a_search_offset_must_be_a_non_negative_integer() -> None:
    """A negative offset is not "the end of the corpus"; it is a request with no meaning."""
    with (
        closing(store_over(lambda request: json_response({"hits": []}))) as client,
        pytest.raises(ValueError, match="offset"),
    ):
        client.search("queue", offset=-1)


# -- client lifecycle -------------------------------------------------------


def test_a_client_used_as_a_context_manager_closes_the_pool_it_opened() -> None:
    """The context manager exists so that the common case cannot leak a connection.

    A leaked pool is a connection to somebody else's knowledge store that outlives
    the process that opened it, and the suite treats that as an error rather than
    as tidiness. This is the one test that builds a real client rather than
    injecting a transport, because the assertion is about pool *ownership* and an
    injected client is one this class does not own.
    """
    client = StoreClient(fake_settings())
    with client:
        assert client.collection == "chronicler-real"
    with pytest.raises(RuntimeError, match="has been closed"):
        client._client.get("/health")  # noqa: SLF001 - the assertion is the ownership rule


def test_the_fake_serves_the_document_body_a_test_wrote_and_nothing_else() -> None:
    """A fake that invented prose would hide the tolerant path a batch projection takes."""
    fake = FakeStore(documents_for(1))
    with closing(fake) as store:
        assert store.get_document(fake.ids[0]).content == "# body\n"


def test_a_search_hit_with_no_id_is_refused_because_a_hit_named_nothing_is_not_a_hit() -> None:
    """``id`` is the only field a hit cannot do without; a score alone identifies nothing."""

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response({"hits": [{"score": 1.0}]})

    with closing(store_over(handler)) as client, pytest.raises(StoreResponseError, match="no id"):
        client.search("queue")


def test_a_search_hit_with_a_non_finite_score_is_refused_because_infinity_has_no_order() -> None:
    """A ranking containing a non-finite score does not sort.

    The body is written raw because strict JSON cannot encode one, which is itself
    the point: a store that sent this is not conforming, and the reader refuses it
    rather than propagating a number that compares false against everything.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b'{"hits": [{"id": "a", "score": Infinity}]}')

    with closing(store_over(handler)) as client, pytest.raises(StoreResponseError, match="score"):
        client.search("queue")


def test_the_authentication_error_says_a_human_must_act_even_when_it_is_caught_as_a_store_error() -> (
    None
):
    """A caller catching the base class still has to be able to tell the two failures apart.

    ``health`` swallows every store failure, so the flag is the only way a caller
    that caught the exception can recover which happened.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response({"error": "nope"}, status=401)

    with closing(store_over(handler)) as client, pytest.raises(StoreError) as raised:
        client.count()
    assert getattr(raised.value, "operator_action_required", False) is True
