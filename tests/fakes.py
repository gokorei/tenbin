"""Fakes for the read seam, shared so that a test states its store rather than building one.

Two shapes, and the split is deliberate. :class:`FakeStore` is a whole
``/v1`` protocol in memory, for the tests that walk a corpus and need the store
to behave like a store -- paging, batch queries that omit what they do not hold,
a ``total`` that can be made to lie. ``store_over`` is the opposite: a single
callable returning a canned response, for a test whose subject is one envelope
or one status code and which should not have to care what the rest of the
protocol does.

Both are built on ``httpx.MockTransport``, so no socket is opened and the
``conftest`` denial stays a real denial rather than a promise. Both take an
injected clock, so the retry path is exercised without spending the backoff --
:func:`closing` exists because the suite treats a leaked connection as an error,
and a fake that could only be abandoned would be a fake every test had to work
around.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qsl

import httpx

from tenbin.config import Settings, settings_from_env
from tenbin.corpus.record import Record
from tenbin.store.client import StoreClient, StoreDocument

#: The collection the fakes serve. Matches the real default so a test that
#: forgets to set one is still testing something real.
FAKE_COLLECTION = "chronicler-real"

#: The key a credential travels under, restated here so a test can assert the
#: header was sent without importing the client's constant and thereby proving
#: only that the constant matches itself.
API_KEY_HEADER = "X-API-Key"


class FakeClock:
    """A stand-in for ``time.sleep`` that records what it was asked to wait for.

    Named ``Fake`` rather than ``Mock`` because nothing is being asserted about
    call counts here -- the only thing that matters is that the backoff
    sequence is observable, so a test can check the store's ``Retry-After`` was
    honoured rather than merely that a retry happened.
    """

    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)

    @property
    def total_delay(self) -> float:
        return sum(self.delays)


@dataclass(frozen=True)
class FakeOptions:
    """The ways a fake store may be made to misbehave, all optional.

    Every knob here corresponds to a thing the real store has actually done or
    could do: a total that disagrees with its own listing, an envelope with the
    array under an unexpected key, a batch query that omits a document that was
    listed a moment ago, a transport error partway through a walk. They are
    options rather than behaviour toggles scattered through the fake so that a
    test's own source says which misbehaviour it is testing.
    """

    collection: str = FAKE_COLLECTION
    api_key: str = "fake-key"
    #: Override the ``total`` the listing and the count report.
    total: int | None = None
    #: Override ``hasMore`` on every page.
    has_more: bool | None = None
    #: Key the document array appears under in the listing envelope.
    array_key: str = "documents"
    #: Send listing entries as bare strings rather than objects carrying an ``id``.
    bare_ids: bool = False
    #: Omit the ``total`` from the listing envelope, as Tanseki's search envelope does.
    omit_total: bool = False
    #: Ids that ``:query`` will not return, simulating a document deleted between
    #: the listing and the fetch.
    omit_ids: frozenset[str] = frozenset()
    #: Listing offsets at which the transport fails, simulating a store that goes
    #: away mid-walk rather than one that was never there.
    fail_at_offsets: frozenset[int] = frozenset()
    #: Fail every request with this status, for a client-level test.
    fail_status: int | None = None
    #: How many attempts to allow, so a retry test can exhaust the budget.
    max_retries: int = 2


def fake_settings(**overrides: Any) -> Settings:
    """Settings pointing at a store that does not exist, for building a client.

    The URL is a loopback address over plain HTTP: allowed by the configuration
    rules, and never dialled, because every fake here installs a
    ``MockTransport`` that answers in-process.
    """
    defaults: dict[str, Any] = {
        "tanseki_url": "http://127.0.0.1:8088/v1",
        "tanseki_collection": FAKE_COLLECTION,
    }
    return settings_from_env(**{**defaults, **overrides})


def kojutsu_frontmatter(
    *,
    tags: Sequence[str] | str | None = None,
    repo: str | None = "acme/widget",
    pr: int | str | None = 42,
    title: str | None = "Why is the queue bounded?",
    author: str | None = "someone",
    updated_at: str | None = "2026-09-28T00:00:00+00:00",
    capture_source: str | None = "webhook",
    **extras: Any,
) -> dict[str, Any]:
    """Frontmatter as Kojutsu writes it: typed keys first, stringified extras.

    Every extra is stringified on the way in, because that is what the writing
    project does and because a fake that handed the client a real ``int`` would
    be testing a leniency the store does not have. The typed parameters accept
    ``None`` so that a test can write a document *missing* a key, which is a
    different corpus from one holding the empty string; ``tags`` also accepts a
    bare string, which is the wrong type for the key and the shape a hand-edited
    document can arrive in.
    """
    frontmatter: dict[str, Any] = {
        "title": title,
        "author": author,
        "tags": tags if tags is not None else ["agent_authored"],
        "updated_at": updated_at,
        "capture_source": capture_source,
    }
    frontmatter = {key: value for key, value in frontmatter.items() if value is not None}
    if repo is not None:
        frontmatter["repo"] = repo
    if pr is not None:
        frontmatter["pr"] = str(pr)
    for key, value in extras.items():
        if value is not None:
            frontmatter[key] = str(value)
    return frontmatter


def document_id(
    repo: str | None,
    pr: int | None,
    entry_id: str,
    *,
    rationale: bool = False,
) -> str:
    """Build an id in the shape Kojutsu's path derivation produces.

    The ``pr-0`` placeholder is what Kojutsu writes when there is no pull
    request, so a fake asking for a record with no PR gets that rather than a
    zero that would have to be special-cased by every test.
    """
    prefix = f"{repo or 'unknown'}/pr-{pr or 0}"
    return f"{prefix}/rationale/{entry_id}" if rationale else f"{prefix}/{entry_id}"


def fake_document(
    doc_id: str,
    frontmatter: Mapping[str, Any],
    *,
    collection: str = FAKE_COLLECTION,
    content: str = "# body\n",
) -> dict[str, Any]:
    """A wire document as Tanseki serialises one, camelCased and complete."""
    return {
        "id": doc_id,
        "path": f"{doc_id}.md",
        "collection": collection,
        "content": content,
        "frontmatter": dict(frontmatter),
        "revision": "rev-1",
        "updatedAt": frontmatter.get("updated_at"),
        "contentHash": "sha256:deadbeef",
    }


def _loose_int(raw: object) -> int | None:
    """Read a number out of frontmatter the way the reader does, leniently.

    So a test can hand ``record_from_frontmatter`` a document whose ``pr`` is
    unreadable -- which is precisely the case whose id cannot be derived, and
    therefore a document a test has to name by hand.
    """
    try:
        return int(str(raw))
    except (TypeError, ValueError):
        return None


def record_from_frontmatter(
    frontmatter: Mapping[str, Any],
    *,
    doc_id: str | None = None,
    collection: str = FAKE_COLLECTION,
    content: str = "# body\n",
) -> Record:
    """The shortest path from "here is a document" to "here is a record".

    Used by the tests whose subject is what a record *says* rather than how it
    was fetched, so that they do not each have to spell out a document and a
    collection to get there. It goes through the real
    :meth:`Record.from_document`, so nothing is bypassed to make a test easier.
    """
    resolved_id = doc_id or document_id(
        str(frontmatter["repo"]) if "repo" in frontmatter else None,
        _loose_int(frontmatter.get("pr")),
        str(frontmatter.get("entry_id", "answer-0")),
    )
    document = fake_document(resolved_id, frontmatter, collection=collection, content=content)
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


class FakeStore(StoreClient):
    """A whole ``/v1`` protocol in memory, for tests that walk a corpus.

    Subclasses the real client rather than standing in for it, so the snapshot
    tests exercise the actual envelope parsing, the actual batch alignment and
    the actual error mapping. A stand-in would test the stand-in.

    ``documents`` maps an id to its frontmatter; the wire document is derived,
    because no fake needs a hand-written ``contentHash``.
    """

    def __init__(
        self,
        documents: Mapping[str, Mapping[str, Any]] | None = None,
        *,
        options: FakeOptions | None = None,
        clock: FakeClock | None = None,
    ) -> None:
        self._options = options or FakeOptions()
        self._documents: dict[str, dict[str, Any]] = {
            doc_id: fake_document(doc_id, frontmatter, collection=self._options.collection)
            for doc_id, frontmatter in (documents or {}).items()
        }
        self._order = list(self._documents)
        self.clock = clock or FakeClock()
        #: Every request the fake saw, in order, for tests about paging and
        #: about what was and was not asked for.
        self.requests: list[httpx.Request] = []
        settings = fake_settings(
            tanseki_api_key=self._options.api_key,
            tanseki_collection=self._options.collection,
        )
        transport = httpx.MockTransport(self._handle)
        super().__init__(
            settings,
            client=httpx.Client(transport=transport, base_url=settings.tanseki_url),
            max_retries=self._options.max_retries,
            sleep=self.clock,
        )

    @property
    def ids(self) -> list[str]:
        """The ids this store holds, in the order it will serve them."""
        return list(self._order)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        options = self._options
        if options.fail_status is not None:
            return httpx.Response(options.fail_status, json={"error": "scripted"})
        path = request.url.path
        if path.endswith("/health"):
            return httpx.Response(200, json={"ok": True})
        if path.endswith("/search"):
            return self._search(request)
        if path.endswith("/documents:get"):
            return self._get(request)
        if path.endswith("/documents:query"):
            return self._query(request)
        if path.endswith("/documents"):
            return self._list(request)
        return httpx.Response(404, json={"error": "no such route"})

    def _report_total(self) -> int:
        options = self._options
        return options.total if options.total is not None else len(self._order)

    def _list(self, request: httpx.Request) -> httpx.Response:
        options = self._options
        params = dict(parse_qsl(request.url.query.decode()))
        offset = int(params.get("offset", 0))
        limit = int(params.get("limit", 1))
        if offset in options.fail_at_offsets:
            raise httpx.ConnectError("scripted transport failure", request=request)
        window = self._order[offset : offset + limit]
        entries: list[Any] = (
            list(window) if options.bare_ids else [{"id": doc_id} for doc_id in window]
        )
        has_more = (
            options.has_more
            if options.has_more is not None
            else offset + len(window) < self._report_total()
        )
        body: dict[str, Any] = {options.array_key: entries, "hasMore": has_more}
        if not options.omit_total:
            body["total"] = self._report_total()
        return httpx.Response(200, json=body)

    def _get(self, request: httpx.Request) -> httpx.Response:
        payload = _json_body(request)
        doc_id = str(payload.get("id"))
        document = self._documents.get(doc_id)
        if document is None:
            return httpx.Response(404, json={"error": "not found"})
        return httpx.Response(200, json=document)

    def _query(self, request: httpx.Request) -> httpx.Response:
        options = self._options
        payload = _json_body(request)
        ids = payload.get("ids") or []
        documents = [
            self._documents[doc_id]
            for doc_id in ids
            if doc_id in self._documents and doc_id not in options.omit_ids
        ]
        return httpx.Response(200, json={"documents": documents})

    def _search(self, request: httpx.Request) -> httpx.Response:
        params = dict(parse_qsl(request.url.query.decode()))
        limit = int(params.get("limit", 10))
        query = params.get("q", "")
        matched = [doc_id for doc_id in self._order if query in doc_id]
        window = matched[:limit]
        return httpx.Response(
            200,
            json={
                "hits": [{"id": doc_id, "score": 1.0, "snippet": "..."} for doc_id in window],
                "limit": limit,
                "offset": int(params.get("offset", 0)),
                "hasMore": len(window) < len(matched),
            },
        )


def _json_body(request: httpx.Request) -> dict[str, Any]:
    return json.loads(request.content.decode())


def store_over(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    clock: FakeClock | None = None,
    max_retries: int = 2,
    **settings_overrides: Any,
) -> StoreClient:
    """A real client over a canned response, for a test about one envelope.

    The handler is a plain function taking the request and returning a
    response, which keeps the test's own source showing exactly what the store
    said -- the alternative, a fake configured with flags, hides the payload
    behind a constructor argument and makes the reader go and look it up.
    """
    settings = fake_settings(**settings_overrides)
    transport = httpx.MockTransport(handler)
    return StoreClient(
        settings,
        client=httpx.Client(transport=transport, base_url=settings.tanseki_url),
        max_retries=max_retries,
        sleep=clock or FakeClock(),
    )


def json_response(payload: Any, *, status: int = 200, **headers: str) -> httpx.Response:
    """A JSON response, for a handler that only has opinions about the body."""
    return httpx.Response(status, json=payload, headers=headers)


@contextmanager
def closing(client: StoreClient) -> Iterator[StoreClient]:
    """Close a client at the end of a ``with`` block, however the block ends.

    A context manager rather than a fixture because ``conftest.py`` is not this
    agent's file, and a fixture declared in one would have to live in another.
    The suite treats a leaked connection as an error, and the simplest way to
    satisfy that without owning the harness is to make closing the path of least
    resistance at the point of construction.
    """
    try:
        yield client
    finally:
        client.close()


def documents_for(
    count: int,
    *,
    repo: str = "acme/widget",
    pr: int = 1,
    tags: Sequence[str] = ("agent_authored",),
) -> dict[str, dict[str, Any]]:
    """A corpus of ``count`` documents, for tests about paging rather than content."""
    return {
        document_id(repo, pr, f"answer-{index:04d}"): kojutsu_frontmatter(
            tags=tags, repo=repo, pr=pr
        )
        for index in range(count)
    }
