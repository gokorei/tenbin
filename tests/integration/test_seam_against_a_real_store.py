"""The seam, checked against a store rather than against a reading of a store.

**Nothing in this file has been run against a real Tanseki store, and every claim it
tests is therefore asserted but unverified.** That is not a caveat about the
author; it is the state of the fact. Each claim below was learned by reading
Tanseki's source, and reading the source again produces the same unverified answer.
What this suite does is make the *first real run* either pass silently or say
exactly what to change, so the gap closes the moment a store is available instead
of staying invisible for as long as the suite is not run.

The claims, each with the observation that would settle it:

1. **The listing envelope.** The document array arrives under one of
   ``documents``/``items``/``results``/``hits``; the source names ``documents``.
   Settled by: reading one listing body and naming the key it used.
2. **``total`` is present, and is the store's own count of the collection.**
   Settled by: reading one listing body and one ``?limit=1`` body.
3. **A batch of one hundred ids is accepted and one hundred and one is refused.**
   Settled by: asking for each and reading the statuses.
4. **Ids the store does not hold are omitted rather than returned as nulls.**
   Settled by: batch-querying a known-absent id beside known-present ones and
   counting what came back.
5. **The offset ceiling is 10 000, and past it the store refuses rather than
   serving a short page.** Settled by: asking at the ceiling and one past it.
6. **The search envelope carries no ``total``.** Settled by: reading one search
   body.

A seventh, not one of the six: the traversal's id-array key, checked the same
way, because ``StoreClient.traverse`` was added today and rests on exactly the
same kind of citation as the other five.

**A skipped run is not a passing run.** Every other test in this repository runs
against ``FakeStore``, which is the right way to test a client and the only kind
of test there is, and it means the seam has never been checked against the thing
it describes. A suite in that position is one skip away from being
indistinguishable from a verified one, so this file refuses that: with no store
opted in, every test skips with a reason naming the variable that would fix it
and the word UNVERIFIED, and ``TANSEKI_SEAM_STRICT`` turns that skip into a failure
so a CI job that was supposed to have a store cannot go green without one.

**The suite writes, and it says so.** ``StoreClient`` holds no write capability
by design, and the :class:`Writer` below is the only code in this repository that
can modify a corpus -- it lives in this file for that reason. Every document it
writes carries a run-unique id prefix and is deleted in a ``finally``; a failing
run leaves the store as it found it as far as its own writes go, and a session
that ends with documents still present fails rather than warning quietly.

**The suite asserts nothing about any corpus it did not create.** It never
compares a count to the store's own count except where that comparison is itself
the claim, it never asserts a document number, and it reads its own documents by
id rather than by enumeration wherever the store allows it. A test that fails the
moment somebody captures something is a test about the corpus, not about the seam.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Final
from uuid import uuid4

import httpx
import pytest

from tenbin.config import Settings, settings_from_env
from tenbin.corpus.snapshot import Completeness, take_snapshot
from tenbin.store.client import (
    LIST_ARRAY_KEYS,
    MAX_BATCH_IDS,
    MAX_PAGE_LIMIT,
    MAX_PAGE_OFFSET,
    TRAVERSE_ARRAY_KEYS,
    StoreClient,
    StoreResponseError,
)

pytestmark = pytest.mark.integration

#: Marks every recorded observation in a run's output. Greppable, because the
#: point of recording a fact is that somebody can find it afterwards.
#:
#: **Two labels, and the run picks one.** ``SEAM-UNVERIFIED`` is what a recorded
#: fact was originally worth, because every claim in this file was written by
#: reading Tanseki's source and nothing had ever checked them. That is no longer
#: true of a run against a real store: on 2026-10-01 the store at
#: ``127.0.0.1:8099`` was reached and answered every one of them. Printing
#: ``UNVERIFIED`` beside an observed fact would be wrong in the same way as
#: printing a bare number was -- it would tell a reader to discount something
#: that has been checked, and a reader who learns to discount the label stops
#: reading the line after it. So a run that reached a store says ``SEAM-OBSERVED``
#: and a run that did not keeps saying ``UNVERIFIED``.
SEAM_UNVERIFIED: Final[str] = "SEAM-UNVERIFIED"
SEAM_OBSERVED: Final[str] = "SEAM-OBSERVED"

#: What a recorded fact was worth before anything had checked it. Kept for the
#: skip reason, which is about a run that never happened.
SEAM_PREFIX: Final[str] = SEAM_UNVERIFIED

#: The opt-in and its neighbours. Named ``TANSEKI_*`` rather than ``TENBIN_*`` on
#: purpose: this suite writes to and deletes from a store, so a variable that
#: reads as "point Tenbin at a store" is the wrong name for a permission, and an
#: ``TENBIN_`` prefix would additionally collide with the harness test that
#: asserts no Tenbin setting survives into a test.
OPTOUT_ENV: Final[str] = "TANSEKI_SEAM_STORE"
URL_ENV: Final[str] = "TANSEKI_SEAM_URL"
COLLECTION_ENV: Final[str] = "TANSEKI_SEAM_COLLECTION"
API_KEY_ENV: Final[str] = "TANSEKI_SEAM_API_KEY"
STRICT_ENV: Final[str] = "TANSEKI_SEAM_STRICT"

#: Where a locally built Tanseki listens, which is its documented default port.
DEFAULT_SEAM_URL: Final[str] = "http://127.0.0.1:8088/v1"

#: A collection of the suite's own, and not the collection Kojutsu writes. A
#: store shared with a real corpus is a store this suite would be writing test
#: documents into, and the walk in the offset test would be a walk of somebody
#: else's knowledge. Overridable, because a developer with a scratch store
#: already pointed somewhere may want to use it.
DEFAULT_SEAM_COLLECTION: Final[str] = "insight-seam-check"

#: The first path segment of every id this run writes, so a leftover is
#: recognisable as this suite's rather than as a document somebody meant.
ID_NAMESPACE: Final[str] = "tenbin-seam"

#: Uniques the run, so two runs against one store -- or one run whose last
#: cleanup failed -- cannot collide on an id. The timestamp is for a human
#: reading a leftover id; the random suffix is what prevents the collision.
RUN_TOKEN: Final[str] = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
RUN_PREFIX: Final[str] = f"{ID_NAMESPACE}/{RUN_TOKEN}"
PROSE_TOKEN: Final[str] = f"seam-prose-{RUN_TOKEN}"
STAMP: Final[str] = f"{datetime.now(UTC):%Y-%m-%dT%H:%M:%S+00:00}"

#: An id the run asks for and the store has never held. Shaped like a Kojutsu
#: id, because a document written outside that shape is a different question and
#: this claim is about an id that simply is not there.
ABSENT_ID: Final[str] = f"{ID_NAMESPACE}/no-such-change/pr-0/answer-absent"


#: Whether this run actually got an answer out of a store. Set by the store
#: fixture rather than inferred from the opt-in variable, because "the operator
#: set TANSEKI_SEAM_STORE" and "the store answered" are different facts and only the
#: second one makes an observation worth anything. It is a module global rather
#: than a fixture return because :func:`record` is called from inside tests that
#: have no interest in it, and threading a label through each of them would put
#: the bookkeeping in seven more places.
_REACHED: SeamStore | None = None


def store_url() -> str:
    """The store this run reached, for a line that says where a fact came from."""
    store = _REACHED
    return store.url if store is not None else ""


def record(fact: str, observed: str) -> None:
    """Announce one observed wire fact where the run's output will show it.

    A warning rather than a print, because pytest shows a warnings summary by
    default and discards a passing test's stdout. The recorded facts are the
    deliverable of this file; the assertions are a by-product, and a run that
    recorded six numbers and passed is more useful than a run that passed.

    **The label says whether this run reached a store**, and the store's URL is
    part of the line, because an observed fact without a source is a fact nobody
    can go and check.
    """
    label = SEAM_OBSERVED if _REACHED is not None else SEAM_PREFIX
    where = f" at {_REACHED.url}" if _REACHED is not None else ""
    warnings.warn(f"{label}{where} {fact}: {observed}", UserWarning, stacklevel=2)


def no_store_reason(detail: str = "") -> str:
    """The sentence every skip in this file carries.

    Named rather than written seven times because a skip reason is the only thing
    a reader sees when the suite did not run, and seven copies of it would be
    seven places to forget the word that makes it honest.
    """
    reason = (
        "no Tanseki store was opted in, so no wire claim was checked. This suite writes to and "
        "deletes from a real store, so it runs only when TANSEKI_SEAM_STORE is set and "
        f"TANSEKI_SEAM_URL (default {DEFAULT_SEAM_URL}) answers. Every wire claim in "
        "test_seam_against_a_real_store.py is UNVERIFIED without this, and a skipped run is "
        f"not a passing one. Set {STRICT_ENV}=1 to make this a failure instead."
    )
    return f"{reason} {detail}" if detail else reason


# ---------------------------------------------------------------------------
# The wire, as it arrived
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WireResponse:
    """One response as it arrived, kept whole.

    ``body`` is the decoded JSON and ``text`` the bytes it came from, because a
    test comparing the client's reading against the wire needs the wire, and a
    test naming what the store actually sent needs the sentence.
    """

    method: str
    path: str
    status: int
    text: str
    body: Any

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    @property
    def error_code(self) -> str | None:
        """The store's own error code, when it sent the documented envelope."""
        if isinstance(self.body, dict):
            error = self.body.get("error")
            if isinstance(error, dict) and isinstance(error.get("code"), str):
                return error["code"]
        return None

    def array_keys(self) -> tuple[str, ...]:
        """Every top-level key holding a list, in the order the store sent them.

        Deliberately unfiltered. The point of the envelope tests here is to be
        told which key the store used, so narrowing this to the keys the client
        accepts would discard the only interesting part of the answer.
        """
        if not isinstance(self.body, dict):
            return ()
        return tuple(key for key, value in self.body.items() if isinstance(value, list))

    def array_under(self, key: str) -> list[Any]:
        """The list under ``key``, or an empty list when the key is not there."""
        if not isinstance(self.body, dict):
            return []
        value = self.body.get(key)
        return value if isinstance(value, list) else []

    def has(self, key: str) -> bool:
        return isinstance(self.body, dict) and key in self.body

    @property
    def keys(self) -> tuple[str, ...]:
        """The envelope's own field names, for a message that has to list them."""
        return tuple(sorted(self.body)) if isinstance(self.body, dict) else ()

    def describe(self) -> str:
        """A one-line account of what arrived, for a failure message."""
        return (
            f"{self.method} {self.path} -> {self.status} "
            f"(error_code={self.error_code}, keys={list(self.keys)})"
        )


def ids_in(entries: list[Any]) -> tuple[str, ...]:
    """The ids a document array holds, read the way the client reads it.

    Written out here rather than imported so that this suite's reading of a
    response is visibly independent of the client's: a comparison between two
    readers that share a function proves that the function agrees with itself.
    """
    found: list[str] = []
    for entry in entries:
        if isinstance(entry, str):
            found.append(entry)
        elif isinstance(entry, dict) and isinstance(entry.get("id"), str):
            found.append(entry["id"])
    return tuple(found)


class WireLog:
    """Every request this run made and the response it got, in order.

    The recording exists so a test can read the exact bytes the client
    interpreted rather than asking the store the same question twice. Two reads
    of a paginated listing are two reads, and comparing the client's answer with
    a second one compares two moments instead of an interpretation with a
    response.
    """

    def __init__(self) -> None:
        self._entries: list[WireResponse] = []

    def append(self, response: WireResponse) -> None:
        self._entries.append(response)

    def mark(self) -> int:
        """The current length, so a caller can name the response it is about to cause."""
        return len(self._entries)

    def at(self, index: int) -> WireResponse:
        assert index < len(self._entries), (
            f"no response was recorded at index {index}; the run made "
            f"{len(self._entries)} request(s)"
        )
        return self._entries[index]

    def clear(self) -> None:
        """Forget the reachability probe, so the first assertion is about a real read."""
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


class RecordingTransport(httpx.BaseTransport):
    """The real transport, with every response read once and kept.

    Delegating rather than mocking, so the bytes under test are the bytes the
    store sent. :meth:`handle_request` reads the body before handing the
    response on, which is correct only because the client here is used
    non-streaming; it also means the content is cached, so the client reads the
    same bytes rather than a drained stream.
    """

    def __init__(self, inner: httpx.BaseTransport, log: WireLog) -> None:
        self._inner = inner
        self._log = log

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        response = self._inner.handle_request(request)
        response.read()
        try:
            body = response.json()
        except ValueError:
            body = None
        self._log.append(
            WireResponse(
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                text=response.text,
                body=body,
            )
        )
        return response

    def close(self) -> None:
        self._inner.close()


# ---------------------------------------------------------------------------
# Reaching the store
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeamStore:
    """Where the store is and which collection this run may touch."""

    url: str
    collection: str
    api_key: str
    settings: Settings

    @property
    def headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers


def seam_store_from_env() -> SeamStore | None:
    """Build the store configuration, or ``None`` when nothing opted in.

    Read at import time, which is the only moment at which the isolation in
    ``tests/conftest.py`` has not already stripped the environment. The ordering
    is not incidental: it is what lets this suite hold a store configuration
    across its tests without asking to be exempted from the policy that removes
    one, and without adding a variable to ``BEHAVIOR_ENV_VARS`` -- which would
    delete it again before the first test ran.
    """
    if not (os.environ.get(OPTOUT_ENV) or "").strip():
        return None
    url = (os.environ.get(URL_ENV) or DEFAULT_SEAM_URL).strip()
    collection = (os.environ.get(COLLECTION_ENV) or DEFAULT_SEAM_COLLECTION).strip()
    api_key = (os.environ.get(API_KEY_ENV) or "").strip()
    return SeamStore(
        url=url,
        collection=collection,
        api_key=api_key,
        settings=settings_from_env(
            tanseki_url=url,
            tanseki_collection=collection,
            tanseki_api_key=api_key,
        ),
    )


def strict_mode() -> bool:
    """Whether an absent store is a failure rather than a skip."""
    return bool((os.environ.get(STRICT_ENV) or "").strip())


def reach(store: SeamStore, http: httpx.Client) -> str | None:
    """``None`` when the store answered, or a sentence naming why it did not.

    The public health route, and deliberately not ``StoreClient.health``: that
    reports a boolean and swallows the reason, and the reason is the difference
    between "start Tanseki" and "your API key is wrong".
    """
    try:
        response = http.get("/health")
    except httpx.HTTPError as exc:
        return f"{type(exc).__name__}: {exc}"
    if not 200 <= response.status_code < 300:
        return f"GET {store.url}/health -> {response.status_code}"
    return None


@dataclass
class Seam:
    """Everything one session needs: a real client, a writer, and the log."""

    store: SeamStore
    http: httpx.Client
    log: WireLog
    reader: StoreClient
    writer: Writer = field(repr=False)
    #: ``True`` once a test in this session has failed, so a cleanup problem is
    #: reported beside the failure that caused it rather than replacing it.
    session_failed: bool = False

    def raw(self, method: str, path: str, **kwargs: Any) -> WireResponse:
        """Issue one request of the suite's own and return what came back.

        Through the same client, so the request is authenticated exactly as the
        client's own requests are and lands in the same log. This is how a test
        asks the store something ``StoreClient`` will not ask -- a hundred and
        one ids, an offset past the ceiling -- without a second code path for
        building requests.
        """
        mark = self.log.mark()
        self.http.request(method, path, **kwargs)
        return self.log.at(mark)

    def list_documents(self, **params: Any) -> WireResponse:
        return self.raw("GET", "/documents", params={"collection": self.store.collection, **params})


@pytest.fixture(scope="session")
def seam(request: pytest.FixtureRequest) -> Iterator[Seam]:
    """A real client over a real store, or a skip that says the seam is unchecked.

    Session-scoped because reachability is a fact about the store, and one probe
    per run is one honest answer; a probe per test would let a transient blip
    look like a different store each time.
    """
    store = seam_store_from_env()
    if store is None:
        reason = no_store_reason()
        if strict_mode():
            pytest.fail(reason)
        pytest.skip(reason)

    global _REACHED  # noqa: PLW0603 -- one store per run, read by record() from seven tests
    _REACHED = store
    log = WireLog()
    transport = RecordingTransport(httpx.HTTPTransport(), log)
    with httpx.Client(
        base_url=store.url,
        headers=store.headers,
        timeout=store.settings.tanseki_timeout_seconds,
        transport=transport,
    ) as http:
        absent = reach(store, http)
        if absent is not None:
            reason = no_store_reason(f"Tried {store.url} and it did not answer: {absent}")
            if strict_mode():
                pytest.fail(reason)
            pytest.skip(reason)
        log.clear()
        writer = Writer(http, log, store.collection)
        # ``client=http`` because the suite owns the connection. A client that
        # closed a transport it did not open would break the next borrower, and
        # the ``with`` above has to be the thing that closes this one.
        reader = StoreClient(store.settings, client=http)
        session = Seam(
            store=store,
            http=http,
            log=log,
            reader=reader,
            writer=writer,
            session_failed=bool(request.session.testsfailed),
        )
        try:
            yield session
        finally:
            session.session_failed = bool(request.session.testsfailed)
            leftovers = audit_leftovers(session)
            if leftovers:
                message = (
                    f"{len(leftovers)} document(s) this run wrote are still in "
                    f"{store.collection}: {', '.join(leftovers)}. A run that leaves "
                    f"documents behind makes the next run's corpus a different corpus."
                )
                if session.session_failed:
                    warnings.warn(f"{SEAM_PREFIX} {message}", UserWarning, stacklevel=1)
                else:
                    pytest.fail(message)


def audit_leftovers(seam: Seam) -> list[str]:
    """Ids this session wrote that the store still holds.

    Asked as a batch read rather than by enumeration, because the ids are known
    and a known id does not need a walk to find. A store that cannot answer is
    reported as "everything" rather than as nothing: a cleanup audit that passes
    because the check itself failed is the exact defect class this repository
    exists to report.
    """
    written = list(seam.writer.written)
    if not written:
        return []
    try:
        resolved = seam.reader.get_documents(written)
    except Exception as exc:  # noqa: BLE001 - the audit must survive anything
        warnings.warn(
            f"{SEAM_PREFIX} could not audit cleanup ({type(exc).__name__}: {exc}); treating "
            f"all {len(written)} written document(s) as unverified",
            UserWarning,
            stacklevel=1,
        )
        return written
    return [
        doc_id for doc_id, document in zip(written, resolved, strict=True) if document is not None
    ]


# ---------------------------------------------------------------------------
# Writing, and taking it back out again
# ---------------------------------------------------------------------------


class Writer:
    """Upsert and delete: the two calls ``StoreClient`` deliberately cannot make.

    A class rather than two module functions so the capability has an address.
    There is exactly one object in this repository that can modify a corpus, it
    is constructed in this file, and it keeps the list of what it wrote so a
    cleanup audit can ask about the right ids.
    """

    def __init__(self, http: httpx.Client, log: WireLog, collection: str) -> None:
        self._http = http
        self._log = log
        self._collection = collection
        #: Every id this run *asked* the store to write, recorded before the
        #: request rather than after it: a write answered with a 500 may still
        #: have created the document, and a cleanup that skipped it on the
        #: strength of the status code would be a cleanup that hoped.
        self.written: list[str] = []

    def upsert(self, doc_id: str, *, content: str, frontmatter: Mapping[str, Any]) -> WireResponse:
        self.written.append(doc_id)
        return self._call(
            "/documents:upsert",
            {
                "id": doc_id,
                "collection": self._collection,
                "content": content,
                "frontmatter": dict(frontmatter),
            },
        )

    def delete(self, doc_id: str) -> WireResponse:
        return self._call(
            "/documents:delete",
            {"id": doc_id, "collection": self._collection},
        )

    def _call(self, path: str, payload: dict[str, Any]) -> WireResponse:
        mark = self._log.mark()
        self._http.post(path, json=payload)
        return self._log.at(mark)


def document_id(index: int) -> str:
    """An id under this run's prefix, shaped the way Kojutsu derives them.

    Path-derived and carrying a slash, because a test that used a flat id would
    not be exercising the addressing the client exists to get right.
    """
    return f"{RUN_PREFIX}/pr-1/answer-{index:04d}"


def seam_frontmatter(index: int, **overrides: Any) -> dict[str, Any]:
    """Frontmatter of the shape Kojutsu writes.

    ``record_kind`` is stated so a record built from one of these is classified
    from a stated kind rather than inferred, and ``seam_run`` names the run so a
    leftover is identifiable from its content as well as from its id.
    """
    frontmatter: dict[str, Any] = {
        "title": f"Seam check {index}",
        "author": "insight-seam-check",
        "tags": ["agent_authored"],
        "updated_at": STAMP,
        "capture_source": "integration",
        "record_kind": "answer",
        "repo": RUN_PREFIX,
        "pr": "1",
        "seam_run": RUN_TOKEN,
    }
    frontmatter.update(overrides)
    return frontmatter


def seam_content(index: int) -> str:
    """A body carrying a token unique to this run.

    Lowercase letters and hyphens only: the store re-renders content canonically
    on write, so a token carrying Markdown punctuation would come back escaped
    and the assertion would be about the serialiser rather than about the seam.
    """
    return f"# Seam check {index}\n\nProse {PROSE_TOKEN} number {index}.\n"


@contextmanager
def created(seam: Seam, count: int, **frontmatter_overrides: Any) -> Iterator[tuple[str, ...]]:
    """Write ``count`` documents under this run's prefix and delete them after.

    The deletion is in a ``finally`` and it is checked rather than assumed. A
    document that survives is a failure on its own, or a warning beside the
    failure that caused it, and never silence: the failure that caused it is
    re-raised rather than replaced, because the defect under test is the one the
    test was written to find.
    """
    ids = tuple(document_id(index) for index in range(count))
    in_flight: BaseException | None = None
    try:
        for index, doc_id in enumerate(ids):
            response = seam.writer.upsert(
                doc_id,
                content=seam_content(index),
                frontmatter=seam_frontmatter(index, **frontmatter_overrides),
            )
            assert response.ok, (
                f"the store refused to write {doc_id}: {response.describe()}. The write "
                f"endpoints are read from Tanseki's source like every other claim here, so a "
                f"refusal is a contradiction of that reading and the body says why: "
                f"{response.text[:300]}"
            )
        yield ids
    except BaseException as exc:  # noqa: BLE001 - re-raised immediately
        in_flight = exc
        raise
    finally:
        survivors = [
            doc_id for doc_id in seam.writer.written[-len(ids) :] if not _deleted(seam, doc_id)
        ]
        if survivors:
            message = (
                f"{len(survivors)} document(s) this test wrote are still in "
                f"{seam.store.collection}: {', '.join(survivors)}"
            )
            if in_flight is None and not seam.session_failed:
                pytest.fail(f"{message}. Cleanup is part of the contract, not a courtesy.")
            warnings.warn(f"{SEAM_PREFIX} {message}", UserWarning, stacklevel=1)


def _deleted(seam: Seam, doc_id: str) -> bool:
    """Whether one document is gone, treating "was never there" as gone."""
    response = seam.writer.delete(doc_id)
    return response.ok or response.status == 404


# ---------------------------------------------------------------------------
# The claims
# ---------------------------------------------------------------------------


def test_a_document_this_run_wrote_is_read_back_through_the_client(seam: Seam) -> None:
    """The round trip a fake cannot make: bytes in, a ``StoreDocument`` out.

    **Unverified.** Nothing here has been run against a real store, so what is
    being tested is only that the write endpoints exist at the paths the writer
    uses, that a document written as one piece of content comes back as one
    ``StoreDocument`` with its id, collection and frontmatter intact, and that
    the body's prose survives the store's canonical re-render.

    Settled by: running this against a store. What would contradict it is a
    store that returns a document whose id is not the one written, or that drops
    a frontmatter key on the way out.

    Content is deliberately **not** compared for equality. The store re-renders
    canonical Markdown on write, so an equality assertion here would be an
    assertion about a serialiser rather than about the seam. What is asserted is
    that the marker prose is present, which is weaker and more durable.
    """
    with created(seam, 3) as ids:
        mark = seam.log.mark()
        fetched = seam.reader.get_documents(ids)
        answered = seam.log.at(mark)
        assert answered.ok, answered.describe()
        assert all(document is not None for document in fetched), (
            f"a batch of {len(ids)} documents this run had just written came back with holes: "
            f"the read succeeded ({answered.describe()}) and a document was absent anyway"
        )
        for index, (requested, document) in enumerate(zip(ids, fetched, strict=True)):
            assert document is not None
            assert document.id == requested
            assert document.collection == seam.store.collection
            assert document.frontmatter.get("title") == f"Seam check {index}"
            assert document.frontmatter.get("seam_run") == RUN_TOKEN
            assert PROSE_TOKEN in document.content
        single = seam.reader.get_document(ids[0])
        assert single.id == ids[0]
        record(
            "round trip",
            f"3 documents written and read back; the response carried "
            f"{len(ids_in(answered.array_under('documents')))} id(s) under "
            f"{list(answered.array_keys())}",
        )


def test_the_listing_sends_its_array_under_a_key_this_client_accepts(seam: Seam) -> None:
    """Claim 1: the listing envelope's array key, recorded and checked.

    **Unverified.** ``LIST_ARRAY_KEYS`` is Kojutsu's defensive set, read from
    Tanseki's source, and the source names ``documents``. If the store sends a
    fourth key then the client works today by accident, and every other test in
    this repository is passing against a fake that was told which key to send.

    Settled by: this test. It reads the raw listing body itself, without asking
    the client, and reports the key the store used. When the key is one the
    client accepts the test passes and records the fact; when it is not, the
    failure message names the key the store actually sent, because a test that
    merely said "no document array" would send the next reader back to the source
    to find out which key it was.

    **The key is settled before the client is asked, deliberately.** A store
    sending a key the client refuses makes ``list_documents`` raise, and a test
    that called the client first would report the client's own
    ``StoreResponseError`` and never get to say which key caused it. The client's
    reading is then compared with this test's reading of *the response the client
    itself received*, which is why the transport records rather than the store
    being asked twice.
    """
    with created(seam, 2):
        probe = seam.list_documents(limit=MAX_PAGE_LIMIT, offset=0)
        assert probe.ok, probe.describe()
        keys = probe.array_keys()
        record("claim 1, listing array key", f"{list(keys)} (accepted: {list(LIST_ARRAY_KEYS)})")
        accepted = [key for key in keys if key in LIST_ARRAY_KEYS]
        assert accepted, (
            f"the store sent the listing's array under {list(keys)}, and none of them is in "
            f"tenbin.store.client.LIST_ARRAY_KEYS ({list(LIST_ARRAY_KEYS)}). The client reads "
            f"listings by looking for the first accepted key, so it refuses this envelope as "
            f"having no document array at all: the listing becomes a StoreResponseError and "
            f"every walk over it becomes a store error. Add the key the store sent to "
            f"LIST_ARRAY_KEYS, or fix the store. The body was: {probe.text[:300]}"
        )
        key = accepted[0]
        mark = seam.log.mark()
        page = seam.reader.list_documents(limit=MAX_PAGE_LIMIT, offset=0)
        listing = seam.log.at(mark)
        assert listing.ok, listing.describe()
        entries = listing.array_under(key)
        assert page.ids == ids_in(entries), (
            f"the client read {len(page.ids)} ids from a response whose {key!r} array holds "
            f"{len(entries)} entries; the two must be the same read of the same bytes"
        )
        assert page.total == (listing.body.get("total") if listing.has("total") else None)


def test_the_store_sends_a_total_and_the_two_ways_of_asking_agree(seam: Seam) -> None:
    """Claim 2: ``total`` is present, and it is the store's own count.

    **Unverified.** The walk in ``tenbin.corpus.snapshot`` refuses to proceed
    without a ``total`` and compares it against what it enumerated, so a missing
    one raises and an inconsistent one is reported as
    ``Completeness.STORE_TOTAL_EXCEEDED``. Every unit test of that behaviour runs
    against a ``total`` the fake invented.

    Settled by: this test, which reads one listing body and one ``?limit=1`` body.
    What would contradict the claim is a listing with no ``total`` at all, or two
    endpoints giving two different answers to the same question.

    No document is written here, deliberately. The two reads are compared with
    each other, and writing documents immediately before asking how many
    documents there are would be a way of manufacturing the disagreement this
    test exists to detect.
    """
    listing = seam.list_documents(limit=1, offset=0)
    assert listing.ok, listing.describe()
    record(
        "claim 2, total present",
        f"keys={list(listing.keys)}; total={'present' if listing.has('total') else 'ABSENT'}",
    )
    assert listing.has("total"), (
        "the listing envelope carried no 'total', so take_snapshot would raise rather than "
        "walk, and no unit test can catch that because FakeStore is configured to send one. "
        f"The keys the store did send: {list(listing.keys)}. Body: {listing.text[:300]}"
    )
    mark = seam.log.mark()
    counted = seam.reader.count()
    count_response = seam.log.at(mark)
    assert count_response.ok, count_response.describe()
    assert count_response.has("total"), (
        "the count endpoint answered without a total, which StoreClient.count refuses as a "
        "broken count; a store that did that would make every snapshot raise"
    )
    listed_total = listing.body["total"]
    record("claim 2, the two counts", f"listing total={listed_total}, count()={counted}")
    assert counted == listed_total, (
        f"the store gave two answers to 'how many documents does this collection hold': the "
        f"listing says {listed_total} and ?limit=1 says {counted}. A count that disagrees "
        f"with itself is not a count of anything, and STORE_TOTAL_EXCEEDED would report the "
        f"disagreement as a finding about the corpus."
    )


def test_a_hundred_ids_are_accepted_and_a_hundred_and_one_are_refused(seam: Seam) -> None:
    """Claim 3: the batch bound of 100, and both sides of it.

    **Unverified.** ``MAX_BATCH_IDS`` was read from Tanseki's source. The client
    chunks to 100 so it never discovers the bound as a 400, which means the bound
    is currently a citation with no consequence attached: a store that accepted
    ten thousand would be reported correctly and this test would not know.

    Settled by: this test, which writes a hundred documents, asks for all a
    hundred, and then asks for a hundred and one. What would contradict the claim
    is a 100-id request that fails, or a 101-id request that succeeds, and the
    failure message names the status in both directions so nobody has to
    reproduce it by hand.

    The hundred documents are the expensive part of this suite, and that is the
    cost of checking a bound at its boundary rather than near it.
    """
    with created(seam, MAX_BATCH_IDS) as ids:
        accepted = seam.raw(
            "POST",
            "/documents:query",
            json={"ids": list(ids), "collection": seam.store.collection},
        )
        assert accepted.ok, (
            f"a batch of exactly {MAX_BATCH_IDS} ids was refused ({accepted.describe()}). "
            f"MAX_BATCH_IDS is read from Tanseki's source and every batch read in this program "
            f"is chunked to it, so a store that refused {MAX_BATCH_IDS} would break every "
            f"walk and no unit test would show it."
        )
        returned = set(ids_in(accepted.array_under("documents")))
        assert returned == set(ids), (
            f"a batch of {MAX_BATCH_IDS} existing ids returned {len(returned)} documents; "
            f"missing: {sorted(set(ids) - returned)[:5]}"
        )
        refused = seam.raw(
            "POST",
            "/documents:query",
            json={"ids": [*ids, ABSENT_ID], "collection": seam.store.collection},
        )
        record(
            "claim 3, the batch bound",
            f"{MAX_BATCH_IDS} ids -> {accepted.status}; {MAX_BATCH_IDS + 1} ids -> "
            f"{refused.status} (error_code={refused.error_code})",
        )
        assert not refused.ok, (
            f"the store accepted {MAX_BATCH_IDS + 1} ids ({refused.describe()}). "
            f"tenbin.store.client.MAX_BATCH_IDS is {MAX_BATCH_IDS} and was read from Tanseki's "
            f"source; the client will keep chunking at 100, which is wasteful rather than "
            f"wrong, but the constant and the citation behind it are now both false."
        )
        mark = seam.log.mark()
        aligned = seam.reader.get_documents(ids)
        client_response = seam.log.at(mark)
        assert client_response.ok, client_response.describe()
        assert all(document is not None for document in aligned), (
            "a full batch of ids the store holds came back with holes in it, which is the "
            "alignment the whole method exists to provide"
        )


def test_an_id_the_store_does_not_hold_is_omitted_rather_than_returned_as_null(
    seam: Seam,
) -> None:
    """Claim 4: absent ids are omitted, and the client's holes land in place.

    **Unverified.** ``get_documents`` returns a positionally aligned list in which
    a ``None`` means "asked for and not held", and ``take_snapshot`` counts those
    holes and downgrades the whole read to ``Completeness.STORE_ERROR`` when it
    finds any. If the store instead returned nulls in the array,
    ``StoreDocument.from_json`` would refuse them as invalid documents and the
    failure would read as a broken envelope rather than as a hole in the corpus.

    Settled by: this test, which asks for two ids the store holds beside one it
    has never held and counts what came back. What would contradict the claim is
    an array of three entries with a null in it, or an array of two entries
    carrying a placeholder for the absent id.

    **What this test cannot do** is provoke the *downgrade*, because that needs a
    document to be listed and then to vanish before it is fetched -- a race no
    single writer can stage against a real store. That half of the claim is still
    tested only against ``FakeStore``, and closing it needs a store that can
    withhold a document on demand.
    """
    with created(seam, 2) as ids:
        asked = [ids[0], ABSENT_ID, ids[1]]
        raw = seam.raw(
            "POST", "/documents:query", json={"ids": asked, "collection": seam.store.collection}
        )
        assert raw.ok, raw.describe()
        entries = raw.array_under("documents")
        record(
            "claim 4, omitted ids",
            f"asked for {len(asked)} ids, received {len(entries)} entries under "
            f"{list(raw.array_keys())}",
        )
        assert all(entry is not None for entry in entries), (
            "the store returned a null where a document was absent, which is the other shape "
            "this claim rules out: a null is an entry the client refuses as an invalid "
            f"document rather than a hole it can count. It sent {len(entries)} entries for "
            f"{len(asked)} ids, one of them null. Body: {raw.text[:300]}"
        )
        assert len(entries) == 2, (
            f"asked for {len(asked)} ids and got {len(entries)} entries "
            f"({raw.describe()}); the claim is that the absent id is omitted, so two is the "
            f"expected length and three would mean the store padded the array with something "
            f"rather than leaving a hole. Body: {raw.text[:300]}"
        )
        assert set(ids_in(entries)) == set(ids), (
            f"the store returned {sorted(ids_in(entries))}, which is not the two ids it was "
            f"asked for: the absent id was echoed back, or another id came in its place"
        )
        resolved = seam.reader.get_documents(asked)
        assert resolved[0] is not None
        assert resolved[1] is None, (
            "the middle position must be the hole, because take_snapshot counts holes by "
            "position and a hole in the wrong place is a document attributed to the wrong id"
        )
        assert resolved[2] is not None


def test_the_offset_ceiling_is_a_claim_about_the_store_and_nothing_has_checked_it(
    seam: Seam,
) -> None:
    """Claim 5: what the store does at the ceiling, and what the walk does next.

    **Unverified.** ``MAX_PAGE_OFFSET`` is 10 000, read from Tanseki's source, and
    two different claims are being carried by one constant: that 10 000 is the
    largest offset the store will serve, and that past it the store *refuses*
    rather than serving a short page. The second is the one the snapshot depends
    on. A refusal is a hole with a status code attached, which
    ``Completeness.OFFSET_CAP`` reports; a short page is a walk that ends quietly
    and reports itself complete over a corpus it did not finish.

    Settled by: this test, which asks at the ceiling and one past it, records
    both statuses, and then checks the walk's own account of itself.

    The walk enumerates the configured collection, so point this suite at a
    collection of its own: otherwise this test is a walk of somebody else's
    corpus, which is slower than it should be and is a claim this file does not
    make.
    """
    with created(seam, 1):
        at_ceiling = seam.list_documents(limit=1, offset=MAX_PAGE_OFFSET)
        past_ceiling = seam.list_documents(limit=1, offset=MAX_PAGE_OFFSET + 1)
        record(
            "claim 5, the offset ceiling",
            f"offset={MAX_PAGE_OFFSET} -> {at_ceiling.status} "
            f"(error_code={at_ceiling.error_code}); offset={MAX_PAGE_OFFSET + 1} -> "
            f"{past_ceiling.status} (error_code={past_ceiling.error_code})",
        )
        assert not past_ceiling.ok, (
            f"the store answered offset={MAX_PAGE_OFFSET + 1} with {past_ceiling.status} "
            f"rather than refusing it ({past_ceiling.describe()}). Either the ceiling is "
            f"higher than tenbin.store.client.MAX_PAGE_OFFSET ({MAX_PAGE_OFFSET}) or the "
            f"store clamps the offset and serves a page, and a clamped page is "
            f"indistinguishable from the end of the collection. That is the silent "
            f"truncation this constant exists to make visible."
        )
        mark = seam.log.mark()
        if at_ceiling.ok:
            page = seam.reader.list_documents(limit=1, offset=MAX_PAGE_OFFSET)
            answer = seam.log.at(mark)
            assert answer.ok, answer.describe()
            assert answer.array_keys(), (
                f"the ceiling page carried no array at all ({answer.describe()}), so there is "
                f"nothing for the client's reading to be compared with"
            )
            assert page.ids == ids_in(answer.array_under(answer.array_keys()[0])), (
                "the client's reading of the ceiling page is not this test's reading of the "
                "same bytes"
            )
        else:
            with pytest.raises(StoreResponseError):
                seam.reader.list_documents(limit=1, offset=MAX_PAGE_OFFSET)
        snapshot = take_snapshot(seam.reader, page_limit=MAX_PAGE_LIMIT)
        record(
            "claim 5, the walk",
            f"completeness={snapshot.completeness}, enumerated={snapshot.enumerated}, "
            f"store_total={snapshot.store_total}, truncation={snapshot.truncation}",
        )
        assert snapshot.completeness is not Completeness.COMPLETE or (
            snapshot.enumerated == snapshot.store_total
        ), (
            f"the walk reported itself complete over {snapshot.enumerated} documents while "
            f"the store says the collection holds {snapshot.store_total}. Completeness is a "
            f"claim about a count, and this is the claim every figure in the program rests "
            f"on."
        )


def test_a_search_records_whether_the_store_sends_a_total(seam: Seam) -> None:
    """Claim 6: the search envelope's missing ``total``, in the direction that matters.

    **Unverified.** ``StoreClient.search`` is documented as tolerating an absent
    ``total`` because Tanseki's search response has never carried one, and it
    rejects a ``total`` smaller than the page it describes. If the store has begun
    sending one, the tolerance is a claim about a world that no longer exists and
    the "has never had one" sentence in the module docstring is a historical note
    wearing a present tense.

    Settled by: this test, which reads the raw search body and then checks the
    client's reading of that same body. The assertion is written in the direction
    the observation points, so it is correct whether the store sends a ``total``
    or not -- and the recorded line says which of the two tolerances this run
    actually covered, because "the test passed" would not tell a reader that.

    A hit is not asserted. A lexical search that finds nothing is a legitimate
    answer, and the interesting part of this envelope is its shape rather than
    whether a token this run invented happened to be indexed.
    """
    params: list[tuple[str, Any]] = [
        ("q", PROSE_TOKEN),
        ("limit", 10),
        ("offset", 0),
        ("collection", seam.store.collection),
    ]
    raw = seam.raw("GET", "/search", params=params)
    assert raw.ok, raw.describe()
    sent_total = raw.has("total")
    hits = raw.array_under("hits")
    record(
        "claim 6, search total",
        f"{'present' if sent_total else 'ABSENT'}; keys={list(raw.keys)}; hits={len(hits)}",
    )
    result = _search_or_explain(seam, raw, hits)
    if sent_total:
        warnings.warn(
            f"{SEAM_PREFIX} the search envelope now carries a total, so the sentence in "
            f"tenbin.store.client saying it 'has never had one' is stale and the "
            f"tolerance-of-absence this test just exercised is no longer the case that "
            f"occurs. Read the docstring before trusting the client on a search total.",
            UserWarning,
            stacklevel=1,
        )
        assert result.total == raw.body["total"], (
            "the store sent a total and the client reported a different one, which is one "
            "response read two ways"
        )
    else:
        assert result.total is None, (
            "the store sent no total and the client reported one, so the client's reading of "
            "the envelope is not this test's reading of the same bytes"
        )


def _search_or_explain(seam: Seam, raw: WireResponse, hits: list[Any]) -> Any:
    """Run the client's search, turning a refusal into a sentence about the envelope.

    A store whose ``total`` contradicts its own hit count makes
    ``StoreClient.search`` raise, and the raise is correct -- but it reaches the
    reader as ``search returned fewer results than its total`` with no hint that
    the interesting fact is the envelope rather than the query. The observation
    is restated here so the failure says what the store sent.
    """
    try:
        return seam.reader.search(PROSE_TOKEN, limit=10)
    except StoreResponseError as exc:
        pytest.fail(
            f"StoreClient.search refused the envelope the store sent ({exc}). The store's "
            f"search body was {raw.describe()} carrying {len(hits)} hit(s) and "
            f"total={raw.body.get('total') if raw.has('total') else 'absent'}. If the total is "
            f"smaller than the page it describes then the client's refusal is right and the "
            f"store's envelope has changed; if the total is absent then the client's reading "
            f"of the envelope is wrong. Body: {raw.text[:300]}"
        )


def test_a_traversal_returns_its_ids_under_a_key_this_client_accepts(seam: Seam) -> None:
    """A seventh claim, checked the same way: the traversal's id-array key.

    Not one of the six, and here because ``StoreClient.traverse`` was added today
    and rests on exactly the same kind of citation as the other five:
    ``TRAVERSE_ARRAY_KEYS`` is a defensive set read from Tanseki's source, and the
    source names ``ids``.

    **Unverified.** Settled by this test. What would contradict the claim is a
    successful traversal whose array arrived under a key the client does not
    accept, which the client reports as "no id array" -- the same accident as
    claim 1, on a method that has never been run against anything.

    A refused request is recorded and not failed on, because the relation name is
    itself unverified: the source derives a ``references`` edge from ``repo``,
    ``pr`` and ``jira``, and a store answering 400 for that relation has not
    contradicted the envelope claim. It has contradicted a different one, and
    skipping with that sentence is the honest reading.
    """
    repo = f"{ID_NAMESPACE}/{RUN_TOKEN}"
    with created(seam, 2, repo=repo, pr="7") as ids:
        raw = seam.raw(
            "POST",
            "/documents:traverse",
            json={
                "id": ids[0],
                "rel": "references",
                "depth": 1,
                "collection": seam.store.collection,
            },
        )
        record(
            "claim 7, traversal array key",
            f"{raw.describe()}; keys={list(raw.keys)}",
        )
        if not raw.ok:
            pytest.skip(
                f"the store refused the traversal ({raw.describe()}, "
                f"error_code={raw.error_code}), so the envelope's array key was not observed "
                f"and claim 7 stays UNVERIFIED. The relation name this test used is itself "
                f"read from source. Body: {raw.text[:200]}"
            )
        accepted = [key for key in raw.array_keys() if key in TRAVERSE_ARRAY_KEYS]
        assert accepted, (
            f"the traversal's id array arrived under {list(raw.array_keys())}, and none of "
            f"them is in tenbin.store.client.TRAVERSE_ARRAY_KEYS ({list(TRAVERSE_ARRAY_KEYS)}). "
            f"The client would raise 'traverse returned no id array' on this response, which "
            f"is a graph this program would report as one the store never described."
        )
        neighbourhood = seam.reader.traverse(ids[0], "references")
        unexpected = [doc_id for doc_id in neighbourhood if doc_id not in {*ids, ABSENT_ID}]
        assert not unexpected, (
            f"a traversal returned ids this run never wrote: {unexpected[:5]}. Either a "
            f"different corpus is leaking into the neighbourhood or the store derived the "
            f"edge from something other than what this run wrote."
        )
        record(
            "claim 7, the neighbourhood",
            f"{len(neighbourhood)} related id(s) at depth 1; an empty one is legitimate",
        )


def test_every_document_this_run_touches_is_named_after_the_run(seam: Seam) -> None:
    """The two properties the other tests are allowed to rely on.

    Criterion-shaped rather than wire-shaped, and deliberately the only test here
    that asserts nothing about a response. It exists so that "this suite writes
    to a collection of its own and deletes what it writes" is a checked property
    rather than a promise in a docstring. It fails if the suite is pointed at a
    collection nobody named, or if an id is ever built without the run's prefix
    -- which is what makes the cleanup audit, and the rule that no assertion may
    depend on a corpus this run did not create, enforceable rather than
    aspirational.
    """
    named = (os.environ.get(COLLECTION_ENV) or "").strip()
    assert seam.store.collection == DEFAULT_SEAM_COLLECTION or named, (
        f"the suite is writing to {seam.store.collection!r}, which is neither its own "
        f"collection nor one {COLLECTION_ENV} named explicitly"
    )
    with created(seam, 2) as ids:
        for doc_id in ids:
            assert doc_id.startswith(f"{RUN_PREFIX}/"), (
                f"{doc_id!r} does not carry this run's prefix, so nothing could tell a "
                f"leftover from a document somebody meant"
            )
        resolved = seam.reader.get_documents(ids)
        assert [document is not None for document in resolved] == [True, True]
