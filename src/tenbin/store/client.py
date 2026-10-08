"""The read seam: one HTTP client, and the only way into the knowledge store.

Tenbin holds no capability over the store beyond reading it, and this module is
where that capability is expressed -- a handful of read operations, an
``httpx.Client`` it does not own, and no writer of any kind. **The absence of a
writer is a design property, not an omission.** There is no upsert, no delete, no
bulk update, and no method that could become one: a measurement program that can
modify the corpus it measures is measuring a corpus it may have edited, and the
edit is invisible in every figure it later reports. Kojutsu owns the records
and the argument about what they are worth; the seam is one-way, and this class
is the whole of Tenbin's side of it.

The wire contract, verified against Tanseki's source rather than inferred from
Kojutsu's use of it, and deliberately stricter here than a client that only
had to work would be. Base URL is ``{TENBIN_TANSEKI_URL}/v1``; the API key travels in
``X-API-Key``. Document ids are path-derived and contain ``/`` -- a repository is
``owner/name`` -- so every single-document read goes through the body of
``POST /v1/documents:get`` and never through a path segment, because a slash in
a path segment is a request for a different URL. The list endpoint caps ``limit``
at 500 and ``offset`` at 10 000; ``:query`` accepts at most 100 ids. Those are
the store's limits, enforced here rather than discovered as a 400 in production.

**Envelopes are read defensively, and absence is not emptiness.**
:meth:`StoreClient.list_documents` accepts the document array under any of the
four keys Tanseki has shipped and accepts entries that are bare ids or objects
carrying an ``id``; but a response with *no* array under any key is a
:exc:`StoreResponseError`, never an empty list, because "I could not find your
documents" and "you have no documents" are the two sentences a measurement
program must never confuse. The mirror case is :meth:`StoreClient.search`, whose
envelope legitimately carries no ``total`` -- Tanseki's search response is a
pagination envelope and has never had one -- so a missing count is accepted
while a count *smaller than the hits returned* is rejected, because that one is
not a missing field but an incoherent one.

**The graph is read, not rebuilt.** Tanseki derives edges server-side from four
frontmatter keys -- ``repo``, ``pr``, ``jira`` and ``files`` -- and
:meth:`StoreClient.traverse` is how this program asks a document for its
neighbourhood instead of enumerating the whole collection to intersect sets by
hand. **It is a ``POST`` and it is still a read:** the verb is forced by the id,
which is path-derived and contains ``/``, exactly as it is for
:meth:`StoreClient.get_document`, and the read seam's guarantee is about the
capability this class holds rather than about the spelling of the method. The
guarantee is mechanical, not stylistic -- this class has no writer to call --
and :meth:`StoreClient.traverse` adds no field to modify.

**Anomalies are typed by who has to act on them.** A transient failure is
retried a bounded number of times and then raised as
:exc:`StoreUnavailableError`; a rejected credential is
:exc:`StoreAuthenticationError`, which carries
:attr:`~StoreAuthenticationError.operator_action_required` so that a caller can
tell a human-actionable stop from a machine-actionable one, because retrying an
API key does anything but fix it.
"""

from __future__ import annotations

import ipaddress
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final
from urllib.parse import urlparse

import httpx

from tenbin.config import MAX_DOCUMENT_ID_LENGTH, Settings

#: Tanseki's ``GET /v1/documents`` cap. Requesting more is a 400, so the bound is
#: applied to the caller rather than learned from a failure.
MAX_PAGE_LIMIT: Final[int] = 500

#: Tanseki's offset ceiling. It is a real limit, not a policy: past it the store
#: refuses the request, which means a corpus larger than this cannot be walked in
#: one pass by anybody. The snapshot names that outcome rather than reporting a
#: short corpus as a whole one.
MAX_PAGE_OFFSET: Final[int] = 10_000

#: Tanseki's ``POST /v1/documents:query`` cap, and therefore the chunk size for
#: every batch read. Requesting 101 ids is a 400, so the client never does.
MAX_BATCH_IDS: Final[int] = 100

#: A bound of this program's own, not the store's. Tanseki documents no ceiling on
#: search results, and a caller who needs the whole corpus must enumerate it
#: rather than search for it; the cap keeps a mistyped ``limit`` from asking a
#: store for half a million hits in one request.
MAX_SEARCH_LIMIT: Final[int] = 500

#: The API key header, named once. Kojutsu uses the same header against the
#: same store, and a second spelling in a second client would be a request that
#: authenticates as nobody.
API_KEY_HEADER: Final[str] = "X-API-Key"

#: Statuses worth retrying: the request is the same and the store is busy,
#: briefly overloaded, or behind a proxy that gave up. Anything else is either
#: the caller's fault or the store's answer, and repeating it unchanged would
#: only arrive at the same answer.
RETRYABLE_STATUSES: Final[frozenset[int]] = frozenset({408, 425, 429, 500, 502, 503, 504})

#: Statuses meaning the credential was refused. Retried or not, the answer will
#: not change until an operator changes the key.
AUTH_STATUSES: Final[frozenset[int]] = frozenset({401, 403})

#: First backoff step, doubled per attempt. Small, because the only statuses it
#: sleeps through are 429 and 5xx, and a program that hangs for a minute inside a
#: snapshot is harder to reason about than one that fails visibly.
RETRY_BACKOFF_SECONDS: Final[float] = 0.2

#: A ``Retry-After`` longer than this is the store misbehaving or an outage
#: lasting longer than a measurement run; honouring it verbatim would turn a
#: snapshot into a hang.
MAX_RETRY_AFTER_SECONDS: Final[float] = 300.0

#: How many times a transient failure is retried before it is raised. Two is
#: enough to ride out a restart and few enough that a dead store fails inside a
#: test's patience.
DEFAULT_MAX_RETRIES: Final[int] = 2

#: Keys the document array has appeared under. Checked in order; the first list
#: wins, so a response carrying two of them is read by the first and the
#: disagreement is not adjudicated.
LIST_ARRAY_KEYS: Final[tuple[str, ...]] = ("documents", "items", "results", "hits")

#: Keys the traversal's id array has appeared under. ``ids`` is the spelling the
#: store's own seam document names and the rest are the same defensive set the
#: listing uses, for the same reason: an envelope shape that has varied is not a
#: broken document, and a reader that insisted on one spelling would refuse real
#: responses rather than refuse broken ones.
TRAVERSE_ARRAY_KEYS: Final[tuple[str, ...]] = ("ids", "documents", "items", "results")

#: How many hops a traversal takes unless the caller says otherwise. One, and the
#: reason is the shape of the question a reader actually asks: "which changes
#: touched this file" and "which records name this ticket" are both
#: one-hop questions, and a default of two would answer a wider one silently.
DEFAULT_TRAVERSE_DEPTH: Final[int] = 1

#: The ceiling on hops, which is a bound of this program's own rather than the
#: store's. Depth is multiplied into the response on every hop, so an unbounded
#: depth is a request whose size is decided by a caller that has not said what it
#: wants; the bound is applied to the caller rather than discovered as a response
#: nobody can use.
MAX_TRAVERSE_DEPTH: Final[int] = 8

#: A relation is a word the store's edge deriver knows, so this is a generous
#: bound on a short name. It exists because the relation goes into a request that
#: a store logs, and an unbounded caller-supplied string should not become an
#: unbounded log line.
MAX_TRAVERSAL_RELATION_LENGTH: Final[int] = 64


class StoreError(RuntimeError):
    """Base for every failure of the read seam.

    ``RuntimeError`` rather than a bare ``Exception`` because a store failure is
    a failure of the world the program is running in, not a mistake in the code
    calling it -- and because catching this one class is the correct way to say
    "the corpus could not be read", which is a claim a program owes its reader.
    """


class StoreConfigurationError(StoreError):
    """The configured store URL is unsafe or unusable.

    A sibling of the other four rather than a ``ValueError``: refusing a URL is
    this class's own concern, and a caller handling store failures should not
    have to know that configuration problems arrive as a different base class
    than transport ones. The message never contains the URL itself, because the
    URL is validated precisely where a credential may have been embedded in it
    and a message is a thing that gets logged.
    """


class StoreUnavailableError(StoreError):
    """The store could not be reached, or was transiently unable to answer.

    Raised for transport errors and for the retryable status set once retries
    are exhausted. Carries ``retry_after`` when the store supplied one, so a
    caller reporting a snapshot knows how long it said to wait rather than
    guessing.
    """

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class StoreAuthenticationError(StoreError):
    """The store refused the credential, so a human has to change something.

    Not retried. ``operator_action_required`` exists so a caller can separate
    "wait and try again" from "the configuration is wrong", and the only honest
    separation available is a property on the exception.
    """

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after

    @property
    def operator_action_required(self) -> bool:
        """Always true, and a property rather than a field so it cannot be unset."""
        return True


class StoreResponseError(StoreError):
    """A well-formed response that violates the contract.

    Distinct from :exc:`StoreUnavailableError` because the remedy is different:
    the store answered, so waiting will not help, and a retry of an unchanged
    request will arrive at the same broken answer. Treating it as transient is
    how a broken envelope becomes a slow, silent, empty report.
    """


class StoreNotFoundError(StoreError):
    """A single document read addressed an id the store does not hold."""


def _text_or_none(data: Mapping[str, Any], key: str, operation: str) -> str | None:
    """Read an optional string, refusing a value that is present and wrong-typed."""
    if key not in data or data[key] is None:
        return None
    value = data[key]
    if not isinstance(value, str) or not value.strip():
        raise StoreResponseError(f"{operation} returned an invalid {key}")
    return value


def _required_text(data: Mapping[str, Any], key: str, operation: str) -> str:
    value = _text_or_none(data, key, operation)
    if value is None:
        raise StoreResponseError(f"{operation} returned no {key}")
    return value


def _optional_count(data: Mapping[str, Any], key: str, operation: str) -> int | None:
    """Read an optional non-negative count, treating an absent field as legitimate.

    Tanseki's search envelope is ``{hits, limit, offset, hasMore}`` and carries no
    ``total``; a field that is present and is not a non-negative integer is still
    a broken response. Booleans are refused explicitly -- ``bool`` is an ``int``
    and ``True`` would otherwise be a count of one.
    """
    if key not in data or data[key] is None:
        return None
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StoreResponseError(f"{operation} returned an invalid {key}")
    return value


def _optional_bool(data: Mapping[str, Any], key: str, operation: str) -> bool | None:
    """Read an optional boolean, treating an absent field as legitimate."""
    if key not in data or data[key] is None:
        return None
    value = data[key]
    if not isinstance(value, bool):
        raise StoreResponseError(f"{operation} returned an invalid {key}")
    return value


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """Honour ``Retry-After`` as seconds, ignoring anything that is not one."""
    raw = response.headers.get("Retry-After")
    if raw is None:
        return None
    try:
        seconds = float(raw)
    except ValueError:
        return None
    return min(MAX_RETRY_AFTER_SECONDS, max(0.0, seconds))


def _validated_base_url(tanseki_url: str) -> str:
    """Reject a store URL that is unsafe, without ever repeating it.

    HTTP(S) only, no embedded credentials, and HTTPS unless the host is
    loopback -- a local daemon is not a network boundary, and a remote one
    would carry the API key and the corpus in the clear. The refusal names the
    rule and not the value, because a URL is exactly where a key gets pasted and
    this message is written to be logged.
    """
    normalized = tanseki_url.strip().rstrip("/")
    parsed = urlparse(normalized)
    if (
        parsed.scheme.casefold() not in {"http", "https"}
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise StoreConfigurationError("Store URL must be a valid HTTP(S) URL without credentials.")
    if parsed.scheme.casefold() == "http" and not _is_loopback(parsed.hostname):
        raise StoreConfigurationError("Store URL must use HTTPS unless it targets loopback.")
    return normalized


def _is_loopback(host: str) -> bool:
    candidate = host.removeprefix("[").removesuffix("]").casefold()
    if candidate == "localhost":
        return True
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True)
class StoreDocument:
    """A document as the store returns it, before any interpretation.

    A value type, not a model of anything: it holds what the wire held and knows
    nothing about what a record is. The split matters because the interpretation
    is where every assumption lives, and a reader that could not see the wire
    shape would have no way to check the reading against it.

    ``content`` defaults to empty because a batch projection is allowed to omit
    the body. Tenbin measures frontmatter, and a document that arrived without
    its prose is still a countable record; refusing it would lose a document
    over the one field this program does not measure.
    """

    id: str
    collection: str
    path: str = ""
    content: str = ""
    content_hash: str | None = None
    updated_at: str | None = None
    revision: str | None = None
    frontmatter: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Same reasoning as ``Record``: a frozen dataclass holding a dict claims
        # to be immutable and is not. The mapping is copied and wrapped so the
        # document a caller passed in cannot change under a record that quotes
        # it.
        object.__setattr__(self, "frontmatter", MappingProxyType(dict(self.frontmatter)))

    @classmethod
    def from_json(
        cls,
        data: Any,
        *,
        collection: str,
        operation: str,
        expected_id: str | None = None,
    ) -> StoreDocument:
        """Validate one document object, refusing anything that is not one.

        ``contentHash`` and ``updatedAt`` are accepted in either spelling because
        Tanseki's serialisers have shipped both and a reader that insisted on one
        would reject real documents. A *present but wrong-typed* field is still
        refused: the tolerance is for the spelling, not for the type.
        """
        if not isinstance(data, dict):
            raise StoreResponseError(f"{operation} returned an invalid document")
        document_id = _required_text(data, "id", operation)
        if expected_id is not None and document_id != expected_id:
            raise StoreResponseError(f"{operation} returned an unexpected document identity")
        resolved = _text_or_none(data, "collection", operation) or collection
        if resolved != collection:
            raise StoreResponseError(f"{operation} returned an unexpected collection")
        frontmatter = data.get("frontmatter", {})
        if not isinstance(frontmatter, dict):
            raise StoreResponseError(f"{operation} returned an invalid frontmatter")
        content = data.get("content", "")
        if not isinstance(content, str):
            raise StoreResponseError(f"{operation} returned an invalid content")
        path = data.get("path", "")
        if not isinstance(path, str):
            raise StoreResponseError(f"{operation} returned an invalid path")
        return cls(
            id=document_id,
            collection=resolved,
            path=path,
            content=content,
            content_hash=_text_or_none(data, "contentHash", operation)
            or _text_or_none(data, "content_hash", operation),
            updated_at=_text_or_none(data, "updatedAt", operation)
            or _text_or_none(data, "updated_at", operation),
            revision=_text_or_none(data, "revision", operation),
            frontmatter=frontmatter,
        )


@dataclass(frozen=True)
class DocumentPage:
    """One page of a listing, with the store's own account of the collection.

    ``total`` and ``has_more`` are returned rather than consumed because a
    snapshot has to be able to say *how* it stopped, and the store's own count is
    the only independent check on that. A caller that discarded them could only
    know the enumeration ended, not whether it ended honestly.
    """

    ids: tuple[str, ...] = ()
    total: int | None = None
    has_more: bool | None = None


@dataclass(frozen=True)
class StoreHit:
    """A ranked search hit: an id, a score, and an optional snippet."""

    id: str
    score: float
    snippet: str | None = None

    @classmethod
    def from_json(cls, data: Any, *, operation: str) -> StoreHit:
        if not isinstance(data, dict):
            raise StoreResponseError(f"{operation} returned an invalid hit")
        score = data.get("score")
        if isinstance(score, bool) or not isinstance(score, int | float):
            raise StoreResponseError(f"{operation} returned an invalid score")
        if not math.isfinite(float(score)):
            raise StoreResponseError(f"{operation} returned an invalid score")
        snippet = data.get("snippet")
        if snippet is not None and not isinstance(snippet, str):
            raise StoreResponseError(f"{operation} returned an invalid snippet")
        return cls(id=_required_text(data, "id", operation), score=float(score), snippet=snippet)


@dataclass(frozen=True)
class SearchResult:
    """A page of search hits, and whatever the envelope said about the rest.

    ``total`` is ``None`` for the normal case, because Tanseki's search response has
    no ``total`` and demanding one would make the whole search surface
    unrefusable. A ``total`` that *is* present is validated against the hits
    returned, because a total smaller than the page it describes is not a missing
    field but a contradiction.
    """

    hits: tuple[StoreHit, ...] = ()
    total: int | None = None
    has_more: bool | None = None


class StoreClient:
    """A read-only client over the store's ``/v1`` API.

    Takes an ``httpx.Client`` rather than building one alone so a test can hand
    it a ``MockTransport`` and be certain no socket is opened -- which is the
    whole reason the suite's network denial can be trusted. An injected client
    belongs to the caller: :meth:`close` will not close it, because a client that
    closed something it did not open would break the next thing that borrowed it.

    Sleeps are injected for the same reason, so the retry path is exercised
    without spending the backoff in wall-clock time.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        client: httpx.Client | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.base_url = _validated_base_url(settings.tanseki_url)
        self.collection = settings.tanseki_collection
        self.max_retries = max_retries
        self._sleep = sleep
        headers: dict[str, str] = {"Accept": "application/json"}
        if settings.tanseki_api_key:
            headers[API_KEY_HEADER] = settings.tanseki_api_key
        self._headers = headers
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=settings.tanseki_timeout_seconds,
        )

    def __enter__(self) -> StoreClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying connection pool, if this client opened one."""
        if self._owns_client:
            self._client.close()

    # -- transport ----------------------------------------------------------

    def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """Issue one request, retrying only what is worth retrying.

        A transport error and the retryable status set are the same kind of
        event -- the request never got a real answer -- so both are retried on
        the same schedule. Everything else is returned to the caller to
        interpret, because a 404 is an answer and only the operation knows what
        it means there.
        """
        headers = {**self._headers, **(kwargs.pop("headers", None) or {})}
        last_error: Exception | None = None
        retry_after: float | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.request(method, url, headers=headers, **kwargs)
            except httpx.TransportError as exc:
                last_error = exc
            else:
                status = response.status_code
                if status in AUTH_STATUSES:
                    # Raised rather than retried, and the one status set that
                    # leaves this loop early on purpose: the credential is the
                    # same on every attempt, so repeating the request only burns
                    # the budget and delays the report that the configuration is
                    # wrong. Kojutsu's client retries these too, which is one
                    # more reason not to mirror a client whose remit includes
                    # writing and therefore has a different cost for waiting.
                    raise StoreAuthenticationError(
                        f"{method} {url} -> {status}",
                        retry_after=_retry_after_seconds(response),
                    )
                if status in RETRYABLE_STATUSES:
                    retry_after = _retry_after_seconds(response)
                    last_error = StoreUnavailableError(
                        f"{method} {url} -> {status}", retry_after=retry_after
                    )
                else:
                    return response
            if attempt < self.max_retries:
                self._sleep(
                    retry_after if retry_after is not None else RETRY_BACKOFF_SECONDS * (2**attempt)
                )
        if isinstance(last_error, StoreUnavailableError):
            raise last_error
        raise StoreUnavailableError(f"{method} {url} failed: {type(last_error).__name__}")

    def _json(self, response: httpx.Response, operation: str) -> dict[str, Any]:
        """Decode a response body, refusing anything that is not an object."""
        try:
            data = response.json()
        except (TypeError, ValueError) as exc:
            raise StoreResponseError(f"{operation} returned invalid JSON") from exc
        if not isinstance(data, dict):
            raise StoreResponseError(f"{operation} returned an invalid response")
        return data

    def _check_status(self, response: httpx.Response, operation: str) -> None:
        """Refuse an error status the request layer deliberately did not classify.

        :meth:`_request` has already turned the transient and authentication sets
        into exceptions, so a response reaching here is either a success or an
        answer the store meant to give -- a 400 for a request it refused, a 422
        for one it could not accept. Neither is transient, so they become
        :exc:`StoreResponseError` and the operation names itself in the message.
        """
        if not 200 <= response.status_code < 300:
            raise StoreResponseError(
                f"{operation} returned unexpected status {response.status_code}"
            )

    # -- operations ---------------------------------------------------------

    def health(self) -> bool:
        """Whether the store answers its health endpoint.

        A probe, so the failures that mean "not up" are reported as ``False``
        rather than raised: a caller asking whether the store is there does not
        want an exception, and a caller that wants the exception can call any
        other method. A status outside 2xx -- including a 404 from a store that
        does not have a health route -- is also ``False``, because the question
        was "is this reachable and healthy", and a 404 is an answer of no.
        """
        try:
            response = self._request("GET", "/health")
        except StoreError:
            return False
        return 200 <= response.status_code < 300

    def count(self) -> int:
        """How many documents the store says the collection holds.

        Strict where :meth:`list_documents` is lenient: the operation *is* a
        count, so a response carrying no total is a broken count and is refused.
        The leniency exists in the listing for a different reason -- an envelope
        shape that has varied is not a broken document -- and borrowing it here
        would let a store answer a count request with a number of nothing.
        """
        response = self._request(
            "GET", "/documents", params={"limit": 1, "collection": self.collection}
        )
        self._check_status(response, "count")
        data = self._json(response, "count")
        total = _optional_count(data, "total", "count")
        if total is None:
            raise StoreResponseError("count returned no total")
        return total

    def list_documents(self, *, limit: int = MAX_PAGE_LIMIT, offset: int = 0) -> DocumentPage:
        """Read one page of document ids, plus the store's count and paging flag.

        The envelope is read defensively: the array may be under any of
        :data:`LIST_ARRAY_KEYS`, and its entries may be bare ids or objects with
        an ``id``. What is *not* tolerated is a response with no array at all --
        that is :exc:`StoreResponseError`, never an empty page, because a caller
        cannot tell an empty page from a broken one and would report a corpus of
        zero over a store it never successfully read.
        """
        _validate_page(limit=limit, offset=offset)
        response = self._request(
            "GET",
            "/documents",
            params={"limit": limit, "offset": offset, "collection": self.collection},
        )
        self._check_status(response, "list_documents")
        data = self._json(response, "list_documents")
        array = next(
            (data[key] for key in LIST_ARRAY_KEYS if isinstance(data.get(key), list)), None
        )
        if array is None:
            raise StoreResponseError("list_documents returned no document array")
        ids: list[str] = []
        for entry in array:
            if isinstance(entry, str):
                ids.append(entry)
            elif isinstance(entry, dict) and isinstance(entry.get("id"), str):
                ids.append(entry["id"])
        return DocumentPage(
            ids=tuple(ids),
            total=_optional_count(data, "total", "list_documents"),
            has_more=_optional_bool(data, "hasMore", "list_documents"),
        )

    def get_document(self, doc_id: str) -> StoreDocument:
        """Read one document by id.

        The id goes in the **body**. Ids are path-derived and contain ``/`` -- a
        repository is ``owner/name`` -- so a path-addressed read would be a
        request for a different URL, and the store offers a body-addressed
        endpoint precisely so that ids like ``org/repo/pr-42/answer-ab12`` can be
        addressed whole.

        Raises :exc:`StoreNotFoundError` for a 404 rather than returning ``None``:
        a single read has exactly one right answer, and there is no batch-alignment
        reason here to turn an absence into a value. Use :meth:`get_documents`
        where a hole has to stay in place.
        """
        _validate_document_id(doc_id)
        response = self._request(
            "POST", "/documents:get", json={"id": doc_id, "collection": self.collection}
        )
        if response.status_code == 404:
            raise StoreNotFoundError(f"document not found: {doc_id}")
        self._check_status(response, "get_document")
        return StoreDocument.from_json(
            self._json(response, "get_document"),
            collection=self.collection,
            operation="get_document",
            expected_id=doc_id,
        )

    def get_documents(self, ids: Sequence[str]) -> list[StoreDocument | None]:
        """Read many documents, returning a list aligned positionally with ``ids``.

        Ids are chunked to :data:`MAX_BATCH_IDS` because the store refuses more,
        and each chunk is ``POST /v1/documents:query``, whose response **omits**
        ids it does not hold. A ``None`` in the returned list therefore means
        exactly one thing -- the store was asked for that id and did not have it
        -- which is a different fact from never having asked, and a caller that
        conflated the two would report a gap in the corpus as a document that
        was never there. Alignment is the whole point: the result can be zipped
        against the request without a lookup.
        """
        if not ids:
            return []
        results: list[StoreDocument | None] = []
        for start in range(0, len(ids), MAX_BATCH_IDS):
            chunk = list(ids[start : start + MAX_BATCH_IDS])
            fetched = self._query_documents(chunk)
            results.extend(_align(chunk, fetched))
        return results

    def _query_documents(self, chunk: Sequence[str]) -> dict[str, StoreDocument]:
        """Run one ``:query`` chunk and index the documents it returned by id."""
        response = self._request(
            "POST", "/documents:query", json={"ids": list(chunk), "collection": self.collection}
        )
        self._check_status(response, "get_documents")
        data = self._json(response, "get_documents")
        documents = data.get("documents")
        if not isinstance(documents, list):
            raise StoreResponseError("get_documents returned no document array")
        fetched: dict[str, StoreDocument] = {}
        for entry in documents:
            document = StoreDocument.from_json(
                entry, collection=self.collection, operation="get_documents"
            )
            # A store returning the same id twice is not a reason to fail the
            # read; the first one wins so the result is deterministic.
            fetched.setdefault(document.id, document)
        return fetched

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        offset: int = 0,
        tags: Sequence[str] | None = None,
        frontmatter: Mapping[str, str] | None = None,
    ) -> SearchResult:
        """Lexical search across the collection, with tag and frontmatter filters.

        A missing ``total`` is accepted, because Tanseki's search envelope has never
        carried one; a ``total`` smaller than the hits returned is refused,
        because that is not an absent field but a response that contradicts
        itself. Frontmatter filters are sent as repeated ``fm=key=value`` pairs
        rather than as a JSON body, because this is a ``GET`` and the store reads
        them from the query string.
        """
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= MAX_SEARCH_LIMIT
        ):
            raise ValueError(f"Search limit must be an integer between 1 and {MAX_SEARCH_LIMIT}.")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("Search offset must be a non-negative integer.")
        params: list[tuple[str, Any]] = [
            ("q", query),
            ("limit", limit),
            ("offset", offset),
            ("collection", self.collection),
        ]
        params.extend(("tags", tag) for tag in tags or ())
        params.extend(("fm", f"{key}={value}") for key, value in (frontmatter or {}).items())
        response = self._request("GET", "/search", params=params)
        self._check_status(response, "search")
        data = self._json(response, "search")
        raw_hits = data.get("hits")
        if not isinstance(raw_hits, list):
            raise StoreResponseError("search returned an invalid response")
        _optional_count(data, "limit", "search")
        _optional_count(data, "offset", "search")
        has_more = _optional_bool(data, "hasMore", "search")
        total = _optional_count(data, "total", "search")
        hits = tuple(StoreHit.from_json(hit, operation="search") for hit in raw_hits)
        if total is not None and total < len(hits):
            raise StoreResponseError("search returned fewer results than its total")
        return SearchResult(hits=hits[:limit], total=total, has_more=has_more)

    def traverse(
        self, doc_id: str, rel: str, *, depth: int = DEFAULT_TRAVERSE_DEPTH
    ) -> tuple[str, ...]:
        """The ids of the documents this one is related to, over one derived edge.

        The graph is derived by the store, not by this program. Tanseki reads four
        frontmatter keys -- ``repo``, ``pr``, ``jira`` and ``files`` -- and builds
        the edges itself, so a neighbourhood is a question the store can answer in
        one request and this client was answering it by enumerating the whole
        collection and intersecting sets by hand. The cost of the hand-rolled
        version was not only requests: it was quadratic in the corpus for any
        per-file question, and it hid that cost inside a read model rather than
        removing it.

        **A 404 is an empty neighbourhood, and this is the one place the method
        departs from :meth:`get_document`'s raise.** A single-document read has
        exactly one right answer and raises; a neighbourhood question has two
        situations -- the anchor is not in the store, and the anchor is in the
        store with nothing related to it -- and both have the same honest answer,
        which is *there is nothing here*. Raising would push the distinction into
        every caller's control flow and buy nothing, because a reader cannot
        distinguish a document that was never written from one with no neighbours:
        both are documents this program holds no evidence about. What the caller
        does need to distinguish is an empty answer from *never having asked*, and
        that is a fact about the caller rather than about the store, so it belongs
        in :class:`tenbin.corpus.graph.Neighbourhood` and not in an exception.

        Deduplicated, because a traversal can legitimately reach the same document
        along two paths at depth two, and a neighbourhood carrying an id twice
        would double-count the record behind it in whatever figure used it. A
        blank id is skipped for the same reason it is refused in a listing: a
        neighbourhood containing the empty string is a relation to nothing.
        """
        _validate_document_id(doc_id)
        relation = _validate_traversal_relation(rel)
        _validate_traverse_depth(depth)
        response = self._request(
            "POST",
            "/documents:traverse",
            json={
                "id": doc_id,
                "rel": relation,
                "depth": depth,
                "collection": self.collection,
            },
        )
        if response.status_code == 404:
            return ()
        self._check_status(response, "traverse")
        data = self._json(response, "traverse")
        array = next(
            (data[key] for key in TRAVERSE_ARRAY_KEYS if isinstance(data.get(key), list)), None
        )
        if array is None:
            # Never an empty tuple. "I could not find the neighbourhood" and "there
            # is no neighbourhood" are the two sentences a caller cannot tell apart,
            # and answering the first with the second would report a graph the store
            # never described.
            raise StoreResponseError("traverse returned no id array")
        ids: list[str] = []
        for entry in array:
            if isinstance(entry, str):
                candidate = entry
            elif isinstance(entry, dict) and isinstance(entry.get("id"), str):
                candidate = entry["id"]
            else:
                continue
            stripped = candidate.strip()
            if stripped and stripped not in ids:
                ids.append(stripped)
        return tuple(ids)


def _validate_page(*, limit: int, offset: int) -> None:
    """Refuse a page request Tanseki would answer with a 400.

    Bounding rather than clamping: a caller asking for more than the store
    permits has a bug, and silently serving it a smaller page would produce a
    walk that skipped documents and reported itself as complete.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_PAGE_LIMIT:
        raise ValueError(f"Page limit must be an integer between 1 and {MAX_PAGE_LIMIT}.")
    if (
        isinstance(offset, bool)
        or not isinstance(offset, int)
        or not 0 <= offset <= MAX_PAGE_OFFSET
    ):
        raise ValueError(f"Page offset must be an integer between 0 and {MAX_PAGE_OFFSET}.")


def _validate_document_id(doc_id: str) -> None:
    """Refuse an id that is empty or unbounded, without repeating it.

    The bound comes from :mod:`tenbin.config` and exists for the same reason it
    does there: a document id is used in error messages, and an unbounded one
    would put an unbounded string into a log.
    """
    if not isinstance(doc_id, str) or not doc_id.strip():
        raise ValueError("Document id must be a non-empty string.")
    if len(doc_id) > MAX_DOCUMENT_ID_LENGTH:
        raise ValueError(f"Document id exceeds {MAX_DOCUMENT_ID_LENGTH} characters.")


def _validate_traversal_relation(rel: str) -> str:
    """Return the relation to traverse, refusing one that is blank or unbounded.

    Stripped and returned rather than merely checked, so the value that goes on
    the wire is the value that was validated. The blank case is the one worth
    having a message for: an empty relation is not a traversal of everything and
    not a traversal of nothing, it is a request the store has no vocabulary for,
    and every store will answer it with something this program would then report.
    """
    if not isinstance(rel, str):
        raise ValueError("Traversal relation must be a string.")
    candidate = rel.strip()
    if not candidate:
        raise ValueError(
            "Traversal relation must not be blank; a neighbourhood is a neighbourhood of "
            "some named edge, and there is no reading of an empty one."
        )
    if len(candidate) > MAX_TRAVERSAL_RELATION_LENGTH:
        raise ValueError(f"Traversal relation exceeds {MAX_TRAVERSAL_RELATION_LENGTH} characters.")
    return candidate


def _validate_traverse_depth(depth: int) -> None:
    """Refuse a traversal depth outside the range this program will ask for.

    Bounding rather than clamping, for the reason :func:`_validate_page` bounds: a
    caller asking for a depth it did not mean has a bug, and silently serving a
    smaller neighbourhood would answer a question the caller did not ask while
    looking like it had answered it. Zero is refused rather than treated as one,
    because a zero-hop traversal is a request whose answer is the anchor -- and
    returning the anchor would put a document in its own neighbourhood.
    """
    if isinstance(depth, bool) or not isinstance(depth, int):
        raise ValueError("Traversal depth must be an integer.")
    if not 1 <= depth <= MAX_TRAVERSE_DEPTH:
        raise ValueError(
            f"Traversal depth must be between 1 and {MAX_TRAVERSE_DEPTH}; a depth of zero "
            "returns the anchor itself and no neighbourhood."
        )


def _align(
    chunk: Sequence[str], fetched: Mapping[str, StoreDocument]
) -> list[StoreDocument | None]:
    """Put a chunk's answers back in the order they were asked for.

    Keyed by id rather than by position because ``:query`` returns documents in
    whatever order it likes, and by Kojutsu's own experience with rank
    preservation: a batch read whose results are re-ordered by the store is a
    batch read whose results are silently attached to the wrong ids.
    """
    return [fetched.get(doc_id) for doc_id in chunk]
