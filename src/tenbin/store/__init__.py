"""The store side of the read seam: one read-only HTTP client and its failures.

Everything in this package is about getting documents *out* of a knowledge store
that another project owns. It notably does not write, and it does not interpret
what it reads: a :class:`~tenbin.store.client.StoreDocument` is what the wire
held, and turning one into a record is :mod:`tenbin.corpus.record`'s job. That
split is deliberate. The place where a document is checked against the wire
contract is the place with the fewest assumptions in it, and the place where a
record's kind is recovered from a tag set is the place with the most -- and a
reader checking one against the other needs them to be separately visible.

The exception hierarchy is the other thing exported here, and it is part of the
contract rather than an implementation detail. A caller has three questions to
ask of a failure -- could not reach it, could not be let in, or answered with
something incoherent -- and each has a different remedy and a different
consequence for the completeness of a snapshot.
"""

from __future__ import annotations

from tenbin.store.client import (
    API_KEY_HEADER,
    AUTH_STATUSES,
    DEFAULT_MAX_RETRIES,
    MAX_BATCH_IDS,
    MAX_PAGE_LIMIT,
    MAX_PAGE_OFFSET,
    MAX_SEARCH_LIMIT,
    RETRYABLE_STATUSES,
    DocumentPage,
    SearchResult,
    StoreAuthenticationError,
    StoreClient,
    StoreConfigurationError,
    StoreDocument,
    StoreError,
    StoreHit,
    StoreNotFoundError,
    StoreResponseError,
    StoreUnavailableError,
)

__all__ = (
    "API_KEY_HEADER",
    "AUTH_STATUSES",
    "DEFAULT_MAX_RETRIES",
    "MAX_BATCH_IDS",
    "MAX_PAGE_LIMIT",
    "MAX_PAGE_OFFSET",
    "MAX_SEARCH_LIMIT",
    "RETRYABLE_STATUSES",
    "DocumentPage",
    "SearchResult",
    "StoreAuthenticationError",
    "StoreClient",
    "StoreConfigurationError",
    "StoreDocument",
    "StoreError",
    "StoreHit",
    "StoreNotFoundError",
    "StoreResponseError",
    "StoreUnavailableError",
)
