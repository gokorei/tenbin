"""Reading each configured store once, and saying so when a store cannot be read.

**This module extends :func:`tenbin.cli._walk`'s failure mapping; it does not invent a
second one.** The command line has one door to the store and one exit point for every
failure through it, so a store that cannot be reached exits ``3`` with nothing written.
That is the right shape for a command, and it has no answer at all for a list of five
stores rendered for a browser: there is nothing to exit with, and refusing the whole page
because one of five stores is down would answer a question nobody asked. So the mapping
is reused case for case -- four typed store failures, each with its own cause and its own
remedy, in the order an operator would want to hear them -- and the only thing that
changes is what happens at the end of it. A command exits; a page renders.

**What does not change is that a failure is not a zero.** :func:`read_store` has exactly
two return values and one of them is not a report of nothing. An unreachable store is
never an empty :class:`~tenbin.corpus.snapshot.CorpusSnapshot`, never an omitted entry,
and never a row of zeroes: it is a :class:`StoreFailure` naming the store, and the
index renders that in the slot the store's report would have occupied. A list of N
stores is the shape that hides a failure -- one row missing reads as a store with nothing
to say -- so :func:`read_all` returns one outcome per configured entry, in configuration
order, and the type makes the second state unrepresentable rather than leaving it to a
loop that has to remember to check.

**The last case is not a store failure at all, and it is caught anyway.**
:func:`read_store` catches ``Exception`` after the typed cases, because the alternative
is that one store raising something nobody anticipated takes down the index and hides
the four stores that were fine -- which is the failure this module exists to prevent,
reached by letting an unexpected exception do it for you. It is reported as a failure
whose remedy says plainly that it is a defect in Tenbin, because that is what it is. It
is caught rather than wrapped silently, and it is never rendered as figures.

**The credential is redacted on the way in, not on the way out.** Every string that
becomes a cause passes through :meth:`~tenbin.webapp.config.StoreEntry.redact` here, at
the point where it is built from an exception, so that a page is not one refactor away
from publishing a key that a transport library decided to quote.

**What this module notably does not do:** it does not render anything, does not cache,
and does not retry beyond what :class:`~tenbin.store.client.StoreClient` already does.
There is deliberately no read model here and no memoisation of a snapshot between
requests: :mod:`tenbin.readmodel` exists precisely to refuse the second, and a summary
that were computed once and served afterwards would be describing a corpus from whenever
that read happened with no way for a reader to tell which they were looking at. Each
request re-reads, which is also why a per-store page can state a completeness that is
true as of now.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final, TypeAlias

from pydantic import ValidationError

from tenbin.claims.gate import ClaimGate
from tenbin.claims.registry import default_registry
from tenbin.config import Settings
from tenbin.corpus.snapshot import CorpusSnapshot, take_snapshot
from tenbin.measures import default_measures
from tenbin.report.build import build_report
from tenbin.report.document import CorpusHeader, Report
from tenbin.store.client import (
    StoreAuthenticationError,
    StoreClient,
    StoreConfigurationError,
    StoreError,
    StoreUnavailableError,
)
from tenbin.webapp.config import StoreEntry, first_line


@dataclass(frozen=True)
class StoreReport:
    """A store that was read, and everything this program will say about it.

    ``report`` rather than ``snapshot``, because what a page needs is the gated,
    ordered document: the figures with their refusals in place of numbers. The snapshot
    is still reachable, through :attr:`header`, for the one summary the index needs, and
    the header is copied from the snapshot rather than reconstructed -- see
    :meth:`~tenbin.report.document.CorpusHeader.from_snapshot` for why that matters.
    """

    entry: StoreEntry
    report: Report

    @property
    def header(self) -> CorpusHeader:
        """The read this report is about, taken from the report rather than rebuilt."""
        return self.report.corpus


@dataclass(frozen=True)
class StoreFailure:
    """A store that could not be read, named, with what to do about it.

    Two sentences and no figures, and that is the whole type. ``cause`` is what happened;
    ``remedy`` is what would change it; a failure with no remedy would leave an operator
    reading a page at seven in the morning with nothing to do next, and one with figures
    would be the dangerous output this program must never produce.

    Every failure carries its :class:`~tenbin.webapp.config.StoreEntry`, so a renderer
    cannot name one store's failure under another store's heading. That is a mistake a
    ``Failure(entry, cause)`` free function would allow and this cannot.
    """

    entry: StoreEntry
    cause: str
    remedy: str


#: The two outcomes, and only the two. A page function takes this, so a store either has
#: a report or has a failure and there is no path through the renderer that produces a
#: store with neither.
StoreOutcome: TypeAlias = StoreReport | StoreFailure

#: How this module reaches the store. The same seam, and for the same reason, as
#: ``tenbin.cli.CLIENT_FACTORY``: a module-level name rather than a constructor call in
#: the body is what lets a test hand it a store which answers in process, and that is the
#: reason every test of the web app runs with the suite's socket denial still in force.
#: Not ``Final`` for the same reason it is not ``Final`` there -- the production value is
#: fixed and the test value is not.
CLIENT_FACTORY: Callable[[Settings], StoreClient] = StoreClient


@dataclass(frozen=True)
class StoreListing:
    """A store that answered, its size, and no report at all.

    **This is the cheap read, and the reason it exists is arithmetic.** A report page is
    one corpus walk; an index page over *N* stores was *N* walks, serially, plus 22
    measures each, to render a list of names. The cost is linear in document count and
    multiplied by store count and by index loads, so a dashboard embedding the index and
    auto-refreshing pays it repeatedly. What the index actually needs is a count and a
    state, and :meth:`~tenbin.store.client.StoreClient.count` answers both in one request
    without enumerating anything.

    ``count`` is the **store's own** figure rather than a number of documents this
    program happened to walk, because a bounded read presenting a partial enumeration as
    a collection's size is the bounding error this project exists to refuse. It is the
    store's answer, and the page says so in the words that render it.
    """

    entry: StoreEntry
    count: int


#: What a listing read produces: a count, or the failure that stopped it. The same two
#: shapes as :data:`StoreOutcome` with :class:`StoreReport` replaced by
#: :class:`StoreListing`, and deliberately the same *arity* -- one outcome per configured
#: store, always, so the totality :func:`read_all` guarantees carries over unchanged.
StoreListingOutcome: TypeAlias = StoreListing | StoreFailure


def read_listing(entry: StoreEntry) -> StoreListingOutcome:
    """Ask one store how many documents it holds, or return the failure that stopped it.

    **Total by construction**, for the same reason and with the same guarantee as
    :func:`read_store`: a store that cannot answer becomes a :class:`StoreFailure` on the
    way out rather than an exception a caller has to handle. An index that turned an
    unreachable store into a silently shorter list would be the failure being hidden by
    the optimisation that was supposed to make the page cheap.

    Reuses :func:`read_store`'s failure sentences by catching the same exceptions, because
    a listing read and a report read fail for the same reasons and an operator reading a
    7am page should not be told two different things about one broken store.
    """
    try:
        with CLIENT_FACTORY(entry.settings) as client:
            return StoreListing(entry=entry, count=client.count())
    except Exception as exc:  # noqa: BLE001 - every failure becomes a StoreFailure
        return _failure_from(entry, exc, doing="counting this store")


def read_listings(entries: Sequence[StoreEntry]) -> tuple[StoreListingOutcome, ...]:
    """One listing outcome per configured store, in configuration order, always.

    The same list-length guarantee as :func:`read_all`, and for the same reason: every
    entry produces exactly one outcome with none dropped or doubled, so a caller cannot
    render four rows for five stores.

    Serial rather than concurrent, matching :func:`read_all`. A listing read is one round
    trip instead of a walk plus 22 measures, so the serial cost that was worth paying for
    a full report is no longer the thing being protected -- but the failure mapping is
    still the valuable part, and it is still not worth a concurrency primitive.
    """
    return tuple(read_listing(entry) for entry in entries)


def read_all(entries: Sequence[StoreEntry]) -> tuple[StoreOutcome, ...]:
    """One outcome per configured store, in configuration order, always.

    **This is the guarantee, and it is a list length rather than a promise.** Every
    entry produces exactly one of the two outcomes, in the order the operator wrote them,
    with no entry dropped and no entry doubled -- so a caller that renders
    ``read_all(entries)`` cannot accidentally render four rows for five stores, which is
    the one way a list view fails quietly.

    Stores are read one after another rather than concurrently. That is a real cost on a
    page of five stores with a slow one among them, and it is paid deliberately: the
    failure mapping that follows it is the most valuable thing in this app, and a
    concurrency primitive here would be a place to get it wrong rather than a saving.
    """
    return tuple(read_store(entry) for entry in entries)


def read_store(entry: StoreEntry) -> StoreOutcome:
    """Read one store and build its report, or return the failure that stopped it.

    **Total by construction.** There is no ``StoreError`` a caller has to handle, because
    every store failure has become a :class:`StoreFailure` on the way out of here, and the
    two failure branches that used to end a command are the two branches that used to
    write nothing.
    """
    try:
        snapshot = _walk(entry)
    except Exception as exc:  # noqa: BLE001 - every failure becomes a StoreFailure
        return _failure_from(entry, exc, doing="reading this store")
    return StoreReport(
        entry=entry,
        report=build_report(
            snapshot, gate=ClaimGate(), registry=default_registry(), measures=default_measures()
        ),
    )


def _failure_from(entry: StoreEntry, exc: Exception, *, doing: str) -> StoreFailure:
    """The failure mapping, in one place, for every way this app reads a store.

    **Shared deliberately.** A listing read and a report read fail for the same reasons,
    and an operator reading a page at seven in the morning must not be told two different
    things about one broken store depending on which page they loaded. The first version
    of the cheap listing read carried its own two catch clauses; it disagreed with these
    five on the unanticipated-exception case, and the test that covers a defect being
    reported rather than propagated caught it. Two sentences for one failure is two
    sentences that can drift apart, which is the same defect this project keeps finding in
    its own registries.

    ``doing`` names the operation that failed so the sentences read correctly from both
    call sites, and nothing else differs between them.
    """
    if isinstance(exc, StoreAuthenticationError):
        return _failure(
            entry,
            cause=f"the store refused the credential ({first_line(exc)})",
            remedy=(
                "This is a configuration problem and retrying will not help: check the "
                f"api_key for this store in {entry.origin}. Nothing was measured for it, and "
                "it contributes no figures to this page."
            ),
        )
    if isinstance(exc, StoreUnavailableError):
        return _failure(
            entry,
            cause=f"the store could not be reached ({first_line(exc)})",
            remedy=(
                "The corpus is not empty -- it is unreachable, and a report of zeroes "
                "would say otherwise. Check that the Tanseki endpoint for this store in "
                f"{entry.origin} is up and reachable from this host. Nothing was measured "
                "for it."
            ),
        )
    if isinstance(exc, (ValidationError, StoreConfigurationError)):
        # Both refusals of one rule, caught together for the reason
        # ``tenbin.cli._walk`` catches them together: two layers enforce the same thing
        # and would otherwise produce two messages for one mistake.
        return _failure(
            entry,
            cause=f"the configured store URL is unusable ({first_line(exc)})",
            remedy=(
                "Set the url for this store in "
                f"{entry.origin} to an HTTP(S) URL, over TLS unless it targets loopback."
            ),
        )
    if isinstance(exc, StoreError):
        # The typed cases above are the ones an operator can act on. This one is a read
        # that stopped for a reason with no better remedy than looking at it, and it is
        # named as its own case rather than folded into "could not be reached" -- a
        # reader told to wait for an envelope that will arrive broken waits for nothing.
        return _failure(
            entry,
            cause=f"the store could not be read ({first_line(exc)})",
            remedy=(
                "The store answered with something this program cannot use, and waiting "
                "will not fix it. Nothing was measured for this store, and it is not "
                "counted among the stores read."
            ),
        )
    # A defect in this program rather than a problem with the store. It is reported
    # instead of propagated because propagating it would take the stores that were read
    # down with it, and a page that omits a store is the failure this app exists to make
    # impossible.
    return _failure(
        entry,
        cause=f"{doing} raised {type(exc).__name__}: {first_line(exc)}",
        remedy=(
            "This is a defect in Tenbin rather than a problem with the store, and it "
            "is not a statement about this store's corpus. Nothing was measured for "
            "it, and it is not counted among the stores read."
        ),
    )


def _walk(entry: StoreEntry) -> CorpusSnapshot:
    """Open the store, walk the collection, and close it -- or raise.

    The one door to the store, and the same shape as :func:`tenbin.cli._walk`: a store
    that cannot be reached raises out of here rather than returning an empty snapshot,
    because the two are the same document to everything downstream and only one of them
    is a fact about the world.
    """
    with CLIENT_FACTORY(entry.settings) as client:
        return take_snapshot(client)


def _failure(entry: StoreEntry, *, cause: str, remedy: str) -> StoreFailure:
    """A named failure, with both sentences scrubbed of this store's credential.

    The redaction happens here rather than in the renderer so that it does not depend on
    every renderer remembering: a cause is built from an exception's text, and an
    exception's text is somebody else's string.
    """
    return StoreFailure(entry=entry, cause=entry.redact(cause), remedy=entry.redact(remedy))


#: Stated once so a caller counting outcomes has a name for the question rather than
#: reaching for an ``isinstance`` and getting the two the wrong way round. A report is
#: not the absence of a failure; that difference is the whole app.
def was_read(outcome: StoreOutcome) -> bool:
    """Whether this store was read, as opposed to failing.

    Exists so the question is asked in one place. It reads positively -- "was it read" --
    rather than "is it a failure", because a renderer that filters on ``is not None`` has
    to decide what ``None`` meant, and this program has exactly one answer to that.
    """
    return isinstance(outcome, StoreReport)


#: The wording a failed store is announced with. A constant because it appears on the
#: index and on a store's own page, and two sentences for one failure is two sentences
#: that can drift apart -- which is the same defect this project keeps finding in its own
#: measure registries, arrived at here by having written one.
FAILED_HEADLINE: Final[str] = "Could not be read"
