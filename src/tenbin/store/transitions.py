"""Walking ticket status transitions, whatever system recorded them.

A second read seam, beside the knowledge store rather than inside it. That
separation is the whole design: this module knows how to page a ticket source
into a snapshot and how to say honestly how much of what it asked for it got.
It knows nothing about flow, cycle time, retrospectives, or what anybody
should conclude from a transition -- those are measures, and they live in
:mod:`tenbin.measures` where they can be given claims. It also knows nothing
about any particular ticketing system: the adapters in
:mod:`tenbin.store.ticket_sources` turn each wire into pages, and this module
turns pages into a snapshot.

**Why a separate walk and not a flag on the store walk.** The store walk
enforces an API-key allowlist and knows about collections and offsets, none of
which apply here. More importantly the two sources fail differently and need
different remedies: a knowledge store that cannot be read is a store problem,
while a ticket source that cannot be read leaves the store perfectly readable.
One walk with a ``which_source`` parameter would make every caller carry that
distinction, and the callers are measures that should not have to know.

**The walk never returns quietly on a failure**, for the same reason the store
walk does not: a snapshot that stopped early is still returned, carrying
``completeness=STORE_ERROR`` and the exception that stopped it, so a caller who
ignores completeness gets a visibly incomplete object rather than a plausible
one. The naming reuses ``Completeness`` on purpose. The four ways a walk can
stop mean the same thing whatever was being walked, and a second enum for the
same four ideas would let a renderer handle one and forget the other.

**What this source does not have.** No principal. A transition says a status
changed and when; it does not say who changed it, so nothing here can support a
per-person measure, and the granularity floor that refuses those elsewhere in
this program is not the only thing standing in the way -- there is no field to
refuse. Adapters drop principal fields (Jira authors, Linear actors) at their
own seam for exactly this reason; see :mod:`tenbin.store.ticket_sources`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import httpx

from tenbin.corpus.snapshot import Completeness, Truncation
from tenbin.corpus.transition import Transition
from tenbin.corpus.window import CorpusWindow
from tenbin.store.client import StoreError

#: The page size the walk asks for when a caller does not say. Neutral rather
#: than any backend's ceiling: each adapter clamps it to what its system
#: serves, so a walk default larger than the smallest ceiling is a request the
#: adapters narrow rather than a contract any of them has to meet.
DEFAULT_PAGE_LIMIT = 200

#: The name transition reads are known by inside a snapshot. Not "store",
#: because a report showing two sources both called "store" is a report that
#: cannot say which corpus a figure came from.
COLLECTION_NAME = "status-transitions"

__all__ = [
    "COLLECTION_NAME",
    "DEFAULT_PAGE_LIMIT",
    "TicketConfigurationError",
    "TransitionPage",
    "TransitionSnapshot",
    "TransitionSource",
    "take_transition_snapshot",
    "transitions_by_ticket",
]


class TicketConfigurationError(ValueError):
    """A ticket source was misconfigured or asked for something it has no equivalent for.

    A ``ValueError`` because it is a wrong input rather than a transport or
    remote failure: an unknown source name, a missing credential, or a scope
    (``--group`` against Jira or Linear) the system cannot express. The command
    layer turns it into a usage error, because retrying a misconfiguration
    against a healthy backend fails the same way every time.
    """


@runtime_checkable
class TransitionSource(Protocol):
    """Anything the transition walk can page.

    The whole adapter contract, and deliberately small: page in, page out,
    close when done. Everything else an adapter does -- which wire, which
    envelope, which bound semantics -- is its own module's business, and the
    walk stays ignorant of it so a fourth ticketing system is a new module
    rather than a change here.
    """

    def list_transitions(
        self,
        *,
        project_id: str | None = None,
        group_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        page: int = 1,
        limit: int = DEFAULT_PAGE_LIMIT,
    ) -> TransitionPage:
        """Read one page of transitions, numbered from 1."""
        ...

    def close(self) -> None:
        """Release whatever the read held open."""
        ...

    def __enter__(self) -> TransitionSource: ...

    def __exit__(self, *exc: object) -> None: ...


@dataclass(frozen=True)
class TransitionPage:
    """One page of transitions, and what the source said it had.

    ``total`` is the source's own count and is kept beside the rows so the walk
    can detect the same disagreement :class:`CorpusSnapshot` detects for the
    knowledge store: a source whose listing count does not match its listing
    cannot be used to say anything about completeness, including the claim that
    the walk finished.
    """

    transitions: tuple[Transition, ...]
    total: int
    page: int
    limit: int
    unreadable: int
    #: Why the rows that could not be read could not be read. Bounded per page,
    #: because a page of ten thousand broken rows is one defect repeated and the
    #: tenth thousand example adds nothing a reader could act on.
    unreadable_examples: tuple[str, ...] = ()

    @property
    def total_pages(self) -> int:
        if self.limit <= 0:
            return 0
        return max(1, -(-self.total // self.limit))


@dataclass(frozen=True)
class TransitionSnapshot:
    """A read of a ticket source's transitions, and everything needed to doubt it.

    Implements :class:`tenbin.corpus.snapshot.Snapshot`, which is why it has no
    ``records`` field: a figure is checked against the protocol's *integrity*
    surface -- what was read, when, how many the source claimed, how far the walk
    got -- and the records themselves are this source's own type, which the
    Kojutsu-shaped ``CorpusSnapshot`` could not hold honestly.

    ``window`` is derived in ``__post_init__`` for the reason it is derived
    there in the other snapshot: a population is a span and a count, and a caller
    who could supply its own window could supply one that disagrees with the
    transitions every figure was computed from.

    ``unreadable_rows`` counts rows the source served that could not be read.
    It is a separate field from ``error`` because they are different findings: a
    failed walk stopped, while unreadable rows were served and could not be
    interpreted, so the walk completed over a corpus it could only partly
    describe.
    """

    transitions: tuple[Transition, ...]
    read_at: datetime
    store_total: int
    enumerated: int
    completeness: Completeness
    truncation: Truncation | None = None
    error: StoreError | None = None
    unreadable_rows: int = 0
    unreadable_examples: tuple[str, ...] = ()
    project_id: str | None = None
    group_id: str | None = None
    window: CorpusWindow = field(init=False)

    #: ``CorpusSnapshot`` names the source here; this one holds it as a field so
    #: two reads of two different backends are distinguishable in a set.
    collection: str = COLLECTION_NAME

    def __post_init__(self) -> None:
        moments = [row.at for row in self.transitions]
        object.__setattr__(
            self,
            "window",
            CorpusWindow(
                earliest=min(moments) if moments else None,
                latest=max(moments) if moments else None,
                unreadable_timestamps=0,
            ),
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.collection,
                self.read_at,
                self.enumerated,
                self.completeness,
                self.project_id,
                self.group_id,
                self.unreadable_rows,
            )
        )

    @property
    def is_complete(self) -> bool:
        """Whether the walk read everything the source said it held."""
        return self.completeness is Completeness.COMPLETE


def take_transition_snapshot(
    client: TransitionSource,
    *,
    project_id: str | None = None,
    group_id: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    page_limit: int = DEFAULT_PAGE_LIMIT,
) -> TransitionSnapshot:
    """Walk the transitions into a snapshot that can be doubted.

    Raises rather than returning when the **first** request fails, for the same
    reason the store walk does: an empty snapshot from a source that was never
    reached is indistinguishable from a corpus of nothing, and that distinction
    is the whole difference between a finding and a failure. Once at least one
    page has arrived, a later failure returns a partial snapshot carrying the
    exception, because a partial corpus is still data.
    """
    read_at = datetime.now(UTC)
    transitions: list[Transition] = []
    unreadable: list[str] = []
    store_total = 0
    enumerated = 0
    reached = 0
    store_error: StoreError | None = None
    completeness = Completeness.COMPLETE

    page = 1
    first_page = True

    while True:
        try:
            result = client.list_transitions(
                project_id=project_id,
                group_id=group_id,
                since=since,
                until=until,
                page=page,
                limit=page_limit,
            )
        except StoreError as exc:
            if first_page:
                raise
            store_error = exc
            completeness = Completeness.STORE_ERROR
            break

        first_page = False
        transitions.extend(result.transitions)
        unreadable.extend(result.unreadable_examples)
        store_total = result.total
        enumerated += len(result.transitions)

        if page >= result.total_pages:
            break
        page += 1

    if store_error is None and store_total != enumerated and unreadable:
        completeness = Completeness.STORE_TOTAL_EXCEEDED

    truncation: Truncation | None = None
    if completeness is Completeness.OFFSET_CAP:
        truncation = Truncation(
            offset_reached=reached,
            records_missing=max(0, store_total - enumerated),
            boundary_repositories={},
            boundary_months={},
        )

    return TransitionSnapshot(
        transitions=tuple(transitions),
        read_at=read_at,
        store_total=store_total,
        enumerated=enumerated,
        completeness=completeness,
        truncation=truncation,
        error=store_error,
        unreadable_rows=len(unreadable),
        unreadable_examples=tuple(unreadable[:5]),
        project_id=project_id,
        group_id=group_id,
    )


def transitions_by_ticket(
    transitions: Iterable[Transition],
) -> Mapping[str, tuple[Transition, ...]]:
    """Group by ticket, each ticket's transitions in the order they happened.

    Sorted by ``at`` rather than kept in read order, because a source's
    pagination order is not a guarantee and a measure that computed an interval
    from two out-of-order rows would invent a negative duration.
    """
    grouped: dict[str, list[Transition]] = {}
    for transition in transitions:
        grouped.setdefault(transition.ticket_id, []).append(transition)
    return MappingProxyType(
        {
            ticket_id: tuple(sorted(rows, key=lambda row: (row.at, row.to_state)))
            for ticket_id, rows in grouped.items()
        }
    )


# Kept for symmetry with the store client's injectable seam, and so a test can
# substitute the transport without reaching into httpx.
TransportFactory = Callable[[], httpx.BaseTransport]
