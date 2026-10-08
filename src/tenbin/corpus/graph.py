"""Which records are related to this one, asked once instead of worked out by enumeration.

Tanseki derives edges server-side. ``EdgeDeriver`` reads exactly four frontmatter
keys -- ``repo``, ``pr``, ``jira`` and, optionally, ``files`` -- so the store
holds a graph whether or not anybody traverses it, and ``POST /v1/documents:traverse``
returns a document's neighbourhood in one request. **Tenbin was paying for that
graph by hand**: it enumerated the collection and intersected sets of edge keys
in Python, which is a request cost, a quadratic for any per-file question, and --
worst of all -- a cost the read model *hid* rather than removed, because a
neighbourhood computed offline is indistinguishable from one computed by the
store that knows more than this program does.

So this module holds both halves and refuses to conflate them.
:func:`traversed` asks the store, in one request, and answers with what the
store said. :func:`neighbourhood` computes the same relation over a read this
program already holds, and is therefore a *second* reader of a graph the store
already owns -- legitimate for a stored read model where there is no client to
ask, and never the answer when there is one. A caller that has a
:class:`~tenbin.store.client.StoreClient` should call :func:`traversed`.

**A neighbourhood that came back empty and a neighbourhood nobody asked about are
different facts, and the type makes the difference unmissable.**
:class:`Neighbourhood` carries ``queried`` beside its ids, and :func:`unqueried`
builds the second kind. This is the same distinction the batch read makes when it
keeps a ``None`` in place of a document the store would not return, and the same
reason: an empty neighbourhood is a fact about the store -- nothing here is
related to that anchor -- while an unqueried one is a fact about the caller, and
a figure that printed the second as the first would be reporting that Kojutsu
recorded nothing about a change when the truth is that nobody asked.

**A pull-request number is not an identity, and the local walk knows it.** Tanseki
derives a ``pr`` edge from the key of the same name; this reader cannot, because
:attr:`~tenbin.corpus.record.Record.pr_key` exists precisely because pull-request
numbers repeat across repositories and joining on the number alone fabricates a
correspondence between two changes that share a digit. So the local ``pr`` walk
matches on ``(repo, pr)`` and refuses a record whose repository is
Kojutsu's placeholder. It is the one place this module is stricter than the
store, and the divergence is stated rather than hidden: a store-side ``pr`` edge
is derived from what the writer put in the key, and a local one is derived from
the address.

**Depth is bounded because an unbounded neighbourhood is not a measurement.** A
neighbourhood is a set, and a set that grows with depth answers a different
question at every depth, so the depth asked for is carried on the answer and a
caller cannot read a two-hop figure as a one-hop one.

**What this module notably does not do:** it does not build the graph, repair it,
or judge an edge. Tanseki owns which documents are related to which, and this
program's job is to ask and to report what came back -- including the ids the
store named that this read does not hold, which is :attr:`Neighbourhood.unread`
and is the visible form of a store that has moved since the read was taken.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from tenbin.corpus.record import PLACEHOLDER, Record
from tenbin.corpus.snapshot import CorpusSnapshot
from tenbin.store.client import (
    DEFAULT_TRAVERSE_DEPTH,
    MAX_TRAVERSE_DEPTH,
    StoreClient,
)

#: The four keys Tanseki's edge deriver recognises, and therefore the only relations
#: a traversal can be asked about. ``files`` is the one Kojutsu makes optional
#: -- it resolves each path against a document that already exists, so a ``files``
#: edge stays dangling until something writes documents for those paths -- and it
#: is here because a dangling edge is still a fact about the graph rather than
#: something a reader should have to guess at. Named once so that a fifth
#: relation arriving upstream is a deliberate edit to a closed set rather than a
#: string that happened to match.
REPO_RELATION: Final[str] = "repo"
PR_RELATION: Final[str] = "pr"
JIRA_RELATION: Final[str] = "jira"
FILES_RELATION: Final[str] = "files"

#: Every relation this reader will traverse, in the order a reader meets them: the
#: change, the ticket, the code. A tuple rather than a set because the order is
#: the order the corpus inventory lists them in, and a message naming the four
#: should name them in the order somebody would say them.
EDGE_KEYS: Final[tuple[str, ...]] = (
    REPO_RELATION,
    PR_RELATION,
    JIRA_RELATION,
    FILES_RELATION,
)

#: The membership test, as a mapping rather than a set so a caller can reach the
#: relation's own name for an error message. Frozen because it is a statement
#: about the store's vocabulary and a caller could not add a relation to it.
RELATIONS: Final[Mapping[str, str]] = MappingProxyType({key: key for key in EDGE_KEYS})


@dataclass(frozen=True)
class Neighbourhood:
    """One document's neighbourhood, and whether anybody asked the store for it.

    ``ids`` are the documents related to ``anchor`` under ``relation``, in the
    order they were reached, and they never include the anchor: a document in its
    own neighbourhood would be a record counted twice by whatever figure used it.
    ``queried`` is the field this type exists for. It is ``False`` only for a
    value built by :func:`unqueried`, and :meth:`__post_init__` refuses to build
    the contradiction -- an unqueried neighbourhood carrying ids -- so the two
    situations cannot be represented as one value.

    ``unread`` names ids the store returned that this read does not hold, and it
    is a second difference rather than a detail: a neighbourhood is drawn over the
    documents in hand, and an id the store names and the read lacks is a store
    that has moved since the walk, not a neighbourhood that reached further than
    the corpus. Empty on a locally computed neighbourhood, where every id came
    out of the read by construction.
    """

    anchor: str
    relation: str
    depth: int
    queried: bool
    ids: tuple[str, ...] = ()
    unread: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.anchor, str) or not self.anchor.strip():
            raise ValueError(
                "Neighbourhood.anchor is required; a neighbourhood of nothing is not a "
                "neighbourhood."
            )
        if not isinstance(self.queried, bool):
            raise ValueError(
                "Neighbourhood.queried must be a bool; a bare truthy value would let an "
                "unqueried neighbourhood be read as a queried one."
            )
        if not self.queried and (self.ids or self.unread):
            raise ValueError(
                "Neighbourhood.queried is False but ids were supplied; nobody asked, so "
                "nothing came back, and a value carrying both is a contradiction rather "
                "than a fact."
            )
        if self.anchor in self.ids:
            raise ValueError(
                "Neighbourhood.ids contains the anchor; a document is not its own "
                "neighbour and counting it twice is how a neighbourhood becomes a rate."
            )

    @property
    def is_empty(self) -> bool:
        """Whether the query returned nothing, which is not the same as not asking."""
        return self.queried and not self.ids


def edge_keys(record: Record, relation: str) -> frozenset[str]:
    """The edge keys one record holds under one relation, as comparable strings.

    Prefixed by the relation name so that two relations cannot collide -- a
    repository called ``jira=ABC-1`` is not a ticket -- and so a key can be read
    back to what produced it by anybody debugging an unexpected neighbourhood.

    An unusable field yields **no keys at all**, never a placeholder key. A
    repository Kojutsu wrote as ``unknown`` is the id's own placeholder and
    matches nothing; a record with no pull request parses to ``None`` and matches
    nothing; a file path is repository-relative, so ``src/queue.py`` in two
    repositories is two edges and the repository is carried in the key. A record
    that yielded a key for a value it does not have would put every unstated
    repository in one neighbourhood, which is the ``unknown``-bucket defect in a
    fourth costume.
    """
    if relation == REPO_RELATION:
        repo = _named(repo=record.repo)
        return frozenset({f"{REPO_RELATION}={repo}"}) if repo else frozenset()
    if relation == PR_RELATION:
        key = record.pr_key
        # ``pr_key`` is already ``None`` unless both halves are readable, so there
        # is no half-key case here to defend against.
        return frozenset({f"{PR_RELATION}={key[0]}#{key[1]}"}) if key else frozenset()
    if relation == JIRA_RELATION:
        ticket = record.jira.strip() if isinstance(record.jira, str) else ""
        return frozenset({f"{JIRA_RELATION}={ticket}"}) if ticket else frozenset()
    if relation == FILES_RELATION:
        repo = _named(repo=record.repo)
        if not repo:
            return frozenset()
        return frozenset(f"{FILES_RELATION}={repo}:{path}" for path in record.files)
    raise ValueError(_unknown_relation(relation))


def neighbourhood(
    snapshot: CorpusSnapshot,
    doc_id: str,
    rel: str,
    *,
    depth: int = DEFAULT_TRAVERSE_DEPTH,
) -> Neighbourhood:
    """The neighbourhood of ``doc_id`` over the records this read already holds.

    **A local computation, and the honest reason to reach for it is that there is
    no client to ask.** The store derives the same edges
    (:func:`traversed`), so wherever a :class:`~tenbin.store.client.StoreClient`
    is available this is the wrong call -- it is the quadratic the read model
    hides, and a caller who picks it when a request was available has traded one
    request for a whole walk without being told.

    The walk is breadth-first and bounded by ``depth``: one hop answers "which
    changes touched this file" and "which records name this ticket", which are the
    questions a reader actually has, and each further hop multiplies the response
    while changing what the answer means. Every record is visited once, so a
    relation that is transitive is walked once rather than once per path.

    An anchor this read does not hold is a queried, empty neighbourhood rather
    than a refusal: the answer to "what is related to a document I have never
    read" is nothing, and raising would make a missing anchor look like a broken
    corpus.
    """
    _validate(anchor=doc_id, relation=rel, depth=depth)
    keys_of, by_key = _index(snapshot.records, rel)
    if doc_id not in keys_of:
        return Neighbourhood(anchor=doc_id, relation=rel, depth=depth, queried=True, ids=())

    visited: set[str] = {doc_id}
    frontier: tuple[str, ...] = (doc_id,)
    found: list[str] = []
    for _ in range(depth):
        reached: list[str] = []
        for member in frontier:
            for key in sorted(keys_of.get(member, ())):
                for candidate in by_key.get(key, ()):
                    if candidate in visited:
                        continue
                    visited.add(candidate)
                    reached.append(candidate)
        if not reached:
            break
        found.extend(reached)
        frontier = tuple(reached)
    return Neighbourhood(anchor=doc_id, relation=rel, depth=depth, queried=True, ids=tuple(found))


def traversed(
    client: StoreClient,
    snapshot: CorpusSnapshot,
    doc_id: str,
    rel: str,
    *,
    depth: int = DEFAULT_TRAVERSE_DEPTH,
) -> Neighbourhood:
    """Ask the store for the neighbourhood, in one request, and report what it said.

    **This is the path that replaces the enumeration.** Everything a caller
    wanted from intersecting edge keys by hand comes back in one
    ``POST /v1/documents:traverse``, answered from an index rather than from a
    walk, and the records are filtered against the read rather than the read
    against the records.

    ``ids`` are the store's answer intersected with what this read holds, and
    ``unread`` are the ids it named that the read does not: a store that has
    written since the walk is a fact about the corpus and not an error, and
    naming it here is what keeps a figure computed over this read from implying
    the read was current.

    An empty answer is ``queried=True`` with no ids, which is what
    :meth:`Neighbourhood.is_empty` is for. A 404 from the store arrives as the
    empty answer too -- see :meth:`tenbin.store.client.StoreClient.traverse` for
    why that is not an exception -- and the two are indistinguishable from outside
    the store, which is why nothing downstream is allowed to treat an empty
    neighbourhood as evidence that the graph has nothing in it.
    """
    _validate(anchor=doc_id, relation=rel, depth=depth)
    held = {record.doc_id for record in snapshot.records}
    named = client.traverse(doc_id, rel, depth=depth)
    return Neighbourhood(
        anchor=doc_id,
        relation=rel,
        depth=depth,
        queried=True,
        ids=tuple(named_id for named_id in named if named_id in held),
        unread=tuple(named_id for named_id in named if named_id not in held),
    )


def unqueried(doc_id: str, rel: str, *, depth: int = DEFAULT_TRAVERSE_DEPTH) -> Neighbourhood:
    """The neighbourhood nobody asked about, which is not an empty one.

    **A figure that has no neighbourhood must not report a neighbourhood of
    zero.** The two are the same integer in a report and opposite facts, and the
    one that matters is the second: "Kojutsu recorded nothing related to this
    change" and "Tenbin did not ask" are indistinguishable to a reader and
    nothing like each other. Built by name rather than by leaving ``queried`` at a
    default so that a caller holding an unqueried neighbourhood is holding a value
    this program built on purpose.
    """
    _validate(anchor=doc_id, relation=rel, depth=depth)
    return Neighbourhood(anchor=doc_id, relation=rel, depth=depth, queried=False, ids=())


def _index(
    records: Iterable[Record], relation: str
) -> tuple[dict[str, frozenset[str]], dict[str, tuple[str, ...]]]:
    """Build both directions of the relation in one pass over the read.

    Two maps because a neighbourhood query asks both questions at once -- which
    edges does this document hold, and which documents hold that edge -- and
    building either alone would mean walking the read again per hop. ``by_key``
    keeps records in the order the read gave them, so two walks of the same
    snapshot produce the same neighbourhood in the same order and a figure
    computed from it does not depend on set iteration.
    """
    keys_of: dict[str, frozenset[str]] = {}
    by_key: dict[str, list[str]] = {}
    for record in records:
        keys = edge_keys(record, relation)
        if not keys:
            continue
        keys_of[record.doc_id] = keys
        for key in sorted(keys):
            by_key.setdefault(key, []).append(record.doc_id)
    return keys_of, {key: tuple(ids) for key, ids in by_key.items()}


def _named(*, repo: str | None) -> str | None:
    """The repository, unless it is Kojutsu's placeholder or nothing at all.

    Spelled here rather than imported from :func:`tenbin.measures.filtering.named_repository`
    because this package is the one below it: the rule is a reading of a document,
    and the corpus layer must not import from the measures layer to obtain it. Both
    spell it against the same public constant --
    :data:`tenbin.corpus.record.PLACEHOLDER` -- so the literal cannot drift even
    though the rule is now stated twice. That duplication is worth one shared
    helper on the record model.
    """
    if repo is None:
        return None
    candidate = repo.strip()
    if not candidate or candidate.casefold() == PLACEHOLDER:
        return None
    return candidate


def _unknown_relation(relation: str) -> str:
    """The message naming a relation the store's deriver does not derive."""
    return (
        f"Relation {relation!r} is not one Tanseki derives edges from. The four are "
        f"{', '.join(repr(key) for key in EDGE_KEYS)}, and a fifth is a change upstream "
        "rather than a spelling to try."
    )


def _validate(*, anchor: str, relation: str, depth: int) -> None:
    """Refuse a query the store could not answer as asked.

    Three conditions and each is a different mistake: an anchor nobody named
    (there is no such thing as the neighbourhood of nothing), a relation outside
    the four the deriver recognises (the store would answer with something this
    program cannot read as an edge), and a depth outside the bound the client
    enforces (an unbounded neighbourhood answers a different question at every
    depth, and the depth is carried on the answer so the difference is visible).
    """
    if not isinstance(anchor, str) or not anchor.strip():
        raise ValueError(
            "A neighbourhood needs an anchor document id; a neighbourhood of nothing is "
            "not a neighbourhood."
        )
    if relation not in RELATIONS:
        raise ValueError(_unknown_relation(relation))
    if isinstance(depth, bool) or not isinstance(depth, int):
        raise ValueError("Neighbourhood depth must be an integer.")
    if not 1 <= depth <= MAX_TRAVERSE_DEPTH:
        raise ValueError(
            f"Neighbourhood depth must be between 1 and {MAX_TRAVERSE_DEPTH}; a depth of "
            "zero returns the anchor itself and no neighbourhood."
        )


def relation_names() -> Sequence[str]:
    """The relations a traversal can be asked about, in the order a reader meets them."""
    return EDGE_KEYS
