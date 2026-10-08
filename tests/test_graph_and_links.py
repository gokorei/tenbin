"""The graph Kojutsu already derives, and the one comparison the corpus holds.

Three things are proved here and they are different kinds of proof. **A neighbourhood
is asked for rather than worked out**: a traversal round trip over the in-memory
protocol fake asserts on the request that left the client, so the claim is that the
store was asked one question in one call -- not that a local walk happens to agree with
one. **The two absences are not the same absence**: a traversal the store answered with
nothing and a neighbourhood nobody asked about are different facts, and the type has to
make telling them apart impossible rather than merely possible. **One vocabulary**: the
declared-versus-reconstructed comparison is asserted against Kojutsu's own source
file rather than against a list of strings somebody typed, because a copy that drifts is
the failure and a copy checked against itself cannot drift visibly.

**The Jira tests are about a key nobody will read back.** ``jira`` is the only field
crossing from the forge into the planning system, it comes off a branch-name pattern,
and Tenbin never opens the ticket system -- so the interesting assertions are about
what the figure refuses to say. A capture naming no ticket is not a capture with no
ticket behind it, and the figure that fails to say so would be a finding about planning
dressed as a count of branch names.

**The rationale tests are about misreading, not arithmetic.** A divergence is not a
defect, a restatement is not corroboration, and whether a particular disagreement
mattered is a question about one change and one person's job. Each is asserted against
the claim rather than against the buckets, because a figure that renders a caveat nobody
required is the failure this package was built to stop.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import httpx
import pytest

from tenbin.claims.mechanism import Granularity
from tenbin.claims.model import ClaimKind
from tenbin.corpus.graph import (
    FILES_RELATION,
    JIRA_RELATION,
    PR_RELATION,
    REPO_RELATION,
    Neighbourhood,
    edge_keys,
    neighbourhood,
    traversed,
    unqueried,
)
from tenbin.corpus.record import Record
from tenbin.corpus.snapshot import CorpusSnapshot
from tenbin.measures.base import CountFigure, DistributionFigure, Figure, FigureGroup
from tenbin.measures.rationale_comparison import (
    KOJUTSU_CHECKOUT,
    RATIONALE_RELATIONSHIP_SLUG,
    REASON_NOT_READABLE,
    SOURCE_NOT_READABLE,
    ComparisonOutcome,
    RationaleRelationshipMeasure,
    RationaleSummary,
    compare_rationales,
    outcome_values_are_kojutsus,
    reason_text,
)
from tenbin.measures.tickets import JiraLinkageMeasure
from tenbin.measures.trust import Independence
from tenbin.store.client import (
    DEFAULT_TRAVERSE_DEPTH,
    MAX_TRAVERSAL_RELATION_LENGTH,
    MAX_TRAVERSE_DEPTH,
    StoreClient,
    StoreResponseError,
)
from tests.fakes import FakeStore, closing, store_over
from tests.fixtures import build_record, build_snapshot

_LINKAGE = JiraLinkageMeasure()
_RELATIONSHIP = RationaleRelationshipMeasure()

_ANCHOR = "acme/widget/pr-42/answer-0001"
_NEIGHBOURS = ("acme/widget/pr-42/answer-0002", "acme/widget/pr-43/answer-0003")


class TraversingStore(FakeStore):
    """``FakeStore`` with the one route it does not serve, so the real client code runs.

    ``FakeStore`` answers five routes and a traversal is a sixth, so a subclass adds it
    rather than a new fake replacing one. Subclassing rather than standing in keeps the
    property that makes this fake worth using at all: the client's own ``_request``,
    ``_json``, ``_check_status`` and 404 handling are what execute, so a round trip here
    is a round trip through the production path with a transport instead of a socket.

    A traversal for an anchor the store does not hold answers 404, which is what Tanseki
    does and what the client's empty-tuple behaviour depends on.
    """

    def __init__(
        self,
        documents: Mapping[str, Mapping[str, Any]] | None = None,
        *,
        edges: Mapping[tuple[str, str], Sequence[str]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(documents, **kwargs)
        self.edges: dict[tuple[str, str], tuple[str, ...]] = {
            key: tuple(value) for key, value in (edges or {}).items()
        }

    def _handle(self, request: httpx.Request) -> httpx.Response:
        """Serve the traversal route, and defer everything else to the parent."""
        if not request.url.path.endswith("/documents:traverse"):
            return super()._handle(request)
        self.requests.append(request)
        payload = json.loads(request.content.decode())
        anchor = str(payload.get("id"))
        if anchor not in self.ids:
            return httpx.Response(404, json={"error": "not found"})
        found = self.edges.get((anchor, str(payload.get("rel"))), ())
        return httpx.Response(200, json={"ids": list(found)})


def _traversing_corpus() -> TraversingStore:
    """A store holding three documents, two of them neighbours of the first.

    The read helper below holds the same three ids, because :func:`traversed`
    intersects the store's answer with the records the read holds -- and an
    intersection with nothing is how a correct traversal reads as an empty one.
    """
    return TraversingStore(
        {doc_id: {"record_kind": "answer"} for doc_id in (_ANCHOR, *_NEIGHBOURS)},
        edges={(_ANCHOR, REPO_RELATION): _NEIGHBOURS},
    )


def _neighbourhood_read() -> CorpusSnapshot:
    """A read holding exactly the documents the traversing corpus serves."""
    return build_snapshot(
        tuple(
            build_record(doc_id=doc_id, repo=None, pr=None, record_kind="answer")
            for doc_id in (_ANCHOR, *_NEIGHBOURS)
        )
    )


def _distributions(figure: Figure) -> tuple[DistributionFigure, ...]:
    """The distribution payloads of a figure, whether it is one or a group of several."""
    figures = figure.figures if isinstance(figure, FigureGroup) else (figure,)
    return tuple(item for item in figures if isinstance(item, DistributionFigure))


def _counts(figure: Figure) -> tuple[CountFigure, ...]:
    """The count payloads of a figure, so a second figure can be asserted on by name."""
    figures = figure.figures if isinstance(figure, FigureGroup) else (figure,)
    return tuple(item for item in figures if isinstance(item, CountFigure))


def _capture(
    *,
    entry: str = "answer-0001",
    pr: int = 1,
    ticket: str | None = None,
    repo: str | None = "acme/widget",
) -> Record:
    """A capture, with or without the branch name carrying a ticket key."""
    return build_record(entry_id=entry, repo=repo, pr=pr, jira=ticket)


def _rationale(
    *,
    pr: int | None,
    entry: str,
    source: str,
    reason: str | None,
    declared_by: str | None = None,
    model: str | None = "model-one",
    revision: int = 1,
    repo: str | None = "acme/widget",
) -> Record:
    """A rationale in the shape Kojutsu's projection writes one.

    The body is the rendered document rather than a bare string, because the reason is
    read out of a ``## Reason`` section and a body without one is a real case with its
    own exclusion. ``reason=None`` writes the heading and nothing under it, which is the
    document a projection would produce for a rationale whose text was empty.

    ``declared_by`` defaults per source -- a stated reason from the author, an inferred
    one from the reviewer. Defaulting both to one account would make every pair a
    restatement and quietly turn a divergence test into a test of the same-acknowledged
    branch, which is the failure the independence axis exists to prevent.
    """
    account = declared_by or ("agent-one" if source == "declared" else "agent-two")
    body = [
        f"# Rationale {revision}",
        "",
        f"## Reason\n{reason if reason is not None else ''}\n",
        "## Attribution",
        f"Declared by: {account}",
        f"Source: {source}",
    ]
    return build_record(
        entry_id=entry,
        repo=repo,
        pr=pr,
        rationale=True,
        record_kind=None,
        tags=["rationale", f"rationale_{source}"],
        content="\n".join(body) + "\n",
        rationale_source=source,
        rationale_revision=revision,
        declared_by=account,
        declared_by_model=model,
    )


def _comparison(
    declared: str | None,
    reconstructed: str | None,
    *,
    declared_by: str = "agent-one",
    declared_model: str | None = "model-one",
    reconstructed_by: str = "agent-two",
    reconstructed_model: str | None = "model-two",
) -> Any:
    """A comparison of two reasons, built through the public comparator."""
    summaries: list[RationaleSummary | None] = []
    for text, account, model in (
        (declared, declared_by, declared_model),
        (reconstructed, reconstructed_by, reconstructed_model),
    ):
        summaries.append(
            None
            if text is None
            else RationaleSummary(
                doc_id=f"doc-{account}",
                source="declared" if declared is text else "reconstructed",
                text=text,
                declared_by=account,
                model=model,
                revision=1,
            )
        )
    return compare_rationales(summaries[0], summaries[1])


# -- the graph is asked for, not rebuilt --------------------------------------------


def test_a_neighbourhood_is_one_request_to_the_store_rather_than_a_walk_over_the_corpus() -> None:
    """Acceptance criterion 1, asserted on the request that left the client.

    The old reading answered "which changes touched this file" by enumerating the
    whole collection and intersecting sets of edge keys in Python. That is a request
    cost, a quadratic in the corpus for any per-file question, and worst of all a cost
    the read model *hides* rather than removes -- a neighbourhood computed offline is
    indistinguishable from one computed by a store that knows more than this program
    does. So the assertion is that exactly one request was issued, it was the traversal,
    and the listing was never touched at all.
    """
    store = _traversing_corpus()
    with closing(store) as client:
        reading = traversed(client, _neighbourhood_read(), _ANCHOR, REPO_RELATION)
        issued = [request.url.path for request in store.requests]

    assert len(issued) == 1
    assert issued[0].endswith("/documents:traverse")
    assert not any(path.endswith("/documents") for path in issued)
    assert not any(path.endswith("/documents:query") for path in issued)
    assert reading.ids == _NEIGHBOURS
    assert reading.queried is True
    assert reading.depth == DEFAULT_TRAVERSE_DEPTH


def test_a_traversal_the_store_answered_with_nothing_is_not_a_neighbourhood_nobody_asked_about() -> (
    None
):
    """Acceptance criterion 6, and the difference the type exists to keep.

    "Kojutsu recorded nothing related to this change" and "Tenbin did not ask" are
    the same integer in a report and opposite facts. The second is the one that matters:
    a figure that printed it as the first would be reporting an absence of knowledge as
    a finding about the corpus. So ``queried`` sits beside the ids, and the two are not
    merely distinguishable -- an unqueried value carrying ids cannot be constructed.
    """
    with closing(_traversing_corpus()) as store:
        empty = traversed(store, build_snapshot([]), _ANCHOR, JIRA_RELATION)
        never = unqueried(_ANCHOR, JIRA_RELATION)

    assert empty.ids == ()
    assert empty.queried is True
    assert empty.is_empty is True

    assert never.ids == ()
    assert never.queried is False
    assert never.is_empty is False

    with pytest.raises(ValueError):
        Neighbourhood(anchor=_ANCHOR, relation=JIRA_RELATION, depth=1, queried=False, ids=("x",))


def test_a_traversal_of_an_anchor_the_store_does_not_hold_is_an_empty_answer_and_not_an_exception() -> (
    None
):
    """The one place ``traverse`` departs from ``get_document``'s raise, and why.

    A single-document read has exactly one right answer and raises. A neighbourhood
    question has two situations -- the anchor is absent, and the anchor is present with
    nothing related to it -- and both have the same honest answer, which is *there is
    nothing here*. A caller cannot tell an unwritten document from one with no
    neighbours: both are documents this program holds no evidence about. Raising would
    push the distinction into every caller's control flow and buy nothing.
    """
    with closing(_traversing_corpus()) as store:
        assert store.traverse("acme/widget/pr-99/answer-0001", REPO_RELATION) == ()
        reading = traversed(
            store, build_snapshot([]), "acme/widget/pr-99/answer-0001", REPO_RELATION
        )
    assert reading.queried is True
    assert reading.is_empty is True


def test_a_traversal_answering_with_no_id_array_is_refused_rather_than_read_as_an_empty_graph() -> (
    None
):
    """The defensive half of the read seam, on the envelope that could lie.

    "I could not find your neighbourhood" and "there is no neighbourhood" are the two
    sentences a caller cannot tell apart, and answering the first with the second would
    report a graph the store never described. ``list_documents`` already refuses a
    listing with no array; this is the same rule on the same shape.
    """
    handler = lambda request: httpx.Response(200, json={"total": 0})  # noqa: E731
    with closing(store_over(handler)) as client, pytest.raises(StoreResponseError):
        client.traverse(_ANCHOR, REPO_RELATION)


def test_a_second_hop_reaching_one_document_along_two_paths_counts_it_once() -> None:
    """Deduplication, because a neighbourhood carrying an id twice double-counts.

    A depth-two traversal can legitimately reach the same document by two routes, and
    a record behind that id appearing twice in whatever figure used it would be a rate
    above one built out of an accident of graph shape rather than a fact about the
    corpus.
    """
    first, second = _NEIGHBOURS
    with closing(
        TraversingStore(
            {_ANCHOR: {"record_kind": "answer"}, first: {"record_kind": "answer"}},
            edges={(_ANCHOR, REPO_RELATION): (first, second), (first, REPO_RELATION): (second,)},
        )
    ) as store:
        found = store.traverse(_ANCHOR, REPO_RELATION, depth=2)
    assert found == (first, second)
    assert len(found) == len(set(found))


def test_a_depth_of_zero_is_refused_because_a_zero_hop_traversal_would_return_the_anchor() -> None:
    """Bounded, not clamped, and the lower bound is the one that matters.

    ``get_document`` takes exactly one id; a traversal that returned its own anchor
    would put a document in its own neighbourhood, and anything counting that
    neighbourhood would be counting a record against itself. Clamping rather than
    refusing would answer a question the caller did not ask while looking as though it
    had answered it.
    """
    with closing(_traversing_corpus()) as store:
        with pytest.raises(ValueError):
            store.traverse(_ANCHOR, REPO_RELATION, depth=0)
        with pytest.raises(ValueError):
            store.traverse(_ANCHOR, REPO_RELATION, depth=MAX_TRAVERSE_DEPTH + 1)


def test_a_store_that_has_written_since_the_read_names_ids_the_read_does_not_hold() -> None:
    """``unread`` is the visible form of a read model that has moved on.

    A neighbourhood is drawn over the documents in hand, and an id the store names and
    the read lacks is not a neighbourhood that reached further than the corpus -- it is a
    store that has been written to since the walk. Naming it is what keeps a figure
    computed over this read from implying the read was current.
    """
    snapshot = build_snapshot([])
    with closing(_traversing_corpus()) as store:
        reading = traversed(store, snapshot, _ANCHOR, REPO_RELATION)
    assert reading.ids == ()
    assert reading.unread == _NEIGHBOURS


def test_a_local_walk_refuses_to_join_a_pull_request_number_without_a_named_repository() -> None:
    """A pull-request number is not an identity, and the local walk knows it.

    ``Record.pr_key`` is a property precisely because pull-request numbers repeat across
    repositories: joining on the number alone fabricates a correspondence between two
    changes that share a digit. Kojutsu writes the literal ``unknown`` where a
    repository was never stated, so a truthiness test here would put every unstated
    repository in this read into one neighbourhood.
    """
    placeholder = build_record(entry_id="answer-0001", repo="unknown", pr=42, jira="ABC-1")
    named = build_record(entry_id="answer-0002", repo="acme/widget", pr=42, jira="ABC-1")
    assert edge_keys(placeholder, JIRA_RELATION) == frozenset({"jira=ABC-1"})
    assert edge_keys(named, REPO_RELATION) == frozenset({"repo=acme/widget"})

    reading = neighbourhood(build_snapshot([placeholder, named]), placeholder.doc_id, REPO_RELATION)
    assert reading.queried is True
    assert reading.ids == ()


def test_a_relation_tanseki_does_not_derive_is_refused_rather_than_tried() -> None:
    """The four keys are a closed set, and a fifth is a change upstream.

    ``repo``, ``pr``, ``jira`` and ``files`` are what ``EdgeDeriver`` reads. A relation
    outside them is not a spelling to try: the store would answer with something this
    program cannot read as an edge, and the caller would report it.
    """
    with closing(_traversing_corpus()) as store, pytest.raises(ValueError):
        traversed(store, build_snapshot([]), _ANCHOR, "branch")
    with pytest.raises(ValueError):
        neighbourhood(build_snapshot([]), _ANCHOR, "branch")


def test_a_blank_relation_is_refused_before_the_store_is_asked_because_it_has_no_vocabulary_for_one() -> (
    None
):
    """An empty relation is neither a traversal of everything nor of nothing.

    It is a request every store would answer with something, and this program would then
    report. The relation also goes into a body a store logs, so an unbounded
    caller-supplied string should not become an unbounded log line -- which is the other
    half of the same bound, and the reason both live in one helper.
    """
    with closing(_traversing_corpus()) as client:
        with pytest.raises(ValueError):
            client.traverse(_ANCHOR, "   ")
        with pytest.raises(ValueError):
            client.traverse(_ANCHOR, "x" * (MAX_TRAVERSAL_RELATION_LENGTH + 1))
        with pytest.raises(ValueError):
            client.traverse(_ANCHOR, REPO_RELATION, depth=True)


def test_a_traversal_entry_that_is_not_an_id_is_dropped_rather_than_published_as_one() -> None:
    """Leniency on the *spelling* of an entry, never on its being absent.

    A neighbourhood containing the empty string is a relation to nothing, and a bare
    number where an id belongs is a store defect rather than a neighbour. Dropping both
    is the same posture :meth:`StoreClient.list_documents` takes -- a shape that has
    varied is not a broken document, and a reader that refused real responses would be
    refusing the right things rather than the broken ones.
    """
    payload = {"ids": [_ANCHOR, "  ", 42, {"id": _NEIGHBOURS[0]}, {"nope": 1}]}
    with closing(store_over(lambda _request: httpx.Response(200, json=payload))) as client:
        assert client.traverse(_ANCHOR, REPO_RELATION) == (_ANCHOR, _NEIGHBOURS[0])


def test_a_document_in_its_own_neighbourhood_is_refused_because_that_is_how_a_rate_appears() -> (
    None
):
    """The anchor invariant, checked where it can be violated.

    Nothing about a store's behaviour stops an id coming back as its own neighbour, and
    a neighbourhood carrying the anchor would have the record behind it counted twice by
    whatever figure used it.
    """
    with pytest.raises(ValueError):
        Neighbourhood(anchor=_ANCHOR, relation=REPO_RELATION, depth=1, queried=True, ids=(_ANCHOR,))


def test_a_local_walk_is_bounded_by_its_depth_and_carries_the_depth_it_was_given() -> None:
    """The inherited read-model walk, checked rather than trusted.

    ``neighbourhood`` is a *second* reader of a graph the store already owns, and the
    only honest reason to reach for it is that there is no client to ask. So it has to
    be bounded -- one hop answers "which changes touched this file", and each further
    hop changes what the answer means -- and the depth has to ride on the answer, so a
    two-hop figure cannot be read as a one-hop one by somebody who never saw the request.

    The chain is over ``files`` because it is the only one of the four that is not an
    equivalence: A touched ``src/queue.py`` and B touched it and ``src/worker.py``, so
    C arrives at hop two and nowhere sooner. A depth test built on ``jira`` would pass at
    depth one for the wrong reason, since every member of a ticket already holds the key
    the anchor holds.
    """
    first = build_record(entry_id="answer-0001", pr=1, frontmatter={"files": ["src/queue.py"]})
    middle = build_record(
        entry_id="answer-0002",
        pr=2,
        frontmatter={"files": ["src/queue.py", "src/worker.py"]},
    )
    last = build_record(entry_id="answer-0003", pr=3, frontmatter={"files": ["src/worker.py"]})
    snapshot = build_snapshot([first, middle, last])

    one_hop = neighbourhood(snapshot, first.doc_id, FILES_RELATION)
    two_hop = neighbourhood(snapshot, first.doc_id, FILES_RELATION, depth=2)

    assert one_hop.ids == (middle.doc_id,)
    assert one_hop.depth == 1
    assert two_hop.ids == (middle.doc_id, last.doc_id)
    assert two_hop.depth == 2
    # A depth past the chain's end returns what the chain held rather than raising or
    # repeating itself: a walk that visited each document once cannot find more.
    assert neighbourhood(snapshot, first.doc_id, FILES_RELATION, depth=3).ids == two_hop.ids


def test_an_anchor_the_read_does_not_hold_is_an_empty_neighbourhood_rather_than_a_refusal() -> None:
    """A missing anchor is not a broken corpus.

    The honest answer to "what is related to a document I have never read" is nothing,
    and raising would make a record the reader lacks look like a defect in the read
    rather than a fact about the read -- which is the distinction
    :attr:`Neighbourhood.queried` exists to keep.
    """
    reading = neighbourhood(build_snapshot([]), "acme/widget/pr-1/answer-0009", JIRA_RELATION)
    assert reading.queried is True
    assert reading.is_empty is True


def test_the_pull_request_edge_needs_both_halves_because_a_number_is_not_an_identity() -> None:
    """``pr`` is keyed on ``(repo, pr)``, and the halves come from different sources.

    The number comes from the id path and the repository from the same path, so a record
    with one but not the other yields no key at all. A partial key would join to every
    repository holding that number, which is a fabricated correspondence and exactly what
    :attr:`~tenbin.corpus.record.Record.pr_key` refuses to return.
    """
    keyed = _capture(entry="answer-0001", pr=7, ticket="ABC-1")
    unnumbered = build_record(entry_id="answer-0002", repo="acme/widget", pr=None, jira="ABC-1")
    assert keyed.pr_key == ("acme/widget", 7)
    assert unnumbered.pr_key is None
    assert edge_keys(keyed, PR_RELATION) == frozenset({"pr=acme/widget#7"})
    assert edge_keys(unnumbered, PR_RELATION) == frozenset()


def test_a_file_edge_carries_its_repository_because_a_path_is_relative_to_one() -> None:
    """``src/queue.py`` exists once per repository, so the key has to say which.

    Without the repository on the key, every repository's ``src/queue.py`` would land in
    one neighbourhood and the file figures above it would be counting a filename rather
    than a file. A record naming no repository yields no file keys at all rather than
    keys under Kojutsu's placeholder, which is the ``unknown``-bucket defect in a
    fourth costume.
    """
    touching = build_record(
        entry_id="answer-0001", repo="acme/widget", pr=1, frontmatter={"files": ["src/queue.py"]}
    )
    orphan = build_record(
        entry_id="answer-0002", repo="unknown", pr=2, frontmatter={"files": ["src/queue.py"]}
    )
    assert edge_keys(touching, FILES_RELATION) == frozenset({"files=acme/widget:src/queue.py"})
    assert edge_keys(orphan, FILES_RELATION) == frozenset()


def test_an_unqueried_neighbourhood_refuses_to_be_built_with_a_blank_anchor() -> None:
    """``unqueried`` is built by name so a caller holds a value this program made on purpose."""
    with pytest.raises(ValueError):
        unqueried("   ", JIRA_RELATION)
    with pytest.raises(ValueError):
        unqueried(_ANCHOR, JIRA_RELATION, depth=0)


# -- jira: the key that is never read back ------------------------------------------


def test_a_ticket_named_by_a_single_capture_has_a_rung_of_its_own() -> None:
    """Acceptance criterion 7: a ticket nobody else referenced is counted.

    It lands on the single-capture rung rather than being folded into a bucket of
    tickets that recurred. Dropping it would report a corpus whose tickets all
    recurred, which is a corpus none of us have -- and it would turn the first rung of a
    five-rung ladder into dead space whose shape depends on the corpus.
    """
    records = (
        _capture(entry="answer-0001", ticket="ABC-1"),
        _capture(entry="answer-0002", ticket="ABC-2"),
        _capture(entry="answer-0003", ticket="ABC-2"),
    )
    figure = _LINKAGE.compute(build_snapshot(records))
    (whole,) = _distributions(figure)

    assert whole.values["a ticket named by exactly 1 capture"] == 1
    assert whole.values["a ticket named by exactly 2 captures"] == 2
    assert sum(whole.values.values()) == 3


def test_the_linkage_claim_says_a_missing_key_means_not_supplied_rather_than_none() -> None:
    """Acceptance criterion 2, asserted on the claim rather than on the count.

    Kojutsu reads ``jira_ticket_key`` from a branch-name pattern
    (``tanseki_mapping.py:136``), so a branch that did not follow the convention carries
    no key. That is a fact about the branch, and a figure that failed to say so would
    have a reader counting the gap as work nobody planned.
    """
    claim = _LINKAGE.claim(build_snapshot([_capture()]))
    rendered = claim.render_text()

    assert "absence here means not supplied, not none" in rendered
    assert claim.granularity is Granularity.team
    assert claim.kind is ClaimKind.descriptive
    assert claim.denominator.size_noun == "capture records"
    assert claim.denominator.size == 1


def test_a_stated_reason_is_named_as_a_separate_gap_because_its_writer_never_emits_a_ticket_key() -> (
    None
):
    """Two absences, not one, and folding them together misreports where the gap is.

    Kojutsu's rationale writer emits ``repo`` and ``pr`` and not ``jira``
    (``tanseki_mapping.py:224``), so every stated reason is a capture that *could not*
    have named a ticket. A capture whose branch simply did not carry a key is a
    different gap, and a reader fixing one of them does not fix the other.
    """
    records = (
        _capture(entry="answer-0001", ticket="ABC-1"),
        _capture(entry="answer-0002"),
        build_record(entry_id="rationale-v1-0", pr=1, rationale=True, record_kind=None),
    )
    (whole,) = _distributions(_LINKAGE.compute(build_snapshot(records)))

    assert sum(whole.values.values()) == 1
    assert len(whole.excluded) == 2
    assert sum(whole.excluded.values()) == 2
    assert whole.counted + whole.excluded_total == whole.claim.denominator.size


def test_the_crowding_ladder_shows_its_whole_shape_because_a_distribution_whose_shape_depends_on_the_corpus_is_one_every_report_has_to_guard_against() -> (
    None
):
    """Every rung materialised at zero, and the top rung open-ended.

    A corpus of five tickets each named once has to render the same five labels as a
    corpus of five thousand, or a reader comparing two runs is comparing two shapes
    rather than two corpora.
    """
    records = tuple(_capture(entry=f"answer-{n:04d}", ticket=f"ABC-{n}") for n in range(3))
    (whole,) = _distributions(_LINKAGE.compute(build_snapshot(records)))
    assert len(whole.values) == 5
    assert whole.values["a ticket named by 5 or more captures"] == 0


# -- one vocabulary, and it is Kojutsu's ----------------------------------------


def test_the_vocabulary_is_exactly_the_six_outcomes_kojutsu_defines() -> None:
    """Acceptance criterion 3, half one: the copy is the six values, verbatim.

    Unconditional, and therefore the half that runs on a machine with only Tenbin in
    it. A member spelled differently from Kojutsu's is the failure this package
    exists to prevent, and it has to be caught by a check that does not depend on a
    neighbouring checkout being present.
    """
    assert {outcome.value for outcome in ComparisonOutcome} == {
        "divergent",
        "concurrent",
        "restatement",
        "declared_only",
        "reconstructed_only",
        "neither",
    }
    assert [outcome.value for outcome in ComparisonOutcome] == [
        "divergent",
        "concurrent",
        "restatement",
        "declared_only",
        "reconstructed_only",
        "neither",
    ]


@pytest.mark.skipif(
    not (KOJUTSU_CHECKOUT / "src" / "kojutsu" / "core" / "rationale_link.py").is_file(),
    reason="Kojutsu is not checked out beside this repository",
)
def test_the_copied_vocabulary_still_equals_kojutsus_own_source_rather_than_a_list_somebody_typed() -> (
    None
):
    """Acceptance criterion 3, half two: checked against the file, not against a literal.

    The test above would prove this enum equals six strings somebody wrote down; it
    would not notice Kojutsu adding a seventh. Reading the source is the only check
    that catches upstream drift, and the drift would be silent -- both enumerations
    would be right about most pairs, so no assertion about buckets would fail and the
    two would simply disagree in a report about a corpus whose author had already
    computed the answer.

    Skipped rather than failed when the checkout is absent, and *skipped*, not
    asserted-false: :func:`outcome_values_are_kojutsus` answers ``False`` for both
    "they differ" and "I cannot read it", and a test that turned that into a failure
    would report a passing machine as broken.
    """
    assert outcome_values_are_kojutsus() is True


def test_every_outcome_kojutsu_defines_is_reachable_from_here() -> None:
    """A closed vocabulary is only a guarantee if each member can actually occur.

    ``NEITHER`` is unreachable from the measure -- a change reaches the population
    because it carries a rationale -- so nothing but a direct test would prove the copy
    has no dead member, and a member nothing can produce is vocabulary inviting a
    reader to expect a population that does not exist.
    """
    assert _comparison(None, None).outcome is ComparisonOutcome.NEITHER
    assert _comparison("because", None).outcome is ComparisonOutcome.DECLARED_ONLY
    assert _comparison(None, "because").outcome is ComparisonOutcome.RECONSTRUCTED_ONLY
    assert _comparison("one reason", "another reason").outcome is ComparisonOutcome.DIVERGENT
    assert _comparison("the same words", "the   same WORDS").outcome is ComparisonOutcome.CONCURRENT
    assert (
        _comparison(
            "the same words",
            "the same words",
            declared_by="one",
            declared_model="model-one",
            reconstructed_by="one",
            reconstructed_model="model-one",
        ).outcome
        is ComparisonOutcome.RESTATEMENT
    )
    assert {comparison.outcome for comparison in (_comparison(None, None),)} == {
        ComparisonOutcome.NEITHER
    }


def test_two_rationales_from_the_same_principal_and_model_are_a_restatement_whether_or_not_they_agree() -> (
    None
):
    """Agreement from the same mind is the expected outcome and carries no information.

    Reporting it as corroboration would reintroduce exactly the confusion the provenance
    axis exists to prevent, one layer up -- which is why ``RESTATEMENT`` is a member with
    the same weight as ``DIVERGENT`` rather than a note attached to it.
    """
    same_mind = _comparison(
        "identical",
        "identical",
        declared_by="one",
        declared_model="model-one",
        reconstructed_by="one",
        reconstructed_model="model-one",
    )
    same_mind_differing = _comparison(
        "one wording",
        "another wording",
        declared_by="one",
        declared_model="model-one",
        reconstructed_by="one",
        reconstructed_model="model-one",
    )
    assert same_mind.independence is Independence.SELF_CERTIFIED
    assert same_mind.is_informative is False
    assert same_mind_differing.outcome is ComparisonOutcome.RESTATEMENT
    assert same_mind_differing.is_informative is False


def test_an_unstated_model_is_not_treated_as_matching_another_because_two_unknowns_may_be_the_same() -> (
    None
):
    """The axis is lossy at the storage boundary and says so rather than filling the gap.

    Kojutsu leaves the key absent where no model was named, so an unstated model is
    absent rather than a model called ``unknown``. Assuming two absences are equal would
    manufacture a worse label than admitting the gap -- and a manufactured
    ``MODEL_SEPARATED`` is a restatement filed as a second opinion, which is the exact
    misreading ``is_informative`` exists to prevent.
    """
    comparison = _comparison(
        "one",
        "two",
        declared_by="one",
        declared_model=None,
        reconstructed_by="one",
        reconstructed_model="model-two",
    )
    assert comparison.independence is Independence.SELF_CERTIFIED
    assert comparison.outcome is ComparisonOutcome.RESTATEMENT


def test_the_claim_refuses_the_three_questions_a_comparison_of_two_statements_cannot_answer() -> (
    None
):
    """Acceptance criterion 4, asserted on the rendered claim.

    This is the nearest thing the corpus holds to a statement about whether two people
    agreed, and it is the most misreadable figure the corpus can produce. The caveat is
    the whole of its value, so it lives in ``does_not_mean`` where a renderer prints it
    rather than in a footnote a reader can skip.
    """
    claim = _RELATIONSHIP.claim(build_snapshot([]))
    rendered = claim.render_text()

    assert claim.slug == RATIONALE_RELATIONSHIP_SLUG
    assert claim.kind is ClaimKind.descriptive
    assert claim.granularity is Granularity.team
    assert "It does not say which is correct" in rendered
    assert "whether the disagreement mattered" in rendered
    assert "A divergence is the finding and not a defect in either record" in rendered


def test_a_change_nobody_reconstructed_lands_in_its_only_one_case_rather_than_being_dropped() -> (
    None
):
    """Acceptance criterion 5: the only-one cases are reported, not excluded.

    A change where nobody reconstructed anything is a fact about the corpus. Excluding it
    would make the denominator "changes somebody checked", and a distribution over
    changes somebody checked cannot report how often nobody reconstructed anything at
    all -- the measure would be structurally unable to show its own most common state.
    """
    records = (
        _rationale(pr=1, entry="rationale-v1-0", source="declared", reason="because"),
        _rationale(pr=2, entry="rationale-v1-1", source="reconstructed", reason="because"),
    )
    (whole,) = _distributions(_RELATIONSHIP.compute(build_snapshot(records)))

    assert whole.values["declared_only"] == 1
    assert whole.values["reconstructed_only"] == 1
    assert whole.values["neither"] == 0
    assert whole.excluded == {}


def test_the_buckets_the_exclusions_and_the_denominator_are_all_counted_in_changes() -> None:
    """One unit throughout, so a reviewer can add them up.

    Every distribution in this package holds that invariant, and it is what turns a
    silent filter into something visible. It also forces the population to be *changes*
    rather than documents: a change carrying two rationales contributes one bucket, and
    the second figure reports how many changes a comparison was actually made on so the
    informative rungs are not read against a denominator that includes the absences.
    """
    records = (
        _rationale(pr=1, entry="rationale-v1-0", source="declared", reason="first"),
        _rationale(pr=1, entry="rationale-v1-1", source="reconstructed", reason="second"),
        _rationale(pr=2, entry="rationale-v1-2", source="declared", reason="only mine"),
        _rationale(pr=3, entry="rationale-v1-3", source="declared", reason=None),
        _rationale(pr=3, entry="rationale-v1-4", source="reconstructed", reason=None),
        _rationale(pr=4, entry="rationale-v1-5", source="unknown", reason="fourth"),
    )
    figure = _RELATIONSHIP.compute(build_snapshot(records))
    (whole,) = _distributions(figure)

    assert whole.values["divergent"] == 1
    assert whole.values["declared_only"] == 1
    assert whole.values["reconstructed_only"] == 0
    assert whole.values["neither"] == 0
    assert whole.excluded == {REASON_NOT_READABLE: 1, SOURCE_NOT_READABLE: 1}
    assert whole.counted + whole.excluded_total == whole.claim.denominator.size


def test_the_second_figure_reports_how_many_comparisons_were_actually_made() -> None:
    """A distribution whose informative rungs can be empty needs its denominator.

    Six buckets where three can only ever be non-empty is not readable without knowing
    how many changes could have been compared at all. Kojutsu's own docstring makes
    the same point about a bounded comparison: a report that compared two of five and
    said "no divergences found" is worse than one that names the three it did not reach.
    """
    records = (
        _rationale(pr=1, entry="rationale-v1-0", source="declared", reason="first"),
        _rationale(pr=1, entry="rationale-v1-1", source="reconstructed", reason="second"),
        _rationale(pr=2, entry="rationale-v1-2", source="declared", reason="only"),
    )
    figure = _RELATIONSHIP.compute(build_snapshot(records))
    (compared,) = _counts(figure)
    (whole,) = _distributions(figure)

    assert whole.values["divergent"] == 1
    assert compared.value == 1
    assert (
        compared.value
        == whole.values["divergent"] + whole.values["concurrent"] + (whole.values["restatement"])
    )


def test_a_superseded_revision_is_not_compared_against_the_current_inference() -> None:
    """The revision chain is walked before the comparison, or history becomes divergence.

    ``rationale_revises`` names what a rationale supersedes, so a change can hold a
    second and third draft. Comparing the *first* draft against the current inference
    would manufacture divergences out of editorial history and report them as a finding
    about how two people reasoned about the change.
    """
    records = (
        _rationale(
            pr=1, entry="rationale-v1-0", source="declared", reason="the first draft", revision=1
        ),
        _rationale(
            pr=1, entry="rationale-v1-1", source="declared", reason="the settled reason", revision=2
        ),
        _rationale(
            pr=1, entry="rationale-v1-2", source="reconstructed", reason="the settled reason"
        ),
    )
    (whole,) = _distributions(_RELATIONSHIP.compute(build_snapshot(records)))
    assert whole.values["concurrent"] == 1
    assert whole.values["divergent"] == 0


def test_the_attribution_prose_is_not_compared_as_a_reason() -> None:
    """The body is a rendering, and comparing it whole would report every pair divergent.

    Kojutsu writes a ``## Reason`` section followed by an ``## Attribution`` block
    repeating the principal, the model and the source (``tanseki_mapping.py:264-284``). Two
    bodies compared in full therefore disagree on the attribution line for every pair
    written by different people -- which is precisely the population this figure is
    about -- and the measure would become a restatement detector wearing the name of a
    disagreement detector.
    """
    body = (
        "# Rationale 1\n\n## Reason\nThe queue is bounded.\n\n## Attribution\n"
        "Declared by: agent-one\nDeclared model: model-one (asserted by the author, "
        "not verified by the platform)\nSource: declared\n"
    )
    assert reason_text(body) == "The queue is bounded."
    assert reason_text("## Reason\n\n") is None
    assert reason_text("# Rationale 1\n\nno heading here\n") is None


def test_a_change_with_both_rationales_and_no_readable_reason_is_excluded_rather_than_called_divergent() -> (
    None
):
    """Two sides and no text is not a disagreement; it is an unreadable comparison.

    Bucketing it as divergent would report an absence as a finding -- the same class of
    error as the one this whole module exists to prevent, committed with our own tool.
    """
    records = (
        _rationale(pr=1, entry="rationale-v1-0", source="declared", reason=None),
        _rationale(pr=1, entry="rationale-v1-1", source="reconstructed", reason=None),
    )
    (whole,) = _distributions(_RELATIONSHIP.compute(build_snapshot(records)))
    assert whole.excluded == {REASON_NOT_READABLE: 1}
    assert whole.values["divergent"] == 0
    assert whole.counted + whole.excluded_total == whole.claim.denominator.size


def test_a_rationale_naming_no_change_is_outside_the_population_and_the_denominator_says_so() -> (
    None
):
    """A document is not a change, so it cannot be an exclusion in a change-shaped figure.

    Mixing the two units into one mapping would break the invariant a reviewer adds up.
    So the restriction is written into the population's own description, which is where
    a reader looks for the scope -- and it is stated as what it is: a fact about this
    reader's ability to place the document, not about the corpus.
    """
    snapshot: CorpusSnapshot = build_snapshot(
        (_rationale(pr=1, entry="rationale-v1-0", source="declared", reason="because"),)
    )
    claim = _RELATIONSHIP.claim(snapshot)
    assert "1 change)" in claim.denominator.render_text()

    placed = _RELATIONSHIP.claim(
        build_snapshot(
            (
                _rationale(pr=1, entry="rationale-v1-0", source="declared", reason="because"),
                _rationale(pr=None, entry="rationale-v1-1", source="declared", reason="orphan"),
                _rationale(
                    pr=2, entry="rationale-v1-2", source="declared", reason="x", repo="unknown"
                ),
            )
        )
    )
    assert "2 rationale documents name no repository and pull request pair" in (
        placed.denominator.description
    )
    assert placed.denominator.size == 1


def test_a_capture_that_is_not_a_rationale_never_enters_the_comparison() -> None:
    """The population is rationales, and a projected request is not one.

    Decision 001 requires a decision request to be counted apart from captures; this
    measure counts a fourth kind apart again, and a request that fell into it would put a
    request for a reason into a distribution of statements about reasons.
    """
    records = (
        _capture(entry="answer-0001", ticket="ABC-1"),
        _rationale(pr=1, entry="rationale-v1-0", source="declared", reason="because"),
    )
    (whole,) = _distributions(_RELATIONSHIP.compute(build_snapshot(records)))
    assert whole.counted == 1
    assert whole.values["declared_only"] == 1


def test_an_empty_corpus_reports_every_rung_at_zero_rather_than_rendering_no_figure() -> None:
    """A read holding no rationale is an answer, and it is the shape of the whole ladder.

    The six rungs are materialised at zero for the reason every distribution in this
    package is: a shape that depends on the corpus is one every report has to guard
    against. The denominator is floored at one rather than raising, which is the
    convention for a subset with no records in it.
    """
    figure = _RELATIONSHIP.compute(build_snapshot([]))
    (whole,) = _distributions(figure)
    assert len(whole.values) == 6
    assert sum(whole.values.values()) == 0
    assert whole.claim.denominator.size == 1


def test_the_client_still_holds_no_writer_after_a_traversal_method_was_added() -> None:
    """The guarantee is mechanical, and it is asserted here rather than described.

    A method named ``traverse`` that issued ``POST`` is still a read -- the verb is
    forced by the id, which is path-derived and contains ``/``, exactly as it is for
    ``get_document``. What makes the seam read-only is that this class has no field to
    modify and no method that could, and the check that it has not grown one is a
    property of the class rather than of anybody's reading of its names.
    """
    assert not hasattr(StoreClient, "upsert")
    assert not hasattr(StoreClient, "delete_document")
    assert callable(StoreClient.traverse)
