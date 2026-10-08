"""The read surface's claim: an agent cannot get a number here that this program
would refuse to publish to a person.

A command line has a reader who can be shown a caveat. An API does not: it has a
caller that reads a field and trusts its name, and a field called ``figure`` is a
number. So the properties asserted here are the ones that only start being at risk when
the output stops being a document -- every figure arrives with its claim, no result is a
bare count, the tool set is three, the bound is in the description rather than only in
the payload, and the corpus is reached over the same read-only seam the command line
uses.

**These tests run without the protocol library installed, and that is the point rather
than a convenience.** The tool table is module-level data and the functions that fill it
are pure, so every property above is assertable against
:mod:`tenbin.mcp_server.tools` with no server and no socket. A surface whose guarantees
can only be checked with a dependency present is a surface whose guarantees stop being
checked the moment that dependency is in flight, and
:func:`test_the_surface_imports_with_the_protocol_package_blocked` is here so that stays
true rather than being a fact somebody remembers.

**The store is a :class:`~tests.fakes.FakeStore` in the tests that need one, injected
through ``tenbin.cli.CLIENT_FACTORY`` -- the same seam the command line's own tests
use.** That is not a convenience either: it is the assertion that the MCP surface has no
second door to the corpus, made structural. The suite's socket denial therefore stays a
real denial, and a test that reached the network would raise rather than quietly pass
against a daemon somebody left running.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

from tenbin import cli
from tenbin.claims.gate import ClaimGate
from tenbin.claims.refusals import NO_TICKET_UNBLOCKS_THIS
from tenbin.claims.registry import default_registry
from tenbin.config import settings_from_env
from tenbin.corpus.snapshot import CorpusSnapshot
from tenbin.mcp_server import server as mcp_server
from tenbin.mcp_server.tools import (
    BOUNDED_BY,
    CLAIM_FIELDS,
    EMPTY_IS_NOT_NOTHING,
    INSTRUCTIONS,
    READ_ONLY_HINTS,
    REFUSAL_FIELDS,
    SELF_SELECTION_BOUND,
    TOOL_NAMES,
    TOOLS,
    TOOLS_BY_NAME,
    ToolSpec,
    call,
    claims_result,
    refusals_result,
    refuse_unbound_numbers,
    report_result,
    unbound_numbers,
)
from tenbin.measures import default_measures
from tenbin.report import build_report, render_json
from tests.fakes import FakeOptions, FakeStore, documents_for
from tests.fixtures import build_answers, build_snapshot

if TYPE_CHECKING:
    pass

#: The three names, written out rather than derived, because the whole assertion is that
#: somebody adds a fourth one on purpose. A test that compared the table to a count, or
#: to whatever it was a moment ago, would pass on a surface nobody decided on. Kojutsu
#: pins its own read table to exactly two, for the same reason.
EXPECTED_TOOLS = ("report", "claims", "refusals")

_REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_INIT = _REPO_ROOT / "src" / "tenbin" / "mcp_server" / "__init__.py"
SERVER_SOURCE = _REPO_ROOT / "src" / "tenbin" / "mcp_server" / "server.py"

#: Identifiers a writer would need, asserted against the adapter's own source rather than
#: against the store client: the client's shape is already asserted in
#: ``tests/test_store_client.py``, and what is unasserted is whether this surface can
#: reach anything but that client. Names rather than English words, for the reason
#: Kojutsu's own test uses names -- an annotations check would pass for a tool that
#: posted a comment and was merely marked read-only by mistake, so the check is that the
#: code that could post it is not imported.
WRITE_PATH_NAMES = (
    "upsert",
    "write_read_model",
    "post_issue_comment",
    "TansekiOutbox",
    "KnowledgeSink",
    "capture_server",
    '"POST"',
    '"PUT"',
    '"PATCH"',
    '"DELETE"',
)

#: Vocabulary that would make an argument a request for a comparison or a chosen
#: population, matched anywhere in a name rather than as whole words: the shapes that get
#: added in a hurry are compounds, and a false positive costs an argument name.
COMPARISON_SHAPED = ("compare", "versus", "baseline", "against", "delta", "diff", "peer")


@pytest.fixture
def corpus() -> CorpusSnapshot:
    """A whole read of four records, from the measures layer's own fixture.

    Through :mod:`tests.fixtures` rather than a fake store, because the subject of most
    of these tests is the shape of a result and not the walk that produced the read. The
    tests that are about reaching the store use the fake instead.
    """
    return build_snapshot(build_answers(4))


@pytest.fixture
def store_url(monkeypatch: pytest.MonkeyPatch) -> str:
    """A store URL in the environment, because :mod:`tenbin.config` has no default.

    A measurement program with a default store is a measurement program that reports a
    real number about whichever store happens to be running, and that is as true of this
    surface as it is of the command line.
    """
    url = "http://127.0.0.1:8088/v1"
    monkeypatch.setenv("TENBIN_TANSEKI_URL", url)
    return url


def a_store(documents: dict[str, dict[str, Any]] | None = None, **options: Any) -> FakeStore:
    """A store that answers in process, and is never retried so a test stays fast."""
    return FakeStore(documents, options=FakeOptions(max_retries=0, **options))


def prose_of(path: Path) -> str:
    """A source file with its line breaks collapsed, for asserting on what it says.

    An argument written for a reader is wrapped at a column width, and a test that
    matches a phrase across that wrap is a test that fails the day somebody rewraps a
    paragraph. Collapsing the whitespace is what makes the assertion about the argument
    rather than about the typesetting, which is the thing under test.
    """
    return " ".join(path.read_text(encoding="utf-8").split())


def use_store(monkeypatch: pytest.MonkeyPatch, store: FakeStore) -> FakeStore:
    """Hand both surfaces this store, through the one seam that reaches a socket."""
    monkeypatch.setattr(cli, "CLIENT_FACTORY", lambda settings: store)
    return store


def refuse_store(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make reaching the corpus impossible, so "was never called" is assertable.

    The factory raises rather than returning a fake, so a test that reaches the store
    fails with this assertion instead of with a network error the reader would have to
    interpret -- which is the whole reason the suite denies sockets.
    """

    def unreachable(settings: Any) -> FakeStore:
        raise AssertionError(f"the store must not be reached, and it was asked for {settings}")

    monkeypatch.setattr(cli, "CLIENT_FACTORY", unreachable)


def unconfigured() -> Any:
    """Settings for a test whose subject is that no configuration was consulted.

    Built with a loopback URL rather than from the environment, so a developer with
    ``TENBIN_TANSEKI_URL`` exported cannot change the outcome of a test about the surface
    answering without one.
    """
    return settings_from_env(tanseki_url="http://127.0.0.1:8088/v1")


def run_report_tool(**arguments: Any) -> dict[str, Any]:
    """Call the report tool the way the adapter does, from the configured environment."""
    return mcp_server._tool_result(
        TOOLS_BY_NAME["report"], settings=settings_from_env(), **arguments
    )


# -- the tool set, pinned --------------------------------------------------------------


def test_the_tool_table_is_exactly_three_tools_because_every_tool_added_is_a_claim_about_the_corpus_somebody_can_act_on() -> (
    None
):
    """Three, by name, in the order they are offered, with the functions they name.

    Not tidiness. A fourth tool is somebody's decision about what this program will let an
    agent conclude, and the refusal catalogue in :mod:`tenbin.claims.refusals` is longer
    than this table, which is the honest measure of how much this corpus supports.
    ``report`` is first because it is the useful one.

    The producing functions are pinned to their names as well, so a tool cannot be
    quietly re-pointed at a different function, and the lookup is asserted to be the same
    tuple rather than a second registry: there is no decorator anywhere to add a tool to,
    and :func:`tenbin.mcp_server.tools.call` is the only way a descriptor is reached.
    """
    assert TOOL_NAMES == EXPECTED_TOOLS
    assert [spec.name for spec in TOOLS] == list(EXPECTED_TOOLS)
    assert {spec.name for spec in TOOLS} == set(EXPECTED_TOOLS)
    assert set(TOOLS_BY_NAME) == set(EXPECTED_TOOLS)
    assert TOOLS_BY_NAME["report"].produce is report_result
    assert TOOLS_BY_NAME["claims"].produce is claims_result
    assert TOOLS_BY_NAME["refusals"].produce is refusals_result
    assert TOOLS_BY_NAME["report"] is TOOLS[0], "the table and the lookup must not be two tables"


def test_a_tool_whose_result_is_not_an_object_cannot_be_built_because_a_bare_number_is_the_output_this_program_refuses() -> (
    None
):
    """The shape is refused at construction, not merely asserted in a test.

    A result schema is a description of intent, and the check has to be on the thing that
    ships: ``{"count": 412}`` passes both a review and a schema. It does not pass this.
    """
    with pytest.raises(ValueError, match="object at its top level"):
        ToolSpec(
            name="count",
            description="A count.",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            result_schema={"type": "integer"},
            produce=lambda snapshot: {"count": 1},
        )


def test_a_schema_holding_something_json_cannot_carry_is_refused_when_the_table_is_built_because_the_protocol_only_fails_when_a_host_tries_to_serialise_it() -> (
    None
):
    """The copy that makes the frozen dataclass honest is also the schema's type check.

    :func:`tenbin.mcp_server.tools.ToolSpec.__post_init__` accepts objects, arrays,
    strings, numbers, booleans and null, and refuses anything else. Without that, a schema
    holding an enum member or a callable would build cleanly here and fail in a host four
    attributes away, which is the kind of defect nobody can find from the traceback.
    """
    with pytest.raises(ValueError, match="cannot carry"):
        ToolSpec(
            name="broken",
            description="A schema with something a JSON object cannot hold.",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            result_schema={"type": "object", "properties": {"x": object()}},
            produce=lambda snapshot: {"x": 1},
        )


def test_every_tool_declares_itself_read_only_because_the_shape_of_the_surface_is_the_guarantee() -> (
    None
):
    """Read-only hints on every tool, as data in the table rather than in the adapter.

    Plain mapping in :mod:`tenbin.mcp_server.tools`, so the declaration is assertable
    with no protocol library present. The hints are a promise a host reads; the structural
    version of the same promise is
    :func:`test_the_server_reaches_the_corpus_through_the_command_lines_own_read_only_seam_and_holds_no_write_path`,
    and a surface with only the promise is the failure the promise is for.
    """
    assert dict(READ_ONLY_HINTS) == {
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    }
    assert READ_ONLY_HINTS["readOnlyHint"] is True
    assert READ_ONLY_HINTS["destructiveHint"] is False


# -- every figure arrives with its claim -----------------------------------------------


def test_a_figure_arrives_with_its_statement_its_negative_space_its_falsifier_and_its_denominator_because_an_agent_reads_a_field_and_trusts_its_name(
    corpus: CorpusSnapshot,
) -> None:
    """Every rendered section, of both kinds, carries what makes it interpretable.

    Asserted over the whole list rather than a sample, because the defect is per-section:
    a schema and a code path that are right for the eleven measures somebody tested and
    wrong for the twelfth. A figure must carry all four of
    :data:`~tenbin.mcp_server.tools.CLAIM_FIELDS`. A refusal must carry its reason, its
    missing fact and its unblocking, and must *not* carry a claim -- showing the rejected
    argument beside the rejection is the contradiction
    :class:`~tenbin.report.base.Section` refuses to be built with, and a payload that
    reintroduced it would be a second way to publish a refused claim.
    """
    sections = report_result(corpus)["sections"]
    figures = [section for section in sections if section["kind"] == "claim"]
    refusals = [section for section in sections if section["kind"] == "refusal"]

    assert figures and refusals, "both shapes must appear, or one half of this is vacuous"

    for figure in figures:
        for field in CLAIM_FIELDS:
            assert isinstance(figure.get(field), str) and figure[field].strip(), (
                f"{figure['slug']} arrived without a {field}; a figure with no caveat beside "
                "it is a claim made silently"
            )
    for refusal in refusals:
        for field in REFUSAL_FIELDS:
            assert field in refusal, f"{refusal['slug']} arrived without a {field}"
        assert isinstance(refusal["reason"], str) and refusal["reason"].strip()
        assert "statement" not in refusal
        assert "falsifier" not in refusal, "a refusal has no falsifier; it asserts no number"


def test_the_report_result_carries_the_completeness_of_the_read_because_a_distribution_over_four_hundred_looks_exactly_like_one_over_four_thousand(
    corpus: CorpusSnapshot,
) -> None:
    """The header says how much arrived, and the verdict is a word rather than a boolean.

    Four verdicts are four different findings, and a boolean would collapse a truncated
    read and a failed walk into one indistinguishable "not complete" -- which is the same
    mistake as a bare count, one level up. ``truncation`` is asserted present-and-null
    rather than absent, because "the walk was whole" and "nobody recorded whether it was"
    are different facts and only the first is true.
    """
    result = report_result(corpus)

    assert result["corpus"]["enumerated"] == 4
    assert result["corpus"]["store_total"] == 4
    assert result["corpus"]["completeness"] == "complete"
    assert "truncation" in result["corpus"]
    assert result["corpus"]["truncation"] is None
    assert not list(unbound_numbers(result)), (
        "the two counts are numbers, and they are publishable only because the object "
        "around them states the read's verdict"
    )


def test_the_report_payload_is_the_command_lines_own_json_because_two_serialisations_of_one_report_would_drift(
    corpus: CorpusSnapshot,
) -> None:
    """Byte-for-byte the payload the command line renders for the same read.

    The whole reason the report tool is thin.
    :func:`~tenbin.mcp_server.tools.report_result` calls
    :func:`~tenbin.report.json_.report_as_dict`, the value behind
    :func:`~tenbin.report.json_.render_json`, so there is no second serialisation here to
    relax a caveat in. It is the reasoning that makes the Markdown and JSON renderers share
    :func:`~tenbin.report.ordering.ordered_parts`: two writers of one format is two
    things to keep true, and the moment somebody relaxes one of them the two surfaces
    disagree about what a number means.
    """
    rendered = build_report(
        corpus, gate=ClaimGate(), registry=default_registry(), measures=default_measures()
    )

    assert json.loads(render_json(rendered)) == report_result(corpus)


def test_the_report_tool_refuses_to_build_without_a_read_because_a_store_that_was_never_reached_and_a_store_holding_nothing_render_the_same_zeroes() -> (
    None
):
    """No read, no report -- raised, rather than returned as an empty document.

    The command line settles this by exiting non-zero having written nothing. A tool has no
    exit code, so the same distinction has to be made by refusing: an empty report is a
    page of zeroes that renders as an answer, and it is the most dangerous output this
    program can emit.
    """
    with pytest.raises(ValueError, match="no read behind it"):
        report_result(None)


def test_integrity_only_selects_the_measures_about_the_capture_system_because_a_caller_asking_whether_to_believe_the_rest_does_not_want_the_rest(
    corpus: CorpusSnapshot,
) -> None:
    """The same four capture-system claims the command line's flag selects.

    Asserted against slugs rather than by calling the command line's own private helper:
    what matters is that both surfaces pick the same measures, and a slugs assertion is a
    behavioural statement that survives either of them being refactored into a published
    function. The refusals stay on an integrity report, because "what did I not get" is the
    question somebody asking this actually has.
    """
    integrity = report_result(corpus, integrity_only=True)
    full = report_result(corpus)

    rendered = [s["slug"] for s in integrity["sections"] if s["kind"] == "claim"]
    assert rendered == [
        "corpus.read_completeness",
        "corpus.lifecycle_shape_checked",
        "corpus.capture_freshness_checked",
        "corpus.trust_profile",
        "corpus.projection_safety_checked",
        "corpus.question_capture_agreement_checked",
    ]
    assert len(integrity["sections"]) < len(full["sections"])
    assert [s for s in integrity["sections"] if s["kind"] == "refusal"] == [
        s for s in full["sections"] if s["kind"] == "refusal"
    ]


# -- no tool returns a bare count ------------------------------------------------------


def test_no_tool_returns_a_bare_count_because_return_len_records_is_one_line_and_a_schema_makes_it_look_deliberate(
    corpus: CorpusSnapshot,
) -> None:
    """Three assertions, because a schema alone is not the guarantee.

    First, every result schema declares an object at its top level, so a ``count`` tool
    cannot be declared whatever its function does. Second, none of the three real results
    contains a number that no claim, reason, verdict or population description accounts
    for -- checked by :func:`~tenbin.mcp_server.tools.unbound_numbers` over the actual
    output rather than over a description of it. Third, the check catches the shapes that
    get written, so the first two are not passing vacuously.
    """
    for spec in TOOLS:
        assert spec.result_schema["type"] == "object", f"{spec.name} does not return an object"
        result = call(spec, snapshot=corpus)
        assert isinstance(result, dict), f"{spec.name} returned {type(result).__name__}"
        assert not list(unbound_numbers(result)), (
            f"{spec.name} would return a number that no claim, reason, verdict or "
            "population description accounts for"
        )

    # The shapes that actually get written, so the two assertions above are not passing
    # because the check does nothing.
    assert list(unbound_numbers({"count": 412})) == ["$"]
    assert list(unbound_numbers({"decisions_recorded": 412})) == ["$"]
    assert list(unbound_numbers({"sections": [{"count": 3}]})) == ["$.sections[0]"]
    assert list(unbound_numbers({"buckets": [1, 2, 3]})) == ["$.buckets"]
    with pytest.raises(ValueError, match="as a bare number"):
        refuse_unbound_numbers({"count": 412}, "count")


def test_a_denominator_size_is_legal_because_it_sits_beside_the_population_it_counts_because_a_size_with_no_description_cannot_be_checked_at_all() -> (
    None
):
    """The check accepts a bounded number and refuses an unbounded one.

    :data:`~tenbin.mcp_server.tools.BOUNDED_BY` is what makes this a rule rather than a
    blanket ban on numbers: a figure is made of counts, and the rule is that a count has to
    arrive inside an object that says what population it counts. The corpus header is the
    other case worth having -- ``enumerated`` beside ``completeness`` is how a reader learns
    a count is conditional on the walk finishing.
    """
    assert "description" in BOUNDED_BY
    assert "completeness" in BOUNDED_BY
    assert not list(unbound_numbers({"description": "records in this read", "size": 4}))
    assert not list(
        unbound_numbers({"completeness": "complete", "enumerated": 4, "store_total": 4})
    )
    assert list(unbound_numbers({"size": 4})), "a size with no population is still a count"


def test_every_schema_is_plain_json_and_cannot_be_edited_through_the_spec_that_holds_it_because_a_frozen_dataclass_holding_a_dict_is_not_frozen(
    corpus: CorpusSnapshot,
) -> None:
    """The schemas round-trip, and the copy is deep rather than a wrapper over one dict.

    Two properties, and the second is the one that bites later. Round-tripping proves the
    copy is plain JSON with no shared structure; attempting to assign proves the spec is
    honestly frozen, which is the same correction
    :class:`~tenbin.store.client.StoreDocument` makes for the same reason -- a caller who
    could change a schema after the table was built would be changing the surface without
    changing the table.
    """
    for spec in TOOLS:
        for schema in (spec.input_schema, spec.result_schema):
            assert json.loads(json.dumps(dict(schema))) == dict(schema)
        with pytest.raises(TypeError):
            cast("Any", spec.result_schema)["type"] = "integer"

    result = report_result(corpus)
    assert json.loads(json.dumps(result)) == result


# -- the refusals, which is the tool the ticket is for ---------------------------------


def test_the_refusals_tool_returns_the_whole_registry_with_its_reasons_and_its_unblocking_because_an_agent_must_tell_a_measure_that_found_nothing_from_a_measure_that_does_not_exist() -> (
    None
):
    """Every entry, verbatim, with ``null`` distinguished from a ticket.

    Two things are protected. The agent gets the whole catalogue rather than a count of it,
    because a count of refusals is a number with no argument attached and the arguments are
    the point. And an entry with no ticket says so in prose, because ``null`` read as
    "nobody has written it down" is how a reader ends up waiting for work that was never
    going to be filed -- which is the specific way one of this project's decisions turns
    back into a queue item.
    """
    result = refusals_result()
    registry = default_registry()

    assert [entry["claim_slug"] for entry in result["refusals"]] == [
        refusal.claim_slug for refusal in registry.all()
    ]
    for entry, refusal in zip(result["refusals"], registry.all(), strict=True):
        assert entry["reason"] == refusal.reason, "the reason is the argument, verbatim"
        assert entry["title"] == refusal.title
        assert "missing_fact" in entry, "an absent key and a null key are different facts"
        assert entry["unblocked_by"] == refusal.unblocked_by
        assert entry["unblocking"].strip()
        if refusal.unblocked_by is None:
            assert entry["unblocking"] == NO_TICKET_UNBLOCKS_THIS

    without_tickets = [r for r in registry.all() if r.unblocked_by is None]
    assert without_tickets, "the catalogue must contain a decision, or this is vacuous"
    assert "whether-agentic-development-is-more-effective" in {
        entry["claim_slug"] for entry in result["refusals"]
    }
    assert result["mechanism"]["assignment_mechanism"] is None
    assert result["mechanism"]["unset_reason"].strip(), (
        "a comparison is refused because the mechanism is absent, and the reason has to "
        "travel with the refusal or a reader goes looking for the missing measurement"
    )


def test_the_two_reasoning_tools_reach_no_store_because_a_refusal_you_cannot_see_without_a_deployment_is_a_property_of_that_deployment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``claims`` and ``refusals`` answer with nothing configured at all.

    The factory here raises rather than returning a fake, so the assertion is not "the fake
    was never called" but "nothing could have been called". It is the reasoning ``tenbin
    claims`` and ``tenbin refusals`` are built on: the catalogue has to be inspectable
    before there is anything to inspect, or it is a property of a deployment rather than of
    the program.

    Both are run through the adapter's dispatcher rather than the functions directly, so
    the branch that skips the read is exercised rather than merely present -- and a
    dispatcher that stopped skipping it would fail here rather than on a deployment.
    """
    refuse_store(monkeypatch)

    claims = mcp_server._tool_result(TOOLS_BY_NAME["claims"], settings=unconfigured())
    refusals = mcp_server._tool_result(TOOLS_BY_NAME["refusals"], settings=unconfigured())

    assert claims["claims"] and refusals["refusals"]
    assert refusals["refusals"] == refusals_result()["refusals"]


def test_the_claims_tool_lists_every_claim_with_its_four_fields_because_a_claim_listed_without_its_negative_space_is_an_argument_with_the_limitation_deleted(
    corpus: CorpusSnapshot,
) -> None:
    """All four fields on all of them, and one entry per measure.

    The claim is this program's reasoning rather than a summary of it, and a reasoning with
    its "what it does not mean" stripped out is an argument somebody can now quote. Asserted
    over the whole list rather than a sample, because the entry that loses a caveat is
    whichever one nobody looked at.
    """
    result = claims_result(corpus)

    assert len(result["claims"]) == len(default_measures())
    assert "corpus.read_completeness" in {entry["slug"] for entry in result["claims"]}
    for entry in result["claims"]:
        for field in ("statement", "does_not_mean", "falsifier"):
            assert isinstance(entry.get(field), str) and entry[field].strip(), (
                f"{entry['slug']} was listed without a {field}"
            )
        # The fourth field is typed rather than rendered: a description so the population
        # can be reproduced and a size so it can be contradicted, which a sentence would
        # let a consumer do neither with.
        assert entry["denominator"]["description"].strip()
        assert "size" in entry["denominator"] and "size_note" in entry["denominator"]
        assert set(CLAIM_FIELDS) <= set(entry)
        assert entry["claim_kind"] and entry["granularity"]


def test_claims_states_a_denominator_size_only_when_a_read_supplied_one_because_a_size_over_an_empty_read_is_a_number_nobody_wrote_down(
    corpus: CorpusSnapshot,
) -> None:
    """The description always; the size only with a read behind it.

    Every denominator in this program is a count over one particular read, so a ``claims``
    listing with no read has to either invent a size or say there is none -- and inventing
    one from the empty corpus standing in for a read would put a number under every claim
    that nobody wrote down. The sentences are :data:`tenbin.cli.DENOMINATOR_FROM_READ` and
    the ``read.taken`` flag, imported rather than restated, so the command line and an
    agent's copy of this payload cannot drift into saying two different things about a
    missing size.
    """
    unsized = claims_result(None)
    sized = claims_result(corpus)

    assert unsized["read"] == {
        "taken": False,
        "read_at": None,
        "collection": None,
        "completeness": None,
    }
    for entry in unsized["claims"]:
        assert entry["denominator"]["size"] is None, "a size over the read that never happened"
        assert entry["denominator"]["size_note"] == cli.DENOMINATOR_FROM_READ
        assert entry["denominator"]["description"].strip()

    assert sized["read"]["taken"] is True
    assert sized["read"]["completeness"] == "complete"
    assert sized["read"]["read_at"] == corpus.read_at.isoformat()
    for entry in sized["claims"]:
        assert entry["denominator"]["size_note"] is None
        assert isinstance(entry["denominator"]["size"], int)


# -- the bound is in the description, before the call -----------------------------------


def test_every_description_states_what_the_numbers_are_not_and_that_the_corpus_is_self_selected_because_the_description_is_the_part_an_agent_reads_before_calling() -> (
    None
):
    """The bound, in prose, in all three descriptions and in the server's instructions.

    A caveat-first layout was argued for a reader who goes top to bottom. An agent does
    not: it reads a field and trusts its name, and it decides whether to call a tool from
    the description alone. So this is the only place the negative space reaches a machine
    that will never render the payload -- which is why the bound lives here rather than only
    inside the results, and why it is one sentence written into all three descriptions
    rather than each tool phrasing its own.
    """
    for spec in TOOLS:
        assert SELF_SELECTION_BOUND in spec.description, (
            f"{spec.name}'s description does not state that the corpus is self-selected"
        )
        assert "self-selected" in spec.description
        assert "not a statement about development" in spec.description

    assert SELF_SELECTION_BOUND in INSTRUCTIONS
    assert INSTRUCTIONS.startswith("Tenbin measures a Kojutsu corpus")
    assert "refusals" in INSTRUCTIONS, (
        "the first instruction an agent reads should be to ask what is not measurable"
    )
    assert EMPTY_IS_NOT_NOTHING in TOOLS_BY_NAME["report"].description


def test_no_tool_offers_a_population_because_a_filter_argument_is_how_a_surface_acquires_a_group_this_corpus_cannot_support() -> (
    None
):
    """No argument anywhere chooses who is counted or what is compared.

    The same rule ``tests/test_cli.py`` applies to flags, for the same reason and one step
    further: a comparison needs an assignment mechanism this corpus does not have
    (:mod:`tenbin.claims.mechanism`), so an argument shaped like a filter or a baseline is
    an argument that will be passed by somebody in a hurry and refused at the far end -- or
    worse, not refused. The two reasoning tools take no arguments at all, and
    ``additionalProperties: False`` is what makes that a fact rather than an intention.
    """
    for spec in TOOLS:
        names = tuple(spec.input_schema.get("properties", {}))
        assert spec.input_schema.get("additionalProperties") is False, (
            f"{spec.name} would accept an argument its table does not declare"
        )
        for name in (*names, spec.name):
            for word in COMPARISON_SHAPED:
                assert word not in name.casefold(), f"{spec.name} offers {name!r}"
        if spec.name != "report":
            assert names == (), f"{spec.name} takes arguments: {names}"

    report_arguments = TOOLS_BY_NAME["report"].input_schema["properties"]
    assert tuple(report_arguments) == ("integrity_only",)
    assert report_arguments["integrity_only"]["type"] == "boolean"


# -- the read seam, and no write path --------------------------------------------------


def test_the_server_reaches_the_corpus_through_the_command_lines_own_read_only_seam_and_holds_no_write_path_because_a_measurement_program_that_can_edit_the_corpus_it_measures_is_measuring_a_corpus_it_may_have_edited(
    monkeypatch: pytest.MonkeyPatch, store_url: str
) -> None:
    """One door -- the command line's -- and no name in this module that could open another.

    Three assertions, each a different kind of evidence. The factory is patched on
    :mod:`tenbin.cli` and the adapter reads it there at call time, so a test that can hand
    the command line a store in process has handed the server one too: that is the
    structural claim, and it is why the adapter imports the module rather than the name.
    The fake records what it served, so the report tool is asserted to have actually read,
    and every request it saw is checked for being a read -- the store client's own shape is
    asserted in ``tests/test_store_client.py``; what is unasserted is whether this surface
    could reach anything but that client. And the adapter's source is read for the
    identifiers a writer would need, because an annotations check would pass for a tool
    that posted a comment and was merely marked read-only by mistake.
    """
    store = use_store(monkeypatch, a_store(documents_for(4)))

    result = run_report_tool()

    assert "error" not in result, result
    assert store.requests, "the fake must have been read for the assertion to mean anything"
    answered = {(request.method, request.url.path) for request in store.requests}
    assert answered <= {("GET", "/v1/documents"), ("POST", "/v1/documents:query")}, (
        f"this surface asked for something other than a read: {sorted(answered)}"
    )
    assert result["corpus"]["enumerated"] == 4

    source = SERVER_SOURCE.read_text(encoding="utf-8")
    for name in WRITE_PATH_NAMES:
        assert name not in source, (
            f"the read adapter references {name!r}. It holds no write capability today; a "
            "measurement program that can modify the corpus it measures is measuring a "
            "corpus it may have edited, and the edit appears in no figure it later reports."
        )
    assert "tenbin.readmodel" not in source, (
        "the read model has a writer, and a read surface that imported it would be one "
        "refactor away from being able to keep a corpus"
    )


def test_an_unreachable_store_comes_back_as_a_refusal_naming_what_did_not_happen_because_a_tool_has_no_exit_code_and_a_document_of_zeroes_is_an_answer(
    monkeypatch: pytest.MonkeyPatch, store_url: str
) -> None:
    """A failure is a refusal, in its own payload, and never a report.

    The command line draws the finding-versus-failure line with an exit code and by writing
    nothing. A tool has neither, so the line has to be drawn inside the result: an error
    that reads like an empty result is the one thing an agent cannot recover from, because it
    will store it as a measurement of a corpus that was never read. So the refusal says in
    its own payload that nothing was measured, and the result has no ``corpus`` and no
    ``sections`` for a caller to mistake for a report of nothing.
    """
    use_store(monkeypatch, a_store(fail_status=503))

    result = run_report_tool()

    assert result["error"] == "corpus_unreadable"
    assert "not read" in result["what_this_is_not"]
    assert "not an empty corpus" in result["what_this_is_not"]
    assert "corpus" not in result, "a failure must not render a corpus, even an empty one"
    assert "sections" not in result


def test_a_defect_in_a_tool_comes_back_as_a_refusal_rather_than_a_traceback_because_a_published_number_is_worse_than_a_named_failure(
    monkeypatch: pytest.MonkeyPatch, store_url: str
) -> None:
    """A ``ValueError`` from the table is reported, not raised into the protocol.

    The two the table raises are a report asked for with no read and a result carrying a
    bare number -- both defects in this package rather than conditions of the corpus, and a
    host that surfaces a traceback is unusable while a host that swallows the message and
    returns ``{}`` is worse. So the message is the payload, and the refusal still says that
    nothing was measured.

    The defect is injected rather than waited for, because a test that only fires when the
    package is already broken is a test that fires after the damage.
    """

    use_store(monkeypatch, a_store(documents_for(2)))

    def bare_count(snapshot: CorpusSnapshot | None) -> dict[str, Any]:
        # The gate is what a real result function calls, so the injection reproduces the
        # whole chain rather than skipping the step under test.
        return dict(refuse_unbound_numbers({"count": 412}, "report"))

    leaky = ToolSpec(
        name="report",
        description=TOOLS_BY_NAME["report"].description,
        input_schema=dict(TOOLS_BY_NAME["report"].input_schema),
        result_schema=dict(TOOLS_BY_NAME["report"].result_schema),
        produce=bare_count,
    )

    result = mcp_server._tool_result(leaky, settings=settings_from_env())

    assert result["error"] == "refused"
    assert "as a bare number" in result["message"]
    assert "not read" in result["what_this_is_not"]


# -- the objection, recorded rather than dismissed -------------------------------------


def test_the_objection_to_exposing_a_measurement_program_over_mcp_is_written_down_and_answered_because_kojutsu_made_the_same_argument_and_resolved_it_by_making_the_log_local() -> (
    None
):
    """The strongest case against this package is in the package, before the code.

    Not a mention -- a paragraph, and an answer that names the precedent. An objection
    answered in a comment is an objection that comes back the next time somebody wants a
    fourth tool, and the reason to write it down is that this particular argument is *good*:
    the corpus was captured with no expectation of being summarised for third parties, and an
    API removes the friction a report's caveats depend on. The answer is not that the
    objection is wrong. It is that the choice is not between the figures and silence, that
    the refusal catalogue names no record and no person, and that an agent which cannot ask
    what is not measurable will infer that everything is.

    The precedent is asserted too, because an answer that cites a resolution rather than
    re-arguing it is the difference between this being a decision and being a preference.
    """
    text = prose_of(PACKAGE_INIT)

    for phrase in (
        "no MCP surface at all",
        "third parties",
        "machine-readable invitation to summarise",
    ):
        assert phrase in text, f"the objection is missing: {phrase!r}"

    for phrase in (
        "not between exposing the corpus and",
        "names no record, no person, and no fact",
        "cannot ask will infer that everything is askable",
    ):
        assert phrase in text, f"the answer is missing: {phrase!r}"

    for phrase in ("read_log.py", "unauthenticated", "single trust domain"):
        assert phrase in text, f"the precedent is missing: {phrase!r}"

    # Before the code, not after it: the first line of the file is prose, so the objection
    # is not something a reader reaches only after the tool table.
    first = PACKAGE_INIT.read_text(encoding="utf-8").splitlines()[0]
    assert first.startswith('"""Tenbin over MCP')


# -- the dependency, and the deferral that makes this file runnable ---------------------


def test_the_surface_imports_with_the_protocol_package_blocked_because_a_test_that_cannot_run_offline_stops_being_run() -> (
    None
):
    """A subprocess with ``mcp`` blocked, so this holds whether or not it is installed.

    Every other test here imports :mod:`tenbin.mcp_server.tools` and never touches the
    adapter, which is only true if importing the package does not drag the protocol in.
    Rather than trust that, both modules are imported in a child process with the protocol
    blocked from ``sys.meta_path``, so the assertion holds in a developer's environment and
    in CI's, and keeps holding the day ``mcp`` is installed. The child prints the tool names,
    because the useful failure here is somebody's own ``ModuleNotFoundError`` rather than
    the protocol's.
    """
    script = (
        "import sys\n"
        "class Blocked:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'mcp' or name.startswith('mcp.'):\n"
        "            raise ImportError('mcp is blocked for this test')\n"
        "        return None\n"
        "sys.meta_path.insert(0, Blocked())\n"
        "import tenbin.mcp_server as package\n"
        "import tenbin.mcp_server.server as adapter\n"
        "print(','.join(package.TOOL_NAMES))\n"
        "print(adapter.SERVER_NAME)\n"
    )
    env = {key: value for key, value in os.environ.items() if not key.startswith("TENBIN_")}
    env["PYTHONPATH"] = str(_REPO_ROOT / "src")
    env["TENBIN_ENV_FILE"] = ""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
        env=env,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    lines = completed.stdout.splitlines()
    assert lines == [",".join(EXPECTED_TOOLS), mcp_server.SERVER_NAME]


def test_a_missing_protocol_library_says_which_line_to_add_rather_than_leaving_a_traceback_as_the_first_thing_an_operator_sees() -> (
    None
):
    """The instruction names the constraint, the file, and the follow-up command.

    Three things a message at seven in the morning has to carry: what is missing, which line
    fixes it, and that the reasoning does not depend on it. ``mcp>=2,<3`` is a range rather
    than a pin because the adapter uses the table's own schemas and reads nothing else from
    the SDK, so a major bump has nothing here to break -- and a pin that quietly went stale
    would be a worse failure than a clear error.
    """
    assert "mcp>=2,<3" in mcp_server.MISSING_DEPENDENCY
    assert "[project] dependencies" in mcp_server.MISSING_DEPENDENCY
    assert "pyproject.toml" in mcp_server.MISSING_DEPENDENCY
    assert "uv lock" in mcp_server.MISSING_DEPENDENCY
    assert "importable without it" in mcp_server.MISSING_DEPENDENCY


def test_the_adapter_registers_the_table_verbatim_because_every_tool_comes_from_the_table_and_the_table_is_the_only_place_one_can_be_added(
    store_url: str,
) -> None:
    """Names, descriptions, schemas and annotations are the table's, not the adapter's.

    Walked rather than grepped, so this holds for a reworded description and a changed
    schema as well as for a new name. A surface whose wording is written twice is a surface
    whose wording will differ, and the difference is the part an agent reads before it acts.

    Both registration shapes the SDK has taken are exercised, against a fake that counts,
    because the two differ only in how the arguments arrive and a version bump between them
    is exactly the moment a tool would quietly stop being registered. The count is
    asserted rather than just the names: an adapter that declared a tool *and* added it
    would answer every call correctly and show the caller the tool twice, which is the
    kind of surface defect nobody goes looking for.
    """
    declared: list[dict[str, Any]] = []

    class FakeSdkServer:
        """Both shapes the SDK's registration call has taken, counted separately."""

        def __init__(self) -> None:
            self.added: list[dict[str, Any]] = []

        def tool(self, **kwargs: Any) -> Any:
            def decorate(handler: Any) -> Any:
                declared.append(kwargs)
                return handler

            return decorate

        def add_tool(self, handler: Any, **kwargs: Any) -> None:
            self.added.append(kwargs)

    decorator_server = FakeSdkServer()
    for spec in TOOLS:
        assert mcp_server._register(decorator_server, spec, dict(READ_ONLY_HINTS)) is None
    assert [entry["name"] for entry in declared] == list(EXPECTED_TOOLS)
    assert not decorator_server.added, "a tool the decorator registered must not also be added"

    add_tool_server = FakeSdkServer()
    add_tool_server.tool = None  # type: ignore[method-assign]
    for spec in TOOLS:
        assert mcp_server._register(add_tool_server, spec, dict(READ_ONLY_HINTS)) is None
    assert [entry["name"] for entry in add_tool_server.added] == list(EXPECTED_TOOLS)
    assert not declared[3:], "the second shape must not reach the decorator"

    for spec, entry in zip(TOOLS, declared[:3], strict=True):
        assert entry["description"] == spec.description
        assert entry["outputSchema"] == dict(spec.result_schema)
        assert entry["inputSchema"] == dict(spec.input_schema)
        assert entry["annotations"]["readOnlyHint"] is True
        assert entry["annotations"]["destructiveHint"] is False


def test_the_adapter_module_names_no_client_because_a_surface_that_could_reach_the_corpus_itself_would_have_a_second_door() -> (
    None
):
    """The adapter's public names, pinned, and no client type among them.

    ``build_server`` and ``main`` are the whole of it. :class:`StoreClient` is imported
    under ``TYPE_CHECKING`` precisely so the assertion is structural rather than a matter of
    what the module happens to do: the namespace contains no client to pass anywhere, and a
    reader asking what this surface can reach finds only the error types and the command
    line's own seam.
    """
    public = {name for name in dir(mcp_server) if not name.startswith("_")}

    assert {"build_server", "main", "SERVER_NAME", "MISSING_DEPENDENCY"} <= public
    assert "StoreClient" not in public, "the adapter borrows a client; it must not name one"
    assert not [name for name in public if "write" in name.casefold()]

    source = SERVER_SOURCE.read_text(encoding="utf-8")
    for name in WRITE_PATH_NAMES:
        assert name not in source, (
            f"the read adapter references {name!r}. It holds no write capability today; a "
            "measurement program that can modify the corpus it measures is measuring a "
            "corpus it may have edited, and the edit appears in no figure it later reports."
        )
    assert "tenbin.readmodel" not in source, (
        "the read model has a writer, and a read surface that imported it would be one "
        "refactor away from being able to keep a corpus"
    )


def test_the_package_re_exports_the_table_so_a_host_can_inspect_the_surface_without_the_protocol() -> (
    None
):
    """The design is importable as data, which is what makes it assertable at all.

    A reviewer asking "what does this surface expose" gets a tuple, and the tests above are
    written against that tuple rather than against a live server.
    """
    package = importlib.import_module("tenbin.mcp_server")

    assert package.TOOL_NAMES == EXPECTED_TOOLS
    assert package.TOOLS is TOOLS
    assert "ToolSpec" in package.__all__
    assert all(name in package.__all__ for name in ("unbound_numbers", "refuse_unbound_numbers"))


def test_the_published_names_of_the_tool_module_are_the_table_and_the_functions_because_a_helper_a_caller_can_reach_is_an_api_nobody_reviewed() -> (
    None
):
    """What :mod:`tenbin.mcp_server.tools` publishes, named and counted.

    The public surface here is larger than the adapter's and deliberately so -- it is a
    library the tests are written against, not an entry point. It is pinned anyway, because
    a helper added for one caller is the first half of a function somebody will use from
    somewhere, and a measurement program accumulates those without anybody deciding to.

    ``__all__`` and not ``dir()``: the point is what the module *declares* it offers, and
    ``dir()`` would also report the eleven names it imported to build the table, which are
    not its surface and are asserted elsewhere to be the packages' own.
    """
    tools_module = importlib.import_module("tenbin.mcp_server.tools")

    assert set(tools_module.__all__) == {
        "BOUNDED_BY",
        "CLAIM_FIELDS",
        "EMPTY_IS_NOT_NOTHING",
        "GRANULARITIES",
        "INSTRUCTIONS",
        "READ_ONLY_HINTS",
        "REFUSAL_FIELDS",
        "REFUSAL_TOOL_BOUND",
        "SELF_SELECTION_BOUND",
        "TOOLS",
        "TOOLS_BY_NAME",
        "TOOL_NAMES",
        "ToolSpec",
        "call",
        "claims_result",
        "refuse_unbound_numbers",
        "refusals_result",
        "report_result",
        "unbound_numbers",
    }


# -- the guards that make the table a design rather than a list ------------------------


def test_a_tool_with_a_blank_name_or_a_blank_description_cannot_be_built_because_a_tool_an_agent_cannot_name_or_a_description_it_cannot_read_is_not_a_tool() -> (
    None
):
    """The constructor's other refusals, all of them the same kind of argument.

    A blank name is a tool no host can offer and a blank description is one an agent
    cannot decide whether to call -- which is the only part of a read surface that
    reaches a machine before the result exists. Both are refused here rather than in a
    test, for the reason the result-shape refusal is: the check has to be on the thing
    that ships.
    """
    valid = TOOLS_BY_NAME["refusals"]

    def build(**overrides: Any) -> ToolSpec:
        fields: dict[str, Any] = {
            "name": valid.name,
            "description": valid.description,
            "input_schema": dict(valid.input_schema),
            "result_schema": dict(valid.result_schema),
            "produce": valid.produce,
        }
        fields.update(overrides)
        return ToolSpec(**fields)

    with pytest.raises(ValueError, match="name must not be blank"):
        build(name="   ")
    with pytest.raises(ValueError, match="description must not be blank"):
        build(description="")
    with pytest.raises(ValueError, match="input_schema must be a mapping"):
        build(input_schema=["not", "a", "mapping"])
    with pytest.raises(ValueError, match="result_schema must be a mapping"):
        build(result_schema=None)
    with pytest.raises(ValueError, match="produce must be callable"):
        build(produce="not callable")
    with pytest.raises(ValueError, match="non-string key"):
        build(result_schema={"type": "object", "properties": {1: "a key that is not a string"}})


def test_a_list_of_numbers_inside_a_bounded_object_is_still_a_list_of_numbers_because_a_rate_written_as_an_array_travels_without_the_population_it_counts() -> (
    None
):
    """The array branch, and the inheritance that keeps a truncation legal.

    Two cases either side of one line. A list of bare numbers under a described
    population is accepted, because a rate written as an array travels with the
    description its sibling carries; the same list under an unbounded key is not, because
    then it travels alone. And the corpus header's truncation -- four numbers with no
    prose of their own -- is accepted, because it is a detail of an object that states
    the walk's verdict. A check that rejected the second would be complaining about the
    payload this program is proud of.
    """
    assert not list(unbound_numbers({"description": "records", "counts": [1, 2, 3]}))
    assert list(unbound_numbers({"counts": [1, 2, 3]}))
    assert not list(
        unbound_numbers(
            {
                "completeness": "offset_cap",
                "truncation": {
                    "offset_reached": 10_000,
                    "records_missing": 500,
                    "boundary_repositories": {"acme/widget": 3},
                    "boundary_months": {"2026-08": 2},
                },
            }
        )
    )
    assert list(unbound_numbers({"truncation": {"offset_reached": 10_000}})), (
        "the same numbers without the verdict above them are a bare count"
    )
    assert list(unbound_numbers(412)) == ["$"], "a bare number is the defect itself"
    assert not list(unbound_numbers("a string is not a number"))
    assert not list(unbound_numbers(True)), "a boolean is not a count"


# -- the adapter's own small functions ------------------------------------------------


def test_the_adapter_builds_its_settings_from_the_same_environment_the_command_line_does_because_two_configurations_would_be_two_answers_to_where_the_corpus_is() -> (
    None
):
    """One configuration, one client, one store.

    The value is the command line's own :func:`~tenbin.config.settings_from_env`, so a
    URL this program refuses to dial is refused identically on both surfaces. A server
    that assembled its own ``Settings`` could accept a store URL the command line rejects,
    and then a report published by an agent would describe a corpus the operator was
    configured never to read.
    """
    from tenbin import __version__

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setenv("TENBIN_TANSEKI_URL", "https://store.example/v1")
        settings = mcp_server._settings()
    finally:
        monkeypatch.undo()

    assert settings.tanseki_url == "https://store.example/v1"
    assert mcp_server._version() == __version__


def test_building_the_server_without_the_protocol_library_says_which_line_to_add_because_an_operator_without_the_protocol_gets_a_traceback_from_inside_the_sdk() -> (
    None
):
    """The missing dependency is a sentence, not an ``ImportError`` with no remedy.

    Skipped once ``mcp`` is installed, because then there is nothing to report -- and the
    skip is written rather than assumed, so the day the dependency lands this test stops
    pretending to cover the path it can no longer reach. Asserted on the message rather
    than on the exception type alone, because the type is the part an operator already
    has and the message is the part they do not.
    """
    if importlib.util.find_spec("mcp") is not None:  # pragma: no cover - environment
        pytest.skip("the protocol library is installed, so there is no failure to report")

    with pytest.raises(ModuleNotFoundError) as caught:
        mcp_server.build_server()

    assert str(caught.value) == mcp_server.MISSING_DEPENDENCY
    assert "mcp>=2,<3" in str(caught.value)
    assert dict(mcp_server._annotations()) == dict(READ_ONLY_HINTS), (
        "with no protocol library the hints degrade to the plain mapping, which is a "
        "documentation gap rather than a capability"
    )
