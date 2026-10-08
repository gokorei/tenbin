"""The tool table as module-level data, and the pure functions that fill it in.

This module holds the whole design and imports none of the protocol. It is a frozen
table of :class:`ToolSpec` -- a name, a description, an input schema, a result
schema, and the function that produces the result -- plus the three functions those
descriptors name. :mod:`tenbin.mcp_server.server` is a thin adapter that walks the
table and hands each descriptor to the MCP SDK; the adapter owns no decision, so
every property argued in this file is a property of data that can be asserted on
without a server, a socket, or the ``mcp`` package installed.

**Why the table is data and not decorators.** ``@server.tool(...)`` at import time
means the tool set is whatever the module decorators happened to produce, and the
only way to assert on it is to stand a server up. A table can be pinned in a test
that imports nothing but a dictionary, and :data:`TOOL_NAMES` can be compared
against a literal the way Kojutsu pins its own two. A fourth tool is then a diff
against a name somebody has to delete, which is the shape of change that gets
reviewed; a fourth decorator is a line of code, which is the shape that does not.

**A tool must never return a bare count, and this is the rule an API breaks
first.** ``return len(records)`` is one line, it is correct, and it is the exact
output this program exists to refuse: a count over a self-selected corpus of
unknown completeness, which an agent then stores as a fact about a team. So the
rule is not left to review. Every :class:`ToolSpec` declares a result schema whose
top level is an object, and every result-producing function runs its own output
through :func:`unbound_numbers` and refuses to hand back a number that is not
sitting inside an object that says what it is. A schema alone would not be enough
-- a schema is a description of intent, and this is a check on the thing that
ships.

**A tool that returns a figure returns the whole section.** :func:`report_result`
calls :func:`~tenbin.report.json_.report_as_dict`, which is the value behind
:func:`~tenbin.report.json_.render_json`, so the statement, the negative space, the
falsifier, the denominator and the corpus completeness travel with every number
because the report already puts them there. There is no second serialisation here to
drift from the CLI's, for the same reason the Markdown and JSON renderers share
:func:`~tenbin.report.ordering.ordered_parts`: two writers of one format is two
things to keep true, and the moment somebody relaxes the caveats in one of them the
two surfaces start disagreeing about what a number means.

**The description is where the bound goes.** A caveat-first layout was argued for a
human who reads top to bottom. An agent does not read top to bottom -- it reads a
field and trusts its name -- so the negative space that the prose renderers put
*above* a figure reaches a machine only in the one place a machine reads before it
acts: the tool description, which is what an agent decides from. So
:data:`SELF_SELECTION_BOUND` is written into every description here rather than only
into the results, and the claim that a result is well-formed does not depend on the
caller having read it.

**The tool set is the size of the refusal set, and no larger.** Three: what was
measured, what is claimed, what is refused. Every tool added is a claim about the
corpus that somebody can act on, and this corpus supports very few -- see
:mod:`tenbin.claims.refusals`, where the catalogue is longer than this table and
every entry names a fact that did not arrive.

**What this module notably does not do:** it does not read the store, hold a
client, or open a socket. :func:`tenbin.cli.CLIENT_FACTORY` reaches the store and
the adapter calls it, so the only door to the corpus is the same one the command
line has and there is no second path to it. It does not render Markdown, and it
does not re-serialise the report: a report here is the CLI's report, in the CLI's
own JSON, and :mod:`tenbin.mcp_server` does not get a way to say something the
command line cannot. Nor does it decide what a claim means, what a refusal covers,
or which measures exist -- those belong to :mod:`tenbin.claims` and
:mod:`tenbin.measures`, and this module asks them the same questions the command
line asks.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

from tenbin.claims.gate import ClaimGate
from tenbin.claims.mechanism import Granularity
from tenbin.claims.model import Claim
from tenbin.claims.refusals import NO_TICKET_UNBLOCKS_THIS, Refusal
from tenbin.claims.registry import RefusalRegistry, default_registry
from tenbin.cli import DENOMINATOR_FROM_READ, NO_READ
from tenbin.corpus.snapshot import CorpusSnapshot
from tenbin.measures import Measure, default_measures
from tenbin.report import build_report, report_as_dict

# -- the bound ---------------------------------------------------------------------

#: What every number this program renders is *not*, in one sentence, so the three
#: descriptions cannot drift apart on the part an agent reads before it acts.
#:
#: The wording is deliberately the package's own rather than a new one: this is the
#: argument in :mod:`tenbin`'s docstring and in the completeness claim's
#: ``does_not_mean``, restated for a reader who is an agent and will not render a
#: report at all. The corpus is self-selected because Kojutsu captures only what
#: it is present for, and self-censored because a capture path that fails writes
#: nothing; the two are the same defect seen from the writer's side and the reader's,
#: and neither is repaired by a summary of what did arrive.
SELF_SELECTION_BOUND: Final[str] = (
    "The corpus is self-selected and self-censored: Kojutsu captures only what it is "
    "present for, a capture path that fails writes nothing, and a change nobody examined is "
    "indistinguishable from a change nobody made. A repository where nobody answers is "
    "byte-identical to a repository with no knowledge debt, and no backfill path exists for "
    "either. So a figure here is a count over one read of a selected population. It is not a "
    "statement about development, about a team, or about anybody's practice, and storing it "
    "does not make it one."
)

#: Why a result that is empty is not a result that says nothing was found. Carried
#: in the descriptions rather than only in the results, because the moment it matters
#: is before the call: an agent that reads "no sections" as a finding has already
#: misread it, and a payload it never looked at cannot correct that.
EMPTY_IS_NOT_NOTHING: Final[str] = (
    "An empty result is not a finding. A measure that returned nothing and a measure that "
    "was not asked both render as nothing here, and only the refusals say which is which."
)

#: The refusal tool's own negative space, and the reason it exists at all: an agent
#: that cannot ask what is not measurable will infer that it is measurable.
REFUSAL_TOOL_BOUND: Final[str] = (
    "The absence of a refusal for a measure is not a claim that the corpus supports it; it "
    "means this program never asked. A refusal whose unblocked_by is null is not a gap in a "
    "plan either -- nothing unblocks it, and that is the answer."
)

#: Every tool is a read, and this is what the protocol is told so it can say so to a
#: host. Plain data rather than an SDK object so the table carries it and the adapter
#: does not decide it; the same reasoning as the schemas.
READ_ONLY_HINTS: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    }
)

#: What the server is for, in the one place a host reads before it offers a tool to a
#: model. An agent's first question about a corpus should be whether the corpus is
#: healthy, and this is the only surface in the system that can answer it.
INSTRUCTIONS: Final[str] = (
    "Tenbin measures a Kojutsu corpus and refuses what that corpus cannot support. "
    "Call refusals before reading a report: it is the answer to 'which measures are not "
    "supported', and it distinguishes a measure that found nothing from a measure that "
    "does not exist. Every figure this server returns arrives with the statement of what "
    "it is, what it does not mean, what would falsify it, and the denominator it is a rate "
    "over. Never quote a figure without them, and never treat a number here as a fact "
    "about a team: " + SELF_SELECTION_BOUND
)

#: The claim's four fields, as the names the payloads use for them. One definition so
#: the schemas, the result builders, and the tests cannot disagree about which four.
CLAIM_FIELDS: Final[tuple[str, ...]] = (
    "statement",
    "does_not_mean",
    "falsifier",
    "denominator",
)

#: The refusal's three fields, which do the same work for a refusal that the four do
#: for a figure: the reason, the fact whose arrival would change it, and what would
#: deliver that fact. A refusal has no falsifier and no denominator because it asserts
#: no number; it has these because it is still a claim about the corpus.
REFUSAL_FIELDS: Final[tuple[str, ...]] = (
    "reason",
    "missing_fact",
    "unblocking",
)

#: Keys that make an object a *statement* rather than a bag of numbers. An object
#: carrying one of these is permitted to carry a number; an object that does not is
#: a count wearing a schema, which is the thing this module exists to refuse.
BOUNDED_BY: Final[tuple[str, ...]] = (
    "statement",
    "reason",
    "completeness",
    "description",
)


# -- the no-bare-count check -------------------------------------------------------


def _is_number(value: object) -> bool:
    """Whether this is a number, and deliberately not a boolean.

    ``bool`` is an ``int`` in Python, and a payload carrying ``True`` where a count
    belongs is a payload whose type has already drifted from what the schema claims.
    """
    return not isinstance(value, bool) and isinstance(value, int | float)


def _carries_a_number(value: Mapping[str, Any]) -> bool:
    """Whether this object holds a number anywhere among its direct values."""
    return any(_is_number(item) for item in value.values())


def unbound_numbers(value: Any, path: str = "$", *, bounded: bool = False) -> Iterator[str]:
    """Every place in ``value`` holds a number that no claim or bound explains.

    The walk is over the *result*, not over the table, because the rule is about
    what ships: a schema can describe a number as a required field of a bounded
    object and a function can still return ``{"count": 412}``. An object is allowed
    a number only when it carries one of :data:`BOUNDED_BY` -- a statement, a reason,
    a corpus verdict, or a population description -- so a rate's size is legal
    because it sits beside the description of the population it counts and a
    ``{"count": 412}`` is not.

    Boundness is inherited by what an object contains, and that is a deliberate leniency
    rather than an oversight. A truncation's four numbers are details of the corpus
    header that states whether the walk finished, and a check that rejected them would
    be complaining about the payload this program is proud of. The price is that a
    number two levels below a claim is not separately checked; the shape this catches is
    the one that actually gets written, which is a count at the top of a result or
    beside a key called ``count``.

    Yields a path per violation rather than raising, so a test can assert on all of
    them at once and the message can name where the defect is.
    """
    if _is_number(value):
        if not bounded:
            yield path
        return
    if isinstance(value, Mapping):
        here = bounded or any(key in value for key in BOUNDED_BY)
        if not here and _carries_a_number(value):
            yield path
            return
        for key, item in value.items():
            yield from unbound_numbers(item, f"{path}.{key}", bounded=here)
        return
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        if any(_is_number(item) for item in value):
            if not bounded:
                yield path
            return
        for index, item in enumerate(value):
            yield from unbound_numbers(item, f"{path}[{index}]", bounded=bounded)


def refuse_unbound_numbers(result: Mapping[str, Any], tool: str) -> Mapping[str, Any]:
    """Hand the result back, or refuse to, because a bare number is not an answer.

    Called by every result-producing function on its own output, so the guarantee
    holds in production and not only in a test: a tool that returned a count would
    raise here rather than publish it, and the failure names the path where the
    number sat so the defect is findable. A tool returning ``{"count": 412}`` is a
    one-line mistake, and this is the line that makes it a test failure instead of
    a figure in somebody's model.
    """
    offenders = tuple(unbound_numbers(result))
    if offenders:
        raise ValueError(
            f"the {tool} tool would return {offenders[0]} as a bare number; a figure is "
            "only publishable inside an object that states what it is, what it does not "
            "mean, what would falsify it, and the population it is over"
        )
    return result


# -- the result builders -----------------------------------------------------------


def report_result(
    snapshot: CorpusSnapshot | None = None, *, integrity_only: bool = False
) -> dict[str, Any]:
    """The whole report, in the command line's own JSON, or refuse to build one.

    The payload is :func:`~tenbin.report.json_.report_as_dict` verbatim -- the value
    behind :func:`~tenbin.report.json_.render_json` -- so a figure arrives with its
    statement, its negative space, its falsifier, its denominator, and the corpus
    header that says how much of the store the read reached. There is no path through
    this function that yields a figure without them, because there is no second
    serialisation to relax them in.

    ``snapshot=None`` is a refusal rather than an empty report. A store that was never
    reached and a store that was reached and holds nothing produce the same zeros, and
    the difference is the whole difference between a finding and a failure; the
    command line settles that by exiting before it renders, and this settles it the
    same way rather than returning a document that looks like an answer.

    ``integrity_only`` selects the measures about the capture system rather than the
    work it captured -- the report a caller wants when the question is "can I believe
    the rest of this?". It is the same opt-in flag on the measure that the command
    line filters on, derived from the same published registry, so a measure added to
    one is added to the other.
    """
    if snapshot is None:
        raise ValueError(
            "the report tool was called with no read behind it; a report describes one read "
            "of one collection, and a document rendered from nothing describes nothing"
        )
    measures: Sequence[Measure] = (
        _measures_about_the_capture_system() if integrity_only else default_measures()
    )
    report = build_report(
        snapshot,
        gate=ClaimGate(),
        registry=default_registry(),
        measures=measures,
    )
    result = report_as_dict(report)
    return dict(refuse_unbound_numbers(result, "report"))


def claims_result(snapshot: CorpusSnapshot | None = None) -> dict[str, Any]:
    """Every claim this program is prepared to make, with its four fields, and no store.

    A listing of claims is the program's reasoning rather than a summary of it, so it
    needs no corpus -- the same reason ``tenbin claims`` needs none. What it cannot do
    without a read is state a denominator's *size*, because every size in this program
    is a count over one particular read. So ``snapshot=None`` gives the description and
    says where the size comes from, rather than inventing one over the empty corpus
    that stands in for a read: a fabricated population under every claim is precisely
    what this package refuses, and it would be easier to fabricate here than at the
    command line because there is no terminal for a reader to notice.

    The sentences are :data:`tenbin.cli.DENOMINATOR_FROM_READ` and
    :data:`tenbin.cli.NO_READ`, imported rather than restated. Two surfaces that each
    wrote their own version of "no read" would agree today and drift on the day one
    of them was edited, and a reader comparing the command line with an agent's copy
    of the payload would be comparing two arguments.
    """
    read = snapshot if snapshot is not None else NO_READ
    sized = snapshot is not None
    result = {
        "claims": [
            _claim_entry(measure.claim(read), sized=sized) for measure in default_measures()
        ],
        "read": _read_entry(read, sized=sized),
    }
    return dict(refuse_unbound_numbers(result, "claims"))


def refusals_result(snapshot: CorpusSnapshot | None = None) -> dict[str, Any]:
    """The refusal registry: what is not measurable, why, and what would change it.

    This is the tool the ticket is for. An agent reasoning over a corpus can ask
    Kojutsu what it knows; nothing could tell it which questions the corpus forecloses,
    and an agent that cannot ask will infer that everything is answerable. So the
    registry goes out whole, with the reasons in the words the catalogue uses -- a
    refusal whose reader has to go and find the reason has already lost the reader who
    asked.

    ``snapshot`` is accepted and ignored, for one reason: one dispatcher line serves
    all three tools, and an arity that varies per tool is a branch somebody will add a
    tool to later. A refusal is a property of the program, not of a read, so nothing
    here depends on a corpus having been walked -- which is the same property the
    command line has and the reason its catalogue is inspectable before there is a
    deployment to inspect.
    """
    del snapshot  # A refusal is a property of the program; a read would only add noise.
    registry = default_registry()
    result = {
        "refusals": [_refusal_entry(refusal) for refusal in registry.all()],
        "mechanism": _mechanism_entry(registry),
    }
    return dict(refuse_unbound_numbers(result, "refusals"))


def call(
    spec: ToolSpec, *, snapshot: CorpusSnapshot | None = None, **arguments: Any
) -> dict[str, Any]:
    """Run one tool by name, so the adapter holds no dispatch logic of its own.

    The adapter's whole job is to hand a descriptor to the SDK, and a dispatcher that
    branched on tool names would be a place a fourth tool could be added without a
    table entry. Uniform ``(snapshot, **arguments)`` is what lets the table be the
    only place a tool is declared.
    """
    return spec.produce(snapshot, **arguments)


def _claim_entry(claim: Claim, *, sized: bool) -> dict[str, Any]:
    """One claim, with the four fields under the names the report payload uses.

    Identity first, then the claim's own three caveats in reading order, then the
    denominator -- the same order :func:`tenbin.cli` prints them in, so a claim
    listed by the command line and a claim listed by a tool are the same document.
    """
    return {
        "slug": claim.slug,
        "claim_kind": claim.kind.value,
        "granularity": claim.granularity.value,
        "statement": claim.statement,
        "does_not_mean": claim.does_not_mean,
        "falsifier": claim.falsifier,
        "denominator": _denominator_entry(claim, sized=sized),
        "unblocked_by": claim.unblocked_by,
        "source": claim.source,
    }


def _denominator_entry(claim: Claim, *, sized: bool) -> dict[str, Any]:
    """The population, with its size only when a read supplied one.

    ``size`` is ``null`` and ``size_note`` says why when there is no read, because a
    number with no description of where it came from is a number a caller will reuse,
    and one with a description and no size is a number a caller can check. Both
    together is the only shape that survives being stored.
    """
    denominator = claim.denominator
    return {
        "description": denominator.description,
        "size": denominator.size if sized else None,
        "size_note": None if sized else DENOMINATOR_FROM_READ,
    }


def _read_entry(read: CorpusSnapshot, *, sized: bool) -> dict[str, Any]:
    """Whether a read backed this listing, and which one.

    Present even when the answer is "no read happened", because the difference
    between "these sizes are from a read you can check" and "these are placeholders"
    is the difference between a payload an agent may act on and one it must not.
    """
    if not sized:
        return {"taken": False, "read_at": None, "collection": None, "completeness": None}
    return {
        "taken": True,
        "read_at": read.read_at.isoformat(),
        "collection": read.collection,
        "completeness": read.completeness.value,
    }


def _refusal_entry(refusal: Refusal) -> dict[str, Any]:
    """One refusal: the reason, the missing fact, and what would deliver it.

    ``unblocking`` is the sentence and ``unblocked_by`` is the ticket or ``null``, and
    both are present. The difference between "nothing will" and "nobody wrote it down"
    is the difference between a decision and a bug, and an agent that stores a
    refusal with a blank unblocking will wait for work that was never going to be
    filed.
    """
    return {
        "claim_slug": refusal.claim_slug,
        "title": refusal.title,
        "reason": refusal.reason,
        "missing_fact": refusal.missing_fact,
        "unblocked_by": refusal.unblocked_by,
        "unblocking": refusal.unblocked_by
        if refusal.unblocked_by is not None
        else NO_TICKET_UNBLOCKS_THIS,
    }


def _mechanism_entry(registry: RefusalRegistry) -> dict[str, Any]:
    """Why a comparison is refused, whether or not a mechanism has been set.

    Carried because the central refusal in the catalogue is a comparison and its
    reason is the absence of this: an agent that reads "no comparison" without reading
    why will look for the missing measurement, and the missing measurement is not a
    fact that could arrive.
    """
    return {
        "assignment_mechanism": None,
        "unset_reason": registry.mechanism_unset_reason,
    }


def _measures_about_the_capture_system() -> tuple[Measure, ...]:
    """The measures about whether the corpus is what it claims to be.

    Selected by the same opt-in flag on the measure that ``tenbin report
    --integrity-only`` selects on, read from the same published registry, rather than
    from a list written out here. The reason is the one :mod:`tenbin.cli` gives its
    own copy: a second list of the same thing is a list nobody reconciles, so a measure
    added to the registry and not to the list would be written, tested, listed by
    ``tenbin claims``, and never appear in an integrity report. The honest form of
    that helper is a published one in :mod:`tenbin.measures` that both call sites
    share; until it exists these two filter the same source with the same four lines,
    and :mod:`tests.test_mcp_surface` asserts they agree on the slugs.
    """
    return tuple(
        measure
        for measure in default_measures()
        if getattr(measure, "about_the_capture_system", False)
    )


# -- the schemas -------------------------------------------------------------------

#: Granularities a rendered claim may carry. ``individual`` is absent because the type
#: refuses it below the floor whatever the floor is, so a schema listing it would be
#: offering a value no configuration can make renderable. Derived from the enum rather
#: than written out, so a granularity added to the model cannot be missing here.
GRANULARITIES: Final[tuple[str, ...]] = tuple(
    kind.value for kind in Granularity if kind is not Granularity.individual
)

#: A non-blank string, which every prose field in every result is.
_TEXT: Final[Mapping[str, Any]] = MappingProxyType({"type": "string", "minLength": 1})

#: A count, in the one shape this program allows a count to take: a number, never a
#: bare integer at the top level of a result.
_COUNT: Final[Mapping[str, Any]] = MappingProxyType({"type": "integer", "minimum": 0})

#: The corpus header, exactly as :func:`~tenbin.report.json_.render_json` emits it.
#: Enumerated and the store's own count beside the verdict that says whether the two
#: may be compared, so a number about the read is never readable without the bound on it.
_CORPUS: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "type": "object",
        "description": "The read every figure below was computed over, and how much of "
        "the store the walk reached.",
        "required": [
            "collection",
            "read_at",
            "enumerated",
            "store_total",
            "completeness",
            "truncation",
        ],
        "additionalProperties": False,
        "properties": {
            "collection": _TEXT,
            "read_at": MappingProxyType({"type": "string"}),
            "enumerated": _COUNT,
            "store_total": _COUNT,
            "completeness": MappingProxyType(
                {
                    "type": "string",
                    "enum": ["complete", "offset_cap", "store_total_exceeded", "store_error"],
                }
            ),
            "truncation": MappingProxyType(
                {
                    "type": ["object", "null"],
                    "description": "What the walk did not reach, and where it stopped.",
                    "required": [
                        "offset_reached",
                        "records_missing",
                        "boundary_repositories",
                        "boundary_months",
                    ],
                    "properties": {
                        "offset_reached": _COUNT,
                        "records_missing": _COUNT,
                        "boundary_repositories": MappingProxyType(
                            {"type": "object", "additionalProperties": _COUNT}
                        ),
                        "boundary_months": MappingProxyType(
                            {"type": "object", "additionalProperties": _COUNT}
                        ),
                    },
                }
            ),
        },
    }
)

#: A rendered figure: the four claim fields required, beside the number. The
#: ``additionalProperties: False`` is the load-bearing part -- a section carrying a
#: count and no caveats would be rejected by the schema as well as by
#: :func:`unbound_numbers`, so the two checks fail together rather than one of them
#: being the one everybody remembers.
_FIGURE_SECTION: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "type": "object",
        "description": "One measure's answer, with the four claim fields that say what it is.",
        "required": [
            "slug",
            "kind",
            "title",
            *CLAIM_FIELDS,
            "figure",
            "claim_kind",
            "granularity",
            "source",
        ],
        "additionalProperties": False,
        "properties": {
            "slug": _TEXT,
            "kind": MappingProxyType({"type": "string", "const": "claim"}),
            "title": _TEXT,
            "statement": _TEXT,
            "does_not_mean": _TEXT,
            "falsifier": _TEXT,
            "figure": MappingProxyType(
                {
                    "type": "string",
                    "description": "The number, preceded by the completeness of the read it "
                    "was computed over.",
                }
            ),
            "denominator": _TEXT,
            "claim_kind": MappingProxyType(
                {
                    "type": "string",
                    "enum": ["descriptive", "comparative", "assignment_mechanism"],
                }
            ),
            "granularity": MappingProxyType({"type": "string", "enum": list(GRANULARITIES)}),
            "source": MappingProxyType({"type": ["string", "null"]}),
        },
    }
)

#: A rendered refusal: the reason, the missing fact, and the unblocking -- the three
#: fields that do for a refusal what the four do for a figure. ``unblocking`` is the
#: sentence and ``unblocked_by`` the ticket or ``null``, and both are required because
#: the difference between a decision and a bug is exactly the difference between them.
_REFUSAL_SECTION: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "type": "object",
        "description": "A measure this corpus cannot support, in place of the figure.",
        "required": ["slug", "kind", "title", *REFUSAL_FIELDS, "unblocked_by"],
        "additionalProperties": False,
        "properties": {
            "slug": _TEXT,
            "kind": MappingProxyType({"type": "string", "const": "refusal"}),
            "title": _TEXT,
            "reason": _TEXT,
            "missing_fact": MappingProxyType({"type": ["string", "null"]}),
            "unblocking": _TEXT,
            "unblocked_by": MappingProxyType({"type": ["string", "null"]}),
        },
    }
)

#: The claims tool's own item: the four claim fields, required, plus the identity an
#: agent needs in order to look one of them up and the typed provenance beside the
#: prose. Same four names as the report's, because it is the same claims.
_CLAIM_ITEM: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "type": "object",
        "required": [
            "slug",
            "claim_kind",
            "granularity",
            *CLAIM_FIELDS,
            "unblocked_by",
            "source",
        ],
        "additionalProperties": False,
        "properties": {
            "slug": _TEXT,
            "claim_kind": MappingProxyType(
                {
                    "type": "string",
                    "enum": ["descriptive", "comparative", "assignment_mechanism"],
                }
            ),
            "granularity": MappingProxyType({"type": "string", "enum": list(GRANULARITIES)}),
            "statement": _TEXT,
            "does_not_mean": _TEXT,
            "falsifier": _TEXT,
            "denominator": MappingProxyType(
                {
                    "type": "object",
                    "description": "The population, and -- only when a read supplied one -- "
                    "its size. A size with no read behind it is null, and says so.",
                    "required": ["description", "size", "size_note"],
                    "additionalProperties": False,
                    "properties": {
                        "description": _TEXT,
                        "size": MappingProxyType({"type": ["integer", "null"], "minimum": 1}),
                        "size_note": MappingProxyType({"type": ["string", "null"]}),
                    },
                }
            ),
            "unblocked_by": MappingProxyType({"type": ["string", "null"]}),
            "source": MappingProxyType({"type": ["string", "null"]}),
        },
    }
)

#: The refusals tool's item. Every field is required, and two of them are nullable on
#: purpose: their absence is the information, and a missing key would be
#: indistinguishable from a key somebody forgot.
_REFUSAL_ITEM: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "type": "object",
        "required": [
            "claim_slug",
            "title",
            "reason",
            *REFUSAL_FIELDS,
            "unblocked_by",
        ],
        "additionalProperties": False,
        "properties": {
            "claim_slug": _TEXT,
            "title": _TEXT,
            "reason": _TEXT,
            "missing_fact": MappingProxyType({"type": ["string", "null"]}),
            "unblocked_by": MappingProxyType({"type": ["string", "null"]}),
            "unblocking": _TEXT,
        },
    }
)

#: An empty argument object, and the reason the two reasoning tools have one. A tool
#: that takes no arguments is a claim that there is nothing to choose between, and
#: ``additionalProperties: False`` is what makes it true rather than merely intended:
#: a filter argument is how a surface acquires a population this corpus cannot
#: support, so the door is closed rather than merely unadvertised.
_NO_ARGUMENTS: Final[Mapping[str, Any]] = MappingProxyType(
    {"type": "object", "properties": {}, "additionalProperties": False}
)

#: The report's one argument. A boolean, not a population: there is no filter, no
#: date range, no repository list, and no comparison, because each of those is a
#: request to be handed a number over a chosen group and this corpus has no mechanism
#: that would make the choice mean anything.
_REPORT_ARGUMENTS: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "type": "object",
        "properties": {
            "integrity_only": MappingProxyType(
                {
                    "type": "boolean",
                    "description": "Run only the measures about the capture system, not the "
                    "work it captured. The report a caller wants when the question is "
                    "'can I believe the rest of this?' rather than 'what is in the corpus?'.",
                }
            )
        },
        "additionalProperties": False,
    }
)


# -- the table ---------------------------------------------------------------------


@dataclass(frozen=True)
class ToolSpec:
    """One tool, described as data: what it is called, what it says, what it returns.

    A frozen dataclass holding the schemas and the producing callable, rather than a
    decorated function, for the reason in the module docstring: a table can be asserted
    on without a server. The schemas are copied and wrapped on construction, because a
    frozen dataclass holding a dict claims to be immutable and is not -- the same
    correction :class:`~tenbin.store.client.StoreDocument` makes for the same reason.
    """

    name: str
    description: str
    input_schema: Mapping[str, Any]
    result_schema: Mapping[str, Any]
    produce: Callable[..., dict[str, Any]]

    def __post_init__(self) -> None:
        for name in ("name", "description"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"a tool's {name} must not be blank; see ToolSpec")
        for name in ("input_schema", "result_schema"):
            schema = getattr(self, name)
            if not isinstance(schema, Mapping):
                raise ValueError(f"{self.name}'s {name} must be a mapping")
            object.__setattr__(
                self, name, MappingProxyType(_plain_copy(schema, owner=f"{self.name}.{name}"))
            )
        if self.result_schema.get("type") != "object":
            # Refused at construction rather than asserted in a test, because a tool
            # whose result is not an object is a tool that can return a bare number,
            # and that is the one shape this package exists to refuse. A list would be
            # nearly as bad: a caller indexing it gets the first entry and no bound.
            raise ValueError(
                f"{self.name}'s result schema must declare an object at its top level; a "
                "result this program publishes is an object, because an object is the only "
                "shape that can carry the claim beside the number"
            )
        if not callable(self.produce):
            raise ValueError(f"{self.name}'s produce must be callable")


def _plain_copy(value: Any, *, owner: str) -> Any:
    """A deep copy of a schema as the plain JSON types, or a refusal naming what is not.

    Two things are being done at once, and the second is the reason this is not a
    ``json.dumps`` round trip. The copy is what makes :class:`ToolSpec` honestly frozen
    -- a frozen dataclass holding a dict claims to be immutable and is not, the same
    defect :class:`~tenbin.store.client.StoreDocument` corrects for the same reason.
    The refusal is what makes a schema the protocol can actually carry: this function
    accepts exactly ``dict``, ``list``, ``str``, ``bool``, ``int``, ``float`` and
    ``None``, so a schema holding a :class:`~enum.StrEnum`, a tuple or a callable fails
    when the table is built rather than when a host tries to serialise it four
    attributes away from here.
    """
    if value is None or isinstance(value, str | bool | int | float):
        return value
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(
                    f"{owner} has a non-string key {key!r}; a schema is a JSON object and "
                    "its keys are strings"
                )
            copied[key] = _plain_copy(item, owner=f"{owner}.{key}")
        return copied
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_plain_copy(item, owner=f"{owner}[{index}]") for index, item in enumerate(value)]
    raise ValueError(
        f"{owner} holds {type(value).__name__}, which a JSON schema cannot carry; a schema "
        "may hold only objects, arrays, strings, numbers, booleans and null"
    )


#: The tool set, in the order an agent should consider it. Three tools, and the order
#: is the argument of the last one: a caller that wants to know whether the corpus can
#: support a question should find the refusals before it finds the figures, because the
#: figures are the part that will mislead.
TOOLS: Final[tuple[ToolSpec, ...]] = (
    ToolSpec(
        name="report",
        description=(
            "Measure the Kojutsu corpus and return the report with every caveat attached. "
            "Read this before you call it, because it is the part that arrives whether or not "
            "you read the result. " + SELF_SELECTION_BOUND + " Every figure comes with four "
            "fields: the statement of what it is, what it does NOT mean, what would falsify "
            "it, and the denominator it is a rate over. The header carries how many documents "
            "the read enumerated, what the store says it holds, and whether the walk reached "
            "the end; every count below is conditional on that and is an undercount when it "
            "did not. " + EMPTY_IS_NOT_NOTHING + " Call the refusals tool first if you want to "
            "know which measures this corpus cannot support at all."
        ),
        input_schema=_REPORT_ARGUMENTS,
        result_schema=MappingProxyType(
            {
                "type": "object",
                "description": "The whole report. Every figure is a section carrying its four "
                "claim fields; every unsupported measure is a section carrying its refusal.",
                "required": ["corpus", "sections"],
                "additionalProperties": False,
                "properties": {
                    "corpus": _CORPUS,
                    "sections": MappingProxyType(
                        {
                            "type": "array",
                            "description": "Findings first, then the completeness of the read, "
                            "then the rest in registry order, then the refusals.",
                            "items": MappingProxyType(
                                {"anyOf": [{"$ref": "#/$defs/figure"}, {"$ref": "#/$defs/refusal"}]}
                            ),
                        }
                    ),
                },
                "$defs": MappingProxyType({"figure": _FIGURE_SECTION, "refusal": _REFUSAL_SECTION}),
            }
        ),
        produce=report_result,
    ),
    ToolSpec(
        name="claims",
        description=(
            "Every claim this program is prepared to make, with the four fields each one "
            "carries: the statement, what it does NOT mean, what would falsify it, and the "
            "denominator. Needs no store, so call it before deciding a question is unaskable. "
            + SELF_SELECTION_BOUND
            + " A claim listed here is a statement this program would "
            "defend, not a fact it has established: a claim's denominator carries its size only "
            "when a read was taken, and a null size with a note saying so is a placeholder you "
            "must not quote."
        ),
        input_schema=_NO_ARGUMENTS,
        result_schema=MappingProxyType(
            {
                "type": "object",
                "description": "Every claim, and whether a read backed the denominator sizes.",
                "required": ["claims", "read"],
                "additionalProperties": False,
                "properties": {
                    "claims": MappingProxyType(
                        {"type": "array", "items": MappingProxyType({"$ref": "#/$defs/claim"})}
                    ),
                    "read": MappingProxyType(
                        {
                            "type": "object",
                            "description": "Whether the sizes above came from a read, and "
                            "which one. Absent information stated as absence rather than as a "
                            "number.",
                            "required": ["taken", "read_at", "collection", "completeness"],
                            "additionalProperties": False,
                            "properties": {
                                "taken": MappingProxyType({"type": "boolean"}),
                                "read_at": MappingProxyType({"type": ["string", "null"]}),
                                "collection": MappingProxyType({"type": ["string", "null"]}),
                                "completeness": MappingProxyType({"type": ["string", "null"]}),
                            },
                        }
                    ),
                },
                "$defs": MappingProxyType({"claim": _CLAIM_ITEM}),
            }
        ),
        produce=claims_result,
    ),
    ToolSpec(
        name="refusals",
        description=(
            "The measures this corpus cannot support, why, and what would change the answer. "
            "Needs no store, and it is the tool that distinguishes a measure that found "
            "nothing from a measure that does not exist. " + REFUSAL_TOOL_BOUND + " Call it "
            "before you assume a missing measurement is missing evidence: an agent that cannot "
            "ask what is not measurable will infer that it is measurable. Read this before you "
            "call it, because a refusal is not the same statement as a figure and an agent "
            "that reads one as the other has drawn the wrong conclusion. "
            + SELF_SELECTION_BOUND
            + " A measure is refused for the specific fact that did not "
            "arrive, so an entry here is not a claim that the question is bad -- it is a claim "
            "that this corpus cannot answer it, and the corpus it means is the self-selected "
            "one described above rather than the work itself."
        ),
        input_schema=_NO_ARGUMENTS,
        result_schema=MappingProxyType(
            {
                "type": "object",
                "description": "The refusal registry, in catalogue order, and why a comparison "
                "is refused.",
                "required": ["refusals", "mechanism"],
                "additionalProperties": False,
                "properties": {
                    "refusals": MappingProxyType(
                        {"type": "array", "items": MappingProxyType({"$ref": "#/$defs/refusal"})}
                    ),
                    "mechanism": MappingProxyType(
                        {
                            "type": "object",
                            "description": "The assignment mechanism, and why it is absent. "
                            "Absence is a decision rather than a gap; see "
                            "docs/decisions/002-no-causal-claims.md.",
                            "required": ["assignment_mechanism", "unset_reason"],
                            "additionalProperties": False,
                            "properties": {
                                "assignment_mechanism": MappingProxyType(
                                    {"type": ["object", "null"]}
                                ),
                                "unset_reason": _TEXT,
                            },
                        }
                    ),
                },
                "$defs": MappingProxyType({"refusal": _REFUSAL_ITEM}),
            }
        ),
        produce=refusals_result,
    ),
)

#: The tool set as names, pinned. Compared against a literal by
#: :mod:`tests.test_mcp_surface` the way Kojutsu pins its own two: a fourth tool is
#: a fourth line somebody has to delete here and in that test, which is the shape of
#: change that gets argued in review. There is no mechanism to add one quietly, because
#: the only way to reach the table is :data:`TOOLS`.
TOOL_NAMES: Final[tuple[str, ...]] = tuple(spec.name for spec in TOOLS)

#: The table by name, for a caller that has been handed a name. Read-only because the
#: table is a design and a mutable registry is a design anybody can extend.
TOOLS_BY_NAME: Final[Mapping[str, ToolSpec]] = MappingProxyType({spec.name: spec for spec in TOOLS})

#: What this module publishes, named rather than left to whatever is imported into it.
#:
#: This is a library the tests are written against rather than an entry point, so the
#: surface is larger than the adapter's on purpose -- but it is written down for the same
#: reason :data:`TOOLS` is: a helper added for one caller is the first half of a function
#: somebody will use from somewhere, and a measurement program accumulates those without
#: anybody deciding to. A name absent from here is private, whatever its underscore says.
__all__ = [
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
]
