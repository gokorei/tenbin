"""The MCP adapter: walk the table, register it, hold no capability of its own.

Every decision in this surface lives in :mod:`tenbin.mcp_server.tools` -- the tool
set, the wording of every description, the schemas, the bounds, and the functions that
produce the results. This module walks that tuple and hands each descriptor to the
SDK. It adds no tool, rewords no description, and decides no schema, which is the
property that makes the design assertable: :mod:`tests.test_mcp_surface` pins the
table without importing this file, and nothing here can widen it.

**The protocol is imported inside a function, on purpose.** The table is module-level
data and the tests are written against the table, so importing
:mod:`tenbin.mcp_server.tools` -- or this package -- must not require the ``mcp``
package. A design whose properties can only be checked with a protocol library
installed is a design that stops being checked the first time a dependency is in
flight, and a test that cannot run offline is a test that stops being run. The import
is therefore deferred to :func:`build_server`, and :func:`missing_dependency` says
which line ``pyproject.toml`` is missing rather than leaving an ``ImportError``
traceback as the first thing an operator sees.

**The store is reached over the command line's own seam, and nothing else.**
:data:`tenbin.cli.CLIENT_FACTORY` is imported rather than replaced: it is the
module-level name the command line reaches the corpus through, a test can already
hand it a store that answers in process, and reusing it is what makes "the same
read-only client as the command line" a structural property instead of a coincidence
two modules could drift on. There is no second path to the corpus from here.

**The only read capability is a walk, and it is taken once per report call.**
:func:`_snapshot` opens the client, calls
:func:`~tenbin.corpus.snapshot.take_snapshot`, and closes it. A walk that fails
raises, and the failure is returned as a structured error rather than as a document
of zeroes -- the same finding-versus-failure distinction the command line draws by
exiting non-zero, arrived at here because an MCP tool has no exit code. A store that
is reached and holds nothing is *not* a failure: it is a report saying the corpus is
empty, and it comes back as a result.

**What this module notably does not do:** it does not write anything -- not to the
store, not to a log, not to a file. Kojutsu resolved the identical objection about
its own read events by keeping them local and unauthenticated rather than by dropping
them, and the argument for a local log here is real but it is a different decision
than this ticket, so it is left visibly undone rather than half-made. It does not
cache a snapshot between calls: a measurement program whose read is reused across
calls is a program reporting a corpus from whenever that read happened, and the
reader's only defence is the header's moment, which a cache would make a lie. And it
does not re-serialise anything -- a result is handed to the SDK as the object the
tool table's function produced, so a figure cannot be rendered one way here and
another way at the command line.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Final, cast

from tenbin import cli
from tenbin.config import Settings, settings_from_env
from tenbin.corpus.snapshot import CorpusSnapshot, take_snapshot
from tenbin.mcp_server.tools import (
    INSTRUCTIONS,
    READ_ONLY_HINTS,
    TOOLS,
    ToolSpec,
    call,
)
from tenbin.store.client import StoreError

if TYPE_CHECKING:
    # Under ``TYPE_CHECKING`` and not at module scope, so the adapter's own namespace
    # holds no client type at all. ``from __future__ import annotations`` means the
    # annotation below never resolves at runtime, and the guarantee is that this module
    # cannot be handed a client, constructed one, or name one -- so a reader asking what
    # this surface can reach finds only the error types and the command line's own seam.
    from tenbin.store.client import StoreClient

#: What to add when the protocol library is missing. Named as a constant because it is
#: read by an operator in a hurry at seven in the morning, and a message that does not
#: say which line to edit becomes a ``ModuleNotFoundError`` in a traceback.
MISSING_DEPENDENCY = (
    "the MCP server needs the protocol library, which is not installed: add "
    "`mcp>=2,<3` to [project] dependencies in pyproject.toml and run `uv lock`. The "
    "tool table in tenbin.mcp_server.tools is importable without it, so nothing about "
    "this program's reasoning depends on the protocol being present."
)

#: The name a host shows for this server. One string rather than a constant the caller
#: supplies, because a measurement program published under a name somebody chose in a
#: hurry is a name an agent will read before it reads anything else.
SERVER_NAME: Final[str] = "tenbin"


def _settings() -> Settings:
    """The same configuration the command line builds, or fail naming the variable.

    Delegated rather than reimplemented so there is one place that knows what
    ``TENBIN_TANSEKI_URL`` means and one place that validates a store URL. Unconfigured
    is a failure here exactly as it is there: a measurement program with a default
    store would report a real number about whichever store happened to be running.
    """
    return settings_from_env()


@contextmanager
def _read_seam(settings: Settings) -> Iterator[StoreClient]:
    """The command line's read-only client, for the length of one tool call.

    Read as ``cli.CLIENT_FACTORY`` at call time rather than imported by value, so it
    is the *same seam* the command line reaches the corpus through and not merely the
    same class: a test that hands a store to ``tenbin.cli.CLIENT_FACTORY`` has handed
    it to this server too, which is what keeps both surfaces exercisable offline under
    the suite's socket denial. Opened per call and closed on the way out, including on
    a failure, because a server that ran for the life of a session would otherwise hold
    one connection to somebody else's corpus for as long as the agent stays up.
    """
    with cli.CLIENT_FACTORY(settings) as client:
        yield client


def _snapshot(settings: Settings) -> CorpusSnapshot:
    """One read of the collection, or raise -- a walk that stopped is not a corpus.

    A store that cannot be reached raises
    :exc:`~tenbin.store.client.StoreError` rather than returning an empty snapshot,
    which is the guarantee that makes a report of zeroes a finding rather than a
    failure. :func:`_tool_result` turns that raise into a structured error saying the
    corpus was not read, so the caller is told the difference instead of being handed
    an empty document that looks like an answer.
    """
    with _read_seam(settings) as client:
        return take_snapshot(client)


def _error_result(message: str, *, code: str) -> dict[str, Any]:
    """A failure, in the shape a caller can act on rather than swallow.

    ``what_this_is_not`` is on the failure too, not only on the successes: the most
    dangerous thing an agent can be handed here is an error that reads like an empty
    result, so the error says in its own payload that no read happened and no figure
    was computed.
    """
    return {
        "error": code,
        "message": message,
        "what_this_is_not": (
            "The corpus was not read and no figure was computed. This is not an empty "
            "corpus; nothing below is a finding, and nothing here may be stored as one."
        ),
    }


def _tool_result(spec: ToolSpec, *, settings: Settings, **arguments: Any) -> dict[str, Any]:
    """Run one tool, reading the corpus only for the tool that needs a read.

    The two reasoning tools take no read, and that is the whole reason they can answer
    before an agent has decided whether to have a store at all. The branch is on the
    table's own name rather than on a per-tool flag, so adding a fourth tool means
    choosing here too -- and choosing is the point.
    """
    try:
        if spec.name == "report":
            return call(spec, snapshot=_snapshot(settings), **arguments)
        return call(spec, snapshot=None, **arguments)
    except StoreError as exc:
        return _error_result(
            f"the corpus could not be read ({exc}); see the command line, which reports "
            "the same failure with the remedy attached",
            code="corpus_unreadable",
        )
    except ValueError as exc:
        # The refusals the tool table itself raises: a report asked for with no read, a
        # result carrying a bare number. Both are defects rather than conditions, and
        # the message names what went wrong so it is findable in review.
        return _error_result(str(exc), code="refused")


def build_server() -> Any:
    """Construct the MCP server, registering every tool in the table and nothing else.

    The table is walked, not transcribed: a tool that is in
    :data:`~tenbin.mcp_server.tools.TOOLS` is registered, and one that is not does not
    exist. Returns ``Any`` because the protocol's server type is resolved at call time
    and this package must import without it -- a type annotation that named the class
    would turn a missing dependency into a failure to *import the module*, which is
    the one thing the deferred import exists to prevent.
    """
    try:
        from mcp.server.mcpserver import MCPServer  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(MISSING_DEPENDENCY) from exc

    server = MCPServer(SERVER_NAME, version=_version(), instructions=INSTRUCTIONS)
    annotations = _annotations()
    for spec in TOOLS:
        _register(server, spec, annotations)
    return server


def _register(server: Any, spec: ToolSpec, annotations: Any) -> None:
    """One descriptor onto the SDK, as a zero-argument handler plus a declaration.

    The handler closes over the descriptor and the settings, and takes no arguments: the
    SDK's own decorator introspects a handler's signature to build the input schema, and a
    ``**kwargs`` handler would hand it a schema of nothing. The declared schema comes from
    the table instead, which is the one place the argument surface is written down, so the
    two cannot disagree about which arguments exist.

    Registration is the last thing that happens here and nothing is returned. An earlier
    version returned the handler and let the caller add it, which is right for the
    ``add_tool`` shape and a double registration for the decorator shape -- and a tool
    listed twice in a host's tool picker is exactly the kind of surface defect nobody
    looks for, because every call still answers correctly. :func:`_declare` owns which of
    the two shapes this SDK version has.
    """
    settings = _settings()

    def handler() -> dict[str, Any]:
        return _tool_result(spec, settings=settings)

    handler.__name__ = spec.name
    handler.__doc__ = spec.description
    _declare(server, spec, handler, annotations)


def _declare(server: Any, spec: ToolSpec, handler: Any, annotations: Any) -> None:
    """Attach the handler to ``server`` under the name, schema and wording in the table.

    Split out from :func:`_register` so the branch is in one place: the SDK's decorator
    form and its ``add_tool`` form differ only in how the name, the description, the
    schemas and the annotations arrive, and a version that has only one of them should be
    a one-line difference rather than a rewrite. Returns nothing, because in both shapes
    the registration is the side effect and there is no value a caller has any use for.
    """
    decorator = cast("Any", getattr(server, "tool", None))
    if callable(decorator):
        decorator(
            name=spec.name,
            description=spec.description,
            inputSchema=dict(spec.input_schema),
            outputSchema=dict(spec.result_schema),
            annotations=annotations,
        )(handler)
        return
    server.add_tool(
        handler,
        name=spec.name,
        description=spec.description,
        inputSchema=dict(spec.input_schema),
        outputSchema=dict(spec.result_schema),
        annotations=annotations,
    )


def _annotations() -> Any:
    """The read-only hints from the table, in whatever shape this SDK version wants.

    Falls back to the plain mapping when the SDK has no ``ToolAnnotations``, because the
    hints are a declaration the host reads and their absence is a documentation gap
    rather than a capability: this server holds no write path either way, and that is
    asserted structurally in the tests rather than asserted here by a type check.
    """
    try:
        from mcp.types import ToolAnnotations  # noqa: PLC0415
    except ModuleNotFoundError:
        return dict(READ_ONLY_HINTS)
    return ToolAnnotations.model_validate(dict(READ_ONLY_HINTS))


def _version() -> str:
    """This package's version, so a host can report which build answered."""
    from tenbin import __version__  # noqa: PLC0415

    return __version__


def main() -> None:
    """Run the server over stdio.

    stdout belongs to the protocol, so anything this process has to say goes to stderr;
    a stray line on stdout desynchronises the session rather than failing loudly. The
    dependency check happens before the server is built so an operator without the
    protocol library gets the instruction rather than a traceback from inside it.
    """
    print("tenbin mcp server: three read tools, no write path", file=sys.stderr)
    server = build_server()
    server.run(transport="stdio")


if __name__ == "__main__":  # pragma: no cover
    main()
