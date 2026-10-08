"""The ``http.server`` shell: three routes, and a sentence for everything else.

This module is deliberately the thinnest thing in the package. It maps a path to
:func:`~tenbin.webapp.pages.index_page` or :func:`~tenbin.webapp.pages.report_page`,
writes the bytes, and does nothing else -- every question about what a store is showing
belongs to :mod:`tenbin.webapp.pages` and every question about whether a store can be
read belongs to :mod:`tenbin.webapp.read`. The reason is testability: :func:`route` is a
pure-ish function of a path and the configured stores, so the whole of the app's
behaviour can be asserted without binding a port, and the socket denial the suite keeps
in force stays a real denial rather than a promise.

**Three routes.** ``/`` is the index, ``/store/<name>`` is one store, and anything else is
a 404 that says what was being looked for rather than serving a bare "not found". There is
no fourth route on purpose. This app holds read-only Tanseki credentials for whatever stores
it is configured with, so every route added to it is a route onto those credentials, and
the absence of routes is the reason it can be bound to a loopback address and run by
whoever develops it.

**It binds loopback by default**, and that is a security decision rather than a
convenience. :data:`DEFAULT_HOST` is ``127.0.0.1`` because this app has no
authentication of its own: bound to every interface it would serve a measurement of a
colleague's corpus, and the presence of a report, to anybody who could reach the host.
An operator who means to publish it has to say so with ``--host``, and having to say so is
the point -- see :func:`main`.

**Nothing is logged but a method and a path.** :meth:`TenbinHandler.log_message`
overrides the base class's, which logs ``self.requestline`` -- the request target exactly
as it arrived, query string and all. A browser that appends a token to a URL would put it
in the log on that path, and this program's stated rule is that an API key never reaches
a page or a log line. So the query is dropped before anything is written, and the path is
the decoded one this handler actually matched rather than the raw bytes, because the raw
bytes are whatever a client chose to send.

**A page that raises does not print a traceback.** :meth:`_respond` catches, answers 500
with the exception's class and nothing else, and the traceback goes to the server's own
error output. That is not politeness about stack traces: a Python traceback renders the
locals it unwinds through, and a local on this path is a
:class:`~tenbin.webapp.config.StoreEntry`, whose ``settings`` hold an API key. The class
name is the most this app will say about an unexpected failure to somebody who can load a
URL.
"""

from __future__ import annotations

import html
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Final, NoReturn
from urllib.parse import unquote, urlsplit

from tenbin.webapp.config import StoreEntry, StoreListError, load_stores
from tenbin.webapp.pages import index_page, report_page
from tenbin.webapp.read import read_listings, read_store

#: Where the server listens unless an operator says otherwise. Loopback, for the reason
#: in the module docstring: this app has no authentication and holds credentials.
DEFAULT_HOST: Final[str] = "127.0.0.1"

#: The default port. Not Tanseki's 8088 or 8099, because those are a store's and this app is
#: not a store; an operator who put this on 8099 would be answering for something else.
DEFAULT_PORT: Final[int] = 8090

#: The prefix a store is addressed under. A path rather than a query parameter so that a
#: store's name is never a URL *value* -- query strings end up in logs, in ``Referer``
#: headers and in browser history, and this app would rather a store's name were in none
#: of those.
STORE_PATH_PREFIX: Final[str] = "/store/"

#: Who may put this app in a frame, when an operator has not said. ``'self'`` rather than
#: no policy at all: absent ``frame-ancestors`` every site on the internet may frame these
#: pages, and a report surface framed by a hostile page is a report surface a reader
#: cannot tell came from here. Same-origin framing is the posture that covers the ordinary
#: deployment -- this app behind the reverse proxy that also serves the page embedding it
#: -- and it is a decision rather than an absence.
DEFAULT_FRAME_ANCESTORS: Final[str] = "'self'"

#: What may not appear in an operator-supplied framing policy. A ``frame-ancestors`` value
#: is concatenated into a response header, so the characters that matter are the ones that
#: would end the header or start another one: CR and LF end it, and ``;`` starts a second
#: CSP directive, which is how a framing allowlist becomes a ``script-src``. Refused rather
#: than escaped, because escaping a value into a header produces a header that says
#: something other than what the operator asked for, and a policy that is not the policy
#: is worse than a refusal that is legible.
_FORBIDDEN_IN_FRAME_ANCESTORS: Final[frozenset[str]] = frozenset({"\r", "\n", ";", "\0"})


class FrameAncestorsError(ValueError):
    """An operator supplied a framing policy this app refuses to send."""


def check_frame_ancestors(value: str) -> str:
    """Return ``value`` if it is a framing policy, and refuse it in a sentence if not.

    Validated rather than concatenated. The check is deliberately about the characters
    that could break out of the header rather than about parsing origins: a full grammar
    for ``frame-ancestors`` would reject configurations that are correct, and the failure
    this guards against is injection, not a typo in a hostname. A bare ``.`` is still
    refused, because it is far more likely to be a shell that ate a wildcard than an
    operator who meant something specific by it.
    """
    if not value:
        raise FrameAncestorsError("the framing policy is empty.")
    found = sorted(_FORBIDDEN_IN_FRAME_ANCESTORS & set(value))
    if found:
        raise FrameAncestorsError(
            "the framing policy may not contain "
            + ", ".join(repr(character) for character in found)
            + ". A semicolon would start a second Content-Security-Policy directive and a "
            "newline would end the header."
        )
    return value


#: What the server says in its own banner. The base class's default sends the Python
#: implementation and version to every client that asks, which is free reconnaissance for
#: anything that reaches the port.
SERVER_NAME: Final[str] = "tenbin"


@dataclass(frozen=True)
class Response:
    """A status and the bytes to send with it.

    A value rather than a write, because :func:`route` has to be able to answer without a
    socket: the handler's whole job is to take one of these and put it on the wire, and a
    function that wrote bytes itself could not be asserted on.
    """

    status: int
    body: str
    content_type: str = "text/html; charset=utf-8"


def _headers(
    response: Response, *, frame_ancestors: str = DEFAULT_FRAME_ANCESTORS
) -> Sequence[tuple[str, str]]:
    """Every header one response carries, as ``(name, value)`` pairs.

    A pure function rather than a run of ``send_header`` calls inside the handler, because
    the property worth testing about a framing policy is *which headers it produced* and a
    test that has to bind a socket to learn a header name has tested the socket. The same
    reason :func:`route` returns a value rather than writing: the suite denies sockets, and
    a policy that can only be verified by opening a port is a policy nobody verifies.

    **No ``X-Frame-Options``, deliberately.** It is ``DENY`` or ``SAMEORIGIN`` only, so it
    cannot express the allowlist an operator may have configured, and emitting both headers
    means the more restrictive one wins -- which would silently defeat a configured
    ``frame-ancestors`` and leave the operator believing their parent page is allowed.
    One header, the one that can say what was meant.
    """
    body = response.body.encode("utf-8")
    return (
        ("Content-Type", response.content_type),
        ("Content-Length", str(len(body))),
        # No caching. Every request re-reads its stores, so a cached copy would be a page
        # describing a corpus from whenever the cache filled -- including its completeness
        # sentence, which is the one line a reader is relying on.
        ("Cache-Control", "no-store"),
        ("Content-Security-Policy", f"frame-ancestors {frame_ancestors}"),
    )


def route(path: str, entries: Sequence[StoreEntry]) -> Response:
    """The page for one path, or the sentence saying there is not one.

    Split out of the handler for the reason in the module docstring: this is the whole of
    the app's behaviour and it is callable without a socket, so the rules worth testing
    are tested directly rather than through a client.

    **A query string is ignored rather than rejected.** It is dropped by the caller before
    the path arrives here, and a cache-buster or a shareable anchor on a link is not this
    app's business. What *is* this app's business is that nothing it renders comes from
    one: no value read from the URL reaches a page.

    **An unknown store is a 404 naming the stores that exist**, which is the useful half
    of the answer. A bare 404 tells an operator their bookmark is stale; a 404 that lists
    the names tells them the name they typed is not one of them.
    """
    target = unquote(urlsplit(path).path)
    if target in ("", "/"):
        return Response(200, index_page(read_listings(entries)))
    if target.startswith(STORE_PATH_PREFIX):
        name = target.removeprefix(STORE_PATH_PREFIX).rstrip("/")
        entry = next((candidate for candidate in entries if candidate.name == name), None)
        if entry is None:
            return Response(404, _not_found_page(entries, name))
        return Response(200, report_page(read_store(entry)))
    return Response(404, _not_found_page(entries, target))


class TenbinServer(ThreadingHTTPServer):
    """The listening socket, holding the configuration it serves.

    The stores live on the server rather than in a global because the handler gets here
    through ``self.server`` and there is no other door: a module-level list of stores
    would be a thing two running servers in one process would share, and this app's
    configuration is exactly the thing that must not be shared between them. The framing
    policy is here for the same reason and not because it needed somewhere to go: two
    running servers in one process may legitimately disagree about who may frame them.
    """

    daemon_threads = True
    entries: tuple[StoreEntry, ...] = ()
    frame_ancestors: str = DEFAULT_FRAME_ANCESTORS


class TenbinHandler(BaseHTTPRequestHandler):
    """GET and nothing else.

    ``do_POST`` and the rest are left undefined, so the base class answers them with a 501
    and a sentence of its own. This app reads and does not write -- not to Tanseki, and not
    to itself -- and a handler that defined a verb it had no business defining would be a
    verb somebody eventually wired to something.
    """

    server_version = SERVER_NAME
    #: The Python version, suppressed. The base class puts it in the ``Server`` header of
    #: every response, which is a version string handed to every client for no reason.
    sys_version = ""
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802 - the name is BaseHTTPRequestHandler's
        """Answer one request, and never let an exception become a traceback on a page."""
        entries = self._entries()
        try:
            response = route(self.path, entries)
        except Exception as exc:  # noqa: BLE001 - the reason is in the docstring above
            response = Response(500, _broken_page(exc))
            self._emit(response)
            self.log_error("a page could not be rendered: %s", type(exc).__name__)
            return
        self._emit(response)

    def _entries(self) -> tuple[StoreEntry, ...]:
        """The configured stores, read off the server this handler is serving."""
        return getattr(self.server, "entries", ())

    def _frame_ancestors(self) -> str:
        """The configured framing policy, read off the server this handler is serving."""
        return getattr(self.server, "frame_ancestors", DEFAULT_FRAME_ANCESTORS)

    def _emit(self, response: Response) -> None:
        """Write one response, with the headers a browser needs to render it correctly."""
        body = response.body.encode("utf-8")
        self.send_response(response.status)
        for name, value in _headers(response, frame_ancestors=self._frame_ancestors()):
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        """Log the method and the matched path, and never the request target.

        Overriding rather than trusting the base class because the base class logs
        ``self.requestline``, which is the request target as it arrived -- query string
        included. Everything in a query string on a page about a credential is something a
        link can carry, so the query is dropped here rather than promised against.
        """
        target = unquote(urlsplit(self.path).path)
        sys.stderr.write(f"{self.address_string()} {self.command} {target}\n")


def build_server(
    entries: Sequence[StoreEntry],
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    frame_ancestors: str = DEFAULT_FRAME_ANCESTORS,
) -> TenbinServer:
    """Bind a server for ``entries``, without serving it.

    Separated from :func:`serve` so a caller can bind an ephemeral port and read the one
    it got -- which is how ``python -m tenbin.webapp.server --port 0`` is useful, and how
    a caller embedding this in something larger would use it.

    The framing policy is checked here rather than at send time, so a caller that embeds
    this in a larger server finds out at bind that its policy is unusable rather than on
    the first request. Refusing to bind is the right time: a server that starts and then
    serves pages nobody may frame has already failed.
    """
    server = TenbinServer((host, port), TenbinHandler)
    server.entries = tuple(entries)
    server.frame_ancestors = check_frame_ancestors(frame_ancestors)
    return server


def serve(
    entries: Sequence[StoreEntry],
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    frame_ancestors: str = DEFAULT_FRAME_ANCESTORS,
) -> NoReturn:
    """Bind and serve until interrupted, saying where it is listening first.

    The address is printed to stderr rather than stdout because a caller may have piped
    stdout somewhere, and this program has a rule about what a pipe receives: the reports,
    and nothing else.
    """
    server = build_server(entries, host=host, port=port, frame_ancestors=frame_ancestors)
    # ``server_port`` rather than slicing ``server_address``: with ``--port 0`` the
    # caller asked for any port and needs to be told which one it got, and this is the
    # attribute the base class sets to it.
    sys.stderr.write(
        f"tenbin: serving {len(entries)} store(s) on http://{host}:{server.server_port}/\n"
    )
    sys.stderr.flush()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        sys.stderr.write("tenbin: stopped\n")
    finally:
        server.server_close()
    raise SystemExit(0)


def main(argv: Sequence[str] | None = None) -> NoReturn:
    """``python -m tenbin.webapp.server <stores.yaml>`` -- run the app.

    **The configuration path is an argument, never an environment variable.** Every other
    setting in this program is ``TENBIN_``-prefixed, and the suite's isolation fixture
    deletes those variables before each test because a behaviour variable a developer
    forgot is a suite that is a function of whoever ran it. A second source of
    configuration outside that list would be one the harness cannot remove -- so this
    takes a path, says so when it cannot read it, and stops.

    A configuration file that cannot be read stops the app before it binds anything,
    rather than serving an index with nothing in it: an empty index looks like a
    deployment where every store is empty, which is the one reading this app exists to
    make impossible.
    """
    arguments = list(sys.argv[1:] if argv is None else argv)
    positional, host, port, frame_ancestors = _arguments(arguments)
    if len(positional) != 1:
        sys.stderr.write(
            "usage: python -m tenbin.webapp.server STORES.yaml [--host HOST] [--port PORT]\n"
            "       [--frame-ancestors POLICY]\n"
            f"  --host defaults to {DEFAULT_HOST} (loopback; this app has no authentication "
            "of its own and holds store credentials)\n"
            f"  --port defaults to {DEFAULT_PORT}\n"
            f"  --frame-ancestors defaults to {DEFAULT_FRAME_ANCESTORS}, which permits "
            "framing by the page that embeds this app and by nobody else. Pass 'none' for "
            "a report surface that is viewable but never embeddable, or a space-separated "
            "list of origins to name the parents that may frame it\n"
        )
        raise SystemExit(2)
    try:
        entries = load_stores(Path(positional[0]))
    except StoreListError as exc:
        sys.stderr.write(f"error: {exc}\n")
        raise SystemExit(2) from exc
    serve(entries, host=host, port=port, frame_ancestors=frame_ancestors)


def _arguments(
    arguments: Sequence[str],
) -> tuple[list[str], str, int, str]:
    """The positional arguments, the host, the port, and the framing policy.

    Parsed here rather than with ``argparse`` because this module must not put a parser's
    help text into a surface whose whole argument is that its failure messages are
    sentences -- and because three flags is not a parser's worth of grammar. A flag without
    a value, or a port that is not a number, stops the run before anything binds.
    """
    positional: list[str] = []
    host, port = DEFAULT_HOST, DEFAULT_PORT
    frame_ancestors = DEFAULT_FRAME_ANCESTORS
    remaining = list(arguments)
    while remaining:
        argument = remaining.pop(0)
        if argument in ("--host", "--port", "--frame-ancestors"):
            if not remaining:
                sys.stderr.write(f"error: {argument} needs a value.\n")
                raise SystemExit(2)
            value = remaining.pop(0)
            if argument == "--host":
                host = value
                continue
            if argument == "--frame-ancestors":
                # Checked here so a typo in a policy is a sentence at the command line
                # rather than a header the operator discovers in a browser's network tab.
                try:
                    frame_ancestors = check_frame_ancestors(value)
                except FrameAncestorsError as exc:
                    sys.stderr.write(f"error: --frame-ancestors: {exc}\n")
                    raise SystemExit(2) from None
                continue
            try:
                port = int(value)
            except ValueError:
                sys.stderr.write(f"error: --port must be a number, not {value!r}.\n")
                raise SystemExit(2) from None
            continue
        if argument.startswith("--"):
            sys.stderr.write(
                f"error: {argument} is not an option this app takes. It takes "
                f"--host, --port and --frame-ancestors, and one path to a store list.\n"
            )
            raise SystemExit(2)
        positional.append(argument)
    return positional, host, port, frame_ancestors


def _not_found_page(entries: Sequence[StoreEntry], asked_for: str) -> str:
    """A 404 that names what exists, because a bare 404 only says the link is stale."""
    names = ", ".join(entry.name for entry in entries) or "none configured"
    return _page(
        "Not found",
        f"<h1>Not found</h1><p>There is no page at "
        f"<code>{_escape(asked_for)}</code> on this app. The configured stores are: "
        f"<strong>{_escape(names)}</strong>.</p>"
        '<p><a href="/">All stores</a></p>',
    )


def _broken_page(exc: Exception) -> str:
    """A 500 that names the class of what went wrong and nothing about the credentials.

    No message text, and deliberately so: an unexpected exception's message is whatever
    the failing layer put in it, and this page is served to anybody who can load the URL.
    The class name is enough for an operator reading a log they already have.
    """
    return _page(
        "Not available",
        f"<h1>Not available</h1><p>This page could not be rendered: "
        f"<code>{_escape(type(exc).__name__)}</code>. Nothing was measured and nothing was "
        "written. The failure is in the server's own output, where the traceback is.</p>",
    )


def _page(title: str, body: str) -> str:
    """The smallest possible document for a response that is not one of the app's pages."""
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{_escape(title)}</title>\n"
        "</head>\n"
        f"<body>\n{body}\n</body>\n"
        "</html>\n"
    )


def _escape(value: object) -> str:
    """Escape a value into a text node, from a store name or an exception's class name."""
    return html.escape(str(value), quote=True)


if __name__ == "__main__":
    main()
