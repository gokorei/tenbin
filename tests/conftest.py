"""Test isolation: a measurement program whose tests can cheat is not measuring.

Three autouse policies, and all three exist because a test that can reach a real
store will eventually be written, and the ones that do are the ones that stop
being run.

**The network is denied -- except to an integration test with an explicit
opt-in.** Tenbin reads a knowledge store over HTTP. A test suite that can open
a socket will grow tests that depend on a store being up, and
the failure mode when it is down is an error that reads like a defect in Tenbin
rather than like a missing daemon. Denying ``connect`` at the socket layer means
a test that needs the store must say so by asking for the fake explicitly, and
the fake is then visible in the test's own source. The exception is a test
marked ``integration`` run with its suite's opt-in set (``TANSEKI_SEAM_STORE``
or ``TICKET_SEAM_BACKEND``): that combination is a test that named the backend
it reaches and the permission to reach it, which is the only shape of network
access this suite tolerates -- and :func:`integration_network_allowed` is the
one place that rule is decided, so a third suite is a variable name rather
than a second rule.

**Credentials and local configuration are removed.** A test that can read a real
``TANSEKI_API_KEY`` is a test that can write to a real store, and the whole point of
the read seam is that Tenbin holds no write capability worth holding. Pointing
``HOME`` at a temporary directory additionally stops a credential file on the
developer's machine from leaking into a test through a settings loader that reads
it.

**Leaks are errors.** ``ResourceWarning`` is promoted to an error, so a handle
a test forgot to close fails that test rather than accumulating. A measurement
program holds a connection to somebody else's data, and a leaked connection
outlives the process that opened it.

There is deliberately no autouse fixture that closes every client a test
constructs. Registering a client somewhere so a fixture can find it is a thing
every test then has to remember, and a test that forgets is worse than one that
had to write two extra lines. Tests that open a client close it, and
``tests/fakes.py`` provides a ``closing()`` context manager so that costs one
line. The ``ResourceWarning`` policy is what catches the ones that do not.

This module is the only ``conftest.py`` in the repository. Per-module fixtures
belong next to the tests that use them.
"""

from __future__ import annotations

import os
import re
import socket
from collections.abc import Generator, Iterable, Mapping
from pathlib import Path
from typing import Final

import pytest

#: Environment variables that change what Tenbin does. Every one is deleted
#: before each test so a developer's shell cannot change a test's outcome.
#: Keep alphabetical, and keep in step with :mod:`tenbin.config`.
#:
#: The list covers **both** settings models, not just ``Settings``. It did not
#: for a while: the three ``ticket_*`` variables were absent while
#: ``TicketSettings`` existed, so a shell carrying a real ticket backend URL
#: could have leaked into any test that read it. The comment in
#: :mod:`tenbin.config` claims every behaviour field is listed here, and that
#: claim is only worth making if it is true -- so it is true, and a new field
#: added to either model without a line here is a gap this harness will not
#: catch on its own.
BEHAVIOR_ENV_VARS = (
    "TENBIN_ENV_FILE",
    "TENBIN_PERIOD_ANCHOR",
    "TENBIN_PERIOD_CADENCE",
    "TENBIN_READ_MODEL_PATH",
    "TENBIN_TANSEKI_API_KEY",
    "TENBIN_TANSEKI_COLLECTION",
    "TENBIN_TANSEKI_TIMEOUT_SECONDS",
    "TENBIN_TANSEKI_URL",
    "TENBIN_TICKET_API_TOKEN",
    "TENBIN_TICKET_API_URL",
    "TENBIN_TICKET_API_USER",
    "TENBIN_TICKET_SOURCE",
    "TENBIN_TICKET_TIMEOUT_SECONDS",
)


def pytest_configure(config: pytest.Config) -> None:
    """Promote resource leaks to errors, because a leak here is a real leak.

    An unclosed client is a connection to the knowledge store that outlives the
    process that opened it. Kojutsu made the same call for the same class of
    resource, and a program that reads other people's records has no business
    being casual about sockets.
    """
    config.addinivalue_line("filterwarnings", "error::ResourceWarning")
    config.addinivalue_line("filterwarnings", "error::pytest.PytestUnraisableExceptionWarning")


#: Terminal styling, matched so assertions read sentences rather than escapes.
#:
#: Two ways styling breaks a substring assertion, and both have bitten: click
#: highlights anything shaped like an option inside an error message, so
#: ``--output`` arrives as two separately-styled spans (``-`` and ``-output``)
#: with no contiguous ``--output`` anywhere; and Rich closes and reopens the
#: surrounding style at every wrapped line, so a phrase spanning a wrap carries
#: an escape and a newline in its middle. GitHub Actions sets ``FORCE_COLOR``,
#: which is why a suite green on a developer's machine goes red there -- the
#: styling is present in one place and absent in the other. Stripping it makes
#: the assertion independent of both the palette and the width.
_ANSI_ESCAPE: Final[re.Pattern[str]] = re.compile(r"\x1b\[[0-9;]*m")


def plain_output(text: str) -> str:
    """Output with terminal styling removed, for assertions on what was said.

    Styling only, never content: wrapping newlines stay for the caller to
    collapse, because a helper that also rewrapped would be asserting on its
    own layout rather than the renderer's.
    """
    return _ANSI_ESCAPE.sub("", text)


def _deny_socket_connect(_socket: socket.socket, address: object) -> None:
    raise RuntimeError(f"Unexpected network access in tests: {address!r}")


def _deny_create_connection(address: object, *args: object, **kwargs: object) -> None:
    raise RuntimeError(f"Unexpected network access in tests: {address!r}")


#: The opt-ins that permit an integration-marked test to open a socket. Either
#: one opens it; each suite still skips unless its own backend is configured,
#: so the permission is never used without the suite having read its own
#: opt-in first. A third suite adds its variable here rather than writing its
#: own socket fixture -- the repository has had one of those, and two ways of
#: permitting the same access is how a third one arrives unnoticed.
SEAM_OPT_IN_VARIABLES: tuple[str, ...] = ("TANSEKI_SEAM_STORE", "TICKET_SEAM_BACKEND")


def integration_network_allowed(marks: Iterable[str], environ: Mapping[str, str]) -> bool:
    """Whether a test with these marks may open a socket in this environment.

    Both halves, because each half alone is a different failure: a mark
    without an opt-in is a test that reaches a backend as a side effect of
    existing, and an opt-in without a mark is a unit test that quietly depends
    on a daemon being up. Pure, so the harness pins it without opening
    anything -- see ``test_integration_marks_keep_the_denial_without_an_opt_in``
    in ``tests/test_harness.py``.
    """
    return "integration" in set(marks) and any(
        environ.get(variable, "").strip() for variable in SEAM_OPT_IN_VARIABLES
    )


@pytest.fixture(autouse=True)
def env_isolate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, request: pytest.FixtureRequest
) -> Generator[None, None, None]:
    """Isolate tests from local configuration, credentials, state, and network.

    ``HOME`` moves to a temporary directory so a settings loader that reads a
    credential file finds nothing, and the socket is closed off so a test that
    forgets to inject a fake fails loudly instead of quietly reaching a store.
    An integration-marked test with its suite's opt-in set keeps the real
    socket instead -- see :func:`integration_network_allowed` for the rule.
    """
    for key in BEHAVIOR_ENV_VARS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TENBIN_ENV_FILE", "")
    monkeypatch.setenv("HOME", str(tmp_path))
    marks = {mark.name for mark in request.node.iter_markers()}
    if not integration_network_allowed(marks, os.environ):
        monkeypatch.setattr(socket.socket, "connect", _deny_socket_connect)
        monkeypatch.setattr(socket.socket, "connect_ex", _deny_socket_connect)
        monkeypatch.setattr(socket, "create_connection", _deny_create_connection)
    yield
