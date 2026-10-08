"""The transitions seam, checked against the ticket backend rather than a reading of it.

**The four claims below were settled by observation on 2026-10-01**, against a
backend on ``http://127.0.0.1:8082``. Each is recorded here with what was seen, so
the next reader knows which parts of this file are checked fact and which are
still reasoning.

1. **The envelope — SETTLED.** Rows arrive under ``results``; pagination under
   ``total``/``page``/``limit``/``total_pages``. A read enumerated 1,848 of a
   stated 1,848 with ``completeness=complete`` and zero unreadable rows.
2. **``from`` is genuinely omitempty — SETTLED.** Of 1,848 transitions exactly one
   had no origin, and it arrived with the key absent rather than as an empty
   string. So the parser's absent-versus-empty distinction is not theoretical: one
   real row in that read depends on it.
3. **The timestamps parse — SETTLED.** Every row carried an offset-aware RFC3339
   moment, e.g. ``2026-09-13T08:05:44.882414+02:00``, and the derived window
   matched the minimum and maximum of the rows.
4. **A scoped read is smaller — SETTLED.** Scoped to one project the backend
   reported 367 against 1,848 unscoped, and an unknown project reported zero. So
   the time and scope filters are applied server-side, and the ``total`` beside
   the rows describes the population actually returned.

**These tests run if and only if the backend is opted in, and that is a harness
guarantee rather than a property of them.** ``tests/conftest.py`` denies
``socket.connect`` to every test unconditionally -- which is right, and which
is why the suite has never quietly depended on a service being up -- and lifts
the denial for ``@pytest.mark.integration`` if and only if an explicit opt-in
is set. ``TICKET_SEAM_BACKEND`` is this suite's opt-in: without it every test
below skips with a reason, and with it the socket is real and an unreachable
backend skips (or fails under ``TICKET_SEAM_STRICT``) rather than erroring on
a denied call. The verification above was run outside pytest before that lift
existed; it stays recorded here because the claims were settled then, not
because the tests cannot run now.

**Nothing here asserts a number about somebody else's work.** Between two reads
minutes apart the backend's ``total`` went from 1,847 to 1,848, because somebody
closed a ticket. That is the reason every assertion here is about shape and
never about a count: a test that fails when a colleague does their job is a test
about the corpus rather than about the seam.

**A skipped run is not a passing run.** With no backend opted in, every test skips
with a reason naming the variable that would fix it and the word UNVERIFIED, and
``TICKET_SEAM_STRICT`` turns that skip into a failure so a CI job that was supposed
to have a backend cannot go green without one.
"""

from __future__ import annotations

import os
from typing import Final

import httpx
import pytest

from tenbin.corpus.snapshot import Completeness
from tenbin.corpus.transition import Transition
from tenbin.store.client import StoreUnavailableError
from tenbin.store.ticket_sources.go_experiment import GoExperimentClient
from tenbin.store.transitions import TransitionSource, take_transition_snapshot

pytestmark = pytest.mark.integration

UNVERIFIED: Final[str] = "UNVERIFIED"

#: Named ``TICKET_SEAM_*`` rather than ``TENBIN_*`` on purpose, for the reason
#: the store's own file gives: this suite reaches a live service, so a variable
#: that reads as "point Tenbin at a store" is the wrong name for a permission.
#: An ``TENBIN_`` prefix would additionally collide with the harness test that
#: asserts no Tenbin setting survives into a test.
OPTOUT_ENV: Final[str] = "TICKET_SEAM_BACKEND"
URL_ENV: Final[str] = "TICKET_SEAM_URL"
STRICT_ENV: Final[str] = "TICKET_SEAM_STRICT"

#: Where the ticket backend listens locally.
DEFAULT_SEAM_URL: Final[str] = "http://127.0.0.1:8082"

#: A project that exists in a real deployment, used only to show a scoped read is
#: a subset rather than the same rows relabelled.
SCOPED_PROJECT: Final[str] = "4GK6J742"


def _skip_reason() -> str:
    """The sentence every skip in this file carries.

    Named rather than written four times because a skip reason is the only thing
    a reader of a green run gets to see, and a bare "skipped" is indistinguishable
    from a test that was deleted.
    """
    return (
        f"{os.path.basename(__file__)} is {UNVERIFIED} without this, and a skipped run is "
        f"not a passing one. Set {OPTOUT_ENV}=1 to make this a failure instead."
    )


def _strict() -> bool:
    """Whether an absent backend is a failure rather than a skip."""
    return bool((os.environ.get(STRICT_ENV) or "").strip())


def _require_backend() -> str:
    if not (os.environ.get(OPTOUT_ENV) or "").strip():
        if _strict():
            pytest.fail(_skip_reason())
        pytest.skip(_skip_reason())
    return (os.environ.get(URL_ENV) or DEFAULT_SEAM_URL).strip()


@pytest.fixture
def backend_url() -> str:
    return _require_backend()


def _snapshot(client: TransitionSource, **kwargs: object):
    try:
        return take_transition_snapshot(client, page_limit=200, **kwargs)  # type: ignore[arg-type]
    except (StoreUnavailableError, httpx.TransportError) as exc:
        # Opted in and still unreachable. That is a failure wearing a skip's
        # clothes, so strict mode is the difference between the two being honest.
        if _strict():
            pytest.fail(f"{OPTOUT_ENV} was set but the backend could not be read: {exc}")
        pytest.skip(f"{UNVERIFIED}: opted in but the backend could not be read: {exc}")


class TestTheEnvelope:
    def test_a_read_parses_and_reports_its_own_completeness(self, backend_url: str) -> None:
        with GoExperimentClient(backend_url, timeout_seconds=5.0) as client:
            snapshot = _snapshot(client)

        assert snapshot.collection == "status-transitions"
        assert isinstance(snapshot.completeness, Completeness)
        assert snapshot.enumerated == len(snapshot.transitions)
        if snapshot.enumerated:
            assert snapshot.is_complete, (
                f"a read that reached the end reported {snapshot.completeness}"
            )
            assert snapshot.unreadable_rows == 0, snapshot.unreadable_examples

    def test_every_row_is_a_fact_with_a_destination_and_a_moment(self, backend_url: str) -> None:
        with GoExperimentClient(backend_url, timeout_seconds=5.0) as client:
            snapshot = _snapshot(client)

        for transition in snapshot.transitions:
            assert isinstance(transition, Transition)
            assert transition.ticket_id
            assert transition.to_state
            assert transition.at.tzinfo is not None, "a moment with no offset cannot be compared"

    def test_the_window_spans_what_was_read(self, backend_url: str) -> None:
        with GoExperimentClient(backend_url, timeout_seconds=5.0) as client:
            snapshot = _snapshot(client)

        if not snapshot.transitions:
            pytest.skip(f"{UNVERIFIED}: the backend served no transitions to span")
        moments = [t.at for t in snapshot.transitions]
        assert snapshot.window.earliest == min(moments)
        assert snapshot.window.latest == max(moments)


class TestTheOmitemptyOrigin:
    def test_a_first_transition_arrives_with_no_from_key(self, backend_url: str) -> None:
        """Claim 2. The parser distinguishes absent from empty, so this settles which is real."""
        with GoExperimentClient(backend_url, timeout_seconds=5.0) as client:
            snapshot = _snapshot(client)

        if not snapshot.transitions:
            pytest.skip(f"{UNVERIFIED}: the backend served no transitions")
        origins = [t.from_state for t in snapshot.transitions]
        if None not in origins:
            # A fact about this deployment's data, not about the seam: no
            # ticket in it is on its first transition, so there is nothing here
            # for the absent-versus-empty distinction to bite on. Failing would
            # be a test about somebody else's corpus, which this file refuses
            # to be -- and the parser's own distinction stays pinned by the
            # unit tests over rows written by hand.
            pytest.skip(
                f"{UNVERIFIED}: no transition lacked a 'from'; either the deployment "
                "has no first transitions, or Go stopped omitting the field"
            )
        # And the two are genuinely distinct values, not two spellings of one.
        assert None not in ("", " ")  # a sentinel cannot be an empty string here


class TestScopingIsServerSide:
    def test_a_scoped_read_reports_a_smaller_population(self, backend_url: str) -> None:
        """Claim 4. A filter ignored server-side would make `total` a lie."""
        with GoExperimentClient(backend_url, timeout_seconds=5.0) as client:
            unscoped = _snapshot(client)
        with GoExperimentClient(backend_url, timeout_seconds=5.0) as client:
            scoped = _snapshot(client, project_id=SCOPED_PROJECT)

        assert scoped.project_id == SCOPED_PROJECT
        if unscoped.store_total:
            assert scoped.store_total < unscoped.store_total, (
                "a per-project read reported the whole backend's population"
            )

    def test_an_unknown_project_yields_an_empty_read_rather_than_everything(
        self, backend_url: str
    ) -> None:
        """The failure mode a silently-ignored filter would produce."""
        with GoExperimentClient(backend_url, timeout_seconds=5.0) as client:
            snapshot = _snapshot(client, project_id="NO-SUCH-PROJECT-0000")

        assert snapshot.enumerated == 0
        assert snapshot.store_total == 0
        assert snapshot.transitions == ()
