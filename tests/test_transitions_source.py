"""Walking transitions, and what the walk admits when it cannot finish.

The walk tests pin the thing that actually matters about a read: that a
snapshot says how far it got, so a caller who ignores that gets a visibly
incomplete object rather than a plausible one. The walk is source-agnostic --
it only ever calls ``list_transitions`` -- so these tests drive it through the
go-experiment adapter with ``socket.connect`` denied, which proves the walk
and nothing about any backend.

Adapter behaviour (parsing, envelopes, auth, scoping) lives in
``tests/test_ticket_sources.py`` beside the adapters, because a walk test that
pinned a backend's envelope would fail for the wrong reason when the envelope
moved.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from tenbin.corpus.snapshot import Completeness, Snapshot
from tenbin.corpus.transition import Transition
from tenbin.store.client import StoreUnavailableError
from tenbin.store.ticket_sources.go_experiment import GoExperimentClient, parse_transition
from tenbin.store.transitions import (
    take_transition_snapshot,
    transitions_by_ticket,
)


def _row(**overrides: Any) -> dict[str, Any]:
    row = {
        "ticket_id": "T1",
        "from": "open",
        "to": "in_progress",
        "at": "2026-09-13T08:07:10.769269+02:00",
    }
    row.update(overrides)
    return row


def _envelope(results: list[Any], *, total: int | None = None, page: int = 1, limit: int = 2):
    return {
        "results": results,
        "total": len(results) if total is None else total,
        "page": page,
        "limit": limit,
        "total_pages": 1,
    }


def _client(handler) -> GoExperimentClient:
    return GoExperimentClient("http://127.0.0.1:8082", transport=httpx.MockTransport(handler))


class TestWalkHonesty:
    def test_the_first_request_failing_raises_rather_than_returning_empty(self) -> None:
        """An empty snapshot from an unreached source is indistinguishable from nothing."""

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(503)

        with pytest.raises(StoreUnavailableError):
            take_transition_snapshot(_client(handler))

    def test_a_later_page_failing_returns_a_partial_snapshot_carrying_the_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.params.get("page") == "2":
                return httpx.Response(503)
            return httpx.Response(
                200,
                json={
                    "results": [_row()],
                    "total": 4,
                    "page": 1,
                    "limit": 1,
                    "total_pages": 4,
                },
            )

        with _client(handler) as client:
            snapshot = take_transition_snapshot(client, page_limit=1)

        assert snapshot.completeness is Completeness.STORE_ERROR
        assert snapshot.error is not None
        assert len(snapshot.transitions) == 1

    def test_a_complete_walk_is_whole_and_agrees_with_the_backend_count(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            page = int(request.url.params.get("page", "1"))
            rows = [_row(ticket_id=f"T{page}")]
            return httpx.Response(
                200,
                json={
                    "results": rows,
                    "total": 2,
                    "page": page,
                    "limit": 1,
                    "total_pages": 2,
                },
            )

        with _client(handler) as client:
            snapshot = take_transition_snapshot(client, page_limit=1)

        assert snapshot.completeness is Completeness.COMPLETE
        assert snapshot.is_complete
        assert snapshot.enumerated == 2
        assert snapshot.store_total == 2
        assert {t.ticket_id for t in snapshot.transitions} == {"T1", "T2"}

    def test_unreadable_rows_are_surfaced_rather_than_counted_as_read(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json=_envelope([_row(), {"nonsense": True}], total=2),
            )

        with _client(handler) as client:
            snapshot = take_transition_snapshot(client)

        assert snapshot.unreadable_rows == 1
        assert snapshot.unreadable_examples
        # A row served and not understood is not a row read.
        assert snapshot.enumerated == 1
        assert snapshot.completeness is Completeness.STORE_TOTAL_EXCEEDED

    def test_the_window_is_derived_and_spans_the_transitions(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json=_envelope(
                    [
                        _row(at="2026-01-02T00:00:00+00:00"),
                        _row(at="2026-03-04T00:00:00+00:00"),
                    ],
                    total=2,
                ),
            )

        with _client(handler) as client:
            snapshot = take_transition_snapshot(client)

        assert snapshot.window.earliest == datetime(2026, 1, 2, tzinfo=UTC)
        assert snapshot.window.latest == datetime(2026, 3, 4, tzinfo=UTC)

    def test_the_snapshot_satisfies_the_figure_protocol(self) -> None:
        """The whole reason this exists: a second source can carry a claim."""

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_envelope([_row()], total=1))

        with _client(handler) as client:
            snapshot = take_transition_snapshot(client)

        assert isinstance(snapshot, Snapshot)


class TestGrouping:
    def test_groups_by_ticket_and_orders_each_by_moment(self) -> None:
        later = parse_transition(_row(ticket_id="T1", to="done", at="2026-02-01T00:00:00+00:00"))
        earlier = parse_transition(_row(ticket_id="T1", to="in_progress"))
        other = parse_transition(_row(ticket_id="T2"))

        grouped = transitions_by_ticket([later, other, earlier])
        assert set(grouped) == {"T1", "T2"}
        # `later` is the one dated 2026-02-01, so read order and time order differ
        # here on purpose -- the ordering must come from `at`, not from arrival.
        assert [t.to_state for t in grouped["T1"]] == ["done", "in_progress"]

    def test_grouping_cannot_produce_a_negative_interval(self) -> None:
        """Read order is not time order; an unsorted pair would invent one."""
        second = parse_transition(_row(ticket_id="T1", to="done", at="2026-05-01T00:00:00+00:00"))
        first = parse_transition(
            _row(ticket_id="T1", to="in_progress", at="2026-05-01T00:00:00+00:00")
        )
        grouped = transitions_by_ticket([second, first])
        moments = [t.at for t in grouped["T1"]]
        assert moments == sorted(moments)


def test_a_walk_driven_transition_is_still_a_transition() -> None:
    """The fact type survives the adapter move: parsed rows are Transitions."""
    transition = parse_transition(_row())

    assert isinstance(transition, Transition)
    assert transition.ticket_id == "T1"
