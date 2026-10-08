"""One adapter per ticketing system, each projecting its wire onto a Transition.

The go-experiment tests moved here from ``test_transitions_source.py`` with
the client: the walk tests stayed there because the walk is source-agnostic,
and everything that pins a backend's envelope moved here because an envelope
belongs to its adapter. The Jira and Linear tests pin shapes that were checked
against public API references rather than a live instance (see each adapter
module for what was verified and what was assumed), so a drift in either API
fails here first -- as unreadable rows in production, as red tests here.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pydantic
import pytest

from tenbin.config import TicketSettings
from tenbin.corpus.transition import TransitionParseError
from tenbin.store.client import StoreResponseError, StoreUnavailableError
from tenbin.store.ticket_sources import (
    SUPPORTED_SOURCES,
    GoExperimentClient,
    JiraClient,
    LinearClient,
    TicketConfigurationError,
    YouTrackClient,
    ticket_source_client,
)
from tenbin.store.ticket_sources.go_experiment import PAGE_LIMIT as GO_EXPERIMENT_LIMIT
from tenbin.store.ticket_sources.go_experiment import _page_from_json, parse_transition
from tenbin.store.ticket_sources.jira import PAGE_LIMIT as JIRA_LIMIT
from tenbin.store.transitions import TransitionSource


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


def _go_client(handler) -> GoExperimentClient:
    return GoExperimentClient("http://127.0.0.1:8082", transport=httpx.MockTransport(handler))


class TestGoExperimentParsing:
    def test_reads_a_complete_row(self) -> None:
        transition = parse_transition(_row())
        assert transition.ticket_id == "T1"
        assert transition.from_state == "open"
        assert transition.to_state == "in_progress"
        # The backend sends an offset-aware RFC3339 moment. `fromisoformat` keeps
        # that offset rather than normalising, and an aware comparison is by
        # instant -- so this asserts 08:07+02:00 equals 06:07Z, which is the
        # property that matters: no information is invented or lost.
        assert transition.at.utcoffset() == timedelta(hours=2)
        assert transition.at == datetime(2026, 9, 13, 6, 7, 10, 769269, tzinfo=UTC)

    def test_an_absent_origin_is_none_not_a_sentinel(self) -> None:
        """An omitted origin is real data: the first transition of a ticket's life."""
        row = _row()
        del row["from"]
        assert parse_transition(row).from_state is None

    def test_an_explicit_null_origin_is_also_none(self) -> None:
        assert parse_transition(_row(**{"from": None})).from_state is None

    @pytest.mark.parametrize(
        "missing", ["ticket_id", "to", "at"], ids=["no-ticket", "no-to", "no-at"]
    )
    def test_refuses_a_row_missing_a_required_field(self, missing: str) -> None:
        row = _row()
        del row[missing]
        with pytest.raises(TransitionParseError, match=missing):
            parse_transition(row)

    @pytest.mark.parametrize(
        "field,value",
        [("ticket_id", ""), ("to", "   "), ("at", 12345), ("at", "not-a-date")],
    )
    def test_refuses_a_row_carrying_the_wrong_type(self, field: str, value: Any) -> None:
        with pytest.raises(TransitionParseError):
            parse_transition(_row(**{field: value}))

    def test_refuses_a_blank_origin_rather_than_treating_it_as_absent(self) -> None:
        """Absent and empty-string are different claims; only one is true."""
        with pytest.raises(TransitionParseError, match="from"):
            parse_transition(_row(**{"from": ""}))

    def test_refuses_a_row_that_is_not_an_object(self) -> None:
        with pytest.raises(TransitionParseError):
            parse_transition(["not", "an", "object"])


class TestGoExperimentClient:
    def test_rejects_a_response_with_no_results_list(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"total": 0, "page": 1, "limit": 2})

        with pytest.raises(StoreResponseError, match="results"), _go_client(handler) as client:
            client.list_transitions()

    def test_a_missing_endpoint_is_a_contract_error_not_a_transport_one(self) -> None:
        """404 means the backend has no such route; retrying will not help."""

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(404)

        with (
            pytest.raises(StoreResponseError, match="no /status-transitions"),
            _go_client(handler) as client,
        ):
            client.list_transitions()

    @pytest.mark.parametrize("status", [401, 403, 500, 503])
    def test_a_refused_or_failing_backend_is_unavailable(self, status: int) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(status)

        with pytest.raises(StoreUnavailableError), _go_client(handler) as client:
            client.list_transitions()

    def test_a_refused_limit_names_the_accepted_range(self) -> None:
        """A 400 from this endpoint is a bad `limit`, and the message says so."""

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(400)

        with pytest.raises(StoreResponseError, match="limit 1..200"), _go_client(handler) as client:
            client.list_transitions()

    def test_the_declared_ceiling_matches_what_a_live_backend_enforced(self) -> None:
        """200 served, 250 refused with a 400, probed against a live backend on 2026-10-01.

        The observation is recorded in the adapter module rather than cited
        into a private repository: the ceiling is a fact about the wire, and
        the clamp test below is what breaks if it moves.
        """
        assert GO_EXPERIMENT_LIMIT == 200

    def test_limit_is_clamped_to_the_backend_ceiling(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["limit"] = request.url.params.get("limit")
            return httpx.Response(200, json=_envelope([], total=0))

        with _go_client(handler) as client:
            client.list_transitions(limit=GO_EXPERIMENT_LIMIT * 10)
        assert int(seen["limit"]) == GO_EXPERIMENT_LIMIT

    def test_the_time_filter_is_sent_rather_than_applied_here(self) -> None:
        """Filtering client-side would make `total` describe another population."""
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["from"] = request.url.params.get("from")
            seen["to"] = request.url.params.get("to")
            return httpx.Response(200, json=_envelope([_row()], total=1))

        with _go_client(handler) as client:
            client.list_transitions(
                since=datetime(2026, 9, 1, tzinfo=UTC), until=datetime(2026, 9, 30, tzinfo=UTC)
            )
        assert seen["from"] == "2026-09-01T00:00:00+00:00"
        assert seen["to"] == "2026-09-30T00:00:00+00:00"

    def test_a_token_is_sent_when_configured_and_omitted_otherwise(self) -> None:
        seen: list[str | None] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.headers.get("authorization"))
            return httpx.Response(200, json=_envelope([], total=0))

        with _go_client(handler) as client:
            client.list_transitions()
        assert seen == [None]

        def handler2(request: httpx.Request) -> httpx.Response:
            seen.append(request.headers.get("authorization"))
            return httpx.Response(200, json=_envelope([], total=0))

        with GoExperimentClient(
            "http://127.0.0.1:8082", token="secret", transport=httpx.MockTransport(handler2)
        ) as client:
            client.list_transitions()
        assert seen[1] == "Bearer secret"

    def test_the_envelope_shape_this_client_expects_is_literal(self) -> None:
        """A literal, so a backend change to the envelope fails here and names itself."""
        import json

        served = json.loads(
            '{"results":[{"ticket_id":"28QKVK61","from":"ready_for_review","to":"done",'
            '"at":"2026-09-13T08:05:44.882414+02:00"}],"total":1847,"page":1,"limit":2,'
            '"total_pages":924}'
        )
        page = _page_from_json(served)
        assert page.total == 1847
        assert page.total_pages == 924
        assert len(page.transitions) == 1
        assert page.transitions[0].ticket_id == "28QKVK61"
        assert page.transitions[0].from_state == "ready_for_review"


def _jira_history(created: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    return {"id": "10001", "created": created, "items": items}


def _status_change(to: str, frm: str | None = "Open") -> dict[str, Any]:
    return {"field": "status", "fieldtype": "jira", "fromString": frm, "toString": to}


def _jira_issue(key: str, histories: list[dict[str, Any]]) -> dict[str, Any]:
    return {"key": key, "changelog": {"histories": histories}}


def _jira_search(issues: list[dict[str, Any]], *, total: int | None = None) -> dict[str, Any]:
    return {
        "startAt": 0,
        "maxResults": 100,
        "total": len(issues) if total is None else total,
        "issues": issues,
    }


def _jira_client(handler, **kwargs: Any) -> JiraClient:
    return JiraClient(
        "https://example.atlassian.net",
        email="reader@example.com",
        token="token",
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


class TestJiraClient:
    def test_searches_the_enhanced_endpoint_with_changelog_expanded(self) -> None:
        """The old `/search` is being removed; the client must not be on it."""
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["path"] = request.url.path
            seen["expand"] = request.url.params.get("expand")
            seen["fields"] = request.url.params.get("fields")
            return httpx.Response(200, json=_jira_search([]))

        with _jira_client(handler) as client:
            client.list_transitions()

        assert seen["path"] == "/rest/api/3/search/jql"
        assert seen["expand"] == "changelog"
        assert seen["fields"] == "key"

    def test_authenticates_with_basic_over_email_and_token(self) -> None:
        import base64

        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["authorization"] = request.headers.get("authorization")
            return httpx.Response(200, json=_jira_search([]))

        with _jira_client(handler) as client:
            client.list_transitions()

        expected = base64.b64encode(b"reader@example.com:token").decode()
        assert seen["authorization"] == f"Basic {expected}"

    def test_construction_without_email_or_token_is_misuse_not_a_401(self) -> None:
        with pytest.raises(TicketConfigurationError, match="TENBIN_TICKET_API_USER"):
            JiraClient("https://example.atlassian.net", token="token")
        with pytest.raises(TicketConfigurationError, match="TENBIN_TICKET_API_TOKEN"):
            JiraClient("https://example.atlassian.net", email="reader@example.com")

    def test_derives_transitions_from_status_entries_and_drops_the_rest(self) -> None:
        """Non-status changelog items are not transitions and are not unreadable rows."""
        issue = _jira_issue(
            "PROJ-1",
            [
                _jira_history(
                    "2026-01-06T09:00:00.000+0000",
                    [
                        _status_change("In Progress", "Open"),
                        {"field": "assignee", "fromString": "ana", "toString": "bo"},
                    ],
                )
            ],
        )

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_jira_search([issue]))

        with _jira_client(handler) as client:
            page = client.list_transitions()

        assert page.total == 1
        (transition,) = page.transitions
        assert transition.ticket_id == "PROJ-1"
        assert transition.from_state == "Open"
        assert transition.to_state == "In Progress"
        assert page.unreadable == 0

    def test_a_null_origin_is_none_and_the_author_is_dropped(self) -> None:
        """The first transition of an issue's life, and the principal left behind."""
        issue = _jira_issue(
            "PROJ-2",
            [
                {
                    "id": "10002",
                    "author": {"displayName": "Ana"},
                    "created": "2026-01-06T09:00:00.000+0000",
                    "items": [_status_change("Open", None)],
                }
            ],
        )

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_jira_search([issue]))

        with _jira_client(handler) as client:
            (transition,) = client.list_transitions().transitions

        assert transition.from_state is None
        assert transition.to_state == "Open"

    def test_a_malformed_status_entry_is_unreadable_not_a_lost_page(self) -> None:
        issue = _jira_issue(
            "PROJ-3",
            [_jira_history("2026-01-06T09:00:00.000+0000", [{"field": "status"}])],
        )

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_jira_search([issue]))

        with _jira_client(handler) as client:
            page = client.list_transitions()

        assert page.transitions == ()
        assert page.unreadable == 1
        assert page.unreadable_examples

    def test_transitions_outside_the_window_are_narrowed_here(self) -> None:
        """The changelog arrives whole; the window is applied to the derived rows."""
        issue = _jira_issue(
            "PROJ-4",
            [
                _jira_history("2025-12-20T09:00:00.000+0000", [_status_change("In Progress")]),
                _jira_history("2026-01-06T09:00:00.000+0000", [_status_change("Done")]),
            ],
        )

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_jira_search([issue]))

        with _jira_client(handler) as client:
            page = client.list_transitions(
                since=datetime(2026, 1, 1, tzinfo=UTC),
                until=datetime(2026, 2, 1, tzinfo=UTC),
            )

        assert [t.to_state for t in page.transitions] == ["Done"]

    def test_the_window_is_sent_as_updated_bounds_and_ordered(self) -> None:
        """JQL scopes the issue population; the narrowing above defines the rows."""
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["jql"] = request.url.params.get("jql")
            return httpx.Response(200, json=_jira_search([]))

        with _jira_client(handler) as client:
            client.list_transitions(
                project_id="PROJ",
                since=datetime(2026, 1, 5, tzinfo=UTC),
                until=datetime(2026, 1, 19, tzinfo=UTC),
            )

        assert seen["jql"] == (
            'project = PROJ AND updated >= "2026-01-05 00:00" '
            'AND updated <= "2026-01-19 00:00" order by updated'
        )

    def test_an_unscoped_search_is_just_an_ordering(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["jql"] = request.url.params.get("jql")
            return httpx.Response(200, json=_jira_search([]))

        with _jira_client(handler) as client:
            client.list_transitions()

        assert seen["jql"] == "order by updated"

    def test_pages_offset_by_startat(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["startAt"] = request.url.params.get("startAt")
            seen["maxResults"] = request.url.params.get("maxResults")
            return httpx.Response(200, json=_jira_search([], total=250))

        with _jira_client(handler) as client:
            page = client.list_transitions(page=3, limit=1000)

        # Clamped to the ceiling, then offset: the third page of a hundred.
        assert seen["maxResults"] == str(JIRA_LIMIT)
        assert seen["startAt"] == str(2 * JIRA_LIMIT)
        assert page.total == 250
        assert page.total_pages == 3

    def test_a_group_scope_is_refused_rather_than_silently_unscoped(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:  # pragma: no cover
            raise AssertionError("no request may be sent for an unscoped read")

        with (
            _jira_client(handler) as client,
            pytest.raises(TicketConfigurationError, match="no group scoping"),
        ):
            client.list_transitions(group_id="anything")

    @pytest.mark.parametrize("status", [401, 403, 500, 503])
    def test_a_refused_or_failing_jira_is_unavailable(self, status: int) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(status)

        with pytest.raises(StoreUnavailableError), _jira_client(handler) as client:
            client.list_transitions()

    def test_a_bad_search_names_the_jql(self) -> None:
        """A 400 is usually an unknown project key; the message carries the query."""

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(400)

        with (
            _jira_client(handler) as client,
            pytest.raises(StoreResponseError, match="project = NOPE"),
        ):
            client.list_transitions(project_id="NOPE")


def _linear_entry(
    created: str, to_name: str | None, from_name: str | None = "Todo"
) -> dict[str, Any]:
    return {
        "createdAt": created,
        "fromWorkflowState": {"name": from_name} if from_name is not None else None,
        "toWorkflowState": {"name": to_name} if to_name is not None else None,
    }


def _linear_issue(
    identifier: str,
    entries: list[dict[str, Any]],
    *,
    uuid: str = "11111111-2222-3333-4444-555555555555",
) -> dict[str, Any]:
    return {"id": uuid, "identifier": identifier, "history": {"nodes": entries}}


def _linear_payload(
    issues: list[dict[str, Any]], *, has_next: bool = False, cursor: str | None = None
) -> dict[str, Any]:
    return {
        "data": {
            "issues": {
                "nodes": issues,
                "pageInfo": {"hasNextPage": has_next, "endCursor": cursor},
            }
        }
    }


def _linear_client(handler, **kwargs: Any) -> LinearClient:
    return LinearClient(
        "https://api.linear.app",
        token="lin_api_key",
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


class TestLinearClient:
    def test_posts_graphql_with_the_team_and_window_as_variables(self) -> None:
        """The filter rides variables, not string interpolation: no GraphQL is assembled."""
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            payload = json.loads(request.content.decode())
            seen["path"] = request.url.path
            seen["variables"] = payload["variables"]
            seen["query"] = payload["query"]
            return httpx.Response(200, json=_linear_payload([]))

        with _linear_client(handler) as client:
            client.list_transitions(
                project_id="ENG",
                since=datetime(2026, 1, 5, tzinfo=UTC),
                until=datetime(2026, 1, 19, tzinfo=UTC),
            )

        assert seen["path"] == "/graphql"
        assert "issues" in seen["query"] and "history" in seen["query"]
        assert seen["variables"]["filter"] == {
            "team": {"key": {"eq": "ENG"}},
            "updatedAt": {"gte": "2026-01-05T00:00:00+00:00", "lte": "2026-01-19T00:00:00+00:00"},
        }

    def test_the_key_rides_the_header_raw(self) -> None:
        """Personal API keys go bare; a Bearer prefix would fail as anonymous."""
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["authorization"] = request.headers.get("authorization")
            return httpx.Response(200, json=_linear_payload([]))

        with _linear_client(handler) as client:
            client.list_transitions()

        assert seen["authorization"] == "lin_api_key"

    def test_construction_without_a_key_is_misuse(self) -> None:
        with pytest.raises(TicketConfigurationError, match="TENBIN_TICKET_API_TOKEN"):
            LinearClient("https://api.linear.app")

    def test_derives_transitions_and_prefers_the_human_key(self) -> None:
        issue = _linear_issue(
            "ENG-12",
            [_linear_entry("2026-01-06T09:00:00.000Z", "In Progress", "Todo")],
        )

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_linear_payload([issue]))

        with _linear_client(handler) as client:
            page = client.list_transitions()

        assert page.total == 1
        (transition,) = page.transitions
        assert transition.ticket_id == "ENG-12"
        assert transition.from_state == "Todo"
        assert transition.to_state == "In Progress"
        assert transition.at == datetime(2026, 1, 6, 9, 0, tzinfo=UTC)

    def test_non_state_history_is_skipped_not_counted(self) -> None:
        """An assignment change is not a transition and not a failure to read one."""
        issue = _linear_issue(
            "ENG-13",
            [
                {"createdAt": "2026-01-06T09:00:00.000Z"},
                _linear_entry("2026-01-07T09:00:00.000Z", "In Review", "In Progress"),
            ],
        )

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_linear_payload([issue]))

        with _linear_client(handler) as client:
            page = client.list_transitions()

        assert [t.to_state for t in page.transitions] == ["In Review"]
        assert page.unreadable == 0

    def test_a_malformed_entry_is_unreadable_not_a_lost_page(self) -> None:
        issue = _linear_issue(
            "ENG-14", [{"createdAt": "not-a-moment", "toWorkflowState": {"name": "Done"}}]
        )

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_linear_payload([issue]))

        with _linear_client(handler) as client:
            page = client.list_transitions()

        assert page.transitions == ()
        assert page.unreadable == 1
        assert page.unreadable_examples

    def test_transitions_outside_the_window_are_narrowed_here(self) -> None:
        issue = _linear_issue(
            "ENG-15",
            [
                _linear_entry("2025-12-20T09:00:00.000Z", "In Progress"),
                _linear_entry("2026-01-06T09:00:00.000Z", "Done", "In Progress"),
            ],
        )

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_linear_payload([issue]))

        with _linear_client(handler) as client:
            page = client.list_transitions(
                since=datetime(2026, 1, 1, tzinfo=UTC),
                until=datetime(2026, 2, 1, tzinfo=UTC),
            )

        assert [t.to_state for t in page.transitions] == ["Done"]

    def test_cursor_pages_become_numbered_pages_with_an_exact_total(self) -> None:
        """The walk speaks page numbers; the cache translates, and the total is issues."""
        first = _linear_issue("ENG-21", [_linear_entry("2026-01-06T09:00:00.000Z", "Done")])
        second = _linear_issue("ENG-22", [_linear_entry("2026-01-07T09:00:00.000Z", "Done")])
        calls: list[Any] = []

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            calls.append(json.loads(request.content.decode())["variables"].get("after"))
            if len(calls) == 1:
                return httpx.Response(
                    200, json=_linear_payload([first], has_next=True, cursor="c1")
                )
            return httpx.Response(200, json=_linear_payload([second]))

        with _linear_client(handler) as client:
            first_page = client.list_transitions(page=1, limit=1)
            second_page = client.list_transitions(page=2, limit=1)

        assert calls == [None, "c1"]
        assert first_page.total == 2
        assert [t.ticket_id for t in first_page.transitions] == ["ENG-21"]
        assert [t.ticket_id for t in second_page.transitions] == ["ENG-22"]

    def test_a_group_scope_is_refused_rather_than_silently_unscoped(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:  # pragma: no cover
            raise AssertionError("no request may be sent for an unscoped read")

        with (
            _linear_client(handler) as client,
            pytest.raises(TicketConfigurationError, match="no group scoping"),
        ):
            client.list_transitions(group_id="anything")

    def test_graphql_errors_are_refusals_even_at_200(self) -> None:
        """Linear answers 200 with the failure inside the envelope."""

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"errors": [{"message": "Cannot query field 'nope'"}]})

        with (
            _linear_client(handler) as client,
            pytest.raises(StoreResponseError, match="Cannot query field"),
        ):
            client.list_transitions()

    @pytest.mark.parametrize("status", [401, 403, 500, 503])
    def test_a_refused_or_failing_linear_is_unavailable(self, status: int) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(status)

        with pytest.raises(StoreUnavailableError), _linear_client(handler) as client:
            client.list_transitions()


def _state_change(to: str, frm: str | None = "Open") -> dict[str, Any]:
    return {
        "id": "0-0",
        "timestamp": 1767225600000,
        "target": {"id": "2-1", "idReadable": "SP-1", "$type": "Issue"},
        "added": [{"name": to, "$type": "StateBundleElement"}],
        "removed": [{"name": frm, "$type": "StateBundleElement"}] if frm is not None else [],
        "$type": "CustomFieldActivityItem",
    }


def _youtrack_client(handler, **kwargs: Any) -> YouTrackClient:
    return YouTrackClient(
        "https://example.youtrack.cloud/api",
        token="perm:key",
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


class TestYouTrackClient:
    def test_reads_the_global_stream_with_state_category_and_window(self) -> None:
        """One endpoint, server-side bounds: no per-issue fetching, no query language."""
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["path"] = request.url.path
            seen.update(request.url.params)
            return httpx.Response(200, json=[])

        with _youtrack_client(handler) as client:
            client.list_transitions(
                project_id="SP",
                since=datetime(2026, 1, 1, tzinfo=UTC),
                until=datetime(2026, 2, 1, tzinfo=UTC),
            )

        assert seen["path"] == "/api/activities"
        assert seen["categories"] == "CustomFieldCategory"
        assert seen["issueQuery"] == "project: {SP}"
        assert seen["start"] == "1767225600000"
        assert seen["end"] == "1769904000000"
        assert seen["reverse"] == "false"

    def test_the_key_rides_the_header_with_a_bearer_prefix(self) -> None:
        """Permanent tokens ride Bearer, unlike Linear's bare personal keys."""
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["authorization"] = request.headers.get("authorization")
            return httpx.Response(200, json=[])

        with _youtrack_client(handler) as client:
            client.list_transitions()

        assert seen["authorization"] == "Bearer perm:key"

    def test_construction_without_a_key_is_misuse(self) -> None:
        with pytest.raises(TicketConfigurationError, match="TENBIN_TICKET_API_TOKEN"):
            YouTrackClient("https://example.youtrack.cloud/api")

    def test_derives_transitions_and_prefers_the_human_key(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=[_state_change("In Progress", "Open")])

        with _youtrack_client(handler) as client:
            page = client.list_transitions()

        assert page.total == 1
        (transition,) = page.transitions
        assert transition.ticket_id == "SP-1"
        assert transition.from_state == "Open"
        assert transition.to_state == "In Progress"
        assert transition.at == datetime(2026, 1, 1, tzinfo=UTC)

    def test_a_non_state_change_is_skipped_not_counted(self) -> None:
        """An assignee change was never a status change and is not a failure to read one."""
        assignment = dict(_state_change("In Progress"))
        assignment["added"] = [{"login": "ana", "$type": "User"}]
        assignment["removed"] = []

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=[assignment, _state_change("Done", "In Progress")])

        with _youtrack_client(handler) as client:
            page = client.list_transitions()

        assert [t.to_state for t in page.transitions] == ["Done"]
        assert page.unreadable == 0

    def test_a_cleared_state_is_unreadable_not_silent(self) -> None:
        """Understood perfectly and unrepresentable: a transition needs a destination."""
        cleared = dict(_state_change("Done"))
        cleared["added"] = []

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=[cleared])

        with _youtrack_client(handler) as client:
            page = client.list_transitions()

        assert page.transitions == ()
        assert page.unreadable == 1
        assert page.unreadable_examples

    def test_a_malformed_item_is_unreadable_not_a_lost_page(self) -> None:
        """State-typed but nameless is malformed, not absent: the change happened."""

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json=[
                    {
                        "id": "0-1",
                        "timestamp": 1767225600000,
                        "target": {"id": "2-1", "idReadable": "SP-1"},
                        "added": [{"$type": "StateBundleElement"}],
                        "removed": [],
                    }
                ],
            )

        with _youtrack_client(handler) as client:
            page = client.list_transitions()

        assert page.transitions == ()
        assert page.unreadable == 1
        assert page.unreadable_examples

    def test_items_outside_the_window_are_narrowed_here(self) -> None:
        """The stream bounds are millisecond-exact, and the half-open rule still applies."""
        early = dict(_state_change("In Progress"))
        early["timestamp"] = 1764547200000
        late = dict(_state_change("Done", "In Progress"))
        late["timestamp"] = 1769904000000

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=[early, _state_change("Done"), late])

        with _youtrack_client(handler) as client:
            page = client.list_transitions(
                since=datetime(2026, 1, 1, tzinfo=UTC),
                until=datetime(2026, 2, 1, tzinfo=UTC),
            )

        # `early` predates the window and `late` sits exactly on its end.
        assert [t.to_state for t in page.transitions] == ["Done"]

    def test_pages_are_slices_of_one_walk_with_an_exact_total(self) -> None:
        """No total on the wire, so the first page walks and the rest slice."""
        from tenbin.store.ticket_sources.youtrack import FETCH_SIZE

        calls: list[Any] = []
        full = [_state_change("In Progress") for _ in range(FETCH_SIZE)]
        rest = [_state_change("Done", "In Progress")]

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request.url.params.get("$skip", "0"))
            if len(calls) == 1:
                return httpx.Response(200, json=full)
            return httpx.Response(200, json=rest)

        with _youtrack_client(handler) as client:
            first_page = client.list_transitions(page=1, limit=1)
            last_page = client.list_transitions(page=FETCH_SIZE + 1, limit=1)

        # A full batch continues the walk; the short second fetch ended it, and
        # both pages were served from the one walk.
        assert calls == ["0", str(FETCH_SIZE)]
        assert first_page.total == FETCH_SIZE + 1
        assert [t.to_state for t in first_page.transitions] == ["In Progress"]
        assert [t.to_state for t in last_page.transitions] == ["Done"]

    def test_a_group_scope_is_refused_rather_than_silently_unscoped(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:  # pragma: no cover
            raise AssertionError("no request may be sent for an unscoped read")

        with (
            _youtrack_client(handler) as client,
            pytest.raises(TicketConfigurationError, match="no group scoping"),
        ):
            client.list_transitions(group_id="anything")

    @pytest.mark.parametrize("status", [401, 403, 500, 503])
    def test_a_refused_or_failing_youtrack_is_unavailable(self, status: int) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(status)

        with pytest.raises(StoreUnavailableError), _youtrack_client(handler) as client:
            client.list_transitions()

    def test_a_bad_query_names_the_issue_query(self) -> None:
        """A 400 is usually an unknown project; the message carries the query."""

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(400)

        with (
            _youtrack_client(handler) as client,
            pytest.raises(StoreResponseError, match=r"project: \{NOPE\}"),
        ):
            client.list_transitions(project_id="NOPE")


class TestSourceSelection:
    def test_every_supported_source_builds_without_touching_the_network(self) -> None:
        """Construction configures; only listing reads."""
        assert isinstance(
            ticket_source_client(source="go-experiment", api_url="http://127.0.0.1:8082"),
            GoExperimentClient,
        )
        assert isinstance(
            ticket_source_client(
                source="jira",
                api_url="https://example.atlassian.net",
                token="token",
                user="reader@example.com",
            ),
            JiraClient,
        )
        assert isinstance(
            ticket_source_client(source="linear", api_url="https://api.linear.app", token="key"),
            LinearClient,
        )
        assert isinstance(
            ticket_source_client(
                source="youtrack", api_url="https://example.youtrack.cloud/api", token="key"
            ),
            YouTrackClient,
        )

    def test_an_unknown_source_names_the_supported_ones(self) -> None:
        with pytest.raises(TicketConfigurationError, match="go-experiment, jira, linear, youtrack"):
            ticket_source_client(source="trac", api_url="http://127.0.0.1:8082")

    def test_the_registry_and_the_supported_names_agree(self) -> None:
        assert set(SUPPORTED_SOURCES) == {"go-experiment", "jira", "linear", "youtrack"}
        for name in SUPPORTED_SOURCES:
            assert isinstance(
                ticket_source_client(
                    source=name,
                    api_url="https://example.com",
                    token="token",
                    user="reader@example.com",
                ),
                TransitionSource,
            )

    def test_go_experiment_needs_no_credential(self) -> None:
        """Auth may be off on a loopback backend; the others always need a key."""
        assert isinstance(
            ticket_source_client(source="go-experiment", api_url="http://127.0.0.1:8082"),
            GoExperimentClient,
        )


class TestTicketSettings:
    def test_a_source_name_is_normalised_not_rejected_on_case(self) -> None:
        assert TicketSettings(ticket_source="Go-Experiment").ticket_source == "go-experiment"

    def test_an_unknown_source_is_rejected_with_the_supported_names(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="go-experiment, jira, linear, youtrack"):
            TicketSettings(ticket_source="trac")

    def test_jira_requires_an_email_and_a_token(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="TENBIN_TICKET_API_USER"):
            TicketSettings(ticket_source="jira", ticket_api_token="token")
        with pytest.raises(pydantic.ValidationError, match="TENBIN_TICKET_API_TOKEN"):
            TicketSettings(ticket_source="jira", ticket_api_user="reader@example.com")

    def test_linear_requires_a_token(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="TENBIN_TICKET_API_TOKEN"):
            TicketSettings(ticket_source="linear")
        assert (
            TicketSettings(ticket_source="linear", ticket_api_token="key").ticket_source == "linear"
        )

    def test_youtrack_requires_a_token(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="TENBIN_TICKET_API_TOKEN"):
            TicketSettings(ticket_source="youtrack")
        assert (
            TicketSettings(ticket_source="youtrack", ticket_api_token="key").ticket_source
            == "youtrack"
        )

    def test_configured_means_a_source_and_an_address(self) -> None:
        assert not TicketSettings().ticket_source_configured
        assert not TicketSettings(ticket_source="go-experiment").ticket_source_configured
        assert not TicketSettings(ticket_api_url="http://127.0.0.1:8082").ticket_source_configured
        assert TicketSettings(
            ticket_source="go-experiment", ticket_api_url="http://127.0.0.1:8082"
        ).ticket_source_configured
