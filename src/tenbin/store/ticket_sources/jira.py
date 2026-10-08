"""Jira Cloud as a ticket source: status changes derived from issue changelogs.

Jira has no transitions endpoint. What it has is per-issue changelogs, so this
adapter pages over *issues* with a JQL search and derives each page's
transitions from the status entries in their changelogs. That shapes everything
below, and each consequence is stated where it lands rather than left for a
reader to triangulate:

- ``total`` counts issues, because ``total`` is what the search reports and the
  walk pages by it. The transitions beside it are the in-window status changes
  derived from the current page's issues -- a different population on purpose,
  and the reason the adapter contract in :mod:`tenbin.store.ticket_sources`
  names the two counts separately.
- The search is scoped server-side with ``project = KEY`` and ``updated``
  bounds, but a changelog arrives whole: an issue updated inside the window
  carries entries from before it. So the derived transitions are narrowed to
  the half-open ``[since, until)`` here, and the JQL bounds are only what keeps
  the read from walking the entire instance.
- JQL dates have minute precision (``updated >= "2026-01-05 00:00"``), so the
  server-side bound is coarser than the window. The local narrowing is what
  makes the boundary exact, not the JQL.
- Changelog authors are dropped at this seam along with every other principal
  field. See :mod:`tenbin.store.ticket_sources` for why that is a decision
  rather than an omission.
- There is no group scoping. Jira's model has projects, not groups, so a
  ``group_id`` is refused as misuse rather than silently unscoped.

The wire shape was checked against Atlassian's public REST v3 reference on
2026-10-07 -- ``GET /rest/api/3/search/jql`` (the enhanced search; the old
``/search`` is being removed), ``expand=changelog``, Basic auth with an email
address and an API token -- and not against a live instance. If Atlassian
renames a field, the rows that stop parsing surface as unreadable rather than
as wrong figures, and the tests pin the assumed shape so the drift fails here
first.
"""

from __future__ import annotations

import base64
from datetime import UTC, datetime
from typing import Any

import httpx

from tenbin.corpus.transition import Transition, TransitionParseError
from tenbin.store.client import (
    StoreResponseError,
    StoreUnavailableError,
)
from tenbin.store.transitions import (
    DEFAULT_PAGE_LIMIT,
    TicketConfigurationError,
    TransitionPage,
)

#: The most rows one search request may ask for. Jira's documented ceiling, so
#: the clamp below enforces a contract rather than guessing one.
PAGE_LIMIT = 100

#: How the one JQL this client sends is ordered. Deterministic output order is
#: not required -- the walk re-sorts per ticket -- but a search that can return
#: rows in any order between pages is a walk that can miss rows, so the order
#: is stated rather than left to the server.
_SEARCH_ORDER = "order by updated"

__all__ = ["JiraClient", "PAGE_LIMIT"]


class JiraClient:
    """A read-only client deriving transitions from Jira Cloud changelogs.

    ``base_url`` is the site address (``https://<site>.atlassian.net``).
    Authentication is Basic with the account email and an API token, which is
    the credential Jira Cloud documents for scripts; OAuth bearer tokens are a
    different flow and are not accepted here, because a client that sent one as
    Basic would fail with a 401 that reads as a wrong token.
    """

    def __init__(
        self,
        base_url: str,
        *,
        email: str = "",
        token: str = "",
        timeout_seconds: float = 10.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not email.strip() or not token.strip():
            raise TicketConfigurationError(
                "jira needs both TENBIN_TICKET_API_USER (the account email) and "
                "TENBIN_TICKET_API_TOKEN (an API token): Jira Cloud authenticates "
                "scripts with Basic auth over the pair, and either half alone is "
                "a 401 that reads as a wrong credential."
            )
        credentials = base64.b64encode(f"{email.strip()}:{token.strip()}".encode()).decode()
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Accept": "application/json", "Authorization": f"Basic {credentials}"},
            timeout=timeout_seconds,
            transport=transport,
            follow_redirects=False,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> JiraClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def list_transitions(
        self,
        *,
        project_id: str | None = None,
        group_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        page: int = 1,
        limit: int = DEFAULT_PAGE_LIMIT,
    ) -> TransitionPage:
        """Read one page of issues, and derive its in-window status changes."""
        if group_id is not None:
            raise TicketConfigurationError(
                "jira has no group scoping: --group names a go-experiment group and "
                "there is nothing to send it as. Scope with --project <KEY> instead, "
                "or run unscoped."
            )
        if page < 1:
            raise ValueError("page is 1-based")
        bounded = max(1, min(limit, PAGE_LIMIT))

        query = _jql(project_id=project_id, since=since, until=until)
        try:
            response = self._client.get(
                "/rest/api/3/search/jql",
                params={
                    "jql": query,
                    "expand": "changelog",
                    "fields": "key",
                    "maxResults": bounded,
                    "startAt": (page - 1) * bounded,
                },
            )
        except httpx.TransportError as exc:
            raise StoreUnavailableError(f"jira is unreachable: {exc}") from exc

        if response.status_code >= 500:
            raise StoreUnavailableError(f"jira answered {response.status_code}")
        if response.status_code in {401, 403}:
            raise StoreUnavailableError(f"jira refused the credential ({response.status_code})")
        if response.status_code == 404:
            raise StoreResponseError("jira has no /rest/api/3/search/jql endpoint")
        if response.status_code == 400:
            raise StoreResponseError(
                "jira refused the search (400) -- usually a project key it does not "
                f"know, in which case the JQL sent was: {query}"
            )
        if response.status_code != 200:
            raise StoreResponseError(f"jira answered {response.status_code} for the search")

        data = _json_object(response)
        issues = data.get("issues")
        if not isinstance(issues, list):
            raise StoreResponseError("jira search response has no 'issues' list")
        total = _required_int(data, "total")

        parsed: list[Transition] = []
        unreadable: list[str] = []
        for issue in issues:
            try:
                parsed.extend(_transitions_from_issue(issue, since=since, until=until))
            except TransitionParseError as exc:
                unreadable.append(str(exc))

        return TransitionPage(
            transitions=tuple(parsed),
            total=total,
            page=page,
            limit=bounded,
            unreadable=len(unreadable),
            unreadable_examples=tuple(unreadable[:5]),
        )


def _jql(
    *,
    project_id: str | None,
    since: datetime | None,
    until: datetime | None,
) -> str:
    """The one search this client sends, with the window as updated-bounds.

    ``updated`` bounds keep the read from walking the whole instance; they do
    not define the population, because a changelog arrives whole and the
    entries are narrowed locally afterwards. The ``order by`` is load-bearing:
    without a stated order, pages of a search are not a partition and the walk
    can see one issue twice and another never.
    """
    clauses: list[str] = []
    if project_id and project_id.strip():
        clauses.append(f"project = {project_id.strip()}")
    if since is not None:
        clauses.append(f'updated >= "{_jql_moment(since)}"')
    if until is not None:
        clauses.append(f'updated <= "{_jql_moment(until)}"')
    if clauses:
        return " AND ".join(clauses) + f" {_SEARCH_ORDER}"
    return _SEARCH_ORDER


def _jql_moment(moment: datetime) -> str:
    """JQL's datetime spelling: minute precision, no zone designator."""
    return moment.astimezone(UTC).strftime("%Y-%m-%d %H:%M")


def _transitions_from_issue(
    issue: Any,
    *,
    since: datetime | None,
    until: datetime | None,
) -> list[Transition]:
    """Every in-window status change one issue's changelog records.

    One issue's failures refuse the issue rather than its neighbours: a row
    the program cannot reason about is counted, not spread.
    """
    if not isinstance(issue, dict):
        raise TransitionParseError(f"jira issue is a {type(issue).__name__}, not an object")
    key = issue.get("key")
    if not isinstance(key, str) or not key.strip():
        raise TransitionParseError("jira issue is missing a usable 'key'")
    ticket_id = key.strip()

    changelog = issue.get("changelog")
    histories = changelog.get("histories") if isinstance(changelog, dict) else None
    if not isinstance(histories, list):
        raise TransitionParseError(f"jira issue {ticket_id} has no changelog histories")

    found: list[Transition] = []
    for history in histories:
        if not isinstance(history, dict):
            raise TransitionParseError(
                f"jira issue {ticket_id} has a history entry that is not an object"
            )
        at = _history_moment(ticket_id, history)
        items = history.get("items")
        if not isinstance(items, list):
            raise TransitionParseError(f"jira issue {ticket_id} has a history entry with no items")
        for item in items:
            if not isinstance(item, dict) or item.get("field") != "status":
                continue
            to_state = item.get("toString")
            if not isinstance(to_state, str) or not to_state.strip():
                raise TransitionParseError(
                    f"jira issue {ticket_id} has a status change with no destination"
                )
            raw_from = item.get("fromString")
            from_state = (
                raw_from.strip() if isinstance(raw_from, str) and raw_from.strip() else None
            )
            if since is not None and at < _as_aware(since):
                continue
            if until is not None and at >= _as_aware(until):
                continue
            found.append(
                Transition(
                    ticket_id=ticket_id, from_state=from_state, to_state=to_state.strip(), at=at
                )
            )
    return found


def _history_moment(ticket_id: str, history: dict[str, Any]) -> datetime:
    created = history.get("created")
    if not isinstance(created, str) or not created.strip():
        raise TransitionParseError(f"jira issue {ticket_id} has a history entry with no moment")
    try:
        return _as_aware(datetime.fromisoformat(created.strip()))
    except ValueError as exc:
        raise TransitionParseError(
            f"jira issue {ticket_id} carries a moment that is not ISO8601: {created!r}"
        ) from exc


def _as_aware(moment: datetime) -> datetime:
    """Treat a naive moment as UTC rather than refusing the comparison.

    Jira documents offset-aware timestamps; a naive one is a server that did
    not follow the contract, and reading it as UTC is stated here rather than
    decided per call site.
    """
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def _json_object(response: httpx.Response) -> dict[str, Any]:
    try:
        data = response.json()
    except ValueError as exc:
        raise StoreResponseError("jira answered with something that is not JSON") from exc
    if not isinstance(data, dict):
        raise StoreResponseError(f"jira answered with a JSON {type(data).__name__}")
    return data


def _required_int(data: dict[str, Any], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise StoreResponseError(f"jira search response has no usable {key!r}")
    return value
