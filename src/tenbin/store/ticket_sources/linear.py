"""Linear as a ticket source: status changes derived from issue history.

Like Jira, Linear has no transitions endpoint: this adapter pages over issues
with a GraphQL search and derives each page's transitions from their history
entries. The Jira module's consequences all transfer -- ``total`` counts
issues, transitions are narrowed locally to the half-open ``[since, until)``,
principals are dropped at the seam, and there is no group scoping -- with two
Linear-specific additions stated here rather than discovered:

- Pagination is cursor-based (Relay ``first``/``after``), but the walk speaks
  page numbers. So the first page of a read walks every cursor internally and
  caches the issues; later pages are slices of that cache. A Linear read is
  therefore atomic: it either arrives whole on the first request or raises
  there, and there are no partial Linear snapshots. The cache key is the
  filter, not the page size, so the existence probe (one row asked) and the
  walk share it rather than walking twice.
- The credential is a personal API key sent raw as the ``Authorization``
  header, which is the scripting credential Linear documents. OAuth access
  tokens ride the same header with a ``Bearer`` prefix, which this client does
  not add -- an OAuth token handed here fails with an authentication error
  that names the header rather than a wrong key.

What was checked against Linear's public developers' reference on 2026-10-07:
the endpoint (``POST https://api.linear.app/graphql``), personal-key auth, the
``gte``/``lte`` date comparators on ``updatedAt``, team scoping through the
``team: { key: { eq } }`` relationship filter, cursor pagination with
``pageInfo { hasNextPage endCursor }``, and ``identifier`` as the human issue
key. What was **not** checked against anything live is the history node shape
-- ``history(first:) { nodes { createdAt fromWorkflowState { name }
toWorkflowState { name } } }`` is read from the schema reference, and a node
that does not match it is an unreadable row rather than a wrong figure. The
tests pin the assumed shape so the drift fails here first.
"""

from __future__ import annotations

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

#: How many issues one GraphQL request asks for. A client choice, not an API
#: ceiling: Linear defaults to 50 and accepts larger, but a read that fetches
#: histories with every issue is already the expensive shape, so this client
#: does not ask for more.
FETCH_SIZE = 50

#: How many history entries one issue's read asks for. A bound, not a claim
#: about any team's activity: an issue with more state changes than this is
#: read partially, and the partiality is silent -- which is why the number is
#: large enough that hitting it means the workflow, not the client, is the
#: thing to look at.
HISTORY_LIMIT = 250

_ISSUES_QUERY = (
    """
query TenbinTransitions($first: Int!, $after: String, $filter: IssueFilter) {
  issues(first: $first, after: $after, filter: $filter, orderBy: updatedAt) {
    nodes {
      id
      identifier
      history(first: """
    + str(HISTORY_LIMIT)
    + """) {
        nodes {
          createdAt
          fromWorkflowState { name }
          toWorkflowState { name }
        }
      }
    }
    pageInfo { hasNextPage endCursor }
  }
}
"""
)

__all__ = ["LinearClient", "FETCH_SIZE"]


class LinearClient:
    """A read-only client deriving transitions from Linear issue history.

    ``base_url`` is the GraphQL endpoint (``https://api.linear.app/graphql``).
    """

    def __init__(
        self,
        base_url: str,
        *,
        token: str = "",
        timeout_seconds: float = 10.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not token.strip():
            raise TicketConfigurationError(
                "linear needs TENBIN_TICKET_API_TOKEN (a personal API key): the "
                "GraphQL endpoint authenticates every request, and an empty "
                "credential fails as an anonymous caller rather than as a "
                "misconfiguration."
            )
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": token.strip(),
            },
            timeout=timeout_seconds,
            transport=transport,
            follow_redirects=False,
        )
        #: Issues already walked for each filter, so cursor pages become the
        #: numbered pages the walk speaks. Keyed by the filter alone -- the
        #: probe and the walk share a read rather than each walking the API.
        self._reads: dict[tuple[str | None, str | None, str | None], list[dict[str, Any]]] = {}

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> LinearClient:
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
                "linear has no group scoping: --group names a go-experiment group and "
                "there is nothing to send it as. Scope with --project <TEAM-KEY> "
                "instead, or run unscoped."
            )
        if page < 1:
            raise ValueError("page is 1-based")
        bounded = max(1, limit)

        issues = self._issues(project_id=project_id, since=since, until=until)
        start = (page - 1) * bounded
        window = issues[start : start + bounded]

        parsed: list[Transition] = []
        unreadable: list[str] = []
        for issue in window:
            try:
                parsed.extend(_transitions_from_issue(issue, since=since, until=until))
            except TransitionParseError as exc:
                unreadable.append(str(exc))

        return TransitionPage(
            transitions=tuple(parsed),
            total=len(issues),
            page=page,
            limit=bounded,
            unreadable=len(unreadable),
            unreadable_examples=tuple(unreadable[:5]),
        )

    def _issues(
        self,
        *,
        project_id: str | None,
        since: datetime | None,
        until: datetime | None,
    ) -> list[dict[str, Any]]:
        """Every issue in the filter, walked once and then served from cache.

        The walk asks for pages 1..N in order and the probe asks for page 1 of
        the same filter, so one walk per filter serves both. A second filter
        starts a second walk; interleaved reads of two filters alternate walks
        rather than sharing them, which is the correct behaviour stated
        wastefully rather than a cache a reader has to reason about.
        """
        key = (
            project_id.strip() if project_id and project_id.strip() else None,
            since.astimezone(UTC).isoformat() if since is not None else None,
            until.astimezone(UTC).isoformat() if until is not None else None,
        )
        cached = self._reads.get(key)
        if cached is not None:
            return cached

        variables: dict[str, Any] = {
            "first": FETCH_SIZE,
            "after": None,
            "filter": _issue_filter(project_id=project_id, since=since, until=until),
        }
        issues: list[dict[str, Any]] = []
        while True:
            payload = self._post(variables)
            nodes, page_info = _issues_page(payload)
            issues.extend(nodes)
            if not page_info.get("hasNextPage"):
                break
            cursor = page_info.get("endCursor")
            if not isinstance(cursor, str) or not cursor:
                break
            variables = {**variables, "after": cursor}

        self._reads[key] = issues
        return issues

    def _post(self, variables: dict[str, Any]) -> Any:
        try:
            response = self._client.post(
                "/graphql", json={"query": _ISSUES_QUERY, "variables": variables}
            )
        except httpx.TransportError as exc:
            raise StoreUnavailableError(f"linear is unreachable: {exc}") from exc

        if response.status_code >= 500:
            raise StoreUnavailableError(f"linear answered {response.status_code}")
        if response.status_code in {401, 403}:
            raise StoreUnavailableError(
                f"linear refused the credential ({response.status_code}): the personal "
                "API key rides the Authorization header raw, without a Bearer prefix."
            )
        if response.status_code == 404:
            raise StoreResponseError("linear has no /graphql endpoint")
        if response.status_code != 200:
            raise StoreResponseError(f"linear answered {response.status_code} for /graphql")

        try:
            body = response.json()
        except ValueError as exc:
            raise StoreResponseError("linear answered with something that is not JSON") from exc
        if not isinstance(body, dict):
            raise StoreResponseError(f"linear answered with a JSON {type(body).__name__}")

        # GraphQL answers 200 with the failure inside the envelope, so a body
        # without data is a refusal even at 200. The first three messages are
        # what an operator can act on; the rest is the same defect repeated.
        errors = body.get("errors")
        if errors:
            messages = [
                str(entry.get("message", entry)) if isinstance(entry, dict) else str(entry)
                for entry in (errors if isinstance(errors, list) else [errors])
            ]
            raise StoreResponseError(f"linear refused the query: {'; '.join(messages[:3])}")
        data = body.get("data")
        if not isinstance(data, dict) or not isinstance(data.get("issues"), dict):
            raise StoreResponseError("linear answered without an issues payload")
        return data["issues"]


def _issue_filter(
    *,
    project_id: str | None,
    since: datetime | None,
    until: datetime | None,
) -> dict[str, Any]:
    """Scope the read server-side, so the walk never pages an instance to filter it here."""
    scope: dict[str, Any] = {}
    if project_id and project_id.strip():
        scope["team"] = {"key": {"eq": project_id.strip()}}
    window: dict[str, str] = {}
    if since is not None:
        window["gte"] = since.astimezone(UTC).isoformat()
    if until is not None:
        window["lte"] = until.astimezone(UTC).isoformat()
    if window:
        scope["updatedAt"] = window
    return scope


def _issues_page(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    nodes = payload.get("nodes")
    if not isinstance(nodes, list) or any(not isinstance(node, dict) for node in nodes):
        raise StoreResponseError("linear issues payload has no usable nodes")
    page_info = payload.get("pageInfo")
    if not isinstance(page_info, dict):
        raise StoreResponseError("linear issues payload has no pageInfo")
    return nodes, page_info


def _transitions_from_issue(
    issue: dict[str, Any],
    *,
    since: datetime | None,
    until: datetime | None,
) -> list[Transition]:
    """Every in-window status change one issue's history records.

    A history entry counts as a status change exactly when it names a
    destination workflow state. Entries about anything else -- assignment,
    titles, estimates -- are not transitions and are skipped rather than
    counted, because a row that was never a status change is not a row this
    program failed to read.
    """
    raw_id = issue.get("identifier") or issue.get("id")
    if not isinstance(raw_id, str) or not raw_id.strip():
        raise TransitionParseError("linear issue is missing a usable identifier")
    ticket_id = raw_id.strip()

    history = issue.get("history")
    entries = history.get("nodes") if isinstance(history, dict) else None
    if not isinstance(entries, list):
        raise TransitionParseError(f"linear issue {ticket_id} has no history nodes")

    found: list[Transition] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise TransitionParseError(
                f"linear issue {ticket_id} has a history entry that is not an object"
            )
        to_state = entry.get("toWorkflowState")
        to_name = to_state.get("name") if isinstance(to_state, dict) else None
        if not isinstance(to_name, str) or not to_name.strip():
            continue
        at = _history_moment(ticket_id, entry)
        from_state = entry.get("fromWorkflowState")
        from_name = from_state.get("name") if isinstance(from_state, dict) else None
        if since is not None and at < _as_aware(since):
            continue
        if until is not None and at >= _as_aware(until):
            continue
        found.append(
            Transition(
                ticket_id=ticket_id,
                from_state=from_name.strip()
                if isinstance(from_name, str) and from_name.strip()
                else None,
                to_state=to_name.strip(),
                at=at,
            )
        )
    return found


def _history_moment(ticket_id: str, entry: dict[str, Any]) -> datetime:
    created = entry.get("createdAt")
    if not isinstance(created, str) or not created.strip():
        raise TransitionParseError(f"linear issue {ticket_id} has a history entry with no moment")
    try:
        return _as_aware(datetime.fromisoformat(created.strip()))
    except ValueError as exc:
        raise TransitionParseError(
            f"linear issue {ticket_id} carries a moment that is not ISO8601: {created!r}"
        ) from exc


def _as_aware(moment: datetime) -> datetime:
    """Treat a naive moment as UTC rather than refusing the comparison."""
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)
