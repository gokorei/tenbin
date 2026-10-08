"""The go-experiment ticket backend: rows that already are transitions.

This adapter is the original ``TransitionClient`` with a name that says which
wire it reads. Its backend serves one row per status change under
``GET /status-transitions`` with ``page``/``limit``/``project_id``/``group_id``
parameters and ``from``/``to`` time bounds, and answers each page with the rows
plus its own ``total``/``page``/``limit`` envelope -- so this client passes the
walk's window through and reports what the backend said, with no local
narrowing beyond what :mod:`tenbin.store.transitions` already documents.

Two behaviours below were settled by probing a live backend on 2026-10-01
rather than read from its source, and the backend stays private, so they are
recorded here as observations with the date rather than as citations:

- The page ceiling is 200. The backend serves ``limit=200`` and refuses
  ``limit=250`` with a 400, so the client clamps rather than discovering.
- The ``to`` bound is inclusive. A request for ``[start, end]`` returns a
  transition recorded exactly at ``end``, which belongs to the next period
  under the half-open convention -- and is why
  :func:`tenbin.report.retrospective.replace_to_period` exists.
"""

from __future__ import annotations

from collections.abc import Mapping
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
    TransitionPage,
)

#: The page ceiling this backend enforces. Observed, not declared: a live
#: backend serves 200 and refuses 250 with a 400 rather than quietly serving a
#: shorter page, so a wrong constant here is a broken client rather than a
#: quietly truncated read.
PAGE_LIMIT = 200


def parse_transition(data: Any) -> Transition:
    """Read one transition from a backend row.

    Refuses a row missing ``ticket_id``, ``to`` or ``at``, or carrying any of
    them with the wrong type. The backend is trusted to be well-formed only as
    far as it has been observed: it writes ``from`` as omitempty and the
    timestamps in RFC3339, and a row that says otherwise is a row this program
    cannot reason about rather than one to guess at.
    """
    if not isinstance(data, dict):
        raise TransitionParseError(f"transition row is a {type(data).__name__}, not an object")

    ticket_id = _required_text(data, "ticket_id")
    to_state = _required_text(data, "to")
    at = _required_moment(data, "at")

    raw_from = data.get("from")
    if raw_from is None:
        from_state = None
    elif isinstance(raw_from, str) and raw_from.strip():
        from_state = raw_from.strip()
    else:
        raise TransitionParseError(f"transition for {ticket_id} has an unusable 'from'")

    return Transition(
        ticket_id=ticket_id,
        from_state=from_state,
        to_state=to_state,
        at=at,
    )


def _required_text(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TransitionParseError(f"transition is missing a usable {key!r}")
    return value.strip()


def _required_moment(data: dict[str, Any], key: str) -> datetime:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TransitionParseError(f"transition is missing a usable {key!r}")
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise TransitionParseError(
            f"transition carries a {key!r} that is not ISO8601: {value!r}"
        ) from exc


class GoExperimentClient:
    """A read-only client for the ticket backend's transition endpoint.

    Injectable and closeable, and the walk takes one as an argument rather
    than constructing it, for the reason every test in this package denies
    ``socket.connect``: a test that reaches a real backend proves nothing about
    the parsing and everything about whether a backend happened to be running.
    """

    def __init__(
        self,
        base_url: str,
        *,
        token: str = "",
        timeout_seconds: float = 10.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        headers = {"Accept": "application/json"}
        if token.strip():
            headers["Authorization"] = f"Bearer {token.strip()}"
        self._client = httpx.Client(
            base_url=self._base_url,
            headers=headers,
            timeout=timeout_seconds,
            transport=transport,
            follow_redirects=False,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> GoExperimentClient:
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
        """Read one page of transitions.

        The time filter is passed to the backend rather than applied here, and
        that matters: filtering client-side over a paginated list would mean the
        ``total`` the backend reports describes a population the caller is not
        looking at, and a denominator that disagrees with the rows beside it is
        the failure this program exists to make visible.
        """
        if page < 1:
            raise ValueError("page is 1-based")
        bounded = max(1, min(limit, PAGE_LIMIT))

        params: dict[str, str | int] = {"page": page, "limit": bounded}
        if project_id:
            params["project_id"] = project_id
        if group_id:
            params["group_id"] = group_id
        if since is not None:
            params["from"] = _rfc3339(since)
        if until is not None:
            params["to"] = _rfc3339(until)

        try:
            response = self._client.get("/status-transitions", params=params)
        except httpx.TransportError as exc:
            raise StoreUnavailableError(f"ticket backend is unreachable: {exc}") from exc

        if response.status_code >= 500:
            raise StoreUnavailableError(f"ticket backend answered {response.status_code}")
        if response.status_code in {401, 403}:
            raise StoreUnavailableError(
                f"ticket backend refused the credential ({response.status_code})"
            )
        if response.status_code == 404:
            raise StoreResponseError("ticket backend has no /status-transitions endpoint")
        if response.status_code == 400:
            # The endpoint validates `page` and `limit` strictly and answers a bad
            # one with a 400 rather than a default. A client that guessed the
            # ceiling wrong lands here, so the message names the parameters
            # rather than leaving the reader to wonder which one was refused.
            raise StoreResponseError(
                "ticket backend refused the request for /status-transitions; it "
                f"accepts limit 1..{PAGE_LIMIT} and a positive page"
            )
        if response.status_code != 200:
            raise StoreResponseError(
                f"ticket backend answered {response.status_code} for /status-transitions"
            )

        return _page_from_json(_json_object(response, "status-transitions"))


def _json_object(response: httpx.Response, operation: str) -> Mapping[str, Any]:
    try:
        data = response.json()
    except ValueError as exc:
        raise StoreResponseError(f"{operation} answered with something that is not JSON") from exc
    if not isinstance(data, dict):
        raise StoreResponseError(f"{operation} answered with a JSON {type(data).__name__}")
    return data


def _page_from_json(data: Mapping[str, Any]) -> TransitionPage:
    """Read one page, counting rows it could not parse rather than dropping them."""
    results = data.get("results")
    if not isinstance(results, list):
        raise StoreResponseError("status-transitions response has no 'results' list")

    total = _required_int(data, "total")
    page = _required_int(data, "page")
    limit = _required_int(data, "limit")

    parsed: list[Transition] = []
    unreadable: list[str] = []
    for row in results:
        try:
            parsed.append(parse_transition(row))
        except TransitionParseError as exc:
            unreadable.append(str(exc))

    return TransitionPage(
        transitions=tuple(parsed),
        total=total,
        page=page,
        limit=limit,
        unreadable=len(unreadable),
        unreadable_examples=tuple(unreadable[:5]),
    )


def _required_int(data: Mapping[str, Any], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise StoreResponseError(f"status-transitions response has no usable {key!r}")
    return value


def _rfc3339(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat()


__all__ = ["GoExperimentClient", "PAGE_LIMIT", "parse_transition"]
