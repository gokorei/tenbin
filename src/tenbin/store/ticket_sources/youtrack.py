"""YouTrack as a ticket source: status changes read from the activity stream.

YouTrack records every custom-field change as an activity item, so unlike Jira
-- whose changelogs arrive per issue -- this adapter reads one global stream:
``GET /api/activities`` with a ``CustomFieldCategory`` filter, an ``issueQuery``
for project scoping, and ``start``/``end`` millisecond bounds for the window.
That shapes the adapter in three ways, each stated here:

- The stream carries *every* custom-field change, not just State ones. A State
  change is recognised by its values' type -- ``$type: StateBundleElement`` --
  rather than by the field's name, so a project that renamed its State field
  still reads correctly. Anything else (assignees, priorities, sprints) is not
  a status change and is skipped rather than counted, because a row that was
  never a status change is not a row this program failed to read. A cleared
  State is the one exception: understood perfectly, unrepresentable as a
  transition with no destination, and therefore unreadable rather than silent.
- The list has no total, so -- like the Linear adapter -- the first page of a
  read walks the whole stream and caches the items, and later pages are slices
  of that cache. ``total`` counts the activity items walked, and the
  transitions beside it are the in-window State changes derived from the
  current page's items. A YouTrack read is therefore atomic the same way a
  Linear one is: it either arrives whole on the first request or raises there.
- The adapter never asks for the author field, so principals cannot leak even
  if the parsing is reused elsewhere. See :mod:`tenbin.store.ticket_sources`
  for why that absence is a decision rather than an omission.

There is no group scoping. YouTrack's model has projects, not groups, so a
``group_id`` is refused as misuse rather than silently unscoped.

The wire shape was checked against JetBrains' public YouTrack developer
reference on 2026-10-07 -- the activities resource with its ``categories``,
``issueQuery``, ``start``/``end``, ``$top``/``$skip`` and ``reverse``
parameters, Bearer permanent-token auth, and the ``added``/``removed`` state
bundle elements -- and not against a live instance. If JetBrains renames a
field, the rows that stop parsing surface as unreadable rather than as wrong
figures, and the tests pin the assumed shape so the drift fails here first.
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

#: How many activity items one request asks for. A client choice, not an API
#: ceiling: the stream is walked whole on the first page regardless, so this
#: only sizes the requests, and a larger number is fewer round trips against a
#: longer wait per trip.
FETCH_SIZE = 100

#: The one activity category this adapter reads. Custom fields cover State
#: along with everything else; the State changes are picked out of the stream
#: by their values' type rather than by asking for a narrower category, which
#: the API does not offer.
CATEGORY = "CustomFieldCategory"

__all__ = ["FETCH_SIZE", "YouTrackClient"]


class YouTrackClient:
    """A read-only client deriving transitions from the YouTrack activity stream.

    ``base_url`` is the REST root (``https://<domain>.youtrack.cloud/api``).
    Authentication is a permanent token on the ``Authorization`` header with a
    ``Bearer`` prefix, which is the scripting credential YouTrack documents.
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
                "youtrack needs TENBIN_TICKET_API_TOKEN (a permanent token): the "
                "activities endpoint authenticates every request, and an empty "
                "credential fails as an anonymous caller rather than as a "
                "misconfiguration."
            )
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {token.strip()}",
            },
            timeout=timeout_seconds,
            transport=transport,
            follow_redirects=False,
        )
        #: Activity items already walked for each filter, so the numbered pages
        #: the walk speaks become slices of one walk. Keyed by the filter alone --
        #: the probe and the walk share a read rather than each walking the API.
        self._reads: dict[tuple[str | None, str | None, str | None], list[dict[str, Any]]] = {}

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> YouTrackClient:
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
        """Read one page of the stream, and derive its in-window State changes."""
        if group_id is not None:
            raise TicketConfigurationError(
                "youtrack has no group scoping: --group names a go-experiment group and "
                "there is nothing to send it as. Scope with --project <SHORT-NAME> "
                "instead, or run unscoped."
            )
        if page < 1:
            raise ValueError("page is 1-based")
        bounded = max(1, limit)

        items = self._items(project_id=project_id, since=since, until=until)
        start = (page - 1) * bounded
        window = items[start : start + bounded]

        parsed: list[Transition] = []
        unreadable: list[str] = []
        for item in window:
            try:
                transition = _transition_from_item(item, since=since, until=until)
            except TransitionParseError as exc:
                unreadable.append(str(exc))
                continue
            if transition is not None:
                parsed.append(transition)

        return TransitionPage(
            transitions=tuple(parsed),
            total=len(items),
            page=page,
            limit=bounded,
            unreadable=len(unreadable),
            unreadable_examples=tuple(unreadable[:5]),
        )

    def _items(
        self,
        *,
        project_id: str | None,
        since: datetime | None,
        until: datetime | None,
    ) -> list[dict[str, Any]]:
        """Every activity item in the filter, walked once and then served from cache.

        The walk asks for pages 1..N in order and the probe asks for page 1 of
        the same filter, so one walk per filter serves both. ``reverse=false``
        keeps the stream oldest-first, because pages of a stream that can
        reorder between requests are not a partition and the walk can see one
        item twice and another never.
        """
        key = (
            project_id.strip() if project_id and project_id.strip() else None,
            _millis(since) if since is not None else None,
            _millis(until) if until is not None else None,
        )
        cached = self._reads.get(key)
        if cached is not None:
            return cached

        params: dict[str, str | int] = {
            "categories": CATEGORY,
            "fields": "id,timestamp,target(id,idReadable),added(name),removed(name)",
            "reverse": "false",
            "$top": FETCH_SIZE,
            "$skip": 0,
        }
        if project_id and project_id.strip():
            params["issueQuery"] = f"project: {{{project_id.strip()}}}"
        if since is not None:
            params["start"] = _millis(since)
        if until is not None:
            params["end"] = _millis(until)

        items: list[dict[str, Any]] = []
        while True:
            batch = self._get(params)
            items.extend(batch)
            if len(batch) < FETCH_SIZE:
                break
            params = {**params, "$skip": len(items)}

        self._reads[key] = items
        return items

    def _get(self, params: dict[str, str | int]) -> list[dict[str, Any]]:
        try:
            response = self._client.get("/activities", params=params)
        except httpx.TransportError as exc:
            raise StoreUnavailableError(f"youtrack is unreachable: {exc}") from exc

        if response.status_code >= 500:
            raise StoreUnavailableError(f"youtrack answered {response.status_code}")
        if response.status_code in {401, 403}:
            raise StoreUnavailableError(f"youtrack refused the credential ({response.status_code})")
        if response.status_code == 404:
            raise StoreResponseError("youtrack has no /activities endpoint")
        if response.status_code == 400:
            raise StoreResponseError(
                "youtrack refused the activity query (400) -- usually a project "
                "it does not know, in which case the issueQuery sent was: "
                f"{params.get('issueQuery', '(unscoped)')}"
            )
        if response.status_code != 200:
            raise StoreResponseError(f"youtrack answered {response.status_code} for /activities")

        try:
            body = response.json()
        except ValueError as exc:
            raise StoreResponseError("youtrack answered with something that is not JSON") from exc
        if not isinstance(body, list) or any(not isinstance(item, dict) for item in body):
            raise StoreResponseError("youtrack activities response is not a list of items")
        return body


def _transition_from_item(
    item: dict[str, Any],
    *,
    since: datetime | None,
    until: datetime | None,
) -> Transition | None:
    """One State change, or None for an activity that is not one.

    None and unreadable are different answers: a sprint assignment was never a
    status change, while a State change with two destinations is a row this
    program cannot reason about. The first is skipped, the second is counted.
    """
    added, saw_added = _state_names(item.get("added"))
    removed, saw_removed = _state_names(item.get("removed"))
    if not saw_added and not saw_removed:
        return None
    if len(added) > 1:
        raise TransitionParseError(
            f"youtrack activity {item.get('id', '?')} adds more than one state"
        )
    if len(removed) > 1:
        raise TransitionParseError(
            f"youtrack activity {item.get('id', '?')} removes more than one state"
        )
    if not added:
        if removed:
            # A cleared State: understood perfectly, unrepresentable as a
            # transition with no destination. Counted, like a go-experiment row
            # with a blank `to`, rather than skipped as though it never happened.
            raise TransitionParseError(
                f"youtrack activity {item.get('id', '?')} clears the State, "
                "which has no destination state"
            )
        raise TransitionParseError(
            f"youtrack activity {item.get('id', '?')} changes State to nothing nameable"
        )
    if saw_removed and not removed:
        raise TransitionParseError(
            f"youtrack activity {item.get('id', '?')} changes State from nothing nameable"
        )

    target = item.get("target")
    ticket_id = _ticket_id(target)
    at = _item_moment(item)
    if since is not None and at < _as_aware(since):
        return None
    if until is not None and at >= _as_aware(until):
        return None
    return Transition(
        ticket_id=ticket_id,
        from_state=removed[0] if removed else None,
        to_state=added[0],
        at=at,
    )


def _state_names(values: Any) -> tuple[list[str], bool]:
    """Usable State names, and whether any State-typed entry was present at all.

    The two answers differ on purpose: an assignee change has no State-typed
    entries and is not a status change, while a State-typed entry with no
    usable name is a status change this program cannot read. Collapsing them
    would count malformed rows as absent and absent rows as malformed.
    """
    if not isinstance(values, list):
        return [], False
    names: list[str] = []
    saw_state = False
    for entry in values:
        if not isinstance(entry, dict) or entry.get("$type") != "StateBundleElement":
            continue
        saw_state = True
        name = entry.get("name")
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return names, saw_state


def _ticket_id(target: Any) -> str:
    if not isinstance(target, dict):
        raise TransitionParseError("youtrack activity has no target issue")
    for key in ("idReadable", "id"):
        value = target.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise TransitionParseError("youtrack activity target has no usable issue id")


def _item_moment(item: dict[str, Any]) -> datetime:
    timestamp = item.get("timestamp")
    if not isinstance(timestamp, int) or isinstance(timestamp, bool):
        raise TransitionParseError(
            f"youtrack activity {item.get('id', '?')} has no millisecond timestamp"
        )
    return datetime.fromtimestamp(timestamp / 1000, tz=UTC)


def _millis(moment: datetime) -> str:
    return str(int(moment.astimezone(UTC).timestamp() * 1000))


def _as_aware(moment: datetime) -> datetime:
    """Treat a naive bound as UTC rather than refusing the comparison."""
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)
