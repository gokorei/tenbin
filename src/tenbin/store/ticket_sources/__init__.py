"""Ticket sources: one adapter per ticketing system, all reading the same fact.

Tenbin measures status transitions -- a ticket moved from one state to another
at a stated moment -- and several systems record that fact behind different
wires. Each adapter here knows exactly one wire and projects it onto
:class:`tenbin.corpus.transition.Transition`; the walk in
:mod:`tenbin.store.transitions` takes any of them, because it only ever calls
``list_transitions`` and only ever asks what that call returned.

**The contract an adapter honours**, beyond the method signature in
:class:`tenbin.store.transitions.TransitionSource`:

- ``total`` counts the *tickets walked*, not the transitions derived. Jira and
  Linear page over issues and derive each page's transitions from the issues on
  it, so a page's transitions and its ``total`` describe different populations
  on purpose. :func:`tenbin.report.retrospective.replace_to_period` resets both
  counts to the narrowed rows for a retrospective, which is where the header's
  "N of N" comes from; anywhere else the two counts are what the adapter said.
- Transitions are narrowed to the half-open window ``[since, until)`` by the
  adapter, not left for the caller. A server-side bound that is inclusive, a
  changelog that arrives whole, and no bound at all are three different
  upstream behaviours, and each adapter states which of them it corrects.
- Principals are dropped at this seam, deliberately. Jira changelog entries
  carry an author and Linear history entries carry an actor; neither survives
  into a :class:`~tenbin.corpus.transition.Transition`. Per-principal output
  is a different product with different consent, retention and access
  requirements rather than this report with a finer filter, so the adapter is
  where that field dies rather than somewhere a measure could quietly revive it.
- Misuse raises :class:`tenbin.store.transitions.TicketConfigurationError`,
  never a transport error. An unknown source name, a missing credential, or a
  scope the system has no equivalent for (``--group`` against Jira or Linear)
  is an operator-shaped failure, and the command layer turns it into a usage
  error rather than a failed read.
"""

from __future__ import annotations

from tenbin.config import SUPPORTED_SOURCES
from tenbin.store.ticket_sources.go_experiment import GoExperimentClient
from tenbin.store.ticket_sources.jira import JiraClient
from tenbin.store.ticket_sources.linear import LinearClient
from tenbin.store.ticket_sources.youtrack import YouTrackClient
from tenbin.store.transitions import TicketConfigurationError, TransitionSource

__all__ = [
    "SUPPORTED_SOURCES",
    "GoExperimentClient",
    "JiraClient",
    "LinearClient",
    "TicketConfigurationError",
    "TransitionSource",
    "YouTrackClient",
    "ticket_source_client",
]


def ticket_source_client(
    *,
    source: str,
    api_url: str,
    token: str = "",
    user: str = "",
    timeout_seconds: float = 10.0,
) -> TransitionSource:
    """Build the adapter a configured source names.

    Takes the settings fields rather than the settings object, so the adapter
    choice stays testable without a settings model and the settings model stays
    ignorant of adapter constructors. :class:`tenbin.config.TicketSettings`
    validates the same requirements first; the errors here are the defence for
    callers who did not come through settings.
    """
    name = source.strip().lower()
    if name == "go-experiment":
        return GoExperimentClient(api_url, token=token, timeout_seconds=timeout_seconds)
    if name == "jira":
        return JiraClient(api_url, email=user, token=token, timeout_seconds=timeout_seconds)
    if name == "linear":
        return LinearClient(api_url, token=token, timeout_seconds=timeout_seconds)
    if name == "youtrack":
        return YouTrackClient(api_url, token=token, timeout_seconds=timeout_seconds)
    raise TicketConfigurationError(
        f"unknown ticket source {source!r}: set TENBIN_TICKET_SOURCE to one of "
        f"{', '.join(SUPPORTED_SOURCES)}."
    )
