"""One ticket status transition, as a fact with a moment on it.

This is a ticket source's counterpart to
:class:`tenbin.corpus.record.Record`, and it exists as its own type rather
than as fields bolted onto a Kojutsu document for one deliberate reason: a
status transition is not a review comment, and a record type that could hold
both would have to pick one vocabulary of kinds. ``RecordKind`` is a closed
five-value set written for pull-request review activity -- answer, review
verdict, inline comment, PR lifecycle, check run -- and a ticket moving from
``ready_for_review`` to ``done`` fits none of them. Forcing it in would have
meant either a sixth kind whose meaning is "not really any of the others", or
overloading ``pr_lifecycle``, which would then have been two unrelated things
sharing a name.

The second reason is honesty about what a transition *is*. A transition records
that a ticket's status field changed and when. It does not record who changed
it, what they were doing, or whether the change was good. Every adapter in
:mod:`tenbin.store.ticket_sources` projects its wire onto these four fields,
dropping whatever principal the wire carried, and a fact type that holds
exactly this cannot be accused of implying a person.

**The moment is a fact and its precision is not.** ``at`` is whatever the
source recorded. That is not necessarily the moment a person acted: it is the
moment the source observed the change, which may be after somebody made it,
and it carries no zone normalisation beyond what the source sent. Measures over
intervals derived from it inherit that, and the ones that exist say so.

Row parsing lives with the adapters rather than here, because each wire has
its own row shape and its own ways of being malformed. What this module owns
is the shape every adapter converges on and the error a row that cannot become
one raises.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

__all__ = ["Transition", "TransitionParseError"]


class TransitionParseError(ValueError):
    """A row the source served does not describe a transition.

    A ``ValueError`` because it is a malformed input rather than a transport or
    configuration failure -- the shape of the data is wrong, not the state of
    the world. The walk treats it as a hole rather than skipping the row, so a
    snapshot can say how many it could not read instead of quietly counting fewer.
    """


@dataclass(frozen=True)
class Transition:
    """A ticket's status changing from one state to another, at a stated moment.

    ``from_state`` is optional because a transition with no recorded origin is
    a real row in real data -- the first transition of a ticket's life. An
    absent origin is therefore ``None`` rather than a sentinel string, because
    "there was no from" and "the from was the empty string" are different claims
    and only one of them is true.

    ``to_state`` is required and non-blank. A row naming no destination state
    describes nothing, so it is refused at the parse boundary rather than
    becoming a transition to nowhere.
    """

    ticket_id: str
    from_state: str | None
    to_state: str
    at: datetime
