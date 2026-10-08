"""What would make a comparison interpretable, and the fact that nothing here is.

An observational corpus cannot support a counterfactual. A principal that happened
to write a change was not assigned to write it, and the two populations differ
*because* of that: agents receive the well-specified tickets and humans receive
the rest, so the selection is the entire mechanism by which the groups differ. A
difference in outcome between them is therefore uninterpretable -- not
small-sample, not confounded, not waiting for more data, but unanswerable, because
the question "what would this change have looked like without an agent" has no
instance in the corpus and no amount of reading would produce one.

That argument lives here, at the point of the emptiness, rather than only in
``docs/decisions/002-no-causal-claims.md``, because the next person to ask "why is
this empty" is not going to read a decision record before they fill it in. So the
empty value is a named constant carrying its own reason, and the reason is
available to anything holding a registry: asking is cheap and the wrong answer is
expensive.

**What this module notably does not do:** it does not attempt to synthesise a
mechanism. There is a natural experiment available -- ticket well-specification is
itself observable, so conditioning on it is a real technique with real literature
-- and it is refused on purpose. Conditioning on an observed confounder yields a
conditional association, and the thing being measured is the *unconditional*
difference a reader actually wants, so the technique answers a different question
while looking like it answers the right one. Keeping the absence a ``None``
rather than a placeholder object that claims to be a mechanism is the same
decision in one line: there is no value here that is safe to fill in.

Granularity is refused from the same posture. ``individual`` sits below the floor,
and the floor exists because a per-principal effectiveness ranking is a different
product -- different consent, retention and access requirements -- and not this
report with a different filter. That it is also methodologically weak is the
weaker of the two reasons, and the one that would erode first.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

#: The reason the corpus has none, attached to the emptiness so the next reader
#: gets the argument rather than a blank. Exported rather than inlined because a
#: refusal that paraphrases it is a second argument to keep true.
MECHANISM_ABSENT_REASON: str = (
    "The corpus is observational and the principal that happened to write a change "
    "was not assigned to write it, so it holds no counterfactual. Agents receive the "
    "well-specified tickets and humans receive the rest, and that selection is the "
    "entire mechanism by which the two groups differ -- a difference in outcome "
    "between them is uninterpretable without knowing what each group was given. No "
    "change to the store produces an assignment mechanism, so this is not a missing "
    "ticket. What is absent is an allocation between groups, not comparison itself: "
    "one population read twice is not two populations, so a measure against its own "
    "past value needs no assignment mechanism because no principal was assigned to "
    "anything."
)

#: What a reader would have to be holding for a comparative claim to be permitted.
#: Named as a constant because it is the answer to "what is missing", and an answer
#: to that question should not be assembled at each call site.
ASSIGNMENT_MECHANISM_MISSING_FACT: str = (
    "an assignment mechanism: a stated way work was allocated between principals, "
    "and the instrument that made the allocation observable to somebody afterwards"
)

#: The absent mechanism. Its value is ``None`` rather than a sentinel object so
#: that a claim rendering without one cannot accidentally receive a plausible
#: looking substitute.
NO_ASSIGNMENT_MECHANISM: None = None


def require_text(owner: str, field: str, value: str | None) -> None:
    """Reject a missing or blank required field, naming the owner and the field.

    Lives in this module because this module is the base of the package's import
    graph, so it is the one place a validator can live without a leaf module
    having to know about a type defined above it. Three dataclasses in the package
    need this and they must produce the same message format, or a reader triaging
    errors has to learn three of them.

    The parameter is typed ``str | None`` even where the field is declared ``str``,
    because a caller that supplies ``None`` for a required field is a mistake this
    has to answer with a ``ValueError`` naming the field. Left to the annotation it
    would be an ``AttributeError`` from inside the check, which reads as a defect
    in the validator rather than as the caller's missing caveat, and the field
    that was left out goes unmentioned.

    A blank string is the other case and it is the one most checks miss. ``""``
    satisfies a required argument in the sense that it was passed, and it is a
    claim that says nothing -- an empty statement, an empty falsifier, an empty
    denominator description. Each of those renders as a figure with a caveat
    attached that is blank, which is the failure the whole package exists to
    prevent, and which looks correct in review.
    """
    if value is None:
        raise ValueError(
            f"{owner}.{field} is required; a missing caveat is a claim with no caveat, "
            "so there is no default to fall back to."
        )
    if not value.strip():
        raise ValueError(
            f"{owner}.{field} must not be blank; a blank caveat is a claim with no caveat."
        )


class Granularity(StrEnum):
    """The population a figure is computed over, from narrowest to widest.

    ``StrEnum`` rather than ``Enum`` so a granularity survives a round trip
    through a report or a JSON fixture as the word a reader sees, and so two
    fixtures cannot disagree about what ``"team"`` means.

    ``individual`` is refused below the floor regardless of what the floor is set
    to, which is why it is a member here and not merely a rejected argument.
    """

    individual = "individual"
    team = "team"
    area = "area"
    repository = "repository"
    corpus = "corpus"


#: Narrowest first, because "narrower than the floor" is the rule the gate applies
#: and a rule expressed as an ordering needs the ordering to be written down. An
#: enum is unordered by construction, so this is the only place it is defined.
GRANULARITY_ORDER: tuple[Granularity, ...] = (
    Granularity.individual,
    Granularity.team,
    Granularity.area,
    Granularity.repository,
    Granularity.corpus,
)

#: The default floor. ``team`` rather than ``area`` because a team is the smallest
#: population in which a rate is not a ranking of people, and ``area`` alone would
#: have been a choice nobody could later distinguish from an accident.
MINIMUM_GRANULARITY: Granularity = Granularity.team


def is_narrower_than(candidate: Granularity, floor: Granularity) -> bool:
    """Whether ``candidate`` resolves to a smaller population than ``floor``.

    Raises rather than returning ``False`` for a value outside
    :data:`GRANULARITY_ORDER`, because a granularity the order does not know has no
    position in it and a silent ``False`` would read as "at the floor" -- which is
    a permission.
    """
    rank = GRANULARITY_ORDER.index(candidate)
    floor_rank = GRANULARITY_ORDER.index(floor)
    return rank < floor_rank


@dataclass(frozen=True)
class AssignmentMechanism:
    """A defensible reason the two populations differ, and how it would be known.

    All three fields are required, and the third is the one that is usually
    missing in a hand-written mechanism. ``method`` and ``description`` are the
    argument; ``instrument`` is what a later reader would use to check it, and a
    mechanism with no instrument is a story rather than a mechanism. A comparative
    claim that names this renders with it attached so the reader can judge the
    mechanism instead of trusting the report's use of it.
    """

    method: str
    description: str
    instrument: str

    def __post_init__(self) -> None:
        require_text("AssignmentMechanism", "method", self.method)
        require_text("AssignmentMechanism", "description", self.description)
        require_text("AssignmentMechanism", "instrument", self.instrument)

    def render_text(self) -> str:
        """Render the mechanism as one line a reader can judge.

        The instrument is last because it is the part that decides whether the
        claim above it is worth anything, and putting it last makes it the
        easiest thing in the line to skip -- which is the correct thing for a
        reader in a hurry to skip.
        """
        return f"{self.method} — {self.description} (instrument: {self.instrument})"
