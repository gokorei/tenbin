"""The gate: one place that decides whether a claim may be rendered, and says why.

Everything else in this package describes; this module refuses. It exists as a
separate object with a ``check`` and a ``require`` because the two callers need
opposite things. A report needs to *list* every refusal it hits, so checking must
return rather than raise -- a report that stopped at the first refusal would
render one of seven and say nothing about the other six, which reads as though
six were permitted. A caller about to render one claim needs the opposite: a
guard that returns quietly is a guard that can be forgotten, so
:meth:`ClaimGate.require` raises and a caller who skips the check fails on the
first claim they try to print.

The rules are three, and the third exists because the first two are
configurable. A comparative claim with no mechanism is refused. A granularity
narrower than the floor is refused. And ``individual`` is refused regardless of
the floor, so lowering the floor to admit it is not a way past the second rule
-- it is a deliberate act that changes nothing. That is the whole argument for
per-principal effectiveness stated as code: it is not this report with a
different filter, it is a different product with different consent, retention and
access requirements, and no floor a caller can set in one argument should be
able to authorise it.

**What the first rule refuses is a comparison between principals, and saying so
is load-bearing.** Decision 002 refuses "agentic development is more effective
than human development" because the corpus holds no assignment mechanism and the
selection of well-specified tickets onto the agent path is the entire mechanism by
which the two groups differ. That argument is about *two populations*, and it does
not reach a claim of kind :attr:`~tenbin.claims.model.ClaimKind.temporal`: one
population read twice has no arm to be assigned to, no selection between groups,
and therefore no counterfactual anybody is being asked to supply.
:func:`compares_principals` is the rule written under a name that says what it
keys on, and :data:`COMPARATIVE_ABSENCE_IS_AN_ASSIGNMENT_MECHANISM` is the
sentence a reader who has just been refused is owed, because **a gate that reads
as a refusal of all comparison costs this programme the ability to say whether a
process is getting better** -- which is the whole purpose of
:mod:`tenbin.report.archive` and the one question an archive of past reads exists
to answer. Silence is the worse failure here: a reader who concludes that no
comparison of any kind is available declines to build a temporal measure, nobody
finds out, and the omission is invisible for a year.

**What this module notably does not do:** it does not read the data. It judges
the declared kind, the declared granularity, and whether a mechanism is set --
never the size of a difference, the cleanliness of a sample, or the significance
of anything. That is the point rather than a limitation to apologise for: a
guard that inspected its input would fire on messy data and stay quiet on a clean
fixture, which is a speed bump a sufficiently confident caller walks around. The
test that pins this is in ``tests/test_refusal.py`` and it is named for what it
protects.
"""

from __future__ import annotations

from typing import Final

from tenbin.claims.mechanism import (
    MINIMUM_GRANULARITY,
    NO_ASSIGNMENT_MECHANISM,
    AssignmentMechanism,
    Granularity,
    is_narrower_than,
)
from tenbin.claims.model import Claim
from tenbin.claims.refusals import ComparativeRefusalError, Refusal, refusal_for_absent_mechanism

#: What is absent when this gate refuses a comparative claim, said in the one place that
#: decides it. Rendered by :mod:`tenbin.cli` beside the refusal and available to the MCP
#: surface, because the refusal's own reason argues about a corpus and does not enumerate the
#: comparisons that survive it -- and the asymmetry is the whole reason this exists. A reader
#: who is told "no counterfactual" and nothing else concludes that no comparison is available;
#: a reader who is told "no assignment mechanism between principals, and a population read twice
#: is not that" can build the most useful figure this corpus supports.
COMPARATIVE_ABSENCE_IS_AN_ASSIGNMENT_MECHANISM: Final[str] = (
    "What is absent is an assignment mechanism between principals: a stated way work was "
    "allocated between two groups, and the instrument that made the allocation observable to "
    "somebody afterwards. It is not an absence of comparison. One population read twice is one "
    "population observed at two times -- no treatment, no arm, no selection between groups, and "
    "so no counterfactual to supply -- and a claim of kind `temporal` says exactly that and is "
    "not refused by this gate."
)


def compares_principals(claim: Claim) -> bool:
    """Whether this claim compares two populations of principals, and so needs a mechanism.

    **The rule this gate applies, in one named place, keyed on the declared kind and nothing
    else.** Not the granularity, not the denominator's size, not how large the difference is,
    and not whether the caller described it as a comparison -- a rule that read anything a
    caller controls would have a verdict that moved when the call site moved, which is the
    failure mode a reader has no way to see. :attr:`Claim.requires_assignment_mechanism` is
    where the derivation lives, because the claim knows its own kind; this is the name for it
    that says what it is *for*, so the gate's rule and the decision record talk about the same
    object.

    False for ``descriptive`` (a count of one population) and false for ``temporal`` (one
    population, read twice). True for ``comparative``, and true for
    ``assignment_mechanism`` because a claim *about* the mechanism is where a comparison
    gets written under a narrower label.
    """
    return claim.requires_assignment_mechanism


class ClaimGate:
    """Whether a claim may be rendered, and the refusal to render when it may not.

    The mechanism defaults to absent, so the gate a caller builds by accident
    refuses every comparison. That is the direction to fail in: an unset
    mechanism takes away a capability nobody can be shown to have earned, while a
    defaulted-present one would hand it out with the constructor.
    """

    def __init__(
        self,
        *,
        mechanism: AssignmentMechanism | None = NO_ASSIGNMENT_MECHANISM,
        granularity_floor: Granularity = MINIMUM_GRANULARITY,
    ) -> None:
        self._mechanism = mechanism
        self._granularity_floor = granularity_floor

    @property
    def mechanism(self) -> AssignmentMechanism | None:
        """The assignment mechanism this gate permits comparisons to name, if any."""
        return self._mechanism

    @property
    def granularity_floor(self) -> Granularity:
        """The narrowest population this gate will render a claim about."""
        return self._granularity_floor

    @property
    def comparison_permitted_without_a_mechanism(self) -> str:
        """The sentence this gate owes a reader it has just refused.

        A property rather than a module constant alone so the note cannot be
        rendered by one surface and forgotten by the other: every place that shows
        a refusal reads it from the gate that produced the refusal. The gate has
        no opinion about how to render it -- ``tenbin refusals`` prints it as a
        paragraph and the MCP surface puts it beside the refusal -- and that is
        right, because the wording is a decision about the argument and not about
        the medium.
        """
        return COMPARATIVE_ABSENCE_IS_AN_ASSIGNMENT_MECHANISM

    def check(self, claim: Claim) -> Refusal | None:
        """Return the refusal that applies to ``claim``, or ``None`` if it may render.

        Returns rather than raises so a report can ask about every claim it holds
        and print all the refusals together. The order of the rules is
        deliberate: the mechanism is checked first because that refusal is about
        the corpus, and a report that showed a caller a granularity complaint
        about a claim the corpus could never have supported would have pointed at
        the easier thing to fix.

        The three rules are checked in the order that puts the unfixable first,
        so a claim with two problems is reported against the one no argument will
        change. The first rule asks :func:`compares_principals` rather than
        comparing the kind here, so that what is refused -- two populations -- and
        what is not -- one population read twice -- is stated once.
        """
        if compares_principals(claim) and self._mechanism is None:
            return refusal_for_absent_mechanism(claim.slug)
        if is_narrower_than(claim.granularity, self._granularity_floor):
            return _refusal_for_granularity(claim, self._granularity_floor)
        if claim.granularity is Granularity.individual:
            return _refusal_for_individual(claim)
        return None

    def require(self, claim: Claim) -> None:
        """Raise unless ``claim`` may be rendered.

        Returns nothing on success rather than returning the claim, so the call
        site reads as a guard -- ``gate.require(claim)`` and then render -- instead
        of as a pipeline step whose result has to be threaded somewhere. The
        render itself is :meth:`render`, which cannot be reached without this
        having passed.
        """
        refusal = self.check(claim)
        if refusal is not None:
            raise ComparativeRefusalError(refusal)

    def render(self, claim: Claim) -> str:
        """Render a claim the gate permits, naming the mechanism when one is set.

        Goes through :meth:`require` rather than trusting its own caller, because
        a ``render`` that assumed it had been checked is a ``render`` that a
        future caller will not check. The mechanism is passed to the claim rather
        than appended here, so the claim owns the sentence that a comparison
        without a mechanism is not renderable.
        """
        self.require(claim)
        return claim.render_text(mechanism=self._mechanism)


def _refusal_for_granularity(claim: Claim, floor: Granularity) -> Refusal:
    """Refuse a claim whose population is narrower than the caller will publish.

    Names the floor in the reason rather than only in the missing fact, so a
    reader who can widen the population knows that widening it is the fix and
    that the refusal is a setting rather than a missing fact.
    """
    return Refusal(
        claim_slug=claim.slug,
        title="Claim below the granularity floor",
        reason=(
            f"This claim is about {claim.granularity.value} and the floor is {floor.value}. "
            "A figure over a population narrower than a team stops being a description of "
            "work and becomes a ranking of people, which is a different artifact with "
            "different consent, retention and access requirements rather than this report "
            "with a different filter."
        ),
        missing_fact=None,
        unblocked_by=None,
    )


def _refusal_for_individual(claim: Claim) -> Refusal:
    """Refuse per-principal claims even when the floor was lowered to allow them.

    Separate from the floor refusal because it is reachable: a caller can pass
    ``granularity_floor=Granularity.individual`` and the floor rule then admits
    the claim, so without this one the prohibition would be a default rather
    than a rule. The missing fact is ``None`` on purpose -- no fact arriving from
    a Kojutsu ticket would make a per-principal ranking the same product as a
    corpus description, so offering one would be a false promise.
    """
    return Refusal(
        claim_slug=claim.slug,
        title="Per-principal claim",
        reason=(
            "Per-principal effectiveness is a different product, not this one at a finer "
            "granularity: it needs consent, retention and access decisions that a corpus "
            "description does not, and it is methodologically weak besides, because a "
            "principal is not assigned the work they did. Lowering the granularity floor "
            "does not reach this rule, and that is the point of having it separately."
        ),
        missing_fact=None,
        unblocked_by=None,
    )
