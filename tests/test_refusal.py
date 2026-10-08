"""The negative test decision 002 asks for, over a fixture designed to be persuasive.

Decision 002 requires that a dataset with a large, unambiguous, genuinely
comparative signal still be refused, and the fixture below is built to make that
refusal as uncomfortable as possible: two groups, five hundred changes each, and
a thirty-point gap in adoption rate in the agents' favour. The gap is not
manufactured, either -- it is exactly what the selection effect predicts, because
the agent path received the well-specified tickets. A guard that inspects its
input would look at this fixture, see a clean and decisive result, and wave it
through, and a sufficiently confident caller would then go looking for a fixture
messy enough to be refused.

So the guard does not read the data. It reads the declared kind, the declared
granularity, and whether a mechanism is set. Everything else in this module
exists to keep that guard from being the only thing tested: a gate that refused
all claims would pass the negative test perfectly and be useless, so the
permitted cases are pinned here too.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from tenbin.claims import (
    MECHANISM_ABSENT_REASON,
    NO_ASSIGNMENT_MECHANISM,
    AssignmentMechanism,
    Claim,
    ClaimGate,
    ClaimKind,
    ComparativeRefusalError,
    Denominator,
    Granularity,
    RefusalRegistry,
    default_registry,
)

#: A controlled run with a defensible instrument, so the permitted path is
#: exercised against a real mechanism rather than a string that says "mechanism".
CONTROLLED_RUN_MECHANISM = AssignmentMechanism(
    method="Randomised allocation within one repository, one quarter",
    description=(
        "Each eligible change was assigned to the human or agent path by a scheduler "
        "before work began, so the two groups received the same work in the same window."
    ),
    instrument="The assignment log, written at allocation time and readable from the store.",
)


@dataclass(frozen=True)
class GroupOutcome:
    """One group's observed outcome, with no room left to argue about the arithmetic.

    Written out rather than computed from the store because the store is not
    reachable and, more to the point, because the point of the fixture is that
    the numbers are *right*. A guard that inspected them would have a real
    measurement to be impressed by, which is the condition the test needs and the
    one a real corpus makes impossible to arrange deliberately.
    """

    label: str
    changes_written: int
    changes_adopted: int

    @property
    def adoption_rate(self) -> float:
        return self.changes_adopted / self.changes_written


#: The persuasive dataset. Five hundred changes a side, no overlap in the outcome
#: distributions worth mentioning, and a gap an effect-size test would call
#: enormous. This is the shape of evidence the refusal exists to survive.
AGENT_GROUP = GroupOutcome(label="agent-authored", changes_written=500, changes_adopted=481)
HUMAN_GROUP = GroupOutcome(label="human-authored", changes_written=500, changes_adopted=312)


def _comparative_claim(**overrides: Any) -> Claim:
    """A claim comparing the two groups above, at the granularity that is permitted.

    ``team`` rather than ``repository`` so the fixture is refused for the corpus
    reason alone. A fixture that tripped two rules at once would not show which
    rule did the refusing, and the whole subject of the negative test is that the
    reason is the absent mechanism.
    """
    fields: dict[str, Any] = {
        "slug": "agent-adoption-rate-gap",
        "statement": (
            f"Agent-authored changes were adopted at {AGENT_GROUP.adoption_rate:.0%} against "
            f"{HUMAN_GROUP.adoption_rate:.0%} for human-authored changes."
        ),
        "does_not_mean": (
            "That agents produce better changes; the two groups were not given the same work."
        ),
        "falsifier": (
            "A randomised allocation showing no difference, or a correction for ticket "
            "well-specification that removes the gap."
        ),
        "denominator": Denominator(
            description="Changes written in the measured quarter, both paths",
            size=AGENT_GROUP.changes_written + HUMAN_GROUP.changes_written,
        ),
        "kind": ClaimKind.comparative,
        "granularity": Granularity.team,
    }
    fields.update(overrides)
    return Claim(**fields)


def test_a_large_comparative_signal_is_still_refused_because_the_guard_must_not_depend_on_the_data() -> (
    None
):
    """The test decision 002 requires, named for what it protects.

    The fixture asserts its own strength before the refusal, so it cannot rot
    into a weak signal that the guard would have refused for uninteresting
    reasons. Then it asserts the refusal is the corpus's argument and not a
    comment on the data -- no digits anywhere in the rendered refusal, because a
    refusal that quoted a sample size would be a data-shaped argument, and the
    next reader would try to argue with the data instead of the mechanism.
    """
    assert AGENT_GROUP.adoption_rate - HUMAN_GROUP.adoption_rate > 0.3, (
        "the fixture must be a large, unambiguous gap or the refusal proves nothing"
    )

    gate = ClaimGate(mechanism=NO_ASSIGNMENT_MECHANISM)
    with pytest.raises(ComparativeRefusalError) as raised:
        gate.require(_comparative_claim())

    refusal = raised.value.refusal
    assert refusal.reason == MECHANISM_ABSENT_REASON
    assert refusal.missing_fact is not None
    assert "assignment mechanism" in refusal.missing_fact
    assert refusal.unblocked_by is None

    rendered = refusal.render_text()
    assert not any(character.isdigit() for character in rendered), (
        "the refusal must be about the corpus, not about the numbers it was handed"
    )
    for word in ("sample", "significant", "p-value", "effect", "confidence"):
        assert word not in rendered.casefold()


def test_a_raised_refusal_names_the_claim_and_the_reason_because_a_stack_trace_is_not_an_explanation() -> (
    None
):
    """The exception message has to survive being read on its own in a log.

    A caller who catches this and prints ``str(exc)`` gets the claim slug and the
    argument; a caller who does not get a ``Refusal`` they can add to a report.
    Both are worth having, and the second is why the whole refusal rides on the
    exception rather than being formatted into the message.
    """
    claim = _comparative_claim()
    with pytest.raises(ComparativeRefusalError) as raised:
        ClaimGate().require(claim)
    assert str(raised.value).startswith(f"{claim.slug}:")
    assert MECHANISM_ABSENT_REASON in str(raised.value)
    assert raised.value.refusal.claim_slug == claim.slug


def test_check_returns_the_refusal_rather_than_raising_because_a_report_has_to_list_every_one_it_hits() -> (
    None
):
    """``check`` for reports, ``require`` for callers, because they want opposites.

    A report raising on the first refused claim would render one refusal out of
    nine and say nothing about the other eight, which reads as though the other
    eight were permitted. So the return carries the refusal and ``require`` is
    the thin wrapper that turns it into a failure.
    """
    gate = ClaimGate()
    refusal = gate.check(_comparative_claim())
    assert refusal is not None
    assert refusal.missing_fact is not None
    assert "assignment mechanism" in refusal.missing_fact
    assert gate.check(_comparative_claim(kind=ClaimKind.descriptive)) is None


def test_a_descriptive_claim_renders_normally_because_a_program_that_refuses_everything_is_useless() -> (
    None
):
    """A count of one population needs no counterfactual, so the gate opens.

    Nearly everything this corpus supports is descriptive. If the gate were
    simply strict, the right response to the first complaint would be to loosen
    it, and the loosening would be the thing nobody reviewed.
    """
    claim = _comparative_claim(
        slug="capture-rate-over-observed-changes",
        statement="Most observed changes carry at least one knowledge record.",
        kind=ClaimKind.descriptive,
        granularity=Granularity.area,
    )
    rendered = ClaimGate().render(claim)
    assert "Most observed changes" in rendered
    assert "Assignment mechanism" not in rendered


def test_a_set_mechanism_lets_a_comparative_claim_render_and_names_it_for_the_reader_to_judge() -> (
    None
):
    """With a mechanism, the same claim renders -- and the mechanism is in it.

    This is the deliberate act decision 002 asks for: turning the refusal off
    takes a mechanism, not a flag. The instrument is asserted on because that is
    the part a reader would be trusting, and a mechanism rendered without one
    would be a story with a citation.
    """
    gate = ClaimGate(mechanism=CONTROLLED_RUN_MECHANISM)
    rendered = gate.render(_comparative_claim())
    assert CONTROLLED_RUN_MECHANISM.method in rendered
    assert CONTROLLED_RUN_MECHANISM.instrument in rendered
    assert gate.check(_comparative_claim()) is None


def test_a_gate_built_without_a_mechanism_refuses_by_default_because_an_accidental_gate_should_fail_closed() -> (
    None
):
    """The mechanism is not required to be *passed* unset; it is unset by default.

    Fails closed, so the constructor a caller reaches for by accident takes away a
    capability rather than handing one out. The reverse default would mean a demo
    or a test fixture quietly licensed a comparison the corpus does not support.
    """
    assert ClaimGate().mechanism is None
    assert ClaimGate().granularity_floor is Granularity.team


def test_individual_granularity_is_refused_even_when_the_floor_was_lowered_because_it_is_a_different_product() -> (
    None
):
    """Lowering the floor is not a way past the rule; the rule is separate.

    Without a rule that ignores the floor, ``granularity_floor=individual`` would
    be a one-argument bypass of the prohibition on per-principal effectiveness.
    And no fact arriving from a Kojutsu ticket can make it the same product as
    a corpus description, so the refusal offers no ticket -- an empty field there
    would be a false promise.
    """
    claim = _comparative_claim(kind=ClaimKind.descriptive, granularity=Granularity.individual)
    refusal = ClaimGate(granularity_floor=Granularity.individual).check(claim)
    assert refusal is not None
    assert "different product" in refusal.reason
    assert refusal.missing_fact is None
    assert refusal.unblocked_by is None


def test_granularity_at_or_above_the_floor_renders_because_the_floor_excludes_only_what_is_narrower() -> (
    None
):
    """``team`` and ``area`` are the two the floor admits, and both are asserted.

    A floor that is off by one is a floor that either refuses a permitted
    population or admits a prohibited one, and the first is the failure that
    gets reported as "the program is broken".
    """
    gate = ClaimGate()
    for granularity in (
        Granularity.team,
        Granularity.area,
        Granularity.repository,
        Granularity.corpus,
    ):
        claim = _comparative_claim(kind=ClaimKind.descriptive, granularity=granularity)
        assert gate.check(claim) is None, f"{granularity} should render"
    below = _comparative_claim(kind=ClaimKind.descriptive, granularity=Granularity.individual)
    refusal = gate.check(below)
    assert refusal is not None
    assert "floor" in refusal.reason


def test_a_freshly_constructed_registry_reports_the_mechanism_unset_with_the_reason_available() -> (
    None
):
    """The corpus's real state, with the argument reachable from the object holding it.

    The reason is available whether or not a mechanism is set, because the second
    question -- what would it take -- is asked by whoever just filled the field
    in, not only by whoever is deciding whether to. And it is available from the
    registry rather than only from a decision record, because the person asking
    is holding the registry.
    """
    registry = default_registry()
    assert registry.mechanism is None
    assert registry.mechanism_unset_reason == MECHANISM_ABSENT_REASON
    assert "assigned" in registry.mechanism_unset_reason

    licensed = RefusalRegistry(mechanism=CONTROLLED_RUN_MECHANISM)
    assert licensed.mechanism_unset_reason == MECHANISM_ABSENT_REASON, (
        "the reason must outlive the emptiness or it stops answering the harder question"
    )
