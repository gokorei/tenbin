"""The claim type, tested for the shape that makes the unclaimed form unbuildable.

These tests are about construction. If a caveat can be left out, the sentence
"a number without a claim attached is a claim" is a preference rather than a
property of the program, and the next person who needs a figure by Friday will
leave the falsifier out. So every test here checks that a missing or blank caveat
is a ``ValueError`` naming the field, rather than a claim that renders with an
empty line where the caveat belongs.

The rendering tests are about the claim alone. ``render_text`` produces a claim
with no figure attached, and separating the two is what lets these assertions
exist: a test that had to build a figure to check a claim would be testing the
figure, and would stop passing for reasons that had nothing to do with the
claim's wording.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tenbin.claims import (
    MECHANISM_ABSENT_REASON,
    AssignmentMechanism,
    Claim,
    ClaimKind,
    ComparativeRefusalError,
    Denominator,
    Granularity,
)
from tenbin.claims.mechanism import is_narrower_than

MECHANISM = AssignmentMechanism(
    method="Randomised allocation of changes to the agent path",
    description="A scheduler assigned each eligible change to one path before work began.",
    instrument="The assignment record, retained in the store and readable after the fact.",
)


def _claim(**overrides: Any) -> Claim:
    """Build a claim that is valid, so a test can invalidate exactly one field.

    Overriding through a dict rather than by writing a fresh literal in every
    test keeps each test's subject visible: what differs from a valid claim is the
    thing under test, and a test that had to restate the other six fields to
    change one would be a test where the interesting difference is easy to miss.
    """
    fields: dict[str, Any] = {
        "slug": "decision-request-capture-rate",
        "statement": "Most captured changes carry at least one knowledge record.",
        "does_not_mean": (
            "That most changes are documented; a change nobody asked about leaves no record."
        ),
        "falsifier": (
            "A census of observed-but-uncaptured changes showing most changes carry none."
        ),
        "denominator": Denominator(description="Pull requests observed by Kojutsu", size=1240),
        "kind": ClaimKind.descriptive,
        "granularity": Granularity.repository,
    }
    fields.update(overrides)
    return Claim(**fields)


def test_a_blank_caveat_is_a_value_error_naming_the_field_because_a_blank_is_a_comment() -> None:
    """The four caveat fields are the claim, and none of them may be empty.

    An empty statement or falsifier satisfies the constructor in the strict sense
    that it was passed, and it renders as a figure with a gap where the caveat
    should be. Nothing downstream can tell that gap from a rendering bug, so it
    has to fail here or not at all.
    """
    for field in ("slug", "statement", "does_not_mean", "falsifier"):
        with pytest.raises(ValueError, match=rf"Claim\.{field} must not be blank"):
            _claim(**{field: "   "})


def test_a_missing_caveat_is_refused_rather_than_defaulted_because_a_default_is_a_decoration() -> (
    None
):
    """``None`` is refused for the same reason a blank is, and names the field.

    A ``None`` caveat is the case a careful reviewer stops to think about, so it
    is the one most likely to be given a default. That default would be a caveat
    that renders on every figure whether or not the figure earned it.
    """
    with pytest.raises(ValueError, match=r"Claim\.falsifier is required"):
        _claim(falsifier=None)


def test_a_blank_denominator_description_is_refused_because_an_unnamed_population_reads_as_general() -> (
    None
):
    """A rate over an unnamed population reads as general and means "these records".

    The count alone cannot be checked against the corpus and the prose alone
    cannot be checked at all, so both travel together or the denominator is a
    caption.
    """
    with pytest.raises(ValueError, match=r"Denominator\.description must not be blank"):
        Denominator(description="", size=1240)


def test_a_denominator_of_zero_is_refused_because_a_rate_over_no_records_has_no_value() -> None:
    """Zero is not a small denominator; it is a division that cannot be made.

    Rendering it would put a number in a report that means nothing, which is the
    single most dangerous output this program can produce -- a real-looking
    figure with no claim behind it.
    """
    with pytest.raises(ValueError, match="Denominator.size must be at least 1"):
        Denominator(description="Pull requests observed by Kojutsu", size=0)


def test_the_kind_is_required_because_a_defaulted_kind_would_default_to_the_weakest_claim() -> None:
    """No default kind, and no bare string standing in for one.

    A default would be ``descriptive``, because that is the kind that demands
    least, so omitting it would quietly downgrade a comparison to a count and let
    the figure render. A bare string is refused for a subtler reason: because
    ``StrEnum`` members compare equal to their values, a string would satisfy
    every check while being invisible to ``isinstance``, so a drifted fixture
    would pass until the day somebody switched a comparison to ``==``.
    """
    with pytest.raises(ValueError, match="Claim.kind must be a ClaimKind"):
        _claim(kind="comparative")
    with pytest.raises(ValueError, match="Claim.granularity must be a Granularity"):
        _claim(granularity="team")


def test_a_blank_ticket_id_is_refused_because_an_empty_reference_looks_like_a_reference() -> None:
    """``None`` says "no ticket unblocks this"; ``""`` says a ticket was forgotten.

    The two are different statements and only one of them is a decision, so the
    field has to distinguish them rather than collapsing both to "empty".
    """
    with pytest.raises(ValueError, match=r"Claim\.unblocked_by must not be blank"):
        _claim(unblocked_by="")
    assert _claim(unblocked_by=None).unblocked_by is None


def test_a_kind_serialises_into_a_fixture_with_the_same_word_a_reader_sees() -> None:
    """``StrEnum`` so a JSON fixture and a report cannot disagree about a kind.

    The value is the word, not an enum name, so a fixture written by hand stays
    valid as the vocabulary changes spelling conventions and does not need
    regenerating because a member was renamed.
    """
    assert ClaimKind.comparative == "comparative"
    assert json.loads(json.dumps({"kind": ClaimKind.descriptive}))["kind"] == "descriptive"
    assert list(ClaimKind) == [
        ClaimKind.descriptive,
        ClaimKind.temporal,
        ClaimKind.comparative,
        ClaimKind.assignment_mechanism,
    ]


def test_a_denominator_counts_in_the_noun_it_counts_and_takes_the_singular_at_one() -> None:
    """A population of one is not a population of one *thing* of that kind.

    ``Denominator.size_noun`` exists because a denominator that counts questions is
    not counting records, and "3 records" beside a description naming decision
    requests has told the reader the population is three documents. The count of
    one is the same problem one unit later: "1 requests" beside a claim a reader is
    meant to take seriously is the sort of detail that teaches a reader to stop
    reading, and the only reliable way not to write it is not to be able to.
    """
    assert (
        Denominator("questions in this read", 3, size_noun="questions")
        .render_text()
        .endswith("(3 questions)")
    )
    assert (
        Denominator("questions in this read", 1, size_noun="questions")
        .render_text()
        .endswith("(1 question)")
    )
    assert (
        Denominator("pull requests in this read", 1, size_noun="pull requests")
        .render_text()
        .endswith("(1 pull request)")
    )
    # The default is the noun almost every denominator in this program is.
    assert Denominator("capture records in this read", 2).render_text().endswith("(2 records)")
    # An unlisted noun is rendered unchanged rather than guessed at, so an unlisted
    # singular is a wart and never a wrong number.
    assert Denominator("widgets", 1, size_noun="widgets").render_text().endswith("(1 widgets)")


def test_the_rendering_order_is_fixed_so_two_claims_in_one_report_can_be_scanned_together() -> None:
    """Statement, then what it does not mean, then the falsifier, then the denominator.

    Fixed rather than arbitrary: a reader who has to search for the caveats stops
    reading them, and a caveat nobody reads is decoration. Asserting the exact
    sequence rather than containment is what makes a reordering a test failure
    instead of a silent change to what a reader is shown first.
    """
    rendered = _claim().render_text().splitlines()
    assert [line.split(":", 1)[0] for line in rendered] == [
        "Statement",
        "What it does not mean",
        "What would falsify it",
        "Denominator",
    ]


def test_the_claim_renders_with_no_figure_attached_because_a_figure_hides_whether_the_claim_holds() -> (
    None
):
    """Four lines, none of which is a measurement.

    The separation is what makes this assertion possible at all. A test that had
    to render a figure to check its claim would be asserting on the figure, and
    would fail for reasons that had nothing to do with whether the claim is
    honest.
    """
    rendered = _claim().render_text()
    assert len(rendered.splitlines()) == 4
    assert "1,240 records" in rendered


def test_a_comparative_claim_cannot_render_without_naming_the_mechanism_that_produced_the_gap() -> (
    None
):
    """The one guard in the model, because there is nowhere else it can live.

    A comparison printed without the mechanism is a number with a caveat
    attached, which is the exact thing this package exists to prevent, and the
    claim is the only layer that can see whether a mechanism was supplied. The
    refusal is the corpus's own argument, verbatim, so the reason a reader sees
    in a traceback is the reason they will read in a report.
    """
    claim = _claim(kind=ClaimKind.comparative, granularity=Granularity.team)
    with pytest.raises(ComparativeRefusalError) as raised:
        claim.render_text()
    assert raised.value.refusal.reason == MECHANISM_ABSENT_REASON
    assert raised.value.refusal.claim_slug == claim.slug


def test_a_comparative_claim_that_names_its_mechanism_renders_it_for_the_reader_to_judge() -> None:
    """With a mechanism supplied, the comparison renders and the mechanism is in it.

    The mechanism is printed rather than referenced so a reader can decide
    whether the instrument is one they accept, instead of trusting the report's
    judgement that it was worth finding. The instrument is asserted on
    specifically, because a mechanism with no instrument is a story and the
    easiest part of one to leave out.
    """
    claim = _claim(kind=ClaimKind.comparative, granularity=Granularity.team)
    rendered = claim.render_text(mechanism=MECHANISM)
    assert "Assignment mechanism:" in rendered
    assert MECHANISM.method in rendered
    assert MECHANISM.instrument in rendered
    assert rendered.splitlines()[-1].startswith("Assignment mechanism:")


def test_a_mechanism_claim_needs_a_mechanism_too_because_it_is_where_a_comparison_hides() -> None:
    """The kind that describes the mechanism cannot assert one that is absent.

    Left open, this kind is where a comparison gets written under a label that
    means something narrower, and the gate would see ``assignment_mechanism``
    and let it past. So the two kinds that touch a counterfactual are held
    together deliberately.
    """
    claim = _claim(kind=ClaimKind.assignment_mechanism, granularity=Granularity.team)
    assert claim.requires_assignment_mechanism is True
    with pytest.raises(ComparativeRefusalError):
        claim.render_text()


def test_a_descriptive_claim_needs_no_mechanism_because_refusing_everything_makes_the_program_useless() -> (
    None
):
    """Most of what this corpus supports is a count, and a count needs no counterfactual.

    Wording it the other way: the guard has to have a door it opens, or the next
    complaint is that the program refuses everything, and the right answer to
    that complaint is a bug rather than a policy change.
    """
    assert _claim().requires_assignment_mechanism is False
    assert "Assignment mechanism" not in _claim().render_text()


def test_granularity_is_ordered_narrowest_first_because_the_floor_is_an_ordering() -> None:
    """An enum is unordered, so the ordering the gate depends on is written down here.

    Getting this backwards inverts the floor rule silently: a corpus-wide claim
    would be refused and a per-principal one rendered, and the test that would
    catch it is a test of the floor, not of the enum.
    """
    assert is_narrower_than(Granularity.individual, Granularity.team) is True
    assert is_narrower_than(Granularity.team, Granularity.area) is True
    assert is_narrower_than(Granularity.team, Granularity.team) is False
    assert is_narrower_than(Granularity.corpus, Granularity.team) is False
