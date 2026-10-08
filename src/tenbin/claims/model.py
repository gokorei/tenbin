"""The claim: what a number means, stated before the number is rendered.

A number without a claim attached is a claim, and that sentence only holds while
the unclaimed form cannot be constructed. So :class:`Claim` has no defaults and
no optional caveats: a caller supplies the statement, what it does not mean, what
would falsify it, and the denominator, or the constructor raises naming the field
it found blank. A default here would be a default caveat, and a default caveat is
decoration that renders on every figure whether or not the figure earned one --
which is how "verified" gets upgraded a degree at a time without anybody
deciding to upgrade it.

``Denominator`` is required for the sharper version of the same reason. A rate
without a population is a number that reads as general and means "these records",
and nothing in the output distinguishes the two. So the denominator carries a
description *and* a count: a bare size would let a report publish a rate over a
population nobody wrote down, and a bare description would let nobody contradict
it.

**What this module notably does not do:** it does not know whether a claim is
permitted to exist. Building a comparative claim is cheap and always succeeds,
because whether it may be *rendered* is a question about the corpus rather than
about the call. Putting that question in the constructor would make the refusal
an exception at construction time, where the cheapest fix is to change the code
and nobody has to understand what the corpus cannot support.
:mod:`tenbin.claims.gate` answers it, and keeps the answers.

One guard does live here, because there is nowhere else it can live.
:func:`Claim.render_text` refuses a comparative claim rendered with no mechanism
to name, so a comparison cannot be printed without the thing that makes it
interpretable. Separating the claim's own text from the figure that goes with it
is the other half of that: this module renders a claim with no figure in sight,
which is the only way the claim can be tested independently of whatever a report
decided to compute.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from tenbin.claims.mechanism import (
    AssignmentMechanism,
    Granularity,
    require_text,
)
from tenbin.claims.refusals import ComparativeRefusalError, refusal_for_absent_mechanism


class ClaimKind(StrEnum):
    """The shapes a claim can take, and the two that cannot go unaccompanied.

    ``StrEnum`` rather than ``Enum`` so a kind serialises into a report and a
    JSON fixture with the same word: ``"comparative"`` in the fixture and
    ``ClaimKind.comparative`` in the code are one string, so a fixture cannot
    drift from the vocabulary by writing an enum name and a consumer cannot read
    a claim it is unable to compare against a slug.

    The members are ordered from least demanding to most, because the order is the
    argument: the first needs nothing, the second needs a past reading, the third
    needs a mechanism nobody has, and the fourth is the kind under which a
    mechanism is written about. A reader who walks the enum top to bottom learns
    what this corpus can and cannot say without being told separately.

    - ``descriptive``: one population, counted. No counterfactual is required
      because no comparison is being made, and this is the kind that carries most
      of the value of a corpus that cannot support anything else.
    - ``temporal``: one population, read twice. A statement about how a measure
      moved between two reads of the same collection -- no treatment was applied,
      no two principals were compared, and so no counterfactual has to be
      supplied. **This is not the comparison ``docs/decisions/002-no-causal-claims.md``
      refuses**, which is a comparison *between principals*; the whole content of
      the distinction is :meth:`Claim.requires_assignment_mechanism`, and it is
      stated here because an enum member with no argument next to it is how a
      decision gets re-read the other way.
    - ``comparative``: two populations against one outcome. Requires a named
      assignment mechanism, because without it a difference between the groups
      is uninterpretable rather than merely uncertain.
    - ``assignment_mechanism``: a claim *about* the mechanism that produced a
      difference -- how work was allocated, and how a later reader would check
      that. Held to the same requirement as ``comparative`` for the same reason:
      this is the kind under which a comparison is most naturally written.
    """

    descriptive = "descriptive"
    temporal = "temporal"
    comparative = "comparative"
    assignment_mechanism = "assignment_mechanism"


#: Plural nouns this package knows how to reduce, and what each becomes. A
#: mapping rather than a rule because English is the problem: ``files`` reduces to
#: ``file`` and ``analyses`` to ``analysis`` and neither falls out of dropping a
#: character, and a denominator whose only job is to say how many of something
#: there were has no business being the place a reader meets that irregularity.
#: A noun not listed here is rendered unchanged rather than guessed at, so an
#: unlisted singular is a wart and never a wrong number.
_SINGULARS: Mapping[str, str] = MappingProxyType(
    {
        "records": "record",
        "requests": "request",
        "questions": "question",
        "files": "file",
        "changes": "change",
        "pull requests": "pull request",
        "gates": "gate",
        "tickets": "ticket",
        "documents": "document",
        "captures": "capture",
        "days": "day",
        "repositories": "repository",
    }
)


def _singular(noun: str) -> str:
    """This package's singular for a counted noun, or the noun unchanged."""
    return _SINGULARS.get(noun, noun)


@dataclass(frozen=True)
class Denominator:
    """The population a rate is a rate over, named and counted together.

    ``size`` is what makes the denominator falsifiable by somebody who
    disagrees with it, and ``description`` is what makes it reproducible. Both
    are required: a count with no description cannot be checked against the
    corpus, and a description with no count cannot be checked at all.
    """

    description: str
    size: int
    size_noun: str = "records"

    def __post_init__(self) -> None:
        require_text("Denominator", "description", self.description)
        require_text("Denominator", "size_noun", self.size_noun)
        if self.size < 1:
            # A rate over zero records has no value, and the honest rendering of
            # one is a refusal rather than a division.
            raise ValueError(
                f"Denominator.size must be at least 1; a rate over {self.size} is not a rate."
            )

    def render_text(self) -> str:
        """Render the population so a figure and its denominator are one string.

        The count is formatted with separators because a bare ``1240`` read
        against a description of unknown scope invites the reader to treat it as
        small, and the size is the part that says whether it is.

        ``size_noun`` exists because a denominator that counts questions is not
        counting records, and a report that says "3 records" beside a
        description naming decision requests has told the reader the population is
        three documents when it is three questions -- which is the one thing the
        denominator exists to prevent. It defaults to ``records`` because that is
        what almost every denominator in this program is, and the exception is the
        point.

        **A count of one takes the singular**, which is why this is a small method
        rather than an f-string. "1 requests" beside a claim a reader is meant to
        take seriously is the sort of detail that teaches a reader to stop
        reading, and the only way to be sure of never writing it is to not be
        able to.
        """
        noun = self.size_noun if self.size != 1 else _singular(self.size_noun)
        return f"{self.description} ({self.size:,} {noun})"


@dataclass(frozen=True)
class Claim:
    """One claim: a statement, its limits, its falsifier, and its population.

    Every caveat field is required and every one is checked for blankness, so
    the unclaimed form is a ``ValueError`` naming the field rather than a claim
    that renders with an empty line where the caveat should be. The two optional
    fields are provenance rather than substance: ``unblocked_by`` names the
    Kojutsu ticket that would deliver a missing fact, and ``source`` names
    where the corpus evidence is documented.

    ``kind`` and ``granularity`` are required for the same reason, and neither
    gets a default. A defaulted kind would default to the *least* demanding one,
    so omitting it would quietly turn a comparative claim into a descriptive one
    and let the figure render; a defaulted granularity would default to the
    floor and make the per-principal refusal a decision each caller has to
    remember to make. Both refusals have to be ones a caller cannot skip.
    """

    slug: str
    statement: str
    does_not_mean: str
    falsifier: str
    denominator: Denominator
    kind: ClaimKind
    granularity: Granularity
    unblocked_by: str | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        for field in ("slug", "statement", "does_not_mean", "falsifier"):
            require_text("Claim", field, getattr(self, field))
        if not isinstance(self.kind, ClaimKind):
            raise ValueError(
                f"Claim.kind must be a ClaimKind, not {self.kind!r}; a bare string would "
                "compare equal to a member and hide a vocabulary that has drifted."
            )
        if not isinstance(self.granularity, Granularity):
            raise ValueError(f"Claim.granularity must be a Granularity, not {self.granularity!r}.")
        if self.unblocked_by is not None:
            require_text("Claim", "unblocked_by", self.unblocked_by)
        if self.source is not None:
            require_text("Claim", "source", self.source)

    @property
    def requires_assignment_mechanism(self) -> bool:
        """Whether the claim may not be rendered without naming a mechanism.

        True for ``comparative`` and for ``assignment_mechanism`` alike. The
        second is the one that closes the hole: a claim *about* the mechanism is
        where a comparison gets written under a different label, so a kind that
        exists to describe the mechanism must not be the one kind that gets to
        assert it while the mechanism is absent.

        **False for ``temporal``, and that is a decision rather than an omission.**
        A claim of that kind compares one population against its own past value:
        the same collection, read twice, with no treatment applied, no arm to
        assign anybody to and no selection between groups. Decision 002 refuses
        the comparison it *does* name -- agent against human, model against model --
        because the corpus holds no assignment mechanism and the selection of
        well-specified tickets onto the agent path is the entire mechanism by which
        those two groups differ. None of that argument touches one population read
        twice, because there is no second population to have been assigned anything.

        **The derivation is a membership test in the declared kind and nothing
        else** -- not the granularity, not the size of a difference, not the size of
        a denominator, not whether the caller described the claim as a comparison.
        A rule that read anything a caller controls would have a verdict that moved
        when the call site moved, and a gate whose verdicts move with the call site
        is a gate nobody can reason about. :func:`tenbin.claims.gate.compares_principals`
        is the same rule under a name that says what it keys on, and the test that
        pins it is named for the distinction rather than for the behaviour.
        """
        return self.kind in (ClaimKind.comparative, ClaimKind.assignment_mechanism)

    def render_text(self, *, mechanism: AssignmentMechanism | None = None) -> str:
        """Render the claim with no figure attached, in a fixed order.

        The order is statement, what it does not mean, what would falsify it,
        denominator, and it is fixed so that two claims in one report are
        scannable against each other -- a reader who has to re-find where the
        caveats are is reading the caveats less often.

        The mechanism is appended when the claim needs one and one is supplied,
        and naming it is the point: a comparison rendered without the mechanism
        that produced the difference is a number with a caveat attached, which is
        the thing this package exists to stop. So a comparative claim rendered
        without a mechanism raises rather than printing -- the one place in this
        module where a limitation is enforced rather than documented, because
        everything else in the package is enforced by
        :mod:`tenbin.claims.gate` and this has no other home.
        """
        if self.requires_assignment_mechanism and mechanism is None:
            raise ComparativeRefusalError(refusal_for_absent_mechanism(self.slug))
        parts = [
            f"Statement: {self.statement}",
            f"What it does not mean: {self.does_not_mean}",
            f"What would falsify it: {self.falsifier}",
            f"Denominator: {self.denominator.render_text()}",
        ]
        if mechanism is not None and self.requires_assignment_mechanism:
            parts.append(f"Assignment mechanism: {mechanism.render_text()}")
        if self.source is not None:
            # Provenance renders, because an unrendered field is a comment and the
            # history of why a refusal was lifted belongs where a reader of the figure
            # will meet it. A measure that exists because a Kojutsu ticket landed
            # says so on the page, so the lifting is auditable from the report rather
            # than only from a merge commit somebody has to go and find.
            parts.append(f"Enabled by: {self.source}")
        return "\n".join(parts)
