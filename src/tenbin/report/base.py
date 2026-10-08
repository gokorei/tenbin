"""One section of a report: exactly one thing to say, and the caveats that travel with it.

There are two shapes a report can take that are both broken and neither of which looks
broken. The first is the hole -- a heading with nothing under it -- which is what a
report renders when a measure produced no figure and the caller had nothing to put
there, and which reads as an absence of findings rather than as a computation that
never ran. The second is the contradiction: a figure together with the refusal that
would have replaced it, so that a reader is left holding a number and a refusal about
the same measure and has to work out which one this program meant. So
:meth:`Section.__post_init__` requires exactly one of the two, and refuses the zero
and the two alike.

**A figure must also bring its claim, and a refusal must not.** The first is this
whole program's central claim: a figure that can be constructed without the thing that
says what it means is a number that renders, and the render is the harm. The second is
the other half of the same thought -- a refused measure renders its refusal, not the
claim that was refused, because a section holding both would be showing the argument
the gate rejected next to the rejection.

``heading`` and ``slug`` are properties rather than fields because both are derivable
from exactly one of the three fields, and a section that stored them separately would
be a section with a fourth way to disagree with itself. ``slug`` in particular is what
a report is keyed by, and it is the join between a figure, a gate refusal and a
catalogue entry -- so it has to be the same string in all three, which is why it is
computed from the claim or the refusal rather than supplied.

**What this module notably does not do:** it does not decide what a section says, and
it holds no opinion about a measure. It does not check a claim against the gate, does
not render, and does not order anything -- a section is a container with one
invariants-bearing constructor, and :mod:`tenbin.report.ordering` is what decides
where its contents appear. Nor does it carry a corpus: the header belongs to the
report as a whole (:mod:`tenbin.report.document`) rather than to each section, so
that a section can be tested without one and the corpus verdict is stated once.
"""

from __future__ import annotations

from dataclasses import dataclass

from tenbin.claims.model import Claim
from tenbin.claims.refusals import Refusal
from tenbin.measures.base import Figure


@dataclass(frozen=True)
class Section:
    """One measure's answer: a claim and the figure it licenses, or the refusal instead.

    All three fields are required and none has a default, so a caller has to say
    which of the two shapes it is building. That is the same posture as
    :class:`~tenbin.claims.model.Claim`: a default here would be a default answer,
    and a section that defaulted to "no figure, no refusal" would render a blank
    heading -- a defect that is invisible in the output and obvious in review, which
    is the order of visibility that gets it through.

    Exactly one of ``figure`` and ``refusal`` must be present. ``claim`` is required
    with ``figure`` and refused with ``refusal``, for the reasons in the module
    docstring: the claim is the interpretation of the figure, and it is precisely what
    a refusal denies the reader.
    """

    claim: Claim | None
    figure: Figure | None
    refusal: Refusal | None

    def __post_init__(self) -> None:
        if self.figure is None and self.refusal is None:
            raise ValueError(
                "a Section needs a figure or a refusal; neither is a hole in the report, "
                "and a heading with nothing under it reads as an absence of findings "
                "rather than as a measure that did not run."
            )
        if self.figure is not None and self.refusal is not None:
            raise ValueError(
                "a Section cannot hold a figure and the refusal that would have replaced "
                "it; a reader holding both has to guess which one this program meant."
            )
        if self.figure is not None and self.claim is None:
            raise ValueError(
                "a Section holding a figure must hold its claim; a number rendered "
                "without the statement of what it means is a claim made silently, and "
                "this is the type that is meant to make that unconstructible."
            )
        if self.refusal is not None and self.claim is not None:
            raise ValueError(
                "a Section holding a refusal must not hold the claim it refuses; the "
                "refusal is the report's answer, and rendering the rejected argument "
                "beside it would put back what the gate took out."
            )

    @property
    def slug(self) -> str:
        """The identifier this section is cited by, and the join to a refusal catalogue.

        Read off the claim or the refusal rather than stored, because a stored slug is
        a fourth field that can disagree with the three it describes, and a section
        whose slug names a different measure than its figure is a figure a reader
        cannot look up.
        """
        if self.claim is not None:
            return self.claim.slug
        assert (
            self.refusal is not None
        )  # guaranteed by __post_init__, and asserted rather than assumed
        return self.refusal.claim_slug

    @property
    def heading(self) -> str:
        """The line that names this section for a reader deciding whether to read it.

        The figure's title for a rendered measure and the refusal's title for a
        refused one, because those are the two sentences each of those types was
        written to be named by -- and a section headed by its slug would be headed by
        an identifier, which tells a reader nothing about what they are about to read.
        """
        if self.figure is not None:
            return self.figure.title
        assert self.refusal is not None
        return self.refusal.title
