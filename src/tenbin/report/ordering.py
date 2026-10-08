"""The one ordering, in one function, because a discipline does not survive a deadline.

A report that renders the caveat before the figure has to be *written* that way, and
writing is the part that slips. A template grows a branch, a hurried fix puts the
number first because the number is what the reader asked for, and a reformat that
preserves every word moves the caveat under the figure where nobody will look at it.
So the order here is a function rather than a convention: :func:`ordered_parts`
returns one section's content as a fixed sequence of typed parts, and both renderers
-- Markdown and JSON -- walk that sequence and decide nothing. Neither renderer has an
opinion about order, which is what makes the property hold for both forms at once:
there is one ordering to be wrong about, and a test can assert on it by index.

**The falsifier sits above the figure, and that is the unfashionable part of it.**
Every report convention in this field puts the caveats at the foot, under a heading
like "Notes", on the reasoning that a reader will come back for them. That reasoning
assumes a reader who reads to the end, and the reader who stops after the number is
the one this program exists for: they are skimming for the figure, they will take the
number away, and whatever they did not read is a caveat that did not travel with it.
So "what would falsify this" goes immediately *above* the number -- a reader who stops
early has already been told what would have made it wrong. It reads oddly in review,
it will be proposed to be moved to the foot for tidiness, and the reason is written
here so that the proposal has to argue with the reason rather than with taste.

The claim's own three caveats come first for the same reason and in decreasing order
of how often they are the thing that matters: what the claim is, what it is not, and
what would prove it wrong. The denominator comes last because it is not a caveat at
all -- it is the population the rate is over, and it is the part a reader looks *up*
when they have already doubted the rate, so putting it above the number would spend
the reader's attention on the one part they did not come for.

**The figure part is the payload preceded by the figure's own completeness sentence.**
:meth:`~tenbin.measures.base.Figure.rate_text` is called, not reimplemented: the
report layer does not know how to summarise a read, and a layer that rebuilt the
verdict from the snapshot could disagree with the figure about whether it is whole
while both looking correct. It is placed inside the figure part rather than beside the
claim's caveats because it is the one caveat specific to *this reading* rather than to
the claim -- a reader who re-runs the measure next month gets the same three caveats
and a different completeness sentence, and only this one changes.

**A refusal renders the same way, or it is a stub.** The reason, the missing fact when
there is one, and the unblocking ticket -- or the explicit statement that nothing
unblocks it, which is an answer rather than a gap. The unblocking line is
unconditional and the missing-fact line is not, and the asymmetry is the argument: a
refusal whose answer is "nothing will" is a different fact from one that says nothing,
and the second is how a reader ends up waiting for a ticket that was never going to be
filed. What a refusal never produces is a stub, an empty chart or a zero: there is no
figure, so there is nothing to render as a number, and the section is the refusal.

**What this module notably does not do:** it does not render. There is no Markdown
here, no JSON, no escaping and no layout, so there is nothing to keep in step with a
template and no second copy of the order. It does not order *sections* -- which
section comes first in a report is :mod:`tenbin.report.build`'s question, and it is a
question about the corpus rather than about how a claim is read. It does not consult
the gate, so a section holding a figure is taken at its word here; the refusal it would
have received is already a section by the time anything is ordered. And it does not
look at the corpus header, which is content in its own right and gets its own ordering
from the same function in :mod:`tenbin.report.document`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from tenbin.claims.mechanism import require_text
from tenbin.claims.refusals import NO_TICKET_UNBLOCKS_THIS, Refusal
from tenbin.report.base import Section


class PartKind(StrEnum):
    """What a part of a section is, in the words a renderer may use for it.

    ``StrEnum`` so the JSON renderer's field name and the Markdown renderer's label
    come from one vocabulary: ``PartKind.does_not_mean`` is ``"does_not_mean"`` in the
    payload and "What it does not mean" on the page, and the two cannot be given
    different words by two renderers because there is only one place they are written
    down.

    The claim members and the refusal members are kept apart rather than shared. A
    refusal has a reason and no falsifier, a claim has a falsifier and no reason, and
    sharing a kind between them would be a way for one renderer's lookup table to
    satisfy a part the other half of the document never emits.
    """

    #: What was read, and how much of it arrived.
    collection = "collection"
    read_at = "read_at"
    #: The declared window a period-scoped read was cut to. Optional -- an unscoped
    #: report has none -- and placed with the header rather than with the sections,
    #: because the window qualifies every figure in the document at once and a reader who
    #: has to find it per figure will not find it before trusting one.
    period = "period"
    enumeration = "enumeration"
    completeness = "completeness"
    truncation = "truncation"

    #: The claim's four fields, in the order the module docstring argues for.
    statement = "statement"
    does_not_mean = "does_not_mean"
    falsifier = "falsifier"
    figure = "figure"
    denominator = "denominator"

    #: What a refusal says instead of a number.
    reason = "reason"
    missing_fact = "missing_fact"
    unblocking = "unblocking"

    @property
    def label(self) -> str:
        """The word this part is introduced by in rendered prose.

        A property on the kind rather than a table in the Markdown renderer, so that
        adding a part is one edit in one file: a new kind with no label would be a
        ``KeyError`` in a renderer, which is loud but late, whereas a missing member
        here is a missing member of an enum every reader of this module can see.
        """
        return _PART_LABELS[self]


#: Every kind with the word that introduces it. A mapping rather than a chain of
#: ``if`` statements so that the vocabulary is one screen, and so that the JSON field
#: name beside each label is visible next to it -- which is what makes the two
#: renderers agreeing on content a checkable claim rather than an assertion.
_PART_LABELS: dict[PartKind, str] = {
    PartKind.collection: "Corpus",
    PartKind.read_at: "Read at",
    PartKind.period: "Period",
    PartKind.enumeration: "Enumerated",
    PartKind.completeness: "Completeness",
    PartKind.truncation: "Truncation",
    PartKind.statement: "Statement",
    PartKind.does_not_mean: "What it does not mean",
    PartKind.falsifier: "What would falsify it",
    PartKind.figure: "Figure",
    PartKind.denominator: "Denominator",
    PartKind.reason: "Why not",
    PartKind.missing_fact: "What is missing",
    PartKind.unblocking: "What would unblock it",
}


@dataclass(frozen=True)
class Part:
    """One piece of a section, typed so a renderer knows what it is holding.

    The type is the point. A renderer handed a list of strings has to decide what each
    one is, and the ways it can decide -- position, prefix matching, a dictionary it
    wrote itself -- are all ways to be wrong quietly. A renderer handed parts asks
    :attr:`kind` and prints :attr:`text`, so the label table lives here and the order
    lives in :func:`ordered_parts`, and neither is repeated per renderer.

    ``text`` is checked for blankness through the package's own validator, because a
    blank part is the exact failure this package exists to prevent: an empty caveat
    line under a number looks like a rendering bug and reads like a claim with no
    caveat. Refusing it here means the defect cannot reach a renderer at all.
    """

    kind: PartKind
    text: str

    def __post_init__(self) -> None:
        require_text("Part", "text", self.text)


def ordered_parts(section: Section) -> Sequence[Part]:
    """This section's content, in the one order there is.

    For a figure: the statement, what it does not mean, what would falsify it, the
    figure with its own completeness sentence, and the denominator. For a refusal: why
    not, what is missing when there is a missing thing, and what would unblock it --
    the last of those unconditionally, because ``None`` there means *nothing unblocks
    this* and that is an answer a reader is entitled to.

    There is no configuration and no parameter for a different order. That is the
    whole argument of this module: an ordering that can be configured is an ordering
    somebody will configure, and the person who configures it is a person in a hurry
    who has been asked for a number. A caller who wants a different order wants a
    different program.

    The figure is asked for its :meth:`~tenbin.measures.base.Figure.rate_text` and
    is not asked what the snapshot said, because the figure is the object that holds
    the read and the sentence about it, and a report that recomputed the verdict would
    be a second implementation of a rule that already exists.
    """
    refusal = section.refusal
    if refusal is not None:
        return _refusal_parts(refusal)
    claim = section.claim
    figure = section.figure
    if claim is None or figure is None:
        # Unreachable: ``Section`` refuses to be constructed in either shape. Raised
        # rather than asserted so that a future relaxation of those rules fails here
        # as a refusal to order a section it cannot describe, not as a ``None``
        # propagating into a renderer.
        raise ValueError("a Section must hold a figure or a refusal; see Section.__post_init__")
    return (
        Part(PartKind.statement, claim.statement),
        Part(PartKind.does_not_mean, claim.does_not_mean),
        Part(PartKind.falsifier, claim.falsifier),
        Part(PartKind.figure, _figure_text(figure.rate_text(), figure.value_text())),
        Part(PartKind.denominator, claim.denominator.render_text()),
    )


def _refusal_parts(refusal: Refusal) -> tuple[Part, ...]:
    """The parts of a refusal, with the missing fact only when there is one.

    Two omissions are possible and they are not the same. A refusal with no missing
    fact -- a per-principal claim, a claim below the granularity floor -- has none
    because no fact arriving would make it this product rather than a different one,
    and naming a missing fact there would be a false promise of a ticket. A refusal
    with no unblocking ticket has none because the correct outcome is that nothing
    unblocks it, and *that* is rendered, in :data:`NO_TICKET_UNBLOCKS_THIS`, because a
    reader who cannot tell those two apart is a reader waiting on a ticket that was
    never going to be filed.

    The title is not here because it is the section's heading rather than its body:
    :attr:`Section.heading` supplies it and every renderer puts a heading first, so
    its position is not a decision any renderer gets to make differently.
    """
    parts = [Part(PartKind.reason, refusal.reason)]
    if refusal.missing_fact is not None:
        parts.append(Part(PartKind.missing_fact, refusal.missing_fact))
    unblocking = (
        refusal.unblocked_by if refusal.unblocked_by is not None else NO_TICKET_UNBLOCKS_THIS
    )
    parts.append(Part(PartKind.unblocking, unblocking))
    return tuple(parts)


def _figure_text(rate_text: str, value_text: str) -> str:
    """The figure's completeness sentence and its payload, in that order.

    Completeness first, for the reason the whole module is about: the sentence that
    qualifies the number belongs above the number. The payload is appended rather than
    merged, so a distribution's buckets and its exclusions keep the line breaks that
    distinguish a bucket that did not occur from a bucket that was filtered out.

    A figure with an empty payload contributes only its completeness sentence. That is
    the honest rendering of a measure that had no number to state, and a measure in
    that position should be a :class:`~tenbin.measures.base.Finding` -- which can
    *say* that it found nothing, where a bare figure can only go blank.
    """
    return f"{rate_text}\n{value_text}" if value_text else rate_text
