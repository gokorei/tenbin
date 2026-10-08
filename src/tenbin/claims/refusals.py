"""The catalogue of measures the corpus cannot support, and the type that carries them.

A refusal is a claim about the corpus that has to be made as carefully as any
claim rendered in a report. Someone asking "why is review throughput not in the
dashboard" deserves a specific answer -- this fact is missing, it dies here, this
Kojutsu ticket would deliver it -- and not a shrug and not silence. So a
refusal names the measure, says what is missing, and says what would change the
answer, and the third of those is the part that is usually wrong.

**``unblocked_by=None`` is an answer, not a gap.** It means no Kojutsu ticket
unblocks this and that is the correct outcome. Three measures in the catalogue are
in that state -- one of them the project's central question, and one of them
refused because the fact that would answer it exists and sits outside this
program's read seam, so no store change could ever deliver it. The distinction
matters because the two mistakes available here are both silent: filing a ticket
that cannot possibly deliver the fact, and leaving a field empty so that a reader
assumes somebody is working on it. An empty string is therefore never a value
for this field -- the constructor refuses it, because a blank ticket id in a
report reads as a reference to something.

**A refusal is retired when its blocker lands, not defended afterwards.** The
temptation is to keep a refusal and soften it, because a measure that was going to
be written anyway deserves a note explaining why it is hard. But a refusal that
outlives its blocker is the worst of both worlds: it tells a reader a thing is
impossible when it is merely unbuilt, and it tells the team an expensive ticket is
still needed. Kojutsu landed ``740111c`` and four entries came out.

What this module notably does not do: it does not decide whether a refusal is
*warranted*, only that one exists and states itself. Whether a comparative claim
is permitted is :mod:`tenbin.claims.gate`'s question. Keeping the catalogue and
the gate apart is what lets a reader of this file see the whole list of what the
corpus cannot say in one screen, without reading the rules that consult it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from tenbin.claims.mechanism import (
    ASSIGNMENT_MECHANISM_MISSING_FACT,
    MECHANISM_ABSENT_REASON,
    require_text,
)

#: What a reader sees in place of a ticket list. A named constant rather than an
#: inline string because this exact sentence is the point of the ``None`` branch,
#: and a branch whose wording is written at the call site is a branch whose
#: wording will differ between call sites.
NO_TICKET_UNBLOCKS_THIS: str = "no Kojutsu ticket unblocks this"

#: Characters that are not slug material, collapsed to a single separator. The
#: collision risk is real and accepted: two measures differing only in punctuation
#: would share a slug, and the registry's duplicate rejection would then surface
#: it as a contradiction rather than letting one silently shadow the other.
_SLUG_SEPARATOR = re.compile(r"[^a-z0-9]+")


def slugify(measure: str) -> str:
    """Derive a claim slug from a measure name, so a document and the code agree.

    The catalogue is keyed by slugs and ``docs/seam.md`` is keyed by measure
    names, so something has to translate. Doing that with one function used by
    both sides is what turns a naming coincidence into a checkable binding: if a
    row in that table is renamed, the completeness test looks for a slug nobody
    registered and fails, rather than the row quietly dropping out of coverage
    because the test matched on a string that moved.

    A name that slugifies to nothing is refused rather than becoming ``""``,
    because an empty slug is a refusal that cannot be looked up.
    """
    slug = _SLUG_SEPARATOR.sub("-", measure.casefold()).strip("-")
    if not slug:
        raise ValueError(f"measure name must not slugify to nothing: {measure!r}")
    return slug


def format_unblocked_by(*tickets: str) -> str:
    """Join ticket ids into the single field that names them, in document order.

    One string rather than a tuple because the field is rendered into a report
    line and appears in prose, and two representations of the same fact is one
    more thing to keep in step. Order is the caller's, and the callers use the
    order ``docs/seam.md`` lists so a reader comparing the two does not have to.
    """
    if not tickets:
        raise ValueError("format_unblocked_by requires at least one ticket; use None instead")
    return ", ".join(tickets)


def _reject_blank_if_present(owner: str, field: str, value: str | None) -> None:
    """Refuse a blank where the field is optional but a value was supplied.

    The distinction this draws is between *absent* and *present and empty*, and
    it is the whole reason ``unblocked_by`` can carry a decision. ``None`` is the
    answer "no ticket unblocks this"; ``""`` is a ticket id that was never
    written down, which is a bug in the caller rather than a statement about the
    corpus, and would render identically to both of the things it is not.
    """
    if value is not None:
        require_text(owner, field, value)


@dataclass(frozen=True)
class Refusal:
    """One measure the corpus cannot support, stated so a reader can act on it.

    ``unblocked_by`` is ``None`` for a measure that no ticket unblocks, and that
    is a decision rather than a missing reference -- see the module docstring. It
    is never an empty string; :meth:`__post_init__` refuses one, because a blank
    ticket list rendered into a report is a reference to nothing that looks like
    a reference to something.

    ``missing_fact`` is optional because not every refusal is about an absent
    fact. A per-principal claim is refused because it is the wrong product, and
    there is no fact whose arrival would make it one.
    """

    claim_slug: str
    title: str
    reason: str
    missing_fact: str | None = None
    unblocked_by: str | None = None

    def __post_init__(self) -> None:
        require_text("Refusal", "claim_slug", self.claim_slug)
        require_text("Refusal", "title", self.title)
        require_text("Refusal", "reason", self.reason)
        _reject_blank_if_present("Refusal", "missing_fact", self.missing_fact)
        _reject_blank_if_present("Refusal", "unblocked_by", self.unblocked_by)

    def render_text(self) -> str:
        """Render the refusal as a reader would need it, with nothing elided.

        The ``what would unblock it`` line is unconditional. A refusal whose
        answer is "nothing will" is a different answer from one that says
        nothing, and the second is how a reader ends up waiting for a ticket that
        was never going to be filed. Every line here is required to be non-empty
        by :meth:`__post_init__`, so a rendered refusal cannot have an empty
        reason next to an empty unblocking.
        """
        lines = [
            f"Refused: {self.title}",
            f"Claim: {self.claim_slug}",
            f"Why: {self.reason}",
        ]
        if self.missing_fact is not None:
            lines.append(f"What is missing: {self.missing_fact}")
        unblocking = self.unblocked_by if self.unblocked_by is not None else NO_TICKET_UNBLOCKS_THIS
        lines.append(f"What would unblock it: {unblocking}")
        return "\n".join(lines)


class ComparativeRefusalError(RuntimeError):
    """Raised when a caller renders a claim the gate will not permit.

    Carries the whole :class:`Refusal` rather than a message built from it,
    because the message is for a human reading a traceback and the refusal is
    for a report listing every measure it hit -- and a caller that catches this
    and wants the second thing should not have to parse the first.

    ``RuntimeError`` rather than ``ValueError`` because nothing about the call is
    malformed. The claim is well-formed, the data may be perfect, and the refusal
    is about what the corpus can support -- which is exactly the condition a
    ``ValueError`` message invites somebody to fix by changing the argument.
    """

    def __init__(self, refusal: Refusal) -> None:
        self.refusal = refusal
        super().__init__(f"{refusal.claim_slug}: {refusal.reason}")


def refusal_for_absent_mechanism(claim_slug: str) -> Refusal:
    """Build the refusal for a comparative claim with no mechanism behind it.

    One factory rather than two call sites constructing it, because the claim
    layer and the gate both need it and the wording of a refusal is part of what
    it asserts. A comparative claim refused by :func:`tenbin.claims.model.Claim.render_text`
    and one refused by the gate have to give the reader the same answer, or the
    answer depends on which path the caller happened to take.
    """
    return Refusal(
        claim_slug=claim_slug,
        title="Comparative claim with no assignment mechanism",
        reason=MECHANISM_ABSENT_REASON,
        missing_fact=ASSIGNMENT_MECHANISM_MISSING_FACT,
        unblocked_by=None,
    )


def refusal_for_measure(
    measure: str,
    *,
    reason: str,
    missing_fact: str | None,
    unblocked_by: str | None = None,
) -> Refusal:
    """Build a catalogue entry, deriving its slug from the measure it blocks.

    The slug is never written out here. Hand-writing it beside the measure name
    invites the two to drift, and a drifted slug is invisible: the entry still
    renders, the report still refuses the measure, and the only symptom is that
    a caller asking ``registry.get("review-verdict-rates")`` gets a ``KeyError``
    on a refusal that exists.
    """
    return Refusal(
        claim_slug=slugify(measure),
        title=measure,
        reason=reason,
        missing_fact=missing_fact,
        unblocked_by=unblocked_by,
    )


#: Every measure this program refuses, in the order ``docs/seam.md`` lists it.
#: The reasons restate the specific fact that died rather than gesturing at the
#: table in the inventory, because a refusal whose reader has to go and find the
#: inventory has already lost the reader who asked.
#:
#: **Entries here are the ones whose blocker is still open, and only those.** Four
#: were removed when Kojutsu landed ``740111c`` and this package gained the four
#: measures that blocker delivered: verdicts by reviewer (``RJNVJK1P``), time to
#: first review (``9PTQE45Y``), change authorship (``3QRPK52A``) and change outcome
#: (``QEVSTMYW``). Two more were removed when ``MWF7Z1EN`` landed and delivered the
#: file anchor -- single-custodian knowledge per area and rationale staleness against
#: code movement -- and the measures that replaced them are in
#: :mod:`tenbin.measures.custody` and :mod:`tenbin.measures.staleness`. A refusal
#: that outlives its blocker is worse than no refusal -- it tells a reader the thing
#: is impossible when it is merely unbuilt, and it tells the team an expensive ticket
#: is still needed.
#:
#: At least one entry is not a row in ``docs/seam.md``: "whether knowledge was
#: retrieved" is a plain count rather than a causal claim, and it is refused
#: because the log that would answer it lives outside this program's read seam.
#: The completeness test in ``tests/test_refusal_registry`` therefore asserts the
#: direction that matters -- every blocked row in the document has an entry here --
#: and deliberately does not assert the reverse, so an entry the document has no
#: row for does not need one to justify itself.
CATALOGUE: tuple[Refusal, ...] = (
    refusal_for_measure(
        "Unanswered decision requests, and their age",
        reason=(
            "This refusal used to name a ticket, and the ticket has landed. Kojutsu's "
            "projection shipped on `AZ8T5XYS` and the measure it unblocked is built: the "
            "corpus now holds the lifecycle of every decision request that reached a terminal "
            "state, and Tenbin reports the wait before the terminal state and the spread "
            "across `answered`, `failed` and `superseded`. What is still missing is the half "
            "this measure is named for. A `pending` or `claimed` request is *unanswered* by "
            "definition and the projection omits both, deliberately: outstanding work is "
            "operational, rewritten on every lease, and the outbox that carries it is built "
            "for immutable records, so a live queue in a knowledge store reads as a set of "
            "open decisions rather than as requests this process is still waiting on. So the "
            "age that is now measurable is the age of requests that already resolved, and "
            "'how long has the queue been' is still unanswerable. No ticket unblocks that, "
            "and that is a structural answer rather than a gap in the plan: delivering it "
            "needs a delivery path Kojutsu does not have, so filing a request for another "
            "run of the same projection would send a reader to wait for work that has already "
            "been done. The `claim_token` half of the old reason still holds and is now "
            "checked rather than promised -- it is the `WHERE`-clause guard on releasing a "
            "claim, and Tanseki is readable by agents, so a projected token is a capability "
            "handed to every reader, and `tenbin.measures.projection_safety` scans every "
            "projected decision request for the keys that must not cross the seam."
        ),
        missing_fact=(
            "the outstanding population the projection omits: the `pending` and `claimed` "
            "requests, which are re-written on every lease and need a delivery path that is "
            "not the one built for immutable records"
        ),
        unblocked_by=None,
    ),
    refusal_for_measure(
        "Coverage of development",
        reason=(
            "Kojutsu writes only on success, so a change that produced a capture and "
            "a change nobody examined are the same observation from outside this corpus, "
            "and no backfill path exists for either. Every rate this program can publish "
            "is therefore a rate over the changes that made a record, while the reader "
            "arriving with 'how much of development is covered' is asking about the "
            "changes that did not. The census of uncaptured changes is the one fact that "
            "decides it, and it has not landed: without it a coverage figure is a "
            "statement about a selected population wearing the name of a statement "
            "about development. This refusal outlived its neighbour -- the merge fact "
            "landed, and the outcome measure reports captured outcomes without claiming "
            "to describe development -- and it is the half of that older refusal that is "
            "still true."
        ),
        missing_fact="a census of changes that produced no capture",
        unblocked_by=format_unblocked_by("EB6FE5ZP"),
    ),
    refusal_for_measure(
        "Whether knowledge was retrieved",
        reason=(
            "The read log exists. Kojutsu shipped one on `2XMZ0TYY` "
            "(`src/kojutsu/core/read_log.py`) and it records tool, query, filters, "
            "result count, outcome and time -- which is exactly the evidence this measure "
            "would need. It is also a local file, deliberately outside Tanseki, and it "
            "carries no authenticated caller identity because the MCP server is stdio "
            "and a single trust domain. So this is not a missing fact any more: it is a "
            "fact on the wrong side of this program's read seam, and no change to the "
            "store will bring it across, because Kojutsu is right that a read event "
            "is not knowledge and Tenbin is right never to read Kojutsu's local "
            "state. The remedy is therefore not a ticket but a decision about what "
            "Tenbin is allowed to read -- and even a log that crossed the seam would "
            "not answer the causal question, because it says what was read and never "
            "what would have happened had it not been (see "
            "`docs/decisions/002-no-causal-claims.md`). Filing a ticket here would "
            "convert a decision about the seam into a queue item, which is the one way "
            "this refusal could be lost."
        ),
        missing_fact=(
            "a read log inside the corpus Tenbin reads, carrying an authenticated "
            "caller identity -- a change to what Tenbin may read, not a Kojutsu ticket"
        ),
        unblocked_by=None,
    ),
    refusal_for_measure(
        "Whether retrieval changed anything",
        reason=(
            "Two independent reasons, and either one alone leaves the measure "
            "unrenderable, so this refusal has to render both. (1) No read log in this "
            "read seam: Kojutsu writes one on `2XMZ0TYY` and keeps it as a local file, "
            "so the treatment cannot be measured here at all, and no amount of reading "
            "Tanseki will supply it. (2) No counterfactual: even a complete read log on the "
            "near side of the seam would say only what was read, never what would have "
            "happened had it not been read, so the claim needs the assignment mechanism "
            "the corpus does not contain -- see "
            "`docs/decisions/002-no-causal-claims.md`. A reader who filed `2XMZ0TYY`, "
            "saw the read log arrive, and expected this measure to light up is "
            "disappointed twice, and both disappointments are the finding: the first is "
            "a seam, the second is a decision that outlives any ticket."
        ),
        missing_fact=(
            "read events inside the corpus Tenbin reads, and a counterfactual Tenbin will not have"
        ),
        unblocked_by=None,
    ),
    refusal_for_measure(
        "Whether agentic development is more effective",
        reason=(
            "This is the project's central question and it is refused for the reason "
            "the whole package exists: the corpus holds no assignment mechanism, and no "
            "change to the store can produce one. Agents receive the well-specified "
            "tickets and humans receive the rest, and that selection is the entire "
            "mechanism by which the two groups differ, so a difference in outcome "
            "between them is uninterpretable rather than merely uncertain. "
            "`docs/seam.md` records that no ticket in the Kojutsu project unblocks "
            "it, and that this is the correct outcome rather than a gap in the plan; "
            "filing a ticket here would convert a decision into a queue item, which is "
            "the specific way this refusal would be lost."
        ),
        missing_fact=ASSIGNMENT_MECHANISM_MISSING_FACT,
        unblocked_by=None,
    ),
)
