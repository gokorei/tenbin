"""The completeness test: what ``docs/seam.md`` forecloses, this package must refuse.

``docs/seam.md`` already states, measure by measure, which parts of the corpus
Tenbin cannot use and which Kojutsu ticket would change that. The catalogue
in :mod:`tenbin.claims.refusals` says the same thing in code. Two statements of
one fact drift, and the drift is invisible: the document keeps reading
correctly while the code stops refusing a measure it once refused, and the first
symptom is a figure in a report that nobody expected.

So this module parses the table out of the document and asserts that every measure
it marks as blocked has an entry here. The parsing is the load-bearing part and it
is the part most completeness tests get wrong. A regex that finds no rows
produces zero failures and a green suite, which is worse than having no test at
all, because it is now evidence of coverage. So the parser is total -- it
recognises exactly the shape the document has and raises
:class:`UnrecognisedTableShapeError` on anything else, including a renamed heading, a
renamed column, an unfamiliar value in the "Unblocked by" cell, and a table with
no data rows -- and a test below proves each of those refusals.

The check runs in one direction only: document to catalogue. Some catalogue entries
have no row in the document, and the reason is in the module docstring of
:mod:`tenbin.claims.refusals`. Asserting the reverse would force the document to
grow rows the author did not intend, and a test that can only be satisfied by
editing a design document is a test that will be edited out.

**When a count moves, the message says which side is stale.** ``docs/seam.md`` and
this catalogue move together and they are edited by different hands, so the
interesting failure is always one of two: the document has been updated and a
refusal outlived its blocker, or the catalogue has been updated and the document
still forecloses a measure somebody built. Both look like a failing count from
here, and sending the next person to the wrong file is how one of the two gets
reverted. So every count assertion below names the two possibilities in the order
they should be checked, and the completeness check names the measure whose row no
longer has an entry.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from tenbin.claims import (
    NO_TICKET_UNBLOCKS_THIS,
    Refusal,
    RefusalRegistry,
    default_registry,
    format_unblocked_by,
    slugify,
)

SEAM_DOCUMENT = Path(__file__).resolve().parents[1] / "docs" / "seam.md"

#: The three counts the document's table must have. Pinned as named constants rather
#: than written into the assertions so that a failure message can quote what the
#: document actually says next to what this file expected -- a bare number in an
#: assertion tells the reader nothing about which of the two files to open.
#:
#: The blocked/permitted split moved from 8/3 to 6/5 when Kojutsu shipped eight
#: commits on ``feat/corpus-answerability`` and two of the rows stopped being
#: blocked, and then from 6/5 to 4/7 when ``MWF7Z1EN`` delivered the file anchor and
#: the two rows that were blocked on it became buildable. These are the document's
#: numbers, verified against Kojutsu's source rather than against its ticket
#: statuses, which had not caught up.
EXPECTED_ROW_COUNT = 19
EXPECTED_BLOCKED_ROWS = 4
EXPECTED_PERMITTED_ROWS = 15

#: The heading the parse is anchored to, matched exactly. An approximate match
#: would find a different section and then complain about its table, which reads
#: as a broken parser rather than as a renamed heading.
SECTION_HEADING = "## What that table forecloses, measure by measure"

#: The three columns the parse requires, in order. Checked rather than assumed so
#: that a column added for somebody else's benefit fails here, loudly, instead of
#: shifting the "Unblocked by" cell out of the index the parse reads.
EXPECTED_COLUMNS = ("measure", "blocked on", "unblocked by")

#: A Kojutsu ticket id, as it appears in the document: eight upper-case
#: alphanumerics in backticks.
TICKET_ID = re.compile(r"`([A-Z0-9]{8})`")

#: The markdown link form used for a decision reference inside an unblocking cell.
MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

#: A cell that says the measure is not blocked at all. Compared casefolded and
#: whitespace-collapsed because a table's alignment padding is not meaning.
PERMITTED_CELLS = frozenset({"available now"})

#: A cell that says no ticket unblocks this. A phrase rather than a set of
#: synonyms on purpose: "nothing in this project" is a decision, and a decision
#: paraphrased into the parse is a decision nobody wrote down.
NO_TICKET_CELLS = frozenset({"nothing in this project"})

#: A near-miss copy of the real section, used to build the shapes the parse must
#: refuse. Kept as a literal rather than derived from the real file so a test of
#: the parser cannot be broken by an edit to the document.
WELL_FORMED_SECTION = """\
## What that table forecloses, measure by measure

| Measure | Blocked on | Unblocked by |
|---|---|---|
| Independence and capture-source distribution | nothing | available now |
| Rationale revision pressure | nothing | available now |
| Time to first review | open time | `9PTQE45Y` |
| Whether retrieval changed anything | read events, and a counterfactual Tenbin will not have | `2XMZ0TYY`, and see [`002-no-causal-claims.md`](decisions/002-no-causal-claims.md) |
| Whether agentic development is more effective | an assignment mechanism | nothing in this project |

The last row is the one worth reading twice.
"""


class UnrecognisedTableShapeError(ValueError):
    """The document's foreclosed-measures table is not the shape this parse requires.

    A ``ValueError`` so that a caller which does not catch it still fails the
    suite, and distinct so that the failure reads as "the parse refused" rather
    than as "the parse matched nothing".
    """


@dataclass(frozen=True)
class ForeclosedRow:
    """One row of the document's table, as this module understands it.

    ``permitted`` and ``decision_refs`` are both needed because a cell naming a
    ticket *and* a decision is neither of the two simple cases. It says a fact is
    missing and that filing the ticket would not be enough, and the registry entry
    that answers it has to be the one carrying the second half of the argument.
    """

    measure: str
    permitted: bool
    unblocked_by: str | None
    decision_refs: tuple[str, ...]

    @property
    def is_blocked(self) -> bool:
        return not self.permitted


def _cells(line: str) -> list[str]:
    """Split a markdown table row into its cells, rejecting a malformed one."""
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        raise UnrecognisedTableShapeError(f"not a table row: {line!r}")
    return [cell.strip() for cell in stripped[1:-1].split("|")]


def _section_body(lines: list[str]) -> list[str]:
    """Return the lines of the named section, or refuse that the section is gone.

    Anchored on the heading as an exact string. A ``startswith`` would survive a
    retitling and then parse whatever table happened to be underneath, which is
    the failure this whole module exists to prevent one level up.
    """
    try:
        start = lines.index(SECTION_HEADING)
    except ValueError:
        raise UnrecognisedTableShapeError(
            f"docs/seam.md has no section headed {SECTION_HEADING!r}; "
            "a renamed heading must fail here rather than skip the completeness check"
        ) from None
    body: list[str] = []
    for line in lines[start + 1 :]:
        if line.startswith("## "):
            break
        body.append(line)
    return body


def _table_lines(body: list[str]) -> list[str]:
    """Collect the consecutive table lines at the top of a section body."""
    lines: list[str] = []
    for line in body:
        if not line.strip():
            if lines:
                break
            continue
        if not line.lstrip().startswith("|"):
            break
        lines.append(line)
    return lines


def _check_header(cells: list[str]) -> None:
    """Refuse a header that is not the three columns this parse indexes by."""
    if tuple(cell.casefold() for cell in cells) != EXPECTED_COLUMNS:
        raise UnrecognisedTableShapeError(
            f"expected the columns {EXPECTED_COLUMNS}, found {tuple(cells)}; "
            "the parse reads 'Unblocked by' by position and will not guess"
        )


def _check_separator(cells: list[str]) -> None:
    """Refuse a table whose second line is not a markdown column separator."""
    if len(cells) != len(EXPECTED_COLUMNS) or not all(
        re.fullmatch(r":?-{2,}:?", cell) for cell in cells
    ):
        raise UnrecognisedTableShapeError(f"expected a column separator, found {tuple(cells)}")


def _classify_unblocking(cell: str) -> tuple[bool, str | None, tuple[str, ...]]:
    """Read the "Unblocked by" cell as exactly one of the states it can be in.

    The three states are: permitted, blocked with no ticket unblocking it, and
    blocked with one or more tickets. Anything else -- an unquoted value, a
    paraphrase of the no-ticket phrase, a cell naming a ticket among prose this
    parse does not recognise -- is refused, because each of those would otherwise
    be treated as "no ticket named" and quietly downgrade a blocked measure to a
    permitted one.
    """
    compact = " ".join(cell.split())
    folded = compact.casefold()
    if folded in PERMITTED_CELLS:
        return True, None, ()
    if folded in NO_TICKET_CELLS:
        return False, None, ()

    tickets: list[str] = []
    references: list[str] = []
    for part in (piece.strip() for piece in compact.split(",")):
        ticket = TICKET_ID.fullmatch(part)
        if ticket is not None:
            tickets.append(ticket.group(1))
            continue
        if part.startswith("and see "):
            link = MARKDOWN_LINK.search(part)
            if link is None:
                raise UnrecognisedTableShapeError(f"unreadable decision reference: {part!r}")
            references.append(link.group(2))
            continue
        raise UnrecognisedTableShapeError(
            f"unrecognised 'Unblocked by' entry: {part!r}; a blocked cell must be "
            "ticket ids, or the no-ticket phrase, or a ticket plus a decision reference"
        )
    if not tickets:
        raise UnrecognisedTableShapeError(
            f"'Unblocked by' names no ticket and is not the no-ticket phrase: {cell!r}"
        )
    return False, format_unblocked_by(*tickets), tuple(references)


def parse_foreclosed_table(markdown: str) -> tuple[ForeclosedRow, ...]:
    """Parse the foreclosed-measures table, or refuse a shape this parse does not know.

    Every way of getting it wrong raises rather than returning fewer rows, because
    every one of them ends in a completeness check that passes on nothing.
    """
    body = _section_body(markdown.splitlines())
    table = _table_lines(body)
    if not table:
        raise UnrecognisedTableShapeError(
            f"no table under {SECTION_HEADING!r}; a renamed heading or a reworded "
            "intro must not leave the completeness check with nothing to compare"
        )
    _check_header(_cells(table[0]))
    if len(table) < 2:
        raise UnrecognisedTableShapeError("the table has a header and no rows")
    _check_separator(_cells(table[1]))

    rows: list[ForeclosedRow] = []
    for offset, line in enumerate(table[2:], start=3):
        cells = _cells(line)
        if len(cells) != len(EXPECTED_COLUMNS):
            raise UnrecognisedTableShapeError(
                f"row {offset} has {len(cells)} cells, expected {len(EXPECTED_COLUMNS)}: {line!r}"
            )
        measure, permitted, tickets, references = cells[0], *_classify_unblocking(cells[2])
        if not measure:
            raise UnrecognisedTableShapeError(f"row {offset} has an empty measure name")
        rows.append(
            ForeclosedRow(
                measure=measure,
                permitted=permitted,
                unblocked_by=tickets,
                decision_refs=references,
            )
        )
    if not rows:
        raise UnrecognisedTableShapeError(
            "the table parsed to zero data rows, which is the one result a completeness "
            "test must never accept"
        )
    return tuple(rows)


def _decision_number(link_target: str) -> str:
    """The number in a decision reference, e.g. ``002`` from ``decisions/002-x.md``."""
    return Path(link_target).name.partition("-")[0]


def _refusal_with(**overrides: Any) -> Refusal:
    """Build a refusal that is valid, so a test can invalidate exactly one field.

    Taken as ``Any`` because one of the tests here has to pass a value the
    annotation forbids on purpose, which is the only honest way to check that a
    caller building a refusal from a parsed document gets a ``ValueError`` naming
    the field rather than an ``AttributeError`` from inside the check.
    """
    fields: dict[str, Any] = {
        "claim_slug": "a-refusal-under-test",
        "title": "A refusal under test",
        "reason": "A reason that is not blank.",
    }
    fields.update(overrides)
    return Refusal(**fields)


def _rows() -> tuple[ForeclosedRow, ...]:
    """The document's foreclosed table, parsed, as every test in this module wants it."""
    return parse_foreclosed_table(SEAM_DOCUMENT.read_text(encoding="utf-8"))


def _count_disagreement(rows: tuple[ForeclosedRow, ...], *, expected: int, what: str) -> str:
    """Say which of the two files is stale, because a count alone does not.

    The two states are: the document moved and a refusal outlived its blocker, or the
    catalogue moved and the document still forecloses a measure that is now built.
    They are indistinguishable from a failing number, and they are fixed in opposite
    places, so the message states both in the order they should be tried -- the
    catalogue first, because a refusal nobody retires is the one that keeps telling a
    reader the impossible is impossible.
    """
    blocked = sum(1 for row in rows if row.is_blocked)
    permitted = len(rows) - blocked
    return (
        f"docs/seam.md lists {len(rows)} rows ({blocked} blocked, {permitted} available now); "
        f"this test expected {expected} {what}. If the document is right, the catalogue in "
        "tenbin.claims.refusals.CATALOGUE is stale: retire the refusal for a measure whose "
        "blocker has landed, and narrow the ones whose blocker is only half delivered. If the "
        "catalogue is right, the document has not caught up with it, and the two counts here "
        "are what should move."
    )


def test_the_parse_finds_the_rows_it_is_asserting_about_because_matching_nothing_looks_like_matching_everything() -> (
    None
):
    """The guard on the guard: assert the parse produced rows before asserting on them.

    A completeness test that parses zero rows and finds nothing missing is green,
    and it is now being cited as evidence that the catalogue is complete. So the
    row count and a spot-check on a known measure are asserted first, and both
    live in their own test so that a future zero-row regression fails with a
    message about the parse rather than about the catalogue.
    """
    rows = _rows()
    assert len(rows) == EXPECTED_ROW_COUNT, _count_disagreement(
        rows, expected=EXPECTED_ROW_COUNT, what="rows"
    )
    measures = {row.measure for row in rows}
    assert "Whether agentic development is more effective" in measures, (
        "the last row is the one the document says is worth reading twice, and a parse "
        "that cannot see it is not a parse of this table"
    )
    assert "Coverage of development" in measures, (
        "the census half of the retired outcome refusal is still refused, so its row must "
        "still be here for the completeness check to be checking anything"
    )
    assert sum(1 for row in rows if row.is_blocked) == EXPECTED_BLOCKED_ROWS, _count_disagreement(
        rows, expected=EXPECTED_BLOCKED_ROWS, what="blocked rows"
    )


def test_every_measure_the_seam_document_marks_blocked_has_a_refusal_here_because_a_foreclosure_can_go_unwritten() -> (
    None
):
    """The completeness claim, in the direction that can be checked.

    A row naming tickets must have an entry listing exactly those tickets, in the
    document's order, because a refusal naming the wrong ticket sends a reader to
    a ticket that will not help. A row naming a ticket *and* a decision must have
    an entry with no ticket at all -- filing the ticket would not finish the
    measure, so listing it would be the same false promise as an empty field.

    The failure message names every disagreement and both files, because a blocked
    row with no entry is one of two states -- a document that still forecloses a
    measure somebody built, or a catalogue that stopped refusing one -- and they are
    fixed in opposite places. All of them are collected rather than raised on the
    first, because the two files are edited by different hands and a person
    reconciling them wants the whole list rather than the first disagreement.
    """
    registry = default_registry()
    problems: list[str] = []
    for row in _rows():
        if row.permitted:
            continue
        slug = slugify(row.measure)
        if slug not in registry:
            problems.append(
                f"  {row.measure!r}: docs/seam.md marks it blocked on "
                f"{row.unblocked_by or 'no ticket'}, and the catalogue has no entry for "
                f"{slug!r}. If the blocker has landed the refusal outlived it and must go; "
                "if it has not, the entry is missing."
            )
            continue
        refusal = registry.get(slug)
        if row.decision_refs:
            if refusal.unblocked_by is not None:
                problems.append(
                    f"  {row.measure!r}: the document names a ticket and a decision, so no "
                    f"ticket finishes it, but the catalogue names {refusal.unblocked_by!r}."
                )
            rendered = refusal.render_text()
            for reference in row.decision_refs:
                if _decision_number(reference) not in rendered:
                    problems.append(
                        f"  {row.measure!r}: the refusal must render the decision "
                        f"({_decision_number(reference)}) that completes it."
                    )
        elif refusal.unblocked_by != row.unblocked_by:
            problems.append(
                f"  {row.measure!r}: the document names {row.unblocked_by!r} as the "
                f"unblocking and the catalogue names {refusal.unblocked_by!r}. One of the "
                "two is quoting a ticket that has already landed, or omitting one that has not."
            )
    assert not problems, (
        "docs/seam.md and tenbin.claims.refusals.CATALOGUE disagree about what this corpus "
        "cannot support:\n" + "\n".join(problems)
    )


def test_a_measure_the_seam_document_marks_available_now_is_not_refused_because_a_catalogue_of_everything_is_useless() -> (
    None
):
    """The permitted rows are asserted absent, which is the other half of the check.

    Without this the parse would be satisfied by a refusal for every row, and a
    report refusing the independence distribution would be refusing the corpus's
    strongest axis on no grounds at all. Every permitted row is checked, and the
    count of them is checked too: a document that has quietly grown a fourth
    available row has changed what this corpus supports, and a test that only
    checked membership would not have noticed.
    """
    rows = _rows()
    permitted = [row.measure for row in rows if row.permitted]
    registry = default_registry()
    assert len(permitted) == EXPECTED_PERMITTED_ROWS, _count_disagreement(
        rows, expected=EXPECTED_PERMITTED_ROWS, what="available-now rows"
    )
    for measure in permitted:
        slug = slugify(measure)
        assert slug not in registry
        with pytest.raises(KeyError, match="no refusal is registered"):
            registry.get(slug)


def test_the_parse_refuses_a_shape_it_does_not_recognise_because_a_renamed_column_must_not_turn_the_check_off() -> (
    None
):
    """Every near-miss of the real table, each of which must raise rather than match nothing.

    Four ways for this table to stop being this table: the heading is retitled,
    a column is renamed, a value in the unblocking column is reworded, and the
    data rows disappear. Each of them is a plausible edit and each of them would
    otherwise leave the completeness test comparing an empty set against the
    catalogue -- which passes, and is then trusted.
    """
    cases = {
        "retitled heading": WELL_FORMED_SECTION.replace(
            "## What that table forecloses, measure by measure", "## Foreclosed measures"
        ),
        "renamed column": WELL_FORMED_SECTION.replace("| Measure |", "| Metric |"),
        "unfamiliar unblocking value": WELL_FORMED_SECTION.replace("`9PTQE45Y`", "soon"),
        "no data rows": "\n".join(WELL_FORMED_SECTION.splitlines()[:4]) + "\n",
    }
    for name, markdown in cases.items():
        with pytest.raises(UnrecognisedTableShapeError):
            parse_foreclosed_table(markdown), name


def test_a_refusal_with_no_ticket_renders_the_answer_because_an_empty_field_looks_like_somebody_is_working_on_it() -> (
    None
):
    """``None`` renders as a sentence, and the sentence is the same one every time.

    Four measures are in this state: the project's central question, the causal
    reading, the read log that exists on the far side of this program's seam, and
    the outstanding decision requests the registry projection deliberately omits.
    All four are cases where the right answer is that nothing will -- three because
    the fact is outside the seam, one because the delivery path the fact needs is
    not the one Kojutsu built -- and an empty field would be read as a reference
    to a ticket that is merely not written down, so the reader would wait for a
    ticket nobody is going to file.
    """
    registry = default_registry()
    without_tickets = [r for r in registry.all() if r.unblocked_by is None]
    assert len(without_tickets) == 4
    for refusal in without_tickets:
        rendered = refusal.render_text()
        assert NO_TICKET_UNBLOCKS_THIS in rendered
        assert f"What would unblock it: {NO_TICKET_UNBLOCKS_THIS}" in rendered


def test_the_read_log_refusal_names_the_seam_and_the_decision_because_the_fact_exists_and_no_ticket_will_deliver_it() -> (
    None
):
    """The refusal says the data is on the wrong side of the seam, and that is a different answer.

    ``2XMZ0TYY`` landed, so the old wording -- nothing observes reads, and the events
    did not happen anywhere that kept them -- is now false, and a refusal that still
    said it would be sending a reader to look for a fact that has been in existence
    for a release. The honest answer names both halves of what is actually true: the
    log is a local file Kojutsu is right not to put in Tanseki, so no change to the
    store brings it across; and even if it did, it says what was read and never what
    would have happened without it, which is decision 002 and is not a ticket.
    """
    refusal = default_registry().get("whether-knowledge-was-retrieved")
    assert refusal.unblocked_by is None, (
        "a fact that exists outside this program's read seam cannot be delivered by any "
        "Kojutsu ticket, so naming one here would be the same false promise as an "
        "empty field"
    )

    rendered = refusal.render_text()

    assert "2XMZ0TYY" in rendered, "the reader has to know the log exists"
    assert "local file" in rendered, "the seam limit is the reason, and it has to be stated"
    assert "stdio" in rendered, "why there is no caller identity is part of the same limit"
    assert "002" in rendered, "the counterfactual is the half no ticket delivers"
    assert "Nothing observes reads" not in rendered, (
        "the read log landed; a refusal still claiming nothing observes reads is a refusal "
        "that outlived its blocker"
    )
    assert refusal.missing_fact is not None
    assert "not a Kojutsu ticket" in refusal.missing_fact


def test_the_retrieval_refusal_renders_both_reasons_because_a_read_log_alone_would_surprise_a_reader_twice() -> (
    None
):
    """Both halves of the argument, and the ticket is withheld from the field.

    A reader who filed ``2XMZ0TYY`` and watched the log land would expect the causal
    measure to light up. It will not, twice over: the log is outside this read seam,
    and the counterfactual is not a thing any ticket delivers. A refusal rendering
    only one half would be the kind of omission that costs somebody a sprint, and the
    ``unblocked_by`` field is where the omission would hide: a ticket id there reads
    as "this is what you are waiting for".
    """
    refusal = default_registry().get("whether-retrieval-changed-anything")
    assert refusal.unblocked_by is None

    rendered = refusal.render_text()
    assert "2XMZ0TYY" in rendered
    assert "002" in rendered
    assert "No read log" in rendered
    assert "No counterfactual" in rendered
    assert "seam" in rendered
    assert refusal.missing_fact is not None
    assert "counterfactual" in refusal.missing_fact


def test_a_refusal_whose_blocker_landed_has_been_retired_because_a_refusal_that_outlives_its_blocker_is_worse_than_no_refusal() -> (
    None
):
    """Four refusals are gone, and each of the four measures that replaced them is here.

    Checked in both directions on purpose. A retired refusal that nobody noticed
    would go on telling a reader that a verdict rate is impossible and that a ticket
    is still needed -- and a reader who checked the ticket would find it closed. The
    replacement measures are asserted too, so the retirement cannot be a deletion
    that left the measure unwritten: if somebody removes a refusal *first* and the
    measure never lands, this fails.
    """
    from tenbin.measures import (
        ChangeAuthorshipMeasure,
        ChangeOutcomeMeasure,
        ReviewVerdictMeasure,
        TimeToReviewMeasure,
    )

    registry = default_registry()
    retired = {
        "Review verdict rates by reviewer": ReviewVerdictMeasure,
        "Time to first review": TimeToReviewMeasure,
        "Outcome of a change": ChangeOutcomeMeasure,
    }
    for measure, replacement in retired.items():
        slug = slugify(measure)
        assert slug not in registry, (
            f"{measure!r} is measured now -- {replacement().slug} -- so its refusal is a "
            "refusal that outlived its blocker. Remove it, and the reader stops being told "
            "the impossible is impossible."
        )
        assert replacement().slug not in registry, (
            "a measure whose slug the registry refuses would be replaced by the refusal "
            "rather than by its own figure"
        )
    assert ChangeAuthorshipMeasure is not None, (
        "authorship was unblocked by 3QRPK52A and had no refusal to retire; the measure is "
        "asserted here so the ticket is not mistaken for one that needed one"
    )


def test_a_refusal_whose_blocker_has_landed_in_full_is_gone_because_half_a_blocker_is_no_longer_what_is_missing() -> (
    None
):
    """The file anchor arrived, so both refusals that named it are gone and replaced.

    Two of them, and they are the pair that proves the point in both directions. Each
    was blocked on ``MWF7Z1EN`` and nothing else, so neither was ever half delivered:
    a rationale with no file anchor had no code to be stale relative to, and a custody
    figure had nothing to attach a custodian to. Both measures exist and both are
    permitted, and a refusal that outlived its blocker would be telling a reader that
    a buildable figure is impossible while naming a ticket that is already closed.

    The document is checked too, because the two files are edited separately and the
    direction that fails silently is the one where only the code was updated.
    """
    from tenbin.measures import RationaleStalenessMeasure, SingleCustodianMeasure

    registry = default_registry()
    retired = {
        "Single-custodian knowledge per area": SingleCustodianMeasure,
        "Rationale staleness against code movement": RationaleStalenessMeasure,
    }
    for measure, replacement in retired.items():
        slug = slugify(measure)
        assert slug not in registry, (
            f"{measure!r} is measured now -- {replacement().slug} -- so its refusal is a "
            "refusal that outlived its blocker. Remove it, and the reader stops being told "
            "the impossible is impossible."
        )
        assert replacement().slug not in registry, (
            "a measure whose slug the registry refuses would be replaced by the refusal "
            "rather than by its own figure"
        )

    blocked = {row.measure for row in _rows() if row.is_blocked}
    for measure in retired:
        assert measure not in blocked, (
            f"docs/seam.md still marks {measure!r} blocked while the catalogue has no entry "
            "for it. One of the two files has not caught up, and the direction that reads as "
            "a working refusal is the one that has not"
        )


def test_a_measure_the_census_alone_forecloses_is_still_refused_because_the_outcome_measure_does_not_describe_development() -> (
    None
):
    """``EB6FE5ZP`` is a refusal in its own right, not half of a retired one.

    The outcome measure landed with the merge fact, and it reports captured changes
    honestly -- over captured changes. So the entry that used to carry both facts was
    split rather than deleted: the census half outlives it, and a coverage figure
    without a census of uncaptured changes is a statement about a selected population
    wearing the name of a statement about development.
    """
    refusal = default_registry().get("coverage-of-development")

    assert refusal.unblocked_by == "EB6FE5ZP"
    assert refusal.missing_fact is not None
    assert "census" in refusal.missing_fact.lower()
    assert "outcome" in refusal.reason.lower(), (
        "the reader has to be able to see that the outcome measure landed and this is the "
        "part of the old refusal that is still true"
    )


def test_a_refusal_can_never_render_with_an_empty_unblocking_and_no_reason_because_that_pair_is_a_citation_to_nothing() -> (
    None
):
    """Both halves of the "what would unblock it" line are unconstructible.

    A blank reason and an empty ticket list together read as a formal reference
    to a formal entry that is not there, and a reader checking the tickets would
    find none and conclude the rest of the refusal was unverified too. So the
    constructor refuses each independently, and every registered refusal is
    checked for a non-empty line in both places.
    """
    with pytest.raises(ValueError, match="Refusal.reason must not be blank"):
        _refusal_with(reason="")
    with pytest.raises(ValueError, match="Refusal.reason is required"):
        # Deliberately past the annotation: a caller that builds a refusal from a
        # parsed document will do this, and the answer has to name the field.
        _refusal_with(reason=None)
    with pytest.raises(ValueError, match="Refusal.unblocked_by must not be blank"):
        Refusal(claim_slug="x", title="X", reason="A reason.", unblocked_by="  ")

    for refusal in default_registry().all():
        rendered = refusal.render_text()
        assert (
            "\n".join(line for line in rendered.splitlines() if line.startswith("Why: ")) != "Why: "
        )
        assert any(
            line.startswith("What would unblock it: ")
            and len(line) > len("What would unblock it: ")
            for line in rendered.splitlines()
        ), f"{refusal.claim_slug} rendered an empty unblocking line"


def test_a_duplicate_slug_is_rejected_because_two_refusals_for_one_measure_is_a_contradiction_not_an_update() -> (
    None
):
    """Registration refuses to shadow, so a second argument cannot quietly replace a first.

    The retrieval refusals rest on a two-part argument, and an overwrite would be
    how the second half got lost: the entry would still render, still be found by
    slug, and would no longer say the second thing.
    """
    registry = default_registry()
    existing = registry.get("coverage-of-development")
    with pytest.raises(ValueError, match="already registered"):
        registry.register(existing)
    assert len(registry) == len(registry.all())


def test_a_registry_holding_nothing_still_reports_its_mechanism_state_because_emptiness_is_an_answer() -> (
    None
):
    """A bare registry is a real configuration, and it is a refusing one.

    A test builds these constantly, and a registry that answered ``None`` for a
    membership test it did not run would make an empty registry indistinguishable
    from a full one that forgot to load.
    """
    empty = RefusalRegistry()
    assert len(empty) == 0
    assert empty.all() == ()
    assert "coverage-of-development" not in empty
    assert empty.mechanism is None
