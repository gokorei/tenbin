"""The report as Markdown: a page a person reads, with the caveats above the numbers.

This renderer is deliberately the boring one. It has no opinion about what a section
contains, in what order, or what a figure says about the read it was computed over --
:func:`~tenbin.report.ordering.ordered_parts` decides the first two and
:meth:`~tenbin.measures.base.Figure.rate_text` decides the third. All that is left is
a heading, a bullet, and the label the part's kind carries. That is what makes the
ordering a property of the program rather than a habit of the author: there is no
branch here that could put a number first, so there is nothing to review for it.

**The corpus header comes first, as a section, not as a preamble.** It gets a heading
of its own and the same bullet treatment as everything below, rather than being a
sentence under the title. That is not styling: a preamble is what a reader skips, and
the header is the part that decides whether any number below it means what the reader
would otherwise assume.

**A multi-line part is indented under its bullet rather than flattened.** A
distribution's payload is several lines, and the exclusions in it are load-bearing --
a bucket that did not occur and a bucket that was filtered out are the same integer in
``values`` and completely different facts. So the line breaks are kept and the
continuation lines are indented, which is what keeps a bucket list readable as a list
instead of arriving as one long line whose missing commas nobody can reconstruct.

**What this module notably does not do:** it does not decide order, does not choose
labels, does not number anything, and does not shorten any caveat except the one
whole-read sentence :data:`WHOLE_READ_SHORT` replaces. It holds no knowledge of what
a claim is or what a truncation means -- a reader who wanted a shorter report would get
it by changing :mod:`tenbin.report.ordering`, not by editing a template here. The
summary table (:data:`SUMMARY_HEADING`) is the one thing it adds above the parts:
one row per section, derived from the sections themselves, so a reader can find a
slug before reading the sections that follow.
"""

from __future__ import annotations

from collections.abc import Iterable

from tenbin.corpus.snapshot import Completeness
from tenbin.report.base import Section
from tenbin.report.document import Report
from tenbin.report.ordering import Part, PartKind, ordered_parts

#: The document title. Constant rather than derived from the collection because the
#: collection is the header's first line and repeating it in the title would be a
#: second copy of the one value a reader most needs to check.
REPORT_TITLE = "# Tenbin report"

#: The heading the corpus read is rendered under. Named because "Contents" would be a
#: lie about a section that contains no findings.
CORPUS_HEADING = "## Corpus"

#: The heading the per-section summary table is rendered under. A table rather
#: than bullets so the bullet-parsing helpers tests use keep comparing the full
#: sections unchanged, and so a reader skimming for one slug finds one row.
SUMMARY_HEADING = "## Summary"

#: What a figure over a whole read carries in place of the full completeness
#: sentence. The header already states the read is whole, and every section
#: repeating the paragraph is what teaches a reader to skip it; the short line
#: keeps the wholeness stated on the figure while pointing at the header that
#: says what "whole" was. Non-complete reads keep their full sentence verbatim,
#: because there the sentence is the news.
WHOLE_READ_SHORT = "Whole read — see Corpus for the read this covers."


def render_markdown(report: Report) -> str:
    """Render the whole report: the read, a summary table, then each section.

    A pure function of ``report`` -- no clock, no store, no formatting option -- so a
    report rendered twice is byte-identical and a rendered report can be checked
    against the value it was rendered from. The trailing newline is there because a
    Markdown file that does not end in one is a file every tool appends to.

    The summary is an entry point, not a replacement: one table row per section
    in report order, with the full sections unchanged below it.
    """
    lines = [REPORT_TITLE, "", CORPUS_HEADING, ""]
    lines.extend(_bullets(report.corpus.parts()))
    lines.extend(("", SUMMARY_HEADING, ""))
    lines.extend(_summary_table(report))
    for section in report.sections:
        lines.extend(("", f"## {section.heading}", ""))
        lines.extend(_section_bullets(section))
    return "\n".join(lines) + "\n"


def _summary_table(report: Report) -> list[str]:
    """One Markdown table row per section, in the order the sections are held.

    A table rather than bullets so the section bullets below stay the document
    of record. Columns are the slug (the identifier to cite), the section
    heading, the first line of the payload (or ``Refused``), and the read's own
    completeness word. The result cell never spans lines: multi-line payloads
    contribute their first line only, with pipes escaped.
    """
    lines = ["| Slug | Section | Result | Read |", "| --- | --- | --- | --- |"]
    for section in report.sections:
        lines.append(
            f"| `{section.slug}` "
            f"| {_cell(section.heading)} "
            f"| {_cell(_summary_result(section))} "
            f"| {_cell(_summary_read(section))} |"
        )
    return lines


def _summary_result(section: Section) -> str:
    """The one-line result a summary row carries for a section."""
    if section.figure is None:
        return "Refused"
    first = section.figure.value_text().splitlines()
    return first[0] if first else "(no payload)"


def _summary_read(section: Section) -> str:
    """The completeness word behind a section, or a dash for a refusal."""
    if section.figure is None:
        return "—"
    return section.figure.snapshot.completeness.value


def _cell(value: str) -> str:
    """One table cell: single line, with pipes escaped so the table survives them."""
    return value.replace("\n", " ").replace("|", "\\|")


def _section_bullets(section: Section) -> list[str]:
    """A section's parts as bullets, with a whole read's sentence shortened.

    The parts come from :func:`~tenbin.report.ordering.ordered_parts` and the
    order is untouched. Only the first line of the figure part changes, and
    only when the figure's own read is whole: the header already states the
    verdict, so the figure keeps a short pointer to it instead of the repeated
    paragraph. Any other read keeps its full sentence verbatim.
    """
    parts = ordered_parts(section)
    if section.figure is not None and section.figure.snapshot.completeness is Completeness.COMPLETE:
        shortened: list[Part] = []
        for part in parts:
            if part.kind is PartKind.figure:
                rest = part.text.splitlines()[1:]
                shortened.append(Part(part.kind, "\n".join([WHOLE_READ_SHORT, *rest])))
            else:
                shortened.append(part)
        parts = shortened
    return _bullets(parts)


def _bullets(parts: Iterable[Part]) -> list[str]:
    """Render parts as a Markdown list, one bullet each, continuation lines indented.

    Takes an iterable rather than a concrete sequence because it is called with the
    header's parts and with a section's parts, and the two are the same type by design
    -- that is the property which lets the header be held to the rule that sections are.
    """
    rendered: list[str] = []
    for part in parts:
        body = part.text.splitlines()
        rendered.append(f"- **{part.kind.label}:** {body[0]}")
        rendered.extend(f"  {line}" for line in body[1:])
    return rendered
