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
labels, does not summarise, and does not number anything. It holds no knowledge of what
a claim is or what a truncation means -- a reader who wanted a shorter report would get
it by changing :mod:`tenbin.report.ordering`, not by editing a template here, which is
the whole reason this file is short.
"""

from __future__ import annotations

from collections.abc import Iterable

from tenbin.report.document import Report
from tenbin.report.ordering import Part, ordered_parts

#: The document title. Constant rather than derived from the collection because the
#: collection is the header's first line and repeating it in the title would be a
#: second copy of the one value a reader most needs to check.
REPORT_TITLE = "# Tenbin report"

#: The heading the corpus read is rendered under. Named because "Contents" would be a
#: lie about a section that contains no findings.
CORPUS_HEADING = "## Corpus"


def render_markdown(report: Report) -> str:
    """Render the whole report: the read, then each section in the order it is held.

    A pure function of ``report`` -- no clock, no store, no formatting option -- so a
    report rendered twice is byte-identical and a rendered report can be checked
    against the value it was rendered from. The trailing newline is there because a
    Markdown file that does not end in one is a file every tool appends to.
    """
    lines = [REPORT_TITLE, "", CORPUS_HEADING, ""]
    lines.extend(_bullets(report.corpus.parts()))
    for section in report.sections:
        lines.extend(("", f"## {section.heading}", ""))
        lines.extend(_bullets(ordered_parts(section)))
    return "\n".join(lines) + "\n"


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
