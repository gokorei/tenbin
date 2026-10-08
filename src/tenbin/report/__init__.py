"""Report layer: a corpus header, sections, and the one order a section is read in.

Everything upstream of this package produces values that are individually correct: a
snapshot that says how far the walk got, a figure that carries the read it was computed
over, a claim that names its four caveats, a refusal that says what is missing. None of
them is a report, and the gap between them is where a measurement program usually loses
its discipline. A report is the one artifact a person reads, and a reader reads it in
whatever order it happens to be in -- caveats first or last is a decision somebody
makes in a hurry, on a Friday, for one figure. So the decision was taken out of the
hands of the person making it: :func:`~tenbin.report.ordering.ordered_parts` returns
one section's content in a fixed sequence, both renderers walk that sequence, and
neither of them has a branch that could put a number first.

**The types make the broken report unconstructible rather than merely discouraged.** A
:class:`~tenbin.report.base.Section` must hold a figure or a refusal, so a blank
heading -- which renders as an absence of findings and is in fact a measure that never
ran -- fails at construction. A figure must bring its claim, so the number and the
statement of what it means cannot be separated. A
:class:`~tenbin.report.document.CorpusHeader` is copied from the snapshot rather than
written out, so the counts describing the read cannot disagree with the read. These are
the same argument :mod:`tenbin.claims` makes about claims, applied to the document: a
convention survives until the deadline, and a type does not.

**A refusal is content, so it renders as a section.** :func:`build_report` runs every
measure, gates every claim, and puts a refusal in the place a figure would have been --
the reason, the missing fact, and the ticket that would deliver it or the explicit
statement that nothing will. A report that refused to build over one unsupported claim
would be a program that can describe this corpus only when asked the questions it can
answer, and the questions it cannot answer are the ones somebody came for.

**What this package notably does not do:** it does not measure, and it holds no opinion
about a record. It opens no socket, reads no clock and writes no file -- a report is a
pure function of the values handed to it, which is what lets a rendered document be
checked against the report it was rendered from. It does not decide what to say, which
is the claims package's and the gate's; it does not decide how a claim reads, which is
:mod:`tenbin.report.ordering`'s alone; and it does not decide what a section contains
or what order the sections are in, which is :mod:`tenbin.report.build`'s. Each of
those three is one function or one type rather than a habit, and that is the whole
design.
"""

from tenbin.report.base import Section
from tenbin.report.build import build_report
from tenbin.report.document import CorpusHeader, Report
from tenbin.report.html_ import render_html
from tenbin.report.json_ import render_json, report_as_dict
from tenbin.report.markdown import render_markdown
from tenbin.report.ordering import Part, PartKind, ordered_parts

__all__ = [
    "CorpusHeader",
    "Part",
    "PartKind",
    "Report",
    "Section",
    "build_report",
    "ordered_parts",
    "render_html",
    "render_json",
    "render_markdown",
    "report_as_dict",
]
