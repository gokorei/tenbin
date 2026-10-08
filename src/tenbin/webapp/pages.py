"""The pages, as pure functions of the outcomes they are handed.

Two functions make HTML: :func:`index_page`, which lists every configured store once,
and :func:`report_page`, which renders one store's own page. **Neither opens a socket,
reads a clock, touches the environment, or knows what an ``http.server`` is** -- that is
the whole reason they are testable without binding a port, and it is why
:mod:`tenbin.webapp.server` is a routing shell and nothing else.

**Every configured store appears on the index exactly once, as one of two things.** A
store that was read contributes its read, its completeness and its window. A store that
was not contributes a named failure: which store, what happened, and what would fix it.
There is no third state and no path that renders one -- :func:`index_page` takes
:class:`~tenbin.webapp.read.StoreOutcome` values, and that type holds a report or a
failure -- so a failed store cannot appear as a store with zero figures, cannot be
silently omitted, and cannot be counted among the stores read. The count line above the
list says how many were read *and* how many failed, because "4 of 4" and "4 of 5, and the
fifth could not be reached" are different reports and only one of them is true.

**The three fields on an index entry are the read, the completeness and the window,
rather than a count of figures.** A figure count answers "how much is in here" and
nothing about whether to believe it; a read over thirteen documents that is complete and
spans half a second says that the collection is fresh, small and wholly read, which is
everything a reader needs before deciding whether to click. So the index carries the
three fields the corpus header exists to state, taken from that header rather than
recomputed, and the figures stay on the page they belong to.

**No page references an external asset.** There is no stylesheet link, no font, no
script and no image: the CSS is inlined in the one document that is served, so a page
loads identically from a laptop, from a locked-down network, and from a browser that has
never heard of this host -- and an operator who serves this on an internal network does
not leak the fact that somebody loaded it to a CDN. It also means there is no second
request to fail, which for a page whose whole job is to say whether a store is up would
be an unfortunate way to fail.

**An API key cannot reach a page.** Not through a template, not through a URL, and not
through an exception message: every piece of text a page renders goes through
:meth:`~tenbin.webapp.config.StoreEntry.redact` and :func:`html.escape` on the way out,
in one place, so the guarantee does not depend on each renderer remembering. The store's
*name* is what appears -- it is the operator's own label for the store, whereas a URL can
carry a host that is itself the sensitive part.
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, TypeAlias
from urllib.parse import quote

from tenbin.corpus.window import CorpusWindow
from tenbin.report.document import CorpusHeader, Report
from tenbin.report.ordering import PartKind
from tenbin.webapp.config import StoreEntry
from tenbin.webapp.read import (
    FAILED_HEADLINE,
    StoreFailure,
    StoreListing,
    StoreListingOutcome,
    StoreOutcome,
)

#: The document title of the index. A constant, and not derived from the configuration,
#: because it is the answer to "which of the tabs am I looking at".
INDEX_TITLE: Final[str] = "Tenbin"

#: The inline stylesheet. One rule set for both pages, inlined rather than linked, for
#: the reason in the module docstring: a page that has to fetch its own appearance is a
#: page that can render blank. The rules are here rather than in a ``.css`` file for the
#: same reason -- a file would be a second thing to deploy, and this app is meant to be
#: runnable by copying a configuration file at it.
STYLE: Final[str] = """
:root { color-scheme: light dark; }
body {
  margin: 0 auto; max-width: 62rem; padding: 2rem 1.25rem 4rem;
  font: 16px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
}
h1 { font-size: 1.6rem; margin: 0 0 .25rem; }
h2 { font-size: 1.1rem; margin: 0 0 .5rem; }
p, li, dt, dd { max-width: 60rem; }
a { color: inherit; }
.counts { font-weight: 600; margin: 0 0 1.5rem; }
.stores { list-style: none; margin: 0; padding: 0; }
.store {
  border: 1px solid rgba(128,128,128,.45); border-left-width: 4px;
  border-radius: 4px; padding: 1rem 1.25rem; margin: 0 0 1.25rem;
}
.store.failed { border-left-color: #b3261e; background: rgba(179,38,30,.06); }
.store.failed h2 { color: #b3261e; }
.summary { display: grid; grid-template-columns: max-content 1fr; gap: .2rem .9rem; margin: 0; }
.summary dt { font-weight: 600; }
.summary dd { margin: 0; }
.failure-cause, .failure-remedy { margin: .4rem 0 0; }
.failure-cause { font-weight: 600; }
.failure-remedy { font-weight: 400; }
.none-figures { margin: .6rem 0 0; font-style: italic; }
.truncation { color: #8a4b00; font-weight: 600; }
.report {
  display: block; width: 100%; height: 80vh; margin-top: 1.5rem;
  border: 1px solid rgba(128,128,128,.45); border-radius: 4px; background: #fff;
}
footer { margin-top: 2.5rem; font-size: .875rem; }
@media (prefers-color-scheme: dark) { .failure-cause, .store.failed h2 { color: #f2b8b5; } }
""".strip()


#: The window a header states when it has none. The same empty window
#: :meth:`tenbin.report.document.CorpusHeader._enumeration_text` builds, so that a
#: header with no window renders the unknown-window sentence here rather than an absent
#: line -- an index entry with a blank window reads like one nobody asked about.
UNKNOWN_WINDOW: Final[CorpusWindow] = CorpusWindow(
    earliest=None, latest=None, unreadable_timestamps=0
)


@dataclass(frozen=True)
class StoreSummary:
    """What an index entry says about a store: three sentences, and a fourth if needed.

    ``read``, ``completeness`` and ``window`` are the three fields, in that order, because
    they answer three questions in that order -- how much and when, how far the walk got,
    and over what span -- and a reader deciding whether to click needs all three before
    any figure. ``truncation`` is ``None`` on a whole read and carries the sentence on a
    truncated one: a summary that reported the count of a truncated read without the
    truncation would be the same defect as a summary of a failed store, one level down.

    Nothing here is a number this program computed. Every value is the read's own, taken
    from the header the report already renders -- see :func:`summarize`.
    """

    read: str
    completeness: str
    window: str
    truncation: str | None = None

    @property
    def is_whole(self) -> bool:
        """Whether this read needs its truncation sentence rendered."""
        return self.truncation is None


def summarize(header: CorpusHeader) -> StoreSummary:
    """The three fields a reader needs before clicking, taken from the read itself.

    **Copied from the header, and never recomputed.** The completeness sentence is
    read out of :meth:`~tenbin.report.document.CorpusHeader.parts` by part kind rather
    than written here, because the plain-language reading of each verdict belongs to the
    report layer and a second copy of it would be a second thing to keep true -- the
    defect this project keeps meeting in its own registries, in a new costume. The counts
    and the window are read from the header's fields for the same reason
    :meth:`~tenbin.corpus.snapshot.CorpusSnapshot.window` is derived rather than passed:
    a summary that recomputed them could disagree with the report rendered underneath it,
    and there would be no test telling anybody.
    """
    parts = {part.kind: part.text for part in header.parts()}
    window = header.window if header.window is not None else UNKNOWN_WINDOW
    span = window.render_text()
    age = window.age_text(header.read_at)
    return StoreSummary(
        read=(
            f"{header.enumerated:,} of {header.store_total:,} documents, "
            f"read at {header.read_at.isoformat()}"
        ),
        completeness=parts[PartKind.completeness],
        window=f"{span} {age}" if age else span,
        truncation=parts.get(PartKind.truncation),
    )


#: What an index row is. **Narrower than :data:`~tenbin.webapp.read.StoreOutcome` on
#: purpose.** An index that also accepted reports could be handed one, and would then have
#: to describe itself as having read those stores while the route that calls it reads
#: nothing — or, the other way round, the counts sentence would have to carry a branch for
#: which kind of row it is holding, and two sentences for one page is two sentences that
#: can drift apart. So the type makes the claim structural: this page can only ever hold
#: counts and failures, and therefore can only ever say it counted.
IndexOutcome: TypeAlias = StoreListingOutcome


def index_page(stores: Sequence[IndexOutcome]) -> str:
    """Every configured store, once each, with a failure where a store could not be read.

    ``stores`` is one listing outcome per configured store in configuration order --
    what :func:`tenbin.webapp.read.read_listings` returns -- and this function renders all
    of them rather than the ones that worked. It cannot do otherwise: it holds outcomes, an
    outcome is a listing or a failure, and there is no code path that skips one. That is the
    shape the rule about unreachable stores needs in a list, and it is enforced by the type
    of the argument rather than by a condition somebody has to remember to write.

    **A row carries a count and no report, and that is a deliberate loss.** It used to carry
    a :class:`~tenbin.webapp.read.StoreReport`'s summary, because the index read every
    configured store in full -- one corpus walk and 22 measures per store, serially, to
    render a list of names. The summary now lives only on the store's own page, beside the
    figures it bounds. That is the honest trade rather than a summary rendered away from
    its caveat; it is recorded in :data:`IndexOutcome` and here because the next reader will
    otherwise take the missing block for an oversight.

    Pure: same outcomes in, same bytes out. No clock, so the page does not drift between
    two renders of one set of outcomes, which is what lets it be asserted on.
    """
    failed_count = sum(1 for store in stores if isinstance(store, StoreFailure))
    listed_count = len(stores) - failed_count
    entries = [_store_entry(store) for store in stores]
    body = "\n".join(
        (
            "<h1>Tenbin</h1>",
            f'<p class="counts">{_index_counts_sentence(len(stores), listed_count, failed_count)}</p>',
            _empty_note() if not entries else f'<ul class="stores">\n{_join(entries)}\n</ul>',
            _footer(failed_count),
        )
    )
    return _document(INDEX_TITLE, body)


def report_page(store: StoreOutcome) -> str:
    """One store's own page: its name, its read, and its report.

    A failure renders here the same way it renders on the index, because the two pages
    answer the same question about the same store and a store that is down must not look
    different depending on which URL was loaded -- a per-store page that showed an empty
    report where the index showed a failure would be the failure being reintroduced
    through a second door.

    Pure, like the index. The report body is delegated to the report layer's own HTML
    renderer rather than rendered here, because :mod:`tenbin.report` owns what a report
    looks like and this module owns where it is put.
    """
    entry = store.entry
    if isinstance(store, StoreFailure):
        body = "\n".join(
            (
                f"<h1>{_text(entry, entry.name)}</h1>",
                _failure_block(entry, store),
                '<p><a href="/">All stores</a></p>',
            )
        )
        return _document(f"{entry.name} — {FAILED_HEADLINE}", body)

    summary = summarize(store.header)
    body = "\n".join(
        (
            f"<h1>{_text(entry, entry.name)}</h1>",
            _summary_block(entry, summary),
            '<p><a href="/">All stores</a></p>',
            _report_body(entry, store.report),
        )
    )
    return _document(entry.name, body, extra_style=report_stylesheet())


def render_report_body(report: Report) -> str:
    """One report as HTML, from the report layer's renderer.

    **Resolved at call time rather than at import time, on purpose.** The renderer lives
    in :mod:`tenbin.report.html_`, and importing it at module scope would make this whole
    package -- and every page, including the ones that never render a report -- fail to
    import on a tree where it does not exist. Resolving inside the function means the
    real renderer is called the moment it is there, with no edit here when it arrives.

    A test may substitute for it by replacing this name, which is how
    ``tests/test_tenbin_index.py`` runs the pages against a corpus while
    :mod:`tenbin.report.html_` is still being written. **The production path is not
    substitutable in practice**: nothing in this package assigns to it, and the only
    caller is :func:`_report_body`, which resolves it from this module's globals on every
    call.
    """
    from tenbin.report.html_ import render_html_body

    return render_html_body(report)


def report_stylesheet() -> str:
    """The rules the body above depends on, resolved late for the same reason.

    **Exposed beside the body because the body is not self-sufficient.** Every class
    :func:`render_report_body` emits is styled by a rule here, so inlining one without
    the other produces a page of valid HTML that looks like plain text -- and a reader
    would see every caveat and miss that the bars are not there.
    """
    from tenbin.report.html_ import report_stylesheet as real

    return real()


def _store_entry(store: IndexOutcome) -> str:
    """One list row: the store's name and size, or the failure that stopped it.

    Two shapes, because a row is one of two things. A store that counted links to its own
    page, because that is where its read and its report are. **A store that failed does not
    link**, because there is no report behind the link and a link labelled as one would be a
    promise the page cannot keep. It is in the list, named, with the cause -- which is the
    only thing a reader needs in order to go and fix it.
    """
    entry = store.entry
    name = _text(entry, entry.name)
    if isinstance(store, StoreFailure):
        return "\n".join(
            (
                '<li class="store failed">',
                f"<h2>{name}</h2>",
                _failure_block(entry, store),
                "</li>",
            )
        )
    return "\n".join(
        (
            '<li class="store">',
            f'<h2><a href="/store/{_href_name(entry, entry.name)}">{name}</a></h2>',
            _listing_block(store),
            "</li>",
        )
    )


def _listing_block(listing: StoreListing) -> str:
    """What the index says about a store it counted but did not read.

    **The count is the store's own figure, and the sentence says which one it is.** A bare
    number here would be a figure without a claim attached, which is the thing this whole
    programme refuses to emit -- worse than useless on an index, because an index is the
    page people read quickly and quote from. So the wording states that the store reports
    this size, that the index did not read the documents to confirm it, and that the store's
    own page is where the read happened.
    """
    noun = "document" if listing.count == 1 else "documents"
    return (
        '<dl class="listing">'
        f"<dt>Documents</dt><dd>The store reports {listing.count} {noun}. This page did "
        "not read them; open the store for what was read and what it means.</dd>"
        "</dl>"
    )


def _index_counts_sentence(total: int, listed: int, failed: int) -> str:
    """The line above the list, which says what the index actually did.

    **Not "were read".** The index no longer walks a store, so a sentence claiming a read
    happened would be false on every successful row -- the exact defect this project keeps
    finding in its own reports, where a number is stated about a read that did not occur.
    A store that answered is counted; a store that did not is listed with its cause; and
    neither is described as having been read.
    """
    if total == 0:
        return (
            "No stores are configured, so this page has nothing to report. That is a "
            "configuration problem and not a finding: an operator expecting stores to be "
            "listed is looking at an empty page that would otherwise read as every store "
            "having nothing to say."
        )
    if failed == 0:
        return (
            f"All {total} configured stores answered a size request. Their contents were "
            "not read to render this list."
        )
    noun = "store" if failed == 1 else "stores"
    return (
        f"{listed} of {total} configured stores answered a size request. "
        f"{failed} {noun} could not be reached"
        + (", and it is" if failed == 1 else ", and they are")
        + " listed below with the cause and what would fix it."
    )


def _summary_block(entry: StoreEntry, summary: StoreSummary) -> str:
    """The read, the completeness and the window, as a definition list.

    Shared by both pages, which is the point of the type: an index entry and a store's own
    page state the same three fields in the same words, so the two cannot disagree about
    what was read -- a summary on the index that did not match the report behind the link
    would be a page making a claim it does not itself hold.
    """
    rows = [
        ("Read", summary.read),
        ("Completeness", summary.completeness),
        ("Window", summary.window),
    ]
    if summary.truncation is not None:
        rows.append(("Not reached", summary.truncation))
    # Interleaved rather than all terms then all definitions, because that is the order a
    # grid places them into columns in -- a list whose labels all arrive first renders as a
    # column of labels beside a column of values with no way to tell which is which. It is
    # also the order a screen reader meets them in, so it is the same fix for both readers.
    pairs = "".join(f"<dt>{label}</dt><dd>{_text(entry, value)}</dd>" for label, value in rows)
    return f'<dl class="summary">{pairs}</dl>'


def _failure_block(entry: StoreEntry, failure: StoreFailure) -> str:
    """A named failure: the headline, the cause, the remedy, and the absence of figures.

    The last line is the load-bearing one. A reader who has skimmed this list sees "could
    not be read", scrolls on, and half an hour later quotes a figure from the page above
    as if the whole page had produced it -- so the entry says in its own words that it
    contributes nothing, which is the sentence a failure in a list owes its reader.
    """
    return "\n".join(
        (
            f'<p class="failure-cause">{FAILED_HEADLINE}. {_text(entry, failure.cause)}.</p>',
            f'<p class="failure-remedy">What would fix it: {_text(entry, failure.remedy)}</p>',
            '<p class="none-figures">This store is not counted among the stores read, '
            "and it contributes no figures to this page.</p>",
        )
    )


def _counts_sentence(total: int, read: int, failed: int) -> str:
    """The line above the list, which says both numbers or it does not say enough.

    A page that reported only the successes would make "4 of 4" and "4 of 5" the same
    sentence, and those are different reports: the second one is a deployment with a
    store nobody can read. So the failed count is stated whenever it is non-zero, in
    words rather than as arithmetic a reader has to perform.
    """
    if total == 0:
        return (
            "No stores are configured, so this page has nothing to report. That is a "
            "configuration problem and not a finding: an operator expecting stores to be "
            "listed is looking at an empty page that would otherwise read as every store "
            "having nothing to say."
        )
    if failed == 0:
        return f"All {total} configured stores were read."
    noun = "store" if failed == 1 else "stores"
    return (
        f"{read} of {total} configured stores were read. "
        f"{failed} {noun} could not be reached"
        + (", and it is" if failed == 1 else ", and they are")
        + " listed below with the cause and what would fix it."
    )


def _empty_note() -> str:
    """A stand-in for an empty list, so the page is a sentence rather than a gap."""
    return (
        '<p class="counts">Nothing is listed below, because no store was configured. '
        "An empty list here is not a list of stores with nothing to say.</p>"
    )


def _footer(failed: int) -> str:
    """The line saying what this page does with a store it could not reach.

    **The wording depends on whether anything actually failed**, and the reason is
    that the unconditional version was a sentence making a claim about the page. "A
    store that could not be reached is listed above as a failure" is true exactly
    when there is one, and on a page where every store was read it tells the reader
    a failure is listed above when the list above them is empty -- which is the same
    defect as the missing entry, one layer down and wearing the voice of a policy
    statement rather than a fact.

    So on a clean page it states the policy in the conditional, which is what a
    policy statement should do: it says what *would* happen, not what did. The
    claim about what happened is the counts sentence's job, and it already carries
    both numbers.
    """
    if failed:
        return (
            "<footer>A store that could not be reached is listed above as a failure. "
            "It is never rendered as a store with no figures, and it is never left out of "
            "this list.</footer>"
        )
    return (
        "<footer>Every store above was read. A store that could not be reached would be "
        "listed here as a failure rather than rendered as a store with no figures, and "
        "would not be left out of this list.</footer>"
    )


def _report_body(entry: StoreEntry, report: Report) -> str:
    """The report's own HTML, carried as a document and scrubbed of this store's key.

    **Scrubbed**, because it is still bytes this page serves and the guarantee about
    credentials is about the whole response rather than about the parts of it somebody
    remembered. **Not escaped as markup**, because it is a document produced by this
    program's own renderer -- which is why it is a distinct function from :func:`_text`
    rather than a flag on it: a function that can emit unescaped text is one whose
    callers have to know which kind they are holding.

    **Spliced, not framed.** This used to hand the report to a ``srcdoc`` ``<iframe>``,
    which had two problems that only showed up once the report was real. It nested a
    document inside a page, so the report's markup ended up escaped into an attribute:
    a 204 KB page whose charts no tool outside a browser could see, and one whose
    nested frame brought its own scrollbar. Fixing the escaping rather than the
    nesting would have kept the second problem, and the fix is the smaller one -- the
    report layer already has a body and a stylesheet, and both are meant to be
    embedded.

    So the body goes in directly and the stylesheet is inlined once in this page's
    own head. One stylesheet, no nested document, no escaped markup: the response is
    the report. **Still no external reference of any kind**, so the page still opens
    with the network off.
    """
    return entry.redact(render_report_body(report))


def _text(entry: StoreEntry, value: str) -> str:
    """A piece of text on a page: this store's credential removed, then escaped.

    The single door for every rendered string, in that order. Redacting after escaping
    would be wrong for two reasons at once -- the marker would be escaped, and a key
    containing ``&`` would have been escaped into something that no longer matched the
    key -- and escaping after redacting means the key is matched against the same bytes
    the configuration holds.
    """
    return html.escape(entry.redact(value), quote=True)


def _href_name(entry: StoreEntry, name: str) -> str:
    """A store name as one URL path segment, percent-encoded.

    Percent-encoded even though :class:`~tenbin.webapp.config.StoreEntry` already
    refuses whitespace, slashes and control characters in a name, because the guarantee
    belongs at the boundary rather than in the thing being encoded: a name that reached
    the encoder from somewhere else would be escaped here, and this function has no way to
    check where its argument came from.
    """
    return quote(entry.redact(name), safe="")


def _document(title: str, body: str, extra_style: str = "") -> str:
    """Wrap rendered content in the one HTML document both pages share.

    ``extra_style`` is carried inline and in the same ``<style>`` element rather than
    in a second one, so the report's rules are cascade-ordered against this page's by
    source order and not by whichever element the browser met last. **It is a
    parameter and not a global** because only the report page needs it: the index
    renders no charts, so paying to resolve the report layer's stylesheet on every
    index request would couple a page that has no report to a module it never uses.
    """
    style = STYLE + extra_style
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{html.escape(title)}</title>\n"
        f"<style>\n{style}\n</style>\n"
        "</head>\n"
        "<body>\n"
        f"{body}\n"
        "</body>\n"
        "</html>\n"
    )


def _join(parts: Sequence[str]) -> str:
    """Join rendered blocks with a blank line between them."""
    return "\n".join(parts)
