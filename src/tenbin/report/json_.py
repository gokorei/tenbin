"""The report as JSON, for the consumer this program is most worried about handing a
number to.

The Markdown renderer is read by somebody who can be shown a caveat. This one is read
by a script, and a script given ``{"figure": "412 of 500 = 82.4%"}`` will put that
string on a wallboard and drop the rest -- not out of malice but because the rest is
not what the field was called. So the JSON is not the Markdown with the formatting
taken out: it carries the same four claim fields, in the same order, as named keys, so
that the cheapest thing a consumer can do with the payload is the responsible thing.
That is the real reason this program has two renderers. One renderer with two output
formats would have been one discipline to keep; the JSON is where the discipline is
easiest to lose and therefore the one that most needs it enforced.

**Key order is the reading order, and it is not sorted.** ``json.dumps`` preserves
insertion order, and the fields are inserted in the order
:func:`~tenbin.report.ordering.ordered_parts` returns them, so a consumer that walks
the object in order meets the statement, the limit and the falsifier before it meets
the figure. Passing ``sort_keys=True`` would make the payload tidier and would undo
the entire argument, which is why it is not an option.

**The corpus header is carried as values, not as prose.** The Markdown needs a
sentence a reader can read; a machine does not -- it needs ``enumerated`` and
``store_total`` as numbers and ``completeness`` as the word the snapshot used, so that
it can compare two reports rather than diff two paragraphs. Both renderings are built
from the same :class:`~tenbin.report.document.CorpusHeader`, so the two cannot
disagree about the read.

**Two fields are typed rather than rendered, because their absence is the information.**
``unblocked_by`` is ``null`` for a refusal that no ticket unblocks and a string for one
that a ticket would, and the difference between ``null`` and a field somebody forgot is
the difference between a decision and a bug. ``missing_fact`` is always present for a
refusal, ``null`` when there is none, for the same reason. Everything else is the part
text verbatim, identical to the Markdown, because a second machine-shaped rendering of
a denominator would be one more thing to keep true and would add nothing a consumer can
do that the text does not already say.

**What this module notably does not do:** it does not decide what a section contains or
in what order -- it takes both from :mod:`tenbin.report.ordering` like every other
consumer. It does not summarise, does not round, does not convert an enum into an
inventoried vocabulary (the values rendered are the ones the snapshot and the claim
already use, so a consumer matches on the words this program uses everywhere else), and
does not emit anything the Markdown does not also say. In particular it never emits a
section with a figure but no caveats: there is no code path that could produce one,
because a section cannot be constructed in that state.
"""

from __future__ import annotations

import json
from typing import Any

from tenbin.report.base import Section
from tenbin.report.document import CorpusHeader, Report
from tenbin.report.ordering import ordered_parts

#: Indentation for the rendered payload. A parameter because a machine consumer does
#: not care and a human debugging a payload does, and two is the value at which a
#: nested field is on its own line rather than four of them sharing one.
DEFAULT_INDENT = 2


def report_as_dict(report: Report) -> dict[str, Any]:
    """The report as nested dictionaries and lists, keys inserted in reading order.

    Returned as a value rather than only as serialised text so that a caller embedding
    a report does not have to parse one back out again, and so that a test can assert
    on the key order without going through a decoder that might reorder it for its own
    reasons.
    """
    return {
        "corpus": _header_as_dict(report.corpus),
        "sections": [_section_as_dict(section) for section in report.sections],
    }


def render_json(report: Report, *, indent: int = DEFAULT_INDENT) -> str:
    """Serialise the report, keeping the order the fields were inserted in.

    A pure function of ``report``: no clock, no store, no filename, and nothing read
    from the environment. Two renders of one report are byte-identical, which is what
    makes a rendered payload diffable and therefore reviewable.
    """
    return json.dumps(report_as_dict(report), indent=indent, sort_keys=False)


def _header_as_dict(header: CorpusHeader) -> dict[str, Any]:
    """The read, as values a machine can compare, with the truncation or ``null``.

    The completeness value is the snapshot's own word rather than a boolean, because
    the four outcomes are four different findings and a boolean would collapse a
    truncated read and a failed walk into one indistinguishable "not complete".

    **``period`` is emitted, and last, for the reason the markdown header emits it.**
    A period-scoped read's ``window`` is derived from the transitions it happened to
    contain, so a quiet period's window is narrower than the period it was cut to, and
    two reads side by side would appear to describe spans of unequal length. The
    declared period rides beside the observed window here for the same reason it does
    in :class:`~tenbin.report.document.CorpusHeader`: without it a JSON reader has a
    span and no way to know it was not the whole of what was asked for. Dropping it
    left ``--format json`` describing an undeclared window, which is the one thing
    ``--source ticket`` refuses to do at the argument parser.
    """
    truncation = header.truncation
    return {
        "collection": header.collection,
        "read_at": header.read_at.isoformat(),
        "enumerated": header.enumerated,
        "store_total": header.store_total,
        "window": None
        if header.window is None
        else {
            "earliest": None
            if header.window.earliest is None
            else header.window.earliest.isoformat(),
            "latest": None if header.window.latest is None else header.window.latest.isoformat(),
            "unreadable_timestamps": header.window.unreadable_timestamps,
        },
        "completeness": header.completeness.value,
        "period": None
        if header.period is None
        else {
            "index": header.period.index,
            "start": header.period.start.isoformat(),
            "end": header.period.end.isoformat(),
        },
        "truncation": None
        if truncation is None
        else {
            "offset_reached": truncation.offset_reached,
            "records_missing": truncation.records_missing,
            "boundary_repositories": dict(truncation.boundary_repositories),
            "boundary_months": dict(truncation.boundary_months),
        },
    }


def _section_as_dict(section: Section) -> dict[str, Any]:
    """One section: what identifies it, its parts in order, then its metadata.

    Identity first because a consumer indexing the array needs the slug before it has
    anything else; the parts next, in reading order; the metadata last, because the
    claim's kind and granularity qualify the parts rather than introduce them.
    """
    body: dict[str, Any] = {
        "slug": section.slug,
        "kind": "claim" if section.figure is not None else "refusal",
        "title": section.heading,
    }
    for part in ordered_parts(section):
        body[part.kind.value] = part.text
    body.update(_section_metadata(section))
    return body


def _section_metadata(section: Section) -> dict[str, Any]:
    """The fields that are not part text, because their shape is the information.

    A consumer can act on ``unblocked_by`` being ``null``; it can only read the
    sentence above it. A consumer can filter on ``claim_kind``; the statement does not
    parse. So these are typed and the parts stay prose, and both are present -- a
    payload with either alone would be worse than no payload, because the consumer
    would have to guess which one the producer meant.
    """
    if section.refusal is not None:
        return {
            # ``missing_fact`` is present even when the part is absent from the ordered
            # parts, so that a consumer can tell "there is no missing fact" from
            # "nobody recorded whether there is one".
            "missing_fact": section.refusal.missing_fact,
            "unblocked_by": section.refusal.unblocked_by,
        }
    claim = section.claim
    if claim is None:
        # Unreachable: ``Section`` refuses to be built without one of the two. Present
        # so that a future relaxation of that rule fails here as a missing field rather
        # than as a ``None`` reaching a consumer's payload.
        raise ValueError("a Section must hold a claim or a refusal; see Section.__post_init__")
    return {
        "claim_kind": claim.kind.value,
        "granularity": claim.granularity.value,
        "source": claim.source,
    }
