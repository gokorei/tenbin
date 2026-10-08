"""The read seam's own vocabulary: what a stored document *is*, read leniently.

Kojutsu decides what to capture and writes it. This package reads what was
written and reports two things about it: what kind of record it is, and how far
its claim to be evidence can be taken. It notably does **not** judge whether a
record is good, correct, or worth having -- that argument belongs to
:mod:`tenbin.claims`, and a corpus layer that started making it would be
measuring its own preferences. It also does not repair anything. A field that
arrived malformed is recorded as malformed, because a reader that quietly
repaired a corpus would report a self-description of the repair rather than of
the corpus, and a measurement program that flatters its input is not a
measurement program.

**Everything here is lenient and nothing here is lossy.** The store stringifies
every frontmatter extra it does not type, so a document read back carries
``"42"`` where the writer held ``42`` and ``"2026-09-28T00:00:00+00:00"`` where it
held a ``datetime``. A strict validator would therefore flag the entire corpus,
and a validator that reports everything reports nothing. So numbers are accepted
in both spellings, timestamps in both spellings, and a field that cannot be read
yields ``None`` rather than an exception: a record with one bad field is still a
record, it is countable, and the gap belongs in
:mod:`tenbin.corpus.provenance` where it is named rather than swallowed.

The one thing this package is *not* lenient about is a kind. Kojutsu now
writes ``record_kind`` on every record it captures and derives the tag set from
it, so the kind is read rather than guessed; a document written before the key
existed -- and every rationale, which is dispatched through a different payload --
has no kind and is classified from its tags, which is a weaker statement about
the same record and is labelled :data:`~tenbin.corpus.record.Certainty.INFERRED`
where it rests on an absence. See :func:`tenbin.corpus.record.classify`.
"""

from __future__ import annotations

__all__ = ("provenance", "record", "snapshot", "values", "window")
