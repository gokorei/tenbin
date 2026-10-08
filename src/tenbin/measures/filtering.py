"""Selecting the records a measure is allowed to count, as a named helper.

Two measures can be a boolean apart and produce figures that read identically.
One that filters to records whose kind was *determined* and one that filters to
every record, side by side, differ in their exclusion counts and nowhere else --
and the one that filtered has a smaller, more confident-looking distribution. So
the certainty a measure requires is stated on the measure
(:data:`requires_certainty` and its per-measure overrides) and applied by
:func:`select`, rather than being an argument each caller passes. A measure that
says what it required can be checked; a measure that was handed a filter cannot.

**Nothing is dropped silently.** :func:`select` returns only the records it was
asked for, so a measure that uses it still has to say what it left out -- in the
``excluded`` mapping of its :class:`~tenbin.measures.base.DistributionFigure`,
under a key naming the condition. There is no helper that turns a filter into a
count, deliberately: a function returning "how many were dropped" would be a
number with no condition attached, and a number with no condition attached is how
an exclusion becomes a bucket.

:func:`named_repository` is here for the same reason, one level down. It is the
other question a grouping has to answer honestly -- not *which kind of record* but
*which value is a name at all* -- and both of the answers this package had to stop
giving were a truthiness test on a string Kojutsu fills with a placeholder. It
is the only value-level rule the measures share, and the only one that was
discovered twice.

**What this module notably does not do:** it does not repair, impute, or decide
which reading of a record is right. A record whose kind is inferred is not a
record that failed to parse and there is nothing here to fix -- there is a real
distinction between "the writer said this is a review verdict" and "the tags
permit reading it as one", and a filter that treated the two as interchangeable
would be discarding the only evidence that a corpus is partly legacy.

**The capture population is an allowlist, and that is the module's newest rule.**
:data:`CAPTURE_KINDS` names every kind that counts towards a capture denominator, and
:func:`capture_population` applies it. The kind it excludes is
:attr:`~tenbin.corpus.record.RecordKind.QUESTION`: Kojutsu now projects terminal
decision requests into the store, and decision 001 requires them to be counted apart
from captures -- never inside one. Six measures iterate the whole read and had no way
to say so, and a projected request carries a repository and a document update time, so
without this it would have landed in the by-repository histogram as a record somebody
examined and in the by-month histogram filed under the day it was written rather than
the day it was asked. Whether a question counts as a capture is not a judgement left to
each measure; it is a fact about what a capture is, stated once.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Final

from tenbin.corpus.record import PLACEHOLDER, Certainty, Record, RecordKind

#: Every kind that counts towards a capture denominator, as an allowlist rather than
#: a deny-list for the same reason Kojutsu's projection is one. The failure mode of
#: a deny-list here is not a bad number, it is a good number about the wrong corpus: a
#: kind added to the store later is counted into every coverage distribution and every
#: capture figure until somebody remembers to exclude it, and nothing anywhere reports
#: the omission. An allowlist puts the safe behaviour first and the remembering second.
#:
#: :attr:`RecordKind.QUESTION` is deliberately absent, and its absence is the reason
#: this constant exists at all. A projected decision request is a request for a reason,
#: so it carries no ``capture_source`` and no ``independence`` -- and decision 001
#: requires it to be counted apart from captures, never inside them. A question
#: reaching the by-repository histogram would put a request into a population of
#: records somebody examined, and reaching the by-month one would file it under the
#: document's update time, which is when it was *projected* rather than when anything
#: was asked.
CAPTURE_KINDS: Final[tuple[RecordKind, ...]] = (
    RecordKind.ANSWER,
    RecordKind.REVIEW_VERDICT,
    RecordKind.INLINE_REVIEW_COMMENT,
    RecordKind.PR_LIFECYCLE,
    RecordKind.RATIONALE,
)

#: The word every capture denominator's description has to use, because the size beside
#: it changed the moment questions were excluded from it. A population described as
#: "records in this read" with the size of the captures is a denominator describing a
#: population nobody measured, which is the one thing a
#: :class:`~tenbin.claims.model.Denominator` exists to make impossible.
CAPTURE_POPULATION_WORD: Final[str] = "capture records"


def named_repository(repo: str | None) -> bool:
    """Whether this string is a repository, as opposed to Kojutsu's placeholder.

    The check is not a truthiness test. Kojutsu writes the literal ``"unknown"``
    where a repository was never stated, and a record with no repository is filed at
    ``unknown/pr-<n>/<entry>`` -- so ``Record.repo`` is the *string* ``"unknown"``
    rather than ``None``, and every measure that groups by repository would count
    the placeholder as one. That is the ``unknown``-model defect in a third costume,
    and it is the reason the rule is a named function rather than a habit.

    Mirrors :func:`tenbin.corpus.provenance._named_repository`, which the
    provenance layer applies to the same field. It is written out here because that
    one is private to a package this one does not own, and the string comes from the
    same public constant -- :data:`tenbin.corpus.record.PLACEHOLDER` -- so the
    literal cannot drift even though the rule is stated twice. That duplication is
    worth a shared public helper on the record model, which is a change to a package
    this ticket does not hold; until then this is the one definition inside
    :mod:`tenbin.measures`, and both callers here use it.
    """
    if repo is None:
        return False
    candidate = repo.strip()
    return bool(candidate) and candidate.casefold() != PLACEHOLDER


#: What a measure counts unless it says otherwise: every record it is handed,
#: whatever certainty the classification carries. Named rather than left as a
#: literal ``None`` at each call site so that a measure declaring
#: ``requires_certainty = Certainty.DETERMINED`` is visibly *opting in* to a
#: stricter population rather than relying on a default somebody might relax.
requires_certainty: Certainty | None = None


def select(
    records: Iterable[Record],
    *,
    certainty: Certainty | None = requires_certainty,
    kinds: Sequence[RecordKind] | None = None,
) -> tuple[Record, ...]:
    """The records matching the required certainty and, if given, the required kinds.

    Both conditions are ``and``: a caller that wanted records of any certainty
    but only of one kind writes ``certainty=None``, and a caller that wanted only
    determined records writes ``kinds=None``. The signature makes both the default
    and the override visible at the call site, which is the only place a reader can
    check that a measure said what it required.

    Order is preserved. Every figure built from the result is a pure function of
    it, and preserving order is what makes two snapshots holding the same records
    in a different order produce the same histogram counts -- the histogram sorts
    by count, so a count is order-free, but a *chain* built from these records is
    not, and a measure that walked them in set order would produce a different
    chain every run.
    """
    wanted_kinds = None if kinds is None else frozenset(kinds)
    return tuple(
        record
        for record in records
        if (certainty is None or record.classification.certainty is certainty)
        and (wanted_kinds is None or record.classification.kind in wanted_kinds)
    )


def capture_population(records: Iterable[Record]) -> tuple[Record, ...]:
    """The records a capture measure is allowed to count, and nothing else.

    Written as one function rather than left to each measure because the alternative
    is six measures independently deciding that a projected decision request is not a
    capture, and the sixth is the one somebody forgets. It takes the whole read rather
    than a kind filter at each call site for the same reason: the *denominator* has to
    exclude them too, and a measure that filtered its loop and not its denominator
    would have produced a histogram of exactly the right numbers over a population that
    no longer existed.

    Order is preserved, for the reason :func:`select` gives.
    """
    return tuple(record for record in records if record.classification.kind in CAPTURE_KINDS)
