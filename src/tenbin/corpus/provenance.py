"""Counting how far each record's claim to be evidence can actually be taken.

Kojutsu writes a ``capture_source`` on every record precisely so a reader can
tell a capture from a claim, and re-validates the anchors on the read path
because the store is a separate service that never ran the constructor check. This
module is Tenbin's side of that check: for each record it names the ways in
which that claim is not supported, as a pure function of the record, and then
counts the results.

**The anomalies are counted, never filtered out.** A record missing its delivery
id is still a record in the corpus, and dropping it would make every rate
computed here a rate over a population nobody had described -- which is the
failure this project exists to report, committed by the reporter. So
:func:`provenance_profile` reports a total, a clean count, and where the
anomalous ones are, and a caller who wants a rate over clean records only has to
say so in the caveat, where it belongs.

**Numbers are read leniently, and the reason is the storage boundary.** Kojutsu
stringifies every frontmatter extra it does not type, so a document read back from
the store carries ``"42"`` where the writer held ``42``. A strict numeric check
would therefore flag every pull request in the corpus, and a validator that
reports everything reports nothing -- worse than no validator, because its output
looks like a finding. ``42`` and ``"42"`` are both accepted; anything else is
absent.

**A present-but-blank field counts as missing**, for ``capture_source``,
``independence`` and ``delivery_id`` specifically. Kojutsu's writer skips empty
values, so a blank key means somebody wrote a document by hand and left the
placeholder in -- which is exactly the record where a validator earns its keep. A
blank *repository* is treated the same way, and so is the literal ``"unknown"``
that stands in for one, because a bucket labelled ``unknown`` is a claim about a
principal and nobody made it.

**This module makes no claims.** It imports nothing from :mod:`tenbin.claims`
and forms no opinion about whether a record is good; it says what the record
claims and whether its claim is anchored. Whether an anchored claim is
*sufficient* is the argument :mod:`tenbin.claims` exists to have, and a
validation layer that started making it would be grading its own homework.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Final, TypeGuard

from tenbin.corpus.record import PLACEHOLDER, Record

#: The literal Kojutsu writes where a value was never stated, used here as the
#: "this is not a real repository" marker. A repository genuinely named
#: ``unknown`` is not a thing, and a bucket labelled that name is a claim about a
#: principal that no writer made.
PLACEHOLDER_VALUE: Final[str] = PLACEHOLDER

#: The three capture sources, in the trust order Kojutsu documents. Spelled
#: here rather than imported because the corpus layer must not depend on the
#: writing project for a three-value vocabulary, and because a *new* value has to
#: be a deliberate act here -- a reader that imported a closed enum would treat a
#: writer's fourth source as a different kind of problem rather than as the
#: invalid one it is.
CAPTURE_WEBHOOK: Final[str] = "webhook"
CAPTURE_COLLECT: Final[str] = "collect"
CAPTURE_ASSERTED: Final[str] = "asserted"

#: The capture sources that claim a real observation, and so must be anchored.
#: ``asserted`` is excluded because it claims no capture at all: a caller typed it
#: in, so there is nothing to point at and nothing is missing.
CAPTURED_SOURCES: Final[frozenset[str]] = frozenset({CAPTURE_WEBHOOK, CAPTURE_COLLECT})

#: Key for a histogram of repositories with no repository to report under. A
#: distinct literal rather than an empty string, so a report cannot accidentally
#: present a missing repository as an empty one.
NO_REPOSITORY: Final[str] = "(no repository)"

#: Key for a histogram of months with no timestamp to report under.
NO_MONTH: Final[str] = "(undated)"


class AnomalyKind(StrEnum):
    """One way a record's provenance claim is not supported by the record.

    A closed set, because these are the only statements this layer is allowed to
    make about a record, and an open string field would let a future author add a
    kind that no reader had agreed to mean. The values are ``StrEnum`` members so
    a report can group by them without converting anything.
    """

    #: No ``capture_source`` at all, or the key left blank. Nothing says how the
    #: record came to exist, so nothing about it is checkable.
    CAPTURE_SOURCE_UNKNOWN = "capture_source_unknown"

    #: A ``capture_source`` outside the three-value vocabulary. A writer inventing
    #: a fourth source, or a document edited by hand; either way a reader cannot
    #: tell what class of claim is being made.
    CAPTURE_SOURCE_INVALID = "capture_source_invalid"

    #: An ``independence`` level on a record that claims no capture. Independence
    #: is computed by the capture pipeline from the accounts involved, so a level
    #: on an ``asserted`` record was asserted by whoever typed it -- which is the
    #: one thing the axis exists to distinguish.
    INDEPENDENCE_WITHOUT_CAPTURE = "independence_without_capture"

    #: A captured record missing the anchor its capture source promises: a
    #: delivery id for ``webhook``, a comment id for ``collect``, and both
    #: alongside a repository, a pull request and a capture time. See
    #: :func:`anchor_gaps` for which ones, per record.
    CAPTURE_ANCHOR_MISSING = "capture_anchor_missing"

    #: No repository. Either the id carried none or it carried the ``"unknown"``
    #: placeholder, and a record with no repository cannot be grouped, cannot be
    #: joined to a change, and cannot be checked against anything.
    REPO_MISSING = "repo_missing"


def _present(value: str | None) -> TypeGuard[str]:
    """True for a value that says something; a blank string says nothing.

    A ``TypeGuard`` rather than a plain ``bool`` so the caller keeps the
    narrowing: every rule below is about whether a key carries a usable value, and
    re-checking ``is not None`` after asking is the sort of repetition that
    eventually disagrees with the helper.
    """
    return value is not None and bool(value.strip())


def _named_repository(repo: str | None) -> TypeGuard[str]:
    """True for a repository that is a repository, and not the ``"unknown"`` slot."""
    return _present(repo) and repo.strip().casefold() != PLACEHOLDER_VALUE


def anchor_gaps(record: Record) -> tuple[str, ...]:
    """Name the anchors this record's capture source promises but does not carry.

    Mirrors Kojutsu's own ``capture_anchor_gaps``, which is the definition of
    "captured means checkable" and lives in the writing project so that both
    sides of the seam use one rule. The names are the storage keys rather than
    Kojutsu's metadata paths, because what a reader needs to act on is the key
    that is empty in the document it is holding.

    ``webhook`` requires a repository, a pull request, a capture time and the
    provider's delivery id -- the four things that make a signed delivery
    re-fetchable. ``collect`` requires a repository, a pull request, a capture
    time and the source comment id, and deliberately **not** a delivery id: the
    ``collect`` path is an authenticated API read rather than a provider
    delivery, so there is no delivery to record and demanding one would flag
    every honestly collected record. ``asserted`` requires nothing, because it
    claims no capture.
    """
    source = record.capture_source
    if source is None or source == CAPTURE_ASSERTED:
        return ()
    if source == CAPTURE_WEBHOOK:
        required: tuple[tuple[str, bool], ...] = (
            ("repo", _named_repository(record.repo)),
            ("pr", record.pr is not None),
            ("captured_at", record.captured_at is not None),
            ("delivery_id", _present(record.delivery_id)),
        )
    elif source == CAPTURE_COLLECT:
        required = (
            ("repo", _named_repository(record.repo)),
            ("pr", record.pr is not None),
            ("captured_at", record.captured_at is not None),
            ("github_comment_id", record.github_comment_id is not None),
        )
    else:
        # An unrecognised source is its own anomaly; inventing a rule for it here
        # would either demand anchors a writer never promised or wave it through
        # as if it were ``asserted``. It is reported, not guessed at.
        return ()
    return tuple(name for name, satisfied in required if not satisfied)


def validate(record: Record) -> frozenset[AnomalyKind]:
    """Every way this record's provenance claim is unsupported, as a pure function.

    Pure and total: same record in, same set out, no state, no clock, and never
    an exception -- a record it cannot reason about is returned as a record with
    every anomaly it warrants, because dropping it would silently shrink the
    denominator it is about to appear in.
    """
    anomalies: set[AnomalyKind] = set()
    source = record.capture_source
    if not _present(source):
        anomalies.add(AnomalyKind.CAPTURE_SOURCE_UNKNOWN)
    elif source not in (CAPTURE_WEBHOOK, CAPTURE_COLLECT, CAPTURE_ASSERTED):
        anomalies.add(AnomalyKind.CAPTURE_SOURCE_INVALID)

    # Independence is derived by the capture pipeline from the accounts that
    # posted, so a level on a record claiming no capture was not derived at all.
    # A blank level is an absence and is not flagged: the writer left a
    # placeholder, which is an absence of a claim rather than a claim.
    if _present(record.independence) and source not in CAPTURED_SOURCES:
        anomalies.add(AnomalyKind.INDEPENDENCE_WITHOUT_CAPTURE)

    if anchor_gaps(record):
        anomalies.add(AnomalyKind.CAPTURE_ANCHOR_MISSING)

    if not _named_repository(record.repo):
        anomalies.add(AnomalyKind.REPO_MISSING)
    return frozenset(anomalies)


def _counts(anomaly_counts: Mapping[AnomalyKind, int]) -> Mapping[AnomalyKind, int]:
    """Freeze a counter, including every kind at zero.

    Every kind is present even when nothing triggered it, because a profile whose
    shape depends on the corpus is a profile every report has to guard against. A
    stable mapping means a report can print a row for every anomaly kind and have
    it mean the same thing on a clean corpus as on a broken one.
    """
    return MappingProxyType({kind: anomaly_counts.get(kind, 0) for kind in AnomalyKind})


def _histogram(counts: Mapping[str, int]) -> Mapping[str, int]:
    """Freeze a counter keyed by text, ordered by count then key.

    Sorted so that two profiles over equal data are equal, which is what makes a
    profile comparable in a test rather than merely comparable by eye.
    """
    return MappingProxyType(dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))))


@dataclass(frozen=True)
class ProvenanceProfile:
    """What a corpus's provenance looks like, and where it stops being anchored.

    Every field is a count, and the three breakdowns count the **anomalous**
    records only -- the field names say so, because ``by_repository`` on a
    profile of anomalies invites exactly the reading that it is a profile of
    everything. ``total`` and ``clean`` are the two numbers a rate needs as its
    numerator and denominator, kept on the same object as the anomalies so that
    nobody can quote one without the other being available.

    The breakdowns are histograms, not lists, and a repository with three
    anomalous records is three times the problem of one with a single one. A
    concentration of broken provenance in one repository is a fact about that
    repository, and it is the fact this object exists to make reportable.
    """

    total: int
    clean: int
    anomalies_by_kind: Mapping[AnomalyKind, int]
    anomalous_by_repository: Mapping[str, int]
    anomalous_by_month: Mapping[str, int]
    records: tuple[Record, ...] = ()

    @property
    def anomalous(self) -> int:
        """How many records carry at least one anomaly."""
        return self.total - self.clean

    def __hash__(self) -> int:
        # A frozen dataclass holding mappings claims to be hashable and is not,
        # so the hash is defined on the population and the shape. Equal profiles
        # hash equal because equality already implies the same counts.
        return hash((self.total, self.clean, tuple(sorted(self.anomalies_by_kind.items()))))


def provenance_profile(records: Iterable[Record]) -> ProvenanceProfile:
    """Describe a corpus's provenance, counting the anomalous records among it.

    Takes any iterable and consumes it once: the anomalous records are kept so
    that a caller can go back to the documents behind a histogram bucket, which is
    the difference between "eight records in ``core`` have no repository" and a
    finding nobody can act on.
    """
    anomalous_records: list[Record] = []
    anomalies_by_kind: Counter[AnomalyKind] = Counter()
    by_repository: Counter[str] = Counter()
    by_month: Counter[str] = Counter()
    total = 0
    clean = 0
    for record in records:
        total += 1
        found = validate(record)
        if not found:
            clean += 1
            continue
        anomalies_by_kind.update(found)
        by_repository[_repository_key(record.repo)] += 1
        by_month[record.month or NO_MONTH] += 1
        anomalous_records.append(record)
    return ProvenanceProfile(
        total=total,
        clean=clean,
        anomalies_by_kind=_counts(anomalies_by_kind),
        anomalous_by_repository=_histogram(by_repository),
        anomalous_by_month=_histogram(by_month),
        records=tuple(anomalous_records),
    )


def _repository_key(repo: str | None) -> str:
    """The histogram key for a record's repository, or the explicit missing one."""
    return repo.strip() if _named_repository(repo) else NO_REPOSITORY
