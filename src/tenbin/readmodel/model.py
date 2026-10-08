"""The derived read model: one walk of the store, kept, and never added to.

Tanseki has no aggregation, so every measure in this program is a full enumeration
over documents fetched by hand (:mod:`tenbin.corpus.snapshot`). At pilot scale
that is fine; the first thing that stops being fine is the same walk run nine
times in one report. The read model is the answer to that, and the answer is a
*rebuildable* artefact rather than a cache with a lifetime: it holds the records
a read produced, the moment it was read, the counts that say whether the read
reached the end, and the classification every record was given. Nothing else.

**The classification is stored, not recomputed.** :func:`tenbin.corpus.record.classify`
is where the corpus layer makes its judgement calls, and a report that recomputed
them would be a second implementation of a rule that already exists -- one that
could disagree with the walk it is describing without either copy looking wrong.
Storing the result makes the model the place that judgement lives, and
:data:`SCHEMA_VERSION` is the handle on it: a change to the classifier is a change
to what a stored classification *means*, so it is a version bump and a rebuild,
never a reinterpretation of the models already on disk.

**There is no update path, and there must not be one.** This module exports
:func:`build`, :func:`write` and :func:`load`, and nothing that adds a record,
removes one, merges two models, or refreshes one in place. The reason is that an
update path is where a derived model quietly becomes a maintained one, and a
maintained model is a source of truth nobody reconciles: it would answer questions
about a corpus that no longer exists, its figures would disagree with a live read
of the same collection for reasons no report could state, and the disagreement
would be settled by whichever copy a reader happened to open. A rebuild is the
expensive operation and that is the correct trade -- the walk costs a round trip
per hundred documents, and being wrong costs a number somebody believed.

**A model records how the read happened, because a read is not one walk.** The
seam refuses a listing offset above :data:`~tenbin.store.client.MAX_PAGE_OFFSET`, so
a read that reaches the ceiling is finished in more than one window, and a model
that could not say so would be indistinguishable from a model of a smaller corpus.
:attr:`ReadModel.windows` is that record: one :class:`WalkWindow` per window walked,
each with the offset range it covered, how many ids it listed, how many documents
it added, what the store's own count was while it ran, and how that window ended.
A reader sees the seams rather than inferring a single walk that never happened, and
a later build can continue from the offset the last window reached. The windows are
also what makes the walk *auditable* rather than merely reproducible: a document
listed twice shows up as a window whose ``enumerated`` exceeds its ``documents``.

**Resuming is not updating, and it lives in a different module.**
:mod:`tenbin.readmodel.resume` is the one place that walks the store in more than
one window, and this module holds no client and offers it nothing to update: a
resumed build still produces a whole model, it just gets there in more than one walk,
and :func:`write` still replaces the file or leaves it. It is deliberately *not*
re-exported from the package, because the package's published surface is the four
verbs above and a function taking a store client on the package's face would make
:mod:`tenbin.readmodel` look like the layer that reads the store -- which it is not,
and which :mod:`tenbin.cli` is.

**A read model is a statement about a past read, or it is nothing.**
:func:`load` returns ``None`` for a file that is absent or cannot be parsed and
never a half-read model, because a model missing its third record is a corpus
with a hole in it that reports itself as a corpus: the completeness sentence every
figure carries would say the walk reached the end of the store while the model
quietly held less of it. A model whose :data:`SCHEMA_VERSION` this program does
not recognise is a different case and is *refused* rather than ignored -- a file
written by another version is not corrupt, and quietly walking the store instead
would hide a version skew that somebody has to know about.

**What this module notably does not do:** it does not read the store, and it
holds no client. It does not measure, gate, or render, and it does not decide
which measures may run over what it holds. It does not compare one model against
another, so there is no way to ask what changed between two reads here -- that
question is answered by reading both, and a diff facility is the first step toward
an update path. Nor does it validate its own contents against the store: nothing
in this file can know whether the corpus changed since the walk, and the field
that records *when* the walk happened is the whole of the answer this module gives
to that question.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from tenbin.corpus.provenance import NO_MONTH, NO_REPOSITORY
from tenbin.corpus.record import Certainty, Classification, Record, RecordKind
from tenbin.corpus.snapshot import Completeness, CorpusSnapshot, Truncation
from tenbin.store.client import (
    StoreAuthenticationError,
    StoreConfigurationError,
    StoreDocument,
    StoreError,
    StoreNotFoundError,
    StoreResponseError,
    StoreUnavailableError,
)

#: The shape of the file this program writes, as a version the loader recognises.
#: A string rather than an integer because a version is written once and read by
#: other programs and by people, and ``"1"`` survives being pasted into a document
#: where ``1`` reads as a count of something. Bump it for any change to what a
#: stored record or classification *means* -- a new field read, a new record kind,
#: a change in what the classifier concludes -- and not for a change of formatting.
#:
#: **Version two is the read's own shape, and the bump is deliberate rather than
#: cautious.** A version-1 model could not say how many windows it was read in,
#: because it did not record them. Defaulting the missing field to a single window
#: would be worse than refusing the file: it would assert "one walk" about a model
#: that was read in three, and the failure that assertion produces is the one this
#: program exists to prevent -- a number that is right once and then invisible,
#: read a hundred times with nobody noticing. **A version-1 file is therefore
#: refused, and the answer is a rebuild**, which is the same remedy the refusal
#: message has always given and the reason it is a raise rather than a ``None``.
SCHEMA_VERSION: Final[str] = "2"

#: The key the moment of the read is stored under. Named so that the payload's
#: vocabulary is written down in one place rather than spelled at the two ends of
#: the round trip, where a typo would be a ``None`` rather than an error.
READ_AT_KEY: Final[str] = "read_at"

#: The four indexes every read model carries, named as constants because they are
#: the reason a read model is worth having: the walk is replaced by a load, and a
#: load that still had to group ten thousand records by month to answer one
#: question would only be a cheaper walk.
BY_ID: Final[str] = "by_id"
BY_REPO: Final[str] = "by_repo"
BY_MONTH: Final[str] = "by_month"
BY_KIND: Final[str] = "by_kind"

#: The store's own error classes, by name, so a walk that failed can be restored
#: as the failure it was. A name is stored rather than a type because a type is a
#: Python object and a file is text; a name this program does not recognise falls
#: back to a :class:`~tenbin.store.client.StoreResponseError` carrying a message
#: that says what it was, because *which* failure stopped the walk changes what an
#: operator should do about it and "some store error" is not an answer.
_STORE_FAILURES: Final[Mapping[str, type[StoreError]]] = MappingProxyType(
    {
        "StoreConfigurationError": StoreConfigurationError,
        "StoreUnavailableError": StoreUnavailableError,
        "StoreAuthenticationError": StoreAuthenticationError,
        "StoreResponseError": StoreResponseError,
        "StoreNotFoundError": StoreNotFoundError,
    }
)


class UnknownSchemaVersionError(RuntimeError):
    """A read model was written by a version of this program that is not this one.

    Raised rather than returning ``None``, because a file this program cannot
    interpret is not a missing file: the model exists, somebody wrote it, and the
    reason it cannot be used is a version skew that a human has to resolve. Answer
    with ``None`` would quietly walk the store instead and produce a correct report
    from a different read, so the skew would never be noticed -- and the whole
    reason the version is stored is so that it can be.

    ``RuntimeError`` and not :class:`ValueError` so that :func:`load` can treat
    "malformed" and "not mine" differently: a ``ValueError`` means absent, and this
    one must survive the handler that turns those into ``None``.
    """


@dataclass(frozen=True)
class StoredFailure:
    """A walk that stopped, as the two things about it that survive a file.

    The class name because the remedy differs by failure -- an operator changes a
    key for one of these and waits for the other -- and the message because the
    exception's own ``str()`` is the only account of *which request* failed. What
    is not kept is the exception object, because an exception is not a value: it
    carries a traceback and a chain that no file can hold and no reader would
    want. A restored failure is therefore a new object carrying the old account of
    what happened, which is exactly as much as a figure renders.
    """

    kind: str
    message: str

    @classmethod
    def from_error(cls, error: StoreError | None) -> StoredFailure | None:
        """The stored form of a walk's failure, or ``None`` when it did not fail."""
        if error is None:
            return None
        return cls(kind=type(error).__name__, message=str(error))

    def restore(self) -> StoreError:
        """The failure as a store error, of the class it was when it stopped the walk."""
        failure = _STORE_FAILURES.get(self.kind)
        if failure is not None:
            return failure(self.message)
        return StoreResponseError(
            f"a read model recorded a walk that failed with {self.kind}, which this version "
            f"of Tenbin does not know: {self.message}"
        )


@dataclass(frozen=True)
class WalkWindow:
    """One stretch of document offsets a walk listed, and what that stretch yielded.

    **A window is a fact about how the read happened, and it is in the model because
    the seam's offset ceiling makes a read more than one walk.** Tanseki refuses a
    listing offset above :data:`~tenbin.store.client.MAX_PAGE_OFFSET`, so a read of
    a collection that reaches the ceiling is finished by walking the tail again from
    inside the range the previous window covered. Without this record the two things
    are indistinguishable: a read that walked one window and a read that walked four
    produce the same kind of file, and a reader cannot tell which is looking at them.

    The five fields are all of it, and each is a number a figure cannot be built
    without:

    - ``offset_start`` and :attr:`offset_end` are the range the window listed, half
      open. The end is derived rather than stored so a caller cannot hold a window
      whose range disagrees with its own count.
    - ``enumerated`` is how many ids the store **listed** in that range, and it counts
      a document again if the range was listed twice. That is not inflation to be
      tidied away: it is the evidence that the walk deliberately re-listed a seam.
    - ``documents`` is how many documents this window **added to the model**, so
      ``enumerated - documents`` is the number of documents it saw and already had.
      The difference sums across the windows to every document listed more than once,
      and that sum is how a re-seen document is shown to have been counted once.
    - ``store_total`` is the store's own count *while this window ran*, kept per
      window because a total that is the same in every window is the only evidence
      available that the collection did not move under the walk.
    - ``completeness`` is how this window ended: the store said there was nothing
      more, the ceiling was reached, or the walk failed here. It is the window's own
      verdict and it is allowed to be ``STORE_TOTAL_EXCEEDED`` because a whole walk
      in one window reaches that verdict on its own; a *resumed* build never puts it
      on a window, because there the disagreement is between windows and belongs to
      the model.

    **The ranges may overlap and the windows are contiguous, and both are enforced.**
    Overlap is how a resumed read does not skip a document that the listing moved
    between two windows; contiguity is how the model refuses to claim a read that
    never listed part of its own range. A gap would be a claim about documents
    nobody asked for, which is the one thing a stored read must not do.
    """

    offset_start: int
    enumerated: int
    documents: int
    store_total: int | None
    completeness: Completeness

    def __post_init__(self) -> None:
        _require_count("WalkWindow.offset_start", self.offset_start)
        _require_count("WalkWindow.enumerated", self.enumerated)
        _require_count("WalkWindow.documents", self.documents)
        if self.store_total is not None:
            _require_count("WalkWindow.store_total", self.store_total)
        if not isinstance(self.completeness, Completeness):
            raise ValueError(
                f"WalkWindow.completeness must be a Completeness, not {self.completeness!r}; a "
                "bare string would restore as a verdict this program never reached."
            )
        if self.documents > self.enumerated:
            # A window cannot contribute more documents than it was shown, so the two
            # disagreeing is a transcription error rather than a strange store, and a
            # strange store is reported by the walk that saw it rather than here.
            raise ValueError(
                f"WalkWindow.documents ({self.documents}) exceeds the ids the window listed "
                f"({self.enumerated}); a window cannot add a document it was not shown."
            )

    @property
    def offset_end(self) -> int:
        """The offset this window's range ends at, exclusive.

        A property rather than a field because it is arithmetic on the two numbers
        that are already here, and a stored copy would be a second account of the
        same fact -- the hazard :func:`tenbin.corpus.record.Record` is built to have
        no version of.
        """
        return self.offset_start + self.enumerated


@dataclass(frozen=True)
class ReadModel:
    """One read of a collection, kept whole, with the indexes a report reads it by.

    Frozen for the reason a :class:`~tenbin.corpus.snapshot.CorpusSnapshot` is: a
    read model is a statement about a moment, and a record appended to it afterwards
    would no longer be the corpus anybody read. Every field is required except
    ``truncation`` and ``failure``, which are the two ways a walk can be less than
    whole and which have to be *declared* rather than defaulted -- a read model with
    no way to say it was truncated is the exact defect the corpus layer refuses.

    **``windows`` is required, and an empty tuple is refused.** It is the record of
    how the read was walked, and a model with no window in it is a read that listed
    no offset range at all: there would be nothing to continue from, and a reader
    would have no way to tell it apart from a model of a collection that was never
    walked. The existing model is a single window from ``[0, enumerated)``, which is
    what :func:`build` writes for a read that fitted in one walk.

    **The indexes are computed here and cannot be supplied.** A caller that could
    pass its own would be able to pass one that disagrees with ``records``, and the
    disagreement is invisible: every figure would render from the records and every
    lookup would answer from the index. So they are ``init=False`` and derived in
    :meth:`__post_init__`, and the values are *document ids* rather than records --
    one copy of each record, with the indexes naming it. ``by_repo``, ``by_month``
    and ``by_kind`` key on the corpus package's own vocabulary, including its
    sentinels for a repository and a month a record does not have, so a caller
    grouping by repository cannot file those records twice.
    """

    schema_version: str
    collection: str
    read_at: datetime
    enumerated: int
    store_total: int
    completeness: Completeness
    windows: tuple[WalkWindow, ...]
    records: tuple[Record, ...] = ()
    truncation: Truncation | None = None
    failure: StoredFailure | None = None

    by_id: Mapping[str, Record] = field(init=False)
    by_repo: Mapping[str, tuple[str, ...]] = field(init=False)
    by_month: Mapping[str, tuple[str, ...]] = field(init=False)
    by_kind: Mapping[str, tuple[str, ...]] = field(init=False)
    classifications: Mapping[str, Classification] = field(init=False)

    def __post_init__(self) -> None:
        if not self.collection.strip():
            raise ValueError("ReadModel.collection is required; a read model names no corpus.")
        if not isinstance(self.completeness, Completeness):
            raise ValueError(
                f"ReadModel.completeness must be a Completeness, not {self.completeness!r}; "
                "a bare string would restore as a verdict this program never reached."
            )
        if not isinstance(self.read_at, datetime) or self.read_at.tzinfo is None:
            # A moment with no zone is not a moment anybody can place, and this
            # program only ever writes aware timestamps. Rather than assume UTC --
            # which would put a corrupted model's read at the right time by accident
            # -- the file is treated as one this version did not write.
            raise ValueError(
                "ReadModel.read_at must be an aware datetime; a model whose read cannot be "
                "placed in time is not a read of anything."
            )
        _require_count("ReadModel.enumerated", self.enumerated)
        _require_count("ReadModel.store_total", self.store_total)
        _require_windows(self.windows)

        by_id: dict[str, Record] = {}
        by_repo: dict[str, list[str]] = {}
        by_month: dict[str, list[str]] = {}
        by_kind: dict[str, list[str]] = {}
        for record in self.records:
            if record.doc_id in by_id:
                # Two records under one id means the walk appended the same document
                # twice, and an index that kept the second would answer every lookup
                # with it while the figures counted both. That is a contradiction, and
                # the honest response to one is to refuse the model.
                raise ValueError(
                    f"ReadModel.records holds two records for {record.doc_id!r}; a walk that "
                    "enumerated one document twice is not a read, and an index over it would "
                    "have to pick a winner."
                )
            by_id[record.doc_id] = record
            by_repo.setdefault(record.repo or NO_REPOSITORY, []).append(record.doc_id)
            by_month.setdefault(record.month or NO_MONTH, []).append(record.doc_id)
            by_kind.setdefault(record.classification.kind.value, []).append(record.doc_id)
        object.__setattr__(self, "by_id", MappingProxyType(by_id))
        object.__setattr__(self, "by_repo", _frozen_groups(by_repo))
        object.__setattr__(self, "by_month", _frozen_groups(by_month))
        object.__setattr__(self, "by_kind", _frozen_groups(by_kind))
        object.__setattr__(
            self,
            "classifications",
            MappingProxyType({r.doc_id: r.classification for r in self.records}),
        )

    def __hash__(self) -> int:
        # The generated hash would walk the records and hash their frontmatter, which
        # raises; and the useful identity of a read model is the read it describes,
        # not the copy of the corpus it holds. This mirrors ``CorpusSnapshot``'s own
        # rule for the same reason.
        return hash(
            (
                self.schema_version,
                self.collection,
                self.read_at,
                self.enumerated,
                self.completeness,
                self.truncation.offset_reached if self.truncation else None,
            )
        )

    def records_in(self, index: Mapping[str, tuple[str, ...]], key: str) -> tuple[Record, ...]:
        """The records an index names under one key, in the order the read produced them.

        A function rather than three attribute lookups so that the two steps -- find
        the ids, resolve them -- cannot be done in the wrong order by a caller, and
        so that a key with no bucket returns nothing rather than raising. Order is the
        walk's, because a chain built over a reordered corpus is a different chain.
        """
        return tuple(self.by_id[doc_id] for doc_id in index.get(key, ()))


def _frozen_groups(groups: Mapping[str, list[str]]) -> Mapping[str, tuple[str, ...]]:
    """Freeze the index values so a caller cannot edit the model through one."""
    return MappingProxyType({key: tuple(ids) for key, ids in groups.items()})


def _require_count(owner: str, value: int) -> None:
    """Refuse a count that is not a count.

    ``bool`` is refused explicitly because it is an ``int`` in Python, and ``True``
    as a document count is a defect that would otherwise reach a rendered header.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{owner} must be a non-negative integer count, not {value!r}")


def _require_windows(windows: tuple[WalkWindow, ...]) -> None:
    """Refuse a window list that does not describe a walk of the whole listing.

    Three rules, and the third is the one a caller is most likely to break by
    accident. **The list is a tuple of windows**, so a model cannot be edited
    through it. **It is not empty**, because a read that listed no offset range is
    not a read of a collection. **The first window starts at zero and no two windows
    leave a gap between them**: a read that began partway down the listing, or that
    skipped a stretch of offsets, is claiming a coverage it does not have, and a
    reader would have no way to see the difference.

    Overlap *is* allowed, and deliberately: re-listing a page is how a resumed read
    avoids skipping a document the listing moved. The check refuses a gap, not an
    overlap.
    """
    if not isinstance(windows, tuple) or not all(
        isinstance(window, WalkWindow) for window in windows
    ):
        raise ValueError("ReadModel.windows must be a tuple of WalkWindow; the walk is the record.")
    if not windows:
        raise ValueError(
            "ReadModel.windows must not be empty; a read that listed no offset window is not a "
            "read of anything, and a model with no window is one nothing could be resumed from."
        )
    if windows[0].offset_start != 0:
        raise ValueError(
            f"the first walk window starts at offset {windows[0].offset_start}, not 0; a read "
            "begins at the top of the listing, and one that did not is missing whatever came "
            "before it."
        )
    previous = windows[0]
    for window in windows[1:]:
        if window.offset_start > previous.offset_end:
            raise ValueError(
                f"ReadModel.windows leaves offsets [{previous.offset_end}, {window.offset_start}) "
                "unwalked; a read with a gap in its own offsets is claiming documents nobody "
                "asked the store for."
            )
        previous = window


def build(snapshot: CorpusSnapshot) -> ReadModel:
    """Keep a read, with everything needed to doubt it.

    A copy of the snapshot's fields rather than a reference to the snapshot itself,
    because the model outlives the walk that produced it: holding the snapshot would
    mean holding the records twice and would let a caller reach the store-error
    object this module is careful to reduce to a name and a message.

    The records are the snapshot's own -- a read model does not re-read, re-parse or
    re-classify anything, which is what makes a figure computed from the model equal
    to the same figure computed live rather than merely similar.

    The walk is recorded as the single window it was, from offset ``0`` to the count
    it enumerated. That is a statement about how this read happened rather than an
    inference: a snapshot cannot say whether a walk that reached the ceiling was
    finished by a second window, and a resumed build in
    :mod:`tenbin.readmodel.resume` writes windows of its own rather than passing
    this function a snapshot, because a snapshot is one walk by construction.
    """
    return ReadModel(
        schema_version=SCHEMA_VERSION,
        collection=snapshot.collection,
        read_at=snapshot.read_at,
        enumerated=snapshot.enumerated,
        store_total=snapshot.store_total,
        completeness=snapshot.completeness,
        windows=(
            WalkWindow(
                offset_start=0,
                enumerated=snapshot.enumerated,
                documents=len(snapshot.records),
                store_total=snapshot.store_total,
                completeness=snapshot.completeness,
            ),
        ),
        records=snapshot.records,
        truncation=snapshot.truncation,
        failure=StoredFailure.from_error(snapshot.error),
    )


def write(path: Path, model: ReadModel) -> Path:
    """Write the model, atomically, and return the path.

    **The write is a temporary file in the same directory followed by a rename.**
    ``os.replace`` is atomic where the two paths share a filesystem, so a reader --
    or the next build -- sees either the previous model or the new one and never a
    half-written file. The alternative, writing in place, leaves a truncated JSON
    document on every interrupted build, and :func:`load` would then treat the
    wreckage as absent: the operator would lose the previous model and learn nothing
    about why. The temporary file is a sibling rather than in the system temporary
    directory for the same reason, and it is removed if anything goes wrong so a
    failed build leaves no litter beside the model it did not replace.

    ``fsync`` before the rename, because atomicity is about what a reader sees and a
    file that is renamed into place and then lost on a power cut is a model that
    existed for a moment and did not.

    A model whose ``schema_version`` is not this program's is refused here rather
    than written, because :func:`load` will refuse to read it back and a file nobody
    can read is not a cache.
    """
    if model.schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"refusing to write a read model at schema version {model.schema_version!r}; this "
            f"program writes and reads version {SCHEMA_VERSION!r}, and a file it cannot read "
            "back is not a cache."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            # Serialised inside the guarded region rather than before it, so that a
            # frontmatter value JSON cannot hold fails here -- with the temporary
            # file cleaned up -- instead of failing one step earlier for no reason a
            # reader could see.
            json.dump(_as_payload(model), stream, indent=2, sort_keys=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        # Only the write and the rename can fail, and in both cases the previous
        # model is still exactly where it was: nothing in this function touches it
        # until the rename succeeds.
        Path(temporary).unlink(missing_ok=True)
        raise
    return path


def load(path: Path) -> ReadModel | None:
    """The model at ``path``, or ``None`` when there is nothing usable there.

    Three outcomes and the middle one is the interesting one. **Absent:** no file,
    unreadable, not JSON, or JSON that does not describe a read model -- all
    ``None``, because the caller's honest options are to walk the store or to report
    on nothing, and this one walks. **Refused:** a file that parses and names a
    schema version this program does not implement, which raises
    :exc:`UnknownSchemaVersionError` rather than returning ``None``, because a
    version skew is a fact about the world that a human has to resolve. **Read:** the
    model, with its classifications taken from the file rather than recomputed.

    Treating a malformed file as absent rather than as half-trusted is the whole
    argument for this function. A model that parsed its header and lost its records
    would render figures whose completeness sentence says the walk reached the end
    of the store, over a corpus that is a fraction of it, and nothing in the output
    would say otherwise.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, ValueError):
        return None
    try:
        payload = json.loads(text)
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    version = payload.get("schema_version")
    if not isinstance(version, str):
        return None
    if version != SCHEMA_VERSION:
        raise UnknownSchemaVersionError(
            f"{path} holds a read model at schema version {version!r}; this program reads "
            f"{SCHEMA_VERSION!r}. Rebuild the model rather than reading it -- an old model is "
            "not a cheaper read, it is a different one."
        )
    try:
        return _from_payload(payload)
    except (KeyError, TypeError, ValueError):
        return None


def _as_payload(model: ReadModel) -> dict[str, Any]:
    """The model as JSON-ready values, with the header first and the records last.

    Key order is the reading order and is not sorted, for the reason the JSON report
    renderer gives: a reader -- or a diff -- should meet the read that was kept
    before the records it consists of.

    The indexes are **not** written. They are derived from the records in
    :meth:`ReadModel.__post_init__`, and a stored index would be a second copy of a
    fact the records already carry: a file whose index disagreed with its own records
    would load into a model that lies about its contents, and the only way to find out
    which half to believe would be to rebuild.
    """
    truncation = model.truncation
    failure = model.failure
    return {
        "schema_version": model.schema_version,
        "collection": model.collection,
        READ_AT_KEY: model.read_at.isoformat(),
        "enumerated": model.enumerated,
        "store_total": model.store_total,
        "completeness": model.completeness.value,
        "truncation": None
        if truncation is None
        else {
            "offset_reached": truncation.offset_reached,
            "records_missing": truncation.records_missing,
            "boundary_repositories": dict(truncation.boundary_repositories),
            "boundary_months": dict(truncation.boundary_months),
        },
        "failure": None if failure is None else {"kind": failure.kind, "message": failure.message},
        "windows": [_window_as_payload(window) for window in model.windows],
        "records": [_record_as_payload(record) for record in model.records],
    }


def _window_as_payload(window: WalkWindow) -> dict[str, Any]:
    """One window: the range it listed, what it yielded, and how it ended.

    Reading order within the object, so a reader -- or a diff -- meets where the
    window started, what came of it, what the store was claiming at the time, and
    only then the verdict. ``documents`` sits next to ``enumerated`` rather than at
    the end because the difference between them is the number this record exists to
    make checkable: a window that listed more ids than it added was looking at
    documents the read already had.
    """
    return {
        "offset_start": window.offset_start,
        "enumerated": window.enumerated,
        "documents": window.documents,
        "store_total": window.store_total,
        "completeness": window.completeness.value,
    }


def _record_as_payload(record: Record) -> dict[str, Any]:
    """One record: the document it was read from, then what it was classified as.

    The document first because that is the order of the reading -- an id, a body and
    a frontmatter are what the store held, and a classification is what this program
    concluded from them. Only the fields :class:`~tenbin.store.client.StoreDocument`
    carries are written, because those are the only ones
    :meth:`~tenbin.corpus.record.Record.from_document` reads; anything else would be
    a copy in the file that no reader of the file would use.
    """
    return {
        "id": record.doc_id,
        "collection": record.collection,
        "path": record.path,
        "content": record.content,
        "content_hash": record.content_hash,
        "updated_at": None if record.updated_at is None else record.updated_at.isoformat(),
        "revision": record.revision,
        "frontmatter": dict(record.frontmatter),
        "classification": {
            "kind": record.classification.kind.value,
            "certainty": record.classification.certainty.value,
            "evidence": list(record.classification.evidence),
            "detail": record.classification.detail,
        },
    }


def _from_payload(payload: Mapping[str, Any]) -> ReadModel:
    """Rebuild a model from parsed JSON, refusing anything that is not one.

    Every failure here is a :class:`ValueError` or a :class:`KeyError` on purpose:
    :func:`load` turns those into "absent", and a file this program cannot read is
    a file it must not read. The stored classification is substituted onto the
    parsed record rather than compared with it -- the file's word is the word, and
    disagreement would mean the classifier changed without a version bump, which is
    the one drift this module is built to make impossible.

    **``windows`` is read as a required key rather than defaulted.** A file without
    it is a version-1 file, and :func:`load` has already refused those by version
    before it gets here; a file that *is* at this version and has no windows is
    either truncated or hand-edited, and both are answered the same way -- by
    rebuilding. Defaulting the key to one window would be the failure this module
    exists to prevent: it would say "one walk" about a read that took three and let
    every figure above it claim a coverage nobody established.
    """
    read_at = datetime.fromisoformat(_text(payload[READ_AT_KEY], READ_AT_KEY))
    if read_at.tzinfo is None:
        read_at = read_at.replace(tzinfo=UTC)
    records = tuple(_record_from_payload(entry) for entry in _sequence(payload["records"]))
    windows = tuple(_window_from_payload(entry) for entry in _sequence(payload["windows"]))
    truncation = payload.get("truncation")
    failure = payload.get("failure")
    return ReadModel(
        schema_version=_text(payload["schema_version"], "schema_version"),
        collection=_text(payload["collection"], "collection"),
        read_at=read_at,
        enumerated=_count(payload["enumerated"], "enumerated"),
        store_total=_count(payload["store_total"], "store_total"),
        completeness=Completeness(_text(payload["completeness"], "completeness")),
        windows=windows,
        records=records,
        truncation=None if truncation is None else _truncation_from_payload(truncation),
        failure=None
        if failure is None
        else StoredFailure(
            kind=_text(failure["kind"], "failure.kind"),
            message=_text(failure["message"], "failure.message"),
        ),
    )


def _window_from_payload(entry: Any) -> WalkWindow:
    """One window from its stored range, counts, and verdict.

    ``store_total`` is the one field allowed to be absent, because a store that does
    not report a count is refused by the walk rather than stored -- so a window
    carrying ``None`` is a model built by hand, and reading it back as ``None`` is
    the honest thing to do with it rather than inventing the number it should have
    held.
    """
    if not isinstance(entry, dict):
        raise ValueError("a read model walk window must be an object")
    store_total = entry.get("store_total")
    return WalkWindow(
        offset_start=_count(entry["offset_start"], "window offset_start"),
        enumerated=_count(entry["enumerated"], "window enumerated"),
        documents=_count(entry["documents"], "window documents"),
        store_total=None if store_total is None else _count(store_total, "window store_total"),
        completeness=Completeness(_text(entry["completeness"], "window completeness")),
    )


def _record_from_payload(entry: Any) -> Record:
    """One record from its stored document and its stored classification."""
    if not isinstance(entry, dict):
        raise ValueError("a read model record must be an object")
    frontmatter = entry.get("frontmatter", {})
    if not isinstance(frontmatter, dict):
        raise ValueError("a read model record carries an invalid frontmatter")
    updated_at = entry.get("updated_at")
    document = StoreDocument(
        id=_text(entry["id"], "record id"),
        collection=_text(entry["collection"], "record collection"),
        path=_optional_text(entry.get("path"), "record path") or "",
        content=_text(entry.get("content", ""), "record content"),
        content_hash=_optional_text(entry.get("content_hash"), "record content_hash"),
        updated_at=None if updated_at is None else _text(updated_at, "record updated_at"),
        revision=_optional_text(entry.get("revision"), "record revision"),
        frontmatter=frontmatter,
    )
    classification = _classification_from_payload(entry.get("classification"))
    return replace(Record.from_document(document), classification=classification)


def _classification_from_payload(entry: Any) -> Classification:
    """The stored classification, read rather than re-derived."""
    if not isinstance(entry, dict):
        raise ValueError("a read model record carries no classification")
    evidence = entry.get("evidence", [])
    if not isinstance(evidence, list) or not all(isinstance(item, str) for item in evidence):
        raise ValueError("a read model classification carries invalid evidence")
    return Classification(
        kind=RecordKind(_text(entry.get("kind"), "classification kind")),
        certainty=Certainty(_text(entry.get("certainty"), "classification certainty")),
        evidence=tuple(evidence),
        detail=_text(entry.get("detail"), "classification detail"),
    )


def _truncation_from_payload(entry: Any) -> Truncation:
    """The truncation, with its two histograms restored as read-only mappings."""
    if not isinstance(entry, dict):
        raise ValueError("a read model truncation must be an object")
    return Truncation(
        offset_reached=_count(entry["offset_reached"], "offset_reached"),
        records_missing=_count(entry["records_missing"], "records_missing"),
        boundary_repositories=_groups(entry.get("boundary_repositories", {})),
        boundary_months=_groups(entry.get("boundary_months", {})),
    )


def _groups(entry: Any) -> Mapping[str, int]:
    """Restore a histogram as a read-only mapping of non-negative counts."""
    if not isinstance(entry, dict):
        raise ValueError("a read model histogram must be an object")
    return MappingProxyType({str(key): _count(value, str(key)) for key, value in entry.items()})


def _text(value: Any, field: str) -> str:
    """A non-blank string, or a refusal naming the field it was reading."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"a read model must hold a non-blank {field}")
    return value


def _optional_text(value: Any, field: str) -> str | None:
    """A string that may be absent, but not one that is present and empty."""
    if value is None:
        return None
    return _text(value, field)


def _count(value: Any, field: str) -> int:
    """A non-negative count, refusing booleans, which are integers in Python."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"a read model must hold a non-negative count for {field}, not {value!r}")
    return value


def _sequence(value: Any) -> Sequence[Any]:
    """A list, refused rather than iterated, so a string of records is not read as one."""
    if not isinstance(value, list):
        raise ValueError("a read model must hold a list of records")
    return value
