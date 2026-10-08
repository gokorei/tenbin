"""An append-only log of what each read rendered, so the programme can say what changed.

**There is no memory between runs, and this file is the memory.** Every read in
:mod:`tenbin.corpus` re-derives everything from the current corpus and compares it to
nothing, so *is the merge rate falling* has no answer -- and a measurement programme whose
stated purpose is measuring process over time, with no record of any past process, is a
snapshot somebody takes repeatedly. :class:`Archive` is the answer to that, and it is the
same reasoning :mod:`tenbin.readmodel` applies to a different subject: derived, rebuildable,
and never consulted for a figure. **A reader who wants today's number gets it from the
corpus.** This log answers *what was true then*, and a programme that let it answer
otherwise would be reporting a corpus that no longer exists and calling it the present.

**An archived figure is never re-computed and never silently replaced.** There is no update
method, no delete method, and no way to rewrite a line: :meth:`Archive.record` only ever
appends. That is the property the whole thing exists for, and it is enforced three ways --
there is no verb that could break it, the file is opened for appending and nothing else, and
:func:`_readings` refuses a line it cannot parse rather than skipping it. If a measure
changes, the archive does not get to fix its own history: the change shows up as a *third*
reading beside the two that disagree, and the reader sees a figure that moved for a reason
nobody can name. **A drift programme that rewrites its own past is a drift programme with no
drift to report**, and the temptation to make it consistent is the exact temptation the
refusal registry exists to refuse.

**The corpus window is stored beside every reading, and that is why a drift can refuse.**
A figure that moved over three weeks and one that moved over three years are different claims
about the same corpus, and nothing in the figure itself says which it is -- so a reader
cannot be given the pair without the two spans. :class:`~tenbin.measures.drift.DriftMeasure`
refuses a pair whose windows differ in length by more than a factor of two, and it cannot do
that unless each reading carries the window of *its own* read. The value is stored exactly as
it was rendered and is never parsed back into numbers, for the reason
:class:`~tenbin.report.document.CorpusHeader` is built by copying rather than by hand: a
reading reconstructed from its own text is a transcription, and transcriptions are where
quietly-wrong numbers come from.

**One ``(slug, read_at)`` is one reading, and a second one is refused rather than appended.**
Two entries for the same measure at the same moment would be two values for one observation,
and every drift computed across them would be a change that never happened -- the sort of
thing that looks like a finding and is an artefact of a re-run. :exc:`DuplicateReadingError`
names the pair so an operator can see which read was recorded twice.

**What this module notably does not do:** it does not measure, and it cannot compute
anything. :meth:`Archive.record` takes a finished :class:`~tenbin.report.document.Report` and
copies out of it; it never re-derives a figure, and a stored value is not a claim that the
figure is still what the measure would produce today. It does not store refusals, because a
refusal is not a figure and ``tenbin claims`` prints the catalogue -- an archive of refusals
would be a second copy of it, and copies of that catalogue are the defect the completeness
test exists to catch. It does not rewrite, prune, compact or expire anything: there is no verb
for it, and a bounded archive is a different artefact with a different promise, which would
have to argue for itself.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from tenbin.claims.mechanism import require_text
from tenbin.corpus.window import CorpusWindow
from tenbin.measures.base import Figure
from tenbin.measures.drift import Reading
from tenbin.report.document import Report

#: The shape of one line of the log. A string rather than an integer because the number is
#: written once and read by people as well as by this program, and ``"1"`` survives being
#: pasted into a ticket where ``1`` reads as a count of something. Bump it for any change to
#: what a stored reading *means* -- a new field, a different window encoding, a change to what
#: ``enumerated`` counts -- and not for a change of formatting.
#:
#: **A line at another version is refused rather than skipped**, following
#: :func:`tenbin.readmodel.load`: a file this program cannot interpret is not a corrupt file,
#: it is a version skew a human has to resolve, and quietly ignoring the odd line would drop
#: a reading from the middle of a drift and report a change that never happened.
ARCHIVE_VERSION: Final[str] = "1"

#: The keys a line is written under, named so the vocabulary is written down in one place
#: rather than spelled at the two ends of the round trip, where a typo is a missing reading
#: rather than an error.
VERSION_KEY: Final[str] = "version"
SLUG_KEY: Final[str] = "slug"
VALUE_KEY: Final[str] = "value"
COLLECTION_KEY: Final[str] = "collection"
READ_AT_KEY: Final[str] = "read_at"
ENUMERATED_KEY: Final[str] = "enumerated"
WINDOW_KEY: Final[str] = "window"


class DuplicateReadingError(RuntimeError):
    """Two readings were offered for one measure at one moment, and one was refused.

    ``RuntimeError`` rather than ``ValueError`` because nothing about either reading is
    malformed: both are well-formed facts, and what is wrong is that they cannot both be true
    of the same read. A ``ValueError`` message would invite somebody to fix one of the
    readings, and the fix is to not have run the second one.
    """

    def __init__(self, slug: str, read_at: datetime, collection: str = "") -> None:
        self.slug = slug
        self.read_at = read_at
        self.collection = collection
        where = f" in {collection}" if collection else ""
        super().__init__(
            f"{slug} is already archived for the read at {read_at.isoformat()}{where}. A "
            "measure read once produces one reading; two entries for one read would be a "
            "drift that never happened, so the second is refused rather than appended."
        )


class CorruptArchiveError(RuntimeError):
    """A line of the log could not be read, and the archive refuses rather than skips it.

    The alternative -- dropping a line this program does not understand -- is what turns a
    truncated write into a fabricated finding: two readings either side of the missing one
    would be compared as adjacent, and the gap would be reported as the change.
    """


@dataclass(frozen=True)
class Archive:
    """An append-only log of rendered figures, one JSON object per line.

    A frozen dataclass rather than a class with ``__init__`` so the path is the identity and
    two handles on the same file are equal, which is what lets a test compare them and what
    lets a caller pass the archive around without a way to swap the file underneath it. The
    file is still created lazily: opening an :class:`Archive` writes nothing, so constructing
    one is not an act that touches the filesystem.

    **The class is the argument as much as the file.** There are four verbs -- construct,
    :meth:`record`, :meth:`readings`, :meth:`slugs` -- and none of them updates or removes a
    line. A caller who wants history rewritten has to edit the file with a text editor, which
    is the loudest possible version of that decision and the one a reviewer can see.
    """

    path: Path

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise ValueError(f"Archive.path must be a Path, not {self.path!r}")

    def record(self, report: Report) -> tuple[Reading, ...]:
        """Append every figure in ``report``, and return what was written.

        One line per section holding a figure, stamped from the report's own corpus header
        rather than from anything the caller supplies -- the moment, the collection, the count
        and the window all come from the read that produced the figures, so a caller cannot
        file a figure under the wrong read and a drift over it will be a drift over the wrong
        pair.

        Sections holding a refusal are skipped, and deliberately: a refusal is not a figure,
        ``tenbin claims`` prints the catalogue, and an archive of refusals would be a second
        copy of a list that already has a completeness test against ``docs/seam.md``.

        **Appending is all-or-nothing across the report, and the check happens before the
        write.** Two runs of the same report against the same archive produce the same refusal
        twice rather than a file holding the second report's figures and not the first's --
        a half-written archive is a report that has been published and retracted in one
        gesture.

        The file is opened for appending and flushed per report. A crash mid-append leaves a
        truncated final line, which :exc:`CorruptArchiveError` will then refuse to read rather
        than skip: the truncated report is caught on the next run instead of quietly shaping
        the next drift.
        """
        header = report.corpus
        window = header.window if header.window is not None else UNKNOWN_WINDOW
        readings = tuple(
            Reading(
                slug=section.claim.slug,
                value=section.figure.value_text(),
                collection=header.collection,
                read_at=header.read_at,
                enumerated=header.enumerated,
                window=window,
            )
            for section in report.sections
            if isinstance(section.figure, Figure) and section.claim is not None
        )
        if not readings:
            return ()
        already = {
            (reading.collection, reading.slug, reading.read_at) for reading in self._all_readings()
        }
        for reading in readings:
            key = (reading.collection, reading.slug, reading.read_at)
            if key in already:
                raise DuplicateReadingError(reading.slug, reading.read_at, reading.collection)
            already.add(key)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            for reading in readings:
                stream.write(json.dumps(_as_payload(reading), sort_keys=False) + "\n")
            stream.flush()
        return readings

    def readings(self, slug: str, *, collection: str | None = None) -> tuple[Reading, ...]:
        """Every archived reading of ``slug`` in one collection, oldest first.

        Ordered by the moment of the read rather than by position in the file, because the
        file is append-only and an operator appending a backfill after the fact would
        otherwise invert the two readings of every drift computed over that measure. Ties on
        the moment cannot happen for one slug and one collection -- :meth:`record` refuses
        them -- so the order is total.

        **One collection, because a slug is not a fact on its own.** A measure called
        ``time-in-state`` over ticket transitions and a measure of the same name over
        documents would share a slug and hold figures in different units, and a drift
        between them would compare a duration in hours to whatever the other one counts.
        That is a category error the archive cannot detect from the figures, because the
        figures are stored as rendered text and never parsed back -- so the scoping has to
        happen here, where the collection is still known.

        ``collection=None`` is permitted and resolves only when the slug lives in exactly
        one collection, which is the whole of the single-source case. When it lives in more
        than one, this **raises** rather than returning every reading: an archive that
        answers an ambiguous question by picking one is worse than one that asks, and a
        drift computed over the union would be a real number about nothing.

        A slug with nothing archived returns an empty tuple rather than raising: "no reading"
        is the answer to "what does the archive hold", and a caller asking it is a drift
        command that should say so rather than a programme that has crashed.
        """
        require_text("Archive.readings", "slug", slug)
        matching = [reading for reading in self._all_readings() if reading.slug == slug]
        if collection is not None:
            require_text("Archive.readings", "collection", collection)
            return tuple(reading for reading in matching if reading.collection == collection)

        collections = {reading.collection for reading in matching}
        if len(collections) > 1:
            listed = ", ".join(sorted(collections))
            raise ValueError(
                f"{slug} is archived from more than one collection ({listed}), and a reading "
                "is a fact about one of them. Pass collection= to say which, because a drift "
                "computed over readings from different sources compares two different things."
            )
        return tuple(matching)

    def slugs(self) -> tuple[str, ...]:
        """Every measure the archive holds a reading of, in alphabetical order.

        Sorted rather than in first-seen order so two archives of the same readings list the
        same things, which is what makes the list usable in a ``diff`` and in a test.
        """
        return tuple(sorted({reading.slug for reading in self._all_readings()}))

    def _all_readings(self) -> tuple[Reading, ...]:
        """Every reading in the file, oldest first, refusing anything unreadable.

        Read whole on every call rather than kept in memory. The file is a log that outlives
        the process that wrote it, and a handle that cached a view of it would report a stale
        archive to the second caller in the same run -- which is the moment a reader stops
        being able to tell which of two archives a figure came from.
        """
        return tuple(sorted(self._iter_readings(), key=lambda item: (item.read_at, item.slug)))

    def _iter_readings(self) -> Iterator[Reading]:
        """Yield each reading in file order, naming the line number on anything malformed."""
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return
        for number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                # A blank line is a line somebody's editor left behind rather than a reading
                # that failed to parse, and refusing one would make a stray newline the end of
                # the programme's history.
                continue
            try:
                yield _reading_from_payload(json.loads(line))
            except (ValueError, TypeError, KeyError) as exc:
                raise CorruptArchiveError(
                    f"{self.path} line {number} could not be read as a reading ({exc}). The "
                    "archive is refused rather than truncated: a reading skipped here is a gap "
                    "that the drift across it would report as the change."
                ) from exc


#: The window stored for a report whose header has none. A hand-written header has no window to
#: copy, and the alternative -- omitting the field -- would restore as an archive with a hole
#: in it that reports itself as complete. This is the same fallback
#: ``CorpusHeader._enumeration_text`` makes when it renders, for the same reason: the honest
#: statement is that the window is unknown, not that the archive forgot to look.
UNKNOWN_WINDOW: Final[CorpusWindow] = CorpusWindow(
    earliest=None, latest=None, unreadable_timestamps=0
)


def _as_payload(reading: Reading) -> dict[str, Any]:
    """One reading as JSON-ready values, in the order a reader of the file meets them.

    Not sorted: the key order is the reading order, so a person opening the archive -- which
    is a thing this file invites, because it is a log and logs get read -- meets *what* was
    measured before *when* it was measured before *over what span*, which is the order those
    three questions get asked in.
    """
    return {
        VERSION_KEY: ARCHIVE_VERSION,
        SLUG_KEY: reading.slug,
        VALUE_KEY: reading.value,
        COLLECTION_KEY: reading.collection,
        READ_AT_KEY: reading.read_at.isoformat(),
        ENUMERATED_KEY: reading.enumerated,
        WINDOW_KEY: {
            "earliest": None
            if reading.window.earliest is None
            else reading.window.earliest.isoformat(),
            "latest": None if reading.window.latest is None else reading.window.latest.isoformat(),
            "unreadable_timestamps": reading.window.unreadable_timestamps,
        },
    }


def _reading_from_payload(payload: Any) -> Reading:
    """One reading back from its stored values, refusing anything that is not one.

    Every failure is a :class:`ValueError`, :class:`TypeError` or :class:`KeyError`, which
    :meth:`Archive._iter_readings` turns into a :exc:`CorruptArchiveError` naming the line. The
    version is read before anything else because it is the field that says whether the rest of
    the line means what this program thinks it means.
    """
    if not isinstance(payload, dict):
        raise ValueError("an archived reading must be an object")
    version = _text(payload.get(VERSION_KEY), VERSION_KEY)
    if version != ARCHIVE_VERSION:
        raise ValueError(
            f"the archive holds a reading at version {version!r} and this program writes "
            f"{ARCHIVE_VERSION!r}"
        )
    window = payload.get(WINDOW_KEY)
    if not isinstance(window, dict):
        raise ValueError("an archived reading must carry its window")
    read_at = _moment(payload.get(READ_AT_KEY), READ_AT_KEY)
    return Reading(
        slug=_text(payload.get(SLUG_KEY), SLUG_KEY),
        value=_text(payload.get(VALUE_KEY), VALUE_KEY),
        collection=_text(payload.get(COLLECTION_KEY), COLLECTION_KEY),
        read_at=read_at,
        enumerated=_count(payload.get(ENUMERATED_KEY), ENUMERATED_KEY),
        window=_window_from_payload(window),
    )


def _window_from_payload(payload: dict[str, Any]) -> CorpusWindow:
    """The stored window, read back rather than re-derived from nothing.

    Both ends or neither, which :class:`~tenbin.corpus.window.CorpusWindow` enforces itself;
    the only thing added here is the refusal of a naive moment, so the error names the archive
    rather than the corpus type. A stored age would have been the alternative and is not here
    for the reason that module gives: an age is a claim about *now*, held by a statement about a
    past read, and :meth:`CorpusWindow.age_at` takes the moment instead.
    """
    unreadable = _count(payload.get("unreadable_timestamps"), "window unreadable_timestamps")
    return CorpusWindow(
        earliest=_optional_moment(payload.get("earliest"), "window earliest"),
        latest=_optional_moment(payload.get("latest"), "window latest"),
        unreadable_timestamps=unreadable,
    )


def _text(value: Any, field: str) -> str:
    """A non-blank string, or a refusal naming the field it was reading."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"an archived reading must hold a non-blank {field}")
    return value


def _count(value: Any, field: str) -> int:
    """A non-negative count, refusing booleans, which are integers in Python."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"an archived reading must hold a non-negative count for {field}")
    return value


def _moment(value: Any, field: str) -> datetime:
    """An aware moment, refusing a naive one rather than assuming UTC.

    A naive timestamp in the log is a line this program did not write, and defaulting it to
    UTC would place a read at the right time by accident -- which is the failure this whole
    package is shaped to prevent, made silently in the one field whose job is to say when
    something was read.
    """
    moment = _text(value, field)
    parsed = datetime.fromisoformat(moment)
    if parsed.tzinfo is None:
        raise ValueError(f"an archived reading must hold an aware {field}, not {moment!r}")
    return parsed.astimezone(UTC)


def _optional_moment(value: Any, field: str) -> datetime | None:
    """A moment that may be absent, but not one that is present and unplaceable."""
    if value is None:
        return None
    return _moment(value, field)
