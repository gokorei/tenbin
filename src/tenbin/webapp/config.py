"""Which stores the web app lists, and the credentials it holds for them.

**This is not :mod:`tenbin.config`, and the separation is not tidiness.** That module
describes one store the command line measures; this one describes *several* stores that
a browser is shown side by side, and the second is a different problem in three ways
that all of them are the same problem. A name is needed, because a list of URLs is not a
list of anything a reader can hold in their head. A key is optional, because the local
Tanseki on 8099 takes none and a configuration that demanded one would refuse the store
everybody actually develops against. And the whole file is loaded from a path rather
than the environment, because ``TENBIN_`` names one store and a second store under the
same prefix would be a prefix whose meaning depends on which command was run.

So this module holds a list, and :class:`~tenbin.config.Settings` still holds the
description of any one of them. **The key is carried inside the ``Settings`` and never
outside it**, which is why :class:`StoreEntry` writes its own ``__repr__``: pydantic's
renders every field, so the default repr of this dataclass would have printed
``tanseki_api_key='hunter2'`` into any log line or traceback that touched an entry. An
entry that reprs itself into a log is a credential published to whoever reads the log,
and a failure message here is a thing that gets logged.

**A key is also removed from every string it could otherwise reach**, by
:meth:`StoreEntry.redact`. That is belt to the braces of a design that never renders the
settings in the first place, and the reason for it is that a defence which depends on
every future renderer remembering is not a defence. A store error message is assembled
from an exception's text, an exception's text is assembled by a transport library, and
a transport library is exactly the kind of component that decides one day to quote a
request header in an error. :meth:`redact` is what makes the guarantee structural rather
than a habit.

**Unknown keys are refused rather than ignored.** ``Settings`` sets ``extra="ignore"``,
which is right for a configuration read from an environment a user's shell also writes
to -- there, an unknown ``TENBIN_`` variable is somebody else's business. Here every
key in the file is this module's business, and ``api_kye:`` is a typo that would mean
"this store needs no key", which surfaces an hour later as an authentication failure
whose message says the credential is wrong. A refusal at load time names the key it did
not recognise.

**What this module notably does not do:** it does not read a store, does not know what a
report is, and does not decide what a page shows -- :mod:`tenbin.webapp.read` walks,
:mod:`tenbin.webapp.pages` renders. It does not resolve ``api_key_env`` from a file it
was told to trust and quietly carry the value around: the *name* of the variable is
resolved here, the value is read once into the ``Settings`` and nowhere else. And it does
not fall back to a default store, for the reason :mod:`tenbin.config` gives at greater
length: a measurement program with a default store is a measurement program that will
happily report a real number about the wrong corpus, and here it would do it over HTTP
to whoever loaded the page.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import yaml
from pydantic import ValidationError

from tenbin.config import DEFAULT_COLLECTION, Settings, settings_from_env

#: The keys a store entry may carry. A closed set, because an unknown one is a typo and
#: a typo in this file is a credential that silently does not get sent.
STORE_KEYS: Final[frozenset[str]] = frozenset(
    {"name", "url", "collection", "api_key", "api_key_env", "timeout_seconds"}
)

#: The top-level key holding the list, when the file is a mapping rather than a bare
#: list. Both shapes are accepted because a file that may later want a ``defaults:``
#: block is a file whose author will reach for a mapping, and refusing it now would mean
#: a breaking change to a configuration file in somebody's repository later.
STORES_KEY: Final[str] = "stores"

#: How long a store name may be. It goes into a URL path and into a log line, and this
#: program does not take unbounded operator-supplied strings into either.
MAX_STORE_NAME_LENGTH: Final[int] = 64

#: A store's read timeout when the file does not name one. The same value and the same
#: bounds as :class:`~tenbin.config.Settings` applies, restated rather than read back
#: out of the model's field metadata: introspecting a default is a way to be wrong
#: quietly, whereas a constant here is one edit in two files at worst and the test suite
#: is what would notice.
DEFAULT_TIMEOUT_SECONDS: Final[float] = 10.0

#: What a removed credential is replaced with. A visible marker rather than a blank or
#: a truncation, so that a redaction is something a reader can see happened -- a
#: cause reading "... with key [redacted] failed" tells an operator the message was
#: scrubbed, where an empty gap would read as a sentence that never had a key in it.
REDACTED: Final[str] = "[redacted]"


class StoreListError(ValueError):
    """The store list could not be read, or a store in it is unusable.

    A ``ValueError`` because this is a mistake in a file rather than a failure of the
    world the program runs in -- the distinction
    :class:`~tenbin.store.client.StoreError` draws for transport problems, and this one
    is deliberately on the other side of it. A configuration file that names a store
    wrongly is wrong before the program starts, and a message about it must not read as
    though a store went down.

    The message names the file and the store and never the value: the same rule
    :mod:`tenbin.config` follows for the URL, and for the same reason, because a URL or
    a key is exactly what gets pasted into this file.
    """


def first_line(exc: BaseException) -> str:
    """The first line of an exception's message, for a one-line failure report.

    A pydantic ``ValidationError`` prints every field, every constraint and a link to
    its documentation, which is the right thing in a traceback and the wrong thing on a
    page. This is the same helper :mod:`tenbin.cli` keeps private, restated rather than
    imported: reaching into the command line from a web app would put Typer in the
    import path of a module whose entire job is to build strings, and the function is
    four lines.
    """
    text = str(exc)
    return text.splitlines()[0] if text else type(exc).__name__


@dataclass(frozen=True, repr=False)
class StoreEntry:
    """One store the app lists: the operator's label for it, and how to read it.

    The name is separate from the settings on purpose. A ``Settings`` describes a
    collection at a URL, and a page that showed it would be publishing an operator's
    internal infrastructure to whoever loaded the URL -- a host name is frequently the
    sensitive half of that sentence -- while the name is the label the same operator
    chose for the thing they want a colleague to recognise. So the name is what every
    page renders, and it is the one field here that was written for a reader.

    ``source`` is where the entry came from, kept so a failure message can name the file
    to edit. It is not part of the store's identity for rendering, and it is empty for
    an entry built by hand, which is why :attr:`origin` exists to give the message a
    phrase that is true in both cases.

    ``repr=False`` on the dataclass, because the generated one prints every field of
    ``settings`` and that includes the API key. See the module docstring.
    """

    name: str
    settings: Settings
    source: str = ""

    def __post_init__(self) -> None:
        normalized = self.name.strip() if isinstance(self.name, str) else ""
        if not normalized:
            raise StoreListError("a store must have a name; a store list is a list of names.")
        if len(normalized) > MAX_STORE_NAME_LENGTH:
            raise StoreListError(
                f"the store name {normalized!r} is longer than {MAX_STORE_NAME_LENGTH} "
                "characters; a name goes into a URL path and a log line."
            )
        for character in normalized:
            # A name becomes a path segment, and a name carrying a slash would address a
            # different store than the one it names. Whitespace and control characters are
            # refused for the same reason plus legibility: percent-encoding them works and
            # produces a link no operator will ever type or read.
            if character == "/" or character.isspace() or not character.isprintable():
                raise StoreListError(
                    f"the store name {normalized!r} may not contain whitespace, control "
                    "characters or '/', because it is also the URL path that addresses it."
                )
        object.__setattr__(self, "name", normalized)

    def __repr__(self) -> str:
        """The name and the collection, and neither the URL nor the key.

        Deliberately not the generated repr. A traceback prints the locals it unwinds
        through, so an entry that reprs its settings has put an API key into a stack
        trace, and a stack trace is the single most-shared artefact in a bug report.
        """
        return f"StoreEntry(name={self.name!r}, collection={self.settings.tanseki_collection!r})"

    @property
    def has_api_key(self) -> bool:
        """Whether this store was given a credential.

        A property so the question is asked rather than inferred from a truthiness test
        on a settings field -- and because the answer is *not* rendered on a page. A page
        that said "authenticated" would be telling a reader about a secret's presence,
        which is less than the secret and more than they need.
        """
        return bool(self.settings.tanseki_api_key.strip())

    @property
    def origin(self) -> str:
        """Where to go and change this store, as a phrase a message can quote."""
        return self.source or "the webapp configuration"

    def redact(self, text: str) -> str:
        """Return ``text`` with this store's API key removed, if it has one.

        The key is matched literally rather than by pattern, so the cost is one scan of
        whatever text is being rendered and the behaviour is total: there is no shape of
        key this could fail to catch and no shape of text it could corrupt, because it
        only ever deletes one exact string. A store with no key returns its input
        unchanged, which is the case that matters most -- the local Tanseki on 8099 takes no
        key, and a redactor that refused to pass text through when there was nothing to
        redact would break exactly the store a developer runs first.
        """
        key = self.settings.tanseki_api_key
        return text.replace(key, REDACTED) if key else text


def load_stores(path: Path) -> tuple[StoreEntry, ...]:
    """Read the store list at ``path``, in configuration order.

    Order is preserved rather than sorted, because the order in the file is the operator
    saying which store they look at first, and a list that reorders itself between two
    loads is a list whose failure entries move.

    ``yaml.safe_load`` and never ``yaml.load``: this parses a file, and the safe loader
    is the one that constructs no Python objects out of the text. A store list is
    operator-supplied input like any other, and the loader that can execute a tag is the
    wrong one for anything a person might edit.

    **An empty file is an empty tuple, and the page says so.** That is not the same as a
    page listing zero healthy stores, and the renderer is careful about the difference:
    a configuration nobody wrote yet must not render as a deployment where every store
    has nothing to say.
    """
    source = str(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise StoreListError(
            f"could not read the store list at {source} ({type(exc).__name__}); "
            f"nothing was listed and nothing was read."
        ) from exc
    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise StoreListError(f"{source} is not valid YAML ({first_line(exc)}).") from exc

    rows = _rows_from(payload, source=source)
    entries = tuple(_entry_from(row, source=source) for row in rows)
    _refuse_duplicate_names(entries, source=source)
    return entries


def _rows_from(payload: Any, *, source: str) -> Sequence[Any]:
    """The store rows in the file, whichever of the two accepted shapes held them."""
    if payload is None:
        return ()
    if isinstance(payload, Mapping):
        rows = payload.get(STORES_KEY)
        if rows is None:
            raise StoreListError(
                f"{source} is a mapping with no {STORES_KEY!r} key; it should hold "
                f"{STORES_KEY!r}: a list of stores."
            )
    elif isinstance(payload, list):
        rows = payload
    else:
        raise StoreListError(
            f"{source} must hold a list of stores, or a mapping with a {STORES_KEY!r} key; "
            f"it holds a {type(payload).__name__}."
        )
    if not isinstance(rows, list):
        raise StoreListError(f"{source} holds {STORES_KEY!r}, which is not a list of stores.")
    return rows


def _entry_from(row: Any, *, source: str) -> StoreEntry:
    """One store row as an entry, refusing a row that cannot be one.

    The row is validated before it becomes a ``Settings``, and the ``ValidationError``
    that comes back is turned into a :class:`StoreListError` naming the store: pydantic's
    own message would say ``tanseki_url`` to a reader who wrote ``url``, and this program's
    reason for hiding its input is that the input is where a credential gets pasted.
    """
    if not isinstance(row, Mapping):
        raise StoreListError(f"{source} holds a store that is not a mapping: {row!r}")
    unknown = sorted(set(row) - STORE_KEYS)
    if unknown:
        raise StoreListError(
            f"{source} names a store with the unrecognised key(s) {', '.join(unknown)}. "
            f"A store may carry only: {', '.join(sorted(STORE_KEYS))}."
        )
    name = _text(row, "name", source=source)
    url = _text(row, "url", source=source)
    collection = _text(row, "collection", source=source, required=False) or DEFAULT_COLLECTION
    api_key = _api_key(row, source=source, name=name)
    timeout = _timeout(row, source=source, name=name)
    try:
        settings = settings_from_env(
            tanseki_url=url,
            tanseki_api_key=api_key,
            tanseki_collection=collection,
            tanseki_timeout_seconds=timeout,
        )
    except ValidationError as exc:
        raise StoreListError(
            f"the store {name!r} in {source} is misconfigured ({first_line(exc)}); "
            "the URL must be HTTP(S), over TLS unless it targets loopback, and the "
            "collection must not be empty."
        ) from exc
    return StoreEntry(name=name, settings=settings, source=source)


def _api_key(row: Mapping[str, Any], *, source: str, name: str) -> str:
    """The credential for a store: literal, named, or none at all.

    ``api_key_env`` names a variable rather than holding a secret, and it is the
    recommended spelling for the same reason ``TENBIN_TANSEKI_API_KEY`` is quoted rather
    than echoed in :mod:`tenbin.cli`: a configuration file is a thing that gets
    committed. **An unset variable is a refusal, not an empty key.** Treating it as
    empty would turn a typo in a variable name into an authentication failure an hour
    later, and one whose message says the credential is wrong rather than that the
    variable was never set.
    """
    literal = _text(row, "api_key", source=source, required=False)
    named = _text(row, "api_key_env", source=source, required=False)
    if literal and named:
        raise StoreListError(
            f"the store {name!r} in {source} carries both api_key and api_key_env; "
            "a credential is one of them."
        )
    if not named:
        return literal
    value = os.environ.get(named)
    if not value:
        raise StoreListError(
            f"the store {name!r} in {source} names the variable {named} for its API key, "
            f"and {named} is unset or empty. Set it, or remove api_key_env for a store "
            "that takes no credential -- the local Tanseki on 8099 takes none."
        )
    return value


def _timeout(row: Mapping[str, Any], *, source: str, name: str) -> float:
    """The store's read timeout, or the settings default when the file names none."""
    if "timeout_seconds" not in row or row["timeout_seconds"] is None:
        return DEFAULT_TIMEOUT_SECONDS
    raw = row["timeout_seconds"]
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise StoreListError(
            f"the store {name!r} in {source} has a timeout_seconds that is not a number: {raw!r}."
        ) from exc


def _text(row: Mapping[str, Any], key: str, *, source: str, required: bool = True) -> str:
    """One field of a store row as a stripped string.

    A present-but-wrong-typed value is refused rather than stringified, for the reason
    ``StoreDocument.from_json`` refuses one: tolerance is for the spelling of a value
    that varies, not for a value of a type nobody meant.
    """
    value = row.get(key)
    if value is None:
        if required:
            raise StoreListError(f"{source} holds a store with no {key}.")
        return ""
    if not isinstance(value, str):
        raise StoreListError(f"{source} holds a store whose {key} is a {type(value).__name__}.")
    return value.strip()


def _refuse_duplicate_names(entries: Sequence[StoreEntry], *, source: str) -> None:
    """Refuse two stores sharing a name, because a name is this app's only address.

    ``/store/<name>`` is the whole routing scheme, so two entries under one name would
    make the index show two identical labels where one of them is unreachable, and the
    reader would have no way to tell which is which. That is the exact failure this app
    exists to make impossible, reached by a different road.
    """
    seen: set[str] = set()
    for entry in entries:
        if entry.name in seen:
            raise StoreListError(
                f"{source} names the store {entry.name!r} twice. Names are how a store is "
                "addressed and how it is told apart from another one, so they must differ."
            )
        seen.add(entry.name)
