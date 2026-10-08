"""Telling a stated value from a missing one, which the corpus cannot afford to blur.

Kojutsu writes the literal string ``"unknown"`` into ``answered_by_model`` and
``declared_by_model`` wherever a principal did not name a model. The key is
therefore present in every document of that shape, and the *value* is doing two
jobs at once: it is sometimes a model, and sometimes the record's own way of
saying it has none. Every measure that groups by model walks straight into that,
and the mistake it invites is a specific and expensive one -- reporting the
``unknown`` bucket as though it were a group of records whose authors declared a
model called ``unknown``, which is a claim about a principal that nobody made.
:func:`parse_stated` exists so that this program cannot make that mistake by
accident: it returns :class:`Unstated` for an absence, and a
:class:`Stated` wrapper for a value, and the two are different types.

**Do not "simplify" this away.** The tempting refactor is to drop the wrapper
and return ``str | None``, or worse, to return the raw value and let a caller
decide. Both deletions are invisible in a diff and both are unrecoverable: a
report grouping by ``answered_by_model`` would again show an ``unknown`` bucket
sitting beside the real models, and this time nothing in the code would say the
bucket is not a group. The distinction is the type, and the type is the only
thing that survives the next person who tidies a module.

There is a third case, and it is the reason a plain ``str | None`` would have
been tempting enough to try. A model could in principle be *named*
``"unknown"`` -- someone declares "the model is unknown" -- and this function
reports that as :class:`Unstated`, which is a wrong answer to a question that
cannot be asked apart. That is not a bug to be fixed by loosening the comparison;
it is a fact about the writer, and it is why the axis is lossy at the storage
boundary rather than at this one. Kojutsu chose the literal for exactly the
fields where the sentence had to be filled in, so the collision is unrecoverable
from the document. What is recoverable is that we know the collision happened,
and a caller holding :class:`Unstated` should say "the writer stated no model"
rather than "the model is unknown".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Final, Generic, Literal, TypeVar

#: Only string values are wrapped, because only strings survive the storage
#: boundary: Kojutsu stringifies every extra, so a model name is always text
#: by the time a reader sees it. Bounding the type parameter is what lets
#: :class:`Stated` be orderable without hand-written comparisons that would then
#: have to decide what to do with two different types.
T = TypeVar("T", bound=str)

#: The literal Kojutsu writes where a principal stated no model. Named here
#: rather than spelled at each comparison so that the string has exactly one
#: definition in this program, and so that a reader searching for it finds the
#: collision documented rather than a bare comparison.
UNSTATED_LITERAL: Final[str] = "unknown"


class Unstated:
    """The singleton meaning "the record says nothing here".

    One instance, ever. ``__new__`` returns the same object for every call, so
    ``is`` is a valid comparison and the value survives copying intact --
    :meth:`__copy__` and :meth:`__deepcopy__` return ``self`` rather than letting
    the default machinery build a second one, because two sentinels for one
    absence is precisely the class of defect this module exists to prevent.

    It is falsy, which makes ``if record.answered_by_model:`` the correct way to
    ask whether anything was stated, and gives a plain ``str | None`` an
    ergonomics argument it does not deserve.

    It does not compare equal to ``None``, ``""`` or ``False``. An absence of a
    model is a fact about the record; ``None`` is a fact about a field, and
    conflating them is how a program ends up asserting that a model was unknown
    when the truth is that nobody said.
    """

    __slots__ = ()

    _instance: ClassVar[Unstated | None] = None

    def __new__(cls) -> Unstated:
        # Refusing subclassing is not decoration: a subclass would share
        # ``_instance`` and silently become a second spelling of the same
        # sentinel, which is the exact failure the singleton exists to prevent.
        if cls is not Unstated:
            raise TypeError("Unstated is a sentinel and is not subclassable.")
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __bool__(self) -> Literal[False]:
        return False

    def __repr__(self) -> str:
        return "UNSTATED"

    def __copy__(self) -> Unstated:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> Unstated:
        return self


@dataclass(frozen=True, order=True)
class Stated(Generic[T]):
    """A value the record actually states, wrapped so it cannot be confused.

    Frozen, orderable and hashable on purpose. Orderable because sorting a
    record's models is a thing a report does, and hashable because a counter of
    stated values has to key on something; neither of those is a reason to let
    the raw string out, and both are reasons not to make this a one-field
    ``NamedTuple`` that a caller unpacks into a bare ``str`` at the first
    convenient line.
    """

    value: T

    def __bool__(self) -> Literal[True]:
        return True

    def __str__(self) -> str:
        return self.value


def parse_stated(raw: object) -> Stated[str] | Unstated:
    """Recover whether a frontmatter value is a stated name or an absence.

    ``"unknown"`` (in any case), ``None``, an absent key, a blank string and any
    non-string value all yield :class:`Unstated`. Everything else yields
    :class:`Stated`, stripped.

    The casefold comparison widens the match to a value a human typed, and it
    costs nothing that matters: a model genuinely named ``"Unknown"`` is already
    indistinguishable from Kojutsu's sentinel in the stored document, so
    reading it as an absence loses no information the writer preserved. A
    non-string yields :class:`Unstated` rather than a coerced ``str`` on purpose:
    a number or a list in this field is a document that states no model name, and
    stringifying it would manufacture a principal called ``"42"``.
    """
    if not isinstance(raw, str):
        return Unstated()
    candidate = raw.strip()
    if not candidate or candidate.casefold() == UNSTATED_LITERAL:
        return Unstated()
    return Stated(candidate)
