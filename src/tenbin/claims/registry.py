"""The refusal catalogue as a lookup, and the mechanism state that licenses a comparison.

Two things live here because they are the same question asked twice. Which
measures are refused, and whether a comparative claim could be rendered at all --
both are statements about what this corpus can support, and a report needs both
before it decides what to put on a page. Keeping them in one object means a
caller cannot hold a catalogue from one place and a mechanism from another and
end up refusing a claim the catalogue says is fine, or permitting one the
mechanism says is not.

The mechanism defaults to absent, and the reason for its absence stays available
whether or not a mechanism is currently set. That is deliberate: the question the
reason answers is asked *after* somebody fills the field in, not only before, so
an accessor that stopped working once a mechanism existed would be answering a
question nobody was asking any more.

**What this module notably does not do:** it does not decide what to refuse. A
refusal registered here is taken at its word -- the registry is a container, and
a container that second-guessed its contents would make the catalogue untestable
and the rules unstated. Likewise it holds no granularity floor, because the floor
is a property of what a caller is willing to publish, and it belongs to
:mod:`tenbin.claims.gate` where that choice is made visibly.
"""

from __future__ import annotations

from collections.abc import Iterable

from tenbin.claims.mechanism import (
    MECHANISM_ABSENT_REASON,
    NO_ASSIGNMENT_MECHANISM,
    AssignmentMechanism,
)
from tenbin.claims.refusals import CATALOGUE, Refusal


class RefusalRegistry:
    """Refusals by measure slug, plus the mechanism state that gates comparisons.

    Registration order is preserved because the report has to list refusals in
    the order a reader can act on them, and a registry that returned them sorted
    alphabetically would make the ordering a decision nobody made explicitly.
    """

    def __init__(
        self,
        *,
        mechanism: AssignmentMechanism | None = NO_ASSIGNMENT_MECHANISM,
        refusals: Iterable[Refusal] = (),
    ) -> None:
        self._mechanism = mechanism
        self._refusals: dict[str, Refusal] = {}
        for refusal in refusals:
            self.register(refusal)

    @property
    def mechanism(self) -> AssignmentMechanism | None:
        """The assignment mechanism, or ``None`` when the corpus has none."""
        return self._mechanism

    @property
    def mechanism_unset_reason(self) -> str:
        """Why there is no assignment mechanism, whether or not one is set.

        Available unconditionally, and that is the design rather than an
        oversight. A registry that exposed this reason only while the field was
        empty would be answering "why is it empty" and not "what would it take" --
        and the second question is the one that arrives from whoever has just
        set the field and wants to check they were not supposed to.
        """
        return MECHANISM_ABSENT_REASON

    def register(self, refusal: Refusal) -> None:
        """Add a refusal, refusing to shadow one already registered for the slug.

        Overwriting would make a contradiction silent: two refusals for one
        measure, with different reasons or different unblocking tickets, and the
        second one simply winning. Since the refusals for the retrieval measures
        in particular rest on a two-part argument, a shadowed entry is how the
        second half would be lost without anything turning red.
        """
        existing = self._refusals.get(refusal.claim_slug)
        if existing is not None:
            raise ValueError(
                f"a refusal for {refusal.claim_slug!r} is already registered; "
                "two refusals for one measure is a contradiction, not an update"
            )
        self._refusals[refusal.claim_slug] = refusal

    def get(self, slug: str) -> Refusal:
        """Return the refusal for a measure slug, or raise if the measure is not refused.

        Raising rather than returning ``None``, because the caller asking this
        question is a report checking whether a measure is blocked, and a
        ``None`` it forgets to test is a measure rendered without its refusal --
        which is the failure this package exists to prevent, arrived at by the
        other route. Use ``slug in registry`` to ask without committing to an
        answer.
        """
        try:
            return self._refusals[slug]
        except KeyError:
            raise KeyError(
                f"no refusal is registered for {slug!r}; {len(self._refusals)} measures are refused"
            ) from None

    def all(self) -> tuple[Refusal, ...]:
        """Every registered refusal, in registration order."""
        return tuple(self._refusals.values())

    def __contains__(self, slug: object) -> bool:
        return slug in self._refusals

    def __len__(self) -> int:
        return len(self._refusals)


def default_registry() -> RefusalRegistry:
    """A registry holding the whole catalogue, with no mechanism set.

    The mechanism is left at the corpus's actual state rather than being a
    parameter, so the thing a caller gets by default refuses exactly the
    comparisons this corpus cannot support. A registry constructed with a
    mechanism is built deliberately, which is what decision 002 asks for: the
    refusal should be something somebody turns off on purpose, not something a
    test fixture or a demo quietly turns off by accident.
    """
    return RefusalRegistry(mechanism=NO_ASSIGNMENT_MECHANISM, refusals=CATALOGUE)
