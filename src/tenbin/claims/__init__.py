"""Claim protocol: the types that make a number-without-a-claim unconstructible.

Tenbin's central claim is that a number without a claim attached is a claim, and
that only holds if the unclaimed form cannot be built. So the package is shaped
around constructors and checks rather than conventions: a claim supplies its
caveats or the constructor raises naming the field, a comparative claim cannot be
rendered without naming the mechanism that produced the difference, a measure the
corpus cannot support has a refusal that says what is missing and what would
change the answer, and the gate raises for a caller who forgets to ask first.

The refusals are listed as data rather than derived from the schema, because
``docs/seam.md`` already states them and a second copy that disagreed with the
first would be a defect nobody would find. A test in ``tests/test_refusal_registry``
parses that table and asserts every measure it marks as blocked has an entry
here, so the two cannot drift apart silently -- and the parse refuses a table
shape it does not recognise, so a renamed heading cannot turn the completeness
check into a test that passes on nothing.

**What this package notably does not do:** it does not compute anything, and it
does not read the store. :mod:`tenbin.claims.model` renders a claim with no
figure attached precisely so the claim can be tested without one, and the store
and corpus layers are somebody else's problem until there is a figure to attach.
Nor does it decide what a claim should be -- the catalogue records what the
corpus forecloses, and a claim that is not in the catalogue and not comparative
is not refused, because a program that refuses everything would be correct and
useless, and uselessness is not a way to be safe.
"""

from tenbin.claims.gate import ClaimGate
from tenbin.claims.mechanism import (
    ASSIGNMENT_MECHANISM_MISSING_FACT,
    GRANULARITY_ORDER,
    MECHANISM_ABSENT_REASON,
    MINIMUM_GRANULARITY,
    NO_ASSIGNMENT_MECHANISM,
    AssignmentMechanism,
    Granularity,
)
from tenbin.claims.model import Claim, ClaimKind, Denominator
from tenbin.claims.refusals import (
    CATALOGUE,
    NO_TICKET_UNBLOCKS_THIS,
    ComparativeRefusalError,
    Refusal,
    format_unblocked_by,
    refusal_for_absent_mechanism,
    refusal_for_measure,
    slugify,
)
from tenbin.claims.registry import RefusalRegistry, default_registry

__all__ = [
    "ASSIGNMENT_MECHANISM_MISSING_FACT",
    "CATALOGUE",
    "GRANULARITY_ORDER",
    "MECHANISM_ABSENT_REASON",
    "MINIMUM_GRANULARITY",
    "NO_ASSIGNMENT_MECHANISM",
    "NO_TICKET_UNBLOCKS_THIS",
    "AssignmentMechanism",
    "Claim",
    "ClaimGate",
    "ClaimKind",
    "ComparativeRefusalError",
    "Denominator",
    "Granularity",
    "Refusal",
    "RefusalRegistry",
    "default_registry",
    "format_unblocked_by",
    "refusal_for_absent_mechanism",
    "refusal_for_measure",
    "slugify",
]
