"""Whose changes this corpus holds evidence about, by the login GitHub reports.

Kojutsu's ticket ``3QRPK52A`` stopped discarding the account that opened the pull
request. The account arrived on every webhook payload, was used to compute
independence, and was then dropped -- so the store could not say whose change it was
holding evidence about, and every independence level was a label with no subject
attached.

**The login is not a person, not a bot, and not a statement about what they were
doing.** That is Kojutsu's own framing of the field and it is repeated in the
claim, because a reader who takes the field for any of the three will draw a
conclusion the store never licensed. A login is what the forge reports opened the
change; an organisation can have shared logins, a person can hold several, and a
login tells you nothing about whether the person was writing the change or
operating something that writes it.

**This measure does not unblock the agentic-effectiveness comparison, and the
reason is not a missing field.** The project's central question is refused because
the corpus holds no assignment mechanism: the principal that happened to write a
change was not assigned to write it, agents receive the well-specified tickets and
humans the rest, and that selection is the entire mechanism by which the two groups
differ. ``change_author_account`` names *who* wrote a change; it says nothing about
*what they were given*. A self-asserted identity field is not an assignment
mechanism, and a measure that treated it as one would be the specific way decision
002 gets lost -- quietly, by a ticket landing. So the field is used for what it is,
a login, and the refusal stays where it is. A test asserts that the gate still
refuses a comparative claim built on this axis.

**A record with no login is counted separately, not bucketed as a name.** An absent
``change_author_account`` means the payload carried no user, and the key is then not
written; putting those records under a placeholder would make the absence look like
a principal, which is the ``unknown``-model defect again and is the reason the
exclusion key here is a sentence.

**What this module notably does not do:** it does not count an agent and a person.
There is nothing in this field that says which kind of account opened a change, and
inferring it from the login's shape would be reading a string as a fact about an
account. It also does not compare two accounts, and the claim is at ``team``
granularity: a histogram of records per login is a description of the corpus, not a
claim about any login in it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from tenbin.claims.mechanism import Granularity
from tenbin.claims.model import Claim, ClaimKind, Denominator
from tenbin.corpus.snapshot import CorpusSnapshot
from tenbin.measures.base import DistributionFigure, Figure, histogram
from tenbin.measures.filtering import capture_population, requires_certainty

#: The slug this measure is cited by, and the ticket that unblocked it. Not a
#: comparison, so there is no refusal to retire: the ticket delivered the field and
#: nothing more, and the two facts are recorded separately for that reason.
AUTHORSHIP_SLUG: Final[str] = "corpus.change_authorship"

#: The ticket named on the claim.
UNBLOCKED_BY: Final[str] = "3QRPK52A"

#: The key for a record the forge reported no user for. A sentence, because a report
#: renders it and an operator acts on it, and ``"(unknown)"`` is a bucket somebody
#: will read as a principal.
NO_CHANGE_AUTHOR_KEY: Final[str] = (
    "no change-author account on the record (the payload named no user)"
)

#: What the population is, in the words the denominator renders.
AUTHORSHIP_POPULATION: Final[str] = (
    "capture records in this read, grouped by the account that opened the change"
)

#: Kojutsu's framing of the field, verbatim in substance, because a paraphrase
#: drops one of the three things it is not.
LOGIN_IS_NOT_A_PERSON: Final[str] = (
    "A change_author_account is the login GitHub reports opened the pull request. It is not a "
    "person, not a bot, and not a statement about what they were doing. An organisation can "
    "hold shared logins, a person can hold several, and a login says nothing about whether a "
    "person wrote a change or operated something that wrote it."
)


def authorship_claim(total: int) -> Claim:
    """The claim behind the authorship distribution, over the records in this read."""
    return Claim(
        slug=AUTHORSHIP_SLUG,
        statement=(
            "The changes this corpus holds records about were opened by the accounts distributed "
            "as the figure below shows."
        ),
        does_not_mean=(
            "That any of these accounts is a person, a bot, or more or less effective. "
            + LOGIN_IS_NOT_A_PERSON
            + " It also does not mean that an account absent from the figure opened no changes, "
            "and it is not a step toward comparing agents with people: this is a distribution of "
            "logins over records, and no account is compared with another or with anything else."
        ),
        falsifier=(
            "A change author login appearing in the records of a change the forge reports a "
            "different login for. Every writer of a record about one change should write the same "
            "login, so a disagreement would mean the grouping is over a field that does not mean "
            "one thing."
        ),
        denominator=Denominator(AUTHORSHIP_POPULATION, max(1, total)),
        kind=ClaimKind.descriptive,
        granularity=Granularity.team,
        unblocked_by=UNBLOCKED_BY,
    )


@dataclass(frozen=True)
class ChangeAuthorshipMeasure:
    """Records per change-author login, with the unnameable ones counted separately.

    Deliberately *not* a step toward the comparison decision 002 refuses, and the
    docstring says why in the terms the decision uses: a login is a self-asserted
    identity, and the comparison is refused for want of a mechanism that says who
    was given what. Naming the author is not that mechanism, so this measure reads
    the field and leaves the refusal alone.
    """

    requires_certainty = requires_certainty

    @property
    def slug(self) -> str:
        """The identifier this measure is cited by."""
        return AUTHORSHIP_SLUG

    def claim(self, snapshot: CorpusSnapshot) -> Claim:
        """The claim, over the capture records this read enumerated."""
        return authorship_claim(len(capture_population(snapshot.records)))

    def compute(self, snapshot: CorpusSnapshot) -> Figure:
        """The histogram of change-author logins, with the absent ones named and counted."""
        logins: list[str] = []
        absent = 0
        for record in capture_population(snapshot.records):
            login = record.change_author_account
            if login:
                logins.append(login)
            else:
                absent += 1
        return DistributionFigure(
            claim=self.claim(snapshot),
            snapshot=snapshot,
            title="Records per account that opened the change",
            values=histogram(logins),
            excluded={NO_CHANGE_AUTHOR_KEY: absent} if absent else {},
        )
