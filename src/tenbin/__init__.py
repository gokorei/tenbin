"""Tenbin -- a measurement program built on the records Kojutsu captures.

Tenbin reads; it does not collect. Kojutsu owns capture and provenance, and
owns the argument about what a record is worth. Tenbin owns measurement and
claims: what the corpus supports saying, what it does not, and what would
falsify each thing it says. The seam between them is one-way and it is a read
seam -- Tenbin never writes to Kojutsu, and Kojutsu never calls Tenbin.

**A number without a claim attached is a claim.** That sentence is the design,
and it is load-bearing in a way that is easy to state and hard to keep. A
dashboard renders numbers, and a number rendered without its caveat is a claim
made silently. So every figure this program produces is bound to a claim, and a
claim names four things: the statement, what it does not mean, what observation
would falsify it, and the denominator the figure is a rate over. The types are
shaped so that the unclaimed form cannot be constructed at all, rather than
merely discouraged by convention, because a convention does not survive the next
refactor or the next person who upgrades a word like "verified" by one degree.

**Every measure here is descriptive, and the refusal is structural.** The corpus
is observational, and the principal that happened to write a change was not
assigned to write it -- agents get the well-specified tickets, and that selection
is the entire mechanism by which the two groups differ. So no comparative
effectiveness claim is renderable: not cautioned, not hedged, refused. See
``docs/decisions/002-no-causal-claims.md``. A limitation recorded in a docstring
is a comment; the limitation here lives in a field defaulted unset and in a
negative test over a clean fixture.

**The corpus is self-selected and self-censored, and that is not repairable
here.** A repository where nobody answers is byte-identical to a repository with
no knowledge debt, and no backfill path exists. What *is* repairable is
legibility -- making the bias visible, so a rate is reported as a rate over a
stated population rather than as a number implying representativeness it does not
have. Until a census of observed-but-uncaptured changes lands, this program
reports counts and says so.
"""

__version__ = "0.1.0"
