# 002 — No causal claims, structurally

**Status:** accepted
**Date:** 2026-09-30

## The decision

Tenbin will not render a comparative effectiveness claim — "agentic development
is more effective than human development" — unless the corpus contains an
assignment mechanism, and the corpus does not and will not. The refusal is
structural: a field in the claim registry, defaulted unset, and a report layer
that refuses to render the comparison when it is unset.

## Why

Observational data on which principal happened to write a change cannot support a
counterfactual. Agents get the well-specified tickets; humans get the rest. The
selection is not noise to be averaged over — it is the entire mechanism by which
the two groups differ, so a difference in outcome between them is uninterpretable
without knowing what each group was given.

Kojutsu already predicts this failure one level down. `docs/github-seam.md:179`
states that an unattended loop built on agreeable answers "manufactures
consensus", and the whole answerer prompt is written to fight it: disagreement is
the expected shape of a useful answer, agreement is the outcome to be suspicious
of. A measurement layer that ranks agent output against human output is the same
loop with a scoreboard attached, and the failure arrives faster because a score
is legible in a way a review is not.

The reasoning is already in the corpus. `docs/design-review/rationale.md:120`
records what would make the whole rationale programme unnecessary, and it is that
storing agent self-assertions is judged too much. A self-asserted rationale is
not a correctness check. A score derived from self-asserted rationales is not an
effectiveness measurement, and it is more confident-sounding than either.

## What is refused

- Any comparison of outcomes between principal types, including agent versus
  human, and including a model-versus-model comparison.
- Any statement of the form "capturing knowledge improved X", which requires a
  counterfactual the deployment does not have.
- Any per-principal ranking. Not because it is methodologically weak — it is
  methodologically weak *and* it is a different product with different consent,
  retention, and access requirements. Granularity defaults to area and team.

## What is permitted

Descriptive aggregates, always with their denominator: how many changes have
captures, how many do not, what the independence distribution looks like, how
long decision requests waited before reaching a terminal state. These are real and
they are useful, and none of them requires a counterfactual.

A comparison between two *conditions within a controlled run* is also permitted —
the five-agent matrix in `scripts/pilot_demo.py` is one, and it is a demo rather
than a sample. A single controlled run is an existence proof, not a result, and
any report built on one has to say so.

### Amended 2026-09-30 — a measure against its own past value

**One population read twice is not two populations.** `tenbin drift` renders a
measure's current value beside a reading it archived earlier, and that is not the
refusal above. There is no arm to assign a principal to and no selection between
groups, so there is no counterfactual to supply: the corpus changing between last
week and this week is not a treatment, it is the same population later. Nothing
about it needs randomising.

`ClaimKind.temporal` is the kind that says so, and
`tenbin.claims.gate.compares_principals` is the rule that keeps it out of the
requirement — the rule keys on the declared kind and on nothing else, and the test
that covers it is named for the distinction rather than for the behaviour. The
refusal itself now says what it is the absence of: an *allocation between groups*,
not comparison as such.

**Why this amendment was written down rather than left in the code.** The risk here
is asymmetric. If this record had stayed silent, the next person to read it would
conclude that any comparison is refused, decline to build the most useful thing
this program could offer, and nobody would find out for a year. A record that says
*this comparison is permitted and that one is refused* is checkable in both
directions.

**What a drift still cannot escape.** Its own falsifier: two reads separated by a
*documented* corpus event — a migration, a reinstalled webhook — did not differ
because time passed, and `tenbin.measures.drift.DriftMeasure` refuses those pairs
rather than reporting them as movement. Tenbin cannot observe a reinstall, so
`--corpus-event` asks the caller rather than inferring it, and a caller who guesses
wrong gets a refusal they did not want.

## How the refusal is enforced

Not by documentation. A limitation in a docstring is a comment; a limitation in a
test survives the next refactor and the next person who upgrades a word like
"verified" by one degree. This is the practice `docs/design-review/README.md:22`
records, and it is the reason the three claims in `rationale.md` have
corresponding tests in `tests/test_provenance.py`.

The test that matters is a negative test with a clean fixture: a dataset with a
large, unambiguous, genuinely comparative signal must still be refused. A guard
that only fires on messy data is a speed bump that a sufficiently confident caller
walks around.

## What would make this wrong

- If a real assignment mechanism becomes available — randomised allocation of
  changes to human and agent paths, or a natural experiment with a defensible
  instrument — then the corpus contains a counterfactual and the refusal is no
  longer the honest position. The gate exists to make that a deliberate act rather
  than a slow erosion.
- If the descriptive measures turn out to be decision-relevant on their own, which
  would mean the need for a causal claim was a framing error rather than a
  requirement.
