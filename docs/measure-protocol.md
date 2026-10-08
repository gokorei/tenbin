# A measure is four required fields and one honest denominator

This is the document to read before writing the next measure. It is written for
the person who has to do it, not for somebody being persuaded of the design: the
design is argued in the module docstrings and in
[`decisions/002-no-causal-claims.md`](decisions/002-no-causal-claims.md), and
re-arguing it here would be a third copy to keep true.

**The short version: a measure is a `Claim` plus a number, and neither half is
optional.** `Claim` has no defaults, so the compiler enforces the four fields
below rather than trusting a review to notice a missing caveat. What the compiler
cannot enforce is whether the denominator is a *population*, whether an excluded
record was counted, and whether the refusal that should have replaced this measure
is in the catalogue. Those three are where measures go wrong, and each has a rule
below.

## The four fields, and why none of them can have a default

`Claim` lives in `tenbin.claims.model` and every field is checked for blankness
in `__post_init__`. A missing or empty caveat raises a `ValueError` naming the
field. The reason each one is required rather than optional is different, and the
difference matters when you are deciding whether to weaken one.

- **`statement` — what is true, said so a reader can check it.** Required because a
  number with no statement attached is a claim made silently, and the reader is
  the only one who can tell. It is the only field that describes the figure at all.
- **`does_not_mean` — the adjacent claim this is not.** Required because the
  interesting error is never misreading a number; it is reading it as the nearest
  stronger thing. A figure about how many records carry a stated model is not a
  figure about how many records were written by a model, and only the second
  sentence stops the substitution. A field with a default here would render the
  same non-claim on every figure, which is a caveat nobody wrote.
- **`falsifier` — what observation would make it wrong.** Required because this is
  what makes a figure falsifiable rather than merely asserted, and because it is
  the field a later reader reaches for when they doubt the number. It goes *above*
  the figure in the rendered output, which is the unfashionable part of
  `tenbin.report.ordering` and the part most likely to be proposed for tidiness.
- **`denominator` — a `description` *and* a `size`, both required.** A count with no
  description cannot be checked against the corpus, and a description with no count
  cannot be checked at all. `Denominator` also refuses a size below 1, because a
  rate over zero records has no value and the honest rendering of one is a refusal
  rather than a division.

Two more fields are required but are not caveats, and they get no default for the
same reason: **`kind`** and **`granularity`**. A defaulted kind would default to
the least demanding one, so omitting it would quietly downgrade a comparative claim
to a descriptive one and let it render. A defaulted granularity would default to
the floor, which is a per-principal refusal that every caller then has to remember
to make.

## Pick the figure type from what you have, not from what reads well

All six live in `tenbin.measures.base`, and every one of them carries a `Claim`, a
`CorpusSnapshot` and a `title`, all required. The snapshot is the part that is easy
to forget and impossible to work around: `Figure.rate_text` renders whether the
corpus behind the number is whole, takes no caller-supplied wording, and cannot be
overridden, so a renderer physically cannot print a number without holding the
object that says whether the number is whole.

| You have | Build | Because |
|---|---|---|
| How many | `CountFigure(value)` | A count needs no population behind it beyond the read, and it is the only honest answer for a store that was reached and holds nothing. |
| A fraction over a population | `RateFigure(rate=Rate(numerator, denominator))` | It refuses a numerator above its denominator, because a rate above 1 is a unit error and would otherwise render as `183.3%`. |
| Buckets | `DistributionFigure(values, excluded)` | Both mappings are required; see the section on exclusions below. |
| Something that looks wrong | `Finding(what_was_observed, cannot_confirm, ...)` | A finding asserts an observation, so it carries the whole apparatus of a claim including the falsifier. `cannot_confirm` is required and cannot be blank. |
| One answer in several parts | `FigureGroup(figures=(...))` | A `Figure` rather than a tuple, so a report iterating measures cannot silently drop the last two. |
| A measure with no number to state | a bare `Figure` | `value_text` is empty rather than abstract, because "the number does not exist" is a real answer and forcing it to invent a zero would be the wrong pressure. Prefer a `Finding`, which can *say* it found nothing. |

`FigureGroup` carries one shared claim, and that is a constraint rather than a
convenience: every member is a statement over the same population. **If you cannot
write a single `Denominator` that fits all of them, you have two measures.** The
way to tell is to try to write the denominator and fail.

## The denominator is a population, and the store's count is not the enumerated count

**This is the rule people get wrong, so it is stated as an absolute: a rate's
denominator is a population, and the population a completeness-style rate is over
is the store's own count — never the number of records you enumerated.**

A rate of enumerated over enumerated is identically 1. It renders as `100.0%`,
it is true on every run against every store that does not fail, and it carries no
information whatsoever. It is the arithmetic of a tautology dressed as a finding,
and it is one line of code, which is exactly why it is not the line of code that
got written. `READ_COMPLETENESS_SLUG` in `tenbin.measures.completeness` is the
one measure whose denominator is `STORE_POPULATION` — "documents the store says
the collection holds" — because that is the only independent check on the walk
available to anybody.

Three consequences you will hit:

- **A store that was reached and holds nothing is a `CountFigure`, not a rate.**
  `Denominator` refuses `size < 1` and `RateFigure` refuses it again, so the case
  has to be rendered as a count with a completeness sentence beside it. `0/0` is
  not one.
- **A store whose own `total` disagrees with its listing is a count too**, and the
  sentence above it says the corpus has no agreed size. A
  `Completeness.STORE_TOTAL_EXCEEDED` read produces exactly the case where a
  numerator exceeds its denominator, and a `RateFigure` would refuse it as a
  defect rather than render it.
- **An unreachable store raises.** It never produces a snapshot with zero records,
  because a snapshot from a store that was never reached is indistinguishable from
  a corpus of nothing. See `take_snapshot` in `tenbin.corpus.snapshot`.

## An unstated value never joins a distribution of stated ones

Kojutsu writes the literal string `"unknown"` into `answered_by_model` and
`declared_by_model` wherever a principal named no model. The key is present in
every document of that shape, so the value is doing two jobs at once: sometimes a
model, sometimes the record's way of saying it has none.

`tenbin.corpus.values.parse_stated` returns `Stated(value)` or the `Unstated`
sentinel, and **they are different types on purpose.** The tempting refactor is to
return `str | None`, or worse the raw string; both deletions are invisible in a
diff and both are unrecoverable, because the only thing that survives the next
person who tidies a module is the type.

So when you group by model:

- **Count `Unstated` into `excluded`, under a key that names the condition**, not
  as a bucket in `values`. `"records_with_no_stated_model"` is a sentence a reader
  can act on; a bucket labelled `unknown` beside the real models is a claim about
  a principal that nobody made.
- **Never write `if record.answered_by_model:` as the check.** `Unstated` is
  falsy so that this happens to work, but the reason to use it is that the type
  makes the *other* mistake unavailable: a `Stated` wrapper that unpacks into a
  bare `str` at the first convenient line.
- **The same rule has a second costume.** `named_repository` in
  `tenbin.measures.filtering` is the repository-level twin: a record with no
  repository is filed at `unknown/pr-<n>/<entry>`, so `Record.repo` is the string
  `"unknown"` rather than `None`, and a truthiness test counts the placeholder as
  a repository.

Kojutsu has since stopped writing `"unknown"` into those two fields and leaves
the key absent instead, and `Unstated` is still the right type here: it covers the
documents written before that change, and it is the only reading that stays correct
across both corpora.

## `DistributionFigure.excluded` exists so a silent filter cannot become a result

**A bucket that did not occur and a bucket that was filtered out are the same
integer in `values` and completely different facts.** `DistributionFigure` carries
`excluded` beside `values` and neither is optional, which is the difference between
this and the `unknown`-bucket bug the whole program exists to have stopped. A
figure that cannot say which it is showing has a silence indistinguishable from a
result.

`select` in `tenbin.measures.filtering` deliberately returns only what it was
asked for and has no companion that turns a filter into a count, because a
function returning "how many were dropped" would be a number with no condition
attached — and a number with no condition attached is how an exclusion becomes a
bucket. **So the count is yours to produce, in `excluded`, under a key that names
the condition.** Every filtered record is either in `values` or in `excluded`, and
a reviewer should be able to add the two up to your denominator.

## `requires_certainty` only counts if you pass it to `select`

`classify` in `tenbin.corpus.record` returns a `Certainty.DETERMINED` reading when
the writer stated `record_kind` and a `Certainty.INFERRED` one when the kind had to
be recovered from the tag set. That distinction is real — "the store says this is
an answer" versus "nothing here said so" — and a measure that needs only the first
has to say so.

Declare it on the class:

```python
requires_certainty = Certainty.DETERMINED
```

and then **pass it at every call site**:

```python
verdicts = select(
    records,
    certainty=self.requires_certainty,
    kinds=(RecordKind.REVIEW_VERDICT,),
)
```

**This was a real bug during the build: a measure declared `requires_certainty`
and then called `select` without it.** Because `select`'s `certainty` parameter
defaults to the module-level `requires_certainty`, which is `None` — meaning *no
filtering* — the omission is invisible. The measure produced a figure over every
record whatever its certainty, its exclusion counts were correspondingly smaller,
and the two figures looked identical in a report. Nothing failed; the filter was
simply absent.

The defence is that the declaration and the use are visible together. A measure
that says what it required can be checked; a measure that was handed a filter
cannot. Two measures can be one boolean apart and produce figures that read
identically, with the difference showing only in an exclusion count — so when you
see a distribution with suspiciously few exclusions, check this first.

When you do filter, the records you dropped go in `excluded`. **A record whose kind
is inferred is not a record that failed to parse**, and a filter that treated the
two as interchangeable would be discarding the only evidence that a corpus is
partly legacy.

## A blocked measure is a `Refusal`, and lifting one is a deliberate act

**If the corpus cannot support a measure, the answer is a catalogue entry, not an
absent function.** The reason is that absence is invisible: nobody asks why review
throughput is missing, and a shrug and a silence are the same thing to the reader.
A refusal is a claim about the corpus that has to be made as carefully as any
claim in a report, so it names the measure, says what is missing, and says what
would change the answer.

`tenbin.claims.refusals.CATALOGUE` holds them, keyed by
`slugify(measure_name)` so the slug can never drift from the name. Write it with
`refusal_for_measure(...)`:

- **`unblocked_by=None` is an answer, not a gap.** It means no Kojutsu ticket
  unblocks this and that is the correct outcome. Two entries are in that state,
  one of them this project's central question. An empty string is never valid,
  because a blank ticket list rendered into a report reads as a reference to
  something.
- **`missing_fact` is optional** because not every refusal is about an absent fact.
  A per-principal claim is refused because it is the wrong product, and no fact
  arriving would make it this one.
- **Order matters and is not arbitrary.** The catalogue is ordered the way
  `docs/seam.md` lists it, and the completeness test in
  `tests/test_refusal_registry.py` parses that document's table and asserts every
  blocked row has an entry naming exactly those tickets. **If you change
  `docs/seam.md`, you are changing the catalogue**, and the test will tell you.

**Lifting a refusal is a deliberate act with a shape.** Remove the entry *and*
change the row in `docs/seam.md` to `available now` in the same change, or the
completeness test fails in one direction or the other. Then write the measure, and
put the blocked-on fact in the new claim's `does_not_mean` if it still applies —
a measure that used to be unrenderable and now renders has a history, and the
history is usually why a reader should be careful with it.

The other direction matters too. `tenbin.claims.gate.ClaimGate` refuses
comparative claims and per-principal claims at render time regardless of what the
catalogue says, because those are not waiting on a ticket. Do not add a measure
whose whole purpose is a comparison **between populations**; see
[`decisions/002-no-causal-claims.md`](decisions/002-no-causal-claims.md). A measure
against the same population's own past value is a different figure and is not
refused — that is `ClaimKind.temporal`, and the distinction is one population
observed twice rather than two populations observed once.

## Test names state the reason, not the mechanics

**A test name that says what the test does has to be rewritten when the test
changes, and a test name that says why it exists does not.** The whole suite here
is written that way, and the convention is not decoration — it is the only thing
that survives the copy-paste from a previous measure.

```python
# Says what it does. Becomes a lie the moment the filter moves.
def test_select_filters_by_kind() -> None: ...

# Says what it protects. Still true after the filter moves, because the thing
# being protected did not change.
def test_a_declared_certainty_that_is_not_passed_to_select_shows_no_exclusions() -> None: ...
```

The pattern for every measure is the same set of shapes, and each is a separate
test rather than a table of assertions, so that a failure names the defect:

- a measure over a clean fixture produces the figure the claim describes;
- the same measure over a truncated read renders the truncation sentence;
- the denominator is the population the claim names, and the size is the store's
  count rather than the enumerated count;
- anything filtered out appears in `excluded` under a condition-phrase, and the
  buckets plus the exclusions add up to the denominator;
- a value that must not be counted is not counted, which for a stated-vs-unstated
  axis is the assertion the type was introduced to make possible.

## What this protocol does not buy

**A well-formed measure is exactly as good as the corpus it reads.** None of the
rules above make a corpus representative. They make a figure *legible*: it says
what it is a rate over, what it does not mean, and whether the read behind it
reached the end of the store. A legible figure over a biased corpus is still a
biased figure, and the bias is upstream of anything in this document.

**No amount of type discipline makes a self-selected, self-censored corpus
representative.** Kojutsu records what somebody chose to answer. A repository
where nobody answers is byte-identical to a repository with no knowledge debt, no
backfill path exists, and nothing in `tenbin.claims` or `tenbin.measures` can
tell the two apart — the missing observation left no trace anywhere. The measures
here report counts over a stated population precisely because that is what the
corpus supports, and a completeness of 100% is a statement about the store's count
matching this read, not about development.

**A required falsifier does not mean anybody will check it.** The field forces the
sentence to exist; whether the observation is ever made is a matter of somebody
being in the position to make it, and this program has no way to arrange that.
Likewise a `denominator.description` is only as honest as the person who wrote it.
The types remove the option of *omitting* the caveat, which is the failure this
project was built to stop; they do not and cannot remove the option of writing a
caveat that is merely present.
