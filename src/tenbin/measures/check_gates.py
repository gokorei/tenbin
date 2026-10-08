"""Which gate failed, because a conclusion with no gate's name on it is not actionable.

Kojutsu writes three fields from one check-run webhook payload and projects all three
into the store verbatim (``answer_collector.py:954-958``, ``tanseki_mapping.py:193-196``):
``check_conclusion``, ``check_id`` and ``check_name``. Until this measure existed, the first
was read and the other two were discarded, and that is backwards. **The conclusion is the
least actionable of the three**: *some check concluded failure 12% of the time* is a number
nobody can act on, because it does not say which check, and *the lint check fails 40% of the
time* is a number somebody can act on this morning.

**A check is a named gate, and a gate's failure rate is a property of the gate -- so the unit
here is the check name and not the commit.** Per-check conclusion distributions answer which
gate is failing, how often, and how that moved. That is a fact about a pipeline and the gates
on it, and it is the first figure in this program that describes something a team can change
without changing how they work.

**One measure per gate, and not a ``FigureGroup`` of per-gate figures -- this is the part the
obvious implementation gets wrong, so it is argued here rather than left to the next reader.**
``FigureGroup`` carries one shared claim, and that is a constraint rather than a convenience:
every member is a statement over the same population, because the claim names the denominator
and a figure carrying a claim whose denominator does not describe it is the exact defect this
package is built to stop. The populations here differ **by construction** -- a repository runs
different checks on different paths, so a lint gate that only runs on Python files is only ever
observed on Python files -- so there is no single ``Denominator`` that describes a lint figure
and a type-check figure together, and no one to try writing it. A ``FigureGroup`` here would be
the tempting version and it is the mistake the class exists to prevent. So
:func:`check_gate_measures` returns one measure instance per gate, each with its own claim and
its own denominator naming the changes that gate ran on.

**The check population is selected, and the denominator says so rather than letting a reader
assume otherwise.** A per-check failure rate is a rate over the changes that ran that check,
which is not the corpus: the path filters, the matrix jobs and the branch protections that
decide where a gate runs are all outside this corpus. The denominator carries the size of that
selected population beside the sentence that says what it is, because a rate whose population
is implied rather than named reads as a rate over everything.

**Two caveats ride in the claim because a reader cannot see either from the figure.** A check
concluding ``success`` is a fact about the build and not about the change -- a green pipeline
on a change that broke production passed its pipeline -- and reverts remain out of reach, so a
change that merged and was later reverted is indistinguishable here from one that stuck. Both
sentences are imported from :mod:`tenbin.measures.check_outcome` rather than restated, because
they are facts about the same store rather than two versions of one caveat.

**A check name is the forge's own label and nothing more, and there is no list of known
names.** The same rule the conclusion distribution already follows: report every value, never
filter to a closed set, and give a value that occurs its own figure rather than folding it into
an ``other`` bucket. **A gate appearing in a corpus is information, and the number of gates
changes over time** -- a bucket labelled ``other`` would make a newly added gate look like a
long-standing one and would make the count of gates a constant.

**A corpus with no check runs yields no figures at all, and that is the whole answer.** Check
runs are an opt-in subscription (``RSF1DK1S``) rather than a capture path, so their absence is
not a defect and not a pipeline with a clean record: a repository nobody subscribed them on has
no gate axis to report on. Every rendering available over an empty one would be a claim about a
pipeline nobody installed -- a distribution of nothing reads as a healthy record, and a
:class:`~tenbin.measures.base.Finding` would put a fact about a subscription in the section of a
report reserved for the capture path's integrity. So :func:`check_gate_measures` returns an empty
tuple and the axis is absent, which is the one honest answer.

**What this module notably does not do:** it does not synthesise a name for a check the forge
named nothing. A nameless gate cannot be acted on, and inventing a label for it would make it
look handled; the runs of such checks are reported under the run id the forge did give, with
the fallback counted in ``excluded`` and said to be a worse identity than a name rather than
hidden. And it does not modify
:class:`~tenbin.measures.check_outcome.CheckOutcomeMeasure`, which answers a different
question -- *how often does anything fail* against *which gate fails*. Collapsing the two would
lose the second of those questions, which is the actionable one, while adding nothing the first
had not already said.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from tenbin.claims.mechanism import Granularity
from tenbin.claims.model import Claim, ClaimKind, Denominator
from tenbin.corpus.record import Record, RecordKind
from tenbin.corpus.snapshot import CorpusSnapshot
from tenbin.measures.base import (
    DistributionFigure,
    Figure,
    Measure,
    histogram,
    subset_window,
)
from tenbin.measures.check_outcome import (
    NO_CONCLUSION_KEY,
    REVERTS_ARE_OUT_OF_REACH,
    SUCCESS_IS_NOT_CORRECTNESS,
)
from tenbin.measures.filtering import requires_certainty, select

#: The namespace every gate's claim slug is published under. Per gate rather than one slug for
#: the whole axis, and the reason is structural rather than cosmetic:
#: :class:`tenbin.report.document.Report` refuses to hold two sections for one claim, so two
#: gate figures sharing a slug would raise on any corpus with two gates -- a crash in the
#: report layer caused by a measure choosing not to be addressable.
CHECK_GATE_SLUG_PREFIX: Final[str] = "corpus.check_gates."

#: The ticket that wrote ``check_id`` and ``check_name`` beside ``check_conclusion``, and which
#: this measure exists because of. ``unblocked_by`` rather than prose only, because a reader
#: asking "what changed" deserves one reference rather than a diff.
UNBLOCKED_BY: Final[str] = "RSF1DK1S"

#: How many hex digits of the identity digest a slug carries when sanitising the name lost
#: anything. Long enough that two gates on one repository cannot collide by accident, short
#: enough that a slug stays readable to somebody looking one up.
SLUG_DIGEST_LENGTH: Final[int] = 8

#: What a gate slug body is made of. Non-alphanumerics collapse to a single separator, the rule
#: :func:`tenbin.claims.refusals.slugify` uses for measure names, applied here to forge free
#: text rather than to a name somebody chose.
_SLUG_SEPARATOR: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9]+")

#: The population a gate figure is a rate over, in the words the denominator renders. "Commits"
#: is the load-bearing word and it is the *whole* of this ticket: a figure over the corpus would
#: answer "how often does anything fail", which is what ``corpus.check_outcome`` already says,
#: while the rate anybody can act on is a rate over the changes that particular gate ran on.
GATE_POPULATION: Final[str] = "check runs of {subject} on commits where that check ran"

#: What a gate denominator's size is counting, in the words it renders. Not *records*: a check
#: run is a document here, but a denominator counting them is counting builds, and the whole
#: point of the field this uses is that a figure whose noun does not match its description tells
#: the reader it covers a different population than the one it was computed over.
GATE_SIZE_NOUN: Final[str] = "check runs"

#: Why the population is not the corpus, and why two gates' figures cannot be added up. The
#: selection is made outside this program -- in the workflow files, the path filters and the
#: branch protections that decide where a gate runs -- and a rate whose denominator is a
#: selected subset reads exactly like a rate over everything unless the selection is named.
SELECTED_POPULATION: Final[str] = (
    "That these are rates over the corpus, or that two of them can be added up or compared. "
    "The population of this figure is the changes on which this check ran, and a repository runs "
    "different checks on different paths: a lint gate that runs only on Python files is only ever "
    "observed on Python files, and a gate that a branch protection runs on the default branch is "
    "only ever observed there. Nothing in this corpus records why a gate did or did not run on a "
    "change -- the selection is made in the pipeline configuration, outside this seam -- so every "
    "rate here is a rate over a selected subset of changes and two gates' figures describe two "
    "different populations."
)

#: What a check name is, which is a label rather than an identity. The forge reuses names: one
#: workflow reported under one name across three matrix jobs is one figure over three gates, and
#: this corpus could not tell, because a check run carries nothing beyond its name and the id of
#: one run.
NAME_IS_NOT_A_GATE: Final[str] = (
    "That the check has an identity in this corpus beyond the label the forge gave it. "
    "`check_name` is the forge's own wording, stored verbatim, and the forge reuses it freely: "
    "one workflow reported under one name across three matrix jobs is one figure over three gates "
    "here, and nothing in the corpus can separate them, because a check run carries no gate "
    "identity beyond its name and the id of one run."
)

#: What the fallback identity is worth, which is less than it looks. Kojutsu writes GitHub's
#: check-*run* id, and a run id identifies one execution of a gate rather than the gate: two
#: pushes through the same workflow arrive under two ids. So a figure built on an id describes a
#: single run, and this sentence is why it says so instead of reading as a gate's failure rate.
RUN_ID_IS_NOT_A_GATE: Final[str] = (
    "That a run id names a gate. Kojutsu writes GitHub's check-run id, and that identifies "
    "one run of a check rather than the check itself: two pushes through the same workflow "
    "arrive under two different ids and are reported here as two figures. These runs are the "
    "ones the forge named nothing for, which is why they are reported at all -- an unnamed gate "
    "cannot be acted on, and dropping it would have made an unhandled gap look like a handled "
    "one -- but a figure over one of them describes one run."
)

#: The exclusion key naming the condition the id-keyed figures exist for, and the count under it
#: is every nameless check run in the read rather than only this figure's own -- including the
#: runs the forge identified by neither field, which have no figure of their own and would
#: otherwise be counted nowhere. **That is the one ``excluded`` entry in this package that is not
#: local to the figure it rides on**, and it is deliberate: the count describes the whole axis,
#: the figures that need the warning are exactly the ones built from the fallback, and there is no
#: figure over a named gate that a reader would want the count attached to. It is written down here
#: rather than left as a surprise for the person adding the two numbers up.
UNNAMED_GATE_KEY: Final[str] = (
    "check runs the forge named nothing for, reported under the run id it did give rather than as "
    "gates; a check-run id names one run and not one gate, so a figure carrying this exclusion "
    "describes a single run, and a nameless run the forge gave no id for is counted here with no "
    "figure of its own"
)


def gate_slug(identity: str) -> str:
    """The claim slug for one gate, derived from its identity and unable to collide.

    The readable part is the identity with non-alphanumerics collapsed, which is
    :func:`tenbin.claims.refusals.slugify`'s rule applied to forge free text. Two identities
    that sanitise to the same readable part would share a claim slug, and two gates sharing a
    claim slug raises in :class:`tenbin.report.document.Report` rather than rendering twice,
    so a digest of the *exact* identity rides along whenever the readable part is not the
    identity itself -- which is every name the fold or the collapse touched, and every name made
    of nothing but punctuation.

    **Case counts as "the fold touched it", and that is the half that is easy to leave out.**
    ``check_name`` is stored verbatim and the forge writes it, so ``Lint`` and ``lint`` are two
    labels two repositories really use for the same workflow, and this corpus is a read across
    repositories rather than one pipeline. Testing the readable part against the *folded* identity
    would hand both the plain slug, and the report would then raise on a collision between two
    gates that both exist and both have a failure rate worth reading -- so the comparison below
    is against the identity, not against its own fold.

    Never raises and never returns an empty body, because a check named ``"!!!"`` is a check
    somebody configured and this program does not get to decide it does not count. Two distinct
    identities could still collide by digest, which needs both the same readable part and the
    same eight hex digits of SHA-256; that is accepted rather than engineered away, and the
    alternative -- a slug a report can raise on -- is the worse failure.
    """
    lowered = identity.casefold()
    body = _SLUG_SEPARATOR.sub("-", lowered).strip("-")
    if body and body == identity:
        return f"{CHECK_GATE_SLUG_PREFIX}{body}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:SLUG_DIGEST_LENGTH]
    return f"{CHECK_GATE_SLUG_PREFIX}{body or 'gate'}-{digest}"


def check_runs(records: Iterable[Record]) -> tuple[Record, ...]:
    """Every check-run record in this read, which is the population this axis counts over.

    Filtered to :attr:`~tenbin.corpus.record.RecordKind.CHECK_RUN` rather than to "any record
    carrying a conclusion", for the reason
    :meth:`~tenbin.measures.check_outcome.CheckOutcomeMeasure._conclusions` gives: the kind is
    what Kojutsu states, and a record that happens to carry the key without being a check
    report is a document this program has no story for. The certainty is passed explicitly at
    the one call site rather than left to :func:`~tenbin.measures.filtering.select`'s default,
    because a filter that was declared and not applied is invisible in the output.
    """
    return select(records, certainty=requires_certainty, kinds=(RecordKind.CHECK_RUN,))


def gate_identity(record: Record) -> tuple[str, bool] | None:
    """The gate a check run belongs to, and whether the forge gave it a name.

    The name is the identity a gate can be acted on by, so it wins. Failing it, the run id is
    the fallback -- a worse identity than a name, which :data:`RUN_ID_IS_NOT_A_GATE` says out
    loud -- and a run the forge identified by neither is not a gate this program can name, so it
    returns ``None`` and the caller counts it rather than inventing a label for it.
    """
    if record.check_name:
        return (record.check_name, True)
    if record.check_id:
        return (record.check_id, False)
    return None


def unnamed_gate_runs(records: Iterable[Record]) -> int:
    """How many of these check runs the forge named nothing for.

    Counted over the check-run population rather than over a gate's own runs, because the
    condition is a fact about the axis and not about one gate: it is the number that says how
    much of this axis rests on a run id instead of a name.

    It is also the only rendered home for the runs the forge identified by *neither* field.
    Those have no identity, so the factory builds no figure for them and no exclusion anywhere
    else would mention them -- and a check run the forge could not identify at all is worth a
    reader's attention precisely because nothing about it can be acted on. They are expected to
    be rare rather than absent: Kojutsu writes a check-run id from every webhook payload, so
    a document with neither field is a hand-edited one, and the count is how that would show.
    """
    return sum(1 for record in records if not record.check_name)


def check_gate_identities(records: Iterable[Record]) -> tuple[tuple[str, bool], ...]:
    """Every gate this read holds, as ``(identity, named)``, in a stable order.

    Sorted and de-duplicated so two reads of the same corpus produce the same list and therefore
    the same report order. The identity is the gate's own free text, which is not sorted, so the
    order is not a ranking of gates -- it is the order a reader scrolls.
    """
    identities = {
        identity
        for identity in (gate_identity(record) for record in check_runs(records))
        if identity is not None
    }
    return tuple(sorted(identities))


def check_gate_measures(records: Iterable[Record]) -> tuple[Measure, ...]:
    """One measure per gate this read holds, and nothing at all when it holds none.

    **A factory rather than a registry entry, because the gates are discovered from the data and
    a registry cannot be assembled at import time.** :func:`tenbin.measures.default_measures` is
    a tuple of instances constructed when it is called, and there is no way for a name nobody has
    seen yet to be in it; :func:`tenbin.report.build.build_report` resolves that tuple by name
    (:data:`tenbin.report.build.PROJECT_MEASURES`) and runs every measure over the snapshot, so
    this is the seam where a data-discovered measure has to enter.

    **The wiring is a parameter on the registry rather than an append in the report layer, and
    the distinction is load-bearing rather than taste.** ``build_report`` resolves exactly one
    registry and hands every member the same snapshot, so the records that name the gates are
    already in hand by the time the registry is read -- ``snapshot.records`` -- and the only
    honest way for a per-gate instance to exist is for the registry to be built knowing them.
    So ``default_measures`` takes ``records`` and splices this factory's result into its tuple,
    and ``build_report`` passes ``snapshot.records`` when it calls the entry point. Appending
    here instead would mean a second registry, and a second registry is a list that can disagree
    with the first; the seam stays one function.

    **There is deliberately no cap on how many gates this returns, because a cap is an ``other``
    bucket under another name.** A repository with sixty checks produces sixty sections, and the
    day that is unreadable the fix is a reader's filter over the sections -- not a line here that
    drops the gates it ranked lowest, which is the same defect the vocabulary rule above refuses
    and would be much harder to notice than a bucket labelled ``other``.

    A corpus holding no check runs yields an empty tuple, which is the correct answer rather than
    a hole -- see the module docstring for why the absent axis is silent.
    """
    return tuple(
        CheckGateMeasure(gate, named=named) for gate, named in check_gate_identities(records)
    )


def _subject(gate: str, named: bool) -> str:
    """How one gate is named in prose: by the forge's label, or by the run id it fell back to.

    One phrase for both cases rather than a template with a hole in it, because it is rendered
    into three places -- the statement, the figure's title and the denominator -- and three
    wordings of one subject is three things that can drift apart.
    """
    if named:
        return f"the check named `{gate}`"
    return f"the check this corpus holds only as the run `{gate}`"


def _gate_denominator(gate: str, named: bool, runs: Sequence[Record]) -> Denominator:
    """This gate's own population, counted, dated and named in the words a reader looks for.

    Built from :func:`~tenbin.measures.base.subset_window` rather than by calling
    :func:`~tenbin.measures.base.subset_denominator`, and that is one of two documented places
    this module declines a helper it would rather have: the helper cannot carry a size noun, and
    a denominator that counts check runs while rendering the word *records* is precisely the
    substitution ``Denominator.size_noun`` exists to prevent -- "12 records" beside a population
    of check runs tells a reader the figure covers twelve documents. The window itself is still
    computed by the one function that owns it, so the only thing duplicated here is the join
    between a description and a window, and giving
    :func:`~tenbin.measures.base.subset_denominator` a size-noun argument would let this call
    it instead.

    ``max(1, ...)`` is the same floor every denominator in this program applies: the factory
    never builds an instance over an empty population, and a corpus that could otherwise reach
    here has a rate over nothing, which is a refusal rather than a division.
    """
    description = GATE_POPULATION.format(subject=_subject(gate, named))
    return Denominator(
        f"{description}; {subset_window(runs).render_text()}",
        max(1, len(runs)),
        size_noun=GATE_SIZE_NOUN,
    )


def check_gate_claim(gate: str, runs: Sequence[Record], *, named: bool) -> Claim:
    """The claim behind one gate's conclusion distribution, over that gate's own check runs.

    ``runs`` is the population rather than a count, for two reasons that are one: the denominator
    is a subset of the corpus and therefore states its own window, and the figure the claim
    licenses is built from exactly the records the denominator counts. A claim carrying a
    population the figure does not cover is the defect this package is built to prevent, and
    passing the records rather than a number makes it impossible rather than merely unlikely.

    ``named`` selects which of the two identities is being described, and it changes the
    statement, the population and the negative space rather than only the last of them: a figure
    over a run id is a weaker claim about a smaller thing and must read as one.
    """
    identity_caveat = NAME_IS_NOT_A_GATE if named else RUN_ID_IS_NOT_A_GATE
    return Claim(
        slug=gate_slug(gate),
        statement=(
            f"The conclusions the builds in this corpus reported for {_subject(gate, named)} are "
            "distributed as the figure below shows, over the changes that check ran on."
        ),
        does_not_mean=(
            SUCCESS_IS_NOT_CORRECTNESS
            + " "
            + REVERTS_ARE_OUT_OF_REACH
            + " "
            + SELECTED_POPULATION
            + " "
            + identity_caveat
        ),
        falsifier=(
            "Two check runs carrying this same identity for two different gates -- the forge "
            "reusing one label for one workflow, or a check-run id that turns out to name a "
            "workflow rather than one run of it. Either would mean these buckets are counting "
            "one gate under two identities or one identity over two gates, and nothing in the "
            "buckets could tell the difference. The opposite observation would be just as "
            "damaging: one check-run id on two records of the same gate would mean the fallback "
            "is splitting one gate across several figures, which is why an id-keyed figure "
            "carries its own caveat instead of reading as a gate's failure rate."
        ),
        denominator=_gate_denominator(gate, named, runs),
        kind=ClaimKind.descriptive,
        granularity=Granularity.repository,
        unblocked_by=UNBLOCKED_BY,
    )


@dataclass(frozen=True)
class CheckGateMeasure:
    """One gate's conclusions, over the changes on which that gate ran.

    Stateless and total, like every measure here: ``compute`` is a pure function of the snapshot
    given, with no clock and no store, so a test builds a snapshot from a fixture and asserts on
    the figure. The only state is which gate this instance speaks for, and that is data rather
    than history.

    Frozen and hashed by its two fields, so a report holding two instances of the same gate --
    which the factory cannot produce -- would also be two sections with the same claim slug, which
    :class:`tenbin.report.document.Report` refuses rather than renders twice.
    """

    #: The forge's ``check_name`` for this gate, or the ``check_id`` it fell back to when the
    #: forge named nothing. Free text either way, which is why :data:`NAME_IS_NOT_A_GATE` and
    #: :data:`RUN_ID_IS_NOT_A_GATE` are two sentences rather than one.
    gate: str

    #: Whether :attr:`gate` is a name the forge gave rather than a run id it fell back to. ``True``
    #: for an ordinary gate and ``False`` for the fallback, and it selects which caveats the claim
    #: carries -- an id-keyed figure has one more thing wrong with it and says so.
    named: bool = True

    #: Every check run counts, whatever certainty its classification carries. A check run read
    #: from Kojutsu's own ``record_kind`` is the ordinary case, and requiring ``DETERMINED``
    #: here would narrow the axis over something this measure never reads.
    requires_certainty = requires_certainty

    @property
    def slug(self) -> str:
        """The identifier this gate's claim is cited by, and the one a refusal would use."""
        return gate_slug(self.gate)

    def claim(self, snapshot: CorpusSnapshot) -> Claim:
        """The claim, over the check runs of *this* gate that the read enumerated."""
        return check_gate_claim(
            self.gate, self._runs(check_runs(snapshot.records)), named=self.named
        )

    def compute(self, snapshot: CorpusSnapshot) -> Figure:
        """The conclusions this gate's runs reported, with what it could not count named.

        The title names the gate and says the unit is the check, because a title reading "check
        conclusions" over two hundred gates in one report is how a reader ends up comparing a
        lint gate's distribution with a type-check gate's. It deliberately does not contain the
        word *commit*: the population is in the denominator, where it is stated rather than
        implied.
        """
        all_runs = check_runs(snapshot.records)
        runs = self._runs(all_runs)
        return DistributionFigure(
            claim=check_gate_claim(self.gate, runs, named=self.named),
            snapshot=snapshot,
            title=f"Conclusions the builds reported for {_subject(self.gate, self.named)}",
            values=histogram(
                record.check_conclusion for record in runs if record.check_conclusion is not None
            ),
            excluded=self._excluded(runs, all_runs),
        )

    def _runs(self, runs: Sequence[Record]) -> tuple[Record, ...]:
        """The check runs in this read that belong to this gate, by the same identity rule."""
        return tuple(run for run in runs if gate_identity(run) == (self.gate, self.named))

    def _excluded(self, runs: Sequence[Record], all_runs: Sequence[Record]) -> Mapping[str, int]:
        """What this figure could not count, under a key naming each condition.

        Two conditions, and they are different gaps with different fixes. A run of this gate that
        concluded nothing is a build that reported nothing, and it shares
        :data:`~tenbin.measures.check_outcome.NO_CONCLUSION_KEY` with the conclusion
        distribution so the two figures cannot describe the same absence in two words. The
        nameless-run count belongs only on an id-keyed figure, for the reason
        :data:`UNNAMED_GATE_KEY` states: those figures exist *because* of the fallback, so they
        are where a reader has to be told the identity underneath them is a run rather than a
        gate.
        """
        silent = sum(1 for run in runs if run.check_conclusion is None)
        excluded: dict[str, int] = {}
        if silent:
            excluded[NO_CONCLUSION_KEY] = silent
        if not self.named:
            excluded[UNNAMED_GATE_KEY] = unnamed_gate_runs(all_runs)
        return excluded
