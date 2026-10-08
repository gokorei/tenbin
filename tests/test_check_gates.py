"""Which gate failed, because the conclusion without the gate's name is not actionable.

`CheckOutcomeMeasure` reports the distribution of conclusions over check runs, which on a real
corpus reads *some check concluded failure 12% of the time* -- a number nobody can act on,
because it does not say which check. The lint check fails 40% of the time is a number somebody
can act on this morning. These tests hold the reframe that makes the second number possible:
**a check is a named gate, a gate's failure rate is a property of the gate, and the unit is
therefore the check name rather than the commit.**

Three things are easier to get wrong than the arithmetic, so each has its own test rather than an
assertion inside a bigger one:

- **The population.** Each gate gets its own measure, claim and denominator, because a `FigureGroup`
  would be a figure carrying a claim whose denominator does not describe it -- the populations
  differ by construction, a lint gate being observed only on Python files. So the figures add up
  to nothing together, and a test holds each denominator to its own gate's runs.
- **The vocabulary.** A check name is forge free text that will gain values, so there is no known
  set and no `other` bucket: a gate appearing in a corpus is information, and a bucket would make
  a new gate look like a long-standing one.
- **The absent axis.** A corpus with no check runs is a repository nobody subscribed check runs
  on, which is not a pipeline with a perfect record, so it yields no figures at all and fires no
  integrity finding.

The nameless case is here too, because it is the one place the fallback is worth having and worth
caving about at the same time: a check run the forge named nothing for is reported under the run
id it did give, which is a worse identity than a name because a check-run id names one *run* of a
gate rather than the gate, and the figure says so rather than reading as a gate's failure rate.
"""

from __future__ import annotations

from tenbin.claims.gate import ClaimGate
from tenbin.corpus.record import Record, RecordKind
from tenbin.measures.base import DistributionFigure, Figure, Finding
from tenbin.measures.check_gates import (
    CHECK_GATE_SLUG_PREFIX,
    RUN_ID_IS_NOT_A_GATE,
    UNNAMED_GATE_KEY,
    CheckGateMeasure,
    check_gate_identities,
    check_gate_measures,
    check_runs,
    gate_slug,
    unnamed_gate_runs,
)
from tenbin.measures.check_outcome import (
    NO_CONCLUSION_KEY,
    REVERTS_ARE_OUT_OF_REACH,
    SUCCESS_IS_NOT_CORRECTNESS,
)
from tests.fixtures import build_record, build_snapshot, truncating_snapshot


def _check(
    conclusion: str | None,
    *,
    name: str | None = "lint",
    check_id: str | None = None,
    pr: int = 1,
    repo: str = "acme/widget",
) -> Record:
    """A check run, in the kind, tag set and three fields Kojutsu writes for one.

    ``name`` and ``check_id`` go through ``build_record`` as ordinary extras, which is what
    puts them on the document the way Kojutsu projects them: omitted entirely where the
    payload carried nothing rather than written as a placeholder. The tag set is derived from the
    conclusion exactly as Kojutsu derives it, so the record classifies as the check run it is
    without the ``record_kind`` key doing all the work.
    """
    tags = ["check"] + ([f"check_state_{conclusion}"] if conclusion else [])
    return build_record(
        entry_id=f"check-{repo.replace('/', '-')}-{pr}-{name or check_id or 'anon'}",
        repo=repo,
        pr=pr,
        record_kind=RecordKind.CHECK_RUN,
        tags=tags,
        head_sha="9f2c1ab",
        capture_source="webhook",
        check_name=name,
        check_id=check_id,
        frontmatter={"check_conclusion": conclusion} if conclusion else {},
    )


def _figure(measure: CheckGateMeasure, records: tuple[Record, ...]) -> DistributionFigure:
    """One gate's figure, asserting the type on the way out so a failure names the defect."""
    figure = measure.compute(build_snapshot(records))
    assert isinstance(figure, DistributionFigure)
    return figure


def _gates(records: tuple[Record, ...]) -> tuple[CheckGateMeasure, ...]:
    """Every measure the factory built for this corpus, narrowed to this module's own type.

    The factory publishes ``tuple[Measure, ...]`` because that is what a registry holds, so a
    test reaching for ``.gate`` has to narrow -- and narrowing here rather than at each call
    site is what keeps every assertion below reading as a statement about a gate.
    """
    built = check_gate_measures(records)
    assert all(isinstance(measure, CheckGateMeasure) for measure in built)
    return tuple(measure for measure in built if isinstance(measure, CheckGateMeasure))


def _gate(records: tuple[Record, ...], gate: str) -> CheckGateMeasure:
    """The instance the factory built for one gate, looked up rather than constructed.

    Looked up so that a test asserting on a gate's figure is asserting on the figure a report
    would run: a measure assembled by hand in a test is a measure nobody wired.
    """
    built = tuple(measure for measure in _gates(records) if measure.gate == gate)
    assert len(built) == 1, f"expected exactly one measure for {gate!r}, got {len(built)}"
    return built[0]


# -- a gate is the unit, not a commit -----------------------------------------------------


def test_one_gate_failing_and_one_passing_is_one_distribution_over_that_gates_runs() -> None:
    """The figure the ticket exists for, in the smallest corpus that produces it.

    One gate, two runs, one failure. The conclusion distribution over all check runs would say
    `failure: 1` of everything; this figure says the same count over the runs of a gate the
    reader can go and look at, which is the whole difference between a number and an action.
    """
    records = (_check("failure", pr=1), _check("success", pr=2))

    figure = _figure(_gate(records, "lint"), records)

    assert dict(figure.values) == {"failure": 1, "success": 1}
    assert figure.claim.denominator.size == 2
    assert figure.counted == 2
    assert figure.excluded == {}


def test_each_gate_gets_its_own_figure_with_its_own_mix_because_the_populations_differ_by_construction() -> (
    None
):
    """Two gates, two figures, two different distributions -- and no shared claim.

    This is the case a ``FigureGroup`` would have got wrong. A group carries one claim, and one
    claim carries one denominator; there is no denominator that describes both a lint gate's runs
    and a type-check gate's, because a repository runs different checks on different paths. So
    the two figures are two measures, and neither can be rendered with the other's population.
    """
    records = (
        _check("failure", name="lint", pr=1),
        _check("failure", name="lint", pr=2),
        _check("success", name="lint", pr=3),
        _check("success", name="typecheck", pr=1),
        _check("success", name="typecheck", pr=2),
    )

    measures = _gates(records)

    assert [measure.slug for measure in measures] == [
        f"{CHECK_GATE_SLUG_PREFIX}lint",
        f"{CHECK_GATE_SLUG_PREFIX}typecheck",
    ]
    lint = _gate(records, "lint").compute(build_snapshot(records))
    typecheck = _gate(records, "typecheck").compute(build_snapshot(records))
    assert isinstance(lint, DistributionFigure)
    assert isinstance(typecheck, DistributionFigure)
    assert dict(lint.values) == {"failure": 2, "success": 1}
    assert dict(typecheck.values) == {"success": 2}
    assert lint.claim.slug != typecheck.claim.slug
    assert lint.claim.denominator.size == 3
    assert typecheck.claim.denominator.size == 2


def test_two_gates_never_share_a_claim_slug_because_a_report_refuses_two_sections_for_one_claim() -> (
    None
):
    """The structural reason a slug is per gate rather than one slug for the whole axis.

    ``tenbin.report.document.Report`` raises when two sections carry one slug, so a shared slug
    would not render twice -- it would stop the report. Names differing only in punctuation are
    the case that would do it, and the case this measure is least able to predict because the
    vocabulary is free text somebody else's workflow files write.
    """
    records = (
        _check("success", name="lint", pr=1),
        _check("failure", name="lint!", pr=2),
        _check("success", name="!!!", pr=3),
    )

    slugs = [measure.slug for measure in _gates(records)]

    assert len(slugs) == len(set(slugs)) == 3
    assert all(slug.startswith(CHECK_GATE_SLUG_PREFIX) for slug in slugs)
    assert gate_slug("lint") == f"{CHECK_GATE_SLUG_PREFIX}lint"


def test_two_gate_names_differing_only_in_case_never_share_a_slug_because_folding_is_not_identity() -> (
    None
):
    """The collision a casefold introduces, which stays invisible until a report holds both gates.

    `check_name` is stored verbatim and the forge writes it, so `Lint` and `lint` are two labels
    two repositories really use for the same workflow -- and this is a read across repositories,
    not one pipeline's check list. A slug built from the *folded* identity hands both the plain
    `corpus.check_gates.lint`, and `Report` refuses two sections for one claim, so the figure
    about the second gate would take the whole report down with it. Readable slugs stay readable:
    the label is still the body, it just carries a digest of the exact identity beside it.
    """
    records = (
        _check("failure", name="Lint", pr=1),
        _check("success", name="lint", pr=2),
    )

    measures = _gates(records)

    slugs = [measure.slug for measure in measures]
    assert len(slugs) == len(set(slugs)) == 2
    assert gate_slug("lint") == f"{CHECK_GATE_SLUG_PREFIX}lint"
    assert gate_slug("Lint").startswith(f"{CHECK_GATE_SLUG_PREFIX}lint-")


def test_the_title_names_the_check_and_not_the_commit_because_a_title_is_the_only_part_a_report_cannot_drop() -> (
    None
):
    """The unit is readable from the figure alone, with no claim in hand.

    A report shows twenty gate figures under headings, and a reader scanning them decides which
    to read from the heading. `Check conclusions` under twenty headings is twenty numbers a reader
    compares; `Conclusions the builds reported for the check named lint` is twenty different
    questions. The word *commit* is kept out of the title on purpose: the population lives in the
    denominator, where it is stated rather than implied.
    """
    records = (_check("failure", name="lint"),)

    figure = _figure(_gate(records, "lint"), records)

    assert "check" in figure.title
    assert "lint" in figure.title
    assert "commit" not in figure.title


# -- the vocabulary is the forge's and it will grow ----------------------------------------


def test_a_check_name_nobody_has_heard_of_is_its_own_figure_and_not_an_other_bucket() -> None:
    """A gate appearing in a corpus is information, and the count of gates changes over time.

    There is no known set here to filter to -- `check_name` is the forge's own label and the
    forge will add workflows -- so the only two ways to lose a new gate are folding it into a
    bucket or refusing it. A bucket called `other` would also make the count of gates look like
    a constant, which is the second fact this axis is reporting.
    """
    records = (
        _check("failure", name="semgrep-linter", pr=1),
        _check("success", name="cargo-audit", pr=1),
    )

    figures = {measure.gate: _figure(measure, records) for measure in _gates(records)}

    assert set(figures) == {"semgrep-linter", "cargo-audit"}
    assert dict(figures["semgrep-linter"].values) == {"failure": 1}
    assert dict(figures["cargo-audit"].values) == {"success": 1}
    for figure in figures.values():
        assert "other" not in figure.values


def test_a_check_run_named_by_nothing_at_all_is_counted_in_the_fallback_and_given_no_name() -> None:
    """A run the forge identified by neither a name nor an id is counted, not invented for.

    The record says a check ran and the forge reported nothing to identify it by. Inventing a
    label would make it look handled and a bucket would hide it, so the run has no identity and
    no figure of its own -- but it is still counted in the exclusion the id-keyed figures carry,
    because that count is the only rendered place a reader would look for a gate this program
    could not name.
    """
    records = (
        _check("failure", name=None, check_id=None, pr=1),
        _check("failure", name=None, check_id="4711", pr=2),
        _check("success", name="lint", pr=3),
    )

    assert check_gate_identities(records) == (("4711", False), ("lint", True))
    assert [measure.gate for measure in _gates(records)] == ["4711", "lint"]
    figure = _figure(_gate(records, "4711"), records)
    assert figure.excluded[UNNAMED_GATE_KEY] == 2
    assert unnamed_gate_runs(check_runs(records)) == 2


# -- the nameless fallback ------------------------------------------------------------------


def test_a_check_run_the_forge_named_nothing_falls_back_to_its_run_id_and_the_fallback_is_counted() -> (
    None
):
    """The fallback, reported and counted, because dropping it would look like handling it.

    Kojutsu writes GitHub's check-run id, so the fallback is real and it is counted rather
    than folded anywhere. The figure it produces describes one *run* rather than one gate, and
    the claim says so in the same words the exclusion key uses -- a reader who skims either one
    has still been told the identity underneath them is weaker than it looks.
    """
    records = (
        _check("failure", name=None, check_id="4711", pr=1),
        _check("failure", name=None, check_id="4712", pr=2),
        _check("success", name="lint", pr=3),
    )

    gate = _gate(records, "4711")
    figure = _figure(gate, records)

    assert gate.named is False
    assert "the check this corpus holds only as the run `4711`" in figure.title
    assert gate.slug == gate_slug("4711")
    assert dict(figure.values) == {"failure": 1}
    assert figure.excluded[UNNAMED_GATE_KEY] == 2
    assert RUN_ID_IS_NOT_A_GATE in figure.claim.does_not_mean


def test_a_named_gate_figure_does_not_carry_the_fallback_count_because_the_fallback_is_not_its_gap() -> (
    None
):
    """The exclusion rides on the figures it explains, and no others.

    ``UNNAMED_GATE_KEY`` is the one entry in this package's ``excluded`` mappings that is not
    local to the figure it sits on -- it counts every nameless check run in the read -- so it
    goes only where it has something to explain. On a named gate it would be a condition that
    does not concern that gate at all, and adding it would put a gap in the arithmetic a reader
    checks the figure with.
    """
    records = (
        _check("failure", name=None, check_id="4711", pr=1),
        _check("success", name="lint", pr=2),
        _check("failure", name="lint", pr=3),
    )

    figure = _figure(_gate(records, "lint"), records)

    assert UNNAMED_GATE_KEY not in figure.excluded
    assert figure.excluded == {}
    assert figure.counted == figure.claim.denominator.size


def test_the_fallback_count_is_the_one_exclusion_that_does_not_add_up_to_the_gate_that_carries_it() -> (
    None
):
    """The arithmetic every other figure in this package satisfies, broken here on purpose.

    `measure-protocol.md` says every record a measure set aside is in `excluded` and the two sums
    add up to the denominator. That holds everywhere except `UNNAMED_GATE_KEY`, which counts
    nameless check runs across the whole axis rather than across one gate -- so on an id-keyed
    figure the two sums run past this gate's denominator, which is the point of the key rather
    than a defect in it. Asserted so the next reader does not "fix" the arithmetic by making the
    count local: that would drop the runs the forge identified by neither field off every figure
    in the report, and they are the runs nothing about can be acted on.
    """
    records = (
        _check("failure", name=None, check_id="4711", pr=1),
        _check("success", name=None, check_id="4712", pr=2),
    )

    figure = _figure(_gate(records, "4711"), records)

    assert figure.claim.denominator.size == 1
    assert figure.counted == 1
    assert figure.excluded[UNNAMED_GATE_KEY] == 2
    assert figure.counted + figure.excluded_total == 3
    assert figure.counted + figure.excluded_total > figure.claim.denominator.size


def test_a_check_run_with_no_conclusion_is_excluded_under_the_key_the_conclusion_measure_uses() -> (
    None
):
    """One condition, one sentence, two figures that cannot describe it differently.

    A gate's runs and the corpus's runs are the same documents, so a silent build is the same
    gap in both axes. Restating it here would be a second wording of one fact, and the two
    figures would drift apart the first time either was edited.
    """
    records = (_check(None, name="lint", pr=1), _check("success", name="lint", pr=2))

    figure = _figure(_gate(records, "lint"), records)

    assert dict(figure.excluded) == {NO_CONCLUSION_KEY: 1}
    assert dict(figure.values) == {"success": 1}
    assert figure.counted + figure.excluded_total == figure.claim.denominator.size


# -- the denominator names the population ---------------------------------------------------


def test_the_denominator_says_the_population_is_the_commits_that_check_ran_on() -> None:
    """The selection is outside this corpus, so it has to be stated rather than implied.

    A lint gate that runs only on Python files is only ever observed on Python files. A rate
    whose denominator is the corpus reads exactly like a rate over everything unless the
    selection is named -- so the size beside it is that selected population's count, and the noun
    it renders is check runs rather than the records the population is made of.
    """
    records = (_check("failure", name="lint", pr=1), _check("failure", name="typecheck", pr=1))

    lint = _figure(_gate(records, "lint"), records)
    typecheck = _figure(_gate(records, "typecheck"), records)

    for figure in (lint, typecheck):
        denominator = figure.claim.denominator
        assert "commits where that check ran" in denominator.description
        assert "check runs" in denominator.description
        assert denominator.size_noun == "check runs"
        assert "1 check runs" in denominator.render_text()
    assert "lint" in lint.claim.denominator.description
    assert "typecheck" in typecheck.claim.denominator.description
    assert "Records span" in lint.claim.denominator.description


def test_the_two_denominators_are_not_interchangeable_because_the_populations_are_different_sets() -> (
    None
):
    """The whole reason there is one measure per gate, asserted on the numbers.

    The two figures together cover three check runs and neither denominator says three. Adding
    them up would be arithmetic over two populations that share nothing, which is the mistake a
    single distribution over all commits would have hidden rather than made.
    """
    records = (
        _check("failure", name="lint", pr=1),
        _check("failure", name="lint", pr=2),
        _check("success", name="typecheck", pr=1),
    )

    sizes = {
        measure.gate: measure.claim(build_snapshot(records)).denominator.size
        for measure in _gates(records)
    }

    assert sizes == {"lint": 2, "typecheck": 1}
    assert sum(sizes.values()) == len(records)


# -- what the claim has to carry -------------------------------------------------------------


def test_a_gate_figure_says_a_green_build_is_a_fact_about_the_build_and_not_about_the_change() -> (
    None
):
    """The same sentence the conclusion measure carries, imported rather than reworded.

    `success` is not `correct`: a green pipeline on a change that broke production passed its
    pipeline. Naming the gate makes the number actionable and does not make it a judgement, and
    the two constants are imported from `check_outcome` so the two figures cannot end up saying
    different things about the same fact.
    """
    figure = _figure(
        _gate((_check("success", name="lint"),), "lint"), (_check("success", name="lint"),)
    )

    assert SUCCESS_IS_NOT_CORRECTNESS in figure.claim.does_not_mean
    assert "broke production passed its pipeline" in figure.claim.does_not_mean


def test_a_gate_figure_says_reverts_are_out_of_reach_because_a_reverted_change_looks_like_one_that_stuck() -> (
    None
):
    """Naming the gate does not bring outcomes into reach, and the claim has to say so.

    Reverting a change is a separate event the store does not record, so a change that merged
    and was then reverted is indistinguishable here from one that stuck. Per-gate reporting makes
    this more tempting rather than less: "the lint gate passed 400 times" beside a reverted
    change reads as evidence about the change.
    """
    records = (_check("success", name="lint"),)

    figure = _figure(_gate(records, "lint"), records)

    assert REVERTS_ARE_OUT_OF_REACH in figure.claim.does_not_mean
    assert "merged and was later reverted" in figure.claim.does_not_mean


def test_every_gate_claim_passes_the_gate_and_names_the_ticket_that_delivered_the_fields() -> None:
    """Descriptive, repository-wide, and cited -- so a report renders it rather than refusing it.

    The gate refuses a comparative claim and anything below the team floor, and a per-gate
    failure rate is neither: it counts one population of check runs and compares nothing. The
    ticket is on the claim because `check_id` and `check_name` are what this measure reads, and
    a reader asking what changed deserves one reference.
    """
    records = (_check("failure", name="lint", pr=1),)

    claim = _gate(records, "lint").claim(build_snapshot(records))

    assert claim.unblocked_by == "RSF1DK1S"
    assert claim.slug == f"{CHECK_GATE_SLUG_PREFIX}lint"
    assert claim.slug not in _refusal_slugs()
    assert ClaimGate().check(claim) is None


def _refusal_slugs() -> set[str]:
    """The slugs the catalogue refuses, so a gate slug can be checked against them."""
    from tenbin.claims.registry import default_registry

    return {refusal.claim_slug for refusal in default_registry().all()}


def test_a_gate_figure_over_a_truncated_read_says_the_counts_are_an_undercount() -> None:
    """The completeness sentence rides with the figure, because a gate count is not less partial
    than any other.

    A walk stopped at the offset ceiling has seen some of each gate's runs and not others, and a
    per-gate distribution is exactly the shape where that is hardest to notice: every figure has
    bars in it, so nothing looks missing.
    """
    records = (_check("failure", name="lint", pr=1),)

    figure = _gate(records, "lint").compute(truncating_snapshot(records, missing=40, offset=100))

    assert isinstance(figure, Figure)
    assert "Truncated read" in figure.rate_text()
    assert "40 records were not reached" in figure.rate_text()


# -- the absent axis -------------------------------------------------------------------------


def test_a_corpus_with_no_check_runs_reports_the_axis_as_absent_and_fires_no_finding() -> None:
    """No figures at all, and no finding, because check runs are an opt-in subscription.

    A repository nobody subscribed check runs on has no check axis, and every rendering this
    module could produce over an empty one would be a claim about a pipeline nobody installed:
    a distribution of nothing reads as a healthy record, and a finding reads as a defect in the
    capture path, which check runs are not part of. So the factory returns nothing, and the
    absence is the whole answer.
    """
    records = (build_record(pr=1), build_record(pr=2))

    measures = check_gate_measures(records)

    assert measures == ()
    assert check_gate_identities(records) == ()
    assert not [
        measure
        for measure in measures
        if isinstance(measure.compute(build_snapshot(records)), Finding)
    ]


def test_no_figure_this_module_produces_is_a_finding_because_a_failing_gate_is_not_a_defect() -> (
    None
):
    """Checked over a corpus that *does* have gates, where the measure has something to say.

    The absent case above cannot distinguish "silent because there was nothing" from "silent
    because it never fires a finding". This one can: a corpus full of failing gates, and not one
    finding among the figures -- a gate that fails is a fact about a pipeline, and this program
    reports facts about pipelines without firing integrity checks at them.
    """
    records = (
        _check("failure", name="lint", pr=1),
        _check("startup_failure", name="lint", pr=2),
        _check("failure", name=None, check_id="4711", pr=3),
    )

    figures = [measure.compute(build_snapshot(records)) for measure in _gates(records)]

    assert len(figures) == 2
    assert not [figure for figure in figures if isinstance(figure, Finding)]


# -- a measure is a pure function of the read -------------------------------------------------


def test_two_reads_of_the_same_gate_produce_the_same_figure_because_a_gate_is_not_a_pagination() -> (
    None
):
    """Order-free and history-free, so a report on the same corpus twice renders the same thing.

    `histogram` sorts by count, so the buckets do not depend on walk order; what could have
    depended on it is which measure came first. The identities are sorted for the same reason
    `select` preserves order: a report that reordered itself between two runs of the same corpus
    would make the reader hunt for the figure they were looking at.
    """
    records = (
        _check("failure", name="typecheck", pr=1),
        _check("success", name="lint", pr=1),
        _check("failure", name="lint", pr=2),
    )

    forwards = [measure.slug for measure in _gates(records)]
    backwards = [measure.slug for measure in _gates(tuple(reversed(records)))]

    assert (
        forwards
        == backwards
        == [f"{CHECK_GATE_SLUG_PREFIX}lint", f"{CHECK_GATE_SLUG_PREFIX}typecheck"]
    )
