"""What got captured, and who was standing when they said it -- and the three ways
each of those is easy to over-read.

Three fields are parsed into :class:`~tenbin.corpus.record.Record` and used by no other
measure, and together they are the shape and the trust of the corpus.
``category`` is the closest thing the corpus has to saying what it is for, and it is
assigned by a model and stored without a review step, so the distribution is a
description of the categoriser rather than of importance.
``github_author_association`` is the half of the trust axis that is a fact about
position rather than a claim about derivation, and it must be reported without naming a
principal. ``independence_reason`` is the auditable phrase behind a level, and its whole
purpose is to show whether one level is doing the work of two situations.

The tests are grouped by the mistake each one would catch. The category half is about a
distribution whose *shape* must not depend on the corpus, about a mix that moves when the
mix of record kinds moves, and about a claim that has to say a model chose the labels. The
standing half is about the two axes being different kinds of fact and about the per-
principal output this must not become. The reason half is about the pairing being visible
at all, which two independent distributions cannot do.

A few of these assert on the *text* of a claim or a title rather than on a sort order,
because a ranking does not need an ordering: "the most trusted records in the corpus" is
a ranking with no sort in it, and a test on the order would pass straight over it.
"""

from __future__ import annotations

from tenbin.claims.gate import ClaimGate
from tenbin.claims.model import ClaimKind
from tenbin.corpus.record import Record, RecordKind
from tenbin.measures.base import DistributionFigure, Figure, FigureGroup
from tenbin.measures.category import (
    CATEGORY_POPULATION,
    CATEGORY_SLUG,
    LIFECYCLE_CATEGORY_TITLE,
    OTHER_CATEGORY_TITLE,
    UNRECOGNISED_CATEGORY_PREFIX,
    UNSTATED_CATEGORY_KEY,
    VERDICT_CATEGORY_TITLE,
    Category,
    CategoryMeasure,
    category_labels,
)
from tenbin.measures.independence_reason import (
    INDEPENDENCE_REASON_SLUG,
    KNOWN_INDEPENDENCE_REASONS,
    PAIRING_SEPARATOR,
    REASON_PAIRING_TITLE,
    UNSTATED_LEVEL_KEY,
    UNSTATED_REASON_KEY,
    IndependenceReasonMeasure,
    known_pairings,
    pairing_key,
)
from tenbin.measures.standing import (
    ASSOCIATION_TITLE,
    AUTHOR_STANDING_SLUG,
    UNRECOGNISED_ASSOCIATION_PREFIX,
    UNSTATED_ASSOCIATION_KEY,
    Association,
    AuthorAssociationMeasure,
)
from tenbin.measures.trust import (
    TRUST_HAS_TWO_AXES,
    TRUST_PROFILE_SLUG,
    Independence,
    TrustProfileMeasure,
)
from tests.fixtures import build_record, build_snapshot, truncating_snapshot

_CATEGORIES = CategoryMeasure()
_STANDING = AuthorAssociationMeasure()
_REASONS = IndependenceReasonMeasure()
_TRUST = TrustProfileMeasure()

#: A login that appears nowhere in any of these figures. Distinctive on purpose: a test
#: asserting that "dana" is absent would also pass if the module had never read a
#: comment author, and the point is that the figure is built *from records that have
#: one*, so the absence has to be a decision rather than an accident of the fixture.
_LOGIN = "quentin-the-owner"


def _lifecycle(*, pr: int, category: str | None = "system_event") -> Record:
    """A pull-request lifecycle record, which is the kind whose category is a constant.

    The tags are Kojutsu's own projection for the kind rather than a bare
    ``record_kind``, because a lifecycle record written with the key and no tags is a
    document the writer does not produce.
    """
    return build_record(
        pr=pr,
        record_kind=RecordKind.PR_LIFECYCLE,
        tags=["pr_state_change", "action_opened"],
        category=category,
    )


def _verdict(*, pr: int, category: str | None = "design_decision") -> Record:
    """A review verdict, which is the other kind whose category is a constant."""
    return build_record(
        pr=pr,
        record_kind=RecordKind.REVIEW_VERDICT,
        tags=["review", "review_state_approved"],
        category=category,
    )


def _answer(
    *,
    pr: int,
    category: str | None = None,
    independence: str | None = None,
    reason: str | None = None,
    association: str | None = None,
    login: str | None = _LOGIN,
) -> Record:
    """A captured answer, which is the kind that reaches all three of the other figures."""
    return build_record(
        pr=pr,
        record_kind=RecordKind.ANSWER,
        tags=["agent_authored"],
        category=category,
        independence=independence,
        independence_reason=reason,
        github_author_association=association,
        comment_author=login,
    )


def _category(figure: Figure, title: str) -> DistributionFigure:
    """The category figure with a given title, found by title rather than by position.

    By title because the order of the group is a presentation decision, and a test that
    indexed it would break the moment somebody decided the remainder read better first --
    without anything about the measure having changed.
    """
    for member in _distributions(figure):
        if member.title == title:
            return member
    raise AssertionError(f"no category figure titled {title!r}")


def _distributions(figure: Figure) -> tuple[DistributionFigure, ...]:
    """Every distribution in a figure or group, narrowed to the payload the tests read.

    Narrowing rather than a cast, because a test that indexes into a payload a figure
    does not have would be a test of a type error rather than of a measure, and the
    assertion is itself the assertion that the group really is a group of distributions.
    """
    members = figure.figures if isinstance(figure, FigureGroup) else (figure,)
    narrowed: list[DistributionFigure] = []
    for member in members:
        assert isinstance(member, DistributionFigure)
        narrowed.append(member)
    return tuple(narrowed)


def _distribution(figure: Figure) -> DistributionFigure:
    """The one distribution a measure returned, narrowed the same way."""
    distributions = _distributions(figure)
    assert len(distributions) == 1
    return distributions[0]


# -- the category distribution --------------------------------------------------------


def test_every_category_rung_is_present_at_zero_because_a_distribution_whose_shape_depends_on_the_corpus_is_one_every_report_has_to_guard_against() -> (
    None
):
    """Six rungs, in Kojutsu's order, whether or not anything landed in them.

    A reader comparing two runs has to be able to see that a category was empty rather
    than infer it from its absence, and an inferred absence is indistinguishable from a
    bucket that was filtered out. The order is the declaration order because these six
    values are a closed set and not a scale: there is no weakest category in this
    vocabulary, and a figure that came out in one would be an argument nobody made.
    """
    figure = _category(
        _CATEGORIES.compute(build_snapshot([_lifecycle(pr=1)])), LIFECYCLE_CATEGORY_TITLE
    )
    assert list(figure.values) == list(category_labels())
    assert category_labels() == (
        "design_decision",
        "trade_off",
        "domain_knowledge",
        "edge_case",
        "dependency",
        "system_event",
    )
    assert dict(figure.values)["system_event"] == 1
    assert all(count == 0 for label, count in figure.values.items() if label != "system_event")
    assert len(Category) == 6


def test_the_category_figure_splits_record_kinds_before_aggregating_because_two_of_the_six_categories_are_constants_of_the_kind() -> (
    None
):
    """The composition change that must not read as a change in what gets captured.

    This is the test that moves the *mix* rather than the labels, and it is the one that
    catches a regression: adding a verdict moves the mix of record kinds and moves every
    label at once on a blended figure, so a reader comparing two runs would see a change
    in what was captured where nothing about capture had changed. On the split figure the
    lifecycle distribution does not move at all, which is the only correct answer.
    """
    lifecycle = (_lifecycle(pr=1), _lifecycle(pr=2))
    before = _category(_CATEGORIES.compute(build_snapshot(lifecycle)), LIFECYCLE_CATEGORY_TITLE)
    after = _category(
        _CATEGORIES.compute(build_snapshot((*lifecycle, _verdict(pr=3)))),
        LIFECYCLE_CATEGORY_TITLE,
    )
    assert before.values == after.values
    assert dict(after.values)["system_event"] == 2

    verdicts = _category(
        _CATEGORIES.compute(build_snapshot((*lifecycle, _verdict(pr=3)))), VERDICT_CATEGORY_TITLE
    )
    assert dict(verdicts.values)["design_decision"] == 1
    assert dict(verdicts.values)["system_event"] == 0
    assert "constant" in LIFECYCLE_CATEGORY_TITLE
    assert "constant" in VERDICT_CATEGORY_TITLE


def test_the_three_category_figures_partition_the_capture_population_because_a_reader_holding_the_whole_thing_has_to_be_able_to_add_them_up() -> (
    None
):
    """Every capture record is in exactly one of the three figures.

    The protocol's rule is that a filtered record is either in ``values`` or in
    ``excluded``, and a partition that overlapped would break it in the direction that
    looks fine: two figures each claiming a record, and their counts adding to more than
    the denominator the claim names. A partition that left a gap would be the same
    arithmetic failing the other way, and a reader adding the three up would be short.
    """
    records = (
        _lifecycle(pr=1),
        _lifecycle(pr=2),
        _verdict(pr=3),
        _verdict(pr=4),
        _answer(pr=5, category="edge_case"),
        _answer(pr=6),
    )
    figure = _CATEGORIES.compute(build_snapshot(records))
    assert isinstance(figure, FigureGroup)
    assert len(figure.figures) == 3
    accounted = sum(member.counted + member.excluded_total for member in _distributions(figure))
    assert accounted == figure.claim.denominator.size == len(records)
    assert figure.claim.denominator.description == CATEGORY_POPULATION


def test_a_record_with_no_category_is_counted_as_an_absence_because_nobody_claimed_anything_about_what_kind_of_thing_it_is() -> (
    None
):
    """A missing category is excluded, not filed under a bucket named for the absence.

    A bucket labelled "no category" sitting beside the six with the same formatting is a
    distribution in which a group of records appears to have been categorised as nothing
    in particular, which is a claim about those records that Kojutsu never made. The
    empty string is here too, because it is the other way the same absence arrives.
    """
    records = (_answer(pr=1, category="edge_case"), _answer(pr=2), _answer(pr=3, category="  "))
    figure = _category(_CATEGORIES.compute(build_snapshot(records)), OTHER_CATEGORY_TITLE)
    assert dict(figure.values)["edge_case"] == 1
    assert dict(figure.excluded) == {UNSTATED_CATEGORY_KEY: 2}
    assert UNSTATED_CATEGORY_KEY not in figure.values
    assert all(count == 0 for label, count in figure.values.items() if label != "edge_case")


def test_a_category_outside_the_vocabulary_is_excluded_under_its_own_value_because_a_seventh_category_is_a_fact_about_a_writer_that_changed() -> (
    None
):
    """The value is in the key, and it is in neither of its neighbours.

    Folding it into ``edge_case`` or ``domain_knowledge`` would report a writer that has
    invented a category as a model that used an existing one, and the two are
    indistinguishable in the rendered figure -- which is the only place the difference
    would ever be visible. Two distinct unexpected values get two keys, because one
    would be a count of something no reader could name.
    """
    records = (
        _answer(pr=1, category="edge_case"),
        _answer(pr=2, category="retrospective_receipt"),
        _answer(pr=3, category="vibe_check"),
    )
    figure = _category(_CATEGORIES.compute(build_snapshot(records)), OTHER_CATEGORY_TITLE)
    assert dict(figure.values) == dict.fromkeys(category_labels(), 0) | {"edge_case": 1}
    assert dict(figure.excluded) == {
        f"{UNRECOGNISED_CATEGORY_PREFIX}retrospective_receipt": 1,
        f"{UNRECOGNISED_CATEGORY_PREFIX}vibe_check": 1,
    }


def test_a_category_read_as_the_corpus_placeholder_is_reported_as_an_absence_because_that_literal_is_not_kojutsus_on_this_field() -> (
    None
):
    """The one consequence of sharing ``parse_stated``, pinned so it cannot drift.

    ``parse_stated`` reads the literal ``"unknown"`` as an absence, because Kojutsu
    writes it into ``answered_by_model`` and ``declared_by_model`` where a principal
    named no model. It does not write it into ``category``, so here the literal is a
    *stated* value outside a closed six-value vocabulary -- and the honest handling of a
    seventh value is to say which one it was. This asserts the other behaviour, on
    purpose, because it is the cost of one reader for every axis and a cost nobody has
    chosen should be invisible.
    """
    figure = _category(
        _CATEGORIES.compute(build_snapshot([_answer(pr=1, category="unknown")])),
        OTHER_CATEGORY_TITLE,
    )
    assert figure.excluded == {UNSTATED_CATEGORY_KEY: 1}
    assert all(count == 0 for count in figure.values.values())


def test_the_category_claim_names_the_categoriser_because_the_labels_were_assigned_by_a_model_and_stored_without_a_review() -> (
    None
):
    """The distribution describes the categoriser, not the importance of what was captured.

    This is the caveat a figure of proportions invites most strongly, because the six
    values read as a taxonomy of what matters: `domain_knowledge` sounds important and
    `system_event` sounds trivial, and a reader who has not read Kojutsu's prompt will
    rank them anyway. Nothing in the corpus can check a single label, so the claim has to
    carry the fact that there was nothing to check them against.
    """
    claim = _CATEGORIES.claim(build_snapshot([_lifecycle(pr=1)]))
    assert _CATEGORIES.slug == CATEGORY_SLUG
    assert claim.slug == CATEGORY_SLUG
    assert claim.kind is ClaimKind.descriptive
    assert "A model assigned every one of these categories" in claim.does_not_mean
    assert "without a review step" in claim.does_not_mean
    assert "describes the categoriser's behaviour" in claim.does_not_mean
    assert "not the importance of what was captured" in claim.does_not_mean
    # And the split has to be in the negative space too, or a reader holding three
    # figures will add them and believe the sum.
    assert "Nor is the sum of the three figures a figure about anything" in claim.does_not_mean


def test_a_truncated_read_renders_the_truncation_sentence_on_every_category_figure_because_a_number_without_it_reads_as_general() -> (
    None
):
    """Every member of the group says how far the read got, with no caller-supplied wording.

    A distribution over four hundred documents looks exactly like a distribution over
    four thousand, and the sentence above it is the only thing standing between the
    number and a reader who assumes it covers everything the store holds.
    """
    records = (_lifecycle(pr=1), _verdict(pr=2))
    figure = _CATEGORIES.compute(truncating_snapshot(records, missing=3, offset=2))
    assert isinstance(figure, FigureGroup)
    for member in figure.figures:
        assert "Truncated read" in member.rate_text()
        assert "3 records were not reached" in member.rate_text()


# -- the standing half ----------------------------------------------------------------


def test_the_association_distribution_reports_who_was_entitled_to_speak_and_names_no_principal_because_that_is_per_principal_output() -> (
    None
):
    """No login anywhere in the figure, over records that all carry one.

    A report of who is an OWNER and who is a MEMBER, attached to a login, is a different
    product with different consent, retention and access requirements -- and the
    granularity floor exists because of exactly that. So the buckets are three words
    about a position and nothing else, and the claim says the absence is deliberate,
    because a figure that simply does not mention a person is indistinguishable from one
    where nobody thought about it.
    """
    records = (
        _answer(pr=1, association=Association.OWNER.value, login="dana"),
        _answer(pr=2, association=Association.OWNER.value, login="dana"),
        _answer(pr=3, association=Association.MEMBER.value, login=_LOGIN),
        _answer(pr=4, association=Association.COLLABORATOR.value, login=_LOGIN),
    )
    figure = _distribution(_STANDING.compute(build_snapshot(records)))
    assert dict(figure.values) == {"OWNER": 2, "MEMBER": 1, "COLLABORATOR": 1}
    rendered = f"{figure.title}\n{figure.value_text()}\n{figure.claim.render_text()}"
    for login in ("dana", _LOGIN, "comment_author", "reviewer"):
        assert login not in rendered, f"{login!r} identifies a principal by standing"
    assert "no account named" in ASSOCIATION_TITLE
    assert "no principal is identified by it" in figure.claim.does_not_mean


def test_the_association_axis_is_reported_beside_independence_because_they_are_two_kinds_of_fact_and_the_claim_says_so() -> (
    None
):
    """The two axes, and the sentence that says why they are two rather than one scale.

    The independence distribution on its own invites the misreading that independence is
    the whole of trustworthiness: it is ordinal, it is documented in ranked prose, and it
    is the strongest thing the corpus has. It is not the whole -- a record asserted by an
    OWNER and one asserted by a COLLABORATOR at the same level are not equally weighted.
    The trust claim has to name the second axis or a reader who never sees the standing
    figure has no way to know one exists, and both claims have to carry the *same*
    sentence or the two figures will drift into describing two different axes.
    """
    records = (
        _answer(
            pr=1,
            independence=Independence.INDEPENDENT.value,
            reason="different posting accounts",
            association=Association.OWNER.value,
        ),
        _answer(
            pr=2,
            independence=Independence.SELF_CERTIFIED.value,
            reason="same account, same model",
            association=Association.COLLABORATOR.value,
        ),
    )
    snapshot = build_snapshot(records)
    trust = _TRUST.claim(snapshot)
    standing = _STANDING.claim(snapshot)

    assert TRUST_HAS_TWO_AXES in trust.does_not_mean
    assert TRUST_HAS_TWO_AXES in standing.does_not_mean
    assert "derived from fields a record asserts about itself" in trust.does_not_mean
    assert "a fact about position" in trust.does_not_mean
    assert "self-asserted and unverifiable" in trust.does_not_mean
    assert AUTHOR_STANDING_SLUG in trust.does_not_mean

    # The same population as the trust figure, so a reader comparing the two axes is
    # comparing the same records and not two different reads.
    assert trust.denominator.size == standing.denominator.size == len(records)
    assert standing.kind is ClaimKind.descriptive
    assert ClaimGate().check(standing) is None


def test_a_record_with_no_association_and_an_association_outside_the_gate_are_two_different_facts() -> (
    None
):
    """An absence is excluded under a key naming it; a fourth association under its value.

    GitHub's own vocabulary for this field is wider than Kojutsu's gate -- ``CONTRIBUTOR``
    and ``NONE`` are real values the forge reports -- so the unrecognised path is a live
    case and not a formality. Folding one into ``COLLABORATOR`` would report a capture the
    gate is supposed to have refused as a capture it accepted, which is the one reading
    this axis exists to distinguish.
    """
    records = (
        _answer(pr=1, association=Association.MEMBER.value),
        _answer(pr=2, association=None),
        _answer(pr=3, association="  "),
        _answer(pr=4, association="CONTRIBUTOR"),
        _answer(pr=5, association="NONE"),
    )
    figure = _distribution(_STANDING.compute(build_snapshot(records)))
    assert dict(figure.values) == {"MEMBER": 1}
    assert dict(figure.excluded) == {
        UNSTATED_ASSOCIATION_KEY: 2,
        f"{UNRECOGNISED_ASSOCIATION_PREFIX}CONTRIBUTOR": 1,
        f"{UNRECOGNISED_ASSOCIATION_PREFIX}NONE": 1,
    }


def test_the_standing_measure_is_its_own_claim_because_a_slug_is_a_reports_key_and_two_measures_under_one_would_be_one_identifier_with_two_figures() -> (
    None
):
    """Its own slug, not the trust profile's, and not refused by the gate or the catalogue.

    Sharing a slug with :class:`~tenbin.measures.trust.TrustProfileMeasure` would give
    a report one identifier with two figures, and a reader matching a rendered figure
    against a slug would have no way to know which they had. Sharing the *sentence*
    about the two axes is what makes them one axis; sharing a slug would make them one
    measure, which they are not.
    """
    from tenbin.claims.registry import default_registry

    claim = _STANDING.claim(build_snapshot([_answer(pr=1, association="OWNER")]))
    assert _STANDING.slug == AUTHOR_STANDING_SLUG
    assert _STANDING.slug != TRUST_PROFILE_SLUG
    assert claim.slug == AUTHOR_STANDING_SLUG
    assert claim.slug not in default_registry()
    assert ClaimGate().check(claim) is None


# -- the reason half ------------------------------------------------------------------


def test_the_reason_distribution_is_a_cross_tabulation_because_two_distributions_leave_the_pairing_in_the_readers_head() -> (
    None
):
    """One bucket per pairing, so a reader can see one level arising from two situations.

    This is the question the axis is really asking: if ``self_certified`` comes from both
    "same account, same model" and "model not stated by both parties", the level is
    collapsing two different situations into one bucket, and a distribution of levels
    cannot show that while a distribution of reasons cannot either. Only the pairing can,
    which is why the key is composite and why it is spelled out in the title.
    """
    records = (
        _answer(
            pr=1,
            independence=Independence.SELF_CERTIFIED.value,
            reason="same account, same model",
        ),
        _answer(
            pr=2,
            independence=Independence.SELF_CERTIFIED.value,
            reason="same account; model not stated by both parties",
        ),
        _answer(
            pr=3,
            independence=Independence.SELF_CERTIFIED.value,
            reason="same account, same model",
        ),
        _answer(
            pr=4,
            independence=Independence.INDEPENDENT.value,
            reason="different posting accounts",
        ),
    )
    figure = _distribution(_REASONS.compute(build_snapshot(records)))
    assert (
        REASON_PAIRING_TITLE
        == "Independence level paired with the reason behind it, as `level/reason`"
    )
    assert figure.title == REASON_PAIRING_TITLE
    assert dict(figure.values)[pairing_key("self_certified", "same account, same model")] == 2
    assert (
        dict(figure.values)[
            pairing_key("self_certified", "same account; model not stated by both parties")
        ]
        == 1
    )
    # And the pairings that did not happen are visible rather than inferred.
    for absent in known_pairings():
        assert absent in figure.values
    assert (
        dict(figure.values)[pairing_key("model_separated", "same account, different models")] == 0
    )
    assert figure.excluded == {}


def test_a_reason_outside_the_four_known_ones_is_reported_under_its_own_key_because_kojutsu_writes_the_phrase_free_text() -> (
    None
):
    """A fifth phrase is information about a corpus that has changed, so it is not filtered.

    Kojutsu returns the phrase as a literal beside the level rather than as a member
    of an enum, so a fifth one is a code change upstream. Filtering on the known four
    would turn that change into silence, and silence is the one thing a measurement program
    must not produce when a writer has changed.
    """
    records = (
        _answer(pr=1, independence="independent", reason="asker vanished mid-thread"),
        _answer(
            pr=2,
            independence=Independence.INDEPENDENT.value,
            reason="different posting accounts",
        ),
    )
    figure = _distribution(_REASONS.compute(build_snapshot(records)))
    assert dict(figure.values)["independent/asker vanished mid-thread"] == 1
    assert figure.excluded == {}
    assert sum(figure.values.values()) == 2


def test_the_known_reasons_are_documented_and_never_used_as_a_filter_because_a_level_that_collapses_two_situations_is_the_finding() -> (
    None
):
    """Four phrases, two of them at the same level, which is the whole reason to hold them.

    The list exists to fill the empty pairings at zero and to give a test its fixtures --
    not to decide which reasons count. ``self_certified`` having two phrases behind it is
    what the falsifier is about, so it is asserted rather than left to be read out of a
    figure: a mapping that grew a fifth entry at a fourth level would be a claim about
    Kojutsu that this reader has no way to check from here.
    """
    levels = list(KNOWN_INDEPENDENCE_REASONS.values())
    assert len(KNOWN_INDEPENDENCE_REASONS) == 4
    assert levels.count(Independence.SELF_CERTIFIED.value) == 2
    assert sorted(set(levels)) == [
        Independence.INDEPENDENT.value,
        Independence.MODEL_SEPARATED.value,
        Independence.SELF_CERTIFIED.value,
    ]
    assert (
        KNOWN_INDEPENDENCE_REASONS["different posting accounts"] == Independence.INDEPENDENT.value
    )
    assert len(known_pairings()) == 4
    assert all(PAIRING_SEPARATOR in key for key in known_pairings())


def test_a_record_with_a_level_and_no_reason_is_counted_separately_from_one_with_no_level_because_those_are_two_different_gaps() -> (
    None
):
    """Two absences, two keys, and neither of them a bucket in the cross-tab.

    A record with no level at all was never assigned one; a record with a level and no
    phrase is a writer that stored the conclusion and not the derivation. Counting them
    together would say one thing about the corpus and the reader would not know which,
    and a record with a level in one bucket and an absence in the other is the
    "pairing" this figure exists to report.
    """
    records = (
        _answer(
            pr=1, independence=Independence.INDEPENDENT.value, reason="different posting accounts"
        ),
        _answer(pr=2, independence=Independence.SELF_CERTIFIED.value),
        _answer(pr=3, independence=Independence.SELF_CERTIFIED.value, reason="   "),
        _answer(pr=4),
    )
    figure = _distribution(_REASONS.compute(build_snapshot(records)))
    assert figure.excluded == {UNSTATED_LEVEL_KEY: 1, UNSTATED_REASON_KEY: 2}
    assert sum(figure.values.values()) + figure.excluded_total == figure.claim.denominator.size
    assert _REASONS.slug == INDEPENDENCE_REASON_SLUG
    assert figure.claim.slug == INDEPENDENCE_REASON_SLUG
    assert ClaimGate().check(figure.claim) is None


def test_a_reason_pairing_a_level_it_is_not_expected_at_keeps_its_own_bucket_because_the_contradiction_is_the_finding() -> (
    None
):
    """Two statements on one record that disagree are reported, not adjudicated.

    The level and the phrase are both written by the same writer at the same moment, and
    a record where they disagree is a contradiction this program cannot resolve. It
    reports both under the key they form; the corpus layer takes the same posture towards
    a tag and an id that disagree. Folding the phrase into the level it was expected at
    would make the cross-tab say the two agree.
    """
    records = (
        _answer(
            pr=1,
            independence=Independence.INDEPENDENT.value,
            reason="same account, same model",
        ),
    )
    figure = _distribution(_REASONS.compute(build_snapshot(records)))
    assert dict(figure.values)["independent/same account, same model"] == 1
    assert dict(figure.values)[pairing_key("self_certified", "same account, same model")] == 0
    assert figure.excluded == {}
