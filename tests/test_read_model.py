"""The read model's claim: it is a kept read, and deleting it changes nothing.

Every test here is about one of two properties, and they are different in kind.
The first is that a read model is **derived**: a figure computed from a rebuilt
model equals a figure computed from the corpus, so the model can be deleted at any
moment without changing a number. That is what makes it safe to keep and what makes
it a cache rather than a source of truth, and it is the property that cannot be
asserted by inspecting the model -- it has to be asserted by throwing the model away
and measuring again.

The second is that the file is either **whole or absent**. There is no third state
in which a model parses, names a collection, and holds a fraction of the records its
completeness sentence claims to cover: that object would render figures whose
caveat says the walk reached the end of the store over a corpus that is part of it,
and nothing in the output would say otherwise. So an unparseable file is absent, a
file written by another schema version is *refused* rather than ignored, and an
interrupted write leaves the previous model rather than a truncated one.

The classification tests are here because "store it rather than recompute it" is a
claim about what the model *holds*, and the only way to check it is to change the
stored answer and see that the change survives. A reader that re-derived the
classification would quietly discard the edit, and a model that disagreed with its
own classifier is precisely the drift ``SCHEMA_VERSION`` exists to prevent.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from tenbin.corpus.provenance import NO_MONTH, NO_REPOSITORY
from tenbin.corpus.record import Certainty, RecordKind
from tenbin.corpus.snapshot import Completeness, CorpusSnapshot, Truncation
from tenbin.measures import CompletenessMeasure, RepositoryCoverageMeasure
from tenbin.readmodel import (
    SCHEMA_VERSION,
    ReadModel,
    UnknownSchemaVersionError,
    build,
    load,
    to_snapshot,
    write,
)
from tenbin.store.client import StoreResponseError
from tests.fixtures import (
    FIXED_READ_AT,
    build_answers,
    build_record,
    build_snapshot,
    legacy_record,
)

#: The measures the derived-not-authoritative property is asserted with. Two, and
#: deliberately of different kinds: one is a distribution over the records, so it
#: fails if a record was lost, and one is a rate whose denominator is the store's own
#: count, so it fails if the model's *header* drifted from its records. A single
#: measure could pass with a model that kept the records and lost the counts.
FIGURES = (RepositoryCoverageMeasure(), CompletenessMeasure())


def a_corpus() -> CorpusSnapshot:
    """A small corpus with one of everything the read model has to survive.

    Records that fall into more than one bucket, because an index is only tested by
    a corpus that has something to group: two repositories, a document whose id
    carries no ``pr`` segment at all, a record with no timestamp of any kind, a
    verdict, a legacy document written before ``record_kind`` existed, and a legacy
    document with no tags either -- so the stored classifications cover both a
    determination and an inference, which are different facts about a record.
    """
    return build_snapshot(
        (
            *build_answers(2, repo="acme/widget"),
            *build_answers(1, repo="acme/gadget"),
            build_record(
                entry_id="review-1",
                pr=9,
                record_kind=RecordKind.REVIEW_VERDICT,
                tags=["review", "review_state_approved"],
                repo="acme/widget",
            ),
            build_record(doc_id="loose-document", repo=None, pr=None, updated_at=None),
            legacy_record(doc_id="acme/widget/pr-11/rationale-v1-0", tags=["rationale"]),
            legacy_record(doc_id="acme/widget/pr-12/answer-0"),
        )
    )


def test_a_figure_computed_from_a_rebuilt_read_model_equals_one_computed_from_the_corpus_because_a_derived_model_is_not_a_second_source_of_truth() -> (
    None
):
    """Build, write, delete, rebuild, and measure: the number does not move.

    The deletion is the test. An assertion that a loaded model equals the model that
    was written only proves the file format is symmetric; what has to be shown is
    that the model carries nothing the corpus does not, and the only evidence of
    that is measuring the corpus again and getting the same answer. So the file is
    deleted between the two builds -- an operator with no read model and an operator
    with one must be able to compare their reports, and this is the comparison.
    """
    snapshot = a_corpus()
    kept = _fresh_dir() / "read-model.json"
    model = build(snapshot)
    write(kept, model)
    assert kept.exists()

    kept.unlink()
    assert not kept.exists(), "the model must be gone before the rebuild, or this proves nothing"

    rebuilt = build(snapshot)
    from_rebuilt = to_snapshot(rebuilt)
    assert rebuilt == model
    for measure in FIGURES:
        assert measure.compute(from_rebuilt) == measure.compute(snapshot)
        assert measure.claim(from_rebuilt) == measure.claim(snapshot)


def test_a_model_written_and_read_back_is_the_model_that_was_written_because_a_cache_that_changes_what_it_holds_is_not_a_cache() -> (
    None
):
    """The round trip is an equality of values, not of renderings.

    Asserted on the whole model rather than on a rendering, because a text
    comparison would pass for a payload that lost a field the readers here do not use
    -- and the fields nobody reads today are the ones that bite when somebody
    rebuilds a corpus with them. The indexes are included: they are derived rather
    than stored, so this also pins that loading rebuilds them from the records
    instead of trusting anything the file said about them.
    """
    model = build(a_corpus())

    restored = _loaded(_written(model))

    assert restored == model
    assert restored.by_id == model.by_id
    assert restored.by_repo == model.by_repo
    assert restored.by_month == model.by_month
    assert restored.by_kind == model.by_kind


def test_the_classification_is_stored_rather_than_recomputed_because_a_model_that_re_derives_it_is_a_second_implementation_of_the_rule() -> (
    None
):
    """An edited classification survives the round trip, which is the proof it was stored.

    The edit is to one record's ``certainty``, ``evidence`` and ``detail``, held in
    the file, and the assertion is that the loaded model carries the edited answer.
    A reader that called :func:`tenbin.corpus.record.classify` again would produce
    the original verdict and pass a test that only compared the two models, so the
    file is edited between the write and the read: the only way this fails is if the
    classification is derived on the way in.
    """
    path = _written(build(a_corpus()))
    payload = _payload(path)
    payload["records"][0]["classification"] = {
        "kind": RecordKind.ANSWER.value,
        "certainty": Certainty.INFERRED.value,
        "evidence": ["no tags"],
        "detail": "edited by a test, to prove this is read rather than derived",
    }
    _save(path, payload)

    restored = _loaded(path)

    first = restored.records[0].classification
    assert first.certainty is Certainty.INFERRED
    assert first.evidence == ("no tags",)
    assert "edited by a test" in first.detail
    assert restored.classifications[restored.records[0].doc_id] == first


def test_a_model_this_program_does_not_implement_is_refused_rather_than_ignored_because_a_version_skew_is_a_fact_a_human_has_to_resolve() -> (
    None
):
    """A file at another schema version raises instead of quietly walking the store.

    The tempting behaviour is to treat it as absent and read the corpus, which
    produces a correct report and hides the skew entirely. That is worse than
    useless here: the whole reason the version is written into the file is so that a
    model this program cannot interpret is *visible*, and a caller cannot rebuild one
    it does not know exists.
    """
    path = _written(build(a_corpus()))
    payload = _payload(path)
    payload["schema_version"] = "a version from the future"
    _save(path, payload)

    with pytest.raises(UnknownSchemaVersionError, match="Rebuild the model"):
        load(path)


def test_a_model_that_cannot_be_parsed_is_treated_as_absent_because_half_a_read_model_is_worse_than_none() -> (
    None
):
    """Five unreadable files, all of them absent, none of them half-trusted.

    The shapes are the ones that actually happen: an interrupted write, a file that
    is not JSON, a JSON document that is not an object, an object with no version to
    check, and an object whose records are the wrong type. Each of them is a file an
    operator can have on disk, and each of them must send the caller back to the
    store -- because a caller that got a model back from any of them would render
    figures over a corpus it cannot describe, with a completeness sentence saying the
    walk reached the end of the store.
    """
    unreadable = {
        "interrupted write": '{"schema_version": "1", "records": [{"id": "acme/wid',
        "not json": "this file is not json",
        "not an object": '["schema_version", "1"]',
        "no version to check": '{"collection": "chronicler-real", "records": []}',
        "records of the wrong type": json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "collection": "chronicler-real",
                "read_at": FIXED_READ_AT.isoformat(),
                "enumerated": 0,
                "store_total": 0,
                "completeness": Completeness.COMPLETE.value,
                "records": "six of them",
            }
        ),
    }
    for name, content in unreadable.items():
        path = _fresh_dir() / f"{name}.json"
        path.write_text(content, encoding="utf-8")
        assert load(path) is None, name

    assert load(_fresh_dir() / "no-such-file.json") is None


def test_an_interrupted_write_leaves_the_previous_model_intact_because_a_truncated_model_reads_as_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two interruptions, and the file on disk is the model that was there before.

    There are two things that can go wrong, and they happen at different moments. A
    payload JSON cannot hold fails while the temporary file is being written, and a
    rename can fail after that. In both cases the previous model survives intact --
    which is the property that matters, because a half-written model does not raise
    on the way back in, it reads as absent, and the operator loses a read without
    being told. The temporary file is checked for as well, because a failed build
    that leaves ``.read-model.json.1234.tmp`` beside the model is a build whose
    leftovers somebody has to learn to ignore.
    """
    path = _written(build(a_corpus()))
    before = path.read_text(encoding="utf-8")
    replacement = build(build_snapshot(build_answers(3, repo="other/place")))

    with pytest.raises(TypeError):
        write(path, _unserialisable(replacement))
    assert path.read_text(encoding="utf-8") == before
    assert _siblings(path) == [path], "a failed write must leave no temporary file behind"

    def refuse(source: object, destination: object) -> None:
        raise OSError(f"rename refused: {source} -> {destination}")

    monkeypatch.setattr(os, "replace", refuse)
    with pytest.raises(OSError, match="rename refused"):
        write(path, replacement)
    assert path.read_text(encoding="utf-8") == before
    assert _siblings(path) == [path], "a failed rename must leave no temporary file behind"

    monkeypatch.undo()
    assert load(path) == build(a_corpus())


def test_the_indexes_name_the_records_by_the_axes_a_report_groups_by_including_the_ones_the_records_do_not_have() -> (
    None
):
    """Four indexes, and the two sentinels the corpus package already owns.

    The sentinels are the interesting half. A record with no repository and a record
    with no month are real documents, and filing them under a bucket some other layer
    invented would give a caller two different names for the same absence -- so the
    keys are ``NO_REPOSITORY`` and ``NO_MONTH``, imported rather than re-spelled, and
    a measure that already groups by them cannot double-count. The legacy records are
    the other half: an index holds them under the kind the classifier gave them, and
    the model carries the certainty that says whether that was a statement or a
    reading of an absence.
    """
    model = build(a_corpus())

    assert set(model.by_id) == {record.doc_id for record in model.records}
    assert model.by_repo[NO_REPOSITORY] == ("loose-document",)
    assert model.by_repo["acme/widget"][0].endswith("answer-0000")
    assert model.by_month[NO_MONTH] == ("loose-document",)
    assert model.by_kind[RecordKind.REVIEW_VERDICT.value] == ("acme/widget/pr-9/review-1",)
    assert model.by_kind[RecordKind.RATIONALE.value] == ("acme/widget/pr-11/rationale-v1-0",)
    assert (
        model.classifications["acme/widget/pr-11/rationale-v1-0"].certainty is Certainty.DETERMINED
    )
    assert "acme/widget/pr-12/answer-0" in model.by_kind[RecordKind.ANSWER.value]
    assert model.classifications["acme/widget/pr-12/answer-0"].certainty is Certainty.INFERRED, (
        "a document with no kind and no tags is inferred, and the stored reading says so"
    )
    assert model.records_in(model.by_repo, "acme/gadget")[0].repo == "acme/gadget"
    assert model.records_in(model.by_repo, "nowhere") == ()


def test_a_read_model_carries_the_verdict_about_how_far_the_walk_got_because_a_read_that_looks_whole_is_the_failure_this_program_exists_to_stop() -> (
    None
):
    """A truncated read restores as truncated, and a failed one still carries its failure.

    Both are the state a figure's completeness sentence is built from, and a model
    that dropped them on the way through the file would render a prefix of the corpus
    while claiming the walk reached the end of it. The failure is restored as the
    class of error that stopped the walk, because "the store could not be reached"
    and "the store answered with something this program cannot use" call for
    different hours of somebody's night.
    """
    truncated = CorpusSnapshot(
        records=(build_record(pr=3),),
        collection="chronicler-real",
        read_at=FIXED_READ_AT,
        store_total=400,
        enumerated=200,
        completeness=Completeness.OFFSET_CAP,
        truncation=Truncation(
            offset_reached=200,
            records_missing=200,
            boundary_repositories={},
            boundary_months={},
        ),
    )

    restored = to_snapshot(_loaded(_written(build(truncated))))

    assert restored.completeness is Completeness.OFFSET_CAP
    assert restored.truncation == truncated.truncation
    assert restored.store_total == 400
    assert restored.enumerated == 200
    assert "records were not reached" in CompletenessMeasure().compute(restored).rate_text()

    failed = CorpusSnapshot(
        records=(build_record(pr=3),),
        collection="chronicler-real",
        read_at=FIXED_READ_AT,
        store_total=400,
        enumerated=1,
        completeness=Completeness.STORE_ERROR,
        error=StoreResponseError("three listed documents were not returned by the store"),
    )

    restored_failure = to_snapshot(_loaded(_written(build(failed))))

    assert restored_failure.completeness is Completeness.STORE_ERROR
    assert isinstance(restored_failure.error, StoreResponseError)
    assert "not returned by the store" in str(restored_failure.error)
    assert "did not finish" in CompletenessMeasure().compute(restored_failure).rate_text()


def test_a_model_whose_records_disagree_with_each_other_is_refused_because_an_index_over_it_would_have_to_pick_a_winner() -> (
    None
):
    """Two records under one document id is a contradiction, not a longer corpus.

    A walk cannot produce it, so a model holding it was built by something else --
    and every index over it would have to keep one of the two while the figures
    counted both. That is the "two refusals for one measure" defect from
    :mod:`tenbin.claims.registry`, in a different layer, and the answer is the
    same: refuse rather than let one silently win.
    """
    model = build(a_corpus())
    doubled = (*model.records, model.records[0])

    with pytest.raises(ValueError, match="enumerated one document twice"):
        replace(model, records=doubled)


def test_the_package_publishes_a_rebuild_and_nothing_else_because_an_update_path_is_where_a_derived_model_becomes_a_maintained_one() -> (
    None
):
    """The public surface is exactly build, write, load and convert.

    The claim in the module docstring is a claim about what this package must never
    grow, and a docstring does not survive a deadline -- somebody under one adds
    ``update``, ``merge`` or ``append`` because a use case is real and the mechanism
    is obvious. So the surface is asserted: eight names, and any ninth is a refusal
    that has to be argued for in this file first.
    """
    import tenbin.readmodel as readmodel

    assert set(readmodel.__all__) == {
        "SCHEMA_VERSION",
        "ReadModel",
        "StoredFailure",
        "UnknownSchemaVersionError",
        "build",
        "load",
        "to_snapshot",
        "write",
    }
    for verb in ("update", "merge", "append", "refresh", "invalidate", "patch"):
        assert not hasattr(readmodel, verb), f"{verb} would be an update path"


def test_a_model_at_another_schema_version_is_refused_at_write_because_a_file_this_program_cannot_read_is_not_a_cache() -> (
    None
):
    """Writing is the other half of the version contract, not just reading it.

    :func:`write` refuses a model whose version is not this program's, so a caller
    cannot hand-build one at version zero, write it, and then discover at read time
    that the file is unusable. The error names both versions, because the fix is to
    rebuild and the person who has to do it should not have to work out which side
    moved.
    """
    model = build(a_corpus())

    with pytest.raises(ValueError, match="refusing to write a read model"):
        write(_fresh_dir() / "model.json", replace(model, schema_version="0"))


def test_a_read_model_states_which_schema_wrote_it_and_stores_no_index_because_a_second_copy_of_a_derived_fact_is_a_second_source_of_truth() -> (
    None
):
    """The version is in the payload, first, and the indexes are not.

    The version first because it is the one field a future reader has to check before
    trusting anything else. The indexes absent because they are derived: a stored
    index is a second copy of a fact the records already carry, and a file whose
    index disagreed with its own records would load into a model that lies about its
    contents with no way to tell which half to believe.
    """
    path = _written(build(a_corpus()))

    payload = _payload(path)

    assert list(payload)[:6] == [
        "schema_version",
        "collection",
        "read_at",
        "enumerated",
        "store_total",
        "completeness",
    ]
    assert payload["schema_version"] == SCHEMA_VERSION
    assert not {"by_id", "by_repo", "by_month", "by_kind"} & set(payload)
    kinds = {kind.value for kind in RecordKind}
    assert {entry["classification"]["kind"] for entry in payload["records"]} <= kinds


def test_a_model_whose_header_says_nonsense_is_refused_rather_than_repaired_because_a_repaired_read_would_report_the_repair() -> (
    None
):
    """Three header fields a read cannot be built from, each refused with its reason.

    A blank collection, a completeness value that is a bare string, and a moment with
    no zone. None of them can come out of a walk, so a model holding one was written by
    something else -- and the answer is a refusal rather than a repair, because a
    repaired read would report whatever the repair decided the corpus was. The
    completeness case is the sharpest: a bare ``"complete"`` would restore as a
    verdict this program never reached.
    """
    model = build(a_corpus())

    with pytest.raises(ValueError, match="collection is required"):
        replace(model, collection="   ")
    with pytest.raises(ValueError, match="must be a Completeness"):
        replace(model, completeness="complete")
    with pytest.raises(ValueError, match="must be an aware datetime"):
        replace(model, read_at=datetime(2026, 9, 30, 12, 0, 0))
    with pytest.raises(ValueError, match="non-negative integer count"):
        replace(model, store_total=-1)


def test_a_failure_this_version_does_not_know_is_still_a_failure_because_the_class_name_decides_what_an_operator_does() -> (
    None
):
    """An unrecognised failure class restores as a store error that names what it was.

    The alternative is to drop it, which would turn a walk that failed into a walk that
    completed and leave every figure in the report claiming a whole corpus. The
    restored error keeps the class name in its message, so the operator can tell what
    the file recorded even though this version cannot raise it itself.
    """
    from tenbin.readmodel import StoredFailure

    restored = StoredFailure(kind="SomeFutureStoreError", message="the walk stopped").restore()

    assert isinstance(restored, StoreResponseError)
    assert "SomeFutureStoreError" in str(restored)
    assert "the walk stopped" in str(restored)


# -- helpers for the tests above ------------------------------------------------------


def _fresh_dir() -> Path:
    """A directory of this test's own, so a leftover file cannot look like a stale model.

    ``tmp_path`` is not used because two tests in this module deliberately write
    files that are meant to be left behind, and the ones that assert "nothing was
    written" have to be able to say which directory they meant.
    """
    return Path(tempfile.mkdtemp())


def _written(model: ReadModel) -> Path:
    """A model written to a fresh file, and the path to read it back from."""
    path = _fresh_dir() / "read-model.json"
    write(path, model)
    return path


def _loaded(path: Path) -> ReadModel:
    """The model at ``path``, having asserted that there is one.

    The assertion is here rather than at each call site because every caller in this
    module needs a model and none of them is testing :func:`load` -- the tests of that
    are the ones that expect ``None`` -- so repeating ``assert ... is not None`` would
    be noise in every test and a missing check in one of them.
    """
    model = load(path)
    assert model is not None, f"{path} was written by this program and must load"
    return model


def _siblings(path: Path) -> list[Path]:
    """Every file in the model's directory, so a leftover temporary file is visible."""
    return sorted(candidate for candidate in path.parent.iterdir() if candidate.is_file())


def _payload(path: Path) -> dict[str, Any]:
    """The file as JSON, for a test that is about to edit it."""
    return json.loads(path.read_text(encoding="utf-8"))


def _save(path: Path, payload: dict[str, Any]) -> None:
    """Write an edited payload back, in the shape a human editing the file would."""
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _unserialisable(model: ReadModel) -> ReadModel:
    """A model whose frontmatter cannot be written as JSON, to fail mid-write.

    The value goes in the frontmatter rather than in a header field because that is
    where a store can put something JSON cannot hold: the header is written by this
    program and is a handful of strings and integers, while the frontmatter is
    whatever a writer stored.
    """
    poisoned = replace(model.records[0], frontmatter={"when": object()})
    return replace(model, records=(poisoned, *model.records[1:]))
