"""The command line's claim: a failure writes nothing, and no flag offers a claim this
corpus cannot support.

The two halves of that sentence are the two halves of what a command line is. It is
the easiest place for this program's discipline to rot, because a flag is a promise
to a reader and nothing enforces it: a measure's caveats are in the type, but a
``--granularity`` that accepts ``individual`` puts a per-principal ranking one
keystroke away from a person in a hurry, and no type is consulted until something
downstream reads it. So the flags are asserted here, by walking the command tree,
rather than trusted.

And the failure half is asserted by looking at the filesystem rather than at the
exit code. A non-zero exit with an empty report left on disk is a command that has
produced the most dangerous output this program can emit: a document that renders,
looks like an answer, and describes a corpus that was never read. So every store
failure test below checks that no file appeared and that nothing went to stdout --
not that the exit code was non-zero, which is the easy half.

The store is a :class:`~tests.fakes.FakeStore` in every test, injected through the
``CLIENT_FACTORY`` seam, and the suite's socket denial is therefore still a real
denial rather than a promise: a test that reached the network would raise.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from typer.main import get_command
from typer.testing import CliRunner

import tenbin.cli as cli
from tenbin.claims.refusals import NO_TICKET_UNBLOCKS_THIS
from tenbin.claims.registry import default_registry
from tenbin.corpus.snapshot import CorpusSnapshot
from tenbin.readmodel import build as build_read_model
from tenbin.readmodel import load as load_read_model
from tenbin.readmodel import write as write_read_model
from tests.conftest import plain_output
from tests.fakes import (
    FakeOptions,
    FakeStore,
    document_id,
    documents_for,
    json_response,
    kojutsu_frontmatter,
    store_over,
)
from tests.fixtures import FIXED_READ_AT, SnapshotBuilder, build_answers, build_snapshot

#: Words that make a flag a request for a comparison, a baseline or a delta. Matched
#: anywhere in a flag name rather than as whole words, because the shapes that get
#: added in a hurry are compounds: ``--compare-to``, ``--versus-baseline``,
#: ``--peer-team``. Substring matching is the conservative direction here: a false
#: positive costs a flag name, a false negative costs a report.
COMPARISON_SHAPED = re.compile(
    r"compare|versus|baseline|benchmark|against|delta|diff|peer|improvement|regression"
)

#: The granularity this corpus refuses in the type, below the floor and regardless of
#: it. A flag offering it would be a flag accepting an illegal value.
REFUSED_GRANULARITY = "individual"

#: How a whole read reads in a report, quoted from the report layer rather than
#: re-spelled here: a test that spelled the sentence itself would go on passing after
#: the wording changed, and this is the sentence that distinguishes an empty corpus
#: from a corpus that was never reached.
WHOLE_READ = "the store was walked to its end and its own count agreed"

runner = CliRunner()


@pytest.fixture
def store_url(monkeypatch: pytest.MonkeyPatch) -> str:
    """A store URL in the environment, for the commands that are allowed to need one.

    The conftest removes every ``TENBIN_`` variable before each test, so a test that
    wants configuration has to ask for it -- which is what makes the two commands that
    need no configuration able to prove they do without one.
    """
    url = "http://127.0.0.1:8088/v1"
    monkeypatch.setenv("TENBIN_TANSEKI_URL", url)
    return url


def a_store(documents: dict[str, dict[str, Any]] | None = None, **options: Any) -> FakeStore:
    """A store that answers in process, and never retried so a test stays fast."""
    return FakeStore(documents, options=FakeOptions(max_retries=0, **options))


def use(monkeypatch: pytest.MonkeyPatch, store: FakeStore) -> FakeStore:
    """Hand the command this store, through the one seam that reaches a socket."""
    monkeypatch.setattr(cli, "CLIENT_FACTORY", lambda settings: store)
    return store


def a_corpus_in_the_store() -> dict[str, dict[str, Any]]:
    """Four answers and a verdict, with the keys the new measures read.

    The verdict carries ``review_id``, ``record_kind`` and ``change_author_account``
    because a report that renders no verdicts would leave the measures this command
    exists to run untested by their own output.
    """
    documents = documents_for(4, repo="acme/widget")
    documents[document_id("acme/widget", 9, "review-1")] = kojutsu_frontmatter(
        tags=["review", "review_state_approved"],
        repo="acme/widget",
        pr=9,
        record_kind="review_verdict",
        review_id="R-1",
        change_author_account="alice",
        pr_opened_at="2026-09-01T10:00:00+00:00",
    )
    return documents


def _descend(command: Any, path: tuple[str, ...] = ()) -> Iterator[tuple[str, Any, Any]]:
    """The sub-commands of one command, recursively, with their full names.

    Duck-typed rather than imported from the click internals: the vendored module is
    private to Typer and a test that reaches into it breaks on a Typer upgrade, while
    everything this module needs -- ``commands``, ``params``, ``opts`` and a
    parameter's ``choices`` -- is public on the objects themselves. Starting from a
    command rather than from the root is what keeps the recursion from re-walking the
    root's own sub-commands at every level.
    """
    for name, sub in getattr(command, "commands", {}).items():
        here = (*path, name)
        yield " ".join(here), here, sub
        yield from _descend(sub, here)


def walk() -> Iterator[tuple[str, Any, Any]]:
    """Every command in the tree below the root, with its name and its parameters."""
    return _descend(get_command(cli.app))


def flags() -> Iterator[tuple[str, str, tuple[str, ...]]]:
    """Every option in the tree: the command it is on, its flag, and its choices."""
    for label, _, command in walk():
        for parameter in command.params:
            if getattr(parameter, "param_type_name", None) != "option":
                continue
            for flag in (*parameter.opts, *getattr(parameter, "secondary_opts", ())):
                choices = tuple(getattr(parameter.type, "choices", ()) or ())
                yield label, flag, choices


# -- the failures, which are the part that matters ------------------------------------


def test_an_unreachable_store_writes_nothing_because_a_report_of_zeroes_is_the_output_this_program_refuses(
    monkeypatch: pytest.MonkeyPatch, store_url: str, tmp_path: Path
) -> None:
    """A store that is down is a failure, and it leaves no document behind.

    The assertion is on the filesystem and on stdout, not on the exit code. An empty
    report and an empty corpus render almost identically -- both are a page of
    zeroes -- so a command that printed its error and also wrote the report would
    have published a finding about a corpus it never read. The message says which of
    the two it was, because "nothing to report" and "nothing to report *from*" are
    different sentences.
    """
    use(monkeypatch, a_store(fail_status=503))
    destination = tmp_path / "report.md"

    result = runner.invoke(cli.app, ["report", "--output", str(destination)])

    assert result.exit_code == cli.EXIT_STORE_UNREADABLE
    assert not destination.exists(), "a report was written about a corpus that was never read"
    assert result.stdout == "", "nothing may go to stdout on this path either"
    assert "could not be reached" in result.output
    assert "nothing was written" in result.output


def test_a_store_that_refused_the_credential_names_the_variable_to_set_and_writes_nothing_because_retrying_a_key_fixes_nothing(
    monkeypatch: pytest.MonkeyPatch, store_url: str, tmp_path: Path
) -> None:
    """A refused key is an operator action, and the message says which line to change.

    The same guarantee as an unreachable store -- nothing written -- and a different
    remedy, which is why it is a separate path rather than one generic error: waiting
    does not fix a credential, and an operator who is told only that "the store
    could not be read" will wait.
    """
    use(monkeypatch, a_store(fail_status=401))
    destination = tmp_path / "report.md"

    result = runner.invoke(cli.app, ["report", "--output", str(destination)])

    assert result.exit_code == cli.EXIT_STORE_UNREADABLE
    assert not destination.exists()
    assert result.stdout == ""
    assert cli.STORE_KEY_VARIABLE in result.output
    assert "retrying will not help" in result.output


def test_an_unconfigured_store_names_the_variable_and_writes_nothing_because_a_default_store_is_a_store_nobody_asked_for(
    tmp_path: Path,
) -> None:
    """No store URL is a failure that says so, not a walk of something plausible.

    ``tenbin.config`` has no default URL on purpose, and this is the test that keeps
    that decision honest at the command line: a program with a default store reports
    a real number about whichever store happens to be running, which is a finding
    about somebody else's corpus.
    """
    destination = tmp_path / "report.md"

    result = runner.invoke(cli.app, ["report", "--output", str(destination)])

    assert result.exit_code == cli.EXIT_STORE_UNREADABLE
    assert not destination.exists()
    assert result.stdout == ""
    assert cli.STORE_URL_VARIABLE in result.output
    assert "not configured" in result.output


def test_a_reachable_store_holding_nothing_is_a_report_saying_so_and_exits_zero_because_that_is_a_finding_rather_than_a_failure(
    monkeypatch: pytest.MonkeyPatch, store_url: str
) -> None:
    """Zero documents is a measurement. Zero documents *because the store was down* is not.

    The distinction is the whole difference between a finding and a failure, and it
    is invisible in the figures: an empty corpus and an unreachable one produce the
    same zeros. So this asserts the exit code and the header together -- the report
    says the store was walked to its end and that it enumerated nothing, and the run
    succeeds, because that is what happened.
    """
    use(monkeypatch, a_store())

    result = runner.invoke(cli.app, ["report"])

    assert result.exit_code == cli.EXIT_OK
    assert "0 of 0 documents" in result.output
    assert WHOLE_READ in result.output


# -- the report, and the model it can be measured from --------------------------------


def test_a_report_written_to_a_file_is_the_report_that_would_have_been_printed_because_the_delivery_is_not_the_document(
    monkeypatch: pytest.MonkeyPatch, store_url: str, tmp_path: Path
) -> None:
    """``--output`` moves the bytes; it does not render anything differently.

    A command that re-rendered for a file would produce a document nobody compared
    with the one on the screen, and the two would drift. So the file is asserted
    equal to the stdout of the same run over the same corpus, and the two runs are
    pinned to one store instance so the only difference is where the text went.
    """
    store = use(monkeypatch, a_store(a_corpus_in_the_store()))
    destination = tmp_path / "nested" / "report.md"

    printed = runner.invoke(cli.app, ["report"])
    written = runner.invoke(cli.app, ["report", "--output", str(destination)])

    assert printed.exit_code == written.exit_code == cli.EXIT_OK
    assert _without_read_at_text(destination.read_text(encoding="utf-8")) == _without_read_at_text(
        printed.stdout
    )
    assert store.requests, "the fake must have been read for the comparison to mean anything"


def test_a_report_measured_from_a_read_model_is_the_report_measured_from_the_store_because_reading_from_the_model_is_a_performance_choice(
    monkeypatch: pytest.MonkeyPatch, store_url: str, tmp_path: Path
) -> None:
    """The two paths differ in one field, and that field is the moment of the read.

    Everything else -- every section, every claim, every caveat, every figure -- is
    asserted equal, with ``read_at`` set aside because it is *meant* to differ: the
    model's report describes the read the model holds and the live one describes the
    read it just made. If anything else moved, the model would be a second source of
    truth wearing the name of a cache, and this is the test that says it is not.
    """
    use(monkeypatch, a_store(a_corpus_in_the_store()))
    model_path = tmp_path / "read-model.json"
    built = runner.invoke(cli.app, ["read-model", "build", "--output", str(model_path)])
    assert built.exit_code == cli.EXIT_OK
    kept_at = _kept_at(built.stdout)

    live = runner.invoke(cli.app, ["report", "--format", "json"])
    from_model = runner.invoke(
        cli.app, ["report", "--format", "json", "--read-model", str(model_path)]
    )

    assert live.exit_code == from_model.exit_code == cli.EXIT_OK
    assert _without_read_at(json.loads(live.stdout)) == _without_read_at(
        json.loads(from_model.stdout)
    )
    # The two differ in exactly one field, and the difference is the point: the model's
    # report describes the read the model holds -- stamped when the model was built,
    # not when the report was rendered -- while the live one describes the walk it just
    # made. Asserted as an identity and an ordering rather than as an inequality, since
    # a clock coarse enough to make two walks agree would make an inequality flaky.
    assert json.loads(from_model.stdout)["corpus"]["read_at"] == kept_at
    assert json.loads(live.stdout)["corpus"]["read_at"] >= kept_at


def test_a_read_model_this_program_does_not_implement_stops_the_run_rather_than_being_ignored_because_a_version_skew_is_a_fact_a_human_has_to_resolve(
    tmp_path: Path,
) -> None:
    """A model at another schema version is an error, and it writes nothing.

    Falling back to a live walk would produce a perfectly good report and hide the
    skew, which is the failure this exists to prevent: the operator cannot rebuild a
    model they were never told was unreadable. Note that no store is configured here
    either -- the run cannot silently have fallen back, because there was nothing to
    fall back to.
    """
    model_path = _a_model_file(tmp_path)
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    payload["schema_version"] = "a version from the future"
    model_path.write_text(json.dumps(payload), encoding="utf-8")
    destination = tmp_path / "report.md"

    result = runner.invoke(
        cli.app, ["report", "--read-model", str(model_path), "--output", str(destination)]
    )

    assert result.exit_code == cli.EXIT_STORE_UNREADABLE
    assert not destination.exists()
    assert result.stdout == ""
    assert "Rebuild the model" in result.output


def test_read_model_build_replaces_the_whole_model_because_an_update_path_is_where_a_derived_model_becomes_a_maintained_one(
    monkeypatch: pytest.MonkeyPatch, store_url: str, tmp_path: Path
) -> None:
    """Two builds, two corpora, and the second is all that survives.

    A partial update would leave a model holding records from two reads, whose
    completeness sentence would describe a walk that never happened. So the second
    build is asserted to hold the second corpus *entirely* -- the first corpus's
    records are gone rather than merged in -- and the directory is checked for the
    temporary file an interrupted rename would leave.
    """
    model_path = tmp_path / "read-model.json"
    use(monkeypatch, a_store(a_corpus_in_the_store()))
    assert (
        runner.invoke(cli.app, ["read-model", "build", "--output", str(model_path)]).exit_code == 0
    )
    first = load_read_model(model_path)
    assert first is not None

    use(monkeypatch, a_store(documents_for(2, repo="other/place")))
    assert (
        runner.invoke(cli.app, ["read-model", "build", "--output", str(model_path)]).exit_code == 0
    )

    second = load_read_model(model_path)
    assert second is not None
    assert {record.doc_id for record in second.records} == {
        document_id("other/place", 1, "answer-0000"),
        document_id("other/place", 1, "answer-0001"),
    }
    assert not {record.doc_id for record in first.records} & {
        record.doc_id for record in second.records
    }
    assert sorted(candidate.name for candidate in tmp_path.iterdir()) == [model_path.name]


def test_integrity_only_runs_the_measures_about_the_capture_system_because_a_reader_asking_whether_to_believe_the_rest_does_not_want_the_rest(
    monkeypatch: pytest.MonkeyPatch, store_url: str
) -> None:
    """Two measures instead of eleven, and the refusals are still all there.

    The refusals stay on an integrity report because they are the answer to "what
    did I not get?", which is the question somebody asking this has. What is left
    out is the descriptions of the work: they are not wrong, they are not what was
    asked, and a report that always contained everything would have no such flag.
    """
    use(monkeypatch, a_store(a_corpus_in_the_store()))

    full = json.loads(runner.invoke(cli.app, ["report", "--format", "json"]).stdout)
    integrity = json.loads(
        runner.invoke(cli.app, ["report", "--format", "json", "--integrity-only"]).stdout
    )

    rendered = [section["slug"] for section in integrity["sections"] if section["kind"] == "claim"]
    # Completeness first, then the rest in registry order. A check that found nothing
    # is a count rather than a finding, so it lands in the "rest" partition -- and the
    # partition is the report's rule about which claims qualify the others, not a
    # statement that an unremarkable check is less important than one that fired.
    assert rendered == [
        "corpus.read_completeness",
        "corpus.lifecycle_shape_checked",
        "corpus.capture_freshness_checked",
        "corpus.trust_profile",
        "corpus.projection_safety_checked",
        "corpus.question_capture_agreement_checked",
    ]
    assert len(integrity["sections"]) < len(full["sections"])
    assert [s for s in integrity["sections"] if s["kind"] == "refusal"] == [
        s for s in full["sections"] if s["kind"] == "refusal"
    ]


# -- the reasoning, which needs no store at all ----------------------------------------


def test_the_reasoning_commands_never_open_the_store_because_a_refusal_you_cannot_see_without_a_deployment_is_a_property_of_that_deployment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``claims`` and ``refusals`` run with no store, no configuration and no socket.

    The factory here raises rather than returning a fake, so the assertion is not
    "the fake was never called" but "nothing could have been called": if either
    command reached the store, this test would fail with the assertion rather than
    with a network error the reader would have to interpret. The reason it matters
    is the one in the module docstring -- the refusal catalogue has to be a property
    of the program, inspectable before there is anything to inspect.
    """

    def unreachable(settings: Any) -> FakeStore:
        raise AssertionError(f"the store must not be reached, and it was asked for {settings}")

    monkeypatch.setattr(cli, "CLIENT_FACTORY", unreachable)

    claims = runner.invoke(cli.app, ["claims"])
    listed = runner.invoke(cli.app, ["refusals"])

    assert claims.exit_code == cli.EXIT_OK
    assert listed.exit_code == cli.EXIT_OK
    assert "corpus.read_completeness" in claims.output
    assert "What would unblock it" in listed.output


def test_claims_names_the_denominator_size_only_when_a_read_supplied_it_because_a_size_over_an_empty_corpus_is_a_number_nobody_wrote_down(
    tmp_path: Path,
) -> None:
    """Without a read, the description is printed and the size is declared dependent.

    Every denominator in this program is a count over a particular read, so a
    ``claims`` listing that wanted a size had to invent one -- and inventing it from
    the empty corpus it is avoiding would print ``(1 records)`` under a dozen claims,
    which is precisely the fabricated population the whole package exists to refuse.
    """
    model_path = _a_model_file(tmp_path)

    without = runner.invoke(cli.app, ["claims"])
    with_read = runner.invoke(cli.app, ["claims", "--read-model", str(model_path)])

    assert without.exit_code == with_read.exit_code == cli.EXIT_OK
    assert cli.DENOMINATOR_FROM_READ in without.output
    assert cli.DENOMINATOR_FROM_READ not in with_read.output
    assert f"({len(build_answers(3)):,} records)" in with_read.output


def test_a_read_model_flag_naming_a_file_that_is_not_there_prints_no_denominator_size_because_the_flag_is_not_the_read(
    tmp_path: Path,
) -> None:
    """The second half of that test: a flag with nothing behind it is still no read.

    ``--read-model`` naming a path that does not exist falls back to the read that
    never happened, and a caller that trusted the flag rather than what it resolved to
    would print a denominator size over an empty corpus. That is a fabricated
    population under a dozen claims, and it is exactly what
    :data:`tenbin.cli.DENOMINATOR_FROM_READ` exists to prevent -- so the check is on
    what the command resolved, not on what it was told.
    """
    result = runner.invoke(
        cli.app, ["claims", "--read-model", str(tmp_path / "no-such-model.json")]
    )

    assert result.exit_code == cli.EXIT_OK
    assert cli.DENOMINATOR_FROM_READ in result.output
    assert "(1 records)" not in result.output, (
        "a count over the empty read this command falls back to is a number nobody wrote down"
    )


def test_require_all_exits_non_zero_because_a_refusal_displayed_on_a_page_is_not_usable_in_a_check() -> (
    None
):
    """The listing is unchanged; only the exit code says anything is refused.

    Both halves of that are the point. Without the flag the command is a page, and a
    page cannot fail a build. With it, the run is red while the catalogue is
    non-empty -- which is today, deliberately, because a pipeline that watches this
    can see the count change, and a check that only fails on a brand new refusal
    would have told the team nothing about the seven that were already there.
    """
    listed = runner.invoke(cli.app, ["claims"])
    checked = runner.invoke(cli.app, ["claims", "--require-all"])

    assert listed.exit_code == cli.EXIT_OK
    assert checked.exit_code == cli.EXIT_REFUSED
    assert checked.stdout == listed.stdout, "the check changes the verdict, not the document"


def test_the_refusal_listing_says_where_no_ticket_will_ever_unblock_it_because_an_empty_field_looks_like_somebody_is_working_on_it() -> (
    None
):
    """Every entry, and the ones with no ticket render the sentence rather than nothing.

    Three refusals are in that state now, and one of them is the project's central
    question. A blank field would be read as "the reference is merely not written
    down", and the reader would wait for a ticket nobody is going to file -- which is
    the specific way a decision turns back into a queue item.
    """
    result = runner.invoke(cli.app, ["refusals"])
    registry = default_registry()

    for refusal in registry.all():
        assert refusal.claim_slug in result.output
        assert refusal.reason in result.output

    without_tickets = [refusal for refusal in registry.all() if refusal.unblocked_by is None]
    assert len(without_tickets) == 4
    assert result.output.count(f"What would unblock it: {NO_TICKET_UNBLOCKS_THIS}") == len(
        without_tickets
    )
    assert "whether-agentic-development-is-more-effective" in result.output


def test_an_unsafe_store_url_is_named_as_a_configuration_problem_and_writes_nothing_because_a_plain_http_store_would_carry_the_key_in_the_clear(
    monkeypatch: pytest.MonkeyPatch, store_url: str, tmp_path: Path
) -> None:
    """A URL this program refuses to dial is reported before anything is measured.

    The same "nothing written" guarantee as the other two store failures, and a
    different cause again: the client refuses a plain-HTTP URL for anything that is
    not loopback, because that URL would carry the API key and the corpus in the
    clear. The refusal happens inside the client, which is why the message names
    ``TENBIN_TANSEKI_URL`` rather than the store.
    """
    monkeypatch.setattr(
        cli,
        "CLIENT_FACTORY",
        lambda settings: store_over(
            lambda request: json_response({}), tanseki_url="http://insecure.example/v1"
        ),
    )
    destination = tmp_path / "report.md"

    result = runner.invoke(cli.app, ["report", "--output", str(destination)])

    assert result.exit_code == cli.EXIT_STORE_UNREADABLE
    assert not destination.exists()
    assert cli.STORE_URL_VARIABLE in result.output
    assert "TLS" in result.output


def test_a_store_that_answers_with_something_this_program_cannot_use_is_named_as_such_because_waiting_does_not_fix_an_envelope(
    monkeypatch: pytest.MonkeyPatch, store_url: str, tmp_path: Path
) -> None:
    """A broken envelope is not an unreachable store, and the message must not conflate them.

    The fake is serving a listing with no ``total`` in it, which is a real shape some
    Tanseki endpoints have and one the walk cannot check itself against -- so the first
    request raises rather than returning an empty snapshot. "The store could not be
    read" is the honest summary, and it is deliberately not "could not be reached":
    a reader who retries will get the same envelope, and one who is told to wait has
    been told to do the wrong thing.
    """
    use(monkeypatch, a_store(omit_total=True))
    destination = tmp_path / "report.md"

    result = runner.invoke(cli.app, ["report", "--output", str(destination)])

    assert result.exit_code == cli.EXIT_STORE_UNREADABLE
    assert not destination.exists()
    assert result.stdout == ""
    assert "could not be read" in result.output
    assert "no total" in result.output, "the message has to carry the cause, not a summary of it"


def test_read_model_build_with_nowhere_to_put_it_says_so_before_reading_the_store_because_a_cache_nobody_can_name_is_not_a_cache(
    monkeypatch: pytest.MonkeyPatch, store_url: str
) -> None:
    """No ``--output`` and no configured path is a usage error, and it is cheap.

    Checked before the walk, which is asserted rather than asserted-in-prose: the fake
    raises if it is asked for a client, so a command that read a thousand documents
    before discovering it had nowhere to put them would fail here. The message names
    both places a destination can come from rather than only the flag the operator did
    not pass.
    """

    def unreachable(settings: Any) -> FakeStore:
        raise AssertionError("the store must not be reached before the destination is known")

    monkeypatch.setattr(cli, "CLIENT_FACTORY", unreachable)

    result = runner.invoke(cli.app, ["read-model", "build"])

    # Styled output would break the flag assertion on terminals that render
    # color (click highlights `--options` as split spans): read the sentence,
    # not the styling. See `plain_output` in `tests/conftest.py`.
    output = plain_output(result.output)
    assert result.exit_code == 2, "a usage error, which is what click reserves 2 for"
    assert "TENBIN_READ_MODEL_PATH" in output
    assert "--output" in output


# -- the flags that must not exist ------------------------------------------------------


def test_no_flag_offers_individual_because_a_flag_accepting_an_illegal_value_is_a_flag_that_will_be_passed() -> (
    None
):
    """No option anywhere in the tree offers the granularity the type refuses.

    ``Granularity.individual`` is refused below the floor *and* refused whatever the
    floor is, so a flag that offered it would be offering a value that no
    configuration can make renderable. It would also be offering it one keystroke
    away from a reader who has just been told this program is about teams.
    """
    offered = [
        (label, flag, choice)
        for label, flag, choices in flags()
        for choice in choices
        if choice == REFUSED_GRANULARITY
    ]

    assert offered == [], f"a flag offers a refused granularity: {offered}"
    assert not [flag for _, flag, _ in flags() if flag.lstrip("-") == "granularity"], (
        "there is no granularity flag at all, and a flag accepting only the legal "
        "values would still be an invitation to pass the illegal one"
    )


def test_no_flag_is_shaped_like_a_comparison_because_a_comparison_needs_a_mechanism_this_corpus_does_not_have() -> (
    None
):
    """Nothing in the tree reads as "against something else".

    A comparative claim needs an assignment mechanism and this corpus has none
    (:mod:`tenbin.claims.mechanism`), so the gate refuses every one of them. A
    comparison-shaped flag is therefore a flag that will be passed by somebody in a
    hurry and refused at the far end -- or, worse, wired to something that quietly
    reads one group as the baseline. The check is on flag *names* across the whole
    tree, including subcommands, because the flag that breaks this is the one
    somebody adds to the command that is easiest to reach for.
    """
    found = [
        (label, flag)
        for label, flag, _ in flags()
        if COMPARISON_SHAPED.search(flag.lstrip("-").casefold())
    ]

    assert found == [], f"comparison-shaped flags exist: {found}"


def test_the_surface_is_six_commands_and_the_read_model_has_exactly_one_verb_because_each_of_them_is_a_decision_somebody_had_to_make() -> (
    None
):
    """The whole tree, pinned.

    Not tidiness: a sixth command is somebody's decision about what this program
    publishes, and a second verb under ``read-model`` is somebody's decision about
    what a kept read can be. Both belong in a review of this file, where the reasons
    are written down, rather than in a diff that only shows the new name.

    Three things have been added to a pinned surface since it was first written, and
    each earned its place by the same argument.

    ``--resume`` on ``read-model build`` is here because a boolean that decides *how
    the read is assembled* is as much a decision about what this program publishes as
    a verb is. It survives the comparison-shaped check above because resuming is not
    a comparison: it changes which windows a read was walked in, and never what one
    window is held against.

    ``drift`` is here because it is the first command that renders a figure about
    *time* rather than about one read. **It passing the comparison-shaped check two
    tests above is itself a claim somebody should check**, and the reason it passes
    is worth stating rather than leaving to a reader: a drift compares one population
    observed twice, so it compares no principal against another and needs no
    assignment mechanism. ``--corpus-event`` and ``--archive`` are here for the same
    reason -- naming the event that would confound a pair, and naming the log the
    readings came from, are not comparisons.

    ``retrospective`` is here because it is the first command over the *second*
    source, and its flags are the interesting part. ``--cadence`` and ``--anchor``
    pass this file's comparison-shaped check because neither compares anything: they
    *declare* a period that nothing in either system declares, and there is no default
    for either because a defaulted cadence would scope every figure in the output to a
    boundary nobody chose while the report said nothing about it. ``--period`` and
    ``--at`` are the two ways of saying which period, and they are mutually exclusive
    by construction rather than by precedence. ``--terminal`` is the one flag here
    that chooses a population, and it is pinned below precisely because that is the
    shape of flag that usually goes wrong: it names which states count as finished,
    and whichever set was used is printed in the claim of every figure that depends
    on it, so the choice cannot be made invisibly.

    **``report``'s second-source flags are pinned for the same reasons, and the reuse is
    the thing being pinned.** ``--cadence``, ``--anchor``, ``--at`` and ``--period`` on
    ``report`` are the *same* flags, resolving through the same two helpers, so the two
    commands cannot drift into disagreeing about which window they measured. They are
    pinned on both commands precisely because a flag set present on one and absent from
    the other is how a reader comes to believe the two commands scope their figures
    differently -- and the second one was added only after the first already existed.

    ``--source`` is here because it decides *which corpus*, not anything about one, and it
    is explicit rather than inferred for the reason ``--read-model`` is explicit above: a
    report that silently preferred a source would describe a population the operator never
    named, and the header's date is not enough to reveal the substitution.

    ``--project`` and ``--group`` are here because they name *where in the backend* to
    look, not which states count as finished. The flag that would choose a population is
    ``--terminal``, and ``report --source ticket`` deliberately has no equivalent: the
    dwell measures it runs report on every state the read observed, so there is no
    population choice available to make invisibly.
    """
    assert {label for label, _, _ in walk()} == {
        "report",
        "claims",
        "refusals",
        "drift",
        "retrospective",
        "read-model",
        "read-model build",
    }
    assert {(label, flag) for label, flag, _ in flags()} == {
        ("report", "--format"),
        ("report", "--output"),
        ("report", "--read-model"),
        ("report", "--integrity-only"),
        ("report", "--archive"),
        ("report", "--source"),
        ("report", "--cadence"),
        ("report", "--anchor"),
        ("report", "--at"),
        ("report", "--period"),
        ("report", "--project"),
        ("report", "--group"),
        ("claims", "--read-model"),
        ("claims", "--require-all"),
        ("drift", "--archive"),
        ("drift", "--corpus-event"),
        ("drift", "--documented-by"),
        ("drift", "--read-model"),
        ("drift", "--format"),
        ("drift", "--output"),
        ("retrospective", "--cadence"),
        ("retrospective", "--anchor"),
        ("retrospective", "--at"),
        ("retrospective", "--period"),
        ("retrospective", "--terminal"),
        ("retrospective", "--project"),
        ("retrospective", "--group"),
        ("retrospective", "--format"),
        ("retrospective", "--output"),
        ("retrospective", "--archive"),
        ("read-model build", "--output"),
        ("read-model build", "--resume"),
    }


def test_the_html_format_is_wired_and_ships_the_caveats_with_the_charts_because_a_chart_is_the_best_way_to_lose_a_bound(
    monkeypatch: pytest.MonkeyPatch,
    store_url: str,
) -> None:
    """**The one thing this test exists to catch is a chart without its bound.**

    A chart is the most effective device ever built for separating a number from the
    caveats that make it true: lift the picture, drop the prose, and a reader has
    "51 MEMBER" with nothing about who was positioned to disagree or what the corpus
    does not cover. So the wiring is asserted here rather than inferred from the
    renderer having its own tests -- ``--format html`` reaching ``render_html`` is a
    line in a dispatch, and a line in a dispatch is exactly the kind of thing that
    silently stops being called.
    """
    use(monkeypatch, a_store(a_corpus_in_the_store()))

    result = runner.invoke(cli.app, ["report", "--format", "html"])

    assert result.exit_code == 0, result.output
    rendered = result.stdout
    assert "<svg" in rendered, "no chart was drawn, so --format html is not reaching render_html"
    assert "<script" not in rendered, (
        "the page carries JavaScript, and this report has to render its own caveats on a "
        "machine with no network -- a page that needs a CDN to show its caveats renders "
        "without them when the CDN is unreachable"
    )
    for field in ("What it does not mean", "What would falsify it", "Denominator"):
        assert field in rendered, f"the page lost {field!r}, which is the caveat a chart strips"


# -- the archive and the one figure that compares time --------------------------------


def test_archiving_appends_and_a_second_read_renders_a_drift_because_a_programme_with_no_memory_is_a_snapshot_repeated(
    tmp_path: Path,
) -> None:
    """The whole loop, through the command line, over two reads of two corpora.

    Written here rather than only in ``test_drift.py`` because the library tests
    prove the measure and the archive are each correct, and this proves the two
    commands are wired to them. Those are different failures: an ``--archive``
    flag that appends nothing, and a ``drift`` command that renders a library
    object nobody can reach, both pass every other test in this file.
    """
    log = tmp_path / "readings.jsonl"
    first = _a_read_model(tmp_path, 3)
    second = _a_read_model(tmp_path, 7, days_later=7)

    wrote = runner.invoke(cli.app, ["report", "--read-model", str(first), "--archive", str(log)])
    assert wrote.exit_code == 0, wrote.output
    assert wrote.stdout.startswith("#"), (
        "the archive confirmation went to stdout, so it became the report's first "
        f"line: {wrote.stdout[:80]!r}"
    )
    after_first = log.read_text().splitlines()
    assert after_first, "nothing was archived"

    again = runner.invoke(cli.app, ["report", "--read-model", str(second), "--archive", str(log)])
    assert again.exit_code == 0, again.output
    lines = log.read_text().splitlines()
    assert len(lines) == 2 * len(after_first), (
        "the second read did not append; an archive that merges is a second source of truth"
    )
    assert lines[: len(after_first)] == after_first, (
        "the first read's lines changed, which is the rewriting this archive refuses"
    )

    drift = runner.invoke(
        cli.app,
        ["drift", "corpus.read_completeness", "--archive", str(log), "--read-model", str(second)],
    )
    assert drift.exit_code == 0, drift.output
    assert "corpus.read_completeness" in drift.stdout


def test_one_archived_read_is_a_sentence_and_a_non_zero_exit_because_a_drift_needs_two(
    tmp_path: Path,
) -> None:
    """The failure mode worth a test is the one that would be a traceback."""
    log = tmp_path / "one.jsonl"
    model = _a_read_model(tmp_path, 2)
    assert (
        runner.invoke(
            cli.app, ["report", "--read-model", str(model), "--archive", str(log)]
        ).exit_code
        == 0
    )

    drift = runner.invoke(
        cli.app,
        ["drift", "corpus.read_completeness", "--archive", str(log), "--read-model", str(model)],
    )
    assert drift.exit_code == cli.EXIT_REFUSED
    assert "needs two" in drift.output
    assert "Traceback" not in drift.output


def test_a_second_read_of_the_same_model_is_refused_rather_than_appended_because_two_entries_for_one_read_is_a_drift_that_never_happened(
    tmp_path: Path,
) -> None:
    """A fixed ``read_at`` is what makes this reachable, and the message says so.

    A read model is stamped once, so ``report --read-model m --archive a`` run
    twice is the ordinary way to hit this.
    """
    log = tmp_path / "twice.jsonl"
    model = _a_read_model(tmp_path, 2)
    assert (
        runner.invoke(
            cli.app, ["report", "--read-model", str(model), "--archive", str(log)]
        ).exit_code
        == 0
    )
    again = runner.invoke(cli.app, ["report", "--read-model", str(model), "--archive", str(log)])
    assert again.exit_code != 0
    assert "Traceback" not in again.output


def test_the_refusal_listing_says_what_the_refusals_do_not_refuse_because_a_silent_gate_reads_as_a_refusal_of_all_comparison() -> (
    None
):
    """A refusal cannot say what survives it, so somewhere else has to.

    Without this, the only way to learn that a temporal claim is permitted is to
    read ``002`` and then read ``gate.py``. Both are real, and neither is where a
    reader who ran ``tenbin refusals`` is standing.
    """
    result = runner.invoke(cli.app, ["refusals"])
    assert result.exit_code == 0, result.output
    assert "WHAT THESE DO NOT REFUSE" in result.output
    assert "read twice" in result.output


# -- helpers ----------------------------------------------------------------------------


def _a_model_file(tmp_path: Path) -> Path:
    """A kept read built from a fixture corpus, with no store involved at all.

    Built through the read model's own API rather than by running the command, so a
    test about what the command *does* with a model is not also a test of the command
    that writes one.
    """
    snapshot: CorpusSnapshot = build_snapshot(build_answers(3))
    path = tmp_path / "read-model.json"
    write_read_model(path, build_read_model(snapshot))
    return path


#: The rendered line naming the moment of the read. Matched rather than compared,
#: because two walks of the same corpus microseconds apart differ in that line and
#: nowhere else, and a test that failed on the difference would be a test about the
#: clock's resolution.
READ_AT_LINE = re.compile(r"^- \*\*Read at:\*\* .*$", re.MULTILINE)

#: The status line ``read-model build`` prints, which is where a test learns the moment
#: the model was taken rather than having to read it back out of the file.
KEPT_AT = re.compile(r"read at (\S+)")


def _a_read_model(tmp_path: Path, answer_count: int, *, days_later: int = 0) -> Path:
    """A kept read of a corpus holding ``answer_count`` answers, at its own path.

    **Two different corpora are not two readings.** The archive identifies a
    reading by ``(slug, read_at)`` and refuses a second one at the same moment,
    because two entries for one read is a drift that never happened. So the second
    model here is both a different size *and* a later read -- otherwise the loop
    under test would be exercising the duplicate refusal instead of the drift, and
    would pass for the wrong reason.
    """
    destination = tmp_path / f"model-{answer_count}-{days_later}.json"
    snapshot = SnapshotBuilder(
        records=build_answers(answer_count),
        read_at=FIXED_READ_AT + timedelta(days=days_later),
    ).build()
    write_read_model(destination, build_read_model(snapshot))
    return destination


def _kept_at(status: str) -> str:
    """The moment the build reported, so a report can be checked against it."""
    found = KEPT_AT.search(status)
    assert found is not None, f"the build did not report when it read: {status!r}"
    return found.group(1)


def _without_read_at_text(rendered: str) -> str:
    """A rendered report with the moment of the read blanked out."""
    return READ_AT_LINE.sub("- **Read at:** (the moment of this read)", rendered)


def _without_read_at(payload: dict[str, Any]) -> dict[str, Any]:
    """A report payload with the moment of the read taken out.

    Only that one field is allowed to differ between a live report and one measured
    from a kept read, so the comparison strips it here rather than the test asserting
    equality and then explaining the exception.
    """
    return {**payload, "corpus": {**payload["corpus"], "read_at": None}}
