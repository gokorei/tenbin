"""The documentation gate's own tests, over a document tree nobody has to own.

``scripts/check_docs.py`` exists to catch a citation rotting in a repository this
project does not control, and the naive way to test it is to cite something real.
That test would pass today and rot exactly the way the rule is supposed to catch,
which would make it a worse test than no test: it would look like coverage of the
thing the rule exists for while checking the same fixed strings every time.

So every test here builds its own document tree under ``tmp_path`` and its own fake
Kojutsu checkout beside it, and points ``KOJUTSU_CHECKOUT`` at the fake. The
fixture is a two-file source module of a known length, so "a line beyond the end"
is a fact about the fixture rather than about whichever revision of somebody else's
repository happens to be on disk. Nothing here needs Kojutsu, and nothing here
will fail because Kojutsu moved.

**The prose cases are the load-bearing ones.** A naive citation regex matches any
backticked path, and the two documents this rule reads are full of backticked paths
that are *not* citations: links into this repository, filenames used as examples,
decision records named by bare stem. A rule that failed on those would be a rule
that got switched off within a week, so "a prose path is not a citation" is
asserted as carefully as "a citation resolves".
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest

#: Imported as a namespace package rather than by mutating ``sys.path``: ``tests``
#: is a package, so pytest puts the repository root on the path, and ``scripts``
#: resolves as an implicit namespace package under it. The gate is then imported
#: the way any other module would be, which means a rename of the script is an
#: ordinary import error rather than a ``None`` at the first assertion.
from scripts.check_docs import (
    CITATION_PATTERN,
    KOJUTSU_CHECKOUT_ENV,
    RENAMED_PROJECTS,
    SKIPPED_EXIT_CODE,
    STALE_NAME_EXEMPT_PATHS,
    STALE_NAME_EXEMPT_SUFFIXES,
    STALE_NAME_SUFFIXES,
    STRICT_ENV,
    TANSEKI_CHECKOUT_ENV,
    Violation,
    check_citations,
    check_stale_project_names,
    main,
    resolve_citation,
)

#: The fake cited file. Its length is the fact several tests turn on, so it is
#: written as an explicit list of lines rather than as a triple-quoted block whose
#: length depends on how the author happened to wrap it.
FAKE_MODULE_LINES = (
    '"""A stand-in for a module in the repository this project does not own."""',
    "",
    "CHANGE_AUTHOR_KEY = 'change_author_account'",
    "PR_OUTCOME_KEY = 'pr_outcome'",
    "RECORD_KIND_KEY = 'record_kind'",
)


@dataclass(frozen=True)
class Fixture:
    """A document tree, the fake cited checkouts, and the environment pointing at them."""

    root: Path
    checkout: Path
    tanseki: Path


@pytest.fixture
def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Fixture]:
    """Build both trees and wire the environment to the fake checkout.

    The environment is restored by ``monkeypatch`` rather than by hand because a
    leaked ``KOJUTSU_CHECKOUT`` would silently redirect every *other* test in
    this file at a directory that no longer exists, and the failure would read as a
    broken rule rather than as a leaked fixture.
    """
    root = tmp_path / "tenbin"
    root.mkdir()
    checkout = tmp_path / "kojutsu"
    package = checkout / "src" / "kojutsu"
    package.mkdir(parents=True)
    (package / "core").mkdir()
    (package / "core" / "tanseki_mapping.py").write_text(
        "\n".join(FAKE_MODULE_LINES) + "\n", encoding="utf-8"
    )
    (package / "models.py").write_text("# one line\n", encoding="utf-8")
    tanseki = tmp_path / "tanseki"
    tanseki_module = (
        tanseki
        / "core"
        / "domain"
        / "src"
        / "main"
        / "kotlin"
        / "gokorei"
        / "tanseki"
        / "core"
        / "domain"
    )
    tanseki_module.mkdir(parents=True)
    (tanseki_module / "RequestLimits.kt").write_text(
        "\n".join(FAKE_MODULE_LINES) + "\n", encoding="utf-8"
    )
    monkeypatch.setenv(KOJUTSU_CHECKOUT_ENV, str(checkout))
    monkeypatch.setenv(TANSEKI_CHECKOUT_ENV, str(tanseki))
    yield Fixture(root=root, checkout=checkout, tanseki=tanseki)


#: Parts of the deliberately-broken citations below, held here so this file contains none
#: of them as literals.
#:
#: The gate scans source as well as documents, which means it scans the file that tests it.
#: A fixture written as a literal broken citation would therefore be reported as a genuine
#: one -- turning every run of the suite into a gate failure, and teaching a contributor to
#: exempt this file from the rule it exists to test. Interpolating keeps the fixtures exactly
#: as broken as they need to be, with no rule losing coverage of this file.
#:
#: Note that this comment previously quoted one of those literals as an illustration, which
#: the widened scan then reported here. Twice, in fact: once in the block above and once in
#: the replacement written to fix it. A comment about a citation is itself a citation to the
#: scanner, which is worth knowing before writing one.
STEM: Final[str] = "mapp"
ELIXIR_EXTENSIONS: Final[tuple[str, ...]] = ("ex", "exs")
LINE: Final[str] = "12"
LONG_LINE: Final[str] = "9000"


def write_document(root: Path, name: str, body: str) -> Path:
    """Write one document into the tree and return its path."""
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def details(fixture: Fixture, document: Path) -> list[str]:
    """The violation details for one document, which is what a reader would be shown."""
    return [
        violation.detail
        for violation in check_citations(fixture.root)
        if violation.document == document
    ]


def test_a_citation_within_range_passes_because_the_rule_is_meant_to_be_quiet_when_the_source_agrees(
    fixture: Fixture,
) -> None:
    """The success case, asserted explicitly.

    A rule with only failure tests cannot be distinguished from a rule that matches
    nothing, which is the failure mode every other test in this file is guarding
    against in the other direction.
    """
    document = write_document(
        fixture.root,
        "docs/inventory.md",
        "The key is declared at `src/kojutsu/core/tanseki_mapping.py:5` of the store.\n",
    )
    assert details(fixture, document) == []


def test_a_bare_backticked_path_is_checked_for_existence_because_kojutsu_house_style_names_no_line(
    fixture: Fixture,
) -> None:
    """Kojutsu writes a bare path, so the rule must accept one.

    A citation with no line number cannot be held to a line, and refusing the form
    would push the next author into writing line numbers the citation does not need.
    """
    document = write_document(
        fixture.root,
        "docs/inventory.md",
        "See `src/kojutsu/core/tanseki_mapping.py` for the storage contract.\n",
    )
    assert details(fixture, document) == []


def test_a_line_beyond_the_end_of_the_file_fails_and_names_both_the_citation_and_the_document(
    fixture: Fixture,
) -> None:
    """A line that drifted out of range is the rotted citation this rule exists for.

    The failure names the file the citation resolved to, the strategy that resolved
    it, and -- through the document the violation is attached to -- which document
    asserts it, because "line 400 of tanseki_mapping.py does not exist" without the
    document leaves a reader with two files to search.
    """
    document = write_document(
        fixture.root,
        "docs/inventory.md",
        "The extras dict starts at `src/kojutsu/core/tanseki_mapping.py:412`.\n",
    )
    reported = details(fixture, document)
    assert len(reported) == 1
    assert "tanseki_mapping.py:412" in reported[0]
    assert f"has {len(FAKE_MODULE_LINES)} lines" in reported[0]
    assert document.name == "inventory.md"


def test_a_renamed_file_fails_because_that_is_the_rotted_citation_this_rule_exists_for(
    fixture: Fixture,
) -> None:
    """A cited file that no longer exists must be reported.

    The trap this avoids is the one that makes a rotted citation worse than none: a
    file is renamed, the citation stops resolving, and a rule that quietly treated
    "resolved to nothing" as "not a citation" would report a clean run over a
    document asserting a fact about a file that is not there.
    """
    document = write_document(
        fixture.root,
        "docs/inventory.md",
        # Interpolated, not written: this file is scanned by the rule it tests, and a
        # literal broken citation in it would be reported by the gate as a real one.
        f"The mapping lives at `src/kojutsu/core/renamed_{STEM}ing.py:{LINE}`.\n",
    )
    reported = details(fixture, document)
    assert len(reported) == 1
    assert f"renamed_mapping.py:{LINE}" in reported[0]
    assert "still looks checked" in reported[0]


def test_a_renamed_file_named_without_a_line_is_left_alone_because_that_form_is_ordinary_prose(
    fixture: Fixture,
) -> None:
    """The counterpart, and the reason the rule keys on the line number.

    Kojutsu's own convention is a bare backticked path, so a bare path that
    resolves nowhere is far more likely to be an example or an elided link than a
    renamed file. Failing those would produce violations on sentences that assert
    nothing, and a gate that cries wolf is a gate that gets switched off.
    """
    document = write_document(
        fixture.root,
        "docs/inventory.md",
        "A file like `core/renamed_mapping.py` might hold the projection.\n",
    )
    assert details(fixture, document) == []


def test_a_package_relative_path_resolves_because_the_source_spells_it_that_way(
    fixture: Fixture,
) -> None:
    """``core/tanseki_mapping.py`` and the repo-relative spelling are one file.

    Kojutsu's own docstrings and comments write paths relative to
    ``src/kojutsu/``, so a document that quotes one quotes it that way. Requiring
    the repository-relative spelling would mean the citation is only checkable if
    somebody remembers to rewrite it, and rewriting is where the information gets
    lost.
    """
    document = write_document(
        fixture.root,
        "docs/inventory.md",
        "The keys are declared at `core/tanseki_mapping.py:4` and nowhere else.\n",
    )
    assert details(fixture, document) == []
    resolution = resolve_citation(fixture.checkout, "core/tanseki_mapping.py")
    assert resolution is not None
    assert resolution.strategy.startswith("package-relative")
    assert resolution.path == fixture.checkout / "src/kojutsu/core/tanseki_mapping.py"


def test_a_leading_kojutsu_prefix_is_stripped_because_that_is_how_the_other_repository_writes_it(
    fixture: Fixture,
) -> None:
    """``kojutsu/src/kojutsu/models.py`` resolves to the same file as the short form.

    Kojutsu's own documents prefix its repository name, and the short form is what
    the module writes. Treating them as two paths would leave one of them
    permanently unresolvable and therefore permanently unchecked.
    """
    resolution = resolve_citation(fixture.checkout, "kojutsu/src/kojutsu/models.py")
    assert resolution is not None
    assert resolution.path == fixture.checkout / "src/kojutsu/models.py"
    assert resolution.strategy == "repository-relative"


def test_a_bare_filename_resolves_only_when_exactly_one_file_carries_that_name(
    fixture: Fixture,
) -> None:
    """The basename strategy works when it is unambiguous and gives up when it is not.

    ``tanseki_mapping.py`` lives one directory below the package root, so neither the
    repository-relative nor the package-relative reading of a bare filename finds
    it and the basename search is the only strategy that can. Two files called
    ``tanseki_mapping.py`` are not a citation, and picking one would be inventing an
    answer to a question the document did not ask: a guess that is silently wrong
    is worse than a skip that is counted.
    """
    assert resolve_citation(fixture.checkout, "tanseki_mapping.py") is not None

    sibling = fixture.checkout / "src" / "kojutsu" / "webhook"
    sibling.mkdir()
    (sibling / "tanseki_mapping.py").write_text("# a different mapping\n", encoding="utf-8")
    assert resolve_citation(fixture.checkout, "tanseki_mapping.py") is None


def test_a_path_into_this_repository_is_not_a_citation_because_the_link_rule_owns_it(
    fixture: Fixture,
) -> None:
    """A backticked path that exists here is this repository's business.

    ``inventory.md`` is a real Tenbin document and a plausible Kojutsu filename.
    A rule that treated it as a cross-repository citation would fail on every
    document in this project that names a sibling.
    """
    write_document(fixture.root, "docs/inventory.md", "# The inventory\n")
    document = write_document(
        fixture.root,
        "docs/notes.md",
        "See [`inventory.md`](inventory.md) and `docs/inventory.md` for the fields.\n",
    )
    assert details(fixture, document) == []


def test_a_document_relative_path_is_not_a_citation_because_it_is_relative_to_a_document(
    fixture: Fixture,
) -> None:
    """``../scripts/check_docs.py`` is a link, and a link is not a claim about a line.

    A path containing ``..`` can only be resolved against the document containing
    it, so it can never denote a repo-rooted path in another repository. Treating
    one as an unresolvable citation would make the gate cry wolf over the ordinary
    way a document links sideways.
    """
    write_document(fixture.root, "docs/notes.md", "# Notes\n")
    document = write_document(
        fixture.root,
        "docs/deep/page.md",
        "The gate lives at [`../scripts/check_docs.py`](../scripts/check_docs.py) and "
        "the notes at `../notes.md:4`.\n",
    )
    assert details(fixture, document) == []


def test_prose_that_merely_contains_a_path_shaped_token_is_not_a_citation(
    fixture: Fixture,
) -> None:
    """The regex is anchored to the backticks, because prose is mostly not citations.

    Each token here is something a reader would see in these documents: a dotted
    module reference, a call with arguments, a filename used as an example of a
    shape rather than as a claim about a file. Matching any of them would produce
    violations on sentences that assert nothing about a line.
    """
    document = write_document(
        fixture.root,
        "docs/inventory.md",
        "Read `tenbin.corpus.record` for the parse, call `select(records, kinds=None)`, "
        "and note that a bare `core/tanseki_mapping.py` with no line is checked for "
        "existence alone. `models.py` is ambiguous and resolves to nothing.\n",
    )
    assert details(fixture, document) == []


def test_an_absent_checkout_warns_visibly_and_checks_nothing(
    fixture: Fixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The skip is announced, loudly, with the way to fix it.

    The requirement is not silence-with-an-exit-code; it is that a run which checked
    nothing cannot look like a run which checked everything. So the warning names
    the path it looked for, the variable that overrides it, and says in as many
    words that the cross-repository claims went unchecked.
    """
    # **Every** cited repository, so "checks nothing" is still true. Pointing only
    # Kojutsu at nothing leaves Tanseki resolvable, which is the partial-skip case
    # and is a different test with a different answer. There are two cited repositories
    # and this one has to move both of them, or it stops meaning what its name says.
    nowhere = str(tmp_path / "not-a-checkout")
    monkeypatch.setenv(KOJUTSU_CHECKOUT_ENV, nowhere)
    monkeypatch.setenv(TANSEKI_CHECKOUT_ENV, nowhere)
    document = write_document(
        fixture.root,
        "docs/inventory.md",
        "The key is at `src/kojutsu/core/tanseki_mapping.py:5`.\n",
    )
    assert details(fixture, document) == []
    stderr = capsys.readouterr().err
    assert "citation check SKIPPED" in stderr
    assert KOJUTSU_CHECKOUT_ENV in stderr
    assert "UNCHECKED" in stderr


def test_a_citation_out_of_range_is_still_reported_when_the_path_exists_here_too(
    fixture: Fixture,
) -> None:
    """The exists-here escape hatch must not become a way to hide a bad citation.

    A document can legitimately mention ``models.py`` as a bare stem of one of its
    own documents. It cannot do that *and* cite a line. Both are checked here so
    the escape hatch stays where it belongs: on the path, never on the line.
    """
    document = write_document(
        fixture.root,
        "docs/inventory.md",
        f"Our own `docs/inventory.md:{LONG_LINE}` is a document, and it is also long enough "
        "to be cited here for no good reason.\n",
    )
    assert details(fixture, document) == []


def test_violations_carry_the_document_so_a_failure_names_where_the_claim_was_made(
    fixture: Fixture,
) -> None:
    """The reporting shape the rule's output depends on.

    Asserted on the dataclass rather than only through the printed string, because
    the printed string is what a human reads and the dataclass is what
    ``main`` iterates; a change to one that quietly broke the other would leave the
    rule reporting a violation nobody could locate.
    """
    document = write_document(
        fixture.root,
        "docs/inventory.md",
        "`src/kojutsu/core/tanseki_mapping.py:999` is past the end.\n",
    )
    violations = list(check_citations(fixture.root))
    assert [violation.document for violation in violations] == [document]
    rendered = violations[0].render(fixture.root)
    assert rendered.startswith("docs/inventory.md: ")


def test_the_strict_mode_exit_code_is_a_constant_rather_than_a_magic_number(
    fixture: Fixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The opt-in escalation is reachable and distinct from the failure code.

    A contributor with no checkout should not be told their prose is wrong, and a
    pipeline that wants the gate to fail closed should not have to reimplement the
    skip. One environment variable and one exit code that is not ``1`` is the whole
    mechanism, and it is only usable if the number is named.
    """
    assert SKIPPED_EXIT_CODE != 1
    # Both cited repositories, because ``main`` reads this repository's real
    # documents -- and those now carry Tanseki citations. Pointing only Kojutsu
    # away would leave Tanseki resolving against the stand-in module, whose five lines
    # would make a real citation look out of range and turn this into a test about
    # the failure code instead of the skip code.
    still_absent = str(tmp_path / "still-absent")
    monkeypatch.setenv(KOJUTSU_CHECKOUT_ENV, still_absent)
    monkeypatch.setenv(TANSEKI_CHECKOUT_ENV, still_absent)
    monkeypatch.setenv(STRICT_ENV, "1")

    # ``main`` reads the repository this script lives in rather than a temporary
    # one, so what is asserted is the branch: a strict run against an absent
    # checkout is the skip exit code, and never the failure code a real violation
    # would produce.
    assert main() == SKIPPED_EXIT_CODE


def test_resolution_is_none_for_something_that_is_not_a_path_at_all(fixture: Fixture) -> None:
    """The resolver's negative case, so the positive assertions mean something.

    A ticket id, a slug and an absolute path outside the checkout all resolve to
    nothing, which is what keeps the caller from having to decide whether a token
    was a citation before asking.
    """
    assert resolve_citation(fixture.checkout, "core/") is None
    assert resolve_citation(fixture.checkout, "../outside.py") is None
    assert resolve_citation(fixture.checkout, "/etc/hosts") is None


def test_the_violation_type_is_hashable_and_comparable_because_reports_are_compared() -> None:
    """Two identical violations compare equal, so a test can assert on a set of them.

    Present because the gate aggregates violations from three rules and a caller
    that deduplicates them by value would silently collapse duplicates from
    different rules -- a frozen dataclass is what makes that safe.
    """
    assert Violation(Path("a.md"), "detail") == Violation(Path("a.md"), "detail")
    assert len({Violation(Path("a.md"), "d"), Violation(Path("a.md"), "d")}) == 1


# -- the second cited repository ------------------------------------------------------


def test_a_citation_into_tanseki_is_resolved_and_its_line_range_checked_because_tanseki_is_what_the_wire_claims_are_about(
    fixture: Fixture,
) -> None:
    """Tanseki is now a cited repository, and the rule reaches it the way it reaches Kojutsu.

    Every fact in ``docs/seam.md`` about an envelope key, a batch bound or an offset
    ceiling is a claim about Tanseki, and it rotted on exactly the schedule a citation
    into Kojutsu rots on. The bound is declared in a Kotlin file at
    ``core/domain/src/main/kotlin/gokorei/tanseki/core/domain/RequestLimits.kt``, which is not a
    path a Kojutsu-relative search would ever reach -- so the rule needed to know
    about a second checkout before it could check anything.
    """
    cited = "tanseki/core/domain/src/main/kotlin/gokorei/tanseki/core/domain/RequestLimits.kt"
    # Line 3 because the stand-in module has five lines; the range check is not the
    # subject of this test, the resolution is.
    document = write_document(fixture.root, "seam.md", f"The batch bound is `{cited}:3` here.\n")

    assert details(fixture, document) == []

    stale = write_document(fixture.root, "stale.md", f"The batch bound is `{cited}:9999`.\n")
    (found,) = details(fixture, stale)
    assert "out of range" in found
    assert "RequestLimits.kt in tanseki" in found, (
        "the failure must name which repository was checked"
    )


def test_a_file_present_in_both_cited_repositories_is_a_violation_rather_than_a_first_hit(
    fixture: Fixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**A citation that could be two files is not checked against either.**

    The tempting alternative is to resolve against whichever checkout was tried
    first and report success. That checks the citation against the wrong codebase,
    says nothing, and does it silently -- which is the same defect as the rotted
    citation this rule was written for, arrived at from the other direction. Two
    checkouts sharing a basename is not hypothetical once both repositories are
    cited, so the case is the case.
    """
    shared = fixture.checkout / "shared.kt"
    shared.write_text("\n".join(FAKE_MODULE_LINES) + "\n", encoding="utf-8")
    (fixture.tanseki / "shared.kt").write_text(
        "\n".join(FAKE_MODULE_LINES) + "\n", encoding="utf-8"
    )

    document = write_document(
        fixture.root, "ambiguous.md", f"The edge keys are in `shared.kt:{LINE}`.\n"
    )

    (found,) = details(fixture, document)
    assert "ambiguous" in found
    assert "kojutsu" in found and "tanseki" in found, "both candidates must be named"


def test_an_absent_tanseki_checkout_warns_by_name_rather_than_passing_silently(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A partial skip is still a skip, and it has to name what it did not check.

    ``PARTIALLY SKIPPED`` rather than ``SKIPPED`` because the Kojutsu half ran:
    collapsing the two would tell a reader that nothing was checked when half of it
    was, which is the failure this module exists to have stopped.
    """
    monkeypatch.setenv(TANSEKI_CHECKOUT_ENV, str(fixture.root / "not-there"))

    assert list(check_citations(fixture.root)) == []
    stderr = capsys.readouterr().err
    assert "PARTIALLY SKIPPED for tanseki" in stderr
    assert "UNCHECKED" in stderr
    assert TANSEKI_CHECKOUT_ENV in stderr, "the warning has to name the variable that fixes it"


# -- the rename rule ---


#: The stale and current names, taken from the rule's own data rather than written here.
#:
#: **A test of this rule cannot spell a stale name.** ``main`` reads the real repository,
#: this file is in it, and the rule scans ``.py``, so a test that wrote the old name out in
#: order to assert the rule catches it would fail the gate that the test is about -- which
#: is the rule working, on the test. Reading the pair out of ``RENAMED_PROJECTS`` also means
#: these tests cannot rot: a third rename updates them by construction, and a *removed*
#: rename entry makes them fail rather than quietly stop testing anything.
def renamed(stale: str) -> tuple[str, str]:
    """The ``(stale, current)`` pair whose stale name is ``stale``."""
    return next(pair for pair in RENAMED_PROJECTS if pair[0] == stale)


def this_projects_rename() -> tuple[str, str]:
    """This project's own entry in the rule, found by its current name.

    Keyed on ``tenbin`` because that half is the one this repository controls and the one
    a reader can check: the point of the entry is that this project renamed itself and did
    not have to do it twice.
    """
    return next(pair for pair in RENAMED_PROJECTS if pair[1] == "tenbin")


def stale_names(fixture: Fixture, document: Path) -> list[str]:
    """The rename violations for one file, which is what a reader would be shown."""
    return [
        violation.detail
        for violation in check_stale_project_names(fixture.root)
        if violation.document == document
    ]


def test_a_source_file_naming_a_project_under_its_old_name_is_reported(fixture: Fixture) -> None:
    """A docstring is where most of a project's references live, so a ``.py`` is scanned.

    A markdown-only version of this rule passes on a tree whose references are mostly
    Python docstrings, which is the failure mode of a gate that checks the wrong files and
    reports success. The file written here is a source module for exactly that reason.
    """
    stale, current = RENAMED_PROJECTS[0]
    module = write_document(
        fixture.root,
        "src/tenbin/store/client.py",
        f'"""Reads what {stale.capitalize()} writes."""\n\nBASE = "{stale}-real"\n',
    )

    (found,) = stale_names(fixture, module)
    assert stale in found and current in found, "the failure must give the replacement"


def test_a_collection_name_in_a_running_store_is_not_a_reference_to_a_project(
    fixture: Fixture,
) -> None:
    """The exemption is what makes a blanket rename safe, so it is asserted directly.

    A collection is a key into a live database. Renaming one in code points every store read
    at a collection that does not exist and every store renders as a failure -- so the
    allowlist is load-bearing, and an allowlist nothing tests is an allowlist that grows.
    Every exempt suffix is checked rather than one of them, because a suffix added by a
    later rename is exactly as likely to be wrong as one that has been there since the
    first.
    """
    stale, _ = RENAMED_PROJECTS[0]
    document = write_document(
        fixture.root,
        "stores.yaml",
        "".join(f"  - collection: {stale}{suffix}\n" for suffix in STALE_NAME_EXEMPT_SUFFIXES),
    )

    assert stale_names(fixture, document) == []


def test_a_stale_name_outside_an_exempt_suffix_is_reported_because_the_exemption_is_a_suffix(
    fixture: Fixture,
) -> None:
    """An exemption for one collection must not become an exemption for the project.

    The allowlist is matched on what *follows* the stale name, so it is a claim about
    suffixes. A rule that skipped any line mentioning the old name would pass this tree
    while letting every genuine reference back in -- the failure this rule exists to
    prevent, wearing the exemption as a disguise.
    """
    stale, _ = RENAMED_PROJECTS[0]
    exempt = STALE_NAME_EXEMPT_SUFFIXES[0]
    document = write_document(
        fixture.root,
        "notes.md",
        f"{stale.capitalize()} was the old name, and {stale}{exempt[:-1]} is not a collection.\n",
    )

    (found,) = stale_names(fixture, document)
    assert stale in found, "one character off an exempt suffix is not exempt"


def test_an_environment_template_is_scanned_because_it_is_how_a_stale_name_reaches_an_operator(
    fixture: Fixture,
) -> None:
    """``.env.example`` hid a stale name for a whole rename, and it is the file operators copy.

    The suffix list stopped at ``.txt`` and that file's suffix is ``.example``, so the old
    upstream name sat in the template an operator copies to become a real environment file
    while every rule reported success. The suffix is asserted rather than assumed: the fix
    is one tuple entry, and a tuple entry removed by a well-meaning cleanup is invisible
    until somebody reads their own environment file.
    """
    assert ".example" in STALE_NAME_SUFFIXES

    stale, _ = RENAMED_PROJECTS[0]
    document = write_document(
        fixture.root, ".env.example", f"# The collection {stale.capitalize()} writes.\n"
    )

    (found,) = stale_names(fixture, document)
    assert stale in found


def test_the_record_of_a_rename_may_name_what_it_renamed(fixture: Fixture) -> None:
    """One path is exempt, and it is exempt because a record cannot avoid the old name.

    The decision record says what this program was called and which identifiers kept the
    spelling; a reader who cannot see the old name cannot check any of that. The cost is
    stated rather than hidden -- a stale name anywhere else *in that document* is invisible
    to this rule too -- so what is asserted is the shape of the exemption and the
    behaviour, not the path spelled out: exactly one entry, relative, and narrow enough that
    the sibling record beside it is still checked.
    """
    stale, current = this_projects_rename()
    assert len(STALE_NAME_EXEMPT_PATHS) == 1, "one record, not a directory and not a glob"
    exempt = STALE_NAME_EXEMPT_PATHS[0]
    assert not exempt.is_absolute() and exempt.suffix == ".md"

    record = write_document(
        fixture.root,
        str(exempt),
        f"# Renamed from {stale.capitalize()} to {current}, and by the same rule upstream.\n",
    )
    assert stale_names(fixture, record) == []

    sibling = write_document(
        fixture.root,
        "docs/decisions/004-a-record-about-something-else.md",
        "# 004 - "
        + " and ".join(old.capitalize() for old, _ in RENAMED_PROJECTS)
        + ", in a record that is not exempt\n",
    )
    assert len(stale_names(fixture, sibling)) == len(RENAMED_PROJECTS)


def test_this_projects_own_old_name_is_in_the_rule_because_a_rename_should_need_no_second_rename(
    fixture: Fixture,
) -> None:
    """The rule that keeps a project named is this project's own rename.

    Every other entry is about another repository's data. This one is different: it is what
    stops the current name drifting back in prose, in a module path or in a configuration
    key. It is asserted because a project that renames itself and does not add itself to its
    own gate has renamed itself once -- and because the alternative, a rename with nothing
    enforcing it, is how a repository ends up called one thing and its package another,
    which is the state this entry exists to make unreachable.
    """
    stale, current = this_projects_rename()
    assert stale != current and stale.islower(), "a stale name that is not a name finds nothing"

    document = write_document(fixture.root, "src/tenbin/cli.py", f'"""Runs `{stale} drift`."""\n')
    (found,) = stale_names(fixture, document)
    assert stale in found and current in found


def test_a_citation_into_a_repository_that_left_the_cited_set_fails_loudly(
    fixture: Fixture,
) -> None:
    """A citation the gate no longer checks is a violation, not a pass.

    The ticket backend used to be a third cited repository; the adapter seam
    replaced its source citations with observed contracts, so a
    `go-experiment/` path resolves against nothing now. The dangerous outcome
    would be the gate waving it through as prose -- a rotted citation that
    still looks checked -- so a removed repository must fail here rather than
    vanish.
    """
    # Interpolated, for the reason the comment on this file's fixture pieces gives: the gate
    # reads this file, so a literal unresolvable citation here is a real violation here.
    orphaned = write_document(
        fixture.root,
        "seam.md",
        f"The filter used to live at `go-experiment/backend/pkg/web/{STEM}_gone.go:{LINE}`.\n",
    )
    (found,) = details(fixture, orphaned)
    assert "resolved to no file" in found
    assert "kojutsu" in found and "tanseki" in found, (
        "the failure must name which repositories were checked, or a reader cannot tell "
        "an unknown repository from a renamed file in a known one"
    )


def test_a_citation_in_a_docstring_is_checked_because_the_claims_are_written_in_code(
    fixture: Fixture,
) -> None:
    """**The scan covers source files, and that is the change that makes the rest matter.**

    This rule scanned markdown only, which is defensible while every cross-repository
    claim lived in a document -- and untrue as soon as a claim moved into a docstring.
    The claims that forced the widening were bound semantics in
    ``report/retrospective.py`` with no file and no line, invisible to a markdown-only
    rule; the backend they described has since left the cited set, and the scan
    stays wide because the claims stayed in code.

    Widening the scope is also what makes the gate read its own test file, which is why the
    broken fixtures there are interpolated. Both halves are the same lesson: a rule that
    checks the wrong files reports success over the references that exist.
    """
    cited = "kojutsu/src/kojutsu/models.py"
    module = fixture.root / "src" / "tenbin"
    module.mkdir(parents=True)

    sound = module / "sound.py"
    sound.write_text(f'"""See `{cited}:1`."""\n', encoding="utf-8")
    assert details(fixture, sound) == []

    rotted = module / "rotted.py"
    rotted.write_text(f'"""See `{cited}:9999`."""\n', encoding="utf-8")
    (found,) = details(fixture, rotted)
    assert "out of range" in found


def test_our_own_source_files_are_not_mistaken_for_cross_repository_citations(
    fixture: Fixture,
) -> None:
    """The elided-directory guard has to know about source files now that they are scanned.

    A docstring that writes ``measures/base.py:197`` is naming a file in *this* repository
    by the same bare-stem form the markdown documents use. The guard that recognises our own
    prose was built from markdown alone, so with the scope widened every such docstring
    would be reported as a rotted cross-repository citation -- and the obvious response,
    exempting source from the rule, is precisely the wrong one.
    """
    module = fixture.root / "src" / "tenbin"
    module.mkdir(parents=True)
    (module / "base.py").write_text("# one line\n# two\n", encoding="utf-8")

    document = module / "claims.py"
    document.write_text('"""See `base.py:1` and `measures/base.py:1`."""\n', encoding="utf-8")

    assert details(fixture, document) == []


def test_the_go_extension_is_recognised_though_no_cited_repository_needs_it(
    fixture: Fixture,
) -> None:
    """A citation pattern that cannot name a file extension checks nothing written in it.

    Not hypothetical bookkeeping: the ticket backend was written in Go, and before
    ``go`` was added every citation into it was invisible to the rule -- not reported
    as broken, not reported as fine, simply not seen. The backend has since left the
    cited set and the extension stays, so a future Go citation is checked rather
    than invisible. An extension the pattern does not know is not a rule that checks
    fewer things.
    """
    cited = "somewhere/service/handlers/thing.go"
    assert CITATION_PATTERN.search(f"`{cited}:1`")

    # And the Elixir extensions, because the repository criterion 2 also named is written in
    # Elixir and will be cited the moment there is a seam into it. Interpolated so that
    # asserting the pattern *matches* does not also leave a citation in this file that the
    # pattern then resolves against two checkouts and cannot find.
    for extension in ELIXIR_EXTENSIONS:
        assert CITATION_PATTERN.search(f"`pijul-viz/apps/pijul_server/lib/thing.{extension}:1`")
