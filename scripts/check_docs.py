"""The documentation gate: this project's prose makes claims that must resolve.

Tenbin is unusual in that its most important documents are claims *about another
codebase's source*. ``docs/corpus-inventory.md`` states what Kojutsu persists,
what it discards, and where each fact dies, and it gives a file and a line for
every claim so a reader can re-check rather than trust. That is the correct way
to write such a document and it is also fragile: Kojutsu is a repository this
project does not own and does not control, so those citations rot silently, and a
rotted citation is worse than no citation because it still looks checked.

The rules are data rather than functions. A third rule is a list entry, not
another function to read, which keeps the cost of adding a check at the level of
one line and makes the whole policy readable in one screen.

Three rules today. Relative markdown links must resolve to a file that exists,
because Tenbin's documents are a web of cross-references and a broken one is a
reader who concludes a section does not exist when it does. A stale-term list,
because a term that has been replaced everywhere except one document is
indistinguishable from a term that is still supported. And a citation check,
because this project demonstrated its own defect rather than predicting it: the
inventory was verified against the wrong Kojutsu branch, went stale in six
places, and nothing noticed until a human went and read the source.
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

LINK_PATTERN = re.compile(r"\[[^\]]+\]\(([^)]+)\)")

#: A backticked cross-repository citation: a whole token that is a repo-rooted path,
#: optionally followed by a line or an inclusive line range. Anchored to the
#: backticks on both sides so that prose containing a path-shaped fragment --
#: ``select(records)``, ``foo.py`` mid-sentence -- cannot produce a match, and so
#: that Kojutsu's own house convention of a bare backticked path with no line
#: number is still caught. A token containing ``..`` is excluded outright: it is
#: relative to a document rather than to the repository root, so it cannot be a
#: citation of anything in the cited checkout.
CITATION_PATTERN = re.compile(
    r"`(?P<path>(?!\.\./)(?!\./)[A-Za-z0-9_.\-/]+"
    # ``go``, ``ex`` and ``exs`` stay recognised after the ticket backend left the
    # cited set. A citation pattern that cannot name a file extension is not a
    # pattern that checks fewer things -- it is a pattern that silently checks nothing
    # about the repository written in it, which is the failure this whole rule exists
    # to prevent. The extensions outlive the repository that needed them.
    r"(?:\.(?:py|md|html|kt|go|ex|exs|json|toml|yaml|yml|txt|sql|sh|ts|js)))"
    r"(?::(?P<lines>\d+(?:-\d+)?))?`"
)

#: Directories that hold no documentation and are not this repository's to lint.
IGNORED_DIRS = frozenset(
    {".git", ".venv", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".pijul", "dist"}
)

#: Terms that have been replaced or abandoned. A surviving occurrence is a
#: document describing a world this project has left.
STALE_TERMS = ("mongodb", "mongodb_uri", "neo4j", "mongodb://")

#: Link schemes that are not filesystem paths and so cannot be resolved here.
EXTERNAL_SCHEMES = ("http://", "https://", "mailto:")


#: One repository this project makes claims about and does not own.
#:
#: The citation rule started with the conversation store because it was the only other
#: codebase whose behaviour Tenbin described -- it was called Chronicler then, and the
#: rule is one of the reasons that rename was a sweep rather than a folder click. Tanseki
#: is now the second, and it is here for a sharper reason than the first: **Tanseki is
#: the thing the wire claims are about.** Every fact in ``docs/seam.md`` about a listing
#: envelope, a batch bound or an offset ceiling is a claim about Tanseki, and a citation
#: into Tanseki's source rots on exactly the schedule a citation into Kojutsu's does.
#:
#: ``prefix`` is stripped before resolution, because a document that writes
#: ``tanseki/core/domain/RequestLimits.kt`` means the same file as the module-relative
#: ``core/domain/RequestLimits.kt`` and should not fail for spelling it the long
#: way. ``package_roots`` is the equivalent of Kojutsu's ``src/kojutsu``: the
#: directory a module-relative citation resolves against, and empty for Tanseki, whose
#: Gradle layout puts each module's package under its own source root.
@dataclass(frozen=True)
class CitedRepository:
    """A checkout this project cites into: its name, where it lives, how it is spelt."""

    name: str
    env: str
    default: str
    prefix: str
    package_roots: tuple[str, ...] = ()

    def checkout(self) -> Path:
        """This repository's checkout, from the environment or the default."""
        configured = os.environ.get(self.env) or self.default
        return Path(configured).expanduser()


#: Every cited repository, in the order a citation is tried against them. The order
#: only decides which resolution wins when exactly one repository has the file; a
#: file present in two is reported as ambiguous rather than resolved, so the order
#: can never quietly decide what a citation was checked against.
CITED_REPOSITORIES: tuple[CitedRepository, ...] = (
    CitedRepository(
        name="kojutsu",
        env="KOJUTSU_CHECKOUT",
        # **The checkout directory moved, and this default was wrong until now.** The
        # project was renamed, its package is `src/kojutsu`, and the clone is at
        # `~/Documents/kojutsu` -- but this line still said `~/Documents/chronicler`,
        # so every run since the rename reported every Kojutsu citation as UNCHECKED
        # and exited zero. It is the exact failure the paragraph above this entry
        # describes, and the reason it is written down here again: an absent checkout
        # is a skip rather than an error, so a stale path does not announce itself.
        # The check that would have caught it is the gate itself, pointed at a
        # directory that exists -- `ls -d ~/Documents/kojutsu`, and the stderr line
        # that names the path it wanted.
        default="~/Documents/kojutsu",
        prefix="kojutsu/",
        package_roots=("src/kojutsu",),
    ),
    CitedRepository(
        name="tanseki",
        env="TANSEKI_CHECKOUT",
        default="~/Documents/tanseki",
        prefix="tanseki/",
    ),
)

#: Kept under the name the rest of the module and its tests already use.
KOJUTSU_CHECKOUT_ENV = "KOJUTSU_CHECKOUT"
TANSEKI_CHECKOUT_ENV = "TANSEKI_CHECKOUT"
DEFAULT_TANSEKI_CHECKOUT = "~/Documents/tanseki"

#: The exit code a strict run uses when the checkout is absent. Zero for a warning,
#: two for "I could not check", because the ordinary gate has exactly one failure
#: meaning and a second one would be a contract every caller has to learn.
SKIPPED_EXIT_CODE = 2

#: Opt in to treating an absent checkout as a failure rather than a warning. See
#: :func:`main` for why the default is the softer of the two.
STRICT_ENV = "TENBIN_DOCS_STRICT_CITATIONS"

#: Directories never walked when looking for a unique basename match. They hold no
#: cited source and a match inside one would be a coincidence rather than a
#: citation -- a virtualenv in particular holds an installed *copy* of every module,
#: so walking one makes every bare-filename citation ambiguous and silently skipped,
#: which is the exact outcome this rule exists to prevent.
CITED_SKIP_DIRS = frozenset(
    {
        ".git",
        ".venv",
        ".package-venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        "node_modules",
        "site-packages",
    }
)


@dataclass(frozen=True)
class Violation:
    """One thing wrong with one document, stated so it can be acted on."""

    document: Path
    detail: str

    def render(self, root: Path) -> str:
        return f"{self.document.relative_to(root)}: {self.detail}"


@dataclass(frozen=True)
class Resolution:
    """How one citation was located in the cited repository.

    ``strategy`` is kept rather than discarded because a failure has to be able to
    say which reading of the path was used, and a reader who has written
    ``core/tanseki_mapping.py`` needs to know whether the checker looked for it at the
    repository root before it decided the file was missing.
    """

    path: Path
    strategy: str


def iter_markdown(root: Path) -> Iterator[Path]:
    """Yield every document worth checking, skipping build and tool directories."""
    for path in sorted(root.rglob("*.md")):
        if not any(part in IGNORED_DIRS for part in path.relative_to(root).parts):
            yield path


def _link_target(target: str) -> str | None:
    """Return the filesystem part of a link target, or None if it is not one."""
    candidate = target.split("#", 1)[0].split("?", 1)[0].strip()
    if not candidate or candidate.startswith(EXTERNAL_SCHEMES):
        return None
    return candidate


def check_links(root: Path) -> Iterator[Violation]:
    """Every relative link must resolve to a file that exists.

    Resolution is relative to the containing document, because that is how a
    reader follows it. A link that escapes the repository is a violation even if
    the target exists, since nothing outside this repository can be part of its
    argument.
    """
    for document in iter_markdown(root):
        text = document.read_text(encoding="utf-8")
        for match in LINK_PATTERN.finditer(text):
            target = _link_target(match.group(1))
            if target is None:
                continue
            resolved = (document.parent / target).resolve()
            if not resolved.is_relative_to(root.resolve()):
                yield Violation(document, f"link target escapes the repository: {target}")
            elif not resolved.exists():
                yield Violation(document, f"missing link target: {target}")


def check_stale_terms(root: Path) -> Iterator[Violation]:
    """No document may mention a term the project has moved off."""
    for document in iter_markdown(root):
        text = document.read_text(encoding="utf-8").casefold()
        for term in STALE_TERMS:
            if term in text:
                yield Violation(document, f"stale term {term}")


def present_repositories() -> tuple[CitedRepository, ...]:
    """The cited repositories that are actually checked out, for citation resolution."""
    return tuple(repository for repository in CITED_REPOSITORIES if repository.checkout().is_dir())


def missing_repositories() -> tuple[CitedRepository, ...]:
    """The cited repositories that are not, so each skip can be announced by name."""
    return tuple(
        repository for repository in CITED_REPOSITORIES if not repository.checkout().is_dir()
    )


def _within(root: Path, candidate: str | Path) -> Path | None:
    """Resolve ``candidate`` under ``root``, or ``None`` if it escapes the checkout.

    The escape check is what stops ``../../etc/passwd`` from being resolved and
    reported as a citation, and it is applied after ``resolve`` so a symlink out of
    the checkout is caught as well as a ``..`` in the text.
    """
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root.resolve()):
        return None
    return resolved


def _unique_basename(checkout: Path, name: str) -> Path | None:
    """The one file in the checkout called ``name``, or ``None`` if it is not unique.

    The last resolution strategy, and the one with a stated cost: a citation
    written as a bare filename resolves by basename, so a file that keeps its name
    while its directory is renamed goes uncaught. What it does buy is that a
    *renamed file* and a *line that drifted out of range* are both caught for the
    bare-filename citations that would otherwise resolve nowhere and be skipped in
    silence -- and a silently skipped citation is the failure this whole rule
    exists to prevent. Ambiguity is a skip rather than a guess: two files called
    ``server.py`` are not a citation, and picking one would be inventing an answer.
    """
    matches = [
        path
        for path in checkout.rglob(name)
        if path.is_file()
        and not any(part in CITED_SKIP_DIRS for part in path.relative_to(checkout).parts)
    ]
    return matches[0] if len(matches) == 1 else None


def resolve_citation(checkout: Path, cited: str) -> Resolution | None:
    """Locate ``cited`` in ``checkout``, or ``None`` if no cited repository has it.

    Deprecated in favour of :func:`resolve_anywhere`, which tries every checkout and
    reports a file present in two. Kept as a single-repository call because the tests
    that pin the three resolution strategies want exactly that and nothing else.
    """
    repository = CITED_REPOSITORIES[0]
    return resolve_in(repository, repository.checkout(), cited)


def resolve_in(repository: CitedRepository, checkout: Path, cited: str) -> Resolution | None:
    """Locate ``cited`` in one repository, or ``None`` if it is not a file there.

    The strategies are tried in order and the first hit wins, so the resolution a
    successful check used is reported rather than being an accident of directory
    listing order:

    1. **Repository-relative**, after stripping the repository's own ``prefix`` -- a
       document that writes ``kojutsu/src/kojutsu/models.py`` where the module itself
       writes ``src/kojutsu/models.py`` means one file, and the prefix is per-repository
       rather than spelled out here so that adding a cited repository cannot leave this
       paragraph describing the wrong one.
    2. **Package-relative**, against each of the repository's ``package_roots``, so
       ``core/tanseki_mapping.py`` resolves the way the module's own docstrings spell it.
    3. **Unique basename**, which catches the bare-filename citations the other two
       cannot reach.

    Returning ``None`` is not a failure. A bare backticked path that resolves
    nowhere in the cited repository is ordinary prose -- a link into *this*
    repository, a filename used as an example -- and a rule that failed on those
    would be a rule that got switched off.
    """
    trimmed = cited
    if repository.prefix and trimmed.startswith(repository.prefix):
        trimmed = trimmed[len(repository.prefix) :]

    direct = _within(checkout, trimmed)
    if direct is not None and direct.is_file():
        return Resolution(direct, "repository-relative")

    for package_root in repository.package_roots:
        packaged = _within(checkout, f"{package_root}/{trimmed}")
        if packaged is not None and packaged.is_file():
            return Resolution(packaged, f"package-relative ({package_root}/)")

    basename = trimmed.rsplit("/", 1)[-1]
    if "/" not in trimmed and (found := _unique_basename(checkout, basename)) is not None:
        return Resolution(found, "unique basename")
    return None


def resolve_anywhere(
    repositories: Sequence[CitedRepository], cited: str
) -> tuple[tuple[str, Resolution], ...]:
    """Every cited repository ``cited`` resolves in, which is usually exactly one.

    **A file present in two cited repositories is a violation, not a first hit.**
    The alternative is to resolve it against whichever repository happened to be
    tried first and report success -- which checks the citation against the wrong
    codebase and says nothing, and does it silently. ``RequestLimits.kt`` and
    ``models.py`` are exactly the kind of basename two checkouts share, so the case
    is not hypothetical: it is the case this function exists for.

    Returning every hit rather than the first one, and an empty tuple rather than
    ``None`` for nothing, is what lets the caller tell the three cases apart without
    a second lookup -- and a two-tuple-per-hit return would have made ``len()`` lie
    about a single hit, which is the sort of bug this module exists to catch.
    """
    return tuple(
        (repository.name, resolution)
        for repository in repositories
        if (resolution := resolve_in(repository, repository.checkout(), cited)) is not None
    )


def render_ambiguity(found: Sequence[tuple[str, Resolution]]) -> str:
    """The names and files a citation matched in more than one cited repository."""
    return ", ".join(f"{name}: {resolution.path.name}" for name, resolution in found)


def _cited_line_range(spec: str | None) -> tuple[int, int]:
    """Parse ``123`` or ``20-207`` into a one-based inclusive range.

    A citation with no line number is checked for existence alone, which is the one
    thing a bare backticked path can be checked for: Kojutsu's house convention is
    to name a file and not a line, and a document that obeys it is not making a
    claim this rule can hold to anything finer.
    """
    if spec is None:
        return (1, 1)
    first, _, last = spec.partition("-")
    start = int(first)
    return (start, int(last) if last else start)


def _line_count(path: Path) -> int:
    """How many lines the cited file has, read as text because it may not be UTF-8."""
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def _own_markdown_names(root: Path) -> frozenset[str]:
    """Every filename this repository owns, for recognising our own prose.

    A document that names a sibling by bare stem -- ``inventory.md:12`` for a file
    that actually lives at ``docs/inventory.md`` -- is writing about *this*
    repository, and the link rule owns it. Without this the elided-directory form
    would look exactly like a cross-repository citation whose file was renamed.

    Source files as well as documents, and necessarily so now that
    :func:`check_citations` scans both: a docstring that writes
    ``corpus/record.py:117`` is naming a file in this repository by the same elided
    form, and if only markdown names were collected then every such docstring would be
    reported as a rotted cross-repository citation. The name is a little historical for
    what it now holds.
    """
    return frozenset(path.name for path in iter_source_and_docs(root))


def check_citations(root: Path) -> Iterator[Violation]:
    """Every cross-repository citation must resolve, and every cited line must be in range.

    **A citation is a backticked path that names a line.** The line number is what
    makes it one: nobody writes ``core/tanseki_mapping.py:61`` as prose, and nobody
    writes it to mean a link into this repository. So a cited path that resolves
    nowhere in the cited checkout *fails*, because that is the rotted citation this
    rule exists to catch -- a file that has been renamed is otherwise a citation
    that keeps looking checked.

    **A bare backticked path is not held to that standard,** because Kojutsu's
    house convention is to name a file and not a line, and its documents are full of
    bare paths that are not citations at all. A bare path that resolves is checked
    for existence; a bare path that resolves nowhere is counted and skipped, since
    it is far more likely to be prose or a link into this repository than a renamed
    file. The two cases are separated on the one signal that actually distinguishes
    them, rather than on a guess about the author's intent.

    **When the checkout is absent this rule skips, and says so on stderr.** The
    decision, written down because it is the decision most likely to be reversed by
    accident: a *warning*, not a distinct exit code. The gate has one failure
    meaning -- this document is wrong -- and a second exit code is a contract every
    caller and every CI step has to learn before it can be trusted. Silence is what
    is unacceptable, so the skip is announced on stderr with a greppable prefix, the
    environment variable that would fix it, and the words UNCHECKED, because a run
    that verified nothing must not be indistinguishable from a run that verified
    everything. ``TENBIN_DOCS_STRICT_CITATIONS=1`` escalates the skip to
    :data:`SKIPPED_EXIT_CODE` for whoever wants the gate to fail closed; the default
    stays open because a contributor with no Kojutsu checkout should not be told
    their prose is wrong.
    """
    repositories = present_repositories()
    for repository in missing_repositories():
        print(
            f"check_docs: citation check PARTIALLY SKIPPED for {repository.name}: no "
            f"checkout at {repository.checkout()} ({repository.env} or the default "
            f"{repository.default} would fix it). Claims in docs/ about "
            f"{repository.name} are UNCHECKED in this run -- a rotted citation still "
            "looks checked.",
            file=sys.stderr,
        )
    if not repositories:
        print(
            "check_docs: citation check SKIPPED entirely: no cited repository is "
            "checked out, so every cross-repository claim in docs/ is UNCHECKED.",
            file=sys.stderr,
        )
        return

    ours = _own_markdown_names(root)
    skipped = 0
    #: Unresolved citations, held back from being called rotted when a checkout is
    #: absent. See the branch below: with one repository missing, a citation into
    #: *that* one cannot be distinguished from prose, and accusing it of being a
    #: renamed file would be a false accusation rather than a cautious one.
    absent = missing_repositories()
    unchecked = 0
    # **Source as well as documents, and that is the whole point of this change.**
    # `iter_source_and_docs` rather than `iter_markdown`: the claims this rule exists to
    # protect are overwhelmingly in *docstrings*, not in prose -- the inclusive-bounds
    # contract that `replace_to_period` exists to correct is a sentence in
    # `report/retrospective.py`, and a markdown-only scan never sees it. Scanning only
    # documents is the same mistake `check_stale_project_names` documents in its own
    # docstring: a rule that checks the wrong files reports success over the references
    # that actually exist. The ticket backend used to be a third cited repository,
    # and removing it left the scope where it was: the claims written in code did
    # not move with it, so neither did the scan.
    for document in iter_source_and_docs(root):
        text = document.read_text(encoding="utf-8")
        for match in CITATION_PATTERN.finditer(text):
            cited = match.group("path")
            lines = match.group("lines")
            if (root / cited).exists() or cited.rsplit("/", 1)[-1] in ours:
                # A path into this repository is this repository's business.
                continue
            hits = resolve_anywhere(repositories, cited)
            if not hits:
                if lines is None:
                    skipped += 1
                    continue
                names = ", ".join(repository.name for repository in repositories)
                if absent:
                    # It might be a citation into a checkout this run could not see.
                    # Calling it rotted here would tell a contributor who has one
                    # repository and not the other that their prose is wrong, which is
                    # the failure the skip exists to avoid -- just in a new shape.
                    unchecked += 1
                    continue
                yield Violation(
                    document,
                    f"citation `{cited}:{lines}` resolved to no file in any of: {names}. "
                    "The file has been renamed or the path is wrong, and a citation "
                    "that does not resolve still looks checked",
                )
                continue
            if len(hits) > 1:
                # Checked against neither, because checking against one and reporting
                # success is the failure this branch exists to prevent.
                yield Violation(
                    document,
                    f"citation `{cited}` is ambiguous: it resolves in "
                    f"{render_ambiguity(hits)}. Spell it repository-relative so it "
                    "names exactly one file, because a citation that could be two files "
                    "is not checked against either",
                )
                continue
            name, resolution = hits[0]
            if lines is None:
                continue
            start, end = _cited_line_range(lines)
            total = _line_count(resolution.path)
            if end > total:
                relative = resolution.path
                yield Violation(
                    document,
                    f"citation `{cited}:{lines}` is out of range: "
                    f"{relative.name} in {name} has {total} lines "
                    f"(resolved by {resolution.strategy})",
                )

    if unchecked:
        names = ", ".join(repository.name for repository in missing_repositories())
        print(
            f"check_docs: {unchecked} cited line(s) resolved to no file in the "
            f"checkouts present, and {names} was not checked out, so they were left "
            "alone rather than called rotted: one of them may be a citation into the "
            "absent repository, and a citation into a checkout nobody has cannot be "
            "called renamed.",
            file=sys.stderr,
        )

    if skipped:
        names = ", ".join(repository.name for repository in repositories)
        print(
            f"check_docs: {skipped} bare backticked path(s) resolved to no file in "
            f"any of: {names}, so they were left alone: that is right for prose and for "
            "links into this repository, and wrong for a citation whose file was "
            "renamed. Cite a line to have the rename caught.",
            file=sys.stderr,
        )


#: A project this codebase used to name, and what it is called now.
#:
#: Held as a pair rather than a bare term because a rename that only says what to
#: stop saying leaves the next reader with a prohibition and no replacement. The
#: replacement is the actionable half.
#:
#: The second entry is this repository's own former name, which makes the rule
#: self-applying: ``tenbin`` cannot drift back to ``insight`` in prose, in a module
#: path or in a configuration key without this reporting it.
RENAMED_PROJECTS = (("chronicler", "kojutsu"), ("insight", "tenbin"))

#: Occurrences of a stale name that are nonetheless correct.
#:
#: **These are collection names in a running store, not references to a project.**
#: ``chronicler-gokorei`` is a key into a live database; renaming it in the code
#: would point every store read at a collection that does not exist, and every store
#: would render as a failure. Renaming the collections themselves is a data migration
#: and is not this rule's business. So a stale name followed by one of these is
#: reported by nothing, and this allowlist is the only reason a blanket rename is safe.
#:
#: ``-seam-check`` is the same kind of name under the second stale term. The seam
#: suite creates it, writes to it and deletes from it, so nothing is lost by leaving
#: it spelled the old way -- and a developer's local store already holds it, which is
#: the situation the rest of this list exists for.
STALE_NAME_EXEMPT_SUFFIXES = (
    "-gokorei",
    "-secondary",
    "-tertiary",
    "-real",
    "-pilot",
    "-seam-check",
)

#: The suffixes this rule scans, rather than every file in the tree.
#:
#: Not a whitelist of interesting files but a statement of where a name can hide:
#: prose in documents, names in source, and configuration that names a repository.
#:
#: ``.example`` is here because ``.env.example`` is the one file that names the
#: environment prefix every operator has to set, and it was missed until the second
#: rename: the suffix list stopped at ``.txt`` and this file's suffix is ``.example``,
#: so a stale project name sat in the template that is copied to become a real
#: environment file. A scan list is a claim about where a name can hide, so a file
#: that demonstrably hid one belongs in it.
STALE_NAME_SUFFIXES = (".py", ".md", ".toml", ".yml", ".yaml", ".cfg", ".txt", ".example")

#: This file, which the rule cannot scan without matching its own search pattern.
#:
#: A rule that searches for a string must contain that string, so scanning itself
#: reports the pattern as a violation of itself on the first run. Exempting one file
#: is a real limitation and it is stated rather than hidden: this file is also the
#: place the exemption is defined, so a stale name added to it is exactly as
#: invisible as one added anywhere else in the codebase.
_SELF = Path(__file__).resolve()

#: Files that must name a project under its old name in order to be legible.
#:
#: **A record of a rename cannot avoid naming what was renamed.** ``003`` exists to say
#: what this program was called, why the name changed, and which two identifiers kept the
#: old spelling; a reader who cannot see the old name cannot check any of that. So the
#: same reasoning that exempts this file from its own pattern exempts that record from
#: the rule -- with the limitation stated rather than hidden, because it is a real one: a
#: stale name added anywhere else *in* that document is now invisible too, and the only
#: defence is that it is short and that every other rule still scans it.
#:
#: A path rather than a glob, so the exemption cannot quietly widen into a directory.
STALE_NAME_EXEMPT_PATHS = (Path("docs/decisions/003-insight-to-tenbin.md"),)


def iter_source_and_docs(root: Path) -> Iterator[Path]:
    """Yield every file a project name could hide in, skipping build and tool dirs."""
    for path in sorted(root.rglob("*")):
        if path.suffix not in STALE_NAME_SUFFIXES or not path.is_file():
            continue
        if any(part in IGNORED_DIRS for part in path.relative_to(root).parts):
            continue
        if path.resolve() == _SELF:
            continue
        if path.relative_to(root) in STALE_NAME_EXEMPT_PATHS:
            continue
        yield path


def check_stale_project_names(root: Path) -> Iterator[Violation]:
    """No source, document or configuration may name a project under its old name.

    Separate from :func:`check_stale_terms`, which covers markdown and predates this.
    **Most of the occurrences this rule exists for are in Python docstrings**, so a
    markdown-only scan would pass on a tree where the majority of its references are
    stale -- which is the failure mode of a gate that checks the wrong files and
    reports success.

    One violation per file per term rather than one per line: a file that names the
    project forty times is one thing wrong with one file, and forty lines of output
    for it makes the real problems harder to find.
    """
    for path in iter_source_and_docs(root):
        text = path.read_text(encoding="utf-8", errors="replace")
        for stale, current in RENAMED_PROJECTS:
            remainder = text
            while (index := remainder.casefold().find(stale)) != -1:
                tail = remainder[index + len(stale) :]
                remainder = tail
                if any(tail.casefold().startswith(suffix) for suffix in STALE_NAME_EXEMPT_SUFFIXES):
                    continue
                yield Violation(path, f"names the project as {stale!r}; it is {current!r}")
                break


#: The whole policy, in order. Append here rather than writing a new function.
RULES = (check_links, check_stale_terms, check_stale_project_names, check_citations)


def main() -> int:
    """Run every rule and report every violation, rather than stopping at the first.

    Reporting one violation per run turns a five-minute fix into five fix cycles,
    and a contributor who fixes a documentation gate once and finds a new problem
    on the next attempt stops running it.
    """
    root = Path(__file__).resolve().parents[1]
    violations = [violation for rule in RULES for violation in rule(root)]
    if violations:
        print("\n".join(violation.render(root) for violation in violations), file=sys.stderr)
        return 1
    if os.environ.get(STRICT_ENV) and missing_repositories():
        return SKIPPED_EXIT_CODE
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
