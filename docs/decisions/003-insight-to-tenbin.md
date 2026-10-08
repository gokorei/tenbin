# 003 — This program is Tenbin, and a Tenbin is an instrument rather than a verdict

**Status:** accepted
**Date:** 2026-10-02
**Affects:** package name, CLI entry point, environment prefix, report title

## The decision

This program is renamed from Insight to Tenbin (天平), the balance. The Python
package is `tenbin`, the CLI is `tenbin`, the environment prefix is `TENBIN_`,
and the rendered report is titled `Tenbin report`.

## Why

**The old name described a benefit, which is a thing a measurement program should
not do.** *Insight* promises that reading the corpus will make something clearer.
That is the claim this project spends its whole design refusing to make: what
follows from a corpus is bounded by what went into it, and the interesting part
is the boundary. A name that advertises understanding invites exactly the reading
the README opens by rejecting — that a report is a window onto how a team works
rather than a statement about what one corpus supports today.

**A balance reports disproportion, and can report nothing else.** That is the
whole semantic content of the instrument: it compares two things against each
other, it cannot produce a verdict on either of them alone, and an unbalanced
pan is a finding rather than a failure of the balance. A scale whose reading *is*
the finding matches a program whose caveat is rendered before the figure, and
whose refusals are structural rather than advisory.

**The family names its tools for the job the tool does, and this one's job is
weighing.** Kojutsu keeps the conversations and elicits the reasoning, Tanseki
holds one corpus under many views, Kude is the editor, Musubime ties the notes.
Nothing in that set is named for a benefit, and an inconsistency here would read
as a claim the rest of the family does not make.

## What the name does not license

The choice of *balance* is a commitment, not a decoration, and it is refused at
the same place every other refusal in this project lives.

- A balance states a comparison, so a program named for one is under a standing
  obligation to say **what is being weighed against what**. An unstated
  denominator is not a shorter report; it is the failure.
- A balance does not adjudicate. Ranking principals is a verdict on a person, and
  `002-no-causal-claims.md` already refuses it because the corpus cannot support
  it, not because the instrument was the wrong shape. The name adds no licence
  to revisit that, and `tenbin.claims.gate.compares_principals` is unchanged by
  this decision.
- A balance with a hidden fulcrum is a rigged one. A measurement whose denominator
  is the set of people who answered a question is balanced against the wrong
  axis, and the name is a standing reminder to state the axis on the page.

## What changed, and what deliberately did not

**The environment prefix is a breaking change.** `INSIGHT_*` becomes `TENBIN_*`
with no compatibility shim, because a setting that silently stops being read
produces an unconfigured store, and this project treats "unreachable corpus" and
"empty corpus" as the same finding for exactly this reason. A shim that accepted
both prefixes would make an operator's stale file keep working until the day it
mattered. Operators rename their variables deliberately.

**One identifier keeps the old name, and the gate says so.** `insight-seam-check` is
a collection in a running store — the same category as `chronicler-gokorei` — so it
is on `STALE_NAME_EXEMPT_SUFFIXES` rather than renamed. The seam suite creates and
deletes it, so nothing is lost by leaving it spelled the old way, and a developer's
local store already holds it. `TENBIN_TANSEKI_COLLECTION` may still be pointed at it.

The seam's *id* namespace did change, from `insight-seam/` to `tenbin-seam/`, and the
distinction is the point rather than an oversight. A namespace is written and read by
the same code in the same run, so renaming both halves leaves the pair consistent;
leaving the prefix alone would have meant a half-renamed marker. The cost is that rows
written by an earlier version keep the old prefix and are no longer recognisable as
seam rows by pattern — which costs a developer one manual delete in the collection
this project already describes as the operator's to clear.

**The gate now applies to this name too, and to one more file.**
`("insight", "tenbin")` joins `RENAMED_PROJECTS`, so the old name cannot come back
in prose, module paths or configuration keys. Verified by reintroducing it: the
gate reports it and exits non-zero.

`STALE_NAME_SUFFIXES` also gained `.example`. `.env.example` still said *Chronicler*
after that project was renamed, and this rule did not catch it, because the suffix
list stopped at `.txt` and that file's suffix is `.example`. A scan list is a claim
about where a name can hide, and a file that demonstrably hid one now belongs in it.
The stale word in that template is fixed as part of this rename rather than
separately, because it is the same sweep.

**The citation check had not been checking one of its two repositories.** The Kojutsu
checkout default still pointed at `~/Documents/chronicler` after the clone moved to
`~/Documents/kojutsu`, and an absent checkout is a *skip* rather than an error — so
every run since that rename reported all 38 Kojutsu citations as UNCHECKED, printed a
warning nobody reads twice, and exited zero. Fixed, and recorded here because it is the
failure this gate exists to prevent arriving through the gate's own default: a citation
that was never checked is indistinguishable from a citation that was checked and agreed.
`missing_repositories()` returning nothing is the check that this stays true, and the
warning on stderr is no longer the only thing standing between a stale path and a
green run.

Two dead constants carrying the old name — `DEFAULT_CHRONICLER_CHECKOUT` and
`CITED_PACKAGE_ROOT` — and a dead `cited_checkout()` were removed rather than renamed.
They had no callers, they were invisible to the rename rule because
`scripts/check_docs.py` is exempt from its own scan, and a constant that documents
resolution the code no longer performs is worse than no constant. The strategy
docstring they sat under now describes `prefix` and `package_roots` as the
per-repository values they are, instead of naming one repository's paths as though
they were the rule's.

## How this is enforced

The same way the previous rename was: by the rule, not by the document. Renaming
120 files and trusting a reviewer to notice a straggler is a rename that holds
until the next person touches the tree. `scripts/check_docs.py` is scanned by
`check_stale_project_names` on every CI run, and it now names this project's own
former name — which is the one rename whose gate is self-applying.

## What would make this wrong

- If the name is read as endorsing the comparison it cannot make. If the balance
  becomes a word for a scoreboard in practice, the name is actively misleading and
  this decision should be revisited rather than defended on the grounds that the
  refusal in `002` already stops the harm.
- If "Tenbin" stops being pronounceable for its audience. A name its own users
  cannot say is not a name, and unlike a package name it is spoken.