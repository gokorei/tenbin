# Tenbin

A measurement program built on the records Kojutsu captures. It reads; it does
not collect.

Kojutsu owns capture and provenance — the trust axes, the anchors, the
evidence fencing, the argument about what a record is worth. Tenbin owns
measurement and claims: what the corpus supports saying, what it does not, and
what would falsify each thing it says.

The seam is one-way and it is a read seam. Tenbin never writes to Kojutsu,
and Kojutsu never calls Tenbin.

## Quickstart

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/gokorei/tenbin.git
cd tenbin
uv sync --locked --all-extras
cp .env.example .env
# Edit .env: set TENBIN_TANSEKI_COLLECTION to the collection your
# Kojutsu instance writes to (read it from `tanseki_collection` in
# your Kojutsu instance's `kojutsu.toml`; there is no usable default).
uv run tenbin claims
uv run tenbin refusals
uv run tenbin report
```

`claims` and `refusals` need no store. `report` needs a reachable Tanseki
store (`TENBIN_TANSEKI_URL`); without one it raises rather than reporting
zero, because an empty corpus and an unreachable corpus look identical from
the outside and only one of them is a finding. See `CONTRIBUTING.md` for the
full verification suite (`ruff`, `mypy`, `pytest`, `scripts/check_docs.py`).

## License

Apache-2.0 — see [LICENSE](LICENSE).

## What this is for

Kojutsu captures what was said about a change. The gap it leaves is that
nobody can say anything about *the process that produced those statements* — how
long a decision request waited, whether anything was recorded at all for most
changes, whether a change was merged or abandoned, what a stated reason looks
like six months after the code moved.

That gap is where a measurement program belongs, and the reason it is a separate
project is that it is a different kind of claim. A captured record is evidence
about a change, subject to the anchor rules. A measurement is a claim about a
corpus, subject to falsification, and the two must not be allowed to borrow each
other's credibility.

## What this is not

It is not a dashboard, and the distinction is the design. A dashboard renders
numbers; a measurement program states what each number means, what it does not
mean, and what observation would prove it wrong, and renders the caveat before
the figure. A number without a claim attached is a claim.

It is not a productivity tool. Per-principal effectiveness is a different
artifact with different consent, retention, and access requirements, not the same
report with a different filter. Granularity defaults to area and team.

It cannot produce a causal claim. Observational data on which principal happened
to write a change cannot support a counterfactual — agents get the well-specified
tickets, and the selection is in the data. Every measure here is descriptive, and
a comparative effectiveness statement is structurally refused rather than
cautioned. See [`docs/decisions/002-no-causal-claims.md`](docs/decisions/002-no-causal-claims.md).

## The corpus, honestly

Tenbin reads a corpus that is **self-selected and self-censored**. Kojutsu
records what someone chose to answer. A repository where nobody answers is
byte-identical to a repository with no knowledge debt.

This is not repairable by anything in this project. It is repairable only by
backfill, and no backfill path exists. What *is* repairable is legibility — a
census of observed-but-uncaptured changes makes the bias visible, so a rate can be
reported as a rate over a stated population rather than as a number that implies
representativeness it does not have. Until that census lands, Tenbin reports
counts and says so.

## What a refusal is, and how to see the whole list

A refusal is a claim about the corpus, made as carefully as any claim a report
renders. Someone asking "why is review throughput not in the dashboard" deserves a
specific answer — this fact is missing, it dies here, this Kojutsu ticket would
deliver it — and not a shrug and not silence.

A refusal has four fields: what the measure is called, why it cannot be rendered,
what is missing, and **what would unblock it**. The last field is the one that is
usually wrong, and the one this program goes out of its way to render.

```console
$ tenbin refusals
```

Two things about that output are load-bearing:

- **`nothing will unblock this` is an answer, not a gap.** Two of the entries are
  in that state, one of them this project's central question. The distinction
  matters because both available mistakes are silent: filing a ticket that cannot
  possibly deliver the fact, and leaving a field empty so a reader assumes somebody
  is working on it. A blank ticket list is never valid, because a blank reference
  in a report reads as a reference to something.
- **A refusal is content, not an exception.** A claim the gate will not permit
  becomes a rendered section — the figure is dropped, since the figure is the thing
  being refused, and the refusal takes its place. What a refusal never produces is
  a stub, an empty chart, or a zero.

`tenbin claims` prints the other half: every claim the program makes alongside the
refusals, with no store at all, so the reasoning can be read without a corpus.

## What the corpus cannot support, however this program is written

The two limits above are the big ones: no counterfactual, no backfill. These are
the smaller ones, and they share the property that no amount of care in a measure
removes them — they are properties of what was captured, not of how it is read.

**A missing fact and an unreachable fact are different refusals, and they look
identical from outside.** Kojutsu's read log records that a retrieval happened.
It is a local append-only file, deliberately not in Tanseki, because a retrieval event
is not knowledge. So the data is real, this program cannot see it, and a reader who
knew only that the log exists would not guess that the corpus itself says nothing.
`docs/seam.md` marks every row of the seam with which of the two it is.

**A fact written down is not a fact anybody checked.** `answered_by_model` is a
self-assertion by the account that posted; independence describes who was
positioned to disagree, not whether anything is correct; a rationale is a claim
about intent and never evidence about code. Tenbin renders all three and says
which it is rendering, which is the same rule that puts the caveat before the
figure.

**A stored record cannot support a conclusion about its own author.** Kojutsu
writes `change_author_account` because a login is an event and not a verdict;
`pr_outcome` because whether GitHub reported a merge is a fact and not whether the
change was any good. Tenbin borrows that vocabulary deliberately. A measure that
ranks authors, or that reads an abandoned change as a rejected one, is drawing a
conclusion the record does not carry — and the record will not stop it.

**The field-by-field inventory of what Kojutsu persists today, what it receives
and discards, and what is therefore unavailable, is
[`docs/corpus-inventory.md`](docs/corpus-inventory.md).** It is the honest answer
to "what can this program possibly say", it names the branch and commit it was
verified against, and it should be read before any measure is designed.

## Commands

| Command | What it does |
|---|---|
| `tenbin report` | Take a read, measure it, and render the report — Markdown by default, `--format json` for the same content as named keys |
| `tenbin claims` | List every claim and every refusal, with no store; `--read-model` fills in the denominator sizes |
| `tenbin refusals` | List the refusal registry: what is not measurable, why, and what would change that |
| `tenbin drift` | Render one measure against its own past value, from two archived readings. This is the only figure here that compares, and it compares *time* rather than people |
| `tenbin retrospective` | Render one **declared** period of ticket flow, read from a ticket source's status transitions. The period's boundaries are stated in the header, and a cadence and an anchor are **required** — see below |
| `tenbin read-model build` | Keep one read of the collection, whole, rebuildable rather than updated. `--resume` continues the read already kept at that path, replacing it whole |

`report` renders the caveats *above* the figures, which is the unfashionable part
of the design and the part that will be proposed for tidiness. `--require-all` on
`claims` exits non-zero when anything is refused, so a refusal is usable in a check.

## Ticket sources

A retrospective reads status transitions from one of three ticketing systems.
`TENBIN_TICKET_SOURCE` names the adapter and `TENBIN_TICKET_API_URL` says where
it lives; one without the other is unconfigured rather than half-configured.

| Source | `TENBIN_TICKET_SOURCE` | Address (`TENBIN_TICKET_API_URL`) | Credential | Project scope |
|---|---|---|---|---|
| go-experiment | `go-experiment` | e.g. `http://localhost:8082` | `TENBIN_TICKET_API_TOKEN`, optional — empty when auth is off | `--project <id>` and `--group <id>` |
| Jira Cloud | `jira` | `https://<site>.atlassian.net` | `TENBIN_TICKET_API_USER` (account email) plus `TENBIN_TICKET_API_TOKEN` ([an API token](https://id.atlassian.com/manage-profile/security/api-tokens)) | `--project <KEY>`; no `--group` |
| Linear | `linear` | `https://api.linear.app/graphql` | `TENBIN_TICKET_API_TOKEN` ([a personal API key](https://linear.app/settings/account/security)) | `--project <TEAM-KEY>`; no `--group` |
| YouTrack | `youtrack` | `https://<domain>.youtrack.cloud/api` | `TENBIN_TICKET_API_TOKEN` (a permanent token) | `--project <SHORT-NAME>`; no `--group` |

Three things hold for every source, and the adapters state the rest:

- A transition is a ticket, a from-state, a to-state and a moment — no
  principal. Jira changelog entries carry an author and Linear history entries
  carry an actor; the adapters drop both at the seam, so *who carried what*
  stays refused however the tickets were read.
- For Jira and Linear, `total` counts the *issues walked*, and the transitions
  beside it are the in-window status changes derived from them. A
  retrospective resets both counts to the narrowed rows, so its header still
  reads "N of N" and means it.
- Jira and Linear shapes were checked against their public API references, not
  a live instance. Each adapter module says what was verified and what was
  assumed; a drifted field surfaces as unreadable rows rather than wrong
  figures.

## A period is declared, never inferred

`tenbin retrospective` will not pick a sprint length. Nothing this program reads
declares one, so a retrospective takes a **cadence** and an **anchor**:

```bash
TENBIN_TICKET_SOURCE=go-experiment \
TENBIN_TICKET_API_URL=http://localhost:8082 \
tenbin retrospective --cadence 14d --anchor 2026-01-05T00:00:00Z --archive retro.jsonl
```

Both can come from `TENBIN_PERIOD_CADENCE` and `TENBIN_PERIOD_ANCHOR` instead.
There is deliberately **no default for either**, because a defaulted cadence
would be invisible in every rendered report: it would scope every figure to a
boundary nobody chose while the output said nothing about it. Undeclared is an
error naming both variables.

Periods are half-open — `[start, end)` — so a transition recorded exactly on a
boundary belongs to one period and not two, and two consecutive retrospectives
add up to the corpus rather than to the corpus plus its seams. A period with
nothing in it says which kind of nothing it is: no transitions with history
before the window means the work did not move tickets, and no transitions *and*
none before it means the log had not started recording, which is a fact about
the installation and says nothing about the team.

Two slides a retrospective audience asks for are **refused in the output rather
than omitted**: *who carried what*, because a status transition records no
principal and there is no field to group by; and *committed versus delivered*,
because no supported source validates estimates against delivery, so they are
empty and a comparison over them would render a shortfall in delivery out of
two unpopulated fields.

`report --archive <path>` appends the run's figures to an append-only readings log,
one line each, and `drift` reads it. **The archive is never re-computed and never
rewritten**: a measure that changes shows up as two readings rather than one
corrected reading, because a drift programme that rewrites its own past has no
drift to report. A drift is one population read twice, which is why it is permitted
where agent-versus-human is refused — see
[`docs/decisions/002-no-causal-claims.md`](docs/decisions/002-no-causal-claims.md).

## Layout

| Path | Contents |
|---|---|
| `src/tenbin/store/` | The read seam's transport: one read-only HTTP client and its failures. No writes, no interpretation. |
| `src/tenbin/corpus/` | What a stored document *is*: records read leniently, the snapshot saying how far the walk got, and the stated/unstated distinction |
| `src/tenbin/claims/` | The claim protocol — `Claim`, `Denominator`, `ClaimGate`, and the catalogue of refusals |
| `src/tenbin/measures/` | Figures and measures: the layer where a claim and a number become one value |
| `src/tenbin/report/` | A corpus header, sections, and the one order a section is read in |
| `src/tenbin/readmodel/` | A derived read of the store, kept whole and rebuilt rather than updated |
| `src/tenbin/cli.py` | The command surface above, and the only place the walking happens |
| `docs/corpus-inventory.md` | What Kojutsu stores, what it drops, what that forecloses — with the branch and commit it was verified against |
| `docs/seam.md` | The read contract Tenbin depends on, and which Kojutsu tickets satisfy it |
| `docs/measure-protocol.md` | How to add a measure, and what it has to get right |
| `docs/decisions/` | Decisions with their reasoning, in the order they were made |
| `scripts/check_docs.py` | The documentation gate: links, stale terms, and cross-repository citations |

## Documentation that makes claims about another repository

`docs/corpus-inventory.md` and `docs/seam.md` state what Kojutsu's source does,
by file and by line, so a reader can re-check rather than trust. Kojutsu is a
repository this project does not own, so those citations rot — and **a rotted
citation is worse than no citation because it still looks checked.**

`scripts/check_docs.py` resolves every one of them against a real checkout and
fails when a file has been renamed or a line has drifted out of range. When it
finds no checkout it says so on stderr rather than reporting a clean run over
claims it never checked. That rule exists because the inventory was verified
against the wrong branch, went stale in six places, and nothing noticed until a
human read the source.

**Two repositories are cited into.** Each has an environment variable and a default:

| Repository | Variable | Default | Public checkout | What Tenbin claims about it |
|---|---|---|---|---|
| Kojutsu | `KOJUTSU_CHECKOUT` | `~/Documents/kojutsu` | [gokorei/kojutsu](https://github.com/gokorei/kojutsu) | The corpus: what is captured, discarded, and refused |
| Tanseki | `TANSEKI_CHECKOUT` | `~/Documents/tanseki` | [gokorei/tanseki](https://github.com/gokorei/tanseki) | The store: its `/v1` contract and its pagination ceiling |

The ticket backend used to be a third. Its adapter seam replaced source
citations with observed contracts instead: the go-experiment adapter documents
what a live backend was probed to enforce (page ceiling, inclusive time
bounds) in its own module, and the Jira and Linear adapters name the public
API references their wire shapes were checked against. A SaaS API has no
checkout to cite into, so those references are URLs rather than file-and-line
citations, and the MockTransport tests pin the assumed shapes so drift fails
in the suite first.

If you do not have a checkout, the gate says so on stderr and passes without
checking those citations (exit 0); set `TENBIN_DOCS_STRICT_CITATIONS=1` to
make an absent checkout fail instead. Either way, a citation you cannot
re-resolve is unverified — treat `docs/corpus-inventory.md` and `docs/seam.md`
accordingly until you have the sources beside them.

The gate scans source files as well as documents, because that is where the
claims are: the inclusive-bounds contract that a retrospective has to correct is a
sentence in a docstring, and a markdown-only rule would never have seen it. Widening
the scan is also what makes the gate read its own tests, so the deliberately-broken
citations in `tests/test_check_docs.py` are interpolated rather than written out.

Adding a cited repository without adding a seam into it would be the wrong order,
so `pijul-viz` is **deliberately absent**: Tenbin cannot read it yet, and a
citation surface for a repository you have no reader into asserts a coupling that
does not exist. It joins this table when SD001ZP1 lands the endpoint.

## Related work

Kojutsu-side changes Tenbin depends on are filed in the
[Kojutsu](https://github.com/gokorei/kojutsu) project
under group `GSZX9JR2` — *Make the captured corpus answerable*. Tenbin does not
own them and does not modify them; the dependency is named in each ticket's
description and the consumer is named in each implementation plan.

## Contributing and security

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and the verification suite,
and [SECURITY.md](SECURITY.md) for how to report vulnerabilities. `stores.yaml`
and `.env` hold credentials and are never committed — do not paste their
contents into issues or pull requests.

> **Pull requests are not being accepted at this time.** There is no demand
> for external contributions yet, so unsolicited PRs are closed without
> review, automatically. If you have a use case that changes that, please open an issue
> first describing the demand.
