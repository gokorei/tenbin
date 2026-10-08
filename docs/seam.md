# The seam, and the honest state of every dependency on it

What Tenbin reads, and which Kojutsu change makes each part available. **A
dependency presented as satisfied is worse than a dependency presented as missing**,
so every row below states the status as the source says it is — including, in four
cases, that the ticket which was supposed to deliver the fact has landed and the
fact is still not there.

That last case is the one the previous revision of this document could not
express, and it is why it named four tickets as blockers that had all shipped.

| | |
|---|---|
| Verified against | `~/Documents/kojutsu`, branch `feat/corpus-answerability` at `22198f5` |
| Date | 2026-09-30 |
| Previously verified against | `fix/worker-durability-and-real-cycle` at `a214cb8` |

Tenbin reads **only** Tanseki, over Kojutsu's documented seam
(`docs/tanseki-seam.md`). Tenbin never reads Kojutsu's local SQLite, never opens
the registry file, never opens the read log, and never calls a Kojutsu process.
That is a deliberate constraint, not a limitation — see
[`decisions/001-registry-through-tanseki.md`](decisions/001-registry-through-tanseki.md).

**"Out of seam" and "does not exist" are different refusals and this document now
distinguishes them.** The read log is the case in point: it exists, it is
deliberately a local file rather than a document, and its absence from this
program's reach is Kojutsu's decision rather than an omission. A dependency that
does not exist is somebody's problem; a dependency that exists out of reach is
somebody's *position*, and only the second can be argued with.

## Transport

| Operation | Endpoint | Availability |
|---|---|---|
| Enumerate documents | `GET /v1/documents?collection=&limit=&offset=` | in use; `limit` capped at 500, `offset` at 10 000 |
| Fetch one document | `POST /v1/documents:get` | in use |
| Batch fetch | `POST /v1/documents:query` | in use; `MAX_BATCH_IDS` is 100 |
| Lexical search | `GET /v1/search?q=&collection=&tags=&fm=&limit=&offset=` | on the client, capped at 500; the walk enumerates rather than searching |
| Graph traversal | `POST /v1/documents:traverse` | in use; `StoreClient.traverse`, `tenbin.corpus.graph.traversed`; depth bounded 1–8, and a 404 reads as an empty neighbourhood rather than an error |
| Count | `GET /v1/documents?limit=1&collection=` | in use, as the store's own total |

**These were checked against a real store on 2026-10-01**, not read from Tanseki's
source. `tests/integration/` records each observation in a warnings summary tagged
`SEAM-OBSERVED` and naming the store it came from, and the answers were:

| Question | Answer | Where Tanseki says so |
|---|---|---|
| Which key carries the listing array | `documents` — the first of the four the client accepts, so the client works because it is right rather than by luck | observed only |
| Is `total` present, and does it agree with the walk | **present**, and it agreed: a 13-document collection read `enumerated=13, store_total=13` | observed only |
| Batch bound | 100 ids accepted with 200; 101 refused with 400 `invalid_argument` | `tanseki/core/domain/src/main/kotlin/gokorei/tanseki/core/domain/RequestLimits.kt:12` |
| Absent ids | **omitted, not nulled** — three asked for, two entries returned | observed only |
| Offset ceiling | `offset=10000` served with 200; `offset=10001` refused with 400 `invalid_argument` | `tanseki/core/domain/src/main/kotlin/gokorei/tanseki/core/domain/RequestLimits.kt:21` |
| Does search carry a `total` | **no** — keys are `hasMore`, `hits`, `limit`, `offset` | observed only |
| Traversal array key | `ids` | observed only |
| Which frontmatter keys become graph edges | `repo`, `pr`, `jira`, and optionally `files` | `tanseki/core/application/src/main/kotlin/gokorei/tanseki/core/application/EdgeDeriver.kt:44` |

**Why some rows say "observed only" and others cite a line.** A row with a citation was
checked twice: the citation says where the bound is written down, and the
integration suite says the running store enforces it. A row that says "observed only"
is a fact about the wire with no single line behind it — an envelope key is chosen by
a serialiser, not declared anywhere, and there is nothing to point at. Saying so is
the honest entry, and it is why the citation rule is not the whole answer to rotted
claims: `scripts/check_docs.py` can tell you a file was renamed, and only a test
against a running store can tell you the serialiser changed its mind.

**The offset result is the one that settles a design question.** The client treats
10 000 as a clean stop and the store agrees by *refusing* past it with a 400 rather
than by returning a short page, so truncation is visible rather than silent — which
is what the snapshot's completeness field is built to report. Had the store answered
past the cap with a short page instead, a walk would have stopped without saying so.

**The binding constraint is `limit`.** Tanseki has no aggregation, no group-by and no
time-series query. Every measure in Tenbin is therefore a full-enumeration
computation over documents fetched in batches, not a query — and the enumeration is
two-step, because the store's own `total` is the only independent check on the walk
available to anybody.

**Two ceilings are real limits rather than policies**, and both are named in the
snapshot rather than discovered as a failure:

- **The offset ceiling is 10 000, and it is a ceiling on the listing rather than on
  the walk.** The largest offset Tanseki will serve is 10 000 and the last page read at
  that offset returns at most one page of documents, so the addressable corpus is
  `10 000 + page_limit` documents and **no number of walks reaches one more**
  (`tenbin.readmodel.resume.reachable_ceiling`). A corpus up to that size is walked
  whole in a single window and there is nothing to resume; a larger one is reported as
  `Completeness.OFFSET_CAP` carrying the exact size of the hole and a histogram of
  where the cut fell. `tenbin read-model build --resume` re-lists the seam to confirm
  the hole and to pick up documents the listing moved into reach, and it finishes a
  walk that failed partway. It is not an update path: the model is still replaced
  whole, so a resume that fails leaves the previous read exactly as a plain build
  would.
- **The response budget is character-based, not row-based**, which is why a batch
  read can return fewer documents than it asked for. The walk steps by the number
  of ids the previous page returned rather than by the page size requested,
  because a short page means the store withheld something.

That is tractable at pilot scale and is the first thing that will not scale. The
honest response when it stops being tractable is a derived read model rebuildable
from the canonical documents — not a second source of truth, and not cached results
presented as measurements.

## The data Tenbin needs, and where each part comes from

| Need | Available | Kojutsu ticket |
|---|---|---|
| Enumerate all knowledge records | now | — |
| Read provenance: `capture_source`, `independence`, `independence_reason` | now | — |
| Read attribution: `comment_author`, `answered_by_agent`, `answered_by_model`, `github_author_association` | now | — |
| Read record kind | **now, written**; legacy records have none and are inferred | `RJNVJK1P` (landed) |
| Read rationale chain: `rationale_source`, `rationale_revision`, `rationale_revises` | now | — |
| Read `review_id`, so a verdict pairs with the review it came from | **now, written** | `RJNVJK1P` (landed) |
| Read who opened a change | **now, written** as `change_author_account` | `3QRPK52A` (landed) |
| Read when a change opened | **now, written** as `pr_opened_at` | `9PTQE45Y` (landed) |
| Read whether a change merged or was abandoned | **now, written** as `pr_outcome` and `pr_merged_at` | `QEVSTMYW` (landed) |
| Read which commit a capture was taken at | **now, written and read** as `head_sha` | `MWF7Z1EN` (landed) |
| Read which files a change touched | **now, written and read** as `files`, bounded and marked; a file past the bound is absent, not undisturbed | `MWF7Z1EN` (landed) |
| Read what a check concluded | **now, written and read** as `check_*`; `RecordKind` knows `check_run` | `RSF1DK1S` (landed) |
| Read which check concluded what, by name | **now, written and read** as `check_name` and `check_id` beside `check_conclusion` | `RSF1DK1S` (landed) |
| Read the declared-vs-reconstructed relationship for a change | **now read and measured**; Tenbin computes it over the projected bodies rather than reading Kojutsu's in-memory field | `XMHKJJ9C` (landed) |
| Read the planning ticket a change was opened against | **now read** as `jira`; presence means the branch name carried one, so absence is *not supplied* rather than *none* | `XMHKJJ9C` (landed) |
| Read decision requests that reached a terminal state | **now, projected** | `AZ8T5XYS` (landed) |
| Read decision requests that are *outstanding* | **not projected, by design** | `AZ8T5XYS` |
| Know which changes had no capture at all | **not recorded** | `EB6FE5ZP` |
| Read whether knowledge was retrieved | **out of seam** — a local JSONL file, not a document | `2XMZ0TYY` (landed) |
| Read defects, incidents, production outcomes | out of reach | — |
| Read work that never became a pull request | out of reach | — |

Those three rows were blocked on **this** repository rather than on Kojutsu,
which is a distinction the previous revision had no way to make. A fact
Kojutsu writes and Tenbin does not read is not a missing fact; it is an unread
one, and the fix is a field on `Record` rather than a ticket in another project.
`Record` now carries `files`, `head_sha` and the check-run kind, so the two
`MWF7Z1EN` rows and the check-run row have left that position.

## What that table forecloses, measure by measure

| Measure | Blocked on | Unblocked by |
|---|---|---|
| Independence and capture-source distribution | nothing | available now |
| Capture mix by repository and author | nothing | available now |
| Rationale revision pressure | nothing | available now |
| Review verdict rates by reviewer | nothing | available now |
| Time to first review | nothing | available now |
| What kind of thing gets captured (`category`) | nothing | available now |
| Who was standing when they said it (`github_author_association`) | nothing | available now |
| The reason behind each independence level (`independence_reason`) | nothing | available now |
| Answers kept, against answers the registry recorded | nothing | available now |
| Disagreements between the question registry and the captured corpus | nothing | available now |
| Time from an answer being written to the corpus holding it | nothing | available now |
| How long a superseded request waited before the code moved | nothing | available now |
| What each named gate concluded, per gate | nothing | available now |
| Unanswered decision requests, and their age | the outstanding population, which the projection deliberately omits | nothing in this project |
| Single-custodian knowledge per area | nothing | available now |
| Rationale staleness against code movement | nothing | available now |
| Coverage of development | a census of changes that produced no capture | `EB6FE5ZP` |
| Whether retrieval changed anything | read events, and a counterfactual Tenbin will not have | `2XMZ0TYY`, and see [`002-no-causal-claims.md`](decisions/002-no-causal-claims.md) |
| Whether agentic development is more effective | an assignment mechanism | nothing in this project |

**Two rows moved from blocked to available, and the reason is that four tickets
shipped.** `review_id` and `record_kind` are written and read, so a verdict pairs
with its review and its kind is stated rather than guessed; `pr_opened_at` is
written and read, so time to first review is a subtraction between the change's own
clock and the review's, rather than a claim about the bot's latency dressed up as a
latency. Both measures are implemented and were blocked only by a catalogue entry.

**Two more rows moved from blocked to available, on `MWF7Z1EN` and `RSF1DK1S`.**
`files` is now read, and it is the first fact that connects a record to code, so
single-custodian knowledge and rationale staleness are both computable; `check_run`
is now a kind this reader knows, so a build's conclusion is a distribution rather
than an unread key. Both refusals are gone rather than softened, because a refusal
that outlives its blocker is the worse of the two failure modes: it tells a reader
the thing is impossible when it is merely unbuilt. What survives is narrower and
lives in the claims instead — a rationale about one file says nothing about the
other files its change touched, a check concluding `success` is a fact about a
build, and reverts are still out of reach, so a merged-then-reverted change is
indistinguishable from one that stuck.

**The census is the one row left, and it is the row that bounds every other
number here.** `pr_outcome` makes a merged change distinguishable from an
abandoned one, so Tenbin reports the merged share of *captured* changes. That is
a true statement about the corpus and a false one about the organisation, and the
distance between the two is exactly what `EB6FE5ZP` closes. Until it lands, every
rate in this program is a rate over changes somebody chose to comment on, and the
sentence saying so belongs in the claim rather than here — which is why
`ChangeOutcomeMeasure` carries it in its own negative space.

**One row is blocked on a ticket that has already landed, and one on nothing at all,
and each is a different thing from being blocked on a missing ticket.** It is the most
useful row in this table, because it is the case a reader cannot otherwise detect:

- **The registry projection ships terminal states only.** "Unanswered decision
  requests, and their age" is about the requests that have *not* reached a terminal
  state, which is exactly the population Kojutsu deliberately excluded: a
  `pending` or `claimed` question is transient and rewritten on every lease, and
  the delivery path is built for immutable records. `AZ8T5XYS` has landed and
  delivered the half it was always going to deliver — the age of requests that
  *reached* a terminal state, which Tenbin now measures — so this row is blocked on
  nothing rather than on that ticket. Delivering the other half needs a mechanism the
  projection is not, not another sweep of the same one.
- **The read log exists and is out of the read seam.** It is a local append-only
  JSON Lines file, deliberately not in Tanseki because a retrieval event is not
  knowledge, and it carries no authenticated caller identity because the MCP server
  is stdio and a single trust domain. It is also off by default, so it has no start
  date until somebody sets it, and appends are not fsynced.

The last row of the table is still the one worth reading twice. No ticket in the
Kojutsu project unblocks it, and that is the correct outcome rather than a gap
in the plan.

## The seam's own limits

Independence says who was positioned to disagree, not whether anything is correct
(`src/kojutsu/models.py:171`). `answered_by_model` is a self-assertion by the
comment author that no part of the system can verify (`docs/github-seam.md:112`). A
rationale is permanently `asserted` and is a claim about intent, never evidence
about code (`docs/design-review/rationale.md:75`).

Tenbin inherits all three. A measure that ranks records by independence is ranking
them by how many people were in a position to object; one that groups by
`answered_by_model` is grouping by an unverifiable self-declaration; one that dates
a rationale against code is dating a statement nobody checked against anything.
All three are permitted, provided the report says which it is doing — which is the
same rule that puts the caveat before the figure.

**The seam is one-way and read-only in both directions.** Tenbin never writes to
Tanseki and Kojutsu never calls Tenbin, so the two projects can move on their own
schedules without coordinating a protocol. The cost of that independence is the
whole reason this document needs re-verifying and the reason the citation gate
exists: a one-way seam has no back-channel that would notice when the other side has
moved.

## What this table does not establish

**It is a snapshot, and the branch and commit at the top are the only thing
distinguishing it from a guess.** Kojutsu ships changes to this surface without
asking, and a row here can be wrong within a day. Nothing in either repository
notices, which is exactly the failure this document's previous revision
demonstrated: it named four tickets as blockers that had all shipped, and every
row above it about record kind, change authors and merge outcomes was wrong for the
same reason.

**"Landed" is not the same as "delivered", and this table now says which is
which.** Three rows are blocked on shipped tickets, so a reader who trusts a ticket
identifier as a promise will be misled by this table — the "Unblocked by" column
names the Kojutsu work that would have to change, not a ticket whose delivery
would be enough. That is a weaker guarantee than the column's name suggests and it
is stated here rather than left to be discovered.

**A row marked `available now` means the corpus supports the measure, not that
Tenbin has built it.** Four of those rows are blocked on nothing on the Kojutsu
side and are implemented or straightforward; none of that is a claim about whether
the resulting figure would be *interesting*, only about whether the fact is there
to compute it from. A complete table and a useful report are different artefacts.

**Nothing here makes the corpus representative.** Every row above describes what
can be read, and none of them describes what was captured in the first place. The
self-selection and self-censoring discussed in
[`corpus-inventory.md`](corpus-inventory.md) sit upstream of this entire table, and
a fully-satisfied seam is what makes that bias visible rather than what removes it.

