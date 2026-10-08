# What Kojutsu stores, and the six facts that changed since this was last checked

What Kojutsu persists, what it receives and discards, and what that forecloses.
Read [`measure-protocol.md`](measure-protocol.md) before designing a measure and
this before designing one at all, because a measure that cannot be built is a
wasted afternoon and this is the document that says so.

**This is a claim about another codebase's source, so it carries a branch, a
commit and a date, and every claim below gives a file and a line.** That is the
correct way to write such a document and it is also fragile, because Kojutsu is
a repository this project does not own and does not control. The previous revision
of this file was verified against `fix/worker-durability-and-real-cycle` and named
no commit at all, and it went stale in six places without anything noticing. The
citation rule in [`../scripts/check_docs.py`](../scripts/check_docs.py) exists
because of that, and it is checked in CI against `KOJUTSU_CHECKOUT`.

| | |
|---|---|
| Verified against | `~/Documents/kojutsu` |
| Branch | `feat/corpus-answerability` |
| Commit | `22198f5` — *feat(capture): record a check run as a machine report about a commit* |
| Date | 2026-09-30 |
| Previously verified against | `fix/worker-durability-and-real-cycle` at `a214cb8` |

---

## 1. What is persisted

Kojutsu writes three document shapes to Tanseki, each with its own id shape and
its own frontmatter builder: `KnowledgeEntry` via `build_frontmatter`
(`src/kojutsu/core/tanseki_mapping.py:101`), `RationaleEntry` via
`build_rationale_frontmatter`
(`src/kojutsu/core/tanseki_mapping.py:224`), and a projected decision request via
`build_question_frontmatter` (`src/kojutsu/core/tanseki_mapping.py:324`).
Between them this is the complete analytical surface a consumer can read.

**Every change-describing key is absent when the payload did not carry it, rather
than defaulted.** `build_frontmatter` skips `None`, and that filter is the only
mechanism in the module that expresses absence honestly — a placeholder string
passes the same filter and is then indistinguishable, to any reader, from a value
somebody stated.

### Knowledge entries — linking

| Field | Source |
|---|---|
| `repo` | metadata |
| `pr` | metadata `pr_number`, stringified |
| `jira` | metadata `jira_ticket_key` |
| `pr_url` | metadata |
| `session_id` | the answer's `session_id` |
| `question_id` | the question this answers |
| `github_comment_id` | metadata |
| `files` | the changed files, as a list — **new** |
| `head_sha` | the commit the capture was taken against — **new** |

`session_id` is a **per-change** identity, not a work-episode identity, and its
name does not say so. It is a deterministic UUID — `uuid5(NAMESPACE_URL,
f"kojutsu:review:{pr_url}")` at `src/kojutsu/cli.py:351` — so it is stable
across runs and derivable from the pull request, which makes it genuinely useful
for grouping records by change. What it is not is a session: a change touched over
three weeks by six agents is one id, and a piece of work spanning three pull
requests is three. Read it as a session and any duration computed from it is a
statement about the pull request, not the work. It also does not exist for work
with no pull request, which is the agentic case the corpus is least able to see.

`files` is the key Kojutsu never used to write, and it is now written on the
review, lifecycle and check-run paths. Two properties matter to a reader:

- **It is a list, not a scalar**, because the extras dict stringifies everything
  and a Python list repr written into YAML is a value no consumer can filter on.
- **It is bounded at 50 and says so when it truncates**
  (`src/kojutsu/core/tanseki_mapping.py:75`). A list quietly shortened reads as a
  complete description of a change. A file list that could not be read is absent
  rather than empty, so a store outage does not look like a file-free change.

**`head_sha` is an anchor, not a verification.** It says which commit the capture
was taken against. It does not establish that the record is still true there, that
the change was reviewed, or that the two correspond to the same code — and
Kojutsu's own comment says a field named `verified_at_commit` would be a claim
the store cannot back.

### Knowledge entries — who

| Field | Meaning | Verified against |
|---|---|---|
| `author` | the answer's author; falls back to `"unknown"` | `src/kojutsu/core/tanseki_mapping.py:117` |
| `comment_author` | the account that posted | metadata |
| `answered_by_agent` | declared agent id, or absent | `extract_agent_claim` |
| `answered_by_model` | declared model, **or the key is absent** | `src/kojutsu/core/answer_collector.py:492` |
| `github_author_association` | OWNER / MEMBER / COLLABORATOR, gated on | metadata |
| `declared_by` | rationale author | `build_rationale_frontmatter` |
| `declared_by_model` | rationale model, **or the key is absent** | `src/kojutsu/core/tanseki_mapping.py:252` |
| `change_author_account` | who opened the pull request — **new** | `src/kojutsu/core/tanseki_mapping.py:179` |

Three properties of these constrain every measure built on them:

- **The model is a self-assertion.** `docs/github-seam.md:112` is explicit that the
  forge proves who posted a comment and nothing more, and that it cannot prove
  which model drafted the text. The claim is trustworthy exactly as far as the
  account making it.
- **`answered_by_model` no longer writes `unknown`.** This is a correction to the
  previous revision of this document. The review and answer paths now leave the key
  absent when no model was named (`src/kojutsu/core/answer_collector.py:492`
  and `:844`), for the same reason a rationale does
  (`src/kojutsu/core/tanseki_mapping.py:252`): a truthy placeholder survives the
  `None` filter and lands in a *queryable* key, where a principal nobody named comes
  to look like one who declared a model called `"unknown"`. Documents written before
  the change still carry the literal, which is why `parse_stated` remains correct.
- **`change_author_account` is a login, not a person.** Kojutsu's own framing,
  repeated here because a reader who treats it as an identity will draw a
  conclusion the store never licensed: it is the account GitHub reports opened the
  pull request — not a person, not a bot, and not a statement about what they were
  doing with it.

### Knowledge entries — trust

| Field | Source |
|---|---|
| `capture_source` | `webhook` / `collect` / `asserted`, defaulted to `asserted` |
| `independence` | `self_certified` / `model_separated` / `independent` |
| `independence_reason` | short auditable phrase |
| `rationale_source` | `declared` / `reconstructed` / `unknown` |
| `delivery_id` | the provider's delivery identifier |
| `github_comment_id` | the source comment |

This axis is the strongest thing Kojutsu has, and it is the reason a downstream
program is worth building at all. It is already a first-class field, already
validated on the write path (`capture_anchor_gaps`,
`src/kojutsu/models.py:56`) and already re-validated on the read path
(`mcp_server/server.py:306`). A rationale is `asserted` permanently and by
default (`src/kojutsu/models.py:293`), because a record has to earn the right
to be anything stronger.

### Knowledge entries — what and when

`category` is a closed six-value vocabulary (`src/kojutsu/models.py:239`) —
`design_decision`, `trade_off`, `domain_knowledge`, `edge_case`, `dependency`,
`system_event`. It is **retrospective by construction**: a rationale states a
reason *before* anyone asks, and adding a forward-looking category would let a
claim carry the label of a decision already taken.

Three clocks, and they are not interchangeable:

- **`answered_at`** is the event the record is about — a review's submission, a
  comment's creation, a close.
- **`captured_at`** is when Kojutsu wrote the record down. It is a fact about
  the bot's latency and never about the work.
- **`pr_opened_at`** and **`pr_merged_at`** — **new** — are the change's own
  timestamps, and `pr_outcome` — **new** — is how it ended. All three are read from
  the payload by the capture paths rather than reconstructed from a hash, which is
  what makes them usable at all.

**`pr_outcome` is written only on a close, and only two values exist.**
`_close_outcome` (`src/kojutsu/core/answer_collector.py:606`) returns `None`
unless the action was a close or the reported state was closed, and otherwise
returns `merged` when a merge timestamp is present and `closed_unmerged` when it
is not — because `state` reads `"closed"` for a merge and for an abandonment alike.
An `opened` record therefore has no outcome at all, which is correct: asserting
either value about it would be asserting something GitHub did not say. **A value
outside that pair must be reported as unrecognised rather than folded into a
neighbour**, because a closed vocabulary on this side would either raise on a
writer's new value or quietly lose the fact that something unfamiliar was written.
`tenbin.corpus.record` keeps `pr_outcome` a plain string for exactly this reason.

### The change a record is about — six keys, newly written

Commit `740111c` (*feat(capture): record what the change was, not just what was
said*) added six keys to `build_frontmatter`, and they are no longer discarded.
The constants are declared in one block
(`src/kojutsu/core/tanseki_mapping.py:54`) and written in the extras dict
(`:168` through `:182`), so the writer imports the name rather than spelling it.

| Key | What it is |
|---|---|
| `record_kind` | `answer`, `review_verdict`, `inline_review_comment`, `pr_lifecycle`, or — for a check — `check_run` |
| `review_id` | the forge's identifier for the review a record came from |
| `change_author_account` | the login that opened the pull request |
| `pr_opened_at` | the change's `created_at`, ISO-8601 |
| `pr_outcome` | `merged` or `closed_unmerged`, on a close only |
| `pr_merged_at` | the change's `merged_at`, a different event from the close |

**Preserve Kojutsu's own reasoning for these keys, because it is the vocabulary
any claim over them has to borrow.** The comment above the constants says it
(`src/kojutsu/core/tanseki_mapping.py:49`): *each names an event rather than a
conclusion, because a stored record cannot support a conclusion about its own
author.* So `change_author_account` is a login and not a person or a bot, and
`pr_outcome` is whether GitHub reported a merge and not whether the change was any
good. A measure that ranks authors, or that reads an abandoned change as a
rejected one, is drawing a conclusion the record does not carry.

`head_sha` joined them in `0d00024`, and the check-run keys in `22198f5`.

### Record kind — written, and a legacy document has none

**This is the second correction to the previous revision, which said there was no
`record_kind` field at all.** It is written on every record the four entry
producers create, and the tags are a *projection* of it computed in one place
(`_record_tags`, `src/kojutsu/core/answer_collector.py:228`, from the
`RECORD_KIND_TAGS` table at `:148`), so the two cannot drift.

The consequence for a reader is sharper than "it is now available". **A legacy
document written before the key existed has no kind at all, and that absence is to
be reported as absence rather than inferred from tags.** Kojutsu states this
reasoning directly at `src/kojutsu/core/tanseki_mapping.py:166`, and it resolves
an ambiguity this project had been modelling: recovering a kind from the tag set
is recording an interpretation as though it had been written down. `classify` in
`tenbin.corpus.record` therefore returns `Certainty.DETERMINED` when the key is
present and readable, and `Certainty.INFERRED` when it is not, and a measure that
needs the first says so with `requires_certainty`.

Two residual limits, both of which matter here:

- **Rationales carry no `record_kind`.** They are dispatched through a different
  payload — `build_rationale_frontmatter` never writes it — so a rationale is still
  classified from its tags, and that is not a gap anybody is going to close.
- **A projected decision request carries no `record_kind` either**
  (`src/kojutsu/core/tanseki_mapping.py:324` writes none). Its tags are `question`
  and `question_<status>`, and neither names a record kind, so Tenbin's
  `classify_by_tags` reads it as an *answer* with inferred certainty. That is a
  live defect on this side and it is recorded in [`seam.md`](seam.md) rather than
  papered over.

### Projected decision requests

The question registry is now projected into Tanseki, in its own namespace:
`<repo>/pr-<n>/question/<question_id>`
(`question_document_id`, `src/kojutsu/core/tanseki_mapping.py:311`). The full
table is in `docs/tanseki-seam.md:85`.

Three decisions shape it, and the second one is the reason the measure this
document previously refused is still refused:

- **Only terminal statuses are projected** — `answered`, `failed`, `superseded`
  (`TERMINAL_QUESTION_STATUSES`,
  `src/kojutsu/core/question_registry.py:162`). A `pending` or `claimed` question
  is operational and transient, rewritten on every lease.
- **`QuestionRecord` is the allowlist.** It has no `claim_token` field and no
  `last_error` field, so the projection *cannot* publish either — not because it
  filters them out, but because there is nowhere for them to be read from
  (`src/kojutsu/core/question_projection.py:22`). A deny-list would make the
  safe behaviour depend on remembering to update a blocklist, and its failure is
  silent. This is a strictly better answer than the one this document used to
  hope for.
- **No `capture_source` and no `independence`,** and the omission is the mechanism:
  that is what excludes a request from an evidence-only query, since nobody stated
  anything.

### Check runs

A check run is a **machine report about a commit**, stored in its own
`record_kind` of `check_run` (`src/kojutsu/core/answer_collector.py:132`) and
never where a review would be. It is the nearest thing GitHub offers to an outcome
signal, and the distinction is carried all the way through:

- the conclusion is stored **verbatim**, as `action_required` rather than
  restated as "failed", because translating a tool's wording into Kojutsu's own
  is editorialising;
- **no independence level**, so an evidence filter cannot return it as a second
  opinion;
- **the author is the check**, not the change's author.

Only `completed` runs are captured, and a re-run arrives under a new check run id,
so a re-run is a separate record rather than an overwrite. A branch check has no
pull request, and rather than invent a `pr_number` the anchor rule accepts a
`check_id` in its place (`capture_anchor_gaps`,
`src/kojutsu/models.py:56`) — a forge-issued, globally unique identifier for
the exact run.

**Revert detection is not implemented, and the gap is deliberate.** A revert is a
commit whose message says so, which means reading commit messages, and a
heuristic that records guesses as facts is the failure this whole exercise is
about.

---

## 2. What changed since the last verification

This section exists because the previous revision was wrong, and **an audit that
never recorded a miss is an audit nobody should trust.** Eight commits landed on
`feat/corpus-answerability` between `a214cb8` and `22198f5`, and this document
tracked none of them.

| Commit | What landed | What this document had to change |
|---|---|---|
| `740111c` | Six change-describing keys written to `build_frontmatter` | §1 gained a section; §2's discarded-facts table lost four rows; the merged-vs-abandoned discussion below is now a stored fact |
| `73818ee` | Dropped a sampling assertion that flaked the injection test | Nothing in this document; recorded because the brief named it as the fixing commit and it is worth being clear that it fixed a test, not a fact |
| `75b9205` | `declared_by_model` left absent when none was stated | §1's *who* table |
| `308f73b` | A local read log, out of Tanseki | §4 no longer says reads are unobserved |
| `0d00024` | `head_sha` and `files` written | §1's linking table; the `files` edge is now written |
| `a947c92` | Terminal decision requests projected into Tanseki | §1 gained a section; the registry is no longer wholly unreachable |
| `690c002` | The rationale comparison behind a CLI command | Nothing here; it is a Kojutsu-side tool, not a stored fact |
| `22198f5` | Check runs captured as a machine report about a commit | §1 gained a section; `RSF1DK1S` clears |

**The two facts the previous revision got most wrong were both about absence.**
It said `record_kind` was dropped by the storage boundary, and it said reads were
not observed at all. In one case a fact exists and is written; in the other a fact
exists and is deliberately out of reach. Neither was "the data does not exist", and
a document that cannot tell those three cases apart is a document that files
tickets for work already done.

---

## 3. What is received and discarded

**After `740111c` this table is short, and that is the headline.** Each of these
arrives on the webhook payload, is used to reach a decision, and now also reaches
the stored record.

| Fact | Where it arrives | Where it went before | Now |
|---|---|---|---|
| Change author account | `pull_request` payload | used by `compute_independence`, then dropped | written as `change_author_account` |
| Merged vs abandoned | `pull_request` close | folded into the identity hash, then dropped | written as `pr_outcome` and `pr_merged_at` |
| PR open time | `pull_request` payload | never read by any capture path | written as `pr_opened_at` |
| `head_sha` | every `pull_request` payload | reached a supersession message only | written as `head_sha` |
| `files_changed` | `pull_request` payload | sized the prompt and reached the plan file | written as `files`, bounded and marked |
| `review_id` | `pull_request_review` payload | not in the extras dict | written as `review_id` |
| `record_kind` | every entry producer | not in the extras dict | written as `record_kind` |
| Check conclusions | `check_run` payload | not subscribed | written as `check_*`, verbatim |

**A merged and an abandoned change are now distinguishable in the record.** That
was the single most consequential gap in the previous revision: they occupied
different entry ids only because the identity hash folded in the merge timestamp,
so a consumer *could* infer the difference by re-deriving the hash, and only if it
knew the derivation, and only for records written after that fact reached the hash.
Nothing in the record said it. `pr_outcome` says it, directly, in one of two
strings, on the close record and nowhere else.

**One thing is still not observable, and it is the important one.** A change with
no capture and a change never examined are the same observation from outside.
Kojutsu writes only on success, so a denominator that means "all changes" is
not available; only "all changes that produced a capture" is, and a rate whose
numerator spans both is a rate over a population nobody can name. `EB6FE5ZP` is
the census that would fix it, and it has not landed.

**The `files` edge resolves against nothing yet.** `repo`, `pr`, `jira` and
`files` are the keys Tanseki's edge deriver recognises
(`docs/tanseki-seam.md:168`), and `files` resolves each path against a document that
already exists — so a `files` edge stays dangling until something writes documents
for those paths. The frontmatter value is useful without the edge; a measure that
wants to *traverse* from a file to its rationale cannot do it yet.

---

## 4. What exists and is out of reach

This section is about the third case, and it is the case the previous revision
could not name at all.

- **Reads are now logged — locally, and deliberately outside Tanseki.** The read log
  lives in `src/kojutsu/core/read_log.py` and is wired into
  `mcp_server/server.py`. It is an **append-only JSON Lines file next to the other
  local state**, and it is not in Tanseki because a retrieval event is not knowledge;
  it is not in the SQLite registry because a read must never take a write lock on
  the single-writer capture path. **It carries no authenticated caller identity**,
  and that is a structural consequence rather than an omission: the MCP server is
  stdio with a single trust domain, so nothing in it can know who asked. What the
  caller *said* about the read is recorded in a field named `caller_claims`
  (`src/kojutsu/core/read_log.py:18`), named as a claim precisely because it is
  not authenticated. Kojutsu says so in the module docstring, and its test
  asserts that no key in an event, and no key inside the claims, reads as an
  identity.

  **The load-bearing consequence for this project is the one the previous revision
  got wrong: the data exists, and it is out of this project's read seam.** That is
  a different refusal from the data not existing. It cannot be retrofitted into
  Tanseki without Kojutsu deciding to do it, and it is off by default
  (`READ_LOG_ENABLED` defaults to `false`), which means it has no start date until
  somebody sets it — a report built on it must state the date rather than draw a
  trend line that implies one. It also does not survive a restart by contract:
  appends are not fsynced, on the argument that telemetry which can fail the read
  it measures has made the store harder to use in order to observe it.

- **Changes that produced no capture.** As above. Kojutsu ticket `EB6FE5ZP`.
- **Reverts.** Not detected, deliberately. See the check-run section above.
- **Defects, incidents, production outcomes, rollback.** Out of reach entirely.
  Kojutsu is a pull-request-comment system with a Jira client that reads bounded
  fields for a branch-name ticket key.
- **Work that never became a pull request.** An agent working in a local
  repository produces no PR, no review, and therefore no record. This is the
  population a "measure agentic development" programme would most want to see.

---

## 5. What this corpus can and cannot support

| Supportable today | Not supportable |
|---|---|
| The trust profile of the store: independence mix, capture-source mix, agent-authored share | Whether a captured decision was correct |
| Review activity and verdict rates by reviewer, over records whose kind the writer stated | Whether a reviewer was right |
| Time to first review, against the change's own `pr_opened_at` | Whether the review was useful |
| Whether a change was merged, abandoned, or is still open | Whether a merged change was an improvement |
| Which files a capture is anchored to, bounded at 50 and marked when truncated | Coverage of files beyond the first 50 |
| What a check concluded about a commit, verbatim, with no independence level | Whether the change that triggered the check was good |
| Decision requests that reached a terminal state: how many were answered, superseded, or hit the ceiling | How long anything is waiting *now* |
| Corpus self-description, once the census lands (`EB6FE5ZP`) | What was decided on changes with no record at all |

**The unstated model must still not be read as a stated one**, and the reasoning
has changed but not the conclusion. Kojutsu now writes the key absent rather
than `"unknown"`, so a *new* corpus cannot produce the bug at all. A corpus written
before the change still carries the literal, and `compute_independence` still treats
an unstated model on either side as `SELF_CERTIFIED` rather than assuming two
unknowns match (`src/kojutsu/models.py:230`). `parse_stated` is what keeps both
corpora readable: it yields `Unstated` for both a missing key and the literal, and a
measure that groups by `answered_by_model` must put `Unstated` in `excluded` rather
than in `values`. See [`measure-protocol.md`](measure-protocol.md).

**Independence describes who was positioned to disagree, not quality.**
`src/kojutsu/models.py:171` states it directly. A distribution of the
independence axis is a statement about the corpus's review structure. It is not a
measure of whether anything is true, and a report that ranks records by
independence is ranking them by how many people were in a position to object.

---

## What this document does not establish

**It is a snapshot of one branch at one commit, and it will be wrong again.** That
is not a caveat that can be engineered away: Kojutsu is a moving repository and
this project does not control its schedule. What can be engineered away is the
*silence* — the citation rule checks every file and line above against a real
checkout and fails when one no longer resolves, so the failure mode is a red gate
rather than a document that quietly describes a repository that no longer exists.

**A citation that resolves is not a claim that is true.** The rule checks that a
file exists and that a line is within it. It cannot check that the line still says
what this document says it says, that a comment was not moved below the fact it
explains, or that a function named at a line is the function this document meant.
Those are the residual rot the rule does not catch, and they are the reason the
branch and commit are in the header rather than in a footer.

**Six corrections is a measure of how quickly this changes, not of how badly it
was written.** The previous revision was careful, cited a file for every claim, and
was still wrong in six places — because the thing it described moved underneath it
and nothing was watching. Read the *What changed* section before trusting any row
above it, and re-verify before designing a measure on it.
