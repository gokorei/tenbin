# 001 — Registry data reaches Tenbin through Tanseki, not a read endpoint

**Status:** accepted
**Date:** 2026-09-30
**Affects:** Kojutsu ticket `AZ8T5XYS`, Tenbin corpus access

## The decision

Kojutsu will project question-registry state into Tanseki as documents. Tenbin
reads it over the same seam it already uses for everything else.

The alternative considered first was a read-only HTTP endpoint over the registry —
either a new server or a route on the existing dev console. That is what the
Kojutsu ticket originally proposed. It is rejected.

## Why

**One read path, not two.** Everything Tenbin needs is in Tanseki except the
registry. Adding a second access mechanism means two authorisation paths, two
paging semantics, two error vocabularies, and a consumer that must understand
which dataset it is looking at before it can decide how much to trust the
answer. `docs/design-review/read-path.md` already records a real defect caused by
exactly this shape: two functions parsed `GITHUB_WEBHOOK_ALLOWED_REPOSITORIES`
with different acceptance criteria, so the write scope was looser than the read
scope and a review could be captured and then be unreachable. That is the failure
mode of two paths, and it is the project's own documented scar tissue.

**The controls already exist and are tested.** The Tanseki read path has the
allowlist predicate, the repository scope check, the bounded response, the
evidence fence, and the read-time provenance re-validation. A registry endpoint
would need its own version of every one.

**The dataset is arguably not "ingestion state" at all.** The README describes the
registry as holding "ingestion state (questions, dedupe, sessions)" and that
framing is why this was never exposed. But a question that was asked, never
answered, and superseded because the code moved is not a fact about ingestion. It
is a fact about a change, and it belongs beside the other facts about that change.

## What it costs, stated plainly

This is a **write path in Kojutsu**, not a read endpoint, and that is a larger
change than the ticket it replaces.

**The projection is eventually consistent.** Today a reader of the registry gets
the current row. A reader of Tanseki gets whatever the last delivery wrote. A
question that has just been claimed may read as `pending` for a moment. For a
consumer measuring outcomes this is acceptable and for one dispatching work it
would not be — which is a real argument that the registry and the store are
answering different questions, and that they are correctly two things.

**Mutable state now travels through a durability path built for immutability.**
The outbox exists so that an Tanseki outage cannot lose a captured answer, and the
README calls a queued row "the only copy of a human's answer". Carrying mutable
operational state through that same path dilutes what the durability guarantee
means. A question row is re-derivable; an answer is not. They should not be
indistinguishable in the queue.

**Write volume goes up.** Registry state changes on every claim, release,
completion and supersession, which is far more often than a capture happens. The
recommendation below reduces this but does not eliminate it.

## The recommendation on write volume

Project on **terminal** transitions — `answered`, `failed`, `superseded` — plus a
periodic sweep so a long-`pending` question still appears. The interesting facts
are the terminal ones: a decision request that was never answered, or that reached
an attempt ceiling, or that was overtaken by its own code moving. `created_at` and
`answered_at` are both stored, so the wait is computable without the transient
states ever being persisted.

Transient states stay in the registry, where they belong. Outstanding work is
operational, and a store of outstanding work in a knowledge store is a queue
pretending to be knowledge.

## Boundaries the projection must not cross

These are the constraints that make this safe, and each is a way the change
could be implemented wrong.

**`claim_token` is never projected.** It is `secrets.token_urlsafe(32)`
(`question_registry.py:1457`) and is the `WHERE`-clause guard on release
(`:1540`) — anyone holding it can release a claim held by someone else. Tanseki is
readable over MCP by agents, so a `claim_token` in a document is a capability
handed to every reader of the store. The projection is an allowlist of columns,
not a deny-list of dangerous ones, precisely so that a column added to the table
later is not projected by default.

**`last_error` is not projected.** It carries internal exception detail and has
no analytical value. The same allowlist covers it.

**A question document is never an answer.** It sits in its own namespace —
`<repo>/pr-<n>/question/<question_id>` — and is counted separately wherever
captures are counted, on the same reasoning the CHANGELOG records for a rationale
rendered as an empty `uncategorized` row. A question with no answer is a request,
not a conclusion, and a store that cannot tell them apart is lying about its
contents.

**A question is never evidence.** It carries no independence level and must sit
below every `min_independence` threshold, because nobody stated anything. A
question asking "why did we choose this?" is not a record of the answer.

## What would make this wrong

- If Tanseki's per-document write cost makes the projection expensive enough to
  threaten the capture path's own reliability. The outbox is shared, and a
  saturating queue would delay answers behind operational state. This is the most
  likely way it fails and it should be measured before the sweep goes in.
- If a consumer needs current rather than eventual state and goes to Tanseki anyway,
  having been told the seam is the way in. The mitigation is to name the
  consistency property in the document shape — `status_as_of` — so a reader can
  see how stale a status is rather than having to guess.
- If the question volume dwarfs the knowledge volume, at which point the "knowledge
  store" contains more requests than records and the corpus is no longer what the
  product says it is.
