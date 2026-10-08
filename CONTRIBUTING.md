# Contributing

> **Pull requests are not being accepted at this time.** There is no demand
> for external contributions yet, so unsolicited PRs are closed without
> review, automatically (a workflow enforces this; GitHub offers no setting
> that disables pull requests). If you have a use case that changes that,
> please open an issue first describing the demand. The setup and verification
> notes below are for anyone running or auditing the code, and for the day
> that changes.

Tenbin reads a knowledge store and states claims about what that store can
support. Two of the rules below exist because of that, and both are enforced
rather than advised.

## Setup

```bash
uv sync --locked --all-extras
```

## Verify

Run the same checks CI runs, in this order. A measurement program that reports a
number nobody re-checked is the thing this project is built to avoid being.

```bash
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv run mypy
uv run pytest -q
uv run python scripts/check_docs.py
```

The unit suite runs with no store and needs nothing. One suite does need a
running store, and it is the only one:

```bash
uv run pytest -m integration -q -rs
```

**These are the only tests that need a running Tanseki store.** They are marked
`integration`, they write to and delete from a store, and with no store
configured every one of them *skips* with a reason rather than passing. A skipped
run here means the seam was not checked: it does not mean the seam is fine. If
you are reading a green run and want to know whether the wire was actually
exercised, look for the `SEAM-OBSERVED` lines in the warnings summary — each one
names the store it came from. A run that skipped says `SEAM-UNVERIFIED` instead,
and that word means the claims are still only read from Tanseki's source.

**Last verified 2026-10-01**, against an Tanseki store on `127.0.0.1:8099` holding 77
documents fetched from a public organisation. All nine tests passed, every wire
claim was observed, and the suite left its collection empty. To reproduce against
that store:

```bash
TANSEKI_SEAM_STORE=1 TANSEKI_SEAM_URL=http://127.0.0.1:8099/v1 \
  uv run pytest -m integration -q -rs
```

`TANSEKI_SEAM_COLLECTION` defaults to `insight-seam-check`, which is a collection of
the suite's own and not the one Kojutsu writes, so a run never touches somebody
else's knowledge.

### Running the seam suite

You need an Tanseki store listening on loopback. Tanseki is a separate repository; its
own checkout is usually at `~/Documents/tanseki`, and the two commands below build
and start it in the `library` profile (SQLite plus Lucene, no Pijul, no
single-writer lock — none of which this suite is testing):

```bash
docker build -t tanseki-seam ~/Documents/tanseki --file ~/Documents/tanseki/service/Dockerfile
docker run --detach --name tanseki-seam --network host \
  --env TANSEKI_PROFILE=library \
  --env TANSEKI_PATH=/vault/library.db \
  --env TANSEKI_INDEX_DIR=/vault/index \
  tanseki-seam
```

`--network host` rather than a published port: Tanseki refuses wildcard binds, and
Tenbin's client refuses plain HTTP to anything that is not loopback, so the
daemon has to be reachable on `127.0.0.1` and a published port reaches the
container's own loopback from nowhere. The port is `8088` unless
`TANSEKI_HTTP_PORT` says otherwise.

The suite reads five environment variables, and **the first one is an opt-in
rather than a setting**: without it the suite does not run, because it writes to
a corpus and that should never be a side effect of pointing Tenbin somewhere.

| Variable | Meaning | Default |
|---|---|---|
| `TANSEKI_SEAM_STORE` | The opt-in. Set it to anything non-empty to let the suite write and delete. | unset, so nothing runs |
| `TANSEKI_SEAM_URL` | The store's `/v1` base URL. Loopback over plain HTTP, or HTTPS. | `http://127.0.0.1:8088/v1` |
| `TANSEKI_SEAM_COLLECTION` | The collection to write to. Give it one of its own. | `insight-seam-check` |
| `TANSEKI_SEAM_API_KEY` | The credential, for a store with a registry. Empty on a loopback store with auth off. | empty |
| `TANSEKI_SEAM_STRICT` | Turn "no store" into a failure instead of a skip. What CI sets. | unset, so a skip is a skip |

```bash
TANSEKI_SEAM_STORE=1 uv run pytest -m integration -q -rs
```

Two things to know before you point it at a store you care about. It **writes**:
it creates documents under a run-unique id prefix and deletes them in a
`finally`, and a session that ends with documents still present fails rather than
warns quietly — but a run killed mid-flight can leave some behind, and they are
identifiable by the `tenbin-seam/<timestamp>-<random>/` prefix. And it
**enumerates**: the offset-ceiling test walks the whole configured collection, so
point `TANSEKI_SEAM_COLLECTION` at a collection of its own rather than at a corpus
with a hundred thousand documents in it. Nothing in the suite asserts anything
about documents it did not write.

The CI job `Seam against a real Tanseki store` does all of the above, and pins the
Tanseki commit whose source the claims in `tenbin.store.client` were read from. If
that job goes red for a reason that is not a red test, read the comment above
`Resolve the Tanseki image` in `.github/workflows/ci.yml` first: no published
image exists, so the job builds one from source, and a from-scratch build was
last seen dying under a 4 GB Docker host in a way that reads as memory rather
than as a broken Dockerfile.

## The two rules that are not negotiable

**A number without a claim attached is a claim.** If you add a measure, you add
its claim: the statement, what it does not mean, what would falsify it, and the
denominator. A measure that cannot name a falsifier is an observation, and the
type system in `tenbin.claims` is shaped so you will find out at construction
time rather than in review.

**A limitation that lives in a docstring is a comment.** If you are tempted to
write "this does not support X" in a docstring, write a test that refuses X
instead. Kojutsu records the same practice in
`docs/design-review/README.md`, and decision 002 in this repository is enforced
by a negative test over a clean fixture rather than by prose.

## Tests

The suite runs with no store. Network access is denied at the socket layer and
the environment is stripped of every Tenbin setting before each test, so a test
that needs a store must ask for the fake explicitly and the fake is then visible
in the test's own source. Write fakes inline in the test module, named `Fake*` or
`Stub*`, and name each test for the reason it exists rather than the mechanism it
exercises:

```python
def test_a_wildcard_is_never_authorisation() -> None:
```

Tests that talk to a real store live in `tests/integration/`, are marked
`integration`, and are the exception to the paragraph above: they opt the socket
back on, they write as well as read, and they skip unless a store was named —
loudly, with a reason, and as failures under `TANSEKI_SEAM_STRICT`. See
[Running the seam suite](#running-the-seam-suite).

## Documentation

Docs are checked by `scripts/check_docs.py` and treated as code. Relative links
must resolve. This project's documents also cite another codebase's source by
file and line, and those citations rot silently, so when you touch a claim about
Kojutsu, re-check it rather than trusting it — the citation existing is not the
same as the citation being right.
