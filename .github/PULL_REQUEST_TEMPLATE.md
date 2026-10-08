> **Pull requests are not being accepted at this time** and will be closed
> without review. Please open an issue first describing the demand.

## What this changes

<!-- One paragraph: the measure, fix, or refusal behavior, in the project's vocabulary. -->

## Claim and falsifier

<!-- Every figure needs a claim. New/changed measures: statement, what it does
not mean, what would falsify it, denominator. Docs-only: N/A. -->

## Verification

<!-- The checks from CONTRIBUTING.md you ran. Paste the `SEAM-OBSERVED` /
`SEAM-UNVERIFIED` line for integration runs. -->

- [ ] `uv run ruff check src tests scripts`
- [ ] `uv run ruff format --check src tests scripts`
- [ ] `uv run mypy`
- [ ] `uv run pytest -q`
- [ ] `uv run python scripts/check_docs.py`

## Credential hygiene

- [ ] No contents of `stores.yaml`, `.env`, or API keys pasted anywhere in this PR.
