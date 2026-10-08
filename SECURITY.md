# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

Tenbin is pre-1.0. Security fixes are released as patch releases on the
current minor line.

## Reporting a Vulnerability

Do not open a public issue for a suspected vulnerability.

Open a
[private security advisory](https://github.com/gokorei/tenbin/security/advisories/new)
on GitHub, or contact the maintainers through the repository's issue tracker
requesting a private channel. Include:

- what you expected vs. what you observed,
- the Tenbin version (`tenbin --version`, or the commit hash),
- minimal reproduction steps that do not include real credentials.

We will acknowledge receipt within 5 working days and aim to ship or
coordinate a fix within 30 days, depending on severity.

## Scope notes specific to this project

- Tenbin is a **read-only** measurement program: it reads a Tanseki knowledge
  store and a ticket backend, and never writes to either. A report that
  exfiltrates data beyond what those sources already expose is in scope.
- `stores.yaml` and `.env` hold credentials (`api_key` / `TENBIN_TANSEKI_API_KEY`
  / `TENBIN_TICKET_API_TOKEN`) and are
  **never committed** (see `.gitignore`). Do not paste their contents into
  issues, advisories, or logs.
- The web app (`src/tenbin/webapp/`) renders store content as HTML: stored
  XSS via malicious record content is in scope; self-XSS requiring the
  operator to paste credentials is not.
- The plain-HTTP allowance is loopback-only by design (see `CONTRIBUTING.md`):
  bypassing that check to reach a non-loopback host over cleartext is in scope.
