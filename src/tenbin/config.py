"""Configuration: one place that knows how to reach each source, and where it is.

Tenbin holds no capability over either source it reads, so the entire configuration
surface is the connection to them. Keeping it in one module is not tidiness -- it is the
reason the read seams are auditable. A measurement program that could reach a store some
other way, without the allowlist and the evidence handling in one named client, would be a
program whose inputs nobody could enumerate.

**Two models, because the two sources are configured independently and a command may need
only one of them.** :class:`Settings` is the knowledge store and requires ``tanseki_url``;
:class:`TicketSettings` is the ticket source and the declared period, and loads with
nothing configured at all. The split exists because ``tenbin retrospective`` measures ticket
flow and has no reason to require a knowledge store, and a command that told an operator to
set ``TENBIN_TANSEKI_URL`` in order to read a ticket source would be issuing a
correctly-shaped instruction about the wrong system.

The defaults are deliberately unhelpful, and unevenly so. ``tanseki_url`` has no default,
because a measurement program with a default store is a measurement program that will
happily report a real number about the wrong corpus the first time somebody runs it on a
machine where a store happens to be running. An unconfigured Tenbin raises rather than
reporting zero, and that distinction is the whole difference between a finding and a
failure. ``period_cadence`` has no default either, and for a sharper reason: a default
cadence would be invisible in every rendered report, scoping each figure to a boundary
nobody chose. ``ticket_source`` and ``ticket_api_url`` *are* empty by default,
because an unconfigured second source means one family of measures is
unavailable rather than that this program has nothing to say -- a different
kind of problem, refused by the command that needs it and in its own words.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: A placeholder, not a real collection name.
#:
#: A default is acceptable here in a way it is not for the URL, because a wrong
#: collection yields an empty corpus and an empty corpus is visible, whereas a
#: wrong URL yields another team's corpus and looks like a finding.
#:
#: **The placeholder is deliberately not a plausible name.** It used to be
#: ``chronicler-real``, which was the collection under the project's former name
#: and is not a collection the writer creates. Nothing caught the mismatch,
#: because Tanseki answers a search against an unknown collection with ``200``
#: and an empty hit list rather than an error -- so the wrong name rendered as a
#: clean report over zero documents, indistinguishable from a real finding.
#: ``CHANGE-ME`` is not a name anything could plausibly have, so a run that
#: forgot to set the collection says so in its own output instead of reporting an
#: empty corpus as though the corpus were empty.
DEFAULT_COLLECTION = "CHANGE-ME"

#: Every ``TENBIN_TICKET_SOURCE`` value that names an adapter, in the order the
#: refusal lists them. Defined here rather than beside the adapters because this
#: module sits below :mod:`tenbin.store` in the dependency graph -- the store
#: client already imports settings, so settings importing an adapter package
#: would be a cycle. One tuple rather than a copy per adapter module, and
#: ``test_the_registry_and_the_supported_names_agree`` builds every name
#: through the factory, so an adapter added without registering here fails
#: loudly instead of constructing but never configuring.
SUPPORTED_SOURCES: tuple[str, ...] = ("go-experiment", "jira", "linear", "youtrack")

#: Document ids are path-derived and contain ``/``; this bounds what a caller may
#: ask for so a malformed identifier cannot become an unbounded request.
MAX_DOCUMENT_ID_LENGTH = 500


def _validate_store_url(value: str) -> str:
    """Accept only an HTTP(S) URL, and require TLS unless it targets loopback.

    Plain HTTP to a remote host would put the API key and the corpus on the wire
    in the clear, so the exception for loopback is narrow and explicit: a local
    daemon is the common case and is not a network boundary.
    """
    normalized = value.strip().rstrip("/")
    parsed = urlparse(normalized)
    if (
        parsed.scheme.casefold() not in {"http", "https"}
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError("Tanseki URL must be a valid HTTP(S) URL without credentials.")
    if parsed.scheme.casefold() == "http":
        host = parsed.hostname.removeprefix("[").removesuffix("]").casefold()
        if host != "localhost" and host not in {"127.0.0.1", "::1"}:
            raise ValueError("Tanseki URL must use HTTPS unless it targets loopback.")
    return normalized


class Settings(BaseSettings):
    """Where the corpus is, read from the environment.

    Every field is a way of changing what Tenbin measures, so every field is
    listed in ``BEHAVIOR_ENV_VARS`` in ``tests/conftest.py`` and deleted before
    each test. A setting that changes behaviour and is not on that list is a
    setting that can make the suite depend on the machine it runs on.
    """

    model_config = SettingsConfigDict(
        env_prefix="TENBIN_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # A validation failure must not echo the value that failed. The store URL
        # is validated here and its rejection may be printed, so a URL that
        # carried a credential in it would hand that credential to every log that
        # caught the error. Kojutsu's allowlist refuses the same way, for the
        # same reason.
        hide_input_in_errors=True,
    )

    tanseki_url: str
    tanseki_api_key: str = ""
    tanseki_collection: str = DEFAULT_COLLECTION
    tanseki_timeout_seconds: float = Field(default=10.0, gt=0, le=120, allow_inf_nan=False)
    read_model_path: str = ""

    #: The ticket source and the declared period are **not** here. They live in
    #: :class:`TicketSettings`, and the reason is that this model requires ``tanseki_url``:
    #: a user with a ticket source and no knowledge store could not load this one, and
    #: would be told to configure a store in order to measure ticket flow. Two models over
    #: one ``TENBIN_`` prefix keeps each source's requirements to itself.

    @field_validator("tanseki_url")
    @classmethod
    def _check_tanseki_url(cls, value: str) -> str:
        return _validate_store_url(value)

    @field_validator("tanseki_collection")
    @classmethod
    def _check_collection(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Tanseki collection must not be empty.")
        return normalized


class TicketSettings(BaseSettings):
    """Where the ticket source is, which one it is, and which period a retrospective is cut to.

    **A separate model from :class:`Settings` because the two sources are configured
    independently and a retrospective needs only one of them.** Before this existed,
    ``tenbin retrospective`` read the whole knowledge-store configuration to reach three
    ticket fields and two period fields -- which meant a user with a ticket source and no
    knowledge store could not run a retrospective at all, and got told to set
    ``TENBIN_TANSEKI_URL`` to measure ticket flow. That is a real misconfiguration
    message: the URL is the right shape, it is simply about the wrong system.

    Same ``env_prefix``, same ``.env``, same refusal to echo a rejected value. Sharing the
    prefix is deliberate rather than convenient -- one prefix for the program is easier to
    remember than one per source -- and it is safe because the field names do not collide:
    ``ticket_*`` and ``period_*`` exist here and nowhere else.

    **The source is named, not inferred from the URL.** Three systems serve
    transitions behind three different wires, and no URL distinguishes them
    reliably -- a Jira site and a self-hosted tracker are both ``https`` with a
    path. So ``ticket_source`` names the adapter (``go-experiment``, ``jira``
    or ``linear``) and the URL only says where it lives. An adapter chosen
    without an address, or an address without an adapter, is unconfigured
    rather than half-configured, because a client built from half a source
    would read the wrong wire with the right credential.

    **The period fields are empty by default, and that is the point rather than an
    oversight.** There is a default collection in :class:`Settings` and there is
    deliberately no default cadence here. Nothing this program reads
    declares a sprint, so a defaulted cadence would produce a retrospective whose every
    figure was scoped to a boundary nobody chose and whose rendered output would not say
    so. Empty means "not declared", and :func:`tenbin.corpus.period.PeriodConfig.from_text`
    is what turns these two strings into arithmetic.

    Strings rather than a ``timedelta`` and a ``datetime`` because a period convention is
    something a person writes down ("every two weeks from the 5th"), and parsing it here
    would put the grammar in the settings model, where a typo produces a validation error
    about a field rather than a sentence about the convention.
    """

    model_config = SettingsConfigDict(
        env_prefix="TENBIN_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # The same reasoning as `Settings`: a rejected value may be printed, and the ticket
        # URL is the field most likely to have a credential typed into it by mistake.
        hide_input_in_errors=True,
    )

    ticket_source: str = ""
    ticket_api_url: str = ""
    ticket_api_token: str = ""
    ticket_api_user: str = ""
    ticket_timeout_seconds: float = Field(default=10.0, gt=0, le=120, allow_inf_nan=False)
    period_cadence: str = ""
    period_anchor: str = ""

    @field_validator("ticket_source")
    @classmethod
    def _check_ticket_source(cls, value: str) -> str:
        # Normalised rather than rejected on case, because "Jira" is a typo no
        # operator should have to retype -- and rejected outright when it names
        # no adapter, because an address without an adapter reads the wrong wire.
        normalized = value.strip().lower()
        if normalized and normalized not in SUPPORTED_SOURCES:
            raise ValueError(
                "Ticket source must name an adapter: "
                f"{', '.join(SUPPORTED_SOURCES)} (TENBIN_TICKET_SOURCE)."
            )
        return normalized

    @field_validator("ticket_api_url")
    @classmethod
    def _check_ticket_url(cls, value: str) -> str:
        # Empty is allowed and means unconfigured, exactly as in `Settings` -- see there
        # for why this is a validator rather than an optional field.
        if not value.strip():
            return ""
        return _validate_store_url(value)

    @model_validator(mode="after")
    def _check_source_requirements(self) -> TicketSettings:
        """Each adapter's credential, required exactly when that adapter is chosen.

        Per-adapter rather than per-field, because no single field is required
        by all four: go-experiment may run with auth off, Jira Cloud always
        wants an email plus an API token, and Linear and YouTrack want a token.
        A requirement stated on the field would refuse a valid go-experiment
        setup or accept a Jira one that can only fail with a 401.
        """
        if self.ticket_source == "jira":
            if not self.ticket_api_user.strip():
                raise ValueError(
                    "The jira adapter authenticates with an account email plus an API "
                    "token (TENBIN_TICKET_API_USER)."
                )
            if not self.ticket_api_token.strip():
                raise ValueError(
                    "The jira adapter authenticates with an account email plus an API "
                    "token (TENBIN_TICKET_API_TOKEN)."
                )
        if self.ticket_source in ("linear", "youtrack") and not self.ticket_api_token.strip():
            system = "linear" if self.ticket_source == "linear" else "youtrack"
            credential = "a personal API key" if system == "linear" else "a permanent token"
            raise ValueError(
                f"The {system} adapter authenticates every request with {credential} "
                "(TENBIN_TICKET_API_TOKEN)."
            )
        return self

    @property
    def ticket_source_configured(self) -> bool:
        """Whether a ticket source has been pointed at.

        Both halves, because an adapter without an address and an address
        without an adapter are the same state: nothing this program may read.
        A property for the reason :attr:`Settings.ticket_source_configured` is one: so a
        caller has to ask the question rather than infer it from a truthiness test it
        writes itself, and so it cannot be set to something disagreeing with the source.
        """
        return bool(self.ticket_source.strip()) and bool(self.ticket_api_url.strip())


def ticket_settings_from_env(**overrides: Any) -> TicketSettings:
    """Build the ticket-source configuration, letting a caller override for tests.

    The twin of :func:`settings_from_env`, and separate because the two sources have
    different requirements rather than different fields: this one loads with nothing
    configured at all, because "no ticket source" and "no period declared" are both
    ordinary states that a command reports in its own words rather than a validation
    failure the operator has to interpret.

    Overrides exist for the reason they do on the store side: a test must be able to point
    at a fake without editing the process environment, which the isolation fixture has
    already stripped.
    """
    return TicketSettings(**overrides)


def settings_from_env(**overrides: Any) -> Settings:
    """Build settings from the environment, letting a caller override for tests.

    Overrides exist so a test can construct a client without editing the
    process environment, which is the only way a test can point at a fake
    without the isolation fixture having to be undone first.
    """
    return Settings(**overrides)
