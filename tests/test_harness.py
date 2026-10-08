"""The harness is itself a claim, so it gets tested like one.

A test harness that has silently stopped isolating anything is the worst possible
failure in a project like this one, because every other test keeps passing. These
tests exist to make that failure loud, and they are the only tests in the suite
that assert about the infrastructure rather than about Tenbin.
"""

from __future__ import annotations

import os
import socket

import pytest

from tenbin.config import DEFAULT_COLLECTION, Settings, settings_from_env
from tests.conftest import BEHAVIOR_ENV_VARS

_LEAKED: str | None = None


def test_every_settings_field_is_on_the_isolation_list_because_a_field_that_is_not_is_a_field_a_shell_can_change() -> (
    None
):
    """Both models, checked against the list that is supposed to cover them.

    This list was three ``ticket_*`` variables short while ``TicketSettings`` existed,
    and nothing failed: every test still passed, because the variables were only ever
    read by commands no offline test reached. The suite cannot detect that kind of gap
    by running -- it can only detect it by comparing the fields to the list, which is
    what this does.

    Derived rather than transcribed, so a new field is covered the moment it is written
    and a field deleted from a model cannot linger here as a line protecting nothing.
    """
    from tenbin.config import TicketSettings

    prefix = Settings.model_config["env_prefix"]
    declared = {
        f"{prefix}{name}".upper()
        for model in (Settings, TicketSettings)
        for name in model.model_fields
    }

    assert declared <= set(BEHAVIOR_ENV_VARS), (
        "settings fields absent from BEHAVIOR_ENV_VARS, so a developer's shell could "
        f"change a test's outcome: {sorted(declared - set(BEHAVIOR_ENV_VARS))}"
    )


def test_the_autouse_fixture_removed_every_behaviour_variable() -> None:
    """No Tenbin setting may survive into a test from the developer's shell.

    Without this the suite is a function of whoever ran it, and a test that
    passes on one machine and fails on another teaches its author to re-run it
    until it goes green.
    """
    visible = [key for key in os.environ if key.startswith("TENBIN_")]
    # ``TENBIN_ENV_FILE`` is emptied rather than removed, which is the same
    # guarantee stated differently: a loader that reads it finds no file.
    assert visible == ["TENBIN_ENV_FILE"], f"behaviour variables leaked in: {visible}"
    assert os.environ["TENBIN_ENV_FILE"] == ""
    assert set(BEHAVIOR_ENV_VARS) >= set(visible)


def test_a_variable_set_by_one_test_does_not_survive_into_the_next() -> None:
    """Isolation is per-test, which is the only kind that helps.

    Set here, gone next. If this fails then a test that configures the store is
    leaking that configuration into whatever ran after it.
    """
    global _LEAKED  # noqa: PLW0603 - the leak is the thing under test
    os.environ["TENBIN_TANSEKI_URL"] = "https://leaked.test/v1"
    _LEAKED = os.environ.get("TENBIN_TANSEKI_URL")
    assert _LEAKED is not None


def test_the_previous_tests_leaked_nothing() -> None:
    """The other half of the pair: the set variable is gone here.

    Two tests rather than one, because a single test cannot observe its own
    cleanup -- it runs before the fixture teardown. The pair is what makes the
    claim checkable.
    """
    assert os.environ.get("TENBIN_TANSEKI_URL") is None


def test_integration_marks_keep_the_denial_without_an_opt_in() -> None:
    """The socket opens for a marked test with an opt-in, and for nothing else.

    All four cells, because each half alone is a different failure: a mark
    without an opt-in reaches a backend as a side effect of existing, an
    opt-in without a mark lets a unit test quietly depend on a daemon, and
    neither without the other opens anything. Asserted on the rule rather
    than by opening a socket, because a test of the guard must not need what
    the guard protects.
    """
    from tests.conftest import integration_network_allowed

    assert not integration_network_allowed({"integration"}, {})
    assert not integration_network_allowed(set(), {"TANSEKI_SEAM_STORE": "1"})
    assert not integration_network_allowed(set(), {"TICKET_SEAM_BACKEND": "1"})
    assert integration_network_allowed({"integration"}, {"TANSEKI_SEAM_STORE": "1"})
    assert integration_network_allowed({"integration"}, {"TICKET_SEAM_BACKEND": "1"})
    assert not integration_network_allowed({"integration"}, {"TANSEKI_SEAM_STORE": "  "})


def test_the_network_is_denied_because_a_silent_store_connection_proves_nothing() -> None:
    """A test that opens a socket must fail, and say why.

    The alternative is a suite that quietly grows a dependency on somebody
    else's daemon, and a measurement program whose tests measure the wrong
    subject is the one failure mode this project cannot recover from quietly.
    """
    with pytest.raises(RuntimeError, match="Unexpected network access in tests"):
        socket.create_connection(("localhost", 9))


def test_socket_connect_is_denied_too_because_create_connection_is_not_the_only_door() -> None:
    """``httpx`` reaches the network through ``socket.connect``, not through this.

    Closing one door and leaving the other open is the exact shape of defect this
    project exists to report, so the harness is held to the same standard. The
    socket is closed in a ``finally`` because a leaked handle is itself an error
    here, and a test that trips the harness must not then trip it a second way.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(RuntimeError, match="Unexpected network access in tests"):
            sock.connect(("localhost", 9))
    finally:
        sock.close()


def test_an_unconfigured_store_raises_rather_than_measuring_nothing() -> None:
    """No URL, no report -- because an empty corpus is not a finding.

    A measurement program that renders zero because it could not reach its
    subject has produced the most dangerous possible output: a real-looking
    number, with no claim attached, meaning nothing at all.
    """
    with pytest.raises(ValueError) as raised:
        settings_from_env()
    assert "tanseki_url" in str(raised.value)


def test_a_plain_http_url_to_a_remote_host_is_refused() -> None:
    """The API key and the corpus must not travel in the clear off the machine."""
    with pytest.raises(ValueError, match="must use HTTPS"):
        settings_from_env(tanseki_url="http://store.example.com/v1")


def test_loopback_is_allowed_over_plain_http() -> None:
    """A local daemon is the common case and is not a network boundary."""
    assert settings_from_env(tanseki_url="http://localhost:8088/v1").tanseki_url == (
        "http://localhost:8088/v1"
    )


def test_the_collection_default_is_a_placeholder_rather_than_a_plausible_name() -> None:
    """The store answers an unknown collection with 200 and no hits, not an error.

    So a shipped default that named a real collection would render a run that
    never configured one as a report over an empty corpus -- indistinguishable
    from a finding. The default has to look wrong in the output instead.
    """
    default = settings_from_env(tanseki_url="https://store.test/v1").tanseki_collection
    assert default == DEFAULT_COLLECTION
    assert DEFAULT_COLLECTION.upper() == DEFAULT_COLLECTION, (
        "an upper-case placeholder is visibly a placeholder in rendered output"
    )


def test_an_empty_collection_is_refused_rather_than_silently_meaning_everything() -> None:
    """No collection means no filter, which would silently read across all of them."""
    with pytest.raises(ValueError, match="collection"):
        settings_from_env(tanseki_url="https://store.test/v1", tanseki_collection="  ")


def test_credentials_never_appear_in_a_rejected_url() -> None:
    """An error that echoes the offending value hands the secret to a log.

    The store URL is validated in a place where the message may be printed, so a
    URL carrying a credential must be refused without being repeated.
    """
    with pytest.raises(ValueError) as raised:
        settings_from_env(tanseki_url="https://user:hunter2@store.test/v1")
    assert "hunter2" not in str(raised.value)


def test_the_settings_object_is_importable_and_requires_a_url() -> None:
    """The configuration object must exist before anything else can be built."""
    with pytest.raises(ValueError):
        Settings(tanseki_url="")
