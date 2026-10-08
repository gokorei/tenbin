"""The rule that has to survive into a list: an unreachable store is a failure, not a zero.

This file exists because the program already has this rule and this surface is where it
was most likely to be lost. :mod:`tenbin.cli` enforces it by exiting non-zero with
nothing written, and :func:`tenbin.corpus.snapshot.take_snapshot` enforces it by raising
on the first request. Both are easy, because each concerns exactly one store, and a
command either renders a report or renders nothing at all. **A list is harder.** One entry
silently missing reads as a store with nothing to say; one entry of zeroes reads as a
healthy store with an empty corpus. Both readings are wrong, and both are what a loop with
a missing branch produces -- so the assertions below are made against rendered bytes, and
the failure test is written to fail if the handling is deleted.

**No test here binds a socket.** Stores are handed over through the same
``CLIENT_FACTORY`` seam :mod:`tenbin.cli` uses, and the shell's routing is called
directly rather than through a client -- the whole of its behaviour is a path and a page,
so it can be asserted on without a port.

``pages.render_report_body`` is replaced for these tests because the report layer's own HTML
renderer has its own tests and is not what this file is about. The seam is one name in one
module, so nothing here reaches into :mod:`tenbin.report.html_`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from tenbin.config import settings_from_env
from tenbin.store.client import StoreAuthenticationError, StoreUnavailableError
from tenbin.webapp import pages, read, server
from tenbin.webapp.config import REDACTED, StoreEntry, StoreListError, load_stores
from tenbin.webapp.read import (
    StoreFailure,
    StoreListing,
    read_all,
    read_listings,
    read_store,
)
from tests.fakes import FAKE_COLLECTION, FakeOptions, FakeStore, document_id, kojutsu_frontmatter

#: A credential that would be unmistakable in a rendered page if it ever reached one.
#: Written out rather than generated so an assertion about it is an assertion about this
#: string rather than about a value that changed under it.
A_KEY = "sk-live-9f2c1d0b4e7a-not-a-real-key"

#: Two URLs, so a test can say which store is which in the store list rather than through a
#: collection name -- a collection is what is measured, and using it as an identity would
#: make these tests depend on the fake's default.
LOCAL_URL = "http://127.0.0.1:8099/v1"
DEAD_URL = "https://prod-tanseki.internal/v1"

#: The report body these tests substitute in. Deliberately trivial: a test here is about
#: what the page *around* the report says, and a real report would make every assertion
#: about the page an assertion about the report too.
STUB_REPORT_BODY = "<p>stub report body</p>"


@pytest.fixture(autouse=True)
def stub_report_renderer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand in for :func:`tenbin.report.html_.render_html`.

    Autouse because every test here renders a page and none is about the report inside it.
    The production path is untouched: nothing in :mod:`tenbin.webapp` assigns to this
    name, and the only caller resolves it from its module's globals on every call.
    """
    monkeypatch.setattr(pages, "render_report_body", lambda report: STUB_REPORT_BODY)


# -- stores under test ------------------------------------------------------------------------


def a_store(documents: Mapping[str, Mapping[str, Any]] | None = None, **options: Any) -> FakeStore:
    """A store that answers in process and never retries, so a test stays fast."""
    return FakeStore(documents, options=FakeOptions(max_retries=0, **options))


class UnreachableStore:
    """A client for a store that is not there.

    Raises the real :exc:`~tenbin.store.client.StoreUnavailableError` a dead Tanseki raises,
    so what is under test is the production failure mapping rather than a lookalike that
    happens to agree with it. Takes the settings it is handed, as
    :class:`~tenbin.store.client.StoreClient` does, rather than being pre-built with
    them -- the credential a message quotes has to be the one the entry actually holds.
    """

    def __init__(self, settings: Any) -> None:
        self.collection = getattr(settings, "tanseki_collection", "")
        self.settings = settings

    def __enter__(self) -> UnreachableStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def list_documents(self, *, limit: int = 500, offset: int = 0) -> Any:
        raise StoreUnavailableError("connect to the tanseki endpoint timed out")

    def count(self) -> int:
        """The index asks for a count, so an unreachable store has to fail at that too.

        Without this the fake answers a request the app never makes and raises
        :exc:`AttributeError` for one it does, which the failure mapping correctly
        reports as a defect in Tenbin rather than as an unreachable store -- true, and
        not what any of these tests are about. A store that is down fails every operation
        a caller might try, including the cheap one.
        """
        raise StoreUnavailableError("connect to the tanseki endpoint timed out")


class LeakyStore(UnreachableStore):
    """A store whose failure text quotes the credential, as a transport library might.

    Not hypothetical. A transport that decides one day to include a request header in its
    error message is exactly the event
    :meth:`tenbin.webapp.config.StoreEntry.redact` exists for, and it is cheaper to assert
    the guarantee against a message that really does carry the key than against a page
    that simply never rendered the key in the first place.
    """

    def list_documents(self, *, limit: int = 500, offset: int = 0) -> Any:
        key = getattr(self.settings, "tanseki_api_key", "")
        raise StoreUnavailableError(f"POST /v1/documents -> refused (sent key {key})")

    def count(self) -> int:
        """The leaky failure has to happen on the operation the index actually performs.

        Otherwise the credential is redacted from a report page that happens to render
        while the index -- which no longer walks -- never triggers the leak at all, and
        the test would pass for the wrong reason on the page that most deserves checking.
        """
        key = getattr(self.settings, "tanseki_api_key", "")
        raise StoreUnavailableError(f"GET /v1/documents -> refused (sent key {key})")


def an_entry(
    name: str, url: str, *, api_key: str = "", collection: str = FAKE_COLLECTION
) -> StoreEntry:
    """A configured store, pointed at whichever client the test wants for it."""
    return StoreEntry(
        name=name,
        settings=settings_from_env(
            tanseki_url=url, tanseki_api_key=api_key, tanseki_collection=collection
        ),
        source="stores.yaml",
    )


def corpus(count: int = 13) -> dict[str, dict[str, Any]]:
    """A small corpus over a window of ``count`` seconds.

    The timestamps differ because the window is one of the three fields an index entry
    states, and a corpus whose every record carries the same moment has a window of no time
    at all -- which would make the assertion about the window true for the wrong reason.
    """
    return {
        document_id("acme/widget", 1, f"answer-{index:04d}"): kojutsu_frontmatter(
            repo="acme/widget", pr=1, updated_at=f"2026-09-28T00:00:{index:02d}+00:00"
        )
        for index in range(count)
    }


def use(monkeypatch: pytest.MonkeyPatch, stores: Mapping[str, Any]) -> None:
    """Hand each configured store the client it should get, keyed by its URL.

    A value that is already a client is handed over as it is, so a :class:`FakeStore` is
    built once and answers every request. A value that is callable is called with the
    settings, which is how a test gives one store a failure whose text depends on the
    credential that store was actually configured with.
    """
    clients = {
        url: value if callable(value) else (lambda _settings, client=value: client)
        for url, value in stores.items()
    }
    monkeypatch.setattr(
        read, "CLIENT_FACTORY", lambda settings: clients[settings.tanseki_url](settings)
    )


def two_stores_one_down(monkeypatch: pytest.MonkeyPatch) -> tuple[StoreEntry, StoreEntry]:
    """One store that answers and one that cannot be reached. The case under test."""
    use(monkeypatch, {LOCAL_URL: a_store(corpus()), DEAD_URL: UnreachableStore})
    return an_entry("local-tanseki", LOCAL_URL), an_entry("prod-tanseki", DEAD_URL)


def two_healthy_stores(monkeypatch: pytest.MonkeyPatch) -> tuple[StoreEntry, StoreEntry]:
    """Two stores that both answer: the case where the page has nothing to apologise for."""
    other = "http://127.0.0.1:8089/v1"
    use(monkeypatch, {LOCAL_URL: a_store(corpus()), other: a_store(corpus())})
    return an_entry("local-tanseki", LOCAL_URL), an_entry("second-tanseki", other)


# -- the index --------------------------------------------------------------------------------


def test_the_index_lists_every_configured_store(monkeypatch: pytest.MonkeyPatch) -> None:
    """One entry per configured store, in configuration order, each with a link.

    The count line is not enough on its own: a page can say "2 of 2 stores were read" and
    still have rendered one of them, and the only way to tell is to look for each name.
    """
    page = pages.index_page(read_listings(two_stores_one_down(monkeypatch)))

    assert page.index("local-tanseki") < page.index("prod-tanseki"), "configuration order is kept"
    assert page.count('<li class="store') == 2
    assert '<a href="/store/local-tanseki">' in page


def test_one_unreachable_store_is_a_named_failure_and_the_others_still_render(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rule, in the shape that makes it hard: a list where one entry failed.

    Three things hold at once, and all three are asserted because a list view can fail any
    one of them while looking perfectly fine. **The failure is named**, so a reader can act
    on it. **The other store is still rendered**, because one dead store must not take the
    page down with it. And **the failed store is on the page at all**, because the failure
    mode this test exists for is the one where it silently is not.
    """
    page = pages.index_page(read_listings(two_stores_one_down(monkeypatch)))

    assert "local-tanseki" in page
    assert "The store reports 13 documents." in page
    assert "This page did not read them" in page

    assert "prod-tanseki" in page
    assert pages.FAILED_HEADLINE in page
    assert "could not be reached" in page
    assert "stores.yaml" in page, "the remedy names the file an operator would edit"

    assert page.count('<li class="store') == 2
    assert page.count('class="store failed"') == 1


def test_a_failed_store_is_not_rendered_with_figures_and_cannot_be_mistaken_for_an_empty_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed store shows no counts at all -- not zero counts, no counts.

    This is what makes the test above bite. A renderer that fell back to an empty report
    for a store it could not read would pass every other assertion in this file: the name
    would be there, the warning would be there, and the page would look like a deployment
    with one store that has nothing to say. What it would be *wrong* about is the one
    output this program most needs never to produce, so the assertions here are about the
    absence of numbers rather than the presence of a warning.
    """
    page = pages.index_page(read_listings(two_stores_one_down(monkeypatch)))
    failure_row = page.split('<li class="store failed">', 1)[1]

    assert "documents, read at" not in failure_row
    assert "0 of 0" not in failure_row
    assert "Records span" not in failure_row
    assert "It is never rendered as a store with no figures" in page


def test_a_failed_store_is_not_counted_among_the_stores_read_and_the_page_says_how_many_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The count line carries both numbers, because "4 of 4" and "4 of 5" are different.

    A page reporting only the successes would make those two the same sentence, and a
    deployment with a store nobody can reach would announce itself as four of four. So the
    line is not "2 of 2", and it does not stop at the successes.
    """
    page = pages.index_page(read_listings(two_stores_one_down(monkeypatch)))

    assert "1 of 2 configured stores answered a size request." in page
    assert "1 store could not be reached" in page
    assert "All 2 configured stores were read." not in page


def test_a_page_where_every_store_was_read_says_so_rather_than_staying_silent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The all-good case says all-good, because a failure count of zero is a fact too.

    Without this, the previous test could pass for the wrong reason: a page that always
    printed "N stores could not be reached" would satisfy it, and a page that printed the
    success count alone would look correct here and wrong in the other case.
    """
    use(monkeypatch, {LOCAL_URL: a_store(corpus(4))})
    page = pages.index_page(read_listings((an_entry("local-tanseki", LOCAL_URL),)))

    assert "All 1 configured stores answered a size request." in page
    assert 'class="store failed"' not in page
    assert pages.FAILED_HEADLINE not in page


def test_an_empty_configuration_says_it_is_empty_because_an_empty_list_is_not_a_list_of_empty_stores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No stores configured is a configuration problem, and the page says so.

    A heading above nothing reads as a deployment where every store has nothing to say,
    which is the same dangerous shape as the one this whole package refuses.
    """
    monkeypatch.setattr(
        read, "CLIENT_FACTORY", lambda settings: pytest.fail("there is no store to read")
    )

    page = pages.index_page(read_listings(()))

    assert "No stores are configured" in page
    assert "not a finding" in page
    assert '<li class="store' not in page


def test_no_page_references_an_external_asset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing a page loads is somewhere else, so a page cannot half-render.

    An external stylesheet or font is a second request on a page whose entire subject is
    whether a store is up -- and the failure would be a page with no styling and no hint
    why, which is a hard thing to diagnose from the page alone. Every ``href`` and ``src``
    is therefore a path on this app or absent.
    """
    rendered = read_all(two_stores_one_down(monkeypatch))
    index = pages.index_page(read_listings(two_stores_one_down(monkeypatch)))
    store = pages.report_page(rendered[0])

    for page in (index, store):
        assert "<link" not in page
        assert "<script" not in page
        references = re.findall(r'(?:href|src)="([^"]*)"', page)
        assert references, "the page should still link to its own routes"
        assert all(reference.startswith("/") for reference in references), references


# -- the report is in the response, not in an attribute -----------------------------------------


def _with_the_real_renderer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Undo the autouse stub, because these tests are about the real renderer.

    **The stub is the reason the escaping survived.** It is right for a test about the
    page around a report and catastrophic for a test about the report: every page here
    rendered a ``<p>stub report body</p>`` into an ``srcdoc`` attribute, so the
    assertion "the page contains the report" was satisfied by markup that never
    reached a reader. A test that cannot see the defect is not evidence.
    """
    monkeypatch.setattr(pages, "render_report_body", _real_render_report_body)


def _real_render_report_body(report: Any) -> str:
    from tenbin.report.html_ import render_html_body

    return render_html_body(report)


def test_the_served_report_page_carries_the_charts_as_markup_rather_than_escaped_into_an_attribute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """**This is the whole reason the ``srcdoc`` frame is gone.**

    The frame escaped a whole document into an attribute, so the response contained no
    chart at all -- ``<svg`` appeared only once a browser unescaped it. Against the
    real 51-document gokorei store that was a 204 KB response in which
    ``curl | grep svg`` found nothing.

    So the assertion is on the served bytes: the charts are there, unescaped, and the
    thing that hid them is not.
    """
    _with_the_real_renderer(monkeypatch)
    rendered = read_all(two_healthy_stores(monkeypatch))
    page = pages.report_page(rendered[0])

    assert "srcdoc" not in page
    assert "<iframe" not in page
    assert page.count("<svg") > 0, "the charts must be in the response itself"
    assert "&lt;svg" not in page, "the charts must not be markup inside an attribute"


def test_the_report_charts_are_styled_because_the_report_stylesheet_is_inlined(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The spliced body is not self-sufficient, so the page must bring the rules.

    Every class in the body is styled by the report layer's stylesheet. Inlining the
    body without it yields valid HTML that renders as plain text -- the reader would
    see every caveat and miss that the bars are absent. That is why
    ``report_stylesheet`` is exposed beside the body rather than folded into it, and
    this is the test that keeps the two from drifting apart.
    """
    from tenbin.report.html_ import report_stylesheet

    _with_the_real_renderer(monkeypatch)
    rendered = read_all(two_healthy_stores(monkeypatch))
    page = pages.report_page(rendered[0])

    first_rule = report_stylesheet().strip().splitlines()[0].strip()
    assert first_rule in page, "a rule the charts depend on must reach the page"
    assert page.count("<style>") == 1, (
        "one stylesheet, so the report rules are cascade-ordered against the page's "
        "by source order rather than by whichever element the browser met last"
    )


def test_the_spliced_report_keeps_the_page_self_contained_with_no_external_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Splicing must not have bought legibility with a network request.

    The frame was attractive partly because it kept the report's styles away from
    this page's. Inlining one stylesheet gives that up, and the guarantee it was
    carrying has to hold directly: the page still opens from ``file://`` with the
    network off.
    """
    _with_the_real_renderer(monkeypatch)
    rendered = read_all(two_healthy_stores(monkeypatch))
    page = pages.report_page(rendered[0])

    assert "<link" not in page
    assert "<script" not in page
    assert "http://" not in page
    assert "https://" not in page


# -- the per-store page -------------------------------------------------------------------------


def test_the_per_store_page_states_the_read_the_completeness_and_the_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The three fields, and the window is one of them.

    The window is the field most often left off a summary and the one that decides whether
    a corpus is a decade of quiet or an afternoon of it: a report over thirteen documents
    reads identically either way, and so does every rate computed from it.
    """
    page = pages.report_page(read_all(two_stores_one_down(monkeypatch))[0])

    assert "<dt>Read</dt>" in page
    assert "13 of 13 documents, read at" in page
    assert "<dt>Completeness</dt>" in page
    assert "the store was walked to its end and its own count agreed" in page
    assert "<dt>Window</dt>" in page
    assert "Records span 2026-09-28T00:00:00+00:00 to 2026-09-28T00:00:12+00:00" in page, (
        "the span, and not only that a window exists"
    )


def test_the_per_store_page_names_the_store_rather_than_printing_its_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The name is the operator's label; the URL can be the sensitive half.

    A host name says which infrastructure somebody runs, which is not this program's to
    publish to whoever loads the page, and it is not what a reader of a list came for.
    """
    page = pages.report_page(read_all(two_stores_one_down(monkeypatch))[0])

    assert "local-tanseki" in page
    assert "prod-tanseki.internal" not in page


def test_a_failed_stores_own_page_shows_the_failure_rather_than_an_empty_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dead store must not look different depending on which URL was loaded.

    An index saying "could not be reached" beside a link to a per-store page holding an
    empty report would reintroduce the defect through a second door: the reader who
    followed the link would be reading a page with no figures and no explanation.
    """
    page = pages.report_page(read_all(two_stores_one_down(monkeypatch))[1])

    assert pages.FAILED_HEADLINE in page
    assert "could not be reached" in page
    assert STUB_REPORT_BODY not in page
    assert "contributes no figures" in page


def test_an_unknown_store_name_is_a_404_naming_the_stores_that_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale bookmark deserves the list of what there is instead of a bare 404."""
    response = server.route("/store/not-a-store", two_stores_one_down(monkeypatch))

    assert response.status == 404
    assert "local-tanseki" in response.body
    assert "prod-tanseki" in response.body
    assert "not-a-store" in response.body


def test_a_query_string_reaches_no_page_because_a_store_name_is_never_a_url_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whatever arrives after the ``?`` is dropped before anything is rendered.

    Query strings end up in logs, in ``Referer`` headers and in browser history, so a value
    that could become a credential must never be read out of one -- and a link that
    appended one is not a thing this app has to defend against being written.
    """
    response = server.route(
        f"/store/local-tanseki?api_key={A_KEY}", two_stores_one_down(monkeypatch)
    )

    assert response.status == 200
    assert A_KEY not in response.body


# -- credentials ---------------------------------------------------------------------------------


def test_no_api_key_appears_anywhere_in_the_output_for_any_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rule, asserted on the key string itself rather than on a rendering of it.

    Every page for every configured store, with a key on the store that was read *and* on
    the store that failed -- because those are two different renderers, and a guarantee
    holding on the healthy path alone would be a guarantee about half the app.

    The failure store here is :class:`LeakyStore`, whose message really does carry the key,
    so this is not a page that happens not to render a credential: it is a page that had
    one and removed it.
    """
    use(monkeypatch, {LOCAL_URL: a_store(corpus()), DEAD_URL: LeakyStore})
    configured = (
        an_entry("local-tanseki", LOCAL_URL, api_key=A_KEY),
        an_entry("prod-tanseki", DEAD_URL, api_key=A_KEY),
    )

    rendered = read_all(configured)
    every_page = [pages.index_page(read_listings(configured))]
    every_page.extend(pages.report_page(store) for store in rendered)

    for page in every_page:
        assert A_KEY not in page, "an API key reached a served page"
    assert REDACTED in every_page[0], (
        "the key really did reach a rendered string and was removed from it; without this "
        "the assertions above would pass for the wrong reason"
    )


def test_a_store_with_no_api_key_still_works(monkeypatch: pytest.MonkeyPatch) -> None:
    """The local Tanseki on 8099 takes no credential, and it is the one anybody runs first.

    A redaction step that refused to pass text through when there was no key to remove
    would break exactly this store, which is why
    :meth:`tenbin.webapp.config.StoreEntry.redact` returns its input unchanged.
    """
    use(monkeypatch, {LOCAL_URL: a_store(corpus(3))})
    entry = an_entry("local-tanseki", LOCAL_URL, api_key="")

    assert entry.has_api_key is False
    page = pages.report_page(read_store(entry))

    assert "3 of 3 documents, read at" in page
    assert entry.redact("nothing to remove here") == "nothing to remove here"


def test_a_credential_quoted_by_a_transport_is_removed_from_the_failure_sentence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The message carries the key, the page does not, and the marker is visible.

    The marker rather than a blank: a cause reading "refused (sent key [redacted])" tells
    an operator the text was scrubbed, where an empty gap reads as a sentence that never
    had a key in it.
    """
    use(monkeypatch, {DEAD_URL: LeakyStore})
    entry = an_entry("prod-tanseki", DEAD_URL, api_key=A_KEY)

    outcome = read_store(entry)

    assert isinstance(outcome, StoreFailure)
    assert "sent key" in outcome.cause, "the transport really did quote the credential"
    page = pages.index_page((outcome,))
    assert A_KEY not in page
    assert "sent key [redacted]" in page


def test_a_stores_repr_carries_neither_its_url_nor_its_key() -> None:
    """A traceback prints the locals it unwinds through, and a local here holds a key.

    The generated dataclass repr would print every field of the settings, so it is
    overridden. This asserts the override rather than trusting the docstring, because a
    future ``@dataclass`` argument bringing the generated repr back would be invisible.
    """
    entry = an_entry("local-tanseki", LOCAL_URL, api_key=A_KEY)

    rendered = repr(entry)

    assert "local-tanseki" in rendered
    assert A_KEY not in rendered
    assert "prod-tanseki.internal" not in rendered


# -- the other failure cases ----------------------------------------------------------------------


def test_a_refused_credential_is_named_as_a_configuration_problem_because_retrying_will_not_help(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One of the four cases from :func:`tenbin.cli._walk`, mapped rather than reinvented.

    A store that refused the key is not a store that is down, and the remedy differs in
    kind: nothing about waiting changes an answer the store has already given.
    """

    def refused(settings: Any) -> Any:
        raise StoreAuthenticationError("GET /v1/documents -> 401")

    monkeypatch.setattr(read, "CLIENT_FACTORY", refused)
    entry = an_entry("prod-tanseki", DEAD_URL, api_key=A_KEY)

    outcome = read_store(entry)

    assert isinstance(outcome, StoreFailure)
    assert "refused the credential" in outcome.cause
    assert "retrying will not help" in outcome.remedy
    assert A_KEY not in outcome.cause
    assert A_KEY not in outcome.remedy


def test_a_store_that_answers_with_something_unusable_is_not_called_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken envelope is its own case, because the two have different remedies.

    This is :mod:`tenbin.cli`'s distinction: a reader told to wait for an envelope that
    will arrive broken waits for nothing.
    """
    use(monkeypatch, {DEAD_URL: a_store(corpus(2), omit_total=True)})

    outcome = read_store(an_entry("odd-tanseki", DEAD_URL))

    assert isinstance(outcome, StoreFailure)
    assert "could not be read" in outcome.cause
    assert "could not be reached" not in outcome.cause
    assert "no total" in outcome.cause


def test_one_store_raising_something_nobody_anticipated_does_not_take_the_page_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unexpected exception is a visible failure, not a missing store.

    Letting it propagate would hide the store that *was* read behind a traceback for the
    one that was not -- the failure this package exists to prevent, reached by a different
    road. The remedy says it is a defect, because that is what it is, and it says the
    failure is not a statement about the store's corpus.
    """

    def broken(settings: Any) -> Any:
        raise KeyError("a key nobody expected")

    monkeypatch.setattr(read, "CLIENT_FACTORY", broken)

    page = pages.index_page(read_listings((an_entry("local-tanseki", LOCAL_URL),)))

    assert "defect in Tenbin" in page
    assert "local-tanseki" in page
    assert "1 store could not be reached" in page
    assert "KeyError" in page


# -- the configuration file -------------------------------------------------------------------------


def _stores_yaml(*entries: str) -> str:
    """A store list on disk, as an operator writes one."""
    return "stores:\n" + "".join(f"  {entry}\n" for entry in entries)


@pytest.fixture
def stores_file(tmp_path: Path) -> Path:
    """Two configured stores, one local and one that names a variable for its key."""
    path = tmp_path / "stores.yaml"
    path.write_text(
        _stores_yaml(
            f"- name: local-tanseki\n    url: {LOCAL_URL}\n    collection: {FAKE_COLLECTION}",
            "- name: prod-tanseki\n    url: https://store.example/v1\n"
            "    collection: chronicler-real\n    api_key_env: TENBIN_TEST_TANSEKI_KEY",
        ),
        encoding="utf-8",
    )
    return path


def test_the_store_list_is_read_from_the_configuration_file_in_order(
    stores_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Names, URLs, collections, and a credential that is optional throughout.

    The local Tanseki carries no key at all, and that is the case which has to work: a
    configuration that demanded a credential would refuse the store a developer runs
    against, which is the first one anybody tries.
    """
    monkeypatch.setenv("TENBIN_TEST_TANSEKI_KEY", A_KEY)

    entries = load_stores(stores_file)

    assert [entry.name for entry in entries] == ["local-tanseki", "prod-tanseki"]
    assert entries[0].settings.tanseki_url == LOCAL_URL
    assert entries[0].settings.tanseki_api_key == ""
    assert entries[0].has_api_key is False
    assert entries[0].settings.tanseki_collection == FAKE_COLLECTION


def test_an_api_key_named_by_variable_is_read_from_the_environment(
    stores_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``api_key_env`` names a variable rather than holding a secret.

    A configuration file gets committed, so a file that can hold a credential is a file
    that will eventually have one committed to it.
    """
    monkeypatch.setenv("TENBIN_TEST_TANSEKI_KEY", A_KEY)

    entries = load_stores(stores_file)

    assert entries[1].settings.tanseki_api_key == A_KEY
    assert entries[1].has_api_key is True


def test_a_named_variable_that_is_unset_is_refused_because_a_typo_should_not_become_an_auth_failure(
    stores_file: Path,
) -> None:
    """An empty key is what "this store takes no credential" looks like from here.

    Treating a mistyped variable name as an absent key would surface an hour later as an
    authentication failure whose message says the credential is wrong, sending the
    operator to rotate a key that was never the problem.
    """
    with pytest.raises(StoreListError) as raised:
        load_stores(stores_file)

    assert "TENBIN_TEST_TANSEKI_KEY" in str(raised.value)


def test_two_stores_named_the_same_are_refused_because_a_name_is_how_a_store_is_addressed(
    tmp_path: Path,
) -> None:
    """The routing scheme is the name, so a duplicate is an unresolvable address.

    And it is the failure this app exists to make impossible, reached from the other
    direction: two entries under one label, one of them down, is a list no reader can
    interpret.
    """
    path = tmp_path / "stores.yaml"
    path.write_text(
        _stores_yaml(
            "- name: same\n    url: http://127.0.0.1:8099/v1",
            "- name: same\n    url: http://127.0.0.1:8098/v1",
        ),
        encoding="utf-8",
    )

    with pytest.raises(StoreListError, match="twice"):
        load_stores(path)


def test_a_store_name_that_cannot_be_addressed_is_refused(tmp_path: Path) -> None:
    """Whitespace or a slash in a name would make the URL address a different store."""
    path = tmp_path / "stores.yaml"
    path.write_text(
        _stores_yaml("- name: 'has a space'\n    url: http://127.0.0.1:8099/v1"),
        encoding="utf-8",
    )

    with pytest.raises(StoreListError, match="whitespace"):
        load_stores(path)


def test_an_unrecognised_key_is_refused_rather_than_ignored(tmp_path: Path) -> None:
    """``api_kye:`` would otherwise mean "this store needs no credential" for ever.

    A configuration file is written by a person, so a typo in it is a mistake about to
    become an authentication failure rather than a message about the typo.
    """
    path = tmp_path / "stores.yaml"
    path.write_text(
        _stores_yaml(
            "- name: local-tanseki\n    url: http://127.0.0.1:8099/v1\n    api_kye: sk-typo"
        ),
        encoding="utf-8",
    )

    with pytest.raises(StoreListError, match="api_kye"):
        load_stores(path)


def test_a_file_that_is_not_there_says_where_it_looked() -> None:
    """A missing file is a sentence naming the path, not a traceback about one."""
    with pytest.raises(StoreListError, match="nothing was listed"):
        load_stores(Path("/nonexistent/stores.yaml"))


# -- the shell -------------------------------------------------------------------------------------


def test_the_index_route_reads_every_store_and_renders_the_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``/`` is the index, and the index is over *all* the configured stores."""
    response = server.route("/", two_stores_one_down(monkeypatch))

    assert response.status == 200
    assert response.content_type == "text/html; charset=utf-8"
    assert "local-tanseki" in response.body
    assert "prod-tanseki" in response.body


def test_a_store_route_renders_that_stores_own_page(monkeypatch: pytest.MonkeyPatch) -> None:
    """``/store/<name>`` reads one store and reports it."""
    response = server.route("/store/prod-tanseki", two_stores_one_down(monkeypatch))

    assert response.status == 200
    assert pages.FAILED_HEADLINE in response.body


def test_a_path_this_app_does_not_serve_is_a_404_rather_than_a_listing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three routes, and the absence of a fourth is why the loopback default is safe."""
    response = server.route("/admin", two_stores_one_down(monkeypatch))

    assert response.status == 404
    assert "Not found" in response.body


def test_the_routes_default_to_loopback_because_this_app_has_no_authentication_of_its_own() -> None:
    """A default is a decision, and this one is about who can read a colleague's corpus.

    Bound to every interface, this would serve a measurement of somebody's corpus to
    anyone who could reach the host. Publishing it has to be something an operator says
    out loud with ``--host``.
    """
    assert server.DEFAULT_HOST == "127.0.0.1"


def test_the_route_reaches_stores_through_a_name_a_test_can_point_somewhere_else(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The seam that keeps the socket denial real, one layer above ``CLIENT_FACTORY``.

    The index route reads through :func:`tenbin.webapp.read.read_listings` resolved from
    the module's globals, so replacing it replaces whatever every request would have read.
    Asserted by handing it a listing of its own and looking for that count on the page: a
    route that built its own clients would render something else here and the test would
    fail with a network error instead, which is the failure mode the harness exists to
    make loud.

    The name is the seam, so it moved with the code: this used to replace ``read_store``
    and assert a report header was rendered, which is the assertion that stopped being
    true when the index stopped reading. The property being tested did not move with it.
    """
    entry = an_entry("local-tanseki", LOCAL_URL)
    monkeypatch.setattr(
        server, "read_listings", lambda configured: (StoreListing(entry=configured[0], count=2),)
    )

    response = server.route("/", (entry,))

    assert response.status == 200
    assert "The store reports 2 documents." in response.body


def test_a_page_where_every_store_was_read_does_not_claim_a_failure_is_listed_above_because_none_was(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """**The footer is the one sentence on the page that can state a falsehood.**

    It read "A store that could not be reached is listed above as a failure",
    unconditionally. That is true exactly when something failed, and on a page
    where every store was read it tells the reader a failure is listed above when
    the list above them is empty -- the missing-entry defect again, one layer down,
    and wearing the voice of a policy statement rather than a fact.

    So the policy is stated in the conditional on a clean page: it says what
    *would* happen. The claim about what did happen belongs to the counts
    sentence, which already carries both numbers.
    """
    page = pages.index_page(read_listings(two_healthy_stores(monkeypatch)))

    assert "All 2 configured stores answered a size request." in page
    assert "is listed above as a failure" not in page, (
        "the page asserts a failure is listed above when nothing failed"
    )
    assert "would be listed here as a failure" in page


def test_a_page_where_a_store_failed_still_says_it_is_listed_above_because_one_was(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The conditional must not have cost the failure case its sentence.

    Fixing the clean page by making the footer vaguer would trade a false claim
    for a missing one, which is the same trade in the other direction.
    """
    page = pages.index_page(read_listings(two_stores_one_down(monkeypatch)))

    assert "is listed above as a failure" in page


# -- the cheap index -------------------------------------------------------------------------------


class _CountingStore:
    """A store that answers a size request and refuses to be walked.

    Stands in for :class:`~tenbin.store.client.StoreClient` on the one operation the
    index performs. ``count`` is the only method it has, on purpose: asking this fake for
    anything else raises :exc:`AttributeError`, which is what makes "the index walked a
    store" a failure with a message rather than a slow test that happens to pass.
    """

    def __init__(self, count: int, calls: list[str], operation: str) -> None:
        self._count = count
        self._calls = calls
        self._operation = operation
        self.collection = FAKE_COLLECTION
        self.settings = None

    def __enter__(self) -> _CountingStore:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        return None

    def count(self) -> int:
        self._calls.append(self._operation)
        return self._count


def test_the_index_walks_nothing_and_only_asks_each_store_for_its_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """**The whole point, asserted on the call rather than on the page.**

    A test that only checked the index still renders would pass against the slow version,
    which is the entire defect: the page looked correct and cost N walks plus 22 measures
    per store. So the walk itself is patched to explode, and the count is recorded.

    ``read_store`` is the expensive door -- it walks and builds a report -- and
    ``take_snapshot`` is underneath it. Both are replaced with a failure rather than a
    spy, because a spy that is merely called would still let the page render and the test
    would pass on the thing it exists to forbid.
    """
    calls: list[str] = []

    def explode(*_args: object, **_kwargs: object) -> Any:
        raise AssertionError("the index walked a store")

    monkeypatch.setattr(read, "read_store", explode)
    monkeypatch.setattr(read, "take_snapshot", explode)
    monkeypatch.setattr(
        read, "CLIENT_FACTORY", lambda _settings: _CountingStore(13, calls, "count")
    )

    page = pages.index_page(read_listings((an_entry("local-tanseki", LOCAL_URL),)))

    assert "The store reports 13 documents." in page
    assert calls == ["count"], "the index asked for something other than a count"


def test_the_index_shows_the_stores_own_count_and_not_a_number_it_enumerated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The figure is the store's answer, and the wording says whose it is.

    A count obtained by walking is a count of whatever this program managed to enumerate,
    which above the offset cap is a prefix. Presenting that as a collection's size is the
    bounding error this project exists to refuse, so the number here is ``count()``'s and
    the sentence names it as the store's report rather than as something measured.
    """
    calls: list[str] = []
    monkeypatch.setattr(
        read, "CLIENT_FACTORY", lambda _settings: _CountingStore(10_000, calls, "count")
    )

    page = pages.index_page(read_listings((an_entry("local-tanseki", LOCAL_URL),)))

    assert "The store reports 10000 documents." in page
    assert "This page did not read them" in page


def test_an_unreachable_store_still_appears_in_the_cheap_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """**The property the optimisation is most likely to break.**

    Making a page cheap by skipping stores would turn an outage into a silently shorter
    list, and a shorter list reads as a deployment with fewer stores. So the unreachable
    one is present, named, with its cause -- and the reachable one is still there beside it.
    """
    page = pages.index_page(read_listings(two_stores_one_down(monkeypatch)))

    assert page.count('<li class="store') == 2
    assert page.count('class="store failed"') == 1
    assert pages.FAILED_HEADLINE in page
    assert "1 of 2 configured stores answered a size request." in page
    assert "1 store could not be reached" in page


def test_the_index_never_says_a_store_was_read_because_it_did_not_read_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ "Were read" would be false on every successful row.

    The counts sentence used to make exactly that claim, and it was true then. It stopped
    being true when the index stopped walking, and a stale sentence is worse than no
    sentence: it is the project's own recurring defect -- a figure stated about a read that
    did not happen -- reproduced in the one component whose job is to be honest about what
    it did.
    """
    for outcomes in (
        read_listings(two_healthy_stores(monkeypatch)),
        read_listings(two_stores_one_down(monkeypatch)),
    ):
        page = pages.index_page(outcomes)
        assert "were read" not in page
        assert "answered a size request" in page


# -- the framing policy -----------------------------------------------------------------------------


class _StubServer:
    """A ``TenbinServer`` that records its address instead of binding one.

    The suite denies the network and promotes a leaked handle to an error, so a test that
    proved what ``build_server`` does to a server would be testing the socket. This stands
    in for the real class so the contract -- a validated policy copied onto the server
    about to be returned -- is asserted without opening a port.
    """

    def __init__(self, address: object, handler: object) -> None:
        self.address = address
        self.entries: tuple[object, ...] = ()
        self.frame_ancestors = server.DEFAULT_FRAME_ANCESTORS


def test_a_response_carries_a_framing_policy_because_absent_one_permits_every_origin() -> None:
    """Absent ``frame-ancestors`` is not a neutral default; it is *every site may frame*.

    A report surface framed by a hostile page is a report surface whose reader cannot
    tell the content came from here. The permissive direction happened to work, which is
    exactly why it survived: nothing failed while it was in place.
    """
    headers = dict(server._headers(server.Response(200, "body")))

    assert headers["Content-Security-Policy"] == "frame-ancestors 'self'"


def test_the_framing_policy_defaults_to_self_because_that_is_what_a_self_hosted_app_needs() -> None:
    """A default is a decision about who may read a colleague's corpus.

    Same-origin framing covers the ordinary deployment -- this app behind the reverse proxy
    that also serves the page embedding it -- so it is the posture that needs no
    configuration and is still not the permissive one.
    """
    assert server.DEFAULT_FRAME_ANCESTORS == "'self'"
    assert server.TenbinServer.frame_ancestors == "'self'"


def test_no_x_frame_options_is_emitted_because_it_would_silently_defeat_a_configured_policy() -> (
    None
):
    """Two headers, and the operator would be wrong about which one won.

    ``X-Frame-Options`` is ``DENY`` or ``SAMEORIGIN`` only, so it cannot express the
    allowlist a named parent needs. Emitted alongside ``frame-ancestors`` the more
    restrictive of the two wins -- so the parent page is blocked and the operator's
    configuration is the thing that silently stops working.
    """
    names = {name for name, _ in server._headers(server.Response(200, "body"))}

    assert "X-Frame-Options" not in names


def test_a_configured_parent_is_named_in_the_policy_and_an_unconfigured_one_is_not() -> None:
    """A test that only checks the header exists would pass on the permissive default.

    The default and the configured value have to be distinguishable, or weakening the
    policy back to ``'self'`` -- or to nothing -- leaves every assertion green.
    """
    allowed = server._headers(
        server.Response(200, "body"), frame_ancestors="'self' https://a.example"
    )
    elsewhere = server._headers(
        server.Response(200, "body"), frame_ancestors="'self' https://b.example"
    )

    assert dict(allowed)["Content-Security-Policy"] == "frame-ancestors 'self' https://a.example"
    assert "https://b.example" not in dict(allowed)["Content-Security-Policy"]
    assert "https://a.example" not in dict(elsewhere)["Content-Security-Policy"]


def test_the_policy_is_present_on_a_report_response_and_not_only_on_the_index() -> None:
    """A framing policy on the landing page and not on the content page is the version
    of this defect that survives review.

    Both are produced by the same ``_emit``, so this is really a statement that the header
    is attached where the bytes are written rather than inside one branch of the route.
    """
    for response in (server.Response(200, "index"), server.Response(200, "report")):
        headers = dict(server._headers(response))
        assert "Content-Security-Policy" in headers


def test_a_policy_carrying_a_newline_is_refused_rather_than_sent_as_a_second_header() -> None:
    """This is the reason the value is validated instead of concatenated.

    ``frame-ancestors 'self'\\r\\nX-Evil: 1`` ends the header and starts another one, so
    an operator-supplied value that is not checked is a way to add headers to every
    response this app serves.
    """
    with pytest.raises(server.FrameAncestorsError, match="may not contain"):
        server.check_frame_ancestors("'self'\r\nX-Evil: 1")


def test_a_policy_carrying_a_semicolon_is_refused_because_it_would_start_a_second_directive() -> (
    None
):
    """A semicolon is how a framing allowlist becomes a ``script-src``."""
    with pytest.raises(server.FrameAncestorsError, match="may not contain"):
        server.check_frame_ancestors("'self'; script-src *")


def test_an_empty_policy_is_refused_rather_than_sending_a_header_that_says_nothing() -> None:
    """``frame-ancestors`` with no value is a directive no browser can act on."""
    with pytest.raises(server.FrameAncestorsError, match="empty"):
        server.check_frame_ancestors("")


def test_a_configured_policy_reaches_the_server_so_the_two_servers_in_a_process_can_disagree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The policy is held on the server for the reason the stores are.

    Two running servers in one process may legitimately have different parents; a module
    level global would make the second server's answer the first server's.
    """
    monkeypatch.setattr(server, "TenbinServer", _StubServer)

    built = server.build_server((), port=0, frame_ancestors="https://a.example")

    assert built.frame_ancestors == "https://a.example"


def test_build_server_refuses_a_policy_it_cannot_send_rather_than_binding_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A server that starts and then serves pages nobody may frame has already failed.

    The check is at bind time so an operator embedding this finds out before serving,
    not on the first request.
    """
    monkeypatch.setattr(server, "TenbinServer", _StubServer)

    with pytest.raises(server.FrameAncestorsError):
        server.build_server((), port=0, frame_ancestors="'self'; script-src *")


def test_a_flag_whose_policy_is_refused_stops_the_run_before_anything_binds(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A typo in a policy is a sentence at the command line, not a header found in a
    browser's network tab after the app has been deployed."""
    with pytest.raises(SystemExit) as stopped:
        server._arguments(["stores.yaml", "--frame-ancestors", "'self'; script-src *"])

    assert stopped.value.code == 2
    assert "--frame-ancestors" in capsys.readouterr().err


def test_the_flag_is_read_into_the_tuple_the_command_line_promise() -> None:
    """Three flags, and the return type is what stops the third being dropped."""
    positional, host, port, policy = server._arguments(
        ["stores.yaml", "--frame-ancestors", "'none'"]
    )

    assert positional == ["stores.yaml"]
    assert (host, port) == (server.DEFAULT_HOST, server.DEFAULT_PORT)
    assert policy == "'none'"
