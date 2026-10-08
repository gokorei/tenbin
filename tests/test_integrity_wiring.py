"""The integrity check has to be reachable from a report, not only callable.

**This module exists because the check was not wired in.** ``LifecycleShapeDetector``
was written, tested, and correct -- and no report ever called it, because nothing
in the registry asked for it. A corpus whose close path had never worked rendered as
a clean one, and the one check the integrity group exists to perform was the one
check that never ran.

The lesson is not "wire it in." It is that a correct function nothing calls is
indistinguishable from a feature nobody built, and the only thing that tells them
apart is a test that goes through the same path a reader does. So every test here
builds a report through ``build_report`` with the **default** measure list and
asserts on the rendered output. None of them calls the detector directly.
"""

from __future__ import annotations

from tenbin.claims.gate import ClaimGate
from tenbin.claims.registry import default_registry
from tenbin.corpus.snapshot import take_snapshot
from tenbin.measures import LifecycleShapeMeasure, default_measures
from tenbin.report.build import build_report
from tenbin.report.markdown import render_markdown
from tests.fakes import FakeStore, document_id

OPENED = "pr_state_change"
CLOSED = "action_closed"


def _transitions(opens: int, closes: int) -> dict[str, dict[str, str]]:
    """A corpus of lifecycle transitions, because the shape is the whole subject."""
    documents: dict[str, dict[str, str]] = {}
    for index in range(opens):
        documents[document_id("acme/app", 100 + index, f"open-{index:04d}")] = {
            "capture_source": "webhook",
            "repo": "acme/app",
            "pr": str(100 + index),
            "captured_at": "2026-09-01T00:00:00+00:00",
            "delivery_id": f"d{index}",
            "title": f"opened {index}",
            "author": "dev",
            "tags": [OPENED, "action_opened"],
        }
    for index in range(closes):
        documents[document_id("acme/app", 200 + index, f"close-{index:04d}")] = {
            "capture_source": "webhook",
            "repo": "acme/app",
            "pr": str(200 + index),
            "captured_at": "2026-09-02T00:00:00+00:00",
            "delivery_id": f"c{index}",
            "title": f"closed {index}",
            "author": "dev",
            "tags": [OPENED, CLOSED],
        }
    return documents


def _render(opens: int, closes: int) -> str:
    """Render a report the way a reader gets one, through the default measures."""
    snapshot = take_snapshot(FakeStore(documents=_transitions(opens, closes)), page_limit=500)
    return render_markdown(build_report(snapshot, gate=ClaimGate(), registry=default_registry()))


def test_a_corpus_that_opens_changes_and_records_no_closes_reports_it() -> None:
    """The finding reaches the report, because the detector is in the default registry.

    This is the test whose failure would have caught the wiring gap. It goes through
    ``build_report`` with no ``measures=`` argument on purpose: a test that passes its
    own list proves the list it passed, which is how an unwired check stays unwired
    while its own unit test stays green.
    """
    assert "Pull-request transitions open changes and record no closes" in _render(12, 0)


def test_a_corpus_with_closes_reports_that_the_shape_was_checked() -> None:
    """Silence is not a result, so a clean check says it ran.

    A reader who cannot tell "checked, and the shape is ordinary" from "nothing here"
    has been handed an absence where a fact belongs -- and the absence is exactly what
    an unwired check produces, so the two are indistinguishable unless the clean case
    renders something of its own.
    """
    rendered = _render(12, 5)
    assert "Pull-request transition shape" in rendered
    assert "Pull-request transitions open changes" not in rendered


def test_the_finding_arrives_before_the_measures_it_qualifies() -> None:
    """A finding about the corpus has to land before the numbers that assume it.

    Every other figure in the report is a rate over whatever this read found, and a
    reader who has already seen the descriptive measures has formed the impression the
    finding exists to correct.
    """
    headings = [line for line in _render(12, 0).splitlines() if line.startswith("## ")]
    finding = next(i for i, h in enumerate(headings) if "record no closes" in h)
    trust = next(i for i, h in enumerate(headings) if "review structure" in h)
    assert finding < trust


def test_the_finding_states_what_it_cannot_confirm() -> None:
    """A named cause Tenbin did not verify is a hypothesis, and says so on the page.

    The cause is real and the mechanism is documented upstream, but Tenbin read a
    store and did not execute Kojutsu's code. Rendering the cause without the
    limit would be this program doing the thing it exists to prevent, in the other
    direction: a confident answer about a system it never ran.
    """
    assert "cannot confirm" in _render(12, 0).lower()


def test_the_default_registry_runs_the_detector_as_a_measure() -> None:
    """The entry point resolves, so a caller that passes no measures still checks.

    ``build_report`` resolves the registry by name, so a registry that does not
    publish it fails with an ``AttributeError`` at report time rather than at import
    time. That is the right place to fail -- but only if something exercises it.
    """
    assert any(isinstance(measure, LifecycleShapeMeasure) for measure in default_measures())


def test_a_corpus_with_no_transitions_still_reports_the_check() -> None:
    """No transitions is not a passing check, and the two are not the same sentence.

    A corpus with nothing to look at is consistent with every cause the detector
    considers, including a collection filtered to exclude closes. Rendering the clean
    claim over zero would claim an ordinary shape where none was observed.
    """
    rendered = _render(0, 0)
    assert "Pull-request transition shape" in rendered
    assert "record no closes" not in rendered
