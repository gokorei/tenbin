"""A read model as the snapshot a measure would have been handed, unchanged.

A measure is a pure function of a :class:`~tenbin.corpus.snapshot.CorpusSnapshot`
and of nothing else, which means the read model is worth having only if it can be
handed over in that shape. :func:`to_snapshot` is the whole of that promise and it
is deliberately narrow: it copies the model's records, its moment, its two counts,
its completeness verdict, its truncation and its failure into a snapshot, and it
computes nothing. There is no second reading of a record here, no re-derivation of
a month, no reconciliation of a count -- so a figure computed over the result is
the *same figure*, with the same claim, the same denominator and the same
completeness sentence, and not a figure that happens to look similar.

**That equality is what makes reading from the model a performance choice rather
than a semantic one.** The alternative is the one this project refuses everywhere
else: a fast path that renders a slightly different number from a slightly
different population, described as an optimisation. Because the two paths are
identical by construction, an operator can delete the model, rebuild it, or read the
store directly, and the report is the same document either way -- so the question
"should this run use the model?" is a question about latency and never about what
the reader will be told.

**The failure is restored as a failure, and that is the limit of the fidelity.** A
walk that stopped carries a :class:`StoreError`, which no file can hold; the model
stores its class name and message and this function builds the same class back, so
every figure's completeness sentence still names the failure that stopped the read.
What is gone is the exception object -- its traceback and its chain -- which no
rendered figure ever printed. A model holding a walk that never failed restores a
snapshot with no error, which is the correct answer rather than a tidy one: there
was no failure to report.

**What this module notably does not do:** it does not validate, reconcile, refresh
or complete anything. It does not compare the model against the store, so it cannot
tell a reader that the corpus has moved since the read -- the read's own
``read_at`` is the whole of the answer to that question, and it travels into the
snapshot unchanged. It does not recompute the indexes, which the model has already
built, and it does not hold the model: everything here is a value transformation, so
a caller can convert the same model twice and get two equal snapshots.
"""

from __future__ import annotations

from tenbin.corpus.snapshot import CorpusSnapshot
from tenbin.readmodel.model import ReadModel


def to_snapshot(model: ReadModel) -> CorpusSnapshot:
    """The read the model holds, as the snapshot a measure accepts.

    Every field of the snapshot is the model's own, including the ones that say the
    read was not whole. A truncated read stays truncated and a failed walk still
    carries its failure, because the whole argument of a read model is that it is
    the read -- and a model that dropped the completeness verdict on the way back
    would be exactly the "reports itself as a whole corpus" failure
    :mod:`tenbin.readmodel.model` refuses to allow.
    """
    return CorpusSnapshot(
        records=model.records,
        collection=model.collection,
        read_at=model.read_at,
        store_total=model.store_total,
        enumerated=model.enumerated,
        completeness=model.completeness,
        truncation=model.truncation,
        error=None if model.failure is None else model.failure.restore(),
    )
