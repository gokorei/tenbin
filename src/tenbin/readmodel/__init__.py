"""The derived read model: a read of the store, kept whole, and rebuilt rather than updated.

This package exists because Tanseki has no aggregation, so every figure in this program
costs a full enumeration of the collection, and a report runs the same enumeration
once per measure it asks. The read model is the replacement for that cost: one walk,
kept, holding the records, the moment, the counts that say whether the walk reached
the end, and the classification every record was given.

**The claim this package makes is narrow, and the narrowness is the point.** A read
model is a *derived* artefact, and a derived artefact that anybody can amend is a
source of truth. So there is no update path here -- no add, no remove, no merge, no
refresh -- and :func:`~tenbin.readmodel.model.write` replaces the whole model or
nothing. The expensive operation is the rebuild, and paying it is the correct price
for a number that can be checked against the store by deleting a file. The
alternative is a model that answers questions about a corpus which no longer
exists, with figures that disagree with a live read for reasons no report could
state and no reader could check.

**Reading from a model is a performance choice and nothing else.**
:func:`~tenbin.readmodel.convert.to_snapshot` hands the read back in exactly the
shape a measure accepts, so a figure computed from a model is the *same* figure,
over the same population, with the same claim and the same completeness sentence.
There is no fast path that renders a different number, and that is what lets an
operator treat the model as a cache: delete it and the report is unchanged, which
is the only evidence that it was never anything else.

**A read is not always one walk, and the walk that finishes it is not published
here.** The seam caps a listing offset, so a collection that reaches the ceiling is
read in more than one window and the model records each of them in
:attr:`~tenbin.readmodel.model.ReadModel.windows`.
:mod:`tenbin.readmodel.resume` is the module that walks those windows, and it is
**deliberately not re-exported from this package**: it takes a store client, while
everything published here is a value transformation over a read somebody else took.
Adding it to :data:`__all__` would put the layer that reads the store on the face of
the package whose entire claim is that it does not, and the surface below is asserted
in ``tests/test_read_model.py`` precisely so that a new name has to be argued for
rather than added under a deadline. Import it from the module.

**What this package notably does not do:** it does not read the store, and it holds
no client -- :mod:`tenbin.cli` does the walking and hands the result to
:func:`~tenbin.readmodel.model.build`. It does not measure, gate, compare or
render, and it holds no opinion about which claims the corpus supports. It does not
answer "what changed since the last read", because a diff facility is the first
step towards an update path and the moment the model records is the whole of the
answer this package gives to staleness. And it does not treat a model it cannot
read as an error worth hiding: a version it does not implement is refused loudly
(:class:`~tenbin.readmodel.model.UnknownSchemaVersionError`) while a file it cannot
parse is reported as absent, because half a read model is worse than none.
"""

from __future__ import annotations

from tenbin.readmodel.convert import to_snapshot
from tenbin.readmodel.model import (
    SCHEMA_VERSION,
    ReadModel,
    StoredFailure,
    UnknownSchemaVersionError,
    build,
    load,
    write,
)

__all__ = (
    "SCHEMA_VERSION",
    "ReadModel",
    "StoredFailure",
    "UnknownSchemaVersionError",
    "build",
    "load",
    "to_snapshot",
    "write",
)
