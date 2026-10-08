"""A read-only web app: one entry per configured store, and each store's report.

**This is a surface over :mod:`tenbin`, not a second implementation of it.** Nothing here
measures anything, decides what a report may say, or renders a figure: a store is read by
:func:`tenbin.corpus.snapshot.take_snapshot`, measured by :mod:`tenbin.measures`, gated
by :mod:`tenbin.claims`, and rendered by :mod:`tenbin.report`. What this package adds is
a way to hold several stores side by side and to put the result where a browser can reach
it -- and the thing it adds carefully is *the list*, because a list is the shape that
hides a failure.

**A store that cannot be reached is a failure, and it survives into the list.** That is the
rule :mod:`tenbin.cli` and :mod:`tenbin.corpus.snapshot` already enforce for one store,
and it is harder in a list of five rather than easier: one row silently missing reads as a
store with nothing to say, and one row of zeroes reads as a healthy store with an empty
corpus. So it is structural rather than careful. :class:`~tenbin.webapp.read.StoreOutcome`
is either a :class:`~tenbin.webapp.read.StoreReport` or a
:class:`~tenbin.webapp.read.StoreFailure`, :func:`~tenbin.webapp.read.read_all` returns
one per configured store, and :func:`~tenbin.webapp.pages.index_page` renders all of them.
There is no value to render that is neither a report nor a failure, and no loop to forget
a branch in.

**No API key reaches a page.** Every store's key is held inside its
:class:`~tenbin.config.Settings` and stripped out of every string this package renders --
see :meth:`tenbin.webapp.config.StoreEntry.redact` -- and what a page shows in a store's
place is the operator's own name for it, not its URL. A credential in a served page is a
credential published to whoever can load the URL.

**The four modules, and the rule each one follows.** :mod:`tenbin.webapp.config` loads the
store list and holds the credentials; it is the only module that knows a key exists.
:mod:`tenbin.webapp.read` reads each store once and maps a failure onto a named sentence,
reusing :func:`tenbin.cli._walk`'s case for case.
:mod:`tenbin.webapp.pages` is pure -- outcomes in, HTML strings out, no clock and no
socket -- and is the whole of what a reader sees. :mod:`tenbin.webapp.server` is the
``http.server`` shell: three routes, a loopback default, and nothing else.

**What this package notably does not do:** it does not write a read model, cache a
snapshot, or refresh a kept read on the way past -- :mod:`tenbin.readmodel` is where that
decision belongs and it is a decision this app does not make. It does not retry a store
beyond what :class:`~tenbin.store.client.StoreClient` already does, does not authenticate
the people who load it, and does not compare two stores with one another: a comparison
needs an assignment mechanism (:mod:`tenbin.claims.mechanism`) and putting two stores on
one page is not a comparison, only the ability to look at both.
"""

from tenbin.webapp.config import (
    StoreEntry,
    StoreListError,
    first_line,
    load_stores,
)
from tenbin.webapp.pages import StoreSummary, index_page, report_page, summarize
from tenbin.webapp.read import (
    StoreFailure,
    StoreOutcome,
    StoreReport,
    read_all,
    read_store,
    was_read,
)
from tenbin.webapp.server import Response, TenbinHandler, TenbinServer, build_server, serve

__all__ = [
    "TenbinHandler",
    "TenbinServer",
    "Response",
    "StoreEntry",
    "StoreFailure",
    "StoreListError",
    "StoreOutcome",
    "StoreReport",
    "StoreSummary",
    "build_server",
    "first_line",
    "index_page",
    "load_stores",
    "read_all",
    "read_store",
    "report_page",
    "serve",
    "summarize",
    "was_read",
]
