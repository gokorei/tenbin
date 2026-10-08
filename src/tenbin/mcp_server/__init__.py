"""Tenbin over MCP: a read surface for a program that refuses to publish a bare number.

**The objection, recorded because it is the strongest argument against this package.**

There is a good case for giving Tenbin no MCP surface at all, and it is not a
hypothetical one. The corpus Kojutsu captures was written down for a local
knowledge system; nobody consented to it being summarised for third parties, and an
MCP server is a machine-readable invitation to summarise. An agent that can call
``report`` can call it in a loop, put the figures in a file, and quote them somewhere
the people who were written about will never see the caveats -- and every one of
those figures is *correct*, because they came from this program, which is exactly what
makes the outcome hard to catch in review. The command line has the same problem in
principle and much less of it in practice: a human has to be at a terminal, and a
report is a document with the caveats rendered above the number. An API removes both
frictions at once. Widening the reach of a corpus is a decision about other people's
data, and it deserves to be argued about rather than shipped with a docstring.

**The answer, which is that the choice is not between exposing the corpus and
silence.** What the objection protects is the *figures* -- the numbers, denormalised
out of the context that makes them legible. The refusal catalogue is not a summary of
the corpus; it is a description of what this program will not say, and it names no
record, no person, and no fact. It is also the part an agent most needs and cannot
get anywhere else. Kojutsu answers "what do you know"; nothing answers "what may I
ask", and an agent that cannot ask will infer that everything is askable, which is how
a comparison this corpus cannot support gets made by an agent three steps from a
number it can quote. So: the tool set is three tools, every one of them returns a
bound with its number, and the two that need no store exist so an agent can ask the
question before it has decided whether to have one.

**The precedent is the same argument, resolved the same way, by the neighbouring
project.** Kojutsu faced the identical objection about read events -- a log of who
read what is a record about people's reading, captured with no expectation of being
published -- and resolved it in ``src/kojutsu/core/read_log.py`` by making the log
a local file rather than a corpus record, and by refusing to give it an authenticated
caller because the stdio server has a single trust domain and cannot know who asked.
A read event is not knowledge; a refusal is not a measurement. Neither belongs in the
store, and neither is made public by being reachable. This package takes the same
shape: it holds no write capability at all, it reaches the corpus through the one
read-only client :mod:`tenbin.cli` already had, and its own read events -- there are
none; recording a read of a measurement program is a different question and is
deliberately left open -- would have been the same local, unauthenticated file.

**What is exported here is the table, not a server.** :mod:`tenbin.mcp_server.tools`
holds the design as data: three :class:`~tenbin.mcp_server.tools.ToolSpec`
descriptors and the pure functions that fill them in, importable with no ``mcp``
package installed so that every property argued about them is assertable without a
protocol, a socket, or a server.
:mod:`tenbin.mcp_server.server` is the adapter that walks that table and hands each
descriptor to the SDK, and it decides nothing: no tool, no wording, no schema lives
there. The whole surface is one frozen tuple, and
:data:`~tenbin.mcp_server.tools.TOOL_NAMES` is the name an agent's host will see.

**The bounds are in the tool descriptions, not only in the results.** Caveat-first
ordering was argued for a reader who goes top to bottom. An agent does not: it reads
a field and trusts its name, and a field named ``figure`` is a number. So
:data:`~tenbin.mcp_server.tools.SELF_SELECTION_BOUND` is written into every tool
description -- the part an agent reads before deciding to call -- and a
:func:`~tenbin.mcp_server.tools.refuse_unbound_numbers` check runs over every result
on its way out, so a tool that returned ``{"count": 412}`` would raise rather than
publish. Neither the check nor the prose is sufficient on its own, which is why both
exist: the description reaches a model that has not called anything yet, and the check
reaches a function that has already been written.
"""

from __future__ import annotations

from tenbin.mcp_server.tools import (
    BOUNDED_BY,
    CLAIM_FIELDS,
    EMPTY_IS_NOT_NOTHING,
    INSTRUCTIONS,
    READ_ONLY_HINTS,
    REFUSAL_FIELDS,
    REFUSAL_TOOL_BOUND,
    SELF_SELECTION_BOUND,
    TOOL_NAMES,
    TOOLS,
    TOOLS_BY_NAME,
    ToolSpec,
    call,
    claims_result,
    refusals_result,
    refuse_unbound_numbers,
    report_result,
    unbound_numbers,
)

__all__ = [
    "BOUNDED_BY",
    "CLAIM_FIELDS",
    "EMPTY_IS_NOT_NOTHING",
    "INSTRUCTIONS",
    "READ_ONLY_HINTS",
    "REFUSAL_FIELDS",
    "REFUSAL_TOOL_BOUND",
    "SELF_SELECTION_BOUND",
    "TOOLS",
    "TOOLS_BY_NAME",
    "TOOL_NAMES",
    "ToolSpec",
    "call",
    "claims_result",
    "refusals_result",
    "report_result",
    "refuse_unbound_numbers",
    "unbound_numbers",
]
