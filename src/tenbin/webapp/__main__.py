"""``python -m tenbin.webapp`` -- run the app.

A separate module rather than a ``__main__`` guard on :mod:`tenbin.webapp.server`,
because the package's ``__init__`` imports ``server`` -- so ``python -m
tenbin.webapp.server`` loads ``tenbin.webapp.server`` twice and Python says so with a
``RuntimeWarning`` about a module that was found in ``sys.modules`` before it was
executed. On a surface whose output a person reads, a warning printed above every
invocation is a bad trade for saving one file, and this one costs four lines.

The entry point is the same function either way: :func:`tenbin.webapp.server.main`
takes the store list as a positional argument and binds loopback unless ``--host`` says
otherwise.
"""

from __future__ import annotations

from tenbin.webapp.server import main

if __name__ == "__main__":
    main()
