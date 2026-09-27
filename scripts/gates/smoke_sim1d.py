"""Assertion smoke suite for LAPDSim1D: the entry point.

    python scripts/gates/smoke_sim1d.py              # full suite, the gate
    python scripts/gates/smoke_sim1d.py --list       # case names, in order
    python scripts/gates/smoke_sim1d.py --only cathode-boundary-beam-terms
    python scripts/gates/smoke_sim1d.py --trace      # log each case name as it starts

The suite lives in the ``smoke`` package beside this file: the harness,
fixtures and command line in ``smoke/_harness.py``, the cases in one module
per subsystem, and the run order in ``smoke/__init__.py``. The exit status
is ``main``'s: 0 when every selected case passes.
"""

import sys
import warnings

from smoke import main

if __name__ == "__main__":
    # Python shows DeprecationWarnings raised from code in ``__main__`` and
    # hides them elsewhere. The cases used to run in ``__main__``; this filter
    # keeps showing the warnings attributed to them, now that they run in the
    # ``smoke`` package.
    warnings.filterwarnings(
        "default", category=DeprecationWarning, module=r"smoke(\.|$)"
    )
    sys.exit(main())
