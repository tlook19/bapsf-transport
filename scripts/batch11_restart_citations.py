"""ASSERT every RESTART.md citation resolves against the tree.

RESTART.md cites the code that writes each piece of carried solver state. Those
citations used to be ``file.py:LINE`` anchors, which rot silently: any edit
above a cited line moves it, and the document then points confidently at the
wrong function. Measured before the recut, EVERY line-number citation in the
file was stale -- ``solver.py:3448``, cited as ``_accept_step_attempt``, had
become a line inside ``_vessel_ion_wall_current_A``.

The citations are now FUNCTION NAMES plus the owning file, which survive edits
above them. This script is the gate that keeps them honest, and it is an
ASSERTION, not an inspection tool: it exits non-zero on a dangling cite.

Two checks:

1. **No line-number citations.** Any surviving ``file.py:LINE`` is a
   regression to the rotting form and fails.
2. **Every cited function exists.** For each ``name`` cited alongside a
   ``file.py``, that file must define it (``def name`` / ``class name``).
   A renamed or deleted function fails here instead of misleading a reader.

Usage: python scripts/batch11_restart_citations.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

SIM1D = Path(__file__).resolve().parents[1] / "cablp" / "solvers" / "_sim1d"
DOC = SIM1D / "RESTART.md"

#: A ``file.py:LINE`` citation -- the form this document no longer uses.
LINE_CITE = re.compile(r"`([A-Za-z_][A-Za-z_0-9]*\.py):(\d+)(?:-(\d+))?`")
#: A backticked module filename.
FILE = re.compile(r"`([A-Za-z_][A-Za-z_0-9]*\.py)`")
#: A FUNCTION cite: one or more backticked names, optionally chained with
#: ``->``, immediately followed by the owning module in parentheses --
#: ``\`_set_state_vector\` (\`solver.py\`)``. Anchoring on that adjacency is
#: what separates a function cite from the STATE NAMES the same rows carry
#: (``_y``, ``_sample_ema``, ``IgnitionMonitor._samples``), which name
#: attributes rather than definitions and must not be looked up.
FUNC_CITE = re.compile(
    r"((?:`[A-Za-z_][A-Za-z_0-9.]*`(?:\(\))?\s*(?:->)?\s*)+)"
    r"(?:\(|in\s+)`([A-Za-z_][A-Za-z_0-9]*\.py)`"
)
#: The individual names inside a matched cite.
CITE_NAME = re.compile(r"`([A-Za-z_][A-Za-z_0-9.]*)`")


def _source(name):
    """Locate a cited module under the sim1d package."""
    for cand in (SIM1D / name, SIM1D / "core" / name, SIM1D / "results" / name,
                 SIM1D / "physics" / name):
        if cand.exists():
            return cand.read_text()
    return None


#: Backticked words that are language literals, not code the file defines.
LITERALS = {"None", "True", "False"}


def _defines(source, name):
    """True if ``source`` defines ``name``.

    A definition is a ``def``/``class`` statement or a module- or class-level
    ASSIGNMENT: RESTART.md legitimately cites constants such as
    ``_RESTART_TRIGGER_ATTRS`` alongside the functions.

    A dotted cite (``IgnitionMonitor.record``) is satisfied by its last segment
    being defined in the file; the owning class is checked separately when it is
    itself cited.
    """
    leaf = name.split(".")[-1]
    return re.search(
        rf"^\s*(?:(?:async\s+)?(?:def|class)\s+{re.escape(leaf)}\b"
        rf"|{re.escape(leaf)}\s*(?::[^=\n]+)?=)",
        source,
        re.M,
    ) is not None


def main():
    doc = DOC.read_text().splitlines()
    failures = []
    checked = 0
    sources = {}

    for n, text in enumerate(doc, start=1):
        for m in LINE_CITE.finditer(text):
            failures.append(
                f"RESTART.md:{n}: line-number citation `{m.group(1)}:"
                f"{m.group(2)}` -- cite the function name instead"
            )

        files = FILE.findall(text)
        if not files:
            continue
        for fname in set(files):
            if fname not in sources:
                sources[fname] = _source(fname)
            if sources[fname] is None:
                failures.append(
                    f"RESTART.md:{n}: cited module {fname} not found under "
                    f"{SIM1D}"
                )

        for names_blob, fname in FUNC_CITE.findall(text):
            source = sources.get(fname)
            if source is None:
                continue
            for raw in CITE_NAME.findall(names_blob):
                if raw in LITERALS:
                    continue
                checked += 1
                if not _defines(source, raw):
                    failures.append(
                        f"RESTART.md:{n}: cited name `{raw}` is not defined "
                        f"in {fname}"
                    )

    print(f"RESTART.md citation gate: {checked} function-name cites checked")
    if failures:
        print(f"FAIL: {len(failures)} dangling citation(s)")
        for line in failures:
            print(f"  {line}")
        return 1
    print("PASS: no line-number citations, every cited name resolves")
    return 0


if __name__ == "__main__":
    sys.exit(main())
