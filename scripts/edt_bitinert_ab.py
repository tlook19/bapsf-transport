"""Bit-inertness A/B for the electron drift-transport operator, default OFF.

The operator's diff reaches ``rhs_terms`` on every step: a new named term is
assembled there whether or not the flag is armed, because the term KEY is what
keeps the saved term structure stable across the pre-breakdown phase change
and across the flag. Unarmed it is the zero state and ``Gamma_d`` is never
evaluated -- and that is what this file measures, by capturing the same short
run at the base commit and at the candidate and comparing raw ``uint64``
digests and the term rows by name.

**A thin wrapper, not a fork.** The arms, the step driver and the row
comparison come from ``b5cj_bitinert_ab``, whose logic is channel-agnostic by
design; only the captured material differs, because the quantity at issue here
is the RHS TERM ROWS rather than a DVM ledger. Two arms, for the same reason
that file has two:

* ``moment`` -- the shipped default and the stance the golden runs.
* ``kinetic_dvm`` -- the other neutral route, walked with the operator at its
  default (off). The registration asks for both routes because the new term is
  assembled on both.

The term rows are compared ROW BY ROW rather than by one digest, because the
candidate legitimately carries a term the base has no name for. The honest
statement is that every term present in BOTH is bit-identical on every field
while every term present in only one is exactly zero there -- a single digest
over the term dict could not distinguish an added zero row from a divergence.
The running state digest is the statement that matters and must be identical.

This file names no configuration key the base commit lacks, so it runs
unchanged there.

Usage (from the checkout root, PYTHONPATH set to it)::

    python scripts/edt_bitinert_ab.py --arm moment --out scripts/edt_ab_moment_<label>.json
    python scripts/edt_bitinert_ab.py --compare A.json B.json
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from cablp.solvers._sim1d import LAPDSim1D

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from b5cj_bitinert_ab import (  # noqa: E402
    DEFAULT_STEPS,
    _compare_rows,
    build_arm,
    step_once,
)

#: The conservative fields a term row can carry. Optional rows read as None
#: and are recorded as absent rather than as zero, so "this term does not
#: write M_n" and "this term writes a zero M_n" stay distinguishable.
FIELDS = ("n", "nn", "M", "Ee", "Ei", "M_n", "nn_a", "M_n_a", "En")


def _term_rows(sim):
    """Return ``({term.field: |row| sum}, {term.field: raw-bytes sha256})``.

    Two mappings because the two questions are different, and neither answers
    the other. The SUM is what makes "present on one side only" checkable
    against zero -- a hash cannot say whether a row it hashed was zero. The
    SHA is what makes "present on both sides" a bit-identity statement rather
    than an agreement of two sums, which a compensating pair of changes would
    pass. Hashing rather than storing the rows keeps the capture small.
    """
    sums = {}
    shas = {}
    for name, term in sorted(sim.rhs_terms().items()):
        for field in FIELDS:
            value = getattr(term, field, None)
            if value is None:
                continue
            arr = np.ascontiguousarray(value, dtype=np.float64)
            key = f"{name}.{field}"
            sums[key] = float(np.abs(arr).sum())
            shas[key] = hashlib.sha256(arr.tobytes()).hexdigest()
    return sums, shas


def capture(arm, steps):
    """Return the running state digest and the final term rows of one arm."""
    params, flags = build_arm(arm)
    sim = LAPDSim1D(input_dict=params, input_flags=flags)
    running = hashlib.sha256()
    taken = 0
    for _ in range(steps):
        step_once(sim)
        running.update(np.asarray(sim._y, dtype=np.float64).tobytes())
        taken += 1
    sums, shas = _term_rows(sim)
    return {
        "arm": arm,
        "steps_requested": int(steps),
        "steps_taken": int(taken),
        "time_s": float(sim.time),
        "cells": int(sim.geometry.cells),
        "digest_sha256": running.hexdigest(),
        "final_state_sha256": hashlib.sha256(
            np.asarray(sim._y, dtype=np.float64).tobytes()
        ).hexdigest(),
        "term_rows": sums,
        "term_shas": shas,
        # Non-vacuity: a "bit-identical" verdict must not be a statement about
        # a window the run never got anywhere in.
        "term_count": len(set(sim.rhs_terms())),
        "nonzero_rows": int(sum(1 for v in sums.values() if v != 0.0)),
    }


def compare(a_path, b_path):
    """Print whether two captured arms are bit-identical; 0 when they are."""
    a = json.loads(Path(a_path).read_text())
    b = json.loads(Path(b_path).read_text())
    same = True
    for key in (
        "arm",
        "steps_taken",
        "time_s",
        "cells",
        "digest_sha256",
        "final_state_sha256",
    ):
        match = a[key] == b[key]
        same = same and match
        print(f"{key:22s} {'MATCH  ' if match else 'DIFFERS'} {a[key]} | {b[key]}")
    print(f"{'term_count':22s} {a['term_count']} | {b['term_count']}")
    print(f"{'nonzero_rows':22s} {a['nonzero_rows']} | {b['nonzero_rows']}")
    ok, summary = _compare_rows("term rows", a["term_rows"], b["term_rows"])
    print(summary)
    same = same and ok
    # Bit-identity on every row both sides carry. The sums above cannot make
    # that statement on their own: two rows can sum alike and differ.
    shared = sorted(set(a["term_shas"]) & set(b["term_shas"]))
    moved = [k for k in shared if a["term_shas"][k] != b["term_shas"][k]]
    print(
        f"term row bytes: {len(shared)} shared rows, {len(moved)} with a "
        f"changed sha256{'; MOVED ' + str(moved[:6]) if moved else ''}"
    )
    same = same and not moved
    one_sided = sorted(
        set(a["term_shas"]).symmetric_difference(b["term_shas"])
    )
    print(f"rows on one side only: {one_sided or 'none'}")
    print("BIT-IDENTICAL" if same else "DIVERGED")
    return 0 if same else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", choices=("moment", "kinetic_dvm"))
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--out")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = parser.parse_args(argv)
    if args.compare:
        return compare(*args.compare)
    if not args.arm or not args.out:
        parser.error("--arm and --out are required unless --compare is given")
    record = capture(args.arm, args.steps)
    Path(args.out).write_text(json.dumps(record, indent=2, sort_keys=True))
    print(
        f"{args.arm}: {record['steps_taken']} steps, t={record['time_s']:.6e} s, "
        f"{record['term_count']} terms, digest={record['digest_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
