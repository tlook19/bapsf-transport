"""Bit-inertness A/B for the A2a anode tail cull, cull OFF and rider OFF.

A2a is a default-off flag build, so the statement it owes is that a run with
``beam_tail_anode_interception`` clear and the rider pair at 0.0 is the run the
base commit produced -- not close to it, the same floats. The production golden
is the fixture for that in general; this is the self-controlled A/B that
carries it on a short horizon, captured once at the base commit and once at the
candidate and compared byte for byte.

Two routes, because the flag family reaches configuration that both of them
resolve differently:

* ``default`` -- ``default_config()``. The anomalous tail is LOCAL here, so no
  walker exists for the cull to miss; this route is the statement that the
  plumbing (the config keys, the construction checks, the lag member, the
  ``J_anode`` term and the deposition module's new arguments) is inert on a
  configuration that never enters the tail walk at all.
* ``g1atrim`` -- the golden-at-stance config (``build_baseline_config``), whose
  tail IS walked, marched and reflected. This route is the statement that the
  cull-off branch of the walk itself is inert: every tail sub-branch runs, and
  every one of them takes the historical arithmetic.

The digest is a running SHA-256 over the packed state vector's raw bytes after
every accepted step, so it is sensitive to any step in the window rather than
only to the final state. Each route also records its step count, its physical
time and the lag member's value, which is 0.0 throughout an unarmed run and
would be the first thing to move if the coupling leaked.

Usage::

    python scripts/a2a_bitinert_ab.py --route g1atrim --steps 300 \\
        --out scripts/a2a_ab_g1atrim_<label>.json
    python scripts/a2a_bitinert_ab.py --compare A.json B.json
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from cablp.solvers._sim1d import LAPDSim1D, default_config

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from baseline_sim1d import build_baseline_config  # noqa: E402

#: Steps each route walks. A cost knob, not physics: the digest is a running
#: hash, so a divergence anywhere in the window reaches the final value.
DEFAULT_STEPS = 300

#: The one departure from the route's own pins: the step cap IS the run length
#: here, so reaching it must stop rather than raise.
STOP_AT_CAP = {"max_steps_action": "stop"}

#: The ``default`` route runs a coarse mesh -- the statement is about plumbing
#: that is per call, not per cell, and a fine mesh only buys wall time.
DEFAULT_ROUTE_PARAMS = {"nx": 32}


def build_route(route):
    """Return the ``(input_dict, input_flags)`` of one A/B route."""
    if route == "g1atrim":
        params, flags = build_baseline_config()
        params = dict(params)
        params.update(STOP_AT_CAP)
        return params, dict(flags)
    params, flags = default_config()
    params = dict(params)
    params.update(DEFAULT_ROUTE_PARAMS)
    params.update(STOP_AT_CAP)
    return params, dict(flags)


def step_once(sim):
    """Advance one step through the production step-acceptance path."""
    split = sim._flags.get("implicit_heat_conduction", False)
    diag = sim.suggest_timestep(include_heat_conduction=not split)

    def generate():
        attempt, retries, reason, events = sim._attempt_step_with_retries(
            dt=diag.dt, operator_split=None, diag=diag,
        )
        return attempt, (retries, reason, events)

    return sim._accept_step_with_picard(generate)


def capture(route, steps):
    """Return the running raw-uint64 digest of ``steps`` accepted steps."""
    params, flags = build_route(route)
    sim = LAPDSim1D(input_dict=params, input_flags=flags)
    running = hashlib.sha256()
    taken = 0
    for _ in range(steps):
        step_once(sim)
        running.update(np.asarray(sim._y, dtype=np.float64).tobytes())
        taken += 1
    # The lag member, read AFTER the window. On an unarmed run it is 0.0 at
    # every instant; a non-zero here would mean the coupling ran without the
    # flag, which is the failure this route exists to catch.
    lag = float(getattr(sim, "_cathode_tail_anode_I", 0.0))
    return {
        "route": route,
        "steps_requested": int(steps),
        "steps_taken": int(taken),
        "time_s": float(sim.time),
        "cells": int(sim.geometry.cells),
        "tail_anode_lag_A": lag,
        "digest_sha256": running.hexdigest(),
        "final_state_sha256": hashlib.sha256(
            np.asarray(sim._y, dtype=np.float64).tobytes()
        ).hexdigest(),
    }


def compare(a_path, b_path):
    """Print whether two captured routes are bit-identical; 0 when they are."""
    a = json.loads(Path(a_path).read_text())
    b = json.loads(Path(b_path).read_text())
    same = True
    for key in (
        "route", "steps_taken", "time_s", "cells", "tail_anode_lag_A",
        "digest_sha256", "final_state_sha256",
    ):
        match = a.get(key) == b.get(key)
        same = same and match
        print(f"{key:20s} {'==' if match else '!='}  {a.get(key)}  |  "
              f"{b.get(key)}")
    print("BIT-IDENTICAL" if same else "DIVERGED")
    return 0 if same else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", choices=("default", "g1atrim"))
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--out")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = parser.parse_args(argv)
    if args.compare:
        return compare(*args.compare)
    if not args.route or not args.out:
        parser.error("--route and --out are required unless --compare is given")
    import cablp
    print(f"cablp.__file__ = {cablp.__file__}")
    from cablp.cathode.kernels import KERNEL_ID
    print(f"KERNEL_ID      = {KERNEL_ID}")
    record = capture(args.route, args.steps)
    Path(args.out).write_text(json.dumps(record, indent=2, sort_keys=True))
    for key, value in sorted(record.items()):
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
