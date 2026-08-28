"""Bit-inertness A/B for the B1 counted boundary-inflow handshake.

The production golden is SUSPENDED BY DESIGN in the R3 window, so the usual
"``--verify`` prints exact=True" evidence is unavailable and a change that
must not move the trajectory has to be proven by a SELF-CONTROLLED A/B
instead: the same short run, digested at raw ``uint64``, captured once at the
base commit and once at the candidate.

Two arms, because the B1 diff touches two things that must not move:

* ``moment`` -- the shipped default and the stance the golden runs. Nothing
  the change adds is reachable here (there is no arm object to engage), so
  this arm is the statement that the plumbing is inert.
* ``kinetic`` -- the K4a steady arm, which is the OTHER consumer of
  ``_kinetic_channel_rates``. B1 factored that method's four source-channel
  expressions into ``_kinetic_source_channel_rows`` so the counted and the
  sampled path cannot drift; this arm is the statement that the extraction
  did not change a single bit of what the method returns.

``kinetic_dvm`` is deliberately NOT an arm here: that is the path the change
is FOR, and it is expected to move.

The digest is a running SHA-256 over the packed state vector's raw bytes after
every accepted step, so it is sensitive to any step in the window and not only
to the final state. Each arm also records its step count and physical time,
which move first when a trajectory diverges.

Alongside it each arm digests the FIVE ARRAYS ``_kinetic_channel_rates``
returns, at every accepted state of the window. That probe is a read: it
evaluates the boundary, reaction and anode terms on the accepted state and
writes nothing, and the state digest is bit-identical with and without it
(measured on the ``moment`` arm, 100 steps). It is what makes the ``moment``
arm say something about the refactor rather than only about the plumbing.

Usage::

    python scripts/dvm_b1_bitinert_ab.py --arm moment --steps 400 \
        --out scripts/dvm_b1_ab_moment_<label>.json
    python scripts/dvm_b1_bitinert_ab.py --compare A.json B.json
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from cablp.solvers._sim1d import LAPDSim1D, default_config
from cablp.solvers._sim1d.core.model_families import (
    KINETIC_DVM_INCOMPATIBLE_DEFAULTS,
)

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from baseline_sim1d import build_baseline_config  # noqa: E402

#: Steps each arm walks. A cost knob, not physics: the digest is a running
#: hash, so a divergence anywhere in the window shows up in the final value.
DEFAULT_STEPS = 400

#: The one departure from the golden pins on the ``moment`` arm: the step cap
#: IS the run length here, so reaching it must stop rather than raise.
MOMENT_PARAM_OVERRIDES = {"max_steps_action": "stop"}

#: The ``kinetic`` arm is built from ``default_config()`` rather than the
#: stance file: the stance arms ``coverage_closure``, which the solver refuses
#: outside ``neutral_model = "moment"``. A coarse mesh keeps the arm cheap --
#: the statement is about an expression, and the expression is per cell. The
#: cleared members are the package defaults a kinetic arm cannot carry, taken
#: from the solver's own list so this fixture cannot drift from what the arm
#: actually refuses.
KINETIC_PARAMS = {
    "nx": 24,
    "neutral_model": "kinetic",
    "max_steps_action": "stop",
}
KINETIC_FLAGS = {"neutral_two_zone": True}


def build_arm(arm):
    """Return the ``(input_dict, input_flags)`` of one A/B arm."""
    if arm == "moment":
        params, flags = build_baseline_config()
        params = dict(params)
        params.update(MOMENT_PARAM_OVERRIDES)
        return params, dict(flags)
    params, flags = default_config()
    params = dict(params)
    flags = dict(flags)
    for space, key, value, _why in KINETIC_DVM_INCOMPATIBLE_DEFAULTS:
        (flags if space == "flags" else params)[key] = value
    params.update(KINETIC_PARAMS)
    flags.update(KINETIC_FLAGS)
    return params, flags


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


def capture(arm, steps):
    """Return the running raw-uint64 digest of ``steps`` accepted steps.

    ``channel_rate_calls`` counts entries into ``_kinetic_channel_rates`` --
    the refactored method -- so a run that never reached it cannot be read as
    evidence about it. A zero on the ``kinetic`` arm means the window was too
    short for the arm to engage, not that the expression is unchanged.

    ``channel_rows_sha256`` is the direct statement about that method: the
    five channel arrays it returns, digested at raw ``uint64`` at every
    accepted state of the window. It is captured on both arms, so the
    ``moment`` arm -- where the method is otherwise never called -- still
    carries the evidence.
    """
    params, flags = build_arm(arm)
    sim = LAPDSim1D(input_dict=params, input_flags=flags)
    calls = 0
    channel_rates = type(sim)._kinetic_channel_rates

    def counted(self, state, derived, time):
        nonlocal calls
        calls += 1
        return channel_rates(self, state, derived, time)

    type(sim)._kinetic_channel_rates = counted
    running = hashlib.sha256()
    rows_running = hashlib.sha256()
    taken = 0
    try:
        for _ in range(steps):
            step_once(sim)
            running.update(np.asarray(sim._y, dtype=np.float64).tobytes())
            probe = channel_rates(sim, sim.state, sim.derived, sim.time)
            for key in sorted(probe):
                rows_running.update(
                    np.asarray(probe[key], dtype=np.float64).tobytes()
                )
            taken += 1
    finally:
        type(sim)._kinetic_channel_rates = channel_rates
    return {
        "arm": arm,
        "steps_requested": int(steps),
        "steps_taken": int(taken),
        "time_s": float(sim.time),
        "cells": int(sim.geometry.cells),
        "channel_rate_calls": int(calls),
        "digest_sha256": running.hexdigest(),
        "channel_rows_sha256": rows_running.hexdigest(),
        "final_state_sha256": hashlib.sha256(
            np.asarray(sim._y, dtype=np.float64).tobytes()
        ).hexdigest(),
    }


def compare(a_path, b_path):
    """Print whether two captured arms are bit-identical; 0 when they are."""
    a = json.loads(Path(a_path).read_text())
    b = json.loads(Path(b_path).read_text())
    same = True
    for key in (
        "arm", "steps_taken", "time_s", "cells", "channel_rate_calls",
        "digest_sha256", "channel_rows_sha256", "final_state_sha256",
    ):
        match = a[key] == b[key]
        same = same and match
        print(f"{key:20s} {'==' if match else '!='}  {a[key]}  |  {b[key]}")
    print("BIT-IDENTICAL" if same else "DIVERGED")
    return 0 if same else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("moment", "kinetic"))
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
    for key, value in sorted(record.items()):
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
