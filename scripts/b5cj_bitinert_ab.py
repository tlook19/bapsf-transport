"""Bit-inertness A/B for the B5 cathode-side energetic recycle, default OFF.

The B5 diff reaches three places the shipped arms walk: the transient DVM's
``update`` (the counted cathode recycle is now split before it is injected),
the solver's step-attempt and accept paths (a second stage accumulator and a
per-step surface debit), and ``rhs_terms`` (a side-channel incident-energy
row). All three are presence-gated on a channel that ships OFF, so none of
them may move a single bit of a run that does not arm it -- and that is what
this file measures, by capturing the same short run at the base commit and at
the candidate and comparing raw ``uint64`` digests.

Two arms, because the diff must be inert on two different paths:

* ``moment`` -- the shipped default and the stance the golden runs. No DVM
  object exists there, so this is the statement that the plumbing added to
  ``_attempt_step`` / ``_accept_step_attempt`` / ``rhs_terms`` costs the
  golden nothing.
* ``kinetic_dvm`` -- the arm the member is FOR, walked with the channel at
  its default (off). This is the statement that matters: the split, the
  accumulator and the debit are all reachable code on this path, and with
  the channel absent the trajectory, every DVM ledger row and the cathode
  surface energy ledger must be what they were.

Neither arm names any B5 configuration key, so this same file runs unchanged
at the base commit.

The state digest is a running SHA-256 over the packed state vector's raw
bytes after every accepted step, so a divergence anywhere in the window
reaches the final value rather than only the endpoint. The ``kinetic_dvm``
arm additionally digests every entry of ``TransientDVM.last_ledger`` -- both
the particle rows and the nested energy ones -- after every neutral tick, and
records the cathode surface energy ledger at the end of the window.

Usage (from the checkout root, PYTHONPATH set to it)::

    python scripts/b5cj_bitinert_ab.py --arm kinetic_dvm \
        --out scripts/b5cj_ab_kinetic_dvm_<label>.json
    python scripts/b5cj_bitinert_ab.py --compare A.json B.json
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

#: Steps each arm walks. A cost knob, not physics: the digest is running.
DEFAULT_STEPS = 400

#: The one departure from the golden pins on the ``moment`` arm: the step cap
#: IS the run length here, so reaching it must stop rather than raise.
MOMENT_PARAM_OVERRIDES = {"max_steps_action": "stop"}

#: The ``kinetic_dvm`` arm. Built from ``default_config()`` rather than the
#: stance file (the stance arms ``coverage_closure``, which the solver refuses
#: outside ``neutral_model = "moment"``), with the family's incompatible
#: defaults taken from the solver's own list so the fixture cannot drift from
#: what the arm actually refuses. A coarse mesh keeps it cheap; the shipped
#: velocity grid is kept, because the channel under test is the one that
#: projects a beam onto it.
DVM_PARAMS = {
    "nx": 24,
    "neutral_model": "kinetic_dvm",
    "neutral_kinetic_dvm_cadence_s": 2.5e-5,
    "max_steps_action": "stop",
}
DVM_FLAGS = {"neutral_two_zone": True}


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
    params.update(DVM_PARAMS)
    flags.update(DVM_FLAGS)
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


def _digest_ledger(running, ledger):
    """Fold one DVM ledger -- rows and nested energy rows -- into a digest."""
    for key in sorted(ledger):
        value = ledger[key]
        if isinstance(value, dict):
            _digest_ledger(running, value)
            continue
        running.update(key.encode())
        running.update(np.float64(value).tobytes())


def capture(arm, steps):
    """Return the running raw-uint64 digest of ``steps`` accepted steps."""
    params, flags = build_arm(arm)
    sim = LAPDSim1D(input_dict=params, input_flags=flags)
    running = hashlib.sha256()
    ledger_running = hashlib.sha256()
    ticks = 0
    taken = 0
    for _ in range(steps):
        step_once(sim)
        running.update(np.asarray(sim._y, dtype=np.float64).tobytes())
        dvm = getattr(sim, "_dvm", None)
        if dvm is not None and dvm.updates > ticks:
            ticks = dvm.updates
            _digest_ledger(ledger_running, dvm.last_ledger)
        taken += 1
    return {
        "arm": arm,
        "steps_requested": int(steps),
        "steps_taken": int(taken),
        "time_s": float(sim.time),
        "cells": int(sim.geometry.cells),
        "dvm_ticks": int(ticks),
        "digest_sha256": running.hexdigest(),
        "dvm_ledger_sha256": ledger_running.hexdigest(),
        "cathode_energy_ledger_J": {
            key: float(value)
            for key, value in sorted(sim._cathode_energy_ledger_J.items())
        },
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
        "arm", "steps_taken", "time_s", "cells", "dvm_ticks",
        "digest_sha256", "dvm_ledger_sha256", "cathode_energy_ledger_J",
        "final_state_sha256",
    ):
        match = a[key] == b[key]
        same = same and match
        print(f"{key:26s} {'==' if match else '!='}  {a[key]}  |  {b[key]}")
    print("BIT-IDENTICAL" if same else "DIVERGED")
    return 0 if same else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
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
    for key, value in sorted(record.items()):
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
