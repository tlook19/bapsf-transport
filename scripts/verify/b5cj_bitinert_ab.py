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

**Reused by B4** (``b4aj_bitinert_ab.py``), which is a thin wrapper rather than
a fork: the logic here is channel-agnostic, so B4 supplies its own velocity
grid, cadence and (for the armed-sanity run) its own channel key through
``--extra`` / ``--extra-flag``, and the two members share one implementation of
the capture, the digest and the row comparison. The ANODE surface energy book
is captured beside the cathode one for the same reason the cathode one is: it
is a solver-side ledger the member can move. Both are read with ``getattr``
and default to an empty mapping, so this file still runs unchanged at any base
commit that has neither.

The state digest is a running SHA-256 over the packed state vector's raw
bytes after every accepted step, so a divergence anywhere in the window
reaches the final value rather than only the endpoint. It is the statement
that matters and it must be identical.

The ``kinetic_dvm`` arm additionally records every entry of
``TransientDVM.last_ledger`` -- the particle rows and the nested energy ones,
per tick, by name -- and the cathode surface energy ledger at the end of the
window. Those are compared ROW BY ROW rather than by one digest, because the
candidate's ledger legitimately carries rows the base has no name for: a new
channel is a new row, and the honest statement is that every row PRESENT IN
BOTH is bit-identical while every row present in only one is exactly zero
there. A single digest over the whole ledger cannot say that -- it would flag
an added zero row as a divergence -- so both are reported: the digest as a
fingerprint, the row comparison as the gate.

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
DEFAULT_STEPS = 800

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


def build_arm(arm, extra_params=None, extra_flags=None):
    """Return the ``(input_dict, input_flags)`` of one A/B arm.

    ``extra_params`` / ``extra_flags`` are layered LAST, so a caller can pin a
    different velocity grid or cadence, or arm the channel under test for a
    sanity run, without either arm's shared construction moving.
    """
    if arm == "moment":
        params, flags = build_baseline_config()
        params = dict(params)
        params.update(MOMENT_PARAM_OVERRIDES)
        flags = dict(flags)
    else:
        params, flags = default_config()
        params = dict(params)
        flags = dict(flags)
        for space, key, value, _why in KINETIC_DVM_INCOMPATIBLE_DEFAULTS:
            (flags if space == "flags" else params)[key] = value
        params.update(DVM_PARAMS)
        flags.update(DVM_FLAGS)
    params.update(extra_params or {})
    flags.update(extra_flags or {})
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


def _flatten_ledger(ledger, prefix=""):
    """Return one DVM ledger as a flat ``{row name: float}`` mapping."""
    flat = {}
    for key in sorted(ledger):
        value = ledger[key]
        if isinstance(value, dict):
            flat.update(_flatten_ledger(value, prefix=f"{prefix}{key}."))
            continue
        flat[f"{prefix}{key}"] = float(value)
    return flat


def _digest_ledger(running, flat):
    """Fold one flattened ledger into a running digest, names included."""
    for key in sorted(flat):
        running.update(key.encode())
        running.update(np.float64(flat[key]).tobytes())


def capture(arm, steps, extra_params=None, extra_flags=None):
    """Return the running raw-uint64 digest of ``steps`` accepted steps."""
    params, flags = build_arm(arm, extra_params, extra_flags)
    sim = LAPDSim1D(input_dict=params, input_flags=flags)
    running = hashlib.sha256()
    ledger_running = hashlib.sha256()
    ledgers = []
    ticks = 0
    taken = 0
    for _ in range(steps):
        step_once(sim)
        running.update(np.asarray(sim._y, dtype=np.float64).tobytes())
        dvm = getattr(sim, "_dvm", None)
        if dvm is not None and dvm.updates > ticks:
            ticks = dvm.updates
            flat = _flatten_ledger(dvm.last_ledger)
            ledgers.append(flat)
            _digest_ledger(ledger_running, flat)
        taken += 1
    return {
        "dvm_ledgers": ledgers,
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
        # Presence-gated on the B4 anode jet, and read with getattr so this
        # file still runs at a base commit that has no such book at all.
        "anode_energy_ledger_J": {
            key: float(value)
            for key, value in sorted(
                (getattr(sim, "_anode_energy_ledger_J", None) or {}).items()
            )
        },
        # Non-vacuity of the walk, so a "bit-identical" verdict cannot be a
        # statement about a window in which the arm never engaged or ticked.
        "dvm_engaged": bool(getattr(sim, "_dvm_engaged", False)),
        "final_state_sha256": hashlib.sha256(
            np.asarray(sim._y, dtype=np.float64).tobytes()
        ).hexdigest(),
    }


def _compare_rows(label, a_rows, b_rows):
    """Compare two ``{row: value}`` mappings; return (ok, one-line summary).

    Shared rows must be BIT-IDENTICAL. A row present on only one side is a
    row that side has a name for and the other does not -- a new channel --
    and it must be exactly zero there, or it is carrying something.
    """
    shared = sorted(set(a_rows) & set(b_rows))
    only_a = sorted(set(a_rows) - set(b_rows))
    only_b = sorted(set(b_rows) - set(a_rows))
    moved = [k for k in shared if a_rows[k] != b_rows[k]]
    nonzero = [k for k in only_a if a_rows[k] != 0.0]
    nonzero += [k for k in only_b if b_rows[k] != 0.0]
    ok = not moved and not nonzero
    summary = (
        f"{label}: {len(shared)} shared rows, {len(moved)} moved; "
        f"rows only on one side {only_a + only_b or 'none'}, of which "
        f"{len(nonzero)} carry a non-zero value"
    )
    if moved:
        summary += f"; MOVED {moved[:6]}"
    if nonzero:
        summary += f"; NON-ZERO {nonzero[:6]}"
    return ok, summary


def compare(a_path, b_path):
    """Print whether two captured arms are bit-identical; 0 when they are."""
    a = json.loads(Path(a_path).read_text())
    b = json.loads(Path(b_path).read_text())
    same = True
    for key in (
        "arm", "steps_taken", "time_s", "cells", "dvm_ticks", "dvm_engaged",
        "digest_sha256", "final_state_sha256",
    ):
        if key not in a or key not in b:
            continue
        match = a[key] == b[key]
        same = same and match
        print(f"{key:26s} {'==' if match else '!='}  {a[key]}  |  {b[key]}")
    # Fingerprint only: it folds ROW NAMES in, so an added zero row moves it.
    print(
        f"{'dvm_ledger_sha256':26s} "
        f"{'==' if a['dvm_ledger_sha256'] == b['dvm_ledger_sha256'] else '!='}"
        f"  {a['dvm_ledger_sha256']}  |  {b['dvm_ledger_sha256']}"
        "   (fingerprint; the row comparison below is the statement)"
    )
    for label, key in (
        ("cathode surface energy ledger", "cathode_energy_ledger_J"),
        ("anode surface energy ledger", "anode_energy_ledger_J"),
    ):
        ok, summary = _compare_rows(
            label, a.get(key, {}), b.get(key, {})
        )
        same = same and ok
        print(summary)
    if len(a["dvm_ledgers"]) != len(b["dvm_ledgers"]):
        same = False
        print(
            f"DVM ledgers: {len(a['dvm_ledgers'])} vs "
            f"{len(b['dvm_ledgers'])} ticks -- different windows"
        )
    for i, (rows_a, rows_b) in enumerate(
        zip(a["dvm_ledgers"], b["dvm_ledgers"]), start=1
    ):
        ok, summary = _compare_rows(f"DVM ledger tick {i}", rows_a, rows_b)
        same = same and ok
        print(summary)
    print("BIT-IDENTICAL" if same else "DIVERGED")
    return 0 if same else 1


def _parse_extra(items):
    """Return ``k=v`` strings as a dict, coercing the JSON-legible values."""
    out = {}
    for item in items or ():
        key, _, raw = item.partition("=")
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("moment", "kinetic_dvm"))
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--out")
    parser.add_argument("--extra", nargs="*", metavar="KEY=VALUE")
    parser.add_argument("--extra-flag", nargs="*", metavar="KEY=VALUE")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = parser.parse_args(argv)
    if args.compare:
        return compare(*args.compare)
    if not args.arm or not args.out:
        parser.error("--arm and --out are required unless --compare is given")
    record = capture(
        args.arm,
        args.steps,
        _parse_extra(args.extra),
        _parse_extra(args.extra_flag),
    )
    Path(args.out).write_text(json.dumps(record, indent=2, sort_keys=True))
    for key, value in sorted(record.items()):
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
