"""Bit-inertness A/B for the B4 anode-side energetic recycle, default OFF.

A WRAPPER around ``b5cj_bitinert_ab.py``, not a fork: the capture, the running
raw-uint64 state digest, the per-tick DVM ledger capture and the row-by-row
comparison are that file's and are shared, because the statement B4 has to make
is the statement B5 made about a different channel. What this file supplies is
the B4 fixture -- the velocity grid and cadence the anode arm is specified at,
and the armed-sanity run that is not an A/B at all.

Three modes:

``--mode moment`` / ``--mode kinetic_dvm``
    Capture one A/B arm at the pinned window. Neither names any B4
    configuration key, so both run unchanged at the base commit and the
    comparison is base-vs-candidate. ``moment`` is the shipped default and the
    stance the golden runs (no DVM object exists there, so it is the statement
    that the plumbing costs the golden nothing); ``kinetic_dvm`` is the arm the
    member is FOR, walked with the channel at its default (off).

``--mode armed``
    The ARMED SANITY run: the same kinetic window with
    ``neutral_kinetic_dvm_anode_jet = True``. It is not compared against
    anything -- it is the statement that the channel constructs, ticks, and
    books, and it prints the birth totals and the two momentum rows.

``--compare A.json B.json``
    Delegates to the shared comparison.

The kinetic fixture DEVIATES from the brief's "KA1c-style" recipe in one
measured respect and it is recorded here rather than remembered: the base
layer is ``default_config()`` and NOT the committed ``g1atrim`` stance, for the
reason ``b5cj_bitinert_ab.py`` records -- the stance arms ``coverage_closure``,
which the solver refuses outside ``neutral_model = "moment"``, so a
stance-based kinetic arm does not construct at all. The pins that make the
fixture the anode arm's -- ``neutral_model = "kinetic_dvm"``, the 3.125e-6 s
cadence, the (64, 24) velocity grid, the family's incompatible defaults
(which include the whole fluid anode trio and the cathode trio) and the
800-step window -- are honoured exactly.

Usage (from the checkout root, PYTHONPATH set to it)::

    python scripts/b4aj_bitinert_ab.py --mode kinetic_dvm \
        --out scripts/b4aj_ab_kinetic_dvm_<label>.json
    python scripts/b4aj_bitinert_ab.py --compare A.json B.json
"""

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from b5cj_bitinert_ab import capture, compare  # noqa: E402

#: Steps each arm walks, pre-registered. A cost knob, not physics.
WINDOW_STEPS = 800

#: The B4 kinetic fixture's pins. The cadence and the velocity grid are the
#: anode arm's operating point; the ``moment`` arm takes none of them, because
#: it builds no DVM at all.
KINETIC_EXTRA = {
    "neutral_kinetic_dvm_cadence_s": 3.125e-6,
    "neutral_kinetic_dvm_nvz": 64,
    "neutral_kinetic_dvm_nvp": 24,
}

#: The one key the ARMED run adds on top of the kinetic fixture.
ARMED_EXTRA = {"neutral_kinetic_dvm_anode_jet": True}


def _armed_report(record):
    """Print the armed run's channel numbers, tick by tick and summed."""
    births = 0.0
    energy = 0.0
    print("-" * 78)
    print("ARMED SANITY -- the anode jet's own rows, per tick")
    for i, rows in enumerate(record["dvm_ledgers"], start=1):
        births += rows.get("birth_anode_jet", 0.0)
        energy += rows.get("energy.birth_anode_jet", 0.0)
        print(
            f"  tick {i}: birth_anode_jet {rows.get('birth_anode_jet', 0.0):.6e} "
            f"particles, {rows.get('energy.birth_anode_jet', 0.0):.6e} erg; "
            f"birth_anode (thermal) {rows.get('birth_anode', 0.0):.6e}; "
            f"momentum_anode_jet {rows.get('momentum_anode_jet', float('nan')):.6e} "
            f"g cm/s; momentum_mesh_absorbed "
            f"{rows.get('momentum_mesh_absorbed', float('nan')):.6e} g cm/s"
        )
    print(
        f"  window total: {births:.6e} particles, {energy:.6e} erg born on "
        f"the jet over {record['dvm_ticks']} ticks"
    )
    print(f"  anode surface energy book [J]: {record['anode_energy_ledger_J']}")
    print(f"  DVM engaged in the window: {record['dvm_engaged']}")
    print("-" * 78)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("moment", "kinetic_dvm", "armed")
    )
    parser.add_argument("--steps", type=int, default=WINDOW_STEPS)
    parser.add_argument("--out")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = parser.parse_args(argv)
    if args.compare:
        return compare(*args.compare)
    if not args.mode or not args.out:
        parser.error("--mode and --out are required unless --compare is given")
    if args.mode == "moment":
        record = capture("moment", args.steps)
    elif args.mode == "kinetic_dvm":
        record = capture("kinetic_dvm", args.steps, dict(KINETIC_EXTRA))
        record["arm"] = "kinetic_dvm"
    else:
        record = capture(
            "kinetic_dvm", args.steps, {**KINETIC_EXTRA, **ARMED_EXTRA}
        )
        record["arm"] = "kinetic_dvm_anode_jet_armed"
    Path(args.out).write_text(json.dumps(record, indent=2, sort_keys=True))
    for key, value in sorted(record.items()):
        if key == "dvm_ledgers":
            print(f"{key}: {len(value)} tick(s) captured")
            continue
        print(f"{key}: {value}")
    if args.mode == "armed":
        _armed_report(record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
