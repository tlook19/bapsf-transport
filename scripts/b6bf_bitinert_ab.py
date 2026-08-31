"""Bit-inertness A/B for the B6 baffle interception channel, default OFF.

A WRAPPER around ``b5cj_bitinert_ab.py``, not a fork -- the same relationship
``b4aj_bitinert_ab.py`` has to it, and for the same reason: the capture, the
running raw-uint64 state digest, the per-tick DVM ledger capture and the
row-by-row comparison are that file's and are channel-agnostic. What this file
supplies is the B6 fixture and the armed-sanity run.

Three modes:

``--mode moment`` / ``--mode kinetic_dvm``
    Capture one A/B arm at the pinned window. Neither names any B6
    configuration key, so both run unchanged at the base commit and the
    comparison is base-vs-candidate. ``moment`` is the shipped default and the
    stance the golden runs (no DVM object exists there, so it is the statement
    that the plumbing costs the golden nothing); ``kinetic_dvm`` is the arm the
    member is FOR, walked with the channel at its default (off).

``--mode armed``
    The ARMED SANITY run: the same kinetic window with the fluid baffles and
    ``neutral_kinetic_dvm_baffles`` both on. It is not compared against
    anything -- it is the statement that the channel constructs, ticks and
    books, and it prints the baffle rows per tick.

``--compare A.json B.json``
    Delegates to the shared comparison.

TWO fixture facts are recorded here rather than remembered:

1. The kinetic base layer is ``default_config()`` and NOT the committed
   ``g1atrim`` stance, for the reason ``b5cj_bitinert_ab.py`` records -- the
   stance arms ``coverage_closure``, which the solver refuses outside
   ``neutral_model = "moment"``. The B4 pins are honoured exactly (the
   3.125e-6 s cadence, the (64, 24) velocity grid, the family's incompatible
   defaults, the 800-step window), so the two members' windows are comparable.
2. The armed run's BAFFLE POSITION is SNAPPED to the nearest interior mesh
   face of that 24-column geometry. The stance's 342.65 cm baffle is measured
   on the stance's own 280-cell mesh; ``core.geometry`` refuses a position
   further than half a cell from its nearest face, and a 24-column mesh has no
   face there. The stance's CLEAR RADIUS is used unchanged -- it is the
   measured CAD number and is what the channel is parameterized by -- and both
   the requested and the used position are printed, so the substitution is
   visible in the transcript rather than implicit in the fixture.

Usage (from the checkout root, PYTHONPATH set to it)::

    python scripts/b6bf_bitinert_ab.py --mode kinetic_dvm \
        --out scripts/b6bf_ab_kinetic_dvm_<label>.json
    python scripts/b6bf_bitinert_ab.py --compare A.json B.json
"""

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import numpy as np  # noqa: E402

from cablp.solvers._sim1d.core.geometry import build_geometry  # noqa: E402

from b5cj_bitinert_ab import (  # noqa: E402
    _parse_extra,
    build_arm,
    capture,
    compare,
)
from stance_config import load_stance  # noqa: E402

#: Steps each arm walks, pre-registered. A cost knob, not physics.
WINDOW_STEPS = 800

#: The B6 kinetic fixture's pins, taken from B4 unchanged so the two members'
#: A/B windows are the same window. The ``moment`` arm takes none of them,
#: because it builds no DVM at all.
KINETIC_EXTRA = {
    "neutral_kinetic_dvm_cadence_s": 3.125e-6,
    "neutral_kinetic_dvm_nvz": 64,
    "neutral_kinetic_dvm_nvp": 24,
}

#: The stance the armed run takes its baffle from.
PRODUCTION_STANCE = "g1atrim"

#: The rows the armed report reads, in the order it prints them.
BAFFLE_ROWS = (
    "loss_baffle_blocked",
    "birth_baffle_reemit",
    "energy.loss_baffle_blocked",
    "energy.birth_baffle_reemit",
    "energy.net_surface_baffle",
    "momentum_baffle_absorbed",
)


def armed_baffle_spec():
    """Return the armed run's baffle params, and what was substituted.

    Builds the kinetic fixture's own geometry with no baffles, then snaps the
    stance's baffle position to the nearest INTERIOR face of that mesh. Returns
    ``(params, note)``.
    """
    stance = load_stance(PRODUCTION_STANCE)
    requested = float(
        np.asarray(stance.params["neutral_baffle_positions_cm"], dtype=float)[0]
    )
    clear = float(
        np.asarray(
            stance.params["neutral_baffle_clear_radii_cm"], dtype=float
        )[0]
    )
    params, flags = build_arm("kinetic_dvm", KINETIC_EXTRA, {})
    geom = build_geometry(params, flags)
    z_edges = np.asarray(geom.z_edges_cm, dtype=float)
    forbidden = set(
        int(f) for f in np.asarray(geom.cathode_face_indices, dtype=int)
    )
    forbidden.update(
        int(f) for f in np.asarray(geom.anode_face_indices, dtype=int)
    )
    interior = [
        f for f in range(1, geom.cells) if f not in forbidden
    ]
    face = min(interior, key=lambda f: abs(z_edges[f] - requested))
    used = float(z_edges[face])
    note = (
        f"stance baffle requested at z = {requested} cm; this fixture's "
        f"{geom.cells}-cell mesh has no face there, so the position is "
        f"SNAPPED to face {face} at z = {used} cm (nearest interior face). "
        f"The stance CLEAR RADIUS {clear} cm is used unchanged."
    )
    return (
        {
            "neutral_baffle_positions_cm": [used],
            "neutral_baffle_clear_radii_cm": [clear],
        },
        note,
        face,
    )


def _armed_report(record):
    """Print the armed run's baffle rows, tick by tick and summed."""
    blocked = 0.0
    energy = 0.0
    first_live = None
    print("-" * 78)
    print("ARMED SANITY -- the baffle channel's own rows, per tick")
    print(f"  {record['baffle_note']}")
    for i, rows in enumerate(record["dvm_ledgers"], start=1):
        n = rows.get("loss_baffle_blocked", 0.0)
        blocked += n
        energy += rows.get("energy.loss_baffle_blocked", 0.0)
        if first_live is None and n > 0.0:
            first_live = i
        if i <= 4:
            print(
                f"  tick {i}: "
                + ", ".join(
                    f"{name} {rows.get(name, float('nan')):.6e}"
                    for name in BAFFLE_ROWS
                )
            )
    finite = all(
        np.isfinite(rows.get(name, 0.0))
        for rows in record["dvm_ledgers"]
        for name in BAFFLE_ROWS
    )
    paired = all(
        rows.get("loss_baffle_blocked", 0.0)
        == rows.get("birth_baffle_reemit", 0.0)
        for rows in record["dvm_ledgers"]
    )
    print(
        f"  window total: {blocked:.6e} particles intercepted, {energy:.6e} "
        f"erg, over {record['dvm_ticks']} ticks"
    )
    print(
        "  FIRST TICK WITH A NON-ZERO INTERCEPTION: "
        + ("none in this window" if first_live is None else str(first_live))
    )
    print(f"  every baffle row finite on every tick: {finite}")
    print(f"  blocked == re-emitted on every tick (particles): {paired}")
    print(f"  DVM engaged in the window: {record['dvm_engaged']}")
    print("-" * 78)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("moment", "kinetic_dvm", "armed"))
    parser.add_argument("--steps", type=int, default=WINDOW_STEPS)
    parser.add_argument("--out")
    parser.add_argument(
        "--extra",
        nargs="*",
        metavar="KEY=VALUE",
        help=(
            "params overrides layered on top of the fixture. The A/B arms are "
            "PRE-REGISTERED and must not be run with this; it exists for "
            "clearly-labelled supplementary runs of the armed mode."
        ),
    )
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = parser.parse_args(argv)
    if args.compare:
        return compare(*args.compare)
    if not args.mode or not args.out:
        parser.error("--mode and --out are required unless --compare is given")
    extra = _parse_extra(args.extra)
    if extra and args.mode != "armed":
        parser.error("--extra is accepted only with --mode armed")
    if args.mode == "moment":
        record = capture("moment", args.steps)
    elif args.mode == "kinetic_dvm":
        record = capture("kinetic_dvm", args.steps, dict(KINETIC_EXTRA))
        record["arm"] = "kinetic_dvm"
    else:
        baffle, note, face = armed_baffle_spec()
        record = capture(
            "kinetic_dvm",
            args.steps,
            {**KINETIC_EXTRA, **baffle, **extra},
            {"neutral_baffles": True, "neutral_kinetic_dvm_baffles": True},
        )
        record["arm"] = "kinetic_dvm_baffles_armed"
        record["baffle_note"] = note
        record["baffle_face"] = int(face)
        record["extra"] = dict(extra)
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
