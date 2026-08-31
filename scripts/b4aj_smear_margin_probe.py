"""[B4] Measure the grid-tied launch smear's MARGIN across the launch band.

The number behind ``NUMERICS.md`` § "The DVM anode jet's launch spectrum" and
behind the AJ gates' choice of velocity grid. The anode jet places a
monoenergetic beam of energy ``e`` as a shifted Maxwellian whose drift is
solved from the energy, ``u^2 = v_back^2 - 3 k T_launch / m``, so it can only
be represented at all while

    margin(e) = e / ((3/2) k T_launch(e))  >  1,

and ``T_launch`` is grid-tied: the width of the axial bin containing the launch
speed. Whether the channel is safe over its whole reachable band is therefore
a property of the GRID, not of the operating point, and it is measured here
rather than assumed.

The ``v_z`` axis is geometrically stretched, so the bin widens with the speed
and the margin is roughly SCALE-FREE -- which is the reason the channel does
not need a launch-energy floor on the shipped grid, and the reason it cannot
run on the K2 suite's coarse default.

Reads only the grid; runs no solver, writes nothing.

Usage (from the checkout root, PYTHONPATH set to it)::

    python scripts/b4aj_smear_margin_probe.py
"""

import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import cablp  # noqa: E402
from cablp.solvers._sim1d.physics.kinetic_dvm import TransientDVM  # noqa: E402
from cablp.solvers._sim1d.physics.kinetic_neutrals import EV, M_HE  # noqa: E402

from verify_sim1d_k2_dvm import uniform_tube  # noqa: E402

#: Grids to report: the shipped one the AJ gates run, the arm's pinned one, and
#: the coarse suite default the channel deliberately cannot run on.
GRIDS = ((48, 12), (64, 24), (16, 6))

#: Launch energies [eV] spanning what the channel can reach: the afterglow's
#: thermal scale at the bottom, a production anode sheath in the middle, and a
#: cathode-class beam at the top.
LAUNCH_EV = (0.005, 0.017, 0.05, 0.2, 0.5, 1.7, 5.0, 20.0, 40.0, 100.0)


def main():
    print("[B4] grid-tied launch smear margin, e / ((3/2) k T_launch)")
    print("=" * 78)
    print(f"cablp.__file__ = {cablp.__file__}")
    print("=" * 78)
    for nvz, nvp in GRIDS:
        dvm = TransientDVM(
            geometry=uniform_tube(8, Rp=15.0, Rm=15.0),
            nvz=nvz,
            nvp=nvp,
            mesh_face=4,
            anode_jet={"R_N": 0.63, "R_E": 0.41, "T_launch_eV": None},
        )
        margins = []
        print(f"--- grid ({nvz}, {nvp}) ---")
        for e_eV in LAUNCH_EV:
            v_back = np.sqrt(2.0 * e_eV * EV / M_HE)
            T = dvm._anode_jet_launch_temperature_eV(v_back)
            margin = e_eV / (1.5 * T)
            margins.append(margin)
            print(
                f"  launch {e_eV:8.4g} eV   T_launch {T:10.5g} eV   "
                f"(3/2) k T {1.5 * T:10.5g} eV   margin {margin:8.3g}"
            )
        lo, hi = min(margins), max(margins)
        print(
            f"  MARGIN RANGE {lo:.3g} - {hi:.3g} over "
            f"{LAUNCH_EV[0]}-{LAUNCH_EV[-1]} eV: "
            + ("representable throughout" if lo > 1.0 else "NOT representable")
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
