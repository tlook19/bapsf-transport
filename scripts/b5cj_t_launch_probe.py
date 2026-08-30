"""MEASURE the cathode jet's launch smear against the registered 0.18 eV pin.

The B5 registration names the launch smear as ``m dv_z(v_back)^2 / k_B``, "≈
0.18 eV at the named (64, 24) grid", and asks for that value as the config
default. Those are two different numbers and only one of them can be the
default, so this probe measures both BEFORE either is committed -- the B3
lesson, applied to a pin that arrives with the brief.

It touches nothing this branch changed: it builds a ``VGrid`` exactly as
``TransientDVM.__init__`` does and calls ``VGrid.maxwellian`` directly, so it
runs unchanged at the base commit and the numbers it prints are the base
commit's numbers.

What it reports, per grid and per launch energy:

* ``dv_z(v_back)``, the axial bin containing the launch speed, and the
  temperature ``m dv_z^2 / k_B`` it corresponds to -- the registration's own
  FORMULA, evaluated;
* for that grid-tied smear and for a fixed 0.18 eV smear, the relative miss
  of the projected spectrum's DISCRETE mean energy against the launch energy
  it is built to carry.

The miss is what decides whether the channel can be cross-booked at all: the
cathode surface is debited the launch energy per atom, and a projection that
misses it hands the gas a different number.

Usage (from the checkout root, PYTHONPATH set to it)::

    python scripts/b5cj_t_launch_probe.py
"""

import numpy as np

from cablp.solvers._sim1d.physics.kinetic_neutrals import (
    EV,
    KB,
    M_HE,
    VGrid,
)

#: Grids probed: the shipped default, the grid the registration names, and a
#: finer one so the trend across resolution is visible rather than inferred.
GRIDS = ((48, 12), (64, 24), (96, 24))

#: Launch energies per atom [eV]. The production band for this channel is the
#: tens of eV -- (R_E/R_N)(phi_c + Ti) at a discharge-class sheath -- and the
#: low end is included because the afterglow collapses towards it.
LAUNCH_EV = (100.0, 60.0, 32.0, 10.0, 2.0, 0.5, 0.1, 0.03)

#: The value the registration quotes for the grid-tied smear.
REGISTERED_T_LAUNCH_EV = 0.18

#: The agreement the cross-book needs. Same number the engine's guard uses.
MOMENT_REL_TOL = 1.0e-10


def build_grid(nvz, nvp, Ti_cap_eV=10.0, u_cap_cm_s=2.0e6, T_wall_K=300.0):
    """Return the velocity grid ``TransientDVM`` builds from these defaults."""
    vmax = 4.0 * np.sqrt(max(Ti_cap_eV, 0.5) * EV / M_HE) + 1.5 * u_cap_cm_s
    v_fine = 0.25 * np.sqrt(KB * T_wall_K / M_HE)
    return VGrid(vmax, vmax, nvz, nvp, v_fine)


def bin_width_at(grid, v):
    """Return the width [cm/s] of the axial bin containing ``v``."""
    edges = grid.vz_edges
    k = int(np.clip(np.searchsorted(edges, abs(v)) - 1, 0, grid.nvz - 1))
    return float(edges[k + 1] - edges[k])


def project(grid, e_launch_erg, T_launch_eV):
    """Return the relative miss of the projected discrete mean energy.

    ``None`` when the smear's own thermal content already exceeds the launch
    energy, which leaves no drift to solve for at all.
    """
    v_back = np.sqrt(2.0 * e_launch_erg / M_HE)
    s2 = T_launch_eV * EV / M_HE
    u2 = v_back * v_back - 3.0 * s2
    if u2 <= 0.0:
        return None
    spec = grid.maxwellian(T_launch_eV, np.sqrt(u2), exact_moments=True)
    got = 0.5 * M_HE * float((spec * grid.V2).sum())
    return abs(got - e_launch_erg) / e_launch_erg


def main():
    print("B5 cathode-jet launch smear: the registered pin, MEASURED")
    print("=" * 78)
    print(f"registered value: m dv_z(v_back)^2 / k_B "
          f"~ {REGISTERED_T_LAUNCH_EV} eV at the (64, 24) grid")
    print(f"agreement the cross-book needs: {MOMENT_REL_TOL:.0e} relative")
    print("=" * 78)
    holds = True
    for nvz, nvp in GRIDS:
        grid = build_grid(nvz, nvp)
        print(f"grid {nvz}x{nvp}  (vmax {grid.vz.max():.4e} cm/s)")
        for e_eV in LAUNCH_EV:
            e_erg = e_eV * EV
            v_back = np.sqrt(2.0 * e_erg / M_HE)
            dv = bin_width_at(grid, v_back)
            T_tied = M_HE * dv * dv / EV
            tied = project(grid, e_erg, T_tied)
            fixed = project(grid, e_erg, REGISTERED_T_LAUNCH_EV)
            if (nvz, nvp) == (64, 24) and fixed is not None:
                holds = holds and fixed <= MOMENT_REL_TOL
            print(
                f"  launch {e_eV:7.2f} eV  v_back {v_back:.4e}  "
                f"dv_z {dv:.4e}  grid-tied T {T_tied:9.4f} eV | "
                f"miss(grid-tied) "
                f"{'n/a    ' if tied is None else f'{tied:.3e}'}  "
                f"miss(0.18 eV) "
                f"{'n/a' if fixed is None else f'{fixed:.3e}'}"
            )
    print("=" * 78)
    print(
        "VERDICT: the registered 0.18 eV "
        + ("HOLDS" if holds else "DOES NOT HOLD")
        + " as a fixed smear at the (64, 24) grid over the production launch "
        "band. This is a MEASUREMENT of a pin the brief supplied, not a gate: "
        "the channel's own gates (CJ1-CJ3) are what it is gated on."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
