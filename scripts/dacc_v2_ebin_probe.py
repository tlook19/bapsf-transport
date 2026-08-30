"""Measure the V2-vs-E_bin recompute agreement at ``kinetic_dvm.py`` ~:1207.

The cathode jet's launch-spectrum moment check forms its mean energy as
``0.5 * M_HE * (spec * self.g.V2).sum()`` while every other energy read in
the module folds the precomputed ``self.E_bin = 0.5 * M_HE * g.V2``
instead. The two differ only by WHERE the constant enters the sum -- scale
the sum, or sum the scaled -- so they must agree to roundoff. This measures
by how much, so the comment recording it quotes a MEASURED number rather
than an asserted one.

The grid is built exactly as ``TransientDVM.__init__`` builds it (the same
helper ``b5cj_t_launch_probe.py`` uses), and the launch band is the
production cathode-sheath band. Nothing here is a gate: it is the
measurement behind a cosmetic comment, and the computation at :1207 is
deliberately left alone.

Run from the worktree root with ``PYTHONPATH=<worktree>``.
"""

import os

import numpy as np

import cablp
from cablp.solvers._sim1d.physics.kinetic_neutrals import (
    EV,
    KB,
    M_HE,
    VGrid,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Velocity grids probed: the shipped ``(nvz, nvp)`` and its neighbours.
GRIDS = ((48, 12), (64, 24), (96, 24))

#: Launch energies [eV] spanning the production cathode-sheath band.
LAUNCHES = (100.0, 60.0, 32.0, 10.0, 2.0, 0.5)


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


def main():
    pkg = os.path.abspath(cablp.__file__)
    print(f"cablp.__file__ = {pkg}")
    assert pkg.startswith(ROOT + os.sep), "PYTHONPATH TRAP"
    print()
    print("V2-form vs E_bin-form mean energy of the cathode-jet launch")
    print("spectrum. Both read the SAME bin masses; they differ only in")
    print("where the 0.5*M_HE enters. Nothing here is a gate.")
    print("=" * 78)

    worst = 0.0
    for nvz, nvp in GRIDS:
        grid = build_grid(nvz, nvp)
        e_bin = 0.5 * M_HE * grid.V2
        print(f"grid {nvz}x{nvp}  (vmax {grid.vz.max():.4e} cm/s)")
        for e_ev in LAUNCHES:
            e_launch = e_ev * EV
            v_back = np.sqrt(2.0 * e_launch / M_HE)
            dv = bin_width_at(grid, v_back)
            t_launch = M_HE * dv * dv / EV
            s2 = t_launch * EV / M_HE
            u2 = v_back * v_back - 3.0 * s2
            if u2 <= 0.0:
                print(
                    f"  launch {e_ev:7.2f} eV  smear exceeds the launch "
                    "energy; no drift to solve for"
                )
                continue
            spec = grid.maxwellian(t_launch, np.sqrt(u2), exact_moments=True)
            as_v2 = 0.5 * M_HE * float((spec * grid.V2).sum())
            as_ebin = float((spec * e_bin).sum())
            rel = abs(as_v2 - as_ebin) / abs(as_ebin)
            worst = max(worst, rel)
            print(
                f"  launch {e_ev:7.2f} eV  V2-form {as_v2:.17e}  "
                f"E_bin-form {as_ebin:.17e}  rel {rel:.3e}"
            )
    print("=" * 78)
    print(f"WORST relative disagreement over the probe: {worst:.3e}")
    print(f"machine epsilon for float64:                {np.finfo(float).eps:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
