"""Flag-on demonstration of the wall-branch momentum partition.

NOT a campaign arm and NOT a score. This runs a short two-moment two-zone
window with ``neutral_wall_momentum_partition`` armed and reports, at the
evolved state, the numbers gate 5 asks for:

  * the He--He mean free path, the annulus optical depth tau = (Rm-Rp)/mfp,
    and the cosine-averaged survival weight 2*E_3(tau) at the run's own
    annulus densities;
  * the momentum re-routed out of the wall branch, as a fraction of the
    wall-branch pool AND as a fraction of the total ion-neutral drag family.

The pre-registered expectation (drag relief 3-6 %, sink cut >= 20 % a red
flag) is adjudicated elsewhere; this script only reports.

The cross section is supplied on the command line -- the solver carries no
default. The script's own default is the repo's one existing boxed He-He
elastic value (``scripts/sp3_build_nn0.py``, hard-sphere LJ 2.551 Angstrom,
Hirschfelder/Curtiss/Bird), used for DEMONSTRATION ONLY.

Usage (from <checkout>/cablp, PYTHONPATH set to that same cablp):

    python scripts/ewp_demo_run.py [--t-end 2e-4] [--sigma-hehe-cm2 S]
"""

import argparse
import warnings

import numpy as np

from cablp.solvers._sim1d import LAPDSim1D, default_config
from cablp.solvers._sim1d.physics.sources import neutral_wall_partition_survival

DEMO_SIGMA_HEHE_CM2 = 2.044e-15


def build(sigma, nx):
    params, flags = default_config()
    params = dict(params)
    flags = dict(flags)
    params["neutral_momentum_radial"] = "kinetic_two_moment"
    params["nx"] = nx
    flags["neutral_two_zone"] = True
    flags["neutral_momentum"] = True
    # The kinetic_two_moment reduction gives the annulus its own momentum row
    # and nothing gives it an energy row, so neutral_energy must be off -- and
    # with it every flag that rides the hot channel neutral_energy builds.
    flags["neutral_energy"] = False
    flags["neutral_hot_internal_wall"] = False
    flags["neutral_hot_birth_drift"] = False
    off = LAPDSim1D(dict(params), dict(flags))
    params["neutral_wall_partition_sigma_hehe_cm2"] = sigma
    flags["neutral_wall_momentum_partition"] = True
    on = LAPDSim1D(dict(params), dict(flags))
    return off, on


def zone_volumes(geometry):
    Vc = np.asarray(geometry.plasma_volume_cm3, dtype=float)
    Va = np.maximum(
        np.asarray(geometry.neutral_volume_cm3, dtype=float) - Vc, 0.0
    )
    return Vc, Va


def report(sim, sigma, label):
    state = sim.state
    geom = sim.geometry
    Vc, Va = zone_volumes(geom)
    nn_a = np.asarray(state.nn_a, dtype=float)
    survival, tau, mfp = neutral_wall_partition_survival(geom, nn_a, sigma)

    terms = sim.rhs_terms()
    radial = terms["neutral_momentum_radial"]

    # The wall-branch pool: rebuild nu_wall from the closure's definition.
    from cablp.vars._cons import kb_cgs
    Rp = np.asarray(geom.Rp_cm, dtype=float)
    Rm = np.asarray(geom.Rm_cm, dtype=float)
    live = Va > 0.0
    vbar_n = np.sqrt(8.0 * 300.0 * kb_cgs / (np.pi * sim.ion_mass_g))
    nu_wall = np.where(
        live, vbar_n * Rm / (2.0 * np.maximum(Rm**2 - Rp**2, 1e-300)), 0.0
    )
    pool = nu_wall * np.asarray(state.M_n_a, dtype=float)
    absorbed = survival * pool
    retained = pool - absorbed

    pool_i = float(np.sum(np.abs(pool) * Va))
    retained_i = float(np.sum(np.abs(retained) * Va))

    # Total ion-neutral drag family: every ledger term that moves neutral
    # momentum by collisional exchange with the ions.
    drag_keys = [
        k for k in terms
        if ("drag" in k or "collision" in k or "cx" in k)
        and getattr(terms[k], "M_n", None) is not None
    ]
    drag_i = 0.0
    for k in drag_keys:
        t = terms[k]
        drag_i += float(np.sum(np.abs(np.asarray(t.M_n)) * Vc))
        if t.M_n_a is not None:
            drag_i += float(np.sum(np.abs(np.asarray(t.M_n_a)) * Va))

    live_tau = tau[live & (nn_a > 0.0)]
    live_s = survival[live & (nn_a > 0.0)]
    live_mfp = mfp[live & (nn_a > 0.0)]

    print(f"--- {label} ---")
    print(f"  t = {sim.time:.6e} s, cells = {geom.cells}")
    print(
        f"  annulus nn_a [cm^-3]: min {nn_a.min():.4e}  "
        f"median {np.median(nn_a):.4e}  max {nn_a.max():.4e}"
    )
    if live_tau.size:
        print(
            f"  He-He mfp [cm]:       min {live_mfp.min():.4g}  "
            f"median {np.median(live_mfp):.4g}  max {live_mfp.max():.4g}"
        )
        print(
            f"  optical depth tau:    min {live_tau.min():.6f}  "
            f"median {np.median(live_tau):.6f}  max {live_tau.max():.6f}"
        )
        print(
            f"  survival 2*E_3(tau):  min {live_s.min():.6f}  "
            f"median {np.median(live_s):.6f}  max {live_s.max():.6f}"
        )
    print(
        f"  wall-branch pool  |nu_wall*M_n_a| dV = {pool_i:.6e} dyn"
    )
    print(
        f"  re-routed to M_n_a                   = {retained_i:.6e} dyn "
        f"({100.0 * retained_i / max(pool_i, 1e-300):.3f}% of the wall pool)"
    )
    print(f"  drag-family terms: {drag_keys}")
    print(
        f"  drag family |dM| dV                  = {drag_i:.6e} dyn "
        f"-> re-routed is "
        f"{100.0 * retained_i / max(drag_i, 1e-300):.3f}% of it"
    )
    return dict(pool=pool_i, retained=retained_i, drag=drag_i)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--t-end", type=float, default=2.0e-4)
    ap.add_argument("--nx", type=int, default=60)
    ap.add_argument("--sigma-hehe-cm2", type=float, default=DEMO_SIGMA_HEHE_CM2)
    args = ap.parse_args()

    print("ewp_demo_run: wall-branch momentum partition, flag-on demonstration")
    print(f"  sigma_HeHe = {args.sigma_hehe_cm2:.6e} cm^2 (demonstration only)")
    print(f"  t_end = {args.t_end:.3e} s")
    print()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        off, on = build(args.sigma_hehe_cm2, args.nx)
        off.run(t_end=args.t_end)
        on.run(t_end=args.t_end)

    report(off, args.sigma_hehe_cm2, "flag OFF (reference ledger)")
    print()
    stats = report(on, args.sigma_hehe_cm2, "flag ON (partitioned wall branch)")

    # State-level divergence between the two arms, for scale.
    print()
    d_ua = np.asarray(on.state.M_n_a) - np.asarray(off.state.M_n_a)
    ua_off = np.asarray(off.state.M_n_a)
    print(
        f"  annulus momentum shift max|dM_n_a| = {np.max(np.abs(d_ua)):.6e} "
        f"on |M_n_a|max = {np.max(np.abs(ua_off)):.6e} "
        f"(rel {np.max(np.abs(d_ua)) / max(np.max(np.abs(ua_off)), 1e-300):.4f})"
    )
    du = np.asarray(on.state.M) - np.asarray(off.state.M)
    print(
        f"  ion momentum shift     max|dM|     = {np.max(np.abs(du)):.6e} "
        f"on |M|max = {np.max(np.abs(off.state.M)):.6e}"
    )
    ui_off = np.asarray(off.state.M) / (off.ion_mass_g * np.maximum(off.state.n, 1.0))
    ui_on = np.asarray(on.state.M) / (on.ion_mass_g * np.maximum(on.state.n, 1.0))
    dui = ui_on - ui_off
    print(
        f"  ion drift shift        p50 |du_i|  = "
        f"{np.median(np.abs(dui)) / 1.0e5:.6f} km/s, "
        f"max {np.max(np.abs(dui)) / 1.0e5:.6f} km/s"
    )
    return stats


if __name__ == "__main__":
    main()
