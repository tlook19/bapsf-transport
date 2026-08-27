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

SHARED ALGEBRA. This module owns the one implementation of the wall-branch
rebuild used by every reader of this mechanism: :func:`zone_volumes`,
:func:`wall_branch_pool`, :func:`partition_split`, :func:`drag_family_keys`
and :func:`drag_family_integral`. ``scripts/ewp_ab_reader.py`` imports them
and runs them over SAVED trajectories rather than over a live solver, so the
construct-time demonstration and the saved-trajectory reader cannot drift
apart. Every one of them takes its rows through a ``get_row`` accessor so the
same code reads a live ``TermRHS`` object and a saved HDF5 term group. The
survival/optical-depth formulas are not restated anywhere: both callers use
``physics.sources.neutral_wall_partition_survival``, which is the in-solver
authority.

Usage (from <checkout>/cablp, PYTHONPATH set to that same cablp):

    python scripts/ewp_demo_run.py [--t-end 2e-4] [--sigma-hehe-cm2 S]
"""

import argparse
import warnings

import numpy as np

from cablp.solvers._sim1d import LAPDSim1D, default_config
from cablp.solvers._sim1d.physics.sources import neutral_wall_partition_survival
from cablp.vars._cons import kb_cgs

DEMO_SIGMA_HEHE_CM2 = 2.044e-15

#: Neutral temperature the free-molecular wall-visit rate is built at [K].
#: The two-zone closure fixes the annulus crossing time at the 300-K
#: free-molecular speed (``core/config.py`` ``neutral_momentum_radial``), so
#: the wall branch is not a function of the evolved neutral energy.
WALL_BRANCH_T_K = 300.0

#: Name test for the ion-neutral drag family: every ledger term that moves
#: neutral momentum by collisional exchange with the ions.
_DRAG_NAME_PARTS = ("drag", "collision", "cx")


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


def attr_row(term, field):
    """Read one ledger row off a live ``TermRHS``; ``None`` when unbooked."""
    return getattr(term, field, None)


def wall_branch_pool(geometry, M_n_a, ion_mass_g):
    """Return ``(nu_wall, pool)`` for the two-zone wall branch.

    ``nu_wall`` is rebuilt from the closure's own definition -- the
    free-molecular annulus crossing rate ``vbar_n Rm / (2 (Rm^2 - Rp^2))`` at
    ``WALL_BRANCH_T_K`` -- and is zero in cells with no annulus. The pool is
    ``nu_wall * M_n_a``, the annulus momentum the un-partitioned ledger sends
    to the vessel wall per unit time and volume.
    """
    Rp = np.asarray(geometry.Rp_cm, dtype=float)
    Rm = np.asarray(geometry.Rm_cm, dtype=float)
    _Vc, Va = zone_volumes(geometry)
    live = Va > 0.0
    vbar_n = np.sqrt(
        8.0 * WALL_BRANCH_T_K * kb_cgs / (np.pi * float(ion_mass_g))
    )
    nu_wall = np.where(
        live, vbar_n * Rm / (2.0 * np.maximum(Rm**2 - Rp**2, 1e-300)), 0.0
    )
    return nu_wall, nu_wall * np.asarray(M_n_a, dtype=float)


def partition_split(pool, survival):
    """Split a wall-branch ``pool`` into ``(absorbed, retained)``."""
    absorbed = survival * pool
    return absorbed, pool - absorbed


def drag_family_keys(terms, get_row=attr_row):
    """Return the ion-neutral drag-family term names present in ``terms``.

    The family is every ledger term that moves neutral momentum by collisional
    exchange with the ions, identified by name and required to carry an
    ``M_n`` row. Reading a SAVED trajectory the presence test necessarily
    degrades: the writer stores a zero row for every packed field, so the test
    becomes "the run evolved ``M_n``" rather than "this term books it". Terms
    that book nothing contribute exactly zero to the integral below, so the
    total is unaffected and only the returned name list can be longer.
    """
    return [
        k for k in terms
        if any(part in k for part in _DRAG_NAME_PARTS)
        and get_row(terms[k], "M_n") is not None
    ]


def drag_family_integral(terms, keys, Vc, Va, get_row=attr_row):
    """Return the volume-integrated |dM| of the drag family [dyn].

    Column rows are charged on ``Vc`` and annulus rows on ``Va``. The absolute
    value is deliberate: this is the SIZE of the family the re-routed momentum
    is compared against, not its net.
    """
    total = 0.0
    for k in keys:
        row = get_row(terms[k], "M_n")
        total += float(np.sum(np.abs(np.asarray(row, dtype=float)) * Vc))
        row_a = get_row(terms[k], "M_n_a")
        if row_a is not None:
            total += float(np.sum(np.abs(np.asarray(row_a, dtype=float)) * Va))
    return total


def report(sim, sigma, label):
    state = sim.state
    geom = sim.geometry
    Vc, Va = zone_volumes(geom)
    nn_a = np.asarray(state.nn_a, dtype=float)
    survival, tau, mfp = neutral_wall_partition_survival(geom, nn_a, sigma)

    terms = sim.rhs_terms()

    # The wall-branch pool: rebuild nu_wall from the closure's definition.
    live = Va > 0.0
    _nu_wall, pool = wall_branch_pool(geom, state.M_n_a, sim.ion_mass_g)
    _absorbed, retained = partition_split(pool, survival)

    pool_i = float(np.sum(np.abs(pool) * Va))
    retained_i = float(np.sum(np.abs(retained) * Va))

    # Total ion-neutral drag family: every ledger term that moves neutral
    # momentum by collisional exchange with the ions.
    drag_keys = drag_family_keys(terms)
    drag_i = drag_family_integral(terms, drag_keys, Vc, Va)

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
