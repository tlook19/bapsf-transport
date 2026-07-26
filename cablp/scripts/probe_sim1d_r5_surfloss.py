"""Pinpoint the cell that binds the surface_loss timestep during the ES1 crawl.

The R5 ES1 run ignites fine but crawls the entire plasma phase at dt~61 ns with
constraint=surface_loss (raw_stage_validation is a default-on R1 stance, so the
resolved plasma-source bound is active). surface_loss = 0.25 * min over
topology-active, DRAINING plasma cells of (n - n_floor)/|dn/dt| (and the Ee/Ei
floor margins). This reports the binding cell so we know whether it is a
physically-active discharge cell (a real stiffness) or a near-floor
boundary/edge cell needlessly gating the global step.

Run from <repo>/cablp with the fenicsx-env interpreter.
"""

import argparse

import numpy as np

from cablp.solvers._sim1d import LAPDSim1D, default_config
from cablp.vars._cons import ev_to_erg
from compare_sim1d_es1 import PARAM_OVERRIDES, FLAG_OVERRIDES


def build(nx, cached):
    params, flags = default_config()
    params.update(PARAM_OVERRIDES)
    flags.update(FLAG_OVERRIDES)
    params["neutral_exchange_model"] = "knudsen"
    params["nx"] = nx
    if cached:
        flags["use_cached_neutral_seed"] = True
        params["neutral_seed_cache_dir"] = cached
    return params, flags


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nx", type=int, default=120)
    ap.add_argument("--t-probe", type=float, default=2.05e-3)
    ap.add_argument("--cached-seed", default="scripts/neutral_seed_db")
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()

    params, flags = build(args.nx, args.cached_seed)
    sim = LAPDSim1D(params, flags)
    sim.start_simulation(t_end=args.t_probe)

    state = sim.state
    geom = sim.geometry
    floors = sim._floors
    time = sim._time
    roles = np.asarray(geom.cell_role)
    active = np.asarray(geom.plasma_active, dtype=bool)

    rhs = sim._plasma_source_timestep_rhs(state, time)
    frac = float(params.get("density_dt_fraction", 0.25))

    n = np.asarray(state.n, float)
    dn = np.asarray(rhs.n, float)
    Ee = np.asarray(state.Ee, float); dEe = np.asarray(rhs.Ee, float)
    Ei = np.asarray(state.Ei, float); dEi = np.asarray(rhs.Ei, float)
    Te_floor = float(floors["Te"]); Ti_floor = float(floors["Ti"]); n_floor = float(floors["n"])

    # Per-cell margins exactly as _negative_margin_timestep forms them.
    m_n = n - n_floor
    m_Ee = Ee - 1.5 * Te_floor * ev_to_erg * n
    dm_Ee = dEe - 1.5 * Te_floor * ev_to_erg * dn
    m_Ei = Ei - 1.5 * Ti_floor * ev_to_erg * n
    dm_Ei = dEi - 1.5 * Ti_floor * ev_to_erg * dn

    def cell_dt(margin, rate):
        dt = np.full_like(margin, np.inf)
        drn = (rate < 0.0) & active
        pos = drn & (margin > 0.0)
        dt[pos] = frac * margin[pos] / (-rate[pos])
        dt[drn & (margin <= 0.0)] = 0.0
        return dt

    dt_n = cell_dt(m_n, dn)
    dt_Ee = cell_dt(m_Ee, dm_Ee)
    dt_Ei = cell_dt(m_Ei, dm_Ei)
    dt_all = np.minimum(np.minimum(dt_n, dt_Ee), dt_Ei)

    print(f"# surface_loss diagnostic  t={time*1e3:.3f} ms  nx={args.nx}  frac={frac}")
    print(f"# n_floor={n_floor:.3e}  Te_floor={Te_floor}  Ti_floor={Ti_floor}")
    print(f"# global surface_loss dt = {np.min(dt_all):.3e} s "
          f"(argmin cell {int(np.argmin(dt_all))}, role {roles[int(np.argmin(dt_all))]})")
    order = np.argsort(dt_all)[: args.top]
    hdr = ("cell", "role", "act", "dt[s]", "which", "n", "n/floor", "dn", "Te", "Ti")
    print(("{:>5}{:>12}{:>4}{:>11}{:>7}{:>11}{:>9}{:>12}{:>8}{:>8}").format(*hdr))
    Te = np.asarray(sim.derived.Te, float); Ti = np.asarray(sim.derived.Ti, float)
    for c in order:
        which = min((("n", dt_n[c]), ("Ee", dt_Ee[c]), ("Ei", dt_Ei[c])), key=lambda kv: kv[1])[0]
        print(("{:>5}{:>12}{:>4}{:>11.3e}{:>7}{:>11.3e}{:>9.2f}{:>12.3e}{:>8.3f}{:>8.3f}").format(
            int(c), str(roles[c]), "Y" if active[c] else "n", float(dt_all[c]), which,
            float(n[c]), float(n[c] / n_floor), float(dn[c]), float(Te[c]), float(Ti[c])))


if __name__ == "__main__":
    main()
