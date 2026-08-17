"""STEP-0 LOCK SCOUT for the corner-arm retry at Rp_end = 25 cm.

Run artifact (untracked).  Adapted from the diagnostician's scratchpad
lockdiag.py: same config construction, same dt-candidate decomposition, plus a
per-step dt HISTORY scan so the PASS/FAIL question ("does it hit the dt_min
lock?") is answered over the WHOLE burst rather than only at its final state.

Lock signature (core/timestep.py:198-200): the accepted dt is
``min(max(raw_dt, dt_min), dt_max)`` and ``active_constraint`` is relabelled to
the literal string ``"dt_min"`` exactly when ``raw_dt < dt_min``.  The solver
records ``accepted_dt`` and ``active_constraint`` for EVERY step in its
``diagnostics``, so a lock cannot hide.

PASS  = zero steps with active_constraint == 'dt_min' AND dt recovers.
FAIL  = any dt_min step -> STOP, do not run the production solve.

Usage:
    python scripts/mn_lockscout_rp25.py --arm both --rp-end 25.0 \
        --nx 120 --max-steps 9000
"""
import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))

from compare_sim1d_es1 import PARAM_OVERRIDES, FLAG_OVERRIDES  # noqa: E402
from run_mechanism_ladder import ES_OPERATING  # noqa: E402
from cablp.solvers._sim1d import default_config, LAPDSim1D  # noqa: E402
from cablp.solvers._sim1d.core.state import derive_state  # noqa: E402
from cablp.vars._cons import ev_to_erg  # noqa: E402


def build(nx, flare, tail, rp_end):
    params, flags = default_config()
    params.update(PARAM_OVERRIDES)
    flags.update(FLAG_OVERRIDES)
    flags["neutral_momentum"] = True
    op = ES_OPERATING[1]
    params.update({
        "neutral_exchange_model": "knudsen", "nx": nx,
        "V_bank": op["V_bank"], "cathode_solver_model": "current_driven",
        "beam_deposition_model": "csda", "beam_anomalous_model": "quasilinear",
        "cathode_emission_profile": "gaussian",
        "cathode_warming_model": "power_balance",
        "cathode_Ts_base_K": op["Ts_standby_K"],
        "cathode_heat_capacity_J_per_K": 120.0, "cathode_emissivity": 0.7,
        "phi_wf": 2.869, "cathode_surface_model": "ads_des",
        "cathode_phiwf_clean_eV": 2.809, "cathode_cleaning_sigma_cm2": 3.5e-16,
        "cathode_cleaning_E_th_eV": 20.0,
        "gas_puff_mode": "square", "S_gp": 6000.0,
        "cathode_sample_smoothing": "presheath",
        "tau_afterglow": 0.006, "ion_neutral_drag_model": "constant",
        "b_ion_neutral_drag": 1.0, "T_s": 1998.15,
        "Te_birth_ionization": "local", "max_steps_action": "stop",
        "max_output_steps": 5,
    })
    if flare:
        params["end_expansion_plasma_radius_cm"] = rp_end
    if tail:
        params["heating_anomalous_transport"] = "tail_walk"
    return params, flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=120)
    ap.add_argument("--arm", required=True,
                    choices=("none", "flare", "tail", "both"))
    ap.add_argument("--rp-end", type=float, default=25.0)
    ap.add_argument("--max-steps", type=int, default=9000)
    a = ap.parse_args()

    flare = a.arm in ("flare", "both")
    tail = a.arm in ("tail", "both")
    p, f = build(a.nx, flare, tail, a.rp_end)
    print("=" * 74)
    print(f"LOCK SCOUT  arm={a.arm}  nx={a.nx}  max_steps={a.max_steps}")
    print(f"  end_expansion_plasma_radius_cm = "
          f"{p.get('end_expansion_plasma_radius_cm')}  (flare={flare})")
    print(f"  heating_anomalous_transport    = "
          f"{p.get('heating_anomalous_transport')}  (tail={tail})")
    print(f"  heating_anomalous_tail_energy_eV = "
          f"{p.get('heating_anomalous_tail_energy_eV')}")
    print(f"  S_gp = {p.get('S_gp')}   dt_min = {p.get('dt_min')}")
    print("=" * 74)

    sim = LAPDSim1D(p, f)
    sim.start_simulation(max_steps=a.max_steps, progress_interval_s=1e-5)
    t = sim._time
    g = sim.geometry
    print(f"\nreached t={t:.9e} s  phase={sim.phase_at_time(t)}")

    # ---- per-step dt HISTORY over the whole burst (the PASS/FAIL evidence) --
    res = sim.get_results()
    diag_hist = getattr(res, "diagnostics", None) or []
    if isinstance(diag_hist, dict):
        adt = np.asarray(diag_hist.get("accepted_dt", ()), dtype=float)
        ac = np.asarray([str(x) for x in diag_hist.get("active_constraint", ())])
    else:
        # list of per-step records (dicts or objects)
        def _get(rec, key):
            if isinstance(rec, dict):
                return rec.get(key)
            return getattr(rec, key, None)
        adt = np.asarray(
            [float(_get(r, "accepted_dt") or _get(r, "dt") or np.nan)
             for r in diag_hist], dtype=float)
        ac = np.asarray([str(_get(r, "active_constraint")) for r in diag_hist])
    dt_min_cfg = float(p.get("dt_min", 1e-10))
    print(f"\n--- dt history over the burst ({adt.size} steps recorded) ---")
    if adt.size:
        print(f"  min accepted_dt  = {adt.min():.6e} s")
        print(f"  max accepted_dt  = {adt.max():.6e} s")
        print(f"  final accepted_dt= {adt[-1]:.6e} s")
        print(f"  dt_min (config)  = {dt_min_cfg:.6e} s")
        print(f"  steps at dt <= dt_min : "
              f"{int((adt <= dt_min_cfg).sum())}")
        n_lock = int((ac == "dt_min").sum())
        print(f"  steps with active_constraint == 'dt_min' : {n_lock}")
        names, counts = np.unique(ac, return_counts=True)
        print("  binding-constraint census over the burst:")
        for nm, c in sorted(zip(names, counts), key=lambda x: -x[1]):
            print(f"     {nm:24s} {c:7d}  ({100.0 * c / ac.size:5.1f} %)")
        # recovery: is dt at the end above its running minimum?
        print(f"  dt recovered from its minimum: "
              f"{bool(adt[-1] > adt.min())}  "
              f"(final/min = {adt[-1] / adt.min():.3g})")
        locked = n_lock > 0
    else:
        print("  NO per-step diagnostics recorded -- cannot certify; treat as FAIL")
        locked = True

    # ---- instantaneous decomposition at the final state (lockdiag's view) ---
    d = sim.suggest_timestep()
    print(f"\nnext dt would be {d.dt:.4e}, active_constraint={d.active_constraint}")
    cands = [("plasma_cfl", d.dt_plasma_cfl), ("front_density", d.dt_front_density),
             ("surface_loss", d.dt_surface_loss),
             ("neutral_exchange", d.dt_neutral_exchange),
             ("neutral_sources", d.dt_neutral_sources), ("reactions", d.dt_reactions),
             ("energy_exchange", d.dt_energy_exchange),
             ("electron_cooling", d.dt_electron_cooling),
             ("ion_charge_exchange", d.dt_ion_charge_exchange),
             ("heat_conduction", d.dt_heat_conduction),
             ("ion_neutral_drag", d.dt_ion_neutral_drag),
             ("neutral_wind", d.dt_neutral_wind), ("dt_max", d.dt_max)]
    print("ALL dt candidates (sorted):")
    for nm, v in sorted(cands, key=lambda x: x[1]):
        print(f"   {nm:20s} {v:.6e}")

    st = sim._state
    rhs = sim._plasma_source_timestep_rhs(state=st, time=t)
    floors = sim._floors
    n = np.asarray(st.n, float)
    dn = np.asarray(rhs.n, float)
    der = derive_state(st, floors=floors, ion_mass_g=sim._ion_mass_g)
    frac = 0.25
    print("\nsurface_loss channel decomposition:")
    for nm, mg, rt in (
        ("density", n - float(floors["n"]), dn),
        ("Ee", np.asarray(st.Ee, float) - 1.5 * float(floors["Te"]) * ev_to_erg * n,
         np.asarray(rhs.Ee, float) - 1.5 * float(floors["Te"]) * ev_to_erg * dn),
        ("Ei", np.asarray(st.Ei, float) - 1.5 * float(floors["Ti"]) * ev_to_erg * n,
         np.asarray(rhs.Ei, float) - 1.5 * float(floors["Ti"]) * ev_to_erg * dn),
    ):
        d2 = np.full_like(mg, np.inf)
        m2 = rt < 0
        d2[m2] = frac * mg[m2] / -rt[m2]
        j = int(np.argmin(d2))
        print(f"  {nm:8s} dt={d2[j]:.6e} cell {j} role={g.cell_role[j]} "
              f"margin={mg[j]:.6e} rate={rt[j]:.6e}")

    # ---- the diagnostic quantity: collector-cell n vs the density floor -----
    nfloor = float(floors["n"])
    print(f"\ncollector/end margin vs ne_floor = {nfloor:.6e}:")
    print("  cell role         n            n/floor    Te        u[cm/s]     nn")
    for i in range(g.cells - 13, g.cells):
        print(f"  {i:4d} {str(g.cell_role[i]):11s} {n[i]:.4e} {n[i] / nfloor:9.4f} "
              f"{der.Te[i]:9.4f} {der.u[i]:11.4e} {np.asarray(st.nn, float)[i]:.4e}")

    print("\n" + "=" * 74)
    print(f"SCOUT VERDICT: {'FAIL -- dt_min LOCK DETECTED' if locked else 'PASS -- no dt_min lock'}")
    print("=" * 74)
    return 2 if locked else 0


if __name__ == "__main__":
    raise SystemExit(main())
