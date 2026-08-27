"""R5.4 collector surface-power ledger line (R3 tail).

Two checks:
  (1) SCALAR correctness: a live sim's saved `collector_surface_power_W` equals a
      direct reconstruction from the boundary term at the collector cell.
  (2) MAGNITUDE on the settled ES artifact: the collector surface power should be
      small in the high-density (detached-ish) ES regime -- the ledger line that
      grows in low-puff/attached runs.

Usage:  python scripts/probe_sim1d_r5_collector.py [--h5 PATH]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

DEFAULT_H5 = (
    "es1_nx120_m6_sq4600_g3200_c120_ts1900_l8p1_mn2mom300k_bmom_"
    "g1vessel150_rp15_baf150p27_r30_es1.h5"
)
WINDOW = (18.81e-3, 23.80e-3)
ERG_TO_W = 1.0e-7


def _collector_power_from_h5(f, sel):
    """Reconstruct the collector surface power [W] from saved per-cell terms."""
    Vp = np.asarray(f["geometry/plasma_volume_cm3"], dtype=float)
    role = np.asarray(f["geometry/cell_role"]).astype(str)
    collector = role == "collector"
    if not np.any(collector):
        return None
    u = np.asarray(f["u"], dtype=float)
    # ion mass: He (params_json), fall back to He
    import json
    # SUPERSEDED 2026-08-21: the unified helium mass is cablp.vars._cons
    # .m_He_cgs = 6.6464790809e-24 g (Ar(4He)*u, CODATA 2022). The literal
    # below is 0.90 ppm low and is left AS A RECORD of what this dated script ran.
    mi = 6.6464731e-24
    try:
        p = json.loads(f.attrs["params_json"])
        if str(p.get("gas_type", "He")) == "H":
            mi = 1.6726219e-24
    except Exception:
        pass
    total = np.zeros(u.shape[0])
    for name in ("characteristic_boundary", "boundary_absorption"):
        if name not in f["electron_energy_terms_W_cm3"]:
            continue
        Ee = np.asarray(f[f"electron_energy_terms_W_cm3/{name}"])  # W/cm^3
        Ei = np.asarray(f[f"ion_energy_terms_W_cm3/{name}"])
        g = f[f"rhs_terms/{name}"]
        M = np.asarray(g["M"]) if "M" in g else np.zeros_like(Ee)
        n = np.asarray(g["n"]) if "n" in g else np.zeros_like(Ee)
        dK = (u * M - 0.5 * mi * u**2 * n) * ERG_TO_W  # W/cm^3
        p_cell = -(Ee + Ei + dK) * Vp[None, :]  # W
        total += np.sum(p_cell[:, collector], axis=1)
    return float(np.median(total[sel]))


def gate_scalar_correctness():
    from cablp.solvers._sim1d import LAPDSim1D, default_config
    params, flags = default_config()
    params.update({
        "ne0": 1e12, "nn0": 1e13, "Te0": 15.0, "Ti0": 2.0,
        "gas_puff_enabled": False, "pump_enabled": False,
        "phase_transition_mode": "scheduled",
        "tau_neutral_prebreakdown": 0.0, "tau_prebreakdown": 0.0,
        "tau_breakdown": 0.0, "tau_discharge": 1.0, "tau_afterglow": 0.0,
        "nx": 60,
    })
    flags.update({"Plasma": True, "cathode_coupling": False,
                  "characteristic_boundary": True})
    sim = LAPDSim1D(params, flags)
    sim.start_simulation(t_end=3.2e-3)
    snap = sim._trajectory_snapshot(sim._time)
    scalar = float(snap["cathode_diagnostics"]["collector_surface_power_W"])
    # direct reconstruction on the same state/terms
    rhs = sim.rhs_terms(include_heat_conduction=True, time=sim._time)
    direct = sim._collector_surface_power_W(rhs, sim.derived)
    ok = abs(scalar - direct) <= 1e-9 * max(abs(scalar), 1.0)
    print(f"[{'PASS' if ok else 'FAIL'}] scalar == direct reconstruction "
          f"(scalar={scalar:.4g} W, direct={direct:.4g} W)")
    return ok


def main(argv=None):
    import h5py
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h5", default=DEFAULT_H5)
    args = ap.parse_args(argv)

    print("R5.4 collector surface-power line")
    print("=" * 60)
    ok = gate_scalar_correctness()

    path = Path(args.h5)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    if path.exists():
        f = h5py.File(path, "r")
        time = np.asarray(f["time"]); phase = np.asarray(f["phase"]).astype(str)
        sel = (time >= WINDOW[0]) & (time <= WINDOW[1]) & (phase == "main_discharge")
        P = _collector_power_from_h5(f, sel)
        print(f"\nsettled ES artifact ({path.name[:40]}...):")
        if P is None:
            print("  no collector cell in this geometry")
        else:
            print(f"  collector surface power (settled median) = {P/1e3:+.3f} kW")
            print("  -> small in the high-density/detached ES regime, as expected; "
                  "the ledger line grows in low-puff/attached runs.")
    else:
        print(f"\n(ES artifact {path.name} not present; skipped magnitude check.)")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
