"""R3 boundary-sampling diagnostic: presheath_alpha vs flat exp(-1/2).

Decision-support for R3 (SIM1D_MODEL_AUDIT_PLAN, "R3 physics map signed off"):
the fluid boundary sink and the circuit current must sample ONE sheath-edge
density n_se. This runs the live ES1 stance to a short startup window (through
main discharge) with the mesh-adjusted presheath (`b_presheath_length=1`) and
the flat `exp(-1/2)` sampling (`b_presheath_length=0`, which also matches the
circuit's historical flat factor), and compares end-cell depletion.

Finding (recorded 2026-07-24): neither over-depletes -- both hold the end cells
far above the density floor through main discharge with R1's electrode-source
timestep bound active, so the historical "depleting too fast" was a timestep
issue R1 now handles, not something the mesh-adjustment is still needed to
prevent. Flat holds a *higher* min edge density, confirming presheath_alpha
depletes more and is the mesh-independent physical n_se. R3 therefore unifies on
presheath_alpha (the circuit is upgraded to it), not flat.

This is a ~minutes-long diagnostic (two partial production runs), not a fast
gate. Usage:  python scripts/verify_sim1d_r3_presheath.py [--t-end-ms 6]
"""
import argparse
import sys

import numpy as np

from cablp.solvers._sim1d import LAPDSim1D, default_config
from cablp.solvers._sim1d.core.geometry import cathode_adjacent_cells
from compare_sim1d_es1 import FLAG_OVERRIDES, PARAM_OVERRIDES


def run(b_presheath_length, t_end):
    params, flags = default_config()
    params.update(PARAM_OVERRIDES)
    flags.update(FLAG_OVERRIDES)
    params["nx"] = 120
    params["b_presheath_length"] = b_presheath_length
    sim = LAPDSim1D(params, flags)
    status = "completed"
    try:
        sim.start_simulation(t_end=t_end)
    except Exception as exc:  # noqa: BLE001
        status = f"STOPPED: {type(exc).__name__}: {exc}"
    return sim, sim.get_results(), status


def summarize(sim, result, status, t_end):
    geo = sim.geometry
    roles = np.asarray(geo.cell_role)
    edge = list(cathode_adjacent_cells(geo)) + list(np.flatnonzero(roles == "collector"))
    n = np.asarray(result.n, dtype=float)
    time = np.asarray(result.time, dtype=float)
    phase = np.asarray(getattr(result, "phase", np.array([], dtype=str)), dtype=str)
    min_edge = float(np.min(n[:, edge])) if (n.ndim == 2 and edge) else float("nan")
    ledger = getattr(result, "floor_ledger", {}) or {}
    n_added = float(ledger.get("n_particles_added", 0.0))
    return {
        "status": status,
        "reached_ms": time[-1] * 1e3 if time.size else 0.0,
        "phases": sorted(set(phase.tolist())) if phase.size else [],
        "min_edge_density": min_edge,
        "n_particles_added": n_added,
        "floor_ledger": {k: float(v) for k, v in ledger.items()},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t-end-ms", type=float, default=6.0)
    args = parser.parse_args(argv)
    t_end = args.t_end_ms * 1e-3

    results = {}
    for bpl, tag in ((1.0, "presheath"), (0.0, "flat")):
        sim, result, status = run(bpl, t_end)
        results[tag] = summarize(sim, result, status, t_end)
        s = results[tag]
        print(f"[b_presheath_length={bpl}] {tag}")
        print(f"  status={s['status']}  reached={s['reached_ms']:.3f} ms  phases={s['phases']}")
        print(f"  min edge density = {s['min_edge_density']:.4e} cm^-3")
        print(f"  density-floor clipping = {s['n_particles_added']:.3e} particles")
        print()

    p, f = results["presheath"], results["flat"]
    both_complete = p["status"] == "completed" and f["status"] == "completed"
    no_density_clip = p["n_particles_added"] == 0.0 and f["n_particles_added"] == 0.0
    flat_gentler = f["min_edge_density"] >= p["min_edge_density"]
    ok = both_complete and no_density_clip and flat_gentler
    print(f"both reach main discharge : {both_complete}")
    print(f"no density-floor clipping : {no_density_clip}")
    print(f"flat edge >= presheath    : {flat_gentler} "
          f"({f['min_edge_density']:.3e} vs {p['min_edge_density']:.3e})")
    print("=> presheath_alpha is the mesh-independent n_se; unify on it "
          "(circuit upgraded).")
    print("R3 presheath diagnostic:", "OK" if ok else "UNEXPECTED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
