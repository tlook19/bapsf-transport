"""Characteristic ghost-cell Bohm outflow: short startup RUN gate.

The boundary sets the edge dynamics, so its decisive validation is a run, not a
static probe. This runs the live ES1 stance to a short startup window (through
main discharge) and checks that the boundary:

  (a) establishes the Bohm outflow at the edge: the source-cathode and collector
      edge cells flow INTO their walls at ~ the sound speed (u sign = outward
      normal, |u| a good fraction of c_s), instead of the A1 "away from the wall"
      signature;
  (b) is a net ENERGY SINK over the main-discharge window (electron internal +
      ion internal + reconstructed kinetic < 0).

RETIRED 2026-08-31 (Tom): gate (c) -- "the reconstructed kinetic is no longer
the A1 source the historical boundary_absorption books at the same stance" --
and with it the second, characteristic-OFF run this script used to make. That
comparison arm is not constructible now that the legacy volumetric absorber and
its flag are gone. The last measurement of record on this fixture, at base
commit 967ff0c, was new -16.01 kW vs old +0.20 kW.

A short startup run is a diagnostic, NOT an ES campaign point; longer runs need
Tom's go-ahead. This is minutes-long (one partial production run).

Usage:  python scripts/verify_sim1d_r3_boundary_startup.py [--t-end-ms 6]
"""
import argparse
import sys

import numpy as np

from cablp.solvers._sim1d import LAPDSim1D, default_config
from cablp.solvers._sim1d.physics.flux import ion_sound_speed
from compare_sim1d_es1 import FLAG_OVERRIDES, PARAM_OVERRIDES

MI = 4.0 * 1.6726e-24  # He ion mass [g]
ERG_TO_W = 1e-7


def run(t_end):
    params, flags = default_config()
    params.update(PARAM_OVERRIDES)
    flags.update(FLAG_OVERRIDES)
    params["nx"] = 120
    sim = LAPDSim1D(params, flags)
    status = "completed"
    try:
        sim.start_simulation(t_end=t_end)
    except Exception as exc:  # noqa: BLE001
        status = f"STOPPED: {type(exc).__name__}: {exc}"
    return sim, sim.get_results(), status


def edge_cells(geo):
    absorbing = np.flatnonzero(np.asarray(geo.plasma_absorbing, bool))
    return {
        int(f): (int(geo.plasma_face_live_cell[f]),
                 -1.0 if int(geo.plasma_face_live_cell[f]) == int(f) else 1.0)
        for f in absorbing
        if int(geo.plasma_face_live_cell[f]) >= 0
    }


def term_power(result, geo, name, sel):
    """Net booked power [W] of one boundary term over the selected snapshots:
    electron internal + ion internal + reconstructed kinetic, median over sel."""
    Vp = np.asarray(geo.plasma_volume_cm3)
    u = np.asarray(result.u)
    e = np.asarray(result.electron_energy_terms_W_cm3[name])
    i = np.asarray(result.ion_energy_terms_W_cm3[name])
    dM = np.asarray(result.rhs_terms[name]["M"])
    dn = np.asarray(result.rhs_terms[name]["n"])
    dK = (u * dM - 0.5 * MI * u**2 * dn) * Vp[None, :] * ERG_TO_W
    internal = (e + i) * Vp[None, :]
    net = np.sum(internal + dK, axis=1)
    return (float(np.median(net[sel])) if np.any(sel) else float("nan"),
            float(np.median(np.sum(internal, axis=1)[sel])) if np.any(sel) else float("nan"),
            float(np.median(np.sum(dK, axis=1)[sel])) if np.any(sel) else float("nan"))


def summarize(sim, result, name):
    geo = sim.geometry
    time = np.asarray(result.time)
    phase = np.asarray(getattr(result, "phase", np.array([], dtype=object))).astype(str)
    u = np.asarray(result.u)
    Te = np.asarray(result.Te) if hasattr(result, "Te") else None
    main = phase == "main_discharge"
    sel = main if np.any(main) else np.ones(time.shape, bool)
    edges = edge_cells(geo)
    # Edge flow at the final main-discharge snapshot.
    idx = np.flatnonzero(sel)[-1]
    print(f"[{name}] reached {time[-1]*1e3:.2f} ms, "
          f"phases={sorted(set(phase.tolist()))}, main-disch snaps={int(main.sum())}")
    edge_ok = True
    for face, (live, outward) in edges.items():
        u_edge = float(u[idx, live])
        cs = float(ion_sound_speed(float(Te[idx, live]), sim._mu)) if Te is not None else float("nan")
        into_wall = (np.sign(u_edge) == np.sign(outward)) and abs(u_edge) > 0.3 * cs
        edge_ok &= into_wall
        role = "cathode " if outward < 0 else "collector"
        print(f"   {role} cell {live:3d}: u={u_edge:+.3e}  outward={outward:+.0f}  "
              f"c_s={cs:.3e}  Mach={u_edge/cs:+.2f}  into-wall={into_wall}")
    net, internal, kinetic = term_power(result, geo, name, sel)
    print(f"   {name} net power (main disch) = {net/1e3:+.2f} kW "
          f"(internal {internal/1e3:+.2f}, kinetic {kinetic/1e3:+.2f})")
    return edge_ok, net, kinetic


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t-end-ms", type=float, default=6.0)
    args = parser.parse_args(argv)
    t_end = args.t_end_ms * 1e-3

    sim_on, res_on, st_on = run(t_end)
    print(f"characteristic boundary: {st_on}")
    edge_ok, net_on, kin_on = summarize(sim_on, res_on, "characteristic_boundary")

    print("\n--- boundary run gate ---")
    a = edge_ok
    b = net_on < 0.0
    print(f"(a) Bohm outflow established at both edges     : {a}")
    print(f"(b) characteristic boundary is a net sink      : {b} ({net_on/1e3:+.2f} kW)")
    print(f"    reconstructed kinetic = {kin_on/1e3:+.2f} kW "
          f"(base-commit reference: -16.01 kW; the old absorber booked +0.20 kW "
          f"before it was retired (Tom, 2026-08-31))")
    ok = a and b and st_on == "completed"
    print("startup run:", "OK" if ok else "FAILED / NULL (deliverable)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
