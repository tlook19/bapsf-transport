"""Run one R5 ES1 refit point (repaired stance) and save an H5 for scoring.

Iteration-grade driver: nx=60, density_dt_fraction=0.5, t_end=22 ms (discharge
only -- the afterglow crawls; stages i+ii don't need it). Uses the ES production
config with circuit overrides for the coupled refit:

  --V-bank            supply setpoint (refit pins 180)
  --R-comp            TOTAL loop series resistance -> sets discharge current
                      (and cools port Te as it lowers V_b -> less plasma power)
  --R-comp-internal   voltage-probe partition -> V_dis = V_b + I*R_comp_internal
                      (raises the MEASURED V_dis without changing current/Te)

Reports plateau current, V_b/V_dis, and the R_comp_internal that WOULD land a
target V_dis (since it is a pure measurement relabel, no re-run needed). Score
with compare_sim1d_es1.py --from-h5 <out> --nx 60 (port Te + gradient are the
targets, not peak Te).
"""

import argparse
import time as _walltime

import numpy as np

from compare_sim1d_es1 import run_model
from cablp.solvers._sim1d.results.io import save_result_hdf5


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nx", type=int, default=60)
    ap.add_argument("--V-bank", type=float, default=180.0)
    ap.add_argument("--R-comp", type=float, default=None)
    ap.add_argument("--R-comp-internal", type=float, default=None)
    ap.add_argument("--density-dt-fraction", type=float, default=0.5)
    ap.add_argument("--t-end", type=float, default=22e-3)
    ap.add_argument("--vdis-target", type=float, default=151.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    extra = {"V_bank": args.V_bank, "density_dt_fraction": args.density_dt_fraction}
    if args.R_comp is not None:
        extra["R_comp"] = args.R_comp
    if args.R_comp_internal is not None:
        extra["R_comp_internal"] = args.R_comp_internal

    # tau_discharge is HARDWARE, not a cost knob: the config.py default 20 ms
    # matches the measured drive (19.97/20.04/20.01 ms on ES1/2/3). It was
    # previously back-derived as max(t_end-2.5e-3, 1e-3), which silently
    # shortened the drive by 0.5 ms (diagnostician, 2026-07-27); run cost is
    # now capped by the explicit t_end passed to run_model.
    extra["tau_discharge"] = 20e-3
    extra["tau_afterglow"] = 0.0
    # Pin the puff to its hardware location: "the physical pipe sits at ~60 cm
    # (anode + 10)" (config.py neutral_defaults docstring). The solver default
    # (None) anchors the cosine_pipe centre to the puff CELL centre, which
    # moves with nx. Pinned at the driver level only -- the solver-default
    # change is deferred to the stance-promotion batch (golden recapture
    # moment). Interim: fixed-cell source region 0-100 cm + pinned puff per
    # Tom 2026-07-27, pending CAD.
    extra["gas_puff_z_cm"] = 60.0

    print(f"# refit: V_bank={args.V_bank} R_comp={args.R_comp} "
          f"R_comp_internal={args.R_comp_internal} nx={args.nx} -> {args.out}")
    t0 = _walltime.time()
    result, geometry, params, flags = run_model(
        nx=args.nx, extra=extra, t_end=args.t_end)
    wall = _walltime.time() - t0

    t = (np.asarray(result.time, float)) * 1e3
    diag = result.cathode_diagnostics
    I = np.asarray(diag["source_I_tot"], float)
    Vb = np.asarray(diag.get("circuit_V_dis_dt_integral", np.zeros_like(I)), float)
    # plateau window on the model clock (~15-19.5 ms of discharge; here absolute)
    m = (t >= 15.0) & (t <= min(19.5, t[-1]))
    Iplat = float(np.median(I[m])) if m.any() else float("nan")
    print(f"# wall={wall:.0f}s saves={len(t)} t_end={t[-1]:.2f}ms  "
          f"I_plateau(15-19.5ms)={Iplat:.0f} A  target 2991 A")
    # V_dis plateau (dt-integrated) and the R_comp_internal for the target
    Vdis = np.asarray(diag.get("source_V_dis", diag.get("circuit_V_dis", [])), float)
    if Vdis.size == I.size and m.any():
        Vd = float(np.median(Vdis[m]))
        need = (args.vdis_target - Vd) / Iplat if Iplat else float("nan")
        rci = float(params.get("R_comp_internal", 0.0))
        # V_b = V_dis - I*R_comp_internal (this run); need for target from V_b:
        Vb_plat = Vd - Iplat * rci
        need_from_Vb = (args.vdis_target - Vb_plat) / Iplat if Iplat else float("nan")
        print(f"# V_dis_plateau={Vd:.1f} V (R_comp_internal={rci:.4g}); "
              f"V_b_plateau={Vb_plat:.1f} V")
        print(f"# R_comp_internal for V_dis={args.vdis_target:.0f} V: "
              f"{need_from_Vb:.4g} Ohm")
    save_result_hdf5(args.out, result, params=params, flags=flags)
    print(f"# saved {args.out}")


if __name__ == "__main__":
    main()
