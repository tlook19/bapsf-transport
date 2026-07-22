"""Run one M6 candidate point with full extra-param control (no driver edits).

Thin wrapper around compare_sim1d_es1.run_model for M6 recalibration points
that need params the ladder driver does not expose (close lag, L, etc.),
kept as a separate file so concurrent sessions editing the shared driver
and solver are never touched.

Usage:
    python scripts/run_m6_point.py --es 1 --sgp 3400 --close-lag 2e-3 \
        --save-h5 out.h5 [--mn] [--L 8.1e-6] [--extra k=v ...]
"""

import argparse
import json

from compare_sim1d_es1 import run_model
from run_mechanism_ladder import ES_OPERATING
from cablp.solvers._sim1d.results.io import save_result_hdf5


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--es", type=int, choices=(1, 2, 3), required=True)
    p.add_argument("--nx", type=int, default=120)
    p.add_argument("--sgp", type=float, required=True)
    p.add_argument("--close-lag", type=float, default=None)
    p.add_argument("--L", type=float, default=None,
                   help="loop inductance override [H]")
    p.add_argument("--g-cond", type=float, default=1200.0)
    p.add_argument("--c-th", type=float, default=120.0)
    p.add_argument("--mn", action="store_true")
    p.add_argument("--two-zone", action="store_true",
                   help="neutral_two_zone particle channel "
                        "(NEUTRAL_TWOZONE_PLAN.md): nn becomes the column "
                        "density, nn_a the annulus")
    p.add_argument("--no-smooth", action="store_true")
    p.add_argument("--extra", nargs="*", default=(),
                   help="additional k=v param overrides (JSON-parsed values)")
    p.add_argument("--save-h5", required=True)
    args = p.parse_args(argv)

    op = ES_OPERATING[args.es]
    extra = {
        "nx": args.nx,
        "V_bank": op["V_bank"],
        "cathode_solver_model": "current_driven",
        "beam_deposition_model": "csda",
        "beam_anomalous_model": "quasilinear",
        "cathode_emission_profile": "gaussian",
        "cathode_warming_model": "power_balance",
        "T_s": op["Ts_standby_K"],
        "cathode_Ts_base_K": op["Ts_standby_K"],
        "cathode_heat_capacity_J_per_K": args.c_th,
        "cathode_conduction_W_per_K": args.g_cond,
        "cathode_emissivity": 0.7,
        "phi_wf": 2.869,
        "cathode_surface_model": "ads_des",
        "cathode_phiwf_clean_eV": 2.809,
        "cathode_cleaning_sigma_cm2": 3.5e-16,
        "cathode_cleaning_E_th_eV": 20.0,
        "gas_puff_mode": "square",
        "S_gp": args.sgp,
    }
    if not args.no_smooth:
        extra["cathode_sample_smoothing"] = "presheath"
    if args.close_lag is not None:
        extra["gas_puff_close_lag_s"] = args.close_lag
    if args.L is not None:
        extra["L_parasitic_H"] = args.L
    flags_extra = {}
    if args.two_zone:
        flags_extra["neutral_two_zone"] = True
    if args.mn:
        extra.update({
            "ion_neutral_drag_model": "constant",
            "b_ion_neutral_drag": 1.0,
            "neutral_momentum_radial": "two_zone",
            "neutral_mesh_accommodation": True,
        })
        flags_extra["neutral_momentum"] = True
    for kv in args.extra:
        k, v = kv.split("=", 1)
        try:
            extra[k] = json.loads(v)
        except json.JSONDecodeError:
            extra[k] = v

    result, geometry, params, flags = run_model(
        resolved=True, nx=args.nx, extra=extra,
        flags_extra=flags_extra or None,
    )
    save_result_hdf5(args.save_h5, result, params=params, flags=flags)
    print(f"saved {args.save_h5}")


if __name__ == "__main__":
    main()
