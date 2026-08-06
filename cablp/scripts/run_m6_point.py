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

from compare_sim1d_es1 import PRODUCTION_NX, run_model
from run_mechanism_ladder import ES_OPERATING
from cablp.solvers._sim1d.results.io import save_result_hdf5
from cablp.solvers._sim1d.results.health import summarize_result


ELECTRON_BIRTH_POLICY = "floor"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--es", type=int, choices=(1, 2, 3), required=True)
    p.add_argument("--nx", type=int, default=PRODUCTION_NX)
    p.add_argument("--sgp", type=float, required=True)
    p.add_argument("--close-lag", type=float, default=None)
    p.add_argument("--L", type=float, default=None,
                   help="loop inductance override [H]")
    p.add_argument("--g-cond", type=float, default=None,
                   help="skin->substrate conduction [W/K]; default None = "
                        "defer to the shared production config "
                        "(compare_sim1d_es1.PARAM_OVERRIDES)")
    p.add_argument("--c-th", type=float, default=120.0)
    p.add_argument("--mn", action="store_true")
    p.add_argument("--two-zone", action="store_true",
                   help="neutral_two_zone particle channel "
                        "-- nn becomes the column "
                        "density, nn_a the annulus")
    p.add_argument("--no-smooth", action="store_true")
    p.add_argument("--extra", nargs="*", default=(),
                   help="additional k=v param overrides (JSON-parsed values)")
    p.add_argument("--extra-flag", nargs="*", default=(),
                   help="additional k=v input_flags overrides (JSON-parsed)")
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
        "cathode_emissivity": 0.7,
        "phi_wf": 2.869,
        "cathode_surface_model": "ads_des",
        "cathode_phiwf_clean_eV": 2.809,
        "cathode_cleaning_sigma_cm2": 3.5e-16,
        "cathode_cleaning_E_th_eV": 20.0,
        # Explicit campaign choice. The notebook carried this override until
        # 0451c97 replaced its local config with the shared production config;
        # never inherit the shared "local" default here again.
        "Te_birth_ionization": ELECTRON_BIRTH_POLICY,
        "gas_puff_mode": "square",
        "S_gp": args.sgp,
    }
    # Passthrough overrides: absent => inherit the shared production config
    # rather than silently reimposing a stale driver default (7c, 2026-07-27 --
    # the old --g-cond default of 1200 overrode the promoted stance value of
    # 8000 on every run that did not pass the flag).
    if args.g_cond is not None:
        extra["cathode_conduction_W_per_K"] = args.g_cond
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
    for kv in args.extra_flag:
        k, v = kv.split("=", 1)
        try:
            flags_extra[k] = json.loads(v)
        except json.JSONDecodeError:
            flags_extra[k] = v

    result, geometry, params, flags = run_model(
        nx=args.nx, extra=extra,
        flags_extra=flags_extra or None,
    )
    save_result_hdf5(args.save_h5, result, params=params, flags=flags)
    # Standing condition for a DVM-arm run: every report quotes the limited
    # step count and the outstanding debt, and any limited > 0 gets a
    # dedicated look. A moment run has no ledger and prints nothing.
    census = summarize_result(result).dvm_transfer_ledger_census
    if census is not None:
        print(
            "dvm transfer ledger: "
            f"engaged={census['engaged']}, "
            f"relax_steps={census['relax_steps']}, "
            f"relax_limited_steps={census['relax_limited_steps']}, "
            f"limited_cells={census['limited_cells']}, "
            f"Ei_debt_total={census['Ei_debt_total']:.6e} erg "
            f"(max/cell {census['Ei_debt_max_abs']:.6e} erg/cm^3), "
            f"M_debt_total={census['M_debt_total']:.6e} g cm/s "
            f"(max/cell {census['M_debt_max_abs']:.6e} g/(cm^2 s)), "
            f"closure |applied+debt-booked|/scale: "
            f"Ei {census['Ei_residual_rel']:.3e}, "
            f"M {census['M_residual_rel']:.3e}"
        )
        if census["relax_limited_steps"] > 0:
            print(
                "dvm transfer ledger: LIMITED STEPS PRESENT -- the standing "
                "condition calls for a dedicated look at this run"
            )
    print(f"saved {args.save_h5}")


if __name__ == "__main__":
    main()
