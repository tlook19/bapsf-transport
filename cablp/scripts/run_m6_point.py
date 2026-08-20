"""Run one M6 candidate point with full extra-param control (no driver edits).

Thin wrapper around compare_sim1d_es1.run_model for M6 recalibration points
that need params the ladder driver does not expose (close lag, L, etc.),
kept as a separate file so concurrent sessions editing the shared driver
and solver are never touched.

The configuration package is named, not accreted: ``--stance NAME`` applies a
committed stance file (``scripts/stances/NAME.toml``) as the base config, and
``--extra`` / ``--extra-flag`` still layer on top of it and say so. Naming the
package is mandatory -- a run either names its stance or acknowledges that it
has none with ``--no-stance``.

Usage:
    python scripts/run_m6_point.py --es 1 --stance g1atrim --sgp 9010 \
        --save-h5 out.h5
    python scripts/run_m6_point.py --es 1 --no-stance --sgp 3400 \
        --close-lag 2e-3 --save-h5 out.h5 [--mn] [--L 8.1e-6] [--extra k=v ...]
"""

import argparse
import json

import numpy as np

from compare_sim1d_es1 import PRODUCTION_NX, run_model
from run_mechanism_ladder import ES_OPERATING
from stance_config import available_stances, load_stance
from cablp.solvers._sim1d.results.io import save_result_hdf5
from cablp.solvers._sim1d.results.health import summarize_result


ELECTRON_BIRTH_POLICY = "floor"


def _brief_value(value):
    """Return a short repr of a config value for a one-line console message.

    Per-cell profiles are hundreds of entries long; they report as their type
    and length so a stance banner stays readable.
    """
    shown = repr(value)
    if len(shown) <= 48:
        return shown
    length = getattr(value, "__len__", None)
    if length is not None:
        return f"<{type(value).__name__} len={len(value)}>"
    return shown[:45] + "..."


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--es", type=int, choices=(1, 2, 3), required=True)
    stance_group = p.add_mutually_exclusive_group()
    stance_group.add_argument(
        "--stance", metavar="NAME", default=None,
        help="committed stance file (scripts/stances/NAME.toml) applied as the "
             "base config, on top of this driver's own defaults and below "
             "--nn0-profile-npz / --extra / --extra-flag. Available: "
             + (", ".join(available_stances()) or "(none committed)"))
    stance_group.add_argument(
        "--no-stance", action="store_true",
        help="acknowledge that this run names no stance and is configured "
             "entirely by this driver's defaults plus the explicit overrides "
             "on this command line")
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
    p.add_argument("--nn0-profile-npz", default=None,
                   help="path to a shaped initial neutral profile written by "
                        "scripts/sp3_build_nn0.py. The DRIVER does the file "
                        "I/O -- the solver never opens a file -- and passes "
                        "the arrays in as input_dict values. It arms the "
                        "neutral_initial_profile flag, sets nn0=None (the "
                        "scalar is superseded, and the solver refuses an "
                        "armed flag alongside an explicit scalar), and passes "
                        "nn0_annulus_profile too when the npz carries one. "
                        "NOT set here: the stance ships neutral_equilibration "
                        "ON and the solver REFUSES it with a shaped IC (the "
                        "seed would overwrite the profile), so a shaped run "
                        "also passes --extra-flag neutral_equilibration=false "
                        "-- a stance delta the arm states rather than "
                        "inherits. Applied AFTER --stance and BEFORE "
                        "--extra/--extra-flag, so it overrides a stance's own "
                        "shaped fill and either of those can still override "
                        "any of it")
    p.add_argument("--extra", nargs="*", default=(),
                   help="additional k=v param overrides (JSON-parsed values)")
    p.add_argument("--extra-flag", nargs="*", default=(),
                   help="additional k=v input_flags overrides (JSON-parsed)")
    p.add_argument("--max-steps", type=int, default=None,
                   help="accepted-step cap; absent (default) is the "
                        "historical uncapped run. Pair it with "
                        "--extra max_steps_action=stop to end cleanly and "
                        "SAVE the partial trajectory at the cap -- on its own "
                        "the cap raises RuntimeError and the h5 is lost")
    p.add_argument("--save-h5", required=True)
    args = p.parse_args(argv)
    # Agent-safety rider: the configuration package a run carries is stated,
    # never inherited by silence. The mis-configured launch this closes is the
    # one that reads as a full stance on the command line while quietly
    # standing on whatever the shared driver dicts happened to hold.
    if args.stance is None and not args.no_stance:
        raise SystemExit(
            "run_m6_point: name the configuration package. Pass "
            "--stance <name> to run a committed stance file "
            f"(available: {', '.join(available_stances()) or '(none committed)'})"
            ", or --no-stance to acknowledge that this run has none and is "
            "configured by this driver's defaults plus the overrides on this "
            "command line."
        )

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
    stance = None
    if args.stance is not None:
        stance = load_stance(args.stance)
        superseded = [
            f"{key}: {_brief_value(extra[key])} -> "
            f"{_brief_value(value)}"
            for key, value in sorted(stance.params.items())
            if key in extra and extra[key] != value
        ] + [
            f"flags:{key}: {_brief_value(flags_extra[key])} -> "
            f"{_brief_value(value)}"
            for key, value in sorted(stance.flags.items())
            if key in flags_extra and flags_extra[key] != value
        ]
        extra.update(stance.params)
        flags_extra.update(stance.flags)
        print(
            f"stance {stance.name} from {stance.path}: "
            f"{len(stance.params)} params, {len(stance.flags)} flags"
        )
        for line in superseded:
            print(f"  stance supersedes this driver's default {line}")
    # Keys this command line supplies above the stance layer, for the
    # departure report below.
    cli_supplied = set()
    cli_supplied_flags = set()
    if args.nn0_profile_npz is not None:
        with np.load(args.nn0_profile_npz, allow_pickle=False) as data:
            if "nn0_profile" not in data:
                raise ValueError(
                    f"{args.nn0_profile_npz} carries no 'nn0_profile' array; "
                    "it is not a shaped-initial-fill npz"
                )
            extra["nn0_profile"] = np.asarray(
                data["nn0_profile"], dtype=float
            ).tolist()
            if "nn0_annulus_profile" in data:
                extra["nn0_annulus_profile"] = np.asarray(
                    data["nn0_annulus_profile"], dtype=float
                ).tolist()
            provenance = (
                str(data["provenance"]) if "provenance" in data else "(absent)"
            )
        extra["nn0"] = None
        flags_extra["neutral_initial_profile"] = True
        cli_supplied.update(("nn0", "nn0_profile", "nn0_annulus_profile"))
        cli_supplied_flags.add("neutral_initial_profile")
        print(f"shaped nn0 from {args.nn0_profile_npz}: {provenance}")
    for kv in args.extra:
        k, v = kv.split("=", 1)
        try:
            extra[k] = json.loads(v)
        except json.JSONDecodeError:
            extra[k] = v
        cli_supplied.add(k)
    for kv in args.extra_flag:
        k, v = kv.split("=", 1)
        try:
            flags_extra[k] = json.loads(v)
        except json.JSONDecodeError:
            flags_extra[k] = v
        cli_supplied_flags.add(k)
    if stance is not None:
        # Every layer above the stance is reported as a DEPARTURE: a run that
        # cites a stance by name must not carry unstated overrides on top of
        # it. Both a changed stance key and a key the stance does not name at
        # all count -- the second is how the shaped-fill and run-cost keys get
        # onto an arm.
        departures = [
            f"{key}: {_brief_value(stance.params[key])} -> "
            f"{_brief_value(extra[key])}"
            for key in sorted(stance.params)
            if extra[key] != stance.params[key]
        ] + [
            f"flags:{key}: {_brief_value(stance.flags[key])} -> "
            f"{_brief_value(flags_extra[key])}"
            for key in sorted(stance.flags)
            if flags_extra[key] != stance.flags[key]
        ] + [
            f"{key}: <not in stance> -> {_brief_value(extra[key])}"
            for key in sorted(cli_supplied)
            if key in extra and key not in stance.params
        ] + [
            f"flags:{key}: <not in stance> -> {_brief_value(flags_extra[key])}"
            for key in sorted(cli_supplied_flags)
            if key in flags_extra and key not in stance.flags
        ]
        if departures:
            print(
                f"WARNING: this run DEPARTS stance {stance.name} -- "
                f"{len(departures)} override(s) applied on top of it:"
            )
            for line in departures:
                print(f"  WARNING: departs {stance.name} {line}")

    result, geometry, params, flags = run_model(
        nx=args.nx, extra=extra,
        flags_extra=flags_extra or None,
        max_steps=args.max_steps,
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
        if "ion_residual_rel" in census:
            print(
                "dvm particle handshake: "
                f"booked={census['ion_booked_total']:.6e}, "
                f"removed={census['ion_removed_total']:.6e}, "
                f"debt={census['ion_debt_total']:.6e} particles "
                f"(max/cell {census['ion_debt_max_abs']:.6e}), "
                f"shortfall_updates={census['ion_shortfall_updates']}, "
                f"closure |removed+debt-booked|/scale: "
                f"{census['ion_residual_rel']:.3e}"
            )
        if census["relax_limited_steps"] > 0:
            print(
                "dvm transfer ledger: LIMITED STEPS PRESENT -- the standing "
                "condition calls for a dedicated look at this run"
            )
    print(f"saved {args.save_h5}")


if __name__ == "__main__":
    main()
