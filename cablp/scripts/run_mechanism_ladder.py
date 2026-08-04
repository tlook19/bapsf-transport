"""Run the candidate config with a chosen warming mechanism across the ladder.

Mechanism-campaign driver: candidate
config = current_driven + manifold-era csda_ql deposition + gaussian
emission profile + Schottky (flag default), with per-campaign inputs
restricted to the MEASURED pair (open-circuit V0; Fig-10 standby T_s) --
transfer across the ladder is the test, so everything else is frozen at
the ES1 calibration.

    python scripts/run_mechanism_ladder.py --es 1 --warming power_balance \
        --phi-wf 2.87 --g-cond 1500 --c-th 120 --save-h5 out.h5
"""

import argparse

from compare_sim1d_es1 import run_model
from cablp.solvers._sim1d.results.io import save_result_hdf5

# Per-campaign operating points: the open-circuit bank voltage V0 and the
# Fig-10 digitized standby surface temperature.
#
# V_bank provenance, corrected 2026-08-03: all four are MEASURED pre-shot
# open-circuit readings on the same probe channel as V_dis (+-0.03 V SEM, with
# a +-1.2% multiplicative instrumental systematic that is unresolved between
# supply regulation and probe gain). The previous comment here called them
# "fitted/nominal" in one breath and mixed conventions: ES1's old 173.6 was
# FITTED (from the near-singular ES1-only circuit fit -- see the PARAM_OVERRIDES
# comment in compare_sim1d_es1.py), while ES2/ES3's were measured. They are now
# uniformly measured. NB ES3 and ES4 previously shared the literal 99.0 and now
# DIFFER -- ES4's reading is window-corrected; do not re-collapse them.
ES_OPERATING = {
    1: {"V_bank": 177.843, "Ts_standby_K": 1910.0},
    2: {"V_bank": 138.303, "Ts_standby_K": 1949.0},
    3: {"V_bank": 98.814, "Ts_standby_K": 1972.0},
    # ES4 (detachment-exacerbation): heater identical to ES3 and the bank set
    # to the same dial; only the puff drive differs (110 V vs the ladder's
    # 76.4 V). The measured V0 differs from ES3 by 0.164 V.
    4: {"V_bank": 98.978, "Ts_standby_K": 1972.0},
}

ELECTRON_BIRTH_POLICY = "floor"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--es", type=int, choices=(1, 2, 3), required=True)
    p.add_argument("--nx", type=int, default=120)
    p.add_argument(
        "--warming",
        choices=("none", "power_balance"),
        default="power_balance",
    )
    p.add_argument("--phi-wf", type=float, default=None,
                   help="shared work function [eV] (emission, Schottky, "
                        "cooling, gaussian inversion -- one constant)")
    p.add_argument("--g-cond", type=float, default=1500.0,
                   help="skin->substrate conduction [W/K] (the one fitted "
                        "knob; frozen after ES1)")
    p.add_argument("--c-th", type=float, default=120.0,
                   help="skin-layer heat capacity [J/K] (honest >=120, "
                        "per the surface energy budget)")
    p.add_argument("--emissivity", type=float, default=0.7)
    p.add_argument("--annuli", type=int, default=None,
                   help="cathode_emission_annuli override (10 -> 30 A/B)")
    p.add_argument("--standby-offset-K", type=float, default=0.0,
                   help="offset added to the measured standby T_s [K] "
                        "(the +-8 K trim-quantum stability-derivative "
                        "probe; NOT a tuning knob)")
    p.add_argument("--surface", action="store_true",
                   help="enable cathode_surface_model='ads_des' (M5a: "
                        "in-shot fluence-cleaning limit)")
    p.add_argument("--phiwf-clean", type=float, default=None,
                   help="per-shot-accessible clean floor phi_clean [eV] "
                        "(required with --surface)")
    p.add_argument("--sigma-clean", type=float, default=0.0,
                   help="ion-stimulated desorption cross section [cm^2]")
    p.add_argument("--E-th", type=float, default=None,
                   help="desorption threshold [eV] (M5a' Bohdansky yield "
                        "factor; omit for the energy-independent limit)")
    p.add_argument("--bridge", action="store_true",
                   help="enable the kT_s emission-release thermal bridge")
    p.add_argument("--mn", action="store_true",
                   help="evolved neutral momentum closure for the jet "
                        "campaign: neutral_momentum + two-zone radial + "
                        "honest b=1 drag + mesh momentum accommodation "
                        "(ALL arms of the jet A/B carry this, reference "
                        "included, so the A/B isolates the jet)")
    p.add_argument("--jet", choices=("cathode", "both"), default=None,
                   help="directed recycle jets "
                        "at literature-boxed (R_N, R_E); requires --mn")
    p.add_argument("--cjet-RN", type=float, default=None,
                   help="cathode jet particle reflection coefficient "
                        "(default the mid box 0.5)")
    p.add_argument("--cjet-RE", type=float, default=None,
                   help="cathode jet energy reflection coefficient "
                        "(default the mid box 0.2)")
    p.add_argument("--ajet-RN", type=float, default=None,
                   help="anode jet particle reflection coefficient "
                        "(default the He->Mo class 0.5)")
    p.add_argument("--ajet-RE", type=float, default=None,
                   help="anode jet energy reflection coefficient "
                        "(default the He->Mo class 0.25)")
    p.add_argument("--jet-debit", action="store_true",
                   help="sensitivity arm: debit the cathode surface's ion "
                        "heating by the reflected-energy fraction "
                        "((1-R_E)*P_i into power_balance)")
    p.add_argument("--square", action="store_true",
                   help="measured square valve waveform (M6): erf rise at "
                        "circuit-on, flat S_gp for the drive, erf close + "
                        "afterglow tail; replaces pulse_decay_to_level")
    p.add_argument("--sgp", type=float, default=None,
                   help="gas puff level [sccm/valve] (the M6 single "
                        "calibration knob; default keeps PARAM_OVERRIDES)")
    p.add_argument("--smooth", action="store_true",
                   help="electrode sample smoothing at the presheath "
                        "transit time (cathode + anode-flank EMA)")
    p.add_argument("--rec-return", action="store_true",
                   help="enable recombination_energy_return (the GCR pair "
                        "+I_ion*S_rec - P_PRB on the electron fluid; the "
                        "stage-(iii) afterglow slowdown candidate)")
    p.add_argument("--save-h5", required=True)
    args = p.parse_args(argv)

    op = dict(ES_OPERATING[args.es])
    op["Ts_standby_K"] = op["Ts_standby_K"] + float(args.standby_offset_K)
    extra = {
        "nx": args.nx,
        "V_bank": op["V_bank"],
        "cathode_solver_model": "current_driven",
        "beam_deposition_model": "csda",
        "beam_anomalous_model": "quasilinear",
        "cathode_emission_profile": "gaussian",
        "cathode_warming_model": args.warming,
        # Explicit campaign choice restored after the notebook override was
        # omitted by 0451c97's shared-config migration.
        "Te_birth_ionization": ELECTRON_BIRTH_POLICY,
    }
    if args.warming == "power_balance":
        extra.update({
            "T_s": op["Ts_standby_K"],
            "cathode_Ts_base_K": op["Ts_standby_K"],
            "cathode_heat_capacity_J_per_K": args.c_th,
            "cathode_conduction_W_per_K": args.g_cond,
            "cathode_emissivity": args.emissivity,
        })
    else:
        extra["T_s"] = op["Ts_standby_K"]
    if args.phi_wf is not None:
        extra["phi_wf"] = args.phi_wf
    if args.annuli is not None:
        extra["cathode_emission_annuli"] = args.annuli
    if args.surface:
        if args.phiwf_clean is None:
            raise SystemExit("--surface requires --phiwf-clean")
        extra.update({
            "cathode_surface_model": "ads_des",
            "cathode_phiwf_clean_eV": args.phiwf_clean,
            "cathode_cleaning_sigma_cm2": args.sigma_clean,
        })
        if args.E_th is not None:
            extra["cathode_cleaning_E_th_eV"] = args.E_th

    flags_extra = {}
    if args.bridge:
        flags_extra["cathode_emission_bridge"] = True
    if args.jet and not args.mn:
        raise SystemExit("--jet requires --mn (the jet is an M_n source)")
    if args.jet_debit and args.jet is None:
        raise SystemExit("--jet-debit requires --jet")
    if args.mn:
        extra.update({
            "ion_neutral_drag_model": "constant",
            "b_ion_neutral_drag": 1.0,
            "neutral_momentum_radial": "two_zone",
            "neutral_mesh_accommodation": True,
        })
        flags_extra["neutral_momentum"] = True
    if args.jet:
        extra["cathode_neutral_jet"] = True
        if args.cjet_RN is not None:
            extra["cathode_jet_R_N"] = args.cjet_RN
        if args.cjet_RE is not None:
            extra["cathode_jet_R_E"] = args.cjet_RE
        if args.jet == "both":
            extra["anode_neutral_jet"] = True
            if args.ajet_RN is not None:
                extra["anode_jet_R_N"] = args.ajet_RN
            if args.ajet_RE is not None:
                extra["anode_jet_R_E"] = args.ajet_RE
        if args.jet_debit:
            extra["cathode_jet_surface_debit"] = True
    if args.rec_return:
        extra["recombination_energy_return"] = True
    if args.square:
        extra["gas_puff_mode"] = "square"
    if args.sgp is not None:
        extra["S_gp"] = args.sgp
    if args.smooth:
        extra["cathode_sample_smoothing"] = "presheath"

    result, geometry, params, flags = run_model(
        nx=args.nx, extra=extra, flags_extra=flags_extra or None
    )
    save_result_hdf5(args.save_h5, result, params=params, flags=flags)
    print(f"saved {args.save_h5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
