"""R5 ES1 tuning-pass iteration driver (density fit + circuit cross-set).

Iteration-grade: nx=60, density_dt_fraction=0.5, t_end=22 ms (discharge only --
the afterglow crawls and stages i+ii don't need it). Uses the ES production
config (compare_sim1d_es1.run_model -> PARAM_OVERRIDES: end-expansion geometry,
tr_bdf2/strang, ADAS, R_comp=5.72 mOhm) with the R5 circuit/thermal overrides:

  --sgp        gas-puff source; raises MID-port density AND current
  --ts-base    cathode standby temperature (cathode_Ts_base_K); a ~29 A/K
               current derivative, but MEASURED, not a tuning knob (the ES
               calibration lives on the effective C_R since 2026-07-29).
               Default = the measured ES standby, no offset.
  --R-comp     TOTAL loop series resistance [Ohm] (sets current)
  --x          R_comp_partition: R_external=x*R_comp (in V_dis),
               R_internal=(1-x)*R_comp (probe->plasma, invisible to V_dis)
  --R-mesh     R_mesh_ohm: separate anode-mesh series R (plasma side)

Neutral seed comes from the signature-keyed DB (--seed-db); a new S_gp is a new
fill (auto-populated, ~16 s at nx60), circuit/Ts refits reuse the entry.

Reports plateau current, V_dis (per-solve + dt-integrated), V_b, and the ES1
mid/far port ne/Te. Score with:
    compare_sim1d_es1.py --from-h5 <out> --nx 60 --es N
    fingerprints_sim1d.py <out>
"""

import argparse
import time as _walltime

import numpy as np

from compare_sim1d_es1 import run_model
from run_mechanism_ladder import ES_OPERATING
from cablp.solvers._sim1d.results.io import save_result_hdf5

# ES1 probe/port layout (compare_sim1d_es1 overlay); z in cm.
MID_PORTS_CM = (470.0, 790.0, 1045.0)   # density-fit targets (~1.0e13)
FAR_PORT_CM = 1716.0                     # diagnostic, NOT a fit target


def _plateau(t_ms, arr, lo=15.0, hi=19.5):
    m = (t_ms >= lo) & (t_ms <= min(hi, t_ms[-1]))
    return float(np.median(arr[m])) if m.any() else float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--es", type=int, choices=(1, 2, 3), default=1)
    ap.add_argument("--nx", type=int, default=60)
    ap.add_argument("--sgp", type=float, required=True)
    ap.add_argument("--ts-base", type=float, default=None,
                    help="cathode_Ts_base_K; default = the MEASURED ES "
                         "standby (no offset)")
    ap.add_argument("--R-comp", type=float, default=None,
                    help="total loop R [Ohm]; default PARAM_OVERRIDES 5.72e-3")
    ap.add_argument("--x", type=float, default=None,
                    help="R_comp_partition (probe fraction); default 1.0")
    ap.add_argument("--R-mesh", type=float, default=None,
                    help="R_mesh_ohm anode-mesh series R [Ohm]; default 0.0")
    ap.add_argument("--c-th", type=float, default=None,
                    help="cathode_heat_capacity_J_per_K (warming TIMESCALE "
                         "tau~C_th/G_cond; default 120 -> tau~100ms >> discharge, "
                         "back-loaded ramp; smaller front-loads it)")
    ap.add_argument("--g-cond", type=float, default=None,
                    help="cathode_conduction_W_per_K (restoring conductance; "
                         "default 1200)")
    ap.add_argument("--L", type=float, default=None,
                    help="L_parasitic_H loop inductance [H] (knee rise time "
                         "tau=L/R_comp; larger L rounds the knee)")
    ap.add_argument("--s-pump", type=float, default=None,
                    help="S_pump_L=S_pump_R [both]; higher = thinner equilibration "
                         "seed (less residual gas for near-source self-ignition)")
    ap.add_argument("--puff-local-ioniz", type=float, default=None,
                    help="gas_puff_local_ionization_fraction: fraction of the "
                         "fresh puff ionized in place (dense clump seed -> front)")
    ap.add_argument("--clump-fraction", type=float, default=None,
                    help="beam_clump_fraction f: flux fraction through dense "
                         "clumps (short l_b, local front seed); rest penetrates")
    ap.add_argument("--clump-enhancement", type=float, default=None,
                    help="beam_clump_enhancement chi: clump nn = chi*background "
                         "(higher = more localized clump deposition)")
    ap.add_argument("--beam-smoothing", type=float, default=None,
                    help="beam_deposition_smoothing_cm: conservative physical "
                         "Gaussian smoothing of the CSDA beam sources [cm] "
                         "(0=off; removes the mesh-scale current-step artifact)")
    ap.add_argument("--density-dt-fraction", type=float, default=0.5)
    ap.add_argument("--t-end", type=float, default=22e-3)
    ap.add_argument("--seed-db", default="scripts/neutral_seed_db")
    ap.add_argument("--two-zone", action="store_true",
                    help="neutral_two_zone particle channel (nn=column, "
                         "nn_a=annulus); momentum stays OFF")
    ap.add_argument("--heat-flux-limit", action="store_true",
                    help="electron_heat_flux_limit (A9): cap the parallel "
                         "electron heat conduction at free-streaming "
                         "(Cowie-McKee); classical is 1.7-3.3x over the ceiling")
    ap.add_argument("--heat-flux-exp", type=float, default=None,
                    help="heat_flux_limiter_exponent p (non-local A9): lambda="
                         "1/(1+Kn^p); p=1 harmonic, p>1 sharper non-local cap")
    ap.add_argument("--heat-flux-f", type=float, default=None,
                    help="heat_flux_limiter_f (free-streaming fraction; default "
                         "0.3; kinetic ~0.1; smaller = harder cap on e- conduction)")
    ap.add_argument("--no-cache", action="store_true",
                    help="run a live neutral equilibration instead of the seed "
                         "DB (needed for a new neutral-flow config, e.g. "
                         "--two-zone, until the auto-populate bug is fixed)")
    ap.add_argument("--baffle-z", default=None,
                    help="comma list of annular-baffle axial positions [cm from "
                         "cathode]; needs --two-zone (restricts the annulus)")
    ap.add_argument("--baffle-r", default=None,
                    help="comma list of baffle clear-aperture radii [cm], one per "
                         "--baffle-z (blocks the annular ring Rp<r<R_clear)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    op = ES_OPERATING[args.es]
    # Default = the MEASURED standby, no offset (cathode calibration
    # reparameterized 2026-07-29: the retired -70 K default reproduced the
    # calibrated stance that now lives on the effective C_R, so leaving it
    # here would double-count the same flat direction on every run that did
    # not pass --ts-base). --ts-base remains available as the explicit
    # stability-derivative probe.
    ts_base = args.ts_base if args.ts_base is not None else op["Ts_standby_K"]

    extra = {
        "V_bank": op["V_bank"],
        "S_gp": args.sgp,
        "cathode_Ts_base_K": ts_base,
        "density_dt_fraction": args.density_dt_fraction,
        # tau_discharge is HARDWARE, not a cost knob: the config.py default
        # 20 ms matches the measured drive (19.97/20.04/20.01 ms on ES1/2/3).
        # It was previously back-derived as max(t_end-2.5e-3, 1e-3), which
        # silently shortened the drive by 0.5 ms (diagnostician, 2026-07-27);
        # run cost is now capped by the explicit t_end passed to run_model.
        "tau_discharge": 20e-3,
        "tau_afterglow": 0.0,
        # Pin the puff to its hardware location: "the physical pipe sits at
        # ~60 cm (anode + 10)" (config.py neutral_defaults docstring). The
        # solver default (None) anchors the cosine_pipe centre to the puff CELL
        # centre, which moves with nx. Pinned at the driver level only -- the
        # solver-default change is deferred to the stance-promotion batch
        # (golden recapture moment). Interim: fixed-cell source region
        # 0-100 cm + pinned puff per Tom 2026-07-27, pending CAD.
        "gas_puff_z_cm": 60.0,
    }
    if args.R_comp is not None:
        extra["R_comp"] = args.R_comp
    if args.x is not None:
        extra["R_comp_partition"] = args.x
    if args.R_mesh is not None:
        extra["R_mesh_ohm"] = args.R_mesh
    if args.c_th is not None:
        extra["cathode_heat_capacity_J_per_K"] = args.c_th
    if args.g_cond is not None:
        extra["cathode_conduction_W_per_K"] = args.g_cond
    if args.beam_smoothing is not None:
        extra["beam_deposition_smoothing_cm"] = args.beam_smoothing
    if args.L is not None:
        extra["L_parasitic_H"] = args.L
    if args.s_pump is not None:
        extra["S_pump_L"] = args.s_pump
        extra["S_pump_R"] = args.s_pump
    if args.puff_local_ioniz is not None:
        extra["gas_puff_local_ionization_fraction"] = args.puff_local_ioniz
    if args.clump_fraction is not None:
        extra["beam_clump_fraction"] = args.clump_fraction
    if args.clump_enhancement is not None:
        extra["beam_clump_enhancement"] = args.clump_enhancement
    if args.baffle_z is not None:
        zs = [float(v) for v in args.baffle_z.split(",")]
        rs = [float(v) for v in args.baffle_r.split(",")]
        if len(zs) != len(rs):
            ap.error("--baffle-z and --baffle-r must have equal length")
        extra["neutral_baffle_positions_cm"] = zs
        extra["neutral_baffle_clear_radii_cm"] = rs
    flags_extra = {}
    if args.baffle_z is not None:
        flags_extra["neutral_baffles"] = True
    if args.two_zone:
        flags_extra["neutral_two_zone"] = True
    if args.heat_flux_limit or args.heat_flux_f is not None:
        flags_extra["electron_heat_flux_limit"] = True
    if args.heat_flux_f is not None:
        extra["heat_flux_limiter_f"] = args.heat_flux_f
    if args.heat_flux_exp is not None:
        extra["heat_flux_limiter_exponent"] = args.heat_flux_exp
        flags_extra["electron_heat_flux_limit"] = True
    if args.no_cache:
        flags_extra["use_cached_neutral_seed"] = False
    else:
        flags_extra["use_cached_neutral_seed"] = True
        extra["neutral_seed_cache_dir"] = args.seed_db

    R_comp = extra.get("R_comp", 5.72e-3)
    print(f"# ES{args.es} V_bank={op['V_bank']} S_gp={args.sgp} "
          f"Ts_base={ts_base:.0f} K R_comp={R_comp:.4g} "
          f"x={extra.get('R_comp_partition', 1.0)} "
          f"R_mesh={extra.get('R_mesh_ohm', 0.0)} nx={args.nx} "
          f"two_zone={args.two_zone} cache={not args.no_cache} "
          f"baffles={args.baffle_z} r={args.baffle_r} -> {args.out}")
    t0 = _walltime.time()
    result, geometry, params, flags = run_model(
        nx=args.nx, extra=extra, flags_extra=flags_extra, t_end=args.t_end)
    wall = _walltime.time() - t0

    t_ms = np.asarray(result.time, float) * 1e3
    diag = result.cathode_diagnostics
    I = np.asarray(diag["source_I_tot"], float)
    Iplat = _plateau(t_ms, I)
    meas_I = {1: 2991.0, 2: 4437.0, 3: 4051.0}[args.es]
    print(f"# wall={wall:.0f}s saves={len(t_ms)} t_end={t_ms[-1]:.2f}ms")
    print(f"# I_plateau(15-19.5ms)={Iplat:.0f} A  measured {meas_I:.0f} A  "
          f"ratio {Iplat/meas_I:.3f}")

    Vb = _plateau(t_ms, np.asarray(diag["source_V_b"], float))
    Vd_step = _plateau(t_ms, np.asarray(diag["circuit_V_dis_step"], float))
    print(f"# V_b_plateau={Vb:.1f} V   V_dis(per-solve step)={Vd_step:.1f} V")
    # Derive the probe partition x that a target V_dis would imply:
    #   V_dis = V_bank - I*(x*R_comp) - L*dI/dt  ->  ignoring L at plateau,
    #   x = (V_bank - V_dis) / (I * R_comp)
    if Iplat and Iplat == Iplat:
        for vtgt in (151.0, 153.0, 155.0):
            xneed = (op["V_bank"] - vtgt) / (Iplat * R_comp)
            print(f"#   x for V_dis={vtgt:.0f} V: {xneed:.3f}")

    # ES1 port densities/temperatures at the plateau (raw z, no origin shift --
    # iteration diagnostic only; the scorer does the origin-relative comparison).
    z = np.asarray(result.z_cm, float)
    n2d = np.asarray(result.n, float)
    Te2d = np.asarray(result.Te, float)
    print("# port     z[cm]     ne[cm-3]     Te[eV]")
    for zc in (*MID_PORTS_CM, FAR_PORT_CM):
        iz = int(np.argmin(np.abs(z - zc)))
        tag = "FAR " if zc == FAR_PORT_CM else "mid "
        print(f"#   {tag} {zc:7.0f}  {_plateau(t_ms, n2d[:, iz]):.3e}  "
              f"{_plateau(t_ms, Te2d[:, iz]):.2f}")

    save_result_hdf5(args.out, result, params=params, flags=flags)
    print(f"# saved {args.out}")


if __name__ == "__main__":
    main()
