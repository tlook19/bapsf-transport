"""Compare a sim1d run against the ES1 experimental overlay.

The point of the model is to reproduce measured behaviour, so the meaningful
benchmark is the data, not a previous version of the model. This samples the
simulation at the five ES1 probe locations and the experimental time base, and
reports the deviation in electron temperature and density against the measured
means and their standard errors.

Comparisons are made on the main-discharge clock, matching the notebook: model
time is shifted so t = 0 is the start of the main discharge, which is what the
experimental ``*_time_ms`` axes are referenced to.

Three comparison stages, in tuning order (each scored independently):

(i)   peak discharge current -- model ``source_I_tot`` vs the measured
      discharge-trace peak. Peak only: the breakdown *rate* is shaped by
      physics the model does not carry, so trace RMS is deliberately not a
      target.
(ii)  bulk Te and density at the five ES1 ports (the original comparison).
(iii) afterglow decay -- per-port e-folding time of the model Isat proxy
      ``n * sqrt(Te)`` against the measured Isat decay traces, over a fit
      window on the main-discharge clock. The default run carries only
      ``tau_afterglow = 5 ms`` past discharge end; use ``--tau-afterglow``
      to extend it toward the measured 27.5 ms tail.

Usage::

    python scripts/compare_sim1d_es1.py                      # resolved + knudsen
    python scripts/compare_sim1d_es1.py --nx 185
    python scripts/compare_sim1d_es1.py --save-h5 run.h5     # keep the run
    python scripts/compare_sim1d_es1.py --from-h5 run.h5     # re-score, no run
    python scripts/compare_sim1d_es1.py --tau-afterglow 0.0275 \
        --decay-window 20.5 40.0
"""

import argparse
from pathlib import Path

import numpy as np

from cablp.solvers._sim1d import (
    BreakdownError,
    LAPDSim1D,
    default_config,
    load_result_hdf5,
)
from cablp.solvers._sim1d.results.io import save_result_hdf5

OVERLAY = Path(__file__).resolve().parent / "data" / "es1_sim1d_overlay.npz"

PARAM_OVERRIDES = {
    # Discharge circuit fitted from the ES1 trace itself (0.14 V rms;
    # scripts/fit_es1_circuit.py): Thevenin V0/R/L/C_eff. Supersedes the
    # nominal V_bank=180 / inferred R_comp=0.010; C_eff carries a hardware
    # caveat (nominal bank <= 4 F) documented in THESIS_NOTES section 2.
    "V_bank": 173.6,
    "R_comp": 5.72e-3,
    "L_parasitic_H": 6.6e-6,
    "C_bank_F": 8.9,
    "T_s": 273.15 + 1725,
    "S_gp": 3000,
    "S_gp_decay_target": 2000,
    "tau_gp_pulse_duration": 1e-3,
    "tau_gp_decay_duration": 5e-3,
    # Ion-neutral closure (2026-07-18 decision): constant drag at the
    # calibrated 0.5 -- a stand-in for the missing neutral-momentum /
    # radial channel, NOT validated physics (THESIS_NOTES gate #2; the
    # "slip" closure is the physical alternative and under-confines).
    # Thermalization is decoupled from the drag scalar, and the
    # momentum-transfer rate is CX-derived rather than a constant sigma.
    "b_ion_neutral_drag": 0.5,
    "ion_neutral_drag_model": "constant",
    "b_ion_neutral_thermalization": 1.0,
    "sigma_in_model": "cx_derived",
    # ADAS GCR rates (see cablp/vars/adas/README.md): effective ionization/
    # recombination and radiation-only cooling, consistent with the separate
    # ionization-cost term. b_Q* = 1 is meaningful under this model.
    "atomic_rate_model": "adas",
    # Beam-driven neutral excitation: 1.0 books the 2^1P channel alone, the
    # extra 0.4 approximates the rest of the singlet manifold. Radiates ~21 eV
    # per event as He I light and shortens the beam deposition length.
    "b_beam_excitation": 1.4,
    "b_Qei": 1,
    "b_Qen": 1,
    "b_Qcx": 1,
    "Rp": 15.0,
    "R_cath": 15.0,
    "implicit_heat_scheme": "tr_bdf2",
    "operator_splitting": "strang",
    "heat_picard_iterations": 2,
    "heat_picard_tol": 1e-10,
}
FLAG_OVERRIDES = {
    "ion_neutral_drag_cx_only": False,
    "ion_neutral_thermalization": True,
}


def _main_discharge_origin(result):
    """Return the model time [s] at which the main discharge begins."""
    phases = np.asarray(getattr(result, "phase", ()), dtype=str)
    times = np.asarray(result.time, dtype=float)
    hits = np.flatnonzero(phases == "main_discharge")
    return float(times[hits[0]]) if hits.size else float(times[0])


def run_model(
    nx=None,
    exchange_model="knudsen",
    extra=None,
    drag_closure=None,
    Rp_model=None,
    flags_extra=None,
):
    params, flags = default_config()
    params.update(PARAM_OVERRIDES)
    flags.update(FLAG_OVERRIDES)
    if flags_extra:
        flags.update(flags_extra)
    params["neutral_exchange_model"] = exchange_model
    if nx is not None:
        params["nx"] = nx
    # A/B instrument for THESIS_NOTES gate #2 (NEUTRAL_MOMENTUM_PLAN.md M4):
    # swap the drag closure without touching the rest of the production
    # config. "constant" is PARAM_OVERRIDES as-is (the calibrated 0.5);
    # "slip" is the entrainment closure; "neutral_momentum" evolves M_n with
    # the honest b = 1 (the field replaces the compensation constant).
    if drag_closure == "slip":
        params["ion_neutral_drag_model"] = "slip"
        params["b_ion_neutral_drag"] = 1.0
    elif drag_closure == "neutral_momentum":
        params["ion_neutral_drag_model"] = "constant"
        params["b_ion_neutral_drag"] = 1.0
        flags["neutral_momentum"] = True
    elif drag_closure == "neutral_momentum_two_zone":
        params["ion_neutral_drag_model"] = "constant"
        params["b_ion_neutral_drag"] = 1.0
        params["neutral_momentum_radial"] = "two_zone"
        flags["neutral_momentum"] = True
    elif drag_closure not in (None, "constant"):
        raise ValueError(f"unknown drag_closure {drag_closure!r}")
    # A/B instrument for CATHODE_IDRIVEN_PLAN.md M1: profile-integrated
    # cathode-anode gap resistance vs the historical single-sample R_p.
    # With the production Rp == R_cath the geometric component vanishes, so
    # this isolates the Te-profile effect on V_dis(t).
    if Rp_model is not None:
        params["cathode_Rp_model"] = Rp_model
    if extra:
        params.update(extra)
    sim = LAPDSim1D(params, flags)
    sim.start_simulation(t_end=None, dt=None, operator_split=None, max_steps=None)
    return sim.get_results(), sim.geometry, params, flags


# --- Measurement error model (adopted 2026-07-22, conservative per Tom:
# "assume experimental errors can be on the large side").  Shot-to-shot SEM
# measures precision only; the sweep systematics dominate:
#   sigma_Te,sys = 0.25*Te + 0.20 eV   (fit-window, EEDF tail, sheath
#       expansion, magnetization, fluctuation smearing, surface drift --
#       added generously; Te < 1 eV is flagged SEMI-QUANTITATIVE, where the
#       transition spans < 1 V and the budget approaches order unity)
#   sigma_n,sys  = n * sqrt((0.5*sigma_Te/Te)^2 + 0.10^2)   (the c_s
#       inversion propagates half the fractional Te error, anti-correlated;
#       10 % interferometer calibration + transfer)
# Deviations are reported against sigma_tot = sqrt(SEM^2 + sigma_sys^2).
# NB the dominant biases push LP Te HIGH and hence inverted n LOW -- the
# model-hot / model-underdense residuals are, if anything, understated.
# The "Isat" rows compare in I_sat space (n*sqrt(Te), both sides), where
# the sweep inversion cancels identically -- the systematics-robust
# magnitude/shape observable (the stage-(iii) tau metric already lives
# there by design).
TE_SYS_FRAC = 0.25
TE_SYS_FLOOR_EV = 0.20
TE_SEMIQUANT_EV = 1.0
N_CAL_FRAC = 0.10


def _sigma_sys(field, exp_values):
    exp_values = np.asarray(exp_values, dtype=float)
    if field == "Te":
        return TE_SYS_FRAC * np.abs(exp_values) + TE_SYS_FLOOR_EV
    if field == "n":
        # 0.5 * sigma_Te/Te with sigma_Te evaluated at the measured Te is
        # applied by the caller (needs the Te trace); this is the
        # calibration part only.
        return N_CAL_FRAC * np.abs(exp_values)
    return np.zeros_like(exp_values)


def compare(result, geometry, overlay):
    """Return per-port deviation of model Te and density from the ES1 means."""
    z_probe = np.asarray(overlay["z_cm"], dtype=float)
    ports = np.asarray(overlay["port"])
    origin = _main_discharge_origin(result)
    t_model_ms = (np.asarray(result.time, dtype=float) - origin) * 1.0e3
    z_model = np.asarray(result.z_cm, dtype=float)

    # Interpolate measured Te onto each field's own time base for the
    # systematic-error propagation (n's sigma_sys needs Te) and the
    # I_sat-space synthesis.
    te_t = np.asarray(overlay["te_time_ms"], dtype=float)
    te_mean_2d = np.asarray(overlay["te_mean_ev"], dtype=float)

    rows = []
    for field, t_key, mean_key, sem_key, unit in (
        ("Te", "te_time_ms", "te_mean_ev", "te_sem_ev", "eV"),
        ("n", "density_time_ms", "density_mean_cm3", "density_total_sem_cm3", "cm^-3"),
        # I_sat space: n*sqrt(Te) on both sides -- the sweep inversion
        # cancels identically on the measured side, so this row carries
        # only SEM + the interferometer calibration.
        ("Isat", "density_time_ms", "density_mean_cm3", "density_total_sem_cm3", "a.u."),
    ):
        t_exp = np.asarray(overlay[t_key], dtype=float)
        mean = np.asarray(overlay[mean_key], dtype=float)
        sem = np.asarray(overlay[sem_key], dtype=float)
        if field == "Isat":
            model_2d = np.asarray(result.n, dtype=float) * np.sqrt(
                np.maximum(np.asarray(result.Te, dtype=float), 0.0)
            )
        else:
            model_2d = np.asarray(getattr(result, field), dtype=float)
        # Only compare where the experiment has data and the model has run.
        window = (t_exp >= t_model_ms.min()) & (t_exp <= t_model_ms.max())
        for p, (z, port) in enumerate(zip(z_probe, ports)):
            iz = int(np.argmin(np.abs(z_model - z)))
            model_t = np.interp(t_exp[window], t_model_ms, model_2d[:, iz])
            exp_t = mean[p, window]
            sem_t = sem[p, window]
            te_exp_t = np.interp(t_exp[window], te_t, te_mean_2d[p])
            te_safe = np.maximum(np.abs(te_exp_t), 1e-3)
            if field == "Isat":
                exp_t = exp_t * np.sqrt(te_safe)
                # SEM propagated; systematics: calibration only (the
                # c_s inversion cancels in n*sqrt(Te)).
                sem_t = sem_t * np.sqrt(te_safe)
                sys_t = N_CAL_FRAC * np.abs(exp_t)
            elif field == "Te":
                sys_t = _sigma_sys("Te", exp_t)
            else:
                sig_te = _sigma_sys("Te", te_exp_t)
                sys_t = np.abs(exp_t) * np.sqrt(
                    (0.5 * sig_te / te_safe) ** 2 + N_CAL_FRAC**2
                )
            err_tot = np.sqrt(sem_t**2 + sys_t**2)
            good = np.isfinite(exp_t) & np.isfinite(model_t) & (exp_t != 0.0)
            if not np.any(good):
                continue
            ratio = float(np.mean(model_t[good] / exp_t[good]))
            rel = float(np.sqrt(np.mean(((model_t - exp_t)[good] / exp_t[good]) ** 2)))
            sigma = float(np.mean(np.abs((model_t - exp_t)[good] / err_tot[good])))
            semiquant = field in ("Te", "n") and float(
                np.mean(te_exp_t[good])
            ) < TE_SEMIQUANT_EV
            rows.append(
                {
                    "field": field,
                    "unit": unit,
                    "port": str(port),
                    "z": float(z),
                    "model": float(np.mean(model_t[good])),
                    "exp": float(np.mean(exp_t[good])),
                    "ratio": ratio,
                    "rms_rel": rel,
                    "sigma": sigma,
                    "semiquant": bool(semiquant),
                }
            )
    return rows


def compare_peak_current(result, overlay):
    """Return the stage (i) figure of merit: model vs measured peak current."""
    diag = getattr(result, "cathode_diagnostics", None) or {}
    I_model = np.asarray(diag.get("source_I_tot", ()), dtype=float)
    t_model_ms = (
        np.asarray(result.time, dtype=float) - _main_discharge_origin(result)
    ) * 1.0e3
    t_exp = np.asarray(overlay["discharge_time_ms"], dtype=float)
    I_exp = np.asarray(overlay["discharge_current_mean_a"], dtype=float)
    sem_exp = np.asarray(overlay["discharge_current_sem_a"], dtype=float)

    out = {"model_peak_a": np.nan, "model_peak_t_ms": np.nan}
    if I_model.size and np.any(np.isfinite(I_model)):
        i_peak = int(np.nanargmax(I_model))
        out["model_peak_a"] = float(I_model[i_peak])
        out["model_peak_t_ms"] = float(t_model_ms[i_peak])
    j_peak = int(np.nanargmax(I_exp))
    out["exp_peak_a"] = float(I_exp[j_peak])
    out["exp_peak_t_ms"] = float(t_exp[j_peak])
    out["exp_peak_sem_a"] = float(sem_exp[j_peak])
    out["ratio"] = out["model_peak_a"] / out["exp_peak_a"]

    # Late-window plateau (15-19.5 ms): the model's early transient carries a
    # known V_dis(t)-trajectory artifact (THESIS_NOTES section 2), so the
    # established current scale is better read from the end of the drive.
    late = (15.0, 19.5)
    m_model = (t_model_ms >= late[0]) & (t_model_ms <= late[1])
    m_exp = (t_exp >= late[0]) & (t_exp <= late[1])
    out["model_late_a"] = (
        float(np.nanmean(I_model[m_model])) if np.any(m_model) else np.nan
    )
    out["exp_late_a"] = float(np.nanmean(I_exp[m_exp])) if np.any(m_exp) else np.nan
    out["late_ratio"] = out["model_late_a"] / out["exp_late_a"]
    return out


def _efold_time_ms(t_ms, y, floor=0.0):
    """Return the log-linear e-folding decay time [ms] of ``y`` over ``t_ms``.

    Positive for a decaying signal. NaN when fewer than 8 samples survive the
    positivity/noise-floor mask, or when the fitted slope is not a decay.
    """
    t_ms = np.asarray(t_ms, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(t_ms) & np.isfinite(y) & (y > max(floor, 0.0))
    if np.count_nonzero(good) < 8:
        return np.nan
    slope = np.polyfit(t_ms[good], np.log(y[good]), 1)[0]
    return -1.0 / slope if slope < 0.0 else np.nan


def compare_decay(result, overlay, window_ms=(20.5, 25.0)):
    """Return per-port stage (iii) rows: model vs measured Isat e-fold times.

    The model Isat proxy is ``n * sqrt(Te)`` at the port cell (the Bohm-flux
    scaling; constants cancel in an e-folding time). Both signals get the same
    log-linear fit over the same window. The experimental noise floor is
    estimated from the final 5 ms of each decay trace (5x its robust sigma).
    """
    t0, t1 = float(window_ms[0]), float(window_ms[1])
    origin = _main_discharge_origin(result)
    t_model_ms = (np.asarray(result.time, dtype=float) - origin) * 1.0e3
    z_model = np.asarray(result.z_cm, dtype=float)
    n_model = np.asarray(result.n, dtype=float)
    Te_model = np.asarray(result.Te, dtype=float)

    t_exp = np.asarray(overlay["isat_decay_time_ms"], dtype=float)
    isat = np.asarray(overlay["isat_decay_mean_a"], dtype=float)
    ports = np.asarray(overlay["isat_decay_port"])
    z_ports = {
        int(p): float(z)
        for p, z in zip(np.asarray(overlay["port"]), overlay["z_cm"])
    }

    t1_model = min(t1, float(t_model_ms.max()))
    rows = []
    for p in range(ports.size):
        z = z_ports.get(int(ports[p]))
        if z is None:
            continue
        exp_window = (t_exp >= t0) & (t_exp <= t1)
        tail = isat[p, t_exp >= t_exp.max() - 5.0]
        noise = 5.0 * 1.4826 * np.nanmedian(np.abs(tail - np.nanmedian(tail)))
        tau_exp = _efold_time_ms(t_exp[exp_window], isat[p, exp_window], noise)

        iz = int(np.argmin(np.abs(z_model - z)))
        model_window = (t_model_ms >= t0) & (t_model_ms <= t1_model)
        proxy = n_model[model_window, iz] * np.sqrt(
            np.maximum(Te_model[model_window, iz], 0.0)
        )
        tau_model = _efold_time_ms(t_model_ms[model_window], proxy)

        rows.append(
            {
                "port": int(ports[p]),
                "z": z,
                "tau_exp_ms": tau_exp,
                "tau_model_ms": tau_model,
                "ratio": tau_model / tau_exp if np.isfinite(tau_exp) else np.nan,
            }
        )
    return rows, (t0, t1_model)


def _report_peak_current(peak):
    print("\n--- stage (i): discharge current ---")
    print(
        f"  peak:    model {peak['model_peak_a']:8.4g} A at {peak['model_peak_t_ms']:+6.2f} ms"
        f" | measured {peak['exp_peak_a']:8.4g} +/- {peak['exp_peak_sem_a']:.2g} A"
        f" at {peak['exp_peak_t_ms']:+6.2f} ms | ratio {peak['ratio']:.3f}"
    )
    print(
        f"  plateau: model {peak['model_late_a']:8.4g} A (15-19.5 ms mean)"
        f" | measured {peak['exp_late_a']:8.4g} A | ratio {peak['late_ratio']:.3f}"
    )


def _report_decay(rows, window):
    print(
        f"\n--- stage (iii): Isat decay e-fold times, window "
        f"{window[0]:.1f}-{window[1]:.1f} ms ---"
    )
    header = f"{'port':>6} {'z [cm]':>8} {'tau_model':>10} {'tau_exp':>9} {'ratio':>7}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['port']:>6} {r['z']:8.0f} {r['tau_model_ms']:9.2f}ms "
            f"{r['tau_exp_ms']:8.2f}ms {r['ratio']:7.2f}"
        )
    ratios = [r["ratio"] for r in rows if np.isfinite(r["ratio"])]
    if ratios:
        print(f"  mean tau_model/tau_exp: {np.mean(ratios):.2f}")


def _report(label, rows):
    print("\n--- stage (ii): bulk Te / density at the ES1 ports ---")
    print("  (sigma = |dev|/sigma_tot, SEM (+) sweep systematics; '~' marks")
    print("   semi-quantitative rows where measured Te < 1 eV)")
    header = (
        f"{'field':>5} {'port':>6} {'z [cm]':>8} {'model':>11} {'measured':>11} "
        f"{'ratio':>7} {'rms rel':>8} {'|dev|/sig':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        flag = "~" if r.get("semiquant") else " "
        print(
            f"{r['field']:>5} {r['port']:>6} {r['z']:8.0f} {r['model']:11.4g} "
            f"{r['exp']:11.4g} {r['ratio']:7.2f} {r['rms_rel']:8.2f} "
            f"{r['sigma']:9.1f}{flag}"
        )
    for field in ("Te", "n", "Isat"):
        sub = [r for r in rows if r["field"] == field]
        if sub:
            print(
                f"  {field}: mean ratio {np.mean([r['ratio'] for r in sub]):.2f}, "
                f"mean rms rel {np.mean([r['rms_rel'] for r in sub]):.2f}, "
                f"mean |dev|/sig {np.mean([r['sigma'] for r in sub]):.1f}"
            )
        if len(sub) >= 2:
            # Axial-gradient figure of merit: far-port / near-port ratio.
            # 1.00 means the model's axial falloff matches the measured one
            # regardless of overall magnitude.
            first, last = sub[0], sub[-1]
            grad_model = last["model"] / first["model"]
            grad_exp = last["exp"] / first["exp"]
            print(
                f"  {field} axial gradient (z={last['z']:.0f}/{first['z']:.0f}): "
                f"model {grad_model:.2f} vs measured {grad_exp:.2f} "
                f"(ratio {grad_model / grad_exp:.2f})"
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=None)
    parser.add_argument(
        "--exchange-model", default="knudsen", choices=("knudsen", "constant")
    )
    parser.add_argument(
        "--tau-afterglow",
        type=float,
        default=None,
        metavar="S",
        help="override the afterglow duration [s] to cover more of the decay",
    )
    parser.add_argument(
        "--decay-window",
        type=float,
        nargs=2,
        default=(20.5, 25.0),
        metavar=("T0", "T1"),
        help="stage (iii) fit window on the main-discharge clock [ms]",
    )
    parser.add_argument(
        "--save-h5",
        type=Path,
        default=None,
        help="save the run result for later re-scoring",
    )
    parser.add_argument(
        "--from-h5",
        type=Path,
        default=None,
        help="score a saved result instead of running the model",
    )
    parser.add_argument(
        "--drag-closure",
        default=None,
        choices=(
            "constant",
            "slip",
            "neutral_momentum",
            "neutral_momentum_two_zone",
        ),
        help=(
            "swap the ion-neutral drag closure for the gate-#2 A/B: "
            "constant (production 0.5), slip (entrainment closure, b=1), "
            "neutral_momentum (evolved M_n wind, b=1), or "
            "neutral_momentum_two_zone (M_n wind + two-zone radial closure)"
        ),
    )
    parser.add_argument(
        "--Rp-model",
        default=None,
        choices=("sample", "resolved_gap"),
        help=(
            "cathode gap-resistance model for the M1 A/B "
            "(CATHODE_IDRIVEN_PLAN.md): sample (historical one-cell "
            "Spitzer) or resolved_gap (profile-integrated over the gap)"
        ),
    )
    parser.add_argument(
        "--beam-excitation",
        default=None,
        choices=("scalar14", "manifold"),
        help=(
            "beam excitation channel for the WP-A A/B "
            "(BEAM_DEPOSITION_PLAN.md A3): scalar14 (production 2p_scalar "
            "with the historical b=1.4 estimate) or manifold (measured "
            "Ralchenko singlet sum, b=1.0)"
        ),
    )
    parser.add_argument(
        "--beam-deposition",
        default=None,
        choices=("beer_lambert", "csda", "csda_ql"),
        help=(
            "beam deposition model for the WP-B B3 A/B "
            "(BEAM_DEPOSITION_PLAN.md): beer_lambert (historical "
            "single-event absorption), csda (slowing-down module, classical "
            "fast-electron Coulomb), or csda_ql (csda + quasilinear "
            "beam-plasma drag)"
        ),
    )
    parser.add_argument(
        "--es",
        type=int,
        choices=(1, 2, 3),
        default=1,
        help=(
            "which experiment-set overlay to score against "
            "(data/es{N}_sim1d_overlay.npz; ES1-3 share fueling and differ "
            "only in heater current and bank voltage — the drive-side "
            "ladder, THESIS_NOTES §2). NB the model config must match the "
            "campaign's operating point; this flag only selects the data."
        ),
    )
    args = parser.parse_args(argv)

    overlay_path = (
        OVERLAY
        if args.es == 1
        else OVERLAY.parent / f"es{args.es}_sim1d_overlay.npz"
    )
    overlay = np.load(overlay_path, allow_pickle=False)
    if args.from_h5 is not None:
        result = load_result_hdf5(args.from_h5)
        geometry = None
        label = f"saved run {args.from_h5}"
    else:
        label = f"resolved ({args.exchange_model}, nx={args.nx or 'default'})"
        if args.drag_closure is not None:
            label += f" [drag={args.drag_closure}]"
        if args.Rp_model is not None:
            label += f" [Rp={args.Rp_model}]"
        if args.beam_excitation is not None:
            label += f" [beam_exc={args.beam_excitation}]"
        extra = {}
        if args.tau_afterglow is not None:
            extra["tau_afterglow"] = args.tau_afterglow
        # A/B instrument for BEAM_DEPOSITION_PLAN.md A3: the measured singlet
        # manifold vs the retired 1.4 estimate. "scalar14" is PARAM_OVERRIDES
        # as-is; "manifold" swaps the cross-section set and drops b to the
        # pure-multiplier benchmark value.
        if args.beam_excitation == "manifold":
            extra["beam_excitation_model"] = "manifold"
            extra["b_beam_excitation"] = 1.0
        if args.beam_deposition is not None:
            label += f" [dep={args.beam_deposition}]"
            extra["beam_deposition_model"] = (
                "csda" if args.beam_deposition.startswith("csda")
                else "beer_lambert"
            )
            if args.beam_deposition == "csda_ql":
                extra["beam_anomalous_model"] = "quasilinear"
        try:
            result, geometry, params, flags = run_model(
                nx=args.nx,
                exchange_model=args.exchange_model,
                extra=extra,
                drag_closure=args.drag_closure,
                Rp_model=args.Rp_model,
            )
        except BreakdownError as error:
            print(f"{label}: no breakdown (I_tot={error.I_tot:.4g} A)")
            return 1
        if args.save_h5 is not None:
            save_result_hdf5(args.save_h5, result, params=params, flags=flags)
            print(f"saved result to {args.save_h5}")
    print(f"\n=== {label} ===")
    _report_peak_current(compare_peak_current(result, overlay))
    _report(label, compare(result, geometry, overlay))
    decay_rows, window = compare_decay(result, overlay, window_ms=args.decay_window)
    _report_decay(decay_rows, window)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
