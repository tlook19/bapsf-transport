"""Compare a sim1d run against the ES1 experimental overlay.

The point of the model is to reproduce measured behaviour, so the meaningful
benchmark is the data, not a previous version of the model. This samples the
simulation at the five ES1 probe locations and the experimental time base, and
reports the deviation in electron temperature and density against the measured
means and their standard errors.

Comparisons are made on the main-discharge clock, matching the notebook: model
time is shifted so t = 0 is the start of the main discharge, which is what the
experimental ``*_time_ms`` axes are referenced to.

Usage::

    python scripts/compare_sim1d_es1.py                      # resolved + knudsen
    python scripts/compare_sim1d_es1.py --legacy             # legacy geometry
    python scripts/compare_sim1d_es1.py --nx 185
"""

import argparse
from pathlib import Path

import numpy as np

from cablp.solvers._sim1d import (
    BreakdownError,
    LAPDSim1D,
    default_config,
)

OVERLAY = Path(__file__).resolve().parent / "data" / "es1_sim1d_overlay.npz"

PARAM_OVERRIDES = {
    "V_bank": 180.0,
    "T_s": 273.15 + 1725,
    "S_gp": 3000,
    "S_gp_decay_target": 2000,
    "tau_gp_pulse_duration": 1e-3,
    "tau_gp_decay_duration": 5e-3,
    "b_ion_neutral_drag": 0.5,
    "b_Qei": 1,
    "b_Qen": 1,
    "b_Qcx": 1,
    "Rp": 15.0,
    "R_cath": 15.0,
    "R_comp": 0.010,
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


def run_model(resolved=True, nx=None, exchange_model="knudsen", extra=None):
    params, flags = default_config()
    params.update(PARAM_OVERRIDES)
    flags.update(FLAG_OVERRIDES)
    flags["resolved_boundaries"] = bool(resolved)
    if resolved:
        params["neutral_exchange_model"] = exchange_model
    if nx is not None:
        params["nx"] = nx
    if extra:
        params.update(extra)
    sim = LAPDSim1D(params, flags)
    sim.start_simulation(t_end=None, dt=None, operator_split=None, max_steps=None)
    return sim.get_results(), sim.geometry


def compare(result, geometry, overlay):
    """Return per-port deviation of model Te and density from the ES1 means."""
    z_probe = np.asarray(overlay["z_cm"], dtype=float)
    ports = np.asarray(overlay["port"])
    origin = _main_discharge_origin(result)
    t_model_ms = (np.asarray(result.time, dtype=float) - origin) * 1.0e3
    z_model = np.asarray(result.z_cm, dtype=float)

    rows = []
    for field, t_key, mean_key, sem_key, unit in (
        ("Te", "te_time_ms", "te_mean_ev", "te_sem_ev", "eV"),
        ("n", "density_time_ms", "density_mean_cm3", "density_total_sem_cm3", "cm^-3"),
    ):
        t_exp = np.asarray(overlay[t_key], dtype=float)
        mean = np.asarray(overlay[mean_key], dtype=float)
        sem = np.asarray(overlay[sem_key], dtype=float)
        model_2d = np.asarray(getattr(result, field), dtype=float)
        # Only compare where the experiment has data and the model has run.
        window = (t_exp >= t_model_ms.min()) & (t_exp <= t_model_ms.max())
        for p, (z, port) in enumerate(zip(z_probe, ports)):
            iz = int(np.argmin(np.abs(z_model - z)))
            model_t = np.interp(t_exp[window], t_model_ms, model_2d[:, iz])
            exp_t = mean[p, window]
            err = sem[p, window]
            good = np.isfinite(exp_t) & np.isfinite(model_t) & (exp_t != 0.0)
            if not np.any(good):
                continue
            ratio = float(np.mean(model_t[good] / exp_t[good]))
            rel = float(np.sqrt(np.mean(((model_t - exp_t)[good] / exp_t[good]) ** 2)))
            sigma = float(np.mean(np.abs((model_t - exp_t)[good] / err[good])))
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
                }
            )
    return rows


def _report(label, rows):
    print(f"\n=== {label} ===")
    header = (
        f"{'field':>5} {'port':>6} {'z [cm]':>8} {'model':>11} {'measured':>11} "
        f"{'ratio':>7} {'rms rel':>8} {'|dev|/SEM':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['field']:>5} {r['port']:>6} {r['z']:8.0f} {r['model']:11.4g} "
            f"{r['exp']:11.4g} {r['ratio']:7.2f} {r['rms_rel']:8.2f} {r['sigma']:10.1f}"
        )
    for field in ("Te", "n"):
        sub = [r for r in rows if r["field"] == field]
        if sub:
            print(
                f"  {field}: mean ratio {np.mean([r['ratio'] for r in sub]):.2f}, "
                f"mean rms rel {np.mean([r['rms_rel'] for r in sub]):.2f}, "
                f"mean |dev|/SEM {np.mean([r['sigma'] for r in sub]):.1f}"
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--nx", type=int, default=None)
    parser.add_argument(
        "--exchange-model", default="knudsen", choices=("knudsen", "molecular_flow")
    )
    args = parser.parse_args(argv)

    overlay = np.load(OVERLAY, allow_pickle=False)
    label = (
        "legacy"
        if args.legacy
        else f"resolved ({args.exchange_model}, nx={args.nx or 'default'})"
    )
    try:
        result, geometry = run_model(
            resolved=not args.legacy, nx=args.nx, exchange_model=args.exchange_model
        )
    except BreakdownError as error:
        print(f"{label}: no breakdown (I_tot={error.I_tot:.4g} A)")
        return 1
    _report(label, compare(result, geometry, overlay))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
