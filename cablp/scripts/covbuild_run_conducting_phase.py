"""Run the conducting-phase window, optionally with the coverage closure on.

Applies ``covbuild_conducting_phase.toml`` as a DELTA over the campaign stance
(``compare_sim1d_es1.PARAM_OVERRIDES`` / ``FLAG_OVERRIDES``) and runs it,
printing the breakdown instant so the window can be checked against the
measured >= 4.5 ms conducting phase.

Shakedown instrument for the coverage-closure build: direction only, never
scored. ``--coverage f0[,r]`` turns the closure on for the A/B arm.
"""

import argparse
import json
import tomllib
from pathlib import Path

import numpy as np

from cablp.solvers._sim1d import LAPDSim1D, ProgressPrinter1D, default_config
from cablp.solvers._sim1d.results.io import save_result_hdf5

from compare_sim1d_es1 import FLAG_OVERRIDES, PARAM_OVERRIDES
from run_mechanism_ladder import ES_OPERATING


DELTA_TOML = Path(__file__).resolve().parent / "covbuild_conducting_phase.toml"


def build_config(nx, coverage=None, extra=None):
    params, flags = default_config()
    params.update(PARAM_OVERRIDES)
    flags.update(FLAG_OVERRIDES)
    op = ES_OPERATING[1]
    params.update({
        "nx": nx,
        "V_bank": op["V_bank"],
        "cathode_solver_model": "current_driven",
        "beam_deposition_model": "csda",
        "beam_anomalous_model": "quasilinear",
        "cathode_emission_profile": "gaussian",
        "cathode_warming_model": "power_balance",
        "T_s": op["Ts_standby_K"],
        "cathode_Ts_base_K": op["Ts_standby_K"],
        "cathode_heat_capacity_J_per_K": 120.0,
        "cathode_emissivity": 0.7,
        "phi_wf": 2.869,
        "cathode_surface_model": "ads_des",
        "cathode_phiwf_clean_eV": 2.809,
        "cathode_cleaning_sigma_cm2": 3.5e-16,
        "cathode_cleaning_E_th_eV": 20.0,
        "Te_birth_ionization": "floor",
        "gas_puff_mode": "square",
        "cathode_sample_smoothing": "presheath",
    })
    delta = tomllib.loads(DELTA_TOML.read_text())
    params.update(delta.get("params", {}))
    flags.update(delta.get("flags", {}))
    if coverage is not None:
        f0, r = coverage
        params["coverage_initial_fraction"] = f0
        params["coverage_growth_rate_per_s"] = r
        flags["coverage_closure"] = True
    if extra:
        params.update(extra)
    return params, flags


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--nx", type=int, default=120)
    p.add_argument("--t-end", type=float, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--coverage", default=None,
                   help="f_cov0[,r] -- turns the coverage closure on")
    p.add_argument("--extra", nargs="*", default=(),
                   help="additional k=v param overrides (JSON-parsed)")
    p.add_argument("--save-h5", required=True)
    args = p.parse_args(argv)

    coverage = None
    if args.coverage is not None:
        parts = args.coverage.split(",")
        coverage = (float(parts[0]), float(parts[1]) if len(parts) > 1 else 0.0)
    extra = {}
    for kv in args.extra:
        k, v = kv.split("=", 1)
        try:
            extra[k] = json.loads(v)
        except json.JSONDecodeError:
            extra[k] = v

    params, flags = build_config(args.nx, coverage=coverage, extra=extra)
    sim = LAPDSim1D(params, flags)
    sim.start_simulation(
        t_end=args.t_end,
        max_steps=args.max_steps,
        progress_tracker=_progress,
        progress_interval_s=0.0,
    )
    result = sim.get_results()
    save_result_hdf5(args.save_h5, result, params=params, flags=flags)

    times = np.asarray(result.time, dtype=float)
    breakdown = None
    events = getattr(result, "phase_events", None) or {}
    for when, phase, reason in zip(
        np.asarray(events.get("time", []), dtype=float),
        events.get("phase", []),
        events.get("reason", []),
    ):
        print(f"phase event: {phase!r} ({reason!r}) at {when * 1e3:.4f} ms")
        if str(phase) == "main_discharge" and breakdown is None:
            breakdown = float(when)
    print(f"window: 0 -> {times[-1] * 1e3:.4f} ms, {times.size} saves")
    if breakdown is not None:
        print(f"conducting phase contained: {breakdown * 1e3:.4f} ms")
    I_loop = np.asarray(
        result.cathode_diagnostics["circuit_I_loop"], dtype=float
    )
    print(f"I_loop: max {I_loop.max():.4g} A, final {I_loop[-1]:.4g} A")
    for threshold in (2.0, 132.0, 1000.0):
        hit = np.flatnonzero(I_loop >= threshold)
        when = f"{times[hit[0]] * 1e3:.4f} ms" if hit.size else "never"
        print(f"I_loop first reaches {threshold:g} A at {when}")
    if coverage is not None:
        f_cov = np.asarray(
            result.cathode_diagnostics["coverage_fraction"], dtype=float
        )
        print(
            f"f_cov trace: start {f_cov[0]:.6f}, end {f_cov[-1]:.6f}, "
            f"reaches 0.5 at "
            + (
                f"{times[np.flatnonzero(f_cov >= 0.5)[0]] * 1e3:.4f} ms"
                if np.any(f_cov >= 0.5)
                else "never"
            )
        )
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            i = min(int(frac * (times.size - 1)), times.size - 1)
            print(
                f"  t={times[i] * 1e3:9.4f} ms  f_cov={f_cov[i]:.6f}  "
                f"nn_deficit_max="
                f"{result.cathode_diagnostics['coverage_nn_deficit_max'][i]:.6e}"
            )
    print(f"saved {args.save_h5}")


# Step-gated only: interval_fraction above 1 can never come due, so the
# cadence is purely `interval_steps`. The solver's own progress_interval_s is
# 0 above, so every accepted step reaches the tracker and the step gate is the
# only one that decides.
_progress = ProgressPrinter1D(interval_fraction=2.0, interval_steps=20000)


if __name__ == "__main__":
    main()
