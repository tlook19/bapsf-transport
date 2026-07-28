"""Validation sweeps for the resolved source/end boundary (M6, plan §10).

Two studies:

``--convergence``
    Refine the cathode-anode gap (``nx_gap``) with the presheath correction on and
    off. ``presheath_alpha`` is *designed* to make the surface Bohm flux
    mesh-independent -- as the near-wall cell thins, the local density falls along
    the Boltzmann profile while the exponent rises to compensate. That is a claim
    about the discretization, so it needs measuring rather than asserting. With
    ``b_presheath_length = 0`` (the historical constant factor) the same sweep
    should drift, which is the control.

``--sensitivity``
    Vary the knobs plan §10 calls out -- ``Lcs``, ``Rcs``, ``Rsup``, ``eta`` and
    pump speed -- around the resolved default, to see which actually move the
    answer. Each is at its legacy limit in the default config, so the deltas here
    are also a check that the knobs are wired to something.

Usage::

    python scripts/sweep_sim1d_resolved.py --convergence
    python scripts/sweep_sim1d_resolved.py --sensitivity
"""

import argparse
import time

import numpy as np

from cablp.solvers._sim1d import (
    BreakdownError,
    LAPDSim1D,
    default_config,
    summarize_result,
)

# Mirrors scripts/sim1d_run_and_plot.ipynb, as baseline_sim1d.py does.
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
    # This sweep classifies a point FROM the BreakdownError in run_case, so it
    # keeps the historical raise-on-timeout behavior rather than the
    # switch-open default (which would return an unignited result instead).
    "prebreakdown_timeout_action": "raise",
}
FLAG_OVERRIDES = {
    "ion_neutral_drag_cx_only": False,
    "ion_neutral_thermalization": True,
}


def run_case(param_overrides=None, exchange_model="knudsen"):
    """Run one resolved discharge and return summary metrics."""
    params, flags = default_config()
    params.update(PARAM_OVERRIDES)
    flags.update(FLAG_OVERRIDES)
    params["neutral_exchange_model"] = exchange_model
    if param_overrides:
        params.update(param_overrides)

    sim = LAPDSim1D(params, flags)
    started = time.time()
    try:
        sim.start_simulation(t_end=None, dt=None, operator_split=None, max_steps=None)
    except BreakdownError as error:
        return {"status": f"no breakdown (I_tot={error.I_tot:.3g} A)"}
    except Exception as error:  # noqa: BLE001 - a sweep should report, not abort
        return {"status": f"{type(error).__name__}: {str(error)[:48]}"}
    result = sim.get_results()
    summary = summarize_result(result)
    geometry = sim.geometry
    thermal = float(
        np.sum((result.Ee[-1] + result.Ei[-1]) * geometry.plasma_volume_cm3)
    )
    plasma_inventory = float(np.sum(result.n[-1] * geometry.plasma_volume_cm3))
    return {
        "status": "ok",
        "wall_s": time.time() - started,
        "cells": geometry.cells,
        "steps": int(result.steps),
        "final_time": float(result.final_time),
        "n_max": float(summary.n_max),
        "Te_max": float(summary.Te_max),
        "thermal_erg": thermal,
        "plasma_inventory": plasma_inventory,
    }


def _print_rows(title, rows):
    print(f"\n=== {title} ===")
    header = (
        f"{'case':28s} {'cells':>6} {'steps':>7} {'final_t [s]':>12} "
        f"{'n_max':>11} {'Te_max':>8} {'thermal [erg]':>14} {'wall':>7}"
    )
    print(header)
    print("-" * len(header))
    for label, metrics in rows:
        if metrics.get("status") != "ok":
            print(f"{label:28s} {metrics['status']}")
            continue
        print(
            f"{label:28s} {metrics['cells']:6d} {metrics['steps']:7d} "
            f"{metrics['final_time']:12.5e} {metrics['n_max']:11.4e} "
            f"{metrics['Te_max']:8.3f} {metrics['thermal_erg']:14.5e} "
            f"{metrics['wall_s']:6.0f}s"
        )


def _spread(rows):
    """Return max relative spread of each metric across successful rows."""
    good = [m for _, m in rows if m.get("status") == "ok"]
    if len(good) < 2:
        return {}
    out = {}
    for key in ("final_time", "n_max", "Te_max", "thermal_erg"):
        values = np.array([m[key] for m in good], dtype=float)
        centre = np.mean(np.abs(values))
        out[key] = float((values.max() - values.min()) / centre) if centre else 0.0
    return out


def convergence(nx_gaps, exchange_model="knudsen"):
    """Refine the gap with the presheath correction on and off."""
    for b_presheath in (1.0, 0.0):
        rows = []
        for nx_gap in nx_gaps:
            label = f"nx_gap={nx_gap} ({50.0 / nx_gap:.2f} cm cells)"
            rows.append(
                (
                    label,
                    run_case(
                        {"nx_gap": nx_gap, "b_presheath_length": b_presheath},
                        exchange_model=exchange_model,
                    ),
                )
            )
        title = (
            "gap refinement, presheath correction ON (b_presheath_length=1)"
            if b_presheath
            else "gap refinement, historical constant factor (b_presheath_length=0)"
        )
        _print_rows(title, rows)
        spread = _spread(rows)
        if spread:
            print(
                "  max relative spread across the sweep: "
                + ", ".join(f"{k}={v:.3%}" for k, v in spread.items())
            )
    return 0


def sensitivity(exchange_model="knudsen"):
    """Vary the §10 knobs around the resolved default."""
    cases = [
        ("resolved default", {}),
        ("Lcs=25, Rcs=25 (duct)", {"Lcs": 25.0, "Rcs": 25.0}),
        ("Rsup=10 (support rods)", {"Rsup": 10.0}),
        ("eta=0 (transparent anode)", {"eta": 0.0}),
        ("eta=0.6 (opaque anode)", {"eta": 0.6}),
        ("S_pump_L x2", {"S_pump_L": 4000.0}),
        ("pump elbow C=2000 L/s", {"pump_elbow_conductance_lps": 2000.0}),
    ]
    rows = [
        (label, run_case(overrides, exchange_model=exchange_model))
        for label, overrides in cases
    ]
    _print_rows(f"resolved sensitivity sweep ({exchange_model})", rows)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--convergence", action="store_true")
    parser.add_argument("--sensitivity", action="store_true")
    parser.add_argument(
        "--nx-gap",
        type=int,
        nargs="+",
        default=[5, 10, 20],
        help="gap cell counts for the convergence study",
    )
    parser.add_argument(
        "--exchange-model",
        default="knudsen",
        choices=("knudsen", "constant"),
        help="resolved neutral transport closure",
    )
    args = parser.parse_args(argv)
    if not (args.convergence or args.sensitivity):
        parser.error("choose --convergence and/or --sensitivity")
    if args.convergence:
        convergence(args.nx_gap, exchange_model=args.exchange_model)
    if args.sensitivity:
        sensitivity(exchange_model=args.exchange_model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
