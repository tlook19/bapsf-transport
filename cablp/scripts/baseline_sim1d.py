"""Golden baseline capture/verify for the 1D source-boundary redesign.

This is the reversibility guarantee from ``BOUNDARY_REGIONS_PLAN.md`` §13: a
committed reference trajectory produced by *today's* solver, plus a checker that
re-runs the solver and asserts it still reproduces that trajectory to tight
tolerance. Every milestone from M1 on must keep ``--verify`` green, and the
``resolved_boundaries`` master switch (default off) plus the degenerate
legacy-limit resolved config must both reproduce it.

The baseline config mirrors the production notebook
``cablp/scripts/sim1d_run_and_plot.ipynb`` (implicit heat + tr_bdf2 + Strang +
Picard, cathode coupling on, real timescales). Keep the two in sync: if the
notebook's overrides change, re-capture the baseline deliberately (a re-baseline
is an explicit, reviewed step, per §10/§13).

Usage::

    # write the golden fixture (run once, before any _sim1d/ change)
    python scripts/baseline_sim1d.py --capture

    # re-run and assert equivalence (run at every milestone boundary)
    python scripts/baseline_sim1d.py --verify

The trajectory is stored as the packed conservative state ``y`` (the solver's
source of truth); all primitive fields derive from it, so comparing ``y`` is the
sharpest single check. A JSON sidecar carries human-readable health scalars and
the exact config used, so a reviewer can see what produced the fixture without
loading the NPZ.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from cablp.solvers._sim1d import (
    LAPDSim1D,
    default_config,
    summarize_result,
)

# Default location of the committed golden fixture (NPZ) and its JSON sidecar.
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BASELINE = SCRIPT_DIR / "baselines" / "notebook_discharge.npz"

# --- Baseline config: mirrors sim1d_run_and_plot.ipynb cell 3 ---------------
# input_dict (parameter) overrides.
BASELINE_PARAM_OVERRIDES = {
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
    # Second-order operator-split time integration (all three knobs needed).
    "implicit_heat_scheme": "tr_bdf2",
    "operator_splitting": "strang",
    "heat_picard_iterations": 2,
    "heat_picard_tol": 1e-10,
}
# input_flags overrides.
BASELINE_FLAG_OVERRIDES = {
    "ion_neutral_drag_cx_only": False,
    "ion_neutral_thermalization": True,
}
# Run controls: None => LAPDSim1D defaults (adaptive dt, dynamic current-trigger
# t_end, unlimited steps -- the notebook's own settings).
BASELINE_RUN_KWARGS = {
    "t_end": None,
    "dt": None,
    "operator_split": None,
    "max_steps": None,
}


def build_baseline_config(param_overrides=None, flag_overrides=None):
    """Return ``(params, flags)`` for the baseline, with optional extra overrides.

    ``param_overrides`` / ``flag_overrides`` layer on top of the baseline (used
    from M1 on to build the degenerate legacy-limit *resolved* config that must
    also reproduce the golden trajectory).
    """
    params, flags = default_config()
    params.update(BASELINE_PARAM_OVERRIDES)
    flags.update(BASELINE_FLAG_OVERRIDES)
    if param_overrides:
        params.update(param_overrides)
    if flag_overrides:
        flags.update(flag_overrides)
    return params, flags


def run_baseline(params, flags):
    """Run the solver and return ``(result, trajectory_dict, summary)``."""
    sim = LAPDSim1D(params, flags)
    sim.start_simulation(**BASELINE_RUN_KWARGS)
    result = sim.get_results()
    y = np.asarray(result.y, dtype=float)
    if y.ndim != 2:
        raise RuntimeError(f"expected 2-D packed trajectory y, got shape {y.shape}")
    trajectory = {
        "time": np.asarray(result.time, dtype=float),
        "y": y,
        "phase": np.asarray(result.phase, dtype="U32"),
    }
    return result, trajectory, summarize_result(result)


def _summary_scalars(summary):
    """Pull JSON-serializable health scalars from a summarize_result namespace."""
    keys = (
        "finite",
        "samples",
        "steps",
        "final_time",
        "n_min",
        "n_max",
        "nn_min",
        "nn_max",
        "Te_min",
        "Te_max",
        "Ti_min",
        "Ti_max",
        "plasma_inventory_relative_drift",
        "neutral_inventory_relative_drift",
        "total_particle_inventory_relative_drift",
        "thermal_energy_relative_drift",
    )
    out = {}
    for key in keys:
        value = getattr(summary, key, None)
        if isinstance(value, np.generic):
            value = value.item()
        out[key] = value
    return out


def capture(baseline_path):
    """Run the baseline config and write the golden NPZ + JSON sidecar."""
    params, flags = build_baseline_config()
    result, trajectory, summary = run_baseline(params, flags)
    baseline_path = Path(baseline_path)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(baseline_path, **trajectory)
    sidecar = baseline_path.with_suffix(".json")
    payload = {
        "description": (
            "Golden baseline for the 1D source-boundary redesign "
            "(BOUNDARY_REGIONS_PLAN.md §13). Produced by baseline_sim1d.py "
            "--capture with the sim1d_run_and_plot.ipynb production config."
        ),
        "result_format": "sim1d packed conservative trajectory y[saves, 5*cells]",
        "cells": int(trajectory["y"].shape[1] // 5),
        "saves": int(trajectory["y"].shape[0]),
        "summary": _summary_scalars(summary),
        "params": _json_safe(params),
        "flags": _json_safe(flags),
    }
    sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    size_mb = baseline_path.stat().st_size / 1e6
    print(
        "baseline captured: "
        f"{baseline_path} ({size_mb:.2f} MB), "
        f"saves={payload['saves']}, cells={payload['cells']}, "
        f"steps={summary.steps}, final_time={summary.final_time:.6e} s"
    )
    print(f"baseline sidecar: {sidecar}")
    return 0


def verify(baseline_path, rtol, atol, param_overrides=None, flag_overrides=None):
    """Re-run and assert the fresh trajectory matches the golden fixture.

    Returns 0 on match, 1 on any mismatch. ``param_overrides`` / ``flag_overrides``
    let a caller check that a *variant* config (e.g. the degenerate legacy-limit
    resolved geometry from M1 on) still reproduces the golden trajectory.
    """
    baseline_path = Path(baseline_path)
    if not baseline_path.exists():
        print(f"baseline missing: {baseline_path} -- run --capture first")
        return 1
    golden = np.load(baseline_path, allow_pickle=False)
    golden_time = golden["time"]
    golden_y = golden["y"]

    params, flags = build_baseline_config(param_overrides, flag_overrides)
    _, trajectory, summary = run_baseline(params, flags)
    fresh_time = trajectory["time"]
    fresh_y = trajectory["y"]

    ok = True
    if fresh_y.shape != golden_y.shape:
        print(
            "MISMATCH shape: "
            f"golden y{golden_y.shape} vs fresh y{fresh_y.shape} "
            f"(golden saves={golden_y.shape[0]}, fresh saves={fresh_y.shape[0]})"
        )
        return 1

    time_abs = float(np.max(np.abs(fresh_time - golden_time))) if golden_time.size else 0.0
    if not np.allclose(fresh_time, golden_time, rtol=1e-12, atol=1e-15):
        ok = False
        print(f"MISMATCH time grid: max|dt|={time_abs:.3e} s")

    diff = np.abs(fresh_y - golden_y)
    scale = np.abs(golden_y) + np.abs(fresh_y)
    rel = np.where(scale > 0.0, 2.0 * diff / scale, 0.0)
    max_abs = float(np.max(diff)) if diff.size else 0.0
    max_rel = float(np.max(rel)) if rel.size else 0.0
    exact = bool(np.array_equal(fresh_y, golden_y))
    if not np.allclose(fresh_y, golden_y, rtol=rtol, atol=atol):
        ok = False
        print(f"MISMATCH trajectory: max_abs={max_abs:.3e} max_rel={max_rel:.3e}")

    status = "OK" if ok else "FAIL"
    print(
        f"baseline verify {status}: "
        f"saves={fresh_y.shape[0]}, exact={exact}, "
        f"max_rel={max_rel:.3e}, max_abs={max_abs:.3e}, "
        f"time_max_abs={time_abs:.3e} s "
        f"(rtol={rtol:.1e}, atol={atol:.1e})"
    )
    return 0 if ok else 1


def _json_safe(mapping):
    """Coerce a params/flags dict to JSON-serializable values."""
    out = {}
    for key, value in mapping.items():
        if isinstance(value, np.generic):
            value = value.item()
        out[key] = value
    return out


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Capture or verify the sim1d golden baseline."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--capture",
        action="store_true",
        help="Run the baseline config and write the golden NPZ + JSON sidecar.",
    )
    mode.add_argument(
        "--verify",
        action="store_true",
        help="Re-run the baseline config and assert it matches the golden fixture.",
    )
    parser.add_argument(
        "--baseline",
        default=str(DEFAULT_BASELINE),
        help="Path to the golden NPZ fixture.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-9,
        help="Relative tolerance for the trajectory comparison (verify).",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=0.0,
        help="Absolute tolerance for the trajectory comparison (verify).",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    if args.capture:
        return capture(args.baseline)
    return verify(args.baseline, rtol=args.rtol, atol=args.atol)


if __name__ == "__main__":
    raise SystemExit(main())
