#!/usr/bin/env python
"""C1: quantify the fueling-bifurcation sensitivity (BEAM_DEPOSITION_PLAN WP-C).

The ES1 operating point sits on the hot-starved / cold-flooded regime boundary
(THESIS_NOTES §3): tuned ``S_gp`` values carry a large, previously
unquantified sensitivity. This sweeps ``S_gp`` (with ``S_gp_decay_target``
scaled proportionally, per the documented bracket protocol) around the
production value on the current-driven benchmark (manifold excitation,
b = 1) and reports:

- per-scale benchmark metrics (peak/plateau current, port Te/n mean ratios,
  axial-gradient ratios, decay-tau mean ratio);
- central finite-difference logarithmic sensitivities
  ``d(metric)/d ln S_gp`` at the operating point;
- where each regime indicator changes sign across the scan (the boundary,
  to the sweep's resolution).

Runs one scale per ``--scales`` entry and writes a JSON metric file per
scale (``--out-dir``); ``--report`` aggregates every JSON found there
without running anything, so scales can be sharded across parallel
invocations.

Usage:
    python scripts/sweep_sim1d_fueling.py --scales 0.8 0.9 1.0
    python scripts/sweep_sim1d_fueling.py --scales 1.1 1.2 1.5
    python scripts/sweep_sim1d_fueling.py --report
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "cmp", Path(__file__).resolve().parent / "compare_sim1d_es1.py"
)
cmp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cmp)

from cablp.solvers._sim1d import BreakdownError  # noqa: E402

S_GP_PRODUCTION = 3000.0
S_GP_DECAY_PRODUCTION = 2000.0
NX = 120


def run_point(scale: float, out_dir: Path, save_h5: bool = True) -> dict:
    extra = {
        "S_gp": S_GP_PRODUCTION * scale,
        "S_gp_decay_target": S_GP_DECAY_PRODUCTION * scale,
        "beam_excitation_model": "manifold",
        "b_beam_excitation": 1.0,
        "cathode_solver_model": "current_driven",
        # This sweep classifies a point FROM the BreakdownError below, so it
        # keeps the historical raise-on-timeout behavior rather than the
        # switch-open default (which would return an unignited result and
        # trip the scorer instead).
        "prebreakdown_timeout_action": "raise",
    }
    metrics: dict = {"scale": scale, "S_gp": extra["S_gp"]}
    try:
        overlay = np.load(cmp.OVERLAY, allow_pickle=False)
        result, geometry, params, flags = cmp.run_model(
            nx=NX, extra=extra
        )
    except BreakdownError as error:
        metrics["status"] = f"no_breakdown (I_tot={error.I_tot:.4g} A)"
        return metrics
    except Exception as error:  # noqa: BLE001 — record, don't kill the sweep
        metrics["status"] = f"failed ({type(error).__name__})"
        return metrics
    metrics["status"] = "ok"
    if save_h5:
        from cablp.solvers._sim1d.results.io import save_result_hdf5

        save_result_hdf5(
            out_dir / f"es1_nx120_fuel_s{scale:g}.h5",
            result,
            params=params,
            flags=flags,
        )

    peak = cmp.compare_peak_current(result, overlay)
    metrics["peak_ratio"] = peak["ratio"]
    metrics["plateau_ratio"] = peak["late_ratio"]

    rows = cmp.compare(result, geometry, overlay)
    for field in ("Te", "n"):
        frows = sorted(
            (r for r in rows if r["field"] == field), key=lambda r: r["z"]
        )
        if not frows:
            continue
        metrics[f"{field}_mean_ratio"] = float(
            np.mean([r["ratio"] for r in frows])
        )
        near, far = frows[0], frows[-1]
        model_grad = far["model"] / near["model"]
        exp_grad = far["exp"] / near["exp"]
        metrics[f"{field}_grad_ratio"] = model_grad / exp_grad

    decay_rows, _window = cmp.compare_decay(result, overlay)
    ratios = [r["ratio"] for r in decay_rows if np.isfinite(r["ratio"])]
    metrics["decay_mean_ratio"] = float(np.mean(ratios)) if ratios else np.nan
    return metrics


def _guarded_run_point(scale, out_dir, save_h5):
    try:
        return run_point(scale, out_dir, save_h5=save_h5)
    except Exception as error:  # noqa: BLE001 — a metric failure loses one
        # point, not the sweep (a transient overlay BadZipFile did exactly
        # this once under two parallel nx=120 shards).
        return {
            "scale": scale,
            "S_gp": S_GP_PRODUCTION * scale,
            "status": f"metrics_failed ({type(error).__name__})",
        }


METRICS = (
    "peak_ratio",
    "plateau_ratio",
    "Te_mean_ratio",
    "n_mean_ratio",
    "Te_grad_ratio",
    "n_grad_ratio",
    "decay_mean_ratio",
)


def report(out_dir: Path) -> None:
    points = []
    for path in sorted(out_dir.glob("fuel_metrics_s*.json")):
        points.append(json.loads(path.read_text()))
    if not points:
        print(f"no fuel_metrics_s*.json under {out_dir}")
        return
    points.sort(key=lambda m: m["scale"])

    print(f"{'scale':>6} {'S_gp':>6} {'status':>14} "
          + " ".join(f"{name:>16}" for name in METRICS))
    for m in points:
        row = f"{m['scale']:6.2f} {m['S_gp']:6.0f} {m['status'][:14]:>14} "
        for name in METRICS:
            v = m.get(name)
            row += f" {v:16.3f}" if isinstance(v, float) else f" {'-':>16}"
        print(row)

    by_scale = {m["scale"]: m for m in points if m["status"] == "ok"}
    lo, hi = 0.9, 1.1
    if lo in by_scale and hi in by_scale:
        dln = np.log(hi) - np.log(lo)
        print("\ncentral d(metric)/dln(S_gp) at the operating point "
              f"(scales {lo}/{hi}):")
        for name in METRICS:
            a, b = by_scale[lo].get(name), by_scale[hi].get(name)
            if isinstance(a, float) and isinstance(b, float) and np.isfinite(a) and np.isfinite(b):
                print(f"  {name:>16}: {(b - a) / dln:+8.3f}")

    # Regime indicators: hot-starved has Te ratio > 1 and density rising
    # downstream (n_grad_ratio > 1); cold-flooded flips both and slows the
    # decay. Report the first sign change of each along the scan.
    print("\nregime-boundary indicators (first crossing along the scan):")
    for name, description in (
        ("Te_mean_ratio", "Te ratio crosses 1 (hot -> cold)"),
        ("decay_mean_ratio", "decay ratio crosses 1 (fast -> slow)"),
    ):
        prev = None
        crossing = None
        for m in points:
            v = m.get(name)
            if not isinstance(v, float) or not np.isfinite(v):
                continue
            if prev is not None and (prev[1] - 1.0) * (v - 1.0) < 0.0:
                crossing = (prev[0], m["scale"])
                break
            prev = (m["scale"], v)
        where = (
            f"between scales {crossing[0]:g} and {crossing[1]:g}"
            if crossing
            else "not crossed in this scan"
        )
        print(f"  {description}: {where}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scales", type=float, nargs="+", default=None)
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument("--no-h5", action="store_true")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    if args.report:
        report(args.out_dir)
        return
    if not args.scales:
        parser.error("pass --scales to run, or --report to aggregate")
    for scale in args.scales:
        print(f"=== running scale {scale:g} (S_gp = "
              f"{S_GP_PRODUCTION * scale:.0f}) ===", flush=True)
        metrics = _guarded_run_point(scale, args.out_dir, save_h5=not args.no_h5)
        out = args.out_dir / f"fuel_metrics_s{scale:g}.json"
        out.write_text(json.dumps(metrics, indent=2))
        print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
