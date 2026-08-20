"""Golden baseline capture/verify for LAPDSim1D.

This is the production reversibility guarantee: a committed reference trajectory
plus a checker that re-runs the solver and asserts bit-exact reproduction. Every
change under ``_sim1d`` must keep ``--verify`` green without recapture.

**The baseline config is ``default_config()`` plus a RUN-SHAPE table, and
nothing else** (R2b re-anchor, 2026-08-20). The shipped defaults are the
production package, so the gate exercises the configuration the campaign
actually runs instead of a frozen historical operating point. Every override
below is a mesh or cost choice; a physics or stance value pinned here would be
a second stance, silently diverging from the shipped one, which is exactly the
drift the old ~30-pin table had accumulated.

**This module is SELF-CONTAINED by rule.** It imports nothing from the campaign
drivers, so ``--verify`` runs with ``compare_sim1d_es1`` and
``run_mechanism_ladder`` absent and no stance edit can reach the anchor.
Re-anchoring is a deliberate, reviewed recapture event, never a side effect.

The retired fixture -- the ~30-pin table holding the 2026-07-22 operating
point -- is reproducible only at the tag ``pre-refactor-2026-08-20`` with its
environment lockfile; its pin table is in this file's git history. The pre-D1
legacy fixture under ``baselines/legacy-final-2026-07-22/`` is likewise a
pinned historical scaffold whose tag is retired.

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
DEFAULT_BASELINE = SCRIPT_DIR / "baselines" / "production_discharge.npz"

# --- Baseline config: the shipped defaults plus run shape ------------------
# RUN SHAPE ONLY (R2b re-anchor, 2026-08-20). Everything else -- the whole
# physics and stance package -- comes from default_config() and is deliberately
# NOT restated here: a value written twice is a value that can disagree with
# itself, which is how the retired ~30-pin table stopped describing anything
# the campaign ran.
#
# The bar for an entry below: it must be a MESH or COST choice. If a physics or
# stance value ever looks necessary here, the honest move is to recapture the
# fixture at the shipped default, not to pin around it.
BASELINE_PARAM_OVERRIDES = {
    # Axial resolution. The campaign runs 268 far-column cells; this gate runs
    # the coarse mesh because a reviewer pays for it on every branch and again
    # post-merge. Pinned rather than inherited so a future default-nx change
    # cannot multiply that cost silently.
    "nx": 60,
    # Stop at BASELINE_RUN_KWARGS["max_steps"] instead of raising. The shipped
    # default is "raise" because for a campaign arm a step cap is a failure;
    # for this gate the cap IS the run length, so reaching it is success. The
    # production stance carries the same setting for the same reason.
    "max_steps_action": "stop",
}
# input_flags overrides. Empty by construction: every flag the production
# package needs is a shipped default, and a flag pinned here would be a stance
# choice, not run shape.
BASELINE_FLAG_OVERRIDES = {}
# Run controls. dt/operator_split stay at the solver defaults (adaptive dt, the
# shipped split). t_end is left dynamic and never reached: max_steps ends the
# run first, by design.
#
# max_steps IS THE COST KNOB, and it is a step count rather than a t_end on
# purpose: a step cap bounds what a reviewer pays even if a future change makes
# the adaptive dt smaller, whereas a duration cap would let the same change
# lengthen the gate without bound.
#
# 40,000 steps is sized to keep this gate at roughly the wall time of the
# fixture it replaced. MEASURED on the shipped defaults at nx=60 (2026-08-20):
# the adaptive dt is held near 3e-8 s by the surface_loss limiter through
# ignition, so running to the dynamic t_end (2.53e-2 s) would take ~4 HOURS --
# ~30x the retired fixture, twice per merge. The capped trajectory covers the
# pre-breakdown foot, breakdown, and the first ~0.7 ms of the discharge; it
# does NOT reach the plateau or the afterglow, so this gate certifies ignition
# physics and everything the construction and equilibration touch, and says
# nothing about late-time behaviour. See golden_baseline_provenance.md.
BASELINE_RUN_KWARGS = {
    "t_end": None,
    "dt": None,
    "operator_split": None,
    "max_steps": 40000,
}


def build_baseline_config(param_overrides=None, flag_overrides=None):
    """Return ``(params, flags)`` for the baseline, with optional extra overrides.

    ``param_overrides`` / ``flag_overrides`` layer on top of the baseline for an
    explicitly requested production variant.
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
            "Golden baseline at the shipped LAPDSim1D defaults "
            "(default_config()) plus the run-shape overrides in "
            "baseline_sim1d.BASELINE_PARAM_OVERRIDES."
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
    rel = np.divide(2.0 * diff, scale, out=np.zeros_like(diff), where=scale > 0.0)
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
