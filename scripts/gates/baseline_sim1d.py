"""Golden baseline capture/verify for LAPDSim1D.

This is the production reversibility guarantee: a committed reference trajectory
plus a checker that re-runs the solver and asserts bit-exact reproduction. Every
change under ``_sim1d`` must keep ``--verify`` green without recapture.

**The baseline config is ``default_config()`` + the committed stance file
(``scripts/stances/g1atrim.toml``) + ``nx = 60``** -- GOLDEN-AT-STANCE, ratified
2026-08-20. The stance is applied minus its mesh-sized package; see the re-cut
note at ``STANCE_MESH_SIZED_PARAMS`` for exactly what is dropped and why.

**The shipped defaults are NOT the production package**, and an earlier draft of
this file said they were. The R2a/R2b folds moved the neutral closure family and
the measured machine into ``default_config()``, but the OPERATING POINT stayed in
the stance file. Captured at bare defaults the fixture gated an unrepresentative
corner -- marginal breakdown, an anode-cathode gap that never filled, and an
anode sheath draining a near-empty flanking cell on a ~100 ns e-fold.

**Consequence of anchoring on the stance, stated plainly: EDITING THE STANCE FILE
BREAKS THIS GATE until the fixture is recaptured.** That is the intended
trade -- the fixture tracks the configuration the campaign actually runs, and the
price is that a stance edit is now a recapture event. It is not a licence to
recapture casually: a recapture is reviewed, authorized, and recorded in
``golden_baseline_provenance.md``.

**No campaign DRIVER is imported.** ``compare_sim1d_es1`` and
``run_mechanism_ladder`` stay unimported, so their override dicts cannot reach
this anchor; the one ``scripts/`` import is ``stance_config``, the loader for the
committed stance artifact.

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
import sys
from pathlib import Path

import numpy as np

from cablp.cathode import kernels as _kernel_selector
from cablp.solvers._sim1d import (
    LAPDSim1D,
    default_config,
    summarize_result,
)

# Default location of the committed golden fixture (NPZ) and its JSON sidecar.
SCRIPT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = SCRIPT_DIR / "baselines" / "production_discharge.npz"

# The stance loader, for the committed stance FILE. This is the module's only
# scripts/ import and it is deliberate: `stance_config` is a small loader for a
# committed artifact, not a campaign driver whose dicts drift. The drivers
# (`compare_sim1d_es1`, `run_mechanism_ladder`) stay unimported.
# scripts/ sibling imports: the seven purpose subdirectories on sys.path.
import sys as _sys
from pathlib import Path as _Path
for _sub in ("atomic", "gates", "kinetic", "run", "score", "stance",
             "verify"):
    _dir = str(_Path(__file__).resolve().parents[1] / _sub)
    if _dir not in _sys.path:
        _sys.path.insert(0, _dir)
from stance_config import load_stance  # noqa: E402

# --- Baseline config: the stance of record, re-cut to the gate mesh --------
# GOLDEN-AT-STANCE (ratified 2026-08-20). The config is
# ``default_config()`` + the committed stance file + ``nx = 60``.
#
# Why the stance and not the bare defaults: the shipped defaults are NOT the
# production package. The R2a/R2b folds moved the closure family and the
# measured machine into the defaults, but the OPERATING POINT stayed in the
# stance file -- the emission constant, the bank voltage, the cathode thermal
# pair, the puff level and its equilibration window. Captured at bare defaults
# this fixture gated an unrepresentative corner: breakdown was marginal, the
# anode-cathode gap never filled, and the anode sheath drained a near-empty
# flanking cell on a ~100 ns e-fold. The stance is what makes the discharge
# behave like the machine.
PRODUCTION_STANCE = "g1atrim"

# THE RE-CUT. Four stance params are per-cell arrays sized to the stance's own
# 280-cell mesh (1 plenum + 5 gap + 5 fixed source + 268 far column + 1
# collector). They cannot travel to nx=60, and they are NOT resampled here:
#
#   * the two radius profiles are built offline by scripts/g1_build_profiles.py
#     from a measured field census, and the vessel profile is a STAIRCASE whose
#     steps interpolation would smear into a bore the machine does not have;
#   * the two nn0 profiles are an equilibrated 4.5 ms foot computed for that
#     mesh -- resampling them changes the neutral inventory and the near-source
#     structure, so it is a new initial condition, not the stance's.
#
# The package is therefore dropped WHOLE, with the two flags that require it,
# rather than half-applied: a prescribed geometry carrying a default fill would
# be a hybrid corner of exactly the kind this re-anchor exists to stop being.
# Everything that is mesh-independent still travels, which is every scalar
# operating-point key plus the baffles (whose arrays are physical cm, not
# per-cell). What the gate loses is the measured flare, the vessel staircase
# and the shaped foot; what it keeps is the operating point.
STANCE_MESH_SIZED_PARAMS = (
    "plasma_radius_profile_cm",
    "machine_radius_profile_cm",
    "nn0_profile",
    "nn0_annulus_profile",
)
STANCE_MESH_SIZED_FLAGS = (
    "prescribed_area_geometry",
    "neutral_initial_profile",
)

BASELINE_PARAM_OVERRIDES = {
    # Axial resolution -- the one run-shape pin. The campaign runs 268 far-column
    # cells; this gate runs the coarse mesh because a reviewer pays for it on the
    # candidate branch and again post-merge. Pinned rather than inherited so a
    # future default-nx change cannot multiply that cost silently.
    #
    "nx": 60,
    # The scalar neutral fill, PINNED. The stance sets nn0 = None because it
    # arms neutral_initial_profile with a per-cell foot; the re-cut above drops
    # that package and clears the flag, which leaves the scalar as the fill the
    # gate actually starts from. Until 2026-08-27 nothing named it, so
    # resolve_nn0 fell through to the frozen gas-puff lookup table and the gate
    # silently inherited its answer. This IS that answer, frozen as a literal at
    # the table's retirement: the value cablp/vars/_nn_table.lookup_nn0 returned
    # for this stance's S_gp = 9010.0 with TwinCathode = False. It is not
    # re-derived here because the table is gone: it was read out of the table
    # programmatically before the deletion, never hand-typed, and checked to
    # round-trip through repr(); pinning it left the constructed initial state
    # bit-identical (0 differing raw uint64 over all 1152 initial-state values,
    # nn and nn_a included). To re-derive it, read the table out of history --
    # it is one commit back from the retirement, under cablp/vars/.
    #
    # It is a LITERAL and not a lookup on purpose. A gate whose initial
    # condition is computed by a frozen table it cannot regenerate is a gate
    # that cannot say what it starts from.
    #
    # WHAT THIS DOES NOT DECIDE: whether this is the RIGHT fill. The value is on
    # the pre-2026-08-21 0 C-sccm convention while S_gp is meter-sccm, so it
    # carries the ~7% conversion inconsistency the retired table's own docstring
    # disclosed. Changing it would move the golden and is a stance-era question,
    # deliberately left on the board rather than resolved by a retirement.
    "nn0": 2725059978765.871,
    # Deliberately OVERRIDES the stance's "stop". For a campaign arm a step cap
    # is a budget and a truncated arm is still data, so "stop" is right there.
    # Here max_steps is not a run length at all -- it is a TRIPWIRE (see
    # BASELINE_RUN_KWARGS). Reaching it means this configuration no longer
    # completes a discharge in a sane number of steps, which is a different
    # failure from "the trajectory moved" and should say so loudly instead of
    # silently handing --verify a short trajectory to report as a shape
    # mismatch.
    "max_steps_action": "raise",
}
# input_flags overrides beyond the stance. The shaped initial fill is gone with
# the mesh-sized package, and the solver refuses a profile and an equilibration
# together, so the equilibrated seed fills the machine again -- at the stance's
# own 25 ms puff window, which is a scalar and travels. This is the substitute
# for the foot, and it is why the gap fills.
BASELINE_FLAG_OVERRIDES = {
    "neutral_equilibration": True,
}
# Run controls. dt/operator_split stay at the solver defaults (adaptive dt, the
# shipped split), and t_end stays dynamic -- the run goes to the current-trigger
# end time, so THE FIXTURE COVERS THE WHOLE CYCLE: ignition, breakdown, the
# plateau and the afterglow.
#
# There is no cost cap any more. The earlier draft capped at 40,000 steps
# because at BARE DEFAULTS the adaptive dt was pinned near 3e-8 s by the
# surface_loss limiter and an uncapped run cost ~4 hours. That was a symptom of
# the unrepresentative corner, not a property of the model: on the stance the
# discharge ignites properly and the limiter relaxes as the column fills.
#
# The capture's own measured cost -- step count, dynamic t_end, wall time,
# saves, and the dt history behind them -- lives in
# scripts/golden_baseline_provenance.md, which is rewritten at every recapture.
# It is deliberately NOT restated here: a runtime figure in a standing comment
# has no way of announcing that it has drifted.
#
# max_steps is a TRIPWIRE, not a run length: roughly twice the step count the
# capture actually measures (that note carries the ratio), paired with
# max_steps_action="raise" above. It exists so that a change which quietly
# destroys the timestep fails loudly and quickly instead of running for hours.
# If it ever fires, the question is "what happened to dt", not "what happened to
# the trajectory". Sized at 2x deliberately: a backstop with only a few percent
# of headroom is not a backstop, it is a second cost cap waiting to truncate the
# gate the first time a legitimate change nudges the timestep.
BASELINE_RUN_KWARGS = {
    "t_end": None,
    "dt": None,
    "operator_split": None,
    "max_steps": 150000,
}


def build_baseline_config(param_overrides=None, flag_overrides=None):
    """Return ``(params, flags)`` for the baseline, with optional extra overrides.

    Layering, in order: ``default_config()``, the committed stance file minus
    its mesh-sized package, then the run-shape overrides. ``param_overrides`` /
    ``flag_overrides`` layer on top for an explicitly requested variant.
    """
    params, flags = default_config()

    stance = load_stance(PRODUCTION_STANCE)
    stance_params = dict(stance.params)
    stance_flags = dict(stance.flags)
    for key in STANCE_MESH_SIZED_PARAMS:
        stance_params.pop(key, None)
    for key in STANCE_MESH_SIZED_FLAGS:
        stance_flags[key] = False
    params.update(stance_params)
    flags.update(stance_flags)

    params.update(BASELINE_PARAM_OVERRIDES)
    flags.update(BASELINE_FLAG_OVERRIDES)
    if param_overrides:
        params.update(param_overrides)
    if flag_overrides:
        flags.update(flag_overrides)
    return params, flags


def run_baseline(params, flags):
    """Run the solver and return ``(result, trajectory_dict, summary, cells)``.

    ``cells`` is the mesh cell count read from the solver's own geometry. It is
    NOT inferred from the width of ``y``: the number of packed fields per cell
    depends on the neutral closure (5 for the cold single-zone layout, 8 once
    evolved neutral momentum, the two-zone split and the neutral energy channel
    are on), so any fixed divisor is wrong for some configuration.
    """
    sim = LAPDSim1D(params, flags)
    sim.start_simulation(**BASELINE_RUN_KWARGS)
    result = sim.get_results()
    y = np.asarray(result.y, dtype=float)
    if y.ndim != 2:
        raise RuntimeError(f"expected 2-D packed trajectory y, got shape {y.shape}")
    cells = int(sim.geometry.cells)
    if y.shape[1] % cells:
        raise RuntimeError(
            f"packed trajectory width {y.shape[1]} is not a whole number of "
            f"fields over {cells} cells"
        )
    trajectory = {
        "time": np.asarray(result.time, dtype=float),
        "y": y,
        "phase": np.asarray(result.phase, dtype="U32"),
    }
    return result, trajectory, summarize_result(result), cells


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
    result, trajectory, summary, cells = run_baseline(params, flags)
    baseline_path = Path(baseline_path)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(baseline_path, **trajectory)
    sidecar = baseline_path.with_suffix(".json")
    payload = {
        "description": (
            "Golden baseline at the stance of record, re-cut to the gate "
            "mesh: default_config() plus the committed stance file "
            "scripts/stances/g1atrim.toml, minus that stance's mesh-sized "
            "package, plus the run-shape overrides in "
            "baseline_sim1d.BASELINE_PARAM_OVERRIDES (nx=60)."
        ),
        "result_format": (
            "sim1d packed conservative trajectory y[saves, fields*cells]"
        ),
        "cells": cells,
        "fields_per_cell": int(trajectory["y"].shape[1] // cells),
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
        f"fields={payload['fields_per_cell']}, "
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
    _, trajectory, summary, _cells = run_baseline(params, flags)
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


def _print_provenance():
    """Print which cathode kernels THIS process loaded, before the run starts.

    A golden transcript is evidence about a code path, and without this line it
    cannot say which path produced it: the compiled and pure runs are bit-exact
    by construction, so their output is identical and a hand-written label is
    the only thing that ever distinguished them. ``_kernel_selector.PROVENANCE``
    is bound at selector import from the loaded module's own ``KERNEL_ID``
    (``"pure"`` when nothing is compiled), so this is an IN-PROCESS probe of
    what actually loaded -- not a restatement of the environment variable, which
    is what a transcript label was.

    Print-only. Nothing here reaches the solver, the trajectory, or the fixture,
    and it is called from ``main`` rather than at module scope so the several
    scripts that import ``build_baseline_config`` stay silent. ``flush`` so the
    line survives a run that is interrupted or piped.
    """
    print(f"provenance: kernels={_kernel_selector.PROVENANCE}", flush=True)


def main(argv=None):
    args = _parse_args(argv)
    _print_provenance()
    if args.capture:
        return capture(args.baseline)
    return verify(args.baseline, rtol=args.rtol, atol=args.atol)


if __name__ == "__main__":
    raise SystemExit(main())
