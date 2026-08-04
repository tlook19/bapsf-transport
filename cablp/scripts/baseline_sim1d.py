"""Golden baseline capture/verify for the 1D source-boundary redesign.

This is the production reversibility guarantee: a committed reference trajectory
plus a checker that re-runs the solver and asserts bit-exact reproduction. Every
change under ``_sim1d`` must keep ``--verify`` green without recapture.

The baseline config is the PRODUCTION configuration
(2026-07-22): current-driven cathode + resolved boundaries + ADAS rates +
knudsen exchange + the measured square fueling waveform + the M6 candidate
constants, IMPORTED from the campaign drivers (compare_sim1d_es1 /
run_mechanism_ladder) so the gate cannot drift from the production stance.
The pre-D1 legacy fixture is archived under
``baselines/legacy-final-2026-07-22/`` and remains reproducible at the tag
of the same name (plus env lockfiles); a re-baseline stays an explicit,
reviewed step.

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

from cablp.solvers._sim1d import (
    LAPDSim1D,
    default_config,
    summarize_result,
)

# Default location of the committed golden fixture (NPZ) and its JSON sidecar.
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BASELINE = SCRIPT_DIR / "baselines" / "production_discharge.npz"

# --- Baseline config: the production stance, imported (no drift) -----------
sys.path.insert(0, str(SCRIPT_DIR))
from compare_sim1d_es1 import (  # noqa: E402
    FLAG_OVERRIDES as PRODUCTION_FLAG_OVERRIDES,
    PARAM_OVERRIDES as PRODUCTION_PARAM_OVERRIDES,
)
from run_mechanism_ladder import ES_OPERATING  # noqa: E402

# The M6 candidate constants (run_m6_point.py, ES1 rung): square waveform at
# sq3400, fitted loop inductance, drive tier, frozen M5a' surface tier,
# presheath sample smoothing.
BASELINE_PARAM_OVERRIDES = {
    **PRODUCTION_PARAM_OVERRIDES,
    # Axial resolution PINNED (stance promotion, 2026-07-27). The campaign
    # drivers promoted their default nx to 240 (compare_sim1d_es1.PRODUCTION_NX,
    # run_m6_point --nx); this gate must NOT inherit it. The golden is a
    # regression scaffold, not a production claim, and quadrupling the cell
    # count would multiply every reviewer gate's runtime. 60 is the value this
    # fixture was captured at (config default at capture time) -- pinned here
    # explicitly so a future config.py nx change cannot move it either.
    "nx": 60,
    # Historical checkpoint seed. The repaired live defaults are intentionally
    # different; this unchanged fixture remains an off-path regression anchor.
    "Te0": 0.1,
    "Ti0": 0.1,
    "V_bank": ES_OPERATING[1]["V_bank"],
    "cathode_solver_model": "current_driven",
    "beam_deposition_model": "csda",
    "beam_anomalous_model": "quasilinear",
    # WP-D product transport: the fixture was captured with local (birth-cell)
    # product deposition. Pin it so a future default promotion to "nonlocal"
    # cannot silently move this anchor (same rule as Te_birth_ionization above).
    "beam_product_transport": "local",
    # WP-E QL heating locality: same rule as beam_product_transport above --
    # the fixture banks the anomalous drag locally; pin it so a future default
    # promotion to "tail_walk" cannot silently move this anchor.
    "heating_anomalous_transport": "local",
    "cathode_emission_profile": "gaussian",
    "cathode_warming_model": "power_balance",
    "T_s": ES_OPERATING[1]["Ts_standby_K"],
    "cathode_Ts_base_K": ES_OPERATING[1]["Ts_standby_K"],
    "cathode_heat_capacity_J_per_K": 120.0,
    "cathode_conduction_W_per_K": 1200.0,
    "cathode_emissivity": 0.7,
    "phi_wf": 2.869,
    "cathode_surface_model": "ads_des",
    "cathode_phiwf_clean_eV": 2.809,
    "cathode_cleaning_sigma_cm2": 3.5e-16,
    "cathode_cleaning_E_th_eV": 20.0,
    # The committed fixture records "local". Pin that historical stance
    # explicitly; R1e must not silently change a physical default to make an
    # old result look like a floor-birth run.
    "Te_birth_ionization": "local",
    "gas_puff_mode": "square",
    "S_gp": 3400,
    "L_parasitic_H": 8.1e-6,
    "cathode_sample_smoothing": "presheath",
    "neutral_exchange_model": "knudsen",
    # R5 STANCE FLIP (2026-07-25): the production defaults moved to conservative
    # ionization birth + the Phelps ion-neutral operator. Pin the historical
    # legacy stance (the ad-hoc constant-0.5 / cx_derived drag + thermalization,
    # now removed from the ES production config) so this checkpoint stays
    # bit-exact.
    "ionization_birth_energy_model": "legacy",
    "b_ion_neutral_drag": 0.5,
    "ion_neutral_drag_model": "constant",
    "sigma_in_model": "cx_derived",
    "b_ion_neutral_thermalization": 1.0,
    # R5 STANCE FLIP part 2 (2026-07-25): the config walkthrough flipped more
    # live defaults. Pin every one the historical fixture ran at its OLD default
    # (and that neither PRODUCTION_PARAM_OVERRIDES nor the pins above cover) so
    # the anchor stays bit-exact:
    "Ti_floor": 0.1,                       # default relaxed to 300 K (0.02585)
    "S_pump_L": 2000,                      # default now matches R (4000)
    "gas_puff_profile": "cell",            # default now "cosine_pipe"
    "hyperbolic_wave_speed": "isothermal",  # default now "adiabatic" (A3)
    # R5 ES1 tuning pass (2026-07-26): the ES production config
    # (PRODUCTION_PARAM_OVERRIDES, inherited above) gained the end-expansion
    # machine geometry (Rcs=40/Lcs=25/Rsup=0 + end_expansion_geometry). The
    # historical golden was captured at the geometry_defaults (Rcs=Lcs=Rsup=0,
    # no end-expansion, 67 cells); pin those back so this anchor stays
    # bit-exact and does NOT track the live ES geometry. (The end_expansion_*
    # params inherited from production are popped in build_baseline_config when
    # the flag is off, since they are presence-gated on it.)
    "Rcs": 0.0,
    "Lcs": 0.0,
    "Rsup": 0.0,
    # Measured 25 ms equilibration puff width (2026-07-29): the ES production
    # config (PRODUCTION_PARAM_OVERRIDES, inherited above) adopted it, but this
    # fixture was captured with the equilibration inheriting tau_discharge as
    # its puff window. Pin the historical stance back (None = the
    # tau_discharge-derived window) so the anchor does NOT track the live ES
    # puff width -- same rule as every pin above.
    "equilibration_gas_puff_on_s": None,
}
# input_flags overrides.
BASELINE_FLAG_OVERRIDES = {
    **PRODUCTION_FLAG_OVERRIDES,
    # Historical R1-off stance, pinned so the checkpoint remains reproducible
    # without making it the future production configuration.
    "active_plasma_topology": False,
    "raw_stage_validation": False,
    "resolved_boundaries": True,
    # R4.1/A15 anode-mesh beam interception is now the production default (on),
    # but this csda checkpoint fixture predates it -- pin it off so the historical
    # trajectory stays reproducible (same pattern as the R1 selectors above; the
    # baseline NPZ is never recaptured to hide a repaired-physics change).
    "beam_anode_interception": False,
    # R5 STANCE FLIP (2026-07-25): the R2/R3/R4.3 repairs are now production
    # defaults. Pin them to their historical-off values here so this checkpoint
    # (which predates the flip) stays bit-exact -- the anchor never recaptures.
    "hyperbolic_energy_consistent": False,
    "characteristic_boundary": False,
    "ion_neutral_moment_closure": False,
    # the historical golden ran the legacy ion-neutral thermalization arm
    "ion_neutral_thermalization": True,
    # R5 STANCE FLIP part 2 (2026-07-25): front_flux default is now False (R2 G7
    # retired the sonic front); the fixture ran it on.
    "front_flux": True,
    # R5 ES1 tuning pass (2026-07-26): the ES production config gained the
    # end-expansion geometry; the historical golden ran without it. Pin off
    # (paired with the Rcs/Lcs/Rsup=0 pins above) so the anchor stays 67 cells.
    "end_expansion_geometry": False,
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
    # The end-expansion params are presence-gated on the flag; the historical
    # anchor pins the flag off, so drop the params inherited from the ES
    # production overrides (else construction raises a loud ValueError).
    if not flags.get("end_expansion_geometry", False):
        for _k in ("end_expansion_cells", "end_expansion_machine_radius_cm",
                   "end_expansion_plasma_radius_cm"):
            params.pop(_k, None)
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
            "Golden baseline at the PRODUCTION configuration: "
            "current-driven + resolved + adas + "
            "knudsen + square waveform + M6 candidate constants."
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
