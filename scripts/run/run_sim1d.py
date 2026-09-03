"""Run a NAMED configuration and save its trajectory.

The plain sim1d driver. It builds no configuration of its own: ``--config``
names one -- a committed configuration by name, or the path of a configuration
file, derived or not -- and this driver adds only the run controls on its
command line. ``default_config()`` is the template of keys and their classes,
not a plasma anyone runs, so there is no bare route here to inherit it by
silence.

The LAPD reference configuration is ``scripts/stances/g1atrim.toml``::

    python scripts/run/run_sim1d.py --config g1atrim --output run.h5
"""

import argparse
from collections import Counter
from pathlib import Path

from cablp.solvers._sim1d import (
    LAPDSim1D,
    ProgressPrinter1D,
    summarize_result,
)

# scripts/ sibling imports: the seven purpose subdirectories on sys.path.
import sys as _sys
from pathlib import Path as _Path
for _sub in ("atomic", "gates", "kinetic", "run", "score", "stance",
             "verify"):
    _dir = str(_Path(__file__).resolve().parents[1] / _sub)
    if _dir not in _sys.path:
        _sys.path.insert(0, _dir)

from stance_config import (  # noqa: E402
    STANCE_DIR,
    SUFFIX,
    available_stances,
    load_configuration,
)

REFERENCE_CONFIGURATION = STANCE_DIR / f"g1atrim{SUFFIX}"


def main(argv=None):
    args = _parse_args(argv)
    # A run names its configuration. The refusal is here rather than in
    # argparse's required= so it can name the reference configuration and the
    # two forms --config takes; "the following arguments are required" cannot.
    if args.config is None:
        raise SystemExit(
            "run_sim1d: name the configuration to run. Pass --config <name> "
            "for a committed configuration (available: "
            f"{', '.join(available_stances()) or '(none committed)'}) or "
            "--config <path.toml> for a configuration file, derived or not. "
            f"The LAPD reference configuration is {REFERENCE_CONFIGURATION}."
        )
    params, flags, configuration = load_configuration(args.config)
    if args.operator_split:
        flags["implicit_heat_conduction"] = True
    if args.neutral_equilibration:
        flags["neutral_equilibration"] = True
    if args.launch_plasma_after_equilibration:
        flags["neutral_equilibration"] = True
        flags["launch_plasma_after_equilibration"] = True
    if args.neutral_equilibration_cycles is not None:
        params["neutral_equilibration_cycles"] = args.neutral_equilibration_cycles
    if args.neutral_equilibration_dt is not None:
        params["neutral_equilibration_dt"] = args.neutral_equilibration_dt

    # The lineage travels with the run, its identity restated over the config
    # this driver actually constructs -- the named configuration plus the run
    # controls above.
    configuration = configuration.with_identity(params, flags)
    sim = LAPDSim1D(params, flags, configuration=configuration)
    progress_tracker = (
        ProgressPrinter1D(interval_fraction=args.progress_interval)
        if args.progress
        else None
    )
    sim.start_simulation(
        t_end=args.t_end,
        dt=args.dt,
        operator_split=args.operator_split,
        max_steps=args.max_steps,
        progress_tracker=progress_tracker,
        progress_interval_s=args.progress_interval_time,
    )
    result = sim.get_results()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sim.save_result(output, result, params=params, flags=flags)
    summary = summarize_result(result)

    constraints = Counter(diag.active_constraint for diag in result.diagnostics)
    constraints_text = ", ".join(
        f"{name}:{count}" for name, count in sorted(constraints.items())
    )
    if not constraints_text:
        constraints_text = "none"
    print(
        f"sim1d configuration: {configuration.name}"
        + (
            f" (derived from {' <- '.join(configuration.base_chain)})"
            if configuration.base_chain
            else ""
        )
        + f", identity={configuration.identity}"
    )
    print(
        "sim1d run complete: "
        f"steps={result.steps}, "
        f"final_time={result.final_time:.6e} s, "
        f"saves={len(result.time)}, "
        f"constraints={constraints_text}, "
        f"output={output}"
    )
    print(
        "sim1d health: "
        f"finite={summary.finite}, "
        f"n=[{summary.n_min:.6e}, {summary.n_max:.6e}] cm^-3, "
        f"Te=[{summary.Te_min:.6e}, {summary.Te_max:.6e}] eV, "
        f"Ti=[{summary.Ti_min:.6e}, {summary.Ti_max:.6e}] eV, "
        "particle_drift="
        f"{summary.total_particle_inventory_relative_drift:.6e}, "
        f"thermal_drift={summary.thermal_energy_relative_drift:.6e}"
    )
    # The clamp no longer appears in the constraint histogram above (it is a
    # fact about the step, not a bound), so report the census explicitly.
    print(
        "sim1d dt_min clamp: "
        f"clamped_steps={summary.dt_min_clamped_step_count}, "
        f"max_consecutive={summary.max_consecutive_dt_min_clamped_steps}, "
        f"hard_zero_steps={summary.dt_min_hard_zero_step_count}"
    )
    if hasattr(result, "neutral_equilibration_summary"):
        neutral_summary = result.neutral_equilibration_summary
        print(
            "sim1d neutral equilibration: "
            f"cycles={neutral_summary.cycles}, "
            f"final_time={neutral_summary.final_time:.6e} s, "
            f"nn_mean={neutral_summary.mean_nn:.6e} cm^-3, "
            f"nn_std={neutral_summary.std_nn:.6e} cm^-3, "
            f"nn_min={neutral_summary.min_nn:.6e} cm^-3, "
            f"nn_max={neutral_summary.max_nn:.6e} cm^-3"
        )
    return 0


def _parse_args(argv):
    parser = argparse.ArgumentParser(description="Run LAPDSim1D and save HDF5.")
    parser.add_argument("--output", required=True, help="Output HDF5 path.")
    parser.add_argument("--t-end", type=float, default=None, help="Final time [s].")
    parser.add_argument("--dt", type=float, default=None, help="Fixed timestep [s].")
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "REQUIRED. The configuration to run: a committed configuration "
            "NAME, or the path of a configuration file. The LAPD reference "
            "configuration is scripts/stances/g1atrim.toml."
        ),
    )
    parser.add_argument(
        "--operator-split",
        action="store_true",
        default=None,
        help="Use explicit non-heat plus implicit heat-conduction splitting.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=(
            "Maximum accepted timesteps before aborting. "
            "Defaults to config max_steps; 0 means unlimited."
        ),
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print lightweight accepted-step progress updates.",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=0.0,
        help="Optional minimum progress fraction between printed updates.",
    )
    parser.add_argument(
        "--progress-interval-time",
        type=float,
        default=1.0e-4,
        help="Minimum simulation time between progress updates [s].",
    )
    parser.add_argument(
        "--neutral-equilibration",
        action="store_true",
        help="Run the configured neutral-only equilibration before returning.",
    )
    parser.add_argument(
        "--launch-plasma-after-equilibration",
        action="store_true",
        help="Seed plasma run with neutral equilibration final state.",
    )
    parser.add_argument(
        "--neutral-equilibration-cycles",
        type=int,
        default=None,
        help="Override neutral equilibration cycle count.",
    )
    parser.add_argument(
        "--neutral-equilibration-dt",
        type=float,
        default=None,
        help="Override neutral equilibration fixed timestep [s].",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
