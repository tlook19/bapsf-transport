import argparse
from collections import Counter
from pathlib import Path

from cablp.solvers._sim1d import LAPDSim1D, default_config, load_config


def main(argv=None):
    args = _parse_args(argv)
    params, flags = load_config(args.config) if args.config else default_config()
    if args.operator_split:
        flags["implicit_heat_conduction"] = True

    sim = LAPDSim1D(params, flags)
    result = sim.run(
        t_end=args.t_end,
        dt=args.dt,
        operator_split=args.operator_split,
        max_steps=args.max_steps,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sim.save_result(output, result, params=params, flags=flags)

    constraints = Counter(diag.active_constraint for diag in result.diagnostics)
    constraints_text = ", ".join(
        f"{name}:{count}" for name, count in sorted(constraints.items())
    )
    if not constraints_text:
        constraints_text = "none"
    print(
        "sim1d run complete: "
        f"steps={result.steps}, "
        f"final_time={result.final_time:.6e} s, "
        f"saves={len(result.time)}, "
        f"constraints={constraints_text}, "
        f"output={output}"
    )
    return 0


def _parse_args(argv):
    parser = argparse.ArgumentParser(description="Run LAPDSim1D and save HDF5.")
    parser.add_argument("--output", required=True, help="Output HDF5 path.")
    parser.add_argument("--t-end", type=float, required=True, help="Final time [s].")
    parser.add_argument("--dt", type=float, default=None, help="Fixed timestep [s].")
    parser.add_argument("--config", default=None, help="Optional TOML config path.")
    parser.add_argument(
        "--operator-split",
        action="store_true",
        help="Use explicit non-heat plus implicit heat-conduction splitting.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100000,
        help="Maximum accepted timesteps before aborting.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
