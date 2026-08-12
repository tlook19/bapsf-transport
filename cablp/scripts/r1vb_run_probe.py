"""R1 ON-path probe: the conducting-phase build leg with the circuit bound.

Reuses ``covbuild_run_conducting_phase.build_config`` verbatim -- the covdecide
recipe family's configuration, unchanged -- and adds ONE delta:
``cathode_circuit_voltage_bound`` in the FLAGS namespace. It needs its own
entry point only because that driver's ``--extra`` writes into ``input_dict``,
and a flag put there would silently do nothing (the two namespaces do not
validate each other's keys).

Usage (from <checkout>/cablp, with PYTHONPATH set to that same cablp):
    python scripts/r1vb_run_probe.py --nx 60 --t-end 1e-4 --max-steps 20000 \
        --bound on --save-h5 scripts/r1vb_probe_on.h5
"""

import argparse
import json

import numpy as np

from cablp.solvers._sim1d import LAPDSim1D, ProgressPrinter1D
from cablp.solvers._sim1d.results.io import save_result_hdf5

from covbuild_run_conducting_phase import build_config


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--nx", type=int, default=60)
    p.add_argument("--t-end", type=float, default=1.0e-4)
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--bound", choices=("on", "off"), required=True)
    p.add_argument("--coverage", default="0.05,1390")
    p.add_argument("--extra", nargs="*", default=())
    p.add_argument("--save-h5", required=True)
    args = p.parse_args(argv)

    parts = args.coverage.split(",")
    coverage = (float(parts[0]), float(parts[1]) if len(parts) > 1 else 0.0)
    extra = {"max_steps_action": "stop",
             "heating_anomalous_transport": "tail_walk",
             "heating_anomalous_tail_ionization": "on"}
    for kv in args.extra:
        k, v = kv.split("=", 1)
        try:
            extra[k] = json.loads(v)
        except json.JSONDecodeError:
            extra[k] = v

    params, flags = build_config(args.nx, coverage=coverage, extra=extra)
    flags["cathode_circuit_voltage_bound"] = args.bound == "on"

    sim = LAPDSim1D(params, flags)
    print(
        f"bound={args.bound} nx={sim.geometry.cells} "
        f"V_bank={params['V_bank']} R_comp={params['R_comp']} "
        f"R_mesh_ohm={params.get('R_mesh_ohm', 0.0)} "
        f"cap={params['cathode_phi_c_cap_V']}",
        flush=True,
    )
    sim.start_simulation(
        t_end=args.t_end,
        max_steps=args.max_steps,
        progress_tracker=ProgressPrinter1D(),
        progress_interval_s=60.0,
    )
    result = sim.get_results()
    save_result_hdf5(args.save_h5, result, params=params, flags=flags)
    times = np.asarray(result.time, dtype=float)
    I_loop = np.asarray(
        result.cathode_diagnostics["circuit_I_loop"], dtype=float
    )
    print(
        f"window: 0 -> {times[-1] * 1e3:.6f} ms, {times.size} saves; "
        f"I_loop max {I_loop.max():.6g} A, final {I_loop[-1]:.6g} A"
    )
    print(f"saved {args.save_h5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
