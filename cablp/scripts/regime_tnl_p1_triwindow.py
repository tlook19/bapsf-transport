"""P1: the long-window tracer run under ``beam_product_transport``.

PRE-REGISTERED BEFORE THE SELECTOR WAS RUN. The scenario is the one of record
-- ``regime_r2_overlap_gate.build_config(nx, tracer_on=True)`` advanced to
t = 1e-4 s at a fixed dt = 1e-8 s -- and the only thing this file changes is
which ``beam_product_transport`` value the arm carries. Nothing here is tuned:
the stance, the window and the step come from the scenario, and the selector
comes from the command line.

REGISTERED OUTCOME BINS, read off the arm under test
-----------------------------------------------------
* BIN (i)   -- the ``TracerBalanceError`` refusal is GONE and the build
               PROCEEDS: at least one cell activates (the tracer hands it to
               the fluid). The long-window tracer program is unblocked on
               measured physics.
* BIN (ii)  -- the refusal is gone but the build DIES: the window runs to its
               end with no activation and gamma <= 0 everywhere, i.e. nothing
               is growing. Retaining the along-ray products is then not
               sufficient either, and that is the result.
* BIN (iii) -- the refusal PERSISTS. The terminal dump was not the whole
               story; the refusal is reported verbatim with the cell and the
               instant it fired on.

The bins are read from the solver's own state, never inferred from prose: the
refusal is the exception the tracer raises, the activations are
``_tracer_first_activation`` and the passive mask, and gamma is the census's
own ``gamma_per_s``.

The ``--baseline`` arm runs the identical scenario under the shipped default
``"local"`` in the same process, so the comparison is against a measurement
rather than against a remembered one. It is also the probe's own VACUITY
CHECK: the bins above only mean anything if the baseline reproduces the
refusal, and an arm under test that lands in the same bin as a baseline which
never refused has measured the scenario rather than the selector.

MEASURED, and the reason ``--dt 0`` exists: at the registered fixed
``dt = 1e-8`` this scenario never starts a discharge. The loop current at a
FIXED elapsed time falls away as the step is shrunk (t = 5e-6 s: 39.2 A
adaptive, 16.0 A at dt = 1e-6, 1.07 A at 1e-7, 0.92 A at 1e-8), the cathode
does not emit that low, the beam source ``S`` is identically zero, and both
arms sit at the density floor for the whole window. ``--dt 0`` runs the
solver's own adaptive stepping instead, which is what the refusal of record
was measured under (``regime_pb_balance_table.py`` section G advances the same
long window with no ``dt`` argument).

Usage (from <checkout>/cablp, with PYTHONPATH set to that same cablp):
    python scripts/regime_tnl_p1_triwindow.py --selector terminal_nonlocal
    python scripts/regime_tnl_p1_triwindow.py --selector terminal_nonlocal --dt 0
"""

import argparse
import sys
import time
import warnings

import numpy as np

from regime_r2_overlap_gate import build_config

from cablp.solvers._sim1d import LAPDSim1D
from cablp.solvers._sim1d.physics.tracer import TracerBalanceError


def run_arm(selector, nx, t_end, dt, dt_save, max_steps, chunk):
    """Advance one arm; return a dict describing which bin it landed in.

    Advanced over a PRECOMPUTED ladder of chunk targets so a long window
    reports progress while it runs rather than only at its end. ``run``
    continues from the state it left, so chunking changes no step: the sequence
    is the same fixed-dt one. The ladder is fixed rather than a
    ``while sim._time < t_end`` loop because the solver lands on its target to
    within float rounding, and a comparison against that lands the loop on a
    target it can no longer advance past.
    """
    params, flags = build_config(nx, True, {"beam_product_transport": selector})
    params["dt_save"] = dt_save
    sim = LAPDSim1D(params, flags)
    out = {"selector": selector, "cells": int(sim.geometry.cells)}
    started = time.time()
    steps = max(1, int(round(t_end / chunk)))
    targets = [t_end * (k + 1) / steps for k in range(steps)]
    targets[-1] = t_end
    # dt = 0 means the solver's own adaptive stepping (the argument is simply
    # not passed), which is how the scenario of record was advanced.
    step_kwargs = {} if dt <= 0.0 else {"dt": dt}
    try:
        for target in targets:
            sim.run(t_end=target, max_steps=max_steps, **step_kwargs)
            print(
                f"     ... t={sim._time:.6g} s  "
                f"passive={int(np.count_nonzero(sim._tracer_passive))}  "
                f"({time.time() - started:.0f} s wall)",
                flush=True,
            )
    except TracerBalanceError as error:
        out["bin"] = "iii"
        out["refusal"] = str(error)
        out["t_refused"] = float(sim._time)
        passive = np.asarray(sim._tracer_passive, dtype=bool)
        active = np.asarray(sim.geometry.plasma_active, dtype=bool)
        out["handed_over"] = np.flatnonzero(active & ~passive).tolist()
        out["n"] = np.asarray(sim.state.n, dtype=float)
        out["I_loop"] = float(sim._circuit_I_loop)
        out["sim"] = sim
        return out
    out["t_reached"] = float(sim._time)
    census = sim._tracer_census or {}
    gamma = np.asarray(census.get("gamma_per_s", np.zeros(0)), dtype=float)
    passive = np.asarray(sim._tracer_passive, dtype=bool)
    active = np.asarray(sim.geometry.plasma_active, dtype=bool)
    handed = np.flatnonzero(active & ~passive)
    out["handed_over"] = handed.tolist()
    out["first_activation"] = sim._tracer_first_activation
    out["gamma_max"] = float(np.max(gamma)) if gamma.size else float("nan")
    out["n"] = np.asarray(sim.state.n, dtype=float)
    out["census"] = sim._tracer_census_line()
    out["bin"] = "i" if handed.size else "ii"
    out["I_loop"] = float(sim._circuit_I_loop)
    # The beam source the balance dilutes into. Zero means no discharge
    # started at all, which is the difference between a measured bin (ii) and
    # a scenario that never ran.
    out["S_max"] = float(np.max(sim._tracer_prepare(1.0e-8)["S"]))
    out["sim"] = sim
    return out


def report(out):
    """Print one arm's outcome in the registered bin language."""
    print(f"   arm beam_product_transport={out['selector']!r}: "
          f"BIN ({out['bin']})")
    if out["bin"] == "iii":
        print(f"     REFUSED at t = {out['t_refused']:.6g} s")
        print(f"     {out['refusal']}")
        print(f"     cells handed to the fluid: {out['handed_over']}")
        n = out["n"]
        print(f"     n range at refusal: {float(np.min(n)):.4g} .. "
              f"{float(np.max(n)):.4g} cm^-3, "
              f"I_loop = {out['I_loop']:.6g} A")
        return
    n = out["n"]
    print(f"     ran to t = {out['t_reached']:.6g} s with NO refusal")
    print(f"     I_loop = {out['I_loop']:.6g} A, beam source S max = "
          f"{out['S_max']:.6g} cm^-3 s^-1"
          + ("   <-- NO DISCHARGE STARTED: this arm measures the scenario, "
             "not the selector" if out["S_max"] == 0.0 else ""))
    print(f"     cells handed to the fluid: {out['handed_over']}")
    first = out["first_activation"]
    print("     first activation: "
          + ("none" if first is None
             else f"t={first[0]:.6g} s cell {first[1]} on {first[2]}"))
    print(f"     max growth rate gamma = {out['gamma_max']:.6g} 1/s")
    print(f"     n range: {float(np.min(n)):.4g} .. {float(np.max(n)):.4g} "
          "cm^-3")
    if out["census"]:
        print(f"     {out['census']}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selector",
        default="terminal_nonlocal",
        choices=("local", "nonlocal", "terminal_nonlocal"),
    )
    parser.add_argument("--nx", type=int, default=20)
    parser.add_argument("--t-end", type=float, default=1.0e-4)
    parser.add_argument("--dt", type=float, default=1.0e-8)
    parser.add_argument("--dt-save", type=float, default=1.0e-5)
    parser.add_argument("--max-steps", type=int, default=2000000)
    parser.add_argument("--chunk", type=float, default=1.0e-5,
                        help="progress-report interval [s of sim time]")
    parser.add_argument("--baseline", dest="baseline", action="store_true",
                        default=True,
                        help="also run the shipped 'local' arm (default)")
    parser.add_argument("--no-baseline", dest="baseline",
                        action="store_false")
    args = parser.parse_args(argv)
    warnings.simplefilter("ignore")

    print("== regime_tnl P1: the long-window tri-bin probe")
    print(f"   scenario regime_r2_overlap_gate.build_config({args.nx}, True), "
          f"t_end={args.t_end:g} s, dt={args.dt:g} s")
    print("   bins: (i) refusal gone AND cells activate; (ii) refusal gone, "
          "no activation, gamma <= 0; (iii) refusal persists")

    arms = []
    if args.baseline and args.selector != "local":
        print()
        print("-- baseline arm (the shipped default, the scenario of record)")
        arms.append(run_arm("local", args.nx, args.t_end, args.dt,
                            args.dt_save, args.max_steps, args.chunk))
        report(arms[-1])
    print()
    print("-- arm under test")
    arms.append(run_arm(args.selector, args.nx, args.t_end, args.dt,
                        args.dt_save, args.max_steps, args.chunk))
    report(arms[-1])

    print()
    print("== outcome: " + ", ".join(
        f"{a['selector']} -> BIN ({a['bin']})" for a in arms
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
