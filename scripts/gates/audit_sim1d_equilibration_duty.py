"""Audit the DELIVERED gas-puff duty of the neutral-equilibration inner sim.

Item 37 (nn-IC diagnostician, 2026-07-27): in the equilibration inner sim
(``flags["Plasma"]=False``) the phase LOOKUP (``_phase_info``, untolerated
modulo) and the phase-boundary SCHEDULE (``next_phase_boundary_after``,
tolerated ``in_run_window``) could disagree about the puff-off instant, so the
puff stayed ON for one extra step and the equilibration over-fuelled relative
to its configured duty.

This script measures the duty the run loop ACTUALLY delivers, with no
re-derivation of the cap logic: ``next_phase_boundary_after`` is called exactly
once per step, at the step's start time, so recording its ``time`` argument
reconstructs the exact step grid.  The phase at each step start is the phase
that step's RHS runs under.

Usage (from <checkout>/cablp, PYTHONPATH set to the same cablp/):

    python scripts/audit_sim1d_equilibration_duty.py
    python scripts/audit_sim1d_equilibration_duty.py --family 0.0195 --nx 60
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cablp.solvers._sim1d import LAPDSim1D, default_config  # noqa: E402


def _instrument(sim, ledger):
    """Record every step-start time the run loop asks a boundary for."""
    original = sim.next_phase_boundary_after

    def recording(time, t_end=None, time_tol=0.0):
        ledger.append(float(time))
        return original(time, t_end=t_end, time_tol=time_tol)

    sim.next_phase_boundary_after = recording
    return sim


def measure(params, flags, cycles, dt, nx):
    """Return the delivered-duty ledger for a neutral-equilibration inner sim.

    Builds the inner sim exactly as ``run_neutral_equilibration`` does.
    """
    params = dict(params)
    flags = dict(flags)
    params["nx"] = int(nx)
    flags["Plasma"] = False
    flags["cathode_coupling"] = False
    flags["neutral_equilibration"] = False
    flags["launch_plasma_after_equilibration"] = False
    flags["use_cached_neutral_seed"] = False
    # The two DVM directed-recycle jets, cleared for the same reason as
    # cathode_coupling above: with no plasma and no cathode solve there is no
    # collected ion flux to split and no phi_c / phi_a to launch against, and
    # the construction guard behind each jet refuses an inner sim that arms one
    # without a cathode solve. Both are input_dict keys, so they go on ``params``.
    params["neutral_kinetic_dvm_cathode_jet"] = False
    params["neutral_kinetic_dvm_anode_jet"] = False
    params["nn0"] = 1e8
    params["cycles"] = int(cycles)
    tau_cycle = float(params.get("tau_cycle", 0.0))
    t_end = cycles * tau_cycle
    params["dt_save"] = t_end
    params["t_save_start"] = 0.0
    params["max_output_steps"] = 0

    sim = LAPDSim1D(params, flags)
    ledger = []
    _instrument(sim, ledger)
    result = sim.run(t_end=t_end, dt=dt)

    starts = np.asarray(ledger, dtype=float)
    edges = np.append(starts, float(result.final_time))
    widths = np.diff(edges)
    phases = np.array([sim.phase_at_time(t) for t in starts], dtype=str)
    on = phases == "equilibrium_puff"

    on_time = float(widths[on].sum())
    # SSPRK2/Heun evaluates the RHS at BOTH step endpoints with weight 1/2, so
    # the fuel the integrator actually books over a step is the trapezoid of the
    # puff indicator. Reported alongside the schedule ledger because a square
    # forcing sampled at endpoints inherently loses ~dt/2 per closing edge --
    # a property of the integrator, NOT of the phase schedule.
    edge_on = np.array(
        [sim.phase_at_time(t) == "equilibrium_puff" for t in edges], dtype=float
    )
    heun_on_time = float((0.5 * (edge_on[:-1] + edge_on[1:]) * widths).sum())
    configured_on = cycles * float(sim._equilibration_puff_on_duration()) if hasattr(
        sim, "_equilibration_puff_on_duration"
    ) else cycles * max(float(params.get("tau_discharge", 0.0)), 0.0)

    # Per-cycle breakdown so a single bad cycle is visible. Attribute a step to
    # the cycle its (lattice-snapped) start lies in.
    snap_tol = 1e-9 * tau_cycle
    cycle_index = np.floor((starts + snap_tol) / tau_cycle).astype(int)
    per_cycle = np.zeros(cycles)
    for k in range(cycles):
        sel = on & (cycle_index == k)
        per_cycle[k] = float(widths[sel].sum())

    final_nn = np.asarray(result.nn[-1], dtype=float)
    return {
        "steps": int(starts.size),
        "t_end": float(result.final_time),
        "on_time": on_time,
        "heun_on_time": heun_on_time,
        "configured_on": float(configured_on),
        "error_pct": 100.0 * (on_time / configured_on - 1.0) if configured_on else 0.0,
        "heun_error_pct": (
            100.0 * (heun_on_time / configured_on - 1.0) if configured_on else 0.0
        ),
        "per_cycle": per_cycle,
        "mean_nn": float(np.mean(final_nn)),
        "max_nn": float(np.max(final_nn)),
    }


FAMILIES = {
    "20ms": 20e-3,
    "0.0195": 0.0195,
    "0.0075": 0.0075,
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", default=None, help="tau_discharge [s] (default: all)")
    ap.add_argument("--cycles", type=int, default=100)
    ap.add_argument("--dt", type=float, default=None, help="default: config")
    ap.add_argument("--nx", type=int, default=60)
    ap.add_argument("--puff-on", type=float, default=None,
                    help="equilibration_gas_puff_on_s override [s]")
    ap.add_argument("--production", action="store_true",
                    help="layer compare_sim1d_es1 PARAM/FLAG_OVERRIDES (the "
                         "production point) on top of default_config")
    args = ap.parse_args(argv)

    base_params, base_flags = default_config()
    if args.production:
        from compare_sim1d_es1 import FLAG_OVERRIDES, PARAM_OVERRIDES

        base_params.update(PARAM_OVERRIDES)
        base_flags.update(FLAG_OVERRIDES)
        if args.family is None:
            args.family = str(base_params.get("tau_discharge", 20e-3))
    dt = args.dt if args.dt is not None else float(
        base_params.get("neutral_equilibration_dt", 1e-2)
    )

    if args.family is None:
        families = list(FAMILIES.items())
    else:
        families = [(args.family, float(args.family))]

    for name, tau_discharge in families:
        params = dict(base_params)
        params["tau_discharge"] = float(tau_discharge)
        if args.puff_on is not None:
            params["equilibration_gas_puff_on_s"] = float(args.puff_on)
        out = measure(params, dict(base_flags), args.cycles, dt, args.nx)
        bad = np.flatnonzero(
            np.abs(out["per_cycle"] - out["per_cycle"][0]) > 1e-15
        )
        print(
            f"family={name} tau_discharge={tau_discharge:g} dt={dt:g} "
            f"cycles={args.cycles} nx={args.nx} "
            f"puff_on_override={args.puff_on}"
        )
        print(
            f"  steps={out['steps']} t_end={out['t_end']:.9g} "
            f"delivered_on={out['on_time']:.9g} s "
            f"configured_on={out['configured_on']:.9g} s "
            f"error={out['error_pct']:+.4f}%"
        )
        print(
            f"  heun-weighted ON={out['heun_on_time']:.9g} s "
            f"error={out['heun_error_pct']:+.4f}%"
        )
        print(
            f"  per-cycle ON: min={out['per_cycle'].min():.9g} "
            f"max={out['per_cycle'].max():.9g} "
            f"n_cycles_differing_from_first={bad.size}"
        )
        print(
            f"  seed: mean_nn={out['mean_nn']:.9e} max_nn={out['max_nn']:.9e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
