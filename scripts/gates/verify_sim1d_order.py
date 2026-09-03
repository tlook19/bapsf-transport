"""Measure the observed temporal order of the sim1d operator-split step.

This is a verification harness, not a physics run. It deliberately builds a
configuration in which order is *meaningful*, which a production discharge is
not:

* **Fixed dt.** Adaptive stepping, retries and the growth limiter are all off,
  so every run in a refinement sequence takes exactly the steps it says.
* **Floors inert.** The initial state is hot and dense, far from every floor.
  Floors are non-smooth projections; wherever one binds, local order collapses.
  The harness watches them and reports the measurement as INVALID if any bind.
* **Single phase.** Phase transitions are threshold-triggered in production,
  which makes the RHS discontinuous and order undefined across the transition.
  Here the whole window sits inside the main discharge.
* **Autonomous RHS.** The gas puff and pump are off, so no explicitly
  time-dependent forcing contributes its own error term.
* **No cathode.** The cathode solve carries a continuation cache between steps,
  which would make a run's result depend on its step history and break
  self-convergence.

Order is estimated from a grid triplet (Richardson), which needs no reference
solution:

    order = log2( ||u_N - u_2N|| / ||u_2N - u_4N|| )

A reference-based estimate against a much finer run is reported alongside it as
a cross-check; the two should agree.

Measured at 62 cells with t_end = 1e-6 s, dt from 1.25e-7 down to 3.13e-8:

    picard  splitting   backward_euler  shifted  crank_nicolson  tr_bdf2
    ------  ---------   --------------  -------  --------------  -------
      0       lie            0.97         1.01        1.01         1.02
      4       lie            0.97         1.00        1.01         1.02
      0       strang         0.98         0.98        1.04         0.98
      4       strang         0.99         0.96        1.99         2.00

Second order needs all three of a second-order substep scheme, a non-frozen
conductivity, and Strang splitting. Each of the first-order terms caps the step
on its own, so knocking out only one changes nothing: --picard alone is still
capped by Lie splitting, and --splitting strang alone is still capped by the
frozen conductivity. Only the last row has none of them.

backward_euler and shifted staying at ~1.0 in every row is the negative
control. Neither can be second-order at any dt, so if either reaches 2.0, the
harness or the scheme is wrong rather than good.

Usage:
    python scripts/gates/verify_sim1d_order.py
    python scripts/gates/verify_sim1d_order.py --picard 4 --splitting strang
    python scripts/gates/verify_sim1d_order.py --schemes crank_nicolson tr_bdf2
    python scripts/gates/verify_sim1d_order.py --t-end 2e-6 --base-steps 8
"""

import argparse
import sys

import numpy as np

import cablp.solvers._sim1d.core.state as state_mod
from cablp.solvers._sim1d import LAPDSim1D, default_config
from cablp.solvers._sim1d.core.state import conservative_from_primitives, pack_state
from cablp.solvers._sim1d.physics.conduction import IMPLICIT_HEAT_SCHEMES
from cablp.constants import ev_to_erg

FLOOR_RTOL = 1e-9

# A state sitting on a floor round-trips to within a few ULP, so only deficits
# deeper than FLOOR_RTOL count as a floor actually doing work.
CLEAN_PARAMS = {
    # Hot and dense: keep every floor far away so the limiters stay inert.
    "ne0": 1e12,
    "nn0": 1e13,
    "Te0": 5.0,
    "Ti0": 2.0,
    "u0": 0.0,
    # No time-dependent forcing -- an autonomous RHS isolates the split step.
    "gas_puff_enabled": False,
    "pump_enabled": False,
    # One phase for the whole window: no threshold discontinuity.
    "phase_transition_mode": "scheduled",
    "tau_neutral_prebreakdown": 0.0,
    "tau_prebreakdown": 0.0,
    "tau_breakdown": 0.0,
    "tau_discharge": 1.0,
    "tau_afterglow": 0.0,
    # Fixed dt: no retries, no growth limiting, no clamping of the test dt.
    "adaptive_retries_enabled": False,
    "dt_growth_enabled": False,
    "dt_min": 1e-16,
    "dt_max": 1.0,
    "max_density_step_fraction": 0.0,
    "max_neutral_step_fraction": 0.0,
    "max_energy_step_fraction": 0.0,
}

CLEAN_FLAGS = {
    "Plasma": True,
    "implicit_heat_conduction": True,
    "neutral_prebreakdown": False,
    "neutral_equilibration": False,
    "launch_plasma_after_equilibration": False,
    # The cathode solve caches a continuation guess across steps, which would
    # make a run depend on its own step history.
    "cathode_coupling": False,
    "debug_checks": False,
}

FIELDS = ("n", "nn", "u", "Te", "Ti")


class FloorWatch:
    """Count floor activations, to invalidate a run whose regime is not clean."""

    def __init__(self):
        self.clips = 0

    def __enter__(self):
        self._orig = state_mod.apply_state_floors

        def probe(state, floors, ion_mass_g):
            n_safe = np.maximum(np.asarray(state.n, dtype=float), floors["n"])
            raw_Te = (2.0 / 3.0) * np.asarray(state.Ee, dtype=float) / (
                n_safe * ev_to_erg
            )
            raw_Ti = (2.0 / 3.0) * np.asarray(state.Ei, dtype=float) / (
                n_safe * ev_to_erg
            )
            for value, floor in (
                (raw_Te, floors["Te"]),
                (raw_Ti, floors["Ti"]),
                (np.asarray(state.n, dtype=float), floors["n"]),
                (np.asarray(state.nn, dtype=float), floors["nn"]),
            ):
                self.clips += int(
                    np.count_nonzero(value < floor * (1.0 - FLOOR_RTOL))
                )
            return self._orig(state, floors=floors, ion_mass_g=ion_mass_g)

        state_mod.apply_state_floors = probe
        return self

    def __exit__(self, *exc):
        state_mod.apply_state_floors = self._orig
        return False


def seeded_state(sim, amplitude):
    """Return a smooth non-uniform state.

    A uniform state is the null mode of the conduction operator (K*1 = 0) and
    carries no gradients for the fluxes either, so it would make convergence
    trivially perfect and measure nothing.
    """
    z = np.asarray(sim._geometry.z_cm, dtype=float)
    span = z[-1] - z[0]
    phase = 2.0 * np.pi * (z - z[0]) / span
    base = sim.state
    n = np.asarray(base.n, dtype=float) * (1.0 + amplitude * np.sin(phase))
    nn = np.asarray(base.nn, dtype=float) * (1.0 + amplitude * np.cos(phase))
    Te = 5.0 * (1.0 + amplitude * np.sin(phase))
    Ti = 2.0 * (1.0 + amplitude * np.sin(phase + 0.7))
    u = 1.0e5 * np.sin(phase)  # subsonic; c_s ~ 1e6 cm/s at these temperatures
    return conservative_from_primitives(n, nn, u, Te, Ti, sim._ion_mass_g)


def run_fixed_dt(scheme, nsteps, t_end, amplitude, picard=0, splitting="lie"):
    """Advance the split step nsteps times at fixed dt; return primitive fields."""
    params, flags = default_config()
    params.update(CLEAN_PARAMS)
    params["implicit_heat_scheme"] = scheme
    params["heat_picard_iterations"] = picard
    params["operator_splitting"] = splitting
    flags.update(CLEAN_FLAGS)

    sim = LAPDSim1D(params, flags)
    sim._set_state_vector(pack_state(seeded_state(sim, amplitude)))

    dt = t_end / nsteps
    watch = FloorWatch()
    with watch:
        for _ in range(nsteps):
            sim.advance_one_step(dt=dt)

    derived = sim.derived
    state = sim.state
    fields = {
        "n": np.asarray(state.n, dtype=float),
        "nn": np.asarray(state.nn, dtype=float),
        "u": np.asarray(derived.u, dtype=float),
        "Te": np.asarray(derived.Te, dtype=float),
        "Ti": np.asarray(derived.Ti, dtype=float),
    }
    if not all(np.all(np.isfinite(v)) for v in fields.values()):
        raise RuntimeError(f"{scheme} at nsteps={nsteps} produced non-finite state")
    return fields, watch.clips


def rel_diff(a, b):
    """Relative L-inf difference, scaled by the magnitude of b."""
    scale = np.max(np.abs(b))
    if scale == 0.0:
        return float(np.max(np.abs(a - b)))
    return float(np.max(np.abs(a - b)) / scale)


def combined(fa, fb):
    return max(rel_diff(fa[k], fb[k]) for k in FIELDS)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schemes", nargs="+", default=list(IMPLICIT_HEAT_SCHEMES))
    parser.add_argument("--t-end", type=float, default=1.0e-6)
    parser.add_argument("--base-steps", type=int, default=8)
    parser.add_argument("--amplitude", type=float, default=0.3)
    parser.add_argument(
        "--picard",
        type=int,
        default=0,
        help="heat_picard_iterations (0 = freeze kappa at the step start)",
    )
    parser.add_argument("--splitting", default="lie", choices=("lie", "strang"))
    parser.add_argument(
        "--ref-factor",
        type=int,
        default=32,
        help="reference run uses base-steps*4*ref-factor steps",
    )
    args = parser.parse_args(argv)

    N = args.base_steps
    counts = (N, 2 * N, 4 * N)
    ref_steps = 4 * N * args.ref_factor

    print("=" * 76)
    print("sim1d SPLIT-STEP TEMPORAL ORDER")
    print("=" * 76)
    print(f"t_end={args.t_end:.2e} s   steps={counts}   reference={ref_steps} steps")
    print(f"heat_picard_iterations={args.picard}  operator_splitting={args.splitting}")
    print(f"dt from {args.t_end/counts[0]:.3e} s down to {args.t_end/counts[-1]:.3e} s")
    print("regime: fixed dt, floors inert, single phase, autonomous RHS, no cathode")

    any_dirty = False
    for scheme in args.schemes:
        runs, clips = {}, 0
        for n in counts:
            runs[n], c = run_fixed_dt(
                scheme, n, args.t_end, args.amplitude, args.picard, args.splitting
            )
            clips += c
        ref, c = run_fixed_dt(
            scheme, ref_steps, args.t_end, args.amplitude, args.picard,
            args.splitting,
        )
        clips += c

        # Richardson triplet: needs no reference solution.
        d1 = combined(runs[counts[0]], runs[counts[1]])
        d2 = combined(runs[counts[1]], runs[counts[2]])
        triplet = np.log2(d1 / d2) if d2 > 0 else float("nan")

        # Reference-based cross-check.
        e = [combined(runs[n], ref) for n in counts]
        ref_rates = [
            np.log2(e[i] / e[i + 1]) if e[i + 1] > 0 else float("nan")
            for i in range(len(e) - 1)
        ]

        flag = ""
        if clips:
            flag = f"   <-- INVALID: {clips} floor activations"
            any_dirty = True
        print(f"\n--- {scheme} ---{flag}")
        print(f"  triplet order            : {triplet:.2f}")
        print(f"  vs reference, per field  :")
        for k in FIELDS:
            ek = [rel_diff(runs[n][k], ref[k]) for n in counts]
            rk = [
                np.log2(ek[i] / ek[i + 1]) if ek[i + 1] > 0 else float("nan")
                for i in range(len(ek) - 1)
            ]
            print(
                f"    {k:3} err={[f'{x:.2e}' for x in ek]} "
                f"order={[f'{x:.2f}' for x in rk]}"
            )
        print(f"  combined ref order       : {[f'{r:.2f}' for r in ref_rates]}")

    print("\n" + "-" * 76)
    if any_dirty:
        print("At least one run activated a floor: those orders are meaningless.")
        print("Raise --amplitude headroom or shorten --t-end.")
    else:
        print("Floors stayed inert in every run: the orders above are meaningful.")
    print("-" * 76)
    return 0


if __name__ == "__main__":
    sys.exit(main())
