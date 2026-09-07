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

RESOLUTION IS A PRECONDITION, not a detail. The seeded state's fastest
conduction mode has lambda_max ~ 1.5e8 s^-1, so at the harness's old default of
8 base-steps the coarsest step ran at dt*lambda_max ~ 18. Nothing about a
scheme's TRUNCATION error is observable there: a Richardson triplet at large
|z| measures the substep's stability function instead, and the two are not the
same ranking. Crank-Nicolson is the trap -- its Strang equilibrium error is
exactly -z^2/16, a clean power law, so it returns 2.00 at EVERY |z| however
unresolved, while the L-stable schemes beside it read ~1. A table built there
says tr_bdf2 is not second order, and that reading is an artefact of the
sampling, not a property of the scheme.

So ``--base-steps`` now DEFAULTS to a value derived from the seed's own
conduction operator (see :func:`resolved_base_steps`), chosen so the coarsest
dt satisfies ``dt * lambda_max <= DT_LAMBDA_RESOLVED``. Every reading is
printed with its ``dt * lambda_max``, a triplet coarser than
``DT_LAMBDA_FLAG`` is flagged PRE-ASYMPTOTIC, and the closing line reports
both preconditions -- floors inert AND the stiff mode resolved.

Measured at 72 cells, t_end = 1e-6 s, in the RESOLVED regime the default now
selects (base-steps 128, dt*lambda_max = 1.15 / 0.58 / 0.29), as the
reference-free triplet order:

    picard  splitting   backward_euler  shifted  crank_nicolson  tr_bdf2
    ------  ---------   --------------  -------  --------------  -------
      0       lie            0.99         1.00        1.00         0.95
      4       lie            0.99         1.00        1.00         0.95
      0       strang         0.82         1.45        1.79         1.74
      4       strang         0.81         1.45        2.00         1.98

Second order needs all three of a second-order substep scheme, a non-frozen
conductivity, and Strang splitting. Each of the first-order terms caps the step
on its own, so knocking out only one changes nothing: --picard alone is still
capped by Lie splitting, which is why every Lie row sits at ~1.0 whatever the
substep scheme is, and --splitting strang alone is still capped by the frozen
conductivity, which is the 1.74-1.79 of the picard-0 Strang row.

The bottom row is the shipped production package, and both second-order
L-stable-or-symmetric substeps reach second order in it: crank_nicolson 2.00
(2.00 -> 2.02 against the reference) and tr_bdf2 1.98 (1.99 -> 2.02). tr_bdf2
is the shipped choice because it is the only one that is second-order AND
L-stable.

backward_euler is the negative control: theta = 1 cannot be second-order at any
dt, so if it reaches 2.0 the harness is wrong rather than good. shifted is
theta = 0.6, first-order for the same reason, and its 1.45 on the Strang rows
is the same band effect described above rather than an order claim -- its
leading first-order coefficient is (theta - 1/2) = 0.1 of backward Euler's, so
the second-order term still contributes at these dt. Read shifted as a scale
check.

EXIT CODE. 0 when BOTH preconditions hold -- floors inert in every run and the
stiffest conduction mode resolved at every dt of the triplet -- and 1 when
either fails, in which case the orders printed are not measurements and the
closing lines say which precondition failed. The exit code gates the
PRECONDITIONS only: which order value counts as passing depends on the scheme,
the splitting and the Picard count being asked about, and is the caller's
question. Whether the numbers mean anything at all is not.

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
from cablp.solvers._sim1d.core.state import (
    NEUTRAL_ENERGY_FLOOR_T_K,
    conservative_from_primitives,
    neutral_energy_floor,
    pack_state,
)
from cablp.solvers._sim1d.physics.conduction import (
    HEAT_DT_FRACTION,
    IMPLICIT_HEAT_SCHEMES,
    heat_conduction_timestep_bound,
)
from cablp.constants import ev_to_erg

FLOOR_RTOL = 1e-9

#: The coarsest ``dt * lambda_max`` at which a Richardson triplet is inside the
#: asymptotic regime, and the value the default ``--base-steps`` is derived to
#: satisfy. Above it the stiff conduction modes are not resolved and the
#: measured order is a property of each scheme's stability function at large
#: ``|z|``, not of its truncation error.
DT_LAMBDA_RESOLVED = 2.0

#: The coarsest ``dt * lambda_max`` beyond which a triplet is reported as
#: PRE-ASYMPTOTIC and its orders as not meaningful.
DT_LAMBDA_FLAG = 4.0

#: Neutral temperature [K] the seed puts the optional ``En`` row at. The ``En``
#: floor clips up to the vessel wall, so a seed AT the wall would sit on its own
#: floor from the first step and any cooling would clip; this leaves the same
#: order-of-magnitude headroom the other fields have.
SEED_TN_K = 4.0 * NEUTRAL_ENERGY_FLOOR_T_K

#: Amplitude of the seeded neutral drift [cm/s], subsonic against the neutral
#: thermal speed at :data:`SEED_TN_K` so the optional momentum row carries a
#: gradient without putting the neutral fluxes in an unrepresentative regime.
SEED_UN_CM_S = 1.0e4

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
            nn_safe = np.maximum(np.asarray(state.nn, dtype=float), floors["nn"])
            raw_Te = (2.0 / 3.0) * np.asarray(state.Ee, dtype=float) / (
                n_safe * ev_to_erg
            )
            raw_Ti = (2.0 / 3.0) * np.asarray(state.Ei, dtype=float) / (
                n_safe * ev_to_erg
            )
            watched = [
                (raw_Te, floors["Te"]),
                (raw_Ti, floors["Ti"]),
                (np.asarray(state.n, dtype=float), floors["n"]),
                (np.asarray(state.nn, dtype=float), floors["nn"]),
            ]
            # The optional rows carry floors of their own wherever the layout
            # includes them: nn_a takes the nn floor, and En takes the wall
            # floor evaluated against the FLOORED nn, exactly as
            # apply_state_floors does. An unwatched floor would let a clipped
            # run report a meaningless order as a meaningful one.
            if state.nn_a is not None:
                watched.append(
                    (np.asarray(state.nn_a, dtype=float), floors["nn"])
                )
            if state.En is not None:
                watched.append(
                    (
                        np.asarray(state.En, dtype=float),
                        neutral_energy_floor(nn_safe),
                    )
                )
            for value, floor in watched:
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
    """Return a smooth non-uniform state in the solver's own packed layout.

    A uniform state is the null mode of the conduction operator (K*1 = 0) and
    carries no gradients for the fluxes either, so it would make convergence
    trivially perfect and measure nothing.

    The packed state is FIVE base rows plus whichever optional neutral rows the
    construction declares (``M_n``, ``nn_a``, ``M_n_a``, ``En``), so a seed that
    supplies only the base rows is rejected by the solver as a width mismatch.
    The seed is therefore presence-gated on the rows the solver's OWN initial
    state carries -- never hand-packed -- and each optional row is built by the
    same constructor the solver builds its initial state with. The optional
    conventions follow that initial state: both neutral zones at one density,
    and a uniform neutral temperature, raised off the wall floor here so the
    ``En`` clip stays inert (see :data:`SEED_TN_K`).
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
    un = SEED_UN_CM_S * np.sin(phase + 1.3)
    return conservative_from_primitives(
        n,
        nn,
        u,
        Te,
        Ti,
        sim._ion_mass_g,
        un=None if base.M_n is None else un,
        nn_a=None if base.nn_a is None else nn.copy(),
        un_a=None if base.M_n_a is None else un.copy(),
        Tn_K=None if base.En is None else SEED_TN_K,
    )


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


def build_sim(scheme, picard, splitting):
    """Return a solver configured exactly as :func:`run_fixed_dt` configures it."""
    params, flags = default_config()
    params.update(CLEAN_PARAMS)
    params["implicit_heat_scheme"] = scheme
    params["heat_picard_iterations"] = picard
    params["operator_splitting"] = splitting
    flags.update(CLEAN_FLAGS)
    return LAPDSim1D(params, flags)


def seed_conduction_lambda_max(scheme, amplitude, picard, splitting):
    """Return the fastest conduction rate [s^-1] of the SEEDED state.

    Read out of the solver's own explicit heat bound rather than restated: that
    bound is ``HEAT_DT_FRACTION / max(cell_coeff)``, and ``max(cell_coeff)`` is
    exactly the largest diagonal rate of the conduction operator this harness's
    implicit substep has to integrate. Inverting the bound therefore reuses the
    operator the solver actually builds -- one source, so the two cannot drift.

    The conduction operator does not depend on which substep scheme integrates
    it, so this is a property of the seed alone and is computed once per run.
    """
    sim = build_sim(scheme, picard, splitting)
    dt_bound = heat_conduction_timestep_bound(
        seeded_state(sim, amplitude),
        floors=sim.floors,
        ion_mass_g=sim.ion_mass_g,
        mu=sim.mu,
        geometry=sim._geometry,
        **sim._heat_conduction_kwargs(),
    )
    if not np.isfinite(dt_bound) or dt_bound <= 0.0:
        raise RuntimeError(
            "the seeded state has no finite explicit heat bound, so the "
            f"resolution of the triplet cannot be established (got {dt_bound})"
        )
    return HEAT_DT_FRACTION / float(dt_bound)


def resolved_base_steps(t_end, lambda_max):
    """Return the smallest power-of-two base-steps that resolves the seed.

    "Resolves" means the COARSEST dt of the triplet satisfies
    ``dt * lambda_max <= DT_LAMBDA_RESOLVED``. A power of two is used so the
    triplet's successive halvings and the reference multiple all land on
    exactly representable steps.
    """
    need = t_end * float(lambda_max) / DT_LAMBDA_RESOLVED
    n = 1
    while n < need:
        n *= 2
    return n


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
    parser.add_argument(
        "--base-steps",
        type=int,
        default=None,
        help=(
            "coarsest step count of the triplet; default is DERIVED from the "
            "seed's own conduction rate so the coarsest dt*lambda_max is at "
            "or below the resolved bound"
        ),
    )
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
        default=8,
        help="reference run uses base-steps*4*ref-factor steps",
    )
    args = parser.parse_args(argv)

    lambda_max = seed_conduction_lambda_max(
        args.schemes[0], args.amplitude, args.picard, args.splitting
    )
    if args.base_steps is None:
        N = resolved_base_steps(args.t_end, lambda_max)
        base_steps_origin = "derived"
    else:
        N = args.base_steps
        base_steps_origin = "given"
    counts = (N, 2 * N, 4 * N)
    ref_steps = 4 * N * args.ref_factor
    z_coarse = args.t_end / counts[0] * lambda_max
    resolved = z_coarse <= DT_LAMBDA_RESOLVED
    pre_asymptotic = z_coarse > DT_LAMBDA_FLAG

    print("=" * 76)
    print("sim1d SPLIT-STEP TEMPORAL ORDER")
    print("=" * 76)
    print(f"t_end={args.t_end:.2e} s   steps={counts}   reference={ref_steps} steps")
    print(f"heat_picard_iterations={args.picard}  operator_splitting={args.splitting}")
    print(f"dt from {args.t_end/counts[0]:.3e} s down to {args.t_end/counts[-1]:.3e} s")
    print("regime: fixed dt, floors inert, single phase, autonomous RHS, no cathode")
    print(
        f"seed conduction lambda_max = {lambda_max:.4e} s^-1 "
        f"(from the solver's own explicit heat bound)"
    )
    print(
        f"base-steps={N} ({base_steps_origin}); dt*lambda_max = "
        + ", ".join(f"{args.t_end/n*lambda_max:.2f}" for n in counts)
        + f"   [resolved bound {DT_LAMBDA_RESOLVED:.0f}, "
        f"flag above {DT_LAMBDA_FLAG:.0f}]"
    )
    if pre_asymptotic:
        print(
            f"  *** PRE-ASYMPTOTIC: coarsest dt*lambda_max = {z_coarse:.2f} > "
            f"{DT_LAMBDA_FLAG:.0f}. ORDERS BELOW ARE NOT MEANINGFUL -- at this "
            "stiffness each\n      scheme reports its stability function at "
            "large |z|, not its truncation error."
        )

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
        if pre_asymptotic:
            flag += (
                f"   <-- PRE-ASYMPTOTIC (dt*lambda_max={z_coarse:.2f}): "
                "orders not meaningful"
            )
        print(f"\n--- {scheme} ---{flag}")
        print(
            f"  triplet order            : {triplet:.2f}"
            f"   [coarsest dt*lambda_max = {z_coarse:.2f}]"
        )
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
    # BOTH preconditions have to hold. Floors are a non-smooth projection and
    # break order wherever they bind; an unresolved stiff mode replaces the
    # truncation error with the substep's large-|z| stability behaviour. Either
    # one alone leaves the orders unreadable, so the closing line states both.
    print(f"precondition 1, floors inert                  : "
          f"{'no' if any_dirty else 'yes'}")
    print(f"precondition 2, coarsest dt*lambda_max <= "
          f"{DT_LAMBDA_RESOLVED:.0f} : "
          f"{'yes' if resolved else 'no'} ({z_coarse:.2f})")
    if any_dirty and not resolved:
        print("NEITHER precondition holds: the orders above are not meaningful.")
        print("Raise --amplitude headroom, and raise --base-steps.")
    elif any_dirty:
        print("At least one run activated a floor: those orders are meaningless.")
        print("Raise --amplitude headroom or shorten --t-end.")
    elif not resolved:
        print(
            "The stiffest conduction mode is UNRESOLVED at the coarsest dt: the "
            "orders above are\nnot meaningful. Raise --base-steps (the default "
            "derives one that resolves it)."
        )
    else:
        print(
            "Floors stayed inert in every run AND the stiffest conduction mode "
            "is resolved at\nevery dt of the triplet: the orders above are "
            "meaningful."
        )
    print("-" * 76)
    # THE EXIT CODE CARRIES THE PRECONDITIONS. This harness exited 0 whichever
    # way the two lines above read, so a run whose orders it had just declared
    # NOT MEANINGFUL was indistinguishable, to anything reading exit codes,
    # from one whose orders it stood behind. The labelled output says which it
    # was; the exit code now says the same thing.
    #
    # It is the PRECONDITIONS this gates on, not the order values: what number
    # counts as passing is the caller's question and depends on the scheme,
    # the splitting and the Picard count being asked about. What is not the
    # caller's question is whether the numbers mean anything at all.
    ok = not any_dirty and resolved
    if not ok:
        print(
            "EXIT 1: at least one precondition failed, so the orders above "
            "are not measurements."
        )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
