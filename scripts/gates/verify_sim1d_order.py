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

THE QUOTED ORDER IS THE SLOPE OF AN ERROR ENVELOPE. Each scheme runs at
``--levels`` (at least four) step counts ``N, 2N, 4N, ...``; each level's
error, per field, is its relative L-inf distance from a converged reference;
and the quoted order is the least-squares slope of ``log(error)`` against
``log(dt)`` over all the levels, printed with the slope's standard error and
the largest residual about the fitted line (:func:`envelope_fit`).

The reference is the Picard-4 second-order package (``tr_bdf2``, Picard 4,
Strang) at the finest level's dt divided by ``--ref-factor`` (default 16).
Every scheme, splitting and Picard count the levels measure converges to the
same semi-discrete solution, so one reference serves them all. At its nominal
second order the reference's error is 16**-2 = 1/256 of a comparable
second-order scheme's finest-level error, and smaller still against a
first-order one. That margin is CHECKED on every run: a second reference at
half the reference dt gives ``d = |ref - check|``, and for any reference order
``q >= 1`` the reference's own error is ``d / (1 - 2**-q) <= 2d``, the bound
printed per field.

A Richardson triplet ratio, ``log2(|u_N - u_2N| / |u_2N - u_4N|)``, is still
printed for every consecutive level triple, labelled a SCREEN, and is never the
quoted order. It needs no reference, but on a state whose error is not smooth
in dt -- a limited electron heat flux whose gradient changes sign inside the
window, where the step that crosses the change carries a phase-dependent
O(dt**2) residual -- a triplet reads erratically at ANY dt (0.88 on one such
state) while the envelope against a converged reference keeps a clean slope
(1.96 over eight halvings). A screen that disagrees with its envelope says to
look; it never replaces the envelope.

PRE-ASYMPTOTIC (:func:`pre_asymptotic_reasons`). A field's envelope is
labelled PRE-ASYMPTOTIC, its slope printed but NOT QUOTED, when any of:

1. an error is zero or non-finite at some level (no slope to fit);
2. the smallest level error is below ``REF_RESOLVE_FACTOR`` (10) times the
   reference's error bound -- the level is not resolved above the reference,
   and the slope would read the reference;
3. the slope's standard error exceeds ``ORDER_SE_BOUND`` (0.10) -- the errors
   do not lie on one power law over the levels, whether from curvature or
   from a wobble the levels taken have not averaged down; a two-sigma band of
   +/- 0.2 is the widest that still separates first from second order;
4. the coarsest ``dt * lambda_max`` exceeds ``DT_LAMBDA_FLAG`` (4) -- see
   below.

The bounds are set from the method, not from any case's reading. Criterion 3
does not catch every departure from one power law: a local slope that drifts
monotonically across the levels can fit with a small standard error. The
successive-level ("local") slopes are printed beside every fit so that drift
is visible.

RESOLUTION IS A PRECONDITION, not a detail. The seeded state's fastest
conduction mode has lambda_max ~ 1.5e8 s^-1, so at 8 base-steps the coarsest
step would run at dt*lambda_max ~ 18. Nothing about a scheme's TRUNCATION
error is observable there: the error at large |z| measures the substep's
stability function instead, and the two are not the same ranking.
Crank-Nicolson is the trap -- its Strang equilibrium error is exactly -z^2/16,
a clean power law, so it returns 2.00 at EVERY |z| however unresolved, while
the L-stable schemes beside it read ~1. So ``--base-steps`` DEFAULTS to a
value derived from the seed's own conduction operator (see
:func:`resolved_base_steps`), chosen so the coarsest dt satisfies ``dt *
lambda_max <= DT_LAMBDA_RESOLVED``, and every level's ``dt * lambda_max`` is
printed.

Measured at 72 cells, t_end = 1e-6 s, levels 128 / 256 / 512 / 1024 steps
(dt*lambda_max = 1.15 / 0.58 / 0.29 / 0.14), reference 16384 steps, as the
envelope order, the range over the five fields n, nn, u, Te, Ti:

    picard  splitting   backward_euler  shifted     crank_nicolson  tr_bdf2
    ------  ---------   --------------  ----------  --------------  ----------
      0       lie        1.00           1.00        1.00            0.94-1.00
      4       lie        1.00           1.00        0.98-1.00       0.90-1.00
      0       strang     0.98-1.03      0.89-1.21   1.40-1.60 (*)   1.42-1.63 (*)
      4       strang     0.97-1.04      0.82-1.20   2.00            1.99-2.00

    (*) Te PRE-ASYMPTOTIC (slope standard error 0.18 and 0.14): not quoted.

Second order needs all three of a second-order substep scheme, a non-frozen
conductivity, and Strang splitting. Each of the first-order terms caps the step
on its own: every Lie row sits at ~1.0 whatever the substep scheme is, and on
the picard-0 Strang row the frozen conductivity caps crank_nicolson and
tr_bdf2 -- their local slopes fall across the levels (about 1.6 to 1.25 on n)
toward first order, so the 1.4-1.6 there is a crossover band, not an order.
The bottom row is the shipped production package, and both second-order
substeps reach second order in it; tr_bdf2 is the shipped choice because it is
the only one that is second-order AND L-stable.

backward_euler is the negative control: theta = 1 cannot be second-order at any
dt, so if it reaches 2.0 the harness is wrong rather than good. shifted is
theta = 0.6, first-order for the same reason; on the Strang rows its local
slopes drift toward 1 across the levels (its leading first-order coefficient
is (theta - 1/2) = 0.1 of backward Euler's, so the second-order term still
contributes at these dt). Read shifted as a scale check.

EXIT CODE. 0 when ALL THREE preconditions hold -- floors inert in every run
(the reference included), the stiffest conduction mode resolved at every
level, and no envelope PRE-ASYMPTOTIC -- and 1 when any fails; the closing
lines say which. The exit code gates the PRECONDITIONS only: which order value
counts as passing depends on the scheme, the splitting and the Picard count
being asked about, and is the caller's question. Whether the numbers mean
anything at all is not.

Usage:
    python scripts/gates/verify_sim1d_order.py
    python scripts/gates/verify_sim1d_order.py --picard 4 --splitting strang
    python scripts/gates/verify_sim1d_order.py --schemes crank_nicolson tr_bdf2
    python scripts/gates/verify_sim1d_order.py --levels 6 --ref-factor 32
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

#: The largest ``dt * lambda_max`` at the coarsest level for which the stiffest
#: conduction mode counts as resolved at every level, and the value the
#: default ``--base-steps`` is derived to satisfy. Above it the error at the
#: coarse levels is a property of each scheme's stability function at large
#: ``|z|``, not of its truncation error.
DT_LAMBDA_RESOLVED = 2.0

#: The coarsest level's ``dt * lambda_max`` beyond which every envelope is
#: labelled PRE-ASYMPTOTIC and its slope is not quoted.
DT_LAMBDA_FLAG = 4.0

#: The fewest dt levels an envelope order is fitted over.
MIN_LEVELS = 4

#: The reference and its check run the Picard-4 second-order package, which
#: converges to the same semi-discrete solution as every scheme, splitting and
#: Picard count the levels measure, so one reference serves them all.
REF_SCHEME = "tr_bdf2"
REF_PICARD = 4
REF_SPLITTING = "strang"

#: Default reference margin: the reference runs at the finest level's dt
#: divided by this. At the reference's nominal second order its error is then
#: 16**-2 = 1/256 of a second-order level's finest error of comparable
#: constant, and far less against a first-order level. The margin is CHECKED,
#: not assumed: a second reference at half the reference dt bounds the
#: reference's own error (see :func:`pre_asymptotic_reasons`).
DEFAULT_REF_FACTOR = 16

#: The smallest level error must exceed the reference's error bound by this
#: factor. Below it, a tenth or more of the measured error may be the
#: reference's own, and the slope is not the scheme's. Measured at the default
#: levels, the smallest margin over every scheme and field is 81x
#: (crank_nicolson Ti, Picard 4 and Picard 2 Strang), against this 10x.
REF_RESOLVE_FACTOR = 10.0

#: The largest standard error of the fitted slope at which the slope is
#: quoted. At 0.10 the two-sigma band (+/- 0.2) separates first from second
#: order with room to spare; a larger error means the levels do not lie on
#: one power law. Measured at the default levels, the largest quoted standard
#: error is 0.07 (shifted Ti, Picard 4 and Picard 2 Strang) and the smallest
#: flagged one is 0.14 (tr_bdf2 Te, Picard 0 Strang), either side of 0.10.
ORDER_SE_BOUND = 0.10

#: The widest span (max - min) of monotone successive-level slopes at which
#: a fit is still quoted as an ORDER; wider, it is labelled DRIFTING. Added
#: after the first results: a four-level fit has two residual degrees of
#: freedom and is blind to monotone curvature, so a slope still drifting
#: across the levels can fit with a small standard error.
DRIFT_SPAN_BOUND = 0.2

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
    "initial_neutral_state": "fill",
}

CLEAN_FLAGS = {
    "Plasma": True,
    "implicit_heat_conduction": True,
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
            "resolution of the stiffest mode at the levels cannot be "
            f"established (got {dt_bound})"
        )
    return HEAT_DT_FRACTION / float(dt_bound)


def resolved_base_steps(t_end, lambda_max):
    """Return the smallest power-of-two base-steps that resolves the seed.

    "Resolves" means the COARSEST level's dt satisfies
    ``dt * lambda_max <= DT_LAMBDA_RESOLVED``. A power of two is used so the
    levels' successive halvings and the reference multiple all land on
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


def envelope_fit(dts, errors):
    """Fit the error envelope ``error = C * dt**order`` by least squares.

    The fit is linear in ``log2(error)`` against ``log2(dt)`` over every level
    at once, so a phase-dependent wobble in the error at one level moves the
    slope by its share of the whole span instead of setting it outright, as a
    triplet ratio does.

    Returns a dict with ``order`` (the fitted slope), ``se`` (its standard
    error, from the fit residuals with ``n - 2`` degrees of freedom),
    ``max_resid`` (the largest ``|log2|`` residual about the fitted line) and
    ``local`` (the successive-level slopes ``log2(e_i / e_{i+1})``, a
    diagnostic only). ``order`` is NaN when any error is not finite and
    positive, because a zero or non-finite error has no logarithm to fit.

    Raises ``ValueError`` for fewer than :data:`MIN_LEVELS` levels or for
    arrays that are not 1-D and of one length.
    """
    dts = np.asarray(dts, dtype=float)
    errors = np.asarray(errors, dtype=float)
    if dts.shape != errors.shape or dts.ndim != 1:
        raise ValueError(
            f"dts and errors must be 1-D and the same length "
            f"(got {dts.shape} and {errors.shape})"
        )
    if dts.size < MIN_LEVELS:
        raise ValueError(
            f"an envelope order needs at least {MIN_LEVELS} dt levels "
            f"(got {dts.size})"
        )
    nan = float("nan")
    if not (np.all(np.isfinite(errors)) and np.all(errors > 0.0)):
        return {"order": nan, "se": nan, "max_resid": nan, "local": []}
    x = np.log2(dts)
    y = np.log2(errors)
    xc = x - x.mean()
    order = float(np.sum(xc * (y - y.mean())) / np.sum(xc * xc))
    resid = y - (y.mean() + order * xc)
    se = float(np.sqrt(np.sum(resid * resid) / (x.size - 2) / np.sum(xc * xc)))
    local = [
        float((y[i] - y[i + 1]) / (x[i] - x[i + 1])) for i in range(x.size - 1)
    ]
    return {
        "order": order,
        "se": se,
        "max_resid": float(np.max(np.abs(resid))),
        "local": local,
    }


def pre_asymptotic_reasons(fit, errors, ref_error_bound, z_coarse):
    """Return why an envelope is PRE-ASYMPTOTIC; an empty list quotes it.

    The envelope is PRE-ASYMPTOTIC when any of these holds:

    1. The fit has no slope (an error at some level is zero or non-finite).
    2. The smallest level error is not resolved above the reference: it is
       below :data:`REF_RESOLVE_FACTOR` times the reference's own error
       bound. The measured error is then partly the reference's, and the
       slope reads the reference, not the scheme.
    3. The slope's standard error exceeds :data:`ORDER_SE_BOUND`: the errors
       do not lie on one power law over the levels, whether from curvature
       (the coarse levels still outside the asymptotic range) or from a
       wobble that the levels taken have not averaged down.
    4. The coarsest ``dt * lambda_max`` exceeds :data:`DT_LAMBDA_FLAG`: the
       stiffest conduction mode is unresolved and the error measures each
       substep's stability function at large ``|z|``.
    """
    reasons = []
    if not np.isfinite(fit["order"]):
        reasons.append("error not positive and finite at every level")
    elif min(errors) < REF_RESOLVE_FACTOR * ref_error_bound:
        reasons.append(
            f"smallest level error {min(errors):.2e} not resolved above "
            f"{REF_RESOLVE_FACTOR:.0f}x the reference error bound "
            f"{ref_error_bound:.2e}"
        )
    if np.isfinite(fit["se"]) and fit["se"] > ORDER_SE_BOUND:
        reasons.append(
            f"slope standard error {fit['se']:.2f} > {ORDER_SE_BOUND:.2f}"
        )
    if z_coarse > DT_LAMBDA_FLAG:
        reasons.append(
            f"coarsest dt*lambda_max {z_coarse:.2f} > {DT_LAMBDA_FLAG:.0f}"
        )
    return reasons


def drift_label(fit):
    """Return the DRIFTING label for a fit whose local slope drifts, else None.

    A fit drifts when its successive-level slopes are monotone across the
    levels and span more than :data:`DRIFT_SPAN_BOUND`: the error is still
    bending between power laws (a crossover band), and the fitted slope is a
    band, not an order. Only MONOTONE drift is labelled: slopes that alternate
    about a steady value are the phase-dependent wobble the envelope averages
    out, and the standard-error bound already judges those. The label is
    printed in place of ORDER and does not enter the exit code.
    """
    local = fit["local"]
    if len(local) < 2 or not np.all(np.isfinite(local)):
        return None
    steps = np.diff(local)
    monotone = bool(np.all(steps >= 0.0) or np.all(steps <= 0.0))
    if monotone and max(local) - min(local) > DRIFT_SPAN_BOUND:
        return (
            f"DRIFTING: local slopes {local[0]:.2f} -> {local[-1]:.2f} "
            "(band, not an order)"
        )
    return None


def triplet_screens(solutions):
    """Return the Richardson triplet ratio of each consecutive level triple.

    ``log2(max|u_N - u_2N| / max|u_2N - u_4N|)`` over the levels in order,
    NaN where either difference is zero. A SCREEN only, never a quoted order:
    on a state where the error is not smooth in dt (a limited-flux gradient
    changing sign inside the window) a triplet reads erratically at any dt
    while the envelope against a converged reference stays clean.
    """
    out = []
    for i in range(len(solutions) - 2):
        num = float(np.max(np.abs(solutions[i] - solutions[i + 1])))
        den = float(np.max(np.abs(solutions[i + 1] - solutions[i + 2])))
        out.append(
            float(np.log2(num / den)) if den > 0.0 and num > 0.0
            else float("nan")
        )
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schemes", nargs="+", default=list(IMPLICIT_HEAT_SCHEMES))
    parser.add_argument("--t-end", type=float, default=1.0e-6)
    parser.add_argument(
        "--base-steps",
        type=int,
        default=None,
        help=(
            "coarsest step count of the refinement levels; default is DERIVED "
            "from the seed's own conduction rate so the coarsest "
            "dt*lambda_max is at or below the resolved bound"
        ),
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=MIN_LEVELS,
        help=(
            f"number of dt levels, each halving the last (at least "
            f"{MIN_LEVELS}; default {MIN_LEVELS})"
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
        default=DEFAULT_REF_FACTOR,
        help=(
            "the reference runs at the finest level's steps times this "
            "factor, and its check at twice that (default "
            f"{DEFAULT_REF_FACTOR})"
        ),
    )
    args = parser.parse_args(argv)
    if args.levels < MIN_LEVELS:
        parser.error(f"--levels must be at least {MIN_LEVELS} (got {args.levels})")
    if args.ref_factor < 2:
        parser.error(f"--ref-factor must be at least 2 (got {args.ref_factor})")

    lambda_max = seed_conduction_lambda_max(
        args.schemes[0], args.amplitude, args.picard, args.splitting
    )
    if args.base_steps is None:
        N = resolved_base_steps(args.t_end, lambda_max)
        base_steps_origin = "derived"
    else:
        N = args.base_steps
        base_steps_origin = "given"
    counts = tuple(N * 2**i for i in range(args.levels))
    dts = [args.t_end / n for n in counts]
    ref_steps = counts[-1] * args.ref_factor
    z_coarse = dts[0] * lambda_max
    resolved = z_coarse <= DT_LAMBDA_RESOLVED

    print("=" * 76)
    print("sim1d SPLIT-STEP TEMPORAL ORDER (error envelope vs a converged reference)")
    print("=" * 76)
    print(f"t_end={args.t_end:.2e} s   levels={counts}")
    print(f"heat_picard_iterations={args.picard}  operator_splitting={args.splitting}")
    print(f"dt from {dts[0]:.3e} s down to {dts[-1]:.3e} s")
    print("regime: fixed dt, floors inert, single phase, autonomous RHS, no cathode")
    print(
        f"seed conduction lambda_max = {lambda_max:.4e} s^-1 "
        f"(from the solver's own explicit heat bound)"
    )
    print(
        f"base-steps={N} ({base_steps_origin}); dt*lambda_max = "
        + ", ".join(f"{dt*lambda_max:.2f}" for dt in dts)
        + f"   [resolved bound {DT_LAMBDA_RESOLVED:.0f}, "
        f"flag above {DT_LAMBDA_FLAG:.0f}]"
    )
    print(
        f"reference: {REF_SCHEME}, heat_picard_iterations={REF_PICARD}, "
        f"{REF_SPLITTING}, {ref_steps} steps (finest level x {args.ref_factor}); "
        f"check: the same at {2 * ref_steps} steps"
    )
    print(
        f"PRE-ASYMPTOTIC when: smallest level error < {REF_RESOLVE_FACTOR:.0f}x "
        f"the reference error bound, OR slope standard error > "
        f"{ORDER_SE_BOUND:.2f}, OR coarsest dt*lambda_max > {DT_LAMBDA_FLAG:.0f}"
    )
    print(
        f"DRIFTING when: local slopes monotone and spanning > "
        f"{DRIFT_SPAN_BOUND:.1f} (not quoted; exit code unaffected)"
    )

    ref, ref_clips = run_fixed_dt(
        REF_SCHEME, ref_steps, args.t_end, args.amplitude, REF_PICARD,
        REF_SPLITTING,
    )
    ref_check, c = run_fixed_dt(
        REF_SCHEME, 2 * ref_steps, args.t_end, args.amplitude, REF_PICARD,
        REF_SPLITTING,
    )
    ref_clips += c
    # For a reference of observed order q >= 1, its own error is
    # d / (1 - 2**-q) <= 2d, where d is its distance to the check at half dt.
    ref_bound = {k: 2.0 * rel_diff(ref[k], ref_check[k]) for k in FIELDS}
    print("reference error bound (2 x |ref - check|, relative L-inf):")
    print("  " + "  ".join(f"{k}={ref_bound[k]:.2e}" for k in FIELDS))
    any_dirty = ref_clips > 0
    if ref_clips:
        print(f"  <-- INVALID reference: {ref_clips} floor activations")

    any_pre = False
    n_drift = 0
    for scheme in args.schemes:
        runs, clips = {}, 0
        for n in counts:
            runs[n], c = run_fixed_dt(
                scheme, n, args.t_end, args.amplitude, args.picard, args.splitting
            )
            clips += c

        flag = ""
        if clips:
            flag = f"   <-- INVALID: {clips} floor activations"
            any_dirty = True
        print(f"\n--- {scheme} ---{flag}")
        for k in FIELDS:
            errs = [rel_diff(runs[n][k], ref[k]) for n in counts]
            fit = envelope_fit(dts, errs)
            screen = triplet_screens([runs[n][k] for n in counts])
            reasons = pre_asymptotic_reasons(fit, errs, ref_bound[k], z_coarse)
            drift = drift_label(fit)
            if reasons:
                any_pre = True
                quoted = (
                    f"slope {fit['order']:.2f} +/- {fit['se']:.2f}  "
                    "NOT QUOTED, PRE-ASYMPTOTIC: " + "; ".join(reasons)
                )
            elif drift:
                n_drift += 1
                quoted = (
                    f"slope {fit['order']:.2f} +/- {fit['se']:.2f}  "
                    f"NOT QUOTED, {drift}"
                )
            else:
                quoted = (
                    f"ORDER {fit['order']:.2f} +/- {fit['se']:.2f}"
                    f"  (max |log2 resid| {fit['max_resid']:.2f})"
                )
            margin = (
                f"{min(errs) / ref_bound[k]:.1e}" if ref_bound[k] > 0.0
                else "inf"
            )
            print(f"  {k:3} {quoted}")
            print(
                f"      err={[f'{x:.2e}' for x in errs]}"
                f"  smallest/ref-bound={margin}"
            )
            print(
                f"      local slopes={[f'{x:.2f}' for x in fit['local']]}"
                f"  triplet SCREEN (not an order)="
                f"{[f'{x:.2f}' for x in screen]}"
            )

    print("\n" + "-" * 76)
    # All three preconditions have to hold. Floors are a non-smooth projection
    # and break order wherever they bind; an unresolved stiff mode replaces the
    # truncation error with the substep's large-|z| stability behaviour; and a
    # PRE-ASYMPTOTIC envelope has no order to quote.
    print(f"precondition 1, floors inert                  : "
          f"{'no' if any_dirty else 'yes'}")
    print(f"precondition 2, coarsest dt*lambda_max <= "
          f"{DT_LAMBDA_RESOLVED:.0f} : "
          f"{'yes' if resolved else 'no'} ({z_coarse:.2f})")
    print(f"precondition 3, no envelope PRE-ASYMPTOTIC    : "
          f"{'no' if any_pre else 'yes'}")
    if any_dirty:
        print("At least one run activated a floor: those orders are meaningless.")
        print("Raise --amplitude headroom or shorten --t-end.")
    if not resolved:
        print(
            "The stiffest conduction mode is UNRESOLVED at the coarsest dt: the "
            "orders above are\nnot meaningful. Raise --base-steps (the default "
            "derives one that resolves it)."
        )
    if any_pre:
        print(
            "At least one envelope is PRE-ASYMPTOTIC: its slope is printed "
            "but not quoted.\nAdd --levels, or raise --base-steps or "
            "--ref-factor, as its stated reason indicates."
        )
    if n_drift:
        print(
            f"{n_drift} field fit(s) DRIFTING: a band, not an order, and not "
            "quoted. The label does not\nenter the exit code."
        )
    ok = not any_dirty and resolved and not any_pre
    if ok:
        print(
            "Floors stayed inert in every run, the stiffest conduction mode is "
            "resolved at every\nlevel, and no envelope is PRE-ASYMPTOTIC: the "
            "ORDERs above are measurements."
        )
    print("-" * 76)
    # THE EXIT CODE CARRIES THE PRECONDITIONS, not the order values: what
    # number counts as passing is the caller's question and depends on the
    # scheme, the splitting and the Picard count being asked about. What is
    # not the caller's question is whether the numbers mean anything at all.
    if not ok:
        print(
            "EXIT 1: at least one precondition failed, so not every order "
            "above is a measurement."
        )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
