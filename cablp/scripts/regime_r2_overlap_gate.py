"""Regime-R2 two-sided overlap gate: tracer vs full fluid where BOTH are valid.

PRE-REGISTERED BEFORE THE COMPARISON WAS IMPLEMENTED. The band and the
tolerance are not chosen here -- they are config keys, so a sweep moves them
without touching this file:

    band      = input_dict["tracer_overlap_band_ne"]  (ships [1.0e10, 1.0e11] cm^-3)
    tolerance = input_dict["tracer_overlap_rtol"]     (ships 0.05)

REGISTERED ACCEPTANCE
---------------------
Two runs from the SAME initial condition and the same configuration, differing
only in the ``regime_tracer`` flag, are advanced over the same window on the
same save lattice. On every saved frame, in every cell whose FLUID density lies
inside the band, the two densities must agree to within ``tracer_overlap_rtol``
relative. PASS iff that holds on every such (frame, cell); FAIL otherwise.

Why the band is the only place this is meaningful: below its low edge the fluid
is floor-poisoned (``ne_floor`` clipping sets the density, not the physics), and
above its high edge the plasma stops being passive so the tracer is out of its
own domain. A comparison outside the band measures the band, not the code.

The comparison is on the fluid's band membership, not the tracer's, so a
disagreement cannot hide by moving a cell out of the window.

A third outcome, BLOCKED, exists and is not a pass: the tracer can refuse to
produce a number at all (``TracerBalanceError`` -- the quasi-static electron
energy balance having no root, or being multi-valued, at the operating point;
see NUMERICS.md, "MEASURED: the local balance has no root at the production
stance"). That is reported as BLOCKED with the refusal verbatim, because a gate
that cannot run has not passed.

MATCHED-CLOSURE READING (registered before the ql_relaxation run)
-----------------------------------------------------------------
``--anomalous-model`` moves BOTH arms together, and moving it changes what the
gate measures. On the shipped ``quasilinear`` the tracer refuses the anomalous
booking on its own cells and the fluid does not, so a disagreement is dominated
by that refusal -- the gate is measuring the closure gap. Under
``ql_relaxation`` a passive cell books the channel exactly as an active one
does, so the closure is MATCHED across the interface and what remains is
tracer-vs-fluid NUMERICS: the tracer's quasi-static ``Te`` and its neglect of
parallel transport against the fluid's resolved equations. Both verdicts are
evidence and neither is tuned for; a FAIL under matched closures is a statement
about the two descriptions, not a defect to be fitted away.

Usage (from <checkout>/cablp, with PYTHONPATH set to that same cablp):
    python scripts/regime_r2_overlap_gate.py --nx 20 --t-end 3e-5
    python scripts/regime_r2_overlap_gate.py --nx 20 --t-end 3e-5 \\
        --anomalous-model ql_relaxation --ql-relaxation-coeff 30
"""

import argparse
import sys
import warnings

import numpy as np

from compare_sim1d_es1 import FLAG_OVERRIDES, PARAM_OVERRIDES

from cablp.solvers._sim1d import LAPDSim1D, default_config
from cablp.solvers._sim1d.physics.tracer import TracerBalanceError


def build_config(nx, tracer_on, extra=None):
    """Return the production-stance config, differing ONLY in the tracer flag."""
    params, flags = default_config()
    params.update(PARAM_OVERRIDES)
    flags.update(FLAG_OVERRIDES)
    params["neutral_exchange_model"] = "knudsen"
    params.update(
        {
            "nx": nx,
            "cathode_solver_model": "current_driven",
            "beam_deposition_model": "csda",
            "beam_anomalous_model": "quasilinear",
            "cathode_emission_profile": "gaussian",
            "cathode_warming_model": "power_balance",
            "cathode_heat_capacity_J_per_K": 120.0,
            "cathode_emissivity": 0.7,
        }
    )
    flags["neutral_equilibration"] = False
    flags["cathode_circuit_voltage_bound"] = True
    flags["regime_tracer"] = bool(tracer_on)
    if extra:
        for key, value in extra.items():
            (flags if key in flags else params)[key] = value
    return params, flags


def run_arm(nx, tracer_on, t_end, dt_save, max_steps, extra=None):
    """Return ``(times, n)`` for one arm, or raise ``TracerBalanceError``."""
    params, flags = build_config(nx, tracer_on, extra)
    params["dt_save"] = dt_save
    sim = LAPDSim1D(params, flags)
    result = sim.run(t_end=t_end, max_steps=max_steps)
    return (
        np.asarray(result.time, dtype=float),
        np.asarray(result.n, dtype=float),
        sim,
    )


def compare(fluid, tracer, band, rtol):
    """Return ``(passed, worst, count, where)`` over the registered band."""
    t_f, n_f = fluid
    t_t, n_t = tracer
    frames = min(t_f.size, t_t.size)
    if frames == 0:
        return False, float("nan"), 0, None
    if not np.allclose(t_f[:frames], t_t[:frames], rtol=0.0, atol=1e-15):
        raise ValueError(
            "the two arms did not land on the same save lattice; the gate "
            "compares frame by frame and cannot interpolate a moving target"
        )
    n_f = n_f[:frames]
    n_t = n_t[:frames]
    low, high = band
    inside = (n_f >= low) & (n_f <= high)
    if not np.any(inside):
        return False, float("nan"), 0, None
    rel = np.abs(n_t - n_f) / np.maximum(np.abs(n_f), low)
    rel_in = np.where(inside, rel, 0.0)
    worst = float(np.max(rel_in))
    idx = np.unravel_index(int(np.argmax(rel_in)), rel_in.shape)
    return worst <= rtol, worst, int(np.count_nonzero(inside)), (
        float(t_f[idx[0]]),
        int(idx[1]),
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=20)
    parser.add_argument("--t-end", type=float, default=3.0e-5)
    parser.add_argument("--dt-save", type=float, default=1.0e-6)
    parser.add_argument("--max-steps", type=int, default=20000)
    # The anomalous closure BOTH arms run. Changing it changes what the gate
    # tests, and that is the point of exposing it: with the two arms on the
    # fiat closure the gate measures the tracer's refusal of it, and with both
    # on `ql_relaxation` -- which a passive cell books like an active one --
    # the closure is matched and what is left is tracer-vs-fluid NUMERICS.
    # Either reading is evidence; neither is tuned for here.
    parser.add_argument(
        "--anomalous-model",
        default="quasilinear",
        choices=("none", "quasilinear", "ql_relaxation"),
    )
    parser.add_argument("--ql-relaxation-coeff", type=float, default=None)
    args = parser.parse_args(argv)

    extra = {"beam_anomalous_model": args.anomalous_model}
    if args.ql_relaxation_coeff is not None:
        extra["ql_relaxation_coeff"] = args.ql_relaxation_coeff

    warnings.simplefilter("ignore")
    params, _flags = build_config(args.nx, False, extra)
    band = [float(v) for v in params["tracer_overlap_band_ne"]]
    rtol = float(params["tracer_overlap_rtol"])
    print(
        f"regime_r2 overlap gate: nx={args.nx} t_end={args.t_end:g} s "
        f"dt_save={args.dt_save:g} s"
    )
    print(
        f"  registered band = [{band[0]:g}, {band[1]:g}] cm^-3, "
        f"registered rtol = {rtol:g} (both from input_dict)"
    )
    print(
        f"  beam_anomalous_model = {args.anomalous_model!r} on BOTH arms"
        + (
            f", ql_relaxation_coeff = {params['ql_relaxation_coeff']:g}"
            if args.anomalous_model == "ql_relaxation"
            else ""
        )
    )

    t_f, n_f, _sim_f = run_arm(
        args.nx, False, args.t_end, args.dt_save, args.max_steps, extra
    )
    print(f"  fluid arm: {t_f.size} frames, t_end={t_f[-1]:.6g} s, "
          f"n max {float(np.max(n_f)):.4g} cm^-3")

    try:
        t_t, n_t, sim_t = run_arm(
            args.nx, True, args.t_end, args.dt_save, args.max_steps, extra
        )
    except TracerBalanceError as error:
        print("  tracer arm: REFUSED to produce a number")
        print(f"    {error}")
        print("regime_r2 overlap gate: BLOCKED (tracer arm could not run)")
        return 2
    print(f"  tracer arm: {t_t.size} frames, t_end={t_t[-1]:.6g} s, "
          f"n max {float(np.max(n_t)):.4g} cm^-3")
    census = sim_t._tracer_census_line()
    if census:
        print(f"  {census}")

    passed, worst, count, where = compare((t_f, n_f), (t_t, n_t), band, rtol)
    if count == 0:
        print(
            "  no (frame, cell) sample landed inside the registered band -- "
            "the window never reached it, so there is nothing to compare"
        )
        print("regime_r2 overlap gate: BLOCKED (empty overlap sample)")
        return 2
    print(
        f"  samples inside band: {count}; worst relative disagreement "
        f"{worst:.4g} at t={where[0]:.6g} s cell {where[1]}"
    )
    print(
        "regime_r2 overlap gate: "
        + ("PASS" if passed else "FAIL")
        + f" (worst {worst:.4g} vs rtol {rtol:g})"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
