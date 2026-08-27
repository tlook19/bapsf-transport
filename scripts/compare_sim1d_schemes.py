"""Controlled fixed-dt scheme comparison in a production-physics regime.

The question this harness answers: does scheme choice change the
*physics* at all?
Every comparison so far has been uncontrolled — adaptive stepping gives each
scheme a different step sequence, and the one spread observed (~0.3% in final
thermal energy) was non-monotone in theta, the signature of trajectory
divergence rather than a scheme effect.

This harness is the complement of ``verify_sim1d_order.py``. That one measures
order in a deliberately clean regime (no cathode, no forcing, floors inert);
this one keeps the production physics ON — cathode drive, gas puff, pump,
floors live, adas rates — and controls the *comparison* instead:

* **Identical initial state** for every scheme (the shared cold prefilled
  column; ignition happens inside the window under the discharge physics,
  which is exactly the front the schemes must agree on).
* **A single scheduled discharge phase**, so no threshold trigger can shift
  by one step between schemes and masquerade as a scheme effect.
* **One shared fixed dt** for every run, chosen from a production adaptive
  pre-pass (its minimum accepted dt times a safety factor by default), so
  every scheme takes exactly the same step sequence.

Any remaining difference between runs is then attributable to the scheme
package, at a dt the production adaptive controller actually uses.

Scheme packages compared (each is a package because the three accuracy knobs
only act together — see NUMERICS.md):

    be_lie     backward_euler + lie    + picard 0   (historical default)
    be_strang  backward_euler + strang + picard 2   (isolates the substep scheme)
    cn         crank_nicolson + strang + picard 2
    tr_bdf2    tr_bdf2        + strang + picard 2   (production; the reference)

Floors are part of production physics here, so activations do not invalidate
the comparison; they are counted and reported per run because a large
imbalance between schemes is itself a finding.

Usage:
    python scripts/compare_sim1d_schemes.py
    python scripts/compare_sim1d_schemes.py --t-end 3e-3 --dt 2e-8
    python scripts/compare_sim1d_schemes.py --packages be_lie tr_bdf2
"""

import argparse
import sys
from time import perf_counter

import numpy as np

import cablp.solvers._sim1d.core.state as state_mod
from cablp.solvers._sim1d import LAPDSim1D, default_config
from cablp.vars._cons import ev_to_erg

FIELDS = ("n", "nn", "u", "Te", "Ti")

SCHEME_PACKAGES = {
    "be_lie": {
        "implicit_heat_scheme": "backward_euler",
        "operator_splitting": "lie",
        "heat_picard_iterations": 0,
    },
    "be_strang": {
        "implicit_heat_scheme": "backward_euler",
        "operator_splitting": "strang",
        "heat_picard_iterations": 2,
    },
    "cn": {
        "implicit_heat_scheme": "crank_nicolson",
        "operator_splitting": "strang",
        "heat_picard_iterations": 2,
    },
    "tr_bdf2": {
        "implicit_heat_scheme": "tr_bdf2",
        "operator_splitting": "strang",
        "heat_picard_iterations": 2,
    },
}
REFERENCE = "tr_bdf2"

# Production physics in a controlled window: single scheduled discharge phase
# from t = 0, prefilled gas standing in for the skipped prebreakdown fill.
WINDOW_PARAMS = {
    "nn0": 1.0e13,
    "phase_transition_mode": "scheduled",
    "tau_neutral_prebreakdown": 0.0,
    "tau_prebreakdown": 0.0,
    "tau_breakdown": 0.0,
    "tau_discharge": 1.0,
    "tau_afterglow": 0.0,
    "heat_picard_tol": 1e-10,
}

# The fixed-dt runs additionally disarm every adaptive intervention so the
# requested dt is the dt taken (same block as verify_sim1d_order.py).
FIXED_DT_PARAMS = {
    "adaptive_retries_enabled": False,
    "dt_growth_enabled": False,
    "dt_min": 1e-16,
    "dt_max": 1.0,
    "max_density_step_fraction": 0.0,
    "max_neutral_step_fraction": 0.0,
    "max_energy_step_fraction": 0.0,
}

WINDOW_FLAGS = {
    "Plasma": True,
    "implicit_heat_conduction": True,
    "cathode_coupling": True,
    "debug_checks": False,
}

FLOOR_RTOL = 1e-9


class FloorWatch:
    """Count floor activations (same probe as verify_sim1d_order.py).

    Here activations do not invalidate the run — floors are production
    physics — but a scheme-to-scheme imbalance is worth reporting.
    """

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


def build_sim(package, fixed_dt, dt_save):
    params, flags = default_config()
    params.update(WINDOW_PARAMS)
    if fixed_dt:
        params.update(FIXED_DT_PARAMS)
    params.update(SCHEME_PACKAGES[package])
    params["dt_save"] = dt_save
    flags.update(WINDOW_FLAGS)
    return LAPDSim1D(params, flags)


def adaptive_prepass(t_end):
    """Run the production package adaptively; return its accepted-dt history."""
    sim = build_sim(REFERENCE, fixed_dt=False, dt_save=0.0)
    result = sim.run(t_end=t_end)
    time = np.asarray(result.time, dtype=float)
    dts = np.diff(time)
    dts = dts[dts > 0.0]
    if dts.size == 0:
        raise RuntimeError("adaptive pre-pass produced no accepted steps")
    return result, dts


def run_package(package, t_end, dt, dt_save):
    sim = build_sim(package, fixed_dt=True, dt_save=dt_save)
    watch = FloorWatch()
    wall = perf_counter()
    with watch:
        result = sim.run(t_end=t_end, dt=dt, operator_split=True)
    wall = perf_counter() - wall
    fields = {
        "time": np.asarray(result.time, dtype=float),
        "n": np.asarray(result.n, dtype=float),
        "nn": np.asarray(result.nn, dtype=float),
        "u": np.asarray(result.u, dtype=float),
        "Te": np.asarray(result.Te, dtype=float),
        "Ti": np.asarray(result.Ti, dtype=float),
    }
    for key in FIELDS:
        if not np.all(np.isfinite(fields[key])):
            raise RuntimeError(f"{package}: non-finite {key} in saved trajectory")
    geom = sim._geometry
    thermal = float(
        np.sum(
            (np.asarray(result.Ee[-1]) + np.asarray(result.Ei[-1]))
            * np.asarray(geom.plasma_volume_cm3)
        )
    )
    return {
        "fields": fields,
        "steps": int(result.steps),
        "clips": watch.clips,
        "thermal_erg": thermal,
        "z_cm": np.asarray(geom.z_cm, dtype=float),
        "wall_s": wall,
    }


def rel_linf(a, b):
    """Relative L-inf difference, scaled by the magnitude of the reference."""
    scale = np.max(np.abs(b))
    if scale == 0.0:
        return float(np.max(np.abs(a - b)))
    return float(np.max(np.abs(a - b)) / scale)


def front_z_cm(z, Te):
    """Rightmost half-max crossing of the Te profile [cm], interpolated."""
    Te = np.asarray(Te, dtype=float)
    half = 0.5 * np.max(Te)
    above = Te >= half
    if not np.any(above) or np.all(above):
        return float("nan")
    idx = np.where(above)[0][-1]
    if idx + 1 >= Te.size:
        return float(z[idx])
    t0, t1 = Te[idx], Te[idx + 1]
    if t0 == t1:
        return float(z[idx])
    frac = (t0 - half) / (t0 - t1)
    return float(z[idx] + frac * (z[idx + 1] - z[idx]))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--t-end", type=float, default=3.0e-3)
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="shared fixed dt [s]; default = safety * min accepted dt of an "
        "adaptive production pre-pass over the same window",
    )
    parser.add_argument(
        "--dt-safety",
        type=float,
        default=0.8,
        help="safety factor on the pre-pass minimum dt",
    )
    parser.add_argument(
        "--dt-save", type=float, default=5.0e-4, help="checkpoint interval [s]"
    )
    parser.add_argument(
        "--packages",
        nargs="+",
        default=list(SCHEME_PACKAGES),
        choices=list(SCHEME_PACKAGES),
    )
    args = parser.parse_args(argv)

    packages = list(dict.fromkeys(args.packages))
    if REFERENCE not in packages:
        packages.append(REFERENCE)

    print("=" * 76)
    print("sim1d SCHEME NULL TEST: fixed-dt comparison, production physics")
    print("=" * 76)

    dt = args.dt
    if dt is None:
        print("adaptive pre-pass (production package) ...", flush=True)
        _, dts = adaptive_prepass(args.t_end)
        dt = args.dt_safety * float(np.min(dts))
        print(
            f"  accepted dt: min={np.min(dts):.3e}  median={np.median(dts):.3e}"
            f"  max={np.max(dts):.3e} s  ({dts.size} steps)"
        )
    nsteps_est = int(np.ceil(args.t_end / dt))
    print(f"window t_end={args.t_end:.2e} s   fixed dt={dt:.3e} s   ~{nsteps_est} steps/run")
    print("regime: identical IC, single scheduled discharge phase, cathode on,")
    print("        puff/pump on, floors live (counted), adas rates")

    runs = {}
    for package in packages:
        print(f"running {package} ...", flush=True)
        runs[package] = run_package(package, args.t_end, dt, args.dt_save)
        r = runs[package]
        print(
            f"  steps={r['steps']}  floor_clips={r['clips']}"
            f"  thermal={r['thermal_erg']:.6e} erg  wall={r['wall_s']:.1f} s"
        )

    ref = runs[REFERENCE]
    ref_times = ref["fields"]["time"]
    z = ref["z_cm"]

    for package in packages:
        times = runs[package]["fields"]["time"]
        if times.shape != ref_times.shape or not np.allclose(
            times, ref_times, rtol=0.0, atol=1e-12
        ):
            raise RuntimeError(
                f"{package}: saved-time grid differs from the reference — "
                "the step sequences were not identical, comparison is invalid"
            )

    print("\n--- vs reference package "
          f"({REFERENCE}: tr_bdf2 + strang + picard 2) ---")
    header = "  ".join(f"{k:>9}" for k in FIELDS)
    for package in packages:
        if package == REFERENCE:
            continue
        fields = runs[package]["fields"]
        print(f"\n{package}:")
        print(f"  checkpoint [ms]   {header}   combined")
        worst = 0.0
        for i, t in enumerate(ref_times):
            if t <= 0.0:
                continue
            diffs = [
                rel_linf(fields[k][i], ref["fields"][k][i]) for k in FIELDS
            ]
            combined = max(diffs)
            worst = max(worst, combined)
            row = "  ".join(f"{d:9.2e}" for d in diffs)
            print(f"  {1e3 * t:11.2f}     {row}   {combined:9.2e}")
        dth = abs(runs[package]["thermal_erg"] - ref["thermal_erg"]) / abs(
            ref["thermal_erg"]
        )
        print(f"  worst combined rel L-inf : {worst:.2e}")
        print(f"  final thermal energy     : {dth:.2e} relative")

    print("\n--- Te front position (rightmost half-max crossing) [cm] ---")
    label = "checkpoint [ms]"
    print(f"  {label:>15}  " + "  ".join(f"{p:>10}" for p in packages))
    for i, t in enumerate(ref_times):
        if t <= 0.0:
            continue
        row = "  ".join(
            f"{front_z_cm(z, runs[p]['fields']['Te'][i]):10.2f}" for p in packages
        )
        print(f"  {1e3 * t:15.2f}  {row}")

    print("\ndone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
