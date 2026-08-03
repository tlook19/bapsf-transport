"""R5 ES1 ignition-threshold isolation probe (2026-07-25).

Diagnosed blocker (R5_STANCE_FLIP_HANDOFF.md, "ES1 IGNITION-THRESHOLD SHIFT"):
the repaired-stance ES config never ignites -- it sits in ``pre_breakdown`` at
``source_I_tot`` ~ 2.76 A (needs 150 A to leave prebreakdown) because the
prebreakdown SEED is ~2.5x below the historical config that ignites. Measured at
t = 2.2 ms, nx = 120: historical/golden collector density n_col ~ 9.8e8 (ignites)
vs repaired-ES n_col ~ 4.0e8 (below threshold). Root cause is under-confinement of
the seed; the fix is a FUELING / seed-retention lever, not raising T_s (I_loop is
plasma-limited, invariant to T_s).

PRE-REGISTERED ISOLATION (this script). Metric: collector-cell plasma density
n_col at t = 2.2 ms (the seed), with source_I_tot and phase as ignition proxies.
From the repaired-ES baseline, single-toggle each named suspect and add an
all-three positive control that reverts toward the historical stance:

  baseline      repaired ES stance as-is                (expect ~4.0e8, ~2.76 A)
  char_off      characteristic_boundary = False         (R3.1 Bohm outflow off)
  birth_legacy  ionization_birth_energy_model = legacy   (hot legacy birth)
  sgp3400       S_gp = 3400 (config default; ES override is 3000)
  all3          char_off + birth_legacy + sgp3400        (positive control)

DECISION RULE (fixed before running): the toggle that lifts n_col closest to the
historical ~9.8e8 identifies the dominant seed-depletion lever and hence the fix
family:
  * char_off dominant  -> R3.1 Bohm outflow over-drains the seed. It is a
    correctness repair we KEEP, so the fix is fueling / seed retention, not a
    revert.
  * birth_legacy dominant -> conservative cold-electron birth suppresses the
    ionization multiplication that builds the seed.
  * sgp3400 dominant   -> fueling-limited; raise the fill.
  * none alone (only all3 recovers) -> combined under-confinement; a
    fueling / seed-retention lever is needed regardless.

This is a diagnostic, not a solver change: no _sim1d/ edit, no gate, no baseline
touch. Run from <repo>/cablp with the fenicsx-env interpreter.
"""

import argparse
import sys
import time as _walltime

import numpy as np

from cablp.solvers._sim1d import LAPDSim1D, default_config
from cablp.solvers._sim1d.solver import ProgressPrinter1D
from cablp.solvers._sim1d.results.io import save_result_hdf5

# Reuse the REAL ES config so the probe cannot drift from the benchmark driver.
from compare_sim1d_es1 import PARAM_OVERRIDES, FLAG_OVERRIDES

VARIANTS = {
    "baseline": (dict(), dict()),
    "char_off": (dict(), {"characteristic_boundary": False}),
    "birth_legacy": ({"ionization_birth_energy_model": "legacy"}, dict()),
    "sgp3400": ({"S_gp": 3400}, dict()),
    "all3": (
        {"ionization_birth_energy_model": "legacy", "S_gp": 3400},
        {"characteristic_boundary": False},
    ),
    # Control: skip the 100-cycle neutral pre-equilibration, so the run starts
    # from the bare nn0 seed and builds its fill only from the in-run puff.
    # Tests whether the diagnosed ~4.0e8 stuck seed was an artifact of a probe
    # that did not equilibrate the neutral inventory first.
    # NB this control used to ALSO carry the in-run tau_neutral_prebreakdown
    # = 2 ms neutral-only accumulation. That window is gone (the default is now
    # 0.0 -- the machine has no pre-drive window, 2026-08-03), so there is no
    # pre-phase accumulation left to fall back on here.
    "no_equil": (dict(), {"neutral_equilibration": False}),
    # Control: the pre-WIP ES geometry (no end-expansion). The end-expansion
    # geometry (Rcs=40/Lcs=25 + expand-to-1m + collector behind 9 "end" cells)
    # was added to PARAM_OVERRIDES AFTER the ignition diagnosis; this isolates
    # whether that geometry edit itself moved the seed.
    "no_endexp": (dict(), {"end_expansion_geometry": False}),
}


def build_config(variant, nx):
    params, flags = default_config()
    params.update(PARAM_OVERRIDES)
    flags.update(FLAG_OVERRIDES)
    # run_model() default: knudsen neutral exchange.
    params["neutral_exchange_model"] = "knudsen"
    params["nx"] = nx
    p_extra, f_extra = VARIANTS[variant]
    params.update(p_extra)
    flags.update(f_extra)
    # The end-expansion params are presence-gated on the flag: strip them when
    # the flag is off (else construction raises a loud ValueError, by design).
    if not flags.get("end_expansion_geometry", False):
        for k in ("end_expansion_cells", "end_expansion_machine_radius_cm",
                  "end_expansion_plasma_radius_cm"):
            params.pop(k, None)
    return params, flags


def probe(variant, nx, t_probe):
    params, flags = build_config(variant, nx)
    sim = LAPDSim1D(params, flags)
    t0 = _walltime.time()
    # start_simulation runs the neutral equilibration, applies it, then runs the
    # plasma -- truncated at t_probe. Exactly the production path, cut short.
    sim.start_simulation(t_end=t_probe, dt=None, operator_split=None, max_steps=None)
    res = sim.get_results()
    wall = _walltime.time() - t0

    roles = np.asarray(sim.geometry.cell_role)
    col = int(np.flatnonzero(roles == "collector")[-1])
    puff = int(np.flatnonzero(roles == "puff")[-1])
    col_mask = roles == "column"

    I = np.asarray(res.cathode_diagnostics["source_I_tot"], float)
    n = np.asarray(res.n, float)
    nn = np.asarray(res.nn, float)
    Te = np.asarray(res.Te, float)
    phases = np.asarray(res.phase, dtype=str)
    t = np.asarray(res.time, float)

    return {
        "variant": variant,
        "wall_s": wall,
        "t_end_ms": t[-1] * 1e3,
        "phase": phases[-1],
        "ever_main": bool(np.any(phases == "main_discharge")),
        "I_A": I[-1],
        "I_max_A": float(np.max(I)),
        "n_col": n[-1, col],
        "n_puff": n[-1, puff],
        "n_col_max_cell": float(np.max(n[-1, col_mask])),
        "nn_col": nn[-1, col],
        "Te_col": Te[-1, col],
    }


class _DualProgress:
    """Progress callback that tolerates BOTH protocols.

    Equilibration calls callback(fraction: float); the plasma run calls
    callback(progress_obj). Throttled so the plasma phase prints periodically
    with t, phase, dt, and the binding timestep constraint (to spot a crawl).
    """

    def __init__(self, every_steps=800):
        self.every = every_steps
        self._last = -every_steps

    def __call__(self, p):
        # As a progress_TRACKER we receive the SimulationProgress1D object
        # (progress_callback would instead receive a bare float -- see
        # solver._emit_progress).
        if isinstance(p, float):
            return
        if p.step - self._last < self.every and p.fraction < 1.0:
            return
        self._last = p.step
        print(f"  t={p.time*1e3:8.3f}ms {p.phase:<14} step={p.step:>8} "
              f"dt={p.accepted_dt:.2e} cap={p.step_cap} constr={p.active_constraint} "
              f"eta={p.wall_remaining_s:6.0f}s", flush=True)


def trace(variant, nx, out_h5, cached_path=None):
    """Run the ES config to completion with progress, save H5.

    Decisive end-to-end test: does it ignite (phase -> main_discharge), how long,
    and does dt crawl (watch active_constraint)? With ``cached_path`` the neutral
    seed is loaded from cache (skips the ~20 s equilibration) -- also a
    correctness check that the cached-seed run matches the live-equilibration one.
    """
    params, flags = build_config(variant, nx)
    if cached_path:
        flags["use_cached_neutral_seed"] = True
        params["neutral_seed_cache_dir"] = cached_path
    sim = LAPDSim1D(params, flags)
    t0 = _walltime.time()
    sim.start_simulation(progress_tracker=_DualProgress(), progress_interval_s=2.0e-5)
    res = sim.get_results()
    wall = _walltime.time() - t0
    phases = np.asarray(res.phase, dtype=str)
    t = np.asarray(res.time, float)
    I = np.asarray(res.cathode_diagnostics["source_I_tot"], float)
    for ph in ("pre_breakdown", "breakdown", "main_discharge", "afterglow"):
        hits = np.flatnonzero(phases == ph)
        if hits.size:
            print(f"# first {ph:<14} at t={t[hits[0]]*1e3:8.3f} ms  I={I[hits[0]]:8.2f} A")
    print(f"# ever main_discharge: {bool(np.any(phases=='main_discharge'))}  "
          f"I_max={np.nanmax(I):.1f} A  t_end={t[-1]*1e3:.3f} ms  wall={wall:.1f} s")
    if out_h5:
        save_result_hdf5(out_h5, res, params=params, flags=flags)
        print(f"# saved {out_h5}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nx", type=int, default=120)
    ap.add_argument("--t-probe", type=float, default=2.2e-3,
                    help="model (plasma) time to stop at [s]")
    ap.add_argument("--variants", default="baseline,char_off,birth_legacy,sgp3400,all3")
    ap.add_argument("--trace", metavar="VARIANT", default=None,
                    help="run VARIANT to completion with progress instead of probing")
    ap.add_argument("--out-h5", default=None, help="save the --trace run to this H5")
    ap.add_argument("--cached-seed", default=None,
                    help="load the neutral seed from this .npz (skip equilibration)")
    args = ap.parse_args()

    if args.trace:
        trace(args.trace, args.nx, args.out_h5, cached_path=args.cached_seed)
        return

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    print(f"# R5 ignition-seed isolation probe  nx={args.nx}  t_probe={args.t_probe*1e3:.3f} ms")
    print(f"# historical reference n_col ~ 9.8e8 (ignites); repaired-ES ~ 4.0e8 (below threshold)")
    hdr = ("variant", "phase", "main?", "I[A]", "Imax[A]", "n_col", "n_col/base",
           "n_puff", "ncolmx", "nn_col", "Te_col", "wall[s]")
    print(("{:<13}{:<14}{:<6}{:>9}{:>9}{:>11}{:>11}{:>11}{:>11}{:>11}{:>9}{:>9}").format(*hdr))
    sys.stdout.flush()

    base_ncol = None
    for v in variants:
        r = probe(v, args.nx, args.t_probe)
        if v == "baseline" or base_ncol is None:
            base_ncol = r["n_col"]
        ratio = r["n_col"] / base_ncol if base_ncol else float("nan")
        print(("{:<13}{:<14}{:<6}{:>9.2f}{:>9.2f}{:>11.3e}{:>11.2f}"
               "{:>11.3e}{:>11.3e}{:>11.3e}{:>9.3f}{:>9.1f}").format(
            r["variant"], r["phase"], "Y" if r["ever_main"] else "n",
            r["I_A"], r["I_max_A"], r["n_col"], ratio, r["n_puff"],
            r["n_col_max_cell"], r["nn_col"], r["Te_col"], r["wall_s"]))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
