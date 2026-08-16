"""SECONDARY seed re-measurement: ne0 decade traces (RUN ARTIFACT, untracked).

Registered 2026-08-04 (Tom). Three trace-to-ignition runs at nx = 120, current
stance, ne0 scaled x0.1 / x1 / x10 and NOTHING else changed. Precedent for
shipping a driver as an untracked run artifact: scripts/invert_un_m7.py.

Stance cannot drift: the config is built by importing ``build_config`` from
``probe_sim1d_r5_ignition_seed`` and taking its ``baseline`` variant, which in
turn imports PARAM_OVERRIDES / FLAG_OVERRIDES from ``compare_sim1d_es1``. The
ONLY mutation applied here is ``params["ne0"] *= scale``.

Ignition time is read from the saved phase events (the M8-record convention):
the timestamp of the ``main_discharge`` event. Under the current stance
``_phase_transition_mode() == "current"``, so that event is emitted once, at
``_t_breakdown_trigger`` with reason ``I_breakdown``. This script does NOT
guess: if the number of ``main_discharge`` events is anything other than one,
it prints the raw phase_events table and exits non-zero.

Wall time is reported in two mechanically-defined segments, because
``start_simulation`` runs the neutral equilibration inline:
  wall_equil = wall elapsed until the first progress event whose phase does NOT
               start with "equilibrium" (i.e. the equilibration segment plus
               whatever startup precedes the first plasma progress event);
  wall_solve = the remainder of the ``start_simulation`` call.
Both, and their total, are printed. ne0 is in the hashed neutral-seed config, so
a cache miss and a live re-equilibration per scaled trace is expected.

One scale per invocation, so the caller can enforce a per-trace wall cap.

Run from <repo>/cablp with the fenicsx-env interpreter and PYTHONPATH=<repo>/cablp.
"""

import argparse
import sys
import time as _walltime

import numpy as np

from cablp.solvers._sim1d import LAPDSim1D
from probe_sim1d_r5_ignition_seed import build_config


class _SegmentTracker:
    """Progress tracker: throttled printout + equilibration/solve wall split."""

    def __init__(self, t0, every_steps=4000):
        self.t0 = t0
        self.every = every_steps
        self._last = -every_steps
        self.first_plasma_wall = None
        self.max_step_plasma = 0
        self.last_t_plasma = float("nan")
        self.phases_seen = []

    def __call__(self, p):
        if isinstance(p, float):
            return
        phase = str(p.phase)
        if phase not in self.phases_seen:
            self.phases_seen.append(phase)
        if not phase.startswith("equilibrium"):
            if self.first_plasma_wall is None:
                self.first_plasma_wall = _walltime.time() - self.t0
                # Reset the throttle baseline at the equilibration->plasma
                # boundary: the plasma run restarts its step counter, so a
                # carried-over baseline suppresses every plasma line.
                self._last = int(p.step) - self.every
            self.max_step_plasma = max(self.max_step_plasma, int(p.step))
            self.last_t_plasma = float(p.time)
        if p.step - self._last < self.every and p.fraction < 1.0:
            return
        self._last = p.step
        # wall= is printed so a KILLED run still yields the wall-vs-model-time
        # curve from the tee'd log alone.
        print(f"  wall={_walltime.time()-self.t0:8.1f}s t={p.time*1e3:9.4f}ms "
              f"{phase:<15} step={p.step:>9} "
              f"dt={p.accepted_dt:.3e} cap={p.step_cap} "
              f"constr={p.active_constraint}", flush=True)


def dump_phase_events(ev):
    print("  RAW phase_events:", flush=True)
    times = np.asarray(ev.get("time", ()), dtype=float)
    phases = np.asarray(ev.get("phase", ()), dtype=object)
    reasons = np.asarray(ev.get("reason", ()), dtype=object)
    print(f"    n_events = {times.size}", flush=True)
    for i in range(times.size):
        print(f"    [{i}] t={times[i]*1e3:12.6f} ms  phase={str(phases[i]):<16} "
              f"reason={str(reasons[i])}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nx", type=int, default=120)
    ap.add_argument("--scale", type=float, required=True,
                    help="multiplier applied to params['ne0']")
    ap.add_argument("--t-end", type=float, default=5.0e-3)
    # progress_interval_s gates emission on MODEL time (solver.py:3115). With dt
    # pinned near dt_min a model-time interval emits essentially nothing, so the
    # default here is 0.0 = emit every step; the tracker throttles the PRINTOUT
    # by step count. Diagnostic only -- emission frequency touches no physics.
    ap.add_argument("--progress-interval", type=float, default=0.0)
    ap.add_argument("--print-every", type=int, default=4000)
    args = ap.parse_args()

    params, flags = build_config("baseline", args.nx)
    ne0_base = float(params["ne0"])
    ne0_used = ne0_base * args.scale
    params["ne0"] = ne0_used

    print("=" * 78, flush=True)
    print(f"# TRACE  variant=baseline  nx={args.nx}  scale={args.scale:g}", flush=True)
    print(f"# ne0_base = {ne0_base:.6e} cm^-3   ne0_used = {ne0_used:.6e} cm^-3", flush=True)
    print(f"# t_end = {args.t_end:.6e} s ({args.t_end*1e3:.3f} ms)", flush=True)
    print(f"# phase_transition: current stance; ignition = main_discharge event time",
          flush=True)
    sys.stdout.flush()

    sim = LAPDSim1D(params, flags)
    t0 = _walltime.time()
    tracker = _SegmentTracker(t0, every_steps=args.print_every)
    sim.start_simulation(t_end=args.t_end, dt=None, operator_split=None,
                         max_steps=None, progress_tracker=tracker,
                         progress_interval_s=args.progress_interval)
    wall_total = _walltime.time() - t0
    res = sim.get_results()

    wall_equil = tracker.first_plasma_wall
    wall_solve = (wall_total - wall_equil) if wall_equil is not None else float("nan")

    ev = getattr(res, "phase_events", {}) or {}
    times = np.asarray(ev.get("time", ()), dtype=float)
    phases = np.asarray(ev.get("phase", ()), dtype=object)
    hits = [i for i in range(times.size) if str(phases[i]) == "main_discharge"]

    t = np.asarray(res.time, float)
    I = np.asarray(res.cathode_diagnostics["source_I_tot"], float)
    n_nonfinite = int(np.count_nonzero(~np.isfinite(np.asarray(res.n, float))))

    print(f"# phases seen in progress stream: {tracker.phases_seen}", flush=True)
    print(f"# n_saves={t.size}  t_end_reached={t[-1]*1e3:.4f} ms  "
          f"last_phase={str(np.asarray(res.phase, dtype=str)[-1])}", flush=True)
    print(f"# I_last={I[-1]:.2f} A  I_max={np.nanmax(I):.2f} A  "
          f"nonfinite_cells_in_n={n_nonfinite}", flush=True)
    print(f"# steps_plasma_last_progress={tracker.max_step_plasma}", flush=True)
    print(f"# wall_equil={wall_equil if wall_equil is None else round(wall_equil,1)} s  "
          f"wall_solve={wall_solve:.1f} s  wall_total={wall_total:.1f} s", flush=True)
    dump_phase_events(ev)

    if len(hits) != 1:
        print(f"!! AMBIGUOUS: {len(hits)} main_discharge events (expected exactly 1). "
              f"NOT guessing an ignition time. RESULT=AMBIGUOUS", flush=True)
        print(f"RESULT scale={args.scale:g} ne0={ne0_used:.6e} t_ign_ms=NONE "
              f"n_main_events={len(hits)} wall_total_s={wall_total:.1f} STATUS=NO_IGNITION_OR_AMBIGUOUS",
              flush=True)
        sys.exit(3)

    t_ign_ms = float(times[hits[0]]) * 1e3
    print(f"RESULT scale={args.scale:g} ne0={ne0_used:.6e} t_ign_ms={t_ign_ms:.6f} "
          f"n_main_events=1 steps={tracker.max_step_plasma} "
          f"wall_equil_s={-1.0 if wall_equil is None else wall_equil:.1f} "
          f"wall_solve_s={wall_solve:.1f} wall_total_s={wall_total:.1f} STATUS=OK",
          flush=True)


if __name__ == "__main__":
    main()
