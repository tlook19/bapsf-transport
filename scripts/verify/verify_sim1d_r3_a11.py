"""R3 A11 convergence gate: sequential fluid->surface->circuit coupling.

Audit A11 (`RETAIN + CONVERGENCE GATE`, P1): the fluid SSPRK stages run at a loop
current FROZEN over the step; after acceptance the cathode T_s/coverage and then
the circuit are advanced from the accepted plasma. Heat Strang splitting does not
make that coupled update second-order. The small production timestep may make the
splitting error negligible -- but that must be SHOWN, by a coupled fixed-dt
refinement around the two places the plasma, T_s, and circuit all move fastest:
the emission knee (early main discharge) and the current fall edge (shutoff).

Method: run the production stance (adaptive) to a target time, snapshot the
coupled state, then advance a fixed window [t0, t0+T] with the FULL coupled step
(`advance_one_step`, which does fluid -> cathode T_s -> circuit) at fixed
dt = T/N for N, 2N, 4N. Self-convergence of the coupled observables
(V_b, phi_c, T_s, I_tot) gives the temporal order p from
`|X(dt/2)-X(dt/4)| / |X(dt/4)-X(dt/8)| ~ 2^p`, and the production-dt error is the
gap to the finest. Clean 2nd-order (or a negligible production-dt error) => the
sequential advance is RETAINED with this gate on record; a stalled/1st-order
coupling error => the finding is it must be coupled (a logged deliverable).

This is a short fixed-dt diagnostic, not an ES campaign point.

Usage:  python scripts/verify_sim1d_r3_a11.py [--phase knee|fall] [--t0-ms X]
                                              [--window-us W] [--n0 N]
"""
import argparse
import sys

import numpy as np

from cablp.solvers._sim1d import LAPDSim1D
# scripts/ sibling imports: the seven purpose subdirectories on sys.path.
import sys as _sys
from pathlib import Path as _Path
for _sub in ("atomic", "gates", "kinetic", "run", "score", "stance",
             "verify"):
    _dir = str(_Path(__file__).resolve().parents[1] / _sub)
    if _dir not in _sys.path:
        _sys.path.insert(0, _dir)

from baseline_sim1d import build_baseline_config


COUPLED_ATTRS = (
    "_time", "_circuit_I_loop", "_cathode_Ts_K",
    "_cathode_x0", "_cathode_x0_twin",
)


def _snapshot(sim):
    snap = {"_y": sim._y.copy()}
    for a in COUPLED_ATTRS:
        snap[a] = getattr(sim, a)
    bc = getattr(sim, "_cathode_beam_cross", None)
    snap["_cathode_beam_cross"] = None if bc is None else np.asarray(bc).copy()
    return snap


def _restore(sim, snap):
    sim._y = snap["_y"].copy()
    for a in COUPLED_ATTRS:
        setattr(sim, a, snap[a])
    bc = snap["_cathode_beam_cross"]
    sim._cathode_beam_cross = None if bc is None else bc.copy()
    # keep the state object in sync with the packed vector
    sim._state = None


def _build_sim(picard=False, picard_tol=None):
    # The exact golden/production stance (baseline_sim1d): the ES1 drive
    # (V_bank etc.) and M6 constants are essential -- without them the discharge
    # never forms and the stepper crawls.
    params, flags = build_baseline_config()
    if picard:
        # R5.1/A11: the gated fluid<->circuit Picard fix under test. A tiny
        # picard_tol FORCES the re-run every driven step (a diagnostic: does
        # full self-consistent coupling remove the knee dt-error, or is the
        # residual the separate SCL-corner chatter?).
        flags = {**flags, "coupled_circuit_picard": True}
        if picard_tol is not None:
            params = {**params, "circuit_picard_tol_rel": picard_tol}
    return LAPDSim1D(params, flags)


def _observables(sim):
    solve = sim._cathode_solve
    r = solve.beam_result.result if (solve and solve.beam_result) else None
    return {
        "V_b": float(r.V_b) if r else np.nan,
        "phi_c": float(r.phi_c) if r else np.nan,
        "T_s": float(sim._cathode_Ts_K) if sim._cathode_Ts_K is not None else np.nan,
        "I_tot": float(sim._circuit_I_loop),
    }


def _fresh_to_t0_then_window(t0, T, n, picard=False, picard_tol=None):
    """Build a FRESH sim, integrate (adaptive, current-gated) to t0, then advance
    the fixed window [t0, t0+T] with the full coupled step at dt=T/n.

    Fresh per refinement guarantees an identical starting state at t0 (the run to
    t0 is deterministic), avoiding the incomplete-snapshot fragility.
    """
    sim = _build_sim(picard=picard, picard_tol=picard_tol)
    sim.start_simulation(t_end=t0)
    dt = T / n
    for _ in range(n):
        sim.advance_one_step(dt=dt)
    obs = _observables(sim)
    if picard:
        obs["_picard_reruns"] = int(getattr(sim, "_picard_extra_solves", 0))
    return obs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=["knee", "fall"], default="knee")
    ap.add_argument("--t0-ms", type=float, default=None,
                    help="absolute model time [ms] to start the refinement window")
    ap.add_argument("--window-us", type=float, default=2.0,
                    help="refinement window length [us]")
    ap.add_argument("--n0", type=int, default=4, help="coarsest step count")
    ap.add_argument("--picard", action="store_true",
                    help="enable the R5.1/A11 gated fluid<->circuit Picard fix")
    ap.add_argument("--picard-tol", type=float, default=None,
                    help="override circuit_picard_tol_rel (tiny => force re-run "
                         "every driven step; diagnostic)")
    args = ap.parse_args(argv)

    # The production stance is CURRENT-GATED (phase_transition_mode="current",
    # I_breakdown=1000 A): breakdown fires dynamically at ~3.07 ms and truncates
    # the prebreakdown, so tau_prebreakdown (50 ms) is only a timeout cap. Place
    # the refinement windows on the LIVE main discharge (onset ~3.08 ms, duration
    # tau_discharge=20 ms), NOT the scheduled clock: the emission knee ~4.5 ms
    # (current ramping to peak) and the fall edge ~22 ms (shutoff).
    if args.t0_ms is not None:
        t0 = args.t0_ms * 1e-3
    elif args.phase == "knee":
        t0 = 4.5e-3
    else:
        t0 = 22.0e-3
    T = args.window_us * 1e-6
    print(f"phase={args.phase}  refinement window "
          f"[{t0*1e3:.4f}, {(t0+T)*1e3:.4f}] ms  (T={T*1e6:.2f} us; fresh-sim mode)")

    ns = [args.n0, 2 * args.n0, 4 * args.n0, 8 * args.n0]
    results = {}
    for n in ns:
        results[n] = _fresh_to_t0_then_window(
            t0, T, n, picard=args.picard, picard_tol=args.picard_tol)
        r = results[n]
        rr = f" reruns={r['_picard_reruns']}" if "_picard_reruns" in r else ""
        print(f"  N={n:3d} dt={T/n*1e9:7.2f} ns : V_b={r['V_b']:.6g} "
              f"phi_c={r['phi_c']:.6g} T_s={r['T_s']:.7g} I_tot={r['I_tot']:.7g}"
              f"{rr}")

    # self-convergence order + production-dt error (coarsest vs finest)
    print("\nself-convergence (order p from successive halvings; "
          "err vs finest):")
    ok = True
    for key in ("V_b", "phi_c", "T_s", "I_tot"):
        seq = [results[n][key] for n in ns]
        d1, d2, d3 = (seq[1] - seq[0]), (seq[2] - seq[1]), (seq[3] - seq[2])
        p_order = (np.log2(abs(d1 / d2)) if d2 else np.nan,
                   np.log2(abs(d2 / d3)) if d3 else np.nan)
        finest = seq[-1]
        rel_coarse = abs(seq[0] - finest) / max(abs(finest), 1e-30)
        print(f"  {key:6s}: p~({p_order[0]:.2f},{p_order[1]:.2f})  "
              f"coarse-dt rel err={rel_coarse:.2e}")
        # gate: the production-scale (coarse) error is small, OR order >= ~1.8
        ok &= (rel_coarse < 1e-3) or (np.isfinite(p_order[1]) and p_order[1] > 1.5)

    print("\nA11 coupling verdict:",
          "RETAIN-WITH-GATE (coupling error negligible / converges)" if ok
          else "MUST-COUPLE (coupling error not negligible) -- logged finding")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
