"""efold1: onset/e-fold ownership probe -- WRAPPER around recertdiag2_traj.

Does NOT modify recertdiag2_traj.py. It imports that module's --set namespace
resolver (``apply_set``) and reproduces its accepted-step loop verbatim
(``sim.run(t_end, max_steps=1)`` with the same exception handling), adding the
per-step density record the original does not keep:

    t, dt, I_loop, phi_c, I_eth_star, <n>_active, n[2], n[7], n_max

Instruments (all definitions imported or transcribed, and disclosed):
  * e-fold tau: ``covcal_read.pedestal_efold`` -- THE F2 ESTIMATOR OF RECORD,
    imported, unmodified, applied to (t, <n>_active, t_breakdown) where
    <n>_active is the active-cell arithmetic mean the estimator's own upstream
    (covcal_efold_read.load) forms. With no breakdown in the window the
    estimator takes its own SURVIVING-BUILD-LEG branch.
  * a disclosed secondary least-squares slope of ln<n> and ln I_loop over a
    stated window, reported alongside -- never in place of -- the estimator.
  * onset delay t_on: first accepted step with I_loop >= 10 * I_wall, where
    I_wall = I_loop linearly interpolated to t = 1e-6 s (pre-registered).

Read-only w.r.t. the repo: writes only efold1_* artifacts.
"""
import argparse
import json
import sys
import time
import warnings

import numpy as np

from cablp.solvers._sim1d import LAPDSim1D

from recertdiag2_traj import apply_set          # the namespace-safe --set
from covcal_read import pedestal_efold          # THE F2 ESTIMATOR OF RECORD

I_WALL_TIME = 1.0e-6
ONSET_FACTOR = 10.0


def _breakdown_time(sim):
    """t of the first 'breakdown' phase event in the window, or None."""
    try:
        ev = sim._phase_events(run_start=0.0, final_time=float(sim._time))
    except Exception as err:                       # instrument, never fatal
        print(f"   [phase_events unavailable: {type(err).__name__}: {err}]")
        return None
    try:
        times = np.asarray(ev["time"], float)
        phases = [v.decode() if isinstance(v, (bytes, bytearray)) else str(v)
                  for v in np.asarray(ev["phase"])]
    except Exception:
        return None
    hits = [t for t, p in zip(times, phases) if p == "breakdown"]
    return float(hits[0]) if hits else None


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--nx", type=int, default=20)
    p.add_argument("--t-target", type=float, default=1.0e-4)
    p.add_argument("--set", dest="sets", nargs="*", default=())
    p.add_argument("--fit-lo", type=float, default=1.0e-6)
    p.add_argument("--fit-hi", type=float, default=None)
    p.add_argument("--npz", default=None)
    args = p.parse_args(argv)
    warnings.simplefilter("ignore")

    from regime_r2_overlap_gate import build_config
    params, flags = build_config(args.nx, False)      # family r2, tracer OFF
    applied = apply_set(params, flags, args.sets)

    from cablp.funcs import _kernels
    kid = str(getattr(_kernels.COMPILED_KERNELS, "KERNEL_ID",
                      _kernels.COMPILED_KERNELS))
    sim = LAPDSim1D(params, flags)
    active = np.asarray(sim.geometry.plasma_active, bool)
    print(f"== efold1_traj label={args.label} family=r2 nx={args.nx} "
          f"tracer=False t_target={args.t_target:g}")
    print(f"   deltas via --set: {applied}")
    print(f"   kernels: {kid}")
    print(f"   active cells: {int(active.sum())} of {active.size}")

    rows = []
    t_prev = 0.0
    t0 = time.time()
    refusal = None
    while sim._time < args.t_target:
        try:
            sim.run(t_end=args.t_target, max_steps=1)
        except RuntimeError:
            pass
        except Exception as err:
            refusal = f"{type(err).__name__}: {err}"
            break
        t = float(sim._time)
        E0 = G0 = float("nan")
        solve = sim._cathode_solve
        br = getattr(solve, "beam_result", None) if solve is not None else None
        res = getattr(br, "result", None) if br is not None else None
        if res is not None:
            E0 = float(res.phi_c)
            G0 = float(res.I_eth_star)
        n = np.asarray(sim.state.n, float)
        rows.append((t, t - t_prev, float(sim._circuit_I_loop), E0, G0,
                     float(n[active].mean()), float(n[2]), float(n[7]),
                     float(n.max())))
        t_prev = t
        if len(rows) > 400000:
            print("   BAILING: >400000 accepted steps")
            break

    wall = time.time() - t0
    arr = np.asarray(rows, dtype=float)
    nsteps = arr.shape[0]
    print(f"   accepted steps: {nsteps}  ({wall:.0f} s wall)")
    if not nsteps:
        print("   NO ACCEPTED STEPS")
        if refusal:
            print(f"   REFUSAL: {refusal}")
        return 0

    t, dt, I, phic, Ieth = (arr[:, i] for i in range(5))
    meann, n2, n7, nmax = (arr[:, i] for i in range(5, 9))

    print(f"   {'t [s]':>12} {'dt [s]':>12} {'I_loop [A]':>12} "
          f"{'phi_c [V]':>10} {'I_eth* [A]':>10} {'<n>_act':>12} "
          f"{'n_max':>12}")
    show = (np.arange(nsteps) if nsteps <= 24 else
            np.unique(np.concatenate([np.arange(6),
                                      np.linspace(6, nsteps - 1, 18).astype(int)])))
    for i in show:
        print(f"   {t[i]:12.6g} {dt[i]:12.6g} {I[i]:12.6g} {phic[i]:10.5g} "
              f"{Ieth[i]:10.5g} {meann[i]:12.6g} {nmax[i]:12.6g}")

    t_bd = _breakdown_time(sim)
    print(f"\n   -- F2 ESTIMATOR (covcal_read.pedestal_efold, imported "
          f"unmodified) on <n>_active --")
    print(f"   breakdown event in window: "
          f"{'t_bd = %.6e s' % t_bd if t_bd is not None else 'NONE'}")
    res_ef = pedestal_efold(t, meann, t_bd)
    tau_us = float("nan")
    if res_ef is None:
        print("   too few steps on the build leg to read a slope")
    else:
        leg, a, b, gr, ef = res_ef
        print(f"   leg: {leg}")
        if gr is None:
            print("   no monotone growth on the leg -> NO E-FOLD")
        else:
            tau_us = ef * 1e6
            print(f"   {meann[a]:.6e} -> {meann[b]:.6e} cm^-3 "
                  f"({meann[b]/meann[a]:.6g}x) over {t[b]-t[a]:.6e} s "
                  f"({b-a+1} steps)")
            print(f"   mean log-slope {gr:.6e} 1/s")
            print(f"   TAU_1 = {ef:.6e} s = {tau_us:.4f} us")
            print(f"   band [713.0, 725.0] us -> "
                  f"{'IN BAND' if 713.0 <= tau_us <= 725.0 else 'OUT OF BAND'}"
                  f"; tau/719 = {tau_us/719.0:.6g}")

    lo = args.fit_lo
    hi = t[-1] if args.fit_hi is None else args.fit_hi
    m = np.flatnonzero((t >= lo) & (t <= hi))
    print(f"\n   -- SECONDARY least-squares slopes over [{lo:g}, {hi:g}] s "
          f"({m.size} steps), DISCLOSED, not the estimator --")
    for nm, y in (("ln<n>_act", meann), ("ln I_loop", I)):
        if m.size >= 2 and np.all(y[m] > 0):
            s = float(np.polyfit(t[m], np.log(y[m]), 1)[0])
            print(f"   d {nm}/dt = {s:.6e} 1/s -> tau = {1e6/s:10.4f} us")
        else:
            print(f"   d {nm}/dt : not fittable on this window")

    I_wall = float(np.interp(I_WALL_TIME, t, I))
    thr = ONSET_FACTOR * I_wall
    on = np.flatnonzero(I >= thr)
    t_on = float(t[on[0]]) if on.size else float("nan")
    print(f"\n   -- ONSET (pre-registered) --")
    print(f"   I_wall = I_loop(t=1e-6 s) = {I_wall:.6g} A "
          f"(interp; bracketing steps t={t[max(0,np.searchsorted(t,I_WALL_TIME)-1)]:.6g}"
          f"/{t[min(nsteps-1,np.searchsorted(t,I_WALL_TIME))]:.6g})")
    print(f"   threshold 10*I_wall = {thr:.6g} A")
    print(f"   t_on = {t_on:.6g} s" if on.size else
          "   t_on = NEVER (I_loop never reaches 10*I_wall in window)")

    print(f"\n   -- ENDPOINT at t={t[-1]:.6g} s --")
    print(f"   I_loop = {I[-1]:.6g} A   n[2] = {n2[-1]:.6g}   "
          f"n[7] = {n7[-1]:.6g}   n_max = {nmax[-1]:.6g} cm^-3")
    print(f"   I_loop max over window: {I.max():.6g} A")
    if refusal:
        print(f"   REFUSAL at t={float(sim._time):.6g}: {refusal}")
    print(f"\n   SUMMARY {args.label}: tau_F2={tau_us:.4f} us  "
          f"t_on={t_on:.6g} s  I_wall={I_wall:.6g} A  "
          f"I_end={I[-1]:.6g} A  n2={n2[-1]:.6g}  n7={n7[-1]:.6g}  "
          f"nmax={nmax[-1]:.6g}  wall={wall:.0f} s")

    if args.npz:
        np.savez(args.npz, trace=arr, n=np.asarray(sim.state.n, float),
                 active=active, meta=np.array(json.dumps(
                     {"label": args.label, "sets": applied, "kid": kid,
                      "t_bd": t_bd, "tau_us": tau_us, "t_on": t_on,
                      "I_wall": I_wall, "wall_s": wall,
                      "refusal": refusal})))
        print(f"   saved {args.npz}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
