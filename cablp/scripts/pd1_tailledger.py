"""pd1: the TAIL END-LEDGER share of P_QL, per accepted step.

WRAPPER around the same accepted-step loop efold1_traj.py and
pd0_endvent_traj.py run (regime_r2_overlap_gate.build_config +
recertdiag2_traj.apply_set, ``sim.run(t_end, max_steps=1)``, same exception
handling). SINGLE-DELTA ON INSTRUMENTATION ONLY: after each accepted step it
reads the deposition rows off a cathode solve taken with update_cache=False,
and the per-step (t, I_loop) trace is compared against the arm's recorded
efold1 npz as the IDENTITY CONTROL -- the same control pd0_endvent_traj prints.

THE ROWS AND THE IDENTITY, both the build's own (smoke_sim1d.py pd1 case (d),
solver.py:9898 for the two ledger diagnostics). Summed over ends:

    P_QL     = heating_anomalous + ionization_cost_tail + radiated_tail
               + end_loss_tail_low + end_loss_tail_high
    launched = tail_power_erg_s                     (= f_Landau-weighted P_QL)
    vented   = end_loss_tail_low + end_loss_tail_high

so the LAUNCHED SHARE is launched/P_QL and the VENTED SHARE is vented/P_QL,
the second being the part of the extracted QL power that leaves the machine at
an end rather than being delivered anywhere in the column. Both are reported
as power-weighted time integrals over the full and build windows (the honest
reduction for a share that varies by two orders of magnitude across the ramp)
with the per-step median alongside, and the identity residual is printed as
its own control.

Writes only pd1_* artifacts. Read-only w.r.t. the repo.
"""
import argparse
import json
import sys
import time
import warnings

import numpy as np

from cablp.solvers._sim1d import LAPDSim1D
from recertdiag2_traj import apply_set

I_WALL_TIME = 1.0e-6
ONSET_FACTOR = 10.0


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--nx", type=int, default=20)
    p.add_argument("--t-target", type=float, default=1.0e-4)
    p.add_argument("--set", dest="sets", nargs="*", default=())
    p.add_argument("--npz", required=True)
    p.add_argument("--ref-npz", default=None)
    args = p.parse_args(argv)
    warnings.simplefilter("ignore")

    from regime_r2_overlap_gate import build_config
    params, flags = build_config(args.nx, False)
    applied = apply_set(params, flags, args.sets)

    from cablp.funcs import _kernels
    kid = str(getattr(_kernels.COMPILED_KERNELS, "KERNEL_ID",
                      _kernels.COMPILED_KERNELS))
    sim = LAPDSim1D(params, flags)
    print(f"== pd1_tailledger label={args.label} nx={args.nx} "
          f"t_target={args.t_target:g}")
    print(f"   deltas via --set: {applied}")
    print(f"   kernels: {kid}")

    rows = []
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
        solve = sim.solve_cathode_boundary(state=sim.state, time=t,
                                           update_cache=False)
        anom = ion = rad = led = launched = 0.0
        for dep in (getattr(solve, "beam_deposition", None) or {}).values():
            if dep is None:
                continue
            anom += float(np.asarray(dep.heating_anomalous_erg_s).sum())
            ion += float(np.asarray(dep.ionization_cost_tail_erg_s).sum())
            rad += float(np.asarray(dep.radiated_tail_erg_s).sum())
            led += (float(dep.end_loss_tail_low_erg_s)
                    + float(dep.end_loss_tail_high_erg_s))
            launched += float(dep.tail_power_erg_s)
        rows.append((t, float(sim._circuit_I_loop), anom, ion, rad, led,
                     launched))
        if len(rows) > 400000:
            print("   BAILING: >400000 accepted steps")
            break

    wall = time.time() - t0
    arr = np.asarray(rows, float)
    ns = arr.shape[0]
    print(f"   accepted steps: {ns}  ({wall:.0f} s wall)")
    if refusal:
        print(f"   REFUSAL at t={float(sim._time):.6g}: {refusal}")
    if not ns:
        return 0
    t, I, anom, ion, rad, led, launched = (arr[:, i] for i in range(7))
    P_QL = anom + ion + rad + led

    if args.ref_npz:
        ref = np.load(args.ref_npz, allow_pickle=True)
        tr = np.asarray(ref["trace"], float)
        m = min(tr.shape[0], ns)
        dt_max = float(np.max(np.abs(tr[:m, 0] - t[:m])))
        dI_max = float(np.max(np.abs(tr[:m, 2] - I[:m])))
        same = tr.shape[0] == ns
        print(f"   IDENTITY CONTROL vs {args.ref_npz}: steps {tr.shape[0]} vs "
              f"{ns} (match={same}); max|dt|={dt_max:.3e} s, "
              f"max|dI_loop|={dI_max:.3e} A "
              f"{'-- IDENTICAL' if same and dt_max == 0.0 and dI_max == 0.0 else '-- DIFFERS'}")

    I_wall = float(np.interp(I_WALL_TIME, t, I))
    on = np.flatnonzero(I >= ONSET_FACTOR * I_wall)
    i_on = int(on[0]) if on.size else 0
    print(f"   onset (efold1 def): t_on={t[i_on]:.6g} s (step {i_on}), "
          f"I_wall={I_wall:.6g} A")

    live = P_QL > 0.0
    print(f"   steps with P_QL > 0: {int(live.sum())} of {ns} "
          f"(first at t={t[live][0]:.6e} s)" if live.any() else
          "   P_QL is identically zero on every step")
    if not live.any():
        np.savez(args.npz, trace=arr, meta=np.array(json.dumps(
            {"label": args.label, "sets": applied, "kid": kid,
             "wall_s": wall, "refusal": refusal})))
        print(f"   saved {args.npz}")
        return 0

    for name, i0 in (("FULL WINDOW", 0), ("BUILD WINDOW (onset->end)", i_on)):
        s = slice(i0, None)
        E_QL = float(np.trapezoid(P_QL[s], t[s]))
        E_lau = float(np.trapezoid(launched[s], t[s]))
        E_ven = float(np.trapezoid(led[s], t[s]))
        E_anom = float(np.trapezoid(anom[s], t[s]))
        E_ion = float(np.trapezoid(ion[s], t[s]))
        E_rad = float(np.trapezoid(rad[s], t[s]))
        print(f"\n   -- TAIL END-LEDGER, {name} [{t[i0]:.4g}, {t[-1]:.4g}] s --")
        print(f"   integrated P_QL = {E_QL:.6e} erg")
        print(f"   LAUNCHED share (tail_power / P_QL)        = "
              f"{E_lau/E_QL:.6f}")
        print(f"   VENTED share  (end_loss_tail / P_QL)      = "
              f"{E_ven/E_QL:.6f}")
        print(f"   delivered-to-column: bulk anomalous {E_anom/E_QL:.6f}  "
              f"tail ionization {E_ion/E_QL:.6f}  tail radiated "
              f"{E_rad/E_QL:.6f}")
        print(f"   vented/launched (escape fraction of what was launched) = "
              f"{E_ven/E_lau:.6f}" if E_lau > 0 else
              "   vented/launched: nothing launched")
        m = live & (np.arange(ns) >= i0)
        print(f"   per-step median over live steps: launched "
              f"{np.median(launched[m]/P_QL[m]):.6f}  vented "
              f"{np.median(led[m]/P_QL[m]):.6f}  ({int(m.sum())} steps)")
        resid = np.abs((anom + ion + rad + led - P_QL)[m]) / P_QL[m]
        print(f"   IDENTITY CONTROL max|resid|/P_QL = {resid.max():.3e}")

    np.savez(args.npz, trace=arr, meta=np.array(json.dumps(
        {"label": args.label, "sets": applied, "kid": kid, "wall_s": wall,
         "refusal": refusal, "i_on": i_on})))
    print(f"   saved {args.npz}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
