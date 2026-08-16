"""ea1v stage-1 READS: post-hoc analysis of the saved arm npz files. No solve.

Reads only what the two named harnesses wrote (efold1_traj traces, pd0_endvent
budgets) plus the femwrap f_em traces, and applies the registered conventions:

  * B3  = measured gain / all-surface end-loss at the seed -- pd0's OWN
          definitions (gain = LS slope of ln N_col; loss = median of
          (vent_coll + vent_cath + anode)/N_col), evaluated on the EARLY window
          t <= 1e-4 s (where f_em is still within ~15% of its seed) and, for
          completeness, on the full 1e-3 s window. B3 PASSES iff ratio < 1.
  * B1  = density LEVEL trajectory vs the F1 band 2-5e11 cm^-3, plus the
          all-surface loss rate lambda at foot conditions (same convention).
  * F2  = covcal_read.pedestal_efold, imported unmodified, on <n>_active.
  * 1 kA tracker + the current-anchored window-start markers 0.34 / 0.47 A.
  * the f_em anti-vacuity read and the measured percolation clock.
  * PROJECTION ONLY (not a B2 claim): t at which the measured f_em trajectory
    reaches the ea0 f_crit class 0.10-0.24.

Writes nothing but stdout.
"""
import sys

import numpy as np

from covcal_read import pedestal_efold          # THE F2 ESTIMATOR OF RECORD

F1_BAND = (2.0e11, 5.0e11)
F_CRIT = (0.10, 0.24)
R_SHARED = 1390.0
EARLY = 1.0e-4
MARKERS = (0.34, 0.47, 1000.0)

ARMS = ("seed", "seed_lo", "seed_hi", "disposal")


def marker(t, I, level):
    hit = np.flatnonzero(I >= level)
    return (float(t[hit[0]]) if hit.size else None)


def b3(t, Ncol, lam_tot, mask, tag):
    m = np.flatnonzero(mask)
    if m.size < 2:
        print(f"   {tag}: fewer than 2 steps -- not fittable")
        return
    g = float(np.polyfit(t[m], np.log(Ncol[m]), 1)[0])
    lo = float(np.median(lam_tot[m]))
    print(f"   {tag}: steps={m.size}  d ln N_col/dt = {g:.6e} 1/s   "
          f"all-surface loss median = {lo:.6e} 1/s")
    print(f"       GAIN/ALL-SURFACE-LOSS = {g/lo:.6g}   "
          f"B3 {'PASS (< 1)' if g/lo < 1.0 else 'FAIL (>= 1)'}")


def main():
    print("== ea1v stage-1 reads (post-hoc; no solve) ==\n")

    a = np.load("scripts/ea1v_seed.npz", allow_pickle=True)["trace"]
    b = np.load("scripts/ea1v_seed_nowrap.npz", allow_pickle=True)["trace"]
    print(f"-- WRAPPER CONTROL: femwrap vs UNWRAPPED efold1_traj, same deltas, "
          f"same window --")
    print(f"   shapes {a.shape} vs {b.shape}; raw bytes identical: "
          f"{a.tobytes() == b.tobytes()}\n")

    bl = np.load("scripts/ea1v_baseline_unarmed.npz", allow_pickle=True)["trace"]
    blf = np.load("scripts/ea1v_baseline_unarmed_fem.npz")["fem"]
    tb, Ib = bl[:, 0], bl[:, 2]
    print("-- UNARMED BASELINE CONTROL (same stance, same 1e-3 s window, "
          "flag OFF) --")
    print(f"   f_em present on {int(np.isfinite(blf[:, 1]).sum())} of "
          f"{blf.shape[0]} steps (None => closure not armed: presence gate)")
    print(f"   steps {bl.shape[0]}   I_loop {Ib[0]:.6g} -> {Ib[-1]:.6g} A "
          f"(max {Ib.max():.6g})   <n>_act {bl[0, 5]:.6e} -> {bl[-1, 5]:.6e}"
          f"   n_max_end {bl[-1, 8]:.6e} cm^-3")
    for lv in MARKERS:
        tm = marker(tb, Ib, lv)
        print(f"   t(first I_loop >= {lv:g} A) = "
              f"{'%.6e s' % tm if tm is not None else 'NEVER in window'}")
    print()

    for arm in ARMS:
        tr = np.load(f"scripts/ea1v_{arm}.npz", allow_pickle=True)["trace"]
        ev = np.load(f"scripts/ea1v_endvent_{arm}.npz", allow_pickle=True)
        fem = np.load(f"scripts/ea1v_{arm}_fem.npz")["fem"]
        t, I = tr[:, 0], tr[:, 2]
        meann, nmax = tr[:, 5], tr[:, 8]
        print(f"================ ARM ea1v_{arm} ================")
        f = fem[:, 1]
        print(f"-- f_em (measured on the scored trajectory) --")
        print(f"   f_em {f[0]:.10e} -> {f[-1]:.10e}  x{f[-1]/f[0]:.6f}  "
              f"= {np.log(f[-1]/f[0]):.6f} e-folds over {t[-1]:.6e} s")
        print(f"   monotone: {bool(np.all(np.diff(f) >= 0))}   "
              f"ANTI-VACUITY: {'PASS' if f[-1] > f[0] else 'VACUOUS'}")
        lg = np.log(f / (1.0 - f))
        r_meas = float(np.polyfit(fem[:, 0], lg, 1)[0])
        print(f"   measured logistic clock r = {r_meas:.6f} /s "
              f"(shared constant {R_SHARED:g} /s; ratio {r_meas/R_SHARED:.6f})")

        Ncol = ev["N_col"]
        lam_tot = np.divide(ev["vent_coll_p"] + ev["vent_cath_p"] + ev["anode_p"],
                            Ncol, out=np.zeros_like(Ncol), where=Ncol > 0)
        lam_coll = np.divide(ev["vent_coll_p"], Ncol,
                             out=np.zeros_like(Ncol), where=Ncol > 0)
        te = ev["t"]
        print(f"-- B3: GAIN vs ALL-SURFACE LOSS (pd0 conventions) --")
        b3(te, Ncol, lam_tot, te <= EARLY, f"EARLY window t<={EARLY:g} s (f_em ~ f_em0)")
        b3(te, Ncol, lam_tot, np.ones_like(te, bool), "FULL window t<=1e-3 s")
        print(f"   lambda (all-surface) at foot: first {lam_tot[0]:.4e}  "
              f"median-early {np.median(lam_tot[te <= EARLY]):.4e}  "
              f"end {lam_tot[-1]:.4e} 1/s")
        print(f"   lambda (collector-only): first {lam_coll[0]:.4e}  "
              f"median-early {np.median(lam_coll[te <= EARLY]):.4e}  "
              f"end {lam_coll[-1]:.4e} 1/s")

        print(f"-- B1: density LEVEL trajectory (F1 band {F1_BAND[0]:.1g}-"
              f"{F1_BAND[1]:.1g} cm^-3) --")
        print(f"   <n>_act {meann[0]:.6e} -> {meann[-1]:.6e} cm^-3   "
              f"n_max {nmax[0]:.6e} -> {nmax[-1]:.6e} cm^-3")
        print(f"   N_col {Ncol[0]:.6e} -> {Ncol[-1]:.6e}   "
              f"(dN/N = {(Ncol[-1]-Ncol[0])/Ncol[0]:+.3%})")
        reach = "REACHED" if nmax[-1] >= F1_BAND[0] else "NOT REACHED"
        print(f"   F1 band lower edge {F1_BAND[0]:.1g}: {reach} "
              f"(n_max/band_lo = {nmax[-1]/F1_BAND[0]:.4g})")
        sl = float(np.polyfit(t[t >= 1e-6], np.log(meann[t >= 1e-6]), 1)[0])
        print(f"   apparent d ln<n>_act/dt = {sl:.6e} 1/s -> "
              f"tau = {1e6/sl:.4f} us  (vs 1/r = 719 us; CALIBRATED, not scored)")

        print(f"-- F2 ESTIMATOR (covcal_read.pedestal_efold, unmodified) --")
        res = pedestal_efold(t, meann, None)
        if res is None:
            print("   too few steps on the build leg to read a slope")
        else:
            leg, i0, i1, gr, ef = res
            print(f"   leg: {leg}")
            if gr is None:
                print("   no monotone growth on the leg -> NO E-FOLD "
                      "(REPORTED-NOT-SCORED)")
            else:
                print(f"   TAU_1 = {ef:.6e} s = {ef*1e6:.4f} us "
                      f"(vs coverage clock 1/r = 719 us)")

        print(f"-- CURRENT markers --")
        print(f"   I_loop start {I[0]:.6g} A  max {I.max():.6g} A  "
              f"end {I[-1]:.6g} A  min {I.min():.6g} A")
        for lv in MARKERS:
            tm = marker(t, I, lv)
            print(f"   t(first I_loop >= {lv:g} A) = "
                  f"{'%.6e s' % tm if tm is not None else 'NEVER in window'}")

        print(f"-- SUPPRESSION vs the unarmed baseline (same window) --")
        Ib_i = np.interp(t, tb, Ib)
        print(f"   I_armed/I_unarmed: t=0 {I[0]/Ib[0]:.6g}  "
              f"t=1e-5 {np.interp(1e-5, t, I)/np.interp(1e-5, tb, Ib):.6g}  "
              f"t=1e-4 {np.interp(1e-4, t, I)/np.interp(1e-4, tb, Ib):.6g}  "
              f"t=1e-3 {I[-1]/Ib[-1]:.6g}")
        print(f"   f_em at those instants: {f[0]:.6g}  "
              f"{np.interp(1e-5, fem[:, 0], f):.6g}  "
              f"{np.interp(1e-4, fem[:, 0], f):.6g}  {f[-1]:.6g}")
        print(f"   (start-of-window ratio vs f_em0: "
              f"{(I[0]/Ib[0])/f[0]:.6g} x f_em0)")

        print(f"-- PROJECTION ONLY (not a B2 claim): f_em -> f_crit class --")
        f0 = float(f[0])
        for fc in F_CRIT:
            tc = (np.log(fc / (1 - fc)) - np.log(f0 / (1 - f0))) / r_meas
            print(f"   f_crit = {fc:.2f} -> t_cross = {tc*1e3:.4f} ms "
                  f"({tc/1e-3:.2f} x the stage-1 window)")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
