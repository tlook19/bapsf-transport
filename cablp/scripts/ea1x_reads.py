"""ea1x ms-class READS: post-hoc over the saved checkpoint traces. NO SOLVE.

Reads only what scripts/ea1x_ckpt.py wrote (columns transcribed verbatim from
efold1_traj.py, proven raw-byte-identical to the stage-1 record in
scripts/ea1x_inertctl_identity.txt), and applies the registered conventions:

  * ANTI-VACUITY: f_em measured on the scored trajectory (femwrap column),
    growth factor, monotonicity, and the measured logistic clock r.
  * THE TURNAROUND: the minimum of <n>_active over the reachable window (the
    instant the decay stops and the forced response takes over) and the
    minimum of I_loop; the apparent d ln<n>/dt after it, against the
    CALIBRATED shared clock r = 1390 /s (reported, never scored).
  * B1: the pedestal LEVEL trajectory vs the F1 band 2-5e11 cm^-3.
  * THE CROSSING: the pre-registered onset t_on (first accepted step with
    I_loop >= 10*I_wall, I_wall = I_loop interpolated to t = 1e-6 s) as the
    avalanche-takeover marker, plus f_em at the end of the reachable window
    against the ea0 f_crit class 0.10-0.24. NOT REACHED is a RESULT: the
    crossing time is then a LOWER BOUND at the last reached time.
  * B2 CURRENT-ANCHORED (the phase ruling): t(first I_loop >= 0.34 / 0.47 /
    1000 A) and the intervals t(0.34->1 kA), t(0.47->1 kA), scored against the
    machine's >= 4.5 ms pedestal.
  * THE SHAPE: the sc1_reads foot-vs-finish tau split, same implementation --
    build window [t_on, t_end], split BY TIME at the first/last 30%, tau =
    1/slope of an OLS fit of ln I_loop; a non-positive slope is reported as a
    decaying leg, never as a tau. Undefined when t_on is never reached.
  * BPD LANDMARK crossings at n_crit = 1e11 cm^-3 on n_max and on the
    active-cell mean. LABEL, travels with every number: "BPD w_pe~w_ce
    landmark, EXTRAPOLATED different-stage physics; reported, never imposed."
  * B4: the two emission-insensitivity arms against the central seed arm --
    matched-step deltas over the common prefix (the registered bin: the
    throttle is area, not temperature).
  * a matched-time table across all arms (the seed bracket).

Writes nothing but stdout.
"""
import sys

import numpy as np

F1_BAND = (2.0e11, 5.0e11)
F_CRIT = (0.10, 0.24)
R_SHARED = 1390.0
N_CRIT = 1.0e11
SPLIT_FRAC = 0.30
I_WALL_TIME = 1.0e-6
ONSET_FACTOR = 10.0
MARKERS = (0.34, 0.47, 1000.0)
MATCH_T = (1.0e-5, 1.0e-4, 5.0e-4, 1.0e-3, 1.2e-3)

ARMS = [
    ("ea1x_seed", "central seed f_em0 = 0.0075"),
    ("ea1x_seed_lo", "bracket low f_em0 = 0.0063"),
    ("ea1x_seed_hi", "bracket high f_em0 = 0.0087"),
    ("ea1x_disposal", "central seed x landau_branched (phi_c fraction 1.0)"),
    ("ea1x_b4_ts", "B4 null: T_s 1998.15 -> 1910.0 K"),
    ("ea1x_b4_tebirth", "B4 null: Te_birth_ionization local -> floor"),
]


def load(stem):
    a = np.load(f"scripts/{stem}_ckpt.npz")["trace"]
    return np.asarray(a, float)


def crossing(t, y, thr):
    hit = np.flatnonzero(y >= thr)
    if not hit.size:
        return None, None
    i = int(hit[0])
    if i == 0 or y[i - 1] <= 0.0 or y[i] <= 0.0:
        return float(t[i]), float(t[i])
    la, lb, lt = np.log(y[i - 1]), np.log(y[i]), np.log(thr)
    if lb == la:
        return float(t[i]), float(t[i])
    w = (lt - la) / (lb - la)
    return float(t[i]), float(t[i - 1] + w * (t[i] - t[i - 1]))


def leg_tau(t, I, lo, hi, name):
    m = np.flatnonzero((t >= lo) & (t <= hi))
    if m.size < 2 or not np.all(I[m] > 0.0):
        return f"   {name}: not fittable ({m.size} steps in window)"
    s = float(np.polyfit(t[m], np.log(I[m]), 1)[0])
    span = f"[{lo:.6e}, {hi:.6e}] s, {m.size} steps"
    if s <= 0.0:
        return (f"   {name}: {span} -> d lnI/dt = {s:.6e} 1/s "
                f"(NON-POSITIVE: decaying/flat leg, NOT a tau)")
    return (f"   {name}: {span} -> d lnI/dt = {s:.6e} 1/s -> "
            f"tau = {1e6 / s:.4f} us")


def one(stem, note):
    a = load(stem)
    t, dt, I, phic, Ieth = (a[:, i] for i in range(5))
    mn, n2, n7, nmax, f = (a[:, i] for i in range(5, 10))
    print(f"================ ARM {stem}  [{note}] ================")
    print(f"   steps={t.size}  t reached {t[-1]:.6e} s of the 5.000000e-03 s "
          f"target ({100*t[-1]/5e-3:.3f}% of the window)")
    print(f"   dt: first {dt[0]:.3e}  median {np.median(dt):.3e}  "
          f"last {dt[-1]:.3e}  min {dt[dt > 0].min():.3e} s")

    print("-- ANTI-VACUITY: f_em measured on the scored trajectory --")
    print(f"   f_em {f[0]:.10e} -> {f[-1]:.10e}  x{f[-1]/f[0]:.6f} "
          f"= {np.log(f[-1]/f[0]):.6f} e-folds")
    print(f"   monotone non-decreasing: {bool(np.all(np.diff(f) >= 0))}   "
          f"ANTI-VACUITY: {'PASS (advanced)' if f[-1] > f[0] else 'VACUOUS'}")
    lg = np.log(f / (1.0 - f))
    r_meas = float(np.polyfit(t, lg, 1)[0])
    print(f"   measured logistic clock r = {r_meas:.6f} /s (shared constant "
          f"{R_SHARED:g} /s; ratio {r_meas/R_SHARED:.6f})")

    print("-- THE TURNAROUND (n) --")
    j = int(np.argmin(mn))
    if j == mn.size - 1:
        print(f"   <n>_act is STILL DECAYING at the last reached step: "
              f"{mn[0]:.6e} -> {mn[-1]:.6e} cm^-3 ({(mn[-1]-mn[0])/mn[0]:+.3%}) "
              f"-- NO TURNAROUND in the reachable window (t_turn > "
              f"{t[-1]:.6e} s, a LOWER BOUND)")
    else:
        print(f"   <n>_act minimum {mn[j]:.6e} cm^-3 at t = {t[j]:.6e} s "
              f"(step {j}); end {mn[-1]:.6e} cm^-3")
        m = np.flatnonzero(t >= t[j])
        if m.size >= 2:
            s = float(np.polyfit(t[m], np.log(mn[m]), 1)[0])
            print(f"   post-turnaround d ln<n>/dt = {s:.6e} 1/s "
                  f"(vs the CALIBRATED clock r = {R_SHARED:g} /s; "
                  f"ratio {s/R_SHARED:.4f}) -- reported, NOT scored")
    k = int(np.argmin(I))
    print(f"   I_loop minimum {I[k]:.6e} A at t = {t[k]:.6e} s (step {k}); "
          f"start {I[0]:.6g} A, end {I[-1]:.6g} A, max {I.max():.6g} A")
    m = np.flatnonzero(t >= t[k])
    if m.size >= 2 and np.all(I[m] > 0):
        s = float(np.polyfit(t[m], np.log(I[m]), 1)[0])
        print(f"   post-minimum d lnI/dt = {s:.6e} 1/s -> tau = "
              f"{1e6/s:.4f} us  (foot e-fold; CALIBRATED by construction "
              f"~719 us, reported-not-scored)")
    sl = float(np.polyfit(t[t >= 1e-6], np.log(mn[t >= 1e-6]), 1)[0])
    print(f"   whole-window apparent d ln<n>/dt = {sl:.6e} 1/s "
          f"-> tau = {1e6/sl:.4f} us")

    print(f"-- B1: pedestal LEVEL vs the F1 band "
          f"{F1_BAND[0]:.1g}-{F1_BAND[1]:.1g} cm^-3 --")
    print(f"   <n>_act {mn[0]:.6e} -> {mn[-1]:.6e}   "
          f"n_max {nmax[0]:.6e} -> {nmax[-1]:.6e} cm^-3")
    print(f"   n_max/band_lo at end = {nmax[-1]/F1_BAND[0]:.6g}  -> "
          f"{'REACHED' if nmax[-1] >= F1_BAND[0] else 'NOT REACHED'} "
          f"(factor {F1_BAND[0]/nmax[-1]:.4g} below the lower edge)")
    print(f"   B1 VERDICT: the forced response does NOT carry the column into "
          f"the band in the reachable window"
          if nmax[-1] < F1_BAND[0] else "   B1 VERDICT: band reached")

    print("-- BPD LANDMARKS (n_crit = 1e11 cm^-3) --")
    for nm, y in (("n_max", nmax), ("<n>_column", mn)):
        raw, itp = crossing(t, y, N_CRIT)
        if raw is None:
            print(f"   t({nm} -> 1e11) = NEVER CROSSED (end {y[-1]:.6g} cm^-3)")
        else:
            print(f"   t({nm} -> 1e11) = {itp:.6e} s (interp; first step "
                  f"at/above {raw:.6e} s)")
    print("     ^ BPD w_pe~w_ce landmark, EXTRAPOLATED different-stage "
          "physics; reported, never imposed.")

    print("-- THE CROSSING / B2 CURRENT-ANCHORED (the phase ruling) --")
    I_wall = float(np.interp(I_WALL_TIME, t, I))
    thr = ONSET_FACTOR * I_wall
    on = np.flatnonzero(I >= thr)
    t_on = float(t[on[0]]) if on.size else None
    print(f"   I_wall = I_loop(1e-6 s) = {I_wall:.6g} A; onset threshold "
          f"10*I_wall = {thr:.6g} A")
    print(f"   t_on (avalanche takeover) = "
          f"{'%.6e s' % t_on if t_on is not None else 'NEVER in the reachable window'}")
    tm = {}
    for lv in MARKERS:
        hit = np.flatnonzero(I >= lv)
        tm[lv] = float(t[hit[0]]) if hit.size else None
        print(f"   t(first I_loop >= {lv:g} A) = "
              f"{'%.6e s' % tm[lv] if tm[lv] is not None else 'NEVER in the reachable window'}")
    for lo in (0.34, 0.47):
        if tm[lo] is not None and tm[1000.0] is not None:
            d = (tm[1000.0] - tm[lo]) * 1e3
            print(f"   B2 interval t({lo} A -> 1 kA) = {d:.4f} ms  -> "
                  f"duration bin vs the machine's >= 4.5 ms pedestal: "
                  f"{'PASS' if d >= 4.5 else 'FAIL (undershoot -- the pre-stated risk)'}")
        else:
            print(f"   B2 interval t({lo} A -> 1 kA): UNQUOTABLE -- the armed "
                  f"foot never reaches {lo} A in the reachable window")
    print(f"   f_em at the last reached step = {f[-1]:.6g} vs the ea0 f_crit "
          f"class {F_CRIT[0]:.2f}-{F_CRIT[1]:.2f}: "
          f"{'INSIDE' if F_CRIT[0] <= f[-1] <= F_CRIT[1] else 'BELOW'} "
          f"(f_crit/f_em_end = {F_CRIT[0]/f[-1]:.4g}-{F_CRIT[1]/f[-1]:.4g}x)")
    for fc in F_CRIT:
        tc = (np.log(fc / (1 - fc)) - np.log(f[0] / (1 - f[0]))) / r_meas
        print(f"   PROJECTION ONLY (not a B2 claim): f_crit {fc:.2f} at "
              f"t = {tc*1e3:.4f} ms")

    print("-- THE SHAPE: foot-vs-finish tau split (sc1_reads implementation) --")
    if t_on is None:
        print("   onset NEVER reached -> the split is UNDEFINED; no "
              "slow-foot-then-fast-finish structure can be read on this arm")
    else:
        t_end = float(t[-1])
        width = t_end - t_on
        print(f"   build window [t_on={t_on:.6e}, t_end={t_end:.6e}] s "
              f"(width {width:.6e} s):")
        print(leg_tau(t, I, t_on, t_on + SPLIT_FRAC * width, "FOOT  (first 30%)"))
        print(leg_tau(t, I, t_end - SPLIT_FRAC * width, t_end, "FINISH (last 30%)"))
    print()
    return a


def main():
    print("== ea1x ms-class reads (post-hoc over the checkpoint traces; NO SOLVE) ==")
    print("   t-target 5e-3 s, 900 s kill-at-cap per arm, campaign @ 1abe696\n")
    data = {}
    for stem, note in ARMS:
        try:
            data[stem] = one(stem, note)
        except FileNotFoundError:
            print(f"================ ARM {stem}: CHECKPOINT ABSENT\n")

    ref = data.get("ea1x_seed")
    print("================ B4: EMISSION-INSENSITIVITY (vs the central arm) ================")
    if ref is None:
        print("   central arm absent -- B4 not evaluable")
    else:
        for stem in ("ea1x_b4_ts", "ea1x_b4_tebirth"):
            a = data.get(stem)
            if a is None:
                print(f"   {stem}: ABSENT")
                continue
            k = min(a.shape[0], ref.shape[0])
            same = np.ascontiguousarray(a[:k]).tobytes() == \
                np.ascontiguousarray(ref[:k]).tobytes()
            d = np.abs(a[:k] - ref[:k])
            names = ("t", "dt", "I_loop", "phi_c", "I_eth*", "<n>_act",
                     "n[2]", "n[7]", "n_max", "f_em")
            print(f"-- {stem}: common prefix {k} steps "
                  f"(arm {a.shape[0]}, central {ref.shape[0]})")
            print(f"   RAW BYTES identical over the common prefix: {same}")
            for i, nm in enumerate(names):
                col = d[:, i]
                den = np.max(np.abs(ref[:k, i]))
                print(f"   max|delta| {nm:8s} = {col.max():.6e}"
                      + (f"   (rel {col.max()/den:.3e})" if den > 0 else ""))
            print(f"   B4 for this arm: "
                  f"{'PASS -- the null survives (no material movement)' if same else 'see the deltas above'}")
    print()

    print("================ MATCHED-TIME TABLE (the seed bracket) ================")
    hdr = "   arm                 " + "".join(f"{tt:>13.3e}" for tt in MATCH_T)
    for q, lab in ((2, "I_loop [A]"), (5, "<n>_act [cm^-3]"), (9, "f_em")):
        print(f"-- {lab}")
        print(hdr)
        for stem, _ in ARMS:
            a = data.get(stem)
            if a is None:
                continue
            vals = []
            for tt in MATCH_T:
                vals.append(np.interp(tt, a[:, 0], a[:, q]) if tt <= a[-1, 0]
                            else float("nan"))
            print(f"   {stem:20s}" + "".join(f"{v:13.5g}" for v in vals))
    print("\n   (NaN = beyond the arm's reachable window)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
