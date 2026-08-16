"""sc1: the NEW reported-not-gated reads, post-hoc over the saved npz traces.

NO SOLVE. Reads only the trajectories efold1_traj.py already wrote, so the
reused efold1 baselines (a1, a4) get the same reads at zero cost without being
re-run. The npz 'trace' columns are, per efold1_traj.py:

    0 t   1 dt   2 I_loop   3 phi_c   4 I_eth*   5 <n>_active   6 n[2]
    7 n[7]   8 n_max

EXACT IMPLEMENTATIONS (disclosed, as the registration requires):

1. ANTI-VACUITY (gating for whether an arm's tau counts). The reference is
   I_wall = I_loop linearly interpolated to t = 1e-6 s -- THE SAME wall-riding
   reference the pre-registered onset definition uses (efold1_traj.py:173).
   decades = log10(max(I_loop) / I_wall). An arm reports NO-BUILD if
   decades < 1.0. The raw first-to-last ratio is printed alongside.

2. BPD LANDMARK CROSSINGS. n_crit = 1e11 cm^-3, the beam-plasma-discharge
   ignition condition w_pe ~ w_ce at 1000 G (0.97e11). Two crossings:
     t(n_max -> 1e11)        on column 8, the max over cells;
     t(<n>_column -> 1e11)   on column 5, the ACTIVE-CELL ARITHMETIC MEAN
                             (40 of 42 cells here) that the F2 estimator's own
                             upstream forms -- i.e. the column mean.
   Each is reported twice: the first accepted step at or above the threshold
   (raw, no interpolation), and a refinement linear in (t, ln n) between that
   step and its predecessor. NEVER CROSSED is a legitimate outcome.
   ***LABEL, TRAVELS WITH EVERY NUMBER: "BPD w_pe~w_ce landmark, EXTRAPOLATED
   different-stage physics" -- the literature anchor is keV beams in
   tens-of-G chambers (Getty & Smullin; Bernstein 1979; Papadopoulos 1982).
   It is a REPORTED LANDMARK, never an imposed switch; nothing in the model
   reads it.***

3. FOOT-vs-FINISH TAU SPLIT on I_loop. The BUILD WINDOW is
   [t_on, t_end], where t_on is the pre-registered onset (first accepted step
   with I_loop >= 10*I_wall) and t_end is the last accepted step (the t-target,
   or the cap if the arm was killed). The window is split BY TIME, not by step
   count: FOOT = the first 30% of (t_end - t_on), FINISH = the last 30%.
   Within each sub-window tau = 1/slope of an ordinary least-squares fit of
   ln I_loop against t (numpy.polyfit degree 1), requiring >= 2 accepted steps
   and I_loop > 0 throughout. A NEGATIVE slope (decaying leg) is reported as
   such rather than as a tau. The qualitative target is the WP-E shape:
   slow foot, fast finish -> tau_foot > tau_finish.
   With t_on NEVER (onset not reached) the split is not defined and says so.
"""
import json
import sys

import numpy as np

I_WALL_TIME = 1.0e-6
ONSET_FACTOR = 10.0
N_CRIT = 1.0e11
SPLIT_FRAC = 0.30

ARMS = [
    ("efold1_a1_baseline", "REUSED efold1 baseline (tau 6.86 us of record)"),
    ("efold1_a4_coverage", "REUSED efold1 coverage arm (tau 52.79 us of record)"),
    ("sc1_b30", "ql_relaxation c=30"),
    ("sc1_b100", "ql_relaxation c=100"),
    ("sc1_bx30", "ql_relaxation c=30 x coverage"),
    ("sc1_bx100", "ql_relaxation c=100 x coverage"),
]


def crossing(t, y, thr):
    """(t_raw, t_interp) of the first step with y >= thr, or (None, None)."""
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


def main():
    print("== sc1_reads: BPD landmarks, foot/finish tau split, anti-vacuity")
    print("   NO SOLVE -- post-hoc over saved efold1_traj npz traces.")
    print(f"   n_crit = {N_CRIT:g} cm^-3; split fraction = {SPLIT_FRAC:g}\n")
    for stem, note in ARMS:
        try:
            z = np.load(f"scripts/{stem}.npz", allow_pickle=False)
        except FileNotFoundError:
            print(f"--- {stem}: ABSENT\n")
            continue
        arr = np.asarray(z["trace"], float)
        meta = json.loads(str(z["meta"]))
        t, I, meann, nmax = arr[:, 0], arr[:, 2], arr[:, 5], arr[:, 8]
        print(f"--- {stem}  [{note}]")
        print(f"    sets={meta['sets']}  kernels={meta['kid']}  "
              f"steps={t.size}  wall={meta['wall_s']:.0f} s  "
              f"tau_F2={meta['tau_us']:.4f} us  refusal={meta['refusal']}")

        I_wall = float(np.interp(I_WALL_TIME, t, I))
        dec = float(np.log10(I.max() / I_wall))
        print(f"    ANTI-VACUITY: I_wall={I_wall:.6g} A, max I_loop="
              f"{I.max():.6g} A -> {dec:.3f} decades "
              f"({'BUILD (tau counts)' if dec >= 1.0 else 'NO-BUILD'}); "
              f"first->last ratio {I[-1] / I[0]:.6g}x")

        for nm, y in (("n_max      ", nmax), ("<n>_column ", meann)):
            raw, itp = crossing(t, y, N_CRIT)
            if raw is None:
                print(f"    BPD landmark t({nm.strip()} -> 1e11) = NEVER CROSSED "
                      f"in window (end value {y[-1]:.6g} cm^-3)")
            else:
                print(f"    BPD landmark t({nm.strip()} -> 1e11) = {itp:.6e} s "
                      f"(interp; first step at/above = {raw:.6e} s)")
        print("      ^ BPD w_pe~w_ce landmark, EXTRAPOLATED different-stage "
              "physics; reported, never imposed.")

        t_on = meta["t_on"]
        if t_on is None or not np.isfinite(t_on):
            print("    FOOT/FINISH: onset NEVER reached -> split undefined\n")
            continue
        t_end = float(t[-1])
        width = t_end - float(t_on)
        print(f"    FOOT/FINISH split over build window [t_on={t_on:.6e}, "
              f"t_end={t_end:.6e}] s (width {width:.6e} s):")
        print(leg_tau(t, I, float(t_on), float(t_on) + SPLIT_FRAC * width,
                      "FOOT  (first 30%)"))
        print(leg_tau(t, I, t_end - SPLIT_FRAC * width, t_end,
                      "FINISH (last 30%)"))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
