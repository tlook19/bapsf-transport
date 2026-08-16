"""ea1x SUPPLEMENTARY READS: the B1 before-vs-after-crossing question, the
matched-time all-surface loss rate lambda (pd0 convention), and the disposal
composition delta. NO SOLVE -- post-hoc over the saved npz artifacts.

B1 as registered asks whether the FORCED RESPONSE carries the column into the
F1 band 2-5e11 cm^-3 BEFORE the crossing. Only one ea1x arm crosses inside its
reachable window (ea1x_seed_hi), so that is where the question has an answer;
for the stalled arms the answer is a bound.

lambda convention (fixed in advance, pd0's own): all-surface end loss =
(vent_coll + vent_cath + anode) / N_col, quoted at matched times.
"""
import sys

import numpy as np

F1_LO, F1_HI = 2.0e11, 5.0e11
MATCH_T = (1.0e-4, 5.0e-4, 1.0e-3, 1.2e-3)


def lam(ev):
    N = ev["N_col"]
    return ev["t"], np.divide(ev["vent_coll_p"] + ev["vent_cath_p"] + ev["anode_p"],
                              N, out=np.zeros_like(N), where=N > 0), N


def main():
    print("== ea1x supplementary reads (no solve) ==\n")

    hi = np.load("scripts/ea1x_seed_hi_ckpt.npz")["trace"]
    t, I, mn, nmax, f = hi[:, 0], hi[:, 2], hi[:, 5], hi[:, 8], hi[:, 9]
    t_on = 2.194904e-03            # from ea1x_reads.txt / efold1 onset definition
    print("-- B1 ON THE ONE ARM THAT CROSSES (ea1x_seed_hi) --")
    print(f"   t_on (avalanche takeover)        = {t_on:.6e} s")
    print(f"   <n>_act at t_on                  = {np.interp(t_on, t, mn):.6e} cm^-3")
    print(f"   n_max   at t_on                  = {np.interp(t_on, t, nmax):.6e} cm^-3")
    print(f"   f_em    at t_on                  = {np.interp(t_on, t, f):.6f}")
    print(f"   <n>_act MAXIMUM strictly BEFORE t_on = "
          f"{mn[t < t_on].max():.6e} cm^-3 (F1 lower edge {F1_LO:.1g})")
    print(f"   -> the forced response reaches "
          f"{100*mn[t < t_on].max()/F1_LO:.4f}% of the F1 lower edge before the "
          f"crossing")
    for thr, nm in ((F1_LO, "F1 lower edge 2e11"), (F1_HI, "F1 upper edge 5e11")):
        hit = np.flatnonzero(mn >= thr)
        if hit.size:
            tt = float(t[hit[0]])
            print(f"   t(<n>_act -> {nm}) = {tt:.6e} s "
                  f"({'AFTER' if tt > t_on else 'BEFORE'} the crossing, "
                  f"{(tt-t_on)*1e3:+.4f} ms relative to t_on)")
        else:
            print(f"   t(<n>_act -> {nm}) = NEVER in the window")
    inb = (mn >= F1_LO) & (mn <= F1_HI)
    if inb.any():
        print(f"   <n>_act is INSIDE the F1 band on t = "
              f"[{t[inb][0]:.6e}, {t[inb][-1]:.6e}] s "
              f"(residence {1e3*(t[inb][-1]-t[inb][0]):.4f} ms); "
              f"end-of-window value {mn[-1]:.6e} cm^-3 is "
              f"{mn[-1]/F1_HI:.4g}x the band's upper edge")
    print()

    print("-- lambda (all-surface, pd0 convention) AT MATCHED TIMES --")
    rows = {}
    for lab, path in (("central seed (t<=1.2 ms)", "scripts/ea1x_endvent_seed.npz"),
                      ("hi seed (full 5 ms)", "scripts/ea1x_endvent_hi.npz")):
        ev = np.load(path, allow_pickle=True)
        te, l, N = lam(ev)
        rows[lab] = (te, l, N)
        print(f"   {lab}: t range [{te[0]:.3e}, {te[-1]:.3e}] s, {te.size} steps")
    hdr = "   arm                        " + "".join(f"{tt:>13.3e}" for tt in MATCH_T)
    print("   lambda_all-surface [1/s]")
    print(hdr)
    for lab, (te, l, N) in rows.items():
        vals = [np.interp(tt, te, l) if tt <= te[-1] else float("nan")
                for tt in MATCH_T]
        print(f"   {lab:27s}" + "".join(f"{v:13.5g}" for v in vals))
    print("   N_col [particles]")
    print(hdr)
    for lab, (te, l, N) in rows.items():
        vals = [np.interp(tt, te, N) if tt <= te[-1] else float("nan")
                for tt in MATCH_T]
        print(f"   {lab:27s}" + "".join(f"{v:13.5g}" for v in vals))
    print()

    print("-- THE DISPOSAL COMPOSITION vs THE CENTRAL ARM (matched steps) --")
    a = np.load("scripts/ea1x_disposal_ckpt.npz")["trace"]
    b = np.load("scripts/ea1x_seed_ckpt.npz")["trace"]
    k = min(a.shape[0], b.shape[0])
    same = np.ascontiguousarray(a[:k]).tobytes() == np.ascontiguousarray(b[:k]).tobytes()
    print(f"   common prefix {k} steps (disposal {a.shape[0]}, central {b.shape[0]})")
    print(f"   RAW BYTES identical over the common prefix: {same}")
    names = ("t", "dt", "I_loop", "phi_c", "I_eth*", "<n>_act", "n[2]", "n[7]",
             "n_max", "f_em")
    d = np.abs(a[:k] - b[:k])
    for i, nm in enumerate(names):
        den = np.max(np.abs(b[:k, i]))
        print(f"   max|delta| {nm:8s} = {d[:, i].max():.6e}"
              + (f"   (rel {d[:, i].max()/den:.3e})" if den > 0 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
