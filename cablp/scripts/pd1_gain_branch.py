"""pd1: the gain/end-loss ratios in BOTH gain conventions, and the branching
z-structure -- post-hoc over the saved pd0_endvent npz traces. NO SOLVE.

1. GAIN / END-LOSS RATIOS. The end-loss rates are pd0_endvent_traj's own,
   recomputed from the arrays it saved with its arithmetic transcribed
   verbatim (solver.py-independent):

       lambda_collector = vent_coll_p / N_col
       lambda_all       = (vent_coll_p + vent_cath_p + anode_p) / N_col

   both over the BUILD WINDOW (steps >= i_on, i_on the efold1 onset index the
   instrument recorded), reduced by the median exactly as it does. The
   collector-only-convention ratio and the pd0 fit-gain ratio are reprinted as
   a CONTROL -- they must reproduce pd0_endvent_*.txt digit for digit.

   TWO GAIN CONVENTIONS, both reported per the dispatch:
     (A) pd0's: gain = d ln N_col/dt, ordinary least squares over the build
         window (the instrument's own fit).
     (B) THE REGISTERED ONE: gain = 1 / tau_F2, tau_F2 the covcal_read
         pedestal_efold value of record for that arm, read from the arm's own
         efold1 npz meta -- NOT refitted here.

2. BRANCHING z-STRUCTURE. No saved per-cell diagnostic carries the solver's
   computed f_Landau, so it is evaluated POST HOC with the SHIPPED
   ``cablp.funcs._beam_deposition.landau_branching_fraction`` -- the same
   function the solver calls -- on the per-cell (n, Te, nn) the pd0 instrument
   recorded and the live phi_c of the same step, which is the E0_eV the
   deposition passes it (_beam_deposition.py:1994). DISCLOSED DELTA: those
   states are the ACCEPTED state after the step, whereas the ray sees the
   state during it; this is pd0_branching.py's convention unchanged.
   Reported at three sampled times (onset, mid, end): f_Landau at the
   density-argmax cell and the active-column median, plus min/max for the
   spread that "bimodality" is about.
"""
import json
import sys

import numpy as np

from cablp.funcs._beam_deposition import landau_branching_fraction

ARMS = [
    ("pd1_f100", "scripts/pd1_endvent_f100.npz", "scripts/pd1_f100.npz",
     "landau_branched f=1.0 (CENTRAL ARM)"),
    ("pd1_f050", "scripts/pd1_endvent_f050.npz", "scripts/pd1_f050.npz",
     "landau_branched f=0.5"),
    ("pd1_f025", "scripts/pd1_endvent_f025.npz", "scripts/pd1_f025.npz",
     "landau_branched f=0.25"),
    ("a1_baseline", "scripts/pd0_endvent_a1.npz",
     "scripts/efold1_a1_baseline.npz", "REUSED baseline, disposal='local'"),
]


def main():
    print("== pd1_gain_branch: gain/end-loss in both conventions + branching")
    print("   NO SOLVE -- post-hoc over pd0_endvent_*.npz and efold1 npz meta.\n")
    for label, ev_path, ef_path, note in ARMS:
        z = np.load(ev_path, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        ef = np.load(ef_path, allow_pickle=True)
        ef_meta = json.loads(str(ef["meta"]))
        tau_F2_us = float(ef_meta["tau_us"])

        t = np.asarray(z["t"], float)
        Ncol = np.asarray(z["N_col"], float)
        i_on = int(meta["i_on"])
        ns = t.size
        mask = np.arange(ns) >= i_on

        lam = np.divide(z["vent_coll_p"], Ncol, out=np.zeros_like(Ncol),
                        where=Ncol > 0)
        lam_tot = np.divide(
            np.asarray(z["vent_coll_p"], float)
            + np.asarray(z["vent_cath_p"], float)
            + np.asarray(z["anode_p"], float),
            Ncol, out=np.zeros_like(Ncol), where=Ncol > 0)
        gfit = float(np.polyfit(t[mask], np.log(Ncol[mask]), 1)[0])
        gtau = 1.0e6 / tau_F2_us

        print(f"--- {label}  [{note}]")
        print(f"    steps={ns}  i_on={i_on}  t_on={meta['t_on']:.6e} s  "
              f"kernels={meta['kid']}")
        print(f"    gain (A) d ln N_col/dt LS fit = {gfit:.6e} 1/s "
              f"(tau {1e6/gfit:.4f} us)   [pd0 CONTROL]")
        print(f"    gain (B) 1/tau_F2 of record   = {gtau:.6e} 1/s "
              f"(tau_F2 {tau_F2_us:.4f} us)   [THE REGISTERED CONVENTION]")
        for nm, arr in (("collector-only", lam), ("all-surface  ", lam_tot)):
            med = float(np.median(arr[mask]))
            print(f"    end-loss {nm}: onset {arr[i_on]:.4e}  median "
                  f"{med:.4e}  end {arr[-1]:.4e} 1/s")
            print(f"      GAIN/END-LOSS  (A) fit-gain  = {gfit/med:.4g}"
                  f"     (B) tau_F2-gain = {gtau/med:.4g}")

        # ---- branching z-structure --------------------------------------
        phic = np.asarray(z["phi_c"], float)
        n = np.asarray(z["cell_n"], float)
        Te = np.asarray(z["cell_Te"], float)
        nn = np.asarray(z["cell_nn"], float)
        act = np.asarray(z["active"]).astype(bool)
        print("    branching f_Landau (SHIPPED landau_branching_fraction, "
              "post-hoc on recorded states; E0 = live phi_c):")
        for nm, i in (("onset", i_on), ("mid", (i_on + ns - 1) // 2),
                      ("end", ns - 1)):
            fl = landau_branching_fraction(n[i, act], Te[i, act], nn[i, act],
                                           float(phic[i]))
            j = int(np.argmax(n[i, act]))
            print(f"      {nm:>5} t={t[i]:.4e} s phi_c={phic[i]:7.2f} V: "
                  f"@n-argmax {fl[j]:.4f} (cell {j}, Te {Te[i,act][j]:.2f} eV, "
                  f"ne {n[i,act][j]:.3e}, nn {nn[i,act][j]:.3e}) | "
                  f"column median {np.median(fl):.4f}  "
                  f"min {fl.min():.4f}  max {fl.max():.4f}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
