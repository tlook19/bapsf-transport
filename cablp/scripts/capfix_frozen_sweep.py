"""capfix: the diagnostician's escape-window sweep, on the frozen state.

Sections A and B of ``phicspike_frozen_sweep.py`` re-run against the frozen
solve captured in ``capfix_escape_case.txt`` -- the state whose J-root escaped
the ceiling -- so the pre-fix and post-fix tables can be laid side by side
without re-marching 39 production steps (post-fix that march never reaches this
state, so the original capture path cannot produce the comparison).

A. sweep the imposed current at a fixed 1000 V cap: the escape window must be
   gone, i.e. phi_c rises monotonically to the cap and then stays flat AT it,
   every shortfall current tagged capability_limited.
B. sweep the cap at the captured current: pre-fix the answer was the same
   1854.195 V at every cap (cap-INDEPENDENT); post-fix it must equal the cap
   until the cap rises above the sheath's own unconstrained root.

Read-only. Run with PYTHONPATH pointed at whichever package is under test.
"""

import argparse
import os

import numpy as np

from cablp.funcs._cathode_solver import DeviceConfig, PlasmaState
from cablp.funcs import _cathode_solver_idriven as idr
from cablp.funcs._kernels import PROVENANCE

# Frozen at step 39 of the f = 1.0 tail-walk arm (capfix_capture_escape.py).
CONFIG = dict(
    A_c=706.8583470577034,
    mu=4,
    V_bank=177.843,
    T_s=1910.0000073162657,
    phi_wf=2.8689998037499964,
    C_R=12.96,
    R_comp=0.0072244,
    R_comp_partition=1.0,
    R_mesh_ohm=0.0,
    eta=0.358,
    Twin=False,
    L_cath=50.0,
    R_cath=15.0,
    phi_sheath_max=None,
    emission_Ts_K=tuple(
        np.float64(v)
        for v in (
            1909.7820600303419, 1908.0402707241847, 1904.566206201057,
            1899.3787650723516, 1892.5059746195277, 1883.9846150874373,
            1873.859733444452, 1862.1840584532556, 1849.0173309615827,
            1834.425564867055,
        )
    ),
    emission_area_cm2=tuple(
        np.float64(v)
        for v in (
            7.0685834705770345, 21.205750411731103, 35.34291735288517,
            49.480084294039244, 63.61725123519331, 77.75441817634739,
            91.89158511750145, 106.02875205865551, 120.16591899980959,
            134.30308594096365,
        )
    ),
    emission_plasma_frac=(1.0,) * 10,
)
PLASMA = dict(
    T_e=2.895378507817385,
    n_e=824479922.2030256,
    n_n=19744589075162.41,
    sigma_b=0.0,
)
KWARGS = dict(
    anode_current_A=0.0424709088850732,
    anode_T_e=2.9855005007754283,
    schottky=True,
    bridge=False,
    phi_c_cap_V=1000.0,
    alpha_sheath=0.8738291673131621,
    alpha_sheath_anode=None,
)
I_CAPTURED = 5.5674329614887945

OUT = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    OUT.append(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = DeviceConfig(**CONFIG)
    pl = PlasmaState(**PLASMA)
    cap = float(KWARGS["phi_c_cap_V"])
    P("# capfix_frozen_sweep -- the escape window at the frozen state")
    P(f"# kernels: {PROVENANCE}")
    P(f"# solver module: {idr.__file__}")
    P(f"# compiled root bound: {idr._COMPILED_ROOT}")
    P(f"FROZEN STATE: T_e={pl.T_e!r} eV  n_e={pl.n_e!r}  n_n={pl.n_n!r}")
    P(f"  captured I_imposed={I_CAPTURED!r} A   cap={cap!r} V")
    P(f"  psi_top(1st grid point) = cap/T_e = {cap / pl.T_e!r};  "
      f"2nd grid point = {2.0 * cap / pl.T_e!r}")
    P("")

    def at(I, cap_V=cap):
        kw = dict(KWARGS)
        kw["phi_c_cap_V"] = float(cap_V)
        return idr.solve_idriven(cfg, pl, I_tot_A=float(I), **kw)

    P("== A. SWEEP THE IMPOSED CURRENT (cap = 1000 V)")
    P(f"{'I_imposed[A]':>13} {'regime':>18} {'net phi_c[V]':>13} "
      f"{'net/cap':>9} {'I_tot[A]':>10} {'I_eth*[A]':>10} {'V_b[V]':>10}")
    prev = None
    monotone = True
    for I in [5.0, 5.3, 5.40, 5.44, 5.45, 5.46, 5.50, I_CAPTURED,
              5.57, 5.58, 5.59, 5.60, 5.65, 6.0, 8.0]:
        r = at(I)
        if prev is not None and float(r.phi_c) < prev - 1.0e-9:
            monotone = False
        prev = float(r.phi_c)
        P(f"{I:>13.6f} {r.regime:>18} {r.phi_c:>13.4f} "
          f"{r.phi_c / cap:>9.4f} {r.I_tot:>10.4f} {r.I_eth_star:>10.4f} "
          f"{r.V_b:>10.3f}")
    P("")
    P(f"   phi_c(I) monotone non-decreasing: {monotone}")
    P("")

    P("== B. SWEEP THE CAP at the captured current")
    P(f"{'cap[V]':>9} {'regime':>18} {'net phi_c[V]':>13} {'net/cap':>9}")
    for c in [500.0, 700.0, 800.0, 900.0, 950.0, 990.0, 1000.0, 1010.0,
              1050.0, 1100.0, 1200.0, 1500.0, 1900.0, 2000.0]:
        r = at(I_CAPTURED, c)
        P(f"{c:>9.1f} {r.regime:>18} {r.phi_c:>13.4f} {r.phi_c / c:>9.4f}")
    P("")
    if args.out:
        with open(args.out, "w") as fh:
            fh.write("\n".join(OUT) + "\n")
        print(f"# wrote {os.path.abspath(args.out)}", flush=True)


if __name__ == "__main__":
    main()
