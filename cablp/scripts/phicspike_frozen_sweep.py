"""phicspike: freeze the offending sheath state and sweep the imposed current
and the cap, to show that the returned NET phi_c escapes the cap only in the
window between the ladder's first and second doubling grid points.
Read-only.
"""
import argparse
import os
import sys

import numpy as np

MAIN_SCRIPTS = "/Users/tlook/bapsf/bapsf-transport/cablp/scripts"
if MAIN_SCRIPTS not in sys.path:
    sys.path.insert(0, MAIN_SCRIPTS)

from compare_sim1d_es1 import FLAG_OVERRIDES, PARAM_OVERRIDES
from run_mechanism_ladder import ES_OPERATING
from k7cbuild_frozen_bitexact import EXTRA

from cablp.funcs import _cathode_solver_idriven as idr
from cablp.funcs._kernels import PROVENANCE
from cablp.solvers._sim1d.physics import cathode as cath
from cablp.solvers._sim1d import LAPDSim1D, default_config

out = []
def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    out.append(s)

SNAP = []
_real_solve = idr.solve_idriven


def traced(config, plasma, **kw):
    res = _real_solve(config, plasma, **kw)
    SNAP.append((config, plasma, dict(kw), res))
    if len(SNAP) > 3000:
        del SNAP[:1500]
    return res


def build():
    params, flags = default_config()
    params.update(PARAM_OVERRIDES)
    flags.update(FLAG_OVERRIDES)
    flags["neutral_two_zone"] = True
    params["neutral_exchange_model"] = "knudsen"
    op = ES_OPERATING[1]
    params["V_bank"] = op["V_bank"]
    params["cathode_Ts_base_K"] = op["Ts_standby_K"]
    params.update(EXTRA)
    params["heating_anomalous_transport"] = "tail_walk"
    params["heating_anomalous_tail_ionization"] = "on"
    params["heating_anomalous_tail_phi_c_fraction"] = 1.0
    return LAPDSim1D(params, flags)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="step39")
    ap.add_argument("--steps", type=int, default=42)
    args = ap.parse_args()
    report = os.path.join(MAIN_SCRIPTS,
                          f"phicspike_frozen_sweep_{args.tag}.txt")
    idr.solve_idriven = traced
    cath.solve_idriven = traced
    P("# phicspike_frozen_sweep -- the excursion window in imposed current "
      "and in the cap")
    P(f"# kernels: {PROVENANCE}")
    P("")
    m = build()
    bad = None
    for step in range(1, args.steps + 1):
        n0 = len(SNAP)
        try:
            m.advance_one_step()
        except ValueError:
            for s in SNAP[n0:]:
                if float(s[3].phi_c) > float(s[2]["phi_c_cap_V"]) * 1.000000001:
                    bad = s
            P(f"# captured the refusing step {step}")
            break
    if bad is None:
        P("# no over-cap solve captured; nothing to sweep")
        return
    config, plasma, kw, res = bad
    cap = float(kw["phi_c_cap_V"])
    T_e = plasma.T_e
    P(f"FROZEN STATE (the solve the EII guard refused): T_e={T_e!r} eV  "
      f"n_e={plasma.n_e!r}  n_n={plasma.n_n!r}")
    P(f"  as-called I_imposed={kw['I_tot_A']!r} A -> regime={res.regime}  "
      f"NET phi_c={res.phi_c!r} V  I_tot={res.I_tot!r} A")
    P(f"  psi_top(1st grid point) = cap/T_e = {cap / T_e!r};  "
      f"2nd grid point = {2.0 * cap / T_e!r}")
    P("")

    def at(I, cap_V=cap):
        k = dict(kw)
        k["I_tot_A"] = float(I)
        k["phi_c_cap_V"] = float(cap_V)
        return _real_solve(config, plasma, **k)

    P("== A. SWEEP THE IMPOSED CURRENT at the frozen state (cap = 1000 V)")
    P(f"{'I_imposed[A]':>13} {'regime':>18} {'net phi_c[V]':>13} "
      f"{'net/cap':>9} {'I_tot[A]':>10} {'I_eth*[A]':>10} {'V_b[V]':>10}")
    for I in [5.0, 5.3, 5.40, 5.44, 5.45, 5.46, 5.50, 5.5674329614887945,
              5.57, 5.58, 5.59, 5.60, 5.65, 6.0, 8.0]:
        r = at(I)
        P(f"{I:>13.6f} {r.regime:>18} {r.phi_c:>13.4f} "
          f"{r.phi_c / cap:>9.4f} {r.I_tot:>10.4f} {r.I_eth_star:>10.4f} "
          f"{r.V_b:>10.3f}")
    P("")
    P("   (the sheath's own capability: I_tot at NET phi_c = cap, and at the")
    P("    doubled grid point, bracket the window in which the ladder can "
      "escape)")
    P("")

    P("== B. SWEEP THE CAP at the as-called current (the escape is "
      "grid-sampled, not physical)")
    P(f"{'cap[V]':>9} {'regime':>18} {'net phi_c[V]':>13} {'net/cap':>9}")
    for c in [500.0, 700.0, 800.0, 900.0, 950.0, 990.0, 1000.0, 1010.0,
              1050.0, 1100.0, 1200.0, 1500.0, 1900.0, 2000.0]:
        r = at(kw["I_tot_A"], c)
        P(f"{c:>9.1f} {r.regime:>18} {r.phi_c:>13.4f} {r.phi_c / c:>9.4f}")
    P("")

    P("== C. THE SAME CURRENT AT THE CIRCUIT'S OWN (UNSMOOTHED) SAMPLE")
    P("   The circuit advance samples the RAW source cell; the beam/RHS "
      "solve samples")
    P("   the EMA-smoothed one (solver.py:4960 vs solver.py:3134). Both at "
      "I as called:")
    from cablp.funcs._cathode_solver import PlasmaState
    for label, Te_s, ne_s in (
        ("EMA-smoothed (what the beam/tail solve got)", T_e, plasma.n_e),
        ("raw x1.5 Te", 1.5 * T_e, plasma.n_e),
        ("raw x1.66 Te (~the step's unsmoothed value)", 1.6564 * T_e,
         plasma.n_e),
        ("2x Te", 2.0 * T_e, plasma.n_e),
    ):
        ps = PlasmaState(T_e=Te_s, n_e=ne_s, n_n=plasma.n_n,
                         sigma_b=plasma.sigma_b)
        r = _real_solve(config, ps, **kw)
        P(f"   {label:<45} T_e={Te_s:7.4f} -> regime={r.regime:>18} "
          f"net phi_c={r.phi_c:10.4f} V  I_eth*={r.I_eth_star:.4f} A")
    P("")
    with open(report, "w") as fh:
        fh.write("\n".join(out) + "\n")
    P(f"# wrote {report}")


if __name__ == "__main__":
    main()
