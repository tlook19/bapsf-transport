"""capfix: the He EII edge guard changes no value at the table edge.

The guard added at ``solve_beam_system_idriven``'s He lookup must be a pure
contract statement: within ``HE_EII_EDGE_REL_TOL`` of the table top it lets the
identical clamped lookup happen, and only a genuine excess raises. The case
that matters is a CAPABILITY-LIMITED solve, where the beam energy is the sheath
ceiling and therefore sits on the table's last node.

Prints the raw bits of the resulting beam cross section (plus the derived beam
arrays) so a pre-fix package and the post-fix package can be compared exactly.
Read-only.

Usage (from <checkout>/cablp, once per package under test):
    PYTHONPATH=<pkg> python scripts/capfix_beam_edge_bitexact.py --out FILE
"""

import argparse
import os

import numpy as np

from capfix_frozen_sweep import CONFIG, KWARGS, PLASMA

from cablp.funcs._cathode_solver import DeviceConfig, PlasmaState
from cablp.funcs._cathode_solver_idriven import solve_beam_system_idriven
from cablp.funcs._kernels import PROVENANCE
from cablp.vars._cons import I_ion

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
    P(f"# capfix_beam_edge_bitexact  kernels: {PROVENANCE}")
    P(f"# I_ion={I_ion!r}  table top = HE_EII_EPS_TOP*I_ion")
    for I in (5.0, 6.0, 8.0):
        res = solve_beam_system_idriven(
            cfg,
            np.array([pl.T_e, pl.T_e]),
            np.array([pl.n_e, pl.n_e]),
            np.array([pl.n_n, pl.n_n]),
            np.zeros(2),
            np.array([cfg.A_c, cfg.A_c]),
            I_ion,
            "He",
            float(I),
            cathode_index=0,
            **KWARGS,
        )
        P(
            f"I={I!r:6} regime={res.result.regime:>18} "
            f"phi_c={float(res.result.phi_c).hex()} "
            f"eps={float(res.result.phi_c / I_ion).hex()}"
        )
        P(
            f"    beam_cross[0]={float(res.beam_cross[0]).hex()} "
            f"v_beam[0]={float(res.v_beam[0]).hex()} "
            f"n_beam[0]={float(res.n_beam[0]).hex()} "
            f"A_ion_beam[0]={float(res.A_ion_beam[0]).hex()}"
        )
    if args.out:
        with open(args.out, "w") as fh:
            fh.write("\n".join(OUT) + "\n")
        print(f"# wrote {os.path.abspath(args.out)}", flush=True)


if __name__ == "__main__":
    main()
