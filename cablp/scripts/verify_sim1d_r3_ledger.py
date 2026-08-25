"""R3 fluid/circuit power-ledger closure probe (audit A16 / section 8).

Static diagnostic for R3: reads the settled M6 artifact
and, on the audit's 18.81-23.80 ms main-discharge window,

  1. reproduces the fluid ``boundary_absorption`` power booking -- showing the
     nominally-absorbing walls are a net **+18.5 kW SOURCE**, driven by the A1
     wrong-sign reconstructed kinetic (u dM/dt - 1/2 m u^2 dn/dt);
  2. reproduces the full audit section-8 plasma power ledger (internal energy
     terms plus reconstructed kinetic), summing to ~-80 kW;
  3. isolates the R3 boundary defect and quantifies the misrouted sheath-phi
     energy: the cathode row as I_i*phi_c (the definition of the solver's
     P_cathode_i_phi), the anode row from the circuit's wall-side vs
     plasma-side (``_pl``) ion powers.

This validated the ledger structure and the R3 boundary term against real
numbers before the boundary/circuit re-derivation. The -81 kW total is the
audit's "not a physical balance": three structural errors partially cancel. No
single repair closes it -- R2 (hyperbolic core, committed), R3 (this boundary),
and R4 (beam interception) close it together.

Needs the gitignored settled M6 artifact; pass --h5 to override the default.

Usage:  python scripts/verify_sim1d_r3_ledger.py [--h5 PATH]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover
    print("h5py required")
    sys.exit(2)

DEFAULT_H5 = (
    "es1_nx120_m6_sq4600_g3200_c120_ts1900_l8p1_mn2mom300k_bmom_"
    "g1vessel150_rp15_baf150p27_r30_es1.h5"
)
MI = 4.0 * 1.6726e-24  # He ion mass [g]
ERG_TO_W = 1e-7
WINDOW = (18.81e-3, 23.80e-3)  # audit settled window [s]


def _open(path):
    p = Path(path)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent / p
    if not p.exists():
        print(f"settled M6 artifact not found: {p}")
        print("This diagnostic reads a gitignored campaign artifact; regenerate "
              "it (compare_sim1d_es1.py --save-h5 ...) or pass --h5 PATH.")
        return None
    return h5py.File(p, "r")


def analyze(f):
    time = np.asarray(f["time"])
    phase = np.asarray(f["phase"]).astype(str)
    Vp = np.asarray(f["geometry/plasma_volume_cm3"])
    u = np.asarray(f["u"])
    sel = (time >= WINDOW[0]) & (time <= WINDOW[1]) & (phase == "main_discharge")
    print(f"settled samples: {sel.sum()}  "
          f"({time[sel][0]*1e3:.2f}-{time[sel][-1]*1e3:.2f} ms)\n")

    def med_int(arr):  # volume-integral -> median over window [W]
        return float(np.median(np.sum(np.asarray(arr) * Vp[None, :], axis=1)[sel]))

    def kinetic(term):  # reconstructed kinetic power for one rhs term [W]
        g = f[f"rhs_terms/{term}"]
        dM = np.asarray(g["M"]) if "M" in g else 0.0
        dn = np.asarray(g["n"]) if "n" in g else 0.0
        dK = (u * dM - 0.5 * MI * u**2 * dn) * Vp[None, :] * ERG_TO_W
        return float(np.median(np.sum(dK, axis=1)[sel]))

    # (1) boundary_absorption booking
    P_be = med_int(f["electron_energy_terms_W_cm3/boundary_absorption"])
    P_bi = med_int(f["ion_energy_terms_W_cm3/boundary_absorption"])
    P_bk = kinetic("boundary_absorption")
    net = P_be + P_bi + P_bk
    print("(1) fluid boundary_absorption booking [kW]:")
    print(f"    electron internal  {P_be/1e3:+7.2f}   ion internal {P_bi/1e3:+7.2f}")
    print(f"    reconstructed KE   {P_bk/1e3:+7.2f}   NET          {net/1e3:+7.2f}"
          f"   (absorbing wall as a SOURCE => A1 sign error)\n")

    # (2) full section-8 ledger
    print("(2) plasma power ledger, settled window [kW]:")
    total = 0.0
    for t in sorted(f["rhs_terms"].keys()):
        e = med_int(f[f"electron_energy_terms_W_cm3/{t}"]) if t in f["electron_energy_terms_W_cm3"] else 0.0
        i = med_int(f[f"ion_energy_terms_W_cm3/{t}"]) if t in f["ion_energy_terms_W_cm3"] else 0.0
        s = e + i + kinetic(t)
        total += s
        if abs(s) > 5e3:
            print(f"    {t:30s} {s/1e3:+8.2f}")
    print(f"    {'TOTAL':30s} {total/1e3:+8.2f}   (audit ~ -80 kW; not a physical balance)\n")

    # (3) misrouted sheath-phi from circuit scalars
    cd = f["cathode_diagnostics"]
    cm = lambda k: float(np.median(np.asarray(cd[k])[sel]))
    # Cathode sheath-phi power, re-derived as I_i*phi_c -- the DEFINITION of the
    # solver's P_cathode_i_phi (_cathode_solver_idriven.py: P_cathode_i_phi =
    # P_cathode_i - P_cathode_i_thermal, with P_cathode_i_thermal = I_i*T_e/2).
    # This row previously took the wall-minus-plasma difference
    # P_cathode_i - P_cathode_i_pl, which is CONTAMINATED by anode ion
    # collection: P_cathode_i_pl is built on the ANODE ion current I_i_a
    # (= 2*eta*I_i), so the difference carried a spare (I_i - I_i_a)*T_e/2
    # thermal term (-0.53 kW, 0.25% of the row at the ES1 point).
    # CONSTRAINT THE CODE CANNOT SHOW: P_cathode_i_phi is not in the saved
    # cathode_diagnostics schema (_CATHODE_RESULT_KEYS), so this is a
    # RE-DERIVATION from published fields, not a read of the solver's value. It
    # is fp-inequivalent (~1 ulp class) to the solver's remainder form, which is
    # negligible here: this row feeds tolerance checks, not machine-zero
    # identities. (The schema key is deferred to a reviewed solver change.)
    phi_c_ion = cm("source_I_i") * cm("source_phi_c")
    # Anode: already exact -- both terms carry I_i_a, so the wall-minus-plasma
    # difference IS P_anode_i_phi = I_i_a*phi_a. Left as-is.
    phi_a_ion = cm("source_P_anode_i") - cm("source_P_anode_i_pl")
    print("(3) sheath-phi routed to walls [kW]:")
    print(f"    cathode ion phi_c   {phi_c_ion/1e3:+7.2f}   anode ion phi_a {phi_a_ion/1e3:+7.2f}")
    print(f"    P_net={cm('source_P_net')/1e3:+.1f} kW  vs  P_net2="
          f"{cm('source_P_net2')/1e3:+.1f} kW  (disagree by "
          f"{(cm('source_P_net2')-cm('source_P_net'))/1e3:.0f} kW => neither closes)")

    # assertions: reproduce the audit within tolerance
    ok = (
        -8.5 < P_be/1e3 < -6.5 and -2.6 < P_bi/1e3 < -1.4
        and 25.0 < P_bk/1e3 < 31.0 and 15.0 < net/1e3 < 22.0
        and -95.0 < total/1e3 < -65.0
    )
    print("\nR3 ledger probe:", "OK (reproduces audit A16/section 8)" if ok
          else "UNEXPECTED (audit numbers not reproduced)")
    return ok


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", default=DEFAULT_H5, help="settled M6 artifact path")
    args = parser.parse_args(argv)
    f = _open(args.h5)
    if f is None:
        return 0  # artifact absent: nothing to check, not a failure
    return 0 if analyze(f) else 1


if __name__ == "__main__":
    sys.exit(main())
