"""R4 item-21 power-ledger re-check after A15 + A14.

Static diagnostic on the settled M6 artifact (audit 18.81-23.80 ms
window). Item 21: two opposite-sign structural source-side errors partially conceal
one another on the settled ES1 fit, so neither is visible in the scalar Te/current
agreement:

  A15  the CSDA beam ray deposits the FULL emitted beam (~470 kW) into the fluid,
       while the circuit books only (1 - eta*bypass) into the plasma (P_prim). The
       ~164 kW difference is the long-mfp beam the anode mesh intercepts -- the
       fluid was depositing it as spurious plasma heating.
  A14  bulk ionization books +43.1 kW of local electron birth energy (3/2 Te S_ion
       with Te_birth="local"), cancelling 92% of the -46.6 kW ionization-potential
       cost -- an electron created with no kinetic energy cannot heat the plasma.

This probe reproduces both numbers from the SAVED terms (like the R3 ledger probe),
then states the repaired booking that R4.1 (A15) and R4.2 (A14) make -- each term
moved to its correct book -- so the two no longer conceal one another. The repaired
values follow from the per-term identities the R4 unit gates prove exactly
(verify_sim1d_r4_beam_interception.py: fluid beam deposition -> P_prim; the
intercepted energy -> anode; verify_sim1d_r4_ionization_birth.py: electron birth
-> 0, dilution).

Needs the gitignored settled M6 artifact; pass --h5 to override the default.

Usage:  python scripts/verify_sim1d_r4_beam_and_birth_ledger.py [--h5 PATH]
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
WINDOW = (18.81e-3, 23.80e-3)  # audit settled window [s]


def _open(path):
    p = Path(path)
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[1] / p
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
    sel = (time >= WINDOW[0]) & (time <= WINDOW[1]) & (phase == "main_discharge")
    print(f"settled samples: {sel.sum()}  "
          f"({time[sel][0]*1e3:.2f}-{time[sel][-1]*1e3:.2f} ms)\n")

    def med_int(term):  # electron-energy term [W/cm^3] -> volume-integral median [W]
        arr = np.asarray(f[f"electron_energy_terms_W_cm3/{term}"])
        return float(np.median(np.sum(arr * Vp[None, :], axis=1)[sel]))

    cd = f["cathode_diagnostics"]

    def cm(key):  # circuit scalar [W], summed over source + twin end (nan-safe:
        # a collector end has NaN cathode diagnostics, contributing zero)
        stack = []
        for pre in ("source_", "end_"):
            name = pre + key
            if name in cd:
                stack.append(np.asarray(cd[name]))
        total = np.nansum(np.vstack(stack), axis=0)
        return float(np.median(total[sel]))

    # --- A15: fluid beam deposition vs circuit into-plasma booking
    P_beam_fluid = med_int("beam_power_deposition")
    P_prim = cm("P_prim")
    P_ohmic = cm("P_ohmic")
    P_into_plasma = P_prim + P_ohmic
    anode_intercepted = P_beam_fluid - P_into_plasma
    # Independent cross-check of the intercepted power: eta*bypass*I_eth_star*phi_c.
    import json
    eta = float(json.loads(f.attrs["params_json"]).get("eta", 0.0))
    # The intercepted beam removed from the PLASMA carries phi_c (the launch
    # energy the fluid was depositing downstream). Of that, the anode SURFACE
    # receives the electron's arrival KE = V_b (it decelerates through the anode
    # sheath phi_a), and phi_a is returned to the anode-sheath field (circuit).
    # So plasma-removed(phi_c) = anode-heat(V_b) + anode-sheath(phi_a) - gap(V_p).
    def circuit_bypass(volt_key):
        stk = []
        for pre in ("source_", "end_"):
            if pre + "I_eth_star" in cd:
                b = np.asarray(cd[pre + "beam_bypass_fraction"])
                I = np.asarray(cd[pre + "I_eth_star"])
                v = np.asarray(cd[pre + volt_key])
                stk.append(eta * b * I * v)
        return float(np.median(np.nansum(np.vstack(stk), axis=0)[sel]))

    xcheck = circuit_bypass("phi_c")   # plasma-removed cross-check
    anode_heat = circuit_bypass("V_b")   # I_bypass * V_b = circuit _P_beam_bypass
    anode_sheath = circuit_bypass("phi_a")   # returned to the anode-sheath field

    print("(A15) anode-mesh beam interception [kW]:")
    print(f"    fluid beam_power_deposition   {P_beam_fluid/1e3:+8.2f}")
    print(f"    circuit into-plasma (P_prim+P_ohmic) {P_into_plasma/1e3:+8.2f}")
    print(f"    REMOVED FROM PLASMA (bypass)  {anode_intercepted/1e3:+8.2f}"
          f"   (eta*bypass*I_eth*phi_c cross-check {xcheck/1e3:+8.2f})")
    print(f"      of which anode heat (I_bypass*V_b) {anode_heat/1e3:+8.2f}"
          f"  + anode-sheath->circuit (*phi_a) {anode_sheath/1e3:+8.2f}")
    print(f"    -> R4.1 repaired: fluid deposits ~P_prim+P_ohmic "
          f"({P_into_plasma/1e3:.1f} kW); the {anode_intercepted/1e3:.0f} kW is "
          f"removed from the plasma (the anode surface takes "
          f"{anode_heat/1e3:.0f} kW, the sheath returns "
          f"{anode_sheath/1e3:.0f} kW to the circuit).\n")

    # --- A14: bulk electron birth vs ionization cost
    P_birth_e = med_int("ionization_birth")
    P_cost = med_int("ionization_energy_cost")
    print("(A14) bulk ionization birth energy [kW]:")
    print(f"    electron birth (3/2 Te S_ion) {P_birth_e/1e3:+8.2f}")
    print(f"    ionization potential cost     {P_cost/1e3:+8.2f}"
          f"   (birth cancels {100*P_birth_e/abs(P_cost):.0f}% of the cost)")
    print(f"    -> R4.2 repaired: electron birth -> 0 (dilution); the "
          f"{P_birth_e/1e3:.0f} kW spurious electron heating is removed.\n")

    # --- Summary: the two spurious electron sources, now un-concealed
    total_spurious = anode_intercepted + P_birth_e
    print("spurious electron-heating sources removed by R4 [kW]:")
    print(f"    A15 bypass beam (plasma-removed) {anode_intercepted/1e3:+8.2f}")
    print(f"    A14 electron birth     {P_birth_e/1e3:+8.2f}")
    print(f"    TOTAL R4 correction    {-total_spurious/1e3:+8.2f}"
          f"   (both were partly hidden behind the R2 -104 kW hyperbolic leak "
          f"and each other)\n")

    # assertions: reproduce the audit item-21 numbers within tolerance
    ok = (
        450.0 < P_beam_fluid/1e3 < 490.0
        and 130.0 < anode_intercepted/1e3 < 190.0
        and abs(anode_intercepted - xcheck)/abs(anode_intercepted) < 0.15
        and 35.0 < P_birth_e/1e3 < 50.0
        and -52.0 < P_cost/1e3 < -42.0
    )
    print("R4 item-21 ledger re-check:",
          "OK (reproduces the A15 ~164 kW + A14 ~43 kW item-21 pair)" if ok
          else "UNEXPECTED (item-21 numbers not reproduced)")
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
