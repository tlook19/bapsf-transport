"""R4.3 static bracket cross-check: Phelps moment operator vs the present quartet.

Offline static probe (no run, no solver advance) on the settled matched-M6
artifact used by the 2026-07-23 collision audit. Reproduces the present
ion-neutral cooling (Q_cx + Q_therm, frictional heating) from the saved
``ion_energy_terms_W_cm3`` rows, then re-evaluates the R4.3 moment-closed operator
(Phelps He+/He rates, T_eff=(Ti+Tn)/2, Tn=300 K) on the saved fluid state over the
same settled window, and prints both against the IAEA-based pre-registration
bracket [-30.40, -22.67] kW (R4.3; item 8).

The IAEA bracket is a CROSS-CHECK: Phelps supersedes the IAEA rate set, so the
Phelps thermal cooling is the primary reported value, not a target. No rate is
tuned. The thermal channel is velocity-independent (exact here); the frictional
channel uses the saved (u, u_n) as the relative drift (an offline approximation of
the live two-momentum column-wind sampling).

Usage:
    python scripts/probe_sim1d_r4_collision_bracket.py [--h5 PATH]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from cablp.funcs._cross import phelps_momentum_transfer_rate_cm3_s
from cablp.vars._cons import ev_to_erg, kb_cgs, m_He_cgs

DEFAULT_H5 = (
    "es1_nx120_m6_sq4600_g3200_c120_ts1900_l8p1_mn2mom300k_bmom_"
    "g1vessel150_rp15_baf150p27_r30_es1.h5"
)
WINDOW = (18.81e-3, 23.80e-3)  # audit settled window [s]
TN_EV = 300.0 * kb_cgs / ev_to_erg  # A8 single cold-gas neutral temperature


def _med_int(arr, Vp, sel, mask):
    """Median over settled samples of the volume-integral [W] of a [W/cm^3] row."""
    per_sample = np.sum(arr[:, mask] * Vp[None, mask], axis=1)[sel]
    return float(np.median(per_sample))


def main(argv=None):
    import h5py

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", default=DEFAULT_H5)
    args = parser.parse_args(argv)

    path = Path(args.h5)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    if not path.exists():
        print(f"settled M6 artifact not found: {path}")
        print("Regenerate it or pass --h5 PATH. (gitignored campaign artifact)")
        return 0

    f = h5py.File(path, "r")
    time = np.asarray(f["time"])
    phase = np.asarray(f["phase"]).astype(str)
    Vp = np.asarray(f["geometry/plasma_volume_cm3"], dtype=float)
    # Every cell carries some ion-neutral cooling; the saved legacy rows are the
    # per-cell truth, so integrate over all cells (Vp-weighted).
    active = np.ones(Vp.shape[0], dtype=bool)
    sel = (time >= WINDOW[0]) & (time <= WINDOW[1]) & (phase == "main_discharge")
    print(f"artifact: {path.name}")
    print(f"settled samples: {sel.sum()} "
          f"({time[sel][0]*1e3:.2f}-{time[sel][-1]*1e3:.2f} ms)\n")

    ion = f["ion_energy_terms_W_cm3"]
    q_cx = _med_int(np.asarray(ion["ion_charge_exchange"]), Vp, sel, active)
    q_therm = _med_int(np.asarray(ion["ion_neutral_thermalization"]), Vp, sel, active)
    q_fric = _med_int(np.asarray(ion["ion_neutral_frictional_heating"]), Vp, sel, active)
    present_cool = q_cx + q_therm

    print("PRESENT (cx_derived, saved rows) [kW]:")
    print(f"    Q_cx                        {q_cx/1e3:+8.2f}")
    print(f"    Q_therm (thermalization)    {q_therm/1e3:+8.2f}")
    print(f"    Q_cx + Q_therm  (cooling)   {present_cool/1e3:+8.2f}   "
          f"(audit: -45.87)")
    print(f"    Q_fric (frictional heating) {q_fric/1e3:+8.2f}   (audit: +11.34)\n")

    # --- Phelps moment operator, re-evaluated on the saved fluid state ---
    n = np.asarray(f["n"], dtype=float)
    nn = np.asarray(f["nn"], dtype=float)
    Ti = np.asarray(f["Ti"], dtype=float)
    u = np.asarray(f["u"], dtype=float)
    u_n = np.asarray(f["u_n"], dtype=float)
    ERG_TO_W = 1.0e-7  # the operator returns CGS erg/cm^3/s; saved rows are W/cm^3
    T_eff = 0.5 * (Ti + TN_EV)
    nu_mt = nn * phelps_momentum_transfer_rate_cm3_s(T_eff, gas_type="He")
    # thermal channel (velocity-independent, exact): 1.5 n nu_mt (Tn - Ti)
    phelps_therm = 1.5 * nu_mt * n * (TN_EV - Ti) * ev_to_erg * ERG_TO_W
    # frictional channel (approx: saved relative drift): 0.5 m n nu_mt (u-u_n)^2
    u_rel = u - u_n
    phelps_fric = 0.5 * m_He_cgs * nu_mt * n * u_rel**2 * ERG_TO_W

    P_therm = _med_int(phelps_therm, Vp, sel, active)
    P_fric = _med_int(phelps_fric, Vp, sel, active)

    print("PHELPS moment operator (T_eff, Tn=300 K) [kW]:")
    print(f"    thermal cooling  1.5 n nu_mt (Tn-Ti) {P_therm/1e3:+8.2f}")
    print(f"    frictional heat  0.5 m n nu_mt u_rel^2 {P_fric/1e3:+8.2f} (approx)")
    print(f"    net (thermal + friction)             {(P_therm+P_fric)/1e3:+8.2f}\n")

    print("BRACKET CROSS-CHECK [kW]:")
    print(f"    present Q_cx+Q_therm cooling  {present_cool/1e3:+8.2f}")
    print(f"    Phelps thermal cooling        {P_therm/1e3:+8.2f}")
    print(f"    IAEA pre-reg bracket          [-30.40, -22.67]  (cross-check only)")
    in_band = -30.40e3 <= P_therm <= -22.67e3
    direction = "reduced" if abs(P_therm) < abs(present_cool) else "NOT reduced"
    print(f"    -> Phelps cooling is {direction} vs present -45.87; "
          f"{'inside' if in_band else 'outside'} the IAEA band "
          f"(supersession: Phelps is primary, band is a sanity cross-check).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
