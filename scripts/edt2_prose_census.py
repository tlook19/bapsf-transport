"""Measure the figures the edt documentation states, so none is uncited.

Every number in MODEL.md's operator section, NUMERICS.md's discretization
section and the provenance entry must trace to a transcript. Most of them come
from ``verify_sim1d_edt.py`` and ``edt_consult_pins.py``. Three do not: they
were quoted from the 2026-08-31 advisor adjudication rather than measured by
this branch, and this file closes that gap so the prose rests on this
branch's own evidence rather than on a citation.

    P_cathode_e            the electron power the cathode surface collects,
                           which is what makes "the returning thermal-electron
                           current is ~0.3 mA" a measurement rather than an
                           assertion;
    gap survival           the beam's ray survival across the gap, which is
                           what motivates the charge-death default;
    l_b_profile vs ray     the factor by which the saved profile disagrees
                           with that survival, which is why a continuous
                           profile is NOT used as the bracket's substitute.

Usage (from the checkout root, PYTHONPATH set to it)::

    python scripts/edt2_prose_census.py --h5 scripts/mgcr1_confirm.h5
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

#: The consult's window, in seconds.
WINDOW = (1.0e-4, 2.01e-2)

#: Elementary charge [C], for the current the collected power implies.
_QE_C = 1.602176634e-19


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--h5",
        default=str(Path(__file__).resolve().parent / "mgcr1_confirm.h5"),
    )
    args = ap.parse_args(argv)
    with h5py.File(args.h5, "r") as h5:
        t = h5["time"][:]
        sel = np.flatnonzero((t >= WINDOW[0]) & (t <= WINDOW[1]))
        cd = h5["cathode_diagnostics"]
        print(f"# {args.h5}")
        print(
            f"# window {WINDOW[0] * 1e3:.3f}-{WINDOW[1] * 1e3:.3f} ms, "
            f"{sel.size} saves"
        )

        P_cath_e = float(np.nanmean(cd["source_P_cathode_e"][sel]))
        Te1 = float(np.nanmean(h5["Te"][sel, 1]))
        phi_c = float(np.nanmean(cd["source_phi_c"][sel]))
        I_loop = float(np.nanmean(cd["circuit_I_loop"][sel]))
        print(f"P_cathode_e (window mean)      = {P_cath_e:.4f} W")
        print(f"    T_e at the cathode cell = {Te1:.3f} eV, phi_c = {phi_c:.2f} V")
        # Which current that power implies depends on what each collected
        # electron carries into the surface. BOTH readings are printed because
        # they differ by the ratio of the sheath fall to the thermal energy,
        # which here is a factor of ~11 -- and only the full-fall reading is
        # the surface power this diagnostic records.
        I_full_mA = 1e3 * P_cath_e / (2.0 * Te1 + phi_c)
        I_thermal_mA = 1e3 * P_cath_e / (2.0 * Te1)
        print(
            f"    at the FULL sheath energy 2 T_e + phi_c = "
            f"{2 * Te1 + phi_c:.2f} eV per electron -> {I_full_mA:.3f} mA "
            "  <- the returning current the prose quotes"
        )
        print(
            f"    at the thermal part 2 T_e = {2 * Te1:.2f} eV alone       "
            f"-> {I_thermal_mA:.3f} mA"
        )
        print(
            f"    against the loop current {I_loop:.1f} A -- a ratio of "
            f"{I_full_mA * 1e-3 / I_loop:.3e}, which is why both flux "
            "channels at that face are taken as exactly zero"
        )

        for name in (
            "source_beam_gap_survival_ray",
            "source_beam_gap_survival_circuit",
            "source_beam_gap_survival_probe",
        ):
            v = cd[name][sel]
            print(f"{name:32s} = {float(np.nanmean(v)):.6f} (window mean)")

        l_b = float(np.nanmean(cd["source_l_b"][sel]))
        L_cath = 53.25
        print(f"source_l_b (window mean)       = {l_b:.3f} cm")
        prof = cd["l_b_profile"][sel, :]
        prof_gap = float(np.nanmean(prof[:, 1:6]))
        print(
            f"l_b_profile over the gap cells = {prof_gap:.2f} cm "
            f"(vs L_cath {L_cath} cm)"
        )
        surv_profile = float(np.exp(-L_cath / prof_gap))
        surv_ray = float(np.nanmean(cd["source_beam_gap_survival_ray"][sel]))
        print(
            f"    survival implied by the PROFILE exp(-L_cath/l_b) = "
            f"{surv_profile:.4f}"
        )
        print(f"    survival measured by the RAY                     = {surv_ray:.6f}")
        if surv_ray > 0.0:
            print(
                "    the profile disagrees with the ray by a factor of "
                f"{surv_profile / surv_ray:.1f} -- which is why a continuous "
                "survival profile is NOT the charge-death bracket's substitute"
            )
        print(
            f"V_p (window mean)              = "
            f"{float(np.nanmean(cd['source_V_p'][sel])):.4f} V"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
