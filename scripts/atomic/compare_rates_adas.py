"""Compare the historical He atomic-rate fits against the packaged ADAS data.

This regenerates the evidence behind ``atomic_rate_model = "adas"``:
the IAEA He I "electron cooling rate" fit
contains the ionization-potential loss (so the model's separate
ionization-cost term double-counts it), the He II fit is roughly 2x high,
and the direct ground-state ionization rate misses the stepwise/metastable
channel that dominates the effective rate at LAPD densities and low Te.

Correspondences (see cablp/atomic/data/adas/README.md for conventions):

    Qen = IAEA_exp1(aHeI)  <->  PLT(1)  [+ I_ion*SCD(1) if the fit includes
                                          the ionization cost -- it does]
    Qei = IAEA_exp4(aHeII) <->  PLT(2)  [+ PRB(1) for the recomb term]
    He_ion_rate_lkup       <->  SCD(1)
    alpha_r + ne*alpha_3   <->  ACD(1)

Usage::

    python scripts/compare_rates_adas.py
    python scripts/compare_rates_adas.py --ne 1e12 1e13
"""

import argparse

import numpy as np

from cablp.atomic.adas import (
    he_ion_line_power,
    he_ionization_rate,
    he_neutral_line_power,
    he_recombination_power,
    he_recombination_rate,
)
from cablp.atomic.cross_sections import He_ion_rate_lkup, alpha_3, alpha_r
from cablp.atomic.fits import IAEA_exp1, IAEA_exp4
from cablp.atomic.coefficients import aHeI, aHeII

TE_EV = np.array(
    [0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0, 50.0, 70.0, 100.0]
)
I_ION_HE = 24.587


def qen_bookkeeping(ne):
    """Does the IAEA He I cooling fit include the ionization cost?"""
    qen_iaea = IAEA_exp1(TE_EV, aHeI)
    plt1 = he_neutral_line_power(ne, TE_EV)
    scd1 = he_ionization_rate(ne, TE_EV)
    with_cost = plt1 + I_ION_HE * scd1
    print(f"\n--- Qen bookkeeping at ne = {ne:.1e} cm^-3 [eV cm^3/s] ---")
    print(f"{'Te':>6} {'IAEA':>11} {'PLT1':>11} {'IAEA/PLT1':>10} {'IAEA/(PLT1+I*SCD1)':>19}")
    for i, t in enumerate(TE_EV):
        print(
            f"{t:6.1f} {qen_iaea[i]:11.3e} {plt1[i]:11.3e} "
            f"{qen_iaea[i] / plt1[i]:10.3g} {qen_iaea[i] / with_cost[i]:19.3g}"
        )


def qei_scale(ne):
    """The He II fit against radiation-only ADAS coefficients."""
    qei_iaea = IAEA_exp4(TE_EV, aHeII, recomb=False)
    rec_iaea = IAEA_exp4(TE_EV, aHeII, recomb=True) - qei_iaea
    plt2 = he_ion_line_power(ne, TE_EV)
    prb1 = he_recombination_power(ne, TE_EV)
    print(f"\n--- Qei scale at ne = {ne:.1e} cm^-3 [eV cm^3/s] ---")
    print(
        f"{'Te':>6} {'IAEA_norec':>11} {'PLT2':>11} {'IAEA/PLT2':>10} "
        f"{'IAEA_rec_term':>13} {'PRB1':>11}"
    )
    for i, t in enumerate(TE_EV):
        print(
            f"{t:6.1f} {qei_iaea[i]:11.3e} {plt2[i]:11.3e} "
            f"{qei_iaea[i] / plt2[i]:10.3g} {rec_iaea[i]:13.3e} {prb1[i]:11.3e}"
        )


def implied_b_factors(ne):
    """b_Q* that would map the IAEA fits onto radiation-only ADAS cooling."""
    qen = IAEA_exp1(TE_EV, aHeI)
    qei = IAEA_exp4(TE_EV, aHeII, recomb=False)
    plt1 = he_neutral_line_power(ne, TE_EV)
    plt2 = he_ion_line_power(ne, TE_EV)
    print(f"\n--- implied janev-mode b(Te) at ne = {ne:.1e} cm^-3 ---")
    print(f"{'Te':>6} {'b_Qen':>7} {'b_Qei':>7}")
    for i, t in enumerate(TE_EV):
        print(f"{t:6.1f} {plt1[i] / qen[i]:7.3f} {plt2[i] / qei[i]:7.3f}")


def reaction_rate_ratios(ne):
    """Direct/summed model rates against the ADAS effective coefficients."""
    scd1 = he_ionization_rate(ne, TE_EV)
    acd1 = he_recombination_rate(ne, TE_EV)
    direct = He_ion_rate_lkup(TE_EV)
    rec_model = alpha_r(TE_EV, I=I_ION_HE) + ne * alpha_3(TE_EV)
    print(f"\n--- particle rates at ne = {ne:.1e} cm^-3 [cm^3/s] ---")
    print(
        f"{'Te':>6} {'direct_ion':>11} {'SCD1':>11} {'direct/SCD':>10} "
        f"{'rec_model':>11} {'ACD1':>11} {'rec/ACD':>8}"
    )
    for i, t in enumerate(TE_EV):
        print(
            f"{t:6.1f} {direct[i]:11.3e} {scd1[i]:11.3e} "
            f"{direct[i] / scd1[i]:10.3f} {rec_model[i]:11.3e} {acd1[i]:11.3e} "
            f"{rec_model[i] / acd1[i]:8.3f}"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ne",
        type=float,
        nargs="+",
        default=[1.0e13],
        help="electron densities [cm^-3] to evaluate at",
    )
    args = parser.parse_args(argv)
    for ne in args.ne:
        qen_bookkeeping(ne)
        qei_scale(ne)
        implied_b_factors(ne)
        reaction_rate_ratios(ne)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
