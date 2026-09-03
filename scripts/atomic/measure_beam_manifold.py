#!/usr/bin/env python
"""Measure the He singlet excitation manifold seen by the cathode beam.

WP-A / A1: replaces the ``b_beam_excitation = 1.4``
estimate (2^1P plus "roughly another 40%" for the rest of the singlet
manifold) with the summed Ralchenko et al. (2008) fits over the full n <= 4
manifold plus the Eq. (5) n >= 5 Rydberg tail.

For each beam energy this prints, per level and summed:

- sigma            [cm^2]  — event cross section (sets the inelastic MFP)
- sigma * E_k      [cm^2 eV] — radiated-power weight (each event books its
                    threshold E_k as line radiation)

and the two equivalent multipliers relative to the historical single-channel
booking (2^1P at 21.218 eV):

- R_events = sum(sigma_k) / sigma_2P            — what the deposition length sees
- R_power  = sum(sigma_k E_k) / (sigma_2P * 21.218)  — what the radiation sees
                                                  (the measured value of the
                                                  old 1.4 estimate)

Usage:
    python scripts/atomic/measure_beam_manifold.py [--energies 60 100 150 180]
"""

from __future__ import annotations

import argparse

from cablp.atomic.cross_sections import (
    He_EIE_cross_manifold,
    He_singlet_tail_cross,
)
from cablp.atomic.coefficients import He_singlet_manifold

E_2P_BOOKED_EV = 21.218  # the historical single-channel radiated energy


def manifold_at(E_eV: float, n_max: int = 20):
    """Per-level and summed (sigma, sigma*E) at beam energy E_eV [cm^2, eV]."""
    rows = {}
    sigma_tot = 0.0
    sigma_E_tot = 0.0
    for name, entry in He_singlet_manifold.items():
        sigma = He_EIE_cross_manifold(E_eV, entry)
        rows[name] = (sigma, sigma * entry["E_eV"])
        sigma_tot += sigma
        sigma_E_tot += sigma * entry["E_eV"]
    tail_sigma, tail_sigma_E = He_singlet_tail_cross(E_eV, n_max=n_max)
    rows["n>=5 tail"] = (tail_sigma, tail_sigma_E)
    sigma_tot += tail_sigma
    sigma_E_tot += tail_sigma_E
    return rows, sigma_tot, sigma_E_tot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--energies",
        type=float,
        nargs="+",
        default=[60.0, 100.0, 150.0, 180.0],
        help="beam energies [eV] to tabulate (default: 60 100 150 180)",
    )
    parser.add_argument(
        "--n-max", type=int, default=20, help="tail truncation (default 20)"
    )
    args = parser.parse_args()

    for E in args.energies:
        rows, sigma_tot, sigma_E_tot = manifold_at(E, n_max=args.n_max)
        sigma_2p = rows["21P"][0]
        print(f"\n=== E_beam = {E:.1f} eV ===")
        print(f"{'level':>10} {'sigma [cm^2]':>13} {'sigma*E [cm^2 eV]':>18} "
              f"{'share of power':>15}")
        for name, (sigma, sigma_E) in rows.items():
            share = sigma_E / sigma_E_tot if sigma_E_tot > 0.0 else 0.0
            print(f"{name:>10} {sigma:13.4e} {sigma_E:18.4e} {share:15.1%}")
        print(f"{'sum':>10} {sigma_tot:13.4e} {sigma_E_tot:18.4e}")
        if sigma_2p > 0.0:
            r_events = sigma_tot / sigma_2p
            r_power = sigma_E_tot / (sigma_2p * E_2P_BOOKED_EV)
            mean_E = sigma_E_tot / sigma_tot
            print(f"R_events = sum(sigma)/sigma_2P            = {r_events:.3f}")
            print(f"R_power  = sum(sigma*E)/(sigma_2P*21.218) = {r_power:.3f}"
                  f"   (the measured '1.4')")
            print(f"mean radiated energy per event            = {mean_E:.2f} eV")

    print(
        "\nRemainder statement: the n >= 5 tail uses the paper's Eq. (5)"
        "\n(4/n)^3 scaling of the fitted n = 4 rows (Born-derived for nP,"
        "\nclassical for nS/nD/nF) with hydrogenic thresholds; it is an"
        "\nestimate, unlike the fitted n <= 4 levels."
    )


if __name__ == "__main__":
    main()
