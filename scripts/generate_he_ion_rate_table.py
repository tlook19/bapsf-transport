"""Generate the helium Maxwellian ionization-rate lookup table.

Run from the ``cablp`` project directory with the project environment::

    python generate_he_ion_rate_table.py
"""

from pathlib import Path

import mpmath as mp
import numpy as np

from cablp.atomic.cross_sections import He_EII_cross, integrate_kern
from cablp.atomic.coefficients import a_11s
from cablp.constants import E_ion


OUT_PATH = Path(__file__).parent.parent / "cablp" / "atomic" / "data" / "he_ion_rate.csv"
TEMPERATURES_EV = np.logspace(-1, 2, 1000)


def main():
    with mp.workdps(30):
        rates = integrate_kern(He_EII_cross, a_11s, TEMPERATURES_EV, E_ion)

    header = (
        "T_eV,rate_cm3_s\n"
        "He electron-impact ionization rate averaged over a 3D Maxwellian\n"
        "Generated with He_EII_cross, a_11s, and E_ion"
    )
    np.savetxt(
        OUT_PATH,
        np.column_stack([TEMPERATURES_EV, rates]),
        delimiter=",",
        header=header,
        comments="# ",
    )
    print(f"Wrote {OUT_PATH} ({len(rates)} points)")


if __name__ == "__main__":
    main()
