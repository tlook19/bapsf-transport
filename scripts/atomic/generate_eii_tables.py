"""
Generate the pre-computed He EII cross section lookup table.

Run once from the repo root:
    conda run -n fenicsx-env python scripts/atomic/generate_eii_tables.py

Output (in cablp/atomic/data/):
    he_eii_cross.csv  -- He electron impact ionization, eps = E/IE_He, 1.001–40.68
"""

import sys
from pathlib import Path

import numpy as np

# Make the package importable from this directory
sys.path.insert(0, str(Path(__file__).parents[2]))

from cablp.atomic.cross_sections import He_EII_cross
from cablp.atomic.coefficients import a_11s
from cablp.constants import I_ion as IE_Helium

OUT_DIR = Path(__file__).parents[2] / "cablp" / "atomic" / "data"
N = 1000

# ── He: eps = E/IE_He from 1.001 to 1000/IE_He ─────────────────────────────
eps_max = 1000.0 / IE_Helium
eps_He = np.logspace(np.log10(1.001), np.log10(eps_max), N)
sigma_He = np.array([float(He_EII_cross(eps, a_11s)) for eps in eps_He])

header_He = (
    "eps,sigma_cm2\n"
    "He electron impact ionization cross section (a_11s coefficients)\n"
    "eps = E_beam/IE_He (dimensionless)  sigma_cm2: cross section [cm^2]"
)
np.savetxt(
    OUT_DIR / "he_eii_cross.csv",
    np.column_stack([eps_He, sigma_He]),
    delimiter=",",
    header=header_He,
    comments="# ",
)
print(f"Wrote {OUT_DIR / 'he_eii_cross.csv'}  ({N} points, eps={eps_He[0]:.4f}–{eps_He[-1]:.4f})")
