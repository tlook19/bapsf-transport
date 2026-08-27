"""
Generate pre-computed EII cross section lookup tables for H and He.

Run once from the cablp/ directory:
    conda run -n fenicsx-env python generate_eii_tables.py

Outputs (in cablp/vars/):
    h_eii_cross.csv   -- H  electron impact ionization, E = 13.61–1000 eV
    he_eii_cross.csv  -- He electron impact ionization, eps = E/IE_He, 1.001–40.68
"""

import sys
from pathlib import Path

import numpy as np

# Make the package importable from this directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from cablp.funcs._cross import H_EII_cross, He_EII_cross
from cablp.vars._coeff import a_11s
from cablp.vars._cons import I_Ry as IE_Hydrogen, I_ion as IE_Helium

OUT_DIR = Path(__file__).parent.parent / "cablp" / "vars"
N = 1000

# ── H: E from IE_H to 1000 eV ──────────────────────────────────────────────
E_H = np.logspace(np.log10(IE_Hydrogen * 1.001), np.log10(1000.0), N)
sigma_H = np.array([H_EII_cross(E) for E in E_H])

header_H = "E_eV,sigma_cm2\nH electron impact ionization cross section\nE_eV: beam energy [eV]  sigma_cm2: cross section [cm^2]"
np.savetxt(
    OUT_DIR / "h_eii_cross.csv",
    np.column_stack([E_H, sigma_H]),
    delimiter=",",
    header=header_H,
    comments="# ",
)
print(f"Wrote {OUT_DIR / 'h_eii_cross.csv'}  ({N} points, {E_H[0]:.3f}–{E_H[-1]:.1f} eV)")

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
