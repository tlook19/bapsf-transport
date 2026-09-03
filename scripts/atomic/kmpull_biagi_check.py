"""Verify the shipped He e-n momentum-transfer table nodes against the
LXCat Biagi pull of 2026-08-13 (~/Downloads/biagi.xml; Magboltz 8.97
transcription, retrieved 2026-08-13).

Checks, per shipped node (5 eV, 25 eV; atomic/cross_sections.py HE_EN_MT_*):
  - Biagi sigma_m at the node energy (log-log and linear interpolation on
    the pulled 36-point table), vs the shipped value and its bracket.
  - Context only (documented conventions, not gated): the shipped
    K_m(Te) = sigma(1.5 Te)*<v> estimate (with its flat clamp above 25 eV)
    vs a full Maxwellian <sigma v> quadrature over the Biagi table.

Read-only on the repo; writes nothing but stdout (tee to
kmpull_biagi_check.txt).
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from cablp.atomic.cross_sections import (  # noqa: E402
    HE_EN_MT_NODE_EV,
    HE_EN_MT_SIGMA_CM2,
    HE_EN_MT_SIGMA_BRACKET_CM2,
    he_electron_momentum_transfer_cm2,
    he_electron_momentum_transfer_rate_cm3_s,
)

XML = str(Path.home() / "Downloads" / "biagi.xml")

root = ET.parse(XML).getroot()
proc = root.find(".//Process")
E_eV = np.array([float(x) for x in proc.find("DataX").text.split()])
sig_m2 = np.array([float(y) for y in proc.find("DataY").text.split()])
sig_cm2 = sig_m2 * 1.0e4
desc = root.find(".//Group[@id='He']/Description").text
print(f"# Biagi He elastic momentum transfer: {len(E_eV)} points, "
      f"E {E_eV[0]:g}-{E_eV[-1]:g} eV")
print(f"# Provenance: {desc.strip()}")
print("# Retrieval: 2026-08-13 (XML export; header warns format under "
      "development -- TXT re-pull recommended for the boxed record)")
print()

pos = E_eV > 0
logE, logS = np.log(E_eV[pos]), np.log(sig_cm2[pos])


def biagi_sigma(E, loglog=True):
    if loglog:
        return float(np.exp(np.interp(np.log(E), logE, logS)))
    return float(np.interp(E, E_eV, sig_cm2))


print("== Node verification ==")
for node, shipped, (lo, hi) in zip(
    HE_EN_MT_NODE_EV, HE_EN_MT_SIGMA_CM2, HE_EN_MT_SIGMA_BRACKET_CM2
):
    b_ll = biagi_sigma(node, loglog=True)
    b_lin = biagi_sigma(node, loglog=False)
    inb = lo <= b_ll <= hi and lo <= b_lin <= hi
    print(f"  {node:g} eV: Biagi {b_ll:.3e} (loglog) / {b_lin:.3e} (linear) "
          f"cm^2 | shipped {shipped:.1e}, bracket [{lo:.1e}, {hi:.1e}] "
          f"-> {'IN BRACKET' if inb else 'OUT OF BRACKET'}; "
          f"delta vs shipped {100 * (b_ll / shipped - 1):+.1f} %")

print()
print("== Context (documented conventions, not gated) ==")
EV_TO_ERG = 1.602176634e-12
M_E = 9.1093837015e-28


def maxwellian_km(Te, Efun):
    # <sigma v> = sqrt(8/(pi m)) (kTe)^-1.5 * Int sigma(E) E exp(-E/Te) dE
    Eg = np.geomspace(1e-4, 60.0 * Te, 4000)
    sig = np.array([Efun(e) for e in Eg])
    integ = np.trapezoid(sig * Eg * np.exp(-Eg / Te), Eg)  # eV^2 * cm^2
    pref = np.sqrt(8.0 / (np.pi * M_E)) * (Te * EV_TO_ERG) ** -1.5
    return pref * integ * EV_TO_ERG**2  # cm^3/s


for Te in (5.0, 25.0):
    shipped_km = he_electron_momentum_transfer_rate_cm3_s(Te)
    biagi_full = maxwellian_km(Te, lambda e: biagi_sigma(e, loglog=False))
    ship_full = maxwellian_km(Te, he_electron_momentum_transfer_cm2)
    print(f"  Te={Te:g} eV: shipped K_m(sigma(1.5Te)<v>) {shipped_km:.3e} | "
          f"Maxwellian<sigma v> Biagi {biagi_full:.3e} | "
          f"Maxwellian over shipped 2-node table {ship_full:.3e} cm^3/s")
    print(f"    shipped-estimate / Biagi-full ratio: "
          f"{shipped_km / biagi_full:.2f}")
print()
print("# NB the shipped estimate evaluates sigma at 1.5*Te with the table")
print("# CLAMPED flat above 25 eV (documented order-of-magnitude stance;")
print("# onset-gate margin is x400-2500, so these ratios are immaterial to")
print("# any gating).")
