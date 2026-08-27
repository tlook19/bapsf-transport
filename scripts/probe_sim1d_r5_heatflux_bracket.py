"""R5.2 static heat-flux-limiter bracket on the settled M6 (audit A9).

Offline static probe (no run): on the settled main-discharge window, reconstruct
the electron Spitzer conductivity from the saved (Te, n) and compare the classical
flux q_SH = kappa_e |dTe/dz| against the free-streaming ceiling q_sat = f n Te
v_the, per cell, across f in {0.1, 0.3, 1.0}. Reports where the limiter engages
(lambda = q_sat/(q_sat+q_SH) < 1) and by how much it caps the flux, split by cell
role. Reproduces the A9 "q_SH reaches 1.7-3.3x n Te v_the" at f=1.

The DYNAMIC bracket (port Te/density from running with the limiter at each f) is a
full ES-class run and needs Tom's go-ahead; this bounds the leading static effect.

Usage:  python scripts/probe_sim1d_r5_heatflux_bracket.py [--h5 PATH]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from cablp.plasma.heat import kappa_par_elec
from cablp.plasma.params import c_log
from cablp.solvers._sim1d.physics.conduction import flux_limited_electron_conductivity
from cablp.constants import ev_to_erg, m_e_cgs

DEFAULT_H5 = (
    "es1_nx120_m6_sq4600_g3200_c120_ts1900_l8p1_mn2mom300k_bmom_"
    "g1vessel150_rp15_baf150p27_r30_es1.h5"
)
WINDOW = (18.81e-3, 23.80e-3)


class _Geom:
    def __init__(self, z_cm):
        self.z_cm = np.asarray(z_cm, dtype=float)


def main(argv=None):
    import h5py

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h5", default=DEFAULT_H5)
    args = ap.parse_args(argv)
    path = Path(args.h5)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    if not path.exists():
        print(f"settled M6 artifact not found: {path}")
        return 0

    f = h5py.File(path, "r")
    time = np.asarray(f["time"])
    phase = np.asarray(f["phase"]).astype(str)
    z_cm = np.asarray(f["geometry/z_cm"], dtype=float)
    role = np.asarray(f["geometry/cell_role"]).astype(str)
    sel = (time >= WINDOW[0]) & (time <= WINDOW[1]) & (phase == "main_discharge")
    geom = _Geom(z_cm)
    gap_mask = np.isin(role, ["gap", "cathode"])

    # window-median Te and n per cell
    Te = np.median(np.asarray(f["Te"])[sel], axis=0)
    n = np.median(np.asarray(f["n"])[sel], axis=0)
    ln = np.maximum(c_log(Te, n, kind="ei"), 1.0)
    kappa_e = kappa_par_elec(Te, n, ln, per_particle=False) * ev_to_erg

    Te_erg = Te * ev_to_erg
    v_the = np.sqrt(Te_erg / m_e_cgs)
    q_free = n * Te_erg * v_the  # n Te v_the (the f=1 free-streaming scale)
    grad = np.gradient(Te, z_cm)
    q_SH = np.abs(kappa_e) * np.abs(grad)

    print(f"artifact: {path.name}")
    print(f"settled samples: {int(sel.sum())} "
          f"({time[sel][0]*1e3:.2f}-{time[sel][-1]*1e3:.2f} ms); "
          f"gap/cathode cells: {int(gap_mask.sum())}\n")
    ratio = q_SH / np.maximum(q_free, 1e-300)
    print(f"A9 check  max q_SH/(n Te v_the) = {np.max(ratio):.2f}  "
          f"(gap/cathode {np.max(ratio[gap_mask]):.2f})   [finding: 1.7-3.3]\n")

    print(f"{'f':>5} {'engaged cells':>14} {'min lambda':>11} "
          f"{'gap flux kept':>13}")
    for ff in (0.1, 0.3, 1.0):
        q_sat = ff * q_free
        lam = q_sat / (q_sat + q_SH)
        # verify against the operator helper (same numbers)
        ke_lim = flux_limited_electron_conductivity(kappa_e, Te, n, geom, ff)
        assert np.allclose(ke_lim, kappa_e * lam, rtol=1e-10)
        engaged = int(np.sum(lam < 0.9))
        gap_kept = float(np.median(lam[gap_mask]))  # fraction of Spitzer flux kept
        print(f"{ff:5.1f} {engaged:14d} {float(np.min(lam)):11.3f} "
              f"{gap_kept:13.3f}")

    print("\n(lambda = fraction of the Spitzer flux retained; 'engaged' = cells "
          "where the limiter removes >10%. Gap/cathode is where A9 bites.)")
    print("DYNAMIC bracket (port Te/density from runs at each f) pending Tom's "
          "ES-run go-ahead.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
