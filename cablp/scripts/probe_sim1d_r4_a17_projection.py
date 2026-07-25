"""R4.3 A17 bracketed projection: velocity-resolved neutral moments vs the moment
two-neutral-momentum operator, on the matched M6 background (SIM1D_MODEL_AUDIT_PLAN
R4.3; audit A17).

A17 asks whether the reduced two-neutral-momentum operator (pressureless donor-cell
M_n/M_n_a advection + Fickian Knudsen diffusion, no neutral pressure/stress moment)
reproduces a velocity-resolved calculation projected onto (nn, M_n, nn_a, M_n_a) on
the same geometry and elapsed time. This probe compares the KN2Zone kinetic drift
(un_col, un_ann -- which carry the velocity-resolved axial streaming / duct tail the
Fickian closure cannot represent) against the saved moment-model drift (u_n, u_n_a)
on the settled window, and the densities as a cross-check.

Bracketed / null-capable: if the drift moments agree within a bounded factor the
reduced moments suffice; a large kinetic/moment divergence indicates a neutral
pressure/energy moment or a cold/fast split is required (the KN2Zone thread). This
is NOT a fit and does not by itself change the solver.

Prerequisite: run
    python scripts/kn2zone.py <M6.h5> --window 18.81 23.80 --out r43_a17_kn2zone
first; this reads its .npz.

Usage:
    python scripts/probe_sim1d_r4_a17_projection.py [--h5 PATH] [--npz PATH]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

DEFAULT_H5 = (
    "es1_nx120_m6_sq4600_g3200_c120_ts1900_l8p1_mn2mom300k_bmom_"
    "g1vessel150_rp15_baf150p27_r30_es1.h5"
)
DEFAULT_NPZ = "r43_a17_kn2zone.npz"
WINDOW = (18.81e-3, 23.80e-3)


def _resolve(p):
    p = Path(p)
    return p if p.is_absolute() else Path(__file__).resolve().parent / p


def main(argv=None):
    import h5py

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h5", default=DEFAULT_H5)
    ap.add_argument("--npz", default=DEFAULT_NPZ)
    args = ap.parse_args(argv)

    npz_path = _resolve(args.npz)
    if not npz_path.exists():
        print(f"KN2Zone output not found: {npz_path}")
        print("Run scripts/kn2zone.py <M6.h5> --window 18.81 23.80 "
              "--out r43_a17_kn2zone first.")
        return 0
    k = np.load(npz_path)
    zk = np.asarray(k["z"])
    nnc_k, nna_k = np.asarray(k["nn_col"]), np.asarray(k["nn_ann"])
    unc_k, una_k = np.asarray(k["un_col"]), np.asarray(k["un_ann"])

    f = h5py.File(_resolve(args.h5), "r")
    time = np.asarray(f["time"])
    phase = np.asarray(f["phase"]).astype(str)
    zc = np.asarray(f["geometry/z_cm"])
    sel = (time >= WINDOW[0]) & (time <= WINDOW[1]) & (phase == "main_discharge")

    def wmean(name):
        return np.median(np.asarray(f[name])[sel], axis=0)

    nn_m = wmean("nn")          # moment-model column density
    nna_m = wmean("nn_a")       # moment-model annulus density
    un_m = wmean("u_n")         # moment-model column neutral drift
    una_m = wmean("u_n_a")      # moment-model annulus neutral drift

    # interpolate moment fields onto the kinetic z grid
    def onto(zc, y):
        return np.interp(zk, zc, y)
    nn_mi, nna_mi = onto(zc, nn_m), onto(zc, nna_m)
    un_mi, una_mi = onto(zc, un_m), onto(zc, una_m)

    # mid-machine region (avoid the source/end cells) for a robust bracket
    mid = (zk > zk.min() + 0.15 * np.ptp(zk)) & (zk < zk.max() - 0.15 * np.ptp(zk))

    def ratio_stats(kin, mom):
        m = mid & (np.abs(mom) > 0) & np.isfinite(kin) & np.isfinite(mom)
        if not np.any(m):
            return float("nan"), float("nan"), float("nan")
        r = kin[m] / mom[m]
        return float(np.nanmin(r)), float(np.nanmedian(r)), float(np.nanmax(r))

    print("R4.3 A17 projection: KN2Zone kinetic vs moment two-momentum "
          "(matched M6, settled window)")
    print("=" * 74)
    print(f"kinetic z grid: {zk.size} pts;  mid-machine cells: {int(mid.sum())}\n")

    for label, kin, mom in [
        ("nn   column density", nnc_k, nn_mi),
        ("nn_a annulus density", nna_k, nna_mi),
        ("M_n  column drift u_n", unc_k, un_mi),
        ("M_n_a annulus drift u_n_a", una_k, una_mi),
    ]:
        lo, md, hi = ratio_stats(kin, mom)
        print(f"  {label:26s}  kinetic/moment  min={lo:8.2g} "
              f"median={md:8.2g} max={hi:8.2g}")

    print("\n" + "=" * 74)
    lo, md, hi = ratio_stats(unc_k, un_mi)
    lo2, md2, hi2 = ratio_stats(una_k, una_mi)
    print("A17 momentum-moment bracket (mid-machine kinetic/moment drift):")
    print(f"    column  M_n   : [{lo:.2g}, {hi:.2g}]  (median {md:.2g})")
    print(f"    annulus M_n_a : [{lo2:.2g}, {hi2:.2g}]  (median {md2:.2g})")
    within = (0.5 < abs(md) < 2.0) and (0.5 < abs(md2) < 2.0)
    print("    verdict:", (
        "reduced moments reproduce the kinetic drift within ~2x (bounded "
        "correction / near-null)" if within else
        "kinetic drift diverges from the reduced moments -> a neutral "
        "pressure/energy moment or cold/fast split is indicated (KN2Zone thread)"
    ))
    print("    (A17 is BRACKETED/exploratory: this does not promote the "
          "two-momentum arm nor change the solver.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
