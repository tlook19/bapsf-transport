"""R5.3 static low-Te clamp-fraction diagnostic (audit A18).

On an afterglow artifact, report the fraction of active-plasma cells below the
0.2 eV ADF11 edge (the A9 "13.5-21.4%" finding) and, at those sub-edge cells, how
much the standard nearest-edge clamp under-estimates the recombination rate (acd)
and its radiated power (prb1) vs the low-Te extension -- the size of the gap the
R5.3 consistency fix closes between the particle-rate and electron-cooling paths.

Usage:  python scripts/probe_sim1d_r5_lowte_bracket.py [--h5 PATH]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from cablp.funcs._adas import he_rates, he_rate_temperature_range_eV

DEFAULT_H5 = "es1_nx120_m6_sq3400_k4t_lowte_icool_es1.h5"


def main(argv=None):
    import h5py

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h5", default=DEFAULT_H5)
    args = ap.parse_args(argv)
    path = Path(args.h5)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    if not path.exists():
        print(f"artifact not found: {path}")
        return 0

    try:
        edge = float(he_rate_temperature_range_eV()[0])
    except Exception:
        edge = 0.2
    f = h5py.File(path, "r")
    time = np.asarray(f["time"])
    Te = np.asarray(f["Te"])
    n = np.asarray(f["n"])
    # active plasma = cells above the density floor (the ADAS domain applies there)
    active = n > 1e9
    below = (Te < edge) & active

    frac_all = float(np.sum(below) / max(np.sum(active), 1))
    last_ms = time >= (time.max() - 1.0e-3)
    fa = active[last_ms]
    fb = below[last_ms]
    frac_last = float(np.sum(fb) / max(np.sum(fa), 1))
    print(f"artifact: {path.name}")
    print(f"ADF11 Te edge = {edge:.4f} eV")
    print(f"sub-edge active-cell fraction: all-time {frac_all:.3f}, "
          f"last-ms {frac_last:.3f}   [A18 finding: 13.5% / 21.4%]\n")

    # under-estimation of acd/prb1 by the clamp at the sub-edge cells
    Te_b = Te[below]
    n_b = np.maximum(n[below], 1.0)
    if Te_b.size:
        rows = []
        for q in ("acd", "prb1"):
            clamp = he_rates(n_b, Te_b, [q], low_te_extension=False)[q]
            ext = he_rates(n_b, Te_b, [q], low_te_extension=True)[q]
            r = ext / np.maximum(clamp, 1e-300)  # how much the clamp misses
            rows.append((q, float(np.median(r)), float(np.max(r))))
        print("clamp under-estimation at sub-edge cells (extension/clamp):")
        for q, med, mx in rows:
            print(f"    {q:5s}  median {med:6.2f}x   max {mx:8.1f}x")
        print("\n(acd and prb1 share the same factor -> before R5.3 the particle "
              "path used the extended acd while cooling kept the clamped prb1; "
              "the fix makes them consistent.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
