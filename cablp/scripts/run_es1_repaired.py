"""Run the R5 repaired-stance ES config and save an H5 for scoring.

Config-complete driver for the first repaired-stance ES1 benchmark (R5 ES1
tuning pass). Uses compare_sim1d_es1.run_model (the exact ES production config)
with two throughput levers for the refit iteration:

  --nx 60                  coarse grid (~2x fewer cells; ES port metrics move
                           <=14% vs nx=120 per THESIS_NOTES item 7 -- iteration
                           grade, peak quantities stay unquotable)
  --density-dt-fraction    relaxes the surface_loss (gap-cell Ee) dt bound; the
                           binding bound during the ES crawl. Reports the
                           floor-clip ledger so the accuracy cost is measured,
                           not assumed.

Score the saved H5 with:
  python scripts/compare_sim1d_es1.py --from-h5 <out.h5> --nx <nx>
  python scripts/fingerprints_sim1d.py <out.h5>
"""

import argparse
import time as _walltime

import numpy as np

from compare_sim1d_es1 import run_model
from cablp.solvers._sim1d.results.io import save_result_hdf5


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nx", type=int, default=60)
    ap.add_argument("--density-dt-fraction", type=float, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    extra = {}
    if args.density_dt_fraction is not None:
        extra["density_dt_fraction"] = args.density_dt_fraction

    print(f"# ES1 repaired run  nx={args.nx}  density_dt_fraction="
          f"{args.density_dt_fraction}  -> {args.out}")
    t0 = _walltime.time()
    result, geometry, params, flags = run_model(nx=args.nx, extra=extra)
    wall = _walltime.time() - t0

    phases = np.asarray(result.phase, dtype=str)
    t = np.asarray(result.time, float)
    I = np.asarray(result.cathode_diagnostics["source_I_tot"], float)
    ledger = getattr(result, "floor_ledger", {}) or {}
    ever_main = bool(np.any(phases == "main_discharge"))

    print(f"# wall={wall:.1f}s  saves={len(t)}  t_end={t[-1]*1e3:.2f}ms  "
          f"ever_main_discharge={ever_main}  I_max={np.nanmax(I):.0f} A")
    print("# floor-clip ledger (accuracy cost of the dt relaxation):")
    for k, v in ledger.items():
        print(f"#   {k}: {float(v):.3e}")
    save_result_hdf5(args.out, result, params=params, flags=flags)
    print(f"# saved {args.out}")


if __name__ == "__main__":
    main()
