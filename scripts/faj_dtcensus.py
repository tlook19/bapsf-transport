"""faj -- EXPLICIT dt census, including the sub-dt_min accepted-step class.

READ-ONLY instrument. No tuning, no fitting, no state mutation.

Why it exists: the fa3 read re-sighted a class of ACCEPTED steps whose dt is
BELOW the configured dt_min, which the ordinary "clamped to dt_min" counter
does not report (that counter counts steps AT the clamp, not steps beneath
it). The brief for faj requires this class to be censused explicitly rather
than inferred from the min-dt line.
"""

import json
import sys

import h5py
import numpy as np


def census(path, label):
    with h5py.File(path, "r") as f:
        p = json.loads(f.attrs["params_json"])
        dt_min_cfg = float(p.get("dt_min", float("nan")))
        # accepted-step dt series: prefer the recorded per-step dt
        dt = None
        for key in ("dt", "dt_history", "timestep"):
            if key in f:
                dt = np.asarray(f[key][:], dtype=float)
                break
        if dt is None and "time" in f:
            dt = np.diff(np.asarray(f["time"][:], dtype=float))
        print(f"--- {label}: {path} ---")
        print(f"  configured dt_min = {dt_min_cfg:.6e} s")
        if dt is None:
            print("  NO per-step dt series in the artifact; census unavailable")
            return
        dt = dt[np.isfinite(dt) & (dt > 0)]
        n = dt.size
        below = dt < dt_min_cfg * (1.0 - 1e-12)
        at = np.isclose(dt, dt_min_cfg, rtol=1e-12, atol=0.0)
        print(f"  accepted dt samples          {n}")
        print(f"  dt min / median / max        {dt.min():.6e} / "
              f"{np.median(dt):.6e} / {dt.max():.6e}")
        print(f"  AT dt_min (clamped)          {int(at.sum())} "
              f"({100.0*at.sum()/n:.3f} %)")
        print(f"  *** BELOW dt_min             {int(below.sum())} "
              f"({100.0*below.sum()/n:.3f} %) ***")
        if below.any():
            b = dt[below]
            print(f"      below-class min          {b.min():.6e} s")
            print(f"      below-class median       {np.median(b):.6e} s")
            print(f"      below-class max          {b.max():.6e} s")
            print(f"      deepest ratio dt/dt_min  {b.min()/dt_min_cfg:.6e}")
            print(f"      total time in class      {b.sum():.6e} s "
                  f"({100.0*b.sum()/dt.sum():.6f} % of elapsed)")
        print()


if __name__ == "__main__":
    for label, path in zip(sys.argv[1::2], sys.argv[2::2]):
        census(path, label)
