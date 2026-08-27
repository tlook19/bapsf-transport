"""fa4 -- EXPLICIT dt census, including the sub-dt_min ACCEPTED-step class.

READ-ONLY instrument. No tuning, no fitting, no state mutation.

WHY THIS EXISTS RATHER THAN faj_dtcensus.py. The copy of faj_dtcensus.py
currently in scripts/ looks for a per-step dt series under the TOP-LEVEL names
"dt" / "dt_history" / "timestep" and, finding none, falls back to
np.diff(time). None of those top-level names exist: the real per-accepted-step
series is diagnostics/accepted_dt, and "time" is the SAVE axis. On fa2_arm.h5
that fallback censuses 2635 SAVE INTERVALS (median 1.0e-05 s) instead of the
88083 accepted steps (median 2.8722e-07 s) -- a census of the wrong quantity,
which by construction can never see a sub-dt_min step.

The recorded faj_dtcensus.txt does NOT have this defect: its own header names
"diagnostics/accepted_dt" and it reports 88083 accepted steps for fa2. So the
file on disk is not the version that produced the recorded output. This script
reproduces the RECORDED instrument's semantics, and is verified against it:
fa2 must come back 88083 steps / median 2.8722e-07 / 153 clamp flags / 0
below-class, matching faj_dtcensus.txt line for line.

Series used:
    diagnostics/accepted_dt        per accepted step [s]
    diagnostics/clamped_to_dt_min  clamp flag (0.0/1.0)
    diagnostics/dt_raw             the unclamped request [s]
    diagnostics/active_constraint  which bound minimized
    params_json["dt_min"]          the configured floor
"""

import json
import sys

import h5py
import numpy as np


def census(path, label):
    with h5py.File(path, "r") as f:
        p = json.loads(f.attrs["params_json"])
        dt_min = float(p.get("dt_min", float("nan")))
        d = f["diagnostics"]
        if "accepted_dt" in d:
            dt, src = d["accepted_dt"][:].astype(float), "diagnostics/accepted_dt"
        elif "dt" in d:
            dt, src = d["dt"][:].astype(float), "diagnostics/dt"
        else:
            print(f"--- {label}: {path} ---\n  NO per-step dt series\n")
            return
        clamp = (d["clamped_to_dt_min"][:].astype(float)
                 if "clamped_to_dt_min" in d else None)
        raw = d["dt_raw"][:].astype(float) if "dt_raw" in d else None

        print(f"--- {label}: {path} ---")
        print(f"  series                       {src}")
        print(f"  configured dt_min            {dt_min:.6e} s")
        ok = np.isfinite(dt) & (dt > 0)
        dtv = dt[ok]
        n = dtv.size
        print(f"  accepted steps               {n}")
        print(f"  dt min / median / max        {dtv.min():.6e} / "
              f"{np.median(dtv):.6e} / {dtv.max():.6e}")
        if clamp is not None:
            nc = int((clamp > 0.5).sum())
            print(f"  clamped_to_dt_min flag set   {nc} "
                  f"({100.0 * nc / clamp.size:.3f} %)")
        below = dtv < dt_min * (1.0 - 1e-12)
        print(f"  *** ACCEPTED dt < dt_min     {int(below.sum())} "
              f"({100.0 * below.sum() / n:.3f} %) ***")
        if below.any():
            b = dtv[below]
            print(f"      below-class min          {b.min():.6e} s   "
                  f"(dt/dt_min = {b.min() / dt_min:.4e})")
            print(f"      below-class median       {np.median(b):.6e} s")
            print(f"      below-class max          {b.max():.6e} s")
            print(f"      elapsed time in class    {b.sum():.6e} s "
                  f"({100.0 * b.sum() / dtv.sum():.6f} % of total)")
            if clamp is not None:
                idx = np.where(ok)[0][below]
                print(f"      overlap with clamp flag  "
                      f"{int((clamp[idx] > 0.5).sum())} of {int(below.sum())}")
        if "active_constraint" in d:
            ac = np.array([x.decode() if isinstance(x, (bytes, np.bytes_))
                           else str(x) for x in d["active_constraint"][:]])
            print("  active constraint census (steps / elapsed):")
            u, c = np.unique(ac, return_counts=True)
            for k, v in sorted(zip(u, c), key=lambda x: -x[1]):
                tf = dt[ac == k].sum() / dt.sum()
                print(f"      {k:<26} {v:8d} ({100.0 * v / c.sum():6.2f}% / "
                      f"{100.0 * tf:6.2f}%)")
        if "dt_neutral_energy" in d:
            dte = d["dt_neutral_energy"][:].astype(float)
            print(f"  dt_neutral_energy            min={np.nanmin(dte):.6e} "
                  f"median={np.nanmedian(dte):.6e}")
            if raw is not None:
                g = np.isfinite(dte) & np.isfinite(raw) & (raw > 0)
                hr = dte[g] / raw[g]
                print(f"  HEADROOM dte/dt_raw          min={hr.min():.4f} "
                      f"p1={np.percentile(hr, 1):.4f} "
                      f"p5={np.percentile(hr, 5):.4f} "
                      f"median={np.median(hr):.4f}")
                print(f"      (>1 = the En bound was not binding; "
                      f"the 7x watch)")
        else:
            print("  dt_neutral_energy            ABSENT")
        print()


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 2 or len(args) % 2:
        raise SystemExit("usage: fa4_dtcensus.py LABEL PATH [LABEL PATH ...]")
    for label, path in zip(args[0::2], args[1::2]):
        census(path, label)
